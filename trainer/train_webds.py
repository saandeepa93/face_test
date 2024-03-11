import sys
sys.path.append('.')
import io
import braceexpand

import torch 
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# DISTRIBUTED COMPUTING
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler # loader
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.distributed import init_process_group
from torchvision.transforms import ToPILImage
import torch.distributed as dist

import webdataset as wds
from transforms import GroupNormalize, ToTorchFormatTensor, Stack
from einops import rearrange

from imports import * 
from utils import * 
from configs import get_cfg_defaults
from dataset2 import BGCDataset
from models import FaceFeatureModel
from supCon import SupConLoss




def ddp_setup(args):
  if "WORLD_SIZE" in os.environ:
    args.world_size = int(os.environ["WORLD_SIZE"])
  args.distributed = args.world_size > 1
  ngpus_per_node = torch.cuda.device_count()
  if args.distributed:
    if args.local_rank != -1: # for torch.distributed.launch
      args.rank = args.local_rank
      args.gpu = args.local_rank
    elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
      args.rank = int(os.environ['SLURM_PROCID'])
      args.gpu = int(os.environ['SLURM_LOCALID'])
    init_process_group(backend='nccl', world_size=args.world_size, rank=args.rank)
    if args.rank!=0:
      def print_pass(*args,  **kwargs):
          pass
      builtins.print = print_pass

    torch.cuda.set_device(args.gpu)
    
def custom_collate_fn(batch):
  x_g = batch[0]
  x_p = batch[1]
  print(x_p.size(), x_g.size())
  return torch.utils.data.dataloader.default_collate(batch)
  batched_json = {
        "gallery": [],
        "probe": [],
        "probe_frames": []
  }
  json_data = [item[2] for item in batch]
  for item in json_data:
    batched_json["gallery"].extend(item["gallery"])
    batched_json["probe"].append(item["probe"])
    batched_json["probe_frames"].extend(item["probe_frames"])
  
  x1_batch = torch.stack([x1 for x1, x2, _ in batch])
  x2_batch = torch.stack([x2 for x1, x2, _ in batch])
  return x1_batch, x2_batch, batched_json 


def preprocess_openface_val(batch):
  gallery_csv_dir_train = "/shares/rra_sarkar-2135-1003-00/faces/openface_feats_extractor/data/openface/val/gallery"
  # gallery_csv_dir_val = "/shares/rra_sarkar-2135-1003-00/faces/openface_feats_extractor/data/openface/gallery"

   # JSON PREPROCESSING
  json_bytes = batch['json']
  json_dict = json.loads(json_bytes.decode('utf-8'))
  
  phase = json_dict['probe'][0].split('/')[0]
  subject = json_dict['probe'][0].split('/')[1]
  all_controlled_fnames = json_dict['gallery']
  all_controlled_fnames = list(itertools.chain.from_iterable(all_controlled_fnames))
  all_controlled_fnames = [k.split('/')[-1] for k in all_controlled_fnames]

  openface_csv_path = os.path.join(gallery_csv_dir_train, phase, subject, "face/face.csv")

  try:
    df_openface = pd.read_csv(openface_csv_path)
    df_openface.loc[1:, 'pose_Rx'] = np.degrees(df_openface['pose_Rx'])
    df_openface.loc[1:,'pose_Ry'] = np.degrees(df_openface['pose_Ry'])
    df_openface.loc[1:, 'pose_Rz'] = np.degrees(df_openface['pose_Rz'])

    df_openface['magnitude'] = np.sqrt(df_openface['pose_Rx']**2 + df_openface['pose_Ry']**2 + df_openface['pose_Rz']**2)
    df_openface = df_openface.sort_values(by='magnitude', ascending=True)

    all_openface_selected = df_openface['frame_name'].tolist()
    all_openface_indeces = [all_controlled_fnames.index(b) for b in all_openface_selected if b in all_controlled_fnames]

    frame_len = 4
    all_field_ind_gallery = all_openface_indeces
    
    if len(all_field_ind_gallery) < frame_len:
      selected_field_ind = list(np.random.choice(all_field_ind_gallery, size=frame_len, replace=True))
    else:
      selected_field_ind = list(np.random.choice(all_field_ind_gallery, size=frame_len, replace=False))

  
  except pd.errors.EmptyDataError:
    frame_len = 4
    all_field_ind_gallery = np.arange(tensor1.size(2))
    random.shuffle(all_field_ind_gallery)
    if len(all_field_ind_gallery) < frame_len:
      index = list(np.random.choice(all_field_ind_gallery, size=frame_len, replace=True))
      selected_field_ind = [all_field_ind_gallery[k] for k in index]
    else:
      selected_field_ind = all_field_ind_gallery[:frame_len]
    print (openface_csv_path, " is empty")


  # GALLERY PREPROCESSING
  data1 = batch['image_g.pt']
  buffer1 = io.BytesIO(data1)
  tensor1 = torch.load(buffer1)

  selected_field_ind = torch.tensor(selected_field_ind)
  x_g = tensor1[:, :, selected_field_ind]

  # PROBE PREPROCESSING
  data2 = batch['image_p.pt']
  buffer2 = io.BytesIO(data2)
  tensor2 = torch.load(buffer2)
  

  all_field_ind_probe = np.arange(tensor2.size(2))
  random.shuffle(all_field_ind_probe)
  if len(all_field_ind_probe) < frame_len:
    index = list(np.random.choice(all_field_ind_probe, size=frame_len, replace=True))
    selected_field_ind_probe = [all_field_ind_probe[k] for k in index]
  else:
    selected_field_ind_probe = all_field_ind_probe[:frame_len]
  selected_field_ind_probe = torch.tensor(selected_field_ind_probe)
  x_p = tensor2[:, :, selected_field_ind_probe]


 
  # gallery_paths = list(itertools.chain(*json_dict['gallery']))
  # gallery_paths = [gallery_paths[k] for k in selected_field_ind]
  # probe_frames = [json_dict['probe_frames'][k] for k in selected_field_ind_probe]
  all_subjects = [k.split('/')[1] for k in json_dict['probe']]
  all_subs_dict = {}
  cnt = 0
  for sub in all_subjects:
    if sub not in all_subs_dict:
      all_subs_dict[sub] = cnt
      cnt += 1
  
  labels = [all_subs_dict[sub] for sub in all_subjects]
  # new_json = {
  #   "gallery": gallery_paths,
  #   "probe": probe_frames
  # }

  # NORMALIZE

  mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1, 1)
  std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1, 1)
  x_g.sub_(mean).div_(std)
  x_p.sub_(mean).div_(std)

  batch['image_g.pt'] = x_g
  batch['image_p.pt'] = x_p
  batch['json'] = torch.tensor(labels)
  return batch

def preprocess_openface(batch):
  gallery_csv_dir_train = "/shares/rra_sarkar-2135-1003-00/faces/openface_feats_extractor/data/openface/train/gallery"
  # gallery_csv_dir_val = "/shares/rra_sarkar-2135-1003-00/faces/openface_feats_extractor/data/openface/gallery"

   # JSON PREPROCESSING
  json_bytes = batch['json']
  json_dict = json.loads(json_bytes.decode('utf-8'))
  
  phase = json_dict['probe'][0].split('/')[0]
  subject = json_dict['probe'][0].split('/')[1]
  all_controlled_fnames = json_dict['gallery']
  all_controlled_fnames = list(itertools.chain.from_iterable(all_controlled_fnames))
  all_controlled_fnames = [k.split('/')[-1] for k in all_controlled_fnames]


  openface_csv_path = os.path.join(gallery_csv_dir_train, phase, subject, "face/face.csv")
  try:
    df_openface = pd.read_csv(openface_csv_path)
    df_openface.loc[1:, 'pose_Rx'] = np.degrees(df_openface['pose_Rx'])
    df_openface.loc[1:,'pose_Ry'] = np.degrees(df_openface['pose_Ry'])
    df_openface.loc[1:, 'pose_Rz'] = np.degrees(df_openface['pose_Rz'])

    df_openface['magnitude'] = np.sqrt(df_openface['pose_Rx']**2 + df_openface['pose_Ry']**2 + df_openface['pose_Rz']**2)
    df_openface = df_openface.sort_values(by='magnitude', ascending=True)

    all_openface_selected = df_openface['frame_name'].tolist()
    all_openface_indeces = [all_controlled_fnames.index(b) for b in all_openface_selected if b in all_controlled_fnames]

    frame_len = 4
    all_field_ind_gallery = all_openface_indeces
    
    if len(all_field_ind_gallery) < frame_len:
      selected_field_ind = list(np.random.choice(all_field_ind_gallery, size=frame_len, replace=True))
    else:
      selected_field_ind = list(np.random.choice(all_field_ind_gallery, size=frame_len, replace=False))

  
  except pd.errors.EmptyDataError:
    frame_len = 4
    all_field_ind_gallery = np.arange(tensor1.size(2))
    random.shuffle(all_field_ind_gallery)
    if len(all_field_ind_gallery) < frame_len:
      index = list(np.random.choice(all_field_ind_gallery, size=frame_len, replace=True))
      selected_field_ind = [all_field_ind_gallery[k] for k in index]
    else:
      selected_field_ind = all_field_ind_gallery[:frame_len]
    print (openface_csv_path, " is empty")

  # GALLERY PREPROCESSING
  data1 = batch['image_g.pt']
  buffer1 = io.BytesIO(data1)
  tensor1 = torch.load(buffer1)

  selected_field_ind = torch.tensor(selected_field_ind)
  x_g = tensor1[:, :, selected_field_ind]

  # PROBE PREPROCESSING
  data2 = batch['image_p.pt']
  buffer2 = io.BytesIO(data2)
  tensor2 = torch.load(buffer2)
  

  all_field_ind_probe = np.arange(tensor2.size(2))
  random.shuffle(all_field_ind_probe)
  if len(all_field_ind_probe) < frame_len:
    index = list(np.random.choice(all_field_ind_probe, size=frame_len, replace=True))
    selected_field_ind_probe = [all_field_ind_probe[k] for k in index]
  else:
    selected_field_ind_probe = all_field_ind_probe[:frame_len]
  selected_field_ind_probe = torch.tensor(selected_field_ind_probe)
  x_p = tensor2[:, :, selected_field_ind_probe]


 
  # gallery_paths = list(itertools.chain(*json_dict['gallery']))
  # gallery_paths = [gallery_paths[k] for k in selected_field_ind]
  # probe_frames = [json_dict['probe_frames'][k] for k in selected_field_ind_probe]
  all_subjects = [k.split('/')[1] for k in json_dict['probe']]
  all_subs_dict = {}
  cnt = 0
  for sub in all_subjects:
    if sub not in all_subs_dict:
      all_subs_dict[sub] = cnt
      cnt += 1
  
  labels = [all_subs_dict[sub] for sub in all_subjects]
  # new_json = {
  #   "gallery": gallery_paths,
  #   "probe": probe_frames
  # }

  # NORMALIZE
  mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1, 1)
  std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1, 1)
  x_g.sub_(mean).div_(std)
  x_p.sub_(mean).div_(std)

  batch['image_g.pt'] = x_g
  batch['image_p.pt'] = x_p
  batch['json'] = torch.tensor(labels)
  return batch


def preprocess(batch):
   # JSON PREPROCESSING
  json_bytes = batch['json']
  json_dict = json.loads(json_bytes.decode('utf-8'))

  # GALLERY PREPROCESSING
  data1 = batch['image_g.pt']
  buffer1 = io.BytesIO(data1)
  tensor1 = torch.load(buffer1)

  frame_len = 4
  all_field_ind_gallery = np.arange(tensor1.size(2))
  random.shuffle(all_field_ind_gallery)
  if len(all_field_ind_gallery) < frame_len:
    index = list(np.random.choice(all_field_ind_gallery, size=frame_len, replace=True))
    selected_field_ind = [all_field_ind_gallery[k] for k in index]
  else:
    selected_field_ind = all_field_ind_gallery[:frame_len]
  selected_field_ind = torch.tensor(selected_field_ind)
  x_g = tensor1[:, :, selected_field_ind]

  # PROBE PREPROCESSING
  data2 = batch['image_p.pt']
  buffer2 = io.BytesIO(data2)
  tensor2 = torch.load(buffer2)
  

  all_field_ind_probe = np.arange(tensor2.size(2))
  random.shuffle(all_field_ind_probe)
  if len(all_field_ind_probe) < frame_len:
    index = list(np.random.choice(all_field_ind_probe, size=frame_len, replace=True))
    selected_field_ind_probe = [all_field_ind_probe[k] for k in index]
  else:
    selected_field_ind_probe = all_field_ind_probe[:frame_len]
  selected_field_ind_probe = torch.tensor(selected_field_ind_probe)
  x_p = tensor2[:, :, selected_field_ind_probe]
  
 
  all_subjects = [k.split('/')[1] for k in json_dict['probe']]
  all_subs_dict = {}
  cnt = 0
  for sub in all_subjects:
    if sub not in all_subs_dict:
      all_subs_dict[sub] = cnt
      cnt += 1
  
  labels = [all_subs_dict[sub] for sub in all_subjects]
  # NORMALIZE
  mean = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1, 1)
  std = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1, 1)
  x_g.sub_(mean).div_(std)
  x_p.sub_(mean).div_(std)
  # normalize = GroupNormalize(mean.squeeze(), std.squeeze())
  # transform = transforms.Compose([ 
  #   Stack(roll=False),
  #   ToTorchFormatTensor(div=True),
  #   normalize,
  # ])
  
  # _, c, t, h, w = x_g.size()
  # imgs = [ ToPILImage()(x_g.squeeze(0)[:, vid, :, :].cpu()) for vid in range(x_g.shape[2])  ]
  # x_g = transform(imgs)
  # x_g = rearrange(x_g, '(b c t) h w -> b c t h w', b=1, c=c, t=t, h=h, w=w)
  
  # imgs = [ ToPILImage()(x_p.squeeze(0)[:, vid, :, :].cpu()) for vid in range(x_p.shape[2])  ]
  # x_p = transform(imgs)
  # x_p = rearrange(x_p, '(b c t) h w -> b c t h w', b=1, c=c, t=t, h=h, w=w)

  # ic(x_g.size(), x_p.size())

  batch['image_g.pt'] = x_g
  batch['image_p.pt'] = x_p
  batch['json'] = torch.tensor(labels)
  return batch

def nodesplitter(src, group=None):
    if torch.distributed.is_initialized():
        if group is None:
            group = torch.distributed.group.WORLD
        rank = torch.distributed.get_rank(group=group)
        size = torch.distributed.get_world_size(group=group)
        
        print(f"nodesplitter: rank={rank} size={size}")
        count = 0
        for i, item in enumerate(src):
            if i % size == rank:
                yield item
                count += 1
        print(f"nodesplitter: rank={rank} size={size} count={count} DONE")
    else:
        yield from src

def prepare_loader(cfg, args):

  # train_url = "./data/face_chips/shard-{000000..000010}.tar"
  # val_url = "./data/face_chips/shard-{000011..000014}.tar"

  train_url = "./data/face_chips/shard-{000000..001037}.tar"
  val_url = "./data/face_chips/val/shard-{000000..000200}.tar"

  if cfg.TRAINING.DISTRIBUTED:
    train_urls = list(braceexpand.braceexpand(train_url))
    val_urls = list(braceexpand.braceexpand(val_url))

    train_ds_size = len(train_urls) * 10
    val_ds_size = len(val_urls) * 10

      # .map(preprocess)\
    train_dataset = wds.WebDataset(train_urls, repeat=False, shardshuffle=False, resampled=True, handler=wds.ignore_and_continue,  nodesplitter=nodesplitter)\
      .map(preprocess_openface)\
      .to_tuple("image_g.pt", "image_p.pt", "json")
      
      # .map(preprocess)\
    val_dataset = wds.WebDataset(val_urls, repeat=False, shardshuffle=False, resampled=True, handler=wds.ignore_and_continue,  nodesplitter=nodesplitter)\
      .map(preprocess_openface_val)\
      .to_tuple("image_g.pt", "image_p.pt", "json")
  else:
    train_dataset = wds.WebDataset(train_url)
    val_dataset = wds.WebDataset(val_url)

    train_dataset = train_dataset\
            .map(preprocess)\
            .to_tuple("image_g.pt", "image_p.pt", "json")\
            .shuffle(5000)\
            .batched(cfg.TRAINING.BATCH_SIZE, partial=False)
    val_dataset = val_dataset\
              .map(preprocess)\
              .to_tuple("image_g.pt", "image_p.pt", "json")\
              .shuffle(0)\
              .batched(cfg.TRAINING.BATCH_SIZE, partial=False) 


  if cfg.TRAINING.DISTRIBUTED:
    world_size =  dist.get_world_size()
    
    train_n_batches =  max(1, train_ds_size // (cfg.TRAINING.BATCH_SIZE * world_size))
    train_loader = wds.WebLoader(train_dataset, batch_size=None, shuffle=False, num_workers=0)
    train_loader = train_loader.unbatched().shuffle(1000).batched(cfg.TRAINING.BATCH_SIZE).with_epoch(train_n_batches)

    val_n_batches =  max(1, val_ds_size // (cfg.TRAINING.BATCH_SIZE * world_size))
    val_loader = wds.WebLoader(val_dataset, batch_size=None, shuffle=False, num_workers=0)
    val_loader = val_loader.unbatched().shuffle(0).batched(cfg.TRAINING.BATCH_SIZE).with_epoch(val_n_batches)
    # train_loader = wds.WebLoader(train_dataset, batch_size=None, shuffle=False, num_workers=0)
    # val_loader = wds.WebLoader(val_dataset, batch_size=None, shuffle=False, num_workers=0)

  else:
    train_loader = wds.WebLoader(train_dataset, batch_size=None, shuffle=False, num_workers=cfg.DATASET.NUM_WORKERS)
    val_loader = wds.WebLoader(val_dataset, batch_size=None, shuffle=False, num_workers=cfg.DATASET.NUM_WORKERS)

  return train_loader, val_loader


def save_images(process_data, mode):
  imgs = [ ToPILImage()(process_data[:, vid, :, :].cpu().clamp(-1.,1.)) for vid in range(process_data.shape[1])  ]
  for id, im in enumerate(imgs):
    im.save(f"./data/vid_loader/{mode}_{id}_1.jpg")

def validate(loader, model, criterion):
  total_loss = []
  model.eval()
  for b, (x_g, x_p, label) in enumerate(loader):
    x = torch.cat([x_g, x_p], dim=2)
    x = x.cuda()
    label = label.cuda()
    z_gallery, z_probe = model(x)
    feats = torch.stack([z_gallery, z_probe], dim=1)
    
    loss = criterion(feats, label)
    loss = loss.mean()
    total_loss.append(loss.item())
  
  loss_tensor = torch.tensor([total_loss], device=args.gpu)
  dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
  loss_tensor /= dist.get_world_size()
  return loss_tensor.mean().item()

def train(loader, model, optimizer, criterion):
  total_loss = []
  model.train()
  for b, (x_g, x_p, label) in enumerate(loader):
    x = torch.cat([x_g, x_p], dim=2)
    x = x.cuda()
    label = label.cuda()

    z_gallery, z_probe = model(x)

    feats = torch.stack([z_gallery, z_probe], dim=1)
    loss = criterion(feats, label)
    loss = loss.mean()

    with torch.no_grad():
      total_loss.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  # total_loss = sum(total_loss)/len(total_loss)
  loss_tensor = torch.tensor([total_loss], device=args.gpu)
  dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
  loss_tensor /= dist.get_world_size()
  return loss_tensor.mean().item()




if __name__ == "__main__":
  seed_everything(42)

  torch.autograd.set_detect_anomaly(True)
  torch.cuda.empty_cache()
  torch.autograd.set_detect_anomaly(True)
  print("GPU: ", torch.cuda.is_available())

  args = get_args()
  db = args.config.split("_")[0]
  config_path = os.path.join(f"./configs/experiments/", f"{args.config}.yaml")

  ckp_path = f"./checkpoints/{args.config}"
  mkdir(ckp_path)

  writer = SummaryWriter(f'./runs/{args.config}')

  # LOAD CONFIGURATION
  cfg = get_cfg_defaults()
  cfg.merge_from_file(config_path)
  cfg.freeze()

  # DDP TRAINING
  if cfg.TRAINING.DISTRIBUTED:
    ddp_setup(args)
    ic("RANK: ", args.rank, args.gpu)

  # LOADER
  train_loader, val_loader = prepare_loader(cfg, args)
  
  # FACE MODEL
  device = torch.device(f"cuda:{args.rank}" if torch.cuda.is_available() else "cpu")
  face_model = FaceFeatureModel(cfg)
  if cfg.TRAINING.DISTRIBUTED:
    # face_model = face_model.to(args.rank % torch.cuda.device_count())
    face_model = face_model.to(args.gpu )
    face_model = DDP(face_model, device_ids=[args.gpu], find_unused_parameters=False, output_device=args.gpu) 
  else:
    # face_model.to(torch.device(args.gpu))
    face_model = face_model.to(device)
  print("number of params: ", sum(p.numel() for p in face_model.parameters() if p.requires_grad))

  # CRITERION 
  criterion = SupConLoss()
  optimizer = optim.AdamW(face_model.parameters(), lr=cfg.TRAINING.LR, betas=(0.9, 0.99), weight_decay=cfg.TRAINING.WT_DECAY)

  # TRAINING
  min_loss = 1e5
  if cfg.TRAINING.DISTRIBUTED:
    is_master = args.rank == 0
    pbar = tqdm(range(cfg.TRAINING.ITER), disable=not is_master)
  else:
    pbar = tqdm(range(cfg.TRAINING.ITER))
  for epoch in pbar:
    train_loss = train(train_loader, face_model, optimizer, criterion)
    with torch.no_grad():
      val_loss = validate(val_loader, face_model, criterion)


    if val_loss < min_loss:
      min_loss = val_loss
      if cfg.TRAINING.DISTRIBUTED and args.rank == 0:
        ckp_dict = {
          'state_dict': face_model.module.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'args': cfg
        }
        torch.save(ckp_dict, f"{ckp_path}/model_final.pt")
      elif not cfg.TRAINING.DISTRIBUTED:
        ckp_dict = {
            'state_dict': face_model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'args': cfg
          }
        torch.save(ckp_dict, f"{ckp_path}/model_final.pt")

    # UDPATE: TOO MANY IFS
    if cfg.TRAINING.DISTRIBUTED and args.rank == 0:
      if epoch%10==0:
        torch.save(ckp_dict, f"{ckp_path}/model_epoch_{epoch}.pt")
    elif not cfg.TRAINING.DISTRIBUTED:
      if epoch%50==0:
        torch.save(ckp_dict, f"{ckp_path}/model_epoch_{epoch}.pt")


    pbar.set_description(
      f"Loss/Train: {round(train_loss, 4)};"\
      f"Loss/Val: {round(val_loss, 4)};"\
      )
    
    
    writer.add_scalar("Loss/Train", round(train_loss, 4), epoch)
    writer.add_scalar("Loss/Val", round(val_loss, 4), epoch)