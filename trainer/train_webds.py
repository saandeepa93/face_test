import sys
sys.path.append('.')
import io
import braceexpand
import tarfile

import torch 
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# DISTRIBUTED COMPUTING
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler # loader
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.distributed import init_process_group
from torchvision.transforms import ToPILImage
import torch.distributed as dist
from torchvision import transforms

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
    init_process_group(backend='gloo', world_size=args.world_size, rank=args.rank)
    # if args.rank!=0:
    #   def print_pass(*args,  **kwargs):
    #       pass
    #   builtins.print = print_pass

    torch.cuda.set_device(args.gpu)
    

def sample_frames(frame_len, all_field_ind_gallery, mode=False):
  if len(all_field_ind_gallery) < frame_len:
    selected_field_ind = list(np.random.choice(all_field_ind_gallery, size=frame_len, replace=True))
  else:
    if mode is False:
      selected_field_ind = list(np.random.choice(all_field_ind_gallery, size=frame_len, replace=False))
    else:
      selected_field_ind = all_field_ind_gallery[:frame_len]
  return selected_field_ind


def preprocess(batch, cfg, openface_mode):

  scale = lambda x: (x - x.min()) / (x.max() - x.min())
  frame_len = cfg.DATASET.CLIP_LGT

   # JSON PREPROCESSING
  json_bytes = batch['json']
  json_dict = json.loads(json_bytes.decode('utf-8'))

  # GALLERY PREPROCESSING
  data1 = batch['image_g.pt']
  buffer1 = io.BytesIO(data1)
  tensor1 = torch.load(buffer1)
  all_field_ind_gallery = np.arange(tensor1.size(2))

  meta_info = json_dict['probe'][0].split('/')
  phase = meta_info[0]
  subject = meta_info[1]

  if cfg.TRAINING.OF:
    #GALLERY

    all_controlled_fnames = json_dict['gallery']
    all_controlled_fnames = list(itertools.chain.from_iterable(all_controlled_fnames))
    all_controlled_fnames = [k.split('/')[-1] for k in all_controlled_fnames]

    try:
      gallery_csv_dir_train = os.path.join(cfg.PATHS.OPENFACE, openface_mode, "gallery")
      openface_csv_path = os.path.join(gallery_csv_dir_train, phase, subject, "face/face.csv")
      df_openface = pd.read_csv(openface_csv_path)

      if cfg.TRAINING.OF_TYPE == "headpose":
        df_openface.loc[1:, 'pose_Rx'] = np.degrees(df_openface['pose_Rx'])
        df_openface.loc[1:,'pose_Ry'] = np.degrees(df_openface['pose_Ry'])
        df_openface.loc[1:, 'pose_Rz'] = np.degrees(df_openface['pose_Rz'])

        df_openface['magnitude'] = np.sqrt(df_openface['pose_Rx']**2 + df_openface['pose_Ry']**2 + df_openface['pose_Rz']**2)
        df_openface = df_openface.sort_values(by='magnitude', ascending=True)

        all_openface_selected = df_openface['frame_name'].tolist()
        all_openface_indeces = [all_controlled_fnames.index(b) for b in all_openface_selected if b in all_controlled_fnames]
      
        all_field_ind_gallery = all_openface_indeces
        selected_field_ind = sample_frames(frame_len, all_field_ind_gallery)

      elif cfg.TRAINING.OF_TYPE == "AU":
        au_int = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]
        au_int_cols = [f"AU{str(AU).zfill(2)}_r" for AU in au_int]

        magnitude_fn = lambda row: np.sqrt(np.sum(row[au_int_cols] ** 2))

        df_openface['magnitude'] = df_openface.apply(magnitude_fn, axis=1)
        df_openface = df_openface.sort_values(by='magnitude', ascending=False)

        all_openface_selected = df_openface['frame_name'].tolist()
        all_openface_indeces = [all_controlled_fnames.index(b) for b in all_openface_selected if b in all_controlled_fnames]

        all_field_ind_gallery = all_openface_indeces
        selected_field_ind = sample_frames(frame_len, all_field_ind_gallery, True)
    
    except pd.errors.EmptyDataError:
      index = sample_frames(frame_len, all_field_ind_gallery)
      selected_field_ind = [all_field_ind_gallery[k] for k in index]

  else:
    index = sample_frames(frame_len, all_field_ind_gallery)
    selected_field_ind = [all_field_ind_gallery[k] for k in index]

  # SELECT GALLERY FRAMES
  selected_field_ind = torch.tensor(selected_field_ind)
  x_g = tensor1[:, :, selected_field_ind]



  # PROBE PREPROCESSING
  data2 = batch['image_p.pt']
  buffer2 = io.BytesIO(data2)
  tensor2 = torch.load(buffer2)

    # PROBE CSV
  distance = meta_info[2]
  dtype = meta_info[3]
  fname = meta_info[-1]
  all_field_ind_probe = np.arange(tensor2.size(2))

  if cfg.TRAINING.OF_PROBE:
    try:
      probe_csv_dir_train = os.path.join(cfg.PATHS.OPENFACE, openface_mode, "probe")
      openface_csv_path = os.path.join(probe_csv_dir_train, phase, subject, distance, dtype, fname, f"{fname}.csv")
      df_openface = pd.read_csv(openface_csv_path)

      if cfg.TRAINING.OF_PROBE_TYPE == "headpose":
        df_openface.loc[1:, 'pose_Rx'] = np.degrees(df_openface['pose_Rx'])
        df_openface.loc[1:,'pose_Ry'] = np.degrees(df_openface['pose_Ry'])
        df_openface.loc[1:, 'pose_Rz'] = np.degrees(df_openface['pose_Rz'])

        df_openface['magnitude'] = np.sqrt(df_openface['pose_Rx']**2 + df_openface['pose_Ry']**2 + df_openface['pose_Rz']**2)
        df_openface = df_openface.sort_values(by='magnitude', ascending=True)

      elif cfg.TRAINING.OF_PROBE_TYPE == "AU":
        au_int = [1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45]
        au_int_cols = [f"AU{str(AU).zfill(2)}_r" for AU in au_int]

        magnitude_fn = lambda row: np.sqrt(np.sum(row[au_int_cols] ** 2))

        df_openface['magnitude'] = df_openface.apply(magnitude_fn, axis=1)
        df_openface = df_openface.sort_values(by='magnitude', ascending=False)

      all_openface_indeces = df_openface['frame'].tolist()
      all_field_ind_probe = all_openface_indeces
      selected_field_ind_probe = sample_frames(frame_len, all_field_ind_probe, True)
        
    except pd.errors.EmptyDataError:
      index = sample_frames(frame_len, all_field_ind_probe)
      selected_field_ind_probe = [all_field_ind_probe[k] for k in index]
    except FileNotFoundError:
      index = sample_frames(frame_len, all_field_ind_probe)
      selected_field_ind_probe = [all_field_ind_probe[k] for k in index]

  else:
    index = sample_frames(frame_len, all_field_ind_probe)
    selected_field_ind_probe = [all_field_ind_probe[k] for k in index]

  # SELECT PROBE FRAMES
  selected_field_ind_probe = torch.tensor(selected_field_ind_probe)
  x_p = tensor2[:, :, selected_field_ind_probe]

  # SEND META INFO
  all_subjects = [k.split('/')[1] for k in json_dict['probe']]
  all_fields = [k.split('/')[2] for k in json_dict['probe']]
  meta = [f"{k}_{i}" for k, i in zip(all_subjects, all_fields)]

  # PREPROCESS BOTH GALLERY AND PROBE
  x_g = scale(x_g)
  x_p = scale(x_p)
  x_g = (x_g - 0.5)/0.5
  x_p = (x_p - 0.5)/0.5

  augmentations = transforms.RandomHorizontalFlip(p=0.5)  # Flip the image horizontally with a probability of 0.5
  
  x_g = augmentations(x_g)
  x_p = augmentations(x_p)

  # SET BATCH
  batch['image_g.pt'] = x_g
  batch['image_p.pt'] = x_p
  batch['json'] = meta
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

  train_url = "./data/face_chips/shard-{000000..002000}.tar"
  val_url = "./data/face_chips/val/shard-{000000..000200}.tar"


  preprocess_of_train = partial(preprocess, cfg=cfg, openface_mode="train")
  preprocess_of_val = partial(preprocess, cfg=cfg, openface_mode="val")

  if cfg.TRAINING.DISTRIBUTED:
    train_urls = list(braceexpand.braceexpand(train_url))
    val_urls = list(braceexpand.braceexpand(val_url))

    train_ds_size = len(train_urls) * 10
    val_ds_size = len(val_urls) * 10

      # .map(preprocess)\
    train_dataset = wds.WebDataset(train_urls, repeat=False, shardshuffle=False, resampled=True, handler=wds.ignore_and_continue,  nodesplitter=nodesplitter)\
      .map(preprocess_of_train)\
      .to_tuple("image_g.pt", "image_p.pt", "json")
      
      # .map(preprocess)\
    val_dataset = wds.WebDataset(val_urls, repeat=False, shardshuffle=False, resampled=True, handler=wds.ignore_and_continue,  nodesplitter=nodesplitter)\
      .map(preprocess_of_val)\
      .to_tuple("image_g.pt", "image_p.pt", "json")
  else:
    train_dataset = wds.WebDataset(train_url)
    val_dataset = wds.WebDataset(val_url)

    train_dataset = train_dataset\
            .map(preprocess_of_train)\
            .to_tuple("image_g.pt", "image_p.pt", "json")\
            .shuffle(5000)\
            .batched(cfg.TRAINING.BATCH_SIZE, partial=False)
    val_dataset = val_dataset\
              .map(preprocess_of_val)\
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


def getOpenfaceDict(cfg):
  openface_dict = {}
  avg_length = {}
  tar_files = glob.glob(os.path.join(cfg.PATHS.TAR, "*.tar"))
  for tar_path in tar_files:
    archive = tarfile.open(tar_path, 'r')
    for member in archive.getmembers():
      if "meta" in member.name:
        meta_file_obj = archive.extractfile(member)
        meta_content = meta_file_obj.read().decode('utf-8')
        df = pd.read_csv(os.path.join(cfg.PATHS.OFROOT, f"{meta_content}", f"{meta_content}.csv"))
        openface_dict[meta_content] = df
        
        task = meta_content.split('_')[1]
        if task not in avg_length:
          avg_length[task] = []
        else:
          avg_length[task].append(df.shape[0])
  
  for key, val in avg_length.items():
    ic(f"{key}: {sum(avg_length[key])/len(avg_length[key])}")
  return openface_dict


def random_indexer(b):
  my_list = list(np.arange(b))
  # The index of the element to keep fixed (e.g., keeping the element '3' fixed)
  fixed_index = random.randint(0, b-1)
  ver_label = [0 if k != fixed_index else 1 for k in range(len(my_list))]

  # Step 1 & 2: Remove the element temporarily
  fixed_element = my_list.pop(fixed_index)

  # Step 3: Shuffle the remaining elements
  random.shuffle(my_list)

  # Step 4: Insert the fixed element back to its original position
  my_list.insert(fixed_index, fixed_element)
  return torch.tensor(my_list), ver_label

def save_images(process_data, mode):
  imgs = [ ToPILImage()(process_data[:, vid, :, :].cpu().clamp(-1.,1.)) for vid in range(process_data.shape[1])  ]
  # imgs = [ ToPILImage()(process_data[:, vid, :, :].cpu()) for vid in range(process_data.shape[1])  ]
  for id, im in enumerate(imgs):
    im.save(f"./data/vid_loader/{mode}_{id}_1.jpg")

def validate(loader, model, criterion):
  total_loss = []
  model.eval()

  g_scores = []
  g_labels = []

  all_x_g = []
  all_x_p = []
  all_meta = []

  for b, (x_g, x_p, string_labels_joint) in enumerate(loader):
    batch, _, t, _, _ = x_p.size()
    shuffle_index, ver_label = random_indexer(x_g.size(0))
    x_g = x_g.cuda()
    x_p = x_p.cuda()

    # all_x_g.append(x_g)
    # all_x_p.append(x_p)
    # all_meta.append(string_labels_joint)

    x_p_clone = x_p.clone()[shuffle_index]
    x_g_clone = x_g.clone()

    string_labels = [k.split("_")[0] for k in string_labels_joint]
    string_fields = [k.split("_")[1] for k in string_labels_joint]
    
    unique_labels = set(string_labels)
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_labels = [label_to_int[label] for label in string_labels]
    tensor_labels = torch.tensor(int_labels)
    tensor_labels = tensor_labels.cuda()

    x_g_clone = rearrange(x_g_clone, 'b c t h w -> (b t) c h w')
    x_p_clone = rearrange(x_p_clone, 'b c t h w -> (b t) c h w')

    z_gallery, _= model.module.model(x_g_clone)
    z_probe, _= model.module.model(x_p_clone)

    f, d = z_gallery.size()
    feats_gallery = rearrange(z_gallery, '(b t) d -> b t d', b=batch, t=t, d=d)
    feats_probe = rearrange(z_probe, '(b t) d -> b t d', b=batch, t=t, d=d)

    p_xg = F.softmax(model.module.get_token_probs(feats_gallery), dim=-1)
    p_xg = torch.nan_to_num(p_xg)
    p_xg = p_xg.view(batch, t, 1)

    p_xp = F.softmax(model.module.get_token_probs(feats_probe), dim=-1)
    p_xp = torch.nan_to_num(p_xp)
    p_xp = p_xp.view(batch, t, 1)

    weighted_emb_gallery = F.normalize((feats_gallery * p_xg).sum(dim=1), dim=-1)
    weighted_emb_probe = F.normalize((feats_probe * p_xp).sum(dim=1), dim=-1)

    similarity_score = torch.diagonal(weighted_emb_gallery @ weighted_emb_probe.T)
    g_scores += similarity_score.cpu()
    g_labels += ver_label


    x = torch.cat([x_g, x_p], dim=2)
    z_gallery, z_probe = model(x)
    feats = torch.stack([z_gallery, z_probe], dim=1)
    loss = criterion(feats, tensor_labels)
    loss = loss.mean()
    total_loss.append(loss.item())

  
  score_matrix = np.array(g_scores).T
  mask_matrix = np.array(g_labels).T

  loss_tensor = torch.tensor([total_loss], device=args.gpu)
  dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
  loss_tensor /= dist.get_world_size()
  return loss_tensor.mean().item(), computeROC(mask_matrix, score_matrix)
  


def train(loader, model, optimizer, criterion):
  total_loss = []
  model.train()

  all_field_dict = {}
  for b, (x_g, x_p, string_labels_joint) in enumerate(loader):
    x = torch.cat([x_g, x_p], dim=2)
    x = x.cuda()

    string_labels = [k.split("_")[0] for k in string_labels_joint]
    string_fields = [k.split("_")[1] for k in string_labels_joint]

    unique_labels = set(string_labels)
    label_to_int = {label: i for i, label in enumerate(unique_labels)}
    int_labels = [label_to_int[label] for label in string_labels]
    tensor_labels = torch.tensor(int_labels)
    tensor_labels = tensor_labels.cuda()

    z_gallery, z_probe = model(x)

    feats = torch.stack([z_gallery, z_probe], dim=1)
    loss = criterion(feats, tensor_labels)
    loss = loss.mean()

    with torch.no_grad():
      total_loss.append(loss.item())

      # for field in string_fields:
      #   if field not in all_field_dict:
      #     all_field_dict[field] = 0
      #   else:
      #     all_field_dict[field] += 1

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  # total_loss = sum(total_loss)/len(total_loss)
  loss_tensor = torch.tensor([total_loss], device=args.gpu).detach()
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
  print(cfg)

  # DDP TRAINING
  if cfg.TRAINING.DISTRIBUTED:
    ddp_setup(args)
    ic("RANK: ", args.rank, args.gpu)

  # LOADER
  train_loader, val_loader = prepare_loader(cfg, args)
  
  # FACE MODEL
  device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
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
      val_loss, val_roc = validate(val_loader, face_model, criterion)


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
        ic(val_roc)
    elif not cfg.TRAINING.DISTRIBUTED:
      if epoch%50==0:
        torch.save(ckp_dict, f"{ckp_path}/model_epoch_{epoch}.pt")

    if dist.get_rank() == 0:
      pbar.set_description(
        f"Loss/Train: {round(train_loss, 4)};"\
        f"Loss/Val: {round(val_loss, 4)};"\
        )
    
    
    writer.add_scalar("Loss/Train", round(train_loss, 4), epoch)
    writer.add_scalar("Loss/Val", round(val_loss, 4), epoch)