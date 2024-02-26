import sys
sys.path.append('.')

import torch 
from torch import optim, nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# DISTRIBUTED COMPUTING
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler # loader
from torch.nn.parallel import DistributedDataParallel as DDP 
from torch.distributed import init_process_group

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
      ic("DISTRIBTUED")
      args.rank = args.local_rank
      args.gpu = args.local_rank
    elif 'SLURM_PROCID' in os.environ: # for slurm scheduler
      ic("SLURM")
      args.rank = int(os.environ['SLURM_PROCID'])
      args.gpu = int(os.environ['SLURM_LOCALID'])
    init_process_group(backend='nccl', world_size=args.world_size, rank=args.rank)
    if args.rank!=0:
      def print_pass(*args,  **kwargs):
          pass
      builtins.print = print_pass

    torch.cuda.set_device(args.gpu)
    # torch.distributed.barrier()

def prepare_loader(cfg):
  train_dataset = BGCDataset(cfg, "train")
  val_dataset = BGCDataset(cfg, "val")
  
  if cfg.TRAINING.DISTRIBUTED:
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=False, \
      num_workers=cfg.DATASET.NUM_WORKERS, pin_memory=True, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=False, \
      num_workers=cfg.DATASET.NUM_WORKERS, pin_memory=True, sampler=val_sampler)
  else:
    train_loader = DataLoader(train_dataset, batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=True, num_workers=cfg.DATASET.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=False, num_workers=cfg.DATASET.NUM_WORKERS)

  return train_loader, val_loader


def validate(loader, model, criterion):
  total_loss = []
  model.eval()
  for b, (x, label) in enumerate(loader):
    x = x.cuda()
    z_gallery, z_probe = model(x)
    feats = torch.stack([z_gallery, z_probe], dim=1)
    
    loss = criterion(feats)
    loss = loss.mean()
    total_loss.append(loss.item())

  total_loss = sum(total_loss)/len(total_loss)
  return total_loss

def train(loader, model, optimizer, criterion):
  total_loss = []
  model.train()
  for b, (x, label) in enumerate(loader):
    x = x.cuda()
    z_gallery, z_probe = model(x)

    feats = torch.stack([z_gallery, z_probe], dim=1)
    print(feats.size())
    loss = criterion(feats)
    loss = loss.mean()


    with torch.no_grad():
      total_loss.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  total_loss = sum(total_loss)/len(total_loss)
  return total_loss

if __name__ == "__main__":
  seed_everything(42)

  torch.autograd.set_detect_anomaly(True)
  torch.cuda.empty_cache()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

  # LOADER
  train_loader, val_loader = prepare_loader(cfg)
  
  # FACE MODEL
  face_model = FaceFeatureModel(cfg)
  if cfg.TRAINING.DISTRIBUTED:
    face_model = face_model.to(args.gpu)
    face_model = DDP(face_model, device_ids=[args.gpu], find_unused_parameters=True) 
  else:
    face_model.to(torch.device(args.gpu))
  print("number of params: ", sum(p.numel() for p in face_model.parameters() if p.requires_grad))
  face_model = face_model.to(device)

  # CRITERION 
  criterion = SupConLoss()
  optimizer = optim.AdamW(face_model.parameters(), lr=cfg.TRAINING.LR, betas=(0.9, 0.99), weight_decay=cfg.TRAINING.WT_DECAY)

  # TRAINING
  min_loss = 1e5
  pbar = tqdm(range(cfg.TRAINING.ITER))
  for epoch in pbar:
    print(f"Epoch: {epoch}")
    train_loss = train(train_loader, face_model, optimizer, criterion)
    with torch.no_grad():
      val_loss = validate(train_loader, face_model, criterion)


    if val_loss < min_loss:
      min_loss = val_loss
      if args.rank == 0:
        ckp_dict = {
          'state_dict': face_model.module.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'args': cfg
        }
        torch.save(ckp_dict, f"{ckp_path}/model_final.pt")

        if epoch%10==0:
          torch.save(ckp_dict, f"{ckp_path}/model_epoch_{epoch}.pt")


    pbar.set_description(
      f"Loss/Train: {round(train_loss, 4)};"\
      f"Loss/Val: {round(val_loss, 4)};"\
      )
    
    writer.add_scalar("Loss/Train", round(train_loss, 4), epoch)
    writer.add_scalar("Loss/Val", round(val_loss, 4), epoch)