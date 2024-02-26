import webdataset as wds
import torch 
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToPILImage
import io

from utils import *
from imports import *
from configs import get_cfg_defaults
from dataset import YoutubeDS
from transforms import *

class DataAugmentationForVideoMAE(object):
  def __init__(self):
    # self.input_mean = [0.485, 0.456, 0.406]  # IMAGENET_DEFAULT_MEAN
    # self.input_std = [0.229, 0.224, 0.225]  # IMAGENET_DEFAULT_STD
    self.input_mean = [0.5, 0.5, 0.5]  # IMAGENET_DEFAULT_MEAN
    self.input_std = [0.5, 0.5, 0.5]  # IMAGENET_DEFAULT_STD
    normalize = GroupNormalize(self.input_mean, self.input_std)
    # self.train_augmentation = GroupMultiScaleCrop(cfg.DATASET.IMG_SIZE, [1, .875, .75, .66])
    self.train_augmentation = GroupScale((224, 224))
    self.transform = transforms.Compose([                            
        self.train_augmentation,
        Stack(roll=False),
        ToTorchFormatTensor(div=True),
        normalize,
    ])

  def __call__(self, images):
    process_data, _ = self.transform(images)
    return process_data

  def __repr__(self):
    repr = "(DataAugmentationForVideoMAE,\n"
    repr += "  transform = %s,\n" % str(self.transform)
    repr += "  Masked position generator = %s,\n" % str(self.masked_position_generator)
    repr += ")"
    return repr

def preprocess(batch):
  transform = DataAugmentationForVideoMAE()

  data = batch['image_g.pt']
  buffer = io.BytesIO(data)
  tensor = torch.load(buffer).squeeze()
  x_g = tensor[:, :]

  meta_g = batch['label_g.cls'].decode("utf-8")

  data = batch['image_p.pt']
  buffer = io.BytesIO(data)
  tensor = torch.load(buffer).squeeze()
  x_p = tensor[:, :]

  meta_p = batch['label_p.cls'].decode("utf-8")

  batch['image_g.pt'] = x_g
  batch['label_g.cls'] = meta_g
  batch['image_p.pt'] = x_p
  batch['label_p.cls'] = meta_p
  return batch




def label_decoder(sample):
  for key, value in sample.items():
    if key.endswith(".pt"):
      # Convert the byte stream back to a PyTorch tensor
      buffer = io.BytesIO(value)
      tensor = torch.load(buffer)
      sample[key] = tensor.squeeze()
    if key == 'label.cls':
      sample[key] =  value.decode("utf-8")
  return sample



# class CustomTarDataset(Dataset):
#   def __init__(self, tar_paths):
        
#     dataset = wds.WebDataset(tar_paths).shuffle(32) \
#               .map(label_decoder)\
#               .to_tuple("image.pt", "label.cls")

#   def __len__(self):
#     return len()


if __name__ == "__main__":
  

  # shard_path = "./data/chips/shard-000000.tar"
  # dataset = wds.WebDataset(shard_path)
  # dataloader = DataLoader(dataset, batch_size=4)
  
  args = get_args()
  seed_everything(42)

  torch.autograd.set_detect_anomaly(True)
  db = args.config.split("_")[0]
  config_path = os.path.join(f"./configs/experiments/{db}", f"{args.config}.yaml")
  ckp_path = os.path.join(f"./checkpoints/{db}")
  mkdir(ckp_path)
  
  #  LOAD CONFIGURATION
  cfg = get_cfg_defaults()
  cfg.merge_from_file(config_path)
  cfg.freeze()
  print(cfg)
  
  url = "./data/chips/shard-{000001..000004}.tar"
  
  dataset = wds.WebDataset(url)
  dataset = dataset.shuffle(5)\
            .map(preprocess)\
            .to_tuple("image_g.pt", "label_g.cls", "image_p.pt", "label_p.cls")
  
  
  dataloader = DataLoader(dataset, num_workers=0, batch_size=1)
  
  for x_gallery, labelg, x_probe, labelp in dataloader:
    print(x_gallery[0].size(), labelg[0])
    
    # print(x_probe.size(), labelp)
    imgs = [ ToPILImage()(x_gallery[0][:, vid, :, :].cpu().clamp(0,0.996)) for vid in range(x_gallery.shape[2])  ]
    for id, im in enumerate(imgs):
      im.save(f"./data/loader/rec_img{id}.jpg")
    e()

    
    #---------------------------------------------------------------------------------------------------------------------------------
    
    
import webdataset as wds
import torch 
from torch.utils.data import DataLoader
import io

from utils import *
from imports import *
from configs import get_cfg_defaults
from dataset import YoutubeDS


if __name__ == "__main__":
  
  shard_path = "./data/chips/shard-000000.tar"
  dataset = wds.WebDataset(shard_path)
  dataloader = DataLoader(dataset, batch_size=4)
  
  args = get_args()
  seed_everything(42)

  torch.autograd.set_detect_anomaly(True)
  db = args.config.split("_")[0]
  config_path = os.path.join(f"./configs/experiments/{db}", f"{args.config}.yaml")
  ckp_path = os.path.join(f"./checkpoints/{db}")
  mkdir(ckp_path)
  
  #  LOAD CONFIGURATION
  cfg = get_cfg_defaults()
  cfg.merge_from_file(config_path)
  cfg.freeze()
  print(cfg)
  
  val_dataset = YoutubeDS(cfg, "val")
  val_loader = DataLoader(val_dataset, batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=True)
  
  output_shard_pattern = "./data/chips/shard-%06d.tar"
  
  sink =  wds.ShardWriter(output_shard_pattern, maxcount=2, maxsize=10e9, verbose=0)
  # sink =  wds.TarWriter(output_shard_pattern)
  for b, (x, label) in enumerate(val_loader): 
    ckp_dict = {
                'data': x.cpu().detach(),
                'class': str(label),
                '__key__': f"sample{b:06d}"
            }
    buffer = io.BytesIO()
    torch.save(x.contiguous().cpu().detach(), buffer)  # Move tensor to CPU before serialization
    image_bytes = buffer.getvalue()
    sample = {  
      "__key__": f"sample{b:06d}",
      "image_g.pt": image_bytes,
      "label_g.cls": str(label).encode("utf-8"),
      "image_p.pt": image_bytes,
      "label_p.cls": str(label).encode("utf-8")
      }
    sink.write(sample)
    
    if b == 10:
      break
  e()
    # for img_b in range(x.size(2)):
    #   x_tensor = x[:, :, img_b].squeeze().detach().cpu()
      
    #   buffer = io.BytesIO()
    #   torch.save(x_tensor.contiguous().cpu().detach(), buffer)  # Move tensor to CPU before serialization
    #   image_bytes = buffer.getvalue()
      
    #   # image_bytes = torch.save(x_tensor, io.BytesIO()).getvalue()
      
    #   sample = {
    #             "__key__": f"sample{b:06d}_{img_b:04d}",
    #             "image.pt": image_bytes,
    #             "label.cls": str(label).encode("utf-8")
    #         }
    #   sink.write(sample)
    
  
  
  
  
  
  