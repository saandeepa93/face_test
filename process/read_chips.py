import os 
import io 

import torch 
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToPILImage

from decord import VideoReader, cpu
import webdataset as wds

from icecream import ic 
from sys import exit as e 
from transforms import * 
import pickle
from tqdm import tqdm





class DataAugmentationForVideoMAE(object):
  def __init__(self):
    self.input_mean = [0.5, 0.5, 0.5]  # IMAGENET_DEFAULT_MEAN
    self.input_std = [0.5, 0.5, 0.5]  # IMAGENET_DEFAULT_STD
    normalize = GroupNormalize(self.input_mean, self.input_std)
    self.train_augmentation = GroupScale((224, 224))
    self.transform = transforms.Compose([                            
        self.train_augmentation,
        Stack(roll=False),
        # ADD ANY AUG BEFORE THIS LINE
        ToTorchFormatTensor(div=True),
    ])

  def __call__(self, images):
    process_data = self.transform(images)
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
  x_g = tensor[:, :100]

  meta_g = batch['meta_g'].decode("utf-8")

  data = batch['image_p.pt']
  buffer = io.BytesIO(data)
  tensor = torch.load(buffer).squeeze()
  x_p = tensor[:, :]

  meta_p = batch['meta_p'].decode("utf-8")
  label = batch['label.cls'].decode("utf-8")

  batch['image_g.pt'] = x_g
  batch['label_g.cls'] = meta_g
  batch['image_p.pt'] = x_p
  batch['label_p.cls'] = meta_p
  return batch




if __name__ == "__main__":
  
  url = "./data/face_chips/shard-000000.tar"
  
  dataset = wds.WebDataset(url)
  dataset = dataset.shuffle(5)\
            .map(preprocess)\
            .to_tuple("image_g.pt", "meta_g", "image_p.pt", "meta_p", "label.cls")
  
  
  dataloader = DataLoader(dataset, num_workers=0, batch_size=1)

  for x_g, meta_g, x_p, meta_p, label in dataloader:
    print(x_g.size(), meta_g)
    print(x_p.size(), meta_p)
    e()