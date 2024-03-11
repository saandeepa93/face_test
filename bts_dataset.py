import os 
import pickle 


import torch 
from torch.utils.data import Dataset 
from torchvision import transforms
from torchvision.transforms import ToPILImage

from decord import VideoReader, cpu

from imports import *
from transforms import * 


#-----------------------------------one time helpers------------------------------------------------
def one_time_mode_save(all_subjects, mode, cfg):
  main_loop_dict = {}
  for subject_phase in all_subjects:
    subject_dict = load_pickle(subject_phase, cfg.PATHS.ANNOT)
    if len(subject_dict['controlled']['fpath']) == 0 or len(subject_dict['field']['fpath']) == 0:
      all_subjects.remove(subject_phase)
    else:
      total_vids = len(subject_dict['field']['fpath'])
      all_vids = list(np.arange(total_vids))
      for v in all_vids:
        main_loop_dict[f"{subject_phase}-{v}"] = subject_dict['label']
    
  all_files = list(main_loop_dict.keys())
  all_labels = list(main_loop_dict.values())

  with open(f'/shares/rra_sarkar-2135-1003-00/faces/face_verification/data/all_files_{mode}.pkl', 'wb') as f:
      pickle.dump(all_files, f)
  with open(f'/shares/rra_sarkar-2135-1003-00/faces/face_verification/data/all_labels_{mode}.pkl', 'wb') as f:
      pickle.dump(all_labels, f)
  ic("here: ", len(all_files))

def one_time_save(all_subjects, cfg):
  all_pickle_path = os.path.join(cfg.PATHS.ANNOT, "main_rel.pickle")
  with open(all_pickle_path, 'rb') as inp:
    all_subjects = pickle.load(inp)
    
  train_len = int(0.8 * len(all_subjects))

  random.shuffle(all_subjects)
  train_subjects = all_subjects[:train_len]
  val_subjects = all_subjects[train_len:]
  ic(len(train_subjects), len(val_subjects), len(all_subjects))
  one_time_mode_save(train_subjects, "train", cfg)
  one_time_mode_save(val_subjects, "val", cfg)

def load_pickle(subject, root):
  fpath = os.path.join(root, f"{subject}")
  with open(fpath, 'rb') as inp:
    pickl_file = pickle.load(inp)
  return pickl_file

def save_images(cfg, process_data, subject, mode):
  imgs = [ ToPILImage()(process_data[:, vid, :, :].cpu().clamp(-1.,1.)) for vid in range(process_data.shape[1])  ]
  for id, im in enumerate(imgs):
    im.save(f"{cfg.PATHS.VIS_PATH}/{mode}_{subject}_rec_img{id}.jpg")


#-----------------------------------Augmentations------------------------------------------------
class DataAugmentationForVideoMAE(object):
  def __init__(self, cfg):
    self.input_mean = [0.5, 0.5, 0.5]  # IMAGENET_DEFAULT_MEAN
    self.input_std = [0.5, 0.5, 0.5]  # IMAGENET_DEFAULT_STD
    normalize = GroupNormalize(self.input_mean, self.input_std)
    self.train_augmentation = GroupScale((cfg.DATASET.IMG_SIZE, cfg.DATASET.IMG_SIZE))
    self.transform = transforms.Compose([                            
        self.train_augmentation,
        Stack(roll=False),
        # ADD ANY AUG BEFORE THIS LINE
        ToTorchFormatTensor(div=True),
        normalize,
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


class BTSTarDataset(Dataset):
  def __init__(self, cfg, mode) -> None:
    super().__init__()

    self.cfg = cfg 
    self.mode = mode 
    
    self.transform = DataAugmentationForVideoMAE(cfg)

    all_files = glob.glob(f"{cfg.PATHS.ANNOT}/{mode}/*.pickle")
    print(len(all_files))

    if self.mode in ["Gallery1", "Gallery2"]:
      ic("In ", self.mode)
      self.all_files = []
      for file_path in all_files:
        pickle_file = load_pickle(f"{file_path}", self.cfg.PATHS.ANNOT)
        if len(pickle_file['jpeg']['bbox']) == 0:
          continue
        self.all_files.append(file_path)
    else:
      self.all_files = all_files

    print(len(self.all_files))
    # random.shuffle(self.all_files)
    self.controlled_process_data_dict = {}
    self.completed = []

  def __len__(self):
    return len(self.all_files)


  def _bboxCrop(self, img_arr, bbox, mode):
    if mode in ["Gallery1", "Gallery2"]:
      x1, y1, x2, y2 = bbox[0]
    else:
      x1, y1, x2, y2 = bbox
    x1 = int(max(0, x1))
    y1 = int(max(0, y1))
    x2 = int(max(0, x2))
    y2 = int(max(0, y2))
    crop_img = img_arr[y1:y2, x1:x2, :]
    crop_img = Image.fromarray(np.array(crop_img)[:, :, ::-1]).convert('RGB')
    return crop_img

  def _loadFrames(self, fpaths, bboxes, mode):
    cropped_images = []
    for fpath, bbox in zip(fpaths, bboxes):
      image = Image.open(fpath).convert('RGB')
      img_arr = np.array(image)
      crop_img = self._bboxCrop(img_arr, bbox, mode)
      cropped_images.append(crop_img)
    return cropped_images
  
  def _loadFieldVid(self, fpath, frames, bboxes, mode):
    decord_vr = VideoReader(fpath, num_threads=1, ctx=cpu(0))
    buffer = decord_vr.get_batch(frames).asnumpy()
    cropped_images = []
    for img_arr, bbox in zip(buffer, bboxes):
      crop_img = self._bboxCrop(img_arr, bbox, mode)
      cropped_images.append(crop_img)
    return cropped_images

  def __getitem__(self, idx):
    subject_pickle = self.all_files[idx]
    subject = subject_pickle.split("/")[-1].split("_")[0]
    subject_dict = load_pickle(f"{subject_pickle}", self.cfg.PATHS.ANNOT)

    if self.mode in ["Gallery1", "Gallery2"]:
        
      # ****LOAD CONTROLLED FRAMES****
      controlled_fpaths = subject_dict['jpeg']['fpath']
      controlled_bboxes = subject_dict['jpeg']['bbox']
      all_controlled_frames = np.arange(len(controlled_fpaths))

      selected_field_ind = all_controlled_frames
      selected_controlled_frames = selected_field_ind
      selected_controlled_fpaths = [controlled_fpaths[k] for k in selected_controlled_frames]
      selected_controlled_bboxes = [controlled_bboxes[k] for k in selected_controlled_frames]

      images = self._loadFrames(selected_controlled_fpaths, selected_controlled_bboxes, self.mode)
      process_data = self.transform(images)

      process_data_g = process_data.contiguous().view((len(selected_controlled_frames), 3) + process_data.size()[-2:]).transpose(0,1)

      # # ****LOAD FIELD VIDEO****
      # img_all = []
      # for selected_vid in range(len( subject_dict['mp4']['frame_num'])):
      #   tmp_path =  subject_dict['mp4']['fpath'][selected_vid]
      #   p1 = self.cfg.PATHS.ROOT
      #   p2 = f"BGC{tmp_path.split('/')[3][-1]}"
      #   p3 = tmp_path.replace("../faces/face_verification", "")
      #   field_fpath = f"{p1}{p2}{p3}"
        
      #   # field_fpath = os.path.join(self.cfg.PATHS.ROOT, subject_dict['mp4']['fpath'][selected_vid])
      #   all_field_frames = subject_dict['mp4']['frame_num'][selected_vid]
      #   all_field_bboxes = subject_dict['mp4']['bbox'][selected_vid]
        
      #   all_field_ind = np.arange(len(all_field_frames))

      #   if len(all_field_ind) > 100:
      #     selected_field_ind = random.sample(list(all_field_ind), 100)
      #   else:
      #     selected_field_ind = all_field_ind


      #   selected_field_frames = [all_field_frames[k] for k in selected_field_ind]
      #   selected_field_bboxes = [all_field_bboxes[k] for k in selected_field_ind]

      #   images = self._loadFieldVid(field_fpath, selected_field_frames, selected_field_bboxes)
      #   process_data = self.transform(images)

      #   process_data_p = process_data.contiguous().view((len(selected_field_ind), 3) + process_data.size()[-2:]).transpose(0,1)
      # img_all.append(process_data)
      # # save_images(self.cfg, process_data, subject, "field")
      # cls_p = f"{subject}"

      # ic(process_data_g.size(), len(img_all))
      meta_info = {
        "gallery": selected_controlled_fpaths,
        # "probe": cls_p, 
        # "probe_frames": selected_field_frames 
      }
      print(process_data_g.size())
      return process_data_g, subject

    else:
      probe_fpath = subject_dict['mp4']['fpath']
      probe_fname = probe_fpath.split("/")[-1].split(".")[0]
      all_field_frames = subject_dict['mp4']['frame_num']
      all_field_bboxes = subject_dict['mp4']['bbox']

      all_field_ind = np.arange(len(all_field_frames))

      if len(all_field_ind) > 32:
        selected_field_ind = random.sample(list(all_field_ind), 32)
      else:
        selected_field_ind = all_field_ind
      
      selected_field_frames = [all_field_frames[k] for k in selected_field_ind]
      selected_field_bboxes = [all_field_bboxes[k] for k in selected_field_ind]

      images = self._loadFieldVid(probe_fpath, selected_field_frames, selected_field_bboxes, self.mode)
      process_data = self.transform(images)

      process_data_p = process_data.contiguous().view((len(selected_field_ind), 3) + process_data.size()[-2:]).transpose(0,1)

      meta_info = {
        # "gallery": selected_controlled_fpaths,
        "probe": probe_fpath, 
        "probe_frames": selected_field_frames 
      }
      return process_data_p, probe_fname


