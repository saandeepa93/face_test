import sys
sys.path.append('.')

import torch 
from torch import optim, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

# DISTRIBUTED COMPUTING
import webdataset as wds

from einops import rearrange

from imports import * 
from utils import * 
from configs import get_cfg_defaults
from bts_dataset import BTSTarDataset
# from dataset import BGCDataset
from models import FaceFeatureModel
# from supCon import SupConLoss


class BTSDataset(Dataset):
  def __init__(self, cfg):
    super().__init__()

    self.cfg = cfg
    self.save = cfg.PATHS.ANNOT

    self.g1_df = pd.read_csv("./data/analysis/eval_4.0.0/Gallery1.csv")
    self.g2_df = pd.read_csv("./data/analysis/eval_4.0.0/Gallery2.csv")

    self.p_face_incl_ctrl_df = pd.read_csv("./data/analysis/eval_4.0.0/Probe_BTS_briar-rd_FaceIncluded_ctrl.csv")
    self.p_face_incl_trt_df = pd.read_csv("./data/analysis/eval_4.0.0/Probe_BTS_briar-rd_FaceIncluded_trt.csv")
    self.p_face_rest_ctrl_df = pd.read_csv("./data/analysis/eval_4.0.0/Probe_BTS_briar-rd_FaceRestricted_ctrl.csv")
    self.p_face_rest_trt_df = pd.read_csv("./data/analysis/eval_4.0.0/Probe_BTS_briar-rd_FaceRestricted_trt.csv")

    # self._getAllGallery(self.g1_df, "Gallery1")
    # self._getAllGallery(self.g2_df, "Gallery2")
    # self._getAllProbe(self.p_face_incl_ctrl_df, "face_incl_ctrl")
    # self._getAllProbe(self.p_face_incl_trt_df, "face_incl_trt")
    # self._getAllProbe(self.p_face_rest_ctrl_df, "face_rest_ctrl")
    # self._getAllProbe(self.p_face_rest_trt_df, "face_rest_trt")

    # YET TO RUN

  def _getAllGallery(self, df, mode):
    not_found = []
    subjects = df['subjectId'].apply(lambda x: x[2:-2]).unique().tolist()
    print(len(subjects))
    for sub in subjects:
      subject_dict = {
          "jpeg":{
            "fpath": [], "bbox": []
          },
          "mp4": {
            "fpath": [], "bbox": [], 'frame_num': []
          }
      } 
      df_filtered = df.loc[df['subjectId'] == f"['{sub}']", ["filepath", "media_format"]]
      df_filtered["bbox_path"] =  self.cfg.PATHS.FACE + "BGC" + df_filtered['filepath'].apply(lambda x: x.split('/')[1][-1]) + "/" + df_filtered['filepath'] + ".csv"
      df_filtered['filepath'] = self.cfg.PATHS.ROOT +  "BGC" + df_filtered['filepath'].apply(lambda x: x.split('/')[1][-1]) + "/" + df_filtered['filepath']

      if sub == "G02419":
        df_filtered.to_csv("./data/sampleG2.csv")
        e()

      cnt = 0
      for _, row in df_filtered.iterrows():
        fpath = row['filepath']
        bbox_path = row['bbox_path']

        if not os.path.isfile(bbox_path):
          continue
        try:
          face_df = pd.read_csv(bbox_path, header=None, usecols=[2, 3, 4, 5, 6])
        except pd.errors.EmptyDataError:
          print (bbox_path, " is empty")
          continue
        except pd.errors.ParserError:
          print (bbox_path, " THIS FILE")
          continue
        cols = ["frame", 'x1', 'y1', 'x2', 'y2']
        face_df.columns = cols
        face_df['bbox_lst'] = face_df.apply(lambda x: [x['x1'], x['y1'], x['x2'], x['y2']], axis=1)


        frame_num = face_df['frame'].tolist()
        bbox = face_df['bbox_lst'].tolist()

        if len(frame_num) == 0:
          continue

        if row['media_format'] == "mp4":
          subject_dict['mp4']['fpath'].append(fpath)
          subject_dict['mp4']['bbox'].append(bbox)
          subject_dict['mp4']['frame_num'].append(frame_num)
        elif row['media_format'] == "jpeg":
          subject_dict['jpeg']['fpath'].append(fpath)
          subject_dict['jpeg']['bbox'].append(bbox)
        cnt += 1

      if cnt == 0:
        print(f"Subject {sub} not found at all!!!")
        not_found.append(sub)
      else:
        with open(f'{self.save}/{mode}/{sub}_rel.pickle', 'wb') as f:
          pickle.dump(subject_dict, f, pickle.HIGHEST_PROTOCOL)
    # with open(f'./data/analysis/not_found.pickle', 'wb') as f:
    #   pickle.dump(not_found, f, pickle.HIGHEST_PROTOCOL)

  def _getAllProbe(self, df, mode):
    subject_dict = {
        "mp4": {
          "fpath": [], "bbox": [], 'frame_num': []
        }
      } 
    
    df["bbox_path"] =  self.cfg.PATHS.FACE + "BGC" + df['media_path'].apply(lambda x: x.split('/')[1][-1]) + "/" + df['media_path'] + ".csv"
    df['media_path'] = self.cfg.PATHS.ROOT +  "BGC" + df['media_path'].apply(lambda x: x.split('/')[1][-1]) + "/" + df['media_path']
    df['fname'] = df['media_path'].apply(lambda x: x.split('/')[-1].split('.')[0])
    # df.to_csv("./data/sampleP.csv")
    not_found_probe = []
    for _, row in df.iterrows():
      fpath = row["media_path"]
      bbox_path = row['bbox_path']
      
      if not os.path.isfile(bbox_path):
        continue
      try:
        face_df = pd.read_csv(bbox_path, header=None, usecols=[2, 3, 4, 5, 6])
      except pd.errors.EmptyDataError:
        print (bbox_path, " is empty")
        continue
      except pd.errors.ParserError:
        print (bbox_path, " THIS FILE")
        continue
      cols = ["frame", 'x1', 'y1', 'x2', 'y2']
      face_df.columns = cols
      face_df['bbox_lst'] = face_df.apply(lambda x: [x['x1'], x['y1'], x['x2'], x['y2']], axis=1)

      frame_num = face_df['frame'].tolist()
      bbox = face_df['bbox_lst'].tolist()
      if len(frame_num) == 0:
        not_found_probe.append(fpath)
        continue
      subject_dict['mp4']['fpath'] = fpath
      subject_dict['mp4']['bbox'] = bbox
      subject_dict['mp4']['frame_num'] = frame_num

      with open(f"{self.save}/{mode}/{row['fname']}_rel.pickle", 'wb') as f:
        pickle.dump(subject_dict, f, pickle.HIGHEST_PROTOCOL)

    with open(f'./data/analysis/not_found_probe.pickle', 'wb') as f:
      pickle.dump(not_found_probe, f, pickle.HIGHEST_PROTOCOL)



def extract_gallery_template(loader,mode):
  template_save_path = f"./data/templates/{mode}"
  subject_lst = []
  prob_lst = []
  feats_lst = []
  with torch.no_grad():
    for b, (x, subject) in enumerate(tqdm(loader)):
      btc, c, t, h, w = x.size()
      x = x.cuda()
      x = rearrange(x, 'b c t h w -> (b t) c h w')
      feats, _ = face_model.model(x)

      f, d = feats.size()
      feats = rearrange(feats, '(b t) d -> b t d', b=btc, t=t, d=d)

      p_x = F.softmax(face_model.get_token_probs(feats), dim=-1)
      p_x = torch.nan_to_num(p_x)
      p_x = p_x.view(btc, t, 1)

      weighted_emb =  F.normalize((feats * p_x).sum(dim=1), dim=-1)

      subject_lst += list(subject)
      prob_lst.append(p_x.cpu().squeeze().tolist())
      feats_lst.append(weighted_emb.cpu())


    embeddings = torch.cat(feats_lst, dim=0)
    meta_info = {}

    for i in range(embeddings.shape[0]):
      emb = embeddings[i]
      idx = subject_lst[i]
      probs = prob_lst[i]
      meta_info[idx] = probs
      torch.save(emb.cpu().detach(), os.path.join(template_save_path, f"{idx}_{i}.pt"))
    with open(os.path.join(template_save_path, f"meta_info_{mode}.json"), 'w') as fp:
      json.dump(meta_info, fp, indent=4)

def extract_probe_template(loader, mode):
  template_save_path = f"./data/templates/{mode}"
  meta_info = {}
  with torch.no_grad():
    for b, (x, subject) in enumerate(tqdm(loader)):
      btc, c, t, h, w = x.size()
      x = x.cuda()
      x = rearrange(x, 'b c t h w -> (b t) c h w')
      feats, _ = face_model.model(x)

      f, d = feats.size()
      feats = rearrange(feats, '(b t) d -> b t d', b=btc, t=t, d=d)

      p_x = F.softmax(face_model.get_token_probs(feats), dim=-1)
      p_x = torch.nan_to_num(p_x)
      p_x = p_x.view(btc, t, 1)

      weighted_emb =  F.normalize((feats * p_x).sum(dim=1), dim=-1)

      meta_info[subject[0]] = p_x.cpu().squeeze().tolist()
      torch.save(weighted_emb.cpu().detach(), os.path.join(template_save_path, f"{subject[0]}_{b}.pt"))
    
    with open(os.path.join(template_save_path, f"meta_info_{mode}.json"), 'w') as fp:
      json.dump(meta_info, fp, indent=4)

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

  ckp_config = "briar_8"
  ckp_path = f"./checkpoints/{ckp_config}"



  # LOAD CONFIGURATION
  cfg = get_cfg_defaults()
  cfg.merge_from_file(config_path)
  cfg.freeze()
  # dataset = BTSDataset(cfg)
  # e()

  # DDP TRAINING
  # output_shard_pattern = "./data/face_chips/test/4.0.0/shard-%06d.tar"
  # sink =  wds.ShardWriter(output_shard_pattern, maxcount=10, maxsize=10e9, verbose=0)

  mode_lst = ['Gallery1', 'Gallery2', 'face_incl_ctrl', 'face_incl_trt', 'face_rest_ctrl', 'face_rest_trt']
  # mode_lst = ['face_incl_ctrl', 'face_incl_trt', 'face_rest_ctrl', 'face_rest_trt']
  mode = "face_rest_trt"
 
  # LOAD MODEL
  face_model = FaceFeatureModel(cfg)
  face_model = face_model.to(device)
  # LOAD CKPT
  checkpt = torch.load(f"{ckp_path}/model_final.pt", map_location=device)
  sd = {k:v for k, v in checkpt['state_dict'].items()}
  state = face_model.state_dict()
  state.update(sd)
  face_model.load_state_dict(state, strict=True)
  face_model.eval()

  # for mode in mode_lst:
  dataset = BTSTarDataset(cfg, mode)
  loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=6)
  
  if mode in ["Gallery1", "Gallery2"]:
    extract_gallery_template(loader, mode)
  else:
    extract_probe_template(loader, mode)
    
     