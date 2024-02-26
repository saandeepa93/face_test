import sys
sys.path.append('.')

from briar_analysis.VerificationResults import VerificationResults, plotVerificationMetrics, plotVerificationMetricsFaceRestricted
from briar_analysis.SearchResults import SearchResults, plotOpenSearchMetrics, plotClosedSearchMetrics
from briar_analysis.CovariateTools import populateMoreFields, multiHistogram
from briar_analysis.Algorithm import Algorithm
import briar_analysis as ba

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

from utils import seed_everything, mkdir, get_args
from configs import get_cfg_defaults
from imports import * 
from models import FaceFeatureModel



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

  test_mode = "incl_ctrl"

  # LOAD CONFIGURATION
  cfg = get_cfg_defaults()
  cfg.merge_from_file(config_path)
  cfg.freeze()

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

  if test_mode == "incl_ctrl":
    df = pd.read_csv("./data/analysis/Probe_BTS_briar-rd_FaceIncluded_ctrl.csv")
    ic(df.head())
  e()




