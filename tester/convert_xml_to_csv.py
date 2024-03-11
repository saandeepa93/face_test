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




def parseBriarSigset(filename):
    """
The parseBriarSigset function parses a Briar XML sigset file and returns a pandas dataframe with the following columns:
    name - The name of the signature.
    subjectId - The ID of the subject who signed this signature.
    filepath - The path to where this sigmember is stored on disk. This may be relative or absolute, depending on how it was specified in the sigset file.
    modality - What type of media is used for this sigmember (e.g., 'online', 'offline').  See https://briarsigset-schema-v2_

:param filename: Specify the path to the xml file
:return: A dataframe with the following columns:
:doc-author: Joel Brogan, BRIAR team, Joel Brogan, BRIAR team, Trelent
"""
    import xml.etree.cElementTree as ET
    import pandas as pd

    tree = ET.parse(filename)

    root = tree.getroot()

    columns = ('entryId', 'subjectId', 'filepath', 'modality', 'media', 'media_format', 'start', 'stop', 'unit')
    # df = pd.DataFrame(columns=columns)
    data_list = []
    rootiter = list(root.iter('{http://www.nist.gov/briar/xml/sigset}signature'))


    for signature in rootiter:
        name = signature.find('{http://www.nist.gov/briar/xml/sigset}name').text
        subjectIds = []
        for element in signature.iter('{http://www.nist.gov/briar/xml/sigset}subjectId'):
            subjectIds.append(element.text)

        sig = list(signature.iter('{http://www.nist.gov/briar/xml/sigset}sigmember'))
        for i, sigmember in enumerate(sig):
            filepath = "ERROR"
            modality = 'ERROR'
            media = "ERROR"
            media_format = 'ERROR'
            start = "NA"
            stop = 'NA'
            unit = 'NA'

            for element in sigmember.iter('{http://www.nist.gov/briar/xml/sigset}filePath'):
                filepath = element.text
            for element in sigmember.iter('{http://www.nist.gov/briar/xml/sigset}modality'):
                modality = element.text
            for element in sigmember.iter('{http://www.nist.gov/briar/xml/sigset}media'):
                media = element.text
            for element in sigmember.iter('{http://www.nist.gov/briar/xml/sigset}mediaFormat'):
                media_format = element.text
            for element in sigmember.iter('{http://www.nist.gov/briar/xml/sigset-eval}start'):
                start = element.text
            for element in sigmember.iter('{http://www.nist.gov/briar/xml/sigset-eval}stop'):
                stop = element.text
            for element in sigmember.iter('{http://www.nist.gov/briar/xml/sigset-eval}unit'):
                unit = element.text
            for element in sigmember.iter('{http://www.nist.gov/briar/xml/sigset}id'):
                entryId = element.text
            # i = len(df)

            data = [entryId, subjectIds, filepath, modality, media, media_format, start, stop, unit]
            # print(columns)
            i = 0
            for field, value in zip(columns, data):
                if value == "ERROR":
                    raise ValueError("Could not determine '{}' for sigmember {} in {}".format(field, i, filename))
            i += 1
            data_list.append(data)

    df = pd.DataFrame(data_list, columns=columns)
    return df


def convert_to_csv(sigset_path, csv_path, verbose=True):
  print("Reading sigset: {}".format(sigset_path))
  df = parseBriarSigset(sigset_path)
  print('DONE PARSING!')

  unique_names = len(set(df['entryId']))
  media_count = len(df)
  image_count = len(df[df['media'] == 'digitalStill'])
  video_count = len(df[df['media'] == 'digitalVideo'])

  videos = df[df['media'] == 'digitalVideo']

  unfiltered_videos = len(videos[videos['unit'] == 'NA'])

  frame_videos = videos[videos['unit'] == 'frame']
  video_frames = frame_videos['stop'].apply(int).sum() - frame_videos['start'].apply(int).sum()

  time_videos = videos[videos['unit'] == 'second']
  video_time = time_videos['stop'].apply(float).sum() - time_videos['start'].apply(float).sum()

  if verbose:
      print('Sigset Fields:')
      print(list(df.columns))
  print("============= Stats =============")
  print('Unique Names:', unique_names)
  print('Media Count:', media_count)
  print('Image Count:', image_count)
  print('Video Count:', video_count)
  print("=================================")
  print('Unfiltered Videos:', unfiltered_videos)
  print('Selected Video Frames:', video_time)
  print('Selected Video Seconds:', video_frames)
  print("=================================")
  print()

  if csv_path is not None:
      print("Saving {} items to: {}".format(len(df), csv_path))

      df.to_csv(csv_path, index=False)



if __name__ == "__main__":
#   sigset_path1 = "./data/analysis/sigsets_gallery/Gallery1.xml"
#   sigset_path2 = "./data/analysis/sigsets_gallery/Gallery2.xml"

#   csv_path1 = "./data/analysis/Gallery1.csv"
#   csv_path2 = "./data/analysis/Gallery2.csv"

#   # convert_to_csv(sigset_path1, csv_path1)
#   convert_to_csv(sigset_path2, csv_path2)
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
  
    


