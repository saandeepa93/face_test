from yacs.config import CfgNode as CN

_C = CN()

# PATHS
_C.PATHS = CN()
_C.PATHS.ROOT = "/dataset/DISFA/Videos_LeftCamera"
_C.PATHS.ANNOT = "/dataset/BRIAR/KITBRC1/data/full"
_C.PATHS.FACE = "/dataset/BRIAR/KITBRC1/data/full"
_C.PATHS.VIS_PATH = "./data/loader"
_C.PATHS.TEMPLATE = ""
_C.PATHS.SO_ROOT = ""
_C.PATHS.OF_SAVE = ""

_C.DATASET = CN()
_C.DATASET.IMG_SIZE=224
_C.DATASET.CLIP_LGT = 4
_C.DATASET.NUM_WORKERS = 4

_C.TRAINING = CN()
_C.TRAINING.ITER=100
_C.TRAINING.LR=1e-4
_C.TRAINING.WT_DECAY=1e-5
_C.TRAINING.BATCH_SIZE=256
_C.TRAINING.DISTRIBUTED=False
_C.TRAINING.FEATS=0
_C.TRAINING.MODEL="ELASTIC"

_C.TEST = CN()
_C.TEST.FOLD = 1

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()