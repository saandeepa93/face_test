
# Update these constants to reflect your paths, filenames, etc.

# Evaluation info
PHASE = 1
PROTOCOL = 'briar_evaluation_v3.1.0'
SIGSET_SUBSET = 'main'
GALLERY_TYPE = 'full'

# Path to directory containing result files
# see README for a list of files that this directory should include for this notebook to run as expected
ALG_DIR = '/shares/rra_sarkar-2135-1003-00/faces/face_verification/data/score_files/final_eval'

# Algorithm/performer shortname
ALG_TAG = ''

# Gallery filenames; it is assumed in Algorithm.py that these will be in ALG_DIR
GALLERY_G1 = '/shares/rra_sarkar-2135-1003-00/faces/face_verification/data/analysis/eval_4.0.0/Gallery1.csv'
GALLERY_G2 = '/shares/rra_sarkar-2135-1003-00/faces/face_verification/data/analysis/eval_4.0.0/Gallery2.csv'

# Path to probe csv file corresponding to probe sigset used
PROBES = '/shares/rra_sarkar-2135-1003-00/faces/face_verification/data/analysis/eval_4.0.0/Probe_BTS_briar-rd_FaceRestricted_ctrl.csv'

# Destination directory for generated plots and curve csv files 
EVAL_DIR = ''

# Program metrics
VER_FACE_INCL = 70
VER_WB_INCL = 85
VER_WB_RESTR = 50
RR_FACE = 80
RR_WB = 90
OS_FACE = 70
OS_WB = 50

# Set this to True if you plan to generate a PDF from the outputs of the current notebook run
# If not, set it to False so that the tables are rendered in a more readable format
IS_NBCONVERT = True

# Set to True if you plan to use plots in a powerpoint and need the font to be larger
POWERPOINT_MODE = False

# Figure size constants
WIDE_FIGURE = (16,8)
EXTRA_WIDE_FIGURE = (24,8)
LARGE_FIGURE = (16,16)
MED_FIGURE = (12,12)
SMALL_FIGURE = (8,8)

# NaN definition
NAN = float('nan')

# Evaluation environment 
GPU_HARDWARE = ''
NUM_GPUS = 0
OS = ''
PODMAN_VERSION = ''
NVIDIA_CONTAINER_RUNTIME_VERSION = ''
NVIDIA_DRIVER_VERSION = ''
CUDA_VERSION = ''

# Thresholds to sample for DET curves
DET_THRESH_COUNT = 10000

# Set to True if you want info to print out that may be useful given common issues that we have encountered
DEBUG = False

# Set to True to generate search results from verification results 
USE_VERIF = False
