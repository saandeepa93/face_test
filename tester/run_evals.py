# imports
import os
import time
import pandas as pd
from pandas.plotting import table
import numpy as np
import scipy as sp
import skimage
import sklearn
import sys
sys.path.append('.')

import matplotlib.pyplot as plt
import seaborn as sns
import json
#from sklearn.metrics import det_curve
from tabulate import tabulate
import sys


import csv
import matplotlib.pyplot as plt
from imports import * 
import os 
import scipy.stats as stats
from scipy import interpolate
import scipy.optimize as opt
from sklearn.metrics import accuracy_score, f1_score, roc_curve,roc_auc_score

import csv
import pickle as pkl

# from briar_analysis.VerificationResults import VerificationResults, plotVerificationMetrics, plotVerificationMetricsFaceRestricted
# from briar_analysis.SearchResults import SearchResults, plotOpenSearchMetrics, plotClosedSearchMetrics
# from briar_analysis.CovariateTools import populateMoreFields, multiHistogram
# from briar_analysis.Algorithm import Algorithm
# import briar_analysis as ba
# from VerificationResults import VerificationResults

from Algorithm import Algorithm

from constants import *





def plot_roc_basic(score_matrix, mask_matrix, fname, cname, rndm=False):

  scores = score_matrix.flatten().copy()
  mask = mask_matrix.flatten()
  ns_probs = [0 for _ in range(len(mask))]
  
  # calculate scores
  ns_auc = roc_auc_score(mask, ns_probs)
  lr_auc = roc_auc_score(mask, scores)
  # summarize scores
  print('No Skill: ROC AUC=%.3f' % (ns_auc))
  print('Logistic: ROC AUC=%.3f' % (lr_auc))
  # calculate roc curves
  ns_fpr, ns_tpr, _ = roc_curve(mask, ns_probs)
  lr_fpr, lr_tpr, _ = roc_curve(mask, scores)


def computeROC(masks, scores):
  scores = scores.flatten().copy()
  labels = masks.flatten()
  fpr, tpr, thresholds = roc_curve(labels, scores)
  fpr_levels = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
  f_interp = interpolate.interp1d(fpr, tpr)
  tpr_ar_fpr = [f_interp(x) for x in fpr_levels]
  for (far, tar) in zip (fpr_levels, tpr_ar_fpr):
    ic(f"TAR @ FAR={far}: {tar}")


def plot_roc_basic(score_matrix, mask_matrix, cname, rndm=False):

  scores = score_matrix.flatten().copy()
  mask = mask_matrix.flatten()
  ns_probs = [0 for _ in range(len(mask))]
  
  # calculate scores
  ns_auc = roc_auc_score(mask, ns_probs)
  lr_auc = roc_auc_score(mask, scores)
  # summarize scores
  print('No Skill: ROC AUC=%.3f' % (ns_auc))
  print('Logistic: ROC AUC=%.3f' % (lr_auc))
  # calculate roc curves
  ns_fpr, ns_tpr, _ = roc_curve(mask, ns_probs)
  lr_fpr, lr_tpr, _ = roc_curve(mask, scores)



  # # ax = plt.subplot(1,2,1)
  plt.title("Receiver Operating Characteristic: Face Included")
  # plt.xlim(1e-5,1.0)
  # plt.ylim(-0.05,1.05)
  plt.xlabel('False Accept Rate')
  plt.ylabel('True Accept Rate')
  # plt.xscale('log')
  linestyle=(0, (7, 10))
  # plot the roc curve for the model
  if rndm:
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='Random', linewidth=2)
  plt.plot(lr_fpr, lr_tpr, marker='.', label=cname, linestyle=linestyle, linewidth=2)
  # axis labels
  # plt.xlabel('False Acceptance Rate')
  # plt.ylabel('True Acceptance Rate')
  # show the legend
  plt.legend()
  # show the plot
  plt.savefig(f"./data/results/rest_{cname}.png")
  # plt.close()

if __name__ == "__main__":

  SOURCE_DIR = "./data/analysis/eval_4.0.0"
  NAN = float('nan')
  FACE_COLOR = 'blue'
  alg_label = "test"
  configs = "briar_9"

  SCORE_DIR = f"./data/score_files/{configs}"

  all_files = glob.glob(os.path.join(SCORE_DIR, "*.npy"))

  gmodes = ["Gallery1"]
  # pmodes = ['face_incl_ctrl','face_incl_trt']
  pmodes = ['face_incl_ctrl']

  for gmode in gmodes:
    for pmode in pmodes:

      score_matrix = np.load(os.path.join(SCORE_DIR,f'face_score_{gmode}_{pmode}.pkl.npy'))
      mask_matrix = np.load(os.path.join(SCORE_DIR,f'face_mask_{gmode}_{pmode}.pkl.npy'))
      plot_roc_basic(score_matrix, mask_matrix, pmode)
      ic(f"MODE: {pmode}, {gmode}")
      computeROC(mask_matrix, score_matrix)
