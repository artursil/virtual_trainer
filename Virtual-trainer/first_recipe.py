"""
@author: Ahmad Kurdi
Test recipe for action classifier model
DSR portfolio project with Artur Silicki
"""

import os
import sys
import errno
import vp3dutils
import torch

import torch.nn as nn
import torch.optim as optim
import numpy as np

from models.semantic import NaiveBaselineModel, NaiveStridedModel
from dataloader import *

try:
    # Create checkpoint directory if it does not exist
    os.makedirs('checkpoint')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', 'checkpoint')
CHECKPATH = os.path('checkpoint')

# Data mountpoint
DATAPOINT = "Data"

# --- Datasets ---
# H36M Ground truths
h36m_file = os.path.join(DATAPOINT,'Keypoints','data_2d_h36m_gt.npz')
action_list=['Walking','Waiting','SittingDown']
subjects = ['S1','S5','S6','S7','S8']
# Instagram Openpose estimations
instagram_file = os.path.join(DATAPOINT,'Keypoints','keypoints.csv')

# --- Parameters ---
batch_size = 1024
epochs = 20
embedding_len = 128
lr, lr_decay = 0.001 , 0.95 

# --- H36M pretrained model settings ---
# checkpoint file
chk_filename = os.path.join(DATAPOINT,'BaseModels', 'epoch_45.bin')
# model architecture
filter_widths = [3,3,3]
channels = 1024
in_joints, in_dims, out_joints = 17, 2, 17

# load dataset
action_op, poses_op = fetch_openpose_keypoints(instagram_file)
action_vp3d, poses_vp3d = fetch_vp3d_keypoints(h36m_file,subjects,action_list,len(np.unique(action_op)),subset=0.5)
actions = action_op + action_vp3d
poses = poses_op + poses_vp3d

# balance the dataset
balanced = balance_dataset(np.array(actions))
actions, poses = actions[balanced] , poses[balanced]



