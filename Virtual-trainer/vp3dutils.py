import numpy as np
import re
import os
import sys
import errno
import torch
import torch.nn as nn


from models.semantic import *
from time import time
from VideoPose3D.common.utils import deterministic_random


def fetch_keypoints(keypoint_file, subjects, action_list, subset=1):
    keypoints = np.load(keypoint_file)['positions_2d'].item() # load keypoints
    action_ident = np.eye(len(action_list)) #identity matrix to generate one hot
    
    actions = []
    out_poses_2d = []

    # traverse dataset and append return lists
    for subject in subjects:
        for action in keypoints[subject].keys(): 
            action_clean = re.sub(r'\s\d+$', '', action)
            if action_clean in action_list: #skip actions not in action_list
                one_hot = action_ident[action_list.index(action_clean)]    
                poses_2d = keypoints[subject][action]
                for i in range(len(poses_2d)): # Iterate across cameras
                    out_poses_2d.append(poses_2d[i])
                    action_oh = np.tile(one_hot,(poses_2d[i].shape[0],1))
                    actions.append(action_oh)
    
    # sample a subset if requested
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i]) * subset))
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start+n_frames]
            actions = actions[i][start:start+n_frames]
    
    return actions, out_poses_2d


def load_VP3D_model(chk_filename, in_joints, in_dims, out_joints, filter_widths, channels):
    
    # instantiate models
    model_pos = ModdedStridedModel(in_joints, in_dims, out_joints,
                                filter_widths=filter_widths, causal=True, dropout=0.25, channels=channels)
    eval_model = ModdedTemporalModel(in_joints, in_dims, out_joints,
                                filter_widths=filter_widths, causal=True, dropout=0.25, channels=channels,
                                dense=False)
    # load model weights from checkpoint
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    model_pos.load_state_dict(checkpoint['model_pos'])
    
    return model_pos, eval_model

def modify_model(model, embedding_len, classes):
    # modify final block and output layer
    dilation = model.layers_conv[-2].dilation
    channels =  model.layers_conv[-2].in_channels
    stride = model.layers_conv[-2].stride
    kernel = model.layers_conv[-2].kernel_size
    model.layers_conv[-2] = nn.Conv1d(channels,embedding_len,kernel, dilation=dilation, stride=stride, bias=False)
    model.layers_conv[-1] = nn.Conv1d(embedding_len,embedding_len,1, dilation=1, bias=False)
    model.layers_bn[-2] = nn.BatchNorm1d(embedding_len, momentum=0.1)
    model.layers_bn[-1] = nn.BatchNorm1d(embedding_len, momentum=0.1)
    model.shrink = nn.Linear(embedding_len,classes)
    return model


