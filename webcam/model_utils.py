import os
import numpy as np
import cv2
from collections import OrderedDict
import torch
from torch import nn
import re
from openpose.mpii_config import *
import os
from models.semantic import HeadlessModule


import sys
sys.path.insert(0, "../../virtual_trainer")

import torch
from openpose.model import get_model

def load_all_models():
    CHECKPATH = 'Virtual_trainer/checkpoint'

    # Data mountpoint
    DATAPOINT = "Virtual_trainer/Data"

    # --- Datasets ---
    # H36M Ground truths
    h36m_file = os.path.join(DATAPOINT,'Keypoints','data_2d_h36m_gt.npz')



    # --- Parameters ---
    batch_size = 2048
    epochs = 20
    embedding_len = 128
    lr, lr_decay = 0.001 , 0.95 
    split_ratio = 0.2

    # --- H36M pretrained model settings ---
    # checkpoint file
    chk_filename = os.path.join(DATAPOINT,'BaseModels', 'epoch_45.bin')
    # model architecture
    filter_widths = [3,3,3]
    channels = 1024
    in_joints, in_dims, out_joints = 17, 2, 17
    weight_name = '../../virtual_trainer/openpose/weights/openpose_mpii_best.pth.tar'
    model = get_model('vgg19')     
    model.load_state_dict(torch.load(weight_name)['state_dict'])
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.float()
    return model

def picture_keypoints(personwiseKeypoints,keypoints_list,frameClone,after_shape):
    mult = frameClone.shape[0]/after_shape[0]
    for i in range(14):
        for n in range(len(personwiseKeypoints)):
            index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
            if -1 in index:
                continue
            B = np.int32(keypoints_list[index.astype(int), 0])
            A = np.int32(keypoints_list[index.astype(int), 1])
            # print(B[0])
            # print(A[0])
            cv2.line(frameClone, (int(B[0]*mult)+10, int(A[0]*mult)),
                                (int(B[1]*mult)+10, int(A[1]*mult)),
                                [102,255,102], 3, cv2.LINE_AA)        
    return frameClone

class HeadlessNet2(nn.Module):
    """
    Headless network
    """
    def __init__(self, class_model):
        super().__init__()
        class_model.top_model.shrink = HeadlessModule()
        self.embed_model = class_model
    def forward(self,x):
        x = self.embed_model(x)
        return x