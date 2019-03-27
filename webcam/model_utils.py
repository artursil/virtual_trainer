import numpy as np
import cv2
from collections import OrderedDict
import torch
from torch import nn
import re
from openpose.mpii_config import *
import os
from models.semantic import TemporalModel, ModdedTemporalModel, ModdedStridedModel, StandardiseKeypoints, HeadlessNet2, SplitModel4,NaiveBaselineModel,HeadlessModule

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
def load_model_weights(chk_filename,base,top, class_mod):
    
    pretrained_weights = torch.load(chk_filename, map_location=lambda storage, loc: storage)['model_state_dict']
    
    base_weights = OrderedDict([(re.sub(r"(base_model.)","",key), value) for key, value in pretrained_weights.items() if key.startswith("base_model")])
    top_weights = OrderedDict([(re.sub(r"(top_model.)","",key), value) for key, value in pretrained_weights.items() if key.startswith("top_model")])
    class_weights = OrderedDict([(re.sub(r"(shrink.)","",key), value) for key, value in top_weights.items() if key.startswith("shrink")])

    drop_keys = [f"shrink.{k}" for k in class_weights.keys()]
    for k in drop_keys:
        top_weights.pop(k)
    base.load_state_dict(base_weights, strict=False)
    top.load_state_dict(top_weights, strict=False)
    class_mod.load_state_dict(class_weights)
    return base, top, class_mod

    
def build_model(chk_filename, in_joints, in_dims, out_joints, filter_widths, causal, channels, embedding_len,classes):

    base= TemporalModel(in_joints,in_dims,out_joints,filter_widths,causal=True,dropout=0.25,channels=channels)
    # top= ModdedStridedModel(in_joints, 3, out_joints, filter_widths, causal=True, dropout=0.25, channels=embedding_len, skip_res=False)
    top= ModdedTemporalModel(in_joints, 3, out_joints, filter_widths, causal=True, dropout=0.25, channels=embedding_len, skip_res=False)
    class_mod = nn.Conv1d( embedding_len, classes, 1)

    base, top, class_mod = load_model_weights(chk_filename,base,top,class_mod)
    model = nn.Sequential(OrderedDict([
          ('base', base) ,
          ('transform', StandardiseKeypoints(True,False)),
          ('embedding', HeadlessNet2(top)),
          ('classifier', SplitModel4(class_mod) )
        ]))
    return model

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
def build_model2(chk_filename,datapoint, in_joints, in_dims, out_joints, filter_widths, causal, channels, embedding_len,classes):

    chk_filename_base = os.path.join(datapoint,'BaseModels', 'epoch_45.bin')
    pretrained_weights = torch.load(chk_filename_base, map_location=lambda storage, loc: storage)

    top = NaiveBaselineModel(in_joints, in_dims, out_joints, filter_widths, pretrained_weights, embedding_len, classes,
                                causal=True, dropout=0.25, channels=channels)
    checkp = torch.load(chk_filename)
    top.load_state_dict(checkp['model_state_dict'])
    class_mod = nn.Conv1d( embedding_len, classes, 1)

    class_mod = load_model_weights2(chk_filename,class_mod)
    model = nn.Sequential(OrderedDict([
          ('embedding', HeadlessNet2(top)),
          ('classifier', SplitModel4(class_mod) )
        ]))
    return model

def load_model_weights2(chk_filename,class_mod):
    
    pretrained_weights = torch.load(chk_filename, map_location=lambda storage, loc: storage)['model_state_dict']
    top_weights = OrderedDict([(re.sub(r"(top_model.)","",key), value) for key, value in pretrained_weights.items() if key.startswith("top_model")])
    class_weights = OrderedDict([(re.sub(r"(shrink.)","",key), value) for key, value in top_weights.items() if key.startswith("shrink")])

    drop_keys = [f"shrink.{k}" for k in class_weights.keys()]
    for k in drop_keys:
        top_weights.pop(k)
    class_mod.load_state_dict(class_weights)
    return class_mod