"""
@author: Ahmad Kurdi
Data loaders for Virtual Trainer portfolio project.
DSR portfolio project with Artur Silicki
"""
#import pdb
import re

import pandas as pd 
import numpy as np

from VideoPose3D.common.utils import deterministic_random
from itertools import zip_longest


def fetch_vp3d_keypoints(keypoint_file, subjects, action_list, class_val, subset=1):
    # fetch 2D keypoints from npz file

    keypoints = np.load(keypoint_file)['positions_2d'].item() # load keypoints
    actions = []
    out_poses_2d = []
    return_idx = []

    # traverse dataset and append return lists
    for subject in subjects:
        for action in keypoints[subject].keys():
            action_clean = re.sub(r'\s\d+$', '', action)
            if action_clean in action_list: #skip actions not in action_list
                poses_2d = keypoints[subject][action]
                for i in range(len(poses_2d)): # Iterate across cameras
                    out_poses_2d.append(poses_2d[i][::2]) #stride 2 to downsample framerate
                    actions.append(class_val)
                    return_idx.append(f"{subject}_{action}")

    # sample a subset if requested
    idx_final = []
    actions_final = []
    out_poses_2d_final = []
    max_frames = 180
    min_frames = 60
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_subclips = out_poses_2d[i].shape[0]//max_frames
            if out_poses_2d[i].shape[0]-max_frames*n_subclips>min_frames:
                n_subclips+=1
            for ns in range(n_subclips):
                out_poses_2d_final.append(out_poses_2d[i][max_frames*ns:max_frames*(ns+1),:,:])
                actions_final.append(actions[i])
                idx_final.append(return_idx[i])

    return actions_final, out_poses_2d_final, idx_final

def fetch_openpose_keypoints(keypoint_file):
    return_idx= []
    keypoints = pd.read_csv(keypoint_file)
    actions = []
    out_poses_2d = []
    vid_idx = keypoints['vid_nr'].unique()
    joint_order = ['16_0','16_1','11_0','11_1','12_0','12_1','13_0','13_1','8_0','8_1','9_0','9_1',
                   '10_0','10_1','14_0','14_1','1_0','1_1','15_0','15_1','0_0','0_1','2_0','2_1',
                   '4_0','4_1','3_0','3_1','5_0','5_1','7_0','7_1','6_0','6_1']
    
    for idx in vid_idx:
        action = int(keypoints.loc[keypoints['vid_nr']==idx]['target'].head(1))
        poses_2d = keypoints.loc[keypoints['vid_nr']==idx].sort_values(by="clip_id")[joint_order]
        poses_2d = np.reshape(poses_2d.values,(-1,17,2))
        if ~np.isnan(poses_2d).any():
            out_poses_2d.append(poses_2d)
            actions.append(action)
            return_idx.append(idx)

    return actions, out_poses_2d, return_idx

def fetch_keypoints(keypoints):
    actions = []
    out_poses_2d = []
    vid_idx = keypoints['vid_nr'].unique()
    joint_order = ['16_0','16_1','11_0','11_1','12_0','12_1','13_0','13_1','8_0','8_1','9_0','9_1',
                   '10_0','10_1','14_0','14_1','1_0','1_1','15_0','15_1','0_0','0_1','2_0','2_1',
                   '4_0','4_1','3_0','3_1','5_0','5_1','7_0','7_1','6_0','6_1']
    
    for idx in vid_idx:
        action = int(keypoints.loc[keypoints['vid_nr']==idx]['target'].head(1))
        poses_2d = keypoints.loc[keypoints['vid_nr']==idx].sort_values(by="clip_id")[joint_order]
        poses_2d = np.reshape(poses_2d.values,(-1,17,2))
        if ~np.isnan(poses_2d).any():
            out_poses_2d.append(poses_2d)
            actions.append(action)

    return actions, out_poses_2d

def balance_dataset(targets,seed):
    np.random.seed(seed)
    classes, counts = np.unique(targets,return_counts=True)
    sm_class = classes[counts.argmin()]
    smpl_size = counts.min()
    idx = np.where(targets == sm_class)[0]
    for cl in classes:
        if cl == sm_class: 
            continue
        ix_ = np.random.choice(np.where(targets == cl)[0],smpl_size,False)
        idx = np.concatenate((idx,ix_))
    return idx

def balance_dataset_recipe2(targets,seed):
    np.random.seed(seed)
    classes, counts = np.unique(targets,return_counts=True)
    sm_class = classes[counts.argmin()]
    smpl_size = counts.min()
    smpl_size = 300
    idx_all = np.array(range(len(targets)))
    idx = np.where(targets == sm_class)[0]
    for cl in range(len(classes)):
        if classes[cl] == sm_class: 
            continue
        elif counts[cl]<=300:
            ix_ = np.random.choice(np.where(targets == classes[cl])[0],counts[cl],False)
        else:
            ix_ = np.random.choice(np.where(targets == classes[cl])[0],smpl_size,False)
        idx = np.concatenate((idx,ix_))
    idx_test = np.array([x for x in idx_all if x not in list(idx)])
    return idx, idx_test
