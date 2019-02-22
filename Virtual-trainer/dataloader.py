"""
@author: Ahmad Kurdi
Data loaders for Virtual Trainer portfolio project.
DSR portfolio project with Artur Silicki
"""
import re

import pandas as pd 
import numpy as np

from VideoPose3D.common.utils import deterministic_random
from itertools import zip_longest

class SequenceGenerator:
    """
    sequence generator for training
    """
    
    def __init__(self, batch_size, actions, poses, pad=0, causal_shift=0, split_ratio=0.2, 
                                    shuffle=True, random_seed=1234, endless=False):
        assert len(actions) == len(poses)
        self.pad = pad
        self.causal_shift = causal_shift
        self.poses = poses
        self.actions = actions
        self.batch_size = batch_size

        self.shuffle = shuffle
        self.endless = endless
        self.state = None
        self.split_ratio = split_ratio
        self.random = np.random.RandomState(random_seed)
        self.trn_idx , self.vld_idx = self.split_dataset()
        self.batch_lineage()

    def next_epoch(self):
        self.trn_idx , self.vld_idx = self.split_dataset()
        self.batch_lineage()

    def split_dataset(self):
        targets = np.array(self.actions)
        classes = np.unique(targets)
        num_samples = self.split_ratio * len(targets) // len(classes)
        vld_idx = []
        for cl in classes:
            vld_idx.append(np.random.choice(np.where(targets == cl),num_samples,False))
        trn_idx = np.setdiff1d(range(len(targets)),vld_idx)
        return trn_idx , vld_idx

    def batch_lineage(self):
        pairs = [] # (seq_idx, start_frame, end_frame) tuples
        poses = self.poses[self.trn_idx]
        actions = self.actions[self.trn_idx]
        for i in range(len(poses)):
            n_chunks = poses[i].shape[0]
            bounds = np.arange(n_chunks+1)
            pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:])

        self.batch_act = np.empty((self.batch_size, 1, actions[0].shape[-1]))
        self.batch_poses = np.empty((self.batch_size, 1+2*self.pad, poses[0].shape[-2], poses[0].shape[-1]))     
        self.num_batches = (len(pairs) + self.batch_size - 1) // self.batch_size
        self.pairs = pairs

    def random_state(self):
        return self.random
    
    def set_random_state(self, random):
        self.random = random
         
    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state

    def num_frames(self):
        trn_count = self.num_batches * self.batch_size
        vld_count = 0
        for p in self.poses[self.vld_idx]:
            vld_count += p.shape[0]
        return trn_count , vld_count

    def next_validation(self):
        vld_actions = self.actions[self.vld_idx]
        vld_poses = self.poses[self.vld_idx]
        for seq_act, seq_poses in zip_longest(vld_actions, vld_poses):
            batch_act = np.expand_dims(seq_act, axis=0)
            batch_poses = np.expand_dims(np.pad(seq_poses,
                            ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)),
                            'edge'), axis=0)
            
            yield batch_act, batch_poses

    def next_batch(self):
        enabled = True
        while enabled:
            start_idx, pairs = self.next_pairs()
            for b_i in range(start_idx, self.num_batches):
                chunks = pairs[b_i*self.batch_size : (b_i+1)*self.batch_size]
                for i, (seq_i, start_act, end_act) in enumerate(chunks):
                    start_poses = start_act - self.pad - self.causal_shift
                    end_poses = end_act + self.pad - self.causal_shift

                    #poses
                    seq_poses = self.poses[seq_i]
                    low_poses = max(start_poses, 0)
                    high_poses = min(end_poses, seq_poses.shape[0])
                    pad_left_poses = low_poses - start_poses
                    pad_right_poses = end_poses - high_poses
                    if pad_left_poses != 0 or pad_right_poses != 0:
                        self.batch_poses[i] = np.pad(seq_poses[low_poses:high_poses], ((pad_left_poses, pad_right_poses), (0, 0), (0, 0)), 'edge')
                    else:
                        self.batch_poses[i] = seq_poses[low_poses:high_poses]
                    
                    # actions                   
                    seq_act = self.actions[seq_i]
                    low_act = max(start_act, 0)
                    high_act = min(end_act, seq_act.shape[0])
                    pad_left_act = low_act - start_act
                    pad_right_act = end_act - high_act
                    if pad_left_act != 0 or pad_right_act != 0:
                        self.batch_act[i] = np.pad(seq_act[low_act:high_act], ((pad_left_act, pad_right_act), (0, 0)), 'edge')
                    else:
                        self.batch_act[i] = seq_act[low_act:high_act]
  
                if self.endless:
                    self.state = (b_i + 1, pairs)

                yield np.squeeze(self.batch_act[:len(chunks)]), self.batch_poses[:len(chunks)]
            
            if self.endless:
                self.state = None
            else:
                enabled = False


def fetch_vp3d_keypoints(keypoint_file, subjects, action_list, class_val, subset=1):
    # fetch 2D keypoints from npz file

    keypoints = np.load(keypoint_file)['positions_2d'].item() # load keypoints
    actions = []
    out_poses_2d = []

    # traverse dataset and append return lists
    for subject in subjects:
        for action in keypoints[subject].keys(): 
            action_clean = re.sub(r'\s\d+$', '', action)
            if action_clean in action_list: #skip actions not in action_list    
                poses_2d = keypoints[subject][action]
                for i in range(len(poses_2d)): # Iterate across cameras
                    out_poses_2d.append(poses_2d[i][::2]) #stride 2 to downsample framerate
                    actions.append(class_val)
    
    # sample a subset if requested
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i]) * subset))
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start+n_frames]
            actions = actions[i][start:start+n_frames]
    
    return actions, out_poses_2d

def fetch_openpose_keypoints(keypoint_file):

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

    return actions, out_poses_2d

def balance_dataset(targets):
    classes, counts = np.unique(targets,return_counts=True)
    sm_class = classes[counts.argmin()]
    smpl_size = counts.min()
    idx = np.where(targets == sm_class)
    for cl in classes:
        if cl == sm_class: 
            continue
        ix_ = np.random.choice(np.where(targets == cl),smpl_size,False)
        idx = np.concatenate((idx,ix_))
    return idx

