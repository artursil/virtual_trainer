"""
@author: Ahmad Kurdi
Test recipe for action classifier model
DSR portfolio project with Artur Silicki
"""


import os
import sys
import errno
import torch

import torch.nn as nn
import torch.optim as optim
import numpy as np

from models.semantic import NaiveBaselineModel
from dataloader import *
from simple_generators import SimpleSequenceGenerator
from itertools import compress

try:
    # Create checkpoint directory if it does not exist
    os.makedirs('checkpoint')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', 'checkpoint')
CHECKPATH = 'checkpoint'

# Data mountpoint
DATAPOINT = "Data"

# --- Datasets ---
# H36M Ground truths
h36m_file = os.path.join(DATAPOINT,'Keypoints','data_2d_h36m_gt.npz')
action_list=['Walking','Waiting','SittingDown']
subjects = ['S1','S5','S6','S7','S8']
# Instagram Openpose estimations
instagram_file = os.path.join(DATAPOINT,'Keypoints','keypoints.csv')
ucf_file = os.path.join(DATAPOINT,'Keypoints','keypoints_rest.csv')
ucf_file_test = os.path.join(DATAPOINT,'Keypoints','keypoints_rest_test.csv')
insta_test_file = os.path.join(DATAPOINT,'Keypoints','keypoints_insta_test2.csv')
# --- Parameters ---
batch_size = 1024
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

# load dataset
action_op, poses_op, vid_idx = fetch_openpose_keypoints(instagram_file)
action_op2, poses_op2, vid_idx2 = fetch_openpose_keypoints(ucf_file)
action_op_test, poses_op_test, vid_idx_test = fetch_openpose_keypoints(ucf_file_test)
actions_op_itest, poses_op_itest, vid_idx_itest = fetch_openpose_keypoints(insta_test_file)


def balance_dataset_recipe2(targets,seed):
    np.random.seed(seed)
    classes, counts = np.unique(targets,return_counts=True)
    sm_class = classes[counts.argmin()]
    smpl_size = counts.min()
    smpl_size = 100
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



# --- Parameters ---
batch_size = 1024
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

action_op_4 = list(compress(action_op2,np.array(action_op2)==4))[31:45]
poses_op_4 = list(compress(poses_op2,np.array(action_op2)==4))[31:45]
action_op_4 += list(compress(action_op2,np.array(action_op2)==7))[:10]
poses_op_4 += list(compress(poses_op2,np.array(action_op2)==7))[:10]

actions = action_op + action_op_test + actions_op_itest + action_op_4
actions = [action if action!=8 else 6 for action in actions]
poses = poses_op + poses_op_test + poses_op_itest + poses_op_4
# vid_idx = vid_idx + vid_idx2

# 4th class


# seed = 1234
test_ids,balanced = balance_dataset_recipe2(np.array(actions),seed)
actions_test = [actions[x] for x in test_ids]
actions_test = [action if action<=7 else 6 for action in actions_test]
poses_test = [poses[x] for x in test_ids]

print(actions_test)


# build models
classes = 8
pretrained_weights = torch.load(chk_filename, map_location=lambda storage, loc: storage)
eval_model = NaiveBaselineModel(in_joints, in_dims, out_joints, filter_widths, pretrained_weights, embedding_len, classes,
                                causal=True, dropout=0.25, channels=channels)


loss_fun = nn.CrossEntropyLoss()


if torch.cuda.is_available():
    eval_model = eval_model.cuda()

# train model
epoch = 0
losses_train = []
losses_test = []
validation_targets = []
vp3d_keypoints = []

# checkp = torch.load('/home/artursil/Documents/vt2/recipe1/checkpoint/model4-19.pth')
checkp = torch.load('/home/artursil/Documents/virtual_trainer/Virtual_trainer/checkpoint/Recipe-2-epoch-19.pth')
# checkp = torch.load('/home/artursil/Documents/virtual_trainer/Virtual_trainer/checkpoint/model-6.pth')
eval_model.load_state_dict(checkp['model_state_dict'])



# generator = SimpleSequenceGenerator(batch_size,actions,poses,pad=pad,causal_shift=causal_shift,test_split=split_ratio)
with torch.no_grad():
    eval_model.eval() 
    if torch.cuda.is_available():
        eval_model = eval_model.cuda()
        # poses = poses.cuda() 
    targets = []
    test_losses = []
    accuracy_test = []
    targets_softmax = []
    for ix,poses in enumerate(poses_test):
        if ix%1000==0:
            print(ix)   
#        poses = np.concatenate(poses)
        poses = np.pad(poses,((54,0),(0,0),(0,0)),'edge')
        poses = torch.Tensor(np.expand_dims(poses,axis=0)).cuda()
        pred = eval_model(poses)
        actions = actions_test[ix]
        orig_action = actions
        if actions>7:
            actions = 6
        act = np.full((1,pred.shape[-1]),actions)

        action = torch.from_numpy(act.astype('long'))
        if torch.cuda.is_available():
            action = action.cuda()

        test_loss  = loss_fun(pred, action)

        test_losses.append(test_loss)
        targets.append((act,pred.detach().cpu().numpy()))
        softmax = torch.nn.Softmax(1)
        pred= softmax(pred)
        pred = pred.detach().cpu().numpy().squeeze()
        preds = np.argmax(pred,axis=0)
        targets_softmax.append((act,preds))
        values, counts = np.unique(preds,return_counts=True)
        ind = np.argmax(counts)
        accuracy_test.append((orig_action,np.sum(values[ind]==actions)))
#        vp3d_keypoints.append(seq_keypoints)
    print(np.mean([x[1] for x in accuracy_test]))
            
torch.save({
        'losses_test': test_losses,
        'test_targets' : targets,
        'targets_softmax': targets_softmax,
        'accuracy' : accuracy_test
        }, os.path.join(CHECKPATH,f'Recipe-2-test-results2.pth') )
   
