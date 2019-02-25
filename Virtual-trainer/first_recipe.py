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

from models.semantic import NaiveBaselineModel, NaiveStridedModel
from dataloader import *
from simple_generators import SimpleSequenceGenerator

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
action_op, poses_op = fetch_openpose_keypoints(instagram_file)
action_vp3d, poses_vp3d = fetch_vp3d_keypoints(h36m_file,subjects,action_list,len(np.unique(action_op)),subset=0.5)
actions = action_op + action_vp3d
poses = poses_op + poses_vp3d

# balance the dataset
balanced = balance_dataset(np.array(actions))
actions = [actions[b] for b in balanced]
poses = [poses[b] for b in balanced]

# build models
classes = len(np.unique(actions))
pretrained_weights = torch.load(chk_filename, map_location=lambda storage, loc: storage)
trn_model = NaiveStridedModel(in_joints, in_dims, out_joints, filter_widths, pretrained_weights, embedding_len, classes,
                                causal=True, dropout=0.25, channels=channels)
eval_model = NaiveBaselineModel(in_joints, in_dims, out_joints, filter_widths, pretrained_weights, embedding_len, classes,
                                causal=True, dropout=0.25, channels=channels)

train_params = list(trn_model.top_model.parameters())

loss_fun = nn.CrossEntropyLoss()
optimizer = optim.Adam(train_params,lr, amsgrad=True)

receptive_field = trn_model.base_model.receptive_field()
pad = (receptive_field - 1) 
causal_shift = pad

# build generator

generator = SimpleSequenceGenerator(batch_len,actions,poses,pad=pad,causal_shift=causal_shift,test_split=split_ratio)

if torch.cuda.is_available():
    trn_model = trn_model.cuda()
    eval_model = eval_model.cuda()

# train model
epoch = 0
losses_train = []
losses_test = []

while epoch < epochs:
    epoch_loss_train = [] 
    trn_model.train()  
    # train minibatches
    for batch_act, batch_2d in generator.next_batch():
        #batch_act = batch_act.reshape(-1,1)
        action = torch.from_numpy(batch_act.astype('long'))
        poses = torch.from_numpy(batch_2d.astype('float32'))
        if torch.cuda.is_available():
            action = action.cuda()
            poses = poses.cuda()
        
        optimizer.zero_grad()
        pred = trn_model(poses)
        batch_loss = loss_fun(pred, action)
        print('{{"metric": "Batch Loss", "value": {}}}'.format(batch_loss))
        epoch_loss_train.append(batch_loss.detach().cpu().numpy()) 
        batch_loss.backward()
        optimizer.step()
        #gc.collect() # only needed in cpu 
    losses_train.append(epoch_loss_train)

    # evaluate every epoch
    with torch.no_grad():
        eval_model.load_state_dict(trn_model.state_dict())
        eval_model.eval()
        epoch_loss_test = []
        for y_val, batch_2d in generator.next_validation():
            #top_pad = trn_model.top_model.receptive_field() - 1 // 2
            #batch_act = np.full(batch_2d.shape[1] - top_pad, y_val)
            #np.expand_dims(batch_act, axis=0)
            action = torch.from_numpy(batch_act.astype('long'))
            poses = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                #action = action.cuda()
                poses = poses.cuda()         
            pred = eval_model(poses)
            action = np.full(pred.shape, y_val)
            action = torch.from_numpy(batch_act.astype('long'))
            if torch.cuda.is_available():
                action = action.cuda()
            batch_loss = loss_fun(pred, action)
            epoch_loss_test.append(batch_loss.detach().cpu().numpy())
            #gc.collect() # only needed in cpu
        losses_test.append(epoch_loss_test)

    print('{{"metric": "Cross Entropy Loss", "value": {}, "epoch": {}}}'.format(
            np.mean(epoch_loss_train), epoch)) 
    print('{{"metric": "Validation Loss", "value": {}, "epoch": {}}}'.format(
            np.mean(epoch_loss_test), epoch)) 

    # checkpoint every epoch
    torch.save({
            'epoch': epoch,
            'model_state_dict': trn_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(CHECKPATH,f'model-{epoch}.pth') )

    lr *= lr_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    epoch += 1
    generator.next_epoch()    
