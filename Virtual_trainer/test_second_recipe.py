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
ucf_file = os.path.join(DATAPOINT,'Keypoints','keypoints_rest.csv')
ucf_file_test = os.path.join(DATAPOINT,'Keypoints','keypoints_rest_test.csv')
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
actions = action_op + action_op2
actions = [action if action!=8 else 6 for action in actions]
poses = poses_op + poses_op2
vid_idx = vid_idx + vid_idx2

# balance the dataset
seed = 1234
balanced, test_ids = balance_dataset_recipe2(np.array(actions),seed)
actions_test = [actions[x] for x in test_ids]
poses_test = [poses[x] for x in test_ids]
actions_test = actions_test + action_op_test
poses_test = poses_test + poses_op_test


# build models
classes = len(np.unique(actions))
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

checkp = torch.load('/home/artursil/Documents/vt2/recipe1/checkpoint/model4-19.pth')
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
    for ix,poses in enumerate(poses_test):
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
        values, counts = np.unique(preds,return_counts=True)
        ind = np.argmax(counts)
        accuracy_test.append((orig_action,np.sum(values[ind]==actions)))
    print(np.mean([x[1] for x in accuracy_test]))
            
torch.save({
        'losses_test': test_losses,
        'test_targets' : targets,
        'accuracy' : accuracy
        }, os.path.join(CHECKPATH,f'Recipe-2-test-results.pth') )





#while epoch < epochs:
#    epoch_loss_train = [] 
#    trn_model.train()  
#    # train minibatches
#    for batch_act, batch_2d in generator.next_batch():
#        #batch_act = batch_act.reshape(-1,1)
#        action = torch.from_numpy(batch_act.astype('long'))
#        poses = torch.from_numpy(batch_2d.astype('float32'))
#        if torch.cuda.is_available():
#            action = action.cuda()
#            poses = poses.cuda()
#        
#        optimizer.zero_grad()
#        pred = trn_model(poses)
#        batch_loss = loss_fun(pred, action)
#        print('{{"metric": "Batch Loss", "value": {}}}'.format(batch_loss))
#        epoch_loss_train.append(batch_loss.detach().cpu().numpy()) 
#        batch_loss.backward()
#        optimizer.step()
#        #gc.collect() # only needed in cpu 
#    losses_train.append(epoch_loss_train)
#
#    # evaluate every epoch
#    with torch.no_grad():
#        eval_model.load_state_dict(trn_model.state_dict())
#        eval_model.eval()
#        epoch_loss_test = []
#        targets = []
#        for y_val, batch_2d in generator.next_validation():          
#            poses = torch.from_numpy(batch_2d.astype('float32'))
#            if torch.cuda.is_available():
#                poses = poses.cuda()         
#            pred = eval_model(poses)
#            batch_act = np.full((1,pred.shape[-1]),y_val)
#            action = torch.from_numpy(batch_act.astype('long'))
#            if torch.cuda.is_available():
#                action = action.cuda()
#            batch_loss = loss_fun(pred, action)
#            epoch_loss_test.append(batch_loss.detach().cpu().numpy())
#            targets.append((batch_act,pred.detach().cpu().numpy()))
#        losses_test.append(epoch_loss_test) 
#        validation_targets.append(targets)  # store (target,prediction) tuple for analysis
#
#    print('{{"metric": "Cross Entropy Loss", "value": {}, "epoch": {}}}'.format(
#            np.mean(epoch_loss_train), epoch)) 
#    print('{{"metric": "Validation Loss", "value": {}, "epoch": {}}}'.format(
#            np.mean(epoch_loss_test), epoch)) 
#
#    # checkpoint every epoch
#    torch.save({
#            'epoch': epoch,
#            'model_state_dict': trn_model.state_dict(),
#            'optimizer_state_dict': optimizer.state_dict(),
#            'losses_test': losses_test,
#            'validation_targets' : validation_targets,
#            'training_set' : vid_idx
#            }, os.path.join(CHECKPATH,f'Recipe-2-epoch-{epoch}.pth') )
#
#    lr *= lr_decay
#    for param_group in optimizer.param_groups:
#        param_group['lr'] *= lr_decay
#    epoch += 1
#    generator.next_epoch()  
#      
