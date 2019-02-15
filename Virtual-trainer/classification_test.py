import os
import gc
import sys
import errno
import vp3dutils
import torch

import torch.nn as nn
import torch.optim as optim
import numpy as np

from generators import SequenceGenerator, BatchedGenerator

try:
    # Create checkpoint directory if it does not exist
    os.makedirs('checkpoint')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', 'checkpoint')
CHECKPATH = 'checkpoint'

# parameters
batch_size = 1024
epochs = 5
embedding_len = 128
lr, lr_decay = 0.001 , 0.95 

# H36M pretrained model settings 
# 2d keypoints file: os.path.join('VideoPose3D','data','data_2d_h36m_gt.npz')
keypoint_file = '/keypoints/data_2d_h36m_gt.npz' #floydhub 
# VP3D pretrained model file: os.path.join('VideoPose3D','checkpoint', 'pretrained_h36m_cpn.bin')
chk_filename = '/pretrained/pretrained_h36m_cpn.bin' #floydhub
# model architecture
filter_widths = [3,3,3,3,3]
channels = 1024
in_joints, in_dims, out_joints = 17, 2, 17

# Split by H36M actors
subjects_train = ['S1','S5','S6','S7','S8']
subjects_test = ['S9','S11']

# Actions to classify
action_list=['Walking','Waiting','SittingDown']
classes = len(action_list)

# load training and validation sets
actions_valid, poses_valid_2d = vp3dutils.fetch_keypoints(keypoint_file, subjects_test, action_list)
actions_train, poses_train_2d = vp3dutils.fetch_keypoints(keypoint_file, subjects_train, action_list)


# load pretrained VP3D model
vp3d_model, eval_model = vp3dutils.load_VP3D_model(chk_filename, in_joints, in_dims, out_joints, filter_widths, channels)

# modify models
vp3d_model = vp3dutils.modify_model(vp3d_model, embedding_len, classes)
eval_model = vp3dutils.modify_model(eval_model, embedding_len, classes)

# create parameter list
train_params = list(vp3d_model.shrink.parameters())
train_params += list(vp3d_model.layers_bn[-1].parameters())
train_params += list(vp3d_model.layers_bn[-2].parameters())
train_params += list(vp3d_model.layers_conv[-1].parameters())
train_params += list(vp3d_model.layers_conv[-2].parameters())

loss_fun = nn.CrossEntropyLoss()
optimizer = optim.Adam(train_params,lr, amsgrad=True)

receptive_field = vp3d_model.receptive_field()
pad = (receptive_field - 1) // 2 
causal_shift = pad

# sequence generators to feed models
train_generator = BatchedGenerator(batch_size, actions_train, poses_train_2d, pad=pad, causal_shift=causal_shift)
#train_generator_eval = SequenceGenerator(actions_train, poses_train_2d, pad=pad, causal_shift=causal_shift)
test_generator = SequenceGenerator(actions_valid, poses_valid_2d, pad=pad, causal_shift=causal_shift)

if torch.cuda.is_available():
    vp3d_model = vp3d_model.cuda()
    eval_model = eval_model.cuda()

# train model
epoch = 0
losses_train = []
losses_test = []

while epoch < epochs:
    epoch_loss_train = 0 
    vp3d_model.train()

    avg_time = []    
    # train minibatches
    for batch_act, batch_2d in train_generator.next_epoch():
        batch_act = np.argmax(batch_act,axis=1).reshape(-1,1)
        action = torch.from_numpy(batch_act.astype('long'))
        poses = torch.from_numpy(batch_2d.astype('float32'))
        if torch.cuda.is_available():
            action = action.cuda()
            poses = poses.cuda()
        
        optimizer.zero_grad()
        pred = vp3d_model(poses)
        batch_loss = loss_fun(pred, action)
        print('{{"metric": "Batch Loss", "value": {}}}'.format(batch_loss))
        epoch_loss_train += batch_loss 
        batch_loss.backward()
        optimizer.step()
        gc.collect() # only needed in cpu 
    losses_train.append(epoch_loss_train)

    # evaluate every epoch
    with torch.no_grad():
        eval_model.load_state_dict(vp3d_model.state_dict())
        eval_model.eval()
        epoch_loss_test = 0
        for batch_act, batch_2d in test_generator.next_epoch():
            batch_act = np.argmax(batch_act,axis=1).reshape(-1,1)
            action = torch.from_numpy(batch_act.astype('long'))
            poses = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                action = action.cuda()
                poses = poses.cuda()         
            pred = eval_model(poses)
            batch_loss = loss_fun(pred, action)
            epoch_loss_test += batch_loss
            gc.collect() # only needed in cpu
        losses_test.append(epoch_loss_test)

    print('{{"metric": "Cross Entropy Loss", "value": {}, "epoch": {}}}'.format(
            losses_train.item(), epoch))
    print('{{"metric": "Validation Loss", "value": {}, "epoch": {}}}'.format(
            losses_test.item(), epoch))

    # checkpoint every epoch
    torch.save({
            'epoch': epoch,
            'model_state_dict': vp3d_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, CHECKPATH)

    lr *= lr_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    epoch += 1    






