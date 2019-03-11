"""
@author: Ahmad Kurdi
Test discrimination model for action classifier
DSR portfolio project with Artur Silicki
"""


import os
import sys
import errno
import time
import torch

import torch.nn as nn
import torch.optim as optim
import numpy as np

from models.semantic import NaiveStridedModel, HeadlessNet
from models.loss import ContrastiveLoss
from dataloader import *
from simple_generators import SimpleSequenceGenerator
from siamese_generator import SiameseGenerator

try:
    # Create checkpoint directory if it does not exist
    os.makedirs('checkpoint')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', 'checkpoint')
CHECKPATH = 'checkpoint'

# Data mountpoint
DATAPOINT = "Data"

def train_epoch():
    epoch_loss_train = [] 
    model.train()  
    # train minibatches
    for X1, X2, diffs in generator.next_batch():
        if X1.shape[0]==0 or X2.shape[0]==0:
            continue
        diffs = torch.from_numpy(diffs.astype('float32'))
        X1 = torch.from_numpy(X1.astype('float32'))
        X2 = torch.from_numpy(X2.astype('float32'))
        if torch.cuda.is_available():
            diffs = diffs.cuda()
            X1 = X1.cuda()
            X2 = X2.cuda()
        
        optimizer.zero_grad()
        embed1, embed2 = model(X1), model(X2)
        batch_loss = loss_fun(embed1, embed2, diffs)
        print('{{"metric": "Batch Loss", "value": {}}}'.format(batch_loss))
        epoch_loss_train.append(batch_loss.detach().cpu().numpy()) 
        batch_loss.backward()
        optimizer.step()  
    return epoch_loss_train

def evaluate_epoch():
    with torch.no_grad():
        model.eval()
        epoch_loss_test = []
        for X1, X2, diffs in generator.next_validation():
            if X1.shape[0]==0 or X2.shape[0]==0:
                continue
            diffs = torch.from_numpy(diffs.astype('float32'))
            X1 = torch.from_numpy(X1.astype('float32'))
            X2 = torch.from_numpy(X2.astype('float32'))
            if torch.cuda.is_available():
                diffs = diffs.cuda()
                X1 = X1.cuda()
                X2 = X2.cuda()
            embed1, embed2 = model(X1), model(X2)
            batch_loss = loss_fun(embed1, embed2, diffs)
            epoch_loss_test.append(batch_loss.detach().cpu().numpy())     
    return epoch_loss_test 

def log_results(epoch, st, epoch_loss_train, epoch_loss_test,loss_tuple):  
    losses_train.append(epoch_loss_train)
    losses_test.append(epoch_loss_test)
    print(f'Time per epoch: {(time.time()-st)//60}')
    print('{{"metric": "Training Loss", "value": {}, "epoch": {}}}'.format(
            np.mean(epoch_loss_train), epoch)) 
    print('{{"metric": "Validation Loss", "value": {}, "epoch": {}}}'.format(
            np.mean(epoch_loss_test), epoch)) 
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses_test': losses_test,
            'loss_tuple': loss_tuple
            }, os.path.join(CHECKPATH,f'siamese-1-{selected_class}-{epoch}.pth') )



def load_keypoints(kpfile_1,kpfile_2,rating_file,seed,target):
    df = fetch_s_df(kpfile_1,kpfile_2,rating_file,seed)
    return fetch_s_keypoints(df,target)


def load_headless_model(chk_filename):
    
    # model architecture
    filter_widths = [3,3,3]
    channels = 1024
    in_joints, in_dims, out_joints = 17, 2, 17
    classes = 8
    embedding_len = 128

    pretrained_weights = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    model = NaiveStridedModel(in_joints, in_dims, out_joints, filter_widths, {}, embedding_len, classes,
                                                                                causal=True, dropout=0.25, channels=channels, loadBase=False)
    return HeadlessNet(model, pretrained_weights)


# --- params ---
pretrained = os.path.join(CHECKPATH,"Recipe-2-epoch-15.pth") # pretrained base
kpfile_1 = os.path.join(DATAPOINT,"Keypoints","keypoints.csv") # squats and dealifts
kpfile_2 = os.path.join(DATAPOINT,"Keypoints", "keypoints_rest.csv") # rest of classes
rating_file = os.path.join(DATAPOINT,"clips-rated.csv") # rating labels file
# selected_class = 0 # Squat
selected_class = 1 # Deadlift
loss_margin = 0.5
rate_range = 3
seed = 1234
lr, lr_decay = 0.001 , 0.95 
split_ratio = 0.2
epochs = 20
batch_size = 32
n_chunks = 8

model = load_headless_model(pretrained)
model.cuda()
receptive_field = model.embed_model.base_model.receptive_field()
pad = (receptive_field - 1) 
causal_shift = pad

# initialise everything
actions, out_poses_2d,from_clip_flgs, return_idx, ratings = load_keypoints(kpfile_1,kpfile_2,
                                                                    rating_file,seed,selected_class)
from siamese_generator import SiameseGenerator
generator = SiameseGenerator(batch_size, actions, out_poses_2d,ratings,from_clip_flgs, pad=pad,
                 causal_shift=causal_shift, test_split=split_ratio,n_chunks=n_chunks, random_seed=seed) # todo

loss_fun = ContrastiveLoss(rate_range,loss_margin)
optimizer = optim.Adam(list(model.embed_model.top_model.parameters()),lr, amsgrad=True)
losses_train = []
losses_test = []

# run training
for epoch in range(1,epochs+1):
    st = time.time()
    epoch_loss_train = train_epoch()
    epoch_loss_test = evaluate_epoch()
    loss_tuple = loss_fun.get_tuples()
    log_results(epoch, st, epoch_loss_train, epoch_loss_test,loss_tuple)
    lr *= lr_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay
    generator.next_epoch()  