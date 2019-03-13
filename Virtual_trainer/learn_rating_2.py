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
import re

import torch.nn as nn
import torch.optim as optim
import numpy as np

from VideoPose3D.common.model import TemporalModel, TemporalModelOptimized1f
from models.semantic import ModdedTemporalModel, ModdedStridedModel, StandardiseKeypoints, HeadlessNet2, RankingEmbedder
from models.loss import CustomRankingLoss
from dataloader import *
from simple_generators import SimpleSequenceGenerator
from siamese_generator import SiameseGenerator
from collections import OrderedDict

try:
    # Create checkpoint directory if it does not exist
    os.makedirs('checkpoint')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', 'checkpoint')
CHECKPATH = 'checkpoint'

# Data mountpoint
DATAPOINT = "Data"

# model architecture
filter_widths = [3,3,3]
channels = 1024
in_joints, in_dims, out_joints = 17, 2, 17
causal = True
embedding_len = 128
embeds=[64,64]
classes=8

# --- params ---
pretrained = os.path.join(CHECKPATH,"Recipe-2-epoch-15.pth") # pretrained base
kpfile_1 = os.path.join(DATAPOINT,"Keypoints","keypoints.csv") # squats and dealifts
kpfile_2 = os.path.join(DATAPOINT,"Keypoints", "keypoints_rest.csv") # rest of classes
rating_file = os.path.join(DATAPOINT,"clips-rated.csv") # rating labels file
loss_margin = 0.3
seed = 1234
lr, lr_decay = 0.001 , 0.95 
split_ratio = 0.2
epochs = 20
batch_size = 512
n_chunks = 8

model, eval_model = build_model(pretrained, in_joints, in_dims, out_joints, filter_widths, causal, channels, embedding_len, embeds,classes)

receptive_field = model.base.receptive_field()
pad = (receptive_field - 1) 
causal_shift = pad

# initialise everything
actions, out_poses_2d,from_clip_flgs, return_idx, ratings = load_keypoints(kpfile_1,kpfile_2, # todo
                                                                    rating_file,seed,selected_class)
from siamese_generator import SiameseGenerator
generator = SiameseGenerator(batch_size, actions, out_poses_2d,ratings,from_clip_flgs, pad=pad,
                 causal_shift=causal_shift, test_split=split_ratio,n_chunks=n_chunks, random_seed=seed) # todo


loss_fun = CustomRankingLoss(margin=loss_margin)
optimizer = optim.Adam(list(model.embedding.parameters()),lr, amsgrad=True)
losses_train = []
losses_test = []

# run training
train_model(model,eval_model,epochs)

def train_model(model,eval_model, epochs):
    loss_tuple = []
    for epoch in range(1,epochs+1):
        if torch.cuda.is_available():
            model.cuda()
            eval_model.cuda()
        st = time.time()
        epoch_loss_train = train_epoch(model)
        eval_model.load_state_dict(model.state_dict)
        epoch_loss_test = evaluate_epoch(eval_model)
        loss_tuple.append(loss_fun.get_pairings())
        log_results(epoch, st, epoch_loss_train, epoch_loss_test)
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        generator.next_epoch()  

def train_epoch(model):
    epoch_loss_train = [] 
    model.train()  
    # train minibatches
    for X, classes, rankings in generator.next_batch():
        X = torch.from_numpy(X.astype('float32'))
        classes = torch.from_numpy(classes.astype('long'))
        rankings = torch.from_numpy(rankings.astype('long'))
        if torch.cuda.is_available():
            X = X.cuda()
            classes = classes.cuda()
            rankings = rankings.cuda()
        
        optimizer.zero_grad()
        embeds = model(X)
        batch_loss = loss_fun(embeds, classes, rankings)
        print('{{"metric": "Batch Loss", "value": {}}}'.format(batch_loss))
        epoch_loss_train.append(batch_loss.detach().cpu().numpy()) 
        batch_loss.backward()
        optimizer.step()  
    return epoch_loss_train

def evaluate_epoch(model):
    with torch.no_grad():
        model.eval()
        epoch_loss_test = []
        for X, classes, rankings in generator.next_validation():
            X = torch.from_numpy(X.astype('float32'))
            classes = torch.from_numpy(classes.astype('long'))
            rankings = torch.from_numpy(rankings.astype('long'))
            if torch.cuda.is_available():
                X = X.cuda()
                classes = classes.cuda()
                rankings = rankings.cuda()
            embeds = model(X)
            batch_loss = loss_fun(embeds, classes, rankings)
            epoch_loss_test.append(batch_loss.detach().cpu().numpy())     
    return epoch_loss_test 

def log_results(epoch, st, epoch_loss_train, epoch_loss_test):  
    losses_train.append(epoch_loss_train)
    losses_test.append(epoch_loss_test)
    print(f'Time per epoch: {(time.time()-st)//60}')
    print('{{"metric": "Training Loss", "value": {}, "epoch": {}}}'.format(
            np.mean(epoch_loss_train), epoch)) 
    print('{{"metric": "Validation Loss", "value": {}, "epoch": {}}}'.format(
            np.mean(epoch_loss_test), epoch))

def save_checkpoint(loss_tuple): 
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


def load_model_weights(chk_filename,base,top):
    

    pretrained_weights = torch.load(chk_filename, map_location=lambda storage, loc: storage)['model_state_dict']
    
    base_weights = OrderedDict([(re.sub(r"(base_model.)","",key), value) for key, value in pretrained_weights.items() if key.startswith("base_model")])
    top_weights = OrderedDict([(re.sub(r"(top_model.)","",key), value) for key, value in pretrained_weights.items() if key.startswith("top_model")])
    base.load_state_dict(base_weights, strict=False)
    top.load_state_dict(top_weights, strict=False)
    return base, top

def build_model(chk_filename, in_joints, in_dims, out_joints, filter_widths, causal, channels, embedding_len, embeds,classes):

    base= TemporalModelOptimized1f(in_joints,in_dims,out_joints,filter_widths,causal=True,dropout=0.25,channels=channels)
    top= ModdedStridedModel(in_joints, 3, out_joints, filter_widths, causal=True, dropout=0.25, channels=embedding_len, skip_res=False)
    base_eval= TemporalModel(in_joints,in_dims,out_joints,filter_widths,causal=True,dropout=0.25,channels=channels)
    top_eval= ModdedTemporalModel(in_joints, 3, out_joints, filter_widths, causal=True, dropout=0.25, channels=embedding_len, skip_res=False)
    top.shrink= nn.Conv1d( embedding_len, classes, 1)
    top_eval.shrink = nn.Conv1d( embedding_len, classes, 1)


    base, top = load_model_weights(chk_filename,base,top)
    model = nn.Sequential(OrderedDict([
          ('base', base) ,
          ('transform', StandardiseKeypoints(True,False)),
          ('top', HeadlessNet2(top)),
          ('embedding', RankingEmbedder(embedding_len,embeds))
        ]))

    eval_model = nn.Sequential(OrderedDict([
          ('base', base_eval) ,
          ('transform', StandardiseKeypoints(True,False)),
          ('top', HeadlessNet2(top_eval)),
          ('embedding', RankingEmbedder(embedding_len,embeds))
        ]))
    return model, eval_model