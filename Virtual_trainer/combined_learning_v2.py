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
from models.semantic import ModdedTemporalModel, ModdedStridedModel, StandardiseKeypoints, HeadlessNet2, SplitModel
from models.loss import CombinedLoss
from dataloader import *
from simple_generators import SimpleSequenceGenerator, SimpleSiameseGenerator
from siamese_generator import SiameseGenerator
from collections import OrderedDict

try:
    # Create checkpoint directory if it does not exist
    os.makedirs('checkpoint')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', 'checkpoint')
CHECKPATH = '/home/ahmad/project/checkpoint'

# Data mountpoint
DATAPOINT = "/home/ahmad/project/Data"


def train_model(model,eval_model, epochs):
    loss_tuple = []
    for epoch in range(1,epochs+1):
        if torch.cuda.is_available():
            model.cuda()
            eval_model.cuda()
        st = time.time()
        #epoch_loss_train = train_epoch(model)
        eval_model.load_state_dict(model.state_dict())
        epoch_loss_test = evaluate_epoch(eval_model)
        loss_tuple.append(loss_fun.get_pairings())
        log_results(epoch, st, epoch_loss_train, epoch_loss_test)
        lr *= lr_decay
        
        weighting *= weighting_decay
        loss_fun.set_weighting(weighting)

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
        embeds, preds = model(X)
        batch_loss = loss_fun(embeds, preds, classes, rankings)
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
            classes = torch.tensor(classes, dtype=torch.long).repeat(X.shape[1] - 52) #(np.array(classes).astype('long'))
            rankings = torch.tensor(rankings, dtype=torch.long).repeat(X.shape[1] - 52) #torch.from_numpy(rankings.astype('long'))
            if torch.cuda.is_available():
                X = X.cuda()
                classes = classes.cuda()
                rankings = rankings.cuda()
            embeds, preds = model(X)
            batch_loss = loss_fun(embeds, preds, classes, rankings)
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
            }, os.path.join(CHECKPATH,f'combinedlearning-{epoch}.pth') )



def load_keypoints(kpfile_1,kpfile_2,rating_file,seed,target):
    df = fetch_s_df(kpfile_1,kpfile_2,rating_file,seed)
    return fetch_s_keypoints(df,target)


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
    top= ModdedStridedModel(in_joints, 3, out_joints, filter_widths, causal=True, dropout=0.25, channels=embedding_len, skip_res=False)
    base_eval= TemporalModel(in_joints,in_dims,out_joints,filter_widths,causal=True,dropout=0.25,channels=channels)
    top_eval= ModdedTemporalModel(in_joints, 3, out_joints, filter_widths, causal=True, dropout=0.25, channels=embedding_len, skip_res=False)
    class_mod = nn.Conv1d( embedding_len, classes, 1)

    base, top, class_mod = load_model_weights(chk_filename,base,top,class_mod)
    model = nn.Sequential(OrderedDict([
          ('base', base) ,
          ('transform', StandardiseKeypoints(True,False)),
          ('embedding', HeadlessNet2(top)),
          ('classifier', SplitModel(class_mod) )
        ]))

    eval_model = nn.Sequential(OrderedDict([
          ('base', base_eval) ,
          ('transform', StandardiseKeypoints(True,False)),
          ('embedding', HeadlessNet2(top_eval)),
          ('classifier', SplitModel(class_mod))
        ]))
    return model, eval_model

# model architecture
filter_widths = [3,3,3]
channels = 1024
in_joints, in_dims, out_joints = 17, 2, 17
causal = True
embedding_len = 128
classes=8

# --- params ---
pretrained = os.path.join(CHECKPATH,"Recipe-2-epoch-19.pth") # pretrained base
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
weighting = 0.99 # classification loss weighting
weighting_decay = 0.9 
supress_cl = 6 # non exercise class id to supress from ranking loss

model, eval_model = build_model(pretrained, in_joints, in_dims, out_joints, filter_widths, causal, channels, embedding_len,classes)

receptive_field = model.base.receptive_field()
pad = (receptive_field - 1) 
causal_shift = pad

# initialise everything
actions, out_poses_2d,from_clip_flgs, return_idx, ratings, filenames_final, targets = load_keypoints(kpfile_1,kpfile_2, # todo
                                                                    rating_file,seed,None)

generator = SimpleSiameseGenerator(batch_size, targets, out_poses_2d,ratings, pad=pad,
                 causal_shift=causal_shift, test_split=split_ratio, random_seed=seed) # todo


loss_fun = CombinedLoss(nn.CrossEntropyLoss(),weighting,supress_cl=supress_cl,margin=loss_margin)
train_parameters = list(model.embedding.parameters()) + list(model.classifier.parameters())
optimizer = optim.Adam(train_parameters,lr, amsgrad=True)
losses_train = []
losses_test = []

# run training
train_model(model,eval_model,epochs)