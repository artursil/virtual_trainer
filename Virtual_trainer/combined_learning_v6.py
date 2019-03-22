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
import neptune
from neptune_connector.np_config import NEPTUNE_TOKEN
from models.semantic import ModdedTemporalModel, ModdedStridedModel, StandardiseKeypoints, HeadlessNet2, SplitModel, SplitModel2
from models.loss import CombinedLoss
from dataloader import *
from simple_generators import SimpleSequenceGenerator, SimpleSiameseGenerator
from siamese_generator import SiameseGenerator
from collections import OrderedDict
from graph_utils import *


try:
    # Create checkpoint directory if it does not exist
    os.makedirs('checkpoint')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', 'checkpoint')
CHECKPATH = 'checkpoint'

# Data mountpoint
DATAPOINT = "../../../Data"
PROJECT_NAME = 'VT-combined'
EXPERIMENT_NAME = 'v6-SGD'
METRICSPATH = os.path.join('metrics',EXPERIMENT_NAME)


try:
    # Create metrics directory if it does not exist
    os.makedirs(METRICSPATH)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create metrics directory:', 'metrics')


loss_margin = 0.3
seed = 1234
lr, lr_decay = 0.001 , 0.95 
split_ratio = 0.2
epochs = 5
batch_size = 512
n_chunks = 8
weighting = 0.999 # classification loss weighting
weighting_decay = 0.95 
supress_cl = 6
class_weight =[1.,1.,3.,3.,3.,3.,3.,3.]
rmse = True


neptune.init(api_token=NEPTUNE_TOKEN,
             project_qualified_name=f'artursil/{PROJECT_NAME}')
neptune.create_experiment(EXPERIMENT_NAME,
                          params={'weighting': weighting,
                                  'weighting_decay': weighting_decay,
                                  'batch_size':batch_size,
                                  'lr':lr,
                                  'lr_decay':lr_decay,
                                  'loss_margin': loss_margin,
                                  'class_weight': f'{class_weight}',
                                  'emb_layer': '[128,128]',
                                  'optimiser': 'sgd'
                                  
                                  })


def train_model(model,epoch_start, epochs,lr,lr_decay,weighting,weighting_decay):
    loss_tuple = []
    for epoch in range(epoch_start,epochs+1):
        if torch.cuda.is_available():
            model.cuda()
        print(f"Starting epoch {epoch}")
        st = time.time()
        epoch_loss_train = train_epoch(model)
        print(f"Epoch {epoch}: Training complete, beginning evaluation")
        epoch_loss_test, val_targets, pairings = evaluate_epoch(model)
        log_results(epoch, st, epoch_loss_train, epoch_loss_test,val_targets, pairings)
        loss_tuple.append(pairings)
        lr *= lr_decay
        
        weighting *= weighting_decay
        loss_fun.set_weighting(weighting)

        #for param_group in optimizer.param_groups:
        #    param_group['lr'] *= lr_decay
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
        neptune.send_metric('batch_loss', batch_loss)
        epoch_loss_train.append(batch_loss.detach().cpu().numpy()) 
        batch_loss.backward()
        optimizer.step()
        
    return epoch_loss_train

def evaluate_epoch(model):
    with torch.no_grad():
        model.eval()
        epoch_loss_test = []
        pairings, targets = [], []
        for X, classes_np, rankings_np in generator.next_validation():
            X = torch.from_numpy(X.astype('float32'))
            classes = torch.from_numpy(classes_np.astype('long'))
            rankings = torch.from_numpy(rankings_np.astype('long'))
            if torch.cuda.is_available():
                X = X.cuda()
                classes = classes.cuda()
                rankings = rankings.cuda()
            embeds, preds = model(X)
            batch_loss = loss_fun(embeds, preds, classes, rankings)
            pairings.append(np.concatenate([p.reshape(-1,4) for p in loss_fun.get_pairings()]))
            epoch_loss_test.append(batch_loss.detach().cpu().numpy())
            targets.append( (classes_np,rankings_np,preds.detach().cpu().numpy().squeeze(),
                            embeds.detach().cpu().numpy().squeeze()) )     
            
    return epoch_loss_test, targets, pairings

def log_results(epoch, st, epoch_loss_train, epoch_loss_test,val_targets, pairings):  
    losses_train.append(epoch_loss_train)
    losses_test.append(epoch_loss_test)
    
    print(f'Time for epoch {epoch}: {(time.time()-st)//60}')
    print(f"Training Loss: {np.mean(epoch_loss_train)} , Validation Loss: {np.mean(epoch_loss_test)}")
    neptune.send_metric('training_loss', np.mean(epoch_loss_train))
    neptune.send_metric('val_loss', np.mean(epoch_loss_test))

    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'losses_test': losses_test,
            'validation_targets':val_targets
            }, os.path.join(CHECKPATH,f'combinedlearning-{EXPERIMENT_NAME}-{epoch}.pth') )

    # prepare charts
    prepare_plots2(pairings, val_targets, epoch, METRICSPATH)



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

def build_model(chk_filename, in_joints, in_dims, out_joints, filter_widths, filter_widths2, causal, channels, embedding_len,classes):

    base= TemporalModel(in_joints,in_dims,out_joints,filter_widths,causal=True,dropout=0.25,channels=channels)
    top= ModdedStridedModel(in_joints, 3, out_joints, filter_widths2, causal=True, dropout=0.25, channels=embedding_len, skip_res=False)
    class_mod = nn.Conv1d( embedding_len, classes, 1)

    base, top, class_mod = load_model_weights(chk_filename,base,top,class_mod)
    model = nn.Sequential(OrderedDict([
          ('base', base) ,
          ('transform', StandardiseKeypoints(True,False)),
          ('embedding', HeadlessNet2(top)),
          ('classifier', SplitModel2(class_mod,embedding_len,128) )
        ]))
    return model 

# model architecture
filter_widths = [3,3,3]
filter_widths2 = [5,3,3]
channels = 1024
in_joints, in_dims, out_joints = 17, 2, 17
causal = True
embedding_len = 128
classes=8

# --- params ---
pretrained = os.path.join(CHECKPATH,"Recipe-2-embedvariant-128-epoch-4.pth") # pretrained base
kpfile_1 = os.path.join(DATAPOINT,"Keypoints","keypoints.csv") # squats and dealifts
kpfile_2 = os.path.join(DATAPOINT,"Keypoints", "keypoints_rest.csv") # rest of classes
rating_file = os.path.join(DATAPOINT,"clips-rated.csv") # rating labels file
ucf_file = os.path.join(DATAPOINT,'Keypoints','keypoints_rest.csv')
# non exercise class id to supress from ranking loss

model = build_model(pretrained, in_joints, in_dims, out_joints, filter_widths, filter_widths2, causal, channels, embedding_len,classes)

epoch=1

resume_cp = os.path.join(CHECKPATH,'combinedlearning-v5-533-rank_embed-128-9.pth')
checkp = torch.load(resume_cp)
epoch = checkp['epoch']+1
model.load_state_dict(checkp['model_state_dict'])

weighting = 0.5

receptive_field = model.embedding.embed_model.receptive_field()
pad = (receptive_field - 1) 
causal_shift = pad

class_weight = torch.tensor(class_weight, dtype=torch.float)

# initialise everything
actions, out_poses_2d,from_clip_flgs, return_idx, ratings, filenames_final, targets = load_keypoints(kpfile_1,kpfile_2, # todo
                                                                    rating_file,seed,None)
action_ucf, poses_ucf, _ = fetch_openpose_keypoints(ucf_file)
action_ucf = [action if action!=8 else 6 for action in action_ucf]
poses_ucf = [p for a,p in zip(action_ucf,poses_ucf) if a==6]
action_ucf = [action for action in action_ucf if action==6]
action_ucf, poses_ucf = action_ucf[:50], poses_ucf[:50]
ratings_ucf = [9] * len(action_ucf)

targets += action_ucf
out_poses_2d += poses_ucf
ratings += ratings_ucf

generator = SimpleSiameseGenerator(batch_size, targets, out_poses_2d,ratings, pad=pad,
                 causal_shift=causal_shift, test_split=split_ratio, random_seed=seed) # todo

if torch.cuda.is_available():
    class_weight = class_weight.cuda()
loss_fun = CombinedLoss(nn.CrossEntropyLoss(weight=class_weight),weighting,supress_cl=supress_cl,rooting=rmse,margin=loss_margin)
train_parameters = list(model.embedding.parameters()) + list(model.classifier.parameters())
optimizer = optim.SGD(train_parameters, lr, weight_decay=lr_decay)
losses_train = []
losses_test = []

# run training
print(f"start epoch: {epoch}")
train_model(model,epoch,epochs,lr,lr_decay,weighting,weighting_decay)
neptune.stop()
