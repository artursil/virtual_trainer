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
from models.semantic import ModdedTemporalModel, ModdedStridedModel, StandardiseKeypoints, HeadlessNet2, SplitModel3, SimpleRegression
from models.loss import  CombinedLoss3
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
DATAPOINT = "Data"
PROJECT_NAME = 'VT-combined'
EXPERIMENT_NAME = 'simple-regressor-grouped-512'
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
epochs = 1000
batch_size = 512
n_chunks = 8
weighting = 0.999 # classification loss weighting
weighting_decay = 0.95 
supress_cl = 6 
freeze_e = 5


neptune.init(api_token=NEPTUNE_TOKEN,
             project_qualified_name=f'artursil/{PROJECT_NAME}')
neptune.create_experiment(EXPERIMENT_NAME,
                          params={'weighting': weighting,
                                  'weighting_decay': weighting_decay,
                                  'batch_size':batch_size,
                                  'lr':lr,
                                  'lr_decay':lr_decay,
                                  'network_layers': '[128,64,32]',
                                  'optimiser': 'rmsprop'
                                  
                                  })
from bokeh.io.export import get_screenshot_as_png
from bokeh.palettes import magma
from bokeh.transform import jitter
from bokeh.layouts import widgetbox
from bokeh.models.widgets import Panel, Tabs, Slider
from bokeh.plotting import figure, save, output_file
from sklearn.metrics import confusion_matrix
from scipy.special import softmax
from PIL import Image

def box_plot(x,y,cl_ar,bok_file,epoch):
    class_names = ['squat', 'deadlift', 'pushups', 'pullups', 'wallpushups', 'lunges', 'other', 'cleanandjerk']
    num_of_classes = len(class_names)
    df = pd.DataFrame(data={ 'tar': x, 'pred': y, 'class': cl_ar})
    palette = magma(num_of_classes + 1)
    p = figure(plot_width=600, plot_height=800, title=f"Ranking by exercise, epoch {epoch}")
    p.xgrid.grid_line_color = None
    p.xaxis.axis_label = 'Target ranking'
    p.yaxis.axis_label = 'Predicted ranking'
    

    
    for cl in range(num_of_classes):
        if cl == 6:
            continue
        df2 = df.loc[df['class']==cl]
        p.circle(x=jitter('tar', 0.5), y='pred', size=8, alpha=0.1, color=palette[cl], legend=class_names[cl], source=df2 )
        p.line(x='tar', y='pred', line_width=2, alpha=0.5, color=palette[cl], source=df2.groupby(by="tar").mean())
    p.legend.location = "top_left"
    p.legend.click_policy="hide"
    output_file(bok_file, title="Ranking by exercise")
    save(p)
    pil_image2 = get_screenshot_as_png(p)
    neptune.send_image('rank_distances', pil_image2)

def train_model(model,epoch_start, epochs,lr,lr_decay):
    loss_tuple = []
    for epoch in range(epoch_start,epochs+1):

        if torch.cuda.is_available():
            model.cuda()
        print(f"Starting epoch {epoch}")
        st = time.time()
        epoch_loss_train = train_epoch(model)
        print(f"Epoch {epoch}: Training complete, beginning evaluation")
        epoch_loss_test, val_targets= evaluate_epoch(model,epoch)
        log_results(epoch, st, epoch_loss_train, epoch_loss_test,val_targets)
        # loss_tuple.append(pairings)
        lr *= lr_decay  

def train_epoch(model):
    epoch_loss_train = [] 
    model.train()  
    # train minibatches
    for X, rankings, _ in generator.next_batch_embeds():

        X = torch.from_numpy(X.astype('float32'))
        rankings = torch.from_numpy(rankings.astype('float32'))
        if torch.cuda.is_available():
            X = X.cuda()
            rankings = rankings.cuda()
        
        optimizer.zero_grad()
        preds = model(X)
        batch_loss = loss_fun(preds,rankings)
        neptune.send_metric('batch_loss', batch_loss)
        epoch_loss_train.append(batch_loss.detach().cpu().numpy()) 
        batch_loss.backward()
        optimizer.step()
    return epoch_loss_train

def evaluate_epoch(model,epoch):
    with torch.no_grad():
        model.eval()
        epoch_loss_test = []
        rankings_list = []
        pred_list = []
        cl_list = []
        targets = []
        for X, rankings_np,classes_np in generator.next_batch_valid():
            X = torch.from_numpy(X.astype('float32'))
            rankings = torch.from_numpy(rankings_np.astype('float32'))
            if torch.cuda.is_available():
                X = X.cuda()
                rankings = rankings.cuda()
            preds = model(X)
            batch_loss = loss_fun(preds,rankings)
            neptune.send_metric('Ranking_validation_loss', batch_loss)
            # pairings.append(np.concatenate([p.reshape(-1,4) for p in loss_fun.get_pairings()]))
            epoch_loss_test.append(batch_loss.detach().cpu().numpy())
            rankings_list.append(rankings.detach().cpu().numpy())
            pred_list.append(preds.detach().cpu().numpy())
            cl_list.append(classes_np)
        rankings = np.concatenate(rankings_list)
        preds = np.concatenate(pred_list)
        classes = np.concatenate(cl_list)
        bok_file = f"{METRICSPATH}/ranking_{epoch}.html"
        targets.append( (classes.squeeze(),rankings.squeeze(),preds.squeeze(),
            preds.squeeze()))
        if epoch%10==0:
            box_plot(rankings.squeeze(),preds.squeeze(),classes.squeeze(),bok_file,epoch)
    return epoch_loss_test, targets

def log_results(epoch, st, epoch_loss_train, epoch_loss_test,val_targets=None, pairings=None):  
    losses_train.append(epoch_loss_train)
    losses_test.append(epoch_loss_test)
    
    print(f'Time for epoch {epoch}: {(time.time()-st)//60}')
    print(f"Training Loss: {np.mean(epoch_loss_train)} , Validation Loss: {np.mean(epoch_loss_test)}")
    neptune.send_metric('training_loss', np.mean(epoch_loss_train))
    neptune.send_metric('val_loss', np.mean(epoch_loss_test))
    if epoch%50==0:
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses_test': losses_test,
                'validation_targets':val_targets
                }, os.path.join(CHECKPATH,f'regressor-{EXPERIMENT_NAME}-{epoch}.pth') )



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
#    top= ModdedStridedModel(in_joints, 3, out_joints, filter_widths, causal=True, dropout=0.25, channels=embedding_len, skip_res=False)
    top= ModdedTemporalModel(in_joints, 3, out_joints, filter_widths, causal=True, dropout=0.25, channels=embedding_len, skip_res=False)
    class_mod = nn.Conv1d( embedding_len, classes, 1)

    base, top, class_mod = load_model_weights(chk_filename,base,top,class_mod)
    model = nn.Sequential(OrderedDict([
          ('base', base) ,
          ('transform', StandardiseKeypoints(True,False)),
          ('embedding', HeadlessNet2(top)),
#          ('classifier', SplitModel(class_mod) )
        ]))
    return model 

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
ucf_file = os.path.join(DATAPOINT,'Keypoints','keypoints_rest.csv')
# non exercise class id to supress from ranking loss

model = build_model(pretrained, in_joints, in_dims, out_joints, filter_widths, causal, channels, embedding_len,classes)

epoch=1


receptive_field = model.base.receptive_field()
pad = (receptive_field - 1) 
causal_shift = pad

# initialise everything
actions, out_poses_2d,from_clip_flgs, return_idx, ratings, filenames_final, targets = load_keypoints(kpfile_1,kpfile_2, # todo
                                                                    rating_file,seed,None)
# action_ucf, poses_ucf, files_ucf = fetch_openpose_keypoints(ucf_file)
# action_ucf = [action if action!=8 else 6 for action in action_ucf]
# poses_ucf = [p for a,p in zip(action_ucf,poses_ucf) if a==6]
# files_ucf = [f for a,f in zip(action_ucf,files_ucf) if a==6]
# action_ucf = [action for action in action_ucf if action==6]
# filenames_ucf = [f'ucf_{x}' for x in files_ucf ]
# action_ucf, poses_ucf, filenames_ucf = action_ucf[:50], poses_ucf[:50], filenames_ucf[:50]
# ratings_ucf = [9] * len(action_ucf)

# targets += action_ucf
# out_poses_2d += poses_ucf
# ratings += ratings_ucf
# filenames_final += filenames_ucf

generator = SimpleSiameseGenerator(batch_size, targets, out_poses_2d,ratings,model,filenames_final, pad=pad,
                 causal_shift=causal_shift, test_split=split_ratio, random_seed=seed,just_emb=True) # todo

model = SimpleRegression([128,64,32])
loss_fun = nn.MSELoss()
optimizer = optim.RMSprop(model.parameters(),lr)
losses_train = []
losses_test = []

# run training
train_model(model,epoch,epochs,lr,lr_decay)
neptune.stop()