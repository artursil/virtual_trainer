import os
import sys
import errno
import time
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from models.semantic import NaiveStridedModel,NaiveBaselineModel, HeadlessNet
from models.loss import ContrastiveLoss
from dataloader import *
from simple_generators import SimpleSequenceGenerator
from siamese_generator import SiameseGenerator
import gc

try:
    # Create checkpoint directory if it does not exist
    os.makedirs('checkpoint')
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', 'checkpoint')
CHECKPATH = 'checkpoint'

# Data mountpoint
DATAPOINT = "Data"

def create_stats_dict(bi,stats_dict,infos,embed1,embed2,batch_loss,diffs,target):
    stats_dict[bi] = []

    batch_loss=batch_loss.detach().cpu().numpy()
    diffs = diffs.detach().cpu().numpy()
    #file1, file2, frame1, frame2, embedding1, embedding2, batch_loss, difference, class
    if embed1 is not None:
        embed1=embed1.detach().cpu().numpy()
        embed2=embed2.detach().cpu().numpy()
        for x in range(len(infos)):
            stats_tuple = (*infos[x],embed1[x,:],embed2[x,:],float(batch_loss),int(diffs[x]),target)
            stats_dict[bi].append(stats_tuple)
    else:
        for x in range(len(infos)):
            stats_tuple = (*infos[x],float(batch_loss),int(diffs[x]),target)
            stats_dict[bi].append(stats_tuple)
    return stats_dict
        

def train_epoch():
    epoch_loss_train = [] 
    model.train()  
    # train minibatches
    num_batches = generator.num_batches
    batch_i=0
    stats_dict = {}
    for X1, X2, diffs, infos in generator.next_batch():
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
        if epoch%20==0:
            stats_dict = create_stats_dict(batch_i,stats_dict,infos,None,None,batch_loss,diffs,selected_class)
        # print('{{"metric": "Batch Loss", "value": {}}}'.format(batch_loss))
        epoch_loss_train.append(batch_loss.detach().cpu().numpy()) 
        batch_loss.backward()
        optimizer.step()  
        batch_i+=1
    return epoch_loss_train, stats_dict

def evaluate_epoch():
    with torch.no_grad():
        model.eval()
        epoch_loss_test = []
        stats_dict = {}
        batch_i=0
        for X1, X2, diffs, infos in generator.next_validation():
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
            if epoch%20==0:
                stats_dict = create_stats_dict(batch_i,stats_dict,infos,embed1,embed2,batch_loss,diffs,selected_class)
            epoch_loss_test.append(batch_loss.detach().cpu().numpy())   
            batch_i+=1
    return epoch_loss_test, stats_dict 

def log_results(epoch, st, epoch_loss_train, epoch_loss_test,loss_tuple, stats_train, stats_val):  
#    losses_train.append(epoch_loss_train)
#    losses_test.append(epoch_loss_test)
    print(f'Time per epoch: {(time.time()-st)}')
    print('{{"metric": "Training Loss", "value": {}, "epoch": {}}}'.format(
            np.mean(epoch_loss_train), epoch)) 
    print('{{"metric": "Validation Loss", "value": {}, "epoch": {}}}'.format(
            np.mean(epoch_loss_test), epoch))
    if epoch%20==0:
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses_test': epoch_loss_test,
#                'loss_tuple': loss_tuple
                }, #os.path.join(CHECKPATH,f'siamese-recipe2-{selected_class}-{epoch}.pth') 
#                f'/media/artursil/DATA/checkpoints/siamese-recipe2-{selected_class}-{epoch}.pth'
                f'/media/artursil/DATA/checkpoints/siamese-recipe3-all-{epoch}.pth'
                
                )
        torch.save({
                'epoch': epoch,
                'stats_train': stats_train,
                'stats_val': stats_val,
                }, #os.path.join(CHECKPATH,f'siamese-recipe2-{selected_class}-{epoch}.pth') 
#                f'/media/artursil/DATA/checkpoints/siamese-recipe2-stats-{selected_class}-{epoch}.pth'
                f'/media/artursil/DATA/checkpoints/siamese-recipe3-stats-all-{epoch}.pth'
                )


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
    model = NaiveBaselineModel(in_joints, in_dims, out_joints, filter_widths, {}, embedding_len, classes,
                                                                                causal=True, dropout=0.25, channels=channels, loadBase=False)
    return HeadlessNet(model, pretrained_weights)



# --- params ---
pretrained = os.path.join(CHECKPATH,"Recipe-2-epoch-19.pth") # pretrained base
kpfile_1 = os.path.join(DATAPOINT,"Keypoints","keypoints.csv") # squats and dealifts
kpfile_2 = os.path.join(DATAPOINT,"Keypoints", "keypoints_rest.csv") # rest of classes
rating_file = os.path.join(DATAPOINT,"clips-rated.csv") # rating labels file
#selected_class = 0 # Squat
selected_class = 1 # Deadlift
loss_margin = 0.5
rate_range = 3
seed = 1234
lr, lr_decay = 0.01 , 0.90 
split_ratio = 0.2
epochs = 100
batch_size = 32
n_chunks = 8

model = load_headless_model(pretrained)
model.cuda()
receptive_field = model.embed_model.base_model.receptive_field()
pad = (receptive_field - 1) 
causal_shift = pad

# initialise everything
actions, out_poses_2d,from_clip_flgs, return_idx, ratings, filenames, targets = load_keypoints(kpfile_1,kpfile_2,
                                                                    rating_file,seed,None)
from faster_generator import SiameseSimpleGenerator
generator = SiameseSimpleGenerator(batch_size, targets, out_poses_2d,ratings,filenames,model, from_clip_flgs, pad=pad,
                 causal_shift=causal_shift,
                 test_split=split_ratio,n_chunks=n_chunks,
                 random_seed=seed,only_rated=False,oversample=True)

# loss_fun = ContrastiveLoss(rate_range,loss_margin)
loss_fun = ContrastiveLoss(loss_margin)
losses_train = []
losses_test = []

class LinearModel(nn.Module):

    def __init__(self, input_dim, output_dim):

        super().__init__() 
        self.linear = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(output_dim,output_dim)

    def forward(self, x):
        out = self.linear2(F.relu(self.linear(x)))
        return out   

model = LinearModel(128,64)
model.cuda()
optimizer = optim.Adam(model.parameters(),lr, amsgrad=True)


# run training
for epoch in range(1,epochs+1):
    st = time.time()
    stats_train_l = []
    stats_val_l = []
    epoch_loss_train,stats_train = train_epoch()
    epoch_loss_test, stats_val = evaluate_epoch()
#    stats_train_l.append(stats_train)
#    stats_val_l.append(stats_val)
#    loss_tuple = loss_fun.get_tuples()
    loss_tuple = []
    log_results(epoch, st, epoch_loss_train, epoch_loss_test,loss_tuple, stats_train, stats_val)
    lr *= lr_decay
    gc.collect()
    for param_group in optimizer.param_groups:
        param_group['lr'] *= lr_decay


#dict1 = torch.load('/media/artursil/DATA/checkpoints/siamese-recipe2-stats-all-1.pth')
#
#stats_train = dict1['stats_train']
#
#batch0 = stats_train[0]
#
#stats_val= dict1['stats_val']
#
#batch0_val = stats_val[0]
