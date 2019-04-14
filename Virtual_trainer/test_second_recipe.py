import os
import sys
import errno
import torch

import torch.nn as nn
import torch.optim as optim
import numpy as np

from models.semantic import BaselineWithkeypoints
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
#filter_widths2 = [7,7]
filter_widths2 = filter_widths

channels = 1024
in_joints, in_dims, out_joints = 17, 2, 17

# load dataset
action_op, poses_op, vid_idx = fetch_openpose_keypoints(instagram_file)
action_op2, poses_op2, vid_idx2 = fetch_openpose_keypoints(ucf_file)
action_op_test, poses_op_test, vid_idx_test = fetch_openpose_keypoints(ucf_file_test)
actions_op_itest, poses_op_itest, vid_idx_itest = fetch_openpose_keypoints(insta_test_file)

actions = action_op + action_op2
actions = [action if action!=8 else 6 for action in actions]
poses = poses_op + poses_op2
vid_idx = vid_idx + vid_idx2

# balance the dataset
seed = 1234
balanced, test_ids = balance_dataset_recipe2(np.array(actions),seed)
actions_test = [actions[x] for x in test_ids]
poses_test = [poses[x] for x in test_ids]
#actions_test = actions_test + action_op_test + actions_op_itest
#poses_test = poses_test + poses_op_test + poses_op_itest
actions_test = actions_op_itest
poses_test = poses_op_itestactions_op_itest, poses_op_itest, vid_idx_itest = fetch_openpose_keypoints(insta_test_file)
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

# actions = action_op + action_op2
# actions = [action if action!=8 else 6 for action in actions]
# poses = poses_op + poses_op2
# vid_idx = vid_idx + vid_idx2

# # balance the dataset
# seed = 1234
# balanced, test_ids = balance_dataset_recipe2(np.array(actions),seed)
# actions_test = [actions[x] for x in test_ids][:200]
# poses_test = [poses[x] for x in test_ids][:200]
# actions_test = actions_test + action_op_test + actions_op_itest
# poses_test = poses_test + poses_op_test + poses_op_itest
# # actions_test = actions_op_itest
# # poses_test = poses_op_itest
print(actions)

#balance squats and deadlifts
actions_test = np.array(actions_test)
poses_test = np.array(poses_test)
actions_ds = list(actions_test[actions_test==1][:100]) + list(actions_test[actions_test==0][:100])
poses_ds = list(poses_test[actions_test==1][:100]) + list(poses_test[actions_test==0][:100])


#
actions_test = actions_ds + action_op_test + actions_op_itest
poses_test = poses_ds + poses_op_test + poses_op_itest
#actions_test = actions_op_itest
#poses_test = poses_op_itest

# build models
classes = len(np.unique(actions))
pretrained_weights = torch.load(chk_filename, map_location=lambda storage, loc: storage)
eval_model = NaiveBaselineModel(in_joints, in_dims, out_joints, filter_widths, pretrained_weights, embedding_len, classes,
                                causal=True, dropout=0.25,
                                channels=channels,filter_widths2=filter_widths2)


loss_fun = nn.CrossEntropyLoss()


if torch.cuda.is_available():
    eval_model = eval_model.cuda()

# train model
epoch = 0
losses_train = []
losses_test = []
validation_targets = []
vp3d_keypoints = []

#checkp = torch.load('/home/artursil/Documents/vt2/recipe1/checkpoint/model4-19.pth')
checkp = torch.load('/home/artursil/Documents/virtual_trainer/Virtual_trainer/checkpoint/Recipe-2-epoch-19.pth')
#checkp = torch.load('/home/artursil/Documents/virtual_trainer/Virtual_trainer/checkpoint/Recipe-2-77-epoch-10.pth')
# checkp = torch.load('/home/artursil/Documents/virtual_trainer/Virtual_trainer/checkpoint/model-6.pth')
eval_model.load_state_dict(checkp['model_state_dict'])


#Load dataframe with addtional exercises for additional info
insta_test = pd.read_csv(insta_test_file)


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
    validation_targets = []
    pred_dict = {}
    for ix,poses in enumerate(poses_test):
        if ix%1000==0:
            print(ix)   
#        poses = np.concatenate(poses)
        poses = np.pad(poses,((54,0),(0,0),(0,0)),'edge')
        poses = torch.Tensor(np.expand_dims(poses,axis=0)).cuda()
        pred = eval_model(poses)
        val_pred = pred.clone()
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
        pred_copy = pred
        preds = np.argmax(pred,axis=0)
        
        values, counts = np.unique(preds,return_counts=True)
        ind = np.argmax(counts)
#        if sum(preds==7)/len(preds)>0.35 and values[ind] in [0,1]:
#            values[ind]=7
#        if sum(preds==5)/len(preds)>0.35 and values[ind] in [0,1]:
#            values[ind]=5
        accuracy = np.sum(values[ind]==actions)
#        print(accuracy)
#        if accuracy==0:
#            name = insta_test.loc[insta_test['vid_nr']==vid_idx_itest[ix],'filename'].iloc[0]
#            pred_dict[name] = [pred_copy,values[ind],orig_action]
#            print(name)
#            print(f'Predicted class: {values[ind]}')
#            print(f'Actual class: {orig_action}')
##            print(pred_copy)
##            print(preds)
        validation_targets.append((act,val_pred.detach().cpu().numpy()))
        accuracy_test.append((orig_action,np.sum(values[ind]==actions)))
        vp3d_keypoints.append(seq_keypoints)
    print(np.mean([x[1] for x in accuracy_test]))
#predss = pred_dict['183_adri-c-utgard-halter-on-instagram-si-tienes-guardi.mp4']
#tens = predss[0]
#preds = np.argmax(tens,axis=0)
#preds
##tens_n = tens.detach().cpu().numpy()
#          
torch.save({
        'losses_test': test_losses,
        'test_targets' : targets,
        'targets_softmax': targets_softmax,
        'accuracy' : accuracy_test,
        'validation_targets' : validation_targets,
        }, os.path.join(CHECKPATH,f'Recipe-2-test-results.pth') )
   
