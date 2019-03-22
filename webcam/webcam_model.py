import sys
sys.path.insert(0, "../../virtual_trainer")
sys.path.insert(0,
        "../../virtual_trainer/Virtual_trainer")


from utils import *
import random
import requests
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import argparse
from easygui import msgbox
from torch.utils.data import DataLoader, RandomSampler
from model_utils import  picture_keypoints, load_model_weights, build_model

from video_extract import VideoScraper
from openpose.model import get_model
from openpose.preprocessing import rtpose_preprocess, vgg_preprocess
from openpose.loader import VideosDataset, EXC_DICT
from openpose.mpii_config import *
from openpose.postprocessing import *
from openpose.create_vp3d_input import delete_nans, rescale_keypoints

from dataloader import *
from simple_generators import SimpleSequenceGenerator
from instagram.exercise_scraper import ExerciseScraper



DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS=1


CHECKPATH = '../../virtual_trainer/Virtual_trainer/checkpoint'

# Data mountpoint
DATAPOINT = "../../virtual_trainer/Virtual_trainer/Data"

# --- Datasets ---
# H36M Ground truths
h36m_file = os.path.join(DATAPOINT,'Keypoints','data_2d_h36m_gt.npz')



# --- Parameters ---
batch_size = 2048
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

#url = 'http://192.168.2.107:8080/shot.jpg' 
#    cv2.imshow('AndroidCam', img)
#
#    
#    if cv2.waitKey(25) & 0xFF == ord('q'):
#        cv2.destroyAllWindows()
#        break

class ModelClass(object):
    def __init__(self,model,index):
        self.model=model

        self.video = cv2.VideoCapture(index)

        self.clip_df = pd.DataFrame()

        self.kp_buff = None

    def __del__(self):
        self.video.release()

    def get_frame(self):
        success, image = self.video.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def get_keypoints(self):
        model = self.model
        with torch.no_grad():
            model.eval()
            ix=0
            while True:
                _, image = self.video.read()
                img, orig_img = VideoScraper.transform_single_frame(image,220)

                X = torch.Tensor(img).unsqueeze(0).to(DEVICE)
                img = orig_img
                
                predicted_outputs, _ = model(X)
        
                output1, output2 = predicted_outputs[-2], predicted_outputs[-1]
                output = output2.detach().cpu().numpy()

                frame = img.copy()
                frameWidth = frame.shape[1]
                frameHeight = frame.shape[0]
                after_shape = img.shape
                detected_keypoints = []
                keypoints_list = np.zeros((0,3))
                keypoint_id = 0

                for part in range(NPOINTS):
                    probMap = output[0,part,:,:]
                    probMap = cv2.resize(probMap, (frame.shape[1], frame.shape[0]))

                    keypoints = getKeypoints(probMap, THRESHOLD)
                    keypoints_with_id = []
                    for i in range(len(keypoints)):
                        keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                        keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                        keypoint_id += 1
                
                    detected_keypoints.append(keypoints_with_id)

                valid_pairs, invalid_pairs = getValidPairs(np.expand_dims(output1[0].detach().cpu().numpy(),axis=0),
                                                            detected_keypoints,frameWidth, frameHeight)
                personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs,keypoints_list)

                spotter=False
                if len(personwiseKeypoints)>1:
                    spotter = detect_spotters(personwiseKeypoints)

                if len(personwiseKeypoints)>1:
                    personwiseKeypoints = more_keypoints(personwiseKeypoints,6)

                if len(personwiseKeypoints)>1:
                    personwiseKeypoints = bigger_person(personwiseKeypoints,keypoints_list,frameWidth, frameHeight)

                if len(personwiseKeypoints)>1:
                    personwiseKeypoints = head_middle(personwiseKeypoints,keypoints_list,frameWidth, frameHeight)
                                
                if len(personwiseKeypoints)>0:
                    final_keypoints={}
                    num_keypoints = 0
                    for k,x in enumerate(personwiseKeypoints[0][:len(personwiseKeypoints[0])-1]):
                        final_keypoints[f"{k}_0"] = np.nan if x==-1 else keypoints_list[int(x)][:2][0]
                        final_keypoints[f"{k}_1"] = np.nan if x==-1 else keypoints_list[int(x)][:2][1]
                        if x!=-1:
                            num_keypoints+=1
                else:
                    final_keypoints={}       
                    for k in range(15):
                        final_keypoints[f"{k}_0"] = np.nan 
                        final_keypoints[f"{k}_1"] = np.nan 
                    num_keypoints = 0

                df_dict =  {"vid_nr":ix,"filename":'filename', "clip_id":0, "target":0, 
                            "num_keypoints":num_keypoints,"spotter":spotter}
                df_dict.update(final_keypoints)
                self.clip_df = self.clip_df.append(pd.DataFrame(df_dict,index=[0]),ignore_index=True)
                
                pic_key = picture_keypoints(personwiseKeypoints,keypoints_list,image,after_shape)

                _, jpeg = cv2.imencode('.jpg', pic_key)
                return  jpeg.tobytes()
        
    def vp3d_model(self):
        
        clip_df = interpolate(self.clip_df,interpolate_feet=False)
        
        clip_df =  delete_nans(clip_df)
        multiplier = round(800/224,2)
        clip_df =  rescale_keypoints(clip_df,multiplier)

        actions, poses = fetch_keypoints(clip_df)
        classes = 8

        chk_filename = os.path.join(CHECKPATH,"Recipe-2-epoch-19.pth")
        model = build_model(chk_filename, in_joints, in_dims, out_joints, filter_widths, True, channels, embedding_len,classes)
        
        pretrained = torch.load('../../virtual_trainer/Virtual_trainer/checkpoint/combinedlearning-2nd_part2-5.pth')
        model.load_state_dict(pretrained['model_state_dict'])

        with torch.no_grad():
            model.eval() 
            if torch.cuda.is_available():
                model = model.cuda()
                # poses = poses.cuda()
            try:
                poses = np.concatenate(poses)
            except ValueError:
                self.prediction = "No human detected"
                return self
            poses = np.pad(poses,((54,0),(0,0),(0,0)),'edge')
            poses = torch.Tensor(np.expand_dims(poses,axis=0)).cuda()
            # print(f'Poses shape: {poses.shape}')
            embeds, preds = model(poses)
            # print(f'Preds shape:{preds.shape}')
            # print(preds)

            #load 3d keypoints from transformer layer
            self.kp_buff = model.transform.get_3dkeys()

            softmax = torch.nn.Softmax(1)
            pred= softmax(preds)
            pred = pred.detach().cpu().numpy().squeeze()
            print(pred)
            preds = np.argmax(pred,axis=1)
            print(preds)
            values, counts = np.unique(preds,return_counts=True)
            # print(values)
            # print(counts)
            ind = np.argmax(counts)
            print(EXC_DICT[values[ind]])
            # msgbox(f'Predicted exercise: {EXC_DICT[values[ind]]}','Result')
            self.prediction = EXC_DICT[values[ind]]
            print(self.prediction)
            return self
