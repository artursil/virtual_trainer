
import sys
sys.path.insert(0, "../../virtual_trainer")
sys.path.insert(0,
        "../../virtual_trainer/Virtual_trainer")


from multiprocessing import Event, Queue, Process
#from torch.multiprocessing import Process
#from multiprocessing.queues import Queue


from utils import *
import random
import requests
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import argparse
from torch.utils.data import DataLoader, RandomSampler
from model_utils import  picture_keypoints, load_model_weights, build_model
#from models.semantic import SimpleRegression

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
from VideoPose3D.common.model import  TemporalModel  


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS=1


CHECKPATH = '../../virtual_trainer/Virtual_trainer/checkpoint'

# Data mountpoint
DATAPOINT = "../../virtual_trainer/Virtual_trainer/Data"


#chk_filename = os.path.join(DATAPOINT,'BaseModels', 'epoch_45.bin')
# model architecture
filter_widths = [3,3,3]
channels = 1024
in_joints, in_dims, out_joints = 17, 2, 17
embedding_len = 128
classes = 8



class ModelClass(Process):
    def __init__(self,index, q, img_q, start_gen):
        super().__init__() 
        self.q = q
        self.img_q = img_q
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        weight_name = '../../virtual_trainer/openpose/weights/openpose_mpii_best.pth.tar'
        model = get_model('vgg19')     
        model.load_state_dict(torch.load(weight_name)['state_dict'])
        model.to(self.device)
        #model = torch.nn.DataParallel(model)
        model.float()
        
        self.model=model
        self.video = cv2.VideoCapture(index)

        self.clip_df_tmp = pd.DataFrame()
        self.clip_df = pd.DataFrame()

        self.kp_buff = None
        self.new_pred = False

        self.gen = start_gen
        self.flip = None
       

    def run(self):
        print("started")
        while True:
            self.img_q.put (self.get_frame())

    def __del__(self):
        self.video.release()

    def get_frame(self):
        if self.gen.is_set():
            print("gen is set")
            if self.flip is None:
                self.flip = True
            self.flip = False if self.flip else True
        if self.flip:    
            return self.get_keypoints()
        success, image = self.video.read()
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()

    def get_keypoints(self):
        model = self.model
        with torch.no_grad():
            model.eval()
            ix=0
            frame_counter=0
            
            _, image = self.video.read()
            img, orig_img = VideoScraper.transform_single_frame(image,220)

            X = torch.Tensor(img).unsqueeze(0).to(self.device)

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
            self.clip_df_tmp = self.clip_df_tmp.append(pd.DataFrame(df_dict,index=[0]),ignore_index=True)
            
            if frame_counter%30==0:
                if len(self.clip_df)<90:
                    self.clip_df = self.clip_df.append(self.clip_df_tmp)
                else:
                    self.clip_df = self.clip_df.iloc[30:,].append(self.clip_df_tmp)
                self.q.put(self.clip_df)
                #print("call recipe2")
                
            

            pic_key = picture_keypoints(personwiseKeypoints,keypoints_list,image,after_shape)

            _, jpeg = cv2.imencode('.jpg', pic_key)
            return jpeg.tobytes()

  

class PredModel:
    def __init__(self,index,img_q, start_gen):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.e = Event()
        self.q = Queue()
        self.img_q = img_q
        
        self.opModel = ModelClass(index, self.q, self.img_q, start_gen)
        print("start")
        self.opModel.start()
        
        self.new_pred = False
        self.prediction = None
        self.rating = None
        chk_filename = os.path.join(CHECKPATH,"Recipe-2-epoch-19.pth")
        self.classModel = build_model(chk_filename, in_joints, in_dims, out_joints, filter_widths, True, channels, embedding_len,classes)
        
        #model_rank =  SimpleRegression([128,64,32])
        #chk_filename = os.path.join(CHECKPATH,"regressor-simple-regressor-3-300.pth")
        #model_rank.load_state_dict[chk_filename['model_state_dict']]
        
        self.rankModel = None # model_rank 


    def vp3d_predict(self,clip_df):
        
        clip_df = interpolate(self.clip_df,interpolate_feet=False)
        
        clip_df =  delete_nans(clip_df)
        multiplier = round(800/224,2)
        clip_df =  rescale_keypoints(clip_df,multiplier)

        actions, poses = fetch_keypoints(clip_df)
        
        model = self.classModel
        #model_rank = self.rankModel
        
        with torch.no_grad():
            model.eval() 
            model.to(self.device)
                #model_rank = self.rankModel.eval().to(self.device)
                # poses = poses.cuda()
            try:
                poses = np.concatenate(poses)
            except ValueError:
                self.prediction = "No human detected"
                return self
            poses = np.pad(poses,((54,0),(0,0),(0,0)),'edge')
            poses = torch.Tensor(np.expand_dims(poses,axis=0), device=self.device)
            # print(f'Poses shape: {poses.shape}')
            embeds, preds = model(poses)
            softmax = torch.nn.Softmax(1)
            pred= softmax(preds)
            pred = pred.detach().cpu().numpy().squeeze()
            preds = np.argmax(pred,axis=1)
            values, counts = np.unique(preds,return_counts=True)
            ind = np.argmax(counts)
            self.prediction = EXC_DICT[values[ind]]
            
            #ratings=model_rank(embeds).detach().detach().cpu().numpy()
            self.rating = 5 # np.mean(ratings)
        return self

    def get_prediction(self):
        return json.dumps({'action':self.prediction,'rating':self.rating})

   
    def runPredict(self):
        
        while True:
            vp3d_predict(self.q.get())
            self.e.set()
            

