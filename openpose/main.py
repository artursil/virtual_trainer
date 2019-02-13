from utils import *
import random
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, RandomSampler
from openpose.model import get_model
from openpose.preprocessing import rtpose_preprocess, vgg_preprocess
from openpose.loader import VideosDataset, EXC_DICT
from openpose.mpii_config import *
from openpose.postprocessing import *


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PATH = "D:/Documents_D/data_science/win/virtual_trainer/instagram/videos/clipped"
EPOCHS=1



def main():
    weight_name = './openpose/weights/openpose_mpii_best.pth.tar'
    model = get_model('vgg19')     
    model.load_state_dict(torch.load(weight_name)['state_dict'])
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.float()
    
    dataset = VideosDataset(PATH,EXC_DICT,200,transform=None)
    X,orig_images,y = dataset[255]
    dl = DataLoader(dataset, batch_size=1,sampler=None)

    outputs_df = pd.DataFrame()
    st=time.time()
    with torch.no_grad():
        model.eval()
        for ix,batch in enumerate(dl):
            if ix==1:
                break
            if ix%50==0:
                print(f"Processed video no: {ix+1}")
            X,orig_images_full, y = batch

            X_full = X.squeeze(0).to(DEVICE)
            y = y.to(DEVICE).detach().cpu().numpy()
            bs = X_full.shape[0]
            third_bs = bs//3
            orig_images_full.squeeze_(0)
            frame_counter = -1
            clip_df = pd.DataFrame()
            rnd_images = [random.randint(0,X_full.shape[0]-1) for x in  range(3)]
            for b in range(3):
                if b==0:
                    X = X_full[:third_bs,:,:,:]
                    orig_images = orig_images_full.squeeze(0)[:third_bs,:,:,:]
                elif b==1:
                    X = X_full[third_bs:2*third_bs,:,:,:]
                    orig_images = orig_images_full[third_bs:2*third_bs,:,:,:]
                else:
                    X = X_full[2*third_bs:,:,:,:]
                    orig_images = orig_images_full[2*third_bs:,:,:,:]          

                predicted_outputs, _ = model(X)
        
                output1, output2 = predicted_outputs[-2], predicted_outputs[-1]
                output = output2.detach().cpu().numpy()
        
                
                for j in range(X.shape[0]):
                    img = orig_images[j].detach().cpu().numpy()
                    frame = img.copy()
                    frame_counter+=1
                    frameWidth = frame.shape[1]
                    frameHeight = frame.shape[0]
        
                    detected_keypoints = []
                    keypoints_list = np.zeros((0,3))
                    keypoint_id = 0
        
                    for part in range(NPOINTS):
                        probMap = output[j,part,:,:]
                        probMap = cv2.resize(probMap, (frame.shape[1], frame.shape[0]))

                        keypoints = getKeypoints(probMap, THRESHOLD)
                        keypoints_with_id = []
                        for i in range(len(keypoints)):
                            keypoints_with_id.append(keypoints[i] + (keypoint_id,))
                            keypoints_list = np.vstack([keypoints_list, keypoints[i]])
                            keypoint_id += 1
                    
                        detected_keypoints.append(keypoints_with_id)
        
                    frameClone = frame.copy()
        
                    valid_pairs, invalid_pairs = getValidPairs(np.expand_dims(output1[j].detach().cpu().numpy(),axis=0),detected_keypoints,frameWidth, frameHeight)
                    personwiseKeypoints = getPersonwiseKeypoints(valid_pairs, invalid_pairs,keypoints_list)

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
        
                    df_dict =  {"vid_nr":ix, "clip_id":frame_counter, "target":y[0][j], "num_keypoints":num_keypoints}
                    df_dict.update(final_keypoints)
                    clip_df = clip_df.append(pd.DataFrame(df_dict,index=[0]),ignore_index=True)
                    
                    
                    if frame_counter in rnd_images:
                        save_picture(f"{PATH.replace('clipped','processed')}/test{ix}_{frame_counter}.png",personwiseKeypoints,keypoints_list,frameClone)
        
            clip_df = interpolate(clip_df)
            for f in rnd_images:
                frame_clone = orig_images_full[f].detach().cpu().numpy()
                draw_interpolated(PATH,ix,f,clip_df,frame_clone) 

            outputs_df = outputs_df.append(clip_df,ignore_index=True)
            if (ix+1) % 100==0:
                outputs_df.to_csv(f"{PATH.replace('clipped','processed')}/outputs_df_{ix}.csv")
            if ix%50==0:
                print(f"Time: {time.time()-st}")
    outputs_df.to_csv(f"{PATH.replace('clipped','processed')}/outputs_df.csv")