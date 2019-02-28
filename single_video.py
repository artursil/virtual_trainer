from utils import *
import random
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import argparse
from torch.utils.data import DataLoader, RandomSampler
from openpose.model import get_model
from openpose.preprocessing import rtpose_preprocess, vgg_preprocess
from openpose.loader import VideosDataset, EXC_DICT
from openpose.mpii_config import *
from openpose.postprocessing import *


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# PATH = "D:/Documents_D/data_science/win/virtual_trainer/instagram/videos/clipped"
EPOCHS=1

# url = "https://scontent-frx5-1.cdninstagram.com/vp/97a7213d8cff329cb014a25ef749352c/5C7A4222/t50.2886-16/53383901_2135602079834457_7199611926907715584_n.mp4?_nc_ht=scontent-frx5-1.cdninstagram.com"


def picture_keypoints(personwiseKeypoints,keypoints_list,frameClone):
    for i in range(14):
        for n in range(len(personwiseKeypoints)):
            index = personwiseKeypoints[n][np.array(POSE_PAIRS[i])]
            if -1 in index:
                continue
            B = np.int32(keypoints_list[index.astype(int), 0])
            A = np.int32(keypoints_list[index.astype(int), 1])
            cv2.line(frameClone, (B[0], A[0]), (B[1], A[1]), [102,255,102], 3, cv2.LINE_AA)        
    return frameClone


def main(url):
    path = '/tmp/'
    weight_name = './openpose/weights/openpose_mpii_best.pth.tar'
    model = get_model('vgg19')     
    model.load_state_dict(torch.load(weight_name)['state_dict'])
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.float()
    dataset = VideosDataset.from_url(url,path)
    
    clip_df = pd.DataFrame()
    
    with torch.no_grad():
        model.eval()
        batch = dataset[0]
        X_full,orig_images_full,_ = batch
        for ix in range(X_full.shape[0]):
            X = X_full[ix,:,:,:].unsqueeze(0).to(DEVICE)
            img = orig_images_full[ix,:,:,:]
            
            predicted_outputs, _ = model(X)
    
            output1, output2 = predicted_outputs[-2], predicted_outputs[-1]
            output = output2.detach().cpu().numpy()

            frame = img.copy()
            frameWidth = frame.shape[1]
            frameHeight = frame.shape[0]

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

            frameClone = frame.copy()

            valid_pairs, invalid_pairs = getValidPairs(np.expand_dims(output1[0].detach().cpu().numpy(),axis=0),detected_keypoints,frameWidth, frameHeight)
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

            df_dict =  {"vid_nr":ix,"filename":url, "clip_id":0, "target":None, "num_keypoints":num_keypoints,"spotter":spotter}
            df_dict.update(final_keypoints)
            clip_df = clip_df.append(pd.DataFrame(df_dict,index=[0]),ignore_index=True)
            

            
            pic_key = picture_keypoints(personwiseKeypoints,keypoints_list,frameClone)
#            plt.imshow(pic_key)
            cv2.imshow('window',cv2.resize(pic_key,(640,360)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        clip_df = interpolate(clip_df,interpolate_feet=False)

parser = argparse.ArgumentParser()
parser.add_argument('--url','-u', help='Url of video')

args = parser.parse_args()

if __name__ == "__main__":
    main(args.url)
            