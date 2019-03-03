import sys
import os
sys.path.insert(0, f"{os.getcwd()}/Virtual_trainer")

from utils import *
import random
import time
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import argparse
from easygui import msgbox
from torch.utils.data import DataLoader, RandomSampler

from openpose.model import get_model
from openpose.preprocessing import rtpose_preprocess, vgg_preprocess
from openpose.loader import VideosDataset, EXC_DICT
from openpose.mpii_config import *
from openpose.postprocessing import *
from openpose.create_vp3d_input import delete_nans, rescale_keypoints

from models.semantic import NaiveBaselineModel, NaiveStridedModel
from dataloader import *
from simple_generators import SimpleSequenceGenerator
# from Virtual_trainer.pose3d_config import *
from instagram.exercise_scraper import ExerciseScraper

EXC_DICT = {
            0:'squat',
            1:'deadlift',
            2:'pushups',
            3:'pullups',
            4:'wallpushups',
            5:'lunges',
            6:'other',
            7:'cleanandjerk',
            8:'jumprope',
            9:'soccerjuggling',
            10:'taichi',
            11:'jumprope',
            12:'golfswing',
            13:'bodyweightsquats'
            
}

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS=1

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


CHECKPATH = 'Virtual_trainer/checkpoint'

# Data mountpoint
DATAPOINT = "Virtual_trainer/Data"

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


url = 'https://www.instagram.com/p/BucCac0h2_L/'
max_duration=6
def main(url,max_duration=6):
    if url.find('instagram.com/p/')>-1:
        print(url.split('/p/')[-1])
        url = ExerciseScraper.get_new_url(url.split('/p/')[-1].replace('/',''))
    print(url)   
    path = '/tmp/'
    weight_name = './openpose/weights/openpose_mpii_best.pth.tar'
    model = get_model('vgg19')     
    model.load_state_dict(torch.load(weight_name)['state_dict'])
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    model.float()
    dataset = VideosDataset.from_url(url,path,max_duration)
    
    clip_df = pd.DataFrame()
    
    with torch.no_grad():
        model.eval()
        batch = dataset[0]
        X_full,orig_images_full,_ = batch
        out = cv2.VideoWriter('deadlift_good_form.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (224,224))
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

            df_dict =  {"vid_nr":ix,"filename":url, "clip_id":0, "target":0, "num_keypoints":num_keypoints,"spotter":spotter}
            df_dict.update(final_keypoints)
            clip_df = clip_df.append(pd.DataFrame(df_dict,index=[0]),ignore_index=True)
            

            
            pic_key = picture_keypoints(personwiseKeypoints,keypoints_list,frameClone)
        #    plt.imshow(pic_key)
            cv2.imshow('window',cv2.resize(pic_key,(640,360)))
            out.write(pic_key)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        clip_df = interpolate(clip_df,interpolate_feet=False)
    out.release()
    clip_df =  delete_nans(clip_df)
    multiplier = round(800/224,2)
    clip_df =  rescale_keypoints(clip_df,multiplier)




    actions, poses = fetch_keypoints(clip_df)
    classes = 8
    pretrained_weights = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    eval_model = NaiveBaselineModel(in_joints, in_dims, out_joints, filter_widths, pretrained_weights, embedding_len, classes,
                                causal=True, dropout=0.25, channels=channels)
    receptive_field = eval_model.base_model.receptive_field()
    pad = (receptive_field - 1) 
    causal_shift = pad

    checkp = torch.load('/home/artursil/Documents/vt2/recipe1/checkpoint/model4-19.pth')
    # checkp = torch.load('/home/artursil/Documents/virtual_trainer/Virtual_trainer/checkpoint/model-6.pth')
    checkp['model_state_dict']
    eval_model.load_state_dict(checkp['model_state_dict'])


    # generator = SimpleSequenceGenerator(batch_size,actions,poses,pad=pad,causal_shift=causal_shift,test_split=split_ratio)
    with torch.no_grad():
        eval_model.eval() 
        if torch.cuda.is_available():
            eval_model = eval_model.cuda()
            # poses = poses.cuda() 
        poses = np.concatenate(poses)
        poses = np.pad(poses,((54,0),(0,0),(0,0)),'edge')
        poses = torch.Tensor(np.expand_dims(poses,axis=0)).cuda()
        pred = eval_model(poses)
        softmax = torch.nn.Softmax(1)
        pred= softmax(pred)
        pred = pred.detach().cpu().numpy().squeeze()
        preds = np.argmax(pred,axis=0)
        print(preds)
        values, counts = np.unique(preds,return_counts=True)
        print(values)
        print(counts)
        ind = np.argmax(counts)
        print(EXC_DICT[values[ind]])
        msgbox(f'Predicted exercise: {EXC_DICT[values[ind]]}','Result')
    # with torch.no_grad():
    #     eval_model.eval()
    #     epoch_loss_test = []
    #     preds = []
    #     preds2 = []
    #     for y_val, batch_2d in generator.next_validation():          
    #         poses = torch.from_numpy(batch_2d.astype('float32'))
    #         if torch.cuda.is_available():
    #             poses = poses.cuda()         
    #         pred = eval_model(poses)
    #         preds2.append(pred/sum(pred))
    #         pred = np.argmax(pred.detach().cpu().numpy())
    #         preds.append(pred)

    # values, counts = np.unique(preds,return_counts=True)
    # print(preds2)
    # print(values)
    # print(counts)
    # ind = np.argmax(counts)
    # print(EXC_DICT[values[ind]])
    # msgbox(f'Predicted exercise: {EXC_DICT[values[ind]]}','Result')


parser = argparse.ArgumentParser()
parser.add_argument('--url','-u', help='Url of video')
parser.add_argument('--max-duration',type=int,default=6, help='Max duration of a clip')

args = parser.parse_args()

if __name__ == "__main__":
    main(args.url,args.max_duration)
