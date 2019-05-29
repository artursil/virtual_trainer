"""
@author: Ahmad Kurdi
Data loaders for Virtual Trainer portfolio project.
DSR portfolio project with Artur Silicki
"""
import re
import random
import pandas as pd 
import numpy as np

from VideoPose3D.common.utils import deterministic_random
from itertools import zip_longest


def fetch_vp3d_keypoints(keypoint_file, subjects, action_list, class_val, subset=1):
    # fetch 2D keypoints from npz file

    keypoints = np.load(keypoint_file)['positions_2d'].item() # load keypoints
    actions = []
    out_poses_2d = []
    return_idx = []

    # traverse dataset and append return lists
    for subject in subjects:
        for action in keypoints[subject].keys():
            action_clean = re.sub(r'\s\d+$', '', action)
            if action_clean in action_list: #skip actions not in action_list
                poses_2d = keypoints[subject][action]
                for i in range(len(poses_2d)): # Iterate across cameras
                    out_poses_2d.append(poses_2d[i][::2]) #stride 2 to downsample framerate
                    actions.append(class_val)
                    return_idx.append(f"{subject}_{action}")

    # sample a subset if requested
    idx_final = []
    actions_final = []
    out_poses_2d_final = []
    max_frames = 180
    min_frames = 60
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_subclips = out_poses_2d[i].shape[0]//max_frames
            if out_poses_2d[i].shape[0]-max_frames*n_subclips>min_frames:
                n_subclips+=1
            for ns in range(n_subclips):
                out_poses_2d_final.append(out_poses_2d[i][max_frames*ns:max_frames*(ns+1),:,:])
                actions_final.append(actions[i])
                idx_final.append(return_idx[i])

    return actions_final, out_poses_2d_final, idx_final

def fetch_openpose_keypoints(keypoint_file):
    return_idx= []
    keypoints = pd.read_csv(keypoint_file)
    actions = []
    out_poses_2d = []
    vid_idx = keypoints['vid_nr'].unique()

     joint_order = ['16_0','16_1','11_0','11_1','12_0','12_1','13_0','13_1',
                     '8_0','8_1','9_0','9_1', '10_0','10_1','14_0','14_1',
                     '1_0','1_1','15_0','15_1','0_0','0_1','2_0','2_1', '3_0',
                     '3_1','4_0','4_1','5_0','5_1','6_0','6_1','7_0','7_1']

#    joint_order = ['16_0','16_1','8_0','8_1','9_0','9_1','10_0','10_1',
#                    '11_0','11_1','12_0','12_1', '13_0','13_1','14_0','14_1',
#                    '1_0','1_1','15_0','15_1','0_0','0_1','5_0','5_1', '6_0',
#                    '6_1','7_0','7_1','2_0','2_1','3_0','3_1','4_0','4_1']
    
    for idx in vid_idx:
                    
        action = int(keypoints.loc[keypoints['vid_nr']==idx]['target'].head(1))
        poses_2d = keypoints.loc[keypoints['vid_nr']==idx].sort_values(by="clip_id")[joint_order]
        poses_2d = np.reshape(poses_2d.values,(-1,17,2))
        if ~np.isnan(poses_2d).any():
            out_poses_2d.append(poses_2d)
            actions.append(action)
            return_idx.append(idx)

    return actions, out_poses_2d, return_idx

def fetch_keypoints(keypoints):
    actions = []
    out_poses_2d = []
    vid_idx = keypoints['vid_nr'].unique()

    joint_order = ['16_0','16_1','11_0','11_1','12_0','12_1','13_0','13_1',
                    '8_0','8_1','9_0','9_1', '10_0','10_1','14_0','14_1',
                    '1_0','1_1','15_0','15_1','0_0','0_1','2_0','2_1', '4_0',
                    '4_1','3_0','3_1','5_0','5_1','7_0','7_1','6_0','6_1']
    
    for idx in vid_idx:
        action = int(keypoints.loc[keypoints['vid_nr']==idx]['target'].head(1))
        poses_2d = keypoints.loc[keypoints['vid_nr']==idx].sort_values(by="clip_id")[joint_order]
        poses_2d = np.reshape(poses_2d.values,(-1,17,2))
        if ~np.isnan(poses_2d).any():
            out_poses_2d.append(poses_2d)
            actions.append(action)

    return actions, out_poses_2d

def balance_dataset(targets,seed):
    np.random.seed(seed)
    classes, counts = np.unique(targets,return_counts=True)
    sm_class = classes[counts.argmin()]
    smpl_size = counts.min()
    idx = np.where(targets == sm_class)[0]
    for cl in classes:
        if cl == sm_class: 
            continue
        ix_ = np.random.choice(np.where(targets == cl)[0],smpl_size,False)
        idx = np.concatenate((idx,ix_))
    return idx

def balance_dataset_recipe2(targets,seed):
    np.random.seed(seed)
    classes, counts = np.unique(targets,return_counts=True)
    sm_class = classes[counts.argmin()]
    smpl_size = counts.min()
    smpl_size = 300
    idx_all = np.array(range(len(targets)))
    idx = np.where(targets == sm_class)[0]
    for cl in range(len(classes)):
        if classes[cl] == sm_class: 
            continue
        elif counts[cl]<=300:
            ix_ = np.random.choice(np.where(targets == classes[cl])[0],counts[cl],False)
        else:
            ix_ = np.random.choice(np.where(targets == classes[cl])[0],smpl_size,False)
        idx = np.concatenate((idx,ix_))
    idx_test = np.array([x for x in idx_all if x not in list(idx)])
    return idx, idx_test

def first_nonemp(df,col1,col2):
    """first_nonemp
        select value from first not empty column

    """
    new_column = df[col1]
    new_column[df[col1].isnull()] = df[col2][df[col1].isnull()]
    return new_column

def get_rating(df,ratings):
    """get_rating
        Assign ratings from chunks to whole videos.

    :param df: DataFrame with keypoints
    :param ratings: DataFrame with sample of manually assigned ratings
    """
    df['video_name'] = ["_".join(x.split('_')[:-1]).replace('-','_') for x in
            df['filename']]   
    ratings = ratings.iloc[:,1:3]
    ratings.columns = ['filename','rating']
    ratings['rating'] =ratings['rating'] - 1 # Rated from 1-10 but 0-9 needed
    ratings['video_name'] = ["_".join(x.split('_')[:-1]).replace('-','_') for x in ratings['filename']]
    ratings_gb = ratings.groupby('video_name')['rating'].mean().reset_index()
    ratings_gb['rating_avg'] = np.floor(ratings_gb['rating'])
    
    df = df.merge(ratings[['filename','rating']],'left','filename')
    df = df.merge(ratings_gb[['video_name','rating_avg']],'left','video_name')
    df['rated'] = False
    df.loc[~df['rating'].isnull(),'rated'] = True 
    df['rating_final'] = first_nonemp(df,'rating','rating_avg')
    return df 

def sample_rating(df,seed):
    """sample_rating
        Given the distribution of ratings per each class assign rating to videos that weren't manually rated.

    :param df: DataFrame with keypoints and manually assigned ratings for some videos.
    :param seed: Seed used for sampling from distribution.
    """
    targets = df['target'].unique()
    np.random.seed(seed)
    vids_ratings = pd.DataFrame()
    for target in targets:
        kp_target = df[df['target']==target]
        kp_target_nn = kp_target[~kp_target['rating_final'].isnull()]
        kp_target_null = kp_target[kp_target['rating_final'].isnull()]
        
        kp_target_nn_vids= kp_target_nn.groupby('video_name')[['rating_final']].max().reset_index()
        values, counts = np.unique(kp_target_nn_vids['rating_final'],return_counts=True)
        for x in range(10):
            if x not in values:
                counts = np.insert(counts,x,0)
        counts_p = counts/sum(counts)
        kp_target_null_vids= pd.DataFrame(kp_target_null['video_name'].unique(),columns=['video_name'])
        kp_target_null_vids['rating_final'] = [np.random.choice(np.arange(0, 10), p=counts_p) for x in range(len(kp_target_null_vids))]
        
        vids_ratings = vids_ratings.append(kp_target_null_vids)
     
    vids_ratings.columns = ['video_name','rating_null']
    vids_ratings['from_distribution'] = True
    df = df.merge(vids_ratings,"left","video_name")
    df['rating_final'] = first_nonemp(df,'rating_final','rating_null')
    df.drop(['rating_null'],axis=1,inplace=True)
    return df
        
    



def fetch_s_df(keypoints_csv,keypoints_rest_csv,ratings_csv,seed):   
    """fetch_s_df
        Assign ratings to clips with keypoints.
    :keypoints_csv: Directory with keypoints for deadlift and squat
    :keypoints_rest_csv: Directory with keypoints for other classes.
    :ratings_csv: Directory with ratings.
    :seed: seed used in sample_rating
    """
    keypoints = pd.read_csv(keypoints_csv)
    keypoints_rest = pd.read_csv(keypoints_rest_csv)
    keypoints_all = keypoints.append(keypoints_rest).reset_index(drop=True)
    ratings = pd.read_csv(ratings_csv)
        
    keypoints_all = get_rating(keypoints_all,ratings)
    
    keypoints_all = sample_rating(keypoints_all,seed)
    keypoints = keypoints_all
    
    return keypoints
    
    
def fetch_s_keypoints(df,target=None,from_distribution=False):
    """fetch_s_keypoints
            Given target and DataFrame with keypoints return tuple for dataloader.

    :param df: DataFrame with keypoints and ratings both for all videos (Created by fetch_s_df).
    :param target: Class of exercise for which we want to return values.
    :param from_distribution: Flag if we want to use samples from rated videos (False) or rest of videos (True)
    """
    out_poses_2d = []
    actions = [] 
    return_idx = []
    from_clip_flgs = []
    ratings = []
    
    if target!=None:
        keypoints = df.loc[df['target']==target]
    else:
        keypoints = df
    
    joint_order = ['16_0','16_1','11_0','11_1','12_0','12_1','13_0','13_1',
                    '8_0','8_1','9_0','9_1', '10_0','10_1','14_0','14_1',
                    '1_0','1_1','15_0','15_1','0_0','0_1','2_0','2_1', '4_0',
                    '4_1','3_0','3_1','5_0','5_1','7_0','7_1','6_0','6_1']
    
    if from_distribution==False:
        keypoints = keypoints.loc[keypoints['from_distribution']!=True]
    else:
        keypoints = keypoints.loc[keypoints['from_distribution']==True]
        
    if len(keypoints)==0:
        raise ValueError(f'''There are no samples for this target ({target}) 
                            and given from_distribution={from_distribution}''')
    filenames = keypoints['filename'].unique()
    filenames_final = []
    targets = []
    for file in filenames:
        action = target
        poses_2d = keypoints.loc[keypoints['filename']==file].sort_values(by="clip_id")[joint_order]
        poses_2d = np.reshape(poses_2d.values,(-1,17,2))
        from_clip_flg = keypoints[keypoints['filename']==file]['rated'].iloc[0]
        rating  = keypoints[keypoints['filename']==file]['rating_final'].iloc[0]
        target  = keypoints[keypoints['filename']==file]['target'].iloc[0]
        if ~np.isnan(poses_2d).any():
            out_poses_2d.append(poses_2d)
            actions.append(action)
            from_clip_flgs.append(from_clip_flg)
            return_idx.append(file)
            ratings.append(rating)
            filenames_final.append(file)
            targets.append(target)

    return actions, out_poses_2d,from_clip_flgs, return_idx, ratings, filenames_final, targets
