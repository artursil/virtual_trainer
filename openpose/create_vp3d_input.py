import os 
import pandas as pd
import numpy as np
import random
path = "/home/artursil/Downloads/processed_other/"


EXC_DICT = {
            0:'squat',
            1:'deadlift',
            2:'pushups',
            3:'pullups',
            4:'wallpushups',
            5:'lunges',
            6:'squats',
            7:'cleanandjerk',
            8:'jumprope',
            9:'soccerjuggling',
            10:'taichi',
            11:'jumprope',
            12:'golfswing',
            13:'bodyweightsquats'
            
}

#Head – 0, Neck – 1, Right Shoulder – 2, Right Elbow – 3, Right Wrist – 4,
#Left Shoulder – 5, Left Elbow – 6, Left Wrist – 7, Right Hip – 8,
#Right Knee – 9, Right Ankle – 10, Left Hip – 11, Left Knee – 12,
#Left Ankle – 13, Chest – 14, Background – 15- Point between neck and head
# 16 - Point between hips

def drop_incomplete(input_df):
    df = input_df.copy()
    df.drop_duplicates(["filename","clip_id"],inplace=True)
    nk_avgs = df.groupby("vid_nr").num_keypoints.mean()
    vids = list(nk_avgs[nk_avgs>=13].index)
    df = df.loc[df["vid_nr"].isin(vids)]
    return df

def drop_spotters(input_df):
    df = input_df.copy()
    df["vid_nr_str"] = [str(x) for x in df["vid_nr"]]
    df["new_id"] = df["filename"] + "_" + df["vid_nr_str"]
    spotter_avgs = df.groupby("new_id").spotter.mean()
    spotter = list(spotter_avgs[spotter_avgs>=0.5].index)
    df = df.loc[~df["new_id"].isin(spotter)]
    return df

def drop_bad_interpolation(input_df):
    df = input_df.copy()
    df2 = input_df.copy()
    df["l130_diff"] = np.abs(df["13_0"] - df["13_0_orig"])
    df["l131_diff"] = np.abs(df["13_1"] - df["13_1_orig"])
    df["l100_diff"] = np.abs(df["10_0"] - df["10_0_orig"])
    df["l101_diff"] = np.abs(df["10_1"] - df["10_1_orig"])
    df_diff = df.groupby(["new_id"])["l130_diff","l131_diff","l100_diff","l101_diff"].mean()
    df2 = df2[~df2['new_id'].isin(df_diff.query('(l130_diff >15) or (l131_diff >15) or (l100_diff >15) or (l101_diff >15)').index.tolist())]
    return df2
    

def delete_nans(input_df):
    df = input_df.copy()
    df = df.loc[df.isnull().sum(axis=1)==0,]
    return df



def rescale_keypoints(input_df, multiplier):
    df = input_df.copy()
    for joint in range(17):
        for coord in range(2):
            df[f'{joint}_{coord}'] = df[f'{joint}_{coord}']*multiplier + 100
            df[f'{joint}_{coord}'] = df[f'{joint}_{coord}'].round(0).astype(int)
            
    return df

def create_exc_other(input_df,list_of_exc_classes):
    df = input_df.copy()
    exc_set = df.loc[df['target'].isin(list_of_exc_classes)]
    exc_files = exc_set['filename'].unique()
    other_set = df.loc[~df['target'].isin(list_of_exc_classes)]
    if len(other_set)>0:
        other_files = list(other_set['filename'].unique())
        sampled_other = random.sample(other_files,len(exc_files)//2)
        other_set = other_set.loc[other_set['filename'].isin(sampled_other)]
        other_set['target'] = 8
        exc_set = exc_set.append(other_set)
    return exc_set



out_files = os.listdir(path)
out_files = [file for file in out_files if file.find("df")>-1]
output_df = pd.DataFrame()

for file in out_files:
    df = pd.read_csv(f"{path}/{file}")
    output_df = output_df.append(df)
    output_df.drop_duplicates(inplace=True)


output_df = drop_incomplete(output_df)
output_df = drop_spotters(output_df)
output_df = drop_bad_interpolation(output_df)
output_df = delete_nans(output_df)
multiplier = round(800/224,2)
output_df = rescale_keypoints(output_df,multiplier)
list_of_exc_classes = [0,1,2,3,4,5,7]
output_df = create_exc_other(output_df,list_of_exc_classes)


output_df.to_csv('/home/artursil/Documents/vt2/recipe1/Data/Keypoints/keypoints_rest2.csv')



