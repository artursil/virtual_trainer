import os
import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from video_extract import VideoScraper
import torchvision.transforms as transforms
from hashlib import md5

# EXC_DICT = {
#             0:'squat',
#             1:'deadlift',
#             2:'pushups',
#             3:'pullups',
#             4:'wallpushups',
#             5:'lunges',
#             6:'squats',
#             7:'cleanandjerk',
#             8:'jumprope',
#             9:'soccerjuggling',
#             10:'taichi',
#             11:'jumprope',
#             12:'golfswing',
#             13:'bodyweightsquats'
#             
# }
EXC_DICT = {
            0:'squat',
            1:'deadlift',
            2:'benchpress',
            3:'pullup',
            4:'overheadpress',
            5:'lunge',
            6:'other',
            7:'cleanandjerk'
            
}

class VideosDataset(Dataset):

    def __len__(self):
        return len(self.metadata_df)-self.starting_point

    def __init__(self, path, exc_dict,resize=220,
                 transform=None,starting_point=0,single_video=False):
        
        self.path = path
        self.transform = transform
        self.exc_dict = exc_dict
        self.resize = resize
        self.starting_point = starting_point
        self.single_video = single_video
        if self.single_video==True:
            self.metadata_df = None
        else:    
            self.metadata_df = self.__get_metadata()


    def __get_metadata(self):
        reverse_dict = {v:k for k,v in EXC_DICT.items()}
        dirs = os.listdir(self.path)
        metadata_df = pd.DataFrame()
        for d in dirs:
            if d.lower() in list(reverse_dict.keys()):
                files = os.listdir(f"{self.path}/{d}/")
                for file in files:
                    row = {
                            "name":file,
                            "path":f"{self.path}/{d}/{file}",
                            "exercise":d,
                            "target":reverse_dict[d]
                    }
                    metadata_df= metadata_df.append(pd.DataFrame(row,index=[0]),ignore_index=True)

        return metadata_df

    def __load_video(self,index):
        filepath = self.metadata_df.iloc[index]["path"]
        exercise = self.metadata_df.iloc[index]["exercise"]
        target = self.metadata_df.iloc[index]["target"]
        video = VideoScraper.from_file(filepath,exercise)
        resize = transforms.Resize(self.resize)
        if self.transform is None:
            prep_imgs, orig_images = video.get_images(resize=resize,fps=30 ,vgg=False)
        else:
            prep_imgs, orig_images = video.get_images(resize=resize,fps=30 ,transform=self.transform)
        target_arr = np.array([target]*prep_imgs.shape[0])
        return prep_imgs, orig_images, target_arr

    def __load_single_video(self):
        filepath = self.path
        exercise = ""
        target = 999
        video = VideoScraper.from_file(filepath,exercise)
        resize = transforms.Resize(self.resize)
        prep_imgs, orig_images = video.get_images(resize=resize,fps=30 ,vgg=False)
        #TODO use normal and swapped images
        target_arr = np.array([target]*prep_imgs.shape[0])
        
        return prep_imgs, orig_images, target_arr



    def __get_file_name(self,index):
        return self.metadata_df.iloc[index]["name"]


    def load_images_and_targets(self,index):
        if self.single_video:
            image,orig_image, y = self.__load_single_video()    
        else:
            image,orig_image, y = self.__load_video(index)
        return image,orig_image, y 
        
    def __getitem__(self, index):
        index+=self.starting_point
        X,orig_image, y = self.load_images_and_targets(index)
        X = torch.tensor(X)
        if self.single_video:
            return X,orig_image, y
        filename = self.__get_file_name(index)

            
        return X,orig_image, y, filename

    @classmethod
    def from_url(cls,url,path,max_duration=6):
        #TODO Change if there is a way to chop video without downloading
        #TODO use tmp file for downloading video and delete it
        #TODO avi and mp4
        filename = md5(str(random.random).encode()).hexdigest()  
        video = VideoScraper(url,path,"from_urls",'mp4',video_type='other',filename=filename)
        video.download_video()
        duration = video.duration
        print(duration)
        print(video.filepath)
        if duration>max_duration:
            start = (duration-max_duration)//2
            video.clip_video(start,start+max_duration,replace=True)

        video_path = f"{path}videos/{video.exercise}/{video.filename}2.{video.quality}"
        return cls(video_path, exc_dict=None,resize=220,transform=None,starting_point=0,single_video =True)

    @classmethod
    def from_file(cls,path,exercise,max_duration=6,clip_video=False,resize=220):
        video = VideoScraper.from_file(path,exercise)
        duration = video.duration
        print(duration)
        print(video.filepath)
        if clip_video:
            if duration>max_duration:
                start = (duration-max_duration)//2
                video.clip_video(start,start+max_duration,replace=True)

        #video_path = f"{path}videos/{video.exercise}/{video.filename}2.{video.quality}"
        video_path=path
        return cls(video_path, exc_dict=None,resize=resize,transform=None,starting_point=0,single_video =True)

    @classmethod
    def single_video(cls,path):
        exc_dict=None
        return cls(path, exc_dict,resize=220,transform=None,starting_point=0,single_video =True)

    def collate_func(self, batch):
        pass
