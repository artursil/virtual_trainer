import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from video_extract import VideoScraper
import torchvision.transforms as transforms

EXC_DICT = {
            0:'squat',
            1:'deadlift'
}

class VideosDataset(Dataset):

    def __len__(self):
        return len(self.metadata_df)

    def __init__(self, path, exc_dict,resize=220,
                 transform=None):
        
        self.path = path
        self.transform = transform
        self.exc_dict = exc_dict
        self.resize = resize
        self.metadata_df = self.__get_metadata()


    def __get_metadata(self):
        reverse_dict = {v:k for k,v in EXC_DICT.items()}
        dirs = os.listdir(self.path)
        metadata_df = pd.DataFrame()
        for d in dirs:
            if d in list(reverse_dict.keys()):
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





    def load_images_and_targets(self,index):

        image,orig_image, y = self.__load_video(index)

        
        return image,orig_image, y 
        
    def __getitem__(self, index):
        X,orig_image, y = self.load_images_and_targets(index)
        X = torch.tensor(X)

            
        return X,orig_image, y

    def collate_func(self, batch):
        pass