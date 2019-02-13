import youtube_dl
import time
import cv2
import numpy as np
from PIL import Image
from utils import *
from openpose.preprocessing import rtpose_preprocess, vgg_preprocess, crop_with_factor

class VideoScraper:
    def __init__(self,url,base_path,exercise,quality="mp4",video_type="youtube", *args, **kwargs):
        self.video_type = video_type
        self.url=self.__get_url(url)
        self.base_path = base_path
        self.quality = quality
        self.exercise = exercise
        self.error = False
        if self.video_type=="youtube":
            try:
                self.video = pafy.new(self.url)
            except OSError:
                self.error = True
                self.filename = None
            else:               
                self.stream = self.select_stream()
                self.filename = self.generate_filename()
        else:
            self.filename = kwargs.get("filename")
        if kwargs.get("filepath") is None:
            self.filepath = self.generate_filepath()
        else:
            self.filepath = kwargs.get("filepath")
        self.clipped_path = self.clip_path()

    def download_video(self,quiet=True):
        try:
            if self.video_type=="youtube":
                self.__download_yt_video(quiet)
            else:
                self.__download_other_video()
        except:
            print(f"Problem with this url: {self.url}")
            return False
        else:
            return True

    def __download_yt_video(self,quiet=True):
        stream = self.select_stream()
        filepath = f'{self.base_path}videos/{self.exercise}/'
        if os.path.isdir(filepath)==False:
            os.makedirs(filepath)
        if os.path.isfile(self.filepath):
            print('File already exists')
        else:
            stream.download(filepath,quiet)

    def __download_other_video(self):
        ydl_opts = {'outtmpl': f"{self.base_path}videos/{self.exercise}/{self.filename}.%(ext)s",
                    'nooverwrites':True,
                    'quiet':True}
        if os.path.isfile(self.filepath):
            print('File already exists')
        else:
            with youtube_dl.YoutubeDL(ydl_opts) as ydl:
                ydl.download([self.url, ])
                    # raise ValueError(f"Problem with this url: {self.url}")
            self.__delete_audio_files()

    def __delete_audio_files(self):
        filepath = f'{self.base_path}videos/{self.exercise}/'
        files = os.listdir(filepath)
        audio_files = [x for x in files if x.lower().find("audio")>-1]
        for audio_file in audio_files:
            os.remove(f'{filepath}{audio_file}')

    def __get_url(self,url):
        if self.video_type=="youtube":
            _url = url.replace("%3D","=").replace("%3F","?")
            _url = re.sub(r"(?<=com/).+(?=watch)","",_url)
            url = _url.split("%")[0]
        return url

    def clip_video(self,start_t,end_t):
        if os.path.isfile(f"{self.filepath}.mp4"):
            return VideoFileClip(f"{self.filepath}.mp4").subclip(start_t,end_t)
        else:
            raise FileNotFoundError(f"{self.filepath}")

    def __duration_other(self):
        file_with_ext = f"{self.filepath}.mp4"
        if os.path.isfile(file_with_ext):          
            vid = VideoFileClip(file_with_ext)
            duration = vid.duration
            vid.close()
            # try:
            #     r =  VideoFileClip(file_with_ext).duration
            # except:
            #     print(file_with_ext)
            #     r =  VideoFileClip(file_with_ext).duration
            return duration
        else:
            raise FileNotFoundError(f"{file_with_ext}")     

    def clip_path(self):
        return self.filepath.replace('/videos/','/videos/clipped/')

    def save_clipped(self,start_t,end_t,n_clip):
        my_clip = self.clip_video(start_t,end_t)
        if os.path.isdir("/".join(self.clipped_path.split("/")[:-1]))==False:
            os.makedirs("/".join(self.clipped_path.split("/")[:-1]))
        if os.path.isfile(self.clipped_path):
            print("Clipped version of this file exists")
        else:
            my_clip.write_videofile(f"{self.clipped_path}_{n_clip}.mp4")
            my_clip.close()

    def generate_filepath(self):
        return f'{self.base_path}videos/{self.exercise}/{self.filename}'

    def remove_video_file(self):
        filepath = self.filepath
        os.remove(filepath)
        print(f"{self.filename}.{self.quality} Removed!")

    def select_stream(self):
        streams = self.video.streams
        stream = None
        poss_qual = []
        for s in streams:
            if str(s).find(self.quality)>-1:
                stream = s
                break
            else:
                poss_qual.append(str(s).split(":")[1].split("@")[0])
        if stream==None:
            poss_qual = list(set(poss_qual))
            raise ValueError(f"Can't find '{self.quality}' quality for this video: {self.url} \n Please select one from {poss_qual}") 
        return stream

    def generate_filename(self):
        return self.stream.generate_filename()

    def gen_img_seq(self,fps=30):
        if os.path.isfile(self.clipped_path ):
            video = VideoFileClip(self.clipped_path)
        else:
            raise FileNotFoundError("Clipped version of this file doesn't exist")      
        split_path = self.filepath.split("/")
        base = "/".join(split_path[:-1])
        base = base.replace("videos","images")
        folder_name = slugify("_".join(split_path[-1].split(" ")[:4]))
        images_path = f"{base}/{folder_name}/"
        if os.path.isdir(images_path)==False:
            os.makedirs(images_path) 
        video.write_images_sequence(f'{images_path}img%03d.png')
        video.close()

    @classmethod
    def from_file(cls,file_path,exercise):
        filepath = file_path.replace(".mp4","")
        filename = filepath.split("/")[-1].replace(".mp4","")
        return cls(url="",base_path="",exercise=exercise,quality="mp4",
                    video_type="other",filepath=filepath,filename=filename)

    def get_images(self,resize,fps=30,transform=None,vgg=True):
        vidcap = cv2.VideoCapture(f"{self.filepath}.{self.quality}")
        success,image = vidcap.read()
        x=0
        n_imgs = int(60*self.duration)
        # images = np.zeros([n_imgs,*image.shape])
        if vgg==True:
            preprocess = vgg_preprocess
        else:
            preprocess = rtpose_preprocess
        orig_images = []
        prep_imgs= []
        while success: 
            image = np.array(resize(Image.fromarray(image)))
            image,_,_ = crop_with_factor(image) 
            # print(type(image))
            if transform is None:  
                orig_image = image            
                image = preprocess(image)
                
            else:
                image = np.array(transform(Image.fromarray(image)))
                orig_image = image
                image = preprocess(image)
            # print("cropping")
             
            orig_images.append(orig_image)
            prep_imgs.append(image)
            x+=1
            success,image = vidcap.read()
        orig_images = np.array(orig_images)
        prep_imgs = np.array(prep_imgs)
        # images = images[:x-1,:,:,:]
        if x/self.duration-fps>2:
            orig_images = orig_images[::2,:,:,:]
            prep_imgs = prep_imgs[::2,:,:,:]
        return prep_imgs, orig_images


    @property
    def title(self):
        return self.video.title

    @property
    def duration(self):
        if self.video_type=="youtube":
            return self.video.duration
        else:
            return self.__duration_other()

    @property
    def streams(self):
        return self.video.streams

    @property
    def description(self):
        return self.video.description