from utils import *

class VideoScraper:
    def __init__(self,url,base_path,exercise,quality="mp4", *args, **kwargs):
        self.url=url
        self.base_path = base_path
        self.quality = quality
        self.exercise = exercise
        self.video = pafy.new(url)
        self.stream = self.select_stream()
        self.filename = self.generate_filename()
        self.filepath = self.generate_filepath()
        self.clipped_path = self.clip_path()

    def download_video(self,quiet=True):
        stream = self.select_stream()
        filepath = f'{self.base_path}videos/{self.exercise}/'
        if os.path.isdir(filepath)==False:
            os.makedirs(filepath)
        if os.path.isfile(self.filepath):
            print('File already exists')
        else:
            stream.download(filepath,quiet)

    def clip_video(self,start_t,end_t):
        if os.path.isfile(self.filepath):
            return VideoFileClip(self.filepath).subclip(start_t,end_t)
        else:
            raise FileNotFoundError

    def clip_path(self):
        return self.filepath.replace('/videos/','/videos/clipped/')

    def save_clipped(self,start_t,end_t):
        my_clip = self.clip_video(start_t,end_t)
        if os.path.isdir("/".join(self.clipped_path.split("/")[:-1]))==False:
            os.makedirs("/".join(self.clipped_path.split("/")[:-1]))
        if os.path.isfile(self.clipped_path ):
            print("Clipped version of this file exists")
        else:
            my_clip.write_videofile(self.clipped_path)

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


    @property
    def title(self):
        return self.video.title

    @property
    def duration(self):
        return self.video.duration

    @property
    def streams(self):
        return self.video.streams

    @property
    def description(self):
        return self.video.description