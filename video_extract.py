from utils import *
import youtube_dl

class VideoScraper:
    def __init__(self,url,base_path,exercise,quality="mp4",video_type="youtube", *args, **kwargs):
        self.video_type = video_type
        self.url=self.__get_url(url)
        self.base_path = base_path
        self.quality = quality
        self.exercise = exercise
        if self.video_type=="youtube":
            self.video = pafy.new(self.url)
            self.stream = self.select_stream()
            self.filename = self.generate_filename()
        else:
            self.filename = kwargs.get("filename")
        
        self.filepath = self.generate_filepath()
        self.clipped_path = self.clip_path()

    def download_video(self,quiet=True):
        if self.video_type=="youtube":
            self.__download_yt_video(quiet)
        else:
            self.__download_other_video()

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
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            ydl.download([self.url, ])
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