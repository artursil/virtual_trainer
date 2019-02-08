import numpy as np
from instagram.create_metadata import create_metadata
from utils import *
from video_extract import VideoScraper

class ExerciseScraper():
    def __init__(self,path,exercise_type):
        self.path = path
        self.exercise = exercise_type
        self.__read_features()
        self.__get_txt_path()

    def __read_config(self):
        json_file_path = f'{self.path}/config_file.json'
        with open(json_file_path, encoding="utf8") as json_file:  
            config_json = json.load(json_file)
        return config_json

    def __save_config(self,json_file):
        json_file_path = f'{self.path}/config_file.json'
        with open(json_file_path, "w") as write_file:
            json.dump(json_file, write_file) 

    def __read_features(self):
        config_json = self.__read_config()
        config_exc = config_json[self.exercise] 
        self.max_kg = config_exc["max_kg"]
        self.max_reps = config_exc["max_reps"]
        self.avg_rep = config_exc["avg_rep"]
        return self
        
    def __get_txt_path(self):
        txt_files_path=f"{self.path}/txt_files/"
        if not os.path.isdir(txt_files_path):
            os.mkdir(txt_files_path)
        self.txt_files_path = txt_files_path
        return self

    def create_metadata(self,hashtag="",max_n_posts=50000):
        if hashtag=="":
            hashtag=self.exercise
        try:
            create_metadata(self.txt_files_path,self.exercise,max_n_posts)
        except RunTimeError:
            print("RunTimeError")

    def __merge_metadata(self):
        files = os.listdir(self.txt_files_path)
        exercise_files = [file for file in files if file.find(self.exercise)>-1 and file.find("json")>-1]
        json_list = []
        if len(exercise_files)==0:
            raise FileNotFoundError("Firstly download metadata file using create_metadata func")
        else:
            for f in exercise_files:
                print(f"Using: {f} file")
        for file in exercise_files:
            file_path = f"{self.txt_files_path}/{file}"
            with open(file_path, encoding="utf8") as json_file:  
                _data = json.load(json_file)
            json_list += _data
        return json_list


    def __get_stop_words(path,exercise):
        _data = self.__read_config()
        exercises_d = _data['exercises']
        exercises = list(chain(*[v for k,v in exercises_d.items() if k!=self.exercise]))
        stop_words = _data['stop_words']
        return exercises, stop_words

    def __filter_row(self,row,ix,stop_words,other_exercises,maximum_tags):
        insta_dict = {}
        is_video = row['is_video'] 
        if is_video==False:
            return insta_dict
        vid = row['id']
        text_container = row['edge_media_to_caption']['edges'] 
        if text_container!=[]:
            text = text_container[0]['node']['text'].replace(":","")
        else:
            text=""
        lang = detect_lang(text)
        tags = [tag.lower() for tag in row.get('tags',[])]
        text_good = analyze_text(text,stop_words+other_exercises)
        len_urls = len(row['urls'])
        len_tags_g = 0<len(tags)<maximum_tags
        tags_good = sum([tag in stop_words+other_exercises for tag in tags])==0
        crawl = lang=="en" and text!="" and len_urls==1 and text_good and tags_good and len_tags_g           
        if crawl:
            m = search_reg(text)
            if m:
                reps_n,weight,mu = get_reps(m,text)
                if weight==None:
                    weight, mu = get_weight(text)
                good_weight = check_max_weight(weight,mu,self.max_kg)
                filename = f"{ix}_{slugify(text)[:50]}"
                if good_weight:
                    insta_dict = {'id':vid,
                                'shortcode':'shortcode',
                                'text':text,
                                'tags':str(tags),
                                'video_url':row['urls'][0],
                                'video_view_count':row['video_view_count'],
                                'likes':row['edge_liked_by']['count'],
                                'exercise':self.exercise,
                                'reps':reps_n,
                                'weight':f"{weight} {mu}",
                                'filename':filename                               
                                }
        return insta_dict


    def fitler_posts(self,max_tags=15):
        data = self.__merge_metadata()
        max_kg = self.max_kg
        other_exercises, stop_words = get_stop_words(f"{self.path}",self.exercise)
        maximum_tags = max_tags
        insta_df = pd.DataFrame()
        ix=0
        for idx,d in enumerate(data):
            if idx%1000==0:
                print(idx)
            insta_dict = self.__filter_row(d,ix,stop_words,other_exercises,maximum_tags)
            if bool(insta_dict):
                ix+=1
                insta_df = insta_df.append(pd.DataFrame(insta_dict,index=[0]),ignore_index=True)
        insta_df.drop_duplicates("shortcode",inplace=True)
        insta_df.to_csv(f"{self.path}/txt_files/{self.exercise}_filtered_df.csv",index=False)     
        return insta_df
    
    @staticmethod
    def append_df(df,vid,filename,filepath,title,duration,n_reps,exercise,text):
        meta_dict={ 'vid':vid, #added for isntagram maybe also useful for reddit
                    'filename':filename,
                    'filepath':filepath,
                    'title':title,
                    'duration':duration,
                    'n_reps':n_reps,
                    'exercise':exercise,
                    "full_text":text # added
                    }
        return df.append(pd.DataFrame(meta_dict,index=[0]),ignore_index=True)

    def __read_download_files(self):
        return pd.read_csv(f"{self.path}/txt_files/{self.exercise}_filtered_df.csv")

    def download_videos(self):
        df = self.__read_download_files()
        down_data_df = pd.DataFrame()
        for ix,row in df.iterrows():
            if ix % 10==0:
                print(ix) 
            vid = row['id']
            url = row['video_url']
            text = slugify(row["text"]).replace("-"," ")
            title = slugify(row['text'])[:50]
            filename = row['filename']
            video = VideoScraper(url,self.path,self.exercise,video_type="other",filename=filename)
            if video.error==False:
                success = video.download_video()
                if success:
                    duration = video.duration
                    down_data_df = self.append_df(down_data_df,vid,video.filename,video.filepath,title,\
                                            duration,row['reps'],self.exercise,text)
        down_data_df.to_csv(f"{self.path}/txt_files/{self.exercise}_dl_files.csv",index=False)

    def __filter_downloaded(self,df):
        files = os.listdir(f"{self.path}/videos/{self.exercise}/")
        files = [file.split(".")[0] for file in files]
        df["not_in_files"] = [file not in files for file in df["filename"]]
        print(sum(df["not_in_files"]))
        df["too_long"] = df['duration']>=50
        df["t_per_rep"] = df['duration']/df['n_reps']
        df["t_short_rep"] = df["t_per_rep"]<=0.6
        df["reps_t_long"] = df["t_per_rep"]>=50
        df["t_many_reps"] = df['n_reps']>self.max_reps
        zipped = zip(df["too_long"],df["reps_t_long"],df["t_short_rep"],df["t_many_reps"],df["not_in_files"])
        df["cond1"] = [any(z) and ~nif for *z,nif in zipped]
        df["cond1"] = df["cond1"]==-1
        zipped = zip(df["too_long"],df["reps_t_long"],df["t_short_rep"],df["t_many_reps"],df["not_in_files"])
        df["cond2"] = [any(z) for z in zipped]
        return df

    def __remove_vids(self,df):
        df_trun = df.loc[df["cond1"]]
        print("vids")
        print(df_trun["filepath"].values)
        for vid in df_trun["filepath"].values:
            try:
                # os.remove(f"{vid}.mp4")
                print(f"Should remove {vid}.mp4")
            except FileNotFoundError as e:
                print(e)

    def __get_vids_to_drop(self,df):
        df_trun = df.loc[df["cond2"]]
        ind_to_drop = list(df_trun.index)
        vids_to_exclude = list(df_trun["vid"].values)
        return ind_to_drop, vids_to_exclude

    def update_filtered(self):
        """[This method compares list of videos in downloaded_files.csv to a list of physical videos.
            If there are videos that were deleted, downloaded_files is updated and ids of those videos are passed to config file]
        
        
        """
        print("Waiting for user to delete videos")
        cleaning_done = input("Are you ready to continue [y/n]")
        if cleaning_done.upper()=="N":
            return

        down_df = pd.read_csv(f"{self.path}/txt_files/{self.exercise}_dl_files.csv")
        print(f"Number of videos before filtering: {len(down_df)}")
        down_df = down_df.reset_index(drop=True)
        config_json = self.__read_config()
        vids_to_exclude = config_json[self.exercise]["ids_to_exclude"]
        ind_to_drop = []

        f_down_df = self.__filter_downloaded(down_df)
        print("removing")
        self.__remove_vids(f_down_df)
        i2d, vid2exc = self.__get_vids_to_drop(f_down_df)
        ind_to_drop += i2d
        print(vids_to_exclude)
        print("vid2exc")
        print(vid2exc)
        vids_to_exclude += vid2exc
                
        config_json[self.exercise]["ids_to_exclude"] = list(set(vids_to_exclude))
        down_df = f_down_df.drop(["cond1","cond2","not_in_files","too_long","reps_t_long","t_many_reps"],axis=1)
        down_df.drop(ind_to_drop,axis=0,inplace=True)
        print(f"Number of videos after filtering: {len(down_df)}")
        down_df.to_csv(f"{self.path}/txt_files/{self.exercise}_dl_files.csv",index=False)
        config_json[self.exercise]["ids_to_exclude"] = list(set([int(vid) for vid in vids_to_exclude]))
        print(config_json)
        self.__save_config(config_json)

    @staticmethod
    def clipped_list(dur,dur2,n_reps,tpr):
        tpr = round(tpr,1)
        start = (dur-dur2)//10*6
        start_rep = start
        clip_list = []
        for rep in range(n_reps):
            end = start_rep+tpr
            if end>dur:
                end=dur
            clip_list.append((start_rep,end))
            start_rep = start_rep+tpr
        return(clip_list)



    def __cr_clipped_intervals(self):
        #TODO
        df= pd.read_csv(f"{self.path}/txt_files/{self.exercise}_dl_files.csv")
        df['t_per_rep2'] = [min(tpr,self.avg_rep) for tpr in df['t_per_rep']]
        df['duration2'] = np.floor(df['n_reps']*df['t_per_rep2'])
        zipped = zip(df['duration'],df['duration2'],df['n_reps'],df['t_per_rep2'])
        df['clips'] = [self.clipped_list(*z) for z in zipped]
        # df.to_csv(f"{self.path}/txt_files/{self.exercise}_clip_files.csv")
        return df
    
    def clip_videos(self):
        videos = self.__cr_clipped_intervals() 
        for _,row in videos.iterrows():
            video = VideoScraper("",self.path,self.exercise,video_type="other",filename=row['filename'])
            for ix,clip in enumerate(row["clips"]):
                video.save_clipped(clip[0],clip[1],ix)
