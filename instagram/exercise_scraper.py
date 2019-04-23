import numpy as np
from slugify import slugify
from instagram.create_metadata import create_metadata
from utils import *
from video_extract import VideoScraper
import requests
from bs4 import BeautifulSoup

class ExerciseScraper():
    def __init__(self,path,exercise_type,read_features=True):
        self.path = path
        self.exercise = exercise_type
        self.__get_txt_path()
        if read_features:
            self.__read_features()

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
        except RuntimeError:
            print("RunTimeError")

    @staticmethod
    def create_json_sample(filepath,sample_size=1000):
        tmp_json = pd.read_json(filepath)
        tmp_json.iloc[:sample_size].to_json(f'{filepath.replace(".json","")}_sample.json')

    def __read_df_json(self,filelist,samples=False):
        full_files = [f for f in filelist if f.find('sample')==-1] 
        if samples==True:
            sample_files = [f'{x.replace(".json","")}_sample.json' for x in full_files] 
            for sample in sample_files:
                if sample not in filelist:
                    import pdb; pdb.set_trace() 
                    filepath = f"{self.txt_files_path}/{sample.replace('_sample','')}"
                    self.create_json_sample(filepath,20000)
            filelist=sample_files
        else:
            filelist=full_files
        cat_df = pd.DataFrame()
        for file in filelist:
            file_path = f"{self.txt_files_path}/{file}"
            tmp_df = pd.read_json(file_path)
            cat_df = cat_df.append(tmp_df)
            del tmp_df
        cat_df.drop_duplicates(['shortcode'],inplace=True)
        return cat_df 

    def __merge_metadata(self,read_df=False,use_sample=False):
        files = os.listdir(self.txt_files_path)
        exercise_files = [file for file in files if file.find(self.exercise)>-1 and file.find("json")>-1]
        json_list = []
        if len(exercise_files)==0:
            raise FileNotFoundError("Firstly download metadata file using create_metadata func")
        else:
            for f in exercise_files:
                print(f"Using: {f} file")
        if read_df==True:
            return self.__read_df_json(exercise_files,samples=use_sample) 
        for file in exercise_files:
            file_path = f"{self.txt_files_path}/{file}"
            with open(file_path, encoding="utf8") as json_file:  
                _data = json.load(json_file)
            json_list += _data
        return json_list


    def __get_stop_words(self,path,exercise):
        _data = self.__read_config()
        exercises_d = _data['exercises']
        exercises = list(chain(*[v for k,v in exercises_d.items() if k!=self.exercise]))
        stop_words = _data['stop_words']
        return exercises, stop_words

    def __filter_row(self,row,ix,stop_words,other_exercises,maximum_tags):
        if isinstance(row,pd.Series):
            row = row.to_dict()
        if isinstance(row['tags'],list)==False:
            row['tags'] = []
        config_json = self.__read_config()
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
        if crawl and row['shortcode'] not in config_json[self.exercise]['shortcodes_to_exclude']:
            m = search_reg(text)
            if m:
                reps_n,weight,mu = get_reps(m,text)
                if weight==None:
                    weight, mu = get_weight(text)
                good_weight = check_max_weight(weight,mu,self.max_kg)
                filename = f"{ix}_{slugify(text)[:50]}"
                if good_weight:
                    insta_dict = {'id':vid,
                                'shortcode':row['shortcode'],
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

    def __filter_df(self,df,stop_words,other_exercises,maximum_tags):
        config_json = self.__read_config()

        print(f'Number of posts before filtering: {len(df)}')

        videos = sum(df['is_video'])
        print(f'Number of videos: {videos}')

        df['text'] = df['edge_media_to_caption'].apply(get_text)
        df['lang'] = df['text'].apply(detect_lang)
        en_posts= sum(df['lang']=='en')
        print(f'Number of english posts: {en_posts}')

        df['text_good'] = [analyze_text(x,stop_words+other_exercises) for x in 
                df['text']]
        good_posts = sum(df['text_good'])
        print(f'Number of posts with good description: {good_posts}')

        df['len_urls'] = [len(x)==1 for x in df['urls']]
        len_urls = sum(df['len_urls'])
        print(f'Number of posts with one url: {len_urls}')

        df['tags'] = [[] if isinstance(tags,list)==False else [x.lower() for x
            in tags] for tags in df['tags']]
        df['len_tags_g'] = [0<len(x)<maximum_tags for x in df['tags']]
        len_tags_g = sum(df['len_tags_g'])
        print(f'Number of posts with tags: {len_tags_g}')

        df['tags_good'] = [sum([tag in stop_words+other_exercises for tag in 
            tags])==0 for tags in df['tags']]
        tags_good = sum(df['tags_good'])
        print(f'Number of posts with only good tags: {tags_good}')

        df['crawl'] = False
        df.loc[(df['lang']=="en") & (df['text']!="") & (df['len_urls']==1) & 
                (df['text_good']==True) & (df['tags_good']==True) & 
                (df['len_tags_g']==True),'crawl']=True
        crawl = sum(df['crawl'])
        print(f'Number of posts after filters mentioned above: {crawl}')

        df = df.loc[(df['crawl']==True) & (~df['shortcode'] \
            .isin(config_json[self.exercise]['shortcodes_to_exclude']))]

        df['m'] = df['text'].apply(search_reg)
        df['m_flg'] = [x==None for x in df['m']]
        df_m = df[df['m_flg']==False]
        del df
        print(f'Number of posts with info about number of reps: {len(df_m)}')

        df_m['reps_n'], df_m['weight'], df_m['mu'] = zip(*[get_reps(m,text) 
            for m,text in zip(df_m['m'],df_m['text'])])
        df_m['weight_flg'] = [x==None for x in df_m['weight']]
        df_m.loc[df_m['weight_flg']==True,'weight'],df_m \
             .loc[df_m['weight_flg']==True,'mu'] = zip(*[get_weight(text) for 
              text in df_m.loc[df_m['weight_flg']==True,'text']])
        df_m['good_weight'] = [check_max_weight(weight,mu,self.max_kg) for 
                weight,mu in zip(df_m['weight'], df_m['mu'])]
        df_m['tags'] = [str(x) for x in df_m['tags']]
        df_m['video_url'] = [x[0] for x in df_m['urls']]
        df_m['weight'] = [f'{x} {y}' for x,y in zip(df_m['weight'],df_m['mu'])]
        df_m['likes'] = [x['count'] for x in df_m['edge_liked_by']]
        df_m['reps'] = df_m['reps_n']

        df_m['exercise'] = self.exercise
        df_m2 = df_m.loc[df_m['good_weight']==True,['id','shortcode','text',
            'tags','video_url','video_view_count','likes',
            'exercise','reps','weight']]
        df_m2['filename'] = [f"{ix}_{slugify(text)[:50]}" for ix,text in 
                enumerate(df_m2['text'])]
        print(f'Final number of posts: {len(df_m2)}')

        return df_m2

    def fitler_posts(self,append,max_tags=15,read_df=False):
        data = self.__merge_metadata(read_df=read_df)
        max_kg = self.max_kg
        other_exercises, stop_words = get_stop_words(f"{self.path}",self.exercise)
        maximum_tags = max_tags
        if append==True:
            insta_df=pd.read_csv(f"{self.path}/txt_files/{self.exercise}_filtered_df.csv")
        else:
            insta_df = pd.DataFrame()
        ix=0
        if read_df:
            insta_df = self.__filter_df(data,stop_words, other_exercises,
                    maximum_tags)
        else:
            for idx,d in enumerate(data):
                if idx%10000==0:
                    print(idx)
                insta_dict = self.__filter_row(d,ix,stop_words,other_exercises,maximum_tags)
                if bool(insta_dict):
                    ix+=1
                    insta_df = insta_df.append(pd.DataFrame(insta_dict,index=[0]),ignore_index=True)
        insta_df.drop_duplicates("shortcode",inplace=True)
        insta_df.to_csv(f"{self.path}/txt_files/{self.exercise}_filtered_df.csv",index=False)     
        save2db(self.path,insta_df,f'{self.exercise}_filtered_df')
        return insta_df
    
    @staticmethod
    def append_df(df,vid,filename,filepath,title,duration,n_reps,shortcode,exercise,text):
        meta_dict={ 'vid':vid, #added for isntagram maybe also useful for reddit
                    'filename':filename,
                    'filepath':filepath,
                    'title':title,
                    'duration':duration,
                    'n_reps':n_reps,
                    'shortcode':shortcode,
                    'exercise':exercise,
                    "full_text":text # added
                    }
        return df.append(pd.DataFrame(meta_dict,index=[0]),ignore_index=True)

    def __read_download_files(self):
        return pd.read_csv(f"{self.path}/txt_files/{self.exercise}_filtered_df.csv")

    def update_urls(self):
        df = self.__read_download_files()
        df_new = df.copy()
        df_new.reset_index(drop=True,inplace=True)
        for ix,row in df.iterrows():
            new_url = self.get_new_url(shortcode=row["shortcode"])
            df_new.loc[ix,"video_url"]= new_url
            assert df_new.iloc[ix]["video_url"]==new_url
        df_new.to_csv(f"{self.path}/txt_files/{self.exercise}_filtered_df2.csv")
        save2db(self.path,df_new,f'{self.exercise}_filtered_df2')


    @staticmethod
    def get_new_url(shortcode):
        headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) \
                     AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.87 Safari/537.36'}
        shortcode_url = f"https://www.instagram.com/p/{shortcode}"
        page = requests.get(shortcode_url,headers=headers)
        soup = BeautifulSoup(page.content,'html.parser')
        if str(soup.title).lower().find("page not found")>-1:
            print(f"Page not found for shortcode: {shortcode}")
            new_url=""
        else:
            new_url = soup.find('meta', attrs={'property': 'og:video'}).get('content')
            if new_url=="":
                print(f"Wasn't able to get new url for shortcode: {shortcode}")
        return new_url

    @classmethod
    def from_file(cls,path,exercise):
        return cls(path,exercise,False)

    def save_filter_csv(self,url,ix):
        filepath = f"{self.path}/txt_files/{self.exercise}_filtered_df.csv"
        if os.path.isfile(filepath):
            insta_df=pd.read_csv(filepath)
        else:
            insta_df = pd.DataFrame()

        headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) \
                     AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3396.87 Safari/537.36'}

        page = requests.get(url,headers=headers)
        soup = BeautifulSoup(page.content,'html.parser')

        vid = soup.find('meta',{'property':'al:ios:url'}).get('content').split('=')[1]
        shortcode = url.split('/p/')[1].replace('/','')
        text = soup.title.text.replace('\n','')
        tags = []
        tags_list = soup.findAll('meta',{'property':'video:tag'})
        for tag in tags_list:
            tags.append(tag.get('content'))
        video_url = soup.find('meta', attrs={'property': 'og:video'}).get('content')
        exercise = self.exercise
        filename = f"{ix}_{slugify(text)[:50]}"

        insta_dict = {'id':vid,
            'shortcode':shortcode,
            'text':text,
            'tags':str(tags),
            'video_url':video_url,
            'video_view_count':0,
            'likes':0,
            'exercise':self.exercise,
            'reps':0,
            'weight':0,
            'filename':filename                               
            }
        insta_df = insta_df.append(pd.DataFrame(insta_dict,index=[0]),ignore_index=True)
        insta_df.to_csv(filepath,index=False) 
        save2db(self.path,insta_df,f'{self.exercise}_filtered_df')

    def __create_path(self,path):
        if os.path.isdir(path):
            pass
        else:
            os.makedirs(path,exist_ok=True)

    def download_videos(self,append):

        print(f'{self.path}videos/{self.exercise}')
        self.__create_path(f'{self.path}videos/{self.exercise}')
        self.__delete_duplicates()
        df = self.__read_download_files()
        if append==True:
            down_data_df = pd.read_csv(f"{self.path}/txt_files/{self.exercise}_dl_files.csv")
        else:
            down_data_df = pd.DataFrame()     
        for ix,row in df.iterrows():
            if ix % 10==0:
                print(ix) 
            vid = row['id']
            url = row['video_url']
            text = slugify(row["text"]).replace("-"," ")
            title = slugify(row['text'])[:50]
            filename = row['filename']
            shortcode = row['shortcode']
            if f"{title}.mp4" in self.downloaded_files_df["files2"]:
                print(f"{title} already downloaded")
                continue
            if type(url)!=str:
                continue
            video = VideoScraper(url,self.path,self.exercise,video_type="other",filename=filename)
            if video.error==False:
                success = video.download_video()
                if success:
                    try:
                        duration = video.duration
                    except FileNotFoundError as e:
                        print(e)
                        continue
                    down_data_df = self.append_df(down_data_df,vid,video.filename,video.filepath,title,\
                                            duration,row['reps'],shortcode,self.exercise,text)
        down_data_df.to_csv(f"{self.path}/txt_files/{self.exercise}_dl_files.csv",index=False)
        save2db(self.path,down_data_df,f'{self.exercise}_dl_files')

    
    def __delete_duplicates(self):
        files_path = f"{self.path}/videos/{self.exercise}/"
        files = os.listdir(files_path)
        files = [x for x in files if x.find("mp4")>-1]
        nrs = [int(x.split("_")[0]) for x in files]
        files2 = [x.split("_")[1] for x in files]

        df = pd.DataFrame({"files":files,"nrs":nrs,"files2":files2})

        df.sort_values(["files2","nrs"],inplace=True)
        df["shifted"] = df["files2"].shift()
        df["to_delete"] = df["files2"]==df["shifted"]

        to_del = df[df["to_delete"]==True]["files"].tolist()

        for file in to_del:
            os.remove(f"{files_path}{file}")
            print(f"Deleting {file}")
        self.downloaded_files_df = df[df["to_delete"]==False]
        return self

    def __delete_additional_videos(self,down_data_df):
        files_path = f"{self.path}/videos/{self.exercise}/"
        files = os.listdir(files_path)
        files = [x for x in files if x.find("mp4")>-1]
    
        for file in files:
            if file.replace("mp4","") in down_data_df["filename"].tolist():
                os.remove(f"{files_path}{file}")
                print(f"Deleting {file}")
        return self

    def update_downloaded_files(self,append):
        self.__delete_duplicates()
        df = self.__read_download_files()
        if append==True:
            down_data_df = pd.read_csv(f"{self.path}/txt_files/{self.exercise}_dl_files.csv")
        else:
            down_data_df = pd.DataFrame() 
        for ix,row in df.iterrows():
            if ix % 10==0:
                print(ix) 
            vid = row['id']
            url = row['video_url']
            text = slugify(row["text"]).replace("-"," ")
            title = slugify(row['text'])[:50]
            filename = row['filename']
            shortcode = row['shortcode']
            if f"{title}.mp4" in self.downloaded_files_df["files2"].tolist():
                title_wn = str(self.downloaded_files_df[self.downloaded_files_df["files2"]==f"{title}.mp4"]["files"].iloc[0])
                video = VideoScraper.from_file(f"{self.path}/videos/{self.exercise}/{title_wn}.mp4",self.exercise)
                if video.error==False:
                    down_data_df = self.append_df(down_data_df,vid,video.filename,video.filepath,title,\
                                                video.duration,row['reps'],shortcode,self.exercise,text)
        down_data_df.to_csv(f"{self.path}/txt_files/{self.exercise}_dl_files.csv",index=False)
        save2db(self.path,down_data_df,f'{self.exercise}_dl_files')
        self.__delete_additional_videos(down_data_df)


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
        sc_to_exclude = list(df_trun["shortcode"].values)
        return ind_to_drop, vids_to_exclude, sc_to_exclude

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
        sc_to_exclude = config_json[self.exercise]["shortcodes_to_exclude"]
        ind_to_drop = []

        f_down_df = self.__filter_downloaded(down_df)
        print("removing")
        self.__remove_vids(f_down_df)
        i2d, vid2exc, sc2exc = self.__get_vids_to_drop(f_down_df)
        ind_to_drop += i2d
        # print(vids_to_exclude)
        # print("vid2exc")
        # print(vid2exc)
        vids_to_exclude += vid2exc
        sc_to_exclude += sc2exc
        # print(sc_to_exclude)
                
        config_json[self.exercise]["ids_to_exclude"] = list(set(vids_to_exclude))
        down_df = f_down_df.drop(["cond1","cond2","not_in_files","too_long","reps_t_long","t_many_reps"],axis=1)
        self.__delete_clipped(down_df,ind_to_drop)
        down_df.drop(ind_to_drop,axis=0,inplace=True)
        print(f"Number of videos after filtering: {len(down_df)}")
        down_df.to_csv(f"{self.path}/txt_files/{self.exercise}_dl_files.csv",index=False)
        save2db(self.path,down_df,f'{self.exercise}_dl_files')
        config_json[self.exercise]["ids_to_exclude"] = list(set([int(vid) for vid in vids_to_exclude]))
        config_json[self.exercise]["shortcodes_to_exclude"] = list(set([str(sc) for sc in sc_to_exclude]))
        # print(config_json)
        self.__save_config(config_json)

    def __delete_clipped(self,down_df,ind_to_drop):
        clipped_files = os.listdir(f"{self.path}videos/clipped/{self.exercise}/")
        clipped_files = [x for x in clipped_files if x.find("mp4")>-1]
        df_delete = [x.split("_")[1] for x in down_df.loc[ind_to_drop,"filename"].tolist()]
        for clip in clipped_files:
            if clip.split("_")[1] in df_delete:
                os.remove(f"{self.path}videos/clipped/{self.exercise}/{clip}")
                print(f"Deleting {clip}")

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
            num_clips = len(row["clips"])
            for ix,clip in enumerate(row["clips"]):
                if num_clips<=5 and ix>=1:
                    video.save_clipped(clip[0],clip[1],ix)
                elif num_clips>5 and ix>=1 and ix!=num_clips:
                    video.save_clipped(clip[0],clip[1],ix)


