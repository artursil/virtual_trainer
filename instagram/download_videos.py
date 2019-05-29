from slugify import slugify
from video_extract import VideoScraper
from utils import *

#TODO move somewhere to not duplicate it
def create_metadata_df(path,filename):
    if os.path.isfile(f"{path}/{filename}"):
        metadata_df = pd.read_csv(f"{path}/{filename}")
    else:
        metadata_df = pd.DataFrame()
    return metadata_df

#TODO move somewhere to not duplicate it
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


def download_videos(df,path):
    # down_data_df = create_metadata_df(f"{path}/txt_files","downloaded_files.csv")
    down_data_df = pd.DataFrame()
    for ix,row in df.iterrows():
        if ix % 10==0:
            print(ix) 
        vid = row['id']
        url = row['video_url']
        exercise = row['exercise']
        base_path = path
        text = slugify(row["text"]).replace("-"," ")
        title = slugify(row['text'])[:50]
        filename = row['filename']
        video = VideoScraper(url,base_path,exercise,video_type="other",filename=filename)
        print(video.error)
        import pdb; pdb.set_trace() 
        if video.error==False:
            success = video.download_video()
            if success:
                duration = video.duration
                down_data_df = append_df(down_data_df,vid,video.filename,video.filepath,title,\
                                        duration,row['reps'],exercise,text)
    down_data_df.to_csv(f"{base_path}/txt_files/{exercise}_dl_files.csv",index=False)


