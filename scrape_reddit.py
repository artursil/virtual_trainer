from connector.redditconsetup import * # import connection settings
from utils import *
import praw
import pandas as pd
import datetime as dt
import re
from video_extract import VideoScraper
import argparse
from tqdm import tqdm

# Setup Reddit connection
reddit = praw.Reddit(client_id=CLIENT_ID, \
                     client_secret=CLIENT_SECRET, \
                     user_agent=USER_AGENT, \
                     username=RDT_USERNAME, \
                     password=RDT_PASS)
rdt_channel = 'formcheck' # subreddit channel
cutoff = 5 # max number of submissions  to gather
scrape_dict = { "title": [], 
                "score": [], 
                "url": [],  
                "is_video": [],
                "duration": [], 
                "link_flair_text": [] }

def scrape_rdt(cutoff):
    # Get Reddit submissions
    subreddit = reddit.subreddit(rdt_channel) # select Reddit channel
    for submission in subreddit.top(limit=cutoff):
        scrape_dict["title"].append(submission.title)
        scrape_dict["score"].append(submission.score)
        scrape_dict["url"].append(submission.url)
        scrape_dict["is_video"].append(submission.is_video)
        if submission.is_video == True : 
            scrape_dict["duration"].append(submission.secure_media["reddit_video"]["duration"])
        else :
            scrape_dict["duration"].append(0)
        scrape_dict["link_flair_text"].append(submission.link_flair_text)
    return pd.DataFrame(scrape_dict)



def get_reps(title):
    reg1 =r"\d+[a-zA-Z]{0,3}\s{0,1}x\s{0,1}\d+[a-zA-Z]{0,3}"
    reg2 = r"\d+\s{0,1}rep"
    reg3 = r"\d+[a-zA-Z]{0,3}\s{0,1}for\s{0,1}\d+[a-zA-Z]{0,3}"
    m = re.search(f"{reg1}|{reg2}|{reg3}",title)
    if m:
        splitter = re.search(r"[a-zA-z]+\s{0,1}[a-zA-z]{0,3}",m.group(0)).group(0)
        m_ls = m.group(0).split(splitter)
        reps_n = min([int(re.sub("[a-zA-Z]{0,3}","",x).replace(" ","")) for x in m_ls if re.sub("[a-zA-Z]{0,3}","",x).replace(" ","")!=''])
    else:
        reps_n=1
    return reps_n

def filter_df(df,stop_words):
    df_tmp = df.copy()
    df_tmp = df_tmp[df_tmp["duration"]<=60]
    df_tmp['insta'] = [x.find("instagram")>-1 for x in df_tmp['url']]
    df_tmp['yt'] = [x.find("youtube")>-1 for x in df_tmp['url']]
    df_tmp = df_tmp[(df_tmp.is_video==True)|(df_tmp.insta==True)|(df_tmp.yt==True)]
    df_tmp['title_lower'] = [x.lower() for x in df_tmp['title']]
    df_tmp["_sumo_q"] = [any(s in x for s in stop_words) for x in df_tmp['title_lower']]
    df_tmp = df_tmp[df_tmp["_sumo_q"]==False]
    df_tmp['_deadlift'] = [any(s in x for s in ["deadlift","dl"]) for x in df_tmp['title_lower']]
    df_tmp['_squat'] = [x.find("squat")>-1 for x in df_tmp['title']]
    dl = zip(list(df_tmp['_deadlift']),list(df_tmp['link_flair_text'].str.lower().fillna("")=="deadlift"))
    df_tmp['deadlift'] = [max(a,b) for a,b in dl]
    sq = zip(list(df_tmp['_squat']),list(df_tmp['link_flair_text'].str.lower().fillna("")=="squat")) 
    df_tmp['squat'] = [max(a,b) for a,b in sq]
    # df_tmp['link_flair_text'] = [x.lower() if x!=None else x for x in df_tmp['link_flair_text']]
    df_tmp = df_tmp[(df_tmp['squat']==True)|(df_tmp['deadlift']==True)]
    df_tmp['reps'] = [get_reps(x) for x in df_tmp['title']]
    col_drop = [x for x in df_tmp.columns if x[0]=="_"]
    df_tmp = df_tmp.drop(col_drop,axis=1)
    return df_tmp


# BASE_PATH="D:/A_Documents/virtual_trainer/"


parser = argparse.ArgumentParser()
parser.add_argument('--path','-p', help='Path where videos should be saved')
parser.add_argument('--number-of-posts','-n',type=int,default=10, help='Number of reddit posts to scrape')

args = parser.parse_args()

def main():
    base_path = args.path
    cutoff = args.number_of_posts
    stop_words=["sumo","?","advice"]
    scrape_df = scrape_rdt(cutoff)
    print(f"Number of posts: {len(scrape_df)}")
    filtered_df = filter_df(scrape_df,stop_words)
    print(f"Number of posts after filtering: {len(filtered_df)}")
    for _,row in tqdm(filtered_df.iterrows()):
        if row["squat"] ==True:
            exercise="squat"
        else:
            exercise="deadlift"
        url = row["url"]
        title = slugify(row['title'])
        if row["yt"]==True:
            video = VideoScraper(url,base_path,exercise,video_type="youtube")
        else:
            video = VideoScraper(url,base_path,exercise,video_type="reddit",filename=title)
        video.download_video()

if __name__=='__main__':
    main()
#def filter_and_dl(rawscrape):
    #filter submissions and download videos
#    filteredsubs = filter_rdt(rawscrape)
#    for