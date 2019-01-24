from connector.redditconsetup import * # import connection settings

import praw
import pandas as pd
import datetime as dt

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

def scrape_rdt():
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

#def filter_and_dl(rawscrape):
    #filter submissions and download videos
#    filteredsubs = filter_rdt(rawscrape)
#    for
