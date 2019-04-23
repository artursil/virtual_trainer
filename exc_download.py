from utils import *
from instagram.create_metadata import create_metadata
from instagram.filter_posts import filter_posts
from instagram.download_videos import download_videos
from instagram.exercise_scraper import ExerciseScraper

PATH = "instagram"
NUM_OF_POSTS = 200000
IF_DOWNLOAD = False
exercises = ['lunge']
all_tags = {'deadlift':[],
        'squat':[],
        'cleanandjerk':['cleanandjerk'],
        'lunge':['lunge'],
        'pullup':['pullup']}


def main():
    for exercise in exercises:
        exc_scraper = ExerciseScraper(PATH,exercise)
        tags = all_tags[exercise]
        for tag in tags:    
            exc_scraper.create_metadata(tag,NUM_OF_POSTS)

if __name__=='__main__':
    main()
