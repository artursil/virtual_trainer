from utils import *
from instagram.create_metadata import create_metadata
from instagram.filter_posts import filter_posts
from instagram.download_videos import download_videos

# PATH = "D:/Documents_D/data_science/win/virtual_trainer/videos/instagram/"
PATH = "D:/Documents_D/data_science/win/virtual_trainer/instagram/"
IF_DOWNLOAD = False
exercises = ['deadlift','squat']
all_tags = {'deadlift':[],
        'squat':[]}
NUM_OF_POSTS = 70000
# TODO parse args


def main():
    if IF_DOWNLOAD==True:
        download=True
    else:
        download=False

    for exercise in exercises:
        exc_scraper = ExerciseScraper(PATH,exercise)
        tags = all_tags[exercise]
        for tag in tags
            exc_scraper.create_metadata(tag,NUM_OF_POSTS)
        exc_scraper.fitler_posts()
        exc_scraper.download_videos()
        exc_scraper.update_filtered()
        exc_scraper.clip_videos()

if __name__=='__main__':
    main()