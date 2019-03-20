from utils import *
from instagram.create_metadata import create_metadata
from instagram.filter_posts import filter_posts
from instagram.download_videos import download_videos
from instagram.exercise_scraper import ExerciseScraper

PATH = "/media/artursil/DATA/Documents_D/data_science/win/virtual_trainer/instagram/other_exercises/"
insta_urls_df = pd.read_csv(f'{PATH}txt_files/additionalclasses.csv')

def change_clasess_name(class_name):
    class_name = class_name.replace('pushup','pushups')\
                           .replace('wall_pushups','wallpushups')\
                           .replace('pullup','pullups')\
                           .replace('clean_and_jerk','cleanandjerk')
    return class_name


classes = insta_urls_df['class']
classes = [change_clasess_name(x) for x in classes]
unique_classes = list(set(classes))
insta_urls_df['class'] = classes

for exercise in unique_classes:
    print(exercise)
    if exercise in ['lunges','wallpushups']:
        continue
    exc_scraper = ExerciseScraper.from_file(PATH,exercise)
    exercise_urls = insta_urls_df[insta_urls_df['class']==exercise]
    for ix, row in exercise_urls.iterrows():
        print(ix)
        ix = unique_classes.index(exercise)*100 + ix
        exc_scraper.save_filter_csv(row['link'],ix)
    exc_scraper.download_videos(append=False)