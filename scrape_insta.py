from utils import *
from instagram.create_metadata import create_metadata
from instagram.filter_posts import filter_posts
from instagram.download_videos import download_videos

# PATH = "D:/Documents_D/data_science/win/virtual_trainer/videos/instagram/"
PATH = "D:/Documents_D/data_science/win/virtual_trainer/instagram/"
IF_DOWNLOAD = True
exercises = ['deadlift','squat']

# TODO parse args

def scrape_exercise(path,exercise,download):
    txt_files_path = f"{path}/txt_files/"
    if download:
        try:
            create_metadata(txt_files_path,exercise,"100000")
        except RunTimeError:
            print("RunTimeError")
    filtered_df = filter_posts(txt_files_path,exercise)
    filtered_df.to_csv(f"{txt_files_path}/{exercise}_filtered_df.csv",index=False)
    filtered_df = pd.read_csv(f"{txt_files_path}/{exercise}_filtered_df.csv")
    print(len(filtered_df))
    print(filtered_df.head())
    download_videos(filtered_df,PATH)
    # create_metadata(PATH,exercise,"100000")  
    filtered_df = filter_posts(PATH,exercise,200)



def main():
    exercise = "squat"
    if IF_DOWNLOAD==True:
        download=True
    for exercise in exercises:
        scrape_exercise(PATH,exercise,download)

if __name__=='__main__':
    main()