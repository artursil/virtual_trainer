from create_metadata import create_metadata
from filter_posts import filter_posts

PATH = ""
def main():
    create_metada(PATH,"deadlift","100000")
    filter_posts(PATH,"deadlift",200)

if __name__=='__main__':
    main()