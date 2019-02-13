from openpose.main import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path','-p', help='Path where clipped videos are')

args = parser.parse_args()

if __name__=='__main__':
    main(args.path)