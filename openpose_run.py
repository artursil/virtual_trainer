from openpose.main import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path','-p', help='Path where clipped videos are')
parser.add_argument('--start',type=int, help='Starting point for a dataloader')

args = parser.parse_args()

if __name__=='__main__':
    main(args.path,args.start)