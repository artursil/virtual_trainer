from openpose.main import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path','-p', help='Path where clipped videos are')
parser.add_argument('--start',type=int, help='Starting point for a dataloader')
parser.add_argument('--save-img',action='store_true', help='Save random images for results validation')
parser.add_argument('--swapped',action='store_true', help='Use swapped images in dataloader')
parser.add_argument('--t-img',action='store_true', help='Print time per image')


args = parser.parse_args()

if __name__=='__main__':
    main(args.path,args.start,args.save_img,args.swapped,args.t_img)