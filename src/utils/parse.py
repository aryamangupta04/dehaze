import argparse
import torch

parser=argparse.ArgumentParser()
parser.add_argument('--steps',type=int)
parser.add_argument('--device',type=str)
parser.add_argument('--epochs',default=100,type=int)
parser.add_argument('--crop_size',type=int)
parser.add_argument('--lr',default=0.0001,type=float)
parser.add_argument('--crop',action='store_true')
#parser.add_argument('--crop_size',type=int,default=240)
parser.add_argument('--bs',default=16,type=int)
parser.add_argument('--itr',type=int)
opt=parser.parse_args()
opt.device='cuda' if torch.cuda.is_available() else 'cpu'