


import torch,os,sys,torchvision,argparse
import torchvision.transforms as tfs
import time,math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch,warnings
from torch import nn
import torchvision.utils as vutils
warnings.filterwarnings('ignore')

parser=argparse.ArgumentParser()
parser.add_argument('-g', '--gpus', default=1, type=int, metavar='N')
parser.add_argument('--steps',type=int,default=1175, help='batches')
parser.add_argument('--device',type=str,default='Automatic detection')
parser.add_argument('--resume',type=bool,default=False) #False True
parser.add_argument('--eval_step',type=int,default=1176, help='batches+1')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--model_dir',type=str,default='./trained_models/')  #保存的模型存放位置
parser.add_argument('--train_folder',type=str,default='./data/')  #数据集位置,根文件夹
parser.add_argument('--log_dir',type=str,default='./logs/', help='tensorboard folder')
parser.add_argument('--net',type=str,default='csa')
parser.add_argument('--bs',type=int,default=1,help='batch size')
parser.add_argument('--testbs',type=int,default=1,help='batch size')
parser.add_argument('--crop',action='store_false')   #因为action='store_true'，所以此参数默认值可能为false 此用法起到开关的作用
parser.add_argument('--crop_size',type=int,default=256,help='Takes effect when using --crop ')
parser.add_argument('--no_lr_sche',action='store_true',help='no lr cos schedule')
parser.add_argument('--perloss',action='store_false',help='perceptual loss')
parser.add_argument('--final_epoch',type=int,default=200)
parser.add_argument('--SW_bias',type=bool,default=False)


opt=parser.parse_args()

opt.device='cuda' if torch.cuda.is_available() else 'cpu'

if not os.path.exists('trained_models'):
	os.mkdir('trained_models')
if not os.path.exists('logs'):
	os.mkdir('logs')

