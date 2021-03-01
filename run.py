#gpu must needed for this process 


import sys
import os
import cv2
import argparse
import torch
from torch.autograd import Variable
from torchvision import transforms
from utils import *
from networks import deeplab_xception_transfer, graph
from networks import custom_transforms as tr



parser = argparse.ArgumentParser()

parser.add_argument('--loadmodel', default='./model/cartoon.pth')
parser.add_argument('--human_img', required=True)
parser.add_argument('--out', default='./out/')
parser.add_argument('--use_gpu', default=1)
args = parser.parse_args()



net = deeplab_xception_transfer.deeplab_xception_transfer_projection_savemem(n_classes=20,
                                                                            hidden_layers=128,
                                                                              source_classes=7, )

x = torch.load(args.loadmodel)
net.load_source_model(x)                                                                       
                                                                              

if args.use_gpu >0 :
    net.cuda()
else:
    raise RuntimeError('must use the gpu!!!!')

human_img = args.human_img
cartoon_img = "./input/animation.jpg"

first_img_seg = inference(net, human_img, use_gpu=1)
second_img_seg = inference(net, cartoon_img, use_gpu=1)

first_img_alpha,first_img_mask = first_img_process(human_img ,first_img_seg)
second_img_head_alpha,second_img_head_mask = first_img_process(cartoon_img ,second_img_seg)
second_img_alpha,second_img_mask = second_img_process(cartoon_img ,second_img_seg)

human_img=read_img(human_img)
cartoon_img=read_img(cartoon_img)

first_img_head,first_img_dim,crop1=landmarks(human_img,first_img_mask,first_img_alpha)

second_img_head,second_img_dim,crop2=landmarks(cartoon_img,second_img_head_mask,second_img_head_alpha)

w_b=cartoon_img.size[0]
h_b=cartoon_img.size[1]
black_dim=(h_b,w_b,3)


x1,y1,h1,w1=crop1
x2,y2,h2,w2=crop2
dim=(w2,h2)
first_img_head = cv2.resize(first_img_head, dim, interpolation = cv2.INTER_NEAREST)
black = np.zeros(black_dim, dtype = "uint8")
black[y2:y2+h2, x2:x2+w2]=first_img_head # ORGINAL

output=final_mate(cartoon_img,second_img_alpha,black)


cv2.imwrite(args.out+"out.jpg",output)
