
#gpu must needed for this process or else process will be delay

import socket
import timeit
import numpy as np
from PIL import Image
from datetime import datetime
import os
import sys
from collections import OrderedDict
sys.path.append('./')
# PyTorch includes
import torch
from torch.autograd import Variable
from torchvision import transforms
import cv2
import face_alignment
import warnings
warnings.filterwarnings("ignore")


# Custom includes
from networks import deeplab_xception_transfer, graph
from networks import custom_transforms as tr

#
import argparse
import torch.nn.functional as F
 
 
label_colours = [(0,0,0)
              , (128,0,0), (255,0,0), (0,85,0), (170,0,51), (255,85,0), (0,0,85), (0,119,221), (85,85,0), (0,85,85), (85,51,0), (52,86,128), (0,128,0)
              , (0,0,255), (51,170,221), (0,255,255), (85,255,170), (170,255,85), (255,255,0), (255,170,0)]



def landmarks(img,white_img,lndmrk_dat_file,alpha_mate):

    
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    input = np.asarray(img)
    landmarks = fa.get_landmarks(input)

    center = landmarks[0][27].tolist()

    top_plt=[]
    right_plt=[]
    left_plt=[]
    bottom_plt=[]
    black=[0 ,0 , 0]

    x,y = int(center[0]),int(center[1])
    
    #right plotting    
    while True:
      color = white_img[y, x].tolist()
      if color==black:
        right_plt.append(x)
        break
      else:
        x+=1

           
    #left plotting  

    x,y = int(center[0]),int(center[1])

    while True:
      color = white_img[y, x].tolist()
      if color==black:
        left_plt.append(x)
        break
      else:
        x-=1
    
    #top plotting   

    x,y = int(center[0]),int(center[1])

    while True:
      color = white_img[y, x].tolist()
      if color==black:
        top_plt.append(y)
        break
      else:
        y-=1

    #bottom plotting   
    x,y = int(center[0]),int(center[1])

    while True:
      color = white_img[y, x].tolist()
      if color==black:
        bottom_plt.append(y)
        break
      else:
        y+=1
      

    y=top_plt[0]
    x=left_plt[0]
    h=bottom_plt[0]-top_plt[0]
    w=right_plt[0]-left_plt[0]
    #print(top_plt,left_plt,bottom_plt,right_plt)
    dim=(h,w)
    crop=(x,y,h,w)
    alpha_mate[y:y+h, x:x+w]
    head = alpha_mate[y:y+h, x:x+w]
    return head,dim,crop


def flip(x, dim):
    indices = [slice(None)] * x.dim()
    indices[dim] = torch.arange(x.size(dim) - 1, -1, -1,
                                dtype=torch.long, device=x.device)
    return x[tuple(indices)]

def flip_cihp(tail_list):
    '''

    :param tail_list: tail_list size is 1 x n_class x h x w
    :return:
    '''
    # tail_list = tail_list[0]
    tail_list_rev = [None] * 20
    for xx in range(14):
        tail_list_rev[xx] = tail_list[xx].unsqueeze(0)
    tail_list_rev[14] = tail_list[15].unsqueeze(0)
    tail_list_rev[15] = tail_list[14].unsqueeze(0)
    tail_list_rev[16] = tail_list[17].unsqueeze(0)
    tail_list_rev[17] = tail_list[16].unsqueeze(0)
    tail_list_rev[18] = tail_list[19].unsqueeze(0)
    tail_list_rev[19] = tail_list[18].unsqueeze(0)
    return torch.cat(tail_list_rev,dim=0)


def decode_labels(mask, num_images=1, num_classes=20):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).

    Returns:
      A batch with num_images RGB images of the same size as the input.
    """
    n, h, w = mask.shape
    assert (n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (
    n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    pixels[k_, j_] = label_colours[k]
        outputs[i] = np.array(img)
    return outputs

def read_img(rgb):

  img=cv2.imread(rgb)
  height,width=img.shape[:2]
  scale_percent = 50 


  while height >= 900 or width >= 900:
    width = int(width * scale_percent / 100)
    height = int(height * scale_percent / 100)

  dim = (width,height)
  
  img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
  img = Image.fromarray(img)

  return img

def img_transform(img, transform=None):
    sample = {'image': img, 'label': 0}

    sample = transform(sample)
    return sample

def inference(net, img_path='', use_gpu=True):
    '''

    :param net:
    :param img_path:
    :param output_path:
    :return:
    '''
    # adj
    adj2_ = torch.from_numpy(graph.cihp2pascal_nlp_adj).float()
    adj2_test = adj2_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 20).cuda().transpose(2, 3)

    adj1_ = Variable(torch.from_numpy(graph.preprocess_adj(graph.pascal_graph)).float())
    adj3_test = adj1_.unsqueeze(0).unsqueeze(0).expand(1, 1, 7, 7).cuda()

    cihp_adj = graph.preprocess_adj(graph.cihp_graph)
    adj3_ = Variable(torch.from_numpy(cihp_adj).float())
    adj1_test = adj3_.unsqueeze(0).unsqueeze(0).expand(1, 1, 20, 20).cuda()

    # multi-scale
    scale_list = [1, 0.5, 0.75, 1.25, 1.5, 1.75]
    img= read_img(img_path)
    testloader_list = []
    testloader_flip_list = []
    for pv in scale_list:
        composed_transforms_ts = transforms.Compose([
            tr.Scale_only_img(pv),
            tr.Normalize_xception_tf_only_img(),
            tr.ToTensor_only_img()])

        composed_transforms_ts_flip = transforms.Compose([
            tr.Scale_only_img(pv),
            tr.HorizontalFlip_only_img(),
            tr.Normalize_xception_tf_only_img(),
            tr.ToTensor_only_img()])

        testloader_list.append(img_transform(img, composed_transforms_ts))
        # print(img_transform(img, composed_transforms_ts))
        testloader_flip_list.append(img_transform(img, composed_transforms_ts_flip))
    # print(testloader_list)
    start_time = timeit.default_timer()
    # One testing epoch
    net.eval()
    # 1 0.5 0.75 1.25 1.5 1.75 ; flip:

    for iii, sample_batched in enumerate(zip(testloader_list, testloader_flip_list)):
        inputs, labels = sample_batched[0]['image'], sample_batched[0]['label']
        inputs_f, _ = sample_batched[1]['image'], sample_batched[1]['label']
        inputs = inputs.unsqueeze(0)
        inputs_f = inputs_f.unsqueeze(0)
        inputs = torch.cat((inputs, inputs_f), dim=0)
        if iii == 0:
            _, _, h, w = inputs.size()
        # assert inputs.size() == inputs_f.size()

        # Forward pass of the mini-batch
        inputs = Variable(inputs, requires_grad=False)

        with torch.no_grad():
            if use_gpu >= 0:
                inputs = inputs.cuda()
            # outputs = net.forward(inputs)
            outputs = net.forward(inputs, adj1_test.cuda(), adj3_test.cuda(), adj2_test.cuda())
            outputs = (outputs[0] + flip(flip_cihp(outputs[1]), dim=-1)) / 2
            outputs = outputs.unsqueeze(0)

            if iii > 0:
                outputs = F.upsample(outputs, size=(h, w), mode='bilinear', align_corners=True)
                outputs_final = outputs_final + outputs
            else:
                outputs_final = outputs.clone()
    ################ plot pic
    predictions = torch.max(outputs_final, 1)[1]
    results = predictions.cpu().numpy()
    vis_res = decode_labels(results)

    parsing_im = Image.fromarray(vis_res[0])
    return parsing_im

def first_img_process(rgb,color_img):
    newimdata = []
    redcolor = (255,0,0)
    bluecolor = (0,0,255)
    white_color = (255,255,255)
    blackcolor = (0,0,0)
    #yellowcolor = (242, 214, 0)
    for color in color_img.getdata():
        if color == redcolor or color == bluecolor :
            newimdata.append(white_color)
        else:
            newimdata.append(blackcolor)
    newim = Image.new(color_img.mode,color_img.size)
    newim.putdata(newimdata)
    mask = np.array(newim)
    rgb = read_img(rgb)
    rgb = np.asarray(rgb)
    

    alpha = cv2.bitwise_and(rgb, mask)
    
    return alpha,mask

def second_img_process(rgb,color_img):
    newimdata = []
    redcolor = (255,0,0)
    bluecolor = (0,0,255)
    white_color = (255,255,255)
    blackcolor = (0,0,0)
    #yellowcolor = (242, 214, 0)
    for color in color_img.getdata():
      if color == redcolor or color == bluecolor or color == blackcolor:
        newimdata.append( blackcolor )
      else:
        newimdata.append( white_color )
    newim = Image.new(color_img.mode,color_img.size)
    newim.putdata(newimdata)
    newim = Image.new(color_img.mode,color_img.size)
    newim.putdata(newimdata)
    mask = np.array(newim)
    rgb = read_img(rgb)
    rgb = np.asarray(rgb)
    alpha = cv2.bitwise_and(rgb, mask)

    return alpha,mask    

def final_mate(cartoon_img,mask,black):
  # Read the images
  foreground = cartoon_img
  foreground = np.asarray(foreground)

  background = black
  background = np.asarray(background)
  alpha = mask

  # Convert uint8 to float
  foreground = foreground.astype(float)
  background = background.astype(float)

  # Normalize the alpha mask to keep intensity between 0 and 1
  alpha = alpha.astype(float)/255

  # Multiply the foreground with the alpha matte
  foreground = cv2.multiply(alpha, foreground)
  # Multiply the background with ( 1 - alpha )
  background = cv2.multiply(1.0 - alpha, background)

  # Add the masked foreground and background.
  outImage = cv2.add(foreground, background)

  return outImage