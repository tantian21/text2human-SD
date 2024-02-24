import pickle

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from torchvision import models
import argparse
import functools
import os
import json
import math
from collections import defaultdict
import random
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import torchvision.transforms as transforms
import cv2
import parser
from PIL import Image
from torch.autograd import Variable
import torchvision.utils
import difflib
import torch.utils.data as data
import scipy.io as scio


from nltk.tokenize import RegexpTokenizer

class Clothing_Dataset(data.Dataset):
    def __init__(self,split='test'):
        self.filenames=[]
        self.masks=[]
        self.all_captions=[]
        self.transform = transforms.Resize([512,512])
        self.resze=transforms.Resize([836,550])
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )
        self.transform64=transforms.Resize([64,64])
        imgs_path='./data/clothing-co-parsing-master/clothing-co-parsing-master/photos/'
        masks_path='./data/clothing-co-parsing-master/clothing-co-parsing-master/annotations/pixel-level/'
        for i in range(1,1005):
            self.filenames.append(imgs_path+str(i).zfill(4)+'.jpg')
            self.masks.append(masks_path+str(i).zfill(4)+'.mat')

            continue
            mask = scio.loadmat(masks_path+str(i).zfill(4)+'.mat')
            mask = dict(mask)['groundtruth']
            height, width = mask.shape[1],mask.shape[0]
            mask=np.array(mask)
            new_mask = np.zeros([width, width])
            new_mask[:, (width - height) // 2:(width - height) // 2 + height] = mask
            mask = new_mask
            mask = Image.fromarray(np.uint8(mask))
            mask = self.transform(mask)
            plt.imsave('./masks/'+str(i-1)+'.png',np.array(mask))
        self.label_list=['null','accessories','bag','belt','blazer','blouse','bodysuit','boots','bra','bracelet','cape','cardigan','clogs','coat','dress','earrings','flats','glasses','gloves','hair','hat','heels','hoodie','intimate','jacket','jeans','jumper','leggings','loafers','necklace','panties','pants','pumps','purse','ring','romper','sandals','scarf','shirt','shoes','shorts','skin','skirt','sneakers','socks','stockings','suit','sunglasses','sweater','sweatshirt','swimwear','t-shirt','tie','tights','top','vest','wallet','watch','wedges']

        self.split=split

        self.train_filenames=[]
        self.train_masks=[]
        for i in range(1005,2099):
            self.train_filenames.append(imgs_path+str(i).zfill(4)+'.jpg')
            self.train_masks.append('./data/clothing-co-parsing-master/clothing-co-parsing-master/annotations/image-level/'+str(i).zfill(4)+'.mat')

    def __getitem__(self, index):
        if self.split=='test':
            img_path = self.filenames[index]
            img = Image.open(img_path).convert('RGB')
            txts=[]
            masks=[]
            tot_txt=''

            img=self.resze(img)
            height, width= img.size#[550,836]
            img=np.array(img)
            new_img=np.zeros([width,width,3])
            new_img[:,(width-height)//2:(width-height)//2+height,:]=img
            img=new_img
            img=Image.fromarray(np.uint8(img))
            img= self.transform(img)
            img= self.norm(img)
            img=torch.tensor(np.array(img),dtype=torch.float32)

            mask_path=self.masks[index]
            mask = scio.loadmat(mask_path)
            mask = dict(mask)['groundtruth']

            ones = np.ones_like(mask)
            zeros = np.zeros_like(mask)

            ones_t = torch.ones([512,512])
            zeros_t = torch.zeros([512,512])
            mask_background=torch.ones([1,512,512])
            for i in range(len(self.label_list)):
                if i in mask and self.label_list[i] not in ['null','hair','skin']:
                    txts.append(self.label_list[i])
                    mask_i=np.where(mask==i, ones, zeros)

                    mask_i=Image.fromarray(np.uint8(mask_i))
                    mask_i=self.resze(mask_i)
                    new_mask_i = np.zeros([width, width])
                    new_mask_i[:,(width - height) // 2:(width - height) // 2 + height] = mask_i
                    mask_i = new_mask_i
                    mask_i=Image.fromarray(np.uint8(mask_i))
                    mask_i= self.transform(mask_i)
                    mask_i=torch.tensor(np.array(mask_i),dtype=torch.float32)
                    mask_i=torch.where(mask_i!=0, ones_t, zeros_t)
                    mask_i=torch.unsqueeze(mask_i,0)

                    masks.append(mask_i)
                    mask_background-=mask_i
                    tot_txt+=self.label_list[i]+' '
            mask_background=torch.clip(mask_background,0,1)

            return img,masks,txts,mask_background,tot_txt[:-1]

        elif self.split=='train':
            img_path = self.train_filenames[index]
            img = Image.open(img_path).convert('RGB')
            tot_txt = ''

            img = self.resze(img)
            height, width = img.size  # [550,836]
            img = np.array(img)
            new_img = np.zeros([width, width, 3])
            new_img[:, (width - height) // 2:(width - height) // 2 + height, :] = img
            img = new_img
            img = Image.fromarray(np.uint8(img))
            img = self.transform(img)
            img = self.norm(img)
            img = torch.tensor(np.array(img), dtype=torch.float32)

            mask_path = self.train_masks[index]
            mask = scio.loadmat(mask_path)['tags'][0]

            for i in range(len(self.label_list)):
                if i in mask and self.label_list[i] not in ['null', 'hair', 'skin']:
                    tot_txt += self.label_list[i] + ' '

            return img,None,None,None, tot_txt[:-1]

    def __len__(self):
        return len(self.filenames)






