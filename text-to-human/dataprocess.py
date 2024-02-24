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




# import nltk
# nltk.download()
# nltk.download('averaged_perceptron_tagger')
class Coco_Dataset(data.Dataset):
    def __init__(self,split):
        self.split = split
        self.transform = transforms.Compose([
            transforms.Resize((256, 256))])
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomHorizontalFlip()])
        self.filenames, self.all_captions, self.attrs, self.ixtoword, self.wordtoix, self.n_words = self.load_coco_dataset(self.split)
        self.class_id = np.arange(self.filenames.__len__())

        # dict={'caps':self.all_captions,'attrs':self.attrs}
        # file=open('./data/coco/'+split+'.pickle','wb')
        # pickle.dump(dict,file)
        # file.close()

    def __getitem__(self, index):
        if self.filenames[index]=='COCO_train2014_000000167126':
            index=0

        img_path = self.filenames[index]
        if self.split=='train':
            img_path='C:/Users/TanTian/pythonproject/image_generation/data/coco/images/train2014/'+img_path+'.jpg'
        else:
            img_path = 'C:/Users/TanTian/pythonproject/image_generation/data/coco/images/test2014/' + img_path+'.jpg'
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        img = self.norm(img)
        img = np.array(img)
        img = torch.tensor(img, dtype=torch.float32)

        cap_idx = random.randint(5 * index, 5 * index + 4)

        cap = self.all_captions[cap_idx]
        attr = self.attrs[cap_idx]


        attr = np.array(attr)

        new_attr=np.zeros([3,5])
        random.shuffle(attr)
        for idxi,lst in enumerate(attr,0):
            for idxj,val in enumerate(lst,0):
                new_attr[idxi][idxj]=val
                if idxj==4:
                    break
            if idxi==2:
                break
        attr=new_attr
        new_cap=np.zeros([30])
        len=min(30,cap.__len__())
        new_cap[:len]=cap[:len]
        cap=new_cap

        cap_len = len

        if np.count_nonzero(attr[1]) == 0:
            attr[1] = attr[0]
        if np.count_nonzero(attr[2]) == 0:
            attr[2] = attr[0]
        attr = attr.astype(np.int32)
        attr = torch.tensor(attr, dtype=torch.int32)
        img = Variable(img)
        attr = Variable(attr)

        img = Variable(img)
        cap = np.array(cap)
        cap = torch.tensor(cap, dtype=torch.int32)
        cap = Variable(cap)

        class_id = self.class_id[index]


        tot_txt=''
        for idx in cap:
            idx=int(idx)
            if idx!=0:
                tot_txt+=self.ixtoword[idx]+' '
        attrs=[]
        for a in attr:
            attr_i=''
            for idx in a:
                idx=int(idx)
                if idx!=0:
                    attr_i+=self.ixtoword[idx]+' '
            attrs.append(attr_i[:-1])

        return img, attrs, tot_txt[:-1]

    def __len__(self):
        return len(self.filenames)

    def load_coco_dataset(self,split):
        import pickle

        filepath = 'C:/Users/TanTian/pythonproject/image_generation/data/coco/captions.pickle'
        file = open('C:/Users/TanTian/pythonproject/image_generation/data/coco/train/filenames.pickle', 'rb')
        train_names = pickle.load(file)
        file = open('C:/Users/TanTian/pythonproject/image_generation/data/coco/test/filenames.pickle', 'rb')
        test_names = pickle.load(file)

        with open(filepath, 'rb') as f:
            x = pickle.load(f)
            train_captions, test_captions, train_attrs, test_attrs = x[0], x[1], x[4], x[5]
            ixtoword, wordtoix = x[2], x[3]
            del x
            n_words = len(ixtoword)
            print('Load from: ', filepath)
        if split == 'train':
            # a list of list: each list contains
            # the indices of words in a sentence
            captions = train_captions
            attrs = train_attrs
            filenames = train_names
        else:  # split=='test'
            captions = test_captions
            attrs = test_attrs
            filenames = test_names
        if(split=='train'):
            file = open('C:/Users/TanTian/pythonproject/image_generation/data/coco/train.pickle', 'rb')
            train_names = pickle.load(file)
            captions=train_names['caps']
            attrs=train_names['attrs']
        else:
            file = open('C:/Users/TanTian/pythonproject/image_generation/data/coco/test.pickle', 'rb')
            test_names = pickle.load(file)
            captions=test_names['caps']
            attrs=test_names['attrs']
        return filenames, captions, attrs, ixtoword, wordtoix, n_words





