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



class Bird_Dataset(data.Dataset):
    def __init__(self,split):
        self.split=split
        self.transform = transforms.Compose([
            transforms.Resize(int(256*76/64))])
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(256),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomHorizontalFlip()])

        import pickle
        file = open('C:/Users/TanTian/pythonproject/image_generation/captions_DAMSM.pickle', 'rb')
        data = pickle.load(file)
        self.ixtoword, self.wordtoix=data[2],data[3]
        self.filenames,self.all_captions,self.bboxs,self.attrs= self.load_bird_dataset(self.split)
        self.all_txt=self.all_captions
        self.class_id=np.arange(self.filenames.__len__())

    def __getitem__(self, index):
        img_path = self.filenames[index]
        bbox=self.bboxs[index]
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        img = img.crop([x1, y1, x2, y2])


        img= self.transform(img)
        img=self.norm(img)
        img=np.array(img)
        img=torch.tensor(img,dtype=torch.float32)

        cap_idx=random.randint(5*index,5*index+4)
        cap=self.all_captions[cap_idx]
        attr=self.attrs[cap_idx]
        cap_len=(cap.__len__()-cap.count(0))

        attr=np.array(attr)
        if np.count_nonzero(attr[1])==0:
            attr[1]=attr[0]
        if np.count_nonzero(attr[2])==0:
            attr[2]=attr[0]

        attr=torch.tensor(attr,dtype=torch.int32)
        img= Variable(img)
        attr = Variable(attr)

        cap=np.array(cap)
        cap = torch.tensor(cap, dtype=torch.int32)
        cap = Variable(cap)

        class_id = self.class_id[index]

        return img,attr,cap,cap_len,class_id

    def __len__(self):
        return len(self.filenames)

    def load_bird_dataset(self,split):
        filenames=[]
        all_captions=[]
        all_attrs=[]
        attrs_exists=False
        if os.path.exists('C:/Users/TanTian/pythonproject/image_generation/birds_data/train_attrs.npy') and os.path.exists('C:/Users/TanTian/pythonproject/image_generation/birds_data/test_attrs.npy'):
            attrs_exists=True
            if self.split=='train':
                all_attrs=np.load('C:/Users/TanTian/pythonproject/image_generation/birds_data/train_attrs.npy',allow_pickle=True)
            else:
                all_attrs=np.load('C:/Users/TanTian/pythonproject/image_generation/birds_data/test_attrs.npy',allow_pickle=True)
        if split=='test':
            f = open("C:/Users/TanTian/pythonproject/image_generation/birds_data/bird_images_test.txt", "r")  # 设置文件对象
        else:
            f = open("C:/Users/TanTian/pythonproject/image_generation/birds_data/bird_images_train.txt", "r")  # 设置文件对象
        line = f.readline()
        while line:  # 直到读取完文件
            if not line:
                break
            line = line.replace('\n','') # 去掉换行符，也可以不去
            filenames.append(line)
            line=line.replace('CUB_200_2011/images','text')
            line=line.replace('.jpg','.txt')
            captions_path=line
            with open(captions_path, "r") as cap_f:
                captions = cap_f.read().encode('utf-8').decode('utf8').split('\n')
                cnt_captions=0
                for cap in captions:
                    if len(cap) == 0:
                        continue
                    while cap.count('��'):
                        cap = cap.replace('��', ' ')
                    while cap.count('.'):
                        cap = cap.replace('.', '')
                    tokenizer = RegexpTokenizer(r'\w+')
                    tokens = tokenizer.tokenize(cap.lower())
                    tokens_new = []
                    for t in tokens:
                        t = t.encode('ascii', 'ignore').decode('ascii')
                        if len(t) > 0:
                            tokens_new.append(t)
                    if tokens_new.__len__() > 24:
                        continue
                    if cnt_captions < 5:
                        all_captions.append(tokens_new)
                        if not attrs_exists:
                            attrs = self.get_attrs(cap)
                            all_attrs.append(attrs)
                    else:
                        break
                    cnt_captions += 1
                if cnt_captions != 5:
                    print('the count of captions is not enough')
                    return 0
            line = f.readline()  # 读取一行文件，包括换行符
        if split=='test':
            bbox_f = open("C:/Users/TanTian/pythonproject/image_generation/birds_data/bboxs_test.txt", "r")  # 设置文件对象
        else:
            bbox_f = open("C:/Users/TanTian/pythonproject/image_generation/birds_data/bboxs_train.txt", "r")  # 设置文件对象
        line = bbox_f.readline()
        bboxs=[]
        while line:  # 直到读取完文件
            if not line:
                break
            line = line.replace('\n', '')  # 去掉换行符，也可以不去
            x1,width,x2,hight=line.split(' ')
            x1,width,x2,hight=float(x1),float(width),float(x2),float(hight)
            bboxs.append([x1,width,x2,hight])
            line = bbox_f.readline()  # 读取一行文件，包括换行符
        if attrs_exists==False:
            if self.split == 'train':
                np.save('C:/Users/TanTian/pythonproject/image_generation/birds_data/train_attrs.npy', all_attrs)
            else:
                np.save('C:/Users/TanTian/pythonproject/image_generation/birds_data/test_attrs.npy', all_attrs)
        return filenames,all_captions,bboxs,all_attrs


class Bird_Dataset_DF_GAN(data.Dataset):
    def __init__(self,split):
        self.filenames=[]
        self.all_captions=[]
        self.bboxs=[]
        self.attrs=[]
        self.all_txt=[]
        import pickle
        if split=='train':
            load_file = open("C:/Users/TanTian/pythonproject/image_generation/filenames_train.pickle", "rb")
        elif split=='test':
            load_file = open("C:/Users/TanTian/pythonproject/image_generation/filenames_test.pickle", "rb")
        data = pickle.load(load_file)
        train_dataset = Bird_Dataset('train')
        test_dataset = Bird_Dataset('test')

        import pickle
        file = open('C:/Users/TanTian/pythonproject/image_generation/captions_DAMSM.pickle', 'rb')
        data2 = pickle.load(file)
        self.ixtoword, self.wordtoix=data2[2],data2[3]

        for i in range(train_dataset.__len__()):
            filename=train_dataset.filenames[i]
            filename=filename.replace('./data/birds/CUB_200_2011/images/','')
            filename=filename.replace('.jpg','')
            if filename in data:
                self.filenames.append(train_dataset.filenames[i])
                self.all_captions.append(train_dataset.all_captions[i*5])
                self.all_captions.append(train_dataset.all_captions[i*5+1])
                self.all_captions.append(train_dataset.all_captions[i*5+2])
                self.all_captions.append(train_dataset.all_captions[i*5+3])
                self.all_captions.append(train_dataset.all_captions[i*5+4])
                self.bboxs.append(train_dataset.bboxs[i])
                self.attrs.append(train_dataset.attrs[i*5])
                self.attrs.append(train_dataset.attrs[i*5+1])
                self.attrs.append(train_dataset.attrs[i*5+2])
                self.attrs.append(train_dataset.attrs[i*5+3])
                self.attrs.append(train_dataset.attrs[i*5+4])

                self.all_txt.append(train_dataset.all_txt[i*5])
                self.all_txt.append(train_dataset.all_txt[i*5+1])
                self.all_txt.append(train_dataset.all_txt[i*5+2])
                self.all_txt.append(train_dataset.all_txt[i*5+3])
                self.all_txt.append(train_dataset.all_txt[i*5+4])

        for i in range(test_dataset.__len__()):
            filename = test_dataset.filenames[i]
            filename = filename.replace('C:/Users/TanTian/pythonproject/image_generation/data/birds/CUB_200_2011/images/', '')
            filename = filename.replace('.jpg', '')
            if filename in data:
                self.filenames.append(test_dataset.filenames[i])
                self.all_captions.append(test_dataset.all_captions[i * 5])
                self.all_captions.append(test_dataset.all_captions[i * 5 + 1])
                self.all_captions.append(test_dataset.all_captions[i * 5 + 2])
                self.all_captions.append(test_dataset.all_captions[i * 5 + 3])
                self.all_captions.append(test_dataset.all_captions[i * 5 + 4])
                self.bboxs.append(test_dataset.bboxs[i])
                self.attrs.append(test_dataset.attrs[i * 5])
                self.attrs.append(test_dataset.attrs[i * 5 + 1])
                self.attrs.append(test_dataset.attrs[i * 5 + 2])
                self.attrs.append(test_dataset.attrs[i * 5 + 3])
                self.attrs.append(test_dataset.attrs[i * 5 + 4])

                self.all_txt.append(test_dataset.all_txt[i* 5])
                self.all_txt.append(test_dataset.all_txt[i* 5 + 1])
                self.all_txt.append(test_dataset.all_txt[i* 5 + 2])
                self.all_txt.append(test_dataset.all_txt[i* 5 + 3])
                self.all_txt.append(test_dataset.all_txt[i* 5 + 4])

        self.transform = transforms.Compose([
            transforms.Resize(int(256*76/64))])
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(256),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomHorizontalFlip()])
        self.class_id=np.arange(self.filenames.__len__())

    def __getitem__(self, index):
        img_path = self.filenames[index]
        bbox=self.bboxs[index]
        img = Image.open(img_path).convert('RGB')
        width, height = img.size
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        img = img.crop([x1, y1, x2, y2])


        img= self.transform(img)
        img=self.norm(img)
        img=np.array(img)
        img=torch.tensor(img,dtype=torch.float32)

        cap_idx=random.randint(5*index,5*index+4)
        cap=self.all_txt[cap_idx]
        attr = self.attrs[cap_idx]

        attrs=[]
        for a in attr:
            attr_i=''
            for word in a:
                attr_i+=word+' '
            attrs.append(attr_i[:-1])
            if attrs.__len__()==3:
                break
        while attrs.__len__()<3:
            attrs.append('')

        img= Variable(img)

        tot_txt = ''
        for c in cap:
            tot_txt += c + ' '
        return img,attrs,tot_txt

    def __len__(self):
        return len(self.filenames)

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





