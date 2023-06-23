import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable
import torchvision.models as models
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import random
from dataloaders.transform import *
import torch.optim as optim
import os

def default_loader(path):
    return cv2.imread(path)
    
class make_dataset(Dataset):
    def __init__(self,txt,loader=default_loader):
        super(make_dataset,self).__init__()
        fh=open(txt,'r')
        slices=[]
        filter_num=0
        for line in fh:
            words=line.strip().split()
            slices,filter_num=self._filter(words,slices,filter_num)
        self.slices=slices
        self.filter_num=filter_num

        self.loader = loader
		self.rand_contrast = rand_contrast
		self.rand_clahe = rand_clahe
		self.rand_hue = rand_hue
        self.rand_flip=rand_flip
        self.rand_trans=rand_trans
        self.rand_rotate=rand_rotate
        self.rand_crop=rand_crop
        self.rand_rotate90=rand_rotate90
        print('Filter_num:' + str(self.filter_num) + ' Slices_num:' + str(len(self.slices)))

    def _filter(self,words, slices, filter_num):
        slice_label_path = words[1]
        slice_label = cv2.imread((slice_label_path))
        if np.sum(slice_label) == 0:
            filter_num = filter_num + 1
        else:
            slices.append((words[0], words[1]))
        return slices, filter_num

    def __getitem__(self, index):
        fn,label=self.slices[index]
        slice_img=self.loader(fn)
        slice_label=self.loader(label)
		scale=random.choice([256,224,168])
        slice_img,slice_label=self.rand_crop(slice_img,slice_label,scale,scale)
        slice_img = cv2.resize(slice_img, (224,224), interpolation=cv2.INTER_AREA)
        slice_label = cv2.resize(slice_label, (224,224), interpolation=cv2.INTER_AREA)

        slice_img=self.rand_contrast(slice_img)
        slice_img=self.rand_clahe(slice_img)
        slice_img=self.rand_hue(slice_img)
		slice_img,slice_label = self.rand_rotate90(slice_img,slice_label)
        slice_img,slice_label = self.rand_flip(slice_img,slice_label)
        slice_img, slice_label = self.rand_trans(slice_img, slice_label)
        slice_img, slice_label = self.rand_rotate(slice_img, slice_label)
		slice_img = (slice_img- slice_img.min())/ (slice_img.max()-slice_img.min()) #min-max
   
        slice_img=slice_img.transpose([2,0,1])
		slice_label=slice_label[:,:,0]
        slice_img = torch.from_numpy(slice_img)
        slice_label=torch.from_numpy(slice_label)

        return fn,slice_img,slice_label

    def __len__(self):
        return len(self.slices)

