#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 15:13:47 2018

@author: huangyin
"""
import torch
from torch.utils.data import Dataset
import numpy as np
import random

def RandomChoice(pair,index):
    #random choice pair of img and label
    img0_tuple = random.choice(pair)[index]
    #we need to make sure approx 50% of images are in the same class
    should_get_same_class = random.randint(0,1) 
    if should_get_same_class:
        while True:
            #keep looping till the same class image is found
            img1_tuple = random.choice(pair) 
            if img0_tuple[1]==img1_tuple[1]:
                break
    else:
        img1_tuple = random.choice(pair)
    
    img0 = img0_tuple[0]
    img1 = img1_tuple[0]
    return img0,img1,torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))

def Misplace(pair,index,misplace):
    
    img0_tuple = pair[(index)%len(pair)]
    img1_tuple = pair[(index+misplace)%len(pair)]
    img0 = img0_tuple[0]
    img1 = img1_tuple[0]
    
    return img0,img1,torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32))
        

class SiameseDataset(Dataset):
    """`Creat Data struct for Siamese Network top on normal Dataset, like mnist.`

    Args:
        root (string): normal Dataset (data,label)
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        misplace : for rpface
    """
    
    def __init__(self,root,transform=None,target_transform=None,misplace=0):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.misplace = misplace
        self.len = len(root)
        
    def __getitem__(self,index):
        if self.misplace==0: 
            img0,img1,label = RandomChoice(self.root,index)
        else:
            img0,img1,label = Misplace(self.root,self.misplace,index)
            
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1 , label
    
    def __len__(self):
        return self.len