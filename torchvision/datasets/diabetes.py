#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 14:46:21 2018

@author: huangyin
"""
import os
import torch
import numpy as np
from torch.utils.data import Dataset
class Diabetes(Dataset):
    """ `Diabetes` Dataset
    Args:
        root (string): Root directory of dataset where directory
            ``omniglot-py`` exists.
        train (bool, optional): If True, creates dataset from the "train" set, otherwise
            creates from the "test" set. 
        transform (callable, optional): A function/transform that  takes in the data
            and returns a transformed version. 
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset zip files from the internet and
            puts it in root directory. If the zip files are already downloaded, they are not
            downloaded again.
    """
 
    folder = 'diabetes'
    download_url_prefix = ' '
    zips_md5 = { }
   
    # Initialize your data, download, etc.
    def __init__(self,root,train=True,transform=None,target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        
        xy = np.loadtxt(root,delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, 0:-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
    
    
    
    
    
    