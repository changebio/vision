#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 17:16:17 2018

@author: huangyin
"""
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import os
import numpy as np
import torch
import random
from PIL import Image
import pickle
# Training settings
batch_size = 64

# MNIST Dataset
train_dsets = datasets.MNIST(root='./data/mnist/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
# Data Loader (Input Pipeline)
train_loader = DataLoader(dataset=train_dsets,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=4)

# CIFAR Dataset

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dsets = datasets.CIFAR10(root='./data', 
                            train=True,
                            transform=transform,
                            download=True)
train_loader = DataLoader(train_dsets, 
                          batch_size=batch_size,
                          shuffle=True, 
                          num_workers=4)



train_omniglot = Omniglot(root='./data/omniglot/',background=True,download=True)
    
       





folder_dataset = datasets.ImageFolder(root='./data/faces/training/')
siamese_dataset = SiameseDataset(root=folder_dataset,
                                        transform=transforms.Compose([transforms.Scale((100,100)),
                                                                      transforms.ToTensor()
                                                                      ]))
vis_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=8)



train = datasets.MNIST(
        root='./Github/data/',
        train=True,
        # transform=transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        # ]),
        download=True
    )

siamese_dataset = SiameseNetworkDataset(root=train,
                                        transform=transforms.ToTensor())
vis_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=8)