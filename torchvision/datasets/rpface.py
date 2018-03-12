#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 12:02:33 2018

@author: huangyin
"""

import torch.utils.data as data

from PIL import Image
import os


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


class RPFaceFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::

        root/xxx.png
        root/xxy.png
        root/xxz.png

        root/train.txt
            xxx.png 1
            xxy.png 1
            xxz.png 2
        root/test.txt

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        imgs (list): List of image path
        labels (list): List  of class
    """
    def __init__(self, root, train=True, transform=None, target_transform=None, loader = default_loader):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        self.loader = loader
        if self.train:
            with open(os.path.join(root,'train.txt')) as input_file:
                lines = input_file.readlines()
        else:
            with open(os.path.join(root,'test.txt')) as input_file:
                lines = input_file.readlines()            
        self.imgs = [os.path.join(self.root, line.strip().split('\t')[0]) for line in lines]
        self.labels = [int(line.strip().split('\t')[-1]) for line in lines]
    

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index],self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str