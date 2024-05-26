from PIL import Image
import torch
from torch.utils.data import Dataset
import logging
import torch
from torch.utils import data
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler
import os
from PIL import Image
import numpy as np
import albumentations as alb
import random
import nibabel as nib



class MyDataSet(Dataset):
    def __init__(self, root):
        imgs = os.listdir(root)
        self.imgs = [os.path.join(root, k) for k in imgs]
        np.random.shuffle(self.imgs)

    def __getitem__(self, index):
        img_path = self.imgs[index]
        label = img_path[-17]
        print(label)
        if label == 'A':
            label = 1
        else:
            label = 0

        data = np.array(nib.load(img_path).get_fdata(), dtype="float32")
        data = np.nan_to_num(data, neginf=0)
        data = self.normalization(data)
        data = torch.tensor(data)
        data = data[:-1, 4:-5, :-1]
        #data = data[4:-5, 9:-8, 4:-5]
        data = torch.unsqueeze(data, dim=0)
        return data, label

    def __len__(self):
        return len(self.imgs)

    def normalization(self, data):
        _range = np.max(data) - np.min(data)
        return (data - np.min(data)) / _range

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels
