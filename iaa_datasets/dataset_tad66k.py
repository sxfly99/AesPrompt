"""
File: dataset_tad66k.py
Created: 2024/6/18
Author: Xiangfei
Description: 
"""
import os
from torchvision import transforms
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import json
IMAGE_NET_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGE_NET_STD = [0.26862954, 0.26130258, 0.27577711]
normalize = transforms.Normalize(
            mean=IMAGE_NET_MEAN,
            std=IMAGE_NET_STD)

class TAD66K_cap(Dataset):
    def __init__(self, path_to_csv, images_path,if_train, json_file=None):
        self.df = pd.read_csv(path_to_csv)
        self.images_path =  images_path
        self.if_train = if_train
        if if_train:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((224, 224)),
                transforms.ToTensor(),
                normalize])
        else:
            self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize])
        # Load captions from JSON file
        self.captions = self._load_captions(json_file)

    def __len__(self):
        return self.df.shape[0]
    def _load_captions(self, json_file):
        captions = {}
        if json_file:
            with open(json_file, 'r') as f:
                data = json.load(f)
                for item in data:
                    captions[item['image_id']] = item['caption']
        return captions

    def __getitem__(self, item):
        row = self.df.iloc[item]
        y = np.array([row['score'] / 10])
        image_id = row['image']
        image_path = os.path.join(self.images_path, f'{image_id}')
        image = default_loader(image_path)
        captions = self.captions.get(image_id, "No caption available")
        x = self.transform(image)
        return x, y.astype('float32'), captions