import os
from torchvision import transforms
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
import torch
from PIL import Image
from torchvision.transforms import Normalize
import pickle
import random
import json
IMAGE_NET_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGE_NET_STD = [0.26862954, 0.26130258, 0.27577711]
normalize = transforms.Normalize(
            mean=IMAGE_NET_MEAN,
            std=IMAGE_NET_STD)

class AVA_zs_cap(Dataset):
    def __init__(self, path_to_csv, images_path, if_train, json_file=None):
        self.df = pd.read_csv(path_to_csv,encoding='ISO-8859-1')
        self.images_path = images_path
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
        scores_names = [f'score{i}' for i in range(2, 12)]
        y = np.array([row[k] for k in scores_names])
        p = y / y.sum()

        image_id = int(row['img_id'])
        # image_name = str(row['img_id'])
        image_path = os.path.join(self.images_path, f'{image_id}.jpg')
        image = default_loader(image_path)
        # Get caption from the loaded captions dictionary
        captions = self.captions.get(image_id, "No caption available")
        x = self.transform(image)
        return x, p.astype('float32'), captions

class AVA_level_cap(Dataset):
    def __init__(self, path_to_csv, images_path, if_train, json_file=None):
        self.df = pd.read_csv(path_to_csv, encoding='ISO-8859-1')
        self.images_path = images_path
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
        scores_names = ['high', 'mid', 'low']
        p = np.array([row[k] for k in scores_names])

        image_id = int(row['image_id'])
        # Get caption from the loaded captions dictionary
        # image_name = str(row['image_id'])
        caption = self.captions.get(image_id, "No caption available")
        image_path = os.path.join(self.images_path, f'{image_id}.jpg')
        # image = default_loader(image_path)
        image = Image.open(image_path).convert('RGB')
        x = self.transform(image)

        return x, p.astype('float32'), caption