import argparse
import os
from pathlib import Path
import sqlite3
import random

from PIL import Image

import torch
from torch import multiprocessing as mp
from torch.utils import data
import torchvision.transforms as transforms
from tqdm import tqdm

import clip
import re
from torch.utils.data.dataloader import default_collate
import json
def clean_path(path):
    # 替换非法字符
    return re.sub(r'[<>:"/\\|?*\n]', '_', path)

# CLIP normalize
IMAGE_NET_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGE_NET_STD = [0.26862954, 0.26130258, 0.27577711]
normalize = transforms.Normalize(
            mean=IMAGE_NET_MEAN,
            std=IMAGE_NET_STD)
def custom_collate_fn(batch):
    # 过滤掉 batch 中的 None 数据
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None
    return default_collate(batch)

class SAC_cap(data.Dataset):

    def __init__(self, images_dir, db, if_train, split='train', split_file=None, test_ratio=0.2, json_file=None):
        self.images_dir = Path(images_dir)
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

        self.db = db
        self.split = split
        self.split_file = split_file
        self.test_ratio = test_ratio
        self.ratings = self._load_ratings()
        self.train_indices, self.test_indices = self._train_test_split()
        self.indices = self.train_indices if split == 'train' else self.test_indices

        # Load captions from JSON file
        self.captions = self._load_captions(json_file)

    def _load_ratings(self):
        conn = sqlite3.connect(self.db)
        ratings = []
        for row in conn.execute(
                'SELECT generations.id, images.idx, paths.path, AVG(ratings.rating) FROM images JOIN generations ON images.gid=generations.id JOIN ratings ON images.id=ratings.iid JOIN paths ON images.id=paths.iid GROUP BY images.id'):
            ratings.append(row)
        conn.close()
        return ratings

    def _train_test_split(self):
        if os.path.exists(self.split_file):
            with open(self.split_file, 'r') as f:
                train_indices = list(map(int, f.readline().strip().split(',')))
                test_indices = list(map(int, f.readline().strip().split(',')))
        else:
            indices = list(range(len(self.ratings)))
            random.shuffle(indices)
            split_point = int(len(indices) * (1 - self.test_ratio))
            train_indices = indices[:split_point]
            test_indices = indices[split_point:]
            with open(self.split_file, 'w') as f:
                f.write(','.join(map(str, train_indices)) + '\n')
                f.write(','.join(map(str, test_indices)) + '\n')
        return train_indices, test_indices

    def _load_captions(self, json_file):
        captions = {}
        if json_file:
            with open(json_file, 'r') as f:
                data = json.load(f)
                for item in data:
                    captions[item['image_id']] = item['caption']
        return captions

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, key):
        try:
            idx = self.indices[key]
            gid, idx, filename, rating = self.ratings[idx]
            clean_filename = clean_path(filename)
            image_path = os.path.join(self.images_dir, clean_filename)
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Error opening image {image_path}: {e}")
                raise

            x = self.transform(image)

            # Get caption from the loaded captions dictionary
            caption = self.captions.get(filename, "No caption available")

            return x, torch.tensor(rating) / 10, caption
        except Exception as e:
            print(f"Error processing item {key}: {e}")
            return None  # 返回 None 表示该数据点无法加载