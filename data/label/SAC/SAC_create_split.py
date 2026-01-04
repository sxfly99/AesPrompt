"""
File: SAC_create_split.py
Created: 2024/6/17
Author: Xiangfei
Description: 按照8:2的比例划分SAC数据集
"""

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

def clean_path(path):
    # 替换非法字符
    return re.sub(r'[<>:"/\\|?*\n]', '_', path)

class SimulacraDataset(data.Dataset):
    """Simulacra dataset
    Args:
        images_dir: directory
        transform: preprocessing and augmentation of the training images
    """

    def __init__(self, images_dir, db, transform=None, split='train', split_file='split.txt', test_ratio=0.2):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.db = db
        self.split = split
        self.split_file = split_file
        self.test_ratio = test_ratio
        self.ratings = self._load_ratings()
        self.train_indices, self.test_indices = self._train_test_split()
        self.indices = self.train_indices if split == 'train' else self.test_indices

    def _load_ratings(self):
        conn = sqlite3.connect(self.db)
        ratings = []
        for row in conn.execute('SELECT generations.id, images.idx, paths.path, AVG(ratings.rating) FROM images JOIN generations ON images.gid=generations.id JOIN ratings ON images.id=ratings.iid JOIN paths ON images.id=paths.iid GROUP BY images.id'):
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

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, key):
        idx = self.indices[key]
        gid, idx, filename, rating = self.ratings[idx]
        filename = clean_path(filename)
        image_path = os.path.join(self.images_dir, filename)
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            raise
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(rating)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--batch-size', '-bs', type=int, default=64,
                   help='the CLIP model')
    p.add_argument('--clip-model', type=str, default='ViT-B/16',
                   help='the CLIP model')
    p.add_argument('--db', type=str, default='sac_public_2022_06_29.sqlite',
                   help='the database location')
    p.add_argument('--images-dir', type=str, default=r'D:\Database\sac\home\jdp\simulacra-aesthetic-captions',
                   help='the dataset images directory')
    p.add_argument('--num-workers', type=int, default=0,
                   help='the number of data loader workers')
    p.add_argument('--output', type=str, default='data',
                   help='the output file')
    p.add_argument('--start-method', type=str, default='spawn',
                   choices=['fork', 'forkserver', 'spawn'],
                   help='the multiprocessing start method')
    p.add_argument('--split-file', type=str, default='sac_split.txt',
                   help='the file to save the train/test split')
    args = p.parse_args()

    mp.set_start_method(args.start_method)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    clip_model, clip_tf = clip.load(args.clip_model, device=device, jit=False)
    clip_model = clip_model.eval().requires_grad_(False)

    train_dataset = SimulacraDataset(args.images_dir, args.db, transform=clip_tf, split='train', split_file=args.split_file)
    test_dataset = SimulacraDataset(args.images_dir, args.db, transform=clip_tf, split='test', split_file=args.split_file)

    train_loader = data.DataLoader(train_dataset, args.batch_size, num_workers=args.num_workers)
    test_loader = data.DataLoader(test_dataset, args.batch_size, num_workers=args.num_workers)

    # for split_name, loader in zip(['train', 'test'], [train_loader, test_loader]):
    #     embeds, ratings = [], []
    #     for batch in tqdm(loader, desc=f'Processing {split_name} split'):
    #         images_batch, ratings_batch = batch
    #         embeds.append(clip_model.encode_image(images_batch.to(device)).cpu())
    #         ratings.append(ratings_batch.clone())
    #
    #     obj = {'clip_model': args.clip_model,
    #            'embeds': torch.cat(embeds),
    #            'ratings': torch.cat(ratings)}
    #
    #     torch.save(obj, f'{args.output}_{split_name}.pt')


if __name__ == '__main__':
    main()

