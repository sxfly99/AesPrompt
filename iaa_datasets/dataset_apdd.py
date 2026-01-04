import os
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import torch
import json
IMAGE_NET_MEAN = [0.48145466, 0.4578275, 0.40821073]
IMAGE_NET_STD = [0.26862954, 0.26130258, 0.27577711]
normalize = transforms.Normalize(
            mean=IMAGE_NET_MEAN,
            std=IMAGE_NET_STD)

def my_collate(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return default_collate(batch)

class APDD_cap(Dataset):
    def __init__(self, csv_file, root_dir, if_train, json_file=None):
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
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
        img_name = row['filename']
        img_path = str(os.path.join(self.root_dir, row['filename']))

        image = Image.open(img_path).convert('RGB')
        x = self.transform(image)
        label = row['Score'] / 10
        captions = self.captions.get(img_name, "No caption available")

        sample = {'image': x, 'label': torch.from_numpy(np.float64([label])).float(), 'caps':captions}
        return sample