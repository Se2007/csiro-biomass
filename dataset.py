from cv2 import transform
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
from torchvision import tv_tensors
import torch.nn.functional as F

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import os



########################


class CsiroDataset(Dataset):
    def __init__(
        self,
        samples_csv,
        targets_csv,
        image_root,
        transform=None
    ):
        self.samples = pd.read_csv(samples_csv)
        self.targets = pd.read_csv(targets_csv)
        self.image_root = image_root
        self.transform = transform

        # merge on sample_id
        self.data = self.samples.merge(
            self.targets,
            on="sample_id",
            how="inner"
        )

        self.target_columns = [
            "Dry_Clover_g",
            "Dry_Dead_g",
            "Dry_Green_g",
            "Dry_Total_g",
            "GDM_g"
        ]

    def __len__(self,):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # --- Image ---
        img_path = os.path.join(self.image_root, row["image_path"])
        image = Image.open(img_path).convert("RGB")
        print(image)

        if self.transform:
            image = self.transform(image)

        # --- Targets ---
        targets = torch.tensor(
            row[self.target_columns].astype(float).values,
            dtype=torch.float32
        )

        # --- Extra numeric features (optional) ---
        extras = torch.tensor(
            [float(row["Pre_GSHH_NDVI"]), float(row["Height_Ave_cm"])],
            dtype=torch.float32
        )

        return {
            "image": image,
            "targets": targets,
            "extras": extras,
            "sample_id": row["sample_id"]
        }
    
    


class BraTS20(object):
    def __init__(self, root, mode, mini=False, memory=True) :
        assert mode in ['train', 'valid', 'test'], 'mode should be train, test or valid'
        self.mini = mini  
        self.memory = memory

        if mode == 'train' :
            self.path_dataset = root + "/train.csv"
        elif mode == 'valid':
            self.path_dataset = root + "/val.csv"
        elif mode == 'test' :
            self.path_dataset = root + "/test.csv"

        # self.path_dataset = root


    def __call__(self, batch_size) :
        dataset = BraTS20_dataset(self.path_dataset, memory=self.memory)   #self.transform,

        if self.mini == False :
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
       

        elif self.mini == True:
            dataset,_ = random_split(dataset,(1000, len(dataset)-1000))
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        return data_loader



if __name__=='__main__':

    dataset = CsiroDataset(
    samples_csv="./csiro-biomass/samples.csv",
    targets_csv="./csiro-biomass/targets.csv",
    image_root="./csiro-biomass/",
    # transform=transform
)
   
    sample = dataset.__getitem__(0)
    print(sample["image"].show())
    print(sample["targets"])
    print(sample["extras"]) 
    print(sample["sample_id"])
   


     