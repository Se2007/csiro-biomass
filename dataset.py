from cv2 import transform
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torch.nn.functional as F

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
import os
from sklearn.preprocessing import LabelEncoder



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

        self.numeric_columns = [
            "Pre_GSHH_NDVI",
            "Height_Ave_cm"
        ]

        # force numeric
        self.data[self.target_columns + self.numeric_columns] = (
            self.data[self.target_columns + self.numeric_columns]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0.0)
        )

        # ---------------- Categorical: State ----------------
        self.state_encoder = LabelEncoder()
        self.data["state_enc"] = self.state_encoder.fit_transform(
            self.data["State"]
        )

        # ---------------- Categorical: Species ----------------
        self.species_encoder = LabelEncoder()
        self.data["species_enc"] = self.species_encoder.fit_transform(
            self.data["Species"]
        )

        # number of categories (for embeddings)
        self.num_states = len(self.state_encoder.classes_)
        self.num_species = len(self.species_encoder.classes_)


    def __len__(self,):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        # print(idx)

        # --- Image ---
        img_path = os.path.join(self.image_root, row["image_path"])
        image = Image.open(img_path).convert("RGB")
        # print(image)

        if self.transform:
            image = self.transform(image)

        # --- Targets ---
        targets = self.normalization(
            torch.tensor(row[self.target_columns].astype(float).values, dtype=torch.float32),
            mean=torch.tensor([ 6.6497, 12.0445, 26.6247, 45.3181, 33.2744]),  
            std=torch.tensor([12.1178, 12.4020, 25.4012, 27.9840, 24.9358])           
        )

        # --- Extra numeric features (optional) ---
        extras = self.normalization(
            torch.tensor([float(row["Pre_GSHH_NDVI"]), float(row["Height_Ave_cm"])], dtype=torch.float32),
            mean=torch.tensor([0.6574229691876751, 7.595985434173669]),    
            std=torch.tensor([0.1521422782849033, 10.285262364329933])
        )
    

        # ---------- Categorical ----------
        state = torch.tensor(row["state_enc"], dtype=torch.long)
        species = torch.tensor(row["species_enc"], dtype=torch.long)

        return image, targets, extras, state, species
    
    def normalization(self, data, mean, std):
        data = ( data - mean ) / ( std + 1e-6 )
        return data

    
    



class Csiro(object):
    def __init__(
        self,
        root,
        image_root,
        transform=None,
        valid_ratio=0.2,
        seed=42,
        mini=False
    ):
        self.samples_csv = root + "/samples.csv"
        self.targets_csv = root + "/targets.csv"
        self.image_root = image_root
        self.transform = transform
        self.valid_ratio = valid_ratio
        self.seed = seed
        self.mini = mini

    def __call__(self, batch_size):

        dataset = CsiroDataset(
            self.samples_csv,
            self.targets_csv,
            self.image_root,
            self.transform
        )

        self.num_species, self.num_states = dataset.num_species, dataset.num_states
        # print(f"Number of species: {self.num_species}")
        # print(f"Number of states  : {self.num_states}")

        # ---------- FIXED RANDOM SPLIT ----------
        total_len = len(dataset)
        val_len = int(total_len * self.valid_ratio)
        train_len = total_len - val_len

        generator = torch.Generator().manual_seed(self.seed)

        train_dataset, val_dataset = random_split(
            dataset,
            [train_len, val_len],
            generator=generator
        )
        

        if self.mini:
            mini_train_dataset, _ = random_split(
                train_dataset,
                (100, len(train_dataset) - 100),    
                generator=generator
            )

            mini_train_loader = DataLoader(
                mini_train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=True
            )

            return mini_train_loader

        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                pin_memory=True
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )

            return train_loader, val_loader




if __name__=='__main__':
    Transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(), 
    ])  


    '''
    dataset = CsiroDataset(
    samples_csv="./csiro-biomass/samples.csv",
    targets_csv="./csiro-biomass/targets.csv",
    image_root="./csiro-biomass/",
    # transform=transform
)
   
    sample = dataset.__getitem__(0)
    # print(sample["image"].show())

    print(f"Sample ID : {sample['sample_id']}")
    print(f"State     : {sample['state']}")
    print(f"Species   : {sample['species']}")

    print("\nTargets")
    print(sample["targets"])

    print("\nExtras")
    print(sample["extras"])

    print("\nState Encoder Classes")
    for i, cls in enumerate(dataset.state_encoder.classes_):
        print(f"  {i:02d} → {cls}")

    print("\nSpecies Encoder Classes")
    for i, cls in enumerate(dataset.species_encoder.classes_):
        print(f"  {i:02d} → {cls}")

    '''

    train_loader = Csiro(
                        root="./csiro-biomass",
                        image_root="./csiro-biomass/",
                        transform=Transform,
                        valid_ratio=0.2,
                        seed=42,
                        mini=True   
                        )(batch_size=6)
    sample = next(iter(train_loader))
    print(sample[0].shape)
    print(sample[1].shape)
    print(sample[2].shape)  
    print(sample[3].shape)
    print(sample[4].shape)