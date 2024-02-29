from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd
from pathlib import Path
import torch
import torchvision
torchvision.disable_beta_transforms_warning()
from torchvision.transforms import InterpolationMode, v2
from torchvision.io import read_image
import lightning as L
from tqdm import tqdm
import numpy as np

torch.manual_seed(1)

train_transforms = v2.Compose([
    v2.Grayscale(1),
    v2.RandomErasing(scale=(0.01, 0.1)),
    v2.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), interpolation=InterpolationMode.BILINEAR),
    v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 5))]),
    v2.ColorJitter(brightness=.1, contrast=.1),
    v2.Resize(size=(224, 224), antialias=True),
    v2.ToDtype(torch.float32),
    # v2.Normalize(mean=[148.23449176623407], std=[18.071799304221553]), # Smoker
    v2.Normalize(mean=[148.48609513022734], std=[18.1603103145614]), # All
])
test_transforms = v2.Compose([
    v2.Grayscale(1),
    v2.Resize(size=(224, 224), antialias=True),
    v2.ToDtype(torch.float32),
    # v2.Normalize(mean=[148.23449176623407], std=[18.071799304221553]), # Smoker
    v2.Normalize(mean=[148.48609513022734], std=[18.1603103145614]), # All
])


class PLCO(Dataset):
    def __init__(self, transforms, csv_path, only_smokers):
        """_summary_

        Args:
            transforms (_type_): _description_
            csv_path (_type_): _description_
            only_smokers (bool, optional): _description_. Defaults to True.
        """
        # Transforms
        self.transforms = transforms
        # Read the csv file
        self.data_info = pd.read_csv(csv_path) # plco_id/pack_year/sex/age
        self.data_info["cig_stat"] = self.data_info["pack_years"] > 0
        self.only_smokers = only_smokers
        if only_smokers:
            self.data_info = self.data_info[self.data_info["cig_stat"]]

    def __getitem__(self, index):
        # Get image name from the pandas df
        data = self.data_info.iloc[index]
        # pack_years,sex,age,image_file_name = 
        # Open image
        # if not isinstance(image_file_name, str):
        #     print(self.data_info.iloc[index])
        img = read_image(str(Path('data/image') / data["image_file_name"]))
        tab = torch.FloatTensor([data["sex"] - 1, data["age"]]) # 0: Man, 1: Woman

        if self.only_smokers:
            return (tab, self.transforms(img)), torch.FloatTensor([data["pack_years"]])
        else:
            return (tab, self.transforms(img)), torch.FloatTensor([data["cig_stat"]])

    def __len__(self):
        return len(self.data_info.index)
    
    def weights(self):
        assert self.only_smokers is False
        # 22920(non) / 34059(smokers)
        return (1./self.data_info.groupby('cig_stat')['cig_stat'].transform('count')).to_numpy()


class PLCODataModule(L.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int, only_smokers: bool):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.only_smokers = only_smokers
        
    def train_dataloader(self):
        dataset = PLCO(train_transforms, self.data_dir / "train.csv", only_smokers=self.only_smokers)
        weights = dataset.weights()
        sampler = None if self.only_smokers else WeightedRandomSampler(weights=weights, num_samples=len(weights))
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True if sampler is None else False, num_workers=self.num_workers, pin_memory=True, sampler=sampler)

    def val_dataloader(self):
        return DataLoader(PLCO(test_transforms, self.data_dir / "val.csv", only_smokers=self.only_smokers), batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(PLCO(test_transforms, self.data_dir / "test.csv", only_smokers=self.only_smokers), batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

if __name__ == "__main__":
    # Call dataset
    paths = pd.read_csv("data/label/train.csv")
    # paths = paths[paths["pack_years"] > 0] # smoker
    paths = paths["image_file_name"].values.tolist()
    transforms = v2.Compose([
        v2.Grayscale(1),
        v2.Resize(size=(224, 224), antialias=True),
    ])
    values = []
    for path in tqdm(paths):
        img = read_image(str(Path('data/image') / path))
        img = transforms(img).numpy()
        values.append(img.mean())
    values = np.array(values)
    print("Mean : ", values.mean())
    print("Std : ", values.std())