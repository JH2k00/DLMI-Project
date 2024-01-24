from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import v2
import pandas as pd 
import os
import torch.nn.functional as F
import torch
import numpy as np


class KC_Dataset(Dataset):
    def __init__(self, csv_path, dataset_path, transforms=None, drop_data_till_balanced=False):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        if(drop_data_till_balanced):
            N_blood = np.sum(self.df["Label"]==1)
            N_drop = len(self.df) - 2*N_blood
            drop_indices = np.random.choice(self.df[self.df["Label"]==0].index, N_drop, replace=False)
            self.df = self.df.drop(drop_indices).reset_index().drop(columns="index")
        self.dataset_path = dataset_path
        self.transforms = transforms
        self.normalize = v2.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) # ImageNet mean and std
              
    def __getitem__(self, index):
        filename, label = self.df.iloc[index]
        image = read_image(os.path.join(self.dataset_path, filename)).float() / 255
        if(self.transforms is not None):
            image = self.transforms(image)
        image = self.normalize(image)
        return image, F.one_hot(torch.tensor(label), num_classes=2).float()

    def __len__(self):
        return len(self.df)

def get_dataloader(csv_path, dataset_path, batch_size, transforms=None, drop_data_till_balanced=False, **kwargs):
    dataset = KC_Dataset(csv_path=csv_path, dataset_path=dataset_path, transforms=transforms, drop_data_till_balanced=drop_data_till_balanced)
    dataloader = DataLoader(dataset, batch_size=batch_size, **kwargs)
    return dataloader

if __name__ == "__main__":
    dataloader = get_dataloader(csv_path=r"C:\Users\JadHa\Desktop\Uni\DLMI-Project\kvasir-capsule-labeled-images\dataset_train.csv", dataset_path=r"C:\Users\JadHa\Desktop\Uni\DLMI-Project\kvasir-capsule-labeled-images\labelled_images", batch_size=32, shuffle=True)
    images, labels = next(iter(dataloader))
    print(len(dataloader.dataset))
    print(images.shape)
    print(labels.shape)