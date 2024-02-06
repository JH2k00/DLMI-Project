from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import v2
import pandas as pd 
import os
import torch.nn.functional as F
import torch
import numpy as np
from feature_extraction import calc_allfeatures
from tqdm import tqdm
import matplotlib.pyplot as plt


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

class KC_Dataset_preloaded(Dataset):
    def __init__(self, start_idx, end_idx, csv_path, dataset_path, transforms=None, drop_data_till_balanced=False, simclr=False):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.dataset_path = dataset_path
        self.transforms = transforms
        self.simclr = simclr
        if(self.simclr):
            assert self.transforms is not None,"We need transformations when using simclr"
        self.normalize = v2.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) # ImageNet mean and std
        if(drop_data_till_balanced):
            N_blood = np.sum(self.df["Label"]==1)
            N_drop = len(self.df) - 2*N_blood
            drop_indices = np.random.choice(self.df[self.df["Label"]==0].index, N_drop, replace=False)
            self.df = self.df.drop(drop_indices).reset_index().drop(columns="index")
            start_idx=0
            end_idx = len(self.df)
        self.x = []
        self.y = []
        for index in tqdm(range(start_idx, end_idx)):
            filename, label = self.df.iloc[index]
            image = read_image(os.path.join(self.dataset_path, filename)).float() / 255
            self.x.append(image)
            self.y.append(label)
              
    def __getitem__(self, index):
        image = self.x[index]
        label = self.y[index]
        if(self.simclr):
            image_aug = self.transforms(image)
            image_all = torch.cat([image, image_aug], dim=2)
            plt.figure(dpi=600)
            plt.imshow(image_all.permute(1,2,0).numpy())
            plt.axis('off')
            plt.savefig(r"images/OriginalvsAug_%d.pdf"%index, bbox_inches='tight')
            plt.show()
            view1 = self.normalize(self.transforms(image))
            view2 = self.normalize(self.transforms(image))
            return view1, view2, label
        else:
            if(self.transforms is not None):
                image = self.transforms(image)
            image = self.normalize(image)
            return image, F.one_hot(torch.tensor(label), num_classes=2).float()
    
    def __len__(self):
        return len(self.x)

class KC_Dataset_Features():
    def __init__(self, start_idx, end_idx, csv_path, dataset_path):
        self.df = pd.read_csv(csv_path)
        self.dataset_path = dataset_path
        self.normalize = v2.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]) # ImageNet mean and std
        self.x = []
        self.y = []
        for index in tqdm(range(start_idx, end_idx)):
            filename, label = self.df.iloc[index]
            image = self.normalize(read_image(os.path.join(self.dataset_path, filename)).float() / 255)
            self.x.append(calc_allfeatures(image.numpy()))
            self.y.append(F.one_hot(torch.tensor(label), num_classes=2).float().numpy())
        self.x = np.stack(self.x, axis=0)
        self.y = np.stack(self.y, axis=0)

    def __getitem__(self, index):
        image = self.x[index, :]
        label = self.y[index, :]
        return image, label
    
    def __len__(self):
        return len(self.x)

def get_dataloader(csv_path, dataset_path, batch_size, use_preloaded, start_idx, end_idx, transforms=None, drop_data_till_balanced=False, simclr=False, **kwargs):
    if(use_preloaded):
        dataset = KC_Dataset_preloaded(start_idx=start_idx, end_idx=end_idx, csv_path=csv_path, dataset_path=dataset_path, transforms=transforms, drop_data_till_balanced=drop_data_till_balanced, simclr=simclr)
        dataloader = DataLoader(dataset, batch_size=batch_size, **kwargs)
    else:
        dataset = KC_Dataset(csv_path=csv_path, dataset_path=dataset_path, transforms=transforms, drop_data_till_balanced=drop_data_till_balanced)
        dataloader = DataLoader(dataset, batch_size=batch_size, **kwargs)
    return dataloader

if __name__ == "__main__":
    dataloader = get_dataloader(csv_path=r"C:\Users\JadHa\Desktop\Uni\DLMI-Project\kvasir-capsule-labeled-images\dataset_train.csv", dataset_path=r"C:\Users\JadHa\Desktop\Uni\DLMI-Project\kvasir-capsule-labeled-images\labelled_images", batch_size=32, shuffle=True)
    images, labels = next(iter(dataloader))
    print(len(dataloader.dataset))
    print(images.shape)
    print(labels.shape)