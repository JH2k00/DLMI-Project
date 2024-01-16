from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import v2
import pandas as pd 
import os
import torch.nn.functional as F
import torch


class KC_Dataset(Dataset):
    def __init__(self, csv_path, dataset_path, transforms=None):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.dataset_path = dataset_path
        self.transforms = transforms
        self.normalize = v2.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
              
    def __getitem__(self, index):
        filename, label = self.df.loc[index]
        image = read_image(os.path.join(self.dataset_path, filename)).float()
        image = self.normalize(image / 255)
        if(self.transforms is not None):
            image = self.transforms(image)
        return image, F.one_hot(torch.tensor(label), num_classes=2).float()

    def __len__(self):
        return len(self.df)

def get_dataloader(csv_path, dataset_path, batch_size, transforms=None, **kwargs):
    dataset = KC_Dataset(csv_path=csv_path, dataset_path=dataset_path, transforms=transforms)
    dataloader = DataLoader(dataset, batch_size=batch_size, **kwargs)
    return dataloader

if __name__ == "__main__":
    dataloader = get_dataloader(csv_path=r"C:\Users\JadHa\Desktop\Uni\DLMI-Project\kvasir-capsule-labeled-images\dataset_train.csv", dataset_path=r"C:\Users\JadHa\Desktop\Uni\DLMI-Project\kvasir-capsule-labeled-images\labelled_images", batch_size=32, shuffle=True)
    images, labels = next(iter(dataloader))
    print(len(dataloader.dataset))
    print(images.shape)
    print(labels.shape)