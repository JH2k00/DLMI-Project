import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import Trainer



class CustomDataset(ImageFolder):
    def __init__(self, csv_file, root_dir, transform=None):
        # Adjust the labels in the dataset to be binary (Blood/Other)
        df = pd.read_csv(csv_file)
        df['label'] = df['label'].apply(lambda x: 1 if x.startswith('Blood') else 0)
        self.root_dir = root_dir
        self.transform = transform
        super(CustomDataset, self).__init__(root=self.root_dir, transform=self.transform)

    def find_classes(self, directory):
        # Override the method to use our custom labels from the CSV
        return ['Blood', 'Other'], {x: i for i, x in enumerate(['Blood', 'Other'])}

class ImageClassifier(pl.LightningModule):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        # Use a pre-trained ResNet model
        

def train_model(csv_file, root_dir, batch_size=32):
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Create dataset
    dataset = CustomDataset(csv_file=csv_file, root_dir=root_dir, transform=transform)

    # Data loader
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    
    # Model
    #model = ImageClassifier()

    # Trainer
    #trainer = Trainer(max_epochs=10)
    #trainer.fit(model, train_loader)


# Paths to the image directories and CSV files
image_dir = '/home/farzaneh/Downloads/DLMI/project/labelled_images'  # Update with the path to our images
csv_file_1 = '/home/farzaneh/Downloads/DLMI/project/split_0.csv'
csv_file_2 = '/home/farzaneh/Downloads/DLMI/project/split_1.csv'

# Train on the first fold
train_model(csv_file_1, image_dir)

# Train on the second fold
train_model(csv_file_2, image_dir)
