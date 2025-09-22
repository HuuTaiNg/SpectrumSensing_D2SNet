from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import torch
import os
import numpy as np

class SemanticSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, transform=None, typeData="train"):  # typeData="val" for validation dataset
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform = transform
        self.image_paths = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir)])
        self.label_paths = sorted([os.path.join(label_dir, lbl) for lbl in os.listdir(label_dir)])
        self.class_colors = {
            (249, 249, 22): 0,     # LTE 
            (16, 190, 186): 1,     # 5G 
            (62, 38, 168): 2       # Noise 
        }
        self.typeData = typeData
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_path = self.label_paths[idx]
        label = cv2.imread(label_path)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        label_mask = np.zeros(label.shape[:2], dtype=np.uint8)
        for rgb, idx in self.class_colors.items():
            label_mask[np.all(label == rgb, axis=-1)] = idx

        image_name = os.path.basename(image_path)
        parts = image_name.split('_')
        if self.typeData=="train":
            parts[3] = '6'  
        elif self.typeData=="val":
            parts[4] = '6'  
        freenoise_name = '_'.join(parts)
        freenoise_path = os.path.join(self.image_dir, freenoise_name)

        image_freenoise = cv2.imread(freenoise_path)
        image_freenoise = cv2.cvtColor(image_freenoise, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image)
            image_freenoise = self.transform(image_freenoise)
            label_mask = torch.from_numpy(label_mask).long()

        return image, image_freenoise, label_mask

dataTransform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

def get_dataset(root_path):
    train_dataset = SemanticSegmentationDataset(
        image_dir=f"{root_path}\\train\\input",
        label_dir=f"{root_path}\\train\\label",
        transform=dataTransform,
        typeData="train"
    )

    val_dataset = SemanticSegmentationDataset(
        image_dir=f"{root_path}\\test\\input",
        label_dir=f"{root_path}\\test\\label",
        transform=dataTransform,
        typeData="val"
    )

    return train_dataset, val_dataset