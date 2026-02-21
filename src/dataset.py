import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import numpy as np
from utils import create_mask


class MultiLabelDataset(Dataset):
    def __init__(self, image_names, labels_dict, images_dir, transform=None):
        self.image_names = image_names
        self.labels_dict = labels_dict
        self.images_dir = Path(images_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        image_name = self.image_names[idx]
        image_path = self.images_dir / image_name
        
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        labels = np.array(self.labels_dict[image_name], dtype=np.float32)
        mask = create_mask(labels)
        labels = np.where(labels == -1, 0, labels)
        labels = torch.tensor(labels, dtype=torch.float32)
        
        return {
            'image': image,
            'labels': labels,
            'mask': mask,
            'image_name': image_name
        }
