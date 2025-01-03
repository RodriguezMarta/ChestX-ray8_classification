import torch
from torch.utils.data import Dataset
import os
import numpy as np
import pandas as pd
from PIL import Image

class ChestXRayDataset(Dataset):
    def __init__(self, data_frame, image_dir, transform=None):
        self.classes = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
                        'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
                        'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']
        
        self.data_frame = data_frame
        self.image_dir = image_dir
        self.transform = transform

        self.size = len(self.data_frame)
        selected_columns = self.data_frame[self.classes]
        self.labels = np.array(selected_columns)
        self.images = self.data_frame['Image Index']
        self.class_count = self.labels.sum(0)
        self.total_labels = self.class_count.sum()

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.images.iloc[idx])
        image = Image.open(img_name).convert("RGB")
        
        labels = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)

        return {'image': image, 'labels': labels}
    
    def __len__(self):
        return self.size
