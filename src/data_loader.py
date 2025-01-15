from PIL import Image
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms, models
import os
class ChestXRayDataset(Dataset):
    def __init__(self, df_dir, image_dir, transform=None):
        self.dataframe = pd.read_csv(df_dir)
        self.image_dir = image_dir
        self.labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
              'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
              'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

        self.transform = transform
        

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.dataframe.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        label = torch.tensor(self.dataframe.iloc[idx, 1:].values, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label