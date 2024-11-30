import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class ChestXray8Dataset(Dataset):
    def __init__(self, img_dir, metadata_file, split_file=None, mode='train', transform=None):
        self.img_dir = img_dir
        self.metadata = pd.read_csv(metadata_file)
        if split_file is not None:
            with open(split_file, 'r') as f:
                split_list = [line.strip() for line in f.readlines()]
            self.metadata = self.metadata[self.metadata['Image Index'].isin(split_list)]
        self.mode = mode
        self.transform = transform or transforms.Compose([transforms.ToTensor()])
        self.label_mapping = {
            "Atelectasis": 0,
            "Consolidation": 1,
            "Infiltration": 2,
            "Pneumothorax": 3,
            "Edema": 4,
            "Emphysema": 5,
            "Fibrosis": 6,
            "Effusion": 7,
            "Pneumonia": 8,
            "Pleural_Thickening": 9,
            "Cardiomegaly": 10,
            "Nodule": 11,
            "Mass": 12,
            "Hernia": 13
        }

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.metadata.iloc[idx, 0])
        image = Image.open(img_name).convert("RGB")  # Convert to grayscale
        raw_labels = self.metadata.iloc[idx, 1]
        labels = raw_labels.split('|')
        label_tensor = torch.zeros(14)
        if 'No Finding' not in labels:
            for label in labels:
                if label in self.label_mapping:
                    label_idx = self.label_mapping[label]
                    label_tensor[label_idx] = 1
        if self.transform:
            image = self.transform(image)
        return image, label_tensor
# from pathlib import Path
# data_dir = Path.cwd().parent / 'data'
# images_dir = data_dir / 'extracted_images' / 'images'
# metadata_dir = data_dir / 'Data_Entry_2017_v2020.csv'
# train_list_path = data_dir / 'train_val_list.txt'
# test_list_path = data_dir / 'test_list.txt'
# preprocessed_dir = data_dir / 'processed/'  # Directory to save processed embeddings

# train_dataset = ChestXray8Dataset(
#     img_dir=images_dir, 
#     metadata_file=metadata_dir, 
#     split_file=train_list_path,
#     mode='train',  # Training mode
# )
# # Create the DataLoader for training
# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# for images, labels in train_loader:
#     print(images.shape, labels.shape)