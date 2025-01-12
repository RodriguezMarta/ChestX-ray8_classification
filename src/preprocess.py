import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2
from multiprocessing import Pool


def load_metadata(metadata_file):
    """
    Load the CSV file with metadata and one-hot encoding.
    """
    metadata = pd.read_csv(metadata_file)
    labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
              'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
              'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

    # One-hot encoding
    for label in labels:
        metadata[label] = metadata['Finding Labels'].apply(lambda x: 1 if label in x else 0)
    
    metadata.drop(columns=['Finding Labels', 'Follow-up #', 'Patient Age', 'Patient Gender'], inplace=True)
    
    return metadata, labels


def split_data(metadata, test_size=0.2, val_size=0.1):
    """
    Split the data into training, validation, and test sets.
    """
    gss = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=42)
    train_val_idx, test_idx = next(gss.split(metadata, groups=metadata['Patient ID']))
    train_val_metadata, test_metadata = metadata.iloc[train_val_idx], metadata.iloc[test_idx]

    train_idx, val_idx = next(GroupShuffleSplit(test_size=val_size, n_splits=1, random_state=42)
                              .split(train_val_metadata, groups=train_val_metadata['Patient ID']))
    train_metadata, val_metadata = train_val_metadata.iloc[train_idx], train_val_metadata.iloc[val_idx]
    
    return train_metadata, val_metadata, test_metadata


def undersample_negatives(df, labels, ratio=0.3):
    """
    Apply undersampling to negative classes.
    """
    negative = df[df[labels].sum(axis=1) == 0]
    positive = df[df[labels].sum(axis=1) > 0]
    print(f'Negatives before undersampling: {len(negative)}')

    negative_resampled = resample(negative, replace=False,
                                  n_samples=int(len(positive) * (ratio / (1 - ratio))),
                                  random_state=42)

    print(f'Negatives after undersampling: {len(negative_resampled)}')
    return pd.concat([positive, negative_resampled])


def augment_image(args):
    """
    Helper function to augment a single image.
    """
    image_path, labels, row, processed_dir, datagen, augmentation_ratio = args
    augmented_images = []
    augmented_labels = []
    augmented_image_indexes = []
    
    if os.path.exists(image_path):
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.resize(image, (224, 224))
            image = image.reshape((1,) + image.shape)
            
            i = 0
            for batch in datagen.flow(image, batch_size=1, save_to_dir=processed_dir,
                                      save_prefix='aug', save_format='png'):
                augmented_images.append(batch[0])
                augmented_labels.append(row[labels].values)
                augmented_image_indexes.append('aug_' + row['Image Index'])
                i += 1
                if i >= augmentation_ratio:
                    break
    
    return augmented_images, augmented_labels, augmented_image_indexes


def batch_data_augmentation(df, images_dir, processed_dir, labels, augmentation_ratio=2.0, batch_size=500):
    """
    Perform data augmentation in batches to avoid memory overload.
    """
    datagen = ImageDataGenerator(zoom_range=0.1, horizontal_flip=True, fill_mode='nearest')
    
    augmented_images = []
    augmented_labels = []
    augmented_image_indexes = []

    for start_idx in range(0, len(df), batch_size):
        batch_df = df.iloc[start_idx:start_idx + batch_size]
        
        args = [(os.path.join(images_dir, row['Image Index']), labels, row, processed_dir, datagen, augmentation_ratio)
                for idx, row in batch_df.iterrows()]

        with Pool(processes=4) as pool:
            results = pool.map(augment_image, args)
        
        for res in results:
            augmented_images.extend(res[0])
            augmented_labels.extend(res[1])
            augmented_image_indexes.extend(res[2])

    augmented_df = pd.DataFrame(augmented_labels, columns=labels)
    augmented_df['Image Index'] = augmented_image_indexes

    return augmented_images, augmented_df


def preprocess_data_in_batches(metadata_file, images_dir, processed_dir, output_dir, 
                               undersample_ratio=0.3, augmentation_ratio=2.0, batch_size=500):
    """
    Preprocess data incrementally using batching for efficient memory management.
    """
    metadata, labels = load_metadata(metadata_file)

    train_metadata, val_metadata, test_metadata = split_data(metadata)

    train_metadata = undersample_negatives(train_metadata, labels, undersample_ratio)
    val_metadata = undersample_negatives(val_metadata, labels, undersample_ratio)

    augmented_images, augmented_df = batch_data_augmentation(train_metadata, images_dir, processed_dir, labels, 
                                                              augmentation_ratio, batch_size)

    combined_train_metadata = pd.concat([train_metadata, augmented_df], ignore_index=True)

    combined_train_metadata.to_csv(os.path.join(output_dir, 'train_metadata.csv'), index=False)
    val_metadata.to_csv(os.path.join(output_dir, 'val_metadata.csv'), index=False)
    test_metadata.to_csv(os.path.join(output_dir, 'test_metadata.csv'), index=False)

