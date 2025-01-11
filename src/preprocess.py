import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2



def load_metadata(metadata_file):
    """
    Load the CSV file with metadata and one-hot encoding
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

def split_data(metadata, test_size=0.2, val_size=0.125):
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
                                  n_samples=int(len(positive) * (ratio / (1 - ratio)) ),
                                  random_state=42)

    print(f'Negatives after undersampling: {len(negative_resampled)}')
    return pd.concat([positive, negative_resampled])



def data_augmentation(df, images_dir, processed_dir, labels, augmentation_ratio=2.0):
    """
    Perform image augmentation and link augmented images back to metadata.
    Each augmented image will have the original Image Index as a reference.
    """
    datagen = ImageDataGenerator(zoom_range=0.1, horizontal_flip=True,
                                  fill_mode='nearest')

    augmented_images = []
    augmented_labels = []
    augmented_image_indexes = []
    
    for idx, row in df.iterrows():
        image_path = os.path.join(images_dir, row['Image Index'])
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.resize(image, (256, 256))  # Resize to fixed size
                image = image.reshape((1,) + image.shape)  # Reshape for the generator
                
                i = 0
                for batch in datagen.flow(image, batch_size=1, save_to_dir=processed_dir,
                                          save_prefix='aug', save_format='png'):
                    # Append the augmented images and their corresponding labels
                    augmented_images.append(batch[0])
                    augmented_labels.append(row[labels].values)
                    augmented_image_indexes.append('aug_' + row['Image Index'])  # Save the original Image Index with prefix
                    i += 1
                    if i >= augmentation_ratio:  # Create only the desired number of augmentations
                        break
    
    # Create a DataFrame for the augmented images with the labels
    augmented_df = pd.DataFrame(augmented_labels, columns=labels)
    augmented_df['Image Index'] = augmented_image_indexes  # Add the original Image Index with prefix
    
    return augmented_images, augmented_df


def preprocess_data(metadata_file, images_dir, processed_dir, output_dir, undersample_ratio=0.3, augmentation_ratio=2.0):
    """
    Preprocess the data, including splitting, undersampling, SMOTE, and image augmentation.
    """
    # Load metadata and labels
    metadata, labels = load_metadata(metadata_file)
    
    # Split the data
    train_metadata, val_metadata, test_metadata = split_data(metadata)
    
    # Apply undersampling
    train_metadata = undersample_negatives(train_metadata, labels, undersample_ratio)
    val_metadata = undersample_negatives(val_metadata, labels, undersample_ratio)
    
    # Perform image augmentation and link to metadata
    augmented_images, augmented_df = data_augmentation(train_metadata, images_dir, processed_dir, labels, augmentation_ratio)
    
    # Combine the original and augmented metadata into one DataFrame
    combined_train_metadata = pd.concat([train_metadata, augmented_df], ignore_index=True)
    
    # Save the results to CSV files
    combined_train_metadata.to_csv(os.path.join(output_dir, 'train_metadata.csv'), index=False)
    val_metadata.to_csv(os.path.join(output_dir, 'val_metadata.csv'), index=False)
    test_metadata.to_csv(os.path.join(output_dir, 'test_metadata.csv'), index=False)