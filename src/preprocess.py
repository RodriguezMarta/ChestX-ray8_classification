import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import cv2

# Funciones de procesamiento
def undersample_negatives(df, labels, ratio=0.3):
    negative = df[df[labels].sum(axis=1) == 0]
    positive = df[df[labels].sum(axis=1) > 0]
    print(f'Negativos antes del undersampling: {len(negative)}')

    negative_resampled = resample(negative, replace=False,
                                  n_samples=int(len(positive) * (ratio / (1 - ratio))),
                                  random_state=42)

    print(f'Negativos después del undersampling: {len(negative_resampled)}')
    return pd.concat([positive, negative_resampled])

def apply_smote(df, labels):
    X = df[labels].values
    y = np.argmax(X, axis=1)
    
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    
    resampled_df = pd.DataFrame(X_res, columns=labels)
    return resampled_df

def augment_images(df, images_dir, processed_dir, labels, augmentation_ratio=2.0):
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2,
                                  height_shift_range=0.2, shear_range=0.2,
                                  zoom_range=0.2, horizontal_flip=True,
                                  fill_mode='nearest')

    augmented_images, augmented_labels = [], []
    for idx, row in df.iterrows():
        image_path = os.path.join(images_dir, row['Image Index'])
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.resize(image, (256, 256))
                image = image.reshape((1,) + image.shape)

                i = 0
                for batch in datagen.flow(image, batch_size=1, save_to_dir=processed_dir,
                                          save_prefix='aug', save_format='png'):
                    augmented_images.append(batch[0])
                    augmented_labels.append(row[labels].values)
                    i += 1
                    if i >= augmentation_ratio:
                        break
    return augmented_images, pd.DataFrame(augmented_labels, columns=labels)

def preprocess_data(metadata_file, images_dir, processed_dir, output_dir, undersample_ratio=0.3, augmentation_ratio=2.0):
    labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion',
              'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule',
              'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

    metadata = pd.read_csv(metadata_file)
    for label in labels:
        metadata[label] = metadata['Finding Labels'].apply(lambda x: 1 if label in x else 0)
    
    metadata.drop(columns=['Finding Labels', 'Follow-up #', 'Patient Age', 'Patient Gender'], inplace=True)
    
    gss = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_val_idx, test_idx = next(gss.split(metadata, groups=metadata['Patient ID']))
    train_val_metadata, test_metadata = metadata.iloc[train_val_idx], metadata.iloc[test_idx]

    train_idx, val_idx = next(GroupShuffleSplit(test_size=0.125, n_splits=1, random_state=42)
                              .split(train_val_metadata, groups=train_val_metadata['Patient ID']))
    train_metadata, val_metadata = train_val_metadata.iloc[train_idx], train_val_metadata.iloc[val_idx]

    # Undersampling
    train_metadata = undersample_negatives(train_metadata, labels, undersample_ratio)
    val_metadata = undersample_negatives(val_metadata, labels, undersample_ratio)

    # SMOTE (Opcional)
    train_metadata_resampled = apply_smote(train_metadata, labels)
    val_metadata_resampled = apply_smote(val_metadata, labels)

    # Data Augmentation
    augment_images(train_metadata, images_dir, processed_dir, labels, augmentation_ratio)

    # Guardar conjuntos
    train_metadata_resampled.to_csv(os.path.join(output_dir, 'train_metadata.csv'), index=False)
    val_metadata_resampled.to_csv(os.path.join(output_dir, 'val_metadata.csv'), index=False)
    test_metadata.to_csv(os.path.join(output_dir, 'test_metadata.csv'), index=False)


