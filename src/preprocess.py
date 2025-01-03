import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os

def undersample_negatives(df, labels, ratio=0.3):
    
    negative = df[df[labels].sum(axis=1) == 0]
    positive = df[df[labels].sum(axis=1) > 0]

    print(f'Negatives before undersampling {len(negative)}')

    negative = resample(negative, 
                        replace=False,
                        n_samples=int(len(positive)*(ratio / (1- ratio))),
                        random_state=42)
    
    print(f'Negatives before undersampling {len(negative)}')
    
    return pd.concat([positive, negative])

def main(args):
    #Get the labels and read the original metadata
    labels = ['Atelectasis',
            'Cardiomegaly',
            'Consolidation',
            'Edema',
            'Effusion',
            'Emphysema',
            'Fibrosis',
            'Hernia',
            'Infiltration',
            'Mass',
            'Nodule',
            'Pleural_Thickening',
            'Pneumonia',
            'Pneumothorax']
    metadata_dir = os.path.join('..', 'data','metadata')
    metadata_file = os.path.join(metadata_dir, 'Data_Entry_2017_v2020.csv')
    metadata = pd.read_csv(metadata_file, delimiter=',')

    #Encode the labels with multi-label friendly encoding
    for label in labels:
        metadata[label] = metadata['Finding Labels'].apply(lambda x: 1 if label in x else 0)

    metadata = metadata.drop(columns=['Finding Labels', 'Follow-up #','Patient Age', 'Patient Gender', 'View Position', 'OriginalImage[Width','Height]', 'OriginalImagePixelSpacing[x', 'y]'])

    #Get the test train and val splits according to the patient ID so no patients end up split between groups
    gss_test = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
    train_val_idx, test_idx = next(gss_test.split(metadata, groups=metadata['Patient ID']))

    train_val_metadata = metadata.iloc[train_val_idx]
    test_metadata = metadata.iloc[test_idx]

    gss_train_val = GroupShuffleSplit(test_size=0.125, n_splits=1, random_state=42)
    train_idx, val_idx = next(gss_train_val.split(train_val_metadata, groups=train_val_metadata['Patient ID']))

    train_metadata = train_val_metadata.iloc[train_idx]
    val_metadata = train_val_metadata.iloc[val_idx]


    #Drop the column of patient ID
    train_metadata = train_metadata.drop(columns=['Patient ID'])
    val_metadata = val_metadata.drop(columns=['Patient ID'])
    test_metadata = test_metadata.drop(columns=['Patient ID'])


    #Undersample "No Findings"
    print(f'Applying a ratio of undersample of: {args}')
    train_metadata = undersample_negatives(train_metadata,labels,args)
    val_metadata = undersample_negatives(val_metadata,labels,args)
    test_metadata = undersample_negatives(test_metadata,labels,args)


    #Write all the new metadata as csv to load easier
    train_metadata_file = os.path.join(metadata_dir,'train_metadata.csv')
    train_metadata.to_csv(train_metadata_file, index=False)
    val_metadata_file = os.path.join(metadata_dir, 'val_metadata.csv')
    val_metadata.to_csv(val_metadata_file, index=False)
    test_metadata_file = os.path.join(metadata_dir,'test_metadata.csv')
    test_metadata.to_csv(test_metadata_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing to the NIH dataset metadata for the model")
    parser.add_argument('--ratio', type=float, default=0.3, help='Ratio of positive to negative samples for undersampling (default: 0.3)')
    args = parser.parse_args()

    main(args.ratio)