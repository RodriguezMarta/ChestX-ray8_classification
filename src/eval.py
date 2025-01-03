import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score
import seaborn as sns

from tqdm import tqdm

from utils import Find_Optimal_Cutoff, ensure_dir


def evaluate_model(device, model, test_loader, criterion, model_dir, use_static_threshold=False):
    
    model_path = os.path.join(model_dir, 'best_model.pth')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No model found at {model_path}. Please train the model before evaluation.")
    
    model.load_state_dict(torch.load(model_path))
    print('Model loaded succesfully and proceeding to eval')

    model.to(device)

    model.eval()
    
    val_loss = 0.0
    all_outputs = []
    all_labels = []
    classes = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion', 
               'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass', 'Nodule', 
               'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

    results_dir = os.path.join(model_dir, 'results')
    ensure_dir(results_dir)

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            images = batch['image'].to(device)
            labels = batch['labels'].to(device).float()

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            all_outputs.append(torch.sigmoid(outputs).cpu().detach().numpy())
            all_labels.append(labels.cpu().detach().numpy())

    val_loss /= len(test_loader)
    all_outputs = np.concatenate(all_outputs)
    all_labels = np.concatenate(all_labels)

    all_preds = np.zeros_like(all_outputs)

    print('Obtaining Cutoff for Predictions')
    
    for i in range(all_labels.shape[1]):
        if use_static_threshold:
            threshold = 0.5
        else:
            threshold = Find_Optimal_Cutoff(all_labels[:, i], all_outputs[:, i])
        all_preds[:, i] = (all_outputs[:, i] > threshold).astype(int)
        
    print('Plotting and storing')

    plt.figure(figsize=(10, 8))
    auc_scores = []

    for i, class_name in enumerate(classes):
        if np.unique(all_labels[:, i]).size > 1:
            fpr, tpr, _ = roc_curve(all_labels[:, i], all_outputs[:, i])
            auc = roc_auc_score(all_labels[:, i], all_outputs[:, i])
            auc_scores.append(auc)
            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.2f})')

    plt.title('ROC Curves per Class')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(results_dir, 'roc_curves.png'))
    plt.close()

    for i, class_name in enumerate(classes):
        cm = confusion_matrix(all_labels[:, i], all_preds[:, i])
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
        plt.title(f'Confusion Matrix for {class_name}')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(results_dir, f'confusion_matrix_{class_name}.png'))
        plt.close()

    print('Classification report')

    report = classification_report(all_labels, all_preds, target_names=classes, output_dict=True, zero_division=0)

    accuracy_per_class = [accuracy_score(all_labels[:, i], all_preds[:, i]) for i in range(all_labels.shape[1])]
    mean_accuracy = np.mean(accuracy_per_class)
    mean_auc = np.nanmean(auc_scores) 

    print(accuracy_per_class)
    print(auc_scores)

    print('Micro')

    precision_micro = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall_micro = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_micro = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    print(f'Validation Loss: {val_loss:.4f}')
    print(f'Average Accuracy per Class: {mean_accuracy:.4f}')
    print(f'Mean AUROC: {mean_auc:.4f}')
    print(f'Weighted-Averaged Precision: {precision_micro:.4f}')
    print(f'Weighted-Averaged Recall: {recall_micro:.4f}')
    print(f'Weighted-Averaged F1-Score: {f1_micro:.4f}')

    report_path = os.path.join(results_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f'Validation Loss: {val_loss:.4f}\n')
        f.write(f'Average Accuracy per Class: {mean_accuracy:.4f}\n')
        f.write(f'Mean AUROC: {mean_auc:.4f}\n')
        f.write(f'Micro-Averaged Precision: {precision_micro:.4f}\n')
        f.write(f'Micro-Averaged Recall: {recall_micro:.4f}\n')
        f.write(f'Micro-Averaged F1-Score: {f1_micro:.4f}\n\n')
        f.write('Classification Report per Class:\n')
        for class_name, metrics in report.items():
            f.write(f'{class_name}:\n')
            for metric_name, metric_value in metrics.items():
                f.write(f'  {metric_name}: {metric_value:.4f}\n')

    return val_loss, mean_auc, precision_micro, recall_micro, f1_micro