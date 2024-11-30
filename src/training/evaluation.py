# src/training/evaluator.py
import torch
from sklearn.metrics import accuracy_score
from torch import nn

def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.sigmoid(outputs).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # Métricas
    accuracy = accuracy_score(all_labels, all_preds.round())
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy
