import time
import numpy as np
import pandas as pd
import copy
import os
from tqdm import tqdm
import torch

from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score


def train_model(device, model, model_dir, train_loader, val_loader, criterion, optimizer,scheduler, num_epochs, steps=None, s_patience=3, patience=15):
    model.to(device)

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    start_epoch, best_val_loss = load_checkpoint(model, optimizer, scheduler, model_dir)

    best_model_wts = copy.deepcopy(model.state_dict())
    epochs_without_improvement = 0

    for epoch in range(start_epoch, start_epoch + num_epochs):
        model.train()
        running_loss = 0.0

        print(f'Starting epoch {epoch}/{start_epoch + num_epochs - 1}')
        
        start_time = time.time() 

        for i, batch in enumerate(tqdm(train_loader, desc="Training")):
            if steps and (i >= steps):
                break

            images = batch['image'].to(device)
            labels = batch['labels'].to(device).float()

            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if isinstance(scheduler, torch.optim.lr_scheduler.CyclicLR):
                scheduler.step()


        train_time = time.time() - start_time  
        start_time_val = time.time() 

        val_loss, val_auc, val_precision, val_recall, val_f1 = validate_model(model, val_loader, criterion)

        val_time = time.time() - start_time_val 

        epoch_time = time.time() - start_time 

        print(f'Epoch [{epoch}/{num_epochs + start_epoch - 1}], Validation Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, '
              f'Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1-score: {val_f1:.4f}, '
              f'Training Time: {train_time:.2f}s, Validation Time: {val_time:.2f}s, Total Time: {epoch_time:.2f}s')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f'No improvement in validation loss for {epochs_without_improvement} epoch(s).')

        if epochs_without_improvement >= patience:
            print(f'Early stopping after {epochs_without_improvement} epochs without improvement.')
            break
        
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        current_lr = scheduler.optimizer.param_groups[0]['lr']

        current_history = pd.DataFrame({'epoch': [epoch],
                                        'val_loss': [val_loss],
                                        'val_auc': [val_auc],
                                        'precision': [val_precision],
                                        'recall': [val_recall],
                                        'f1_score': [val_f1],
                                        'lr': [current_lr],
                                        'train_time': [train_time],
                                        'val_time': [val_time],
                                        'epoch_time': [epoch_time]})
        
        current_history.to_csv(os.path.join(model_dir, 'history.csv'), mode='a', header=False, index=False)

        save_checkpoint(model, optimizer, scheduler, epoch, model_dir, best_val_loss)

    model.load_state_dict(best_model_wts)
    print('Training complete. Best Validation Loss:', best_val_loss)

    torch.save(model.state_dict(), os.path.join(model_dir, 'best_model.pth'))
    print(f'Best model saved to {os.path.join(model_dir, "best_model.pth")}')

    return model

def validate_model(model, val_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    val_loss = 0.0
    all_outputs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            images = batch['image'].to(device)
            labels = batch['labels'].to(device).float()

            outputs = model(images)
            
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            all_outputs.append(torch.sigmoid(outputs).cpu().detach().numpy())
            all_labels.append(labels.cpu().detach().numpy())

    val_loss /= len(val_loader) 
    all_outputs = np.concatenate(all_outputs) 
    all_labels = np.concatenate(all_labels) 
    all_preds = (all_outputs > 0.5).astype(int)

    auc_scores = []
    for i in range(all_labels.shape[1]):
        if np.unique(all_labels[:, i]).size > 1: 
            auc = roc_auc_score(all_labels[:, i], all_outputs[:, i])
            auc_scores.append(auc)
        else:
            auc_scores.append(np.nan)

    mean_auc = np.nanmean(auc_scores)

    precision = precision_score(all_labels, all_preds, average='micro', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='micro')
    f1 = f1_score(all_labels, all_preds, average='micro')

    return val_loss, mean_auc, precision, recall, f1  

def save_checkpoint(model, optimizer, scheduler, epoch, model_dir, best_val_loss):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'best_val_loss': best_val_loss
    }
    torch.save(checkpoint, os.path.join(model_dir, 'checkpoint.pth'))
    print(f'Model checkpoint saved at epoch {epoch}.')

def load_checkpoint(model, optimizer, scheduler, model_dir):
    checkpoint_path = os.path.join(model_dir, 'checkpoint.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}.")
        return checkpoint['epoch'] + 1, checkpoint['best_val_loss']
    else:
        print("No checkpoint found, starting from scratch.")
        return 1, float('inf')