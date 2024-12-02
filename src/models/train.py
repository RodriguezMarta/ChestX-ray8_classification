import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import time

def train_model(model, train_loader, num_epochs, lr,device):
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss()  # Para clasificación multiclase
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in num_epochs:
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Poner a cero los gradientes de los optimizadores
            optimizer.zero_grad()

            # Hacer la predicción
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Retropropagar los gradientes
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Medir precisión
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}, Accuracy: {100*correct/total}%")
    
    print("Entrenamiento finalizado.")

def evaluate_model(model, val_loader,device):
    """
    Evaluación del modelo.
    """
    model.eval()  # Establecer el modelo en modo evaluación
    correct = 0
    total = 0
    model = model.to(device)
    
    with torch.no_grad():  # Desactivar el cálculo de gradientes
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Precisión en el conjunto de validación: {100*correct/total}%")

def test_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())
            
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct_preds += (preds == labels).sum().item()
            total_preds += labels.size(0)
    
    avg_loss = running_loss / len(dataloader)
    accuracy = correct_preds / total_preds
    return avg_loss, accuracy

