import torch
import torch.nn as nn
import torch.optim as optim

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Hacer la predicción
        outputs = model(inputs)

        # Calcular la pérdida
        loss = criterion(outputs, labels)

        # Retropropagar los gradientes
        loss.backward()

        # Actualizar los pesos
        optimizer.step()

        # Acumular el loss
        running_loss += loss.item()

        # Calcular la precisión (aproximada para multilabel)
        # Usamos un umbral de 0.5 para decidir si una clase está presente o no
        predicted = (torch.sigmoid(outputs) > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / (total * labels.size(1))  # Precisión multilabel
    return avg_loss, accuracy