import torch
def test(model, test_loader, criterion, device):
    model.eval()  # Establecer el modelo en modo de evaluación
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Desactivar el cálculo de gradientes
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Hacer la predicción
            outputs = model(inputs)

            # Calcular la pérdida
            loss = criterion(outputs, labels)

            # Acumular el loss
            running_loss += loss.item()

            # Calcular la precisión
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(test_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy
