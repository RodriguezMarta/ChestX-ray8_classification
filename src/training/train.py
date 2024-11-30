import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data.dataset import ChestXrayDataset
from src.models.cnn import get_resnet50

# src/training/trainer.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.data_preparation.dataset import ChestXrayDataset
from src.data_preparation.transforms import get_transforms
from src.models.resnet import ResNet50
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_model(csv_file, img_dir, train_list, test_list, model, num_epochs=10, batch_size=32, lr=0.001):
    transform = get_transforms()

    train_dataset = ChestXrayDataset(csv_file, img_dir, train_list, transform)
    test_dataset = ChestXrayDataset(csv_file, img_dir, test_list, transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Entrenamiento
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

    return model
