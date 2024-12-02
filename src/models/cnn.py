import torch
import torch.nn as nn
import torchvision.models as models

class ResNet50(nn.Module):
    def __init__(self, num_classes=14):
        super(ResNet50, self).__init__()
        # Cargar ResNet50 preentrenado
        self.resnet = models.resnet50(pretrained=True)
        # Cambiar la capa final para que tenga 14 salidas
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)
