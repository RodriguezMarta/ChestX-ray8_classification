import torch.nn as nn
from torchvision import models

def get_resnet50(num_classes):
    model = models.resnet50(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
