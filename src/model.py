import torch
import torch.nn as nn
import torchvision.models as models


def create_model(num_classes=4):
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    return model


class MultiLabelModel(nn.Module):
    def __init__(self, num_classes=4):
        super(MultiLabelModel, self).__init__()
        self.resnet = create_model(num_classes)
    
    def forward(self, x):
        return self.resnet(x)
