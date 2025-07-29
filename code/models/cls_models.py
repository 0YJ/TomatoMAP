#!/usr/bin/env python3
"""
Classification model definitions for TomatoMAP
"""

import torch.nn as nn
from torchvision import models
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    MobileNet_V2_Weights,
    ResNet18_Weights,
)


def get_model(name, num_classes, pretrained=True):
    print(f"Building model: {name}, classes: {num_classes}")
    
    if name == 'mobilenet_v3_large':
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        
    elif name == 'mobilenet_v3_small':
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        
    elif name == 'mobilenet_v2':
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        
    elif name == 'resnet18':
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        
    else:
        raise ValueError(f"Model {name} not supported")
    
    # print parameter information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters - Total: {total_params:,}, Trainable: {trainable_params:,}")
    
    return model
