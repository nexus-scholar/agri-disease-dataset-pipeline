"""
MobileNetV3-Small definition.
"""
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

def get_mobilenet_v3_small(num_classes: int, pretrained: bool = True):
    """
    Returns MobileNetV3-Small model.
    """
    weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    
    # Replace classifier head
    # MobileNetV3 Small classifier structure:
    # (classifier): Sequential(
    #   (0): Linear(in_features=576, out_features=1024, bias=True)
    #   (1): Hardswish()
    #   (2): Dropout(p=0.2, inplace=True)
    #   (3): Linear(in_features=1024, out_features=1000, bias=True)
    # )
    
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    
    return model
