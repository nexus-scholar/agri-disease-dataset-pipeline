"""
Evaluation metrics including Accuracy, F1, and Healthy False Positive Rate.
"""
import torch
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

def calculate_accuracy(outputs, targets):
    """Computes top-1 accuracy."""
    _, preds = torch.max(outputs, 1)
    correct = (preds == targets).sum().item()
    return correct / targets.size(0)

def calculate_metrics(outputs, targets, average='macro'):
    """Computes Accuracy and F1 Score."""
    _, preds = torch.max(outputs, 1)
    preds = preds.cpu().numpy()
    targets = targets.cpu().numpy()
    
    acc = np.mean(preds == targets)
    f1 = f1_score(targets, preds, average=average, zero_division=0)
    
    return acc, f1

def calculate_healthy_fpr(outputs, healthy_class_idx):
    """
    Calculates Healthy False Positive Rate (FPR) for a batch of KNOWN NOISE/BACKGROUND images.
    
    Definition: Of all the "Background/Soil" images, what percentage did the model classify as "Disease"?
    
    Args:
        outputs: Logits or probabilities from the model for the noise images.
        healthy_class_idx: The index of the "Healthy" class.
        
    Returns:
        FPR: Fraction of images predicted as any Disease class (i.e., NOT Healthy).
    """
    _, preds = torch.max(outputs, 1)
    preds = preds.cpu().numpy()
    
    # If prediction is NOT Healthy, it is a False Positive (Disease)
    # We want the model to predict "Healthy" (or reject) for these noise images.
    # In this specific metric definition, "Healthy" is the safe fallback.
    
    false_positives = (preds != healthy_class_idx)
    fpr = np.mean(false_positives)
    return fpr
