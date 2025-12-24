"""
FixMatch Algorithm Implementation.
"""
import torch
import torch.nn.functional as F

class FixMatchLoss(object):
    def __init__(self, threshold=0.95, lambda_u=1.0):
        self.threshold = threshold
        self.lambda_u = lambda_u

    def __call__(self, logits_u_w, logits_u_s):
        """
        logits_u_w: Logits of Weakly augmented unlabeled images
        logits_u_s: Logits of Strongly augmented unlabeled images
        """
        # Generate pseudo-labels using weak augmentation
        with torch.no_grad():
            probs_u_w = torch.softmax(logits_u_w, dim=1)
            max_probs, targets_u = torch.max(probs_u_w, dim=1)
            mask = max_probs.ge(self.threshold).float()

        # Compute loss on strong augmentation
        loss_u = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()
        
        return loss_u * self.lambda_u, mask
