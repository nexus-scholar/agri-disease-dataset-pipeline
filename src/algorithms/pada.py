"""
Partial Adversarial Domain Adaptation (PADA) Logic.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DomainDiscriminator(nn.Module):
    def __init__(self, in_feature=1024, hidden_size=1024):
        super(DomainDiscriminator, self).__init__()
        self.layer1 = nn.Linear(in_feature, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.bn1(self.layer1(x)))
        x = self.relu2(self.bn2(self.layer2(x)))
        x = self.sigmoid(self.layer3(x))
        return x

def pada_loss(source_features, target_features, domain_discriminator, class_weights, alpha=1.0):
    """
    Computes PADA loss.
    """
    # Domain labels
    source_domain_label = torch.zeros(source_features.size(0)).float().to(source_features.device)
    target_domain_label = torch.ones(target_features.size(0)).float().to(target_features.device)
    
    # Reverse Gradient
    source_features_rev = ReverseLayerF.apply(source_features, alpha)
    target_features_rev = ReverseLayerF.apply(target_features, alpha)
    
    # Discriminator output
    source_pred = domain_discriminator(source_features_rev)
    target_pred = domain_discriminator(target_features_rev)
    
    # Weighted Target Loss (Partial Adaptation)
    # We weight the target samples by their predicted class probability (averaged over batch)
    # But PADA usually weights by the class importance.
    # Here we assume class_weights are passed (gamma in the paper).
    
    loss_s = F.binary_cross_entropy(source_pred.squeeze(), source_domain_label)
    loss_t = F.binary_cross_entropy(target_pred.squeeze(), target_domain_label, weight=class_weights)
    
    return loss_s + loss_t

class ClassWeightAccumulator:
    def __init__(self, num_classes, decay=0.9):
        self.class_weight = torch.ones(num_classes).cuda()
        self.decay = decay

    def update(self, target_logits, target_pred_domain):
        """
        target_logits: output of classifier on target data
        target_pred_domain: output of discriminator on target data
        """
        with torch.no_grad():
            softmax_out = nn.Softmax(dim=1)(target_logits)
            # Weight = Avg probability of this class, weighted by how "source-like" the domain D thinks it is
            # Actually standard PADA just averages the softmax outputs
            batch_weight = torch.mean(softmax_out, dim=0)
            self.class_weight = self.decay * self.class_weight + (1 - self.decay) * batch_weight
            
    def get_weights(self):
        # Normalize so max is 1 (avoid vanishing gradients)
        w = self.class_weight / torch.max(self.class_weight)
        return w.detach()
