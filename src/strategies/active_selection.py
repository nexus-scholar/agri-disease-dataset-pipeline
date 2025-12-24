"""
Active Learning Strategies.
"""
import numpy as np
import torch
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader
from tqdm import tqdm

def get_embeddings(model, loader, device):
    """Extracts embeddings (penultimate layer) from the model."""
    model.eval()
    embeddings = []
    indices = []
    
    # Hook to get features before classifier
    features = []
    def hook(module, input, output):
        features.append(output.flatten(start_dim=1))
    
    # MobileNetV3 Small: classifier[0] is the pooling/flattened layer input usually
    # Or we can hook onto the avgpool layer.
    # model.avgpool is AdaptiveAvgPool2d(1).
    handle = model.avgpool.register_forward_hook(hook)
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(tqdm(loader, desc="Extracting embeddings")):
            inputs = inputs.to(device)
            _ = model(inputs)
            # features[-1] contains the batch features
            embeddings.append(features[-1].cpu().numpy())
            # We need to track which sample corresponds to which embedding if loader is shuffled
            # But usually for AL selection we use a non-shuffled loader.
            
    handle.remove()
    return np.vstack(embeddings)

def kmeans_sampling(model, dataset, unlabeled_indices, budget, device):
    """
    Selects samples using K-Means clustering on embeddings.
    1. Extract embeddings for all unlabeled data.
    2. Cluster into K=budget clusters.
    3. Select sample nearest to each centroid.
    """
    subset = torch.utils.data.Subset(dataset, unlabeled_indices)
    loader = DataLoader(subset, batch_size=64, shuffle=False, num_workers=4)
    
    embeddings = get_embeddings(model, loader, device)
    
    # K-Means
    kmeans = KMeans(n_clusters=budget, random_state=42, n_init=10)
    kmeans.fit(embeddings)
    
    # Find nearest neighbors to centroids
    centers = kmeans.cluster_centers_
    selected_indices_local = []
    
    from sklearn.metrics import pairwise_distances_argmin_min
    closest, _ = pairwise_distances_argmin_min(centers, embeddings)
    
    # Map back to global indices
    selected_indices_global = [unlabeled_indices[i] for i in closest]
    
    return selected_indices_global

def random_sampling(unlabeled_indices, budget):
    """Randomly selects samples."""
    return np.random.choice(unlabeled_indices, size=budget, replace=False).tolist()

def entropy_sampling(model, dataset, unlabeled_indices, budget, device):
    """Selects samples with highest entropy."""
    subset = torch.utils.data.Subset(dataset, unlabeled_indices)
    loader = DataLoader(subset, batch_size=64, shuffle=False, num_workers=4)
    
    model.eval()
    uncertainties = []
    
    with torch.no_grad():
        for inputs, _ in tqdm(loader, desc="Calculating entropy"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum(dim=1)
            uncertainties.extend(entropy.cpu().numpy())
            
    # Select top-k entropy
    top_k_indices = np.argsort(uncertainties)[-budget:]
    selected_indices_global = [unlabeled_indices[i] for i in top_k_indices]
    
    return selected_indices_global
