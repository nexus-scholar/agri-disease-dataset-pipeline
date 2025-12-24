"""
Active Learning Selection Strategies.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from tqdm import tqdm

def random_sampler(dataset, budget, seed=42):
    """Randomly selects indices."""
    np.random.seed(seed)
    indices = np.random.choice(len(dataset), budget, replace=False)
    return indices.tolist()

def entropy_sampler(model, dataset, budget, device):
    """Selects samples with highest entropy (uncertainty)."""
    model.eval()
    uncertainties = []
    indices = []
    
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(tqdm(loader, desc="Entropy Calculation")):
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = torch.softmax(logits, dim=1)
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum(dim=1)
            
            uncertainties.extend(entropy.cpu().numpy())
            start_idx = i * loader.batch_size
            indices.extend(range(start_idx, start_idx + inputs.size(0)))
            
    # Select top-k highest entropy
    top_k_indices = np.argsort(uncertainties)[-budget:]
    return [indices[i] for i in top_k_indices]

def kmeans_sampler(model, dataset, budget, device, seed=42):
    """
    Selects samples using K-Means clustering on the feature space.
    Ensures diversity by picking centroids.
    """
    print("Extracting embeddings for K-Means Selection...")
    model.eval()
    embeddings = []
    indices = []
    
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    with torch.no_grad():
        for i, (inputs, _) in enumerate(tqdm(loader, desc="Embedding Extraction")):
            inputs = inputs.to(device)
            # Extract features (before classifier)
            # Assuming MobileNetV3 structure
            if hasattr(model, 'features') and hasattr(model, 'avgpool'):
                features = model.features(inputs)
                features = model.avgpool(features)
                features = torch.flatten(features, 1)
            elif hasattr(model, 'fc'): # ResNet
                # Hook or modify forward needed, but for now let's assume we can get it
                # Simplified: just use logits as proxy if features hard to get (suboptimal)
                # Or better: Assume MobileNetV3 as per spec
                features = model.features(inputs)
                features = model.avgpool(features)
                features = torch.flatten(features, 1)
            else:
                raise ValueError("Model architecture not supported for feature extraction")
            
            embeddings.append(features.cpu().numpy())
            start_idx = i * loader.batch_size
            indices.extend(range(start_idx, start_idx + inputs.size(0)))
            
    embeddings = np.concatenate(embeddings, axis=0)
    
    print(f"Running KMeans on {embeddings.shape[0]} samples...")
    kmeans = KMeans(n_clusters=budget, random_state=seed)
    kmeans.fit(embeddings)
    
    # Find nearest neighbors to centroids
    selected_indices = []
    for center in kmeans.cluster_centers_:
        dists = np.linalg.norm(embeddings - center, axis=1)
        nearest_idx = np.argmin(dists)
        selected_indices.append(indices[nearest_idx])
        
    return selected_indices
