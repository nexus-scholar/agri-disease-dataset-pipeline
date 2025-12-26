"""
Main entry point for training and evaluation.
Implements the 3-Phase Protocol:
1. Source Pre-training
2. Active Selection (KMeans)
3. FixMatch Training with Partial Domain Adaptation
"""
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from pathlib import Path

from config import cfg
from src.data.dataset import PlantVillageDataset, PlantDocDataset, get_transform, TransformFixMatch
from src.models import get_mobilenet_v3_small
from src.algorithms.fixmatch import FixMatchLoss
from src.algorithms.pada import ClassWeightAccumulator
from src.algorithms.active_selection import kmeans_sampler, random_sampler
from src.utils.metrics import calculate_healthy_fpr
from sklearn.cluster import KMeans

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_source_only(model, loader, criterion, optimizer, device, epochs):
    """Phase 0: Train on Source Domain"""
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc=f"Source Pretrain Epoch {epoch+1}/{epochs}", leave=False)
        for inputs, targets in pbar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': correct/total})
            
    return model

def train_fixmatch(model, source_loader, labeled_target_loader, unlabeled_target_loader, 
                   optimizer, device, epochs, num_classes):
    """Phase 2: FixMatch Training"""
    fixmatch_loss_fn = FixMatchLoss(threshold=0.95)
    
    # Iterators for infinite looping
    iter_source = iter(source_loader)
    iter_labeled = iter(labeled_target_loader)
    iter_unlabeled = iter(unlabeled_target_loader)
    
    # Steps per epoch (based on source dataset size usually, or fixed)
    steps_per_epoch = len(source_loader)
    
    model.train()
    criterion_sup = nn.CrossEntropyLoss(ignore_index=-1)
    
    for epoch in range(epochs):
        pbar = tqdm(range(steps_per_epoch), desc=f"FixMatch Epoch {epoch+1}/{epochs}")
        total_loss = 0
        
        for _ in pbar:
            # 1. Get Source Batch
            try:
                inputs_s, targets_s = next(iter_source)
            except StopIteration:
                iter_source = iter(source_loader)
                inputs_s, targets_s = next(iter_source)
                
            # 2. Get Labeled Target Batch
            try:
                inputs_t, targets_t = next(iter_labeled)
            except StopIteration:
                iter_labeled = iter(labeled_target_loader)
                inputs_t, targets_t = next(iter_labeled)
                
            # 3. Get Unlabeled Target Batch (Weak + Strong Aug)
            try:
                (inputs_u_w, inputs_u_s), _ = next(iter_unlabeled)
            except StopIteration:
                iter_unlabeled = iter(unlabeled_target_loader)
                (inputs_u_w, inputs_u_s), _ = next(iter_unlabeled)
            
            # Move to device
            inputs_s, targets_s = inputs_s.to(device), targets_s.to(device)
            inputs_t, targets_t = inputs_t.to(device), targets_t.to(device)
            inputs_u_w, inputs_u_s = inputs_u_w.to(device), inputs_u_s.to(device)
            
            # Forward Pass
            # Concatenate all inputs to save forward passes (optional optimization)
            # Here we do separate for clarity
            logits_s = model(inputs_s)
            logits_t = model(inputs_t)
            logits_u_w = model(inputs_u_w)
            logits_u_s = model(inputs_u_s)
            
            # Losses
            loss_s = criterion_sup(logits_s, targets_s)
            loss_t = criterion_sup(logits_t, targets_t)
            loss_u, mask = fixmatch_loss_fn(logits_u_w, logits_u_s)
            
            # Total Loss (Weights from FixMatch paper: lambda_u=1.0)
            loss = loss_s + loss_t + loss_u
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': total_loss / (pbar.n + 1), 'mask': mask.mean().item()})

def evaluate(model, loader, device, healthy_class_idx=None):
    model.eval()
    correct = 0
    total = 0
    
    # For Healthy FPR calculation
    # We need a way to identify "Noise" images.
    # In our setup, PlantDocDataset with include_unknown=False (test set) only has known classes.
    # So we can't measure FPR on the test set unless the test set has noise.
    # The spec says: "Of all the 'Background/Soil' images in PlantDoc..."
    # We should probably have a separate evaluation on the "Unknown" split.
    
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Evaluating", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
            
    return correct / total

def evaluate_fpr(model, dataset, device, healthy_class_idx):
    """
    Evaluates Healthy False Positive Rate on the Noise/Unknown subset.
    """
    # Create a loader that ONLY loads the unknown/noise images
    # We can do this by filtering the dataset or creating a new one
    # Since PlantDocDataset with include_unknown=True loads everything,
    # we need to filter for targets == -1.
    
    # Efficient way: Create a Subset
    noise_indices = [i for i, (_, target) in enumerate(dataset.samples) if target == -1]
    if not noise_indices:
        print("No noise images found for FPR evaluation.")
        return 0.0
        
    noise_subset = Subset(dataset, noise_indices)
    loader = DataLoader(noise_subset, batch_size=32, shuffle=False, num_workers=4)
    
    model.eval()
    all_preds = []
    
    with torch.no_grad():
        for inputs, _ in tqdm(loader, desc="Evaluating FPR", leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            
    # Calculate FPR: Fraction of predictions that are NOT Healthy
    all_preds = np.array(all_preds)
    false_positives = (all_preds != healthy_class_idx)
    fpr = np.mean(false_positives)
    
    return fpr

def main():
    parser = argparse.ArgumentParser(description="PDA Experiment Runner")
    parser.add_argument("--mode", type=str, default="full_pipeline", 
                        choices=["source_only", "full_pipeline", "random_baseline"],
                        help="Training mode")
    args = parser.parse_args()
    
    set_seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running mode: {args.mode} on {device}")
    
    # --- Data Loading ---
    print("Loading datasets...")
    
    # Source Domain (PlantVillage)
    source_dataset = PlantVillageDataset(
        root_dir=cfg.PLANT_VILLAGE_DIR,
        transform=get_transform(train=True)
    )
    source_loader = DataLoader(
        source_dataset, 
        batch_size=cfg.BATCH_SIZE_SOURCE, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Target Domain (PlantDoc) - For Evaluation (Clean, Shared Classes Only)
    target_test_dataset = PlantDocDataset(
        root_dir=cfg.PLANT_DOC_DIR,
        transform=get_transform(train=False),
        split="test", # Use test split for final eval
        class_to_idx=source_dataset.class_to_idx,
        include_unknown=False # Only evaluate on known classes
    )
    target_test_loader = DataLoader(
        target_test_dataset,
        batch_size=cfg.BATCH_SIZE_TARGET_UNLABELED,
        shuffle=False,
        num_workers=4
    )
    
    # Target Domain - Candidate Pool for Active Selection (Shared Classes Only)
    # We simulate that we only label "Known" classes.
    target_candidate_dataset = PlantDocDataset(
        root_dir=cfg.PLANT_DOC_DIR,
        transform=get_transform(train=False), # No aug for selection
        split="train",
        class_to_idx=source_dataset.class_to_idx,
        include_unknown=True
    )
    
    # Target Domain - Unlabeled Pool for FixMatch (Includes Noise/Unknown)
    target_unlabeled_dataset = PlantDocDataset(
        root_dir=cfg.PLANT_DOC_DIR,
        transform=TransformFixMatch(), # Weak + Strong Aug
        split="train",
        class_to_idx=source_dataset.class_to_idx,
        include_unknown=True # CRITICAL: Include noise for training
    )
    
    # Target Domain - Noise Evaluation Set (For FPR)
    # We use the training split's noise for monitoring, or test split if available.
    # Assuming noise is in 'train' split mostly.
    target_noise_dataset = PlantDocDataset(
        root_dir=cfg.PLANT_DOC_DIR,
        transform=get_transform(train=False),
        split="train",
        class_to_idx=source_dataset.class_to_idx,
        include_unknown=True
    )

    # --- Model Setup ---
    num_classes = len(source_dataset.classes)
    model = get_mobilenet_v3_small(num_classes=num_classes, pretrained=True)
    model = model.to(device)
    
    # Identify Healthy Class Index
    healthy_class_name = "tomato_healthy" # Based on PlantVillage class names
    # Check actual class name in dataset
    # PlantVillage classes are like 'tomato_early_blight', 'tomato_healthy', etc.
    # We need to find the index.
    healthy_idx = source_dataset.class_to_idx.get(healthy_class_name)
    if healthy_idx is None:
        # Fallback or search
        for name, idx in source_dataset.class_to_idx.items():
            if "healthy" in name.lower():
                healthy_idx = idx
                break
    print(f"Healthy Class Index: {healthy_idx}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=cfg.LEARNING_RATE, momentum=cfg.MOMENTUM, weight_decay=cfg.WEIGHT_DECAY)
    
    # --- Phase 0: Source Pre-training ---
    print("\n=== Phase 0: Source Pre-training ===")
    if Path("source_pretrained.pth").exists():
        print("Loading pretrained source model...")
        model.load_state_dict(torch.load("source_pretrained.pth"))
    else:
        model = train_source_only(model, source_loader, criterion, optimizer, device, epochs=10)
        torch.save(model.state_dict(), "source_pretrained.pth")
        
    baseline_acc = evaluate(model, target_test_loader, device)
    print(f"Baseline Source-Only Accuracy: {baseline_acc:.4f}")
    
    if args.mode == "source_only":
        return

    # --- Phase 1: Active Selection ---
    print(f"\n=== Phase 1: Active Selection ({'Random' if args.mode == 'random_baseline' else 'KMeans'}) ===")
    
    if args.mode == "random_baseline":
        selected_indices = random_sampler(target_candidate_dataset, budget=50, seed=cfg.SEED)
    else:
        selected_indices = kmeans_sampler(model, target_candidate_dataset, budget=50, device=device)
    
    # Create Labeled Dataset from selection
    labeled_target_dataset = Subset(target_candidate_dataset, selected_indices)
    
    labeled_target_loader = DataLoader(
        labeled_target_dataset,
        batch_size=cfg.BATCH_SIZE_TARGET_LABELED,
        shuffle=True,
        num_workers=0
    )
    
    unlabeled_target_loader = DataLoader(
        target_unlabeled_dataset,
        batch_size=cfg.BATCH_SIZE_TARGET_UNLABELED * 3, # Reduced Mu from 7 to 3 for M1200 GPU
        shuffle=True,
        num_workers=4,
        persistent_workers=True
    )
    
    # --- Phase 2: FixMatch Training ---
    print("\n=== Phase 2: FixMatch Training ===")
    # Re-initialize optimizer for fine-tuning
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    
    train_fixmatch(
        model, 
        source_loader, 
        labeled_target_loader, 
        unlabeled_target_loader, 
        optimizer, 
        device, 
        epochs=5,
        num_classes=num_classes
    )
    
    final_acc = evaluate(model, target_test_loader, device)
    print(f"Final Accuracy after FixMatch: {final_acc:.4f}")
    
    # Calculate Healthy FPR
    if healthy_idx is not None:
        fpr = evaluate_fpr(model, target_noise_dataset, device, healthy_idx)
        print(f"Healthy False Positive Rate (FPR): {fpr:.4f}")
    
    torch.save(model.state_dict(), f"final_model_{args.mode}.pth")
    print("Done.")

if __name__ == "__main__":
    main()
