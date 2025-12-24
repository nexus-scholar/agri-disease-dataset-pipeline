"""
Dataset definitions for Source (PlantVillage) and Target (PlantDoc) domains.
Handles loading, transformations, and splitting for Partial Domain Adaptation.
"""
import json
import os
from pathlib import Path
from typing import Callable, Optional, Tuple, List, Dict

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# Standard ImageNet normalization
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

class BaseDataset(Dataset):
    """Base class for image datasets."""
    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        self.classes: List[str] = []
        self.class_to_idx: Dict[str, int] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, target = self.samples[idx]
        img_path = self.root_dir / path
        
        try:
            with open(img_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            img = Image.new('RGB', (224, 224))

        if self.transform:
            img = self.transform(img)
            
        return img, target

class PlantVillageDataset(BaseDataset):
    """
    Source Domain Dataset (Laboratory Conditions).
    Contains all 10 Tomato classes.
    """
    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        super().__init__(root_dir, transform)
        self._load_metadata()

    def _load_metadata(self):
        metadata_path = self.root_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        # Filter for Tomato classes only
        self.classes = sorted([c for c in metadata['classes'].keys() if c.startswith('tomato_')])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # Load samples
        labels_csv = self.root_dir / "labels.csv"
        import csv
        with open(labels_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                label = row['label']
                if label in self.class_to_idx:
                    filename = row['filename']
                    rel_path = Path(label) / filename
                    self.samples.append((str(rel_path), self.class_to_idx[label]))

class PlantDocDataset(BaseDataset):
    """
    Target Domain Dataset (Field Conditions).
    Contains subset of 6 common classes.
    """
    # Mapping from PlantDoc class names to PlantVillage class names (Source)
    CLASS_MAPPING = {
        "tomato_early_blight": "tomato_early_blight",
        "tomato_late_blight": "tomato_late_blight",
        "tomato_septoria_spot": "tomato_septoria_spot",
        "tomato_mosaic_virus": "tomato_mosaic_virus",
        "tomato_yellow_virus": "tomato_yellow_curl_virus", # Mapping
        "tomato_healthy": "tomato_healthy"
    }
    
    # The 6 shared classes defined in the plan
    SHARED_CLASSES = {
        "tomato_early_blight",
        "tomato_late_blight",
        "tomato_septoria_spot",
        "tomato_mosaic_virus",
        "tomato_yellow_curl_virus",
        "tomato_healthy"
    }

    def __init__(self, root_dir: str, transform: Optional[Callable] = None, split: str = "all", class_to_idx: Dict[str, int] = None, include_unknown: bool = False):
        super().__init__(root_dir, transform)
        self.split = split
        self.provided_class_to_idx = class_to_idx
        self.include_unknown = include_unknown
        self._load_metadata()

    def _load_metadata(self):
        metadata_path = self.root_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at {metadata_path}")
            
        # If class_to_idx is provided (from Source), use it.
        if self.provided_class_to_idx:
            self.class_to_idx = self.provided_class_to_idx
        else:
            # Fallback (should not happen in correct usage)
            self.class_to_idx = {} 

        # Load samples
        labels_csv = self.root_dir / "labels.csv"
        import csv
        with open(labels_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                raw_label = row['label']
                filename = row['filename']
                
                # CASE 1: Known/Shared Class (For Evaluation/Labeling)
                if raw_label in self.CLASS_MAPPING and self.CLASS_MAPPING[raw_label] in self.class_to_idx:
                    mapped_label = self.CLASS_MAPPING[raw_label]
                    target_idx = self.class_to_idx[mapped_label]
                    
                    # Check split filter
                    if self.split == "all" or row.get('original_split') == self.split:
                        rel_path = Path(raw_label) / filename
                        self.samples.append((str(rel_path), target_idx))

                # CASE 2: Unknown/Noise (For Unlabeled Training ONLY)
                elif self.include_unknown:
                    # We treat these as "Unlabeled" (index -1 usually, or just ignored in loss)
                    # Ideally, for "Healthy FPR" calculation, we track them as specific index
                    # But for Unlabeled training, we just need the image.
                    rel_path = Path(raw_label) / filename
                    self.samples.append((str(rel_path), -1)) # -1 indicates Unknown/Outlier

def get_transform(train: bool = True) -> transforms.Compose:
    """Standard transforms for MobileNetV3."""
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])

class TransformFixMatch(object):
    def __init__(self, mean=MEAN, std=STD):
        self.weak = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
        self.strong = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=10), # RandAugment for strong
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

    def __call__(self, x):
        return self.weak(x), self.strong(x)
