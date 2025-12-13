"""
Crop-specific label configuration for PDA experiments.

This module provides canonical class definitions for each crop, handling
the label mismatches between PlantVillage (source/lab) and PlantDoc (target/field).

The key insight is that we must only use classes that exist in BOTH datasets,
otherwise the model will predict classes that don't exist in the target domain.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set


@dataclass
class CropConfig:
    """Configuration for a single crop experiment."""
    name: str
    # Classes that exist in PlantVillage (source)
    source_classes: List[str]
    # Classes that exist in PlantDoc (target)
    target_classes: List[str]
    # Mapping from source folder names to canonical names
    source_mapping: Dict[str, str]
    # Mapping from target folder names to canonical names
    target_mapping: Dict[str, str]
    # Classes that exist in BOTH (intersection) - these are the canonical classes
    canonical_classes: List[str]
    # Classes only in source (will cause negative transfer if model predicts them)
    source_only_classes: List[str]
    # Classes only in target (model can't predict these)
    target_only_classes: List[str]

    @property
    def num_classes(self) -> int:
        return len(self.canonical_classes)

    @property
    def is_partial_domain(self) -> bool:
        """True if source has classes not in target (partial domain adaptation)."""
        return len(self.source_only_classes) > 0


# =============================================================================
# TOMATO CONFIGURATION
# =============================================================================
# PlantVillage has 10 tomato classes, PlantDoc has 9
# Key mismatch: spider_mites naming differs, target_spot and yellow_curl_virus only in PV

TOMATO_SOURCE_CLASSES = [
    "tomato_bacterial_spot",
    "tomato_early_blight",
    "tomato_healthy",
    "tomato_late_blight",
    "tomato_mold",
    "tomato_mosaic_virus",
    "tomato_septoria_spot",
    "tomato_spider_mites_two_spotted_spider_mite",  # PV naming
    "tomato_target_spot",
    "tomato_yellow_curl_virus",
]

TOMATO_TARGET_CLASSES = [
    "tomato_bacterial_spot",
    "tomato_early_blight",
    "tomato_healthy",
    "tomato_late_blight",
    "tomato_mold",
    "tomato_mosaic_virus",
    "tomato_septoria_spot",
    "tomato_two_spotted_spider_mites",  # PD naming (different!)
    "tomato_yellow_virus",  # PD naming (different!)
]

# Canonical names we'll use (intersection with normalization)
TOMATO_CANONICAL = [
    "tomato_bacterial_spot",
    "tomato_early_blight",
    "tomato_healthy",
    "tomato_late_blight",
    "tomato_mold",
    "tomato_mosaic_virus",
    "tomato_septoria_spot",
]

# Mappings to canonical
TOMATO_SOURCE_MAP = {
    "tomato_bacterial_spot": "tomato_bacterial_spot",
    "tomato_early_blight": "tomato_early_blight",
    "tomato_healthy": "tomato_healthy",
    "tomato_late_blight": "tomato_late_blight",
    "tomato_mold": "tomato_mold",
    "tomato_mosaic_virus": "tomato_mosaic_virus",
    "tomato_septoria_spot": "tomato_septoria_spot",
    # These don't have equivalents in target - exclude from canonical
    "tomato_spider_mites_two_spotted_spider_mite": None,  # Exclude (only 2 in target)
    "tomato_target_spot": None,  # Not in target
    "tomato_yellow_curl_virus": None,  # Different naming, exclude for safety
}

TOMATO_TARGET_MAP = {
    "tomato_bacterial_spot": "tomato_bacterial_spot",
    "tomato_early_blight": "tomato_early_blight",
    "tomato_healthy": "tomato_healthy",
    "tomato_late_blight": "tomato_late_blight",
    "tomato_mold": "tomato_mold",
    "tomato_mosaic_virus": "tomato_mosaic_virus",
    "tomato_septoria_spot": "tomato_septoria_spot",
    # These are excluded
    "tomato_two_spotted_spider_mites": None,  # Only 2 samples
    "tomato_yellow_virus": None,  # Different from source naming
}

TOMATO_CONFIG = CropConfig(
    name="tomato",
    source_classes=TOMATO_SOURCE_CLASSES,
    target_classes=TOMATO_TARGET_CLASSES,
    source_mapping=TOMATO_SOURCE_MAP,
    target_mapping=TOMATO_TARGET_MAP,
    canonical_classes=TOMATO_CANONICAL,
    source_only_classes=["tomato_spider_mites_two_spotted_spider_mite", "tomato_target_spot", "tomato_yellow_curl_virus"],
    target_only_classes=["tomato_two_spotted_spider_mites", "tomato_yellow_virus"],
)


# =============================================================================
# POTATO CONFIGURATION
# =============================================================================
# PlantVillage has 3 classes (early_blight, healthy, late_blight)
# PlantDoc has only 2 classes (early_blight, late_blight) - NO HEALTHY!
# This is the PARTIAL DOMAIN ADAPTATION scenario

POTATO_SOURCE_CLASSES = [
    "potato_early_blight",
    "potato_healthy",
    "potato_late_blight",
]

POTATO_TARGET_CLASSES = [
    "potato_early_blight",
    "potato_late_blight",
    # NO potato_healthy in PlantDoc!
]

# For potato, we use ALL source classes but acknowledge healthy is missing in target
# This creates the partial domain adaptation scenario we want to study
POTATO_CANONICAL = [
    "potato_early_blight",
    "potato_healthy",  # Keep this! It's the PDA scenario
    "potato_late_blight",
]

POTATO_SOURCE_MAP = {
    "potato_early_blight": "potato_early_blight",
    "potato_healthy": "potato_healthy",
    "potato_late_blight": "potato_late_blight",
}

POTATO_TARGET_MAP = {
    "potato_early_blight": "potato_early_blight",
    "potato_late_blight": "potato_late_blight",
    # potato_healthy doesn't exist in target
}

POTATO_CONFIG = CropConfig(
    name="potato",
    source_classes=POTATO_SOURCE_CLASSES,
    target_classes=POTATO_TARGET_CLASSES,
    source_mapping=POTATO_SOURCE_MAP,
    target_mapping=POTATO_TARGET_MAP,
    canonical_classes=POTATO_CANONICAL,
    source_only_classes=["potato_healthy"],  # The missing class!
    target_only_classes=[],
)


# =============================================================================
# PEPPER CONFIGURATION
# =============================================================================
# PlantVillage: pepper_bell_bacterial_spot, pepper_bell_healthy
# PlantDoc: pepper_bell (=healthy), pepper_bell_spot (=bacterial_spot)

PEPPER_SOURCE_CLASSES = [
    "pepper_bell_bacterial_spot",
    "pepper_bell_healthy",
]

PEPPER_TARGET_CLASSES = [
    "pepper_bell",  # This is actually healthy
    "pepper_bell_spot",  # This is bacterial spot
]

PEPPER_CANONICAL = [
    "pepper_bell_bacterial_spot",
    "pepper_bell_healthy",
]

PEPPER_SOURCE_MAP = {
    "pepper_bell_bacterial_spot": "pepper_bell_bacterial_spot",
    "pepper_bell_healthy": "pepper_bell_healthy",
}

PEPPER_TARGET_MAP = {
    "pepper_bell": "pepper_bell_healthy",  # Map to canonical
    "pepper_bell_spot": "pepper_bell_bacterial_spot",  # Map to canonical
}

PEPPER_CONFIG = CropConfig(
    name="pepper",
    source_classes=PEPPER_SOURCE_CLASSES,
    target_classes=PEPPER_TARGET_CLASSES,
    source_mapping=PEPPER_SOURCE_MAP,
    target_mapping=PEPPER_TARGET_MAP,
    canonical_classes=PEPPER_CANONICAL,
    source_only_classes=[],
    target_only_classes=[],
)


# =============================================================================
# REGISTRY
# =============================================================================

CROP_CONFIGS: Dict[str, CropConfig] = {
    "tomato": TOMATO_CONFIG,
    "potato": POTATO_CONFIG,
    "pepper": PEPPER_CONFIG,
}


def get_crop_config(crop_name: str) -> CropConfig:
    """Get configuration for a specific crop."""
    crop_name = crop_name.lower().strip()
    if crop_name not in CROP_CONFIGS:
        raise ValueError(f"Unknown crop: {crop_name}. Available: {list(CROP_CONFIGS.keys())}")
    return CROP_CONFIGS[crop_name]


def get_canonical_classes(crop_filter: str) -> List[str]:
    """
    Get canonical class list for one or more crops.

    Args:
        crop_filter: Single crop name or comma-separated list (e.g., "tomato,potato,pepper")

    Returns:
        Sorted list of canonical class names.
    """
    crops = [c.strip().lower() for c in crop_filter.split(",") if c.strip()]

    all_classes = set()
    for crop in crops:
        config = get_crop_config(crop)
        all_classes.update(config.canonical_classes)

    return sorted(all_classes)


def get_source_to_canonical_mapping(crop_filter: str) -> Dict[str, Optional[str]]:
    """Get mapping from source folder names to canonical names."""
    crops = [c.strip().lower() for c in crop_filter.split(",") if c.strip()]

    mapping = {}
    for crop in crops:
        config = get_crop_config(crop)
        mapping.update(config.source_mapping)

    return mapping


def get_target_to_canonical_mapping(crop_filter: str) -> Dict[str, Optional[str]]:
    """Get mapping from target folder names to canonical names."""
    crops = [c.strip().lower() for c in crop_filter.split(",") if c.strip()]

    mapping = {}
    for crop in crops:
        config = get_crop_config(crop)
        mapping.update(config.target_mapping)

    return mapping


def print_crop_summary(crop_filter: str) -> None:
    """Print summary of crop configuration for debugging."""
    crops = [c.strip().lower() for c in crop_filter.split(",") if c.strip()]

    print(f"\n{'='*60}")
    print(f"CROP CONFIGURATION SUMMARY: {crop_filter}")
    print(f"{'='*60}")

    for crop in crops:
        config = get_crop_config(crop)
        print(f"\n[{config.name.upper()}]")
        print(f"  Canonical classes ({config.num_classes}): {config.canonical_classes}")
        print(f"  Source-only (PDA): {config.source_only_classes or 'None'}")
        print(f"  Target-only: {config.target_only_classes or 'None'}")
        print(f"  Is Partial Domain: {config.is_partial_domain}")

    all_canonical = get_canonical_classes(crop_filter)
    print(f"\n[COMBINED]")
    print(f"  Total canonical classes: {len(all_canonical)}")
    print(f"  Classes: {all_canonical}")

