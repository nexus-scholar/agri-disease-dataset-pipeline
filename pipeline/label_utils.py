"""Label normalization helpers shared across datasets."""
from __future__ import annotations

from functools import lru_cache

CROP_KEYWORDS = {
    "tomato": "tomato",
    "tom": "tomato",
    "potato": "potato",
    "pepper": "pepper",
    "bell_pepper": "pepper",
    "bell pepper": "pepper",
    "apple": "apple",
    "corn": "corn",
    "grape": "grape",
    "peach": "peach",
    "cherry": "cherry",
    "blueberry": "blueberry",
    "strawberry": "strawberry",
    "raspberry": "raspberry",
    "soyabean": "soybean",
    "soybean": "soybean",
    "squash": "squash",
}

LABEL_ALIASES = {
    # Tomato mites variants
    "tomato_spider_mites_two_spotted_spider_mite": "tomato_spider_mites",
    "tomato_two_spotted_spider_mites": "tomato_spider_mites",
    # Tomato yellow leaf curl naming differences
    "tomato_yellow_curl_virus": "tomato_yellow_leaf_curl_virus",
    "tomato_yellow_virus": "tomato_yellow_leaf_curl_virus",
    # Pepper naming differences
    "pepper_bell": "pepper_bell_healthy",
    "pepper_bell_spot": "pepper_bell_bacterial_spot",
}


@lru_cache(maxsize=None)
def normalize_label(folder_name: str) -> tuple[str, str, str]:
    """Convert folder name to standardized label format: crop_disease."""
    name = folder_name.replace("__", "_").replace("  ", " ")
    name_lower = name.lower()

    crop = "unknown"
    for keyword, crop_name in CROP_KEYWORDS.items():
        if keyword in name_lower:
            crop = crop_name
            break

    disease_part = name_lower
    for keyword in CROP_KEYWORDS.keys():
        disease_part = disease_part.replace(keyword, "")

    for word in ["leaf", "leaves", "__", "  "]:
        disease_part = disease_part.replace(word, " ")

    disease_part = "_".join(disease_part.split()).strip("_")
    if not disease_part or disease_part in {"healthy", "normal"}:
        disease_part = "healthy"

    label = f"{crop}_{disease_part}" if crop != "unknown" else disease_part
    while "__" in label:
        label = label.replace("__", "_")
    return label.strip("_"), crop, disease_part


def canonicalize_label(label: str) -> str:
    """Map dataset-specific label variants to canonical snake_case labels."""
    cleaned = (label or "").strip().lower()
    return LABEL_ALIASES.get(cleaned, cleaned)


def extract_crop(label: str) -> str:
    """Return the crop portion of a canonical label (best-effort)."""
    canonical = canonicalize_label(label)
    if not canonical:
        return "unknown"
    if "_" in canonical:
        return canonical.split("_", 1)[0]
    return canonical.split()[0]

