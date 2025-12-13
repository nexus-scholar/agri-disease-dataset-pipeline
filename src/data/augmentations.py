import torch
from torchvision import transforms
from torchvision.transforms import AutoAugmentPolicy


class WeakStrongTransform:
    """
    Returns (weak, strong) augmentations for a single input image.
    """

    def __init__(self, image_size: int = 224):
        normalize_mean = [0.485, 0.456, 0.406]
        normalize_std = [0.229, 0.224, 0.225]

        self.weak = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, padding=int(image_size * 0.125), padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ])

        self.strong = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(image_size, padding=int(image_size * 0.125), padding_mode='reflect'),
            transforms.AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
            transforms.ToTensor(),
            transforms.Normalize(normalize_mean, normalize_std),
        ])

    def __call__(self, x):
        weak_img = self.weak(x)
        strong_img = self.strong(x)
        return weak_img, strong_img

