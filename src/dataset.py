"""Dataset utilities for NEU Surface Defect Detection."""

import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


# NEU Surface Defect Dataset classes
NEU_CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

# For segmentation we use a simplified binary approach per-class
# or multi-class classification with 6 defect types + background


class NEUDataset(Dataset):
    """NEU Surface Defect Dataset for classification.

    The NEU dataset contains 1800 grayscale images (200x200) of 6 defect types.
    Each class has 300 samples.

    Args:
        root_dir: Path to the dataset root directory.
        transform: Optional transforms to apply to images.
        split: 'train' or 'val' split.
        val_ratio: Fraction of data to use for validation.
    """

    def __init__(
        self,
        root_dir: str,
        transform: transforms.Compose | None = None,
        split: str = "train",
        val_ratio: float = 0.2,
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        self.images = []
        self.labels = []

        for class_idx, class_name in enumerate(NEU_CLASSES):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue

            image_files = sorted(class_dir.glob("*.bmp")) + sorted(class_dir.glob("*.jpg"))
            n_val = int(len(image_files) * val_ratio)

            if split == "val":
                image_files = image_files[:n_val]
            else:
                image_files = image_files[n_val:]

            for img_path in image_files:
                self.images.append(str(img_path))
                self.labels.append(class_idx)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, label


class NEUSegmentationDataset(Dataset):
    """NEU Surface Defect Dataset with synthetic segmentation masks.

    Since NEU doesn't provide pixel-level annotations by default,
    this dataset generates approximate masks using thresholding on
    the grayscale images for demonstration purposes.

    Args:
        root_dir: Path to the dataset root directory.
        transform: Optional transforms to apply to images.
        mask_transform: Optional transforms to apply to masks.
        split: 'train' or 'val' split.
        val_ratio: Fraction of data to use for validation.
        image_size: Target image size (height, width).
    """

    def __init__(
        self,
        root_dir: str,
        transform: transforms.Compose | None = None,
        mask_transform: transforms.Compose | None = None,
        split: str = "train",
        val_ratio: float = 0.2,
        image_size: tuple[int, int] = (200, 200),
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.mask_transform = mask_transform
        self.split = split
        self.image_size = image_size
        self.images = []
        self.labels = []

        for class_idx, class_name in enumerate(NEU_CLASSES):
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue

            image_files = sorted(class_dir.glob("*.bmp")) + sorted(class_dir.glob("*.jpg"))
            n_val = int(len(image_files) * val_ratio)

            if split == "val":
                image_files = image_files[:n_val]
            else:
                image_files = image_files[n_val:]

            for img_path in image_files:
                self.images.append(str(img_path))
                self.labels.append(class_idx)

    def __len__(self) -> int:
        return len(self.images)

    def _generate_mask(self, image: Image.Image, label: int) -> np.ndarray:
        """Generate a synthetic segmentation mask from the image.

        Uses adaptive thresholding to detect defect regions.
        """
        gray = np.array(image.convert("L"), dtype=np.float32)

        # Normalize
        gray = (gray - gray.mean()) / (gray.std() + 1e-8)

        # Threshold to find anomalous regions
        mask = np.zeros_like(gray, dtype=np.int64)
        threshold = 1.5  # Standard deviations from mean

        # Different defects have different characteristics
        if label in [0, 4, 5]:  # crazing, rolled-in_scale, scratches (darker)
            mask[gray < -threshold] = label + 1
        elif label in [1, 2, 3]:  # inclusion, patches, pitted_surface (brighter/darker spots)
            mask[np.abs(gray) > threshold] = label + 1

        return mask

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.images[idx]).convert("RGB")
        label = self.labels[idx]

        # Generate synthetic mask
        mask = self._generate_mask(image, label)

        # Resize
        image = image.resize(self.image_size, Image.BILINEAR)
        mask = Image.fromarray(mask.astype(np.uint8)).resize(
            self.image_size, Image.NEAREST
        )
        mask = np.array(mask, dtype=np.int64)

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        mask = torch.from_numpy(mask).long()
        return image, mask


def get_train_transforms(image_size: tuple[int, int] = (200, 200)) -> transforms.Compose:
    """Get training data transforms with augmentation."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(image_size: tuple[int, int] = (200, 200)) -> transforms.Compose:
    """Get validation data transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
