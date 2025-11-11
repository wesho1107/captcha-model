"""
Data utilities for creating and managing CAPTCHA datasets for Faster RCNN.

Provides functions for creating data loaders and managing train/val splits.
"""

from typing import Tuple, Optional
from torch.utils.data import DataLoader
from captcha_coco_dataset import COCOCaptchaDataset, collate_fn


def create_data_loaders(
    coco_json_path: str,
    image_dir: str,
    train_batch_size: int = 8,
    val_batch_size: int = 8,
    train_ratio: float = 0.8,
    num_workers: int = 4,
    seed: int = 42,
    augment: bool = True,
    remove_noise: bool = True,
) -> Tuple[DataLoader, DataLoader, COCOCaptchaDataset]:
    """
    Create training and validation DataLoaders.

    Args:
        coco_json_path: Path to COCO format JSON file
        image_dir: Directory containing CAPTCHA images
        train_batch_size: Batch size for training
        val_batch_size: Batch size for validation
        train_ratio: Fraction of data for training (0.8 = 80% train, 20% val)
        num_workers: Number of worker processes for data loading
        seed: Random seed for reproducibility
        augment: Whether to apply data augmentations
        remove_noise: Whether to apply black line removal preprocessing

    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        train_dataset: Training dataset (for getting metadata like num_classes)
    """

    train_dataset = COCOCaptchaDataset(
        coco_json_path=coco_json_path,
        image_dir=image_dir,
        train=True,
        train_ratio=train_ratio,
        augment=augment,
        remove_noise=remove_noise,
        seed=seed,
    )

    val_dataset = COCOCaptchaDataset(
        coco_json_path=coco_json_path,
        image_dir=image_dir,
        train=False,
        train_ratio=train_ratio,
        augment=False,  # Never augment validation data
        remove_noise=remove_noise,
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, train_dataset


if __name__ == '__main__':
    # Example usage
    train_loader, val_loader, train_dataset = create_data_loaders(
        coco_json_path='/Users/wesho/repos/captcha-model/output.json',
        image_dir='/Users/wesho/repos/captcha-model/data/train',
        train_batch_size=8,
        val_batch_size=16,
    )

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    print(f"Number of classes: {train_dataset.get_num_classes()}")
    print(f"Class names: {train_dataset.get_class_names()}")

    # Load a batch
    images, targets = next(iter(train_loader))
    print(f"\nBatch size: {len(images)}")
    print(f"Image shapes: {[img.shape for img in images]}")
    print(f"Sample target keys: {targets[0].keys()}")
    print(f"Sample boxes shape: {targets[0]['boxes'].shape}")
