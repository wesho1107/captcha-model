"""
COCO Dataset loader for CAPTCHA character detection.

Loads CAPTCHA images and annotations from COCO format JSON file.
Supports train/val split and augmentations for Faster RCNN training.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from PIL import Image
import cv2
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F


def remove_black_lines(image: Image.Image) -> Image.Image:
    """
    Remove near-black lines and artifacts from CAPTCHA image using inpainting.

    Args:
        image: PIL Image in RGB format

    Returns:
        Cleaned PIL Image with black lines removed
    """
    # Convert PIL to numpy array for OpenCV processing
    img_array = np.array(image)

    # Convert to grayscale for processing
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # Create mask for near-black pixels (< 10 grayscale value)
    mask = (gray < 10).astype(np.uint8) * 255

    # Only inpaint if there are black pixels to remove
    if cv2.countNonZero(mask) > 0:
        # Inpaint using TELEA algorithm
        img_array = cv2.inpaint(img_array, mask, 3, cv2.INPAINT_TELEA)

    # Convert back to PIL Image
    return Image.fromarray(img_array)


class COCOCaptchaDataset(Dataset):
    """
    PyTorch Dataset for CAPTCHA character detection from COCO format.

    Loads CAPTCHA images and character annotations in COCO format.
    Each image contains multiple character bounding boxes with class labels (0-9, a-z).

    Args:
        coco_json_path: Path to COCO format JSON file
        image_dir: Directory containing CAPTCHA images (parent directory of image files)
        train: If True, use training split; if False, use validation split
        train_ratio: Fraction of data to use for training (default 0.8)
        augment: If True, apply data augmentations
        remove_noise: If True, apply black line removal preprocessing
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        coco_json_path: str,
        image_dir: str,
        train: bool = True,
        train_ratio: float = 0.8,
        augment: bool = True,
        remove_noise: bool = True,
        seed: int = 42,
    ):
        self.image_dir = Path(image_dir)
        self.train = train
        self.augment = augment and train  # Only augment training data
        self.remove_noise = remove_noise
        self.seed = seed

        # Load COCO JSON
        with open(coco_json_path, 'r') as f:
            self.coco_data = json.load(f)

        # Build category mapping
        self.category_id_to_label = {}
        self.label_to_category_id = {}
        for cat in self.coco_data['categories']:
            cat_id = cat['id']
            label = int(cat['name']) if cat['name'].isdigit() else ord(cat['name'].lower()) - ord('a') + 10
            self.category_id_to_label[cat_id] = label
            self.label_to_category_id[label] = cat_id

        # Build image id to annotations mapping
        self.image_id_to_annots = {}
        for annot in self.coco_data['annotations']:
            img_id = annot['image_id']
            if img_id not in self.image_id_to_annots:
                self.image_id_to_annots[img_id] = []
            self.image_id_to_annots[img_id].append(annot)

        # Split into train/val
        np.random.seed(seed)
        image_ids = np.array([img['id'] for img in self.coco_data['images']])
        n_train = int(len(image_ids) * train_ratio)

        perm = np.random.permutation(len(image_ids))
        train_indices = perm[:n_train]
        val_indices = perm[n_train:]

        self.image_ids = image_ids[train_indices] if train else image_ids[val_indices]

        # Define augmentations
        self.augmentation_transforms = T.Compose([
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        ])

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict]:
        """
        Get image and targets for Faster RCNN.

        Returns:
            image: Tensor of shape (3, H, W)
            target: Dict with keys:
                - boxes: Tensor of shape (N, 4) in format [x1, y1, x2, y2]
                - labels: Tensor of shape (N,) with class indices 0-35
                - image_id: int
                - area: Tensor of shape (N,)
                - iscrowd: Tensor of shape (N,)
        """
        image_id = self.image_ids[idx]
        image_info = next(img for img in self.coco_data['images'] if img['id'] == image_id)

        # Load image
        image_path = self.image_dir / image_info['file_name']
        image = Image.open(image_path).convert('RGB')

        # Remove noise if requested
        if self.remove_noise:
            image = remove_black_lines(image)

        # Get annotations for this image
        annots = self.image_id_to_annots.get(image_id, [])

        # Extract boxes and labels
        boxes = []
        labels = []
        areas = []
        iscrowds = []

        for annot in annots:
            # Convert COCO bbox [x, y, w, h] to [x1, y1, x2, y2]
            x, y, w, h = annot['bbox']
            boxes.append([x, y, x + w, y + h])

            # Convert category_id to label (0-35)
            cat_id = annot['category_id']
            label = self.category_id_to_label[cat_id]
            labels.append(label)

            areas.append(annot.get('area', w * h))
            iscrowds.append(annot.get('iscrowd', 0))

        # Convert to tensors
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowds = torch.as_tensor(iscrowds, dtype=torch.uint8)
        else:
            # No annotations - create empty tensors
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowds = torch.zeros((0,), dtype=torch.uint8)

        # Apply augmentations
        if self.augment:
            image = self.augmentation_transforms(image)

        # Convert image to tensor and normalize
        image_tensor = F.to_tensor(image)

        # Normalize with ImageNet stats (common for transfer learning)
        normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        image_tensor = normalize(image_tensor)

        # Build target dict for Faster RCNN
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor(image_id),
            'area': areas,
            'iscrowd': iscrowds,
        }

        return image_tensor, target

    def get_num_classes(self) -> int:
        """Return number of character classes (36: 0-9 + a-z)."""
        return len(self.category_id_to_label)

    def get_class_names(self) -> List[str]:
        """Return list of class names in order (0-9, then a-z)."""
        names = [''] * len(self.category_id_to_label)
        for cat in self.coco_data['categories']:
            cat_id = cat['id']
            label = self.category_id_to_label[cat_id]
            names[label] = cat['name']
        return names


def collate_fn(batch: List[Tuple[torch.Tensor, Dict]]) -> Tuple[List[torch.Tensor], List[Dict]]:
    """
    Custom collate function for DataLoader.

    Returns batches of images and targets as lists (required by Faster RCNN).

    Args:
        batch: List of (image, target) tuples

    Returns:
        images: List of image tensors
        targets: List of target dicts
    """
    images = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    return images, targets


if __name__ == '__main__':
    # Test dataset loading
    dataset = COCOCaptchaDataset(
        coco_json_path='/Users/wesho/repos/captcha-model/output.json',
        image_dir='/Users/wesho/repos/captcha-model/data/train',
        train=True,
        augment=False,
        remove_noise=True,
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {dataset.get_num_classes()}")
    print(f"Class names: {dataset.get_class_names()}")

    # Load a sample
    image, target = dataset[0]
    print(f"\nSample image shape: {image.shape}")
    print(f"Sample boxes shape: {target['boxes'].shape}")
    print(f"Sample labels: {target['labels']}")
    print(f"Sample image_id: {target['image_id'].item()}")
