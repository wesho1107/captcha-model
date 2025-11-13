"""
Test script to verify preprocessing and augmentation changes.

This script loads a few samples and displays the preprocessing results.
"""

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from captcha_coco_dataset import COCOCaptchaDataset, remove_black_lines
from PIL import Image


def test_black_line_removal():
    """Test the black line removal function on a sample image."""
    print("Testing black line removal...")

    dataset = COCOCaptchaDataset(
        coco_json_path='/Users/wesho/repos/captcha-model/output.json',
        image_dir='/Users/wesho/repos/captcha-model/data/train',
        train=True,
        augment=False,
        remove_noise=False,  # Disable to test manually
    )

    # Get first image
    image_id = dataset.image_ids[20]
    image_info = next(img for img in dataset.coco_data['images'] if img['id'] == image_id)
    image_path = dataset.image_dir / image_info['file_name']

    # Load original
    original = Image.open(image_path).convert('RGB')

    # Apply preprocessing
    cleaned = remove_black_lines(original)

    # Display results
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].imshow(original)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(cleaned)
    axes[1].set_title('After Black Line Removal')
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('/Users/wesho/repos/captcha-model/rcnn/test_black_line_removal.png')
    print("Saved comparison to: rcnn/test_black_line_removal.png")
    plt.close()


def test_dataset_loading():
    """Test basic dataset loading and statistics."""
    print("\nTesting dataset loading...")

    dataset = COCOCaptchaDataset(
        coco_json_path='/Users/wesho/repos/captcha-model/output.json',
        image_dir='/Users/wesho/repos/captcha-model/data/train',
        train=True,
        augment=False,
        remove_noise=True,
    )

    print(dataset.image_ids)

    print(f"✓ Dataset size: {len(dataset)}")
    print(f"✓ Number of classes: {dataset.get_num_classes()}")
    print(f"✓ Class names: {dataset.get_class_names()}")

    # Load a sample
    image, target = dataset[20]
    print(f"\n✓ Sample image shape: {image.shape}")
    print(f"✓ Sample boxes shape: {target['boxes'].shape}")
    print(f"✓ Sample labels: {target['labels']}")
    print(f"✓ Sample image_id: {target['image_id'].item()}")

    print("\n✓ All tests passed!")


if __name__ == '__main__':
    print("=" * 60)
    print("CAPTCHA Dataset Preprocessing Test Suite")
    print("=" * 60)

    # Test 1: Black line removal
    test_black_line_removal()

    # Test 2: Dataset loading
    test_dataset_loading()

    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
