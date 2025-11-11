"""
Data validation utilities for COCO CAPTCHA dataset.

Validates COCO JSON format, checks image existence, and provides dataset statistics.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image


class COCOValidator:
    """Validates COCO format CAPTCHA dataset."""

    def __init__(self, coco_json_path: str, image_dir: str):
        self.coco_json_path = Path(coco_json_path)
        self.image_dir = Path(image_dir)
        self.coco_data = None
        self.issues = []

    def load_json(self) -> bool:
        """Load and validate JSON structure."""
        try:
            with open(self.coco_json_path, 'r') as f:
                self.coco_data = json.load(f)
            return True
        except FileNotFoundError:
            self.issues.append(f"JSON file not found: {self.coco_json_path}")
            return False
        except json.JSONDecodeError as e:
            self.issues.append(f"Invalid JSON: {e}")
            return False

    def validate_json_structure(self) -> bool:
        """Validate COCO JSON has required fields."""
        if not self.coco_data:
            return False

        required_keys = ['images', 'annotations', 'categories']
        for key in required_keys:
            if key not in self.coco_data:
                self.issues.append(f"Missing required key: {key}")
                return False

        # Check image structure
        for img in self.coco_data['images']:
            if not all(k in img for k in ['id', 'file_name', 'width', 'height']):
                self.issues.append(f"Image {img.get('id', '?')} missing required fields")
                return False

        # Check annotation structure
        for annot in self.coco_data['annotations']:
            if not all(k in annot for k in ['id', 'image_id', 'category_id', 'bbox']):
                self.issues.append(f"Annotation {annot.get('id', '?')} missing required fields")
                return False

        # Check category structure
        for cat in self.coco_data['categories']:
            if not all(k in cat for k in ['id', 'name']):
                self.issues.append(f"Category {cat.get('id', '?')} missing required fields")
                return False

        return True

    def validate_images_exist(self) -> bool:
        """Check that all image files exist."""
        missing_count = 0
        for img in self.coco_data['images']:
            image_path = self.image_dir / img['file_name']
            if not image_path.exists():
                self.issues.append(f"Image not found: {image_path}")
                missing_count += 1

        if missing_count > 0:
            print(f"Warning: {missing_count}/{len(self.coco_data['images'])} images missing")
            return False
        return True

    def validate_image_dimensions(self) -> bool:
        """Check that image dimensions match metadata."""
        issues_count = 0
        for img in self.coco_data['images'][:100]:  # Sample first 100
            image_path = self.image_dir / img['file_name']
            if not image_path.exists():
                continue

            try:
                pil_img = Image.open(image_path)
                width, height = pil_img.size
                if width != img['width'] or height != img['height']:
                    self.issues.append(
                        f"Image {img['file_name']}: "
                        f"metadata ({img['width']}, {img['height']}) "
                        f"!= actual ({width}, {height})"
                    )
                    issues_count += 1
            except Exception as e:
                self.issues.append(f"Error loading {image_path}: {e}")
                issues_count += 1

        if issues_count > 0:
            print(f"Warning: {issues_count} dimension mismatches found")
            return False
        return True

    def validate_bbox_format(self) -> bool:
        """Check that bboxes are in correct COCO format [x, y, w, h]."""
        issues_count = 0
        for annot in self.coco_data['annotations'][:1000]:  # Sample first 1000
            bbox = annot['bbox']
            if len(bbox) != 4:
                self.issues.append(f"Annotation {annot['id']}: bbox length != 4")
                issues_count += 1
                continue

            x, y, w, h = bbox
            if w <= 0 or h <= 0:
                self.issues.append(
                    f"Annotation {annot['id']}: invalid bbox dimensions "
                    f"({w}, {h}) - must be positive"
                )
                issues_count += 1

            # Check image bounds
            img_id = annot['image_id']
            img_info = next((img for img in self.coco_data['images'] if img['id'] == img_id), None)
            if img_info and (x + w > img_info['width'] or y + h > img_info['height']):
                self.issues.append(
                    f"Annotation {annot['id']}: bbox exceeds image bounds"
                )
                issues_count += 1

        if issues_count > 0:
            print(f"Warning: {issues_count} bbox issues found")
            return False
        return True

    def validate_category_ids(self) -> bool:
        """Check that all annotations reference valid categories."""
        valid_cat_ids = {cat['id'] for cat in self.coco_data['categories']}

        for annot in self.coco_data['annotations']:
            if annot['category_id'] not in valid_cat_ids:
                self.issues.append(
                    f"Annotation {annot['id']}: "
                    f"invalid category_id {annot['category_id']}"
                )
                return False

        return True

    def get_statistics(self) -> Dict:
        """Compute dataset statistics."""
        if not self.coco_data:
            return {}

        stats = {
            'num_images': len(self.coco_data['images']),
            'num_annotations': len(self.coco_data['annotations']),
            'num_classes': len(self.coco_data['categories']),
            'classes': {cat['name']: 0 for cat in self.coco_data['categories']},
            'image_sizes': {'min_width': float('inf'), 'max_width': 0, 'avg_width': 0},
            'bbox_sizes': {'min_area': float('inf'), 'max_area': 0, 'avg_area': 0},
            'annot_per_image': {'min': float('inf'), 'max': 0, 'avg': 0},
        }

        # Count annotations per class
        for annot in self.coco_data['annotations']:
            cat_id = annot['category_id']
            cat_name = next(
                (cat['name'] for cat in self.coco_data['categories'] if cat['id'] == cat_id),
                'unknown'
            )
            stats['classes'][cat_name] += 1

        # Image dimension stats
        widths = [img['width'] for img in self.coco_data['images']]
        stats['image_sizes']['min_width'] = min(widths)
        stats['image_sizes']['max_width'] = max(widths)
        stats['image_sizes']['avg_width'] = np.mean(widths)

        # Bbox area stats
        areas = [annot['bbox'][2] * annot['bbox'][3] for annot in self.coco_data['annotations']]
        if areas:
            stats['bbox_sizes']['min_area'] = min(areas)
            stats['bbox_sizes']['max_area'] = max(areas)
            stats['bbox_sizes']['avg_area'] = np.mean(areas)

        # Annotations per image
        annot_counts = {}
        for annot in self.coco_data['annotations']:
            img_id = annot['image_id']
            annot_counts[img_id] = annot_counts.get(img_id, 0) + 1

        if annot_counts:
            counts = list(annot_counts.values())
            stats['annot_per_image']['min'] = min(counts)
            stats['annot_per_image']['max'] = max(counts)
            stats['annot_per_image']['avg'] = np.mean(counts)

        return stats

    def run_validation(self, check_images: bool = True) -> bool:
        """Run all validation checks."""
        print("Validating COCO dataset...")

        if not self.load_json():
            return False

        checks = [
            ("JSON structure", self.validate_json_structure()),
            ("Category IDs", self.validate_category_ids()),
            ("BBox format", self.validate_bbox_format()),
        ]

        if check_images:
            checks.extend([
                ("Images exist", self.validate_images_exist()),
                ("Image dimensions", self.validate_image_dimensions()),
            ])

        all_passed = all(passed for _, passed in checks)

        for name, passed in checks:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {status}: {name}")

        if self.issues:
            print(f"\n{len(self.issues)} issues found:")
            for issue in self.issues[:10]:  # Show first 10
                print(f"  - {issue}")
            if len(self.issues) > 10:
                print(f"  ... and {len(self.issues) - 10} more")

        return all_passed

    def print_statistics(self):
        """Print dataset statistics."""
        stats = self.get_statistics()

        if not stats:
            print("No statistics available")
            return

        print("\n=== Dataset Statistics ===")
        print(f"Images: {stats['num_images']}")
        print(f"Annotations: {stats['num_annotations']}")
        print(f"Classes: {stats['num_classes']}")

        print("\nClass distribution:")
        sorted_classes = sorted(stats['classes'].items(), key=lambda x: x[1], reverse=True)
        for class_name, count in sorted_classes:
            print(f"  {class_name}: {count}")

        print(f"\nImage dimensions:")
        print(f"  Min width: {stats['image_sizes']['min_width']}")
        print(f"  Max width: {stats['image_sizes']['max_width']}")
        print(f"  Avg width: {stats['image_sizes']['avg_width']:.1f}")

        print(f"\nBounding box sizes:")
        print(f"  Min area: {stats['bbox_sizes']['min_area']:.1f}")
        print(f"  Max area: {stats['bbox_sizes']['max_area']:.1f}")
        print(f"  Avg area: {stats['bbox_sizes']['avg_area']:.1f}")

        print(f"\nAnnotations per image:")
        print(f"  Min: {int(stats['annot_per_image']['min'])}")
        print(f"  Max: {int(stats['annot_per_image']['max'])}")
        print(f"  Avg: {stats['annot_per_image']['avg']:.1f}")


if __name__ == '__main__':
    validator = COCOValidator(
        coco_json_path='/Users/wesho/repos/captcha-model/output.json',
        image_dir='/Users/wesho/repos/captcha-model/data/train',
    )

    if validator.run_validation():
        print("\n✓ All validation checks passed!")
    else:
        print("\n✗ Validation failed!")

    validator.print_statistics()
