# Uses DBSCAN to segment characters from a CAPTCHA image

# imports
import os
import cv2
import numpy as np
import torch
from sklearn.cluster import DBSCAN
from PIL import Image
from typing import List, Dict, Tuple, Optional, Union
import argparse
import random
import json
import shutil


class DBSCANSegmentor:
    """
    DBSCAN-based character segmentor for CAPTCHA images.
    Converts CAPTCHA images into RCNN training data format with bounding boxes and labels.
    """
    
    def __init__(self, 
                 eps: float = 3.0,
                 spatial_weight: float = 1.0,
                 color_weight: float = 100.0,
                 pad: int = 3,
                 min_pixels_ratio: float = 0.01,
                 wb_threshold_std: float = 3.0,
                 show_roi: bool = False):
        """
        Initialize DBSCAN segmentor with configuration parameters.
        
        Args:
            eps: DBSCAN eps parameter for clustering
            spatial_weight: Weight for spatial coordinates in feature vector
            color_weight: Weight for color values in feature vector
            pad: Padding around bounding boxes
            min_pixels_ratio: Minimum pixels as ratio of ROI size
            wb_threshold_std: Standard deviations for white/black ratio threshold
            show_roi: Whether to display ROIs during processing
        """
        self.eps = eps
        self.spatial_weight = spatial_weight
        self.color_weight = color_weight
        self.pad = pad
        self.min_pixels_ratio = min_pixels_ratio
        self.wb_threshold_std = wb_threshold_std
        self.show_roi = show_roi
        
        # Will be populated when processing images
        self.charset = None
        self.char_to_idx = None
        self.idx_to_char = None
    
    def _flatten_and_append_coordinates(self, roi: np.ndarray) -> np.ndarray:
        """
        Flatten ROI and append normalized coordinates to color features.
        
        Args:
            roi: Region of interest image (H, W, 3)
            
        Returns:
            Feature matrix (H*W, 5) with [x_norm, y_norm, r, g, b]
        """
        h, w, _ = roi.shape
        Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        coords = np.stack((X / w, Y / h), axis=-1).astype(np.float32)  # normalize to [0,1]
        colors = (roi.astype(np.float32) / 255.0)                      # normalize to [0,1]

        # apply weights to control influence
        coords *= self.spatial_weight
        colors *= self.color_weight

        flat_features = np.concatenate(
            (coords.reshape(-1, 2), colors.reshape(-1, 3)), axis=1
        )
        return flat_features

    def _dbscan_collect_meta(self, roi: np.ndarray, min_samples: int) -> List[Dict]:
        """
        Run DBSCAN on ROI and collect cluster metadata.
        
        Args:
            roi: Region of interest image (H, W, 3)
            min_samples: Minimum samples for DBSCAN cluster
            
        Returns:
            List of metadata dicts with bbox, mask, wb_ratio, etc.
        """
        h, w, _ = roi.shape
        flat_features = self._flatten_and_append_coordinates(roi)

        # Build a keep-mask that removes near-white and near-black pixels
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        keep_mask = ((gray < 245) & (gray > 10))      # 0-10 ≈ black, 245-255 ≈ white
        keep_idx = np.where(keep_mask.flatten())[0]

        if len(keep_idx) == 0:
            return []

        filtered_features = flat_features[keep_idx]

        # Run DBSCAN only on the remaining pixels
        db = DBSCAN(eps=self.eps, min_samples=min_samples)
        labels_partial = db.fit_predict(filtered_features)

        # Reconstruct full-size label image (background = -1)
        labels = np.full((h * w), -1, dtype=np.int32)
        labels[keep_idx] = labels_partial
        labels = labels.reshape(h, w)

        # Collect metadata for each cluster
        meta = []
        min_pixels = int(w * h * self.min_pixels_ratio)
        
        for label in np.unique(labels):
            if label == -1:
                continue

            mask = (labels == label).astype(np.uint8) * 255
            if cv2.countNonZero(mask) < min_pixels:
                continue

            ys, xs = np.where(mask > 0)
            if len(xs) == 0:
                continue

            x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
            x1, y1 = max(0, x1 - self.pad), max(0, y1 - self.pad)
            x2, y2 = min(w - 1, x2 + self.pad), min(h - 1, y2 + self.pad)

            cropped = roi[y1:y2+1, x1:x2+1].copy()
            cropped_mask = mask[y1:y2+1, x1:x2+1]
            cropped[~(cropped_mask.astype(bool))] = 255
            gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            total = gray_crop.size
            wb_ratio = (np.sum(gray_crop >= 250) + np.sum(gray_crop <= 5)) / total

            meta.append({
                "label": label,
                "bbox": (x1, y1, x2, y2),  # ROI-relative coordinates
                "mask": mask,
                "left_x": xs.min(),
                "wb_ratio": wb_ratio,
            })

        return meta

    def _find_rois(self, img: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Find regions of interest using contour detection.
        
        Args:
            img: Grayscale image
            
        Returns:
            List of (x, y, w, h) bounding boxes
        """
        mask = (img < 5).astype(np.uint8) * 255
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        ret, thresh = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY_INV)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        bounding_boxes = []
        for i, c in enumerate(contours):
            parent = hierarchy[0][i][3]
            if parent != -1:  # Skip if it has a parent contour
                continue
            x, y, w, h = cv2.boundingRect(c)
            bounding_boxes.append((x, y, w, h))
        
        return sorted(bounding_boxes, key=lambda box: box[0])

    def _estimate_min_samples(self, roi: np.ndarray) -> int:
        """
        Estimate min_samples for DBSCAN based on color distribution in ROI.
        
        Args:
            roi: Region of interest image
            
        Returns:
            Estimated min_samples value
        """
        h, w = roi.shape[:2]
        roi_rgb = roi.reshape(-1, 3)
        colors, counts = np.unique(roi_rgb, axis=0, return_counts=True)

        top_color_counts = sorted(zip(counts, colors), key=lambda x: x[0], reverse=True)
        usable_counts = []
        for count, color in top_color_counts[:8]:
            # Filter out white, black, and very small noise
            if np.linalg.norm(color - 255) > 5 and np.linalg.norm(color) > 5 and count > w * h * self.min_pixels_ratio:
                usable_counts.append(count)

        if len(usable_counts) < 1:
            return 0
        elif len(usable_counts) == 1:
            return usable_counts[0]
        else:
            ratios = usable_counts[:-1] / (np.array(usable_counts[1:]) + 1e-5)
            est_k = np.argmax(ratios) + 1
            return usable_counts[est_k - 1]

    def _process_roi(self, roi: np.ndarray, roi_bbox: Tuple[int, int, int, int]) -> List[Dict]:
        """
        Process a single ROI and return cluster metadata.
        
        Args:
            roi: Region of interest image
            roi_bbox: (x, y, w, h) of ROI in original image
            
        Returns:
            List of metadata dicts with ROI-relative bboxes
        """
        min_samples = self._estimate_min_samples(roi)
        if min_samples == 0:
            return []
        
        metadata = self._dbscan_collect_meta(roi, min_samples=min_samples)
        return metadata

    def _filter_clusters(self, all_meta: List[Dict]) -> Tuple[List[Dict], float]:
        """
        Filter clusters based on white/black ratio threshold.
        
        Args:
            all_meta: List of all cluster metadata
            
        Returns:
            Tuple of (filtered_meta, threshold)
        """
        if not all_meta:
            return [], 0.9
        
        wb_values = np.array([m["wb_ratio"] for m in all_meta])
        wb_mean, wb_std = wb_values.mean(), wb_values.std()
        threshold = min(wb_mean + self.wb_threshold_std * wb_std, 0.9)
        
        filtered = [m for m in all_meta if m["wb_ratio"] <= threshold]
        return filtered, threshold

    def _build_charset(self, image_paths: List[str]) -> None:
        """
        Build character set and mappings from image filenames.
        
        Args:
            image_paths: List of image file paths
        """
        charset = set()
        for path in image_paths:
            filename = os.path.basename(path)
            captcha_name = os.path.splitext(filename)[0]
            # Remove suffix like "-0" if present
            if captcha_name.endswith('-0') or captcha_name.endswith('-1'):
                captcha_name = captcha_name[:-2]
            charset.update(captcha_name)
        
        self.charset = sorted(list(charset))
        # 0 is reserved for background in PyTorch, so start from 1
        self.char_to_idx = {c: i + 1 for i, c in enumerate(self.charset)}
        self.idx_to_char = {i + 1: c for i, c in enumerate(self.charset)}

    def segment_image(self, image_path: str) -> Dict:
        """
        Segment a single CAPTCHA image and return results.
        
        Args:
            image_path: Path to CAPTCHA image
            
        Returns:
            Dict with:
                - 'image': Full image array
                - 'rois': List of (roi, metadata, roi_bbox) tuples
                - 'captcha_name': Extracted captcha name from filename
        """
        filename = os.path.basename(image_path)
        captcha_name = os.path.splitext(filename)[0]
        # Remove suffix like "-0" if present
        if captcha_name.endswith('-0') or captcha_name.endswith('-1'):
            captcha_name = captcha_name[:-2]
        
        # Load image
        img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img_gray is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        img_color = cv2.imread(image_path)
        if img_color is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Find ROIs
        bounding_boxes = self._find_rois(img_gray)
        
        rois_data = []
        for x, y, w, h in bounding_boxes:
            roi = img_color[y:y + h, x:x + w]
            
            if self.show_roi:
                vis_img = img_color.copy()
                cv2.rectangle(vis_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.imshow("ROI", vis_img)
                cv2.imshow("ROI raw", roi)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
            metadata = self._process_roi(roi, (x, y, w, h))
            if metadata:
                rois_data.append((roi, metadata, (x, y, w, h)))
        
        return {
            'image': img_color,
            'rois': rois_data,
            'captcha_name': captcha_name,
            'image_path': image_path
        }

    def get_training_data(self, 
                         data_path: str, 
                         charset: Optional[List[str]] = None) -> List[Dict]:
        """
        Process all images in data_path and return RCNN training data format.
        
        Args:
            data_path: Path to directory containing CAPTCHA images
            charset: Optional pre-defined charset. If None, builds from filenames.
            
        Returns:
            List of dicts, each containing:
                - 'boxes': Bounding boxes as list [N, 4] in [x1, y1, x2, y2] format (absolute coordinates)
                - 'labels': Character labels as list [N] (1-indexed, 0=background)
                - 'image_id': Original filename
        """
        # Collect all image paths
        image_paths = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith('.png') or file.endswith('.jpg'):
                    image_paths.append(os.path.join(root, file))
        
        if not image_paths:
            raise ValueError(f"No images found in {data_path}")
        
        # Build charset
        if charset is None:
            self._build_charset(image_paths)
        else:
            self.charset = charset
            self.char_to_idx = {c: i + 1 for i, c in enumerate(self.charset)}
            self.idx_to_char = {i + 1: c for i, c in enumerate(self.charset)}
        
        # Process all images and collect all metadata first for global threshold calculation
        all_segments = []
        all_meta_global = []
        for image_path in image_paths:
            result = self.segment_image(image_path)
            all_segments.append(result)
            # Collect metadata for global threshold
            for roi, meta, _ in result['rois']:
                all_meta_global.extend(meta)
        
        # Calculate global WB threshold
        _, global_threshold = self._filter_clusters(all_meta_global)
        
        # Build training data
        training_data = []
        for seg in all_segments:
            captcha_name = seg['captcha_name']
            captcha_chars = list(captcha_name)
            img = seg['image']
            
            boxes = []
            box_metadata = []  # Store metadata with boxes for sorting
            
            # Collect all valid clusters from this image
            for roi, meta, roi_bbox in seg['rois']:
                x_roi, y_roi, w_roi, h_roi = roi_bbox
                
                # Filter clusters by global threshold and convert coordinates
                for m in meta:
                    if m['wb_ratio'] > global_threshold:
                        continue
                    
                    # Convert ROI-relative bbox to absolute coordinates
                    x1_rel, y1_rel, x2_rel, y2_rel = m['bbox']
                    x1_abs = x_roi + x1_rel
                    y1_abs = y_roi + y1_rel
                    x2_abs = x_roi + x2_rel
                    y2_abs = y_roi + y2_rel
                    
                    boxes.append([x1_abs, y1_abs, x2_abs, y2_abs])
                    box_metadata.append(m['left_x'] + x_roi)  # Store absolute left_x for sorting
            
            # Sort boxes by left x coordinate
            if boxes:
                sorted_indices = sorted(range(len(boxes)), key=lambda i: box_metadata[i])
                boxes = [boxes[i] for i in sorted_indices]

                # Assign labels based on sorted order
                if len(boxes) != len(captcha_chars):
                    print(f"Skipping {seg['image_path']}: expected {len(captcha_chars)} chars but detected {len(boxes)} boxes.")
                    continue

                # Convert labels to indices
                label_indices = [self.char_to_idx.get(c, 0) for c in captcha_chars]

                training_data.append({
                    'boxes': boxes,  # Already a list, no need for numpy array
                    'labels': label_indices,  # Already a list
                    'image_id': os.path.basename(seg['image_path']),
                    'width': img.shape[1],
                    'height': img.shape[0]
                })

        return training_data

    def _convert_to_native_types(self, obj):
        """
        Recursively convert NumPy types to native Python types for JSON serialization.
        
        Args:
            obj: Object that may contain NumPy types
            
        Returns:
            Object with all NumPy types converted to native Python types
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_to_native_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_native_types(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(self._convert_to_native_types(item) for item in obj)
        else:
            return obj

    def save_training_data_to_json(self, 
                                  training_data: List[Dict], 
                                  output_path: str,
                                  charset: Optional[List[str]] = None) -> None:
        """
        Save training data to a JSON file.
        
        Args:
            training_data: Output from get_training_data() method
            output_path: Path to save JSON file
            charset: Optional charset to include in JSON metadata
        """
        # Prepare data for JSON serialization
        if charset is not None:
            self.charset = charset
            self.char_to_idx = {c: i + 1 for i, c in enumerate(self.charset)}
        if not self.char_to_idx:
            raise ValueError("Character to index mapping is not initialized.")

        # COCO components
        images = []
        annotations = []
        categories = [
            {
                'id': idx,
                'name': char
            }
            for char, idx in sorted(self.char_to_idx.items(), key=lambda kv: kv[1])
        ]

        image_id_mapping = {}
        annotation_id = 1

        for idx, item in enumerate(training_data, start=1):
            filename = item['image_id']
            width = item.get('width')
            height = item.get('height')
            if width is None or height is None:
                raise ValueError(f"Training data item for {filename} is missing width/height.")

            images.append({
                'id': idx,
                'file_name': filename,
                'width': width,
                'height': height
            })
            image_id_mapping[filename] = idx

            boxes = item['boxes']
            labels = item['labels']
            if len(boxes) != len(labels):
                raise ValueError(f"Mismatched boxes/labels lengths for {filename}.")

            for box, label in zip(boxes, labels):
                if label == 0:
                    continue  # skip background if present
                x1, y1, x2, y2 = box
                width_box = max(0, x2 - x1)
                height_box = max(0, y2 - y1)
                area = width_box * height_box
                annotations.append({
                    'id': annotation_id,
                    'image_id': idx,
                    'category_id': int(label),
                    'bbox': [x1, y1, width_box, height_box],
                    'area': area,
                    'iscrowd': 0,
                    'segmentation': [] # Optional: Add segmented data
                })
                annotation_id += 1

        json_data = {
            'info': {
                'description': 'DBSCAN Segmentor Dataset',
                'version': '1.0',
                'num_images': len(images),
                'num_annotations': len(annotations)
            },
            'licenses': [],
            'images': images,
            'annotations': annotations,
            'categories': categories
        }
        
        json_data = self._convert_to_native_types(json_data)
        
        # Save to JSON file
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Saved training data to {output_path}")
        print(f"  - {len(training_data)} images")

    def split_and_save_training_data(self,
                                     training_data: List[Dict],
                                     original_data_path: str,
                                     output_dir: str,
                                     train_ratio: float = 0.7,
                                     charset: Optional[List[str]] = None,
                                     seed: int = 42) -> None:
        """
        Split training data into train/test sets, copy images to respective folders,
        and save COCO format JSON files for each split.
        
        Args:
            training_data: Output from get_training_data() method
            original_data_path: Path to directory containing original CAPTCHA images
            output_dir: Path to directory where train/ and test/ folders will be created
            train_ratio: Ratio of data to use for training (default: 0.7, so 70% train, 30% test)
            charset: Optional charset to include in JSON metadata
            seed: Random seed for reproducibility (default: 42)
        """
        if not self.char_to_idx:
            if charset is not None:
                self.charset = charset
                self.char_to_idx = {c: i + 1 for i, c in enumerate(self.charset)}
            else:
                raise ValueError("Character to index mapping is not initialized. "
                               "Either provide charset or call get_training_data first.")
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Shuffle training data
        shuffled_data = training_data.copy()
        random.shuffle(shuffled_data)
        
        # Split into train and test
        split_idx = int(len(shuffled_data) * train_ratio)
        train_data = shuffled_data[:split_idx]
        test_data = shuffled_data[split_idx:]
        
        print(f"Splitting {len(training_data)} images into:")
        print(f"  - Train: {len(train_data)} images ({len(train_data)/len(training_data)*100:.1f}%)")
        print(f"  - Test: {len(test_data)} images ({len(test_data)/len(training_data)*100:.1f}%)")
        
        # Create output directories
        train_dir = os.path.join(output_dir, 'train')
        test_dir = os.path.join(output_dir, 'test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)
        
        # Helper function to find original image path
        def find_image_path(image_id: str, search_path: str) -> Optional[str]:
            """Find the full path to an image file by searching in search_path."""
            for root, dirs, files in os.walk(search_path):
                if image_id in files:
                    return os.path.join(root, image_id)
            return None
        
        # Helper function to save split data
        def save_split(split_data: List[Dict], split_dir: str, split_name: str) -> None:
            """Save images and COCO JSON for a split."""
            images = []
            annotations = []
            categories = [
                {
                    'id': idx,
                    'name': char
                }
                for char, idx in sorted(self.char_to_idx.items(), key=lambda kv: kv[1])
            ]
            
            annotation_id = 1
            image_id = 1
            copied_count = 0
            missing_count = 0
            
            for item in split_data:
                filename = item['image_id']
                width = item.get('width')
                height = item.get('height')
                if width is None or height is None:
                    print(f"Warning: Skipping {filename} - missing width/height.")
                    continue
                
                # Find and copy image
                original_image_path = find_image_path(filename, original_data_path)
                if original_image_path is None:
                    print(f"Warning: Could not find image {filename} in {original_data_path}")
                    missing_count += 1
                    continue
                
                # Copy image to split directory
                dest_image_path = os.path.join(split_dir, filename)
                try:
                    shutil.copy2(original_image_path, dest_image_path)
                    copied_count += 1
                except Exception as e:
                    print(f"Warning: Failed to copy image {filename}: {e}")
                    missing_count += 1
                    continue
                
                # Add to images list
                current_image_id = image_id
                images.append({
                    'id': current_image_id,
                    'file_name': filename,
                    'width': width,
                    'height': height
                })
                image_id += 1
                
                # Add annotations
                boxes = item['boxes']
                labels = item['labels']
                if len(boxes) != len(labels):
                    print(f"Warning: Mismatched boxes/labels lengths for {filename}.")
                    continue
                
                for box, label in zip(boxes, labels):
                    if label == 0:
                        continue  # skip background if present
                    x1, y1, x2, y2 = box
                    width_box = max(0, x2 - x1)
                    height_box = max(0, y2 - y1)
                    area = width_box * height_box
                    annotations.append({
                        'id': annotation_id,
                        'image_id': current_image_id,
                        'category_id': int(label),
                        'bbox': [x1, y1, width_box, height_box],
                        'area': area,
                        'iscrowd': 0,
                        'segmentation': []
                    })
                    annotation_id += 1
            
            # Create COCO JSON structure
            json_data = {
                'info': {
                    'description': f'DBSCAN Segmentor Dataset - {split_name}',
                    'version': '1.0',
                    'num_images': len(images),
                    'num_annotations': len(annotations)
                },
                'licenses': [],
                'images': images,
                'annotations': annotations,
                'categories': categories
            }
            
            json_data = self._convert_to_native_types(json_data)
            
            # Save JSON file
            json_path = os.path.join(split_dir, 'annotations.json')
            with open(json_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            print(f"Saved {split_name} split to {split_dir}")
            print(f"  - Copied {copied_count} images")
            print(f"  - Missing {missing_count} images")
            print(f"  - {len(images)} images in JSON")
            print(f"  - {len(annotations)} annotations")
            print(f"  - JSON saved to {json_path}")
        
        # Save train and test splits
        save_split(train_data, train_dir, 'train')
        save_split(test_data, test_dir, 'test')
        
        print(f"\nSuccessfully split and saved training data to {output_dir}")

    def save_to_files(self, 
                     roi: np.ndarray, 
                     meta: List[Dict], 
                     out_dir: str, 
                     captcha_name: str, 
                     idx: int = 0) -> None:
        """
        Save segmented characters to files (for debugging/visualization).
        
        Args:
            roi: Region of interest image
            meta: List of cluster metadata
            out_dir: Output directory
            captcha_name: Captcha name string
            idx: Starting index for character labels
        """
        os.makedirs(out_dir, exist_ok=True)
        meta.sort(key=lambda x: x["left_x"])
        
        wb_values = np.array([m["wb_ratio"] for m in meta])
        wb_mean, wb_std = wb_values.mean(), wb_values.std()
        wb_threshold = min(wb_mean + wb_std, 0.9)
        
        if self.show_roi:
            cv2.imshow("ROI", roi)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        for order, cluster in enumerate(meta):
            if cluster["wb_ratio"] > wb_threshold:
                print(f"Skipping cluster {order} due to high WB ratio: {cluster['wb_ratio']:.3f}")
                continue

            x1, y1, x2, y2 = cluster["bbox"]
            mask = cluster["mask"]

            cropped = roi[y1:y2+1, x1:x2+1].copy()
            cropped_mask = mask[y1:y2+1, x1:x2+1]
            cropped[~(cropped_mask.astype(bool))] = 255  # white background

            # Create unique filename
            base = f"{captcha_name[idx]}_{captcha_name}"
            counter = 0
            filename = os.path.join(out_dir, f"{base}.png")
            while os.path.exists(filename):
                counter += 1
                filename = os.path.join(out_dir, f"{base}-{counter}.png")

            cv2.imwrite(filename, cropped)
            print(f"Saved cluster {order}: {filename}")
            idx += 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DBSCAN Segmentor")
    parser.add_argument("data_path", type=str, help="Path to directory containing CAPTCHA images")
    parser.add_argument("out_path", type=str, help="Path to output directory where train/ and test/ folders will be created")
    args = parser.parse_args()
    
    segmentor = DBSCANSegmentor()

    training_data = segmentor.get_training_data(args.data_path)
    # segmentor.save_training_data_to_json(training_data, args.out_path)  # Old: expects file path like 'output.json'
    segmentor.split_and_save_training_data(training_data, args.data_path, args.out_path)
    # To run and split data into train/test:
    # python rcnn/dbscan_segmentor.py data/train output/split_data