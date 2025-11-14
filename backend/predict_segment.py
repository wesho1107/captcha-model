#!/usr/bin/env python3
"""
Segment-then-Predict CAPTCHA prediction script.
Supports CNN, ResNet, and SqueezeNet models.
"""
import sys
import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from sklearn.cluster import DBSCAN

# Add parent directory to path to import models
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import get_device, get_model_path

# Character mapping (same as training)
CHARS = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
         "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


def flatten_and_append_coordinates(roi, spatial_weight=1.0, color_weight=1.0):
    """Flatten ROI and append normalized coordinates for DBSCAN."""
    h, w, _ = roi.shape
    Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coords = np.stack((X / w, Y / h), axis=-1).astype(np.float32)
    colors = (roi.astype(np.float32) / 255.0)
    coords *= spatial_weight
    colors *= color_weight
    flat_features = np.concatenate((coords.reshape(-1, 2), colors.reshape(-1, 3)), axis=1)
    return flat_features


def dbscan_collect_meta(roi, min_samples, eps=10, pad=3, min_pixels=30,
                        spatial_weight=1.0, color_weight=1.0):
    """Collect metadata from DBSCAN clustering."""
    h, w, _ = roi.shape
    flat_features = flatten_and_append_coordinates(roi, spatial_weight, color_weight)

    # Build a keep-mask that removes near-white and near-black pixels
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    keep_mask = ((gray < 245) & (gray > 10))
    keep_idx = np.where(keep_mask.flatten())[0]

    if len(keep_idx) == 0:
        return []

    filtered_features = flat_features[keep_idx]

    # Run DBSCAN only on the remaining pixels
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels_partial = db.fit_predict(filtered_features)

    # Reconstruct full-size label image (background = -1)
    labels = np.full((h * w), -1, dtype=np.int32)
    labels[keep_idx] = labels_partial
    labels = labels.reshape(h, w)

    # Collect metadata
    meta = []
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
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(w - 1, x2 + pad), min(h - 1, y2 + pad)
        
        # Convert to native Python ints
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

        cropped = roi[y1:y2+1, x1:x2+1].copy()
        cropped_mask = mask[y1:y2+1, x1:x2+1]
        cropped[~(cropped_mask.astype(bool))] = 255
        gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        total = gray_crop.size
        wb_ratio = float((np.sum(gray_crop >= 250) + np.sum(gray_crop <= 5)) / total)

        meta.append({
            "label": int(label),
            "bbox": (x1, y1, x2, y2),
            "mask": mask,
            "left_x": int(xs.min()),
            "wb_ratio": wb_ratio,
        })

    return meta


def segment_image_for_prediction(img_path_or_array, eps=3, spatial_weight=1.0, color_weight=100.0):
    """Segment a captcha image into individual character images."""
    # Load image
    if isinstance(img_path_or_array, str):
        img = cv2.imread(img_path_or_array)
        if img is None:
            raise ValueError(f"Could not load image from {img_path_or_array}")
    else:
        img = img_path_or_array.copy()
    
    # Convert to grayscale for contour detection
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Remove black lines via inpainting
    mask = (img_gray < 5).astype(np.uint8) * 255
    img_gray = cv2.inpaint(img_gray, mask, 3, cv2.INPAINT_TELEA)
    
    # Threshold and find contours
    ret, thresh = cv2.threshold(img_gray, 250, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get bounding boxes (skip nested contours)
    bounding_boxes = []
    for i, c in enumerate(contours):
        parent = hierarchy[0][i][3]
        if parent != -1:  # Skip if it has a parent contour
            continue
        x, y, w, h = cv2.boundingRect(c)
        bounding_boxes.append((x, y, w, h))
    
    bounding_boxes = sorted(bounding_boxes, key=lambda box: box[0])
    
    character_images = []
    char_bounding_boxes = []
    
    # Process each bounding box
    for x, y, w, h in bounding_boxes:
        roi = img[y:y + h, x:x + w]
        
        # Find usable colors
        roi_rgb = roi.reshape(-1, 3)
        colors, counts = np.unique(roi_rgb, axis=0, return_counts=True)
        
        top_color_counts = sorted(zip(counts, colors), key=lambda x: x[0], reverse=True)
        usable_colors = []
        usable_counts = []
        
        for count, color in top_color_counts[:8]:
            # Filter out white, black, and very small noise
            if np.linalg.norm(color - 255) > 5 and np.linalg.norm(color) > 5 and count > w * h * 0.01:
                usable_colors.append(color)
                usable_counts.append(count)
        
        if len(usable_counts) < 1:
            continue
        elif len(usable_counts) == 1:
            est_k = len(usable_counts)
        else:
            ratios = usable_counts[:-1] / (np.array(usable_counts[1:]) + 1e-5)
            est_k = np.argmax(ratios) + 1
        
        # Run DBSCAN on ROI
        metadata = dbscan_collect_meta(
            roi, 
            min_samples=usable_counts[est_k - 1], 
            eps=eps, 
            pad=3, 
            min_pixels=w * h * 0.01,
            spatial_weight=spatial_weight, 
            color_weight=color_weight
        )
        
        # Sort by left_x and extract character images
        metadata.sort(key=lambda x: x["left_x"])
        
        # Calculate WB threshold from all metadata
        if len(metadata) > 0:
            wb_values = np.array([m["wb_ratio"] for m in metadata])
            wb_mean, wb_std = wb_values.mean(), wb_values.std()
            wb_threshold = min(wb_mean + 3 * wb_std, 0.9)
            
            # Extract valid character images
            for cluster in metadata:
                if cluster["wb_ratio"] > wb_threshold:
                    continue
                
                x1, y1, x2, y2 = cluster["bbox"]
                mask = cluster["mask"]
                
                cropped = roi[y1:y2+1, x1:x2+1].copy()
                cropped_mask = mask[y1:y2+1, x1:x2+1]
                cropped[~(cropped_mask.astype(bool))] = 255  # white background
                
                # Convert relative coordinates to absolute coordinates (ensure native Python types)
                abs_x1 = int(x + x1)
                abs_y1 = int(y + y1)
                abs_x2 = int(x + x2)
                abs_y2 = int(y + y2)
                
                character_images.append(cropped)
                char_bounding_boxes.append((abs_x1, abs_y1, abs_x2, abs_y2))
    
    return character_images, char_bounding_boxes


def preprocess_character_image(char_img):
    """Preprocess a single character image for model prediction."""
    # Convert BGR to RGB
    char_img_rgb = cv2.cvtColor(char_img, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_img = Image.fromarray(char_img_rgb)
    
    # Apply same transforms as training
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    return transform(pil_img)


# Model definitions
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, 36)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class Fire(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand_channels):
        super().__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_bn = nn.BatchNorm2d(squeeze_channels)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand_channels, kernel_size=3, padding=1)
        self.expand_bn = nn.BatchNorm2d(2 * expand_channels)

    def forward(self, x):
        x = torch.nn.functional.relu(self.squeeze_bn(self.squeeze(x)))
        out1 = torch.nn.functional.relu(self.expand1x1(x))
        out3 = torch.nn.functional.relu(self.expand3x3(x))
        out = torch.cat([out1, out3], dim=1)
        out = self.expand_bn(out)
        return out


class SqueezeNetMini(nn.Module):
    def __init__(self, num_classes=36):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fire2 = Fire(32, 16, 32)
        self.fire3 = Fire(64, 16, 32)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fire4 = Fire(64, 32, 48)
        self.fire5 = Fire(96, 32, 48)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv_final = nn.Conv2d(96, num_classes, kernel_size=1)
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = torch.nn.functional.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.fire2(x)
        x = self.fire3(x)
        x = self.pool2(x)
        x = self.fire4(x)
        x = self.fire5(x)
        x = self.pool3(x)
        x = self.conv_final(x)
        x = self.global_avgpool(x)
        x = torch.flatten(x, 1)
        return torch.nn.functional.log_softmax(x, dim=1)


def predict_captcha(img_path, model, device, model_type='squeezenet', 
                    eps=3, spatial_weight=1.0, color_weight=100.0):
    """Complete pipeline: segments a captcha image and predicts the text."""
    # Segment image into characters
    character_images, bounding_boxes = segment_image_for_prediction(
        img_path, 
        eps=eps, 
        spatial_weight=spatial_weight, 
        color_weight=color_weight
    )
    
    if len(character_images) == 0:
        return "", [], []
    
    # Preprocess all characters
    processed_chars = []
    for char_img in character_images:
        processed = preprocess_character_image(char_img)
        processed_chars.append(processed)
    
    # Stack into batch
    batch = torch.stack(processed_chars).to(device)
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(batch)
        
        # Handle different model output formats
        if model_type == 'squeezenet':
            # SqueezeNet outputs log_softmax, so we need to exp it to get probabilities
            probs = torch.exp(outputs)
        else:
            # CNN and ResNet output raw logits, apply softmax to get probabilities
            probs = torch.softmax(outputs, dim=1)
        
        # Get predicted class and its confidence score
        scores, predicted = torch.max(probs.data, 1)
        predicted_chars = [CHARS[pred.item()] for pred in predicted]
        confidence_scores = [score.item() for score in scores]
    
    prediction_string = ''.join(predicted_chars)
    return prediction_string, bounding_boxes, confidence_scores


def main():
    if len(sys.argv) != 3:
        print("Usage: python predict_segment.py <model_name> <image_path>", file=sys.stderr)
        sys.exit(1)
    
    model_name = sys.argv[1].lower()
    image_path = sys.argv[2]
    
    if model_name not in ['cnn', 'resnet', 'squeezenet']:
        print(f"Error: Unknown model '{model_name}'. Must be one of: cnn, resnet, squeezenet", file=sys.stderr)
        sys.exit(1)
    
    device = get_device()
    
    # Load model
    model_path = get_model_path(model_name)
    
    if model_name == 'cnn':
        model = CNN().to(device)
    elif model_name == 'resnet':
        from torchvision.models import resnet18
        model = resnet18(weights=None)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, 36)
        model = model.to(device)
    elif model_name == 'squeezenet':
        model = SqueezeNetMini(num_classes=36).to(device)
    
    model.load_state_dict(torch.load(model_path, weights_only=False))
    
    # Make prediction
    try:
        prediction, bboxes, scores = predict_captcha(
            image_path, 
            model, 
            device, 
            model_type=model_name
        )
        
        # Convert bounding boxes to list format with native Python types
        bounding_boxes = []
        for b in bboxes:
            bounding_boxes.append({
                "x1": float(b[0]),
                "y1": float(b[1]),
                "x2": float(b[2]),
                "y2": float(b[3])
            })
        
        # Create characters list
        characters = list(prediction)
        
        # Convert scores to native Python floats
        scores_list = [float(score) for score in scores]
        
        # Output JSON
        result = {
            "prediction": prediction,
            "boundingBoxes": bounding_boxes,
            "scores": scores_list,
            "characters": characters
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

