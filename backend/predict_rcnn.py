#!/usr/bin/env python3
"""
RCNN CAPTCHA prediction script.
Uses Faster R-CNN for both segmentation and classification.
"""
import sys
import os
import json
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision.transforms.functional as F

from utils import get_device, get_model_path

# Category mapping from COCO format to characters
# Based on the annotations.json structure: category_id -> character
COCO_CLASSES = {
    1: '0', 2: '1', 3: '2', 4: '3', 5: '4', 6: '5', 7: '6', 8: '7', 9: '8', 10: '9',
    11: 'a', 12: 'b', 13: 'c', 14: 'd', 15: 'e', 16: 'f', 17: 'g', 18: 'h', 19: 'i', 20: 'j',
    21: 'k', 22: 'l', 23: 'm', 24: 'n', 25: 'o', 26: 'p', 27: 'q', 28: 'r', 29: 's', 30: 't',
    31: 'u', 32: 'v', 33: 'w', 34: 'x', 35: 'y', 36: 'z'
}


def get_model(num_classes):
    """Get Faster R-CNN model with ResNet50 backbone."""
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def prepare_image(image_path):
    """Prepare image for model inference."""
    image = Image.open(image_path).convert("RGB")
    image_tensor = F.to_tensor(image).unsqueeze(0)
    return image_tensor


def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) between two bounding boxes."""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Intersection coordinates
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    # Check if there's an intersection
    if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
        return 0.0
    
    # Calculate intersection area
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    
    # Calculate union area
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    # Avoid division by zero
    if union_area == 0:
        return 0.0
    
    # Calculate IoU
    iou = inter_area / union_area
    return iou


def remove_overlapping_boxes(prediction, iou_threshold=0.5):
    """Remove overlapping bounding boxes, keeping highest score."""
    boxes = prediction[0]["boxes"].cpu().numpy()
    labels = prediction[0]["labels"].cpu().numpy()
    scores = prediction[0]["scores"].cpu().numpy()
    
    # Sort by score (descending)
    sorted_indices = np.argsort(scores)[::-1]
    
    keep_indices = []
    
    for i in sorted_indices:
        current_box = boxes[i]
        should_keep = True
        
        # Check if this box overlaps significantly with any already-kept box
        for kept_idx in keep_indices:
            kept_box = boxes[kept_idx]
            iou = calculate_iou(current_box, kept_box)
            
            if iou > iou_threshold:
                should_keep = False
                break
        
        if should_keep:
            keep_indices.append(i)
    
    # Sort keep_indices to maintain left-to-right order
    keep_indices = sorted(keep_indices, key=lambda i: boxes[i][0])  # Sort by x1
    
    # Filter predictions
    filtered_boxes = torch.tensor(boxes[keep_indices], dtype=torch.float32)
    filtered_labels = torch.tensor(labels[keep_indices], dtype=torch.int64)
    filtered_scores = torch.tensor(scores[keep_indices], dtype=torch.float32)
    
    return {
        "boxes": filtered_boxes,
        "labels": filtered_labels,
        "scores": filtered_scores
    }


def predict_rcnn(image_path, model, device):
    """Predict CAPTCHA using RCNN model."""
    # Prepare image
    image_tensor = prepare_image(image_path).to(device)
    
    # Set model thresholds
    model.roi_heads.nms_thresh = 0.05
    model.roi_heads.score_thresh = 0.5
    
    # Predict
    model.eval()
    with torch.no_grad():
        prediction = model(image_tensor)
    
    # Remove overlapping boxes
    filtered = remove_overlapping_boxes(prediction, iou_threshold=0.5)
    
    # Extract results
    boxes = filtered["boxes"].cpu().numpy()
    labels = filtered["labels"].cpu().numpy()
    scores = filtered["scores"].cpu().numpy()
    
    # Convert to output format
    bounding_boxes = []
    characters = []
    score_list = []
    
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = box.tolist()
        bounding_boxes.append({
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2)
        })
        # Convert label to native Python int
        label_int = int(label.item()) if hasattr(label, 'item') else int(label)
        characters.append(COCO_CLASSES.get(label_int, "?"))
        score_list.append(float(score.item()) if hasattr(score, 'item') else float(score))
    
    prediction_string = ''.join(characters)
    
    return prediction_string, bounding_boxes, score_list, characters


def main():
    if len(sys.argv) != 2:
        print("Usage: python predict_rcnn.py <image_path>", file=sys.stderr)
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    device = get_device()
    
    # Load model
    model_path = get_model_path("rcnn")
    num_classes = 36 + 1  # 36 characters + background
    model = get_model(num_classes)
    model.load_state_dict(torch.load(model_path, weights_only=False))
    model.to(device)
    
    # Make prediction
    try:
        prediction, bboxes, scores, characters = predict_rcnn(image_path, model, device)
        
        # Output JSON
        result = {
            "prediction": prediction,
            "boundingBoxes": bboxes,
            "scores": scores,
            "characters": characters
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

