# imports
import os
import cv2
import numpy as np
import torch
from sklearn.cluster import KMeans
from PIL import Image
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

chars = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z",
         "0","1","2","3","4","5","6","7","8","9"]

#### MODEL DECLARATIONS ####
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# most basic CNN model
class CNN(nn.Module):
  def __init__(self):
    super(CNN, self).__init__()
    self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
    self.fc1 = nn.Linear(64 * 8 * 8, 128)
    self.fc2 = nn.Linear(128, 36)  # Assuming 36 classes (0-9, A-Z)
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
  
cnn_model = CNN().to(device)

resnet_model = resnet18(weights=None)
resnet_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, 36)  # Assuming 36 classes (0-9, A-Z)
resnet_model = resnet_model.to(device)

import torch.nn.functional as F


class Fire(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand_channels):
        super().__init__()
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_bn = nn.BatchNorm2d(squeeze_channels)

        self.expand1x1 = nn.Conv2d(squeeze_channels, expand_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand_channels, kernel_size=3, padding=1)
        self.expand_bn = nn.BatchNorm2d(2 * expand_channels)

    def forward(self, x):
        x = F.relu(self.squeeze_bn(self.squeeze(x)))
        out1 = F.relu(self.expand1x1(x))
        out3 = F.relu(self.expand3x3(x))
        out = torch.cat([out1, out3], dim=1)
        out = self.expand_bn(out)
        return out

class SqueezeNetMini(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Input: 3×32×32
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # 32→16

        self.fire2 = Fire(32, 16, 32)   # 16 squeeze, 32 expand
        self.fire3 = Fire(64, 16, 32)
        self.pool2 = nn.MaxPool2d(2, 2)  # 16→8

        self.fire4 = Fire(64, 32, 48)
        self.fire5 = Fire(96, 32, 48)
        self.pool3 = nn.MaxPool2d(2, 2)  # 8→4

        self.conv_final = nn.Conv2d(96, num_classes, kernel_size=1)
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
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
        return F.log_softmax(x, dim=1)

squeezenet_model = SqueezeNetMini(num_classes=36).to(device)

class ConvBNReLU(nn.Sequential):
    """Standard 2D conv + batchnorm + ReLU6."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )

class InvertedResidual(nn.Module):
    """MobileNetV2 residual block."""
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        self.stride = stride
        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # 1×1 expand
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1))
        # 3×3 depthwise
        layers.append(ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim))
        # 1×1 linear projection
        layers.append(nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetMini(nn.Module):
    def __init__(self, num_classes=36, width_mult=1.0):
        super().__init__()
        input_channel = int(32 * width_mult)
        last_channel = int(1280 * width_mult)

        # Input: grayscale 1×32×32
        features = [ConvBNReLU(1, input_channel, stride=1)]  # keep resolution

        # (t, c, n, s): expand ratio, channels, repeats, stride
        inverted_residual_setting = [
            # Designed for 32×32 (reduced depth)
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 2, 1],
        ]

        for t, c, n, s in inverted_residual_setting:
            out_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(InvertedResidual(input_channel, out_channel, stride, expand_ratio=t))
                input_channel = out_channel

        # Last conv + pooling
        features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(last_channel, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

mobilenet_model = MobileNetMini(num_classes=36).to(device)

### HELPERS ###
from sklearn.cluster import DBSCAN

def flatten_and_append_coordinates(roi, color_weight=1.0, spatial_weight=1.0):
    h, w, _ = roi.shape
    Y, X = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    coords = np.stack((X / w, Y / h), axis=-1).astype(np.float32)  # normalize to [0,1]
    colors = (roi.astype(np.float32) / 255.0)                      # normalize to [0,1]

    # apply weights to control influence
    coords *= spatial_weight
    colors *= color_weight

    flat_features = np.concatenate(
        (coords.reshape(-1, 2), colors.reshape(-1, 3)), axis=1
    )
    return flat_features

def dbscan_collect_meta(roi, min_samples, eps=10, pad=3, min_pixels=30,
                        spatial_weight=1.0, color_weight=1.0):
    h, w, _ = roi.shape
    flat_features = flatten_and_append_coordinates(roi, spatial_weight=spatial_weight, color_weight=color_weight)

    # --- 1️⃣ build a keep-mask that removes near-white and near-black pixels ---
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    keep_mask = ((gray < 245) & (gray > 10))      # 0-10 ≈ black, 245-255 ≈ white
    keep_idx = np.where(keep_mask.flatten())[0]

    if len(keep_idx) == 0:
        print("Warning: no usable pixels for DBSCAN.")
        return []

    filtered_features = flat_features[keep_idx]

    # --- 2️⃣ run DBSCAN only on the remaining pixels ---
    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels_partial = db.fit_predict(filtered_features)

    # --- 3️⃣ reconstruct full-size label image (background = -1) ---
    labels = np.full((h * w), -1, dtype=np.int32)
    labels[keep_idx] = labels_partial
    labels = labels.reshape(h, w)

    # --- 4️⃣ continue as before ---
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

        cropped = roi[y1:y2+1, x1:x2+1].copy()
        cropped_mask = mask[y1:y2+1, x1:x2+1]
        cropped[~(cropped_mask.astype(bool))] = 255
        gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        total = gray_crop.size
        wb_ratio = (np.sum(gray_crop >= 250) + np.sum(gray_crop <= 5)) / total

        meta.append({
            "label": label,
            "bbox": (x1, y1, x2, y2),
            "mask": mask,
            "left_x": xs.min(),
            "wb_ratio": wb_ratio,
        })

    return meta

### INFERENCE FUNCTIONS ###

def inference(model, weight_path, image_path, show_rois=False):
    model.load_state_dict(torch.load(weight_path, map_location=torch.device('cpu')))
    model.eval()
    return dbscan_inference(model, image_path, show_roi=show_rois)

def predict_partial(model, roi, metadata):
    confidence_results = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    predicted_partial = ""
    for cluster in metadata:
        x1, y1, x2, y2 = cluster["bbox"]
        mask = cluster["mask"]

        cropped = roi[y1:y2+1, x1:x2+1].copy()
        cropped_mask = mask[y1:y2+1, x1:x2+1]
        cropped[~(cropped_mask.astype(bool))] = 255  # white background

        gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        pil_img = Image.fromarray(gray_crop).convert('L')
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        input_tensor = transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            conf, predicted = torch.max(torch.softmax(output, dim=1), 1)
            confidence = conf.item()
            predicted_char = chars[predicted.item()]
            predicted_partial += predicted_char
            confidence_results.append({
                "char": predicted_char,
                "confidence": confidence,
                "bbox": cluster["bbox"],
            })
    return predicted_partial, confidence_results

def dbscan_inference(model, img_path, eps=10, spatial_weight=1.0, color_weight=100.0, show_roi=False):
    bounding_boxes = []
    index = 0
    rois_to_analyse = []
    
    # threshold letters and find contours
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return
    mask = (img < 5).astype(np.uint8) * 255
    img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    ret, thresh = cv2.threshold(img, 250, 255, cv2.THRESH_BINARY_INV)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for i, c in enumerate(contours):
        parent = hierarchy[0][i][3]
        if parent != -1:  # Skip if it has a parent contour    
            continue
        x, y, w, h = cv2.boundingRect(c)
        bounding_boxes.append((x, y, w, h))
        
    bounding_boxes = sorted(bounding_boxes, key=lambda box: box[0])

    
    
    for b in bounding_boxes:
        x, y, w, h = b
        # for this local scope, COLOR is important
        img = cv2.imread(img_path)
        roi = img[y:y + h, x:x + w]
        # try and see how many unique colors are in this roi
        roi_rgb = roi.reshape(-1, 3)
        colors, counts = np.unique(roi_rgb, axis=0, return_counts=True)
        # print(f"Unique colors: {len(colors)}")

        top_color_counts = sorted(zip(counts, colors), key=lambda x: x[0], reverse=True)
        usable_colors = []
        usable_counts = []
        for count, color in top_color_counts:
        # filter out white, black, and very small noise
            if np.linalg.norm(color - 255) > 5 and np.linalg.norm(color) > 5 and count > w * h * 0.01:
                usable_colors.append(color)
                usable_counts.append(count)

        # safety check: If there's only the background color or less, skip
        est_k = 0
        if len(usable_counts) < 1:
            continue
        elif len(usable_counts) == 1:
            print(usable_counts)
            est_k = len(usable_counts)
        else:
            ratios = usable_counts[:-1] / (np.array(usable_counts[1:]) + 1e-5)  # avoid div by 0
            est_k = np.argmax(ratios) + 1 # This is needed to get the elbow value for DBSCAN

        # print(f"Usable counts: {usable_counts[est_k - 1]}")
        metadata = dbscan_collect_meta(roi, min_samples=usable_counts[est_k - 1], eps=eps, pad=3, min_pixels=w * h * 0.01,
                        spatial_weight=spatial_weight, color_weight=color_weight)
        
        # Both bounding boxes and DBSCAN sort from left to right, so the order is preserved
        # print(f"DBSCAN found {len(metadata)} clusters in ROI.")
        rois_to_analyse.append((roi, metadata, b))

    # read in a fresh img in case show_roi is on
    img = cv2.imread(img_path)
    predicted_string = ""
    for roi, metadata, bbox in rois_to_analyse:
        partial_string, confidence_results = predict_partial(model, roi, metadata)
        predicted_string += partial_string
        for i, res in enumerate(confidence_results):
            x, y, w, h = bbox
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, f"{res['char']}:{res['confidence']:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)

        
    # print(f"Predicted string: {predicted_string}")
    if show_roi:
        cv2.imshow("Predictions", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return predicted_string, img


### CREATE A MINI SERVER ###
from flask import Flask, request, jsonify, send_file
import os
import tempfile
import base64

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided.'}), 400

    image_file = request.files['image']
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_image:
        image_path = temp_image.name
        image_file.save(image_path)

    try:
        model_name: str = request.form.get('model', 'cnn') # default to cnn

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if "cnn" in model_name.lower():
            model = CNN().to(device)
            weight_path = 'captcha_model_cnn.pth'
        elif "resnet" in model_name.lower():
            model = resnet_model.to(device)
            weight_path = 'captcha_model_resnet.pth'
        elif "squeezenet" in model_name.lower():
            model = squeezenet_model.to(device)
            weight_path = 'captcha_model_squeezenet.pth'
        elif "mobilenet" in model_name.lower():
            model = mobilenet_model.to(device)
            weight_path = 'captcha_model_mobilenet.pth'
        else:
            # default to cnn
            model = CNN().to(device)
            weight_path = 'captcha_model_cnn.pth'
      
        predicted_string, annotated_img = inference(model, weight_path, image_path, show_rois=True)

        temp_out_path = image_path.replace('.png', '_annotated.png')
        cv2.imwrite(temp_out_path, annotated_img)

        # try send both predicted string and image, make img base64
        _, buffer = cv2.imencode('.png', annotated_img)
        img_bytes = base64.b64encode(buffer).decode('utf-8')
        response = jsonify({'predicted_string': predicted_string, 'annotated_image': img_bytes})
        return response
    finally:
        os.remove(image_path)
        if os.path.exists(temp_out_path):
            os.remove(temp_out_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)