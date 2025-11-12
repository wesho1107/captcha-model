from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import kagglehub
import os
import random
import string
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# -------------------------
# CONFIG
# -------------------------
CKPT_DIR = "/content/drive/MyDrive/Alphanum_GAN/output/checkpoints"
OUTPUT_DIR = "/content/drive/MyDrive/Alphanum_GAN/captcha_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# params
CAPTCHA_WIDTH = 800
CAPTCHA_HEIGHT = 100
CHAR_SIZE = 64  # Size of gend character
MIN_CHARS = 4
MAX_CHARS = 6
NUM_SAMPLES = 10  # Number of CAPTCHAs to generate for viewing

# Noise params
ADD_LINES = True
MIN_LINES = 0
MAX_LINES = 3
LINE_WIDTH_RANGE = (1, 3)

# Character params
ROTATION_RANGE = (-10, 10)  
VERTICAL_OFFSET_RANGE = (-15, 15)  
CHAR_SPACING_RANGE = (50, 70)  # pixels between chars
EDGE_CROP = 6  # to remove edge artifact

# Char colors
COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 0, 255),    # Magenta
    (255, 165, 0),    # Orange
    (128, 0, 128),    # Purple
    (0, 128, 128),    # Teal
    (255, 20, 147),   # Deep Pink
    (0, 100, 0),      # Dark Green
    (139, 69, 19),    # Brown
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# -------------------------
# Alphanumeric character set - MUST MATCH TRAINING ORDER
# -------------------------
# Reconstruct the exact same mapping as during training
# The training code filters from all available folders in sorted order

# First, we need to get the actual characters from the dataset folders
import glob as glob_module

def build_training_mapping(train_dir):
    """Rebuild the exact mapping used during training"""
    # Get foldres
    label_folders = sorted([p for p in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, p))])
    ascii_vals = [int(x) for x in label_folders]
    all_chars = [chr(a) for a in ascii_vals]

    alphanumeric_set = []
    alphanumeric_set.extend([str(i) for i in range(10)])  # 0-9
    alphanumeric_set.extend([chr(i) for i in range(65, 91)])  # A-Z
    alphanumeric_set.extend([chr(i) for i in range(97, 123)])  # a-z

    chars = [ch for ch in all_chars if ch in alphanumeric_set]

    mapping = {chars[i]: i for i in range(len(chars))}
    inv_mapping = {i: chars[i] for i in range(len(chars))}

    return chars, mapping, inv_mapping

# rebuild same mapping
TRAIN_DIR_FOR_MAPPING = TRAIN_DIR if 'TRAIN_DIR' in globals() else os.path.join(DATA_DIR if 'DATA_DIR' in globals() else kagglehub.dataset_download("lopalp/alphanum"), "train")

chars, mapping, inv_mapping = build_training_mapping(TRAIN_DIR_FOR_MAPPING)
NUM_CLASSES = len(chars)
ALPHANUMERIC_CHARS = chars  # Use actual order from training

print(f"Reconstructed training mapping:")
print(f"  Total classes: {NUM_CLASSES}")
print(f"  Characters: {''.join(chars)}")
print(f"  Sample mapping: {list(mapping.items())[:10]}")

# -------------------------
# Generator Architecture (SAME as training)
# -------------------------
Z_DIM = 100
EMBED_DIM = 50
G_FEATURES = 64

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.query = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.size()

        query = self.query(x).view(B, -1, H * W).permute(0, 2, 1)
        key = self.key(x).view(B, -1, H * W)
        attention = F.softmax(torch.bmm(query, key), dim=-1)

        value = self.value(x).view(B, -1, H * W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)

        return self.gamma * out + x

class Generator(nn.Module):
    def __init__(self, z_dim=Z_DIM, embed_dim=EMBED_DIM, num_classes=NUM_CLASSES, features=G_FEATURES):
        super().__init__()

        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        self.z_dim = z_dim
        self.embed_dim = embed_dim

        self.project = nn.Sequential(
            nn.Linear(z_dim + embed_dim, features * 8 * 4 * 4),
            nn.BatchNorm1d(features * 8 * 4 * 4),
            nn.ReLU(True)
        )

        self.conv1 = self._conv_block(features * 8, features * 4, upsample=True)
        self.conv2 = self._conv_block(features * 4, features * 2, upsample=True)
        self.attention = SelfAttention(features * 2)
        self.conv3 = self._conv_block(features * 2, features, upsample=True)
        self.conv4 = self._conv_block(features, features, upsample=True)

        self.final = nn.Sequential(
            nn.Conv2d(features, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def _conv_block(self, in_channels, out_channels, upsample=False):
        layers = []
        if upsample:
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
        layers.extend([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        ])
        return nn.Sequential(*layers)

    def forward(self, z, labels):
        label_embed = self.label_embedding(labels)
        x = torch.cat([z, label_embed], dim=1)

        x = self.project(x)
        x = x.view(x.size(0), -1, 4, 4)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.final(x)

        return x

# -------------------------
# Load Generator
# -------------------------
def load_generator(checkpoint_path):
    generator = Generator().to(DEVICE)
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    generator.eval()
    print(f"Loaded generator from epoch {checkpoint['epoch']}")
    return generator

# Find latest ckpt
checkpoints = sorted([f for f in os.listdir(CKPT_DIR) if f.endswith('.pth')])
if not checkpoints:
    raise FileNotFoundError(f"No checkpoints found in {CKPT_DIR}")

latest_checkpoint = os.path.join(CKPT_DIR, checkpoints[-1])
print(f"Loading checkpoint: {latest_checkpoint}")
generator = load_generator(latest_checkpoint)

# -------------------------
# Character Generation & Processing
# -------------------------
def generate_character(generator, char, device=DEVICE):
    if char not in mapping:
        raise ValueError(f"Character '{char}' not in training set")

    label = mapping[char]

    with torch.no_grad():
        z = torch.randn(1, Z_DIM).to(device)
        labels = torch.tensor([label]).to(device)
        fake_img = generator(z, labels)

        # Denormalize from [-1, 1] to [0, 1]
        fake_img = (fake_img + 1) / 2.0
        fake_img = fake_img.clamp(0, 1)

        # Convert to numpy (H, W, C)
        img_np = fake_img.squeeze(0).cpu().numpy().transpose(1, 2, 0)

    return img_np

def clean_character_image(img_np, threshold=0.9, edge_crop=EDGE_CROP):
    # Convert to grayscale
    if len(img_np.shape) == 3:
        gray = np.mean(img_np, axis=2)
    else:
        gray = img_np

    # remove border artifact
    if edge_crop > 0:
        gray = gray[edge_crop:-edge_crop, edge_crop:-edge_crop]

    # pixels above threshold become white (background)
    # pixels below threshold become black (character)
    binary = (gray > threshold).astype(np.uint8) * 255

    binary_img = Image.fromarray(binary.astype('uint8'), 'L')

    binary_img = binary_img.filter(ImageFilter.MedianFilter(size=3))

    return binary_img

def colorize_character(char_img, color):
    char_img = char_img.convert('RGBA')

    # Create colored version
    colored = Image.new('RGBA', char_img.size, (255, 255, 255, 0))
    pixels = np.array(char_img)

    colored_array = np.ones((char_img.size[1], char_img.size[0], 4), dtype=np.uint8) * 255
    colored_array[:, :, 3] = 0  # Set alpha to 0

    mask = pixels[:, :, 0] < 128  # Use first channel for grayscale
    colored_array[mask] = list(color) + [255]  # Add alpha channel

    colored = Image.fromarray(colored_array, 'RGBA')

    return colored

def add_noise_lines(img, num_lines=2):
    draw = ImageDraw.Draw(img)
    width, height = img.size

    for _ in range(num_lines):
        x1 = random.randint(0, width)
        y1 = random.randint(0, height)
        x2 = random.randint(0, width)
        y2 = random.randint(0, height)
        line_width = random.randint(*LINE_WIDTH_RANGE)

        draw.line([(x1, y1), (x2, y2)], fill=(0, 0, 0), width=line_width)

    return img

# -------------------------
# CAPTCHA Generation
# -------------------------
def generate_captcha(generator, length=None):
    if length is None:
        length = random.randint(MIN_CHARS, MAX_CHARS)

    # Generate random string
    captcha_text = ''.join(random.choices(ALPHANUMERIC_CHARS, k=length))

    # Create white background
    captcha_img = Image.new('RGB', (CAPTCHA_WIDTH, CAPTCHA_HEIGHT), (255, 255, 255))

    # Calculate starting position to center the text
    total_spacing = CHAR_SPACING_RANGE[0] * (length - 1)
    start_x = (CAPTCHA_WIDTH - total_spacing - CHAR_SIZE) // 2

    # Generate and place each character
    current_x = start_x

    for char in captcha_text:
        # Generate character
        char_np = generate_character(generator, char)

        # Clean and binarize
        char_img = clean_character_image(char_np, threshold=0.65)

        # color
        color = random.choice(COLORS)
        char_colored = colorize_character(char_img, color)

        # rotation
        rotation = random.uniform(*ROTATION_RANGE)
        char_rotated = char_colored.rotate(rotation, expand=True, fillcolor=(255, 255, 255, 0))

        # offset
        y_offset = random.randint(*VERTICAL_OFFSET_RANGE)
        y_pos = (CAPTCHA_HEIGHT - char_rotated.size[1]) // 2 + y_offset

        # Paste onto captcha 
        captcha_img.paste(char_rotated, (current_x, y_pos), char_rotated)

        # Move to next character position
        current_x += random.randint(*CHAR_SPACING_RANGE)

    # Add noise lines
    if ADD_LINES:
        num_lines = random.randint(MIN_LINES, MAX_LINES)
        captcha_img = add_noise_lines(captcha_img, num_lines)

    return captcha_img, captcha_text

# -------------------------
# Generate Multiple CAPTCHAs
# -------------------------
print(f"\nGenerating {NUM_SAMPLES} CAPTCHA images...")
print("=" * 80)

for i in range(NUM_SAMPLES):
    try:
        captcha_img, captcha_text = generate_captcha(generator)
        
        filename = f"captcha_{i+1:04d}_{captcha_text}.png"
        filepath = os.path.join(OUTPUT_DIR, filename)
        captcha_img.save(filepath)

        if (i + 1) % 10 == 0:
            print(f"Generated {i+1}/{NUM_SAMPLES} CAPTCHAs")

    except Exception as e:
        print(f"Error generating CAPTCHA {i+1}: {e}")
        continue

print("=" * 80)
print(f"✓ Successfully generated {NUM_SAMPLES} CAPTCHA images")
print(f"✓ Saved to: {OUTPUT_DIR}")

# -------------------------
# Display Sample CAPTCHAs
# -------------------------
print("\nDisplaying sample CAPTCHAs...")

# Load and display first 9 generated images
sample_files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')])[:9]

fig, axes = plt.subplots(3, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, filename in enumerate(sample_files):
    img = Image.open(os.path.join(OUTPUT_DIR, filename))
    axes[idx].imshow(img)
    axes[idx].axis('off')
    # Extract text from filename
    text = filename.split('_')[2].replace('.png', '')
    axes[idx].set_title(f"Text: {text}", fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'sample_captchas.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f"\n✓ Sample grid saved to: {os.path.join(OUTPUT_DIR, 'sample_captchas.png')}")