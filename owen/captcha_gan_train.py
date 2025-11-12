from google.colab import drive
drive.mount('/content/drive', force_remount=False)

import os
import glob
import random
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import kagglehub

# -------------------------
# CONFIG
# -------------------------
# CHANGE BASED ON COLAB/LOCAL
DATA_DIR = kagglehub.dataset_download("lopalp/alphanum")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
OUT_DIR = "/content/drive/MyDrive/Alphanum_GAN/output"

os.makedirs(OUT_DIR, exist_ok=True)
SAMPLES_DIR = os.path.join(OUT_DIR, "samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)
CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)

# params
IMG_SIZE = 64
IN_SIZE = 24
BATCH_SIZE = 64  
Z_DIM = 100  
EMBED_DIM = 50  
G_FEATURES = 64
D_FEATURES = 64  

LR_D = 4e-4
LR_G = 1e-4
BETA1 = 0.0  
BETA2 = 0.9

EPOCHS = 150
SAVE_EVERY = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PRINT_EVERY = 50

# Label smoothing for better training
REAL_LABEL_SMOOTH = 0.9
FAKE_LABEL = 0.0

print(f"Using device: {DEVICE}")

# -------------------------
# Need to rebuild mapping from dataset
# -------------------------
ALPHANUMERIC_CHARS = []
ALPHANUMERIC_CHARS.extend([str(i) for i in range(10)])  # 0-9
ALPHANUMERIC_CHARS.extend([chr(i) for i in range(65, 91)])  # A-Z
ALPHANUMERIC_CHARS.extend([chr(i) for i in range(97, 123)])  # a-z

print(f"Target alphanumeric characters ({len(ALPHANUMERIC_CHARS)}): {''.join(ALPHANUMERIC_CHARS)}")

# Get all available folders
label_folders = sorted([p for p in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, p))])
ascii_vals = [int(x) for x in label_folders]
all_chars = [chr(a) for a in ascii_vals]

# Filter until only alphanumeric characters
chars = [ch for ch in all_chars if ch in ALPHANUMERIC_CHARS]

# Create mappings 
mapping = {chars[i]: i for i in range(len(chars))}
inv_mapping = {i: chars[i] for i in range(len(chars))}
NUM_CLASSES = len(chars)

print(f"Found {NUM_CLASSES} alphanumeric classes in dataset: {''.join(sorted(chars))}")
print(f"Missing characters: {set(ALPHANUMERIC_CHARS) - set(chars)}")

# -------------------------
# Dataset & DataLoader
# -------------------------
class AlphanumDataset(Dataset):
    def __init__(self, root_dir, mapping, img_size=IMG_SIZE, train=True):
        self.samples = []
        self.mapping = mapping

        for ch, idx in mapping.items():
            ascii_folder = str(ord(ch))
            folder = os.path.join(root_dir, ascii_folder)
            if not os.path.isdir(folder):
                continue
            for f in glob.glob(os.path.join(folder, "*.png")) + glob.glob(os.path.join(folder, "*.jpg")):
                self.samples.append((f, idx))

        # DATA TRANSFORMS FOR TRAINING
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size), interpolation=Image.BICUBIC),
            transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L")
        img = self.transform(img)
        img = img.repeat(3, 1, 1)
        return img, label

train_dataset = AlphanumDataset(TRAIN_DIR, mapping, img_size=IMG_SIZE)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True, drop_last=True)
print(f"Train dataset size: {len(train_dataset)}")

# -------------------------
# Generator with Self-Attention
# -------------------------
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

        # Project and reshape: (z_dim + embed_dim) -> features*8 x 4 x 4
        self.project = nn.Sequential(
            nn.Linear(z_dim + embed_dim, features * 8 * 4 * 4),
            nn.BatchNorm1d(features * 8 * 4 * 4),
            nn.ReLU(True)
        )

        # Upsampling blocks: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
        self.conv1 = self._conv_block(features * 8, features * 4, upsample=True)  # 8x8
        self.conv2 = self._conv_block(features * 4, features * 2, upsample=True)  # 16x16
        self.attention = SelfAttention(features * 2)  # Self-attention at 16x16
        self.conv3 = self._conv_block(features * 2, features, upsample=True)      # 32x32
        self.conv4 = self._conv_block(features, features, upsample=True)          # 64x64

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
        # Embed labels and concat with noise
        label_embed = self.label_embedding(labels)
        x = torch.cat([z, label_embed], dim=1)

        # Project and reshape
        x = self.project(x)
        x = x.view(x.size(0), -1, 4, 4)

        # Generate image
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.final(x)

        return x

# -------------------------
# Discriminator with Spectral Normalization
# -------------------------
def spectral_norm(module):
    return nn.utils.spectral_norm(module)

class Discriminator(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, features=D_FEATURES, embed_dim=EMBED_DIM):
        super().__init__()

        self.label_embedding = nn.Embedding(num_classes, embed_dim)

        # Downsampling: 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.conv1 = self._conv_block(3, features, spectral=True)           # 32x32
        self.conv2 = self._conv_block(features, features * 2, spectral=True)  # 16x16
        self.attention = SelfAttention(features * 2)
        self.conv3 = self._conv_block(features * 2, features * 4, spectral=True)  # 8x8
        self.conv4 = self._conv_block(features * 4, features * 8, spectral=True)  # 4x4

        # Final classification
        self.final = nn.Sequential(
            spectral_norm(nn.Conv2d(features * 8, features * 8, kernel_size=4, stride=1)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Output heads
        self.adv_layer = spectral_norm(nn.Linear(features * 8, 1))  # Real/Fake
        self.aux_layer = spectral_norm(nn.Linear(features * 8, num_classes))  # Class prediction

    def _conv_block(self, in_channels, out_channels, spectral=False):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)
        if spectral:
            conv = spectral_norm(conv)
        return nn.Sequential(
            conv,
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3)
        )

    def forward(self, img, labels=None):
        x = self.conv1(img)
        x = self.conv2(x)
        x = self.attention(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.final(x)

        x = x.view(x.size(0), -1)

        # Real/Fake prediction
        validity = self.adv_layer(x)

        # Class prediction (auxiliary classifier)
        class_pred = self.aux_layer(x)

        return validity, class_pred

# -------------------------
# Initialize models
# -------------------------
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

generator = Generator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)

generator.apply(weights_init)
discriminator.apply(weights_init)

print(f"Generator parameters: {sum(p.numel() for p in generator.parameters()):,}")
print(f"Discriminator parameters: {sum(p.numel() for p in discriminator.parameters()):,}")

# -------------------------
# Optimizers
# -------------------------
optimizer_G = torch.optim.Adam(generator.parameters(), lr=LR_G, betas=(BETA1, BETA2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR_D, betas=(BETA1, BETA2))

# Learning rate schedulers
scheduler_G = torch.optim.lr_scheduler.StepLR(optimizer_G, step_size=50, gamma=0.5)
scheduler_D = torch.optim.lr_scheduler.StepLR(optimizer_D, step_size=50, gamma=0.5)

# Loss functions
adversarial_loss = nn.BCEWithLogitsLoss()
auxiliary_loss = nn.CrossEntropyLoss()

# -------------------------
# Training utilities
# -------------------------
def save_samples(generator, epoch, num_samples=NUM_CLASSES):
    generator.eval()
    with torch.no_grad():
        # Generate one sample per class
        z = torch.randn(num_samples, Z_DIM).to(DEVICE)
        labels = torch.arange(num_samples).to(DEVICE)
        fake_imgs = generator(z, labels)

        # Denormalize
        fake_imgs = (fake_imgs + 1) / 2.0

        # Create grid
        grid = utils.make_grid(fake_imgs, nrow=10, padding=2, normalize=False)

        # Save
        plt.figure(figsize=(15, 8))
        plt.imshow(grid.cpu().permute(1, 2, 0))
        plt.axis('off')
        plt.title(f'Generated Characters - Epoch {epoch}')
        plt.tight_layout()
        plt.savefig(os.path.join(SAMPLES_DIR, f'epoch_{epoch:03d}.png'), dpi=150, bbox_inches='tight')
        plt.close()
    generator.train()

# -------------------------
# Training Loop
# -------------------------
print("\nStarting training...")
print("=" * 80)

# Reusme from latest ckpt if found
latest_ckpt = None
if os.path.exists(CKPT_DIR):
    ckpt_files = sorted(glob.glob(os.path.join(CKPT_DIR, "checkpoint_epoch_*.pth")))
    if ckpt_files:
        latest_ckpt = ckpt_files[-1]

start_epoch = 1
g_losses, d_losses = [], []

if latest_ckpt:
    print(f"Found checkpoint: {latest_ckpt}")
    checkpoint = torch.load(latest_ckpt, map_location=DEVICE)

    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])

    if 'g_losses' in checkpoint:
        g_losses = checkpoint['g_losses']
    if 'd_losses' in checkpoint:
        d_losses = checkpoint['d_losses']

    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming training from epoch {start_epoch}...")
else:
    print("No checkpoint found. Starting training from nothing.")

for epoch in range(start_epoch, EPOCHS + 1):
    epoch_g_loss = 0
    epoch_d_loss = 0

    for i, (real_imgs, labels) in enumerate(train_loader):
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        # Labels for adversarial loss
        real_labels = torch.full((batch_size, 1), REAL_LABEL_SMOOTH, device=DEVICE)
        fake_labels = torch.full((batch_size, 1), FAKE_LABEL, device=DEVICE)

        # ---------------------
        # Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Real images
        real_validity, real_class = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(real_validity, real_labels)
        d_real_aux_loss = auxiliary_loss(real_class, labels)

        # Fake images
        z = torch.randn(batch_size, Z_DIM).to(DEVICE)
        gen_labels = torch.randint(0, NUM_CLASSES, (batch_size,)).to(DEVICE)
        fake_imgs = generator(z, gen_labels)
        fake_validity, fake_class = discriminator(fake_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(fake_validity, fake_labels)
        d_fake_aux_loss = auxiliary_loss(fake_class, gen_labels)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2 + (d_real_aux_loss + d_fake_aux_loss) / 2
        d_loss.backward()
        optimizer_D.step()

        # ---------------------
        # Train Generator
        # ---------------------
        optimizer_G.zero_grad()

        # Generate fake images
        z = torch.randn(batch_size, Z_DIM).to(DEVICE)
        gen_labels = torch.randint(0, NUM_CLASSES, (batch_size,)).to(DEVICE)
        fake_imgs = generator(z, gen_labels)

        # Generator wants discriminator to think fake images are real
        fake_validity, fake_class = discriminator(fake_imgs, gen_labels)
        g_adv_loss = adversarial_loss(fake_validity, real_labels)  # Classification as real/fake
        g_aux_loss = auxiliary_loss(fake_class, gen_labels)  # Prob of each class

        g_loss = g_adv_loss + g_aux_loss
        g_loss.backward()
        optimizer_G.step()

        # Statistics
        epoch_g_loss += g_loss.item()
        epoch_d_loss += d_loss.item()

        # Print progress
        if (i + 1) % PRINT_EVERY == 0:
            print(f"Epoch [{epoch}/{EPOCHS}] Batch [{i+1}/{len(train_loader)}] "
                  f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")

    # Epoch statistics
    avg_g_loss = epoch_g_loss / len(train_loader)
    avg_d_loss = epoch_d_loss / len(train_loader)
    g_losses.append(avg_g_loss)
    d_losses.append(avg_d_loss)

    print(f"Epoch [{epoch}/{EPOCHS}] Average - D_loss: {avg_d_loss:.4f}, G_loss: {avg_g_loss:.4f}")

    # Step schedulers
    scheduler_G.step()
    scheduler_D.step()

    save_samples(generator, epoch)

    if epoch % SAVE_EVERY == 0 or epoch == 1:
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'optimizer_G_state_dict': optimizer_G.state_dict(),
            'optimizer_D_state_dict': optimizer_D.state_dict(),
            'g_losses': g_losses,
            'd_losses': d_losses,
        }, os.path.join(CKPT_DIR, f'checkpoint_epoch_{epoch:03d}.pth'))

        print(f"Saved checkpoint and samples for epoch {epoch}")

print("\n" + "=" * 80)
print("Training complete")
