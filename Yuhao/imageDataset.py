import os
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
  def __init__(self, image_dir, transform=None):
    self.image_dir = image_dir
    self.transform = transform
    # accept png or jpg files
    self.image_files, self.labels = cleanup(image_dir)

  def __len__(self):
    return len(self.image_files)
  
  def __getitem__(self, idx):
    img_path = self.image_files[idx]
    image = Image.open(img_path).convert('L')  # Convert to grayscale
    label = self.labels[idx]
    
    if self.transform:
      image = self.transform(image)
    
    return image, label


def cleanup(image_dir):
  image_files = []
  labels = []
  for filename in os.listdir(image_dir):
    if filename.endswith('.png') or filename.endswith('.jpg'):
      # all files are expected to be named with ONE letter and a number (e.g. a1, z69420)
      label = filename[0]  # Get the first character as label
      image_files.append(os.path.join(image_dir, filename))
      labels.append(label)
  return image_files, labels