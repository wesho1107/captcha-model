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

chars = ["a","b","c","d","e","f","g","h","i","j","k","l","m","n","o","p","q","r","s","t","u","v","w","x","y","z",
         "0","1","2","3","4","5","6","7","8","9"]
char_to_class = {c: i for i, c in enumerate(chars)}

class ImageDatasetFromTextFile(Dataset):
  def __init__(self, text_file, transform=None):
    self.image_files = []
    self.labels = []
    self.transform = transform
    
    with open(text_file, 'r') as f:
      for line in f:
        img_path = line.strip()
        if not img_path:
          continue
        alphanumeric_char = os.path.basename(img_path)[0].lower()  # The first char of the filename but not including the folders
        label = char_to_class[alphanumeric_char]
        self.image_files.append(img_path)
        self.labels.append(label)
  
  def __len__(self):
    return len(self.image_files)
  
  def __getitem__(self, idx):
    img_path = self.image_files[idx]
    image = Image.open(img_path).convert('L')  # Convert to grayscale
    label = self.labels[idx]
    
    if self.transform:
      image = self.transform(image)
    
    return image, label