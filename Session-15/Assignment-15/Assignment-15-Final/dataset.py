from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image
from pathlib import Path,PurePath
import os 

class ImageDataset(Dataset):
  def __init__(self, root, transformations):
    
    labelsPath = os.path.join(root,'labels.txt')
    with open(labelsPath) as f:
        images = f.readlines()

    self.bg_files = [] 
    self.bg_fg_files = []
    self.masks_files = []
    self.depths_files = []
    
    for img in images:
        img = img.strip()
        bgName = img.split('_fg')[0] + '.jpg'

        self.bg_files.append(os.path.join(root+'/bg/', bgName))
        self.bg_fg_files.append(os.path.join(root+'/bg_fg/',img))
        self.masks_files.append(os.path.join(root+'/masks/',img))
        self.depths_files.append(os.path.join(root+'/depths/',img))
    
    self.transformations = transformations
    self.simpletransformation = transforms.Compose([transforms.ToTensor(),])
    
    images.clear()
    
  def __len__(self):
    return len(self.bg_fg_files)

  def __getitem__(self, index):
    
    bg_img = self.transformations(Image.open(self.bg_files[index]))
    bg_fg_img = self.transformations(Image.open(self.bg_fg_files[index]))
    mask_img = self.simpletransformation(Image.open(self.masks_files[index]))
    depth_img = self.simpletransformation(Image.open(self.depths_files[index]))

    return {'bg': bg_img, 'bg_fg': bg_fg_img, 'mask': mask_img, 'depth': depth_img}