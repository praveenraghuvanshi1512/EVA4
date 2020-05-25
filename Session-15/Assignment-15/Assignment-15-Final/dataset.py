import os 
from PIL import Image
from pathlib import Path,PurePath
from zipfile import ZipFile 
from torch.utils.data import Dataset
from torchvision.transforms import transforms


class ImageDataset(Dataset):
  def __init__(self, zipfile, transformations):
    self.zipobj = ZipFile(zipfile)
    self.zipfiles = self.zipobj.namelist()

    self.bg_fg_files = []
    self.masks_files = []
    self.depths_files = []
    
    for item in self.zipfiles:
        if item.startswith('bg_fg_1/'):
              self.bg_fg_files.append(item)
        if item.startswith('bg_fg_mask_1/'):
              self.masks_files.append(item)
        if item.startswith('depthMap/'):
              self.depths_files.append(item)
    
    self.transformations = transformations
    self.simpletransformation = transforms.Compose([transforms.ToTensor(),])
    
    print(len(self.zipfiles))
    self.zipfiles.clear()
    
  def __len__(self):
    return len(self.bg_fg_files)

  def __getitem__(self, index):
    bg_fg_img = self.transformations(Image.open(self.zipobj.open(self.bg_fg_files[index])))
    mask_img = self.simpletransformation(Image.open(self.zipobj.open(self.masks_files[index])))
    depth_img = self.simpletransformation(Image.open(self.zipobj.open(self.depths_files[index])))

    return {'bg_fg': bg_fg_img, 'mask': mask_img, 'depth': depth_img}