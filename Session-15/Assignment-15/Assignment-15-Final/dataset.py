from torch.utils.data import Dataset
from torchvision.transforms import transforms
from PIL import Image

class ImageDataset(Dataset):
  def __init__(self, root, transformations):
    self.bg_fg_files = list(bg_fg.glob('*.jpg'))
    self.masks_files = list(masks.glob('*.jpg'))
    self.depths_files = list(depths.glob('*.jpg'))
    self.transformations = transformations
    
  def __len__(self):
    return len(self.bg_fg_files)

  def __getitem__(self, index):
    bg_fg_img = self.transformations(Image.open(self.bg_fg_files[index]))
    mask_img = self.transformations(Image.open(self.masks_files[index]))
    depth_img = self.transformations(Image.open(self.depths_files[index]))

    return {'bg_fg': bg_fg_img, 'mask': mask_img, 'depth': depth_img}