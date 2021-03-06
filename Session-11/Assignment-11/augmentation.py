import numpy as np
import albumentations as A
import albumentations.pytorch.transforms as T
import cv2 as cv2

class AlbumentationTransformTrain:
    def __init__(self):
        
        self.transform = A.Compose([
            A.PadIfNeeded(36,36, p=1.),
            A.RandomCrop(32,32, p=1.),
            A.HorizontalFlip(p=1),
            A.Cutout(num_holes=2, max_h_size=8, max_w_size=8, fill_value=[0.4914, 0.4822, 0.4465], always_apply=False, p=0.5),
            A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            T.ToTensor()
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.transform(image=img)['image']
        return img

class AlbumentationTransformTest:
    def __init__(self):

        self.transform = A.Compose([
            A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            T.ToTensor()
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.transform(image=img)['image']
        return img

