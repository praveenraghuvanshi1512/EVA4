import numpy as np
import albumentations as A
import albumentations.pytorch.transforms as T
import cv2 as cv2

class AlbumentationTransformTrain:
    def __init__(self):
        
        self.transform = A.Compose([
            A.Rotate((-30.0,30.0)),
            A.ToGray(),
            A.RandomBrightnessContrast(),
            A.HorizontalFlip(p=0.5),
            A.RandomResizedCrop(64, 64, scale=(0.75, 1.0), ratio=(0.9, 1.1), p=0.75),
            #alb.Cutout(num_holes=4,max_h_size=16, max_w_size=16,fill_value=0.4421*255),
            A.CoarseDropout(max_holes=8, max_height=16, max_width=16, min_holes=1, min_height=8, min_width=8,fill_value=0, p=1.0),
            A.Normalize((0.4804, 0.4482, 0.3976), (0.277, 0.269, 0.282)),
            T.ToTensor(),
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.transform(image=img)['image']
        return img

class AlbumentationTransformTest:
    def __init__(self):

        self.transform = A.Compose([
            A.Normalize((0.4804, 0.4482, 0.3976), (0.227, 0.269, 0.282)),
            T.ToTensor()
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.transform(image=img)['image']
        return img

