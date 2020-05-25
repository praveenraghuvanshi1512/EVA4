import numpy as np
import albumentations as A
import albumentations.pytorch.transforms as T
import cv2 as cv2

class AlbumentationTransformTrain:
    def __init__(self):
        
        self.transform = A.Compose([
            A.Rotate((-30.0,30.0)),
            A.HorizontalFlip(p=0.5),
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

