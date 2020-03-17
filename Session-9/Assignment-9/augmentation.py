import numpy as np
import albumentations as A
import albumentations.pytorch.transforms as T
import cv2 as cv2

train = A.Compose([
    A.HorizontalFlip(p=1),
    T.ToTensor(),
    A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

test = A.Compose([
    T.ToTensor(),
    A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

class AlbumentationTransformTrain:
    def __init__(self):
        
        self.transform = A.Compose([
            A.HorizontalFlip(p=1),
            A.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            A.Cutout(num_holes=1, max_h_size=12, max_w_size=12, fill_value=[0.4914, 0.4822, 0.4465], always_apply=False, p=0.5),
            A.RandomContrast(limit=0.2),
            A.RandomBrightness(limit=0.2),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1,rotate_limit=15, border_mode=cv2.BORDER_REFLECT_101),
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

