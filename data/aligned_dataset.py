import os.path
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)
import numpy as np
import random
from itertools import combinations


def _not_used_selective_strong_aug(p=0.7, K=2):
    """
    Why select augmentations by K?:
        Based on a paper to find better augmentation efficiently.
        'RandAugment: Practical automated data augmentation with a reduced search space'
        Ekin D. Cubuk, Barret Zoph, Jonathon Shlens, Quoc V. Le (2019)
    """
    augs = [
        RandomRotate90(),
        # Flip(), already there
        Transpose(),
        OneOf([
            MotionBlur(p=0.3),
            #MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=10, p=0.3),
        ], p=0.3),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.2),
        ], p=0.3),
        OneOf([
            #CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=.3),
        #HueSaturationValue(p=0.3),
    ]
    if K is None:
        selected = augs
    else:
        all_possibles = list(reversed(list(combinations(augs, K))))
        selected = random.choice(all_possibles)
    print(f'[{K} Augmentations]')
    for a in selected:
        print(a)
    return Compose(selected, p=p)


def strong_aug(p=0.7):
    augs = [
        OneOf([
            MotionBlur(p=0.3),
            #MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=10, p=0.3),
        ], p=0.3),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            #GridDistortion(p=0.1), # causes segmentation fault, due to version mismatch with open cv...
            IAAPiecewiseAffine(p=0.2),
        ], p=0.3),
    ]
    return Compose(augs, p=p)


class AugStrong:
    instance = None
    @staticmethod
    def augment_strong(p=0.7):
        if AugStrong.instance is None:
            AugStrong.instance = strong_aug(p=p)
        return AugStrong.instance


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc
        
        self.strong_aug = AugStrong.augment_strong(p=0.7)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # strong augmentation is also applied if flip is enabled.
        if not self.opt.no_flip and not self.opt.no_strong_aug:
            data = {"image": np.array(A), "mask": np.array(B)}
            augmented = self.strong_aug(**data)
            A, B = Image.fromarray(augmented["image"]), Image.fromarray(augmented["mask"])

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
