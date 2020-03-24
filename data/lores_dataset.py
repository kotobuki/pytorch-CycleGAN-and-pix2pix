import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, AugStrong
from data.image_folder import make_dataset
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import random
import numpy as np
from util.util import to_np


def lo_res_pil(img, blur=4, mosaic=0, posterize=0):
    if blur > 0:
        img = img.filter(ImageFilter.GaussianBlur(blur))
    if mosaic > 0:
        img = img.resize([xy // mosaic for xy in img.size]).resize(img.size)
    if 8 > posterize and posterize > 0:
        img = ImageOps.posterize(img, posterize).convert('L').convert('RGB')
    return img


def lo_res(np_img, blur=4, mosaic=0, posterize=0):
    img = lo_res_pil(Image.fromarray(np_img), blur=blur, mosaic=mosaic, posterize=posterize)
    return np.array(img)


class LoResDataset(BaseDataset):
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
        self.strong_aug = AugStrong.get_aug() if self.opt.aug_strong else None
        self.width_min, self.width_max, self.length_max, self.color_max = 2, 8, 8, 255 #1, 16, opt.load_size//5, 127


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

        A = lo_res_pil(AB, blur=self.opt.lo_blur, mosaic=self.opt.lo_mosaic,
                       posterize=self.opt.lo_posterize)
        B = AB

        # strong augmentation is also applied if flip is enabled.
        if self.strong_aug:
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
