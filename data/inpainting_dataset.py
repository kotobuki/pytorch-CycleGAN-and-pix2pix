import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, AugStrong
from data.image_folder import make_dataset
from PIL import Image, ImageDraw, ImageOps
import random
import numpy as np


def flat_top_hill(size, zero, r, m):
    (w, h), (x0, y0), (rx, ry) = size, zero, r
    def tantan(a, a0, r0):
        return (np.tanh((a - a0)/m) - np.tanh((a - a0 - r0)/m)) / 2
    return np.matmul(tantan(np.linspace(0, h, h), y0, ry)[:, np.newaxis],
                     tantan(np.linspace(0, w, w), x0, rx)[np.newaxis, :])


def rand_flat_top_hill(size, pcts=(0.1, 0.5), m=None):
    def rand_r(r, pcts):
        return random.randint(int(r*pcts[0]), int(r*pcts[1]))
    W, H = size
    w, h = rand_r(W, pcts), rand_r(H, pcts)
    x0, y0 = rand_r(W - w, (0., 1.)), rand_r(H - h, (0., 1.))
    m = W//20 if m is None else m
    pl = flat_top_hill(size, (x0, y0), (w, h), m)
    return pl


def rand_smooth_cutout(np_img, pcts=(0.05, 0.4), depth=(1., 0.), max_k=1, m=None):
    """Cut out from image, and edges of rectangles are smooth.

    Returns:
        applied image, cut mask
    """
    cut = np.ones(np_img.shape[:2])
    k = random.randint(1, max_k)
    for _ in range(k):
        d = random.random() * depth[0] + depth[1]
        hill = rand_flat_top_hill((np_img.shape[1], np_img.shape[0]), pcts=pcts, m=m)
        cut = cut * (1 -  d * hill)
    return np_img * cut[..., np.newaxis], (cut < 0.9).astype(np.int8)


def rand_solid_hill(size, pcts=(0.1, 0.5)):
    def rand_r(r, pcts):
        return random.randint(int(r*pcts[0]), int(r*pcts[1]))
    W, H = size
    w, h = rand_r(W, pcts), rand_r(H, pcts)
    x0, y0 = rand_r(W - w, (0., 1.)), rand_r(H - h, (0., 1.))
    hill = np.zeros((H, W))
    hill[y0:y0+h, x0:x0+w] = 1.0
    return hill


def rand_cutout(np_img, pcts=(0.05, 0.4), depth=(1., 0.), max_k=1):
    """Cut out from image, and edges of rectangles are smooth.

    Returns:
        applied image, cut mask
    """
    cut = np.ones(np_img.shape[:2])
    k = random.randint(1, max_k)
    for _ in range(k):
        d = random.random() * depth[0] + depth[1]
        hill = rand_solid_hill((np_img.shape[1], np_img.shape[0]), pcts=pcts)
        cut = cut * (1 -  d * hill)
    return np_img * cut[..., np.newaxis], (cut < 0.9).astype(np.int8)


class InpaintingDataset(BaseDataset):
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

        A = self.cutout(AB)
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

    def cutout(self, A):
        """Default anomaly twin maker."""
        A = np.array(A)/255.
        if self.opt.co_m > 0:
            A, _ = rand_smooth_cutout(A, pcts=(self.opt.co_pct0, self.opt.co_pct1),
                                      depth=(self.opt.co_depth_a, self.opt.co_depth_b),
                                      max_k=self.opt.co_k, m=self.opt.co_m)
        else:
            A, _ = rand_cutout(A, pcts=(self.opt.co_pct0, self.opt.co_pct1),
                               depth=(self.opt.co_depth_a, self.opt.co_depth_b),
                               max_k=self.opt.co_k)
        A = Image.fromarray((A * 255).astype(np.uint8))
        return A
