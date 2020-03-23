import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, AugStrong
from data.image_folder import make_dataset
from PIL import Image, ImageDraw, ImageOps
import random
import numpy as np

def plateau(size, zero, r, m):
    (w, h), (x0, y0), (rx, ry) = size, zero, r
    def tantan(a, a0, r0):
        return (np.tanh((a - a0)/m) - np.tanh((a - a0 - r0)/m)) / 2
    return np.matmul(tantan(np.linspace(0, w, w), x0, rx)[:, np.newaxis],
                     tantan(np.linspace(0, h, h), y0, ry)[np.newaxis, :])

def rand_plateau(size, pcts=(0.1, 0.5), m=None):
    def rand_r(r, pcts):
        return random.randint(int(r*pcts[0]), int(r*pcts[1]))
    W, H = size
    w, h = rand_r(W, pcts), rand_r(H, pcts)
    x0, y0 = rand_r(W - w, (0., 1.)), rand_r(H - h, (0., 1.))
    m = W//20 if m is None else m
    pl = plateau(size, (x0, y0), (w, h), m)
    return pl

def rand_dent(np_img, pcts=(0.05, 0.4), scale=(0.3, 0.7), m=None):
    scale = random.random() * scale[0] + scale[1]
    pl = 1 -  scale * rand_plateau((np_img.shape[1], np_img.shape[0]), pcts=pcts, m=m)
    return (pl < 0.9).astype(np.int8), np_img * pl[..., np.newaxis]


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
        self.strong_aug = AugStrong.get_aug() if self.opt.aug_strong else None
        self.width_min, self.width_max, self.length_max, self.color_max = 2, 8, 8, 255 #1, 16, opt.load_size//5, 127
        self.n_dots, self.n_clusters = 50, 5

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
        A, B = self.anomaly_twin(AB, self.opt.invert)

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

    def random_pick_point(self, image: Image, base_pt, shape, scale=0.05):
        # randomly choose a point from entire image
        W, H = image.size
        if scale < 1.0:
            w, h = int(W * scale * shape[0]), int(H * scale * shape[1])
            x0, x1 = max(base_pt[0] - w // 2, 0), min(base_pt[0] + w // 2, W)
            y0, y1 = max(base_pt[1] - h // 2, 0), min(base_pt[1] + h // 2, H)
        else:
            w, h = W, H
            x0, x1 = 0, w
            y0, y1 = 0, h
        return random.randint(x0, x1), random.randint(y0, y1)

    def random_pick_points(self, image, base_pt, shape, scale=0.05):
        x, y = self.random_pick_point(image, base_pt, shape, scale)
        w, h = image.size
        # randomly choose other parameters
        w_half, h_half = w // 2, h // w
        dx, dy = random.randint(0, self.length_max), random.randint(0, self.length_max)
        x2, y2 = x + dx if x < w_half else x - dx, y + dy if y < h_half else y - dy
        return x, y, x2, y2

    def random_pick_color(self, invert):
        # randomly choose a color
        c = (random.randint(0, self.color_max+1),
            random.randint(0, self.color_max+1),
            random.randint(0, self.color_max+1))
        # return c
        return (c[0], c[0], c[0])

    def anomaly_twin(self, A, invert):
        """Default anomaly twin maker."""
        # randomly choose a point on object
        if invert:
            A = ImageOps.invert(A)
        A = np.array(A)/255.
        B = np.zeros(A.shape[:2], dtype=np.uint8)
        nc = random.randint(1, self.n_clusters)
        for _ in range(nc):
            if self.opt.test_fake or self.opt.phase != 'test':
                mask, A = rand_dent(A, pcts=(0.05, 0.10), scale=(0.4, 0.5), m=8)
                B = B | mask
        A = Image.fromarray((A * 255).astype(np.uint8))
        B = B[..., np.newaxis]*255
        B = Image.fromarray(np.c_[B, B, B].astype(np.uint8))
        return A, B
