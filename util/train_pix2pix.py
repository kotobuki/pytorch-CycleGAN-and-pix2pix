import numpy as np
from dlcliche.utils import Path, ensure_delete, ensure_folder
from PIL import Image


class TrainPix2Pix(object):
    def __init__(self, project_name, train_files, load_size, crop_size, suffix, model='pix2pix',
                 pre_crop_rect=None, extra='', skip_file_creation=False):
        self.project, self.load_size, self.crop_size = project_name, load_size, crop_size
        self.suffix, self.model, self.pre_crop_rect, self.extra = suffix, model, pre_crop_rect, extra
        self.root = Path(f'./datasets/{self.project}')
        self.train = self.root/'train'
        self.test = self.root/'test'
        self.prepare_train(train_files, skip_file_creation)

    def copy_files_pre_crop(self, dest, files, skip_file_creation):
        new_filenames = []
        for f in files:
            file_name = (dest/f.name).with_suffix(self.suffix)
            if not skip_file_creation:
                img = Image.open(f).convert('RGB')
                if self.pre_crop_rect is not None:
                    img = img.crop(self.pre_crop_rect)
                img = img.resize((self.load_size, self.load_size))
                img.save(file_name)
            new_filenames.append(file_name)
        return new_filenames

    def prepare_train(self, train_files, skip_file_creation):
        if not skip_file_creation:
            ensure_delete(self.train)
            ensure_folder(self.train)
        self.train_files = self.copy_files_pre_crop(self.train, train_files, skip_file_creation)

    def prepare_test(self, test_files, skip_file_creation=False):
        if not skip_file_creation:
            ensure_delete(self.test)
            ensure_folder(self.test)
        self.test_files = self.copy_files_pre_crop(self.test, test_files, skip_file_creation=skip_file_creation)

    def train_command_line(self, n_epochs=100):
        return (f'python train.py --dataroot {self.root} --direction AtoB --model {self.model} --n_epochs {n_epochs - 100} '
                f'--name {self.project} --display_id 0 '
                f'--load_size {self.load_size} --crop_size {self.crop_size} {self.extra}'
               )
