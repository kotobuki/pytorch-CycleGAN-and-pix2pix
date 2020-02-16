import numpy as np
import cv2
from torchvision.transforms import ToTensor
from options.test_options import TestOptions
from models import create_model
from util.functions import abstract_image, abstract_image_array
from util.util import tensor2im
from util.run_cyclegan import RunCycleGAN


class RunPix2Pix(RunCycleGAN):
    """Pix2pix wrapper class for runtime inference.

    Example:
        PROJECT = 'YOUR FOLDER NAME UNDER checkpoint'
        SIZE = 'YOUR DATA PX SIZE'
        GPU = '-1' # if you use CPU, else 0 or whatever.
        K = 5      # your configuration.
        options = (f'--dataroot dummy --direction BtoA --model pix2pix --name {PROJECT} ' +
                   f'--load_size {SIZE} --crop_size {SIZE} --gpu_ids {GPU}')
        pix2pix = RunPix2Pix(options)
          :
        _, abst_img = abstract_image_array(image_array_RGB_HWC, K)
        converted = pix2pix.convert(abst_img)
    """
    
    def __init__(self, options, K=5):
        super().__init__(options)
        self.K = K

    def preprocess_file(self, file_name):
        org_img, abst_img = abstract_image(file_name, K=self.K, add_edge=True,
            resize=(self.opt.crop_size, self.opt.crop_size))
        return org_img, abst_img

    def preprocess(self, image):
        abst_img = abstract_image_array(image, K=self.K, add_edge=True)
        return abst_img

    def convert(self, abst_img):
        AorB = self.normalize(self.totensor(abst_img)).unsqueeze(0)
        data = {
            'A': AorB, 'A_paths': 'dummy', 
            'B': AorB, 'B_paths': 'dummy',
        }
        self.model.set_input(data)  # unpack data from data loader
        self.model.test()           # run inference
        visuals = self.model.get_current_visuals()  # get image results
        return tensor2im(visuals['fake_B'])

    def convert_file(self, file_name, out_file_name=None):
        org_img, abst_img = self.preprocess_file(file_name)
        converted = self.convert(abst_img)
        if out_file_name is not None:
            bgr = cv2.cvtColor(converted, cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_file_name, bgr)
        return org_img, converted
