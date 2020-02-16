import cv2
from torchvision.transforms import ToTensor
from options.test_options import TestOptions
from models import create_model
from util.util import tensor2im


class RunCycleGAN:
    """CycleGAN wrapper class for runtime inference.

    Example:
        PROJECT = 'YOUR FOLDER NAME UNDER checkpoint'
        SIZE = 'YOUR DATA PX SIZE'
        GPU = '-1' # if you use CPU, else 0 or whatever.
        options = f'--dataroot dummy --name {PROJECT} --model test --no_dropout  --load_size {SIZE} --crop_size {SIZE} --gpu_ids {GPU}'
        cyclegan = RunCycleGAN(options)
          :
        converted = cyclegan.convert(image_array_RGB_HWC)
    """

    def __init__(self, options):
        opt = TestOptions(options).parse()  # get test options
        # hard-code some parameters for test
        opt.num_threads = 0   # test code only supports num_threads = 1
        opt.batch_size = 1    # test code only supports batch_size = 1
        opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
        opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
        model = create_model(opt)      # create a model given opt.model and other options
        model.setup(opt)               # regular setup: load and print networks; create schedulers
        model.eval()
        self.opt, self.model = opt, model
        self.totensor = ToTensor()

    def convert(self, img):
        data = {'A': self.totensor(img).unsqueeze(0), 'A_paths': 'dummy'}
        self.model.set_input(data)  # unpack data from data loader
        self.model.test()           # run inference
        visuals = self.model.get_current_visuals()  # get image results
        return tensor2im(visuals['fake'])

    def convert_file(self, file_name, out_file_name=None):
        img = cv2.imread(str(file_name))
        assert img is not None, f'Cannot read {file_name}'
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.opt.crop_size, self.opt.crop_size))
        converted = self.convert(img)
        if out_file_name is not None:
            bgr = cv2.cvtColor(converted, cv2.COLOR_RGB2BGR)
            cv2.imwrite(out_file_name, bgr)
        return img, converted

