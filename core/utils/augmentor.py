from core.utils.augmentations import *
import numpy as np


name_to_aug = {
    'resize': Resize,
    't_resize': TemporalResize,
    'random_crop': RandomCrop,
    't_random_crop': TemporalRandomCrop,
    't_center_crop': TemporalCenterCrop,
    't_to_tensor': TemporalToTensor,
    't_normalize': TemporalNormalize,
    't_color_jitter': TemporalColorJitter,
    # 'random_hrz_flip': RandomHorizontalFlip,
    # 't_random_hrz_flip': TemporalRandomHorizontalFlip,
    # 'color_jitter': transforms.ColorJitter,
    't_s_resize': TemporalSegResize,
    'to_tensor': ToTensor,
    'normalize': Normalize,
}


class Augmentor():
    def __init__(self, aug_params):
        self.aug_list = []
        
        for key, val in aug_params.items():
            if isinstance(val, int) or isinstance(val, float):
                self.aug_list.append(name_to_aug[key](val))
            elif isinstance(val, list):
                self.aug_list.append(name_to_aug[key](*val))
            else:
                self.aug_list.append(name_to_aug[key]())

    def __call__(self, img):
        if isinstance(img, list):
            img = [np.array(_img) for _img in img]
        else:
            img = np.array(img)

        for aug in self.aug_list:
            img = aug(img)

        return img

name_to_sig_aug = {
    'sig_jitter': SigJittering,
    'sig_scale': SigScale,
    'sig_magwarp': SigMagWarping,
}


class SignalAugmentor():
    def __init__(self, aug_params):
        self.aug_list = []
        
        for key, val in aug_params.items():
            if isinstance(val, int) or isinstance(val, float):
                self.aug_list.append(name_to_sig_aug[key](val))
            elif isinstance(val, list):
                self.aug_list.append(name_to_sig_aug[key](*val))
            else:
                self.aug_list.append(name_to_sig_aug[key]())

    def __call__(self, signal):
        for aug in self.aug_list:
            signal = aug(signal)

        return signal