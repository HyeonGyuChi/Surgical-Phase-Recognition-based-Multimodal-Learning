import cv2
import torch
import numpy as np


# TODO 수정 필요

# Image Augmentations
class Resize(object):
    def __init__(self, re_sz):
        self.re_sz = re_sz

    def __call__(self, img):
        return cv2.resize(img, dsize=(self.re_sz, self.re_sz), interpolation=cv2.INTER_NEAREST)

class RandomCrop(object):
    def __init__(self, crop_sz):
        self.crop_sz = crop_sz

    def __call__(self, img):
        h, w, c = img.shape
        mh = (h - self.crop_sz)//2
        mw = w // 2
        hw = self.crop_sz // 2

        img_list = []
        _img = img[mh:-mh, :self.crop_sz]
        img_list.append(cv2.resize(_img, dsize=(self.crop_sz, self.crop_sz), interpolation=cv2.INTER_NEAREST))
        _img = img[mh:-mh, mw-hw:mw+hw]
        img_list.append(cv2.resize(_img, dsize=(self.crop_sz, self.crop_sz), interpolation=cv2.INTER_NEAREST))

        imgs = torch.vstack([torch.Tensor(_img).permute(2, 0, 1) for _img in img_list])

        return imgs        

class ToTensor(object):
    def __init__(self):
        self.test = None

    def __call__(self, img):
        img = img / 255.0
        return torch.Tensor(img)

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        return (img - self.mean) / self.std

# Segmentation Augmentations
class TemporalSegResize(object):
    def __init__(self, re_sz):
        self.re_sz = re_sz

    def __call__(self, img_list):
        return [cv2.resize(img, dsize=(self.re_sz, self.re_sz), interpolation=cv2.INTER_NEAREST) for img in img_list]

# Video Augmentations
class TemporalResize(object):
    def __init__(self, re_sz):
        self.re_sz = re_sz

    def __call__(self, img_list):
        return [cv2.resize(img, dsize=(self.re_sz, self.re_sz)) for img in img_list]

class TemporalColorJitter(object):
    def __init__(self, brightness, contrast):
        # self.hue = hue
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, img_list):
        res_list = []

        for img in img_list:
            img = img.astype('float')
            # if self.hue is not None:
            #     img = 
            if self.brightness is not None:
                if np.random.random() >= 0.5:
                    b = np.random.random_sample() * (self.brightness[1]-self.brightness[0]) + self.brightness[0]
                    img += b
                    img = np.clip(img, 0, 255)
            if self.contrast is not None:
                if np.random.random() >= 0.5:
                    c = np.random.random_sample() * (self.contrast[1]-self.contrast[0]) + self.contrast[0]
                    img *= c
                    img = np.clip(img, 0, 255)

            img = img.astype('uint8')
            res_list.append(img)

        return res_list

class TemporalRandomCrop(object):
    def __init__(self, crop_sz, consistency=True):
        self.crop_sz = crop_sz
        self.consistency = consistency

    def __call__(self, img_list):
        h, w, c = img_list[0].shape

        if self.consistency:
            i = np.random.randint(0, h-self.crop_sz, 1)[0]
            j = np.random.randint(0, w-self.crop_sz, 1)[0]
            
            _img_list = [img[i:i+self.crop_sz, j:j+self.crop_sz, ] for img in img_list]
        else:
            _img_list = []
            for idx in range(len(img_list)):
                i = np.random.randint(0, h-self.crop_sz, 1)
                j = np.random.randint(0, w-self.crop_sz, 1)

                _img_list.append(img_list[idx,i:i+self.crop_sz, j:j+self.crop_sz, ])

        # img_list = torch.stack([torch.Tensor(_img).permute(2, 0, 1) for _img in _img_list], dim=0)
        img_list = [torch.Tensor(_img).permute(2, 0, 1) for _img in _img_list]

        return img_list 


class TemporalCenterCrop(object):
    def __init__(self, crop_sz):
        self.crop_sz = crop_sz

    def __call__(self, img_list):
        h, w, c = img_list[0].shape
        hh = (h-self.crop_sz) // 2
        hw = (w-self.crop_sz) // 2
        
        _img_list = [img[hh:h-hh, hw:w-hw, ] for img in img_list]

        img_list = [torch.Tensor(_img).permute(2, 0, 1) for _img in _img_list]

        return img_list 


class TemporalFlip(object):
    def __init__(self, prob, axis='hor'):
        self.prob = prob
        self.axis = axis

    def __call__(self, img_list):
        if np.random.random() >= self.prob:
            _img_list = []

            for img in img_list:
                if self.axis == 'hor':
                    img = torch.flip(img, [-1])

                elif self.axis == 'ver':
                    img = torch.flip(img, [-2])

                _img_list.append(img)

        else:
            _img_list = img_list

        return _img_list


class TemporalToTensor(object):
    def __init__(self):
        self.test = None

    def __call__(self, img_list):
        return [img / 255.0 for img in img_list]
        

class TemporalNormalize(object):
    def __init__(self, mean, std, is_rgb):
        self.mean = mean
        self.std = std
        self.is_rgb = is_rgb

    def __call__(self, img_list):
        if isinstance(self.mean, list):
            tmp = []

            for img in img_list:
                if self.is_rgb:
                    img = torch.flip(img, [0])
                    
                for i in range(3):
                    img[i, :, :] = (img[i, :, :] - self.mean[i]) / self.std[i]

                tmp.append(img)
            
            return tmp
        else:
            if self.is_rgb:
                _img_list = []

                for img in img_list:
                    img = torch.flip(img, [0])
                    _img_list.append((img-self.mean) / self.std)

                return _img_list
            else:
                return [(img - self.mean) / self.std for img in img_list]


##################################################################
################## Signal Augmentations ##########################
##################################################################
class SigJittering(object):
    def __init__(self, sigma=0.05):
        self.sigma = sigma

    def __call__(self, signal):
        return signal + torch.normal(0, self.sigma, signal.size())

class SigScale(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, signal):
        return signal * torch.normal(1.0, self.sigma, signal.size())

class SigMagWarping(object):
    def __init__(self, sigma=0.01):
        self.sigma = sigma

    def __call__(self, signal):
        return signal * torch.normal(1.0, self.sigma, signal.size())

