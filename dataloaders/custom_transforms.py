from os import O_WRONLY
import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter
from dataloaders.synthesis.synthesize_sample import *
from PIL import Image, ImageChops
import cv2
import torchvision.transforms.functional as TF

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std
        
        
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # img_name = sample['img_name']
        img = np.array(img).astype(np.float32)
        if isinstance(mask, Image.Image):
            mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std
        # if img_name:
        #     return {'image': img,
        #         'label': mask,
        #         'img_name':img_name}
        # else:
        #     return {'image': img,
        #         'label': mask}
        sample['image'] = img
        sample['label'] = mask
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        # img_name = sample['img_name']
        
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        img = torch.from_numpy(img).float()
        if isinstance(mask, Image.Image):
            mask = np.array(mask).astype(np.float32)
            mask = torch.from_numpy(mask).float()

        # if img_name:
        #     return {'image': img,
        #         'label': mask,
        #         'img_name':img_name}
        # else:
        #     return {'image': img,
        #         'label': mask}
        sample['image'] = img
        sample['label'] = mask
        return sample


class RandomHorizontalFlip(object):
    def __init__(self, args = None):
        self.args = args
    def __call__(self, sample):
        if self.args.distinguish_left_right_semantic:
            return sample
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        # return {'image': img,
        #         'label': mask}
        sample['image'] = img
        sample['label'] = mask
        return sample

class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            rotate_degree = random.uniform(-1*self.degree, self.degree)
            img = img.rotate(rotate_degree, Image.BILINEAR)
            mask = mask.rotate(rotate_degree, Image.NEAREST)
            # return {'image': img,
            #         'label': mask}
            sample['image'] = img
            sample['label'] = mask
        return sample


class RandomGaussianBlur(object):
    def __call__(self, sample):
        if random.random() < 0.65:
            return sample
        img = sample['image']
        mask = sample['label']
        

        img = img.filter(ImageFilter.GaussianBlur(
            radius=random.random()))

        sample['image'] = img
        sample['label'] = mask
        return sample


class RandomScaleCrop(object):
    def __init__(self, base_size, crop_size, fill=0, args = None):
        self.base_size = base_size
        self.crop_size = crop_size
        self.fill = fill
        self.args = args

    def __call__(self, sample):
        if self.args.distinguish_left_right_semantic:
            short_size = random.randint(int(self.base_size * 0.85), int(self.base_size * 1.15))
        else:
            short_size = random.randint(int(self.base_size * 0.75), int(self.base_size * 1.25))
        img = sample['image']
        mask = sample['label']
        # random scale (short edge)
        
        w, h = img.size
        if h > w:
            ow = short_size
            oh = int(1.0 * h * ow / w)
        else:
            oh = short_size
            ow = int(1.0 * w * oh / h)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # pad crop
        if short_size < self.crop_size:
            padh = self.crop_size - oh if oh < self.crop_size else 0
            padw = self.crop_size - ow if ow < self.crop_size else 0
            img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
            mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # random crop crop_size
        w, h = img.size
        x1 = random.randint(0, w - self.crop_size)
        y1 = random.randint(0, h - self.crop_size)
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        # return {'image': img,
        #         'label': mask}
        sample['image'] = img
        sample['label'] = mask
        return sample


class RandomVerticalCrop(object):
    def __init__(self, base_size, crop_size, fill=0, args = None):
        self.fill = fill
        self.args = args

    def __call__(self, sample):
        seed = random.random()
        if seed < 0.75:
            return sample
        else:
            img = sample['image']
            mask = sample['label']
            # random scale (short edge)
            
            w, h = img.size
            if seed < 0.85:
                pass


class FixScaleCrop(object):
    def __init__(self, crop_size, args = None):
        self.crop_size = crop_size
        self.args = args

    def __call__(self, sample):
        if self.args.testValTrain <= 1:
            return sample
        if random.random() < 0.5:
            return sample
        # print('sample', sample)
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        # return {'image': img,
        #         'label': mask}
        sample['image'] = img
        sample['label'] = mask
        return sample

class ShortEdgeCrop(object):
    def __init__(self, hw_ratio = 1.5, args = None):
        self.hw_ratio = hw_ratio
        self.args = args

    def __call__(self, sample):
        if self.args.testValTrain <= 1:
            return sample
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            if w/h <= self.hw_ratio:
                return sample
            oh = h
            ow = int(self.hw_ratio*h)
        else:
            if h/w <= self.hw_ratio:
                return sample
            oh = int(self.hw_ratio*w)
            ow = w

        # center crop
        x1 = int(round((w - ow) / 2.))
        y1 = int(round((h - oh) / 2.))
        img = img.crop((x1, y1, x1 + ow, y1 + oh))
        mask = mask.crop((x1, y1, x1 + ow, y1 + oh))

        # return {'image': img,
        #         'label': mask}
        sample['image'] = img
        sample['label'] = mask
        return sample

# sample = {'image': _img, 'label': _target, 'img_name': img_name}

class FixedResize(object):
    def __init__(self, size, args = None):
        self.size = (size, size)  # size: (h, w)
        self.args = args

    def __call__(self, sample):
        if self.args.testValTrain <= 1:
            return sample
        img = sample['image']
        mask = sample['label']
        # img_name = sample['img_name']
        if isinstance(mask, Image.Image):
            assert img.size == mask.size
            mask = mask.resize(self.size, Image.NEAREST)
        img = img.resize(self.size, Image.BILINEAR)
        # if img_name:
        #     return {'image': img,
        #         'label': mask,
        #         'img_name':img_name}
        # else:
        #     return {'image': img,
        #         'label': mask}
        sample['image'] = img
        sample['label'] = mask
        return sample

class LimitResize(object):
    def __init__(self, size, args = None):
        self.size = size  # size: (h, w)
        self.args = args
        
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # print('type(mask)', type(mask))
        # img_name = sample['img_name']
        w, h = img.size
        resize_flag = True
        if w > h and w > self.size:
            oh = int(self.size/w*h)
            ow = self.size
        elif h > w and h > self.size:
            oh = self.size
            ow = int(self.size/h*w)
        else:
            resize_flag = False
        if resize_flag:
            if isinstance(mask, Image.Image):
                assert img.size == mask.size
                mask = mask.resize((ow, oh), Image.NEAREST)
            img = img.resize((ow, oh), Image.BILINEAR)
        # if img_name:
        #     return {'image': img,
        #         'label': mask,
        #         'img_name':img_name}
        # else:
        #     return {'image': img,
        #         'label': mask}
        
        sample['image'] = img
        sample['label'] = mask
        return sample
    
    

class RandomHorizontalFlipImageMask(object):
    def __init__(self, args = None):
        self.args = args
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if isinstance(mask, Image.Image): 
                mask = np.array(mask)
            if mask.max() > 1:
                mask_copy = mask.copy()
                mask[(mask_copy%2 == 0) * (mask_copy > 0)] -= 1
                mask[mask_copy%2==1] += 1
            mask = Image.fromarray(mask)
        sample['image'] = img
        sample['label'] = mask
        return sample


class RandomAddNegSample(object):
    def __init__(self, args = None):
        self.args = args
        self.AddNegSample = AddNegSample(coco_root = '/comp_robot/cv_public_dataset/COCO2017/')
        
    def __call__(self, sample):
        if self.args.testValTrain <= 1:
            return sample
        # return sample

        if self.args.add_neg_pixels_on_rails:
            return self.AddNegSample.forward(sample)
        else:
            return sample



class RandomShadows(object):
    def __init__(self, p=0.25, high_ratio=(1,2), low_ratio=(0.8, 1.), left_low_ratio =(0.4, 0.8), 
                    left_high_ratio=(0, 0.3), right_low_ratio=(0.4,0.8), right_high_ratio=(0, 0.3), args = None):
        self.p = p
        self.high_ratio = high_ratio
        self.low_ratio = low_ratio
        self.left_low_ratio = left_low_ratio
        self.left_high_ratio = left_high_ratio
        self.right_low_ratio = right_low_ratio
        self.right_high_ratio = right_high_ratio
        self.args = args

    @staticmethod
    def process(img, high_ratio, low_ratio, left_low_ratio, left_high_ratio, \
            right_low_ratio, right_high_ratio):

        w, h = img.size
        high_bright_factor = random.uniform(high_ratio[0], high_ratio[1])
        low_bright_factor = random.uniform(low_ratio[0], low_ratio[1])

        left_low_factor = random.uniform(left_low_ratio[0]*h, left_low_ratio[1]*h)
        left_high_factor = random.uniform(left_high_ratio[0]*h, left_high_ratio[1]*h)
        right_low_factor = random.uniform(right_low_ratio[0]*h, right_low_ratio[1]*h)
        right_high_factor = random.uniform(right_high_ratio[0]*h, right_high_ratio[1]*h)

        tl = (0, left_high_factor)
        bl = (0, left_high_factor+left_low_factor)

        tr = (w, right_high_factor)
        br = (w, right_high_factor+right_low_factor)

        contour = np.array([tl, tr, br, bl], dtype=np.int32)

        mask = np.zeros([h, w, 3],np.uint8)
        cv2.fillPoly(mask,[contour],(255,255,255))
        inverted_mask = cv2.bitwise_not(mask)
        # we need to convert this cv2 masks to PIL images
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # we skip the above convertion because our mask is just black and white
        mask_pil = Image.fromarray(mask)
        inverted_mask_pil = Image.fromarray(inverted_mask)

        low_brightness = TF.adjust_brightness(img, low_bright_factor)
        low_brightness_masked = ImageChops.multiply(low_brightness, mask_pil)
        high_brightness = TF.adjust_brightness(img, high_bright_factor)
        high_brightness_masked = ImageChops.multiply(high_brightness, inverted_mask_pil)

        return ImageChops.add(low_brightness_masked, high_brightness_masked)

    def __call__(self, sample):
        if self.args.testValTrain <= 1 or not self.args.use_RandomShadows:
            return sample
        
        if random.uniform(0, 1) < self.p:
            img = sample['image']        
            img = self.process(img, self.high_ratio, self.low_ratio, \
            self.left_low_ratio, self.left_high_ratio, self.right_low_ratio, \
            self.right_high_ratio)
            sample['image'] = img
        return sample