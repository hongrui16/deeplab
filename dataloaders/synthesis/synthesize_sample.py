from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
import os, sys
from skimage.filters import gaussian
# from color_transfer import color_transfer
import albumentations as albu
from PIL import Image, ImageOps, ImageFilter

from dataloaders.synthesis.gen_utils import *
from dataloaders.synthesis.coco_gen import CocoGen


from itertools import accumulate
from tools.util import *



class AddNegSample(object):
    def __init__(self, coco_root = '/comp_robot/cv_public_dataset/COCO2017/', coco_data_type='train2017', args = None):
        # self.coco_generator = CocoGen(coco_root, coco_data_type)
        self.scope = (0.04, 0.15)
  

    ## Color Correction
    def correct_colours(self, im1, im2, box):
        COLOUR_CORRECT_BLUR_FRAC = 0.75
        #LEFT_EYE_POINTS = list(range(42, 48))
        #RIGHT_EYE_POINTS = list(range(36, 42))

        #blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) - np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
        x, y, w, h = box
        blur_amount = max(w, h)
        #blur_amount /= 3
        blur_amount = 25


        blur_amount = int(blur_amount)
        if blur_amount % 2 == 0:
            blur_amount += 35
        im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
        im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)

        # Avoid divide-by-zero errors.
        im2_blur = im2_blur.astype(int)
        im2_blur += 128*(im2_blur <= 1)

        result = im2.astype(np.float64) * im1_blur.astype(np.float64) / im2_blur.astype(np.float64)
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result

    def check_intersection(self, rail_mask, mask, min_inter_ratio):
        inter = rail_mask * mask 
        
        if inter.sum()/mask.sum()<min_inter_ratio:
            return False
        else:
            return True

    def check(self, rail_mask, mask, min_pixels):
        inter = rail_mask * mask 

        inter_sum = inter.sum()
        
        if inter_sum>min_pixels or inter_sum>mask.sum()*0.25:
            return True
        else:
            return False

        


    def image_copy_paste(self, img, paste_img, alpha, blend=False, sigma=1):
        if alpha is not None:
            if blend:
                alpha = gaussian(alpha, sigma=sigma, preserve_range=True)

            img_dtype = img.dtype
            alpha = alpha[..., None]
            img = paste_img * alpha + img * (1 - alpha)
            img = img.astype(img_dtype)

        return img


    def forward(self, sample):
        rail_img = np.array(sample['image'])
        rail_mask = np.array(sample['label'])
        if not rail_mask.any() > 0:
            return sample
        
        # print('np.unique(rail_mask)', np.unique(rail_mask))
        left_mask, right_mask = sort_left_right_lane(rail_mask)
        if not isinstance(left_mask, np.ndarray) and not isinstance(left_mask, np.ndarray):
            return sample
        # print('left_mask.shape', left_mask.shape)
        # print('right_mask.shape', right_mask.shape)
        l_neg_img, l_neg_mask = self.randomly_get_neg_sample(rail_img.copy(), rail_mask.copy())
        r_neg_img, r_neg_mask = self.randomly_get_neg_sample(rail_img.copy(), rail_mask.copy())
        #print("mode {}".format(mode))
        # print('l_neg_mask.shape', l_neg_mask.shape)
        # print('r_neg_mask.shape', r_neg_mask.shape)
        l_zero_mask, l_zero_img = self.randomly_select_a_certer_on_a_rail(left_mask.copy(), l_neg_mask.copy(), l_neg_img.copy())
        r_zero_mask, r_zero_img = self.randomly_select_a_certer_on_a_rail(right_mask.copy(), r_neg_mask.copy(), r_neg_img.copy())

        inter_section_mask = (l_zero_mask > 0)*(r_zero_mask > 0)
        l_mask = (l_zero_mask > 0)*(~inter_section_mask)
        r_mask = (r_zero_mask > 0)*(~inter_section_mask)
        neg_img = self.mutitly_brg_img_with_mask(l_zero_img.copy(), l_mask) + \
                    self.mutitly_brg_img_with_mask(r_zero_img.copy(), r_mask)
        inter_section_img = 0.5*self.mutitly_brg_img_with_mask(l_zero_img.copy(), inter_section_mask)\
                            + 0.5*self.mutitly_brg_img_with_mask(r_zero_img.copy(), inter_section_mask)
        neg_img += inter_section_img
        # print(f'call {sys._getframe().f_code.co_name, sys._getframe().f_lineno}')
        # print('neg_img.shape', neg_img.shape)
        # for c in range(3):
        #     img_ = neg_img[:,:,c].copy()
        #     neg_img[:,:,c][img_ > 255] *= 0.5
        # neg_img[neg_img>255] *= 0.5
        # print(f'call {sys._getframe().f_code.co_name, sys._getframe().f_lineno}')
        
        neg_img = neg_img.astype(np.uint8)

        # print('neg_img.shape', neg_img.shape)
        neg_mask = l_zero_mask.astype(np.uint16) + r_zero_mask.astype(np.uint16)
        neg_mask[neg_mask>=255] = 1
        neg_mask = neg_mask.astype(np.uint8)

        # if random.random()<0.: #donot apply color transfer 
        #     #src_img = self.correct_colours(rail_img, src_img, bbox)
        #     rail_img = color_transfer(rail_img, neg_img) #颜色匹配

        rail_img = self.image_copy_paste(rail_img, neg_img, neg_mask, blend=True, sigma=1)
        rail_mask[neg_mask > 0] = 0
        rail_img = Image.fromarray(rail_img)
        rail_mask = Image.fromarray(rail_mask)
        sample['image'] = rail_img
        sample['label'] = rail_mask
        # img_name = sample['img_name']
        # mask_name = img_name.replace('.jpg', '.png')
        # rail_img.save(f'temp/{img_name}')
        # rail_mask.save(f'temp/{mask_name}')
        # print('----------')
        return sample

    def mutitly_brg_img_with_mask(self, img, mask):
        for c in range(3):
            img[:,:,c] *= mask
        return img

    def randomly_select_a_certer_on_a_rail(self, rail_mask, neg_sample_mask, neg_sample_img):
        HH, WW = rail_mask.shape
        zero_img = np.zeros((HH, WW, 3))
        if not rail_mask.any() > 0:
            return rail_mask, zero_img
        try:
            nonzeros = rail_mask.nonzero()
            nonzero_y = nonzeros[0]
            nonzero_x = nonzeros[1]

            random_index = int(random.random()*len(nonzero_y))
            cy = nonzero_y[random_index]
            cx = nonzero_x[random_index]

            
            h, w = neg_sample_mask.shape
            xmin = cx - w //2
            ymin = cy - h //2
            
            xmax = cx + w//2 if cx + w//2 <= WW else WW
            ymax = cy + h//2 if cy + h//2 <= HH else HH
            zero_mask = np.zeros(rail_mask.shape)
            offset_x = xmax-xmin
            offset_y = ymax-ymin
            zero_mask[ymin:ymin+offset_y, xmin:xmin+ offset_x] = neg_sample_mask[0:offset_y, 0:offset_x]
            zero_img[ymin:ymin+offset_y, xmin:xmin+ offset_x] = neg_sample_img[0:offset_y, 0:offset_x]
        except Exception as e:
            # print(offset_x, offset_y)
            return rail_mask, zero_img 
        return zero_mask, zero_img


    def randomly_get_neg_sample(self, rail_img, rail_mask):
        # print(f'call {sys._getframe().f_code.co_name, sys._getframe().f_lineno}')
        # sel = random.random()
        sel = 1
        scope = self.scope
        if sel <= 0.5:
            hei, wid, _ = rail_img.shape
            img_block = self.coco_generator.get_img_block()  
            h, w, _ = img_block.shape
            random_mask = get_random_shape(width=w, height=h)
            random_mask = cv2.resize(random_mask, (w,h))
            # print('random_mask', random_mask.shape, 'img_block', img_block.shape)
            for c in range(3):
                img_block[:,:, c] *= random_mask
            w = np.random.randint(int(scope[0]*wid), int(scope[1]*wid))
            h = np.random.randint(int(scope[0]*hei), int(scope[1]*hei))
            neg_img = cv2.resize(img_block, (w,h))
            neg_mask = cv2.resize(random_mask, (w, h))
            neg_mask[neg_mask>0] = 255
        else:
            neg_img, neg_mask = self.randomly_get_neg_block_from_rail_images(rail_img.copy(), rail_mask.copy(), scope)
            neg_mask[neg_mask>0] = 255

        return neg_img, neg_mask


    def get_centeral_coors_of_mask(self, mask):
        non_y, non_x = np.nonzero(mask)
        n_x_min = non_x.min()
        n_x_max = non_x.max()
        n_y_min = non_y.min()
        n_y_max = non_y.max()
        c_x = (n_x_max + n_x_min)//2
        c_y = (n_y_min + n_y_max)//2
        return (c_x, c_y)

    def get_nonzero_area_bbox(self, mask):
        non_y, non_x = np.nonzero(mask)
        n_x_min = non_x.min()
        n_x_max = non_x.max()
        n_y_min = non_y.min()
        n_y_max = non_y.max()
        return (n_x_min, n_y_min, n_x_max, n_y_max)


    def randomly_get_neg_block_from_rail_images(self, img, mask, scope):
        # print(f'call {sys._getframe().f_code.co_name, sys._getframe().f_lineno}')
        hei, wid, _ = img.shape
        # scope = (0.08, 0.2)
        while True:
            w = np.random.randint(int(scope[0]*wid), int(scope[1]*wid))
            h = np.random.randint(int(scope[0]*hei), int(scope[1]*hei))
            random_mask = get_random_shape(width=w, height=h)
            random_mask = cv2.resize(random_mask, (w,h))
            # print(f'call {sys._getframe().f_code.co_name, sys._getframe().f_lineno}')
            while True:
                expand_mask = self.randomly_pad_mask_with_zero(random_mask, hei, wid)
                inter = (mask>0)*(expand_mask>0)
                if not inter.any() == True:
                    break
            # print(f'call {sys._getframe().f_code.co_name, sys._getframe().f_lineno}')
            # print('expand_mask.shape', expand_mask.shape)
            # print('img.shape', img.shape)
            for c in range(3):
                img[:,:, c] *= expand_mask > 0

            # print(f'call {sys._getframe().f_code.co_name, sys._getframe().f_lineno}')
            x_min, y_min, x_max, y_max = self.get_nonzero_area_bbox(expand_mask)
            if x_max - x_min > wid*scope[0]/2 and y_max - y_min > hei*scope[0]/2:
                break
        # print(f'call {sys._getframe().f_code.co_name, sys._getframe().f_lineno}')
        return img[y_min:y_max,x_min:x_max],  expand_mask[y_min:y_max,x_min:x_max]


    def randomly_pad_mask_with_zero(self, mask, tar_h, tar_w):
        zeros = np.zeros((tar_h, tar_w))
        h, w = mask.shape
        offset_w = tar_w - w
        offset_h = tar_h - h
        start_y = np.random.randint(0, offset_h)
        start_x = np.random.randint(0, offset_w)
        zeros[start_y:start_y+h, start_x:start_x+w] = mask
        return zeros

if __name__ == '__main__':
    pass
    # OBJ = AddNegSample()
    # rail_img = np.zeros((233, 322, 3)) 
    # rail_mask = np.zeros((233, 322)) 
    # OBJ.randomly_get_neg_sample(rail_img, rail_mask)