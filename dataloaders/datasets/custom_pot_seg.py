import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from torchvision import transforms
from dataloaders import custom_transforms as tr
# from dataloaders import joint_transforms as jo_trans
# from dataloaders.synthesis.synthesize_sample import AddNegSample as RandomAddNegSample

import logging
from os import listdir
from os.path import splitext
import torch
from torch.utils.data import Dataset
import cv2
import sys
import albumentations as albu
import random
from tools.util import *

class CustomPotSeg(Dataset):

    def __init__(self, args, root=None, split="train"):
        self.args = args
        self.root = args.dataset_dir
        self.split = split
        self.ignore_index = args.ignore_index
        
        # print('args.ignore_index', args.ignore_index)
        self.spatial_trans, self.pixel_trans = self.albumentations_aug()
        if not self.args.use_txtfile:
            self.base_dir = os.path.join(self.root, self.split)
            if args.testset_dir:
                self.images_base =  args.testset_dir
                self.annotations_base = ''
            else:
                self.images_base = os.path.join(self.base_dir, 'image')
                self.annotations_base = os.path.join(self.base_dir, 'label')
            # print('annotations_base', self.annotations_base)
            # self.ids = [splitext(file)[0] for file in listdir(self.images_base) if not file.startswith('.')]
            # self.img_ids = [file for file in listdir(self.images_base) if not file.startswith('.')]
            # random.shuffle(self.img_ids)
            self.img_filepaths = []
            for file in listdir(self.images_base):
                img_filepath = os.path.join(self.images_base, file)
                self.img_filepaths.append(img_filepath)
            random.shuffle(self.img_filepaths)
            
        else:
            txt_filepath = os.path.join(self.root, f'{self.split}.txt')
            self.img_filepaths = read_txt_to_list(txt_filepath)
            


    def __len__(self):
        # return len(self.img_ids)
        return len(self.img_filepaths)

    def __getitem__(self, index):
        # print(f'calling {__file__}, {sys._getframe().f_lineno}')
        if self.split == "train" and self.args.use_albu:

            img_path = self.img_filepaths[index]
            _img = cv2.imread(img_path)
            # print('img_path', _img.shape, img_path)
            img_name = img_path.split('/')[-1]
            # lbl_path = img_path.replace('image', 'label').replace('.jpg', '.png')
            lbl_path = img_path.replace('.jpg', '.png')
            _tmp = cv2.imread(lbl_path, 0)
            _tmp, _img = self.encode_segmap(_tmp, _img)
            # print('lbl_path', _tmp.shape, lbl_path)

            
            _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
            _target = _tmp

            sample = {'image': Image.fromarray(_img), 'label': Image.fromarray(_target), 'img_name': img_name}
            # print('sample', sample)
            sample = self.transform_train1(sample)
            _img = np.array(sample['image'])
            _target = np.array(sample['label'])        

            _img = self.pixel_trans(image=_img)['image']

            #to tensor
            _img = _img.transpose( (2, 0, 1) )
            # print('_img', _img.shape, '_target', _target.shape)
            _img = torch.from_numpy(_img).float()
            _target = torch.from_numpy(_target).float()            

            sample = {'image': _img, 'label': _target, 'img_name': img_name}
            return sample

        else:
            img_path = self.img_filepaths[index]
            img_name = img_path.split('/')[-1]
            lbl_path = img_path.replace('.jpg', '.png')
            # lbl_path = img_path.replace('image', 'label').replace('.jpg', '.png')
            _img = cv2.imread(img_path)
            # _img = Image.open(img_path).convert('RGB')
            h, w, _ = _img.shape 
            if self.args.testset_dir:
                # w, h = _img.size
                # _target = np.zeros((h,w))
                # _target = Image.fromarray(_target)
                # lbl_path = img_path.replace('image', 'label').replace('.jpg', '.png')
                lbl_path = img_path.replace('.jpg', '.png')
                if os.path.exists(lbl_path):
                    # print('lbl_path', lbl_path)
                    _tmp = cv2.imread(lbl_path, 0).astype(np.uint8)
                else:
                    # print('lbl_path none')
                    _tmp = np.zeros((h,w), dtype=np.uint8)
                # _tmp = self.encode_segmap(_tmp)
                # _target = Image.fromarray(_tmp)
            else:
                _tmp = cv2.imread(lbl_path, 0).astype(np.uint8)
            _tmp, _img = self.encode_segmap(_tmp, _img)
            # print('_img.shape', _img.shape)
            _img = cv2.cvtColor(_img, cv2.COLOR_BGR2RGB)
            _img = Image.fromarray(_img)

                
            _target = Image.fromarray(_tmp)

            sample = {'image': _img, 'label': _target, 'img_name': img_name}

            if self.split == 'train':
                return self.transform_train(sample)
            elif self.split == 'val':
                return self.transform_val(sample)
            elif self.split == 'test' or self.args.testset_dir:
                return self.transform_test(sample)

    def encode_segmap(self, mask, img = None):
        '''
        label_dict = {'lasi_heavy': 11, 'lasi_medium':12, 'lasi_slight':13,
        'gengshang_heavy':21, 'gengshang_medium':22, 'gengshang_slight':23,  
        'gengshi_heavy':31, 'gengshi_medium':32, 'gengshi_slight':33,
        'shayan_heavy':41, 'shayan_medium':42, 'shayan_medium':43,
        'huahen_heavy':51, 'huahen_medium':52, 'huahen_medium':53,
        'zhoubian_heavy':61, 'zhoubian_medium':62, 'zhoubian_medium':63,
        'bowen_heavy':71, 'bowen_medium':72, 'bowen_medium':73,
        'youwu_heavy':81, 'youwu_medium':82, 'youwu_medium':83,
        }

        '''        
        if mask.any() > 0:
            mask_bk = mask.copy()
            mask[mask_bk == 13] = 0
            mask[mask_bk == 23] = 0
            mask[mask_bk == 33] = 0
            mask[mask_bk == 43] = 0
            mask[mask_bk == 53] = 0
            if self.args.pot_train_mode == 1: #不区分类别
                mask[mask_bk >= 61] = 0
                mask[mask_bk>0] = 1
            
            mask[mask_bk==self.args.ignore_index] = self.args.ignore_index #255
        return mask, img.astype(np.uint8)

    def transform_train(self, sample):
        composed_transforms = transforms.Compose([
            tr.RamdomCutPostives(size=self.args.base_size, args = self.args),
            tr.ShortEdgePad(size=self.args.base_size, args = self.args),
            # tr.ShortEdgeCrop(hw_ratio= self.args.hw_ratio, args = self.args),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=self.ignore_index, args = self.args),
            # tr.RandomAddNegSample(args = self.args),
            tr.RandomHorizontalFlip(self.args),
            tr.RandomVerticalFlip(self.args),            
            tr.RandomRotate(degree = self.args.rotate_degree),
            tr.RandomGaussianBlur(),
            # tr.FixScaleCrop(crop_size=self.args.crop_size, args = self.args),
            # tr.RandomHorizontalFlipImageMask(self.args),
            tr.FixedResize(size=self.args.base_size, args = self.args),
            tr.RandomShadows(args = self.args),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):
        composed_transforms = transforms.Compose([
            # tr.RamdomCutPostives(size=self.args.base_size, args = self.args),
            tr.ShortEdgePad(size=self.args.base_size, args = self.args),
            # tr.ShortEdgeCrop(hw_ratio= self.args.hw_ratio, args = self.args),
            # tr.RandomAddNegSample(args = self.args),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=self.ignore_index, args = self.args),
            tr.FixScaleCrop(crop_size=self.args.crop_size, args = self.args),
            tr.FixedResize(size=self.args.base_size, args = self.args),
            # tr.LimitResize(size=self.args.max_size, args = self.args),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

    def transform_test(self, sample):
        composed_transforms = transforms.Compose([
            tr.RamdomCutPostives(size=self.args.base_size, args = self.args),
            tr.ShortEdgePad(size=self.args.base_size, args = self.args),
            # tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=self.ignore_index, args = self.args),
            # tr.RandomAddNegSample(args = self.args),
            # tr.CenterPadAndCrop(size=self.args.base_size, args = self.args),
            tr.FixedResize(size=self.args.base_size, args = self.args),
            # tr.LimitResize(size=self.args.max_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])
        return composed_transforms(sample)

    def transform_train1(self, sample):
        composed_transforms = transforms.Compose([
            tr.RamdomCutPostives(size=self.args.base_size, args = self.args),
            tr.ShortEdgePad(size=self.args.base_size, args = self.args),
            # tr.ShortEdgeCrop(hw_ratio= self.args.hw_ratio, args = self.args),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=self.ignore_index, args = self.args),
            tr.RandomHorizontalFlip(self.args),
            tr.RandomVerticalFlip(self.args),
            # tr.RandomAddNegSample(args = self.args),
            tr.RandomRotate(degree = self.args.rotate_degree),
            tr.RandomShadows(args = self.args),
            tr.RandomGaussianBlur(),
            tr.FixScaleCrop(crop_size=self.args.crop_size, args = self.args),
            # tr.RandomHorizontalFlipImageMask(self.args),
            ])

        return composed_transforms(sample)

    def albumentations_aug(self):
        args = self.args

        spatial_trans = albu.Compose([
            #albu.RandomSizedCrop(args.base_size, scale=(0.6, 1.0), ratio=(0.75, 1.3333333333333333)),
            albu.SmallestMaxSize(args.base_size, p=1.),
            albu.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=20, interpolation=1, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=args.ignore_index, p=0.5),
            #albu.ShiftScaleRotate(shift_limit=0.0, scale_limit=0.0, rotate_limit=90, interpolation=1, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=args.ignore_index, p=0.5),
            albu.PadIfNeeded(min_height=args.base_size, min_width=args.base_size, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=args.ignore_index, p=1.),
            #albu.RandomCrop(args.base_size, args.base_size, p=1.),
            albu.CenterCrop(args.base_size, args.base_size, p=1.),
            albu.Flip(p=0.5),              
            albu.OneOf([
                #albu.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
                albu.GridDistortion(p=0.5),
                albu.OpticalDistortion(distort_limit=2, shift_limit=0.5, p=0.5)                  
                ], p=0.3),
            #albu.CoarseDropout (max_holes=8, max_height=int(args.base_size*0.1), max_width=int(args.base_size*0.1), fill_value=0, mask_fill_value=args.ignore_index, p=0.3)
            albu.CoarseDropout (max_holes=32, max_height=20, max_width=20, fill_value=255, mask_fill_value=0, p=0.3)
            ], p=1.)


        pixel_trans = albu.Compose([
            #albu.OneOf([
            #    albu.CLAHE(clip_limit=2, p=.5),
            #    albu.Sharpen(p=.25),
            #    ], p=0.35),
            albu.RandomBrightnessContrast(p=.4),
            albu.OneOf([
                # albu.GaussNoise(p=.2),
                albu.ISONoise(p=.2),
                albu.ImageCompression(quality_lower=75, quality_upper=100, p=.4)
                ], p=.4),
            albu.RGBShift(p=.4),
            albu.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=12, val_shift_limit=8, p=.4),
            #albu.ToGray(p=.2),
            albu.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), p=1.0),
            ], p=1.)
        return spatial_trans, pixel_trans

    # skip boundary pixels to handle nosiy annotation
    def skip_boundary(self, mask):
        mat = mask >= 1
        mat = mat.astype(np.uint8)*255        
        edges = cv2.Canny(mat,240,240)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)), iterations=1) 
        indices = edges == 255 
        mask[indices] = self.ignore_index

        return mask


if __name__ == '__main__':
    # from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse
    import os
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    parser = argparse.ArgumentParser()
    parser.add_argument('-im', '--input_dir', type=str, default=None)
    parser.add_argument('-om', '--output_dir', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=16,
                    metavar='N', help='input batch size for \
                            training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=8,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--hw_ratio', type=float, default=1.25)
    parser.add_argument('--ignore_index', type=int, default=255)

    parser.add_argument('--base_size', type=int, default=640)
    parser.add_argument('--crop_size', type=int, default=640)
    parser.add_argument('--max_size', type=int, default=1080)


    parser.add_argument('--rotate_degree', type=int, default=15)
    parser.add_argument('--dataset', type=str, default='basicDataset')
    parser.add_argument('--dataset_dir', type=str, default=None, help='dataset dir')
    parser.add_argument('--testValTrain', type=int, default=-2, help='-1: infer, 0: test, 1: testval, 2: train, 3: trainval, 4: trainvaltest')
    parser.add_argument('--testset_dir', type=str, default=None, help='input test image dir')
    parser.add_argument('--testOut_dir', type=str, default=None, help='test image output dir')
    parser.add_argument('--distinguish_left_right_semantic', action='store_true', default=True, help='distinguish left and right rail semantic segmentation')
    parser.add_argument('--skip_boundary', action='store_true', default=False, help="skip boundary pixel to handle annotation noise")

    parser.add_argument('--use_albu', action='store_true', default=False, help="indicate wheather to use albumentation in training phase for data augmentation")


    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    root = args.input_dir

    basicDataset_train = BasicDataset(args, root, split="train")
    basicDataset_test = BasicDataset(args, root, split="test")

    train_loader = DataLoader(basicDataset_train, batch_size=2, shuffle=False, num_workers=2)
    test_loader = DataLoader(basicDataset_test, batch_size=2, shuffle=False, num_workers=2)

    def save_img_mask(loader):
        for ii, sample in enumerate(loader):
            if ii == 3:
                break
            batch_size = sample["image"].size()[0]
            # print('batch_size: ', batch_size)
            for jj in range(batch_size):

                img = sample['image'].numpy()
                gt = sample['label'].numpy()
                img_name =  sample['img_name']
                img_name_perfix = img_name.split('.')[0]
                segmap = np.array(gt[jj]).astype(np.uint8)
                # segmap = decode_segmap(segmap, dataset='cityscapes')
                img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
                img_tmp *= (0.229, 0.224, 0.225)
                img_tmp += (0.485, 0.456, 0.406)
                img_tmp *= 255.0
                img_tmp = img_tmp.astype(np.uint8)
                
                # plt.figure()
                # plt.title('display')
                # plt.subplot(211)
                # plt.imshow(img_tmp)
                # plt.subplot(212)
                # plt.imshow(segmap)
                # ax = plt.subplot(4, batch_size*2, ii*batch_size*2 + 2*jj+1), plt.imshow(img_tmp), plt.title(f'img_{ii*batch_size + jj}'), plt.xticks([]), plt.yticks([])
                # ax = plt.subplot(4, batch_size*2, ii*batch_size*2 + 2*jj+2), plt.imshow(segmap*60), plt.title(f'mask_{ii*batch_size + jj}'), plt.xticks([]), plt.yticks([])
                # if segmap.ndim == 2:
                #     plt.gray()

                cv2.imwrite(os.path.join(output_dir, f'{img_name_perfix}.jpg'), img_tmp)
                cv2.imwrite(os.path.join(output_dir, f'{img_name_perfix}.png'), segmap*60)

    save_img_mask(train_loader)
    save_img_mask(test_loader)
    # plt.show()
    # plt.show(block=True)

