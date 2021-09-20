import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
from torchvision import transforms
from dataloaders import custom_transforms as tr
# from dataloaders import joint_transforms as jo_trans

import logging
from os import listdir
from os.path import splitext
from mypath import Path
import torch
from torch.utils.data import Dataset
import cv2
import sys

class BasicDataset(Dataset):

    def __init__(self, args, root=Path.db_root_dir('basicDataset'), split="train"):
        if args.dataset_dir:
            self.root = args.dataset_dir
        else:
            self.root = root
        self.split = split
        self.base_dir = os.path.join(self.root, self.split)
        self.ignore_index = args.ignore_index
        self.args = args
        # print('args.ignore_index', args.ignore_index)
        self.images_base = os.path.join(self.base_dir, 'image')
        self.annotations_base = os.path.join(self.base_dir, 'label')
        # self.ids = [splitext(file)[0] for file in listdir(self.images_base) if not file.startswith('.')]
        self.img_ids = [file for file in listdir(self.images_base) if not file.startswith('.')]
        
        # self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        # self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        # self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', \
        #                     'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain', \
        #                     'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', \
        #                     'motorcycle', 'bicycle']
        # self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))


    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, index):
        # print(f'calling {__file__}, {sys._getframe().f_lineno}')
        
        img_name = self.img_ids[index]
        img_path = os.path.join(self.images_base, img_name)
        lbl_path = os.path.join(self.annotations_base, splitext(img_name)[0]+'.png')

        _img = Image.open(img_path).convert('RGB')
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _tmp = self.encode_segmap(_tmp)
        _target = Image.fromarray(_tmp)


        sample = {'image': _img, 'label': _target, 'img_name': img_name}

        if self.split == 'train':
            return self.transform_train(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_test(sample)

    def encode_segmap(self, mask):
        # Put all void classes to zero
        # for _voidc in self.void_classes:
        #     mask[mask == _voidc] = self.ignore_index
        # for _validc in self.valid_classes:
        #     mask[mask == _validc] = self.class_map[_validc]
        # mask //= 2
        mask[mask>0] = 1
        return mask

    def transform_train(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=self.ignore_index),
            tr.RandomHorizontalFlip(),
            tr.RandomRotate(degree = self.args.rotate_degree),
            # tr.RandomCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=self.ignore_index),
            
            tr.RandomGaussianBlur(),
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_test(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

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
    parser.add_argument('--ignore_label', type=int, default=255)
    parser.add_argument('--base_size', type=int, default=640)
    parser.add_argument('--crop_size', type=int, default=640)
    parser.add_argument('--rotate_degree', type=int, default=15)
    args = parser.parse_args()

    output_dir = args.output_dir
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    root = args.input_dir

    basicDataset_train = BasicDataset(args, root, split="train")

    dataloader = DataLoader(basicDataset_train, batch_size=2, shuffle=False, num_workers=2)

    for ii, sample in enumerate(dataloader):
        batch_size = sample["image"].size()[0]
        # print('batch_size: ', batch_size)
        for jj in range(batch_size):

            img = sample['image'].numpy()
            gt = sample['label'].numpy()
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

            cv2.imwrite(os.path.join(output_dir, f'{ii*batch_size + jj}.jpg'), img_tmp)
            cv2.imwrite(os.path.join(output_dir, f'{ii*batch_size + jj}.png'), segmap*60)
        if ii == 3:
            break
    # plt.show()
    # plt.show(block=True)

