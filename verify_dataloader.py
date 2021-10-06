# from labelme2coco.labelme2coco import labelme2coco_custom
# from labelme2coco import convert_customer_dataset
import labelme2coco
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse
import os
# from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd, basicDataset
from dataloaders.datasets import basicDataset
from torch.utils.data import DataLoader
import torch
import sys
import cv2
import numpy as np
from tools.custom_train import instance_custom_training


def save_img_mask(loader, output_dir):
    for ii, sample in enumerate(loader):
        if ii == 3:
            break
        batch_size = sample["image"].size()[0]
        img = sample['image'].numpy()
        gt = sample['label'].numpy()
        img_names =  sample['img_name']
        for jj in range(batch_size):
            segmap = gt[jj].astype(np.uint8)
            img_name = img_names[jj]
            img_name_perfix = img_name.split('.')[0]
            # segmap = decode_segmap(segmap, dataset='cityscapes')
            img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp.astype(np.uint8)
            # segmap[segmap>0] = 255
            
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

def varify_forward(args):
    # output_dir = args.output_dir
    # if output_dir and not os.path.exists(output_dir):
    #     os.makedirs(output_dir)

    if args.output_dir and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    root = args.input_dir

    basicDataset_train = basicDataset.BasicDataset(args, root, split="train")
    basicDataset_test = basicDataset.BasicDataset(args, root, split="test")

    train_loader = DataLoader(basicDataset_train, batch_size=2, shuffle=False, num_workers=2)
    # test_loader = DataLoader(basicDataset_test, batch_size=2, shuffle=False, num_workers=2)


    save_img_mask(train_loader, args.output_dir)


def pixellib_vis(args):
    input_dir   = args.inp


    vis_img = instance_custom_training()
    # vis_img = instance_custom_dataset_model_training()

    # vis_img.load_dataset(input_dir)
    vis_img.load_customer_dataset(input_dir)

    # vis_img.load_dataset("Nature")
    vis_img.visualize_sample()


if __name__ == '__main__':
    # from dataloaders.utils import decode_segmap

    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    parser = argparse.ArgumentParser()
    parser.add_argument('--inp', type=str, default=None)

    parser.add_argument('-im', '--input_dir', type=str, default='/home/hongrui/project/metro_pro/dataset/1st_2000')
    parser.add_argument('-om', '--output_dir', type=str, default='temp')
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
    parser.add_argument('--rotate_degree', type=int, default=15)
    parser.add_argument('--dataset', type=str, default='basicDataset')
    parser.add_argument('--dataset_dir', type=str, default=None, help='dataset dir')
    parser.add_argument('--testValTrain', type=int, default=-1, help='-1: no, 0: test, 1: testval, 2: trainval, 3: train')
    parser.add_argument('--testset_dir', type=str, default=None, help='input test image dir')
    parser.add_argument('--testOut_dir', type=str, default=None, help='test image output dir')
    args = parser.parse_args()

    
    # save_img_mask(test_loader, args.output_dir)
    # plt.show()
    # plt.show(block=True)

