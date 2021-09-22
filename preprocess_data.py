import argparse
import os
import numpy as np
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt
import sys
import shutil
import random

def sort_data(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    in_label_dir = os.path.join(input_dir, '500pic_label')
    in_img_dir = os.path.join(input_dir, 'Part1_2000pic')
    label_names = os.listdir(in_label_dir)
    img_names = os.listdir(in_img_dir)

    # out_label_dir = os.path.join(output_dir, 'test', 'label')
    # out_img_dir = os.path.join(output_dir, 'test', 'image')
    out_label_dir = os.path.join(output_dir, 'label')
    out_img_dir = os.path.join(output_dir, 'image')
    print(len(label_names), len(img_names))

    for i, l_name in enumerate(label_names):
        print(f'processing {l_name} {i+1}/{len(label_names)}')
        i_label_filepath = os.path.join(in_label_dir, l_name)
        l_name_prefix = l_name.split('.')[0]
        o_l_name = f'{l_name_prefix}.png'
        img_name = f'{l_name_prefix}.jpg'
        i_img_filepath = os.path.join(in_img_dir, img_name)
        if not os.path.exists(i_img_filepath):
            continue
        o_img_filepath = os.path.join(out_img_dir, img_name)
        o_label_filepath = os.path.join(out_label_dir, o_l_name)

        label = cv2.imread(i_label_filepath)
        cv2.imwrite(o_label_filepath, label)
        shutil.move(i_img_filepath, o_img_filepath)


def split_train_val_data(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    in_label_dir = os.path.join(input_dir, '1500pic_label')
    in_img_dir = os.path.join(input_dir, 'Part1_2000pic')
    label_names = os.listdir(in_label_dir)
    img_names = os.listdir(in_img_dir)

    val_out_label_dir = os.path.join(output_dir, 'val', 'label')
    val_out_img_dir = os.path.join(output_dir, 'val', 'image')

    train_out_label_dir = os.path.join(output_dir, 'train', 'label')
    train_out_img_dir = os.path.join(output_dir, 'train', 'image')
    # out_label_dir = os.path.join(output_dir, 'label')
    # out_img_dir = os.path.join(output_dir, 'image')
    print(len(label_names), len(img_names))
    random.shuffle(label_names)
    for i, l_name in enumerate(label_names):
        print(f'processing {l_name} {i+1}/{len(label_names)}')
        i_label_filepath = os.path.join(in_label_dir, l_name)
        l_name_prefix = l_name.split('.')[0]
        o_l_name = f'{l_name_prefix}.png'
        img_name = f'{l_name_prefix}.jpg'
        i_img_filepath = os.path.join(in_img_dir, img_name)
        if not os.path.exists(i_img_filepath):
            continue
        if i < 300:
            out_img_dir = val_out_img_dir
            out_label_dir = val_out_label_dir
        else:
            out_img_dir = train_out_img_dir
            out_label_dir = train_out_label_dir
        o_img_filepath = os.path.join(out_img_dir, img_name)
        o_label_filepath = os.path.join(out_label_dir, o_l_name)

        label = cv2.imread(i_label_filepath)
        cv2.imwrite(o_label_filepath, label)
        shutil.move(i_img_filepath, o_img_filepath)

def print_size(args):
    in_img_dir = args.im
    img_names = os.listdir(in_img_dir)


    for i, img_name in enumerate(img_names):
        print(f'processing {img_name} {i+1}/{len(img_names)}')        
        i_img_filepath = os.path.join(in_img_dir, img_name)
        img = cv2.imread(i_img_filepath)
        print(f'{img.shape}\n')

def pring_data(args):
    in_img_filepath = args.im

    th = 30
    
    img = cv2.imread(in_img_filepath, 0)
    h, w = img.shape
    temp = img[h//2 - 30:h//2+30, w//2-30:w//2+30]
    img[img>th] = 255
    img[img<=th] = 0
    cv2.imwrite('youtube_000859_q.png', img)
    print(temp)
    opt=vars(args)
    for key, val in opt.items():
        print(key, val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IDEA Training')
    parser.add_argument('--im', '--input_dir', type=str, default=None)
    parser.add_argument('-om', '--output_dir', type=str, default=None)

    args = parser.parse_args()
    # sort_data(args)
    # split_train_val_data(args)
    # print_size(args)
    pring_data(args)