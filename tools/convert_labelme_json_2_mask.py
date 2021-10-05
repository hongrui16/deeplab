import scipy
import numpy
from PIL import Image
import math
from math import pi
import imageio
import cv2
import numpy as np
import argparse
import os
import sys
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from scipy import ndimage
from scipy.ndimage.filters import convolve
from scipy.special import softmax
import time
from time import gmtime, strftime

import shutil
import json
import skimage.exposure
import base64
import os.path as osp
from labelme import utils
import imgviz
import random

def select_images(args):
    img_filepath    = args.img_filepath
    input_img_dir   = args.input_dir
    output_dir      = args.output_dir


    for root, dirs, files in os.walk(input_img_dir, topdown=True):
        for name in files:
            if '.json' in name:
                json_filepath = os.path.join(root, name)
                img_filepath = json_filepath.replace('.json', '.jpg')
                if os.path.exists(img_filepath):
                    out_json_filepath = os.path.join(output_dir, name)
                    out_img_filepath = out_json_filepath.replace('.json', '.jpg')
                    shutil.move(json_filepath, out_json_filepath)
                    shutil.move(img_filepath, out_img_filepath)


def labelme_json_to_dataset_fun(json_filepath, out_dir):
 
    if not osp.exists(out_dir):
        os.mkdir(out_dir)
       
    if '\\' in json_filepath:
        json_name = json_filepath.split('\\')[-1]
    elif '/' in json_filepath:
        json_name = json_filepath.split('/')[-1]
    else:
        json_name = json_filepath

    prefix_name = json_name.split('.')[0]

    data = json.load(open(json_filepath))
    imageData = data.get("imageData")

    if not imageData:
        imagePath = os.path.join(os.path.dirname(json_filepath), data["imagePath"])
        with open(imagePath, "rb") as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode("utf-8")
    img = utils.img_b64_to_arr(imageData)

    label_name_to_value = {"_background_": 0}
    for shape in sorted(data["shapes"], key=lambda x: x["label"]):
        label_name = shape["label"]
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value
    lbl, _ = utils.shapes_to_label(
        img.shape, data["shapes"], label_name_to_value
    )

    # label_names = [None] * (max(label_name_to_value.values()) + 1)
    # for name, value in label_name_to_value.items():
    #     label_names[value] = name

    # lbl_viz = imgviz.label2rgb(
    #     label=lbl, img=imgviz.asgray(img), label_names=label_names, loc="rb"
    # )
    out_label_filepath = osp.join(out_dir, f'{prefix_name}.png')
    # Image.fromarray(img).save(osp.join(out_dir, "img.png"))
    utils.lblsave(out_label_filepath, lbl)
    print('Saved to: %s' % out_label_filepath)


def delete_imageData_from_labelme_json(json_filepath, out_dir):
 
    if not osp.exists(out_dir):
        os.mkdir(out_dir)
       
    if '\\' in json_filepath:
        json_name = json_filepath.split('\\')[-1]
    elif '/' in json_filepath:
        json_name = json_filepath.split('/')[-1]
    else:
        json_name = json_filepath

    prefix_name = json_name.split('.')[0]

    data = json.load(open(json_filepath))
    if not 'imageData' in data:
        return None
    # print(data, type(data))
    data.pop('imageData')
    return data
    


def convert_json_2_mask(args):
    img_filepath    = args.img_filepath
    input_img_dir   = args.input_dir
    output_dir      = args.output_dir

    for root, dirs, files in os.walk(input_img_dir, topdown=True):
        for name in files:
            if '.json' in name:
                json_filepath = os.path.join(root, name)
                img_filepath = json_filepath.replace('.json', '.jpg')
                img_name = name.replace('.json', '.jpg')
                img_name_prefix = img_name.split('.')[0]
                # Opening JSON file
                if os.path.exists(img_filepath):
                    labelme_json_to_dataset_fun(json_filepath, output_dir)
                    # return 

def split_train_val_dataset(args):
    input_img_dir = args.input_dir
    output_img_dir      = args.output_dir

    train_2_val_ratio = 0.25
    input_label_dir = input_img_dir.replace('image', 'label')
    output_label_dir = output_img_dir.replace('image', 'label')
    if output_img_dir and not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)

    if output_label_dir and not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)

    train_images = os.listdir(input_img_dir)
    random.shuffle(train_images)
    val_images = train_images[:int(0.1*len(train_images))]

    for img_name in val_images:
        label_name = img_name.replace('.jpg', '.png') 
        ori_img_filepath = os.path.join(input_img_dir, img_name)
        ori_label_filepath = os.path.join(input_label_dir, label_name)

        val_img_filepath = ori_img_filepath.replace('train', 'val')
        val_label_filepath = ori_label_filepath.replace('train', 'val')

        shutil.move(ori_img_filepath, val_img_filepath)
        shutil.move(ori_label_filepath, val_label_filepath)

def split_train_val_dataset_2nd(args):
    input_img_dir = args.input_dir
    output_dir      = args.output_dir

    val_ratio = 0.25
    input_mask_dir = input_img_dir.replace('img', 'ano')

    output_tr_img_dir = os.path.join(output_dir, 'train', 'image')
    output_tr_mask_dir = os.path.join(output_dir, 'train', 'mask')

    output_val_img_dir = os.path.join(output_dir, 'val', 'image')
    output_val_mask_dir = os.path.join(output_dir, 'val', 'mask')

    if output_tr_img_dir and not os.path.exists(output_tr_img_dir):
        os.makedirs(output_tr_img_dir)
    if output_tr_mask_dir and not os.path.exists(output_tr_mask_dir):
        os.makedirs(output_tr_mask_dir)
    if output_val_img_dir and not os.path.exists(output_val_img_dir):
        os.makedirs(output_val_img_dir)
    if output_val_mask_dir and not os.path.exists(output_val_mask_dir):
        os.makedirs(output_val_mask_dir)

    img_names = os.listdir(input_img_dir)
    random.shuffle(img_names)
    total_num = len(img_names)
    val_num = int(val_ratio*total_num)
    for i, img_name in enumerate(img_names):
        print(f'processing {img_name} {i+1}/{len(img_names)}')
        if not '.jpg' in img_name:
            continue

        if i > val_num:
            output_img_dir = output_tr_img_dir
            output_mask_dir = output_tr_mask_dir
        else:
            output_img_dir = output_val_img_dir
            output_mask_dir = output_val_mask_dir

        ori_img_filepath = os.path.join(input_img_dir, img_name)
        out_img_flepath = os.path.join(output_img_dir, img_name)
        shutil.move(ori_img_filepath, out_img_flepath)

        label_name = img_name.replace('.jpg', '.png') 
        ori_label_filepath = os.path.join(input_mask_dir, label_name)
        if not os.path.exists(ori_label_filepath):
            continue
        out_label_filepath = os.path.join(output_mask_dir, label_name)
        shutil.move(ori_label_filepath, out_label_filepath)


def split_train_val_dataset_3rd(args):
    input_dir = args.inp
    output_dir      = args.oup

    val_ratio = 0.25

    output_tr_img_dir = os.path.join(output_dir, 'train', 'image')
    output_tr_mask_dir = os.path.join(output_dir, 'train', 'json')

    output_val_img_dir = os.path.join(output_dir, 'val', 'image')
    output_val_mask_dir = os.path.join(output_dir, 'val', 'json')

    if output_tr_img_dir and not os.path.exists(output_tr_img_dir):
        os.makedirs(output_tr_img_dir)
    if output_tr_mask_dir and not os.path.exists(output_tr_mask_dir):
        os.makedirs(output_tr_mask_dir)
    if output_val_img_dir and not os.path.exists(output_val_img_dir):
        os.makedirs(output_val_img_dir)
    if output_val_mask_dir and not os.path.exists(output_val_mask_dir):
        os.makedirs(output_val_mask_dir)

    names = os.listdir(input_dir)
    json_names = []
    for name in names:
        if '.json' in name:
            json_names.append(name)
    random.shuffle(json_names)
    total_num = len(json_names)
    val_num = int(val_ratio*total_num)
    cnt = 1
    for i, json_name in enumerate(json_names):
        print(f'processing {json_name} {i+1}/{len(json_names)}')
        img_name = json_name.replace('.json', '.jpg')
        ori_img_filepath = os.path.join(input_dir, img_name)
        if not os.path.exists(ori_img_filepath):
            continue
        
        ori_json_filepath = os.path.join(input_dir, json_name)
        
        if cnt < val_num:
            output_img_dir = output_val_img_dir
            output_mask_dir = output_val_mask_dir
        else:
            output_img_dir = output_tr_img_dir
            output_mask_dir = output_tr_mask_dir

        out_img_flepath = os.path.join(output_img_dir, img_name)
        out_json_filepath = os.path.join(output_mask_dir, json_name)

        shutil.move(ori_img_filepath, out_img_flepath)
        shutil.move(ori_json_filepath, out_json_filepath)
        cnt += 1

def match_mask_for_dataset(args):
    input_img_dir = args.input_dir
    output_dir      = args.output_dir


    output_tr_img_dir = os.path.join(output_dir, 'train', 'image')
    output_tr_mask_dir = os.path.join(output_dir, 'train', 'mask')

    output_val_img_dir = os.path.join(output_dir, 'val', 'image')
    output_val_mask_dir = os.path.join(output_dir, 'val', 'mask')

    if output_tr_img_dir and not os.path.exists(output_tr_img_dir):
        os.makedirs(output_tr_img_dir)
    if output_tr_mask_dir and not os.path.exists(output_tr_mask_dir):
        os.makedirs(output_tr_mask_dir)
    if output_val_img_dir and not os.path.exists(output_val_img_dir):
        os.makedirs(output_val_img_dir)
    if output_val_mask_dir and not os.path.exists(output_val_mask_dir):
        os.makedirs(output_val_mask_dir)

    train_img_dir = '/home/hongrui/project/metro_pro/dataset/1st_5000/train/image'
    val_img_dir = '/home/hongrui/project/metro_pro/dataset/1st_5000/val/image'
    img_names = os.listdir(input_img_dir)
    train_img_names = os.listdir(train_img_dir)
    val_img_names = os.listdir(val_img_dir)

    # random.shuffle(img_names)
    # val_num = int(val_ratio*total_num)

    for i, img_name in enumerate(img_names):
        print(f'processing {img_name} {i+1}/{len(img_names)}')
        if not '.jpg' in img_name:
            continue

        if img_name in train_img_names:
            output_img_dir = output_tr_img_dir
            output_mask_dir = output_tr_mask_dir
        elif img_name in val_img_names:
            output_img_dir = output_val_img_dir
            output_mask_dir = output_val_mask_dir

        ori_img_filepath = os.path.join(input_img_dir, img_name)

        label_name = img_name.replace('.jpg', '.png') 
        out_label_filepath = os.path.join(output_mask_dir, label_name)

        shutil.move(ori_img_filepath, out_label_filepath)


def match_json_for_dataset(args):
    input_json_dir = args.input_dir
    output_dir      = args.output_dir


    output_tr_json_dir = os.path.join(output_dir, 'train', 'json')
    output_test_json_dir = os.path.join(output_dir, 'test', 'json')
    output_val_json_dir = os.path.join(output_dir, 'val', 'json')

    if output_tr_json_dir and not os.path.exists(output_tr_json_dir):
        os.makedirs(output_tr_json_dir)
    if output_test_json_dir and not os.path.exists(output_test_json_dir):
        os.makedirs(output_test_json_dir)
    if output_val_json_dir and not os.path.exists(output_val_json_dir):
        os.makedirs(output_val_json_dir)
    
    train_img_dir = '/home/hongrui/project/metro_pro/dataset/1st_5000/train/image'
    val_img_dir = '/home/hongrui/project/metro_pro/dataset/1st_5000/val/image'
    test_img_dir = '/home/hongrui/project/metro_pro/dataset/1st_5000/test/image'

    train_img_names = os.listdir(train_img_dir)
    val_img_names = os.listdir(val_img_dir)
    test_img_names = os.listdir(test_img_dir)

    json_names = os.listdir(input_json_dir)

    # random.shuffle(img_names)
    # val_num = int(val_ratio*total_num)

    for i, j_name in enumerate(json_names):
        print(f'processing {j_name} {i+1}/{len(json_names)}')
        if not '.json' in j_name:
            continue
        img_name = j_name.replace('.json', '.jpg')
        if img_name in train_img_names:
            output_json_dir = output_tr_json_dir
        elif img_name in val_img_names:
            output_json_dir = output_val_json_dir
        elif img_name in test_img_names:
            output_json_dir = output_test_json_dir

        ori_json_filepath = os.path.join(input_json_dir, j_name)
        out_json_filepath = os.path.join(output_json_dir, j_name)

        shutil.move(ori_json_filepath, out_json_filepath)

def convert_json(args):
    input_dir   = args.input_dir
    output_dir      = args.output_dir

    files = os.listdir(input_dir)
    for i, name in enumerate(files):
        print(f'processing {name} {i+1}/{len(files)}')
        if '.json' in name:
            json_filepath = os.path.join(input_dir, name)
            anno_dict = delete_imageData_from_labelme_json(json_filepath, output_dir)
            if anno_dict is None:
                continue
            new_json_filepath = os.path.join(output_dir, name)

            with open(new_json_filepath, 'w') as f:
                json.dump(anno_dict, f, indent=4)
            
def find_unmatched_image(args):
    input_dir   = args.inp
    output_dir      = args.oup


    dirs = ['train', 'test', 'val']
    for d in dirs:
        input_d_dir = os.path.join(input_dir, d)
        input_img_dir = os.path.join(input_d_dir, 'image')
        input_json_dir = os.path.join(input_d_dir, 'json')

        img_names = os.listdir(input_img_dir)
        json_names = os.listdir(input_json_dir)
        output_img_dir = os.path.join(output_dir, 'unmatched', d, 'image')

        for i, img_name in enumerate(img_names):
            print(f'processing {img_name} {i+1}/{len(img_names)}')
            if not '.jpg' in img_name:
                continue
            j_name = img_name.replace('.jpg', '.json')
            if j_name in json_names:
                continue
            ori_img_filepath = os.path.join(input_img_dir, img_name)
            out_img_filepath = os.path.join(output_img_dir, img_name)
            if not os.path.exists(output_img_dir):
                os.makedirs(output_img_dir)
            shutil.move(ori_img_filepath, out_img_filepath)

            
def find_unmatched_image_and_json(args):
    input_dir   = args.inp
    output_dir      = args.oup

    output_dir = 'todo'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dirs = ['train', 'val']
    for d in dirs:
        input_d_dir = os.path.join(input_dir, d)
        input_img_dir = os.path.join(input_d_dir, 'image')
        input_json_dir = os.path.join(input_d_dir, 'json')
        input_mask_dir = os.path.join(input_d_dir, 'mask')

        ref_mask_names = os.listdir(input_mask_dir)
        img_names = os.listdir(input_img_dir)
        json_names = os.listdir(input_json_dir)

        for i, img_name in enumerate(img_names):
            print(f'processing {img_name} {i+1}/{len(img_names)}')
            if not '.jpg' in img_name:
                continue
            mask_name = img_name.replace('.jpg', '.png')
            if mask_name in ref_mask_names:
                continue
            ori_img_filepath = os.path.join(input_img_dir, img_name)
            out_img_filepath = os.path.join(output_dir, img_name)
            
            shutil.move(ori_img_filepath, out_img_filepath)

        for i, img_name in enumerate(json_names):
            print(f'processing {img_name} {i+1}/{len(img_names)}')
            if not '.json' in img_name:
                continue
            mask_name = img_name.replace('.json', '.png')
            if mask_name in ref_mask_names:
                continue
            ori_json_filepath = os.path.join(input_json_dir, img_name)
            out_img_filepath = os.path.join(output_dir, img_name)
            
            shutil.move(ori_json_filepath, out_img_filepath)
           
# def find_json_for_val(args):
#     input_dir   = args.inp
#     output_dir      = args.oup


#     val_img_dir = os.path.join(input_dir, 'train', 'image')
#     train_json_dir = os.path.join(input_dir, 'val', 'json')

#     img_names = os.listdir(val_img_dir)
#     json_names = os.listdir(train_json_dir)
#     output_img_dir = os.path.join(output_dir, 'unmatched', d, 'image')

#     for i, img_name in enumerate(img_names):
#         print(f'processing {img_name} {i+1}/{len(img_names)}')
#         if not '.jpg' in img_name:
#             continue
#         j_name = img_name.replace('.jpg', '.json')
        
#         ori_img_filepath = os.path.join(input_img_dir, img_name)
#         out_img_filepath = os.path.join(output_img_dir, img_name)
#         if not os.path.exists(output_img_dir):
#             os.makedirs(output_img_dir)
#         shutil.move(ori_img_filepath, out_img_filepath)

def count_dataset():

    train_img_dir = '/home/hongrui/project/metro_pro/dataset/1st_5000/train/image'
    val_img_dir = '/home/hongrui/project/metro_pro/dataset/1st_5000/val/image'
    test_img_dir = '/home/hongrui/project/metro_pro/dataset/1st_5000/test/image'

    train_mask_dir = '/home/hongrui/project/metro_pro/dataset/1st_5000/train/mask'
    val_mask_dir = '/home/hongrui/project/metro_pro/dataset/1st_5000/val/mask'
    test_mask_dir = '/home/hongrui/project/metro_pro/dataset/1st_5000/test/mask'

    train_json_dir = '/home/hongrui/project/metro_pro/dataset/1st_5000/train/json'
    val_json_dir = '/home/hongrui/project/metro_pro/dataset/1st_5000/val/json'
    test_json_dir = '/home/hongrui/project/metro_pro/dataset/1st_5000/test/json'

    train_label_dir = '/home/hongrui/project/metro_pro/dataset/1st_5000/train/label'
    val_label_dir = '/home/hongrui/project/metro_pro/dataset/1st_5000/val/label'
    test_label_dir = '/home/hongrui/project/metro_pro/dataset/1st_5000/test/label'

    todo_dir = '/home/hongrui/project/metro_pro/dataset/1st_5000/todo'


    train_img_names = os.listdir(train_img_dir)
    val_img_names = os.listdir(val_img_dir)
    test_img_names = os.listdir(test_img_dir)
    
    train_json_names = os.listdir(train_json_dir)
    val_json_names = os.listdir(val_json_dir)
    test_json_names = os.listdir(test_json_dir)

    train_mask_names = os.listdir(train_mask_dir)
    val_mask_names = os.listdir(val_mask_dir)
    test_mask_names = os.listdir(test_mask_dir) 
    
    
    train_label_names = os.listdir(train_label_dir)
    val_label_names = os.listdir(val_label_dir)
    test_label_names = os.listdir(test_label_dir)

    todo_names = os.listdir(todo_dir)

    print(f'image train: {len(train_img_names)}, test: {len(test_img_names)}, val: {len(val_img_names)}')
    print(f'json  train: {len(train_json_names)}, test: {len(test_json_names)}, val: {len(val_json_names)}')
    print(f'label train: {len(train_label_names)}, test: {len(test_label_names)}, val: {len(val_label_names)}')
    print(f'mask  train: {len(train_mask_names)}, test: {len(test_mask_names)}, val: {len(val_mask_names)}')

    print(f'todo: {len(todo_names)}')

def convert_json_to_label(args):
    input_dir   = args.inp
    # output_dir      = args.oup


    dirs = ['train', 'test', 'val']
    for d in dirs:
        input_d_dir = os.path.join(input_dir, d)
        input_img_dir = os.path.join(input_d_dir, 'image')
        input_json_dir = os.path.join(input_d_dir, 'json')
        # input_mask_dir = os.path.join(input_d_dir, 'mask')
        output_label_dir = os.path.join(input_d_dir, 'label')
        if not os.path.exists(output_label_dir):
            os.makedirs(output_label_dir)
        # img_names = os.listdir(inpßut_img_dir)
        json_names = os.listdir(input_json_dir)

        for i, json_name in enumerate(json_names):
            print(f'processing {json_name} {i+1}/{len(json_names)}')
            img_name = json_name.replace('.json', '.jpg')
            ori_img_filepath = os.path.join(input_img_dir, img_name)
            if not os.path.exists(ori_img_filepath):
                continue
            label_file_name = json_name.replace('.json', '.png')
            ori_json_filepath = os.path.join(input_json_dir, json_name)
            data = json.load(open(ori_json_filepath))
            img = cv2.imread(ori_img_filepath)
            
            metro_label_name_to_value = {"left_1": 1, "right_1": 2, "left_2": 3, "right_2": 4, 
                                         "left_3": 5, "right_3": 6, "left_4": 7, "right_4": 8,
                                         "left_5": 9, "right_5": 10, "left_6": 11, "right_6": 12}
            for shape in sorted(data["shapes"], key=lambda x: x["label"]):
                label_name = shape["label"]
                if label_name in metro_label_name_to_value:
                    label_value = (metro_label_name_to_value[label_name])*20
                    metro_label_name_to_value[label_name] = label_value
                else:
                    label_value = 250
                    metro_label_name_to_value[label_name] = label_value
            lbl, _ = utils.shapes_to_label(
                img.shape, data["shapes"], metro_label_name_to_value
            )
            # lbl *= 100
            # print(lbl[lbl>100], lbl.max())
            # print('lbl', lbl.shape)
            out_label_filepath = os.path.join(output_label_dir, label_file_name)
            # utils.lblsave(out_label_filepath, lbl)
            cv2.imwrite(out_label_filepath, lbl)
            
            print('Saved to: %s' % out_label_filepath)
            # return
                    
            


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='aligment')
    # parser.add_argument('-if', '--img_filepath', type=str, default='images\\rail_06.jpg')
    parser.add_argument('-if', '--img_filepath', type=str, default=None)

    parser.add_argument('--inp', type=str, default=None)
    parser.add_argument('--oup', type=str, default=None)

    parser.add_argument('-im', '--input_dir', type=str, default=None)

    parser.add_argument('-om', '--output_dir', type=str, default=None)
    parser.add_argument('-th', '--target_height', type=int, default=1440)
    parser.add_argument('-tw', '--target_width', type=int, default=1920)

    args = parser.parse_args()
    # split_train_val_dataset(args)
    # split_train_val_dataset_2nd(args)
    # compose()
    # test()
    # open_alg_fun()
    # convert_json_2_mask(args)
    # convert_json(args) 
    # match_json_for_dataset(args)
    # find_unmatched_image(args)
    # find_unmatched_image_and_json(args)


    # split_train_val_dataset_3rd(args)
    convert_json_to_label(args)

    count_dataset()