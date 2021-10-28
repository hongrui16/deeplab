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
from util import *

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

    dirs = ['train', 'val', 'test']
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

            
def find_unannonated_image_and_json(args):
    input_dir   = args.inp
    # output_dir      = args.oup

    output_dir = input_dir

    dirs = ['train', 'val', 'test']
    for d in dirs:
        input_d_dir = os.path.join(input_dir, d)
        input_img_dir = os.path.join(input_d_dir, 'image')
        input_json_dir = os.path.join(input_d_dir, 'json')
        input_mask_dir = os.path.join(input_d_dir, 'mask')

        output_img_dir = os.path.join(output_dir, 'EmptyJson', d, 'image')
        output_json_dir = os.path.join(output_dir, 'EmptyJson', d, 'json')

        if not os.path.exists(output_img_dir):
            os.makedirs(output_img_dir)        
        if not os.path.exists(output_json_dir):
            os.makedirs(output_json_dir)

        json_names = os.listdir(input_json_dir)

        for i, json_name in enumerate(json_names):
            if not '.json' in json_name:
                continue
            
            ori_json_filepath = os.path.join(input_json_dir, json_name)
            print(f'processing {ori_json_filepath} {i+1}/{len(json_names)}')

            data = json.load(open(ori_json_filepath))
            # print(data)
            if 'shapes' in data and len(data['shapes']) > 0:
                if 'points' in data['shapes'][0] and len(data['shapes'][0]['points']) > 0:
                    pass
            else:
                # print('error')
                # print(data)
                # print('shapes' in data)
                # print(len(data['shapes']) > 0)
                # print('points' in data['shapes'][0])
                # print(len(data['shapes'][0]['points']))
            
                img_name = json_name.replace('.json', '.jpg')

                out_json_filepath = os.path.join(output_json_dir, json_name)
                # print('output_img_dir', output_img_dir)
                print('out_json_filepath', out_json_filepath)
                
                ori_img_filepath = os.path.join(input_img_dir, img_name)
                # print('ori_img_filepath', ori_img_filepath)
                out_img_filepath = os.path.join(output_img_dir, img_name)
                print('out_img_filepath', out_img_filepath)

                shutil.move(ori_json_filepath, out_json_filepath)   
                shutil.move(ori_img_filepath, out_img_filepath)
            print()
      
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

    print(f'image train: {len(train_img_names)},   test: {len(test_img_names)},   val: {len(val_img_names)}')
    print(f'json  train: {len(train_json_names)},  test: {len(test_json_names)},  val: {len(val_json_names)}')
    print(f'label train: {len(train_label_names)}, test: {len(test_label_names)}, val: {len(val_label_names)}')
    print(f'mask  train: {len(train_mask_names)},  test: {len(test_mask_names)},  val: {len(val_mask_names)}')

    print(f'todo: {len(todo_names)}')

                    
def find_images_for_train_and_val(args):
    input_dir   = args.inp
    # output_dir      = args.oup


    dirs = ['train', 'test', 'val']
    ori_img_dir = '/home/hongrui/project/metro_pro/dataset/1st_5000/Part3'
    ori_names = os.listdir(ori_img_dir)

    for d in dirs:
        input_d_dir = os.path.join(input_dir, d)
        img_dir = os.path.join(input_d_dir, 'image')
        img_names = os.listdir(img_dir)

        for i, img_name in enumerate(img_names):
            print(f'processing {img_name} {i+1}/{len(img_names)}')
            if not '.jpg' in img_name:
                continue
            if not img_name in ori_names:
                continue
            ori_img_filepath = os.path.join(ori_img_dir, img_name)
            out_img_filepath = os.path.join(img_dir, img_name)
            
            shutil.copy(ori_img_filepath, out_img_filepath)

def find_GT_for_inference(args):
    input_dir   = args.inp
    
    test_label_dir = '/home/hongrui/project/metro_pro/dataset/1st_5000/test/label'
    

    ori_names = os.listdir(input_dir)

    img_names = []
    for i, ori_name in enumerate(ori_names):
        if '.jpg' in ori_name:
            img_names.append(ori_name)
    for i, img_name in enumerate(img_names):
        print(f'processing {img_name} {i+1}/{len(img_names)}')
        
        ori_gt_name = img_name.replace('.jpg', '.png')
        out_gt_name = f"{img_name.split('.')[0]}_GT.png"
        ori_gt_filepath = os.path.join(test_label_dir, ori_gt_name)
        if os.path.exists(ori_gt_filepath):
            out_gt_filepath = os.path.join(input_dir, out_gt_name)
            print(out_gt_filepath)
            shutil.copy(ori_gt_filepath, out_gt_filepath)
        print()



def relocate_rail_regin_in_images(args):
    # old_test_img_dir = '/home/hongrui/project/metro_pro/dataset/1st_5000/test_old_ori/image'
    # old_test_img_dir = '/comp_robot/hongrui/metro_pro/dataset/1st_5000/val/image'
    # old_test_img_dir = '/comp_robot/hongrui/metro_pro/dataset/1st_5000_2nd_round/test_ori/image'
    old_test_img_dir = '/comp_robot/hongrui/metro_pro/dataset/1st_5000_2nd_round/val/image'

    # old_test_json_dir = '/home/hongrui/project/metro_pro/dataset/1st_5000/test_old_ori/json'
    # old_test_label_dir = '/home/hongrui/project/metro_pro/dataset/1st_5000/test_old_ori/label'
    # old_test_label_dir = '/comp_robot/hongrui/metro_pro/dataset/1st_5000/val/label'
    # old_test_label_dir = '/comp_robot/hongrui/metro_pro/dataset/1st_5000_2nd_round/test_ori/label'
    old_test_label_dir = '/comp_robot/hongrui/metro_pro/dataset/1st_5000_2nd_round/val/label'

    output_img_dir = '/comp_robot/hongrui/metro_pro/dataset/1st_5000_2nd_round/val_for_infer/image'
    # output_json_dir = '/home/hongrui/project/metro_pro/dataset/1st_5000/test/json'
    output_label_dir = '/comp_robot/hongrui/metro_pro/dataset/1st_5000_2nd_round/val_for_infer/label'

    if not os.path.exists(output_img_dir):
        os.makedirs(output_img_dir)        
    # if not os.path.exists(output_json_dir):
    #     os.makedirs(output_json_dir)
    if not os.path.exists(output_label_dir):
        os.makedirs(output_label_dir)
    
    # sorted_test_imgs_filepath = '/home/hongrui/project/metro_pro/dataset/1st_5000/sorted_test_imgs.txt'
    # sorted_test_imgs = read_txt_to_list(sorted_test_imgs_filepath)
    sorted_test_imgs = []
    img_names = os.listdir(old_test_img_dir)
    for img_name in img_names:
        if '.jpg' in img_name:
            sorted_test_imgs.append(img_name)
    cnt = 1
    for i, img_name in enumerate(sorted_test_imgs):
        
        print(f'processing {img_name} {cnt} {i+1}/{len(sorted_test_imgs)}')
        img_prefix = img_name.split('.')[0]
        
        img_filepath = os.path.join(old_test_img_dir, img_name)
        label_filepath = os.path.join(old_test_label_dir, img_prefix + '.png')
        # json_filepath = os.path.join(old_test_json_dir, img_prefix + '.json')
        

        out_img_filepath = os.path.join(output_img_dir, img_name)
        out_label_filepath = os.path.join(output_label_dir, img_prefix + '.png')
        # out_json_filepath = os.path.join(output_json_dir, img_prefix + '.json')

        img = cv2.imread(img_filepath).astype(np.uint8)
        h, w, _ = img.shape
        label = cv2.imread(label_filepath, 0).astype(np.uint8)
        try:
            left_x_pos, right_x_pos = find_bottom_lane_location_in_labels(label)
        except Exception as e:
            continue
            
        # print(f"h {h}, w {w}, left_x_pos {left_x_pos}, right_x_pos {right_x_pos}")
        rail_width = right_x_pos - left_x_pos
        if left_x_pos + rail_width//4 < w//2 < right_x_pos - rail_width//4:
            shutil.copy(img_filepath, out_img_filepath)
            shutil.copy(label_filepath, out_label_filepath)
            # shutil.copy(json_filepath, out_json_filepath)
        else:

            mid_x_pos = (left_x_pos + right_x_pos)//2
            offset = w//2 - mid_x_pos
            abs_offset = int(abs(offset))
            zero_img_block = np.zeros((h,abs_offset,3), dtype=np.uint8)
            zero_label_block = np.zeros((h,abs_offset), dtype=np.uint8)
            if offset < 0:
                new_img = np.concatenate((img[:,abs_offset:], zero_img_block), axis=1)
                new_label = np.concatenate((label[:,abs_offset:], zero_label_block), axis=1)

            else:
                new_img = np.concatenate((zero_img_block, img[:,:w-abs_offset]), axis=1)
                new_label = np.concatenate((zero_label_block, label[:,:w-abs_offset]), axis=1)

            cv2.imwrite(out_img_filepath, new_img)
            cv2.imwrite(out_label_filepath, new_label)
        cnt += 1
        # return

def copy_train_val_json_2nd_round(args):
    ori_json_dir = '/comp_robot/hongrui/metro_pro/dataset/1st_5000_2nd_round/std_json_2nd_round/'
    input_dir = '/comp_robot/hongrui/metro_pro/dataset/1st_5000_2nd_round/'
    dirs = ['train', 'val']
    for d in dirs:
        input_d_dir = os.path.join(input_dir, d)
        input_img_dir = os.path.join(input_d_dir, 'image')
        # input_mask_dir = os.path.join(input_d_dir, 'mask')
        output_json_dir = os.path.join(input_d_dir, 'json')
        if not os.path.exists(output_json_dir):
            os.makedirs(output_json_dir)
        img_names = os.listdir(input_img_dir)

        for i, img_name in enumerate(img_names):
            print(f'processing {img_name} {i+1}/{len(img_names)}')
            json_name = img_name.replace('.jpg', '.json')
            ori_json_filepath = os.path.join(ori_json_dir, json_name)
            if not os.path.exists(ori_json_filepath):
                continue

            out_json_filepath = os.path.join(output_json_dir, json_name)
            shutil.copy(ori_json_filepath, out_json_filepath)


def copy_test_json_2nd_round(args):
    ori_json_dir = '/comp_robot/hongrui/metro_pro/dataset/1st_5000_2nd_round/std_json_2nd_round/'
    ori_img_dir =  '/comp_robot/hongrui/metro_pro/dataset/1st_5000_2nd_round/test_ori/image'
    input_dir =    '/comp_robot/hongrui/metro_pro/dataset/1st_5000_2nd_round/test_ori'

    output_json_dir = os.path.join(input_dir, 'json')
    if not os.path.exists(output_json_dir):
        os.makedirs(output_json_dir)
    img_names = os.listdir(ori_img_dir)

    for i, img_name in enumerate(img_names):
        print(f'processing {img_name} {i+1}/{len(img_names)}')
        json_name = img_name.replace('.jpg', '.json')
        ori_json_filepath = os.path.join(ori_json_dir, json_name)
        if not os.path.exists(ori_json_filepath):
            continue

        out_json_filepath = os.path.join(output_json_dir, json_name)
        shutil.copy(ori_json_filepath, out_json_filepath)

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
    # convert_json_to_label(args)
    # find_images_for_train_and_val(args)

    # find_unannonated_image_and_json(args)
    # find_GT_for_inference(args)


    relocate_rail_regin_in_images(args)
    # count_dataset()
    # copy_train_val_json_2nd_round(args)
    # copy_test_json_2nd_round(args)