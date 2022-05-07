from unicodedata import category
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
import csv
import datetime
import shutil
import json
import skimage.exposure
import base64
import os.path as osp
from labelme import utils
import imgviz
import random
import glob
from pascal import PascalVOC
from zipfile import ZipFile

from pathlib import Path


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

def count_dataset_v2():

    parent_dir = '/comp_robot/hongrui/metro_pro/dataset/twoRail/sorted'
    parent_dir = '/comp_robot/hongrui/metro_pro/dataset/1st_5000_2nd_round/'
    
    to_dirs =  ['test'  ,   'test_ori', 'train',     'val',     'val_ori']
    to_dirs =  ['test',  'train',  'val', ]
    for i, td in enumerate(to_dirs):
        ori_dir = os.path.join(parent_dir, td)
        in_img_dir = os.path.join(parent_dir, td, 'image')
        in_label_dir = os.path.join(parent_dir, td, 'label')

        img_names = os.listdir(in_img_dir)
        label_names = os.listdir(in_label_dir)
        print(f'{td}/image {len(img_names)}')
        print(f'{td}/label {len(label_names)}')

        # print()

                    
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
    # old_test_img_dir = '/home/hongrui/project/metro_pro/dataset/v2_2rails/sorted/test_ori/image'

    # old_test_json_dir = '/home/hongrui/project/metro_pro/dataset/1st_5000/test_old_ori/json'
    # old_test_label_dir = '/home/hongrui/project/metro_pro/dataset/1st_5000/test_old_ori/label'
    # old_test_label_dir = '/comp_robot/hongrui/metro_pro/dataset/1st_5000/val/label'
    # old_test_label_dir = '/comp_robot/hongrui/metro_pro/dataset/1st_5000_2nd_round/test_ori/label'
    # old_test_label_dir = '/comp_robot/hongrui/metro_pro/dataset/1st_5000_2nd_round/val/label'

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



def relocate_rail_regins_in_all_images_once(args):

    parent_dir = '/home/hongrui/project/metro_pro/dataset/v2_2rails/sorted/'

    ori_dirs = ['test_ori', 'val_ori']
    new_dirs = ['test', 'val']
    out_img_dirs = []
    out_label_dirs = []

    for nd in new_dirs:
        output_img_dir = os.path.join(parent_dir, nd, 'image')
        output_label_dir = os.path.join(parent_dir, nd, 'label')
        out_img_dirs.append(output_img_dir)
        out_label_dirs.append(output_label_dir)
        if not os.path.exists(output_img_dir):
            os.makedirs(output_img_dir)        
        
        if not os.path.exists(output_label_dir):
            os.makedirs(output_label_dir)
            
    # sorted_test_imgs_filepath = '/home/hongrui/project/metro_pro/dataset/1st_5000/sorted_test_imgs.txt'
    # sorted_test_imgs = read_txt_to_list(sorted_test_imgs_filepath)
    for id, od in enumerate(ori_dirs):
        ori_img_dir = os.path.join(parent_dir, od, 'image')
        ori_label_dir = os.path.join(parent_dir, od, 'label')
        output_img_dir = out_img_dirs[id]
        output_label_dir = out_label_dirs[id]

        img_names = os.listdir(ori_img_dir)
        for i, img_name in enumerate(img_names):
            if not '.jpg' in img_name:
                continue

            print(f'processing {img_name} {od} {i+1}/{len(img_names)}')
            img_prefix = img_name.split('.')[0]
            
            img_filepath = os.path.join(ori_img_dir, img_name)
            label_filepath = os.path.join(ori_label_dir, img_prefix + '.png')
            
            out_img_filepath = os.path.join(output_img_dir, img_name)
            out_label_filepath = os.path.join(output_label_dir, img_prefix + '.png')
            # out_json_filepath = os.path.join(output_json_dir, img_prefix + '.json')
            # if os.path.exists(out_img_filepath) and os.path.exists(out_label_filepath):
            #     continue
            img = cv2.imread(img_filepath).astype(np.uint8)
            h, w, _ = img.shape
            label = cv2.imread(label_filepath, 0).astype(np.uint8)
            try:
                left_x_pos, right_x_pos = find_bottom_lane_location_in_labels(label)
            except Exception as e:
                continue
            if left_x_pos < 0 and right_x_pos < 0:
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
                # print('w',w, 'mid_x_pos', mid_x_pos, 'offset',offset )
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

def rename_video(args):
    input_dir  = args.input_dir
    img_names = os.listdir(input_dir)
    img_names.sort()
    print('img_names', img_names)

    for i, v in enumerate(img_names):
        v_filepath = os.path.join(input_dir, v)
        new_name = str(i)+'.avi'
        new_v_filepath = os.path.join(input_dir, new_name)

        os.rename(v_filepath, new_v_filepath)





def check_annotation(args):
    ori_json_dir = '/comp_robot/hongrui/metro_pro/dataset/1st_5000_2nd_round/std_json_2nd_round/'
    input_dir =  '/comp_robot/hongrui/metro_pro/dataset/1st_5000_2nd_round/'

    output_dir = os.path.join('temp')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dirs = ['train', 'val', 'test']
    img_filepaths = []
    for d in dirs:
        input_d_dir = os.path.join(input_dir, d)
        input_img_dir = os.path.join(input_d_dir, 'image')
        img_names = os.listdir(input_img_dir)
        for i, img_name in enumerate(img_names):
            img_filepath = os.path.join(input_img_dir, img_name)
            img_filepaths.append(img_filepath)

    random.shuffle(img_filepaths)
    img_filepaths = img_filepaths[:500]

    for i, img_filepath in enumerate(img_filepaths):
        img_name = img_filepath.split('/')[-1]
        print(f'processing {img_name} {i+1}/{len(img_filepaths)}')
        json_name = img_name.replace('.jpg', '.json')
        ori_json_filepath = os.path.join(ori_json_dir, json_name)
        if os.path.exists(ori_json_filepath):
            out_json_filepath = os.path.join(output_dir, json_name)
            shutil.copy(ori_json_filepath, out_json_filepath)
            out_img_filepath = os.path.join(output_dir, img_name)
            shutil.copy(img_filepath, out_img_filepath)

def convet_to_numerical_index(label):
    if label.any() > 0:
        mask_min_nonzero = label[label>0].min()
        label = label//mask_min_nonzero
    return label

def colorize_anno(args):
    input_dir =  '/comp_robot/hongrui/metro_pro/dataset/1st_5000_2nd_round/'

    output_dir = os.path.join('temp')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dirs = ['train', 'val', 'test']
    img_filepaths = []
    for d in dirs:
        input_d_dir = os.path.join(input_dir, d)
        input_img_dir = os.path.join(input_d_dir, 'image')
        img_names = os.listdir(input_img_dir)
        for i, img_name in enumerate(img_names):
            img_filepath = os.path.join(input_img_dir, img_name)
            img_filepaths.append(img_filepath)

    random.shuffle(img_filepaths)
    img_filepaths = img_filepaths[:400]

    for i, img_filepath in enumerate(img_filepaths):
        img_name = img_filepath.split('/')[-1]
        print(f'processing {img_name} {i+1}/{len(img_filepaths)}')
        label_filepath = img_filepath.replace('.jpg', '.png').replace('image', 'label')
        if not os.path.exists(label_filepath):
            continue
        img = cv2.imread(img_filepath)
        label = cv2.imread(label_filepath, 0)
        ori_label_col = cv2.imread(label_filepath)

        label = convet_to_numerical_index(label)
        label_bgr = colorize_mask_to_bgr(label)
        cat_img_label = np.concatenate((img, ori_label_col), axis=1)

        composed = 0.7*img + 0.3*label_bgr
        cat_label_bgr_composed = np.concatenate((label_bgr, composed), axis=1)

        out_img = np.concatenate((cat_img_label, cat_label_bgr_composed), axis=0)
        out_img_filepath = os.path.join(output_dir, img_name)
        cv2.imwrite(out_img_filepath, out_img)



def find_all_images(args):
    input_dir =  '/comp_robot/hongrui/metro_pro/dataset/1st_5000_2nd_round/'

    output_dir = '/comp_robot/hongrui/metro_pro/dataset/v1_v2_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # dirs = ['train', 'val', 'test_old_ori']
    dirs = ['test_ori']

    for d in dirs:
        input_d_dir = os.path.join(input_dir, d)
        input_img_dir = os.path.join(input_d_dir, 'image')
        img_names = os.listdir(input_img_dir)
        for i, img_name in enumerate(img_names):
            if not '.jpg' in img_name:
                continue
            img_filepath = os.path.join(input_img_dir, img_name)
            out_img_filepath = os.path.join(output_dir, img_name)
            shutil.copy(img_filepath, out_img_filepath)



def find_jsons(args):

    input_img_dir = '/comp_robot/hongrui/metro_pro/dataset/v1_v2_images'
    v1_json_dir = '/comp_robot/hongrui/metro_pro/dataset/v1_json'
    v2_json_dir = '/comp_robot/hongrui/metro_pro/dataset/v2_json'

    if not os.path.exists(v1_json_dir):
        os.makedirs(v1_json_dir)
        
    if not os.path.exists(v2_json_dir):
        os.makedirs(v2_json_dir)

    in_dirs = ['/comp_robot/hongrui/metro_pro/dataset/1st_5000/std_json',
        '/comp_robot/hongrui/metro_pro/dataset/1st_5000_2nd_round/std_json_2nd_round']

    out_dirs = [v1_json_dir, v2_json_dir]
    img_names = os.listdir(input_img_dir)
    for i, d in enumerate(in_dirs):
        json_names = os.listdir(d)
        output_dir = out_dirs[i]
        for i, json_name in enumerate(json_names):
            if not '.json' in json_name:
                continue
            img_name = json_name.replace('.json', '.jpg')
            if not img_name in img_names:
                continue
            json_filepath = os.path.join(d, json_name)
            
            out_json_filepath = os.path.join(output_dir, json_name)
            shutil.copy(json_filepath, out_json_filepath)

def split_rails2_images(args):

    csv_dir = '/home/hongrui/project/metro_pro/deeplab/ious'
    ori_img_dir = '/comp_robot/hongrui/metro_pro/dataset/v1_v2_images'
    out_dir = '/comp_robot/hongrui/metro_pro/dataset/v1_v2_images_sorted'
    json_dir = '/comp_robot/hongrui/metro_pro/dataset/v2_json'

    csv_files = os.listdir(csv_dir)
    two_rails_img_names = []
    for csv_file in csv_files:
        if not '.csv' in csv_file:
            continue
        csv_filepath = os.path.join(csv_dir, csv_file)
        with open(csv_filepath) as csv_file:
            readCSV = csv.reader(csv_file)
            for row in readCSV:
                img_name = row[0]   
                if not '.jpg' in img_name:
                    continue
                n_rails = int(row[1])
                if not n_rails <= 2:
                    continue
                two_rails_img_names.append(img_name)
    n_imgs = len(two_rails_img_names)
    print(n_imgs)
    random.shuffle(two_rails_img_names)
    for i, img_name in enumerate(two_rails_img_names):
        
        img_filepath = os.path.join(ori_img_dir, img_name) 
        json_name = img_name.replace('.jpg', '.json')
        json_filepath = os.path.join(json_dir, json_name) 
        if i < 0.33*n_imgs:
            outdir_name = 'two_rails_1'
        elif 0.33*n_imgs <= i <= 0.66*n_imgs:
            outdir_name = 'two_rails_2'
        else:
            outdir_name = 'two_rails_3'
        output_dir = os.path.join(out_dir, outdir_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        out_img_filepath = os.path.join(output_dir, img_name)
        out_json_filepath = os.path.join(output_dir, json_name)
        shutil.copy(img_filepath, out_img_filepath)
        shutil.copy(json_filepath, out_json_filepath)
        print(f'processing {img_name} to {outdir_name}, {i}/{n_imgs}')


def del_unclear_images(args):
    parent_dir = '/comp_robot/hongrui/metro_pro/dataset/twoRail/sorted'
    # to_dirs =  ['test'  ,   'test_ori', 'train',     'val',     'val_ori']
    # del_dirs = ['test_del', 'test_del', 'train_del', 'val_del', 'val_del']
    to_dirs =  ['test'  ,   'test_ori']
    del_dirs = ['test_del', 'test_del']
    for i, td in enumerate(to_dirs):
        ori_dir = os.path.join(parent_dir, td)
        del_dir = os.path.join(parent_dir, del_dirs[i])
        in_img_dir = os.path.join(parent_dir, td, 'image')
        in_label_dir = os.path.join(parent_dir, td, 'label')

        output_dir = os.path.join(parent_dir, td, 'del')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # del_img_names = os.listdir(del_dir)
        del_img_names = ['camera_000086.jpg', 'camera_000087.jpg', 'camera_000090.jpg', 'camera_000093.jpg']
        for j, del_img_name in enumerate(del_img_names):
            
            del_label_name = del_img_name.replace('.jpg', '.png')
            ori_img_filepath = os.path.join(in_img_dir, del_img_name)
            ori_label_filepath = os.path.join(in_label_dir, del_label_name)
            
            out_img_filepath = os.path.join(output_dir, del_img_name)
            out_label_filepath = os.path.join(output_dir, del_label_name)
            print(f'processing {td} {out_label_filepath}, {j}/{len(del_img_names)}')
            shutil.move(ori_img_filepath, out_img_filepath)
            shutil.move(ori_label_filepath, out_label_filepath)
        # print()

def create_imgfilepath_txt(args):
    parent_dir = '/comp_robot/hongrui/metro_pro/dataset/twoRail/sorted'
    source_dir = '/comp_robot/hongrui/metro_pro/dataset/1st_5000_2nd_round/'
    dirs =  ['val',   'test', 'train']

    for i, td in enumerate(dirs):
        in_img_dir = os.path.join(parent_dir, td, 'image')
        tar_label_dir = os.path.join(source_dir, td, 'label')
        tar_img_dir = os.path.join(source_dir, td, 'image')
        img_names = os.listdir(in_img_dir)
        txt_filepath = os.path.join(source_dir, f'{td}.txt')
        if os.path.exists(txt_filepath):
            os.remove(txt_filepath)
        random.shuffle(img_names)
        img_filepaths_list = []
        for j, img_name in enumerate(img_names):
            label_name = img_name.replace('.jpg', '.png')
            tar_img_filepath = os.path.join(tar_img_dir, img_name)
            tar_label_filepath = os.path.join(tar_label_dir, label_name)
            print(f'processing {td} {tar_img_filepath}, {j}/{len(img_names)}')
            if os.path.exists(tar_label_filepath) and os.path.exists(tar_img_filepath):
                img_filepaths_list.append(tar_img_filepath)
        
        write_list_to_txt(txt_filepath, img_filepaths_list)
        
        lists = read_txt_to_list(txt_filepath)
        print(f'---------------------------{lists[-1]}')


def create_imgfilepath_txt_two(args):
    output_dir = '/comp_robot/hongrui/metro_pro/dataset/compose_v2v3/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    parent_dir = '/comp_robot/hongrui/metro_pro/dataset/twoRail/sorted'
    source_dir = '/comp_robot/hongrui/metro_pro/dataset/1st_5000_2nd_round/'
    dirs =  ['val',   'test', 'train']

    
    for i, td in enumerate(dirs):
        in_img_dir = os.path.join(parent_dir, td, 'image')
        img_names = os.listdir(in_img_dir)
        txt_filepath = os.path.join(output_dir, f'{td}.txt')
        # if os.path.exists(txt_filepath):
        #     os.remove(txt_filepath)
        random.shuffle(img_names)
        img_filepaths_list = []
        for j, img_name in enumerate(img_names):
            tar_img_filepath = os.path.join(in_img_dir, img_name)
            img_filepaths_list.append(tar_img_filepath)

        write_list_to_txt(txt_filepath, img_filepaths_list)

    for i, td in enumerate(dirs):
        in_img_dir = os.path.join(parent_dir, td, 'image')
        tar_label_dir = os.path.join(source_dir, td, 'label')
        tar_img_dir = os.path.join(source_dir, td, 'image')
        delete_dir = os.path.join(parent_dir, td, 'del')
        del_img_names = os.listdir(in_img_dir)
        delete_names = os.listdir(delete_dir)
        del_img_names += delete_names
        img_names = os.listdir(tar_img_dir)

        txt_filepath = os.path.join(output_dir, f'{td}.txt')
        # if os.path.exists(txt_filepath):
        #     os.remove(txt_filepath)
        random.shuffle(img_names)
        img_filepaths_list = []
        for j, img_name in enumerate(img_names):
            if img_name in del_img_names:
                continue
            label_name = img_name.replace('.jpg', '.png')
            tar_img_filepath = os.path.join(tar_img_dir, img_name)
            tar_label_filepath = os.path.join(tar_label_dir, label_name)
            print(f'processing {td} {tar_img_filepath}, {j}/{len(img_names)}')
            if os.path.exists(tar_label_filepath) and os.path.exists(tar_img_filepath):
                img_filepaths_list.append(tar_img_filepath)
        
        write_list_to_txt(txt_filepath, img_filepaths_list)
        
        lists = read_txt_to_list(txt_filepath)
        print(f'---------------------------{lists[-1]}')

def find_matched_images_json(args):

    input_img_dir   = args.input_dir
    output_dir      = args.output_dir
    
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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
                    print(f'move {img_filepath} -> {out_img_filepath}')



def find_images_json(args):

    # input_img_dir   = '/home/hongrui/project/metro_pro/dataset/pot/pot_20220222'
    # output_dir      = '/home/hongrui/project/metro_pro/dataset/pot/20220222'
    
    input_img_dir   = '/home/hongrui/project/metro_pro/dataset/pot/pot_20220108'
    output_dir      = '/home/hongrui/project/metro_pro/dataset/pot/20220108'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, dirs, files in os.walk(input_img_dir, topdown=True):
        for name in files:
            if '.json' in name:
                json_filepath = os.path.join(root, name)
                out_json_filepath = os.path.join(output_dir, name)
                shutil.move(json_filepath, out_json_filepath)
                print(f'move {json_filepath} -> {out_json_filepath}')
            elif '.jpg' in name:
                img_filepath = os.path.join(root, name)
                out_img_filepath = os.path.join(output_dir, name)
                shutil.move(img_filepath, out_img_filepath)
                print(f'move {img_filepath} -> {out_img_filepath}')
                    
def split_train_val_test_filelist():
    # input_dir = '/home/hongrui/project/metro_pro/dataset/pot_20220108_cut'
    # output_dir = '/home/hongrui/project/metro_pro/dataset/'

    # input_dir = '/home/hongrui/project/metro_pro/dataset/pot/pot_20220108_obvious_defect_0/data'
    # output_dir = '/home/hongrui/project/metro_pro/dataset/pot/pot_20220108_obvious_defect_0/'
    
    # input_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_2/data'
    # output_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_2/'

    input_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_0328/data'
    output_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_0328/'


    # ref_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_1/data'
    # ref_img_names = os.listdir(ref_dir)
    ref_img_names = []
    
    dirs =  ['val',   'test', 'train']
    label_files = glob.glob(osp.join(input_dir, "*.png"))
    total_num = len(label_files)
    
    
    
    test_filepaths = []
    train_filepaths = []
    val_filepaths = []
    
    test_list_txt = os.path.join(output_dir, 'test.txt')
    val_list_txt = os.path.join(output_dir, 'val.txt')
    train_list_txt = os.path.join(output_dir, 'train.txt')
    
    test_filepaths = read_txt_to_list(test_list_txt)
    val_filepaths = read_txt_to_list(val_list_txt)
    train_filepaths = read_txt_to_list(train_list_txt)
    
    test_filepaths = list(set(test_filepaths))
    val_filepaths = list(set(val_filepaths))
    train_filepaths = list(set(train_filepaths))
    
    os.remove(test_list_txt)
    os.remove(val_list_txt)
    os.remove(train_list_txt)
    
    random.shuffle(label_files)
    random.shuffle(label_files)
    for i, label_filepath in enumerate(label_files):
        label_name = label_filepath.split('/')[-1]
        if label_name in ref_img_names:
            continue
        label_filepath = label_filepath.replace('.png', '.jpg')
        if i < 0.15*total_num:
            test_filepaths.append(label_filepath)
        elif 0.15*total_num <= i < 0.3*total_num:
            val_filepaths.append(label_filepath)
        else:
            train_filepaths.append(label_filepath)

    # test_filepaths *=3
    # val_filepaths *=3
    # train_filepaths *=2

    write_list_to_txt(test_list_txt, test_filepaths)
    write_list_to_txt(val_list_txt, val_filepaths)
    write_list_to_txt(train_list_txt, train_filepaths)

    total_num = 0
    total_num += len(test_filepaths)
    total_num += len(val_filepaths)
    total_num += len(train_filepaths)
    
    print('total_num', total_num)


def sort_GC10_DET():
    par_dir = '/comp_robot/hongrui/pot_pro/GC10-DET'
    cls_names = ['1_chongkong', '2_hanfeng', '3_yueyawan', '4_shuiban', '5_youban', '6_siban', '7_yiwu', '8_yahen', '9_zhehen', '10_yaozhe']

    dirs = ['train', 'test', 'val']
    class_names = []
    for d in dirs:
        input_dir = os.path.join(par_dir, d, 'xml')
        output_dir = os.path.join(par_dir, d, 'label')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        ds = Path(input_dir)
        xml_files = ds.glob("*.xml")
        xmls = os.listdir(input_dir)
        i = 1
        # for file in xml_files:
        for fl in xmls:
            file = os.path.join(input_dir, fl)
            print(f'processing {d}, {i}/{len(os.listdir(input_dir))}, {file}')
            ann = PascalVOC.from_xml(file)
            label_filepath = file.replace('xml/', 'label/').replace('.xml', '.png')
            img_filepath = file.replace('xml/', 'image/').replace('.xml', '.jpg')
            # print('img_filepath', img_filepath)
            img = cv2.imread(img_filepath)
            mask = np.zeros(img.shape[:2])
            for obj in ann.objects:
                xmin, ymin = (obj.bndbox.xmin, obj.bndbox.ymin)
                xmax, ymax = (obj.bndbox.xmax, obj.bndbox.ymax)
                label_name = obj.name
                if not label_name in cls_names:
                    continue
                label_index = cls_names.index(label_name) + 1
                mask[ymin:ymax, xmin:xmax] = label_index
                # if not label_name in class_names:
                #     class_names.append(label_name)
            cv2.imwrite(label_filepath, mask.astype(np.uint8))
            i += 1
    # print(class_names)


def calculate_mean_value():
    input_dir = '/comp_robot/hongrui/pot_pro/severstal-steel-defect-detection/train_images/'
    
    img_names = os.listdir(input_dir)
    random.shuffle(img_names)
    
    mean_value = 0
    for i, img_name in enumerate(img_names):
        print(f'{i}/500')
        img_filepath = os.path.join(input_dir, img_name)
        img = cv2.imread(img_filepath, 0)
        mean_v = img.mean()
        mean_value += mean_v
        if i > 499:
            break
    
    print(mean_value/500)

def read_imgfilepath():
    input_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_1/'
    splits = ['train', 'test', 'val']
    for sp in splits:
        txt_filepath = os.path.join(input_dir, f'{sp}.txt')
        img_filepaths = read_txt_to_list(txt_filepath)
        for img_filepath in img_filepaths:
            if not os.path.exists(img_filepath):
                print(f'not exists {img_filepath}')
                
            img = cv2.imread(img_filepath)
            label_filepath = img_filepath.replace('.jpg', '.png')
            label = cv2.imread(label_filepath)
            if not isinstance(img, np.ndarray):
                print('img', img_filepath)
            if not isinstance(label, np.ndarray):
                print('label', label_filepath)
                
def verify_exist():
    input_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_2/'
    txt_files = ['train.txt', 'val.txt', 'test.txt']
    
    for txt in txt_files:
        txt_filepath = os.path.join(input_dir, txt)
        img_filepaths = read_txt_to_list(txt_filepath)
        for img_filepath in img_filepaths:
            label_filepath = img_filepath.replace('.jpg', '.png')
            if not os.path.exists(img_filepath) or not os.path.exists(label_filepath):
                print(txt, img_filepath)
                print(txt, label_filepath)
            # print('..')

def del_all_checkpoint_pth_tar():
    input_dir = '/home/hongrui/project/metro_pro/deeplab'
    tar = 'checkpoint.pth.tar'
    for root, dirs, files in os.walk(input_dir, topdown=True):
        for name in files:
            if tar == name:
                tar_filepath = os.path.join(root, name)
                os.remove(tar_filepath)
                print(f'remove {tar_filepath}')

def calculate_pos_vs_neg():
    input_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_v3'
    # input_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_0328'
    txt_files = [
                'val.txt',    
                'test.txt',
                ]
    
    if input_dir.endswith('/'):
        input_dir = input_dir[:-1]
    
        
    label_dict = {
        'lasi_heavy': 11, 'lasi_medium':12, 'lasi_slight':13,
        'gengshang_heavy':21, 'gengshang_medium':22, 'gengshang_slight':23,  
        'gengshi_heavy':31, 'gengshi_medium':32, 'gengshi_slight':33,
        'shayan_heavy':41, 'shayan_medium':42, 'shayan_slight':43,
        'huahen_heavy':51, 'huahen_medium':52, 'huahen_slight':53,
        'zhoubian_heavy':61, 'zhoubian_medium':62, 'zhoubian_slight':63,
        'bowen_heavy':71, 'bowen_medium':72, 'bowen_slight':73,
        'youwu_heavy':81, 'youwu_medium':82, 'youwu_slight':83,
        }
    
    label_names = label_dict.keys()
    # print('label_names', label_names)
    num_classes = len(label_names)//3 - 3
    cate_names = []
    for i, name in enumerate(label_names):
        if i % 3 == 0:
            cat_id = i // 3
            if cat_id >= num_classes:
                continue
            cate_names.append(name.split('_')[0])
    
    for txt_file in txt_files:
        txt_filepath = os.path.join(input_dir, txt_file)
        split = txt_file.split('.')[0]

        
        img_filepaths = read_txt_to_list(txt_filepath)
        img_filepaths = list(set(img_filepaths))
        pos, neg = 0, 0
        for i, img_filepath in enumerate(img_filepaths):
            label_filepath = img_filepath.replace('.jpg', '.png')
            label = cv2.imread(label_filepath, 0)
            # print(label_filepath, np.unique(label))
            ids = np.unique(label)
            ids = ids[(ids > 0) * (ids < 61)]
            ids = ids.tolist()
            flag = False
            for c in ids:
                if not c % 10 == 3:
                    pos += 1
                    flag = True
                    break
            if flag:
                continue
            else:
                neg += 1
        print(split, f'pos:neg = {pos}:{neg}')
            
def calculate_pos_vs_neg_v2():
    input_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_0328_block'
    # input_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_0328'
    txt_files = [
                'val.txt',    
                'test.txt',
                ]
    # val pos:neg = 149:127
    # test pos:neg = 140:98

    if input_dir.endswith('/'):
        input_dir = input_dir[:-1]
    
        
    label_dict = {
        'lasi_heavy': 11, 'lasi_medium':12, 'lasi_slight':13,
        'gengshang_heavy':21, 'gengshang_medium':22, 'gengshang_slight':23,  
        'gengshi_heavy':31, 'gengshi_medium':32, 'gengshi_slight':33,
        'shayan_heavy':41, 'shayan_medium':42, 'shayan_slight':43,
        'huahen_heavy':51, 'huahen_medium':52, 'huahen_slight':53,
        'zhoubian_heavy':61, 'zhoubian_medium':62, 'zhoubian_slight':63,
        'bowen_heavy':71, 'bowen_medium':72, 'bowen_slight':73,
        'youwu_heavy':81, 'youwu_medium':82, 'youwu_slight':83,
        }

    for txt_file in txt_files:
        txt_filepath = os.path.join(input_dir, txt_file)
        split = txt_file.split('.')[0]

        
        img_filepaths = read_txt_to_list(txt_filepath)
        img_filepaths = list(set(img_filepaths))
        pos, neg = 0, 0
        for i, img_filepath in enumerate(img_filepaths):
            img_name = img_filepath.split('/')[-1]
            tag = img_name.split('_')[-3]
            if tag == 'P':
                pos += 1
            else:
                neg += 1
        print(split, f'pos:neg = {pos}:{neg}')


def calculate_pixels_and_plot_dist():
    # input_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_2'
    # input_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_v3'
    input_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_0328_block'
    
    
    
    # txt_files = [
    #             'train.txt', 
    #             'val.txt',    
    #             'test.txt',
    #             ]

    txt_files = [
                'val.txt',    
                'test.txt',
                ]
    
    if input_dir.endswith('/'):
        input_dir = input_dir[:-1]
    
    dir_name = input_dir.split('/')[-1]
    csv_file = os.path.join(input_dir, f'{dir_name}_dist.csv')
    if os.path.exists(csv_file):
        os.remove(csv_file)
    
    plot_file = os.path.join(input_dir, f'{dir_name}_dist.jpg')
    if os.path.exists(plot_file):
        os.remove(plot_file)
        
    label_dict = {
        'lasi_heavy': 11,       'lasi_medium':12,       'lasi_slight':13,
        'gengshang_heavy':21,   'gengshang_medium':22,  'gengshang_slight':23,  
        'gengshi_heavy':31,     'gengshi_medium':32,    'gengshi_slight':33,
        'shayan_heavy':41,      'shayan_medium':42,     'shayan_slight':43,
        'huahen_heavy':51,      'huahen_medium':52,     'huahen_slight':53,
        'zhoubian_heavy':61,    'zhoubian_medium':62,   'zhoubian_slight':63,
        'bowen_heavy':71,       'bowen_medium':72,      'bowen_slight':73,
        'youwu_heavy':81,       'youwu_medium':82,      'youwu_slight':83,
        }
    
    label_names = label_dict.keys()
    # print('label_names', label_names)
    num_classes = len(label_names)//3 - 3
    cate_names = []
    for i, name in enumerate(label_names):
        if i % 3 == 0:
            cat_id = i // 3
            if cat_id >= num_classes:
                continue
            cate_names.append(name.split('_')[0])
    
    write_list_to_row_in_csv(csv_file, ['', 'heavy', 'medium', 'slight'])
    
    all_counter = []
    splits = []
    for txt_file in txt_files:
        txt_filepath = os.path.join(input_dir, txt_file)
        split = txt_file.split('.')[0]
        splits.append(split)
        write_list_to_row_in_csv(csv_file, [f'{split}'])
        
        img_filepaths = read_txt_to_list(txt_filepath)
        img_filepaths = list(set(img_filepaths))
        counter = np.zeros((num_classes, 3)).astype(np.float64)
        bk_num = 0
        total_num = 0
        for i, img_filepath in enumerate(img_filepaths):
            label_filepath = img_filepath.replace('.jpg', '.png')
            label = cv2.imread(label_filepath, 0)
            # print(label_filepath, np.unique(label))
            bk_num += len(label[label==0])
            total_num += len(label[label<255])
            for j, label_name in enumerate(label_names):
                label_index = label_dict[label_name]
                num = len(label[label==label_index])
                row = label_index // 10 - 1
                col = label_index % 10 - 1
                if row >= num_classes:
                    continue
                counter[row, col] += num
            # print(label_name, row, col, num)
        counter /= total_num
        write_list_to_row_in_csv(csv_file, ['bk', f'{round(bk_num/total_num, 4)}'])
        for i, category in enumerate(cate_names):
            write_list_to_row_in_csv(csv_file, [f'{category}'] + counter[i].tolist())
        write_list_to_row_in_csv(csv_file, ['total', f'{total_num}'])
        
        write_list_to_row_in_csv(csv_file, [''])
        all_counter.append(counter)
    
    # set width of bar
    barWidth = 0.25
    # fig = plt.subplots(figsize =(12, 8))

    # Set position of bar on X axis
    br1 = np.arange(num_classes)
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    ax = plt.subplot(1, 2, 1)
    
    all_counter = np.array(all_counter)
    # # print('all_counter', all_counter)
    all_counter *= 100 ### show percentage
    # print('all_counter', all_counter)
    
    # Make the plot
    plt.grid(True, color = "grey", linewidth = "1.4", linestyle = "-.")
    plt.bar(br1, all_counter[0][:,:2].sum(axis = 1), color ='r', width = barWidth,
            edgecolor ='grey', label = f'{splits[0]}')
    plt.bar(br2, all_counter[1][:,:2].sum(axis = 1), color ='g', width = barWidth,
            edgecolor ='grey', label = f'{splits[1]}')
    # plt.bar(br3, all_counter[2][:,:2].sum(axis = 1), color ='b', width = barWidth,
    #         edgecolor ='grey', label = f'{splits[2]}')
    
    # Adding Xticks
    plt.xlabel('heavy & medium', fontweight ='bold', fontsize = 12)
    plt.ylabel('Percentage (%)', fontweight ='bold', fontsize = 12)
    plt.xticks([r + barWidth for r in range(num_classes)],
            cate_names)
    plt.legend()

    ax = plt.subplot(1, 2, 2)
    # Make the plot
    plt.grid(True, color = "grey", linewidth = "1.4", linestyle = "-.")
    plt.bar(br1, all_counter[0][:,2:].sum(axis = 1), color ='r', width = barWidth,
            edgecolor ='grey', label = f'{splits[0]}')
    plt.bar(br2, all_counter[1][:,2:].sum(axis = 1), color ='g', width = barWidth,
            edgecolor ='grey', label = f'{splits[1]}')
    # plt.bar(br3, all_counter[2][:,2:].sum(axis = 1), color ='b', width = barWidth,
    #         edgecolor ='grey', label = f'{splits[2]}')
    
    # Adding Xticks
    plt.xlabel('slight', fontweight ='bold', fontsize = 12)
    plt.ylabel('Percentage (%)', fontweight ='bold', fontsize = 12)
    plt.xticks([r + barWidth for r in range(num_classes)],
            cate_names)
    plt.legend()
    
    plt.suptitle('Positive Pixel Percentate Calculated respectively', y = 0.92, verticalalignment = 'center', fontweight ='bold', fontsize = 15)
    plt.show()
    figure = plt.gcf()  
    figure.set_size_inches(20, 10)
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
def cut_block():
    # txt_files = [
    #         # '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_2/train.txt', 
    #             '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_2/val.txt',    
    #         '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_2/test.txt']
    
    # out_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_2_block/'
    # out_data_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_2_block/data'
    
    txt_files = [
            # '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_2/train.txt', 
                '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_v3/val.txt',    
            '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_v3/test.txt']
    
    out_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_v3_block/'
    out_data_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_v3_block/data'
    
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)
        
    # ignore_index = 255
    size = 480
    for txt_file in txt_files:
        split = txt_file.split('/')[-1].split('.')[0]
        new_txt_filepath = os.path.join(out_dir, f'{split}.txt')
        new_filepath_list = []
        img_filepaths = read_txt_to_list(txt_file)
        img_filepaths = list(set(img_filepaths))
        for i, img_filepath in enumerate(img_filepaths):
            ori_img_name = img_filepath.split('/')[-1]
            
            print(f'processing {split}, {i}/{len(img_filepaths)}, {ori_img_name}')
            label_filepath = img_filepath.replace('.jpg', '.png')
            label = cv2.imread(label_filepath, 0)
            img = cv2.imread(img_filepath)
            mask = label.copy()
            mask[mask==255] = 0
            if not mask.any() > 0:
                continue
            connectivity = 4  
        
            # Perform the operation
            output = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)
            # Get the results

            # # The first cell is the number of labels
            # num_labels = output[0]
            # print('num_labels', num_labels)

            # # The second cell is the label matrix
            # labels = output[1]
            # print('labels', labels)

            # # The third cell is the stat matrix, (xmin, ymin, width, height, area)
            stats = output[2]
            # print('stats', stats)

            # # The fourth cell is the centroid matrix, (cx, cy)
            centroids = output[3]
            # print('centroids', centroids)
            if len(stats) == 1:
                continue
            stats = stats[1:]
            centroids = centroids[1:]
            # img = Image.fromarray(img)
            # mask = Image.fromarray(label)
            h, w, _ = img.shape
            n_cen = len(centroids)
            idx_list = [i for i in range(n_cen)]
            if n_cen > 5:
                random.shuffle(idx_list)
                random.shuffle(idx_list)
                idx_list = idx_list[:5]
            
            for j, idx in enumerate(idx_list):
                x0, y0, width, height, _ = stats[idx].astype(np.uint32)
                cx, cy = centroids[idx].astype(np.uint32)
                
                if width >= size :
                    x_offset = size//2
                else:
                    x_offset = (size - width)//2
                
                if height >= size :
                    y_offset = size//2
                else:
                    y_offset = (size - height)//2
                
                xmin = x0 - x_offset if x0 - x_offset > 0 else 0
                ymin = y0 - y_offset if y0 - y_offset > 0 else 0
                xmax = x0 + width + x_offset if x0 + width + x_offset < w else w
                ymax = y0 + height + y_offset if y0 + height + y_offset < h else h
                
                img_block = img[ymin:ymax, xmin:xmax]
                label_block = label[ymin:ymax, xmin:xmax]

                h_block, w_block = label_block.shape
                row = round(h_block/size)
                col = round(w_block/size)
                
                row = 1 if row == 0 else row
                col = 1 if col == 0 else col
                base_h = int(h_block/row)
                base_w = int(w_block/col)
                for r in range(row):
                    for c in range(col):
                        temp_img = img[ymin + r*base_h: ymin + (r+1)*base_h, xmin + c*base_w: xmin + (c+1)*base_w]
                        temp_label = label[ymin + r*base_h: ymin + (r+1)*base_h, xmin + c*base_w: xmin + (c+1)*base_w]
                        new_img_name = ori_img_name.replace('.jpg', f'_{j}_{r}_{c}.jpg')
                        new_label_name = ori_img_name.replace('.jpg', f'_{j}_{r}_{c}.png')
                        out_img_filepath = os.path.join(out_data_dir, new_img_name)
                        out_label_filepath = os.path.join(out_data_dir, new_label_name)
                        cv2.imwrite(out_img_filepath, temp_img)
                        cv2.imwrite(out_label_filepath, temp_label)
                
                        new_filepath_list.append(out_img_filepath)
        write_list_to_txt(new_txt_filepath, new_filepath_list)

    
def cut_pos_neg_block():
    # txt_files = [
    #         # '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_2/train.txt', 
    #             '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_2/val.txt',    
    #         '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_2/test.txt']
    
    # out_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_2_block/'
    # out_data_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_2_block/data'
    
    txt_files = [
            # '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_2/train.txt', 
                '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_0328/val.txt',    
            '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_0328/test.txt']
    
    out_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_0328_block/'
    out_data_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_0328_block/data'
    
    if not os.path.exists(out_data_dir):
        os.makedirs(out_data_dir)
        
    # ignore_index = 255
    size = 480
    for txt_file in txt_files:
        split = txt_file.split('/')[-1].split('.')[0]
        new_txt_filepath = os.path.join(out_dir, f'{split}.txt')
        new_filepath_list = []
        img_filepaths = read_txt_to_list(txt_file)
        img_filepaths = list(set(img_filepaths))
        for i, img_filepath in enumerate(img_filepaths):
            ori_img_name = img_filepath.split('/')[-1]
            
            print(f'processing {split}, {i}/{len(img_filepaths)}, {ori_img_name}')
            label_filepath = img_filepath.replace('.jpg', '.png')
            label = cv2.imread(label_filepath, 0)
            img = cv2.imread(img_filepath)
            mask = label.copy()
            mask[mask==255] = 0
            mask[mask==13] = 0
            mask[mask==23] = 0
            mask[mask==33] = 0
            mask[mask==43] = 0
            mask[mask==53] = 0
            mask[mask>=61] = 0
            # if not mask.any() > 0:
            #     continue
            connectivity = 4  
        
            # Perform the operation
            output = cv2.connectedComponentsWithStats(mask, connectivity, cv2.CV_32S)
            # Get the results

            # # The first cell is the number of labels
            # num_labels = output[0]
            # print('num_labels', num_labels)

            # # The second cell is the label matrix
            # labels = output[1]
            # print('labels', labels)

            # # The third cell is the stat matrix, (xmin, ymin, width, height, area)
            stats = output[2]
            # print('stats', stats)

            # # The fourth cell is the centroid matrix, (cx, cy)
            centroids = output[3]
            # print('centroids', centroids)
            h, w, _ = img.shape
            if len(stats) > 1:
                stats = stats[1:]
                centroids = centroids[1:]
                # img = Image.fromarray(img)
                # mask = Image.fromarray(label)
                
                n_cen = len(centroids)
                idx_list = [i for i in range(n_cen)]
                if n_cen > 5:
                    random.shuffle(idx_list)
                    random.shuffle(idx_list)
                    idx_list = idx_list[:5]
                
                for j, idx in enumerate(idx_list):
                    # x0, y0, width, height, _ = stats[idx].astype(np.uint32)
                    cx, cy = centroids[idx].astype(np.uint32)
                    
                    xmin = cx - size // 2 if cx - size // 2 > 0 else 0
                    ymin = cy - size // 2 if cy - size // 2 > 0 else 0
                    xmax = cx + size // 2 if cx + size // 2 < w else w
                    ymax = cy + size // 2 if cy + size // 2 < h else h
                    
                    img_block = img[ymin:ymax, xmin:xmax]
                    label_block = label[ymin:ymax, xmin:xmax]
                    new_img_name = ori_img_name.replace('.jpg', f'_P_c_{j}.jpg')
                    new_label_name = ori_img_name.replace('.jpg', f'_P_c_{j}.png')
                    out_img_filepath = os.path.join(out_data_dir, new_img_name)
                    out_label_filepath = os.path.join(out_data_dir, new_label_name)
                    cv2.imwrite(out_img_filepath, img_block)
                    cv2.imwrite(out_label_filepath, label_block)
                    new_filepath_list.append(out_img_filepath)


            row = round(h/size)
            col = round(w/size)
            
            row = 1 if row == 0 else row
            col = 1 if col == 0 else col
            base_h = int(h/row)
            base_w = int(w/col)
            for r in range(row):
                for c in range(col):
                    xmin = 0
                    ymin = 0
                    temp_img = img[ymin + r*base_h: ymin + (r+1)*base_h, xmin + c*base_w: xmin + (c+1)*base_w]
                    temp_label = label[ymin + r*base_h: ymin + (r+1)*base_h, xmin + c*base_w: xmin + (c+1)*base_w]
                    temp_mask = mask[ymin + r*base_h: ymin + (r+1)*base_h, xmin + c*base_w: xmin + (c+1)*base_w]
                    nonzero = temp_mask[temp_mask>0]
                    if len(nonzero) > 12:
                        tag = 'P'
                    else:
                        tag = 'N'
                    new_img_name = ori_img_name.replace('.jpg', f'_{tag}_{r}_{c}.jpg')
                    new_label_name = ori_img_name.replace('.jpg', f'_{tag}_{r}_{c}.png')
                    out_img_filepath = os.path.join(out_data_dir, new_img_name)
                    out_label_filepath = os.path.join(out_data_dir, new_label_name)
                    cv2.imwrite(out_img_filepath, temp_img)
                    cv2.imwrite(out_label_filepath, temp_label)
            
                    new_filepath_list.append(out_img_filepath)
        write_list_to_txt(new_txt_filepath, new_filepath_list)


    
def cut_black_bk_to_block():
    # txt_files = [
    #         # '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_2/train.txt', 
    #             '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_2/val.txt',    
    #         '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_2/test.txt']
    
    # out_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_2_block/'
    # out_data_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_2_block/data'
    
    input_dir = '/home/hongrui/project/metro_pro/dataset/pot/test_val_blank_bk'
    
    splits = ['test', 'val']
    out_dir = '/home/hongrui/project/metro_pro/dataset/pot/test_val_blank_bk/block'
    
    
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    # ignore_index = 255
    size = 480
    tag = 'N'
    for split in splits:
        new_txt_filepath = os.path.join(out_dir, f'{split}.txt')
        new_filepath_list = []
        in_img_dir = os.path.join(input_dir, split)
        img_filepaths = glob.glob(osp.join(in_img_dir, "*.jpg"))
        out_img_dir = os.path.join(out_dir, split)
        if not os.path.exists(out_img_dir):
            os.makedirs(out_img_dir)
        for i, img_filepath in enumerate(img_filepaths):
            ori_img_name = img_filepath.split('/')[-1]
            
            print(f'processing {split}, {i}/{len(img_filepaths)}, {ori_img_name}')
            label_filepath = img_filepath.replace('.jpg', '.png')
            label = cv2.imread(label_filepath, 0)
            img = cv2.imread(img_filepath)
            
            h, w = label.shape
            row = round(h/size)
            col = round(w/size)
            
            row = 1 if row == 0 else row
            col = 1 if col == 0 else col
            base_h = int(h/row)
            base_w = int(w/col)
            for r in range(row):
                for c in range(col):
                    xmin = 0
                    ymin = 0
                    temp_img = img[ymin + r*base_h: ymin + (r+1)*base_h, xmin + c*base_w: xmin + (c+1)*base_w]
                    temp_label = label[ymin + r*base_h: ymin + (r+1)*base_h, xmin + c*base_w: xmin + (c+1)*base_w]
                    # temp_mask = mask[ymin + r*base_h: ymin + (r+1)*base_h, xmin + c*base_w: xmin + (c+1)*base_w]
                    # nonzero = temp_mask[temp_mask>0]
                    # if len(nonzero) > 12:
                    #     tag = 'P'
                    # else:
                    #     tag = 'N'
                    new_img_name = ori_img_name.replace('.jpg', f'_{tag}_{r}_{c}.jpg')
                    new_label_name = ori_img_name.replace('.jpg', f'_{tag}_{r}_{c}.png')
                    out_img_filepath = os.path.join(out_img_dir, new_img_name)
                    out_label_filepath = os.path.join(out_img_dir, new_label_name)
                    cv2.imwrite(out_img_filepath, temp_img)
                    cv2.imwrite(out_label_filepath, temp_label)
            
                    new_filepath_list.append(out_img_filepath)
        write_list_to_txt(new_txt_filepath, new_filepath_list)



def delete_repeat_in_txt():

    # input_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_0/'
    # input_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_1/'
    # input_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_2/'
    # input_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_2_block/'
    
    # input_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_2/data'
    # output_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_2/'

    # input_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_0328/data'
    # input_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_0328/'
    input_dir = '/home/hongrui/project/metro_pro/dataset/pot/test_val_blank_bk/block'


    # ref_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_1/data'
    # ref_img_names = os.listdir(ref_dir)
    # ref_img_names = []
    
    # splits =  ['val',   'test', 'train']
    splits =  ['val',   'test']
    
    total_num = 0
    for split in splits:
        txt_filepath = os.path.join(input_dir, f'{split}.txt')
        img_filepaths = read_txt_to_list(txt_filepath)
        os.remove(txt_filepath)
        img_filepaths = list(set(img_filepaths))
        random.shuffle(img_filepaths)
        random.shuffle(img_filepaths)
        write_list_to_txt(txt_filepath, img_filepaths)
        total_num += len(img_filepaths)
    
    print('total_num', total_num)
    
    
    
    
    
def count_images_nums():
    
    # input_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_2/'
    input_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_0328/'


    # ref_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_1/data'
    # ref_img_names = os.listdir(ref_dir)
    # ref_img_names = []
    
    dirs =  ['val',   'test', 'train']
    
    
    test_list_txt = os.path.join(input_dir, 'test.txt')
    val_list_txt = os.path.join(input_dir, 'val.txt')
    train_list_txt = os.path.join(input_dir, 'train.txt')
    
    test_filepaths = read_txt_to_list(test_list_txt)
    val_filepaths = read_txt_to_list(val_list_txt)
    train_filepaths = read_txt_to_list(train_list_txt)
    
    test_filepaths = list(set(test_filepaths))
    val_filepaths = list(set(val_filepaths))
    train_filepaths = list(set(train_filepaths))
    
    
    total_num = 0
    total_num += len(test_filepaths)
    total_num += len(val_filepaths)
    total_num += len(train_filepaths)
    
    print('total_num', total_num)

def write_zip_file():
    txt_files = [
            # '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_2/train.txt', 
                '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_0328/val.txt',    
            '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_0328/test.txt']
    
    out_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_0328/'
    
    input_dir = '/home/hongrui/project/metro_pro/dataset/pot/'
    dir_names = ['20220108', '20220222', '20220328']

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        

    for txt_file in txt_files:
        split = txt_file.split('/')[-1].split('.')[0]
        zip_filepath = os.path.join(out_dir, f'{split}.zip')

        img_filepaths = read_txt_to_list(txt_file)
        img_filepaths = list(set(img_filepaths))
        with ZipFile(zip_filepath,'w') as zip:
            for i, img_filepath in enumerate(img_filepaths):
                img_name = img_filepath.split('/')[-1]
                for dir_name in dir_names:
                    filepath = os.path.join(input_dir, dir_name, 'image', img_name)
                    # print(filepath)
                    if os.path.exists(filepath):
                        zip.write(filepath)



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


    # relocate_rail_regin_in_images(args)
    # relocate_rail_regins_in_all_images_once(args)
    
    # copy_train_val_json_2nd_round(args)
    # copy_test_json_2nd_round(args)
    # rename_video(args)    
    # check_annotation(args)
    # colorize_anno(args)
    # find_all_images_jsons(args)
    # find_jsons(args)
    # split_rails2_images(args)
    # del_unclear_images(args)

    # count_dataset_v2()
    # create_imgfilepath_txt(args)
    # create_imgfilepath_txt_two(args)
    # find_images_json(args)
    # creat_train_val_test_filelist()
    # sort_GC10_DET()

    # calculate_mean_value()
    # read_imgfilepath()
    # split_train_val_test_filelist()
    
    # verify_exist()
    # del_all_checkpoint_pth_tar()
    calculate_pixels_and_plot_dist()
    # cut_block()
    # split_train_val_test_filelist()
    # delete_repeat_in_txt()
    # count_images_nums()

    # calculate_pos_vs_neg()
    # calculate_pos_vs_neg_v2()

    # cut_pos_neg_block()
    # write_zip_file()
    # cut_black_bk_to_block()