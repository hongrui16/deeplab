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
import pixellib
# from pixellib.custom_train import instance_custom_training
# from custom_train import instance_custom_dataset_model_training
# from pixellib_local.custom_train import instance_custom_training



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


def convert_json_to_label(args):
    # input_dir   = args.inp
    # output_dir      = args.oup

    input_dir = '/comp_robot/hongrui/metro_pro/dataset/1st_5000_2nd_round/'
    # dirs = ['train', 'test', 'val']
    # dirs = ['train', 'val']
    dirs = ['test_ori']

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



def convert_json_and_mosaic_image(args):
    # input_dir   = args.inp
    # output_dir      = args.oup

    input_dir = '/home/hongrui/project/metro_pro/dataset/v2_2rails/unsorted'
    refer_parent_dir  = '/comp_robot/hongrui/metro_pro/dataset/1st_5000_2nd_round/'
    output_dir = '/home/hongrui/project/metro_pro/dataset/v2_2rails/sorted'
    dirs = ['train', 'test', 'val']

    image_names_list = [[] for _ in range(3)]
    out_img_dirs = []
    out_label_dirs = []
    for i, d in enumerate(dirs):
        ref_d_dir = os.path.join(refer_parent_dir, d)
        ref_img_dir = os.path.join(ref_d_dir, 'image')
        ref_img_names = os.listdir(ref_img_dir)
        image_names_list[i] = ref_img_names
        
        output_image_dir = os.path.join(output_dir, d, 'image')
        output_label_dir = os.path.join(output_dir, d, 'label')
        out_img_dirs.append(output_image_dir)
        out_label_dirs.append(output_label_dir)
        
        if not os.path.exists(output_image_dir):
            os.makedirs(output_image_dir)

        if not os.path.exists(output_label_dir):
            os.makedirs(output_label_dir)
            
    files = os.listdir(input_dir)
    ori_images = []
    ori_jsons = []
    for file in files:
        if '.jpg' in file:
            ori_images.append(file)
        elif '.json' in file:
            ori_jsons.append(file)


    random.shuffle(ori_jsons)
    for i, json_name in enumerate(ori_jsons):
        print(f'processing {json_name} {i+1}/{len(ori_jsons)}')
        img_name = json_name.replace('.json', '.jpg')
        ori_img_filepath = os.path.join(input_dir, img_name)
        if not os.path.exists(ori_img_filepath):
            continue
        for j, temp_img_names in enumerate(image_names_list):
            if img_name in temp_img_names:
                output_image_dir = out_img_dirs[j]
                output_label_dir = out_label_dirs[j]
        img = cv2.imread(ori_img_filepath)
        mask_name = json_name.replace('.json', '.png')
        
        ori_json_filepath = os.path.join(input_dir, json_name)
        data = json.load(open(ori_json_filepath))
        

        metro_label_name_to_value = {"left_1": 1, "right_1": 2, "left_2": 3, "right_2": 4, 
                                        "left_3": 5, "right_3": 6, "left_4": 7, "right_4": 8,
                                        "left_5": 9, "right_5": 10, "left_6": 11, "right_6": 12}
        negs_label_name_to_value = {"miss": 251, "too_much": 252}

        rails_info = []
        negs_info = []
        delete_label = 'remove_from_training'
        label_info = data["shapes"]
        delete_flag = False
        for shape in label_info:
            label_name = shape["label"]
            if label_name == delete_label:
                delete_flag = True
                break
            elif label_name in metro_label_name_to_value:
                rails_info.append(shape)
            elif label_name in negs_label_name_to_value:
                negs_info.append(shape)
        if delete_flag:
            continue
        
        rail_mask, _ = utils.shapes_to_label(
            img.shape, rails_info, metro_label_name_to_value
        )

        neg_mask, _ = utils.shapes_to_label(
            img.shape, negs_info, negs_label_name_to_value
        )

        noise = np.random.randint(0, 255, (img.shape))
        for c in range(3):
            img[:,:,c] = img[:,:,c]*(neg_mask<=0)
            noise[:,:,c] = noise[:,:,c]*(neg_mask > 0)
        img = img + noise

        rail_mask *= neg_mask<=0
        rail_mask *= 20

        # lbl *= 100
        # print(lbl[lbl>100], lbl.max())
        # print('rail_mask', rail_mask, rail_mask.shape)
        out_label_filepath = os.path.join(output_label_dir, mask_name)
        out_img_filepath = os.path.join(output_image_dir, img_name)
        # utils.lblsave(out_label_filepath, lbl)
        # print('Saved to: %s' % out_label_filepath)
        cv2.imwrite(out_label_filepath, rail_mask.astype(np.uint8))
        cv2.imwrite(out_img_filepath, img.astype(np.uint8))
        
        
        # if i > 10:
        #     break

def read_all_label_name_from_json(args):
    input_json_dir = '/home/hongrui/project/metro_pro/dataset/pot_20220108'

    json_names = os.listdir(input_json_dir)
    
    labelnames = []
    for i, json_name in enumerate(json_names):
        print(f'processing {json_name} {i+1}/{len(json_names)}')
        if not '.json' in json_name:
            continue
        ori_json_filepath = os.path.join(input_json_dir, json_name)
        data = json.load(open(ori_json_filepath))
        for shape in sorted(data["shapes"], key=lambda x: x["label"]):
            label_name = shape["label"]
            if label_name in labelnames:
                continue
            else:
                labelnames.append(label_name)

    print('labelnames', labelnames)

def convert_pot_json_to_mask():
    label_names = ['LaSi_rect', 'TuQi', 'ZhouBian', 'HuaHen_rect', 'pot', 'HuaHen']

    output_dir = '/home/hongrui/project/metro_pro/dataset/pot_seg'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    input_json_dir = '/home/hongrui/project/metro_pro/dataset/pot_20220108'

    json_names = os.listdir(input_json_dir)
    
    labelnames = []
    for i, json_name in enumerate(json_names):
        print(f'processing {json_name} {i+1}/{len(json_names)}')
        if not '.json' in json_name:
            continue
        ori_json_filepath = os.path.join(input_json_dir, json_name)
        data = json.load(open(ori_json_filepath))
        for shape in sorted(data["shapes"], key=lambda x: x["label"]):
            label_name = shape["label"]

    

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
    # pixellib_vis(args)
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
    # convert_json_and_mosaic_image(args)
    # read_all_label_name_from_json(args)
    convert_json(args)