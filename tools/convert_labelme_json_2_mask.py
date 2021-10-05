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
            


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='aligment')
    # parser.add_argument('-if', '--img_filepath', type=str, default='images\\rail_06.jpg')
    parser.add_argument('-if', '--img_filepath', type=str, default=None)

    parser.add_argument('-im', '--input_dir', type=str, default=None)

    parser.add_argument('-om', '--output_dir', type=str, default=None)
    parser.add_argument('-th', '--target_height', type=int, default=1440)
    parser.add_argument('-tw', '--target_width', type=int, default=1920)

    args = parser.parse_args()
    # split_train_val_dataset(args)
    # compose()
    # test()
    # open_alg_fun()
    # convert_json_2_mask(args)
    convert_json(args)