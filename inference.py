# -*- coding: utf-8 -*-
# @Last Modified by:   Hong Rui
import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.special import softmax

import sys
import time
from PIL import Image
from torchvision import transforms
import random
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from tools.util import *
# print(f'calling {__file__}, {sys._getframe().f_lineno}')



class RailInference(object):
    def __init__(self, args):
        self.args = args
        self.mean=(0.485, 0.456, 0.406)
        self.std=(0.229, 0.224, 0.225)
        self.transform = transforms.Compose([            #[1]
             transforms.ToTensor(),                     #[4]
             transforms.Normalize(                      #[5]
             mean=[0.485, 0.456, 0.406],                #[6]
             std=[0.229, 0.224, 0.225]                  #[7]
             )])
        # num_classes   = args.n_classes    if args.n_classes else 3
        # backbone      = args.backbone     if args.backbone  else 'resnet'
        # output_stride = args.out_stride   if args.out_stride else 8
        # sync_bn       = args.sync_bn      if not args.sync_bn is None else False
        # freeze_bn     = args.freeze_bn    if not args.freeze_bn is None else False
        # max_size      = args.max_size     if args.max_size else 1080

        num_classes   = 3
        backbone      = 'resnet'
        output_stride = 8
        sync_bn       = False
        freeze_bn     = False
        max_size      = 1080

        self.max_size = max_size

        print(f'Define network...')
        
        self.model = DeepLab(num_classes   = num_classes,
                             backbone      = backbone,
                             output_stride = output_stride,
                             sync_bn       = sync_bn,
                             freeze_bn     = freeze_bn)

        
        if args.gpu_id:
            os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Load weight
        print(f'Load weight from {args.resume}')
        checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['state_dict'])
        
        self.model.eval()
        print("Initialization finished")

    def preprocess_np(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resize_flag = True
        h, w, _ = image.shape
        if w > h and w > self.max_size:
            oh = int(self.max_size/w*h)
            ow = self.max_size
        elif h > w and h > self.max_size:
            oh = self.max_size
            ow = int(self.max_size/h*w)
        else:
            resize_flag = False
        if resize_flag:
            image = cv2.resize(image, (ow, oh))
        image = image.astype("float32") / 255.0
        # subtract ImageNet mean, divide by ImageNet standard deviation,
        # set "channels first" ordering, and add a batch dimension
        image -= self.mean
        image /= self.std
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(self.device)
        return image

    def preprocess_pil(self, image):
        w, h = image.size
        resize_flag = True
        if w > h and w > self.max_size:
            oh = int(self.max_size/w*h)
            ow = self.max_size
        elif h > w and h > self.max_size:
            oh = self.max_size
            ow = int(self.max_size/h*w)
        else:
            resize_flag = False
        if resize_flag:
            image = image.resize((ow, oh), Image.BILINEAR)
        image = self.transform(image)
        image = torch.unsqueeze(image, 0).to(self.device)
        return image

    def inference(self, image):
        if isinstance(image, Image.Image):
            image = self.preprocess_pil(image)
        elif isinstance(image, np.ndarray):
            image = self.preprocess_np(image)
        else:
            print('wrong data type')
        with torch.no_grad():
            output = self.model(image)
        # print('output', output)
        pred = output.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        pred = np.squeeze(pred, 0)
        return pred
    
    
    def inference_only(self, image):
        if isinstance(image, Image.Image):
            image = self.preprocess_pil(image)
        elif isinstance(image, np.ndarray):
            image = self.preprocess_np(image)
        else:
            print('wrong data type')
        with torch.no_grad():
            output = self.model(image)
        # print('output', output)
        pred = output.data.cpu().numpy()
        # pred = np.argmax(pred, axis=1)
        pred = np.squeeze(pred, 0)
        pred = np.transpose(pred, axes=[1, 2, 0])
        pred = softmax(pred, axis=2)
        return pred
      
    def postprocess(self, infer):
        # max_id = infer.max()
        # ratio = 255//max_id
        # infer *= ratio
        infer *= 40
        return infer



                

def verify_threshold(args):
    input_dir   = args.testset_dir
    output_dir  = args.testOut_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    inference_engine = RailInference(args)

    img_names = os.listdir(input_dir)

    def based_on_threshold(pre, thres):
        # print('pre.ndim', pre.ndim)
        # res = np.zeros(pre.shape[:2])
        for i in range(pre.ndim):
            p = pre[:,:,i]
            p[p < thres] = 0
            p[p >= thres] = i
            # res += p
            # pre[:,:,i][pre[:,:,i] < thres] = 0
            # pre[:,:,i][pre[:,:,i] >= thres] = i
        res = np.sum(pre, axis=2)
        # print('res.shape', res.shape)
        # print(res)
        return res

    thres = [0.1, 0.2, 0.33, 0.5, 0.6, 0.8, 0.9]
    random.shuffle(img_names)
    for i, img_name in enumerate(img_names):
        mask_name = []
        res = []
        print(f'processing {img_name} {i+1}/{len(img_names)}')
        ori_img_filepath = os.path.join(input_dir, img_name)

        img = cv2.imread(ori_img_filepath)
        res.append(img)
        mask_name.append('img')

        ori_infer = inference_engine.inference_only(img)

        arg_max = np.argmax(ori_infer, axis=2)
        arg_max = colorize_mask_to_bgr(arg_max)
        res.append(arg_max)
        mask_name.append('arg_max')

        for j in range(len(thres)):
            temp = based_on_threshold(ori_infer.copy(), thres[j])
            temp = colorize_mask_to_bgr(temp)
            res.append(temp.copy())
            mask_name.append(f'thres {thres[j]}')

        infer_mask_name = f"{img_name.split('.')[0]}_infer.jpg"                    
        out_infer_filepath = os.path.join(output_dir, infer_mask_name)
        plot_and_save_complex_func(res,  mask_name, out_infer_filepath)
        # cv2.imwrite(out_infer_filepath, infer)
        if i > 100:
            return

def main(args):
    input_dir   = args.testset_dir
    output_dir  = args.testOut_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    inference_engine = RailInference(args)

    img_names = os.listdir(input_dir)

    for i, img_name in enumerate(img_names):
        print(f'processing {img_name} {i+1}/{len(img_names)}')
        ori_img_filepath = os.path.join(input_dir, img_name)

        img = Image.open(ori_img_filepath)
        infer = inference_engine.inference_only(img)
        # infer = inference_engine.postprocess(infer)

        infer_mask_name = f"{img_name.split('.')[0]}_infer.png"                    
        out_infer_filepath = os.path.join(output_dir, infer_mask_name)
        # cv2.imwrite(out_infer_filepath, infer)
        cv2.imwrite(out_infer_filepath, (infer*255).astype(np.uint8))
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='IDEA Training')
    parser.add_argument('--backbone', type=str, default='resnet',
                            choices=['resnet', 'xception', 'drn', 'mobilenet'],
                            help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=8,
                        help='network output stride (default: 8)')
    parser.add_argument('--sync-bn', type=bool, default=False,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--max_size', type=int, default=1080)
    parser.add_argument('--n_classes', type=int, default=3)

    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('-g', '--gpu_id', default=None, type=str,
                        help='GPU id to use.')
    parser.add_argument('-im', '--testset_dir', type=str, default=None, help='input test or inference image dir')
    parser.add_argument('-om', '--testOut_dir', type=str, default=None, help='inference and test output dir')
    
    args = parser.parse_args()
    # main(args)
    verify_threshold(args)
    # plot_image()