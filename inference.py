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
import sys
import time
from PIL import Image
from torchvision import transforms

from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *

# print(f'calling {__file__}, {sys._getframe().f_lineno}')



class Inference(object):
    def __init__(self, args):
        self.args = args
        self.max_size = args.max_size
        self.mean=(0.485, 0.456, 0.406)
        self.std=(0.229, 0.224, 0.225)
        self.transform = transforms.Compose([            #[1]
             transforms.ToTensor(),                     #[4]
             transforms.Normalize(                      #[5]
             mean=[0.485, 0.456, 0.406],                #[6]
             std=[0.229, 0.224, 0.225]                  #[7]
             )])

        print(f'Define network...')
        self.model = DeepLab(num_classes   = args.n_classes,
                             backbone      = args.backbone,
                             output_stride = args.out_stride,
                             sync_bn       = args.sync_bn,
                             freeze_bn     = args.freeze_bn)

        # Load weight
        print(f'Load weight from {args.resume}')
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.model.load_state_dict(torch.load(args.resume), strict=False)
        self.model.load_state_dict(torch.load(args.resume))
        self.model = self.model.to(device)
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
            image = image.resize((ow, oh), Image.BILINEAR)
        image = transform(image)
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
        pred = torch.max(output, dim=1)
        pred = pred.squeeze(0).cpu().numpy()
        return pred
      
    def postprocess(self, infer):
        max_id = infer.max()
        ratio = 255//max_id
        infer *= ratio
        return infer

                

def main(args):
    input_dir   = args.testset_dir
    output_dir  = args.testOut_dir

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    inference_engine = Inference(args)

    img_names = os.listdir(input_dir)

    for i, img_name in enumerate(img_names):
        print(f'processing {img_name} {i+1}/{len(img_names)}')
        ori_img_filepath = os.path.join(input_dir, img_name)

        img = Image.open(ori_img_filepath)
        infer = inference_engine.inference(img)
        infer = inference_engine.postprocess(infer)

        infer_mask_name = f"{img_name.split('.')[0]}_infer.png"                    
        out_infer_filepath = os.path.join(output_dir, infer_mask_name)
        cv2.imwrite(out_infer_filepath, infer)
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='IDEA Training')

    
    parser.add_argument('--gpu', default='0', type=str,
                        help='GPU id to use.')

    parser.add_argument('--backbone', type=str, default='resnet',
                            choices=['resnet', 'xception', 'drn', 'mobilenet'],
                            help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=8,
                        help='network output stride (default: 8)')

    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: False)')

    parser.add_argument('--sync-bn', type=bool, default=False,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--max_size', type=int, default=1080)
    parser.add_argument('--n_classes', type=int, default=3)
    parser.add_argument('--testset_dir', type=str, default=None, help='input test or inference image dir')
    parser.add_argument('--testOut_dir', type=str, default=None, help='inference and test output dir')
    
    args = parser.parse_args()
    main(args)
    # plot_image()