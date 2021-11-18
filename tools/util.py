import sys
import time
import os
import numpy as np
import random
import shutil
import cv2
import scipy

from PIL import Image
import math
from math import pi
import imageio

from matplotlib import pyplot as plt


rgb_palette = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]]
                       
rgb_palette = [item for sublist in rgb_palette for item in sublist]
zero_pad = 256 * 3 - len(rgb_palette)
for i in range(zero_pad):
    rgb_palette.append(0)

def colorize_mask_to_bgr(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(rgb_palette)
    mask_rgb = new_mask.convert("RGB")
    mask_rgb = np.array(mask_rgb)
    mask_bgr = mask_rgb[:,:,::-1]
    return mask_bgr

def write_list_to_txt(txt_filepath, lists):
    f = open(txt_filepath, "a+")
    for l in lists:
        f.write(str(l)+'\n')
    f.close()#

def read_txt_to_list(txt_filepath):
    lists = []
    with open(txt_filepath) as f:
        lines = f.readlines()
        for line in lines:
            lists.append(line.strip('\n'))
    # print(len(lists))
    return lists

def sort_left_right_lane(label):
    left_0 = min(label[label>0])
    # print(label[label>left_0].min())
    right_0 = 2*left_0
    # print('left_0 ',left_0, 'right_0 ', right_0)
    return label*(label==left_0), label*(label==right_0)

def find_bottom_lane_location_in_labels(label):
    left_lane, right_lane = sort_left_right_lane(label.copy())
    # cv2.imwrite('left_lane.png', left_lane)
    # cv2.imwrite('right_lane.png', right_lane)
    assert label.ndim == 2
    h, w = label.shape
    max_y_left = left_lane.nonzero()[0].max()
    max_y_right = right_lane.nonzero()[0].max()
    max_y = min(max_y_left, max_y_right)
    left_x_pos = left_lane[max_y-20:max_y].nonzero()[1].mean()
    right_x_pos = right_lane[max_y-20:max_y].nonzero()[1].mean()
    return left_x_pos, right_x_pos


def plot_and_save_complex_func(res,  mask_name = None, out_img_filepath = None, text_str = None, debug = False):
    if debug:
        print(f'call {sys._getframe().f_code.co_name}')
    if 1 <= len(res) <= 3:
        row = 1
        col = len(res)

    elif 4 <= len(res) <= 8:
        row = 2
        col = len(res)//2 + len(res)%2

    elif 9 <= len(res) <= 12:
        # col = math.sqrt(len(res)) if math.sqrt(len(res))%1 == 0 else int(math.sqrt(len(res))) + 1
        # row = int(math.sqrt(len(res)))
        row = 3
        col = len(res)//3 + len(res)%3

    else:
        row = len(res)//5 + 1
        col = 5
        
    
    for i in range(len(res)):
        # img = res[i].astype(np.uint8)
        img = res[i]
        if img.ndim == 2:
            height, width = img.shape
        elif img.ndim == 3:
            height, width, _ = img.shape
        else:
            return
        if i == 0:
            ori_img = img.copy()
            
            ax = plt.subplot(row, col, i+1), plt.imshow(img), plt.title(mask_name[i]), plt.xticks([]), plt.yticks([])
            if text_str:
                # ax.text(2.0, 9.5, text_str, fontsize=10)
                # ax.text(.05, .95, text_str, color = 'red', transform=ax.transAxes, ha="left", va="top")
                plt.text(.05, .95, text_str, fontsize = 6, color = 'red', ha = "left", va = "top", rotation = 0, wrap = True)

        else:
            if mask_name:
                if 'fitted_curves' in mask_name[i]:
                    # ax = plt.subplot(row, col, i+1), plt.imshow(np.zeros((height,width))), plt.title(mask_name[i]), plt.xticks([]), plt.yticks([])
                    ax = plt.subplot(row, col, i+1), plt.imshow(ori_img), plt.title(mask_name[i]), plt.xticks([]), plt.yticks([])
                    x_plot, curve_fits = img
                    lw = 0.2
                    for j in range(len(curve_fits)):
                        plt.plot(np.polyval(curve_fits[j], x_plot), x_plot, color='lightgreen', linestyle='--', linewidth=lw)
                    plt.xlim(0, width)
                    plt.ylim(height, 0)
                    
                else:
                    ax = plt.subplot(row, col, i+1), plt.imshow(img), plt.title(mask_name[i]), plt.xticks([]), plt.yticks([])
            else:    
                ax = plt.subplot(row, col, i+1), plt.imshow(img), plt.title(f'res_{i}'), plt.xticks([]), plt.yticks([])
            if text_str:
                # ax.text(2.0, 9.5, text_str, fontsize=10)
                # ax.text(.05, .95, text_str, color = 'red', transform=ax.transAxes, ha="left", va="top")
                plt.text(.05, .95, text_str, fontsize = 6, color = 'red', ha = "left", va = "top", rotation = 0, wrap = True)

        if isinstance(res[i], np.ndarray) and res[i].ndim == 2:
            plt.gray()

    plt.subplots_adjust(wspace=0)
    if out_img_filepath:
        print(f'saving {out_img_filepath}')
        figure = plt.gcf()  
        figure.set_size_inches(16, 9)
        plt.savefig(out_img_filepath, dpi=900, bbox_inches='tight')
        plt.close()
    # else:
    #     plt.show()
    #     time.sleep(3)
    #     plt.close()
    print()

if __name__ == '__main__':
    # read_txt_to_list('/home/hongrui/project/metro_pro/edge_detection/chdis_v2.txt')
    pass