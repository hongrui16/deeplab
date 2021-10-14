import sys
import time
import os
import numpy as np
import random
import shutil
import cv2

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

if __name__ == '__main__':
    read_txt_to_list('/home/hongrui/project/metro_pro/edge_detection/chdis_v2.txt')
