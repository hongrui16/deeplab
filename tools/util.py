import sys
import time
import os
import numpy as np
import random
import shutil

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


if __name__ == '__main__':
    read_txt_to_list('/home/hongrui/project/metro_pro/edge_detection/chdis_v2.txt')
