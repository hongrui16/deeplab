import matplotlib.patches as patches
from matplotlib.path import Path
import os
import sys
import io
import cv2
import time
import argparse
import shutil
import numpy as np

from PIL import Image, ImageOps, ImageDraw, ImageFilter

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('agg')


def get_random_shape(edge_num=3, ratio=0.7, width=400, height=300):
    '''
      There is the initial point and 3 points per cubic bezier curve. 
      Thus, the curve will only pass though n points, which will be the sharp edges.
      The other 2 modify the shape of the bezier curve.
      edge_num, Number of possibly sharp edges
      points_num, number of points in the Path
      ratio, (0, 1) magnitude of the perturbation from the unit circle, 
    '''
    points_num = edge_num*3 + 1
    angles = np.linspace(0, 2*np.pi, points_num)
    codes = np.full(points_num, Path.CURVE4)
    codes[0] = Path.MOVETO
    # Using this instad of Path.CLOSEPOLY avoids an innecessary straight line
    verts = np.stack((np.cos(angles), np.sin(angles))).T * \
        (2*ratio*np.random.random(points_num)+1-ratio)[:, None]
    verts[-1, :] = verts[0, :]
    path = Path(verts, codes)
    # draw paths into images
    fig = plt.figure()
    ax = fig.add_subplot(111)
    patch = patches.PathPatch(path, facecolor='black', lw=2)
    ax.add_patch(patch)
    ax.set_xlim(np.min(verts)*1.1, np.max(verts)*1.1)
    ax.set_ylim(np.min(verts)*1.1, np.max(verts)*1.1)
    ax.axis('off')  # removes the axis to leave only the shape
    fig.canvas.draw()
    # convert plt images into numpy images
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape((fig.canvas.get_width_height()[::-1] + (3,)))
    plt.close(fig)
    # postprocess
    data = cv2.resize(data, (width, height))[:, :, 0]
    data = (1 - np.array(data > 0).astype(np.uint8))*255
    corrdinates = np.where(data > 0)
    xmin, xmax, ymin, ymax = np.min(corrdinates[0]), np.max(
        corrdinates[0]), np.min(corrdinates[1]), np.max(corrdinates[1])
    #region = Image.fromarray(data).crop((ymin, xmin, ymax, xmax))
    data = data[xmin:xmax, ymin:ymax]


    return data



def get_bbox(mask):
    #compute bounding box
    hor = np.sum(mask, axis=0)
    hor_idx = np.nonzero(hor)[0]
    x = hor_idx[0]
    width = hor_idx[-1] - x + 1
    vert = np.sum(mask, axis=1)
    vert_idx = np.nonzero(vert)[0]
    y = vert_idx[0]
    height = vert_idx[-1] - y + 1
    bbox = [int(x), int(y), int(width), int(height)]

    return bbox


def iou(boxA, boxB):
    
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return iou



def draw_one_box_cls(im, bbox, cls):
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(im,(x_min,y_min),(x_max,y_max),(0,255,0), 2)
    
    org= (x_min,y_min)
    font = cv2.FONT_HERSHEY_SIMPLEX 
    fontScale = 1
    color = (255, 0, 0)
    thickness = 2

    im = cv2.putText(im, str(cls), org, font,  fontScale, color, thickness, cv2.LINE_AA)
    
    return im



def list_image_in(folder):
    images = list()
    for root, dirs, files in os.walk(folder):
        for name in files:
            im = os.path.join(root, name)
            #if 'jpeg' in im.lower() or 'jpg' in im.lower() or 'png' in im.lower():
            im_lower = im.lower()
            if im_lower.endswith('jpeg') or im_lower.endswith('jpg') or im_lower.endswith('png'):
                images.append(im)
    return images

    

if __name__ == "__main__":
    h = np.random.randint(200, 350)
    w = np.random.randint(300, 450)
    region = get_random_shape(width = w, height= h)
    region = np.array(region)
    print(region.shape)
    print(region.max())
    print(region.min())
    
    print('unique ', np.unique(region))
    # plt.imshow(region)
    # plt.show()
    cv2.imshow('shape', region)
    # time.sleep(5)
    if cv2.waitKey(1000) == 27:# 按下esc退出
        cv2.destroyAllWindows()

