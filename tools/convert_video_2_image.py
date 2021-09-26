import cv2
import logging
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import numpy as np


import argparse
import os
import sys
import time
import shutil

def convert(args):

    input_dir = args.input_dir
    output_dir = args.output_dir

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    videoCapture = cv2.VideoCapture()
    video_names = os.listdir(input_dir)

    for i, v_name in enumerate(video_names):
        print(f'processing {v_name}, {i}/{len(video_names)}')
        if not '.MP4' in v_name:
            continue
        video_filepath = os.path.join(input_dir, v_name)

        v_name_prefix = v_name.split('.')[0]

        videoCapture.open(video_filepath)

        if videoCapture.isOpened():
            success = True
        else:
            success = False
            print(f"读取 {video_filepath} 失败!")
            continue

        frame_index = 0
        frame_count = 0
        interval = 15

        fps = videoCapture.get(cv2.CAP_PROP_FPS)
        frames = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
        wid = int(videoCapture.get(3))
        hei = int(videoCapture.get(4))
        #fps是帧率，意思是每一秒刷新图片的数量，frames是一整段视频中总的图片数量。
        # print("fps=",fps,"frames=",frames)
        # print("wid=",wid,"hei=",hei)
        # res = np.zeros((hei, wid, 3))
        # videoCapture.set(cv2.CAP_PROP_POS_FRAMES, end)

        while success:
            success, frame = videoCapture.read()
            if frame_index % interval == 0 and success:

                out_img_name = f"{v_name_prefix}_{frame_index}.jpg"
                out_img_filepath = os.path.join(output_dir, out_img_name)
                cv2.imwrite(out_img_filepath, frame)
                frame_count += 1
            # res += (frame/num)
            frame_index += 1
        videoCapture.release()   


def select_images(args):
    input_dir = args.input_dir
    output_dir = args.output_dir

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_names = os.listdir(input_dir)
    for i, v_name in enumerate(video_names):
        print(f'processing {v_name}, {i}/{len(video_names)}')
        if not i % 4 == 0:
            continue
        video_filepath = os.path.join(input_dir, v_name)
        out_img_filepath = os.path.join(output_dir, v_name)
        shutil.move(video_filepath, out_img_filepath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-im', '--input_dir', type=str, default=None)
    parser.add_argument('-om', '--output_dir', type=str, default=None)
    args = parser.parse_args()
    # convert(args)
    select_images(args)
# cv2.imshow('13456', frame)
# cv2.waitKey(1000)
# plt.imshow(frame)
# plt.show()

# for i in range(int(frames)):
#     ret,frame = videoCapture.read()
#     cv2.imwrite("E:/video/pictures/1-1.avi(%d).jpg"%i,frame)
#     