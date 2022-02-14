#!/usr/bin/env python

import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys
import uuid
import cv2
import imgviz
import numpy as np
import random
import labelme
from util import *
try:
    import pycocotools.mask
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    sys.exit(1)


def main(args):
    

    if osp.exists(args.output_dir):
        print("Output directory already exists:", args.output_dir)
        sys.exit(1)
    os.makedirs(args.output_dir)
    os.makedirs(osp.join(args.output_dir, "JPEGImages"))
    if not args.noviz:
        os.makedirs(osp.join(args.output_dir, "Visualization"))
    print("Creating dataset:", args.output_dir)

    now = datetime.datetime.now()

    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[dict(url=None, id=0, name=None,)],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type="instances",
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()
        if class_id == -1:
            assert class_name == "__ignore__"
            continue
        class_name_to_id[class_name] = class_id
        data["categories"].append(
            dict(supercategory=None, id=class_id, name=class_name,)
        )

    out_ann_file = osp.join(args.output_dir, "annotations.json")
    label_files = glob.glob(osp.join(args.input_dir, "*.json"))
    for image_id, filename in enumerate(label_files):
        print(f"processing {filename}")

        label_file = labelme.LabelFile(filename=filename)

        base = osp.splitext(osp.basename(filename))[0]
        out_img_file = osp.join(args.output_dir, "JPEGImages", base + ".jpg")

        img = labelme.utils.img_data_to_arr(label_file.imageData)
        imgviz.io.imsave(out_img_file, img)
        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name=osp.relpath(out_img_file, osp.dirname(out_ann_file)),
                height=img.shape[0],
                width=img.shape[1],
                date_captured=None,
                id=image_id,
            )
        )

        masks = {}  # for area
        segmentations = collections.defaultdict(list)  # for segmentation
        for shape in label_file.shapes:
            points = shape["points"]
            label = shape["label"]
            group_id = shape.get("group_id")
            shape_type = shape.get("shape_type", "polygon")
            mask = labelme.utils.shape_to_mask(
                img.shape[:2], points, shape_type
            )

            if group_id is None:
                group_id = uuid.uuid1()

            instance = (label, group_id)

            if instance in masks:
                masks[instance] = masks[instance] | mask
            else:
                masks[instance] = mask

            if shape_type == "rectangle":
                (x1, y1), (x2, y2) = points
                x1, x2 = sorted([x1, x2])
                y1, y2 = sorted([y1, y2])
                points = [x1, y1, x2, y1, x2, y2, x1, y2]
            if shape_type == "circle":
                (x1, y1), (x2, y2) = points
                r = np.linalg.norm([x2 - x1, y2 - y1])
                # r(1-cos(a/2))<x, a=2*pi/N => N>pi/arccos(1-x/r)
                # x: tolerance of the gap between the arc and the line segment
                n_points_circle = max(int(np.pi / np.arccos(1 - 1 / r)), 12)
                i = np.arange(n_points_circle)
                x = x1 + r * np.sin(2 * np.pi / n_points_circle * i)
                y = y1 + r * np.cos(2 * np.pi / n_points_circle * i)
                points = np.stack((x, y), axis=1).flatten().tolist()
            else:
                points = np.asarray(points).flatten().tolist()

            segmentations[instance].append(points)
        segmentations = dict(segmentations)

        for instance, mask in masks.items():
            cls_name, group_id = instance
            if cls_name not in class_name_to_id:
                continue
            cls_id = class_name_to_id[cls_name]

            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)
            area = float(pycocotools.mask.area(mask))
            bbox = pycocotools.mask.toBbox(mask).flatten().tolist()

            data["annotations"].append(
                dict(
                    id=len(data["annotations"]),
                    image_id=image_id,
                    category_id=cls_id,
                    segmentation=segmentations[instance],
                    area=area,
                    bbox=bbox,
                    iscrowd=0,
                )
            )

        if not args.noviz:
            viz = img
            if masks:
                labels, captions, masks = zip(
                    *[
                        (class_name_to_id[cnm], cnm, msk)
                        for (cnm, gid), msk in masks.items()
                        if cnm in class_name_to_id
                    ]
                )
                viz = imgviz.instances2rgb(
                    image=img,
                    labels=labels,
                    masks=masks,
                    captions=captions,
                    font_size=15,
                    line_width=2,
                )
            out_viz_file = osp.join(
                args.output_dir, "Visualization", base + ".jpg"
            )
            imgviz.io.imsave(out_viz_file, viz)

    with open(out_ann_file, "w") as f:
        json.dump(data, f)


def convert_info(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    input_dir = '/home/hongrui/project/metro_pro/dataset/pot_20220108'
    output_dir = '/home/hongrui/project/metro_pro/dataset/pot_20220108_cut'
    # output_dir = 'temp'

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    label_names = ['pot', 'LaSi_rect', 'TuQi', 'ZhouBian', 'HuaHen_rect', 'HuaHen']

    

    # json_files = ['IMG_20220104_094754.json', 'IMG_20220104_101800.json']
    # json_files = os.listdir(input_dir)
    label_files = glob.glob(osp.join(input_dir, "*.json"))
    random.shuffle(label_files)
    for idx, filename in enumerate(label_files):
        if not '.json' in filename:
            continue
        print(f"processing {idx}/{len(label_files)}, {filename}")
        # filepath = os.path.join(input_dir, filename)
        base = osp.splitext(osp.basename(filename))[0]
        # print(base)
        in_img_filepath = osp.join(input_dir, base + ".jpg")
        out_img_filepath = osp.join(output_dir, base + ".jpg")
        label_info = labelme.LabelFile(filename=filename)
        img = cv2.imread(in_img_filepath)

        masks = np.zeros(img.shape[:2], dtype=np.uint8)  # for area
 
        # print(img.shape)
        for cnt, shape in enumerate(label_info.shapes):
            points = shape["points"]
            label = shape["label"]
            if not label in label_names:
                continue
            label_index = label_names.index(label)

            shape_type = shape.get("shape_type", "polygon")
            mask = labelme.utils.shape_to_mask(
                img.shape[:2], points, shape_type
            )
            
            if label == 'pot':
                mask_t = np.asfortranarray(mask.astype(np.uint8))
                mask_t = pycocotools.mask.encode(mask_t)
                area = float(pycocotools.mask.area(mask_t))
                pot_bbox = pycocotools.mask.toBbox(mask_t).flatten().tolist()
                # print('pot_bbox', pot_bbox)
                pot_bbox = list(map(int, pot_bbox))                
            else:
                mask_t = np.asfortranarray(mask.astype(np.uint8))
                if label == 'HuaHen':
                    mask_t = morphologyEx_close(mask_t, kernel_size = 7)
                    mask_t[mask_t > 0] = 1
                masks += mask_t*label_index
            # cv2.imwrite(os.path.join(output_dir, base + f"_{cnt}_{label}.jpg"), )

        xmin, ymin, w, h = pot_bbox
        
        img = multiple_img_with_binary_mask(img.copy(), mask)
        img = img[ymin:ymin+h, xmin:xmin+w]

        masks = multiple_img_with_binary_mask(masks.copy(), mask)
        masks[mask<=0] = 255
        masks = masks[ymin:ymin+h, xmin:xmin+w]
        # print('img.shape', img.shape)
        cv2.imwrite(os.path.join(output_dir, base + ".jpg"), img)
        cv2.imwrite(os.path.join(output_dir, base + ".png"), masks)

        masks_col = colorize_mask_to_bgr(masks)
        cv2.imwrite(os.path.join(output_dir, base + f"_blend.jpg"), masks_col)

        compose = 0.65*img + 0.35*masks_col
        cv2.imwrite(os.path.join(output_dir, base + f"_compose.jpg"), compose.astype(np.uint8))

        # if idx > 4:
        #     break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir", default=None, type = str, help="input annotated directory")
    parser.add_argument("--output_dir", default=None, type = str, help="output dataset directory")
    parser.add_argument("--labels",  default=None, type = str, help="labels file")
    parser.add_argument(
        "--noviz", help="no visualization", action="store_true"
    )
    args = parser.parse_args()

    # main(args)
    convert_info(args)