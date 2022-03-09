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


def convert_labelme_to_mask(args):
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
                # area = float(pycocotools.mask.area(mask_t))
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


def convert_labelme_to_bbox(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    input_dir = '/home/hongrui/project/metro_pro/dataset/pot/pot_20220108'
    output_dir = '/home/hongrui/project/metro_pro/dataset/pot/pot_20220108_cut/annos'
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
        out_txt_filepath = osp.join(output_dir, base + ".txt")
        label_info = labelme.LabelFile(filename=filename)
        img = cv2.imread(in_img_filepath)

        # masks = np.zeros(img.shape[:2], dtype=np.uint8)  # for area
 
        # print(img.shape)
        anno_list  = []
        for cnt, shape in enumerate(label_info.shapes):
            points = shape["points"]
            label = shape["label"]
            if not label in label_names:
                continue
            

            shape_type = shape.get("shape_type", "polygon")
            mask = labelme.utils.shape_to_mask(
                img.shape[:2], points, shape_type
            )
            
            mask_t = np.asfortranarray(mask.astype(np.uint8))
            mask_t = pycocotools.mask.encode(mask_t)
            # area = float(pycocotools.mask.area(mask_t))
            bbox = pycocotools.mask.toBbox(mask_t).flatten().tolist()
            
            if label == 'pot':
                pot_bbox = list(map(int, bbox))                
            else:
                obj_bbox = list(map(int, bbox))
                label_index = label_names.index(label) - 1
            anno_list.append([label_index] + obj_bbox)

        pot_xmin, pot_ymin, pot_w, pot_h = pot_bbox
        annos = []
        for anno in anno_list:
            cls_idx = anno[0]
            xmin = (anno[1] - pot_xmin)/pot_w
            ymin = (anno[2]-pot_ymin)/pot_h
            w = anno[3]/pot_w
            h = anno[4]/pot_h
            xmin = round(xmin, 4)
            ymin = round(ymin, 4)
            w = round(w, 4)
            h = round(h, 4)
            ele = f'{cls_idx} {xmin} {ymin} {w} {h}'
            annos.append(ele)
            # cv2.imwrite(os.path.join(output_dir, base + f"_{cnt}_{label}.jpg"), )

        
        write_list_to_txt(out_txt_filepath, annos)
        # if idx > 4:
        #     break




def convert_labelme_to_mask_and_cut_block(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    input_dir_img = '/home/hongrui/project/metro_pro/dataset/pot/pot_20220108'
    input_ano_dir = 'temp'
    
    output_dir = '/home/hongrui/project/metro_pro/dataset/pot/pot_20220108_cutblock'
    # output_dir = 'temp'

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    label_names = ['pot', 'LaSi_rect', 'TuQi', 'ZhouBian', 'HuaHen_rect', 'HuaHen', 'block', 'neg_block']

    

    # json_files = ['IMG_20220104_094754.json', 'IMG_20220104_101800.json']
    # json_files = os.listdir(input_dir)
    label_files = glob.glob(osp.join(input_ano_dir, "*.json"))
    random.shuffle(label_files)
    for idx, filename in enumerate(label_files):
        if not '.json' in filename:
            continue
        print(f"processing {idx}/{len(label_files)}, {filename}")
        # filepath = os.path.join(input_dir, filename)
        base = osp.splitext(osp.basename(filename))[0]
        # print(base)
        in_img_filepath = osp.join(input_dir_img, base + ".jpg")
        out_img_filepath = osp.join(output_dir, base + ".jpg")
        label_info = labelme.LabelFile(filename=filename)
        img = cv2.imread(in_img_filepath)

        masks = np.zeros(img.shape[:2], dtype=np.int64)  # for area
 
        # print(img.shape)

        for cnt, shape in enumerate(label_info.shapes):
            points = shape["points"]
            label = shape["label"]
            if not label in label_names:
                continue
            label_index = label_names.index(label)

            shape_type = shape.get("shape_type", "polygon")
            # print('shape_type', shape_type, 'label', label)
            mask = labelme.utils.shape_to_mask(
                img.shape[:2], points, shape_type
            )
            # print('mask', mask[mask == True])
            mask = mask.astype(np.uint8)
            if label == 'block':
                mask_t = np.asfortranarray(mask)
                mask_t = pycocotools.mask.encode(mask_t)
                # area = float(pycocotools.mask.area(mask_t))
                block_bbox = pycocotools.mask.toBbox(mask_t).flatten().tolist()
                # print('block_bbox', block_bbox)
                block_bbox = list(map(int, block_bbox))      

            elif label == 'neg_block':
                mask_t = np.asfortranarray(mask)
                mask_t = pycocotools.mask.encode(mask_t)
                # area = float(pycocotools.mask.area(mask_t))
                neg_block_bbox = pycocotools.mask.toBbox(mask_t).flatten().tolist()                
                neg_block_bbox = list(map(int, neg_block_bbox))      
                # print('neg_block_bbox', neg_block_bbox)        
            else:
                masks += mask*label_index
            # cv2.imwrite(os.path.join(output_dir, base + f"_{cnt}_{label}.jpg"), )
        masks *= 50
        if len(block_bbox):
            xmin, ymin, w, h = block_bbox
        else:
            xmin, ymin, w, h = neg_block_bbox
        
        img = multiple_img_with_binary_mask(img.copy(), mask)
        img = img[ymin:ymin+h, xmin:xmin+w]

        masks = multiple_img_with_binary_mask(masks.copy(), mask)
        # masks[mask<=0] = 255
        masks = masks[ymin:ymin+h, xmin:xmin+w]
        # print('img.shape', img.shape)
        cv2.imwrite(os.path.join(output_dir, base + ".jpg"), img)
        cv2.imwrite(os.path.join(output_dir, base + ".png"), masks)

        # masks_col = colorize_mask_to_bgr(masks)
        # cv2.imwrite(os.path.join(output_dir, base + f"_blend.jpg"), masks_col)

        # compose = 0.65*img + 0.35*masks_col
        # cv2.imwrite(os.path.join(output_dir, base + f"_compose.jpg"), compose.astype(np.uint8))

def convert_labelme_jsonfile():

    label_names = ['lasi_heavy', 'lasi_medium', 'lasi_slight',
        'gengshang_heavy', 'gengshang_medium', 'gengshang_slight',  
        'gengshi_heavy', 'gengshi_medium', 'gengshi_slight', ]


    in_img_dir = '/home/hongrui/project/metro_pro/dataset/pot/pot_20220108'
    ano_fore_dir = 'temp'
    ano_bk_dir = '/home/hongrui/project/metro_pro/dataset/pot/pot_20220108'

    
    output_dir = '/home/hongrui/project/metro_pro/dataset/pot/pot_20220108_obvious_defect_0/data'
    # output_dir = 'tempout2'

    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    label_filepaths = glob.glob(osp.join(ano_fore_dir, "*.json"))
    # random.shuffle(label_filepaths)
    for idx, filepath in enumerate(label_filepaths):
        if not '.json' in filepath:
            continue
        # if idx == 0:
        #     continue
        print(f"processing {idx}/{len(label_filepaths)}, {filepath}")
        # filepath = os.path.join(input_dir, filename)
        base = osp.splitext(osp.basename(filepath))[0]
        # print(base)
        in_img_filepath = osp.join(in_img_dir, base + ".jpg")
        out_img_filepath = osp.join(output_dir, base + ".jpg")
        fore_label_info = labelme.LabelFile(filename=filepath)
        bk_filepath = os.path.join(ano_bk_dir, base + ".json")
        bk_label_info = labelme.LabelFile(filename=bk_filepath)
        img = cv2.imread(in_img_filepath)

    
        for cnt, shape in enumerate(bk_label_info.shapes):
            points = shape["points"]
            label = shape["label"]
            if not label == 'pot':
                continue
            
            shape_type = shape.get("shape_type", "polygon")
            maskpot = labelme.utils.shape_to_mask(
                img.shape[:2], points, shape_type
            )
        
            mask_t = np.asfortranarray(maskpot.astype(np.uint8))
            mask_t = pycocotools.mask.encode(mask_t)
            # area = float(pycocotools.mask.area(mask_t))
            pot_bbox = pycocotools.mask.toBbox(mask_t).flatten().tolist()
            # print('pot_bbox', pot_bbox)
            pot_bbox = list(map(int, pot_bbox))     

        masks = np.zeros(img.shape[:2], dtype=np.uint8)  # for area
        # masks_t = np.zeros(img.shape[:2], dtype=np.uint8)  # for area
        # print('fore_label_info.shapes', fore_label_info.shapes)

        for cnt, shape in enumerate(fore_label_info.shapes):
            points = shape["points"]
            label = shape["label"]
            # print('label', label)
            if not label in label_names:
                continue
            label_index = label_names.index(label) 
            product = label_index // 3
            basis = label_index % 3 + 1
            multipler = product*10 + basis

            # print('label', label, 'multipler', multipler)
            shape_type = shape.get("shape_type", "polygon")
            # print('shape_type', shape_type, 'label', label)
            mask = labelme.utils.shape_to_mask(
                img.shape[:2], points, shape_type
            )
            # print('mask', mask[mask == True])
            mask = mask.astype(np.uint8)
            mask[mask > 0] = multipler
            masks += mask
            # cv2.imwrite(os.path.join(output_dir, base + f"_{cnt}_{label}.jpg"), )
        
        # masks[masks > 0] = 1
        height, width = masks.shape
        ynon, xnon = masks.nonzero()
        y_start, y_end = ynon.min(), ynon.max()
        x_start, x_end = xnon.min(), xnon.max()
        target_length = 640
        if y_end - y_start > target_length or x_end - x_start > target_length:
            offset_x, offset_y = 500, 500
        else:
            h, w = target_length, target_length
            ori_h = y_end - y_start
            ori_w = x_end - x_start
            offset_x = w - ori_w
            offset_y = h - ori_h
        xmin = x_start - offset_x // 2 if x_start - offset_x // 2 >= 0 else 0
        xmax = x_end + offset_x // 2 if x_end + offset_x // 2 <= width else width
        ymin = y_start - offset_y // 2 if y_start - offset_y // 2 >= 0 else 0
        ymax = y_end + offset_y // 2 if y_end + offset_x // 2 <= height else height
        h = ymax - ymin
        w = xmax - xmin

        img = multiple_img_with_binary_mask(img.copy(), maskpot)
        masks = multiple_img_with_binary_mask(masks.copy(), maskpot)

        masks[maskpot<=0] = 255

        img = img[ymin:ymin+h, xmin:xmin+w]
        masks = masks[ymin:ymin+h, xmin:xmin+w]

        # print('img.shape', img.shape)
        cv2.imwrite(os.path.join(output_dir, base + ".jpg"), img)
        cv2.imwrite(os.path.join(output_dir, base + ".png"), masks)
        # cv2.imwrite(os.path.join(output_dir, base + "_t.png"), masks_t)

        masks_col = colorize_mask_to_bgr(masks)
        cv2.imwrite(os.path.join(output_dir, base + f"_blend.jpg"), masks_col)

        compose = 0.65*img + 0.35*masks_col
        cv2.imwrite(os.path.join(output_dir, base + f"_compose.jpg"), compose.astype(np.uint8))
        print(img.shape)
        print()
        # if idx > 6:
        #     break


def newly_convert_labelme_jsonfile():
    
    label_dict = {'lasi_heavy': 11, 'lasi_medium':12, 'lasi_slight':13,
        'gengshang_heavy':21, 'gengshang_medium':22, 'gengshang_slight':23,  
        'gengshi_heavy':31, 'gengshi_medium':32, 'gengshi_slight':33,
        'shayan_heavy':41, 'shayan_medium':42, 'shayan_medium':43,
        'huahen_heavy':51, 'huahen_medium':52, 'huahen_medium':53,
        'zhoubian_heavy':61, 'zhoubian_medium':62, 'zhoubian_medium':63,
        'bowen_heavy':71, 'bowen_medium':72, 'bowen_medium':73,
        'youwu_heavy':81, 'youwu_medium':82, 'youwu_medium':83,
        }
    
    label_names = label_dict.keys()
    # 
    in_img_dir = '/home/hongrui/project/metro_pro/dataset/pot/20220108/image'
    in_anno_dir = '/home/hongrui/project/metro_pro/dataset/pot/20220108/json_3'

    # in_img_dir = '/home/hongrui/project/metro_pro/dataset/pot/20220222/image'
    # in_anno_dir = '/home/hongrui/project/metro_pro/dataset/pot/20220222/json'

    
    output_dir = '/home/hongrui/project/metro_pro/dataset/pot/0108_0222_obvious_defect_1/data'
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    label_filepaths = glob.glob(osp.join(in_anno_dir, "*.json"))
    # random.shuffle(label_filepaths)

    for idx, filepath in enumerate(label_filepaths):
        pot_bbox = []
        if not '.json' in filepath:
            continue
        # if idx == 0:
        #     continue
        # print(f"processing {idx}/{len(label_filepaths)}, {filepath}")
        # filepath = os.path.join(input_dir, filename)
        base = osp.splitext(osp.basename(filepath))[0]
        # print(base)
        in_img_filepath = osp.join(in_img_dir, base + ".jpg")
        out_img_filepath = osp.join(output_dir, base + ".jpg")
        label_info = labelme.LabelFile(filename=filepath)
        img = cv2.imread(in_img_filepath)

        masks = np.zeros(img.shape[:2], dtype=np.uint8)  # for area
        col_masks = np.zeros(img.shape[:2], dtype=np.uint8)  # for area
        
        for cnt, shape in enumerate(label_info.shapes):
            points = shape["points"]
            label_name = shape["label"]
            shape_type = shape.get("shape_type", "polygon")
            mask = labelme.utils.shape_to_mask(
                img.shape[:2], points, shape_type
            )
            
            if label_name == 'pot':
                maskpot = mask.copy()
                mask = np.asfortranarray(mask.astype(np.uint8))
                mask = pycocotools.mask.encode(mask)
                # area = float(pycocotools.mask.area(mask_t))
                pot_bbox = pycocotools.mask.toBbox(mask).flatten().tolist()
                # print('pot_bbox', pot_bbox)
                pot_bbox = list(map(int, pot_bbox))     

            elif label_name in label_names:
                col_masks[mask] = list(label_names).index(label_name) + 1
                mask = mask.astype(np.uint8)
                label_idx = label_dict[label_name]
                category_id = label_idx // 10
                defect_level = label_idx % 10
                
                mask[mask > 0] = label_idx
                masks += mask
                
        
        if len(pot_bbox) == 0:
            continue
        
        height, width = masks.shape
        ynon, xnon = masks.nonzero()
        y_start, y_end = ynon.min(), ynon.max()
        x_start, x_end = xnon.min(), xnon.max()
        target_length = 640
        if y_end - y_start > target_length or x_end - x_start > target_length:
            offset_x, offset_y = 500, 500
        else:
            h, w = target_length, target_length
            ori_h = y_end - y_start
            ori_w = x_end - x_start
            offset_x = w - ori_w
            offset_y = h - ori_h
        xmin = x_start - offset_x // 2 if x_start - offset_x // 2 >= 0 else 0
        xmax = x_end + offset_x // 2 if x_end + offset_x // 2 <= width else width
        ymin = y_start - offset_y // 2 if y_start - offset_y // 2 >= 0 else 0
        ymax = y_end + offset_y // 2 if y_end + offset_x // 2 <= height else height
        
        # pxmin, pymin, pw, ph = pot_bbox
        # pxmax = pxmin + pw
        # pymax = pymin + ph
        
        # xmin = xmin if xmin <= pxmin else pxmin
        # ymin = ymin if ymin <= pymin else pymin
        # xmax = xmax if xmax >= pxmax else pxmax
        # ymax = ymax if ymax >= pymax else pymax
        
        h = ymax - ymin
        w = xmax - xmin

        img = multiple_img_with_binary_mask(img.copy(), maskpot)
        masks = multiple_img_with_binary_mask(masks.copy(), maskpot)
        col_masks = multiple_img_with_binary_mask(col_masks.copy(), maskpot)

        masks[maskpot<=0] = 255
        col_masks[col_masks<=0] = 255
        

        img = img[ymin:ymin+h, xmin:xmin+w]
        masks = masks[ymin:ymin+h, xmin:xmin+w]
        col_masks = col_masks[ymin:ymin+h, xmin:xmin+w]

        # print('img.shape', img.shape)
        cv2.imwrite(os.path.join(output_dir, base + ".jpg"), img)
        cv2.imwrite(os.path.join(output_dir, base + ".png"), masks)
        # cv2.imwrite(os.path.join(output_dir, base + "_t.png"), masks_t)

        col_masks = colorize_mask_to_bgr(col_masks)
        cv2.imwrite(os.path.join(output_dir, base + f"_blend.jpg"), col_masks)

        compose = 0.65*img + 0.35*col_masks
        cv2.imwrite(os.path.join(output_dir, base + f"_compose.jpg"), compose.astype(np.uint8))
        print(f"processing {img.shape} {idx}/{len(label_filepaths)}, {filepath}")
        # print(img.shape)
        print()
        # if idx > 6:
        #     break
        

def print_label_names():
    input_ano_dir = '/home/hongrui/project/metro_pro/dataset/pot/20220222/json'
    
    label_files = glob.glob(osp.join(input_ano_dir, "*.json"))
    random.shuffle(label_files)
    label_names = []
    num = 0
    in_flag = False
    for idx, filename in enumerate(label_files):
        if not '.json' in filename:
            continue
        print(f"processing {idx}/{len(label_files)}, {filename}")
        label_info = labelme.LabelFile(filename=filename)
        for cnt, shape in enumerate(label_info.shapes):
            label_name = shape["label"]
            
            if not label_name in label_names:
                label_names.append(label_name)
            if 'medium' in label_name or 'heavy' in label_name:
                in_flag = True
            # print(label_name, in_flag)
        if in_flag:
            num += 1
            in_flag = False
        # print(num,'\n')
    # label_names = ['lasi_slight', 'gengshi_medium', 'gengshang_slight', 
    #         'lasi_heavy', 'lasi_medium', 'gengshi_heavy', 'gengshi_slight', 'gengshang_medium']
    label_names.sort()
    print(label_names)
    print(num)

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
    # convert_labelme_to_bbox(args)
    # convert_labelme_to_mask_and_cut_block(args)
    # print_label_names()
    # convert_labelme_jsonfile()
    newly_convert_labelme_jsonfile()