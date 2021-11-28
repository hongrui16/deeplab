from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import cv2
import numpy as np
import random
import os

from .gen_utils import get_bbox, iou, get_random_shape

    
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


class CocoGen():
    def __init__(self, coco_root, data_type, anno_file=None, size=112):
        if anno_file is None:
            ann_file = os.path.join(coco_root, f'annotations/instances_{data_type}.json')
        
        self.coco_root = coco_root
        self.data_type = data_type
        self.size = 112
    
        self.coco = COCO(ann_file)

        self.cat_ids = self.coco.getCatIds()
        categories = self.coco.loadCats(self.cat_ids)
        #print(categories)

        self.cat_names = dict()
        for cat in categories:
            cat_id = cat['id']
            cat_name = cat['name']

            self.cat_names[cat_id] = cat_name 

        #self.cat_names = [x['name'] for x in categories]




    def get_anno(self, cat_id=None ):
        if cat_id is None:
            cat_ids = self.coco.getCatIds()
            cat_id = random.choice(cat_ids)

        anno_ids = self.coco.getAnnIds(catIds=[cat_id])
        chosen_id = random.choice(anno_ids)
        #print(f"chosen anno id: {chosen_id}.")

        anno = self.coco.loadAnns([chosen_id])[0]

        return anno
    
    def get_crop_img(self, cat_id = None):
        anno = self.get_anno(cat_id)
        cat_name = self.cat_names[anno['category_id']]
        mask = self.coco.annToMask(anno)
        image_name = str(anno['image_id']).zfill(12)
        img_path = os.path.join(self.coco_root, self.data_type, f"{image_name}.jpg")
        img = cv2.imread(img_path)

        x, y, w, h = [int(i) for i in anno['bbox']]
        
        img = draw_one_box_cls(img, [x,y, x+w, y+h], cat_name)

        return img

        
    
    def get_masked_img(self, cat_id = None, min_pixel=20*20):
    
        while True:
            anno = self.get_anno(cat_id)
            if anno['area'] >= min_pixel:
                break
        
        # cat_name = self.cat_names[anno['category_id']]
        mask = self.coco.annToMask(anno)
        image_name = str(anno['image_id']).zfill(12)
        img_path = os.path.join(self.coco_root, self.data_type, f"{image_name}.jpg")
        img = cv2.imread(img_path)

        x, y, w, h = [int(i) for i in anno['bbox']]
        
        crop_img = img[y:y+h, x:x+w]
        mask = mask[y:y+h, x:x+w]
        
        #crop_img[mask==0] = 255
        
        #compute bounding box
        '''
        hor = np.sum(mask, axis=0)
        hor_idx = np.nonzero(hor)[0]
        x = hor_idx[0]
        width = hor_idx[-1] - x + 1
        vert = np.sum(mask, axis=1)
        vert_idx = np.nonzero(vert)[0]
        y = vert_idx[0]
        height = vert_idx[-1] - y + 1
        bbox = [int(x), int(y), int(width), int(height)]
        '''
        return crop_img, mask


    def get_img_block(self, cat_id = None, min_pixel=20*20):
    
        while True:
            anno = self.get_anno(cat_id)
            if anno['area'] >= min_pixel:
                break
        
        # cat_name = self.cat_names[anno['category_id']]
        mask = self.coco.annToMask(anno)
        image_name = str(anno['image_id']).zfill(12)
        img_path = os.path.join(self.coco_root, self.data_type, f"{image_name}.jpg")
        img = cv2.imread(img_path)

        x, y, w, h = [int(i) for i in anno['bbox']]
        
        crop_img = img[y:y+h, x:x+w]
        
        return crop_img


    def get_negative_sample(self, cat_id = None):
        anno = self.get_anno(cat_id)

        cat_name = self.cat_names[anno['category_id']]
        mask = self.coco.annToMask(anno)
        image_name = str(anno['image_id']).zfill(12)
        img_path = os.path.join(self.coco_root, self.data_type, f"{image_name}.jpg")
        img = cv2.imread(img_path)
        im_h, im_w, _ = img.shape

        x, y, w, h = [int(i) for i in anno['bbox']]


         
        feasible = False
        '''
        while not feasible:
            #generate random position for top left point
            bg_y = random.randint(0, im_h-self.size)
            bg_x = random.randint(0, im_w-self.size)

            boxA = [x, y, x+w, y+h]
            boxB = [bg_x, bg_y, bg_x+self.size, bg_y+self.size]
            if iou( boxA, boxB) <=0:
                feasible = True
        '''
        patch_y = random.randint(0, im_h-self.size)
        patch_x = random.randint(0, im_w-self.size)
        
        patch = img[patch_y:patch_y+self.size,  patch_x:patch_x+self.size]

        return patch
