from __future__ import absolute_import

__version__ = "0.1.2"

from labelme2coco.labelme2coco import labelme2coco
from labelme2coco.labelme2coco import labelme2coco_custom


def convert(labelme_folder: str, save_json_path: str):
    """
    Args:
        labelme_folder: folder that contains labelme annotations and image files
        save_json_path: oath for coco json to be saved
    """
    labelme2coco(labelme_folder, save_json_path)


def convert_customer_dataset(data_folder: str, save_json_path: str):
    """
    Args:
        data_folder: folder that contains labelme annotations and image files
        save_json_path: oath for coco json to be saved
    """
    labelme2coco_custom(data_folder, save_json_path)
