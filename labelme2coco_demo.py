import argparse
import sys
# from tools.custom_train import instance_custom_training
from pixellib import instance_custom_training

def pixellib_vis(args):
    input_dir   = args.inp


    vis_img = instance_custom_training()
    # vis_img = instance_custom_dataset_model_training()

    # vis_img.load_dataset(input_dir)
    vis_img.load_customer_dataset(input_dir)

    # vis_img.load_dataset("Nature")
    vis_img.visualize_sample()


if __name__ == '__main__':
    # from dataloaders.utils import decode_segmap


    parser = argparse.ArgumentParser()
    parser.add_argument('--inp', type=str, default=None)

    parser.add_argument('-im', '--input_dir', type=str, default='/home/hongrui/project/metro_pro/dataset/1st_2000')
    parser.add_argument('-om', '--output_dir', type=str, default='temp')
    
    args = parser.parse_args()
    pixellib_vis(args)