# -*- coding: utf-8 -*-
# @Author: Hong Rui
# @Date:   2021-09-16 16:53:03
# @Last Modified by:   Hong Rui
import argparse
import random
import shutil
import time
import warnings
import sys
from datetime import datetime
import socket

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import os

from train_engine import main_worker

parser = argparse.ArgumentParser(description='IDEA Training')

parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-file', default='dist_file', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--work_dirs', default="log", type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
# parser.add_argument('--seed', default=-1, type=int,
#                     help='random seed')
parser.add_argument('--multiprocessing-distributed', action='store_true', default=True,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
parser.add_argument('--out-stride', type=int, default=8,
                    help='network output stride (default: 8)')

parser.add_argument('--use-sbd', action='store_true', default=False,
                    help='whether to use SBD dataset (default: False)')

parser.add_argument('--sync-bn', type=bool, default=False,
                    help='whether to use sync bn (default: auto)')
parser.add_argument('--freeze-bn', type=bool, default=False,
                    help='whether to freeze bn parameters (default: False)')


# optimizer params
parser.add_argument('--lr', type=float, default=None, metavar='LR',
                    help='learning rate (default: auto)')
parser.add_argument('--lr-scheduler', type=str, default='poly',
                    choices=['poly', 'step', 'cos'],
                    help='lr scheduler mode: (default: poly)')
parser.add_argument('--momentum', type=float, default=0.9,
                    metavar='M', help='momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=5e-4,
                    metavar='M', help='w-decay (default: 5e-4)')
parser.add_argument('--nesterov', action='store_true', default=False,
                    help='whether use nesterov (default: False)')
# cuda, seed and logging
parser.add_argument('--no-cuda', action='store_true', default=
                    False, help='disables CUDA training')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training. ')
# checking point
parser.add_argument('--resume', type=str, default=None,
                    help='put the path to resuming file if needed')
parser.add_argument('--checkname', type=str, default=None,
                    help='set the checkpoint name')
# finetuning pre-trained models
parser.add_argument('--ft', action='store_true', default=False,
                    help='finetuning on a different dataset')
# evaluation option
parser.add_argument('--eval-interval', type=int, default=1,
                    help='evaluuation interval (default: 1)')
parser.add_argument('--no-val', action='store_true', default=False,
                    help='skip validation during training')


# training hyper params
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: auto)')
parser.add_argument('--start_epoch', type=int, default=0,
                    metavar='N', help='start epochs (default:0)')
parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                    help='whether to use balanced weights (default: False)')

parser.add_argument('--loss_type', type=str, default='focal',
                    choices=['ce', 'focal', 'FSOhemCELoss'],
                    help='loss func type (default: focal)')
parser.add_argument('--batch-size', type=int, default=8,
                    metavar='N', help='input batch size for \
                            training (default: auto)')
parser.add_argument('--test-batch-size', type=int, default=8,
                    metavar='N', help='input batch size for \
                            testing (default: auto)')
parser.add_argument('--workers', type=int, default=4,
                    metavar='N', help='dataloader threads')
parser.add_argument('--hw_ratio', type=float, default=1.25)
parser.add_argument('--ignore_index', type=int, default=255)
parser.add_argument('--base_size', type=int, default=640)
parser.add_argument('--crop_size', type=int, default=640)
parser.add_argument('--max_size', type=int, default=1080)
parser.add_argument('--rotate_degree', type=int, default=15)
parser.add_argument('--n_classes', type=int, default=2)
parser.add_argument('--dataset', type=str, default='basicDataset')
parser.add_argument('--dataset_dir', type=str, default='/home/hongrui/project/metro_pro/dataset/1st_2000', help='dataset dir')
parser.add_argument('--testValTrain', type=int, default=-1, help='-1: no, 0: infer, 1: test, 2: train, 3: trainval, 4: trainvaltest')
parser.add_argument('--testset_dir', type=str, default=None, help='input test or inference image dir')
parser.add_argument('--testOut_dir', type=str, default=None, help='test image output dir')
parser.add_argument('--dump_image', action='store_true', default=False,
                    help='dump image when test')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='focal loss alpha')
args = parser.parse_args()


def find_free_port():
    import socket
    s = socket.socket()
    s.bind(('', 0))            # Bind to a free port provided by the host.
    return s.getsockname()[1]  # Return the port number assigned.

    
def main(args):
    import os

    if args.testValTrain == 0:
        print('only inference')
    elif args.testValTrain == 1:
        print('test (calculate metrics)')
    elif args.testValTrain == 2:
        print('only training')
    elif args.testValTrain == 3:
        print('train and val')
    elif args.testValTrain == 4:
        print('train, val, and test')
    else:
        print('please select a mode (0: infer, 1: test, 2: train, 3: trainval, 4: trainvaltest)')
        return
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    
    if args.world_size == -1 and "SLURM_NPROCS" in os.environ:
        args.world_size = int(os.environ["SLURM_NPROCS"])
        args.rank = int(os.environ["SLURM_PROCID"])
        jobid = os.environ["SLURM_JOBID"]
    else:
        args.world_size = 1
        args.rank = 0
        jobid = f'{random.randint(1,20)}'
    hostfile = "dist_url." + jobid  + ".txt"
    if os.path.exists(hostfile):
        os.remove(hostfile)
    if args.dist_file is not None:
        url_file = f"{os.path.realpath(args.dist_file)}.{jobid}"
        if os.path.exists(url_file):
            os.remove(url_file)
        args.dist_url = f"file://{url_file}"
    elif args.rank == 0:
        import socket
        ip = socket.gethostbyname(socket.gethostname())
        port = find_free_port()
        args.dist_url = "tcp://{}:{}".format(ip, port)
        with open(hostfile, "w") as f:
            f.write(args.dist_url)
    else:
        import os
        import time
        while not os.path.exists(hostfile):
            time.sleep(1)
        with open(hostfile, "r") as f:
            args.dist_url = f.read()
    print("dist-url:{} at PROCID {} / {}".format(args.dist_url, args.rank, args.world_size))
    # if args.dist_url == "env://" and args.world_size == -1:
    #     args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    args.ngpus_per_node = ngpus_per_node
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        # print(f'calling {__file__}, {sys._getframe().f_lineno}')
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)
    
if __name__ == '__main__':
    main(args)

