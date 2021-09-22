# -*- coding: utf-8 -*-
# @Author: Yihao Chen
# @Date:   2021-08-18 14:11:25
# @Last Modified by:   Yihao Chen
# @Last Modified time: 2021-08-24 10:42:51
import time
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from train_worker import distTrainer
# from idea import build_module, Config, Runner
# from idea.utils import get_logger, random_seed
import os

def main_worker(gpu, ngpus_per_node, args):
    # print(f'calling {__file__}, {sys._getframe().f_lineno}')
    # return
    args.gpu = gpu #gpu idx, equal to local_rank
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu # euqal to global rank

        # return
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # print(f'calling {__file__}, {sys._getframe().f_lineno}')
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        args.master = True
    else:
        args.master = False

    if args.sync_bn is None:
        if args.cuda and args.rank > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
            'basicdataset': 50,
        }
        args.epochs = epoches[args.dataset.lower()]

    # if args.batch_size is None:
    #     args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
            'basicdataset': 0.01,

        }
        # args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size
        args.lr = lrs[args.dataset.lower()]# / (4 * len(args.gpu_ids)) * args.batch_size

    # print(f'calling {__file__}, {sys._getframe().f_lineno}')

    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    if args.master:
        print(args)
    torch.manual_seed(args.seed)
    # return
    # trainer = Trainer(args)
    trainer = distTrainer(args)
    print(f'rank {args.rank} Starting Epoch: {trainer.args.start_epoch}')
    print(f'rank {args.rank} Total Epoches: {trainer.args.epochs}')
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)
    if args.master:
        trainer.writer.close()





if __name__ == "__main__":
    pass
