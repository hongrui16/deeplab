# -*- coding: utf-8 -*-
# @Author: Hong Rui
# @Date:   2021-09-16 16:53:03
# @Last Modified by:   Hong Rui
import time
import sys
import torch
import torch.nn as nn
import torch.distributed as dist
from train_worker import distWorker
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
    # print(f'rank: {args.rank}, gpu: {gpu}')
    print(f'calling {__file__}, {sys._getframe().f_lineno}, rank: {args.rank}, gpu: {gpu}')

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        args.master = True
    else:
        args.master = False
    args.cuda = not args.no_cuda and torch.cuda.is_available()


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

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
            'basicdataset': 0.01,

        }
        args.lr = lrs[args.dataset.lower()]

    if args.checkname is None:
        args.checkname = 'deeplab-'+str(args.backbone)
    if args.master:
        print(args)
    torch.manual_seed(args.seed)
    # return
    worker = distWorker(args)
    # print('args.log_file', args.log_file)
    # if args.master:
    #     args.log_file.write('\n')

    # return
    print(f'rank {args.rank} Starting Epoch: {worker.args.start_epoch}, Total Epoches: {worker.args.epochs}')
    if args.testValTrain >= 2:
        for epoch in range(worker.args.start_epoch, worker.args.epochs):
            worker.training(epoch)
            # if not worker.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            if args.testValTrain > 2 and epoch % args.eval_interval == (args.eval_interval - 1):
                worker.validation(epoch)
            if args.testValTrain == 4 and epoch % 5 == 0:
                worker.test(epoch)
            if args.master:
                worker.saver.write_log_to_txt('\n')
    elif args.testValTrain >= 0:
        worker.validation()
        worker.test()
    elif args.testValTrain >= -1:
        worker.test()
    else:
        print('error, please specify a mode')
    if args.master:
        worker.writer.close()




if __name__ == "__main__":
    pass
