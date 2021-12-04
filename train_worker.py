# -*- coding: utf-8 -*-
# @Last Modified by:   Hong Rui
import argparse
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys
import time
from scipy.special import softmax
import torch.distributed as dist

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator, EvaluatorForeground
from runx.logx import logx
from tools.util import *

# print(f'calling {__file__}, {sys._getframe().f_lineno}')

class distWorker(object):
    def __init__(self, args):
        self.args = args
        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        if self.args.master:
            # Define Tensorboard Summary
            self.summary = TensorboardSummary(self.saver.experiment_dir, args = args)
            self.writer = self.summary.create_summary()
        
        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)
        
        # Define network
        print(f'rank {args.rank} Define network')
        self.model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        if args.testValTrain > 1:#train
            # Define Optimizer
            train_params = [{'params': self.model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': self.model.get_10x_lr_params(), 'lr': args.lr * 10}]
            self.optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                        weight_decay=args.weight_decay, nesterov=args.nesterov)
            # Define lr scheduler
            self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                                args.epochs, len(self.train_loader), args = args)
        if args.testValTrain >= 0:#test and val
            # Define Criterion
            # whether to use class balanced weights
            if args.use_balanced_weights:
                classes_weights_path = os.path.join(Path.db_root_dir(args.dataset, args), args.dataset+f'_c{self.nclass}_classes_weights.npy')
                if os.path.isfile(classes_weights_path):
                    weight = np.load(classes_weights_path)
                else:
                    weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass, args,
                                                        classes_weights_path = classes_weights_path)
                weight = torch.from_numpy(weight.astype(np.float32))
            else:
                weight = None
            self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda, args = args).build_loss(type=args.loss_type)
            
            # Define Evaluator
            if self.args.infer_thresholds and (1 >= self.args.testValTrain >= 0 ):
                # self.evaluators = [Evaluator(self.nclass) for _ in range(len(self.args.infer_thresholds))]
                self.evaluators = [EvaluatorForeground(self.nclass) for _ in range(len(self.args.infer_thresholds))]
            else:
                self.evaluators = []
            # self.evaluator = Evaluator(self.nclass)
            self.evaluator = EvaluatorForeground(self.nclass)
            
        # Using cuda
        if args.distributed:
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu) 
                self.model = self.model.cuda(args.gpu)
                self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[args.gpu])
                # patch_replication_callback(self.model)
            else:
                model.cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                self.model = torch.nn.parallel.DistributedDataParallel(self.model)
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            self.model = self.model.cuda(args.gpu)
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            self.model = torch.nn.DataParallel(self.model).cuda()

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'" .format(args.resume))
            checkpoint = torch.load(args.resume, map_location=torch.device('cpu'))
            # model.load_state_dict(checkpoint["state_dict"])
            # 使用下面这种load方式会导致每个进程在GPU0多占用一部分显存，原因是默认load的位置是GPU0
            # checkpoint = torch.load("checkpoint.pth")

            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft and args.testValTrain > 1: #train
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0
        if -1 < args.testValTrain < 2 and self.args.master:
            task_id = args.resume.split('/')[-6]
            # self.saver.write_log_to_csv([''])
            # self.saver.write_log_to_csv([''])
            self.saver.write_log_to_csv([f'{task_id}', ''])

    def training(self, epoch):
        train_loss = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_iter_tr = len(self.train_loader)
        # if self.args.master:
        #     print(f'rank {self.args.rank} num_img_tr: {num_img_tr}')
        start = 0
        for i, sample in enumerate(tbar):
            if not i % (200//self.args.world_size) == 0 and self.args.debug:
                continue
            # print(f'rank {self.args.rank} dataload time {round(time.time() - start, 3)}')
            # start = time.time()
            image, target, _ = sample['image'], sample['label'], sample['img_name']
            # print('target', target.size(), image.size())
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            # start = time.time()
            output = self.model(image)
            # print(f'rank {self.args.rank} inference time {round(time.time() - start, 3)}')
            # start = time.time()
            
            loss = self.criterion(output, target)
            # print(f'rank {self.args.rank} loss calculation time {round(time.time() - start, 3)}')
            # start = time.time()
            loss.backward()
            # print(f'rank {self.args.rank} loss backward time {round(time.time() - start, 3)}')
            # start = time.time()
            self.optimizer.step()
            train_loss += loss.item()
            
            if self.args.master:
                tbar.set_description('Train loss: %.5f' % (train_loss / (i + 1)))
                self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_iter_tr * epoch)

                # # Show 10 * 3 inference results each epoch
                interval = num_iter_tr // 5 if num_iter_tr // 5 else 1
                if i % interval == 0:
                    global_step = i + num_iter_tr * epoch
                    self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step, 'train')
            
        if self.args.master:
            self.writer.add_scalar('train/total_loss_epoch', train_loss/num_iter_tr, epoch)
            # print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
            print('Loss: %.5f' % (train_loss/num_iter_tr))

        # if self.args.no_val and self.args.master:
        if self.args.testValTrain == 2 and self.args.master:
            # testValTrain == 2 only train
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)
        # start = time.time()

    def validation(self, epoch = 0):
        self.model.eval()
        self.evaluator.reset()
        self.reset_evaluators(self.evaluators)
        tbar = tqdm(self.val_loader, desc='\r')
        num_iter_val = len(self.val_loader)
        is_best = False
        
        val_loss = 0.0
        # return
        for i, sample in enumerate(tbar):
            if not i % 20 == 0 and self.args.debug:
                continue
            image, target, img_names = sample['image'], sample['label'], sample['img_name']
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)
            if self.args.testValTrain > 1:                    
                loss = self.criterion(output, target)
                val_loss += loss.item()
            if self.args.master and self.args.testValTrain > 1:
                tbar.set_description('Val loss: %.5f' % (val_loss / (i + 1)))
                interval = num_iter_val // 8 if num_iter_val // 8 else 2
                if i % interval == 0:
                    global_step = i + num_iter_val * epoch
                    self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step, 'val')
                    
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            ori_infer = pred.copy()
            ori_infer = softmax(ori_infer, axis=1)
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)
            if self.args.infer_thresholds and self.evaluators:
                for j in range(len(self.evaluators)):
                    thres = self.args.infer_thresholds[j]
                    mask_by_thres = self.sel_ch_based_on_threshold(ori_infer.copy(), thres)
                    mask_by_thres[mask_by_thres == 3] = pred[mask_by_thres == 3]
                    # print('mask_by_thres.max', mask_by_thres.max(), mask_by_thres[mask_by_thres==3])
                    # print('target.dtype', target.dtype, target.shape, 'mask_by_thres.dtype', mask_by_thres.dtype, mask_by_thres.shape)
                    self.evaluators[j].add_batch(target, mask_by_thres)

            if self.args.master:
                if self.args.dump_raw_prediction or self.args.dump_image_for_cal_chamferDist:
                    self.dump_thre_pre_gt(ori_infer.copy(), target, img_names, output_mask_dir = os.path.join(self.saver.output_mask_dir, 'val'))

                if self.args.dump_image and len(image)*i < 200:
                    # self.dump_argmax_pre_gt_img(image, pred, target, img_names, output_mask_dir = self.saver.output_mask_dir)
                    self.dump_composed_img_pre_label(image, pred, target, img_names, output_mask_dir = self.saver.output_mask_dir)

        if self.args.dump_image_for_cal_chamferDist:
            return
        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        
        # FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        # global_mIoU = self.get_global_mIoU(mIoU)
        global_mIoU = self.reduce_tensor(mIoU)
        # print(f'test/rank@{self.args.rank} mIoU: {mIoU}, global_mIoU: {global_mIoU}')
        if self.args.master:
            if self.args.testValTrain > 1: # during training
                self.writer.add_scalar('val/total_loss_epoch', val_loss/num_iter_val, epoch)
                self.writer.add_scalar('val/mIoU', mIoU, epoch)
                self.writer.add_scalar('val/Acc', Acc, epoch)
                self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
                # self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
                self.writer.add_scalar('val/global_mIoU', global_mIoU, epoch)
                print('Validation:')
                print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
                print("Acc:{}, Acc_class:{}, mIoU:{}, global_mIoU: {}".format(Acc, Acc_class, mIoU, global_mIoU))
                print('Loss: %.5f' % (val_loss/num_iter_val))
                self.saver.write_log_to_txt("Epoch: {}, Val, Acc:{}, Acc_class:{}, mIoU:{}, global_mIoU: {}".format(epoch, Acc, Acc_class, mIoU, global_mIoU) + '\n')

                if global_mIoU > self.best_pred:
                    is_best = True
                    self.saver.write_log_to_txt("Best Epoch: {}, Val, Acc:{}, Acc_class:{}, mIoU:{}, global_mIoU: {}".format(epoch, Acc, Acc_class, mIoU, global_mIoU) + '\n')
                    self.best_pred = global_mIoU
                self.saver.save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.model.module.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'best_pred': self.best_pred,
                }, is_best)
            elif self.args.testValTrain >= 0:
                self.saver.write_log_to_txt(f'val/mIoU@argmax: {global_mIoU}\n')
                self.saver.write_log_to_csv([f'val/mIoU@argmax', f'{global_mIoU}'])

                if self.args.infer_thresholds:
                    for i in range(len(self.args.infer_thresholds)):
                        global_mIoU = self.evaluators[i].Mean_Intersection_over_Union()
                        # global_mIoU = self.reduce_tensor(global_mIoU)
                        # self.saver.write_log_to_txt(f'val/mIoU@thres_{self.args.infer_thresholds[i]}: {global_mIoU}')
                        self.saver.write_log_to_csv([f'val/mIoU@thres_{self.args.infer_thresholds[i]}', f'{global_mIoU}'])
                    self.saver.write_log_to_csv([''])
                self.saver.write_log_to_txt('\n')

        del val_loss

    def test(self, epoch = 0):
        self.model.eval()
        if self.args.testValTrain >= 0:
            self.evaluator.reset()
        self.reset_evaluators(self.evaluators)
        tbar = tqdm(self.test_loader, desc='\r')
        test_loss = 0.0
        # return
        # label_normalize_unit = 20
        num_iter_test = len(self.test_loader)
        # print('num_img_tr', num_img_tr)
        for i, sample in enumerate(tbar):
            if not i % 20 == 0 and self.args.debug:
                continue
            image, target, img_names = sample['image'], sample['label'], sample['img_name']
            if self.args.sync_single_pair_rail and not target.numpy().any() > 0:                
                continue
            if self.args.cuda:
                image, target = image.cuda(), target.cuda()
            with torch.no_grad():
                output = self.model(image)

            if self.args.master and self.args.testValTrain > 1: # only in training mode
                interval = num_iter_test // 5 if num_iter_test // 5 else 1
                if i % interval == 0:
                    global_step = i + num_iter_test * epoch
                    self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step, 'test')
            
            infer = output.data.cpu().numpy()
            ori_infer = infer.copy()
            ori_infer = softmax(ori_infer, axis=1)
            pred = np.argmax(infer, axis=1)
            if self.args.testValTrain >= 0:
                # if self.args.testValTrain > 1:
                    # loss = self.criterion(output, target)
                    # test_loss += loss.item()
                    # if self.args.master:
                    #     tbar.set_description('Test loss: %.5f' % (test_loss / (i + 1)))
            
                target = target.cpu().numpy()
                # Add batch sample into evaluator
                self.evaluator.add_batch(target, pred)
                # print('')
                if self.args.infer_thresholds and self.evaluators:
                    for j in range(len(self.evaluators)):
                        thres = self.args.infer_thresholds[j]
                        mask_by_thres = self.sel_ch_based_on_threshold(ori_infer.copy(), thres)
                        mask_by_thres[mask_by_thres == 3] = pred[mask_by_thres == 3]
                        self.evaluators[j].add_batch(target, mask_by_thres)
                # print('end')
            if self.args.master:
                if self.args.dump_raw_prediction or self.args.dump_image_for_cal_chamferDist:
                    self.dump_thre_pre_gt(ori_infer.copy(), target, img_names, output_mask_dir = os.path.join(self.saver.output_mask_dir, 'test'))

                if self.args.dump_image and len(image)*i < 200:
                    # self.dump_argmax_pre_gt_img(image, pred, target, img_names, output_mask_dir = self.saver.output_mask_dir)
                    self.dump_composed_img_pre_label(image, pred, target, img_names, output_mask_dir = self.saver.output_mask_dir)
                    
        if self.args.dump_image_for_cal_chamferDist:
            return
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        # FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        # global_mIoU = self.get_global_mIoU(mIoU)
        global_mIoU = self.reduce_tensor(mIoU)
        if self.args.master:
            if self.args.testValTrain > 1: # only in training mode
                # self.writer.add_scalar('test/total_loss_epoch', test_loss/num_iter_test, epoch)
                self.writer.add_scalar('test/mIoU', mIoU, epoch)
                self.writer.add_scalar('test/global_mIoU', global_mIoU, epoch)
                self.writer.add_scalar('test/Acc', Acc, epoch)
                self.writer.add_scalar('test/Acc_class', Acc_class, epoch)
                # self.writer.add_scalar('test/fwIoU', FWIoU, epoch)
                print('Test:')
                print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
                print("Acc:{}, Acc_class:{}, mIoU:{}, global_mIoU: {}".format(Acc, Acc_class, mIoU, global_mIoU))
                # print('Loss: %.5f' % (test_loss/num_iter_test))
                self.saver.write_log_to_txt("Epoch: {}, Tes, Acc:{}, Acc_class:{}, mIoU:{}, global_mIoU: {}".format(epoch, Acc, Acc_class, mIoU, global_mIoU) + '\n')
            elif self.args.testValTrain >= 0:
                self.saver.write_log_to_txt(f'test/mIoU@argmax: {global_mIoU}\n')
                self.saver.write_log_to_csv([f'test/mIoU@argmax', f'{global_mIoU}'])
                if self.args.infer_thresholds:
                    for i in range(len(self.args.infer_thresholds)):
                        global_mIoU = self.evaluators[i].Mean_Intersection_over_Union()
                        # global_mIoU = self.reduce_tensor(global_mIoU)
                        self.saver.write_log_to_csv([f'test/mIoU@thres_{self.args.infer_thresholds[i]}', f'{global_mIoU}'])
                    self.saver.write_log_to_csv([''])
                    self.saver.write_log_to_csv([''])
                    
    def postprocess(self, img):
        # max_id = img.max()
        # ratio = 255//max_id
        # img *= ratio
        img *= 40
        return img

    def sel_ch_based_on_threshold(self, pre, thres):
        # print('pre.ndim', pre.ndim)
        # res = np.zeros(pre.shape[1:])
        bt, ch, h, w = pre.shape
        for i in range(ch):
            p = pre[:,i]
            p[p < thres] = 0
            p[p >= thres] = i
        res = np.sum(pre, axis=1)
        # print('res.shape', res.shape)
        # print(res)
        return res.astype(np.int64)

    def get_global_mIoU(self, mIoU):
        mIoU_t = torch.tensor(mIoU).cuda()
        gather_t = [torch.ones_like(mIoU_t)] * self.args.world_size
        dist.all_gather(gather_t, mIoU_t)
        global_mIoU = 0
        # for id_ in range(self.args.world_size):
        #     global_mIoU += gather_t[id_].item()
        # print('gather_t', gather_t)
        inputs = torch.stack(gather_t, dim=0)
        global_mIoU = torch.sum(inputs, dim=0).item()
        global_mIoU /= self.args.world_size
        global_mIoU = round(global_mIoU, 4)
        return global_mIoU

    def reduce_tensor(self, mIoU):
        mIoU_t = torch.tensor(mIoU).cuda()
        # mIoU_t = mIoU.clone()
        dist.all_reduce(mIoU_t, op=dist.ReduceOp.SUM)
        global_mIoU = mIoU_t.item() / self.args.world_size
        # print('global_mIoU', global_mIoU)
        global_mIoU = round(global_mIoU, 4)
        return global_mIoU

    def reset_evaluators(self, evaluators):
        if evaluators:
            for i in range(len(evaluators)):
                evaluators[i].reset()

    def dump_argmax_pre_gt_img(self, images, pres, GTs, img_names, output_mask_dir = None):
        results = self.postprocess(pres.copy())
        if isinstance(GTs, np.ndarray):                    
            labels = GTs.copy()
        elif isinstance(GTs, torch.Tensor): 
            labels = GTs.cpu().numpy()
        else:
            pass
        if isinstance(images, np.ndarray):                    
            images = images.copy()
        elif isinstance(images, torch.Tensor): 
            images = images.cpu().numpy()
        else:
            pass
        labels = self.postprocess(labels)
        # print('len(image)', len(image))
        for _id in range(len(images)):
            img_tmp = np.transpose(images[_id].cpu().numpy(), axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp[:,:,::-1]
            img_tmp = img_tmp.astype(np.uint8)
            img_name = img_names[_id]
            out_img_filepath = os.path.join(output_mask_dir, img_name)
            cv2.imwrite(out_img_filepath, img_tmp)

            infer_mask_name = f"{img_name.split('.')[0]}_infer.png"                    
            out_infer_mask_filepath = os.path.join(output_mask_dir, infer_mask_name)
            cv2.imwrite(out_infer_mask_filepath, results[_id])
            
            # if self.args.testValTrain >= 1:
            if labels[_id].any() > 0:
                label_name = f"{img_name.split('.')[0]}_GT.png"
                out_label_filepath = os.path.join(output_mask_dir, label_name)
                cv2.imwrite(out_label_filepath, labels[_id])

    def dump_thre_pre_gt(self, pres, GTs, img_names, images = None, output_mask_dir = None):
        if not os.path.exists(output_mask_dir):
            os.makedirs(output_mask_dir)
        # print('pres.shape', pres.shape)
        raw_pre = np.transpose(pres.copy(), axes=[0, 2, 3, 1])

        if isinstance(GTs, np.ndarray):                    
            labels = GTs.copy()
        elif isinstance(GTs, torch.Tensor): 
            labels = GTs.cpu().numpy()
        else:
            pass
        labels = self.postprocess(labels)
        
        # print('raw_pre.shape', raw_pre.shape)
        assert raw_pre.shape[-1] == 3
        for _id in range(len(raw_pre)):
            img_name = img_names[_id]
            infer_mask_name = f"{img_name.split('.')[0]}_infer.png"                    
            out_infer_mask_filepath = os.path.join(output_mask_dir, infer_mask_name)
            cv2.imwrite(out_infer_mask_filepath, (255*raw_pre[_id]).astype(np.uint8))

            label_name = f"{img_name.split('.')[0]}_GT.png"
            out_label_filepath = os.path.join(output_mask_dir, label_name)
            cv2.imwrite(out_label_filepath, labels[_id])

    def dump_composed_img_pre_label(self, images, pres, GTs, img_names, output_mask_dir = None):
        if isinstance(GTs, np.ndarray):                    
            labels = GTs.copy()
        elif isinstance(GTs, torch.Tensor): 
            labels = GTs.cpu().numpy()
        else:
            pass
        if isinstance(images, np.ndarray):                    
            images = images.copy()
        elif isinstance(images, torch.Tensor): 
            images = images.cpu().numpy()
        else:
            pass
        if isinstance(pres, np.ndarray):                    
            preds = pres.copy()
        elif isinstance(pres, torch.Tensor): 
            preds = pres.cpu().numpy()
        else:
            pass
        # print('len(image)', len(image))
        for _id in range(len(images)):
            img = images[_id]
            label = labels[_id]
            pre = preds[_id]
            img_tmp = np.transpose(img, axes=[1, 2, 0])
            img_tmp *= (0.229, 0.224, 0.225)
            img_tmp += (0.485, 0.456, 0.406)
            img_tmp *= 255.0
            img_tmp = img_tmp[:,:,::-1]
            img_tmp = img_tmp.astype(np.uint8)
            img_name = img_names[_id]
            composed = compose_img_label_pre(img_tmp, label, pre)
            out_img_filepath = os.path.join(output_mask_dir, img_name)
            cv2.imwrite(out_img_filepath, composed)

def plot_image():
    print('call plot image fun')
    img = cv2.imread('images/composited_sf.png')
    print('img shape', img.shape)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    plt.show()
    print('end')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test-batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
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
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
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

    args = parser.parse_args()
    
    # plot_image()