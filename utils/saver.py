import os
import shutil
import torch
from collections import OrderedDict
import glob

class Saver(object):

    def __init__(self, args):
        self.args = args
        if self.args.testValTrain > 1:
            self.directory = os.path.join('run', args.dataset, args.checkname)
            self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')))
            run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0
            self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
        elif 0<=self.args.testValTrain<=1 and self.args.resume:
            if self.args.testOut_dir:
                self.experiment_dir = self.args.testOut_dir
            else:
                if not '/' in self.args.resume:
                    self.directory = f'./'
                else:
                    self.directory = self.args.resume[self.args.resume.find(self.args.resume.split('/')[-1])-1] 
                self.runs = sorted(glob.glob(os.path.join(self.directory, 'testResult_*')))
                run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0
                self.experiment_dir = os.path.join(self.directory, 'testResult_{}'.format(str(run_id)))
        else:
            return
        # print('self.args.rank', self.args.rank)
        if not self.args.master:
            return
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        if not self.args.master:
            return
        filename = os.path.join(self.experiment_dir, filename)
        torch.save(state, filename)
        if is_best:
            best_pred = state['best_pred']
            with open(os.path.join(self.experiment_dir, 'best_pred.txt'), 'w') as f:
                f.write(str(best_pred))
            if self.runs:
                previous_miou = [0.0]
                for run in self.runs:
                    run_id = run.split('_')[-1]
                    path = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)), 'best_pred.txt')
                    if os.path.exists(path):
                        with open(path, 'r') as f:
                            miou = float(f.readline())
                            previous_miou.append(miou)
                    else:
                        continue
                max_miou = max(previous_miou)
                if best_pred > max_miou:
                    shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.directory, 'model_best.pth.tar'))

    def save_experiment_config(self):
        if not self.args.master:
            return
        logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        log_file = open(logfile, 'w')
        # p = OrderedDict()
        # p['datset'] = self.args.dataset
        # p['backbone'] = self.args.backbone
        # p['out_stride'] = self.args.out_stride
        # p['lr'] = self.args.lr
        # p['lr_scheduler'] = self.args.lr_scheduler
        # p['loss_type'] = self.args.loss_type
        # p['epoch'] = self.args.epochs
        # p['base_size'] = self.args.base_size
        # p['crop_size'] = self.args.crop_size
        p=vars(self.args)
        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.close()