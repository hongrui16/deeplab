import os
import shutil
import torch
from collections import OrderedDict
import glob

class Saver(object):

    def __init__(self, args):
        self.args = args
        self.output_mask_dir = None
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
                    checkpoint_file = self.args.resume.split('/')[-1]
                    start = self.args.resume.find(checkpoint_file)
                    self.directory = self.args.resume[:start-1] 
                # print('self.directory', self.directory)
                self.runs = sorted(glob.glob(os.path.join(self.directory, 'testResult_*')))
                run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0
                self.experiment_dir = os.path.join(self.directory, 'testResult_{}'.format(str(run_id)))
                self.output_mask_dir = os.path.join(self.experiment_dir, 'infer_mask')
        else:
            return

        self.logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        # print('self.args.rank', self.args.rank)
        if not self.args.master:
            return
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        if self.output_mask_dir and not os.path.exists(self.output_mask_dir):
            os.makedirs(self.output_mask_dir)
        if os.path.exists(self.logfile):
            os.remove(self.logfile)

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
                    shutil.copyfile(filename, os.path.join(self.experiment_dir, 'model_best.pth.tar'))
            else:
                shutil.copyfile(filename, os.path.join(self.experiment_dir, 'model_best.pth.tar'))

    def save_experiment_config(self):
        if not self.args.master:
            return
        p=vars(self.args)
        log_file = open(self.logfile, "a+")
        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.write('\n')
        log_file.close()# 
        # return log_file
        # logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        # self.args.log_file = open(logfile, 'w')
        # for key, val in p.items():
        #     self.args.log_file.write(key + ':' + str(val) + '\n')
        # self.args.log_file.write('\n')

    def write_log_to_txt(self, data):
        assert isinstance(data, str)
        if not self.args.master:
            return
        log_file = open(self.logfile, "a+")
        if data.endswith('\n'):
            log_file.write(data)
        else:
            log_file.write(data+'\n')
        log_file.close()# 

