import os
import shutil
import torch
import glob
from time import gmtime, strftime
import csv
import pytz
import datetime

class Saver(object):

    def __init__(self, args):
        self.args = args
        self.output_mask_dir = None
        self.best_model_filepath = None
        if self.args.testValTrain > 1:
            self.directory = os.path.join('run', args.dataset, args.checkname)
            self.runs = sorted(glob.glob(os.path.join(self.directory, 'experiment_*')), key=lambda x: int(x.split('_')[-1]))
            run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0
            self.experiment_dir = os.path.join(self.directory, 'experiment_{}'.format(str(run_id)))
            
        elif 0<=self.args.testValTrain<=1 and self.args.resume:
            if self.args.testOut_dir:
                self.directory = self.args.testOut_dir
            else:
                if not '/' in self.args.resume:
                    self.directory = f'./'
                else:
                    checkpoint_file = self.args.resume.split('/')[-1]
                    start = self.args.resume.find(checkpoint_file)
                    self.directory = self.args.resume[:start-1] 
                # print('self.directory', self.directory)
            self.runs = sorted(glob.glob(os.path.join(self.directory, 'testResult_*')), key=lambda x: int(x.split('_')[-1]))
            # self.runs = sorted(glob.glob(os.path.join(self.directory, 'testResult_*')))
            run_id = int(self.runs[-1].split('_')[-1]) + 1 if self.runs else 0
            self.experiment_dir = os.path.join(self.directory, 'testResult_{}'.format(str(run_id)))
            self.output_mask_dir = os.path.join(self.experiment_dir, 'infer_mask')
        else:
            return

        self.logfile = os.path.join(self.experiment_dir, 'parameters.txt')
        
        self.csvlogfile = os.path.join(self.experiment_dir, 'log.csv')
        # self.csvlogfile = 'log.csv'
       
        self.lossfile = 'loss.txt'
        # print('self.args.rank', self.args.rank)
        if not self.args.master:
            return
        if not os.path.exists(self.experiment_dir):
            os.makedirs(self.experiment_dir)
        # if self.output_mask_dir and not os.path.exists(self.output_mask_dir):
        #     os.makedirs(self.output_mask_dir)
        # if os.path.exists(self.logfile):
        #     os.remove(self.logfile)
        # if os.path.exists(self.csvlogfile):
        #     os.remove(self.csvlogfile)
        self.lossfile = 'loss.txt'

    def save_checkpoint(self, state, is_best, filename='checkpoint.pth.tar'):
        """Saves checkpoint to disk"""
        self.best_model_filepath = os.path.join(self.experiment_dir, 'model_best.pth.tar')
        if not self.args.master:
            return
        filename = os.path.join(self.experiment_dir, filename)
        # torch.save(state, filename)
        if is_best:   
            torch.save(state, self.best_model_filepath)

    def save_experiment_config(self):
        if not self.args.master:
            return
        p=vars(self.args)
        log_file = open(self.logfile, "a+")
        for key, val in p.items():
            log_file.write(key + ':' + str(val) + '\n')
        log_file.write('\n')
        # current_time = strftime("%Y-%m-%d-%H-%M-%S", gmtime())
        current_time = self.get_current_time()
        log_file.write('current_time' + ':' + current_time + '\n')
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


    def write_loss_to_txt(self, data):
        assert isinstance(data, str)
        # if not self.args.master:
        #     return
        loss_file = open(self.lossfile, "a+")
        if data.endswith('\n'):
            loss_file.write(data)
        else:
            loss_file.write(data+'\n')
        loss_file.close()# 

    
    def write_log_to_csv(self, row, head = None):
        with open(self.csvlogfile, 'a+', newline='') as csvfile:
            # fieldnames = ['first_name', 'last_name']
            # writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # writer.writeheader()
            # writer.writerow({'first_name': 'Baked', 'last_name': 'Beans'})
            # writer.writerow({'first_name': 'Lovely', 'last_name': 'Spam'})
            # writer.writerow({'first_name': 'Wonderful', 'last_name': 'Spam'})
            writer = csv.writer(csvfile, dialect='excel')
            if head is not None:
                writer.writerow(head)
            writer.writerow(row)

        with open('log.csv', 'a+', newline='') as csvfile:
            # fieldnames = ['first_name', 'last_name']
            # writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            # writer.writeheader()
            # writer.writerow({'first_name': 'Baked', 'last_name': 'Beans'})
            # writer.writerow({'first_name': 'Lovely', 'last_name': 'Spam'})
            # writer.writerow({'first_name': 'Wonderful', 'last_name': 'Spam'})
            writer = csv.writer(csvfile, dialect='excel')
            if head is not None:
                writer.writerow(head)
            writer.writerow(row)
            
    def get_current_time(self):
        tz = pytz.timezone('Asia/Shanghai')
        current_time = datetime.datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")
        return str(current_time)