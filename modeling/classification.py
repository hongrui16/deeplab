import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
# from sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor



class ClsHead(nn.Module):
    def __init__(self, 
        num_classes, backbone, BatchNorm):
        super(ClsHead, self).__init__()
        
        if backbone == 'drn':
            inplanes = 512
        elif backbone == 'mobilenet':
            inplanes = 320
        else:
            inplanes = 2048
        

        self.last_conv = nn.Sequential(nn.Conv2d(inplanes, 256, kernel_size=3, stride=2, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.3),
                                       
                                       nn.Conv2d(256, 128, kernel_size=3, stride=2, padding=1, bias=False),                                       
                                       BatchNorm(128),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       
                                       nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
                                       BatchNorm(128),                
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       
                                       nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                       BatchNorm(64),                
                                       nn.ReLU(),
                                       )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)
        
        self._init_weight()


    def forward(self, x):
        x = self.last_conv(x)
        # low_featuer = x
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # return x, low_featuer
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    
def build_cls_head(num_classes, backbone, BatchNorm):
    return ClsHead(num_classes, backbone, BatchNorm)




if __name__ == "__main__":
    import torch
    model = ClsHead(2, 'drn' ,nn.BatchNorm2d)
    print('model', model)
    input = torch.rand(1, 512, 64, 64)
    output, low_featuer = model(input)
    print(output.size()) #torch.Size([1, 2])
    print(low_featuer.size()) #torch.Size([1, 64, 4, 4])
