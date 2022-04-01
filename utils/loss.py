import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import sys

print(os.getcwd())
sys.path.append(os.getcwd())
from utils.lovasz_losses import lovasz_softmax
# from lovasz_losses import lovasz_softmax




class FocalLossObj(nn.Module):
    def __init__(self, gamma=2, weight=None, ignore_index=255, size_average=True):
        super(FocalLossObj, self).__init__()
        self.gamma = gamma
        self.size_average = size_average
        self.CE_loss = nn.CrossEntropyLoss(reduce=False, ignore_index=ignore_index, weight=weight)

    def forward(self, output, target):
        logpt = self.CE_loss(output, target)
        pt = torch.exp(-logpt)
        loss = ((1-pt)**self.gamma) * logpt
        if self.size_average:
            return loss.mean()
        return loss.sum()


class LovaszSoftmax(nn.Module):
    def __init__(self, classes='present', per_image=False, ignore_index=255):
        super(LovaszSoftmax, self).__init__()
        self.smooth = classes
        self.per_image = per_image
        self.ignore_index = ignore_index
    
    def forward(self, output, target):
        logits = F.softmax(output, dim=1)
        loss = lovasz_softmax(logits, target, ignore=self.ignore_index)
        return loss

class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False, args = None):
        if args and args.ignore_index >= 0:
            self.ignore_index = args.ignore_index
        else:
            self.ignore_index = ignore_index
 
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

        self.focal = FocalLossObj(weight=weight, ignore_index=ignore_index, size_average=True)
        self.lovas = LovaszSoftmax()

    def build_loss(self, type='ce'):
        """Choices: ['ce' or 'focal']"""
        if type == 'ce':
            return self.CrossEntropyLoss
        elif type == 'focal':
            return self.FocalLoss
        elif type == 'FSOhemCELoss':
            return self.FSOhemCELoss
        elif type == 'FocalLovas':
            return self.FocalLovas
        elif type == 'LovasLoss':
            return self.LovasLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        # print('loss target', target.size(), target.type())
        if not self.weight is None and self.cuda:
            weight = self.weight.cuda()
        else:
            weight = None 
    
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average and not self.size_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)

        if alpha:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average and not self.size_average:
            loss /= n

        return loss

    def FSOhemCELoss(self, predict, target):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        # print('predict.size(), target.size()', predict.size(), target.size())
        self.reduction = 'elementwise_mean'
        self.thresh = 0.7
        self.min_kept = 50000
        if not self.weight is None and self.cuda:
            weight = self.weight.cuda()
        else:
            weight = None 
        # ce_weight = [0.1, 3]
        # weight = torch.FloatTensor(ce_weight).cuda()
        self.ce_loss = nn.CrossEntropyLoss(weight=weight, ignore_index=self.ignore_index, reduction='none')

        prob_out = F.softmax(predict, dim=1)
        tmp_target = target.clone()
        tmp_target[tmp_target == self.ignore_index] = 0
        # print('type(tmp_target.unsqueeze(1))', type(tmp_target.unsqueeze(1)))
        tmp_target = tmp_target.to(torch.int64)
        prob = prob_out.gather(1, tmp_target.unsqueeze(1))
        mask = target.contiguous().view(-1,) != self.ignore_index
        mask[0] = 1  # Avoid `mask` being empty
        sort_prob, sort_indices = prob.contiguous().view(-1,)[mask].contiguous().sort()
        min_threshold = sort_prob[min(self.min_kept, sort_prob.numel() - 1)]
        threshold = max(min_threshold, self.thresh)
        # print('loss target', target.size(), target.type())
        target = target.to(torch.int64)
        loss_matirx = self.ce_loss(predict, target).contiguous().view(-1,)
        sort_loss_matirx = loss_matirx[mask][sort_indices]
        select_loss_matrix = sort_loss_matirx[sort_prob < threshold]
        if self.reduction == 'sum':
            return select_loss_matrix.sum()
        elif self.reduction == 'elementwise_mean':
            return select_loss_matrix.mean()
        else:
            raise NotImplementedError('Reduction Error!')

    def FocalLovas(self, predict, target):
        #print("predict device {}".format(predict.device))
        #print("target device {}".format(target.device))
        self.focal = self.focal.to(predict.device)
        self.lovas = self.lovas.to(predict.device)

        focal_loss = self.focal(predict, target.long())
        lovas_loss = self.lovas(predict, target.long())

        return focal_loss + lovas_loss

    def LovasLoss(self, predict, target):
        #print("predict device {}".format(predict.device))
        #print("target device {}".format(target.device))
        self.lovas = self.lovas.to(predict.device)

        lovas_loss = self.lovas(predict, target.long())

        return lovas_loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float, int, int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1).long()

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()



if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    focal_loss = FocalLoss(gamma=2, alpha=0.5)
    a = torch.rand(3, 3, 5, 5).cuda()
    b = torch.rand(3, 5, 5).cuda()
    
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
    print(focal_loss(a, b).item())




