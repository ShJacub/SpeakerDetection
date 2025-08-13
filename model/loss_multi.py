import torch
import torch.nn as nn
import torch.nn.functional as F


class lossAV(nn.Module):

    def __init__(self):
        super(lossAV, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.FC = nn.Linear(256, 2)

    def forward(self, x):
        out = x.squeeze(1)
        out = self.FC(out)
        # return x
        
        # custom
        predScore = F.softmax(out, dim=-1)
        return predScore

class lossA(nn.Module):

    def __init__(self):
        super(lossA, self).__init__()
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.FC = nn.Linear(128, 2)

    def forward(self, x, labels, masks=None):
        x = x.squeeze(1)
        x = self.FC(x)
        nloss = self.criterion(x, labels) * masks
        num_valid = masks.sum().float()
        if self.training:
            [num_valid] = du.all_reduce([num_valid],average=True)
        nloss = torch.sum(nloss) / num_valid
        #nloss = torch.sum(nloss) / torch.sum(masks)
        return nloss


class lossV(nn.Module):

    def __init__(self):
        super(lossV, self).__init__()

        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.FC = nn.Linear(128, 2)

    def forward(self, x, labels, masks=None):
        x = x.squeeze(1)
        x = self.FC(x)
        nloss = self.criterion(x, labels) * masks
        # nloss = torch.sum(nloss) / torch.sum(masks)
        num_valid = masks.sum().float()
        if self.training:
            [num_valid] = du.all_reduce([num_valid],average=True)
        nloss = torch.sum(nloss) / num_valid
        return nloss
