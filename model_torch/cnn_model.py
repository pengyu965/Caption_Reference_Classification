import torch
import torch.nn as nn 
import torch.nn.functional as F

import torch.optim as optim
from torch.optim import lr_scheduler

import time 

import json 


class Model(nn.Module):
    def __init__(self, batch_size = 50, lr = 0.001, keep_prob = 0.4, class_num = 3, is_training = True):
        super(Model, self).__init__()

        self.batch_size = batch_size
        self.lr = lr 
        self.keep_prob = keep_prob
        self.class_num = class_num
        self.is_training = is_training

        self.conv1 = nn.Conv2d(2, 32, (5,5), stride = (2,2), padding = (2,2))
        self.batch_norm1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 128, (3,3), stride = (2,2), padding = (1,1))
        self.batch_norm2 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(128*50*25, 128)
        self.fc2 = nn.Linear(128, 64)
        self.logits = nn.Linear(64, self.class_num)

        self.layer_output = {}

    def forward(self, input):
        x = F.avg_pool2d(self.batch_norm1(F.relu(self.conv1(input))))
        x = F.avg_pool2d(self.batch_norm2(F.relu(self.conv2(x))))

        x = F.dropout2d(F.relu(self.fc1(x.view(-1,128*50*25))), p = self.keep_prob)
        x = F.dropout2d(F.relu(self.fc2(x)), p = self.keep_prob)
        x = self.logits(x)

        return x

class Trainer:
    



