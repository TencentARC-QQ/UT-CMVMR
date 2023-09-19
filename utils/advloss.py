import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.parameter import Parameter

class AdvCrossModalLoss(nn.Module):
    def __init__(self, data_dim):
        super(AdvCrossModalLoss, self).__init__()
        '''
           data_dim: int; the second dim of the input data
        '''
        w1 = torch.normal(0, 0.01, size=(data_dim//2, data_dim), requires_grad=True)
        w2 = torch.normal(0, 0.01, size=(data_dim//4, data_dim//2), requires_grad=True)
        w3 = torch.normal(0, 0.01, size=(2, data_dim//4), requires_grad=True)
        linear1 = nn.Linear(data_dim, data_dim//2)
        linear1.weight = Parameter(w1)
        linear2 = nn.Linear(data_dim//2, data_dim//4)
        linear2.weight = Parameter(w2)
        linear3 = nn.Linear(data_dim//4, 2)
        linear3.weight = Parameter(w3)
        self.discriminator = nn.Sequential(linear1, linear2, linear3)
        self.cross_entropy_loss = nn.CrossEntropyLoss()
    def forward(self, x_embed, y_embed):
        '''
            x_embed: tensor; the input of embeded audio vec
            y_embed: tensor; the input of embeded video vec
        '''
        audio_logits = self.discriminator(x_embed)
        video_logits = self.discriminator(y_embed)
        #  if the tensor is from audio, then it's label is set as 1; Vice versa
        y_1 = np.repeat([1], x_embed.shape[0], axis=0)
        y_0 = np.repeat([0], y_embed.shape[0], axis=0)
        y_1 = torch.Tensor(y_1)
        y_0 = torch.Tensor(y_0)
        if torch.cuda.is_available():
           y_1 = y_1.cuda()
           y_0 = y_0.cuda()
        outputs = torch.concat((audio_logits, video_logits), axis=0)
        y = torch.concat((y_1, y_0), axis=0)
        crossentropyloss_output = self.cross_entropy_loss(outputs, y.long())
        top_pred = outputs.argmax(1, keepdim=True)
        correct = top_pred.eq(y.view_as(top_pred)).sum()
        acc = correct.float() / y.shape[0]
        return crossentropyloss_output, acc






