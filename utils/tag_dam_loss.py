import torch
import torch.nn as nn
import numpy as np


class ClusterClassificationLoss(nn.Module):
    def __init__(self, target_dim = 100, drop_rate = 0.1, init_method = "kaiming"):
        super(ClusterClassificationLoss, self).__init__()
        self.target_dim = target_dim
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = drop_rate)
        self.bn1 = nn.BatchNorm1d(1024, eps = 1e-3, momentum = 0.1)
        self.bn2 = nn.BatchNorm1d(512, eps = 1e-3, momentum = 0.1)
        self.bn3 = nn.BatchNorm1d(256, eps = 1e-3, momentum = 0.1)

        self.linear_net = nn.Sequential(
            nn.Linear(1024, 1024),
            self.bn1,
            self.relu,
            self.dropout,
            nn.Linear(1024, 512),
            self.bn2,
            self.relu,
            self.dropout,
            nn.Linear(512, 256),
            self.bn3,
            self.relu,
            self.dropout,
            nn.Linear(256, self.target_dim)
        )
        self.init_linear_weights(self.linear_net, init_method)

    def init_linear_weights(self, model, init_method):
        for m in model:
            if isinstance(m, nn.Linear):
                if init_method == "xavier":
                    nn.init.xavier_normal_(m.weight)
                elif init_method == "kaiming":
                    nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.01)

    def forward(self, x_embed: torch.Tensor, y_embed: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        if torch.cuda.is_available():
            label = label.cuda()
        final_input = torch.concat((x_embed, y_embed), axis = 1)
        outputs = self.linear_net(final_input)
        loss_func = nn.CrossEntropyLoss()
        loss = loss_func(outputs, label.long())
        top_pred = outputs.argmax(1, keepdim=True)
        correct = top_pred.eq(label.view_as(top_pred)).sum()
        acc = correct.float() / label.shape[0]
        return loss, acc

class ClusterSoftmaxLoss(ClusterClassificationLoss):
    def __init__(self, input_dim, target_dim = 100, drop_rate = 0.1, init_method = "kaiming", m=0.35, s=30):
        super(ClusterSoftmaxLoss, self).__init__(target_dim, drop_rate, init_method)
        self.W1 = torch.nn.Parameter(torch.normal(0, 1, size=(input_dim//2, input_dim)), requires_grad=True)
        self.W2 = torch.nn.Parameter(torch.normal(0, 1, size=(input_dim//2, input_dim)), requires_grad=True)
        self.m = m
        self.s = s
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_rate)
        self.bn1 = nn.BatchNorm1d(input_dim//2, eps=1e-3, momentum=0.1)
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, input_dim//2),
            self.bn1,
            self.relu,
            self.dropout,
            nn.Linear(input_dim//2, target_dim)
        )
        self.init_linear_weights(self.linear_net, init_method)

    def forward(self, x_embed=None, y_embed=None, label=None, if_testing=False):
        w_norm_1 = torch.norm(self.W1, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm_1 = torch.div(self.W1, w_norm_1)
        w_norm_2 = torch.norm(self.W2, p=2, dim=0, keepdim=True).clamp(min=1e-12)
        w_norm_2 = torch.div(self.W2, w_norm_2)
        if torch.cuda.is_available():
            w_norm_1 = w_norm_1.cuda()
            w_norm_2 = w_norm_2.cuda()
        # normalization
        x_norm = torch.norm(x_embed, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = torch.div(x_embed, x_norm)
        y_norm = torch.norm(y_embed, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        y_norm = torch.div(y_embed, y_norm)
        costh_x = torch.mm(x_norm, w_norm_1.T)
        costh_y = torch.mm(y_norm, w_norm_2.T)
        # concat x and y
        costh = torch.concat((costh_x, costh_y), axis=1)
        costh = self.linear_net(costh)
        if torch.cuda.is_available():
            costh = costh.cuda()
        if if_testing == True:
            return costh
        else:
            if torch.cuda.is_available():
                label = label.cuda()
            delt_costh = torch.zeros(costh.size()).cuda().scatter_(1, label.to(torch.int64).reshape(-1,1), self.m)
            costh_m = costh - delt_costh
            outputs = self.s * costh_m
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(outputs, label.long())
            top_pred = outputs.argmax(1, keepdim=True)
            correct = top_pred.eq(label.view_as(top_pred)).sum()
            acc = correct.float() / label.shape[0]
            return loss, acc





