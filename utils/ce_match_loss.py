import queue
import torch
import torch.nn as nn
import numpy as np

class CEMatchingLoss(nn.Module):
    def __init__(self, input_dim, drop_out = 0.1, init_method = "kaiming"):
        '''
           :param drop_out: the dropout rate
           :param init_method: the method to init the linear
        '''
        super(CEMatchingLoss, self).__init__()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = drop_out)
        self.bn1 = nn.BatchNorm1d(input_dim, eps = 1e-3, momentum = 0.1)
        self.bn2 = nn.BatchNorm1d(input_dim//2, eps = 1e-3, momentum = 0.1)

        self.linear_net = nn.Sequential(
            nn.Linear(input_dim*2, input_dim),
            self.bn1,
            self.relu,
            self.dropout,
            nn.Linear(input_dim, input_dim//2),
            self.bn2,
            self.relu,
            self.dropout,
            nn.Linear(input_dim//2, 2)
        )
        self.init_linear_weights(self.linear_net, init_method)

        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def init_linear_weights(self, model, init_method):
        for m in model:
            if isinstance(m, nn.Linear):
                if init_method == "xavier":
                    nn.init.xavier_normal_(m.weight)
                elif init_method == "kaiming":
                    nn.init.kaiming_normal_(m.weight)
                m.bias.data.fill_(0.01)


    def forward(self, x_embed: torch.Tensor, y_embed: torch.Tensor, n_x_embed: torch.Tensor, n_y_embed: torch.Tensor) -> torch.Tensor:
        """
            x_embed: torch.Tensor; embedded audio features from music subnetwork, dimension is
                     [batch_num, audio_feature_dim]; audio_feature_dim: 512
            y_embed: torch.Tensor; embedded video features from video subnetwork, dimension is
                     [batch_num, video_feature_dim]; video_feature_dim: 512
        """
        # to get the negative samples
        current_batch_size = x_embed.shape[0]
        y_1 = np.repeat([1], current_batch_size)
        y_0 = np.repeat([0], current_batch_size * 2)
        y = np.concatenate([y_1, y_0])
        x_new = torch.concat((x_embed, x_embed, n_x_embed), axis=0)
        y_new = torch.concat((y_embed, n_y_embed, y_embed), axis=0)
        final_input = torch.concat((x_new, y_new), axis = 1)
        label = torch.Tensor(y)
        if torch.cuda.is_available():
           label = label.cuda()
        outputs = self.linear_net(final_input)
        ce_loss = self.cross_entropy_loss(outputs, label.long())
        _, preds = torch.max(outputs, 1)
        acc_sum = torch.sum(torch.eq(preds ,label))
        return ce_loss, acc_sum / preds.shape[0]
