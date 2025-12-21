import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.Attention import MultiheadAttention

class CMBlockChannelAttention(nn.Module):
    def __init__(self, n_vars=7, reduction=2, avg_flag=True, max_flag=True):
        super(CMBlockChannelAttention, self).__init__()
        # parameters
        self.avg_flag = avg_flag
        self.max_flag = max_flag
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.reduction = reduction

        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Sequential(nn.Linear(n_vars, n_vars // self.reduction, bias=True),
                                nn.GELU(),
                                nn.Linear(n_vars // self.reduction, n_vars, bias=True))

    def forward(self, x):
        batch_size, channels, d_model = x.shape
        device = x.device
        x = x.reshape(batch_size, d_model, channels, -1).to(device)
        out = torch.zeros_like(x).to(device)
        if self.avg_flag:
            tmp = self.avg_pool(x.to(device)).to(device)
            tmp = self.fc(tmp.reshape(batch_size, d_model).to(device)).to(device)
            tmp = tmp.reshape(batch_size, d_model, 1, 1).to(device)
            out += tmp.to(device)
            out = out.to(device)
        if self.max_flag:
            out += self.fc(self.max_pool(x).reshape(batch_size, d_model)).reshape(batch_size, d_model, 1, -1)
        ans = self.sigmoid(out) * x
        ans = ans.reshape(batch_size, channels, d_model)
        return ans

class CMBlock(nn.Module):
    def __init__(self, n_vars=7, reduction=2, avg_flag=True, max_flag=True):
        super(CMBlock, self).__init__()
        self.CAttention = CMBlockChannelAttention(n_vars, reduction=reduction, avg_flag=avg_flag, max_flag=max_flag)

    def forward(self, input_x, x):
        device = input_x.device
        input_x = input_x.to(device)
        return input_x + self.CAttention(x)