import torch
import torch.nn.functional as F

# FastSTL
class FastSTL(torch.nn.Module):
    def __init__(self, period=24, trend_kernel=25, device='cuda'):
        super(FastSTL, self).__init__()
        self.period = period
        self.trend_kernel = trend_kernel
        self.device = device

    def moving_avg(self, x, kernel_size):
        padding = kernel_size // 2

        x_padded = F.pad(x, (padding, padding), mode='replicate')

        weight = torch.ones(1, 1, kernel_size, device=x.device) / kernel_size

        return F.conv1d(x_padded, weight)

    def forward(self, x):

        B, T, N = x.shape
        x = x.permute(0, 2, 1).to(self.device)

        trend = self.moving_avg(x, self.trend_kernel)

        seasonal = x - trend

        return seasonal.permute(0, 2, 1), trend.permute(0, 2, 1)
