import torch
import torch.nn.functional as F

class FastSTL(torch.nn.Module):
    def __init__(self, period=24, trend_kernel=25, device='cuda'):
        super(FastSTL, self).__init__()
        self.period = period
        self.trend_kernel = trend_kernel
        self.device = device

    def moving_avg(self, x, kernel_size):
        padding = kernel_size // 2

        x_padded = F.pad(x, (padding, padding), mode='replicate')  # [B, N, T + 2*padding]

        weight = torch.ones(1, 1, kernel_size, device=x.device) / kernel_size

        return F.conv1d(x_padded, weight)

    def forward(self, x):
        # x: [B, T, N]
        B, T, N = x.shape
        x = x.permute(0, 2, 1).to(self.device)  # [B, N, T]

        # Trend
        trend = self.moving_avg(x, self.trend_kernel)

        # Seasonal pattern from first cycle
        seasonal = x - trend

        return seasonal.permute(0, 2, 1), trend.permute(0, 2, 1)
