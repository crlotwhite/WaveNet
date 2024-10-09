import torch.nn as nn
import torch.nn.functional as F

class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, **kwargs):
        super(CausalConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, **kwargs)

    def forward(self, x):
        # 오른쪽으로 패딩을 추가하여 Causal 패딩 적용
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)