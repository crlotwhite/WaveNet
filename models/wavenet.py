import torch
import torch.nn as nn
import torch.nn.functional as F

from models.causal_conv_1d import CausalConv1d
from models.residual_block import ResidualBlock

class WaveNet(nn.Module):
    def __init__(self, cfg):
        super(WaveNet, self).__init__()
        self.initial_conv = CausalConv1d(cfg.model.in_channels,
                                         cfg.model.residual_channels,
                                         kernel_size=1,
                                         dilation=1)
        self.residual_blocks = nn.ModuleList()
        self.num_blocks = cfg.model.num_blocks
        self.num_layers = cfg.model.num_layers

        for b in range(cfg.model.num_blocks):
            for n in range(cfg.model.num_layers):
                dilation = 2 ** n
                self.residual_blocks.append(ResidualBlock(cfg.model.residual_channels,
                                                          cfg.model.skip_channels,
                                                          cfg.model.kernel_size,
                                                          dilation))

        self.final_conv1 = nn.Conv1d(cfg.model.skip_channels,
                                     cfg.model.skip_channels,
                                     kernel_size=1)
        self.final_conv2 = nn.Conv1d(cfg.model.skip_channels,
                                     cfg.model.in_channels,
                                     kernel_size=1)  # in_channels=256

    def forward(self, x):
        # causal conv
        x = self.initial_conv(x)
        skip_connections = []

        # k layers
        for block in self.residual_blocks:
            x, skip = block(x)
            skip_connections.append(skip)

        # output
        skip_connections = torch.cat(skip_connections, dim=1)
        skip_out = F.relu(skip_connections)
        skip_out = self.final_conv1(skip_out)
        skip_out = F.relu(skip_out)
        out = self.final_conv2(skip_out)

        # CrossEntropyLoss에서 내부적으로 softmax를 수행하기 때문에 softmax 생략

        return out