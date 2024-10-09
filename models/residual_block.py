import torch
import torch.nn as nn

from models.causal_conv_1d import CausalConv1d

class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, skip_channels, kernel_size, dilation):
        super(ResidualBlock, self).__init__()
        self.causal_conv = CausalConv1d(residual_channels, 2 * residual_channels, kernel_size, dilation)
        self.residual_conv = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, kernel_size=1)

    def forward(self, x):
        # dilated conv
        out = self.causal_conv(x)

        # gate activation unit
        tanh_out = torch.tanh(out[:, :out.shape[1] // 2, :])
        sigm_out = torch.sigmoid(out[:, out.shape[1] // 2:, :])
        gated_out = tanh_out * sigm_out

        # Residual connection
        residual_out = self.residual_conv(gated_out)
        residual_out = residual_out + x[:, :, -residual_out.shape[2]:]

        # Skip connection
        skip_out = self.skip_conv(gated_out)
        return residual_out, skip_out
