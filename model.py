import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.weight)
        nn.init.zeros_(self.bias)


class RFF_MLP_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.RFF_freq = nn.Parameter(
            16 * torch.randn([1, 32]), requires_grad=False)
        self.MLP = nn.ModuleList([
            nn.Linear(64, 128),
            nn.Linear(128, 256),
            nn.Linear(256, 512),
        ])

    def forward(self, sigma):
        """
        Arguments:
          sigma:
              (shape: [B, 1], dtype: float32)

        Returns:
          x: embedding of sigma
              (shape: [B, 512], dtype: float32)
        """
        x = self._build_RFF_embedding(sigma)
        for layer in self.MLP:
            x = F.relu(layer(x))
        return x

    def _build_RFF_embedding(self, sigma):
        """
        Arguments:
          sigma:
              (shape: [B, 1], dtype: float32)
        Returns:
          table:
              (shape: [B, 64], dtype: float32)
        """
        freqs = self.RFF_freq
        freqs = freqs.to(device=torch.device("cuda"))
        table = 2 * np.pi * sigma * freqs
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)
        return table


class Film(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.output_layer = nn.Linear(512, 2 * output_dim)

    def forward(self, sigma_encoding):
        sigma_encoding = self.output_layer(sigma_encoding)
        sigma_encoding = sigma_encoding.unsqueeze(-1)
        gamma, beta = torch.chunk(sigma_encoding, 2, dim=1)
        return gamma, beta


class UBlock(nn.Module):
    def __init__(self, input_size, hidden_size, factor, dilation):
        super().__init__()
        assert isinstance(dilation, (list, tuple))
        assert len(dilation) == 4

        self.factor = factor
        self.residual_dense = Conv1d(input_size, hidden_size, 1)
        self.convs = nn.ModuleList([
            Conv1d(2 * input_size, hidden_size, 3,
                   dilation=dilation[0], padding=dilation[0]),
            Conv1d(hidden_size, hidden_size, 3,
                   dilation=dilation[1], padding=dilation[1]),
            Conv1d(hidden_size, hidden_size, 3,
                   dilation=dilation[2], padding=dilation[2]),
            Conv1d(hidden_size, hidden_size, 3,
                   dilation=dilation[3], padding=dilation[3])
        ])

    def forward(self, x, x_dblock):
        size = x.shape[-1] * self.factor

        residual = F.interpolate(x, size=size)
        residual = self.residual_dense(residual)

        x = torch.cat([x, x_dblock], dim=1)
        x = F.leaky_relu(x, 0.2)
        x = F.interpolate(x, size=size)
        for layer in self.convs:
            x = F.leaky_relu(x, 0.2)
            x = layer(x)
        return x + residual


class DBlock(nn.Module):
    def __init__(self, input_size, hidden_size, factor):
        super().__init__()
        self.factor = factor
        self.residual_dense = Conv1d(input_size, hidden_size, 1)
        self.layer_1 = Conv1d(input_size, hidden_size,
                              3, dilation=1, padding=1)
        self.convs = nn.ModuleList([
            Conv1d(hidden_size, hidden_size, 3, dilation=2, padding=2),
            Conv1d(hidden_size, hidden_size, 3, dilation=4, padding=4),
            Conv1d(hidden_size, hidden_size, 3, dilation=8, padding=8),

        ])

    def forward(self, x, gamma, beta):
        size = x.shape[-1] // self.factor

        residual = self.residual_dense(x)
        residual = F.interpolate(residual, size=size)

        x = F.interpolate(x, size=size)
        x = F.leaky_relu(x, 0.2)
        x = self.layer_1(x)
        x = gamma * x + beta
        for layer in self.convs:
            x = F.leaky_relu(x, 0.2)
            x = layer(x)

        return x + residual


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_1 = Conv1d(1, 32, 5, padding=2)
        self.embedding = RFF_MLP_Block()
        self.downsample = nn.ModuleList([
            DBlock(32, 128, 2),
            DBlock(128, 128, 2),
            DBlock(128, 256, 3),
            DBlock(256, 512, 5),
            DBlock(512, 512, 5),
        ])
        self.gamma_beta = nn.ModuleList([
            Film(128),
            Film(128),
            Film(256),
            Film(512),
            Film(512),
        ])
        self.upsample = nn.ModuleList([
            UBlock(512, 512, 5, [1, 2, 4, 8]),
            UBlock(512, 256, 5, [1, 2, 4, 8]),
            UBlock(256, 128, 3, [1, 2, 4, 8]),
            UBlock(128, 128, 2, [1, 2, 4, 8]),
            UBlock(128, 128, 2, [1, 2, 4, 8]),
        ])
        self.last_conv = Conv1d(128, 1, 3, padding=1)

    def forward(self, audio, sigma):
        x = audio.unsqueeze(1)
        x = self.conv_1(x)
        downsampled = []
        sigma_encoding = self.embedding(sigma)

        for film, layer in zip(self.gamma_beta, self.downsample):
            gamma, beta = film(sigma_encoding)
            x = layer(x, gamma, beta)
            downsampled.append(x)

        for layer, x_dblock in zip(self.upsample, reversed(downsampled)):
            x = layer(x, x_dblock)
        x = self.last_conv(x)
        x = x.squeeze(1)
        return x
