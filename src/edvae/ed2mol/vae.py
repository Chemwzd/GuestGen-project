"""ED-VAE model components.

The encoder/decoder follow the 3D convolutional architecture of the ED
generator in ED2Mol (https://github.com/pineappleK/ED2Mol, MIT License); the
variational formulation and the training loop are part of this project.

The decoder output is not squashed into [0, 1]: electron-density values span
several orders of magnitude and are reconstructed with a plain MSE loss.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from typing_extensions import List, Tuple


class VAEEncoderBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        use_bias: bool = True,
        dropout_ratio: float = 0.2,
        use_norm: bool = True,
        use_pool: bool = False,
    ):
        super(VAEEncoderBlock, self).__init__()

        self.use_norm = use_norm
        self.use_pool = use_pool

        self.encoder_block = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=use_bias,
        )
        if self.use_norm:
            self.norm = nn.InstanceNorm3d(out_channels)
        if self.use_pool:
            self.maxpool = nn.MaxPool3d(2)
        self.act = nn.GELU()
        self.dropout = nn.Dropout3d(dropout_ratio)

    def forward(self, x: Tensor):
        x = self.encoder_block(x)
        if self.use_norm:
            x = self.norm(x)
        x = self.act(x)
        if self.use_pool:
            x = self.maxpool(x)
        x = self.dropout(x)

        return x


class VAEEncoder(nn.Module):
    """3D-CNN encoder.

    Input shape: ``[B, 1, 48, 48, 48]``.
    """

    def __init__(
        self,
        max_channels: int,
        kernel_sizes: List[int] = [5, 5, 4, 4, 3],
        strides: List[int] = [2, 2, 2, 2, 2],
        paddings: List[int] = [2, 2, 1, 1, 1],
        dropout_ratio: float = 0.2,
    ):
        super(VAEEncoder, self).__init__()

        self.encoder_block_1 = VAEEncoderBlock(
            1,
            max_channels // 8,
            kernel_size=kernel_sizes[0],
            stride=strides[0],
            padding=paddings[0],
            use_bias=True,
            dropout_ratio=dropout_ratio,
            use_norm=False,
        )  # [1, 1, 48, 48, 48] -> [1, 16, 24, 24, 24]

        self.encoder_block_2 = VAEEncoderBlock(
            max_channels // 8,
            max_channels // 4,
            kernel_size=kernel_sizes[1],
            stride=strides[1],
            padding=paddings[1],
            use_bias=True,
            dropout_ratio=dropout_ratio,
            use_norm=False,
        )  # [1, 16, 24, 24, 24] -> [1, 32, 12, 12, 12]

        self.encoder_block_3 = VAEEncoderBlock(
            max_channels // 4,
            max_channels // 2,
            kernel_size=kernel_sizes[2],
            stride=strides[2],
            padding=paddings[2],
            use_bias=True,
            dropout_ratio=dropout_ratio,
        )  # [1, 32, 12, 12, 12] -> [1, 64, 6, 6, 6]

        self.encoder_block_4 = VAEEncoderBlock(
            max_channels // 2,
            max_channels,
            kernel_size=kernel_sizes[3],
            stride=strides[3],
            padding=paddings[3],
            use_bias=True,
            dropout_ratio=dropout_ratio,
        )  # [1, 64, 6, 6, 6] -> [1, 128, 3, 3, 3]

        self.encoder_block_5 = VAEEncoderBlock(
            max_channels,
            max_channels * 2,
            kernel_size=kernel_sizes[4],
            stride=strides[4],
            padding=paddings[4],
            use_bias=True,
            dropout_ratio=dropout_ratio,
        )  # [1, 128, 3, 3, 3] -> [1, 256, 3, 3, 3]

    def forward(self, x: Tensor):
        x = self.encoder_block_1(x)
        x = self.encoder_block_2(x)
        x = self.encoder_block_3(x)
        x = self.encoder_block_4(x)
        x = self.encoder_block_5(x)

        return x


class VAEDecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size: int = 3,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 1,
        use_bias: bool = True,
    ):
        super(VAEDecoderBlock, self).__init__()

        self.decoder_block = nn.ConvTranspose3d(
            in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            bias=use_bias,
        )
        self.act = nn.GELU()

    def forward(self, x: Tensor):
        x = self.decoder_block(x)
        x = self.act(x)
        return x


class VAEDecoder(nn.Module):
    """3D transposed-convolution decoder.

    Output size per axis: ``(D_in - 1) * stride - 2 * padding + kernel_size + output_padding``.
    """

    def __init__(
        self,
        max_channels: int,
        kernel_sizes: List[int] = [1, 4, 4, 5, 5],
        strides: List[int] = [1, 2, 2, 2, 2],
        paddings: List[int] = [0, 1, 1, 2, 2],
        output_paddings: List[int] = [0, 0, 0, 1, 1],
    ):
        super(VAEDecoder, self).__init__()

        self.decoder_block_1 = VAEDecoderBlock(
            max_channels,
            max_channels // 4,
            kernel_size=kernel_sizes[0],
            stride=strides[0],
            padding=paddings[0],
            output_padding=output_paddings[0],
            use_bias=True,
        )  # [1, 128, 3, 3, 3] -> [1, 32, 3, 3, 3]

        self.decoder_block_2 = VAEDecoderBlock(
            max_channels // 4,
            max_channels // 8,
            kernel_size=kernel_sizes[1],
            stride=strides[1],
            padding=paddings[1],
            output_padding=output_paddings[1],
            use_bias=True,
        )  # [1, 32, 3, 3, 3] -> [1, 16, 6, 6, 6]

        self.decoder_block_3 = VAEDecoderBlock(
            max_channels // 8,
            max_channels // 16,
            kernel_size=kernel_sizes[2],
            stride=strides[2],
            padding=paddings[2],
            output_padding=output_paddings[2],
            use_bias=True,
        )  # [1, 16, 6, 6, 6] -> [1, 8, 12, 12, 12]

        self.decoder_block_4 = VAEDecoderBlock(
            max_channels // 16,
            max_channels // 32,
            kernel_size=kernel_sizes[3],
            stride=strides[3],
            padding=paddings[3],
            output_padding=output_paddings[3],
            use_bias=True,
        )  # [1, 8, 12, 12, 12] -> [1, 4, 24, 24, 24]

        self.decoder_block_5 = VAEDecoderBlock(
            max_channels // 32,
            1,
            kernel_size=kernel_sizes[4],
            stride=strides[4],
            padding=paddings[4],
            output_padding=output_paddings[4],
            use_bias=True,
        )  # [1, 4, 24, 24, 24] -> [1, 1, 48, 48, 48]

    def forward(self, x: Tensor):
        x = self.decoder_block_1(x)
        x = self.decoder_block_2(x)
        x = self.decoder_block_3(x)
        x = self.decoder_block_4(x)
        x = self.decoder_block_5(x)

        return x


class VAE(nn.Module):
    def __init__(
        self,
        max_channels: int = 128,
        encoder_kernel_sizes: List[int] = [5, 5, 4, 4, 1],
        encoder_strides: List[int] = [2, 2, 2, 2, 1],
        encoder_paddings: List[int] = [2, 2, 1, 1, 0],
        decoder_kernel_sizes: List[int] = [1, 4, 4, 5, 5],
        decoder_strides: List[int] = [1, 2, 2, 2, 2],
        decoder_paddings: List[int] = [0, 1, 1, 2, 2],
        decoder_output_paddings: List[int] = [0, 0, 0, 1, 1],
        latent_dim: int = 128,
        dropout_ratio: float = 0.2,
    ):
        super(VAE, self).__init__()

        self.max_channels = max_channels
        self.num_flatten_feats = max_channels * 2 * 3 * 3 * 3

        self.encoder = VAEEncoder(
            max_channels=max_channels,
            kernel_sizes=encoder_kernel_sizes,
            strides=encoder_strides,
            paddings=encoder_paddings,
            dropout_ratio=dropout_ratio,
        )
        self.decoder = VAEDecoder(
            max_channels=max_channels,
            kernel_sizes=decoder_kernel_sizes,
            strides=decoder_strides,
            paddings=decoder_paddings,
            output_paddings=decoder_output_paddings,
        )

        self.proj_mu = nn.Linear(self.num_flatten_feats, latent_dim)
        self.proj_logvar = nn.Linear(self.num_flatten_feats, latent_dim)
        self.proj_latent = nn.Linear(latent_dim, latent_dim * 3 * 3 * 3)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick: ``z = mu + std * eps``, ``eps ~ N(0, 1)``."""

        std = torch.exp(logvar * 0.5)
        eps = torch.randn_like(std, device=mu.device)

        return mu + std * eps

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = x[:, None, ...]

        x = self.encoder(x)

        x = x.view(x.size(0), -1)

        mu = self.proj_mu(x)
        logvar = self.proj_logvar(x)

        # Clamp logvar to keep the variance term numerically stable.
        logvar = torch.clamp(logvar, min=-10, max=10)

        latent_z = self.reparameterize(mu, logvar)

        x = self.proj_latent(latent_z)
        x = x.view(x.shape[0], self.max_channels, 3, 3, 3)

        x = self.decoder(x)

        return x, mu, logvar
