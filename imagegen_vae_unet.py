import torch
import torch.nn as nn
import torch.nn.functional as F
import wget
import json
import os

IMAGEGEN_FOLDER = "./ImageGenModel"
IMAGEGEN_MODEL_WEIGHTS = "diffusion_pytorch_model.bin"
IMAGEGEN_CONFIG = "config.json"
IMAGEGEN_MODEL_URL = "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/diffusion_pytorch_model.bin"
IMAGEGEN_CONFIG_URL = "https://huggingface.co/stabilityai/sd-vae-ft-mse/resolve/main/config.json"
IMAGEGEN_FILES_URLS = [
    (IMAGEGEN_MODEL_URL, IMAGEGEN_MODEL_WEIGHTS),
    (IMAGEGEN_CONFIG_URL, IMAGEGEN_CONFIG),
]

def ensure_imagegen_files_exist():
    os.makedirs(IMAGEGEN_FOLDER, exist_ok=True)
    for url, filename in IMAGEGEN_FILES_URLS:
        filepath = os.path.join(IMAGEGEN_FOLDER, filename)
        if not os.path.exists(filepath):
            wget.download(url, out=filepath)

class UNet2DConditionModelConfig:
    def __init__(self, **kwargs):
        self.sample_size = 64
        self.layers_per_block = 2
        self.block_out_channels = [320, 640, 1280, 1280]
        self.downsample = [2, 2, 2, 2]
        self.upsample = [2, 2, 2, 2]
        self.cross_attention_dim = 768
        self.act_fn = "silu"
        self.norm_num_groups = 32
        self.num_attention_heads = 8
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)

class UNet2DConditionModel(nn.Module):
    def __init__(self, config: UNet2DConditionModelConfig):
        super().__init__()
        self.conv_in = nn.Conv2d(4, config.block_out_channels[0], kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList([])
        for i in range(len(config.block_out_channels)):
            is_final_block = i == len(config.block_out_channels) - 1
            downsample_factor = 1 if is_final_block else config.downsample[i]
            out_channels = config.block_out_channels[i]
            layers_per_block = config.layers_per_block
            self.down_blocks.append(DownBlock(out_channels, layers_per_block, downsample_factor))
        self.mid_block = MidBlock(config.block_out_channels[-1])
        self.up_blocks = nn.ModuleList([])
        reversed_block_out_channels = list(reversed(config.block_out_channels))
        reversed_upsample_factors = list(reversed(config.upsample))
        for i in range(len(config.block_out_channels)):
            is_final_block = i == len(config.block_out_channels) - 1
            upsample_factor = 1 if is_final_block else reversed_upsample_factors[i]
            out_channels = reversed_block_out_channels[i]
            layers_per_block = config.layers_per_block
            self.up_blocks.append(UpBlock(out_channels, layers_per_block, upsample_factor))
        self.norm_out = nn.GroupNorm(num_groups=config.norm_num_groups, num_channels=config.block_out_channels[0])
        self.conv_norm_out = nn.Conv2d(config.block_out_channels[0], config.block_out_channels[0], kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(config.block_out_channels[0], 4, kernel_size=3, padding=1)

    def forward(self, sample: torch.FloatTensor, timestep: torch.IntTensor, encoder_hidden_states: torch.FloatTensor):
        sample = self.conv_in(sample)
        for down_block in self.down_blocks:
            sample = down_block(sample)
        sample = self.mid_block(sample)
        for up_block in self.up_blocks:
            sample = up_block(sample)
        sample = self.norm_out(sample)
        sample = F.silu(sample)
        sample = self.conv_norm_out(sample)
        sample = F.silu(sample)
        sample = self.conv_out(sample)
        return {"sample": sample}

class DownBlock(nn.Module):
    def __init__(self, out_channels, layers_per_block, downsample_factor):
        super().__init__()
        self.layers = nn.ModuleList([ResnetBlock(out_channels) for _ in range(layers_per_block)])
        if downsample_factor > 1:
            self.downsample = Downsample2D(out_channels, downsample_factor)
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.downsample(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, out_channels, layers_per_block, upsample_factor):
        super().__init__()
        self.layers = nn.ModuleList([ResnetBlock(out_channels) for _ in range(layers_per_block)])
        if upsample_factor > 1:
            self.upsample = Upsample2D(out_channels, upsample_factor)
        else:
            self.upsample = nn.Identity()

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.upsample(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.residual_conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        return x + self.residual_conv(residual)

class MidBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.norm1(x)
        x = F.silu(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = F.silu(x)
        x = self.conv2(x)
        return x

class Downsample2D(nn.Module):
    def __init__(self, channels, factor):
        super().__init__()
        self.factor = factor
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=factor, padding=1)

    def forward(self, x):
        return self.conv(x)

class Upsample2D(nn.Module):
    def __init__(self, channels, factor):
        super().__init__()
        self.factor = factor
        self.conv = nn.ConvTranspose2d(channels, channels, kernel_size=factor, stride=factor)

    def forward(self, x):
        return self.conv(x)