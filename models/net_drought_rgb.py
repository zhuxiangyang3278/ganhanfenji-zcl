"""
单模态RGB干旱分级模型
基于Restormer架构，仅使用RGB图像进行干旱分级
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Layer Norm
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        return self.body(x)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        # 对于2D卷积特征图，在通道维度上进行归一化
        if x.dim() == 4:  # (B, C, H, W)
            mu = x.mean(dim=1, keepdim=True)
            sigma = x.var(dim=1, keepdim=True, unbiased=False)
            x = (x - mu) / torch.sqrt(sigma + 1e-5)
            # 调整权重和偏置的维度以匹配输入
            weight = self.weight.view(1, -1, 1, 1)
            bias = self.bias.view(1, -1, 1, 1)
            return x * weight + bias
        else:
            # 对于1D向量，使用原来的实现
            mu = x.mean(-1, keepdim=True)
            sigma = x.var(-1, keepdim=True, unbiased=False)
            return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


# ---------------------------------------------------------------------------
# Multi-DConv Head Transposed Self-Attention (MDTA)
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


# ---------------------------------------------------------------------------
# Gated-DConv Feed-Forward Network (GDFN)
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=False):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(
            dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super().__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Restormer Encoder for RGB
# ---------------------------------------------------------------------------

class RestormerEncoder(nn.Module):
    """
    Restormer编码器，专门用于RGB图像特征提取
    """

    def __init__(self, inp_channels=3, dim=48, num_blocks=[4, 6], heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super().__init__()

        self.patch_embed = nn.Conv2d(
            inp_channels, dim, kernel_size=3, stride=1, padding=1, bias=bias)

        # 编码器块
        self.encoder_blocks = nn.ModuleList()
        for i, num_block in enumerate(num_blocks):
            self.encoder_blocks.append(nn.Sequential(*[
                TransformerBlock(dim=dim, num_heads=heads[i],
                                 ffn_expansion_factor=ffn_expansion_factor,
                                 bias=bias, LayerNorm_type=LayerNorm_type)
                for _ in range(num_block)
            ]))

        # 输出层
        self.output_conv = nn.Conv2d(
            dim, dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        # 补丁嵌入
        x = self.patch_embed(x)

        # 编码器块
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)

        # 输出
        x = self.output_conv(x)

        return x


# ---------------------------------------------------------------------------
# 单模态RGB干旱分类器
# ---------------------------------------------------------------------------

class DroughtClassifierRGB(nn.Module):
    """
    单模态RGB干旱分类器
    仅使用RGB图像进行干旱分级
    """

    def __init__(self, dim=48, num_blocks=[4, 6], heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2, num_classes=5):
        super().__init__()

        # RGB编码器
        self.encoder_rgb = RestormerEncoder(
            inp_channels=3,
            dim=dim,
            num_blocks=num_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor
        )

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, rgb):
        """
        前向传播
        Args:
            rgb: RGB图像张量 (B, 3, H, W)
        Returns:
            分类结果 (B, num_classes)
        """
        # RGB特征提取
        rgb_features = self.encoder_rgb(rgb)  # (B, dim, H, W)

        # 全局池化
        pooled_features = self.global_pool(rgb_features)  # (B, dim, 1, 1)
        pooled_features = pooled_features.view(
            pooled_features.size(0), -1)  # (B, dim)

        # 分类
        output = self.classifier(pooled_features)  # (B, num_classes)

        return output


# ---------------------------------------------------------------------------
# 简化版本（用于快速实验）
# ---------------------------------------------------------------------------

class DroughtClassifierRGBLite(nn.Module):
    """
    简化版单模态RGB干旱分类器
    更小的模型，训练更快
    """

    def __init__(self, dim=32, num_blocks=[2, 3], heads=[1, 2],
                 ffn_expansion_factor=2, num_classes=5):
        super().__init__()

        # RGB编码器
        self.encoder_rgb = RestormerEncoder(
            inp_channels=3,
            dim=dim,
            num_blocks=num_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor
        )

        # 全局平均池化
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, rgb):
        """
        前向传播
        Args:
            rgb: RGB图像张量 (B, 3, H, W)
        Returns:
            分类结果 (B, num_classes)
        """
        # RGB特征提取
        rgb_features = self.encoder_rgb(rgb)

        # 全局池化
        pooled_features = self.global_pool(rgb_features)
        pooled_features = pooled_features.view(pooled_features.size(0), -1)

        # 分类
        output = self.classifier(pooled_features)

        return output
