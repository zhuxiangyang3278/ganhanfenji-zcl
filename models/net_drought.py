"""
net_drought.py
三分支干旱分级分类网络，基于 CDDFuse 的 Restormer_Encoder。

网络结构：
  分支1: RGB (3通道)           → Restormer_Encoder → base_feature1, detail_feature1
  分支2: 热红外 (3通道)         → Restormer_Encoder → base_feature2, detail_feature2
  分支3: 多光谱+VI (8通道)      → Restormer_Encoder → base_feature3, detail_feature3
              ↓
      特征拼接 [f1, f2, f3]  (B, 192, H, W)
              ↓
      全局平均池化 (GAP)      (B, 192)
              ↓
      分类头 FC: 192 → 256 → 5
              ↓
      输出: 5类干旱等级 (0-4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers


# ---------------------------------------------------------------------------
# LayerNorm helpers
# ---------------------------------------------------------------------------

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        # x: (B, C, H, W)
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


# ---------------------------------------------------------------------------
# Feed-Forward Network (Gated-DConv Feed-Forward, GDFN)
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=False):
        super().__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(
            dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(
            hidden_features * 2, hidden_features * 2,
            kernel_size=3, stride=1, padding=1,
            groups=hidden_features * 2, bias=bias
        )
        self.project_out = nn.Conv2d(
            hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


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
            dim * 3, dim * 3,
            kernel_size=3, stride=1, padding=1,
            groups=dim * 3, bias=bias
        )
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

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        return self.project_out(out)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, ffn_expansion_factor=2,
                 bias=False, LayerNorm_type='WithBias'):
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
# Overlapped Patch Embedding
# ---------------------------------------------------------------------------

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


# ---------------------------------------------------------------------------
# Base-feature extraction (low-frequency / global)
# ---------------------------------------------------------------------------

class AttentionBase(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)',
                      head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, bias=False):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features *
                             2, kernel_size=1, bias=bias)
        self.fc2 = nn.Conv2d(hidden_features, in_features,
                             kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.fc2(x)


class BaseFeatureExtraction(nn.Module):
    def __init__(self, dim, num_heads=8, ffn_expansion_factor=2, bias=False):
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = AttentionBase(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * ffn_expansion_factor))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# Detail-feature extraction (high-frequency / invertible NN)
# ---------------------------------------------------------------------------

class InvertedResidualBlock(nn.Module):
    def __init__(self, inp, oup, expand_ratio=2):
        super().__init__()
        hidden_dim = int(inp * expand_ratio)
        self.bottleneckBlock = nn.Sequential(
            nn.Conv2d(inp, hidden_dim, 1, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1,
                      groups=hidden_dim, bias=False),
            nn.ReLU6(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, bias=False),
        )

    def forward(self, x):
        return self.bottleneckBlock(x)


class DetailNode(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Split dim into two halves
        c = dim // 2
        self.theta_phi = InvertedResidualBlock(c, c)
        self.theta_rho = InvertedResidualBlock(c, c)
        self.theta_eta = InvertedResidualBlock(c, c)

    def nodeblock(self, z1, z2):
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return z1, z2

    def forward(self, z):
        z1, z2 = z.chunk(2, dim=1)
        z1, z2 = self.nodeblock(z1, z2)
        return torch.cat([z1, z2], dim=1)


class DetailFeatureExtraction(nn.Module):
    def __init__(self, dim, num_layers=3):
        super().__init__()
        self.layers = nn.Sequential(*[DetailNode(dim)
                                    for _ in range(num_layers)])

    def forward(self, x):
        return self.layers(x)


# ---------------------------------------------------------------------------
# Restormer Encoder
# ---------------------------------------------------------------------------

class Restormer_Encoder(nn.Module):
    """
    Restormer encoder. Encodes an input image into base and detail features.

    Args:
        inp_channels (int): number of input channels (default 3).
        dim (int): embedding dimension (default 64).
        num_blocks (list): [blocks_encoder, blocks_decoder].
        heads (list): attention heads per stage.
        ffn_expansion_factor (float): expansion ratio in FFN.
        bias (bool): use bias in convolutions.
        LayerNorm_type (str): 'WithBias' or 'BiasFree'.
    """

    def __init__(
        self,
        inp_channels=3,
        dim=64,
        num_blocks=None,
        heads=None,
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type='WithBias',
    ):
        super().__init__()
        if num_blocks is None:
            num_blocks = [4, 4]
        if heads is None:
            heads = [8, 8, 8]

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_blocks = nn.Sequential(
            *[TransformerBlock(dim, heads[0], ffn_expansion_factor, bias, LayerNorm_type)
              for _ in range(num_blocks[0])]
        )

        self.base_layer = BaseFeatureExtraction(
            dim, heads[2], ffn_expansion_factor, bias)
        self.detail_layer = DetailFeatureExtraction(dim)

    def forward(self, inp_img):
        inp_enc = self.patch_embed(inp_img)
        out_enc = self.encoder_blocks(inp_enc)
        base_feature = self.base_layer(out_enc)
        detail_feature = self.detail_layer(out_enc)
        return base_feature, detail_feature, out_enc


# ---------------------------------------------------------------------------
# Three-Branch Drought Classification Network
# ---------------------------------------------------------------------------

class DroughtClassifier(nn.Module):
    """
    三分支干旱分级五分类网络。

    分支1: RGB (3通道)
    分支2: 热红外 (3通道)
    分支3: 多光谱+植被指数 (8通道)

    每个分支输出 base_feature (B, dim, H, W)，三个分支特征在通道维度上拼接后
    经全局平均池化得到全局描述子，再通过全连接分类头输出5类预测。
    """

    def __init__(
        self,
        dim=64,
        num_blocks=None,
        heads=None,
        ffn_expansion_factor=2,
        bias=False,
        LayerNorm_type='WithBias',
        num_classes=5,
    ):
        super().__init__()
        if num_blocks is None:
            num_blocks = [4, 4]
        if heads is None:
            heads = [8, 8, 8]

        # RGB branch
        self.encoder_rgb = Restormer_Encoder(
            inp_channels=3,
            dim=dim,
            num_blocks=num_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )

        # Thermal infrared branch
        self.encoder_tir = Restormer_Encoder(
            inp_channels=3,
            dim=dim,
            num_blocks=num_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )

        # Multispectral + vegetation indices branch (8 channels)
        self.encoder_ms = Restormer_Encoder(
            inp_channels=8,
            dim=dim,
            num_blocks=num_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type,
        )

        # Classification head: 3 branches × dim → num_classes
        fused_dim = dim * 3  # 192 when dim=64
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, rgb, tir, ms):
        """
        Args:
            rgb: (B, 3, H, W)  visible light
            tir: (B, 3, H, W)  thermal infrared
            ms:  (B, 8, H, W)  multispectral + vegetation indices
        Returns:
            logits: (B, num_classes)
        """
        base1, _, _ = self.encoder_rgb(rgb)
        base2, _, _ = self.encoder_tir(tir)
        base3, _, _ = self.encoder_ms(ms)

        # Concatenate along channel dimension: (B, 3*dim, H, W)
        fused = torch.cat([base1, base2, base3], dim=1)

        # Global Average Pooling → (B, 3*dim)
        fused = fused.mean(dim=[2, 3])

        return self.classifier(fused)
