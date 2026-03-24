"""
net_drought_multiarch.py
多架构组合干旱分类模型。

支持为每个模态（RGB / TIR / MS）独立选择编码器架构，方便逐分支实验与对比。

可选架构：
  arch_rgb  : 'restormer'  (默认，使用 Restormer_Encoder)
              'efficientnet' (轻量 EfficientNet 变体，从零实现，无需 timm)
  arch_tir  : 'restormer'  (默认)
              'densenet'   (DenseNet-121，来自 net_drought_densenet_tir)
  arch_ms   : 'restormer'  (默认)
              'hybridsn'   (HybridSN，来自 net_drought_hybridsn_ms)

分类头保持与原 DroughtClassifier 一致：
  特征拼接 (B, 3*dim) → GAP → FC 192→256→5

用法示例：
    model = DroughtClassifierMultiArch(
        arch_rgb='restormer',
        arch_tir='densenet',
        arch_ms='hybridsn',
        dim=64,
        num_classes=5,
    )
    out = model(rgb, tir, ms)   # (B, 5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import numbers

# 从项目模型文件导入专用编码器
from models.net_drought_densenet_tir import DenseNetTIREncoder
from models.net_drought_hybridsn_ms import HybridSNMSEncoder


# ---------------------------------------------------------------------------
# 复用 net_drought.py 中的 Restormer 组件（避免重复依赖导入问题）
# ---------------------------------------------------------------------------

class _BiasFreeLayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class _WithBiasLayerNorm(nn.Module):
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


class _LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='WithBias'):
        super().__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = _BiasFreeLayerNorm(dim)
        else:
            self.body = _WithBiasLayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return _to_4d(self.body(_to_3d(x)), h, w)


def _to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def _to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class _FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2, bias=False):
        super().__init__()
        hidden = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden * 2, hidden * 2, kernel_size=3,
                                stride=1, padding=1,
                                groups=hidden * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        return self.project_out(F.gelu(x1) * x2)


class _Attention(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super().__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                    stride=1, padding=1,
                                    groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        out = attn.softmax(dim=-1) @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class _TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, ffn_expansion_factor=2,
                 bias=False, LayerNorm_type='WithBias'):
        super().__init__()
        self.norm1 = _LayerNorm(dim, LayerNorm_type)
        self.attn = _Attention(dim, num_heads, bias)
        self.norm2 = _LayerNorm(dim, LayerNorm_type)
        self.ffn = _FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class _OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super().__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3,
                              stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class _AttentionBase(nn.Module):
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
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        out = attn.softmax(dim=-1) @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w',
                        head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class _Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, bias=False):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2,
                             kernel_size=1, bias=bias)
        self.fc2 = nn.Conv2d(hidden_features, in_features,
                             kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.fc1(x)
        x1, x2 = x.chunk(2, dim=1)
        return self.fc2(F.gelu(x1) * x2)


class _BaseFeatureExtraction(nn.Module):
    def __init__(self, dim, num_heads=8, ffn_expansion_factor=2, bias=False):
        super().__init__()
        self.norm1 = _LayerNorm(dim)
        self.attn = _AttentionBase(dim, num_heads, bias)
        self.norm2 = _LayerNorm(dim)
        self.mlp = _Mlp(dim, int(dim * ffn_expansion_factor))

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class _InvertedResidualBlock(nn.Module):
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


class _DetailNode(nn.Module):
    def __init__(self, dim):
        super().__init__()
        c = dim // 2
        self.theta_phi = _InvertedResidualBlock(c, c)
        self.theta_rho = _InvertedResidualBlock(c, c)
        self.theta_eta = _InvertedResidualBlock(c, c)

    def forward(self, z):
        z1, z2 = z.chunk(2, dim=1)
        z2 = z2 + self.theta_phi(z1)
        z1 = z1 * torch.exp(self.theta_rho(z2)) + self.theta_eta(z2)
        return torch.cat([z1, z2], dim=1)


class _DetailFeatureExtraction(nn.Module):
    def __init__(self, dim, num_layers=3):
        super().__init__()
        self.layers = nn.Sequential(*[_DetailNode(dim) for _ in range(num_layers)])

    def forward(self, x):
        return self.layers(x)


class RestormerBranchEncoder(nn.Module):
    """
    与 net_drought.py 中 Restormer_Encoder 等价的独立实现。
    输出 base_feature (B, dim, H, W)，再由调用方 GAP 压缩为 (B, dim)。
    """

    def __init__(self, inp_channels=3, dim=64, num_blocks=None,
                 heads=None, ffn_expansion_factor=2, bias=False,
                 LayerNorm_type='WithBias'):
        super().__init__()
        if num_blocks is None:
            num_blocks = [4, 4]
        if heads is None:
            heads = [8, 8, 8]

        self.patch_embed = _OverlapPatchEmbed(inp_channels, dim)
        self.encoder_blocks = nn.Sequential(
            *[_TransformerBlock(dim, heads[0], ffn_expansion_factor,
                                bias, LayerNorm_type)
              for _ in range(num_blocks[0])]
        )
        self.base_layer = _BaseFeatureExtraction(
            dim, heads[2], ffn_expansion_factor, bias)
        self.detail_layer = _DetailFeatureExtraction(dim)

    def forward(self, x):
        """Returns base_feature (B, dim, H, W)."""
        enc = self.encoder_blocks(self.patch_embed(x))
        return self.base_layer(enc)  # (B, dim, H, W)


# ---------------------------------------------------------------------------
# 轻量 EfficientNet 风格编码器（从零实现，无需 timm/torchvision）
# ---------------------------------------------------------------------------

class _MBConv(nn.Module):
    """Mobile Inverted Bottleneck Conv（EfficientNet 基础块）。"""

    def __init__(self, in_ch, out_ch, expand_ratio=6, stride=1):
        super().__init__()
        mid_ch = in_ch * expand_ratio
        layers = []
        if expand_ratio != 1:
            layers += [nn.Conv2d(in_ch, mid_ch, 1, bias=False),
                       nn.BatchNorm2d(mid_ch), nn.SiLU(inplace=True)]
        layers += [
            nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1,
                      groups=mid_ch, bias=False),
            nn.BatchNorm2d(mid_ch), nn.SiLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)
        self.use_skip = (stride == 1 and in_ch == out_ch)

    def forward(self, x):
        out = self.conv(x)
        if self.use_skip:
            out = out + x
        return out


class EfficientNetRGBEncoder(nn.Module):
    """
    轻量 EfficientNet-B0 风格编码器，适配 RGB（3 通道）输入。
    输出 (B, out_dim)。
    """

    def __init__(self, inp_channels: int = 3, out_dim: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(inp_channels, 32, kernel_size=3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            _MBConv(32, 16, expand_ratio=1),
            _MBConv(16, 24, expand_ratio=6, stride=2),
            _MBConv(24, 24, expand_ratio=6),
            _MBConv(24, 40, expand_ratio=6, stride=2),
            _MBConv(40, 40, expand_ratio=6),
            _MBConv(40, 80, expand_ratio=6, stride=2),
            _MBConv(80, 80, expand_ratio=6),
            _MBConv(80, 112, expand_ratio=6),
            _MBConv(112, 192, expand_ratio=6, stride=2),
            _MBConv(192, 192, expand_ratio=6),
            _MBConv(192, 320, expand_ratio=6),
        )
        self.head = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(1280, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.gap(x).view(x.size(0), -1)
        return self.proj(x)


# ---------------------------------------------------------------------------
# 多架构组合干旱分类模型
# ---------------------------------------------------------------------------

class DroughtClassifierMultiArch(nn.Module):
    """
    多架构组合干旱五分类网络。

    可为每个模态独立选择编码器架构：
      arch_rgb : 'restormer' | 'efficientnet'
      arch_tir : 'restormer' | 'densenet'
      arch_ms  : 'restormer' | 'hybridsn'

    所有编码器输出统一压缩至 dim 维，再拼接 → 分类头。

    Args:
        arch_rgb  (str): RGB 分支架构，默认 'restormer'。
        arch_tir  (str): TIR 分支架构，默认 'restormer'。
        arch_ms   (str): MS 分支架构，默认 'restormer'。
        dim       (int): 每个分支输出特征维度，默认 64。
        num_classes (int): 分类数，默认 5。
        num_blocks (list): Restormer 编码器块数，默认 [4, 4]。
        heads     (list): Restormer 注意力头数，默认 [8, 8, 8]。
    """

    _SUPPORTED_RGB = ('restormer', 'efficientnet')
    _SUPPORTED_TIR = ('restormer', 'densenet')
    _SUPPORTED_MS  = ('restormer', 'hybridsn')

    def __init__(
        self,
        arch_rgb: str = 'restormer',
        arch_tir: str = 'restormer',
        arch_ms: str = 'restormer',
        dim: int = 64,
        num_classes: int = 5,
        num_blocks=None,
        heads=None,
    ):
        super().__init__()
        if num_blocks is None:
            num_blocks = [4, 4]
        if heads is None:
            heads = [8, 8, 8]

        # 验证架构选择
        if arch_rgb not in self._SUPPORTED_RGB:
            raise ValueError(
                f"arch_rgb must be one of {self._SUPPORTED_RGB}, got '{arch_rgb}'")
        if arch_tir not in self._SUPPORTED_TIR:
            raise ValueError(
                f"arch_tir must be one of {self._SUPPORTED_TIR}, got '{arch_tir}'")
        if arch_ms not in self._SUPPORTED_MS:
            raise ValueError(
                f"arch_ms must be one of {self._SUPPORTED_MS}, got '{arch_ms}'")

        self.arch_rgb = arch_rgb
        self.arch_tir = arch_tir
        self.arch_ms  = arch_ms
        self.dim = dim

        # ------------------------------------------------------------------
        # RGB 分支
        # ------------------------------------------------------------------
        if arch_rgb == 'restormer':
            self.encoder_rgb = RestormerBranchEncoder(
                inp_channels=3, dim=dim,
                num_blocks=num_blocks, heads=heads)
            self.rgb_gap = nn.AdaptiveAvgPool2d(1)
            self._rgb_out_spatial = True
        else:  # efficientnet
            self.encoder_rgb = EfficientNetRGBEncoder(
                inp_channels=3, out_dim=dim)
            self._rgb_out_spatial = False

        # ------------------------------------------------------------------
        # TIR 分支
        # ------------------------------------------------------------------
        if arch_tir == 'restormer':
            self.encoder_tir = RestormerBranchEncoder(
                inp_channels=3, dim=dim,
                num_blocks=num_blocks, heads=heads)
            self.tir_gap = nn.AdaptiveAvgPool2d(1)
            self._tir_out_spatial = True
        else:  # densenet
            self.encoder_tir = DenseNetTIREncoder(
                inp_channels=3, out_dim=dim)
            self._tir_out_spatial = False

        # ------------------------------------------------------------------
        # MS 分支
        # ------------------------------------------------------------------
        if arch_ms == 'restormer':
            self.encoder_ms = RestormerBranchEncoder(
                inp_channels=8, dim=dim,
                num_blocks=num_blocks, heads=heads)
            self.ms_gap = nn.AdaptiveAvgPool2d(1)
            self._ms_out_spatial = True
        else:  # hybridsn
            self.encoder_ms = HybridSNMSEncoder(
                inp_channels=8, out_dim=dim)
            self._ms_out_spatial = False

        # ------------------------------------------------------------------
        # 分类头（与原 DroughtClassifier 一致）
        # ------------------------------------------------------------------
        fused_dim = dim * 3
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def _encode(self, encoder, x, out_spatial, gap_module):
        """统一处理空间图（需 GAP）和向量（直接使用）两种输出。"""
        feat = encoder(x)
        if out_spatial:
            feat = gap_module(feat).view(feat.size(0), -1)
        return feat

    def forward(self, rgb, tir, ms):
        """
        Args:
            rgb: (B, 3, H, W)  可见光
            tir: (B, 3, H, W)  热红外
            ms:  (B, 8, H, W)  多光谱 + 植被指数
        Returns:
            logits: (B, num_classes)
        """
        rgb_feat = self._encode(self.encoder_rgb, rgb,
                                self._rgb_out_spatial,
                                getattr(self, 'rgb_gap', None))
        tir_feat = self._encode(self.encoder_tir, tir,
                                self._tir_out_spatial,
                                getattr(self, 'tir_gap', None))
        ms_feat  = self._encode(self.encoder_ms, ms,
                                self._ms_out_spatial,
                                getattr(self, 'ms_gap', None))

        fused = torch.cat([rgb_feat, tir_feat, ms_feat], dim=1)  # (B, 3*dim)
        return self.classifier(fused)

    def extra_repr(self):
        return (f"arch_rgb={self.arch_rgb}, arch_tir={self.arch_tir}, "
                f"arch_ms={self.arch_ms}, dim={self.dim}")
