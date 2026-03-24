"""
net_drought_densenet_tir.py
热红外 (TIR) 专用 DenseNet-121 编码器。

DenseNet 的密集连接特性使低层特征可以直接参与分类，适合热成像中地表温度细节的传播。

架构：
  DenseNet-121 骨干（修改输入为 3 通道）
  → 全局平均池化
  → 投影层 (→ 64 维)，与原 Restormer 分支输出维度保持一致

可用类：
  DroughtClassifierTIR       - 完整独立 TIR 分类器（含分类头）
  DenseNetTIREncoder         - 仅编码器部分，输出 (B, out_dim, 1, 1)，用于多分支融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Dense Block 基础组件
# ---------------------------------------------------------------------------

class _DenseLayer(nn.Module):
    """DenseNet 中单个 Dense Layer（BN-ReLU-Conv1x1-BN-ReLU-Conv3x3）。"""

    def __init__(self, num_input_features: int, growth_rate: int,
                 bn_size: int = 4, drop_rate: float = 0.0):
        super().__init__()
        inter_features = bn_size * growth_rate
        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.conv1 = nn.Conv2d(num_input_features, inter_features,
                               kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_features)
        self.conv2 = nn.Conv2d(inter_features, growth_rate,
                               kernel_size=3, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, inputs):
        # inputs: Tensor 或 Tensor 列表（密集连接）
        if isinstance(inputs, (list, tuple)):
            x = torch.cat(inputs, dim=1)
        else:
            x = inputs
        out = self.conv1(F.relu(self.bn1(x), inplace=True))
        out = self.conv2(F.relu(self.bn2(out), inplace=True))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return out


class _DenseBlock(nn.Module):
    """若干 DenseLayer 组成一个 DenseBlock。"""

    def __init__(self, num_layers: int, num_input_features: int,
                 growth_rate: int, bn_size: int = 4, drop_rate: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                _DenseLayer(num_input_features + i * growth_rate,
                            growth_rate, bn_size, drop_rate)
            )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_feat = layer(features)
            features.append(new_feat)
        return torch.cat(features, dim=1)


class _Transition(nn.Module):
    """DenseBlock 之间的过渡层（BN-ReLU-Conv1x1-AvgPool2x2）。"""

    def __init__(self, num_input_features: int, num_output_features: int):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        self.conv = nn.Conv2d(num_input_features, num_output_features,
                              kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.pool(self.conv(F.relu(self.bn(x), inplace=True)))


# ---------------------------------------------------------------------------
# DenseNet-121 骨干（从零实现，无需 torchvision）
# ---------------------------------------------------------------------------

class DenseNet121(nn.Module):
    """
    DenseNet-121 骨干网络。

    Args:
        inp_channels (int): 输入通道数，默认 3（TIR）。
        growth_rate (int): 每层新增特征数 k，DenseNet-121 默认 32。
        block_config (tuple): 各 block 层数，DenseNet-121 = (6, 12, 24, 16)。
        num_init_features (int): 初始卷积输出通道数，默认 64。
        bn_size (int): bottleneck 放大系数，默认 4。
        drop_rate (float): dropout 比率，默认 0。
    """

    def __init__(
        self,
        inp_channels: int = 3,
        growth_rate: int = 32,
        block_config: tuple = (6, 12, 24, 16),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0.0,
    ):
        super().__init__()

        # 初始卷积层（与原论文一致，stride=2 下采样）
        self.features = nn.Sequential(
            nn.Conv2d(inp_channels, num_init_features,
                      kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
            )
            self.features.add_module(f"denseblock{i+1}", block)
            num_features = num_features + num_layers * growth_rate

            if i != len(block_config) - 1:
                out_features = num_features // 2
                trans = _Transition(num_features, out_features)
                self.features.add_module(f"transition{i+1}", trans)
                num_features = out_features

        self.features.add_module("norm_final", nn.BatchNorm2d(num_features))
        self.out_channels = num_features  # 记录最终输出通道数（1024 for DenseNet-121）

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)
        return x


# ---------------------------------------------------------------------------
# TIR 专用 DenseNet 编码器（供多分支融合使用）
# ---------------------------------------------------------------------------

class DenseNetTIREncoder(nn.Module):
    """
    热红外 (TIR) 专用编码器，输出固定维度的特征图。

    输入:  (B, 3, H, W)
    输出:  (B, out_dim)  经全局平均池化并投影后
    """

    def __init__(self, inp_channels: int = 3, out_dim: int = 64,
                 drop_rate: float = 0.0):
        super().__init__()
        self.backbone = DenseNet121(
            inp_channels=inp_channels, drop_rate=drop_rate)
        backbone_out = self.backbone.out_channels  # 1024

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(backbone_out, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W)
        Returns:
            feat: (B, out_dim)
        """
        feat = self.backbone(x)          # (B, 1024, H', W')
        feat = self.gap(feat)            # (B, 1024, 1, 1)
        feat = feat.view(feat.size(0), -1)  # (B, 1024)
        feat = self.proj(feat)           # (B, out_dim)
        return feat


# ---------------------------------------------------------------------------
# 独立 TIR 分类器（用于单模态训练）
# ---------------------------------------------------------------------------

class DroughtClassifierTIR(nn.Module):
    """
    热红外 (TIR) 单模态干旱五分类器。

    使用 DenseNet-121 骨干提取特征，经分类头输出 5 类预测。

    Args:
        inp_channels (int): 输入通道数，默认 3。
        dim (int): 编码器输出特征维度，默认 64（与 Restormer 分支对齐）。
        num_classes (int): 分类数，默认 5。
        drop_rate (float): DenseNet 内部 dropout 比率。
    """

    def __init__(
        self,
        inp_channels: int = 3,
        dim: int = 64,
        num_classes: int = 5,
        drop_rate: float = 0.0,
    ):
        super().__init__()
        self.encoder = DenseNetTIREncoder(
            inp_channels=inp_channels, out_dim=dim, drop_rate=drop_rate)

        self.classifier = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, tir):
        """
        Args:
            tir: (B, 3, H, W)
        Returns:
            logits: (B, num_classes)
        """
        feat = self.encoder(tir)          # (B, dim)
        return self.classifier(feat)


class DroughtClassifierTIRLite(nn.Module):
    """
    轻量版 TIR 分类器，使用较小的 DenseNet 变体（block_config 缩减）。

    适合快速实验与资源受限场景。
    """

    def __init__(
        self,
        inp_channels: int = 3,
        dim: int = 64,
        num_classes: int = 5,
    ):
        super().__init__()
        # 使用更小的 DenseNet 配置（block_config=(6, 12, 8, 4)）
        self.backbone = DenseNet121(
            inp_channels=inp_channels,
            growth_rate=32,
            block_config=(6, 12, 8, 4),
            num_init_features=64,
        )
        backbone_out = self.backbone.out_channels

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(backbone_out, dim)

        self.classifier = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, tir):
        feat = self.backbone(tir)
        feat = self.gap(feat).view(feat.size(0), -1)
        feat = F.relu(self.proj(feat), inplace=True)
        return self.classifier(feat)
