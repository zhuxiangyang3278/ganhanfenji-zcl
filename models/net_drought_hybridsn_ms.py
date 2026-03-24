"""
net_drought_hybridsn_ms.py
多光谱 (MS) 专用 HybridSN 编码器。

HybridSN 结合 3D-CNN（捕捉光谱依赖）和 2D-CNN（捕捉空间特征），
专为多光谱/高光谱数据设计，适合本项目的 8 通道 MS 输入
（NIR, Red, Blue, Green, RedEdge 5个多光谱通道 + NDVI/GNDVI/SAVI 3个植被指数）。

参考论文：
  Roy, S.K. et al. "HybridSN: Exploring 3-D–2-D CNN Feature Hierarchy
  for Hyperspectral Image Classification", IEEE GRSL, 2020.

架构（针对遥感分类任务改编）：
  输入 (B, 8, H, W)
  → reshape 为 (B, 1, 8, H, W)（将通道视为光谱维度）
  → 3D-CNN 模块（光谱-空间联合特征提取）
  → reshape 为 (B, C', H', W')
  → 2D-CNN 模块（空间细节特征提取）
  → 全局平均池化
  → 投影层 (→ 64 维)

可用类：
  DroughtClassifierMSHybridSN   - 完整独立 MS 分类器（含分类头）
  HybridSNMSEncoder             - 仅编码器部分，输出 (B, out_dim)，用于多分支融合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# HybridSN 核心模块
# ---------------------------------------------------------------------------

class HybridSNEncoder(nn.Module):
    """
    HybridSN 主体编码器。

    3D-CNN 处理光谱维度（将 8 通道视为"光谱帧"），
    2D-CNN 在 3D 特征 reshape 后提取空间特征。

    Args:
        in_channels (int): 输入通道数（光谱维度），默认 8。
        out_dim (int): 输出特征维度，默认 64。
        spatial_size (int): 输入空间分辨率，默认 224。
    """

    def __init__(self, in_channels: int = 8, out_dim: int = 64,
                 spatial_size: int = 224):
        super().__init__()
        self.in_channels = in_channels

        # 3D-CNN
        self.conv3d_1 = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(7, 3, 3),
                      padding=(3, 1, 1), bias=False),  # padding 保持光谱维度不变
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
        )
        self.conv3d_2 = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=(5, 3, 3),
                      padding=(2, 1, 1), bias=False),  # padding 保持光谱维度不变
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.conv3d_3 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3),
                      padding=(1, 1, 1), bias=False),  # padding 保持光谱维度不变
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        # 3D 输出通道数 = 32 * in_channels（光谱维度保持不变，与通道合并）
        channels_after_3d = 32 * in_channels

        # ------------------------------------------------------------------
        # 2D-CNN 模块（在 3D 特征 reshape 后应用）
        # ------------------------------------------------------------------
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(channels_after_3d, 64, kernel_size=3,
                      padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # 全局平均池化 + 投影
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(128, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
        )
        self.out_dim = out_dim

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)  C=8 多光谱通道
        Returns:
            feat: (B, out_dim)
        """
        B, C, H, W = x.shape

        # 将通道维扩展为光谱维：(B, 1, C, H, W)
        x = x.unsqueeze(1)

        # 3D-CNN
        x = self.conv3d_1(x)   # (B, 8,  C, H, W)
        x = self.conv3d_2(x)   # (B, 16, C, H, W)
        x = self.conv3d_3(x)   # (B, 32, C, H, W)

        # reshape 为 2D 特征图：(B, 32*C, H, W)
        B2, F3d, Cs, Hs, Ws = x.shape
        x = x.view(B2, F3d * Cs, Hs, Ws)

        # 2D-CNN
        x = self.conv2d_1(x)   # (B, 64,  H, W)
        x = self.conv2d_2(x)   # (B, 128, H, W)

        # 全局平均池化 + 投影
        x = self.gap(x).view(B2, -1)   # (B, 128)
        feat = self.proj(x)             # (B, out_dim)
        return feat


# ---------------------------------------------------------------------------
# MS 专用编码器（对外接口，与 DenseNetTIREncoder 保持一致）
# ---------------------------------------------------------------------------

class HybridSNMSEncoder(nn.Module):
    """
    多光谱 (MS) 专用 HybridSN 编码器，用于多分支融合。

    输入:  (B, 8, H, W)
    输出:  (B, out_dim)
    """

    def __init__(self, inp_channels: int = 8, out_dim: int = 64,
                 spatial_size: int = 224):
        super().__init__()
        self.encoder = HybridSNEncoder(
            in_channels=inp_channels, out_dim=out_dim,
            spatial_size=spatial_size)
        self.out_dim = out_dim

    def forward(self, x):
        return self.encoder(x)


# ---------------------------------------------------------------------------
# 独立 MS 分类器（用于单模态训练）
# ---------------------------------------------------------------------------

class DroughtClassifierMSHybridSN(nn.Module):
    """
    多光谱 (MS) 单模态干旱五分类器，使用 HybridSN 编码器。

    Args:
        inp_channels (int): 输入通道数，默认 8。
        dim (int): 编码器输出特征维度，默认 64（与 Restormer 分支对齐）。
        num_classes (int): 分类数，默认 5。
        spatial_size (int): 输入空间分辨率，默认 224。
    """

    def __init__(
        self,
        inp_channels: int = 8,
        dim: int = 64,
        num_classes: int = 5,
        spatial_size: int = 224,
    ):
        super().__init__()
        self.encoder = HybridSNMSEncoder(
            inp_channels=inp_channels, out_dim=dim,
            spatial_size=spatial_size)

        self.classifier = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, ms):
        """
        Args:
            ms: (B, 8, H, W)
        Returns:
            logits: (B, num_classes)
        """
        feat = self.encoder(ms)          # (B, dim)
        return self.classifier(feat)


class DroughtClassifierMSHybridSNLite(nn.Module):
    """
    轻量版 HybridSN MS 分类器，使用更少的 3D/2D 卷积滤波器。

    适合快速实验与资源受限场景。
    """

    def __init__(
        self,
        inp_channels: int = 8,
        dim: int = 64,
        num_classes: int = 5,
    ):
        super().__init__()

        # 轻量 3D-CNN
        self.conv3d = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1),
                      bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1),
                      bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        # 16 * inp_channels 通道输入 2D-CNN
        channels_2d = 16 * inp_channels

        self.conv2d = nn.Sequential(
            nn.Conv2d(channels_2d, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(64, dim)

        self.classifier = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, ms):
        B, C, H, W = ms.shape
        x = ms.unsqueeze(1)           # (B, 1, C, H, W)
        x = self.conv3d(x)            # (B, 16, C, H, W)
        x = x.view(B, -1, H, W)      # (B, 16*C, H, W)
        x = self.conv2d(x)            # (B, 64, H, W)
        x = self.gap(x).view(B, -1)  # (B, 64)
        x = F.relu(self.proj(x), inplace=True)   # (B, dim)
        return self.classifier(x)
