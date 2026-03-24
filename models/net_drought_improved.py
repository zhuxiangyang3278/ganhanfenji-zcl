"""
改进版干旱分类模型（集成跨模态注意力融合）
基于原始 net_drought.py，添加注意力融合模块
"""

# 复制原始模型代码
import torch
import torch.nn as nn
from fusion_module import ModalityFusionBlock
from net_drought import (
    Restormer_Encoder,
    MS_Encoder,
    CrossModalFusion,
    DroughtClassifier
)
import sys
sys.path.insert(0, '/home/zcl/addfuse')


class DroughtClassifierImproved(nn.Module):
    """
    改进版多模态干旱分类器
    - 添加跨模态注意力融合
    - 保持原有 Restormer 架构
    """

    def __init__(self, dim=48, num_blocks=[3, 3], heads=[6, 6, 6],
                 ffn_expansion_factor=2, num_classes=5):
        super().__init__()

        # === RGB Encoder ===
        self.encoder_rgb = Restormer_Encoder(
            inp_channels=3,
            dim=dim,
            num_blocks=num_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=False,
            LayerNorm_type='WithBias',
        )

        # === TIR Encoder ===
        self.encoder_tir = Restormer_Encoder(
            inp_channels=3,
            dim=dim,
            num_blocks=num_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=False,
            LayerNorm_type='WithBias',
        )

        # === MS Encoder ===
        self.encoder_ms = Restormer_Encoder(
            inp_channels=8,
            dim=dim,
            num_blocks=num_blocks,
            heads=heads,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=False,
            LayerNorm_type='WithBias',
        )

        # === 跨模态注意力融合（新增）===
        self.fusion_block = ModalityFusionBlock(dim)

        # === 分类头 ===
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, rgb, tir, ms):
        """
        rgb: (B, 3, H, W)
        tir: (B, 3, H, W)
        ms:  (B, 8, H, W)
        """
        # 特征提取
        rgb_feat, _, _ = self.encoder_rgb(rgb)  # (B, dim, H', W')
        tir_feat, _, _ = self.encoder_tir(tir)
        ms_feat, _, _ = self.encoder_ms(ms)

        # 跨模态注意力融合（核心改进）
        fused_feat = self.fusion_block(rgb_feat, tir_feat, ms_feat)

        # 全局池化 + 分类
        pooled = self.global_pool(fused_feat)  # (B, dim, 1, 1)
        logits = self.classifier(pooled)        # (B, num_classes)

        return logits
