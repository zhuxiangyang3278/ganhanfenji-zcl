"""
advanced_augmentation.py
遥感图像数据增强模块，专门针对干旱分级任务设计。

支持的数据增强：
1. 几何变换：旋转、缩放、翻转、裁剪
2. 颜色变换：亮度、对比度、饱和度调整
3. 噪声添加：高斯噪声、椒盐噪声
4. 弹性变换：模拟地形变化
5. 混合增强：CutMix, MixUp
"""

import numpy as np
import random
from scipy import ndimage

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️ OpenCV未安装，部分增强功能将使用替代方法")


class RemoteSensingAugmentation:
    """遥感图像数据增强类"""

    def __init__(self, augmentation_strength=0.5):
        """
        Args:
            augmentation_strength: 增强强度 (0.0-1.0)
        """
        self.augmentation_strength = augmentation_strength

    def apply_geometric_augmentation(self, rgb, tir, ms):
        """应用几何变换增强"""
        if random.random() < 0.7 * self.augmentation_strength:
            # 随机旋转 (-30° 到 30°)
            angle = random.uniform(-30, 30)
            rgb = self._rotate_image(rgb, angle)
            tir = self._rotate_image(tir, angle)
            ms = self._rotate_image(ms, angle)

        if random.random() < 0.6 * self.augmentation_strength:
            # 随机缩放 (0.8 到 1.2)
            scale = random.uniform(0.8, 1.2)
            rgb = self._scale_image(rgb, scale)
            tir = self._scale_image(tir, scale)
            ms = self._scale_image(ms, scale)

        if random.random() < 0.5 * self.augmentation_strength:
            # 随机翻转
            if random.random() < 0.5:
                rgb = np.flip(rgb, axis=-1).copy()
                tir = np.flip(tir, axis=-1).copy()
                ms = np.flip(ms, axis=-1).copy()
            if random.random() < 0.5:
                rgb = np.flip(rgb, axis=-2).copy()
                tir = np.flip(tir, axis=-2).copy()
                ms = np.flip(ms, axis=-2).copy()

        return rgb, tir, ms

    def apply_color_augmentation(self, rgb, tir, ms):
        """应用颜色变换增强（主要对RGB图像）"""
        if random.random() < 0.6 * self.augmentation_strength:
            # 亮度调整
            brightness = random.uniform(0.7, 1.3)
            rgb = rgb * brightness
            rgb = np.clip(rgb, 0, 1)

        if random.random() < 0.5 * self.augmentation_strength:
            # 对比度调整
            contrast = random.uniform(0.8, 1.2)
            mean = np.mean(rgb)
            rgb = (rgb - mean) * contrast + mean
            rgb = np.clip(rgb, 0, 1)

        if random.random() < 0.4 * self.augmentation_strength:
            # 高斯噪声
            noise_std = random.uniform(0, 0.05)
            noise = np.random.normal(0, noise_std, rgb.shape)
            rgb = rgb + noise
            rgb = np.clip(rgb, 0, 1)

        return rgb, tir, ms

    def apply_elastic_transform(self, rgb, tir, ms):
        """应用弹性变换（模拟地形变化）"""
        if random.random() < 0.3 * self.augmentation_strength:
            if not CV2_AVAILABLE:
                # 如果没有OpenCV，使用简单的几何变换替代
                return self.apply_geometric_augmentation(rgb, tir, ms)

            alpha = random.uniform(50, 150)  # 变形强度
            sigma = random.uniform(4, 8)    # 平滑度

            # 生成随机位移场
            shape = rgb.shape[1:]  # H, W
            dx = np.random.uniform(-1, 1, shape) * alpha
            dy = np.random.uniform(-1, 1, shape) * alpha

            # 平滑位移场
            dx = cv2.GaussianBlur(dx, (0, 0), sigma)
            dy = cv2.GaussianBlur(dy, (0, 0), sigma)

            # 应用弹性变换
            rgb = self._elastic_transform_image(rgb, dx, dy)
            tir = self._elastic_transform_image(tir, dx, dy)
            ms = self._elastic_transform_image(ms, dx, dy)

        return rgb, tir, ms

    def apply_mixup(self, rgb1, tir1, ms1, label1, rgb2, tir2, ms2, label2):
        """应用MixUp数据增强"""
        if random.random() < 0.3 * self.augmentation_strength:
            # 检查图像尺寸是否一致
            if rgb1.shape != rgb2.shape:
                # 尺寸不一致，返回原始样本
                return rgb1, tir1, ms1, label1

            lam = random.betavariate(0.4, 0.4)  # Beta分布参数

            # 混合图像
            rgb = lam * rgb1 + (1 - lam) * rgb2
            tir = lam * tir1 + (1 - lam) * tir2
            ms = lam * ms1 + (1 - lam) * ms2

            # 混合标签（软标签）
            label = lam * label1 + (1 - lam) * label2

            return rgb, tir, ms, label

        return rgb1, tir1, ms1, label1

    def apply_cutmix(self, rgb1, tir1, ms1, label1, rgb2, tir2, ms2, label2):
        """应用CutMix数据增强"""
        if random.random() < 0.3 * self.augmentation_strength:
            # 检查图像尺寸是否一致
            if rgb1.shape != rgb2.shape:
                # 尺寸不一致，返回原始样本
                return rgb1, tir1, ms1, label1

            H, W = rgb1.shape[1:]

            # 随机生成裁剪区域
            lam = np.random.beta(1.0, 1.0)
            cut_ratio = np.sqrt(1. - lam)
            cut_w = int(W * cut_ratio)
            cut_h = int(H * cut_ratio)

            # 随机位置
            cx = np.random.randint(0, W)
            cy = np.random.randint(0, H)

            # 边界处理
            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)

            # 应用CutMix
            rgb1[:, bby1:bby2, bbx1:bbx2] = rgb2[:, bby1:bby2, bbx1:bbx2]
            tir1[:, bby1:bby2, bbx1:bbx2] = tir2[:, bby1:bby2, bbx1:bbx2]
            ms1[:, bby1:bby2, bbx1:bbx2] = ms2[:, bby1:bby2, bbx1:bbx2]

            # 调整lambda值
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
            label = lam * label1 + (1 - lam) * label2

            return rgb1, tir1, ms1, label

        return rgb1, tir1, ms1, label1

    def __call__(self, rgb, tir, ms, label, second_sample=None):
        """应用完整的数据增强流程"""

        # 基础几何变换
        rgb, tir, ms = self.apply_geometric_augmentation(rgb, tir, ms)

        # 颜色变换（主要对RGB）
        rgb, tir, ms = self.apply_color_augmentation(rgb, tir, ms)

        # 弹性变换
        rgb, tir, ms = self.apply_elastic_transform(rgb, tir, ms)

        # 如果提供了第二个样本，应用混合增强
        if second_sample is not None:
            rgb2, tir2, ms2, label2 = second_sample

            if random.random() < 0.5:
                rgb, tir, ms, label = self.apply_mixup(rgb, tir, ms, label,
                                                       rgb2, tir2, ms2, label2)
            else:
                rgb, tir, ms, label = self.apply_cutmix(rgb, tir, ms, label,
                                                        rgb2, tir2, ms2, label2)

        return rgb, tir, ms, label

    # 辅助函数
    def _rotate_image(self, img, angle):
        """旋转图像"""
        if img.ndim == 3:
            rotated = []
            for i in range(img.shape[0]):
                rotated.append(ndimage.rotate(
                    img[i], angle, reshape=False, mode='reflect'))
            return np.stack(rotated, axis=0)
        else:
            return ndimage.rotate(img, angle, reshape=False, mode='reflect')

    def _scale_image(self, img, scale):
        """缩放图像"""
        if scale == 1.0:
            return img

        H, W = img.shape[-2:]
        new_H, new_W = int(H * scale), int(W * scale)

        if CV2_AVAILABLE:
            # 使用OpenCV进行高质量缩放
            if img.ndim == 3:
                scaled = []
                for i in range(img.shape[0]):
                    channel = cv2.resize(
                        img[i], (new_W, new_H), interpolation=cv2.INTER_LINEAR)
                    # 保持原始尺寸
                    if scale > 1:
                        # 裁剪中心区域
                        start_H = (new_H - H) // 2
                        start_W = (new_W - W) // 2
                        channel = channel[start_H:start_H+H, start_W:start_W+W]
                    else:
                        # 填充到原始尺寸
                        pad_H = (H - new_H) // 2
                        pad_W = (W - new_W) // 2
                        channel = np.pad(channel, ((pad_H, H-new_H-pad_H),
                                                   (pad_W, W-new_W-pad_W)),
                                         mode='reflect')
                    scaled.append(channel)
                return np.stack(scaled, axis=0)
            else:
                scaled = cv2.resize(img, (new_W, new_H),
                                    interpolation=cv2.INTER_LINEAR)
                if scale > 1:
                    start_H = (new_H - H) // 2
                    start_W = (new_W - W) // 2
                    return scaled[start_H:start_H+H, start_W:start_W+W]
                else:
                    pad_H = (H - new_H) // 2
                    pad_W = (W - new_W) // 2
                    return np.pad(scaled, ((pad_H, H-new_H-pad_H),
                                           (pad_W, W-new_W-pad_W)),
                                  mode='reflect')
        else:
            # 使用NumPy进行简单缩放（质量较低但可用）
            from scipy.ndimage import zoom

            if img.ndim == 3:
                zoom_factors = (1, scale, scale)
            else:
                zoom_factors = (scale, scale)

            scaled = zoom(img, zoom_factors, order=1)  # 双线性插值

            # 调整到原始尺寸
            if scaled.shape[-2] != H or scaled.shape[-1] != W:
                if scaled.shape[-2] > H or scaled.shape[-1] > W:
                    # 裁剪中心区域
                    start_H = (scaled.shape[-2] - H) // 2
                    start_W = (scaled.shape[-1] - W) // 2
                    if img.ndim == 3:
                        scaled = scaled[:, start_H:start_H +
                                        H, start_W:start_W+W]
                    else:
                        scaled = scaled[start_H:start_H+H, start_W:start_W+W]
                else:
                    # 填充到原始尺寸
                    pad_H = (H - scaled.shape[-2]) // 2
                    pad_W = (W - scaled.shape[-1]) // 2
                    if img.ndim == 3:
                        scaled = np.pad(scaled, ((0, 0), (pad_H, H-scaled.shape[-2]-pad_H),
                                                 (pad_W, W-scaled.shape[-1]-pad_W)),
                                        mode='reflect')
                    else:
                        scaled = np.pad(scaled, ((pad_H, H-scaled.shape[-2]-pad_H),
                                                 (pad_W, W-scaled.shape[-1]-pad_W)),
                                        mode='reflect')
            return scaled

    def _elastic_transform_image(self, img, dx, dy):
        """应用弹性变换到单张图像"""
        if img.ndim == 3:
            transformed = []
            for i in range(img.shape[0]):
                channel = img[i]
                H, W = channel.shape

                # 创建坐标网格
                x, y = np.meshgrid(np.arange(W), np.arange(H))

                # 应用位移
                indices_x = np.clip(x + dx, 0, W-1).astype(np.float32)
                indices_y = np.clip(y + dy, 0, H-1).astype(np.float32)

                # 重映射
                transformed_channel = cv2.remap(channel, indices_x, indices_y,
                                                cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                transformed.append(transformed_channel)

            return np.stack(transformed, axis=0)
        else:
            H, W = img.shape

            # 创建坐标网格
            x, y = np.meshgrid(np.arange(W), np.arange(H))

            # 应用位移
            indices_x = np.clip(x + dx, 0, W-1).astype(np.float32)
            indices_y = np.clip(y + dy, 0, H-1).astype(np.float32)

            # 重映射
            return cv2.remap(img, indices_x, indices_y,
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def create_augmented_dataset(original_dataset, augmentation_factor=5):
    """
    创建增强后的数据集

    Args:
        original_dataset: 原始数据集
        augmentation_factor: 增强倍数（每个样本生成多少增强版本）

    Returns:
        增强后的数据集（原始数据 + 增强数据）
    """
    import torch
    from torch.utils.data import Dataset

    class AugmentedDataset(Dataset):
        def __init__(self, original_dataset, augmentation_factor):
            self.original_dataset = original_dataset
            self.augmentation_factor = augmentation_factor
            self.augmenter = RemoteSensingAugmentation(
                augmentation_strength=0.7)

        def __len__(self):
            return len(self.original_dataset) * (self.augmentation_factor + 1)

        def __getitem__(self, idx):
            if idx < len(self.original_dataset):
                # 返回原始样本，但将标签转换为浮点数以保持一致性
                rgb, tir, ms, label = self.original_dataset[idx]
                return (
                    rgb.float(),
                    tir.float(),
                    ms.float(),
                    torch.tensor(float(label), dtype=torch.float32)
                )
            else:
                # 返回增强样本
                original_idx = (idx - len(self.original_dataset)
                                ) % len(self.original_dataset)
                rgb, tir, ms, label = self.original_dataset[original_idx]

                # 随机选择另一个样本用于混合增强
                second_idx = random.randint(0, len(self.original_dataset) - 1)
                rgb2, tir2, ms2, label2 = self.original_dataset[second_idx]

                # 应用增强
                rgb_aug, tir_aug, ms_aug, label_aug = self.augmenter(
                    rgb.numpy(), tir.numpy(), ms.numpy(), label,
                    second_sample=(rgb2.numpy(), tir2.numpy(),
                                   ms2.numpy(), label2)
                )

                return (
                    torch.from_numpy(rgb_aug).float(),
                    torch.from_numpy(tir_aug).float(),
                    torch.from_numpy(ms_aug).float(),
                    torch.tensor(float(label_aug), dtype=torch.float32)
                )

    return AugmentedDataset(original_dataset, augmentation_factor)
