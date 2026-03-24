"""
单模态RGB干旱分级模型评估
专门用于评估单模态RGB模型性能
"""

from models.net_drought_rgb import DroughtClassifierRGB
import argparse
import sys
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 命令行参数解析
parser = argparse.ArgumentParser(description='评估单模态RGB干旱分级模型')
parser.add_argument('--model_path', type=str, required=True,
                    help='模型文件路径')
parser.add_argument('--csv_path', type=str, default='/home/zcl/addfuse/2025label_classic5.csv',
                    help='CSV文件路径')
parser.add_argument('--data_root', type=str, default='/home/zcl/addfuse/dataset/',
                    help='数据根目录')
parser.add_argument('--batch_size', type=int, default=8,
                    help='批次大小')
args = parser.parse_args()

CSV_PATH = args.csv_path
DATA_ROOT = args.data_root
MODEL_PATH = args.model_path
BATCH_SIZE = args.batch_size
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['Level 0', 'Level 1', 'Level 2', 'Level 3', 'Level 4']
RESULT_DIR = './results_rgb'


def load_rgb_data(csv_path, data_root, ids):
    """加载RGB数据"""
    from datasets.dataset_drought import read_envi_bands, get_file_paths
    import cv2

    rgb_data = []
    labels = []

    df = pd.read_csv(csv_path)
    df = df[df['id'].isin(ids)].reset_index(drop=True)

    for sample_id in tqdm(ids, desc='加载RGB数据'):
        paths = get_file_paths(sample_id, data_root)

        # 仅加载RGB图像
        rgb = read_envi_bands(paths['rgb'], [0, 1, 2])  # (3, H, W)

        # 调整图像尺寸到统一大小
        target_size = (224, 224)
        if rgb.shape[1:] != target_size:
            rgb_resized = np.transpose(rgb, (1, 2, 0))
            rgb_resized = cv2.resize(rgb_resized, target_size[::-1])
            rgb = np.transpose(rgb_resized, (2, 0, 1))

        rgb_data.append(rgb)

        # 获取标签
        label = df[df['id'] == sample_id]['label'].values[0]
        labels.append(label)

    return rgb_data, labels


class RGBDataset(torch.utils.data.Dataset):
    """单模态RGB数据集"""

    def __init__(self, csv_path, data_root, ids):
        self.csv_path = csv_path
        self.data_root = data_root
        self.ids = ids
        self.target_size = (224, 224)

        # 存储文件路径和标签
        df = pd.read_csv(csv_path)
        df = df[df['id'].isin(ids)].reset_index(drop=True)
        self.labels = df['label'].tolist()

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        sample_id = self.ids[idx]
        label = self.labels[idx]

        # 实时加载RGB数据
        from datasets.dataset_drought import read_envi_bands, get_file_paths
        import cv2

        paths = get_file_paths(sample_id, self.data_root)
        rgb = read_envi_bands(paths['rgb'], [0, 1, 2])  # (3, H, W)

        # 调整图像尺寸到统一大小
        if rgb.shape[1:] != self.target_size:
            rgb_resized = np.transpose(rgb, (1, 2, 0))
            rgb_resized = cv2.resize(rgb_resized, self.target_size[::-1])
            rgb = np.transpose(rgb_resized, (2, 0, 1))

        # 转换为张量
        rgb_tensor = torch.tensor(rgb, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return rgb_tensor, label_tensor


def evaluate_model(model, dataloader, device):
    """评估模型"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for rgb, labels in tqdm(dataloader, desc='评估'):
            rgb, labels = rgb.to(device), labels.to(device)

            outputs = model(rgb)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return all_preds, all_labels


def main():
    print("=" * 60)
    print("单模态RGB干旱分级模型评估")
    print("=" * 60)

    # 创建结果目录
    os.makedirs(RESULT_DIR, exist_ok=True)

    # 加载数据
    print("\n加载数据...")
    df = pd.read_csv(CSV_PATH)
    ids = df['id'].tolist()
    labels = df['label'].tolist()

    # 使用全部数据进行评估
    dataset = RGBDataset(CSV_PATH, DATA_ROOT, ids)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print(f"评估数据集: {len(dataset)} 个样本")

    # 加载模型
    print("\n加载模型...")
    model = DroughtClassifierRGB(
        dim=48,
        num_blocks=[4, 6],
        heads=[1, 2, 4, 8],
        num_classes=5,
    ).to(DEVICE)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 获取最佳准确率
    best_val_acc = checkpoint.get('val_acc', 0.0)
    best_epoch = checkpoint.get('epoch', 0)

    print(f"最佳模型: Epoch {best_epoch}, 验证准确率: {best_val_acc:.2%}")

    # 评估模型
    print("\n评估模型...")
    all_preds, all_labels = evaluate_model(model, dataloader, DEVICE)

    # 计算准确率
    accuracy = accuracy_score(all_labels, all_preds)

    print("\n" + "=" * 60)
    print("评估结果")
    print("=" * 60)
    print(f"总体准确率: {accuracy:.2%}")

    # 分类报告
    print("\n分类报告:")
    report = classification_report(
        all_labels, all_preds, target_names=CLASS_NAMES, digits=4)
    print(report)

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title(f'单模态RGB混淆矩阵\n准确率: {accuracy:.2%}')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_DIR, 'confusion_matrix_rgb.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 每个类别的准确率
    print("\n每个类别准确率:")
    for i in range(5):
        class_mask = np.array(all_labels) == i
        if class_mask.any():
            class_acc = accuracy_score(
                np.array(all_labels)[class_mask],
                np.array(all_preds)[class_mask]
            )
            print(f"  {CLASS_NAMES[i]}: {class_acc:.2%}")

    # 保存评估结果
    results = {
        'accuracy': accuracy,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'best_epoch': best_epoch,
        'best_val_acc': best_val_acc
    }

    import json
    with open(os.path.join(RESULT_DIR, 'evaluation_results_rgb.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n评估结果已保存到: {RESULT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
