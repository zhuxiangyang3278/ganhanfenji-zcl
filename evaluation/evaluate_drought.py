"""评估模型"""
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from datasets.dataset_drought import build_datasets
from models.net_drought import DroughtClassifier
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
parser = argparse.ArgumentParser(description='评估干旱分级模型')
parser.add_argument('--model_path', type=str, required=True,
                    help='模型文件路径')
parser.add_argument('--csv_path', type=str, default='/home/zcl/addfuse/2025label_classic5.csv',
                    help='CSV文件路径')
parser.add_argument('--data_root', type=str, default='/home/zcl/addfuse/dataset/',
                    help='数据根目录')
parser.add_argument('--batch_size', type=int, default=4,
                    help='批次大小')
args = parser.parse_args()

CSV_PATH = args.csv_path
DATA_ROOT = args.data_root
MODEL_PATH = args.model_path
BATCH_SIZE = args.batch_size
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['Level 0', 'Level 1', 'Level 2', 'Level 3', 'Level 4']
RESULT_DIR = './results'


def evaluate_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for rgb, tir, ms, labels in tqdm(dataloader, desc='Evaluating'):
            rgb = rgb.to(device)
            tir = tir.to(device)
            ms = ms.to(device)

            outputs = model(rgb, tir, ms)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return np.array(all_preds), np.array(all_labels)


print("=" * 60)
print("干旱分级模型评估")
print("=" * 60)

# 加载数据
print(f"\n加载数据...")
_, val_ds = build_datasets(CSV_PATH, DATA_ROOT, target_size=(224, 224))
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=4)
print(f"验证集: {len(val_ds)} 个样本")

# 加载模型
print(f"\n加载模型: {MODEL_PATH}")
model = DroughtClassifier(
    dim=48, num_blocks=[3, 3], heads=[4, 4, 4],
    ffn_expansion_factor=2, num_classes=5
).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"✅ 模型加载成功 (Epoch: {checkpoint['epoch']})")

# 评估
print(f"\n开始评估...")
y_pred, y_true = evaluate_model(model, val_loader, DEVICE)

# 准确率
acc = accuracy_score(y_true, y_pred)
print("\n" + "=" * 60)
print(f"总体准确率: {acc:.4f} ({acc*100:.2f}%)")
print(f"{'='*60}")

# 分类报告
print(f"\n分类报告:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))

# 保存指标
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, average=None)

metrics_data = []
for i in range(len(CLASS_NAMES)):
    metrics_data.append({
        'Class': CLASS_NAMES[i],
        'Precision': f'{precision[i]:.4f}',
        'Recall': f'{recall[i]:.4f}',
        'F1-Score': f'{f1[i]:.4f}',
        'Support': int(support[i])
    })

df = pd.DataFrame(metrics_data)
csv_path = os.path.join(RESULT_DIR, 'metrics.csv')
df.to_csv(csv_path, index=False)
print(f"\n✅ 指标已保存: {csv_path}")

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
cm_path = os.path.join(RESULT_DIR, 'confusion_matrix.png')
plt.savefig(cm_path, dpi=300, bbox_inches='tight')
print(f"✅ 混淆矩阵已保存: {cm_path}")

print(f"\n{'='*60}")
print(f"✅ 评估完成！")
print(f"{'='*60}")
