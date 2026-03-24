"""测试脚本"""
import torch
from net_drought import DroughtClassifier
from dataset_drought import DroughtDataset
import pandas as pd

CSV_PATH = '/home/zcl/addfuse/2025label_classic5_complete.csv'
DATA_ROOT = '/home/zcl/addfuse/dataset/'
MODEL_PATH = './models/drought_best.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['Level 0', 'Level 1', 'Level 2', 'Level 3', 'Level 4']

print("=" * 60)
print("干旱分级模型测试")
print("=" * 60)

# 加载模型
print(f"\n加载模型...")
model = DroughtClassifier(
    dim=48, num_blocks=[3, 3], heads=[6, 6, 6],
    ffn_expansion_factor=2, num_classes=5
).to(DEVICE)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
print(f"✅ 模型加载成功！")

# 加载数据
df = pd.read_csv(CSV_PATH)
all_ids = df['id'].tolist()
dataset = DroughtDataset(CSV_PATH, DATA_ROOT, all_ids,
                         augment=False, target_size=(224, 224))

print(f"\n{'='*60}")
print(f"测试前 10 个样本:")
print(f"{'='*60}")
print(f"{'ID':<6} {'真实':<10} {'预测':<10} {'置信度':<10} {'正确':<6}")
print(f"{'-'*60}")

correct = 0
with torch.no_grad():
    for i in range(min(10, len(dataset))):
        rgb, tir, ms, true_label = dataset[i]
        sample_id = dataset.ids[i]

        rgb = rgb.unsqueeze(0).to(DEVICE)
        tir = tir.unsqueeze(0).to(DEVICE)
        ms = ms.unsqueeze(0).to(DEVICE)

        outputs = model(rgb, tir, ms)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_class = outputs.argmax(dim=1).item()
        confidence = probs[pred_class].item()

        is_correct = '✅' if pred_class == true_label else '❌'
        if pred_class == true_label:
            correct += 1

        print(f"{sample_id:<6} {CLASS_NAMES[true_label]:<10} {CLASS_NAMES[pred_class]:<10} "
              f"{confidence*100:>6.2f}%    {is_correct}")

print(f"{'-'*60}")
print(f"准确率: {correct}/10 = {correct/10*100:.1f}%")
print(f"{'='*60}")
