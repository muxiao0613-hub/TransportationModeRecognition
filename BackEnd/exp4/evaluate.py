"""
exp4/evaluate.py
======================
基于exp3，使用LabelSmoothingFocalLoss
直接读取exp3/cache/processed_features.pkl
"""
import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
EXP3_DIR = os.path.join(PARENT_DIR, 'exp3')
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)
sys.path.insert(0, EXP3_DIR)
os.chdir(SCRIPT_DIR)

from exp3.src.model_weather import TransportationModeClassifierWithWeather
from common.focal_loss import LabelSmoothingFocalLoss

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

TRAJECTORY_FEATURE_DIM = 21
WEATHER_FEATURE_DIM    = 10

EXP3_CACHE_PATH = os.path.join(EXP3_DIR, 'cache', 'processed_features.pkl')
OUTPUT_DIR         = 'evaluation_results'


class TrajectoryDataset(Dataset):
    def __init__(self, all_features,
                 traj_mean=None, traj_std=None,
                 weather_mean=None, weather_std=None,
                 stats_mean=None, stats_std=None):
        self.data = all_features
        self.traj_mean  = traj_mean
        self.traj_std   = traj_std
        self.weather_mean = weather_mean
        self.weather_std  = weather_std
        self.stats_mean = stats_mean
        self.stats_std  = stats_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        traj, stats, label, weather = self.data[idx]
        traj    = traj.astype(np.float32)
        stats   = stats.astype(np.float32)
        weather = weather.astype(np.float32)

        if self.traj_mean is not None:
            traj    = (traj    - self.traj_mean)    / (self.traj_std    + 1e-8)
        if self.weather_mean is not None:
            weather = (weather - self.weather_mean) / (self.weather_std + 1e-8)
        if self.stats_mean is not None:
            stats   = (stats   - self.stats_mean)   / (self.stats_std   + 1e-8)

        return (torch.FloatTensor(traj),
                torch.FloatTensor(weather),
                torch.FloatTensor(stats),
                torch.LongTensor([label]).squeeze())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/exp4_model.pth')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    DEVICE = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print("Exp4 模型评估 (LabelSmoothing + Focal Loss)")
    print("=" * 60)

    # 1. 加载模型
    print(f"\n[1/4] 加载模型: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"找不到模型: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    label_encoder = checkpoint['label_encoder']
    config = checkpoint.get('model_config', {})
    norm_params = checkpoint.get('norm_params', {})

    traj_mean    = norm_params.get('traj_mean', None)
    traj_std     = norm_params.get('traj_std', None)
    weather_mean = norm_params.get('weather_mean', None)
    weather_std  = norm_params.get('weather_std', None)
    stats_mean   = norm_params.get('stats_mean', None)
    stats_std    = norm_params.get('stats_std', None)

    if traj_mean is None:
        print("❌ checkpoint 缺少 norm_params，请重新训练 exp4/train.py")
        return

    model = TransportationModeClassifierWithWeather(
        TRAJECTORY_FEATURE_DIM, WEATHER_FEATURE_DIM, 18,
        config.get('hidden_dim', 128), config.get('num_layers', 2),
        len(label_encoder.classes_), config.get('dropout', 0.3),
        num_segments=5, local_hidden=64, global_hidden=128,
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"   ✓ 模型加载完成")

    # 2. 加载测试数据
    print(f"\n[2/4] 加载测试数据...")
    if not os.path.exists(EXP3_CACHE_PATH):
        raise FileNotFoundError(f"EXP3 缓存不存在: {EXP3_CACHE_PATH}\n请先运行 exp3/train.py")

    with open(EXP3_CACHE_PATH, 'rb') as f:
        cache = pickle.load(f)
    all_data = cache[0]

    # 使用与train.py相同的划分逻辑（70/10/20，random_state=42）
    from sklearn.model_selection import train_test_split
    all_indices = np.arange(len(all_data))
    labels_encoded = [item[2] for item in all_data]

    train_indices, temp_indices = train_test_split(
        all_indices, test_size=0.3, random_state=42, stratify=labels_encoded
    )
    temp_labels = [labels_encoded[i] for i in temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.6667, random_state=42, stratify=temp_labels
    )

    test_data = [all_data[i] for i in test_indices]
    print(f"   ✓ 测试集: {len(test_data)} 个样本")

    # 3. 构建测试集
    print(f"\n[3/4] 构建测试集...")
    dataset = TrajectoryDataset(
        test_data,
        traj_mean=traj_mean, traj_std=traj_std,
        weather_mean=weather_mean, weather_std=weather_std,
        stats_mean=stats_mean, stats_std=stats_std
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"   ✓ 最终测试样本: {len(test_data)}")

    # 4. 推理与报告
    print(f"\n[4/4] 推理与报告...")
    class_names_str = [str(name) for name in label_encoder.classes_]
    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for traj, weather, stats, labels in tqdm(loader, desc="Evaluating"):
            traj    = traj.to(DEVICE)
            weather = weather.to(DEVICE)
            stats   = stats.to(DEVICE)
            logits  = model(traj, weather, segment_stats=stats)
            probs   = torch.softmax(logits, dim=1)
            preds   = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    y_probs = np.array(y_probs)

    # 报告
    print("\n" + "=" * 60)
    print("分类报告")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names_str,
                                 zero_division=0, digits=4))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_dict = classification_report(y_true, y_pred, target_names=class_names_str,
                                         output_dict=True, zero_division=0)

    with open(os.path.join(OUTPUT_DIR, 'evaluation_report.json'), 'w', encoding='utf-8-sig') as f:
        json.dump(report_dict, f, indent=4, ensure_ascii=False)
    print(f"   ✓ 保存: evaluation_results/evaluation_report.json")

    conf_list = [float(y_probs[i, p]) for i, p in enumerate(y_pred)]
    pd.DataFrame({
        'true_label': [class_names_str[i] for i in y_true],
        'pred_label': [class_names_str[i] for i in y_pred],
        'confidence': conf_list,
        'correct':    y_true == y_pred
    }).to_csv(os.path.join(OUTPUT_DIR, 'predictions_exp4.csv'),
              index=False, encoding='utf-8-sig')
    print(f"   ✓ 保存: evaluation_results/predictions_exp4.csv")

    try:
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                    xticklabels=class_names_str, yticklabels=class_names_str,
                    annot_kws={'size': 20})
        plt.title('Exp4 Confusion Matrix (LabelSmoothing + Focal Loss)', fontsize=26)
        plt.xlabel('Predicted', fontsize=22); plt.ylabel('True', fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300)
        plt.close()
        print(f"   ✓ 保存: evaluation_results/confusion_matrix.png")
    except Exception as e:
        print(f"   ⚠️ 混淆矩阵生成失败: {e}")

    try:
        f1_scores = [report_dict[cls]['f1-score'] for cls in class_names_str]
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(class_names_str), y=f1_scores, color='darkorange')
        plt.title('Exp4 F1-Score by Transportation Mode', fontsize=14)
        plt.xlabel('Transportation Mode'); plt.ylabel('F1-Score')
        plt.ylim(0, 1.0)
        for i, v in enumerate(f1_scores):
            plt.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=10)
        plt.xticks(rotation=45); plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'per_class_f1_scores.png'), dpi=300)
        plt.close()
        print(f"   ✓ 保存: evaluation_results/per_class_f1_scores.png")
    except Exception as e:
        print(f"   ⚠️ F1图生成失败: {e}")

    errors_df = pd.DataFrame({
        'true_label': [class_names_str[i] for i in y_true],
        'pred_label': [class_names_str[i] for i in y_pred],
        'confidence': conf_list
    })
    errors_df[errors_df['true_label'] != errors_df['pred_label']].to_csv(
        os.path.join(OUTPUT_DIR, 'error_analysis.csv'),
        index=False, encoding='utf-8-sig'
    )
    print(f"   ✓ 保存: evaluation_results/error_analysis.csv")

    print("\n" + "=" * 60)
    print("评估汇总")
    print("=" * 60)
    print(f"总样本数: {len(y_true)}")
    print(f"正确预测: {(y_true == y_pred).sum()}")
    print(f"错误预测: {(y_true != y_pred).sum()}")
    print(f"准确率:   {report_dict['accuracy']:.4f}")
    print(f"加权 F1:  {report_dict['weighted avg']['f1-score']:.4f}")
    print(f"宏平均 F1:{report_dict['macro avg']['f1-score']:.4f}")
    print(f"\n✅ 结果保存至: {os.path.abspath(OUTPUT_DIR)}")


if __name__ == '__main__':
    main()