"""
Exp1 评估脚本
======================
仅使用轨迹特征（9维）+ 段级统计特征（18维）。
从 exp1/cache/exp1_processed_features.pkl 加载缓存特征，
并使用与 train.py 一致的 70/10/20 划分方式生成测试集评估结果。
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
from sklearn.model_selection import train_test_split
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)
os.chdir(SCRIPT_DIR)

from src.model import TransportationModeClassifier

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

EXP1_FEATURE_CACHE = os.path.join(SCRIPT_DIR, 'cache', 'exp1_processed_features.pkl')
OUTPUT_DIR = 'evaluation_results'


class TrajectoryDataset(Dataset):
    """对应 exp1 缓存格式：(traj_9, stats_18, label_encoded)。"""

    def __init__(self, data, traj_mean=None, traj_std=None,
                 stats_mean=None, stats_std=None):
        self.data = data
        self.traj_mean = traj_mean
        self.traj_std = traj_std
        self.stats_mean = stats_mean
        self.stats_std = stats_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        traj_9, stats_18, label_encoded = self.data[idx]

        traj = traj_9.astype(np.float32)
        stats = stats_18.astype(np.float32)

        if self.traj_mean is not None:
            traj = (traj - self.traj_mean) / (self.traj_std + 1e-8)
        if self.stats_mean is not None:
            stats = (stats - self.stats_mean) / (self.stats_std + 1e-8)

        return (
            torch.FloatTensor(traj),
            torch.FloatTensor(stats),
            torch.LongTensor([label_encoded]).squeeze(),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='checkpoints/exp1_model.pth')
    parser.add_argument('--model_path', default=None,
                        help='兼容旧参数名，优先级低于 --checkpoint')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    checkpoint_path = args.checkpoint if args.checkpoint else args.model_path
    if args.model_path is not None:
        checkpoint_path = args.model_path

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print("\n" + "=" * 60)
    print("Exp1 模型评估 (仅轨迹特征)")
    print("=" * 60)
    print(f"设备: {device}")

    # 1. 加载模型
    print(f"\n[1/4] 加载模型: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"找不到模型: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    label_encoder = checkpoint['label_encoder']
    config = checkpoint.get('model_config', {})
    norm_params = checkpoint.get('norm_params', {})

    traj_mean = norm_params.get('traj_mean', None)
    traj_std = norm_params.get('traj_std', None)
    stats_mean = norm_params.get('stats_mean', None)
    stats_std = norm_params.get('stats_std', None)

    if traj_mean is None or stats_mean is None:
        raise ValueError("checkpoint 缺少 norm_params，请重新训练 exp1/train.py")

    model = TransportationModeClassifier(
        trajectory_feature_dim=config.get('trajectory_feature_dim', config.get('input_dim', 9)),
        segment_stats_dim=config.get('segment_stats_dim', 18),
        hidden_dim=config.get('hidden_dim', 128),
        num_layers=config.get('num_layers', 2),
        num_classes=config.get('num_classes', len(label_encoder.classes_)),
        dropout=config.get('dropout', 0.3),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    class_names = [str(x) for x in label_encoder.classes_]
    print(f"   ✓ 模型加载完成 | 类别={class_names}")

    # 2. 加载测试数据
    print("\n[2/4] 加载测试数据...")
    if not os.path.exists(EXP1_FEATURE_CACHE):
        raise FileNotFoundError(
            f"EXP1 特征缓存不存在: {EXP1_FEATURE_CACHE}\n请先运行 exp1/train.py"
        )

    with open(EXP1_FEATURE_CACHE, 'rb') as f:
        cache = pickle.load(f)

    all_data = cache[0] if isinstance(cache, tuple) else cache

    all_indices = np.arange(len(all_data))
    labels_encoded = [item[2] for item in all_data]

    train_indices, temp_indices = train_test_split(
        all_indices, test_size=0.3, random_state=42, stratify=labels_encoded
    )
    temp_labels = [labels_encoded[i] for i in temp_indices]
    _, test_indices = train_test_split(
        temp_indices, test_size=0.6667, random_state=42, stratify=temp_labels
    )

    test_data = [all_data[i] for i in test_indices]
    print(f"   ✓ 测试样本: {len(test_data)}")

    # 3. 推理
    print("\n[3/4] 推理中...")
    dataset = TrajectoryDataset(
        test_data,
        traj_mean=traj_mean,
        traj_std=traj_std,
        stats_mean=stats_mean,
        stats_std=stats_std,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for traj, stats, labels in tqdm(loader, desc='Evaluating'):
            traj = traj.to(device)
            stats = stats.to(device)
            logits = model(traj, segment_stats=stats)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_probs = np.array(y_probs)

    # 4. 生成报告
    print("\n[4/4] 生成报告...")
    print("\n" + "=" * 60)
    print("分类报告")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names,
                                zero_division=0, digits=4))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_dict = classification_report(y_true, y_pred, target_names=class_names,
                                        output_dict=True, zero_division=0)

    with open(os.path.join(OUTPUT_DIR, 'evaluation_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=4, ensure_ascii=False)
    print("   ✓ 保存: evaluation_results/evaluation_report.json")

    conf_list = [float(y_probs[i, p]) for i, p in enumerate(y_pred)]
    pd.DataFrame({
        'true_label': [class_names[i] for i in y_true],
        'pred_label': [class_names[i] for i in y_pred],
        'confidence': conf_list,
        'correct': y_true == y_pred,
    }).to_csv(os.path.join(OUTPUT_DIR, 'predictions_exp1.csv'),
              index=False, encoding='utf-8-sig')
    print("   ✓ 保存: evaluation_results/predictions_exp1.csv")

    try:
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={'size': 20})
        plt.title('Exp1 Confusion Matrix (Trajectory Only)', fontsize=26)
        plt.xlabel('Predicted', fontsize=22)
        plt.ylabel('True', fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300)
        plt.close()
        print("   ✓ 保存: evaluation_results/confusion_matrix.png")
    except Exception as e:
        print(f"   ⚠️ 混淆矩阵生成失败: {e}")

    try:
        f1_scores = [report_dict[cls]['f1-score'] for cls in class_names]
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(class_names), y=f1_scores, color='steelblue')
        plt.title('Exp1 F1-Score by Transportation Mode', fontsize=14)
        plt.xlabel('Transportation Mode')
        plt.ylabel('F1-Score')
        plt.ylim(0, 1.0)
        for i, v in enumerate(f1_scores):
            plt.text(i, v + 0.02, f"{v:.3f}", ha='center', fontsize=10)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'per_class_f1_scores.png'), dpi=300)
        plt.close()
        print("   ✓ 保存: evaluation_results/per_class_f1_scores.png")
    except Exception as e:
        print(f"   ⚠️ F1图生成失败: {e}")

    errors_df = pd.DataFrame({
        'true_label': [class_names[i] for i in y_true],
        'pred_label': [class_names[i] for i in y_pred],
        'confidence': conf_list,
    })
    errors_df[errors_df['true_label'] != errors_df['pred_label']].to_csv(
        os.path.join(OUTPUT_DIR, 'error_analysis.csv'),
        index=False,
        encoding='utf-8-sig',
    )
    print("   ✓ 保存: evaluation_results/error_analysis.csv")

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
