"""
Exp2 评估脚本（修复版）
修复内容：
1. 移除实时OSM特征提取，直接从 exp2/cache/processed_features.pkl 加载
2. 使用 shared_split.pkl（新格式，非旧的 shared_test_indices.pkl）
3. 模型调用改为 model(traj, segment_stats=stats)，不传 spatial 参数
4. Dataset 格式对应缓存：(traj_21, spatial_placeholder, stats_18, label_encoded)
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)
os.chdir(SCRIPT_DIR)

from src.model import TransportationModeClassifier

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

MODEL_PATH         = 'checkpoints/exp2_model.pth'
EXP2_FEATURE_CACHE = os.path.join(SCRIPT_DIR, 'cache', 'processed_features.pkl')
OUTPUT_DIR         = 'evaluation_results'
DEVICE             = 'cuda' if torch.cuda.is_available() else 'cpu'


class TrajectoryDataset(Dataset):
    """
    对应 EXP2 缓存格式：(traj_21, spatial_placeholder, stats_18, label_encoded)
    模型只需要 traj_21 和 stats_18。
    """
    def __init__(self, data, traj_mean=None, traj_std=None,
                 stats_mean=None, stats_std=None):
        self.data       = data
        self.traj_mean  = traj_mean
        self.traj_std   = traj_std
        self.stats_mean = stats_mean
        self.stats_std  = stats_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        traj_21, stats, label_encoded = self.data[idx]
        traj  = traj_21.astype(np.float32)
        stats = stats.astype(np.float32)
        if self.traj_mean is not None:
            traj  = (traj  - self.traj_mean)  / (self.traj_std  + 1e-8)
        if self.stats_mean is not None:
            stats = (stats - self.stats_mean) / (self.stats_std + 1e-8)
        return (torch.FloatTensor(traj),
                torch.FloatTensor(stats),
                torch.LongTensor([label_encoded])[0])


def main():
    print("\n" + "=" * 60)
    print("Exp2 模型评估 (点级融合：轨迹 + OSM空间特征)")
    print("=" * 60)
    print(f"设备: {DEVICE}")

    # ── 1. 加载模型 ─────────────────────────────────────────────
    print(f"\n[1/4] 加载模型: {MODEL_PATH}")
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型: {MODEL_PATH}")
        return

    checkpoint    = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    label_encoder = checkpoint['label_encoder']
    config        = checkpoint['model_config']
    norm_params   = checkpoint.get('norm_params', {})
    class_names   = label_encoder.classes_

    traj_mean  = norm_params.get('traj_mean',  None)
    traj_std   = norm_params.get('traj_std',   None)
    stats_mean = norm_params.get('stats_mean', None)
    stats_std  = norm_params.get('stats_std',  None)

    if traj_mean is None:
        print("❌ checkpoint 缺少 norm_params，请重新训练 exp2/train.py")
        return

    input_dim = config.get('input_dim', config.get('trajectory_feature_dim', 21))

    model = TransportationModeClassifier(
        trajectory_feature_dim=input_dim,
        segment_stats_dim=config.get('segment_stats_dim', 18),
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes'],
        dropout=config.get('dropout', 0.3)
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"   ✓ 加载完成 | 输入维度={input_dim} | 类别={list(class_names)}")

    # ── 2. 从缓存加载测试数据（自行划分）───────────────────────
    print(f"\n[2/4] 加载测试数据...")
    if not os.path.exists(EXP2_FEATURE_CACHE):
        print(f"❌ EXP2 特征缓存不存在: {EXP2_FEATURE_CACHE}")
        print("   请先运行 exp2/train.py")
        return

    with open(EXP2_FEATURE_CACHE, 'rb') as f:
        cache = pickle.load(f)
    # 兼容两种缓存格式：(data, mu, sigma) 或 仅 data
    all_data = cache[0] if isinstance(cache, tuple) else cache

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
    print(f"   ✓ 测试样本: {len(test_data)}")

    # ── 3. 推理 ──────────────────────────────────────────────────
    print(f"\n[3/4] 推理中...")
    dataset = TrajectoryDataset(
        test_data,
        traj_mean=traj_mean, traj_std=traj_std,
        stats_mean=stats_mean, stats_std=stats_std
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    y_true, y_pred, y_probs = [], [], []
    with torch.no_grad():
        for traj, stats, labels in tqdm(loader, desc="Evaluating"):
            traj   = traj.to(DEVICE)
            stats  = stats.to(DEVICE)
            # ★ 单编码器，不传 spatial 参数
            logits = model(traj, segment_stats=stats)
            probs  = torch.softmax(logits, dim=1)
            preds  = torch.argmax(logits, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    y_true  = np.array(y_true)
    y_pred  = np.array(y_pred)
    y_probs = np.array(y_probs)

    # ── 4. 生成报告 ───────────────────────────────────────────────
    print(f"\n[4/4] 生成报告...")
    print("\n" + "=" * 60)
    print("分类报告")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names,
                                 zero_division=0, digits=4))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_dict = classification_report(y_true, y_pred, target_names=class_names,
                                         output_dict=True, zero_division=0)

    # JSON
    with open(os.path.join(OUTPUT_DIR, 'evaluation_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=4, ensure_ascii=False)
    print(f"   ✓ 保存: evaluation_results/evaluation_report.json")

    # CSV 预测
    conf_list = [float(y_probs[i, p]) for i, p in enumerate(y_pred)]
    pd.DataFrame({
        'true_label': [class_names[i] for i in y_true],
        'pred_label': [class_names[i] for i in y_pred],
        'confidence': conf_list,
        'correct':    y_true == y_pred
    }).to_csv(os.path.join(OUTPUT_DIR, 'predictions_exp2.csv'),
              index=False, encoding='utf-8-sig')
    print(f"   ✓ 保存: evaluation_results/predictions_exp2.csv")

    # 混淆矩阵
    try:
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                    xticklabels=class_names, yticklabels=class_names,
                    annot_kws={'size': 20})
        plt.title('Exp2 Confusion Matrix (Trajectory + OSM Spatial Features)', fontsize=26)
        plt.xlabel('Predicted', fontsize=22); plt.ylabel('True', fontsize=22)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300)
        plt.close()
        print(f"   ✓ 保存: evaluation_results/confusion_matrix.png")
    except Exception as e:
        print(f"   ⚠️ 混淆矩阵生成失败: {e}")

    # F1图
    try:
        f1_scores = [report_dict[cls]['f1-score'] for cls in class_names]
        plt.figure(figsize=(12, 6))
        sns.barplot(x=list(class_names), y=f1_scores, color='mediumslateblue')
        plt.title('Exp2 F1-Score by Transportation Mode', fontsize=14)
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

    # 错误分析
    errors_df = pd.DataFrame({
        'true_label': [class_names[i] for i in y_true],
        'pred_label': [class_names[i] for i in y_pred],
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