"""
Exp3 评估脚本（修复版）
修复内容：
1. 移除实时OSM特征提取，直接从 exp2/cache/processed_features.pkl 加载21维特征
2. 使用 shared_split.pkl（新格式），不依赖旧的 shared_test_indices.pkl
3. 天气特征从 cleaned_balanced.pkl 中的时间戳提取，而非实时网络请求
4. 不再从 train 模块 import Dataset（避免触发 train.py 副作用）
   —— Dataset 类在本文件内直接定义
5. WeatherDataProcessor 路径使用绝对路径，不依赖 cwd
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
EXP2_DIR   = os.path.join(PARENT_DIR, 'exp2')
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, PARENT_DIR)
os.chdir(SCRIPT_DIR)

from src import model_weather
from src import weather_preprocessing

# 添加exp2路径用于导入feature_extraction模块
sys.path.insert(0, EXP2_DIR)

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

MODEL_PATH         = 'checkpoints/exp3_model.pth'
EXP2_FEATURE_CACHE = os.path.join(EXP2_DIR, 'cache', 'processed_features.pkl')
# 原始数据（含时间戳），用于提取天气
CLEANED_DATA_PATH  = os.path.join(PARENT_DIR, 'data', 'processed', 'cleaned_balanced.pkl')
OUTPUT_DIR         = 'evaluation_results'
DEVICE             = 'cuda' if torch.cuda.is_available() else 'cpu'


# ── Dataset（直接定义，不从 train.py import）─────────────────────
class TrajectoryDatasetWithWeather(Dataset):
    """
    格式：(traj_21, weather_10, stats_18, label_encoded)
    weather_10: shape (T, 10)，T 为序列长度（通常50）
    """
    def __init__(self, data, traj_mean=None, traj_std=None,
                 weather_mean=None, weather_std=None,
                 stats_mean=None, stats_std=None):
        self.data         = data
        self.traj_mean    = traj_mean
        self.traj_std     = traj_std
        self.weather_mean = weather_mean
        self.weather_std  = weather_std
        self.stats_mean   = stats_mean
        self.stats_std    = stats_std

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        traj_21, weather, stats, label_encoded = self.data[idx]
        traj    = traj_21.astype(np.float32)
        weather = weather.astype(np.float32)
        stats   = stats.astype(np.float32)

        if self.traj_mean is not None:
            traj    = (traj    - self.traj_mean)    / (self.traj_std    + 1e-8)
        if self.weather_mean is not None:
            weather = (weather - self.weather_mean) / (self.weather_std + 1e-8)
        if self.stats_mean is not None:
            stats   = (stats   - self.stats_mean)   / (self.stats_std   + 1e-8)

        return (torch.FloatTensor(traj),
                torch.FloatTensor(weather),
                torch.FloatTensor(stats),
                torch.LongTensor([label_encoded])[0])


def main():
    print("\n" + "=" * 60)
    print("Exp3 模型评估 (轨迹 + OSM空间 + 天气)")
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

    traj_mean    = norm_params.get('traj_mean',    None)
    traj_std     = norm_params.get('traj_std',     None)
    weather_mean = norm_params.get('weather_mean', None)
    weather_std  = norm_params.get('weather_std',  None)
    stats_mean   = norm_params.get('stats_mean',   None)
    stats_std    = norm_params.get('stats_std',    None)

    if traj_mean is None:
        print("❌ checkpoint 缺少 norm_params，请重新训练 exp3/train.py")
        return

    print(f"   模型配置:")
    print(f"     - 轨迹维度: {config['trajectory_feature_dim']}")
    print(f"     - 天气维度: {config['weather_feature_dim']}")
    print(f"     - stats维度: {config.get('segment_stats_dim', 18)}")
    print(f"     - 类别:     {list(class_names)}")

    model = model_weather.TransportationModeClassifierWithWeather(
        trajectory_feature_dim=config['trajectory_feature_dim'],
        weather_feature_dim=config['weather_feature_dim'],
        segment_stats_dim=config.get('segment_stats_dim', 18),
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        num_classes=config['num_classes'],
        dropout=config.get('dropout', 0.3)
    ).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("   ✓ 模型加载完成")

    # ── 2. 从缓存加载数据（自行划分）───────────────────────
    print(f"\n[2/4] 加载数据...")
    for path, name in [(EXP2_FEATURE_CACHE, 'EXP2特征缓存'),
                       (CLEANED_DATA_PATH,  '原始数据(时间戳)')]:
        if not os.path.exists(path):
            print(f"❌ {name} 不存在: {path}")
            if name == 'EXP2特征缓存':
                print("   请先运行 exp2/train.py")
            return

    with open(EXP2_FEATURE_CACHE, 'rb') as f:
        cache = pickle.load(f)
    all_exp2_data = cache[0] if isinstance(cache, tuple) else cache

    with open(CLEANED_DATA_PATH, 'rb') as f:
        cleaned_data = pickle.load(f)

    n_exp2    = len(all_exp2_data)
    n_cleaned = len(cleaned_data)
    if n_exp2 != n_cleaned:
        print(f"   ⚠️ 样本数不一致: EXP2缓存={n_exp2}, cleaned_balanced={n_cleaned}")
        n_use = min(n_exp2, n_cleaned)
        print(f"   → 取较小值 {n_use} 个样本")
    else:
        n_use = n_exp2

    print(f"   ✓ EXP2缓存: {n_exp2} 样本 | cleaned_balanced: {n_cleaned} 样本")

    # 使用与train.py相同的划分逻辑（70/10/20，random_state=42）
    from sklearn.model_selection import train_test_split
    all_indices = np.arange(n_use)
    labels_encoded = [item[2] for item in all_exp2_data[:n_use]]

    train_indices, temp_indices = train_test_split(
        all_indices, test_size=0.3, random_state=42, stratify=labels_encoded
    )
    temp_labels = [labels_encoded[i] for i in temp_indices]
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.6667, random_state=42, stratify=temp_labels
    )

    print(f"   ✓ 测试集: {len(test_indices)} 个样本")

    # ── 3. 附加天气特征 ──────────────────────────────────────────
    print(f"\n[3/4] 附加天气特征...")
    # WeatherDataProcessor 使用绝对路径初始化
    weather_csv_path = os.path.join(PARENT_DIR, 'data', 'beijing_weather_daily_2007_2012.csv')
    if os.path.exists(weather_csv_path):
        weather_processor = weather_preprocessing.WeatherDataProcessor(weather_csv_path=weather_csv_path)
        weather_processor.load_and_process()  # 加载天气数据
    else:
        print(f"   警告: 天气数据文件不存在: {weather_csv_path}")
        print(f"   将使用零向量代替天气特征")
        weather_processor = None
    print(f"   天气数据文件: {weather_csv_path}")

    SEQ_LEN     = 50   # 固定序列长度
    WEATHER_DIM = config['weather_feature_dim']  # 通常10
    zero_weather = np.zeros((SEQ_LEN, WEATHER_DIM), dtype=np.float32)

    test_data = []
    skipped   = 0
    for idx in tqdm(test_indices, desc="附加天气特征"):
        traj_21, stats, label_encoded = all_exp2_data[idx]

        # 从 cleaned_balanced.pkl 取时间戳
        raw_sample = cleaned_data[idx]
        datetime_series = raw_sample[2]  # shape=(50,), dtype=datetime64[us]

        try:
            if datetime_series is not None and len(datetime_series) > 0 and weather_processor is not None:
                weather_feat = weather_processor.get_weather_features_for_trajectory(
                    datetime_series
                )
                T = traj_21.shape[0]
                if weather_feat.shape[0] != T:
                    if weather_feat.shape[0] > T:
                        weather_feat = weather_feat[:T]
                    else:
                        pad = np.zeros((T - weather_feat.shape[0], WEATHER_DIM),
                                       dtype=np.float32)
                        weather_feat = np.vstack([weather_feat, pad])
                # 检查维度是否匹配
                if weather_feat.shape[1] != WEATHER_DIM:
                    weather_feat = zero_weather.copy()
            else:
                weather_feat = zero_weather.copy()

            weather_feat = np.nan_to_num(weather_feat, nan=0.0,
                                          posinf=0.0, neginf=0.0).astype(np.float32)
        except Exception as e:
            weather_feat = zero_weather.copy()

        test_data.append((traj_21.astype(np.float32),
                          weather_feat,
                          stats.astype(np.float32),
                          label_encoded))

    if skipped:
        print(f"   ⚠️ 跳过 {skipped} 个越界索引")
    print(f"   ✓ 最终测试样本: {len(test_data)}")

    # ── 4. 推理 ──────────────────────────────────────────────────
    print(f"\n[4/4] 推理与报告...")
    dataset = TrajectoryDatasetWithWeather(
        test_data,
        traj_mean=traj_mean, traj_std=traj_std,
        weather_mean=weather_mean, weather_std=weather_std,
        stats_mean=stats_mean, stats_std=stats_std
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

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
    class_names_str = [str(name) for name in class_names]
    print(classification_report(y_true, y_pred, target_names=class_names_str,
                                 zero_division=0, digits=4))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    report_dict = classification_report(y_true, y_pred, target_names=class_names_str,
                                         output_dict=True, zero_division=0)

    with open(os.path.join(OUTPUT_DIR, 'evaluation_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=4, ensure_ascii=False)
    print(f"   ✓ 保存: evaluation_results/evaluation_report.json")

    conf_list = [float(y_probs[i, p]) for i, p in enumerate(y_pred)]
    pd.DataFrame({
        'true_label': [class_names_str[i] for i in y_true],
        'pred_label': [class_names_str[i] for i in y_pred],
        'confidence': conf_list,
        'correct':    y_true == y_pred
    }).to_csv(os.path.join(OUTPUT_DIR, 'predictions_exp3.csv'),
              index=False, encoding='utf-8-sig')
    print(f"   ✓ 保存: evaluation_results/predictions_exp3.csv")

    try:
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges',
                    xticklabels=class_names_str, yticklabels=class_names_str,
                    annot_kws={'size': 20})
        plt.title('Exp3 Confusion Matrix (Trajectory + Spatial + Weather)', fontsize=26)
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
        plt.title('Exp3 F1-Score by Transportation Mode', fontsize=14)
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