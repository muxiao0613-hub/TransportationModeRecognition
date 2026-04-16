"""
Exp1 预测器 (独立版)
只需要模型文件，不需要任何额外的 pkl 缓存
"""
import torch
import numpy as np
import os
from src.model import TransportationModeClassifier


class TrajectoryPredictor:
    def __init__(self, checkpoint_path="checkpoints/exp1_model.pth"):
        """
        初始化实验一预测器
        :param checkpoint_path: 训练好的模型权重路径（包含 label_encoder）
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 加载模型 Checkpoint（包含 label_encoder）
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"❌ 未找到模型文件: {checkpoint_path}")

        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # 从 checkpoint 中获取 label_encoder
        self.label_encoder = ckpt['label_encoder']
        self.class_names = self.label_encoder.classes_
        config = ckpt['model_config']

        # 初始化模型架构
        self.model = TransportationModeClassifier(
            trajectory_feature_dim=config.get('trajectory_feature_dim', config.get('input_dim', 9)),
            segment_stats_dim=18,
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            num_classes=config['num_classes'],
            dropout=config.get('dropout', 0.3)
        ).to(self.device)

        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()

        norm_params = ckpt.get('norm_params', {})
        self.traj_mean = norm_params.get('traj_mean', None)
        self.traj_std = norm_params.get('traj_std', None)
        self.stats_mean = norm_params.get('stats_mean', None)
        self.stats_std = norm_params.get('stats_std', None)

        print(f"✅ Exp1 模型加载成功！")
        print(f"📊 特征配置: 轨迹维度={config.get('trajectory_feature_dim', config.get('input_dim', 9))}, 统计特征=18")
        print(f"🏷️  支持类别: {list(self.class_names)}")

    def predict(self, trajectory_features, segment_stats=None):
        """
        输入:
            trajectory_features: np.ndarray, 形状为 (seq_len, 9) 或 (batch, seq_len, 9)
            segment_stats: np.ndarray, 形状为 (18,) 或 (batch, 18)，可选
        输出:
            pred_labels: 预测标签字符串数组
            confidences: 置信度分数数组
        """
        # 转换数据类型
        if isinstance(trajectory_features, list):
            trajectory_features = np.array(trajectory_features)

        # 归一化
        if self.traj_mean is not None:
            trajectory_features = (trajectory_features - self.traj_mean) / self.traj_std

        # 如果是单条数据 (seq_len, 9)，增加 batch 维度 -> (1, seq_len, 9)
        if len(trajectory_features.shape) == 2:
            trajectory_features = np.expand_dims(trajectory_features, axis=0)

        x = torch.FloatTensor(trajectory_features).to(self.device)

        # 处理 segment_stats
        if segment_stats is not None:
            if isinstance(segment_stats, list):
                segment_stats = np.array(segment_stats)
            if self.stats_mean is not None:
                segment_stats = (segment_stats - self.stats_mean) / self.stats_std
            if len(segment_stats.shape) == 1:
                segment_stats = np.expand_dims(segment_stats, axis=0)
            stats = torch.FloatTensor(segment_stats).to(self.device)
            if stats.dim() == 1:
                stats = stats.unsqueeze(0)
        else:
            # 无 segment_stats 时用零向量（向后兼容）
            stats = torch.zeros(x.size(0), 18).to(self.device)

        with torch.no_grad():
            logits = self.model(x, segment_stats=stats)
            probs = torch.softmax(logits, dim=1)
            conf, preds = torch.max(probs, dim=1)

        # 映射回标签名称
        pred_labels = self.label_encoder.inverse_transform(preds.cpu().numpy())
        confidences = conf.cpu().numpy()

        return pred_labels, confidences


# ==========================================
# 示例用法
# ==========================================
if __name__ == "__main__":
    predictor = TrajectoryPredictor()

    # 模拟输入一条 9 维特征的轨迹段 (假设长度为 50)
    dummy_input = np.random.randn(50, 9)

    labels, scores = predictor.predict(dummy_input)

    print("\n" + "=" * 40)
    print(f"【实验一预测结论】")
    print(f"识别模式: {labels[0]}")
    print(f"置信水平: {scores[0]:.4%}")
    print("=" * 40)