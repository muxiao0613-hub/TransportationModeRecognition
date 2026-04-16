"""
Exp2 预测器 (独立版)
只需要模型文件，不需要任何额外的 pkl 缓存
"""
import torch
import numpy as np
import os
from src.model import TransportationModeClassifier


class TransportationPredictorExp2:
    def __init__(self, checkpoint_path="checkpoints/exp2_model.pth"):
        """
        初始化实验二预测器
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

        # 初始化模型架构 (Exp2: 轨迹 21 维)
        input_dim = config.get('trajectory_feature_dim', 21)
        self.model = TransportationModeClassifier(
            trajectory_feature_dim=input_dim,
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

        print(f"✅ Exp2 模型加载成功！")
        print(f"📊 特征配置: 轨迹={input_dim}维, 统计特征=18维")
        print(f"🏷️  支持类别: {list(self.class_names)}")

    def predict(self, traj_features, segment_stats=None):
        """
        输入参数:
            traj_features: (seq_len, 21) 的轨迹特征矩阵
            segment_stats: (18,) 的统计特征向量
        返回:
            pred_label: 预测的交通方式字符串
            confidence: 置信度分数
        """
        # 1. 轨迹特征维度处理 (batch_size, seq_len, 21)
        if isinstance(traj_features, list):
            traj_features = np.array(traj_features)
        if traj_features.ndim == 2:
            traj_features = np.expand_dims(traj_features, axis=0)

        # 2. 归一化
        if self.traj_mean is not None:
            traj_features = (traj_features - self.traj_mean) / self.traj_std

        # 3. 统计特征处理
        if segment_stats is not None:
            if isinstance(segment_stats, list):
                segment_stats = np.array(segment_stats)
            if self.stats_mean is not None:
                segment_stats = (segment_stats - self.stats_mean) / self.stats_std
            if segment_stats.ndim == 1:
                segment_stats = np.expand_dims(segment_stats, axis=0)
            stats = torch.FloatTensor(segment_stats).to(self.device)
        else:
            stats = torch.zeros(traj_features.shape[0], 18).to(self.device)

        # 4. 转换为 Tensor 并移动到设备
        traj_tensor = torch.FloatTensor(traj_features).to(self.device)

        # 5. 执行推理
        with torch.no_grad():
            logits = self.model(traj_tensor, segment_stats=stats)
            probs = torch.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)

        # 6. 获取结果
        pred_label = self.label_encoder.inverse_transform(pred_idx.cpu().numpy())[0]
        score = confidence.cpu().item()

        return pred_label, score


# ==========================================
# 示例用法
# ==========================================
if __name__ == "__main__":
    predictor = TransportationPredictorExp2()

    # 模拟输入：
    # 轨迹特征: 长度50, 维度21
    dummy_traj = np.random.randn(50, 21)
    # 统计特征: 维度18
    dummy_stats = np.random.randn(18)

    label, score = predictor.predict(dummy_traj, dummy_stats)

    print("\n" + "=" * 40)
    print(f"【实验二预测结论】")
    print(f"识别模式: {label}")
    print(f"置信水平: {score:.4%}")
    print("=" * 40)
