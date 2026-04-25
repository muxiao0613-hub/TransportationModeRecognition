# 交通方式识别系统 (Transportation Mode Recognition)

一个基于轨迹数据、地理空间特征和气象信息的**多模态深度学习系统**，用于自动识别用户的交通方式（步行、骑车、公交、汽车/出租车、火车、地铁、飞机）。

## 📋 项目概述

### 核心特性
- **多模态融合**：结合轨迹特征、OSM地理空间特征和天气数据
- **递进式实验设计**：从简单到复杂，系统地评估各模态贡献
- **轨迹分段处理**：将长轨迹分解为固定长度段，提取统计特征
- **Bi-LSTM架构**：双向长短期记忆网络处理时间序列数据
- **类别均衡处理**：支持class weights和Focal Loss处理不平衡数据
- **灵活的特征缓存**：加速迭代训练过程

### 数据集
- **GeoLife Dataset**：微软亚洲研究院发布的真实用户GPS轨迹数据
- **时间跨度**：2007-2012年
- **地理范围**：北京市及其他城市
- **目标分类**：7种交通方式

---

## 📁 项目结构

```
TransportationModeRecognition/
├── config.yaml                    # 全局配置文件（模型参数、数据路径）
├── requirements.txt               # 依赖包列表
├── README.md                      # 本文件
│
├── common/                        # ⭐ 共享基础库（所有实验通用）
│   ├── __init__.py
│   ├── base_model.py             # 基础分类器和层级模型
│   ├── base_preprocessor.py      # GeoLife数据预处理（所有实验通用）
│   ├── base_adapter.py           # 数据适配器
│   ├── adapters.py               # 各实验的数据适配器实现
│   ├── train_utils.py            # 通用训练函数（train_epoch, evaluate）
│   ├── trajectory_cleaner.py     # 轨迹数据清洗
│   ├── focal_loss.py             # Focal Loss实现
│   └── base_preprocessor.py      # 基础预处理器
│
├── exp1/                         # 📊 实验1：纯轨迹特征
│   ├── train.py                 # 训练脚本
│   ├── evaluate.py              # 评估脚本
│   ├── predict.py               # 预测脚本
│   └── src/
│       ├── data_loader.py       # 数据加载器
│       └── model.py             # 分类模型
│
├── exp2/                         # 📊 实验2：轨迹 + OSM空间特征
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── src/
│       ├── data_preprocessing.py
│       ├── feature_extraction.py # 特征提取
│       ├── osm_feature_extractor.py  # OSM空间特征提取
│       └── model.py
│
├── exp3/                         # 📊 实验3：轨迹 + 增强空间特征 + 天气
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── src/
│       ├── model_weather.py     # 支持天气特征的模型
│       └── weather_preprocessing.py   # 天气数据处理
│
├── exp4/                         # 📊 实验4：使用Focal Loss改进
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── src/
│
└── scripts/                      # 🛠️ 数据处理脚本
    ├── generate_base_data.py     # 一键生成基础数据
    ├── prepare_data.py           # 数据准备和清洗
    ├── download_osm.py           # OSM数据下载
    └── download_weather.py       # 天气数据下载
```

---

## 🔄 实验设计对比

| 维度 | Exp1 | Exp2 | Exp3 | Exp4 |
|------|------|------|------|------|
| **轨迹特征** | ✅ 9维 | ✅ 9维 | ✅ 9维 | ✅ 9维 |
| **OSM空间特征** | ❌ | ✅ 12维 | ✅ 15维(增强) | ✅ 15维 |
| **天气特征** | ❌ | ❌ | ✅ 10维 | ✅ 10维 |
| **损失函数** | Cross Entropy | CE | CE | Focal Loss |
| **特征维度** | 27 | 48 | 58 | 58 |
| **关键改进** | 基础版本 | 加入地理信息 | 引入气象数据 | 处理类别不平衡 |

---

## 🎯 特征详解

### 1. 轨迹特征 (9维)
GPS轨迹点级特征，计算方式：
```
[0] latitude           - 纬度
[1] longitude          - 经度
[2] speed             - 速度 (m/s)
[3] acceleration      - 加速度 (m/s²)
[4] bearing_change    - 方向变化率 (°/s)
[5] distance          - 相邻点距离 (m)
[6] time_diff         - 相邻点时间差 (s)
[7] total_distance    - 轨迹总距离 (m) [滑动窗口]
[8] total_time        - 轨迹总时间 (s)  [滑动窗口]
```

### 2. 段级统计特征 (18维)
对固定长度轨迹段（50个点）的统计：
```
- 轨迹特征的均值、方差、最大值、最小值 (9×2=18维)
```

### 3. OSM空间特征 (12~15维)
基于OpenStreetMap的本地地理信息：
```
[0-3]  道路类型特征 - 主干道/次干道/小街/步行街覆盖率
[4-5]  建筑物信息 - POI密度/建筑频率  
[6-9]  交通设施 - 公交站/停车场/地铁站/最近POI距离
[10]   道路密度  - 100m范围内道路节点数
[11-14] 增强特征 - 街道连通度等(Exp3+)
```

### 4. 天气特征 (10维，Exp3+)
```
[0]  temp          - 平均气温 (°C)
[1]  tmin          - 最低气温 (°C)
[2]  tmax          - 最高气温 (°C)
[3]  prcp          - 降水量 (mm)
[4]  wspd          - 风速 (km/h)
[5]  is_rainy      - 是否降雨 (prcp > 0.5mm)
[6]  is_heavy_rain - 是否大雨 (prcp > 10mm)
[7]  is_snowy      - 是否降雪 
[8]  is_cold       - 是否寒冷 (temp < 0°C)
[9]  is_hot        - 是否炎热 (temp > 30°C)
```

---

## 🚀 快速开始

### 环境配置

```bash
# 1. 创建虚拟环境（可选）
python -m venv venv
source venv/Scripts/activate  # Windows: venv\Scripts\activate

# 2. 安装依赖
pip install -r requirements.txt
```

### 依赖包
- **深度学习**：torch>=2.0.0, numpy>=1.24.0
- **数据处理**：pandas>=2.0.0, scikit-learn>=1.3.0
- **地理计算**：geopy>=2.3.0, networkx>=3.0, scipy>=1.10.0
- **天气数据**：meteostat>=1.6.0
- **其他**：matplotlib, seaborn, tqdm, pyyaml

### 数据准备

```bash
# 1. 下载原始数据
#    - GeoLife数据集：放到 data/Geolife Trajectories 1.3/
#    - Beijing OSM数据：data/beijing_osm_full_enhanced_verified.geojson
#    - 天气数据：data/beijing_weather_daily_2007_2012.csv

# 2. 一键生成基础数据（所有实验通用）
python scripts/generate_base_data.py

# 3. 数据清洗和准备（交互模式）
python scripts/prepare_data.py

#    或指定模式
python scripts/prepare_data.py --mode balanced
```

**数据处理流程**：
```
原始GeoLife数据
    ↓
BaseGeoLifePreprocessor (base_segments.pkl)
    ↓ 轨迹清洗 + 特征提取
TrajectoryCleaner + 特征工程
    ↓ 数据适配
各实验的DataAdapter
    ↓ 类别均衡
cleaned_balanced.pkl
    ↓ 特征编码
各实验特定的feature缓存
```

### 训练实验

```bash
# Exp1: 纯轨迹特征
cd exp1
python train.py --epochs 100 --batch_size 32 --lr 0.001

# Exp2: 轨迹 + 空间特征
cd ../exp2
python train.py --epochs 100 --batch_size 32 --lr 0.001 --use_spatial

# Exp3: 轨迹 + 空间 + 天气特征  
cd ../exp3
python train.py --epochs 100 --batch_size 32 --lr 0.001 --use_weather

# Exp4: 使用Focal Loss版本
cd ../exp4
python train.py --epochs 100 --batch_size 32 --lr 0.001 --focal_loss
```

### 评估和预测

```bash
# 评估模型
cd exp1  # 或其他实验目录
python evaluate.py --model_path checkpoints/exp1_model.pth

# 单条轨迹预测
python predict.py --trajectory_path path/to/trajectory.csv
```

---

## 🧠 核心模型架构

### BaseTransportationClassifier
```python
class BaseTransportationClassifier(nn.Module):
    """
    基础分类器：支持多输入模态的融合框架
    
    流程：
    1. 每个模态独立Bi-LSTM编码
    2. 截取最后时间步的隐层状态
    3. 拼接所有模态的表示
    4. 通过融合层和分类层输出
    """
    
    input_dims  : List[int]   # 各模态维度，如[9, 12]
    hidden_dims : List[int]   # LSTM隐层维度
    num_layers  : int = 2
    num_classes : int = 7
```

### HierarchicalTransportationClassifier
```python
class HierarchicalTransportationClassifier(nn.Module):
    """
    层级分类器：额外支持轨迹分段处理
    
    特点：
    - 处理固定长度轨迹段（固定为50个点）
    - 同时输入：轨迹特征序列 + 段级统计特征
    - 融合两类特征后做预测
    - 更好地捕捉全局统计特性
    """
```

---

## 📊 训练流程

### 数据流向
```
Data
 ├─ TrajectoryDataset
 │   ├─ 遍历segment索引
 │   └─ 返回(features_tuple, label)
 │
 └─ DataLoader
     └─ 生成batch

Model
 ├─ Bi-LSTM编码器（每个模态一个）
 │   └─ 输出sequence的最后隐层[B, hidden_dim]
 │
 ├─ 拼接所有模态表示
 │   └─ [B, sum(hidden_dims)]
 │
 ├─ 融合层（可选）
 │   └─ [B, fusion_dim]
 │
 └─ 分类头
     └─ logits [B, num_classes]

Loss & Optimization
 ├─ Cross Entropy Loss 或 Focal Loss
 ├─ Backward pass
 ├─ Gradient clipping (max_norm=1.0)
 └─ Parameter update
```

### 关键函数

#### train_epoch()
```python
def train_epoch(model, dataloader, criterion, optimizer, device,
                max_grad_norm=1.0, use_cca=False, context_loss_weight=0.1):
    """
    训练单个epoch
    - 支持任意数量隐层状态输入
    - 梯度裁剪防止爆炸
    - 可选的CCA对比损失
    """
```

#### evaluate()
```python
def evaluate(model, dataloader, criterion, device, num_classes=7):
    """
    评估模型性能
    - 计算loss和accuracy
    - 返回分类报告
    - 支持多分类评估
    """
```

---

## ⚙️ 配置文件说明

### config.yaml

```yaml
# 数据路径
data:
  geolife_root: "data/Geolife Trajectories 1.3"
  osm_path: "data/beijing_osm_full_enhanced_verified.geojson"
  weather_data_path: "data/weather_data"

# 模型参数
model:
  hidden_dim: 128          # LSTM隐层维度
  num_layers: 2            # LSTM层数
  num_classes: 7           # 分类类别数
  dropout: 0.3             # Dropout比率
  num_segments: 5          # 轨迹分割数

# 各实验配置
exp1:
  trajectory_feature_dim: 9
  segment_stats_dim: 18

exp2:
  trajectory_feature_dim: 9
  spatial_feature_dim: 12  # Exp2中是12维，Exp3中是15维
  segment_stats_dim: 18

exp3:
  trajectory_feature_dim: 9
  spatial_feature_dim: 15
  weather_feature_dim: 10
  segment_stats_dim: 18

# 训练参数
training:
  batch_size: 32
  learning_rate: 0.001
  num_epochs: 100
  early_stopping_patience: 20
  device: "cuda"  # 或 "cpu"
```

---

## 🔬 技术亮点

### 1. 轨迹分段处理
- 将长轨迹分解为固定长度段（50个点）
- 提取段级统计特征，增强模型的全局理解
- 避免过长序列导致的梯度问题

### 2. 多模态融合
- **早期融合**：特征级拼接
- **晚期融合**：编码器分离，隐层拼接
- 支持不同模态的独立处理

### 3. 类别不平衡处理
- **Class Weights**：根据样本数计算权重
- **Focal Loss**：重点关注困难样本
- **数据均衡**：采样平衡数据集

### 4. 缓存机制
- 空间特征缓存：避免重复计算OSM查询
- 处理后特征缓存：加速迭代训练
- 网格索引缓存：提高查询速度

### 5. 灵活的特征工程
- 动态特征维度管理
- 支持特征归一化
- 易于扩展新特征

---

## 📈 预期结果

基于GeoLife数据集的实验结果（参考）：
- **Exp1** (纯轨迹)：Accuracy ~75-80%
- **Exp2** (轨迹+OSM)：Accuracy ~80-85%
- **Exp3** (轨迹+OSM+天气)：Accuracy ~82-87%
- **Exp4** (Focal Loss)：Accuracy ~83-88%

注：实际结果取决于数据集大小、模型超参数等因素

---

## 🛠️ 常见问题

### Q1: 如何使用自己的轨迹数据？
```python
# 准备CSV文件，包含列：latitude, longitude, timestamp, label(可选)
# 修改BaseGeoLifePreprocessor的数据读取逻辑
# 或创建新的Adapter继承BaseDataAdapter
```

### Q2: 如何调整模型大小？
```yaml
# 在config.yaml中修改：
model:
  hidden_dim: 256        # 增大LSTM隐层
  num_layers: 3          # 增加LSTM层数
  dropout: 0.5           # 增加正则化
```

### Q3: 如何使用GPU加速训练？
```bash
# 确保已安装CUDA版本的PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 在代码中会自动检测GPU：device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### Q4: 内存不足怎么办？
```python
# 减小batch_size
training:
  batch_size: 16  # 从32改为16

# 减少模型参数
model:
  hidden_dim: 64  # 从128改为64
```

---

## 📚 文件说明

### common/ 核心库

| 文件 | 说明 |
|------|------|
| `base_model.py` | BaseTransportationClassifier、HierarchicalTransportationClassifier |
| `base_preprocessor.py` | GeoLife数据加载和预处理 |
| `trajectory_cleaner.py` | 轨迹清洗（去重、异常值处理） |
| `base_adapter.py` | 数据适配器基类 |
| `adapters.py` | 各实验的具体DataAdapter实现 |
| `train_utils.py` | train_epoch()、evaluate()等通用函数 |
| `focal_loss.py` | Focal Loss和LabelSmoothing实现 |

### scripts/ 工具脚本

| 脚本 | 说明 |
|------|------|
| `generate_base_data.py` | 一键生成所有实验通用的基础数据 |
| `prepare_data.py` | 数据清洗和类别均衡 |
| `download_osm.py` | 下载OpenStreetMap数据 |
| `download_weather.py` | 下载北京区域天气数据 |

---

## 🔗 依赖库详解

- **PyTorch**：深度学习框架，用于模型定义和训练
- **pandas/numpy**：数据操作和数值计算
- **scikit-learn**：数据处理和评估指标
- **geopy**：地理距离计算
- **networkx**：OSM图处理  
- **scipy KDTree**：空间索引加速
- **meteostat**：天气数据API

---

## 📝 许可证和引用

如果使用此项目中的代码或想法，请引用：

```bibtex
@misc{transportationmoderecognition,
  title={Transportation Mode Recognition System},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo}
}
```

数据集引用：
```bibtex
@inproceedings{zheng2008understanding,
  title={Understanding mobility based on GPS data},
  author={Yu, Z. and others},
  booktitle={UbiComp 2008},
  year={2008}
}
```

---

## 💬 联系方式

有问题或建议？
- 提交Issue或Pull Request
- 发邮件至 your-email@example.com

---

**最后更新**：2024年3月18日
