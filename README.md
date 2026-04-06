# 城市出行智能感知平台

基于深度学习的交通方式识别系统，支持多种轨迹数据格式上传和实时预测。

## 项目结构

```
交通方式识别/
├── BackEnd/                 # 后端服务
│   ├── api/                # FastAPI 后端接口
│   │   ├── routers/        # API 路由
│   │   ├── main.py         # FastAPI 主入口
│   │   └── schemas.py      # 数据模型定义
│   ├── common/             # 通用工具
│   ├── exp1/               # 实验1：9维轨迹特征
│   ├── exp2/               # 实验2：9维轨迹特征 + OSM
│   ├── exp3/               # 实验3：9维轨迹特征 + 天气
│   ├── exp4/               # 实验4：9维轨迹特征 + OSM + 天气
│   └── scripts/            # 数据准备脚本
├── FrontEnd/               # 前端应用
│   ├── src/
│   │   ├── views/          # 页面组件
│   │   ├── components/     # 通用组件
│   │   ├── stores/         # Pinia 状态管理
│   │   ├── api/            # API 调用
│   │   └── types/          # TypeScript 类型定义
│   └── package.json
└── data/                   # 数据目录（已在 .gitignore 中）
```

## 功能特性

### 前端页面
- **地图分析**：上传轨迹文件，实时预测交通方式并在地图上可视化
- **模型对比**：对比四个实验的训练结果和性能指标
- **数据概览**：展示数据集统计信息和特征维度说明
- **关于**：项目介绍和技术架构

### 后端功能
- 支持 CSV、JSON、PLT (Geolife) 格式的轨迹文件上传
- 四个实验的模型预测接口
- 实验结果和数据统计 API
- 实时交通方式识别

## 快速开始

### 环境要求
- Python 3.8+
- Node.js 16+
- 推荐使用 conda 管理 Python 环境

### 后端启动

1. 创建并激活 conda 环境：
```bash
conda create -n transportation python=3.11
conda activate transportation
```

2. 安装依赖：
```bash
cd BackEnd
pip install -r requirements.txt
```

3. 启动后端服务：
```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

后端服务将在 http://localhost:8000 启动

### 前端启动

1. 安装依赖：
```bash
cd FrontEnd
npm install
```

2. 启动开发服务器：
```bash
npm run dev
```

前端应用将在 http://localhost:5173 启动

## 使用说明

### 文件上传格式

系统支持三种轨迹文件格式：

**1. CSV 格式**
```csv
latitude,longitude,timestamp
39.9042,116.4074,2024-01-01 08:00:00
39.9050,116.4080,2024-01-01 08:00:05
```

**2. JSON 格式**
```json
[
  {
    "latitude": 39.9042,
    "longitude": 116.4074,
    "timestamp": "2024-01-01 08:00:00"
  }
]
```

**3. PLT 格式 (Geolife)**
```
Geolife trajectory
WGS84
Altitude is in Feet
Reserved
3
1
0
10
39.9042,116.4074,0,100,39245.3333,2024-01-01,08:00:00
```

### 模型选择

在地图分析页面，可以选择以下模型进行预测：
- **exp1**：使用 9 维轨迹特征的深度学习模型（推荐）
- **exp2**：需要 OSM 数据（当前使用规则预测）
- **exp3**：需要天气数据（当前使用规则预测）
- **exp4**：需要 OSM + 天气数据（当前使用规则预测）

## 实验说明

### 四个实验的对比

| 实验 | 特征组合 | 模型结构 | 损失函数 |
|------|----------|----------|----------|
| exp1 | 9维轨迹特征 | LSTM | CrossEntropy |
| exp2 | 9维轨迹特征 + OSM | LSTM | CrossEntropy |
| exp3 | 9维轨迹特征 + 天气 | LSTM + Weather | Focal Loss |
| exp4 | 9维轨迹特征 + OSM + 天气 | LSTM + Weather | LabelSmoothingFocalLoss |

## 技术栈

### 后端
- FastAPI - Web 框架
- PyTorch - 深度学习框架
- Pandas - 数据处理
- NumPy - 数值计算

### 前端
- Vue 3 - 前端框架
- TypeScript - 类型安全
- Vite - 构建工具
- Pinia - 状态管理
- Element Plus - UI 组件库
- ECharts - 数据可视化
- Leaflet - 地图组件

## 开发说明

### 数据目录

`data/` 目录已添加到 `.gitignore`，不会被提交到版本控制。

### 模型文件

各实验的训练好的模型权重保存在：
- `BackEnd/exp1/checkpoints/exp1_model.pth`
- `BackEnd/exp2/checkpoints/exp2_model.pth`
- `BackEnd/exp3/checkpoints/exp3_model.pth`
- `BackEnd/exp4/checkpoints/exp4_model.pth`

## 许可证

本项目仅供学习和研究使用。
