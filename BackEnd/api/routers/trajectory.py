from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import io
import sys
import os
from pathlib import Path
from typing import List, Tuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from api.schemas import (
    TrajectoryPrediction,
    TrajectoryPoint,
    TrajectoryStats,
    TransportMode,
    SegmentPrediction,  # 新增
    ModeBreakdown,       # 新增
)

router = APIRouter()

TRANSPORT_MODES = [
    {"id": "walk", "name": "步行", "color": "#4A90E2", "icon": "walking"},
    {"id": "bike", "name": "自行车", "color": "#52C41A", "icon": "bicycle"},
    {"id": "bus", "name": "公交", "color": "#FA8C16", "icon": "bus"},
    {"id": "car", "name": "汽车/出租", "color": "#F5222D", "icon": "car"},
    {"id": "subway", "name": "地铁", "color": "#722ED1", "icon": "subway"},
    {"id": "train", "name": "火车", "color": "#13C2C2", "icon": "train"},
]

predictors = {}
weather_processor = None
osm_extractor = None

# ========== 原有的加载函数保持不变 ==========

def load_osm_data():
    """加载真实OSM数据（优先使用缓存）"""
    global osm_extractor
    try:
        import pickle
        from pathlib import Path
        
        exp2_path = str(Path(__file__).parent.parent.parent / "exp2")
        if exp2_path not in sys.path:
            sys.path.insert(0, exp2_path)
        
        from exp2.src.osm_feature_extractor import OsmSpatialExtractor
        
        spatial_cache_path = Path(__file__).parent.parent.parent / "exp2" / "cache" / "spatial_data.pkl"
        if spatial_cache_path.exists():
            print(f"📋 正在从缓存加载OSM数据: {spatial_cache_path}")
            with open(spatial_cache_path, 'rb') as f:
                osm_extractor = pickle.load(f)
            
            grid_cache_path = Path(__file__).parent.parent.parent / "exp2" / "cache" / "spatial_grid_cache.pkl"
            if grid_cache_path.exists():
                osm_extractor.load_cache(str(grid_cache_path))
            
            print(f"✅ OSM数据从缓存加载成功")
            return
        
        osm_geojson_path = Path(__file__).parent.parent.parent / "data" / "exp2.geojson"
        if osm_geojson_path.exists():
            print(f"📋 正在加载OSM数据: {osm_geojson_path}")
            
            from exp2.src.data_preprocessing import OSMDataLoader
            osm_loader = OSMDataLoader(str(osm_geojson_path))
            osm_data = osm_loader.load_osm_data()
            
            road_network = osm_loader.extract_road_network(osm_data)
            pois = osm_loader.extract_pois(osm_data)
            
            osm_extractor = OsmSpatialExtractor()
            osm_extractor.build_from_osm(road_network, pois)
            
            print(f"✅ OSM数据加载成功")
        else:
            print(f"⚠️ OSM数据文件不存在: {osm_geojson_path}")
    except Exception as e:
        print(f"⚠️ 加载OSM数据失败: {e}")
        import traceback
        traceback.print_exc()


def load_weather_data():
    """加载真实天气数据"""
    global weather_processor
    try:
        from exp3.src.weather_preprocessing import WeatherDataProcessor
        weather_csv_path = Path(__file__).parent.parent.parent / "data" / "beijing_weather_daily_2007_2012.csv"
        if weather_csv_path.exists():
            weather_processor = WeatherDataProcessor(str(weather_csv_path))
            weather_processor.load_and_process()
            print(f"✅ 天气数据加载成功")
        else:
            print(f"⚠️ 天气数据文件不存在: {weather_csv_path}")
    except Exception as e:
        print(f"⚠️ 加载天气数据失败: {e}")


def load_predictors():
    """加载所有预测器"""
    base_dir = Path(__file__).parent.parent.parent
    
    for exp in ['exp1', 'exp2', 'exp3', 'exp4']:
        exp_path = str(base_dir / exp)
        if exp_path not in sys.path:
            sys.path.insert(0, exp_path)
    
    # 加载 exp1
    try:
        from exp1.predict import TrajectoryPredictor
        model_path = base_dir / "exp1" / "checkpoints" / "exp1_model.pth"
        if model_path.exists():
            predictors['exp1'] = TrajectoryPredictor(str(model_path))
        else:
            print(f"⚠️ exp1 模型文件不存在: {model_path}")
    except Exception as e:
        print(f"⚠️ 加载 exp1 预测器失败: {e}")
    
    # 加载其他实验...（省略，与原代码相同）


def load_plt_file(content: bytes) -> pd.DataFrame:
    """加载 PLT 格式文件（Geolife 格式）"""
    try:
        lines = content.decode('utf-8').splitlines()
        
        if len(lines) < 7:
            raise ValueError("PLT 文件格式错误：文件过短")
        
        data_lines = lines[6:]
        
        data = []
        for line in data_lines:
            parts = line.strip().split(',')
            if len(parts) >= 6:
                try:
                    lat = float(parts[0])
                    lon = float(parts[1])
                    date_str = parts[-2]
                    time_str = parts[-1]
                    timestamp = f"{date_str} {time_str}"
                    data.append({
                        'latitude': lat,
                        'longitude': lon,
                        'timestamp': timestamp
                    })
                except (ValueError, IndexError):
                    continue
        
        if not data:
            raise ValueError("PLT 文件中没有有效的数据点")
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except Exception as e:
        raise ValueError(f"解析 PLT 文件失败: {str(e)}")


# ========== 新增：分段预测核心函数 ==========

def segment_trajectory_sliding_window(
    df: pd.DataFrame,
    window_size: int = 50,
    stride: int = 25,
    min_segment_points: int = 20
) -> List[Tuple[int, int, pd.DataFrame]]:
    """
    滑动窗口分段
    
    Args:
        df: 轨迹数据 DataFrame
        window_size: 窗口大小（点数）
        stride: 滑动步长
        min_segment_points: 最小段长度
        
    Returns:
        List of (start_idx, end_idx, segment_df)
    """
    segments = []
    n_points = len(df)
    
    if n_points < min_segment_points:
        # 轨迹太短，作为单段处理
        return [(0, n_points, df)]
    
    start = 0
    while start < n_points:
        end = min(start + window_size, n_points)
        
        # 确保最后一段不会太短
        if n_points - end < min_segment_points and end < n_points:
            end = n_points
        
        segment_df = df.iloc[start:end].reset_index(drop=True)
        
        if len(segment_df) >= min_segment_points:
            segments.append((start, end, segment_df))
        
        if end >= n_points:
            break
            
        start += stride
    
    return segments


def detect_change_points(df: pd.DataFrame, speed_threshold: float = 8.0) -> List[int]:
    """
    基于速度变化检测交通方式切换点
    
    Args:
        df: 带有 speed 列的 DataFrame
        speed_threshold: 速度变化阈值
        
    Returns:
        切换点的索引列表
    """
    if 'speed' not in df.columns or len(df) < 3:
        return []
    
    change_points = [0]
    speeds = df['speed'].fillna(0).values
    
    # 使用滑动窗口平滑
    window = 5
    smoothed = np.convolve(speeds, np.ones(window)/window, mode='same')
    
    for i in range(1, len(smoothed) - 1):
        # 检测速度突变
        speed_change = abs(smoothed[i] - smoothed[i-1])
        
        # 检测长时间停留（速度接近0持续一段时间）
        is_stopped = smoothed[i] < 0.5
        was_moving = smoothed[i-1] > 2.0
        
        # 检测从停到动或从动到停
        if (is_stopped and was_moving) or (speed_change > speed_threshold):
            # 避免切换点过于密集
            if i - change_points[-1] > 10:
                change_points.append(i)
    
    change_points.append(len(df))
    return change_points


def predict_segment(
    segment_df: pd.DataFrame,
    model_id: str = 'exp1'
) -> Tuple[str, float]:
    """
    对单个轨迹段进行预测
    
    Returns:
        (predicted_mode, confidence)
    """
    # 计算特征
    features, df_with_stats, segment_stats = compute_trajectory_features(segment_df)
    features_normalized = normalize_sequence_length(features, 50)
    
    # 模型预测
    if model_id in predictors:
        try:
            if model_id == 'exp1':
                pred_labels, confidences = predictors['exp1'].predict(
                    features_normalized, segment_stats
                )
                return pred_labels[0], float(confidences[0])
        except Exception as e:
            print(f"模型预测失败: {e}")
    
    # 规则预测作为后备
    speed = features_normalized[:, 2]
    avg_speed = float(np.mean(speed[speed > 0])) if np.any(speed > 0) else 0
    max_speed = float(np.max(speed))
    
    if max_speed > 50:
        return "train", 0.85
    elif max_speed > 30:
        return "train", 0.80
    elif avg_speed > 15:
        return "subway", 0.78
    elif avg_speed > 8:
        return "car", 0.75
    elif avg_speed > 4:
        return "bus", 0.72
    elif avg_speed > 2:
        return "bike", 0.70
    else:
        return "walk", 0.68


def merge_consecutive_segments(
    segments: List[dict],
    min_duration: float = 60.0
) -> List[dict]:
    """
    合并相邻的相同交通方式段，并移除过短的段
    
    Args:
        segments: 分段预测结果列表
        min_duration: 最小段时长（秒），小于此值的段会被合并
        
    Returns:
        合并后的分段列表
    """
    if not segments:
        return []
    
    merged = [segments[0].copy()]
    
    for seg in segments[1:]:
        last = merged[-1]
        
        # 如果模式相同，合并
        if seg['predicted_mode'] == last['predicted_mode']:
            last['end_index'] = seg['end_index']
            last['end_time'] = seg['end_time']
            last['duration'] += seg['duration']
            last['distance'] += seg['distance']
            last['points'].extend(seg['points'])
            # 重新计算平均速度和置信度
            last['avg_speed'] = last['distance'] / (last['duration'] + 1e-6)
            last['confidence'] = (last['confidence'] + seg['confidence']) / 2
        else:
            # 检查当前段是否太短
            if seg['duration'] < min_duration and len(merged) > 0:
                # 太短的段，合并到前一段
                last['end_index'] = seg['end_index']
                last['end_time'] = seg['end_time']
                last['duration'] += seg['duration']
                last['distance'] += seg['distance']
                last['points'].extend(seg['points'])
                last['avg_speed'] = last['distance'] / (last['duration'] + 1e-6)
            else:
                merged.append(seg.copy())
    
    # 重新编号
    for i, seg in enumerate(merged):
        seg['segment_id'] = i
    
    return merged


def compute_mode_breakdown(segments: List[dict]) -> List[dict]:
    """
    计算各交通方式的占比统计
    """
    mode_stats = defaultdict(lambda: {'distance': 0.0, 'duration': 0.0})
    total_distance = 0.0
    total_duration = 0.0
    
    for seg in segments:
        mode = seg['predicted_mode']
        mode_stats[mode]['distance'] += seg['distance']
        mode_stats[mode]['duration'] += seg['duration']
        total_distance += seg['distance']
        total_duration += seg['duration']
    
    breakdown = []
    for mode, stats in mode_stats.items():
        breakdown.append({
            'mode': mode,
            'percentage': stats['distance'] / (total_distance + 1e-6),
            'distance': stats['distance'],
            'duration': stats['duration']
        })
    
    # 按占比排序
    breakdown.sort(key=lambda x: x['percentage'], reverse=True)
    return breakdown


# ========== 原有的特征计算函数 ==========

def compute_trajectory_features(df: pd.DataFrame) -> tuple:
    """计算轨迹特征（9维）和统计特征（18维）"""
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    df['time_diff'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
    
    # 计算距离
    lat1 = df['latitude'].shift(1).fillna(df['latitude'].iloc[0])
    lon1 = df['longitude'].shift(1).fillna(df['longitude'].iloc[0])
    lat2 = df['latitude']
    lon2 = df['longitude']
    
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    df['distance'] = 6371000 * c
    
    # 计算速度
    df['speed'] = df['distance'] / df['time_diff'].replace(0, np.nan)
    df['speed'] = df['speed'].fillna(0)
    
    # 计算加速度
    df['acceleration'] = df['speed'].diff() / df['time_diff'].replace(0, np.nan)
    df['acceleration'] = df['acceleration'].fillna(0)
    
    # 计算方向变化
    df['bearing'] = np.degrees(np.arctan2(
        np.sin(lon2_rad - lon1_rad) * np.cos(lat2_rad),
        np.cos(lat1_rad) * np.sin(lat2_rad) - np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(lon2_rad - lon1_rad)
    ))
    df['bearing_change'] = df['bearing'].diff().fillna(0)
    df['bearing_change'] = df['bearing_change'].apply(lambda x: x if abs(x) <= 180 else x - 360 * np.sign(x))
    
    # 累计距离和时间
    df['total_distance'] = df['distance'].cumsum()
    df['total_time'] = df['time_diff'].cumsum()
    
    # 构建特征矩阵
    features = np.column_stack([
        df['latitude'].values,
        df['longitude'].values,
        df['speed'].values,
        df['acceleration'].values,
        df['bearing_change'].values,
        df['distance'].values,
        df['time_diff'].values,
        df['total_distance'].values,
        df['total_time'].values
    ]).astype(np.float32)
    
    # 计算统计特征
    segment_stats = compute_segment_stats(features)
    
    return features, df, segment_stats


def compute_segment_stats(features: np.ndarray) -> np.ndarray:
    """计算18维统计特征"""
    eps = 1e-8
    N = len(features)
    
    if N == 0:
        return np.zeros(18, dtype=np.float32)
    
    speed = features[:, 2]
    acceleration = features[:, 3]
    bearing_change = features[:, 4]
    
    speed_mean = float(np.mean(speed))
    speed_std = float(np.std(speed))
    speed_max = float(np.max(speed))
    speed_cv = speed_std / (speed_mean + eps)
    
    accel_mean = float(np.mean(np.abs(acceleration)))
    accel_std = float(np.std(acceleration))
    accel_max = float(np.max(np.abs(acceleration)))
    
    bearing_mean = float(np.mean(np.abs(bearing_change)))
    bearing_std = float(np.std(bearing_change))
    
    total_dist = float(np.sum(features[:, 5]))
    total_time = float(np.sum(features[:, 6]))
    
    stop_ratio = float(np.mean(speed < 0.5))
    high_speed_ratio = float(np.mean(speed > 15.0))
    
    # 线性度
    if total_dist > eps:
        lat_start, lon_start = features[0, 0], features[0, 1]
        lat_end, lon_end = features[-1, 0], features[-1, 1]
        straight_dist = np.sqrt(
            ((lat_end - lat_start) * 111300) ** 2 +
            ((lon_end - lon_start) * 111300 * np.cos(np.radians(lat_start))) ** 2
        )
        linearity = float(min(straight_dist / (total_dist + eps), 1.0))
    else:
        linearity = 0.0
    
    avg_segment_speed = float(total_dist / (total_time + eps))
    
    # 速度熵
    if speed_max > eps:
        hist, _ = np.histogram(speed, bins=10, range=(0, speed_max + eps))
        hist = hist / (hist.sum() + eps)
        hist = hist[hist > 0]
        speed_entropy = float(-np.sum(hist * np.log(hist + eps)))
    else:
        speed_entropy = 0.0
    
    # 加速度符号变化率
    if N > 1:
        sign_changes = np.sum(np.diff(np.sign(acceleration)) != 0)
        accel_sign_changes = float(sign_changes / (N - 1))
    else:
        accel_sign_changes = 0.0
    
    # 高速持续比例
    high_speed_mask = speed > 10.0
    max_sustained = 0
    current_run = 0
    for v in high_speed_mask:
        if v:
            current_run += 1
            max_sustained = max(max_sustained, current_run)
        else:
            current_run = 0
    max_sustained_speed = float(max_sustained / N)
    
    stats = np.array([
        speed_mean, speed_std, speed_max, speed_cv,
        accel_mean, accel_std, accel_max,
        bearing_mean, bearing_std,
        stop_ratio, high_speed_ratio,
        linearity,
        total_dist, total_time, avg_segment_speed,
        speed_entropy,
        accel_sign_changes,
        max_sustained_speed
    ], dtype=np.float32)
    
    stats = np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
    
    return stats


def normalize_sequence_length(features: np.ndarray, target_length: int = 50) -> np.ndarray:
    """统一序列长度"""
    if len(features) == target_length:
        return features
    elif len(features) > target_length:
        indices = np.linspace(0, len(features) - 1, target_length, dtype=int)
        return features[indices]
    else:
        feature_dim = features.shape[1] if features.ndim > 1 else 1
        padding = np.zeros((target_length - len(features), feature_dim), dtype=np.float32)
        return np.vstack([features, padding])


# ========== 更新后的预测端点 ==========

@router.post("/predict", response_model=TrajectoryPrediction)
async def predict_trajectory(
    file: UploadFile = File(...),
    model: str = Form('exp1'),
    enable_segmentation: bool = Form(True),  # 新增：是否启用分段
    window_size: int = Form(50),              # 新增：窗口大小
    min_segment_duration: float = Form(60.0)  # 新增：最小段时长
):
    """
    上传GPS文件并预测交通方式
    
    新增功能：
    - enable_segmentation: 是否启用多段识别
    - window_size: 滑动窗口大小
    - min_segment_duration: 最小段时长（秒）
    """
    try:
        content = await file.read()
        
        # 解析文件
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
        elif file.filename.endswith('.json'):
            data = pd.read_json(io.StringIO(content.decode('utf-8')))
            df = pd.DataFrame(data)
        elif file.filename.endswith('.plt'):
            df = load_plt_file(content)
        else:
            raise HTTPException(status_code=400, detail="不支持的文件格式")
        
        # 验证必需列
        required_columns = ['latitude', 'longitude', 'timestamp']
        for col in required_columns:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"缺少必需列: {col}")
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        if len(df) < 10:
            raise HTTPException(status_code=400, detail="轨迹点太少")
        
        # 计算基础特征（用于整体统计）
        features, df_with_stats, _ = compute_trajectory_features(df)
        
        segments_result = []
        
        if enable_segmentation and len(df) > window_size:
            # ========== 分段预测 ==========
            print(f"📊 启用分段预测，窗口大小: {window_size}")
            
            # 方案1：滑动窗口分段
            raw_segments = segment_trajectory_sliding_window(
                df, window_size=window_size, stride=window_size // 2
            )
            
            # 对每段预测
            for start_idx, end_idx, seg_df in raw_segments:
                mode, confidence = predict_segment(seg_df, model)
                
                # 计算段统计
                seg_features, seg_df_stats, _ = compute_trajectory_features(seg_df)
                seg_distance = float(seg_df_stats['distance'].sum())
                seg_duration = float(seg_df_stats['time_diff'].sum())
                
                segments_result.append({
                    'segment_id': len(segments_result),
                    'predicted_mode': mode,
                    'confidence': confidence,
                    'start_index': start_idx,
                    'end_index': end_idx,
                    'start_time': seg_df['timestamp'].iloc[0].isoformat(),
                    'end_time': seg_df['timestamp'].iloc[-1].isoformat(),
                    'duration': seg_duration,
                    'distance': seg_distance,
                    'avg_speed': seg_distance / (seg_duration + 1e-6),
                    'points': [
                        TrajectoryPoint(
                            lat=row['latitude'],
                            lng=row['longitude'],
                            timestamp=row['timestamp'].isoformat(),
                            speed=row.get('speed', 0)
                        )
                        for _, row in seg_df_stats.iterrows()
                    ]
                })
            
            # 合并相邻相同模式的段
            segments_result = merge_consecutive_segments(
                segments_result, min_duration=min_segment_duration
            )
            
            print(f"✅ 分段结果: {len(segments_result)} 段")
            for seg in segments_result:
                print(f"   段 {seg['segment_id']}: {seg['predicted_mode']} "
                      f"(置信度: {seg['confidence']:.2f}, "
                      f"距离: {seg['distance']:.0f}m, "
                      f"时长: {seg['duration']:.0f}s)")
        
        else:
            # ========== 单段预测（原逻辑） ==========
            mode, confidence = predict_segment(df, model)
            
            total_distance = float(df_with_stats['distance'].sum())
            total_time = float(df_with_stats['time_diff'].sum())
            
            segments_result.append({
                'segment_id': 0,
                'predicted_mode': mode,
                'confidence': confidence,
                'start_index': 0,
                'end_index': len(df),
                'start_time': df['timestamp'].iloc[0].isoformat(),
                'end_time': df['timestamp'].iloc[-1].isoformat(),
                'duration': total_time,
                'distance': total_distance,
                'avg_speed': total_distance / (total_time + 1e-6),
                'points': [
                    TrajectoryPoint(
                        lat=row['latitude'],
                        lng=row['longitude'],
                        timestamp=row['timestamp'].isoformat(),
                        speed=row.get('speed', 0)
                    )
                    for _, row in df_with_stats.iterrows()
                ]
            })
        
        # 计算各模式占比
        mode_breakdown = compute_mode_breakdown(segments_result)
        
        # 确定主要交通方式（占比最大的）
        primary_mode = mode_breakdown[0]['mode'] if mode_breakdown else 'unknown'
        primary_confidence = segments_result[0]['confidence'] if segments_result else 0.0
        
        # 标签映射
        mode_mapping = {
            'Walk': 'walk',
            'Bike': 'bike',
            'Bus': 'bus',
            'Car & taxi': 'car',
            'Subway': 'subway',
            'Train': 'train'
        }
        primary_mode = mode_mapping.get(primary_mode, primary_mode.lower())
        
        # 对所有段的标签也做映射
        for seg in segments_result:
            seg['predicted_mode'] = mode_mapping.get(
                seg['predicted_mode'], seg['predicted_mode'].lower()
            )
        
        for mb in mode_breakdown:
            mb['mode'] = mode_mapping.get(mb['mode'], mb['mode'].lower())
        
        # 整体统计
        total_distance = float(df_with_stats['distance'].sum())
        total_time = float(df_with_stats['time_diff'].sum())
        avg_speed = total_distance / (total_time + 1e-6)
        max_speed = float(df_with_stats['speed'].max())
        
        trajectory_id = f"traj_{hash(file.filename) % 1000000}"
        
        # 所有轨迹点
        all_points = [
            TrajectoryPoint(
                lat=row['latitude'],
                lng=row['longitude'],
                timestamp=row['timestamp'].isoformat(),
                speed=row['speed']
            )
            for _, row in df_with_stats.iterrows()
        ]
        
        stats = TrajectoryStats(
            distance=total_distance,
            duration=total_time,
            avg_speed=avg_speed,
            max_speed=max_speed
        )
        
        # 构建分段对象列表
        segment_predictions = [
            SegmentPrediction(**seg) for seg in segments_result
        ]
        
        mode_breakdown_objs = [
            ModeBreakdown(**mb) for mb in mode_breakdown
        ]
        
        return TrajectoryPrediction(
            trajectory_id=trajectory_id,
            predicted_mode=primary_mode,
            confidence=primary_confidence,
            segments=segment_predictions,          # 新增
            mode_breakdown=mode_breakdown_objs,    # 新增
            points=all_points,
            stats=stats
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@router.get("/modes", response_model=List[TransportMode])
async def get_transport_modes():
    """获取所有交通方式的配置"""
    return [TransportMode(**mode) for mode in TRANSPORT_MODES]