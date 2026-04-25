from fastapi import APIRouter, HTTPException
from pathlib import Path
import pickle
import numpy as np
from collections import Counter
from typing import Dict, List

from api.schemas import DatasetStats, DataCleaningStats, DataCleaningStep

router = APIRouter()

# Resolve project paths from this file location so runtime cwd never affects data loading.
BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED_DIR = (BASE_DIR / "data" / "processed").resolve()
CLEANED_DATA_PATH = PROCESSED_DIR / "cleaned_balanced.pkl"
BASE_SEGMENTS_PATH = PROCESSED_DIR / "base_segments.pkl"


@router.get("/stats", response_model=DatasetStats)
async def get_dataset_stats():
    """获取数据集统计信息"""
    if not PROCESSED_DIR.exists():
        raise HTTPException(status_code=404, detail=f"处理后数据目录不存在: {PROCESSED_DIR}")

    if not CLEANED_DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="清洗后的数据文件不存在，请先运行数据准备脚本")
    
    try:
        with open(CLEANED_DATA_PATH, 'rb') as f:
            segments = pickle.load(f)
        
        total_trajectories = len(segments)
        
        labels = [seg[3] for seg in segments]
        mode_distribution = dict(Counter(labels))
        
        trajectory_lengths = [len(seg[0]) for seg in segments]
        avg_trajectory_length = float(np.mean(trajectory_lengths))
        
        if BASE_SEGMENTS_PATH.exists():
            with open(BASE_SEGMENTS_PATH, 'rb') as f:
                base_segments = pickle.load(f)
            users = set()
            for seg in base_segments:
                user_id = seg.get('user_id')
                if user_id:
                    users.add(user_id)
            total_users = len(users)
        else:
            total_users = 182
        
        date_range = {
            "start": "2007-04-12",
            "end": "2012-08-01"
        }
        
        return DatasetStats(
            total_trajectories=total_trajectories,
            total_users=total_users,
            mode_distribution=mode_distribution,
            avg_trajectory_length=avg_trajectory_length,
            date_range=date_range,
            total_distance="2.5M km"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取数据集统计失败: {str(e)}")


@router.get("/mode-distribution")
async def get_mode_distribution():
    """获取交通方式分布详情"""
    if not PROCESSED_DIR.exists():
        raise HTTPException(status_code=404, detail=f"处理后数据目录不存在: {PROCESSED_DIR}")

    if not CLEANED_DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="清洗后的数据文件不存在")
    
    try:
        with open(CLEANED_DATA_PATH, 'rb') as f:
            segments = pickle.load(f)
        
        labels = [seg[3] for seg in segments]
        counter = Counter(labels)
        
        total = len(labels)
        distribution = []
        
        for mode, count in sorted(counter.items()):
            percentage = (count / total) * 100
            distribution.append({
                "mode": mode,
                "count": count,
                "percentage": round(percentage, 2)
            })
        
        return {
            "total": total,
            "distribution": distribution
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"读取交通方式分布失败: {str(e)}")


@router.get("/cleaning-stats", response_model=DataCleaningStats)
async def get_cleaning_stats():
    """获取数据清洗流程统计"""
    return DataCleaningStats(
        steps=[
            DataCleaningStep(name="原始数据", count=17621),
            DataCleaningStep(name="基础过滤", count=15847),
            DataCleaningStep(name="深度清洗", count=14236),
            DataCleaningStep(name="归一化", count=14236)
        ]
    )
