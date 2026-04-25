#!/usr/bin/env python3
"""
统计训练中 Geolife / OSM / 天气数据的使用量。

口径说明：
1) Geolife：使用 cleaned_<mode>.pkl 的样本数（与 exp1/exp2/exp3 的主样本源一致）
2) OSM：统计 exp2.geojson 的原始要素数、道路数、POI数；
   同时统计已提取成功的 OSM 融合样本数（优先读取 exp2 缓存）
3) 天气：统计天气 CSV 的日记录数；
   同时统计参与多模态训练样本涉及的唯一日期数
4) 训练样本量：按与训练脚本一致的 70/10/20 划分计算
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent
DATA_DIR = BACKEND_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

EXP2_CACHE = BACKEND_DIR / "exp2" / "cache" / "processed_features.pkl"
EXP3_CACHE = BACKEND_DIR / "exp3" / "cache" / "processed_features.pkl"
OSM_GEOJSON = DATA_DIR / "exp2.geojson"
WEATHER_CSV = DATA_DIR / "beijing_weather_daily_2007_2012.csv"


def pick_cleaned_path(mode: str) -> Path:
    preferred = PROCESSED_DIR / f"cleaned_{mode}.pkl"
    fallback = PROCESSED_DIR / "cleaned_balanced.pkl"
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        f"未找到清洗数据文件: {preferred} 或 {fallback}"
    )


def safe_load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def split_701020(total_count: int, labels: list) -> tuple[int, int, int]:
    if total_count == 0:
        return 0, 0, 0

    indices = np.arange(total_count)

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.3,
        random_state=42,
        stratify=labels if labels else None,
    )
    temp_labels = [labels[i] for i in temp_idx] if labels else None
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.6667,
        random_state=42,
        stratify=temp_labels if temp_labels else None,
    )
    return len(train_idx), len(val_idx), len(test_idx)


def parse_datetime_series(series_obj) -> list[pd.Timestamp]:
    if series_obj is None:
        return []

    values = series_obj
    if isinstance(values, pd.Series):
        values = values.tolist()

    timestamps: list[pd.Timestamp] = []
    for v in values:
        try:
            ts = pd.Timestamp(v)
            if pd.isna(ts):
                continue
            timestamps.append(ts)
        except Exception:
            continue
    return timestamps


def count_osm_raw(geojson_path: Path) -> dict:
    if not geojson_path.exists():
        return {
            "exists": False,
            "total_features": 0,
            "road_features": 0,
            "poi_features": 0,
        }

    with geojson_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    features = data.get("features", [])
    road_count = 0
    poi_count = 0
    poi_types = {"bus_stop", "station", "parking", "taxi", "subway_entrance"}

    for feat in features:
        props = feat.get("properties", {}) if isinstance(feat, dict) else {}
        highway = props.get("highway", "")
        railway = props.get("railway", "")
        amenity = props.get("amenity", "")

        if highway or railway:
            road_count += 1

        if (
            highway in poi_types
            or railway in poi_types
            or amenity in poi_types
            or highway == "bus_stop"
            or railway == "station"
            or amenity in {"parking", "taxi"}
        ):
            poi_count += 1

    return {
        "exists": True,
        "total_features": len(features),
        "road_features": road_count,
        "poi_features": poi_count,
    }


def to_date_set_from_cleaned(cleaned_data: list, limit: Optional[int] = None) -> set[pd.Timestamp]:
    date_set: set[pd.Timestamp] = set()
    data = cleaned_data if limit is None else cleaned_data[:limit]

    for item in data:
        if len(item) < 3:
            continue
        dt_series = item[2]
        for ts in parse_datetime_series(dt_series):
            date_set.add(ts.normalize())

    return date_set


def closest_within_7_days(date_index: pd.DatetimeIndex, d: pd.Timestamp) -> bool:
    if d in date_index:
        return True
    if len(date_index) == 0:
        return False

    deltas = np.abs((date_index - d).days)
    return bool(deltas.min() <= 7)


def main() -> None:
    parser = argparse.ArgumentParser(description="统计模型训练数据使用量")
    parser.add_argument("--mode", default="balanced", choices=["strict", "balanced", "gentle"])
    args = parser.parse_args()

    cleaned_path = pick_cleaned_path(args.mode)
    cleaned_data = safe_load_pickle(cleaned_path)

    geolife_total = len(cleaned_data)
    geolife_labels = [item[3] for item in cleaned_data if len(item) >= 4]
    geolife_train, geolife_val, geolife_test = split_701020(geolife_total, geolife_labels)

    exp2_total = 0
    exp2_labels = []
    exp2_train = exp2_val = exp2_test = 0
    if EXP2_CACHE.exists():
        exp2_cached = safe_load_pickle(EXP2_CACHE)
        exp2_samples = exp2_cached[0]
        exp2_total = len(exp2_samples)
        exp2_labels = [item[2] for item in exp2_samples if len(item) >= 3]
        exp2_train, exp2_val, exp2_test = split_701020(exp2_total, exp2_labels)

    exp3_total = 0
    exp3_labels = []
    exp3_train = exp3_val = exp3_test = 0
    if EXP3_CACHE.exists():
        exp3_cached = safe_load_pickle(EXP3_CACHE)
        exp3_samples = exp3_cached[0]
        exp3_total = len(exp3_samples)
        exp3_labels = [item[2] for item in exp3_samples if len(item) >= 3]
        exp3_train, exp3_val, exp3_test = split_701020(exp3_total, exp3_labels)

    osm_raw = count_osm_raw(OSM_GEOJSON)

    weather_total_days = 0
    weather_start = None
    weather_end = None
    weather_index = pd.DatetimeIndex([])
    if WEATHER_CSV.exists():
        weather_df = pd.read_csv(WEATHER_CSV, index_col=0, parse_dates=True)
        weather_total_days = len(weather_df)
        if weather_total_days > 0:
            weather_index = pd.DatetimeIndex(weather_df.index.normalize().unique())
            weather_start = str(weather_df.index.min().date())
            weather_end = str(weather_df.index.max().date())

    # 多模态训练口径：exp3 可用样本；若无exp3缓存则回退到 min(cleaned, exp2)
    multimodal_total = exp3_total if exp3_total > 0 else min(geolife_total, exp2_total if exp2_total > 0 else geolife_total)

    used_dates = to_date_set_from_cleaned(cleaned_data, limit=multimodal_total)
    weather_exact_match_days = 0
    weather_match_with_fallback_days = 0
    if len(weather_index) > 0 and used_dates:
        for d in used_dates:
            if d in weather_index:
                weather_exact_match_days += 1
            if closest_within_7_days(weather_index, d):
                weather_match_with_fallback_days += 1

    print("=" * 70)
    print("训练数据使用量统计")
    print("=" * 70)
    print(f"统计时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"清洗数据文件: {cleaned_path}")

    print("\n[Geolife 轨迹数据]")
    print(f"- 清洗后总样本数: {geolife_total}")
    print(f"- 训练/验证/测试(70/10/20): {geolife_train}/{geolife_val}/{geolife_test}")
    if geolife_labels:
        dist = dict(Counter(geolife_labels))
        print(f"- 标签分布: {dist}")

    print("\n[OSM 空间数据]")
    if osm_raw["exists"]:
        print(f"- 原始 GeoJSON 特征数: {osm_raw['total_features']}")
        print(f"- 其中道路特征数: {osm_raw['road_features']}")
        print(f"- 其中POI特征数: {osm_raw['poi_features']}")
    else:
        print("- 未找到 OSM 文件")

    if exp2_total > 0:
        print(f"- OSM融合后可用样本数(exp2缓存): {exp2_total}")
        print(f"- 训练/验证/测试(70/10/20): {exp2_train}/{exp2_val}/{exp2_test}")
    else:
        print("- 未找到 exp2 特征缓存，暂无法给出 OSM 融合后样本数")

    print("\n[天气数据]")
    if weather_total_days > 0:
        print(f"- 天气日记录数: {weather_total_days}")
        print(f"- 覆盖时间: {weather_start} ~ {weather_end}")
    else:
        print("- 未找到天气CSV或记录为空")

    print(f"- 多模态样本涉及唯一日期数: {len(used_dates)}")
    print(f"- 与天气表精确匹配日期数: {weather_exact_match_days}")
    print(f"- 考虑±7天回退后可匹配日期数: {weather_match_with_fallback_days}")

    if exp3_total > 0:
        print(f"- 天气融合后可用样本数(exp3缓存): {exp3_total}")
        print(f"- 训练/验证/测试(70/10/20): {exp3_train}/{exp3_val}/{exp3_test}")
    else:
        print("- 未找到 exp3 特征缓存，暂无法给出天气融合后样本数")

    print("\n[结论口径建议]")
    print(f"- 仅轨迹训练(exp1)可认为使用了 {geolife_train} 条训练样本")
    if exp2_total > 0:
        print(f"- 轨迹+OSM 训练(exp2)可认为使用了 {exp2_train} 条训练样本")
    if exp3_total > 0:
        print(f"- 轨迹+OSM+天气 训练(exp3/exp4)可认为使用了 {exp3_train} 条训练样本")

    print("=" * 70)


if __name__ == "__main__":
    main()
