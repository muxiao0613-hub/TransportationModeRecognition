"""
OSM 空间特征提取模块 (exp2 - 基础版)

功能:
    基于 OpenStreetMap 道路网络和 POI 数据，
    为轨迹点提取 11 维空间上下文特征。

特征组成（共11维）:
    [0:6]  道路类型 one-hot: walk/bike/car/bus/train/unknown
    [6:10] POI 邻近标记: 公交站/停车场/地铁站/最近POI距离(归一化)
    [10]   道路密度: 100m 范围内道路节点数(归一化)

依赖:
    - osmnx:  OSM 数据获取
    - scipy:  KDTree 空间索引
    - numpy:  特征矩阵运算

注意:
    本模块与"知识图谱"无关，仅是基于地理空间的特征工程。
    所有查询结果会缓存到磁盘以加速后续运行。
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from geopy.distance import geodesic
import networkx as nx
from collections import defaultdict
from tqdm import tqdm
from scipy.spatial import KDTree
import pickle
import os
from decimal import Decimal


class OsmSpatialExtractor:
    """
    OSM 空间特征提取器 (基础版)

    职责:
        从 OSM 数据构建空间索引，为轨迹点提取空间上下文特征。

    主要属性:
        graph: NetworkX 图对象，存储道路节点和 POI 节点
        road_kdtree: 道路节点的 KDTree 空间索引
        poi_kdtree: POI 节点的 KDTree 空间索引
        _grid_cache: 网格缓存字典，加速重复查询
        _cache_resolution: 网格缓存分辨率（约 111 米）
    """

    def __init__(self):
        """
        初始化 OSM 空间特征提取器。

        初始化内容:
            - 创建空的 NetworkX 图
            - 初始化道路类型映射字典
            - 初始化 KDTree 索引为 None
            - 初始化网格缓存字典
        """
        self.graph = nx.MultiDiGraph()
        self.road_network = None
        self.pois = None
        self.road_type_mapping = {
            'footway': 'walk',
            'cycleway': 'bike',
            'primary': 'car',
            'secondary': 'car',
            'tertiary': 'car',
            'residential': 'car',
            'bus_stop': 'bus',
            'station': 'train',
            'subway_entrance': 'subway',
            'rail': 'train',
            'subway': 'subway',
            'parking': 'car',
            'taxi': 'car'
        }

        # KDTree 索引
        self.road_kdtree = None
        self.road_coords = None
        self.road_types = None

        self.poi_kdtree = None
        self.poi_coords = None
        self.poi_types = None

        # 网格缓存
        self._grid_cache = {}
        self._cache_resolution = 0.001  # 约 111 米
        self._cache_hits = 0
        self._cache_misses = 0

    def build_from_osm(self, road_network: pd.DataFrame, pois: pd.DataFrame):
        """
        从 OSM 数据构建空间特征提取器。

        构建流程:
            1. 添加道路网络节点和路段到图
            2. 添加 POI 节点到图
            3. 构建 KDTree 空间索引
            4. 关联 POI 到最近的道路节点

        参数:
            road_network (pd.DataFrame): 道路网络数据，包含 id、highway、coordinates 等字段
            pois (pd.DataFrame): POI 数据，包含 id、type、coordinates 等字段
        """
        self.road_network = road_network
        self.pois = pois

        print(" -> 正在添加道路节点和路段...")
        self._add_road_network()
        print(f" -> 道路网络添加完成。当前图节点数: {self.graph.number_of_nodes()}")

        print(" -> 正在添加 POI 节点...")
        self._add_pois()
        print(f" -> POI 节点添加完成。当前图节点数: {self.graph.number_of_nodes()}")

        print(" -> 正在构建 KDTree 空间索引...")
        self._build_spatial_indices()
        print(" -> KDTree 索引构建完成！")

        self._link_roads_to_pois()

    def _convert_to_float(self, value):
        """
        转换任意数值类型为 float。

        用于处理 Decimal 类型（来自 GeoJSON 解析），
        确保所有坐标数据为 float 类型以兼容 numpy。

        参数:
            value: 任意数值类型（int、float、Decimal 等）

        返回:
            float: 转换后的浮点数
        """
        if isinstance(value, Decimal):
            return float(value)
        return float(value)

    def _add_road_network(self):
        """
        添加道路网络到空间特征提取器。

        处理逻辑:
            1. 遍历道路网络 DataFrame
            2. 对每条道路，解析其坐标序列
            3. 为每个坐标点创建道路节点
            4. 在相邻节点之间创建道路路段边
            5. 计算路段距离（使用 Haversine 公式）

        注意:
            支持 LineString 和 Polygon 两种几何类型
            所有坐标值都会转换为 float 类型
        """
        for _, road in self.road_network.iterrows():
            road_id = road['id']
            road_type = road.get('highway') or road.get('railway', '')
            coordinates = road.get('coordinates', [])

            if not coordinates:
                continue

            if road['geometry_type'] == 'LineString':
                coords = coordinates
            elif road['geometry_type'] == 'Polygon':
                coords = coordinates[0]
            else:
                continue

            for i, coord in enumerate(coords):
                node_id = f"{road_id}_node_{i}"
                # 修复：转换 Decimal 为 float
                lon, lat = self._convert_to_float(coord[0]), self._convert_to_float(coord[1])

                self.graph.add_node(node_id,
                                  type='road_node',
                                  road_id=road_id,
                                  road_type=road_type,
                                  latitude=lat,
                                  longitude=lon)

                if i > 0:
                    prev_node_id = f"{road_id}_node_{i-1}"
                    # Haversine 距离计算（单位：米）
                    distance = geodesic((coords[i-1][1], coords[i-1][0]),
                                       (lat, lon)).meters

                    self.graph.add_edge(prev_node_id, node_id,
                                      type='road_segment',
                                      road_type=road_type,
                                      distance=distance)

    def _add_pois(self):
        """
        添加 POI 到空间特征提取器。

        处理逻辑:
            1. 遍历 POI DataFrame
            2. 解析 POI 坐标（支持 Point 和 MultiPoint）
            3. 为每个 POI 创建节点，存储类型和名称信息

        注意:
            所有坐标值都会转换为 float 类型
        """
        for _, poi in self.pois.iterrows():
            poi_id = poi['id']
            poi_type = poi['type']
            coordinates = poi.get('coordinates', [])

            if not coordinates:
                continue

            if isinstance(coordinates[0], list):
                # 修复：转换 Decimal 为 float
                lon, lat = self._convert_to_float(coordinates[0][0]), self._convert_to_float(coordinates[0][1])
            else:
                # 修复：转换 Decimal 为 float
                lon, lat = self._convert_to_float(coordinates[0]), self._convert_to_float(coordinates[1])

            self.graph.add_node(poi_id,
                              type='poi',
                              poi_type=poi_type,
                              name=poi.get('name', ''),
                              latitude=lat,
                              longitude=lon)

    def _build_spatial_indices(self):
        """
        预构建 KDTree 空间索引。

        构建内容:
            1. 道路节点 KDTree: 用于快速查询最近道路
            2. POI 节点 KDTree: 用于快速查询附近 POI

        性能优化:
            - 预先构建索引，避免每次查询时重新构建
            - 使用 KDTree 的 query 和 query_ball_point 方法加速空间查询
        """

        # 1. 构建道路节点索引
        road_nodes_data = [
            (d['latitude'], d['longitude'], d['road_type'])
            for n, d in self.graph.nodes(data=True)
            if d.get('type') == 'road_node'
        ]

        if road_nodes_data:
            # 确保坐标是 float 类型
            self.road_coords = np.array([
                (float(lat), float(lon)) for lat, lon, _ in road_nodes_data
            ], dtype=np.float64)
            self.road_types = [
                self.road_type_mapping.get(road_type, 'unknown')
                for _, _, road_type in road_nodes_data
            ]
            self.road_kdtree = KDTree(self.road_coords)
            print(f"   -> 道路 KDTree: {len(self.road_coords)} 个节点")

        # 2. 构建 POI 索引
        poi_nodes_data = [
            (d['latitude'], d['longitude'], d['poi_type'])
            for n, d in self.graph.nodes(data=True)
            if d.get('type') == 'poi'
        ]

        if poi_nodes_data:
            # 确保坐标是 float 类型
            self.poi_coords = np.array([
                (float(lat), float(lon)) for lat, lon, _ in poi_nodes_data
            ], dtype=np.float64)
            self.poi_types = [poi_type for _, _, poi_type in poi_nodes_data]
            self.poi_kdtree = KDTree(self.poi_coords)
            print(f"   -> POI KDTree: {len(self.poi_coords)} 个节点")

    # ========== 核心：混合查询策略 ==========
    def extract_spatial_features(self, trajectory: np.ndarray) -> np.ndarray:
        """
        为轨迹序列的每个点提取 OSM 空间上下文特征。

        算法流程:
            1. 对每个轨迹点计算网格 key（精度约 111m）
            2. 命中网格缓存则直接读取，否则加入待查询列表
            3. 对未缓存点批量执行 KDTree 最近邻查询
            4. 将查询结果写回网格缓存

        参数:
            trajectory (np.ndarray): 形状 (N, 9) 的轨迹数组，
                                     第0列=纬度，第1列=经度。

        返回:
            np.ndarray: 形状 (N, 12) 的特征矩阵，
                        dtype=float32，不含 NaN 或 Inf。
        """
        if self.road_kdtree is None or self.poi_kdtree is None:
            return np.zeros((trajectory.shape[0], 12), dtype=np.float32)

        N = trajectory.shape[0]
        spatial_features = np.zeros((N, 12), dtype=np.float32)

        uncached_indices = []
        uncached_coords = []

        # 步骤1: 检查缓存
        for i in range(N):
            lat, lon = float(trajectory[i, 0]), float(trajectory[i, 1])
            grid_key = self._get_grid_key(lat, lon)

            if grid_key in self._grid_cache:
                spatial_features[i] = self._grid_cache[grid_key]
                self._cache_hits += 1
            else:
                uncached_indices.append(i)
                uncached_coords.append([lat, lon])
                self._cache_misses += 1

        # 步骤2: 批量查询未缓存的点
        if uncached_indices:
            uncached_coords = np.array(uncached_coords, dtype=np.float64)
            uncached_features = self._batch_query_all(uncached_coords)

            # 步骤3: 更新缓存和结果
            for i, idx in enumerate(uncached_indices):
                lat, lon = float(trajectory[idx, 0]), float(trajectory[idx, 1])
                grid_key = self._get_grid_key(lat, lon)
                self._grid_cache[grid_key] = uncached_features[i]
                spatial_features[idx] = uncached_features[i]

        # ========== 统计分析（覆盖率 + road_type分布）==========
        if not hasattr(self, '_coverage_stats'):
            self._coverage_stats = []
        if not hasattr(self, '_type_counts'):
            self._type_counts = {'walk':0,'bike':0,'car':0,'bus':0,
                                 'train':0,'subway':0,'unknown':0}
            self._total_points = 0
        if not hasattr(self, '_traj_count'):
            self._traj_count = 0

        # 覆盖率（非unknown点的比例）
        road_hits = int(spatial_features[:, :6].sum(axis=1).astype(bool).sum())
        coverage = road_hits / N
        self._coverage_stats.append(coverage)

        # road_type分布（每个点累计）
        type_names = ['walk','bike','car','bus','train','subway','unknown']
        for j, t in enumerate(type_names):
            self._type_counts[t] += int(spatial_features[:, j].sum())
        self._total_points += N
        self._traj_count += 1

        # 每1000条轨迹打印一次
        if self._traj_count % 1000 == 0:
            avg_cov = float(np.mean(self._coverage_stats[-1000:]))
            print(f"\n   [Road Type分布] 累计{self._total_points}个点 "
                  f"(覆盖率{avg_cov:.1%}):")
            for t in type_names:
                pct = self._type_counts[t] / max(self._total_points, 1) * 100
                bar = '█' * int(pct / 2)
                print(f"     {t:10s}: {pct:5.1f}%  {bar}")

        return spatial_features.astype(np.float32)

    def _get_grid_key(self, lat: float, lon: float) -> Tuple[int, int]:
        """
        将坐标映射到网格 (约 111 米精度)。

        参数:
            lat (float): 纬度
            lon (float): 经度

        返回:
            Tuple[int, int]: 网格键值 (lat_key, lon_key)
        """
        return (
            round(lat / self._cache_resolution),
            round(lon / self._cache_resolution)
        )

    def _batch_query_all(self, coords: np.ndarray) -> np.ndarray:
        """
        批量查询所有空间特征。

        查询内容:
            1. 道路类型特征（6维 one-hot）
            2. POI 邻近特征（4维）
            3. 道路密度特征（1维）

        参数:
            coords (np.ndarray): 形状 (N, 2) 的坐标数组，每行为 (lat, lon)

        返回:
            np.ndarray: 形状 (N, 11) 的特征矩阵
        """
        # 特征1: 道路类型 (6维)
        road_type_features = self._batch_query_road_types(coords)

        # 特征2: 附近 POI (4维)
        poi_features = self._batch_query_pois(coords)

        # 特征3: 道路密度 (1维)
        road_density = self._batch_query_road_density(coords)

        return np.concatenate([
            road_type_features,
            poi_features,
            road_density
        ], axis=1)

    def _batch_query_road_types(self, coords: np.ndarray,
                                 max_distance: float = 150.0) -> np.ndarray:
        """
        批量查询道路类型。

        查询逻辑:
            1. 使用 KDTree 查询每个坐标最近的道路节点
            2. 如果距离小于阈值，使用该道路类型
            3. 如果是 car 类道路且附近有 bus_stop POI，改为 bus
            4. 否则标记为 unknown
            5. 将道路类型转换为 one-hot 编码

        参数:
            coords (np.ndarray): 形状 (N, 2) 的坐标数组
            max_distance (float): 最大查询距离（米），默认 150.0

        返回:
            np.ndarray: 形状 (N, 7) 的 one-hot 编码矩阵
        """
        N = coords.shape[0]

        # 查询最近的道路节点
        distances, indices = self.road_kdtree.query(coords, k=1)
        distances = distances * 111300.0  # 转换为米

        # 先查POI：找附近bus_stop
        bus_stop_radius = 50.0 / 111300.0  # 50米内有bus_stop则认为是bus环境
        if self.poi_kdtree is not None:
            poi_indices = self.poi_kdtree.query_ball_point(coords, r=bus_stop_radius)
            has_bus_stop = np.array([
                any(self.poi_types[j] == 'bus_stop' for j in idx_list)
                for idx_list in poi_indices
            ])
        else:
            has_bus_stop = np.zeros(N, dtype=bool)

        # 构建 one-hot 编码
        type_names = ['walk', 'bike', 'car', 'bus', 'train', 'subway', 'unknown']
        road_type_features = np.zeros((N, 7), dtype=np.float32)

        for i in range(N):
            if distances[i] < max_distance:
                road_type = self.road_types[indices[i]]
                # 如果是car类道路且附近有bus_stop，改为bus
                if road_type == 'car' and has_bus_stop[i]:
                    road_type_features[i, 3] = 1.0  # bus
                elif road_type in type_names:
                    road_type_features[i, type_names.index(road_type)] = 1.0
                else:
                    road_type_features[i, 6] = 1.0
            else:
                road_type_features[i, 6] = 1.0  # unknown

        return road_type_features

    def _batch_query_pois(self, coords: np.ndarray,
                          max_distance: float = 300.0) -> np.ndarray:
        """
        批量查询 POI 信息。

        查询逻辑:
            1. 使用 KDTree 查询每个坐标附近的所有 POI
            2. 检查是否有公交站、地铁站、停车场
            3. 计算最近 POI 的距离并归一化

        参数:
            coords (np.ndarray): 形状 (N, 2) 的坐标数组
            max_distance (float): 最大查询距离（米），默认 200.0

        返回:
            np.ndarray: 形状 (N, 4) 的 POI 特征矩阵
                        [公交站标记, 地铁站标记, 停车场标记, 最近POI距离(归一化)]
        """
        N = coords.shape[0]
        poi_features = np.zeros((N, 4), dtype=np.float32)

        # 查询附近所有 POI
        max_degree = max_distance / 111300.0
        indices = self.poi_kdtree.query_ball_point(coords, r=max_degree)

        for i in range(N):
            if len(indices[i]) > 0:
                # 获取附近 POI 类型
                nearby_types = [self.poi_types[j] for j in indices[i]]

                # 特征编码
                if 'bus_stop' in nearby_types:
                    poi_features[i, 0] = 1.0
                if 'station' in nearby_types:
                    poi_features[i, 1] = 1.0
                if 'parking' in nearby_types:
                    poi_features[i, 2] = 1.0

                # 最近 POI 距离（修复 Decimal 问题）
                poi_coords_nearby = self.poi_coords[indices[i]]
                # 确保 coords 和 poi_coords_nearby 都是 float64
                dists = np.linalg.norm(
                    poi_coords_nearby.astype(np.float64) - coords[i:i+1].astype(np.float64),
                    axis=1
                ) * 111300.0
                poi_features[i, 3] = min(dists) / 200.0  # 归一化

        return poi_features

    def _batch_query_road_density(self, coords: np.ndarray,
                                   radius: float = 100.0) -> np.ndarray:
        """
        批量查询道路密度。

        查询逻辑:
            1. 使用 KDTree 查询每个坐标半径内的道路节点数量
            2. 将数量归一化到 [0, 1] 区间

        参数:
            coords (np.ndarray): 形状 (N, 2) 的坐标数组
            radius (float): 查询半径（米），默认 100.0

        返回:
            np.ndarray: 形状 (N, 1) 的道路密度矩阵
        """
        N = coords.shape[0]

        # 查询附近道路节点数量
        radius_degree = radius / 111300.0
        indices = self.road_kdtree.query_ball_point(coords, r=radius_degree)

        # 计算密度并归一化
        densities = np.array([len(idx) for idx in indices], dtype=np.float32)
        densities = np.clip(densities / 50.0, 0, 1)

        return densities.reshape(-1, 1)

    # ========== 缓存管理 ==========
    def get_cache_stats(self) -> Dict:
        """
        获取缓存统计信息。

        返回:
            Dict: 包含以下键的字典
                - cache_size: 缓存条目数
                - cache_hits: 缓存命中次数
                - cache_misses: 缓存未命中次数
                - hit_rate: 缓存命中率（百分比字符串）
                - cache_memory_mb: 缓存占用内存（MB）
        """
        total = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total if total > 0 else 0

        return {
            'cache_size': len(self._grid_cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': f"{hit_rate:.2%}",
            'cache_memory_mb': len(self._grid_cache) * 11 * 4 / (1024 * 1024)
        }

    def save_cache(self, cache_path: str):
        """
        保存缓存到文件。

        参数:
            cache_path (str): 缓存文件保存路径
        """
        with open(cache_path, 'wb') as f:
            pickle.dump(self._grid_cache, f)
        print(f" -> 缓存已保存到: {cache_path}")

    def load_cache(self, cache_path: str):
        """
        从文件加载缓存。

        参数:
            cache_path (str): 缓存文件路径
        """
        if os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self._grid_cache = pickle.load(f)
            print(f" -> 缓存已加载: {len(self._grid_cache)} 个网格")
        else:
            print(f" -> 缓存文件不存在: {cache_path}")

    def clear_cache(self):
        """
        清空缓存。

        清空内容:
            - 网格缓存字典
            - 缓存命中/未命中计数器
        """
        self._grid_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0

    # ========== 保留原有接口 ==========
    def _link_roads_to_pois(self, max_distance: float = 100.0):
        """
        使用 KDTree 加速 POI 到最近道路节点的链接。

        链接逻辑:
            1. 对每个 POI，使用 KDTree 查询附近的所有道路节点
            2. 计算每个道路节点到 POI 的 Haversine 距离
            3. 选择距离最近的道路节点建立连接
            4. 在图中添加双向边（POI->道路，道路->POI）

        参数:
            max_distance (float): 最大链接距离（米），默认 100.0
        """
        poi_nodes = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'poi']
        road_nodes_data = [(n, d) for n, d in self.graph.nodes(data=True) if d.get('type') == 'road_node']

        if not road_nodes_data:
            print("警告: 无道路节点，跳过关联。")
            return

        print(f" -> 正在关联 {len(poi_nodes)} 个POI到 {len(road_nodes_data)} 个道路节点...")

        road_coords = np.array([[d['latitude'], d['longitude']] for n, d in road_nodes_data])
        road_node_ids = [n for n, d in road_nodes_data]
        road_tree = KDTree(road_coords)

        poi_coords_list = []
        poi_ids_list = []
        for poi_node in poi_nodes:
            poi_data = self.graph.nodes[poi_node]
            poi_coords_list.append([poi_data['latitude'], poi_data['longitude']])
            poi_ids_list.append(poi_node)

        poi_coords = np.array(poi_coords_list)
        max_degree_distance = max_distance / 111300.0
        indices = road_tree.query_ball_point(poi_coords, r=max_degree_distance)

        link_count = 0
        for i, neighbors_indices in enumerate(tqdm(indices, desc="   [空间特征关联进度]", leave=False)):
            poi_node = poi_ids_list[i]
            poi_lat, poi_lon = poi_coords[i]

            min_dist = float('inf')
            nearest_road = None

            for j in neighbors_indices:
                road_node = road_node_ids[j]
                road_lat, road_lon = road_coords[j]
                # Haversine 距离计算（单位：米）
                dist = geodesic((poi_lat, poi_lon), (road_lat, road_lon)).meters

                if dist < min_dist:
                    min_dist = dist
                    nearest_road = road_node

            if nearest_road is not None:
                self.graph.add_edge(poi_node, nearest_road,
                                  type='nearby_road',
                                  distance=min_dist)
                self.graph.add_edge(nearest_road, poi_node,
                                  type='has_poi',
                                  distance=min_dist)
                link_count += 1

        print(f" -> 空间特征道路-POI关联完成。共添加 {link_count} 条关联边。")

    def get_graph_statistics(self) -> Dict:
        """
        获取空间特征提取器统计信息。

        返回:
            Dict: 包含以下键的字典
                - num_nodes: 图中节点总数
                - num_edges: 图中边总数
                - road_nodes: 道路节点数
                - poi_nodes: POI 节点数
                - poi_links: POI-道路关联边数
        """
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'road_nodes': len([n for n, d in self.graph.nodes(data=True)
                             if d.get('type') == 'road_node']),
            'poi_nodes': len([n for n, d in self.graph.nodes(data=True)
                            if d.get('type') == 'poi']),
            'poi_links': len([u for u, v, k, d in self.graph.edges(data=True, keys=True)
                              if d.get('type') == 'nearby_road'])
        }
