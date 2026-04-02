export interface TrajectoryPoint {
  lat: number
  lng: number
  timestamp: string
  speed?: number
}

export interface TrajectoryStats {
  distance: number
  duration: number
  avg_speed: number
  max_speed: number
}

// ========== 新增：分段预测类型 ==========

export interface SegmentPrediction {
  segment_id: number
  predicted_mode: string
  confidence: number
  start_index: number
  end_index: number
  start_time: string
  end_time: string
  duration: number      // 秒
  distance: number      // 米
  avg_speed: number     // m/s
  points: TrajectoryPoint[]
}

export interface ModeBreakdown {
  mode: string
  percentage: number    // 0-1
  distance: number      // 米
  duration: number      // 秒
}

// ========== 更新：TrajectoryPrediction 添加分段字段 ==========

export interface TrajectoryPrediction {
  trajectory_id: string
  predicted_mode: string
  confidence: number
  points: TrajectoryPoint[]
  stats: TrajectoryStats
  // 新增：分段预测结果
  segments?: SegmentPrediction[]
  mode_breakdown?: ModeBreakdown[]
}

// ========== 以下为原有类型，保持不变 ==========

export interface TransportMode {
  id: string
  name: string
  color: string
  icon: string
}

export interface ExperimentInfo {
  id: string
  name: string
  description: string
  features: string[]
  status: 'completed' | 'not_trained' | 'training'
}

export interface EvaluationReport {
  accuracy: number
  precision: Record<string, number>
  recall: Record<string, number>
  f1_score: Record<string, number>
  classification_report: Record<string, any>
}

export interface PredictionSummary {
  total_predictions: number
  mode_distribution: Record<string, number>
  accuracy: number
}

export interface DatasetStats {
  total_trajectories: number
  total_users: number
  mode_distribution: Record<string, number>
  avg_trajectory_length: number
  date_range: {
    start: string
    end: string
  }
  total_distance: string
}

export interface DataCleaningStep {
  name: string
  count: number
}

export interface DataCleaningStats {
  steps: DataCleaningStep[]
}

export interface TrainingProgress {
  task_id: string
  exp_name: string
  epoch: number
  total_epochs: number
  loss: number
  accuracy: number
  status: 'training' | 'completed' | 'failed' | 'cancelled'
}

export interface TrainingRequest {
  exp_name: string
  epochs?: number
  batch_size?: number
  learning_rate?: number
}

export interface TrainingResponse {
  task_id: string
  status: string
  message: string
}

// ========== 新增：预测选项类型 ==========

export interface PredictOptions {
  model: string
  enableSegmentation?: boolean
  windowSize?: number
  minSegmentDuration?: number
}

// ========== 新增：常量映射 ==========

export const MODE_COLORS: Record<string, string> = {
  walk: '#4A90E2',
  bike: '#52C41A',
  bus: '#FA8C16',
  car: '#F5222D',
  subway: '#722ED1',
  train: '#13C2C2',
}

export const MODE_NAMES: Record<string, string> = {
  walk: '步行',
  bike: '自行车',
  bus: '公交',
  car: '汽车/出租',
  subway: '地铁',
  train: '火车',
}

export const MODE_ICONS: Record<string, string> = {
  walk: '🚶',
  bike: '🚴',
  bus: '🚌',
  car: '🚗',
  subway: '🚇',
  train: '🚄',
}