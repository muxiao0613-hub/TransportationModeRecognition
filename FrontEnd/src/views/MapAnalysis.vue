<template>
  <div class="map-analysis">
    <div class="sidebar">
      <div class="sidebar-section">
        <h3>文件上传</h3>
        <TrajectoryUpload @upload="handleUpload" />
      </div>

      <div v-if="predictions.length > 0" class="sidebar-section">
        <div class="section-header">
          <h3>轨迹列表</h3>
          <el-button
            text
            size="small"
            type="danger"
            @click="handleClearAll"
          >
            清空全部
          </el-button>
        </div>
        <div class="trajectory-list">
          <div
            v-for="traj in predictions"
            :key="traj.trajectory_id"
            class="trajectory-item"
            :class="{ 
              active: selectedTrajectory?.trajectory_id === traj.trajectory_id,
              hidden: !visibleTrajectoryIds.has(traj.trajectory_id)
            }"
          >
            <div class="trajectory-item-main" @click="handleSelectTrajectory(traj)">
              <div class="trajectory-icon">
                <el-icon><Location /></el-icon>
              </div>
              <div class="trajectory-info">
                <div class="trajectory-name">{{ '轨迹 ' + traj.trajectory_id.slice(0, 8) }}</div>
                <div class="trajectory-meta">
                  <span>{{ formatDistance(traj.stats.distance) }}</span>
                  <span>•</span>
                  <span>{{ formatDuration(traj.stats.duration) }}</span>
                </div>
              </div>
              <ModeTag :mode="traj.predicted_mode" class="trajectory-mode" />
            </div>
            <div class="trajectory-actions">
              <el-button
                :icon="visibleTrajectoryIds.has(traj.trajectory_id) ? View : Hide"
                text
                size="small"
                :type="visibleTrajectoryIds.has(traj.trajectory_id) ? 'primary' : 'info'"
                @click.stop="handleToggleVisibility(traj.trajectory_id)"
                :title="visibleTrajectoryIds.has(traj.trajectory_id) ? '隐藏轨迹' : '显示轨迹'"
              />
              <el-button
                :icon="Delete"
                text
                size="small"
                type="danger"
                @click.stop="handleDeleteTrajectory(traj.trajectory_id)"
                title="删除轨迹"
              />
            </div>
          </div>
        </div>
      </div>

      <div class="sidebar-section">
        <h3>交通方式图例</h3>
        <div class="mode-filters">
          <div
            v-for="mode in transportModes"
            :key="mode.id"
            class="mode-filter-item"
            :class="{ active: modeFilters[mode.id] }"
            @click="toggleModeFilter(mode.id)"
          >
            <div
              class="mode-color-dot"
              :style="{ background: mode.color }"
            ></div>
            <span class="mode-name">{{ mode.name }}</span>
          </div>
        </div>
      </div>

      <div v-if="selectedTrajectory" class="sidebar-section">
        <h3>轨迹详情</h3>
        <div class="trajectory-details">
          <div class="detail-item">
            <span class="detail-label">识别结果：</span>
            <ModeTag :mode="selectedTrajectory.predicted_mode" />
          </div>
          <div class="detail-item">
            <span class="detail-label">置信度：</span>
            <el-progress
              :percentage="selectedTrajectory.confidence * 100"
              :color="getConfidenceColor(selectedTrajectory.confidence)"
              :stroke-width="12"
            />
          </div>
          <div class="detail-item">
            <span class="detail-label">总距离：</span>
            <span class="detail-value">{{ formatDistance(selectedTrajectory.stats.distance) }}</span>
          </div>
          <div class="detail-item">
            <span class="detail-label">总时长：</span>
            <span class="detail-value">{{ formatDuration(selectedTrajectory.stats.duration) }}</span>
          </div>
          <div class="detail-item">
            <span class="detail-label">平均速度：</span>
            <span class="detail-value">{{ formatSpeed(selectedTrajectory.stats.avg_speed) }}</span>
          </div>
          <div class="detail-item">
            <span class="detail-label">最高速度：</span>
            <span class="detail-value">{{ formatSpeed(selectedTrajectory.stats.max_speed) }}</span>
          </div>
          <div class="detail-item">
            <span class="detail-label">起止时间：</span>
            <span class="detail-value">{{ formatTimeRange(selectedTrajectory.points) }}</span>
          </div>
        </div>
      </div>

      <div v-if="selectedTrajectory && selectedTrajectory.segments && selectedTrajectory.segments.length > 1" class="sidebar-section">
        <SegmentTimeline
          :segments="selectedTrajectory.segments"
          :mode-breakdown="selectedTrajectory.mode_breakdown"
          :active-segment-id="activeSegmentId"
          @segment-click="handleSegmentClick"
          @segment-hover="handleSegmentHover"
        />
      </div>
    </div>

    <div class="map-container">
      <MapView
        :trajectories="filteredPredictions"
        :selected-trajectory="selectedTrajectory"
        :show-segments="true"
        @select="handleSelectTrajectory"
        @select-segment="handleSegmentClick"
      />
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Location, View, Hide, Delete } from '@element-plus/icons-vue'
import { useTrajectoryStore } from '@/stores/trajectory'
import MapView from '@/components/MapView.vue'
import TrajectoryUpload from '@/components/TrajectoryUpload.vue'
import ModeTag from '@/components/ModeTag.vue'
import SegmentTimeline from '@/components/SegmentTimeline.vue'
import type { SegmentPrediction, TrajectoryPrediction } from '@/types'

const trajectoryStore = useTrajectoryStore()

const transportModes = computed(() => trajectoryStore.transportModes)
const predictions = computed(() => trajectoryStore.predictions)
const selectedTrajectory = computed(() => trajectoryStore.selectedTrajectory)
const visibleTrajectoryIds = computed(() => trajectoryStore.visibleTrajectoryIds)
const filteredPredictions = computed(() => trajectoryStore.filteredPredictions)

const modeFilters = ref<Record<string, boolean>>({
  walk: true,
  bike: true,
  bus: true,
  car: true,
  subway: true,
  train: true,
})

const activeSegmentId = ref<number | null>(null)

onMounted(() => {
  trajectoryStore.loadTransportModes()
})

async function handleUpload(file: File, model: string) {
  try {
    await trajectoryStore.predictTrajectory(file, model, true)
    activeSegmentId.value = null
    ElMessage.success('预测完成！')
  } catch (error) {
    ElMessage.error('预测失败，请检查文件格式')
  }
}

function handleSelectTrajectory(trajectory: TrajectoryPrediction) {
  trajectoryStore.selectTrajectory(trajectory)
  activeSegmentId.value = null
}

function handleToggleVisibility(trajectoryId: string) {
  trajectoryStore.toggleTrajectoryVisibility(trajectoryId)
}

async function handleDeleteTrajectory(trajectoryId: string) {
  try {
    await ElMessageBox.confirm(
      '确定要删除这条轨迹吗？',
      '删除确认',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning',
      }
    )
    trajectoryStore.deleteTrajectory(trajectoryId)
    ElMessage.success('轨迹已删除')
  } catch {
  }
}

async function handleClearAll() {
  try {
    await ElMessageBox.confirm(
      '确定要清空所有轨迹吗？',
      '清空确认',
      {
        confirmButtonText: '确定',
        cancelButtonText: '取消',
        type: 'warning',
      }
    )
    trajectoryStore.clearPredictions()
    ElMessage.success('已清空所有轨迹')
  } catch {
  }
}

function handleSegmentClick(segment: SegmentPrediction) {
  activeSegmentId.value = segment.segment_id
}

function handleSegmentHover(segment: SegmentPrediction | null) {
}

function toggleModeFilter(mode: string) {
  modeFilters.value[mode] = !modeFilters.value[mode]
}

function getConfidenceColor(confidence: number): string {
  if (confidence >= 0.8) return '#67C23A'
  if (confidence >= 0.6) return '#E6A23C'
  return '#F56C6C'
}

function formatDistance(meters: number): string {
  if (meters < 1000) return `${meters.toFixed(0)} m`
  return `${(meters / 1000).toFixed(2)} km`
}

function formatDuration(seconds: number): string {
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = Math.floor(seconds % 60)
  
  if (hours > 0) return `${hours}h ${minutes}m ${secs}s`
  if (minutes > 0) return `${minutes}m ${secs}s`
  return `${secs}s`
}

function formatSpeed(mps: number): string {
  return `${mps.toFixed(2)} m/s`
}

function formatTimeRange(points: any[]): string {
  if (points.length === 0) return 'N/A'
  const start = new Date(points[0].timestamp)
  const end = new Date(points[points.length - 1].timestamp)
  
  const formatDate = (date: Date) => {
    return date.toLocaleString('zh-CN', {
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
    })
  }
  
  return `${formatDate(start)} - ${formatDate(end)}`
}
</script>

<style scoped>
.map-analysis {
  display: flex;
  flex: 1;
  min-height: 0;
  background: #0f1117;
}

.sidebar {
  width: 340px;
  background: #1a1f2e;
  border-right: 1px solid rgba(255, 255, 255, 0.08);
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 20px;
  padding: 20px;
}

.sidebar-section {
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
  padding-bottom: 20px;
}

.sidebar-section:last-child {
  border-bottom: none;
}

.section-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.section-header h3 {
  margin: 0;
  font-size: 16px;
  font-weight: 600;
  color: #fff;
}

.sidebar-section h3 {
  margin: 0 0 12px 0;
  font-size: 16px;
  font-weight: 600;
  color: #fff;
}

.trajectory-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.trajectory-item {
  background: rgba(255, 255, 255, 0.02);
  border: 1px solid rgba(255, 255, 255, 0.06);
  border-radius: 8px;
  overflow: hidden;
  transition: all 0.2s;
}

.trajectory-item:hover {
  background: rgba(255, 255, 255, 0.04);
  border-color: rgba(255, 255, 255, 0.1);
}

.trajectory-item.active {
  background: rgba(74, 144, 226, 0.1);
  border-color: rgba(74, 144, 226, 0.3);
}

.trajectory-item.hidden {
  opacity: 0.3;
  border-color: rgba(255, 255, 255, 0.03);
  background: rgba(255, 255, 255, 0.01);
}

.trajectory-item-main {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px;
  cursor: pointer;
}

.trajectory-icon {
  width: 36px;
  height: 36px;
  background: rgba(74, 144, 226, 0.15);
  border-radius: 8px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #4a90e2;
  flex-shrink: 0;
}

.trajectory-icon .el-icon {
  font-size: 18px;
}

.trajectory-info {
  flex: 1;
  min-width: 0;
}

.trajectory-name {
  font-size: 14px;
  font-weight: 500;
  color: #e5e8eb;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.trajectory-meta {
  font-size: 12px;
  color: #909399;
  margin-top: 2px;
  display: flex;
  gap: 6px;
}

.trajectory-mode {
  flex-shrink: 0;
}

.trajectory-actions {
  display: flex;
  gap: 4px;
  padding: 0 12px 12px 12px;
  border-top: 1px solid rgba(255, 255, 255, 0.04);
  margin-top: -1px;
}

.mode-filters {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.mode-filter-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s;
  background: rgba(255, 255, 255, 0.02);
}

.mode-filter-item:hover {
  background: rgba(255, 255, 255, 0.05);
}

.mode-filter-item.active {
  background: rgba(74, 144, 226, 0.15);
}

.mode-color-dot {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  flex-shrink: 0;
}

.mode-name {
  flex: 1;
  font-size: 14px;
  color: #e5e8eb;
}

.trajectory-details {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.detail-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 14px;
}

.detail-label {
  color: #909399;
}

.detail-value {
  font-weight: 600;
  color: #e5e8eb;
}

.map-container {
  flex: 1;
  position: relative;
}
</style>