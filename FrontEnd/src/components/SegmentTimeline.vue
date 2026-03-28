<template>
  <div class="segment-timeline">
    <!-- 标题栏 -->
    <div class="timeline-header">
      <span class="title">行程分段</span>
      <span class="segment-count">共 {{ segments.length }} 段</span>
    </div>
    
    <!-- 时间轴条 -->
    <div class="timeline-bar">
      <div
        v-for="(segment, index) in segments"
        :key="segment.segment_id"
        class="segment-block"
        :class="{ active: activeSegmentId === segment.segment_id }"
        :style="{
          width: getSegmentWidth(segment) + '%',
          backgroundColor: getModeColor(segment.predicted_mode),
        }"
        @click="handleSegmentClick(segment)"
        @mouseenter="handleSegmentHover(segment)"
        @mouseleave="handleSegmentLeave"
      >
        <span class="segment-icon">{{ getModeIcon(segment.predicted_mode) }}</span>
      </div>
    </div>
    
    <!-- 时间标签 -->
    <div class="time-labels">
      <span class="start-time">{{ formatTime(segments[0]?.start_time) }}</span>
      <span class="end-time">{{ formatTime(segments[segments.length - 1]?.end_time) }}</span>
    </div>
    
    <!-- 分段详情列表 -->
    <div class="segment-list">
      <div
        v-for="(segment, index) in segments"
        :key="segment.segment_id"
        class="segment-item"
        :class="{ active: activeSegmentId === segment.segment_id }"
        @click="handleSegmentClick(segment)"
      >
        <div class="segment-indicator" :style="{ backgroundColor: getModeColor(segment.predicted_mode) }"></div>
        <div class="segment-info">
          <div class="segment-mode">
            <span class="icon">{{ getModeIcon(segment.predicted_mode) }}</span>
            <span class="name">{{ getModeName(segment.predicted_mode) }}</span>
            <span class="confidence">({{ (segment.confidence * 100).toFixed(0) }}%)</span>
          </div>
          <div class="segment-stats">
            <span class="stat">
              <i class="icon-distance"></i>
              {{ formatDistance(segment.distance) }}
            </span>
            <span class="stat">
              <i class="icon-time"></i>
              {{ formatDuration(segment.duration) }}
            </span>
            <span class="stat">
              <i class="icon-speed"></i>
              {{ segment.avg_speed.toFixed(1) }} m/s
            </span>
          </div>
        </div>
        <div class="segment-arrow" v-if="index < segments.length - 1">→</div>
      </div>
    </div>
    
    <!-- 交通方式占比统计 -->
    <div class="mode-breakdown" v-if="modeBreakdown && modeBreakdown.length > 0">
      <div class="breakdown-title">交通方式占比</div>
      <div class="breakdown-bars">
        <div
          v-for="item in modeBreakdown"
          :key="item.mode"
          class="breakdown-item"
        >
          <div class="breakdown-label">
            <span class="icon">{{ getModeIcon(item.mode) }}</span>
            <span class="name">{{ getModeName(item.mode) }}</span>
          </div>
          <div class="breakdown-bar-container">
            <div
              class="breakdown-bar"
              :style="{
                width: (item.percentage * 100) + '%',
                backgroundColor: getModeColor(item.mode),
              }"
            ></div>
          </div>
          <span class="breakdown-value">{{ (item.percentage * 100).toFixed(1) }}%</span>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import type { SegmentPrediction, ModeBreakdown } from '@/types'

interface Props {
  segments: SegmentPrediction[]
  modeBreakdown?: ModeBreakdown[]
  activeSegmentId?: number | null
}

const props = withDefaults(defineProps<Props>(), {
  activeSegmentId: null,
})

const emit = defineEmits<{
  segmentClick: [segment: SegmentPrediction]
  segmentHover: [segment: SegmentPrediction | null]
}>()

// 交通方式配置
const MODE_COLORS: Record<string, string> = {
  walk: '#4A90E2',
  bike: '#52C41A',
  bus: '#FA8C16',
  car: '#F5222D',
  subway: '#722ED1',
  train: '#13C2C2',
}

const MODE_NAMES: Record<string, string> = {
  walk: '步行',
  bike: '自行车',
  bus: '公交',
  car: '汽车/出租',
  subway: '地铁',
  train: '火车',
}

const MODE_ICONS: Record<string, string> = {
  walk: '🚶',
  bike: '🚴',
  bus: '🚌',
  car: '🚗',
  subway: '🚇',
  train: '🚄',
}

// 计算总距离
const totalDistance = computed(() => {
  return props.segments.reduce((sum, seg) => sum + seg.distance, 0)
})

// 获取段宽度百分比（基于距离）
function getSegmentWidth(segment: SegmentPrediction): number {
  if (totalDistance.value === 0) return 100 / props.segments.length
  return (segment.distance / totalDistance.value) * 100
}

function getModeColor(mode?: string): string {
  if (!mode) return '#666'
  const parts = mode.split(' ')
  const cleanMode = (parts[0] || '').toLowerCase()
  return MODE_COLORS[cleanMode] || '#666'
}

function getModeName(mode?: string): string {
  if (!mode) return '未知'
  const parts = mode.split(' ')
  const cleanMode = (parts[0] || '').toLowerCase()
  return MODE_NAMES[cleanMode] || mode
}

function getModeIcon(mode?: string): string {
  if (!mode) return '❓'
  const parts = mode.split(' ')
  const cleanMode = (parts[0] || '').toLowerCase()
  return MODE_ICONS[cleanMode] || '❓'
}

function formatTime(isoString?: string): string {
  if (!isoString) return '--:--'
  try {
    const date = new Date(isoString)
    return date.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })
  } catch {
    return '--:--'
  }
}

function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${Math.round(seconds)}秒`
  } else if (seconds < 3600) {
    return `${Math.round(seconds / 60)}分钟`
  } else {
    const hours = Math.floor(seconds / 3600)
    const mins = Math.round((seconds % 3600) / 60)
    return `${hours}时${mins}分`
  }
}

function formatDistance(meters: number): string {
  if (meters < 1000) {
    return `${Math.round(meters)}米`
  } else {
    return `${(meters / 1000).toFixed(2)}公里`
  }
}

function handleSegmentClick(segment: SegmentPrediction) {
  emit('segmentClick', segment)
}

function handleSegmentHover(segment: SegmentPrediction) {
  emit('segmentHover', segment)
}

function handleSegmentLeave() {
  emit('segmentHover', null)
}
</script>

<style scoped>
.segment-timeline {
  background: #fff;
  border-radius: 8px;
  padding: 16px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.timeline-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.timeline-header .title {
  font-weight: 600;
  font-size: 14px;
  color: #333;
}

.timeline-header .segment-count {
  font-size: 12px;
  color: #999;
}

/* 时间轴条 */
.timeline-bar {
  display: flex;
  height: 32px;
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 8px;
}

.segment-block {
  display: flex;
  align-items: center;
  justify-content: center;
  min-width: 24px;
  cursor: pointer;
  transition: all 0.2s ease;
  position: relative;
}

.segment-block:hover {
  filter: brightness(1.1);
  transform: scaleY(1.1);
}

.segment-block.active {
  transform: scaleY(1.15);
  box-shadow: 0 0 0 2px #fff, 0 0 0 4px currentColor;
  z-index: 1;
}

.segment-icon {
  font-size: 14px;
}

/* 时间标签 */
.time-labels {
  display: flex;
  justify-content: space-between;
  font-size: 11px;
  color: #999;
  margin-bottom: 16px;
}

/* 分段列表 */
.segment-list {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-bottom: 16px;
}

.segment-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  background: #f5f5f5;
  border-radius: 6px;
  cursor: pointer;
  transition: all 0.2s ease;
  flex: 1;
  min-width: 180px;
}

.segment-item:hover {
  background: #e8e8e8;
}

.segment-item.active {
  background: #e6f7ff;
  border: 1px solid #1890ff;
}

.segment-indicator {
  width: 4px;
  height: 100%;
  min-height: 36px;
  border-radius: 2px;
}

.segment-info {
  flex: 1;
}

.segment-mode {
  display: flex;
  align-items: center;
  gap: 4px;
  margin-bottom: 4px;
}

.segment-mode .icon {
  font-size: 14px;
}

.segment-mode .name {
  font-weight: 500;
  font-size: 13px;
  color: #333;
}

.segment-mode .confidence {
  font-size: 11px;
  color: #999;
}

.segment-stats {
  display: flex;
  gap: 12px;
  font-size: 11px;
  color: #666;
}

.segment-stats .stat {
  display: flex;
  align-items: center;
  gap: 2px;
}

.segment-arrow {
  color: #ccc;
  font-size: 16px;
}

/* 交通方式占比 */
.mode-breakdown {
  border-top: 1px solid #f0f0f0;
  padding-top: 12px;
}

.breakdown-title {
  font-size: 12px;
  color: #666;
  margin-bottom: 8px;
}

.breakdown-bars {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.breakdown-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.breakdown-label {
  display: flex;
  align-items: center;
  gap: 4px;
  min-width: 80px;
  font-size: 12px;
}

.breakdown-label .icon {
  font-size: 14px;
}

.breakdown-label .name {
  color: #333;
}

.breakdown-bar-container {
  flex: 1;
  height: 8px;
  background: #f0f0f0;
  border-radius: 4px;
  overflow: hidden;
}

.breakdown-bar {
  height: 100%;
  border-radius: 4px;
  transition: width 0.3s ease;
}

.breakdown-value {
  min-width: 45px;
  text-align: right;
  font-size: 12px;
  color: #666;
}
</style>