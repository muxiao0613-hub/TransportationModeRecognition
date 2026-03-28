<template>
  <div class="map-view" ref="mapContainer"></div>
</template>

<script setup lang="ts">
import { ref, onMounted, watch } from 'vue'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import type { TrajectoryPoint, TrajectoryPrediction, SegmentPrediction } from '@/types'

interface Props {
  trajectories: TrajectoryPrediction[]
  selectedTrajectory: TrajectoryPrediction | null
  showSpeedHeatmap?: boolean
  showSegments?: boolean  // 新增：是否显示分段
}

const props = withDefaults(defineProps<Props>(), {
  showSpeedHeatmap: false,
  showSegments: true,  // 默认显示分段
})

const emit = defineEmits<{
  select: [trajectory: TrajectoryPrediction]
  selectSegment: [segment: SegmentPrediction]  // 新增：选中某段
}>()

const mapContainer = ref<HTMLElement>()
let map: L.Map | null = null
let trajectoryLayers: L.LayerGroup[] = []
let markers: L.Marker[] = []
let segmentLayers: Map<number, L.Polyline[]> = new Map()

// 交通方式颜色映射
const MODE_COLORS: Record<string, string> = {
  walk: '#4A90E2',
  bike: '#52C41A',
  bus: '#FA8C16',
  car: '#F5222D',
  subway: '#722ED1',
  train: '#13C2C2',
}

// 交通方式名称映射
const MODE_NAMES: Record<string, string> = {
  walk: '步行',
  bike: '自行车',
  bus: '公交',
  car: '汽车/出租',
  subway: '地铁',
  train: '火车',
}

onMounted(() => {
  if (!mapContainer.value) return

  map = L.map(mapContainer.value).setView([39.9042, 116.4074], 12)

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    attribution: '© OpenStreetMap contributors',
    maxZoom: 19,
  }).addTo(map)
})

watch(() => props.trajectories, (newTrajectories) => {
  updateMap(newTrajectories)
}, { deep: true })

watch(() => props.selectedTrajectory, (selected) => {
  highlightTrajectory(selected)
})

watch(() => props.showSegments, () => {
  updateMap(props.trajectories)
})

/**
 * 更新地图 - 支持分段渲染
 */
function updateMap(trajectories: TrajectoryPrediction[]) {
  if (!map) return

  clearMap()

  trajectories.forEach((traj, trajIndex) => {
    const layerGroup = L.layerGroup().addTo(map!)
    
    // 判断是否有分段数据
    const hasSegments = traj.segments && traj.segments.length > 1 && props.showSegments
    
    if (hasSegments) {
      // ========== 多段渲染 ==========
      const segmentPolylines: L.Polyline[] = []
      const segments = traj.segments || []
      
      segments.forEach((segment, segIndex) => {
        const points = segment.points.map(p => [p.lat, p.lng] as [number, number])
        const color = getModeColor(segment.predicted_mode)
        
        // 绘制该段轨迹
        const polyline = L.polyline(points, {
          color,
          weight: 4,
          opacity: 0.85,
        }).addTo(layerGroup)
        
        // 点击事件
        polyline.on('click', () => {
          emit('selectSegment', segment)
        })
        
        // 悬停显示段信息
        polyline.on('mouseover', (e) => {
          const popup = L.popup()
            .setLatLng(e.latlng)
            .setContent(createSegmentPopup(segment))
            .openOn(map!)
        })
        
        polyline.on('mouseout', () => {
          map?.closePopup()
        })
        
        segmentPolylines.push(polyline)
        
        // 在段起点添加小标记（除了第一段）
        if (segIndex > 0 && points.length > 0 && segIndex - 1 < segments.length) {
          const prevSegment = segments[segIndex - 1]
          const firstPoint = points[0] as [number, number]
          const transitionMarker = L.circleMarker(firstPoint, {
            radius: 6,
            fillColor: color,
            color: '#fff',
            weight: 2,
            fillOpacity: 0.9,
          }).addTo(layerGroup)
          
          transitionMarker.bindTooltip(
            `切换: ${getModeName(prevSegment?.predicted_mode)} → ${getModeName(segment.predicted_mode)}`,
            { permanent: false, direction: 'top' }
          )
        }
      })
      
      segmentLayers.set(trajIndex, segmentPolylines)
      
    } else {
      // ========== 单段渲染（原逻辑） ==========
      const points = traj.points.map(p => [p.lat, p.lng] as [number, number])
      const color = getModeColor(traj.predicted_mode)

      const polyline = L.polyline(points, {
        color,
        weight: 3,
        opacity: 0.8,
      }).addTo(layerGroup)

      polyline.on('click', () => {
        emit('select', traj)
      })

      polyline.on('mouseover', (e) => {
        const point = e.latlng
        const nearestPoint = findNearestPoint(traj, point)
        
        if (nearestPoint) {
          const popup = L.popup()
            .setLatLng(point)
            .setContent(`
              <div>
                <strong>${getModeName(traj.predicted_mode)}</strong><br>
                速度: ${nearestPoint.speed?.toFixed(2) || 'N/A'} m/s<br>
                置信度: ${(traj.confidence * 100).toFixed(1)}%
              </div>
            `)
            .openOn(map!)
        }
      })

      polyline.on('mouseout', () => {
        map?.closePopup()
      })
    }
    
    trajectoryLayers.push(layerGroup)

    // 添加起点和终点标记
    if (traj.points.length > 0) {
      const startPoint = traj.points[0] as TrajectoryPoint
      const endPoint = traj.points[traj.points.length - 1] as TrajectoryPoint
      
      // 起点：绿色圆形
      const startMarker = L.circleMarker([startPoint.lat, startPoint.lng], {
        radius: 10,
        fillColor: '#52C41A',
        color: '#fff',
        weight: 3,
        fillOpacity: 1,
      }).addTo(map!)
      startMarker.bindTooltip('起点', { permanent: false, direction: 'top' })

      // 终点：红色方形
      const endMarker = L.marker([endPoint.lat, endPoint.lng], {
        icon: L.divIcon({
          className: 'end-marker',
          html: '<div style="width: 16px; height: 16px; background: #F5222D; border: 2px solid #fff; border-radius: 2px;"></div>',
          iconSize: [16, 16],
          iconAnchor: [8, 8],
        }),
      }).addTo(map!)
      endMarker.bindTooltip('终点', { permanent: false, direction: 'top' })

      markers.push(startMarker as any, endMarker)
    }
  })

  // 调整视图
  if (trajectories.length > 0) {
    const allPoints = trajectories.flatMap(t => t.points.map(p => [p.lat, p.lng] as [number, number]))
    if (allPoints.length > 0) {
      const bounds = L.latLngBounds(allPoints)
      map.fitBounds(bounds, { padding: [50, 50] })
    }
  }
}

/**
 * 创建分段弹窗内容
 */
function createSegmentPopup(segment: SegmentPrediction): string {
  const duration = formatDuration(segment.duration)
  const distance = formatDistance(segment.distance)
  
  return `
    <div class="segment-popup">
      <div style="font-weight: bold; color: ${getModeColor(segment.predicted_mode)}; margin-bottom: 4px;">
        ${getModeName(segment.predicted_mode)}
      </div>
      <div style="font-size: 12px; color: #666;">
        <div>📏 距离: ${distance}</div>
        <div>⏱️ 时长: ${duration}</div>
        <div>🚀 平均速度: ${segment.avg_speed.toFixed(1)} m/s</div>
        <div>📊 置信度: ${(segment.confidence * 100).toFixed(1)}%</div>
      </div>
    </div>
  `
}

/**
 * 格式化时长
 */
function formatDuration(seconds: number): string {
  if (seconds < 60) {
    return `${Math.round(seconds)}秒`
  } else if (seconds < 3600) {
    return `${Math.round(seconds / 60)}分钟`
  } else {
    const hours = Math.floor(seconds / 3600)
    const mins = Math.round((seconds % 3600) / 60)
    return `${hours}小时${mins}分钟`
  }
}

/**
 * 格式化距离
 */
function formatDistance(meters: number): string {
  if (meters < 1000) {
    return `${Math.round(meters)}米`
  } else {
    return `${(meters / 1000).toFixed(2)}公里`
  }
}

function clearMap() {
  trajectoryLayers.forEach(layer => map?.removeLayer(layer))
  markers.forEach(marker => map?.removeLayer(marker))
  trajectoryLayers = []
  markers = []
  segmentLayers.clear()
}

function highlightTrajectory(selected: TrajectoryPrediction | null) {
  const selectedIndex = selected ? props.trajectories.indexOf(selected) : -1
  
  trajectoryLayers.forEach((layerGroup, index) => {
    layerGroup.eachLayer((layer) => {
      if (layer instanceof L.Polyline) {
        if (selectedIndex === index) {
          layer.setStyle({ weight: 6, opacity: 1 })
        } else {
          layer.setStyle({ weight: 4, opacity: 0.7 })
        }
      }
    })
  })
}

function findNearestPoint(trajectory: TrajectoryPrediction, point: L.LatLng): TrajectoryPoint | null {
  let nearest = null
  let minDist = Infinity

  trajectory.points.forEach(p => {
    const dist = Math.sqrt(
      Math.pow(p.lat - point.lat, 2) + Math.pow(p.lng - point.lng, 2)
    )
    if (dist < minDist) {
      minDist = dist
      nearest = p
    }
  })

  return nearest
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

// 暴露方法供父组件调用
defineExpose({
  fitToTrajectory: (traj: TrajectoryPrediction) => {
    if (!map || !traj.points.length) return
    const points = traj.points.map(p => [p.lat, p.lng] as [number, number])
    map.fitBounds(L.latLngBounds(points), { padding: [50, 50] })
  },
  highlightSegment: (segmentId: number, trajIndex: number = 0) => {
    const polylines = segmentLayers.get(trajIndex)
    if (!polylines) return
    
    polylines.forEach((line, idx) => {
      if (idx === segmentId) {
        line.setStyle({ weight: 8, opacity: 1 })
        line.bringToFront()
      } else {
        line.setStyle({ weight: 4, opacity: 0.6 })
      }
    })
  }
})
</script>

<style scoped>
.map-view {
  width: 100%;
  height: 100%;
  min-height: 600px;
}

:deep(.segment-popup) {
  min-width: 150px;
}

:deep(.end-marker) {
  background: transparent;
}
</style>