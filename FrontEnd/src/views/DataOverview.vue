<template>
  <div class="data-overview">
    <div class="page-header">
      <h2>数据概览</h2>
      <p>GeoLife 轨迹数据集统计与特征说明</p>
    </div>

    <div class="stats-cards">
      <StatCard
        label="总轨迹数"
        :value="datasetStats.total_trajectories"
        icon="DataLine"
        icon-bg="#4A90E2"
      />
      <StatCard
        label="用户数"
        :value="datasetStats.total_users"
        icon="User"
        icon-bg="#52C41A"
      />
      <StatCard
        label="总里程（估算）"
        :value="datasetStats.total_distance"
        icon="Location"
        icon-bg="#FA8C16"
      />
      <StatCard
        label="时间跨度"
        :value="`${datasetStats.date_range.start.split('-')[0]}-${datasetStats.date_range.end.split('-')[0]}`"
        icon="Calendar"
        icon-bg="#722ED1"
      />
    </div>

    <div class="charts-row">
      <div class="chart-card">
        <h3>各交通方式样本数量</h3>
        <div ref="barChartRef" class="chart-container"></div>
      </div>

      <div class="chart-card">
        <h3>样本分布</h3>
        <div ref="pieChartRef" class="chart-container"></div>
      </div>
    </div>

    <div class="info-cards">
      <div class="info-card">
        <h3>数据清洗流程</h3>
        <div class="process-flow">
          <template v-for="(step, index) in cleaningStats.steps" :key="index">
            <div class="process-step">
              <div class="step-icon">
                <el-icon>
                  <component :is="stepIcons[index]" />
                </el-icon>
              </div>
              <div class="step-content">
                <h4>{{ step.name }}</h4>
                <p>{{ stepDescriptions[index] }}</p>
                <div class="step-count">{{ step.name }}: {{ step.count.toLocaleString() }}</div>
              </div>
            </div>
            <div v-if="index < cleaningStats.steps.length - 1" class="process-arrow">
              <el-icon><ArrowRight /></el-icon>
            </div>
          </template>
        </div>
      </div>

      <div class="info-card">
        <h3>特征维度说明</h3>
        <div class="feature-sections">
          <div class="feature-section">
            <h4>9维轨迹特征</h4>
            <ul class="feature-list">
              <li>平均速度 (avg_speed)</li>
              <li>最大速度 (max_speed)</li>
              <li>速度标准差 (speed_std)</li>
              <li>平均加速度 (avg_acceleration)</li>
              <li>最大加速度 (max_acceleration)</li>
              <li>加速度标准差 (acceleration_std)</li>
              <li>平均转向率 (avg_turning_rate)</li>
              <li>最大转向率 (max_turning_rate)</li>
              <li>转向率标准差 (turning_rate_std)</li>
            </ul>
          </div>

          <div class="feature-section">
            <h4>18维段级统计特征</h4>
            <ul class="feature-list">
              <li>段距离统计 (mean, std, max, min)</li>
              <li>段时间统计 (mean, std, max, min)</li>
              <li>段速度统计 (mean, std, max, min)</li>
              <li>段加速度统计 (mean, std, max, min)</li>
              <li>段转向率统计 (mean, std, max, min)</li>
              <li>段数 (segment_count)</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
defineOptions({
  name: 'DataOverview'
})

import { ref, onMounted, onUnmounted, onActivated, onDeactivated } from 'vue'
import { datasetApi } from '@/api/dataset'
import StatCard from '@/components/StatCard.vue'
import * as echarts from 'echarts'
import type { ECharts } from 'echarts'
import { Document, Filter, Brush, ScaleToOriginal, ArrowRight } from '@element-plus/icons-vue'
import type { DataCleaningStats, DatasetStats } from '@/types'

let cachedDatasetStats: DatasetStats | null = null
let cachedCleaningStats: DataCleaningStats | null = null
let isDataLoaded = false

const datasetStats = ref<DatasetStats>({
  total_trajectories: 0,
  total_users: 0,
  avg_trajectory_length: 0,
  mode_distribution: {} as Record<string, number>,
  date_range: { start: '', end: '' },
  total_distance: ''
})

const cleaningStats = ref<DataCleaningStats>({
  steps: []
})

const stepIcons = [Document, Filter, Brush, ScaleToOriginal]
const stepDescriptions = [
  '182用户，17,621轨迹段',
  '移除低质量轨迹，GPS点数<10',
  '异常点检测、轨迹平滑、去噪',
  '特征标准化、数据归一化'
]

const barChartRef = ref<HTMLElement>()
const pieChartRef = ref<HTMLElement>()
let barChart: ECharts | null = null
let pieChart: ECharts | null = null

async function loadData() {
  if (isDataLoaded && cachedDatasetStats && cachedCleaningStats) {
    datasetStats.value = cachedDatasetStats
    cleaningStats.value = cachedCleaningStats
    return
  }

  await Promise.all([
    loadDatasetStats(),
    loadCleaningStats()
  ])
  
  cachedDatasetStats = datasetStats.value
  cachedCleaningStats = cleaningStats.value
  isDataLoaded = true
}

async function loadDatasetStats() {
  if (cachedDatasetStats) {
    datasetStats.value = cachedDatasetStats
    return
  }
  
  try {
    const stats = await datasetApi.getStats()
    datasetStats.value = stats
    cachedDatasetStats = stats
  } catch (error) {
    console.error('加载数据集统计失败:', error)
  }
}

async function loadCleaningStats() {
  if (cachedCleaningStats) {
    cleaningStats.value = cachedCleaningStats
    return
  }
  
  try {
    const stats = await datasetApi.getCleaningStats()
    cleaningStats.value = stats
    cachedCleaningStats = stats
  } catch (error) {
    console.error('加载数据清洗统计失败:', error)
  }
}

function initCharts() {
  initBarChart()
  initPieChart()
}

function initBarChart() {
  if (!barChartRef.value) return
  
  if (barChart) {
    barChart.resize()
    return
  }

  barChart = echarts.init(barChartRef.value)

  const modes = Object.keys(datasetStats.value.mode_distribution)
  const counts = modes.map(m => datasetStats.value.mode_distribution[m])

  const modeNames: Record<string, string> = {
    walk: '步行',
    bike: '自行车',
    bus: '公交',
    car: '汽车/出租',
    subway: '地铁',
    train: '火车',
  }

  const colors = ['#4A90E2', '#52C41A', '#FA8C16', '#F5222D', '#722ED1', '#13C2C2']

  const option = {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow',
      },
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true,
    },
    xAxis: {
      type: 'value',
      name: '样本数',
      nameLocation: 'middle',
      nameGap: 30,
    },
    yAxis: {
      type: 'category',
      data: modes.map(m => modeNames[m] || m),
    },
    series: [
      {
        type: 'bar',
        data: counts,
        itemStyle: {
          color: (params: any) => colors[params.dataIndex % colors.length],
        },
        label: {
          show: true,
          position: 'right',
        },
      },
    ],
  }

  barChart.setOption(option)
}

function initPieChart() {
  if (!pieChartRef.value) return
  
  if (pieChart) {
    pieChart.resize()
    return
  }

  pieChart = echarts.init(pieChartRef.value)

  const modes = Object.keys(datasetStats.value.mode_distribution)
  const counts = modes.map(m => datasetStats.value.mode_distribution[m])

  const modeNames: Record<string, string> = {
    walk: '步行',
    bike: '自行车',
    bus: '公交',
    car: '汽车/出租',
    subway: '地铁',
    train: '火车',
  }

  const colors = ['#4A90E2', '#52C41A', '#FA8C16', '#F5222D', '#722ED1', '#13C2C2']

  const option = {
    tooltip: {
      trigger: 'item',
      formatter: '{b}: {c} ({d}%)',
    },
    legend: {
      orient: 'vertical',
      left: 'left',
      data: modes.map(m => modeNames[m] || m),
    },
    series: [
      {
        type: 'pie',
        radius: '60%',
        data: modes.map((m, i) => ({
          name: modeNames[m] || m,
          value: counts[i],
          itemStyle: {
            color: colors[i % colors.length],
          },
        })),
        emphasis: {
          itemStyle: {
            shadowBlur: 10,
            shadowOffsetX: 0,
            shadowColor: 'rgba(0, 0, 0, 0.5)',
          },
        },
        label: {
          formatter: '{b}\n{d}%',
        },
      },
    ],
  }

  pieChart.setOption(option)
}

function handleResize() {
  barChart?.resize()
  pieChart?.resize()
}

onMounted(async () => {
  await loadData()
  initCharts()
  window.addEventListener('resize', handleResize)
})

onActivated(() => {
  handleResize()
})

onUnmounted(() => {
  window.removeEventListener('resize', handleResize)
  barChart?.dispose()
  pieChart?.dispose()
  barChart = null
  pieChart = null
})
</script>

<style scoped>
.data-overview {
  padding: 24px;
  background: #0f1117;
  flex: 1;
  min-height: 0;
  overflow-y: auto;
}

.page-header {
  margin-bottom: 32px;
  text-align: center;
}

.page-header h2 {
  margin: 0 0 12px 0;
  font-size: 28px;
  color: #fff;
}

.page-header p {
  margin: 0;
  font-size: 14px;
  color: #909399;
}

.stats-cards {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 24px;
  margin-bottom: 32px;
}

.charts-row {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
  margin-bottom: 32px;
}

.chart-card {
  background: #1a1f2e;
  border-radius: 8px;
  padding: 24px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255, 255, 255, 0.08);
}

.chart-card h3 {
  margin: 0 0 20px 0;
  font-size: 18px;
  font-weight: 600;
  color: #fff;
}

.chart-container {
  width: 100%;
  height: 400px;
}

.info-cards {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}

.info-card {
  background: #1a1f2e;
  border-radius: 8px;
  padding: 24px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255, 255, 255, 0.08);
}

.info-card h3 {
  margin: 0 0 24px 0;
  font-size: 18px;
  font-weight: 600;
  color: #fff;
}

.process-flow {
  display: flex;
  align-items: flex-start;
  justify-content: center;
  margin-bottom: 24px;
  padding: 20px 0;
  gap: 8px;
}

.process-arrow {
  display: flex;
  align-items: center;
  justify-content: center;
  color: #909399;
  font-size: 20px;
  padding: 28px 8px 0;
  flex-shrink: 0;
}

.process-step {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
  min-width: 0;
}

.step-icon {
  width: 56px;
  height: 56px;
  border-radius: 12px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  font-size: 24px;
  margin-bottom: 12px;
  flex-shrink: 0;
}

.step-content {
  width: 100%;
  min-width: 0;
}

.step-content h4 {
  margin: 0 0 8px 0;
  font-size: 14px;
  font-weight: 600;
  color: #fff;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.step-content p {
  margin: 0 0 8px 0;
  font-size: 11px;
  color: #909399;
  line-height: 1.4;
  word-break: break-word;
}

.step-count {
  font-size: 12px;
  color: #4A90E2;
  font-weight: 600;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.cleaning-stats {
  display: flex;
  gap: 32px;
  padding-top: 16px;
  border-top: 1px solid rgba(255, 255, 255, 0.08);
}

.stat-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.stat-label {
  font-size: 14px;
  color: #e5e8eb;
}

.stat-value {
  font-size: 18px;
  font-weight: 600;
  color: #4A90E2;
}

.feature-sections {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}

.feature-section h4 {
  margin: 0 0 16px 0;
  font-size: 14px;
  font-weight: 600;
  color: #fff;
  padding-bottom: 8px;
  border-bottom: 2px solid #4A90E2;
}

.feature-list {
  margin: 0;
  padding: 0;
  list-style: none;
}

.feature-list li {
  padding: 6px 0;
  font-size: 13px;
  color: #e5e8eb;
  position: relative;
  padding-left: 16px;
}

.feature-list li::before {
  content: '•';
  position: absolute;
  left: 0;
  color: #4A90E2;
  font-weight: bold;
}
</style>
