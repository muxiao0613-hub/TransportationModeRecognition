<template>
  <div class="model-comparison">
    <div class="experiments-header">
      <h2>实验对比</h2>
      <p>对比不同特征组合对交通方式识别性能的影响</p>
    </div>

    <div class="experiments-cards">
      <div
        v-for="exp in experiments"
        :key="exp.id"
        class="experiment-card"
        :class="{ active: selectedExperiment?.id === exp.id, completed: exp.status === 'completed' }"
        @click="selectExperiment(exp)"
      >
        <div class="card-header">
          <h3>{{ exp.name }}</h3>
          <el-tag :type="getStatusType(exp.status)" size="small">
            {{ getStatusText(exp.status) }}
          </el-tag>
        </div>

        <div class="card-body">
          <p class="description">{{ exp.description }}</p>
          <div class="features">
            <span
              v-for="(feature, index) in exp.features"
              :key="index"
              class="feature-tag"
            >
              {{ feature }}
            </span>
          </div>
        </div>

        <div v-if="exp.status === 'completed'" class="card-footer">
          <el-button type="primary" @click="showDetails(exp)">
            查看详情
          </el-button>
        </div>
      </div>
    </div>

    <div class="comparison-section">
      <h3>准确率对比</h3>
      <div ref="accuracyChartRef" class="chart-container"></div>
    </div>

    <div class="comparison-section">
      <h3>各类别 F1 分数对比</h3>
      <div class="radar-selector">
        <el-checkbox-group v-model="selectedExpsForRadar" @change="updateRadarChart">
          <el-checkbox-button v-for="exp in experiments" :key="exp.id" :label="exp.id">
            {{ exp.name }}
          </el-checkbox-button>
        </el-checkbox-group>
      </div>
      <div ref="radarChartRef" class="chart-container"></div>
    </div>

    <div class="comparison-section">
      <h3>各交通方式在不同实验中的 F1 分数</h3>
      <div class="mode-selector">
        <el-checkbox-group v-model="selectedModesForBar" @change="updateModeBarChart">
          <el-checkbox-button v-for="mode in modeList" :key="mode.key" :label="mode.key">
            {{ mode.label }}
          </el-checkbox-button>
        </el-checkbox-group>
      </div>
      <div ref="modeBarChartRef" class="chart-container"></div>
    </div>

    <div class="comparison-section">
      <h3>实验指标对比</h3>
      <div ref="metricsChartRef" class="chart-container"></div>
    </div>

    <div class="comparison-section">
      <h3>实验设置说明</h3>
      <el-alert type="info" :closable="false" style="margin-bottom: 16px">
        <template #title>
          <strong>数据划分与训练策略</strong>
        </template>
        <ul style="margin-top: 12px; padding-left: 20px;">
          <li>训练/验证/测试集划分：70% / 10% / 20%</li>
          <li>随机种子：random_state = 42</li>
          <li>分层采样：stratify 保证类别分布一致</li>
          <li>最优模型保存标准：验证集损失最小</li>
        </ul>
      </el-alert>
      <el-table :data="comparisonData" border stripe>
        <el-table-column prop="feature" label="配置项" width="200" fixed />
        <el-table-column prop="exp1" label="Exp1 (基线)" width="180" />
        <el-table-column prop="exp2" label="Exp2 (+OSM)" width="180" />
        <el-table-column prop="exp3" label="Exp3 (+天气)" width="180" />
        <el-table-column prop="exp4" label="Exp4 (方法对比)" width="180" />
      </el-table>
    </div>

    <el-dialog v-model="showDetailsDialog" :title="`${selectedExp?.name} 详细评估结果`" width="1200px">
      <div v-loading="loading" class="exp-details">
        <div class="details-layout">
          <div class="details-left">
            <h4>混淆矩阵</h4>
            <div class="confusion-matrix-container">
              <img
                v-if="confusionMatrixUrl"
                :src="confusionMatrixUrl"
                alt="混淆矩阵"
                class="confusion-matrix-img"
              />
              <el-empty v-else description="暂无混淆矩阵数据" />
            </div>
          </div>

          <div class="details-right">
            <div class="accuracy-display">
              <div class="accuracy-label">总体准确率</div>
              <div class="accuracy-value">
                {{ currentReport?.accuracy ? (currentReport.accuracy * 100).toFixed(2) : 'N/A' }}%
              </div>
            </div>

            <div class="metrics-chart">
              <h5>各类别性能指标</h5>
              <div ref="singleMetricsChartRef" class="chart-container"></div>
            </div>

            <div class="f1-display">
              <h5>Macro 平均 F1 分数</h5>
              <div class="f1-value">
                {{ calculateMacroF1() }}
              </div>
            </div>
          </div>
        </div>
      </div>
    </el-dialog>
  </div>
</template>

<script setup lang="ts">
defineOptions({
  name: 'ModelComparison'
})

import { ref, computed, onMounted, nextTick, watch } from 'vue'
import { useExperimentStore } from '@/stores/experiment'
import { experimentApi } from '@/api/experiment'
import * as echarts from 'echarts'
import type { ECharts } from 'echarts'
import type { ExperimentInfo, EvaluationReport } from '@/types'

const experimentStore = useExperimentStore()

const experiments = computed(() => experimentStore.experiments)
const selectedExperiment = computed(() => experimentStore.selectedExperiment)
const currentReport = computed(() => experimentStore.currentReport)
const loading = computed(() => experimentStore.loading)

const showDetailsDialog = ref(false)
const selectedExp = ref<ExperimentInfo | null>(null)
const confusionMatrixUrl = ref('')
const selectedExpsForRadar = ref<string[]>(['exp1', 'exp2', 'exp3', 'exp4'])

const accuracyChartRef = ref<HTMLElement>()
const radarChartRef = ref<HTMLElement>()
const modeBarChartRef = ref<HTMLElement>()
const metricsChartRef = ref<HTMLElement>()
const singleMetricsChartRef = ref<HTMLElement>()

let accuracyChart: ECharts | null = null
let radarChart: ECharts | null = null
let modeBarChart: ECharts | null = null
let metricsChart: ECharts | null = null
let singleMetricsChart: ECharts | null = null

const allReports = ref<Record<string, EvaluationReport>>({})

const modeList = ref<{ key: string; label: string }[]>([])
const selectedModesForBar = ref<string[]>([])

const modeNames: Record<string, string> = {
  Walk: '步行',
  Bike: '自行车',
  'Car & taxi': '汽车/出租',
  Bus: '公交',
  Subway: '地铁',
  Train: '火车',
}

const expColors = {
  exp1: '#4A90E2',
  exp2: '#52C41A',
  exp3: '#FA8C16',
  exp4: '#722ED1',
}

const expNames = {
  exp1: 'Exp1 (基线)',
  exp2: 'Exp2 (+OSM)',
  exp3: 'Exp3 (+天气)',
  exp4: 'Exp4 (方法对比)',
}

onMounted(async () => {
  experimentStore.loadExperiments()
  await loadAllReports()
  await nextTick()
  initAccuracyChart()
  initRadarChart()
  initModeBarChart()
  initMetricsChart()
})

watch(() => modeList.value, () => {
  nextTick(() => {
    if (radarChart) updateRadarChart()
    if (modeBarChart) updateModeBarChart()
  })
})

async function loadAllReports() {
  const expIds = ['exp1', 'exp2', 'exp3', 'exp4']
  for (const expId of expIds) {
    try {
      const report = await experimentApi.getExperimentReport(expId)
      allReports.value[expId] = report
    } catch (error) {
      console.error(`加载 ${expId} 报告失败:`, error)
    }
  }
  
  const allModes = new Set<string>()
  Object.values(allReports.value).forEach(report => {
    if (report?.f1_score) {
      Object.keys(report.f1_score).forEach(mode => allModes.add(mode))
    }
  })
  
  modeList.value = Array.from(allModes).map(key => ({
    key,
    label: modeNames[key] || key
  })).sort((a, b) => {
    const order = ['Walk', 'Bike', 'Bus', 'Car & taxi', 'Subway', 'Train']
    return order.indexOf(a.key) - order.indexOf(b.key)
  })
  
  selectedModesForBar.value = modeList.value.map(m => m.key)
}

function initAccuracyChart() {
  if (!accuracyChartRef.value) return

  if (accuracyChart) {
    accuracyChart.dispose()
  }

  accuracyChart = echarts.init(accuracyChartRef.value)

  const expIds = ['exp1', 'exp2', 'exp3', 'exp4']
  const accuracies = expIds.map(id => {
    const report = allReports.value[id]
    return report ? report.accuracy * 100 : 0
  })

  const option = {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow',
      },
      formatter: (params: any) => {
        return `<strong>${params[0].axisValue}</strong><br/>` +
          `${params[0].marker} 准确率: ${params[0].data}%`
      },
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true,
    },
    xAxis: {
      type: 'category',
      data: expIds.map(id => expNames[id as keyof typeof expNames]),
      axisLabel: {
        interval: 0,
        rotate: 15,
      },
    },
    yAxis: {
      type: 'value',
      name: '准确率 (%)',
      min: 0,
      max: 100,
    },
    series: [
      {
        name: '准确率',
        type: 'bar',
        data: accuracies,
        itemStyle: {
          color: (params: any) => {
            const colors = ['#4A90E2', '#52C41A', '#FA8C16', '#722ED1']
            return colors[params.dataIndex]
          },
        },
        label: {
          show: false,
        },
        barWidth: '50%',
      },
    ],
  }

  accuracyChart.setOption(option)
}

function initRadarChart() {
  if (!radarChartRef.value) return

  if (radarChart) {
    radarChart.dispose()
  }

  radarChart = echarts.init(radarChartRef.value)
  updateRadarChart()
}

function initModeBarChart() {
  if (!modeBarChartRef.value) return

  if (modeBarChart) {
    modeBarChart.dispose()
  }

  modeBarChart = echarts.init(modeBarChartRef.value)
  updateModeBarChart()
}

function updateModeBarChart() {
  if (!modeBarChart) return

  const expIds = ['exp1', 'exp2', 'exp3', 'exp4']
  const selectedModeKeys = selectedModesForBar.value

  const series = expIds.map(expId => {
    const report = allReports.value[expId]
    const data = selectedModeKeys.map(modeKey => {
      return report?.f1_score?.[modeKey] ? report.f1_score[modeKey] * 100 : 0
    })
    return {
      name: expNames[expId as keyof typeof expNames],
      type: 'bar',
      data: data,
      itemStyle: {
        color: expColors[expId as keyof typeof expColors],
      },
      label: {
        show: false,
      },
    }
  })

  const option = {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow',
      },
      formatter: (params: any) => {
        let result = `<strong>${params[0].axisValue}</strong><br/>`
        params.forEach((param: any) => {
          result += `${param.marker} ${param.seriesName}: ${param.data}%<br/>`
        })
        return result
      },
    },
    legend: {
      data: expIds.map(id => expNames[id as keyof typeof expNames]),
      top: 0,
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      top: '10%',
      containLabel: true,
    },
    xAxis: {
      type: 'category',
      data: selectedModeKeys.map(key => modeNames[key] || key),
    },
    yAxis: {
      type: 'value',
      name: 'F1分数 (%)',
      min: 0,
      max: 100,
    },
    series: series,
  }

  modeBarChart.setOption(option)
}

function updateRadarChart() {
  if (!radarChart) return

  const selectedExpIds = selectedExpsForRadar.value
  
  if (selectedExpIds.length === 0) {
    radarChart.setOption({
      series: [],
      radar: {
        indicator: [],
      },
    })
    return
  }

  const modes = modeList.value.map(m => m.key)
  const modeLabels = modes.map(m => modeNames[m] || m)

  const series = selectedExpsForRadar.value.map(expId => {
    const report = allReports.value[expId]
    const data = modes.map(m => {
      return report?.f1_score?.[m] ? report.f1_score[m] * 100 : 0
    })
    return {
      name: expNames[expId as keyof typeof expNames],
      value: data,
      lineStyle: {
        color: expColors[expId as keyof typeof expColors],
      },
      itemStyle: {
        color: expColors[expId as keyof typeof expColors],
      },
      areaStyle: {
        color: expColors[expId as keyof typeof expColors],
        opacity: 0.2,
      },
    }
  })

  const option = {
    tooltip: {
      trigger: 'item',
      formatter: (params: any) => {
        let result = `<strong>${params.seriesName}</strong><br/>`
        params.value.forEach((val: number, idx: number) => {
          const mode = modeList.value[idx]
          if (mode) {
            result += `${mode.label}: ${val.toFixed(2)}%<br/>`
          }
        })
        return result
      },
    },
    legend: {
      data: selectedExpsForRadar.value.map(id => expNames[id as keyof typeof expNames]),
      bottom: 0,
    },
    radar: {
      indicator: modeLabels.map(label => ({
        name: label,
        max: 100,
      })),
      radius: '60%',
    },
    series: [
      {
        name: 'F1分数',
        type: 'radar',
        data: series,
      },
    ],
  }

  radarChart.setOption(option)
}

function initMetricsChart() {
  if (!metricsChartRef.value) return

  if (metricsChart) {
    metricsChart.dispose()
  }

  metricsChart = echarts.init(metricsChartRef.value)

  const expIds = ['exp1', 'exp2', 'exp3', 'exp4']
  const metrics = ['准确率', '精确率', '召回率', 'F1分数']

  const seriesData = metrics.map((metric, idx) => {
    const data = expIds.map(id => {
      const report = allReports.value[id]
      if (!report) return 0

      if (metric === '准确率') {
        return report.accuracy * 100
      }

      const values = Object.values(report.precision || {})
      const avg = values.reduce((sum, v) => sum + v, 0) / values.length

      if (metric === '精确率') {
        return avg * 100
      } else if (metric === '召回率') {
        const recallValues = Object.values(report.recall || {})
        return (recallValues.reduce((sum, v) => sum + v, 0) / recallValues.length) * 100
      } else {
        const f1Values = Object.values(report.f1_score || {})
        return (f1Values.reduce((sum, v) => sum + v, 0) / f1Values.length) * 100
      }
    })

    return {
      name: metric,
      type: 'line',
      data: data,
      smooth: true,
      symbol: 'circle',
      symbolSize: 10,
      label: {
        show: false,
      },
    }
  })

  const option = {
    tooltip: {
      trigger: 'axis',
      formatter: (params: any) => {
        let result = `<strong>${params[0].axisValue}</strong><br/>`
        params.forEach((param: any) => {
          result += `${param.marker} ${param.seriesName}: ${param.data}%<br/>`
        })
        return result
      },
    },
    legend: {
      data: metrics,
      top: 0,
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      top: '15%',
      containLabel: true,
    },
    xAxis: {
      type: 'category',
      data: expIds.map(id => expNames[id as keyof typeof expNames]),
    },
    yAxis: {
      type: 'value',
      name: '百分比 (%)',
      min: 0,
      max: 100,
    },
    series: seriesData,
  }

  metricsChart.setOption(option)
}

watch(showDetailsDialog, async (show) => {
  if (show && selectedExp.value) {
    await loadExperimentData(selectedExp.value.id)
  } else {
    if (confusionMatrixUrl.value) {
      URL.revokeObjectURL(confusionMatrixUrl.value)
      confusionMatrixUrl.value = ''
    }
    if (singleMetricsChart) {
      singleMetricsChart.dispose()
      singleMetricsChart = null
    }
  }
})

async function loadExperimentData(expId: string) {
  confusionMatrixUrl.value = ''
  if (singleMetricsChart) {
    singleMetricsChart.dispose()
    singleMetricsChart = null
  }

  await Promise.all([
    experimentStore.loadExperimentReport(expId),
    loadConfusionMatrix(expId),
  ])

  await nextTick()
  initSingleMetricsChart()
}

async function loadConfusionMatrix(expId: string) {
  try {
    const blob = await experimentApi.getExperimentConfusionMatrix(expId)
    confusionMatrixUrl.value = URL.createObjectURL(blob)
  } catch (error) {
    console.error('加载混淆矩阵失败:', error)
  }
}

function showDetails(exp: ExperimentInfo) {
  selectedExp.value = exp
  showDetailsDialog.value = true
}

function selectExperiment(exp: any) {
  experimentStore.selectExperiment(exp)
}

function getStatusType(status: string): string {
  const statusMap: Record<string, string> = {
    completed: 'success',
    not_trained: 'info',
    training: 'warning',
  }
  return statusMap[status] || 'info'
}

function getStatusText(status: string): string {
  const statusMap: Record<string, string> = {
    completed: '已完成',
    not_trained: '未训练',
    training: '训练中',
  }
  return statusMap[status] || '未知'
}

function initSingleMetricsChart() {
  if (!singleMetricsChartRef.value || !currentReport.value) {
    return
  }

  if (singleMetricsChart) {
    singleMetricsChart.dispose()
  }

  singleMetricsChart = echarts.init(singleMetricsChartRef.value)

  const report = currentReport.value
  const modes = Object.keys(report.precision || {})

  const precisionData = modes.map(m => report.precision?.[m] || 0)
  const recallData = modes.map(m => report.recall?.[m] || 0)
  const f1Data = modes.map(m => report.f1_score?.[m] || 0)

  const option = {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'shadow',
      },
      formatter: (params: any) => {
        let result = `<strong>${params[0].axisValue}</strong><br/>`
        params.forEach((param: any) => {
          const value = (param.data * 100).toFixed(2)
          result += `${param.marker} ${param.seriesName}: ${value}%<br/>`
        })
        return result
      },
    },
    legend: {
      data: ['精确率', '召回率', 'F1分数'],
      top: 10,
    },
    xAxis: {
      type: 'category',
      data: modes.map(m => modeNames[m] || m),
      axisLabel: {
        interval: 0,
        rotate: 30,
      },
    },
    yAxis: {
      type: 'value',
      max: 1,
      axisLabel: {
        formatter: '{value}',
      },
    },
    series: [
      {
        name: '精确率',
        type: 'bar',
        data: precisionData,
        itemStyle: {
          color: '#4A90E2',
        },
        label: {
          show: false,
        },
      },
      {
        name: '召回率',
        type: 'bar',
        data: recallData,
        itemStyle: {
          color: '#52C41A',
        },
        label: {
          show: false,
        },
      },
      {
        name: 'F1分数',
        type: 'bar',
        data: f1Data,
        itemStyle: {
          color: '#FA8C16',
        },
        label: {
          show: false,
        },
      },
    ],
  }

  singleMetricsChart.setOption(option)
}

function calculateMacroF1(): string {
  if (!currentReport.value || !currentReport.value.f1_score) return 'N/A'

  const f1Scores = Object.values(currentReport.value.f1_score)
  const macroF1 = f1Scores.reduce((sum, val) => sum + val, 0) / f1Scores.length

  return macroF1.toFixed(3)
}

const comparisonData = [
  {
    feature: '轨迹特征',
    exp1: 'traj_9',
    exp2: 'traj_21',
    exp3: 'traj_21',
    exp4: 'traj_21',
  },
  {
    feature: '段级统计',
    exp1: 'stats_18',
    exp2: 'stats_18',
    exp3: 'stats_18',
    exp4: 'stats_18',
  },
  {
    feature: 'OSM空间',
    exp1: '-',
    exp2: '✓',
    exp3: '✓',
    exp4: '✓',
  },
  {
    feature: '天气特征',
    exp1: '-',
    exp2: '-',
    exp3: '✓',
    exp4: '✓',
  },
  {
    feature: '损失函数',
    exp1: '带类别权重的CrossEntropyLoss',
    exp2: '带类别权重的CrossEntropyLoss',
    exp3: '带类别权重的CrossEntropyLoss',
    exp4: 'LabelSmoothingFocalLoss (γ=2.0, α=0.1)',
  },
  {
    feature: '模型结构',
    exp1: '层次化Bi-LSTM',
    exp2: '层次化Bi-LSTM',
    exp3: '双路编码器<br/>(轨迹hidden=128 + 天气hidden=32)<br/>AttentionPooling后拼接',
    exp4: '双路编码器<br/>(轨迹hidden=128 + 天气hidden=32)<br/>AttentionPooling后拼接',
  },
]
</script>

<style scoped>
.model-comparison {
  padding: 24px;
  background: #0f1117;
  flex: 1;
  min-height: 0;
  overflow-y: auto;
}

.experiments-header {
  margin-bottom: 32px;
  text-align: center;
}

.experiments-header h2 {
  margin: 0 0 12px 0;
  font-size: 28px;
  color: #fff;
}

.experiments-header p {
  margin: 0;
  font-size: 14px;
  color: #909399;
}

.experiments-cards {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 24px;
  margin-bottom: 40px;
}

.experiment-card {
  background: #1a1f2e;
  border-radius: 8px;
  padding: 20px;
  cursor: pointer;
  transition: all 0.3s;
  border: 2px solid rgba(255, 255, 255, 0.08);
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.3);
}

.experiment-card:hover {
  transform: translateY(-4px);
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
}

.experiment-card.active {
  border-color: #4A90E2;
}

.experiment-card.completed {
  border-left: 4px solid #67C23A;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.card-header h3 {
  margin: 0;
  font-size: 18px;
  font-weight: 600;
  color: #fff;
}

.card-body {
  margin-bottom: 16px;
}

.description {
  font-size: 14px;
  color: #e5e8eb;
  line-height: 1.6;
  margin-bottom: 12px;
}

.features {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}

.feature-tag {
  background: rgba(74, 144, 226, 0.15);
  color: #4A90E2;
  padding: 4px 12px;
  border-radius: 4px;
  font-size: 12px;
}

.card-footer {
  text-align: center;
}

.comparison-section {
  background: #1a1f2e;
  border-radius: 8px;
  padding: 24px;
  margin-bottom: 24px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255, 255, 255, 0.08);
}

.comparison-section h3 {
  margin: 0 0 20px 0;
  font-size: 18px;
  font-weight: 600;
  color: #fff;
  padding-bottom: 12px;
  border-bottom: 2px solid #4A90E2;
}

.radar-selector {
  margin-bottom: 20px;
}

.mode-selector {
  margin-bottom: 20px;
}

.chart-container {
  width: 100%;
  height: 450px;
}

.exp-details {
  min-height: 400px;
}

.details-layout {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 24px;
}

.details-left h4,
.details-right h4 {
  margin: 0 0 16px 0;
  font-size: 16px;
  font-weight: 600;
  color: #fff;
}

.confusion-matrix-container {
  background: rgba(255, 255, 255, 0.02);
  border-radius: 8px;
  padding: 16px;
  text-align: center;
  border: 1px solid rgba(255, 255, 255, 0.08);
}

.confusion-matrix-img {
  max-width: 100%;
  height: auto;
  border-radius: 4px;
}

.accuracy-display {
  text-align: center;
  padding: 24px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 12px;
  margin-bottom: 24px;
}

.accuracy-label {
  font-size: 14px;
  color: #fff;
  margin-bottom: 8px;
}

.accuracy-value {
  font-size: 48px;
  font-weight: 700;
  color: #fff;
}

.metrics-chart {
  margin-bottom: 24px;
}

.f1-display {
  text-align: center;
  padding: 20px;
  background: rgba(255, 255, 255, 0.02);
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.08);
}

.f1-display h5 {
  margin: 0 0 8px 0;
  font-size: 14px;
  color: #909399;
}

.f1-value {
  font-size: 36px;
  font-weight: 600;
  color: #4A90E2;
}
</style>
