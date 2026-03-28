import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
import type { TrajectoryPrediction, TransportMode } from '@/types'
import { trajectoryApi } from '@/api/trajectory'

export const useTrajectoryStore = defineStore('trajectory', () => {
  const predictions = ref<TrajectoryPrediction[]>([])
  const selectedTrajectory = ref<TrajectoryPrediction | null>(null)
  const transportModes = ref<TransportMode[]>([])
  const loading = ref(false)
  const error = ref<string | null>(null)
  const visibleTrajectoryIds = ref<Set<string>>(new Set())

  const filteredPredictions = computed(() => {
    return predictions.value.filter(pred => visibleTrajectoryIds.value.has(pred.trajectory_id))
  })

  async function loadTransportModes() {
    try {
      transportModes.value = await trajectoryApi.getModes()
    } catch (e) {
      error.value = '加载交通方式配置失败'
      console.error(e)
    }
  }

  async function predictTrajectory(file: File, model: string = 'exp1', enableSegmentation: boolean = true) {
    loading.value = true
    error.value = null

    try {
      const result = await trajectoryApi.predict(file, model, enableSegmentation)
      predictions.value.push(result)
      visibleTrajectoryIds.value.add(result.trajectory_id)
      selectedTrajectory.value = result
      return result
    } catch (e) {
      error.value = '预测失败，请检查文件格式'
      console.error(e)
      throw e
    } finally {
      loading.value = false
    }
  }

  function selectTrajectory(trajectory: TrajectoryPrediction) {
    selectedTrajectory.value = trajectory
  }

  function toggleTrajectoryVisibility(trajectoryId: string) {
    if (visibleTrajectoryIds.value.has(trajectoryId)) {
      visibleTrajectoryIds.value.delete(trajectoryId)
    } else {
      visibleTrajectoryIds.value.add(trajectoryId)
    }
  }

  function deleteTrajectory(trajectoryId: string) {
    predictions.value = predictions.value.filter(pred => pred.trajectory_id !== trajectoryId)
    visibleTrajectoryIds.value.delete(trajectoryId)
    if (selectedTrajectory.value?.trajectory_id === trajectoryId) {
      selectedTrajectory.value = predictions.value.length > 0 ? predictions.value[predictions.value.length - 1] || null : null
    }
  }

  function clearPredictions() {
    predictions.value = []
    selectedTrajectory.value = null
    visibleTrajectoryIds.value.clear()
  }

  return {
    predictions,
    selectedTrajectory,
    transportModes,
    loading,
    error,
    visibleTrajectoryIds,
    filteredPredictions,
    loadTransportModes,
    predictTrajectory,
    selectTrajectory,
    toggleTrajectoryVisibility,
    deleteTrajectory,
    clearPredictions,
  }
})
