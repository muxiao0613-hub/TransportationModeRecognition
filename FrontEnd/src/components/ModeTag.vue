<template>
  <el-tag :color="color" size="large">
    <el-icon style="margin-right: 4px"><component :is="iconComponent" /></el-icon>
    {{ name }}
  </el-tag>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'

interface Props {
  mode: string
}

const props = defineProps<Props>()

const modeConfig: Record<string, { name: string; color: string; icon: string }> = {
  walk: { name: '步行', color: '#4A90E2', icon: 'Walking' },
  bike: { name: '自行车', color: '#52C41A', icon: 'Bicycle' },
  bus: { name: '公交', color: '#FA8C16', icon: 'Van' },
  car: { name: '汽车/出租', color: '#F5222D', icon: 'Car' },
  subway: { name: '地铁', color: '#722ED1', icon: 'Subway' },
  train: { name: '火车', color: '#13C2C2', icon: 'Train' },
}

const config = computed(() => modeConfig[props.mode] || { name: props.mode, color: '#666', icon: 'QuestionFilled' })

const name = computed(() => config.value.name)
const color = computed(() => config.value.color)
const iconComponent = computed(() => (ElementPlusIconsVue as any)[config.value.icon] || ElementPlusIconsVue.QuestionFilled)
</script>
