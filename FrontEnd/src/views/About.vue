<template>
  <div class="about-page">
    <div class="page-header">
      <h1>城市出行智能感知平台</h1>
      <p>Urban Travel Intelligent Perception Platform</p>
    </div>

    <div class="content-section">
      <div class="section-card">
        <h3>研究背景</h3>
        <div class="background-content">
          <p>
            随着城市化进程的加速，城市交通问题日益凸显。准确识别城市居民的出行方式对于优化交通管理、
            改善城市规划、提升出行体验具有重要意义。传统的交通调查方法成本高、周期长、样本量有限，
            难以满足精细化管理的需求。
          </p>
          <p>
            本平台基于深度学习技术，利用GPS轨迹数据自动识别6种主要交通方式（步行、自行车、公交、
            汽车/出租、火车、地铁），为智慧城市建设提供数据支撑。系统采用Bi-LSTM多模态网络架构，
            结合轨迹特征、空间特征和环境特征，实现了高精度的交通方式识别。
          </p>
          <p>
            平台集成了数据预处理、模型训练、评估分析和可视化展示等功能，适用于物流管理、执法监管、
            城市规划、交通研究等多种应用场景的研究演示。
          </p>
        </div>
      </div>

      <div class="section-card">
        <h3>技术架构</h3>
        <div class="architecture-diagram">
          <svg viewBox="0 0 800 400" class="arch-svg">
            <defs>
              <linearGradient id="grad1" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" style="stop-color:#4A90E2;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#667eea;stop-opacity:1" />
              </linearGradient>
              <linearGradient id="grad2" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" style="stop-color:#52C41A;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#73d13d;stop-opacity:1" />
              </linearGradient>
              <linearGradient id="grad3" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" style="stop-color:#FA8C16;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#ffc53d;stop-opacity:1" />
              </linearGradient>
            </defs>

            <rect x="50" y="30" width="180" height="60" rx="8" fill="url(#grad1)" />
            <text x="140" y="65" text-anchor="middle" fill="white" font-size="14" font-weight="bold">数据采集层</text>

            <rect x="310" y="30" width="180" height="60" rx="8" fill="url(#grad2)" />
            <text x="400" y="65" text-anchor="middle" fill="white" font-size="14" font-weight="bold">数据处理层</text>

            <rect x="570" y="30" width="180" height="60" rx="8" fill="url(#grad3)" />
            <text x="660" y="65" text-anchor="middle" fill="white" font-size="14" font-weight="bold">模型训练层</text>

            <line x1="230" y1="60" x2="310" y2="60" stroke="#909399" stroke-width="2" marker-end="url(#arrow)" />
            <line x1="490" y1="60" x2="570" y2="60" stroke="#909399" stroke-width="2" marker-end="url(#arrow)" />

            <rect x="50" y="150" width="180" height="60" rx="8" fill="url(#grad1)" />
            <text x="140" y="185" text-anchor="middle" fill="white" font-size="14" font-weight="bold">模型推理层</text>

            <rect x="310" y="150" width="180" height="60" rx="8" fill="url(#grad2)" />
            <text x="400" y="185" text-anchor="middle" fill="white" font-size="14" font-weight="bold">API服务层</text>

            <rect x="570" y="150" width="180" height="60" rx="8" fill="url(#grad3)" />
            <text x="660" y="185" text-anchor="middle" fill="white" font-size="14" font-weight="bold">可视化层</text>

            <line x1="230" y1="180" x2="310" y2="180" stroke="#909399" stroke-width="2" marker-end="url(#arrow)" />
            <line x1="490" y1="180" x2="570" y2="180" stroke="#909399" stroke-width="2" marker-end="url(#arrow)" />

            <line x1="660" y1="90" x2="660" y2="150" stroke="#909399" stroke-width="2" marker-end="url(#arrow)" />
            <line x1="140" y1="90" x2="140" y2="150" stroke="#909399" stroke-width="2" marker-end="url(#arrow)" />

            <rect x="200" y="280" width="400" height="80" rx="8" fill="#f5f7fa" stroke="#4A90E2" stroke-width="2" />
            <text x="400" y="315" text-anchor="middle" fill="#303133" font-size="14" font-weight="bold">核心技术栈</text>
            <text x="400" y="340" text-anchor="middle" fill="#606266" font-size="12">Python + PyTorch + FastAPI + Vue 3 + TypeScript + ECharts</text>

            <line x1="400" y1="210" x2="400" y2="280" stroke="#909399" stroke-width="2" marker-end="url(#arrow)" />
          </svg>
        </div>
      </div>

      <div class="section-card">
        <h3>应用场景</h3>
        <div class="scenarios-grid">
          <div class="scenario-item">
            <div class="scenario-icon">
              <el-icon><Van /></el-icon>
            </div>
            <h4>物流管理</h4>
            <p>优化配送路线，提升物流效率，降低运营成本</p>
          </div>

          <div class="scenario-item">
            <div class="scenario-icon">
              <el-icon><Monitor /></el-icon>
            </div>
            <h4>执法监管</h4>
            <p>交通违规检测，车辆轨迹追踪，安全监控</p>
          </div>

          <div class="scenario-item">
            <div class="scenario-icon">
              <el-icon><OfficeBuilding /></el-icon>
            </div>
            <h4>城市规划</h4>
            <p>出行模式分析，交通设施规划，城市设计优化</p>
          </div>

          <div class="scenario-item">
            <div class="scenario-icon">
              <el-icon><DataAnalysis /></el-icon>
            </div>
            <h4>交通研究</h4>
            <p>出行行为研究，交通流分析，政策评估</p>
          </div>
        </div>
      </div>

      <div class="section-card">
        <h3>数据集说明</h3>
        <div class="dataset-info">
          <h4>数据来源</h4>
          <p>
            本项目使用 <strong>GeoLife Trajectories 1.3</strong> 数据集，该数据集由微软亚洲研究院于2007年4月至2012年8月收集，包含了182个用户的GPS轨迹数据。
          </p>
          
          <h4>数据筛选</h4>
          <p>
            原始数据集包含17,621条带标签的轨迹段，但并非所有数据都适合用于模型训练。经过数据筛选和清洗后：
          </p>
          <ul class="dataset-stats">
            <li><strong>原始用户数</strong>：182个用户</li>
            <li><strong>当前用户数</strong>：61个用户（拥有足够标注数据的用户）</li>
            <li><strong>原始轨迹段</strong>：17,621段</li>
            <li><strong>清洗后轨迹段</strong>：9,531段（基础清洗）</li>
            <li><strong>均衡采样后</strong>：8,841段（用于模型训练）</li>
          </ul>
          
          <h4>筛选原因</h4>
          <p>
            1. <strong>数据质量过滤</strong>：移除GPS点数少于10个的轨迹段，确保轨迹具有足够的特征信息
          </p>
          <p>
            2. <strong>异常值处理</strong>：检测和移除物理异常点（如速度过快、位置跳跃等）
          </p>
          <p>
            3. <strong>用户筛选</strong>：只保留拥有足够标注数据的用户，确保每个用户有足够的样本
          </p>
          <p>
            4. <strong>类别均衡</strong>：为避免模型偏向样本多的类别，对各类别进行了均衡采样
          </p>
          
          <h4>清洗后的6种交通方式</h4>
          <ul class="mode-list">
            <li>步行 (Walk)</li>
            <li>自行车 (Bike)</li>
            <li>公交 (Bus)</li>
            <li>汽车/出租 (Car &amp; taxi)</li>
            <li>火车 (Train)</li>
            <li>地铁 (Subway)</li>
          </ul>
        </div>
      </div>

      <div class="section-card">
        <h3>技术栈</h3>
        <div class="tech-cloud">
          <el-tag v-for="tech in techStack" :key="tech" size="large" class="tech-tag">
            {{ tech }}
          </el-tag>
        </div>
      </div>

      <div class="section-card">
        <h3>项目信息</h3>
        <div class="project-info">
          <div class="info-item">
            <span class="info-label">数据集：</span>
            <span class="info-value">GeoLife Trajectories 1.3</span>
          </div>
          <div class="info-item">
            <span class="info-label">模型架构：</span>
            <span class="info-value">Bi-LSTM + Attention Mechanism</span>
          </div>
          <div class="info-item">
            <span class="info-label">识别类别：</span>
            <span class="info-value">6种交通方式</span>
          </div>
          <div class="info-item">
            <span class="info-label">开发框架：</span>
            <span class="info-value">Vue 3 + FastAPI</span>
          </div>
          <div class="info-item">
            <span class="info-label">项目性质：</span>
            <span class="info-value">研究展示型平台</span>
          </div>
        </div>

        <div class="github-link">
          <el-button type="primary" :icon="Link" size="large">
            https://github.com/muxiao0613-hub/TransportationModeRecognition.git
          </el-button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { Van, Monitor, OfficeBuilding, DataAnalysis, Link } from '@element-plus/icons-vue'

const techStack = [
  'Python',
  'PyTorch',
  'FastAPI',
  'Vue 3',
  'TypeScript',
  'Vite',
  'Element Plus',
  'ECharts',
  'Leaflet',
  'Pinia',
  'Axios',
  'Bi-LSTM',
  'Attention',
  'Focal Loss',
  'GeoLife',
]
</script>

<style scoped>
.about-page {
  padding: 24px;
  background: #0f1117;
  flex: 1;
  min-height: 0;
  overflow-y: auto;
}

.page-header {
  text-align: center;
  margin-bottom: 48px;
  padding: 40px 0;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  border-radius: 12px;
  color: white;
}

.page-header h1 {
  margin: 0 0 12px 0;
  font-size: 36px;
  font-weight: 700;
}

.page-header p {
  margin: 0;
  font-size: 16px;
  opacity: 0.9;
}

.content-section {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.section-card {
  background: #1a1f2e;
  border-radius: 8px;
  padding: 32px;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.3);
  border: 1px solid rgba(255, 255, 255, 0.08);
}

.section-card h3 {
  margin: 0 0 24px 0;
  font-size: 20px;
  font-weight: 600;
  color: #fff;
  padding-bottom: 12px;
  border-bottom: 2px solid #4A90E2;
}

.background-content {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.background-content p {
  margin: 0;
  font-size: 15px;
  line-height: 1.8;
  color: #e5e8eb;
  text-align: justify;
}

.architecture-diagram {
  display: flex;
  justify-content: center;
  padding: 20px 0;
}

.arch-svg {
  max-width: 100%;
  height: auto;
}

.scenarios-grid {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 24px;
}

.scenario-item {
  text-align: center;
  padding: 24px;
  background: rgba(255, 255, 255, 0.02);
  border-radius: 8px;
  transition: all 0.3s;
  border: 1px solid rgba(255, 255, 255, 0.08);
}

.scenario-item:hover {
  transform: translateY(-4px);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.4);
}

.scenario-icon {
  width: 64px;
  height: 64px;
  border-radius: 50%;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  display: flex;
  align-items: center;
  justify-content: center;
  margin: 0 auto 16px;
  color: white;
  font-size: 32px;
}

.scenario-item h4 {
  margin: 0 0 8px 0;
  font-size: 16px;
  font-weight: 600;
  color: #fff;
}

.scenario-item p {
  margin: 0;
  font-size: 13px;
  color: #909399;
  line-height: 1.5;
}

.dataset-info {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.dataset-info h4 {
  margin: 0 0 12px 0;
  font-size: 16px;
  font-weight: 600;
  color: #4A90E2;
}

.dataset-info p {
  margin: 0;
  font-size: 14px;
  line-height: 1.8;
  color: #e5e8eb;
}

.dataset-stats {
  margin: 0;
  padding: 16px 24px;
  background: rgba(74, 144, 226, 0.1);
  border-radius: 8px;
  border: 1px solid rgba(74, 144, 226, 0.3);
}

.dataset-stats li {
  padding: 8px 0;
  font-size: 14px;
  color: #e5e8eb;
  list-style: none;
  border-bottom: 1px solid rgba(255, 255, 255, 0.08);
}

.dataset-stats li:last-child {
  border-bottom: none;
}

.mode-list {
  margin: 0;
  padding: 16px 24px;
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
  background: rgba(82, 196, 26, 0.1);
  border-radius: 8px;
  border: 1px solid rgba(82, 196, 26, 0.3);
}

.mode-list li {
  padding: 10px 16px;
  font-size: 14px;
  color: #e5e8eb;
  list-style: none;
  background: rgba(255, 255, 255, 0.02);
  border-radius: 4px;
  border-left: 3px solid #52C41A;
}

.tech-cloud {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  padding: 20px;
  background: rgba(255, 255, 255, 0.02);
  border-radius: 8px;
  border: 1px solid rgba(255, 255, 255, 0.08);
}

.tech-tag {
  font-size: 14px;
  padding: 8px 16px;
  background: #1a1f2e;
  color: #4A90E2;
  border: 1px solid #4A90E2;
  transition: all 0.3s;
}

.tech-tag:hover {
  background: #4A90E2;
  color: white;
  transform: scale(1.05);
}

.project-info {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
  margin-bottom: 32px;
}

.info-item {
  display: flex;
  align-items: center;
  padding: 12px;
  background: rgba(255, 255, 255, 0.02);
  border-radius: 6px;
  border: 1px solid rgba(255, 255, 255, 0.08);
}

.info-label {
  font-size: 14px;
  color: #909399;
  margin-right: 12px;
  min-width: 100px;
}

.info-value {
  font-size: 14px;
  color: #e5e8eb;
  font-weight: 500;
}

.github-link {
  text-align: center;
  padding-top: 24px;
  border-top: 1px solid rgba(255, 255, 255, 0.08);
}
</style>
