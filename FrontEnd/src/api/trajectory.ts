import axios from 'axios'
import type { TrajectoryPrediction, TransportMode } from '@/types'

const API_BASE_URL = 'http://localhost:8000/api'

export const trajectoryApi = {
  async predict(file: File, model: string = 'exp1', enableSegmentation: boolean = true): Promise<TrajectoryPrediction> {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('model', model)
    formData.append('enable_segmentation', String(enableSegmentation))

    const response = await axios.post<TrajectoryPrediction>(
      `${API_BASE_URL}/trajectory/predict`,
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    )

    return response.data
  },

  async getModes(): Promise<TransportMode[]> {
    const response = await axios.get<TransportMode[]>(
      `${API_BASE_URL}/trajectory/modes`
    )

    return response.data
  },
}
