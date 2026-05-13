import axios from 'axios'
import { toast } from 'sonner'
import type {
  AnalysisResponse,
  BatchTaskResponse,
  TaskStatusResponse,
  HealthResponse,
  FusionResponse,
  LogsResponse,
  StatsResponse,
  SessionsResponse,
  AdminDebugInfo,
  ModelType,
} from '@/types'

const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '/api/v1',
  timeout: 120000,
})

api.interceptors.response.use(
  (res) => res,
  (err) => {
    const message =
      err.response?.data?.detail || err.message || '网络请求失败'
    if (err.response?.status === 503) {
      toast.error('服务暂停', { description: message })
    } else if (err.response?.status === 429) {
      toast.warning('请求限制', { description: message })
    }
    return Promise.reject(new Error(message))
  },
)

export async function checkHealth(): Promise<HealthResponse> {
  const { data } = await api.get<HealthResponse>('/health')
  return data
}

export async function detectImage(
  file: File,
  modelType: ModelType = 'swin',
): Promise<AnalysisResponse> {
  const form = new FormData()
  form.append('file', file)
  form.append('model_type', modelType)
  const { data } = await api.post<AnalysisResponse>('/detect/image', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
  return data
}

export async function detectBatch(
  files: File[],
  modelType: ModelType = 'swin',
  onProgress?: (progress: number) => void,
): Promise<BatchTaskResponse> {
  const form = new FormData()
  files.forEach((f) => form.append('files', f))
  form.append('model_type', modelType)
  const { data } = await api.post<BatchTaskResponse>('/detect/batch', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => {
      if (e.total && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100))
      }
    },
  })
  return data
}

export async function getTaskStatus(
  taskId: string,
): Promise<TaskStatusResponse> {
  const { data } = await api.get<TaskStatusResponse>(`/task/${taskId}`)
  return data
}

export async function detectFusion(
  files: File[],
  frameInterval: number = 30,
  modelType: ModelType = 'swin',
  onProgress?: (progress: number) => void,
): Promise<FusionResponse> {
  const form = new FormData()
  const isVideo = files.length === 1 && files[0].type.startsWith('video/')
  if (isVideo) {
    form.append('video', files[0])
    form.append('frame_interval', String(frameInterval))
  } else {
    files.forEach((f) => form.append('files', f))
  }
  form.append('model_type', modelType)
  const { data } = await api.post<FusionResponse>('/detect/fusion', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    timeout: 300000,
    onUploadProgress: (e) => {
      if (e.total && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100))
      }
    },
  })
  return data
}

export async function getSessions(params: {
  page?: number
  size?: number
  mode?: string
  model_name?: string
  date_from?: string
  date_to?: string
}): Promise<SessionsResponse> {
  const { data } = await api.get<SessionsResponse>('/sessions', { params })
  return data
}

export async function deleteSession(sessionId: string): Promise<void> {
  await api.delete(`/sessions/${sessionId}`)
}

export async function getLogs(params: {
  page?: number
  size?: number
  mode?: string
  model_name?: string
  date_from?: string
  date_to?: string
  session_id?: string
}): Promise<LogsResponse> {
  const { data } = await api.get<LogsResponse>('/logs', { params })
  return data
}

export async function getLogStats(): Promise<StatsResponse> {
  const { data } = await api.get<StatsResponse>('/logs/stats')
  return data
}

export async function deleteLog(id: number): Promise<void> {
  await api.delete(`/logs/${id}`)
}

export async function deleteLogsBySession(sessionId: string): Promise<void> {
  await api.delete('/logs', { params: { session_id: sessionId } })
}

export async function getAdminDebug(password: string): Promise<AdminDebugInfo> {
  const { data } = await api.get<AdminDebugInfo>('/admin/debug', {
    params: { key: password },
  })
  return data
}

export async function exportPDF(taskId: string): Promise<Blob> {
  const { data } = await api.post(
    '/export/pdf',
    { task_id: taskId, include_images: true },
    { responseType: 'blob' },
  )
  return data
}

export async function exportImagesZip(taskId: string): Promise<Blob> {
  const { data } = await api.post(
    '/export/images',
    { task_id: taskId, include_images: true },
    { responseType: 'blob' },
  )
  return data
}

// ---------- Admin write operations ----------

export async function adminDeleteSession(
  sessionId: string, key: string,
): Promise<void> {
  await api.delete(`/admin/sessions/${sessionId}`, { params: { key } })
}

export async function adminRestoreSession(
  sessionId: string, key: string,
): Promise<void> {
  await api.post(`/admin/sessions/${sessionId}/restore`, null, {
    params: { key },
  })
}

export async function adminPurgeDeleted(
  key: string,
): Promise<{ purged_sessions: number; purged_logs: number }> {
  const { data } = await api.post('/admin/sessions/purge-deleted', null, {
    params: { key },
  })
  return data
}

export async function adminDeleteLog(
  logId: number, key: string,
): Promise<void> {
  await api.delete(`/admin/logs/${logId}`, { params: { key } })
}

export async function adminTogglePhoneModel(
  modelId: number, key: string,
): Promise<{ is_active: boolean }> {
  const { data } = await api.post(`/admin/phone-models/${modelId}/toggle`, null, {
    params: { key },
  })
  return data
}

export default api
