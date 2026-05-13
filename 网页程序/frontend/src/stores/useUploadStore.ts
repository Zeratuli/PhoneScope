import { create } from 'zustand'
import type {
  FileWithPreview,
  TaskStatusResponse,
  FusionResult,
  ModelType,
  NumberedTask,
  ProcessingMode,
  ClassificationItem,
} from '@/types'
import { detectImage, detectBatch, detectFusion, getTaskStatus } from '@/services/api'

interface UploadState {
  files: FileWithPreview[]
  tasks: Map<string, TaskStatusResponse>
  numberedTasks: NumberedTask[]
  taskCounter: number
  isUploading: boolean
  uploadProgress: number
  error: string | null
  fusionResult: FusionResult | null
  frameInterval: number
  modelType: ModelType
  processingMode: ProcessingMode

  addFiles: (files: File[]) => void
  removeFile: (id: string) => void
  clearFiles: () => void
  submitDetect: () => Promise<string | null>
  submitFusion: () => Promise<string | null>
  pollTaskStatus: (taskId: string) => void
  setError: (error: string | null) => void
  setFrameInterval: (n: number) => void
  setModelType: (t: ModelType) => void
  setProcessingMode: (m: ProcessingMode) => void
  updateTaskSummary: (taskId: string, summary: string) => void
}

let pollTimers: Map<string, ReturnType<typeof setInterval>> = new Map()

function formatModelLabel(item: Pick<ClassificationItem, 'display_name' | 'model_name'>) {
  return item.display_name || item.model_name.replace(/_/g, ' ')
}

function buildSummary(results: TaskStatusResponse['results']): string {
  if (!results || results.length === 0) return '无结果'
  const phones = results.flatMap((r) => r.classifications)
  if (phones.length === 0) return '未检测到手机'
  const names = phones.map((c) => formatModelLabel(c))
  return `检测到 ${phones.length} 台：${names.slice(0, 3).join('、')}`
}

export const useUploadStore = create<UploadState>((set, get) => ({
  files: [],
  tasks: new Map(),
  numberedTasks: [],
  taskCounter: 0,
  isUploading: false,
  uploadProgress: 0,
  error: null,
  fusionResult: null,
  frameInterval: 30,
  modelType: 'swin',
  processingMode: 'single',

  addFiles: (newFiles) => {
    const items: FileWithPreview[] = newFiles.map((file) => ({
      id: crypto.randomUUID(),
      file,
      preview: file.type.startsWith('image/') ? URL.createObjectURL(file) : '',
      type: file.type.startsWith('image/') ? 'image' : 'video',
    }))
    set((s) => ({ files: [...s.files, ...items] }))
  },

  removeFile: (id) => {
    set((s) => {
      const f = s.files.find((f) => f.id === id)
      if (f?.preview) URL.revokeObjectURL(f.preview)
      return { files: s.files.filter((f) => f.id !== id) }
    })
  },

  clearFiles: () => {
    get().files.forEach((f) => {
      if (f.preview) URL.revokeObjectURL(f.preview)
    })
    set({ files: [], uploadProgress: 0 })
  },

  updateTaskSummary: (taskId, summary) => {
    set((s) => ({
      numberedTasks: s.numberedTasks.map((t) =>
        t.taskId === taskId ? { ...t, summary } : t,
      ),
    }))
  },

  submitDetect: async () => {
    const { files, modelType, processingMode } = get()
    if (files.length === 0) return null
    set({ isUploading: true, error: null })
    const taskNumber = get().taskCounter + 1
    set({ taskCounter: taskNumber })

    try {
      if (processingMode === 'single' && files.length === 1) {
        const res = await detectImage(files[0].file, modelType)
        const taskId = res.result.image_id
        const tasks = new Map(get().tasks)
        const taskStatus: TaskStatusResponse = {
          task_id: taskId,
          status: 'completed',
          progress: 1,
          current_file: null,
          results: [res.result],
          created_at: new Date().toISOString(),
          error: null,
        }
        tasks.set(taskId, taskStatus)
        const summary = buildSummary(taskStatus.results)
        set((s) => ({
          tasks,
          isUploading: false,
          files: [],
          numberedTasks: [
            {
              taskNumber,
              taskId,
              mode: 'single',
              modelType,
              status: 'completed',
              summary,
              createdAt: new Date().toISOString(),
            },
            ...s.numberedTasks,
          ],
        }))
        return taskId
      } else {
        const rawFiles = files.map((f) => f.file)
        const res = await detectBatch(rawFiles, modelType, (p) =>
          set({ uploadProgress: p }),
        )
        const taskId = res.task_id
        const tasks = new Map(get().tasks)
        tasks.set(taskId, {
          task_id: taskId,
          status: 'pending',
          progress: 0,
          current_file: null,
          results: null,
          created_at: new Date().toISOString(),
          error: null,
        })
        set((s) => ({
          tasks,
          isUploading: false,
          files: [],
          uploadProgress: 100,
          numberedTasks: [
            {
              taskNumber,
              taskId,
              mode: 'batch',
              modelType,
              status: 'pending',
              summary: '处理中...',
              createdAt: new Date().toISOString(),
            },
            ...s.numberedTasks,
          ],
        }))
        get().pollTaskStatus(taskId)
        return taskId
      }
    } catch (e: any) {
      set({ isUploading: false, error: e.message })
      return null
    }
  },

  submitFusion: async () => {
    const { files, frameInterval, modelType, processingMode } = get()
    if (files.length === 0) return null
    set({ isUploading: true, error: null, fusionResult: null })
    const taskNumber = get().taskCounter + 1
    set({ taskCounter: taskNumber })

    try {
      const rawFiles = files.map((f) => f.file)
      const res = await detectFusion(rawFiles, frameInterval, modelType, (p) =>
        set({ uploadProgress: p }),
      )
      const sessionId = res.result.session_id
      const summary = res.result.final_model_name
        ? `检测到 ${res.result.valid_frames} 帧有效：${res.result.final_display_name || res.result.final_model_name.replace(/_/g, ' ')} (${((res.result.final_confidence || 0) * 100).toFixed(1)}%)`
        : '未检测到手机'

      set((s) => ({
        fusionResult: res.result,
        isUploading: false,
        uploadProgress: 100,
        files: [],
        numberedTasks: [
            {
              taskNumber,
              taskId: sessionId,
              mode: processingMode === 'fusion_video' ? 'fusion_video' : 'fusion_images',
              modelType,
              status: 'completed',
              summary,
            createdAt: new Date().toISOString(),
          },
          ...s.numberedTasks,
        ],
      }))
      return sessionId
    } catch (e: any) {
      set({ isUploading: false, error: e.message })
      return null
    }
  },

  pollTaskStatus: (taskId) => {
    if (pollTimers.has(taskId)) return
    const timer = setInterval(async () => {
      try {
        const status = await getTaskStatus(taskId)
        const tasks = new Map(get().tasks)
        tasks.set(taskId, status)
        set({ tasks })
        if (status.status === 'completed' || status.status === 'failed') {
          clearInterval(timer)
          pollTimers.delete(taskId)
          const summary =
            status.status === 'completed'
              ? buildSummary(status.results)
              : `失败: ${status.error || '未知错误'}`
          get().updateTaskSummary(taskId, summary)
          set((s) => ({
            numberedTasks: s.numberedTasks.map((t) =>
              t.taskId === taskId ? { ...t, status: status.status as any } : t,
            ),
          }))
        }
      } catch {
        clearInterval(timer)
        pollTimers.delete(taskId)
      }
    }, 1500)
    pollTimers.set(taskId, timer)
  },

  setError: (error) => set({ error }),
  setFrameInterval: (n) => set({ frameInterval: n }),
  setModelType: (t) => set({ modelType: t }),
  setProcessingMode: (m) => set({ processingMode: m }),
}))
