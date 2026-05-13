import { create } from 'zustand'
import { toast } from 'sonner'
import type { ImageResult } from '@/types'
import { exportImagesZip, exportPDF } from '@/services/api'

interface ResultState {
  currentResult: ImageResult | null
  batchResults: ImageResult[]
  isExporting: boolean
  isExportingImages: boolean
  activeTaskId: string | null

  setCurrentResult: (result: ImageResult) => void
  setBatchResults: (results: ImageResult[], taskId: string) => void
  downloadPDF: (taskId: string) => Promise<void>
  downloadImagesZip: (taskId: string) => Promise<void>
  clearResults: () => void
}

function triggerDownload(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

export const useResultStore = create<ResultState>((set) => ({
  currentResult: null,
  batchResults: [],
  isExporting: false,
  isExportingImages: false,
  activeTaskId: null,

  setCurrentResult: (result) => set({ currentResult: result }),

  setBatchResults: (results, taskId) =>
    set({
      batchResults: results,
      activeTaskId: taskId,
      currentResult: results[0] ?? null,
    }),

  downloadPDF: async (taskId) => {
    set({ isExporting: true })
    try {
      const blob = await exportPDF(taskId)
      triggerDownload(blob, `PhoneScope_Report_${taskId}.pdf`)
      toast.success('PDF 报告已下载')
    } catch (e: any) {
      toast.error('导出 PDF 失败', { description: e?.message || '请稍后再试' })
    } finally {
      set({ isExporting: false })
    }
  },

  downloadImagesZip: async (taskId) => {
    set({ isExportingImages: true })
    try {
      const blob = await exportImagesZip(taskId)
      triggerDownload(blob, `PhoneScope_Images_${taskId}.zip`)
      toast.success('标注图压缩包已下载')
    } catch (e: any) {
      toast.error('导出图片失败', { description: e?.message || '请稍后再试' })
    } finally {
      set({ isExportingImages: false })
    }
  },

  clearResults: () =>
    set({ currentResult: null, batchResults: [], activeTaskId: null }),
}))
