import { useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import { useUploadStore } from '@/stores/useUploadStore'
import { useResultStore } from '@/stores/useResultStore'
import DropZone from '@/components/upload/DropZone'
import FilePreview from '@/components/upload/FilePreview'
import CameraView from '@/components/upload/CameraView'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import {
  Loader2,
  Trash2,
  SendHorizonal,
  Camera,
  ImageUp,
  CheckCircle,
  AlertCircle,
  Clock,
  ChevronRight,
} from 'lucide-react'
import { useState } from 'react'
import type { ModelType, NumberedTask, ProcessingMode } from '@/types'

type Tab = 'detect' | 'camera'

const MODEL_OPTIONS: { value: ModelType; label: string; desc: string }[] = [
  { value: 'swin', label: 'Swin Transformer', desc: '精度更高，适合样本充足场景' },
  { value: 'mobilenet', label: 'MobileNetV3', desc: '速度更快，适合小样本场景' },
]

const MODE_OPTIONS: { value: ProcessingMode; label: string; desc: string }[] = [
  { value: 'single', label: '单张检测', desc: '上传 1 张图片时做单张检测，上传多张图片时自动转为批量单张检测' },
  { value: 'fusion_images', label: '多图融合', desc: '上传 2~5 张图片，综合多张证据输出 1 个结论' },
  { value: 'fusion_video', label: '视频融合', desc: '上传 1 个视频，抽帧后执行多证据融合识别' },
]

const MODE_LABEL: Record<NumberedTask['mode'], string> = {
  single: '单图/批量',
  batch: '批量',
  fusion_images: '多图融合',
  fusion_video: '视频融合',
}

function TaskRow({
  task,
  onView,
}: {
  task: NumberedTask
  onView: (t: NumberedTask) => void
}) {
  const icons = {
    pending: <Clock className="h-4 w-4 text-muted-foreground" />,
    processing: <Loader2 className="h-4 w-4 animate-spin text-primary" />,
    completed: <CheckCircle className="h-4 w-4 text-green-500" />,
    failed: <AlertCircle className="h-4 w-4 text-destructive" />,
  }

  return (
    <div className="flex items-center gap-3 rounded-lg border border-border bg-card px-4 py-3">
      {icons[task.status]}
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium">检测任务 {task.taskNumber}</span>
          <Badge variant="secondary" className="text-xs px-1.5 py-0">
            {MODE_LABEL[task.mode]}
          </Badge>
          <Badge variant="outline" className="text-xs px-1.5 py-0">
            {task.modelType === 'swin' ? 'Swin' : 'MobileNet'}
          </Badge>
        </div>
        <p className="text-xs text-muted-foreground mt-0.5 truncate">{task.summary}</p>
      </div>
      {(task.status === 'completed') && (
        <button
          onClick={() => onView(task)}
          className="flex items-center gap-1 text-sm text-primary hover:underline shrink-0"
        >
          查看结果 <ChevronRight className="h-3.5 w-3.5" />
        </button>
      )}
    </div>
  )
}

export default function UploadPage() {
  const [tab, setTab] = useState<Tab>('detect')
  const navigate = useNavigate()

  const {
    files,
    tasks,
    numberedTasks,
    isUploading,
    error,
    frameInterval,
    modelType,
    processingMode,
    addFiles,
    removeFile,
    clearFiles,
    submitDetect,
    submitFusion,
    setFrameInterval,
    setModelType,
    setProcessingMode,
  } = useUploadStore()

  const { setBatchResults, setCurrentResult } = useResultStore()

  const hasVideo = files.some((f) => f.type === 'video')
  const imageCount = files.filter((f) => f.type === 'image').length

  const modeRules: Record<ProcessingMode, string> = {
    single: '支持 1 张或多张图片；1 张时单张检测，多张时自动走批量单张检测',
    batch: '支持 2~20 张图片，逐张独立检测',
    fusion_images: '支持 2~5 张图片，输出 1 个融合结论',
    fusion_video: '仅支持 1 个 MP4 / WebM 视频',
  }

  const submitLabelMap: Record<ProcessingMode, string> = {
    single: imageCount > 1 ? `开始批量检测 (${imageCount} 张)` : '开始单张检测',
    batch: imageCount > 0 ? `开始批量检测 (${imageCount} 张)` : '开始批量检测',
    fusion_images: imageCount > 0 ? `开始多图融合 (${imageCount} 张)` : '开始多图融合',
    fusion_video: '开始视频融合识别',
  }

  const validateSelection = () => {
    if (files.length === 0) return '请先选择文件'

    if (processingMode === 'single') {
      if (hasVideo) return '单张检测模式仅支持图片'
      if (imageCount < 1 || imageCount > 20) return '单张检测模式支持 1~20 张图片'
    }

    if (processingMode === 'fusion_images') {
      if (hasVideo) return '多图融合模式仅支持图片'
      if (imageCount < 2 || imageCount > 5) return '多图融合需上传 2~5 张图片'
    }

    if (processingMode === 'fusion_video') {
      if (files.length !== 1 || !hasVideo) return '视频融合仅支持上传 1 个视频文件'
    }

    return null
  }

  const handleSubmit = async () => {
    const validationError = validateSelection()
    if (validationError) {
      toast.error('文件与模式不匹配', { description: validationError })
      return
    }

    if (processingMode === 'fusion_images' || processingMode === 'fusion_video') {
      const sessionId = await submitFusion()
      if (sessionId) {
        toast.success('融合识别完成')
        navigate(`/results/fusion/${sessionId}`)
      } else {
        toast.error('识别失败', { description: useUploadStore.getState().error || '请重试' })
      }
    } else {
      const taskId = await submitDetect()
      if (taskId) {
        toast.success('检测完成')
        const task = useUploadStore.getState().tasks.get(taskId)
        if (task?.results?.[0]) {
          setCurrentResult(task.results[0])
          navigate(`/results/${taskId}`)
        }
      } else {
        toast.error('检测失败', { description: useUploadStore.getState().error || '请重试' })
      }
    }
  }

  const handleViewTask = (task: NumberedTask) => {
    if (task.mode === 'fusion_images' || task.mode === 'fusion_video') {
      navigate(`/results/fusion/${task.taskId}`)
    } else {
      const t = tasks.get(task.taskId)
      if (t?.results && t.results.length > 0) {
        setBatchResults(t.results, task.taskId)
        navigate(`/results/${task.taskId}`)
      }
    }
  }

  return (
    <div className="mx-auto max-w-5xl px-4 pt-24 pb-16 sm:px-6 lg:px-8">
      <h1 className="mb-2 text-3xl font-bold">智能检测</h1>
      <p className="mb-8 text-muted-foreground">
        单张检测支持 1~20 张图片自动分流，多图融合与视频融合保留单独入口
      </p>

      <div className="mb-6 flex gap-2">
        <Button
          variant={tab === 'detect' ? 'default' : 'secondary'}
          onClick={() => setTab('detect')}
          className="gap-2"
        >
          <ImageUp className="h-4 w-4" /> 上传检测
        </Button>
        <Button
          variant={tab === 'camera' ? 'default' : 'secondary'}
          onClick={() => setTab('camera')}
          className="gap-2"
        >
          <Camera className="h-4 w-4" /> 实时摄像头
        </Button>
      </div>

      {tab === 'detect' ? (
        <div className="space-y-6">
          <div className="rounded-xl border border-border bg-card p-4">
            <p className="text-sm font-medium mb-3">处理模式</p>
            <div className="grid gap-3 sm:grid-cols-2">
              {MODE_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  onClick={() => setProcessingMode(opt.value)}
                  className={`rounded-lg border p-3 text-left transition-all ${
                    opt.value === 'fusion_video' ? 'sm:col-span-2' : ''
                  } ${
                    processingMode === opt.value
                      ? 'border-primary bg-primary/5 ring-1 ring-primary/30'
                      : 'border-border hover:border-primary/40'
                  }`}
                >
                  <p className="text-sm font-medium">{opt.label}</p>
                  <p className="text-xs text-muted-foreground mt-0.5">{opt.desc}</p>
                </button>
              ))}
            </div>
            <div className="mt-3 rounded-lg border border-primary/20 bg-primary/5 p-3 text-xs text-primary">
              当前模式说明：{modeRules[processingMode]}
            </div>
          </div>

          <div className="rounded-xl border border-border bg-card p-4">
            <p className="text-sm font-medium mb-3">分类模型选择</p>
            <div className="flex gap-3">
              {MODEL_OPTIONS.map((opt) => (
                <button
                  key={opt.value}
                  onClick={() => setModelType(opt.value)}
                  className={`flex-1 rounded-lg border p-3 text-left transition-all ${
                    modelType === opt.value
                      ? 'border-primary bg-primary/5 ring-1 ring-primary/30'
                      : 'border-border hover:border-primary/40'
                  }`}
                >
                  <p className="text-sm font-medium">{opt.label}</p>
                  <p className="text-xs text-muted-foreground mt-0.5">{opt.desc}</p>
                </button>
              ))}
            </div>
          </div>

          <DropZone onFilesSelected={addFiles} disabled={isUploading} />

          {files.length > 0 && (
            <>
              {processingMode === 'fusion_video' && hasVideo && (
                <div className="flex items-center gap-4 rounded-lg border border-border bg-card p-4">
                  <label className="text-sm font-medium whitespace-nowrap">帧间隔</label>
                  <input
                    type="range"
                    min={5}
                    max={120}
                    step={5}
                    value={frameInterval}
                    onChange={(e) => setFrameInterval(Number(e.target.value))}
                    className="flex-1"
                    disabled={isUploading}
                  />
                  <span className="text-sm tabular-nums text-muted-foreground w-20 text-right">
                    每 {frameInterval} 帧
                  </span>
                </div>
              )}

              {processingMode === 'fusion_images' && !hasVideo && (
                <div className="rounded-lg border border-primary/20 bg-primary/5 p-3 text-sm text-primary">
                  已选择 {imageCount} 张图片，将执行多图融合识别（两阶段最优帧选择）
                </div>
              )}

              {processingMode === 'single' && imageCount > 1 && !hasVideo && (
                <div className="rounded-lg border border-primary/20 bg-primary/5 p-3 text-sm text-primary">
                  已选择 {imageCount} 张图片，将逐张独立检测并在结果页中支持翻页查看
                </div>
              )}

              <div className="flex items-center justify-between">
                <span className="text-sm text-muted-foreground">
                  已选择 {files.length} 个文件
                </span>
                <div className="flex gap-2">
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={clearFiles}
                    disabled={isUploading}
                    className="gap-1 text-muted-foreground"
                  >
                    <Trash2 className="h-4 w-4" /> 清空
                  </Button>
                  <Button
                    size="sm"
                    onClick={handleSubmit}
                    disabled={isUploading}
                    className="gap-1"
                  >
                    {isUploading ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <SendHorizonal className="h-4 w-4" />
                    )}
                    {isUploading ? '处理中...' : submitLabelMap[processingMode]}
                  </Button>
                </div>
              </div>

              <FilePreview files={files} onRemove={removeFile} />
            </>
          )}

          {error && (
            <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-4 text-sm text-destructive">
              {error}
            </div>
          )}

          {numberedTasks.length > 0 && (
            <>
              <Separator />
              <div className="space-y-2">
                <h3 className="text-sm font-semibold text-muted-foreground">
                  历史任务 ({numberedTasks.length})
                </h3>
                {numberedTasks.map((t) => (
                  <TaskRow key={t.taskId} task={t} onView={handleViewTask} />
                ))}
              </div>
            </>
          )}
        </div>
      ) : (
        <CameraView />
      )}
    </div>
  )
}
