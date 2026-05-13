import { useEffect, useRef, useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { toast } from 'sonner'
import { Camera, CameraOff, Aperture, Film, Loader2, SwitchCamera } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { useUploadStore } from '@/stores/useUploadStore'
import { useResultStore } from '@/stores/useResultStore'

type CaptureMode = 'single' | 'burst'

function blobToFile(blob: Blob, name: string): File {
  return new File([blob], name, { type: 'image/jpeg' })
}

async function captureFrame(video: HTMLVideoElement): Promise<Blob> {
  const canvas = document.createElement('canvas')
  canvas.width = video.videoWidth || 640
  canvas.height = video.videoHeight || 480
  const ctx = canvas.getContext('2d')!
  ctx.drawImage(video, 0, 0)
  return new Promise((resolve, reject) => {
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('capture failed'))), 'image/jpeg', 0.92)
  })
}

export default function CameraView() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const navigate = useNavigate()

  const [available, setAvailable] = useState<boolean | null>(null)
  const [active, setActive] = useState(false)
  const [capturing, setCapturing] = useState(false)
  const [captureMode, setCaptureMode] = useState<CaptureMode>('single')
  const [burstProgress, setBurstProgress] = useState(0)
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([])
  const [deviceIdx, setDeviceIdx] = useState(0)

  const {
    submitDetect,
    submitFusion,
    modelType,
    setModelType,
    setProcessingMode,
    addFiles,
    clearFiles,
  } = useUploadStore()
  const { setCurrentResult } = useResultStore()

  useEffect(() => {
    navigator.mediaDevices
      .enumerateDevices()
      .then((d) => {
        const cams = d.filter((x) => x.kind === 'videoinput')
        setDevices(cams)
        setAvailable(cams.length > 0)
      })
      .catch(() => setAvailable(false))
    return () => stopCamera()
  }, [])

  const startCamera = async (idx = deviceIdx) => {
    stopCamera()
    try {
      const constraints: MediaStreamConstraints = {
        video: devices[idx]?.deviceId
          ? { deviceId: { exact: devices[idx].deviceId } }
          : { facingMode: 'environment' },
      }
      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      streamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      setActive(true)
    } catch {
      toast.error('无法访问摄像头，请检查浏览器权限')
      setActive(false)
    }
  }

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop())
    streamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setActive(false)
  }

  const switchCamera = async () => {
    if (devices.length < 2) return
    const next = (deviceIdx + 1) % devices.length
    setDeviceIdx(next)
    await startCamera(next)
  }

  const handleCaptureSingle = useCallback(async () => {
    if (!videoRef.current || capturing) return
    setCapturing(true)
    try {
      const blob = await captureFrame(videoRef.current)
      const file = blobToFile(blob, `camera_${Date.now()}.jpg`)
      setProcessingMode('single')
      clearFiles()
      addFiles([file])
      const taskId = await submitDetect()
      if (taskId) {
        toast.success('检测完成')
        const task = useUploadStore.getState().tasks.get(taskId)
        if (task?.results?.[0]) {
          setCurrentResult(task.results[0])
          navigate(`/results/${taskId}`)
        }
      } else {
        toast.error('检测失败', { description: useUploadStore.getState().error || '' })
      }
    } catch (e: any) {
      toast.error('拍照失败', { description: e.message })
    } finally {
      setCapturing(false)
    }
  }, [capturing, clearFiles, addFiles, submitDetect, setCurrentResult, navigate, setProcessingMode])

  const handleCaptureBurst = useCallback(async () => {
    if (!videoRef.current || capturing) return
    setCapturing(true)
    setBurstProgress(0)
    const FRAMES = 5
    const INTERVAL_MS = 600
    try {
      const blobs: Blob[] = []
      for (let i = 0; i < FRAMES; i++) {
        const blob = await captureFrame(videoRef.current)
        blobs.push(blob)
        setBurstProgress(i + 1)
        if (i < FRAMES - 1) await new Promise((r) => setTimeout(r, INTERVAL_MS))
      }
      const files = blobs.map((b, i) => blobToFile(b, `burst_${i + 1}.jpg`))
      setProcessingMode('fusion_images')
      clearFiles()
      addFiles(files)
      const sessionId = await submitFusion()
      if (sessionId) {
        toast.success('融合识别完成')
        navigate(`/results/fusion/${sessionId}`)
      } else {
        toast.error('融合识别失败', { description: useUploadStore.getState().error || '' })
      }
    } catch (e: any) {
      toast.error('连拍失败', { description: e.message })
    } finally {
      setCapturing(false)
      setBurstProgress(0)
    }
  }, [capturing, clearFiles, addFiles, submitFusion, navigate, setProcessingMode])

  if (available === null) {
    return (
      <div className="flex items-center justify-center p-12 text-muted-foreground">
        <Loader2 className="h-6 w-6 animate-spin mr-2" /> 检测摄像头...
      </div>
    )
  }

  if (available === false) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-border p-12 text-center">
        <CameraOff className="h-12 w-12 text-muted-foreground" />
        <p className="text-lg font-medium">未检测到可用摄像头</p>
        <p className="text-sm text-muted-foreground">请确认设备已连接摄像头并授予浏览器访问权限</p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* 模型选择 */}
      <div className="rounded-xl border border-border bg-card p-3">
        <p className="text-xs font-medium text-muted-foreground mb-2">分类模型</p>
        <div className="flex gap-2">
          {(['swin', 'mobilenet'] as const).map((m) => (
            <button
              key={m}
              onClick={() => setModelType(m)}
              className={`flex-1 rounded-lg border px-3 py-2 text-sm transition-all ${
                modelType === m
                  ? 'border-primary bg-primary/5 text-primary font-medium'
                  : 'border-border text-muted-foreground hover:border-primary/40'
              }`}
            >
              {m === 'swin' ? 'Swin Transformer' : 'MobileNetV3'}
            </button>
          ))}
        </div>
      </div>

      <div className="relative rounded-2xl overflow-hidden bg-black aspect-video border border-border">
        <video
          ref={videoRef}
          autoPlay
          playsInline
          muted
          className="w-full h-full object-cover"
        />
        {!active && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 bg-black/60">
            <Camera className="h-12 w-12 text-white/60" />
            <Button onClick={() => startCamera()} className="gap-2">
              <Camera className="h-4 w-4" /> 开启摄像头
            </Button>
          </div>
        )}
        {active && (
          <div className="absolute top-3 right-3 flex gap-2">
            {devices.length > 1 && (
              <button
                onClick={switchCamera}
                className="flex items-center justify-center h-8 w-8 rounded-full bg-black/50 text-white hover:bg-black/70 transition"
              >
                <SwitchCamera className="h-4 w-4" />
              </button>
            )}
            <Badge variant="secondary" className="bg-black/50 text-white border-0">
              {devices[deviceIdx]?.label || `摄像头 ${deviceIdx + 1}`}
            </Badge>
          </div>
        )}
        {capturing && captureMode === 'burst' && (
          <div className="absolute bottom-0 left-0 right-0 bg-black/60 p-3 text-center text-white text-sm">
            正在连拍第 {burstProgress} / 5 张...
          </div>
        )}
      </div>

      <div className="rounded-xl border border-border bg-card p-4 space-y-3">
        <p className="text-sm font-medium">拍摄模式</p>
        <div className="flex gap-3">
          <button
            onClick={() => setCaptureMode('single')}
            className={`flex-1 flex items-center gap-2 rounded-lg border p-3 transition-all ${
              captureMode === 'single'
                ? 'border-primary bg-primary/5 ring-1 ring-primary/30'
                : 'border-border hover:border-primary/40'
            }`}
          >
            <Aperture className="h-5 w-5 shrink-0" />
            <div className="text-left">
              <p className="text-sm font-medium">单张拍照</p>
              <p className="text-xs text-muted-foreground">拍1张直接检测</p>
            </div>
          </button>
          <button
            onClick={() => setCaptureMode('burst')}
            className={`flex-1 flex items-center gap-2 rounded-lg border p-3 transition-all ${
              captureMode === 'burst'
                ? 'border-primary bg-primary/5 ring-1 ring-primary/30'
                : 'border-border hover:border-primary/40'
            }`}
          >
            <Film className="h-5 w-5 shrink-0" />
            <div className="text-left">
              <p className="text-sm font-medium">连拍5帧融合</p>
              <p className="text-xs text-muted-foreground">自动截取5张融合识别</p>
            </div>
          </button>
        </div>

        <div className="flex gap-2 pt-1">
          {active ? (
            <>
              <Button
                className="flex-1 gap-2"
                disabled={capturing}
                onClick={captureMode === 'single' ? handleCaptureSingle : handleCaptureBurst}
              >
                {capturing ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : captureMode === 'single' ? (
                  <Aperture className="h-4 w-4" />
                ) : (
                  <Film className="h-4 w-4" />
                )}
                {capturing
                  ? captureMode === 'burst'
                    ? `连拍中 ${burstProgress}/5`
                    : '处理中...'
                  : captureMode === 'single'
                  ? '拍照检测'
                  : '连拍融合识别'}
              </Button>
              <Button variant="outline" onClick={stopCamera} disabled={capturing}>
                关闭
              </Button>
            </>
          ) : (
            <Button className="flex-1 gap-2" onClick={() => startCamera()}>
              <Camera className="h-4 w-4" /> 开启摄像头
            </Button>
          )}
        </div>
      </div>
    </div>
  )
}
