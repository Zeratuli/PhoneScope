import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, ImagePlus } from 'lucide-react'
import { cn } from '@/lib/utils'

interface DropZoneProps {
  onFilesSelected: (files: File[]) => void
  disabled?: boolean
}

const ACCEPT = {
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/png': ['.png'],
  'image/webp': ['.webp'],
  'image/bmp': ['.bmp'],
  'video/mp4': ['.mp4'],
  'video/webm': ['.webm'],
}

export default function DropZone({ onFilesSelected, disabled }: DropZoneProps) {
  const onDrop = useCallback(
    (accepted: File[]) => {
      if (accepted.length > 0) onFilesSelected(accepted)
    },
    [onFilesSelected],
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPT,
    maxSize: 50 * 1024 * 1024,
    disabled,
    multiple: true,
  })

  return (
    <div
      {...getRootProps()}
      className={cn(
        'relative cursor-pointer rounded-2xl border-2 border-dashed p-12 text-center transition-all',
        isDragActive
          ? 'border-primary bg-primary/5 scale-[1.02]'
          : 'border-border hover:border-primary/50 hover:bg-muted/30',
        disabled && 'pointer-events-none opacity-50',
      )}
    >
      <input {...getInputProps()} />
      <div className="flex flex-col items-center gap-4">
        {isDragActive ? (
          <>
            <ImagePlus className="h-12 w-12 text-primary" />
            <p className="text-lg font-medium text-primary">松开以添加文件</p>
          </>
        ) : (
          <>
            <Upload className="h-12 w-12 text-muted-foreground" />
            <div>
              <p className="text-lg font-medium">
                拖拽文件到此处，或点击选择
              </p>
              <p className="mt-1 text-sm text-muted-foreground">
                支持 JPG / PNG / WebP / BMP / MP4 / WebM，单文件最大 50MB
              </p>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
