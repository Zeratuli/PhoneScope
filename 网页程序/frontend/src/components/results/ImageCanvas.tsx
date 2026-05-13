import { useEffect, useRef } from 'react'
import type { DetectionItem, ClassificationItem } from '@/types'

interface ImageCanvasProps {
  imageSrc: string
  detections: DetectionItem[]
  classifications: ClassificationItem[]
}

export default function ImageCanvas({
  imageSrc,
  detections,
  classifications,
}: ImageCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    const container = containerRef.current
    if (!canvas || !container) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const img = new Image()
    img.onload = () => {
      const containerW = container.clientWidth
      const scale = containerW / img.width
      const displayH = img.height * scale

      canvas.width = containerW
      canvas.height = displayH
      canvas.style.width = `${containerW}px`
      canvas.style.height = `${displayH}px`

      ctx.drawImage(img, 0, 0, containerW, displayH)

      detections.forEach((det, i) => {
        const { x1, y1, x2, y2 } = det.bbox
        const sx = x1 * scale
        const sy = y1 * scale
        const sw = (x2 - x1) * scale
        const sh = (y2 - y1) * scale

        const color =
          det.confidence > 0.85
            ? '#10b981'
            : det.confidence > 0.7
              ? '#f59e0b'
              : '#ef4444'

        ctx.strokeStyle = color
        ctx.lineWidth = 3
        ctx.strokeRect(sx, sy, sw, sh)

        const cls = classifications[i]
        if (cls) {
          const modelLabel = cls.display_name || cls.model_name
          const label = `${modelLabel} ${(det.confidence * 100).toFixed(0)}%`
          ctx.font = 'bold 14px Inter, sans-serif'
          const textW = ctx.measureText(label).width
          const textH = 20
          ctx.fillStyle = color
          ctx.fillRect(sx, sy - textH - 2, textW + 12, textH + 2)
          ctx.fillStyle = '#ffffff'
          ctx.fillText(label, sx + 6, sy - 6)
        }
      })
    }
    img.src = `data:image/jpeg;base64,${imageSrc}`
  }, [imageSrc, detections, classifications])

  return (
    <div ref={containerRef} className="w-full overflow-hidden rounded-xl border border-border">
      <canvas ref={canvasRef} className="block w-full" />
    </div>
  )
}
