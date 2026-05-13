import { Badge } from '@/components/ui/badge'
import { Target } from 'lucide-react'
import type { DetectionItem } from '@/types'

interface DetectionPanelProps {
  detections: DetectionItem[]
}

export default function DetectionPanel({ detections }: DetectionPanelProps) {
  return (
    <div className="rounded-xl border border-border bg-card p-5">
      <div className="mb-4 flex items-center gap-2">
        <Target className="h-5 w-5 text-primary" />
        <h3 className="text-sm font-semibold">检测结果</h3>
        <Badge variant="secondary" className="ml-auto">
          {detections.length} 个目标
        </Badge>
      </div>

      <div className="space-y-3">
        {detections.map((det, i) => (
          <div
            key={i}
            className="flex items-center justify-between rounded-lg bg-muted/50 px-4 py-3"
          >
            <div className="flex items-center gap-3">
              <span className="flex h-6 w-6 items-center justify-center rounded-full bg-primary/10 text-xs font-bold text-primary">
                {i + 1}
              </span>
              <span className="text-sm font-medium">{det.label}</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="h-1.5 w-16 overflow-hidden rounded-full bg-muted">
                <div
                  className="h-full rounded-full bg-primary transition-all"
                  style={{ width: `${det.confidence * 100}%` }}
                />
              </div>
              <span className="text-xs font-mono text-muted-foreground w-12 text-right">
                {(det.confidence * 100).toFixed(1)}%
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
