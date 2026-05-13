import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Smartphone } from 'lucide-react'
import type { ClassificationItem } from '@/types'

interface ClassificationCardProps {
  classification: ClassificationItem
  index: number
}

function formatModelName(name: string) {
  const labels: Record<string, string> = {
    HUAWEI_NOVA_10: 'HUAWEI nova 10',
    REDMI_K80_Pro: 'Redmi K80 Pro',
    iPhone_13: 'iPhone 13',
  }
  return labels[name] || name.replace(/_/g, ' ')
}

export default function ClassificationCard({
  classification,
  index,
}: ClassificationCardProps) {
  const displayName =
    classification.display_name || formatModelName(classification.model_name)

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-lg bg-primary/10">
            <Smartphone className="h-5 w-5 text-primary" />
          </div>
          <div>
            <CardTitle className="text-base">
              目标 #{index + 1}
            </CardTitle>
            <p className="text-lg font-bold text-primary">
              {displayName}
            </p>
          </div>
        </div>
      </CardHeader>
      <CardContent>
        <p className="mb-3 text-xs font-semibold uppercase tracking-wider text-muted-foreground">
          Top-K 预测
        </p>
        <div className="space-y-2.5">
          {classification.top_k.map((item, i) => (
            <div key={i} className="space-y-1">
              <div className="flex items-center justify-between text-sm">
                <span className={i === 0 ? 'font-semibold' : 'text-muted-foreground'}>
                  {formatModelName(item.name)}
                </span>
                <span className="font-mono text-xs">
                  {(item.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-muted">
                <div
                  className={`h-full rounded-full transition-all ${
                    i === 0 ? 'bg-primary' : 'bg-muted-foreground/30'
                  }`}
                  style={{ width: `${item.confidence * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
