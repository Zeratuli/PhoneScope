import { Progress } from '@/components/ui/progress'
import { Badge } from '@/components/ui/badge'
import { CheckCircle, Loader2, AlertCircle, Clock } from 'lucide-react'
import type { TaskStatusResponse } from '@/types'

const statusConfig = {
  pending: { label: '等待中', icon: Clock, variant: 'secondary' as const },
  processing: { label: '处理中', icon: Loader2, variant: 'default' as const },
  completed: { label: '已完成', icon: CheckCircle, variant: 'default' as const },
  failed: { label: '失败', icon: AlertCircle, variant: 'destructive' as const },
}

interface UploadProgressProps {
  tasks: Map<string, TaskStatusResponse>
  onViewResult: (taskId: string) => void
}

export default function UploadProgress({ tasks, onViewResult }: UploadProgressProps) {
  const taskList = Array.from(tasks.entries())
  if (taskList.length === 0) return null

  return (
    <div className="space-y-3">
      <h3 className="text-sm font-semibold text-muted-foreground">任务列表</h3>
      {taskList.map(([id, task]) => {
        const cfg = statusConfig[task.status]
        const Icon = cfg.icon

        return (
          <div
            key={id}
            className="flex items-center gap-4 rounded-xl border border-border bg-card p-4"
          >
            <Icon
              className={`h-5 w-5 shrink-0 ${
                task.status === 'processing' ? 'animate-spin text-primary' :
                task.status === 'completed' ? 'text-success' :
                task.status === 'failed' ? 'text-destructive' :
                'text-muted-foreground'
              }`}
            />
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-1">
                <span className="truncate text-sm font-medium">
                  {task.current_file || `任务 ${id.slice(0, 8)}`}
                </span>
                <Badge variant={cfg.variant} className="text-xs shrink-0">
                  {cfg.label}
                </Badge>
              </div>
              <Progress value={task.progress * 100} className="h-1.5" />
            </div>
            {task.status === 'completed' && (
              <button
                onClick={() => onViewResult(id)}
                className="shrink-0 text-sm font-medium text-primary hover:underline"
              >
                查看结果
              </button>
            )}
          </div>
        )
      })}
    </div>
  )
}
