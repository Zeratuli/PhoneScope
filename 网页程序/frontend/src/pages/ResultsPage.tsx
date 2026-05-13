import { useEffect } from 'react'
import { useParams, Link, useNavigate } from 'react-router-dom'
import { useResultStore } from '@/stores/useResultStore'
import { useUploadStore } from '@/stores/useUploadStore'
import ImageCanvas from '@/components/results/ImageCanvas'
import DetectionPanel from '@/components/results/DetectionPanel'
import ClassificationCard from '@/components/results/ClassificationCard'
import PhoneInfoPanel from '@/components/results/PhoneInfoPanel'
import ExportBar from '@/components/results/ExportBar'
import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { ArrowLeft, ChevronLeft, ChevronRight } from 'lucide-react'

export default function ResultsPage() {
  const { taskId } = useParams<{ taskId: string }>()
  const navigate = useNavigate()
  const { currentResult, batchResults, setBatchResults, setCurrentResult } = useResultStore()
  const tasks = useUploadStore((s) => s.tasks)

  useEffect(() => {
    if (!currentResult && taskId) {
      const task = tasks.get(taskId)
      if (task?.results && task.results.length > 0) {
        setBatchResults(task.results, taskId)
      }
    }
  }, [taskId, currentResult, tasks, setBatchResults])

  if (!currentResult) {
    return (
      <div className="flex min-h-[60vh] flex-col items-center justify-center gap-4 pt-20">
        <p className="text-lg text-muted-foreground">暂无检测结果</p>
        <Button asChild variant="secondary">
          <Link to="/upload">返回上传</Link>
        </Button>
      </div>
    )
  }

  const currentIdx = batchResults.findIndex(
    (r) => r.image_id === currentResult.image_id,
  )
  const hasPrev = currentIdx > 0
  const hasNext = currentIdx < batchResults.length - 1

  return (
    <div className="mx-auto max-w-7xl px-4 pt-24 pb-16 sm:px-6 lg:px-8">
      <div className="mb-6 flex flex-wrap items-center gap-4">
        <Button
          variant="ghost"
          size="sm"
          onClick={() => navigate('/upload')}
          className="gap-1"
        >
          <ArrowLeft className="h-4 w-4" /> 返回
        </Button>

        {batchResults.length > 1 && (
          <div className="flex items-center gap-2 ml-auto">
            <Button
              variant="outline"
              size="icon"
              disabled={!hasPrev}
              onClick={() => setCurrentResult(batchResults[currentIdx - 1])}
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <span className="text-sm text-muted-foreground">
              {currentIdx + 1} / {batchResults.length}
            </span>
            <Button
              variant="outline"
              size="icon"
              disabled={!hasNext}
              onClick={() => setCurrentResult(batchResults[currentIdx + 1])}
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        )}
      </div>

      <div className="grid gap-6 lg:grid-cols-[1fr_380px]">
        <div className="space-y-6">
          <ImageCanvas
            imageSrc={currentResult.annotated_image_base64}
            detections={currentResult.detections}
            classifications={currentResult.classifications}
          />
          <DetectionPanel detections={currentResult.detections} />
        </div>

        <div className="space-y-6">
          {currentResult.classifications.map((cls, i) => (
            <ClassificationCard key={i} classification={cls} index={i} />
          ))}
          {currentResult.classifications.map((cls, i) =>
            cls.phone_spec ? (
              <PhoneInfoPanel
                key={`spec-${i}`}
                spec={cls.phone_spec}
                modelName={cls.display_name || cls.model_name}
              />
            ) : null,
          )}
        </div>
      </div>

      <Separator className="my-8" />
      <ExportBar taskId={taskId || ''} />
    </div>
  )
}
