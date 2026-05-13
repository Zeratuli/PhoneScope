import { Link } from 'react-router-dom'
import { useEffect } from 'react'
import { useUploadStore } from '@/stores/useUploadStore'
import { useResultStore } from '@/stores/useResultStore'
import PhoneInfoPanel from '@/components/results/PhoneInfoPanel'
import ExportBar from '@/components/results/ExportBar'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { ArrowLeft, CheckCircle, XCircle, Star } from 'lucide-react'

function formatModelName(name: string) {
  const labels: Record<string, string> = {
    HUAWEI_NOVA_10: 'HUAWEI nova 10',
    REDMI_K80_Pro: 'Redmi K80 Pro',
    iPhone_13: 'iPhone 13',
  }
  return labels[name] || name.replace(/_/g, ' ')
}

export default function FusionResultsPage() {
  const fusionResult = useUploadStore((s) => s.fusionResult)
  const setBatchResults = useResultStore((s) => s.setBatchResults)

  // 让 ExportBar 的 disabled 判定生效（基于 batchResults.length > 0）
  useEffect(() => {
    if (fusionResult) {
      // 以融合的单元素占位 —— ExportBar 只用它来判 disabled
      setBatchResults([fusionResult as any], fusionResult.session_id)
    }
  }, [fusionResult, setBatchResults])

  if (!fusionResult) {
    return (
      <div className="flex min-h-[60vh] flex-col items-center justify-center gap-4 pt-20">
        <p className="text-lg text-muted-foreground">暂无融合识别结果</p>
        <Button asChild variant="secondary">
          <Link to="/upload">返回上传</Link>
        </Button>
      </div>
    )
  }

  const r = fusionResult

  return (
    <div className="mx-auto max-w-7xl px-4 pt-24 pb-16 sm:px-6 lg:px-8">
      <div className="mb-6 flex flex-wrap items-center gap-3">
        <Button variant="ghost" size="sm" asChild className="gap-1">
          <Link to="/upload">
            <ArrowLeft className="h-4 w-4" /> 返回
          </Link>
        </Button>
        <div className="ml-auto">
          <ExportBar taskId={r.session_id} />
        </div>
      </div>

      <div className="rounded-2xl border border-border bg-card p-6 mb-8">
        <h2 className="text-xl font-bold mb-4">融合识别结论</h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <div>
            <p className="text-xs text-muted-foreground mb-1">最终型号</p>
            <p className="text-lg font-semibold">
              {r.final_display_name || r.final_model_name?.replace(/_/g, ' ') || '未检测到手机'}
            </p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground mb-1">置信度</p>
            <p className="text-lg font-semibold">
              {r.final_confidence != null ? `${(r.final_confidence * 100).toFixed(1)}%` : '—'}
            </p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground mb-1">有效帧 / 总帧</p>
            <p className="text-lg font-semibold">
              {r.valid_frames} / {r.total_frames}
            </p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground mb-1">处理耗时</p>
            <p className="text-lg font-semibold">{r.processing_time_ms.toFixed(0)} ms</p>
          </div>
        </div>

        {r.final_top_k && r.final_top_k.length > 0 && (
          <div className="mt-4">
            <p className="text-xs text-muted-foreground mb-2">Top-K 分类概率</p>
            <div className="flex flex-wrap gap-2">
              {r.final_top_k.map((t, i) => (
                <Badge key={i} variant={i === 0 ? 'default' : 'secondary'}>
                  {formatModelName(t.name)} {(t.confidence * 100).toFixed(1)}%
                </Badge>
              ))}
            </div>
          </div>
        )}

        {r.best_crop_base64 && (
          <div className="mt-4">
            <p className="text-xs text-muted-foreground mb-2">最优裁剪区域</p>
            <img
              src={`data:image/jpeg;base64,${r.best_crop_base64}`}
              alt="Best crop"
              className="max-h-48 rounded-lg border border-border"
            />
          </div>
        )}
      </div>

      {r.final_phone_spec && r.final_model_name && (
        <>
          <PhoneInfoPanel
            spec={r.final_phone_spec}
            modelName={r.final_display_name || r.final_model_name}
          />
          <Separator className="my-8" />
        </>
      )}

      <h3 className="text-lg font-semibold mb-4">帧证据详情 ({r.mode === 'video' ? '视频抽帧' : '多图上传'})</h3>
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        {r.frames.map((frame) => (
          <div
            key={frame.frame_index}
            className={`rounded-xl border overflow-hidden ${
              frame.frame_index === r.best_frame_index
                ? 'border-primary ring-2 ring-primary/30'
                : 'border-border'
            }`}
          >
            {frame.annotated_image_base64 && (
              <img
                src={`data:image/jpeg;base64,${frame.annotated_image_base64}`}
                alt={frame.filename}
                className="w-full aspect-video object-cover"
              />
            )}
            <div className="p-3 space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium truncate">{frame.filename}</span>
                {frame.frame_index === r.best_frame_index && (
                  <Badge variant="default" className="gap-1 shrink-0">
                    <Star className="h-3 w-3" /> 最优帧
                  </Badge>
                )}
              </div>
              <div className="flex items-center gap-3 text-xs text-muted-foreground">
                <span className="flex items-center gap-1">
                  {frame.is_valid ? (
                    <CheckCircle className="h-3 w-3 text-green-500" />
                  ) : (
                    <XCircle className="h-3 w-3 text-red-400" />
                  )}
                  {frame.is_valid ? '有效检测' : '无有效检测'}
                </span>
                <span>质量分: {frame.quality_score.toFixed(4)}</span>
                <span>检测数: {frame.detections.length}</span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
