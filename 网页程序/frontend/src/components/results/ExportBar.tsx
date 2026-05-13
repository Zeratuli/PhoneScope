import { Button } from '@/components/ui/button'
import { useResultStore } from '@/stores/useResultStore'
import { FileDown, FileImage, Loader2 } from 'lucide-react'

interface ExportBarProps {
  taskId: string
}

export default function ExportBar({ taskId }: ExportBarProps) {
  const {
    isExporting,
    isExportingImages,
    downloadPDF,
    downloadImagesZip,
    batchResults,
  } = useResultStore()

  const disabled = !taskId || batchResults.length === 0

  return (
    <div className="flex flex-wrap gap-2">
      <Button
        onClick={() => downloadPDF(taskId)}
        disabled={disabled || isExporting}
        className="gap-2"
      >
        {isExporting ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <FileDown className="h-4 w-4" />
        )}
        {isExporting ? '生成中...' : '导出 PDF 报告'}
      </Button>
      <Button
        variant="secondary"
        onClick={() => downloadImagesZip(taskId)}
        disabled={disabled || isExportingImages}
        className="gap-2"
      >
        {isExportingImages ? (
          <Loader2 className="h-4 w-4 animate-spin" />
        ) : (
          <FileImage className="h-4 w-4" />
        )}
        {isExportingImages ? '打包中...' : '导出标注图 (ZIP)'}
      </Button>
    </div>
  )
}
