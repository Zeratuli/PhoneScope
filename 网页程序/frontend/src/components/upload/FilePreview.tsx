import { X, Film, Image as ImageIcon } from 'lucide-react'
import type { FileWithPreview } from '@/types'

interface FilePreviewProps {
  files: FileWithPreview[]
  onRemove: (id: string) => void
}

export default function FilePreview({ files, onRemove }: FilePreviewProps) {
  if (files.length === 0) return null

  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4">
      {files.map((f) => (
        <div
          key={f.id}
          className="group relative aspect-square overflow-hidden rounded-xl border border-border bg-card"
        >
          {f.type === 'image' && f.preview ? (
            <img
              src={f.preview}
              alt={f.file.name}
              className="h-full w-full object-cover"
            />
          ) : (
            <div className="flex h-full w-full flex-col items-center justify-center gap-2 text-muted-foreground">
              <Film className="h-8 w-8" />
              <span className="text-xs">{f.file.name}</span>
            </div>
          )}

          <button
            onClick={() => onRemove(f.id)}
            className="absolute right-1.5 top-1.5 flex h-7 w-7 items-center justify-center rounded-full bg-background/80 text-muted-foreground opacity-0 backdrop-blur transition-opacity group-hover:opacity-100 hover:text-destructive"
          >
            <X className="h-4 w-4" />
          </button>

          <div className="absolute bottom-0 left-0 right-0 bg-gradient-to-t from-black/60 to-transparent px-2 py-1.5">
            <div className="flex items-center gap-1">
              {f.type === 'image' ? (
                <ImageIcon className="h-3 w-3 text-white/70" />
              ) : (
                <Film className="h-3 w-3 text-white/70" />
              )}
              <span className="truncate text-xs text-white/90">
                {f.file.name}
              </span>
            </div>
          </div>
        </div>
      ))}
    </div>
  )
}
