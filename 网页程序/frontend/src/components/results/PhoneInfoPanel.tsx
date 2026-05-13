import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import {
  Factory, Tag, Smartphone, Calendar, Monitor,
  Cpu, MemoryStick, HardDrive, Camera, Battery,
  AppWindow, Ruler, Weight, Palette,
} from 'lucide-react'
import type { PhoneSpec } from '@/types'

interface PhoneInfoPanelProps {
  spec: PhoneSpec
  modelName: string
}

const specRows = (s: PhoneSpec) => [
  { icon: Factory, label: '制造商', value: s.manufacturer },
  { icon: Tag, label: '品牌', value: s.brand },
  { icon: Smartphone, label: '型号', value: s.model },
  { icon: Calendar, label: '发布日期', value: s.released },
  { icon: Monitor, label: '屏幕', value: s.screen },
  { icon: Cpu, label: '处理器', value: s.processor },
  { icon: MemoryStick, label: '运行内存', value: s.ram },
  { icon: HardDrive, label: '存储', value: s.storage },
  { icon: Camera, label: '后置相机', value: s.rear_camera },
  { icon: Camera, label: '前置相机', value: s.front_camera },
  { icon: Battery, label: '电池', value: s.battery },
  { icon: AppWindow, label: '操作系统', value: s.os },
  { icon: Ruler, label: '尺寸', value: s.dimensions },
  { icon: Weight, label: '重量', value: s.weight },
  { icon: Palette, label: '配色', value: s.colors },
]

export default function PhoneInfoPanel({ spec, modelName }: PhoneInfoPanelProps) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="text-base flex items-center gap-2">
          <Smartphone className="h-4 w-4 text-primary" />
          {modelName.replace(/_/g, ' ')} 详细参数
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-2">
        {specRows(spec).map((row, i) => {
          const Icon = row.icon
          return (
            <div
              key={i}
              className="flex items-start gap-3 rounded-lg px-2 py-1.5 odd:bg-muted/30"
            >
              <Icon className="mt-0.5 h-3.5 w-3.5 shrink-0 text-muted-foreground" />
              <div className="min-w-0">
                <span className="text-xs text-muted-foreground">{row.label}</span>
                <p className="text-sm font-medium leading-snug">{row.value || '—'}</p>
              </div>
            </div>
          )
        })}
      </CardContent>
    </Card>
  )
}
