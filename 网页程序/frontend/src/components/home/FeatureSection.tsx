import { useRef } from 'react'
import { useGSAP } from '@/hooks/useGSAP'
import { Upload, ScanSearch, Tag, FileText } from 'lucide-react'

const steps = [
  {
    icon: Upload,
    title: '上传图片',
    desc: '支持单张、批量图片或视频上传，拖拽即可开始',
  },
  {
    icon: ScanSearch,
    title: '智能检测',
    desc: 'YOLO 深度学习模型精准定位画面中的智能手机',
  },
  {
    icon: Tag,
    title: '型号识别',
    desc: '分类网络自动识别手机品牌与型号，输出置信度',
  },
  {
    icon: FileText,
    title: '生成报告',
    desc: '一键导出 PDF 检测报告，包含标注图与详细数据',
  },
]

export default function FeatureSection() {
  const sectionRef = useRef<HTMLElement>(null)
  const trackRef = useRef<HTMLDivElement>(null)

  useGSAP(
    (gsap) => {
      const section = sectionRef.current
      const track = trackRef.current
      if (!section || !track) return

      const cards = track.querySelectorAll<HTMLElement>('[data-step-card]')
      const totalScroll = track.scrollWidth - section.offsetWidth

      gsap.to(track, {
        x: -totalScroll,
        ease: 'none',
        scrollTrigger: {
          trigger: section,
          start: 'top top',
          end: () => `+=${totalScroll}`,
          scrub: 1,
          pin: true,
          anticipatePin: 1,
        },
      })

      cards.forEach((card) => {
        gsap.from(card, {
          opacity: 0,
          y: 60,
          duration: 0.6,
          ease: 'power2.out',
          scrollTrigger: {
            trigger: card,
            containerAnimation: gsap.getById?.('featureScroll') ?? undefined,
            start: 'left 80%',
            toggleActions: 'play none none reverse',
          },
        })
      })
    },
    [],
    sectionRef,
  )

  return (
    <section
      ref={sectionRef}
      className="relative overflow-hidden bg-background"
    >
      <div className="flex h-dvh items-center">
        <div ref={trackRef} className="flex gap-8 px-8 sm:px-16 lg:px-24">
          <div className="flex w-[80vw] max-w-md shrink-0 flex-col justify-center pr-8 sm:w-[50vw]">
            <p className="mb-2 text-sm font-semibold uppercase tracking-widest text-primary">
              How it works
            </p>
            <h2 className="text-3xl font-bold sm:text-4xl">工作流程</h2>
            <p className="mt-4 text-muted-foreground">
              四步完成从图片上传到检测报告的全流程
            </p>
          </div>

          {steps.map((step, i) => {
            const Icon = step.icon
            return (
              <div
                key={i}
                data-step-card
                className="flex w-[75vw] max-w-sm shrink-0 flex-col justify-center rounded-2xl border border-border bg-card p-8 shadow-lg sm:w-[40vw]"
              >
                <div className="mb-6 flex h-14 w-14 items-center justify-center rounded-xl bg-primary/10 text-primary">
                  <Icon className="h-7 w-7" />
                </div>
                <div className="mb-2 text-xs font-bold uppercase tracking-widest text-muted-foreground">
                  Step {i + 1}
                </div>
                <h3 className="mb-3 text-xl font-bold">{step.title}</h3>
                <p className="text-sm leading-relaxed text-muted-foreground">
                  {step.desc}
                </p>
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
