import { useRef } from 'react'
import { useGSAP } from '@/hooks/useGSAP'
import { Cpu, Eye, Layout, Zap } from 'lucide-react'

const techs = [
  {
    icon: Eye,
    name: 'YOLO',
    desc: '实时目标检测，精准定位画面中的手机位置',
    color: 'text-emerald-400',
    bg: 'bg-emerald-400/10',
  },
  {
    icon: Cpu,
    name: 'MobileNet / Swin',
    desc: '轻量级分类网络，高效识别手机品牌与型号',
    color: 'text-cyan-400',
    bg: 'bg-cyan-400/10',
  },
  {
    icon: Layout,
    name: 'React',
    desc: '现代化前端框架，流畅的交互与动画体验',
    color: 'text-blue-400',
    bg: 'bg-blue-400/10',
  },
  {
    icon: Zap,
    name: 'FastAPI',
    desc: '高性能 Python 后端，异步处理推理请求',
    color: 'text-amber-400',
    bg: 'bg-amber-400/10',
  },
]

export default function TechSection() {
  const sectionRef = useRef<HTMLElement>(null)

  useGSAP(
    (gsap) => {
      gsap.from('[data-tech-card]', {
        y: 60,
        opacity: 0,
        duration: 0.7,
        stagger: 0.15,
        ease: 'power2.out',
        scrollTrigger: {
          trigger: sectionRef.current,
          start: 'top 75%',
          toggleActions: 'play none none reverse',
        },
      })
    },
    [],
    sectionRef,
  )

  return (
    <section ref={sectionRef} className="px-4 py-24 sm:px-6 lg:px-8">
      <div className="mx-auto max-w-6xl">
        <p className="mb-2 text-center text-sm font-semibold uppercase tracking-widest text-primary">
          Tech Stack
        </p>
        <h2 className="mb-12 text-center text-3xl font-bold sm:text-4xl">
          技术架构
        </h2>

        <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-4">
          {techs.map((tech, i) => {
            const Icon = tech.icon
            return (
              <div
                key={i}
                data-tech-card
                className="group rounded-2xl border border-border bg-card p-6 transition-colors hover:border-primary/40"
              >
                <div
                  className={`mb-4 flex h-12 w-12 items-center justify-center rounded-xl ${tech.bg}`}
                >
                  <Icon className={`h-6 w-6 ${tech.color}`} />
                </div>
                <h3 className="mb-2 text-lg font-bold">{tech.name}</h3>
                <p className="text-sm leading-relaxed text-muted-foreground">
                  {tech.desc}
                </p>
              </div>
            )
          })}
        </div>
      </div>
    </section>
  )
}
