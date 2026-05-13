import { useRef, useState } from 'react'
import { useGSAP, ScrollTrigger } from '@/hooks/useGSAP'

const stats = [
  { value: 99.5, suffix: '%', label: 'mAP@0.5 检测精度', decimals: 1 },
  { value: 97, suffix: '%', label: 'mAP@0.5:0.95 定位精度', decimals: 0 },
  { value: 26, suffix: 'ms', label: '单帧推理耗时', decimals: 0 },
  { value: 3, suffix: '+', label: '可识别手机型号', decimals: 0 },
]

function AnimatedNumber({
  value,
  suffix,
  decimals,
  active,
}: {
  value: number
  suffix: string
  decimals: number
  active: boolean
}) {
  const [display, setDisplay] = useState(0)
  const ref = useRef<HTMLSpanElement>(null)

  useGSAP(
    (gsap) => {
      if (!active) return
      const obj = { val: 0 }
      gsap.to(obj, {
        val: value,
        duration: 2,
        ease: 'power2.out',
        onUpdate: () => setDisplay(obj.val),
      })
    },
    [active],
  )

  return (
    <span ref={ref} className="text-4xl font-extrabold tabular-nums sm:text-5xl">
      {display.toFixed(decimals)}
      <span className="text-2xl text-primary sm:text-3xl">{suffix}</span>
    </span>
  )
}

export default function StatsSection() {
  const sectionRef = useRef<HTMLElement>(null)
  const [active, setActive] = useState(false)

  useGSAP(
    (gsap) => {
      ScrollTrigger.create({
        trigger: sectionRef.current,
        start: 'top 75%',
        once: true,
        onEnter: () => setActive(true),
      })

      gsap.from('[data-stat-item]', {
        y: 40,
        opacity: 0,
        duration: 0.6,
        stagger: 0.12,
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
    <section
      ref={sectionRef}
      className="relative overflow-hidden border-y border-border bg-card/50 px-4 py-20 sm:px-6 lg:px-8"
    >
      <div className="mx-auto max-w-5xl">
        <p className="mb-2 text-center text-sm font-semibold uppercase tracking-widest text-primary">
          Performance
        </p>
        <h2 className="mb-12 text-center text-3xl font-bold sm:text-4xl">
          核心指标
        </h2>

        <div className="grid grid-cols-2 gap-8 lg:grid-cols-4">
          {stats.map((stat, i) => (
            <div
              key={i}
              data-stat-item
              className="flex flex-col items-center gap-2 text-center"
            >
              <AnimatedNumber
                value={stat.value}
                suffix={stat.suffix}
                decimals={stat.decimals}
                active={active}
              />
              <span className="text-sm text-muted-foreground">{stat.label}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
