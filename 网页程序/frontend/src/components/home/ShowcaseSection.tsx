import { useRef, useEffect, useCallback } from 'react'
import { useGSAP } from '@/hooks/useGSAP'
import { Smartphone } from 'lucide-react'

const PARTICLE_COUNT = 80

function createParticles(container: HTMLDivElement) {
  const particles: HTMLDivElement[] = []
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    const el = document.createElement('div')
    const size = Math.random() * 4 + 2
    el.className = 'absolute rounded-full pointer-events-none'
    el.style.width = `${size}px`
    el.style.height = `${size}px`
    el.style.background =
      i % 3 === 0
        ? 'rgba(99,102,241,0.5)'
        : i % 3 === 1
          ? 'rgba(34,211,238,0.4)'
          : 'rgba(139,92,246,0.3)'
    el.style.left = `${Math.random() * 100}%`
    el.style.top = `${Math.random() * 100}%`
    container.appendChild(el)
    particles.push(el)
  }
  return particles
}

export default function ShowcaseSection() {
  const sectionRef = useRef<HTMLElement>(null)
  const phoneRef = useRef<HTMLDivElement>(null)
  const particleContainerRef = useRef<HTMLDivElement>(null)
  const particlesRef = useRef<HTMLDivElement[]>([])
  const mouseRef = useRef({ x: 0, y: 0 })

  useEffect(() => {
    const container = particleContainerRef.current
    if (!container) return
    particlesRef.current = createParticles(container)
    return () => {
      particlesRef.current.forEach((p) => p.remove())
      particlesRef.current = []
    }
  }, [])

  useGSAP(
    (gsap) => {
      gsap.from('[data-showcase-text]', {
        x: -60,
        opacity: 0,
        duration: 0.8,
        ease: 'power2.out',
        scrollTrigger: {
          trigger: sectionRef.current,
          start: 'top 70%',
          toggleActions: 'play none none reverse',
        },
      })

      const phone = phoneRef.current
      if (!phone) return

      gsap.fromTo(
        phone,
        { rotateY: -20, rotateX: 8, opacity: 0, scale: 0.85 },
        {
          rotateY: 0,
          rotateX: 0,
          opacity: 1,
          scale: 1,
          duration: 1,
          ease: 'power3.out',
          scrollTrigger: {
            trigger: sectionRef.current,
            start: 'top 65%',
            toggleActions: 'play none none reverse',
          },
        },
      )

      particlesRef.current.forEach((p, i) => {
        gsap.to(p, {
          x: `random(-30, 30)`,
          y: `random(-30, 30)`,
          duration: gsap.utils.random(3, 6),
          ease: 'sine.inOut',
          repeat: -1,
          yoyo: true,
          delay: i * 0.05,
        })
      })

      gsap.to('[data-scan-line]', {
        y: 280,
        duration: 2,
        ease: 'power1.inOut',
        repeat: -1,
        yoyo: true,
      })
    },
    [],
    sectionRef,
  )

  const handleMouseMove = useCallback(
    (e: React.MouseEvent<HTMLDivElement>) => {
      const rect = e.currentTarget.getBoundingClientRect()
      const cx = rect.left + rect.width / 2
      const cy = rect.top + rect.height / 2
      const mx = (e.clientX - cx) / (rect.width / 2)
      const my = (e.clientY - cy) / (rect.height / 2)
      mouseRef.current = { x: mx, y: my }

      const phone = phoneRef.current
      if (phone) {
        phone.style.transform = `rotateY(${mx * 15}deg) rotateX(${-my * 10}deg)`
      }

      particlesRef.current.forEach((p) => {
        const px = parseFloat(p.style.left) / 100
        const py = parseFloat(p.style.top) / 100
        const dx = (mx - (px - 0.5) * 2) * 25
        const dy = (my - (py - 0.5) * 2) * 25
        const dist = Math.sqrt(dx * dx + dy * dy)
        if (dist < 40) {
          const pushX = dx > 0 ? -15 : 15
          const pushY = dy > 0 ? -15 : 15
          p.style.transition = 'transform 0.4s ease-out'
          p.style.transform = `translate(${pushX}px, ${pushY}px)`
        }
      })
    },
    [],
  )

  const handleMouseLeave = useCallback(() => {
    const phone = phoneRef.current
    if (phone) {
      phone.style.transition = 'transform 0.6s ease-out'
      phone.style.transform = 'rotateY(0deg) rotateX(0deg)'
      setTimeout(() => {
        if (phone) phone.style.transition = ''
      }, 600)
    }
    particlesRef.current.forEach((p) => {
      p.style.transition = 'transform 0.8s ease-out'
      p.style.transform = ''
    })
  }, [])

  return (
    <section
      ref={sectionRef}
      className="relative overflow-hidden px-4 py-24 sm:px-6 lg:px-8"
    >
      <div className="mx-auto grid max-w-6xl gap-12 lg:grid-cols-2 lg:gap-16 items-center">
        <div data-showcase-text className="space-y-6">
          <p className="text-sm font-semibold uppercase tracking-widest text-primary">
            Real-Time Detection
          </p>
          <h2 className="text-3xl font-bold sm:text-4xl">
            实时检测与识别
          </h2>
          <p className="text-lg leading-relaxed text-muted-foreground">
            将手机对准目标，系统即刻完成检测定位与型号识别。
            YOLO 深度学习模型在毫秒级时间内精准定位手机位置，
            随后分类网络自动判别手机品牌与型号。
          </p>
          <div className="space-y-4">
            {[
              { label: '检测精度', value: 'mAP@0.5 达到 99.5%' },
              { label: '推理速度', value: '单帧 < 30ms（GPU）' },
              { label: '支持型号', value: '华为 / 红米 / iPhone 系列' },
            ].map((item, i) => (
              <div key={i} className="flex items-start gap-3">
                <div className="mt-1 h-2 w-2 shrink-0 rounded-full bg-primary" />
                <div>
                  <span className="font-semibold">{item.label}</span>
                  <span className="text-muted-foreground"> — {item.value}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div
          className="relative flex justify-center"
          style={{ perspective: '1000px' }}
          onMouseMove={handleMouseMove}
          onMouseLeave={handleMouseLeave}
        >
          <div
            ref={particleContainerRef}
            className="pointer-events-none absolute inset-0"
          />

          <div
            ref={phoneRef}
            className="relative cursor-pointer"
            style={{ transformStyle: 'preserve-3d', willChange: 'transform' }}
          >
            <div className="relative h-[420px] w-[220px] rounded-[2.5rem] border-[3px] border-border bg-gradient-to-b from-card to-secondary shadow-2xl shadow-primary/10 overflow-hidden">
              <div className="absolute inset-x-0 top-0 flex justify-center pt-2">
                <div className="h-6 w-20 rounded-full bg-background/80" />
              </div>

              <div className="absolute inset-4 top-12 rounded-xl bg-background/50 overflow-hidden">
                <div className="flex h-full flex-col items-center justify-center gap-3 p-4">
                  <Smartphone className="h-10 w-10 text-primary/60" />
                  <span className="text-xs font-medium text-muted-foreground">
                    PhoneScope
                  </span>

                  <div className="mt-4 w-full space-y-2">
                    <div className="flex items-center gap-2 rounded-lg bg-primary/10 px-3 py-2">
                      <div className="h-2 w-2 rounded-full bg-emerald-400" />
                      <span className="text-[10px] font-medium text-foreground">
                        iPhone 13 — 96.2%
                      </span>
                    </div>
                    <div className="flex items-center gap-2 rounded-lg bg-primary/5 px-3 py-2">
                      <div className="h-2 w-2 rounded-full bg-cyan-400" />
                      <span className="text-[10px] font-medium text-muted-foreground">
                        REDMI K80 Pro — 2.1%
                      </span>
                    </div>
                    <div className="flex items-center gap-2 rounded-lg bg-primary/5 px-3 py-2">
                      <div className="h-2 w-2 rounded-full bg-amber-400" />
                      <span className="text-[10px] font-medium text-muted-foreground">
                        HUAWEI Nova 10 — 1.7%
                      </span>
                    </div>
                  </div>
                </div>

                <div
                  data-scan-line
                  className="absolute left-0 right-0 h-[2px] bg-gradient-to-r from-transparent via-primary to-transparent opacity-60"
                  style={{ top: 0 }}
                />
              </div>

              <div className="absolute inset-x-0 bottom-2 flex justify-center">
                <div className="h-1 w-24 rounded-full bg-muted-foreground/30" />
              </div>
            </div>

          </div>
        </div>
      </div>
    </section>
  )
}
