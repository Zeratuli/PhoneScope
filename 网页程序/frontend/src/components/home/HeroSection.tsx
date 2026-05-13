import { useRef } from 'react'
import { Link } from 'react-router-dom'
import { useGSAP, ScrollTrigger } from '@/hooks/useGSAP'
import { useAppStore } from '@/stores/useAppStore'
import { Button } from '@/components/ui/button'
import { ChevronDown } from 'lucide-react'

const TITLE_CHARS = '智能手机识别系统'.split('')

export default function HeroSection() {
  const containerRef = useRef<HTMLElement>(null)
  const setNavTransparent = useAppStore((s) => s.setNavTransparent)

  useGSAP(
    (gsap) => {
      const tl = gsap.timeline({ defaults: { ease: 'power3.out' } })

      tl.from('[data-hero-char]', {
        y: 50,
        opacity: 0,
        duration: 0.6,
        stagger: 0.04,
      })
        .from('[data-hero-sub]', { y: 20, opacity: 0, duration: 0.5 }, '-=0.2')
        .from(
          '[data-hero-cta]',
          { y: 30, opacity: 0, scale: 0.95, duration: 0.5, ease: 'back.out(1.7)' },
          '-=0.2',
        )
        .from('[data-hero-scroll]', { opacity: 0, duration: 0.4 }, '-=0.1')

      gsap.to('[data-hero-scroll]', {
        y: 10,
        duration: 1.5,
        ease: 'sine.inOut',
        repeat: -1,
        yoyo: true,
      })

      ScrollTrigger.create({
        trigger: containerRef.current,
        start: 'top top',
        end: 'bottom top',
        onLeave: () => setNavTransparent(false),
        onEnterBack: () => setNavTransparent(true),
      })
    },
    [],
    containerRef,
  )

  return (
    <section
      ref={containerRef}
      className="relative flex h-dvh flex-col items-center justify-center overflow-hidden px-4"
    >
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_center,_rgba(99,102,241,0.15)_0%,_transparent_70%)]" />
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_80%_20%,_rgba(34,211,238,0.08)_0%,_transparent_50%)]" />

      <h1 className="relative z-10 mb-6 text-center text-5xl font-extrabold tracking-tight sm:text-6xl lg:text-7xl">
        {TITLE_CHARS.map((char, i) => (
          <span key={i} data-hero-char className="inline-block">
            {char}
          </span>
        ))}
      </h1>

      <p
        data-hero-sub
        className="relative z-10 mb-10 max-w-2xl text-center text-lg text-muted-foreground sm:text-xl"
      >
        基于 YOLO 目标检测与深度学习分类，实现智能手机的精准定位与型号识别
      </p>

      <div data-hero-cta className="relative z-10">
        <Button
          asChild
          size="lg"
          className="h-12 px-8 text-base font-semibold shadow-lg shadow-primary/25 transition-transform hover:scale-105"
        >
          <Link to="/upload">开始使用</Link>
        </Button>
      </div>

      <div
        data-hero-scroll
        className="absolute bottom-8 left-1/2 z-10 -translate-x-1/2 text-muted-foreground"
      >
        <ChevronDown className="h-8 w-8" />
      </div>
    </section>
  )
}
