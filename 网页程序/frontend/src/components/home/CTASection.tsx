import { useRef } from 'react'
import { Link } from 'react-router-dom'
import { useGSAP } from '@/hooks/useGSAP'
import { Button } from '@/components/ui/button'
import { ArrowRight } from 'lucide-react'

export default function CTASection() {
  const sectionRef = useRef<HTMLElement>(null)

  useGSAP(
    (gsap) => {
      gsap.from('[data-cta-content]', {
        y: 40,
        opacity: 0,
        duration: 0.6,
        ease: 'power2.out',
        scrollTrigger: {
          trigger: sectionRef.current,
          start: 'top 80%',
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
      className="relative overflow-hidden px-4 py-24 sm:px-6 lg:px-8"
    >
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(ellipse_at_bottom,_rgba(99,102,241,0.1)_0%,_transparent_60%)]" />

      <div
        data-cta-content
        className="relative z-10 mx-auto flex max-w-2xl flex-col items-center text-center"
      >
        <h2 className="mb-4 text-3xl font-bold sm:text-4xl">立即体验</h2>
        <p className="mb-8 text-muted-foreground">
          上传一张照片，几秒钟内即可获得检测与识别结果
        </p>
        <Button
          asChild
          size="lg"
          className="h-12 px-8 text-base font-semibold shadow-lg shadow-primary/25 transition-transform hover:scale-105"
        >
          <Link to="/upload" className="flex items-center gap-2">
            开始检测 <ArrowRight className="h-5 w-5" />
          </Link>
        </Button>
      </div>
    </section>
  )
}
