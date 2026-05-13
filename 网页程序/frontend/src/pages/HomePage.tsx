import { useEffect } from 'react'
import { useAppStore } from '@/stores/useAppStore'
import HeroSection from '@/components/home/HeroSection'
import ShowcaseSection from '@/components/home/ShowcaseSection'
import StatsSection from '@/components/home/StatsSection'
import FeatureSection from '@/components/home/FeatureSection'
import TechSection from '@/components/home/TechSection'
import CTASection from '@/components/home/CTASection'

export default function HomePage() {
  const setNavTransparent = useAppStore((s) => s.setNavTransparent)

  useEffect(() => {
    setNavTransparent(true)
    return () => setNavTransparent(false)
  }, [setNavTransparent])

  return (
    <>
      <HeroSection />
      <ShowcaseSection />
      <StatsSection />
      <FeatureSection />
      <TechSection />
      <CTASection />
    </>
  )
}
