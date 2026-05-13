import { useLayoutEffect, useRef, type RefObject } from 'react'
import gsap from 'gsap'
import { ScrollTrigger } from 'gsap/ScrollTrigger'

gsap.registerPlugin(ScrollTrigger)

export function useGSAP(
  callback: (gsapInstance: typeof gsap) => void | (() => void),
  deps: unknown[] = [],
  scope?: RefObject<HTMLElement | null>,
) {
  const cleanup = useRef<(() => void) | void>(undefined)

  useLayoutEffect(() => {
    const ctx = gsap.context(() => {
      cleanup.current = callback(gsap)
    }, scope?.current ?? undefined)

    return () => {
      if (typeof cleanup.current === 'function') cleanup.current()
      ctx.revert()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
}

export { gsap, ScrollTrigger }
