"use client"
import { useEffect, useState } from 'react'
import { ThemeToggle } from './ThemeToggle'

const LINKS: [string, string][] = [['Predict', 'predict'], ['Model', 'model'], ['Limits', 'limits']]

export function SiteNav() {
  const [active, setActive] = useState('top')
  const [showBrand, setShowBrand] = useState(false)

  useEffect(() => {
    // 1. Observe sections for nav active indicator
    const els = ['top', 'predict', 'model', 'limits'].map((i) => document.getElementById(i)).filter(Boolean) as HTMLElement[]
    const io = new IntersectionObserver((es) => es.forEach((e) => e.isIntersecting && setActive(e.target.id)), { rootMargin: '-45% 0px -50% 0px' })
    els.forEach((e) => io.observe(e))

    // 2. Observe giant hero wordmark: fade in nav wordmark only when hero wordmark scrolls out of view
    const wm = document.getElementById('hero-wordmark')
    let wmIo: IntersectionObserver | null = null
    if (wm) {
      wmIo = new IntersectionObserver(([entry]) => {
        setShowBrand(!entry.isIntersecting)
      }, { rootMargin: '-56px 0px 0px 0px', threshold: 0 })
      wmIo.observe(wm)
    }

    return () => {
      io.disconnect()
      if (wmIo) wmIo.disconnect()
    }
  }, [])

  return (
    <header className="sticky top-0 z-40 flex h-14 items-center justify-between border-b border-line bg-canvas/80 px-3 backdrop-blur-md sm:px-5">
      <a
        href="#top"
        className={`font-display text-xl font-extrabold tracking-[-0.04em] text-ink transition-opacity duration-200 ${
          showBrand ? 'opacity-100' : 'pointer-events-none opacity-0'
        }`}
      >
        triage
      </a>
      <nav aria-label="Primary" className="flex items-center gap-1 text-sm text-ink sm:gap-2">
        {LINKS.map(([l, id]) => (
          <a key={id} href={`#${id}`} aria-current={active === id ? 'true' : undefined} className={`hidden rounded-full px-3 py-2 transition-colors hover:bg-hover md:block ${active === id ? 'bg-hover font-medium' : 'text-muted'}`}>{l}</a>
        ))}
        <ThemeToggle />
        <a href="#predict" className="inline-flex h-9 items-center rounded-full bg-ink px-5 text-sm font-medium text-canvas transition-opacity hover:opacity-85">Try a patient</a>
      </nav>
    </header>
  )
}
