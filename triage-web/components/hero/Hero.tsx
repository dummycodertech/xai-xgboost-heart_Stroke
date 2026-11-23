'use client'
import dynamic from 'next/dynamic'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useHeroFeatures } from '@/lib/api'
import { HudOverlay } from './HudOverlay'
import { useTelemetry } from '@/lib/telemetry'
import { createBus, type Mode } from './tracker'

const HeartCanvas = dynamic(() => import('./HeartCanvas'), { ssr: false })

const Arrow = () => (
  <svg width="16" height="11" viewBox="0 0 16 11" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden><path d="M0 5.5h14M9.5 1l4.5 4.5L9.5 10" /></svg>
)
const MODES: { id: Mode; n: string; label: string; tone: string }[] = [
  { id: 'heart', n: '01', label: 'DRIVERS', tone: 'var(--color-burgundy)' },
  { id: 'brain', n: '02', label: 'OUTCOME', tone: 'var(--color-olive)' },
]

export function Hero() {
  const hud = useMemo(createBus, [])
  const tel = useTelemetry()
  const heroFeatures = useHeroFeatures()
  const [mode, setMode] = useState<Mode>('heart')
  const [shown, setShown] = useState<Mode>('heart') // caption follows the visual swap, not the click
  const [active, setActive] = useState(true)
  const [reduce, setReduce] = useState(false)
  const frame = useRef<HTMLDivElement>(null)

  useEffect(() => {
    setReduce(matchMedia('(prefers-reduced-motion: reduce)').matches)
    const onVisibility = () => {
      if (document.hidden) setActive(false)
      else if (frame.current) {
        const rect = frame.current.getBoundingClientRect()
        setActive(rect.top < window.innerHeight && rect.bottom > 0)
      }
    }
    document.addEventListener('visibilitychange', onVisibility)
    const io = new IntersectionObserver(([e]) => {
      if (!document.hidden) setActive(e.isIntersecting)
    }, { threshold: 0.05 })
    io.observe(frame.current!)
    return () => {
      document.removeEventListener('visibilitychange', onVisibility)
      io.disconnect()
    }
  }, [])

  useEffect(() => { const id = setTimeout(() => setShown(mode), reduce ? 0 : 220); return () => clearTimeout(id) }, [mode, reduce])

  const toggle = () => setMode((m) => (m === 'heart' ? 'brain' : 'heart'))
  const move = (e: React.PointerEvent) => {
    const b = e.currentTarget.getBoundingClientRect()
    hud.pointer.x = (e.clientX - b.left) / b.width; hud.pointer.y = 1 - (e.clientY - b.top) / b.height
  }

  return (
    <section id="top" className="relative overflow-hidden pb-4 sm:pb-6" style={{ ['--wm' as string]: 'clamp(84px, 21.5vw, 400px)' }}>
      <header className="flex flex-col gap-2 px-3 pb-2 pt-2 sm:flex-row sm:items-baseline sm:justify-between sm:gap-6 sm:px-5 sm:pb-2.5 sm:pt-2.5">
        <div>
          <p className="max-w-[24ch] text-[18px] font-medium leading-[1.2] tracking-[-0.01em] text-ink sm:text-[21px]">
            Not just a score, a reason. Every risk comes with its why.
          </p>
        </div>
        <div className="font-mono text-[11px] tracking-wide text-muted sm:text-right md:text-[12px]">
          XGBoost · Isotonic Calibration · Held-out N = 1,022 · AUC 0.790
        </div>
      </header>

      {/* Dark Diagnostic Monitor inset: warm obsidian surface with data-theme="dark" */}
      <div className="px-3 sm:px-5">
        <div
          ref={frame} data-theme="dark" role="group" tabIndex={0}
          aria-label="Diagnostic scan viewport. Click or press Enter to switch between heart and brain."
          onPointerMove={move} onPointerLeave={() => { hud.pointer.x = 0.5; hud.pointer.y = 0.5 }} onClick={toggle}
          onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); toggle() } }}
          style={{ height: 'calc(100svh - var(--hdr) - 0.48 * var(--wm))' }}
          className="relative min-h-[440px] cursor-pointer overflow-hidden border border-line bg-[#11100F] focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-burgundy"
        >
          <HeartCanvas mode={mode} hud={hud} reduce={reduce} active={active} />
          <HudOverlay hud={hud} features={heroFeatures} />

          <div className="pointer-events-none absolute left-4 top-4 font-mono text-[10px] leading-relaxed text-[#A8A29E]">
            <p className="font-semibold tracking-wider text-ink">{shown === 'heart' ? 'CARDIOVASCULAR DRIVERS' : 'CEREBRAL OUTCOME'}</p>
            <p className="hidden text-[#A8A29E] sm:block">{shown === 'heart' ? 'Top 4 SHAP drivers of the sample patient' : 'The same drivers, mapped to the organ the model protects'}</p>
            <p className="mt-1 text-[#9CA3AF]">{tel.version} / {tel.source === 'api' ? 'live model' : 'demo data'}</p>
          </div>

          <div className="absolute left-4 flex gap-2" style={{ bottom: 'calc(0.36 * var(--wm) + 1rem)' }} onClick={(e) => e.stopPropagation()}>
            {MODES.map((m) => (
              <button key={m.id} onClick={() => setMode(m.id)} aria-pressed={mode === m.id} aria-label={`Switch to ${m.label} scan view`}
                className={`group flex h-14 w-[104px] flex-col justify-between border p-2 text-left font-mono text-[10px] backdrop-blur-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-burgundy ${mode === m.id ? 'bg-ink text-canvas' : 'bg-surface/70 text-muted hover:bg-hover hover:text-ink'}`}
                style={{ borderColor: mode === m.id ? m.tone : 'var(--border-subtle)' }}>
                <span className="flex items-center justify-between"><span className="font-semibold" style={{ color: mode === m.id ? undefined : '#E57373' }}>{m.n}</span><span className="text-muted transition-transform group-hover:-translate-y-px group-hover:translate-x-px">&#8599;</span></span>
                <span>{m.label}</span>
              </button>
            ))}
          </div>
          <p className="sr-only" aria-live="polite" role="status">{mode.toUpperCase()} scan active. Top 4 SHAP risk drivers updated.</p>
        </div>
      </div>

      {/* Wordmark straddles the monitor edge: cream copy over dark, espresso copy over cream */}
      <div id="hero-wordmark" aria-hidden className="wm pointer-events-none relative select-none" style={{ fontSize: 'var(--wm)', marginTop: '-0.36em', paddingBottom: '0.45em' }}>
        <span className="wm-a">triage</span>
        <span className="wm-b">triage</span>
      </div>
    </section>
  )
}
