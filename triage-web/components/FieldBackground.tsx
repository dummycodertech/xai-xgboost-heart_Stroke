"use client"
import React, { useEffect, useRef } from 'react'

// >>> Asymmetric probability contour + flowing wave vector backdrop
// Flowing chromatic currents: Crimson, Burnt Orange, Golden Yellow, and Dark Glowing Red.
// Exactly 15 total curves: 6 top concentric contour loops, 5 bottom contour loops, 4 sweeping waves.
// Pure single-stroke vector lines with zero SVG/CSS filters for locked 60fps rendering performance.

type ContourDef = {
  rx: number
  ry: number
  skewX: number
  skewY: number
  isFocal?: boolean
}

// 6 topographic probability density rings anchored top-right (1060, 260)
const TOP_RINGS: ContourDef[] = [
  { rx: 110, ry: 75, skewX: 0.05, skewY: -0.04 },
  { rx: 195, ry: 135, skewX: 0.07, skewY: -0.05 },
  { rx: 290, ry: 205, skewX: 0.09, skewY: -0.06, isFocal: true },
  { rx: 410, ry: 290, skewX: 0.11, skewY: -0.07 },
  { rx: 560, ry: 395, skewX: 0.13, skewY: -0.08 },
  { rx: 740, ry: 520, skewX: 0.15, skewY: -0.09, isFocal: true },
]

// 5 topographic probability density rings anchored lower-left (340, 680)
const BOT_RINGS: ContourDef[] = [
  { rx: 130, ry: 90, skewX: -0.05, skewY: 0.04 },
  { rx: 230, ry: 160, skewX: -0.07, skewY: 0.05, isFocal: true },
  { rx: 360, ry: 250, skewX: -0.09, skewY: 0.06 },
  { rx: 520, ry: 360, skewX: -0.11, skewY: 0.07 },
  { rx: 710, ry: 490, skewX: -0.13, skewY: 0.08, isFocal: true },
]

// 4 flowing wave curves sweeping across the canvas
const WAVES = [
  { baseY: 170, amp: 18, freq: 0.0028, phase: 0.4 },
  { baseY: 380, amp: 22, freq: 0.0024, phase: 2.1 },
  { baseY: 540, amp: 20, freq: 0.0026, phase: 4.2 },
  { baseY: 750, amp: 24, freq: 0.0022, phase: 1.0 },
]

function smoothClosedLoop(points: { x: number; y: number }[]): string {
  const n = points.length
  if (n < 3) return ''
  const mid = (p1: { x: number; y: number }, p2: { x: number; y: number }) => ({
    x: ((p1.x + p2.x) / 2).toFixed(1),
    y: ((p1.y + p2.y) / 2).toFixed(1),
  })
  const start = mid(points[n - 1], points[0])
  let d = `M ${start.x} ${start.y}`
  for (let i = 0; i < n; i++) {
    const curr = points[i]
    const next = points[(i + 1) % n]
    const end = mid(curr, next)
    d += ` Q ${curr.x.toFixed(1)} ${curr.y.toFixed(1)}, ${end.x} ${end.y}`
  }
  return d + ' Z'
}

function generateContourPath(
  cx: number,
  cy: number,
  rx: number,
  ry: number,
  deg: number,
  skewX: number,
  skewY: number,
  mx: number,
  my: number,
  time: number,
  sc: number
): string {
  const n = 22
  const rad = (deg * Math.PI) / 180
  const cos = Math.cos(rad), sin = Math.sin(rad)
  const pts: { x: number; y: number }[] = []

  // Subtle gravitational drift of center towards cursor
  const cdx = cx - mx, cdy = cy - my
  const cg = Math.exp(-(cdx * cdx + cdy * cdy) / 180000)
  const effectiveCx = cx - cdx * cg * 0.08
  const effectiveCy = cy - cdy * cg * 0.08 + Math.sin(time + sc * 2) * 4

  for (let i = 0; i < n; i++) {
    const a = (i / n) * Math.PI * 2
    const rX = rx * (1 + skewX * Math.cos(a))
    const rY = ry * (1 + skewY * Math.sin(a))
    const px = Math.cos(a) * rX
    const py = Math.sin(a) * rY
    const x0 = effectiveCx + px * cos - py * sin
    const y0 = effectiveCy + px * sin + py * cos

    // Localized cursor repulsion/expansion (fluid probability surface)
    const dx = x0 - mx, dy = y0 - my
    const g = Math.exp(-(dx * dx + dy * dy) / 44000)
    const wave = Math.sin(a * 4 + time * 1.2) * 3
    const x = x0 + dx * g * 0.16 + Math.cos(a) * wave
    const y = y0 + dy * g * 0.16 + Math.sin(a) * wave
    pts.push({ x, y })
  }
  return smoothClosedLoop(pts)
}

function generateWavePath(
  baseY: number,
  amp: number,
  freq: number,
  phase: number,
  mx: number,
  my: number,
  time: number,
  sc: number
): string {
  const steps = 28
  const points: { x: number; y: number }[] = []
  for (let i = 0; i <= steps; i++) {
    const x = -40 + (i / steps) * (1440 + 80)
    const natural = Math.sin(x * freq + phase + time * 1.5 + sc * 3) * amp + Math.sin(x * 0.0012 - time * 0.8) * 8
    const dx = x - mx, dy = baseY + natural - my
    const g = Math.exp(-(dx * dx + dy * dy) / 38000)
    const y = baseY + natural - g * 46
    points.push({ x, y })
  }
  let d = `M ${points[0].x.toFixed(1)} ${points[0].y.toFixed(1)}`
  for (let i = 1; i < points.length; i++) {
    d += ` L ${points[i].x.toFixed(1)} ${points[i].y.toFixed(1)}`
  }
  return d
}

export function FieldBackground() {
  const svgRef = useRef<SVGSVGElement>(null)

  // Direct element references for maximum performance (15 paths total)
  const topRefs = useRef<(SVGPathElement | null)[]>([])
  const botRefs = useRef<(SVGPathElement | null)[]>([])
  const waveRefs = useRef<(SVGPathElement | null)[]>([])

  useEffect(() => {
    const el = svgRef.current
    if (!el) return

    const topGradEl = el.querySelector<SVGLinearGradientElement>('#flow-top')
    const botGradEl = el.querySelector<SVGLinearGradientElement>('#flow-bot')
    const waveGradEl = el.querySelector<SVGLinearGradientElement>('#flow-wave')

    const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches
    if (reduce) return

    let raf = 0
    let vis = !document.hidden
    let mx = -1e4, my = -1e4, tx = mx, ty = my
    let last = 0

    const onPointerMove = (e: PointerEvent) => {
      const rect = el.getBoundingClientRect()
      if (rect.width > 0 && rect.height > 0) {
        tx = ((e.clientX - rect.left) / rect.width) * 1440
        ty = ((e.clientY - rect.top) / rect.height) * 900
      }
    }
    window.addEventListener('pointermove', onPointerMove, { passive: true })

    const onVisibilityChange = () => { vis = !document.hidden }
    document.addEventListener('visibilitychange', onVisibilityChange)

    const loop = (t: number) => {
      raf = requestAnimationFrame(loop)
      if (!vis || t - last < 24) return
      last = t

      // Smooth cursor easing
      mx += (tx - mx) * 0.12
      my += (ty - my) * 0.12

      const time = t * 0.0006
      const sc = window.scrollY * 0.0005

      // Dynamic color flow: rotate & translate gradients continuously
      if (topGradEl) {
        const rotTop = (t * 0.02) % 360
        topGradEl.setAttribute('gradientTransform', `rotate(${rotTop.toFixed(2)} 1060 260)`)
      }
      if (botGradEl) {
        const rotBot = (-t * 0.016) % 360
        botGradEl.setAttribute('gradientTransform', `rotate(${rotBot.toFixed(2)} 340 680)`)
      }
      if (waveGradEl) {
        const transX = (t * 0.15) % 1440
        waveGradEl.setAttribute('gradientTransform', `translate(${transX.toFixed(1)} 0)`)
      }

      // Update 6 Top-Right Contours
      TOP_RINGS.forEach((c, i) => {
        const pathEl = topRefs.current[i]
        if (pathEl) {
          pathEl.setAttribute('d', generateContourPath(1060, 260, c.rx, c.ry, -18, c.skewX, c.skewY, mx, my, time, sc))
        }
      })

      // Update 5 Lower-Left Contours
      BOT_RINGS.forEach((c, i) => {
        const pathEl = botRefs.current[i]
        if (pathEl) {
          pathEl.setAttribute('d', generateContourPath(340, 680, c.rx, c.ry, 22, c.skewX, c.skewY, mx, my, time, sc))
        }
      })

      // Update 4 Flowing Waves
      WAVES.forEach((w, i) => {
        const pathEl = waveRefs.current[i]
        if (pathEl) {
          pathEl.setAttribute('d', generateWavePath(w.baseY, w.amp, w.freq, w.phase, mx, my, time, sc))
        }
      })
    }

    raf = requestAnimationFrame(loop)

    return () => {
      cancelAnimationFrame(raf)
      window.removeEventListener('pointermove', onPointerMove)
      document.removeEventListener('visibilitychange', onVisibilityChange)
    }
  }, [])

  return (
    <svg
      ref={svgRef}
      aria-hidden="true"
      preserveAspectRatio="xMidYMid slice"
      viewBox="0 0 1440 900"
      className="pointer-events-none fixed inset-0 -z-10 h-full w-full"
    >
      <defs>
        {/* Continuous neon flow for Top-Right cluster: Crimson -> Burnt Orange -> Golden Yellow -> Dark Glowing Red */}
        <linearGradient id="flow-top" gradientUnits="userSpaceOnUse" x1="460" y1="0" x2="1660" y2="520">
          <stop offset="0%" stopColor="var(--flow-dark-red)" />
          <stop offset="20%" stopColor="var(--flow-crimson)" />
          <stop offset="44%" stopColor="var(--flow-orange)" />
          <stop offset="70%" stopColor="var(--flow-gold)" />
          <stop offset="86%" stopColor="var(--flow-crimson)" />
          <stop offset="100%" stopColor="var(--flow-dark-red)" />
        </linearGradient>

        {/* Counter-flowing neon gradient for Lower-Left cluster */}
        <linearGradient id="flow-bot" gradientUnits="userSpaceOnUse" x1="-160" y1="480" x2="840" y2="880">
          <stop offset="0%" stopColor="var(--flow-gold)" />
          <stop offset="24%" stopColor="var(--flow-orange)" />
          <stop offset="50%" stopColor="var(--flow-crimson)" />
          <stop offset="75%" stopColor="var(--flow-dark-red)" />
          <stop offset="100%" stopColor="var(--flow-gold)" />
        </linearGradient>

        {/* Horizontally streaming radiant neon gradient for Flowing Waves */}
        <linearGradient id="flow-wave" gradientUnits="userSpaceOnUse" x1="0" y1="0" x2="1440" y2="0" spreadMethod="repeat">
          <stop offset="0%" stopColor="var(--flow-dark-red)" />
          <stop offset="18%" stopColor="var(--flow-crimson)" />
          <stop offset="38%" stopColor="var(--flow-orange)" />
          <stop offset="58%" stopColor="var(--flow-gold)" />
          <stop offset="75%" stopColor="var(--flow-orange)" />
          <stop offset="88%" stopColor="var(--flow-crimson)" />
          <stop offset="100%" stopColor="var(--flow-dark-red)" />
        </linearGradient>
      </defs>

      <g fill="none">
        {/* 6 Top-Right Contours */}
        {TOP_RINGS.map((c, i) => (
          <path
            key={`top-${i}`}
            ref={(el) => { topRefs.current[i] = el }}
            d={generateContourPath(1060, 260, c.rx, c.ry, -18, c.skewX, c.skewY, -1e4, -1e4, 0, 0)}
            vectorEffect="non-scaling-stroke"
            stroke="url(#flow-top)"
            strokeWidth={c.isFocal ? 1.92 : 1.5}
            style={{
              opacity: c.isFocal ? 'var(--flow-accent-opacity)' : 'var(--flow-opacity)',
            }}
          />
        ))}

        {/* 5 Lower-Left Contours */}
        {BOT_RINGS.map((c, i) => (
          <path
            key={`bot-${i}`}
            ref={(el) => { botRefs.current[i] = el }}
            d={generateContourPath(340, 680, c.rx, c.ry, 22, c.skewX, c.skewY, -1e4, -1e4, 0, 0)}
            vectorEffect="non-scaling-stroke"
            stroke="url(#flow-bot)"
            strokeWidth={c.isFocal ? 1.92 : 1.5}
            style={{
              opacity: c.isFocal ? 'var(--flow-accent-opacity)' : 'var(--flow-opacity)',
            }}
          />
        ))}

        {/* 4 Flowing Waves */}
        {WAVES.map((w, i) => (
          <path
            key={`wave-${i}`}
            ref={(el) => { waveRefs.current[i] = el }}
            d={generateWavePath(w.baseY, w.amp, w.freq, w.phase, -1e4, -1e4, 0, 0)}
            vectorEffect="non-scaling-stroke"
            stroke="url(#flow-wave)"
            strokeWidth={1.56}
            style={{
              opacity: 'var(--flow-opacity)',
            }}
          />
        ))}
      </g>
    </svg>
  )
}
