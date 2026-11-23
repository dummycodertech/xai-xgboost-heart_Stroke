'use client'
import { useEffect, useRef } from 'react'
import { GL } from '@/lib/palette'
import type { HudBus } from './tracker'

type Item = { x: number; y: number; tx: number; ty: number; has: boolean }
type Feature = { id: string; shap: number; value?: string }
const N_MICRO = 14
const LINK = 'rgba(246,242,233,0.22)'

// >>> 2D overlay. Reads detections from the shared bus every frame and eases reticles toward them.
export function HudOverlay({ hud, features }: { hud: HudBus; features: readonly Feature[] }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const featuresRef = useRef(features)
  featuresRef.current = features

  useEffect(() => {
    const cv = ref.current!, ctx = cv.getContext('2d')!
    const font = getComputedStyle(document.documentElement).getPropertyValue('--font-mono').trim() || 'monospace'
    const mk = (): Item => ({ x: 0.5, y: 0.5, tx: 0.5, ty: 0.5, has: false })
    const prim = Array.from({ length: 4 }, mk), micro = Array.from({ length: N_MICRO }, mk)
    let W = 1, H = 1, dpr = 1, raf = 0, last = performance.now(), epoch = hud.epoch, lockUntil = 0

    const fit = () => { dpr = Math.min(devicePixelRatio, 2); W = cv.clientWidth; H = cv.clientHeight; cv.width = W * dpr; cv.height = H * dpr }
    const ro = new ResizeObserver(fit); ro.observe(cv); fit()

    const match = () => {
      const peaks = hud.peaks, taken = new Set<number>()
      const grab = (it: Item, maxD: number) => {
        let best = -1
        if (!it.has) {
          best = peaks.findIndex((_, i) => !taken.has(i))
        } else {
          let bd = maxD
          peaks.forEach((p, i) => {
            if (taken.has(i)) return
            const d = Math.hypot(p.x - it.tx, p.y - it.ty)
            if (d < bd) { bd = d; best = i }
          })
          // If tracked peak moved beyond maxD, acquire nearest available untaken peak
          if (best < 0) {
            let nearestDist = 999
            peaks.forEach((p, i) => {
              if (taken.has(i)) return
              const d = Math.hypot(p.x - it.x, p.y - it.y)
              if (d < nearestDist) { nearestDist = d; best = i }
            })
          }
        }
        if (best >= 0) {
          taken.add(best)
          it.tx = peaks[best].x
          it.ty = peaks[best].y
          it.has = true
        } else {
          it.has = false
        }
      }
      prim.forEach((it) => grab(it, 0.25))
      micro.forEach((it) => grab(it, 0.18))
    }

    const bracket = (x: number, y: number, s: number, a: number) => {
      ctx.beginPath()
      for (const [sx, sy] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) {
        const cx = x + (sx * s) / 2, cy = y + (sy * s) / 2
        ctx.moveTo(cx - sx * a, cy); ctx.lineTo(cx, cy); ctx.lineTo(cx, cy - sy * a)
      }
      ctx.stroke()
    }
    const box = (x: number, y: number, s: number) => ctx.strokeRect(Math.round(x - s / 2) + 0.5, Math.round(y - s / 2) + 0.5, s, s)

    const loop = (now: number) => {
      raf = requestAnimationFrame(loop)
      const dt = Math.min(0.05, (now - last) / 1000); last = now
      if (hud.epoch !== epoch) { epoch = hud.epoch; [...prim, ...micro].forEach((i) => (i.has = false)); lockUntil = now + 500 }
      match()
      const k = 1 - Math.exp(-dt * 9)
      for (const it of [...prim, ...micro]) { it.x += (it.tx - it.x) * k; it.y += (it.ty - it.y) * k }

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H)
      const vis = 1 - Math.min(1, hud.fx.ascii * 1.6)
      if (vis < 0.03 || !hud.peaks.length) return
      const jit = hud.fx.glitch
      ctx.globalAlpha = vis; ctx.lineWidth = 1; ctx.font = `10px ${font}`
      const J = () => (jit ? (Math.random() - 0.5) * 12 * jit : 0)
      const P = [...prim, ...micro].map((i) => ({ x: i.x * W + J(), y: i.y * H }))

      // constellation: each point to its two nearest neighbours
      ctx.strokeStyle = LINK; ctx.beginPath()
      const seen = new Set<string>(), maxL = Math.min(W, H) * 0.34
      P.forEach((a, i) => {
        P.map((b, j) => ({ j, d: Math.hypot(a.x - b.x, a.y - b.y) })).filter((o) => o.j !== i && o.d < maxL).sort((p, q) => p.d - q.d).slice(0, 2).forEach((o) => {
          const key = i < o.j ? `${i}-${o.j}` : `${o.j}-${i}`
          if (seen.has(key)) return; seen.add(key)
          ctx.moveTo(Math.round(a.x) + 0.5, Math.round(a.y) + 0.5); ctx.lineTo(Math.round(P[o.j].x) + 0.5, Math.round(P[o.j].y) + 0.5)
        })
      })
      ctx.stroke()

      // micro trackers: outer square, inner square, centre dot
      ctx.strokeStyle = 'rgba(246,242,233,0.8)'; ctx.fillStyle = GL.cream
      for (let i = prim.length; i < P.length; i++) { box(P[i].x, P[i].y, 12); box(P[i].x, P[i].y, 5); ctx.fillRect(Math.round(P[i].x) - 0.5, Math.round(P[i].y) - 0.5, 2, 2) }

      // Primary reticles with callouts: centered around the subject, balanced 2 on left and 2 on right
      const cx = W * 0.5, cy = H * 0.5
      const feats = featuresRef.current
      const isMobile = W < 640
      const LW = isMobile ? Math.min(152, Math.floor(W * 0.38)) : Math.min(236, Math.max(188, W * 0.22))
      const LH = isMobile ? 40 : 46

      // Rank the 4 targets by x position: 2 leftmost go to left flank, 2 rightmost go to right flank
      const sortedByX = prim.map((_, i) => ({ i, x: P[i].x, y: P[i].y })).sort((a, b) => a.x - b.x)
      const leftGroup = sortedByX.slice(0, 2).sort((a, b) => a.y - b.y)
      const rightGroup = sortedByX.slice(2, 4).sort((a, b) => a.y - b.y)

      const slot: Record<number, { side: number; row: number }> = {}
      leftGroup.forEach((item, row) => { slot[item.i] = { side: 0, row } })
      rightGroup.forEach((item, row) => { slot[item.i] = { side: 1, row } })

      // Distance from center: keeps the callout blocks closely framing the organ, not pinned to outer edges
      const span = Math.min(W * 0.24, 210)
      const leftX = isMobile ? 8 : Math.max(16, cx - span - LW)
      const rightX = isMobile ? W - LW - 8 : Math.min(W - LW - 16, cx + span)

      prim.forEach((_, i) => {
        const f = feats[i] ?? { id: 'PARAM', shap: 0 }, p = P[i], col = f.shap > 0 ? GL.burgundy : GL.olive, s = slot[i] ?? { side: 0, row: 0 }
        ctx.strokeStyle = col; bracket(p.x, p.y, 42, 12); box(p.x, p.y, 8); ctx.fillStyle = col; ctx.fillRect(Math.round(p.x) - 1, Math.round(p.y) - 1, 3, 3)
        const left = s.side === 0, x0 = left ? leftX : rightX
        const ey = s.row === 0 ? Math.round(cy - (isMobile ? 60 : 68)) : Math.round(cy + (isMobile ? 60 : 68))
        const ax = left ? x0 + LW : x0, dir = left ? 1 : -1
        const halfH = isMobile ? 20 : 23

        // Angled leader lines with horizontal stubs into the callout blocks
        ctx.strokeStyle = col; ctx.lineWidth = 1.25; ctx.beginPath()
        ctx.moveTo(p.x - dir * 21, p.y)
        ctx.lineTo(ax + dir * 26, ey)
        ctx.lineTo(ax, ey)
        ctx.stroke()

        // Card background & high-contrast border
        ctx.fillStyle = 'rgba(17, 16, 15, 0.95)'
        ctx.fillRect(Math.round(x0), Math.round(ey - halfH), LW, LH)
        ctx.strokeStyle = col
        ctx.lineWidth = 1.5
        ctx.strokeRect(Math.round(x0) + 0.5, Math.round(ey - halfH) + 0.5, LW, LH)

        // Top line: Feature Tag and SHAP value
        ctx.font = isMobile ? `700 11px ${font}` : `700 13px ${font}`
        ctx.textAlign = 'left'; ctx.fillStyle = GL.cream
        ctx.fillText(f.id, x0 + (isMobile ? 8 : 12), ey - (isMobile ? 3 : 4))
        ctx.textAlign = 'right'; ctx.fillStyle = col
        ctx.fillText(`${f.shap > 0 ? '+' : '-'}${Math.abs(f.shap).toFixed(2)}`, x0 + LW - (isMobile ? 8 : 12), ey - (isMobile ? 3 : 4))

        // Bottom line: Risk direction indicator on left, sample patient clinical value on right
        ctx.font = isMobile ? `10px ${font}` : `11px ${font}`
        ctx.textAlign = 'left'
        ctx.fillStyle = f.shap > 0 ? '#f08fa5' : '#a8c99e'
        ctx.fillText(f.shap > 0 ? '▲ raises' : '▼ lowers', x0 + (isMobile ? 8 : 12), ey + (isMobile ? 12 : 14))

        const valText = f.value ?? (
          f.id.includes('AGE') ? '62 yrs' :
          f.id.includes('GLUCOSE') ? '195 mg/dL' :
          f.id.includes('RISK') ? '2 active' :
          f.id.includes('HYPERTEN') ? 'Yes' :
          f.id.includes('SMOK') ? 'former' :
          f.id.includes('BMI') ? '32.4' :
          'sample'
        )
        ctx.textAlign = 'right'; ctx.fillStyle = '#A8A29E'
        ctx.fillText(valText, x0 + LW - (isMobile ? 8 : 12), ey + (isMobile ? 12 : 14))
      })

      ctx.textAlign = 'right'; ctx.fillStyle = now < lockUntil ? GL.ochre : '#A8A29E'
      ctx.fillText(now < lockUntil ? 'ACQUIRING TARGETS' : `TRACKING ${prim.filter((p) => p.has).length} OF ${prim.length} DRIVERS`, W - 16, 22)
    }
    raf = requestAnimationFrame(loop)
    return () => { cancelAnimationFrame(raf); ro.disconnect() }
  }, [hud])

  return <canvas ref={ref} className="pointer-events-none absolute inset-0 h-full w-full" aria-hidden />
}
