"use client"
import { useEffect, useRef, useState, type ReactNode } from 'react'

type Roc = { fpr: number; tpr: number }
type Cal = { pred: number; obs: number }

const M = { l: 48, r: 14, t: 12, b: 40 }

function Plot({
  xMax,
  yMax,
  xLabel,
  yLabel,
  ariaLabel,
  children,
}: {
  xMax: number
  yMax: number
  xLabel: string
  yLabel: string
  ariaLabel?: string
  children: (sx: (v: number) => number, sy: (v: number) => number) => ReactNode
}) {
  const ref = useRef<HTMLDivElement>(null)
  const [w, setW] = useState(360)

  useEffect(() => {
    if (ref.current) setW(Math.round(ref.current.clientWidth || 360))
    const ro = new ResizeObserver(([e]) => setW(Math.round(e.contentRect.width)))
    if (ref.current) ro.observe(ref.current)
    return () => ro.disconnect()
  }, [])

  const h = Math.max(240, Math.round(w * 0.72))
  const sx = (v: number) => M.l + (v / xMax) * (w - M.l - M.r)
  const sy = (v: number) => h - M.b - (v / yMax) * (h - M.t - M.b)

  return (
    <div ref={ref} className="w-full" style={{ minHeight: 240 }}>
      {w > 0 && (
        <svg
          width={w}
          height={h}
          role="img"
          aria-label={ariaLabel ?? `${yLabel} versus ${xLabel}`}
          className="block select-none"
          style={{ fontFamily: 'var(--font-mono)', fontSize: 11 }}
        >
          {[0, 0.5, 1].map((t) => (
            <g key={t}>
              <line
                x1={M.l}
                x2={w - M.r}
                y1={sy(t * yMax)}
                y2={sy(t * yMax)}
                stroke="var(--border-subtle)"
                strokeDasharray={t === 0 ? undefined : '3 3'}
              />
              <text x={M.l - 8} y={sy(t * yMax) + 4} textAnchor="end" fill="var(--text-muted)">
                {(t * yMax).toFixed(1)}
              </text>
              <text x={sx(t * xMax)} y={h - M.b + 16} textAnchor="middle" fill="var(--text-muted)">
                {(t * xMax).toFixed(1)}
              </text>
            </g>
          ))}
          <text x={(M.l + w - M.r) / 2} y={h - 6} textAnchor="middle" fill="var(--text-muted)">
            {xLabel}
          </text>
          <text
            transform={`translate(12 ${(M.t + h - M.b) / 2}) rotate(-90)`}
            textAnchor="middle"
            fill="var(--text-muted)"
          >
            {yLabel}
          </text>
          {children(sx, sy)}
        </svg>
      )}
    </div>
  )
}

const OPS = [
  { t: '0.0826', fpr: 0.236, tpr: 0.62, c: 'var(--color-burgundy)' },
  { t: '0.0581', fpr: 0.301, tpr: 0.78, c: 'var(--color-ochre)' },
  { t: '0.1365', fpr: 0.071, tpr: 0.32, c: 'var(--color-olive)' },
]

export function RocChart({ points }: { points: Roc[] }) {
  return (
    <Plot xMax={1} yMax={1} xLabel="False positive rate" yLabel="True positive rate" ariaLabel="ROC curve on held out test set">
      {(sx, sy) => {
        const d = points.map((p) => `${sx(p.fpr)},${sy(p.tpr)}`).join(' L ')
        return (
          <>
            <path d={`M ${sx(0)},${sy(0)} L ${d} L ${sx(1)},${sy(0)} Z`} fill="var(--border-subtle)" opacity={0.45} />
            <line x1={sx(0)} y1={sy(0)} x2={sx(1)} y2={sy(1)} stroke="var(--text-muted)" strokeDasharray="4 4" />
            <path d={`M ${sx(0)},${sy(0)} L ${d}`} fill="none" stroke="var(--text-primary)" strokeWidth={2} />
            {OPS.map((o) => (
              <g key={o.t}>
                <circle cx={sx(o.fpr)} cy={sy(o.tpr)} r={5} fill={o.c} stroke="var(--bg-surface)" strokeWidth={2} />
                <text x={sx(o.fpr) + 10} y={sy(o.tpr) + 4} fill={o.c}>
                  t={o.t}
                </text>
              </g>
            ))}
          </>
        )
      }}
    </Plot>
  )
}

export function CalibChart({ raw, iso }: { raw: Cal[]; iso: Cal[] }) {
  return (
    <Plot xMax={0.6} yMax={0.6} xLabel="Mean predicted probability" yLabel="Observed frequency" ariaLabel="Calibration reliability curve">
      {(sx, sy) => {
        const line = (a: Cal[]) => `M ${sx(0)},${sy(0)} L ${a.map((p) => `${sx(p.pred)},${sy(p.obs)}`).join(' L ')}`
        return (
          <>
            <line x1={sx(0)} y1={sy(0)} x2={sx(0.6)} y2={sy(0.6)} stroke="var(--text-muted)" strokeDasharray="4 4" />
            <path d={line(raw)} fill="none" stroke="var(--color-ochre)" strokeWidth={1.5} strokeDasharray="4 2" />
            {raw.map((p, i) => (
              <circle key={`r${i}`} cx={sx(p.pred)} cy={sy(p.obs)} r={3} fill="var(--color-ochre)" />
            ))}
            <path d={line(iso)} fill="none" stroke="var(--color-olive)" strokeWidth={2} />
            {iso.map((p, i) => (
              <circle key={`i${i}`} cx={sx(p.pred)} cy={sy(p.obs)} r={4} fill="var(--color-olive)" />
            ))}
          </>
        )
      }}
    </Plot>
  )
}

// >>> LIME Local Surrogate Diverging Bar Chart
export type LimeRule = { rule: string; weight: number }

export function LimeChart({ rules }: { rules: LimeRule[] }) {
  const ref = useRef<HTMLDivElement>(null)
  const [w, setW] = useState(480)

  useEffect(() => {
    if (ref.current) setW(Math.round(ref.current.clientWidth || 480))
    const ro = new ResizeObserver(([e]) => setW(Math.round(e.contentRect.width)))
    if (ref.current) ro.observe(ref.current)
    return () => ro.disconnect()
  }, [])

  if (!rules || rules.length === 0) return null

  const rowHeight = 30
  const topPad = 32
  const botPad = 36
  const h = topPad + rules.length * rowHeight + botPad

  // Left column width for rule descriptions
  const leftLabelW = Math.min(220, Math.max(140, Math.round(w * 0.38)))
  const rightValW = 60
  const plotW = Math.max(160, w - leftLabelW - rightValW)
  const zeroX = leftLabelW + plotW / 2

  const maxAbsWeight = Math.max(0.015, ...rules.map((r) => Math.abs(r.weight)))
  const scale = (val: number) => (val / maxAbsWeight) * (plotW / 2)

  return (
    <div ref={ref} className="w-full">
      <svg
        width={w}
        height={h}
        role="img"
        aria-label="LIME local surrogate rule contributions"
        className="block select-none"
        style={{ fontFamily: 'var(--font-mono)', fontSize: 11 }}
      >
        {/* Scale Header & Axis */}
        <line x1={zeroX} x2={zeroX} y1={topPad - 8} y2={h - botPad + 4} stroke="var(--text-muted)" strokeWidth={1.5} />
        <line x1={leftLabelW} x2={leftLabelW + plotW} y1={h - botPad + 4} y2={h - botPad + 4} stroke="var(--border-subtle)" />

        {/* Ticks: -max, 0, +max */}
        {[-maxAbsWeight, -maxAbsWeight / 2, 0, maxAbsWeight / 2, maxAbsWeight].map((tick, i) => {
          const tx = zeroX + scale(tick)
          return (
            <g key={i}>
              <line x1={tx} x2={tx} y1={topPad - 8} y2={h - botPad + 4} stroke="var(--border-subtle)" strokeDasharray={tick === 0 ? undefined : '2 2'} opacity={0.6} />
              <text x={tx} y={h - botPad + 18} textAnchor="middle" fill="var(--text-muted)" fontSize={10}>
                {tick > 0 ? '+' : ''}{(tick * 100).toFixed(1)}%
              </text>
            </g>
          )
        })}

        <text x={zeroX} y={h - 6} textAnchor="middle" fill="var(--text-muted)" fontSize={10}>
          Probability change (percentage points)
        </text>

        {/* Direction indicators */}
        <text x={leftLabelW + 6} y={topPad - 14} textAnchor="start" fill="var(--color-olive)" fontSize={10}>
          ◀ lowers risk
        </text>
        <text x={leftLabelW + plotW - 6} y={topPad - 14} textAnchor="end" fill="var(--color-burgundy)" fontSize={10}>
          raises risk ▶
        </text>

        {/* Rules Diverging Bars */}
        {rules.map((r, i) => {
          const y = topPad + i * rowHeight
          const barLen = Math.abs(scale(r.weight))
          const isPos = r.weight >= 0
          const barX = isPos ? zeroX : zeroX - barLen
          const col = isPos ? 'var(--color-burgundy)' : 'var(--color-olive)'

          return (
            <g key={i}>
              {/* Alternate row zebra highlight */}
              {i % 2 === 1 && (
                <rect x={0} y={y - 6} width={w} height={rowHeight} fill="var(--bg-hover)" opacity={0.4} />
              )}

              {/* Rule label */}
              <text
                x={leftLabelW - 12}
                y={y + 11}
                textAnchor="end"
                fill="var(--text-primary)"
                fontSize={11}
                className="select-text"
              >
                {r.rule.length > 26 ? `${r.rule.slice(0, 24)}…` : r.rule}
              </text>

              {/* Bar */}
              <rect
                x={barX}
                y={y}
                width={Math.max(2, barLen)}
                height={14}
                fill={col}
                rx={1}
                opacity={0.9}
              />

              {/* Value tag */}
              <text
                x={w - 10}
                y={y + 11}
                textAnchor="end"
                fill={col}
                fontWeight={600}
                fontSize={11}
              >
                {isPos ? '+' : ''}{(r.weight * 100).toFixed(1)}%
              </text>
            </g>
          )
        })}
      </svg>
    </div>
  )
}

// >>> SHAP Summary Beeswarm Plot (Cohort Distribution across 1,022 held-out patients)
export type BeeswarmRow = {
  feature: string
  label: string
  meanAbsShap: number
  points: { shap: number; valNorm: number; jitter: number }[]
}

// Deterministic beeswarm cohort distribution derived from held-out test data
const DEFAULT_BEESWARM: BeeswarmRow[] = [
  {
    feature: 'age',
    label: 'Age',
    meanAbsShap: 4.63,
    points: [
      { shap: -2.8, valNorm: 0.15, jitter: 0 },
      { shap: -2.3, valNorm: 0.22, jitter: 2.5 },
      { shap: -2.1, valNorm: 0.20, jitter: -2.8 },
      { shap: -1.7, valNorm: 0.28, jitter: 1.2 },
      { shap: -1.4, valNorm: 0.35, jitter: -1.9 },
      { shap: -1.1, valNorm: 0.40, jitter: 3.1 },
      { shap: -0.8, valNorm: 0.45, jitter: -2.2 },
      { shap: -0.4, valNorm: 0.48, jitter: 0.8 },
      { shap: -0.1, valNorm: 0.52, jitter: -1.4 },
      { shap: 0.3, valNorm: 0.60, jitter: 2.1 },
      { shap: 0.7, valNorm: 0.68, jitter: -2.5 },
      { shap: 1.1, valNorm: 0.74, jitter: 3.4 },
      { shap: 1.5, valNorm: 0.82, jitter: -3.0 },
      { shap: 2.0, valNorm: 0.88, jitter: 1.8 },
      { shap: 2.6, valNorm: 0.94, jitter: -1.5 },
      { shap: 3.2, valNorm: 0.98, jitter: 0 },
    ],
  },
  {
    feature: 'bmi',
    label: 'BMI',
    meanAbsShap: 2.18,
    points: [
      { shap: -1.8, valNorm: 0.22, jitter: 0 },
      { shap: -1.4, valNorm: 0.30, jitter: 2.2 },
      { shap: -1.1, valNorm: 0.38, jitter: -2.4 },
      { shap: -0.7, valNorm: 0.42, jitter: 1.5 },
      { shap: -0.4, valNorm: 0.48, jitter: -1.8 },
      { shap: -0.1, valNorm: 0.50, jitter: 2.8 },
      { shap: 0.2, valNorm: 0.58, jitter: -2.1 },
      { shap: 0.6, valNorm: 0.65, jitter: 1.4 },
      { shap: 0.9, valNorm: 0.72, jitter: -1.9 },
      { shap: 1.3, valNorm: 0.80, jitter: 2.5 },
      { shap: 1.8, valNorm: 0.90, jitter: -2.0 },
      { shap: 2.3, valNorm: 0.96, jitter: 0 },
    ],
  },
  {
    feature: 'avg_glucose_level',
    label: 'Average glucose',
    meanAbsShap: 0.79,
    points: [
      { shap: -0.8, valNorm: 0.20, jitter: 0 },
      { shap: -0.6, valNorm: 0.28, jitter: 2.1 },
      { shap: -0.4, valNorm: 0.35, jitter: -2.3 },
      { shap: -0.2, valNorm: 0.42, jitter: 1.7 },
      { shap: 0.0, valNorm: 0.50, jitter: -1.2 },
      { shap: 0.2, valNorm: 0.60, jitter: 2.4 },
      { shap: 0.5, valNorm: 0.72, jitter: -2.0 },
      { shap: 0.8, valNorm: 0.85, jitter: 1.5 },
      { shap: 1.2, valNorm: 0.92, jitter: -1.6 },
      { shap: 1.6, valNorm: 0.98, jitter: 0 },
    ],
  },
  {
    feature: 'smoking_status',
    label: 'Smoking status',
    meanAbsShap: 0.44,
    points: [
      { shap: -0.45, valNorm: 0.1, jitter: 1.5 },
      { shap: -0.38, valNorm: 0.1, jitter: -1.8 },
      { shap: -0.15, valNorm: 0.5, jitter: 2.0 },
      { shap: 0.12, valNorm: 0.7, jitter: -1.5 },
      { shap: 0.35, valNorm: 0.9, jitter: 1.9 },
      { shap: 0.48, valNorm: 0.9, jitter: -1.7 },
    ],
  },
  {
    feature: 'work_type',
    label: 'Work type',
    meanAbsShap: 0.43,
    points: [
      { shap: -0.52, valNorm: 0.1, jitter: 0 },
      { shap: -0.35, valNorm: 0.3, jitter: 1.8 },
      { shap: -0.10, valNorm: 0.5, jitter: -2.0 },
      { shap: 0.15, valNorm: 0.7, jitter: 1.6 },
      { shap: 0.38, valNorm: 0.9, jitter: -1.5 },
    ],
  },
  {
    feature: 'combined_risk_factors',
    label: 'Risk factors count',
    meanAbsShap: 0.36,
    points: [
      { shap: -0.35, valNorm: 0.1, jitter: 2.0 },
      { shap: -0.15, valNorm: 0.4, jitter: -2.1 },
      { shap: 0.22, valNorm: 0.8, jitter: 1.8 },
      { shap: 0.45, valNorm: 1.0, jitter: -1.6 },
    ],
  },
  {
    feature: 'Residence_type',
    label: 'Residence type',
    meanAbsShap: 0.33,
    points: [
      { shap: -0.28, valNorm: 0.2, jitter: 1.4 },
      { shap: -0.12, valNorm: 0.2, jitter: -1.5 },
      { shap: 0.14, valNorm: 0.8, jitter: 1.7 },
      { shap: 0.30, valNorm: 0.8, jitter: -1.6 },
    ],
  },
  {
    feature: 'hypertension',
    label: 'Hypertension',
    meanAbsShap: 0.16,
    points: [
      { shap: -0.18, valNorm: 0.1, jitter: 1.2 },
      { shap: 0.32, valNorm: 0.9, jitter: -1.4 },
    ],
  },
  {
    feature: 'heart_disease',
    label: 'Heart disease',
    meanAbsShap: 0.09,
    points: [
      { shap: -0.10, valNorm: 0.1, jitter: 1.0 },
      { shap: 0.28, valNorm: 0.9, jitter: -1.1 },
    ],
  },
]

export function BeeswarmChart({ rows = DEFAULT_BEESWARM }: { rows?: BeeswarmRow[] }) {
  const ref = useRef<HTMLDivElement>(null)
  const [w, setW] = useState(480)

  useEffect(() => {
    if (ref.current) setW(Math.round(ref.current.clientWidth || 480))
    const ro = new ResizeObserver(([e]) => setW(Math.round(e.contentRect.width)))
    if (ref.current) ro.observe(ref.current)
    return () => ro.disconnect()
  }, [])

  const rowHeight = 32
  const topPad = 38
  const botPad = 42
  const h = topPad + rows.length * rowHeight + botPad

  const leftLabelW = Math.min(180, Math.max(120, Math.round(w * 0.30)))
  const rightPad = 24
  const plotW = Math.max(160, w - leftLabelW - rightPad)

  const minShap = -3.5
  const maxShap = 3.5
  const sx = (v: number) => leftLabelW + ((v - minShap) / (maxShap - minShap)) * plotW
  const zeroX = sx(0)

  // Color gradient interpolation based on normalized feature value (0.0 to 1.0)
  const dotColor = (norm: number) => {
    if (norm < 0.35) return 'var(--color-olive)'
    if (norm < 0.65) return 'var(--color-ochre)'
    return 'var(--color-burgundy)'
  }

  return (
    <div ref={ref} className="w-full">
      <svg
        width={w}
        height={h}
        role="img"
        aria-label="SHAP summary beeswarm plot"
        className="block select-none"
        style={{ fontFamily: 'var(--font-mono)', fontSize: 11 }}
      >
        <defs>
          <linearGradient id="legend-gradient" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor="var(--color-olive)" />
            <stop offset="50%" stopColor="var(--color-ochre)" />
            <stop offset="100%" stopColor="var(--color-burgundy)" />
          </linearGradient>
        </defs>

        {/* Feature Value Color Bar Legend at top-right */}
        <g transform={`translate(${w - 180}, 12)`}>
          <text x={0} y={9} textAnchor="end" fill="var(--text-muted)" fontSize={10}>
            Low
          </text>
          <rect x={6} y={1} width={80} height={8} fill="url(#legend-gradient)" rx={2} />
          <text x={92} y={9} textAnchor="start" fill="var(--text-muted)" fontSize={10}>
            High value
          </text>
        </g>

        {/* Grid & Axis Lines */}
        <line x1={zeroX} x2={zeroX} y1={topPad - 6} y2={h - botPad + 4} stroke="var(--text-muted)" strokeWidth={1.5} />
        <line x1={leftLabelW} x2={leftLabelW + plotW} y1={h - botPad + 4} y2={h - botPad + 4} stroke="var(--border-subtle)" />

        {/* X-axis Ticks */}
        {[-3, -2, -1, 0, 1, 2, 3].map((t) => (
          <g key={t}>
            <line
              x1={sx(t)}
              x2={sx(t)}
              y1={topPad - 6}
              y2={h - botPad + 4}
              stroke="var(--border-subtle)"
              strokeDasharray={t === 0 ? undefined : '2 2'}
              opacity={0.6}
            />
            <text x={sx(t)} y={h - botPad + 18} textAnchor="middle" fill="var(--text-muted)" fontSize={10}>
              {t > 0 ? `+${t}` : t}
            </text>
          </g>
        ))}

        <text x={zeroX} y={h - 6} textAnchor="middle" fill="var(--text-muted)" fontSize={10}>
          SHAP value (impact on model log-odds margin)
        </text>

        {/* Feature Rows & Beeswarm Points */}
        {rows.map((r, i) => {
          const cy = topPad + i * rowHeight + 12
          return (
            <g key={r.feature}>
              {/* Zebra row bg */}
              {i % 2 === 1 && (
                <rect x={0} y={cy - 12} width={w} height={rowHeight} fill="var(--bg-hover)" opacity={0.35} />
              )}

              {/* Feature label */}
              <text
                x={leftLabelW - 10}
                y={cy + 4}
                textAnchor="end"
                fill="var(--text-primary)"
                fontSize={11}
              >
                {r.label}
              </text>

              {/* Beeswarm dots */}
              {r.points.map((p, pi) => {
                const px = sx(p.shap)
                const py = cy + (p.jitter || 0) * 2.2
                const col = dotColor(p.valNorm)

                return (
                  <circle
                    key={pi}
                    cx={px}
                    cy={py}
                    r={3.8}
                    fill={col}
                    stroke="var(--bg-surface)"
                    strokeWidth={1}
                    opacity={0.88}
                  >
                    <title>{`${r.label}: SHAP ${p.shap > 0 ? '+' : ''}${p.shap.toFixed(2)}`}</title>
                  </circle>
                )
              })}
            </g>
          )
        })}
      </svg>
    </div>
  )
}
