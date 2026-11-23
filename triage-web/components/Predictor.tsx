'use client'
import { useEffect, useMemo, useState } from 'react'
import { useThreshold } from './ThresholdContext'
import { DEFAULT_PATIENT, predict, STRATEGIES, syncHealth, type PatientInput, type Prediction } from '@/lib/api'
import { LimeChart } from './charts'

const fmt = (v: number) => `${v > 0 ? '+' : v < 0 ? '-' : ''}${Math.abs(v).toFixed(2)}`
// >>> 5-step diverging scale, graded by magnitude relative to the largest bar
const scale = (v: number, max: number) => {
  const m = Math.abs(v) / max
  if (m < 0.06) return v < 0 ? 'var(--scale-p)' : 'var(--scale-0)'
  return v > 0 ? (m > 0.55 ? 'var(--scale-rr)' : 'var(--scale-r)') : (m > 0.55 ? 'var(--scale-pp)' : 'var(--scale-p)')
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return <label className="block"><span className="mb-1.5 block text-[13px] text-muted">{label}</span>{children}</label>
}
const selectCls = 'h-11 w-full rounded-none border border-control bg-canvas px-3 pr-8 text-sm text-ink'

function Slider({ label, unit, value, min, max, step, onChange }: { label: string; unit?: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void }) {
  const fill = ((value - min) / (max - min)) * 100
  return (
    <div>
      <div className="flex items-center justify-between text-[13px] text-muted">
        <span>{label}{unit ? <span className="ml-1 font-mono text-[11px]">{unit.trim()}</span> : null}</span>
        <input type="number" aria-label={`${label} value`} min={min} max={max} step={step} value={value} onChange={(e) => { const v = Number(e.target.value); if (!Number.isNaN(v)) onChange(Math.min(max, Math.max(min, v))) }} className="h-8 w-20 border border-control bg-canvas px-2 text-right font-mono text-ink" />
      </div>
      <input type="range" aria-label={label} min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} style={{ ['--fill' as string]: `${fill}%` }} />
      <div className="flex justify-between font-mono text-[10px] text-muted"><span>{min}</span><span>{max}</span></div>
    </div>
  )
}
function Seg({ label, value, options, onChange }: { label: string; value: string | number; options: [string | number, string][]; onChange: (v: string) => void }) {
  return (
    <div role="radiogroup" aria-label={label}>
      <span className="mb-1.5 block text-[13px] text-muted">{label}</span>
      <div className="grid h-11 grid-flow-col auto-cols-fr border border-control">
        {options.map(([v, t]) => (
          <button key={String(v)} type="button" role="radio" aria-checked={String(value) === String(v)} onClick={() => onChange(String(v))}
            className={`px-2 text-sm transition-colors ${String(value) === String(v) ? 'bg-ink text-canvas' : 'bg-canvas text-ink hover:bg-hover'}`}>{t}</button>
        ))}
      </div>
    </div>
  )
}

const PRESETS: { name: string; p: PatientInput }[] = [
  { name: 'Lower risk', p: { age: 34, avg_glucose_level: 88, bmi: 23.5, hypertension: 0, heart_disease: 0, gender: 'Female', ever_married: 'No', work_type: 'Private', Residence_type: 'Urban', smoking_status: 'never smoked' } },
  { name: 'Mid risk', p: { age: 55, avg_glucose_level: 125, bmi: 29, hypertension: 0, heart_disease: 0, gender: 'Male', ever_married: 'Yes', work_type: 'Private', Residence_type: 'Rural', smoking_status: 'formerly smoked' } },
  { name: 'Higher risk', p: { age: 74, avg_glucose_level: 228, bmi: 33, hypertension: 1, heart_disease: 1, gender: 'Male', ever_married: 'Yes', work_type: 'Self-employed', Residence_type: 'Urban', smoking_status: 'smokes' } },
]

export function Predictor() {
  const [p, setP] = useState<PatientInput>(DEFAULT_PATIENT)
  const { id: sid, setId: setSid } = useThreshold()
  const [res, setRes] = useState<Prediction | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const strat = STRATEGIES.find((s) => s.id === sid)!
  const set = <K extends keyof PatientInput>(k: K, v: PatientInput[K]) => setP((o) => ({ ...o, [k]: v }))

  useEffect(() => { syncHealth() }, [])
  useEffect(() => {
    let stale = false
    const id = setTimeout(() => { predict(p, strat.tau).then((r) => { if (!stale) { setRes(r); setErr(null) } }).catch((e) => { if (!stale) { console.error('[predict]', e); setErr('request failed') } }) }, 120)
    return () => { stale = true; clearTimeout(id) }
  }, [p, strat.tau])

  const max = useMemo(() => Math.max(0.01, ...(res?.contributions.map((c) => Math.abs(c.shap)) ?? [0])), [res])
  const tier = !res ? null : res.probability >= res.threshold ? 'high' : res.probability >= res.threshold * 0.7 ? 'near' : 'low'
  const badge = { high: ['Above threshold', 'bg-burgundy-dim text-burgundy'], near: ['Near threshold', 'bg-ochre-dim text-ochre'], low: ['Below threshold', 'bg-olive-dim text-olive'] }[tier ?? 'low']
  const pct = res ? res.probability * 100 : 0
  const scaleMax = Math.max(25, Math.ceil((res?.threshold ?? 0.14) * 200), Math.ceil(pct / 5) * 5)

  return (
    <section id="predict" className="scroll-mt-4 px-3 pb-24 pt-28 sm:px-5">
      <div className="mb-10 max-w-3xl">
        <h2 className="text-4xl font-semibold leading-[1.05] tracking-[-0.03em] text-ink sm:text-6xl">Predict, then check the reasoning</h2>
        <p className="mt-4 max-w-[60ch] text-base leading-relaxed text-muted">Move any input. The probability, the alert decision and every SHAP contribution update together, so you can see which factor pushed the score and by how much.</p>
      </div>

      <div className="grid items-start gap-5 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
        {/* Inputs */}
        <div className="border border-line bg-surface p-5 sm:p-6 lg:sticky lg:top-20">
          <div className="mb-5">
            <span className="mb-1.5 block text-[13px] text-muted">Example profiles (illustrative, not real patients)</span>
            <div className="grid grid-cols-3 border border-control" role="group" aria-label="Example profiles">
              {PRESETS.map((x) => (
                <button key={x.name} type="button" onClick={() => setP(x.p)} aria-pressed={JSON.stringify(p) === JSON.stringify(x.p)}
                  className={`px-2 py-2 text-sm transition-colors ${JSON.stringify(p) === JSON.stringify(x.p) ? 'bg-ink text-canvas' : 'bg-canvas text-ink hover:bg-hover'}`}>{x.name}</button>
              ))}
            </div>
          </div>
          <Field label="Alert threshold strategy">
            <select className={selectCls} value={sid} onChange={(e) => setSid(e.target.value)}>
              {STRATEGIES.map((s) => <option key={s.id} value={s.id}>{s.label} (t = {s.tau})</option>)}
            </select>
          </Field>
          <p className="mt-2 text-[13px] leading-relaxed text-muted">{strat.note}</p>

          <div className="mt-6 space-y-4">
            <Slider label="Age" value={p.age} min={1} max={100} step={1} onChange={(v) => set('age', v)} />
            <Slider label="Average glucose" unit=" mg/dL" value={p.avg_glucose_level} min={55} max={272} step={1} onChange={(v) => set('avg_glucose_level', v)} />
            <Slider label="BMI" value={p.bmi} min={10} max={60} step={0.1} onChange={(v) => set('bmi', Math.round(v * 10) / 10)} />
          </div>

          <div className="mt-6 grid grid-cols-2 gap-4">
            <Seg label="Hypertension" value={p.hypertension} options={[[0,"No"],[1,"Yes"]] as [number, string][]} onChange={(v) => set("hypertension", Number(v) as 0 | 1)} />
            <Seg label="Heart disease" value={p.heart_disease} options={[[0,"No"],[1,"Yes"]] as [number, string][]} onChange={(v) => set("heart_disease", Number(v) as 0 | 1)} />
            <Seg label="Gender" value={p.gender} options={[["Male","Male"],["Female","Female"]]} onChange={(v) => set("gender", v as PatientInput["gender"])} />
            <Seg label="Ever married" value={p.ever_married} options={[["Yes","Yes"],["No","No"]]} onChange={(v) => set("ever_married", v as PatientInput["ever_married"])} />
            <Field label="Work type"><select className={selectCls} value={p.work_type} onChange={(e) => set('work_type', e.target.value as PatientInput['work_type'])}>{['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'].map((o) => <option key={o}>{o}</option>)}</select></Field>
            <Seg label="Residence" value={p.Residence_type} options={[["Urban","Urban"],["Rural","Rural"]]} onChange={(v) => set("Residence_type", v as PatientInput["Residence_type"])} />
          </div>
          <div className="mt-4">
            <Field label="Smoking status"><select className={selectCls} value={p.smoking_status} onChange={(e) => set('smoking_status', e.target.value as PatientInput['smoking_status'])}>{['formerly smoked', 'never smoked', 'smokes', 'Unknown'].map((o) => <option key={o}>{o}</option>)}</select></Field>
          </div>
        </div>

        {/* Result */}
        <div className="border border-line bg-surface p-5 sm:p-8" aria-live="polite">
          {err && <p role="alert" className="mb-4 border border-burgundy bg-burgundy-dim p-3 text-sm text-burgundy">The scoring service did not respond. Please try again in a moment.</p>}

          <p className="text-sm text-muted">Estimated stroke probability (calibrated)</p>
          <div className="mt-2 flex flex-wrap items-end justify-between gap-4">
            <p className="font-mono text-7xl font-medium tabular-nums tracking-[-0.04em] text-ink sm:text-8xl">{res ? pct.toFixed(1) : '--'}<span className="text-4xl text-muted sm:text-5xl">%</span></p>
            {res && <span className={`rounded-full px-4 py-1.5 text-sm font-medium ${badge[1]}`}>{badge[0]}</span>}
          </div>

          {res && (
            <div className="mt-5">
              <div className="relative h-2 bg-hover" role="img" aria-label={`Probability ${pct.toFixed(1)} percent against a ${(res.threshold * 100).toFixed(1)} percent alert threshold`}>
                <div className="h-full transition-[width] duration-300" style={{ width: `${Math.min(100, (pct / scaleMax) * 100)}%`, background: tier === 'high' ? 'var(--color-burgundy)' : tier === 'near' ? 'var(--color-ochre)' : 'var(--color-olive)' }} />
                <div className="absolute -top-1.5 h-5 w-px bg-ink" style={{ left: `${(res.threshold * 100 / scaleMax) * 100}%` }} />
              </div>
              <div className="mt-2 flex justify-between font-mono text-[11px] text-muted"><span>0%</span><span>threshold {(res.threshold * 100).toFixed(1)}%</span><span>{scaleMax}%</span></div>
              <p className="mt-4 max-w-[62ch] text-[15px] leading-relaxed text-muted">
                {tier === 'high' ? `At or above the ${(res.threshold * 100).toFixed(1)}% alert threshold (${strat.label}). This profile would be flagged for screening triage.` : `Under the ${(res.threshold * 100).toFixed(1)}% alert threshold (${strat.label}). No single risk factor pushes the score into the alert range.`}
              </p>
            </div>
          )}

          <div className="mt-9 border-t border-line pt-6">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <h3 className="text-lg font-semibold tracking-[-0.01em] text-ink">SHAP feature contributions</h3>
              <span className="font-mono text-[11px] text-muted">log-odds, before calibration</span>
            </div>
            <p className="mt-1 text-[13px] leading-relaxed text-muted">Bars to the right raise risk, bars to the left lower it. The headline probability is calibrated separately.</p>

            {res && (
              <ul className="mt-5">
                <li className="grid grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)_52px] items-center gap-3 border-b border-line py-2 text-[13px]">
                  <span className="text-ink">Baseline <span className="text-muted">average patient</span></span><span /><span className="text-right font-mono text-ink">{fmt(res.baseline)}</span>
                </li>
                {res.contributions.map((c) => (
                  <li key={c.feature} className="grid grid-cols-[minmax(0,1fr)_minmax(0,1.6fr)_52px] items-center gap-3 border-b border-line py-2 text-[13px]">
                    <span className="min-w-0 truncate text-ink">{c.label} <span className="text-muted">{c.display}</span></span>
                    <span className="relative h-3" title={`${c.label}: ${fmt(c.shap)} log-odds`}>
                      <span className="absolute -inset-y-1 left-1/2 w-px bg-muted opacity-60" />
                      <span className="absolute inset-y-0 transition-[width] duration-300" style={{ background: scale(c.shap, max), width: `${(Math.abs(c.shap) / max) * 50}%`, [c.shap >= 0 ? 'left' : 'right']: '50%' }} />
                    </span>
                    <span className={`text-right font-mono ${c.shap > 0 ? 'text-burgundy' : 'text-olive'}`}>{fmt(c.shap)}</span>
                  </li>
                ))}
              </ul>
            )}

            {res && res.local_surrogate && res.local_surrogate.length > 0 && (
              <div className="mt-8 border-t border-line pt-6">
                <div className="flex flex-wrap items-baseline justify-between gap-2">
                  <h3 className="text-lg font-semibold tracking-[-0.01em] text-ink">Local surrogate (LIME-style)</h3>
                  <span className="font-mono text-[11px] text-muted">calibrated-probability units</span>
                </div>
                <p className="mt-1 text-[13px] leading-relaxed text-muted">
                  Sparse linear approximation fitted around this patient&apos;s neighborhood. Bars show shift in estimated probability.
                </p>
                <div className="mt-4">
                  <LimeChart rules={res.local_surrogate} />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </section>
  )
}
