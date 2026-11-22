import { useEffect, useRef, useState } from 'react'
import Heart from './Heart'
import {
  API,
  DEFAULT_META,
  ModelMeta,
  Patient,
  Result,
  fetchMetadata,
  predict,
  warmupService,
} from './lib'

const init: Patient = {
  age: 62,
  glucose: 195,
  bmi: 32.4,
  gender: 'Male',
  hypertension: 1,
  heart: 0,
  married: 'Yes',
  work: 'Self-employed',
  residence: 'Urban',
  smoking: 'formerly smoked',
}

const pct = (x: number) => `${(x * 100).toFixed(1)}%`
const sgn = (x: number) => `${x >= 0 ? '+' : ''}${x.toFixed(2)}`

function Waterfall({ r }: { r: Result }) {
  const steps = [{ label: 'Baseline', note: 'average patient', from: 0, to: r.base_value, kind: 'base' }]
  let cur = r.base_value
  r.shap.forEach(s => {
    steps.push({
      label: s.feature.replace(/_/g, ' '),
      note: s.value,
      from: cur,
      to: cur + s.contribution,
      kind: s.contribution >= 0 ? 'up' : 'down',
    })
    cur += s.contribution
  })
  const all = [r.base_value, 0, ...steps.map(s => s.to)]
  const lo = Math.min(...all)
  const hi = Math.max(...all)
  const span = hi - lo || 1
  const x = (v: number) => ((v - lo) / span) * 100

  return (
    <div className="wf" role="table" aria-label="SHAP waterfall in log-odds">
      {steps.map(s => (
        <div className="wf-row" key={s.label} role="row">
          <div className="wf-l">
            <span>{s.label}</span>
            <em>{s.note}</em>
          </div>
          <div className="wf-t">
            <i
              className={`wf-b ${s.kind}`}
              style={{
                left: `${x(Math.min(s.from, s.to))}%`,
                width: `${Math.max(0.8, Math.abs(x(s.to) - x(s.from)))}%`,
              }}
            />
          </div>
          <b className="mono">{s.kind === 'base' ? sgn(s.to) : sgn(s.to - s.from)}</b>
        </div>
      ))}
    </div>
  )
}

function Lime({ r }: { r: Result }) {
  const m = Math.max(...r.lime.map(l => Math.abs(l.weight))) || 1
  return (
    <div className="lime">
      {r.lime.map(l => (
        <div className="lime-row" key={l.rule}>
          <span className="lime-l">{l.rule}</span>
          <div className="lime-t">
            <i
              className={l.weight >= 0 ? 'up' : 'down'}
              style={{
                width: `${(Math.abs(l.weight) / m) * 50}%`,
                [l.weight >= 0 ? 'left' : 'right']: '50%',
              }}
            />
          </div>
          <b className="mono">{sgn(l.weight)}</b>
        </div>
      ))}
    </div>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  )
}

export default function App() {
  const [theme, setTheme] = useState<'dark' | 'light'>('dark')
  const [p, setP] = useState<Patient>(init)
  const [opKey, setOpKey] = useState<string>('cost_based_10')
  const [meta, setMeta] = useState<ModelMeta>(DEFAULT_META)
  const [apiConnected, setApiConnected] = useState<boolean>(false)
  const [r, setR] = useState<Result | null>(null)
  const [busy, setBusy] = useState(false)
  const [slow, setSlow] = useState(false)
  const [err, setErr] = useState('')
  const [live, setLive] = useState(true)
  const ac = useRef<AbortController>()

  useEffect(() => {
    document.documentElement.dataset.theme = theme
  }, [theme])

  useEffect(() => {
    warmupService()
    fetchMetadata()
      .then(m => {
        setMeta(m)
        if (API) setApiConnected(true)
      })
      .catch(() => {
        // Retain fallback metadata
      })
  }, [])

  const run = async () => {
    ac.current?.abort()
    const c = (ac.current = new AbortController())
    setBusy(true)
    setErr('')
    setSlow(false)
    const t = setTimeout(() => setSlow(true), 3000)

    try {
      const res = await predict(p, opKey, c.signal)
      setR(res)
      if (API) setApiConnected(true)
    } catch (e) {
      if (!c.signal.aborted) {
        setErr(e instanceof Error ? e.message : 'The model service is unreachable. Try again in a minute.')
        setR(null) // Never display stale or simulated prediction when real service fails
      }
    } finally {
      clearTimeout(t)
      if (!c.signal.aborted) setBusy(false)
    }
  }

  useEffect(() => {
    if (!live && r) return
    const id = setTimeout(run, r ? 280 : 0)
    return () => clearTimeout(id)
  }, [p, opKey, live]) // eslint-disable-line

  const set = <K extends keyof Patient>(k: K, v: Patient[K]) => setP(o => ({ ...o, [k]: v }))

  const currentOp = meta.operating_points[opKey] ?? meta.operating_points['cost_based_10'] ?? DEFAULT_META.operating_points['cost_based_10']
  const high = r?.decision === 'HIGH_RISK'
  const top = r ? [...r.shap].filter(s => s.contribution > 0).slice(0, 3) : []
  const sh3 = r ? r.shap.slice(0, 3).map(s => s.feature) : []
  const li3 = r ? r.lime.slice(0, 3).map(l => l.rule) : []
  const agree = sh3.filter(f => li3.some(l => l.startsWith(f))).length
  const M = currentOp.matrix

  return (
    <>
      <header className="nav">
        <a className="brand" href="#top">
          <svg viewBox="0 0 100 90" aria-hidden>
            <path d="M50 85C10 55 0 35 0 22 0 8 12 0 25 0c10 0 20 6 25 16C55 6 65 0 75 0c13 0 25 8 25 22 0 13-10 33-50 63z" />
          </svg>
          Stroke triage
        </a>
        <nav>
          <a href="#predict">Predict</a>
          <a href="#model">Model</a>
          <a href="#limits">Limits</a>
        </nav>
        <span className={`chip ${API && apiConnected ? 'ok' : ''}`}>
          {API ? (apiConnected ? 'API v2.0 connected' : 'Connecting to API...') : 'Demo mode, simulated outputs'}
        </span>
        <button
          className="ghost"
          onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
          aria-label="Toggle color theme"
        >
          {theme === 'dark' ? 'Light' : 'Dark'}
        </button>
      </header>

      <main id="top">
        <section className="hero">
          <h1>Stroke risk screening that shows its work</h1>
          <p className="lede">
            Enter a patient profile and inspect the calibrated risk probability alongside the exact factors that moved the score.
          </p>
          <div className="stage">
            <Heart />
            <div className="callout l">
              <b className="mono">{pct(currentOp.recall)}</b>
              <span>test sensitivity [{pct(currentOp.recall_ci[0])}, {pct(currentOp.recall_ci[1])}]</span>
            </div>
            <div className="callout r">
              <b className="mono">{currentOp.threshold.toFixed(4)}</b>
              <span>alert threshold ({currentOp.label})</span>
            </div>
          </div>
          <div className="cta">
            <a className="btn" href="#predict">Try a patient</a>
            <a className="btn ghost" href="#model">See the evidence</a>
          </div>
        </section>

        <section id="predict" className="wrap">
          <h2>Predict and explain</h2>
          <div className="grid">
            <form
              className="card form"
              onSubmit={e => {
                e.preventDefault()
                run()
              }}
            >
              <Field label="Triage threshold strategy">
                <select value={opKey} onChange={e => setOpKey(e.target.value)}>
                  {Object.values(meta.operating_points).map(op => (
                    <option key={op.id} value={op.id}>
                      {op.label} (τ = {op.threshold})
                    </option>
                  ))}
                </select>
              </Field>
              <small className="mut" style={{ marginTop: '-8px' }}>
                {currentOp.description}
              </small>

              <Field label={`Age: ${p.age}`}>
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={p.age}
                  onChange={e => set('age', +e.target.value)}
                />
              </Field>

              <Field label={`Average glucose: ${p.glucose} mg/dL`}>
                <input
                  type="range"
                  min={50}
                  max={300}
                  value={p.glucose}
                  onChange={e => set('glucose', +e.target.value)}
                />
              </Field>

              <Field label={`BMI: ${p.bmi}`}>
                <input
                  type="range"
                  min={10}
                  max={100}
                  step={0.1}
                  value={p.bmi}
                  onChange={e => set('bmi', +e.target.value)}
                />
              </Field>

              <div className="two">
                <Field label="Hypertension">
                  <select
                    value={p.hypertension}
                    onChange={e => set('hypertension', +e.target.value as 0 | 1)}
                  >
                    <option value={0}>No</option>
                    <option value={1}>Yes</option>
                  </select>
                </Field>
                <Field label="Heart disease">
                  <select
                    value={p.heart}
                    onChange={e => set('heart', +e.target.value as 0 | 1)}
                  >
                    <option value={0}>No</option>
                    <option value={1}>Yes</option>
                  </select>
                </Field>
                <Field label="Gender">
                  <select
                    value={p.gender}
                    onChange={e => set('gender', e.target.value)}
                  >
                    <option>Male</option>
                    <option>Female</option>
                  </select>
                </Field>
                <Field label="Ever married">
                  <select
                    value={p.married}
                    onChange={e => set('married', e.target.value)}
                  >
                    <option>Yes</option>
                    <option>No</option>
                  </select>
                </Field>
                <Field label="Work type">
                  <select
                    value={p.work}
                    onChange={e => set('work', e.target.value)}
                  >
                    {['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'].map(o => (
                      <option key={o}>{o}</option>
                    ))}
                  </select>
                </Field>
                <Field label="Residence">
                  <select
                    value={p.residence}
                    onChange={e => set('residence', e.target.value)}
                  >
                    <option>Urban</option>
                    <option>Rural</option>
                  </select>
                </Field>
              </div>

              <Field label="Smoking status">
                <select
                  value={p.smoking}
                  onChange={e => set('smoking', e.target.value)}
                >
                  {['formerly smoked', 'never smoked', 'smokes', 'Unknown'].map(o => (
                    <option key={o}>{o}</option>
                  ))}
                </select>
              </Field>

              <label className="check">
                <input
                  type="checkbox"
                  checked={live}
                  onChange={e => setLive(e.target.checked)}
                />
                Update results as I change inputs
              </label>

              <button className="btn" type="submit" disabled={busy}>
                {busy ? 'Calculating...' : 'Predict stroke risk'}
              </button>
            </form>

            <div className={`card out ${busy ? 'busy' : ''}`} aria-live="polite">
              {err && (
                <div className="alert">
                  <span>{err}</span>
                  <button className="ghost" onClick={run}>Retry</button>
                </div>
              )}
              {slow && !err && (
                <div className="note">
                  The model service is waking up on Render free tier (spin-up takes ~1 minute). Please wait...
                </div>
              )}
              {!r && !err && <div className="skeleton" />}
              {r && (
                <>
                  <div className="res">
                    <div>
                      <span className="mut">Estimated stroke probability (calibrated)</span>
                      <div className="big mono">{pct(r.probability)}</div>
                    </div>
                    <span className={`badge ${high ? 'hi' : 'lo'}`}>
                      {high ? 'High risk' : 'Below threshold'}
                    </span>
                  </div>

                  <div
                    className="gauge"
                    role="img"
                    aria-label={`Probability ${pct(r.probability)}, threshold ${pct(r.threshold)}`}
                  >
                    <i
                      style={{ width: `${Math.min(100, r.probability * 100)}%` }}
                      className={high ? 'hi' : 'lo'}
                    />
                    <u style={{ left: `${r.threshold * 100}%` }}>
                      <em>{pct(r.threshold)} threshold</em>
                    </u>
                  </div>

                  <p className="sum">
                    {high
                      ? `Exceeds the ${pct(r.threshold)} alert threshold (${currentOp.label}), driven primarily by ${top.map(t => t.feature.replace(/_/g, ' ')).join(', ') || 'combined patient factors'}.`
                      : `Under the ${pct(r.threshold)} alert threshold (${currentOp.label}). No single risk factor pushes the score into the screening triage alert range.`}
                  </p>

                  <h3>
                    SHAP: feature contributions <small>units: log-odds (raw model score)</small>
                  </h3>
                  <p className="mut" style={{ fontSize: '0.8rem', margin: '4px 0 10px' }}>
                    SHAP bars explain the booster's raw log-odds score before probability calibration. Positive values increase risk. The headline stroke risk is isotonic-calibrated.
                  </p>
                  <Waterfall r={r} />

                  <h3>
                    Local surrogate (LIME-style) rules <small>units: calibrated probability points (% risk)</small>
                  </h3>
                  <p className="mut" style={{ fontSize: '0.8rem', margin: '4px 0 10px' }}>
                    Local surrogate (LIME-style) linear rules estimate the direct change in calibrated risk probability around this patient's profile. Agrees with SHAP on {agree} of 3 top factors.
                  </p>
                  <Lime r={r} />

                  {r.demo && (
                    <p className="note">
                      Running in demo mode with a local toy model. Configure VITE_API_URL to connect the live API.
                    </p>
                  )}
                </>
              )}
            </div>
          </div>
        </section>

        <section id="model" className="wrap">
          <h2>What the model has shown so far</h2>
          <p className="mut narrow">
            {meta.evaluation_dataset.split_label}. Evaluated on held-out test data ({meta.evaluation_dataset.n_test} patients, {meta.evaluation_dataset.n_positives} strokes) with 1,000-sample bootstrap 95% confidence intervals.
          </p>

          <div style={{ margin: '0 0 20px', display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center' }}>
            <span className="mut" style={{ fontSize: '0.85rem' }}>Select operating point:</span>
            {Object.values(meta.operating_points).map(op => (
              <button
                key={op.id}
                className="ghost"
                style={
                  opKey === op.id
                    ? { borderColor: 'var(--powder)', color: 'var(--powder)', fontWeight: 600 }
                    : {}
                }
                onClick={() => setOpKey(op.id)}
              >
                {op.label} (τ = {op.threshold})
              </button>
            ))}
          </div>

          <div className="metrics">
            <div className="card">
              <span className="mut">Recall (Sensitivity)</span>
              <b className="mono">{pct(currentOp.recall)}</b>
              <small>95% CI: [{pct(currentOp.recall_ci[0])}, {pct(currentOp.recall_ci[1])}]</small>
            </div>
            <div className="card">
              <span className="mut">Precision (PPV)</span>
              <b className="mono">{pct(currentOp.precision)}</b>
              <small>95% CI: [{pct(currentOp.precision_ci[0])}, {pct(currentOp.precision_ci[1])}]</small>
            </div>
            <div className="card">
              <span className="mut">False Alarms (FP)</span>
              <b className="mono">{currentOp.false_alarms}</b>
              <small>95% CI: [{currentOp.false_alarms_ci[0]}, {currentOp.false_alarms_ci[1]}] out of {meta.evaluation_dataset.n_negatives}</small>
            </div>
            <div className="card">
              <span className="mut">Number Needed to Screen</span>
              <b className="mono">{currentOp.nns.toFixed(1)}</b>
              <small>95% CI: [{currentOp.nns_ci[0].toFixed(1)}, {currentOp.nns_ci[1].toFixed(1)}] to catch 1 stroke</small>
            </div>
          </div>

          <div className="cm card" aria-label="Confusion matrix">
            <div className="tn">
              <b className="mono">{M.tn}</b>
              healthy, cleared
            </div>
            <div className="fp">
              <b className="mono">{M.fp}</b>
              healthy, flagged for check
            </div>
            <div className="fn">
              <b className="mono">{M.fn}</b>
              strokes missed
            </div>
            <div className="tp">
              <b className="mono">{M.tp}</b>
              strokes caught ({M.tp} / {meta.evaluation_dataset.n_positives})
            </div>
          </div>
        </section>

        <section id="limits" className="wrap narrow">
          <h2>Limits & Clinical Scope</h2>
          <p>
            This is an explainable AI research prototype trained on a public dataset of 5,109 patients (249 stroke cases, 4.87% prevalence). It is not an FDA-cleared or CE-marked diagnostic device and must not guide acute clinical care decisions.
          </p>
          <p>
            Screening triage deliberately sets operating thresholds to prioritize sensitivity (e.g. 62%–78% recall), which means the majority of positive alerts are false alarms requiring follow-up clinical examination (PPV 11%–19%).
          </p>
          <p>
            External validity has not been evaluated on secondary cohorts (e.g. BRFSS lacks serum glucose, residence, and employment features and relies on self-report). All patient inputs are evaluated purely in volatile memory; no patient data is logged, stored, or transmitted to third-party analytics.
          </p>
        </section>
      </main>

      <footer className="wrap">
        <span>Stroke triage research prototype · Model {meta.model_version}</span>
        <a href="https://github.com/dummycodertech/xai-xgboost-heart_Stroke">Source on GitHub</a>
      </footer>
    </>
  )
}
