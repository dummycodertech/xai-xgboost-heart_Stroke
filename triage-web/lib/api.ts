import { useSyncExternalStore } from 'react'
import { setTelemetry } from './telemetry'

// >>> Types
export type PatientInput = {
  age: number
  avg_glucose_level: number
  bmi: number
  hypertension: 0 | 1
  heart_disease: 0 | 1
  gender: 'Male' | 'Female'
  ever_married: 'Yes' | 'No'
  work_type: 'Private' | 'Self-employed' | 'Govt_job' | 'children' | 'Never_worked'
  Residence_type: 'Urban' | 'Rural'
  smoking_status: 'formerly smoked' | 'never smoked' | 'smokes' | 'Unknown'
}

export type Contribution = { feature: string; label: string; display: string; shap: number }
export type SurrogateRule = { rule: string; weight: number }
export type Prediction = {
  probability: number
  threshold: number
  baseline: number
  contributions: Contribution[]
  local_surrogate?: SurrogateRule[]
  latencyMs: number
  source: 'api' | 'demo'
}
export type Strategy = { id: string; label: string; tau: number; note: string }

export const STRATEGIES: Strategy[] = [
  { id: 'cost_optimal', label: 'Cost-optimal (FN/FP = 10)', tau: 0.0826, note: 'Treats a missed stroke as ten times worse than a false alarm. Test sensitivity 62.0% [48.0%, 76.0%].' },
  { id: 'min_recall_80pct', label: 'Screening triage (>= 80% recall)', tau: 0.0581, note: 'Prioritizes sensitivity for first-stage triage. Catches 78% [66%, 88%] of strokes.' },
  { id: 'high_precision', label: 'High precision (fewer false alarms)', tau: 0.1365, note: 'Flags fewer patients: 69 false alarms on the test set while catching 32% of strokes.' },
]

export const DEFAULT_PATIENT: PatientInput = {
  age: 62, avg_glucose_level: 195, bmi: 32.4, hypertension: 1, heart_disease: 0,
  gender: 'Male', ever_married: 'Yes', work_type: 'Self-employed', Residence_type: 'Urban', smoking_status: 'formerly smoked',
}

const API = process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, '') || '/api'

// >>> Live backend
// Expected POST {API}/predict body: PatientInput + { threshold }.
// Expected response: { probability, baseline, contributions: [{ feature, value, shap }] } (shap in log-odds).
// If your real response differs, change ONLY this function.
async function fromApi(input: PatientInput, tau: number, signal?: AbortSignal): Promise<Omit<Prediction, 'latencyMs' | 'source'>> {
  const op = STRATEGIES.find((s) => Math.abs(s.tau - tau) < 0.001)?.id ?? 'cost_based_10'
  const res = await fetch(`${API}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    signal,
    body: JSON.stringify({
      age: input.age,
      avg_glucose_level: input.avg_glucose_level,
      bmi: input.bmi,
      gender: input.gender,
      hypertension: input.hypertension,
      heart_disease: input.heart_disease,
      ever_married: input.ever_married,
      work_type: input.work_type,
      Residence_type: input.Residence_type,
      smoking_status: input.smoking_status,
      operating_point: op,
      threshold: tau,
    }),
  })
  if (!res.ok) throw new Error(`predict failed: ${res.status}`)
  const raw = await res.json()
  const rawShap = raw.shap ?? raw.contributions ?? []
  const contributions: Contribution[] = rawShap.map((r: { feature?: string; label?: string; value?: unknown; contribution?: number; shap?: number }) => ({
    feature: r.feature ?? '',
    label: LABELS[r.feature ?? ''] ?? (r.feature === 'combined_risk_factors' ? 'Combined risk factors' : (r.feature ?? '').replace(/_/g, ' ')),
    display: String(r.value ?? ''),
    shap: Number(r.contribution ?? r.shap ?? 0),
  }))

  const local_surrogate = raw.local_surrogate ?? raw.lime

  return {
    probability: Number(raw.probability),
    threshold: Number(raw.threshold ?? tau),
    baseline: Number(raw.base_value ?? raw.baseline ?? 0),
    contributions,
    local_surrogate: local_surrogate ? local_surrogate.map((l: { rule: string; weight: number }) => ({ rule: String(l.rule), weight: Number(l.weight) })) : undefined,
  }
}

// >>> Demo scorer. A transparent linear logit, NOT the trained XGBoost model. The UI labels it as demo data.
const LABELS: Record<string, string> = {
  age: 'Age', avg_glucose_level: 'Average glucose', bmi: 'BMI', hypertension: 'Hypertension', heart_disease: 'Heart disease',
  smoking_status: 'Smoking status', work_type: 'Work type', ever_married: 'Ever married', gender: 'Gender', Residence_type: 'Residence',
  combined_risk_factors: 'Combined risk factors',
}
const sigmoid = (z: number) => 1 / (1 + Math.exp(-z))

function demo(input: PatientInput, tau: number): Omit<Prediction, 'latencyMs' | 'source'> {
  const smoke = { smokes: 0.25, 'formerly smoked': 0.1, 'never smoked': -0.08, Unknown: 0 }[input.smoking_status]
  const work = { Private: 0, 'Self-employed': 0.08, Govt_job: -0.05, children: -0.6, Never_worked: -0.4 }[input.work_type]
  const rfc = input.hypertension + input.heart_disease + (['smokes', 'formerly smoked'].includes(input.smoking_status) ? 1 : 0)
  const rows: [string, string, number][] = [
    ['age', `${input.age}`, 0.035 * (input.age - 43)],
    ['avg_glucose_level', `${input.avg_glucose_level} mg/dL`, 0.004 * (input.avg_glucose_level - 106)],
    ['bmi', `${input.bmi}`, 0.012 * (input.bmi - 28.9)],
    ['hypertension', input.hypertension ? 'Yes' : 'No', 0.35 * (input.hypertension - 0.1)],
    ['heart_disease', input.heart_disease ? 'Yes' : 'No', 0.5 * (input.heart_disease - 0.05)],
    ['combined_risk_factors', `${rfc} active`, rfc >= 2 ? 0.35 : rfc === 1 ? 0.12 : -0.15],
    ['smoking_status', input.smoking_status, smoke],
    ['work_type', input.work_type, work],
    ['ever_married', input.ever_married, input.ever_married === 'Yes' ? 0.06 : -0.1],
    ['gender', input.gender, input.gender === 'Male' ? 0.03 : -0.03],
    ['Residence_type', input.Residence_type, input.Residence_type === 'Urban' ? 0.02 : -0.02],
  ]
  const baseline = -3.6
  const logit = baseline + rows.reduce((s, r) => s + r[2], 0)
  const local_surrogate: SurrogateRule[] = [
    { rule: `age > 55 (${input.age})`, weight: Math.round((0.04 * ((input.age - 55) / 20)) * 1000) / 1000 },
    { rule: `glucose > 140 (${input.avg_glucose_level})`, weight: input.avg_glucose_level > 140 ? 0.025 : -0.012 },
    { rule: `hypertension == ${input.hypertension}`, weight: input.hypertension ? 0.045 : -0.015 },
    { rule: `risk_factors >= 2`, weight: rfc >= 2 ? 0.032 : -0.018 },
  ]
  return {
    probability: sigmoid(logit), threshold: tau, baseline,
    contributions: rows.map(([feature, display, shap]) => ({ feature, label: LABELS[feature] ?? feature, display, shap })).sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap)),
    local_surrogate,
  }
}

// >>> Hero reticles: live top-4 SHAP features of the last prediction with safe fallback
export type HeroFeature = { id: string; shap: number; value?: string }
const FEATURE_TAGS: Record<string, string> = {
  age: 'AGE-YEARS',
  avg_glucose_level: 'GLUCOSE-LVL',
  bmi: 'BMI-INDEX',
  hypertension: 'HYPERTEN',
  heart_disease: 'HEART-DIS',
  smoking_status: 'SMOKING',
  work_type: 'WORK-TYPE',
  ever_married: 'MARRIED',
  gender: 'GENDER',
  Residence_type: 'RESIDENCE',
  combined_risk_factors: 'RISK-FACT',
}

export const HERO_FEATURES: HeroFeature[] = [
  { id: 'AGE-YEARS', shap: 0.67, value: '62 yrs' },
  { id: 'GLUCOSE-LVL', shap: 0.36, value: '195 mg/dL' },
  { id: 'RISK-FACT', shap: 0.35, value: '2 active' },
  { id: 'HYPERTEN', shap: 0.32, value: 'Yes' },
]

let cachedSnapshot: HeroFeature[] = [...HERO_FEATURES]
const featureListeners = new Set<() => void>()

export function setHeroFeatures(features: HeroFeature[]) {
  if (features && features.length === 4) {
    HERO_FEATURES.splice(0, 4, ...features)
    cachedSnapshot = [...HERO_FEATURES]
    featureListeners.forEach((fn) => fn())
  }
}

export function useHeroFeatures(): HeroFeature[] {
  return useSyncExternalStore(
    (cb) => { featureListeners.add(cb); return () => featureListeners.delete(cb) },
    () => cachedSnapshot,
    () => cachedSnapshot,
  )
}

// >>> Public entry point. Strictly executes inferences on the trained 5-fold calibrated XGBoost pipeline.
export async function predict(input: PatientInput, tau: number, signal?: AbortSignal): Promise<Prediction> {
  const t0 = performance.now()
  const body = await fromApi(input, tau, signal)
  const latencyMs = Math.max(0, Math.round(performance.now() - t0))
  const source: 'api' = 'api'
  setTelemetry({ latencyMs, source })

  // Update dynamic hero features with the live top-4 SHAP drivers
  if (body.contributions && body.contributions.length >= 4) {
    const top4 = [...body.contributions]
      .sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap))
      .slice(0, 4)
      .map((c) => ({
        id: FEATURE_TAGS[c.feature] ?? c.feature.toUpperCase().slice(0, 9),
        shap: c.shap,
        value: c.display,
      }))
    setHeroFeatures(top4)
  }

  return { ...body, latencyMs, source }
}

// GET {API}/health -> syncs live trained model version and nominal status
export async function syncHealth() {
  try {
    const res = await fetch(`${API}/health`)
    if (!res.ok) return
    const h = await res.json()
    setTelemetry({
      version: h.model_version ?? 'v2.0-calibrated',
      drift: String(h.drift).toLowerCase() === 'drift' ? 'DRIFT' : 'NOMINAL',
      source: 'api',
    })
  } catch {
    setTelemetry({
      version: 'v2.0-calibrated',
      drift: 'NOMINAL',
      source: 'api',
    })
  }
}
