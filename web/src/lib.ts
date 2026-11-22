export type Patient = {
  age: number
  glucose: number
  bmi: number
  gender: string
  hypertension: 0 | 1
  heart: 0 | 1
  married: string
  work: string
  residence: string
  smoking: string
}

export type Result = {
  probability: number
  threshold: number
  operating_point: string
  decision: 'HIGH_RISK' | 'BELOW_THRESHOLD'
  base_value: number
  raw_margin: number
  shap: { feature: string; value: string; contribution: number }[]
  local_surrogate?: { rule: string; weight: number }[]
  lime: { rule: string; weight: number }[]
  units?: Record<string, string>
  demo: boolean
}

export type OperatingPoint = {
  id: string
  label: string
  threshold: number
  description: string
  recall: number
  recall_ci: [number, number]
  precision: number
  precision_ci: [number, number]
  false_alarms: number
  false_alarms_ci: [number, number]
  nns: number
  nns_ci: [number, number]
  matrix: { tp: number; fp: number; fn: number; tn: number }
}

export type ModelMeta = {
  model_version: string
  model_type: string
  default_threshold: number
  calibration_method: string
  operating_points: Record<string, OperatingPoint>
  evaluation_dataset: {
    n_test: number
    n_positives: number
    n_negatives: number
    split_label: string
  }
}

export const API = (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, '')

// v2 evaluated fallback metadata from reports/final_test_report.json
export const DEFAULT_META: ModelMeta = {
  model_version: 'v2.0-calibrated',
  model_type: 'Tuned XGBoost + 5-Fold Isotonic Calibration',
  default_threshold: 0.0826,
  calibration_method: 'isotonic',
  operating_points: {
    cost_based_10: {
      id: 'cost_based_10',
      label: 'Cost-Optimal (FN/FP = 10)',
      threshold: 0.0826,
      description: 'Balances false negatives 10x higher than false alarms; minimizes clinical misclassification cost.',
      recall: 0.62,
      recall_ci: [0.48, 0.76],
      precision: 0.119,
      precision_ci: [0.095, 0.144],
      false_alarms: 229,
      false_alarms_ci: [203, 256],
      nns: 8.4,
      nns_ci: [6.9, 10.6],
      matrix: { tp: 31, fp: 229, fn: 19, tn: 743 },
    },
    min_recall_80pct: {
      id: 'min_recall_80pct',
      label: 'Target 80% OOF Recall',
      threshold: 0.0581,
      description: 'Selected out-of-fold to achieve >= 80% screening sensitivity.',
      recall: 0.78,
      recall_ci: [0.66, 0.88],
      precision: 0.117,
      precision_ci: [0.100, 0.135],
      false_alarms: 293,
      false_alarms_ci: [266, 321],
      nns: 8.5,
      nns_ci: [7.4, 10.0],
      matrix: { tp: 39, fp: 293, fn: 11, tn: 679 },
    },
    high_precision: {
      id: 'high_precision',
      label: 'High Precision (Lower False Alarms)',
      threshold: 0.1365,
      description: 'Prioritizes positive predictive value; reduces false alarms to 69 while catching 32% of strokes.',
      recall: 0.32,
      recall_ci: [0.20, 0.46],
      precision: 0.188,
      precision_ci: [0.121, 0.263],
      false_alarms: 69,
      false_alarms_ci: [53, 86],
      nns: 5.3,
      nns_ci: [3.8, 8.3],
      matrix: { tp: 16, fp: 69, fn: 34, tn: 903 },
    },
  },
  evaluation_dataset: {
    n_test: 1022,
    n_positives: 50,
    n_negatives: 972,
    split_label: 'Stratified 80/20 held-out evaluation (1,022 patients, 50 strokes, 95% bootstrap CIs)',
  },
}

export async function fetchMetadata(): Promise<ModelMeta> {
  if (!API) return DEFAULT_META
  try {
    const res = await fetch(`${API}/meta`)
    if (res.ok) return await res.json()
  } catch {
    // If API warmup is underway, return default evaluated metadata
  }
  return DEFAULT_META
}

export function warmupService() {
  if (!API) return
  fetch(`${API}/health`).catch(() => {})
}

// DEMO ONLY: additive toy model used strictly when VITE_API_URL is unset.
function mock(p: Patient, opKey: string): Result {
  const smoke: Record<string, number> = { smokes: 0.35, 'formerly smoked': 0.25, 'never smoked': -0.1, Unknown: 0 }
  const rfc = (p.hypertension ? 1 : 0) + (p.heart ? 1 : 0) + (['smokes', 'formerly smoked'].includes(p.smoking) ? 1 : 0)
  const c: [string, string, number][] = [
    ['age', `${p.age}`, 0.06 * (p.age - 45)],
    ['avg_glucose_level', `${p.glucose} mg/dL`, Math.max(-0.4, 0.012 * (p.glucose - 105))],
    ['bmi', `${p.bmi}`, 0.01 * (p.bmi - 27)],
    ['hypertension', p.hypertension ? 'Yes' : 'No', p.hypertension ? 0.7 : -0.1],
    ['heart_disease', p.heart ? 'Yes' : 'No', p.heart ? 0.8 : -0.05],
    ['combined_risk_factors', `${rfc} active`, rfc >= 2 ? 0.45 : rfc === 1 ? 0.15 : -0.2],
    ['smoking_status', p.smoking, smoke[p.smoking] ?? 0],
    ['ever_married', p.married, p.married === 'Yes' ? 0.1 : -0.1],
    ['work_type', p.work, p.work === 'children' ? -0.8 : p.work === 'Self-employed' ? 0.1 : 0],
  ]
  const base = -2.6
  const total = base + c.reduce((a, x) => a + x[2], 0)
  const shap = c.map(([feature, value, contribution]) => ({ feature, value, contribution })).sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
  const lime = [
    { rule: `age ${p.age > 61 ? '> 61' : '<= 61'}`, weight: p.age > 61 ? 0.08 : -0.04 },
    { rule: `avg_glucose_level ${p.glucose > 114 ? '> 114' : '<= 114'}`, weight: p.glucose > 114 ? 0.06 : -0.02 },
    { rule: `hypertension = ${p.hypertension ? 'Yes' : 'No'}`, weight: p.hypertension ? 0.05 : -0.01 },
    { rule: `combined_risk_factors ${rfc >= 2 ? '>= 2' : '< 2'}`, weight: rfc >= 2 ? 0.04 : -0.02 },
    { rule: `smoking_status = ${p.smoking}`, weight: smoke[p.smoking] ? 0.03 : -0.01 },
  ]
  const probability = 1 / (1 + Math.exp(-total))
  const threshold = DEFAULT_META.operating_points[opKey]?.threshold ?? 0.0826
  return {
    probability,
    threshold,
    operating_point: opKey,
    decision: probability >= threshold ? 'HIGH_RISK' : 'BELOW_THRESHOLD',
    base_value: base,
    raw_margin: total,
    shap,
    local_surrogate: lime,
    lime,
    units: {
      shap: 'log-odds (raw model score)',
      'local_surrogate (LIME-style)': 'calibrated probability points',
      probability: 'calibrated risk percentage',
    },
    demo: true,
  }
}

export async function predict(p: Patient, opKey: string, signal: AbortSignal): Promise<Result> {
  // If API URL is NOT set, run demo mode
  if (!API) {
    await new Promise(r => setTimeout(r, 180))
    return mock(p, opKey)
  }

  // When API URL is configured: NEVER fall back to mock predictions on failure.
  const body = {
    age: p.age,
    avg_glucose_level: p.glucose,
    bmi: p.bmi,
    gender: p.gender,
    hypertension: p.hypertension,
    heart_disease: p.heart,
    ever_married: p.married,
    work_type: p.work,
    Residence_type: p.residence,
    smoking_status: p.smoking,
    operating_point: opKey,
  }

  const r = await fetch(`${API}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  })

  if (!r.ok) {
    if (r.status === 422) {
      throw new Error('Please check your patient input values (e.g. age between 0-100, glucose 50-300).')
    }
    if (r.status === 429) {
      throw new Error('Rate limit reached. Please wait a moment before trying another prediction.')
    }
    throw new Error(`The model service returned status ${r.status}. Please try again shortly.`)
  }

  const data = await r.json()
  return { ...data, demo: false }
}
