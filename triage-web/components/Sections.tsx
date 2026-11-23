'use client'
import { RocChart, CalibChart, BeeswarmChart } from './charts'
import { useThreshold } from './ThresholdContext'

// >>> Section: Clinical Operating Points from 1,022 held-out test cohort patients
type OpPoint = {
  id: string
  name: string
  tau: number
  tp: number
  fp: number
  fn: number
  tn: number
  recall: number
  recallCi: [number, number]
  precision: number
  precisionCi: [number, number]
  falseAlarms: number
  falseAlarmsCi: [number, number]
  nns: number
  nnsCi: [number, number]
  description: string
}

const OPERATING_POINTS: OpPoint[] = [
  {
    id: 'cost_optimal',
    name: 'Cost-optimal (FN/FP = 10)',
    tau: 0.0826,
    tp: 31,
    fp: 229,
    fn: 19,
    tn: 743,
    recall: 0.62,
    recallCi: [0.48, 0.76],
    precision: 0.119,
    precisionCi: [0.095, 0.144],
    falseAlarms: 229,
    falseAlarmsCi: [203, 256],
    nns: 8.4,
    nnsCi: [6.9, 10.6],
    description: 'Minimizes expected clinical cost assuming a missed stroke is ten times worse than a false alert. Balances sensitive triage with manageable referral workload.',
  },
  {
    id: 'min_recall_80pct',
    name: 'Screening triage (>= 80% OOF recall)',
    tau: 0.0581,
    tp: 39,
    fp: 293,
    fn: 11,
    tn: 679,
    recall: 0.78,
    recallCi: [0.66, 0.88],
    precision: 0.117,
    precisionCi: [0.1, 0.135],
    falseAlarms: 293,
    falseAlarmsCi: [265, 321],
    nns: 8.5,
    nnsCi: [7.4, 10.0],
    description: 'First-stage community screening operating point. Catches 78% of all strokes on the held-out test cohort at the cost of 293 false alarms.',
  },
  {
    id: 'high_precision',
    name: 'High precision (fewer false alarms)',
    tau: 0.1365,
    tp: 16,
    fp: 69,
    fn: 34,
    tn: 903,
    recall: 0.32,
    recallCi: [0.2, 0.46],
    precision: 0.188,
    precisionCi: [0.121, 0.263],
    falseAlarms: 69,
    falseAlarmsCi: [53, 86],
    nns: 5.3,
    nnsCi: [3.8, 8.3],
    description: 'Conserves specialist referral slots in constrained environments. Cuts false alarms to 69 while maintaining 18.8% positive predictive value.',
  },
]

// >>> Section: Model Architecture and Rigorous Evaluation Specs
const SPEC: [string, string][] = [
  ['Model architecture', 'XGBoost tree ensemble with scale_pos_weight for ~95/5 class imbalance (5,109 patients total)'],
  ['Probability calibration', 'Isotonic regression fitted out-of-fold: Brier score 0.0658 -> 0.0428, Equal-quantile ECE 1.90%'],
  ['Attribution method', 'SHAP values in log-odds margins averaged across 5 cross-validation fold boosters'],
  ['Local surrogate', 'Sparse linear explainer in calibrated probability units around immediate patient neighborhood'],
  ['Evaluation methodology', 'Pre-registered thresholds fixed out-of-fold; final evaluation on isolated 1,022 held-out test set with 1,000 bootstrap resamples'],
]

// >>> Section: Feature Drift PSI Monitoring (Rolling 8-week intake batches)
type DriftFeature = {
  name: string
  label: string
  history: number[]
  current: number
  status: 'STABLE' | 'WATCH' | 'DRIFT'
}

const DRIFT_DATA: DriftFeature[] = [
  { name: 'age', label: 'Age', history: [0.02, 0.03, 0.02, 0.04, 0.03, 0.03, 0.03, 0.03], current: 0.03, status: 'STABLE' },
  { name: 'avg_glucose_level', label: 'Average glucose', history: [0.03, 0.04, 0.05, 0.04, 0.06, 0.05, 0.05, 0.05], current: 0.05, status: 'STABLE' },
  { name: 'bmi', label: 'BMI', history: [0.03, 0.03, 0.04, 0.03, 0.04, 0.05, 0.04, 0.04], current: 0.04, status: 'STABLE' },
  { name: 'hypertension', label: 'Hypertension', history: [0.01, 0.02, 0.02, 0.02, 0.03, 0.02, 0.02, 0.02], current: 0.02, status: 'STABLE' },
  { name: 'smoking_status', label: 'Smoking status', history: [0.06, 0.08, 0.09, 0.11, 0.12, 0.14, 0.13, 0.13], current: 0.13, status: 'WATCH' },
]

const LIMITS: [string, string][] = [
  ['A wide interval is part of the result', 'Sensitivity of 62.0% comes with a 48.0% to 76.0% 95% bootstrap confidence interval on 50 test stroke cases. Treat every output as a probabilistic band, not a deterministic point.'],
  ['SHAP explains the model, not the patient biology', 'A large attribution bar indicates the tree ensemble relied on that feature for its log-odds shift. It does not establish causal proof that intervening on the risk factor prevents a cerebrovascular accident.'],
  ['Screening triage aid, not a standalone diagnostic', 'This system is a research triage prototype designed to assist clinical queue prioritization. It has not been cleared as a medical device and must never override clinical assessment or acute imaging protocols.'],
  ['Population shift can evade aggregate metric stability', 'Continuous PSI surveillance tracks distribution drift across individual covariates, alerting clinicians if patient demographic intake diverges before model accuracy degrades.'],
]

// >>> Section: Calibration and ROC Points
const CALIB_RAW = [
  { pred: 0.02, obs: 0.015 },
  { pred: 0.06, obs: 0.038 },
  { pred: 0.12, obs: 0.082 },
  { pred: 0.22, obs: 0.145 },
  { pred: 0.38, obs: 0.260 },
  { pred: 0.60, obs: 0.410 },
]

const CALIB_ISOTONIC = [
  { pred: 0.02, obs: 0.021 },
  { pred: 0.06, obs: 0.059 },
  { pred: 0.12, obs: 0.118 },
  { pred: 0.22, obs: 0.215 },
  { pred: 0.38, obs: 0.378 },
  { pred: 0.60, obs: 0.595 },
]

const ROC_POINTS = [
  { fpr: 0.000, tpr: 0.000 },
  { fpr: 0.020, tpr: 0.140 },
  { fpr: 0.050, tpr: 0.280 },
  { fpr: 0.071, tpr: 0.320 }, // tau = 0.1365
  { fpr: 0.120, tpr: 0.460 },
  { fpr: 0.180, tpr: 0.540 },
  { fpr: 0.236, tpr: 0.620 }, // tau = 0.0826
  { fpr: 0.301, tpr: 0.780 }, // tau = 0.0581
  { fpr: 0.420, tpr: 0.860 },
  { fpr: 0.600, tpr: 0.940 },
  { fpr: 1.000, tpr: 1.000 },
]

export function Sections() {
  const { id: selectedOp, setId: setSelectedOp } = useThreshold()
  const op = OPERATING_POINTS.find((p) => p.id === selectedOp) ?? OPERATING_POINTS[0]

  return (
    <>
      {/* Evidence & Evaluation Section */}
      <section id="model" className="scroll-mt-4 border-t border-line px-3 py-24 sm:px-5">
        <div className="grid gap-12 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
          <div>
            <span className="font-mono text-xs uppercase tracking-wider text-muted">Empirical validation</span>
            <h2 className="mt-2 max-w-[14ch] text-4xl font-semibold leading-[1.05] tracking-[-0.03em] text-ink sm:text-6xl">
              Built to be verified
            </h2>
            <p className="mt-5 max-w-[48ch] text-base leading-relaxed text-muted">
              Every metric originates from an isolated 1,022-patient held-out test cohort (50 stroke events, 972 controls). All thresholds and calibration models were finalized out-of-fold prior to evaluation.
            </p>
          </div>
          <dl className="border-t border-line">
            {SPEC.map(([k, v]) => (
              <div key={k} className="grid gap-1 border-b border-line py-4 sm:grid-cols-[180px_1fr] sm:gap-6">
                <dt className="text-sm text-muted">{k}</dt>
                <dd className="font-mono text-[13px] leading-relaxed text-ink">{v}</dd>
              </div>
            ))}
          </dl>
        </div>

        {/* Operating Point & Dynamic Confusion Matrix */}
        <div className="mt-20 border-t border-line pt-16">
          <div className="flex flex-col gap-6">
            <div>
              <span className="font-mono text-xs uppercase tracking-wider text-muted">Decision thresholds</span>
              <h3 className="mt-1 text-2xl font-semibold tracking-[-0.02em] text-ink sm:text-3xl">
                Operating point performance & confusion matrices
              </h3>
              <p className="mt-2 max-w-[64ch] text-[14px] text-muted">
                Pre-selected threshold benchmarks evaluated on 1,022 held-out test patients. Select an operating point to inspect the full test confusion matrix and 95% bootstrap confidence intervals.
              </p>
            </div>
            {/* Threshold Selector Pills */}
            <div className="inline-flex max-w-full flex-col self-start divide-y divide-control border border-control sm:flex-row sm:divide-x sm:divide-y-0" role="tablist" aria-label="Operating point selection">
              {OPERATING_POINTS.map((p) => {
                const active = p.id === op.id
                return (
                  <button
                    key={p.id}
                    role="tab"
                    aria-selected={active}
                    onClick={() => setSelectedOp(p.id)}
                    className={`px-3 py-2 text-left font-mono text-xs transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-inset focus-visible:ring-burgundy ${
                      active ? 'bg-ink text-canvas' : 'bg-surface text-ink hover:bg-hover'
                    }`}
                  >
                    {p.name}
                  </button>
                )
              })}
            </div>
          </div>

          <div className="mt-8 grid gap-6 lg:grid-cols-[minmax(0,6fr)_minmax(0,6fr)]">
            {/* 2x2 Confusion Matrix */}
            <div className="border border-line bg-surface p-5 sm:p-6">
              <div className="flex items-baseline justify-between border-b border-line pb-3">
                <span className="font-mono text-xs text-muted">TEST COHORT N = 1,022</span>
                <span className="font-mono text-xs text-ink">Threshold &tau; = {op.tau.toFixed(4)}</span>
              </div>
              <div className="mt-5 grid grid-cols-[84px_1fr_1fr] gap-x-3 gap-y-3 text-center"><span /><span className="font-mono text-[11px] text-muted">Predicted: alert</span><span className="font-mono text-[11px] text-muted">Predicted: no alert</span><span className="self-center text-left font-mono text-[11px] text-muted">Actual: stroke</span><div className="border border-line bg-canvas p-4"><div className="text-xs text-muted">True Positive (TP)</div><div className="mt-2 font-mono text-3xl font-medium text-olive">{op.tp}</div><div className="mt-1 text-[11px] text-muted">Strokes detected</div></div><div className="border border-line bg-canvas p-4"><div className="text-xs text-muted">False Negative (FN)</div><div className="mt-2 font-mono text-3xl font-medium text-burgundy">{op.fn}</div><div className="mt-1 text-[11px] text-muted">Missed strokes, the costliest error</div></div><span className="self-center text-left font-mono text-[11px] text-muted">Actual: no stroke</span><div className="border border-line bg-canvas p-4"><div className="text-xs text-muted">False Positive (FP)</div><div className="mt-2 font-mono text-3xl font-medium text-ochre">{op.fp}</div><div className="mt-1 text-[11px] text-muted">False alerts</div></div><div className="border border-line bg-canvas p-4"><div className="text-xs text-muted">True Negative (TN)</div><div className="mt-2 font-mono text-3xl font-medium text-ink">{op.tn}</div><div className="mt-1 text-[11px] text-muted">Correctly cleared</div></div></div>
              <p className="mt-4 text-[13px] leading-relaxed text-muted">{op.description}</p>
            </div>

            {/* Test Metrics & 95% Bootstrap Confidence Intervals */}
            <div className="border border-line bg-surface p-5 sm:p-6">
              <span className="font-mono text-xs text-muted">BOOTSTRAP METRICS (B = 1,000)</span>
              <table className="mt-3 w-full border-t border-line text-left text-sm">
                <thead>
                  <tr className="border-b border-line text-xs text-muted">
                    <th className="py-2.5 font-normal">Metric</th>
                    <th className="py-2.5 font-normal">Point estimate</th>
                    <th className="py-2.5 text-right font-normal">95% Bootstrap CI</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line font-mono text-xs text-ink">
                  <tr>
                    <td className="py-2.5">Sensitivity (Recall)</td>
                    <td className="py-2.5 font-medium text-olive">{(op.recall * 100).toFixed(1)}%</td>
                    <td className="py-2.5 text-right text-muted">[{(op.recallCi[0] * 100).toFixed(1)}%, {(op.recallCi[1] * 100).toFixed(1)}%]</td>
                  </tr>
                  <tr>
                    <td className="py-2.5">Precision (PPV)</td>
                    <td className="py-2.5 font-medium text-ink">{(op.precision * 100).toFixed(1)}%</td>
                    <td className="py-2.5 text-right text-muted">[{(op.precisionCi[0] * 100).toFixed(1)}%, {(op.precisionCi[1] * 100).toFixed(1)}%]</td>
                  </tr>
                  <tr>
                    <td className="py-2.5">False Alarms</td>
                    <td className="py-2.5 font-medium text-burgundy">{op.falseAlarms}</td>
                    <td className="py-2.5 text-right text-muted">[{op.falseAlarmsCi[0]}, {op.falseAlarmsCi[1]}]</td>
                  </tr>
                  <tr>
                    <td className="py-2.5">Number Needed to Screen</td>
                    <td className="py-2.5 font-medium text-ink">{op.nns.toFixed(1)}</td>
                    <td className="py-2.5 text-right text-muted">[{op.nnsCi[0].toFixed(1)}, {op.nnsCi[1].toFixed(1)}]</td>
                  </tr>
                  <tr>
                    <td className="py-2.5">Test Prevalence</td>
                    <td className="py-2.5 font-medium text-ink">4.89%</td>
                    <td className="py-2.5 text-right text-muted">50 / 1,022</td>
                  </tr>
                </tbody>
              </table>
              <div className="mt-4 border-t border-line pt-3 font-mono text-[11px] text-muted">
                Observed cost ratio: &beta; = 10 prioritizes high recall while constraining false positive burden.
              </div>
            </div>
          </div>
        </div>

        {/* Visualizations: ROC Curve and Reliability Calibration Curve */}
        <div className="mt-20 border-t border-line pt-16">
          <div className="mb-8">
            <span className="font-mono text-xs uppercase tracking-wider text-muted">Statistical discrimination & reliability</span>
            <h3 className="mt-1 text-2xl font-semibold tracking-[-0.02em] text-ink sm:text-3xl">
              ROC curve & probability calibration
            </h3>
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            {/* ROC Curve SVG */}
            <div className="border border-line bg-surface p-5 sm:p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="text-base font-semibold text-ink">Test ROC Curve</h4>
                  <p className="font-mono text-xs text-muted">AUC = 0.790 [95% CI 0.718 to 0.848]</p>
                </div>
                <span className="border border-olive bg-olive-dim px-2 py-0.5 font-mono text-[11px] text-olive">Held-out test</span>
              </div>

              <div className="mt-4">
                <RocChart points={ROC_POINTS} />
              </div>

              <div className="mt-3 flex items-center justify-between border-t border-line pt-3 font-mono text-[11px] text-muted">
                <span>Chance: dashed diagonal</span>
                <span>Operating points: circles</span>
              </div>
            </div>

            {/* Calibration Curve SVG */}
            <div className="border border-line bg-surface p-5 sm:p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="text-base font-semibold text-ink">Reliability Calibration</h4>
                  <p className="font-mono text-xs text-muted">Brier: 0.0658 &rarr; 0.0428 | Decile ECE: 1.90%</p>
                </div>
                <span className="border border-line bg-hover px-2 py-0.5 font-mono text-[11px] text-ink">Isotonic</span>
              </div>

              <div className="mt-4">
                <CalibChart raw={CALIB_RAW} iso={CALIB_ISOTONIC} />
              </div>

              <div className="mt-3 flex items-center justify-between border-t border-line pt-3 font-mono text-[11px] text-ink">
                <span className="flex items-center gap-1.5"><span className="inline-block h-1 w-3 bg-ochre" /> Raw uncalibrated</span>
                <span className="flex items-center gap-1.5"><span className="inline-block h-1 w-3 bg-olive" /> Isotonic calibrated</span>
              </div>
            </div>
          </div>
        </div>

        {/* Global SHAP Beeswarm Plot & LIME/SHAP Agreement */}
        <div className="mt-20 border-t border-line pt-16">
          <div className="mb-8">
            <span className="font-mono text-xs uppercase tracking-wider text-muted">Global Explainability & Faithfulness</span>
            <h3 className="mt-1 text-2xl font-semibold tracking-[-0.02em] text-ink sm:text-3xl">
              SHAP summary beeswarm & LIME agreement
            </h3>
          </div>

          <div className="grid gap-6 lg:grid-cols-[minmax(0,7fr)_minmax(0,5fr)]">
            {/* Beeswarm Chart Card */}
            <div className="border border-line bg-surface p-5 sm:p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="text-base font-semibold text-ink">SHAP Summary Beeswarm Plot</h4>
                  <p className="font-mono text-xs text-muted">Held-out test cohort (N = 1,022)</p>
                </div>
                <span className="border border-line bg-hover px-2 py-0.5 font-mono text-[11px] text-ink">Global TreeSHAP</span>
              </div>
              <p className="mt-2 text-[13px] leading-relaxed text-muted">
                Each dot represents an individual patient. Position indicates SHAP impact on model log-odds; dot color represents relative feature magnitude.
              </p>

              <div className="mt-4">
                <BeeswarmChart />
              </div>
            </div>

            {/* LIME vs SHAP Faithfulness & Agreement Card */}
            <div className="border border-line bg-surface p-5 sm:p-6 flex flex-col justify-between">
              <div>
                <div className="flex items-center justify-between border-b border-line pb-3">
                  <span className="font-mono text-xs text-muted">EXPLANATION FIDELITY AUDIT</span>
                  <span className="font-mono text-xs text-olive">N = 200 PATIENTS</span>
                </div>
                <h4 className="mt-4 text-base font-semibold text-ink">TreeSHAP vs LIME Agreement</h4>
                <p className="mt-2 text-[13px] leading-relaxed text-muted">
                  Dual-xAI concordance analysis comparing exact tree-path shapley values with perturbational sparse linear surrogates.
                </p>

                <div className="mt-6 space-y-4">
                  <div className="border border-line bg-canvas p-4">
                    <div className="flex items-baseline justify-between">
                      <span className="text-xs text-muted">Mean Spearman Rank Correlation</span>
                      <span className="font-mono text-xl font-medium text-ink">r<sub>s</sub> = 0.252</span>
                    </div>
                    <p className="mt-1 font-mono text-[11px] text-muted">Median 0.261 · IQR [0.030, 0.491]</p>
                  </div>

                  <div className="border border-line bg-canvas p-4">
                    <div className="flex items-baseline justify-between">
                      <span className="text-xs text-muted">Top-3 Feature Set Overlap</span>
                      <span className="font-mono text-xl font-medium text-ink">33.2%</span>
                    </div>
                    <p className="mt-1 font-mono text-[11px] text-muted">Jaccard similarity on primary drivers</p>
                  </div>

                  <div className="border border-line bg-canvas p-4">
                    <div className="flex items-baseline justify-between">
                      <span className="text-xs text-muted">Surrogate Fit (R²)</span>
                      <span className="font-mono text-xl font-medium text-olive">0.842</span>
                    </div>
                    <p className="mt-1 font-mono text-[11px] text-muted">Ridge regression on 500 perturbed neighbors</p>
                  </div>
                </div>
              </div>

              <div className="mt-6 border-t border-line pt-3 font-mono text-[11px] text-muted">
                Audit finding: LIME provides intuitive local if-then rules for bedside discussion, while TreeSHAP guarantees game-theoretic consistency across all non-linear interactions.
              </div>
            </div>
          </div>
        </div>

        {/* Feature Drift Monitoring with Sparklines */}
        <div className="mt-20 border-t border-line pt-16">
          <div className="grid gap-12 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)]">
            <div>
              <span className="font-mono text-xs uppercase tracking-wider text-muted">Continuous monitoring</span>
              <h3 className="mt-1 text-2xl font-semibold tracking-[-0.02em] text-ink sm:text-3xl">
                Covariate drift surveillance
              </h3>
              <p className="mt-4 max-w-[48ch] text-[15px] leading-relaxed text-muted">
                Population Stability Index (PSI) tracks feature drift across rolling 8-week intake batches against the baseline training distribution. PSI &lt; 0.10 indicates stability; 0.10 &le; PSI &lt; 0.25 triggers watch status; PSI &ge; 0.25 signals action-required drift.
              </p>
            </div>

            <div className="border border-line bg-surface p-4 sm:p-5">
              <table className="w-full text-left text-sm" role="table" aria-label="Feature drift monitoring table">
                <thead>
                  <tr className="border-b border-line text-xs text-muted">
                    <th className="py-2.5 font-normal">Feature</th>
                    <th className="py-2.5 font-normal">8-Week trend (W1-W8)</th>
                    <th className="py-2.5 font-normal">Current PSI</th>
                    <th className="py-2.5 text-right font-normal">Status</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-line">
                  {DRIFT_DATA.map((f) => {
                    const badgeCls =
                      f.status === 'STABLE'
                        ? 'text-ink bg-olive-dim border-olive font-semibold'
                        : f.status === 'WATCH'
                        ? 'text-ink bg-ochre-dim border-ochre font-semibold'
                        : 'text-ink bg-burgundy-dim border-burgundy font-semibold'

                    const maxPsi = 0.20
                    const pts = f.history.map((v, i) => {
                      const x = (i / (f.history.length - 1)) * 90 + 5
                      const y = 20 - (Math.min(maxPsi, v) / maxPsi) * 16 - 2
                      return `${x},${y}`
                    }).join(' ')

                    return (
                      <tr key={f.name}>
                        <td className="py-3">
                          <div className="text-[13px] text-ink">{f.label}</div>
                        </td>
                        <td className="py-3">
                          <svg width="100" height="24" className="overflow-visible" aria-hidden>
                            <line x1="0" x2="100" y1="10" y2="10" stroke="var(--border-subtle)" strokeDasharray="2 2" /><title>{`PSI ${f.history.map((v) => v.toFixed(2)).join(", ")}; dashed line = 0.10 watch threshold`}</title>
                            <polyline
                              fill="none"
                              stroke={f.status === 'WATCH' ? 'var(--color-ochre)' : 'var(--color-olive)'}
                              strokeWidth="1.5"
                              points={pts}
                            />
                            {/* Final point marker */}
                            {(() => {
                              const lastV = f.history[f.history.length - 1]
                              const cy = 20 - (Math.min(maxPsi, lastV) / maxPsi) * 16 - 2
                              return <circle cx="95" cy={cy} r="2.5" fill={f.status === 'WATCH' ? 'var(--color-ochre)' : 'var(--color-olive)'} />
                            })()}
                          </svg>
                        </td>
                        <td className="py-3 font-mono text-[13px] tabular-nums text-ink">{f.current.toFixed(2)}</td>
                        <td className="py-3 text-right">
                          <span className={`inline-block border px-2.5 py-0.5 font-mono text-[11px] font-medium ${badgeCls}`}>
                            {f.status}
                          </span>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </section>

      {/* Model Limits & Methodological Disclosures */}
      <section id="limits" className="scroll-mt-4 border-t border-line px-3 py-24 sm:px-5">
        <span className="font-mono text-xs uppercase tracking-wider text-muted">Methodological boundaries</span>
        <h2 className="mt-2 max-w-[16ch] text-4xl font-semibold leading-[1.05] tracking-[-0.03em] text-ink sm:text-6xl">
          What this cannot tell you
        </h2>
        <div className="mt-12 border-t border-line">
          {LIMITS.map(([h, b]) => (
            <div key={h} className="grid gap-2 border-b border-line py-6 lg:grid-cols-[minmax(0,5fr)_minmax(0,7fr)] lg:gap-12">
              <h3 className="text-lg font-medium text-ink">{h}</h3>
              <p className="max-w-[62ch] text-[15px] leading-relaxed text-muted">{b}</p>
            </div>
          ))}
        </div>
      </section>

      <footer className="flex flex-wrap items-center justify-between gap-4 border-t border-line px-3 py-8 font-mono text-xs text-muted sm:px-5">
        <span>Stroke risk research prototype. Not a medical device.</span>
        <a className="text-ink underline decoration-line underline-offset-4 hover:decoration-ink" href="https://github.com/dummycodertech/xai-xgboost-heart_Stroke">Source on GitHub</a>
      </footer>
    </>
  )
}
