import { useSyncExternalStore } from 'react'

export type Telemetry = {
  version: string
  latencyMs: number | null
  drift: 'NOMINAL' | 'DRIFT'
  source: 'api' | 'demo'
}

// >>> Tiny external store so the monitor bar and the predictor share one source of truth
let state: Telemetry = { version: 'XGB-SHAP v2.4', latencyMs: null, drift: 'NOMINAL', source: 'demo' }
const subs = new Set<() => void>()

export function setTelemetry(patch: Partial<Telemetry>) {
  state = { ...state, ...patch }
  subs.forEach((f) => f())
}

export function useTelemetry(): Telemetry {
  return useSyncExternalStore(
    (cb) => { subs.add(cb); return () => { subs.delete(cb) } },
    () => state,
    () => state,
  )
}
