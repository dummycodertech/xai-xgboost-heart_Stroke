'use client'
import { useTelemetry } from '@/lib/telemetry'

// >>> MLOps strip: model version, measured inference latency, drift status
export function TelemetryBar() {
  const t = useTelemetry()
  const lat = t.latencyMs === null ? '--' : t.latencyMs < 1 ? '<1ms' : `${t.latencyMs}ms`
  const slow = (t.latencyMs ?? 0) > 200
  const drift = t.drift === 'DRIFT'
  const Cell = ({ k, children }: { k: string; children: React.ReactNode }) => (
    <span className="flex items-center gap-2"><span className="text-muted">{k}</span>{children}</span>
  )
  return (
    <div className="flex h-9 items-center justify-between gap-4 border-b border-line px-3 font-mono text-[11px] sm:px-5">
      <div className="flex min-w-0 items-center gap-5 sm:gap-8">
        <Cell k="model"><span className="text-ink">{t.version}</span></Cell>
        <Cell k="latency"><span className={slow ? 'text-ochre' : 'text-ink'}>{lat}</span></Cell>
        <Cell k="drift">
          <span className={`flex items-center gap-1.5 ${drift ? 'text-burgundy' : 'text-olive'}`}>
            <span className={`h-1.5 w-1.5 rounded-full ${drift ? 'bg-burgundy' : 'bg-olive'}`} />{t.drift}
          </span>
        </Cell>
      </div>
      <span className="hidden text-muted sm:block">{t.source === 'api' ? 'API connected' : 'Demo data, no API connected'}</span>
    </div>
  )
}
