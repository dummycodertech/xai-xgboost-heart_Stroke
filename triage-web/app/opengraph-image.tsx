import { ImageResponse } from 'next/og'

export const runtime = 'edge'
export const alt = 'Triage - Stroke Risk & Explainability System'
export const size = { width: 1200, height: 630 }
export const contentType = 'image/png'

export default async function Image() {
  return new ImageResponse(
    (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          flexDirection: 'column',
          justifyContent: 'space-between',
          padding: '80px',
          background: '#141311',
          color: '#F6F2E9',
          fontFamily: 'sans-serif',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
          <div
            style={{
              padding: '6px 14px',
              border: '1px solid #801835',
              background: '#231519',
              color: '#D44D6E',
              fontFamily: 'monospace',
              fontSize: '18px',
              letterSpacing: '0.1em',
            }}
          >
            TRIAGE // XAI-EVAL
          </div>
          <div style={{ color: '#A39E93', fontFamily: 'monospace', fontSize: '18px' }}>
            RESEARCH PROTOTYPE
          </div>
        </div>

        <div>
          <div
            style={{
              fontSize: '64px',
              fontWeight: 700,
              letterSpacing: '-0.03em',
              lineHeight: 1.1,
              maxWidth: '900px',
            }}
          >
            Not just a score, a reason. Every risk comes with its why.
          </div>
          <div
            style={{
              marginTop: '24px',
              fontSize: '24px',
              color: '#A39E93',
              maxWidth: '800px',
              lineHeight: 1.4,
            }}
          >
            Calibrated XGBoost stroke risk scoring, multi-fold SHAP explanations, and real-time input drift monitoring.
          </div>
        </div>

        <div
          style={{
            display: 'flex',
            gap: '40px',
            borderTop: '1px solid #2E2C27',
            paddingTop: '32px',
            fontFamily: 'monospace',
            fontSize: '16px',
            color: '#8FA885',
          }}
        >
          <div>AUC: 0.790 [0.718, 0.848]</div>
          <div>SENSITIVITY: 62.0% [48.0%, 76.0%]</div>
          <div>BRIER: 0.0427</div>
        </div>
      </div>
    ),
    { ...size }
  )
}
