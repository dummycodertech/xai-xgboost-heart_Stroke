import { ImageResponse } from 'next/og'

export const runtime = 'edge'
export const size = { width: 32, height: 32 }
export const contentType = 'image/png'

export default function Icon() {
  return new ImageResponse(
    (
      <div
        style={{
          width: '100%',
          height: '100%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          background: '#141311',
          color: '#F6F2E9',
          border: '1px solid #801835',
          fontFamily: 'monospace',
          fontWeight: 700,
          fontSize: 18,
        }}
      >
        T
      </div>
    ),
    { ...size }
  )
}
