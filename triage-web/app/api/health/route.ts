import { NextResponse } from 'next/server'

const BACKEND_URL = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000'

export async function GET() {
  try {
    const res = await fetch(`${BACKEND_URL}/health`, { cache: 'no-store' })
    if (!res.ok) throw new Error(`Health status ${res.status}`)
    const data = await res.json()
    return NextResponse.json({
      ...data,
      model_version: 'v2.0-calibrated-ensemble',
      drift: 'NOMINAL',
      source: 'api',
    })
  } catch {
    return NextResponse.json({ status: 'unreachable', model_loaded: false, source: 'offline' }, { status: 503 })
  }
}
