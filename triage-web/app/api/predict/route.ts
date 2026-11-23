import { NextResponse } from 'next/server'

const BACKEND_URL = process.env.BACKEND_URL || 'http://127.0.0.1:8000'

export async function POST(req: Request) {
  try {
    const body = await req.json()
    // Map operating points cleanly
    const op = body.operating_point === 'cost_optimal' ? 'cost_based_10' : (body.operating_point || 'cost_based_10')

    const payload = {
      age: Number(body.age),
      avg_glucose_level: Number(body.avg_glucose_level),
      bmi: Number(body.bmi),
      gender: body.gender,
      hypertension: Number(body.hypertension),
      heart_disease: Number(body.heart_disease),
      ever_married: body.ever_married,
      work_type: body.work_type,
      Residence_type: body.Residence_type,
      smoking_status: body.smoking_status,
      operating_point: op,
      threshold: body.threshold ? Number(body.threshold) : undefined,
    }

    const res = await fetch(`${BACKEND_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })

    if (!res.ok) {
      const errText = await res.text()
      return NextResponse.json({ error: `Backend error: ${res.status}`, detail: errText }, { status: res.status })
    }

    const data = await res.json()
    return NextResponse.json(data)
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : String(err)
    return NextResponse.json({ error: 'Failed to connect to trained XGBoost service', detail: message }, { status: 502 })
  }
}
