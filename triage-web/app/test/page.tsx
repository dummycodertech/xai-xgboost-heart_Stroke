'use client'
import { useEffect, useState } from 'react'
import { Hero } from '@/components/hero/Hero'

// >>> Test page: slow performance.now to 1/50 speed to capture transitions reliably
if (typeof window !== 'undefined') {
  const origNow = performance.now.bind(performance)
  const t0 = origNow()
  const factor = 1 / 50
  // @ts-expect-error override for test page
  window.__origPerformanceNow = origNow
  performance.now = () => t0 + (origNow() - t0) * factor
}

export default function TestHarnessPage() {
  const [mounted, setMounted] = useState(false)
  useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) return null

  return (
    <main>
      <div id="test-banner" className="bg-line px-4 py-2 font-mono text-xs text-ink">
        TEST HARNESS: clock running at 1/50 speed (50ms real = 1ms virtual)
      </div>
      <Hero />
    </main>
  )
}
