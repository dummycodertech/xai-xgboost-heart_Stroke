'use client'
import { useEffect, useRef, useState } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { Engine } from './engine'
import type { HudBus, Mode } from './tracker'

type Props = { mode: Mode; hud: HudBus; reduce: boolean; active: boolean }

// >>> R3F owns the canvas, resize and render loop. The Engine owns the actual multi-pass pipeline.
function Stage({ mode, hud, reduce }: Omit<Props, 'active'>) {
  const gl = useThree((s) => s.gl)
  const size = useThree((s) => s.size)
  const eng = useRef<Engine | null>(null)

  useEffect(() => {
    const font = getComputedStyle(document.documentElement).getPropertyValue('--font-mono').trim() || 'monospace'
    const e = new Engine(gl, hud, reduce, font)
    eng.current = e
    e.resize(size.width, size.height, gl.getPixelRatio())
    e.setMode(mode)

    // Reduced motion: render one static frame and stop loop entirely
    if (reduce) {
      e.frame(0.016)
    }

    // WebGL context loss & restore handlers
    const canvas = gl.domElement
    const onLost = (ev: Event) => {
      ev.preventDefault()
      console.warn('WebGL context lost')
    }
    const onRestored = () => {
      console.info('WebGL context restored, restoring scene')
      if (eng.current) {
        eng.current.resize(size.width, size.height, gl.getPixelRatio())
        eng.current.setMode(mode)
      }
    }
    canvas.addEventListener('webglcontextlost', onLost)
    canvas.addEventListener('webglcontextrestored', onRestored)

    return () => {
      canvas.removeEventListener('webglcontextlost', onLost)
      canvas.removeEventListener('webglcontextrestored', onRestored)
      e.dispose()
      eng.current = null
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gl, hud, reduce])

  useEffect(() => {
    eng.current?.resize(size.width, size.height, gl.getPixelRatio())
  }, [size, gl])

  useEffect(() => {
    eng.current?.setMode(mode)
    if (reduce) eng.current?.frame(0.016)
  }, [mode, reduce])

  // Stop render loop completely when reduced motion is preferred
  useFrame((_, dt) => {
    if (!reduce) {
      eng.current?.frame(Math.min(dt, 0.05))
    }
  }, 1)

  return null
}

function Fallback() {
  return (
    <div className="absolute inset-0 grid place-items-center bg-canvas text-center font-mono text-xs text-muted">
      <p>3D view needs WebGL 2.<br />Predictions below are unaffected.</p>
    </div>
  )
}

export default function HeartCanvas({ active, ...rest }: Props) {
  const [ok, setOk] = useState<boolean | null>(null)
  useEffect(() => {
    try {
      setOk(!!document.createElement('canvas').getContext('webgl2'))
    } catch {
      setOk(false)
    }
  }, [])

  if (ok === null) return null
  if (!ok) return <Fallback />

  // frameloop stops completely when off-screen, tab hidden, or prefers-reduced-motion
  const loopMode = active && !rest.reduce ? 'always' : 'never'

  return (
    <Canvas
      className="!absolute inset-0"
      role="img"
      dpr={[1, 1.5]}
      flat
      linear
      frameloop={loopMode}
      gl={{ antialias: false, alpha: false, powerPreference: 'high-performance' }}
      aria-label="Interactive 3D anatomical scan"
    >
      <Stage {...rest} />
    </Canvas>
  )
}
