// >>> Real pixel-brightness tracking. Input: RGBA bytes of a small render of the subject's luminance.
// Output: local maxima in normalised, top-left-origin coordinates, strongest first.
export type Peak = { x: number; y: number; s: number }

export function findPeaks(buf: Uint8Array, w: number, h: number, max = 28, minDist = 2, floor = 0.05): Peak[] {
  const a = new Float32Array(w * h)
  for (let i = 0; i < w * h; i++) a[i] = buf[i * 4] / 255

  // 5-tap Gaussian-like separable blur to eliminate single-pixel spikes and prevent jitter
  const b = new Float32Array(w * h)
  for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
    let s = a[y * w + x] * 4
    s += (a[y * w + x - 1] + a[y * w + x + 1] + a[(y - 1) * w + x] + a[(y + 1) * w + x]) * 2
    s += a[(y - 1) * w + x - 1] + a[(y - 1) * w + x + 1] + a[(y + 1) * w + x - 1] + a[(y + 1) * w + x + 1]
    b[y * w + x] = s / 16
  }

  const out: Peak[] = []
  while (out.length < max) {
    let best = -1, bi = 0
    for (let i = 0; i < b.length; i++) if (b[i] > best) { best = b[i]; bi = i }
    if (best < floor) break
    const px = bi % w, py = (bi / w) | 0
    out.push({ x: (px + 0.5) / w, y: 1 - (py + 0.5) / h, s: best }) // readPixels is bottom-up
    for (let y = Math.max(0, py - minDist); y <= Math.min(h - 1, py + minDist); y++)
      for (let x = Math.max(0, px - minDist); x <= Math.min(w - 1, px + minDist); x++) b[y * w + x] = 0
  }
  return out
}

// >>> Shared bus between the WebGL engine (writer) and the 2D HUD (reader). Mutable on purpose: no React renders per frame.
export type Mode = 'heart' | 'brain'
export type HudBus = {
  peaks: Peak[]
  pointer: { x: number; y: number }
  fx: { glitch: number; ascii: number }
  fps: number
  frame: number
  epoch: number
  mode: Mode
}
export const createBus = (): HudBus => ({ peaks: [], pointer: { x: 0.5, y: 0.5 }, fx: { glitch: 0, ascii: 0 }, fps: 60, frame: 0, epoch: 0, mode: 'heart' })
