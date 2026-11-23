// >>> Shader-side copy of the Diagnostic Monitor palette. Keep in sync with globals.css [data-theme='dark'].
export const GL = {
  ink: '#11100f',
  surface: '#1c1b18',
  line: '#2e2c27',
  cream: '#f6f2e9',
  stone: '#a39e93',
  olive: '#8fa885',
  burgundy: '#d44d6e',
  ochre: '#d9a05b',
} as const

// Raw sRGB floats. Not THREE.Color: that would convert to linear and darken everything.
export function vec3(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16)
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255]
}
