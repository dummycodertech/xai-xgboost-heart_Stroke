import { useEffect, useRef } from 'react'
import * as THREE from 'three'
import { ParametricGeometry } from 'three/examples/jsm/geometries/ParametricGeometry.js'

const S = 0.1
const pt = (u: number, v: number, t: THREE.Vector3) => {
  const a = u * Math.PI * 2, p = v * Math.PI, s = Math.sin(p)
  const X = 16 * Math.sin(a) ** 3
  const Y = 13 * Math.cos(a) - 5 * Math.cos(2 * a) - 2 * Math.cos(3 * a) - Math.cos(4 * a)
  t.set(X * s * S, (Y + 2.5) * s * S, 8 * Math.cos(p) * S)
}
const dot = () => { const c = document.createElement('canvas'); c.width = c.height = 64; const g = c.getContext('2d')!; const r = g.createRadialGradient(32, 32, 0, 32, 32, 32); r.addColorStop(0, 'rgba(255,255,255,1)'); r.addColorStop(1, 'rgba(255,255,255,0)'); g.fillStyle = r; g.fillRect(0, 0, 64, 64); return new THREE.CanvasTexture(c) }
const bump = (x: number, c: number, w: number) => Math.exp(-(((x - c) / w) ** 2))

export default function Heart() {
  const ref = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const el = ref.current!
    let renderer: THREE.WebGLRenderer
    try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true }) } catch { el.classList.add('nogl'); return }
    const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2)); el.appendChild(renderer.domElement)
    const scene = new THREE.Scene(), cam = new THREE.PerspectiveCamera(38, 1, 0.1, 50); cam.position.z = 6.4
    const red = new THREE.Color('#F24452'), blue = new THREE.Color('#93C5FD')
    scene.add(new THREE.AmbientLight('#93C5FD', 0.55))
    const key = new THREE.DirectionalLight('#F8FAFC', 1.5); key.position.set(3, 4, 5); scene.add(key)
    const rim = new THREE.DirectionalLight('#93C5FD', 1.3); rim.position.set(-4, -2, -3); scene.add(rim)
    const pl = new THREE.PointLight('#F24452', 8, 12); pl.position.set(0, 0, 2.5); scene.add(pl)
    const tex = dot()
    const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, color: red, transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending, depthWrite: false }))
    glow.scale.setScalar(8); glow.position.z = -1; scene.add(glow)
    const geo = new ParametricGeometry(pt, 26, 15), group = new THREE.Group(); scene.add(group)
    group.add(new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: red, emissive: red, emissiveIntensity: 0.35, flatShading: true, roughness: 0.35, metalness: 0.15 })))
    const wire = new THREE.LineSegments(new THREE.WireframeGeometry(geo), new THREE.LineBasicMaterial({ color: '#F8FAFC', transparent: true, opacity: 0.22, blending: THREE.AdditiveBlending }))
    wire.scale.setScalar(1.004); group.add(wire)
    // packets: data points running across the heart surface
    const M = 80, pk = Array.from({ length: M }, () => ({ u: Math.random(), v0: Math.random(), sp: 0.04 + Math.random() * 0.08 }))
    const pkGeo = new THREE.BufferGeometry(); pkGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(M * 3), 3))
    group.add(new THREE.Points(pkGeo, new THREE.PointsMaterial({ map: tex, color: blue, size: 0.16, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false })))
    // orbiting protective signals
    const N = matchMedia('(max-width: 700px)').matches ? 120 : 240
    const orb = Array.from({ length: N }, () => ({ r: 2.1 + Math.random() * 1.3, th: Math.random() * 6.28, sp: (0.15 + Math.random() * 0.35) * (Math.random() < 0.5 ? -1 : 1), tilt: (Math.random() - 0.5) * 1.6 }))
    const orGeo = new THREE.BufferGeometry(); orGeo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(N * 3), 3))
    scene.add(new THREE.Points(orGeo, new THREE.PointsMaterial({ map: tex, color: blue, size: 0.09, transparent: true, opacity: 0.8, blending: THREE.AdditiveBlending, depthWrite: false })))

    let vx = 0, vy = 0, drag = false, lx = 0, ly = 0, vis = true, raf = 0
    const q = new THREE.Quaternion(), e = new THREE.Euler(), tmp = new THREE.Vector3(), cv = renderer.domElement
    const down = (ev: PointerEvent) => { drag = true; lx = ev.clientX; ly = ev.clientY; cv.setPointerCapture(ev.pointerId) }
    const move = (ev: PointerEvent) => { if (!drag) return; vy = (ev.clientX - lx) * 0.01; vx = (ev.clientY - ly) * 0.01; lx = ev.clientX; ly = ev.clientY }
    const up = () => { drag = false }
    cv.addEventListener('pointerdown', down); cv.addEventListener('pointermove', move); cv.addEventListener('pointerup', up); cv.addEventListener('pointercancel', up)
    const resize = () => { const w = el.clientWidth, h = el.clientHeight; renderer.setSize(w, h, false); cam.aspect = w / h; cam.updateProjectionMatrix() }
    const ro = new ResizeObserver(resize); ro.observe(el); resize()
    const io = new IntersectionObserver(([en]) => { vis = en.isIntersecting }); io.observe(el)
    const clock = new THREE.Clock()
    const tick = () => {
      raf = requestAnimationFrame(tick); if (!vis) return
      const t = clock.getElapsedTime(), ph = (t % 1.15) / 1.15
      const b = reduce ? 0 : bump(ph, 0.06, 0.05) + 0.65 * bump(ph, 0.3, 0.06)
      group.scale.setScalar(1 + 0.05 * b); glow.material.opacity = 0.42 + 0.3 * b; glow.scale.setScalar(7.5 + 1.2 * b); pl.intensity = 6 + 6 * b
      if (!drag) { vx *= 0.95; vy = vy * 0.95 + (reduce ? 0 : 0.004) * 0.05 }
      q.setFromEuler(e.set(vx, vy, 0)); group.quaternion.premultiply(q); if (drag) { vx = 0; vy = 0 }
      const pp = pkGeo.attributes.position as THREE.BufferAttribute
      pk.forEach((k, i) => { pt(k.u, (k.v0 + t * k.sp) % 1, tmp); pp.setXYZ(i, tmp.x * 1.03, tmp.y * 1.03, tmp.z * 1.03) }); pp.needsUpdate = true
      const op = orGeo.attributes.position as THREE.BufferAttribute
      orb.forEach((o, i) => { const a = o.th + t * o.sp; op.setXYZ(i, o.r * Math.cos(a), o.r * Math.sin(a) * Math.sin(o.tilt), o.r * Math.sin(a) * Math.cos(o.tilt)) }); op.needsUpdate = true
      renderer.render(scene, cam)
    }
    tick()
    return () => { cancelAnimationFrame(raf); ro.disconnect(); io.disconnect(); cv.removeEventListener('pointerdown', down); cv.removeEventListener('pointermove', move); cv.removeEventListener('pointerup', up); cv.removeEventListener('pointercancel', up); geo.dispose(); renderer.dispose(); cv.remove() }
  }, [])
  return <div ref={ref} className="heart" role="img" aria-label="Interactive 3D red heart. Drag to rotate."><svg className="fallback" viewBox="0 0 100 90" aria-hidden><path d="M50 85C10 55 0 35 0 22 0 8 12 0 25 0c10 0 20 6 25 16C55 6 65 0 75 0c13 0 25 8 25 22 0 13-10 33-50 63z" /></svg></div>
}
