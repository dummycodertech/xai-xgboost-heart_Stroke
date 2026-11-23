import * as THREE from 'three'
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'

// >>> Helpers
const V3 = THREE.Vector3
const sstep = (t: number) => t * t * (3 - 2 * t)
const hash = (x: number, y: number, z: number) => {
  const h = Math.sin(x * 127.1 + y * 311.7 + z * 74.7) * 43758.5453
  return h - Math.floor(h)
}
function noise(x: number, y: number, z: number) {
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z)
  const u = sstep(x - xi), v = sstep(y - yi), w = sstep(z - zi)
  const L = (a: number, b: number, t: number) => a + (b - a) * t
  const c = (dx: number, dy: number, dz: number) => hash(xi + dx, yi + dy, zi + dz)
  return L(L(L(c(0, 0, 0), c(1, 0, 0), u), L(c(0, 1, 0), c(1, 1, 0), u), v), L(L(c(0, 0, 1), c(1, 0, 1), u), L(c(0, 1, 1), c(1, 1, 1), u), v), w)
}
function finish(g: THREE.BufferGeometry, tone: number) {
  g.deleteAttribute('uv')
  g.setAttribute('aTone', new THREE.BufferAttribute(new Float32Array(g.attributes.position.count).fill(tone), 1))
  return g
}
const tube = (pts: number[][], r: number, tone: number) =>
  finish(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts.map(([x, y, z]) => new V3(x, y, z))), 56, r, 14, false), tone)

// >>> Heart implicit surface (apex down-left)
const F = (x: number, y: number, z: number) => {
  const a = x * x + 2.25 * y * y + z * z - 1
  return a * a * a - x * x * z * z * z - 0.1125 * y * y * z * z * z
}
function heartPoint(d: THREE.Vector3, k = 1): THREE.Vector3 {
  let lo = 0, hi = 2.2
  for (let i = 0; i < 26; i++) {
    const m = (lo + hi) / 2
    F(m * d.x, m * d.z, m * d.y) < 0 ? (lo = m) : (hi = m)
  }
  const egg = 0.95 + 0.28 * d.y - 0.22 * d.y * d.y
  const r = (lo * 0.55 + egg * 0.45) * (1 + 0.03 * noise(d.x * 5 + 9, d.y * 5, d.z * 5)) * k
  const lv = 1 + 0.12 * sstep(Math.max(0, Math.min(1, d.x * 1.6 + 0.2))) * sstep(Math.max(0, Math.min(1, -d.y + 0.6)))
  const rx = d.x * r * lv, ry = d.y * r, rz = d.z * r * 1.3
  const cosA = Math.cos(0.45), sinA = Math.sin(0.45)
  return new V3(rx * cosA - ry * sinA, rx * sinA + ry * cosA, rz)
}

// >>> Brain implicit/ridged surface
function brainPoint(d: THREE.Vector3): THREE.Vector3 {
  const n = noise(d.x * 3.4 + 4, d.y * 3.4, d.z * 3.4) * 0.65 + noise(d.x * 8 + 1, d.y * 8, d.z * 8) * 0.35
  const ridge = Math.pow(1 - Math.abs(2 * n - 1), 2.2)
  let r = 1 + 0.085 * ridge
  const fissure = Math.exp(-Math.pow(d.x / 0.07, 2)) * sstep(Math.max(0, Math.min(1, d.y * 2 + 0.3)))
  r -= 0.16 * fissure
  const x = d.x * 1.05 * r, z = d.z * 1.22 * r
  let y = d.y * 0.8 * r
  if (d.y < 0) y *= 0.82
  return new V3(x * 1.15, y * 1.15, z * 1.15)
}

// >>> Translucent bioluminescent bell creature (Video A)
function bellPoint(d: THREE.Vector3): THREE.Vector3 {
  const angle = Math.atan2(d.z, d.x)
  const scallop = 1 + 0.065 * Math.cos(8 * angle)
  if (d.y >= -0.15) {
    const r = (1.25 - 0.35 * d.y * d.y) * scallop
    return new V3(d.x * r, d.y * 0.75 + 0.22, d.z * r)
  } else {
    const t = (-d.y - 0.15) / 0.85
    const wave = 1 + 0.18 * Math.sin(t * 12 + angle * 3)
    const r = (0.95 - 0.65 * t) * wave
    return new V3(d.x * r, -0.1 - t * 0.95, d.z * r)
  }
}

// >>> Shared-topology morph mesh: bioluminescent bell + trailing ribbons + heart + brain (Video A & B)
export function buildSharedMorphMesh(): THREE.BufferGeometry {
  const sphere = new THREE.SphereGeometry(1, 90, 70)
  const pCount = sphere.attributes.position.count
  const posBell = new Float32Array(pCount * 3)
  const posHeart = new Float32Array(pCount * 3)
  const posBrain = new Float32Array(pCount * 3)
  const aRibbon = new Float32Array(pCount)
  const aTone = new Float32Array(pCount)

  const d = new V3()
  const pos = sphere.attributes.position
  for (let i = 0; i < pCount; i++) {
    d.set(pos.getX(i), pos.getY(i), pos.getZ(i)).normalize()
    const pb = bellPoint(d)
    const ph = heartPoint(d)
    const pbr = brainPoint(d)

    posBell[i * 3] = pb.x; posBell[i * 3 + 1] = pb.y; posBell[i * 3 + 2] = pb.z
    posHeart[i * 3] = ph.x; posHeart[i * 3 + 1] = ph.y; posHeart[i * 3 + 2] = ph.z
    posBrain[i * 3] = pbr.x; posBrain[i * 3 + 1] = pbr.y; posBrain[i * 3 + 2] = pbr.z

    aRibbon[i] = 0 // main body
    aTone[i] = d.y < -0.1 ? 0.35 : 0.0
  }

  // 16 trailing ribbons driven by noise (Video A)
  const ribbons: THREE.BufferGeometry[] = []
  const N_RIBBONS = 16
  const SEGS = 24
  for (let r = 0; r < N_RIBBONS; r++) {
    const angle = (r / N_RIBBONS) * Math.PI * 2
    const ribGeo = new THREE.PlaneGeometry(0.04, 1.8, 1, SEGS)
    const rp = ribGeo.attributes.position
    const rCount = rp.count
    const rBell = new Float32Array(rCount * 3)
    const rHeart = new Float32Array(rCount * 3)
    const rBrain = new Float32Array(rCount * 3)
    const rRibbon = new Float32Array(rCount)
    const rTone = new Float32Array(rCount)

    for (let j = 0; j < rCount; j++) {
      const u = rp.getX(j)
      const v = (rp.getY(j) + 0.9) / 1.8 // [0, 1] from top to bottom
      rRibbon[j] = v
      rTone[j] = 0.8

      // Bell state: streams downward with organic flare
      const rad = 1.1 + 0.35 * v
      rBell[j * 3] = Math.cos(angle + u * 4) * rad
      rBell[j * 3 + 1] = -0.15 - v * 1.85
      rBell[j * 3 + 2] = Math.sin(angle + u * 4) * rad

      // Heart state: wraps around heart into coronary arteries / aortic arch
      const hRad = 0.85 * (1 - 0.4 * v)
      const hAngle = angle + v * 2.5
      rHeart[j * 3] = Math.cos(hAngle) * hRad - 0.2 * v
      rHeart[j * 3 + 1] = 0.6 - v * 1.2
      rHeart[j * 3 + 2] = Math.sin(hAngle) * hRad

      // Brain state: gathers into brainstem / descending spinal column
      const bRad = 0.22 * (1 - 0.3 * v)
      rBrain[j * 3] = Math.cos(angle) * bRad
      rBrain[j * 3 + 1] = -0.6 - v * 1.1
      rBrain[j * 3 + 2] = Math.sin(angle) * bRad - 0.3
    }

    ribGeo.setAttribute('aPosBell', new THREE.BufferAttribute(rBell, 3))
    ribGeo.setAttribute('aPosHeart', new THREE.BufferAttribute(rHeart, 3))
    ribGeo.setAttribute('aPosBrain', new THREE.BufferAttribute(rBrain, 3))
    ribGeo.setAttribute('aRibbon', new THREE.BufferAttribute(rRibbon, 1))
    ribGeo.setAttribute('aTone', new THREE.BufferAttribute(rTone, 1))
    ribbons.push(ribGeo)
  }

  sphere.setAttribute('aPosBell', new THREE.BufferAttribute(posBell, 3))
  sphere.setAttribute('aPosHeart', new THREE.BufferAttribute(posHeart, 3))
  sphere.setAttribute('aPosBrain', new THREE.BufferAttribute(posBrain, 3))
  sphere.setAttribute('aRibbon', new THREE.BufferAttribute(aRibbon, 1))
  sphere.setAttribute('aTone', new THREE.BufferAttribute(aTone, 1))

  const finalMesh = mergeGeometries([sphere, ...ribbons], false)!
  return normalise(finalMesh, 1.35)
}

function atrium(x: number, y: number, z: number, sx: number, sy: number, sz: number) {
  const g = new THREE.SphereGeometry(1, 40, 30); const p = g.attributes.position as THREE.BufferAttribute
  for (let i = 0; i < p.count; i++) { const a = p.getX(i), b = p.getY(i), c = p.getZ(i); const f = 1 + 0.05 * noise(a * 4 + 3, b * 4, c * 4); p.setXYZ(i, x + a * sx * f, y + b * sy * f, z + c * sz * f) }
  g.computeVertexNormals(); return finish(g, 0.35)
}

export function buildHeart(): THREE.BufferGeometry {
  const body = new THREE.SphereGeometry(1, 150, 110)
  const p = body.attributes.position as THREE.BufferAttribute
  const d = new V3()
  for (let i = 0; i < p.count; i++) { d.fromBufferAttribute(p, i).normalize(); const q = heartPoint(d); p.setXYZ(i, q.x, q.y, q.z) }
  body.computeVertexNormals()
  const sweep = (a: number[][]) => a.map(([x, y, z]) => heartPoint(new V3(x, y, z).normalize(), 1.015).toArray())
  const lad = sweep([[0.1, 0.8, 0.7], [0.25, 0.4, 0.85], [0.35, 0.0, 0.85], [0.4, -0.45, 0.8], [0.3, -0.85, 0.55]])
  const rca = sweep([[-0.2, 0.7, 0.75], [-0.6, 0.45, 0.7], [-0.85, 0.0, 0.5], [-0.75, -0.5, 0.55], [-0.4, -0.8, 0.6]])
  const parts = [
    finish(body, 0),
    tube([[0.1, 0.7, -0.05], [0.15, 1.1, -0.05], [0.35, 1.45, -0.15], [0.75, 1.45, -0.3], [0.98, 1.1, -0.4], [1.02, 0.6, -0.45]], 0.12, 1),
    tube([[-0.12, 0.72, 0.3], [-0.18, 1.08, 0.35], [-0.45, 1.34, 0.3], [-0.8, 1.38, 0.15]], 0.1, 1),
    tube([[-0.68, 0.5, -0.2], [-0.7, 1.1, -0.25], [-0.7, 1.5, -0.25]], 0.085, 1),
    tube([[-0.5, -0.55, -0.15], [-0.55, -0.95, -0.2], [-0.55, -1.3, -0.2]], 0.09, 1),
    atrium(-0.62, 0.42, -0.12, 0.36, 0.3, 0.32), atrium(0.5, 0.78, 0.2, 0.22, 0.2, 0.2),
    tube(lad, 0.022, 1), tube(rca, 0.022, 1),
  ]
  const g = mergeGeometries(parts, false)!
  g.rotateZ(0.45)
  return normalise(g, 1.35)
}

export function buildBrain(): THREE.BufferGeometry {
  const cortex = new THREE.SphereGeometry(1, 150, 110)
  const p = cortex.attributes.position as THREE.BufferAttribute
  const d = new V3()
  for (let i = 0; i < p.count; i++) {
    d.fromBufferAttribute(p, i).normalize()
    const pb = brainPoint(d)
    p.setXYZ(i, pb.x, pb.y, pb.z)
  }
  cortex.computeVertexNormals()
  const cb = new THREE.SphereGeometry(1, 70, 50)
  const q = cb.attributes.position as THREE.BufferAttribute
  for (let i = 0; i < q.count; i++) {
    d.fromBufferAttribute(q, i).normalize()
    const f = 1 + 0.06 * Math.sin(d.y * 26 + noise(d.x * 3, d.y * 3, d.z * 3) * 3)
    q.setXYZ(i, d.x * 0.6 * f, d.y * 0.36 * f - 0.64, d.z * 0.5 * f - 0.86)
  }
  cb.computeVertexNormals()
  const g = mergeGeometries([finish(cortex, 0), finish(cb, 0.55), tube([[0, -0.5, -0.4], [0, -0.95, -0.5], [0, -1.35, -0.55]], 0.13, 1)], false)!
  return normalise(g, 1.3)
}

function normalise(g: THREE.BufferGeometry, radius: number) {
  g.computeBoundingSphere()
  const s = g.boundingSphere!
  g.translate(-s.center.x, -s.center.y, -s.center.z)
  g.scale(radius / s.radius, radius / s.radius, radius / s.radius)
  return g
}

// >>> Optional real anatomy: any .glb, merged into one geometry and normalised. Resolves null on failure.
export function loadGlb(url: string): Promise<THREE.BufferGeometry | null> {
  return new Promise((resolve) => {
    new GLTFLoader().load(url, (gltf) => {
      const geos: THREE.BufferGeometry[] = []
      gltf.scene.updateMatrixWorld(true)
      gltf.scene.traverse((o) => {
        const m = o as THREE.Mesh
        if (!m.isMesh) return
        const g = m.geometry.clone().applyMatrix4(m.matrixWorld)
        for (const k of Object.keys(g.attributes)) if (k !== 'position' && k !== 'normal') g.deleteAttribute(k)
        if (!g.attributes.normal) g.computeVertexNormals()
        geos.push(finish(g.index ? g.toNonIndexed() : g, 0))
      })
      const merged = geos.length ? mergeGeometries(geos, false) : null
      resolve(merged ? normalise(merged, 1.35) : null)
    }, undefined, () => resolve(null))
  })
}
