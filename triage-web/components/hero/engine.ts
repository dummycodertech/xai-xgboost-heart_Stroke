import * as THREE from 'three'
import { GL, vec3 } from '@/lib/palette'
import { buildBrain, buildHeart, buildSharedMorphMesh, loadGlb } from './geometry'
import { BG_FS, BODY_FS, BODY_VS, POST_FS, QUAD_VS, TRACK_FS } from './shaders'
import { findPeaks, type HudBus, type Mode } from './tracker'

const TRACK = 64
const GLITCH_MS = 75, ASCII_MS = 160, REVEAL_MS = 75 // tightened to ~310ms total (spec: 280-350ms)

const u3 = (hex: string) => ({ value: new THREE.Vector3(...vec3(hex)) })

// >>> Glyph ramp atlas: 10 cells, dark to bright, drawn with the page's monospace face
function makeAtlas(font: string) {
  const w = 40, h = 64, c = document.createElement('canvas')
  c.width = w * 10; c.height = h
  const x = c.getContext('2d')!
  x.fillStyle = '#000'; x.fillRect(0, 0, c.width, c.height)
  x.fillStyle = '#fff'; x.textAlign = 'center'; x.textBaseline = 'middle'; x.font = `700 ${h * 0.78}px ${font}`
  ' .:-=+*#%@'.split('').forEach((ch, i) => x.fillText(ch, i * w + w / 2, h / 2 + 2))
  const t = new THREE.CanvasTexture(c); t.minFilter = THREE.LinearFilter; t.generateMipmaps = false
  return t
}

export type QualityTier = 'high' | 'medium' | 'low'

export class Engine {
  private gl: THREE.WebGLRenderer
  private world = new THREE.Scene()
  private cam = new THREE.PerspectiveCamera(34, 1, 0.1, 40)
  private rt = new THREE.WebGLRenderTarget(2, 2)
  private small = new THREE.WebGLRenderTarget(TRACK, TRACK)
  private buf = new Uint8Array(TRACK * TRACK * 4)
  private quad = new THREE.PlaneGeometry(2, 2)
  private post = new THREE.Scene()
  private trackScene = new THREE.Scene()
  private postCam = new THREE.Camera()
  private bg: THREE.ShaderMaterial
  private postMat: THREE.ShaderMaterial
  private trackMat: THREE.ShaderMaterial
  private subjectMat: THREE.ShaderMaterial
  private group = new THREE.Group()
  private morphMesh: THREE.Mesh
  private loaded = new Set<Mode>()
  private meshes: Record<Mode, THREE.Mesh>
  private hasGlb: Record<Mode, boolean> = { heart: false, brain: false }
  private atlas: THREE.CanvasTexture
  private t = 0; private n = 0; private fpsAcc = 0; private fpsN = 0
  private mouse = new THREE.Vector2(0.5, 0.5)
  private aim = new THREE.Vector2(0.5, 0.5)
  private tx: { start: number; to: Mode; swapped: boolean } | null = null
  private shown: Mode = 'heart'
  private alive = true
  private reading = false
  private quality: QualityTier = 'high'
  private trackStride = 4

  constructor(gl: THREE.WebGLRenderer, private hud: HudBus, private reduce: boolean, font: string) {
    this.gl = gl
    const bgMat = (this.bg = new THREE.ShaderMaterial({
      vertexShader: QUAD_VS, fragmentShader: BG_FS, depthTest: false, depthWrite: false,
      uniforms: { uRes: { value: new THREE.Vector2(1, 1) }, uT: { value: 0 }, uM: { value: this.mouse }, uOff: { value: new THREE.Vector2(0.18, 0) }, uScale: { value: 1 },
        uSteps: { value: 44 },
        uBase: u3(GL.ink), uC0: u3(GL.burgundy), uC1: u3(GL.ochre), uC2: u3(GL.cream), uC3: u3(GL.olive) },
    }))
    const bgMesh = new THREE.Mesh(this.quad, bgMat); bgMesh.frustumCulled = false; bgMesh.renderOrder = -1
    this.world.add(this.group)

    const w = typeof window !== 'undefined' ? window.innerWidth : 1200
    const dpr = typeof window !== 'undefined' ? window.devicePixelRatio : 1
    if (w < 640) {
      this.setQuality('low')
    } else if (w < 1024 || dpr > 2.5) {
      this.setQuality('medium')
    } else {
      this.setQuality('high')
    }

    // Subject material: shared-topology Fresnel X-ray with morph targets and trailing ribbons
    this.subjectMat = new THREE.ShaderMaterial({
      vertexShader: BODY_VS, fragmentShader: BODY_FS, side: THREE.DoubleSide, transparent: true, depthWrite: true, depthTest: true,
      blending: THREE.NormalBlending,
      uniforms: {
        uT: { value: 0 }, uBeat: { value: 0 }, uGain: { value: 1 },
        uMorph: { value: 0 }, uModeFrom: { value: 1 }, uModeTo: { value: 1 },
        uA: u3(GL.burgundy), uB: u3('#801835'), uCream: u3(GL.cream),
      },
    })

    // Shared morph mesh with bell creature (video A) + trailing ribbons + heart + brain
    this.morphMesh = new THREE.Mesh(buildSharedMorphMesh(), this.subjectMat)
    this.group.add(this.morphMesh)

    // Procedural fallback meshes, replaceable by real anatomical GLB
    this.meshes = {
      heart: new THREE.Mesh(buildHeart(), this.subjectMat),
      brain: new THREE.Mesh(buildBrain(), this.subjectMat),
    }
    this.meshes.heart.visible = false
    this.meshes.brain.visible = false
    this.group.add(this.meshes.heart, this.meshes.brain)

    this.cam.position.z = 6.4
    this.loadModels(['heart', 'brain'])

    this.atlas = makeAtlas(font)
    this.postMat = new THREE.ShaderMaterial({
      vertexShader: QUAD_VS, fragmentShader: POST_FS, depthTest: false,
      uniforms: { tScene: { value: this.rt.texture }, tAtlas: { value: this.atlas }, uRes: { value: new THREE.Vector2(1, 1) }, uDpr: { value: 1 },
        uT: { value: 0 }, uGlitch: { value: 0 }, uAscii: { value: 0 }, uWipe: { value: 0 },
        uBurg: u3(GL.burgundy), uOlive: u3(GL.olive), uCream: u3(GL.cream), uInk: u3(GL.ink) },
    })
    const pq = new THREE.Mesh(this.quad, this.postMat); pq.frustumCulled = false; this.post.add(pq)
    this.trackMat = new THREE.ShaderMaterial({ vertexShader: QUAD_VS, fragmentShader: TRACK_FS, depthTest: false, uniforms: { tScene: { value: this.rt.texture } } })
    const tq = new THREE.Mesh(this.quad, this.trackMat); tq.frustumCulled = false; this.trackScene.add(tq)
  }

  // Real anatomy GLBs: heart.glb (HuBMAP CC-BY 4.0) and brain.glb (HuBMAP/Allen CC-BY 4.0)
  private async loadModels(which: Mode[] = ['heart']) {
    const urls: Record<Mode, string | undefined> = {
      heart: process.env.NEXT_PUBLIC_HEART_GLB || '/models/heart.glb',
      brain: process.env.NEXT_PUBLIC_BRAIN_GLB || '/models/brain.glb',
    }
    for (const m of which) {
      if (this.loaded.has(m)) continue
      this.loaded.add(m)
      const u = urls[m]; if (!u) continue
      const g = await loadGlb(u)
      if (g && this.alive) {
        this.meshes[m].geometry.dispose()
        this.meshes[m].geometry = g
        this.hasGlb[m] = true
        if (this.shown === m && !this.tx) {
          this.meshes[m].visible = true
          this.morphMesh.visible = false
        }
      }
    }
  }

  setQuality(q: QualityTier) {
    this.quality = q
    if (this.bg?.uniforms?.uSteps) {
      this.bg.uniforms.uSteps.value = q === 'low' ? 24 : q === 'medium' ? 32 : 44
    }
    this.trackStride = q === 'low' ? 8 : q === 'medium' ? 6 : 4
  }

  resize(w: number, h: number, dpr: number) {
    const maxDpr = this.quality === 'low' ? 1.0 : this.quality === 'medium' ? 1.25 : 1.5
    const effDpr = Math.min(dpr, maxDpr)
    const pw = Math.max(2, Math.floor(w * effDpr)), ph = Math.max(2, Math.floor(h * effDpr))
    this.rt.setSize(pw, ph)
    this.bg.uniforms.uRes.value.set(pw, ph); this.postMat.uniforms.uRes.value.set(pw, ph); this.postMat.uniforms.uDpr.value = effDpr
    this.cam.aspect = w / h
    this.cam.position.z = w / h < 1 ? 6.4 / Math.min(1, (w / h) * 1.15) : 6.4
    this.bg.uniforms.uOff.value.set(w / h < 1 ? 0 : 0.18, 0)
    this.cam.updateProjectionMatrix()
  }

  setMode(m: Mode) {
    if (m === 'brain') this.loadModels(['brain'])
    if (m === this.shown && !this.tx) return
    const fromMode = this.shown === 'heart' ? 1 : 2
    const toMode = m === 'heart' ? 1 : 2
    this.subjectMat.uniforms.uModeFrom.value = fromMode
    this.subjectMat.uniforms.uModeTo.value = toMode
    this.subjectMat.uniforms.uMorph.value = 0
    this.tx = { start: performance.now(), to: m, swapped: false }
    // Morph mesh is active during transition
    this.morphMesh.visible = true
    this.meshes.heart.visible = false
    this.meshes.brain.visible = false
  }

  private swap(m: Mode) {
    this.shown = m
    this.hud.mode = m
    this.hud.epoch++
    const isHeart = m === 'heart'
    this.subjectMat.uniforms.uA.value.copy(new THREE.Vector3(...vec3(isHeart ? GL.burgundy : GL.olive)))
    this.subjectMat.uniforms.uB.value.copy(new THREE.Vector3(...vec3(isHeart ? '#801835' : '#556b4e')))
  }

  private stepTx(now: number) {
    const fx = this.postMat.uniforms
    let glitch = 0, ascii = 0, wipe = 0
    if (this.tx) {
      const e = now - this.tx.start
      if (this.reduce) {
        // prefers-reduced-motion: clean, smooth opacity/morph crossfade with zero glitch and zero ASCII
        const DURATION = 300
        const progress = Math.min(1, e / DURATION)
        const smoothK = progress * progress * (3 - 2 * progress)
        this.subjectMat.uniforms.uMorph.value = smoothK
        if (!this.tx.swapped && progress > 0.5) {
          this.swap(this.tx.to)
          this.tx.swapped = true
        }
        if (progress >= 1) {
          if (!this.tx.swapped) this.swap(this.tx.to)
          this.subjectMat.uniforms.uMorph.value = 1
          this.subjectMat.uniforms.uModeFrom.value = this.tx.to === 'heart' ? 1 : 2
          this.tx = null
          if (this.hasGlb[this.shown]) {
            this.meshes[this.shown].visible = true
            this.morphMesh.visible = false
          } else {
            this.morphMesh.visible = true
          }
        }
      } else {
        // Standard motion: slice glitch (~75ms) -> ASCII flash (~160ms) -> reveal (~75ms)
        const a0 = GLITCH_MS, a1 = a0 + ASCII_MS, a2 = a1 + REVEAL_MS
        if (e < a0) {
          glitch = e / a0
          this.subjectMat.uniforms.uMorph.value = 0
        } else if (e < a1) {
          glitch = 1
          ascii = 1
          const progress = (e - a0) / ASCII_MS
          // Shared-topology vertex morphing occurs underneath the ASCII flash
          this.subjectMat.uniforms.uMorph.value = progress
          if (!this.tx.swapped && progress > 0.45) {
            this.swap(this.tx.to)
            this.tx.swapped = true
          }
        } else if (e < a2) {
          const k = (e - a1) / REVEAL_MS
          glitch = 1 - k
          ascii = 1
          wipe = k
          this.subjectMat.uniforms.uMorph.value = 1
        } else {
          if (!this.tx.swapped) this.swap(this.tx.to)
          this.subjectMat.uniforms.uMorph.value = 1
          this.subjectMat.uniforms.uModeFrom.value = this.tx.to === 'heart' ? 1 : 2
          this.tx = null
          if (this.hasGlb[this.shown]) {
            this.meshes[this.shown].visible = true
            this.morphMesh.visible = false
          } else {
            this.morphMesh.visible = true
          }
        }
      }
    }
    fx.uGlitch.value = glitch
    fx.uAscii.value = ascii
    fx.uWipe.value = wipe
    this.hud.fx.glitch = glitch
    this.hud.fx.ascii = ascii * (1 - wipe)
  }

  frame(dt: number) {
    const now = performance.now()
    if (!this.reduce) this.t += dt
    this.stepTx(now)
    this.mouse.lerp(this.aim.set(this.hud.pointer.x, this.hud.pointer.y), 1 - Math.exp(-dt * 3))
    const ph = (this.t * 1.1) % 1 // lub-dub envelope, about 66 bpm
    const beat = this.reduce ? 0 : Math.exp(-Math.pow((ph - 0.08) / 0.06, 2)) + 0.55 * Math.exp(-Math.pow((ph - 0.28) / 0.07, 2))
    const mx = this.mouse.x - 0.5, my = this.mouse.y - 0.5
    this.group.rotation.y = Math.sin(this.t * 0.35) * 0.45 + mx * 0.5
    this.group.rotation.x = -my * 0.25
    this.group.position.x = this.cam.aspect < 1 ? 0 : -0.15

    this.subjectMat.uniforms.uT.value = this.t
    this.subjectMat.uniforms.uBeat.value = this.shown === 'heart' ? beat : beat * 0.3
    this.bg.uniforms.uT.value = this.t
    this.postMat.uniforms.uT.value = this.t

    this.gl.setRenderTarget(this.rt); this.gl.setClearColor(0x000000, 0); this.gl.clear()
    this.gl.render(this.world, this.cam)
    if (this.n++ % this.trackStride === 0) this.track()
    this.gl.setRenderTarget(null); this.gl.render(this.post, this.postCam)

    this.hud.frame++
    this.fpsAcc += dt; this.fpsN++
    if (this.fpsAcc > 0.5) {
      this.hud.fps = Math.round(this.fpsN / this.fpsAcc)
      if (this.hud.fps < 32 && this.quality !== 'low') {
        this.setQuality(this.quality === 'high' ? 'medium' : 'low')
      }
      this.fpsAcc = 0; this.fpsN = 0
    }
  }

  private track() {
    if (this.reading) return
    this.gl.setRenderTarget(this.small); this.gl.render(this.trackScene, this.postCam)
    this.reading = true
    this.gl.readRenderTargetPixelsAsync(this.small, 0, 0, TRACK, TRACK, this.buf)
      .then(() => { if (this.alive && this.hud.fx.ascii < 0.5) this.hud.peaks = findPeaks(this.buf, TRACK, TRACK) })
      .catch(() => { /* context lost: keep last peaks */ })
      .finally(() => { this.reading = false })
  }

  dispose() {
    this.alive = false
    this.rt.dispose(); this.small.dispose(); this.quad.dispose(); this.atlas.dispose()
    this.morphMesh.geometry.dispose()
    for (const m of Object.values(this.meshes)) { m.geometry.dispose() }
    this.subjectMat.dispose(); this.bg.dispose(); this.postMat.dispose(); this.trackMat.dispose()
  }
}
