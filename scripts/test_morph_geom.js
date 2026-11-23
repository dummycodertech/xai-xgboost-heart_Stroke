const THREE = require(require('path').resolve(__dirname, '../triage-web/node_modules/three'));

const V3 = THREE.Vector3;
const sstep = (t) => t * t * (3 - 2 * t);
const hash = (x, y, z) => {
  const h = Math.sin(x * 127.1 + y * 311.7 + z * 74.7) * 43758.5453;
  return h - Math.floor(h);
};
function noise(x, y, z) {
  const xi = Math.floor(x), yi = Math.floor(y), zi = Math.floor(z);
  const u = sstep(x - xi), v = sstep(y - yi), w = sstep(z - zi);
  const L = (a, b, t) => a + (b - a) * t;
  const c = (dx, dy, dz) => hash(xi + dx, yi + dy, zi + dz);
  return L(L(L(c(0, 0, 0), c(1, 0, 0), u), L(c(0, 1, 0), c(1, 1, 0), u), v), L(L(c(0, 0, 1), c(1, 0, 1), u), L(c(0, 1, 1), c(1, 1, 1), u), v), w);
}

const F = (x, y, z) => {
  const a = x * x + 2.25 * y * y + z * z - 1;
  return a * a * a - x * x * z * z * z - 0.1125 * y * y * z * z * z;
};

function heartPoint(d, k = 1) {
  let lo = 0, hi = 2.2;
  for (let i = 0; i < 26; i++) {
    const m = (lo + hi) / 2;
    F(m * d.x, m * d.z, m * d.y) < 0 ? (lo = m) : (hi = m);
  }
  const egg = 0.95 + 0.28 * d.y - 0.22 * d.y * d.y;
  const r = (lo * 0.55 + egg * 0.45) * (1 + 0.03 * noise(d.x * 5 + 9, d.y * 5, d.z * 5)) * k;
  const lv = 1 + 0.12 * sstep(Math.max(0, Math.min(1, d.x * 1.6 + 0.2))) * sstep(Math.max(0, Math.min(1, -d.y + 0.6)));
  // Rotate Z by 0.45 so apex points down-left
  const rx = d.x * r * lv, ry = d.y * r, rz = d.z * r * 1.3;
  const cosA = Math.cos(0.45), sinA = Math.sin(0.45);
  return new V3(rx * cosA - ry * sinA, rx * sinA + ry * cosA, rz);
}

function brainPoint(d) {
  const n = noise(d.x * 3.4 + 4, d.y * 3.4, d.z * 3.4) * 0.65 + noise(d.x * 8 + 1, d.y * 8, d.z * 8) * 0.35;
  const ridge = Math.pow(1 - Math.abs(2 * n - 1), 2.2);
  let r = 1 + 0.085 * ridge;
  const fissure = Math.exp(-Math.pow(d.x / 0.07, 2)) * sstep(Math.max(0, Math.min(1, d.y * 2 + 0.3)));
  r -= 0.16 * fissure;
  const x = d.x * 1.05 * r, z = d.z * 1.22 * r;
  let y = d.y * 0.8 * r;
  if (d.y < 0) y *= 0.82;
  return new V3(x * 1.15, y * 1.15, z * 1.15);
}

function bellPoint(d) {
  // Translucent bell dome (Video A)
  const angle = Math.atan2(d.z, d.x);
  const scallop = 1 + 0.065 * Math.cos(8 * angle);
  if (d.y >= -0.15) {
    // Upper bell umbrella
    const r = (1.25 - 0.35 * d.y * d.y) * scallop;
    return new V3(d.x * r, d.y * 0.75 + 0.22, d.z * r);
  } else {
    // Subumbrella and fluted oral arms
    const t = (-d.y - 0.15) / 0.85;
    const wave = 1 + 0.18 * Math.sin(t * 12 + angle * 3);
    const r = (0.95 - 0.65 * t) * wave;
    return new V3(d.x * r, -0.1 - t * 0.95, d.z * r);
  }
}

function buildMorphMesh() {
  const sphere = new THREE.SphereGeometry(1, 90, 70);
  const pCount = sphere.attributes.position.count;
  const posBell = new Float32Array(pCount * 3);
  const posHeart = new Float32Array(pCount * 3);
  const posBrain = new Float32Array(pCount * 3);
  const aRibbon = new Float32Array(pCount);
  const aTone = new Float32Array(pCount);

  const d = new V3();
  const pos = sphere.attributes.position;
  for (let i = 0; i < pCount; i++) {
    d.set(pos.getX(i), pos.getY(i), pos.getZ(i)).normalize();
    const pb = bellPoint(d);
    const ph = heartPoint(d);
    const pbr = brainPoint(d);

    posBell[i * 3] = pb.x; posBell[i * 3 + 1] = pb.y; posBell[i * 3 + 2] = pb.z;
    posHeart[i * 3] = ph.x; posHeart[i * 3 + 1] = ph.y; posHeart[i * 3 + 2] = ph.z;
    posBrain[i * 3] = pbr.x; posBrain[i * 3 + 1] = pbr.y; posBrain[i * 3 + 2] = pbr.z;

    aRibbon[i] = 0; // main body
    aTone[i] = (d.y < -0.1 ? 0.35 : 0.0);
  }

  // Add trailing ribbons
  const ribbons = [];
  const N_RIBBONS = 16;
  const SEGS = 24;
  for (let r = 0; r < N_RIBBONS; r++) {
    const angle = (r / N_RIBBONS) * Math.PI * 2;
    const ribGeo = new THREE.PlaneGeometry(0.04, 1.8, 1, SEGS);
    const rp = ribGeo.attributes.position;
    const rCount = rp.count;
    const rBell = new Float32Array(rCount * 3);
    const rHeart = new Float32Array(rCount * 3);
    const rBrain = new Float32Array(rCount * 3);
    const rRibbon = new Float32Array(rCount);
    const rTone = new Float32Array(rCount);

    for (let j = 0; j < rCount; j++) {
      const u = rp.getX(j); // [-0.02, 0.02]
      const v = (rp.getY(j) + 0.9) / 1.8; // [0, 1] from top to bottom
      rRibbon[j] = v;
      rTone[j] = 0.8;

      // Bell ribbon: streams downward with flare
      const rad = 1.1 + 0.35 * v;
      rBell[j * 3] = Math.cos(angle + u * 4) * rad;
      rBell[j * 3 + 1] = -0.15 - v * 1.85;
      rBell[j * 3 + 2] = Math.sin(angle + u * 4) * rad;

      // Heart ribbon: wraps around heart into coronary vessels / aorta arch
      const hRad = 0.85 * (1 - 0.4 * v);
      const hAngle = angle + v * 2.5;
      rHeart[j * 3] = Math.cos(hAngle) * hRad - 0.2 * v;
      rHeart[j * 3 + 1] = 0.6 - v * 1.2;
      rHeart[j * 3 + 2] = Math.sin(hAngle) * hRad;

      // Brain ribbon: gathers into brainstem / spinal cord
      const bRad = 0.22 * (1 - 0.3 * v);
      rBrain[j * 3] = Math.cos(angle) * bRad;
      rBrain[j * 3 + 1] = -0.6 - v * 1.1;
      rBrain[j * 3 + 2] = Math.sin(angle) * bRad - 0.3;
    }

    ribGeo.setAttribute('aPosBell', new THREE.BufferAttribute(rBell, 3));
    ribGeo.setAttribute('aPosHeart', new THREE.BufferAttribute(rHeart, 3));
    ribGeo.setAttribute('aPosBrain', new THREE.BufferAttribute(rBrain, 3));
    ribGeo.setAttribute('aRibbon', new THREE.BufferAttribute(rRibbon, 1));
    ribGeo.setAttribute('aTone', new THREE.BufferAttribute(rTone, 1));
    ribbons.push(ribGeo);
  }

  sphere.setAttribute('aPosBell', new THREE.BufferAttribute(posBell, 3));
  sphere.setAttribute('aPosHeart', new THREE.BufferAttribute(posHeart, 3));
  sphere.setAttribute('aPosBrain', new THREE.BufferAttribute(posBrain, 3));
  sphere.setAttribute('aRibbon', new THREE.BufferAttribute(aRibbon, 1));
  sphere.setAttribute('aTone', new THREE.BufferAttribute(aTone, 1));

  const { mergeGeometries } = require(require('path').resolve(__dirname, '../triage-web/node_modules/three/examples/jsm/utils/BufferGeometryUtils.js'));
  const finalMesh = mergeGeometries([sphere, ...ribbons], false);
  console.log('Final shared morph mesh created! Total vertices:', finalMesh.attributes.position.count);
  return finalMesh;
}

const g = buildMorphMesh();
console.log('Success! Attributes:', Object.keys(g.attributes));
