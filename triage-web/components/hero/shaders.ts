// >>> Full-screen quad vertex shader (clip space, no camera)
export const QUAD_VS = `varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`

// >>> Background: raymarched iridescent glass blob (burgundy / ochre / cream / olive thin-film ramp). Alpha stays 0 so the tracker ignores it.
export const BG_FS = `precision highp float;
uniform vec2 uRes; uniform float uT; uniform vec2 uM; uniform vec2 uOff; uniform float uScale;
uniform int uSteps;
uniform vec3 uBase, uC0, uC1, uC2, uC3;

mat2 rot(float a){ float c = cos(a), s = sin(a); return mat2(c, -s, s, c); }

// Distance estimator with interactive cursor magnetic morphing (Video B)
float map(vec3 p){
  float t = uT * 0.24;
  vec3 mTarget = vec3((uM.x - 0.5) * 2.8 + uOff.x * 1.2, (uM.y - 0.5) * 1.8 + uOff.y, 0.2);
  float mDist = length(p - mTarget);
  float pull = exp(-mDist * 1.5) * 0.45;
  p -= normalize(p - mTarget) * pull;

  p.xz *= rot(p.y * 0.5 + t);
  p.yz *= rot(sin(t * 1.2) * 0.45);

  float d = length(p * vec3(1.0, 0.82, 1.08)) - 1.05;
  d += 0.18 * sin(2.5 * p.x + t * 1.8) * sin(2.2 * p.y - t * 1.4) * sin(2.6 * p.z + t * 0.9);
  d += 0.06 * sin(6.5 * p.x + t * 2.8) * sin(6.5 * p.y) * sin(6.5 * p.z);
  d += 0.025 * sin(12.0 * p.x - t * 4.0) * sin(12.0 * p.z);
  return d * 0.58;
}

// 4-color thin-film chromatic dispersion ramp (strictly burgundy, ochre, cream, olive)
vec3 ramp(float v){
  v = fract(v) * 4.0;
  float k = fract(v); k = k * k * (3.0 - 2.0 * k);
  if (v < 1.0) return mix(uC0, uC1, k);
  if (v < 2.0) return mix(uC1, uC2, k);
  if (v < 3.0) return mix(uC2, uC3, k);
  return mix(uC3, uC0, k);
}

void main(){
  vec2 uv = (gl_FragCoord.xy - 0.5 * uRes) / uRes.y;
  vec3 ro = vec3(-(uM.x - 0.5) * 1.1 - uOff.x, -(uM.y - 0.5) * 0.8 - uOff.y, 5.2);
  vec3 rd = normalize(vec3(uv, -1.75));

  float t = 0.0; bool hit = false;
  int maxSteps = uSteps > 0 ? uSteps : 44;
  for (int i = 0; i < 48; i++){
    if (i >= maxSteps) break;
    float d = map(ro + rd * t);
    if (d < 0.0018){ hit = true; break; }
    t += d;
    if (t > 9.5) break;
  }

  vec3 col = uBase;
  if (hit){
    vec3 p = ro + rd * t;
    vec2 e = vec2(0.0035, 0.0);
    vec3 n = normalize(vec3(
      map(p + e.xyy) - map(p - e.xyy),
      map(p + e.yxy) - map(p - e.yxy),
      map(p + e.yyx) - map(p - e.yyx)
    ));

    float ndv = clamp(dot(n, -rd), 0.0, 1.0);
    float fr = pow(1.0 - ndv, 2.2);

    // Refraction into the glass interior (Snell's law ray)
    vec3 refr = refract(rd, n, 1.0 / 1.35);
    if (length(refr) < 0.01) refr = rd;

    // Internal caustics / streaks raymarching (Video B)
    float th = 0.0; float streaks = 0.0;
    vec3 q = p + refr * 0.03;
    for (int i = 0; i < 12; i++){
      float dd = -map(q);
      float dense = max(dd, 0.0);
      th += dense;
      streaks += max(0.0, sin(q.x * 14.0 + q.y * 7.0 + uT * 1.5) * cos(q.z * 12.0 - uT)) * dense;
      q += refr * max(dd, 0.055);
    }

    // Thin-film interference fringe rings
    float fringe = sin(fr * 14.0 + th * 2.8 + uT * 0.25) * 0.5 + 0.5;
    float rampPos = fr * 1.8 + th * 0.6 + streaks * 0.4 + dot(n, normalize(vec3(0.35, 0.8, 0.5))) * 0.4 + uT * 0.05;
    vec3 c = ramp(rampPos);

    // Brighter, vivid glass lighting with specular and chromatic rim
    vec3 glow = c * (0.05 + 1.65 * pow(fr, 1.25)) + c * th * 0.06 + c * streaks * 0.12;
    glow += mix(uC0, uC2, fringe) * pow(fr, 4.2) * 1.1;
    vec3 refl = reflect(rd, n);
    float spec = pow(max(dot(refl, normalize(vec3(-0.25, 0.85, 0.45))), 0.0), 32.0);
    glow += uC2 * spec * 0.95;

    col += glow * uScale;
  }

  col += uC0 * 0.035 * exp(-length(uv) * 2.0);
  gl_FragColor = vec4(col, 0.0);
}`

// >>> Subject: Shared-topology Fresnel X-ray vertex shader with noise-driven live trailing ribbons (Video A)
export const BODY_VS = `
attribute vec3 aPosBell;
attribute vec3 aPosHeart;
attribute vec3 aPosBrain;
attribute float aRibbon;
attribute float aTone;

uniform float uMorph;
uniform float uModeFrom;
uniform float uModeTo;
uniform float uBeat;
uniform float uT;

varying vec3 vN, vV, vP;
varying float vTone, vRibbon;

vec3 getBasePos(float mode){
  if (mode < 0.5) return aPosBell;
  if (mode < 1.5) return aPosHeart;
  return aPosBrain;
}

void main(){
  float k = smoothstep(0.0, 1.0, uMorph);
  vec3 pFrom = getBasePos(uModeFrom);
  vec3 pTo = getBasePos(uModeTo);
  vec3 p = mix(pFrom, pTo, k);

  // If morph attributes not provided (e.g. standalone GLB mesh), fallback to standard position
  if (length(p) < 0.0001) p = position;

  // Live noise-driven trailing ribbons (Video A)
  if (aRibbon > 0.01){
    float waveX = sin(uT * 2.8 - aRibbon * 6.5 + p.y * 2.0) * cos(uT * 1.7 + p.z * 3.0);
    float waveZ = cos(uT * 2.4 - aRibbon * 5.8 + p.x * 2.5);
    float waveY = sin(uT * 3.2 + aRibbon * 4.0) * 0.06;
    float amp = mix(0.18, 0.05, step(0.5, mix(uModeFrom, uModeTo, k)));
    p.x += waveX * amp * aRibbon;
    p.z += waveZ * amp * aRibbon;
    p.y += waveY * aRibbon;
  } else {
    // Cardiac / respiratory living pulse
    float pulse = (1.0 + 0.05 * uBeat * (1.0 - aTone * 0.6));
    p *= pulse;
    // Gentle biological undulation
    p.x += sin(uT * 1.2 + p.y * 2.5) * 0.02;
    p.z += cos(uT * 1.1 + p.x * 2.5) * 0.02;
  }

  vec4 mv = modelViewMatrix * vec4(p, 1.0);
  vV = mv.xyz;
  vN = normalize(normalMatrix * normal);
  vP = p;
  vTone = aTone;
  vRibbon = aRibbon;
  gl_Position = projectionMatrix * mv;
}`

// >>> Subject: Bioluminescent anatomical shader with sculptural depth and crisp sulci / gyri contrast
export const BODY_FS = `precision highp float;
uniform vec3 uA, uB, uCream; uniform float uT, uBeat, uGain;
varying vec3 vN, vV, vP; varying float vTone, vRibbon;
void main(){
  vec3 n = normalize(vN); if (!gl_FrontFacing) n = -n;
  vec3 viewDir = normalize(-vV);
  float ndv = clamp(abs(dot(n, viewDir)), 0.0, 1.0);
  float fr = pow(1.0 - ndv, 2.0);

  // Directional lighting to reveal sculptural folds, crevices & anatomy
  vec3 l1 = normalize(vec3(0.45, 0.75, 0.65));
  vec3 l2 = normalize(vec3(-0.6, -0.3, 0.4));
  float diff1 = max(dot(n, l1), 0.0);
  float diff2 = max(dot(n, l2), 0.0) * 0.35;
  float ambient = 0.28;
  float lighting = ambient + 0.62 * diff1 + 0.22 * diff2;

  // Specular highlight catching gyri and blood vessel crests
  vec3 h1 = normalize(l1 + viewDir);
  float spec = pow(max(dot(n, h1), 0.0), 24.0) * 0.45;

  // Subtle biological band undulation
  float bands = smoothstep(0.88, 1.0, sin(vP.y * 32.0 + vP.x * 8.0 - uT * 1.2) * 0.5 + 0.5);
  float ribbonGlow = vRibbon * (0.6 + 0.4 * sin(uT * 3.0 - vRibbon * 8.0));

  vec3 baseTone = mix(uA, uB, vTone);
  vec3 core = baseTone * lighting + baseTone * bands * 0.15;
  vec3 rim = mix(baseTone, uCream, 0.45) * fr * 0.95 + uCream * pow(fr, 3.8) * 0.45;

  vec3 col = (core + rim + uCream * spec + ribbonGlow) * uGain;
  col = min(col, vec3(0.92));

  // Solid opacity with subtle rim transparency for anatomical crispness
  float a = clamp(0.82 + 0.18 * fr, 0.0, 0.98);
  gl_FragColor = vec4(col, a);
}`

// >>> Tracker downsample: 16-tap average of the scene's alpha (subject luminance) into a 64x64 target
export const TRACK_FS = `precision highp float; uniform sampler2D tScene; varying vec2 vUv;
void main(){ float a = 0.; for (int j = 0; j < 4; j++) for (int i = 0; i < 4; i++) a += texture2D(tScene, vUv + (vec2(float(i), float(j)) / 4. - .375) / 64.).a; a /= 16.; gl_FragColor = vec4(a, a, a, 1.); }`

// >>> Composite: slice glitch -> organ silhouette ASCII -> ordered dither + dot grid + vignette
export const POST_FS = `precision highp float;
uniform sampler2D tScene; uniform sampler2D tAtlas; uniform vec2 uRes; uniform float uDpr, uT, uGlitch, uAscii, uWipe;
uniform vec3 uBurg, uOlive, uCream, uInk; varying vec2 vUv;
float h11(float x){ return fract(sin(x * 127.1) * 43758.5453); }
float bayer(vec2 p){
  const float B[16] = float[16](0.,8.,2.,10.,12.,4.,14.,6.,3.,11.,1.,9.,15.,7.,13.,5.);
  return (B[int(mod(p.y, 4.)) * 4 + int(mod(p.x, 4.))] + .5) / 16.;
}
void main(){
  float organA = texture2D(tScene, vUv).a;
  vec2 uv = vUv; float tick = floor(uT * 30.); float on = 0.; float r2 = 0.;
  // Confine slice glitch strictly to the organ's silhouette
  if (uGlitch > 0. && organA > 0.03){
    float row = floor(uv.y * 26.); float r = h11(row + tick); r2 = h11(row * 3.1 + tick);
    on = step(1. - uGlitch * 0.75, r) * smoothstep(0.03, 0.2, organA);
    uv.x += on * (r2 - 0.5) * 0.18 * uGlitch;
  }
  vec3 col = texture2D(tScene, uv).rgb;
  col = mix(col, col * 0.55 + mix(uBurg, uOlive, step(0.5, r2)) * 0.32 * uGlitch, on);

  // Confine ASCII flash strictly to organ silhouette (no empty background flooding)
  vec2 cell = vec2(7., 12.) * uDpr; vec2 cid = floor(gl_FragCoord.xy / cell);
  vec2 cuv = (cid + 0.5) * cell / uRes;
  float cellA = texture2D(tScene, cuv).a;
  float inAscii = uAscii * step(uWipe, 1. - vUv.y) * smoothstep(0.12, 0.35, min(organA, cellA));
  if (inAscii > 0.){
    vec3 c = (texture2D(tScene, cuv).rgb + texture2D(tScene, cuv + cell * 0.3 / uRes).rgb) * 0.5;
    float L = max(max(c.r, c.g), c.b) + (h11(dot(cid, vec2(1., 57.)) + floor(uT * 18.)) - 0.5) * 0.08;
    float idx = floor(clamp(L * 1.7, 0., 0.999) * 10.);
    vec2 loc = fract(gl_FragCoord.xy / cell);
    float g = texture2D(tAtlas, vec2((idx + loc.x) / 10., loc.y)).r;
    vec3 tint = c / max(L, 0.001);
    vec3 a = tint * g * 1.35 + uCream * g * 0.22;
    col = mix(col, a, inAscii);
  }
  // dither + dot grid + vignette (the 'video B' texture)
  float lv = 14.; col = floor(col * lv + bayer(floor(gl_FragCoord.xy / uDpr))) / lv;
  vec2 gq = fract(gl_FragCoord.xy / (uDpr * 3.));
  col *= mix(.82, 1., 1. - smoothstep(.26, .48, length(gq - .5)));
  vec2 vv = vUv - .5; col *= 1. - .6 * dot(vv, vv);
  gl_FragColor = vec4(col, 1.);
}`
