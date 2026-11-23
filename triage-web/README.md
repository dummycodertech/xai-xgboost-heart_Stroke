# Triage web (Next.js 14 + Tailwind + React Three Fiber)

    npm install
    npm run dev          # http://localhost:3000
    npm run build

## Wiring your API (one function)
Set `NEXT_PUBLIC_API_URL`. Edit only `fromApi()` in `lib/api.ts` to match your response shape.
Without it the site runs on a labelled demo scorer (not your XGBoost model).

## Real anatomy
Drop `heart.glb` / `brain.glb` in `public/models/` and set `NEXT_PUBLIC_HEART_GLB=/models/heart.glb`
(and `..._BRAIN_GLB`). Any mesh is merged, centred and rendered with the same Fresnel X-ray shader.
Unset = the procedural meshes in `components/hero/geometry.ts`.

### 3D Model Licences & Sources
- **Heart (`public/models/heart.glb`)**: Sourced from the **HuBMAP Consortium** (NIH Common Fund Human Reference Atlas CCF releases, `VH_M_Heart.glb`). Licenced under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Source URL: `https://github.com/hubmapconsortium/ccf-releases`. Apex oriented down-left.
- **Brain (`public/models/brain.glb`)**: Sourced from the **HuBMAP Consortium / Allen Brain Atlas** integration (`Allen_M_Brain.glb`). Licenced under **Creative Commons Attribution 4.0 International (CC BY 4.0)**. Source URL: `https://github.com/hubmapconsortium/ccf-releases`. Optimized and quadric-decimated to 2.0 MB for real-time WebGL rendering.

## Where things live
- `app/globals.css`: palette tokens (exact spec values), SHAP scale, wordmark split
- `components/hero/engine.ts`: render pipeline (blob + subject -> FBO -> glitch/ASCII/dither composite)
- `components/hero/shaders.ts`: all GLSL
- `components/hero/tracker.ts`: brightness peaks read back from the rendered frame
- `components/hero/HudOverlay.tsx`: brackets, micro-squares, constellation lines, telemetry
- `lib/api.ts`: types, strategies, demo scorer, HERO_FEATURES (swap for live top-4 SHAP)
