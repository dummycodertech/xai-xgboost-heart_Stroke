# Deployment & Hosting Guide: Stroke Risk Triage & Explainable AI

This project consists of two distinct components:
1. **Backend API (`api/`)**: High-performance FastAPI service running a 5-fold calibrated XGBoost pipeline with TreeSHAP and local surrogate explainability. Self-contained in `api/artifacts/`.
2. **Frontend UI (`triage-web/`)**: Next.js 14 App Router with Tailwind CSS, React Three Fiber, Three.js 3D bioluminescent anatomical hero, calibrated clinical charts, and interactive risk operating-point explorer.

---

## Architecture & Hosting Plan

| Component | Technology | Recommended Host | Cost | Setup Time |
| :--- | :--- | :--- | :--- | :--- |
| **Backend API** | FastAPI / Python 3.12 | **Render** (or Railway / Fly.io) | Free Tier | ~5 mins |
| **Frontend Web** | Next.js 14 / TypeScript | **Vercel** (or Netlify / Cloudflare) | Free Tier | ~3 mins |

---

## Part 1: Deploy the Backend API (Render)

### Option A: 1-Click Blueprint (Recommended)
This repository contains a pre-configured `render.yaml` file in the root directory.

1. Push your repository to GitHub (or create a private fork/repo).
2. Go to [dashboard.render.com](https://dashboard.render.com) and sign in with GitHub.
3. Click **New +** → **Blueprint**.
4. Connect your GitHub repository.
5. Render will automatically detect `render.yaml` and configure:
   - **Service Name**: `stroke-xai-api`
   - **Runtime**: Python 3.12.8
   - **Build Command**: `pip install -r api/requirements.txt`
   - **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT --proxy-headers --forwarded-allow-ips='*'`
   - **Health Check Path**: `/health`
6. Click **Apply**.
7. Once deployed, copy your API URL (e.g., `https://stroke-xai-api.onrender.com`).

### Option B: Manual Setup on Render
1. Click **New +** → **Web Service**.
2. Connect your repository.
3. Configure settings:
   - **Name**: `stroke-xai-api`
   - **Region**: Choose closest to you (e.g., Oregon or Frankfurt)
   - **Root Directory**: leave blank (or `.`)
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r api/requirements.txt`
   - **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT --proxy-headers --forwarded-allow-ips='*'`
4. Under **Advanced** → **Add Environment Variables**:
   - `PYTHON_VERSION`: `3.12.8`
   - `ENVIRONMENT`: `production`
   - `FRONTEND_URL`: `https://your-frontend-domain.vercel.app` (you can update this after deploying the frontend)
   - `TRUSTED_PROXIES`: `127.0.0.1,::1`
5. Under **Health Check Path**: set `/health`.
6. Click **Create Web Service**.

---

## Part 2: Deploy the Frontend (Vercel)

Vercel is built by the creators of Next.js and provides zero-config deployment.

1. Go to [vercel.com](https://vercel.com) and log in with GitHub.
2. Click **Add New...** → **Project**.
3. Import your GitHub repository.
4. **IMPORTANT**: In the **Configure Project** screen:
   - **Framework Preset**: `Next.js`
   - **Root Directory**: Click **Edit** and select `triage-web` (do NOT leave it at `./`).
5. Open **Environment Variables** and add:
   - `NEXT_PUBLIC_API_URL`: Your backend URL from Render (e.g. `https://stroke-xai-api.onrender.com`).
     *(Note: If left blank, the frontend runs in built-in demo mode).*
   - `NEXT_PUBLIC_HEART_GLB`: `/models/heart.glb`
   - `NEXT_PUBLIC_BRAIN_GLB`: `/models/brain.glb`
6. Click **Deploy**.
7. Vercel will install dependencies, build the Next.js bundle, and give you a live production URL (e.g., `https://triage-stroke.vercel.app`).

---

## Part 3: Connect Frontend & Backend (CORS)

To ensure secure cross-origin requests between your Vercel frontend and Render backend:

1. Copy your Vercel production domain (e.g. `https://triage-stroke.vercel.app`).
2. Go to your Render Dashboard → select `stroke-xai-api` → **Environment**.
3. Set `FRONTEND_URL` to your exact Vercel URL (e.g. `https://triage-stroke.vercel.app`).
4. If you use Vercel Preview Deployments, set `VERCEL_PREVIEW_REGEX` to match your team/project preview URLs:
   ```env
   VERCEL_PREVIEW_REGEX=^https:\/\/triage-stroke(-[a-z0-9-]+)?-[a-z0-9-]+\.vercel\.app$
   ```
5. Click **Save Changes**. Render will automatically redeploy with the updated CORS policy.

---

## Part 4: Verification & Smoke Test

### 1. Test Backend Health & Prediction
Run in your terminal or browser:
```bash
# 1. Health check
curl -X GET https://YOUR-API-URL.onrender.com/health
# Expected output: {"status":"healthy"}

# 2. Test prediction endpoint
curl -X POST https://YOUR-API-URL.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 67.0,
    "hypertension": 0,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked"
  }'
```

### 2. Test Frontend
1. Open your Vercel URL in your browser.
2. Confirm the 3D anatomical model (Heart/Brain) renders and rotates smoothly in dark mode.
3. Switch between **01 DRIVERS** and **02 OUTCOME** to observe the vertex morphing transition.
4. Adjust patient sliders in the Predictor to see live updates from the model.
5. Inspect the calibration curve, ROC curve, and confusion matrix under the Model section.
