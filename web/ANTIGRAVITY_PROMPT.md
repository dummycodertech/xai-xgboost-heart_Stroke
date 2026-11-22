# Task for Antigravity: merge this frontend with the stroke-risk backend

You are merging `stroke-xai-frontend/` (Vite + React + TypeScript + three.js) with the existing Python project in `xai-xgboost-heart_Stroke`. Work on a new branch. Plan first, then implement, and summarize what you changed.

## Hard rules
- Do NOT change the color palette in `src/styles.css` (dark "Midnight Telemetry" default, light "Clinical Publication" via `[data-theme=light]`). Red is reserved for risk-increasing signals, powder blue for protective ones.
- Do NOT replace or simplify the 3D heart in `src/Heart.tsx` (drag to rotate, heartbeat glow, surface data points, orbiting particles). Only fix bugs if the build fails.
- Never claim metrics that are not measured. Every number shown must come from the backend `/meta` endpoint or a generated report file.
- No browser storage of patient inputs, no analytics on the form, no logging of request bodies on the server.

## 1. Run and verify the frontend
`npm install && npm run dev`, then `npm run build`. Fix any TypeScript or dependency errors without changing the design. Open it in the browser agent and check the hero, form, SHAP and LIME panels, light/dark toggle, and a 390px mobile width.

## 2. Build the FastAPI backend (new folder `api/`)
- Load the trained pipeline once at startup (prefer one joblib pipeline plus XGBoost JSON model over separate pickles).
- Compute SHAP with `booster.predict(dmatrix, pred_contribs=True)` (exact TreeSHAP, no `shap` library at inference). Sum one-hot columns back to the original feature so each feature has one contribution.
- Compute LIME with at most 1,500 samples. If it is slow, expose it as `POST /explain/lime` and have the UI load it after SHAP.
- Endpoints: `GET /health`, `GET /meta`, `POST /predict`.
- CORS: allow only the Vercel production and preview origins plus `http://localhost:5173`. Add basic rate limiting. Validate inputs with Pydantic using the same bounds as the Streamlit app (age 0-100, glucose 50-300, BMI 10-100).
- Contract for `POST /predict` (request uses dataset field names):
```json
{"age":62,"avg_glucose_level":195,"bmi":32.4,"gender":"Male","hypertension":1,"heart_disease":0,
 "ever_married":"Yes","work_type":"Self-employed","Residence_type":"Urban","smoking_status":"formerly smoked"}
```
Response:
```json
{"probability":0.31,"threshold":0.15,"decision":"HIGH_RISK","base_value":-2.6,
 "shap":[{"feature":"age","value":"62","contribution":1.02}],
 "lime":[{"rule":"age > 61","weight":0.31}]}
```
`decision` is `HIGH_RISK` or `BELOW_THRESHOLD`. SHAP contributions are in log-odds and `base_value + sum(contributions)` equals the model log-odds. `GET /meta` returns model version, threshold and how it was chosen, and metrics with confidence intervals when available.
- Add pytest tests for the endpoints, including that the SHAP sum matches the model output.

## 3. Wire the frontend
- Set `VITE_API_URL`; demo mode turns off automatically and the header chip changes to "Model v1 connected". Delete the `mock` function in `src/lib.ts` only after the API works.
- Replace the hard-coded `MODEL` object in `src/lib.ts` with data from `GET /meta`.
- Keep the cold-start message (free hosting can take about a minute to wake) and the error and retry state.

## 4. Deploy
- Frontend on Vercel (root `stroke-xai-frontend`, build `npm run build`, output `dist`, env `VITE_API_URL`).
- Backend on Render as a web service with a `render.yaml`. Keep dependencies lean for 512 MB RAM.
- Keep the old Streamlit app working until I approve removing it.

## Done when
The deployed UI shows real predictions with SHAP and LIME from the API, the build passes, tests pass, and the README explains how to run and deploy both parts.
