# Deploying PlantGuard AI — "works once and for all" guide

## Why it was broken

The web app (frontend) was deployed on **Vercel**, but Vercel only serves
static files / small serverless functions. The backend is a **FastAPI app
running a PyTorch model** (~800 MB of dependencies) — it can never run on
Vercel. So every API call (`/health`, `/predict`, `/plants`...) returned
Vercel's `index.html` instead of JSON, which is why you saw:

- `Backend offline` on the home page, and
- `Unexpected token '<', "<!doctype"... is not valid JSON`

**Fix:** the backend runs on **Render** (free tier, `render.yaml` included),
and the frontend knows its URL.

## One-time setup

### 1. Deploy the backend on Render

1. Push this repo to GitHub (if not already).
2. Render Dashboard → **New +** → **Blueprint** → select the repo.
   Render reads `render.yaml` and creates a web service named
   `plantguard-backend`.
3. Wait for the first build (~5–10 min; torch is big). The health check is
   `https://<service-url>/health`.

> If Render names the service `plantguard-backend-ab12cd` (it appends random
> characters when the name is taken), copy the real URL for step 2.

### 2. Point the frontend at the backend

Either rely on the default (`https://plantguard-backend.onrender.com`), or set
an env var on the **Vercel project**:

```
VITE_API_URL = https://<your-render-service-url>
```

Then **redeploy** the frontend on Vercel (env vars are baked at build time).

### 3. Lock down CORS (optional but recommended)

On the Render service, set:

```
CORS_ORIGINS = https://<your-vercel-app>.vercel.app
```

and restart. (`*` works during testing.)

## Alternative: one service for everything (no CORS, no Vercel)

The backend can also serve the built frontend:

```bash
cd frontend_vite
npm install && npm run build   # produces frontend_vite/dist
```

If `frontend_vite/dist` exists when the backend starts, the same Render
service serves both the API and the site at one URL. Deploy the **repo root**
as the Render service and you get the whole app at
`https://plantguard-backend.onrender.com`.

## Notes on the free tier

- Render free services **sleep** after ~15 min idle; the first request takes
  ~30–60 s to wake. The home page retries every 30 s and has a **Retry**
  button. Keep-alive pings (e.g. cron-job.org hitting `/health`) prevent
  sleeping if you want.
- Uploads and the SQLite DB live in `/tmp` on Render — they reset on deploy/
  restart. That's fine for a demo; use Postgres + S3 for persistence.

## Local development

```bash
pip install -r requirements.txt
python -m uvicorn backend.app.main:app --reload --port 8000   # terminal 1
cd frontend_vite && npm install && npm run dev                # terminal 2
```

The dev server proxies `/api/*` to `http://127.0.0.1:8000` automatically.
