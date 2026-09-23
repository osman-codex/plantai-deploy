// Resolve the backend base URL once, at module load.
//
// Priority:
//   1. VITE_API_URL env var (set in Vercel project settings or .env) — always wins.
//   2. In dev (vite server), use "/api" and let the vite proxy forward to the
//      local backend (see vite.config.js).
//   3. In a production build, default to the Render backend deployed from this
//      repo (render.yaml -> service name "plantguard-backend").
export const API_URL = (() => {
  const env = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");
  if (env) return env;
  if (import.meta.env.DEV) return "/api";
  // Page served by the backend itself (single-service deploy) -> same-origin.
  if (/onrender\.com$/.test(window.location.hostname)) return "";
  // Split deploy (frontend on Vercel) -> Render backend from this repo.
  return "https://plantguard-backend.onrender.com";
})();

async function handle(res) {
  if (!res.ok) {
    // Prefer the API's JSON error detail; if the server answered with HTML
    // (e.g. a misconfigured proxy), say so instead of throwing a cryptic
    // "Unexpected token '<'" JSON parse error.
    const ct = res.headers.get("content-type") || "";
    let detail = res.statusText || `Request failed (${res.status})`;
    if (ct.includes("application/json")) {
      try {
        const body = await res.json();
        detail = body.detail || body.message || detail;
      } catch {
        /* keep statusText */
      }
    } else if (ct.includes("text/html")) {
      detail = `Backend returned an HTML page instead of JSON (HTTP ${res.status}). Check that the backend URL is correct and the service is awake.`;
    }
    throw new Error(detail);
  }
  const ct = res.headers.get("content-type") || "";
  if (!ct.includes("application/json")) {
    throw new Error(
      "Backend returned a non-JSON response. The API URL is probably wrong (points at the site itself, not the backend)."
    );
  }
  return res.json();
}

export async function api(path, options = {}) {
  let res;
  try {
    res = await fetch(`${API_URL}${path}`, options);
  } catch (networkErr) {
    // fetch only rejects for network/CORS failures — give these a human message.
    if (networkErr instanceof TypeError) {
      throw new Error(
        `Cannot reach the backend at ${API_URL} (network or CORS error). If the backend was just deployed or woken up, wait a few seconds and retry — free tiers sleep.`
      );
    }
    throw networkErr;
  }
  return handle(res);
}

export function predictImage(file) {
  const fd = new FormData();
  fd.append("file", file);
  return api("/predict", { method: "POST", body: fd });
}

export function createPlant(profile) {
  return api("/plants", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(profile),
  });
}

export function uploadObservation(plantId, file, notes = "") {
  const fd = new FormData();
  fd.append("file", file);
  fd.append("notes", notes);
  return api(`/plants/${plantId}/observations`, { method: "POST", body: fd });
}

export function fetchHealth() {
  return api("/health");
}
