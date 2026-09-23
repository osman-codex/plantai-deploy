import { useEffect, useState } from "react";
import { api } from "../api";
import Diagnostic from "../components/Diagnostic";

export default function Home() {
  const [health, setHealth] = useState(null);
  const [error, setError] = useState("");
  const [checking, setChecking] = useState(true);

  function check() {
    setChecking(true);
    setError("");
    api("/health")
      .then((h) => {
        setHealth(h);
        setError("");
      })
      .catch((e) => {
        setHealth(null);
        setError(e.message);
      })
      .finally(() => setChecking(false));
  }

  useEffect(() => {
    check();
    const iv = setInterval(check, 30000); // free tiers sleep; retry quietly
    return () => clearInterval(iv);
  }, []);

  return (
    <div>
      <div className="hero">
        <div>
          <h1>PlantGuard <em className="hero-em">AI</em></h1>
          <p>
            Analyze a leaf photo with a real, locally trained neural network
            (EfficientNet-B0, 38 classes). Diagnostics live right here on the main
            interface — monitoring and research are one click away.
          </p>
        </div>
        <div className="hero-status">
          <span className={`status-chip ${health ? "ok" : checking ? "connecting" : "down"}`}>
            <span className="dot" />
            {health ? `Backend online · model ${health.model}` : checking ? "Connecting…" : "Backend offline"}
          </span>
          {!health && (
            <button className="retry-btn" onClick={check} disabled={checking}>
              {checking ? "Checking…" : "Retry"}
            </button>
          )}
        </div>
      </div>

      {error && <div className="err">API error: {error}</div>}

      <Diagnostic />

      <div className="grid">
        <div className="card">
          <h2>What it can do</h2>
          <ul>
            <li>Disease classification across 38 classes</li>
            <li>Plant identification + health status</li>
            <li>Grad-CAM heatmap of the focused region</li>
            <li>Confidence thresholding with a clear "unsure" result</li>
            <li>Per-plant progress tracking in <a href="/monitor">Monitor</a></li>
          </ul>
        </div>
        <div className="card">
          <h2>What it will never fake</h2>
          <ul>
            <li>Severity estimation (no labels exist — no model)</li>
            <li>Nutrient deficiency (requires a dedicated model)</li>
            <li>Confidence or metrics pulled from imagination</li>
          </ul>
        </div>
      </div>
    </div>
  );
}
