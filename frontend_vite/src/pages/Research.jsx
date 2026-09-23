import { useEffect, useState } from "react";
import { api } from "../api";

export default function Research() {
  const [info, setInfo] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [metricsStatus, setMetricsStatus] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api("/model-info")
      .then(setInfo)
      .catch((e) => setError(e.message));
    api("/eval-metrics")
      .then((r) => {
        setMetricsStatus(r.status);
        setMetrics(r.metrics);
      })
      .catch(() => setMetricsStatus("unavailable"));
  }, []);

  return (
    <div>
      <div className="page-head">
        <h1>Model Research</h1>
        <p className="muted">Architecture, honest capability boundaries, and dataset provenance.</p>
      </div>

      <section className="card">
        <div className="section-head">
          <h2>Model card</h2>
          {info && <span className={`badge ${info.status === "loaded" ? "" : "warn"}`}>{info.status}</span>}
        </div>
        {error && <div className="err">{error}</div>}
        {info && (
          <table className="kv-table">
            <tbody>
              <tr><th>Model</th><td>{info.model_name}</td></tr>
              <tr><th>Version</th><td>{info.version}</td></tr>
              <tr><th>Trained tasks</th><td>{info.available_tasks.join(", ") || "none yet"}</td></tr>
              <tr><th>Unsupported tasks</th><td>{info.unsupported_tasks.join(", ")}</td></tr>
              <tr><th>Disclaimer</th><td>{info.disclaimer}</td></tr>
            </tbody>
          </table>
        )}
      </section>

      <div className="grid">
        <section className="card">
          <h2>Test-set metrics</h2>
          <p className="muted small">
            Metrics are produced by <code>ml/evaluate.py</code> on the held-out test
            split. If a metric file is missing, it has not been computed yet — none
            are invented here.
          </p>
          {metricsStatus === "unavailable" && (
            <div className="err">Could not reach the metrics endpoint.</div>
          )}
          {metrics ? (
            <table className="kv-table">
              <tbody>
                <tr><th>Accuracy</th><td>{(metrics.accuracy * 100).toFixed(1)}%</td></tr>
                <tr><th>Macro F1</th><td>{metrics.macro_f1?.toFixed(3) ?? "—"}</td></tr>
                <tr><th>ECE (calibration)</th><td>{metrics.ece_calibration_error?.toFixed(3) ?? "—"}</td></tr>
                <tr><th>Test images</th><td>{metrics.test_images}</td></tr>
                <tr><th>Coverage @ {metrics.threshold}</th><td>{metrics.coverage_at_threshold != null ? `${(metrics.coverage_at_threshold * 100).toFixed(1)}%` : "—"}</td></tr>
                <tr><th>Accuracy (accepted)</th><td>{metrics.accuracy_accepted != null ? `${(metrics.accuracy_accepted * 100).toFixed(1)}%` : "—"}</td></tr>
              </tbody>
            </table>
          ) : metricsStatus === "not_computed" ? (
            <p className="muted">
              Not computed yet. Run <code>python -m ml.evaluate</code> on the test set —
              real numbers will appear here once produced.
            </p>
          ) : !metricsStatus ? null : (
            <p className="muted">No evaluation results available.</p>
          )}
        </section>

        <section className="card">
          <h2>Training data</h2>
          <ul>
            <li>PlantVillage — 54,305 images, 38 classes</li>
            <li>PlantDoc — 2,552 field images (generalisation check)</li>
            <li>Nutrition — 17,627 (not accepted: label semantics + heavy duplication)</li>
          </ul>
          <p className="metric">
            Split 70/15/15 with near-duplicate leakage control (cluster-level split).
          </p>
        </section>
      </div>
    </div>
  );
}
