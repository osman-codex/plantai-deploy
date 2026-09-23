import { useRef, useState } from "react";
import { predictImage } from "../api";

const fmtPct = (v) => `${(v * 100).toFixed(1)}%`;

export default function Diagnostic({ compact = false }) {
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [preview, setPreview] = useState(null);
  const inputRef = useRef(null);

  async function onFile(file) {
    if (!file) return;
    setError("");
    setResult(null);
    setPreview(URL.createObjectURL(file));
    setLoading(true);
    try {
      const data = await predictImage(file);
      setResult(data);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const pred = result?.prediction;

  return (
    <section className={`card diagnostic${compact ? " compact" : ""}`}>
      <div className="section-head">
        <h2>Diagnostics</h2>
        <span className="pill">real model · 38 classes</span>
      </div>
      <p className="muted">
        Upload a leaf photo. The model reports a diagnosis only when its confidence
        passes the threshold — otherwise it says so.
      </p>

      <div
        className={`file-drop${loading ? " busy" : ""}`}
        onClick={() => !loading && inputRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          if (!loading) onFile(e.dataTransfer.files?.[0]);
        }}
      >
        <input
          ref={inputRef}
          type="file"
          accept="image/*"
          hidden
          onChange={(e) => onFile(e.target.files[0])}
        />
        {loading ? (
          <span className="drop-busy"><span className="spinner" /> Analyzing…</span>
        ) : (
          <span className="drop-idle">
            <strong>Click or drop an image</strong>
            <small>JPG · PNG · WebP</small>
          </span>
        )}
      </div>

      {error && <div className="err">{error}</div>}

      {pred && (
        <div className="diag-result">
          <div className="grid media-grid">
            {preview && (
              <figure>
                <img className="preview" src={preview} alt="uploaded leaf" />
                <figcaption>Your photo</figcaption>
              </figure>
            )}
            {pred.explainability?.grad_cam?.overlay && (
              <figure>
                <img
                  className="preview"
                  src={pred.explainability.grad_cam.overlay}
                  alt="Grad-CAM overlay"
                />
                <figcaption>Model attention (Grad-CAM)</figcaption>
              </figure>
            )}
          </div>

          {!pred.accepted ? (
            <div className="warn">
              <strong>Insufficient confidence.</strong> {pred.reason}
            </div>
          ) : (
            <>
              <div className="verdict">
                <div>
                  <span className="muted small">Diagnosis</span>
                  <div className="verdict-name">{pred.diagnosis.disease}</div>
                </div>
                <div className="verdict-conf">
                  <span className="muted small">Confidence</span>
                  <div className="verdict-pct">{fmtPct(pred.diagnosis.confidence)}</div>
                </div>
                <div>
                  <span className="muted small">Health status</span>
                  <div>
                    <span className={`badge ${/healthy/i.test(pred.disease_health_status || "") ? "" : "warn"}`}>
                      {pred.disease_health_status}
                    </span>
                  </div>
                </div>
              </div>

              <div className="progress lg">
                <div style={{ width: `${pred.diagnosis.confidence * 100}%` }} />
              </div>

              <h3>Other candidates</h3>
              <div className="bars">
                {pred.matches.slice(1).map((m) => (
                  <div className="bar" key={m.class}>
                    <span className="bar-label">{m.display_name}</span>
                    <span className="progress"><div style={{ width: `${m.confidence * 100}%` }} /></span>
                    <span className="bar-val">{fmtPct(m.confidence)}</span>
                  </div>
                ))}
              </div>

              {pred.care_plan ? (
                <>
                  <h3>Suggested treatment & next steps</h3>
                  <div className="care-plan">
                    <div className="care-block">
                      <h4>① Do this now</h4>
                      <ul>
                        {pred.care_plan.actions.map((a) => <li key={a}>{a}</li>)}
                      </ul>
                    </div>
                    {pred.care_plan.products.length > 0 && (
                      <div className="care-block">
                        <h4>② Sprays & products</h4>
                        <ul>
                          {pred.care_plan.products.map((p) => <li key={p}>{p}</li>)}
                        </ul>
                      </div>
                    )}
                    {pred.care_plan.nutrition.length > 0 && (
                      <div className="care-block">
                        <h4>③ Fertilizer & nutrients</h4>
                        <ul>
                          {pred.care_plan.nutrition.map((n) => <li key={n}>{n}</li>)}
                        </ul>
                      </div>
                    )}
                    {pred.care_plan.prevention.length > 0 && (
                      <div className="care-block">
                        <h4>④ Prevent it next season</h4>
                        <ul>
                          {pred.care_plan.prevention.map((p) => <li key={p}>{p}</li>)}
                        </ul>
                      </div>
                    )}
                    <p className="metric">{pred.care_plan.scope}</p>
                  </div>
                </>
              ) : pred.treatment_recommendations?.length > 0 && (
                <>
                  <h3>Reference guidance</h3>
                  {pred.treatment_recommendations.map((t) => (
                    <div key={t.disease} className="subcard">
                      <strong>{t.disease}</strong>
                      <p>{t.description}</p>
                      <p className="guidance">{t.guidance}</p>
                      <p className="metric">{t.scope}</p>
                    </div>
                  ))}
                </>
              )}
            </>
          )}

          {pred.quality_warnings?.length > 0 && (
            <div className="warn">
              <strong>Quality warnings:</strong> {pred.quality_warnings.join(" · ")}
            </div>
          )}
          {pred.grad_cam?.available && (
            <p className="metric">
              Model focused on {fmtPct(pred.grad_cam.high_attention_fraction_0_4)} of the
              image at a high-attention level (Grad-CAM threshold 0.4).
            </p>
          )}
          <p className="metric">
            model {pred.model_version} · latency{" "}
            {(result.latency_ms ?? pred.latency_ms ?? 0).toFixed(0)} ms
          </p>
          {pred.notes?.length > 0 && (
            <ul className="notes">
              {pred.notes.map((n) => <li key={n}>{n}</li>)}
            </ul>
          )}
        </div>
      )}
    </section>
  );
}
