/**
 * Plant progress graph — renders real timeline data only (no fabricated points).
 * Shows: confidence trend line, health status markers, per-observation thumbnails.
 * Pure SVG, no chart library needed.
 */
import { API_URL } from "../api";

const W = 640;
const H = 240;
const PAD = { t: 18, r: 16, b: 34, l: 44 };

const fmtPct = (v) => `${(v * 100).toFixed(0)}%`;

function niceTime(iso) {
  const d = new Date(iso);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" }) +
    " " + d.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

export default function ProgressChart({ observations, message }) {
  const obs = (observations || []).map((o, i) => ({
    ...o,
    conf: typeof o.confidence === "number" ? o.confidence : null,
    healthy: /healthy/i.test(o.health_status || ""),
    idx: i + 1,
  }));

  if (obs.length === 0) {
    return (
      <div className="chart-empty">
        <p>{message || "No observations yet."}</p>
      </div>
    );
  }

  const withConf = obs.filter((o) => o.conf != null);
  const n = obs.length;

  // X positions: evenly spaced by observation number (time is monotonic from API)
  const xFor = (i) => PAD.l + (i * (W - PAD.l - PAD.r)) / Math.max(n - 1, 1);

  // Y scale: focus on the observed confidence range (min 0.4 span so line is readable)
  let lo = withConf.length ? Math.min(...withConf.map((o) => o.conf)) : 0;
  let hi = withConf.length ? Math.max(...withConf.map((o) => o.conf)) : 1;
  if (hi - lo < 0.2) { const mid = (hi + lo) / 2; lo = Math.max(0, mid - 0.1); hi = Math.min(1, mid + 0.1); }
  lo = Math.max(0, lo - 0.08);
  hi = Math.min(1, hi + 0.08);
  const yFor = (v) => PAD.t + (1 - (v - lo) / (hi - lo)) * (H - PAD.t - PAD.b);

  const pts = withConf.map((o) => {
    const i = obs.indexOf(o);
    return { x: xFor(i), y: yFor(o.conf), o };
  });
  const path = pts.map((p, i) => `${i === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(" ");
  const area = pts.length > 1
    ? `${path} L${pts[pts.length - 1].x.toFixed(1)},${H - PAD.b} L${pts[0].x.toFixed(1)},${H - PAD.b} Z`
    : "";

  const healthyCount = obs.filter((o) => o.healthy).length;

  return (
    <div className="chart-wrap">
      <div className="chart-head">
        <h3>Progress graph</h3>
        <div className="chart-legend">
          <span><i className="lg-line" /> confidence</span>
          <span><i className="lg-dot ok" /> healthy</span>
          <span><i className="lg-dot bad" /> diseased</span>
        </div>
      </div>

      <svg
        className="progress-svg"
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label="Plant confidence trend across observations"
        preserveAspectRatio="xMidYMid meet"
      >
        <defs>
          <linearGradient id="pg-area-grad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor="#4a7c23" stopOpacity="0.28" />
            <stop offset="1" stopColor="#4a7c23" stopOpacity="0.02" />
          </linearGradient>
        </defs>
        {/* horizontal gridlines + labels */}
        {[0, 0.25, 0.5, 0.75, 1].map((t) => {
          const v = lo + t * (hi - lo);
          const y = yFor(v);
          return (
            <g key={t}>
              <line x1={PAD.l} y1={y} x2={W - PAD.r} y2={y} className="gridline" />
              <text x={PAD.l - 6} y={y + 4} textAnchor="end" className="tick">{fmtPct(v)}</text>
            </g>
          );
        })}

        {/* 0.55 confidence threshold marker */}
        {(() => {
          if (0.55 < lo || 0.55 > hi) return null;
          const y = yFor(0.55);
          return (
            <g>
              <line x1={PAD.l} y1={y} x2={W - PAD.r} y2={y} className="threshold" />
              <text x={W - PAD.r} y={y - 5} textAnchor="end" className="tick threshold-label">
                accept threshold 55%
              </text>
            </g>
          );
        })()}

        {area && <path d={area} className="chart-area" />}

        {pts.length > 1 && <path d={path} className="chart-line" />}

        {pts.map((p, k) => (
          <g key={k} className="pt">
            <circle
              cx={p.x}
              cy={p.y}
              r={6}
              className={`pt-hit`}
              data-tip={`${niceTime(p.o.date)} — ${p.o.disease || "?"} — ${fmtPct(p.o.conf)}`}
            />
            <circle
              cx={p.x}
              cy={p.y}
              r={4.5}
              className={`pt-dot ${p.o.healthy ? "ok" : "bad"}`}
            />
            <text x={p.x} y={H - PAD.b + 16} textAnchor="middle" className="tick">#{p.o.idx}</text>
          </g>
        ))}
      </svg>

      <div className="chart-summary">
        <div className="stat">
          <span className="stat-num">{n}</span>
          <span className="stat-label">observations</span>
        </div>
        <div className="stat">
          <span className="stat-num">{healthyCount}</span>
          <span className="stat-label">healthy checks</span>
        </div>
        <div className="stat">
          <span className="stat-num">
            {withConf.length ? fmtPct(withConf.reduce((s, o) => s + o.conf, 0) / withConf.length) : "—"}
          </span>
          <span className="stat-label">mean confidence</span>
        </div>
        <div className="stat">
          <span className="stat-num">
            {withConf.length > 1
              ? (withConf[withConf.length - 1].conf - withConf[0].conf) >= 0 ? "▲" : "▼"
              : "—"}
          </span>
        <span className="stat-label">confidence trend</span>
        </div>
      </div>

      <p className="muted small">
        Confidence is how sure the model was at each visit — it is not a severity measure.
        The backend does not compute severity trends (no severity labels exist in the data).
      </p>

      {obs.length > 0 && (
        <div className="thumbs">
          {obs.map((o) => {
            const src = o.image_path
              ? `${API_URL}/data/plants/${o.image_path.split(/[\\/]/).slice(-2).join("/")}`
              : null;
            return (
              <figure key={o.idx} className={`thumb ${o.healthy ? "ok" : "bad"}`} title={`${niceTime(o.date)} — ${o.disease || "?"}`}>
                {src ? <img src={src} alt={`observation ${o.idx}`} /> : <div className="thumb-ph">no img</div>}
                <figcaption>#{o.idx} {fmtPct(o.conf ?? 0)}</figcaption>
                <span className={`badge ${o.healthy ? "" : "warn"}`}>{o.disease || o.health_status}</span>
              </figure>
            );
          })}
        </div>
      )}
    </div>
  );
}
