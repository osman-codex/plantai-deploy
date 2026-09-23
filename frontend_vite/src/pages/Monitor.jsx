import { useEffect, useState } from "react";
import { api, createPlant, uploadObservation } from "../api";
import ProgressChart from "../components/ProgressChart";

export default function Monitor() {
  const [plants, setPlants] = useState([]);
  const [selected, setSelected] = useState(null);
  const [timeline, setTimeline] = useState(null);
  const [error, setError] = useState("");
  const [crop, setCrop] = useState("");
  const [notes, setNotes] = useState("");
  const [busy, setBusy] = useState(false);
  const [file, setFile] = useState(null);

  async function refresh() {
    try {
      setPlants(await api("/plants"));
    } catch (e) {
      setError(e.message);
    }
  }

  async function highlight(id) {
    setSelected(id);
    setError("");
    try {
      setTimeline(await api(`/plants/${id}/timeline`));
    } catch (e) {
      setError(e.message);
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function addPlant(e) {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      const r = await createPlant({ crop });
      setCrop("");
      await refresh();
      await highlight(r.plant_id);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  async function addObservation(e) {
    e.preventDefault();
    if (!selected || !file) return;
    setBusy(true);
    setError("");
    try {
      await uploadObservation(selected, file, notes);
      setFile(null);
      setNotes("");
      setTimeline(await api(`/plants/${selected}/timeline`));
      await refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Plant Monitoring</h1>
          <p className="muted">
            Register plants, upload leaf photos per visit, and follow each plant's
            progress graph over time.
          </p>
        </div>
      </div>

      <div className="monitor-grid">
        <div>
          <section className="card">
            <div className="section-head">
              <h2>Register a plant</h2>
            </div>
            <form onSubmit={addPlant} className="row-form">
              <input
                value={crop}
                onChange={(e) => setCrop(e.target.value)}
                placeholder="Crop name, e.g. Tomato"
                required
              />
              <button className="primary" disabled={busy} type="submit">
                Register plant
              </button>
            </form>
            {error && <div className="err">{error}</div>}
          </section>

          <section className="card">
            <div className="section-head">
              <h2>Registered plants</h2>
              <span className="pill">{plants.length}</span>
            </div>
            {plants.length === 0 && (
              <p className="muted">No plants yet — register one above.</p>
            )}
            <div className="plant-list">
              {plants.map((p) => (
                <button
                  key={p.plant_id}
                  className={`plant-row${selected === p.plant_id ? " active" : ""}`}
                  onClick={() => highlight(p.plant_id)}
                >
                  <span className="plant-crop">{p.crop}</span>
                  <span className="plant-meta">
                    {p.observations_count} obs · {p.latest_status ?? "no data"}
                  </span>
                </button>
                ))}
            </div>
          </section>
        </div>

        <div>
          {selected && (
            <section className="card">
              <div className="section-head">
                <h2>Add observation — {selected}</h2>
              </div>
              <form onSubmit={addObservation}>
                <div
                  className="file-drop slim"
                  onClick={() => document.getElementById("obs-file")?.click()}
                >
                  <input
                    id="obs-file"
                    type="file"
                    accept="image/*"
                    hidden
                    onChange={(e) => setFile(e.target.files[0] ?? null)}
                  />
                  {file ? `${file.name} — click to change` : "Choose leaf photo"}
                </div>
                <input
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                  placeholder="Observation notes (optional)"
                />
                <button className="primary" disabled={busy || !file} type="submit">
                  Upload & analyze
                </button>
              </form>
            </section>
          )}

          {timeline && (
            <section className="card">
              <div className="section-head">
                <h2>{timeline.crop ? `${timeline.crop} — ` : ""}progress report</h2>
                <span className="pill">{timeline.total_observations} obs</span>
              </div>
              <ProgressChart
                observations={timeline.observations}
                message={timeline.message}
              />
            </section>
          )}
        </div>
      </div>
    </div>
  );
}
