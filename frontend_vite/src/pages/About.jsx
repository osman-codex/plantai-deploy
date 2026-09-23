export default function About() {
  return (
    <div>
      <div className="page-head">
        <h1>About</h1>
        <p className="muted">What PlantGuard AI is, and the rule it never breaks.</p>
      </div>

      <section className="card">
        <h2>About PlantGuard AI</h2>
        <p>
          PlantGuard AI trains a real convolution network (EfficientNet-B0) on the
          PlantVillage dataset and serves it through a FastAPI backend with a React
          frontend and an Expo-powered mobile client.
        </p>
        <p>
          The key rule of the project: <strong>never fabricate</strong>. A prediction is
          only emitted when the model&apos;s softmax confidence passes a fixed threshold;
          everything else is honestly labelled &ldquo;insufficient evidence&rdquo;.
        </p>
      </section>

      <div className="grid">
        <section className="card">
          <h2>Stack</h2>
          <ul>
            <li>PyTorch (CPU optimised) + timm</li>
            <li>FastAPI + SQLAlchemy/Alembic (SQLite local, Postgres ready)</li>
            <li>React 18 + Vite</li>
            <li>React Native / Expo for mobile</li>
          </ul>
        </section>
        <section className="card">
          <h2>Datasets trained on</h2>
          <ul>
            <li>PlantVillage — 54,305 images, 38 classes</li>
            <li>PlantDoc — 2,552 field images (generalisation check)</li>
            <li>Nutrition — 17,627 (not yet accepted: label semantics + heavy duplication)</li>
          </ul>
        </section>
      </div>
    </div>
  );
}
