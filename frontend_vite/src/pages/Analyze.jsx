import Diagnostic from "../components/Diagnostic";

export default function Analyze() {
  return (
    <div>
      <div className="page-head">
        <h1>Analyze</h1>
        <p className="muted">
          The full diagnostic workflow lives on the main interface — this page
          focuses on the standalone analysis run.
        </p>
      </div>
      <Diagnostic />
    </div>
  );
}
