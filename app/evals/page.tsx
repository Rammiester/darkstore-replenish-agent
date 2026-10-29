import fs from "fs";
import path from "path";

export const dynamic = "force-dynamic";

interface Report {
  summary: {
    generatedAt: string;
    model: string;
    dryRun: boolean;
    totalTestRows: number;
    totalAnomalyRows: number;
    evaluatedAnomalyRows: number;
    maeAnomalyMlOnly: number | null;
    maeAnomalyAgent: number | null;
    maeNonAnomalyMlOnly: number | null;
    adjustedCount: number;
  };
  rows: {
    store_id: string;
    sku_id: string;
    date: string;
    actual: number;
    ml_forecast: number;
    is_anomaly_day: boolean;
    matched_events: string[];
    agent_forecast: number | null;
    agent_adjusted: boolean | null;
    agent_error: string | null;
  }[];
}

function loadReport(): Report | null {
  const p = path.join(process.cwd(), "evals", "report.json");
  if (!fs.existsSync(p)) return null;
  return JSON.parse(fs.readFileSync(p, "utf-8"));
}

export default function EvalsPage() {
  const report = loadReport();

  return (
    <div className="wrap">
      <div className="hgroup">
        <h1>Evals: ML-only vs. agent-adjusted</h1>
        <p>
          Comparing forecast error on anomaly days (festival/heatwave/competitor-outage days in
          the held-out test window) against normal days. The ML model is trained only on
          historical sales, so it has no way to know an anomaly is coming — this measures whether
          the agent&apos;s retrieval + tool calling actually closes that gap.
        </p>
      </div>
      <nav className="tabs">
        <a href="/">Dashboard</a>
        <a className="active" href="/evals">Evals</a>
      </nav>

      {!report && (
        <div className="card">
          <p>
            No eval report yet. Run <code>npm run eval -- --dry-run</code> to check the anomaly-day
            detection logic for free, or <code>npm run eval</code> (requires{" "}
            <code>ANTHROPIC_API_KEY</code>, makes real API calls) for the full comparison.
          </p>
        </div>
      )}

      {report && (
        <>
          <div className="metric-row">
            <div className="metric">
              <div className="label">ML-only MAE — anomaly days</div>
              <div className="value bad">
                {report.summary.maeAnomalyMlOnly?.toFixed(2) ?? "—"}
              </div>
            </div>
            <div className="metric">
              <div className="label">Agent MAE — anomaly days</div>
              <div className="value good">
                {report.summary.maeAnomalyAgent?.toFixed(2) ?? "—"}
              </div>
            </div>
            <div className="metric">
              <div className="label">ML-only MAE — normal days</div>
              <div className="value">
                {report.summary.maeNonAnomalyMlOnly?.toFixed(2) ?? "—"}
              </div>
            </div>
          </div>

          {report.summary.dryRun && (
            <div className="card">
              This report is from a <strong>--dry-run</strong>: it confirms which days are flagged
              as anomaly days and the ML-only baseline error on them, but makes no agent calls.
              The ML-only MAE above being much higher on anomaly days than normal days (
              {report.summary.maeAnomalyMlOnly?.toFixed(2)} vs{" "}
              {report.summary.maeNonAnomalyMlOnly?.toFixed(2)}) is exactly the gap the agent is
              meant to close — run <code>npm run eval</code> with an API key to measure that.
            </div>
          )}

          <table>
            <thead>
              <tr>
                <th>Store/SKU</th>
                <th>Date</th>
                <th>Event</th>
                <th>Actual</th>
                <th>ML forecast</th>
                <th>Agent forecast</th>
              </tr>
            </thead>
            <tbody>
              {report.rows
                .filter((r) => r.is_anomaly_day)
                .map((r) => (
                  <tr key={`${r.store_id}-${r.sku_id}-${r.date}`}>
                    <td>
                      {r.store_id} / {r.sku_id}
                    </td>
                    <td>{r.date}</td>
                    <td>{r.matched_events.join(", ")}</td>
                    <td>{r.actual}</td>
                    <td>{r.ml_forecast.toFixed(1)}</td>
                    <td>
                      {r.agent_forecast !== null
                        ? r.agent_forecast.toFixed(1)
                        : r.agent_error
                        ? `error: ${r.agent_error}`
                        : "—"}
                    </td>
                  </tr>
                ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  );
}
