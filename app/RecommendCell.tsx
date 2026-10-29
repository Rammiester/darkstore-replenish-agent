"use client";

import { useState } from "react";

interface Props {
  storeId: string;
  skuId: string;
  date: string;
  mlForecast: number;
}

interface Recommendation {
  agent_forecast: number;
  confidence: number;
  adjusted: boolean;
  rationale: string;
  retrieved_context: string[];
}

export default function RecommendCell({ storeId, skuId, date, mlForecast }: Props) {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [rec, setRec] = useState<Recommendation | null>(null);

  async function run() {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch("/api/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ storeId, skuId, date, mlForecast }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "request failed");
      setRec(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "unknown error");
    } finally {
      setLoading(false);
    }
  }

  if (rec) {
    return (
      <div>
        <span>{rec.agent_forecast.toFixed(1)} units</span>{" "}
        <span className={`badge ${rec.adjusted ? "adjusted" : "unadjusted"}`}>
          {rec.adjusted ? "adjusted" : "confirmed ML"}
        </span>
        <div className="rationale">{rec.rationale}</div>
        {rec.retrieved_context.length > 0 && (
          <div className="rationale">
            Context used: {rec.retrieved_context.join(", ")}
          </div>
        )}
      </div>
    );
  }

  return (
    <div>
      <button className="recommend-btn" onClick={run} disabled={loading}>
        {loading ? "Thinking…" : "Get agent recommendation"}
      </button>
      {error && <div className="rationale" style={{ color: "var(--bad)" }}>{error}</div>}
    </div>
  );
}
