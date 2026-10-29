/**
 * Eval harness: compares ML-only forecast error against agent-adjusted
 * forecast error, specifically on "anomaly days" in the held-out test
 * window — days where a context_docs event (festival, heatwave, etc.)
 * matches the store's region and/or the SKU's category. The base ML model
 * never sees those events; this measures whether the agent's RAG + tool
 * calling actually closes that gap, or just adds noise.
 *
 * This makes real Claude API calls (one run per evaluated row) and costs
 * real money — it does not run automatically. Requires ANTHROPIC_API_KEY.
 *
 * Usage:
 *   npm run eval                  # evaluate every anomaly-day row in the test window
 *   npm run eval -- --limit 5     # cap how many agent calls are made (cost control)
 *   npm run eval -- --dry-run     # skip agent calls entirely; just report the
 *                                  ML-only baseline and which rows are anomaly days
 *                                  (useful for sanity-checking the eval logic itself
 *                                  without spending anything)
 */
import fs from "fs";
import path from "path";
import { getDb, getStores, getSkus } from "../lib/db";
import { runAgent } from "../lib/agent";

interface EvalRow {
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
}

function parseArgs() {
  const args = process.argv.slice(2);
  const limit = args.includes("--limit")
    ? parseInt(args[args.indexOf("--limit") + 1], 10)
    : undefined;
  const dryRun = args.includes("--dry-run");
  return { limit, dryRun };
}

function getTestWindowWithActuals() {
  const db = getDb();
  return db
    .prepare(
      `SELECT f.store_id, f.sku_id, f.date, f.ml_forecast, s.units_sold AS actual
       FROM forecasts f
       JOIN sales s ON s.store_id = f.store_id AND s.sku_id = f.sku_id AND s.date = f.date`
    )
    .all() as unknown as {
    store_id: string;
    sku_id: string;
    date: string;
    ml_forecast: number;
    actual: number;
  }[];
}

function getContextDocsRaw() {
  const db = getDb();
  return db.prepare("SELECT * FROM context_docs").all() as unknown as {
    date: string;
    region: string;
    category: string;
    title: string;
  }[];
}

async function main() {
  const { limit, dryRun } = parseArgs();
  const stores = Object.fromEntries(getStores().map((s) => [s.id, s]));
  const skus = Object.fromEntries(getSkus().map((s) => [s.id, s]));
  const contextDocs = getContextDocsRaw();
  const testRows = getTestWindowWithActuals();

  const annotated: EvalRow[] = testRows.map((r) => {
    const store = stores[r.store_id];
    const sku = skus[r.sku_id];
    const matched = contextDocs.filter(
      (d) =>
        d.date === r.date &&
        (d.region === "all" || d.region === store.region) &&
        (d.category === "all" || d.category === sku.category)
    );
    return {
      store_id: r.store_id,
      sku_id: r.sku_id,
      date: r.date,
      actual: r.actual,
      ml_forecast: r.ml_forecast,
      is_anomaly_day: matched.length > 0,
      matched_events: matched.map((m) => m.title),
      agent_forecast: null,
      agent_adjusted: null,
      agent_error: null,
    };
  });

  const anomalyRows = annotated.filter((r) => r.is_anomaly_day);
  const toRun = limit ? anomalyRows.slice(0, limit) : anomalyRows;

  console.log(
    `${annotated.length} test rows total, ${anomalyRows.length} are anomaly days.` +
      (dryRun
        ? " (--dry-run: skipping agent calls)"
        : ` Running the agent on ${toRun.length} of them.`)
  );

  if (!dryRun) {
    for (const row of toRun) {
      const store = stores[row.store_id];
      const sku = skus[row.sku_id];
      try {
        const rec = await runAgent({
          store,
          sku,
          date: row.date,
          mlForecast: row.ml_forecast,
        });
        row.agent_forecast = rec.agent_forecast;
        row.agent_adjusted = rec.adjusted;
      } catch (err) {
        row.agent_error = err instanceof Error ? err.message : String(err);
        console.error(`  agent failed for ${row.store_id}/${row.sku_id}/${row.date}: ${row.agent_error}`);
      }
    }
  }

  const mae = (rows: EvalRow[], field: "ml_forecast" | "agent_forecast") => {
    const vals = rows
      .map((r) => (field === "ml_forecast" ? r.ml_forecast : r.agent_forecast))
      .filter((v): v is number => v !== null)
      .map((v, i) => Math.abs(v - rows[i].actual));
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : null;
  };

  const evaluatedAnomalyRows = toRun.filter((r) => r.agent_forecast !== null || dryRun);
  const nonAnomalyRows = annotated.filter((r) => !r.is_anomaly_day);

  const summary = {
    generatedAt: new Date().toISOString(),
    model: process.env.ANTHROPIC_MODEL || "claude-opus-5-5",
    dryRun,
    totalTestRows: annotated.length,
    totalAnomalyRows: anomalyRows.length,
    evaluatedAnomalyRows: evaluatedAnomalyRows.length,
    maeAnomalyMlOnly: mae(evaluatedAnomalyRows, "ml_forecast"),
    maeAnomalyAgent: dryRun ? null : mae(evaluatedAnomalyRows, "agent_forecast"),
    maeNonAnomalyMlOnly: mae(nonAnomalyRows, "ml_forecast"),
    adjustedCount: toRun.filter((r) => r.agent_adjusted).length,
  };

  console.log("\nSummary:", JSON.stringify(summary, null, 2));

  const outDir = path.join(process.cwd(), "evals");
  fs.mkdirSync(outDir, { recursive: true });
  fs.writeFileSync(
    path.join(outDir, "report.json"),
    JSON.stringify({ summary, rows: annotated }, null, 2)
  );
  console.log(`\nWrote ${path.join(outDir, "report.json")}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
