import { NextRequest, NextResponse } from "next/server";
import { getStores, getSkus, saveRecommendation } from "../../../lib/db";
import { runAgent } from "../../../lib/agent";

// Very small in-memory rate limit: fine for a single-instance demo deploy,
// not for a multi-instance production setup (use a shared store like Redis
// there — called out in the README).
const RATE_LIMIT = 10; // requests per IP per window
const WINDOW_MS = 10 * 60 * 1000;
const hits = new Map<string, { count: number; resetAt: number }>();

function checkRateLimit(ip: string): boolean {
  const now = Date.now();
  const entry = hits.get(ip);
  if (!entry || now > entry.resetAt) {
    hits.set(ip, { count: 1, resetAt: now + WINDOW_MS });
    return true;
  }
  if (entry.count >= RATE_LIMIT) return false;
  entry.count += 1;
  return true;
}

export async function POST(req: NextRequest) {
  const ip = req.headers.get("x-forwarded-for") || "local";
  if (!checkRateLimit(ip)) {
    return NextResponse.json(
      { error: "Rate limit exceeded for this demo. Try again later." },
      { status: 429 }
    );
  }

  let body: { storeId?: string; skuId?: string; date?: string; mlForecast?: number };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "invalid JSON body" }, { status: 400 });
  }

  const { storeId, skuId, date, mlForecast } = body;
  if (!storeId || !skuId || !date || typeof mlForecast !== "number") {
    return NextResponse.json(
      { error: "storeId, skuId, date, mlForecast are all required" },
      { status: 400 }
    );
  }

  const store = getStores().find((s) => s.id === storeId);
  const sku = getSkus().find((s) => s.id === skuId);
  if (!store || !sku) {
    return NextResponse.json({ error: "unknown storeId or skuId" }, { status: 404 });
  }

  try {
    const rec = await runAgent({ store, sku, date, mlForecast });
    saveRecommendation({
      store_id: storeId,
      sku_id: skuId,
      date,
      ml_forecast: mlForecast,
      agent_forecast: rec.agent_forecast,
      confidence: rec.confidence,
      adjusted: rec.adjusted ? 1 : 0,
      rationale: rec.rationale,
    });
    return NextResponse.json(rec);
  } catch (err) {
    const message = err instanceof Error ? err.message : "agent failed";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
