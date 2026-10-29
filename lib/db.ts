import { DatabaseSync } from "node:sqlite";
import path from "path";
import fs from "fs";

// A single shared SQLite connection, via Node's built-in node:sqlite
// (stable-enough-for-a-demo as of Node 22.5+, still flagged experimental —
// chosen specifically to avoid a native-compiled dependency like
// better-sqlite3, which needs a C++ toolchain to install). SQLite (not
// Postgres/pgvector) is used deliberately so the whole project runs with
// zero external infra after `npm install` — see README "Why SQLite" for the
// production swap-in notes (Postgres + pgvector for the RAG corpus at real
// scale).
let _db: DatabaseSync | null = null;

export function getDb(): DatabaseSync {
  if (!_db) {
    const dbPath = path.join(process.cwd(), "data", "darkstore.db");
    if (!fs.existsSync(dbPath)) {
      throw new Error(
        `${dbPath} does not exist. Run "python ml/generate_data.py && python ml/train_model.py" first (see README).`
      );
    }
    _db = new DatabaseSync(dbPath);
    _db.exec("PRAGMA journal_mode = WAL");
  }
  return _db;
}

export interface StoreRow {
  id: string;
  name: string;
  city: string;
  region: string;
}

export interface SkuRow {
  id: string;
  name: string;
  category: string;
  base_demand: number;
}

export interface ForecastRow {
  store_id: string;
  sku_id: string;
  date: string;
  ml_forecast: number;
}

export interface StockRow {
  store_id: string;
  sku_id: string;
  on_hand: number;
}

export interface IncomingPoRow {
  id: number;
  store_id: string;
  sku_id: string;
  quantity: number;
  eta_date: string;
}

export interface ContextDocRow {
  id: number;
  date: string;
  region: string;
  category: string;
  title: string;
  body: string;
}

export interface RecommendationRow {
  id: number;
  store_id: string;
  sku_id: string;
  date: string;
  ml_forecast: number;
  agent_forecast: number;
  confidence: number;
  adjusted: number;
  rationale: string;
  created_at: string;
}

export function getStores(): StoreRow[] {
  return getDb().prepare("SELECT * FROM stores ORDER BY name").all() as unknown as StoreRow[];
}

export function getSkus(): SkuRow[] {
  return getDb().prepare("SELECT * FROM skus ORDER BY name").all() as unknown as SkuRow[];
}

/** Latest forecast per store/SKU (the most recent date in the forecasts table). */
export function getLatestForecasts(): ForecastRow[] {
  return getDb()
    .prepare(
      `SELECT f.* FROM forecasts f
       INNER JOIN (
         SELECT store_id, sku_id, MAX(date) AS max_date FROM forecasts GROUP BY store_id, sku_id
       ) latest ON f.store_id = latest.store_id AND f.sku_id = latest.sku_id AND f.date = latest.max_date`
    )
    .all() as unknown as ForecastRow[];
}

export function getStock(storeId: string, skuId: string): StockRow | undefined {
  return getDb()
    .prepare("SELECT * FROM stock WHERE store_id = ? AND sku_id = ?")
    .get(storeId, skuId) as unknown as StockRow | undefined;
}

export function getIncomingPos(storeId: string, skuId: string): IncomingPoRow[] {
  return getDb()
    .prepare("SELECT * FROM incoming_pos WHERE store_id = ? AND sku_id = ? ORDER BY eta_date")
    .all(storeId, skuId) as unknown as IncomingPoRow[];
}

export function getContextDocs(): ContextDocRow[] {
  return getDb().prepare("SELECT * FROM context_docs").all() as unknown as ContextDocRow[];
}

export function getForecastRow(
  storeId: string,
  skuId: string,
  date: string
): ForecastRow | undefined {
  return getDb()
    .prepare("SELECT * FROM forecasts WHERE store_id = ? AND sku_id = ? AND date = ?")
    .get(storeId, skuId, date) as unknown as ForecastRow | undefined;
}

export function saveRecommendation(rec: Omit<RecommendationRow, "id" | "created_at">) {
  getDb()
    .prepare(
      `INSERT INTO recommendations
       (store_id, sku_id, date, ml_forecast, agent_forecast, confidence, adjusted, rationale, created_at)
       VALUES (@store_id, @sku_id, @date, @ml_forecast, @agent_forecast, @confidence, @adjusted, @rationale, @created_at)`
    )
    .run({ ...rec, created_at: new Date().toISOString() });
}
