"""
Synthetic data generator for the dark-store replenishment agent.

Everything here is synthetic — there is no real quick-commerce dataset behind
this. The generator creates:
  - stores, skus              (dimension tables)
  - sales                     (daily units sold per store/SKU, with weekly
                               seasonality + a handful of injected "anomaly"
                               days tied to synthetic real-world events)
  - context_docs               (short text blurbs — festival/weather/local
                               event/competitor-promo notes — the RAG corpus)
  - stock                      (current on-hand units per store/SKU)
  - incoming_pos                (purchase orders in transit, with an ETA)

The last TEST_DAYS days of the sales history are held out as a test window
(see train_model.py / scripts/eval). Anomaly events are deliberately placed
so some land inside the test window and some inside the training window,
so the eval can compare ML-only vs agent-adjusted error specifically on
anomaly days the base model never saw a labeled cause for.

Usage:
    python ml/generate_data.py
"""
import random
import sqlite3
import datetime
from pathlib import Path

SEED = 42
random.seed(SEED)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "darkstore.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

HISTORY_DAYS = 120
TEST_DAYS = 14          # last N days held out for eval / "current" forecast window
FORECAST_HORIZON = 7    # days the model forecasts ahead

STORES = [
    {"id": "DS-BLR-01", "name": "Indiranagar", "city": "Bengaluru", "region": "south"},
    {"id": "DS-DEL-01", "name": "Hauz Khas", "city": "Delhi", "region": "north"},
    {"id": "DS-PUN-01", "name": "Baner", "city": "Pune", "region": "west"},
]

SKUS = [
    {"id": "SKU-MILK-500", "name": "Toned Milk 500ml", "category": "dairy", "base_demand": 60},
    {"id": "SKU-BREAD-WHT", "name": "White Bread 400g", "category": "bakery", "base_demand": 35},
    {"id": "SKU-EGG-12", "name": "Eggs (12 pack)", "category": "dairy", "base_demand": 28},
    {"id": "SKU-ICE-TUB1L", "name": "Ice Cream Tub 1L", "category": "frozen", "base_demand": 15},
    {"id": "SKU-WATER-1L", "name": "Packaged Water 1L", "category": "beverages", "base_demand": 50},
    {"id": "SKU-CHIPS-LAYS", "name": "Potato Chips 90g", "category": "snacks", "base_demand": 40},
    {"id": "SKU-SODA-COLA", "name": "Cola 750ml", "category": "beverages", "base_demand": 32},
    {"id": "SKU-DIAPER-M", "name": "Diapers Medium (pack)", "category": "baby", "base_demand": 10},
]

# Synthetic anomaly events: (date_offset_from_today's start, region or "all", sku category or "all",
# event description, demand multiplier)
# date_offset is negative = days before the generation date (today), so these are historical.
TODAY = datetime.date.today()


def d(days_ago):
    return TODAY - datetime.timedelta(days=days_ago)


EVENTS = [
    {"date": d(100), "region": "all", "category": "all",
     "title": "Diwali week", "multiplier": 2.2,
     "text": "Diwali falls this week across India. Expect a sharp, broad spike in grocery and snack demand as households stock up for gatherings and gifting."},
    {"date": d(63), "region": "south", "category": "beverages",
     "title": "Heatwave in Bengaluru", "multiplier": 1.8,
     "text": "IMD heatwave advisory for Bengaluru this week; daytime temperatures 4-5C above normal. Cold beverage and water demand typically spikes during heatwaves."},
    {"date": d(40), "region": "north", "category": "snacks",
     "title": "IPL knockout match, Delhi", "multiplier": 1.6,
     "text": "High-profile cricket match being watched widely in Delhi NCR tonight. Snack and soft-drink demand usually rises sharply during big match nights."},
    {"date": d(21), "region": "west", "category": "dairy",
     "title": "Local festival, Pune", "multiplier": 1.7,
     "text": "A regional festival is being celebrated in Pune this week, with increased home cooking and dairy consumption reported in prior years."},
    {"date": d(9), "region": "all", "category": "all",
     "title": "Competitor outage", "multiplier": 1.4,
     "text": "A major competitor dark-store chain is reporting app outages and stock-outs across metro cities today; some overflow demand is being observed at nearby stores."},
    {"date": d(3), "region": "south", "category": "frozen",
     "title": "Unseasonal heat, Bengaluru", "multiplier": 1.9,
     "text": "Bengaluru is seeing an unseasonal heat spell this week. Ice cream and frozen dessert sales tend to rise sharply in similar past conditions."},
]


def seasonal_factor(dow):
    # Weekends (Fri/Sat/Sun) a bit busier than weekdays.
    return {0: 0.95, 1: 0.9, 2: 0.95, 3: 1.0, 4: 1.15, 5: 1.3, 6: 1.1}[dow]


def event_for(day, store, sku):
    for ev in EVENTS:
        if ev["date"] != day:
            continue
        region_match = ev["region"] == "all" or ev["region"] == store["region"]
        cat_match = ev["category"] == "all" or ev["category"] == sku["category"]
        if region_match and cat_match:
            return ev
    return None


def build_schema(conn):
    conn.executescript(
        """
        DROP TABLE IF EXISTS stores;
        DROP TABLE IF EXISTS skus;
        DROP TABLE IF EXISTS sales;
        DROP TABLE IF EXISTS context_docs;
        DROP TABLE IF EXISTS stock;
        DROP TABLE IF EXISTS incoming_pos;
        DROP TABLE IF EXISTS forecasts;
        DROP TABLE IF EXISTS recommendations;

        CREATE TABLE stores (
            id TEXT PRIMARY KEY, name TEXT, city TEXT, region TEXT
        );
        CREATE TABLE skus (
            id TEXT PRIMARY KEY, name TEXT, category TEXT, base_demand INTEGER
        );
        CREATE TABLE sales (
            store_id TEXT, sku_id TEXT, date TEXT, units_sold INTEGER,
            PRIMARY KEY (store_id, sku_id, date)
        );
        CREATE TABLE context_docs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT, region TEXT, category TEXT, title TEXT, body TEXT
        );
        CREATE TABLE stock (
            store_id TEXT, sku_id TEXT, on_hand INTEGER,
            PRIMARY KEY (store_id, sku_id)
        );
        CREATE TABLE incoming_pos (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            store_id TEXT, sku_id TEXT, quantity INTEGER, eta_date TEXT
        );
        -- forecasts/recommendations are populated by train_model.py and the
        -- Node agent respectively; declared here so the schema is all in one place.
        CREATE TABLE forecasts (
            store_id TEXT, sku_id TEXT, date TEXT, ml_forecast REAL,
            PRIMARY KEY (store_id, sku_id, date)
        );
        CREATE TABLE recommendations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            store_id TEXT, sku_id TEXT, date TEXT,
            ml_forecast REAL, agent_forecast REAL, confidence REAL,
            adjusted INTEGER, rationale TEXT, created_at TEXT
        );
        """
    )


def main():
    conn = sqlite3.connect(DB_PATH)
    build_schema(conn)

    for s in STORES:
        conn.execute("INSERT INTO stores VALUES (?,?,?,?)", (s["id"], s["name"], s["city"], s["region"]))
    for sku in SKUS:
        conn.execute("INSERT INTO skus VALUES (?,?,?,?)", (sku["id"], sku["name"], sku["category"], sku["base_demand"]))

    for i in range(HISTORY_DAYS, -1, -1):
        day = d(i)
        dow = day.weekday()
        for store in STORES:
            for sku in SKUS:
                base = sku["base_demand"] * seasonal_factor(dow)
                noise = random.gauss(0, base * 0.08)
                ev = event_for(day, store, sku)
                mult = ev["multiplier"] if ev else 1.0
                units = max(0, round(base * mult + noise))
                conn.execute(
                    "INSERT INTO sales VALUES (?,?,?,?)",
                    (store["id"], sku["id"], day.isoformat(), units),
                )

    # One context_docs row per event (not per store/SKU match) — this is the RAG corpus.
    for ev in EVENTS:
        conn.execute(
            "INSERT INTO context_docs (date, region, category, title, body) VALUES (?,?,?,?,?)",
            (ev["date"].isoformat(), ev["region"], ev["category"], ev["title"], ev["text"]),
        )

    # Current stock + one incoming PO per store/SKU, randomly timed.
    for store in STORES:
        for sku in SKUS:
            on_hand = random.randint(0, int(sku["base_demand"] * 1.5))
            conn.execute("INSERT INTO stock VALUES (?,?,?)", (store["id"], sku["id"], on_hand))
            if random.random() < 0.6:
                eta = TODAY + datetime.timedelta(days=random.randint(1, 5))
                qty = random.randint(int(sku["base_demand"] * 0.5), sku["base_demand"] * 2)
                conn.execute(
                    "INSERT INTO incoming_pos (store_id, sku_id, quantity, eta_date) VALUES (?,?,?,?)",
                    (store["id"], sku["id"], qty, eta.isoformat()),
                )

    conn.commit()
    n_sales = conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0]
    n_docs = conn.execute("SELECT COUNT(*) FROM context_docs").fetchone()[0]
    conn.close()
    print(f"Wrote {DB_PATH}")
    print(f"  stores={len(STORES)} skus={len(SKUS)} sales_rows={n_sales} context_docs={n_docs}")
    print(f"  history: {d(HISTORY_DAYS)} .. {TODAY}  (test window: last {TEST_DAYS} days)")


if __name__ == "__main__":
    main()
