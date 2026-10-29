# Dark-Store Replenishment Agent

A hybrid **ML + LLM-agent** demand-replenishment system for quick-commerce dark stores
(Blinkit/Zepto/Instamart-style). Not another forecasting notebook — the point is the
architecture: a classical ML model gives a quantitative baseline forecast; a Claude agent
with retrieval and tool calling catches what the model is structurally blind to, and
explains itself.

**Everything here is synthetic data**, clearly disclosed — there is no real dark-store
dataset behind this (see [Data](#data)).

## The gap this demonstrates

A gradient-boosting model trained on 120 days of synthetic sales history forecasts demand
for the next period using lags, rolling averages, and day-of-week. It has **no way to know**
that a festival, a heatwave, or a competitor's app outage is about to spike demand — it only
knows what happened historically.

Running the eval's dry-run (`npm run eval -- --dry-run`, no API key needed, included as
[`evals/report.json`](evals/report.json)) on the held-out test window:

| | Mean absolute error (units/day) |
|---|---|
| ML-only forecast, **normal** days | 2.72 |
| ML-only forecast, **anomaly** days (festival/heatwave/competitor-outage) | **9.88** |

The model is **3.6x worse** on exactly the days a human planner would have seen coming. That
gap — not "can an LLM forecast demand" — is what the agent layer is built to close: retrieve
the relevant context, check live stock and incoming purchase orders, and decide whether (and
how much) to adjust the ML number, with a stated rationale and confidence.

Running `npm run eval` (requires `ANTHROPIC_API_KEY`, makes real API calls, not included in
the repo since it costs money) extends that same report with the agent-adjusted error on
those same anomaly days, so the two numbers are directly comparable.

## Architecture

```
synthetic data (Python) ──> SQLite ──> ML forecast (Python, scikit-learn)
                                │
                                ├──> Next.js dashboard (reads forecasts)
                                │
                                └──> Claude agent (Node/TypeScript)
                                       ├─ retrieval: festival/weather/competitor
                                       │   context docs, keyword-scored (see
                                       │   "Why not embeddings" below)
                                       ├─ tool calls: get_current_stock,
                                       │   get_incoming_purchase_orders
                                       └─ submit_recommendation (structured,
                                           strict-schema tool call — not
                                           free-text JSON parsing)
```

- **`ml/`** — Python. `generate_data.py` builds the synthetic dataset (stores, SKUs, 120 days
  of sales with injected anomaly events) into `data/darkstore.db`. `train_model.py` trains a
  per-store/SKU gradient-boosting forecaster (lag + calendar features) on everything except
  the last 14 days, and writes held-out forecasts back to the same SQLite file.
- **`lib/agent.ts`** — the Claude agent. A manual tool-use loop (not the beta tool runner,
  so there's no beta-dependency and the loop is easy to read end-to-end): retrieves context,
  calls tools, and must finish by calling `submit_recommendation` — a `strict: true` tool,
  so the final output is schema-validated by the API itself rather than parsed out of free text.
- **`lib/retrieval.ts`** — keyword/region/category-scored retrieval over `context_docs`.
- **`app/`** — Next.js dashboard (`/`) and eval report viewer (`/evals`); `app/api/recommend`
  is the one API route, with a small in-memory per-IP rate limit.
- **`scripts/eval.ts`** — the eval harness described above.

### Why SQLite, not Postgres + pgvector

This is a from-scratch demo meant to run with zero external infra after `npm install` — no
Postgres instance to provision, no connection string to manage. At real scale, the two things
that would change are: the RAG corpus would move to Postgres + pgvector with real embeddings
(see below), and the structured tables would move to whatever OLTP store the rest of the
stack already uses. Nothing in `lib/agent.ts` or the API route would need to change — only
`lib/db.ts` and `lib/retrieval.ts`.

### Why not real embeddings for retrieval

The RAG corpus here is six short event blurbs. Keyword + region/category scoring
(`lib/retrieval.ts`) finds the right one every time at this scale, and a vector index would be
pure overhead. At hundreds-plus of documents, swap in Postgres + pgvector and an embeddings
model — the call site doesn't change.

### Guardrails

- The agent's system prompt explicitly marks retrieved context as **data, not instructions**
  — it comes from an internal notes database and is never trusted as commands, regardless of
  what its text says.
- Tool-call loop is capped at 6 rounds per request.
- The final answer must come through the `submit_recommendation` tool (schema-validated), not
  parsed out of free text.
- `/api/recommend` has a small in-memory per-IP rate limit (10 requests / 10 minutes) — fine
  for a single-instance demo deploy, not for a multi-instance production one (use a shared
  store like Redis there).

## Data

Fully synthetic, seeded for reproducibility (`ml/generate_data.py`, seed 42): 3 stores
(Bengaluru/Delhi/Pune), 8 SKUs, 120 days of history with weekly seasonality and 6 injected
anomaly events (Diwali, a heatwave, an IPL match night, a local festival, a competitor
outage, an unseasonal heat spell) tied to specific store regions and SKU categories. The last
14 days are held out as the test window used for forecasting and evals.

## Running it

Requires Node 22.5+ (uses the built-in `node:sqlite` module — see note below) and Python 3.10+.

```bash
# 1. Generate synthetic data and train the model
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r ml/requirements.txt
python ml/generate_data.py
python ml/train_model.py

# 2. Run the app (data/darkstore.db is committed, so this works without step 1 too)
npm install
npm run dev
# -> http://localhost:3000  (dashboard)
# -> http://localhost:3000/evals  (eval report)

# 3. To exercise the agent, set an API key first
cp .env.example .env.local   # fill in ANTHROPIC_API_KEY
npm run eval -- --dry-run    # free: sanity-checks anomaly-day detection, no API calls
npm run eval                 # real: makes API calls, costs money
```

### Why `node:sqlite` instead of `better-sqlite3`

`better-sqlite3` needs a native (node-gyp) build step, which needs a C++ toolchain installed
— an easy thing for a recruiter cloning this to get stuck on. Node's built-in `node:sqlite`
(stable enough for a demo as of Node 22.5+, though still flagged experimental upstream) needs
nothing beyond Node itself.

## Deploying

Frontend + API routes deploy cleanly to Vercel (`vercel --prod` from this directory, or
drag-and-drop at vercel.com/new) — no build config needed beyond the committed `data/darkstore.db`
and an `ANTHROPIC_API_KEY` environment variable set in the Vercel project settings. The
in-memory rate limiter resets per cold start / per instance, which is a known limitation
called out above, not a bug.

## What's verified vs. not

- Synthetic data generation, ML training, the dashboard, the API route's validation/rate-limit/
  error paths, and the eval harness's anomaly-detection + MAE logic are all run and verified in
  this repo (`npm run build` passes, `npm run eval -- --dry-run` produces the numbers quoted
  above).
- The actual Claude agent calls (`lib/agent.ts`, the non-dry-run eval, the dashboard's "Get
  agent recommendation" button) require `ANTHROPIC_API_KEY` and have **not** been exercised
  against the live API while building this — there was no key available in the environment
  this was built in. The error path (missing key) was verified to fail gracefully rather than
  crash. Run `npm run eval` yourself to get the real agent-adjusted numbers for the headline
  comparison above.
