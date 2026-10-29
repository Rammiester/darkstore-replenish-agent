"""
Trains a per-store/SKU demand forecaster and writes forecasts back to SQLite.

Model: gradient boosting regression (scikit-learn) on lag + calendar features,
trained on all days EXCEPT the last TEST_DAYS (see generate_data.py). This is
deliberately a single global model (not one model per SKU) with store/SKU
identity as categorical features — simpler to train well on a small synthetic
dataset than 24 separate tiny per-series models, while still being able to
learn per-series baselines via the ohe-encoded identity features.

The model has no knowledge of `context_docs` (the RAG corpus) at all — that's
the point. It only ever sees historical sales + calendar features, so it is
systematically blind to the anomaly events, which is exactly the gap the
Node agent (lib/agent.ts) is built to close using retrieval + tool calls.

Usage:
    python ml/train_model.py
"""
import sqlite3
import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "darkstore.db"
TEST_DAYS = 14
LAGS = [1, 7, 14]


def load_sales(conn):
    df = pd.read_sql_query("SELECT * FROM sales", conn, parse_dates=["date"])
    return df.sort_values(["store_id", "sku_id", "date"])


def build_features(df):
    df = df.copy()
    for lag in LAGS:
        df[f"lag_{lag}"] = df.groupby(["store_id", "sku_id"])["units_sold"].shift(lag)
    df["dow"] = df["date"].dt.day_of_week
    df["rolling_mean_7"] = df.groupby(["store_id", "sku_id"])["units_sold"].transform(
        lambda s: s.shift(1).rolling(7).mean()
    )
    return df.dropna()


def main():
    conn = sqlite3.connect(DB_PATH)
    sales = load_sales(conn)
    feats = build_features(sales)

    max_date = feats["date"].max()
    cutoff = max_date - pd.Timedelta(days=TEST_DAYS)
    train = feats[feats["date"] <= cutoff]
    test = feats[feats["date"] > cutoff]

    feature_cols = ["store_id", "sku_id", "dow"] + [f"lag_{l}" for l in LAGS] + ["rolling_mean_7"]
    X_train, y_train = train[feature_cols], train["units_sold"]
    X_test, y_test = test[feature_cols], test["units_sold"]

    pre = ColumnTransformer(
        [("cat", OneHotEncoder(handle_unknown="ignore"), ["store_id", "sku_id", "dow"])],
        remainder="passthrough",
    )
    model = Pipeline([
        ("pre", pre),
        ("gbr", GradientBoostingRegressor(random_state=42, n_estimators=200, max_depth=3)),
    ])
    model.fit(X_train, y_train)

    test = test.copy()
    test["ml_forecast"] = np.clip(model.predict(X_test), 0, None)
    mae = mean_absolute_error(y_test, test["ml_forecast"])
    print(f"Trained on {len(train)} rows, evaluated on {len(test)} held-out rows. Test MAE: {mae:.2f} units/day")

    conn.execute("DELETE FROM forecasts")
    conn.executemany(
        "INSERT INTO forecasts (store_id, sku_id, date, ml_forecast) VALUES (?,?,?,?)",
        [
            (r.store_id, r.sku_id, r.date.date().isoformat(), float(r.ml_forecast))
            for r in test.itertuples()
        ],
    )
    conn.commit()
    conn.close()
    print(f"Wrote {len(test)} forecast rows to {DB_PATH} (forecasts table).")


if __name__ == "__main__":
    main()
