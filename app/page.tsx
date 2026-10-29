import { getStores, getSkus, getLatestForecasts } from "../lib/db";
import RecommendCell from "./RecommendCell";

export const dynamic = "force-dynamic";

export default function DashboardPage() {
  const stores = getStores();
  const skus = getSkus();
  const forecasts = getLatestForecasts();

  const storeById = Object.fromEntries(stores.map((s) => [s.id, s]));
  const skuById = Object.fromEntries(skus.map((s) => [s.id, s]));

  const rows = forecasts
    .map((f) => ({
      ...f,
      store: storeById[f.store_id],
      sku: skuById[f.sku_id],
    }))
    .filter((r) => r.store && r.sku)
    .sort((a, b) =>
      a.store.name === b.store.name
        ? a.sku.name.localeCompare(b.sku.name)
        : a.store.name.localeCompare(b.store.name)
    );

  return (
    <div className="wrap">
      <div className="hgroup">
        <h1>Dark-Store Replenishment Agent</h1>
        <p>
          A gradient-boosting model forecasts baseline demand from sales history alone — it has
          no idea a festival, heatwave, or competitor outage is coming. The agent column lets a
          Claude agent retrieve that context, check live stock/incoming POs, and decide whether to
          adjust the number. Synthetic data — see README.
        </p>
      </div>
      <nav className="tabs">
        <a className="active" href="/">Dashboard</a>
        <a href="/evals">Evals</a>
      </nav>
      <table>
        <thead>
          <tr>
            <th>Store</th>
            <th>SKU</th>
            <th>Date</th>
            <th>ML forecast</th>
            <th>Agent recommendation</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={`${r.store_id}-${r.sku_id}`}>
              <td>{r.store.name}</td>
              <td>{r.sku.name}</td>
              <td>{r.date}</td>
              <td>{r.ml_forecast.toFixed(1)}</td>
              <td>
                <RecommendCell
                  storeId={r.store_id}
                  skuId={r.sku_id}
                  date={r.date}
                  mlForecast={r.ml_forecast}
                />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
