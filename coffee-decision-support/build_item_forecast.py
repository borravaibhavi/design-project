"""
build_item_forecast.py

Adds a SECOND, independent real-data forecast: a genuine 14-day seasonal-naive
forecast for finished menu items (Latte, Drip coffee, Scone, ...), built from
the Maven Roasters "Coffee Shop Sales" Kaggle dataset
(https://www.kaggle.com/datasets/ahmedabbas757/coffee-sales), which has 181
days of real daily transactions (2023-01-01 to 2023-06-30) across 3 NYC
locations -- enough history for pipeline.py's forecast_demand() to actually
show weekly seasonality, unlike the 6-day viramatv order window.

This is DELIBERATELY NOT reconciled with the rest of the pipeline: it's a
different coffee shop, a different time period, and product-level (not
ingredient-level) data. It does not feed EOQ, reorder points, supplier
selection, or the LLM recommendations -- those stay on the real
ingredient/recipe/inventory data from viramatv/coffee-shop-data. This script
only produces a second forecast panel so the dashboard has one real chart
that actually shows shape, clearly labeled as a separate source.

Reuses pipeline.py's forecast_demand() directly (same seasonal-naive method,
same confidence-band math) so both forecast panels are methodologically
identical -- only the amount of real history behind them differs.
"""

import json
from pathlib import Path

import pandas as pd

from pipeline import forecast_demand

RAW_PATH = Path(__file__).parent / "maven_raw" / "Coffee Shop Sales.xlsx"
OUT_PATH = Path(__file__).parent / "data" / "item_forecasts.json"


def main():
    df = pd.read_excel(RAW_PATH, sheet_name="Transactions")

    # aggregate across all 3 Maven Roasters locations -- a different shop
    # entirely from the ingredient-level data, deliberately not blended with it
    daily = (
        df.groupby([df["transaction_date"].dt.date, "product_type"])["transaction_qty"]
        .sum()
        .reset_index()
        .rename(columns={"transaction_date": "date", "transaction_qty": "units_sold_clean"})
    )

    forecasts = {}
    for product_type, g in daily.groupby("product_type"):
        f, _, _ = forecast_demand(g, horizon=14)
        forecasts[product_type] = f

    payload = {
        "source": "Maven Roasters \"Coffee Shop Sales\" (Kaggle: ahmedabbas757/coffee-sales)",
        "note": "Different shop, different time period from the ingredient/inventory data -- "
                "product-level only, not used for EOQ/reorder/supplier logic.",
        "date_range": f"{daily['date'].min()} to {daily['date'].max()}",
        "n_days": int(daily["date"].nunique()),
        "product_types": sorted(forecasts.keys()),
        "forecasts": forecasts,
    }

    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[done] wrote {OUT_PATH} -- {payload['n_days']} real days, "
          f"{len(payload['product_types'])} product types, {payload['date_range']}")


if __name__ == "__main__":
    main()
