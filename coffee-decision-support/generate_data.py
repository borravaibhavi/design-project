"""
generate_data.py

Generates synthetic historical data for a coffee shop supply chain,
standing in until a real data partner is found. Structure is modeled
after real public datasets (Kaggle coffee-sales datasets, the TTB
brewery-materials dataset, and generic supply-chain inventory datasets)
so it can be swapped for real data later with minimal changes to the
pipeline downstream.

Outputs (all under ./data/):
  - daily_demand.csv     : date, sku, units_sold, on_hand, stockout_flag
  - suppliers.csv        : sku, supplier, price_per_unit, lead_time_days,
                            reliability, moq, availability
  - skus.csv             : sku, unit, holding_cost_per_unit_day,
                            ordering_cost, safety_stock_days

Explicitly synthetic. Seeded for reproducibility. Includes deliberate
outliers/missing values so the cleaning step in the pipeline has
something real to do.
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)

START_DATE = "2021-01-01"
END_DATE = "2026-08-31"
dates = pd.date_range(START_DATE, END_DATE, freq="D")
n_days = len(dates)

# ---------------------------------------------------------------------
# 1. SKUs — the raw materials a coffee shop actually reorders
# ---------------------------------------------------------------------
SKUS = {
    "Espresso Beans (Arabica)":   {"unit": "lb",   "base_demand": 18, "weekly_amp": 0.20, "annual_amp": 0.10, "holding_cost": 0.06, "ordering_cost": 40, "safety_days": 5},
    "Whole Milk":                 {"unit": "gal",  "base_demand": 22, "weekly_amp": 0.30, "annual_amp": 0.05, "holding_cost": 0.10, "ordering_cost": 25, "safety_days": 2},
    "Oat Milk":                   {"unit": "gal",  "base_demand": 9,  "weekly_amp": 0.25, "annual_amp": 0.15, "holding_cost": 0.12, "ordering_cost": 25, "safety_days": 3},
    "Paper Cups (12oz)":          {"unit": "case", "base_demand": 6,  "weekly_amp": 0.35, "annual_amp": 0.20, "holding_cost": 0.03, "ordering_cost": 15, "safety_days": 7},
    "Vanilla Syrup":              {"unit": "btl",  "base_demand": 4,  "weekly_amp": 0.15, "annual_amp": 0.30, "holding_cost": 0.04, "ordering_cost": 15, "safety_days": 6},
    "Sugar":                      {"unit": "lb",   "base_demand": 7,  "weekly_amp": 0.10, "annual_amp": 0.05, "holding_cost": 0.02, "ordering_cost": 10, "safety_days": 10},
}

SUPPLIERS = {
    "Espresso Beans (Arabica)": [
        {"supplier": "Highland Roasters Co-op", "price": 6.75, "lead_time": 6, "reliability": 0.95, "moq": 25, "availability": "In stock"},
        {"supplier": "Continental Green Coffee", "price": 6.40, "lead_time": 9, "reliability": 0.90, "moq": 50, "availability": "In stock"},
        {"supplier": "Direct Trade Partners", "price": 7.20, "lead_time": 4, "reliability": 0.98, "moq": 10, "availability": "Limited"},
    ],
    "Whole Milk": [
        {"supplier": "Garden State Dairy", "price": 3.10, "lead_time": 1, "reliability": 0.99, "moq": 5, "availability": "In stock"},
        {"supplier": "Regional Dairy Co-op", "price": 2.95, "lead_time": 2, "reliability": 0.94, "moq": 10, "availability": "In stock"},
    ],
    "Oat Milk": [
        {"supplier": "PlantBase Foods", "price": 4.50, "lead_time": 3, "reliability": 0.93, "moq": 5, "availability": "In stock"},
        {"supplier": "Regional Dairy Co-op", "price": 4.80, "lead_time": 2, "reliability": 0.96, "moq": 5, "availability": "Limited"},
    ],
    "Paper Cups (12oz)": [
        {"supplier": "EcoPack Supply", "price": 32.00, "lead_time": 8, "reliability": 0.92, "moq": 3, "availability": "In stock"},
        {"supplier": "Metro Foodservice", "price": 35.50, "lead_time": 4, "reliability": 0.97, "moq": 2, "availability": "In stock"},
    ],
    "Vanilla Syrup": [
        {"supplier": "Torani Distribution", "price": 9.25, "lead_time": 5, "reliability": 0.96, "moq": 6, "availability": "In stock"},
        {"supplier": "Metro Foodservice", "price": 9.90, "lead_time": 4, "reliability": 0.97, "moq": 4, "availability": "In stock"},
    ],
    "Sugar": [
        {"supplier": "Regional Dairy Co-op", "price": 0.85, "lead_time": 3, "reliability": 0.98, "moq": 20, "availability": "In stock"},
        {"supplier": "Metro Foodservice", "price": 0.90, "lead_time": 4, "reliability": 0.97, "moq": 10, "availability": "In stock"},
    ],
}


def seasonal_multiplier(day_of_year, annual_amp):
    # Peaks in winter (hot drinks) and back-to-school/holiday season;
    # dips in mid-summer. Purely stylistic — tune freely.
    return 1 + annual_amp * np.cos(2 * np.pi * (day_of_year - 15) / 365.25)


def weekly_multiplier(dow, weekly_amp):
    # Higher on weekday mornings (commuters), a bump on weekends for cafes
    weekday_curve = [1.05, 1.0, 1.0, 1.02, 1.15, 1.25, 1.10]  # Mon..Sun
    base = weekday_curve[dow]
    return 1 + weekly_amp * (base - 1)


records = []
for sku, cfg in SKUS.items():
    on_hand = cfg["base_demand"] * cfg["safety_days"] * 2  # starting stock
    for i, d in enumerate(dates):
        seasonal = seasonal_multiplier(d.dayofyear, cfg["annual_amp"])
        weekly = weekly_multiplier(d.weekday(), cfg["weekly_amp"])
        noise = rng.normal(1.0, 0.12)
        demand = max(0, cfg["base_demand"] * seasonal * weekly * noise)

        # deliberate injected data-quality issues, ~1.5% of rows
        if rng.random() < 0.008:
            demand *= rng.choice([0, 4.0])  # dropout or spike
        if rng.random() < 0.005:
            demand = np.nan  # missing read

        units_sold = 0 if np.isnan(demand) else round(demand)
        on_hand -= units_sold
        stockout = on_hand < 0
        if stockout:
            on_hand = 0

        # simple periodic replenishment so on_hand doesn't just drain to zero
        reorder_point = cfg["base_demand"] * cfg["safety_days"]
        if on_hand < reorder_point and i % 3 == 0:
            on_hand += cfg["base_demand"] * cfg["safety_days"] * 2.5

        records.append({
            "date": d.date().isoformat(),
            "sku": sku,
            "units_sold": units_sold if not np.isnan(demand) else np.nan,
            "on_hand": round(on_hand, 1),
            "stockout_flag": stockout,
        })

demand_df = pd.DataFrame(records)

# For demo purposes, force the final week for two SKUs into a genuine
# low-stock situation (busy week, no restock yet) so the dashboard has
# something live to recommend against, rather than everything sitting
# comfortably at "OK" on the last day by coincidence of the replenishment
# cycle above.
LOW_STOCK_DEMO_SKUS = {"Oat Milk": 6.0, "Vanilla Syrup": 8.0}
for sku, forced_on_hand in LOW_STOCK_DEMO_SKUS.items():
    mask = demand_df["sku"] == sku
    last_idx = demand_df[mask].index[-1]
    demand_df.loc[last_idx, "on_hand"] = forced_on_hand

demand_df.to_csv(OUT_DIR / "daily_demand.csv", index=False)

supplier_rows = []
for sku, sup_list in SUPPLIERS.items():
    for s in sup_list:
        supplier_rows.append({"sku": sku, **s})
suppliers_df = pd.DataFrame(supplier_rows)
suppliers_df.to_csv(OUT_DIR / "suppliers.csv", index=False)

sku_rows = []
for sku, cfg in SKUS.items():
    sku_rows.append({
        "sku": sku,
        "unit": cfg["unit"],
        "holding_cost_per_unit_day": cfg["holding_cost"],
        "ordering_cost": cfg["ordering_cost"],
        "safety_stock_days": cfg["safety_days"],
    })
skus_df = pd.DataFrame(sku_rows)
skus_df.to_csv(OUT_DIR / "skus.csv", index=False)

print(f"Generated {n_days} days x {len(SKUS)} SKUs -> {len(demand_df)} demand rows")
print(f"Missing values injected: {demand_df['units_sold'].isna().sum()}")
print(f"Stockout days: {int(demand_df['stockout_flag'].sum())}")
print("Files written to:", OUT_DIR)
