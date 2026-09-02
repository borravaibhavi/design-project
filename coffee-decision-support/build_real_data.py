"""
build_real_data.py

Builds data/daily_demand.csv, data/skus.csv, and data/suppliers.csv from the
real "Coffee Shop Sales/Inventory/Staff" Kaggle dataset
(https://www.kaggle.com/datasets/viramatv/coffee-shop-data), in the exact
schema pipeline.py already expects. Raw source files (downloaded via
`kaggle datasets download viramatv/coffee-shop-data --unzip`) are read from
./kaggle_raw/.

Pipeline: orders.csv -> items.csv (item_id -> sku) -> recipe.csv (sku ->
ingredient quantities) -> aggregate by date + ingredient. inventory.csv is
used as the STARTING stock snapshot (as instructed) and depleted forward day
by day using the derived consumption, exactly like generate_data.py's
synthetic on_hand simulation.

REAL vs ESTIMATED, flagged inline and in the summary this script prints:
  - REAL: ingredient names/prices, recipe quantities, order volumes/dates,
    starting inventory levels.
  - ESTIMATED (not present in the source data at all):
    skus.csv: holding_cost_per_unit_day, ordering_cost, safety_stock_days
    suppliers.csv: entirely synthetic (source has no multi-supplier data) --
    adapted from generate_data.py's synthetic supplier generator, anchored to
    each ingredient's real unit price.

Known real-data limitations (see the summary this script prints):
  - orders.csv only spans 2024-02-12 to 2024-02-17 (6 calendar days) -- far
    too little history for real seasonality; pipeline.py's forecaster will
    fall back to its flat/no-seasonality branch (it requires 28+ days for
    the seasonal-naive model).
  - 55 of 521 order rows (10.6%) reference item_id values (It025-It028) that
    don't exist in items.csv -- dropped; there's no items/recipe data to
    resolve them to an ingredient.
  - one row references "It0010" (extra zero), an evident typo for "It010" --
    corrected here.
  - every order line has quantity == 1 in the source data (no bulk orders).
  - 2 of the 18 ingredients (Vanilla syrup, Vanilla extract) are never used
    by any recipe in recipe.csv, so no demand can ever be derived for them --
    excluded from the pipeline inputs entirely (see summary).
"""

import numpy as np
import pandas as pd
from pathlib import Path

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

RAW_DIR = Path(__file__).parent / "kaggle_raw"
OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)

# --- ESTIMATED cost parameters -- not present anywhere in the source data ---
ANNUAL_HOLDING_RATE = 0.20   # common inventory-textbook heuristic: ~20%/yr of unit cost
FLAT_ORDERING_COST = 12.0    # flat $/order placeholder (source has no fixed-cost-per-order data)
FLAT_SAFETY_STOCK_DAYS = 3   # flat placeholder (source has no lead-time-variability data)

# grams/ml -> kg/L so quantities and $/unit read at a normal cafe scale
# (recipe/inventory data is natively in grams and ml, which produces
# unreadably large EOQ/on-hand numbers if left unscaled)
DISPLAY_SCALE = {"grams": (0.001, "kg"), "ml": (0.001, "L"), "units": (1.0, "unit")}


def load_raw():
    ingredients = pd.read_csv(RAW_DIR / "ingredients.csv")
    inventory = pd.read_csv(RAW_DIR / "inventory.csv")
    items = pd.read_csv(RAW_DIR / "items.csv")
    orders = pd.read_csv(RAW_DIR / "orders.csv")
    recipe = pd.read_csv(RAW_DIR / "recipe.csv")
    return ingredients, inventory, items, orders, recipe


def clean_orders(orders, items):
    orders = orders.copy()
    orders.loc[orders["item_id"] == "It0010", "item_id"] = "It010"  # obvious source typo

    known = set(items["item_id"])
    bad_mask = ~orders["item_id"].isin(known)
    n_bad = int(bad_mask.sum())
    bad_ids = sorted(orders.loc[bad_mask, "item_id"].unique())
    if n_bad:
        print(f"[clean] dropping {n_bad}/{len(orders)} order rows referencing "
              f"item_id values not in items.csv: {bad_ids}")
    orders = orders.loc[~bad_mask].copy()

    orders["created_at"] = pd.to_datetime(orders["created_at"])
    orders["date"] = orders["created_at"].dt.date
    return orders, n_bad, bad_ids


def build_ingredient_demand(orders, items, recipe, ing_scale):
    order_sku = orders.merge(items[["item_id", "sku"]], on="item_id", how="left")
    exploded = order_sku.merge(recipe, left_on="sku", right_on="recipe_id", how="left")
    # quantity_x = order-line qty (source data: always 1), quantity_y = recipe qty
    # per single menu item, in the ingredient's raw ing_meas unit (grams/ml/units)
    exploded["ingredient_used_raw"] = exploded["quantity_x"] * exploded["quantity_y"]
    exploded["ingredient_used"] = exploded["ingredient_used_raw"] * exploded["ing_id"].map(ing_scale)

    daily = (
        exploded.groupby(["date", "ing_id"])["ingredient_used"]
        .sum()
        .reset_index()
        .rename(columns={"ingredient_used": "units_sold"})
    )
    return daily


def build_demand_and_stock(daily, ingredients, inventory, ing_scale, active_ing_ids):
    # inventory.quantity = number of the ingredient's purchase pack (ing_weight
    # sized) on hand. Raw on-hand in ing_meas units = quantity * ing_weight.
    # (Inferred -- the source doesn't document the unit. Sanity-checked against
    # recipe consumption: treating quantity as literal raw grams/ml would leave
    # several ingredients below a single drink's worth of stock, which isn't a
    # plausible "current inventory" snapshot for an operating cafe.)
    inv = inventory.merge(ingredients[["ing_id", "ing_weight"]], on="ing_id")
    inv["start_on_hand"] = inv["quantity"] * inv["ing_weight"] * inv["ing_id"].map(ing_scale)
    start_stock = inv.set_index("ing_id")["start_on_hand"].to_dict()

    all_dates = sorted(daily["date"].unique())
    full_index = pd.MultiIndex.from_product([all_dates, active_ing_ids], names=["date", "ing_id"])
    dense = (
        daily.set_index(["date", "ing_id"])
        .reindex(full_index, fill_value=0.0)
        .reset_index()
        .sort_values(["ing_id", "date"])
    )

    running = dict(start_stock)
    on_hand_col, stockout_col = [], []
    for _, r in dense.iterrows():
        ing_id, sold = r["ing_id"], r["units_sold"]
        balance = running[ing_id] - sold
        stockout_col.append(balance < 0)
        balance = max(0.0, balance)
        running[ing_id] = balance
        on_hand_col.append(round(balance, 4))
    dense["on_hand"] = on_hand_col
    dense["stockout_flag"] = stockout_col
    return dense


def build_suppliers(ingredients, active_ing_ids, ing_scale, ing_display_unit):
    # Fully synthetic -- the Kaggle dataset has no multi-supplier pricing or
    # lead-time data. Adapted from generate_data.py: 2 synthetic suppliers per
    # ingredient, price anchored to the ingredient's REAL unit cost with a
    # +/-15% spread, other fields randomized in the same ranges
    # generate_data.py used.
    SUPPLIER_NAME_POOL = [
        "Highland Roasters Co-op", "Continental Green Coffee", "Direct Trade Partners",
        "Garden State Dairy", "Regional Dairy Co-op", "PlantBase Foods",
        "EcoPack Supply", "Metro Foodservice", "Torani Distribution",
        "Golden Valley Wholesale", "Union Square Provisions", "Coastal Foods Group",
    ]
    rows = []
    ing = ingredients.set_index("ing_id")
    for i, ing_id in enumerate(active_ing_ids):
        r = ing.loc[ing_id]
        scale = ing_scale[ing_id]
        base_price = r["ing_price"] / (r["ing_weight"] * scale)  # $ per display unit
        n_sup = 2
        names = rng.choice(SUPPLIER_NAME_POOL, size=n_sup, replace=False)
        for s_idx in range(n_sup):
            price_mult = rng.uniform(0.88, 1.15)
            rows.append({
                "sku": r["ing_name"],
                "supplier": names[s_idx],
                "price": round(base_price * price_mult, 3),
                "lead_time": int(rng.integers(1, 10)),
                "reliability": round(float(rng.uniform(0.90, 0.99)), 2),
                "moq": round(float(rng.uniform(2, 20)), 1) if ing_display_unit[ing_id] != "unit"
                       else int(rng.integers(5, 30)),
                "availability": rng.choice(["In stock", "In stock", "Limited"]),
            })
    return pd.DataFrame(rows)


def build_skus(ingredients, active_ing_ids, ing_scale, ing_display_unit):
    rows = []
    ing = ingredients.set_index("ing_id")
    for ing_id in active_ing_ids:
        r = ing.loc[ing_id]
        scale = ing_scale[ing_id]
        unit_price = r["ing_price"] / (r["ing_weight"] * scale)
        holding_cost = round(unit_price * ANNUAL_HOLDING_RATE / 365, 5)
        rows.append({
            "sku": r["ing_name"],
            "unit": ing_display_unit[ing_id],
            "holding_cost_per_unit_day": holding_cost,   # ESTIMATED (see header)
            "ordering_cost": FLAT_ORDERING_COST,          # ESTIMATED (see header)
            "safety_stock_days": FLAT_SAFETY_STOCK_DAYS,  # ESTIMATED (see header)
        })
    return pd.DataFrame(rows)


def main():
    ingredients, inventory, items, orders, recipe = load_raw()

    ing_scale = {}
    ing_display_unit = {}
    for _, r in ingredients.iterrows():
        factor, label = DISPLAY_SCALE[r["ing_meas"]]
        ing_scale[r["ing_id"]] = factor
        ing_display_unit[r["ing_id"]] = label

    orders_clean, n_bad_orders, bad_item_ids = clean_orders(orders, items)

    used_ing_ids = set(recipe["ing_id"].unique())
    all_ing_ids = set(ingredients["ing_id"].unique())
    unused_ing_ids = sorted(all_ing_ids - used_ing_ids)
    active_ing_ids = sorted(used_ing_ids)  # only ingredients that appear in >=1 recipe

    daily = build_ingredient_demand(orders_clean, items, recipe, ing_scale)
    dense = build_demand_and_stock(daily, ingredients, inventory, ing_scale, active_ing_ids)

    ing_name = ingredients.set_index("ing_id")["ing_name"].to_dict()
    demand_out = pd.DataFrame({
        "date": dense["date"],
        "sku": dense["ing_id"].map(ing_name),
        "units_sold": dense["units_sold"].round(4),
        "on_hand": dense["on_hand"],
        "stockout_flag": dense["stockout_flag"],
    })
    demand_out.to_csv(OUT_DIR / "daily_demand.csv", index=False)

    skus_out = build_skus(ingredients, active_ing_ids, ing_scale, ing_display_unit)
    skus_out.to_csv(OUT_DIR / "skus.csv", index=False)

    suppliers_out = build_suppliers(ingredients, active_ing_ids, ing_scale, ing_display_unit)
    suppliers_out.to_csv(OUT_DIR / "suppliers.csv", index=False)

    # ---------------- summary ----------------
    n_days = demand_out["date"].nunique()
    date_min, date_max = demand_out["date"].min(), demand_out["date"].max()
    stockout_days = int(demand_out["stockout_flag"].sum())

    print()
    print("=" * 70)
    print("[done] wrote data/daily_demand.csv, data/skus.csv, data/suppliers.csv")
    print(f"  - {len(active_ing_ids)} active ingredient SKUs (of 18 total in ingredients.csv)")
    print(f"  - date range: {date_min} to {date_max} ({n_days} calendar days)")
    print(f"  - {len(demand_out)} daily_demand.csv rows, {stockout_days} simulated stockout-days")
    print(f"  - {n_bad_orders} order rows dropped (unresolvable item_id: {bad_item_ids})")
    print(f"  - ingredients with NO recipe usage, excluded entirely: "
          f"{[ing_name[i] for i in unused_ing_ids]}")
    print("=" * 70)


if __name__ == "__main__":
    main()
