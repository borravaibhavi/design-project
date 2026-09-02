"""
pipeline.py

The four-layer system from the project brief, running end to end on
the synthetic data in ./data/:

  Layer 1: Ingest + clean (drop/interpolate bad rows, flag outliers)
  Layer 2: Optimization models
              - demand forecast (Holt-Winters-style seasonal naive + CI)
              - EOQ reorder point/quantity
              - LP-based supplier selection (minimize cost subject to
                lead-time and reliability constraints)
  Layer 3: Translate structured output into plain-English recommendations
           (rule-based fallback here; swap in a real Claude API call by
           setting ANTHROPIC_API_KEY and using translate_with_claude())
  Layer 4: Package everything the dashboard needs into recommendations.json

Run: python3 pipeline.py
"""

import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog

DATA_DIR = Path(__file__).parent / "data"
OUT_PATH = Path(__file__).parent / "recommendations.json"

SERVICE_Z = 1.65  # ~95% service level


# ---------------------------------------------------------------------
# Layer 1: clean
# ---------------------------------------------------------------------
def load_and_clean():
    demand = pd.read_csv(DATA_DIR / "daily_demand.csv", parse_dates=["date"])
    before = len(demand)

    # interpolate missing units_sold per SKU
    demand["units_sold"] = demand.groupby("sku")["units_sold"].transform(
        lambda s: s.interpolate(limit_direction="both")
    )

    # outlier clipping: cap at 4 std devs from each SKU's rolling mean
    def clip_outliers(s):
        mean, std = s.mean(), s.std()
        return s.clip(lower=max(0, mean - 4 * std), upper=mean + 4 * std)

    demand["units_sold_clean"] = demand.groupby("sku")["units_sold"].transform(clip_outliers)
    n_clipped = (demand["units_sold_clean"] != demand["units_sold"]).sum()

    print(f"[clean] {before} rows loaded, {n_clipped} outliers clipped")
    return demand


# ---------------------------------------------------------------------
# Layer 2a: demand forecast with confidence interval
# ---------------------------------------------------------------------
def forecast_demand(sku_df, horizon=14):
    y = sku_df.sort_values("date")["units_sold_clean"].to_numpy()
    # 7-day seasonal naive baseline + trailing 28-day level, blended
    level = y[-28:].mean()
    seasonal = y[-28:].reshape(-1, 7).mean(axis=0) if len(y) >= 28 else np.full(7, level)
    seasonal_factor = seasonal / seasonal.mean()

    resid_std = y[-28:].std()
    forecasts = []
    for h in range(horizon):
        dow = h % 7
        point = level * seasonal_factor[dow]
        forecasts.append({
            "day": h + 1,
            "point": round(float(point), 1),
            "low": round(max(0, point - SERVICE_Z * resid_std), 1),
            "high": round(point + SERVICE_Z * resid_std, 1),
        })
    return forecasts, level, resid_std


# ---------------------------------------------------------------------
# Layer 2b: EOQ + reorder point
# ---------------------------------------------------------------------
def eoq_calc(avg_daily_demand, ordering_cost, holding_cost_per_unit_day,
             lead_time_days, demand_std, safety_stock_days):
    annual_demand = avg_daily_demand * 365
    holding_cost_annual = holding_cost_per_unit_day * 365
    if holding_cost_annual <= 0 or annual_demand <= 0:
        return {"eoq": 0, "reorder_point": 0, "safety_stock": 0}

    eoq = math.sqrt((2 * annual_demand * ordering_cost) / holding_cost_annual)
    safety_stock = SERVICE_Z * demand_std * math.sqrt(lead_time_days)
    reorder_point = avg_daily_demand * lead_time_days + safety_stock
    return {
        "eoq": round(eoq, 1),
        "reorder_point": round(reorder_point, 1),
        "safety_stock": round(safety_stock, 1),
    }


# ---------------------------------------------------------------------
# Layer 2c: LP supplier selection
#   Minimize total cost = price*qty, subject to:
#     - sum(qty) >= EOQ (must cover the order)
#     - qty <= availability-implied cap per supplier
#     - a soft preference against low-reliability/limited-availability
#       suppliers via a cost penalty rather than a hard constraint
# ---------------------------------------------------------------------
def select_supplier_lp(eoq_qty, supplier_rows):
    n = len(supplier_rows)
    if n == 0 or eoq_qty <= 0:
        return None

    # effective cost per unit penalizes low reliability and limited stock
    eff_cost = []
    for s in supplier_rows:
        penalty = (1 - s["reliability"]) * s["price"] * 2
        if s["availability"] == "Limited":
            penalty += 0.10 * s["price"]
        eff_cost.append(s["price"] + penalty)

    c = np.array(eff_cost)
    # cap per supplier at 3x MOQ or the full EOQ, whichever smaller — keeps
    # the LP from dumping the whole order on one tiny/limited supplier
    upper_bounds = [min(eoq_qty, max(s["moq"] * 3, eoq_qty)) for s in supplier_rows]
    A_ub = [[-1] * n]  # -sum(qty) <= -eoq_qty  =>  sum(qty) >= eoq_qty
    b_ub = [-eoq_qty]
    bounds = [(s["moq"] if eoq_qty >= s["moq"] else 0, ub) for s, ub in zip(supplier_rows, upper_bounds)]

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        # fallback: just rank by effective cost and take cheapest wholesale
        ranked = sorted(zip(supplier_rows, eff_cost), key=lambda t: t[1])
        best = ranked[0][0]
        return {"allocation": [{"supplier": best["supplier"], "qty": eoq_qty}], "lp_scores": None}

    allocation = []
    for s, qty in zip(supplier_rows, res.x):
        if qty > 0.5:
            allocation.append({"supplier": s["supplier"], "qty": round(float(qty), 1)})

    # normalize an "LP score" 0-1 per supplier for the dashboard (cheaper/more
    # reliable/faster = higher), same spirit as Figure 3's supplier table
    scores = {}
    max_cost, min_cost = max(eff_cost), min(eff_cost)
    for s, ec in zip(supplier_rows, eff_cost):
        norm_cost = 1 - (ec - min_cost) / (max_cost - min_cost + 1e-9)
        scores[s["supplier"]] = round(0.5 * norm_cost + 0.3 * s["reliability"] +
                                       0.2 * (1 / s["lead_time"]) * 3, 2)
    return {"allocation": allocation, "lp_scores": scores}


# ---------------------------------------------------------------------
# Layer 3: plain-English translation (rule-based fallback)
# ---------------------------------------------------------------------
def translate_rule_based(sku, eoq_result, supplier_choice, days_until_reorder, current_stock):
    if not supplier_choice or not supplier_choice["allocation"]:
        return f"No viable supplier allocation found for {sku} — needs manual review."

    top = max(supplier_choice["allocation"], key=lambda a: a["qty"])
    if days_until_reorder <= 0:
        urgency = f"{sku} is at or below its reorder point now."
    else:
        urgency = f"{sku} will hit its reorder point in about {days_until_reorder} day(s)."

    return (
        f"{urgency} Recommend ordering {eoq_result['eoq']} units from "
        f"{top['supplier']} ({top['qty']} units), which balances price, "
        f"lead time, and reliability best among current options. "
        f"Current stock: {current_stock} units; reorder point: "
        f"{eoq_result['reorder_point']} units."
    )


def translate_with_claude(sku, eoq_result, supplier_choice, days_until_reorder, current_stock):
    """
    Optional: real Claude API call for Layer 3, matching Figure 1's
    'Claude API (Anthropic) — Translates numerical outputs into plain
    English recommendations'. Requires ANTHROPIC_API_KEY in the
    environment. Falls back to the rule-based version if unset or on
    any error, so the pipeline never breaks without a key.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return translate_rule_based(sku, eoq_result, supplier_choice, days_until_reorder, current_stock)
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        payload = {
            "sku": sku, "eoq": eoq_result, "supplier_choice": supplier_choice,
            "days_until_reorder": days_until_reorder, "current_stock": current_stock,
        }
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": (
                    "You are the plain-English translation layer of an inventory "
                    "decision-support system. Given this structured JSON, write a "
                    "2-3 sentence recommendation for the shop owner, in the tone of "
                    "a knowledgeable operations assistant, no jargon:\n\n"
                    f"{json.dumps(payload)}"
                ),
            }],
        )
        return msg.content[0].text
    except Exception as e:
        return translate_rule_based(sku, eoq_result, supplier_choice, days_until_reorder, current_stock)


# ---------------------------------------------------------------------
# Layer 4: run everything, assemble dashboard payload
# ---------------------------------------------------------------------
def run():
    demand = load_and_clean()
    skus_df = pd.read_csv(DATA_DIR / "skus.csv")
    suppliers_df = pd.read_csv(DATA_DIR / "suppliers.csv")

    sku_cards = []
    recommendation_queue = []
    all_forecasts = {}

    for _, row in skus_df.iterrows():
        sku = row["sku"]
        sku_df = demand[demand["sku"] == sku]
        forecasts, level, resid_std = forecast_demand(sku_df)
        all_forecasts[sku] = forecasts

        current_stock = float(sku_df.sort_values("date")["on_hand"].iloc[-1])

        sup_rows = suppliers_df[suppliers_df["sku"] == sku].to_dict("records")
        eoq_result = eoq_calc(
            avg_daily_demand=level,
            ordering_cost=float(row["ordering_cost"]),
            holding_cost_per_unit_day=float(row["holding_cost_per_unit_day"]),
            lead_time_days=float(np.mean([s["lead_time"] for s in sup_rows])) if sup_rows else 3,
            demand_std=resid_std,
            safety_stock_days=float(row["safety_stock_days"]),
        )
        supplier_choice = select_supplier_lp(eoq_result["eoq"], sup_rows)

        days_until_reorder = 0
        if level > 0:
            days_until_reorder = max(0, round((current_stock - eoq_result["reorder_point"]) / level))

        status = "OK"
        if current_stock <= eoq_result["reorder_point"]:
            status = "Low"
        elif current_stock <= eoq_result["reorder_point"] * 1.3:
            status = "Watch"

        recommendation_text = translate_with_claude(
            sku, eoq_result, supplier_choice, days_until_reorder, current_stock
        )

        sku_cards.append({
            "sku": sku,
            "unit": row["unit"],
            "on_hand": round(current_stock, 1),
            "reorder_point": eoq_result["reorder_point"],
            "eoq": eoq_result["eoq"],
            "safety_stock": eoq_result["safety_stock"],
            "status": status,
            "avg_daily_demand": round(level, 1),
            "suppliers": sup_rows,
            "lp_scores": supplier_choice["lp_scores"] if supplier_choice else None,
            "allocation": supplier_choice["allocation"] if supplier_choice else [],
        })

        if status in ("Low", "Watch"):
            recommendation_queue.append({
                "sku": sku,
                "status": status,
                "text": recommendation_text,
            })

    # headline metrics for the dashboard header row
    total_stockout_days = int(demand["stockout_flag"].sum())
    metrics = {
        "active_skus": len(skus_df),
        "low_stock_alerts": sum(1 for c in sku_cards if c["status"] == "Low"),
        "pending_recommendations": len(recommendation_queue),
        "historical_stockout_days": total_stockout_days,
        "data_range": f"{demand['date'].min().date()} to {demand['date'].max().date()}",
    }

    payload = {
        "metrics": metrics,
        "sku_cards": sku_cards,
        "recommendation_queue": recommendation_queue,
        "forecasts": all_forecasts,
    }

    OUT_PATH.write_text(json.dumps(payload, indent=2, default=str))
    print(f"[done] wrote {OUT_PATH} ({OUT_PATH.stat().st_size} bytes)")
    print(f"[done] {metrics}")


if __name__ == "__main__":
    run()
