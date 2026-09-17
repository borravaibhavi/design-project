"""
agents/or_agent.py

Wraps pipeline.py's existing EOQ / reorder-point / LP supplier-selection
logic behind the shared state-in / decision-out interface (see
backtest/state.py), so the classical OR pipeline can run as one "policy"
inside the walk-forward backtest, scored identically to llm_agent and
historical_agent by the same simulator.

Decision rule -- unchanged from pipeline.py, just re-triggered at every
decision date instead of once over the whole dataset: classic
continuous-review (s, Q) policy. If current on-hand is at or below the
reorder point, order the EOQ, split across suppliers by the same LP.
Otherwise, hold.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline import eoq_calc, forecast_demand, select_supplier_lp  # noqa: E402


def _clean_window(demand_history: list[dict]) -> pd.DataFrame:
    """The same interpolate/clip cleaning pipeline.py's load_and_clean()
    applies, scoped to just this agent's visible window -- mean/std used
    for outlier clipping come only from history strictly before as_of,
    so this stays no-lookahead-safe."""
    df = pd.DataFrame(demand_history)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df["units_sold"] = df["units_sold"].interpolate(limit_direction="both")

    mean, std = df["units_sold"].mean(), df["units_sold"].std()
    if std and not np.isnan(std):
        df["units_sold_clean"] = df["units_sold"].clip(lower=max(0, mean - 4 * std), upper=mean + 4 * std)
    else:
        df["units_sold_clean"] = df["units_sold"]
    return df


def decide(state: dict) -> dict:
    """state (see backtest.state.build_state) -> decision dict. Same
    output shape llm_agent.decide() and historical_agent.decide() use,
    so backtest_engine/simulator can treat all three policies uniformly."""
    sku = state["sku"]
    window = _clean_window(state["demand_history"])

    forecasts, level, resid_std = forecast_demand(window, horizon=14)

    sku_params = state["sku_params"]
    sup_rows = state["suppliers"]
    lead_time = float(np.mean([s["lead_time"] for s in sup_rows])) if sup_rows else 3.0

    eoq_result = eoq_calc(
        avg_daily_demand=level,
        ordering_cost=sku_params["ordering_cost"],
        holding_cost_per_unit_day=sku_params["holding_cost_per_unit_day"],
        lead_time_days=lead_time,
        demand_std=resid_std,
        safety_stock_days=sku_params["safety_stock_days"],
    )

    current_on_hand = state["current_on_hand"]
    should_order = current_on_hand <= eoq_result["reorder_point"]

    allocation = []
    if should_order:
        supplier_choice = select_supplier_lp(eoq_result["eoq"], sup_rows)
        allocation = supplier_choice["allocation"] if supplier_choice else []
        reasoning = (
            f"On-hand ({current_on_hand:.1f}) is at or below the reorder point "
            f"({eoq_result['reorder_point']:.1f}). Ordering {eoq_result['eoq']} units "
            f"across {len(allocation)} supplier(s) per LP allocation."
        )
    else:
        reasoning = (
            f"On-hand ({current_on_hand:.1f}) is above the reorder point "
            f"({eoq_result['reorder_point']:.1f}); no action needed."
        )

    return {
        "sku": sku,
        "date": state["as_of_date"],
        "policy": "or_baseline",
        "action": "order" if should_order else "hold",
        "order_qty": eoq_result["eoq"] if should_order else 0.0,
        "allocation": allocation,
        "forecast_level": round(level, 2),
        "reorder_point": eoq_result["reorder_point"],
        "safety_stock": eoq_result["safety_stock"],
        "reasoning": reasoning,
    }


if __name__ == "__main__":
    from backtest.state import build_state, get_decision_dates, load_dataset

    data_dir = Path(__file__).parent.parent / "data_synthetic_backup"
    demand, skus, suppliers = load_dataset(data_dir)

    for sku in ["Espresso Beans (Arabica)", "Oat Milk"]:
        dates = get_decision_dates(demand, sku, history_days=90)
        print(f"\n=== {sku} ===")
        for d in [dates[0], dates[len(dates) // 2], dates[-1]]:
            state = build_state(demand, skus, suppliers, sku, d, history_days=28)
            decision = decide(state)
            print(f"{decision['date']}: action={decision['action']:5s} "
                  f"qty={decision['order_qty']:>6.1f}  reorder_pt={decision['reorder_point']:>6.1f}  "
                  f"on_hand={state['current_on_hand']:>6.1f}")
