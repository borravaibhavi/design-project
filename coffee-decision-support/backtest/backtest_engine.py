"""
backtest/backtest_engine.py

The walk-forward loop: for one sku and one policy, step through every
calendar day in order. On that policy's decision days, build_state()
(no-lookahead, and given the SIMULATOR's own current on-hand -- not the
dataset's original replenishment trajectory, see state.py) -> agent.
decide() -> apply_decision() registers any new order's arrivals. Every
single day, step_day() consumes that day's REAL realized demand and
accrues holding cost, regardless of whether a decision was made today --
so every policy is scored against the exact same demand sequence.

Feeds each policy's own outcome back to itself: right before its next
decide() call, the interval since its last decision (days, holding
cost, stockouts) is attached to that prior decision_log entry as
"outcome", so llm_agent can react to what its last call actually
caused -- the "trains itself" loop from the project brief. No
fine-tuning involved, just in-context reflection on real outcomes.

decision_interval_days lets a policy decide on a coarser cadence (e.g.
weekly for the LLM agent, to bound API cost) while still being scored
against daily consumption -- see the recommended fixed 28-day/weekly
config for llm_agent in agents/llm_agent.py.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd

from backtest.simulator import SimLedger, apply_decision, step_day
from backtest.state import build_state, load_dataset


def _outcome_since(ledger: SimLedger, snapshot: dict, days: int) -> dict:
    return {
        "days": days,
        "holding_cost": round(ledger.holding_cost_total - snapshot["holding_cost_total"], 2),
        "stockout_units": round(ledger.stockout_units_total - snapshot["stockout_units_total"], 2),
        "stockout_days": ledger.stockout_days - snapshot["stockout_days"],
    }


def _snapshot(ledger: SimLedger) -> dict:
    return {
        "holding_cost_total": ledger.holding_cost_total,
        "stockout_units_total": ledger.stockout_units_total,
        "stockout_days": ledger.stockout_days,
    }


def run_backtest(
    demand: pd.DataFrame,
    skus: pd.DataFrame,
    suppliers: pd.DataFrame,
    sku: str,
    agent_module,
    policy_name: str,
    start_date: dt.date | None = None,
    end_date: dt.date | None = None,
    history_days: int = 90,
    state_window_days: int = 28,
    decision_interval_days: int = 1,
    log_dir: Path | str | None = None,
) -> dict:
    all_dates = sorted(demand.loc[demand["sku"] == sku, "date"].unique())
    if len(all_dates) <= history_days:
        raise ValueError(f"not enough history for {sku!r} to run a backtest")

    sim_start_date = all_dates[history_days]
    if start_date is not None:
        sim_start_date = max(sim_start_date, start_date)
    sim_end_date = all_dates[-1] if end_date is None else min(all_dates[-1], end_date)
    sim_dates = [d for d in all_dates if sim_start_date <= d <= sim_end_date]
    if not sim_dates:
        raise ValueError(f"empty date range for {sku!r}: {sim_start_date}..{sim_end_date}")

    sku_row = skus.loc[skus["sku"] == sku].iloc[0]
    holding_cost = float(sku_row["holding_cost_per_unit_day"])
    ordering_cost = float(sku_row["ordering_cost"])
    sup_rows = suppliers.loc[suppliers["sku"] == sku].to_dict("records")

    demand_by_date = demand.loc[demand["sku"] == sku].set_index("date")["units_sold"].to_dict()

    starting_row = (
        demand.loc[(demand["sku"] == sku) & (demand["date"] < sim_start_date)]
        .sort_values("date")
        .iloc[-1]
    )
    ledger = SimLedger(on_hand=float(starting_row["on_hand"]),
                        currently_stockout=bool(starting_row["stockout_flag"]))

    decision_log: list[dict] = []
    daily_log: list[dict] = []
    last_decision_entry: dict | None = None
    last_snapshot = _snapshot(ledger)
    days_since_decision = 0

    for i, date in enumerate(sim_dates):
        is_decision_day = i % decision_interval_days == 0

        if is_decision_day:
            if last_decision_entry is not None:
                last_decision_entry["outcome"] = _outcome_since(ledger, last_snapshot, days_since_decision)
                last_snapshot = _snapshot(ledger)
                days_since_decision = 0

            state = build_state(
                demand, skus, suppliers, sku, date,
                history_days=state_window_days,
                decision_log=decision_log,
                current_on_hand=ledger.on_hand,
                currently_stockout=ledger.currently_stockout,
            )
            decision = agent_module.decide(state)
            decision["policy"] = policy_name
            apply_decision(ledger, decision, date, sup_rows, ordering_cost)
            decision_log.append(decision)
            last_decision_entry = decision

        actual = demand_by_date.get(date)
        daily_log.append(step_day(ledger, date, actual, holding_cost))
        days_since_decision += 1

    if last_decision_entry is not None and "outcome" not in last_decision_entry:
        last_decision_entry["outcome"] = _outcome_since(ledger, last_snapshot, days_since_decision)

    result = {
        "policy": policy_name,
        "sku": sku,
        "start_date": sim_start_date.isoformat(),
        "end_date": sim_end_date.isoformat(),
        "days_simulated": ledger.days_simulated,
        "total_holding_cost": round(ledger.holding_cost_total, 2),
        "total_ordering_cost": round(ledger.ordering_cost_total, 2),
        "total_cost": round(ledger.holding_cost_total + ledger.ordering_cost_total, 2),
        "order_count": ledger.order_count,
        "stockout_days": ledger.stockout_days,
        "stockout_units": round(ledger.stockout_units_total, 2),
        "total_demand": round(ledger.total_demand, 2),
        "service_level": (
            round(1 - ledger.stockout_units_total / ledger.total_demand, 4)
            if ledger.total_demand else None
        ),
        "decisions": decision_log,
        "daily_log": daily_log,
    }

    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        safe_sku = sku.replace(" ", "_").replace("(", "").replace(")", "")
        out_path = log_dir / f"{policy_name}__{safe_sku}.json"
        out_path.write_text(json.dumps(result, indent=2, default=str))

    return result


if __name__ == "__main__":
    import agents.or_agent as or_agent

    data_dir = Path(__file__).parent.parent / "data_synthetic_backup"
    demand, skus, suppliers = load_dataset(data_dir)
    log_dir = Path(__file__).parent.parent / "results" / "runs"

    print(f"{'sku':30s} {'cost':>10s} {'orders':>7s} {'stockout_days':>14s} {'service_level':>14s}")
    for sku in skus["sku"]:
        result = run_backtest(demand, skus, suppliers, sku, or_agent, "or_baseline", log_dir=log_dir)
        print(f"{sku:30s} {result['total_cost']:>10.2f} {result['order_count']:>7d} "
              f"{result['stockout_days']:>14d} {str(result['service_level']):>14s}")
