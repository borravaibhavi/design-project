"""
backtest/state.py

The shared "what would an agent have known?" object. Every agent
(or_agent, llm_agent, historical_agent) and the backtest engine itself
call build_state() to get the exact same view of the world at a given
decision date -- this is the single integrity guardrail that makes the
walk-forward backtest meaningful. If this leaks future data, every
downstream comparison is worthless.

Convention: a decision made "as of" date D is made at the start of day
D, having observed everything through the close of day D-1. So:
  - demand_history only includes rows with date < D
  - current_on_hand / currently_stockout come from the last close
    at or before D-1
  - decision_history (this agent's own past decisions + outcomes)
    only includes entries with date < D
  - sku_params and suppliers are treated as a static, always-known
    catalog/contract snapshot (the dataset has no time dimension for
    them) -- a known simplification, not a lookahead risk, since real
    supplier terms don't change day to day in this data at all.

validate_no_lookahead() is a self-check build_state() runs on its own
output before returning, so a bug here fails loudly instead of quietly
corrupting a multi-year backtest.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

DATE_FMT = "%Y-%m-%d"


def _to_date(d) -> dt.date:
    if isinstance(d, dt.date) and not isinstance(d, dt.datetime):
        return d
    if isinstance(d, dt.datetime):
        return d.date()
    return pd.Timestamp(d).date()


def _date_str(d: dt.date) -> str:
    return d.strftime(DATE_FMT)


# ---------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------
def load_dataset(data_dir: Path | str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load demand/skus/suppliers from a data directory in pipeline.py's schema.

    Defaults used by the backtest point this at data_synthetic_backup/
    (5.7 years of history) rather than data/ (6 real days) -- see the
    project decision recorded in README / check-in: the real Kaggle
    ingredient data doesn't span enough days for a walk-forward backtest.
    """
    data_dir = Path(data_dir)
    demand = pd.read_csv(data_dir / "daily_demand.csv", parse_dates=["date"])
    demand["date"] = demand["date"].dt.date
    skus = pd.read_csv(data_dir / "skus.csv")
    suppliers = pd.read_csv(data_dir / "suppliers.csv")
    return demand, skus, suppliers


def get_decision_dates(demand: pd.DataFrame, sku: str, history_days: int = 90) -> list[dt.date]:
    """Dates for which build_state() can produce a state with a full
    history_days window for this sku, i.e. valid walk-forward steps."""
    dates = sorted(demand.loc[demand["sku"] == sku, "date"].unique())
    if len(dates) <= history_days:
        return []
    return list(dates[history_days:])


# ---------------------------------------------------------------------
# Guardrail
# ---------------------------------------------------------------------
class LookaheadError(AssertionError):
    pass


def validate_no_lookahead(state: dict, as_of: dt.date) -> None:
    """Recursively scan the state dict for any "date" field on or after
    as_of. Raises LookaheadError if found. This is the guardrail: every
    agent interface funnels through build_state(), and build_state()
    always runs this on itself before returning."""

    def walk(node, path):
        if isinstance(node, dict):
            for k, v in node.items():
                if k == "date" and v is not None:
                    d = _to_date(v)
                    if d >= as_of:
                        raise LookaheadError(
                            f"lookahead violation at {path}.date = {v} "
                            f"(as_of = {_date_str(as_of)})"
                        )
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, item in enumerate(node):
                walk(item, f"{path}[{i}]")

    walk(state, "state")


# ---------------------------------------------------------------------
# State builder
# ---------------------------------------------------------------------
def build_state(
    demand: pd.DataFrame,
    skus: pd.DataFrame,
    suppliers: pd.DataFrame,
    sku: str,
    as_of,
    history_days: int = 90,
    decision_log: list[dict] | None = None,
) -> dict:
    """Everything an agent would have known when deciding for `sku` at
    the start of `as_of`. No row dated >= as_of is ever included.

    decision_log: this agent's own prior decisions for this sku, each a
    dict with at least a "date" key (plus whatever decide()/simulate()
    produced -- qty, supplier, reasoning, outcome). The backtest engine
    owns this list and appends to it after each simulated outcome;
    build_state() re-filters it to date < as_of defensively, so even a
    caller bug can't leak a same-day-or-future entry into the prompt.
    """
    as_of = _to_date(as_of)
    decision_log = decision_log or []

    sku_demand = demand.loc[demand["sku"] == sku].sort_values("date")
    past = sku_demand.loc[sku_demand["date"] < as_of]
    if past.empty:
        raise ValueError(f"no history before {as_of} for sku={sku!r}; not a valid decision date")

    window = past.tail(history_days)
    last_row = past.iloc[-1]

    sku_row = skus.loc[skus["sku"] == sku]
    if sku_row.empty:
        raise ValueError(f"sku {sku!r} not found in skus.csv")
    sku_row = sku_row.iloc[0]

    sup_rows = suppliers.loc[suppliers["sku"] == sku].to_dict("records")

    # decision_log is expected to already be strictly historical (the
    # backtest engine only ever appends past outcomes before calling
    # decide() for the next date). A same-day-or-future entry here means
    # a bug in the caller, not dirty input to clean up -- fail loudly
    # rather than silently dropping it and masking the bug.
    for d in decision_log:
        d_date = _to_date(d["date"])
        if d_date >= as_of:
            raise LookaheadError(
                f"decision_log entry dated {_date_str(d_date)} is not "
                f"strictly before as_of={_date_str(as_of)} -- caller bug"
            )

    state = {
        "sku": sku,
        "unit": sku_row["unit"],
        "as_of_date": _date_str(as_of),
        "last_observed_date": _date_str(last_row["date"]),
        "sku_params": {
            "holding_cost_per_unit_day": float(sku_row["holding_cost_per_unit_day"]),
            "ordering_cost": float(sku_row["ordering_cost"]),
            "safety_stock_days": float(sku_row["safety_stock_days"]),
        },
        "suppliers": [
            {
                "supplier": s["supplier"],
                "price": float(s["price"]),
                "lead_time": float(s["lead_time"]),
                "reliability": float(s["reliability"]),
                "moq": float(s["moq"]),
                "availability": s["availability"],
            }
            for s in sup_rows
        ],
        "current_on_hand": float(last_row["on_hand"]),
        "currently_stockout": bool(last_row["stockout_flag"]),
        "demand_history": [
            {
                "date": _date_str(r["date"]),
                "units_sold": None if pd.isna(r["units_sold"]) else float(r["units_sold"]),
                "stockout_flag": bool(r["stockout_flag"]),
            }
            for _, r in window.iterrows()
        ],
        "decision_history": decision_log,
    }

    validate_no_lookahead(state, as_of)
    return state


if __name__ == "__main__":
    # smoke test: build one state, confirm the guardrail fires on a
    # deliberately corrupted decision_log entry
    data_dir = Path(__file__).parent.parent / "data_synthetic_backup"
    demand, skus, suppliers = load_dataset(data_dir)

    sku = "Espresso Beans (Arabica)"
    dates = get_decision_dates(demand, sku, history_days=90)
    print(f"{len(dates)} valid decision dates for {sku!r}, "
          f"first={dates[0]} last={dates[-1]}")

    sample_date = dates[100]
    state = build_state(demand, skus, suppliers, sku, sample_date, history_days=28)
    print(f"\nstate as_of={state['as_of_date']} last_observed={state['last_observed_date']}")
    print(f"current_on_hand={state['current_on_hand']} suppliers={len(state['suppliers'])}")
    print(f"demand_history: {len(state['demand_history'])} days, "
          f"first={state['demand_history'][0]['date']} last={state['demand_history'][-1]['date']}")

    print("\n[guardrail check] clean state passes validate_no_lookahead: OK")

    try:
        bad_log = [{"date": _date_str(sample_date), "qty": 10}]  # same-day, should be rejected
        build_state(demand, skus, suppliers, sku, sample_date, history_days=28, decision_log=bad_log)
        print("[guardrail check] FAILED to catch same-day decision_log leak")
    except LookaheadError as e:
        print(f"[guardrail check] PASSED -- correctly rejected same-day decision_log entry: {e}")

    try:
        future_date = dates[105]
        bad_log2 = [{"date": _date_str(future_date), "qty": 10}]
        build_state(demand, skus, suppliers, sku, sample_date, history_days=28, decision_log=bad_log2)
        print("[guardrail check] FAILED to catch future decision_log leak")
    except LookaheadError as e:
        print(f"[guardrail check] PASSED -- correctly rejected future decision_log entry: {e}")
