"""
backtest/metrics.py

Aggregates backtest_engine run logs (results/runs/*.json) into a
cross-policy comparison: a summary table (cost, orders, service level,
LLM-fallback count) per (policy, sku), and a cumulative-cost time
series per (policy, sku) for the dashboard's policy-comparison chart.

results/runs/ can hold logs from different-scoped runs (e.g. the full
5.7-year, 6-SKU or_baseline run from backtest_engine.py's own __main__,
alongside the narrow 2-SKU/1-year/weekly LLM-vs-OR comparison from
run_comparison.py) -- each row keeps its own start/end date and sku so
scope is never ambiguous, but build_comparison_payload() only includes
runs that share a common (start_date, end_date, sku) triple with at
least one llm_agent run, since mixing a full-history baseline into a
narrow LLM comparison chart would be an apples-to-oranges chart.

Run: python -m backtest.metrics
"""

from __future__ import annotations

import json
from pathlib import Path


def load_runs(log_dir) -> list[dict]:
    log_dir = Path(log_dir)
    return [json.loads(p.read_text()) for p in sorted(log_dir.glob("*.json"))]


def summarize(results: list[dict]) -> list[dict]:
    return [
        {
            "policy": r["policy"],
            "sku": r["sku"],
            "start_date": r["start_date"],
            "end_date": r["end_date"],
            "total_cost": r["total_cost"],
            "total_holding_cost": r["total_holding_cost"],
            "total_ordering_cost": r["total_ordering_cost"],
            "order_count": r["order_count"],
            "stockout_days": r["stockout_days"],
            "stockout_units": r["stockout_units"],
            "service_level": r["service_level"],
            "llm_fallback_count": sum(1 for d in r["decisions"] if d.get("llm_error_fallback")),
        }
        for r in results
    ]


def cumulative_cost_series(result: dict, holding_cost_per_unit_day: float, ordering_cost: float) -> list[dict]:
    """Per-day cumulative (holding + ordering) cost -- the line the
    dashboard's policy-comparison chart plots. Recomputes holding cost
    from each day's on_hand rather than trusting a precomputed field on
    the log, so this stays correct even against run logs collected
    before simulator.py started recording holding_cost_today directly
    (recomputing here is free; re-running an llm_agent backtest to
    regenerate logs is not)."""
    order_dates = {d["date"] for d in result["decisions"] if d.get("action") == "order"}
    cum = 0.0
    series = []
    for day in result["daily_log"]:
        cum += holding_cost_per_unit_day * day["on_hand"]
        if day["date"] in order_dates:
            cum += ordering_cost
        series.append({"date": day["date"], "cumulative_cost": round(cum, 2)})
    return series


def build_comparison_payload(results: list[dict], skus_df) -> dict:
    """Everything dashboard.html's policy-comparison tab needs, scoped
    to just the runs that share a window with at least one llm_agent
    run (see module docstring)."""
    ordering_cost_by_sku = dict(zip(skus_df["sku"], skus_df["ordering_cost"]))
    holding_cost_by_sku = dict(zip(skus_df["sku"], skus_df["holding_cost_per_unit_day"]))

    llm_windows = {(r["sku"], r["start_date"], r["end_date"]) for r in results if r["policy"] == "llm_agent"}
    in_scope = [r for r in results if (r["sku"], r["start_date"], r["end_date"]) in llm_windows]

    series_by_policy_sku = {
        f"{r['policy']}::{r['sku']}": cumulative_cost_series(
            r, holding_cost_by_sku[r["sku"]], ordering_cost_by_sku[r["sku"]]
        )
        for r in in_scope
    }

    sample_decisions = {
        f"{r['policy']}::{r['sku']}": [
            {k: d[k] for k in ("date", "action", "order_qty", "reasoning") if k in d}
            for d in r["decisions"] if d.get("action") == "order"
        ][:5]
        for r in in_scope
    }

    return {
        "summary": summarize(in_scope),
        "cumulative_cost_series": series_by_policy_sku,
        "sample_decisions": sample_decisions,
    }


if __name__ == "__main__":
    import pandas as pd

    log_dir = Path(__file__).parent.parent / "results" / "runs"
    skus_df = pd.read_csv(Path(__file__).parent.parent / "data_synthetic_backup" / "skus.csv")

    results = load_runs(log_dir)
    if not results:
        raise SystemExit(f"no run logs found in {log_dir} -- run backtest_engine.py or run_comparison.py first")

    payload = build_comparison_payload(results, skus_df)
    if not payload["summary"]:
        raise SystemExit("no llm_agent runs found in results/runs/ -- run run_comparison.py first")

    out_path = Path(__file__).parent.parent / "results" / "comparison.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str))

    print(f"[done] wrote {out_path} ({out_path.stat().st_size} bytes)")
    for row in payload["summary"]:
        print(f"{row['policy']:22s} {row['sku']:26s} cost=${row['total_cost']:>9.2f} "
              f"orders={row['order_count']:>3d} stockout_days={row['stockout_days']:>3d} "
              f"service_level={row['service_level']} fallbacks={row['llm_fallback_count']}")
