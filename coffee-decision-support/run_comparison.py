"""
run_comparison.py

Runs the narrow LLM-vs-OR walk-forward comparison: 2 SKUs, the final
year of the synthetic dataset, weekly decisions for both policies
(matched cadence, so the comparison isolates decision QUALITY rather
than decision FREQUENCY). Also runs the OR baseline at its natural
daily cadence as a second reference point -- what a fully automated,
zero-marginal-cost OR system could do if it re-evaluated every day.

This is a real-money run: ~2 skus x ~52 weekly decisions = ~104 Claude
API calls total for llm_agent. Scope (2 skus / 1 year / weekly) was
confirmed with the project owner before running -- see session log.

Requires ANTHROPIC_API_KEY (and ANTHROPIC_WORKSPACE_ID if the key is
workspace-scoped) set in the environment.

Run: python run_comparison.py
"""

import datetime as dt
from pathlib import Path

import agents.llm_agent as llm_agent
import agents.or_agent as or_agent
from backtest.backtest_engine import run_backtest
from backtest.state import load_dataset

DATA_DIR = Path(__file__).parent / "data_synthetic_backup"
LOG_DIR = Path(__file__).parent / "results" / "runs"

SKUS = ["Espresso Beans (Arabica)", "Oat Milk"]
WINDOW_DAYS = 365
DECISION_INTERVAL_DAYS = 7


def main():
    demand, skus_df, suppliers = load_dataset(DATA_DIR)
    all_dates = sorted(demand["date"].unique())
    end_date = all_dates[-1]
    start_date = end_date - dt.timedelta(days=WINDOW_DAYS)

    print(f"Comparison window: {start_date} to {end_date} ({WINDOW_DAYS} days)")
    print(f"SKUs: {SKUS}\n")

    results = []
    for sku in SKUS:
        or_daily = run_backtest(
            demand, skus_df, suppliers, sku, or_agent, "or_baseline_daily",
            start_date=start_date, end_date=end_date,
            decision_interval_days=1, log_dir=LOG_DIR,
        )
        or_weekly = run_backtest(
            demand, skus_df, suppliers, sku, or_agent, "or_baseline_weekly",
            start_date=start_date, end_date=end_date,
            decision_interval_days=DECISION_INTERVAL_DAYS, log_dir=LOG_DIR,
        )
        llm_weekly = run_backtest(
            demand, skus_df, suppliers, sku, llm_agent, "llm_agent",
            start_date=start_date, end_date=end_date,
            decision_interval_days=DECISION_INTERVAL_DAYS, log_dir=LOG_DIR,
        )
        results += [or_daily, or_weekly, llm_weekly]

        print(f"=== {sku} ===")
        for r in (or_daily, or_weekly, llm_weekly):
            fallbacks = sum(1 for d in r["decisions"] if d.get("llm_error_fallback"))
            flag = f"  fallbacks={fallbacks}" if fallbacks else ""
            print(f"  {r['policy']:20s} cost=${r['total_cost']:>9.2f}  orders={r['order_count']:>3d}  "
                  f"stockout_days={r['stockout_days']:>3d}  service_level={r['service_level']}{flag}")
        print()

    return results


if __name__ == "__main__":
    main()
