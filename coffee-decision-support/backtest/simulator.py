"""
backtest/simulator.py

Given any policy's decision at a decision date, simulates what actually
happens going forward: orders arrive after each allocated supplier's
lead time, real historical demand consumes on-hand inventory, unmet
demand is logged as a stockout (a lost sale -- on-hand floors at 0, it
is not backordered), and holding/ordering costs accrue. Policy-agnostic:
the exact same functions score or_agent, llm_agent, and historical_agent
identically, so the walk-forward comparison is apples to apples.

Deliberately does NOT invent a dollar cost for stockouts -- the source
data has no real markup/margin figures, so a "cost of a lost sale"
number would be fabricated. Instead stockouts are reported as their own
metric (days and units short) alongside total holding+ordering cost: a
textbook cost-vs-service-level framing rather than one fudged blended
score.

Purchase cost (qty x price) IS tracked, separately from holding/ordering
cost, using whatever price each allocation line was actually decided
against (state["suppliers"][*]["price"] at decision time -- see
backtest/state.py's build_effective_supplier_price). On the static
data_synthetic_backup dataset this price never moves, so purchase cost
timing is irrelevant there; once prices follow real market history
(data_real_seeded/), this is what makes ordering ahead of a price rise
actually save real simulated dollars, instead of just narrating a trend
with no effect on the numbers.
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field


@dataclass
class SimLedger:
    """Running state for one (policy, sku) simulation."""

    on_hand: float
    pending: list = field(default_factory=list)  # [(arrival_date, qty), ...]
    holding_cost_total: float = 0.0
    ordering_cost_total: float = 0.0
    purchase_cost_total: float = 0.0
    stockout_units_total: float = 0.0
    stockout_days: int = 0
    order_count: int = 0
    total_demand: float = 0.0
    days_simulated: int = 0
    currently_stockout: bool = False


def _lookup(supplier_name: str, suppliers: list[dict], field_name: str, default: float) -> float:
    for s in suppliers:
        if s["supplier"] == supplier_name:
            return float(s[field_name])
    return default  # fallback if a decision names a supplier not in the current catalog


def apply_decision(ledger: SimLedger, decision: dict | None, as_of: dt.date,
                    suppliers: list[dict], ordering_cost: float) -> None:
    """Register a new order's arrivals and its real purchase cost. Called
    once, on the day the decision is made -- consumption/holding accrue
    separately every day via step_day(), regardless of whether an order
    was placed.

    `suppliers` should be the CURRENT state's supplier list (i.e.
    state["suppliers"] from the same build_state() call that produced
    `decision`), not a static snapshot -- its price field may already be
    adjusted for real market movement (see module docstring), and that's
    the price this function charges against."""
    if not decision or decision.get("action") != "order":
        return
    allocation = decision.get("allocation") or []
    if not allocation:
        return
    ledger.ordering_cost_total += ordering_cost  # one order event -> one flat ordering cost,
    ledger.order_count += 1                      # even if the LP splits it across suppliers
    for line in allocation:
        lead_time = _lookup(line["supplier"], suppliers, "lead_time", default=3.0)
        price = _lookup(line["supplier"], suppliers, "price", default=0.0)
        arrival = as_of + dt.timedelta(days=round(lead_time))
        ledger.pending.append((arrival, float(line["qty"])))
        ledger.purchase_cost_total += price * float(line["qty"])


def step_day(ledger: SimLedger, date: dt.date, actual_units_sold,
             holding_cost_per_unit_day: float) -> dict:
    """Advance the simulation by exactly one calendar day: receive any
    orders arriving today, consume that day's REAL realized demand,
    floor at zero on a stockout, accrue holding cost on the end-of-day
    balance."""
    arrived_today = [qty for (arrival, qty) in ledger.pending if arrival == date]
    ledger.on_hand += sum(arrived_today)
    ledger.pending = [(a, q) for (a, q) in ledger.pending if a != date]

    is_missing = actual_units_sold is None or (
        isinstance(actual_units_sold, float) and math.isnan(actual_units_sold)
    )
    # a missing meter reading in the source data means zero recorded
    # consumption that day (see generate_data.py) -- not "unknown", zero
    demand_today = 0.0 if is_missing else float(actual_units_sold)

    balance = ledger.on_hand - demand_today
    stockout_units = max(0.0, -balance)
    ledger.on_hand = max(0.0, balance)
    ledger.currently_stockout = stockout_units > 0

    if stockout_units > 0:
        ledger.stockout_units_total += stockout_units
        ledger.stockout_days += 1

    holding_cost_today = holding_cost_per_unit_day * ledger.on_hand
    ledger.holding_cost_total += holding_cost_today
    ledger.total_demand += demand_today
    ledger.days_simulated += 1

    return {
        "date": date.isoformat() if hasattr(date, "isoformat") else str(date),
        "on_hand": round(ledger.on_hand, 2),
        "holding_cost_today": round(holding_cost_today, 4),
        "demand": round(demand_today, 2),
        "stockout_units": round(stockout_units, 2),
        "arrived_qty": round(sum(arrived_today), 2),
    }


if __name__ == "__main__":
    # smoke test: one order, verify it sits pending until lead time,
    # then arrives; verify a stockout floors on_hand at 0 and is logged
    today = dt.date(2024, 1, 1)
    ledger = SimLedger(on_hand=10.0)
    suppliers = [{"supplier": "Acme", "lead_time": 3, "price": 1.0, "reliability": 0.9,
                  "moq": 1, "availability": "In stock"}]

    apply_decision(ledger, {"action": "order", "allocation": [{"supplier": "Acme", "qty": 50}]},
                    today, suppliers, ordering_cost=12.0)
    assert ledger.order_count == 1 and ledger.ordering_cost_total == 12.0
    assert ledger.pending == [(dt.date(2024, 1, 4), 50.0)]
    assert ledger.purchase_cost_total == 50.0, ledger.purchase_cost_total  # 50 units @ $1.0

    step_day(ledger, dt.date(2024, 1, 1), 8.0, holding_cost_per_unit_day=0.05)
    assert ledger.on_hand == 2.0, ledger.on_hand
    step_day(ledger, dt.date(2024, 1, 2), 5.0, holding_cost_per_unit_day=0.05)  # stockout: 2 - 5 = -3
    assert ledger.on_hand == 0.0 and ledger.stockout_units_total == 3.0 and ledger.stockout_days == 1
    step_day(ledger, dt.date(2024, 1, 3), 0.0, holding_cost_per_unit_day=0.05)
    assert ledger.on_hand == 0.0
    step_day(ledger, dt.date(2024, 1, 4), 1.0, holding_cost_per_unit_day=0.05)  # order arrives: 0+50-1
    assert ledger.on_hand == 49.0, ledger.on_hand

    print("[smoke test] all assertions passed")
    print(f"holding_cost_total={ledger.holding_cost_total:.2f} "
          f"stockout_units_total={ledger.stockout_units_total} "
          f"order_count={ledger.order_count} purchase_cost_total={ledger.purchase_cost_total}")

    # a second order at a higher price should cost more for the same qty --
    # this is the mechanism that makes ordering ahead of a price rise
    # actually pay off in the simulator, not just in an agent's narration
    ledger2 = SimLedger(on_hand=0.0)
    cheap_suppliers = [{"supplier": "Acme", "lead_time": 1, "price": 1.0}]
    pricey_suppliers = [{"supplier": "Acme", "lead_time": 1, "price": 2.0}]
    apply_decision(ledger2, {"action": "order", "allocation": [{"supplier": "Acme", "qty": 10}]},
                    today, cheap_suppliers, ordering_cost=0.0)
    apply_decision(ledger2, {"action": "order", "allocation": [{"supplier": "Acme", "qty": 10}]},
                    today, pricey_suppliers, ordering_cost=0.0)
    assert ledger2.purchase_cost_total == 30.0, ledger2.purchase_cost_total  # 10@$1 + 10@$2
    print("[smoke test] purchase cost correctly reflects the price at each order's own decision time")
