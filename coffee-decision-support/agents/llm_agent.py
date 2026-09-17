"""
agents/llm_agent.py

The LLM policy: same state-in / decision-out interface as or_agent and
historical_agent, but the decision comes from a real Claude API call
instead of EOQ/LP math. This is the "trains itself" story from the
project brief -- no fine-tuning, no model weights touched. Each call
sees a compact summary of recent demand, current stock, the supplier
catalog, and (via state["decision_history"]) this agent's own last few
decisions and what actually happened after them (outcome: stockouts,
holding cost) -- an in-context reflection loop, the same way a manager
reading last week's report would adjust course.

Uses forced tool-use rather than asking for freeform JSON, so the
decision shape is guaranteed structured instead of parsed out of prose.
Falls back to a safe "hold" decision on any API error so one bad call
can't crash a multi-year backtest -- but the fallback is flagged
(llm_error_fallback: True) rather than silently counted as a real
zero-cost decision, since that would quietly flatter the LLM's numbers
in the final comparison.

If the state includes a "market_trend" field (see backtest/state.py's
market_trends param and market_data/build_market_trends.py), the LLM
also sees a real external commodity/producer-price signal for this
ingredient -- e.g. "coffee is up 13% year over year" -- and can weigh
proactively ordering ahead of a rising trend. This is a genuine
LLM-only capability: the classical EOQ/LP math in or_agent.py has no
mechanism to use a forward-looking signal like this at all, so
or_agent's own state calls never include it.

decide_live() is a separate, non-backtested capability: it gives the
model a real web-search tool (server-side, no beta header, no round
trip needed) so it can look up actual current news/prices before
deciding "today". It is NOT used inside the walk-forward backtest --
live search results aren't reproducible or point-in-time-safe the way
the historical market_trend signal is, so mixing them into the
backtest would quietly break the no-lookahead guarantee. Treat it as a
demo of what a live deployment could do, scored separately if at all.

IMPORTANT (per the project brief): sanity-check this on a handful of
hand-picked days (run this file directly) BEFORE running it across
years of history through backtest_engine -- each call costs real API
spend, and a bug found after a wide run has already burned the budget.
"""

from __future__ import annotations

import json
import os

import anthropic

MODEL = "claude-sonnet-5"

DECISION_TOOL = {
    "name": "record_decision",
    "description": "Record the inventory reorder decision for this SKU as of today.",
    "input_schema": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["order", "hold"]},
            "order_qty": {
                "type": "number",
                "description": "Total units to order, across all suppliers combined. 0 if action is 'hold'.",
            },
            "allocation": {
                "type": "array",
                "description": "How order_qty is split across suppliers. Empty if action is 'hold'.",
                "items": {
                    "type": "object",
                    "properties": {
                        "supplier": {"type": "string"},
                        "qty": {"type": "number"},
                    },
                    "required": ["supplier", "qty"],
                },
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "2-3 sentence explanation a shop owner could read, covering why this "
                    "action and, if ordering, why this supplier choice."
                ),
            },
        },
        "required": ["action", "order_qty", "allocation", "reasoning"],
    },
}

SYSTEM_PROMPT = (
    "You are an experienced coffee shop operations manager responsible for "
    "reordering one raw-material SKU. You will see recent demand history, "
    "current stock on hand, this ingredient's cost/ordering parameters, the "
    "available suppliers (price, lead time, reliability, availability, "
    "minimum order quantity), your own recent decisions and what actually "
    "happened after them (stockouts, holding cost incurred), and sometimes a "
    "market_trend field with a real external commodity/producer-price signal "
    "for this ingredient's raw input. Decide today's action: place a "
    "replenishment order now, or hold. Balance the risk of a stockout "
    "(running out before a new order could arrive, given lead time) against "
    "the cost of ordering too early or too much (cash tied up, holding cost, "
    "needing somewhere to store it). If market_trend is present, weigh "
    "whether a rising price trend justifies ordering more now to hedge "
    "against paying more later -- but check its 'is_proxy' and 'note' fields "
    "first: a proxy signal (e.g. a manufacturing cost index standing in for a "
    "raw commodity) deserves less weight than a direct global spot price. "
    "Use ONLY the data in this message -- do not assume information you were "
    "not given. Call record_decision exactly once with your decision."
)

LIVE_SYSTEM_PROMPT = (
    SYSTEM_PROMPT
    + " You also have a web_search tool. You may search a few times (2-3 max) "
    "for real current news or prices that could affect this ingredient's supply "
    "or cost (e.g. a drought or harvest report in a major growing region, a "
    "recent commodity price move) before deciding -- only if it seems likely "
    "to change the decision, not by default. Then call record_decision exactly "
    "once with your final decision, citing anything you found in your reasoning."
)


def _compact_state_for_prompt(state: dict) -> dict:
    payload = {
        "sku": state["sku"],
        "unit": state["unit"],
        "as_of_date": state["as_of_date"],
        "current_on_hand": state["current_on_hand"],
        "currently_stockout": state["currently_stockout"],
        "sku_params": state["sku_params"],
        "suppliers": state["suppliers"],
        "recent_demand_history": state["demand_history"],
        "your_recent_decisions_and_outcomes": state["decision_history"],
    }
    if state.get("market_trend") is not None:
        payload["market_trend"] = state["market_trend"]
    return payload


def _fallback_decision(sku: str, date: str, reason: str) -> dict:
    return {
        "sku": sku,
        "date": date,
        "policy": "llm_agent",
        "action": "hold",
        "order_qty": 0.0,
        "allocation": [],
        "reasoning": f"[fallback -- no LLM decision applied] {reason}",
        "llm_error_fallback": True,
    }


def decide(state: dict, model: str = MODEL, client: "anthropic.Anthropic | None" = None) -> dict:
    sku = state["sku"]
    date = state["as_of_date"]

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and client is None:
        return _fallback_decision(sku, date, "ANTHROPIC_API_KEY not set")

    if client is None:
        # identity-linked API keys require the target workspace on every request
        workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
        default_headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
        client = anthropic.Anthropic(api_key=api_key, default_headers=default_headers)

    payload = _compact_state_for_prompt(state)
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            tools=[DECISION_TOOL],
            tool_choice={"type": "tool", "name": "record_decision"},
            messages=[{"role": "user", "content": json.dumps(payload, default=str)}],
        )
        tool_call = next(b for b in msg.content if b.type == "tool_use" and b.name == "record_decision")
        result = tool_call.input
    except Exception as e:
        return _fallback_decision(sku, date, f"{type(e).__name__}: {e}")

    action = result.get("action")
    if action not in ("order", "hold"):
        return _fallback_decision(sku, date, f"model returned invalid action={action!r}")

    allocation = result.get("allocation") or []
    order_qty = float(result.get("order_qty") or 0.0)
    if action == "hold":
        allocation, order_qty = [], 0.0

    return {
        "sku": sku,
        "date": date,
        "policy": "llm_agent",
        "action": action,
        "order_qty": order_qty,
        "allocation": allocation,
        "reasoning": result.get("reasoning", ""),
        "llm_error_fallback": False,
    }


def decide_live(state: dict, model: str = MODEL, client: "anthropic.Anthropic | None" = None,
                 max_search_uses: int = 3) -> dict:
    """Live-only variant of decide(): the model may use a real, server-side
    web_search tool before committing to record_decision. NOT used inside
    the walk-forward backtest -- see module docstring for why. tool_choice
    is "auto" here (not forced) since forcing record_decision would prevent
    the model from searching first.
    """
    sku = state["sku"]
    date = state["as_of_date"]

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and client is None:
        return _fallback_decision(sku, date, "ANTHROPIC_API_KEY not set")

    if client is None:
        workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
        default_headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
        client = anthropic.Anthropic(api_key=api_key, default_headers=default_headers)

    payload = _compact_state_for_prompt(state)
    web_search_tool = {"type": "web_search_20260209", "name": "web_search", "max_uses": max_search_uses}
    try:
        msg = client.messages.create(
            model=model,
            max_tokens=3000,
            system=LIVE_SYSTEM_PROMPT,
            tools=[web_search_tool, DECISION_TOOL],
            tool_choice={"type": "auto"},
            messages=[{"role": "user", "content": json.dumps(payload, default=str)}],
        )
        tool_call = next(
            (b for b in msg.content if b.type == "tool_use" and b.name == "record_decision"), None
        )
        if tool_call is None:
            return _fallback_decision(sku, date, "model did not call record_decision")
        result = tool_call.input
        searches = [b for b in msg.content if b.type == "web_search_tool_result"]
    except Exception as e:
        return _fallback_decision(sku, date, f"{type(e).__name__}: {e}")

    action = result.get("action")
    if action not in ("order", "hold"):
        return _fallback_decision(sku, date, f"model returned invalid action={action!r}")

    allocation = result.get("allocation") or []
    order_qty = float(result.get("order_qty") or 0.0)
    if action == "hold":
        allocation, order_qty = [], 0.0

    return {
        "sku": sku,
        "date": date,
        "policy": "llm_agent_live",
        "action": action,
        "order_qty": order_qty,
        "allocation": allocation,
        "reasoning": result.get("reasoning", ""),
        "web_searches_used": len(searches),
        "llm_error_fallback": False,
    }


if __name__ == "__main__":
    # Sanity-check step (per the project brief): run on a handful of
    # hand-picked days BEFORE the wide backtest. This costs a small,
    # bounded number of real API calls -- not years of history.
    from pathlib import Path

    from backtest.state import build_state, get_decision_dates, load_dataset, load_market_trends

    data_dir = Path(__file__).parent.parent / "data_synthetic_backup"
    demand, skus, suppliers = load_dataset(data_dir)
    market_trends = load_market_trends(Path(__file__).parent.parent / "market_data" / "market_trends.csv")

    SAMPLE_SKUS = ["Espresso Beans (Arabica)", "Oat Milk"]
    N_SAMPLES = 3

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY not set -- set it before running this sanity check.")
        raise SystemExit(1)

    for sku in SAMPLE_SKUS:
        dates = get_decision_dates(demand, sku, history_days=90)
        idxs = sorted(set([0, len(dates) // 3, 2 * len(dates) // 3, len(dates) - 1]))[:N_SAMPLES]
        sample_dates = [dates[i] for i in idxs]
        print(f"\n=== {sku} ===")
        for d in sample_dates:
            state = build_state(demand, skus, suppliers, sku, d, history_days=28)
            decision = decide(state)
            print(f"{decision['date']}: action={decision['action']:5s} "
                  f"qty={decision['order_qty']:>6.1f}  allocation={decision['allocation']}")
            print(f"  on_hand={state['current_on_hand']:.1f}  reasoning: {decision['reasoning']}")
            if decision.get("llm_error_fallback"):
                print("  *** FALLBACK USED -- investigate before running the full backtest ***")

    # Market-trend sanity check: same idea, but now with a real external
    # commodity signal wired in, on the SKU with the strongest real data
    # (coffee, up ~13% YoY as of mid-2026 -- see backtest/state.py's smoke test)
    print("\n=== Espresso Beans (Arabica) -- WITH market_trend ===")
    trend_sku = "Espresso Beans (Arabica)"
    trend_dates = get_decision_dates(demand, trend_sku, history_days=90)
    for d in [trend_dates[-30], trend_dates[-1]]:
        state = build_state(demand, skus, suppliers, trend_sku, d, history_days=28,
                             market_trends=market_trends)
        decision = decide(state)
        mt = state.get("market_trend")
        print(f"{decision['date']}: action={decision['action']:5s} qty={decision['order_qty']:>6.1f}")
        if mt:
            print(f"  market_trend: {mt['source_name']} pct_change_yoy={mt['pct_change_yoy']}% "
                  f"is_proxy={mt['is_proxy']}")
        print(f"  reasoning: {decision['reasoning']}")
        if decision.get("llm_error_fallback"):
            print("  *** FALLBACK USED ***")

    # Live-mode sanity check: ONE call, web search enabled, no backtest
    # involvement. Real API spend -- kept to a single call here on purpose.
    print("\n=== Espresso Beans (Arabica) -- LIVE mode (web search) ===")
    live_state = build_state(demand, skus, suppliers, trend_sku, trend_dates[-1], history_days=28,
                              market_trends=market_trends)
    live_decision = decide_live(live_state)
    print(f"action={live_decision['action']} qty={live_decision.get('order_qty')} "
          f"web_searches_used={live_decision.get('web_searches_used')}")
    print(f"reasoning: {live_decision['reasoning']}")
    if live_decision.get("llm_error_fallback"):
        print("  *** FALLBACK USED ***")
