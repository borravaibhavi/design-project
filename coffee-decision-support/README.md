# Coffee shop decision-support prototype

A working prototype of the LLM-based supply-chain decision support
system from the project brief, rebuilt around a coffee shop and
running on synthetic placeholder data (no real data partner yet).

## What's here

- `generate_data.py` — generates synthetic historical data (2021-08-31 by
  default) for 6 SKUs (espresso beans, whole milk, oat milk, paper cups,
  vanilla syrup, sugar), with realistic seasonality, weekly patterns,
  and deliberately injected outliers/missing values so the cleaning
  step has real work to do. Writes to `data/`.
- `pipeline.py` — the four-layer system end to end:
  1. **Clean** — interpolates missing reads, clips outliers
  2. **Forecast** — seasonal-naive demand forecast with a 95%-style
     confidence band
  3. **EOQ + reorder point** — classic economic order quantity plus a
     safety-stock buffer sized off demand variability and lead time
  4. **Supplier selection (LP)** — `scipy.optimize.linprog` minimizes
     an effective cost that penalizes low reliability and limited
     availability, subject to covering the EOQ
  5. **Plain-English translation** — rule-based by default;
     `translate_with_claude()` in the same file switches to a real
     Claude API call if you set `ANTHROPIC_API_KEY` (falls back
     automatically if the key or the `anthropic` package isn't there)
  Writes `recommendations.json`.
- `dashboard.html` — a self-contained, no-build dashboard (open it
  directly in a browser) that reads the generated data and mirrors the
  Streamlit mockup from the brief: stock levels, a recommendation
  queue with Approve/Adjust, a supplier comparison table with LP
  scores, and a 14-day forecast chart per SKU.
- `data/*.csv` and `recommendations.json` — the already-generated
  synthetic dataset and pipeline output, so the dashboard works out of
  the box without rerunning anything.

## Running it yourself

```bash
pip install numpy pandas scipy
python3 generate_data.py   # regenerate synthetic data (change the seed/date range inside)
python3 pipeline.py        # rerun the pipeline -> recommendations.json
```

Then open `dashboard.html` in any browser — it embeds the JSON output
directly, no server needed.

## Swapping in real data later

Point `generate_data.py`'s output format is the contract: `pipeline.py`
only expects three CSVs (`daily_demand.csv`, `suppliers.csv`,
`skus.csv`) with the columns shown in each file. Replace those three
files with a real POS/inventory export in the same shape and the rest
of the pipeline runs unchanged.

## Known simplifications (be upfront about these in the showcase)

- All data is synthetic — clearly label it as such in any report.
- The LP "reliability/availability penalty" weighting is a modeling
  choice, not derived from real cost data; explain the reasoning if asked.
- The rule-based Layer 3 fallback is intentionally simple; the real
  Claude API path is wired up but needs an API key to demonstrate live.
