"""
market_data/build_market_trends.py

Builds market_data/market_trends.csv from real historical commodity/
producer-price series downloaded from FRED (Federal Reserve Economic
Data, fred.stlouisfed.org), each ultimately sourced from the IMF or the
US Bureau of Labor Statistics. Raw files live in market_data/raw/,
fetched via FRED's public, no-API-key CSV endpoint:

    https://fred.stlouisfed.org/graph/fredgraph.csv?id=<SERIES_ID>

(This machine's shell has no direct internet egress -- these were
fetched once via an authenticated fetch tool and committed as raw
inputs. To refresh: re-download each URL below and overwrite the
matching file in market_data/raw/.)

SKU -> series mapping, and what kind of signal each one actually is
(some are real global spot prices; some are the best free real proxy
available, which is NOT the same thing -- see `note` in the output):

  Espresso Beans (Arabica) -> PCOFFOTMUSDM
      IMF Global price of Coffee, Other Mild Arabica (monthly, since
      1992). https://fred.stlouisfed.org/series/PCOFFOTMUSDM
      REAL GLOBAL SPOT PRICE -- direct match.

  Sugar -> PSUGAISAUSDM
      IMF Global price of Sugar, No. 11, World (monthly, since 1990).
      https://fred.stlouisfed.org/series/PSUGAISAUSDM
      REAL GLOBAL SPOT PRICE -- direct match.

  Vanilla Syrup -> PSUGAISAUSDM (reused)
      No free public global vanilla-bean price index exists (vanilla
      is thinly traded and its price is extremely volatile). Vanilla
      syrup is majority sugar, so the sugar price is used as a rough
      PROXY -- it does not capture vanilla-specific volatility at all
      and is clearly flagged as a proxy in the output.

  Oat Milk -> WPU012203
      BLS Producer Price Index, Farm Products: Oats (monthly, since
      1971). https://fred.stlouisfed.org/series/WPU012203
      REAL, direct raw-input match, but a US farm-gate price, not a
      global spot price.

  Whole Milk -> PCU31153115
      BLS Producer Price Index by Industry: Dairy Product
      Manufacturing (monthly, since 1984).
      https://fred.stlouisfed.org/series/PCU31153115
      REAL but WEAK signal -- an industry-level US manufacturing cost
      index, not a raw milk commodity price. No free real raw-milk
      farm-gate series was found.

  Paper Cups (12oz) -> WPU0911
      BLS Producer Price Index by Commodity: Wood Pulp (monthly, since
      1926). https://fred.stlouisfed.org/series/WPU0911
      REAL, indirect input match (pulp feeds into paper -> cups), US
      producer price, not a global spot price.

Run: python market_data/build_market_trends.py
"""

from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).parent / "raw"
OUT_PATH = Path(__file__).parent / "market_trends.csv"

SKU_SOURCES = {
    "Espresso Beans (Arabica)": {
        "series_id": "PCOFFOTMUSDM",
        "source_name": "IMF Global price of Coffee, Other Mild Arabica (via FRED)",
        "source_url": "https://fred.stlouisfed.org/series/PCOFFOTMUSDM",
        "is_proxy": False,
        "note": "Real global spot price -- direct match for this ingredient.",
    },
    "Sugar": {
        "series_id": "PSUGAISAUSDM",
        "source_name": "IMF Global price of Sugar, No. 11, World (via FRED)",
        "source_url": "https://fred.stlouisfed.org/series/PSUGAISAUSDM",
        "is_proxy": False,
        "note": "Real global spot price -- direct match for this ingredient.",
    },
    "Vanilla Syrup": {
        "series_id": "PSUGAISAUSDM",
        "source_name": "IMF Global price of Sugar, No. 11, World (via FRED)",
        "source_url": "https://fred.stlouisfed.org/series/PSUGAISAUSDM",
        "is_proxy": True,
        "note": "PROXY: no free global vanilla-bean price index exists. Sugar price "
                "used as a rough stand-in since syrup is majority sugar -- does NOT "
                "capture vanilla-specific volatility (e.g. Madagascar crop shocks).",
    },
    "Oat Milk": {
        "series_id": "WPU012203",
        "source_name": "BLS Producer Price Index, Farm Products: Oats (via FRED)",
        "source_url": "https://fred.stlouisfed.org/series/WPU012203",
        "is_proxy": False,
        "note": "Real direct raw-input price, but a US farm-gate producer price, "
                "not a global spot commodity price.",
    },
    "Whole Milk": {
        "series_id": "PCU31153115",
        "source_name": "BLS PPI by Industry: Dairy Product Manufacturing (via FRED)",
        "source_url": "https://fred.stlouisfed.org/series/PCU31153115",
        "is_proxy": True,
        "note": "WEAK PROXY: an industry-level US manufacturing cost index, not a "
                "raw milk farm-gate or global spot price. No free real alternative "
                "was found -- treat this signal with more skepticism than coffee/sugar.",
    },
    "Paper Cups (12oz)": {
        "series_id": "WPU0911",
        "source_name": "BLS Producer Price Index by Commodity: Wood Pulp (via FRED)",
        "source_url": "https://fred.stlouisfed.org/series/WPU0911",
        "is_proxy": True,
        "note": "Indirect input proxy (wood pulp feeds into paper cup manufacturing), "
                "US producer price, not a global spot price.",
    },
}


def main():
    rows = []
    loaded_series = {}
    for sku, cfg in SKU_SOURCES.items():
        series_id = cfg["series_id"]
        if series_id not in loaded_series:
            path = RAW_DIR / f"{series_id}.csv"
            df = pd.read_csv(path, parse_dates=["observation_date"])
            df = df.rename(columns={"observation_date": "date", series_id: "value"})
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            df = df.dropna(subset=["value"]).sort_values("date")
            loaded_series[series_id] = df
        df = loaded_series[series_id]
        for _, r in df.iterrows():
            rows.append({
                "date": r["date"].date().isoformat(),
                "sku": sku,
                "value": float(r["value"]),
                "series_id": series_id,
                "source_name": cfg["source_name"],
                "source_url": cfg["source_url"],
                "is_proxy": cfg["is_proxy"],
                "note": cfg["note"],
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)

    print(f"[done] wrote {OUT_PATH} ({len(out)} rows)")
    for sku, cfg in SKU_SOURCES.items():
        n = len(out[out["sku"] == sku])
        proxy_flag = " (PROXY)" if cfg["is_proxy"] else ""
        print(f"  {sku:30s} <- {cfg['series_id']:15s}{proxy_flag}  {n} monthly points")


if __name__ == "__main__":
    main()
