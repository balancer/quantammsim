"""Prepare price data for BOLD/USDC reCLAMM simulations.

1. Fetches BOLD/USD daily price + traded volume from CoinGecko (free tier =
   last 365 days), expands to a minute grid by forward-fill, and writes
   quantammsim/data/BOLD_USD.parquet.  The real daily traded volume is spread
   evenly across the day so that market_features' daily resample recovers the
   true daily USD volume (used by the organic-volume noise model).

2. Rebuilds quantammsim/data/USDC_USD.parquet pegged at $1.00 over the union
   of its existing coverage and the BOLD grid, so both the BOLD/USDC window
   and the earlier Monad window stay covered.

BOLD (Liquity V2, CoinGecko id "liquity-bold-2") is not on Binance/Coinbase,
so CoinGecko is the only source.  Free-tier history is capped at 365 days —
this bounds the earliest possible simulation start date.

Usage (from repo root):
    python scripts/prepare_bold_usdc_data.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "quantammsim" / "data"

CG_ID = "liquity-bold-2"
_CG_URL = f"https://api.coingecko.com/api/v3/coins/{CG_ID}/market_chart"


def _fetch_coingecko_daily() -> tuple[list, list]:
    """Return (prices, total_volumes) as [[unix_ms, value], ...] for last 365d."""
    params = {"vs_currency": "usd", "days": "365", "interval": "daily"}
    for attempt in range(4):
        try:
            resp = requests.get(_CG_URL, params=params, timeout=40)
            if resp.status_code == 429:
                print("  rate-limited, sleeping 60s …")
                time.sleep(60)
                continue
            resp.raise_for_status()
            j = resp.json()
            return j.get("prices", []), j.get("total_volumes", [])
        except Exception as exc:
            print(f"  CoinGecko attempt {attempt+1} failed: {exc}")
            time.sleep(10)
    raise RuntimeError("CoinGecko fetch failed for BOLD")


def build_bold_parquet() -> pd.DataFrame:
    prices, volumes = _fetch_coingecko_daily()
    if len(prices) < 100:
        raise RuntimeError(f"BOLD price series too short: {len(prices)}")

    # Daily frame: price (close) + daily USD volume, indexed by midnight-UTC day.
    pdf = pd.DataFrame(prices, columns=["unix", "close"])
    vdf = pd.DataFrame(volumes, columns=["unix", "vol_usd"])
    pdf["day"] = pd.to_datetime(pdf["unix"], unit="ms", utc=True).dt.normalize()
    vdf["day"] = pd.to_datetime(vdf["unix"], unit="ms", utc=True).dt.normalize()
    daily = (pdf.groupby("day")["close"].last()
             .to_frame()
             .join(vdf.groupby("day")["vol_usd"].last()))
    daily["vol_usd"] = daily["vol_usd"].fillna(0.0)
    daily = daily.dropna(subset=["close"]).sort_index()

    start = daily.index.min()
    end = daily.index.max()
    print(f"  BOLD daily: {start.date()} → {end.date()}  ({len(daily)} days)")
    print(f"  price ${daily['close'].min():.4f}–${daily['close'].max():.4f}, "
          f"last ${daily['close'].iloc[-1]:.4f}")
    print(f"  daily vol (USD): median ${daily['vol_usd'].median():,.0f}, "
          f"mean ${daily['vol_usd'].mean():,.0f}")

    # Expand to a complete minute grid covering full days [start, end+1day).
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    grid_end = end + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    min_range = pd.date_range(start=start, end=grid_end, freq="1min")
    min_idx = ((min_range - epoch).total_seconds() * 1000).astype(np.int64)

    out = pd.DataFrame(index=min_idx)
    out.index.name = "unix"
    out["date"] = min_range
    out["day"] = min_range.normalize()

    # Map close + per-minute volume (daily vol spread across 1440 minutes so the
    # daily resample-sum in market_features recovers the real daily volume).
    close_map = daily["close"].to_dict()
    vol_map = (daily["vol_usd"] / 1440.0).to_dict()
    out["close"] = out["day"].map(close_map).astype(float)
    out["Volume USD"] = out["day"].map(vol_map).astype(float)
    out["close"] = out["close"].ffill().bfill()
    out["Volume USD"] = out["Volume USD"].fillna(0.0)
    out["open"] = out["high"] = out["low"] = out["close"]
    out["Volume BOLD"] = out["Volume USD"] / out["close"].clip(lower=1e-9)
    out["symbol"] = "BOLD"
    out = out.drop(columns=["day"])
    out = out[["date", "symbol", "open", "high", "low", "close",
               "Volume USD", "Volume BOLD"]]

    path = DATA_DIR / "BOLD_USD.parquet"
    out.to_parquet(path, engine="pyarrow")
    print(f"  Saved {len(out):,} rows → {path}")
    return out


def rebuild_usdc_parquet(bold_df: pd.DataFrame) -> None:
    """USDC pegged at $1.00 over the union of its current grid and BOLD's."""
    path = DATA_DIR / "USDC_USD.parquet"
    start_ms = int(bold_df.index.min())
    end_ms = int(bold_df.index.max())
    if path.exists():
        existing = pd.read_parquet(path)
        start_ms = min(start_ms, int(existing.index.min()))
        end_ms = max(end_ms, int(existing.index.max()))

    idx = np.arange(start_ms, end_ms + 60_000, 60_000, dtype=np.int64)
    n = len(idx)
    out = pd.DataFrame({
        "date": pd.to_datetime(idx, unit="ms", utc=True),
        "symbol": "USDC",
        "open": np.ones(n), "high": np.ones(n),
        "low": np.ones(n), "close": np.ones(n),
        "Volume USD": np.zeros(n), "Volume USDC": np.zeros(n),
    }, index=idx)
    out.index.name = "unix"
    out.to_parquet(path, engine="pyarrow")
    print(f"  Saved {n:,} rows → {path}  (peg $1.00, "
          f"{out['date'].iloc[0].date()} → {out['date'].iloc[-1].date()})")


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print("Fetching BOLD/USD from CoinGecko …")
    bold = build_bold_parquet()
    print("Rebuilding USDC peg parquet …")
    rebuild_usdc_parquet(bold)

    print("\n--- Data summary ---")
    for token in ["BOLD", "USDC"]:
        df = pd.read_parquet(DATA_DIR / f"{token}_USD.parquet")
        dates = pd.to_datetime(df.index, unit="ms", utc=True)
        print(f"  {token:5s}: {dates.min().date()} → {dates.max().date()}  "
              f"({len(df):,} rows)  close=[{df['close'].min():.4f}, "
              f"{df['close'].max():.4f}]")


if __name__ == "__main__":
    main()
