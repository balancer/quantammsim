"""CoinGecko data source for the historic-data pipeline.

Generalizes the one-off ``scripts/prepare_bold_usdc_data.py`` so that any token
listed on CoinGecko can be fetched through the same waterfall as Binance/Coinbase.

Two public entry points:

- :func:`get_coingecko_data` — fetch a token's daily price + volume from CoinGecko
  and expand it to a complete 1-minute grid in the pipeline's canonical schema
  (unix-indexed; columns ``date, symbol, open, high, low, close, Volume USD,
  Volume <TOKEN>``). Daily USD volume is spread evenly across the 1440 minutes of
  each day so a daily resample-sum downstream recovers the true daily volume.
- :func:`rebuild_stable_peg` — write/extend ``<STABLE>_USD.parquet`` at a flat
  $1.00 peg over the union of any existing coverage and a requested window.

CoinGecko free/demo tier caps ``market_chart`` history at the last **365 days**,
which bounds the earliest possible simulation start for CoinGecko-only tokens.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Symbol -> CoinGecko id registry (single source of truth; imported by
# scripts/fetch_token_mcaps.py). Covers the tokens in our matched pools plus
# common Balancer tokens. Pass --cg-id on the command line for anything absent.
# ---------------------------------------------------------------------------
COINGECKO_IDS = {
    # Blue-chip / wrapped natives
    "WETH": "ethereum",
    "ETH": "ethereum",
    "WBTC": "wrapped-bitcoin",
    "BTC": "bitcoin",
    "cbBTC": "bitcoin",   # Coinbase wrapped BTC — use BTC mcap
    "USDC": "usd-coin",
    "USDT": "tether",
    "DAI": "dai",
    "wstETH": "wrapped-steth",
    "stETH": "staked-ether",
    "rETH": "rocket-pool-eth",
    "cbETH": "coinbase-wrapped-staked-eth",
    "WMATIC": "polygon-ecosystem-token",
    "MATIC": "polygon-ecosystem-token",
    "POL": "polygon-ecosystem-token",
    "WAVAX": "avalanche-2",
    "AVAX": "avalanche-2",
    "GNO": "gnosis",
    "WXDAI": "dai",       # Wrapped xDAI ≈ DAI
    "xDAI": "dai",
    "S": "sonic-3",
    "wS": "sonic-3",
    # Mid-cap DeFi
    "AAVE": "aave",
    "LINK": "chainlink",
    "UNI": "uniswap",
    "BAL": "balancer",
    "MKR": "maker",
    "CRV": "curve-dao-token",
    "COMP": "compound-governance-token",
    "SNX": "havven",
    "LDO": "lido-dao",
    "RPL": "rocket-pool",
    "SUSHI": "sushi",
    "YFI": "yearn-finance",
    "1INCH": "1inch",
    "ENS": "ethereum-name-service",
    "ARB": "arbitrum",
    "OP": "optimism",
    "PENDLE": "pendle",
    "ENA": "ethena",
    "EIGEN": "eigenlayer",
    "COW": "cow-protocol",
    "SAFE": "safe",
    # Smaller / specific tokens in our pools
    "ACX": "across-protocol",
    "ALCX": "alchemix",
    "QI": "benqi",
    "QNT": "quant-network",
    "RDNT": "radiant-capital",
    "TREE": "treehouse",
    "HYPE": "hyperliquid",       # Hyperliquid L1 native (underlying of tHYPE)
    "wHYPE": "hyperliquid",      # Wrapped HYPE — same underlying price
    "XAI": "xai-blockchain",
    # Wrapped aTokens — use underlying
    "waEthLidoWETH": "ethereum",
    "waEthLidowstETH": "wrapped-steth",
    "waBasWETH": "ethereum",
    "waBasUSDC": "usd-coin",
    "waEthUSDC": "usd-coin",
    "waGnoGNO": "gnosis",
    "waGnowstETH": "wrapped-steth",
    # Additional tokens from expanded pool set
    "wPOL": "polygon-ecosystem-token",
    "stS": "sonic-3",        # Staked Sonic — use S mcap
    "JitoSOL": "jito-governance-token",
    "scUSD": "usd-coin",     # Rings scUSD stablecoin — use USDC mcap as proxy
    "DOLA": "dola-usd",
    # CoinGecko-only tokens (not on Binance/Coinbase)
    "BOLD": "liquity-bold-2",
}

# Asset type classification
STABLECOINS = {
    "USDC", "USDT", "DAI", "WXDAI", "xDAI", "GHO", "LUSD", "crvUSD",
    "FRAX", "sDAI", "scUSD", "DOLA",
    "waBasUSDC", "waEthUSDC",
}
NATIVE_LST = {
    "WETH", "ETH", "wstETH", "stETH", "rETH", "cbETH",
    "WBTC", "BTC", "cbBTC",
    "WMATIC", "MATIC", "POL", "wPOL",
    "WAVAX", "AVAX",
    "GNO", "S", "wS", "stS",
    "JitoSOL",
    "waEthLidoWETH", "waEthLidowstETH",
    "waBasWETH", "waGnoGNO", "waGnowstETH",
}

_CG_BASE = "https://api.coingecko.com/api/v3"


def resolve_cg_id(token: str, override: str | None = None) -> str:
    """Resolve a token symbol to its CoinGecko id.

    ``override`` (the ``--cg-id`` flag) wins; otherwise look up the built-in
    registry. Raises a clear error if neither is available.
    """
    if override:
        return override
    cg_id = COINGECKO_IDS.get(token) or COINGECKO_IDS.get(token.upper())
    if cg_id:
        return cg_id
    raise ValueError(
        f"No CoinGecko id known for token '{token}'. Pass it explicitly with "
        f"--cg-id <coingecko-id> (find the id in the token's CoinGecko URL, "
        f"e.g. https://www.coingecko.com/en/coins/<coingecko-id>)."
    )


def fetch_coingecko_daily(cg_id: str, days: str = "365") -> tuple[list, list]:
    """Return ``(prices, total_volumes)`` as ``[[unix_ms, value], ...]``.

    Retries with a back-off on transient failures and honours 429 rate limits.
    Free/demo tier caps ``days`` at 365.
    """
    url = f"{_CG_BASE}/coins/{cg_id}/market_chart"
    params = {"vs_currency": "usd", "days": str(days), "interval": "daily"}
    for attempt in range(4):
        try:
            resp = requests.get(url, params=params, timeout=40)
            if resp.status_code == 429:
                print("  rate-limited, sleeping 60s …")
                time.sleep(60)
                continue
            resp.raise_for_status()
            j = resp.json()
            return j.get("prices", []), j.get("total_volumes", [])
        except Exception as exc:
            print(f"  CoinGecko attempt {attempt + 1} failed: {exc}")
            time.sleep(10)
    raise RuntimeError(f"CoinGecko fetch failed for id '{cg_id}'")


def get_coingecko_data(
    token: str,
    root: str,
    cg_id: str | None = None,
    days: str = "365",
    numeraire: str = "USD",
) -> pd.DataFrame | None:
    """Fetch ``token`` from CoinGecko as a unix-indexed 1-minute canonical frame.

    Matches the schema the exchange helpers feed into ``update_historic_data``:
    unix (int64 ms) index; columns ``date, symbol, open, high, low, close,
    Volume USD, Volume <TOKEN>``. Daily close is forward-filled across each day;
    daily USD volume is spread ``/1440`` so a daily resample-sum recovers the
    real daily volume. ``root`` is accepted for signature parity with the other
    ``get_*_data`` source helpers (CoinGecko needs no local cache dir).

    Returns ``None`` if CoinGecko yields too little history to be usable.
    """
    resolved_id = resolve_cg_id(token, cg_id)
    print(f"Fetching {token} ({resolved_id}) daily data from CoinGecko …")
    prices, volumes = fetch_coingecko_daily(resolved_id, days=days)
    if len(prices) < 2:
        print(f"CoinGecko returned too little data for {token} ({len(prices)} points)")
        return None

    # Daily frame: close price + daily USD volume, indexed by midnight-UTC day.
    pdf = pd.DataFrame(prices, columns=["unix", "close"])
    vdf = pd.DataFrame(volumes, columns=["unix", "vol_usd"])
    pdf["day"] = pd.to_datetime(pdf["unix"], unit="ms", utc=True).dt.normalize()
    vdf["day"] = pd.to_datetime(vdf["unix"], unit="ms", utc=True).dt.normalize()
    daily = (
        pdf.groupby("day")["close"].last().to_frame()
        .join(vdf.groupby("day")["vol_usd"].last())
    )
    daily["vol_usd"] = daily["vol_usd"].fillna(0.0)
    daily = daily.dropna(subset=["close"]).sort_index()

    start = daily.index.min()
    end = daily.index.max()
    print(
        f"  {token} daily: {start.date()} → {end.date()}  ({len(daily)} days), "
        f"close ${daily['close'].min():.4f}–${daily['close'].max():.4f}"
    )

    # Expand to a complete 1-minute grid covering full days [start, end+1day).
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    grid_end = end + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)
    min_range = pd.date_range(start=start, end=grid_end, freq="1min")
    unix_idx = ((min_range - epoch).total_seconds() * 1000).astype(np.int64)

    out = pd.DataFrame(index=unix_idx)
    out.index.name = "unix"
    out["date"] = min_range.tz_localize(None).strftime("%Y-%m-%d %H:%M:%S")
    day_key = min_range.normalize()

    close_map = daily["close"].to_dict()
    vol_map = (daily["vol_usd"] / 1440.0).to_dict()
    out["close"] = pd.Series(day_key.map(close_map), index=unix_idx).astype(float)
    out["Volume USD"] = pd.Series(day_key.map(vol_map), index=unix_idx).astype(float)
    out["close"] = out["close"].ffill().bfill()
    out["Volume USD"] = out["Volume USD"].fillna(0.0)
    out["open"] = out["high"] = out["low"] = out["close"]
    out[f"Volume {token}"] = out["Volume USD"] / out["close"].clip(lower=1e-9)
    out["symbol"] = f"{token}/{numeraire}"

    out = out[
        ["date", "symbol", "open", "high", "low", "close",
         "Volume USD", f"Volume {token}"]
    ]
    return out


def rebuild_stable_peg(
    stable_token: str,
    root: str,
    cover_start_ms: int,
    cover_end_ms: int,
) -> None:
    """Write/extend ``<STABLE>_USD.parquet`` at a flat $1.00 peg.

    Covers the union of any existing coverage and ``[cover_start_ms,
    cover_end_ms]`` on a 1-minute grid. Volume columns are zero. ``root`` is the
    data directory (with trailing slash), matching ``update_historic_data``.
    """
    path = Path(root) / f"{stable_token}_USD.parquet"
    start_ms = int(cover_start_ms)
    end_ms = int(cover_end_ms)
    if path.exists():
        existing = pd.read_parquet(path)
        idx_vals = (
            existing.index if existing.index.name == "unix" else existing["unix"]
        )
        start_ms = min(start_ms, int(np.asarray(idx_vals).min()))
        end_ms = max(end_ms, int(np.asarray(idx_vals).max()))

    idx = np.arange(start_ms, end_ms + 60_000, 60_000, dtype=np.int64)
    n = len(idx)
    out = pd.DataFrame(
        {
            "unix": idx,
            "date": pd.to_datetime(idx, unit="ms", utc=True)
            .tz_localize(None)
            .strftime("%Y-%m-%d %H:%M:%S"),
            "symbol": f"{stable_token}/USD",
            "open": np.ones(n),
            "high": np.ones(n),
            "low": np.ones(n),
            "close": np.ones(n),
            "Volume USD": np.zeros(n),
            f"Volume {stable_token}": np.zeros(n),
        }
    )
    out.to_parquet(path, engine="pyarrow", index=False)
    print(
        f"  Saved {n:,} rows → {path}  (peg $1.00, "
        f"{out['date'].iloc[0]} → {out['date'].iloc[-1]})"
    )
