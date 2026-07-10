import os
import argparse
from pathlib import Path
from quantammsim.utils.data_processing.historic_data_utils import (
    update_historic_data,
)
from quantammsim.utils.data_processing.coingecko_data import (
    rebuild_stable_peg,
)

# Get absolute paths relative to this script
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "quantammsim" / "data"
DATA_DIR.mkdir(exist_ok=True)
DATA_DIR_STR = str(DATA_DIR) + '/'

TICKER_FILE = SCRIPT_DIR / "ticker_list.txt"


def _token_unix_span(parquet_path):
    """Return (min_unix_ms, max_unix_ms) for a written token parquet."""
    import pandas as pd

    df = pd.read_parquet(parquet_path, columns=["unix"])
    return int(df["unix"].min()), int(df["unix"].max())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download historic data for tickers.")
    parser.add_argument(
        "tickers",
        nargs="*",
        help="List of tickers to process. If provided, overrides ticker_list.txt",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "binance", "coingecko"],
        default="auto",
        help=(
            "auto (default): try Binance/Coinbase/etc, fall back to CoinGecko if "
            "nothing is found. binance: exchange waterfall only. coingecko: fetch "
            "straight from CoinGecko."
        ),
    )
    parser.add_argument(
        "--cg-id",
        default=None,
        help=(
            "CoinGecko id override for a token not in the built-in registry "
            "(e.g. --cg-id liquity-bold-2). Single-token runs only."
        ),
    )
    parser.add_argument(
        "--cg-days",
        default="365",
        help="CoinGecko history window in days (free tier caps at 365).",
    )
    parser.add_argument(
        "--peg",
        action="append",
        default=[],
        metavar="STABLE",
        help=(
            "After downloading, write/extend <STABLE>_USD.parquet at a flat $1.00 "
            "peg covering the downloaded token's window (repeatable). "
            "e.g. BOLD --cg-id liquity-bold-2 --peg USDC"
        ),
    )
    args = parser.parse_args()

    if args.tickers:
        tickers = args.tickers
    else:
        with open(TICKER_FILE, "r") as f:
            tickers = f.readlines()

    if (args.cg_id or args.peg) and len(tickers) != 1:
        parser.error("--cg-id and --peg require exactly one ticker.")

    for token in tickers:
        token = token.strip().upper()
        print(f"Processing {token}")
        update_historic_data(
            token,
            DATA_DIR_STR,
            source=args.source,
            cg_id=args.cg_id,
            cg_days=args.cg_days,
        )
        print(f"Finished processing {token}")
        token_parquet = DATA_DIR_STR + token + "_USD.parquet"
        os.rename(
            DATA_DIR_STR + "combined_data/" + token + "_USD.parquet",
            token_parquet,
        )
        os.rename(
            DATA_DIR_STR + "combined_data/" + token + "_USD_daily.csv",
            DATA_DIR_STR + token + "_USD_daily.csv",
        )

        # Rebuild/extend any requested stablecoin pegs to cover this token's window.
        if args.peg:
            start_ms, end_ms = _token_unix_span(token_parquet)
            for stable in args.peg:
                stable = stable.strip().upper()
                print(f"Rebuilding {stable} peg to cover {token} window")
                rebuild_stable_peg(stable, DATA_DIR_STR, start_ms, end_ms)
