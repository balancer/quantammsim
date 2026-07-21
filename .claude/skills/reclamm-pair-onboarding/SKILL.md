---
name: reclamm-pair-onboarding
description: Onboard a new token pair for reCLAMM simulations — fetch price data, run the Optuna parameter sweep, produce final-sim CSVs/plots with candidate pool params, and optionally build a Balancer AutoRange investor pitch deck. Use when asked to simulate a reCLAMM pool for a pair, gather candidate price_ratio/margin/shift params, add a pair to run_final_sims.py, or make a pitch/proposal deck for a pair.
---

# reCLAMM pair onboarding

Pipeline: **price data → Optuna sweep → register pair → `run_final_sims.py`**.
Worked examples on branch `btc-eth-pair-onboarding`: BTC/ETH (Binance data,
real pool coefficients) and BOLD/USDC (CoinGecko data, median-fallback noise).

**Never retrain the noise model.** The checked-in artifacts
(`results/mm_noise/{model.npz,meta.json}`, `results/competitor_tvl/competitor_tvl.npz`)
are fit across 38 pools and reused for every pair; per-pair arrays are built
and cached automatically under `results/mm_noise/_sim_arrays/`.

## Step 1 — price data (`quantammsim/data/<TOKEN>_USD.parquet`)

`BTC_USD.parquet` is required for **every** pair (market features), plus one
parquet per pool token (`TOKEN_MAP` in `quantammsim/calibration/market_features.py`
maps WBTC→BTC, WETH→ETH, USDT→USDC, …).

`scripts/download_data.py` handles both sources — one command, any token:

- **Binance-listed token**: `python scripts/download_data.py <TOKEN>`.
- **Not on Binance**: same script auto-falls back to CoinGecko. If the token is
  in the built-in registry (`COINGECKO_IDS` in
  `quantammsim/utils/data_processing/coingecko_data.py`) just run
  `python scripts/download_data.py <TOKEN>`; otherwise pass the id explicitly.
  For a stable-paired volatile token, add `--peg <STABLE>` to write/extend the
  stablecoin's flat-$1 parquet over the same window. BOLD/USDC example:
  `python scripts/download_data.py BOLD --cg-id liquity-bold-2 --peg USDC`.
  (`--source coingecko` forces CoinGecko even for a Binance-listed token.)

  CoinGecko builds the minute grid by forward-fill with daily volume spread /1440
  (so the daily resample recovers real volume). Free tier = **last 365 days
  only** — this bounds the earliest simulation start and usually forces a
  per-pair train window (Step 4).

Check coverage: the parquet must span train start → test end.

## Step 2 — pool id and the noise fallback

Look the pair up in `results/mm_noise/meta.json` (`pool_ids` / `pool_tokens`).

- **Pair present** (e.g. WBTC/WETH → `0xa6f548df93de92`): use that id; the
  noise model gets real per-pool coefficients and competitor TVL.
- **Pair absent**: any placeholder id works — the builder falls back to
  median coefficients + K=$10M (`noise_model_arrays.py`). Median alpha is
  tiny, so organic volume (and fee revenue) will be near zero. If realistic
  fees matter, re-anchor: after `build_mm_simulator_arrays`, shift
  `noise_base += log(V_max_daily) - mean(noise_base)` and optionally set a
  constant competitor K — see `scripts/run_rpl_eth_sweep.py` (~lines 124-134)
  for the pattern and the pair's real daily volume for V_max.

## Step 3 — Optuna sweep

Demo scale (~50 trials, minutes; the team's production scale is
`scripts/run_full_sweep.sh`: 300 trials × 4 objectives × 3 penalties per TVL):

```
python scripts/tune_reclamm_calibrated_noise.py \
  --noise-model mm_observed --artifact-dir results/mm_noise \
  --tokens BTC ETH --pool-id 0xa6f548df93de92 \
  --gas-cost 1.0 --fees 0.0025 --initial-pool-value 5000000 \
  --objective returns_over_hodl --n-trials 50 --pr-max 5.0 \
  --start-date "2025-01-01 00:00:00" --end-date "2025-10-05 00:00:00" \
  --output results/full_sweep/returns_over_hodl_<pair>_<tvl>.json
```

- One sweep per (pair, TVL). Search space: price_ratio 1.01–200 (log),
  margin 0.01–0.99, shift_exponent 1e-5–125 (log). Cap `--pr-max` sensibly:
  ~5 for correlated majors, ~1.05 for stable/stable.
- Dates: default train window is 2025-01-01 → 2025-10-05 (pre flash-crash);
  keep it unless data starts later. `--end-test-date` sets the OOS span used
  for validation metrics.
- Side effect (the part `run_final_sims.py` actually reads): a trajectory
  file `results/run_<hash>.json` written by `train_on_historic_data`.

## Step 4 — register the pair and run final sims

Add an entry to `PAIR_CONFIGS` in `scripts/run_final_sims.py` (tokens,
pool_id, gas_cost, fees, the swept TVLs; `--pair` choices follow the dict).
If the pair's data can't cover the default windows, add per-pair overrides:

```python
"train": ("2025-07-15 00:00:00", "2025-10-05 00:00:00"),   # optional
"test":  ("2025-10-25 00:00:00", "2026-05-01 00:00:00"),   # optional
```

The sweep's train window **must match** the pair's train window — trials are
filtered by it (`filter_trials_to_window`).

```
python scripts/run_final_sims.py --pair <name>    # or --all
```

Selection: best trial per TVL by OOS `returns_over_hodl` (override with
`--metric`). Outputs in `results/final_sims/`:
`run_{Value,Reserves,TokenValues}_<pair>_<tvl>_{train,test}_<hash>.csv`,
themed plots `<pair>_{train,test}[_weights]_{light,dark}.png`, and
`<pair>_sim_results.pkl`. The printed summary gives train/test RoH and fees.

## Step 5 — investor deck (optional)

A ready-made Balancer-v3 "AutoRange" pitch deck lives at
`.claude/skills/reclamm-pair-onboarding/assets/autorange-deck-example.html`
(single self-contained HTML file, opens in any browser, has an **Export PDF**
button). To make a deck for a new pair, **duplicate that file** and edit only
the four things below. **No extra scripts are needed** — all numbers come
straight from the `run_final_sims.py` output you already produced (printed
train/test RoH + fees, and the `run_Value_*` CSVs normalised to 100). **Do not
change any simulation logic** to build a deck.

Everything is driven by two JS objects near the bottom of the file
(`const CONFIG = {…}` and `const DECK_DATA = {…}`). Only touch:

1. **Token images** — `CONFIG.tokenLogoUrl` and `CONFIG.tokenBLogoUrl`
   (CoinGecko `image.large` URLs, e.g.
   `https://api.coingecko.com/api/v3/coins/<id>?...` → `.image.large`).

2. **Token names / branding** — `CONFIG.tokenA`, `CONFIG.tokenB`,
   `CONFIG.brandColor`, `CONFIG.date` (these auto-fill the page title, cover
   logos and date). Then find-and-replace the hardcoded pair text: the cover
   `<h1 id="cover-title">`, cover placeholder letters (e.g. `IN`/`DO`), the
   `X / Y AutoRange - Balancer v3` footer labels, the proposal hero
   `<h2 class="proposal-config">`, the range-chart CoinGecko fetch id +
   `'<TOKEN> / USD'` label, and the `pdf.save('…')` filename.

3. **Results slide** (`#slide-results`) — the graphs + LP-return comparison.
   Fill `DECK_DATA.series` (`hodl`, `balancer` = the passive full-range
   reference, `pr<N>` = the simulated AutoRange configs — each an array
   normalised to 100 from the `run_Value_*` CSVs) and `DECK_DATA.headline`
   (`pr<N>.roh_pct`, `pr<N>.fee_yield_pct`, `balancer.roh_pct`) from the sim
   summary. Also update the hardcoded `Backtest · <dates>` eyebrow, the
   `$<X> pool` chart title, and the results-note sentence.

4. **Proposal slide** (`#slide-proposal`) — the recommendation. Set
   `DECK_DATA.recommend.price_ratio` (controls which results row is starred)
   and the four `<span class="proposal-param-val">` values (Price Ratio,
   Margin, Shift Speed, Swap Fee).

**Leave everything else as-is.** In particular the parameter-sweep slide
(`#slide-sweep`, the 3-D heat-map) is a **standard illustration** — it does
**not** need per-project data; keep `DECK_DATA.heatmap` and the sweep slide
unchanged. The "What is AutoRange", methodology, and security slides are also
generic and stay put.

## Gotchas

- `--method` on `run_final_sims.py` is currently a no-op.
- First run per (pool, window) builds noise arrays (needs network-free local
  parquets only); subsequent runs hit the `_sim_arrays` cache.
- Stable pairs: expect PR near the 1.01 floor and near-zero fees under the
  median-fallback noise — re-anchor (Step 2) before trusting fee numbers.
- Sweeps and final sims are pure JAX forward passes — a laptop handles demo
  scale; production sweeps want the parallel `run_full_sweep.sh` machinery.
