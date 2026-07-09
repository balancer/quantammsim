---
name: reclamm-pair-onboarding
description: Onboard a new token pair for reCLAMM simulations — fetch price data, run the Optuna parameter sweep, and produce final-sim CSVs/plots with candidate pool params. Use when asked to simulate a reCLAMM pool for a pair, gather candidate price_ratio/margin/shift params, or add a pair to run_final_sims.py.
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

- **Binance-listed token**: `python scripts/download_data.py <TOKEN>`.
- **Not on Binance**: copy `scripts/prepare_bold_usdc_data.py` (CoinGecko):
  daily close+volume → minute grid by forward-fill, daily volume spread /1440
  (so the daily resample recovers real volume), stables as flat $1.00 peg.
  CoinGecko free tier = **last 365 days only** — this bounds the earliest
  simulation start and usually forces a per-pair train window (Step 4).

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

## Gotchas

- `--method` on `run_final_sims.py` is currently a no-op.
- First run per (pool, window) builds noise arrays (needs network-free local
  parquets only); subsequent runs hit the `_sim_arrays` cache.
- Stable pairs: expect PR near the 1.01 floor and near-zero fees under the
  median-fallback noise — re-anchor (Step 2) before trusting fee numbers.
- Sweeps and final sims are pure JAX forward passes — a laptop handles demo
  scale; production sweeps want the parallel `run_full_sweep.sh` machinery.
