"""Reproducible headline run for Study 697 — Wolfe Waves.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached daily basket tapes under
``_cache/`` if present (fetching once on a miss), and always runs the synthetic positive
control with no network.

    python examples/verify.py

Headline is the **pooled** basket test (more wedges = more power); SPY is the named single
tape. All real-tape runs are trimmed to the as-of date (last complete calendar month).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quantlab.repro import data_stamp  # noqa: E402

from wolfe_waves import data, strategy as st  # noqa: E402

PCT = 0.04
MAX_HORIZON = 90
N_DRAWS = 1000
SEED = 697

print("# Wolfe Waves — does the EPA target get hit more than a same-distance random target?")
print(f"# ZigZag reversal threshold: {PCT*100:.0f}%  |  target-hit walk-forward horizon: "
      f"{MAX_HORIZON} bars  |  as-of {data.AS_OF}")

have = [tk for tk in data.TICKERS if data.have_real(tk)]
print(f"# basket cached: {have}")
if not have:
    print("(cache miss — fetching the basket once)")
    data.fetch_all()
    have = list(data.TICKERS)

basket = data.load_basket(have)
for tk in have:
    print(data_stamp(f"{tk} OHLC", basket[tk], cols=["close"], asof=data.AS_OF))

# --------------------------------------------------------------------------- #
# THE HEADLINE — pooled target-hit test vs the same-distance random-day placebo
# --------------------------------------------------------------------------- #
print("\n# THE HEADLINE — pooled EPA target-hit test (basket)")
pooled = st.pooled_target_hit(basket, pct=PCT, max_horizon=MAX_HORIZON)
ledger = pooled["ledger"]
for tk, c in pooled["counts"].items():
    print(f"  {tk:>7}: pivots={c['pivots']:>4}  Wolfe candidates={c['entries']:>3}")
print(f"  pooled candidates: {len(ledger)}")

hr = st.hit_rate_summary(ledger)
print(f"  decided (win/loss, timeouts excluded): {hr['n_decided']} "
      f"(timeouts: {hr['n_timeout']})")
print(f"  hit rate: {hr['n_win']}/{hr['n_decided']} = {hr['hit_rate']*100:.1f}%  "
      f"(Wilson 95% [{hr['wilson_lo']*100:.1f}%, {hr['wilson_hi']*100:.1f}%])")

per_ticker_ledgers = {tk: ledger[ledger["ticker"] == tk].reset_index(drop=True) for tk in have}
placebo = st.pooled_random_target_placebo(basket, per_ticker_ledgers, max_horizon=MAX_HORIZON,
                                          n_draws=N_DRAWS, seed=SEED)
print(f"\n  same-distance random-day placebo ({N_DRAWS} draws/ticker, pooled):")
print(f"    observed hit rate {placebo['obs_hit_rate']*100:.1f}%  vs  placebo mean "
      f"{placebo['placebo_mean']*100:.1f}% (sd {placebo['placebo_sd']*100:.1f}pp)")
print(f"    z = {placebo['z']:+.2f}   p = {placebo['p_value']:.3f}")

tta = st.time_target_accuracy(ledger)
print(f"\n  the EPA TIME target (point5 + (point4-point1) bars), winning trades only "
      f"(n={tta['n']}):")
print(f"    actual bars-to-hit: mean {tta['mean_actual']:.1f}   "
      f"projected: mean {tta['mean_predicted']:.1f}   "
      f"corr(actual, projected) = {tta['corr']:+.3f}   MAE = {tta['mae_days']:.1f} bars")

# --------------------------------------------------------------------------- #
# SECONDARY — fixed-horizon directional return (geometry-free robustness cut)
# --------------------------------------------------------------------------- #
print("\n# SECONDARY — fixed-horizon directional return (pooled basket, no stop/target geometry)")
rng = np.random.default_rng(SEED)
rets = {h: [] for h in st.HORIZONS}
dirs = {h: [] for h in st.HORIZONS}
for tk in have:
    b = basket[tk]
    piv = st.zigzag(b["close"].to_numpy(float), pct=PCT)
    ent = st.wolfe_candidates(piv)
    for h in st.HORIZONS:
        led = st.run_trades(b, ent, h, cost_bps=0.0)
        rets[h].append(led["ret_gross"].to_numpy())
        dirs[h].append(led["dir"].to_numpy())

print(f"  {'H':>3} {'n':>5} {'gross_bps':>9} {'net_bps':>8} {'win%':>5} "
      f"{'t':>6} {'hac_t':>6} {'coin_p':>7}")
for h in st.HORIZONS:
    r = np.concatenate(rets[h])
    d = np.concatenate(dirs[h])
    s = st.summarize(pd.DataFrame({"ret_gross": r, "ret_net": r - 2e-4}), "ret_gross")
    sn = st.summarize(pd.DataFrame({"ret_gross": r, "ret_net": r - 2e-4}), "ret_net")
    unsigned = r * d
    draws = np.array([(rng.choice([-1, 1], size=len(unsigned)) * unsigned).mean()
                      for _ in range(N_DRAWS)])
    p = float((draws >= r.mean()).mean())
    print(f"  {h:>3} {s['n_trades']:>5} {s['mean_bps']:>9.0f} {sn['mean_bps']:>8.0f} "
          f"{s['win_rate']*100:>4.0f} {s['t']:>6.2f} {s['hac_t']:>6.2f} {p:>7.3f}")

# --------------------------------------------------------------------------- #
# ROBUSTNESS — ZigZag threshold sweep (headline metric: pooled hit rate at H=90)
# --------------------------------------------------------------------------- #
print("\n# ROBUSTNESS — ZigZag reversal threshold sweep")
for pct in (0.03, 0.04, 0.05, 0.08):
    p2 = st.pooled_target_hit(basket, pct=pct, max_horizon=MAX_HORIZON)
    h2 = st.hit_rate_summary(p2["ledger"])
    print(f"  pct={pct:.2f}: candidates={len(p2['ledger']):>4}  decided={h2['n_decided']:>4}  "
          f"hit_rate={h2['hit_rate']*100:>5.1f}%")

# --------------------------------------------------------------------------- #
# SPY-only — the named single tape (lower power, same question)
# --------------------------------------------------------------------------- #
print("\n# SPY-only (named single tape)")
spy = basket["SPY"]
piv = st.zigzag(spy["close"].to_numpy(float), pct=PCT)
ent = st.wolfe_candidates(piv)
led_spy = st.target_hit_ledger(spy, ent, max_horizon=MAX_HORIZON)
hr_spy = st.hit_rate_summary(led_spy)
print(f"  candidates={len(ent)}  decided={hr_spy['n_decided']}  "
      f"hit_rate={hr_spy['hit_rate']*100:.1f}% "
      f"(Wilson [{hr_spy['wilson_lo']*100:.1f}%, {hr_spy['wilson_hi']*100:.1f}%])")

# --------------------------------------------------------------------------- #
# Synthetic positive control — deterministic, no network
# --------------------------------------------------------------------------- #
print("\n# Synthetic positive control — deterministic, no network (H=20 fixed-horizon return)")
print("  the detector must NOT manufacture significance on a pure random-walk null (edge=0)")
print("  and must light up on a planted post-point-5 reversal (edge=0.30). Null over 20 seeds.")
null_ts = []
for s_ in range(20):
    bars, _ = data.synthetic_panel(edge=0.0, seed=SEED + s_, n_days=9000)
    piv = st.zigzag(bars["close"].to_numpy(float), pct=PCT)
    ent = st.wolfe_candidates(piv)
    led = st.run_trades(bars, ent, horizon=20, cost_bps=0.0)
    null_ts.append(st.summarize(led, "ret_gross")["t"])
null_ts = np.asarray(null_ts)
print(f"  null (edge=0), 20 seeds: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(np.abs(null_ts) >= 2).sum()}/20 seeds")

print("  planted edge=0.30 (seed 697), by horizon:")
bars, _ = data.synthetic_panel(edge=0.30, seed=SEED, n_days=9000)
piv = st.zigzag(bars["close"].to_numpy(float), pct=PCT)
ent = st.wolfe_candidates(piv)
for h in st.HORIZONS:
    led = st.run_trades(bars, ent, horizon=h, cost_bps=0.0)
    s = st.summarize(led, "ret_gross")
    print(f"    H={h:>2}: n={s['n_trades']:>3}  mean={s['mean_bps']:>6.0f} bps  "
          f"t={s['t']:>6.2f}  hac_t={s['hac_t']:>6.2f}")
