"""Reproducible headline run for Study 763 — Puell-Multiple.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached BTC-USD tape under ``_cache/``
(fetching once on a cache miss). The Puell Multiple is a pure function of that tape and the
hardcoded halving schedule — no network, no digitised proxy. The synthetic control always runs.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from puell_multiple import data, strategy as st  # noqa: E402

print("# Puell-Multiple — does daily miner issuance vs its 365-day average time BTC tops/bottoms?")

if not data.have_real():
    print("(cache miss — fetching BTC-USD once)")
    data.fetch_btc(end="2026-07-01")

btc = data.load_btc()
print(data_stamp("BTC-USD daily close", btc.to_frame(), asof=data.AS_OF))

pm = data.puell_multiple(btc).dropna()
print(f"\n# THE METRIC — Puell Multiple reconstructed from price + halving schedule (issuance-only)")
print(f"  Puell = 144 * block_reward(t) * price(t) / trailing-365d mean of that (144 cancels).")
print(f"  {len(pm)} daily points {pm.index.min().date()} -> {pm.index.max().date()}  "
      f"(min {pm.min():.2f}, median {pm.median():.2f}, max {pm.max():.2f})")
print(f"  reward steps applied: 25 (pre-2016-07-09) -> 12.5 -> 6.25 -> 3.125 (post-2024-04-20)")

print("\n# 1) PREDICTIVE REGRESSION — forward log-return on log(Puell), HAC (lag = horizon)")
for h in (30, 90, 180):
    r = st.predictive_regression(pm, btc, horizon=h)
    print(f"  +{h:>3d}d: slope {r['slope_puell']:+.4f}  HAC t = {r['t_puell']:+.2f}  "
          f"R^2 {r['r2']:.4f}  n={r['n']}")
rc = st.predictive_regression(pm, btc, horizon=90, add_price_control=True)
print(f"  horse race (+90d, add trailing-180d price momentum):  "
      f"Puell HAC t = {rc['t_puell']:+.2f}  |  price-mom HAC t = {rc['t_price']:+.2f}")

print("\n# 2) BAND FORWARD RETURNS — mean forward BTC return by Puell band vs unconditional")
print("  bands: top (Puell>=4, contrarian SELL), bottom (Puell<=0.5, contrarian BUY), neutral")
for h in (30, 90, 180):
    tab = st.state_forward_stats(pm, btc, horizon=h, high=4.0, low=0.5)
    print(f"  +{h}d:")
    for band, row in tab.iterrows():
        p = "   —  " if np.isnan(row["placebo_p"]) else f"{row['placebo_p']:.3f}"
        print(f"    {band:<13}: {row['mean_pct']:+7.2f}%  (median {row['median_pct']:+7.2f}%, "
              f"hit {row['hit']:.2f}, n={int(row['n']):>4d})  Welch t {row['welch_t']:+7.2f}  "
              f"placebo p {p}")

print("\n# 3) BUY-LOW/SELL-HIGH TIMER (cash when Puell>=4, else long) vs continuous buy-and-hold")
print("  one-day entry lag, 10 bps one-way x NAV per flip, price-only, long/flat (no borrow)")
tb = st.backtest_timing(pm, btc, high=4.0, cost_bps=10.0)
print(f"  timer      : total {tb['strat_total_pct']:+10.1f}%  CAGR {tb['strat_cagr_pct']:+6.2f}%  "
      f"Sharpe {tb['strat_sharpe']:+.3f}  (exposed {tb['exposure_pct']:.1f}% of the time, "
      f"{tb['n_flips']} flips)")
print(f"  buy & hold : total {tb['bh_total_pct']:+10.1f}%  CAGR {tb['bh_cagr_pct']:+6.2f}%  "
      f"Sharpe {tb['bh_sharpe']:+.3f}  (exposed 100.0% of the time)")
print(f"  timer - buy&hold: {tb['excess_cagr_pct']:+.2f}%/yr  (HAC t = {tb['excess_t']:+.2f})")
print("  threshold sweep (total return, exposure, Sharpe):")
for hi in (3.0, 3.5, 4.0, 4.5):
    t = st.backtest_timing(pm, btc, high=hi, cost_bps=10.0)
    print(f"    Puell>={hi}: {t['strat_total_pct']:+10.1f}%  exp {t['exposure_pct']:5.1f}%  "
          f"SR {t['strat_sharpe']:.2f}   (buy&hold {t['bh_total_pct']:+.1f}%, SR {t['bh_sharpe']:.2f})")

print("\n# 4) SYNTHETIC POSITIVE CONTROL — the regression must find a planted contrarian link")
pu0, pr0 = data.synthetic_world(beta=0.0, seed=763)
pu1, pr1 = data.synthetic_world(beta=0.5, seed=763)
r0 = st.synthetic_detect(pu0, pr0, horizon=30)
r1 = st.synthetic_detect(pu1, pr1, horizon=30)
print(f"  planted contrarian beta=0.50 (seed 763): slope {r1['slope_puell']:+.3f}  "
      f"HAC t = {r1['t_puell']:+.2f}")
null_t = []
for s in range(20):
    pu, pr = data.synthetic_world(beta=0.0, seed=763 + s)
    null_t.append(st.synthetic_detect(pu, pr, horizon=30)["t_puell"])
null_t = np.asarray(null_t)
print(f"  null beta=0.00, 20 seeds: mean HAC t = {null_t.mean():+.2f} (sd {null_t.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_t) >= 2).sum()}/20 seeds")
print(f"\nfingerprint (BTC-USD last row): {data.fingerprint(btc.to_frame())}")
