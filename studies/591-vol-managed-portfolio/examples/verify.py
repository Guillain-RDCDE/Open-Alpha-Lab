"""Reproducible headline run for Study 591 — Vol-Managed Portfolio (Moreira-Muir 2017).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached yfinance tape under
``_cache/`` if present (the real-tape numbers), and always runs the synthetic control
with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np

from vol_managed_portfolio import data, strategy as st

N_PLACEBO = 200

print("# Vol-Managed Portfolio (Moreira-Muir 2017) — SPY headline + QQQ/EFA/IWM robustness")
print(f"rule: w(m+1) = min({st.CAP}, expanding-mean(RV)/RV(m)); monthly rebalance, one clean lag")
print(f"as-of {data.AS_OF} (last complete month)\n")

if data.have_real():
    try:
        from quantlab.repro import fingerprint
        fp = fingerprint(data.load_prices())
    except Exception:
        fp = "n/a"

    # ---------------- headline: SPY ----------------
    df = data.complete_months(data.daily_frames("SPY"))
    tbl = st.monthly_table(df)
    print(f"SPY tape        : {df.index.min().date()} -> {df.index.max().date()}  "
          f"({len(df):,} days, {len(tbl)} complete months)   fingerprint {fp}")

    g = st.race_summary(tbl["exc"].values, tbl["rv"].values)
    print(f"strategy months : {g['n_months']} (after {st.BURN_IN}-month RV burn-in)  "
          f"avg weight {g['avg_w']:.2f}  capped-at-{st.CAP}x share {g['share_capped']*100:.1f}%  "
          f"avg |dw|/mo {g['avg_turnover']:.3f}")

    print("\n# The race — excess-vs-excess, gross")
    print(f"  Sharpe (ann)  : managed {g['sharpe_managed']:.3f}  vs  buy&hold {g['sharpe_bh']:.3f}")
    print(f"  HAC alpha     : {g['reg_alpha_ann_pct']:+.2f}%/yr   NW t = {g['reg_t_alpha']:+.3f}  "
          f"(lags={g['reg_lags']})")
    print(f"  beta          : {g['reg_beta']:.3f}   appraisal ratio = {g['reg_appraisal']:.3f}  "
          f"(resid vol {g['reg_resid_vol_ann_pct']:.1f}%/yr)")

    print("\n# Net of costs (one-way bps x |dw| x NAV + retail borrow spread on w>1)")
    for cb, sp in ((0.0, 0.0), (5.0, 0.01), (10.0, 0.02)):
        n = st.race_summary(tbl["exc"].values, tbl["rv"].values,
                            cost_bps=cb, borrow_spread_ann=sp)
        print(f"  {cb:>4.0f} bps + {sp*100:.0f}% spread: alpha {n['reg_alpha_ann_pct']:+.2f}%/yr  "
              f"t = {n['reg_t_alpha']:+.2f}   Sharpe {n['sharpe_managed']:.3f} vs B&H {n['sharpe_bh']:.3f}")

    print(f"\n# Placebo — shuffle the monthly RV signal, {N_PLACEBO} seeds, full rule rebuilt per seed")
    pl = st.placebo_shuffle(tbl["exc"].values, tbl["rv"].values, n_seeds=N_PLACEBO)
    print(f"  observed alpha {pl['obs_alpha_ann_pct']:+.2f}%/yr (t={pl['obs_t']:+.2f})  vs  shuffled: "
          f"mean t {pl['placebo_mean_t']:+.2f} (sd {pl['placebo_sd_t']:.2f}), "
          f"mean alpha {pl['placebo_mean_alpha']:+.2f}%/yr")
    print(f"  p[shuffled alpha >= observed] = {pl['p_alpha']:.3f}")

    print("\n# Third axis — does it dodge crashes? (max DD, monthly total-return NAVs)")
    ct = st.crash_table(tbl)
    for k, v in ct.items():
        print(f"  {k:<12}: managed {v['managed']*100:+.1f}%   buy&hold {v['bh']*100:+.1f}%")

    print("\n# Robustness — same rule on QQQ / EFA / IWM")
    for tk in ("QQQ", "EFA", "IWM"):
        d2 = data.complete_months(data.daily_frames(tk))
        t2 = st.monthly_table(d2)
        r2 = st.race_summary(t2["exc"].values, t2["rv"].values)
        print(f"  {tk}: {r2['n_months']} mo  alpha {r2['reg_alpha_ann_pct']:+.2f}%/yr  "
              f"t = {r2['reg_t_alpha']:+.2f}   Sharpe {r2['sharpe_managed']:.3f} "
              f"vs B&H {r2['sharpe_bh']:.3f}")

    print("\n# Robustness — the cap")
    for cap in (1.0, 1.5, 2.0):
        rc = st.race_summary(tbl["exc"].values, tbl["rv"].values, cap=cap)
        print(f"  cap {cap:.1f}x: alpha {rc['reg_alpha_ann_pct']:+.2f}%/yr  t = {rc['reg_t_alpha']:+.2f}  "
              f"Sharpe {rc['sharpe_managed']:.3f} vs B&H {rc['sharpe_bh']:.3f}")
else:
    print("(no _cache/vmp_prices.csv — run data.fetch_all() once to build the cache)")

print("\n# Synthetic control — deterministic, no network, 20 seeds averaged, 30y worlds")
print("  NULL (d=0, mean scales with variance — risk fully priced): overlay must NOT win.")
print("  FLAT-MEAN (d=1) and PLANTED LEVERAGE-EFFECT (d=2, mean falls as variance")
print("  rises — the Moreira-Muir disconnect): the harness must bank the planted effect.")
for d, label in ((0.0, "NULL     "), (1.0, "FLAT-MEAN"), (2.0, "LEVERAGE ")):
    s = st.synthetic_check(disconnect=d, n_seeds=20)
    print(f"  {label} disconnect={d:.0f}: mean HAC alpha t = {s['mean_t']:+.2f} "
          f"(sd {s['sd_t']:.2f}), mean alpha {s['mean_alpha_ann_pct']:+.2f}%/yr, "
          f"share of seeds with t>=2: {s['share_t_ge_2']*100:.0f}%")
