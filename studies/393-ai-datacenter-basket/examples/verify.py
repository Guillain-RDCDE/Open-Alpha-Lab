"""Reproducible headline run for Study 393 (AI-Datacenter-Basket).

Prints every number quoted in docs/results.md and frozen into notebooks/build_notebooks.py
(the ``R`` dict). Offline & deterministic: the real-tape panel is read from ``_cache/`` (populated
once via load_real(fetch=True, ...)); the random-basket distribution and the synthetic control are
fixed-seed.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ai_datacenter_basket import data, strategy as st

print("# AI-Datacenter-Basket — riding the AI build-out by buying the picks-and-shovels?\n")

if data.have_real():
    # --- Real tape (cache-first; look-ahead opt-in is the subject of the study) --------------
    rets, bench = data.load_real(allow_lookahead_selection=True)
    print("# Real tape — yfinance monthly total return, datacenter/power field + SPY + QQQ")
    print(f"window         : {rets.index.min().strftime('%Y-%m')} -> "
          f"{rets.index.max().strftime('%Y-%m')}  ({len(rets)} months, {len(rets)/12:.1f} years)")
    print(f"field size     : {rets.shape[1]} names  (the 8 + {rets.shape[1]-8} also-rans)")
    print(f"fingerprint    : {data.fingerprint(rets)}")

    R = st.race(rets, bench, data.BASKET, k=8, n_draws=2000, cost_bps=10.0,
                allow_lookahead_selection=True)

    print("\n# The race — equal-weight basket vs benchmarks (CAGR / Sharpe / maxDD)")
    for name, ser in [("AI-datacenter basket", R["basket"]), ("S&P 500 (SPY)", R["spy"]),
                      ("Nasdaq-100 (QQQ)", R["qqq"]), ("Equal-weight field", R["equal_field"]),
                      ("Ex-post '8 winners'", R["expost"])]:
        s = st.summarize(ser)
        print(f"  {name:22s} CAGR={s['cagr']*100:6.2f}%  Sharpe={s['sharpe']:.2f}  "
              f"maxDD={s['max_dd']*100:6.1f}%")
    print(f"  {'Basket net (10 bps)':22s} CAGR={R['basket_net_cagr']*100:6.2f}%")

    print("\n# Signal-axis test — HAC t-stat of the spread (monthly rebalance)")
    for label, key in [("Basket - SPY        ", "test_vs_spy"),
                       ("Basket - QQQ        ", "test_vs_qqq"),
                       ("Basket - equal-field", "test_vs_equal")]:
        t = R[key]
        print(f"  {label}: mean {t['mean_diff_ann']*100:+.1f}%/yr   HAC t = {t['tstat']:.2f}  "
              f"(n={t['n']})")

    print("\n# The decomposition — is it a theme tilt, or selection-after-the-fact?")
    print(f"  Basket CAGR spread vs SPY     : {R['basket_cagr_spread']*100:+.2f}%/yr")
    print(f"  Ex-post '8 winners' spread    : {R['expost_cagr_spread']*100:+.2f}%/yr")
    print(f"  selection reproduces          : {R['selection_share']*100:.0f}% of the basket spread")
    print(f"  ex-post winners (look-ahead)  : {R['expost_members']}")
    print(f"  basket percentile in 2000 random 8-baskets : {R['basket_pctile']:.1f}")
    print(f"  random blind 8-basket spread (mean/median)  : "
          f"{R['rand_spread'].mean()*100:+.2f}% / {np.median(R['rand_spread'])*100:+.2f}% per yr")

    print("\n# Single-name decomposition — HAC t of each member's excess over SPY")
    for tk, v in sorted(R["single_names"].items(), key=lambda kv: -kv[1]["tstat"]):
        print(f"  {tk:5s} mean excess {v['mean_ann']*100:6.1f}%/yr   HAC t = {v['tstat']:.2f}")
else:
    print("(no _cache/datacenter_panel.parquet — run "
          "data.load_real(fetch=True, allow_lookahead_selection=True) once to build it)")

print("\n# Synthetic positive control — selection manufactures a spread on a NO-EDGE tape")
print("  pre-named basket must be significant ONLY when an edge is planted; the ex-post")
print("  'pick the winners' placebo conjures a spread even at zero true edge.")
for a in (0.0, 0.012):
    rdf, b, truth = data.synthetic_panel(alpha_spread=a)
    named = st.basket_returns(rdf, truth["true_basket_cols"])
    expo = st.basket_returns(rdf, st.expost_winners(rdf, k=8, allow_lookahead_selection=True))
    sn = st.summarize(named)["cagr"] - st.summarize(b)["cagr"]
    se = st.summarize(expo)["cagr"] - st.summarize(b)["cagr"]
    tn = st.hac_tstat_diff(named, b)["tstat"]
    tag = "NULL (no true edge)" if a == 0 else "real pre-named edge"
    print(f"  alpha={a:.3f} [{tag:19s}]: pre-named basket spread {sn*100:+.2f}%/yr (t={tn:+.2f})"
          f"   ex-post placebo spread {se*100:+.2f}%/yr")
