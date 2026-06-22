"""Reproducible headline run for Study 355 — prints every number quoted in docs/results.md
and frozen into notebooks/build_notebooks.py (the ``R`` dict). Offline & deterministic: the
real-tape panel is read from ``_cache/`` (populated once via load_real(fetch=True, ...)); the
random-basket distribution and the synthetic control are fixed-seed.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from magnificent_seven import data, strategy as st

# --- Real tape (cache-first; look-ahead opt-in is the subject of the study) -----------------
rets, bench = data.load_real(allow_lookahead_selection=True)
print("# Real tape — yfinance monthly total return, mega-cap field + SPY")
print(f"window         : {rets.index.min().date()} -> {rets.index.max().date()}")
print(f"months         : {len(rets)}")
print(f"universe size  : {rets.shape[1]} names  (Mag 7 + {rets.shape[1]-7} also-rans)")
print(f"fingerprint    : {data.fingerprint(rets)}")

R = st.race(rets, bench, data.MAG7, k=7, n_draws=2000, cost_bps=10.0,
            allow_lookahead_selection=True)

print("\n# The race — equal-weight basket vs benchmark (CAGR / Sharpe / maxDD)")
for name, ser in [("Mag 7 (named)", R["mag7"]), ("S&P 500 (SPY)", R["bench"]),
                  ("Equal-weight field", R["equal_field"]),
                  ("Ex-post '7 winners'", R["expost"])]:
    s = st.summarize(ser)
    print(f"  {name:22s} CAGR={s['cagr']*100:6.2f}%  Sharpe={s['sharpe']:.2f}  "
          f"maxDD={s['max_dd']*100:6.1f}%")
print(f"  {'Mag 7 net (10 bps)':22s} CAGR={R['mag7_net_cagr']*100:6.2f}%")

print("\n# Signal-axis test — HAC t-stat of the spread (one execution model: monthly rebalance)")
tb, te = R["test_vs_bench"], R["test_vs_equal"]
print(f"  Mag 7 - SPY         : mean {tb['mean_diff_ann']*100:+.1f}%/yr   HAC t = {tb['tstat']:.2f}  (n={tb['n']})")
print(f"  Mag 7 - equal-field : mean {te['mean_diff_ann']*100:+.1f}%/yr   HAC t = {te['tstat']:.2f}  (n={te['n']})")

print("\n# The decomposition — is it a factor, or selection-after-the-fact?")
print(f"  Mag 7 CAGR spread vs SPY      : {R['mag7_cagr_spread']*100:+.2f}%/yr")
print(f"  Ex-post '7 winners' spread    : {R['expost_cagr_spread']*100:+.2f}%/yr")
print(f"  selection reproduces          : {R['selection_share']*100:.0f}% of the Mag-7 spread")
print(f"  ex-post winners (look-ahead)  : {R['expost_members']}")
print(f"  Mag 7 percentile in 2000 random 7-baskets : {R['mag7_pctile']:.1f}")
print(f"  random blind 7-basket spread (mean/median): "
      f"{R['rand_spread'].mean()*100:+.2f}% / {np.median(R['rand_spread'])*100:+.2f}% per yr")

print("\n# Synthetic positive control — selection manufactures a spread on a NO-EDGE tape")
for a in (0.0, 0.004):
    rdf, b, truth = data.synthetic_panel(alpha_spread=a)
    named = st.basket_returns(rdf, truth["true_basket_cols"])
    expo = st.basket_returns(rdf, st.expost_winners(rdf, k=7, allow_lookahead_selection=True))
    sn = st.summarize(named)["cagr"] - st.summarize(b)["cagr"]
    se = st.summarize(expo)["cagr"] - st.summarize(b)["cagr"]
    tn = st.hac_tstat_diff(named, b)["tstat"]
    tag = "NULL (no true edge)" if a == 0 else "real pre-named edge"
    print(f"  alpha={a:.3f} [{tag:19s}]: pre-named basket spread {sn*100:+.2f}%/yr (t={tn:+.2f})"
          f"   ex-post placebo spread {se*100:+.2f}%/yr")
