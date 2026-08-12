"""Reproducible headline run for Study 867 — Currency Crash Risk.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached weekly FX panel under
``_cache/`` (fetching once on a cache miss via yfinance), and always runs the synthetic
control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from fx_crash import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")


def _fingerprint(df: pd.DataFrame) -> str:
    try:
        from quantlab.repro import fingerprint  # noqa: E402
        return fingerprint(df)
    except Exception:  # noqa: BLE001
        import hashlib
        return hashlib.sha1(
            pd.util.hash_pandas_object(df, index=True).values.tobytes()
        ).hexdigest()[:12]


print("# Currency Crash Risk — are high-carry currencies negatively skewed (BNP)?")

if not data.have_real():
    print("(cache miss — fetching the weekly FX panel once via yfinance)")
    data.fetch()

b = data.load_real()
panel = data.load_panel()
print(f"[data] {panel.shape[1]} currencies, {len(b['total_ret'])} weekly returns  "
      f"{b['total_ret'].index.min().date()} -> {b['total_ret'].index.max().date()}  "
      f"as-of {data.AS_OF}  fingerprint(panel)={_fingerprint(panel)}")
print("  SURVIVORSHIP: fixed CURRENT G10+MXN membership (no defunct/de-pegged legs) — "
      "magnitudes are an upper bound. Named on the Signal axis.")
print(f"  carry proxy (ann. %): {b['carry']}")

# ---- THE HEADLINE: the carry basket's crash skew + premium -------------------
bs = st.basket_stats(b, k=3)
print("\n# THE HEADLINE — long top-3 / short bottom-3 carry basket (dollar-neutral)")
print(f"  realized skew : {bs['skew']:+.2f}   NW(6) skew t = {bs['skew_t']:+.2f}   "
      f"(n = {bs['n_weeks']} weeks)")
print(f"  premium       : {bs['mean_bps']:+.2f} bps/wk  ({bs['ann_pct']:+.2f}%/yr)  "
      f"NW premium t = {bs['premium_t']:+.2f}  Sharpe {bs['sharpe']:.2f}")
print(f"  crash shape   : worst week {bs['worst_week_pct']:+.2f}%  "
      f"max DD {bs['max_dd_pct']:+.1f}%")

bb = st.basket_returns(b["total_ret"], b["carry"], k=3).to_numpy()
cs = st.crash_split(bb, q=0.05)
print(f"  crash split   : calm weeks {cs['calm_mean']*52*100:+.1f}%/yr vs "
      f"worst-5% weeks {cs['off_mean']*52*100:+.1f}%/yr annualised")

# ---- THE SKEW-CARRY CROSS-SECTION -------------------------------------------
reg = st.skew_carry_regression(b)
print("\n# SKEW-CARRY CROSS-SECTION — does higher carry predict more negative skew?")
sk = st.per_currency_skew(b)
print("  per-currency realized skew: " +
      ", ".join(f"{c} {sk[c]:+.2f}" for c in sorted(sk.index, key=lambda c: -b["carry"][c])))
print(f"  slope(skew on carry) = {reg['slope']:+.4f}  t = {reg['t_slope']:+.2f}  "
      f"R^2 = {reg['r2']:.2f}  Spearman = {reg['spearman']:+.3f}  (n = {reg['n']})")
ls = st.leg_skews(b, k=3)
print(f"  high-carry leg skew {ls['hi_skew']:+.2f} vs low-carry leg skew "
      f"{ls['lo_skew']:+.2f}  (diff {ls['diff']:+.2f})")

# ---- PLACEBO ----------------------------------------------------------------
print("\n# PLACEBO — shuffle which currency owns which carry (2,000 relabelings)")
pl = st.label_shuffle_placebo(b, k=3, n_perm=2000)
print(f"  observed basket skew {pl['obs_skew']:+.2f} vs placebo mean {pl['placebo_mean']:+.3f} "
      f"(sd {pl['placebo_sd']:.3f}) -> left-tail p = {pl['p_value']:.4f}")

# ---- ROBUSTNESS -------------------------------------------------------------
print("\n# ROBUSTNESS — two eras (split 2015-01-01)")
era = st.era_stats(b, split="2015-01-01", k=3)
for lbl, e in [("2004-2014", era["early"]), ("2015-2026", era["late"])]:
    print(f"  {lbl}: n={e['n_weeks']}  basket skew {e['skew']:+.2f} (NW t {e['skew_t']:+.2f})  "
          f"premium {e['ann_pct']:+.2f}%/yr (t {e['premium_t']:+.2f})")
for lo, hi, lbl in [("2003-01-01", "2015-01-01", "2004-2014"),
                    ("2015-01-01", "2026-07-01", "2015-2026")]:
    sub = {"spot_ret": b["spot_ret"][(b["spot_ret"].index >= lo) & (b["spot_ret"].index < hi)],
           "total_ret": b["total_ret"], "carry": b["carry"]}
    r = st.skew_carry_regression(sub)
    print(f"    skew-carry {lbl}: slope {r['slope']:+.3f} (t {r['t_slope']:+.2f}, "
          f"Spearman {r['spearman']:+.3f})")

# ---- THE TIMER --------------------------------------------------------------
print("\n# THE TIMER — costed carry book (2 bps/side rebalance + borrow on the short)")
for bcost in (0.0, 25.0, 50.0, 100.0):
    tm = st.timer_stats(b, k=3, cost_bps=2.0, borrow_bps_ann=bcost)
    print(f"  borrow={bcost:>5.0f} bps/yr: gross {tm['gross_ann_pct']:+.2f}% -> net "
          f"{tm['net_ann_pct']:+.2f}%/yr  Sharpe {tm['sharpe_net']:.2f}  "
          f"(t_net {tm['t_net']:+.2f}, skew {tm['skew']:+.2f})")

# ---- SYNTHETIC POSITIVE CONTROL ---------------------------------------------
print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
null_t, null_slope = [], []
for s_ in range(20):
    nb = data.synthetic_panel(edge=0.0, seed=867 + s_, n_weeks=1100)
    d = st.synthetic_detect(nb)
    null_t.append(d["skew_t"]); null_slope.append(d["slope"])
null_t = np.asarray(null_t); null_slope = np.asarray(null_slope)
print(f"  null (edge=0), 20 seeds: basket skew t mean {null_t.mean():+.2f} "
      f"(sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t) >= 2).sum()}/20; "
      f"skew-carry slope mean {null_slope.mean():+.4f}")
pb = data.synthetic_panel(edge=0.02, seed=867, n_weeks=1100)
dp = st.synthetic_detect(pb); bp = st.basket_stats(pb); rp = st.skew_carry_regression(pb)
print(f"  planted (edge=0.02, seed 867): basket skew {bp['skew']:+.2f}, "
      f"skew-carry slope {rp['slope']:+.3f} (t {rp['t_slope']:+.2f}, Spearman {rp['spearman']:+.3f})")
