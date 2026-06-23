"""Reproducible headline run for Study 370 — Zero-DTE Options & the SPX tape.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached SPY OHLC + option-chain snapshot
under ``_cache/`` if present (the real-tape numbers), and always runs the synthetic pinning
control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from zero_dte_options import data, strategy as st

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "..", "_cache")

print("# Zero-DTE Options & the SPX tape — daily proxy (intraday open->close leg) + chain snapshot")

if data.have_real():
    F = data.load_real()
    bd = data.BREAK_DATE
    span = (F.index.max() - F.index.min()).days / 365.25
    print(f"tape days      : {len(F)}  ({F.index.min().date()} -> {F.index.max().date()}, "
          f"{span:.1f} years)")
    print(f"break date     : {bd.date()}  (same-day options went daily through 2022)")

    print("\n# Signal — lag-1 autocorr of the intraday leg, pre vs post 2022 (PROXY)")
    for col in ("intraday", "oc_ret"):
        rt = st.regime_table(F, bd, col)
        w = st.welch_t_autocorr(F, bd, col)
        pb = st.placebo_break_pvalue(F, col) if col == "intraday" else None
        tail = f"  placebo_p={pb['p_value']:.3f}" if pb else ""
        print(f"  {col:9s}: ac_pre={rt['ac_pre']:+.4f} ac_post={rt['ac_post']:+.4f} "
              f"d_ac={rt['d_ac']:+.4f}  Welch_t={w['t']:+.2f}{tail}")

    print("\n# Robustness — vary the assumed break date (effect grows only as you fish later)")
    for y in ("2021-01-01", "2022-01-01", "2022-05-01", "2023-01-01"):
        b = pd.Timestamp(y)
        rt = st.regime_table(F, b, "intraday"); w = st.welch_t_autocorr(F, b, "intraday")
        print(f"  break={y}: d_ac={rt['d_ac']:+.4f}  t={w['t']:+.2f}  n_post={rt['n_post']}")

    print("\n# Tradability — fade the prior intraday leg (1-day lag, 1bp one-way x turnover)")
    mr = st.mean_reversion_trade(F, bd, "intraday", cost_bps=1.0)
    for seg in ("all", "pre", "post"):
        s = mr[seg]
        print(f"  {seg:4s}: n={s['n']:>4}  gross={s['gross_mean']*1e4:+.2f}bp  "
              f"net@1bp={s['net_mean']*1e4:+.2f}bp  win={s['win']*100:.1f}%  "
              f"net_Sharpe={s['sharpe_net']:+.2f}")

    meta_path = os.path.join(CACHE, "option_meta.json")
    if data.have_option_chain() and os.path.exists(meta_path):
        meta = json.load(open(meta_path))
        ch = data.load_option_chain()
        cc = st.chain_concentration(ch, meta["spot"], band=0.01)
        print("\n# 0DTE chain snapshot — the boom is real (current snapshot only)")
        print(f"  as-of expiry {meta['snapshot_expiry']}  spot ~{meta['spot']}  "
              f"listed expiries={meta['n_expiries']}")
        print(f"  total volume={cc['total_volume']:,.0f}  within +-1% of spot="
              f"{cc['share_near']*100:.1f}%")
else:
    print("(no _cache/spy_ohlc.csv — run data.fetch_spy() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  detector must find a PLANTED pinning regime switch and must NOT invent one (pin=0).")
for pin in (0.0, 0.30):
    syn = data.synthetic_tape(n_days=4000, pin=pin, seed=370)
    sw = syn.index[2000]
    rt = st.regime_table(syn, sw, "intraday")
    w = st.welch_t_autocorr(syn, sw, "intraday")
    pb = st.placebo_break_pvalue(syn, "intraday", break_date=sw)
    print(f"  pin={pin:.2f}: ac_pre={rt['ac_pre']:+.4f} ac_post={rt['ac_post']:+.4f} "
          f"d_ac={rt['d_ac']:+.4f}  Welch_t={w['t']:+.2f}  placebo_p={pb['p_value']:.3f}")
