"""Real-panel verification -- Study 512 (High-Volume-Return-Premium). Prints every number.

Reads the study's own yfinance cache (studies/512-high-volume-return-premium/_cache/), forms
the weekly abnormal-volume sort (high-volume quintile long, low-volume quintile short, one
execution lag), and reports the long-short with one-sample & HAC t, a label-shuffle placebo
p-value, gross-vs-net costs, a horizon robustness sweep, and the synthetic positive control.

    python studies/512-high-volume-return-premium/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from high_volume_return_premium import data, strategy as st  # noqa: E402

# Cost assumptions (stated): large-cap one-way bps and an annual short-borrow.
COST_BPS = 5.0          # one-way, per leg, per rebalance
BORROW_ANN_BPS = 50.0   # annual borrow on the low-volume short leg
ASOF = "2026-06-26"


def main() -> None:
    ret, vol = data.fetch_panel()
    if ret.empty:
        print("Cache not found. Run fetch_panel(fetch=True) from a network-enabled host.")
        return
    ret = data.drop_partial_last_week(ret)
    vol = vol.reindex(ret.index)

    print("Study 512 -- High-Volume-Return-Premium -- real yfinance panel\n")
    print(f"Returns panel: {ret.shape}, fingerprint={data.fingerprint(ret)}")
    print(f"Volume panel:  {vol.shape}, fingerprint={data.fingerprint(vol)}")
    print(f"Date range: {ret.index[0].date()} -- {ret.index[-1].date()}")
    print(f"As-of: {ASOF} | Source: yfinance daily adjusted close + raw volume\n")

    # --- Headline: 1-week-horizon long-short, gross and net ---
    df = st.long_short(ret, vol, horizon=1, cost_bps=0.0)
    df_net = st.long_short(ret, vol, horizon=1, cost_bps=COST_BPS, borrow_ann_bps=BORROW_ANN_BPS)
    s_gross = st.summary(df["ls_gross"])
    s_net = st.summary(df_net["ls_net"])
    s_long = st.summary(df["long_ret"])
    s_short = st.summary(df["short_ret"])

    print("=== Long-short (high-vol quintile long, low-vol quintile short, horizon=1wk) ===")
    print(f"  N weeks:             {s_gross['n']}")
    print(f"  GROSS mean (ann):    {s_gross['mean']*100:+.2f}%/yr")
    print(f"  GROSS vol (ann):     {s_gross['vol']*100:.2f}%/yr")
    print(f"  GROSS Sharpe:        {s_gross['sharpe']:+.3f}")
    print(f"  GROSS one-sample t:  {s_gross['t_os']:+.3f}")
    print(f"  GROSS HAC t:         {s_gross['t_hac']:+.3f}")
    print(f"  GROSS hit rate:      {s_gross['hit_rate']*100:.1f}%")
    print(f"  GROSS max drawdown:  {s_gross['max_dd']*100:.1f}%")
    print(f"  NET mean (ann):      {s_net['mean']*100:+.2f}%/yr  (cost={COST_BPS}bps/leg, "
          f"borrow={BORROW_ANN_BPS}bps/yr)")
    print(f"  NET one-sample t:    {s_net['t_os']:+.3f}")
    print(f"  NET HAC t:           {s_net['t_hac']:+.3f}")

    print("\n=== Legs (forward 1-week return of each quintile) ===")
    print(f"  High-volume leg (long, ann):  {s_long['mean']*100:+.2f}%/yr")
    print(f"  Low-volume leg (short, ann):  {s_short['mean']*100:+.2f}%/yr")
    turn = st.avg_turnover(ret, vol, horizon=1)
    print(f"  Avg one-sided long-leg turnover/week: {turn*100:.1f}%")

    # --- Placebo / label-shuffle null ---
    placebo = st.placebo_pvalue(ret, vol, real_mean=float(df["ls_gross"].mean()),
                                n_shuffles=200, horizon=1)
    print("\n=== Label-shuffle placebo (permute volume labels across names each week) ===")
    print(f"  N shuffles:          {placebo['n_shuffles']}")
    print(f"  Null mean (weekly):  {placebo['null_mean']*100:+.4f}%")
    print(f"  Null std (weekly):   {placebo['null_std']*100:.4f}%")
    print(f"  Real mean (weekly):  {float(df['ls_gross'].mean())*100:+.4f}%")
    print(f"  Placebo p-value:     {placebo['p']:.3f}")

    # --- Horizon robustness sweep ---
    print("\n=== Horizon robustness (gross long-short) ===")
    print("  horizon(wk) | mean%/yr |  t_os  |  Sharpe")
    horizons = {}
    for h in (1, 2, 4, 8):
        dh = st.long_short(ret, vol, horizon=h, cost_bps=0.0)
        if dh.empty:
            continue
        sh = st.summary(dh["ls_gross"])
        horizons[h] = sh
        print(f"     {h:>2d}       | {sh['mean']*100:+7.2f}  | {sh['t_os']:+.3f} | {sh['sharpe']:+.3f}")

    # --- Seed robustness of the placebo p (different base seeds) ---
    print("\n=== Placebo p across base seeds (robustness) ===")
    for bs in (11, 512, 2026):
        pb = st.placebo_pvalue(ret, vol, real_mean=float(df["ls_gross"].mean()),
                               n_shuffles=120, horizon=1, base_seed=bs)
        print(f"  base_seed={bs:>4d}: p={pb['p']:.3f}")

    # --- Synthetic positive control (faithful-engine check only) ---
    print("\n=== Synthetic positive control (engine faithfulness; NOT a real-tape claim) ===")
    sweep = st.sweep_synthetic()
    print(sweep.round(3).to_string(index=False))

    print("\nNOTE Survivorship bias: universe = 40 current mega/large-caps projected backwards.")
    print("     Delisted / failed names (natural volume-spike candidates) are absent ->")
    print("     positive results are UPPER-BOUND estimates. One execution lag (enter next week).")


if __name__ == "__main__":
    main()
