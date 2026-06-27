"""Real-tape verification -- Study 506 (Industry-Momentum). Regenerates docs/results.md numbers.

Reads the study's own yfinance cache (studies/506-industry-momentum/_cache/*.parquet), builds
the 12-1 momentum long-short on the 11 SPDR sector ETFs (industry momentum) and on the
large-cap survivor basket (single-name momentum), and prints every number, gross and net, with
a Newey-West HAC t and a label-shuffle placebo p-value.

    python studies/506-industry-momentum/examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from industry_momentum import data, strategy as st  # noqa: E402

# Costs: 5 bps/leg one-way, 50 bps/yr borrow on the short leg (liquid ETFs / large-caps).
COST_BPS = 5.0
BORROW_BPS = 50.0
TOP_K_SEC = 3       # 3 winners / 3 losers out of 11 sectors
TOP_K_NAME = 8      # 8 winners / 8 losers out of ~40 names


def _print_summary(label: str, s: dict) -> None:
    print(f"=== {label} ===")
    print(f"  N months:          {s['n']}")
    print(f"  Mean (ann):        {s['mean']*100:+.2f}%/yr")
    print(f"  Vol (ann):         {s['vol']*100:.2f}%/yr")
    print(f"  Sharpe (ann):      {s['sharpe']:+.3f}")
    print(f"  HAC t-stat:        {s['tstat']:+.3f}")
    print(f"  Hit rate:          {s['hit_rate']*100:.1f}%")
    print(f"  Max drawdown:      {s['max_drawdown']*100:.1f}%")


def main() -> None:
    sec_px, spy, names_px = data.fetch_prices()
    if sec_px.empty:
        print("Cache not found. Run with fetch=True and network access to populate the cache.")
        return

    sec_ret = data.drop_partial_last(data.to_monthly_returns(sec_px))
    name_ret = data.drop_partial_last(data.to_monthly_returns(names_px))
    spy_ret = data.drop_partial_last(data.to_monthly_returns(spy))

    # SPY common-window restriction: sector ETFs include XLRE(2015)/XLC(2018); the engine
    # already drops NaN scores, so each sector enters once it has 13 months of history.
    print("Study 506 -- Industry-Momentum -- real yfinance tape\n")
    print(f"Sector ETFs:      {sec_px.shape} fp={data.fingerprint(sec_px)}")
    print(f"SPY:              {spy.shape} fp={data.fingerprint(spy)}")
    print(f"Single-names:     {names_px.shape} fp={data.fingerprint(names_px)}")
    print(f"Daily range:      {sec_px.index[0].date()} -- {sec_px.index[-1].date()}")
    print(f"Monthly sectors:  {sec_ret.index[0].date()} -- {sec_ret.index[-1].date()} ({len(sec_ret)} months)")
    print("As-of: 2026-06-26 | Source: yfinance daily adjusted-close (auto_adjust=True)\n")

    # ---- Industry momentum (sector ETFs) ----
    sec_g = st.long_short(sec_ret, top_k=TOP_K_SEC)
    sec_n = st.long_short(sec_ret, top_k=TOP_K_SEC, cost_bps=COST_BPS, borrow_ann_bps=BORROW_BPS)
    s_sec_g = st.summary(sec_g["ls_gross"])
    s_sec_n = st.summary(sec_n["ls_net"])
    _print_summary("INDUSTRY momentum (sector ETFs, gross)", s_sec_g)
    _print_summary("INDUSTRY momentum (sector ETFs, NET 5bps+50bps borrow)", s_sec_n)
    print(f"  Avg turnover/leg:  {sec_g['turnover'].mean()*100:.1f}%/mo")

    pb_sec = st.placebo_pvalue(sec_ret, real_mean=float(sec_g["ls_gross"].mean()),
                               top_k=TOP_K_SEC, n_shuffles=500)
    print(f"  Placebo p-value:   {pb_sec['p_value']:.3f} "
          f"(null mean {pb_sec['null_mean']*100:+.2f}%/yr +/- {pb_sec['null_std']*100:.2f}, "
          f"{pb_sec['n_shuffles']} shuffles)\n")

    # ---- Single-name momentum (large-cap survivor basket) ----
    nm_g = st.long_short(name_ret, top_k=TOP_K_NAME)
    nm_n = st.long_short(name_ret, top_k=TOP_K_NAME, cost_bps=COST_BPS, borrow_ann_bps=BORROW_BPS)
    s_nm_g = st.summary(nm_g["ls_gross"])
    s_nm_n = st.summary(nm_n["ls_net"])
    _print_summary("SINGLE-NAME momentum (large-cap survivors, gross)", s_nm_g)
    _print_summary("SINGLE-NAME momentum (large-cap survivors, NET)", s_nm_n)
    print(f"  Avg turnover/leg:  {nm_g['turnover'].mean()*100:.1f}%/mo")

    pb_nm = st.placebo_pvalue(name_ret, real_mean=float(nm_g["ls_gross"].mean()),
                              top_k=TOP_K_NAME, n_shuffles=500)
    print(f"  Placebo p-value:   {pb_nm['p_value']:.3f} "
          f"(null mean {pb_nm['null_mean']*100:+.2f}%/yr +/- {pb_nm['null_std']*100:.2f}, "
          f"{pb_nm['n_shuffles']} shuffles)\n")

    # ---- Long-only winner sleeves vs SPY (context) ----
    spy_s = st.summary(spy_ret.reindex(sec_g.index).dropna())
    print("=== Context: long winners only vs SPY ===")
    print(f"  Sector winners (ann):  {st.summary(sec_g['long_ret'])['mean']*100:+.2f}%/yr")
    print(f"  Sector losers  (ann):  {st.summary(sec_g['short_ret'])['mean']*100:+.2f}%/yr")
    print(f"  SPY (ann):             {spy_s['mean']*100:+.2f}%/yr\n")

    # ---- Robustness: top_k sweep on sectors ----
    print("=== Robustness: sector top_k sweep (gross) ===")
    for k in (1, 2, 3, 4):
        r = st.long_short(sec_ret, top_k=k)
        s = st.summary(r["ls_gross"])
        print(f"  top_k={k}: mean {s['mean']*100:+.2f}%/yr, HAC t {s['tstat']:+.3f}, Sharpe {s['sharpe']:+.3f}")
    print()

    # ---- Synthetic positive control (seed-robust) ----
    print("=== Synthetic positive control (seed-averaged over 20 seeds) ===")
    ctrl = st.control_sweep(data, moms=[0.0, 0.06, 0.12, 0.20], n_years=18, top_k=3, n_seeds=20)
    for _, row in ctrl.iterrows():
        print(f"  industry_mom={row['industry_mom']:.2f}: "
              f"mean {row['mean_ann']*100:+.2f}%/yr, seed-avg HAC t {row['tstat']:+.3f}")
    print()

    print("NOTE Survivorship: the single-name basket is large-cap names still trading in 2026 ")
    print("     (loser leg is an upper bound). The sector-ETF book is survivorship-free.")
    print("     One execution lag: signal at close of t, hold realised t+1 return.")
    print("     Costs: 5 bps/leg one-way x turnover + 50 bps/yr borrow on the short leg.")


if __name__ == "__main__":
    main()
