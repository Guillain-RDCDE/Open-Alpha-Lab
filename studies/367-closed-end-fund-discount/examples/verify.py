"""Reproducible headline run for Study 367 — Closed-End Fund Discount.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached CEF + benchmark prices under
``_cache/`` if present (the real-tape numbers), and always runs the synthetic control with
no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from closed_end_fund_discount import data, strategy as st

print("# Closed-End Fund Discount — proxy discount (price relative to a per-fund benchmark)")
if data.have_real():
    prices = data.load_prices()
    real = data.load_real()
    disc, px = real["disc"], real["px"]
    span = (prices.index.max() - prices.index.min()).days / 365.25
    print(f"price days     : {len(prices)}  ({prices.index.min().date()} -> "
          f"{prices.index.max().date()}, {span:.1f} years)")
    print(f"CEFs usable    : {disc.shape[1]}  -> {list(disc.columns)}")
    print(f"NAV proxy      : log(price) - log(benchmark), demeaned per fund (PROXY for price/NAV discount)")

    mp = st.monthly_panels(disc, px)
    ls = st.long_short_returns(mp["disc"], mp["fret"])
    lo = st.long_only_excess(mp["disc"], mp["fret"])
    print(f"months         : {mp['disc'].shape[0]}  "
          f"({mp['disc'].index.min().date()} -> {mp['disc'].index.max().date()})")

    print("\n# Long-short: widest-discount tercile minus narrowest (1-month lag, equal-wt)")
    s = st.summarize(ls)
    print(f"  n={s['n']}  mean={s['mean']*100:.3f}%/mo  ann={s['ann_mean']*100:.2f}%  "
          f"win={s['win']*100:.0f}%  Welch_t={s['t']:.2f}  boot_p={s['p_boot']:.3f}  "
          f"Sharpe={s['sharpe']:.2f}")

    print("\n# Long-only: widest-discount tercile minus equal-weight CEF universe")
    s = st.summarize(lo)
    print(f"  n={s['n']}  mean={s['mean']*100:.3f}%/mo  ann={s['ann_mean']*100:.2f}%  "
          f"win={s['win']*100:.0f}%  Welch_t={s['t']:.2f}  boot_p={s['p_boot']:.3f}  "
          f"Sharpe={s['sharpe']:.2f}")

    print("\n# Net of costs (20 bps one-way x 2.0 turnover/mo on the long-short)")
    c = st.net_of_costs(ls, cost_bps=20.0, turnover=2.0)
    print(f"  gross={c['ann_gross']*100:.2f}%/yr  net={c['ann_net']*100:.2f}%/yr  "
          f"breakeven={c['breakeven_bps']:.1f} bps one-way")

    print("\n# Robustness — tercile fraction q")
    for q in (0.25, 1 / 3, 0.40):
        s = st.summarize(st.long_short_returns(mp["disc"], mp["fret"], q=q))
        print(f"  q={q:.2f}: n={s['n']:>3}  ann={s['ann_mean']*100:>5.2f}%  "
              f"t={s['t']:>4.2f}  p={s['p_boot']:.3f}  Sharpe={s['sharpe']:.2f}")

    print("\n# Sub-period decay — the modern era fades")
    x = ls.dropna(); h = len(x) // 2
    for name, seg in (("first half ", x.iloc[:h]), ("second half", x.iloc[h:])):
        s = st.summarize(seg)
        print(f"  {name}: n={s['n']:>3}  ann={s['ann_mean']*100:>5.2f}%  "
              f"t={s['t']:>4.2f}  Sharpe={s['sharpe']:.2f}")
else:
    print("(no _cache/cef_prices.csv — run data.fetch_cefs() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  edge=0 must NOT manufacture significance; a large planted edge MUST light up.")
for edge in (0.0, 0.30):
    syn = data.synthetic_panel(edge=edge, seed=367)
    ls_s = st.long_short_returns(syn["disc"], syn["ret"], min_names=6)
    s = st.summarize(ls_s)
    print(f"  planted edge={edge:.2f}: n={s['n']}  ann={s['ann_mean']*100:>7.2f}%  "
          f"t={s['t']:>6.2f}  p_boot={s['p_boot']:.3f}  Sharpe={s['sharpe']:.2f}")
