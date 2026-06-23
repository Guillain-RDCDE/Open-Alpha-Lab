"""Reproducible headline run for Study 373 — Dispersion-Trade.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached basket prices under
``_cache/`` if present (the real-tape numbers), and always runs the synthetic control with
no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dispersion_trade import data, strategy as st

print("# Dispersion-Trade — realized-vol proxy: SPY index vol vs avg single-name vol")
if data.have_real():
    f = data.load_real()
    span = (f.index.max() - f.index.min()).days / 365.25
    print(f"proxy days     : {len(f)}  ({f.index.min().date()} -> {f.index.max().date()}, "
          f"{span:.1f} years)")

    gs = st.gap_sign_stats(f)
    print("\n# The dispersion gap (avg single-name vol - index vol) — the identity")
    print(f"mean index vol : {gs['mean_index_vol']*100:5.1f}%")
    print(f"mean avg vol   : {gs['mean_avg_vol']*100:5.1f}%")
    print(f"mean gap       : {gs['mean_gap']*100:5.1f} vol-pts")
    print(f"gap > 0 share  : {gs['share_positive']*100:5.1f}%")
    print(f"impl. corr     : {gs['mean_impl_corr']:.3f}  ((idx/avg)^2, 1-factor)")

    pnl = st.carry_pnl(f)
    s = st.summarize_carry(pnl)
    print("\n# The carry book (long single var / short index var, trailing-63d strike, 1d lag)")
    print(f"  n={s['n']}  Sharpe={s['sharpe']:.2f}  win={s['win']*100:.0f}%  "
          f"naive_t={s['t']:.2f}  HAC_t={s['t_hac']:.2f}  placebo_p={s['p_placebo']:.3f}")
    print(f"  daily carry skew={pnl.skew():.2f}  median={pnl.median():.5f}  "
          f"worst_day={pnl.min():.4f} ({pnl.idxmin().date()})")

    c = st.net_of_costs(pnl)
    print("\n# Net of vega/bid-ask cost (5e-4 var-pt x 12 re-strikes/yr)")
    print(f"  gross_ann={c['gross_ann']*100:.1f} var-pts  net_ann={c['net_ann']*100:.1f} var-pts  "
          f"net_Sharpe={c['net_sharpe']:.2f}")

    print("\n# Robustness — realized-vol window")
    prices = data.load_prices()
    for w in (10, 21, 42, 63):
        fw = data.dispersion_proxy(prices, window=w)
        sw = st.summarize_carry(st.carry_pnl(fw))
        gw = st.gap_sign_stats(fw)["share_positive"]
        print(f"  w={w:>2}: gap+={gw*100:5.1f}%  Sharpe={sw['sharpe']:.2f}  "
              f"naive_t={sw['t']:.2f}  HAC_t={sw['t_hac']:.2f}  p={sw['p_placebo']:.3f}")
else:
    print("(no _cache/basket_prices.csv — run data.fetch_basket() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  engine must recover a PLANTED carry and must NOT manufacture a carry from the")
print("  mechanical (subadditivity) gap when the true planted carry is zero.")
for carry in (0.0, 0.0004, 0.0010):
    syn = data.synthetic_basket(n_days=4000, n_names=40, carry=carry, seed=373)
    prox = data.dispersion_proxy(syn, window=21)
    gp = st.gap_sign_stats(prox)["share_positive"]
    s = st.summarize_carry(st.synthetic_carry_pnl(syn))
    print(f"  planted carry={carry:+.4f}: gap+={gp*100:5.1f}%  mean={s['mean']:+.5f}  "
          f"Sharpe={s['sharpe']:5.2f}  HAC_t={s['t_hac']:5.2f}  p={s['p_placebo']:.3f}")
print("  implied-corr recovery:")
for bc in (0.2, 0.4, 0.6):
    syn = data.synthetic_basket(n_days=4000, carry=0.0, base_corr=bc, seed=373)
    ic = st.gap_sign_stats(data.dispersion_proxy(syn, window=21))["mean_impl_corr"]
    print(f"    planted rho={bc:.1f} -> measured impl_corr={ic:.3f}")
