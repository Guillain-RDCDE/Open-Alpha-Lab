"""Reproducible headline run for Study 761 — Hotel-RevPAR.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached HST + lodging-basket tapes
under ``_cache/`` (the RevPAR proxy table is always available), and always runs the
synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hotel_revpar import data, strategy as st

print("# Hotel-RevPAR — a cited RevPAR proxy (STR/CoStar-anchored) vs hotel REITs")
if data.have_real():
    f = data.build_real("hst")
    yrs = (f.index.max() - f.index.min()).days / 365.25
    mom = st.revpar_momentum(f)
    print(f"monthly frame  : {len(f)} months  ({f.index.min().date()} -> "
          f"{f.index.max().date()}, {yrs:.1f} years)   asset = HST (flagship lodging REIT)")
    print(f"RevPAR momentum: YoY log growth, mean {mom.mean():+.3f}  std {mom.std():.3f}  "
          f"frac>0 {(mom > 0).mean():.3f}  (RevPAR is a PROXY, STR/CoStar-anchored)")

    print("\n# Forward HST returns after an UPCYCLE month (RevPAR YoY>0) vs base "
          "(1-month release lag)")
    print(f"  {'H':>3} {'n_up':>4} {'up_mean':>8} {'dn_mean':>8} {'base':>8} "
          f"{'up_win':>7} {'Welch_t':>8} {'HAC_beta':>9} {'HAC_t':>7} {'placebo_p':>10}")
    for h in st.HORIZONS:
        s = st.summarize(f, h)
        print(f"  {str(h)+'m':>3} {s['n_up']:>4} {s['up_mean']*100:>7.2f}% "
              f"{s['dn_mean']*100:>7.2f}% {s['base_mean']*100:>7.2f}% "
              f"{s['up_win']*100:>6.0f}% {s['t_welch']:>8.2f} {s['beta_hac']:>9.3f} "
              f"{s['t_hac']:>7.2f} {s['p_placebo']:>10.3f}")

    print("\n# Lead-lag corr of RevPAR momentum(t) with hotel return over [t+L, t+L+1]")
    print("#   L<0 => momentum LAGS the equity (the stock led); L>0 => momentum LEADS it")
    ll = st.lead_lag(f)
    for L in (-6, -3, 0, 3, 6):
        print(f"  L={L:+d}: corr={ll[L]:+.3f}")
    print(f"  peak correlation at L = {ll.idxmax():+d}  (a leading indicator peaks at L>0)")

    print("\n# Timing overlay — long HST when RevPAR YoY>0 (1-month lag, 10bps/turn)")
    for short in (False, True):
        b = st.timing_backtest(f, cost_bps=10.0, allow_short=short)
        tag = "long/short" if short else "long/flat "
        print(f"  {tag}: exposure={b['exposure']:.2f}  turns={b['n_turns']:.0f}  "
              f"net Sharpe={b['net']['sharpe']:.2f}  net ann={b['net']['ann_ret']*100:.1f}%  "
              f"(buy&hold Sharpe={b['buy_hold']['sharpe']:.2f})")

    print("\n# Robustness — shift the UPCYCLE threshold (12-month horizon)")
    for thr in (-0.05, 0.0, 0.05, 0.10):
        s = st.summarize(f, 12, thr=thr)
        print(f"  thr={thr:+.2f}: n_up={s['n_up']:>3}  up12={s['up_mean']*100:>5.1f}%  "
              f"Welch_t={s['t_welch']:>5.2f}  HAC_t={s['t_hac']:>5.2f}  p={s['p_placebo']:.3f}")

    print("\n# Robustness — the lodging-REIT basket (equal-weight, total-return)")
    fb = data.build_real("basket")
    for h in (6, 12):
        s = st.summarize(fb, h)
        print(f"  basket H={h:>2}: up={s['up_mean']*100:>5.1f}%  base={s['base_mean']*100:>5.1f}%"
              f"  Welch_t={s['t_welch']:>5.2f}  HAC_t={s['t_hac']:>5.2f}")
else:
    print("(no _cache — run data.fetch_hst() and data.fetch_basket() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  inference must recover a PLANTED positive lead and must NOT manufacture")
print("  significance when the true edge is 0 (RevPAR momentum carries no forward info).")
for edge in (0.0, 0.03):
    syn = data.synthetic(edge=edge, seed=761)
    s6 = st.summarize(syn, 6)
    print(f"  planted edge={edge:+.2f}: n_up={s6['n_up']:>3}  up6={s6['up_mean']*100:>6.2f}%  "
          f"base6={s6['base_mean']*100:>5.2f}%  Welch_t={s6['t_welch']:>5.2f}  "
          f"HAC_t={s6['t_hac']:>5.2f}  p={s6['p_placebo']:.3f}")
