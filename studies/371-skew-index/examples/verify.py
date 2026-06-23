"""Reproducible headline run for Study 371 — CBOE SKEW Index.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached SKEW/VIX/SPY tape under
``_cache/`` if present (the real-tape numbers), and always runs the synthetic control with no
network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from skew_index import data, strategy as st

print("# CBOE SKEW vs forward SPY — is SKEW informative beyond VIX?")
if data.have_real():
    f = data.load_real()
    span = (f.index.max() - f.index.min()).days / 365.25
    print(f"tape          : {len(f)} joint days  ({f.index.min().date()} -> "
          f"{f.index.max().date()}, {span:.1f} years)")
    print(f"SKEW mean     : {f['skew'].mean():.1f}   VIX mean: {f['vix'].mean():.1f}   "
          f"corr(SKEW,VIX): {f['skew'].corr(f['vix']):+.2f}")

    print("\n# Decisive test — forward return on SKEW controlling for VIX (HAC / Newey-West)")
    print(f"  {'H':>4} {'n':>5} {'b_skew%':>8} {'t_skew':>7} {'b_vix%':>8} {'t_vix':>7}")
    for m in (1, 3):
        r = st.regress_forward(f, m)
        print(f"  {str(m)+'m':>4} {r['n']:>5} {r['beta_skew']*100:>+7.3f} "
              f"{r['t_skew']:>+7.2f} {r['beta_vix']*100:>+7.3f} {r['t_vix']:>+7.2f}")

    print("\n# Directional bucket — high-SKEW (top decile) forward returns vs rest")
    print(f"  {'H':>4} {'n_hi':>5} {'hi%':>7} {'rest%':>7} {'hi_win':>7} {'welch_t':>8} "
          f"{'placebo_p':>10}")
    for m in (1, 3):
        s = st.decile_summary(f, m)
        print(f"  {str(m)+'m':>4} {s['n_hi']:>5} {s['hi_mean']*100:>+6.2f} "
              f"{s['rest_mean']*100:>+6.2f} {s['hi_win']*100:>6.0f}% {s['t']:>+8.2f} "
              f"{s['p_placebo']:>10.3f}")

    print("\n# ...and the autocorrelation-respecting block bootstrap (the honest t)")
    for m in (1, 3):
        b = st.block_bootstrap_decile(f, m)
        print(f"  {m}m: obs(hi-rest)={b['obs_diff']*100:>+5.2f}%  block_se="
              f"{b['block_se']*100:>4.2f}%  block_t={b['block_t']:>+5.2f}  "
              f"distinct episodes~{b['n_episodes']}")

    print("\n# Tail-warning — forward 1-month tail-event rate after high SKEW vs base rate")
    for dd in (-0.10, -0.05):
        tw = st.tail_warning(f, 1, dd=dd)
        print(f"  drawdown < {dd*100:>+5.0f}%: hi-SKEW={tw['hi_tail_rate']*100:>4.1f}%  "
              f"base={tw['base_tail_rate']*100:>4.1f}%  placebo_p={tw['p_placebo']:.3f}")

    print("\n# Tradability — fade-high-SKEW sleeve vs buy-and-hold SPY (1d lag, 2bps/turn)")
    fs = st.fade_sleeve(f)
    print(f"  sleeve : CAGR={fs['sleeve_cagr']*100:>5.2f}%  Sharpe={fs['sleeve_sharpe']:.2f}  "
          f"(in cash {fs['pct_in_cash']*100:.0f}% of days)")
    print(f"  buyhold: CAGR={fs['bh_cagr']*100:>5.2f}%  Sharpe={fs['bh_sharpe']:.2f}")
else:
    print("(no _cache/skew_vix_spy.csv — run data.fetch() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  VIX-controlled HAC t on SKEW must stay quiet at edge=0 and light up at a real edge.")
for edge in (0.0, 0.0005):
    syn = data.synthetic_panel(edge=edge, seed=371)
    r = st.regress_forward(syn, 1)
    print(f"  planted edge={edge:+.4f}: n={r['n']}  beta_skew={r['beta_skew']*100:>+6.3f}%  "
          f"t_skew={r['t_skew']:>+5.2f}")
