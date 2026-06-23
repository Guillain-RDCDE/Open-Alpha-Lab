"""Reproducible headline run for Study 398 — Entropy-Efficiency.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached SPY prices under ``_cache/`` if
present (the real-tape numbers), and always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from entropy_efficiency import data, strategy as st

print("# Entropy-Efficiency — the 'efficiency clock' on SPY (permutation entropy of returns)")
if data.have_real():
    F = data.load_real()
    span_years = (F.index.max() - F.index.min()).days / 365.25
    ent = st.rolling_entropy(F["ret"], window=60, kind="perm")
    mask = st.low_entropy_mask(ent, q=0.20)
    print(f"return days    : {len(F)}  ({F.index.min().date()} -> {F.index.max().date()}, "
          f"{span_years:.1f} years)")
    print(f"perm entropy   : mean {ent.mean():.3f}  range [{ent.min():.3f}, {ent.max():.3f}] "
          f"(near the white-noise ceiling of 1.0)")
    print(f"low-entropy    : bottom 20% -> {int(mask.sum())} days ({mask.mean()*100:.0f}% of tape)")

    print("\n# Forward SPY returns: low-entropy vs high-entropy days (1-day entry lag)")
    print(f"  {'H':>4} {'n_low':>5} {'n_high':>6} {'low%':>8} {'high%':>8} {'gap%':>8} "
          f"{'lowwin':>7} {'basewin':>8} {'HAC_t':>7} {'boot_p':>7}")
    for h in (1, 5, 21):
        s = st.summarize(F, mask, h)
        print(f"  {str(h)+'d':>4} {s['n_low']:>5} {s['n_high']:>6} {s['low_mean']*100:>7.3f}% "
              f"{s['high_mean']*100:>7.3f}% {s['gap']*100:>7.3f}% {s['low_win']*100:>6.0f}% "
              f"{s['base_win']*100:>7.0f}% {s['t_hac']:>7.2f} {s['p_boot']:>7.3f}")

    print("\n# Second estimator — sample entropy (Richman-Moorman); the clock disappears")
    ent_s = st.rolling_entropy(F["ret"], window=60, kind="sample")
    mask_s = st.low_entropy_mask(ent_s, q=0.20)
    for h in (1, 5, 21):
        s = st.summarize(F, mask_s, h)
        print(f"  {str(h)+'d':>4}  gap={s['gap']*100:>6.3f}%  lowwin={s['low_win']*100:>3.0f}%  "
              f"basewin={s['base_win']*100:>3.0f}%  HAC_t={s['t_hac']:>5.2f}  boot_p={s['p_boot']:.3f}")

    print("\n# Robustness — low-entropy quantile sweep (21-day horizon)")
    for q in (0.10, 0.20, 0.30):
        m = st.low_entropy_mask(ent, q=q)
        s = st.summarize(F, m, 21)
        print(f"  q={q:.2f}: n_low={s['n_low']:>4}  gap={s['gap']*100:>5.3f}%  "
              f"HAC_t={s['t_hac']:>5.2f}  boot_p={s['p_boot']:.3f}")

    print("\n# Tradability — long-in-low-entropy / flat, 1 bp per turn")
    c = st.net_of_costs(F, mask, cost_bps=1.0)
    print(f"  exposure={c['exposure']*100:.0f}%  turnover={c['turnover_annual']:.1f}x/yr")
    print(f"  gross  {c['gross_ann']*100:>6.2f}%/yr  Sharpe {c['gross_sharpe']:.2f}")
    print(f"  net    {c['net_ann']*100:>6.2f}%/yr  Sharpe {c['net_sharpe']:.2f}")
    print(f"  buy&hold {c['bh_ann']*100:>5.2f}%/yr  Sharpe {c['bh_sharpe']:.2f}")
else:
    print("(no _cache/spy_prices.csv — run data.fetch_spy() once to build it)")

print("\n# Synthetic positive control — deterministic, no network")
print("  random vs zero-sum-cyclic STRUCTURED (low sample-entropy) regimes; the detector must")
print("  find the structured regime, recover a PLANTED edge, and NOT fake significance at edge=0.")
for edge in (0.0, 0.0025):
    syn = data.synthetic_returns(n_days=3000, block=120, edge=edge, seed=398)
    ent_y = st.rolling_entropy(syn["ret"], window=40, kind="sample")
    truth = syn["regime"].astype(bool)
    mask_y = st.low_entropy_mask(ent_y, q=0.30)
    recall = (mask_y & truth).sum() / max(1, int(mask_y.sum()))
    s = st.summarize(syn, mask_y, 5)
    print(f"  planted edge={edge*100:+.2f}%/day: detected={int(mask_y.sum())} recall={recall*100:.0f}% "
          f"gap={s['gap']*100:+.2f}% HAC_t={s['t_hac']:>5.2f} boot_p={s['p_boot']:.3f}")
