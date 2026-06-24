"""Reproducible headline run for Study 408 — Three Black Crows.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached basket OHLCV under ``_cache/``
if present (the real-tape numbers), and always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from three_black_crows import data, strategy as st

NDRAWS = 5_000

print("# Three Black Crows — signed-SHORT event study on a 30-name large-cap + SPY basket")
if data.have_real():
    panel = data.load_real(allow_fetch=False)
    fp = data.fingerprint(panel)
    mins = min(b.index.min() for b in panel.values())
    maxs = max(b.index.max() for b in panel.values())
    span = (maxs - mins).days / 365.25
    tot = sum(len(b) for b in panel.values())
    print(f"names          : {len(panel)}  (fingerprint {fp})")
    print(f"window         : {mins.date()} -> {maxs.date()}  ({span:.1f} years, {tot:,} bars)")

    print("\n# Signed-SHORT forward return after three black crows (enter next open, 1-day lag)")
    print(f"  {'H':>3} {'n':>5} {'mean%':>7} {'base%':>7} {'win%':>5} "
          f"{'t_hac':>6} {'t_one':>6} {'t_welch':>7} {'p_plac':>7} {'net%':>7}")
    for h in st.HORIZONS:
        s = st.summarize(panel, h, n_draws=NDRAWS)
        print(f"  {h:>3} {s['n_events']:>5} {s['mean']*100:>6.3f} {s['base_mean']*100:>6.3f} "
              f"{s['win']*100:>4.0f} {s['t_hac']:>6.2f} {s['t_one']:>6.2f} "
              f"{s['t_welch']:>7.2f} {s['p_placebo']:>7.3f} {s['net']*100:>6.3f}")

    print("\n# Myth check #1 — the STRICT long-bodied crow (close near each low)")
    for h in st.HORIZONS:
        s = st.summarize(panel, h, n_draws=NDRAWS, strict=True)
        print(f"  H={h:>2}: n={s['n_events']:>4} mean={s['mean']*100:>6.3f}%  "
              f"t_hac={s['t_hac']:>6.2f}  p={s['p_placebo']:.3f}  net={s['net']*100:>6.3f}%")

    print("\n# Myth check #2 — only crows after a PRIOR UPTREND (a genuine reversal)")
    for h in st.HORIZONS:
        s = st.summarize(panel, h, n_draws=NDRAWS, require_trend=True)
        print(f"  H={h:>2}: n={s['n_events']:>4} mean={s['mean']*100:>6.3f}%  "
              f"t_hac={s['t_hac']:>6.2f}  p={s['p_placebo']:.3f}  net={s['net']*100:>6.3f}%")
else:
    print("(no _cache/tbc_*.parquet — run data.load_real() with network once to build the cache)")

print("\n# Synthetic positive control — deterministic, no network")
print("  detector + inference must recover a PLANTED post-pattern crash and must NOT")
print("  manufacture significance when the true edge is 0.")
for edge in (0.0, 0.004, 0.008):
    px, truth = data.synthetic_panel(edge=edge, seed=408)
    s = st.summarize(px, 5, n_draws=4_000)
    print(f"  planted edge={edge:+.3f}/day: planted_days={truth['n_planted_days']:>4}  "
          f"n={s['n_events']:>4}  mean={s['mean']*100:>6.3f}%  t_hac={s['t_hac']:>6.2f}  "
          f"t_one={s['t_one']:>6.2f}  p={s['p_placebo']:.3f}  win={s['win']*100:.0f}%")
