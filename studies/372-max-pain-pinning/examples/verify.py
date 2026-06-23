"""Reproducible headline run for Study 372 — Max-Pain Pinning.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached option-chain snapshot under
``_cache/`` if present (the real-tape gap numbers), and always runs the synthetic pinning
control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from max_pain_pinning import data, strategy as st

print("# Max-Pain Pinning — option-chain max-pain snapshot + synthetic pinning control")

if data.have_real():
    snap = data.load_real()
    g = st.snapshot_gap_stats(snap)
    print(f"\n# REAL SNAPSHOT (as-of {snap['asof'].iloc[0]}) — {g['n']} liquid underlyings")
    print("  ticker     spot   maxpain    gap%")
    for _, r in snap.iterrows():
        print(f"  {r['ticker']:<6} {r['spot']:>8.2f} {r['maxpain']:>8.2f}  {r['gap_pct']:>+7.2f}%")
    print(f"\n  median |gap|      : {g['median_abs_gap']:.2f}%")
    print(f"  mean   |gap|      : {g['mean_abs_gap']:.2f}%")
    print(f"  within 0.5% of MP : {g['frac_within_0p5pct']*100:.0f}%")
    print(f"  within 1.0% of MP : {g['frac_within_1pct']*100:.0f}%")
    print(f"  max    |gap|      : {g['max_abs_gap']:.2f}%")
    print("  (snapshot only — yfinance retains no expiry-day close, so the LANDING claim")
    print("   cannot be tested on this tape; it is deferred to the synthetic control below.)")
else:
    print("(no _cache/mp_snapshot.csv — run data.fetch_snapshot() once to build it)")

print("\n# SYNTHETIC PINNING CONTROL — deterministic, no network")
print("  detector must NOT manufacture significance with pin=0, and MUST light up with a")
print("  large planted pin. Paired t on (dist_spot_anchor - dist_maxpain); win-rate vs 50%.")
print(f"  {'pin':>5} {'n':>4} {'d_mp':>6} {'d_anc':>6} {'diff':>6} {'paired_t':>9} "
      f"{'win':>5} {'p_plac':>7} {'fade_g%':>8} {'fade_n%':>8} {'hit':>5}")
for pin in (0.0, 0.05, 0.15, 0.40):
    ep = data.synthetic_episodes(n_episodes=400, pin=pin, seed=372)
    s = st.summarize(ep)
    f = st.fade_trade(ep)
    print(f"  {pin:>5.2f} {s['n']:>4} {s['mean_d_mp']:>6.3f} {s['mean_d_anchor']:>6.3f} "
          f"{s['mean_diff']:>+6.3f} {s['t']:>9.2f} {s['win_rate']*100:>4.0f}% "
          f"{s['p_placebo']:>7.3f} {f['gross_mean']*100:>+7.2f}% {f['net_mean']*100:>+7.2f}% "
          f"{f['hit_rate']*100:>4.0f}%")

print("\n# Robustness — false-positive check: more episodes never fakes a pin at pin=0")
for n in (200, 400, 800):
    s = st.summarize(data.synthetic_episodes(n_episodes=n, pin=0.0, seed=372))
    print(f"  n={n:>4}: paired_t={s['t']:>7.2f}  placebo_p={s['p_placebo']:.3f}")

print("\n# Robustness — pinning needs days to act (pin=0.15, vary days-to-expiry)")
for dte in (3, 5, 10):
    s = st.summarize(data.synthetic_episodes(n_episodes=400, pin=0.15, seed=372,
                                             days_to_expiry=dte))
    print(f"  dte={dte:>2}: paired_t={s['t']:>6.2f}  win={s['win_rate']*100:.0f}%")
