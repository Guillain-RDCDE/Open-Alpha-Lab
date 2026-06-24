"""Reproducible headline run for Study 414 — Falling Wedge.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached basket OHLC under ``_cache/`` if
present (the real-tape numbers), and always runs the synthetic positive control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from falling_wedge import data, strategy as st

NDRAWS = 5_000

print("# Falling Wedge — confirmed-breakout forward returns on SPY + 29 US large-caps (yfinance)")
if data.have_real():
    panel = data.load_real()
    closes = panel["close"]
    span = (closes.index.max() - closes.index.min()).days / 365.25
    bk_up = st.collect_breakouts(panel, side="up")
    bk_dn = st.collect_breakouts(panel, side="down")
    n_up = sum(len(v) for v in bk_up.values())
    n_dn = sum(len(v) for v in bk_dn.values())
    print(f"window         : {closes.index.min().date()} -> {closes.index.max().date()} "
          f"({span:.1f} years, {len(closes)} daily bars)")
    print(f"fingerprint    : {data.fingerprint(panel)}")
    print(f"up-break wedges: {n_up}  (SPY alone: {len(bk_up.get('SPY', []))})  "
          f"| down-break wedges: {n_dn}")

    print("\n# UP-break forward return, EXCESS over each name's base rate (1-day entry lag)")
    print(f"  {'H':>3} {'n':>5} {'exc%':>7} {'raw%':>7} {'win%':>5} {'t':>6} {'HACt':>6} "
          f"{'p_plac':>7} {'net%':>7}")
    for h in st.HORIZONS:
        r = st.run_experiment(panel, horizon=h, side="up", n_draws=NDRAWS)
        print(f"  {h:>3} {r['n_events']:>5} {r['mean']*100:>6.2f} {r['raw_mean']*100:>6.2f} "
              f"{r['win']*100:>4.0f} {r['t']:>6.2f} {r['hac_t']:>6.2f} "
              f"{r['p_placebo']:>7.3f} {r['net']*100:>6.2f}")

    print("\n# Myth check — DOWN-break of the same figure (does it break UP specifically?)")
    print(f"  {'H':>3} {'n':>5} {'exc%':>7} {'raw%':>7} {'win%':>5} {'t':>6} {'p_plac':>7}")
    for h in st.HORIZONS:
        r = st.run_experiment(panel, horizon=h, side="down", n_draws=NDRAWS)
        print(f"  {h:>3} {r['n_events']:>5} {r['mean']*100:>6.2f} {r['raw_mean']*100:>6.2f} "
              f"{r['win']*100:>4.0f} {r['t']:>6.2f} {r['p_placebo']:>7.3f}")

    print("\n# Robustness — strictness sweep at H=20 (moves the t, not the placebo)")
    for name, kw in [("strict (narrow>=.35, buf=.5%)", dict(min_narrow=0.35, breakout_buf=0.005)),
                     ("headline (narrow>=.25)", dict(min_narrow=0.25)),
                     ("loose (narrow>=.15)", dict(min_narrow=0.15))]:
        r = st.run_experiment(panel, horizon=20, side="up", n_draws=NDRAWS, **kw)
        print(f"  {name:<28} n={r['n_events']:>4}  exc={r['mean']*100:>5.2f}%  "
              f"t={r['t']:>5.2f}  p={r['p_placebo']:.3f}")

    print("\n# README hook — SPY only (too few clean wedges to certify)")
    for h in st.HORIZONS:
        r = st.run_experiment(panel, horizon=h, side="up", names=["SPY"], n_draws=NDRAWS)
        print(f"  H={h:>2}: n={r['n_events']:>2}  exc={r['mean']*100:>6.2f}%  raw={r['raw_mean']*100:>6.2f}%  "
              f"t={r['t']:>5.2f}  p={r['p_placebo']:.3f}")
else:
    print("(no _cache/wedge_close.parquet — run data.fetch_panel() once to build the cache)")

print("\n# Synthetic positive control — deterministic, no network (H=20, excess over base rate)")
print("  detector + inference must NOT manufacture a positive edge at edge=0, and must light up")
print("  with a large planted continuation — and the placebo is the right arbiter.")
for edge in (0.0, 0.40):
    panel_s, truth = data.synthetic_panel(edge=edge, seed=414)
    det = sum(len(v) for v in st.collect_breakouts(panel_s, side="up").values())
    r = st.run_experiment(panel_s, horizon=20, side="up", n_draws=4_000, seed=414)
    print(f"  planted edge={edge:+.2f}: planted={truth['n_planted_total']} detected={det} "
          f"n={r['n_events']:>4}  exc20={r['mean']*100:>6.2f}%  t={r['t']:>6.2f}  "
          f"p_placebo={r['p_placebo']:.3f}  win={r['win']*100:.0f}%")
