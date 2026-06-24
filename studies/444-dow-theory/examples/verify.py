"""Reproducible headline run for Study 444 — Dow Theory confirmation.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached ETF panel (DIA/IYT) and
the long index panel (^DJI/^DJT) under ``_cache/`` if present, pinned to the as-of
date below, and always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dow_theory import data, strategy as st

ASOF = "2026-05-31"          # last full calendar month before the build (no partial period)


def _pin(df):
    return df[df.index <= ASOF]


print("# Dow Theory — Industrials/Transports confirmation, tradable ETF proxies (DIA/IYT)")
if data.have_real():
    df = _pin(data.load_real())
    print(f"ETF tape : {df.index.min().date()} -> {df.index.max().date()}  "
          f"({len(df)} days, fingerprint {data.fingerprint(df)})")

    print("\n# Buy-and-hold the Industrials  vs  Dow-Theory-timed (long only when confirmed)")
    print(f"  {'mode':<14} {'on%':>5} {'CAGR%':>7} {'Sharpe':>7} {'MDD%':>7} "
          f"{'active/yr%':>10} {'HAC t':>7}")
    for latched, label in ((True, "latched"), (False, "instant flag")):
        s = st.summarize(df, latched=latched, cost_bps=2.0)
        if latched:
            print(f"  {'buy & hold':<14} {'--':>5} {s['bh_cagr']*100:>7.2f} "
                  f"{s['bh_sharpe']:>7.3f} {s['bh_mdd']*100:>7.1f} {'--':>10} {'--':>7}")
        print(f"  {label:<14} {s['on_frac']*100:>5.1f} {s['st_cagr']*100:>7.2f} "
              f"{s['st_sharpe']:>7.3f} {s['st_mdd']*100:>7.1f} "
              f"{s['active_mean_ann']*100:>+10.2f} {s['active_t']:>7.3f}")

    print("\n# Content test — Industrials next-day return on CONFIRMED vs NON-confirmed days")
    for latched, label in ((True, "latched"), (False, "instant flag")):
        c = st.confirmed_vs_not(df, latched=latched)
        print(f"  {label:<14}: confirmed {c['mean_on_ann']*100:>+6.2f}%/yr  "
              f"non {c['mean_off_ann']*100:>+6.2f}%/yr  spread {c['spread_ann']*100:>+6.2f}%/yr  "
              f"Welch t {c['welch_t']:>6.3f}")

    print("\n# Placebo — random regime, same on-fraction + persistence (latched)")
    reg = st.latched_regime(df)
    pl = st.placebo_pvalue(df, reg, n_draws=2000, cost_bps=2.0)
    print(f"  observed net Sharpe {pl['obs_sharpe']:.3f}  placebo-mean {pl['placebo_mean']:.3f}  "
          f"p(random >= real) = {pl['p_value']:.3f}")

    print("\n# Cost sweep (latched) — active HAC t barely moves; the edge was never there")
    for c in (0.0, 2.0, 5.0, 10.0):
        s = st.summarize(df, latched=True, cost_bps=c)
        print(f"  {c:>5.1f} bps: active {s['active_mean_ann']*100:>+6.2f}%/yr  "
              f"HAC t {s['active_t']:>6.3f}  Strat Sharpe {s['st_sharpe']:.3f}")

    print("\n# Lookback robustness (latched, 2 bps) — negative active t at every window")
    for lb in (21, 42, 63, 126, 200):
        s = st.summarize(df, lookback=lb, latched=True, cost_bps=2.0)
        print(f"  LB {lb:>3}: on {s['on_frac']:.2f}  Strat Sharpe {s['st_sharpe']:.3f} "
              f"vs BH {s['bh_sharpe']:.3f}  active t {s['active_t']:>6.3f}")
else:
    print("(no _cache/dow_theory_etf.parquet — run data.fetch_etf() once to build the cache)")

if data.have_idx():
    idx = _pin(data.load_idx())
    s = st.summarize(idx, latched=True, cost_bps=2.0)
    c = st.confirmed_vs_not(idx, latched=True)
    print(f"\n# Long index robustness (^DJI/^DJT price-only, "
          f"{idx.index.min().date()} -> {idx.index.max().date()}, fp {data.fingerprint(idx)})")
    print(f"  BH Sharpe {s['bh_sharpe']:.3f}  Strat Sharpe {s['st_sharpe']:.3f}  "
          f"active {s['active_mean_ann']*100:+.2f}%/yr  HAC t {s['active_t']:.3f}  "
          f"content spread {c['spread_ann']*100:+.2f}%/yr (Welch t {c['welch_t']:.3f})")

print("\n# Synthetic positive control — deterministic, no network")
print("  the content test must recover a PLANTED confirmation edge and must NOT")
print("  manufacture one when the true edge is 0.")
for edge in (0.0, 0.003):
    px, truth = data.synthetic_panel(edge=edge, seed=444, n_days=8000)
    c = st.confirmed_vs_not(px, latched=False)
    print(f"  planted edge={edge:+.4f}/day  conf_frac={truth['confirmed_frac']:.3f}: "
          f"confirmed {c['mean_on_ann']*100:>+7.2f}%/yr  non {c['mean_off_ann']*100:>+6.2f}%/yr  "
          f"spread {c['spread_ann']*100:>+7.2f}%/yr  Welch t {c['welch_t']:>6.3f}")
