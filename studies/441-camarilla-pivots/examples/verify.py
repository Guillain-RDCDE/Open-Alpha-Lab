"""Reproducible headline run for Study 441 — Camarilla Pivots.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached intraday bars under ``_cache/`` if
present (the real-tape numbers, with the in-progress final session dropped) and always runs the
synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd

from camarilla_pivots import data, strategy as st


def _trimmed():
    frames = data.load_real()
    out = {}
    for t, b in frames.items():
        last = b["session"].max()           # drop in-progress final session
        out[t] = b[b["session"] < last].copy()
    return out


print("# Camarilla pivots — L3/H3 first-touch reversion on 5m intraday bars (yfinance)")
if data.have_real():
    FRAMES = _trimmed()
    EV = pd.concat([st.collect_events(b).assign(ticker=t) for t, b in FRAMES.items()],
                   ignore_index=True)
    CTRL = pd.concat([st.collect_random_controls(b, n_draws=50) for b in FRAMES.values()],
                     ignore_index=True)
    smin = min(b["session"].min() for b in FRAMES.values())
    smax = max(b["session"].max() for b in FRAMES.values())
    sess = max(b["session"].nunique() for b in FRAMES.values())
    print(f"names         : {len(FRAMES)}  ({', '.join(FRAMES)})")
    print(f"window        : {smin.date()} -> {smax.date()}  ({sess} RTH sessions/name, 5m bars)")
    print(f"events        : {len(EV)} L3/H3 touches over {EV.groupby(['ticker','session']).ngroups} name-days")
    print(f"control draws : {len(CTRL):,} random control reversions")
    print(f"fingerprint   : {data.fingerprint(FRAMES)}")

    rev = EV["rev"].values
    daily = EV.groupby(["ticker", "session"])["rev"].mean().values
    print("\n# Pooled reversion (positive = level respected, toward pivot)")
    print(f"  rev_mean = {rev.mean()*100:+.4f}%   one-sample t = {st.ttest_vs_zero(rev):+.3f}   "
          f"HAC t = {st.hac_t(daily):+.3f}   hit-rate = {st.hit_rate(rev)*100:.1f}%")

    d = st.bootstrap_diff_p(rev, CTRL["rev"].values)
    print("\n# Random-control placebo (the headline)")
    print(f"  control_mean = {d['control_mean']*100:+.4f}%   real-control = {d['obs_diff']*100:+.4f}%"
          f"   bootstrap p(level beats random) = {1-d['p_value']:.3f}")

    c = st.net_of_costs(rev.mean(), cost_bps=2.0)
    print(f"\n# Net of a 2 bps round-trip: gross {c['gross']*100:+.4f}%  net {c['net']*100:+.4f}%")

    print("\n# By level")
    for lvl in ("L3", "H3"):
        s = EV[EV.level == lvl]["rev"].values
        print(f"  {lvl}: n={len(s):>3}  mean={s.mean()*100:+.4f}%  t={st.ttest_vs_zero(s):+.2f}  "
              f"hit={st.hit_rate(s)*100:.1f}%")

    print("\n# Hold-bars sweep")
    for h in (6, 12, 24):
        e = pd.concat([st.collect_events(b, hold=h) for b in FRAMES.values()], ignore_index=True)
        s = e["rev"].values
        print(f"  hold={h:>2} bars: n={len(s)}  mean={s.mean()*100:+.4f}%  t={st.ttest_vs_zero(s):+.2f}")

    print("\n# Per name")
    for t in EV["ticker"].unique():
        s = EV[EV.ticker == t]["rev"].values
        print(f"  {t:>4}: n={len(s):>3}  mean={s.mean()*100:+.4f}%  t={st.ttest_vs_zero(s):+.2f}")
else:
    print("(no _cache/bars_SPY_5m.parquet — run data.fetch_intraday(...) once to build the cache)")

print("\n# Synthetic positive control — deterministic, no network")
print("  zero planted respect must NOT manufacture significance / beat its control;")
print("  a large planted respect must light up AND beat the random twin.")
for respect in (0.0, 0.5):
    b, _ = data.synthetic_panel(respect=respect, seed=441)
    r = st.run_experiment(b, n_control=50)
    print(f"  respect={respect}: n={r['n_events']:>3}  rev={r['rev_mean']*100:+.4f}%  "
          f"t={r['t']:+.2f}  HAC={r['hac_t']:+.2f}  hit={r['hit']*100:.1f}%  "
          f"ctrl={r['control_mean']*100:+.4f}%  diff={r['diff']*100:+.4f}%  diff_p={r['diff_p']:.3f}")
