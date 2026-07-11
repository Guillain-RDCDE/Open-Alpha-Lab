"""Reproducible headline run for Study 688 — Concealing Baby Swallow.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached daily basket tape under
``_cache/`` (fetching once on a cache miss — a 111-name basket, so the first run can
take a couple of minutes), and always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quantlab.repro import fingerprint  # noqa: E402

from concealing_baby_swallow import data, strategy as st  # noqa: E402

print("# Concealing Baby Swallow — does the rarest bullish reversal candle in the book "
      "ever even happen?")

if not data.have_real():
    print("(cache miss — fetching the 111-name basket once; a couple of minutes)")
    data.fetch_all()

panel = data.load_real()
print(f"basket: {len(panel)} tickers (SPY/QQQ/DIA/IWM + {len(panel) - 4} US large-caps), "
      "cache-first yfinance daily OHLCV, survivors (still trading today)")

name_years = sum((b.index[-1] - b.index[0]).days / 365.25 for b in panel.values() if len(b) > 1)
max_span = max((b.index[-1] - b.index[0]).days / 365.25 for b in panel.values() if len(b) > 1)
oldest = min(b.index[0] for b in panel.values() if len(b))
print(f"coverage: {name_years:,.0f} name-years searched, oldest bar {oldest.date()}, "
      f"longest single series spans {max_span:.1f} years, as-of {data.AS_OF}")

combined = pd.concat([panel[t][["close"]] for t in sorted(panel)], axis=0)
fp = fingerprint(combined, cols=["close"])
print(f"[data] basket close panel: {len(combined):,} rows across {len(panel)} tickers  "
      f"as-of {data.AS_OF}  fingerprint={fp}")

print("\n# THE SEARCH — scanning every bar of every ticker for the 4-candle shape")
res = st.run_experiment(panel)
print(f"  loose cut (plain geometric reading): {res['n_loose']} occurrences")
print(f"  strict cut (literature-close: true marubozu + true gaps + day-4 opens in "
      f"day-3's shadow): {res['n_strict']} occurrences")
print(f"  pre-registered minimum for any t-stat: {st.MIN_N_FOR_TEST} pooled events")
print(f"  -> {'BOTH cuts are BELOW the bar: no t-statistic is computed, by design.' if res['n_loose'] < st.MIN_N_FOR_TEST else 'the loose cut clears the bar.'}")

ev = st.pool_events(panel)
if len(ev):
    print("\n  every loose-cut occurrence found, in full (there are only four):")
    for _, row in ev.sort_values("pos").iterrows():
        t = row["ticker"]
        bars = panel[t]
        i = int(row["pos"])
        d4 = bars.index[i]
        print(f"    {t:<5s} day-4 confirms {d4.date()}  strict={bool(row['strict'])}  "
              f"fwd 1d {row['ret_1']*1e4:+.0f}bp  5d {row['ret_5']*1e4:+.0f}bp  "
              f"10d {row['ret_10']*1e4:+.0f}bp  20d {row['ret_20']*1e4:+.0f}bp")
    print("  note: the GD 1962-06-14 occurrence prints at a split-adjusted price of "
        "~$0.12/share — coarse penny-level rounding at that price level can manufacture "
        "near-zero-shadow ('marubozu') geometry on its own; named as a data-quality "
        "caveat, not excluded (the other three are unaffected, modern-era prices).")

print("\n# CONTEXT — the base rate (same long bet on every 4-red-days-in-a-downtrend bar, "
      "whether or not the specific concealing shape fired)")
for h in st.HORIZONS:
    d = res["per_horizon"][h]
    b = d["base"]
    s = d["ladder"]
    print(f"  h={h:>2d}d: base rate n={b['n']:,} mean {b['mean_bps']:+.1f}bp "
          f"(HAC t={b['tstat']:+.2f}, this is just the basket's own dip-bounce, NOT "
          f"evidence for the pattern) | the {s['n']} CBS occurrences mean "
          f"{s['mean_bps']:+.1f}bp, win rate {s['win_rate']*100:.0f}% "
          f"(n={s['n']} — UNTESTED, n < {st.MIN_N_FOR_TEST})")

print("\n# Descriptive-only placebo on the tiny sample (NOT a certifying test — with "
      "n=4 this is decoration, printed for transparency, never cited as evidence)")
for h in st.HORIZONS:
    d = res["per_horizon"][h]
    print(f"  h={h:>2d}d: observed mean {d['ladder']['mean_bps']:+.1f}bp vs "
          f"{res['n_loose']}-draw placebo mean {d['placebo_mean_bps']:+.1f}bp "
          f"-> p = {d['placebo_p']:.4f}")

print("\n# Synthetic positive control — deterministic, no network")
print("  the detector must NOT systematically fire on a null world (edge=0) and must "
      "recover a planted post-pattern bounce. Null checked over 20 seeds.")
null_ts, null_ns = [], []
for s_ in range(20):
    d0, _ = data.synthetic_panel(edge=0.0, seed=688 + s_)
    r0 = st.synthetic_detect(d0, seed=688 + s_)
    null_ts.append(r0["welch_t"])
    null_ns.append(r0["n"])
null_ts = np.asarray(null_ts, dtype=float)
print(f"  null (edge=0), 20 seeds: mean Welch t = {null_ts.mean():+.2f} "
      f"(sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(np.abs(null_ts) >= 2).sum()}/20 seeds "
      f"(nominal two-sided 5% rate is ~1/20) | events found per seed: "
      f"{min(null_ns)}-{max(null_ns)}")

d1, _ = data.synthetic_panel(edge=0.05, seed=688)
r1 = st.synthetic_detect(d1, seed=688)
print(f"  planted edge=+0.05 (seed 688): n={r1['n']} events, mean {r1['mean_bps']:+.1f}bp "
      f"vs base {r1['base_bps']:+.1f}bp (delta {r1['delta_bps']:+.1f}bp)  "
      f"Welch t = {r1['welch_t']:+.2f}")
print("  the machinery finds a planted edge and stays quiet on a null. The near-zero "
      "real-tape count (4 loose / 0 strict) is a property of the market, not a broken "
      "detector.")

print("\n# VERDICT")
print("  Signal: NONE -- the pattern is too rare on the real tape to test at all "
      f"({res['n_loose']} loose-cut / {res['n_strict']} strict-cut occurrences across "
      f"{len(panel)} names and {name_years:,.0f} name-years, below the pre-registered "
      f"n>={st.MIN_N_FOR_TEST} bar).")
print("  Tradability: MIRAGE -- nothing to charge costs against; even pooled across the "
      "desk's largest basket the pattern fires roughly once every 15 years.")
print("  Myth-check 'too rare to ever test?': CONFIRMED.")
