"""Reproducible headline run for Study 709 — World-Series-Effect.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached ^GSPC tape under
``_cache/`` (fetching once on a cache miss), and always runs the synthetic control
with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from world_series_effect import data, strategy as st  # noqa: E402

print("# World-Series-Effect — does the champion's league (or hometown) predict next year's market?")

ws = data.ws_table()
print(f"calendar: {len(ws)} played World Series {int(ws['ws_year'].min())} -> "
      f"{int(ws['ws_year'].max())} (hardcoded MLB postseason table; 1994 strike excluded)")

if not data.have_real():
    print("(cache miss — fetching ^GSPC once)")
    data.fetch()

gspc = data.load_real()
print(data_stamp("^GSPC daily close", gspc, asof=data.AS_OF))

ann = data.annual_returns(gspc)
print(f"complete calendar-year returns on tape: {len(ann)} "
      f"({ann.index.min()} -> {ann.index.max()}, price-only, no dividends)")

ev = st.build_events(ws, ann)
print(f"scoreable events (WS season -> complete next-year return): {len(ev)} "
      f"({int(ev['ws_year'].min())} -> {int(ev['ws_year'].max())})")
n_al = int((ev["league"] == "AL").sum())
n_nl = int((ev["league"] == "NL").sum())
n_ny = int(ev["is_ny"].sum())
print(f"league split: AL={n_al}  NL={n_nl}   |   New York champion seasons: {n_ny}")

print("\n# THE HEADLINE — NL win -> bullish next year (the Super Bowl NFC mnemonic, ported)")
bull_nl = (ev["league"] == "NL").to_numpy()
s1 = st.omen_stats(ev, bull_nl)
print(f"  mean next-year return | NL-preceded {s1['mean_bull_pct']:+.2f}%  vs  "
      f"AL-preceded {s1['mean_bear_pct']:+.2f}%   contrast {s1['contrast_pct']:+.2f} pp")
print(f"  Welch t = {s1['welch_t']:+.2f}   permutation p (20,000 draws) = {s1['perm_p']:.4f}")
print(f"  omen hit rate = {s1['hit_rate_pct']:.1f}% (Wilson 95% [{s1['hit_lo_pct']:.1f}%, "
      f"{s1['hit_hi_pct']:.1f}%])  vs unconditional up-rate {s1['uncond_up_pct']:.1f}%")
print(f"  binomial p (H0: hit-in-bull-years rate = unconditional up-rate): {s1['binom_p']:.4f}")
print(f"  myth-check — beats a flat coin (p=0.5, two-sided)? p = {s1['coin_p']:.4f}")

print("\n# THE CITY VARIANT — a New York franchise wins -> bullish next year")
bull_ny = ev["is_ny"].to_numpy()
s2 = st.omen_stats(ev, bull_ny)
print(f"  mean next-year return | NY-preceded {s2['mean_bull_pct']:+.2f}%  vs  "
      f"non-NY-preceded {s2['mean_bear_pct']:+.2f}%   contrast {s2['contrast_pct']:+.2f} pp")
print(f"  Welch t = {s2['welch_t']:+.2f}   permutation p (20,000 draws) = {s2['perm_p']:.4f}")
print(f"  omen hit rate = {s2['hit_rate_pct']:.1f}% (Wilson 95% [{s2['hit_lo_pct']:.1f}%, "
      f"{s2['hit_hi_pct']:.1f}%])  vs unconditional up-rate {s2['uncond_up_pct']:.1f}%")
print(f"  binomial p (H0: hit-in-bull-years rate = unconditional up-rate): {s2['binom_p']:.4f}")
print(f"  myth-check — beats a flat coin (p=0.5, two-sided)? p = {s2['coin_p']:.4f}")

print("\n# COULD YOU TRADE IT? — hold the S&P only after a bull-flagged season, else cash")
t1 = st.timing_strategy(ev, bull_nl)
print(f"  NL-omen timing: {t1['n_held']}/{t1['n_years']} years held -> "
      f"{t1['strat_ann_pct']:+.2f}%/yr vs buy-and-hold {t1['bah_ann_pct']:+.2f}%/yr "
      f"(advantage {t1['ann_advantage_pct']:+.2f} pp/yr)")
t2 = st.timing_strategy(ev, bull_ny)
print(f"  NY-omen timing: {t2['n_held']}/{t2['n_years']} years held -> "
      f"{t2['strat_ann_pct']:+.2f}%/yr vs buy-and-hold {t2['bah_ann_pct']:+.2f}%/yr "
      f"(advantage {t2['ann_advantage_pct']:+.2f} pp/yr)")

print("\n# Synthetic positive control — deterministic, no network")
print("  the detector must NOT fire on a null world (boost=0) and must recover a")
print("  planted omen. Null checked over 20 seeds (never a single stream).")
null_ts = []
for s_ in range(20):
    syn, _ = data.synthetic_world(boost=0.0, seed=709 + s_)
    null_ts.append(st.synthetic_detect(syn)["welch_t"])
null_ts = np.asarray(null_ts)
print(f"  null (boost=0), 20 seeds: mean Welch t = {null_ts.mean():+.2f} "
      f"(sd {null_ts.std(ddof=1):.2f}), |t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
syn, _ = data.synthetic_world(boost=10.0, seed=709)
sy = st.synthetic_detect(syn)
print(f"  planted boost=+10.0 pp (seed 709): bull mean {sy['mean_bull_pct']:+.2f}% vs bear "
      f"{sy['mean_bear_pct']:+.2f}%  Welch t = {sy['welch_t']:+.2f}")

print("\n# Verdict: Signal=NONE, Tradability=MIRAGE, myth-check=BUSTED — a coincidence dressed as an omen.")
