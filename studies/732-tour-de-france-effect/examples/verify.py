"""Reproducible headline run for Study 732 — Tour-de-France-Effect.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached EWQ / ^FCHI / VGK tapes
under ``_cache/`` (fetching once on a cache miss), and always runs the synthetic control
with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quantlab.repro import data_stamp  # noqa: E402

from tour_de_france_effect import data as dt, strategy as st  # noqa: E402

print("# Tour-de-France-Effect — do French equities get a July 'Grande Boucle' bump?")

print(f"calendar: {len(dt.EVENTS)} editions {dt.EVENTS[0][0]}->{dt.EVENTS[-1][0]}, "
      "hardcoded from Wikipedia (2020 shifted July->Aug/Sep by COVID-19)")

if not dt.have_real():
    print("(cache miss — fetching EWQ + ^FCHI + VGK once)")
    dt.fetch()

prices = dt.load_real()
panel = pd.DataFrame({t: s for t, s in prices.items()})
print(data_stamp("TdF panel (EWQ + ^FCHI + VGK)", panel, asof=dt.AS_OF))

ev = st.build_event_table(prices, cost_bps=5.0)
inc = ev[ev["included"]]
sub = inc[inc["has_ar"]]                                   # 2005+ (VGK coverage)
print(f"\neditions resolved: {len(inc)}/{len(ev)} covered by EWQ; "
      f"{len(sub)} also have VGK (2005+) for the abnormal test; "
      f"window length {int(inc['win_sessions'].min())}-{int(inc['win_sessions'].max())} "
      f"sessions (mean {inc['win_sessions'].mean():.1f})")

print("\n# THE RAW SEASONAL — EWQ total return over the Tour window (entry->exit)")
for name, col in (("EWQ raw (1996-2025)", "raw_ret"),
                  ("EWQ raw NET @5bps", "raw_net"),
                  ("CAC 40 price-only", "cac_raw")):
    s = st.one_sample_t(inc[col].values)
    hr = st.hit_rate(inc[col].values)
    print(f"  {name:<22s} n={s['n']:2d}  mean={s['mean']*100:+.3f}%  t={s['t']:+.3f}  "
          f"hit {hr['k']}/{hr['n']}={hr['rate']*100:.1f}% "
          f"(Wilson [{hr['lo']*100:.1f}%, {hr['hi']*100:.1f}%])")

print("\n# THE ABNORMAL (France-specific) — EWQ minus VGK Europe, same window (2005+)")
ewq, vgk = prices[dt.FRANCE_ETF], prices[dt.EUROPE_BENCHMARK]
common = ewq.index.intersection(vgk.index).sort_values()
vgk_rets = []
for _, r in sub.iterrows():
    e, x = st._window_positions(common, r["grand_depart"], r["final_stage"])
    vgk_rets.append(float(vgk.loc[common[x]] / vgk.loc[common[e]] - 1.0))
s_ewq = st.one_sample_t(sub["raw_ret"].values)
s_vgk = st.one_sample_t(np.asarray(vgk_rets))
s_ar = st.one_sample_t(sub["ar"].values)
hr_ar = st.hit_rate(sub["ar"].values)
print(f"  EWQ France (2005+ subset) mean={s_ewq['mean']*100:+.3f}%  t={s_ewq['t']:+.3f}")
print(f"  VGK Europe (2005+ subset) mean={s_vgk['mean']*100:+.3f}%  t={s_vgk['t']:+.3f}  "
      "<- the summer-doldrums component, shared region-wide")
print(f"  ABNORMAL EWQ-VGK          mean={s_ar['mean']*100:+.3f}%  t={s_ar['t']:+.3f}  "
      f"hit {hr_ar['k']}/{hr_ar['n']}={hr_ar['rate']*100:.1f}% "
      f"(Wilson [{hr_ar['lo']*100:.1f}%, {hr_ar['hi']*100:.1f}%])")
print(f"  Welch t (France - Europe, unpaired) = "
      f"{st.welch_t(sub['raw_ret'].values, np.asarray(vgk_rets)):+.3f}")

print("\n# RANDOM-WINDOW PLACEBO (20 seeds x 200 draws; same-length windows elsewhere in "
      "the tape)")
for kind, cost, tail, lab in (("raw", 0.0, "left", "raw, left (as weak or weaker)"),
                              ("raw", 5.0, "left", "raw net@5bps, left"),
                              ("ar", 0.0, "left", "abnormal, left")):
    pl = st.placebo_pvalue(ev, prices, kind, cost_bps=cost, tail=tail)
    print(f"  {lab:<28s} observed {pl['obs']*100:+.3f}%  vs placebo mean "
          f"{pl['placebo_mean']*100:+.3f}% (sd {pl['placebo_sd']*100:.3f}%) "
          f"over {pl['n_draws']:,} draws -> p = {pl['p_value']:.3f}")
for kind in ("raw", "ar"):
    pl = st.placebo_pvalue(ev, prices, kind, tail="right")
    print(f"  {kind} right-tail (share of ordinary windows that BEAT the Tour mean) = "
          f"{pl['p_value']:.3f}")

print("\n# EVENT ANATOMY — mean cumulative return by session offset from entry")
cp_raw = st.car_path(ev, prices, "raw", max_k=16)
cp_ar = st.car_path(ev, prices, "ar", max_k=16)
for k in (0, 4, 8, 12, 16):
    print(f"  day {k:>2d}: raw {cp_raw[k]*100:+.3f}%   abnormal {cp_ar[k]*100:+.3f}%")

print("\n# ROBUSTNESS — the 2020 'calendar vs race' probe (race moved to Aug-Sep)")
r20 = inc[inc["year"] == 2020].iloc[0]
no20 = inc[inc["year"] != 2020]
s_no20 = st.one_sample_t(no20["raw_ret"].values)
print(f"  2020 actual (Aug-Sep) window: raw {r20['raw_ret']*100:+.3f}%  "
      f"abnormal {r20['ar']*100:+.3f}%  (the worst single edition — a market crash month, "
      "not a Tour effect)")
print(f"  RAW excluding 2020: n={s_no20['n']}  mean={s_no20['mean']*100:+.3f}%  "
      f"t={s_no20['t']:+.3f}  (still nothing)")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
print("  the one-sample-t abnormal detector must NOT fire on a null world (bump=0) and "
      "must recover a planted per-day July bump. Null checked over 20 seeds.")
null_ts = np.array([st.synthetic_detect(bump=0.0, seed=732 + s)["t"] for s in range(20)])
print(f"  null (bump=0), 20 seeds: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), "
      f"|t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
p5 = st.synthetic_detect(bump=0.0005, seed=732)
p10 = st.synthetic_detect(bump=0.0010, seed=732)
print(f"  planted +5bp/day  (seed 732): mean AR {p5['mean']*100:+.3f}%  t = {p5['t']:+.2f}")
print(f"  planted +10bp/day (seed 732): mean AR {p10['mean']*100:+.3f}%  t = {p10['t']:+.2f}")

print("\n# VERDICT")
print("  Signal:      NONE   -- raw t=-0.37, abnormal (France-Europe) t=-0.78; both slightly")
print("                        NEGATIVE. The July 'bump' is a mild drag indistinguishable from")
print("                        an ordinary three weeks (placebo p=0.167).")
print("  Tradability: MIRAGE -- the raw seasonal is negative before costs (-0.35%, -0.45% net);")
print("                        there is no positive edge to trade, only summer weakness to pay for.")
print("  France-specific? NOT SUPPORTED -- France does not beat Europe during the Tour (it slightly")
print("                        lags); the mild July softness is ordinary pan-European summer beta.")
