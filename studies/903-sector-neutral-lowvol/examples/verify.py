"""Reproducible headline run for Study 903 — Sector-Neutral Low-Vol.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached cross-section panel under
``_cache/`` (fetching once on a cache miss through the quantlab.universe survivorship
guard), and always runs the synthetic control with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys
import warnings

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from quantlab.repro import fingerprint  # noqa: E402

from sn_lowvol import data, strategy as st  # noqa: E402

warnings.filterwarnings("ignore")
TD = 252


def leg_sharpe(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    sd = x.std(ddof=1)
    return x.mean() / sd * np.sqrt(TD) if sd > 0 else float("nan")


def leg_vol(x):
    x = np.asarray(x, dtype=float); x = x[~np.isnan(x)]
    return x.std(ddof=1) * np.sqrt(TD)


print("# Sector-Neutral Low-Vol — does the low-vol edge survive once the sector bet is stripped?")

if not data.have_real():
    print("(cache miss — fetching the cross-section panel once, through the survivorship guard)")
    data.fetch()

panel = data.load_panel()
sectors = data.sector_series(panel)
closes = pd.DataFrame({s: panel[s]["Close"] for s in data.UNIVERSE if s in panel})
print(f"[data] {len(panel)} names, {len(closes)} rows  "
      f"{closes.index.min().date()} -> {closes.index.max().date()}  as-of {data.AS_OF}  "
      f"fingerprint(Close)={fingerprint(closes)}")
print(f"  sectors: {dict(sectors.value_counts())}")
print("  SURVIVORSHIP: current-membership mega-cap panel — magnitudes are an upper bound "
      "(delisted names absent). Named on the Signal axis.")

ret = st.close_returns(panel)
WIN, FRAC = 63, 0.3

for neutral in (False, True):
    tag = "SECTOR-NEUTRAL" if neutral else "RAW (sector-tilted)"
    sp = st.vol_spreads(ret, sectors, window=WIN, frac=FRAC, neutral=neutral)
    h = st.vol_stats(sp)
    dt = st.defensive_tilt(ret, sectors, data.DEFENSIVE_SECTORS, WIN, FRAC, neutral)
    print(f"\n=== {tag}: trailing-{WIN}d vol, long bottom{int(FRAC*100)}% / short top{int(FRAC*100)}%, "
          f"{h['n_days']} days (median {int(sp['n'].median())} names/day) ===")
    print(f"  spread (low-vol - high-vol): {h['spread_bps']:+.2f} bps/day  "
          f"NW(10) t = {h['t_nw']:+.2f}  one-sample t = {h['t_1s']:+.2f}")
    print(f"  books : low-vol {h['lo_bps']:+.2f} vs high-vol {h['hi_bps']:+.2f} bps  "
          f"(Welch t = {h['welch_t']:+.2f})  gross spread Sharpe {h['gross_sharpe']:.2f}")
    print(f"  Sharpe race (each leg, excess-of-cash~raw): "
          f"low-vol {leg_sharpe(sp['lo']):.2f} (vol {leg_vol(sp['lo']):.2f}) vs "
          f"high-vol {leg_sharpe(sp['hi']):.2f} (vol {leg_vol(sp['hi']):.2f})")
    print(f"  DEFENSIVE tilt: long book {dt['long_defensive_share']:.1%} defensive vs "
          f"short {dt['short_defensive_share']:.1%} (universe {dt['universe_defensive_share']:.1%}); "
          f"long-short = {dt['long_minus_short_defensive']:+.1%}")

print("\n# THE HEADLINE COMPARISON")
raw = st.vol_stats(st.vol_spreads(ret, sectors, WIN, FRAC, neutral=False))
neu = st.vol_stats(st.vol_spreads(ret, sectors, WIN, FRAC, neutral=True))
print(f"  raw spread {raw['spread_bps']:+.2f} bps (t={raw['t_nw']:+.2f})  ->  "
      f"sector-neutral {neu['spread_bps']:+.2f} bps (t={neu['t_nw']:+.2f})")
print(f"  removing the sector bet changed the spread by "
      f"{neu['spread_bps'] - raw['spread_bps']:+.2f} bps; the (wrong-signed) stock-level "
      f"effect {'remains' if neu['t_nw'] < -2 else 'weakens'}.")

sp = st.vol_spreads(ret, sectors, WIN, FRAC, neutral=True)
print("\n# PLACEBO (sector-neutral) — column-permute the forward returns (1,000 permutations)")
pl = st.placebo_pvalue(ret, sectors, WIN, FRAC, neutral=True, n_seeds=20, n_draws_per_seed=50)
sig = (pl["obs_bps"] - pl["placebo_mean_bps"]) / pl["placebo_sd_bps"]
print(f"  observed {pl['obs_bps']:+.2f} bps vs placebo mean {pl['placebo_mean_bps']:+.3f} "
      f"(sd {pl['placebo_sd_bps']:.3f}) over {pl['n_draws']:,} draws -> "
      f"~{abs(sig):.2f}sigma into the left tail (right-tail p = {pl['p_value']:.4f})")

print("\n# ROBUSTNESS — two eras (split 2018-01-01), sector-neutral")
for lo, hi, lbl in [("2010-01-01", "2018-01-01", "2010-2017"),
                    ("2018-01-01", "2026-07-01", "2018-2026")]:
    sub = sp[(sp.index >= lo) & (sp.index < hi)]
    ts = st.vol_stats(sub)
    print(f"  {lbl}: n={ts['n_days']}  spread {ts['spread_bps']:+.2f} bps (NW t={ts['t_nw']:+.2f})")

print("\n# THE TIMER — sector-neutral long-low-vol / short-high-vol, costed")
print("  2 sides x one-way cost x NAV per day; short (high-vol) book pays 50 bps/yr borrow")
for cb in (1.0, 5.0):
    tm = st.timer_stats(sp, cost_bps=cb, borrow_bps_yr=50.0)
    print(f"  cost={cb:>4.1f} bps/side: gross {tm['gross_bps']:+.2f} -> net {tm['net_bps']:+.2f} "
          f"bps/day (cost {tm['cost_bps_per_day']:.2f}/day, t = {tm['t_net']:+.2f}, "
          f"Sharpe {tm['sharpe_net']:.2f}, ~{tm['ann_net_pct']:+.1f}%/yr)")

print("\n# SYNTHETIC POSITIVE CONTROL — deterministic, no network")
secmap = pd.Series(data.synthetic_sectors(40, 8))
null_t = np.array([st.synthetic_detect(
    data.synthetic_panel(edge=0.0, seed=903 + s, n_assets=40, n_days=1200, n_sectors=8),
    secmap, neutral=True)["t_nw"] for s in range(20)])
print(f"  null (edge=0), 20 seeds, sector-neutral sort: NW t mean {null_t.mean():+.2f} "
      f"(sd {null_t.std(ddof=1):.2f}), |t|>=2 in {(abs(null_t) >= 2).sum()}/20")
p1 = data.synthetic_panel(edge=0.1, seed=903, n_assets=40, n_days=1500, n_sectors=8)
sy = st.synthetic_detect(p1, secmap, neutral=True)
print(f"  planted stock-level low-vol effect (edge=0.1), sector-neutral: NW t = {sy['t_nw']:+.2f}, "
      f"Welch t = {sy['welch_t']:+.2f}")
raw_t = np.array([st.synthetic_detect(
    data.synthetic_panel(edge=0.0, seed=903 + s, n_assets=40, n_days=1200, n_sectors=8),
    secmap, neutral=False)["t_nw"] for s in range(20)])
print(f"  CONFOUND CHECK (edge=0, only a sector premium present): RAW sort fires "
      f"(NW t mean {raw_t.mean():+.2f}, |t|>=2 in {(abs(raw_t) >= 2).sum()}/20) but the "
      f"SECTOR-NEUTRAL sort stays silent ({(abs(null_t) >= 2).sum()}/20) — the demean strips the sector bet.")
