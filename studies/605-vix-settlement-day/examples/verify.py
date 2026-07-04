"""Reproducible headline run for Study 605 — VIX Settlement Day.

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; real-tape numbers come from the cached ^VIX/^GSPC
OHLC under ``_cache/`` (cache-first — refetched only if missing), and the synthetic control
always runs with no network.

    python examples/verify.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quantlab.repro import data_stamp  # noqa: E402
from vix_settlement_day import data, strategy as st  # noqa: E402


def pct(x: float) -> str:
    return f"{x * 100:+.3f}%"


print("# VIX Settlement Day — does the monthly settlement Wednesday leave tracks in ^VIX?")

n_known = data.verify_calendar()
sett = data.settlement_calendar()
tuesdays = [str(d.date()) for d in sett[sett.weekday == 1]]
print(f"calendar: {len(sett)} monthly settlements {sett.min().date()} -> {sett.max().date()}, "
      f"built by rule and verified against {n_known} known CBOE dates")
print(f"holiday-shifted Tuesday settlements: {tuesdays}")

if not data.have_real():
    print("(cache miss — fetching ^VIX/^GSPC once)")
    data.fetch()

vix, spx = data.load_real()
print(data_stamp("^VIX OHLC", vix, asof=data.AS_OF))
print(data_stamp("^GSPC OHLC", spx, asof=data.AS_OF))

df = st.day_frame(vix, spx, settlements=sett)
on_tape = int(df["sett"].sum())
overlap = int((df["sett"] & df["fomc"]).sum())
print(f"settlement days on tape: {on_tape}  |  other Wednesdays: "
      f"{int((df['wed'] & ~df['sett']).sum())}  |  settlements that are ALSO FOMC statement "
      f"days: {overlap}")

print("\n# Level tests — settlement days vs other Wednesdays (Welch t), ALL days")
lv = st.level_tests(df, ex_fomc=False)
for key, lbl in [("gap", "open gap  ln(O/C-1)"), ("abs_gap", "|open gap|"),
                 ("intr", "intraday  ln(C/O)"), ("abs_intr", "|intraday|"),
                 ("cc", "close-close"), ("abs_cc", "|close-close|"),
                 ("rng", "range (H-L)/C-1"), ("spx", "SPX close-close")]:
    v = lv[key]
    print(f"  {lbl:20s}: sett {pct(v['sett'])}  vs other-Wed {pct(v['other'])}   "
          f"Welch t = {v['t']:+.2f}")

print("\n# Same level tests EXCLUDING FOMC statement days (the Wednesday confounder)")
lx = st.level_tests(df, ex_fomc=True)
for key, lbl in [("abs_gap", "|open gap|"), ("abs_cc", "|close-close|"),
                 ("rng", "range"), ("spx", "SPX close-close")]:
    v = lx[key]
    print(f"  {lbl:20s}: sett {pct(v['sett'])}  vs other-Wed {pct(v['other'])}   "
          f"Welch t = {v['t']:+.2f}")
print(f"  (n = {lx['n_sett']} settlement days vs {lx['n_other']} other Wednesdays ex-FOMC)")

print("\n# THE SIGNATURE — settlement x gap continuation (intr = a + b*gap + c*sett + d*gap*sett)")
print("  White (HC1) robust t; sample = settlement days + other Wednesdays")
ia = st.interaction(df, ex_fomc=False)
ix = st.interaction(df, ex_fomc=True)
for lbl, r in (("ALL days ", ia), ("ex-FOMC  ", ix)):
    print(f"  {lbl}: base slope b = {r['base_slope']:+.3f} (t={r['base_t']:+.2f})  ->  "
          f"settlement slope b+d = {r['sett_slope']:+.3f}  |  interaction d = "
          f"{r['inter']:+.3f}  (t = {r['inter_t']:+.2f}, n={r['n']})")

print("\n# Random-calendar placebo (one fake settlement Wednesday per month, ex-FOMC)")
pl = st.placebo_interaction(df, ex_fomc=True, n_seeds=25, n_per=80)
print(f"  observed interaction {pl['obs']:+.3f}  vs placebo mean {pl['mean']:+.3f} "
      f"(sd {pl['sd']:.3f})  over {pl['n_draws']:,} draws / {pl['n_seeds']} seeds  ->  "
      f"two-sided p = {pl['p_two_sided']:.4f}")

print("\n# Robustness — the signature under data-quality and tail screens (ex-FOMC)")
r1 = st.interaction(df, ex_fomc=True, drop_zero_gaps=True)
r2 = st.interaction(df, ex_fomc=True, winsor=0.01)
print(f"  drop exact-zero gaps (Yahoo stale-open artefact): d = {r1['inter']:+.3f} "
      f"(t = {r1['inter_t']:+.2f}, n={r1['n']})")
print(f"  winsorise gap/intr at 1%/99%                    : d = {r2['inter']:+.3f} "
      f"(t = {r2['inter_t']:+.2f}, n={r2['n']})")

print("\n# Third axis — did it fade after Griffin-Shams (2018) + the manipulation lawsuits?")
sp = st.subperiod_interactions(df, ex_fomc=True)
for lbl, r in (("2004-2017", sp["pre"]), ("2018-2026", sp["post"])):
    print(f"  {lbl} signature: interaction d = {r['inter']:+.3f}  (t = {r['inter_t']:+.2f}, "
          f"n_sett = {r['n_sett']})")
sl = st.subperiod_levels(df)
for tag, lbl in (("all", "with FOMC"), ("ex_fomc", "ex-FOMC ")):
    a, b = sl[tag]["pre"], sl[tag]["post"]
    print(f"  |close-close| level effect ({lbl}): 2004-2017 t = {a['t']:+.2f}  ->  "
          f"2018-2026 t = {b['t']:+.2f}")

print("\n# Tradability arithmetic — ride the settlement-morning gap open->close (INDEX units)")
print("  the VIX index is NOT tradable; this is the paper value of the signature, then")
print("  VIX-futures-level costs (25 bps one-way x 2) are charged per event")
tp = st.continuation_pnl(df, ex_fomc=False, cost_bps_oneway=25.0)
print(f"  settlement days : gross {pct(tp['gross_per_event'])} /event  (t vs 0 = "
      f"{tp['t_vs_zero']:+.2f}, t vs other-Wed = {tp['t_vs_otherwed']:+.2f}, n={tp['n']})")
print(f"  other Wednesdays: {pct(tp['other_wed'])} /event")
print(f"  net of 2 x 25 bps: {pct(tp['net_per_event'])} /event x 12 events/yr = "
      f"{tp['net_per_year'] * 100:+.2f}%/yr of notional — before the futures-vs-spot basis")

print("\n# Synthetic control — machinery proof only, never market evidence")
print("  planted = extra settlement-day gap->close slope; null (0) must stay quiet")
for planted in (0.0, 0.5):
    world = data.synthetic_world(planted=planted, seed=605)
    sdf = st.day_frame(world, spx=None)
    r = st.interaction(sdf, ex_fomc=False)
    print(f"  planted d = {planted:+.2f}: recovered d = {r['inter']:+.3f}  "
          f"(t = {r['inter_t']:+.2f}, n={r['n']}, n_sett={r['n_sett']})")
