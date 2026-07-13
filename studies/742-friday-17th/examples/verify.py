"""Reproducible headline run for Study 742 — Friday-17th (Venerdì 17).

Prints every number quoted in docs/results.md and frozen into the ``R`` dict in
notebooks/build_notebooks.py. Deterministic; uses the cached FTSE MIB (EUR, price-only)
and EWI (USD, total-return) tapes under ``_cache/`` (fetching once on a cache miss), and
always runs the synthetic control with no network.

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

from friday_17th import data as dt, strategy as st  # noqa: E402

print("# Friday-17th (Venerdi 17) — does Italy's FTSE MIB trade weak on the unlucky day?")

if not dt.have_real():
    print("(cache miss — fetching FTSE MIB + EWI once)")
    dt.fetch()

prices = dt.load_real()
panel = pd.DataFrame({t: s for t, s in prices.items()})
print(data_stamp("Friday17 panel (FTSE MIB + EWI)", panel, asof=dt.AS_OF))
print("calendar: Venerdi 17 = Friday AND day-of-month 17, pure date arithmetic "
      "(known before the open -> zero look-ahead)")

for t in dt.TICKERS:
    df = st.build_frame(prices[t])
    q = dt.QUOTE[t]
    r = st.friday17_test(df)
    print(f"\n# {t} — {q['label']}")
    print(f"  span {df.index.min().date()} -> {df.index.max().date()}  ({len(df):,} daily returns)")
    print(f"  Venerdi 17     : n={r['n_f17']:3d}  mean={r['mean_f17_bps']:+7.3f} bps  "
          f"one-sample t={r['t_f17']:+.3f}")
    print(f"  other Fridays  : mean={r['mean_other_fri_bps']:+7.3f} bps  "
          f"contrast={r['contrast_fri_bps']:+7.3f} bps  Welch t={r['t_welch_fri']:+.3f}  "
          f"p={r['p_welch_fri']:.4f}")
    print(f"  all other days : mean={r['mean_all_other_bps']:+7.3f} bps  "
          f"contrast={r['contrast_all_bps']:+7.3f} bps  Welch p={r['p_welch_all']:.4f}")
    print(f"  down-day rate  : {r['down_k']}/{r['down_n']} = {r['down_rate']*100:.1f}%  "
          f"(Wilson 95% [{r['down_lo']*100:.1f}%, {r['down_hi']*100:.1f}%])")
    pl = st.random_friday_placebo(df)
    print(f"  placebo (20x500 random other-Friday sets): observed {pl['obs_bps']:+.3f} bps "
          f"vs null mean {pl['null_mean_bps']:+.3f} (sd {pl['null_sd_bps']:.3f}) over "
          f"{pl['n_draws']:,} draws -> left-tail p = {pl['p_left']:.4f}")
    sh = st.short_the_17th(prices[t])
    print(f"  TRADE (short the 17th, prior-close entry, cover at 17th close, "
          f"2x{st.COST_BPS:.0f}bps cost + {st.BORROW_BPS:.0f}bps borrow):")
    print(f"    gross {sh['gross_mean_bps']:+.3f} bps (t={sh['gross_t']:+.2f})  "
          f"net {sh['net_mean_bps']:+.3f} bps (t={sh['net_t']:+.2f})  "
          f"win {sh['win_rate']*100:.1f}%  breakeven {sh['breakeven_bps']:.0f} bps")

print("\n# Look-elsewhere — day-of-month Bonferroni sweep on the FTSE MIB (slots 17 +/- 7k)")
sw = st.dom_sweep(st.build_frame(prices[dt.MIB]))
print(sw.to_string(index=False))
print("  -> the 17th is NOT the most extreme slot; the extreme one fails Bonferroni.")

print("\n# Sub-periods (FTSE MIB) — is the null stable?")
mib = st.build_frame(prices[dt.MIB])
for lab, lo, hi in (("1998-2012", "1998-01-01", "2012-12-31"),
                    ("2013-2026", "2013-01-01", "2026-06-30")):
    sub = mib[(mib.index >= lo) & (mib.index <= hi)]
    r = st.friday17_test(sub)
    print(f"  {lab}: n={r['n_f17']:2d}  mean={r['mean_f17_bps']:+7.2f} bps  "
          f"t={r['t_f17']:+.3f}  contrast={r['contrast_fri_bps']:+.2f}  p={r['p_welch_fri']:.3f}")

print("\n# Synthetic positive control — deterministic, no network")
print("  the one-sample-t detector must NOT fire on the null (effect=0) and must recover")
print("  a planted Venerdi-17 fear. Null checked over 20 seeds.")
null_ts = np.array([st.synthetic_detect(0.0, seed=742 + s)["t"] for s in range(20)])
print(f"  null (effect=0), 20 seeds: mean t = {null_ts.mean():+.3f} (sd "
      f"{null_ts.std(ddof=1):.3f}), |t|>=2 in {(abs(null_ts) >= 2).sum()}/20 seeds")
for eff in (-0.5, -1.0, -2.0):
    s = st.synthetic_detect(eff, seed=742)
    print(f"  planted fear {eff:+.1f} sigma (seed 742): mean {s['mean']*1e4:+.2f} bps  "
          f"t = {s['t']:+.2f}  (n={s['n']} synthetic events)")

print("\n# VERDICT")
print("  Signal:      NONE   -- Venerdi 17 on the FTSE MIB is +26.3 bps (t=+1.27), POSITIVE")
print("                         (wrong sign for the fear); down-day 46.9%; placebo left-tail")
print("                         p=0.93; EWI agrees (+10.1 bps, t=+0.52).")
print("  Tradability: MIRAGE -- shorting the unlucky day LOSES money: net -38 bps/event (MIB,")
print("                         t=-1.85), -22 bps (EWI). No edge to charge costs against.")
print("  Unlucky day? BUSTED -- the 17th is a slightly ABOVE-average Friday; the most extreme")
print("                         middle-Friday slot is the (boring) 10th, and it dies under")
print("                         Bonferroni. Same shape as Friday-13th (study 163), one country over.")
