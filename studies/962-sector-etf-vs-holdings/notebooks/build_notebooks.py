"""Generate the two narrative notebooks for Study 962 (Do It Yourself).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Every real-tape number is quoted from the
frozen ``R`` dict below (a mirror of docs/results.md); the only live cells run the fast
synthetic control, and they are never presented under a real-tape banner.
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# Frozen real-tape headline — mirror of docs/results.md. XLK/XLE/XLF + 44 constituent
# names + BIL, daily total return, 2011-01-03 -> 2026-06-30, monthly, 5 bps one-way.
R = dict(
    start="2011-01-03", end="2026-06-30", n_days=3895, fp="1f3a3deb14a7",
    n_tickers=48, fee=0.08,
    # blended headline (equal-weight across the three sectors, top-10)
    blend_gap=-0.76, blend_t=-0.58, blend_te=5.13,
    blend_ci_lo=-3.21, blend_ci_hi=1.69, blend_worst12=-12.3,
    look_gap=6.08, look_t=6.10, look_te=3.85,
    look_ci_lo=4.07, look_ci_hi=8.11, look_worst12=-8.1,
    raw_premium=6.84,          # cap2026 - eq2011: TWO changes at once, not the look-ahead
    # decomposition against the eq2026 control (today's names, EQUAL weight)
    eq2026_gap=5.06, eq2026_t=5.13,
    vintage_premium=5.81,      # eq2026 - eq2011, weighting held fixed = THE look-ahead
    weighting_premium=1.02,    # cap2026 - eq2026, names held fixed = not hindsight
    decomp={"XLK": (-3.94, 4.16, 9.83), "XLE": (-0.30, 5.56, 3.35),
            "XLF": (2.07, 5.46, 5.17)},   # (eq2011, eq2026, cap2026) per sector
    # per sector, contemporaneous top-10
    xlk_gap=-3.94, xlk_t=-1.86, xlk_te=9.13, xlk_ci_lo=-7.77, xlk_ci_hi=0.10,
    xlk_corr=0.915, xlk_beta=0.76, xlk_sh_diy=0.86, xlk_sh_fund=0.90,
    xlk_dd_diy=-29.9, xlk_dd_fund=-33.6, xlk_worst12=-23.3,
    xle_gap=-0.30, xle_t=-0.12, xle_te=9.75, xle_ci_lo=-5.24, xle_ci_hi=4.58,
    xle_corr=0.969, xle_beta=1.19, xle_sh_diy=0.25, xle_sh_fund=0.32,
    xle_dd_diy=-81.8, xle_dd_fund=-71.3, xle_worst12=-25.7,
    xlf_gap=2.07, xlf_t=1.32, xlf_te=6.25, xlf_ci_lo=-0.94, xlf_ci_hi=5.14,
    xlf_corr=0.976, xlf_beta=1.14, xlf_sh_diy=0.55, xlf_sh_fund=0.55,
    xlf_dd_diy=-47.3, xlf_dd_fund=-42.9, xlf_worst12=-11.1,
    # depth x weighting sweep: (gap, t) per fund/scheme/N
    sweep={
        ("XLK", "hindsight"): [(13.62, 3.66), (12.93, 4.16), (9.83, 3.89)],
        ("XLK", "contemp"): [(-1.75, -0.72), (-2.90, -1.19), (-3.94, -1.86)],
        ("XLE", "hindsight"): [(1.46, 0.72), (2.46, 1.57), (3.35, 2.82)],
        ("XLE", "contemp"): [(-0.74, -0.42), (-0.12, -0.07), (-0.30, -0.12)],
        ("XLF", "hindsight"): [(4.62, 2.52), (5.47, 3.20), (5.17, 4.02)],
        ("XLF", "contemp"): [(2.25, 1.32), (2.17, 1.13), (2.07, 1.32)],
    },
    # power arithmetic
    se_ann=1.31, se_over_fee=16, years_needed=16472, sample_years=15.4,
    # era cut (split 2019-01-01)
    era={"XLK": ((-1.35, -0.82, 5.02), (-6.94, -1.73, 12.04)),
         "XLE": ((-1.42, -0.74, 5.72), (0.80, 0.17, 12.72)),
         "XLF": ((-0.25, -0.17, 4.38), (4.46, 1.58, 7.76))},
    # cost sweep (XLK contemporaneous top-10): cost_bps -> (gap, t)
    cost={0.0: (-3.92, -1.85), 5.0: (-3.94, -1.86),
          25.0: (-4.04, -1.91), 50.0: (-4.17, -1.97)},
    # rebalance sweep: freq -> (gap, t, te, turnover)
    freq={"M": (-3.94, -1.86, 9.13, 0.52), "Q": (-3.65, -1.73, 9.01, 0.31),
          "A": (-3.66, -1.75, 8.94, 0.14), "N": (-3.79, -2.38, 7.08, 0.00)},
    # calendar-year gaps (pp) for the three contemporaneous top-10 baskets
    cal_years=[2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018,
               2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026],
    cal_xlk=[6.9, -0.9, -2.8, -3.7, -4.2, 3.3, -12.8, 1.7,
             -10.2, -24.0, -7.6, 6.3, -20.5, -4.0, 8.6, -10.0],
    cal_xle=[-4.2, -4.3, 0.9, 2.2, -2.9, 5.1, -2.7, -6.6,
             -8.2, 3.8, 12.2, 10.8, -6.2, -20.2, -2.7, 0.4],
    cal_xlf=[-5.1, 5.3, 5.4, -1.6, -2.8, 0.2, -1.1, -2.0,
             4.3, -6.8, 0.5, 0.5, 2.9, 7.7, 14.0, 8.2],
    # synthetic control
    syn_fee=2.00, syn_pl_gap=1.957, syn_pl_sd=0.131, syn_pl_t=12.52, syn_pl_fire=6,
    syn_nl_gap=-0.062, syn_nl_sd=0.128, syn_nl_t=-0.39, syn_nl_fire=0, syn_te=0.69,
)


HEADER = f"""# Study 962 — Do It Yourself 🧰

**A sector fund is mostly a handful of mega-caps. So why not just buy them and skip the fee?**

The Technology Select Sector SPDR keeps about two thirds of its money in ten names. With
zero-commission trading and fractional shares, the forum argument writes itself: hold the
ten, skip the **{R['fee']:.2f}%/yr** expense ratio, keep the difference.

We build that basket for **XLK**, **XLE** and **XLF** at depths 3 / 5 / 10, rebalance it
monthly with a one-day execution lag, charge 5 bps one-way on every trade, and race it
against the fund it is meant to replace. Daily total-return closes, {R['n_tickers']} tickers,
{R['start']} → {R['end']} ({R['n_days']:,} days).

*Real-tape numbers below are the frozen headline (`docs/results.md`, Fingerprint
`{R['fp']}`); the live cells run only the offline synthetic control. As-of 2026-06-30.*
"""


def build_curious():
    nb = new_notebook()
    cells = [
        md(HEADER),
        md("## 1. The argument, and the trap inside it\n\n"
           "The argument is arithmetically fine *if* the ten names behave like the fund. "
           "The trap is in the phrase \"the ten names\". Which ten? If you look up the "
           "holdings today and run them backwards, you have quietly given your 2011 self "
           "a list of the companies that would go on to dominate the next fifteen years.\n\n"
           "So we run it both ways: **hindsight** (today's published weights, applied from "
           "2011) and **contemporaneous** (the top ten as published in January 2011, equal "
           "weight, fixed from day one). Same machinery, same costs, same lag."),
        code(
            "R = dict(look_gap=%r, look_t=%r, blend_gap=%r, blend_t=%r, eq2026_gap=%r,\n"
            "         raw_premium=%r, vintage_premium=%r, weighting_premium=%r,\n"
            "         blend_te=%r, fee=%r)\n"
            "print('hindsight basket      : %%+.2f%%%% a year vs the fund   (t = %%+.2f)'\n"
            "      %% (R['look_gap'], R['look_t']))\n"
            "print('contemporaneous basket: %%+.2f%%%% a year vs the fund   (t = %%+.2f)'\n"
            "      %% (R['blend_gap'], R['blend_t']))\n"
            "print()\n"
            "print('raw difference between them          : %%+.2f%%%% a year' %% R['raw_premium'])\n"
            "print('  of which knowing the names (look-ahead): %%+.2f%%%%' %% R['vintage_premium'])\n"
            "print('  of which just cap- vs equal-weighting  : %%+.2f%%%%' %% R['weighting_premium'])"
            % (R["look_gap"], R["look_t"], R["blend_gap"], R["blend_t"], R["eq2026_gap"],
               R["raw_premium"], R["vintage_premium"], R["weighting_premium"],
               R["blend_te"], R["fee"])
        ),
        md("> ⚖️ **Why the split matters.** The hindsight basket changes **two** things at "
           "once — *which* names, and *how* they are weighted. So we ran a third basket: "
           "today's names at **equal** weight. That pins the weighting scheme and leaves "
           "only the hindsight, which is worth "
           f"**{R['vintage_premium']:+.2f}%/yr** — not the raw {R['raw_premium']:+.2f}%. "
           "The leftover is cap-weighting, and it is not a peek at the answer sheet."),
        md(f"## 2. What you were actually hunting\n\n"
           f"The prize is the expense ratio: **{R['fee']:.2f}% a year**. Eight basis points. "
           f"The bill is the **tracking error** — how far your ten names wander from the fund "
           f"in a typical year. Blended across the three sectors that is **{R['blend_te']:.2f}% "
           f"a year**; inside technology and energy it is **{R['xlk_te']:.1f}%** and "
           f"**{R['xle_te']:.1f}%**.\n\n"
           f"That is not a rounding error on the fee. It is **{R['blend_te'] / R['fee']:.0f} times** "
           f"the fee, and it points in both directions."),
        code(
            "for tag, te in [('blended top-10', %r), ('XLK top-10', %r), ('XLE top-10', %r)]:\n"
            "    print('%%-15s tracking error %%5.2f%%%%/yr  = %%3.0fx the %%.2f%%%% fee you saved'\n"
            "          %% (tag, te, te / %r, %r))"
            % (R["blend_te"], R["xlk_te"], R["xle_te"], R["fee"], R["fee"])
        ),
        md("> 🔬 **For the quants.** The standard error on the annualised gap is "
           f"**{R['se_ann']:.2f}%/yr** — about **{R['se_over_fee']}×** the fee under test. At this "
           f"tracking error you would need roughly **{R['years_needed']:,} years** of tape for the "
           f"fee saving to reach |*t*| = 2. The honest statement is not \"the saving isn't "
           "there\"; it is that no test of this shape can ever see it."),
        md("## 3. What 5% tracking error feels like, year by year\n\n"
           "Averages hide the experience. Here is the gap between each do-it-yourself "
           "basket and its fund, calendar year by calendar year, in percentage points of "
           "total return. Double-digit misses in **both** directions are routine — you are "
           "not saving eight basis points, you are running an undiagnosed active bet."),
        code(
            "years = %r\nxlk = %r\nxle = %r\nxlf = %r\n"
            "print('year    XLK     XLE     XLF')\n"
            "for y, a, b, c in zip(years, xlk, xle, xlf):\n"
            "    star = '  <-- double-digit miss' if min(a, b, c) <= -10 else ''\n"
            "    print('%%d  %%+6.1f  %%+6.1f  %%+6.1f%%s' %% (y, a, b, c, star))\n"
            "print('\\n(2026 is a half-year, to 2026-06-30)')"
            % (R["cal_years"], R["cal_xlk"], R["cal_xle"], R["cal_xlf"])
        ),
        md(f"## 4. And you inherit the concentration\n\n"
           f"Ten names are not a sector. The energy basket's worst twelve months cost "
           f"**{R['xle_worst12']:.1f}%** relative to simply owning XLE, and its worst drawdown was "
           f"**{R['xle_dd_diy']:.1f}%** against the fund's **{R['xle_dd_fund']:.1f}%** — over ten "
           f"percentage points deeper. None of that risk is paid for: the excess-of-cash Sharpe "
           f"advantage over the fund is **{R['xlk_sh_diy'] - R['xlk_sh_fund']:+.2f}** in tech, "
           f"**{R['xle_sh_diy'] - R['xle_sh_fund']:+.2f}** in energy, "
           f"**{R['xlf_sh_diy'] - R['xlf_sh_fund']:+.2f}** in financials."),
        md("## 5. Live check — the machinery is unbiased (offline synthetic)\n\n"
           "Could the harness simply be blind to fees? No. On a synthetic sector world "
           "where the fund skims a deliberately fat 2%/yr, the same code recovers most of "
           "that fee with an enormous *t*. Switch the fee off and it recovers "
           "nothing. So the real-tape non-result is a fact about markets, not a bug."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "from diy_sector import data, strategy as st\n"
            "p1, t1 = data.synthetic_panel(signal_strength=1.0, seed=962)\n"
            "p0, t0 = data.synthetic_panel(signal_strength=0.0, seed=962)\n"
            "d1 = st.synthetic_detect(p1, t1, cost_bps=0.0)\n"
            "d0 = st.synthetic_detect(p0, t0, cost_bps=0.0)\n"
            "print('fund skims %.2f%%/yr -> basket recovers %+.2f%%/yr (t = %+.2f)'\n"
            "      % (t1['planted_fee_ann']*100, d1['ann_gap']*100, d1['t_diff']))\n"
            "print('fund skims  0.00%%/yr -> basket recovers %+.2f%%/yr (t = %+.2f)'\n"
            "      % (d0['ann_gap']*100, d0['t_diff']))"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** With a contemporaneous holdings list the gap is "
           f"**{R['blend_gap']:+.2f}%/yr** (HAC *t* = {R['blend_t']:+.2f}), the confidence interval "
           f"[{R['blend_ci_lo']:+.2f}%, {R['blend_ci_hi']:+.2f}%] straddles zero, and no sector "
           f"clears |*t*| = 2. The only impressive number in the study — "
           f"**{R['look_gap']:+.2f}%/yr, *t* = {R['look_t']:+.2f}** — belongs to the hindsight "
           f"basket, of which **{R['vintage_premium']:+.2f}%/yr** is pure look-ahead and "
           f"{R['weighting_premium']:+.2f}%/yr is just cap-weighting.\n"
           f"- **Tradability — Mirage.** You would be swapping a certain "
           f"**{R['fee']:.2f}%/yr** saving for **{R['blend_te']:.2f}%/yr** of two-sided tracking "
           f"error, a worst twelve months of **{R['blend_worst12']:.1f}%**, and a deeper drawdown "
           f"in the sector where it hurt most. Buy the fund.")
    ]
    nb["cells"] = cells
    return nb


def build_quants():
    nb = new_notebook()
    cells = [
        md("# Study 962 — Do It Yourself — the teardown\n\n"
           "The depth × weighting sweep, the HAC *t* on the daily gap, block-bootstrap CIs, "
           "the power arithmetic, the era cut, the cost and rebalance-frequency sweeps, and "
           "the live synthetic control. Every real number is frozen from `docs/results.md` "
           "(Fingerprint `%s`), sample %s → %s, %s days, monthly rebalance, one-day "
           "execution lag, 5 bps one-way × NAV, long-only (no borrow leg)."
           % (R["fp"], R["start"], R["end"], f"{R['n_days']:,}")),
        code("R = %r" % (R,)),
        md("## 1. The identification problem\n\n"
           "A top-N basket built from **today's** holdings list is a survivorship portfolio: "
           "the 2011 investor could not have known it. The study's whole design is the "
           "contrast between that basket and one built from the **January-2011** list, held "
           "equal weight and fixed from the first day of the sample.\n\n"
           "> 💡 **In plain words:** one basket is allowed to peek at the answer sheet; the "
           "other is not."),
        code(
            "print('depth x weighting sweep — annualised gap (HAC t), monthly, 5 bps')\n"
            "print('%-6s %-12s %14s %14s %14s' % ('fund','scheme','N=3','N=5','N=10'))\n"
            "for fund in ('XLK','XLE','XLF'):\n"
            "    for scheme in ('hindsight','contemp'):\n"
            "        cells = R['sweep'][(fund, scheme)]\n"
            "        print('%-6s %-12s ' % (fund, scheme) +\n"
            "              ' '.join('%+8.2f%% (%+.2f)' % (g, t) for g, t in cells))\n"
            "hits_look = sum(abs(t) >= 2 for k, v in R['sweep'].items() if k[1]=='hindsight' for _, t in v)\n"
            "hits_cont = sum(abs(t) >= 2 for k, v in R['sweep'].items() if k[1]=='contemp' for _, t in v)\n"
            "print('\\n|t| >= 2 cells: hindsight %d/9, contemporaneous %d/9' % (hits_look, hits_cont))"
        ),
        md("## 2. The blended headline, and what the raw premium actually contains\n\n"
           "One number per scheme: the equal-weight blend of the three sectors' top-10 gap "
           "series, with a 21-day circular block-bootstrap CI on the annualised mean.\n\n"
           "The naive move here is to call the whole hindsight-minus-contemporaneous "
           "difference the look-ahead. It is not: `cap2026` and `eq2011` differ in **two** "
           "factors at once — the *membership vintage* and the *weighting scheme*. The "
           "`eq2026` basket (today's names, **equal** weight) is the control that "
           "identifies them separately.\n\n"
           "> 💡 **In plain words:** two knobs were turned, so measure them one at a time."),
        code(
            "print('%-22s %10s %8s %8s %20s %10s' % ('basket','ann gap','HAC t','TE','95% CI','worst 12m'))\n"
            "print('%-22s %+9.2f%% %+8.2f %7.2f%% [%+6.2f%%, %+6.2f%%] %9.1f%%'\n"
            "      % ('contemporaneous 2011', R['blend_gap'], R['blend_t'], R['blend_te'],\n"
            "         R['blend_ci_lo'], R['blend_ci_hi'], R['blend_worst12']))\n"
            "print('%-22s %+9.2f%% %+8.2f %7.2f%% [%+6.2f%%, %+6.2f%%] %9.1f%%'\n"
            "      % ('hindsight 2026', R['look_gap'], R['look_t'], R['look_te'],\n"
            "         R['look_ci_lo'], R['look_ci_hi'], R['look_worst12']))\n"
            "print('%-22s %+9.2f%% %+8.2f' % ('eq2026 control', R['eq2026_gap'], R['eq2026_t']))\n"
            "\n"
            "print('\\ndecomposition of the %+.2f%%/yr raw difference' % R['raw_premium'])\n"
            "print('  name vintage (eq2026 - eq2011, weighting fixed): %+.2f%%/yr  <- the look-ahead'\n"
            "      % R['vintage_premium'])\n"
            "print('  weighting    (cap2026 - eq2026, names fixed)   : %+.2f%%/yr  <- not hindsight'\n"
            "      % R['weighting_premium'])\n"
            "print('\\nper sector: eq2011 -> eq2026 -> cap2026   (vintage | weighting)')\n"
            "for f, (a, b, c) in R['decomp'].items():\n"
            "    print('  %-4s %+6.2f%% -> %+6.2f%% -> %+6.2f%%   (%+6.2f | %+6.2f)'\n"
            "          % (f, a, b, c, b - a, c - b))\n"
            "print('\\nvintage term is positive in 3/3 sectors; the weighting term is not.')"
        ),
        md("## 3. Per sector — the excess-of-cash race and the risk you inherit\n\n"
           "Both legs are measured excess of BIL's actual total return, so the cash leg "
           "cancels in the difference and the Sharpe comparison is like for like."),
        code(
            "rows = [('XLK', R['xlk_gap'], R['xlk_t'], R['xlk_te'], R['xlk_corr'], R['xlk_beta'],\n"
            "         R['xlk_sh_diy'], R['xlk_sh_fund'], R['xlk_dd_diy'], R['xlk_dd_fund'], R['xlk_worst12']),\n"
            "        ('XLE', R['xle_gap'], R['xle_t'], R['xle_te'], R['xle_corr'], R['xle_beta'],\n"
            "         R['xle_sh_diy'], R['xle_sh_fund'], R['xle_dd_diy'], R['xle_dd_fund'], R['xle_worst12']),\n"
            "        ('XLF', R['xlf_gap'], R['xlf_t'], R['xlf_te'], R['xlf_corr'], R['xlf_beta'],\n"
            "         R['xlf_sh_diy'], R['xlf_sh_fund'], R['xlf_dd_diy'], R['xlf_dd_fund'], R['xlf_worst12'])]\n"
            "print('%-5s %8s %7s %7s %6s %6s %14s %16s %10s'\n"
            "      % ('fund','gap','HAC t','TE','corr','beta','exSharpe D/F','maxDD D/F','worst12m'))\n"
            "for f, g, t, te, c, b, sd, sf, dd, df, w in rows:\n"
            "    print('%-5s %+7.2f%% %+7.2f %6.2f%% %6.3f %6.2f  %+5.2f/%+5.2f (%+.2f) %6.1f%%/%6.1f%% %9.1f%%'\n"
            "          % (f, g, t, te, c, b, sd, sf, sd - sf, dd, df, w))"
        ),
        md("## 4. Power — the test cannot see an 8 bp fee\n\n"
           "This is the finding, not a caveat. The replication noise is three orders of "
           "magnitude larger than the quantity being estimated.\n\n"
           "> 💡 **In plain words:** you are trying to hear a whisper next to a jet engine."),
        code(
            "te, fee, yrs = R['blend_te'], R['fee'], R['sample_years']\n"
            "se = te / yrs**0.5\n"
            "print('blended TE            : %.2f%%/yr' % te)\n"
            "print('sample length         : %.1f yr' % yrs)\n"
            "print('SE of annualised gap  : %.2f%%/yr  (= %.0fx the %.2f%% fee)' % (se, se/fee, fee))\n"
            "print('years needed for |t|=2: %s' % format(int(round((2*te/fee)**2)), ','))\n"
            "print('\\nfrozen values: SE %.2f%%, ratio %dx, years %s'\n"
            "      % (R['se_ann'], R['se_over_fee'], format(R['years_needed'], ',')))"
        ),
        md("## 5. Era cut (split 2019-01-01)\n\n"
           "No sector holds its sign, and the tracking error roughly doubles in the recent "
           "era: the funds have grown more top-heavy, so a list fixed in 2011 drifts further "
           "from the thing it is replacing."),
        code(
            "print('%-5s %26s %26s' % ('fund','2011-2018  gap (t) / TE','2019-2026  gap (t) / TE'))\n"
            "for f, ((g1,t1,e1),(g2,t2,e2)) in R['era'].items():\n"
            "    print('%-5s %+14.2f%% (%+.2f) %5.2f%% %+14.2f%% (%+.2f) %5.2f%%'\n"
            "          % (f, g1, t1, e1, g2, t2, e2))"
        ),
        md("## 6. Cost and rebalance sweeps (XLK, contemporaneous top-10)\n\n"
           "Turnover is ~0.5×/yr, so friction decides nothing — gross and net differ by 2 bps "
           "at the house rate. Long-only, so there is no borrow rate to sweep.\n\n"
           "Across the **full** contemporaneous grid — 3 funds × 3 depths × 4 frequencies = "
           "**36 cells** — exactly one clears |*t*| = 2: the never-rebalanced XLK top-10, at "
           "−2.38, *losing* to its fund. We do not bank that either. Thirty-six correlated "
           "cells at 5% size would be expected to throw up **≈1.8** hits under a true null, "
           "so one hit is if anything fewer than noise predicts.\n\n"
           "> 💡 **In plain words:** we searched hard for a winner and did not even find "
           "enough losers to be surprising."),
        code(
            "print('one-way cost sweep')\n"
            "for c in sorted(R['cost']):\n"
            "    g, t = R['cost'][c]\n"
            "    print('  %5.1f bps: gap %+.2f%% (t %+.2f)' % (c, g, t))\n"
            "print('\\nrebalance-frequency sweep')\n"
            "names = {'M':'monthly','Q':'quarterly','A':'annual','N':'never'}\n"
            "for k in ('M','Q','A','N'):\n"
            "    g, t, te, tu = R['freq'][k]\n"
            "    flag = '  <-- |t| >= 2, wrong sign' if abs(t) >= 2 else ''\n"
            "    print('  %-10s gap %+.2f%% (t %+.2f)  TE %5.2f%%  turnover %.2f/yr%s'\n"
            "          % (names[k], g, t, te, tu, flag))"
        ),
        md("## 7. Live synthetic control — the detector is unbiased\n\n"
           "A one-factor sector world: ten big names, a forty-name tail the basket omits, and "
           "a fund that charges a **deliberately fat** 2%/yr against a **deliberately thin** "
           "10% tail — the only regime in which a test of this shape has power. The detector "
           "must recover the planted fee and stay silent when it is switched off."),
        code(
            "import os, sys\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "sys.path.insert(0, os.path.abspath(os.path.join('..','..','..')))\n"
            "import numpy as np\n"
            "from diy_sector import data, strategy as st\n"
            "for ss, tag in [(1.0, 'planted fee'), (0.0, 'null (no fee)')]:\n"
            "    gaps, ts = [], []\n"
            "    for s in range(4):\n"
            "        p, t = data.synthetic_panel(n_years=10, signal_strength=ss, seed=962+s)\n"
            "        d = st.synthetic_detect(p, t, cost_bps=0.0)\n"
            "        gaps.append(d['ann_gap']); ts.append(d['t_diff'])\n"
            "    gaps, ts = np.array(gaps), np.array(ts)\n"
            "    print('%-14s planted %.2f%%/yr -> gap %+.3f%% (sd %.3f%%), mean t %+.2f, |t|>=2 in %d/4'\n"
            "          % (tag, t['planted_fee_ann']*100, gaps.mean()*100, gaps.std(ddof=1)*100,\n"
            "             ts.mean(), (np.abs(ts) >= 2).sum()))"
        ),
        md(f"## Verdict\n\n"
           f"- **Signal — None.** Contemporaneous blend **{R['blend_gap']:+.2f}%/yr**, HAC *t* = "
           f"**{R['blend_t']:+.2f}**, bootstrap CI [{R['blend_ci_lo']:+.2f}%, {R['blend_ci_hi']:+.2f}%] "
           f"straddling zero; 0/9 contemporaneous cells clear |*t*| = 2 against 7/9 hindsight "
           f"cells; no sign survives the era cut. Decomposed against the `eq2026` control, the "
           f"look-ahead premium is **{R['vintage_premium']:+.2f}%/yr** (positive in 3/3 sectors) "
           f"and the remaining {R['weighting_premium']:+.2f}%/yr of the raw "
           f"{R['raw_premium']:+.2f}% is cap- versus equal-weighting, which is not hindsight at "
           f"all — it is +5.67 pp in XLK and *negative* in XLE and XLF. Survivorship is "
           f"named on this axis twice: the hindsight list *is* a survivor list, and the 2011 "
           f"control is survivor-conditional throughout (in energy explicitly — two acquired "
           f"constituents have no retrievable tape), a bias that runs in the do-it-yourself "
           f"basket's favour and still fails to produce an edge. The synthetic control recovers "
           f"**{R['syn_pl_gap']:+.2f}%** of a planted **{R['syn_fee']:.2f}%** fee "
           f"(*t* = {R['syn_pl_t']:+.2f}, {R['syn_pl_fire']}/6 seeds) and returns "
           f"**{R['syn_nl_gap']:+.3f}%** on the null ({R['syn_nl_fire']}/6), so the harness is "
           f"sound.\n"
           f"- **Tradability — Mirage.** Fee saved **{R['fee']:.2f}%/yr**; tracking error accepted "
           f"**{R['blend_te']:.2f}%/yr** blended and {R['xlk_te']:.1f}–{R['xle_te']:.1f}%/yr per "
           f"sector — **{R['blend_te']/R['fee']:.0f}×** to **{R['xle_te']/R['fee']:.0f}×** the "
           f"prize. Worst rolling twelve months {R['blend_worst12']:.1f}% blended, "
           f"{R['xle_worst12']:.1f}% in energy; energy drawdown "
           f"{R['xle_dd_diy']:.1f}% vs the fund's {R['xle_dd_fund']:.1f}%. None of it is "
           f"compensated: excess-of-cash Sharpe advantage "
           f"{R['xlk_sh_diy'] - R['xlk_sh_fund']:+.2f} / "
           f"{R['xle_sh_diy'] - R['xle_sh_fund']:+.2f} / "
           f"{R['xlf_sh_diy'] - R['xlf_sh_fund']:+.2f}.\n"
           f"- **What it really is.** A concentrated, fixed-list active sector bet wearing an "
           f"indexing costume. The fee is the smallest term in the problem by two orders of "
           f"magnitude."),
    ]
    nb["cells"] = cells
    return nb


def main():
    for name, nb in [("01_for_the_curious", build_curious()),
                     ("02_for_the_quants", build_quants())]:
        path = os.path.join(HERE, f"{name}.ipynb")
        with open(path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)
        print("wrote", path)


if __name__ == "__main__":
    main()
