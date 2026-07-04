"""Generate the two narrative notebooks for Study 605 (VIX Settlement Day).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached ^VIX/^GSPC OHLC
under ../_cache/ and quote the frozen headline numbers in ``R`` (mirroring docs/results.md).
The synthetic control runs anywhere with no network. Heavy pieces (the 2,000-draw placebo)
are run REDUCED in-notebook and the canonical number is quoted from ``R``.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance ^VIX + ^GSPC OHLC,
# 2004-01-02 -> 2026-06-30; 270 rule-built settlements verified vs 18 known CBOE dates).
R = dict(
    start="2004-01-02", end="2026-06-30", asof="2026-06-30",
    n_sett=270, n_wed=899, n_overlap=40, n_known=18,
    tuesdays=["2008-02-19", "2014-03-18", "2019-03-19", "2022-03-15",
              "2024-06-18", "2025-03-18", "2026-05-19"],
    # level tests: (label, sett%, other%, welch_t_all, welch_t_exfomc or None)
    levels=[("open gap ln(O/C-1), mean", 0.043, 0.016, 0.11, None),
            ("|open gap|", 2.410, 2.185, 1.30, 0.90),
            ("intraday ln(C/O), mean", -0.238, -0.762, 1.06, None),
            ("|intraday|", 5.000, 4.468, 1.48, None),
            ("|close-close|", 5.861, 4.880, 2.42, 1.73),
            ("range (H-L)/C-1", 10.088, 9.078, 2.17, 0.98),
            ("SPX close-close, mean", -0.079, 0.090, -2.06, -1.77)],
    n_sett_exfomc=230, n_wed_exfomc=797,
    # interaction: (base_slope, base_t, sett_slope, d, t_d, n)
    inter_all=(-0.179, -2.30, 0.249, 0.427, 2.56, 1169),
    inter_ex=(-0.195, -2.42, 0.302, 0.497, 2.86, 1027),
    placebo=dict(obs=0.497, mean=-0.023, sd=0.172, p=0.0025, draws=2000, seeds=25),
    # robustness: (label, d, t, n)
    robust=[("drop exact-zero gaps (stale-open artefact)", 0.496, 2.86, 986),
            ("winsorise gap/intr at 1%/99%", 0.447, 2.81, 1027)],
    # fade: interaction ex-FOMC pre/post + |cc| level t's
    fade_inter=dict(pre=(0.468, 2.31, 148), post=(0.490, 1.53, 82)),
    fade_level=dict(all_pre=3.10, all_post=0.18, ex_pre=2.21, ex_post=-0.20),
    split="2018-01-01",
    # tradability (index log units, % per event)
    trade=dict(gross=-0.024, t0=-0.05, t_vs=0.67, n=262, other=-0.361,
               net=-0.524, net_yr=-6.29, cost_bps=25.0),
    # synthetic: (planted, recovered, t)
    syn=[(0.0, 0.025, 0.25), (0.5, 0.525, 5.22)],
    fp_vix="2f08bb9304e8", fp_spx="731fc130d08d",
)

BADGES = (
    "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Faded after 2018?: Mixed](https://img.shields.io/badge/Faded_after_2018%3F-Mixed-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from vix_settlement_day import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    VIX, SPX = data.load_real()
    SETT = data.settlement_calendar()
    DF = st.day_frame(VIX, SPX, settlements=SETT)
else:
    VIX = SPX = SETT = DF = None
print("real cache present:", HAVE_REAL,
      "| settlements:", (0 if SETT is None else len(SETT)),
      "| days:", (0 if DF is None else len(DF)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The one Wednesday a month the VIX won't calm down 🎯\n"
            "### Did the VIX settlement auction — the one a famous paper accused of being pushed — leave tracks in the tape?\n\n"
            + BADGES +
            "Every month, billions of dollars of VIX futures and options stop trading and get paid out "
            "at a **single number**: the *settlement print*, computed from a special auction of S&P-500 "
            "options at the Wednesday open, exactly 30 days before the next monthly S&P option expiry.\n\n"
            "In 2018 two academics, **Griffin & Shams**, published a paper with a needle in the title — "
            "*\"Manipulation in the VIX?\"* — showing that at that precise auction, trading volume spikes "
            "in exactly the deep out-of-the-money options that move the formula, and the settlement print "
            "systematically lands **away** from where the VIX was trading just before and after. Lawsuits "
            "followed. CBOE tweaked the auction.\n\n"
            "We ask a simpler, testable question: **can you see that Wednesday in the ordinary daily VIX "
            "chart?** No tick data, no settlement prints — just the open, high, low and close that anyone "
            "can download.\n\n"
            "> 📓 **Plain-language layer.** Want the regressions, the placebo and the robustness screens? "
            "See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Up front:** the VIX index itself is **not tradable** — this is a forensics study, "
            "not a strategy. Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Can you see settlement Wednesday in the daily VIX bar? | **Yes — but not where you'd look.** "
            "The day isn't reliably *wilder* (that turned out to be mostly Fed days in disguise). The real "
            "track is subtler: it's the one day the **morning move refuses to fade**. |\n"
            "| Is it luck? | Unlikely — placebo calendars produce it ~**0.3%** of the time. |\n"
            "| Can you trade it? | **No.** The index isn't tradable, and the tradable echo is worth "
            "roughly zero before costs, negative after. |\n"
            "| Did it stop after the paper and the lawsuits? | **Not really** — the loud part faded, the "
            "fingerprint didn't. |"
        ),

        md(
            "## The settlement calendar — a rule, not a mystery\n\n"
            "The settlement Wednesday is fixed by contract: **30 days before the third Friday of the "
            "following month** (with holiday adjustments — occasionally it's a Tuesday). We rebuilt the "
            "whole 2004-2026 calendar from that rule and checked it against "
            f"{R['n_known']} official CBOE dates, including all {len(R['tuesdays'])} holiday-shifted "
            "Tuesdays. That gives us **270 settlement days** to compare against **899 ordinary "
            "Wednesdays**."
        ),
        code(
            "assert data.verify_calendar() >= 18\n"
            "if HAVE_REAL:\n"
            "    tue = [str(d.date()) for d in SETT[SETT.weekday == 1]]\n"
            "    print(len(SETT), 'settlements on the calendar |', int(DF['sett'].sum()), 'on the tape')\n"
            "    print('holiday-shifted Tuesdays:', tue)\n"
            "else:\n"
            "    print(R['n_sett'], 'settlements | Tuesdays:', R['tuesdays'])"
        ),

        md(
            "## First look: is settlement day just louder? Mostly a red herring\n\n"
            "The obvious guess — the VIX jumps around more on settlement day — *looks* true at first "
            f"(the average full-day move is {R['levels'][4][1]:.1f}% of the index level vs "
            f"{R['levels'][4][2]:.1f}% on other Wednesdays). But **40 of the 270 settlement days are also "
            "FOMC announcement days** — the Fed's Wednesday press conferences land in exactly the same "
            "mid-month slot. Strip those out and the \"louder day\" almost entirely evaporates. The chart "
            "shows the before/after."
        ),
        code(
            "labels = ['|open gap|', '|close-close|', 'day range', 'SPX return']\n"
            "t_all  = [1.30, 2.42, 2.17, -2.06]\n"
            "t_ex   = [0.90, 1.73, 0.98, -1.77]\n"
            "x = np.arange(len(labels))\n"
            "fig, ax = plt.subplots()\n"
            "ax.bar(x - .18, t_all, .34, color=AMBER, label='all settlement days')\n"
            "ax.bar(x + .18, t_ex, .34, color=GREY, label='FOMC days removed')\n"
            "for yy in (2, -2): ax.axhline(yy, color=RED, lw=1, ls='--')\n"
            "ax.set_xticks(x, labels); ax.set_ylabel('Welch t vs other Wednesdays')\n"
            "ax.set_title('The \"wilder day\" story mostly IS the Fed — level effects vs the t=2 bar')\n"
            "ax.legend(); plt.show()"
        ),
        md(
            "> 🔬 **For the quants.** Every pre-registered level metric (jump size, day volatility, "
            "range, SPX drift) is below *t* = 2 once the FOMC overlap is controlled — the full Welch "
            "table with both columns is in notebook 02."
        ),

        md(
            "## The real track: the gap that refuses to fade\n\n"
            "Here's the VIX's normal morning habit: when it *opens* well above or below yesterday's "
            "close, the day tends to **walk it back** — opening gaps fade into the close. It's one of "
            "the most reliable small regularities in the vol tape.\n\n"
            "**Except on settlement Wednesday.** On the 270 settlement days, the relationship flips: "
            "the morning gap **carries on** into the close instead of fading. That is exactly the shape "
            "you'd expect if the settlement auction *pushes option prices* at the open — the push is in "
            "the actual traded book, so the index doesn't snap back the way a noise gap does.\n\n"
            "The scatter below shows both regimes: each dot is a day, morning gap on the x-axis, rest of "
            "the day on the y-axis."
        ),
        code(
            "if HAVE_REAL:\n"
            "    d = st.event_sample(DF, ex_fomc=True)\n"
            "    s, o = d[d['sett']], d[~d['sett']]\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.scatter(o['gap']*100, o['intr']*100, s=8, alpha=.25, color=GREY,\n"
            "               label='other Wednesdays')\n"
            "    ax.scatter(s['gap']*100, s['intr']*100, s=18, alpha=.6, color=RED,\n"
            "               label='settlement days')\n"
            "    for g, c, lb in ((o, GREY, 'fade'), (s, RED, 'continuation')):\n"
            "        b = np.polyfit(g['gap'], g['intr'], 1)\n"
            "        xs = np.linspace(-8, 8, 20)\n"
            "        ax.plot(xs, (b[0]*xs/100 + b[1])*100, color=c, lw=2.5,\n"
            "                label=f'{lb}: slope {b[0]:+.2f}')\n"
            "    ax.set_xlim(-9, 9); ax.set_ylim(-25, 25)\n"
            "    ax.set_xlabel('morning gap: open vs yesterday close (%)')\n"
            "    ax.set_ylabel('rest of the day: close vs open (%)')\n"
            "    ax.set_title('Normal Wednesdays fade the morning gap — settlement Wednesdays extend it')\n"
            "    ax.legend(); plt.show()\n"
            "else:\n"
            "    print('cache missing — canonical slopes:', R['inter_ex'][0], 'vs', R['inter_ex'][2])"
        ),
        md(
            f"The grey line tilts **down** (slope {R['inter_ex'][0]:+.2f}: a gap up tends to be given "
            f"back); the red line tilts **up** (slope {R['inter_ex'][2]:+.2f}: a gap up keeps going). "
            "The flip is worth a robust *t* of "
            f"**{R['inter_ex'][4]:+.2f}** with the Fed days removed, and fake calendars only produce it "
            f"~{R['placebo']['p']*100:.1f}% of the time. Not proof of manipulation — but exactly the "
            "fingerprint the manipulation story predicts, in data anyone can download."
        ),

        md(
            "## Did the paper and the lawsuits kill it?\n\n"
            "Griffin & Shams published in 2018; class actions were filed the same year and CBOE adjusted "
            "the auction. You'd hope the fingerprint disappeared. The honest answer is **mixed**: the "
            "*louder-day* part did vanish — but the fade-refusal fingerprint is, if anything, the same "
            "size after 2018 as before. There are just fewer settlements since then, so the post-2018 "
            "sample alone can't certify it."
        ),
        code(
            "pre, post = R['fade_inter']['pre'], R['fade_inter']['post']\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.6))\n"
            "bars = ax.bar(['2004-2017\\n(pre-paper)', '2018-2026\\n(post-paper)'],\n"
            "              [pre[0], post[0]], color=[RED, AMBER], width=.5)\n"
            "for b, (d_, t_, n_) in zip(bars, [pre, post]):\n"
            "    ax.text(b.get_x()+b.get_width()/2, d_+.02, f'{d_:+.2f}  (t={t_:+.2f}, {n_} settlements)',\n"
            "            ha='center', fontsize=10)\n"
            "ax.set_ylabel('settlement continuation (interaction d, ex-FOMC)')\n"
            "ax.set_ylim(0, .65)\n"
            "ax.set_title('The fingerprint did not shrink after the 2018 paper — it just got harder to certify')\n"
            "plt.show()"
        ),

        md(
            "## Why you can't cash it\n\n"
            "Three walls, each fatal on its own:\n\n"
            f"1. **The VIX index is not a thing you can buy.** The fingerprint lives in the index print.\n"
            "2. **The obvious rule earns nothing.** \"At the settlement open, ride the gap's direction to "
            f"the close\" is worth **{R['trade']['gross']:+.3f}% per event** before costs (statistically "
            f"zero) and about **{R['trade']['net_yr']:+.1f}%/yr** after futures-level costs.\n"
            "3. **The contract that settles has already settled.** By the time the distorted open exists, "
            "the expiring future's payout is fixed; the surviving contracts track the index loosely.\n\n"
            "A real footprint, an empty wallet: **Mirage**."
        ),

        md(
            "## Verdict\n\n"
            "| Axis | Stamp |\n|---|---|\n"
            "| Signal | **MIXED** — real on the *fade-refusal* fingerprint (robust *t* ≈ 2.9, placebo "
            "p ≈ 0.003); none on the loud \"wilder day\" claims once the Fed overlap is stripped |\n"
            "| Tradability | **MIRAGE** — untradable index, ~zero gross, negative net |\n"
            "| Faded after 2018? | **MIXED** — the level effect died, the fingerprint didn't budge |\n\n"
            "*Full numbers: [docs/results.md](../docs/results.md) · stats teardown: "
            "[02_for_the_quants.ipynb](02_for_the_quants.ipynb)*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata={"language_info": {"name": "python"}})
    nbf.write(nb, os.path.join(HERE, "01_for_the_curious.ipynb"))


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# VIX Settlement Day — the stats teardown 🎯\n\n"
            + BADGES +
            "**Claim under test** (Griffin & Shams 2018, RFS): the monthly VIX-derivative settlement "
            "auction systematically deviates. **Our question:** does the settlement Wednesday leave "
            "tracks in the daily ^VIX bar (2004-2026)?\n\n"
            f"Sample: **{R['n_sett']}** rule-built settlements (verified against {R['n_known']} CBOE "
            f"dates) vs **{R['n_wed']}** other Wednesdays; as-of **{R['asof']}**; fingerprints "
            f"`{R['fp_vix']}` / `{R['fp_spx']}`. Welch *t* for group splits, White (HC1) *t* for "
            "regressions (single-day non-overlapping events). Canonical numbers: "
            "[docs/results.md](../docs/results.md)."
        ),
        code(BOOT_CELL),

        md(
            "## 1 · Calendar construction + verification\n\n"
            "Final settlement = the Wednesday 30 days before the *following* month's SPX expiry (third "
            "Friday, holiday-stepped), itself holiday-stepped — Good Friday via the Gregorian Easter "
            "algorithm, Juneteenth from 2022. Asserted against 18 known CBOE dates including all 7 "
            "holiday-shifted Tuesdays.\n\n"
            "> 💡 **In plain words** — the settlement date is pure arithmetic known years in advance, "
            "so there is no look-ahead anywhere in this study: the *only* same-day input is the open "
            "print itself."
        ),
        code(
            "print('verified against', data.verify_calendar(), 'known CBOE dates')\n"
            "cal = data.settlement_calendar()\n"
            "print(len(cal), 'settlements', cal.min().date(), '->', cal.max().date())\n"
            "print('Tuesdays:', [str(d.date()) for d in cal[cal.weekday == 1]])\n"
            "if HAVE_REAL:\n"
            "    from quantlab.repro import data_stamp\n"
            "    print(data_stamp('^VIX OHLC', VIX, asof=data.AS_OF))\n"
            "    print(data_stamp('^GSPC OHLC', SPX, asof=data.AS_OF))"
        ),

        md(
            "## 2 · Level tests — and the FOMC confounder\n\n"
            "The planned Welch splits (settlement days vs other Wednesdays), run twice: on all days, "
            "and with **FOMC statement days removed** — 40 of the 270 settlements coincide with FOMC "
            "Wednesdays (source: the Fed calendar table shared with studies 67/135/517).\n\n"
            "> 💡 **In plain words** — mid-month Wednesday is *also* where the Fed lives. Any settlement "
            "study that skips this control is measuring monetary policy."
        ),
        code(
            "if HAVE_REAL:\n"
            "    for tag, ex in (('ALL', False), ('ex-FOMC', True)):\n"
            "        lv = st.level_tests(DF, ex_fomc=ex)\n"
            "        print(f\"--- {tag}: n={lv['n_sett']} vs {lv['n_other']}\")\n"
            "        for k in ('gap','abs_gap','intr','abs_intr','cc','abs_cc','rng','spx'):\n"
            "            v = lv[k]\n"
            "            print(f\"  {k:8s} sett {v['sett']*100:+7.3f}%  other {v['other']*100:+7.3f}%  \"\n"
            "                  f\"Welch t={v['t']:+.2f}\")\n"
            "else:\n"
            "    for row in R['levels']: print(row)"
        ),
        md(
            f"**Read-out.** All-days, two level stats clear the bar — |close-close| (*t* = +2.42) and "
            f"range (*t* = +2.17) — and SPX prints red (*t* = −2.06). Ex-FOMC (n = {R['n_sett_exfomc']} "
            f"vs {R['n_wed_exfomc']}) **all of them fall under 2**: |gap| +0.90, |close-close| +1.73, "
            "range +0.98, SPX −1.77. The level story does not survive the confounder. *(No level claim "
            "is stamped Real.)*"
        ),

        md(
            "## 3 · The signature — settlement × gap continuation\n\n"
            "`intr = a + b·gap + c·sett + d·(gap × sett)` on settlement days + other Wednesdays, "
            "White (HC1) robust *t*. `b` is the normal-Wednesday gap fade; `d` is the settlement "
            "distortion of that fade.\n\n"
            "> 💡 **In plain words** — we're not asking whether settlement days move more; we're asking "
            "whether the *morning move behaves differently*: normally the VIX takes back its opening "
            "gap during the day; on settlement day it keeps it. That's what a pushed opening auction "
            "should look like from daily bars."
        ),
        code(
            "if HAVE_REAL:\n"
            "    for tag, ex in (('ALL days', False), ('ex-FOMC ', True)):\n"
            "        r = st.interaction(DF, ex_fomc=ex)\n"
            "        print(f\"{tag}: b={r['base_slope']:+.3f} (t={r['base_t']:+.2f})  \"\n"
            "              f\"b+d={r['sett_slope']:+.3f}  d={r['inter']:+.3f} (t={r['inter_t']:+.2f})  \"\n"
            "              f\"n={r['n']}\")\n"
            "else:\n"
            "    print('all  :', R['inter_all']); print('exfomc:', R['inter_ex'])"
        ),
        code(
            "if HAVE_REAL:\n"
            "    d = st.event_sample(DF, ex_fomc=True)\n"
            "    s, o = d[d['sett']], d[~d['sett']]\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.scatter(o['gap']*100, o['intr']*100, s=8, alpha=.25, color=GREY, label='other Wed')\n"
            "    ax.scatter(s['gap']*100, s['intr']*100, s=18, alpha=.6, color=RED, label='settlement')\n"
            "    for g, c in ((o, GREY), (s, RED)):\n"
            "        b = np.polyfit(g['gap'], g['intr'], 1)\n"
            "        xs = np.linspace(-8, 8, 20)\n"
            "        ax.plot(xs, (b[0]*xs/100 + b[1])*100, color=c, lw=2.5, label=f'slope {b[0]:+.2f}')\n"
            "    ax.set_xlim(-9, 9); ax.set_ylim(-25, 25)\n"
            "    ax.set_xlabel('overnight gap ln(O/C$_{-1}$) (%)'); ax.set_ylabel('intraday ln(C/O) (%)')\n"
            "    ax.set_title('Gap fade (grey) flips to gap continuation (red) on settlement days — ex-FOMC')\n"
            "    ax.legend(); plt.show()"
        ),

        md(
            "## 4 · Random-calendar placebo (≥ 20 seeds)\n\n"
            "One fake settlement Wednesday per month drawn among non-settlement Wednesdays (ex-FOMC), "
            "same interaction regression. Canonical run: **2,000 draws over 25 seeds** → two-sided "
            f"**p = {R['placebo']['p']:.4f}** (observed {R['placebo']['obs']:+.3f} vs placebo "
            f"{R['placebo']['mean']:+.3f} ± {R['placebo']['sd']:.3f}). The cell below re-runs a "
            "REDUCED version (200 draws / 5 seeds) to keep the notebook light."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_interaction(DF, ex_fomc=True, n_seeds=5, n_per=40)\n"
            "    print(f\"reduced placebo: obs {pl['obs']:+.3f} vs {pl['mean']:+.3f} (sd {pl['sd']:.3f}) \"\n"
            "          f\"over {pl['n_draws']} draws -> p={pl['p_two_sided']:.4f}\")\n"
            "print('canonical (2,000 draws / 25 seeds): p =', R['placebo']['p'])"
        ),

        md(
            "## 5 · Robustness — data quirks and tails\n\n"
            "Yahoo's ^VIX open exactly equals the prior close on ~4% of days (~30% in 2008-09) — a "
            "stale-open artefact that only *dilutes* the gap variable. And fat-tailed days could own "
            "an OLS slope. Neither carries the result:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    r1 = st.interaction(DF, ex_fomc=True, drop_zero_gaps=True)\n"
            "    r2 = st.interaction(DF, ex_fomc=True, winsor=0.01)\n"
            "    print(f\"drop zero gaps : d={r1['inter']:+.3f} (t={r1['inter_t']:+.2f}, n={r1['n']})\")\n"
            "    print(f\"winsor 1%/99%  : d={r2['inter']:+.3f} (t={r2['inter_t']:+.2f}, n={r2['n']})\")\n"
            "else:\n"
            "    for row in R['robust']: print(row)"
        ),

        md(
            "## 6 · Third axis — the 2018 fade test\n\n"
            f"Split at **{R['split']}** (RFS publication + the consolidated antitrust litigation + "
            "CBOE auction tweaks — a justified, not snooped, split).\n\n"
            "> 💡 **In plain words** — if the print was being pushed and the pushing stopped when the "
            "lawyers arrived, the fingerprint should disappear after 2018. The *noise* part did; the "
            "*continuation* part didn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sp = st.subperiod_interactions(DF, ex_fomc=True)\n"
            "    for lbl, r in (('2004-2017', sp['pre']), ('2018-2026', sp['post'])):\n"
            "        print(f\"{lbl}: d={r['inter']:+.3f} (t={r['inter_t']:+.2f}, n_sett={r['n_sett']})\")\n"
            "    sl = st.subperiod_levels(DF)\n"
            "    for tag in ('all', 'ex_fomc'):\n"
            "        print(f\"|cc| level t ({tag}): {sl[tag]['pre']['t']:+.2f} -> {sl[tag]['post']['t']:+.2f}\")\n"
            "else:\n"
            "    print(R['fade_inter']); print(R['fade_level'])"
        ),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))\n"
            "pre, post = R['fade_inter']['pre'], R['fade_inter']['post']\n"
            "axes[0].bar(['2004-2017', '2018-2026'], [pre[0], post[0]], color=[RED, AMBER], width=.5)\n"
            "axes[0].set_title('continuation d (ex-FOMC): did not fade')\n"
            "axes[0].set_ylabel('interaction d')\n"
            "fl = R['fade_level']\n"
            "axes[1].bar(['pre (FOMC in)', 'post (FOMC in)', 'pre (ex-FOMC)', 'post (ex-FOMC)'],\n"
            "            [fl['all_pre'], fl['all_post'], fl['ex_pre'], fl['ex_post']],\n"
            "            color=[GREY, GREY, AMBER, AMBER], width=.6)\n"
            "axes[1].axhline(2, color=RED, lw=1, ls='--')\n"
            "axes[1].set_title('|close-close| level effect (Welch t): faded')\n"
            "axes[1].tick_params(axis='x', labelsize=8)\n"
            "plt.tight_layout(); plt.show()"
        ),

        md(
            "## 7 · Tradability — the honest cash-out attempt\n\n"
            "Rule: at the settlement-day open (calendar known years ahead; the open print is the only "
            "same-day input — **one clean lag**), position = sign(gap), exit at the close. Index log "
            "units — **the index is not tradable**; costs at VIX-futures retail level "
            f"({R['trade']['cost_bps']:.0f} bps one-way × 2), and even this flatters the trade (the "
            "futures under-react to spot; the expiring contract has already printed its SOQ)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tp = st.continuation_pnl(DF, ex_fomc=False, cost_bps_oneway=25.0)\n"
            "    print(f\"gross {tp['gross_per_event']*100:+.3f}%/event (t vs 0 = {tp['t_vs_zero']:+.2f}, \"\n"
            "          f\"t vs other Wed = {tp['t_vs_otherwed']:+.2f}, n={tp['n']})\")\n"
            "    print(f\"other Wednesdays {tp['other_wed']*100:+.3f}%/event\")\n"
            "    print(f\"net {tp['net_per_event']*100:+.3f}%/event x 12/yr = {tp['net_per_year']*100:+.2f}%/yr\")\n"
            "else:\n"
            "    print(R['trade'])"
        ),
        md(
            "> 💡 **In plain words** — the fingerprint is a *covariance* (big gaps carry through in "
            "proportion), not a directional edge you can harvest with a sign rule 12 times a year. "
            "Gross ≈ 0, net deeply negative. **Mirage.**"
        ),

        md(
            "## 8 · Synthetic control — machinery proof (never market evidence)\n\n"
            "A deterministic 44-year world where normal-day gaps fade (slope −0.20) and settlement days "
            "carry a **planted** extra gap→close slope. The pipeline must stay quiet at 0 and light up "
            "at +0.50:"
        ),
        code(
            "for planted in (0.0, 0.5):\n"
            "    world = data.synthetic_world(planted=planted, seed=605)\n"
            "    sdf = st.day_frame(world, spx=None)\n"
            "    r = st.interaction(sdf, ex_fomc=False)\n"
            "    print(f\"planted {planted:+.2f}: recovered d={r['inter']:+.3f} (t={r['inter_t']:+.2f}, \"\n"
            "          f\"n_sett={r['n_sett']})\")"
        ),

        md(
            "## Verdict\n\n"
            f"- **Signal — MIXED.** Real on the structure: settlement × gap continuation d = "
            f"{R['inter_all'][3]:+.3f} (White t = {R['inter_all'][4]:+.2f}) all days, "
            f"{R['inter_ex'][3]:+.3f} (t = {R['inter_ex'][4]:+.2f}) ex-FOMC, placebo p = "
            f"{R['placebo']['p']:.4f}, robust to the stale-open screen and winsorising. None on the "
            "level: every planned Welch split dies ex-FOMC (all |t| < 2).\n"
            f"- **Tradability — MIRAGE.** Untradable index; sign rule {R['trade']['gross']:+.3f}%/event "
            f"gross (t ≈ 0), {R['trade']['net_yr']:+.1f}%/yr net.\n"
            f"- **Faded after 2018? — MIXED.** Level effect gone (+3.10 → +0.18); continuation "
            f"unchanged in point estimate ({R['fade_inter']['pre'][0]:+.3f} → "
            f"{R['fade_inter']['post'][0]:+.3f}) but under-powered post-2018 (t = "
            f"{R['fade_inter']['post'][1]:+.2f}, n = {R['fade_inter']['post'][2]}).\n\n"
            "*Canonical numbers: [docs/results.md](../docs/results.md) · plain-words tour: "
            "[01_for_the_curious.ipynb](01_for_the_curious.ipynb)*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata={"language_info": {"name": "python"}})
    nbf.write(nb, os.path.join(HERE, "02_for_the_quants.ipynb"))


if __name__ == "__main__":
    build_curious()
    build_quants()
    print("wrote 01_for_the_curious.ipynb + 02_for_the_quants.ipynb")
