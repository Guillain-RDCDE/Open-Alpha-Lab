"""Generate the two narrative notebooks for Study 644 (CPI-Day-Drift).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached SPY/TLT tapes
under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance SPY 1997-01-02 ->
# 2026-06-30, TLT 2002-07-30 -> 2026-06-30; 353 hardcoded actual CPI release dates).
R = dict(
    start="1997-01-02", end="2026-06-30", n_cpi=353, n_spy_rest=7065, n_tlt_rest=5731,
    n_tlt_events=286,
    cal_lo="1997-01-14", cal_hi="2026-06-10",
    # headline return
    spy_cpi_bps=+3.73, spy_rest_bps=+3.80, spy_gap_bps=-0.07,
    spy_welch=-0.01, spy_nw=-0.01, spy_hit=198, spy_hit_pct=56.1, spy_wilson=(50.9, 61.2),
    tlt_cpi_bps=+5.24, tlt_rest_bps=+1.26, tlt_gap_bps=+3.98,
    tlt_welch=+0.67, tlt_nw=+0.67, tlt_hit=156, tlt_hit_pct=54.5, tlt_wilson=(48.8, 60.2),
    # placebos
    plac_spy_ret_obs=+3.73, plac_spy_ret_mean=+3.87, plac_spy_ret_sd=6.34, plac_spy_ret_p=0.62605,
    plac_tlt_rng_obs=1.001, plac_tlt_rng_mean=0.884, plac_tlt_rng_sd=0.031, plac_tlt_rng_p=0.00025,
    plac_spy_rng_obs=1.392, plac_spy_rng_mean=1.346, plac_spy_rng_sd=0.052, plac_spy_rng_p=0.18350,
    # event window: offset -> (mean bps, welch t)
    event={-3: (+4.94, +0.27), -2: (+3.66, +0.11), -1: (+8.01, +0.65),
           0: (+3.73, +0.10), 1: (-2.63, -0.74), 2: (+16.35, +2.04), 3: (+3.51, +0.07)},
    runup_bps=+16.62, runup_t=+1.67,
    # realized range
    spy_rng_cpi=1.392, spy_rng_rest=1.346, spy_rng_t=+0.82,
    tlt_rng_cpi=1.001, tlt_rng_rest=0.884, tlt_rng_t=+2.70,
    # regime contrast (split 2022-01-01)
    split="2022-01-01",
    era_ret_pre=+2.74, era_ret_pre_n=300, era_ret_pre_t=-0.13,
    era_ret_post=+9.35, era_ret_post_n=53, era_ret_post_t=+0.24, era_ret_diff_t=+0.30,
    era_rng_pre=1.391, era_rng_pre_t=+0.36, era_rng_post=1.398, era_rng_post_t=+1.33,
    era_rng_diff_t=+0.05,
    # third axis: biggest day of the month
    bd_null=4.8,
    bd_spy_ret_pre=4.3, bd_spy_ret_pre_ci=(2.5, 7.3), bd_spy_ret_pre_n=300,
    bd_spy_ret_post=9.4, bd_spy_ret_post_ci=(4.1, 20.3), bd_spy_ret_post_n=53,
    bd_spy_ret_difft=+1.21,
    bd_spy_rng_pre=4.7, bd_spy_rng_post=3.8, bd_spy_rng_difft=-0.31,
    bd_tlt_ret_pre=4.7, bd_tlt_ret_post=7.5, bd_tlt_ret_difft=+0.72,
    bd_tlt_rng_pre=8.2, bd_tlt_rng_pre_ci=(5.3, 12.4), bd_tlt_rng_pre_n=233,
    bd_tlt_rng_post=15.1, bd_tlt_rng_post_ci=(7.9, 27.1), bd_tlt_rng_post_n=53,
    bd_tlt_rng_difft=+1.31,
    # tradability — naive SPY timer
    tm_gross=+3.73, tm_net5=-6.27, tm_net10=-16.27, tm_ann5=-0.75, tm_ann10=-1.95,
    tm_rest=+3.80, tm_t=-0.01, tm_hit=56.1, tm_worst=-6.6,
    # synthetic control (two independent knobs)
    syn_null_ret_mean=+0.24, syn_null_ret_sd=1.26, syn_null_ret_fire=2,
    syn_null_rng_mean=-0.11, syn_null_rng_sd=0.91, syn_null_rng_fire=0,
    syn_planted_ret_t=+3.46, syn_planted_rng_t=+13.07, syn_cross_ret_t=+0.87,
    fp_spy="3ebc1d6324c3", fp_tlt="745c19792ff8",
)

BADGES = (
    "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Biggest_day%3F: Busted](https://img.shields.io/badge/Biggest_day%3F-Busted-8b949e?style=flat-square)\n\n"
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

from cpi_day_drift import data, strategy as st

CPI = data.cpi_calendar()
HAVE_REAL = data.have_real()
if HAVE_REAL:
    SPY_PX, TLT_PX = data.load_real()
    SPY_SESSIONS, _ = data.map_to_sessions(SPY_PX.index, CPI)
    TLT_SESSIONS, _ = data.map_to_sessions(TLT_PX.index, CPI)
    SPY = st.day_frame(SPY_PX, SPY_SESSIONS)
    TLT = st.day_frame(TLT_PX, TLT_SESSIONS)
else:
    SPY_PX = TLT_PX = SPY = TLT = None
    SPY_SESSIONS = TLT_SESSIONS = None
print("real cache present:", HAVE_REAL, "| CPI releases:", len(CPI),
      "| SPY tape days:", (0 if SPY is None else len(SPY)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Is CPI morning really the market's biggest day? 📊🌡️\n"
            "### The Fed's favorite number, put through the desk's honesty rails\n\n"
            + BADGES +
            "Twelve times a year, before the opening bell, the Bureau of Labor Statistics drops "
            "the inflation report. Since the Fed started hiking rates in 2022, trading desks call "
            "it the biggest print of the month — bigger than earnings, bigger than jobs day, "
            "bigger than anything the Fed itself says. *This* number decides whether the Fed hikes, "
            "holds, or cuts.\n\n"
            "So does the market actually **move** on CPI morning — and does it move in a direction "
            "you could bank? We tested it on 353 real CPI releases since 1997, on both stocks "
            "(SPY) and the long bond (TLT).\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 353 actual CPI release dates (1997→2026) hardcoded from the "
            "BLS's own archive — not a weekday-pattern guess. Every chart is drawn by the code "
            "beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the stock market move a particular way on CPI day? | **No.** SPY's CPI-day "
            f"return ({R['spy_cpi_bps']:+.2f} bps) is statistically identical to an ordinary day "
            f"({R['spy_rest_bps']:+.2f} bps) — a coin flip, not a trend. |\n"
            f"| Does the bond market? | **Also no**, on direction — TLT's average CPI-day return "
            "isn't certifiable either. |\n"
            f"| Does *something* systematic happen? | **Yes — bonds get louder.** TLT's actual "
            f"daily *swing* (high minus low) is bigger on CPI mornings than on ordinary days, and "
            "this one clears the statistical bar convincingly. Stocks don't show the same "
            "loudness. |\n"
            f"| Is CPI really \"the biggest day of the month\" now? | **Not proven.** The rate at "
            f"which CPI day is literally the single loudest day of its month did rise after the "
            f"Fed's 2022 hiking cycle began — but the increase itself can't be certified, and "
            "even at its best it's a small minority of months, not \"always.\" |\n\n"
            "> The print makes bonds jumpier. It does not hand you a trade."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"CPI morning is the single most important economic release of the month. Every "
            "trading desk clears its calendar for 8:30 am. The Fed's entire rate path hangs on "
            "this one number — of course the market moves on it, and of course it's become the "
            "biggest trading day of the month since the Fed started hiking.\"*\n\n"
            "It's a reasonable-sounding claim with a clean mechanism: inflation surprises change "
            "the Fed's next move, the Fed's next move changes every discount rate in the market, "
            "so a CPI surprise should ripple through stocks and (especially) bonds the moment it "
            "hits the tape."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If true in a *bankable* sense, this is a free, publicly-known calendar edge: buy "
            "before the print if you expect a cool number, sell if you expect hot, or simply "
            "harvest the extra realized move with an options strategy. Even if the direction isn't "
            "predictable in advance, if CPI mornings are reliably *louder*, that's valuable "
            "information for anyone sizing positions or selling options around the print.\n\n"
            "So we ask three separate things: does the market actually move (direction), does it "
            "move *more* than usual (loudness), and did that second thing get more true once the "
            "Fed started hanging rate decisions on this exact number?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The calendar.** All **{R['n_cpi']}** actual CPI release dates from "
            f"{R['cal_lo']} to {R['cal_hi']}, hardcoded from the BLS's own news-release archive. "
            "CPI lands at 8:30 am ET — before the 9:30 am open — so the day's closing price "
            "already contains the market's full reaction.\n"
            "- **The comparison.** SPY's and TLT's daily change and daily high-low swing on CPI "
            "days vs every other trading day.\n"
            "- **The luck check.** Draw random days of the same count instead — how often does a "
            "random calendar look this special?\n"
            f"- **The regime check.** Split the tape at {R['split']} — the Fed's hawkish pivot — "
            "and ask whether the *difference* between the two eras is real, not just eyeballed."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the direction question.** Average daily return on CPI days vs every other "
            "day, for both stocks and bonds."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s_spy = st.cpi_day_stats(SPY); s_tlt = st.cpi_day_stats(TLT)\n"
            "    vals = [s_spy['cpi_bps'], s_spy['rest_bps'], s_tlt['cpi_bps'], s_tlt['rest_bps']]\n"
            "else:\n"
            "    vals = [R['spy_cpi_bps'], R['spy_rest_bps'], R['tlt_cpi_bps'], R['tlt_rest_bps']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.6))\n"
            "labels = ['SPY\\nCPI day','SPY\\nother day','TLT\\nCPI day','TLT\\nother day']\n"
            "ax.bar(labels, vals, color=[GREY, GREY, GREY, GREY], width=.6)\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.2f}',(i,v),ha='center',\n"
            "    va='top' if v<0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('average daily return (bps)')\n"
            "ax.set_title('Neither asset picks a direction on CPI morning')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'SPY: CPI {vals[0]:+.2f} vs other {vals[1]:+.2f} bps')\n"
            "print(f'TLT: CPI {vals[2]:+.2f} vs other {vals[3]:+.2f} bps')"
        ),
        md(
            f"CPI-day and other-day returns are practically the same bar for both assets. SPY's "
            f"gap is Welch *t* = **{R['spy_welch']:.2f}** — statistically nothing — and TLT's, "
            f"despite looking a bit bigger on the page, is *t* = **{R['tlt_welch']:.2f}**, also "
            "well under the bar. A random-calendar check confirms it: a random 353-day calendar "
            f"produces a mean this size or bigger **{R['plac_spy_ret_p']*100:.0f}% of the time** — "
            "there's no direction here to trade.\n\n"
            "**Now the loudness question** — this is where it gets interesting:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rg_spy = st.range_stats(SPY); rg_tlt = st.range_stats(TLT)\n"
            "    vals = [rg_spy['cpi_range_pct'], rg_spy['rest_range_pct'],\n"
            "            rg_tlt['cpi_range_pct'], rg_tlt['rest_range_pct']]\n"
            "else:\n"
            "    vals = [R['spy_rng_cpi'], R['spy_rng_rest'], R['tlt_rng_cpi'], R['tlt_rng_rest']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.6))\n"
            "labels = ['SPY\\nCPI day','SPY\\nother day','TLT\\nCPI day','TLT\\nother day']\n"
            "cols = [GREY, GREY, RED, GREY]\n"
            "ax.bar(labels, vals, color=cols, width=.6)\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:.2f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('average high-low range (% of prev close)')\n"
            "ax.set_title('Bonds get louder on CPI day. Stocks barely notice.')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'SPY range: CPI {vals[0]:.3f}% vs other {vals[1]:.3f}%')\n"
            "print(f'TLT range: CPI {vals[2]:.3f}% vs other {vals[3]:.3f}%')"
        ),
        md(
            f"There it is: **TLT's** daily high-low swing jumps from **{R['tlt_rng_rest']:.2f}%** "
            f"on an ordinary day to **{R['tlt_rng_cpi']:.2f}%** on a CPI day — a real, "
            f"statistically certified effect (Welch *t* = **{R['tlt_rng_t']:.2f}**, and a "
            "random-calendar check finds a random day producing this only "
            f"**{R['plac_tlt_rng_p']*100:.2f}%** of the time). **SPY's** range barely moves "
            f"({R['spy_rng_rest']:.2f}% → {R['spy_rng_cpi']:.2f}%, *t* = **{R['spy_rng_t']:.2f}**, "
            "not certifiable). Makes sense: CPI is an inflation number, and inflation is exactly "
            "what re-prices the bond market's discount rate. Stocks care too, eventually — just "
            "not enough, on the average day, to show up in the statistics.\n\n"
            "**And here's the myth we actually set out to test** — \"CPI day is the biggest "
            "trading day of the month\":"
        ),
        code(
            "if HAVE_REAL:\n"
            "    bd = st.biggest_day_of_month(TLT, TLT_SESSIONS, data.REGIME_SPLIT, metric='range_pct')\n"
            "    pre, post = bd['pre_rate']*100, bd['post_rate']*100\n"
            "else:\n"
            "    pre, post = R['bd_tlt_rng_pre'], R['bd_tlt_rng_post']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['before the 2022\\nhiking cycle','since the 2022\\nhiking cycle'], [pre, post],\n"
            "       color=[GREY, AMBER], width=.55)\n"
            "ax.axhline(R['bd_null'], ls='--', c=GREY, lw=1.3, label=f\"pure chance ({R['bd_null']:.1f}%)\")\n"
            "for i,v in enumerate([pre, post]): ax.annotate(f'{v:.1f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('% of months TLT is loudest on its CPI day')\n"
            "ax.set_title('Elevated since 2022 - but not statistically certified')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'TLT range \"biggest day of month\" rate: pre {pre:.1f}% -> post {post:.1f}%')"
        ),
        md(
            f"Even on TLT's realized range — our *strongest* leg of evidence — the \"biggest day "
            f"of the month\" rate only rises from **{R['bd_tlt_rng_pre']:.1f}%** of months to "
            f"**{R['bd_tlt_rng_post']:.1f}%** since the Fed started hiking. That's a real-looking "
            "jump on the page, but the sample of post-2022 months is small, the statistical test "
            "of the *difference* falls short of our bar, and even 15% of months is nowhere close "
            "to \"the biggest day, every month.\" The catchphrase overstates what the data show.\n\n"
            "**Finally — could you actually trade any of this?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    tc5 = st.timer_capture(SPY, cost_bps=5.0)\n"
            "    tc10 = st.timer_capture(SPY, cost_bps=10.0)\n"
            "    g, n5, n10 = tc5['gross_bps'], tc5['net_bps'], tc10['net_bps']\n"
            "else:\n"
            "    g, n5, n10 = R['tm_gross'], R['tm_net5'], R['tm_net10']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['gross','net (5 bps)','net (10 bps)'], [g, n5, n10],\n"
            "       color=[GREY, RED, RED], width=.55)\n"
            "for i,v in enumerate([g, n5, n10]): ax.annotate(f'{v:+.1f}',(i,v),ha='center',\n"
            "    va='top' if v<0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('bps per CPI-day \"own SPY only\" bet')\n"
            "ax.set_title('There was never a gross edge to begin with')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.2f} -> net {n5:+.2f} / {n10:+.2f} bps')"
        ),
        md(
            f"Not really. The \"own SPY only on CPI day\" timer starts from a gross edge of "
            f"**{R['tm_gross']:+.2f} bps** — practically nothing above the market's own average "
            f"day — so a normal 5-basis-point round-trip cost flips it to **{R['tm_net5']:+.2f} "
            "bps net**, a loser. There's no free lunch here: the real effect we found lives in "
            "bond *volatility*, not in a direction you can just buy and hold."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Mixed.** Real for one thing (TLT gets louder on CPI mornings), none for "
            "everything else (no direction in stocks or bonds, no loudness bump in stocks).\n"
            "- **Tradability — Mirage.** No direction to bank, and the one real effect isn't "
            "something a simple buy-and-hold position can monetize.\n"
            "- **\"Biggest day of the month\"? — Busted.** Directionally plausible since the Fed's "
            "2022 pivot, but the data don't certify it, and even the best case is a small "
            "minority of months, not a rule."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The next question is options, not stock.** If bonds really do get louder on CPI "
            "day, the honest way to test whether that's *bankable* is a straddle or variance-swap "
            "study on TLT/rates around the print — not a directional equity bet.\n"
            "- **The Fed's own vol crush is a cautionary tale.** Our sibling study "
            "[637-fomc-vol-crush](../../637-fomc-vol-crush/) found a real, certified implied-vol "
            "collapse on FOMC decision afternoons — and it turned out to be untradable, because "
            "futures pre-price the scheduled event. The same skepticism should apply here before "
            "anyone gets excited about CPI-day loudness.\n"
            "- **Sibling studies:** [602-macro-announcement-premium](../../602-macro-announcement-premium/) "
            "(the pooled FOMC+CPI+NFP equity premium) and "
            "[643-payrolls-day-effect](../../643-payrolls-day-effect/) (the same protocol on "
            "Nonfarm Payrolls mornings) — ask related but distinct questions.\n\n"
            "*Think CPI-day loudness is bankable through options? Show a net, certifiable edge "
            "after the market-maker's spread on release mornings — then we'll talk.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# CPI-Day-Drift — a quantitative teardown 🔬\n"
            "### SPY/TLT Welch+HAC splits on return AND range · a two-sided/one-sided "
            "random-calendar placebo · the [−3..+3] event anatomy · a justified regime split · "
            "the \"biggest day of the month\" hit-rate test · an honest naive-timer cost sweep · "
            "a two-knob synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The claim — **CPI mornings move stocks and bonds systematically** — splits cleanly "
            "into a *direction* question and a *loudness* question, and this notebook measures "
            "both, on both assets, honestly.\n\n"
            "> ⚠️ **Data note.** SPY daily raw OHLC + adjusted close (1997→2026) and TLT daily "
            "raw OHLC + adjusted close (2002→2026, its inception), yfinance, cached; **353 "
            "hardcoded actual CPI release dates** from the BLS archive (identical table to "
            "sibling study 602). No survivorship (SPY/TLT are index-tracking ETFs). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprints `" + R["fp_spy"] + "` / `" +
            R["fp_tlt"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | TLT range Welch **t = {R['tlt_rng_t']:.2f}** (placebo "
            f"*p* = {R['plac_tlt_rng_p']:.5f}) — real; SPY return **t = {R['spy_welch']:.2f}**, "
            f"TLT return **t = {R['tlt_welch']:.2f}**, SPY range **t = {R['spy_rng_t']:.2f}** — "
            "all none |\n"
            f"| **Tradability** | `MIRAGE` | naive SPY timer: gross {R['tm_gross']:+.2f} bps -> "
            f"net **{R['tm_net5']:+.2f} bps** at 5 bps costs (**{R['tm_ann5']:+.2f}%/yr**) |\n"
            f"| **\"Biggest day of month\"?** | `BUSTED` | best case (TLT range) diff *t* = "
            f"**{R['bd_tlt_rng_difft']:.2f}**, rate {R['bd_tlt_rng_pre']:.1f}% -> "
            f"{R['bd_tlt_rng_post']:.1f}% — elevated, uncertified |\n\n"
            "> 💡 In plain words: the print re-prices the bond market's volatility for real; it "
            "does not hand equities, or anyone, a direction to trade."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t$ be the daily close-to-close log return of SPY or TLT and $D_t \\in \\{0,1\\}$ "
            "the scheduled-CPI-day flag (known *ex ante* — the BLS calendar is public months "
            "ahead). CPI prints at 08:30 ET, before the 09:30 open, so the daily bar fully "
            "contains the reaction. The claims:\n\n"
            "- **H₁ (direction).** $E[r_t \\mid D_t=1] \\ne E[r_t \\mid D_t=0]$ — CPI days move a "
            "particular way, systematically, across 353 releases.\n"
            "- **H₂ (loudness).** Realized high-low range is elevated on CPI days vs other days — "
            "the market reacts *harder*, whichever way it goes.\n"
            "- **H₃ (regime).** The loudness/direction effect got structurally stronger once the "
            "Fed's rate path became explicitly data-dependent on CPI (2022+).\n"
            "- **H₄ (\"biggest day\").** CPI day is disproportionately likely to be the single "
            "largest-move trading day of its calendar month, especially since 2022.\n\n"
            "We find **H₁ not supported** on either asset, **H₂ supported for TLT only** "
            f"(*t* = {R['tlt_rng_t']:.2f}), **H₃ not certified** on any leg tested, and "
            "**H₄ directionally suggestive but not certified** on any of the four metrics tested."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "CPI releases are **single, non-overlapping events**, so the planned primary for the "
            "headline split is a **Welch t** on the group difference — direction gets a "
            "**two-sided** placebo (no a-priori sign), loudness gets a **one-sided** placebo "
            "(\"louder\" is an inherently non-negative claim). Because daily returns carry mild "
            "serial correlation, we cross-check the return split with a **Newey-West (5-lag) t** "
            "on the dummy regression $r_t = a + b D_t$ — the slope *is* the mean gap. Hit rates "
            "carry **Wilson intervals**; the regime split (2022-01-01, the Fed's Dec-2021 "
            "hawkish pivot) is justified *ex ante* from the FOMC calendar, not snooped, and "
            "tested as a **difference**, not eyeballed; the \"biggest day of the month\" claim is "
            "operationalized as a rate against an honest chance baseline (1/n for an n-session "
            "month)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Calendar.** {R['n_cpi']} actual CPI releases {R['cal_lo']} → {R['cal_hi']}, "
            "hardcoded (BLS archive; identical table to sibling study 602). All map onto the SPY "
            f"tape; {R['n_tlt_events']} onto the shorter TLT tape (2002+ inception).\n"
            f"- **Tape.** SPY + TLT raw OHLC + adjusted close, {R['start']} → {R['end']} "
            "(TLT from inception). As-of 2026-06-30 (last complete month).\n"
            "- **Headline.** Welch t (return, two-sided placebo; range, one-sided placebo) + "
            "NW(5) t on return + Wilson hit rates.\n"
            "- **Anatomy.** Event window [−3..+3], per-offset Welch t vs far days; cumulative "
            "run-up per event, one-sample t.\n"
            "- **Regime.** Pre/post 2022-01-01 within-era Welch t's + Welch t of the difference, "
            "on both return and range.\n"
            "- **Myth-check.** Per-month rank test: is the CPI session the largest-move day of "
            "its month, on |return| and on range, for both assets; Wilson intervals + Welch t of "
            "the pre/post difference.\n"
            "- **Execution (tradability).** Enter SPY at the prior close (calendar public months "
            "ahead — zero look-ahead), exit the CPI close; 2 × one-way cost × NAV per event; "
            "long-only, no borrow.\n"
            "- **Control.** Synthetic daily-return process, TWO independent planted knobs "
            "(return-shift, vol-multiplier); the null must not fire on either axis across "
            "20 seeds, and each knob must be detected without contaminating the other."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline split — direction, both assets, plus its placebo\n\n"
            "Welch t + NW(5) t on close-to-close return, and a two-sided random-calendar null. "
            "In the notebook we run a lighter placebo (4 seeds × 500 draws) and quote the "
            "canonical 20,000-draw p from `results.md`."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s_spy = st.cpi_day_stats(SPY); s_tlt = st.cpi_day_stats(TLT)\n"
            "    print(f\"SPY: CPI {s_spy['cpi_bps']:+.2f} bps vs other {s_spy['rest_bps']:+.2f} bps  \"\n"
            "          f\"Welch t={s_spy['welch_t']:+.2f}  NW(5) t={s_spy['nw_t']:+.2f}\")\n"
            "    print(f\"TLT: CPI {s_tlt['cpi_bps']:+.2f} bps vs other {s_tlt['rest_bps']:+.2f} bps  \"\n"
            "          f\"Welch t={s_tlt['welch_t']:+.2f}  NW(5) t={s_tlt['nw_t']:+.2f}\")\n"
            "    pl = st.placebo_pvalue(SPY, col='ret', two_sided=True, n_draws_per_seed=500, n_seeds=4)\n"
            "    obs, draws = pl['obs'], pl['draws']\n"
            "else:\n"
            "    obs = R['spy_cpi_bps'] / 1e4\n"
            "    rng = np.random.default_rng(644)\n"
            "    draws = rng.normal(R['plac_spy_ret_mean']/1e4, R['plac_spy_ret_sd']/1e4, 2000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws*1e4, bins=50, color=GREY, alpha=.85,\n"
            "        label='null: random calendars of 353 days (light in-notebook run)')\n"
            "ax.axvline(obs*1e4, c=RED, lw=2.5, label=f'observed SPY CPI-day mean {obs*1e4:+.2f} bps')\n"
            "ax.set_xlabel('mean SPY return of a random 353-day calendar (bps)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"Squarely inside the luck cloud: canonical p = {R['plac_spy_ret_p']:.5f} \"\n"
            "             '(20 seeds x 1,000 draws)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"canonical placebo (results.md): mean {R['plac_spy_ret_mean']:+.2f} bps, \"\n"
            "      f\"sd {R['plac_spy_ret_sd']:.2f}, p = {R['plac_spy_ret_p']:.5f}\")"
        ),
        md(
            f"> 💡 In plain words: SPY's observed **{R['spy_cpi_bps']:+.2f} bps** sits right in "
            f"the middle of the null cloud (**p = {R['plac_spy_ret_p']:.5f}**) — a random calendar "
            "of the same size looks *this* special about 63% of the time. TLT's return "
            f"(*t* = {R['tlt_welch']:.2f}) is smaller but still nowhere near the bar. H₁ fails on "
            "both assets."
        ),
        md(
            "### 4b · Loudness — the realized-range split, both assets, one-sided placebo\n\n"
            "(H−L)/prev close on CPI days vs other days. Because \"louder\" has no a-priori sign "
            "problem, the placebo is **one-sided** (share of draws with mean ≥ observed)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rg_spy = st.range_stats(SPY); rg_tlt = st.range_stats(TLT)\n"
            "    a1v = [rg_spy['cpi_range_pct'], rg_spy['rest_range_pct']]\n"
            "    a2v = [rg_tlt['cpi_range_pct'], rg_tlt['rest_range_pct']]\n"
            "    t1, t2 = rg_spy['welch_t'], rg_tlt['welch_t']\n"
            "else:\n"
            "    a1v = [R['spy_rng_cpi'], R['spy_rng_rest']]\n"
            "    a2v = [R['tlt_rng_cpi'], R['tlt_rng_rest']]\n"
            "    t1, t2 = R['spy_rng_t'], R['tlt_rng_t']\n"
            "fig, (b1, b2) = plt.subplots(1, 2, figsize=(10.6, 4.3), sharey=True)\n"
            "b1.bar(['CPI day','other day'], a1v, color=[AMBER, GREY], width=.55)\n"
            "b1.set_title(f'SPY range (Welch t = {t1:+.2f}) - not certified')\n"
            "for i,v in enumerate(a1v): b1.annotate(f'{v:.3f}%',(i,v),ha='center',va='bottom')\n"
            "b2.bar(['CPI day','other day'], a2v, color=[RED, GREY], width=.55)\n"
            "b2.set_title(f'TLT range (Welch t = {t2:+.2f}) - CERTIFIED')\n"
            "for i,v in enumerate(a2v): b2.annotate(f'{v:.3f}%',(i,v),ha='center',va='bottom')\n"
            "b1.set_ylabel('mean (H-L)/prev close (%)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'SPY range: {a1v[0]:.3f}% vs {a1v[1]:.3f}%  t={t1:+.2f}')\n"
            "print(f'TLT range: {a2v[0]:.3f}% vs {a2v[1]:.3f}%  t={t2:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: TLT's range gap (**{R['tlt_rng_rest']:.2f}% → "
            f"{R['tlt_rng_cpi']:.2f}%**) clears Welch **t = {R['tlt_rng_t']:.2f}**, confirmed by "
            f"a one-sided placebo (**p = {R['plac_tlt_rng_p']:.5f}**) — this is this study's one "
            f"real, certified effect. SPY's equivalent gap (*t* = {R['spy_rng_t']:.2f}, placebo "
            f"*p* = {R['plac_spy_rng_p']:.5f}) does not clear the bar. H₂ holds for bonds, not "
            "stocks — mechanically consistent with CPI's direct role in re-pricing the discount "
            "curve that dominates a duration instrument like TLT."
        ),
        md(
            "### 4c · Event anatomy — pre-release drift and post-release persistence (SPY)\n\n"
            "Per-offset means with Welch t vs far-from-release days; the run-up is tested as a "
            "**cumulative per-event** quantity (one-sample t across 353 events)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ev = st.event_study(SPY, SPY_SESSIONS)\n"
            "    ks = list(ev.index); ms = list(ev['mean_bps']); ts = list(ev['welch_t'])\n"
            "    ru = st.runup_stats(SPY, SPY_SESSIONS)\n"
            "    ru_m, ru_t = ru['mean_runup_bps'], ru['t']\n"
            "else:\n"
            "    ks = sorted(R['event']); ms = [R['event'][k][0] for k in ks]\n"
            "    ts = [R['event'][k][1] for k in ks]; ru_m, ru_t = R['runup_bps'], R['runup_t']\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.4, 6.4), sharex=True,\n"
            "                             gridspec_kw={'height_ratios': [2, 1]})\n"
            "cols = [RED if k==0 else GREY for k in ks]\n"
            "a1.bar([str(k) for k in ks], ms, color=cols, width=.62)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean return (bps)')\n"
            "a1.set_title('No systematic pattern survives around the CPI day')\n"
            "a2.bar([str(k) for k in ks], ts, color=[RED if abs(t)>=2 else GREY for t in ts], width=.62)\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.axhline(-2, ls='--', c=RED, lw=1); a2.axhline(2, ls='--', c=RED, lw=1)\n"
            "a2.set_ylabel('Welch t'); a2.set_xlabel('offset (sessions from CPI release)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'cumulative run-up [-3..-1]: {ru_m:+.2f} bps/event (t = {ru_t:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: day 0 itself is flat (**{R['event'][0][0]:+.2f}** bps, "
            f"*t* = {R['event'][0][1]:+.2f}). Day +2 clears *t* = {R['event'][2][1]:.2f}, but "
            "that's one flagged offset out of seven with no multiple-comparison correction "
            "(≈0.3 false positives expected by chance alone), no stated mechanism, and no support "
            "from its neighbors (+1, +3) — treated as noise. The pre-release run-up "
            f"(**{R['runup_bps']:+.2f} bps/event**, *t* = {R['runup_t']:+.2f}) is suggestive but "
            "uncertified."
        ),
        md(
            "### 4d · Regime contrast — justified split, tested as a difference\n\n"
            "Split at **2022-01-01** (the Fed's 2021-12-15 hawkish pivot — accelerated taper, dot "
            "plot signaling 2022 hikes; chosen ex ante from the FOMC calendar, not snooped)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ec_r = st.era_contrast(SPY, data.REGIME_SPLIT, col='ret')\n"
            "    ec_g = st.era_contrast(SPY, data.REGIME_SPLIT, col='range_pct')\n"
            "    e, l = ec_r['early'], ec_r['late']; et, lt, dt = ec_r['welch_t_early'], ec_r['welch_t_late'], ec_r['welch_t_diff']\n"
            "    ge, gl = ec_g['early'], ec_g['late']; get_, glt, gdt = ec_g['welch_t_early'], ec_g['welch_t_late'], ec_g['welch_t_diff']\n"
            "else:\n"
            "    e, l = R['era_ret_pre'], R['era_ret_post']\n"
            "    et, lt, dt = R['era_ret_pre_t'], R['era_ret_post_t'], R['era_ret_diff_t']\n"
            "    ge, gl = R['era_rng_pre'], R['era_rng_post']\n"
            "    get_, glt, gdt = R['era_rng_pre_t'], R['era_rng_post_t'], R['era_rng_diff_t']\n"
            "fig, (c1, c2) = plt.subplots(1, 2, figsize=(11.0, 4.4))\n"
            "c1.bar(['pre-2022','post-2022'], [e, l], color=[GREY, AMBER], width=.55)\n"
            "for i,(v,t_) in enumerate([(e,et),(l,lt)]):\n"
            "    c1.annotate(f'{v:+.2f} bps\\n(t={t_:+.2f})',(i,v),ha='center',va='bottom')\n"
            "c1.axhline(0, c='k', lw=.8); c1.set_ylabel('SPY CPI-day return (bps)')\n"
            "c1.set_title(f'Return: diff t = {dt:+.2f} (not certified)')\n"
            "c2.bar(['pre-2022','post-2022'], [ge, gl], color=[GREY, AMBER], width=.55)\n"
            "for i,(v,t_) in enumerate([(ge,get_),(gl,glt)]):\n"
            "    c2.annotate(f'{v:.3f}%\\n(t={t_:+.2f})',(i,v),ha='center',va='bottom')\n"
            "c2.set_ylabel('SPY CPI-day range (%)')\n"
            "c2.set_title(f'Range: diff t = {gdt:+.2f} (not certified)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'return: pre {e:+.2f} (t={et:+.2f}) post {l:+.2f} (t={lt:+.2f}) diff t={dt:+.2f}')\n"
            "print(f'range: pre {ge:.3f}% (t={get_:+.2f}) post {gl:.3f}% (t={glt:+.2f}) diff t={gdt:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: SPY's own return and range barely budge across the 2022 "
            f"regime split (diff *t* = {R['era_ret_diff_t']:+.2f} and {R['era_rng_diff_t']:+.2f} "
            "respectively) — whatever changed about CPI's market salience since the hiking cycle "
            "began, it isn't visible as a mean shift in SPY's own numbers. (The most convincing "
            "regime move, on TLT range, is tested directly in the myth-check below.)"
        ),
        md(
            "### 4e · The myth-check — is CPI day the SINGLE BIGGEST trading day of its month?\n\n"
            "Per calendar month with a mapped CPI session: is that session's largest-magnitude "
            "day of the month, on |return| or on range? A random day in an average 21-session "
            "month clears this bar ≈4.8% of the time by chance — the honest baseline, not zero."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for name, df_, mask, met in [('SPY |ret|', SPY, SPY_SESSIONS, 'ret'),\n"
            "                                  ('SPY range', SPY, SPY_SESSIONS, 'range_pct'),\n"
            "                                  ('TLT |ret|', TLT, TLT_SESSIONS, 'ret'),\n"
            "                                  ('TLT range', TLT, TLT_SESSIONS, 'range_pct')]:\n"
            "        bd = st.biggest_day_of_month(df_, mask, data.REGIME_SPLIT, metric=met)\n"
            "        rows.append((name, bd['pre_rate']*100, bd['post_rate']*100, bd['welch_t_diff']))\n"
            "else:\n"
            "    rows = [('SPY |ret|', R['bd_spy_ret_pre'], R['bd_spy_ret_post'], R['bd_spy_ret_difft']),\n"
            "            ('SPY range', R['bd_spy_rng_pre'], R['bd_spy_rng_post'], R['bd_spy_rng_difft']),\n"
            "            ('TLT |ret|', R['bd_tlt_ret_pre'], R['bd_tlt_ret_post'], R['bd_tlt_ret_difft']),\n"
            "            ('TLT range', R['bd_tlt_rng_pre'], R['bd_tlt_rng_post'], R['bd_tlt_rng_difft'])]\n"
            "labels = [r[0] for r in rows]; pre_v = [r[1] for r in rows]; post_v = [r[2] for r in rows]\n"
            "x = np.arange(len(labels)); w = 0.35\n"
            "fig, ax = plt.subplots(figsize=(10.2, 4.6))\n"
            "ax.bar(x - w/2, pre_v, w, label='pre-2022', color=GREY)\n"
            "ax.bar(x + w/2, post_v, w, label='post-2022', color=AMBER)\n"
            "ax.axhline(R['bd_null'], ls='--', c='k', lw=1, label=f\"chance ({R['bd_null']:.1f}%)\")\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('% of months this is the biggest day')\n"
            "ax.set_title('Every metric drifts up since 2022 - none clears t = 2')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for name, pre, post, dt in rows: print(f'{name:11s}: pre {pre:5.1f}%  post {post:5.1f}%  diff t={dt:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: every one of the four metrics moves the \"more true since "
            f"2022\" direction, and the best case — TLT range — reaches **"
            f"{R['bd_tlt_rng_post']:.1f}%** of months post-2022 (up from {R['bd_tlt_rng_pre']:.1f}"
            f"%), but the pre/post difference tops out at *t* = **{R['bd_tlt_rng_difft']:.2f}**, "
            "short of the bar. H₄ is directionally plausible, statistically unconfirmed — and "
            "even the best point estimate (15%) means CPI is the month's loudest bond day roughly "
            "one month in seven, not \"the biggest day of the month\" as a rule."
        ),
        md(
            "### 4f · Tradability — the naive timer, cost-swept\n\n"
            "Enter SPY at the prior close (zero look-ahead — the calendar is public months "
            "ahead), exit the CPI-day close, pay 2 × one-way costs per event."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = [st.timer_capture(SPY, cost_bps=cb) for cb in (5.0, 10.0)]\n"
            "    g = rows[0]['gross_bps']; n5, n10 = rows[0]['net_bps'], rows[1]['net_bps']\n"
            "    tv, worst = rows[0]['welch_t'], rows[0]['worst_day_pct']\n"
            "else:\n"
            "    g, n5, n10 = R['tm_gross'], R['tm_net5'], R['tm_net10']\n"
            "    tv, worst = R['tm_t'], R['tm_worst']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['gross','net 5 bps','net 10 bps'], [g, n5, n10], color=[GREY, RED, RED], width=.6)\n"
            "for i,v in enumerate([g, n5, n10]): ax.annotate(f'{v:+.2f}',(i,v),ha='center',\n"
            "    va='top' if v<0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('bps per CPI-day timer bet')\n"
            "ax.set_title(f'No gross edge to start from (Welch t = {tv:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.2f} -> net {n5:+.2f} / {n10:+.2f} bps  (t={tv:+.2f}, worst day {worst:+.1f}%)')"
        ),
        md(
            f"> 💡 In plain words: the timer starts from **{R['tm_gross']:+.2f} bps** gross — "
            f"barely distinguishable from the {R['tm_rest']:+.2f} bps SPY earns on an ordinary "
            f"day (*t* = {R['tm_t']:.2f}) — so any realistic cost (5-10 bps one-way) flips it "
            f"negative (**{R['tm_net5']:+.2f} / {R['tm_net10']:+.2f} bps**, "
            f"**{R['tm_ann5']:+.2f}% / {R['tm_ann10']:+.2f}%/yr**). There was never a directional "
            "edge to begin with; charging costs against nothing just produces a loss."
        ),
        md(
            "### 4g · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic daily-return process, scheduled pseudo-CPI days every 21st business day, "
            "TWO independently tunable planted effects (a return-mean shift and a volatility "
            "multiplier). The null is checked over **20 seeds** — never a single stream — on "
            "BOTH axes, and each knob must be recovered WITHOUT contaminating the other."
        ),
        code(
            "null_ret, null_rng = [], []\n"
            "for s_ in range(20):\n"
            "    close, dec = data.synthetic_world(mu_shift=0.0, vol_mult=1.0, seed=644 + s_)\n"
            "    d = st.synthetic_detect(close, dec)\n"
            "    null_ret.append(d['welch_t']); null_rng.append(d['range_welch_t'])\n"
            "null_ret = np.asarray(null_ret); null_rng = np.asarray(null_rng)\n"
            "close, dec = data.synthetic_world(mu_shift=0.0015, vol_mult=1.0, seed=644)\n"
            "planted_ret_t = st.synthetic_detect(close, dec)['welch_t']\n"
            "close, dec = data.synthetic_world(mu_shift=0.0, vol_mult=1.6, seed=644)\n"
            "d_rng = st.synthetic_detect(close, dec)\n"
            "planted_rng_t, cross_ret_t = d_rng['range_welch_t'], d_rng['welch_t']\n"
            "fig, (p1, p2) = plt.subplots(1, 2, figsize=(11.0, 4.4))\n"
            "p1.scatter(np.zeros(20) + np.linspace(-.12,.12,20), null_ret, color=GREY, s=40, label='null x 20')\n"
            "p1.scatter([1], [planted_ret_t], color=RED, s=90, zorder=5, label='planted return shift')\n"
            "p1.axhline(-2, ls='--', c=RED, lw=1); p1.axhline(2, ls='--', c=RED, lw=1)\n"
            "p1.set_xticks([0,1]); p1.set_xticklabels(['null','planted']); p1.set_ylabel('Welch t (return)')\n"
            "p1.set_title('Return-shift detector'); p1.legend()\n"
            "p2.scatter(np.zeros(20) + np.linspace(-.12,.12,20), null_rng, color=GREY, s=40, label='null x 20')\n"
            "p2.scatter([1], [planted_rng_t], color=RED, s=90, zorder=5, label='planted vol x1.6')\n"
            "p2.axhline(-2, ls='--', c=RED, lw=1); p2.axhline(2, ls='--', c=RED, lw=1)\n"
            "p2.set_xticks([0,1]); p2.set_xticklabels(['null','planted']); p2.set_ylabel('Welch t (range)')\n"
            "p2.set_title('Vol-multiplier detector'); p2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'null: return mean t={null_ret.mean():+.2f} (sd {null_ret.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ret)>=2).sum()}/20  |  range mean t={null_rng.mean():+.2f} '\n"
            "      f'(sd {null_rng.std(ddof=1):.2f}), |t|>=2 in {(abs(null_rng)>=2).sum()}/20')\n"
            "print(f'planted return shift t={planted_ret_t:+.2f}  |  planted vol-mult range '\n"
            "      f't={planted_rng_t:+.2f} (cross-contamination check, return t={cross_ret_t:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds neither detector fires "
            f"(return mean *t* = {R['syn_null_ret_mean']:+.2f}, {R['syn_null_ret_fire']}/20 "
            f"crossing the bar by chance; range mean *t* = {R['syn_null_rng_mean']:+.2f}, "
            f"{R['syn_null_rng_fire']}/20). A planted +15 bp return shift reads "
            f"*t* = {R['syn_planted_ret_t']:.2f}; a planted 1.6× vol multiplier reads range "
            f"*t* = {R['syn_planted_rng_t']:.2f} **without** contaminating the return detector "
            f"(*t* = {R['syn_cross_ret_t']:.2f}, small). The machinery can catch either the "
            "direction effect or the loudness effect independently — exactly what the real-tape "
            "split (real on TLT range, none on return) requires it to be capable of. *(A "
            "faithful-engine / power check only — never cited in support of the real-tape "
            "stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `MIXED`** — real on one leg, none on the rest: TLT's realized range is "
            f"genuinely elevated on CPI mornings (Welch *t* = **{R['tlt_rng_t']:.2f}**, one-sided "
            f"placebo *p* = **{R['plac_tlt_rng_p']:.5f}**), mechanically sound (CPI directly "
            f"re-prices rate expectations). SPY return (*t* = {R['spy_welch']:.2f}), TLT return "
            f"(*t* = {R['tlt_welch']:.2f}) and SPY's own range (*t* = {R['spy_rng_t']:.2f}, "
            f"placebo *p* = {R['plac_spy_rng_p']:.5f}) all fail the **t ≥ 2** bar.\n"
            f"- **Tradability `MIRAGE`** — no directional edge exists to bank; the naive SPY "
            f"timer nets **{R['tm_net5']:+.2f} bps/event** (≈{R['tm_ann5']:+.2f}%/yr) at 5 bps "
            "costs, starting from a non-edge. The one certified effect (TLT range) needs an "
            "options/vol instrument to harvest, which this study doesn't test and which the FOMC "
            "precedent (637) suggests is pre-priced ahead of the print.\n"
            f"- **\"Biggest day of the month\"? `BUSTED`** — every metric drifts the \"more true "
            f"since 2022\" way, best case TLT range diff *t* = **{R['bd_tlt_rng_difft']:.2f}**, "
            "but none of the four pre/post differences clears *t* = 2, and even the strongest "
            f"post-2022 rate ({R['bd_tlt_rng_post']:.1f}% of months) is a small minority — not "
            "literally \"the\" biggest day."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general object is the scheduled-data-release premium.** Faust, Rogers, Wang "
            "& Wright (2007) and Balduzzi, Elton & Green (2001) document fast, sharp fixed-income "
            "reactions to macro surprises specifically — the mechanism behind why TLT, not SPY, "
            "carries this study's certified effect.\n"
            "- **The natural sequel is an options-based test.** If bond volatility genuinely "
            "spikes on CPI day, the honest way to ask whether that's bankable is a straddle or "
            "variance-swap backtest on rates around the print — not a directional cash-bond "
            "position, which this study shows has nothing to offer.\n"
            "- **Dedup map:** [602-macro-announcement-premium](../../602-macro-announcement-premium/) "
            "(the pooled FOMC+CPI+NFP Savor-Wilson premium — traces almost entirely to FOMC, not "
            "CPI), [643-payrolls-day-effect](../../643-payrolls-day-effect/) (identical protocol, "
            "Nonfarm Payrolls mornings instead), [637-fomc-vol-crush](../../637-fomc-vol-crush/) "
            "(the FOMC decision *afternoon* — implied vol, not realized range, and the cautionary "
            "tale that a real vol effect can still be untradable).\n\n"
            "*The reproducible core is offline and deterministic; frozen numbers live in "
            "[`docs/results.md`](../docs/results.md), sources in "
            "[`docs/references.md`](../docs/references.md).*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "02_for_the_quants.ipynb")


def _meta():
    return {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }


def _write(nb, name):
    path = os.path.join(HERE, name)
    with open(path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", path)


if __name__ == "__main__":
    build_curious()
    build_quants()
