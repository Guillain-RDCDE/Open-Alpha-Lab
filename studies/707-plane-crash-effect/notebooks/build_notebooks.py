"""Generate the two narrative notebooks for Study 707 (Plane-Crash-Effect).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached
SPY/AAL/DAL/UAL/LUV tapes under ../_cache/ and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md). The synthetic positive control runs
anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance SPY/AAL/DAL/UAL/LUV
# 2000-01-03 -> 2026-06-30; 36 hardcoded major aviation disasters 2000-01-31 -> 2025-06-12).
R = dict(
    n_events=36, cal_lo="2000-01-31", cal_hi="2025-06-12",
    day0_mean_bps=0.41, day0_t=0.02,
    hit=17, hit_n=36, hit_pct=47.2, wilson=(32.0, 63.0),
    placebo_mean_bps=0.11, placebo_sd_bps=20.13, placebo_p=0.506, placebo_draws=20000,
    # event window: offset -> (mean_bps, car_bps, t)
    event={-1: (-11.86, 0.00, -0.72), 0: (0.41, 0.41, 0.02), 1: (-0.05, 0.36, -0.00),
           2: (-49.21, -48.84, -2.38), 3: (30.70, -18.15, 1.81), 4: (-18.71, -36.86, -0.98),
           5: (-0.80, -37.66, -0.06)},
    reversal_mean_bps=-38.07, reversal_t=-0.90,
    air_mean_bps=43.22, spy_mean_bps=0.41, air_diff_bps=42.81, air_t=1.11, air_n=36,
    air_full_diff_bps=37.29, air_full_t=0.79, air_full_n=27,
    # buy-the-dip timer: hold -> (gross_bps, net5_bps, t_net5, baseline_bps)
    timer={1: (3.85, -6.15, -0.39, 3.90), 3: (-7.03, -17.03, -0.55, 11.49),
           5: (-18.30, -28.30, -0.68, 18.97), 10: (-30.70, -40.70, -0.62, 37.37)},
    timer_net10_5d=-38.30, timer_net10_5d_t=-0.91,
    # synthetic control
    syn_null_mean=-0.32, syn_null_sd=0.82, syn_null_fire=0,
    syn_planted_t=-14.59, syn_planted_bps=-205.9,
    fp_spy="0732ff66475c", fp_aal="ebf1693dea47", fp_dal="6d728d6446e7",
    fp_ual="4cbbbdd0ad8e", fp_luv="5c6b37f8ffb1",
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Flinch%3F: Busted](https://img.shields.io/badge/Flinch%3F-Busted-8b949e?style=flat-square)\n\n"
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

from plane_crash_effect import data, strategy as st

EVENTS = data.disaster_table()
HAVE_REAL = data.have_real()
if HAVE_REAL:
    SPY, AIRLINES = data.load_real()
    RET_SPY = st.daily_returns(SPY)
    AR_SPY = st.abnormal_returns(RET_SPY)
    BASKET, COVERAGE = st.airline_basket_returns(AIRLINES)
    AR_AIR = st.abnormal_returns(BASKET)
else:
    SPY = AIRLINES = RET_SPY = AR_SPY = BASKET = COVERAGE = AR_AIR = None
print("real cache present:", HAVE_REAL, "| disasters in table:", len(EVENTS))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the market flinch when a plane goes down? ✈️📉\n"
            "### The \"plane-crash effect\" — a tidy behavioral-finance story that "
            "quietly doesn't show up anymore\n\n"
            + BADGES +
            "Every time a major airliner goes down, some corner of financial media "
            "repeats the same idea: a tragedy this vivid puts investors in a bad mood, "
            "and a bad mood makes people sell — a small, fear-driven dip in the whole "
            "market, not just airline stocks, that fades again within days. It's a real, "
            "published academic finding (Kaplanski & Levy, 2010) with a real mechanism: "
            "human psychology doesn't cleanly separate \"this news is sad\" from \"I should "
            "sell my portfolio.\"\n\n"
            "That's the claim we test on **36 of the most infamous crashes since 2000** — "
            "does the market actually dip? Do airline stocks, which are genuinely exposed, "
            "drop harder? And could you have traded any of it?\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the paired "
            "airline test? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Ethical note, stated plainly.** The \"buy the dip\" test near the end is "
            "a clinical statistical exercise — a test of whether a pattern exists in public "
            "market data — not a suggestion to trade around tragedy. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the market dip the day a crash makes the news? | **No — not "
            f"detectably.** The S&P 500 moves **{R['day0_mean_bps']:+.2f} bps** on the "
            f"average crash day — statistically nothing (odds a random day beats it: "
            f"about **{R['placebo_p']*100:.0f}%**, i.e. a coin flip). |\n"
            "| Do airline stocks fall harder than the market? | **No.** Paired against "
            f"the S&P on the same day, the 4-carrier airline basket (AAL/DAL/UAL/LUV) "
            f"actually moves *less* negatively on average — the sign is backwards for "
            "the claim. |\n"
            "| Does the dip reverse within a few days? | **There's no dip to reverse.** "
            "The days after a crash drift mildly but not meaningfully. |\n"
            "| Could you trade it? | **No edge to trade.** \"Buying the dip\" after a "
            "crash loses to simply holding the S&P at every horizon we tried (1 to 10 "
            "days), before *and* after costs. |\n\n"
            "> A well-known behavioral-finance result, tested again on today's market and "
            "today's instruments — and it doesn't show up."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A major air disaster is front-page news everywhere, instantly and "
            "viscerally. Even investors with zero exposure to airlines absorb the mood — "
            "and a soured mood shows up, briefly, as selling across the whole market. "
            "Airline stocks, which actually have something at stake, should fall "
            "further still.\"*\n\n"
            "This isn't fringe folklore — it's a published finding (Kaplanski & Levy, "
            "2010, *Journal of Financial and Quantitative Analysis*) in the same family "
            "as \"sunny days lift stock returns\" and \"a home team's tournament "
            "elimination sours the home market\": non-fundamental news moving prices "
            "through mood, not information."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real and tradable, this would be a clean demonstration that **markets "
            "aren't purely rational information processors** — that raw sentiment, "
            "unconnected to any company's actual cash flows, can move prices in a "
            "predictable direction. And the trade write itself: buy the dip a mood shock "
            "creates, once the mood (but nothing fundamental) has passed.\n\n"
            "So we ask three things: is the market-wide dip real, do airlines actually "
            "get hit harder, and does \"buy the disaster dip\" pay after costs?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The calendar.** **{R['n_events']}** major commercial-aviation disasters "
            f"from {R['cal_lo']} to {R['cal_hi']} — hand-picked as the crashes any "
            "reasonable person would call front-page news. (9/11 and MH17 are left out on "
            "purpose — both are terror/war events already tested in a sibling study, not "
            "accidents.)\n"
            "- **The comparison.** The S&P 500's return on the first tradable session "
            "after each crash vs. every other day since 2000.\n"
            "- **The luck check.** Draw 36 random days instead, 20,000 times — how often "
            "does a random calendar produce a dip this size (or bigger)?\n"
            "- **The airline check.** Same day, same crash: does a 4-carrier US airline "
            "basket (American, Delta, United, Southwest) move *more* negatively than the "
            "S&P?\n"
            "- **The trade check.** Buy the S&P at the crash-day close, sell a few "
            "sessions later, pay costs, compare to just holding."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Average S&P 500 move on the day a crash breaks, "
            "vs. what a random day looks like."
        ),
        code(
            "if HAVE_REAL:\n"
            "    d0 = st.day0_stats(AR_SPY, EVENTS['date'], pre=1, post=5)\n"
            "    day0_bps = d0['mean'] * 1e4\n"
            "else:\n"
            "    day0_bps = R['day0_mean_bps']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['crash day\\n(n=36)', 'random day\\n(placebo mean)'],\n"
            "       [day0_bps, R['placebo_mean_bps']], color=[RED, GREY], width=.55)\n"
            "for i, v in enumerate([day0_bps, R['placebo_mean_bps']]):\n"
            "    ax.annotate(f'{v:+.2f} bps', (i, v), ha='center',\n"
            "                va='top' if v < 0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('average S&P 500 move (bps)')\n"
            "ax.set_title('The crash-day bar is not taller than the random-day bar')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'crash day {day0_bps:+.2f} bps vs random day {R[\"placebo_mean_bps\"]:+.2f} bps')"
        ),
        md(
            f"That's it — **{R['day0_mean_bps']:+.2f} bps** on the average crash day, "
            f"barely different from **{R['placebo_mean_bps']:+.2f} bps** on a random day. "
            f"The market fell on only **{R['hit']}/{R['hit_n']} = {R['hit_pct']:.0f}%** "
            "of crash days — a coin flip. The quants notebook shows a random 36-day "
            "calendar beats this about half the time.\n\n"
            "**Next, the week around the crash.** Does a dip show up a day or two later "
            "instead?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cp = st.car_path_stats(AR_SPY, EVENTS['date'], pre=1, post=5)\n"
            "    ks, ms = list(cp.index), list(cp['mean_ar'] * 1e4)\n"
            "else:\n"
            "    ks = sorted(R['event']); ms = [R['event'][k][0] for k in ks]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.5))\n"
            "cols = [RED if k == 0 else (AMBER if k == 2 else GREY) for k in ks]\n"
            "ax.bar([str(k) for k in ks], ms, color=cols, width=.62)\n"
            "for i, v in enumerate(ms):\n"
            "    ax.annotate(f'{v:+.1f}', (i, v), ha='center',\n"
            "                va='top' if v < 0 else 'bottom', fontsize=9)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('trading days relative to the crash (0 = first tradable session)')\n"
            "ax.set_ylabel('average S&P 500 abnormal return (bps)')\n"
            "ax.set_title('No pattern that survives a straight face — one noisy bar, no story')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('offsets (bps):', {k: round(m, 2) for k, m in zip(ks, ms)})"
        ),
        md(
            "The only bar that stands out is **+2 days** — a noisy, unexplained dip with "
            "no mechanism attached to it. When you test 7 different days, one of them "
            "crossing the \"interesting\" threshold by pure chance is *exactly* what "
            "you'd expect (that's the quants notebook's job to check properly) — it isn't "
            "a second, delayed version of the effect.\n\n"
            "**Now, the part that should matter most if the story is true.** Airlines are "
            "the sector that actually has skin in the game — fleets, lawsuits, "
            "reputational damage. Do they drop harder?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    extra = st.airline_extra_drop(AR_SPY, AR_AIR, EVENTS['date'], pre=1, post=5)\n"
            "    air_b, spy_b = extra['air_mean'] * 1e4, extra['spy_mean'] * 1e4\n"
            "else:\n"
            "    air_b, spy_b = R['air_mean_bps'], R['spy_mean_bps']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['S&P 500', 'airline basket\\n(AAL/DAL/UAL/LUV)'], [spy_b, air_b],\n"
            "       color=[GREY, AMBER], width=.55)\n"
            "for i, v in enumerate([spy_b, air_b]):\n"
            "    ax.annotate(f'{v:+.1f} bps', (i, v), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('average crash-day move (bps)')\n"
            "ax.set_title('Airlines do not fall harder than the market on their own bad news')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'S&P {spy_b:+.2f} bps  vs  airline basket {air_b:+.2f} bps')"
        ),
        md(
            "Airlines don't fall harder — if anything, in this sample they move slightly "
            "*less* negatively than the broad market on the crash's first session. Most "
            "of the crashes in the table are foreign carriers with no direct fundamental "
            "link to American/Delta/United/Southwest, so this isn't shocking — but it's "
            "also exactly the piece of the claim that should be the *easiest* to find if "
            "the sentiment story were strong, and it isn't there.\n\n"
            "**Finally, the trade.** Could you have made money buying the dip?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    vals = []\n"
            "    for h in (1, 3, 5, 10):\n"
            "        ledger = st.buy_the_dip(SPY, EVENTS['date'], hold=h, cost_bps=5.0)\n"
            "        s = st.summarize_dip(ledger)\n"
            "        vals.append(s['mean_bps'])\n"
            "    holds = [1, 3, 5, 10]\n"
            "else:\n"
            "    holds = sorted(R['timer'])\n"
            "    vals = [R['timer'][h][1] for h in holds]\n"
            "base = [R['timer'][h][3] for h in holds]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.5))\n"
            "x = np.arange(len(holds)); w = 0.36\n"
            "ax.bar(x - w/2, vals, width=w, color=RED, label='buy the crash dip (net, 5 bps)')\n"
            "ax.bar(x + w/2, base, width=w, color=GREY, label='just hold the S&P (unconditional)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in holds])\n"
            "ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('Buying the dip loses to just holding the index, every time')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('net buy-the-dip (bps):', dict(zip(holds, [round(v,1) for v in vals])))\n"
            "print('unconditional baseline (bps):', dict(zip(holds, [round(v,1) for v in base])))"
        ),
        md(
            "At every holding period we tried, buying the S&P right after a crash comes "
            "out **behind** simply holding the index the whole time — the point estimates "
            "are all negative net of costs, though none of them are far enough from zero "
            "to call it a real *losing* strategy either. There is simply no dip-buying "
            "edge to find, and the ethical caveat that comes with even asking the "
            "question is moot: the honest answer and the comfortable answer are the same "
            "one."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The S&P moves an indistinguishable-from-zero "
            f"**{R['day0_mean_bps']:+.2f} bps** on the average crash day, the airline "
            "basket doesn't fall harder, and a random calendar reproduces this about "
            "half the time. The famous \"aviation-disaster sentiment\" effect doesn't "
            "show up on today's tape.\n"
            "- **Tradability — Mirage.** There's no edge to deploy — buying the dip "
            "loses to just holding, before you even get to costs.\n"
            "- **\"Does the market flinch at a plane crash?\" — Busted.** Not on the "
            "modern, tradable version of the test."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **This doesn't erase Kaplanski & Levy's original finding.** Their sample "
            "runs 1950–2005 and screens a much broader accident register by fatality "
            "count; a small, real effect in a much longer, differently-built sample can "
            "genuinely be invisible in 36 hand-picked headline crashes since 2000 — the "
            "honest conclusion is \"not detectable at this size on this tape\", not "
            "\"proven false everywhere, always.\"\n"
            "- **Where a real version might live:** narrower sentiment proxies (put-call "
            "skew, news-sentiment indices, retail order flow) rather than the blunt "
            "instrument of a broad-index close-to-close return, or a larger accident "
            "register that includes smaller, less headline-grabbing crashes.\n"
            "- **Sibling studies:** [313-geopolitical-shock](../../313-geopolitical-shock/) "
            "(wars/attacks, including 9/11 and MH17), "
            "[300-sports-sentiment](../../300-sports-sentiment/) (the Edmans \"loss "
            "effect\") and [279-geomagnetic-storms](../../279-geomagnetic-storms/) "
            "(a physiological mood channel) — the same \"does mood move markets\" "
            "question, different triggers.\n\n"
            "*Think a narrower sentiment measure would catch what a broad-index close "
            "misses? Show it — on out-of-sample crashes, after costs — then we'll talk.*"
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
            "# The Plane-Crash-Effect — a quantitative teardown 🔬\n"
            "### The crash-day event study · a 20-seed random-calendar placebo · the "
            "[−1..+5] anatomy and its look-elsewhere caveat · a paired airline-extra-drop "
            "test · a costed buy-the-dip timer · a 20-seed synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious]"
            "(01_for_the_curious.ipynb). The claim — Kaplanski & Levy (2010): a major "
            "aviation disaster triggers a transient, market-wide sentiment dip, sharper "
            "in airline stocks — has a real academic anchor and a stated mood mechanism. "
            "The job here is to measure it honestly on today's tradable tape, then ask "
            "the only question that pays: *is any of it real, and if so, tradable?*\n\n"
            "> ⚠️ **Data note.** SPY + AAL/DAL/UAL/LUV total-return closes (2000→2026), "
            "yfinance, cached; **36 hardcoded major commercial-aviation disasters** "
            "2000→2025 (9/11 and MH17 excluded — terror/war events already tested in "
            "313-geopolitical-shock). No survivorship on the SPY axis; **airline-basket "
            "coverage is named** (full 4-carrier only from 2008; a full-coverage-only "
            "n=27 subsample cross-checks the full n=36). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprints `" + R["fp_spy"] +
            "` SPY / `" + R["fp_aal"] + "` AAL / `" + R["fp_dal"] + "` DAL / `" +
            R["fp_ual"] + "` UAL / `" + R["fp_luv"] + "` LUV).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to "
            "intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | crash-day abnormal SPY return "
            f"**{R['day0_mean_bps']:+.2f} bps**, one-sample **t = {R['day0_t']:+.2f}**, "
            f"hit rate {R['hit_pct']:.1f}% (Wilson [{R['wilson'][0]:.1f}%, "
            f"{R['wilson'][1]:.1f}%]), placebo **p = {R['placebo_p']:.3f}** |\n"
            f"| **Tradability** | `MIRAGE` | buy-the-dip underperforms the unconditional "
            f"baseline at every horizon (1/3/5/10d), worst-case *t* = "
            f"{R['timer_net10_5d_t']:.2f} at 10 bps costs |\n"
            f"| **Airline extra drop?** | `NONE` | paired (airline − SPY) diff "
            f"**{R['air_diff_bps']:+.2f} bps**, *t* = **{R['air_t']:.2f}** — wrong sign "
            "for the claim |\n\n"
            "> 💡 In plain words: the folklore has a real mechanism and a real citation, "
            "but the modern, tradable version of the test comes back empty on every axis."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t$ be SPY's daily close-to-close return and $a_t = r_t - \\bar{r}$ "
            "its abnormal return under a constant-mean market model (Brown & Warner "
            "1985). For each crash $i$ with first-tradable-session date $\\tau_i$, the "
            "claims:\n\n"
            "- **H₁ (market dip).** $E[a_{\\tau_i}] < 0$, large and systematic across "
            "events — not a calendar coincidence.\n"
            "- **H₂ (reversal).** The dip is *temporary*: $\\sum_{k=1}^{5} a_{\\tau_i+k}$ "
            "is positive and offsets it.\n"
            "- **H₃ (sector amplification).** Airline-basket abnormal returns on the "
            "same day are *more negative* than SPY's: $E[a^{air}_{\\tau_i} - "
            "a^{spy}_{\\tau_i}] < 0$.\n"
            "- **H₄ (capture).** A dip-buy overlay on SPY, held a few sessions, beats "
            "the unconditional baseline net of costs.\n\n"
            "We find **H₁ not supported** (t = 0.02), **H₂ moot** (nothing to reverse), "
            "**H₃ rejected on sign** (t = 1.11, positive not negative), **H₄ not "
            "supported** (negative point estimates at every horizon, never significant "
            "either way)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Crash events are **independent, non-overlapping calendar dates** (the "
            "closest pair in the table is still weeks apart), so the planned primary is "
            "a **one-sample t-test** across the 36 (or fewer, for coverage-limited tests) "
            "per-event abnormal returns — no HAC correction is needed the way a daily "
            "panel regression would require, because the unit of analysis is already "
            "\"one number per event\", not a daily series. The hit rate carries a "
            "**Wilson interval**; the placebo draws 36 random non-crash dates **20,000 "
            "times (20 seeds × 1,000)**; the [−1..+5] anatomy is read as a **7-offset "
            "multiple-comparison** exercise, not 7 independent shots at significance; "
            "the airline test **pairs** each event's SPY and airline-basket move on the "
            "same date before t-testing the difference, so any common market-wide move "
            "cancels out of H₃ by construction."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Calendar.** {R['n_events']} major disasters {R['cal_lo']} → "
            f"{R['cal_hi']}, hardcoded (9/11, MH17 excluded — see references.md). All "
            "36 sit on the SPY tape.\n"
            f"- **Tape.** SPY + AAL/DAL/UAL/LUV total-return closes, 2000 → 2026-06-30 "
            "(as-of, last complete month). Each airline ticker starts at its "
            "current-entity trading start.\n"
            "- **Headline.** Crash-day (offset 0) one-sample t + Wilson hit rate + "
            "20-seed placebo.\n"
            "- **Anatomy.** Event window [−1..+5], per-offset mean + t, read with the "
            "multiple-comparison caveat; a dedicated H₂ reversal test on the summed "
            "[+1..+5] abnormal return.\n"
            "- **Sector test.** Paired (airline basket − SPY) day-0 difference, full "
            "sample and a full-4-carrier-coverage-only subsample (n=27, events ≥ 2008).\n"
            "- **Execution (timer).** Enter SPY at the crash-day close (zero look-ahead "
            "— the crash date predates that session's close by construction), exit "
            "`hold` sessions later; 2 × one-way cost × NAV per event; long-only.\n"
            "- **Control.** Synthetic random-walk tape, planted mean-reverting crash-day "
            "dip; the null must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline split and its placebo\n\n"
            "One-sample t on the crash-day (offset 0) abnormal SPY return, Wilson hit "
            "rate, and the random-calendar null. In the notebook we run a lighter "
            "placebo (4 seeds × 500 draws) and quote the canonical 20,000-draw p from "
            "`results.md`."
        ),
        code(
            "if HAVE_REAL:\n"
            "    d0 = st.day0_stats(AR_SPY, EVENTS['date'], pre=1, post=5)\n"
            "    hit = int(sum(1 for d in d0['kept_dates']\n"
            "                   if st.event_window(AR_SPY, d, 1, 5)[1] < 0))\n"
            "    print(f\"crash-day abnormal SPY return {d0['mean']*1e4:+.2f} bps  \"\n"
            "          f\"one-sample t = {d0['t']:+.3f}  (n={d0['n']})\")\n"
            "    print(f\"hit rate: {hit}/{d0['n']} = {hit/d0['n']*100:.1f}% down\")\n"
            "    pl = st.placebo_distribution(AR_SPY, d0['n'], pre=1, post=5,\n"
            "                                  n_draws=500, seed=707, stat='day0')\n"
            "    for s_ in range(1, 4):\n"
            "        pl = np.concatenate([pl, st.placebo_distribution(\n"
            "            AR_SPY, d0['n'], pre=1, post=5, n_draws=500,\n"
            "            seed=707 + s_, stat='day0')])\n"
            "    obs, draws = d0['mean'], pl\n"
            "else:\n"
            "    obs = R['day0_mean_bps'] / 1e4\n"
            "    rng = np.random.default_rng(707)\n"
            "    draws = rng.normal(R['placebo_mean_bps'] / 1e4, R['placebo_sd_bps'] / 1e4, 2000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws * 1e4, bins=50, color=GREY, alpha=.85,\n"
            "        label='null: random calendars of 36 days (light in-notebook run)')\n"
            "ax.axvline(obs * 1e4, c=RED, lw=2.5,\n"
            "           label=f'observed crash-day mean {obs*1e4:+.2f} bps')\n"
            "ax.set_xlabel('mean abnormal SPY return of a random 36-day calendar (bps)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"Squarely inside the luck cloud: canonical p = {R['placebo_p']:.3f} \"\n"
            "             '(20 seeds x 1,000 draws)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"canonical placebo (results.md): mean {R['placebo_mean_bps']:+.2f} bps, \"\n"
            "      f\"sd {R['placebo_sd_bps']:.2f} bps, p = {R['placebo_p']:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: the observed **{R['day0_mean_bps']:+.2f} bps** sits "
            f"almost exactly at the center of the null's distribution "
            f"({R['placebo_mean_bps']:+.2f} ± {R['placebo_sd_bps']:.2f} bps); "
            f"**p = {R['placebo_p']:.3f}** means a random 36-day calendar beats this "
            f"observation about half the time. With one-sample t = **{R['day0_t']:.2f}** "
            "and a hit rate indistinguishable from 50%, H₁ does not clear the desk bar "
            "by any measure."
        ),
        md(
            "### 4b · Anatomy — the [−1..+5] window, read with the multiple-comparison caveat\n\n"
            "Per-offset means with each offset's own one-sample t. With 7 offsets tested, "
            "roughly a 1-in-7 chance one crosses |t| ≥ 2 by pure chance at the "
            "conventional 5% level — exactly what happens at offset +2, with no "
            "mechanism attached to it."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cp = st.car_path_stats(AR_SPY, EVENTS['date'], pre=1, post=5)\n"
            "    ks = list(cp.index); ms = list(cp['mean_ar'] * 1e4); ts = list(cp['t'])\n"
            "    rev = st.reversal_stats(AR_SPY, EVENTS['date'], pre=1, post=5)\n"
            "    rev_m, rev_t = rev['mean'] * 1e4, rev['t']\n"
            "else:\n"
            "    ks = sorted(R['event']); ms = [R['event'][k][0] for k in ks]\n"
            "    ts = [R['event'][k][2] for k in ks]\n"
            "    rev_m, rev_t = R['reversal_mean_bps'], R['reversal_t']\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.4, 6.4), sharex=True,\n"
            "                             gridspec_kw={'height_ratios': [2, 1]})\n"
            "cols = [RED if k == 0 else GREY for k in ks]\n"
            "a1.bar([str(k) for k in ks], ms, color=cols, width=.62)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean abnormal return (bps)')\n"
            "a1.set_title('Event anatomy: no crash-day dip, one unexplained noisy bar')\n"
            "a2.bar([str(k) for k in ks], ts,\n"
            "       color=[RED if abs(t) >= 2 else GREY for t in ts], width=.62)\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.axhline(-2, ls='--', c=RED, lw=1); a2.axhline(2, ls='--', c=RED, lw=1)\n"
            "a2.set_ylabel('one-sample t'); a2.set_xlabel('offset (sessions from the crash)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'post-crash [+1..+5] reversal test: {rev_m:+.2f} bps (t = {rev_t:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: day 0 is **{R['event'][0][0]:+.2f} bps** "
            f"(t = {R['event'][0][2]:+.2f}) — nothing. Offset +2 stands out at "
            f"**{R['event'][2][0]:+.2f} bps** (t = {R['event'][2][2]:.2f}), but testing "
            "7 offsets means one crossing the bar by chance is the *expected* outcome, "
            "not a discovery — it has no mechanism (why would news from two days ago "
            f"hit *now*?) and doesn't replicate in the headline or the placebo. The "
            f"reversal test (H₂) is moot: cumulative [+1..+5] is "
            f"**{R['reversal_mean_bps']:+.2f} bps** at t = **{R['reversal_t']:.2f}** — "
            "there's no dip to reverse."
        ),
        md(
            "### 4c · The airline test — paired extra drop\n\n"
            "Same date, SPY vs. the 4-carrier basket: pairing removes the common "
            "market-wide move, isolating whether airlines carry an incremental reaction."
        ),
        code(
            "if HAVE_REAL:\n"
            "    extra = st.airline_extra_drop(AR_SPY, AR_AIR, EVENTS['date'], pre=1, post=5)\n"
            "    full_cov = EVENTS.loc[EVENTS['date'] >= '2008-01-01', 'date']\n"
            "    extra_full = st.airline_extra_drop(AR_SPY, AR_AIR, full_cov, pre=1, post=5)\n"
            "    air_b, spy_b = extra['air_mean'] * 1e4, extra['spy_mean'] * 1e4\n"
            "    diff, t_, n_ = extra['mean_diff'] * 1e4, extra['t'], extra['n']\n"
            "    diff_f, t_f, n_f = (extra_full['mean_diff'] * 1e4, extra_full['t'],\n"
            "                        extra_full['n'])\n"
            "else:\n"
            "    air_b, spy_b = R['air_mean_bps'], R['spy_mean_bps']\n"
            "    diff, t_, n_ = R['air_diff_bps'], R['air_t'], R['air_n']\n"
            "    diff_f, t_f, n_f = R['air_full_diff_bps'], R['air_full_t'], R['air_full_n']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.8, 4.3))\n"
            "a1.bar(['S&P 500', 'airline basket'], [spy_b, air_b], color=[GREY, AMBER], width=.55)\n"
            "for i, v in enumerate([spy_b, air_b]):\n"
            "    a1.annotate(f'{v:+.1f} bps', (i, v), ha='center', va='bottom')\n"
            "a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_ylabel('crash-day abnormal return (bps)')\n"
            "a1.set_title(f'Paired diff = {diff:+.1f} bps (t = {t_:+.2f}, n={n_})')\n"
            "a2.bar(['full sample\\n(n=' + str(n_) + ')', 'coverage>=2008\\n(n=' + str(n_f) + ')'],\n"
            "       [diff, diff_f], color=[AMBER, GREY], width=.55)\n"
            "for i, (v, t2) in enumerate([(diff, t_), (diff_f, t_f)]):\n"
            "    a2.annotate(f'{v:+.1f} bps\\n(t={t2:+.2f})', (i, v), ha='center', va='bottom')\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_ylabel('airline - SPY paired diff (bps)')\n"
            "a2.set_title('Robust to dropping the thin-coverage early years')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'airline {air_b:+.2f} bps vs SPY {spy_b:+.2f} bps -> diff {diff:+.2f} bps '\n"
            "      f'(t={t_:+.2f}); full-coverage-only diff {diff_f:+.2f} bps (t={t_f:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: H₃ predicts a **negative** paired difference (airlines "
            f"fall harder). We measure **{R['air_diff_bps']:+.2f} bps** at "
            f"t = **{R['air_t']:.2f}** — positive, and nowhere near significant either "
            "way. Restricting to the 27 events with full 4-carrier coverage (dropping "
            f"the thin early years) gives **{R['air_full_diff_bps']:+.2f} bps** at "
            f"t = **{R['air_full_t']:.2f}** — same story, not a coverage artifact. Most "
            "crashes in the table are foreign carriers with no direct fundamental link "
            "to AAL/DAL/UAL/LUV, which is itself an honest limit on what this specific "
            "basket could ever detect."
        ),
        md(
            "### 4d · The timer — an honest \"buy the dip\" cost sweep\n\n"
            "*Ethical caveat, stated plainly: this is a clinical test of whether a "
            "pattern in public market data is tradable, not encouragement to trade "
            "around tragedy.* Enter SPY at the crash-day close (zero look-ahead by "
            "construction — the crash predates that session's close), hold `h` "
            "sessions, one round trip of one-way costs charged twice, vs. the "
            "unconditional `h`-day SPY baseline over the same sample."
        ),
        code(
            "if HAVE_REAL:\n"
            "    holds = [1, 3, 5, 10]\n"
            "    gross, net5, net10, ts, base = [], [], [], [], []\n"
            "    for h in holds:\n"
            "        lg = st.buy_the_dip(SPY, EVENTS['date'], hold=h, cost_bps=0.0)\n"
            "        g = st.summarize_dip(lg, col='ret_gross')\n"
            "        ln5 = st.buy_the_dip(SPY, EVENTS['date'], hold=h, cost_bps=5.0)\n"
            "        n5 = st.summarize_dip(ln5, col='ret_net')\n"
            "        ln10 = st.buy_the_dip(SPY, EVENTS['date'], hold=h, cost_bps=10.0)\n"
            "        n10 = st.summarize_dip(ln10, col='ret_net')\n"
            "        gross.append(g['mean_bps']); net5.append(n5['mean_bps'])\n"
            "        net10.append(n10['mean_bps']); ts.append(n5['t'])\n"
            "        base.append(float((SPY.shift(-h) / SPY - 1.0).mean() * 1e4))\n"
            "else:\n"
            "    holds = sorted(R['timer'])\n"
            "    gross = [R['timer'][h][0] for h in holds]\n"
            "    net5 = [R['timer'][h][1] for h in holds]\n"
            "    ts = [R['timer'][h][2] for h in holds]\n"
            "    base = [R['timer'][h][3] for h in holds]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "x = np.arange(len(holds)); w = 0.27\n"
            "ax.bar(x - w, gross, width=w, color=GREY, label='gross')\n"
            "ax.bar(x, net5, width=w, color=AMBER, label='net (5 bps)')\n"
            "ax.bar(x + w, base, width=w, color=GREEN, label='unconditional baseline')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in holds])\n"
            "ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('The dip-buy trade never beats plain buy-and-hold, at any horizon')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('gross (bps):', dict(zip(holds, [round(v,1) for v in gross])))\n"
            "print('net 5bps (bps):', dict(zip(holds, [round(v,1) for v in net5])))\n"
            "print('t (net 5bps):', dict(zip(holds, [round(t,2) for t in ts])))\n"
            "print('unconditional baseline (bps):', dict(zip(holds, [round(v,1) for v in base])))"
        ),
        md(
            f"> 💡 In plain words: at 5 bps costs the 5-day hold nets "
            f"**{R['timer'][5][1]:+.2f} bps** (t = {R['timer'][5][2]:.2f}) against a "
            f"**{R['timer'][5][3]:+.2f} bps** unconditional baseline — buying the dip "
            "loses to doing nothing special, at every horizon we tried, though never at "
            f"a statistically certifiable margin. At 10 bps costs the 5-day hold falls "
            f"to **{R['timer_net10_5d']:+.2f} bps** (t = {R['timer_net10_5d_t']:.2f}). "
            "H₄ is not supported. There is no edge here to weigh against the study's "
            "ethical caveat — the honest number and the comfortable number coincide."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic random-walk tape, 36 scheduled pseudo-crash dates, TUNABLE "
            "planted day-0 dip that reverts over the next 5 sessions. The null (dip=0) "
            "is checked over **20 seeds** — never a single stream."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    close, ev = data.synthetic_world(dip=0.0, seed=707 + s_)\n"
            "    null_ts.append(st.synthetic_detect(close, ev)['t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "close, ev = data.synthetic_world(dip=-0.02, seed=707)\n"
            "planted_t = st.synthetic_detect(close, ev)['t']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (dip=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=RED, s=90, zorder=5,\n"
            "           label='planted dip = -2.0%')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('one-sample t (crash-day dip)')\n"
            "ax.set_title('Control: no null fires; a planted dip lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(np.abs(null_ts) >= 2).sum()}/20 seeds  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and **never** "
            f"crosses the bar; a planted 2% crash-day dip reads t = "
            f"{R['syn_planted_t']:.2f}. The machinery is unbiased — the real-tape "
            "t ≈ 0.02 is the genuine, honest reading, not a detector that's asleep. "
            "*(A faithful-engine / power check only — never cited in support of the "
            "real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — crash-day abnormal SPY return "
            f"**{R['day0_mean_bps']:+.2f} bps**, one-sample t = **{R['day0_t']:.2f}**, "
            f"hit rate {R['hit_pct']:.1f}% (Wilson [{R['wilson'][0]:.1f}%, "
            f"{R['wilson'][1]:.1f}%]), placebo p = **{R['placebo_p']:.3f}**. The airline "
            f"basket shows no extra drop (paired diff {R['air_diff_bps']:+.2f} bps, "
            f"t = {R['air_t']:.2f}, wrong sign), robust to a full-coverage-only "
            "subsample. The one nominally interesting offset (+2, t = "
            f"{R['event'][2][2]:.2f}) is a look-elsewhere artifact among 7 offsets "
            "tested.\n"
            f"- **Tradability `MIRAGE`** — buy-the-dip underperforms plain buy-and-hold "
            "at every horizon (1/3/5/10 days), gross and net, worst case t = "
            f"{R['timer_net10_5d_t']:.2f} at 10 bps costs. No edge exists to charge "
            "costs against.\n"
            "- **\"Does the market flinch at a plane crash?\" `BUSTED`** — on the "
            "modern, tradable version of the test. This does not overturn Kaplanski & "
            "Levy's original 1950–2005 finding on a broader accident register; it says "
            "a contemporary 36-event replication on SPY and a 4-carrier US airline "
            "basket does not detect it."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Power is the honest limitation.** n=36 independent events is a small "
            "sample for a small hypothesized effect; Kaplanski & Levy's original study "
            "used a much longer sample and a broader accident register. A pre-registered "
            "extension of this table back further (with the same 9/11/MH17 exclusion "
            "discipline) is the natural next step, not a change of method.\n"
            "- **A narrower sentiment proxy** — options skew, a news-sentiment index, "
            "retail order-flow imbalance — might catch a mood shock a blunt daily "
            "close-to-close S&P return cannot.\n"
            "- **Dedup map:** [313-geopolitical-shock](../../313-geopolitical-shock/) "
            "(wars/attacks, including the excluded 9/11 and MH17), "
            "[300-sports-sentiment](../../300-sports-sentiment/) (the Edmans loss "
            "effect) and [279-geomagnetic-storms](../../279-geomagnetic-storms/) (a "
            "non-newsworthy mood channel) — same research question, different "
            "triggers, all tested on this desk with the same honesty rails.\n\n"
            "*The reproducible core is offline and deterministic; frozen numbers live "
            "in [`docs/results.md`](../docs/results.md), sources in "
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
