"""Generate the two narrative notebooks for Study 736 (Sportsbook-Playoffs).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached
DKNG / PENN / CZR / MGM / RSI / BETZ / SPY tapes under ../_cache/ and otherwise quote
the frozen headline numbers in ``R`` (mirroring docs/results.md). The synthetic
positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance total-return
# DKNG 2020-04-24 -> 2026-06-30 + PENN/CZR/MGM/RSI/BETZ/SPY; 12 hardcoded betting-season
# starts 2021-01-09 -> 2026-03-19). RUN_UP = 10 sessions.
R = dict(
    n=12, cal_lo="2021-01-09", cal_hi="2026-03-19", run_up=10,
    run_up_pct=-1.93, run_up_t=-0.55,
    hits=5, hit_n=12, hit_pct=41.7, wilson=(19.3, 68.0),
    boot=(-8.37, 4.63),
    placebo_mean_pct=-0.038, placebo_sd_pct=3.59, placebo_p=0.698, placebo_draws=20000,
    nfl=(-0.76, -0.14, 6, 3), ncaa=(-3.10, -0.63, 6, 2), welch=0.32,
    madj_mean=-1.28, madj_t=-0.41,
    basket_mean=-3.24, basket_t=-1.19, basket_hits=3, basket_wilson=(8.9, 53.2),
    betz_mean=-1.24, betz_t=-0.74, betz_hits=4,
    post_mean=-2.80, post_t=-1.03,
    # event window offset -> (mean_bps, car_bps, t)
    event={-10: (-171.7, -171.7, -1.54), -9: (-193.8, -365.5, -2.76),
           -8: (-207.4, -572.9, -1.45), -7: (156.7, -416.2, 1.41),
           -6: (91.4, -324.8, 1.14), -5: (-73.3, -398.1, -0.88),
           -4: (-8.9, -407.0, -0.07), -3: (-52.1, -459.1, -0.61),
           -2: (91.5, -367.7, 0.82), -1: (174.2, -193.4, 1.19),
           0: (127.7, -65.8, 1.54), 1: (-77.9, -143.7, -0.93),
           2: (-61.1, -204.8, -0.61), 3: (34.8, -169.9, 0.32),
           4: (-245.5, -415.5, -2.09), 5: (-58.2, -473.7, -0.72)},
    # timer: run_up -> (gross_bps, net5_bps, net15_bps, t_net15, win_pct)
    timer={5: (245.9, 235.9, 215.9, 0.87, 50), 10: (52.2, 42.2, 22.2, 0.07, 42),
           20: (-178.2, -188.2, -208.2, -0.54, 58)},
    # synthetic control
    syn_null_mean=0.14, syn_null_sd=0.63, syn_null_fire=0,
    syn_planted_mean=14.2, syn_planted_t=4.37,
    fp_dkng="72bd1b0f5d55", fp_penn="e888bef384fc", fp_czr="48824d07b05a",
    fp_mgm="9ca308e4bd97", fp_rsi="793daf5af45c", fp_betz="5438abdf1003",
    fp_spy="c33b73c750b6",
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Rally%3F: Busted](https://img.shields.io/badge/Rally%3F-Busted-8b949e?style=flat-square)\n\n"
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

from sportsbook_playoffs import data, strategy as st

EVENTS = data.event_table()
RUN_UP = 10
HAVE_REAL = data.have_real()
if HAVE_REAL:
    CLOSES = data.load_real()
    DKNG = CLOSES["DKNG"]
    RET = st.daily_returns(DKNG)
    AR = st.abnormal_returns(RET)
    BENCH_RET = st.daily_returns(CLOSES["SPY"])
    AR_MADJ = st.abnormal_returns(st.market_adjusted_returns(RET, BENCH_RET))
    BASKET, COVERAGE = st.basket_returns(CLOSES, data.BASKET_TICKERS)
    AR_BASKET = st.abnormal_returns(BASKET)
    AR_BETZ = st.abnormal_returns(st.daily_returns(CLOSES["BETZ"]))
else:
    CLOSES = DKNG = RET = AR = AR_MADJ = BASKET = COVERAGE = AR_BASKET = AR_BETZ = None
print("real cache present:", HAVE_REAL, "| betting seasons in table:", len(EVENTS))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do sportsbook stocks rally *into* the big betting seasons? 🎰\n"
            "### The \"buy DraftKings before the NFL playoffs\" trade — a tidy story that "
            "the tape doesn't tell\n\n"
            + BADGES +
            "Every January and every March, the same idea makes the rounds: the NFL "
            "playoffs and March Madness are about to unleash a *wall* of betting handle — "
            "record parlays, new-customer promos, deposits pouring in — so the stocks that "
            "run the sportsbooks (DraftKings and friends) should rally *ahead* of the "
            "games, as the market prices the coming boom.\n\n"
            "And the boom is real: Americans really do bet far more in the NFL/March "
            "window than in the summer. So the premise isn't crazy. The question is the "
            "*second* step — does that known, calendar-fixed spike in betting actually "
            "lift the **stocks** in the weeks before tip-off? Or has an efficient market "
            "already priced a schedule everyone can read a year in advance?\n\n"
            "That's what we test on **DraftKings' entire public life as an operating "
            "company (2020→2026)** and **12 flagship betting seasons** — 6 NFL Wild-Card "
            "weekends and 6 March-Madness Round-of-64s.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo, the "
            "beta-adjusted and basket-wide checks? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice** — a clinical statistical test of public "
            "market-price data. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do the stocks rally into the season? | **No — the opposite, if anything.** "
            f"DraftKings' average move over the ~2 weeks *before* the first game is "
            f"**{R['run_up_pct']:+.2f}%** — negative, and statistically nothing (a random "
            f"pair of weeks beats it about **{R['placebo_p']*100:.0f}%** of the time). |\n"
            "| Is it just an NFL thing, or a March thing? | **Neither.** Both seasons "
            f"drift *down* into the games on their own (NFL {R['nfl'][0]:+.2f}%, March "
            f"Madness {R['ncaa'][0]:+.2f}%). |\n"
            "| Does the whole betting sector do better? | **No.** The 5-name basket "
            f"({R['basket_mean']:+.2f}%) and the BETZ pure-play ETF ({R['betz_mean']:+.2f}%) "
            "drift down too. |\n"
            "| Could you trade it? | **No edge to trade.** \"Buy the run-up\" produces no "
            "cost-surviving profit at any horizon we tried. |\n\n"
            "> The *betting* really does spike on schedule. The tradable *stock rally* "
            "the folklore bolts onto it does not."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"NFL playoffs and March Madness are the Super Bowl and the World Cup of "
            "the betting business — the two biggest handle events of the year. Anticipation "
            "of that flood of wagering should be priced into DraftKings and the sportsbook "
            "names in the weeks *before* the games, so you buy the run-up and ride the "
            "hype into tip-off.\"*\n\n"
            "This is a fixture of betting-industry trade press and broker previews every "
            "autumn and January. It has a real, checkable premise — betting *activity* is "
            "strongly seasonal — and a plausible-sounding mechanism (markets price ahead). "
            "We take the strongest version: not \"the stocks are volatile around the "
            "games\" but \"they systematically **rally into** them.\""
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — and is the premise even true?\n\n"
            "First, the honest half of the story: US sports-betting **handle** really is "
            "seasonal. Here's the shape the industry reports every year (an illustrative, "
            "clearly-labelled approximation of American Gaming Association / state-regulator "
            "monthly handle — *not* market data, never traded, just to show the premise is "
            "real)."
        ),
        code(
            "months = list(range(1, 13))\n"
            "seas = [data.HANDLE_SEASONALITY[m] for m in months]\n"
            "names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.2))\n"
            "cols = [RED if m in (1, 3) else GREY for m in months]\n"
            "ax.bar(names, seas, color=cols, width=.66)\n"
            "ax.axhline(1.0, c='k', lw=.8, ls='--')\n"
            "ax.set_ylabel('handle vs annual average (proxy, mean=1.0)')\n"
            "ax.set_title('The premise is real: betting handle peaks around the NFL playoffs (Jan) '\n"
            "             'and March Madness (Mar)')\n"
            "ax.annotate('NFL\\nplayoffs', (0, seas[0]), ha='center', va='bottom', fontsize=9, color=RED)\n"
            "ax.annotate('March\\nMadness', (2, seas[2]), ha='center', va='bottom', fontsize=9, color=RED)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('LABELLED PROXY (AGA / state-regulator seasonal shape) — illustrative, not traded')"
        ),
        md(
            "So if the stocks *did* price this in ahead of time, it would be a clean little "
            "seasonal edge: a known calendar, a known catalyst, buy two weeks early. That's "
            "worth checking — and it's exactly the kind of \"obvious\" trade that an "
            "efficient market tends to have already eaten.\n\n"
            "So we ask three plain things: does DraftKings actually rally into the season, "
            "does the rest of the betting sector, and could you have made money buying the "
            "run-up?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The calendar.** **{R['n']}** flagship betting-season starts — 6 NFL "
            f"Wild-Card weekends and 6 March-Madness Round-of-64s, {R['cal_lo']} to "
            f"{R['cal_hi']}. That's every one since DraftKings became a real public "
            "company (its SPAC merger closed April 2020).\n"
            f"- **The run-up.** DraftKings' return over the **{R['run_up']} trading days "
            "just before** the first game, measured against its *own* average (so a "
            "\"rally\" means faster than usual, not just \"the stock went up\").\n"
            "- **The luck check.** Draw 12 random dates instead, 20,000 times — how often "
            "does a random pair of weeks produce a run-up this big *or bigger*?\n"
            "- **The sector check.** Same test on a 5-name betting basket "
            "(DraftKings/Penn/Caesars/MGM/Rush Street) and on BETZ, the pure-play "
            "sports-betting ETF.\n"
            "- **The trade check.** Buy the run-up, sell before the first game, pay costs, "
            "see if anything's left.\n\n"
            "**No look-ahead, and none needed:** the NFL and NCAA schedules are public "
            "months ahead, so \"buy N days before the first game\" uses only dates you'd "
            "have known a year in advance."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** DraftKings' average run-up into a betting season, "
            "vs. what a random stretch of the same length looks like."
        ),
        code(
            "if HAVE_REAL:\n"
            "    r = st.run_up_stats(AR, EVENTS['date'], run_up=RUN_UP)\n"
            "    run_up_pct = r['mean'] * 100\n"
            "else:\n"
            "    run_up_pct = R['run_up_pct']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['into the season\\n(DKNG, n=12)', 'random 2 weeks\\n(placebo mean)'],\n"
            "       [run_up_pct, R['placebo_mean_pct']], color=[RED, GREY], width=.55)\n"
            "for i, v in enumerate([run_up_pct, R['placebo_mean_pct']]):\n"
            "    ax.annotate(f'{v:+.2f}%', (i, v), ha='center',\n"
            "                va='top' if v < 0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('average DKNG run-up (%)')\n"
            "ax.set_title('The \"rally\" bar points the wrong way — and it is not taller than random')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'run-up into the season {run_up_pct:+.2f}% vs random {R[\"placebo_mean_pct\"]:+.2f}%')"
        ),
        md(
            f"That's the whole story in one chart: **{R['run_up_pct']:+.2f}%** on the "
            f"average run-up — *negative*, the opposite of a rally — and barely different "
            f"from **{R['placebo_mean_pct']:+.2f}%** on a random stretch. DraftKings rallied "
            f"into only **{R['hits']}/{R['hit_n']} = {R['hit_pct']:.0f}%** of these seasons "
            "— a coin flip. A random calendar beats this about 70% of the time.\n\n"
            "**Maybe it's really an NFL thing, or a March thing?** Let's split them."
        ),
        code(
            "if HAVE_REAL:\n"
            "    nfl = st.run_up_stats(AR, EVENTS[EVENTS['family']=='NFL']['date'], run_up=RUN_UP)\n"
            "    ncaa = st.run_up_stats(AR, EVENTS[EVENTS['family']=='NCAA']['date'], run_up=RUN_UP)\n"
            "    vals = [nfl['mean']*100, ncaa['mean']*100]\n"
            "else:\n"
            "    vals = [R['nfl'][0], R['ncaa'][0]]\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['NFL playoffs\\n(n=6)', 'March Madness\\n(n=6)'], vals,\n"
            "       color=[RED, RED], width=.5)\n"
            "for i, v in enumerate(vals):\n"
            "    ax.annotate(f'{v:+.2f}%', (i, v), ha='center', va='top' if v < 0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('average DKNG run-up (%)')\n"
            "ax.set_title('Neither season rallies on its own — both drift down into the games')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'NFL {vals[0]:+.2f}%  vs  March Madness {vals[1]:+.2f}%')"
        ),
        md(
            "Nope — both are negative and both are noise. There's no \"it's really the NFL\" "
            "rescue hiding inside the average.\n\n"
            "**Is it just DraftKings, or the whole sector?** Let's widen to a 5-name "
            "betting basket and the BETZ pure-play ETF."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rb = st.run_up_stats(AR_BASKET, EVENTS['date'], run_up=RUN_UP)\n"
            "    re = st.run_up_stats(AR_BETZ, EVENTS['date'], run_up=RUN_UP)\n"
            "    vals = [run_up_pct, rb['mean']*100, re['mean']*100]\n"
            "else:\n"
            "    vals = [R['run_up_pct'], R['basket_mean'], R['betz_mean']]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(['DKNG', '5-name basket\\n(DKNG/PENN/CZR/MGM/RSI)', 'BETZ ETF'], vals,\n"
            "       color=[RED, RED, RED], width=.6)\n"
            "for i, v in enumerate(vals):\n"
            "    ax.annotate(f'{v:+.2f}%', (i, v), ha='center', va='top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('average run-up into the season (%)')\n"
            "ax.set_title('The whole betting complex drifts down into the games, not up')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'DKNG {vals[0]:+.2f}%  basket {vals[1]:+.2f}%  BETZ {vals[2]:+.2f}%')"
        ),
        md(
            "Widening the net makes it slightly *worse*, not better — the diversified "
            "basket drifts down a touch more than DraftKings alone. Whatever the folklore "
            "is describing, it isn't in the tape.\n\n"
            "**Finally, the trade.** Could you have made money buying the run-up?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    holds = [5, 10, 20]\n"
            "    net = []\n"
            "    for h in holds:\n"
            "        lg = st.run_up_timer(DKNG, EVENTS['date'], run_up=h, cost_bps=15.0)\n"
            "        net.append(st.summarize_timer(lg, 'ret_net')['mean_bps'])\n"
            "else:\n"
            "    holds = [5, 10, 20]\n"
            "    net = [R['timer'][h][2] for h in holds]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar([f'{h}-day run-up' for h in holds], net,\n"
            "       color=[GREY if v <= 0 else AMBER for v in net], width=.55)\n"
            "for i, v in enumerate(net):\n"
            "    ax.annotate(f'{v:+.0f} bps', (i, v), ha='center', va='bottom' if v > 0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('net return, 15 bps costs (bps)')\n"
            "ax.set_title('Buying the run-up: no reliable edge, and it flips sign with the window')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net (15 bps) by run-up:', dict(zip(holds, [round(v) for v in net])))"
        ),
        md(
            f"The only positive bar (the 5-day run-up, **{R['timer'][5][2]:+.0f} bps** net) "
            f"comes with a *t* of just **{R['timer'][5][3]}** and a **{R['timer'][5][4]}%** "
            "win rate over 12 events — a coin flip that happened to land heads — and it "
            "**flips to a loss** as you lengthen the window. Stretch it to 20 days and "
            f"you're down **{abs(R['timer'][20][2]):.0f} bps**. There's no stable edge to "
            "trade, and nothing for costs to eat because there was nothing there to begin "
            "with."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** DraftKings' run-up into a betting season is "
            f"**{R['run_up_pct']:+.2f}%** (*wrong sign*), a coin-flip hit rate, and a "
            "random calendar reproduces it ~70% of the time. The basket, the ETF and "
            "the market-adjusted version all agree; neither season rescues it.\n"
            "- **Tradability — Mirage.** No cost-surviving edge at any horizon; the one "
            "positive point estimate is noise that flips sign as the window widens.\n"
            "- **\"Do betting stocks rally into the season?\" — Busted.** The handle "
            "seasonality is real; the tradable stock-price rally isn't."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **A schedule everyone can read a year ahead is the least likely thing to "
            "surprise a market.** The cleanest reading of this null: the betting-season "
            "calendar is *maximally* public, so if there were an easy pre-season rally it "
            "would have been arbitraged into the earlier weeks (or into the *prior* "
            "off-season) long ago. Efficient-markets 1, folklore 0.\n"
            "- **Where a real version might live:** the *surprise* component — a season "
            "that shatters handle records vs. a merely-typical one — measured against "
            "*analyst expectations* rather than the calendar, or the earnings prints that "
            "actually report the handle a month or two later. That's a fundamentals event "
            "study, not a calendar one, and a different study.\n"
            "- **Short history is the honest limit.** DraftKings has only ~6 years and 12 "
            "flagship seasons as a public company; a small, real seasonal tilt could be "
            "invisible at n=12. The honest conclusion is \"not detectable, wrong sign, at "
            "this size on this tape\", not \"provably impossible forever.\"\n"
            "- **Sibling studies:** [707-plane-crash-effect](../../707-plane-crash-effect/) "
            "and [708-eurovision-effect](../../708-eurovision-effect/) (the same event-study "
            "machinery on other folklore) and "
            "[300-sports-sentiment](../../300-sports-sentiment/) (sports *results* moving "
            "the *home* market).\n\n"
            "*Think the surprise-vs-expectation version has a pulse? Show it — out of "
            "sample, after costs — then we'll talk.*"
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
            "# The Sportsbook-Playoffs trade — a quantitative teardown 🔬\n"
            "### The run-up event study · a right-tail random-calendar placebo · the "
            "NFL-vs-NCAA split · beta-adjusted and basket/ETF cross-checks · the [−10..+5] "
            "anatomy and its look-elsewhere caveat · a costed run-up timer · a 20-seed "
            "synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious]"
            "(01_for_the_curious.ipynb). The claim — betting stocks rally *into* the NFL "
            "playoffs and March Madness — rests on a genuinely real premise (handle "
            "seasonality) and a strong-form-efficiency-tempting mechanism (price the "
            "known catalyst ahead). The job here is to measure the pre-season run-up "
            "honestly on DraftKings' full public tape, then ask the only question that "
            "pays: *is any of it real, and if so, tradable?*\n\n"
            "> ⚠️ **Data note.** DKNG + PENN/CZR/MGM/RSI + BETZ + SPY total-return closes "
            "(2020→2026), yfinance, cached; **DKNG floored at its 2020-04-24 SPAC-merger "
            "close** (earlier tape = DEAC cash shell). **12 hardcoded betting-season "
            "starts** (6 NFL + 6 NCAA, 2021→2026). **Survivorship named** on the basket "
            "(current survivors of the 2021-22 shakeout — a bias that points *for* the "
            "rally). Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprints `" + R["fp_dkng"] +
            "` DKNG / `" + R["fp_betz"] + "` BETZ / `" + R["fp_spy"] + "` SPY).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | run-up CAR **{R['run_up_pct']:+.2f}%** (wrong sign), "
            f"one-sample **t = {R['run_up_t']:+.2f}**, hit rate {R['hit_pct']:.1f}% "
            f"(Wilson [{R['wilson'][0]:.1f}%, {R['wilson'][1]:.1f}%]), placebo (right-tail) "
            f"**p = {R['placebo_p']:.3f}** |\n"
            f"| **Tradability** | `MIRAGE` | run-up timer nets **{R['timer'][10][2]:+.1f} "
            f"bps** at 10d (t = {R['timer'][10][3]}); the only positive leg (5d, "
            f"{R['timer'][5][2]:+.0f} bps) is t = {R['timer'][5][3]}, 50% win, and flips "
            "sign by 20d |\n"
            f"| **Rally into the season?** | `BUSTED` | basket **{R['basket_t']:+.2f}**, "
            f"BETZ **{R['betz_t']:+.2f}**, market-adj **{R['madj_t']:+.2f}** — every "
            "cross-check agrees the pre-season drift is *down*, not up |\n\n"
            "> 💡 In plain words: the betting-handle seasonality is real, but the "
            "stock-price rally folklore attaches to it isn't — every way we slice the "
            "run-up, it comes back negative and insignificant."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t$ be DKNG's daily total-return and $a_t = r_t - \\bar{r}$ its abnormal "
            "return under a constant-mean market model (Brown & Warner 1985). For each "
            "betting season $i$ with public first-game date $\\tau_i$, snapped to the first "
            "session on/after it, define the **run-up** "
            "$U_i = \\sum_{k=1}^{K} a_{\\tau_i - k}$ over the $K=10$ sessions before the "
            "games. The claims:\n\n"
            "- **H₁ (pre-season rally).** $E[U_i] > 0$, large and systematic across events "
            "— the stocks price the coming handle ahead of tip-off.\n"
            "- **H₂ (sector-wide).** The same holds for a betting basket and the BETZ ETF, "
            "not just DKNG.\n"
            "- **H₃ (not just beta).** The run-up survives subtracting the market's return "
            "over the same window.\n"
            "- **H₄ (capture).** A buy-the-run-up overlay beats zero net of costs.\n\n"
            "We find **H₁ rejected on sign** (run-up "
            f"{R['run_up_pct']:+.2f}%, t = {R['run_up_t']:+.2f} — *negative*), **H₂ "
            f"rejected** (basket t = {R['basket_t']:+.2f}, BETZ t = {R['betz_t']:+.2f}, "
            f"both negative), **H₃ moot** (market-adj t = {R['madj_t']:+.2f}, still "
            "negative), **H₄ not supported** (no cost-surviving edge at any horizon)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Betting seasons are **independent, non-overlapping calendar dates** (the "
            "closest pair — a January NFL and the following March — is ~9 weeks apart, and "
            "the run-up windows never overlap), so the planned primary is a **one-sample "
            "t-test** across the 12 per-event run-up figures — the correct unit of "
            "analysis is \"one run-up per season\", not a daily panel, so no HAC "
            "correction is needed the way a daily regression would require. The hit rate "
            "carries a **Wilson interval**; the placebo draws 12 random dates **20,000 "
            "times (20 seeds × 1,000)** and is read on the **right tail** because the "
            "claim predicts a *positive* run-up; the [−10..+5] anatomy is read as a "
            "**16-offset multiple-comparison** exercise, not 16 independent shots. Because "
            "the season-start dates are **public months ahead**, the entry/exit sessions "
            "are known ex-ante — a **calendar-known rule with zero look-ahead and no "
            "execution lag**, applied once and documented once."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Calendar.** {R['n']} flagship betting-season starts {R['cal_lo']} → "
            f"{R['cal_hi']}, hardcoded (6 NFL Wild-Card weekends + 6 March-Madness "
            "Round-of-64s). All 12 sit on the DKNG tape.\n"
            "- **Tape.** DKNG (from its 2020-04-24 merger close) + PENN/CZR/MGM/RSI + BETZ "
            "+ SPY total-return closes, → 2026-06-30 (as-of, last complete month).\n"
            "- **Headline.** Run-up CAR over the 10 sessions before the first game, "
            "one-sample t + Wilson hit rate + 20-seed right-tail placebo + event-bootstrap "
            "CI.\n"
            "- **Splits.** NFL vs NCAA one-sample t's and a Welch t of the difference.\n"
            "- **Cross-checks.** The 5-name basket, the BETZ ETF, and a beta≈1 "
            "market-adjusted (DKNG − SPY) run-up.\n"
            "- **Anatomy.** Event window [−10..+5], per-offset mean + t, read with the "
            "multiple-comparison caveat; a post-event [0..+5] sell-the-news test.\n"
            "- **Execution (timer).** Buy DKNG N sessions before the first game, sell the "
            "session before it (zero look-ahead — schedule public); 2 × one-way cost × NAV "
            "per event; long-only.\n"
            "- **Control.** Synthetic random-walk tape, planted pre-event run-up; the null "
            "must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline run-up and its right-tail placebo\n\n"
            "One-sample t on the 10-day run-up CAR, Wilson hit rate, and the "
            "random-calendar null. In the notebook we run a lighter placebo (4 seeds × 500 "
            "draws) and quote the canonical 20,000-draw p from `results.md`."
        ),
        code(
            "if HAVE_REAL:\n"
            "    r = st.run_up_stats(AR, EVENTS['date'], run_up=RUN_UP)\n"
            "    wl, wh = st.wilson_interval(r['hits'], r['n'])\n"
            "    print(f\"run-up CAR {r['mean']*100:+.2f}%  one-sample t = {r['t']:+.3f}  (n={r['n']})\")\n"
            "    print(f\"hit rate: {r['hits']}/{r['n']} = {r['hits']/r['n']*100:.1f}% positive  \"\n"
            "          f\"(Wilson [{wl*100:.1f}%, {wh*100:.1f}%])\")\n"
            "    lo, hi = st.bootstrap_ci(r['per_event'])\n"
            "    print(f\"event-bootstrap 95% CI on the mean run-up: [{lo*100:+.2f}%, {hi*100:+.2f}%]\")\n"
            "    pl = np.concatenate([st.placebo_distribution(AR, r['n'], run_up=RUN_UP,\n"
            "                          n_draws=500, seed=736 + s_) for s_ in range(4)])\n"
            "    obs, draws = r['mean'], pl\n"
            "else:\n"
            "    obs = R['run_up_pct'] / 100\n"
            "    rng = np.random.default_rng(736)\n"
            "    draws = rng.normal(R['placebo_mean_pct'] / 100, R['placebo_sd_pct'] / 100, 2000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws * 100, bins=50, color=GREY, alpha=.85,\n"
            "        label='null: random 12-date calendars (light in-notebook run)')\n"
            "ax.axvline(obs * 100, c=RED, lw=2.5, label=f'observed run-up {obs*100:+.2f}%')\n"
            "ax.axvline(0, c='k', lw=.8, ls='--')\n"
            "ax.set_xlabel('mean run-up of a random 12-date calendar (%)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"Left of centre, not right: canonical right-tail p = {R['placebo_p']:.3f} \"\n"
            "             '(20 seeds x 1,000 draws)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"canonical placebo (results.md): mean {R['placebo_mean_pct']:+.3f}%, \"\n"
            "      f\"sd {R['placebo_sd_pct']:.2f}%, right-tail p = {R['placebo_p']:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: the observed run-up **{R['run_up_pct']:+.2f}%** sits "
            f"*below* the centre of the null cloud ({R['placebo_mean_pct']:+.3f} ± "
            f"{R['placebo_sd_pct']:.2f}%). The right-tail **p = {R['placebo_p']:.3f}** means "
            "a random calendar produces a run-up this big or bigger ~70% of the time. With "
            f"one-sample t = **{R['run_up_t']:.2f}** and a bootstrap CI "
            f"[{R['boot'][0]:+.2f}%, {R['boot'][1]:+.2f}%] straddling zero, H₁ is rejected "
            "— and on the *wrong side*."
        ),
        md(
            "### 4b · The season split and the sector/beta cross-checks\n\n"
            "If a rally hides in one season, or in the wider sector, or is just market "
            "beta, this is where it shows up. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    nfl = st.run_up_stats(AR, EVENTS[EVENTS['family']=='NFL']['date'], run_up=RUN_UP)\n"
            "    ncaa = st.run_up_stats(AR, EVENTS[EVENTS['family']=='NCAA']['date'], run_up=RUN_UP)\n"
            "    rb = st.run_up_stats(AR_BASKET, EVENTS['date'], run_up=RUN_UP)\n"
            "    re = st.run_up_stats(AR_BETZ, EVENTS['date'], run_up=RUN_UP)\n"
            "    rm = st.run_up_stats(AR_MADJ, EVENTS['date'], run_up=RUN_UP)\n"
            "    welch = st.welch_t(nfl['per_event'], ncaa['per_event'])\n"
            "    labels = ['NFL', 'March\\nMadness', '5-name\\nbasket', 'BETZ\\nETF', 'DKNG\\n-minus-SPY']\n"
            "    vals = [nfl['mean']*100, ncaa['mean']*100, rb['mean']*100, re['mean']*100, rm['mean']*100]\n"
            "    ts = [nfl['t'], ncaa['t'], rb['t'], re['t'], rm['t']]\n"
            "else:\n"
            "    welch = R['welch']\n"
            "    labels = ['NFL', 'March\\nMadness', '5-name\\nbasket', 'BETZ\\nETF', 'DKNG\\n-minus-SPY']\n"
            "    vals = [R['nfl'][0], R['ncaa'][0], R['basket_mean'], R['betz_mean'], R['madj_mean']]\n"
            "    ts = [R['nfl'][1], R['ncaa'][1], R['basket_t'], R['betz_t'], R['madj_t']]\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.5))\n"
            "ax.bar(labels, vals, color=[RED]*len(vals), width=.62)\n"
            "for i, (v, t) in enumerate(zip(vals, ts)):\n"
            "    ax.annotate(f'{v:+.2f}%\\n(t={t:+.2f})', (i, v), ha='center', va='top', fontsize=9)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean run-up (%)')\n"
            "ax.set_title('Every cut is negative and insignificant — no season, sector or beta rescue')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('run-ups (%):', dict(zip([l.replace(chr(10),' ') for l in labels], [round(v,2) for v in vals])))\n"
            "print(f'Welch t (NFL - NCAA) = {welch:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: NFL ({R['nfl'][0]:+.2f}%, t = {R['nfl'][1]:+.2f}) and "
            f"March Madness ({R['ncaa'][0]:+.2f}%, t = {R['ncaa'][1]:+.2f}) are the same "
            f"nothing (Welch t = {R['welch']:+.2f}); the 5-name basket "
            f"({R['basket_mean']:+.2f}%, t = {R['basket_t']:+.2f}) and BETZ "
            f"({R['betz_mean']:+.2f}%, t = {R['betz_t']:+.2f}) drift down too; and "
            f"subtracting the market (DKNG − SPY, {R['madj_mean']:+.2f}%, t = "
            f"{R['madj_t']:+.2f}) doesn't turn it positive. H₂ and H₃ both fail."
        ),
        md(
            "### 4c · Anatomy — the [−10..+5] window, read with the multiple-comparison caveat\n\n"
            "Per-offset means with each offset's own one-sample t. With 16 offsets tested, "
            "~0.8 crossing |t| ≥ 2 by pure chance at the 5% level is expected — exactly "
            "two do (offsets −9 and +4), and **both are negative**, pointing the wrong way "
            "for a rally and attached to no mechanism."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cp = st.car_path_stats(AR, EVENTS['date'], pre=RUN_UP, post=5)\n"
            "    ks = list(cp.index); ms = list(cp['mean_ar'] * 1e4); ts = list(cp['t'])\n"
            "    cars = list(cp['car'] * 1e4)\n"
            "    pe = st.post_event_stats(AR, EVENTS['date'], pre=RUN_UP, post=5)\n"
            "    post_m, post_t = pe['mean']*100, pe['t']\n"
            "else:\n"
            "    ks = sorted(R['event']); ms = [R['event'][k][0] for k in ks]\n"
            "    cars = [R['event'][k][1] for k in ks]; ts = [R['event'][k][2] for k in ks]\n"
            "    post_m, post_t = R['post_mean'], R['post_t']\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.2, 6.6), sharex=True,\n"
            "                             gridspec_kw={'height_ratios': [2, 1]})\n"
            "cols = [RED if k == 0 else GREY for k in ks]\n"
            "a1.bar([str(k) for k in ks], ms, color=cols, width=.7)\n"
            "a1.plot(range(len(ks)), cars, color=RED, lw=1.6, marker='o', ms=3, label='CAR (bps)')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('abnormal return / CAR (bps)')\n"
            "a1.axvline(RUN_UP - 0.5, c=GREY, ls=':', lw=1)\n"
            "a1.set_title('Event anatomy: the CAR drifts DOWN into the games (offset 0), no run-up hump')\n"
            "a1.legend(loc='lower left')\n"
            "a2.bar([str(k) for k in ks], ts,\n"
            "       color=[RED if abs(t) >= 2 else GREY for t in ts], width=.7)\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.axhline(-2, ls='--', c=RED, lw=1); a2.axhline(2, ls='--', c=RED, lw=1)\n"
            "a2.set_ylabel('one-sample t'); a2.set_xlabel('offset (sessions; 0 = first game)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'post-game [0..+5] sell-the-news test: {post_m:+.2f}% (t = {post_t:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: the cumulative abnormal return actually *falls* into the "
            f"first game (CAR ≈ {R['event'][-1][1]:.0f} bps at offset −1) — the opposite of "
            "a run-up hump. The two |t| ≥ 2 bars (offset −9 at "
            f"t = {R['event'][-9][2]:.2f}, offset +4 at t = {R['event'][4][2]:.2f}) are "
            "both *down*, mechanism-free, and exactly the ~1 spurious crossing you expect "
            "from 16 offsets. The post-game [0..+5] leg is "
            f"**{R['post_mean']:+.2f}%** (t = {R['post_t']:+.2f}) — no sell-the-news round "
            "trip either, because there was no run-up to sell."
        ),
        md(
            "### 4d · The timer — an honest \"buy the run-up\" cost sweep\n\n"
            "Buy DKNG at the close N sessions before the first game, sell at the close the "
            "session before it (zero look-ahead — the schedule is public months ahead), one "
            "round trip of one-way costs charged twice, across the 12 seasons. Gross, net "
            "at 5 and 15 bps."
        ),
        code(
            "if HAVE_REAL:\n"
            "    holds = [5, 10, 20]\n"
            "    gross, net5, net15, tt, win = [], [], [], [], []\n"
            "    for h in holds:\n"
            "        g = st.summarize_timer(st.run_up_timer(DKNG, EVENTS['date'], run_up=h, cost_bps=0.0), 'ret_gross')\n"
            "        n5 = st.summarize_timer(st.run_up_timer(DKNG, EVENTS['date'], run_up=h, cost_bps=5.0), 'ret_net')\n"
            "        n15 = st.summarize_timer(st.run_up_timer(DKNG, EVENTS['date'], run_up=h, cost_bps=15.0), 'ret_net')\n"
            "        gross.append(g['mean_bps']); net5.append(n5['mean_bps'])\n"
            "        net15.append(n15['mean_bps']); tt.append(n15['t']); win.append(g['win_rate']*100)\n"
            "else:\n"
            "    holds = [5, 10, 20]\n"
            "    gross = [R['timer'][h][0] for h in holds]\n"
            "    net5 = [R['timer'][h][1] for h in holds]\n"
            "    net15 = [R['timer'][h][2] for h in holds]\n"
            "    tt = [R['timer'][h][3] for h in holds]; win = [R['timer'][h][4] for h in holds]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "x = np.arange(len(holds)); w = 0.27\n"
            "ax.bar(x - w, gross, width=w, color=GREY, label='gross')\n"
            "ax.bar(x, net5, width=w, color=AMBER, label='net (5 bps)')\n"
            "ax.bar(x + w, net15, width=w, color=RED, label='net (15 bps)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}-day\\nrun-up' for h in holds])\n"
            "ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('No stable edge: the one positive leg is a 50%-win coin flip that flips sign by 20d')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('gross (bps):', dict(zip(holds, [round(v,1) for v in gross])))\n"
            "print('net 15bps (bps):', dict(zip(holds, [round(v,1) for v in net15])))\n"
            "print('t (net15):', dict(zip(holds, [round(t,2) for t in tt])))\n"
            "print('win rate (%):', dict(zip(holds, [round(w) for w in win])))"
        ),
        md(
            f"> 💡 In plain words: the 5-day run-up nets **{R['timer'][5][2]:+.1f} bps** at "
            f"15 bps costs — but on t = {R['timer'][5][3]}, a {R['timer'][5][4]}% win rate "
            "across 12 events, it's a coin flip that landed heads. Lengthen the window and "
            f"it evaporates: 10-day nets **{R['timer'][10][2]:+.1f} bps** (t = "
            f"{R['timer'][10][3]}), 20-day **{R['timer'][20][2]:+.1f} bps** (t = "
            f"{R['timer'][20][3]}). H₄ is not supported — there is no stable edge, so the "
            "cost question is moot."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic random-walk tape, 12 scheduled pseudo-season dates, TUNABLE planted "
            "run-up that accumulates over the 10 sessions before each event. The null "
            "(bump=0) is checked over **20 seeds** — never a single stream."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    close, ev = data.synthetic_world(bump=0.0, seed=736 + s_)\n"
            "    null_ts.append(st.synthetic_detect(close, ev)['t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "close, ev = data.synthetic_world(bump=0.15, seed=736)\n"
            "planted_t = st.synthetic_detect(close, ev)['t']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (bump=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=RED, s=90, zorder=5, label='planted run-up = +15%')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('one-sample t (run-up)')\n"
            "ax.set_title('Control: no null fires; a planted run-up lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(np.abs(null_ts) >= 2).sum()}/20 seeds  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and **never** "
            f"crosses the bar; a planted +15% run-up reads t = {R['syn_planted_t']:.2f}. "
            "The machinery is unbiased — the real-tape t ≈ −0.55 is the genuine, honest "
            "reading, not a detector asleep at the wheel. *(A faithful-engine / power "
            "check only — never cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — run-up CAR **{R['run_up_pct']:+.2f}%** (wrong sign), "
            f"one-sample t = **{R['run_up_t']:.2f}**, hit rate {R['hit_pct']:.1f}% (Wilson "
            f"[{R['wilson'][0]:.1f}%, {R['wilson'][1]:.1f}%]), right-tail placebo "
            f"p = **{R['placebo_p']:.3f}**. The basket (t = {R['basket_t']:.2f}), BETZ "
            f"(t = {R['betz_t']:.2f}) and market-adjusted (t = {R['madj_t']:.2f}) "
            "cross-checks all agree; neither season rescues it. The two |t| ≥ 2 offsets "
            "(−9, +4) are look-elsewhere artifacts among 16 offsets — and both negative.\n"
            f"- **Tradability `MIRAGE`** — the run-up timer nets "
            f"{R['timer'][10][2]:+.1f} bps at 10d (t = {R['timer'][10][3]}); the one "
            f"positive leg (5d, {R['timer'][5][2]:+.0f} bps) is t = {R['timer'][5][3]}, "
            "50% win, and flips sign by 20d. No stable edge to charge costs against.\n"
            "- **\"Do betting stocks rally into the season?\" `BUSTED`** — the handle "
            "seasonality is real (Beat 2's labelled proxy), but the tradable stock-price "
            "rally is not: on DraftKings' full public tape the stocks drift *down* into "
            "the games. A schedule everyone can read a year ahead is already in the price."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Efficiency is the honest explanation.** The betting calendar is maximally "
            "public, so an easy pre-season rally would have been arbitraged into earlier "
            "weeks (or the prior off-season) long ago. A null here is what an efficient "
            "market *should* deliver on a fully-anticipated catalyst.\n"
            "- **The surprise version is a different study.** Test the *unexpected* "
            "component — a record-shattering handle vs. a merely-typical season, measured "
            "against analyst expectations, or the earnings prints that report the handle "
            "weeks later — a fundamentals event study, not a calendar one.\n"
            "- **Power is a real limit.** n=12 seasons over ~6 years of DKNG history is "
            "small; a tiny real seasonal tilt could be invisible. The honest conclusion is "
            "\"not detectable, wrong sign, at this size on this tape.\"\n"
            "- **Dedup map:** [707-plane-crash-effect](../../707-plane-crash-effect/) and "
            "[708-eurovision-effect](../../708-eurovision-effect/) share this exact "
            "event-study machinery on other folklore; "
            "[300-sports-sentiment](../../300-sports-sentiment/) tests sports *results* on "
            "the *home* market. None test whether the betting stocks themselves rally into "
            "the betting calendar — that scheduled-anticipation axis is this study's own.\n\n"
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
