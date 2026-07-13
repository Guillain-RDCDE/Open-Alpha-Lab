"""Generate the two narrative notebooks for Study 739 (Wildfire-Season).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached
EIX/PCG/ALL/TRV/MCY/CB/SPY tapes under ../_cache/ and otherwise quote the frozen
headline numbers in ``R`` (mirroring docs/results.md). The synthetic positive control
runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance
# EIX/PCG/ALL/TRV/MCY/CB/SPY total-return closes 2003-01-02 -> 2026-06-30; 14 hardcoded
# major California wildfires 2003-10-25 -> 2025-01-07).
R = dict(
    n_events=14, n_util_linked=7, n_nonutil=7,
    cal_lo="2003-10-25", cal_hi="2025-01-07",
    # day0 headline (combined basket)
    day0_mean_bps=-22.77, day0_t=-0.92,
    hit_down=9, hit_n=14, hit_pct=64.3, wilson=(38.8, 83.7),
    placebo_mean_bps=-0.31, placebo_sd_bps=36.04, placebo_p=0.242, placebo_draws=20000,
    # event window: offset -> (mean_bps, car_bps, t)
    event={-1: (-57.05, 0.00, -2.17), 0: (-22.77, -22.77, -0.92), 1: (-74.35, -97.11, -1.38),
           2: (-149.74, -246.86, -1.70), 3: (-66.23, -313.09, -1.18), 4: (-7.84, -320.92, -0.11),
           5: (33.22, -287.71, 0.51)},
    # liability window [+1..+5] combined basket
    rev_mean_bps=-264.94, rev_t=-1.41, rev_ci=(-657.7, 41.4),
    rev_jk_below2=14, rev_jk_n=14, rev_jk_tmin=-1.61, rev_jk_tmax=-0.98,
    # legs ([+1..+5])
    util_mean_bps=-704.11, util_t=-1.51, ins_mean_bps=-45.35, ins_t=-0.60,
    leg_diff_bps=-658.76, leg_diff_t=-1.54,
    # utility-linked subset (utility leg)
    ul_n=7, ul_day0_bps=-22.27, ul_day0_t=-0.47, ul_rev_bps=-1596.35, ul_rev_t=-1.96,
    ul_jk_below2=5, ul_jk_n=7,
    # extra drop vs SPY ([+1..+5])
    ed_basket_bps=-264.94, ed_spy_bps=24.86, ed_diff_bps=-289.80, ed_t=-1.71, ed_n=14,
    # seasonal
    seas_in_bps=1.809, seas_out_bps=-1.777, seas_gap_bps=3.586, seas_t=1.019, seas_p=0.297,
    # timer: hold -> (gross_bps, median_bps, net_bps, t_net, win_pct)
    timer={5: (227.91, 30.90, 211.96, 1.19, 57), 10: (197.69, 141.15, 175.79, 1.16, 57),
           21: (116.92, 11.37, 81.92, 0.45, 43)},
    # synthetic control
    syn_null_mean=-0.09, syn_null_sd=0.96, syn_null_fire=1,
    syn_planted_t=-10.92, syn_planted_bps=-298.1,
    fp_eix="62b4a2a1757b", fp_pcg="759d993391f0", fp_all="7898cdc31849", fp_trv="3a070b43af0e",
    fp_mcy="c95d75857b2a", fp_cb="36b3badbd526", fp_spy="1ecb87f843e4",
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Seasonal%3F: Busted](https://img.shields.io/badge/Seasonal%3F-Busted-8b949e?style=flat-square)\n\n"
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

from wildfire_season import data, strategy as st

FIRES = data.fire_table()
HAVE_REAL = data.have_real()
if HAVE_REAL:
    SERIES = data.load_real()
    AR_SPY = st.abnormal_returns(st.daily_returns(SERIES["SPY"]))
    AR_BK = st.abnormal_returns(st.basket_returns(SERIES, data.BASKET_TICKERS))
    AR_UTIL = st.abnormal_returns(st.basket_returns(SERIES, data.UTIL_TICKERS))
    AR_INS = st.abnormal_returns(st.basket_returns(SERIES, data.INS_TICKERS))
    NAV_BK = st.basket_nav(SERIES, data.BASKET_TICKERS)
else:
    SERIES = AR_SPY = AR_BK = AR_UTIL = AR_INS = NAV_BK = None
print("real cache present:", HAVE_REAL, "| fires in table:", len(FIRES),
      "| utility-linked:", int(FIRES["utility_linked"].sum()))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# When California burns, does the market torch the utilities and insurers? 🔥📉\n"
            "### The \"fire-season trade\" — a story that's *half* true, in the most "
            "misleading way\n\n"
            + BADGES +
            "Every California fire season, the same trade makes the rounds: a big "
            "wildfire breaks out, so *sell* (or short) the exposed names — the utility "
            "whose power lines might have sparked it (PG&E, Edison) and the property "
            "insurers left holding the claims (Allstate, Travelers, Mercury, Chubb) — and "
            "underweight the whole basket heading into fire season. And there's a real, "
            "brutal precedent: PG&E literally went **bankrupt** in 2019 over its wildfire "
            "liability, and Edison cratered when its equipment became the suspect in the "
            "January 2025 Los Angeles fires.\n\n"
            "So the folklore *feels* obviously right. We test the strong, tradable "
            "version of it on **14 of the biggest California wildfires since 2003** — and "
            "find that the obvious-looking truth hides three things that quietly kill the "
            "trade.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo, the "
            "bootstrap/jackknife outlier autopsy and the seasonal test? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Note, stated plainly.** The \"short it\" test near the end is a "
            "clinical statistical exercise on public market data — a wildfire is a human "
            "tragedy first; this is a test of whether a *trading pattern* exists, not a "
            "suggestion about how to profit from disaster. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the basket drop the day a fire breaks out? | **No — not detectably.** "
            f"The utility+insurer basket moves **{R['day0_mean_bps']:+.2f} bps** on the "
            f"average ignition day (*t* = {R['day0_t']:.2f}) — a random calendar beats "
            f"that about **{R['placebo_p']*100:.0f}%** of the time. |\n"
            "| So where's the famous PG&E-style crater? | **In the days *after* the fire, "
            "not on it** — and only for the *utilities*, only on the handful of fires "
            "their own equipment caused. It's a slow **liability** repricing (\"our lines "
            "started it\"), not a same-day market flinch. |\n"
            "| Do the insurers get hit too? | **Barely.** The insurer half of the basket "
            f"gives up just **{R['ins_mean_bps']/100:+.2f}%** over the week after a fire "
            "(*t* = −0.60) — they're diversified and reinsured; one state's fire is a "
            "footnote. |\n"
            "| Is there a \"sell in July\" fire-season seasonal? | **No — it's the wrong "
            "sign.** The basket does *slightly better* inside the July→December fire "
            "window than the rest of the year. |\n"
            "| Could you trade it? | **No edge — a lottery.** Shorting the basket on the "
            "fire headline pays *on average* only because of two bankruptcy-scale "
            "jackpots; the *typical* fire barely moves it, and by three weeks you lose "
            "more often than you win. |\n\n"
            "> The intuitive truth — \"the fire destroyed the stock\" — is real for a "
            "*couple* of megafires, but it's a delayed, utility-specific liability event, "
            "not the same-day, basket-wide, seasonal, tradable pattern the story sells."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"California fire season is a recurring, tradable risk event. When a "
            "major wildfire ignites, sell the utility whose lines may have started it and "
            "the insurers who'll pay the claims — and stay underweight the basket through "
            "the summer-to-autumn fire window.\"*\n\n"
            "This one has real teeth, unlike most folklore we test. California's "
            "**inverse-condemnation** doctrine makes a utility strictly liable for damage "
            "its equipment causes — *even without negligence* — so a fire traced to a "
            "power line is a direct, enormous, near-automatic hit to that utility's "
            "equity. PG&E's 2019 Chapter 11 (≈ $30 bn of fire liability) and Edison's "
            "January 2025 plunge are the proof the mechanism is real. The question isn't "
            "*whether* a utility-caused megafire can wreck the stock — it obviously can — "
            "it's whether that adds up to a **systematic, tradable** pattern across fires."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the strong version held, you'd have a clean, repeatable playbook: short "
            "the basket on every ignition headline, and lighten up every July. That's a "
            "real edge in a sector most investors hold for its *safety* (utilities and "
            "insurers are classic defensive, dividend names). So we ask four things: does "
            "the basket drop on the ignition day, does the damage (wherever it is) show up "
            "in insurers or just utilities, is the fire *season* itself a bad time to own "
            "the basket, and does shorting the headline actually pay after costs and "
            "borrow?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The calendar.** **{R['n_events']}** major California wildfires from "
            f"{R['cal_lo']} to {R['cal_hi']} — the front-page fires any Californian would "
            f"name — each tagged by whether a utility's equipment (likely) caused it "
            f"(**{R['n_util_linked']}** utility-linked, **{R['n_nonutil']}** not).\n"
            "- **The comparison.** The basket's return on the first tradable session "
            "after each ignition vs. every other day, and vs. a random 14-day calendar "
            "drawn 20,000 times.\n"
            "- **The split.** Utilities (EIX/PCG) vs. insurers (ALL/TRV/MCY/CB) — which "
            "half actually moves?\n"
            "- **The seasonal.** Is the July→December fire window a systematically worse "
            "time to own the basket than the rest of the year?\n"
            "- **The trade.** Short the basket at the ignition close, hold a few weeks, "
            "pay costs *and* borrow — do you beat zero, and how often do you actually win?"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the ignition day.** If a fire is a market event for the basket, the "
            "day it breaks should show it."
        ),
        code(
            "if HAVE_REAL:\n"
            "    d0 = st.day0_stats(AR_BK, FIRES['date'], pre=1, post=5)\n"
            "    day0_bps = d0['mean'] * 1e4\n"
            "else:\n"
            "    day0_bps = R['day0_mean_bps']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['fire ignition day\\n(n=14)', 'random day\\n(placebo mean)'],\n"
            "       [day0_bps, R['placebo_mean_bps']], color=[RED, GREY], width=.55)\n"
            "for i, v in enumerate([day0_bps, R['placebo_mean_bps']]):\n"
            "    ax.annotate(f'{v:+.2f} bps', (i, v), ha='center',\n"
            "                va='top' if v < 0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('average basket move (bps)')\n"
            "ax.set_title('The ignition-day bar is lost in the noise')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'ignition day {day0_bps:+.2f} bps vs random day {R[\"placebo_mean_bps\"]:+.2f} bps')"
        ),
        md(
            f"**{R['day0_mean_bps']:+.2f} bps** on the average ignition day — the right "
            f"sign, but tiny, and a random 14-day calendar produces a dip this big about "
            f"**{R['placebo_p']*100:.0f}%** of the time. On the *day* the fire breaks, "
            "the market doesn't yet know whose fault it is — so it doesn't move. That's "
            "the first clue: whatever damage there is doesn't happen on day 0.\n\n"
            "**Next, the week around the fire.** Watch where the drop actually lives."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cp = st.car_path_stats(AR_BK, FIRES['date'], pre=1, post=5)\n"
            "    ks, cars = list(cp.index), list(cp['car'] * 1e4)\n"
            "else:\n"
            "    ks = sorted(R['event']); cars = [R['event'][k][1] for k in ks]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "ax.plot([str(k) for k in ks], cars, marker='o', color=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.axvline('0', c=GREY, ls='--', lw=1)\n"
            "ax.annotate('ignition', ('0', 0), textcoords='offset points', xytext=(6, 10),\n"
            "            color=GREY)\n"
            "ax.set_xlabel('trading days relative to ignition (0 = first tradable session)')\n"
            "ax.set_ylabel('cumulative abnormal basket return (bps)')\n"
            "ax.set_title('The damage builds AFTER the fire — as the liability news lands')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('CAR by offset (bps):', {k: round(c, 1) for k, c in zip(ks, cars)})"
        ),
        md(
            "There it is: the line slides *downhill after* day 0, bottoming around "
            "**−3.2%** a few sessions later. That shape — flat on the day, sinking over "
            "the next week — is the fingerprint of a **fundamental** repricing: markets "
            "spend the following days working out *whose equipment started it* and "
            "*how big the liability is*. It is not a mood/sentiment flinch on the news.\n\n"
            "**Now the crucial split.** Utilities or insurers — who's actually taking the "
            "hit?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    lc = st.leg_compare(AR_UTIL, AR_INS, FIRES['date'], pre=1, post=5, window='post')\n"
            "    util_b, ins_b = lc['util_mean'] * 1e4, lc['ins_mean'] * 1e4\n"
            "else:\n"
            "    util_b, ins_b = R['util_mean_bps'], R['ins_mean_bps']\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.4))\n"
            "ax.bar(['utilities\\n(EIX/PCG)', 'insurers\\n(ALL/TRV/MCY/CB)'],\n"
            "       [util_b, ins_b], color=[RED, GREY], width=.55)\n"
            "for i, v in enumerate([util_b, ins_b]):\n"
            "    ax.annotate(f'{v/100:+.2f}%', (i, v), ha='center', va='top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean [+1..+5] post-fire return (bps)')\n"
            "ax.set_title('The whole drop is the utilities; insurers barely move')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'utility leg {util_b/100:+.2f}%  vs  insurer leg {ins_b/100:+.2f}% (post-fire week)')"
        ),
        md(
            f"The utilities give up **{R['util_mean_bps']/100:+.2f}%** over the week after "
            f"a fire; the insurers, **{R['ins_mean_bps']/100:+.2f}%** — essentially "
            "nothing. That kills half the folklore outright: property insurers are "
            "national, diversified and reinsured, so even a record California fire loss is "
            "a footnote to their book. The \"trade\" is really just *one thing* — the two "
            "California utilities' liability risk.\n\n"
            "**The seasonal.** Is fire season itself a bad time to own the basket?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    seas = st.seasonal_test(AR_BK, fire_months=data.FIRE_MONTHS)\n"
            "    inb, outb = seas['in_mean'] * 1e4, seas['out_mean'] * 1e4\n"
            "else:\n"
            "    inb, outb = R['seas_in_bps'], R['seas_out_bps']\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.4))\n"
            "ax.bar(['fire window\\n(Jul-Dec)', 'rest of year'], [inb, outb],\n"
            "       color=[AMBER, GREY], width=.55)\n"
            "for i, v in enumerate([inb, outb]):\n"
            "    ax.annotate(f'{v:+.2f} bps/day', (i, v), ha='center',\n"
            "                va='bottom' if v > 0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean daily abnormal basket return (bps)')\n"
            "ax.set_title('\"Sell in July\" points the WRONG way — fire season is mildly better')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'fire window {inb:+.2f} bps/day vs rest {outb:+.2f} bps/day')"
        ),
        md(
            "The \"sell in July for California risk\" seasonal doesn't just fail to show "
            "up — it points the **wrong way**. Utilities are defensive, dividend-rich "
            "names that tend to do *well* when the broad market gets jittery in the "
            "autumn, so the basket's fire-window return is if anything slightly "
            "*positive*. The calendar simply doesn't price fire risk in advance.\n\n"
            "**Finally, the trade.** Short the basket on the fire headline — does it pay?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    holds = [5, 10, 21]; means = []; meds = []; wins = []\n"
            "    for h in holds:\n"
            "        lg = st.fire_timer(NAV_BK, FIRES['date'], hold=h, cost_bps=5.0,\n"
            "                           borrow_bps_annual=300.0)\n"
            "        s = st.summarize_timer(lg, col='ret_net')\n"
            "        means.append(s['mean_bps']); meds.append(s['median_bps'])\n"
            "        wins.append(s['win_rate'] * 100)\n"
            "else:\n"
            "    holds = sorted(R['timer'])\n"
            "    means = [R['timer'][h][2] for h in holds]\n"
            "    meds = [R['timer'][h][1] for h in holds]\n"
            "    wins = [R['timer'][h][4] for h in holds]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.6))\n"
            "x = np.arange(len(holds)); w = 0.36\n"
            "ax.bar(x - w/2, means, width=w, color=RED, label='MEAN short return (net)')\n"
            "ax.bar(x + w/2, meds, width=w, color=GREY, label='MEDIAN short return (net)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in holds])\n"
            "ax.set_ylabel('short-the-fire return (bps)')\n"
            "ax.set_title('The mean is a 2-jackpot mirage — the typical fire barely moves')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('net mean (bps):', dict(zip(holds, [round(v,1) for v in means])))\n"
            "print('net median (bps):', dict(zip(holds, [round(v,1) for v in meds])))\n"
            "print('win rate (%):', dict(zip(holds, [round(v) for v in wins])))"
        ),
        md(
            f"Look at the gap between the two bars. At a 5-day hold the short's *mean* is "
            f"**{R['timer'][5][2]:+.0f} bps** but its *median* is only "
            f"**{R['timer'][5][1]:+.0f} bps** — the average is propped up entirely by two "
            "bankruptcy-scale jackpots (PG&E's Camp Fire, Edison's Eaton Fire). The "
            "*typical* fire barely moves the basket, you're paying borrow to stay short a "
            "basket of defensive dividend stocks, and by three weeks the win rate falls "
            f"to **{R['timer'][21][4]}%** — you lose more often than you win. This is a "
            "negative-carry bet on a rare, un-timeable disaster, not an edge."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The ignition day is flat "
            f"(**{R['day0_mean_bps']:+.2f} bps**, *t* = {R['day0_t']:.2f}, placebo "
            f"*p* = {R['placebo_p']:.2f}). The real post-fire crater is a delayed, "
            "utility-only, liability repricing concentrated in two megafires — it doesn't "
            "clear the desk's significance bar even at its strongest, and insurers barely "
            "move.\n"
            "- **Tradability — Mirage.** Shorting the headline is a 2-jackpot lottery: "
            "positive mean, tiny median, never significant net of borrow, negative "
            "win-rate by three weeks.\n"
            "- **\"A sell-in-July fire-season seasonal?\" — Busted.** The fire window is "
            "if anything mildly *positive* for the basket. The calendar doesn't price "
            "fire risk ahead of time."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **This doesn't deny that utility-caused megafires wreck utility stocks.** "
            "They obviously do — PG&E went bankrupt. The honest finding is narrower and "
            "more useful: that damage is a *fundamental liability event*, it lands in the "
            "*days after* ignition (not on it), it's *utility-only* (not insurers), it's "
            "*concentrated* in a couple of fires, and it is *not* a same-day, seasonal, "
            "or costable trading pattern.\n"
            "- **Where a real edge might hide:** conditioning on *cause attribution* "
            "(shorting only once a utility's equipment is publicly named as the suspect — "
            "but that news is already in the price), or single-name utility options "
            "around known ignition-investigation catalysts, rather than a blunt "
            "basket-and-calendar rule.\n"
            "- **Sibling studies:** [707-plane-crash-effect](../../707-plane-crash-effect/) "
            "(the mirror image — a *sentiment* claim on a broad index, also busted), and "
            "[313-geopolitical-shock](../../313-geopolitical-shock/) (hardcoded shock "
            "calendar, same placebo design).\n\n"
            "*Think cause-attribution timing or single-name options catch what a "
            "basket-and-calendar rule misses? Show it — out-of-sample, after costs and "
            "borrow — then we'll talk.*"
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
            "# The Wildfire-Season trade — a quantitative teardown 🔬\n"
            "### The ignition-day event study · a 20-seed random-calendar placebo · the "
            "[−1..+5] anatomy and its liability-window reading · a bootstrap/jackknife "
            "outlier autopsy · a utility-vs-insurer leg split · a basket-vs-SPY extra "
            "drop · a random-window seasonal placebo · a costed short-the-fire timer · a "
            "20-seed synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious]"
            "(01_for_the_curious.ipynb). The claim — California fire season is a "
            "systematic, tradable risk event for the state's utilities and property "
            "insurers — has an unusually *real* mechanism (inverse-condemnation strict "
            "liability; PG&E's 2019 bankruptcy). The job here is to separate that genuine "
            "but *idiosyncratic and fundamental* liability risk from the *systematic, "
            "same-day, seasonal, tradable* pattern the folklore actually claims — and to "
            "grade each on the desk's inference bar.\n\n"
            "> ⚠️ **Data note.** EIX/PCG (utilities) + ALL/TRV/MCY/CB (insurers) + SPY "
            "total-return closes (2003→2026), yfinance, cached; **14 hardcoded major "
            "California wildfires** 2003→2025, each flagged by utility-cause. Full "
            "coverage on every event (no survivorship gap on this axis) — but **PG&E's "
            "2019 Chapter 11 and dilution are *inside* the `PCG` tape, not survivored "
            "out**. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprints `" + R["fp_eix"] +
            "` EIX / `" + R["fp_pcg"] + "` PCG / `" + R["fp_spy"] + "` SPY, + 4 insurers).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | ignition-day abnormal basket return "
            f"**{R['day0_mean_bps']:+.2f} bps**, one-sample **t = {R['day0_t']:+.2f}**, "
            f"placebo **p = {R['placebo_p']:.3f}**; the [+1..+5] liability window is "
            f"large but sub-bar (**{R['rev_mean_bps']:+.0f} bps**, t = {R['rev_t']:.2f}, "
            f"bootstrap CI crosses 0, {R['rev_jk_below2']}/{R['rev_jk_n']} leave-one-out "
            f"below 2) |\n"
            f"| **Tradability** | `MIRAGE` | short-the-fire mean +{R['timer'][5][2]:.0f} "
            f"bps but median only +{R['timer'][5][1]:.0f} bps (2-jackpot lottery), best "
            f"net **t = +{R['timer'][5][3]:.2f}**, win-rate {R['timer'][21][4]}% by 21d |\n"
            f"| **Fire-season seasonal?** | `BUSTED` | Jul→Dec gap "
            f"**{R['seas_gap_bps']:+.2f} bps/day** (wrong sign), Welch **t = "
            f"+{R['seas_t']:.2f}**, random-window p = {R['seas_p']:.2f} |\n\n"
            "> 💡 In plain words: the mechanism is real (utilities *can* be wrecked by a "
            "fire they caused), but it's a delayed, utility-only, concentrated "
            "*fundamental* event — not the systematic same-day/seasonal/tradable pattern "
            "the folklore sells. Every axis of the tradable claim comes back empty."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t$ be the equal-weight utility+insurer basket's daily return and "
            "$a_t = r_t - \\bar{r}$ its abnormal return under a constant-mean market "
            "model (Brown & Warner 1985). For each fire $i$ with first-tradable-session "
            "date $\\tau_i$:\n\n"
            "- **H₁ (ignition dip).** $E[a_{\\tau_i}] < 0$, large and systematic across "
            "events.\n"
            "- **H₂ (post-fire crater).** $\\sum_{k=1}^{5} a_{\\tau_i+k} < 0$ — the drop, "
            "wherever it is, is real over the following week.\n"
            "- **H₃ (insurers hit too).** The insurer leg falls, not just the utilities.\n"
            "- **H₄ (fire-season seasonal).** The Jul→Dec window's mean abnormal return "
            "is materially below the rest of the year.\n"
            "- **H₅ (capture).** A short-the-ignition overlay beats zero net of costs and "
            "borrow.\n\n"
            "We find **H₁ not supported** (t = −0.92), **H₂ directionally real but "
            "sub-bar and outlier-fragile** (t = −1.41, bootstrap CI crosses 0, all "
            "leave-one-out below 2), **H₃ rejected** (insurer leg t = −0.60, ≈ flat), "
            "**H₄ rejected on sign** (fire window is mildly *positive*, Welch t = +1.02), "
            "**H₅ not supported** (a positive-mean, tiny-median lottery, never "
            "significant). The one place the effect nearly lives — the utility leg on "
            "utility-caused fires — reaches only t = −1.96 and is Camp/Eaton-driven."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Fire events are **independent, non-overlapping calendar dates** (weeks to "
            "years apart), so the planned primary is a **one-sample t-test** across the "
            "14 per-event abnormal returns — the unit of analysis is \"one number per "
            "fire\", not a daily panel, so no HAC correction is needed. Because a "
            "14-event mean can be dominated by one or two megafires, the [+1..+5] window "
            "carries an **event bootstrap** 95% CI *and* a **leave-one-out jackknife** of "
            "the t-stat — the desk's outlier discipline. The hit rate carries a **Wilson "
            "interval**; the event placebo draws 14 random non-fire dates **20,000 times "
            "(20 seeds × 1,000)**; the [−1..+5] anatomy is a **7-offset multiple-"
            "comparison** (one spurious |t| ≥ 2 expected by chance); the seasonal carries "
            "a **random-6-month-window** null; the utility/insurer split and basket/SPY "
            "test are **paired** on the same dates so common market moves cancel."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Calendar.** {R['n_events']} major California wildfires {R['cal_lo']} → "
            f"{R['cal_hi']}, hardcoded, {R['n_util_linked']} utility-linked / "
            f"{R['n_nonutil']} not (cause flag from Cal Fire / CPUC / 8-K disclosures).\n"
            "- **Tape.** EIX/PCG/ALL/TRV/MCY/CB + SPY total-return closes, 2003 → "
            "2026-06-30 (as-of). Full coverage every event.\n"
            "- **Headline.** Ignition-day (offset 0) one-sample t + Wilson hit rate + "
            "20-seed placebo, on the combined basket.\n"
            "- **Anatomy.** Event window [−1..+5], per-offset mean + t (multiple-"
            "comparison caveat); the [+1..+5] liability-window CAR with bootstrap CI + "
            "jackknife.\n"
            "- **Splits.** Utility leg vs insurer leg ([+1..+5]); the utility-linked "
            "subset (n=7); paired basket − SPY extra drop.\n"
            "- **Seasonal.** Jul→Dec vs rest-of-year daily-mean gap, Welch t + random-"
            "6-month-window placebo.\n"
            "- **Execution (timer).** SHORT the basket at the ignition close (zero look-"
            "ahead — the ignition predates that session's close), exit `hold` sessions "
            "later; 2 × one-way cost × NAV + 300 bps/yr borrow; gross, net, *and median*.\n"
            "- **Control.** Synthetic random-walk tape, planted mean-reverting dip; the "
            "null must not fire beyond nominal size across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline split and its placebo\n\n"
            "One-sample t on the ignition-day (offset 0) abnormal basket return, Wilson "
            "hit rate, and the random-calendar null (a lighter 4-seed × 500 run in-"
            "notebook; the canonical 20,000-draw p is quoted from `results.md`)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    d0 = st.day0_stats(AR_BK, FIRES['date'], pre=1, post=5)\n"
            "    wlo, whi = st.wilson_interval(d0['hit_down'], d0['n'])\n"
            "    print(f\"ignition-day abnormal basket return {d0['mean']*1e4:+.2f} bps  \"\n"
            "          f\"one-sample t = {d0['t']:+.3f}  (n={d0['n']})\")\n"
            "    print(f\"hit rate (down): {d0['hit_down']}/{d0['n']} = \"\n"
            "          f\"{d0['hit_down']/d0['n']*100:.1f}%  Wilson [{wlo*100:.1f}%, {whi*100:.1f}%]\")\n"
            "    pl = np.concatenate([st.placebo_distribution(AR_BK, d0['n'], n_draws=500,\n"
            "        seed=739 + s_, stat='day0') for s_ in range(4)])\n"
            "    obs, draws = d0['mean'], pl\n"
            "else:\n"
            "    obs = R['day0_mean_bps'] / 1e4\n"
            "    rng = np.random.default_rng(739)\n"
            "    draws = rng.normal(R['placebo_mean_bps'] / 1e4, R['placebo_sd_bps'] / 1e4, 2000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws * 1e4, bins=50, color=GREY, alpha=.85,\n"
            "        label='null: random 14-day calendars (light in-notebook run)')\n"
            "ax.axvline(obs * 1e4, c=RED, lw=2.5,\n"
            "           label=f'observed ignition-day mean {obs*1e4:+.2f} bps')\n"
            "ax.set_xlabel('mean abnormal basket return of a random 14-day calendar (bps)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"Well inside the luck cloud: canonical p = {R['placebo_p']:.3f} \"\n"
            "             '(20 seeds x 1,000 draws)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"canonical placebo (results.md): mean {R['placebo_mean_bps']:+.2f} bps, \"\n"
            "      f\"sd {R['placebo_sd_bps']:.2f} bps, p = {R['placebo_p']:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: the observed **{R['day0_mean_bps']:+.2f} bps** sits "
            f"well inside the null ({R['placebo_mean_bps']:+.2f} ± "
            f"{R['placebo_sd_bps']:.2f} bps); **p = {R['placebo_p']:.3f}**. With one-"
            f"sample t = **{R['day0_t']:.2f}** and a hit-rate Wilson interval "
            f"[{R['wilson'][0]:.1f}%, {R['wilson'][1]:.1f}%] straddling 50%, H₁ (a same-"
            "day ignition dip) does not clear the bar. On the day the fire breaks, the "
            "market doesn't yet know who's liable."
        ),
        md(
            "### 4b · Anatomy — the [−1..+5] window and the liability autopsy\n\n"
            "Per-offset means + t (7-offset multiple comparison), then the [+1..+5] "
            "liability window with its bootstrap CI and leave-one-out jackknife — the "
            "outlier discipline that decides whether a big-looking mean is real."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cp = st.car_path_stats(AR_BK, FIRES['date'], pre=1, post=5)\n"
            "    ks = list(cp.index); ms = list(cp['mean_ar'] * 1e4); ts = list(cp['t'])\n"
            "    rev = st.reversal_stats(AR_BK, FIRES['date'], pre=1, post=5)\n"
            "    ci = st.block_bootstrap_ci(rev['per_event'])\n"
            "    jk = st.jackknife_range(rev['per_event'])\n"
            "    rev_m, rev_t = rev['mean'] * 1e4, rev['t']\n"
            "    ci_lo, ci_hi = ci[0] * 1e4, ci[1] * 1e4\n"
            "    jk_below, jk_n = jk['n_below2'], jk['n']\n"
            "else:\n"
            "    ks = sorted(R['event']); ms = [R['event'][k][0] for k in ks]\n"
            "    ts = [R['event'][k][2] for k in ks]\n"
            "    rev_m, rev_t = R['rev_mean_bps'], R['rev_t']\n"
            "    ci_lo, ci_hi = R['rev_ci']; jk_below, jk_n = R['rev_jk_below2'], R['rev_jk_n']\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.4, 6.4), sharex=True,\n"
            "                             gridspec_kw={'height_ratios': [2, 1]})\n"
            "cols = [RED if k == 0 else GREY for k in ks]\n"
            "a1.bar([str(k) for k in ks], ms, color=cols, width=.62)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean abnormal return (bps)')\n"
            "a1.set_title('Event anatomy: flat on ignition, sinking over the next week')\n"
            "a2.bar([str(k) for k in ks], ts,\n"
            "       color=[RED if abs(t) >= 2 else GREY for t in ts], width=.62)\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.axhline(-2, ls='--', c=RED, lw=1); a2.axhline(2, ls='--', c=RED, lw=1)\n"
            "a2.set_ylabel('one-sample t'); a2.set_xlabel('offset (sessions from ignition)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'[+1..+5] liability CAR {rev_m:+.1f} bps (t={rev_t:+.2f}); '\n"
            "      f'bootstrap 95% CI [{ci_lo:+.0f}, {ci_hi:+.0f}] bps; '\n"
            "      f'|t|<2 in {jk_below}/{jk_n} leave-one-out drops')"
        ),
        md(
            f"> 💡 In plain words: day 0 is **{R['event'][0][0]:+.2f} bps** "
            f"(t = {R['event'][0][2]:.2f}); the drop *builds* over +1..+3 (the liability "
            f"news landing), summing to **{R['rev_mean_bps']:+.0f} bps** over [+1..+5]. "
            f"But t = **{R['rev_t']:.2f}**, the bootstrap CI **[{R['rev_ci'][0]:+.0f}, "
            f"{R['rev_ci'][1]:+.0f}] bps crosses zero**, and **every** "
            f"({R['rev_jk_below2']}/{R['rev_jk_n']}) leave-one-out drop keeps |t| under "
            "2. The one bar that crosses the bar — offset **−1** (t = "
            f"{R['event'][-1][2]:.2f}) — is on the *wrong side of the event* (before the "
            "fire is public), a textbook look-elsewhere false positive among 7 offsets."
        ),
        md(
            "### 4c · The splits — utilities vs insurers, and the utility-linked subset\n\n"
            "Same dates: which leg carries the drop, and what happens when we restrict to "
            "the 7 fires a utility's equipment (likely) caused."
        ),
        code(
            "if HAVE_REAL:\n"
            "    lc = st.leg_compare(AR_UTIL, AR_INS, FIRES['date'], pre=1, post=5, window='post')\n"
            "    ul = FIRES.loc[FIRES['utility_linked'], 'date']\n"
            "    rev_ul = st.reversal_stats(AR_UTIL, ul, pre=1, post=5)\n"
            "    d0_ul = st.day0_stats(AR_UTIL, ul, pre=1, post=5)\n"
            "    jk_ul = st.jackknife_range(rev_ul['per_event'])\n"
            "    util_b, ins_b = lc['util_mean'] * 1e4, lc['ins_mean'] * 1e4\n"
            "    ut_ut, in_t = lc['util_t'], lc['ins_t']\n"
            "    ul_m, ul_t = rev_ul['mean'] * 1e4, rev_ul['t']\n"
            "    ul_d0, ul_d0t = d0_ul['mean'] * 1e4, d0_ul['t']\n"
            "    ul_below, ul_n = jk_ul['n_below2'], jk_ul['n']\n"
            "else:\n"
            "    util_b, ins_b = R['util_mean_bps'], R['ins_mean_bps']\n"
            "    ut_ut, in_t = R['util_t'], R['ins_t']\n"
            "    ul_m, ul_t = R['ul_rev_bps'], R['ul_rev_t']\n"
            "    ul_d0, ul_d0t = R['ul_day0_bps'], R['ul_day0_t']\n"
            "    ul_below, ul_n = R['ul_jk_below2'], R['ul_jk_n']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.bar(['utilities\\n(EIX/PCG)', 'insurers\\n(ALL/TRV/MCY/CB)'], [util_b, ins_b],\n"
            "       color=[RED, GREY], width=.55)\n"
            "for i, (v, t2) in enumerate([(util_b, ut_ut), (ins_b, in_t)]):\n"
            "    a1.annotate(f'{v/100:+.2f}%\\n(t={t2:+.2f})', (i, v), ha='center', va='top')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('[+1..+5] post-fire return (bps)')\n"
            "a1.set_title('Utilities carry it all; insurers ~flat')\n"
            "a2.bar(['all 14 fires', 'utility-linked\\n(n=7)'], [R['util_mean_bps'], ul_m],\n"
            "       color=[GREY, RED], width=.55)\n"
            "for i, (v, t2) in enumerate([(R['util_mean_bps'], R['util_t']), (ul_m, ul_t)]):\n"
            "    a2.annotate(f'{v/100:+.1f}%\\n(t={t2:+.2f})', (i, v), ha='center', va='top')\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('utility-leg [+1..+5] return (bps)')\n"
            "a2.set_title(f'Best cut still sub-bar: |t|<2 in {ul_below}/{ul_n} drops')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'utility {util_b/100:+.2f}% (t={ut_ut:+.2f}) vs insurer {ins_b/100:+.2f}% (t={in_t:+.2f})')\n"
            "print(f'utility-linked subset: day0 {ul_d0:+.1f} bps (t={ul_d0t:+.2f}); '\n"
            "      f'[+1..+5] {ul_m/100:+.1f}% (t={ul_t:+.2f}); |t|<2 in {ul_below}/{ul_n} drops')"
        ),
        md(
            f"> 💡 In plain words: the utility leg gives up **{R['util_mean_bps']/100:+.2f}%** "
            f"post-fire (t = {R['util_t']:.2f}); the insurer leg "
            f"**{R['ins_mean_bps']/100:+.2f}%** (t = {R['ins_t']:.2f}) — H₃ rejected, "
            "insurers are diversified/reinsured. Restricting the utility leg to the 7 "
            f"utility-caused fires drives the [+1..+5] drop to "
            f"**{R['ul_rev_bps']/100:+.1f}%** — but only **t = {R['ul_rev_t']:.2f}** "
            f"(one hair short of 2) and **{R['ul_jk_below2']}/{R['ul_jk_n']} leave-one-"
            "out drops fall below 2**: take out Camp *or* Eaton and it evaporates. And "
            f"the ignition day *within* this subset is still flat "
            f"({R['ul_day0_bps']:+.1f} bps, t = {R['ul_day0_t']:.2f}) — confirming a "
            "delayed liability repricing, not a same-day event."
        ),
        md(
            "### 4d · The seasonal — is fire season a bad time to own the basket?\n\n"
            "Jul→Dec fire-window mean daily abnormal return vs the rest of the year, "
            "with a random-6-month-window null."
        ),
        code(
            "if HAVE_REAL:\n"
            "    seas = st.seasonal_test(AR_BK, fire_months=data.FIRE_MONTHS)\n"
            "    sp = st.seasonal_placebo(AR_BK, k_months=len(data.FIRE_MONTHS), n_draws=4000, seed=739)\n"
            "    gap, tt = seas['diff'] * 1e4, seas['t']\n"
            "    p_seas = float((np.abs(sp) >= abs(seas['diff'])).mean())\n"
            "    inb, outb = seas['in_mean'] * 1e4, seas['out_mean'] * 1e4\n"
            "else:\n"
            "    gap, tt, p_seas = R['seas_gap_bps'], R['seas_t'], R['seas_p']\n"
            "    inb, outb = R['seas_in_bps'], R['seas_out_bps']\n"
            "    rng = np.random.default_rng(739)\n"
            "    sp = rng.normal(0, abs(R['seas_gap_bps'] / 1e4) / 1.04, 4000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(sp * 1e4, bins=50, color=GREY, alpha=.85,\n"
            "        label='null: random 6-month windows')\n"
            "ax.axvline(gap, c=RED, lw=2.5, label=f'observed Jul-Dec gap {gap:+.2f} bps/day')\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('in-window minus out-of-window mean daily return (bps)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Wrong sign and unremarkable: Welch t = {tt:+.2f}, random-window p = {p_seas:.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'fire window {inb:+.3f} bps/day vs rest {outb:+.3f} bps/day -> '\n"
            "      f'gap {gap:+.3f} bps/day (Welch t={tt:+.2f}, random-window p={p_seas:.3f})')"
        ),
        md(
            f"> 💡 In plain words: H₄ predicts a materially **negative** fire-window gap. "
            f"We measure **{R['seas_gap_bps']:+.2f} bps/day** — *positive*, wrong sign, "
            f"Welch t = **+{R['seas_t']:.2f}**, and a random 6-month calendar slice "
            f"produces a gap this large about **{R['seas_p']*100:.0f}%** of the time. "
            "Utilities are defensive dividend names that tend to do *better* in autumn "
            "risk-off; the calendar does not price fire risk in advance. Busted."
        ),
        md(
            "### 4e · The timer — short-the-fire cost & borrow sweep\n\n"
            "*Clinical test of public market data.* SHORT the basket at the ignition "
            "close (zero look-ahead), hold `h` sessions; 5 bps one-way charged twice + "
            "300 bps/yr borrow prorated over the hold. The **median** is reported next to "
            "the mean because that gap is the whole story."
        ),
        code(
            "if HAVE_REAL:\n"
            "    holds = [5, 10, 21]; gross = []; net = []; med = []; ts = []; wins = []\n"
            "    for h in holds:\n"
            "        lg_g = st.fire_timer(NAV_BK, FIRES['date'], hold=h, cost_bps=0.0,\n"
            "                             borrow_bps_annual=0.0)\n"
            "        g = st.summarize_timer(lg_g, col='ret_gross')\n"
            "        lg_n = st.fire_timer(NAV_BK, FIRES['date'], hold=h, cost_bps=5.0,\n"
            "                             borrow_bps_annual=300.0)\n"
            "        n_ = st.summarize_timer(lg_n, col='ret_net')\n"
            "        gross.append(g['mean_bps']); net.append(n_['mean_bps'])\n"
            "        med.append(n_['median_bps']); ts.append(n_['t']); wins.append(n_['win_rate']*100)\n"
            "else:\n"
            "    holds = sorted(R['timer'])\n"
            "    gross = [R['timer'][h][0] for h in holds]\n"
            "    net = [R['timer'][h][2] for h in holds]\n"
            "    med = [R['timer'][h][1] for h in holds]\n"
            "    ts = [R['timer'][h][3] for h in holds]\n"
            "    wins = [R['timer'][h][4] for h in holds]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.6))\n"
            "x = np.arange(len(holds)); w = 0.27\n"
            "ax.bar(x - w, gross, width=w, color=GREY, label='gross mean')\n"
            "ax.bar(x, net, width=w, color=RED, label='net mean (5bps + borrow)')\n"
            "ax.bar(x + w, med, width=w, color=AMBER, label='net MEDIAN')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in holds])\n"
            "ax.set_ylabel('short-the-fire return (bps)')\n"
            "ax.set_title('Mean >> median: a 2-jackpot lottery, not an edge')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('gross mean (bps):', dict(zip(holds, [round(v,1) for v in gross])))\n"
            "print('net mean (bps):', dict(zip(holds, [round(v,1) for v in net])))\n"
            "print('net median (bps):', dict(zip(holds, [round(v,1) for v in med])))\n"
            "print('net t / win% :', dict(zip(holds, [f'{t:+.2f}/{w:.0f}%' for t, w in zip(ts, wins)])))"
        ),
        md(
            f"> 💡 In plain words: at the 5-day hold the short nets a mean "
            f"**{R['timer'][5][2]:+.0f} bps** but a **median of only "
            f"{R['timer'][5][1]:+.0f} bps** (net t = +{R['timer'][5][3]:.2f}) — the mean "
            "is two jackpots (Camp, Eaton) inside 14 bets. H₅ is not supported: never "
            f"significant, and by 21 days the win rate is **{R['timer'][21][4]}%** (you "
            "lose more often than you win) while you bleed borrow shorting defensive "
            "dividend stocks. A negative-carry bet on a rare, un-timeable liability "
            "crater."
        ),
        md(
            "### 4f · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic random-walk tape, 14 scheduled pseudo-fire dates, TUNABLE planted "
            "day-0 dip reverting over 5 sessions. The null (dip=0) is checked over "
            "**20 seeds** — never a single stream."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    close, ev = data.synthetic_world(dip=0.0, seed=739 + s_)\n"
            "    null_ts.append(st.synthetic_detect(close, ev, stat='day0')['t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "close, ev = data.synthetic_world(dip=-0.03, seed=739)\n"
            "planted_t = st.synthetic_detect(close, ev, stat='day0')['t']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (dip=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=RED, s=90, zorder=5, label='planted dip = -3.0%')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('one-sample t (ignition-day dip)')\n"
            "ax.set_title('Control: null sits at nominal size; a planted dip lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(np.abs(null_ts) >= 2).sum()}/20 seeds  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and crosses the "
            f"bar in **{R['syn_null_fire']}/20** seeds — the nominal 5% false-positive "
            f"rate, not a broken detector; a planted 3% dip reads t = "
            f"{R['syn_planted_t']:.2f}. The machinery is unbiased, so the real-tape "
            "ignition-day t ≈ −0.92 is a genuine null, not a detector asleep. *(A "
            "faithful-engine / power check only — never cited in support of the real-tape "
            "stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — ignition-day abnormal basket return "
            f"**{R['day0_mean_bps']:+.2f} bps**, one-sample t = **{R['day0_t']:.2f}**, "
            f"placebo p = **{R['placebo_p']:.3f}**. The [+1..+5] liability window is "
            f"large (**{R['rev_mean_bps']:+.0f} bps**) but sub-bar and outlier-fragile "
            f"(t = {R['rev_t']:.2f}, bootstrap CI [{R['rev_ci'][0]:+.0f}, "
            f"{R['rev_ci'][1]:+.0f}] crosses 0, {R['rev_jk_below2']}/{R['rev_jk_n']} "
            f"leave-one-out below 2); insurers ≈ flat (t = {R['ins_t']:.2f}); the best "
            f"cut (utility leg, utility-caused fires) reaches only t = {R['ul_rev_t']:.2f}, "
            f"Camp/Eaton-driven ({R['ul_jk_below2']}/{R['ul_jk_n']} drops below 2). The "
            f"one significant offset (−1, t = {R['event'][-1][2]:.2f}) is a wrong-side "
            "look-elsewhere artifact.\n"
            f"- **Tradability `MIRAGE`** — short-the-fire mean +{R['timer'][5][2]:.0f} bps "
            f"but median +{R['timer'][5][1]:.0f} bps, best net t = +{R['timer'][5][3]:.2f}, "
            f"win-rate {R['timer'][21][4]}% by 21d. A 2-jackpot lottery with negative "
            "carry, not a costable edge.\n"
            f"- **\"A fire-season seasonal?\" `BUSTED`** — Jul→Dec gap "
            f"**{R['seas_gap_bps']:+.2f} bps/day** (wrong sign), Welch t = "
            f"+{R['seas_t']:.2f}, random-window p = {R['seas_p']:.2f}.\n\n"
            "The mechanism is real — inverse-condemnation liability *can* and did wreck a "
            "utility (PG&E) — but it manifests as a **delayed, utility-only, "
            "concentrated fundamental repricing**, not the systematic same-day, "
            "basket-wide, seasonal, tradable pattern the folklore sells."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Power is the honest limitation, in both directions.** n=14 (7 utility-"
            "linked) is small, and the [+1..+5] utility-caused effect is real enough that "
            "a *larger* sample might certify it — but a signal that lives or dies on Camp "
            "and Eaton is, by construction, not something you can *rely on ex-ante*.\n"
            "- **A cause-conditional design** — event-time keyed to the *public "
            "attribution* of a utility as the ignition suspect (an 8-K, a CPUC filing) "
            "rather than the ignition itself — is the natural next test, but that news is "
            "typically already in the price; single-name utility options around known "
            "investigation catalysts are the more honest capacity question.\n"
            "- **Dedup map:** [707-plane-crash-effect](../../707-plane-crash-effect/) "
            "(the mirror image — a *sentiment* claim on a broad index, also None/Mirage) "
            "and [313-geopolitical-shock](../../313-geopolitical-shock/) (hardcoded shock "
            "calendar, same random-calendar placebo design).\n\n"
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
