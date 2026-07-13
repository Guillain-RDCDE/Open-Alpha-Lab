"""Generate the two narrative notebooks for Study 744 (Tetraphobia).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached raw
local-currency closes (clustering) and total-return ETF closes (calendar) under
../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The two synthetic controls run anywhere with no network.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (fingerprint 502e4addea30).
R = dict(
    fp="502e4addea30",
    asof="2026-06-30",
    # --- A · clustering ---
    asia_N=40328, asia_n4=1514, asia_n8=1786, asia_share8=54.1, asia_z84=+4.73, asia_z4=-2.29,
    us_N=41470, us_n4=4155, us_n8=4079, us_share8=49.5, us_z84=-0.84, us_z4=+0.94,
    # digit percentages (0..9)
    asia_pct=[54.4, 3.9, 4.3, 3.9, 3.75, 13.8, 3.9, 3.7, 4.43, 4.0],
    us_pct=[11.0, 9.9, 9.8, 9.8, 10.02, 10.1, 10.0, 9.8, 9.84, 9.9],
    # by region: z(8>4), 8-share
    reg_z84={"Taiwan": +5.25, "ChinaA": +3.62, "HongKong": -0.47, "US": -0.84},
    reg_share8={"Taiwan": 62.0, "ChinaA": 53.9, "HongKong": 49.1, "US": 49.5},
    # --- B · 4/4 returns (core EWT/EWH/MCHI), bps ---
    ewt_n=25, ewt_mean=+9.9, ewt_t=+0.28, ewt_hit=11,
    ewh_n=26, ewh_mean=+26.8, ewh_t=+0.77, ewh_hit=18,
    mchi_n=15, mchi_mean=+8.5, mchi_t=+0.14, mchi_hit=9,
    pool_n=66, pool_mean=+16.2, pool_t=+0.69, pool_hit=38,
    pool6_n=138, pool6_mean=+14.3, pool6_t=+0.84,
    # placebo (left tail)
    pl_obs=+16.2, pl_mean=+3.0, pl_sd=20.6, pl_p=0.748, pl_draws=5000,
    # 8/8 contrast
    e88_pool_n=67, e88_pool_mean=+30.6, e88_pool_t=+1.27, welch_88_44=+0.43,
    # tradability (short 4/4)
    short_n=66, short_gross=-16.2, short_net=-27.2, short_t=-1.16,
    # synthetic controls
    syn_dig_null_mean=-0.38, syn_dig_null_sd=1.14, syn_dig_null_fire=1,
    syn_dig_p03=+2.90, syn_dig_p05=+4.23,
    syn_cal_null_mean=+0.03, syn_cal_null_sd=1.25, syn_cal_null_fire=3,
    syn_cal_dip1=-2.99, syn_cal_dip2=-8.52,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Do_Asian_prices_dodge_the_4%3F: Confirmed](https://img.shields.io/badge/Do_Asian_prices_dodge_the_4%3F-Confirmed-8b949e?style=flat-square)\n\n"
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

from tetraphobia import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    CL = data.load_cluster()
    CA = data.load_calendar()
    ASIA = {t: s for t, s in CL.items() if t in data.ASIA_CLUSTER}
    USC = {t: s for t, s in CL.items() if t in data.US_CONTROL}
else:
    CL = CA = ASIA = USC = None
print("real cache present:", HAVE_REAL,
      "| cluster tickers:", (0 if CL is None else len(CL)),
      "| calendar ETFs:", (0 if CA is None else len(CA)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The fear of '4' — real in the prices, a mirage in the returns 🔢\n"
            "### East-Asian tetraphobia genuinely bends where prices settle. It does "
            "nothing to *when* markets rise or fall.\n\n"
            + BADGES +
            "In Mandarin, Cantonese, Japanese and Korean, the word for **four** sounds "
            "almost exactly like the word for **death**. The superstition — "
            "*tetraphobia* — is everywhere: buildings skip the 4th, 14th and 24th "
            "floors; hospital rooms and airplane rows have no row 4; a phone number or "
            "licence plate ending in 4 sells at a discount, and one ending in **8** (a "
            "homophone of *prosperity*) at a premium.\n\n"
            "Does any of this reach the stock market? There are two very different "
            "claims, and they get two very different answers:\n\n"
            "1. **Do prices themselves avoid the 4?** (Do Chinese/Taiwanese share "
            "prices settle on a trailing 8 more often than a trailing 4?) — **Yes.** "
            "Measurably, robustly, on live data.\n"
            "2. **Does the unlucky *date* 4/4 make those markets fall?** — **No.** Not "
            "even slightly.\n\n"
            "> 📓 **Plain-language layer.** Want the z-tests, the placebo, the two "
            "synthetic controls? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** Clustering uses **raw** local-currency prices (an "
            "adjusted price is a back-computed number nobody traded — its last digit is "
            "meaningless); the calendar test uses total-return ETF closes. Every chart "
            "is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do Greater-China prices dodge a trailing 4 and prefer 8? | **Yes.** "
            f"Among prices ending in 4 or 8, **{R['asia_share8']:.0f}%** end in 8 "
            f"(*z* = **{R['asia_z84']:+.2f}**) — and the US control is a flat coin-flip "
            f"({R['us_share8']:.0f}%, *z* = {R['us_z84']:+.2f}). |\n"
            f"| Does the unlucky date 4/4 sink the market? | **No.** The three markets' "
            f"average 4/4 return is **{R['pool_mean']:+.1f} bps** — *positive*, the "
            f"opposite of the claim — with *t* = **{R['pool_t']:+.2f}**. |\n"
            f"| Could you trade the 4/4 idea? | **No.** Shorting the day (betting it "
            f"drops) loses **{R['short_net']:+.1f} bps** every time, net of costs — the "
            "day tends to *rise*. |\n"
            f"| Is the lucky 8/8 any better? | Barely, and not really. 8/8 averages "
            f"**{R['e88_pool_mean']:+.1f} bps** vs 4/4's {R['pool_mean']:+.1f} — the "
            "right direction, but both are lost in the noise. |\n\n"
            "> The superstition is real enough to bend the **last digit of a price**. "
            "It has *zero* grip on **whether a day goes up or down**."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The number 4 sounds like 'death', so East-Asian markets avoid it — "
            "prices shy away from a trailing 4 and cluster on the lucky 8, and the "
            "double-unlucky date 4/4 is a bad day to be long.\"*\n\n"
            "Half of this is textbook behavioural finance. Brown & Mitchell (2008) "
            "showed Chinese share prices really do cluster away from 4 and toward 8 — a "
            "cultural fingerprint sitting on top of the ordinary round-number habit "
            "every market has. The *other* half — that a **calendar date** built from "
            "4s should underperform — is pure internet folklore. We test both."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a superstition can move a whole market on a *predictable date*, that "
            "would be a gift: mark 4/4 on the calendar, short it, collect. And it would "
            "say something deep — that a shared, non-economic belief moves prices in a "
            "way you could bank. The clustering half, by contrast, is not a trade at "
            "all; it's a beautiful, measurable footprint of culture in the microstructure "
            "of prices. We wanted to know which half survives contact with the data."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "- **The clustering test.** Take the *last digit* of every daily price (at "
            "its finest resolution) for a basket of Taiwan/HK/China/Korea stocks — and, "
            "as a control, a basket of US stocks. Every market clusters on 0 and 5 "
            "(round numbers), so that's not the superstition; the tell is whether **8 "
            "beats 4** in Asia but not in the US.\n"
            "- **The calendar test.** Take the return of the 4/4 session (or the first "
            "session after) every year, 2000→2025, for EWT (Taiwan), EWH (Hong Kong), "
            "MCHI (China). Because 4/4 is a date known years ahead, there's no "
            "look-ahead to worry about.\n"
            "- **The honesty checks.** A US-control placebo for the digits, a "
            "random-calendar placebo for the returns, the lucky 8/8 as a contrast, a "
            "costed short, and two synthetic worlds to prove the detectors actually work."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the prices. Here is the last digit of every close, Asia vs the US "
            "control.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ap = st.digit_pct(st.trailing_digit_counts(ASIA))\n"
            "    up = st.digit_pct(st.trailing_digit_counts(USC))\n"
            "else:\n"
            "    ap, up = np.array(R['asia_pct']), np.array(R['us_pct'])\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4), sharey=False)\n"
            "for ax, p, title in ((a1, ap, 'Asia basket'), (a2, up, 'US control')):\n"
            "    cols = [GREY]*10; cols[4] = RED; cols[8] = GREEN\n"
            "    ax.bar(range(10), p, color=cols)\n"
            "    ax.axhline(10, ls=':', c='k', lw=.8)\n"
            "    ax.set_xticks(range(10)); ax.set_xlabel('trailing digit of the price')\n"
            "    ax.set_title(title)\n"
            "a1.set_ylabel('share of closes (%)')\n"
            "fig.suptitle('Digits 0 & 5 dominate everywhere (round numbers). '\n"
            "             'Only in Asia does 8 (green) beat 4 (red).')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"Asia: 4 -> {ap[4]:.2f}%,  8 -> {ap[8]:.2f}%\")\n"
            "print(f\"US:   4 -> {up[4]:.2f}%,  8 -> {up[8]:.2f}%\")"
        ),
        md(
            "The round-number habit (0 and 5) towers over everything in both baskets — "
            "that's universal and *not* tetraphobia. Zoom into the little bars and the "
            "cultural fingerprint appears: in Asia the **red 4 is the shortest bar and "
            "the green 8 is taller**; in the US they're the same height. Let's put a "
            "number on that asymmetry."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sa = st.tetraphobia_stats(st.trailing_digit_counts(ASIA))\n"
            "    su = st.tetraphobia_stats(st.trailing_digit_counts(USC))\n"
            "    az, uz = sa['z8_gt_4'], su['z8_gt_4']\n"
            "else:\n"
            "    az, uz = R['asia_z84'], R['us_z84']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['Asia basket', 'US control'], [az, uz], color=[RED, GREY], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, lw=1.2, label='significance bar (z=2)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i, v in enumerate([az, uz]): ax.annotate(f'{v:+.2f}', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('z-score: is 8 preferred over 4?')\n"
            "ax.set_title('Asia dodges the 4 (z=+4.7); the US is a coin flip')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"In the Asia basket, of all the prices that end in 4 or 8, "
            f"**{R['asia_share8']:.0f}%** end in the lucky 8 — that's *z* = "
            f"**{R['asia_z84']:+.2f}**, far past the point where you'd call it luck. In "
            f"the US control it's **{R['us_share8']:.0f}%**, a dead coin flip "
            f"(*z* = {R['us_z84']:+.2f}). And it tracks how strong the superstition is: "
            "Taiwan hardest, mainland China strong, cosmopolitan Hong Kong basically not "
            "at all.\n\n"
            "**So the belief is real. Now — does it move returns? Here's every 4/4.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    m = st.market_date_stats(CA, 4, 4, tickers=data.CALENDAR_CORE)\n"
            "    names = ['EWT', 'EWH', 'MCHI', 'POOLED']\n"
            "    means = [m[t]['mean']*1e4 for t in names]; ts = [m[t]['t'] for t in names]\n"
            "else:\n"
            "    names = ['EWT', 'EWH', 'MCHI', 'POOLED']\n"
            "    means = [R['ewt_mean'], R['ewh_mean'], R['mchi_mean'], R['pool_mean']]\n"
            "    ts = [R['ewt_t'], R['ewh_t'], R['mchi_t'], R['pool_t']]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(names, means, color=[GREY, GREY, GREY, AMBER])\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i, v in enumerate(means): ax.annotate(f'{v:+.1f}', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('mean 4/4 session return (bps)')\n"
            "ax.set_title('Every 4/4 bar is ABOVE zero — the opposite of a sell-off')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('pooled 4/4:', round(means[-1],1), 'bps, t =', round(ts[-1],2))"
        ),
        md(
            f"Every market's 4/4 return is **positive**. The pooled average is "
            f"**{R['pool_mean']:+.1f} bps** with *t* = **{R['pool_t']:+.2f}** — not just "
            "insignificant, but pointing the *wrong way* for the folklore. Maybe 4/4 "
            "just happens to be an unremarkable day? That's exactly what a placebo "
            "checks: draw the same number of *random* days thousands of times and see "
            "where 4/4 lands."
        ),
        code(
            "if HAVE_REAL:\n"
            "    m = st.market_date_stats(CA, 4, 4, tickers=data.CALENDAR_CORE)\n"
            "    obs = m['POOLED']['mean']; n_per = {t: m[t]['n'] for t in data.CALENDAR_CORE}\n"
            "    pl = st.placebo_pvalue(CA, obs, n_per, tickers=data.CALENDAR_CORE,\n"
            "                           n_seeds=20, n_draws_per_seed=250)\n"
            "    draws = np.random.default_rng(744).normal(pl['placebo_mean'], pl['placebo_sd'], 5000)*1e4\n"
            "    obs_bps = obs*1e4\n"
            "else:\n"
            "    draws = np.random.default_rng(744).normal(R['pl_mean']/1e4, R['pl_sd']/1e4, 5000)*1e4\n"
            "    obs_bps = R['pl_obs']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='random-calendar days (same sample size)')\n"
            "ax.axvline(obs_bps, c=AMBER, lw=2.4, label=f'observed 4/4 mean {obs_bps:+.1f} bps')\n"
            "ax.set_xlabel('mean return of a random-day draw (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'4/4 sits at the 75th percentile of random days (p_left = {R[\"pl_p\"]:.3f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"4/4 sits at the **{int(R['pl_p']*100)}th percentile** of the random-day "
            "cloud — i.e. a random calendar of the same size *underperforms* 4/4 about "
            "three times out of four. There is simply no underperformance to find.\n\n"
            "**And the trade?** For completeness, short 4/4 (bet on the drop), net of costs:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    tr = st.short_the_unlucky_day(CA, tickers=data.CALENDAR_CORE)\n"
            "    g, n = tr['gross_mean']*1e4, tr['mean']*1e4\n"
            "else:\n"
            "    g, n = R['short_gross'], R['short_net']\n"
            "fig, ax = plt.subplots(figsize=(7.2, 4.2))\n"
            "ax.bar(['gross', 'net of costs'], [g, n], color=[GREY, RED], width=.5)\n"
            "for i, v in enumerate([g, n]): ax.annotate(f'{v:+.1f}', (i, v), ha='center', va='top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('short-the-4/4 P&L per event (bps)')\n"
            "ax.set_title('Betting against 4/4 loses money before you even pay costs')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"The short is underwater **gross** ({R['short_gross']:+.1f} bps — the day "
            f"rose), and costs drag it to **{R['short_net']:+.1f} bps** net. The tradable "
            "folklore is a mirage."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The 4/4 date does not underperform. Pooled "
            f"**{R['pool_mean']:+.1f} bps**, *t* = **{R['pool_t']:+.2f}**, every market "
            f"positive, placebo *p* = {R['pl_p']:.3f}. No return footprint at all.\n"
            "- **Tradability — Mirage.** Shorting 4/4 loses "
            f"**{R['short_net']:+.1f} bps** per event net.\n"
            "- **Do Asian prices dodge the 4? — Confirmed.** The trailing digit really "
            f"does avoid 4 and prefer 8 (*z* = **{R['asia_z84']:+.2f}**), with a flat US "
            "control. The superstition lives in *where prices settle*, not in *when they "
            "move*."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **This is the cleanest split on the desk.** One belief, two footprints: a "
            "*real, confirmed* one in the microstructure (the digits) and a *nonexistent* "
            "one in the returns (the date). A belief can be genuinely, measurably true "
            "about prices and pure fiction about profits at the same time.\n"
            "- **Sibling folklore studies:** the "
            "[Eurovision effect](../../708-eurovision-effect/) and "
            "[plane-crash effect](../../707-plane-crash-effect/) (same event-study "
            "machinery), the [Super Bowl indicator](../../158-super-bowl/) and "
            "[Olympic years](../../234-olympic-year/) (calendar folklore) — every one "
            "tested the same honest way.\n\n"
            "*Think a tetraphobic date really moves markets? Try intraday data around a "
            "4/4 open, or the full '4-days' set (the 4th, 14th, 24th), or Lunar-calendar "
            "unlucky dates — and show a net, replicated, placebo-surviving edge. We'll "
            "publish the teardown.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 -- FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Tetraphobia — a quantitative teardown 🔬\n"
            "### A one-proportion z-battery on trailing digits (by region, vs a US "
            "placebo) · a 4/4 one-sample-*t* event study · a random-calendar placebo · "
            "an 8/8 contrast · a costed short · two synthetic controls\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious]"
            "(01_for_the_curious.ipynb). The superstition has two testable halves: a "
            "**price-clustering** claim with a genuine academic anchor (Brown & Mitchell "
            "2008 — Chinese prices avoid a trailing 4, prefer 8), and a **calendar-return** "
            "claim (the date 4/4 underperforms) that is pure folklore. The job here is to "
            "certify the first on live tape and falsify the second honestly.\n\n"
            "> ⚠️ **Data note.** Clustering = **raw, un-adjusted, local-currency** closes "
            "(an adjustment destroys the traded last digit → **price-only** by design), "
            "10 Asian + 10 US tickers, 2010→2026-06-30. Calendar = **total-return** "
            "closes, 6 China-sphere ETFs, one panel, 2000→2026-06-30. The **US basket is "
            "the placebo**; Korea contributes no trailing 4/8 digit (won-priced). Methods "
            "in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fp"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | 4/4 pooled **{R['pool_mean']:+.1f} bps**, "
            f"*t* = **{R['pool_t']:+.2f}** (wrong sign), every market positive, "
            f"random-calendar placebo *p* = **{R['pl_p']:.3f}** |\n"
            f"| **Tradability** | `MIRAGE` | short-4/4 net **{R['short_net']:+.1f} bps** "
            f"/event, *t* = {R['short_t']:+.2f} |\n"
            f"| **Do Asian prices dodge the 4?** | `CONFIRMED` | Asia *z*(8>4) = "
            f"**{R['asia_z84']:+.2f}** (Taiwan {R['reg_z84']['Taiwan']:+.2f}, China "
            f"{R['reg_z84']['ChinaA']:+.2f}); US control {R['us_z84']:+.2f} |\n\n"
            "> 💡 In plain words: the belief is bulletproof in the microstructure and "
            "invisible in the returns. Two footprints, opposite verdicts, one superstition."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "**Clustering.** Let the trailing digit of a raw price $P$ be "
            "$d(P) = \\lfloor 100P \\rceil \\bmod 10$. Round-number clustering makes "
            "$d \\in \\{0, 5\\}$ over-frequent in *every* market, so the tetraphobia "
            "test conditions on the non-round digits and asks a single one-proportion "
            "question: among prices with $d \\in \\{4, 8\\}$, is\n\n"
            "$$\\Pr[d = 8 \\mid d \\in \\{4,8\\}] > \\tfrac12 \\quad\\Longleftrightarrow\\quad "
            "z = \\frac{n_8 - n_4}{\\sqrt{n_8 + n_4}} > 0\\,?$$\n\n"
            "The **US basket, run through the identical statistic, is the placebo** — it "
            "must be flat, or the Asian asymmetry is generic microstructure, not culture.\n\n"
            "**Calendar.** For year $y$, let $r_y$ be the close-to-close return of the "
            "first session on/after 4 April. Because the date is fixed and public years "
            "ahead, there is **no execution lag**. Each year is independent and "
            "non-overlapping, so the primary statistic is the **one-sample t** of "
            "$\\{r_y\\}$ (per market and pooled). Claims:\n\n"
            "- **H1 (clustering).** $z(8{>}4) > 0$ in Asia, $\\approx 0$ in the US.\n"
            "- **H2 (date effect).** $E[r_y] < 0$ on 4/4 (underperformance).\n"
            "- **H3 (lucky contrast).** $E[r]_{8/8} > E[r]_{4/4}$.\n\n"
            "We find **H1 strongly supported**, **H2 rejected** (the sign is *positive*), "
            "**H3 directionally true but insignificant**."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "The two halves need different units. Clustering is a **one-proportion "
            "z-test** on ~40k prices per basket — enormous power, so the interesting "
            "question is *effect size and the placebo*, not significance. The calendar "
            "test is **tiny-n** (≈15-26 events per market): the right unit is a "
            "one-sample *t* across years, a Wilson interval on the hit rate, and a "
            "**random-calendar placebo** (same-size draws of random days) because with "
            "n≈25 a naive *t* is fragile. Both detectors get a **synthetic positive "
            "control** — a planted-digit-bias world and a planted-4/4-dip world — so we "
            "know the machinery fires when there IS something and stays quiet when there "
            "isn't."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Clustering.** Trailing digit of every raw local-currency close, Asia "
            "(Taiwan/HK/China/Korea, 10 tickers) vs US control (10 tickers), 2010→as-of. "
            "Exclude digits 0/5; headline $z(8{>}4)$; by-region cut; US placebo.\n"
            "- **Calendar.** 4/4 session return, EWT/EWH/MCHI (core) + EWY/EWJ/FXI "
            "(extended), 2000→2025. One-sample *t* per market and pooled; Wilson hit "
            "rate.\n"
            "- **Robustness.** 20×250 random-calendar placebo (left tail); 8/8 lucky "
            "contrast + Welch; short-the-day P&L net of 2×5 bps + 1 bp borrow.\n"
            "- **Controls.** Synthetic digit stream (tunable 4→8 bias) and synthetic "
            "daily tape (tunable 4/4 dip); nulls checked over 20 seeds.\n"
            "- **Kill criterion.** Signal is `NONE` unless 4/4 underperforms with "
            "*t* ≤ −2 and a placebo *p* < 0.05; the clustering axis is `CONFIRMED` only "
            "if Asia clears \\|z\\| ≥ 2 **and** the US placebo does not."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Trailing-digit distribution — Asia vs the US placebo\n\n"
            "Round digits 0/5 dominate both baskets (universal). The tell is the 4-vs-8 "
            "asymmetry among the rest."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ap = st.digit_pct(st.trailing_digit_counts(ASIA))\n"
            "    up = st.digit_pct(st.trailing_digit_counts(USC))\n"
            "else:\n"
            "    ap, up = np.array(R['asia_pct']), np.array(R['us_pct'])\n"
            "x = np.arange(10); w = 0.4\n"
            "fig, ax = plt.subplots(figsize=(10, 4.4))\n"
            "ax.bar(x - w/2, ap, w, label='Asia basket', color=RED, alpha=.85)\n"
            "ax.bar(x + w/2, up, w, label='US control', color=GREY, alpha=.85)\n"
            "ax.axhline(10, ls=':', c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xlabel('trailing digit'); ax.set_ylabel('share (%)')\n"
            "ax.set_title('Log-scale the small bars and 8 > 4 only in Asia'); ax.set_yscale('log')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('Asia 4/8:', round(ap[4],2), round(ap[8],2), '| US 4/8:', round(up[4],2), round(up[8],2))"
        ),
        md(
            f"> 💡 In plain words: on a log axis the round-number towers (0, 5) stop "
            "hiding the story. In Asia the 8-bar clears the 4-bar; in the US they're "
            "level. The next cell tests that asymmetry directly."
        ),
        md("### 4b · The headline z-test, and the region cut"),
        code(
            "if HAVE_REAL:\n"
            "    reg = st.region_tetraphobia(CL)\n"
            "    order = ['Taiwan', 'ChinaA', 'HongKong', 'US']\n"
            "    zs = [reg[r]['z8_gt_4'] for r in order]\n"
            "else:\n"
            "    order = ['Taiwan', 'ChinaA', 'HongKong', 'US']\n"
            "    zs = [R['reg_z84'][r] for r in order]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "cols = [RED if z >= 2 else GREY for z in zs]\n"
            "ax.bar(order, zs, color=cols, width=.6)\n"
            "ax.axhline(2, ls='--', c=RED, lw=1.2, label='significance bar (z=2)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i, v in enumerate(zs): ax.annotate(f'{v:+.2f}', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('z: is trailing 8 preferred over 4?')\n"
            "ax.set_title('Effect tracks the strength of the superstition; US placebo flat')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: Taiwan {R['reg_z84']['Taiwan']:+.2f}, mainland China "
            f"{R['reg_z84']['ChinaA']:+.2f} — both far past *z* = 2. Internationalised "
            f"Hong Kong is flat ({R['reg_z84']['HongKong']:+.2f}), and the US placebo is "
            f"flat ({R['reg_z84']['US']:+.2f}). Whole-basket Asia is "
            f"**{R['asia_z84']:+.2f}** on {R['asia_N']:,} closes. This is a real, "
            "replicated Brown-&-Mitchell footprint — **third axis = CONFIRMED**. (It is a "
            "*price-clustering* fact, not a tradable edge — nobody profits from where the "
            "last digit lands.)"
        ),
        md(
            "### 4c · The 4/4 event study — one-sample t, per market and pooled"
        ),
        code(
            "if HAVE_REAL:\n"
            "    m = st.market_date_stats(CA, 4, 4, tickers=data.CALENDAR_CORE)\n"
            "    me = st.market_date_stats(CA, 4, 4, tickers=data.CALENDAR_ALL)['POOLED']\n"
            "    rows = [(t, m[t]['n'], m[t]['mean']*1e4, m[t]['t']) for t in ['EWT','EWH','MCHI','POOLED']]\n"
            "    rows.append(('POOLED6', me['n'], me['mean']*1e4, me['t']))\n"
            "else:\n"
            "    rows = [('EWT', R['ewt_n'], R['ewt_mean'], R['ewt_t']),\n"
            "            ('EWH', R['ewh_n'], R['ewh_mean'], R['ewh_t']),\n"
            "            ('MCHI', R['mchi_n'], R['mchi_mean'], R['mchi_t']),\n"
            "            ('POOLED', R['pool_n'], R['pool_mean'], R['pool_t']),\n"
            "            ('POOLED6', R['pool6_n'], R['pool6_mean'], R['pool6_t'])]\n"
            "for r in rows: print(f'{r[0]:8s} n={r[1]:3d}  mean={r[2]:+6.1f} bps  t={r[3]:+.2f}')\n"
            "labels = [r[0] for r in rows]; means = [r[2] for r in rows]; ts = [r[3] for r in rows]\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(8.8, 6.4), sharex=True,\n"
            "                             gridspec_kw={'height_ratios': [2, 1]})\n"
            "a1.bar(labels, means, color=[RED if t<=-2 else GREY for t in ts])\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean 4/4 return (bps)')\n"
            "a1.set_title('Every cut is POSITIVE — the wrong sign for an unlucky day')\n"
            "a2.bar(labels, ts, color=[RED if abs(t)>=2 else GREY for t in ts])\n"
            "a2.axhline(-2, ls='--', c=RED, lw=1); a2.axhline(2, ls='--', c=RED, lw=1)\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('t-stat')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: the folklore predicts a *negative* 4/4 return. Every "
            f"single market delivers a *positive* one, and the pooled *t* is "
            f"**{R['pool_t']:+.2f}** (six-ETF pooled {R['pool6_t']:+.2f}). You cannot "
            "reject zero, and the point estimate is on the wrong side of it. **H2 rejected.**"
        ),
        md(
            "### 4d · Random-calendar placebo — is 4/4 unusual at all?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    m = st.market_date_stats(CA, 4, 4, tickers=data.CALENDAR_CORE)\n"
            "    obs = m['POOLED']['mean']; n_per = {t: m[t]['n'] for t in data.CALENDAR_CORE}\n"
            "    pl = st.placebo_pvalue(CA, obs, n_per, tickers=data.CALENDAR_CORE,\n"
            "                           n_seeds=20, n_draws_per_seed=250)\n"
            "    pm, ps, pp = pl['placebo_mean']*1e4, pl['placebo_sd']*1e4, pl['p_value']\n"
            "    obs_bps = obs*1e4\n"
            "else:\n"
            "    pm, ps, pp, obs_bps = R['pl_mean'], R['pl_sd'], R['pl_p'], R['pl_obs']\n"
            "draws = np.random.default_rng(744).normal(pm, ps, 6000)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.hist(draws, bins=55, color=GREY, alpha=.85, label='random-calendar draws (same n)')\n"
            "ax.axvline(obs_bps, c=AMBER, lw=2.4, label=f'observed 4/4 {obs_bps:+.1f} bps')\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('mean return of a random-day draw (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Left-tail p = {R[\"pl_p\"]:.3f}: 4/4 is a mildly GOOD day vs random')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs_bps:+.1f} bps vs placebo {pm:+.1f} bps (sd {ps:.1f}); left-tail p={pp:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the left-tail *p* (share of random calendars that "
            f"underperform 4/4) is **{R['pl_p']:.3f}** — 4/4 beats three random days out "
            "of four. Not only is there no underperformance, 4/4 is on the mildly *lucky* "
            "side of ordinary. **Signal = NONE.**"
        ),
        md("### 4e · The 8/8 lucky-date contrast"),
        code(
            "if HAVE_REAL:\n"
            "    m44 = st.market_date_stats(CA, 4, 4, tickers=data.CALENDAR_CORE)['POOLED']\n"
            "    m88 = st.market_date_stats(CA, 8, 8, tickers=data.CALENDAR_CORE)['POOLED']\n"
            "    w = st.welch_t(st.pooled_returns(CA,8,8), st.pooled_returns(CA,4,4))\n"
            "    v44, v88 = m44['mean']*1e4, m88['mean']*1e4\n"
            "else:\n"
            "    v44, v88, w = R['pool_mean'], R['e88_pool_mean'], R['welch_88_44']\n"
            "fig, ax = plt.subplots(figsize=(7.2, 4.3))\n"
            "ax.bar(['4/4 (unlucky)', '8/8 (lucky)'], [v44, v88], color=[RED, GREEN], width=.5)\n"
            "for i, v in enumerate([v44, v88]): ax.annotate(f'{v:+.1f}', (i, v), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('pooled mean return (bps)')\n"
            "ax.set_title(f'8/8 edges 4/4 (Welch t = {w:+.2f}) — right direction, deep in the noise')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: 8/8 ({R['e88_pool_mean']:+.1f} bps) does beat 4/4 "
            f"({R['pool_mean']:+.1f} bps) — the folklore's direction — but the Welch *t* "
            f"of the difference is **{R['welch_88_44']:+.2f}**, and neither date is "
            "individually significant. A cute coincidence, not an effect. **H3 "
            "directional but not certified.**"
        ),
        md(
            "### 4f · Tradability — short the unlucky day, net of costs"
        ),
        code(
            "if HAVE_REAL:\n"
            "    tr = st.short_the_unlucky_day(CA, tickers=data.CALENDAR_CORE)\n"
            "    g, n, tt = tr['gross_mean']*1e4, tr['mean']*1e4, tr['t']\n"
            "else:\n"
            "    g, n, tt = R['short_gross'], R['short_net'], R['short_t']\n"
            "fig, ax = plt.subplots(figsize=(7.2, 4.2))\n"
            "ax.bar(['gross', 'net @5bps+borrow'], [g, n], color=[GREY, RED], width=.5)\n"
            "for i, v in enumerate([g, n]): ax.annotate(f'{v:+.1f}', (i, v), ha='center', va='top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('short-4/4 P&L per event (bps)')\n"
            "ax.set_title(f'Short bleeds gross AND net (t = {tt:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'short 4/4: gross {g:+.1f} bps, net {n:+.1f} bps, t={tt:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: because 4/4 tends to *rise*, a short is underwater "
            f"before costs ({R['short_gross']:+.1f} bps) and worse after "
            f"({R['short_net']:+.1f} bps, *t* = {R['short_t']:+.2f}). There is no side of "
            "this date to be on. **Tradability = MIRAGE.**"
        ),
        md(
            "### 4g · Faithful-engine & power controls\n\n"
            "Two synthetic worlds, nulls over 20 seeds. The digit detector on a stream "
            "with a tunable 4→8 bias; the 4/4 detector on a tape with a tunable planted dip."
        ),
        code(
            "dig_null = np.array([st.synthetic_digit_detect(0.0, seed=744+s)['z8_gt_4'] for s in range(20)])\n"
            "dig_p03 = st.synthetic_digit_detect(0.03)['z8_gt_4']; dig_p05 = st.synthetic_digit_detect(0.05)['z8_gt_4']\n"
            "cal_null = np.array([st.synthetic_calendar_detect(0.0, seed=744+s)['t'] for s in range(20)])\n"
            "cal_d1 = st.synthetic_calendar_detect(-0.01)['t']; cal_d2 = st.synthetic_calendar_detect(-0.02)['t']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.scatter(np.linspace(-.12,.12,20), dig_null, color=GREY, s=36, label='null (bias=0) x20')\n"
            "a1.scatter([1], [dig_p03], color=AMBER, s=90, zorder=5, label='planted 0.03')\n"
            "a1.scatter([2], [dig_p05], color=RED, s=90, zorder=5, label='planted 0.05')\n"
            "a1.axhline(2, ls='--', c=RED, lw=1); a1.axhline(-2, ls='--', c=RED, lw=1)\n"
            "a1.set_xticks([0,1,2]); a1.set_xticklabels(['null x20','0.03','0.05'])\n"
            "a1.set_ylabel('z(8>4)'); a1.set_title('Digit detector'); a1.legend(fontsize=8)\n"
            "a2.scatter(np.linspace(-.12,.12,20), cal_null, color=GREY, s=36, label='null (dip=0) x20')\n"
            "a2.scatter([1], [cal_d1], color=AMBER, s=90, zorder=5, label='planted -1%')\n"
            "a2.scatter([2], [cal_d2], color=RED, s=90, zorder=5, label='planted -2%')\n"
            "a2.axhline(2, ls='--', c=RED, lw=1); a2.axhline(-2, ls='--', c=RED, lw=1)\n"
            "a2.set_xticks([0,1,2]); a2.set_xticklabels(['null x20','-1%','-2%'])\n"
            "a2.set_ylabel('one-sample t'); a2.set_title('4/4 detector'); a2.legend(fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'digit null mean z={dig_null.mean():+.2f} (|z|>=2 in {(abs(dig_null)>=2).sum()}/20); '\n"
            "      f'planted 0.03->{dig_p03:+.2f}, 0.05->{dig_p05:+.2f}')\n"
            "print(f'cal null mean t={cal_null.mean():+.2f} (|t|>=2 in {(abs(cal_null)>=2).sum()}/20); '\n"
            "      f'planted -1%->{cal_d1:+.2f}, -2%->{cal_d2:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: both detectors sit at a standard-normal null "
            f"(digit mean z = {R['syn_dig_null_mean']:+.2f}, fires {R['syn_dig_null_fire']}/20; "
            f"calendar mean t = {R['syn_cal_null_mean']:+.2f}, fires {R['syn_cal_null_fire']}/20) "
            f"and light up hard when an effect is planted (digit "
            f"{R['syn_dig_p05']:+.2f} at bias 0.05; calendar {R['syn_cal_dip2']:+.2f} at "
            "a −2% dip). So the CONFIRMED clustering and the NONE calendar are the tape's "
            "honest answers, not a broken pipeline. *(Machinery/power check only — never "
            "cited for a real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — 4/4 pooled **{R['pool_mean']:+.1f} bps**, *t* = "
            f"**{R['pool_t']:+.2f}** (six-ETF pooled {R['pool6_t']:+.2f}), every market "
            f"positive, random-calendar placebo left-tail *p* = **{R['pl_p']:.3f}**. The "
            "predicted underperformance is not merely insignificant — it has the wrong "
            "sign.\n"
            f"- **Tradability `MIRAGE`** — shorting 4/4 loses **{R['short_net']:+.1f} bps** "
            f"per event net (*t* = {R['short_t']:+.2f}); the day tends to rise.\n"
            f"- **\"Do Asian prices dodge the 4?\" `CONFIRMED`** — trailing-digit "
            f"*z*(8>4) = **{R['asia_z84']:+.2f}** in Asia (Taiwan "
            f"{R['reg_z84']['Taiwan']:+.2f}, China A-shares {R['reg_z84']['ChinaA']:+.2f}), "
            f"a flat US placebo ({R['us_z84']:+.2f}), a real replication of Brown & "
            "Mitchell (2008). The superstition shapes where prices settle, not when "
            "returns happen — and even the confirmed half is a behavioural fact, not an edge."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general lesson: separate the two footprints of a belief.** A "
            "superstition can be *measurably real* in microstructure (the last digit a "
            "trader picks) and *completely absent* in returns (whether a date goes up). "
            "Collapsing the two — 'tetraphobia is real, therefore short 4/4' — is exactly "
            "the leap this desk exists to catch.\n"
            "- **A stronger calendar test would need power.** Intraday returns around the "
            "4/4 open, the full '4-days' family (4th/14th/24th of the month), or "
            "Lunar-calendar unlucky dates would raise n — but the sign here is wrong, not "
            "just weak, so the prior on finding anything is low.\n"
            "- **Dedup map:** [708-eurovision-effect](../../708-eurovision-effect/) and "
            "[707-plane-crash-effect](../../707-plane-crash-effect/) (same one-sample-*t* "
            "event-study engine, no clustering half); "
            "[158-super-bowl](../../158-super-bowl/), "
            "[234-olympic-year](../../234-olympic-year/) (calendar folklore, single "
            "market). None pair a confirmed microstructure footprint with an absent "
            "return footprint of the *same* belief — that contrast is this study's "
            "contribution.\n\n"
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
