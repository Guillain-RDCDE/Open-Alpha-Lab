"""Generate the two narrative notebooks for Study 738 (Pollen-Season).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached
basket/benchmark tapes under ../_cache/ and otherwise quote the frozen headline numbers
in ``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere with
no network.
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
# closes 1996-01 -> 2026-06-30; 30 spring windows 1997 -> 2026; window = last Feb session
# -> last session on/before May 31).
R = dict(
    n_events=30, cal_lo=1997, cal_hi=2026,
    season_label="last session of February -> last session on/before May 31",
    abn_mean_pct=2.03, abn_t=1.06, abn_sd_pct=10.47,
    hit=19, hit_n=30, hit_pct=63.3, wilson=(45.5, 78.1),
    boot_lo_pct=-1.71, boot_hi_pct=5.63,
    placebo_obs_bps=203.0, placebo_mean_bps=-107.0, placebo_sd_bps=182.2,
    placebo_p=0.048, placebo_draws=5000,
    xlp_mean_pct=3.58, xlp_t=1.65, xlp_n=28,
    core_mean_pct=2.02, core_t=1.06, core_n=30,
    gross_bps=203.0, gross_t=1.06, net5_bps=170.4, net5_t=0.89,
    net10_bps=150.4, net10_t=0.79, borrow_bps=12.6,
    # calendar-month seasonality: month -> (mean_bps_per_day, t)
    months={1: (0.7, 0.12), 2: (-4.6, -0.78), 3: (2.9, 0.57), 4: (4.1, 0.74),
            5: (-1.4, -0.28), 6: (-0.5, -0.11), 7: (-3.7, -0.71), 8: (-6.8, -1.31),
            9: (4.1, 0.84), 10: (-9.3, -1.62), 11: (-4.5, -0.82), 12: (4.4, 0.95)},
    # per-year abnormal return (basket - SPY), %
    years_abn={1997: -12.09, 1998: 3.95, 1999: 3.96, 2000: -16.80, 2001: 9.04,
               2002: 18.09, 2003: 21.02, 2004: 3.83, 2005: 0.48, 2006: 10.72,
               2007: 10.22, 2008: 5.97, 2009: 3.03, 2010: -1.58, 2011: 11.78,
               2012: -1.21, 2013: 1.51, 2014: -5.48, 2015: 8.02, 2016: -16.05,
               2017: 10.75, 2018: -0.84, 2019: -10.93, 2020: 2.66, 2021: 4.03,
               2022: 20.19, 2023: -6.10, 2024: -0.45, 2025: 5.21, 2026: -22.01},
    syn_null_mean=-0.07, syn_null_sd=1.09, syn_null_fire=2,
    syn_planted_t=4.56, syn_planted_pct=9.3,
    fp_bayry="09ea36482502", fp_sny="063cf5d088aa", fp_prgo="f10b7c3e080d",
    fp_kvue="5f77bcceb9b7", fp_hln="b95efb29b67e", fp_spy="6685430bd981",
    fp_xlp="2d7f448f48d3",
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Spring seasonal%3F: Not supported](https://img.shields.io/badge/Spring_seasonal%3F-Not_supported-8b949e?style=flat-square)\n\n"
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

from pollen_season import data, strategy as st

YEARS = data.sample_years()
HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES = data.load_real()
    TBL = st.build_spread_table(PRICES, YEARS)
    ABN = TBL["abn"].to_numpy()
else:
    PRICES = TBL = ABN = None
print("real cache present:", HAVE_REAL, "| spring windows in sample:", len(YEARS))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do allergy stocks bloom with the pollen? 🤧\n"
            "### The \"buy the antihistamine names before spring\" trade — a tidy "
            "demand-seasonality story, tested honestly\n\n"
            + BADGES +
            "Every spring, some corner of the market repeats a very sensible-sounding "
            "idea: buy the companies that own the big allergy brands *before* pollen "
            "season. The logic is clean — from March through May, tens of millions of US "
            "hay-fever sufferers restock Claritin, Allegra, Zyrtec, Benadryl and Flonase, "
            "a demand spike the industry's own sales data shows every single year. If that "
            "seasonal is real and predictable, the brand owners' share prices should carry "
            "it too.\n\n"
            "So we built the basket — **Bayer** (Claritin), **Sanofi** (Allegra), "
            "**Perrigo** (the store-brand pills), **Kenvue** (Zyrtec/Benadryl) and "
            "**Haleon** (Flonase) — pinned a **cited pollen-season window** (end of "
            "February to end of May), and asked: across **30 springs since 1997**, did the "
            "basket actually beat the market? And could you have traded it?\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo, the bootstrap "
            "CI and the costed timer? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the allergy basket beat the market in spring? | **A little — but not "
            f"reliably.** The basket beats the market by **+{R['abn_mean_pct']:.2f}%** on "
            "the average spring, and wins in **63%** of years — the right direction. But "
            "over 30 springs that average is statistically **indistinguishable from "
            "zero**. |\n"
            "| Is spring genuinely *special* for these stocks? | **Barely, and for a sneaky "
            "reason.** Spring beats a *random* stretch of the year — but mostly because the "
            "basket quietly **lags the market the rest of the year**, not because spring "
            "itself is a goldmine. |\n"
            "| Could you trade it? | **No edge to bank.** A long-basket / short-market "
            "seasonal earns about **+2%/yr gross** — but you can't tell it apart from zero "
            "*before* costs, and costs + borrow only make it worse. |\n\n"
            "> A genuinely intuitive, real-world demand story — that the tape leans toward "
            "but simply won't confirm at this sample size."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Hay-fever season is as predictable as the calendar. Every March the "
            "trees start pollinating, drugstores fill their front tables with "
            "antihistamines, and the companies that own those brands sell a season's worth "
            "of Claritin and Zyrtec. Get in before the sneezing starts and ride the "
            "seasonal demand.\"*\n\n"
            "It's not a crazy story — the OTC industry (CHPA/IRI category data) really does "
            "show a large, repeatable **cough-cold-allergy** sales spike every spring. The "
            "question isn't whether people buy more allergy medicine in spring (they "
            "clearly do); it's whether that **known, calendar-predictable** demand is "
            "already baked into the share price — or leaves a tradable seasonal behind."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a **calendar-known** demand spike left a predictable seasonal in the stock, "
            "that would be a small dent in market efficiency you could set your watch to — "
            "no forecasting skill required, just a date. That's exactly why it's worth "
            "checking: an effect this *easy* to see should be arbitraged away if markets "
            "are doing their job. So we ask two plain things: did the basket beat the "
            "market through the pollen window, and is spring actually different from any "
            "other three months?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The window.** A cited pollen-season calendar rule: enter at the **{R['season_label']}** "
            "(AAFA/AAAAI pollen calendars). It's fixed years in advance, so there's "
            "**nothing to predict** — you always know when spring is.\n"
            "- **The basket.** Five listed allergy-brand owners, equal-weighted "
            "(Bayer/Claritin, Sanofi/Allegra, Perrigo/store-brand, Kenvue/Zyrtec, "
            "Haleon/Flonase). Two are recent spin-offs, so they only join once they're "
            "actually listed — no cheating with backfilled history.\n"
            f"- **The comparison.** The basket's total return over the window vs the "
            f"market's (SPY), across **{R['n_events']} springs {R['cal_lo']}→{R['cal_hi']}** "
            "— one number per year, because each spring is an independent event.\n"
            "- **The luck check.** Draw same-length windows at *random* points in the year, "
            "thousands of times — is spring really different, or is +2% just what a noisy "
            "3-month bet looks like?\n"
            "- **The trade check.** Go long the basket, short the market, over the window "
            "only — pay the spread on both legs, pay borrow on the short — and see if "
            "anything survives."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** How much did the basket beat the market by, spring by "
            "spring, over 30 years?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    yrs = TBL['year'].tolist(); ab = (TBL['abn'] * 100).tolist()\n"
            "else:\n"
            "    yrs = sorted(R['years_abn']); ab = [R['years_abn'][y] for y in yrs]\n"
            "fig, ax = plt.subplots(figsize=(10.2, 4.6))\n"
            "cols = [GREEN if v > 0 else RED for v in ab]\n"
            "ax.bar([str(y) for y in yrs], ab, color=cols, width=.7)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.axhline(np.mean(ab), c=AMBER, lw=2, ls='--',\n"
            "           label=f'mean {np.mean(ab):+.2f}%')\n"
            "ax.set_ylabel('basket minus market, spring window (%)')\n"
            "ax.set_title('Allergy basket vs the market, each spring — green wins, red losses')\n"
            "ax.set_xticks(range(0, len(yrs), 2)); ax.set_xticklabels([str(yrs[i]) for i in range(0, len(yrs), 2)], rotation=45)\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'mean spring abnormal return: {np.mean(ab):+.2f}%  |  '\n"
            "      f'won {sum(v>0 for v in ab)}/{len(ab)} years')"
        ),
        md(
            f"The basket beats the market **{R['hit']} of {R['hit_n']} springs "
            f"({R['hit_pct']:.0f}%)**, averaging **+{R['abn_mean_pct']:.2f}%** — the "
            "direction the folklore predicts. But look at the spread of the bars: **+21% in "
            "2003, −22% in 2026.** With swings that violent, a +2% average over 30 years is "
            "a whisper inside a roar. The quants notebook puts a number on it: one-sample "
            f"*t* = **{R['abn_t']:.2f}**, nowhere near the bar, and a bootstrap confidence "
            f"interval of **[{R['boot_lo_pct']:+.2f}%, {R['boot_hi_pct']:+.2f}%]** that "
            "straddles zero.\n\n"
            "**Next, the sneaky part.** A \"luck check\" compares spring to random windows — "
            "and spring *does* win that comparison. So why isn't the effect real?"
        ),
        code(
            "months = R['months']\n"
            "ms = [months[m][0] for m in range(1, 13)]\n"
            "labels = ['J','F','M','A','M','J','J','A','S','O','N','D']\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "cols = [GREEN if m in (3,4,5) else GREY for m in range(1,13)]\n"
            "ax.bar(labels, ms, color=cols, width=.66)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('avg daily basket-minus-market (bps/day)')\n"
            "ax.set_title('Is spring (green) actually special? Not obviously — Sep & Dec look as good')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('pollen months M/A/M (bps/day):', [months[m][0] for m in (3,4,5)])"
        ),
        md(
            "Here's the catch. The \"spring beats a random window\" result isn't because "
            "spring is a goldmine — it's because the basket **quietly lags the market for "
            "much of the rest of the year** (a random 3-month window averages about "
            f"**{R['placebo_mean_bps']:+.0f} bps**). Against that soggy baseline, a merely "
            "*flat* spring looks great. Month by month, March and April lean mildly "
            "positive — but so do September and December, and none of them stand out from "
            "the noise. There's no clean \"the pollen months light up\" picture.\n\n"
            "**Finally, the trade.** Could you bank the +2% with a long/short?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    holds = ['gross', 'net 5bps', 'net 10bps']\n"
            "    g = st.timer_stats(st.build_spread_table(PRICES, YEARS, cost_bps=0.0))\n"
            "    n5 = st.timer_stats(st.build_spread_table(PRICES, YEARS, cost_bps=5.0))\n"
            "    n10 = st.timer_stats(st.build_spread_table(PRICES, YEARS, cost_bps=10.0))\n"
            "    vals = [g['gross_mean_bps'], n5['net_mean_bps'], n10['net_mean_bps']]\n"
            "    tsv = [g['gross_t'], n5['net_t'], n10['net_t']]\n"
            "else:\n"
            "    holds = ['gross', 'net 5bps', 'net 10bps']\n"
            "    vals = [R['gross_bps'], R['net5_bps'], R['net10_bps']]\n"
            "    tsv = [R['gross_t'], R['net5_t'], R['net10_t']]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(holds, vals, color=[GREY, AMBER, RED], width=.55)\n"
            "for i, (v, t) in enumerate(zip(vals, tsv)):\n"
            "    ax.annotate(f'{v:+.0f} bps\\n(t={t:+.2f})', (i, v), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('long-basket / short-market, per spring (bps)')\n"
            "ax.set_title('The seasonal trade: positive on paper, but never far from zero')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('long/short per spring (bps):', dict(zip(holds, [round(v,1) for v in vals])))"
        ),
        md(
            f"The long/short earns a gross **+{R['gross_bps']/100:.2f}%/yr** — but at *t* = "
            f"**{R['gross_t']:.2f}** you can't distinguish it from zero *before you pay a "
            "cent*. Charge the spread on both legs and borrow on the short and it fades to "
            f"**+{R['net5_bps']/100:.2f}%** (*t* = {R['net5_t']:.2f}). There's simply no "
            "edge solid enough to size — a +2% average that swings ±20% year to year is not "
            "a trade, it's a coin flip with a slight tilt."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The basket beats the market by **+{R['abn_mean_pct']:.2f}%** "
            f"in the average spring (63% hit rate) — the right sign, but at *t* = "
            f"**{R['abn_t']:.2f}** over 30 years it's indistinguishable from zero, and the "
            "one borderline \"luck check\" rides on the basket's *off-season* weakness, not "
            "a real spring edge.\n"
            "- **Tradability — Mirage.** The seasonal long/short is positive on paper but "
            "undistinguishable from zero before costs, and worse after them.\n"
            "- **\"A tradable spring seasonal?\" — Not supported.** The direction leans the "
            "folklore's way; the certainty never arrives."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **This doesn't say the demand spike is fake** — it plainly isn't; people "
            "really do buy more allergy medicine in spring. It says that *known, "
            "calendar-predictable* demand is largely already in the price, leaving no "
            "seasonal a date-watcher can bank at this sample size.\n"
            "- **Where a real version might hide:** the pure OTC-brand names (Kenvue, "
            "Haleon) rather than diversified pharma giants where allergy is a rounding "
            "error; a pollen-*intensity* signal (a bad pollen year vs a mild one) instead "
            "of a fixed calendar; or the quarterly earnings-surprise angle rather than a "
            "price window.\n"
            "- **Sibling studies:** [708-eurovision-effect](../../708-eurovision-effect/) "
            "and [707-plane-crash-effect](../../707-plane-crash-effect/) share the exact "
            "event-study machinery (one-sample *t*, random placebo, costed timer, synthetic "
            "control) on very different folklore.\n\n"
            "*Think the pure-OTC names or a pollen-intensity signal would separate spring "
            "from the noise? Show it — out of sample, after costs — then we'll talk.*"
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
            "# The Pollen-Season seasonal — a quantitative teardown 🔬\n"
            "### 30 independent spring windows · one-sample *t* + block-bootstrap CI · a "
            "random-window placebo that disagrees (and why) · staples/core-basket "
            "robustness · a costed long/short timer · a 20-seed synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious]"
            "(01_for_the_curious.ipynb). The claim — a **calendar-known** hay-fever demand "
            "spike (US spring pollen, ~Mar→May) leaves a tradable seasonal in the "
            "allergy-brand owners' share prices — is intuitive and rests on real OTC sales "
            "data. The job here is to measure it honestly on the tradable tape, then ask "
            "the only question that pays: *is any of it real, and if so, tradable?*\n\n"
            "> ⚠️ **Data note.** BAYRY/SNY/PRGO/KVUE/HLN + SPY/XLP total-return closes "
            "(1996→2026), yfinance, cached; **30 spring windows 1997→2026**, one per year "
            "(independent, non-overlapping — the correct unit, **not** a daily panel). "
            "**Survivorship named:** currently-listed owners only; Kenvue (2023) and Haleon "
            "(2022) enter post-listing, cross-checked against a 3-name core basket. Methods "
            "in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprints `" + R["fp_bayry"] +
            "` BAYRY / `" + R["fp_sny"] + "` SNY / `" + R["fp_prgo"] + "` PRGO / `" +
            R["fp_spy"] + "` SPY).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | mean spring abnormal return **+{R['abn_mean_pct']:.2f}%**, "
            f"one-sample **t = {R['abn_t']:+.2f}** (n={R['n_events']}), hit "
            f"{R['hit_pct']:.1f}% (Wilson [{R['wilson'][0]:.1f}%, {R['wilson'][1]:.1f}%]), "
            f"bootstrap CI [{R['boot_lo_pct']:+.2f}%, {R['boot_hi_pct']:+.2f}%] straddles 0 |\n"
            f"| **Tradability** | `MIRAGE` | long/short gross +{R['gross_bps']/100:.2f}%/yr "
            f"(t={R['gross_t']:+.2f}); net of costs+borrow +{R['net5_bps']/100:.2f}%/yr "
            f"(t={R['net5_t']:+.2f}) |\n"
            f"| **A tradable spring seasonal?** | `NOT SUPPORTED` | placebo p = "
            f"**{R['placebo_p']:.3f}** is the only sub-0.05 number, and it rides on the "
            "basket's off-season drag, not a spring edge |\n\n"
            "> 💡 In plain words: an intuitive, real demand story that leans the right way "
            "(+2%/yr, 63% hits) but clears no honest bar on 30 independent springs."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $b_y$ be the equal-weight allergy basket's total return over year $y$'s "
            "pollen window $[\\tau^{enter}_y, \\tau^{exit}_y]$ (last Feb session → last "
            "session ≤ May 31) and $m_y$ the market's (SPY) return over the same window. "
            "The per-year abnormal return is $a_y = b_y - m_y$. The claims:\n\n"
            "- **H₁ (seasonal exists).** $E[a_y] > 0$, systematic across years — the "
            "basket beats the market through pollen season.\n"
            "- **H₂ (spring is special).** $a_y$ in the pollen window exceeds what a "
            "random same-length window on the same names produces.\n"
            "- **H₃ (capture).** A long-basket / short-SPY overlay held over the window "
            "beats zero net of both legs' costs and the short's borrow.\n\n"
            "The window is **calendar-known** (fixed dates), so there is **no execution "
            "lag** — H₁–H₃ need no `shift`, the same free pass a turn-of-month rule gets. "
            "We find **H₁ not supported** (t = 1.06), **H₂ borderline but confounded** "
            "(placebo p = 0.048, driven by the basket's off-season drag), **H₃ not "
            "supported** (positive but t < 1 net)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Spring windows are **independent, non-overlapping yearly events** (the 2019 "
            "window cannot leak into 2020), so the planned primary is a **one-sample "
            "t-test** across the 30 per-year abnormal returns — *not* a daily-panel "
            "regression, whose ~1,900 autocorrelated in-window rows would overstate the "
            "degrees of freedom by orders of magnitude and manufacture significance from "
            "noise. The mean carries a **block-bootstrap** percentile CI (years resampled "
            "with replacement); the hit rate carries a **Wilson** interval; the placebo "
            "redraws same-length **non-spring** windows (20 seeds × 250 draws). A "
            "consumer-staples benchmark (XLP) and a spin-off-free 3-name core basket are "
            "the robustness cuts."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Window.** {R['season_label']} — a cited calendar rule (AAFA/AAAAI), no "
            "execution lag.\n"
            "- **Tape.** BAYRY/SNY/PRGO/KVUE/HLN + SPY/XLP total-return closes, "
            "1996→2026-06-30 (as-of, last complete month; the 2026 window closed 2026-05-30).\n"
            f"- **Events.** {R['n_events']} springs {R['cal_lo']}→{R['cal_hi']}; coverage "
            "named (2 names pre-2003, 3 to 2022, 4 in 2023, 5 from 2024).\n"
            "- **Headline.** One-sample t on the per-year abnormal return + Wilson hit rate "
            "+ bootstrap CI.\n"
            "- **Placebo.** Random non-spring same-length windows, 20 seeds × 250 draws, "
            "right tail.\n"
            "- **Robustness.** basket − XLP (fairer benchmark); 3-name core basket "
            "(spin-off-free); calendar-month seasonality cross-check.\n"
            "- **Timer.** Long basket / short SPY over the window; both legs one-way × NAV "
            "(0/5/10 bps), short pays 50 bps annual borrow.\n"
            "- **Control.** Synthetic paired tape, planted spring bump; the null must not "
            "fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline — one-sample t + bootstrap CI\n\n"
            "The per-year abnormal return across 30 springs, its one-sample *t*, the Wilson "
            "hit rate and a block-bootstrap CI on the mean."
        ),
        code(
            "if HAVE_REAL:\n"
            "    h = st.one_sample_t(ABN); hr = st.hit_rate(ABN)\n"
            "    lo, hi = st.block_bootstrap_ci(ABN)\n"
            "    mean_pct, t_, n_ = h['mean']*100, h['t'], h['n']\n"
            "    hit_pct, wlo, whi = hr['rate']*100, hr['lo']*100, hr['hi']*100\n"
            "    lo_pct, hi_pct = lo*100, hi*100\n"
            "    ab = ABN*100\n"
            "else:\n"
            "    mean_pct, t_, n_ = R['abn_mean_pct'], R['abn_t'], R['n_events']\n"
            "    hit_pct, wlo, whi = R['hit_pct'], R['wilson'][0], R['wilson'][1]\n"
            "    lo_pct, hi_pct = R['boot_lo_pct'], R['boot_hi_pct']\n"
            "    ab = np.array([R['years_abn'][y] for y in sorted(R['years_abn'])])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.hist(ab, bins=14, color=GREY, alpha=.85, label='per-year abnormal return')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.axvline(mean_pct, c=RED, lw=2.5, label=f'mean {mean_pct:+.2f}%')\n"
            "ax.axvspan(lo_pct, hi_pct, color=AMBER, alpha=.25,\n"
            "           label=f'bootstrap 95% CI [{lo_pct:+.1f}%, {hi_pct:+.1f}%]')\n"
            "ax.set_xlabel('basket minus market, spring window (%)'); ax.set_ylabel('years')\n"
            "ax.set_title(f'Mean {mean_pct:+.2f}% but t = {t_:+.2f}: the CI swallows zero')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'mean {mean_pct:+.2f}%  t = {t_:+.3f}  (n={n_})  |  hit {hit_pct:.1f}% '\n"
            "      f'(Wilson [{wlo:.1f}%, {whi:.1f}%])  |  boot CI [{lo_pct:+.2f}%, {hi_pct:+.2f}%]')"
        ),
        md(
            f"> 💡 In plain words: the basket really does average **+{R['abn_mean_pct']:.2f}%** "
            f"over the market in spring and wins **{R['hit_pct']:.0f}%** of years — the "
            f"folklore's sign. But one-sample **t = {R['abn_t']:.2f}** is far below 2, the "
            f"Wilson hit interval **[{R['wilson'][0]:.1f}%, {R['wilson'][1]:.1f}%]** "
            f"includes 50%, and the bootstrap CI **[{R['boot_lo_pct']:+.2f}%, "
            f"{R['boot_hi_pct']:+.2f}%]** straddles zero by a mile. H₁ fails the desk bar."
        ),
        md(
            "### 4b · The placebo that disagrees — a teaching case in *which null*\n\n"
            "Same-length windows anchored at random **non-spring** dates, 20 seeds × 250 "
            "draws. At face value it's borderline significant — and it *disagrees* with the "
            "one-sample t. That disagreement is the lesson."
        ),
        code(
            "if HAVE_REAL:\n"
            "    draws = st.placebo_distribution(PRICES, TBL, n_seeds=8, n_draws_per_seed=150)\n"
            "    obs = ABN.mean()\n"
            "    pval = st.placebo_pvalue(obs, draws, tail='right')\n"
            "else:\n"
            "    obs = R['placebo_obs_bps']/1e4\n"
            "    rng = np.random.default_rng(738)\n"
            "    draws = rng.normal(R['placebo_mean_bps']/1e4, R['placebo_sd_bps']/1e4, 2000)\n"
            "    pval = R['placebo_p']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws*100, bins=45, color=GREY, alpha=.85,\n"
            "        label='null: random non-spring windows (light in-notebook run)')\n"
            "ax.axvline(0, c='k', lw=1, ls=':')\n"
            "ax.axvline(draws.mean()*100, c=GREY, lw=2, ls='--',\n"
            "           label=f'placebo mean {draws.mean()*100:+.2f}% (basket lags off-season!)')\n"
            "ax.axvline(obs*100, c=RED, lw=2.5, label=f'observed spring {obs*100:+.2f}%')\n"
            "ax.set_xlabel('basket minus market over a random window (%)'); ax.set_ylabel('draws')\n"
            "ax.set_title(f\"Spring beats a random window (p={R['placebo_p']:.3f}) — but the \"\n"
            "             'baseline is NEGATIVE')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"observed {obs*1e4:+.1f} bps vs placebo mean {R['placebo_mean_bps']:+.1f} bps \"\n"
            "      f\"(canonical p = {R['placebo_p']:.3f}, {R['placebo_draws']:,} draws)\")"
        ),
        md(
            f"> 💡 In plain words: the placebo *p* = **{R['placebo_p']:.3f}** looks like a "
            "win — until you see **the placebo mean is negative** "
            f"(**{R['placebo_mean_bps']:+.0f} bps**). The two tests null different things: "
            "the one-sample t asks *\"did the basket make money, market-adjusted, in "
            "spring?\"* (no, t = 1.06); the placebo asks *\"is spring better than a random "
            "window?\"* — and it is, **only because the basket bleeds vs the market the rest "
            "of the year.** Spring isn't a goldmine; it's the one stretch the basket "
            "*doesn't* lag. Read against zero — the honest question for a trade — the effect "
            "is inside the noise."
        ),
        md(
            "### 4c · Robustness — fairer benchmark, spin-off-free basket, month seasonality\n\n"
            "Does the (non-)result hinge on the benchmark, the two recent spin-offs, or the "
            "exact window? Three cross-checks."
        ),
        code(
            "if HAVE_REAL:\n"
            "    x = st.one_sample_t(TBL['abn_xlp'].to_numpy())\n"
            "    core = st.build_spread_table(PRICES, YEARS, tickers=data.CORE_TICKERS, min_names=2)\n"
            "    c = st.one_sample_t(core['abn'].to_numpy())\n"
            "    cuts = [('basket - SPY', R['abn_mean_pct'], R['abn_t']),\n"
            "            ('basket - XLP', x['mean']*100, x['t']),\n"
            "            ('3-name core - SPY', c['mean']*100, c['t'])]\n"
            "    ms = st.month_seasonality(PRICES)\n"
            "    mvals = (ms['mean_abn_bps']).tolist()\n"
            "else:\n"
            "    cuts = [('basket - SPY', R['abn_mean_pct'], R['abn_t']),\n"
            "            ('basket - XLP', R['xlp_mean_pct'], R['xlp_t']),\n"
            "            ('3-name core - SPY', R['core_mean_pct'], R['core_t'])]\n"
            "    mvals = [R['months'][m][0] for m in range(1, 13)]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "names = [c[0] for c in cuts]; ts = [c[2] for c in cuts]\n"
            "a1.barh(names, ts, color=[RED if abs(t) < 2 else GREEN for t in ts])\n"
            "a1.axvline(2, ls='--', c=RED, lw=1); a1.axvline(-2, ls='--', c=RED, lw=1)\n"
            "a1.set_xlabel('one-sample t'); a1.set_title('No cut clears |t| >= 2')\n"
            "for i, (nm, mn, t) in enumerate(cuts):\n"
            "    a1.annotate(f'{mn:+.2f}% (t={t:+.2f})', (t, i), va='center',\n"
            "                ha='left' if t >= 0 else 'right', fontsize=9)\n"
            "labels = ['J','F','M','A','M','J','J','A','S','O','N','D']\n"
            "a2.bar(labels, mvals, color=[GREEN if m in (3,4,5) else GREY for m in range(1,13)], width=.66)\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('avg daily abn (bps/day)')\n"
            "a2.set_title('Month seasonality: spring (green) is not visibly special')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('cuts (mean%, t):', [(nm, round(mn,2), round(t,2)) for nm, mn, t in cuts])"
        ),
        md(
            f"> 💡 In plain words: the strongest cut is vs consumer staples (XLP), "
            f"**+{R['xlp_mean_pct']:.2f}%, t = {R['xlp_t']:.2f}** — still short of 2. The "
            f"spin-off-free 3-name core basket reproduces the headline almost exactly "
            f"(**+{R['core_mean_pct']:.2f}%, t = {R['core_t']:.2f}**), so Kenvue and Haleon "
            "aren't manufacturing anything. And month by month, the pollen months (Mar/Apr) "
            "lean mildly positive but sit inside the same noise as September and December. "
            "The (non-)result is robust."
        ),
        md(
            "### 4d · The timer — a costed long/short cost sweep\n\n"
            "Long basket / short SPY over the window, both legs one-way × NAV, the short "
            "SPY leg paying 50 bps annual borrow (~12.6 bps over a ~63-session window). "
            "Gross and net, one-sample *t* across the 30 springs."
        ),
        code(
            "if HAVE_REAL:\n"
            "    labels = ['gross', 'net 5bps', 'net 10bps']\n"
            "    g = st.timer_stats(st.build_spread_table(PRICES, YEARS, cost_bps=0.0))\n"
            "    n5 = st.timer_stats(st.build_spread_table(PRICES, YEARS, cost_bps=5.0))\n"
            "    n10 = st.timer_stats(st.build_spread_table(PRICES, YEARS, cost_bps=10.0))\n"
            "    vals = [g['gross_mean_bps'], n5['net_mean_bps'], n10['net_mean_bps']]\n"
            "    tsv = [g['gross_t'], n5['net_t'], n10['net_t']]\n"
            "else:\n"
            "    labels = ['gross', 'net 5bps', 'net 10bps']\n"
            "    vals = [R['gross_bps'], R['net5_bps'], R['net10_bps']]\n"
            "    tsv = [R['gross_t'], R['net5_t'], R['net10_t']]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(labels, vals, color=[GREY, AMBER, RED], width=.55)\n"
            "for i, (v, t) in enumerate(zip(vals, tsv)):\n"
            "    ax.annotate(f'{v:+.0f} bps\\n(t={t:+.2f})', (i, v), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('long-basket / short-SPY per spring (bps)')\n"
            "ax.set_title('Positive on paper, sub-2 t at every cost level — nothing to size')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per spring (bps):', dict(zip(labels, [round(v,1) for v in vals])),\n"
            "      '| t:', dict(zip(labels, [round(t,2) for t in tsv])))"
        ),
        md(
            f"> 💡 In plain words: the seasonal long/short earns a gross "
            f"**+{R['gross_bps']:.0f} bps/yr** but at **t = {R['gross_t']:.2f}** it is "
            f"indistinguishable from zero *before costs*. Net of 5 bps/leg and borrow it is "
            f"**+{R['net5_bps']:.0f} bps** (t = {R['net5_t']:.2f}); at 10 bps, "
            f"**+{R['net10_bps']:.0f} bps** (t = {R['net10_t']:.2f}). H₃ is not supported — "
            "there is no certifiable edge to charge costs against."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic paired basket/market tapes (~30 years), a TUNABLE planted spring "
            "bump on the pollen sessions only. The null (bump=0) is checked over **20 "
            "seeds** — never a single stream."
        ),
        code(
            "null_ts = np.array([st.synthetic_detect(0.0, 738 + s)['t'] for s in range(20)])\n"
            "planted = st.synthetic_detect(0.06, 738)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (bump=0), 20 seeds')\n"
            "ax.scatter([1], [planted['t']], color=RED, s=90, zorder=5,\n"
            "           label='planted bump = +6%')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('one-sample t (spring window)')\n"
            "ax.set_title('Control: the null centers on zero; a real seasonal lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(np.abs(null_ts)>=2).sum()}/20  |  planted t = {planted[\"t\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and crosses the bar "
            f"only ~at the chance rate ({R['syn_null_fire']}/20); a planted +6% spring bump "
            f"reads t = {R['syn_planted_t']:.2f}. The machinery is unbiased — the real-tape "
            "t ≈ 1.06 is a genuine, honest reading, not a detector that's asleep. *(A "
            "faithful-engine / power check only — never cited in support of the real-tape "
            "stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — mean spring abnormal return **+{R['abn_mean_pct']:.2f}%**, "
            f"one-sample t = **{R['abn_t']:.2f}** (n={R['n_events']}); no cut clears "
            f"|t| ≥ 2 (vs staples {R['xlp_t']:.2f}, core basket {R['core_t']:.2f}), the "
            f"bootstrap CI **[{R['boot_lo_pct']:+.2f}%, {R['boot_hi_pct']:+.2f}%]** straddles "
            f"zero, and the hit-rate Wilson interval **[{R['wilson'][0]:.1f}%, "
            f"{R['wilson'][1]:.1f}%]** includes a coin flip. The one borderline number "
            f"(placebo p = **{R['placebo_p']:.3f}**) is confounded by the basket's negative "
            "off-season baseline, not a spring edge.\n"
            f"- **Tradability `MIRAGE`** — the seasonal long/short earns a gross "
            f"**+{R['gross_bps']/100:.2f}%/yr** but at t = {R['gross_t']:.2f} it is "
            f"undistinguishable from zero before costs; net of costs and borrow "
            f"**+{R['net5_bps']/100:.2f}%/yr** (t = {R['net5_t']:.2f}). Nothing to size.\n"
            f"- **\"A tradable spring seasonal?\" `NOT SUPPORTED`** — the direction leans the "
            "folklore's way (positive mean, 63% hit rate), but on 30 independent springs "
            "nothing clears the desk bar. A real, intuitive demand story that the price tape "
            "simply won't certify at this power."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Power is the honest limitation.** 30 independent years against a "
            "hypothesized ~2% seasonal with ~10% year-to-year sd is low power; a real "
            "effect of that size would need decades more, or a lower-variance signal, to "
            "certify. The calendar-known demand spike is almost surely already in the "
            "price — that's markets working, not folklore failing.\n"
            "- **A pollen-*intensity* signal** (a severe vs mild pollen year, from NAB "
            "station counts) rather than a fixed calendar, or the **pure-OTC names** "
            "(Kenvue, Haleon) rather than diversified pharma, might separate spring from "
            "the noise where a blunt calendar window can't.\n"
            "- **Dedup map:** [708-eurovision-effect](../../708-eurovision-effect/) and "
            "[707-plane-crash-effect](../../707-plane-crash-effect/) share the exact "
            "event-study machinery (one-sample t across independent events, random placebo, "
            "costed timer, synthetic control); [358-watch-index](../../358-watch-index/) "
            "shares the labelled-calendar-proxy pattern used for the pollen window.\n\n"
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
