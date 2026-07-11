"""Generate the two narrative notebooks for Study 661 (USO-Roll-Decay).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached USO/CL=F tapes
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance USO/CL=F 2006-04-10
# -> 2026-06-30; 2 hardcoded contango-stress windows).
R = dict(
    start="2006-04-11", end="2026-06-30", years=20.22, n=5080,
    tr_uso=-80.5, tr_clf=+0.8, cagr_uso=-7.76, cagr_clf=+0.04, cagr_gap=-7.80,
    gap_mean_pct=-0.0362, gap_ann_pct=-9.13,
    naive_t=-2.11, nw5=-2.33, nw21=-1.91, nw63=-1.67,
    hit=2579, hit_pct=50.8, wilson=(49.4, 52.1),
    boot_lo=-20.2, boot_hi=-1.7, boot_p=0.0038,
    stress_n=229, stress_pct_days=4.5, stress_ann=-155.4, stress_t=-2.06,
    rest_n=4851, rest_ann=-2.23, rest_t=-0.80, welch_stress_rest=-2.03,
    stress_share=76.7,
    w09_n=146, w09_ann=-61.1, w09_t=-0.96,
    w20_n=83, w20_ann=-321.1, w20_t=-1.84,
    # April 2020 case (date -> (uso_close, uso_ret_pct, clf_close, clf_ret_pct))
    apr20=[
        ("2020-04-17", 33.68, -3.44, 18.27, -8.05),
        ("2020-04-20", 30.00, -10.93, -37.63, -305.97),
        ("2020-04-21", 22.48, -25.07, 10.01, -126.60),
        ("2020-04-22", 20.08, -10.68, 13.78, +37.66),
        ("2020-04-23", 21.12, +5.18, 16.50, +19.74),
    ],
    week_dd=-40.4,
    gross_ann=9.13, gross_vol=19.4, gross_sharpe=0.47, gross_dd=-58.8, gross_worst=-14.2,
    hac21_t=1.91, placebo_p=0.0130,
    net5_ann=7.17, net5_sharpe=0.37, net5_dd=-62.9,
    net10_ann=5.97, net10_sharpe=0.31, net10_dd=-65.2,
    ex_stress_ann=0.27, ex_stress_sharpe=0.02,
    stress_only_ann=153.4, stress_only_sharpe=2.13,
    syn_null_mean=-0.09, syn_null_sd=0.80, syn_null_fire=0,
    syn_planted_naive=-16.31, syn_planted_nw5=-7.64,
    fp_uso="b9dde5e9cdd0", fp_clf="0e2eae3d2942",
    # Persistence check (adversarial re-audit, 2026-07): is the drag a per-year structural
    # constant, or does it hinge on the two named crises and reverse once they roll off?
    years_total=21, years_neg=13, years_pos=8,
    ex2020_n=4829, ex2020_ann=-3.86, ex2020_naive_t=-1.15, ex2020_nw21_t=-1.92,
    ex2020_boot_lo=-7.80, ex2020_boot_hi=-0.11, ex2020_boot_p=0.023,
    post2020_start="2021-01-04", post2020_years=5.48, post2020_n=1378,
    post2020_tr_uso=228.7, post2020_tr_clf=45.9, post2020_cagr_gap=17.10,
    post2020_ann=14.84, post2020_naive_t=2.52, post2020_nw21_t=4.47,
    post2020_boot_lo=8.72, post2020_boot_hi=21.84, post2020_hit_pct=47.0,
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Owns_oil%3F: Busted](https://img.shields.io/badge/Owns_oil%3F-Busted-8b949e?style=flat-square)\n\n"
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

from uso_roll_decay import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    USO, CLF = data.load_real()
    DF = st.gap_frame(USO, CLF)
    STRESS = data.stress_mask(DF.index)
else:
    USO = CLF = DF = STRESS = None
print("real cache present:", HAVE_REAL, "| daily gap obs:", (0 if DF is None else len(DF)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the oil ETF in your brokerage app actually own oil? 🛢️📉\n"
            "### USO-Roll-Decay — a fund down **80%** while \"the price of oil\" was flat\n\n"
            + BADGES +
            "Crude oil roughly round-tripped over the last twenty years — the front-month price "
            "you see quoted on every news ticker is almost exactly where it started. USO, the "
            "biggest retail oil ETF, is down **four-fifths of its value** over the same stretch. "
            "Same commodity, same two decades, completely different outcome.\n\n"
            "That's the claim we test: *USO doesn't track oil — it tracks a leaky bucket that "
            "happens to be shaped like oil.* And on the single most extreme day in oil-market "
            "history, it did the opposite of what everyone assumed, too.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the bootstrap and the cost math? "
            "See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** \"Oil\" here means CL=F, the continuously-rolled NYMEX "
            "front-month futures print — the exact number every ticker calls \"the price of "
            "oil\" (no free physical-spot series exists, and this is what the folklore actually "
            "means). Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does USO actually track the oil price? | **No — not even close.** Over "
            f"**{R['years']:.0f} years** USO lost **{abs(R['tr_uso']):.0f}%** while the "
            f"quoted oil price gained **{R['tr_clf']:.1f}%**. That's a **{abs(R['cagr_gap']):.1f} "
            "percentage-point** annual gap, every year, compounding. |\n"
            "| Why? | Contango. USO holds oil *futures*, and rolling from an expiring contract "
            "to the next one usually means selling cheap and buying dear — a toll a real barrel "
            "in a tank never pays. |\n"
            f"| Is the damage spread evenly across 20 years? | **No — it's lumpy.** "
            f"**{R['stress_share']:.0f}%** of the entire two-decade gap happened in just "
            f"**{R['stress_pct_days']:.1f}%** of trading days — two historic storage crises, "
            "2009 and 2020. |\n"
            "| What happened the day oil went negative? | USO barely blinked. The front "
            "contract crashed to **-$37.63**; USO fell a completely ordinary **-10.9%** that "
            "day, because it wasn't fully holding that exact contract. |\n"
            f"| Can you get paid for knowing this? | **Not reliably.** Shorting USO against "
            "long oil futures made money on paper — but almost *all* of it came from those same "
            "two crises. Miss them and the trade earns roughly nothing. |\n"
            f"| Does the drag keep happening every year? | **No.** Split by calendar year, USO "
            f"underperformed in **{R['years_neg']}/{R['years_total']}** years — a majority, not "
            f"a rule — and for the last **{R['post2020_years']:.1f} years running** (2021 → "
            f"2026) it has done the *opposite*: USO actually **beat** the headline oil price by "
            f"**{R['post2020_ann']:+.1f}%/yr**. |\n\n"
            "> The two crisis episodes really happened and really hurt. But \"USO mechanically "
            "loses to oil\" turns out to describe two storms, not a standing law of physics — "
            "the mental model \"USO = oil\" is wrong in both directions, and so, it turns out, "
            "is \"USO always loses to oil.\""
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Buy USO, own the oil price.\"* It's on the tin — United States **Oil** Fund — "
            "and it trades on your brokerage app right next to the S&P 500 and gold. Millions "
            "of retail investors have bought it expecting exactly that: exposure to crude.\n\n"
            "The honest mechanism, steelmanned: USO holds NYMEX WTI **futures contracts**, not "
            "barrels in a tank. Every contract expires, so the fund must constantly \"roll\" — "
            "sell the contract about to expire, buy the next month out. When the market is in "
            "**contango** (later months cost more than the current one), that roll is "
            "mechanically unprofitable, over and over, regardless of what the actual oil price "
            "does — but when the curve flips to **backwardation** the exact same roll turns "
            "profitable, so how true this is depends entirely on which regime the market is in "
            "at the time, not on a fixed law."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If this is real, it means one of the most-traded commodity ETFs in the world "
            "systematically misleads its own holders about what they own — and it means the "
            "flip side, \"short the ETF, capture the toll,\" might be a genuine, repeatable "
            "trade. Billions of retail dollars have flowed through USO since 2006; a structural "
            "80% gap against the thing it's named after is not a footnote."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The race.** USO's split-adjusted price vs CL=F (the front-month oil price "
            f"everyone quotes), {R['start']} → {R['end']}, cumulative and annualized.\n"
            "- **The daily grind.** Is USO's daily return systematically a little worse than "
            "oil's, day after day — and is that reliable, not luck?\n"
            "- **The regime check.** Is the damage a steady bleed, or does it come from a "
            "handful of specific crisis episodes (2009's storage glut, 2020's COVID collapse)?\n"
            "- **The extreme case.** What actually happened to USO the day oil futures went "
            "negative — did the \"USO = oil\" story hold up when it mattered most?\n"
            "- **The trade check.** Short USO, go long oil futures, pay realistic costs and "
            "borrow — does the obvious trade actually pay, reliably?"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Twenty years, same commodity, two completely different "
            "lines."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cs = st.cumulative_stats(DF)\n"
            "    uso_cum = (1 + USO['Close'].pct_change().fillna(0)).cumprod()\n"
            "    clf_cum = (1 + CLF['Close'].pct_change().fillna(0)).cumprod()\n"
            "    tr_u, tr_c = cs['total_return_uso']*100, cs['total_return_clf']*100\n"
            "else:\n"
            "    tr_u, tr_c = R['tr_uso'], R['tr_clf']\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.6))\n"
            "if HAVE_REAL:\n"
            "    ax.plot(uso_cum.index, uso_cum.values, color=RED, lw=1.6, label='USO (fund)')\n"
            "    ax.plot(clf_cum.index, clf_cum.values, color=GREY, lw=1.6,\n"
            "            label='CL=F (\"the oil price\")')\n"
            "    ax.set_yscale('log'); ax.set_ylabel('growth of $1 (log scale)')\n"
            "    ax.legend()\n"
            "else:\n"
            "    ax.bar(['USO','CL=F (\"oil price\")'], [tr_u, tr_c], color=[RED, GREY], width=.55)\n"
            "    for i,v in enumerate([tr_u, tr_c]): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',\n"
            "        va='top' if v<0 else 'bottom')\n"
            "    ax.set_ylabel('total return (%)')\n"
            "ax.set_title('Same 20 years, same commodity, opposite outcome')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'USO total return {tr_u:+.1f}%   CL=F total return {tr_c:+.1f}%')"
        ),
        md(
            f"USO: **{R['tr_uso']:.1f}%**. The oil price: **{R['tr_clf']:+.1f}%**. That's a "
            f"**{abs(R['cagr_gap']):.2f} percentage-point** compounding gap *every single year* "
            "for two decades. This isn't a bad year or a bad decade — it's the whole sample.\n\n"
            "**Next: is it a steady leak, or does it come in bursts?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    stress_ann = DF.loc[STRESS, 'gap'].mean()*252*100\n"
            "    rest_ann = DF.loc[~STRESS, 'gap'].mean()*252*100\n"
            "else:\n"
            "    stress_ann, rest_ann = R['stress_ann'], R['rest_ann']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(['2009 + 2020\\ncrisis windows\\n(4.5% of days)','every other day\\n(95.5%)'],\n"
            "       [stress_ann, rest_ann], color=[RED, GREY], width=.55)\n"
            "for i,v in enumerate([stress_ann, rest_ann]): ax.annotate(f'{v:+.1f}%/yr',(i,v),\n"
            "    ha='center', va='top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel(\"USO's drag vs oil, annualized\")\n"
            "ax.set_title(f'{R[\"stress_share\"]:.0f}% of 20 years of damage from 4.5% of days')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'stress windows {stress_ann:+.1f}%/yr   rest of tape {rest_ann:+.2f}%/yr')"
        ),
        md(
            f"There's the shape of it: during the two named storage crises the drag runs at "
            f"**{R['stress_ann']:.0f}%/yr**; every other day it's a mild **{R['rest_ann']:.1f}%/yr** "
            f"— barely distinguishable from nothing. **{R['stress_share']:.0f}%** of the entire "
            "20-year gap happened in **4.5%** of the trading days. The \"USO bleeds a little "
            "every day\" story is only half right — most of the bleeding happened during two "
            "storms.\n\n"
            "**So does the bleeding come back once the storm passes? Check every single "
            "year.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    yearly = DF['gap'].groupby(DF.index.year).sum() * 100\n"
            "    n_neg = int((yearly < 0).sum())\n"
            "else:\n"
            "    yearly, n_neg = None, R['years_neg']\n"
            "fig, ax = plt.subplots(figsize=(10.5, 4.6))\n"
            "if yearly is not None:\n"
            "    cols = [RED if v < 0 else GREEN for v in yearly.values]\n"
            "    ax.bar(yearly.index.astype(str), yearly.values, color=cols, width=.65)\n"
            "    ax.set_xticklabels(yearly.index.astype(str), rotation=60)\n"
            "    ax.set_ylabel('cumulative log-gap that year (%)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_title(f'USO vs oil, year by year -- red in {n_neg}/{R[\"years_total\"]} '\n"
            "             'years, not every year')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'years USO underperformed: {n_neg}/{R[\"years_total\"]}')"
        ),
        md(
            f"**{R['years_neg']} of {R['years_total']}** calendar years are red — a majority, "
            "but nowhere near \"every year, like clockwork.\" And every single year from "
            f"**2021 through 2026** ({R['post2020_years']:.1f} years running) is green: USO "
            f"gained **{R['post2020_tr_uso']:+.0f}%** while CL=F gained only "
            f"**{R['post2020_tr_clf']:+.0f}%** — USO *beat* the oil price it supposedly bleeds "
            f"against, by **{R['post2020_ann']:+.1f}%/yr**, and that gap is itself statistically "
            f"real (*t* ≈ {R['post2020_naive_t']:.1f}-{R['post2020_nw21_t']:.1f}). Same "
            "mechanism, opposite sign: when the futures curve flips from contango to "
            "**backwardation** (the regime oil sat in through most of 2021-2026's "
            "supply-constrained market), the monthly roll *collects* rather than pays a toll. "
            "\"USO mechanically loses to oil\" isn't a standing law — it's a bet on which side "
            "of the curve the market happens to sit on, and the market has sat on the other "
            "side for over five years running.\n\n"
            "**Now the most dramatic single day of all — April 20, 2020, when oil went "
            "negative.**"
        ),
        code(
            "labels = [d for d,_,_,_,_ in R['apr20']]\n"
            "uso_r = [ur for _,_,ur,_,_ in R['apr20']]\n"
            "clf_r = [cr for _,_,_,_,cr in R['apr20']]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.6))\n"
            "x = np.arange(len(labels))\n"
            "ax.bar(x-0.2, uso_r, width=0.38, color=AMBER, label='USO daily return')\n"
            "ax.bar(x+0.2, clf_r, width=0.38, color=RED, label='CL=F (front WTI) daily return')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20)\n"
            "ax.set_ylabel('daily return (%)')\n"
            "ax.set_title('The day oil went negative: USO barely noticed')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for d,uc,ur,cc,cr in R['apr20']:\n"
            "    print(f'{d}: USO {ur:+7.2f}%   CL=F {cr:+8.2f}%')"
        ),
        md(
            f"On **2020-04-20** the front WTI contract cratered to **-$37.63** — the first "
            "negative oil price in history — a **-306%** one-day move. USO that same day: "
            f"**{R['apr20'][1][2]:+.1f}%**. Painful, but a completely ordinary bad day, because "
            "by then USO already held a laddered basket across contract months, not the "
            f"collapsing front contract alone. It still lost **{abs(R['week_dd']):.1f}%** "
            "cumulatively over that week — it didn't escape the storm — but the folklore image "
            "of \"USO went negative too\" is simply false. The \"USO = oil\" mental model broke "
            "in both directions on the one day it mattered most.\n\n"
            "**Finally: can you get paid for knowing all this?**"
        ),
        code(
            "labels = ['full sample\\n(net, 5 bps)', 'excluding 2009/2020\\ncrises (net)',\n"
            "          'inside crises only\\n(net, illustrative)']\n"
            "vals = [R['net5_ann'], R['ex_stress_ann'], R['stress_only_ann']]\n"
            "cols = [AMBER, RED, GREEN]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "ax.bar(labels, vals, color=cols, width=.55)\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.1f}%/yr',(i,v),ha='center',\n"
            "    va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('long-oil / short-USO book, net of costs (%/yr)')\n"
            "ax.set_title('Almost the whole trade is two crises you could not have timed')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"full sample net {R['net5_ann']:+.2f}%/yr | ex-crisis "
            "{R['ex_stress_ann']:+.2f}%/yr | crisis-only {R['stress_only_ann']:+.1f}%/yr\")"
        ),
        md(
            f"The full-sample trade looks decent: **+{R['net5_ann']:.1f}%/yr** net, "
            f"Sharpe {R['net5_sharpe']:.2f}. But take out the 2009 and 2020 crisis windows and "
            f"the net return is **+{R['ex_stress_ann']:.2f}%/yr** — statistically nothing. You'd "
            "have needed to correctly hold this trade through two once-a-decade storage crises "
            "to earn anything at all; the other 95% of the time it barely covers its own costs. "
            "That's not a repeatable edge — it's a backtest that happened to straddle two rare "
            "disasters."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** USO really has lost **{abs(R['tr_uso']):.0f}%** against a "
            f"flat oil price over {R['years']:.0f} years, and two real crisis episodes (2009, "
            "2020) really did hurt it — but that's not the same as a persistent, mechanical "
            f"per-year decay. Excluding all of 2020, the daily drag is no longer statistically "
            f"distinguishable from zero by the standard test (naive *t* = "
            f"{R['ex2020_naive_t']:.2f}); year-by-year the sign splits "
            f"**{R['years_neg']}/{R['years_total']}** negative, not \"almost every year\"; and "
            f"the most recent **{R['post2020_years']:.1f} years running** (2021-2026) went the "
            f"*other* way, USO beating oil by **{R['post2020_ann']:+.1f}%/yr** with its own real "
            "*t*-stat. The mechanism is genuine (roll yield), but it cuts both ways with the "
            "curve's shape — it is not the one-way structural tax the folklore describes.\n"
            "- **Tradability — Mirage.** The obvious \"short USO, own oil futures\" trade nets "
            "to roughly nothing once you remove the two crises that made it look good — and for "
            "over five years now it would have actively lost money.\n"
            "- **\"Does USO let you own oil?\" — Busted.** Wrong direction on the 20-year "
            "magnitude, wrong again in the other direction on the most extreme single day, and "
            "wrong in a *third* direction for the last five-plus years. USO is not a spot-oil "
            "proxy in any regime."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The general lesson is roll yield.** Any futures-based ETF (VXX/VIXY on vol, "
            "BITO on bitcoin, most commodity trackers) pays or collects a toll that has nothing "
            "to do with the price of the thing it's named after — read the prospectus before "
            "you assume \"the fund = the asset.\"\n"
            "- **Where the pros actually play it** is directly in the futures curve (calendar "
            "spreads, term-structure positioning) rather than through a retail ETP wrapper — "
            "the wrapper adds its own costs and structural constraints on top of the curve.\n"
            "- **Sibling studies:** [100-melting-ice](../../100-melting-ice/) (the general "
            "commodity-contango story), [375-vxx-roll-decay](../../375-vxx-roll-decay/) (the "
            "same mechanism in VIX futures) and [619-bito-roll-drag](../../619-bito-roll-drag/) "
            "(the same family, bitcoin) — none of them race USO against the exact number its "
            "own holders think it tracks.\n\n"
            "*Think the next storage crisis is coming and you can time it? That's a forecast, "
            "not a backtest — show it forward, out of sample, then we'll talk.*"
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
            "# USO-Roll-Decay — a quantitative teardown 🔬\n"
            "### Cumulative divergence · naive + HAC one-sample *t*'s · a circular block-"
            "bootstrap · the 2009/2020 contango-stress regime split · a year-by-year persistence "
            "check (ex-2020, post-2020) · the April-2020 case study · an honest carry-capture "
            "book with its ex-crisis decomposition · a planted-drag synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The claim — **USO structurally underperforms the front-month oil price it's built "
            "from, because of contango roll cost** — has a stated mechanism (futures-curve roll "
            "yield, Erb & Harvey 2006), a dramatic historical confirmation (2020-04-20), and a "
            "twenty-year real tape. The job here is to measure it honestly, characterize *where* "
            "the decay lives, and then ask the only question that pays: *is any of it "
            "tradable?*\n\n"
            "> ⚠️ **Data note.** USO adjusted close + CL=F front-month close (2006→2026), "
            "yfinance, cached; **2 hardcoded contango-stress windows** (2008-12→2009-06, "
            "2020-03→2020-06) from EIA storage reports. CL=F is the continuously-rolled "
            "front-month futures print — no free physical-spot series exists, and this *is* "
            "what \"the oil price\" means to the folklore. No survivorship (single live "
            "instrument on each leg). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprints `" + R["fp_uso"] +
            "` / `" + R["fp_clf"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | full-sample naive *t* = **{R['naive_t']:.2f}** looks "
            f"significant, but excluding 2020 alone drops it to **{R['ex2020_naive_t']:.2f}**; "
            f"the drag is negative in only **{R['years_neg']}/{R['years_total']}** calendar "
            f"years, and the most recent **{R['post2020_years']:.1f} years** (2021-2026) reverse "
            f"HARD positive (**{R['post2020_ann']:+.1f}%/yr**, *t* = "
            f"{R['post2020_naive_t']:.2f}-{R['post2020_nw21_t']:.2f}) |\n"
            f"| **Tradability** | `MIRAGE` | net Sharpe {R['net5_sharpe']:.2f} full-sample "
            f"collapses to **{R['ex_stress_ann']:+.2f}%/yr** (Sharpe {R['ex_stress_sharpe']:.2f}) "
            "ex the 2 named crisis windows, and runs net negative since 2021 |\n"
            "| **Owns oil?** | `BUSTED` | -80% vs flat oil over 20 yr, then +14.8%/yr *ahead* of "
            "oil since 2021; **-10.9%**, not -306%, on the day WTI settled at -$37.63 |\n\n"
            "> 💡 In plain words: two real, un-forecastable crises (2009, 2020) did almost all "
            "of the 20-year damage; strip them out, or look at any year since 2021, and the "
            "\"USO structurally loses to oil\" claim does not hold up — the roll mechanism runs "
            "in both directions depending on whether the curve is in contango or backwardation, "
            "so this is a regime-dependent wedge, not a persistent structural decay."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^{USO}_t$ and $r^{CLF}_t$ be the daily log returns of USO and the CL=F "
            "front-month print, and $g_t = r^{USO}_t - r^{CLF}_t$ the daily gap. The claims:\n\n"
            "- **H₁ (cumulative divergence).** $\\prod(1+r^{USO}_t) \\ll \\prod(1+r^{CLF}_t)$ "
            "over the full sample — large, not a rounding error.\n"
            "- **H₂ (systematic daily drag).** $E[g_t] < 0$, reliably (not just in the sample "
            "mean by chance).\n"
            "- **H₃ (regime concentration).** The drag concentrates in known contango-stress "
            "episodes rather than being i.i.d. across the sample.\n"
            "- **H₄ (the extreme case).** On the most extreme realized WTI move in history, USO "
            "does *not* mirror the front contract's move.\n"
            "- **H₅ (capture).** A costed \"long CL=F / short USO\" book banks the drag net of "
            "realistic frictions, *repeatably*.\n"
            "- **H₆ (persistence).** The per-year sign of $g_t$ is negative in most years, and "
            "the drag survives dropping the single worst episode — i.e. it is a standing "
            "structural tax, not two crisis prints.\n\n"
            "We find **H₁ and H₄ strongly supported**; **H₂-H₃ supported only *inside* the two "
            "named crisis windows** — outside them the gap is not separately certifiable; "
            "**H₅ not supported once regime-conditioned**; and **H₆ rejected**: only "
            f"{R['years_neg']}/{R['years_total']} years are negative, dropping 2020 alone drops "
            "the naive *t* below 2, and the most recent 5.5 years reverse sign with their own "
            "statistically real *t*-stat. The honest read is a regime-dependent roll wedge that "
            "hurt badly twice, not a persistent mechanical decay."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "There is no natural \"control group\" for a paired daily differential the way "
            "FOMC-day-vs-not works — every day carries a gap. So H₂ is tested with a **naive "
            "one-sample *t*** and a **Newey-West (1987) HAC one-sample *t*** at three lags "
            "(5/21/63 sessions), plus a **circular block-bootstrap** (block ≈ 21 sessions, "
            "5,000 draws) for a distribution-free CI and $P[\\bar g \\geq 0]$. H₃ (the regime "
            "split) *is* a genuine two-group comparison and uses **Welch (1947)** — the "
            "2008-12→2009-06 and 2020-03→2020-06 windows are named and cited *ex ante* (EIA "
            "storage reports), not fit to the outcome."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** USO adjusted close + CL=F front-month close, {R['start']} → "
            f"{R['end']} ({R['n']:,} paired daily observations). As-of 2026-06-30 (last "
            "complete month).\n"
            "- **Cumulative.** Total return and CAGR, USO vs CL=F.\n"
            "- **Daily drag.** naive *t* + NW(5/21/63) *t* + Wilson hit rate + block-bootstrap "
            "CI/placebo on the gap.\n"
            "- **Regime.** Welch *t*, stress (2009+2020, hardcoded) vs rest; per-window "
            "one-sample *t*; share of cumulative log-drag from stress.\n"
            "- **Persistence.** Calendar-year sign count of the gap; naive/NW/bootstrap "
            "re-estimate excluding all of 2020; a strict post-2020 (2021-01-04 onward) subsample "
            "estimate, to test whether the drag is a standing per-year constant or an artifact "
            "of the two named crisis prints.\n"
            "- **Case study.** Simple (not log) returns around 2020-04-20 — log is undefined "
            "across the sign flip, so this window is analyzed on its own, non-parametrically.\n"
            "- **Execution (third axis).** Constant-notional long-CL=F/short-USO, rebalanced "
            "monthly; 0.75%/yr borrow on the short, 2 × one-way cost × NAV on rebalance days "
            "only; gross/net Sharpe, drawdown, HAC(21) *t*, sign-shuffle placebo, and the "
            "ex-crisis vs crisis-only decomposition.\n"
            "- **Control.** Synthetic (spot, fund) pair, planted daily + stress-block drag; the "
            "null must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Cumulative divergence and the daily gap\n\n"
            "Total return / CAGR since USO's 2006-04-10 inception, then the daily gap's naive "
            "and HAC *t*'s plus the block-bootstrap CI."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cs = st.cumulative_stats(DF)\n"
            "    hs = st.headline_drag_stats(DF)\n"
            "    bb = st.block_bootstrap_mean_ci(DF['gap'].values, block=21, n_boot=2000, seed=661)\n"
            "    tr_u, tr_c, gap_ann = cs['total_return_uso']*100, cs['total_return_clf']*100, hs['ann_pct']\n"
            "    naive_t, nw5, nw21, nw63 = hs['naive_t'], hs['nw_t_5'], hs['nw_t_21'], hs['nw_t_63']\n"
            "    boot_lo, boot_hi = bb['lo']*252*100, bb['hi']*252*100\n"
            "else:\n"
            "    tr_u, tr_c, gap_ann = R['tr_uso'], R['tr_clf'], R['gap_ann_pct']\n"
            "    naive_t, nw5, nw21, nw63 = R['naive_t'], R['nw5'], R['nw21'], R['nw63']\n"
            "    boot_lo, boot_hi = R['boot_lo'], R['boot_hi']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.4))\n"
            "a1.bar(['USO','CL=F'], [tr_u, tr_c], color=[RED, GREY], width=.55)\n"
            "for i,v in enumerate([tr_u, tr_c]): a1.annotate(f'{v:+.1f}%',(i,v),ha='center',\n"
            "    va='top' if v<0 else 'bottom')\n"
            "a1.set_title('20-year total return'); a1.set_ylabel('%')\n"
            "ts = [naive_t, nw5, nw21, nw63]\n"
            "a2.bar(['naive','NW(5)','NW(21)','NW(63)'], ts,\n"
            "       color=[RED if abs(t)>=2 else AMBER for t in ts], width=.6)\n"
            "a2.axhline(-2, ls='--', c=RED, lw=1); a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_title('Daily gap t-stat by lag'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'USO {tr_u:+.1f}%  CL=F {tr_c:+.1f}%  |  gap {gap_ann:+.2f}%/yr')\n"
            "print(f'naive t={naive_t:+.2f}  NW(5)={nw5:+.2f}  NW(21)={nw21:+.2f}  NW(63)={nw63:+.2f}')\n"
            "print(f'block-bootstrap 95% CI (ann.): [{boot_lo:+.1f}%, {boot_hi:+.1f}%]')"
        ),
        md(
            f"> 💡 In plain words: the cumulative gap ({R['tr_uso']:.1f}% vs {R['tr_clf']:+.1f}%) "
            f"is enormous and the daily mean clears *t* ≥ 2 at the standard lag "
            f"(naive {R['naive_t']:.2f}, NW(5) {R['nw5']:.2f}) and via the block-bootstrap "
            f"(95% CI [{R['boot_lo']:+.1f}%, {R['boot_hi']:+.1f}%]/yr, excludes 0). Longer HAC "
            f"lags weaken (NW(21) {R['nw21']:.2f}, NW(63) {R['nw63']:.2f}) — a first hint that "
            "the process is not smoothly i.i.d., confirmed in 4b."
        ),
        md(
            "### 4b · The regime split — is it a grind or a series of shocks?\n\n"
            "Split the daily gap by the two hardcoded, cited-*ex-ante* contango-stress windows "
            "(2008-12→2009-06, 2020-03→2020-06) vs the rest of the tape."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rs = st.regime_stats(DF, STRESS)\n"
            "    stress_ann, rest_ann = rs['stress_ann_pct'], rs['rest_ann_pct']\n"
            "    stress_t, rest_t, welch_t = rs['one_t_stress'], rs['one_t_rest'], rs['welch_t_stress_vs_rest']\n"
            "    share = rs['stress_share_of_cum']*100\n"
            "else:\n"
            "    stress_ann, rest_ann = R['stress_ann'], R['rest_ann']\n"
            "    stress_t, rest_t, welch_t = R['stress_t'], R['rest_t'], R['welch_stress_rest']\n"
            "    share = R['stress_share']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(['stress windows\\n(2009+2020)','rest of tape'], [stress_ann, rest_ann],\n"
            "       color=[RED, GREY], width=.55)\n"
            "for i,(v,t_) in enumerate([(stress_ann,stress_t),(rest_ann,rest_t)]):\n"
            "    ax.annotate(f'{v:+.1f}%/yr\\n(t={t_:+.2f})',(i,v),ha='center',va='top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('daily gap, annualized')\n"
            "ax.set_title(f'{share:.0f}% of cumulative divergence from 4.5% of days '\n"
            "             f'(Welch t={welch_t:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'stress {stress_ann:+.1f}%/yr (t={stress_t:+.2f})  rest {rest_ann:+.2f}%/yr '\n"
            "      f'(t={rest_t:+.2f})  Welch t={welch_t:+.2f}  share={share:.1f}%')"
        ),
        md(
            f"> 💡 In plain words: **{R['stress_share']:.0f}%** of the entire 20-year "
            f"cumulative log-divergence comes from **{R['stress_pct_days']:.1f}%** of trading "
            f"days. Outside the two named windows the drag is **{R['rest_ann']:.2f}%/yr at "
            f"*t* = {R['rest_t']:.2f}** — not separately certifiable. This *is* the mechanism "
            "behind 4a's lag-sensitivity: the series is regime-switching, not stationary, so a "
            "single long-lag HAC estimate gets diluted by two extreme, short episodes. It also "
            f"explains the coin-flip hit rate: **{R['hit_pct']:.1f}%** of days USO "
            "underperforms — a magnitude/skew story, not a frequency one."
        ),
        md(
            "### 4c · Persistence — does the drag survive dropping the worst episode?\n\n"
            "4a-4b establish the drag is real in-sample and regime-concentrated. That is *not* "
            "the same claim as \"a structural, mechanical, per-year tax\" — this beat tests "
            "persistence directly: exclude 2020 entirely and re-run the naive/HAC/bootstrap "
            "machinery on what remains, check every calendar year's sign, then isolate the "
            "years after the last named crisis rolled off (2021-01-04 onward)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    df_ex2020 = DF[DF.index.year != 2020]\n"
            "    hs_ex = st.headline_drag_stats(df_ex2020, nw_lags=(5, 21, 63))\n"
            "    bb_ex = st.block_bootstrap_mean_ci(df_ex2020['gap'].values, block=21,\n"
            "                                        n_boot=2000, seed=661)\n"
            "    yearly = DF['gap'].groupby(DF.index.year).sum()\n"
            "    n_neg, n_tot = int((yearly < 0).sum()), len(yearly)\n"
            "    post = DF[DF.index >= R['post2020_start']]\n"
            "    hs_post = st.headline_drag_stats(post, nw_lags=(5, 21, 63))\n"
            "    cs_post = st.cumulative_stats(post)\n"
            "    ex_ann, ex_nt, ex_nw21 = hs_ex['ann_pct'], hs_ex['naive_t'], hs_ex['nw_t_21']\n"
            "    ex_lo, ex_hi = bb_ex['lo']*252*100, bb_ex['hi']*252*100\n"
            "    post_ann, post_nt, post_nw21 = hs_post['ann_pct'], hs_post['naive_t'], hs_post['nw_t_21']\n"
            "    post_tru, post_trc = cs_post['total_return_uso']*100, cs_post['total_return_clf']*100\n"
            "else:\n"
            "    ex_ann, ex_nt, ex_nw21 = R['ex2020_ann'], R['ex2020_naive_t'], R['ex2020_nw21_t']\n"
            "    ex_lo, ex_hi = R['ex2020_boot_lo'], R['ex2020_boot_hi']\n"
            "    n_neg, n_tot = R['years_neg'], R['years_total']\n"
            "    post_ann, post_nt, post_nw21 = R['post2020_ann'], R['post2020_naive_t'], R['post2020_nw21_t']\n"
            "    post_tru, post_trc = R['post2020_tr_uso'], R['post2020_tr_clf']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.4))\n"
            "labs = ['full\\nsample', 'ex-2020', 'ex-crisis\\nwindows', '2021-2026\\n(post)']\n"
            "vals = [R['gap_ann_pct'], ex_ann, R['rest_ann'], post_ann]\n"
            "cols = [RED if v < 0 else GREEN for v in vals]\n"
            "a1.bar(labs, vals, color=cols, width=.6)\n"
            "for i, v in enumerate(vals):\n"
            "    a1.annotate(f'{v:+.1f}%/yr', (i, v), ha='center', va='top' if v < 0 else 'bottom')\n"
            "a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_ylabel('daily gap, annualized'); a1.set_title('The \"decay\" shrinks, then flips')\n"
            "a2.bar(['naive t\\n(full)', 'naive t\\n(ex-2020)', 'naive t\\n(post-2020)'],\n"
            "       [R['naive_t'], ex_nt, post_nt],\n"
            "       color=[RED if abs(t) >= 2 else AMBER for t in [R['naive_t'], ex_nt, post_nt]], width=.6)\n"
            "a2.axhline(-2, ls='--', c=RED, lw=1); a2.axhline(2, ls='--', c=RED, lw=1)\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_title('Significance is not robust to dropping 2020')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'ex-2020: {ex_ann:+.2f}%/yr, naive t={ex_nt:+.2f}, NW(21) t={ex_nw21:+.2f}, '\n"
            "      f'bootstrap 95% CI [{ex_lo:+.1f}%, {ex_hi:+.1f}%]/yr')\n"
            "print(f'years USO underperformed: {n_neg}/{n_tot}')\n"
            "print(f'post-2020 (2021-01-04 -> 2026-06-30): USO {post_tru:+.1f}% vs CL=F {post_trc:+.1f}%, '\n"
            "      f'gap {post_ann:+.2f}%/yr, naive t={post_nt:+.2f}, NW(21) t={post_nw21:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: drop 2020 and the naive *t* falls from {R['naive_t']:.2f} to "
            f"**{R['ex2020_naive_t']:.2f}** — no longer separately certifiable — though the "
            f"block-bootstrap (autocorrelation-robust) still narrowly excludes zero "
            f"([{R['ex2020_boot_lo']:+.1f}%, {R['ex2020_boot_hi']:+.1f}%]/yr). Calendar-year "
            f"sign count: **{R['years_neg']}/{R['years_total']}** negative — a majority, not "
            "\"structurally every year.\" Decisive: the **post-2020 subsample** "
            f"({R['post2020_start']} → 2026-06-30, {R['post2020_years']:.1f} years, "
            f"{R['post2020_n']:,} obs) shows USO **{R['post2020_tr_uso']:+.0f}%** vs CL=F "
            f"**{R['post2020_tr_clf']:+.0f}%** — a gap of **{R['post2020_ann']:+.1f}%/yr in "
            f"USO's favor**, itself statistically real (naive *t* = {R['post2020_naive_t']:.2f}, "
            f"NW(21) *t* = {R['post2020_nw21_t']:.2f}, block-bootstrap 95% CI "
            f"[{R['post2020_boot_lo']:+.1f}%, {R['post2020_boot_hi']:+.1f}%]/yr, fully positive). "
            "This rejects a persistent mechanical tax: it is a regime-dependent roll wedge — "
            "negative in contango, positive in backwardation — that happened to sit on the "
            "losing side of that divide during two historic storage crises and on the winning "
            "side of it for the five-plus years since."
        ),
        md(
            "### 4d · The April-2020 case study\n\n"
            "Simple (not log) returns around 2020-04-20, where the front WTI contract settled "
            "at -$37.63 — log is undefined across a sign flip, so this window gets its own, "
            "non-parametric look rather than folding into 4a/4b."
        ),
        code(
            "labels = [d for d,_,_,_,_ in R['apr20']]\n"
            "uso_r = [ur for _,_,ur,_,_ in R['apr20']]\n"
            "clf_r = [cr for _,_,_,_,cr in R['apr20']]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.6))\n"
            "x = np.arange(len(labels))\n"
            "ax.bar(x-0.2, uso_r, width=0.38, color=AMBER, label='USO')\n"
            "ax.bar(x+0.2, clf_r, width=0.38, color=RED, label='CL=F (front WTI)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20)\n"
            "ax.set_ylabel('daily return (%, simple)')\n"
            "ax.set_title('Front-month WTI settles at -$37.63; USO decouples')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for d,uc,ur,cc,cr in R['apr20']:\n"
            "    print(f'{d}: USO close {uc:6.2f} ({ur:+7.2f}%)   CL=F close {cc:8.2f} ({cr:+8.2f}%)')"
        ),
        md(
            f"> 💡 In plain words: on the day of the -$37.63 settlement (a -306% one-day move "
            f"in the front contract), USO fell **{R['apr20'][1][2]:+.1f}%** — an ordinary bad "
            "day, not a historic one. By April 2020 USO's holdings were already laddered across "
            "contract months (a structure pushed further that same month, alongside the "
            f"2020-04-29 1-for-8 reverse split). USO still lost **{abs(R['week_dd']):.1f}%** "
            "over the surrounding week — it didn't dodge the crisis — but H₄ holds: the fund "
            "does not mirror the front contract's move, in *either* direction."
        ),
        md(
            "### 4e · The honest carry-capture test — and its ex-crisis decomposition\n\n"
            "Constant-notional long-CL=F / short-USO, rebalanced monthly; 0.75%/yr borrow on "
            "the short, 2 × one-way cost × NAV charged on rebalance days only."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cc = st.carry_capture_summary(DF, STRESS, borrow_annual=0.0075,\n"
            "                                  cost_bps_sweep=(5.0, 10.0))\n"
            "    g, n5, n10 = cc['gross'], cc['net_5'], cc['net_10']\n"
            "    ex, so = cc['net5_ex_stress'], cc['net5_stress_only']\n"
            "    hac_t, pval = cc['hac_t'], cc['placebo_p']\n"
            "    gross_ann, net5_ann, net10_ann = g['ann_ret']*100, n5['ann_ret']*100, n10['ann_ret']*100\n"
            "    ex_ann, ex_sh = ex['ann_ret']*100, ex['sharpe']\n"
            "    so_ann, so_sh = so['ann_ret']*100, so['sharpe']\n"
            "    net5_sh = n5['sharpe']\n"
            "else:\n"
            "    gross_ann, net5_ann, net10_ann = R['gross_ann'], R['net5_ann'], R['net10_ann']\n"
            "    ex_ann, ex_sh = R['ex_stress_ann'], R['ex_stress_sharpe']\n"
            "    so_ann, so_sh = R['stress_only_ann'], R['stress_only_sharpe']\n"
            "    hac_t, pval, net5_sh = R['hac21_t'], R['placebo_p'], R['net5_sharpe']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.4))\n"
            "a1.bar(['gross','net 5bps','net 10bps'], [gross_ann, net5_ann, net10_ann],\n"
            "       color=[GREY, AMBER, AMBER], width=.6)\n"
            "for i,v in enumerate([gross_ann, net5_ann, net10_ann]):\n"
            "    a1.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom')\n"
            "a1.set_ylabel('%/yr'); a1.set_title(f'Positive on paper (HAC(21) t={hac_t:+.2f})')\n"
            "a2.bar(['full sample\\n(net, 5bps)','ex-crisis\\n(net, 5bps)'], [net5_ann, ex_ann],\n"
            "       color=[AMBER, RED], width=.55)\n"
            "for i,(v,s) in enumerate([(net5_ann,net5_sh),(ex_ann,ex_sh)]):\n"
            "    a2.annotate(f'{v:+.2f}%/yr\\n(Sharpe {s:.2f})',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('%/yr'); a2.set_title('...and it was almost all 2 crisis windows')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {gross_ann:+.2f}%/yr -> net {net5_ann:+.2f}/{net10_ann:+.2f}%/yr  '\n"
            "      f'(HAC(21) t={hac_t:+.2f}, placebo p={pval:.4f})')\n"
            "print(f'ex-crisis net {ex_ann:+.2f}%/yr (Sharpe {ex_sh:.3f})  |  '\n"
            "      f'crisis-only net {so_ann:+.1f}%/yr (Sharpe {so_sh:.2f}, illustrative)')"
        ),
        md(
            f"> 💡 In plain words: +{R['net5_ann']:.2f}%/yr net at Sharpe {R['net5_sharpe']:.2f} "
            f"looks like a real, deployable carry — until the ex-crisis split: "
            f"**+{R['ex_stress_ann']:.2f}%/yr at Sharpe {R['ex_stress_sharpe']:.2f}** for the "
            "95.5% of calendar time outside the two named windows. The HAC(21) *t* on the daily "
            f"book return (**{R['hac21_t']:.2f}**) already misses the certification bar; the "
            "regime decomposition shows *why* — this is not a repeatable carry, it is two "
            "historical disasters correctly held in a backtest that cannot be timed forward. Add "
            f"a **{R['gross_dd']:.1f}%** max drawdown even gross, and short-borrow that "
            "tightens exactly when a new storage crisis hits — **H₅ not supported; "
            "Tradability = MIRAGE**."
        ),
        md(
            "### 4f · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic (spot, fund) pair: spot is a plain random walk, fund = spot minus a "
            "TUNABLE daily drag plus extra drag on two synthetic stress blocks, plus small "
            "independent tracking noise. The null (drag = 0) is checked over **20 seeds**."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    spot_r, fund_r, _ = data.synthetic_world(seed=2024 + s_, drag_daily_bps=0.0,\n"
            "                                              stress_extra_bps=0.0)\n"
            "    null_ts.append(st.synthetic_detect(spot_r, fund_r)['nw_t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "spot_r, fund_r, _ = data.synthetic_world(seed=2024, drag_daily_bps=3.0,\n"
            "                                          stress_extra_bps=300.0)\n"
            "sy = st.synthetic_detect(spot_r, fund_r)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12,.12,20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (drag=0), 20 seeds')\n"
            "ax.scatter([1], [sy['nw_t']], color=RED, s=90, zorder=5, label='planted drag')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('NW(5) t (fund - spot)')\n"
            "ax.set_title('Control: no null fires; a planted drag lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds  |  planted NW(5) t = {sy[\"nw_t\"]:+.2f}, '\n"
            "      f'naive t = {sy[\"naive_t\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and **never** crosses "
            f"the bar; a planted drag reads naive t = {R['syn_planted_naive']:.2f}, "
            f"NW(5) t = {R['syn_planted_nw5']:.2f}. The machinery is unbiased — the real-tape "
            "signal is the genuine article. *(A faithful-engine / power check only — never "
            "cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — USO **{R['tr_uso']:.1f}%** vs CL=F **{R['tr_clf']:+.1f}%** "
            f"over {R['years']:.2f} years (CAGR gap **{R['cagr_gap']:+.2f} pp/yr**) looks like a "
            f"strong full-sample result (naive t = {R['naive_t']:.2f}, NW(5) t = {R['nw5']:.2f}, "
            f"bootstrap excludes 0 at p = {R['boot_p']:.4f}) — but H₆ (persistence) rejects the "
            f"structural-decay framing: **{R['stress_share']:.0f}%** of the cumulative "
            f"divergence comes from just **{R['stress_pct_days']:.1f}%** of days (the "
            "2009/2020 crisis windows); "
            f"excluding all of 2020 drops the naive t to **{R['ex2020_naive_t']:.2f}** (no "
            f"longer separately certifiable); the calendar-year sign count is "
            f"**{R['years_neg']}/{R['years_total']}** negative, not \"almost every year\"; and "
            f"the **post-2020 subsample** ({R['post2020_years']:.1f} years, 2021-2026) reverses "
            f"sign entirely, with USO **beating** CL=F by **{R['post2020_ann']:+.1f}%/yr** at "
            f"its own real *t*-stat (naive {R['post2020_naive_t']:.2f}, NW(21) "
            f"{R['post2020_nw21_t']:.2f}). Two real crisis episodes did real, large, "
            "well-measured damage; that is not the same claim as a persistent, mechanical, "
            "per-year contango tax — the mechanism is genuine but regime-dependent, and the "
            "regime has run the other way for over five years.\n"
            f"- **Tradability `MIRAGE`** — the long-CL=F/short-USO book nets "
            f"+{R['net5_ann']:.2f}%/yr at Sharpe {R['net5_sharpe']:.2f} full-sample, but HAC(21) "
            f"t = **{R['hac21_t']:.2f}** misses the bar, and stripping the 2 named crisis "
            f"windows collapses it to **+{R['ex_stress_ann']:.2f}%/yr at Sharpe "
            f"{R['ex_stress_sharpe']:.2f}** — the whole edge is two unforecastable historical "
            f"events, against a **{R['gross_dd']:.1f}%** max drawdown even gross, and the trade "
            "would have run net negative through the entire post-2020 subsample.\n"
            "- **\"Owns oil?\" `BUSTED`** — 80% behind a flat headline price over 20 years, then "
            f"**{R['post2020_ann']:+.1f}%/yr ahead** of it since 2021, and the ETF barely moved "
            "the day oil made history in yet a third direction. Not a spot-crude proxy in any "
            "regime."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general object is roll yield.** Erb & Harvey (2006) show it drives most "
            "commodity-index/ETF returns; any futures-based product (VXX/VIXY, BITO, most "
            "commodity trackers) inherits the same wedge from spot. The professional expression "
            "lives directly in the futures curve (calendar spreads), not in a retail ETP "
            "wrapper that adds its own frictions on top.\n"
            "- **Why the regime concentration matters for the sequel:** if the roll-drag is "
            "genuinely regime-dependent (crisis-driven storage stress, not a smooth grind), the "
            "natural next study is a **contango-level-conditioned** carry book — sizing the "
            "short by the observed term-structure slope rather than holding constant notional "
            "through calm regimes where the edge is statistically indistinguishable from zero.\n"
            "- **Dedup map:** [100-melting-ice](../../100-melting-ice/) (general commodity "
            "contango decay), [226-crude-seasonality](../../226-crude-seasonality/) (calendar "
            "effects, not the futures-spot wedge), "
            "[375-vxx-roll-decay](../../375-vxx-roll-decay/) (same mechanism, VIX futures) and "
            "[619-bito-roll-drag](../../619-bito-roll-drag/) (same family, bitcoin, monthly CME "
            "roll).\n\n"
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
