"""Generate the two narrative notebooks for Study 651 (Sugar-Seasonality).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached CANE/SB=F tapes
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance CANE adj close + SB=F
# raw close, 2011-10-03 -> 2026-06-30, 176 monthly obs/series).
R = dict(
    start="2011-10-03", end="2026-06-30", n_months=176,
    bonf_crit=3.47, bonf_survive=0, bonf_n=12,
    # per-month: mean%, n, t_naive, t_hac
    month={
        1: (0.27, 15, 0.17, 0.18), 2: (0.05, 15, 0.04, 0.07), 3: (-1.31, 15, -0.48, -0.47),
        4: (-0.68, 15, -0.30, -0.31), 5: (-2.11, 15, -1.80, -1.90), 6: (0.23, 15, 0.17, 0.25),
        7: (-1.29, 14, -0.85, -1.26), 8: (-0.46, 14, -0.31, -0.31), 9: (1.44, 14, 0.96, 1.50),
        10: (-0.03, 14, -0.02, -0.02), 11: (-0.63, 15, -0.52, -0.71), 12: (-1.76, 15, -1.15, -1.12),
    },
    best_month=9, best_mean=1.44, best_t=1.36, best_n=14,
    worst_month=5, worst_mean=-2.11, worst_t=-1.35, worst_n=15,
    # tight vs crush
    tight_mean=-0.33, tight_n=45, crush_mean=-0.96, crush_n=59,
    spread=0.63, spread_t=0.46, ci_lo=-1.89, ci_hi=2.98, ci_boot=5000,
    # roll drag: CANE bps/mo, SB=F bps/mo, drag bps/mo, t, n
    etf_bps=-53.4, fut_bps=-33.3, drag_bps=-20.1, drag_t=-0.80, drag_n=176,
    # seasonal timer: buy&hold sharpe/cagr%, timer-gross sharpe/cagr%, timer-net sharpe/cagr%
    bh_sharpe=-0.29, bh_cagr=-8.5, tg_sharpe=0.16, tg_cagr=1.2, tn_sharpe=0.14, tn_cagr=0.8,
    # coin-flip hit rate on the timer's active legs
    hit_k=61, hit_n=104, hit_rate=58.7, hit_lo=49.0, hit_hi=67.6, hit_t=0.60,
    # synthetic control
    syn_null_mean=0.23, syn_null_sd=1.01, syn_null_fire=0, syn_planted_t=4.73, syn_planted_pct=12.0,
    fp_cane="c0c35fdc0e66", fp_sb="88229d212e57",
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Beats_a_coin%3F: Busted](https://img.shields.io/badge/Beats_a_coin%3F-Busted-8b949e?style=flat-square)\n\n"
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

from sugar_seasonality import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    ETF, FUT = data.load_real()
    MONTHLY = st.monthly_log_returns(ETF)
    FUT_MONTHLY = st.monthly_log_returns(FUT)
else:
    ETF = FUT = MONTHLY = FUT_MONTHLY = None
print("real cache present:", HAVE_REAL, "| months on tape:", (0 if MONTHLY is None else len(MONTHLY)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does sugar really have a winter high and a Brazilian-harvest low? 🍬\n"
            "### The crush calendar — a sugar-desk legend, tested on the one ETF anyone can "
            "actually buy\n\n"
            + BADGES +
            "Sugar traders have a story: Brazil, the world's biggest cane grower, crushes its crop "
            "**April through November**; India, the second biggest, crushes **October through "
            "April**. In between — the Northern-Hemisphere winter — old-crop stocks are supposedly "
            "at their scarcest, right before Brazil's new crush ramps up. So prices should be "
            "**tight and firm into January-March**, then give it all back as the Brazilian harvest "
            "floods the market every spring.\n\n"
            "It's a great story with real agronomics behind it. So we tested it on **CANE** — the "
            "Teucrium Sugar Fund, the ETF a retail account can actually hold — from 2011 to 2026.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the Bonferroni table and the "
            "roll-drag math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** The Brazil/India crush windows come from USDA FAS and UNICA's "
            "own published crush-progress reports — hardcoded facts, not a fit. Every chart is "
            "drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is the pre-harvest winter really more expensive than the Brazilian crush window? | "
            f"**Not on this tape.** The tight window averages **{R['spread']:+.2f}%** more than "
            "the crush window — nominally the right direction, but that tiny gap could easily be "
            f"zero (its uncertainty band runs from {R['ci_lo']:+.2f}% to {R['ci_hi']:+.2f}%). |\n"
            "| What's the single best or worst calendar month, then? | Worst is **May** "
            f"({R['worst_mean']:+.2f}%), inside the crush window — a point *for* the story. Best "
            f"is **September** ({R['best_mean']:+.2f}%), which is deep in Brazil's *own* crush "
            "window too — a point against a clean \"winter high\" story. Neither is statistically "
            "meaningful on its own. |\n"
            "| Can you at least trade the *idea* — long winter, short the crush? | On paper, a "
            "little. Test the *active* trades on their own and they hit "
            f"**{R['hit_rate']:.0f}%** of the time — nominally above a coin flip, but not "
            "reliably so. |\n"
            "| Is there a real cost hiding underneath all this? | Some — CANE gives back a nominal "
            f"**{-R['drag_bps']:.0f} basis points a month** to its own roll, though that number "
            "itself isn't statistically certain. |\n\n"
            "> The agronomics are real. The calendar trade, on this tape, is not."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Brazil crushes cane April through November. India crushes it October through "
            "April. Between the two, world raw-sugar stocks are tightest in the Northern-"
            "Hemisphere winter — right before Brazil's new crop hits full stride — so prices "
            "carry a seasonal premium into that window and give it back every spring as the "
            "Brazilian harvest floods the market.\"*\n\n"
            "It's the same shape of story grain desks tell about corn and wheat — a scarcity "
            "premium ahead of a predictable harvest — applied to the world's two biggest cane "
            "suppliers instead of two Midwestern row crops."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If this pattern is real and tradable, it's a clean calendar trade with a genuine "
            "supply-and-demand mechanism behind it, on a single liquid ETF anyone can buy with no "
            "futures account. If it's *not* real — or real but too small/noisy to bank — that's an "
            "equally useful answer: it tells you why \"everyone knows\" a seasonal that nobody can "
            "actually trade for a living."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "- **The calendar.** Brazil's and India's own crush-season windows, hardcoded from USDA "
            "FAS and UNICA's published crop reports — not fitted to the price data.\n"
            "- **The comparison.** Every calendar month's average CANE return — 12 numbers — "
            "checked against a strict bar that accounts for testing 12 things at once "
            "(Bonferroni), so a single lucky month can't sneak through.\n"
            "- **The trade check.** Buy the winter-tight window, short the crush window, sit in "
            "cash otherwise — costs included — and separately ask: on the months the trade is "
            "actually *on*, does it win more than half the time?\n"
            "- **The roll check.** Compare CANE's own return to a simple futures-splice spot "
            "proxy, to see how much the ETF's own mechanics already eat before any seasonal math "
            "even starts."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Average return in the claimed winter-tight window vs the "
            "claimed crush-glut window."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tc = st.tight_crush_tstat(MONTHLY, data.TIGHT_MONTHS, data.CRUSH_MONTHS)\n"
            "    tight_v, crush_v = tc['tight_mean']*100, tc['crush_mean']*100\n"
            "else:\n"
            "    tight_v, crush_v = R['tight_mean'], R['crush_mean']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['tight (Jan-Mar)\\nclaimed high', 'crush (Apr-Jul)\\nclaimed low'],\n"
            "       [tight_v, crush_v], color=[AMBER, GREY], width=.55)\n"
            "for i,v in enumerate([tight_v, crush_v]): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',\n"
            "    va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean monthly return (%)')\n"
            "ax.set_title('The winter-tight window is NOT reliably higher than the crush window')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'tight {tight_v:+.2f}%   crush {crush_v:+.2f}%')"
        ),
        md(
            f"The spread points the folklore's direction (**{R['spread']:+.2f}%/month** in favor "
            "of the winter-tight window) — but it's a rounding error with a huge uncertainty band "
            f"around it, from **{R['ci_lo']:+.1f}%** to **{R['ci_hi']:+.1f}%**, which comfortably "
            "contains zero (and the opposite sign).\n\n"
            "**Next, the full calendar.** Which single month is best or worst?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ms = st.month_stats(MONTHLY)\n"
            "    means = list(ms['mean']*100)\n"
            "else:\n"
            "    means = [R['month'][m][0] for m in range(1, 13)]\n"
            "cols = [RED if m in data.CRUSH_MONTHS else (AMBER if m in data.TIGHT_MONTHS else GREY)\n"
            "        for m in range(1, 13)]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.6))\n"
            "ax.bar(['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],\n"
            "       means, color=cols, width=.62)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean monthly return (%)')\n"
            "ax.set_title('Amber = claimed tight window, red = claimed crush window')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({m: round(v,2) for m, v in zip(range(1,13), means)})"
        ),
        md(
            f"The worst month is **May** ({R['worst_mean']:+.2f}%) — inside the crush window, a "
            f"point *for* the folklore. But the best month is **September** ({R['best_mean']:+.2f}%)"
            " — deep inside Brazil's *own* crush window, months after the tight window is long "
            "over, which is not the clean \"winter high, spring low\" shape the story predicts. "
            "None of these 12 individual months clears a strict bar that accounts for testing all "
            "12 at once — a single standout month out of 12 draws is exactly what you'd expect "
            "from chance alone, and none of ours even reaches the weaker, uncorrected bar.\n\n"
            "**Now, the trade.** Long the tight window, short the crush window, cash otherwise — "
            "does it beat just holding the ETF?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    timer = st.seasonal_timer(MONTHLY, data.TIGHT_MONTHS, data.CRUSH_MONTHS)\n"
            "    net = st.apply_costs(timer, n_trades_per_year=4, cost_bps_one_way=10.0)\n"
            "    bh = st.summary(MONTHLY)['sharpe']; tn = st.summary(net)['sharpe']\n"
            "else:\n"
            "    bh, tn = R['bh_sharpe'], R['tn_sharpe']\n"
            "fig, ax = plt.subplots(figsize=(6.8, 4.4))\n"
            "ax.bar(['buy & hold\\nCANE', 'timer\\n(net of costs)'], [bh, tn], color=[GREY, AMBER], width=.5)\n"
            "for i, v in enumerate([bh, tn]): ax.annotate(f'{v:+.2f}', (i, v), ha='center',\n"
            "    va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('Sharpe')\n"
            "ax.set_title('The timer looks better only because it is IN CASH half the year')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'buy&hold {bh:+.2f}   timer net {tn:+.2f}')"
        ),
        md(
            "The timer's Sharpe does creep above buy-and-hold — but that's a magic trick, not a "
            "discovery: the timer sits in cash 5 months a year, and CANE has *lost money on "
            "average since 2011*. Doing less of a losing trade looks better almost by definition. "
            "The honest test is to check **only the months the trade is actually on**: those "
            f"active months win **{R['hit_rate']:.1f}%** of the time — nominally above 50%, but "
            "the uncertainty band comfortably includes a coin flip, and the average return on "
            "those active months is statistically indistinguishable from zero.\n\n"
            "**Finally, the hidden cost.** Even if the calendar edge were real, can you actually "
            "collect it in the ETF, or does the ETF's own machinery eat it first?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    drag = st.roll_drag(MONTHLY, FUT_MONTHLY)['drag_bps']\n"
            "else:\n"
            "    drag = R['drag_bps']\n"
            "fig, ax = plt.subplots(figsize=(6.4, 4.4))\n"
            "ax.bar(['CANE vs SB=F splice'], [drag], color=AMBER, width=.4)\n"
            "ax.annotate(f'{drag:+.1f} bps/mo', (0, drag), ha='center', va='top' if drag<0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('CANE return minus its own futures-splice proxy (bps/month)')\n"
            "ax.set_title('CANE quietly pays away some return to its own roll every month')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'roll drag: {drag:+.1f} bps/mo')"
        ),
        md(
            f"CANE gives back a nominal **{-R['drag_bps']:.0f} basis points a month** to its own "
            "rolling mechanics, relative to a simple futures-price proxy — though that gap itself "
            "isn't statistically certain either. Either way, it's not the decisive blow here: the "
            "calendar seasonal is already indistinguishable from noise before any roll cost gets a "
            "vote."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** No calendar month clears the bar once you account for testing "
            "12 things at once — not one even clears the far weaker uncorrected bar. The pooled "
            "tight-vs-crush spread points the folklore's direction but is buried inside its own "
            "uncertainty band.\n"
            "- **Tradability — Mirage.** The seasonal timer's apparent improvement is a cash-drag "
            "artifact, not skill (its active months hit at roughly coin-flip odds), and CANE's own "
            "roll works against a long-tight bet even if the roll gap itself isn't certain.\n"
            "- **\"Beats a coin?\" — Busted.** The trade's active legs win 58.7% of the time — "
            "nominally above 50%, but not reliably so."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The mechanism might still live in the raw futures curve** — this study tests the "
            "*ETF*, the thing you can actually buy, not the raw contract-to-contract calendar "
            "spread directly. Measuring the ICE No.11 term structure around the crush windows is "
            "the natural sequel.\n"
            "- **Weather years and El Niño cycles might matter more than calendar months.** A "
            "single Brazilian drought or an early Indian monsoon can dominate 15 years of \"average "
            "May,\" and averaging it away is exactly what a clean seasonality test is supposed to "
            "do — which is one reason this one comes up empty.\n"
            "- **Sibling studies:** [307-coffee-seasonality](../../307-coffee-seasonality/) (a "
            "frost story, not a crush story) and "
            "[648-grain-seasonality](../../648-grain-seasonality/) (the same *shape* of "
            "old-crop/new-crop test, different crop family) ask the same shape of question about "
            "different crops — worth comparing notes.\n\n"
            "*Think the calendar edge is real somewhere this study didn't look — the raw futures "
            "curve, a specific weather year, a different window? Show a net, certifiable edge on "
            "a tradable instrument, then we'll talk.*"
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
            "# Sugar-Seasonality — a quantitative teardown 🔬\n"
            "### A 12-cell Bonferroni month table · a tight-vs-crush Welch/HAC split · a "
            "block-bootstrap CI · the ETF-vs-futures roll-drag test · a costed calendar timer · "
            "a coin-flip hit-rate test · a 20-seed synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — **raw sugar carries a pre-harvest-tight premium that decays into the "
            "Brazilian crush** — has a real agronomic mechanism (Brazil's and India's crush "
            "seasons) and plenty of trading-floor conviction behind it. The job here is to test it "
            "on the instrument a retail account can actually hold, with the multiple-testing "
            "correction a 12-cell month grid demands.\n\n"
            "> ⚠️ **Data note.** CANE daily adjusted closes + SB=F daily raw closes, yfinance, "
            "2011-10-03 → 2026-06-30 (176 monthly observations per series); **hardcoded Brazil/"
            "India crush windows** (facts, no network). No survivorship (single continuously-"
            "listed tickers). Methods in [`docs/references.md`](../docs/references.md), numbers "
            "in [`docs/results.md`](../docs/results.md) (fingerprints `" + R["fp_cane"] + "` / `" +
            R["fp_sb"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | tight-vs-crush spread **{R['spread']:+.2f}%/mo**, Welch "
            f"**t = {R['spread_t']:+.2f}**, bootstrap CI **[{R['ci_lo']:+.2f}%, {R['ci_hi']:+.2f}%]**; "
            f"**{R['bonf_survive']}/{R['bonf_n']}** month cells clear the Bonferroni bar "
            f"(\\|t\\| ≥ {R['bonf_crit']:.2f}) |\n"
            f"| **Tradability** | `MIRAGE` | roll drag **{R['drag_bps']:.1f} bps/mo** "
            f"(t = {R['drag_t']:.2f}, uncertified); timer net Sharpe **{R['tn_sharpe']:+.2f}** vs "
            f"buy-and-hold **{R['bh_sharpe']:+.2f}** |\n"
            f"| **Beats a coin?** | `BUSTED` | active-leg hit rate **{R['hit_k']}/{R['hit_n']} = "
            f"{R['hit_rate']:.1f}%** (Wilson [{R['hit_lo']:.1f}%, {R['hit_hi']:.1f}%]), "
            f"t = {R['hit_t']:.2f} |\n\n"
            "> 💡 In plain words: the crush calendar is real agronomics; the ETF tape shows no "
            "certified trace of it, and what little the timer earns is a cash-drag illusion, not "
            "skill."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t$ be CANE's monthly log return and $T, C \\subset \\{1..12\\}$ the claimed "
            "pre-harvest-tight (Jan-Mar) and crush-glut (Apr-Jul) month sets, chosen from the "
            "hardcoded Brazil/India crush calendar. The claims:\n\n"
            "- **H₁ (level).** $E[r_t \\mid m_t \\in T] \\gg E[r_t \\mid m_t \\in C]$ — the winter "
            "months systematically outperform the early-crush months.\n"
            "- **H₂ (concentration).** The seasonal is a genuine calendar effect, not one lucky "
            "month out of the 12 cells on the board.\n"
            "- **H₃ (tradability).** A calendar-known long-tight/short-crush timer on the *ETF* "
            "earns net of costs and beats a coin on its active legs.\n"
            "- **H₄ (roll).** Whatever the raw futures curve shows, the ETF the retail account "
            "actually holds captures most of it (the roll doesn't eat the edge).\n\n"
            "We find **H₁ rejected** (t = +0.46, right sign but noise-sized), **H₂ rejected** "
            "(0/12 Bonferroni survivors, 0/12 even clear the naive |t| ≥ 2 line), **H₃ rejected** "
            "(58.7% active-leg hit rate, uncertified) and **H₄ inconclusive but adverse** (a "
            "nominal, uncertified drag against the long-tight leg)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Calendar months within a year are **not** independent draws (a Brazilian drought or "
            "an early/late Indian monsoon is autocorrelated across adjacent months), so every "
            "per-month test reports both a naive one-sample *t* and a **Newey-West (HAC)** *t*. "
            "Testing 12 months at once means the honest bar is **Bonferroni**-corrected: "
            "$\\alpha/12 \\approx 0.0042$, which at $df \\approx 13$ needs $|t| \\ge 3.47$ — "
            "notably stricter than the naive $|t| \\ge 2$, though (unlike study 648's 3-grain, "
            "36-cell grid) this single-instrument test only needs the /12 bar, not /36. The "
            "tight-vs-crush headline uses a **Welch** two-sample *t* (unequal variances, unequal "
            "group sizes by construction) plus a **circular block-bootstrap** (12-month blocks, "
            "respecting the annual seasonal structure) for a distribution-free CI on the spread."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** CANE adjusted closes + SB=F raw closes, {R['start']} → {R['end']}, "
            f"resampled to **{R['n_months']} monthly observations**. As-of 2026-06-30 (last "
            "complete month).\n"
            "- **Calendar.** Hardcoded Brazil Center-South crush (Apr-Nov), India crush (Oct-Apr) "
            "and the claimed pre-harvest-tight (Jan-Mar) / crush-glut (Apr-Jul) test windows "
            "(`SUGAR_CALENDAR` in `data.py`) — facts, not a fit.\n"
            "- **Headline.** Per-month naive + HAC *t* (12 cells, Bonferroni-corrected); best/"
            "worst month Welch *t* vs rest; tight-vs-crush Welch *t* with a 5,000-draw "
            "block-bootstrap CI.\n"
            "- **Execution (calendar-known).** The seasonal timer's position is set from the fixed "
            "crush calendar alone — no signal-to-trade lag is needed, because the calendar repeats "
            "every year and is known at the start of it.\n"
            "- **Costs.** 4 one-way legs/yr × 10 bps × NAV, spread evenly across the 12 months.\n"
            "- **Roll cross-check.** CANE monthly return minus the roll-naive SB=F-splice monthly "
            "return, one-sample *t* of the gap — sizes what the ETF's own mechanics already take "
            "before any calendar math starts.\n"
            "- **Control.** i.i.d. synthetic monthly-return world, planted tight/crush spread "
            "knob; the null must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The 12-cell month table and the Bonferroni bar\n\n"
            "Per-month naive + HAC *t*-stats. The Bonferroni-corrected bar for 12 simultaneous "
            "tests dwarfs the naive |t| ≥ 2 line."
        ),
        code(
            "if HAVE_REAL:\n"
            "    crit = st.bonferroni_crit_t(12, df=13)\n"
            "    ms = st.month_stats(MONTHLY)\n"
            "    hac = ms['tstat_hac'].to_numpy()\n"
            "else:\n"
            "    crit = R['bonf_crit']\n"
            "    hac = [R['month'][m][3] for m in range(1, 13)]\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.6))\n"
            "cols = [RED if abs(v) >= crit else GREY for v in hac]\n"
            "ax.bar(range(1, 13), hac, color=cols, width=.62)\n"
            "ax.axhline(crit, ls='--', c=RED, lw=1); ax.axhline(-crit, ls='--', c=RED, lw=1)\n"
            "ax.axhline(2, ls=':', c=AMBER, lw=1); ax.axhline(-2, ls=':', c=AMBER, lw=1)\n"
            "ax.axhline(0, c='k', lw=.6)\n"
            "ax.set_xlabel('calendar month'); ax.set_ylabel('HAC t-stat (one-sample, vs 0)')\n"
            "ax.set_title(f'Bonferroni bar |t| >= {crit:.2f} (dashed red) vs naive |t| >= 2 (dotted amber)')\n"
            "plt.tight_layout(); plt.show()\n"
            "n_survive = sum(int(abs(v) >= crit) for v in hac)\n"
            "n_naive = sum(int(abs(v) >= 2) for v in hac)\n"
            "print(f'cells clearing Bonferroni: {n_survive}/12   cells clearing naive |t|>=2: {n_naive}/12')"
        ),
        md(
            f"**{R['bonf_survive']}/{R['bonf_n']}** cells clear the Bonferroni bar "
            f"(|t| ≥ {R['bonf_crit']:.2f}) — and, notably, **none clear even the naive |t| ≥ 2 "
            "line either**. The largest-magnitude cell is May (mean "
            f"{R['month'][5][0]:+.2f}%, t<sub>HAC</sub> = {R['month'][5][3]:+.2f}) — inside the "
            "claimed crush window, pointing the claimed (negative) direction, but short of even "
            "nominal significance."
        ),
        md(
            "### 4b · Best/worst month and tight-vs-crush\n\n"
            "Welch *t* of the single best/worst month vs every other month pooled, and of the "
            "claimed tight-vs-crush window."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bw = st.best_worst_vs_rest(MONTHLY)\n"
            "    tc = st.tight_crush_tstat(MONTHLY, data.TIGHT_MONTHS, data.CRUSH_MONTHS)\n"
            "    spread, spread_t = tc['spread']*100, tc['t']\n"
            "else:\n"
            "    bw = {'best_month': R['best_month'], 'best_mean': R['best_mean'], 'best_t': R['best_t'],\n"
            "          'worst_month': R['worst_month'], 'worst_mean': R['worst_mean'], 'worst_t': R['worst_t']}\n"
            "    spread, spread_t = R['spread'], R['spread_t']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.4))\n"
            "labels = [f\"best\\n(month {bw['best_month']})\", f\"worst\\n(month {bw['worst_month']})\"]\n"
            "vals = [bw['best_mean']*100 if HAVE_REAL else bw['best_mean'],\n"
            "        bw['worst_mean']*100 if HAVE_REAL else bw['worst_mean']]\n"
            "ts = [bw['best_t'], bw['worst_t']]\n"
            "a1.bar(labels, vals, color=[RED if abs(t)>=2 else GREY for t in ts], width=.5)\n"
            "for i,(v,t) in enumerate(zip(vals, ts)): a1.annotate(f'{v:+.2f}%\\n(t={t:+.2f})',(i,v),\n"
            "    ha='center', va='bottom' if v>=0 else 'top')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean monthly return (%)')\n"
            "a1.set_title('Neither clears |t| >= 2 vs the rest')\n"
            "a2.bar(['tight - crush\\nspread'], [spread], color=RED if abs(spread_t)>=2 else GREY, width=.4)\n"
            "a2.annotate(f'{spread:+.2f}%\\n(t={spread_t:+.2f})', (0, spread), ha='center',\n"
            "    va='bottom' if spread>=0 else 'top')\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('spread (%)')\n"
            "a2.set_title('The decisive pooled number')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'best month t={bw[\"best_t\"]:+.2f}  worst month t={bw[\"worst_t\"]:+.2f}  spread t={spread_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the best month (September, +{R['best_mean']:.2f}%, "
            f"t = {R['best_t']:+.2f}) and worst month (May, {R['worst_mean']:+.2f}%, "
            f"t = {R['worst_t']:+.2f}) are both inside |t| < 2 of the rest of the calendar. The "
            f"pooled tight-vs-crush spread ({R['spread']:+.2f}%, t = {R['spread_t']:+.2f}) is "
            "closer to zero than either individual month."
        ),
        md(
            "### 4c · The bootstrap CI on the decisive spread\n\n"
            "A circular block-bootstrap (12-month blocks, respecting the annual seasonal "
            "structure) gives a distribution-free CI on the tight-minus-crush spread."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ci = st.spread_bootstrap_ci(MONTHLY, data.TIGHT_MONTHS, data.CRUSH_MONTHS)\n"
            "    spread, lo, hi = tc['spread']*100, ci['lo']*100, ci['hi']*100\n"
            "else:\n"
            "    spread, lo, hi = R['spread'], R['ci_lo'], R['ci_hi']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.0))\n"
            "ax.barh(['tight - crush spread'], [spread], color=GREY, height=.4)\n"
            "ax.errorbar([spread], ['tight - crush spread'], xerr=[[spread-lo],[hi-spread]],\n"
            "            fmt='none', ecolor=RED, capsize=8, lw=2)\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('tight - crush spread (%), 95% block-bootstrap CI')\n"
            "ax.set_title(f'Spread {spread:+.2f}%, CI [{lo:+.2f}%, {hi:+.2f}%] - straddles zero by a mile')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'pooled spread {spread:+.2f}%  CI [{lo:+.2f}%, {hi:+.2f}%]')"
        ),
        md(
            f"> 💡 In plain words: the CI width (roughly {R['ci_hi']-R['ci_lo']:.1f} points) is "
            f"about {(R['ci_hi']-R['ci_lo'])/abs(R['spread']):.0f}x the point estimate itself. "
            "H₁ is rejected — not \"weak,\" genuinely indistinguishable from zero on this tape."
        ),
        md(
            "### 4d · The ETF's own roll — the 'contango' caveat, sized\n\n"
            "CANE's monthly return minus the roll-naive SB=F-splice monthly return, one-sample *t* "
            "of the gap."
        ),
        code(
            "if HAVE_REAL:\n"
            "    drag = st.roll_drag(MONTHLY, FUT_MONTHLY)\n"
            "    d, dt = drag['drag_bps'], drag['t']\n"
            "else:\n"
            "    d, dt = R['drag_bps'], R['drag_t']\n"
            "fig, ax = plt.subplots(figsize=(6.6, 4.4))\n"
            "ax.bar(['CANE vs SB=F splice'], [d], color=RED if abs(dt)>=2 else AMBER, width=.4)\n"
            "ax.annotate(f'{d:+.1f} bps/mo\\n(t={dt:+.2f})', (0, d), ha='center', va='top' if d<0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('CANE minus futures-splice, mean monthly gap (bps)')\n"
            "ax.set_title('A nominal roll drag - not itself certified')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'drag {d:+.1f} bps/mo  t={dt:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: CANE's drag ({R['drag_bps']:.1f} bps/mo, t = {R['drag_t']:.2f}) "
            "works against a long-tight bet in direction, but does **not** itself clear |t| ≥ 2 — "
            "unlike study 648's WEAT (certified −90.3 bps/mo, t = −5.09), sugar's roll is not the "
            "decisive blow here. The seasonal itself is already indistinguishable from noise "
            "before any roll cost is charged."
        ),
        md(
            "### 4e · The costed timer and the coin-flip test\n\n"
            "Long tight, short crush, cash otherwise (no execution lag — the crush calendar "
            "repeats every year); 4 one-way legs/yr × 10 bps × NAV. The honest read isolates the "
            "*active* legs from the cash months."
        ),
        code(
            "if HAVE_REAL:\n"
            "    timer = st.seasonal_timer(MONTHLY, data.TIGHT_MONTHS, data.CRUSH_MONTHS)\n"
            "    net = st.apply_costs(timer, n_trades_per_year=4, cost_bps_one_way=10.0)\n"
            "    bh, tg, tn = st.summary(MONTHLY)['sharpe'], st.summary(timer)['sharpe'], st.summary(net)['sharpe']\n"
            "else:\n"
            "    bh, tg, tn = R['bh_sharpe'], R['tg_sharpe'], R['tn_sharpe']\n"
            "fig, ax = plt.subplots(figsize=(7.2, 4.4))\n"
            "ax.bar(['buy & hold', 'timer (gross)', 'timer (net)'], [bh, tg, tn], color=[GREY, AMBER, AMBER], width=.55)\n"
            "for i, v in enumerate([bh, tg, tn]): ax.annotate(f'{v:+.2f}', (i, v), ha='center', va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('Sharpe'); ax.set_title('Timer vs buy-and-hold')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'buy&hold {bh:+.2f}   timer gross {tg:+.2f}   timer net {tn:+.2f}')"
        ),
        code(
            "if HAVE_REAL:\n"
            "    active = timer[timer != 0.0].dropna()\n"
            "    x = active.to_numpy()\n"
            "    k, n = int((x > 0).sum()), len(x)\n"
            "    rate = k / n\n"
            "    t = st._one_sample_t(x)\n"
            "else:\n"
            "    k, n, rate, t = R['hit_k'], R['hit_n'], R['hit_rate']/100, R['hit_t']\n"
            "lo, hi = st.wilson_interval(k, n)\n"
            "fig, ax = plt.subplots(figsize=(7.0, 4.2))\n"
            "ax.bar(['active-leg hit rate'], [rate*100], color=AMBER if rate >= 0.5 else RED, width=.4)\n"
            "ax.errorbar([0], [rate*100], yerr=[[(rate-lo)*100],[(hi-rate)*100]], fmt='none', ecolor='k', capsize=8)\n"
            "ax.axhline(50, ls='--', c='k', lw=1.2, label='coin flip')\n"
            "ax.set_ylabel('% of active months with a positive return')\n"
            "ax.set_title(f'{k}/{n} = {rate*100:.1f}% (Wilson [{lo*100:.1f}%, {hi*100:.1f}%]) - not certifiably better than a coin')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'hit rate {k}/{n} = {rate*100:.1f}%  Wilson [{lo*100:.1f}%, {hi*100:.1f}%]  t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the Sharpe bar chart flatters the timer (it's mostly just cash "
            "on an already-losing ETF), but the coin-flip test strips that out — on the months the "
            f"trade is actually live, it wins **{R['hit_rate']:.1f}%** of the time (Wilson "
            f"[{R['hit_lo']:.1f}%, {R['hit_hi']:.1f}%]), a point estimate above 50% whose interval "
            f"still straddles it. Mean per active month is statistically zero "
            f"(t = {R['hit_t']:.2f}). **H₃ rejected.**"
        ),
        md(
            "### 4f · Faithful-engine & power control — we know the truth here\n\n"
            "Deterministic i.i.d. monthly-return world with a TUNABLE planted tight-premium/"
            "crush-discount pair. The null is checked over **20 seeds** — never a single stream."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    df = data.synthetic_world(seasonal=0.0, seed=651 + s_)\n"
            "    null_ts.append(st.synthetic_detect(df, data.TIGHT_MONTHS, data.CRUSH_MONTHS)['t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "df = data.synthetic_world(seasonal=0.12, seed=651)\n"
            "planted_t = st.synthetic_detect(df, data.TIGHT_MONTHS, data.CRUSH_MONTHS)['t']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (seasonal=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=RED, s=90, zorder=5, label='planted seasonal = +12%/yr')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('Welch t (tight vs crush)')\n"
            "ax.set_title('Control: the null never fires; a planted spread lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and **never** crosses the "
            f"bar ({R['syn_null_fire']}/20); a planted {R['syn_planted_pct']:.0f}%/yr spread reads "
            f"t = {R['syn_planted_t']:.2f}. The machinery is unbiased — the real-tape t = "
            f"{R['spread_t']:+.2f} is the genuine article, not a broken detector. *(A "
            "faithful-engine / power check only — never cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — tight-vs-crush spread **{R['spread']:+.2f}%/month**, Welch "
            f"t = **{R['spread_t']:+.2f}**, bootstrap CI **[{R['ci_lo']:+.2f}%, {R['ci_hi']:+.2f}%]**; "
            f"**{R['bonf_survive']}/{R['bonf_n']}** grid cells clear Bonferroni, and none even "
            "clear the weaker naive |t| ≥ 2 line; the nominal direction agrees with the folklore "
            "(worst month May, inside the crush window) but every number sits well inside noise.\n"
            f"- **Tradability `MIRAGE`** — CANE's roll drag is a nominal, uncertified "
            f"**{R['drag_bps']:.1f} bps/month (t = {R['drag_t']:.2f})** against a long-tight bet; "
            f"the timer's net Sharpe is **{R['tn_sharpe']:+.2f}** vs buy-and-hold's "
            f"**{R['bh_sharpe']:+.2f}**, and the apparent gain is a cash-drag artifact of an "
            "already-losing ETF.\n"
            f"- **\"Beats a coin?\" `BUSTED`** — active-leg hit rate "
            f"**{R['hit_k']}/{R['hit_n']} = {R['hit_rate']:.1f}%** (Wilson "
            f"[{R['hit_lo']:.1f}%, {R['hit_hi']:.1f}%]), mean per active month t = "
            f"**{R['hit_t']:.2f}**. Not *wrong* about the agronomics existing — wrong that it's "
            "still there, tradable, in this ETF, today."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The mechanism could still live in the raw futures basis.** This study tests the "
            "*ETF*, not the contract-to-contract calendar spread directly. A term-structure study "
            "(front vs deferred, calendar-spread level around the crush windows) is the natural "
            "sequel, the way study 648's roll-drag test separates the curve from the holder.\n"
            "- **Weather-year and El Niño clustering could be masking a conditional effect** — a "
            "seasonal that only shows up in *drought* years is a different, testable claim; this "
            "study deliberately averages across all weather regimes, which is why an unconditional "
            "calendar test comes up empty even if a conditional one wouldn't.\n"
            "- **Dedup map:** [307-coffee-seasonality](../../307-coffee-seasonality/) (a frost tail "
            "event, not a crush calendar), [308-cocoa-squeeze](../../308-cocoa-squeeze/) (a squeeze, "
            "not a calendar), [648-grain-seasonality](../../648-grain-seasonality/) (the same "
            "old-crop/new-crop shape, a different crop family and mechanism) — all reach the same "
            "honest `NONE`/`MIRAGE` shape independently.\n\n"
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
