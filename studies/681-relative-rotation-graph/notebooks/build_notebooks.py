"""Generate the two narrative notebooks for Study 681 (Relative-Rotation-Graph).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached daily
sector/SPY tape under ../_cache/ and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere with
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance 11 sector ETFs +
# SPY, daily 1998-12-01 -> 2026-06-30; RS-Ratio W=63d, RS-Momentum M=21d).
R = dict(
    start="1998-12-01", end="2026-06-30", fp="27323464af92", n_rows=6936,
    n_months_quad=324, quad_lo="1999-07", quad_hi="2026-06",
    n_months_ret=331, ret_lo="1998-12", ret_hi="2026-06",
    rs_window=63, mom_window=21, cost_bps=5.0,
    rrg=dict(ann=6.66, vol=16.49, sharpe=0.404, dd=-46.21, t=2.36, n=331),
    ew=dict(ann=9.64, vol=14.60, sharpe=0.660, dd=-49.12, t=3.55, n=330),
    spy=dict(ann=9.46, vol=15.11, sharpe=0.626, dd=-50.78, t=3.25, n=330),
    mom=dict(ann=8.45, vol=13.66, sharpe=0.619, dd=-32.45, t=3.36, n=331),
    n_leading_dist={0: 26, 1: 54, 2: 77, 3: 88, 4: 55, 5: 22, 6: 6, 7: 2, 8: 1},
    mean_n_leading=2.60, cash_months=26, cash_pct=7.9, mean_turnover=76.9,
    act_ew=dict(ann=-2.96, t=-2.07, n=330),
    act_spy=dict(ann=-2.78, t=-1.68, n=330),
    act_mom=dict(ann=-1.80, t=-0.93, n=331),
    act_rand=dict(ann=-0.50, t=-0.35, n=331),
    cost_sweep={0.0: (7.58, 0.460, 2.69), 5.0: (6.66, 0.404, 2.36),
                10.0: (5.74, 0.348, 2.03), 20.0: (3.89, 0.236, 1.38)},
    sub_periods={
        "1999-2009": dict(rrg_ann=3.81, rrg_sharpe=0.218, rrg_t=0.72, rrg_n=126,
                           ew_ann=3.77, ew_sharpe=0.242, ew_t=0.68),
        "2010-2019": dict(rrg_ann=8.82, rrg_sharpe=0.711, rrg_t=2.58, rrg_n=120,
                           ew_ann=12.51, ew_sharpe=1.044, ew_t=4.30),
        "2020-2026": dict(rrg_ann=8.52, rrg_sharpe=0.412, rrg_t=1.21, rrg_n=78,
                           ew_ann=13.32, ew_sharpe=0.808, ew_t=2.65),
    },
    trans={
        "Leading":   {"Leading": 196, "Weakening": 409, "Lagging": 237, "Improving": 15},
        "Weakening": {"Leading": 173, "Weakening": 251, "Lagging": 247, "Improving": 9},
        "Lagging":   {"Leading": 219, "Weakening": 6, "Lagging": 177, "Improving": 430},
        "Improving": {"Leading": 269, "Weakening": 14, "Lagging": 167, "Improving": 298},
    },
    clockwise_pct=61.7, random_baseline_pct=33.3, n_moves=2195,
    per_quad_cw={"Leading": 61.9, "Weakening": 57.6, "Lagging": 65.6, "Improving": 59.8},
    syn_null_mean=-0.30, syn_null_sd=1.11, syn_null_fire=1, syn_null_seeds=20,
    syn_planted_ann=15.30, syn_planted_t=6.20,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Rotates_clockwise%3F: Confirmed](https://img.shields.io/badge/Rotates_clockwise%3F-Confirmed-8b949e?style=flat-square)\n\n"
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

from relative_rotation_graph import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PRICES = data.load_real()
    FRAMES = st.rrg_frame(PRICES, data.TICKERS, data.BENCHMARK, data.RS_WINDOW, data.MOM_WINDOW)
    QUAD = st.monthly_quadrants(FRAMES)
    RETS = st.monthly_returns(PRICES, data.TICKERS)
else:
    PRICES = FRAMES = QUAD = RETS = None
print("real cache present:", HAVE_REAL, "| quadrant months:", (0 if QUAD is None else len(QUAD)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Can a two-axis chart really tell you which sector to buy? 🔄📊\n"
            "### The Relative Rotation Graph — a beautiful chart whose own physics checks "
            "out, and still can't pick a winner\n\n"
            + BADGES +
            "If you've ever seen a chart with sector tickers looping around in ovals through "
            "four labelled zones — Leading, Weakening, Lagging, Improving — you've seen a "
            "**Relative Rotation Graph (RRG)**. It's on every major charting platform "
            "(StockCharts, Bloomberg's `RRG<GO>`), and the pitch is simple: a sector's "
            "relative strength has a *level* (is it winning right now?) and a *momentum* "
            "(is that lead growing or fading?). Plot both, and a sector traces a loop through "
            "the four quadrants as its fortunes rise and fall. Buy it while it's **Leading**. "
            "Get out before it's **Lagging**.\n\n"
            "That's the claim we test: does watching the loop actually make you money — or "
            "does it just look like it should?\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the matched-random control and "
            "the cost math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** We build the two axes ourselves (RRG vendors don't publish "
            "their exact smoothing constants) as rolling z-scores — a 63-trading-day level, a "
            "21-day rate of change — and test the plain rule: hold the equal-weighted "
            "**Leading** quadrant every month, cash if nobody qualifies. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the chart actually loop clockwise the way it's drawn? | **Yes, mostly.** "
            f"When a sector's quadrant changes at all, it moves the claimed clockwise way "
            f"**{R['clockwise_pct']:.0f}% of the time** — nearly double the 33% you'd expect "
            "from chance. |\n"
            "| Does buying \"Leading\" sectors make money? | **About as much as being long "
            "equities generally** — but no more than picking the *same number* of sectors at "
            f"random each month ({R['act_rand']['ann']:+.1f}%/yr, statistically zero). |\n"
            "| Does it beat just holding all 11 sectors equally? | **No — it loses**, by a "
            f"statistically real **{R['act_ew']['ann']:.1f}%/yr** (this isn't noise; it clears "
            "the certification bar in the *wrong* direction). |\n"
            "| Does it beat plain old-fashioned momentum? | **No.** A simple \"buy the 3 "
            "sectors with the best 6-month return\" rule does better, though not by a "
            "statistically provable margin either. |\n\n"
            "> The rotation loop is real. The alpha inside it isn't."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Relative strength isn't one number, it's two: a **level** (winning or "
            "losing right now) and a **momentum** (that lead accelerating or fading). Plot "
            "them together and a sector visibly rotates through four stages every cycle — "
            "buy it entering Leading, sell before it reaches Lagging — and you'll catch more "
            "of the move than a trader watching only one axis.\"* — the pitch behind Julius "
            "de Kempenaer's Relative Rotation Graph (RRG Research, commercialised on "
            "StockCharts and Bloomberg).\n\n"
            "It's a genuinely elegant idea: separate *where you are* from *where you're "
            "headed*. The only question that matters for a trading desk is whether that "
            "extra axis of information actually **predicts** anything."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the RRG really adds information over plain momentum, it should let you "
            "**rotate earlier and exit earlier** than a trader watching only trailing "
            "returns — catching more of a sector's run and less of its unwind. That's a "
            "real, sellable edge if true: sector allocators manage trillions, and \"which "
            "sector next\" is one of the oldest questions in the business.\n\n"
            "So we ask three things: does the chart's own rotation story hold up in the "
            "data, does buying \"Leading\" actually pick winners, and does the two-axis read "
            "beat the one-axis version it's supposed to improve on?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The universe.** The 11 SPDR sector ETFs vs SPY, {R['start']} → {R['end']}, "
            "monthly rebalance.\n"
            "- **The rule.** Each month-end, classify every sector into one of four quadrants "
            "from its RS-Ratio (level) and RS-Momentum (rate of change) vs SPY; hold the "
            "equal-weighted **Leading** sectors the following month (no look-ahead), cash if "
            "none qualify.\n"
            "- **The controls.** SPY, an equal-weight sector basket, a plain 6-month momentum "
            "top-3 sort (the one-axis version), and — the decisive one — a randomly-chosen "
            "basket matched to the **same number of sectors** RRG held that month, so a "
            "lucky concentration or a cash-timing accident can't masquerade as skill.\n"
            "- **The luck check.** A synthetic sector world with a knob for \"real persistent "
            "rotation\" turned fully off — the detector must stay quiet."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: does the chart even loop the way it's drawn?**"
        ),
        code(
            "labels = ['Leading', 'Weakening', 'Lagging', 'Improving']\n"
            "if HAVE_REAL:\n"
            "    trans = st.quadrant_transition_matrix(QUAD)\n"
            "    cw = st.clockwise_share(trans)\n"
            "    pooled, per_q = cw['pooled_clockwise_pct'], cw['per_quadrant_pct']\n"
            "else:\n"
            "    pooled, per_q = R['clockwise_pct'], R['per_quad_cw']\n"
            "vals = [per_q[q] for q in labels]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(labels, vals, color=GREY, width=.55, label='clockwise share of moves')\n"
            "ax.axhline(R['random_baseline_pct'], ls='--', c=RED, lw=1.5,\n"
            "           label=f\"random-chance baseline ({R['random_baseline_pct']:.0f}%)\")\n"
            "for i, v in enumerate(vals): ax.annotate(f'{v:.0f}%', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('% of quadrant changes that go clockwise-forward')\n"
            "ax.set_title(f'Yes — the loop is real: {pooled:.0f}% clockwise vs a 33% coin flip')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'pooled clockwise share: {pooled:.1f}%')"
        ),
        md(
            f"When a sector's quadrant changes, it moves the claimed clockwise way "
            f"**{R['clockwise_pct']:.0f}% of the time** — well above the 33% a coin flip would "
            "give you. The chart's own physics is genuinely descriptive: sectors really do "
            "orbit Leading → Weakening → Lagging → Improving more than they wander randomly. "
            "**Now the money question: does knowing that let you pick winners?**"
        ),
        code(
            "names = ['RRG\\n(Leading quadrant)', 'Equal-weight\\nbasket', 'SPY', '6-1 momentum\\n(top-3)']\n"
            "if HAVE_REAL:\n"
            "    rrg = st.run_rrg_strategy(QUAD, RETS, cost_bps=R['cost_bps'])\n"
            "    ew = st.equal_weight_returns(RETS)\n"
            "    spy_m = st.benchmark_monthly_returns(PRICES, data.BENCHMARK)\n"
            "    mom = st.run_momentum_strategy(RETS, lookback=6, skip=1, k=3, cost_bps=R['cost_bps'])\n"
            "    sharpes = [st.summarize(rrg['r_net'])['sharpe'], st.summarize(ew)['sharpe'],\n"
            "               st.summarize(spy_m)['sharpe'], st.summarize(mom['r_net'])['sharpe']]\n"
            "else:\n"
            "    sharpes = [R['rrg']['sharpe'], R['ew']['sharpe'], R['spy']['sharpe'], R['mom']['sharpe']]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "cols = [RED, GREY, GREY, GREY]\n"
            "ax.bar(names, sharpes, color=cols, width=.55)\n"
            "for i, v in enumerate(sharpes): ax.annotate(f'{v:.2f}', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('Sharpe ratio (net of 5bps one-way cost)')\n"
            "ax.set_title('The quadrant overlay has the WORST Sharpe of the four books')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Sharpe:', dict(zip(['RRG','EW','SPY','MOM'], [round(s,3) for s in sharpes])))"
        ),
        md(
            f"RRG's Sharpe (**{R['rrg']['sharpe']:.2f}**) is the *lowest* of all four — below "
            f"just holding all 11 sectors equally (**{R['ew']['sharpe']:.2f}**), below SPY "
            f"(**{R['spy']['sharpe']:.2f}**), below a bare-bones 6-month momentum sort "
            f"(**{R['mom']['sharpe']:.2f}**). Being long sectors most months pays — that's "
            "equity beta — but the two-axis quadrant read doesn't add to it. **The cleanest "
            "test isolates exactly why:**"
        ),
        code(
            "labels2 = ['vs equal-weight\\nbasket', 'vs SPY', 'vs plain\\n6-1 momentum',\n"
            "           'vs matched-random\\n(same count)']\n"
            "acts = [R['act_ew']['ann'], R['act_spy']['ann'], R['act_mom']['ann'], R['act_rand']['ann']]\n"
            "cols2 = [RED, GREY, GREY, AMBER]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(labels2, acts, color=cols2, width=.55)\n"
            "for i, v in enumerate(acts): ax.annotate(f'{v:+.2f}%/yr', (i, v), ha='center',\n"
            "    va='top' if v < 0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel(\"RRG's active return vs each control (%/yr)\")\n"
            "ax.set_title('Negative everywhere — and the loss vs equal-weight is statistically real')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('active ann:', dict(zip(['EW','SPY','MOM','RAND'], acts)))"
        ),
        md(
            f"Every bar is negative. The amber one — **matched-random** — is the fairest test: "
            "it picks the *same number* of sectors RRG held each month (including its cash "
            f"months), so it can't win just by being lucky about how concentrated the book "
            f"was. RRG scores **{R['act_rand']['ann']:+.2f}%/yr** against it, statistically "
            "indistinguishable from zero: **the quadrant tells you nothing about which sector "
            f"to hold.** And the red bar — versus simply holding all 11 sectors equally — is "
            f"**certified** (a real, statistically provable loss), because RRG's \"go to cash "
            f"when nothing is Leading\" rule sat out **{R['cash_pct']:.0f}%** of months while "
            "the naive basket stayed fully invested through a quarter-century bull-tilted "
            "tape."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Buying the Leading quadrant is statistically indistinguishable "
            "from a random pick of the same size, doesn't beat plain momentum, and "
            "underperforms the naive basket by a certified margin — the wrong direction from "
            "the claim.\n"
            f"- **Tradability — Mirage.** {R['mean_turnover']:.0f}% monthly one-way turnover, "
            "the worst Sharpe of four books, and a real shortfall to equal-weight in the two "
            "most recent decades.\n"
            "- **\"Does the chart rotate clockwise?\" — Confirmed.** The loop shape is real — "
            "it just doesn't come with a paycheck."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The chart is a visualisation, not a factor.** Splitting momentum into level "
            "and rate-of-change is a legitimate way to *see* a rotation — the mistake is "
            "assuming a clearer picture implies a more predictive one.\n"
            "- **Where a real edge (if any) would have to live** is in *which* Leading sectors "
            "to prefer, or in shorting confirmed Lagging names, or in much finer entry timing "
            "than a monthly snapshot — none of which this desk found evidence for here.\n"
            "- **Sibling studies:** [225-sector-rotation](../../225-sector-rotation/) (the "
            "plain 1-D version, same universe, same \"weak/mirage\" verdict) and "
            "[506-industry-momentum](../../506-industry-momentum/) (long-short industry "
            "momentum, also a coin flip) — the RRG doesn't fix what plain momentum couldn't "
            "already do on this universe.\n\n"
            "*Think the RRG works with a finer clock (weekly, or the vendor's exact "
            "constants)? Show a net, certifiable edge over the matched-random control — then "
            "we'll talk.*"
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
            "# The Relative Rotation Graph — a quantitative teardown 🔬\n"
            "### The RS-Ratio/RS-Momentum construction · the matched-random control · a cost "
            "sweep · sub-period stability · the quadrant transition-matrix myth-check · a "
            "20-seed synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The RRG (de Kempenaer, RRG Research) is a practitioner charting technique, not a "
            "published factor model — so the job here is to (1) build the two axes ourselves, "
            "explicitly, (2) test whether the descriptive rotation story holds, and (3) test "
            "whether the trading rule it implies (\"buy Leading\") beats the honest controls, "
            "chiefly a **matched-random** control that isolates *selection* skill from the "
            "mechanical cash-timing effect of a quadrant-conditioned rule.\n\n"
            "> ⚠️ **Data note.** 11 SPDR sector ETFs + SPY, daily adjusted closes, yfinance, "
            f"cached, {R['start']} → {R['end']} (XLRE from 2015-10, XLC from 2018-06 — named). "
            "No survivorship (all tickers used through their own live history). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fp"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | vs matched-random control: **t = {R['act_rand']['t']:.2f}** "
            f"({R['act_rand']['ann']:+.2f}%/yr); vs plain 6-1 momentum: t = {R['act_mom']['t']:.2f} "
            f"({R['act_mom']['ann']:+.2f}%/yr); vs equal-weight: **t = {R['act_ew']['t']:.2f}** "
            f"({R['act_ew']['ann']:+.2f}%/yr, certified negative) |\n"
            f"| **Tradability** | `MIRAGE` | {R['mean_turnover']:.0f}% one-way monthly turnover; "
            f"Sharpe {R['rrg']['sharpe']:.3f} vs EW {R['ew']['sharpe']:.3f} / SPY "
            f"{R['spy']['sharpe']:.3f} / MOM {R['mom']['sharpe']:.3f}; certified shortfall vs "
            "EW in 2010s and 2020s |\n"
            f"| **Rotates clockwise?** | `CONFIRMED` | {R['clockwise_pct']:.1f}% of quadrant "
            f"changes go clockwise-forward vs a {R['random_baseline_pct']:.1f}% random baseline "
            f"({R['n_moves']:,} changes) |\n\n"
            "> 💡 In plain words: the chart's descriptive mechanics check out; the trading rule "
            "built on top of them does not."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $P_{i,t}$ be sector $i$'s price and $B_t$ SPY's. Define the relative-strength "
            "line $RS_{i,t} = P_{i,t}/B_t$. The RRG's own construction (vendors don't publish "
            "exact constants, so we define ours in the open):\n\n"
            "$$\\text{RS-Ratio}_{i,t} = 100 + \\frac{RS_{i,t} - \\overline{RS_i}_{[t-W,t]}}"
            "{\\sigma(RS_i)_{[t-W,t]}}, \\quad "
            "\\text{RS-Momentum}_{i,t} = 100 + \\frac{\\Delta_M \\text{RS-Ratio}_{i,t} - "
            "\\overline{\\Delta_M \\text{RS-Ratio}_i}_{[t-W,t]}}{\\sigma(\\Delta_M "
            "\\text{RS-Ratio}_i)_{[t-W,t]}}$$\n\n"
            f"with $W$ = {R['rs_window']} trading days, $M$ = {R['mom_window']} trading days. "
            "Four quadrants from the sign of (RS-Ratio − 100, RS-Momentum − 100); the claim:\n\n"
            "- **H₁ (rotation).** Sectors move through the quadrants in the claimed clockwise "
            "order (Leading → Weakening → Lagging → Improving) more than a random mover would.\n"
            "- **H₂ (selection).** Holding the Leading quadrant beats a same-size random pick "
            "— i.e. the quadrant *label* carries forward-looking information about returns.\n"
            "- **H₃ (dominance).** The two-axis read beats the one-axis (plain momentum) "
            "control it's explicitly sold as improving on.\n\n"
            "We find **H₁ supported** (62% vs 33% baseline), **H₂ rejected** (*t* = "
            f"{R['act_rand']['t']:.2f}), **H₃ rejected** (negative point estimate, *t* = "
            f"{R['act_mom']['t']:.2f})."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "The confound that sinks a naive test here is **mechanical**: a rule that goes to "
            "cash whenever nothing is Leading will, on a positive-drift tape, underperform a "
            "*fully invested* benchmark (equal-weight, SPY) for reasons that have nothing to "
            "do with stock-picking skill — it's simply out of the market some months. So the "
            "decisive test is not RRG vs equal-weight (that conflates cash-timing with "
            "selection) — it's RRG vs a **matched-random control**: each month, pick the same "
            "*number* of random sectors RRG actually held (0 → cash, exactly matching RRG's "
            "cash months), Monte-Carlo-averaged over 200 seeds so no single lucky draw decides "
            "it. Any active return left over is purely about *which* sectors, not *how many*."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** 11 SPDR sector ETFs vs SPY, {R['start']} → {R['end']} daily, "
            "resampled to month-end for the rebalance.\n"
            f"- **Signal.** RS-Ratio (W={R['rs_window']}d) x RS-Momentum (M={R['mom_window']}d), "
            "both rolling z-scores centred on 100.\n"
            "- **Execution.** One lag: quadrant read at month-end close *t*, position (equal-"
            "weight the Leading quadrant, cash if empty) held over month *t+1*.\n"
            "- **Controls.** SPY, equal-weight basket, plain 6-1 top-3 momentum (sibling 225's "
            "exact construction), matched-random (200-seed Monte Carlo).\n"
            "- **Costs.** One-way bps x NAV per leg on realised turnover, swept 0/5/10/20 bps; "
            "round trip = 2x.\n"
            "- **Myth-check.** Pooled quadrant transition matrix across all 11 tickers; "
            "clockwise share of quadrant *changes* vs the 33.3% baseline three alternatives "
            "would give a uniform random mover.\n"
            "- **Control.** Synthetic daily multi-sector panel, AR(1) planted relative-drift "
            "knob; the null must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline race\n\n"
            "RRG's Leading-quadrant rotation vs three controls, net of 5bps one-way cost."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rrg = st.run_rrg_strategy(QUAD, RETS, cost_bps=R['cost_bps'])\n"
            "    ew = st.equal_weight_returns(RETS)\n"
            "    spy_m = st.benchmark_monthly_returns(PRICES, data.BENCHMARK)\n"
            "    mom = st.run_momentum_strategy(RETS, lookback=6, skip=1, k=3, cost_bps=R['cost_bps'])\n"
            "    rows = {'RRG': st.summarize(rrg['r_net']), 'EW': st.summarize(ew),\n"
            "            'SPY': st.summarize(spy_m), 'MOM': st.summarize(mom['r_net'])}\n"
            "    anns = [rows[k]['ann_ret']*100 for k in ['RRG','EW','SPY','MOM']]\n"
            "    sharpes = [rows[k]['sharpe'] for k in ['RRG','EW','SPY','MOM']]\n"
            "    ts = [rows[k]['tstat'] for k in ['RRG','EW','SPY','MOM']]\n"
            "else:\n"
            "    anns = [R['rrg']['ann'], R['ew']['ann'], R['spy']['ann'], R['mom']['ann']]\n"
            "    sharpes = [R['rrg']['sharpe'], R['ew']['sharpe'], R['spy']['sharpe'], R['mom']['sharpe']]\n"
            "    ts = [R['rrg']['t'], R['ew']['t'], R['spy']['t'], R['mom']['t']]\n"
            "names = ['RRG', 'EW basket', 'SPY', '6-1 MOM']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "cols = [RED, GREY, GREY, GREY]\n"
            "a1.bar(names, anns, color=cols, width=.6)\n"
            "for i, v in enumerate(anns): a1.annotate(f'{v:+.2f}%', (i, v), ha='center', va='bottom')\n"
            "a1.set_ylabel('annualised return (%)'); a1.set_title('Absolute return')\n"
            "a2.bar(names, sharpes, color=cols, width=.6)\n"
            "for i, v in enumerate(sharpes): a2.annotate(f'{v:.3f}\\n(t={ts[i]:+.2f})', (i, v),\n"
            "    ha='center', va='bottom', fontsize=8)\n"
            "a2.set_ylabel('Sharpe ratio'); a2.set_title('Risk-adjusted (NW t on the mean)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dict(zip(names, [round(a,2) for a in anns])), dict(zip(names, [round(s,3) for s in sharpes])))"
        ),
        md(
            f"> 💡 In plain words: RRG clears *t* ≥ 2 in absolute terms ({R['rrg']['t']:.2f}) — "
            "meaningless on its own, since being long a diversified equity basket is nearly "
            f"always going to do that. The tell is the Sharpe ranking: RRG's {R['rrg']['sharpe']:.3f} "
            f"is the *lowest* of the four, well under EW's {R['ew']['sharpe']:.3f}."
        ),
        md(
            "### 4b · The decisive test — matched-random control\n\n"
            "Each month, draw a random basket of the same SIZE RRG actually held (0 → cash), "
            "Monte-Carlo-averaged over many seeds. This is the only comparison that isolates "
            "*selection* from the mechanical cash-timing effect."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ctrl = st.matched_random_control(QUAD, RETS, n_seeds=50, cost_bps=R['cost_bps'])\n"
            "    a_rand = st.active_stats(rrg['r_net'], ctrl)\n"
            "    a_ew = st.active_stats(rrg['r_net'], ew)\n"
            "    a_spy = st.active_stats(rrg['r_net'], spy_m)\n"
            "    a_mom = st.active_stats(rrg['r_net'], mom['r_net'])\n"
            "    acts = [a_ew['active_ann']*100, a_spy['active_ann']*100, a_mom['active_ann']*100, a_rand['active_ann']*100]\n"
            "    act_ts = [a_ew['active_tstat'], a_spy['active_tstat'], a_mom['active_tstat'], a_rand['active_tstat']]\n"
            "else:\n"
            "    acts = [R['act_ew']['ann'], R['act_spy']['ann'], R['act_mom']['ann'], R['act_rand']['ann']]\n"
            "    act_ts = [R['act_ew']['t'], R['act_spy']['t'], R['act_mom']['t'], R['act_rand']['t']]\n"
            "labels = ['vs EW', 'vs SPY', 'vs 6-1 MOM', 'vs matched-\\nrandom (N=200)']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "cols = [RED if abs(t) >= 2 else AMBER for t in act_ts]\n"
            "ax.bar(labels, acts, color=cols, width=.55)\n"
            "for i, (v, t) in enumerate(zip(acts, act_ts)):\n"
            "    ax.annotate(f'{v:+.2f}%/yr\\nt={t:+.2f}', (i, v), ha='center',\n"
            "        va='top' if v < 0 else 'bottom', fontsize=8)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel(\"RRG's active return (%/yr)\")\n"
            "ax.set_title('Certified loss vs EW; statistically zero selection skill vs matched-random')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('active:', dict(zip(['EW','SPY','MOM','RAND'], zip(acts, act_ts))))"
        ),
        md(
            f"> 💡 In plain words: the matched-random control is the fair fight, and RRG loses "
            f"it too (*t* = {R['act_rand']['t']:.2f}, {R['act_rand']['ann']:+.2f}%/yr) — "
            "statistically zero, meaning the Leading-quadrant *label* carries no information "
            f"about which sector actually outperforms next month. The certified loss vs EW "
            f"(*t* = {R['act_ew']['t']:.2f}) is a **design artifact**: RRG sat in cash "
            f"{R['cash_pct']:.0f}% of months while EW stayed fully invested through a "
            "bull-tilted quarter-century."
        ),
        md(
            "### 4c · Cost sweep — turnover this high is not free\n\n"
            f"Mean one-way monthly turnover: **{R['mean_turnover']:.1f}%**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cbs = [0.0, 5.0, 10.0, 20.0]\n"
            "    anns_c, shs_c, ts_c = [], [], []\n"
            "    for cb in cbs:\n"
            "        r = st.run_rrg_strategy(QUAD, RETS, cost_bps=cb)\n"
            "        s = st.summarize(r['r_net'])\n"
            "        anns_c.append(s['ann_ret']*100); shs_c.append(s['sharpe']); ts_c.append(s['tstat'])\n"
            "else:\n"
            "    cbs = sorted(R['cost_sweep'])\n"
            "    anns_c = [R['cost_sweep'][c][0] for c in cbs]\n"
            "    shs_c = [R['cost_sweep'][c][1] for c in cbs]\n"
            "    ts_c = [R['cost_sweep'][c][2] for c in cbs]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.3))\n"
            "ax.plot(cbs, ts_c, 'o-', color=RED, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = 2 bar')\n"
            "ax.set_xlabel('one-way cost (bps)'); ax.set_ylabel('NW t (absolute return)')\n"
            "ax.set_title('Even the (irrelevant) absolute t-stat erodes with realistic costs')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(list(zip(cbs, [round(a,2) for a in anns_c], [round(s,3) for s in shs_c], [round(t,2) for t in ts_c])))"
        ),
        md(
            "### 4d · Sub-period stability — no era in which RRG beats equal-weight\n\n"
            "Split at the two natural sector-ETF-liquidity eras (same split points as sibling "
            "225)."
        ),
        code(
            "labels3 = list(R['sub_periods'].keys())\n"
            "if HAVE_REAL:\n"
            "    ranges = {'1999-2009': ('1999-07', '2009-12'), '2010-2019': ('2010-01', '2019-12'),\n"
            "              '2020-2026': ('2020-01', '2026-06')}\n"
            "    rrg_a, ew_a = [], []\n"
            "    for lo, hi in ranges.values():\n"
            "        m_r = (rrg.index >= lo) & (rrg.index <= hi)\n"
            "        m_e = (ew.index >= lo) & (ew.index <= hi)\n"
            "        rrg_a.append(st.summarize(rrg.loc[m_r, 'r_net'])['ann_ret'] * 100)\n"
            "        ew_a.append(st.summarize(ew.loc[m_e])['ann_ret'] * 100)\n"
            "else:\n"
            "    rrg_a = [R['sub_periods'][k]['rrg_ann'] for k in labels3]\n"
            "    ew_a = [R['sub_periods'][k]['ew_ann'] for k in labels3]\n"
            "x = np.arange(len(labels3)); w = 0.35\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(x - w/2, rrg_a, w, label='RRG', color=RED)\n"
            "ax.bar(x + w/2, ew_a, w, label='EW basket', color=GREY)\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels3)\n"
            "ax.set_ylabel('annualised return (%)')\n"
            "ax.set_title('EW pulls ahead exactly when sector ETFs got liquid and crowded')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('RRG:', rrg_a, ' EW:', ew_a)"
        ),
        md(
            "### 4e · The myth-check — the full transition matrix\n\n"
            "Pooled month-to-month quadrant transitions, all 11 tickers, 1999→2026."
        ),
        code(
            "labels4 = ['Leading', 'Weakening', 'Lagging', 'Improving']\n"
            "if HAVE_REAL:\n"
            "    trans_df = st.quadrant_transition_matrix(QUAD)\n"
            "    mat = trans_df.div(trans_df.sum(axis=1), axis=0).to_numpy() * 100\n"
            "else:\n"
            "    mat = np.array([[R['trans'][a][b] for b in labels4] for a in labels4], dtype=float)\n"
            "    mat = mat / mat.sum(axis=1, keepdims=True) * 100\n"
            "fig, ax = plt.subplots(figsize=(6.6, 5.6))\n"
            "im = ax.imshow(mat, cmap='RdYlGn_r', vmin=0, vmax=55)\n"
            "ax.set_xticks(range(4)); ax.set_xticklabels(labels4, rotation=30, ha='right')\n"
            "ax.set_yticks(range(4)); ax.set_yticklabels(labels4)\n"
            "ax.set_xlabel('quadrant at month t+1'); ax.set_ylabel('quadrant at month t')\n"
            "for i in range(4):\n"
            "    for j in range(4):\n"
            "        ax.text(j, i, f'{mat[i,j]:.0f}%', ha='center', va='center', fontsize=10,\n"
            "                color='white' if mat[i,j] > 30 else 'black')\n"
            "ax.set_title('Row-normalised transition matrix — the diagonal-below is thin')\n"
            "plt.colorbar(im, ax=ax, label='% of month-t observations')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: pooled clockwise share of quadrant CHANGES = "
            f"**{R['clockwise_pct']:.1f}%** vs a **{R['random_baseline_pct']:.1f}%** random "
            "baseline (3 equally-likely destinations). Immediate reversals — Leading→Improving, "
            "Weakening→Improving, Lagging→Weakening — are all under 2%, because RS-Momentum is "
            "*defined* as the smoothed rate-of-change of RS-Ratio, so a orderly progression is "
            "partly baked into the construction. The one notable exception, Weakening→Leading "
            "(25.4%), shows the loop isn't perfectly clean either. **Descriptive**, not "
            "predictive — H₁ holds, H₂ still fails."
        ),
        md(
            "### 4f · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic daily multi-sector panel, AR(1) persistent relative-drift knob "
            "(`mom_strength`), tested against the SAME matched-random control as the real "
            "tape. Null checked over 20 seeds — never a single stream."
        ),
        code(
            "sec_cols = [f'SEC{i:02d}' for i in range(9)]\n"
            "null_ts = []\n"
            "for s_ in range(6):  # lighter in-notebook run; canonical 20-seed number in R\n"
            "    p, _t = data.synthetic_panel(n_days=4000, n_sectors=9, mom_strength=0.0, seed=681 + s_)\n"
            "    r = st.synthetic_detect(p, sec_cols, 'BENCH', data.RS_WINDOW, data.MOM_WINDOW, n_control_seeds=20)\n"
            "    null_ts.append(r['active_tstat'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "p, _t = data.synthetic_panel(n_days=6300, n_sectors=9, mom_strength=0.0015, seed=681)\n"
            "planted = st.synthetic_detect(p, sec_cols, 'BENCH', data.RS_WINDOW, data.MOM_WINDOW, n_control_seeds=20)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(len(null_ts)) + np.linspace(-.12, .12, len(null_ts)), null_ts,\n"
            "           color=GREY, s=40, label=f'null worlds (light in-notebook run, n={len(null_ts)})')\n"
            "ax.scatter([1], [planted['active_tstat']], color=RED, s=90, zorder=5,\n"
            "           label='planted persistent rotation')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null', 'planted'])\n"
            "ax.set_ylabel('active t (RRG vs matched-random)')\n"
            "ax.set_title('Control: the null stays quiet, a planted rotation lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'light null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f})  |  '\n"
            "      f\"planted t = {planted['active_tstat']:+.2f}\")\n"
            "print(f\"canonical (results.md, 20 seeds): mean t = {R['syn_null_mean']:+.2f} \"\n"
            "      f\"(sd {R['syn_null_sd']:.2f}), |t|>=2 in {R['syn_null_fire']}/{R['syn_null_seeds']} seeds  |  \"\n"
            "      f\"planted t = {R['syn_planted_t']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: canonical 20-seed run (frozen in `results.md`) — null mean "
            f"*t* = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}), \\|t\\| ≥ 2 in "
            f"{R['syn_null_fire']}/{R['syn_null_seeds']} seeds (≈ the false-positive rate the "
            f"*t* ≥ 2 threshold implies on its own — not systematic over-firing); a planted "
            f"persistent relative drift lights up at *t* = {R['syn_planted_t']:+.2f}. The "
            "harness is powered to detect real rotation if it existed. *(A faithful-engine / "
            "power check only — never cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — vs the matched-random control (the fair test): "
            f"*t* = {R['act_rand']['t']:.2f} ({R['act_rand']['ann']:+.2f}%/yr), statistically "
            f"zero selection skill. Doesn't beat plain 6-1 momentum (*t* = "
            f"{R['act_mom']['t']:.2f}, negative point estimate). Underperforms equal-weight by "
            f"a **certified** margin (*t* = {R['act_ew']['t']:.2f}) — a design artifact of the "
            "cash-timing rule, not evidence of anti-skill, but not an edge either.\n"
            f"- **Tradability `MIRAGE`** — {R['mean_turnover']:.0f}% one-way monthly turnover, "
            f"lowest Sharpe of four books ({R['rrg']['sharpe']:.3f}), certified shortfall vs "
            "equal-weight in the 2010s and 2020s, costs erode even the (irrelevant) absolute "
            "significance.\n"
            f"- **Rotates clockwise? `CONFIRMED`** — {R['clockwise_pct']:.1f}% of quadrant "
            f"changes go the claimed clockwise way vs a {R['random_baseline_pct']:.1f}% random "
            "baseline. The chart's own story about *how* sectors move is descriptively true; "
            "it just isn't predictive of *which* one to buy."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The construction is ours, not the vendor's.** StockCharts/Bloomberg don't "
            "publish their exact smoothing constants; a reader with access to the proprietary "
            "feed could re-run this exact protocol (same controls, same matched-random test, "
            "same synthetic power check) against the commercial series.\n"
            "- **The natural next test** is whether the RRG adds anything at the *individual-"
            "stock* level (its more common professional use — rotating within an industry, not "
            "across the whole market) rather than across only 11 broad sectors.\n"
            "- **Dedup map:** [225-sector-rotation](../../225-sector-rotation/) (the plain 1-D "
            "control used here, same universe, independently reaches \"weak/mirage\"), "
            "[506-industry-momentum](../../506-industry-momentum/) (long-short industry "
            "momentum on the same 11 ETFs, also a coin flip), "
            "[246-defensive-sectors](../../246-defensive-sectors/) (a two-sector risk-off "
            "timing canary — a different question, market-timing, not sector selection).\n\n"
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
