"""Generate (and execute) the two narrative notebooks for Study 290 (September-Effect).

    python notebooks/build_notebooks.py

This writes BOTH notebooks and then executes them in-place with nbconvert
(no --allow-errors).  The synthetic generator and the frozen ``R`` dict run
anywhere, offline and deterministically; the real S&P 500 cells use the
repo-level Shiller parquet at ``_cache/shiller_sp500.parquet`` if present, and
otherwise fall back to the frozen headline numbers in ``R`` (mirroring
docs/results.md) plus a synthetic tape, so the notebooks re-run for any reader.

The _write convention (each build_*() ends by calling _write) is kept so the
repo's intro-restyle tooling can monkeypatch it.
"""

from __future__ import annotations

import os
import subprocess
import sys

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-17).
R = dict(
    n_total=912,
    n_sep=76,
    n_other=836,
    sep_mean_pct=0.0,
    other_mean_pct=0.79,
    gap_pct=-0.79,
    sep_up_rate=0.579,
    all_up_rate=0.627,
    hac_t_sep=0.0,
    hac_t_gap=-1.926,
    welch_t=-1.807,
    welch_p=0.0743,
    perm_p=0.0345,
    bt_buyhold_ann_pct=8.253,
    bt_strat_gross_ann_pct=8.326,
    bt_strat_net_ann_pct=8.226,
    bt_gross_edge_pp=0.073,
    bt_net_edge_pp=-0.027,
    oct_mean_pct=-0.063,
    year_start=1950,
    year_end=2025,
    fp="b4e287ff931c",
)

R_CELL = (
    "# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-17)\n"
    "R = dict(\n"
    "    n_total=912, n_sep=76, n_other=836,\n"
    "    sep_mean_pct=0.0, other_mean_pct=0.79, gap_pct=-0.79,\n"
    "    sep_up_rate=0.579, all_up_rate=0.627,\n"
    "    hac_t_sep=0.0, hac_t_gap=-1.926, welch_t=-1.807, welch_p=0.0743, perm_p=0.0345,\n"
    "    bt_buyhold_ann_pct=8.253, bt_strat_gross_ann_pct=8.326,\n"
    "    bt_strat_net_ann_pct=8.226, bt_gross_edge_pp=0.073, bt_net_edge_pp=-0.027,\n"
    "    oct_mean_pct=-0.063, year_start=1950, year_end=2025,\n"
    ")\n"
)

# ---------------------------------------------------------------------------
# Shared preamble
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from september_effect import data, strategy as st

def _have_cache():
    try:
        data.fetch_sp500_monthly()
        return True
    except FileNotFoundError:
        return False

HAVE_REAL = _have_cache()
print("Shiller cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# September-Effect -- is September really the market's worst month?\n"
            "### The most famous calendar omen on Wall Street, tested honestly, in plain English\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "Every late summer the financial press dusts off the same warning: **September is "
            "the stock market's worst month** -- the one calendar month with a negative long-run "
            "average return. It even has a sibling slogan, *'Sell in May and go away.'* This "
            "notebook asks the only questions that matter: is the September drag actually real, "
            "and -- even if it is -- could you ever make money dodging it?\n\n"
            "> **This is the plain-language layer.** Want the Newey-West t-stat, the permutation "
            "distribution, and the sub-period persistence check? That's "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn by "
            "the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\n" + R_CELL),

        # ---- BEAT 0 -- VERDICT ----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is September's average return really negative/flat? | **Almost exactly flat:** "
            "**+0.00%/mo** vs **+0.79%/mo** for every other month -- a real **-0.79pp/mo** gap. |\n"
            "| Is that gap statistically solid? | **Borderline.** Newey-West *t* = **-1.93** -- "
            "it leans negative but does *not* clear the robust |t| >= 2 bar. |\n"
            "| Is September actually the *worst* month? | **No** -- in this 1950-2025 window "
            "**October** is marginally worse (**-0.06%/mo**). September is the famous one, not the worst. |\n"
            "| Could you trade it? | **No.** Sitting out September gains **+0.07pp/yr gross** -- "
            "and goes **negative net** of the two trades a year and taxes it costs. |\n\n"
            "> September really is a soft month -- but the drag is small, statistically borderline, "
            "and worth essentially nothing once you try to trade it."
        ),

        # ---- BEAT 1 -- THE CLAIM --------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"September is the worst month for stocks. On average the S&P 500 actually "
            "loses money in September -- it is the only month with a negative long-run average "
            "return. Smart money lightens up before Labor Day.\"*\n\n"
            "This is folklore with real academic pedigree: the 'September effect' (and its cousin "
            "the Halloween / 'Sell in May' indicator) has been documented across decades and "
            "across many countries. Unlike the Super Bowl indicator, there is no obvious "
            "data-snooping origin -- September genuinely is soft. The interesting question is "
            "*how* soft, and whether 'soft' is the same thing as 'tradable.'"
        ),

        # ---- BEAT 2 -- SO WHAT ----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If September carried a reliable, large negative premium, the fix would be trivial: "
            "spend one month a year in cash. That's the dream version of a calendar anomaly -- "
            "a free lunch you can put in your calendar app years ahead.\n\n"
            "The honest version is more interesting. A real-but-tiny seasonal can be both "
            "academically true *and* completely useless to a trader, because the edge is smaller "
            "than the costs of harvesting it. September is the textbook case."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW ----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three traps to avoid:\n\n"
            "1. **The base-rate trap.** The S&P rises in ~63% of *all* months. So 'September is "
            "down' has to be measured against the fact that most months are up -- the right "
            "comparison is September vs the **pooled other-month** mean, not against zero in a "
            "vacuum.\n"
            "2. **Serial correlation.** Monthly equity returns are mildly autocorrelated, which "
            "inflates a naive t-stat. We use a **Newey-West (HAC)** t-stat, which corrects for "
            "this -- and the robust bar for calling a signal REAL is |t| >= 2.\n"
            "3. **One month a year.** With 76 years of data there are only **76 Septembers**. "
            "That is a small sample, and a single bad September (1974, 2001, 2008) moves the "
            "average a lot.\n\n"
            "We test on the Shiller S&P 500 monthly price series, 1950-2025 (912 months, "
            "76 Septembers)."
        ),

        # ---- BEAT 4 -- THE TEARDOWN -----------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First, the monthly profile:** the average S&P price return for each calendar "
            "month, with September called out."
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = data.fetch_sp500_monthly()\n"
            "    prof = st.monthly_profile(df)\n"
            "    means = prof['mean_pct'].values\n"
            "    labels = list(prof.index)\n"
            "else:\n"
            "    labels = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']\n"
            "    means = [1.413,0.754,0.430,1.224,0.720,0.465,0.707,0.426,\n"
            "             R['sep_mean_pct'],R['oct_mean_pct'],1.352,1.262]\n"
            "\n"
            "colors = [RED if m < 0.05 else GREY for m in means]\n"
            "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
            "bars = ax.bar(labels, means, color=colors)\n"
            "bars[8].set_color(AMBER)  # September\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Mean monthly price return (%)')\n"
            "ax.set_title('September (amber) is dead flat -- October is marginally the worst')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"September mean: {means[8]:+.3f}%/mo  |  October mean: {means[9]:+.3f}%/mo\")\n"
            "print('Every other month averages clearly positive; September stalls out.')"
        ),
        md(
            "September sits at **+0.00%/mo** -- essentially flat -- while the other eleven months "
            "average comfortably positive. It is the famous laggard. (Note the honest twist: in "
            "this exact window **October** is the only month that is actually *negative*, at "
            "-0.06%/mo. 'September is the worst' is a slogan, not a fact.)"
        ),
        md("**The base-rate view -- September vs the pooled rest.**"),
        code(
            "if HAVE_REAL:\n"
            "    s = st.september_stats(df, n_permutations=2000, seed=290)\n"
            "    sep_m, oth_m = s['sep_mean_pct'], s['other_mean_pct']\n"
            "    sep_up, all_up = s['sep_up_rate'], s['all_up_rate']\n"
            "else:\n"
            "    sep_m, oth_m = R['sep_mean_pct'], R['other_mean_pct']\n"
            "    sep_up, all_up = R['sep_up_rate'], R['all_up_rate']\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "axes[0].bar(['September','All other\\nmonths'], [sep_m, oth_m],\n"
            "            color=[AMBER, GREEN]); axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_ylabel('Mean return (%/mo)'); axes[0].set_title('Mean return')\n"
            "axes[1].bar(['September','Any month'], [sep_up*100, all_up*100],\n"
            "            color=[AMBER, GREEN])\n"
            "axes[1].axhline(50, c=RED, ls=':', lw=1)\n"
            "axes[1].set_ylabel('Up-rate (%)'); axes[1].set_title('Months that finished up')\n"
            "axes[1].set_ylim(0, 80)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'September up-rate: {sep_up:.1%}  vs  any month: {all_up:.1%}')\n"
            "print(f'Mean gap: {sep_m-oth_m:+.2f}pp/mo')"
        ),
        md(
            "September finished up only **57.9%** of the time vs **62.7%** for an average month -- "
            "and its mean return runs **0.79pp/mo below** the rest. So the drag is real in the "
            "sample. The question the quants notebook settles is whether 76 Septembers are enough "
            "to call it *statistically* real."
        ),

        # ---- BEAT 5 -- THE VERDICT ------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal -- Weak.** September genuinely lags by **-0.79pp/mo** and the one-sided "
            "permutation test flags it (p = **0.03**), but the serial-correlation-robust "
            "Newey-West *t* = **-1.93** falls just short of the |t| >= 2 bar. Real-leaning, "
            "not robustly proven.\n"
            "- **Tradability -- Mirage.** Avoiding September gains a microscopic **+0.07pp/yr** "
            "gross and turns **negative** once you pay for two trades a year and the taxes on "
            "realising the dodge. There is no money here.\n"
            "- **Honest footnote.** The drag is **price-only**; September pays a roughly normal "
            "dividend, so the total-return gap is even smaller. And September isn't even the "
            "worst month in this window -- October is."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT -----------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "The natural rule: hold the S&P all year *except* September, when you sit in cash. "
            "Here's what that buys you."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bt = st.avoid_september_backtest(df, one_way_bps=5.0)\n"
            "    bh, gross, net = bt['buyhold_ann_pct'], bt['strat_gross_ann_pct'], bt['strat_net_ann_pct']\n"
            "else:\n"
            "    bh, gross, net = R['bt_buyhold_ann_pct'], R['bt_strat_gross_ann_pct'], R['bt_strat_net_ann_pct']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "bars = ax.bar(['Buy & hold','Avoid Sep\\n(gross)','Avoid Sep\\n(net of costs)'],\n"
            "              [bh, gross, net], color=[GREEN, GREY, RED], width=0.55)\n"
            "ax.set_ylabel('Annualised return (%)')\n"
            "ax.set_title('Dodging September: a rounding error gross, a loss net')\n"
            "ax.set_ylim(min(bh,net)-0.3, max(bh,gross)+0.3)\n"
            "for b, v in zip(bars, [bh, gross, net]):\n"
            "    ax.annotate(f'{v:.2f}%', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Buy & hold:        {bh:.3f}%/yr')\n"
            "print(f'Avoid Sep (gross): {gross:.3f}%/yr  (edge {gross-bh:+.3f}pp)')\n"
            "print(f'Avoid Sep (net):   {net:.3f}%/yr  (edge {net-bh:+.3f}pp)')"
        ),
        md(
            "Gross, avoiding September adds **+0.07pp/yr** -- a rounding error. Net of just two "
            "trades a year at 5 bps one-way it is **negative**, and we haven't even charged the "
            "capital-gains tax that selling every August would trigger on a taxable account. The "
            "soft month is real; the trade is a mirage."
        ),

        # ---- BEAT 7 -- GOING FURTHER ----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The positive control:** the companion notebook plants a known September drag in a "
            "synthetic tape and shows the same test recovers it cleanly -- so the borderline real "
            "result is about the *market and sample size*, not a broken method.\n"
            "- **The sibling seasonals:** the broader autumn / 'Sell in May' family lives in the "
            "October-effect and Halloween-indicator studies; September is the single-month slice.\n\n"
            "*Think September is a tradable omen? Fork this, add the risk-free rate to the cash "
            "leg and a realistic tax model, and show a net edge that clears costs. That's the "
            "bar -- and a 0.07pp/yr gross edge doesn't get close.*"
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
            "# September-Effect -- a quantitative teardown\n"
            "### 912 months * Shiller S&P 500 * Newey-West HAC t * permutation * "
            "Welch t * sub-period persistence * n=76 power\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.* We isolate September "
            "against the pooled other-month baseline, run a Newey-West HAC t-test on the September "
            "dummy (the headline), a one-sided permutation test, a Welch two-sample test, a "
            "sub-period persistence check, and a buy-and-hold benchmark for the avoidance rule.\n\n"
            "> **Not investment advice.** Real data: Shiller S&P 500 monthly *price* index "
            "(1950-2025, price-only, no dividends). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The plain-English notes** translate each result back to intuition."
        ),
        code(BOOT + "\n" + R_CELL + "\nfrom scipy import stats as scipy_stats\n"),

        # ---- BEAT 0 ---------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            "| **Signal** | `WEAK` | September mean **+0.00%/mo** vs **+0.79%/mo** rest "
            "(gap **-0.79pp**); permutation p = **0.034** but HAC *t* = **-1.93** "
            "(below the |t| >= 2 bar). |\n"
            "| **Tradability** | `MIRAGE` | Avoid-September gross edge **+0.07pp/yr**, "
            "**net -0.03pp/yr** after two trades a year -- before tax. |\n\n"
            "> The drag is real and well-attested in the literature, but on the modern S&P it does "
            "not clear a serial-correlation-robust significance bar, and it is far too small to "
            "trade. Real-leaning Signal, Mirage Tradability."
        ),

        # ---- BEAT 1 ---------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $r_t$ be the S&P 500 monthly price return and $D_t = \\mathbf{1}[\\text{month}(t)=\\text{Sep}]$. "
            "The hypotheses:\n\n"
            "- **H1 (drag).** In the regression $r_t = \\alpha + \\beta D_t + \\varepsilon_t$, "
            "$\\beta < 0$ with a serial-correlation-robust |t| >= 2. This is the headline test.\n"
            "- **H2 (two-sample).** $E[r_t \\mid \\text{Sep}] < E[r_t \\mid \\text{not Sep}]$ "
            "by a Welch t-test.\n"
            "- **H3 (persistence).** The drag is present across sub-periods, not driven by one "
            "era (or one catastrophic September).\n\n"
            "We find H1 **borderline** (HAC t = -1.93, just under the bar), H2 directionally "
            "supportive but not significant (p = 0.074), and H3 **failing** -- the drag is "
            "concentrated in 1986-2005."
        ),

        # ---- BEAT 2 ---------------------------------------------------------
        md(
            "## 2 - So what? -- what rides on each answer\n\n"
            "The September effect is the rare calendar anomaly with genuine academic support "
            "(Bouman & Jacobsen 2002 on the broader Halloween seasonal; many country studies). "
            "If H1-H3 held robustly it would be a real, exploitable seasonal. The finding here is "
            "subtler: the *direction* is right and the literature is real, but on a "
            "serial-correlation-robust test with n=76 Septembers the effect is statistically "
            "borderline and economically negligible -- a cautionary tale about the gap between "
            "'documented' and 'tradable.'"
        ),

        # ---- BEAT 3 ---------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Data.** Shiller S&P 500 monthly nominal *price* index, December-anchored "
            "month-over-month simple returns, 1950-2025 (912 months, 76 Septembers).\n"
            "- **Headline test.** OLS $r_t = \\alpha + \\beta D_t + \\varepsilon_t$ with a "
            "**Newey-West HAC** covariance (Bartlett kernel, $\\lfloor 4(n/100)^{2/9}\\rfloor$ lags).\n"
            "- **Permutation.** One-sided: shuffle which 76 months carry the September label "
            "10,000 times; p = fraction of shuffles with mean <= observed September mean.\n"
            "- **Welch t-test.** September returns vs the pooled rest, unequal variance.\n"
            "- **Persistence.** Sub-period September means and HAC t's.\n"
            "- **Positive control.** Synthetic tape with a planted September drag, swept in size.\n"
        ),

        # ---- BEAT 4 ---------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - The headline: Newey-West HAC t on the September dummy\n\n"
            "Monthly returns are mildly autocorrelated, so a naive OLS t overstates significance. "
            "The HAC t is the honest number."
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = data.fetch_sp500_monthly()\n"
            "    s = st.september_stats(df, n_permutations=10_000, seed=290)\n"
            "    gap, hac_gap = s['gap_pct'], s['hac_t_gap']\n"
            "    welch_t, welch_p = s['welch_t'], s['welch_p']\n"
            "    perm_p = s['perm_p']\n"
            "    sep_m, oth_m = s['sep_mean_pct'], s['other_mean_pct']\n"
            "else:\n"
            "    gap, hac_gap = R['gap_pct'], R['hac_t_gap']\n"
            "    welch_t, welch_p = R['welch_t'], R['welch_p']\n"
            "    perm_p = R['perm_p']\n"
            "    sep_m, oth_m = R['sep_mean_pct'], R['other_mean_pct']\n"
            "\n"
            "print(f'September mean: {sep_m:+.3f}%/mo   Other months: {oth_m:+.3f}%/mo')\n"
            "print(f'Gap (Sep - rest): {gap:+.3f}pp/mo')\n"
            "print()\n"
            "print(f'Newey-West HAC t on the Sep dummy: {hac_gap:+.3f}')\n"
            "print(f'Welch two-sample t (Sep vs rest):  {welch_t:+.3f}  (p = {welch_p:.4f})')\n"
            "print()\n"
            "print('Robust bar for Signal = REAL: |HAC t| >= 2.')\n"
            "print(f'We observe |{hac_gap:.2f}| -- just short. Verdict: WEAK, not REAL.')"
        ),
        md(
            "> The September dummy carries a HAC *t* of **-1.93** -- it leans clearly negative but "
            "does not clear the |t| >= 2 robustness bar. The Welch test agrees: p = **0.074**, "
            "directionally right, not significant at 5%. The effect is *real-leaning*, not "
            "*robustly real*."
        ),
        md(
            "### 4b - The permutation test -- and why it disagrees with HAC\n\n"
            "Shuffle the September label 10,000 times and see where the real September mean lands."
        ),
        code(
            "if HAVE_REAL:\n"
            "    perm_means = s['perm_means'] * 100\n"
            "    obs = s['sep_mean_pct']\n"
            "else:\n"
            "    rng0 = np.random.default_rng(290)\n"
            "    allr = rng0.normal(R['other_mean_pct']/100, 0.037, R['n_total'])\n"
            "    perm_means = np.array([rng0.choice(allr, R['n_sep'], replace=False).mean()\n"
            "                           for _ in range(4000)]) * 100\n"
            "    obs = R['sep_mean_pct']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(perm_means, bins=40, color=GREY, alpha=0.75,\n"
            "        label='Shuffled September label')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'Observed Sep mean: {obs:+.2f}%')\n"
            "ax.set_xlabel('Mean monthly return (%)'); ax.set_ylabel('Count')\n"
            "ax.set_title('One-sided permutation: the real September lands in the left tail')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'One-sided permutation p (mean <= observed): {perm_p:.4f}')\n"
            "print('Note the tension: the permutation flags it (p<0.05) but assumes i.i.d. months;')\n"
            "print('the HAC t corrects for serial correlation and does NOT clear the bar.')"
        ),
        md(
            "The naive permutation p = **0.034** *looks* significant -- but it treats months as "
            "i.i.d. The HAC t-test, which corrects for the mild autocorrelation in monthly "
            "returns, is the more conservative and more honest number, and it stalls at -1.93. "
            "When two valid tests disagree at the margin, the robust one wins: **Weak**, not Real."
        ),
        md(
            "### 4c - Persistence: is the drag stable across eras?\n\n"
            "A real seasonal should show up in every sub-period, not hide in one."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sub = st.subperiod_table(df)\n"
            "    periods = list(sub.index)\n"
            "    smeans = sub['sep_mean_pct'].values\n"
            "    stt = sub['sep_hac_t'].values\n"
            "else:\n"
            "    periods = ['1950-1985','1986-2005','2006-2025']\n"
            "    smeans = [0.225, -0.923, 0.518]\n"
            "    stt = [0.413, -1.093, 0.899]\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "colors = [RED if m < 0 else GREEN for m in smeans]\n"
            "bars = ax.bar(periods, smeans, color=colors, width=0.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('September mean (%/mo)')\n"
            "ax.set_title('The drag lives almost entirely in 1986-2005 -- not a stable seasonal')\n"
            "for b, m, t in zip(bars, smeans, stt):\n"
            "    ax.annotate(f'{m:+.2f}%\\nt={t:+.2f}', (b.get_x()+b.get_width()/2, m),\n"
            "                ha='center', va='bottom' if m>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "for p, m, t in zip(periods, smeans, stt):\n"
            "    print(f'{p}: Sep mean {m:+.2f}%/mo, HAC t {t:+.2f}')"
        ),
        md(
            "> The September drag is **not** stable: it is mildly positive in 1950-1985 "
            "(+0.23%/mo), strongly negative only in 1986-2005 (-0.92%/mo, the era of 1987, 2001, "
            "2008 autumn shocks), and positive again in 2006-2025 (+0.52%/mo). A genuine "
            "structural seasonal would persist; this looks like a few bad autumns clustered in one "
            "window. Strong evidence against a robust, exploitable effect."
        ),
        md(
            "### 4d - Power: how big a drag could n=76 even detect?\n\n"
            "The minimum detectable September gap at 80% power, two-sided, alpha=0.05."
        ),
        code(
            "vol_mo = 0.0367   # September monthly vol (~3.7%)\n"
            "n_sep = R['n_sep']\n"
            "n_oth = R['n_other']\n"
            "alpha = 0.05\n"
            "from scipy.stats import norm\n"
            "z_crit = norm.ppf(1 - alpha/2)\n"
            "z_pow = norm.ppf(0.80)\n"
            "mde = (z_crit + z_pow) * vol_mo * np.sqrt(1/n_sep + 1/n_oth)\n"
            "print(f'Minimum detectable Sep-vs-rest gap (80% power, 2-sided, alpha={alpha}):')\n"
            "print(f'  vol/mo = {vol_mo:.1%}, n_sep = {n_sep}, n_other = {n_oth}')\n"
            "print(f'  MDE = {mde*100:.2f}pp/mo')\n"
            "print()\n"
            "print(f'Observed gap: {R[\"gap_pct\"]:+.2f}pp/mo')\n"
            "print(f'That is {abs(R[\"gap_pct\"])/(mde*100):.0%} of the MDE -- right on the edge of detectability.')"
        ),

        # ---- BEAT 5 ---------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal `WEAK`** -- gap -0.79pp/mo, HAC *t* = -1.93 (below |t| >= 2), Welch "
            "p = 0.074, permutation p = 0.034 (but i.i.d.-assuming), and the drag fails the "
            "persistence check (concentrated in 1986-2005). Real-leaning, not robustly real.\n"
            "- **Tradability `MIRAGE`** -- the avoidance rule gains +0.07pp/yr gross, -0.03pp/yr "
            "net, before tax. No exploitable edge."
        ),

        # ---- BEAT 6 ---------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- the buy-and-hold benchmark\n\n"
            "Hold the S&P all year except September (cash, 0%); charge two trades a year."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bt = st.avoid_september_backtest(df, one_way_bps=5.0)\n"
            "    bh, gross, net = bt['buyhold_ann_pct'], bt['strat_gross_ann_pct'], bt['strat_net_ann_pct']\n"
            "    work = df.copy()\n"
            "    years = work['year'].values\n"
            "    bh_cum = (1+work['ret']).cumprod().values\n"
            "    flat = np.where(work['month'].values==9, 0.0, work['ret'].values)\n"
            "    st_cum = (1+flat).cumprod()\n"
            "    dates = work['date'].values\n"
            "else:\n"
            "    bh, gross, net = R['bt_buyhold_ann_pct'], R['bt_strat_gross_ann_pct'], R['bt_strat_net_ann_pct']\n"
            "    rng2 = np.random.default_rng(290)\n"
            "    dates = pd.date_range('1950-01-31', periods=R['n_total'], freq='ME')\n"
            "    r = rng2.normal(R['other_mean_pct']/100, 0.037, R['n_total'])\n"
            "    bh_cum = (1+r).cumprod()\n"
            "    flat = np.where(dates.month==9, 0.0, r)\n"
            "    st_cum = (1+flat).cumprod()\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10.0, 4.5))\n"
            "ax.plot(dates, bh_cum, lw=1.6, c=GREEN, label=f'Buy & hold: {bh:.2f}%/yr')\n"
            "ax.plot(dates, st_cum, lw=1.6, c=RED, ls='--', label=f'Avoid Sep (gross): {gross:.2f}%/yr')\n"
            "ax.set_yscale('log'); ax.set_xlabel('Year')\n"
            "ax.set_ylabel('Growth of 1 (log scale)')\n"
            "ax.set_title('Avoid-September vs buy-and-hold: visually indistinguishable')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Buy & hold:        {bh:.3f}%/yr')\n"
            "print(f'Avoid Sep (gross): {gross:.3f}%/yr  (edge {gross-bh:+.3f}pp)')\n"
            "print(f'Avoid Sep (net):   {net:.3f}%/yr  (edge {net-bh:+.3f}pp) -- before tax')"
        ),
        md(
            "> The two equity curves lie on top of each other. The gross edge is a rounding error; "
            "net of trading it is negative; and on a taxable book selling every August realises "
            "gains that compound the drag. There is no trade."
        ),

        # ---- BEAT 7 ---------------------------------------------------------
        md(
            "## 7 - Going further -- the synthetic positive control\n\n"
            "Does the machinery detect a September drag *when one exists*? We plant a known "
            "September premium in a synthetic tape and sweep its size."
        ),
        code(
            "planted = [0, -100, -200, -500, -1000]\n"
            "rows = []\n"
            "for bps in planted:\n"
            "    df_s, _ = data.synthetic_monthly(n_years=76, sep_bps=float(bps), seed=290)\n"
            "    rr = st.september_stats(df_s, n_permutations=500, seed=99)\n"
            "    rows.append({'sep_bps': bps, 'gap_pct': rr['gap_pct'],\n"
            "                 'hac_t_gap': rr['hac_t_gap'], 'perm_p': rr['perm_p']})\n"
            "ctrl = pd.DataFrame(rows)\n"
            "\n"
            "colors = [GREEN if abs(r['hac_t_gap']) >= 2 else RED for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar([str(r['sep_bps']) for r in rows], [r['hac_t_gap'] for r in rows], color=colors)\n"
            "ax.axhline(-2, c='k', ls=':', lw=1, label='|HAC t| = 2 bar')\n"
            "ax.axhline(2, c='k', ls=':', lw=1)\n"
            "ax.set_xlabel('Planted September drag (bps/mo)')\n"
            "ax.set_ylabel('HAC t on the Sep dummy')\n"
            "ax.set_title('The engine clears the bar once the planted drag is real (green)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(ctrl.to_string(index=False))"
        ),
        md(
            "The engine recovers a planted September drag cleanly: once the drag is around "
            "-200 bps/mo or larger the HAC t blows past the |t| >= 2 bar. The real tape's "
            "~-79 bps/mo gap sits right at the edge of detectability with only 76 Septembers -- "
            "the verdict is a statement about the **market and sample size**, not the method."
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


def _execute(name):
    path = os.path.join(HERE, name)
    subprocess.run(
        [
            sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
            "--inplace", "--ExecutePreprocessor.timeout=600", path,
        ],
        check=True,
    )
    print("executed", path)


if __name__ == "__main__":
    build_curious()
    build_quants()
    _execute("01_for_the_curious.ipynb")
    _execute("02_for_the_quants.ipynb")
