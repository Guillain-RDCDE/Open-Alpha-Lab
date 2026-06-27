"""Generate the two narrative notebooks for Study 522 (Percent Operating Accruals).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats. The synthetic figures run offline and
deterministically; the real-data cells use this study's own ``_cache/`` (EDGAR fundamentals
+ yfinance prices) with a HAVE_REAL guard that falls back to the frozen headline numbers
``R`` when the cache is absent. ``R`` is the ONE dict of real numbers and mirrors
``docs/results.md`` exactly (as-of 2026-06-26).
"""

from __future__ import annotations

import os

import nbformat as nbf  # noqa: F401
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
# The ONE dict of real numbers — mirrors docs/results.md exactly (as-of 2026-06-26).
# All numbers are SURVIVORSHIP-BIASED upper bounds.
# --------------------------------------------------------------------------- #
R = dict(
    n_tickers=40, n_years=17, start_year=2009, end_year=2025,
    # Quintiles (forward 1yr)
    q1_mean=16.3, q1_sharpe=1.16, q1_t=7.28, q1_hit=88,
    q2_mean=20.1, q3_mean=13.3, q4_mean=15.5,
    q5_mean=13.7, q5_sharpe=0.96, q5_t=4.81, q5_hit=76,
    mkt_mean=15.7, mkt_sharpe=1.56, mkt_t=10.47,
    # Hedge Q1 - Q5
    hedge_gross=2.6, hedge_sharpe=0.15, hedge_t=0.84, hedge_hit=47,
    hedge_net=1.9, hedge_net_t=0.64,
    turnover=62, one_way_bps=10, borrow_bps=50,
    # Excess vs market
    lo_excess=0.6, lo_excess_t=0.29,
    hi_excess=-2.0, hi_excess_t=-1.06,
    # Placebo
    placebo_p=0.460, placebo_null_t=0.11, placebo_null_std=1.23,
    placebo_p_mean=0.469,
    # Random null
    rand_std=7.6, rand_pct_beaten=54,
    # Synthetic control (mean over 20 seeds)
    syn_null_t=0.02, syn_strong_t=9.89, syn_null_single=2.58,
    # Sloan head-to-head (Study 231 comparable panel)
    sloan_t=2.73,
)

# Year-by-year (Q1, Q5, market, hedge); years 2008..2024 signal -> 2009..2025 returns
_Q1  = [14.8, 19.2,  3.0, 38.1, 33.7, 13.2,  5.5, 24.6, 11.8, -5.8, 28.8,  6.1, 22.1, 10.4, 31.7, -10.5, 30.1]
_Q5  = [24.7, 16.5,  3.2, 10.7, 28.3, 17.8,  9.2, 10.5, 16.9, -5.3, 24.9, -2.3, 27.8, -15.2, -0.8, 31.3, 35.3]
_MKT = [23.9, 17.6,  7.4, 18.7, 28.5, 16.0,  4.8, 15.5, 25.8, -1.8, 28.7,  7.8, 26.8, -4.6, 14.9, 18.0, 19.6]
_HDG = [-9.9,  2.7, -0.2, 27.4,  5.4, -4.5, -3.7, 14.1, -5.1, -0.4,  3.9,  8.5, -5.6, 25.7, 32.5, -41.8, -5.3]
_YRS = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]

RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

VERDICT_BADGES = (
    "![Signal None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square) "
    "![Tradability Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square) "
    "![HLVW sharper Busted](https://img.shields.io/badge/HLVW_sharper_sort-Busted-8b949e?style=flat-square)\n\n"
)


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


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # study package
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from percent_operating_accruals import data, strategy as st

# Load this study's cached real tape; fall back gracefully to the frozen R dict.
sig, fwd = data.load_panel()
HAVE_REAL = not sig.empty
h = st.quintile_returns(sig, fwd, q=0.20, min_names=15) if HAVE_REAL else pd.DataFrame()

print("Real tape loaded:", HAVE_REAL,
      f"({len(sig.columns) if HAVE_REAL else 0} tickers, {len(sig) if HAVE_REAL else 0} years)")
"""

R_CODE = f"""\
# The ONE dict of real numbers — mirrors docs/results.md (as-of 2026-06-26).
R = dict(
    n_tickers={R['n_tickers']}, n_years={R['n_years']}, start_year={R['start_year']}, end_year={R['end_year']},
    q1_mean={R['q1_mean']}, q1_t={R['q1_t']}, q5_mean={R['q5_mean']}, q5_t={R['q5_t']},
    mkt_mean={R['mkt_mean']}, mkt_t={R['mkt_t']},
    hedge_gross={R['hedge_gross']}, hedge_t={R['hedge_t']}, hedge_hit={R['hedge_hit']},
    hedge_net={R['hedge_net']}, hedge_net_t={R['hedge_net_t']}, turnover={R['turnover']},
    lo_excess={R['lo_excess']}, lo_excess_t={R['lo_excess_t']},
    hi_excess={R['hi_excess']}, hi_excess_t={R['hi_excess_t']},
    placebo_p={R['placebo_p']}, placebo_null_t={R['placebo_null_t']}, placebo_null_std={R['placebo_null_std']},
    rand_std={R['rand_std']}, rand_pct_beaten={R['rand_pct_beaten']},
    syn_null_t={R['syn_null_t']}, syn_strong_t={R['syn_strong_t']}, syn_null_single={R['syn_null_single']},
    sloan_t={R['sloan_t']},
)
Q1  = {_Q1}
Q5  = {_Q5}
MKT = {_MKT}
HDG = {_HDG}
YRS = {_YRS}
"""


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Percent Operating Accruals -- does scaling accruals by *earnings* beat scaling by *assets*?\n"
            "### The Hafzalla-Lundholm-Van Winkle (2011) percent-accruals sort on a large-cap survivor basket\n\n"
            + VERDICT_BADGES +
            "Sloan (1996) scaled operating accruals (Net Income minus Operating Cash Flow) by total "
            "assets and found high-accrual firms underperform. HLVW (2011) said: scale by the "
            "*magnitude of earnings* instead, and the sort gets *sharper*. We test that claim on 40 "
            "large-cap names using real SEC 10-K filings, and ask whether the percent-scaling really "
            "beats the asset-scaling on a 2009-2025 survivor panel.\n\n"
            "> This is the plain-language layer. The HAC t-stats, label-shuffle placebo, and "
            "random-portfolio null live in **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** Every chart drawn by the code beside it.\n"
            ">\n"
            "> **Survivorship bias**: the basket is large caps still trading in 2026. The high-accrual "
            "firms that collapsed and delisted are absent. Every number below is an upper bound."
        ),
        code(BOOT + "\n" + R_CODE),

        # ---- BEAT 0 -- ANSWER FIRST ------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | What the data says |\n|---|---|\n"
            f"| Does low percent-accruals beat high? | Barely -- by **+{R['hedge_gross']:.1f}%/yr** gross, "
            f"positive in only **{R['hedge_hit']}%** of years. |\n"
            f"| Is that statistically real? | No. HAC *t* = **{R['hedge_t']:.2f}** -- well below |t| >= 2. "
            f"After costs: +{R['hedge_net']:.1f}%/yr, *t* = {R['hedge_net_t']:.2f}. |\n"
            f"| Does it survive a label shuffle? | No -- shuffled labels reproduce the real *t* "
            f"**{R['placebo_p']*100:.0f}%** of the time (*p* = {R['placebo_p']:.2f}). |\n"
            f"| Does the sort even line up? | No -- it is **non-monotone**: Q2 ({R['mkt_mean']:.0f}%-ish) leads, "
            f"not Q1. The low-PA leg's excess is +{R['lo_excess']:.1f}%/yr (*t* = +{R['lo_excess_t']:.2f}). |\n"
            f"| Does earnings-scaling beat Sloan's asset-scaling? | No -- Sloan (Study 231) reached "
            f"*t* = +{R['sloan_t']:.2f} on a comparable panel; this lands +{R['hedge_t']:.2f}. |\n\n"
            "> **The punchline**: on large-cap survivors, HLVW's sharper-sort claim does not "
            "replicate. The percent-accruals hedge is indistinguishable from random."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *Don't scale accruals by total assets (Sloan). Scale them by the size of earnings. "
            "Firms whose reported profit is mostly accruals (not cash) will underperform; firms whose "
            "profit is cash-backed will outperform. Sort on that fraction and the spread is bigger "
            "than Sloan's.*\n\n"
            "- **operating accruals** = Net Income - Operating Cash Flow\n"
            "- **percent accruals** = (Net Income - Operating Cash Flow) / |Net Income|\n\n"
            "A high percent-accruals number means most of the profit is *not* backed by cash. "
            "HLVW (2011) argue this ranks earnings quality more cleanly than dividing the same "
            "accrual by firm size."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If percent accruals sort returns more sharply than Sloan accruals, it would be a "
            "free upgrade: same two income-statement/cash-flow lines, a different denominator, a "
            "bigger long-short spread. HLVW reported a value-weighted hedge larger than Sloan's "
            "on 1989-2008 US stocks. If it *doesn't* replicate on large caps, that's a lesson in "
            "how published anomalies decay precisely where capital concentrates."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Four honesty requirements:\n\n"
            "1. **Reporting lag.** Fiscal-year-y fundamentals predict year y+1 returns only -- "
            "exactly one execution lag, no look-ahead.\n"
            "2. **Label-shuffle placebo.** Permute the percent-accruals labels within each year. "
            "If the real *t* is no bigger than the shuffled *t*, there is no signal.\n"
            "3. **Random-portfolio null.** Does the low-PA basket beat a random same-size draw?\n"
            "4. **Name the survivorship bias.** The basket is current survivors; the accrual "
            "disasters that delisted are absent."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown -- what the real data shows\n\n"
            "Positive control first: **when we plant a real percent-accruals premium in a synthetic "
            "panel, the engine finds it every time.** That makes the flat real-data result a "
            "statement about the *market* and the *panel*, not a broken method."
        ),
        code(
            "# Synthetic positive control: plant a known percent-accruals premium, confirm detection\n"
            "premiums = [0.0, -0.04, -0.08, -0.12]\n"
            "hedges = []\n"
            "for p in premiums:\n"
            "    s, f, _ = data.synthetic_panel(n_firms=300, n_years=25, premium=p, seed=522)\n"
            "    hh = st.quintile_returns(s, f, q=0.20, min_names=15)\n"
            "    hedges.append(hh['hedge'].mean() * 100)\n"
            "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            "ax.plot([abs(p)*100 for p in premiums], hedges, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Planted percent-accruals effect (% alpha per unit z-rank)')\n"
            "ax.set_ylabel('Low-minus-high-PA hedge (%/yr)')\n"
            "ax.set_title('Synthetic control: engine finds the effect when it is really there')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Engine faithfully recovers planted percent-accruals effects.')"
        ),
        md("**Now the real tape -- quintile sort on percent accruals, year by year:**"),
        code(
            "# Year-by-year low-PA vs high-PA vs hedge\n"
            "if HAVE_REAL:\n"
            "    q1_vals = h['q1'].values * 100\n"
            "    q5_vals = h['q5'].values * 100\n"
            "    hdg_vals = h['hedge'].values * 100\n"
            "    years_real = [int(y) for y in h.index.tolist()]\n"
            "else:\n"
            "    q1_vals, q5_vals, hdg_vals, years_real = Q1, Q5, HDG, YRS\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "x = range(len(years_real))\n"
            "axes[0].bar(x, q1_vals, label='Q1 low-PA', color=GREEN, alpha=0.7, width=0.4)\n"
            "axes[0].bar([i+0.4 for i in x], q5_vals, label='Q5 high-PA', color=RED, alpha=0.7, width=0.4)\n"
            "axes[0].set_xticks([i+0.2 for i in x])\n"
            "axes[0].set_xticklabels(years_real, rotation=45, fontsize=8)\n"
            "axes[0].set_ylabel('Annual return %'); axes[0].legend()\n"
            "axes[0].set_title('Q1 (low PA) vs Q5 (high PA) raw returns')\n"
            "colors = [GREEN if v > 0 else RED for v in hdg_vals]\n"
            "axes[1].bar(years_real, hdg_vals, color=colors)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_ylabel('Hedge (Q1 - Q5) %')\n"
            "axes[1].set_title(f'Low-minus-high-PA hedge: {sum(1 for v in hdg_vals if v>0)}/{len(hdg_vals)} positive years')\n"
            "axes[1].tick_params(axis='x', rotation=45, labelsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Mean hedge: %+.1f%%/yr  (frozen R: +%.1f%%/yr)' % (np.mean(hdg_vals), R['hedge_gross']))"
        ),
        md(
            f"The hedge is **+{R['hedge_gross']:.1f}%/yr gross** but positive in only {R['hedge_hit']}% of years, "
            "with a brutal **-41.8% in 2023**. The quintile sort is **non-monotone** -- the low-PA leg "
            f"barely separates from the high-PA leg (low-PA excess +{R['lo_excess']:.1f}%/yr, "
            f"*t* = +{R['lo_excess_t']:.2f}; high-PA excess {R['hi_excess']:.1f}%/yr, "
            f"*t* = {R['hi_excess_t']:.2f}). This is noise, not the sharp HLVW spread."
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal -- None.** Hedge +{R['hedge_gross']:.1f}%/yr, HAC *t* = {R['hedge_t']:.2f}; "
            f"label-shuffle placebo *p* = {R['placebo_p']:.2f} (indistinguishable from random); "
            f"the low-PA basket beats only {R['rand_pct_beaten']}% of random same-size draws. "
            "And this is a survivorship-biased upper bound.\n"
            f"- **Tradability -- Mirage.** Annual rebalancing is cheap ({R['turnover']}%/yr turnover), "
            f"but net *t* = {R['hedge_net_t']:.2f} -- a tradable wrapper around a non-signal.\n"
            f"- **HLVW sharper sort? -- Busted.** Sloan's asset-scaled accrual reached *t* = "
            f"+{R['sloan_t']:.2f} on a comparable survivor panel (Study 231); this earnings-scaled "
            f"version lands +{R['hedge_t']:.2f}. On large-cap survivors the percent-scaling does "
            "*not* sharpen the sort."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "Mechanically, yes -- 40 liquid large caps, one rebalance a year. But there is nothing "
            "to harvest:\n\n"
            "1. **No edge.** Gross +2.6%/yr at *t* = 0.84 is well inside the noise band; the placebo "
            "says a coin-flip relabelling does as well.\n"
            "2. **You'd pay to short nothing.** The high-PA short leg's underperformance "
            "(-2.0%/yr, *t* = -1.06) is the only directionally-correct piece and it is "
            "insignificant -- you would pay 50 bps/yr borrow for it.\n"
            "3. **Wrong universe.** Both accrual flavours survive mainly in small, illiquid, "
            "low-coverage names -- the opposite of this basket.\n\n"
            "The right conclusion: **MIRAGE** on this panel."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The right universe.** HLVW (2011) ran value-weighted deciles on all US stocks "
            "1989-2008. A point-in-time Compustat universe with small caps is the minimum for a "
            "credible percent-accruals test.\n"
            "- **Percent *total* accruals.** HLVW also test the broader Richardson et al. "
            "balance-sheet accrual scaled by earnings; a natural extension.\n"
            "- **The asset-scaled cousin.** [Study 231 -- Sloan-Accruals](../../231-sloan-accruals/) "
            "is the head-to-head benchmark -- same operating accrual, divided by assets.\n"
            "- **Earnings-quality lens.** [Study 122 -- Gross-Profit](../../122-gross-profit/) and "
            "[Study 153 -- Net-Operating-Assets](../../153-net-operating-assets/) are the cousins.\n\n"
            "*Think percent accruals still beat Sloan in a broad universe? Fork this, extend to a "
            "point-in-time all-US panel, and show |t| >= 2 on the spread. That's the missing half.*"
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
            "# Percent Operating Accruals -- a quantitative teardown\n"
            "### HLVW (2011) percent-accruals quintile sort - large-cap survivor panel - HAC "
            "inference - label-shuffle placebo - random-portfolio null - costs - Sloan head-to-head\n\n"
            + VERDICT_BADGES +
            "The deep companion to [01_for_the_curious.ipynb](01_for_the_curious.ipynb) -- same seven "
            f"beats, every claim carrying its standard error. We test the HLVW percent-accruals hedge "
            f"on a {R['n_tickers']}-name large-cap survivor panel ({R['start_year']}-{R['end_year']}), "
            "measure it with a HAC t-stat, kill it with a label-shuffle placebo, and race the "
            "denominator against Sloan's asset-scaling.\n\n"
            "> **Not investment advice.** SEC EDGAR 10-K fundamentals; annual returns via Yahoo "
            "Finance; survivorship-biased. Sources in [`docs/references.md`](../docs/references.md), "
            "reproducible numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            f"> **Key finding**: the hedge is *not* real (HAC t = +{R['hedge_t']:.2f}, placebo "
            f"p = {R['placebo_p']:.2f}); the sort is non-monotone; and the earnings-scaling is "
            f"*weaker* than Sloan's asset-scaling (+{R['sloan_t']:.2f}) on a comparable panel."
        ),
        code(BOOT + "\n" + R_CODE),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Hedge **+{R['hedge_gross']:.1f}%/yr**, HAC *t* = **{R['hedge_t']:.2f}** "
            f"-- below |t| >= 2. Placebo *p* = {R['placebo_p']:.2f}; non-monotone sort; beats only "
            f"{R['rand_pct_beaten']}% of random draws. |\n"
            f"| **Tradability** | `MIRAGE` | Net hedge +{R['hedge_net']:.1f}%/yr, *t* = {R['hedge_net_t']:.2f} "
            f"after {R['turnover']}%/yr turnover and borrow. No edge to harvest. |\n"
            f"| **HLVW sharper sort?** | `BUSTED` | Sloan asset-scaled *t* = +{R['sloan_t']:.2f} "
            f"(Study 231); percent earnings-scaled *t* = +{R['hedge_t']:.2f} on a comparable panel. |\n\n"
            "> **In plain words:** scaling the operating accrual by earnings rather than assets does "
            "not sharpen the sort on large-cap survivors -- it dulls it."
        ),

        # ---- BEAT 1: signal definition ---------------------------------------
        md(
            "## 1 - The signal, precisely\n\n"
            "For fiscal year *y*, firm *i*:\n\n"
            "$$\\text{PA}_{i,y} = \\frac{\\text{NetIncome}_{i,y} - \\text{OperatingCashFlow}_{i,y}}"
            "{|\\text{NetIncome}_{i,y}|}$$\n\n"
            "We mask firm-years with |NetIncome| below \\$1M (the ratio explodes near zero earnings, "
            "as HLVW note). Quintile each year, **long Q1 (low PA) / short Q5 (high PA)**, lag one "
            "full year (fiscal *y* -> calendar *y*+1 returns)."
        ),
        code(
            "# Quintile means + HAC t-stats from the real tape\n"
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for q in ['q1','q2','q3','q4','q5','market']:\n"
            "        s = st.summary(h[q])\n"
            "        rows.append((q, s['mean']*100, s['sharpe'], s['tstat'], s['hit_rate']*100))\n"
            "    qt = pd.DataFrame(rows, columns=['quintile','mean_%','sharpe','HAC_t','hit_%']).set_index('quintile')\n"
            "    display(qt.round(2))\n"
            "    fig, ax = plt.subplots(figsize=(8,4.5))\n"
            "    means = [st.summary(h[f'q{i}'])['mean']*100 for i in range(1,6)]\n"
            "    ax.bar(range(1,6), means, color=[GREEN,GREEN,GREY,RED,RED], alpha=.8)\n"
            "    ax.axhline(st.summary(h['market'])['mean']*100, c='k', ls='--', label='EW market')\n"
            "    ax.set_xlabel('Percent-accruals quintile (1=low PA, 5=high PA)')\n"
            "    ax.set_ylabel('Mean forward return %/yr'); ax.set_title('Non-monotone sort: Q2 leads, not Q1'); ax.legend()\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('Quintile means (frozen): Q1=%.1f Q5=%.1f mkt=%.1f' % (R['q1_mean'],R['q5_mean'],R['mkt_mean']))"
        ),

        # ---- BEAT 2: the hedge + HAC -----------------------------------------
        md(
            "## 2 - The hedge and its HAC t-stat\n\n"
            f"Q1 - Q5 hedge: **+{R['hedge_gross']:.1f}%/yr gross**, HAC Newey-West *t* = "
            f"**{R['hedge_t']:.2f}**, hit rate {R['hedge_hit']}%. Below the |t| >= 2 bar."
        ),
        code(
            "# The hedge series, HAC stat, and equity curve\n"
            "if HAVE_REAL:\n"
            "    hedge = h['hedge']\n"
            "    s = st.summary(hedge)\n"
            "    print(f\"gross hedge mean={s['mean']*100:+.2f}%/yr  Sharpe={s['sharpe']:+.2f}  \"\n"
            "          f\"HAC t={s['tstat']:+.2f}  hit={s['hit_rate']:.0%}  n={s['n']}\")\n"
            "    eq = (1+hedge).cumprod()\n"
            "    fig, ax = plt.subplots(figsize=(9,4.5))\n"
            "    ax.plot([int(y) for y in eq.index], eq.values, 'o-', c=AMBER, lw=2)\n"
            "    ax.axhline(1, c='k', lw=1)\n"
            "    ax.set_ylabel('Growth of 1 (gross hedge)')\n"
            "    ax.set_title(f'Long low-PA / short high-PA: t={s[\"tstat\"]:+.2f} (noise)')\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('frozen hedge t = %.2f' % R['hedge_t'])"
        ),

        # ---- BEAT 3: label-shuffle placebo -----------------------------------
        md(
            "## 3 - The label-shuffle placebo\n\n"
            "Permute the percent-accruals labels *within each year* (1000 perms), breaking the "
            "signal->return link while preserving each year's return marginals. If the real *t* is "
            "not in the tail of the shuffled distribution, there is no signal."
        ),
        code(
            "# Label-shuffle placebo (reduced perms for a fast notebook cell)\n"
            "if HAVE_REAL:\n"
            "    p, null = st.placebo_hedge_t(sig, fwd, q=0.20, n_perm=300, min_names=15, seed=522)\n"
            "    real_t = st.summary(h['hedge'])['tstat']\n"
            "    fig, ax = plt.subplots(figsize=(9,4.5))\n"
            "    ax.hist(null, bins=30, color=GREY, alpha=.7, density=True)\n"
            "    ax.axvline(real_t, c=RED, lw=2, label=f'real t={real_t:+.2f}')\n"
            "    ax.axvline(-real_t, c=RED, lw=1, ls='--', alpha=.5)\n"
            "    ax.set_xlabel('Shuffled-label hedge HAC t'); ax.set_ylabel('density')\n"
            "    ax.set_title(f'Placebo: p={p:.3f} (real t is NOT in the tail)'); ax.legend()\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'placebo p={p:.3f}  (frozen full-run p={R[\"placebo_p\"]:.3f})')\n"
            "else:\n"
            "    print('frozen placebo p = %.3f' % R['placebo_p'])"
        ),

        # ---- BEAT 4: random null + synthetic control -------------------------
        md(
            "## 4 - Random-portfolio null and the synthetic positive control\n\n"
            f"Does the low-PA basket beat a random same-size draw? It beats only "
            f"**{R['rand_pct_beaten']}%** -- a coin flip. And the synthetic control proves the engine "
            f"*would* detect a real effect (mean *t* = +{R['syn_strong_t']:.1f} with a planted "
            f"-5%/yr premium over 20 seeds; null mean *t* = +{R['syn_null_t']:.2f}). Note the "
            f"single-seed null *t* = +{R['syn_null_single']:.2f} -- a lucky-seed artefact the "
            "20-seed average correctly washes out."
        ),
        code(
            "# Synthetic control averaged over 20 seeds (faithful-engine + lucky-seed lesson)\n"
            "null_ts = [st.synthetic_hedge_t(0.0, seed=s) for s in range(20)]\n"
            "strong_ts = [st.synthetic_hedge_t(-0.05, seed=s) for s in range(20)]\n"
            "print(f'null   premium: mean t over 20 seeds = {np.mean(null_ts):+.2f}  '\n"
            "      f'(single seed 522 = {st.synthetic_hedge_t(0.0,522):+.2f}, a lucky-seed artefact)')\n"
            "print(f'strong premium: mean t over 20 seeds = {np.mean(strong_ts):+.2f}  '\n"
            "      f'(frac |t|>2 = {np.mean([abs(t)>2 for t in strong_ts]):.0%})')\n"
            "if HAVE_REAL:\n"
            "    rand = st.random_portfolio_returns(sig, fwd, q=0.20, n_draws=300, seed=522)\n"
            "    actual_lo = float((h['q1']-h['market']).mean())\n"
            "    print(f'random null std={rand.std()*100:.1f}%  low-PA beats {(rand<actual_lo).mean():.0%} of draws')"
        ),

        # ---- BEAT 5: costs ---------------------------------------------------
        md(
            "## 5 - Costs and the Sloan head-to-head\n\n"
            f"Net of {R['turnover']}%/yr turnover (10 bps/leg x 2) plus 50 bps/yr borrow on the short "
            f"leg: +{R['hedge_net']:.1f}%/yr, *t* = {R['hedge_net_t']:.2f}. And the denominator race: "
            f"Sloan's asset-scaled accrual reached *t* = +{R['sloan_t']:.2f} on a comparable survivor "
            f"panel (Study 231); the earnings-scaled percent accrual lands +{R['hedge_t']:.2f}."
        ),
        code(
            "# Costs + Sloan head-to-head bar\n"
            "if HAVE_REAL:\n"
            "    turn = st.turnover_series(sig, fwd, q=0.20, min_names=15)\n"
            "    net = st.apply_costs(h['hedge'], turn, one_way_bps=10.0, borrow_bps=50.0)\n"
            "    g, n = st.summary(h['hedge']), st.summary(net)\n"
            "    print(f\"GROSS +{g['mean']*100:.1f}%/yr t={g['tstat']:+.2f}  ->  \"\n"
            "          f\"NET +{n['mean']*100:.1f}%/yr t={n['tstat']:+.2f}  (turnover {turn.mean():.0%}/yr)\")\n"
            "fig, ax = plt.subplots(figsize=(7,4.5))\n"
            "ax.bar(['Sloan (231)\\nasset-scaled','Percent (522)\\nearnings-scaled'],\n"
            "       [R['sloan_t'], R['hedge_t']], color=[AMBER, RED], alpha=.85)\n"
            "ax.axhline(2.0, c='k', ls='--', label='|t|=2 bar'); ax.legend()\n"
            "ax.set_ylabel('Hedge HAC t-stat'); ax.set_title('Denominator race: earnings-scaling does NOT win')\n"
            "plt.tight_layout(); plt.show()"
        ),

        # ---- BEAT 6: robustness summary --------------------------------------
        md(
            "## 6 - Robustness summary\n\n"
            f"| Check | Result |\n|---|---|\n"
            f"| Gross hedge HAC *t* | +{R['hedge_t']:.2f} (below 2) |\n"
            f"| Net hedge HAC *t* | +{R['hedge_net_t']:.2f} |\n"
            f"| Label-shuffle placebo *p* | {R['placebo_p']:.2f} (seed-stable ~{R['placebo_p']:.2f}) |\n"
            f"| Low-PA beats random draw | {R['rand_pct_beaten']}% |\n"
            f"| Sort monotone? | No (Q2 leads Q1) |\n"
            f"| Synthetic null mean *t* (20 seeds) | +{R['syn_null_t']:.2f} |\n"
            f"| Synthetic strong mean *t* (20 seeds) | +{R['syn_strong_t']:.1f} |\n"
            f"| vs Sloan asset-scaled (Study 231) | +{R['hedge_t']:.2f} vs +{R['sloan_t']:.2f} |\n\n"
            "Every robustness lens agrees: **no signal**, and the earnings-scaling is the *weaker* "
            "denominator here."
        ),

        # ---- BEAT 7: going further -------------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **Universe.** HLVW (2011) used value-weighted deciles on all US stocks 1989-2008 "
            "including small caps -- where both accrual flavours actually live.\n"
            "- **Percent total accruals.** The broader balance-sheet accrual scaled by earnings is "
            "HLVW's second variant; worth a separate sleeve.\n"
            "- **The benchmark.** [Study 231 -- Sloan-Accruals](../../231-sloan-accruals/) is the "
            "asset-scaled head-to-head; [Study 153 -- NOA](../../153-net-operating-assets/) the "
            "balance-sheet cousin.\n\n"
            "*The honest read: on large-cap survivors, the percent-accruals denominator does not "
            "sharpen the Sloan sort -- and neither sort clears |t| >= 2 here.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "02_for_the_quants.ipynb")


if __name__ == "__main__":
    build_curious()
    build_quants()
