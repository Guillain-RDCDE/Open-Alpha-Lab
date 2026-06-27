"""Generate the two narrative notebooks for Study 529 (Inventory-Growth).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the desk beats. The synthetic figures run offline and
deterministically; the real-data cells use this study's own ``_cache/`` with a HAVE_REAL
guard that falls back to the frozen headline numbers ``R`` when the cache is absent.
``R`` is the ONE dict of real numbers and mirrors ``docs/results.md`` exactly.
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# THE ONE DICT OF REAL NUMBERS — mirrors docs/results.md (as-of 2026-06-26).
# Survivor-biased, n=2 usable hedge years, signal NOT certifiable.
# ---------------------------------------------------------------------------
R = dict(
    n_basket=40, n_usable=37, n_hedge_years=2,
    invg_fp="044108b907ab",
    # Quintiles (forward 1yr, INVG low->high)
    q1_mean=17.3, q2_mean=-5.9, q3_mean=12.3, q4_mean=21.1, q5_mean=12.7, mkt_mean=11.3,
    lo_mean=17.3, hi_mean=12.7,
    # Hedge (Q1 - Q5): long low-INVG, short high-INVG
    hedge_gross=4.6, hedge_gross_sharpe=0.18, hedge_gross_t=0.50, hedge_hit=50,
    cost_drag=0.6, hedge_net=4.0, hedge_net_t=0.44,
    # Placebo / label-shuffle null
    placebo_p=0.376, null_mean=0.6, null_std=13.2,
    # Random-portfolio control
    lo_excess=6.0, rand_beat=71, rand_std=10.2,
    # Synthetic positive control (seed-robust, 20 seeds)
    sc_null_t=-0.22, sc_03_t=6.59, sc_06_t=13.40, sc_10_t=22.49,
    sc_null_frac=5, sc_03_frac=100, sc_06_frac=100, sc_10_frac=100,
)

# Year-by-year frozen (2 years)
_YRS = [2023, 2024]
_Q1  = [28.8, 5.7]
_Q5  = [5.9, 19.4]
_MKT = [13.8, 8.7]
_HDG = [22.9, -13.7]

VERDICT_BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Data depth: Insufficient](https://img.shields.io/badge/Data_depth-Insufficient-8b949e?style=flat-square)\n\n"
)

# ---------------------------------------------------------------------------
# Shared boot code
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # study package (inventory_growth/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from inventory_growth import data, strategy as st

# Try this study's own cache; fall back gracefully to the frozen R dict.
invg, fwd = data.fetch_panel()
HAVE_REAL = not invg.empty
if HAVE_REAL:
    h = st.quintile_returns(invg, fwd, q=0.20, min_names=10)
else:
    h = pd.DataFrame()
print("Real cache loaded:", HAVE_REAL,
      f"({invg.shape[1] if HAVE_REAL else 0} names, {invg.shape[0] if HAVE_REAL else 0} signal years)")
"""

R_CODE = f"""\
R = dict(
    n_basket={R['n_basket']}, n_usable={R['n_usable']}, n_hedge_years={R['n_hedge_years']},
    invg_fp="{R['invg_fp']}",
    q1_mean={R['q1_mean']}, q2_mean={R['q2_mean']}, q3_mean={R['q3_mean']}, q4_mean={R['q4_mean']},
    q5_mean={R['q5_mean']}, mkt_mean={R['mkt_mean']}, lo_mean={R['lo_mean']}, hi_mean={R['hi_mean']},
    hedge_gross={R['hedge_gross']}, hedge_gross_sharpe={R['hedge_gross_sharpe']},
    hedge_gross_t={R['hedge_gross_t']}, hedge_hit={R['hedge_hit']},
    cost_drag={R['cost_drag']}, hedge_net={R['hedge_net']}, hedge_net_t={R['hedge_net_t']},
    placebo_p={R['placebo_p']}, null_mean={R['null_mean']}, null_std={R['null_std']},
    lo_excess={R['lo_excess']}, rand_beat={R['rand_beat']}, rand_std={R['rand_std']},
    sc_null_t={R['sc_null_t']}, sc_03_t={R['sc_03_t']}, sc_06_t={R['sc_06_t']}, sc_10_t={R['sc_10_t']},
)
YRS = {_YRS}
Q1  = {_Q1}
Q5  = {_Q5}
MKT = {_MKT}
HDG = {_HDG}
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
            "# Inventory-Growth -- do firms that pile up stock go on to underperform?\n"
            "### The Belo-Lin (2012) / Thomas-Zhang (2002) inventory-growth anomaly\n\n"
            + VERDICT_BADGES +
            "Frederico Belo & Xiaoji Lin (2012) and Jacob Thomas & Huai Zhang (2002) argue "
            "that firms which build inventory aggressively earn *lower* future returns -- the "
            "inventory bulge signals over-investment, or demand that fails to show up, and the "
            "stock disappoints. We test it on a basket of inventory-heavy survivors (retailers, "
            "consumer-goods makers, industrials, automakers) using real Yahoo balance sheets.\n\n"
            "> This is the plain-language layer. The HAC *t*-stats, the label-shuffle placebo, "
            "and the synthetic control live in "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** Every chart drawn by the code beside it. "
            "House style in [METHODOLOGY.md](../../../METHODOLOGY.md).\n"
            ">\n"
            "> **Two big caveats up front:** (1) the basket is names *still trading in 2026* -- "
            "the over-builders that failed are missing (survivorship bias); (2) Yahoo serves "
            "only ~4 years of balance sheets, leaving just **two** usable test years."
        ),
        code(BOOT + "\n" + R_CODE),

        # ---- BEAT 0 -- ANSWER FIRST ------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | What the data says |\n|---|---|\n"
            f"| Do lean-inventory firms beat the over-builders? | The hedge is "
            f"**{R['hedge_gross']:+.1f}%/yr** but with only **{R['n_hedge_years']}** "
            f"usable years -- and it flips sign between them. |\n"
            f"| Is that statistically real? | One-sample HAC *t* = "
            f"**{R['hedge_gross_t']:.2f}** -- nowhere near |t| >= 2. |\n"
            f"| Does it beat a coin-flip label shuffle? | No -- placebo *p* = "
            f"**{R['placebo_p']:.2f}**. The real sort is inside the noise. |\n"
            "| Could you trade it? | No. Two years can't certify anything, and the basket "
            "excludes the very firms the anomaly is built around. |\n\n"
            "> **The catch**: Yahoo's `balance_sheet` only goes ~4 fiscal years deep. After a "
            "one-year reporting lag and a complete-calendar-year return filter, only **2** "
            "hedge years survive. The honest verdict here is *untestable*, not *disproven*."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *Sell stocks whose inventory grew fast (relative to their balance sheet). "
            "Buy stocks whose inventory stayed lean. Rebalance annually. The over-builders "
            "will disappoint as the unsold stock is discounted or written down.*\n\n"
            "The signal is the inventory *change* scaled by last year's total assets:\n\n"
            "- **INVG_t** = (Inventory_t - Inventory_{t-1}) / Total Assets_{t-1}\n\n"
            "Scaling by lagged assets (Thomas & Zhang 2002) keeps the number comparable across "
            "firms of very different inventory intensity. A high INVG means the firm bulked up "
            "stock fast -- maybe ahead of demand that never came."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If the inventory-growth effect is real and tradable, it is a low-turnover, "
            "fundamentals-only signal needing only annual reports -- and it slots into the "
            "broader real-investment factor (high investment -> low returns) that underpins "
            "the Fama-French CMA factor. If it only shows up on a broad, decades-deep "
            "Compustat universe and vanishes on a shallow survivor slice, that is itself the "
            "lesson: the right test needs the right data."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three honesty requirements:\n\n"
            "1. **One reporting lag.** Fiscal-year-y inventory growth only predicts year-y+1 "
            "returns -- no peeking.\n"
            "2. **A placebo.** Shuffle the inventory-growth labels within each year and re-run "
            "the sort. If the real hedge doesn't beat the shuffled null, there is no signal.\n"
            "3. **Name the limits.** The basket is 2026 survivors (failed over-builders are "
            "gone), and Yahoo gives only ~4 years of statements. Both are stated, not hidden."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown -- the engine works, the data doesn't reach\n\n"
            "Positive control first: **when we plant a real inventory-growth effect in a "
            "synthetic panel, the engine finds it every time.** So the flat real result is "
            "about the *data*, not the *method*."
        ),
        code(
            "# Synthetic positive control: plant a known INVG premium, confirm the engine detects it\n"
            "premiums = [0.0, -0.03, -0.06, -0.10]\n"
            "hedges = []\n"
            "for p in premiums:\n"
            "    r = st.seed_robust_synthetic_t(p, n_seeds=20, n_firms=200, n_years=20)\n"
            "    hedges.append(r['mean_hedge'] * 100)\n"
            "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            "ax.plot([abs(p)*100 for p in premiums], hedges, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Planted INVG effect (% alpha per unit z-rank)')\n"
            "ax.set_ylabel('Low-minus-high-INVG hedge (%/yr)')\n"
            "ax.set_title('Synthetic control: engine recovers the effect when planted (20-seed avg)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Engine is faithful -> the real result is a data statement, not a method bug.')"
        ),
        md("**Now the real tape -- the two usable years, side by side:**"),
        code(
            "if HAVE_REAL:\n"
            "    q1_vals = [h.loc[y, 'q1']*100 for y in h.index]\n"
            "    q5_vals = [h.loc[y, 'q5']*100 for y in h.index]\n"
            "    hdg_vals = [h.loc[y, 'hedge']*100 for y in h.index]\n"
            "    years_real = list(h.index)\n"
            "else:\n"
            "    q1_vals, q5_vals, hdg_vals, years_real = Q1, Q5, HDG, YRS\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "x = range(len(years_real))\n"
            "axes[0].bar(x, q1_vals, label='Q1 low-INVG (lean)', color=GREEN, alpha=0.8, width=0.4)\n"
            "axes[0].bar([i+0.4 for i in x], q5_vals, label='Q5 high-INVG (build)', color=RED, alpha=0.8, width=0.4)\n"
            "axes[0].set_xticks([i+0.2 for i in x]); axes[0].set_xticklabels(years_real)\n"
            "axes[0].set_ylabel('Annual return %'); axes[0].legend()\n"
            "axes[0].set_title('Lean vs over-builder quintiles')\n"
            "colors = [GREEN if v > 0 else RED for v in hdg_vals]\n"
            "axes[1].bar([str(y) for y in years_real], hdg_vals, color=colors)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_ylabel('Hedge return (Q1 - Q5) %')\n"
            "axes[1].set_title('The hedge FLIPS sign between the only two years')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Mean hedge: {np.mean(hdg_vals):+.1f}%/yr  (frozen R: {R[\"hedge_gross\"]:+.1f}%/yr)')"
        ),
        md(
            f"The hedge swings **{_HDG[0]:+.1f}%** in {_YRS[0]} then **{_HDG[1]:+.1f}%** in "
            f"{_YRS[1]} -- exactly the coin-flip pattern a two-observation sample produces when "
            "there is no underlying signal. The mean is "
            f"**{R['hedge_gross']:+.1f}%/yr** at HAC *t* = **{R['hedge_gross_t']:.2f}**, and the "
            f"label-shuffle placebo clears it at *p* = **{R['placebo_p']:.2f}**."
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal -- None.** Hedge {R['hedge_gross']:+.1f}%/yr, HAC *t* = "
            f"{R['hedge_gross_t']:.2f}, placebo *p* = {R['placebo_p']:.2f}, sign flips "
            "year-to-year. No certifiable effect.\n"
            f"- **Tradability -- Mirage.** Nets {R['hedge_net']:+.1f}%/yr (*t* "
            f"{R['hedge_net_t']:.2f}) after costs + borrow over two years. Nothing to trade.\n"
            "- **Data depth -- Insufficient.** ~4 balance-sheet years from Yahoo -> 2 usable "
            "hedge years. The anomaly is untestable here, not disproven."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "No -- and the reasons are about the test, not the idea:\n\n"
            "1. **Two years.** You cannot certify a cross-sectional anomaly on two annual "
            "hedge observations. The standard error swamps everything.\n"
            "2. **Survivorship.** The basket holds 2026 survivors. The firms that drowned in "
            "unsold inventory -- the ones that would drive the predicted sign -- are gone.\n"
            "3. **Wrong-depth data.** Belo & Lin (2012) and Thomas & Zhang (2002) used decades "
            "of point-in-time Compustat data across the full cross-section. Yahoo's four-deep "
            "annual history is simply the wrong tool for this job.\n\n"
            "The right conclusion: **explore this on a deep, point-in-time, broad universe "
            "before treating it as tradable.**"
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The right data.** A Compustat (or SEC-EDGAR-XBRL) pull going back to the 1990s "
            "across all US stocks is the minimum for a fair test -- that's what the original "
            "papers used.\n"
            "- **Asset-growth parent.** [Study 244 — Asset-Growth](../244-asset-growth/) "
            "tests the total-asset version of the same real-investment idea; inventory is one "
            "component of it.\n"
            "- **Accrual cousins.** [Study 231 — Sloan-Accruals](../231-sloan-accruals/) and "
            "[Study 522 — Percent-Operating-Accruals](../522-percent-operating-accruals/) "
            "sit on the working-capital side of the same balance-sheet-bloat family.\n\n"
            "*Think the inventory-growth spread is real? Fork this, plug in a deep point-in-time "
            "universe, and show |t| >= 2 on the hedge with no survivorship shortcut. That's the bar.*"
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
            "# Inventory-Growth -- a quantitative teardown\n"
            "### INVG quintile sort - survivor basket - one-sample HAC - label-shuffle placebo - "
            "synthetic control\n\n"
            + VERDICT_BADGES +
            "The deep companion to [01_for_the_curious.ipynb](01_for_the_curious.ipynb) -- "
            "every claim now carrying its standard error. We test the inventory-growth anomaly "
            "(Belo-Lin 2012; Thomas-Zhang 2002) on a 37-name inventory-heavy survivor basket, "
            "measure the low-minus-high hedge with a one-sample HAC *t*-stat, pin it against a "
            "label-shuffle placebo and a random-portfolio null, charge costs + short borrow, and "
            "prove the engine faithful with a 20-seed synthetic positive control.\n\n"
            "> **Not investment advice.** Yahoo `balance_sheet` + daily adjusted closes; "
            "survivorship-biased; only **2** usable hedge years. Sources in "
            "[`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Data-depth warning**: the binding constraint is that Yahoo serves ~4 annual "
            "statements per name -- this is a *power* problem, stated openly."
        ),
        code(BOOT + "\n" + R_CODE),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Hedge **{R['hedge_gross']:+.1f}%/yr**, one-sample HAC "
            f"*t* = **{R['hedge_gross_t']:+.2f}**, placebo *p* = **{R['placebo_p']:.2f}**, "
            f"sign flips across the only 2 years. |\n"
            f"| **Tradability** | `MIRAGE` | Nets **{R['hedge_net']:+.1f}%/yr** (*t* "
            f"{R['hedge_net_t']:.2f}) after {R['cost_drag']:.1f}%/yr costs+borrow. n=2. |\n"
            f"| **Data depth** | `INSUFFICIENT` | ~4 Yahoo balance-sheet years -> "
            f"{R['n_hedge_years']} usable hedge years. Untestable, not disproven. |\n\n"
            "> **In plain words:** the academic effect is documented on decades of broad "
            "Compustat data. On a shallow, survivor-biased, 2-year Yahoo slice it cannot be "
            "certified -- and the synthetic control proves that's the data's fault, not the "
            "engine's."
        ),

        # ---- BEAT 1 -----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $r_{i,t+1}$ be firm $i$'s return in year $t+1$. Define:\n\n"
            "$$\\text{INVG}_{i,t} = \\frac{\\text{Inv}_{i,t} - \\text{Inv}_{i,t-1}}"
            "{A_{i,t-1}}$$\n\n"
            "where $A$ = total assets. The **H1 (signal)** claim (Belo-Lin 2012; "
            "Thomas-Zhang 2002):\n\n"
            "$$\\mathbb{E}\\left[\\bar{r}_{\\text{low-INVG},t+1} - "
            "\\bar{r}_{\\text{high-INVG},t+1}\\right] > 0$$\n\n"
            "with a one-sample HAC *t* >= 2. **H2 (not a fluke)** requires the real hedge to "
            "beat a within-year label shuffle. **H3 (tradable)** requires it to survive costs "
            "and borrow. Here H1 fails on power before it can be judged on merit."
        ),

        # ---- BEAT 2 -----------------------------------------------------------
        md(
            "## 2 - So what? -- the stakes\n\n"
            "A real, tradable inventory-growth spread would be a clean low-turnover "
            "fundamentals signal and a building block of the real-investment / CMA factor. "
            "But the original evidence lives on a deep broad universe; the question here is "
            "whether a retail-accessible Yahoo slice can even see it. It cannot -- and saying "
            "so honestly is the point."
        ),

        # ---- BEAT 3 -----------------------------------------------------------
        md(
            "## 3 - The protocol\n\n"
            "- **Data**: Yahoo `balance_sheet` (Inventory, Total Assets) + daily adjusted "
            "closes, cached to this study's own `_cache/`.\n"
            f"- **Universe**: {R['n_basket']}-name inventory-heavy survivor basket; "
            f"{R['n_usable']} carry a usable INVG (survivorship-biased).\n"
            f"- **Window**: signal years 2023-2026; after the lag + complete-year filter, "
            f"**{R['n_hedge_years']}** usable hedge years.\n"
            "- **Signal**: inventory change / lagged total assets (higher = aggressive build).\n"
            "- **Sort**: quintile (q=0.20); Q1 = low INVG (lean), Q5 = high INVG (build).\n"
            "- **Inference**: one-sample HAC Newey-West *t* on the annual hedge series.\n"
            "- **Nulls**: within-year label shuffle (1000x) + random same-size portfolio.\n"
            "- **Costs**: 5 bps/leg x 2 legs + 50 bps short borrow.\n"
            "- **Positive control**: 20-seed synthetic panel with a planted INVG premium."
        ),

        # ---- BEAT 4 -----------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - Synthetic positive control -- the engine works when an effect is planted\n\n"
            "Sweep the planted INVG premium and average the hedge *t* over 20 seeds (a "
            "seed-robust power check -- never cited for the real stamp)."
        ),
        code(
            "premiums = [0.0, -0.03, -0.06, -0.10]\n"
            "rows = []\n"
            "for p in premiums:\n"
            "    r = st.seed_robust_synthetic_t(p, n_seeds=20, n_firms=200, n_years=20)\n"
            "    rows.append((abs(p)*100, r['mean_hedge']*100, r['mean_t'], r['frac_t_gt2']*100))\n"
            "sydf = pd.DataFrame(rows, columns=['premium_%', 'hedge_%/yr', 'mean_HAC_t', 'frac_t>2_%'])\n"
            "fig, ax = plt.subplots(figsize=(9, 4))\n"
            "colors = [GREEN if t > 2 else (RED if t < -2 else GREY) for t in sydf['mean_HAC_t']]\n"
            "ax.bar(sydf['premium_%'], sydf['mean_HAC_t'], color=colors, width=1.2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Planted INVG effect (%/yr per unit z-rank)')\n"
            "ax.set_ylabel('Mean HAC t (20 seeds)')\n"
            "ax.set_title('Positive control: t>2 (green) appears exactly where the premium is real')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(sydf.round(2).to_string(index=False))"
        ),
        md(
            f"> The engine returns mean *t* = **{R['sc_null_t']:.2f}** under the null and "
            f"**{R['sc_06_t']:.1f}** at a -6%/yr planted premium (t>2 in 100% of seeds). "
            "Faithful. The flat real result is therefore a **data** statement."
        ),
        md("### 4b - Real tape -- quintile returns and the hedge with one-sample HAC inference"),
        code(
            "if HAVE_REAL:\n"
            "    q_means = [float(h[f'q{i}'].mean()*100) for i in range(1,6)]\n"
            "    s_hedge = st.summary(h['hedge'])\n"
            "    net = st.net_hedge(h['hedge']); s_net = st.summary(net)\n"
            "else:\n"
            "    q_means = [R['q1_mean'], R['q2_mean'], R['q3_mean'], R['q4_mean'], R['q5_mean']]\n"
            "    s_hedge = dict(mean=R['hedge_gross']/100, tstat=R['hedge_gross_t'], hit_rate=R['hedge_hit']/100)\n"
            "    s_net = dict(mean=R['hedge_net']/100, tstat=R['hedge_net_t'])\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "x = [1, 2, 3, 4, 5]\n"
            "bar_colors = [GREEN, GREY, GREY, GREY, RED]\n"
            "ax.bar(x, q_means, color=bar_colors)\n"
            "ax.axhline(R['mkt_mean'], ls='--', c='k', lw=1, label=f'EW mkt {R[\"mkt_mean\"]:.1f}%')\n"
            "ax.set_xticks(x); ax.set_xticklabels(['Q1 (lean)', 'Q2', 'Q3', 'Q4', 'Q5 (build)'])\n"
            "ax.set_ylabel('Mean annual return (%/yr)'); ax.legend()\n"
            "ax.set_title('No monotone low->high pattern (n=2 years -- pure noise)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'GROSS hedge: {s_hedge[\"mean\"]*100:+.1f}%/yr  HAC t={s_hedge[\"tstat\"]:+.2f}  hit={s_hedge[\"hit_rate\"]:.0%}')\n"
            "print(f'NET   hedge: {s_net[\"mean\"]*100:+.1f}%/yr  HAC t={s_net[\"tstat\"]:+.2f}  (after costs+borrow)')"
        ),
        md(
            f"> Gross hedge **{R['hedge_gross']:+.1f}%/yr**, one-sample HAC *t* = "
            f"**{R['hedge_gross_t']:+.2f}**; net **{R['hedge_net']:+.1f}%/yr** after "
            f"{R['cost_drag']:.1f}%/yr of costs + borrow. There is no monotone low->high "
            "ordering -- the quintiles are noise on two years."
        ),
        md(
            "### 4c - Label-shuffle placebo + random-portfolio null\n\n"
            "Permute the INVG labels within each year (1000x) and re-run the hedge; then ask "
            "whether the low-INVG basket beats a blind same-size draw."
        ),
        code(
            "if HAVE_REAL:\n"
            "    p_val, null_means = st.placebo_hedge_t(invg, fwd, n_shuffles=1000, min_names=10)\n"
            "    real_mean = float(h['hedge'].mean())\n"
            "    rand = st.random_portfolio_returns(invg, fwd, n_draws=1000, min_names=10)\n"
            "    lo_ex = float((h['q1'] - h['market']).mean())\n"
            "    rand_beat = float((rand < lo_ex).mean())\n"
            "else:\n"
            "    rng_nb = np.random.default_rng(529)\n"
            "    null_means = rng_nb.normal(R['null_mean']/100, R['null_std']/100, 1000)\n"
            "    real_mean = R['hedge_gross']/100; p_val = R['placebo_p']\n"
            "    rand = rng_nb.normal(0.0, R['rand_std']/100, 2000)\n"
            "    lo_ex = R['lo_excess']/100; rand_beat = R['rand_beat']/100\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "axes[0].hist(np.asarray(null_means)*100, bins=40, color=GREY, alpha=0.7)\n"
            "axes[0].axvline(real_mean*100, c=RED, lw=2.5, label=f'real hedge {real_mean*100:+.1f}%')\n"
            "axes[0].axvline(0, c='k', lw=1); axes[0].legend()\n"
            "axes[0].set_xlabel('Shuffled hedge mean (%/yr)'); axes[0].set_ylabel('Count')\n"
            "axes[0].set_title(f'Label-shuffle placebo: p = {p_val:.2f} (real is INSIDE the noise)')\n"
            "axes[1].hist(np.asarray(rand)*100, bins=40, color=GREY, alpha=0.7)\n"
            "axes[1].axvline(lo_ex*100, c=RED, lw=2.5, label=f'low-INVG excess {lo_ex*100:+.1f}%')\n"
            "axes[1].axvline(0, c='k', lw=1); axes[1].legend()\n"
            "axes[1].set_xlabel('Excess vs EW mkt (%/yr)'); axes[1].set_ylabel('Count')\n"
            "axes[1].set_title(f'Random-portfolio null: low-INVG beats {rand_beat:.0%} of draws')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Placebo p = {p_val:.2f}; low-INVG beats {rand_beat:.0%} of random same-size draws')"
        ),
        md(
            f"> Placebo *p* = **{R['placebo_p']:.2f}** -- the real hedge sits squarely inside "
            f"the shuffled-label distribution. The low-INVG basket beats **{R['rand_beat']}%** "
            "of random draws (a noisy 2-year read, not a 50%-vs-significant separation)."
        ),

        # ---- BEAT 5 -----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal `NONE`** -- hedge {R['hedge_gross']:+.1f}%/yr, HAC *t* = "
            f"{R['hedge_gross_t']:+.2f}, placebo *p* = {R['placebo_p']:.2f}, sign flips.\n"
            f"- **Tradability `MIRAGE`** -- net {R['hedge_net']:+.1f}%/yr (*t* "
            f"{R['hedge_net_t']:.2f}) over 2 years; survivor basket; nothing to trade.\n"
            "- **Data depth `INSUFFICIENT`** -- ~4 Yahoo balance-sheet years -> 2 usable "
            "hedge years. The binding constraint is power, not direction."
        ),

        # ---- BEAT 6 -----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- the honest answer\n\n"
            "No tradable signal can be established on this tape:\n\n"
            "1. **Power.** Two annual hedge observations cannot reject the null; the standard "
            "error is enormous. The synthetic control shows the engine *would* find a real "
            "effect with enough years -- we just don't have them.\n"
            "2. **Survivorship.** The basket excludes the over-builders that failed -- the very "
            "names that drive the predicted sign in the academic samples.\n"
            "3. **Depth mismatch.** Belo & Lin (2012) and Thomas & Zhang (2002) used decades of "
            "point-in-time Compustat data. Yahoo's four-deep annual history is the wrong tool.\n\n"
            "The correct conclusion: **MIRAGE** on this test, with the honest footnote that the "
            "anomaly is *untestable here*, not disproven."
        ),

        # ---- BEAT 7 -----------------------------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **Deep universe.** A point-in-time Compustat / EDGAR-XBRL pull back to the 1990s "
            "across all US stocks is the minimum for a fair test.\n"
            "- **Sales-adjusted variant.** Thomas & Zhang (2002) use inventory growth *net of "
            "sales growth*; a sales adjustment sharpens the over-build signal -- worth adding "
            "with deeper income-statement data.\n"
            "- **Family.** [Study 244 — Asset-Growth](../244-asset-growth/) (total assets), "
            "[Study 231 — Sloan-Accruals](../231-sloan-accruals/) and "
            "[Study 522 — Percent-Operating-Accruals](../522-percent-operating-accruals/) "
            "are the neighbours on the balance-sheet-bloat block.\n\n"
            "*Want to certify the inventory-growth spread? Fork this, plug in a deep "
            "point-in-time universe, and show |t| >= 2 on the hedge with no survivorship "
            "shortcut. That's the bar.*"
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
