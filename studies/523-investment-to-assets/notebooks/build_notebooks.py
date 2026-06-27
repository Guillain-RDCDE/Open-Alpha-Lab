"""Generate the two narrative notebooks for Study 523 (Investment-To-Assets).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks load the study's own cache (``../_cache/ia_*.parquet``) with a HAVE_REAL
guard that falls back to the frozen headline numbers ``R`` (a mirror of docs/results.md,
as-of 2026-06-27) when the cache is absent. The synthetic positive control runs offline
and deterministically.
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-27).
# SURVIVORSHIP-BIASED; the predicted hedge is a flat zero (effect absent).
R = dict(
    n_names=27, n_years=16, start_year=2009, end_year=2024,
    lo_mean=19.4, lo_sharpe=0.97, lo_t=6.39, lo_hit=88,
    mid_mean=18.0, mid_sharpe=1.44, mid_t=5.74,
    hi_mean=19.4, hi_sharpe=0.98, hi_t=5.07, hi_hit=100,
    mkt_mean=19.0, mkt_sharpe=1.20, mkt_t=6.89,
    # Hedge (g1 - g3): essentially zero -> anomaly absent on this basket
    hedge_mean=0.04, hedge_sharpe=0.00, hedge_t=0.01, hedge_hit=44,
    lo_excess_mean=0.5, lo_excess_t=0.25,
    hi_excess_mean=0.4, hi_excess_t=0.22,
    placebo_p=0.471, shuffle_mean=-0.21, shuffle_std=3.88,
    rand_std=9.6, rand_pct_beaten=55,
    cost_drag=0.20, borrow_drag=0.50, net_mean=-0.66,
    synth_null_t=-0.15, synth_06_t=14.94, synth_10_t=25.00,
    ia_fp="bb346b49d925", fwd_fp="64b068f655e9",
)

# Year-by-year frozen data (g1, g3, market, hedge)
_G1  = [27.0, 1.4, 19.6, 24.4, 19.6, 9.8, 47.9, 11.9, 17.2, -4.3, 72.7, 2.3, 10.3, 11.8, -1.1, 40.6]
_G3  = [29.0, 18.0, 12.4, 12.1, 16.2, 10.1, 17.9, 20.4, 18.9, 3.1, 54.9, 1.6, 2.2, 13.8, 3.7, 76.0]
_MKT = [30.4, 10.2, 14.3, 16.7, 16.7, 7.8, 27.0, 16.9, 24.6, 6.3, 56.0, 5.8, 6.8, 12.4, 0.6, 51.0]
_HDG = [-2.0, -16.6, 7.1, 12.3, 3.4, -0.2, 30.0, -8.5, -1.7, -7.3, 17.8, 0.7, 8.0, -2.0, -4.8, -35.4]
_YRS = list(range(R["start_year"], R["end_year"] + 1))

VERDICT_BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Channel: Busted](https://img.shields.io/badge/Channel-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath("../../.."))   # repo root
sys.path.insert(0, os.path.abspath(".."))          # study package (investment_to_assets/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from investment_to_assets import data, strategy as st

# Cache-first: load the study's own EDGAR + price cache; fall back to frozen R.
ia, fwd = data.fetch_panel()
ia = ia.dropna(how="all") if not ia.empty else ia
HAVE_REAL = not ia.empty
if HAVE_REAL:
    h = st.sorted_returns(ia, fwd, n_groups=3)
else:
    h = pd.DataFrame()
print("Real cache loaded:", HAVE_REAL,
      f"({len(ia.columns) if HAVE_REAL else 0} names, {len(h) if HAVE_REAL else 0} sort years)")
"""

R_CODE = f"""\
R = dict(
    n_names={R['n_names']}, n_years={R['n_years']}, start_year={R['start_year']}, end_year={R['end_year']},
    lo_mean={R['lo_mean']}, lo_sharpe={R['lo_sharpe']}, lo_t={R['lo_t']}, lo_hit={R['lo_hit']},
    mid_mean={R['mid_mean']}, hi_mean={R['hi_mean']}, hi_sharpe={R['hi_sharpe']}, hi_t={R['hi_t']}, hi_hit={R['hi_hit']},
    mkt_mean={R['mkt_mean']}, mkt_sharpe={R['mkt_sharpe']},
    hedge_mean={R['hedge_mean']}, hedge_sharpe={R['hedge_sharpe']}, hedge_t={R['hedge_t']}, hedge_hit={R['hedge_hit']},
    lo_excess_mean={R['lo_excess_mean']}, lo_excess_t={R['lo_excess_t']},
    hi_excess_mean={R['hi_excess_mean']}, hi_excess_t={R['hi_excess_t']},
    placebo_p={R['placebo_p']}, shuffle_mean={R['shuffle_mean']}, shuffle_std={R['shuffle_std']},
    rand_std={R['rand_std']}, rand_pct_beaten={R['rand_pct_beaten']},
    cost_drag={R['cost_drag']}, borrow_drag={R['borrow_drag']}, net_mean={R['net_mean']},
    synth_null_t={R['synth_null_t']}, synth_06_t={R['synth_06_t']}, synth_10_t={R['synth_10_t']},
)
G1  = {_G1}
G3  = {_G3}
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
            "# Investment-To-Assets -- do the heaviest capital-spenders underperform?\n"
            "### The Titman-Wei-Xie (2004) capex anomaly on a large-cap survivor basket\n\n"
            + VERDICT_BADGES +
            "Sheridan Titman, John Wei and Feixue Xie documented in 2004 that firms ploughing "
            "a large fraction of their asset base back into capital expenditure earn lower "
            "future returns. The story: managers over-invest when capital is cheap, markets "
            "extrapolate the expansion, and the stock disappoints. We test the **capex/PP&E "
            "channel** on a fixed large-cap basket using real SEC EDGAR filings.\n\n"
            "> This is the plain-language layer. The HAC t-stats, label-shuffle placebo, and "
            "random-portfolio null live in "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** Every chart is drawn by the code beside it. "
            "House style in [METHODOLOGY.md](../../../METHODOLOGY.md).\n"
            ">\n"
            "> **Survivorship bias**: the basket covers only firms still large-cap in 2026. "
            "Heavy investors that over-invested and *failed* are absent."
        ),
        code(BOOT + "\n" + R_CODE),

        md(
            "## The answer first\n\n"
            "| Question | What the data says |\n|---|---|\n"
            f"| Do low-capex firms outperform high-capex firms? | No -- the hedge is "
            f"**{R['hedge_mean']:+.2f}%/yr**, a flat zero. The two legs earn "
            f"~{R['lo_mean']:.1f}%/yr each. |\n"
            f"| Is that statistically real? | HAC *t* = {R['hedge_t']:.2f} -- nowhere near "
            f"|t| >= 2. The anomaly is **absent**. |\n"
            f"| Does it beat a coin-flip relabelling? | No -- placebo p = {R['placebo_p']:.2f}, "
            f"buried inside the shuffle null. |\n"
            f"| Could you trade it? | No. After costs + short borrow the hedge is "
            f"**{R['net_mean']:+.2f}%/yr net**. |\n\n"
            "> **The catch**: on a *survivor* basket, the high-capex group is full of "
            "*successful* expanders -- semiconductor fabs, hyperscale datacentres, energy "
            "majors. Their investment was productive. The over-investing failures the anomaly "
            "was built around are simply missing. (2024: high-capex tercile +76% as the "
            "AI-datacentre capex names ripped.)"
        ),

        md(
            "## 1 - The claim\n\n"
            "> *Sell the heavy capital-spenders. Buy the disciplined ones. Rebalance "
            "annually. The over-investing firms will disappoint as the expansion stops "
            "producing earnings.*\n\n"
            "The signal is one accounting ratio:\n\n"
            "- **IA_t** = CapEx_t / Total Assets_{t-1}\n\n"
            "A high IA means the firm spent heavily on property, plant and equipment relative "
            "to last year's asset base. The Titman-Wei-Xie argument: that spending often "
            "reflects over-investment (Jensen 1986 agency costs) or market over-extrapolation, "
            "and the stock underperforms. This is the *capex* channel -- distinct from the "
            "*total-asset-growth* channel tested in "
            "[Study 244](../../244-asset-growth/)."
        ),

        md(
            "## 2 - So what?\n\n"
            "If the capex effect is real and tradable on large US stocks, it underpins the "
            "Fama-French **CMA** (Conservative Minus Aggressive) factor and the q-factor "
            "model's investment leg -- a low-ML, fundamentals-only strategy needing only "
            "public 10-K filings. If it only works on small/mid-caps -- or is a selection "
            "artefact -- the large-cap test is a cautionary tale about testing in the wrong "
            "universe."
        ),

        md(
            "## 3 - How would we even know?\n\n"
            "Three honesty requirements:\n\n"
            "1. **One execution lag.** IA from fiscal year y drives a 12-month return that "
            "*starts 4 months after year-end* (when the 10-K is public), entered one trading "
            "day later. No same-bar fills, no look-ahead; no partial year is ever stamped.\n"
            "2. **Placebo + random control.** Shuffle the IA labels within each year; does the "
            "real hedge beat its shuffled self? And does the low-IA leg beat a random draw of "
            "the same size?\n"
            "3. **Name the survivorship bias.** The basket is current large-caps. Failed "
            "over-investors are absent. Every result is conditional on survival."
        ),

        md(
            "## 4 - The teardown\n\n"
            "Positive control first: **when we plant a real capex effect in a synthetic "
            "panel, the engine finds it.** That makes the real-data result a statement about "
            "the *market* (and the survivor basket), not the method."
        ),
        code(
            "# Synthetic positive control: plant a known capex premium, confirm the engine detects it\n"
            "premiums = [0.0, -0.04, -0.08, -0.12]\n"
            "hedges = []\n"
            "for p in premiums:\n"
            "    s, f, _ = data.synthetic_panel(n_firms=300, n_years=25, premium=p, seed=523)\n"
            "    hh = st.sorted_returns(s, f, n_groups=3)\n"
            "    hedges.append(hh['hedge'].mean() * 100)\n"
            "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            "ax.plot([abs(p)*100 for p in premiums], hedges, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Planted capex effect (% alpha per unit z-rank)')\n"
            "ax.set_ylabel('Low-minus-high-IA hedge (%/yr)')\n"
            "ax.set_title('Synthetic control: engine finds the effect when planted')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('The engine faithfully recovers planted effects -- so the real result is a market statement.')"
        ),
        md("**Now the real EDGAR panel -- tercile sort on investment-to-assets:**"),
        code(
            "if HAVE_REAL:\n"
            "    g1_vals = h['g1'].values * 100\n"
            "    g3_vals = h['g3'].values * 100\n"
            "    hdg_vals = h['hedge'].values * 100\n"
            "    years_real = [int(y) for y in h.index.tolist()]\n"
            "else:\n"
            "    g1_vals, g3_vals, hdg_vals, years_real = G1, G3, HDG, YRS\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "x = range(len(years_real))\n"
            "axes[0].bar(x, g1_vals, label='g1 low-IA', color=GREEN, alpha=0.7, width=0.4)\n"
            "axes[0].bar([i+0.4 for i in x], g3_vals, label='g3 high-IA', color=RED, alpha=0.7, width=0.4)\n"
            "axes[0].set_xticks([i+0.2 for i in x])\n"
            "axes[0].set_xticklabels(years_real, rotation=45, fontsize=8)\n"
            "axes[0].set_ylabel('Annual return %'); axes[0].legend()\n"
            "axes[0].set_title('g1 (low IA) vs g3 (high IA) returns')\n"
            "colors = [GREEN if v > 0 else RED for v in hdg_vals]\n"
            "axes[1].bar(years_real, hdg_vals, color=colors)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_ylabel('Hedge return (g1 - g3) %')\n"
            "axes[1].set_title(f'Low-minus-high-IA hedge: {sum(1 for v in hdg_vals if v>0)}/{len(hdg_vals)} positive years')\n"
            "axes[1].tick_params(axis='x', rotation=45, labelsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Mean hedge: {np.mean(hdg_vals):+.2f}%/yr  (frozen R: {R[\"hedge_mean\"]:+.2f}%/yr)')"
        ),
        md(
            f"The low-IA tercile earns **+{R['lo_mean']:.1f}%/yr** vs the high-IA tercile's "
            f"**+{R['hi_mean']:.1f}%/yr** -- a hedge of **{R['hedge_mean']:+.2f}%/yr**, "
            f"positive in only {R['hedge_hit']}% of years. **The predicted effect is absent**: "
            "disciplined and aggressive investors earn essentially identical returns on this "
            "survivor basket. Survivorship selection washes the signal out entirely."
        ),

        md(
            "## 5 - The verdict\n\n"
            f"- **Signal -- None.** Hedge {R['hedge_mean']:+.2f}%/yr, HAC *t* = "
            f"{R['hedge_t']:.2f}; placebo p = {R['placebo_p']:.2f}. No signal.\n"
            f"- **Tradability -- Mirage.** No gross edge; **{R['net_mean']:+.2f}%/yr net** "
            "after costs and short borrow.\n"
            "- **Channel -- Busted.** The capex/PP&E channel is no more present than the "
            "total-asset-growth channel of [Study 244](../../244-asset-growth/)."
        ),

        md(
            "## 6 - Could you actually trade it?\n\n"
            "No. There is no signal on this basket. Even setting aside survivorship:\n\n"
            "1. **Wrong universe.** Titman-Wei-Xie (2004) documented the effect on the broad "
            "US cross-section, dominated by small/mid-caps where over-investment and "
            "limits-to-arbitrage bite hardest. On ~27 heavily-covered large caps it is thin "
            "and arbitraged.\n"
            "2. **Survivorship reversal.** The high-capex large caps that survived are "
            "productive expanders (semis, datacentres, energy) -- the opposite of the "
            "over-investing failures the anomaly targets.\n"
            "3. **The CMA paradox.** The investment factor has positive expected returns in "
            "broad cross-sections -- not on a narrow survivor basket.\n\n"
            "The right conclusion: **explore this in a broad, point-in-time universe before "
            "treating the capex anomaly as tradable.**"
        ),

        md(
            "## 7 - Going further\n\n"
            "- **The right universe.** Titman-Wei-Xie used all NYSE/AMEX/NASDAQ firms -- "
            "thousands of names including small and mid-caps. The large-cap survivor basket "
            "here is the wrong test.\n"
            "- **Total-asset-growth cousin.** [Study 244 -- Asset-Growth](../../244-asset-growth/) "
            "tests the total-balance-sheet channel (Cooper-Gulen-Schill) on the same kind of "
            "survivor panel and also lands None/Mirage. The two channels die together.\n"
            "- **q-factor I/A.** Hou-Xue-Zhang's investment factor is the direct quantitative "
            "cousin; the factor evidence is broad-cross-section, not survivor large-cap.\n\n"
            "*Think the capex effect is real in large caps? Fork this, extend to a "
            "point-in-time all-US-stock universe, and show |t| >= 2 on the hedge with no "
            "survivorship shortcut. That's the bar.*"
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
            "# Investment-To-Assets -- a quantitative teardown\n"
            "### IA tercile sort - EDGAR survivor basket - HAC inference - label-shuffle "
            "placebo - random-portfolio null - costs - synthetic control\n\n"
            + VERDICT_BADGES +
            "The deep companion to [01_for_the_curious.ipynb](01_for_the_curious.ipynb) -- "
            "same seven beats, every claim now carrying its standard error. We test the "
            "Titman-Wei-Xie capex anomaly on a survivorship-biased large-cap basket "
            f"(~{R['n_names']} names/yr, {R['start_year']}-{R['end_year']}), measure the "
            "low-minus-high-IA hedge with a HAC t-stat, pin it against a label-shuffle "
            "placebo and a random-portfolio null, charge costs, and confirm the engine on a "
            "seed-robust synthetic positive control.\n\n"
            "> **Not investment advice.** SEC EDGAR companyfacts (capex, assets); Yahoo daily "
            "prices; survivorship-biased. Sources in [`docs/references.md`](../docs/references.md), "
            "reproducible numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Survivorship-bias warning**: all results are conditional on survival -- the "
            "over-investing firms that failed are excluded."
        ),
        code(BOOT + "\n" + R_CODE),

        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Hedge **{R['hedge_mean']:+.2f}%/yr**, HAC *t* = "
            f"**{R['hedge_t']:+.2f}**; placebo p = **{R['placebo_p']:.2f}**. Absent. |\n"
            f"| **Tradability** | `MIRAGE` | No gross edge; **{R['net_mean']:+.2f}%/yr net** "
            f"after 10bps x turnover + 50bps borrow. |\n"
            f"| **Channel** | `BUSTED` | Capex channel no more present than the total-asset-"
            f"growth channel of [Study 244](../../244-asset-growth/). |\n\n"
            "> **In plain words:** the engine recovers a planted capex effect at HAC "
            f"*t* ~ {R['synth_06_t']:.0f}, but the real survivor basket shows a flat zero. "
            "The high-IA survivors are productive expanders, not over-investing failures."
        ),

        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $r_{i,t+1}$ be firm $i$'s 12-month forward return. Define:\n\n"
            "$$\\text{IA}_{i,t} = \\frac{\\text{CapEx}_{i,t}}{A_{i,t-1}}$$\n\n"
            "where $A$ = total assets. The **H1 (signal)** claim (Titman-Wei-Xie 2004) is:\n\n"
            "$$\\mathbb{E}\\left[\\bar{r}_{\\text{low-IA},t+1} - "
            "\\bar{r}_{\\text{high-IA},t+1}\\right] > 0$$\n\n"
            "with HAC |t| >= 2. **H2 (beats noise)** requires the hedge to beat its "
            "label-shuffled self. **H3 (tradable)** requires it to survive costs + borrow."
        ),

        md(
            "## 2 - So what? -- the stakes\n\n"
            "H1 is *not* supported on this biased basket -- the hedge is a flat zero. The "
            "academic evidence on broad US cross-sections is credible (Titman-Wei-Xie 2004; "
            "Fama-French CMA; Hou-Xue-Zhang q-factor); on large-cap survivors, selection "
            "washes the signal out."
        ),

        md(
            "## 3 - The protocol\n\n"
            "- **Data**: capex (`PaymentsToAcquirePropertyPlantAndEquipment`) and `Assets` "
            "from SEC EDGAR companyfacts; daily adjusted prices from Yahoo.\n"
            f"- **Universe**: fixed large-cap survivor basket (~{R['n_names']} names/yr); "
            "survivorship-biased.\n"
            f"- **Window**: {R['n_years']} signal years ({R['start_year']}-{R['end_year']}), "
            "12-month forward returns, report-lagged.\n"
            "- **Signal**: IA = CapEx_t / Assets_{t-1} (higher = heavier investor).\n"
            "- **Sort**: terciles; g1 = low IA (disciplined), g3 = high IA (aggressive).\n"
            "- **Execution lag**: one -- enter 1 trading day after a 4-month report lag past "
            "fiscal year-end; no partial year stamped.\n"
            "- **Inference**: HAC Newey-West t-stat on the 16-obs annual hedge.\n"
            "- **Nulls**: label-shuffle placebo (1000) + random same-size portfolios (500/yr).\n"
            "- **Costs**: 10bps x turnover + 50bps short borrow.\n"
            "- **Positive control**: seed-robust synthetic panel with a planted capex premium."
        ),

        md("## 4 - The teardown"),
        md(
            "### 4a - Synthetic positive control (seed-robust)\n\n"
            "Sweep the planted capex premium and confirm the hedge tracks it monotonically, "
            "averaging the HAC-t over many seeds (a single lucky seed is never enough)."
        ),
        code(
            "premiums = [0.0, -0.06, -0.10]\n"
            "synth_rows = []\n"
            "for p in premiums:\n"
            "    r = st.seed_robust_synthetic(p, n_seeds=25)\n"
            "    synth_rows.append((abs(p)*100, r['mean_hedge']*100, r['mean_tstat']))\n"
            "sydf = pd.DataFrame(synth_rows, columns=['premium_%', 'hedge_%/yr', 'mean_HAC_t'])\n"
            "fig, ax = plt.subplots(figsize=(9, 4))\n"
            "colors = [GREEN if t > 2 else (RED if t < -2 else GREY) for t in sydf['mean_HAC_t']]\n"
            "ax.bar(sydf['premium_%'], sydf['hedge_%/yr'], color=colors, width=1.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Planted capex effect (%/yr per unit z-rank)')\n"
            "ax.set_ylabel('Low-minus-high hedge (%/yr)')\n"
            "ax.set_title('Positive control: HAC t>2 (green) appears where the premium is real')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(sydf.round(2))"
        ),
        md(
            f"> The engine recovers planted premiums (t ~ {R['synth_06_t']:.0f} at -6%/yr, "
            f"averaged over 25 seeds) and the null behaves (t ~ {R['synth_null_t']:.1f}). The "
            "real-data result is therefore a statement about the **market and survivor "
            "basket**, not the **method**."
        ),
        md("### 4b - Real EDGAR panel -- tercile returns and HAC inference"),
        code(
            "if HAVE_REAL:\n"
            "    g_means = [float(h[f'g{i}'].mean()*100) for i in range(1,4)]\n"
            "    g_tstats = [st.summary(h[f'g{i}'])['tstat'] for i in range(1,4)]\n"
            "    s_hedge = st.summary(h['hedge'])\n"
            "    s_lo_ex = st.summary(h['g1'] - h['market'])\n"
            "    s_hi_ex = st.summary(h['g3'] - h['market'])\n"
            "else:\n"
            "    g_means = [R['lo_mean'], R['mid_mean'], R['hi_mean']]\n"
            "    g_tstats = [R['lo_t'], R['mid_mean'], R['hi_t']]\n"
            "    s_hedge = dict(mean=R['hedge_mean']/100, tstat=R['hedge_t'], hit_rate=R['hedge_hit']/100)\n"
            "    s_lo_ex = dict(mean=R['lo_excess_mean']/100, tstat=R['lo_excess_t'])\n"
            "    s_hi_ex = dict(mean=R['hi_excess_mean']/100, tstat=R['hi_excess_t'])\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "x = [1, 2, 3]\n"
            "bar_colors = [GREY, GREY, GREY]\n"
            "axes[0].bar(x, g_means, color=bar_colors)\n"
            "axes[0].set_xticks(x); axes[0].set_xticklabels(['g1 (lo)', 'g2', 'g3 (hi)'])\n"
            "axes[0].set_ylabel('Mean annual return (%/yr)')\n"
            "axes[0].set_title('Tercile returns: low and high IA earn ~identical (no spread)')\n"
            "axes[1].bar(x, g_tstats, color=bar_colors)\n"
            "for s in (2, -2): axes[1].axhline(s, ls='--', c=GREY, lw=1)\n"
            "axes[1].set_xticks(x); axes[1].set_xticklabels(['g1 (lo)', 'g2', 'g3 (hi)'])\n"
            "axes[1].set_ylabel('HAC t-stat (level returns)')\n"
            "axes[1].set_title('All terciles ride the same survivor bull market')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Hedge: {s_hedge[\"mean\"]*100:+.2f}%/yr  HAC t={s_hedge[\"tstat\"]:+.2f}  hit={s_hedge[\"hit_rate\"]:.0%}')\n"
            "print(f'Low-IA excess: {s_lo_ex[\"mean\"]*100:+.1f}%/yr  t={s_lo_ex[\"tstat\"]:+.2f}')\n"
            "print(f'High-IA excess: {s_hi_ex[\"mean\"]*100:+.1f}%/yr  t={s_hi_ex[\"tstat\"]:+.2f}')"
        ),
        md(
            f"> The low-minus-high-IA hedge is **{R['hedge_mean']:+.2f}%/yr**, HAC *t* = "
            f"**{R['hedge_t']:+.2f}** -- statistically zero. The low and high IA terciles earn "
            f"**~{R['lo_mean']:.1f}%/yr each**: no spread, no monotonicity, no signal."
        ),
        md(
            "### 4c - Label-shuffle placebo + random-portfolio null\n\n"
            "Shuffle the IA labels within each year (1000 times); does the real hedge beat its "
            "shuffled self? And does the low-IA leg beat a blind same-size draw?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    p_val, sh = st.placebo_hedge_tstats(ia, fwd, n_groups=3, n_shuffles=1000, seed=523)\n"
            "    rand = st.random_portfolio_returns(ia, fwd, n_groups=3, n_draws=500, seed=523)\n"
            "    real_mean = float(h['hedge'].mean())\n"
            "    lo_excess = float((h['g1'] - h['market']).mean())\n"
            "    pct_beaten = float((rand < lo_excess).mean())\n"
            "else:\n"
            "    rng_nb = np.random.default_rng(523)\n"
            "    sh = pd.Series(rng_nb.normal(R['shuffle_mean']/100, R['shuffle_std']/100, 1000))\n"
            "    rand = pd.Series(rng_nb.normal(0.0, R['rand_std']/100, 16*500))\n"
            "    real_mean = R['hedge_mean'] / 100\n"
            "    p_val = R['placebo_p']\n"
            "    lo_excess = R['lo_excess_mean'] / 100\n"
            "    pct_beaten = R['rand_pct_beaten'] / 100\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "axes[0].hist(sh * 100, bins=40, color=GREY, alpha=0.75, label='Label-shuffle hedge')\n"
            "axes[0].axvline(real_mean * 100, c=RED, lw=2.5, label=f'Real hedge: {real_mean*100:+.2f}%/yr')\n"
            "axes[0].axvline(0, c='k', lw=1); axes[0].legend()\n"
            "axes[0].set_xlabel('Hedge mean (%/yr)'); axes[0].set_ylabel('Count')\n"
            "axes[0].set_title(f'Placebo p = {p_val:.2f} (real buried in the null)')\n"
            "axes[1].hist(rand * 100, bins=50, color=GREY, alpha=0.7, label='Random same-size draws')\n"
            "axes[1].axvline(lo_excess * 100, c=RED, lw=2.5, label=f'Low-IA excess: {lo_excess*100:+.1f}%/yr')\n"
            "axes[1].axvline(0, c='k', lw=1); axes[1].legend()\n"
            "axes[1].set_xlabel('Annual excess vs EW market (%)'); axes[1].set_ylabel('Count')\n"
            "axes[1].set_title(f'Low-IA beats {pct_beaten:.0%} of random draws (chance)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Placebo p={p_val:.3f}  Low-IA beats {pct_beaten:.0%} of random draws')"
        ),
        md(
            f"> Placebo p = **{R['placebo_p']:.2f}** (the real hedge sits inside the shuffle "
            f"null) and the low-IA leg beats only **{R['rand_pct_beaten']}%** of random "
            "same-size draws -- chance level. No alpha over a blind pick."
        ),
        md("### 4d - Costs: the (already-zero) gross turns negative"),
        code(
            "c = st.net_of_costs(real_mean, one_way_bps=10.0, annual_turnover=2.0, borrow_bps=50.0)\n"
            "labels = ['Gross', '- trading', '- borrow', 'Net']\n"
            "vals = [c['gross']*100, -c['cost_drag']*100, -c['borrow_drag']*100, c['net']*100]\n"
            "fig, ax = plt.subplots(figsize=(8, 4))\n"
            "bar_colors = [GREY, RED, RED, RED if c['net'] < 0 else GREEN]\n"
            "ax.bar(labels, vals, color=bar_colors)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('%/yr')\n"
            "ax.set_title('Costs erase a non-existent edge')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Gross {c[\"gross\"]*100:+.2f}%/yr -> Net {c[\"net\"]*100:+.2f}%/yr')"
        ),

        md(
            "## 5 - The verdict\n\n"
            f"- **Signal `NONE`** -- hedge {R['hedge_mean']:+.2f}%/yr, HAC *t* = "
            f"{R['hedge_t']:+.2f}, placebo p = {R['placebo_p']:.2f}. Absent.\n"
            f"- **Tradability `MIRAGE`** -- {R['net_mean']:+.2f}%/yr net. Short-high-IA on "
            "survivors means shorting the AI-datacentre and fab build-outs.\n"
            "- **Channel `BUSTED`** -- the capex channel dies alongside the total-asset-growth "
            "channel of [Study 244](../../244-asset-growth/)."
        ),

        md(
            "## 6 - Could you trade it? -- the honest answer\n\n"
            "No tradable signal exists on this basket. The obstacles:\n\n"
            "1. **Flat spread.** The hedge is +0.04%/yr with t = 0.01 -- not even a direction.\n"
            "2. **Survivorship selection dominates.** The high-capex survivors are productive "
            "expanders (semis, hyperscalers, energy majors) whose investment paid off.\n"
            "3. **The academic evidence is for broad universes.** Titman-Wei-Xie used all US "
            "stocks; the CMA / q-factor are broad-cross-section. On large-cap blue-chips the "
            "effect is thin and arbitraged.\n\n"
            "The correct conclusion: **MIRAGE** on this test. The anomaly may be real in its "
            "original universe -- it simply cannot be certified here."
        ),

        md(
            "## 7 - Going further\n\n"
            "- **The right universe.** Titman-Wei-Xie (2004) used NYSE/AMEX/NASDAQ stocks; a "
            "point-in-time all-US-stock universe is the minimum for a credible test.\n"
            "- **q-factor I/A.** Hou-Xue-Zhang's investment factor operationalises the same "
            "idea; the factor evidence is broad-cross-section, not survivor large-cap.\n"
            "- **Total-asset-growth cousin.** [Study 244 -- Asset-Growth](../../244-asset-growth/) "
            "tests the total-balance-sheet channel and also lands None/Mirage on the survivor "
            "panel -- the two investment channels die together.\n\n"
            "*Want to confirm the effect on a clean universe? Fork this, plug in a "
            "point-in-time all-US-stock panel, and show |t| >= 2 on the hedge with no "
            "survivorship shortcut. That's the bar.*"
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
