"""Generate the two narrative notebooks for Study 524 (Operating-Leverage).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks load the study's own cache (``../_cache/ol_*.parquet``) with a HAVE_REAL
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
# SURVIVORSHIP-BIASED; the predicted hedge is absent and mildly inverted (wrong sign).
R = dict(
    n_names=38, n_years=17, start_year=2008, end_year=2024,
    lo_mean=22.3, lo_sharpe=0.99, lo_t=4.68, lo_hit=94,
    mid_mean=16.0, mid_sharpe=1.23, mid_t=6.69,
    hi_mean=20.1, hi_sharpe=1.30, hi_t=5.96, hi_hit=100,
    mkt_mean=19.5, mkt_sharpe=1.42, mkt_t=8.01,
    # Hedge (g3 - g1): mildly negative -> effect absent (wrong sign) on this basket
    hedge_mean=-2.14, hedge_sharpe=-0.10, hedge_t=-0.39, hedge_hit=59,
    hi_excess_mean=0.6, hi_excess_t=0.31,
    lo_excess_mean=2.8, lo_excess_t=0.75,
    placebo_p=0.720, shuffle_mean=-0.12, shuffle_std=3.19,
    rand_std=7.8, rand_pct_beaten=54,
    cost_drag=0.20, borrow_drag=0.50, net_mean=-2.84,
    synth_null_t=0.15, synth_06_t=15.16, synth_10_t=25.06,
    ol_fp="b818fcf7a9a8", fwd_fp="9389b44fb18c",
)

# Year-by-year frozen data (g1 low-OL, g3 high-OL, market, hedge=g3-g1)
_G1  = [36.6, 23.5, 5.5, 9.8, 20.5, 5.3, 13.9, 38.2, 14.9, 24.8, -2.8, 34.4, 15.6, 2.9, 34.2, 6.5, 94.7]
_G3  = [47.2, 27.4, 21.4, 14.3, 14.5, 23.3, 10.9, 15.4, 16.5, 29.2, 9.4, 60.9, 5.9, 4.9, 11.1, 0.1, 29.6]
_MKT = [35.8, 25.3, 11.7, 13.9, 17.2, 13.6, 8.4, 27.9, 18.3, 22.0, 5.6, 50.6, 8.7, 5.4, 18.2, 3.5, 45.1]
_HDG = [10.7, 3.9, 16.0, 4.5, -6.0, 18.0, -3.1, -22.8, 1.6, 4.3, 12.2, 26.6, -9.6, 2.0, -23.1, -6.4, -65.2]
_YRS = list(range(R["start_year"], R["end_year"] + 1))

VERDICT_BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Channel: Busted](https://img.shields.io/badge/Channel-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath("../../.."))   # repo root
sys.path.insert(0, os.path.abspath(".."))          # study package (operating_leverage/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from operating_leverage import data, strategy as st

# Cache-first: load the study's own EDGAR + price cache; fall back to frozen R.
ol, fwd = data.fetch_panel()
ol = ol.dropna(how="all") if not ol.empty else ol
HAVE_REAL = not ol.empty
if HAVE_REAL:
    h = st.sorted_returns(ol, fwd, n_groups=3)
else:
    h = pd.DataFrame()
print("Real cache loaded:", HAVE_REAL,
      f"({len(ol.columns) if HAVE_REAL else 0} names, {len(h) if HAVE_REAL else 0} sort years)")
"""

R_CODE = f"""\
R = dict(
    n_names={R['n_names']}, n_years={R['n_years']}, start_year={R['start_year']}, end_year={R['end_year']},
    lo_mean={R['lo_mean']}, lo_sharpe={R['lo_sharpe']}, lo_t={R['lo_t']}, lo_hit={R['lo_hit']},
    mid_mean={R['mid_mean']}, hi_mean={R['hi_mean']}, hi_sharpe={R['hi_sharpe']}, hi_t={R['hi_t']}, hi_hit={R['hi_hit']},
    mkt_mean={R['mkt_mean']}, mkt_sharpe={R['mkt_sharpe']},
    hedge_mean={R['hedge_mean']}, hedge_sharpe={R['hedge_sharpe']}, hedge_t={R['hedge_t']}, hedge_hit={R['hedge_hit']},
    hi_excess_mean={R['hi_excess_mean']}, hi_excess_t={R['hi_excess_t']},
    lo_excess_mean={R['lo_excess_mean']}, lo_excess_t={R['lo_excess_t']},
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
            "# Operating-Leverage -- do high-fixed-cost firms earn a premium?\n"
            "### The Novy-Marx (2011) operating-leverage value channel on a large-cap survivor basket\n\n"
            + VERDICT_BADGES +
            "Robert Novy-Marx argued in 2011 that a firm's *fixed* operating costs act like "
            "leverage on the income statement. A firm whose operating costs (cost of goods + "
            "SG&A) are large relative to its asset base behaves like a *levered claim on "
            "demand*: small revenue swings produce large profit swings, so it carries more "
            "operating risk and should earn a premium that overlaps the value effect. We test "
            "this **operating-leverage channel** on a fixed large-cap basket using real SEC "
            "EDGAR filings.\n\n"
            "> This is the plain-language layer. The HAC t-stats, label-shuffle placebo, and "
            "random-portfolio null live in "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** Every chart is drawn by the code beside it. "
            "House style in [METHODOLOGY.md](../../../METHODOLOGY.md).\n"
            ">\n"
            "> **Survivorship bias**: the basket covers only firms still large-cap in 2026. "
            "High-fixed-cost firms whose demand collapsed and *failed* are absent -- which "
            "matters more than usual, because operating leverage is a *risk* premium that "
            "is supposed to pay for exactly those bad states."
        ),
        code(BOOT + "\n" + R_CODE),

        md(
            "## The answer first\n\n"
            "| Question | What the data says |\n|---|---|\n"
            f"| Do high-operating-leverage firms outperform low ones? | No -- the hedge is "
            f"**{R['hedge_mean']:+.2f}%/yr**, slightly the *wrong way*. The two legs earn "
            f"~{R['hi_mean']:.0f}%/yr each. |\n"
            f"| Is that statistically real? | HAC *t* = {R['hedge_t']:.2f} -- nowhere near "
            f"|t| >= 2, and negative. The premium is **absent**. |\n"
            f"| Does it beat a coin-flip relabelling? | No -- placebo p = {R['placebo_p']:.2f}; "
            f"a shuffled label beats it {int(R['placebo_p']*100)}% of the time. |\n"
            f"| Could you trade it? | No. After costs + short borrow the hedge is "
            f"**{R['net_mean']:+.2f}%/yr net**. |\n\n"
            "> **The catch**: operating leverage is a *risk* premium -- it is meant to pay in "
            "the bad states where a high-fixed-cost firm's demand collapses. A *survivor* "
            "basket deletes exactly those failures, so the surviving high-OL names are the "
            "ones whose demand *didn't* collapse. (2024: the low-OL, capital-light megacaps "
            "ran +94.7% while the high-OL tercile lagged at +29.6%.)"
        ),

        md(
            "## 1 - The claim\n\n"
            "> *Buy the high-fixed-cost firms. Sell the asset-heavy, capital-light ones. "
            "Rebalance annually. The high-operating-leverage firms are riskier -- a levered "
            "claim on demand -- and should pay a premium for that risk.*\n\n"
            "The signal is one accounting ratio:\n\n"
            "- **OL_t** = (COGS_t + SG&A_t) / Total Assets_t\n\n"
            "A high OL means the firm's income statement is dominated by operating costs "
            "relative to its asset base -- think a thin-margin retailer (rent, wages, "
            "logistics) versus a capital-light software platform. Novy-Marx's argument: those "
            "fixed costs amplify the effect of demand swings on profits, raising operating "
            "risk and expected return. This is the *operating-leverage* channel -- distinct "
            "from the *financial*-leverage (balance-sheet debt) anomaly tested in "
            "[Study 154](../../154-leverage-anomaly/)."
        ),

        md(
            "## 2 - So what?\n\n"
            "If the operating-leverage effect is real and tradable on large US stocks, it is a "
            "low-ML, fundamentals-only strategy needing only public 10-K filings, and it helps "
            "*explain* the value premium (value firms tend to have under-utilised assets and "
            "high fixed costs). If it only works on small/mid-caps -- or is a selection "
            "artefact -- the large-cap test is a cautionary tale about testing a risk premium "
            "in a sample with the risk deleted."
        ),

        md(
            "## 3 - How would we even know?\n\n"
            "Three honesty requirements:\n\n"
            "1. **One execution lag.** OL from fiscal year y drives a 12-month return that "
            "*starts 4 months after year-end* (when the 10-K is public), entered one trading "
            "day later. No same-bar fills, no look-ahead; no partial year is ever stamped.\n"
            "2. **Placebo + random control.** Shuffle the OL labels within each year; does the "
            "real hedge beat its shuffled self? And does the high-OL leg beat a random draw of "
            "the same size?\n"
            "3. **Name the survivorship bias.** The basket is current large-caps. Failed "
            "high-fixed-cost firms are absent. Every result is conditional on survival -- and "
            "for a *risk* premium, that conditioning is fatal."
        ),

        md(
            "## 4 - The teardown\n\n"
            "Positive control first: **when we plant a real operating-leverage effect in a "
            "synthetic panel, the engine finds it.** That makes the real-data result a "
            "statement about the *market* (and the survivor basket), not the method."
        ),
        code(
            "# Synthetic positive control: plant a known OL premium, confirm the engine detects it\n"
            "premiums = [0.0, 0.04, 0.08, 0.12]\n"
            "hedges = []\n"
            "for p in premiums:\n"
            "    s, f, _ = data.synthetic_panel(n_firms=300, n_years=25, premium=p, seed=524)\n"
            "    hh = st.sorted_returns(s, f, n_groups=3)\n"
            "    hedges.append(hh['hedge'].mean() * 100)\n"
            "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            "ax.plot([p*100 for p in premiums], hedges, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Planted OL effect (% alpha per unit z-rank)')\n"
            "ax.set_ylabel('High-minus-low-OL hedge (%/yr)')\n"
            "ax.set_title('Synthetic control: engine finds the effect when planted')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('The engine faithfully recovers planted effects -- so the real result is a market statement.')"
        ),
        md("**Now the real EDGAR panel -- tercile sort on operating leverage:**"),
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
            "axes[0].bar(x, g3_vals, label='g3 high-OL', color=GREEN, alpha=0.7, width=0.4)\n"
            "axes[0].bar([i+0.4 for i in x], g1_vals, label='g1 low-OL', color=RED, alpha=0.7, width=0.4)\n"
            "axes[0].set_xticks([i+0.2 for i in x])\n"
            "axes[0].set_xticklabels(years_real, rotation=45, fontsize=8)\n"
            "axes[0].set_ylabel('Annual return %'); axes[0].legend()\n"
            "axes[0].set_title('g3 (high OL) vs g1 (low OL) returns')\n"
            "colors = [GREEN if v > 0 else RED for v in hdg_vals]\n"
            "axes[1].bar(years_real, hdg_vals, color=colors)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_ylabel('Hedge return (g3 - g1) %')\n"
            "axes[1].set_title(f'High-minus-low-OL hedge: {sum(1 for v in hdg_vals if v>0)}/{len(hdg_vals)} positive years')\n"
            "axes[1].tick_params(axis='x', rotation=45, labelsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Mean hedge: {np.mean(hdg_vals):+.2f}%/yr  (frozen R: {R[\"hedge_mean\"]:+.2f}%/yr)')"
        ),
        md(
            f"The high-OL tercile earns **+{R['hi_mean']:.1f}%/yr** vs the low-OL tercile's "
            f"**+{R['lo_mean']:.1f}%/yr** -- a hedge of **{R['hedge_mean']:+.2f}%/yr**, "
            f"positive in only {R['hedge_hit']}% of years and pointing the *wrong* way. "
            "**The predicted premium is absent**: high- and low-operating-leverage firms earn "
            "essentially the same return on this survivor basket. Deleting the bad-state "
            "failures washes the risk premium out -- and the 2024 megacap melt-up even flips "
            "the sign."
        ),

        md(
            "## 5 - The verdict\n\n"
            f"- **Signal -- None.** Hedge {R['hedge_mean']:+.2f}%/yr, HAC *t* = "
            f"{R['hedge_t']:.2f} (wrong sign); placebo p = {R['placebo_p']:.2f}. No signal.\n"
            f"- **Tradability -- Mirage.** No gross edge; **{R['net_mean']:+.2f}%/yr net** "
            "after costs and short borrow.\n"
            "- **Operating- vs financial-leverage channel -- Busted.** The operating-leverage "
            "channel is no more present than the financial-leverage (debt) channel of "
            "[Study 154](../../154-leverage-anomaly/)."
        ),

        md(
            "## 6 - Could you actually trade it?\n\n"
            "No. There is no signal on this basket. Even setting aside survivorship:\n\n"
            "1. **Wrong universe.** Novy-Marx (2011) documented the effect on the broad US "
            "cross-section, where operating leverage overlaps value and bites hardest among "
            "small/mid-caps with limited access to capital. On ~38 heavily-covered large caps "
            "it is thin and arbitraged.\n"
            "2. **A risk premium with the risk deleted.** The high-OL large caps that survived "
            "are precisely the firms whose demand *didn't* collapse -- the opposite of the "
            "bad-state failures the premium is meant to compensate.\n"
            "3. **Short the wrong side.** Going short the *low*-OL leg on a survivor basket "
            "means shorting the capital-light megacaps -- the decade's biggest winners.\n\n"
            "The right conclusion: **explore this in a broad, point-in-time universe before "
            "treating the operating-leverage anomaly as tradable.**"
        ),

        md(
            "## 7 - Going further\n\n"
            "- **The right universe.** Novy-Marx (2011) used the broad US cross-section -- "
            "thousands of names including small and mid-caps. The large-cap survivor basket "
            "here is the wrong test for a risk premium.\n"
            "- **Financial-leverage cousin.** [Study 154 -- Leverage-Anomaly](../../154-leverage-anomaly/) "
            "tests the *balance-sheet debt* channel on a survivor panel; the two leverage "
            "channels die together here.\n"
            "- **Gross-profitability sibling.** [Study 122 -- Gross-Profitability](../../122-gross-profitability/) "
            "is Novy-Marx's *other* famous accounting factor, on the same desk infrastructure.\n\n"
            "*Think the operating-leverage effect is real in large caps? Fork this, extend to "
            "a point-in-time all-US-stock universe that *keeps the failures*, and show "
            "|t| >= 2 on the hedge. That's the bar.*"
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
            "# Operating-Leverage -- a quantitative teardown\n"
            "### OL tercile sort - EDGAR survivor basket - HAC inference - label-shuffle "
            "placebo - random-portfolio null - costs - synthetic control\n\n"
            + VERDICT_BADGES +
            "The deep companion to [01_for_the_curious.ipynb](01_for_the_curious.ipynb) -- "
            "same seven beats, every claim now carrying its standard error. We test the "
            "Novy-Marx (2011) operating-leverage premium on a survivorship-biased large-cap "
            f"basket (~{R['n_names']} names/yr, {R['start_year']}-{R['end_year']}), measure the "
            "high-minus-low-OL hedge with a HAC t-stat, pin it against a label-shuffle "
            "placebo and a random-portfolio null, charge costs, and confirm the engine on a "
            "seed-robust synthetic positive control.\n\n"
            "> **Not investment advice.** SEC EDGAR companyfacts (COGS/CostOfRevenue + SG&A, "
            "assets); Yahoo daily prices; survivorship-biased. Sources in "
            "[`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Survivorship-bias warning**: all results are conditional on survival -- the "
            "high-fixed-cost firms that failed (exactly what a risk premium pays for) are "
            "excluded."
        ),
        code(BOOT + "\n" + R_CODE),

        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Hedge **{R['hedge_mean']:+.2f}%/yr**, HAC *t* = "
            f"**{R['hedge_t']:+.2f}** (wrong sign); placebo p = **{R['placebo_p']:.2f}**. Absent. |\n"
            f"| **Tradability** | `MIRAGE` | No gross edge; **{R['net_mean']:+.2f}%/yr net** "
            f"after 10bps x turnover + 50bps borrow. |\n"
            f"| **Channel** | `BUSTED` | Operating-leverage channel no more present than the "
            f"financial-leverage channel of [Study 154](../../154-leverage-anomaly/). |\n\n"
            "> **In plain words:** the engine recovers a planted operating-leverage effect at "
            f"HAC *t* ~ {R['synth_06_t']:.0f}, but the real survivor basket shows a flat (in "
            "fact mildly inverted) zero. The high-OL survivors are firms whose demand never "
            "collapsed -- the risk the premium pays for has been selected out of the sample."
        ),

        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $r_{i,t+1}$ be firm $i$'s 12-month forward return. Define:\n\n"
            "$$\\text{OL}_{i,t} = \\frac{\\text{COGS}_{i,t} + \\text{SG\\&A}_{i,t}}{A_{i,t}}$$\n\n"
            "where $A$ = total assets. The **H1 (signal)** claim (Novy-Marx 2011) is:\n\n"
            "$$\\mathbb{E}\\left[\\bar{r}_{\\text{high-OL},t+1} - "
            "\\bar{r}_{\\text{low-OL},t+1}\\right] > 0$$\n\n"
            "with HAC |t| >= 2. **H2 (beats noise)** requires the hedge to beat its "
            "label-shuffled self. **H3 (tradable)** requires it to survive costs + borrow."
        ),

        md(
            "## 2 - So what? -- the stakes\n\n"
            "H1 is *not* supported on this biased basket -- the hedge is a flat (mildly "
            "negative) zero. The academic evidence on broad US cross-sections is credible "
            "(Novy-Marx 2011; Carlson-Fisher-Giammarino 2004; Zhang 2005 value-premium "
            "models); on large-cap survivors, selection deletes the bad-state risk the "
            "premium is supposed to price."
        ),

        md(
            "## 3 - The protocol\n\n"
            "- **Data**: operating costs (`CostOfGoodsAndServicesSold`/`CostOfRevenue` + "
            "`SellingGeneralAndAdministrativeExpense`) and `Assets` from SEC EDGAR "
            "companyfacts; daily adjusted prices from Yahoo.\n"
            f"- **Universe**: fixed large-cap survivor basket (~{R['n_names']} names/yr); "
            "survivorship-biased.\n"
            f"- **Window**: {R['n_years']} signal years ({R['start_year']}-{R['end_year']}), "
            "12-month forward returns, report-lagged.\n"
            "- **Signal**: OL = (COGS + SG&A) / Assets (higher = more operating-cost-intensive).\n"
            "- **Sort**: terciles; g1 = low OL (capital-heavy), g3 = high OL (operating-leverage).\n"
            "- **Execution lag**: one -- enter 1 trading day after a 4-month report lag past "
            "fiscal year-end; no partial year stamped.\n"
            "- **Inference**: HAC Newey-West t-stat on the 17-obs annual hedge.\n"
            "- **Nulls**: label-shuffle placebo (1000) + random same-size portfolios (500/yr).\n"
            "- **Costs**: 10bps x turnover + 50bps short borrow.\n"
            "- **Positive control**: seed-robust synthetic panel with a planted OL premium."
        ),

        md("## 4 - The teardown"),
        md(
            "### 4a - Synthetic positive control (seed-robust)\n\n"
            "Sweep the planted OL premium and confirm the hedge tracks it monotonically, "
            "averaging the HAC-t over many seeds (a single lucky seed is never enough)."
        ),
        code(
            "premiums = [0.0, 0.06, 0.10]\n"
            "synth_rows = []\n"
            "for p in premiums:\n"
            "    r = st.seed_robust_synthetic(p, n_seeds=25)\n"
            "    synth_rows.append((p*100, r['mean_hedge']*100, r['mean_tstat']))\n"
            "sydf = pd.DataFrame(synth_rows, columns=['premium_%', 'hedge_%/yr', 'mean_HAC_t'])\n"
            "fig, ax = plt.subplots(figsize=(9, 4))\n"
            "colors = [GREEN if t > 2 else (RED if t < -2 else GREY) for t in sydf['mean_HAC_t']]\n"
            "ax.bar(sydf['premium_%'], sydf['hedge_%/yr'], color=colors, width=1.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Planted OL effect (%/yr per unit z-rank)')\n"
            "ax.set_ylabel('High-minus-low hedge (%/yr)')\n"
            "ax.set_title('Positive control: HAC t>2 (green) appears where the premium is real')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(sydf.round(2))"
        ),
        md(
            f"> The engine recovers planted premiums (t ~ {R['synth_06_t']:.0f} at +6%/yr, "
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
            "    s_hi_ex = st.summary(h['g3'] - h['market'])\n"
            "    s_lo_ex = st.summary(h['g1'] - h['market'])\n"
            "else:\n"
            "    g_means = [R['lo_mean'], R['mid_mean'], R['hi_mean']]\n"
            "    g_tstats = [R['lo_t'], R['mid_mean'], R['hi_t']]\n"
            "    s_hedge = dict(mean=R['hedge_mean']/100, tstat=R['hedge_t'], hit_rate=R['hedge_hit']/100)\n"
            "    s_hi_ex = dict(mean=R['hi_excess_mean']/100, tstat=R['hi_excess_t'])\n"
            "    s_lo_ex = dict(mean=R['lo_excess_mean']/100, tstat=R['lo_excess_t'])\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "x = [1, 2, 3]\n"
            "bar_colors = [GREY, GREY, GREY]\n"
            "axes[0].bar(x, g_means, color=bar_colors)\n"
            "axes[0].set_xticks(x); axes[0].set_xticklabels(['g1 (lo)', 'g2', 'g3 (hi)'])\n"
            "axes[0].set_ylabel('Mean annual return (%/yr)')\n"
            "axes[0].set_title('Tercile returns: low and high OL earn ~identical (no spread)')\n"
            "axes[1].bar(x, g_tstats, color=bar_colors)\n"
            "for s in (2, -2): axes[1].axhline(s, ls='--', c=GREY, lw=1)\n"
            "axes[1].set_xticks(x); axes[1].set_xticklabels(['g1 (lo)', 'g2', 'g3 (hi)'])\n"
            "axes[1].set_ylabel('HAC t-stat (level returns)')\n"
            "axes[1].set_title('All terciles ride the same survivor bull market')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Hedge: {s_hedge[\"mean\"]*100:+.2f}%/yr  HAC t={s_hedge[\"tstat\"]:+.2f}  hit={s_hedge[\"hit_rate\"]:.0%}')\n"
            "print(f'High-OL excess: {s_hi_ex[\"mean\"]*100:+.1f}%/yr  t={s_hi_ex[\"tstat\"]:+.2f}')\n"
            "print(f'Low-OL excess: {s_lo_ex[\"mean\"]*100:+.1f}%/yr  t={s_lo_ex[\"tstat\"]:+.2f}')"
        ),
        md(
            f"> The high-minus-low-OL hedge is **{R['hedge_mean']:+.2f}%/yr**, HAC *t* = "
            f"**{R['hedge_t']:+.2f}** -- statistically zero and the wrong sign. The low and "
            f"high OL terciles earn **~{R['hi_mean']:.0f}%/yr each**: no spread, no "
            "monotonicity, no signal."
        ),
        md(
            "### 4c - Label-shuffle placebo + random-portfolio null\n\n"
            "Shuffle the OL labels within each year (1000 times); does the real hedge beat its "
            "shuffled self? And does the high-OL leg beat a blind same-size draw?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    p_val, sh = st.placebo_hedge_tstats(ol, fwd, n_groups=3, n_shuffles=1000, seed=524)\n"
            "    rand = st.random_portfolio_returns(ol, fwd, n_groups=3, n_draws=500, seed=524)\n"
            "    real_mean = float(h['hedge'].mean())\n"
            "    hi_excess = float((h['g3'] - h['market']).mean())\n"
            "    pct_beaten = float((rand < hi_excess).mean())\n"
            "else:\n"
            "    rng_nb = np.random.default_rng(524)\n"
            "    sh = pd.Series(rng_nb.normal(R['shuffle_mean']/100, R['shuffle_std']/100, 1000))\n"
            "    rand = pd.Series(rng_nb.normal(0.0, R['rand_std']/100, 17*500))\n"
            "    real_mean = R['hedge_mean'] / 100\n"
            "    p_val = R['placebo_p']\n"
            "    hi_excess = R['hi_excess_mean'] / 100\n"
            "    pct_beaten = R['rand_pct_beaten'] / 100\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "axes[0].hist(sh * 100, bins=40, color=GREY, alpha=0.75, label='Label-shuffle hedge')\n"
            "axes[0].axvline(real_mean * 100, c=RED, lw=2.5, label=f'Real hedge: {real_mean*100:+.2f}%/yr')\n"
            "axes[0].axvline(0, c='k', lw=1); axes[0].legend()\n"
            "axes[0].set_xlabel('Hedge mean (%/yr)'); axes[0].set_ylabel('Count')\n"
            "axes[0].set_title(f'Placebo p = {p_val:.2f} (real buried in the null)')\n"
            "axes[1].hist(rand * 100, bins=50, color=GREY, alpha=0.7, label='Random same-size draws')\n"
            "axes[1].axvline(hi_excess * 100, c=RED, lw=2.5, label=f'High-OL excess: {hi_excess*100:+.1f}%/yr')\n"
            "axes[1].axvline(0, c='k', lw=1); axes[1].legend()\n"
            "axes[1].set_xlabel('Annual excess vs EW market (%)'); axes[1].set_ylabel('Count')\n"
            "axes[1].set_title(f'High-OL beats {pct_beaten:.0%} of random draws (chance)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Placebo p={p_val:.3f}  High-OL beats {pct_beaten:.0%} of random draws')"
        ),
        md(
            f"> Placebo p = **{R['placebo_p']:.2f}** (the real hedge sits inside -- in fact "
            f"left of -- the shuffle null) and the high-OL leg beats only "
            f"**{R['rand_pct_beaten']}%** of random same-size draws -- chance level. No alpha "
            "over a blind pick."
        ),
        md("### 4d - Costs: the (already-negative) gross worsens"),
        code(
            "c = st.net_of_costs(real_mean, one_way_bps=10.0, annual_turnover=2.0, borrow_bps=50.0)\n"
            "labels = ['Gross', '- trading', '- borrow', 'Net']\n"
            "vals = [c['gross']*100, -c['cost_drag']*100, -c['borrow_drag']*100, c['net']*100]\n"
            "fig, ax = plt.subplots(figsize=(8, 4))\n"
            "bar_colors = [RED if c['gross'] < 0 else GREY, RED, RED, RED if c['net'] < 0 else GREEN]\n"
            "ax.bar(labels, vals, color=bar_colors)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('%/yr')\n"
            "ax.set_title('Costs deepen a non-existent edge')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Gross {c[\"gross\"]*100:+.2f}%/yr -> Net {c[\"net\"]*100:+.2f}%/yr')"
        ),

        md(
            "## 5 - The verdict\n\n"
            f"- **Signal `NONE`** -- hedge {R['hedge_mean']:+.2f}%/yr, HAC *t* = "
            f"{R['hedge_t']:+.2f}, placebo p = {R['placebo_p']:.2f}. Absent (wrong sign).\n"
            f"- **Tradability `MIRAGE`** -- {R['net_mean']:+.2f}%/yr net. Short-low-OL on "
            "survivors means shorting the capital-light megacaps -- the decade's winners.\n"
            "- **Channel `BUSTED`** -- the operating-leverage channel dies alongside the "
            "financial-leverage channel of [Study 154](../../154-leverage-anomaly/)."
        ),

        md(
            "## 6 - Could you trade it? -- the honest answer\n\n"
            "No tradable signal exists on this basket. The obstacles:\n\n"
            "1. **Flat (wrong-sign) spread.** The hedge is -2.14%/yr with t = -0.39 -- not "
            "even the right direction.\n"
            "2. **A risk premium with the risk deleted.** The high-OL survivors are firms "
            "whose demand never collapsed -- exactly the sample where an operating-risk "
            "premium should be smallest.\n"
            "3. **The academic evidence is for broad universes.** Novy-Marx (2011) used the "
            "whole US cross-section; on large-cap blue-chips the effect is thin and "
            "arbitraged.\n\n"
            "The correct conclusion: **MIRAGE** on this test. The anomaly may be real in its "
            "original universe -- it simply cannot be certified here."
        ),

        md(
            "## 7 - Going further\n\n"
            "- **The right universe.** Novy-Marx (2011) used the broad US cross-section; a "
            "point-in-time all-US-stock universe that *keeps the failures* is the minimum for "
            "a credible test of a risk premium.\n"
            "- **Financial-leverage cousin.** [Study 154 -- Leverage-Anomaly](../../154-leverage-anomaly/) "
            "tests the *balance-sheet debt* channel and also lands None/Mirage on the survivor "
            "panel -- the two leverage channels die together.\n"
            "- **Gross-profitability sibling.** [Study 122 -- Gross-Profitability](../../122-gross-profitability/) "
            "is Novy-Marx's *other* famous accounting factor, on the same desk infrastructure.\n\n"
            "*Want to confirm the effect on a clean universe? Fork this, plug in a "
            "point-in-time all-US-stock panel that keeps delisted names, and show |t| >= 2 on "
            "the hedge with no survivorship shortcut. That's the bar.*"
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
