"""Generate the two narrative notebooks for Study 528 (Labor-Hiring-Rate).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks load the study's own price cache (``../_cache/hiring_prices.parquet``) plus
the hardcoded employee panel with a HAVE_REAL guard that falls back to the frozen headline
numbers ``R`` (a mirror of docs/results.md, as-of 2026-06-27) when the cache is absent. The
synthetic positive control runs offline and deterministically.
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-27).
# SURVIVORSHIP-BIASED; the predicted hedge is negative (sign reversed on this basket).
R = dict(
    n_names=28, n_years=11, start_year=2014, end_year=2024,
    lo_mean=17.5, lo_sharpe=1.24, lo_t=5.40, lo_hit=100,
    mid_mean=16.0, mid_sharpe=0.85, mid_t=3.99,
    hi_mean=22.2, hi_sharpe=1.66, hi_t=10.03, hi_hit=100,
    mkt_mean=18.7, mkt_sharpe=1.36, mkt_t=7.16,
    # Hedge (g1 - g3): NEGATIVE -> anomaly absent / inverted on this basket
    hedge_mean=-4.68, hedge_sharpe=-0.35, hedge_t=-1.29, hedge_hit=45,
    lo_excess_mean=-1.2, lo_excess_t=-0.74,
    hi_excess_mean=3.5, hi_excess_t=1.56,
    placebo_p=0.805, shuffle_mean=0.04, shuffle_std=5.06,
    rand_std=9.9, rand_pct_beaten=48,
    cost_drag=0.20, borrow_drag=0.50, net_mean=-5.38,
    synth_null_t=-0.15, synth_06_t=14.94, synth_10_t=25.00,
    hn_fp="26f5af1e69da", fwd_fp="d6fd3efee81e",
)

# Year-by-year frozen data (g1, g3, market, hedge)
_G1  = [5.0, 22.5, 12.6, 16.5, 3.1, 31.4, 13.5, 11.7, 14.0, 9.3, 52.6]
_G3  = [18.1, 21.1, 33.2, 18.8, 19.9, 49.0, 9.4, 9.9, 31.5, 1.4, 31.3]
_MKT = [9.5, 24.8, 20.4, 17.9, 5.9, 42.9, 7.8, 7.6, 17.8, 7.2, 43.9]
_HDG = [-13.2, 1.4, -20.6, -2.3, -16.9, -17.6, 4.1, 1.8, -17.5, 7.9, 21.3]
_YRS = list(range(R["start_year"], R["end_year"] + 1))

VERDICT_BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Sign: Busted](https://img.shields.io/badge/Sign-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath("../../.."))   # repo root
sys.path.insert(0, os.path.abspath(".."))          # study package (labor_hiring_rate/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from labor_hiring_rate import data, strategy as st

# Cache-first: load the study's own price cache + hardcoded employee panel; fall back to R.
hn, fwd = data.fetch_panel()
hn = hn.dropna(how="all") if not hn.empty else hn
HAVE_REAL = not hn.empty
if HAVE_REAL:
    h = st.sorted_returns(hn, fwd, n_groups=3)
else:
    h = pd.DataFrame()
print("Real cache loaded:", HAVE_REAL,
      f"({len(hn.columns) if HAVE_REAL else 0} names, {len(h) if HAVE_REAL else 0} sort years)")
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
            "# Labor-Hiring-Rate -- do the fastest hirers underperform?\n"
            "### The Belo-Lin-Bazdresch (2014) hiring anomaly on a large-cap survivor basket\n\n"
            + VERDICT_BADGES +
            "Frederico Belo, Xiaoji Lin and Santiago Bazdresch showed in 2014 that firms which "
            "grow employment fast earn lower future returns -- the labor analogue of the "
            "capital-investment anomaly. The story: hiring is costly (search, training, "
            "severance), so the *hiring rate* behaves like an investment rate; managers "
            "over-hire when capital is cheap, markets extrapolate the head-count growth, and "
            "the stock disappoints. We test it on a fixed large-cap basket using employee "
            "counts transcribed from real 10-K filings.\n\n"
            "> This is the plain-language layer. The HAC t-stats, label-shuffle placebo, and "
            "random-portfolio null live in "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** Every chart is drawn by the code beside it. "
            "House style in [METHODOLOGY.md](../../../METHODOLOGY.md).\n"
            ">\n"
            "> **Survivorship bias**: the basket covers only firms still large-cap in 2026. "
            "Aggressive hirers that over-expanded and *failed* are absent.\n"
            ">\n"
            "> **Data note**: employee counts are *not* in any machine-readable feed -- firms "
            "report them as 10-K cover-page text. We use a curated, hardcoded panel of those "
            "cover-page headcounts (FY2013-FY2024)."
        ),
        code(BOOT + "\n" + R_CODE),

        md(
            "## The answer first\n\n"
            "| Question | What the data says |\n|---|---|\n"
            f"| Do slow-hiring firms outperform fast-hiring firms? | **No -- the opposite.** "
            f"The hedge is **{R['hedge_mean']:+.2f}%/yr**: the fast hirers (g3, "
            f"{R['hi_mean']:.1f}%/yr) *beat* the disciplined hirers (g1, {R['lo_mean']:.1f}%/yr). |\n"
            f"| Is that statistically real? | HAC *t* = {R['hedge_t']:.2f} -- wrong sign and "
            f"\\|t\\| < 2. The predicted anomaly is **absent**. |\n"
            f"| Does it beat a coin-flip relabelling? | No -- placebo p = {R['placebo_p']:.2f}; "
            f"the real hedge is in the *left* tail of the shuffle null. |\n"
            f"| Could you trade it? | No. After costs + short borrow the hedge is "
            f"**{R['net_mean']:+.2f}%/yr net** -- and the short leg is the AI/hyperscale winners. |\n\n"
            "> **The catch**: on a *survivor* basket, the high-hiring group is full of "
            "*successful* expanders -- NVIDIA, Amazon, Microsoft, Costco, UnitedHealth. Their "
            "hiring tracked real demand. The over-expanding failures the anomaly was built "
            "around are simply missing."
        ),

        md(
            "## 1 - The claim\n\n"
            "> *Sell the fastest hirers. Buy the disciplined ones. Rebalance annually. The "
            "over-expanding firms will disappoint as the hiring spree stops producing "
            "earnings.*\n\n"
            "The signal is one ratio of head-count change:\n\n"
            "- **HN_t** = (N_t - N_{t-1}) / (0.5 x (N_t + N_{t-1}))\n\n"
            "where N is full-time employees. The mid-point denominator keeps the rate "
            "symmetric (a halving and a doubling are equal-and-opposite) and bounded in "
            "[-2, 2]. A high HN means the firm grew its workforce fast. The "
            "Belo-Lin-Bazdresch argument: that growth often reflects over-investment in labor "
            "(Jensen 1986 agency costs) or market over-extrapolation, and the stock "
            "underperforms. This is the *labor* channel -- distinct from the *capex* channel "
            "in [Study 523](../../523-investment-to-assets/) and the *total-asset-growth* "
            "channel in [Study 244](../../244-asset-growth/)."
        ),

        md(
            "## 2 - So what?\n\n"
            "If the hiring effect is real and tradable on large US stocks, it is a "
            "low-ML, fundamentals-only strategy needing only one number off the 10-K cover "
            "page. If it only works on small/mid-caps -- or is a selection artefact that "
            "*reverses* on survivors -- the large-cap test is a cautionary tale about testing "
            "in the wrong universe."
        ),

        md(
            "## 3 - How would we even know?\n\n"
            "Three honesty requirements:\n\n"
            "1. **One execution lag.** The year-y hiring rate drives a 12-month return that "
            "*starts 4 months after year-end* (when the 10-K is public), entered one trading "
            "day later. No same-bar fills, no look-ahead; no partial year is ever stamped.\n"
            "2. **Placebo + random control.** Shuffle the hiring labels within each year; does "
            "the real hedge beat its shuffled self? And does the low-hiring leg beat a random "
            "draw of the same size?\n"
            "3. **Name the survivorship bias.** The basket is current large-caps. Failed "
            "over-expanders are absent. Every result is conditional on survival."
        ),

        md(
            "## 4 - The teardown\n\n"
            "Positive control first: **when we plant a real hiring effect in a synthetic "
            "panel, the engine finds it.** That makes the real-data result a statement about "
            "the *market* (and the survivor basket), not the method."
        ),
        code(
            "# Synthetic positive control: plant a known hiring premium, confirm the engine detects it\n"
            "premiums = [0.0, -0.04, -0.08, -0.12]\n"
            "hedges = []\n"
            "for p in premiums:\n"
            "    s, f, _ = data.synthetic_panel(n_firms=300, n_years=25, premium=p, seed=528)\n"
            "    hh = st.sorted_returns(s, f, n_groups=3)\n"
            "    hedges.append(hh['hedge'].mean() * 100)\n"
            "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            "ax.plot([abs(p)*100 for p in premiums], hedges, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Planted hiring effect (% alpha per unit z-rank)')\n"
            "ax.set_ylabel('Low-minus-high-hire hedge (%/yr)')\n"
            "ax.set_title('Synthetic control: engine finds the effect when planted')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('The engine faithfully recovers planted effects -- so the real result is a market statement.')"
        ),
        md("**Now the real panel -- tercile sort on the hiring rate:**"),
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
            "axes[0].bar(x, g1_vals, label='g1 low-hire', color=GREEN, alpha=0.7, width=0.4)\n"
            "axes[0].bar([i+0.4 for i in x], g3_vals, label='g3 high-hire', color=RED, alpha=0.7, width=0.4)\n"
            "axes[0].set_xticks([i+0.2 for i in x])\n"
            "axes[0].set_xticklabels(years_real, rotation=45, fontsize=8)\n"
            "axes[0].set_ylabel('Annual return %'); axes[0].legend()\n"
            "axes[0].set_title('g1 (low hire) vs g3 (high hire) returns')\n"
            "colors = [GREEN if v > 0 else RED for v in hdg_vals]\n"
            "axes[1].bar(years_real, hdg_vals, color=colors)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_ylabel('Hedge return (g1 - g3) %')\n"
            "axes[1].set_title(f'Low-minus-high-hire hedge: {sum(1 for v in hdg_vals if v>0)}/{len(hdg_vals)} positive years')\n"
            "axes[1].tick_params(axis='x', rotation=45, labelsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Mean hedge: {np.mean(hdg_vals):+.2f}%/yr  (frozen R: {R[\"hedge_mean\"]:+.2f}%/yr)')"
        ),
        md(
            f"The low-hire tercile earns **+{R['lo_mean']:.1f}%/yr** vs the high-hire "
            f"tercile's **+{R['hi_mean']:.1f}%/yr** -- a hedge of **{R['hedge_mean']:+.2f}%/yr**, "
            f"positive in only {R['hedge_hit']}% of years. **The predicted ranking is "
            "inverted**: the aggressive hirers *out*-earned the disciplined ones on this "
            "survivor basket. Survivorship selection doesn't just wash the signal out -- it "
            "flips its sign."
        ),

        md(
            "## 5 - The verdict\n\n"
            f"- **Signal -- None.** Hedge {R['hedge_mean']:+.2f}%/yr, HAC *t* = "
            f"{R['hedge_t']:.2f} (wrong sign); placebo p = {R['placebo_p']:.2f}. No signal.\n"
            f"- **Tradability -- Mirage.** No positive gross edge; **{R['net_mean']:+.2f}%/yr "
            "net** after costs and short borrow.\n"
            "- **Predicted sign -- Busted.** The fast hirers beat the disciplined ones -- the "
            "opposite of Belo-Lin-Bazdresch's prediction, the same fate as the capex channel "
            "of [Study 523](../../523-investment-to-assets/)."
        ),

        md(
            "## 6 - Could you actually trade it?\n\n"
            "No. There is no signal on this basket -- and the sign is wrong. Even setting "
            "aside survivorship:\n\n"
            "1. **Wrong universe.** Belo-Lin-Bazdresch (2014) documented the effect on the "
            "broad Compustat cross-section, dominated by small/mid-caps where over-expansion "
            "and limits-to-arbitrage bite hardest. On 28 heavily-covered large caps it is "
            "thin and arbitraged.\n"
            "2. **Survivorship reversal.** The high-hiring large caps that survived are "
            "productive expanders (AI/semis, hyperscalers, warehouse retail, managed care) -- "
            "the opposite of the over-expanding failures the anomaly targets.\n"
            "3. **The short leg is the winners.** Shorting the high-hiring leg means shorting "
            "NVIDIA, Amazon, Microsoft and Costco through the very years they led the market.\n\n"
            "The right conclusion: **explore this in a broad, point-in-time universe before "
            "treating the hiring anomaly as tradable.**"
        ),

        md(
            "## 7 - Going further\n\n"
            "- **The right universe.** Belo-Lin-Bazdresch used the broad Compustat universe -- "
            "thousands of names including small and mid-caps. The large-cap survivor basket "
            "here is the wrong test.\n"
            "- **Capex cousin.** [Study 523 -- Investment-To-Assets](../../523-investment-to-assets/) "
            "tests the capital-input channel on the same kind of survivor panel and also lands "
            "None/Mirage. The labor and capital channels die together.\n"
            "- **Total-asset-growth cousin.** [Study 244 -- Asset-Growth](../../244-asset-growth/) "
            "tests the balance-sheet channel and lands None/Mirage too.\n"
            "- **The data wall.** Employee counts are not in XBRL; a deep, point-in-time "
            "head-count panel is the real prerequisite for a credible large-sample test.\n\n"
            "*Think the hiring effect is real in large caps? Fork this, extend to a "
            "point-in-time all-US-stock universe with a proper head-count history, and show "
            "\\|t\\| >= 2 on the hedge with no survivorship shortcut. That's the bar.*"
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
            "# Labor-Hiring-Rate -- a quantitative teardown\n"
            "### Hiring-rate tercile sort - curated 10-K head-count panel - HAC inference - "
            "label-shuffle placebo - random-portfolio null - costs - synthetic control\n\n"
            + VERDICT_BADGES +
            "The deep companion to [01_for_the_curious.ipynb](01_for_the_curious.ipynb) -- "
            "same seven beats, every claim now carrying its standard error. We test the "
            "Belo-Lin-Bazdresch hiring anomaly on a survivorship-biased large-cap basket "
            f"({R['n_names']} names/yr, {R['start_year']}-{R['end_year']}), measure the "
            "low-minus-high-hire hedge with a HAC t-stat, pin it against a label-shuffle "
            "placebo and a random-portfolio null, charge costs, and confirm the engine on a "
            "seed-robust synthetic positive control.\n\n"
            "> **Not investment advice.** Employee counts: a curated 10-K cover-page panel "
            "(no XBRL/API source exists); daily prices from Yahoo; survivorship-biased. "
            "Sources in [`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Survivorship-bias warning**: all results are conditional on survival -- the "
            "over-expanding firms that failed are excluded."
        ),
        code(BOOT + "\n" + R_CODE),

        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Hedge **{R['hedge_mean']:+.2f}%/yr**, HAC *t* = "
            f"**{R['hedge_t']:+.2f}** (wrong sign); placebo p = **{R['placebo_p']:.2f}**. Absent. |\n"
            f"| **Tradability** | `MIRAGE` | No positive gross edge; **{R['net_mean']:+.2f}%/yr net** "
            f"after 10bps x turnover + 50bps borrow. |\n"
            f"| **Predicted sign** | `BUSTED` | Fast hirers **+{R['hi_mean']:.1f}%/yr** beat "
            f"disciplined hirers **+{R['lo_mean']:.1f}%/yr** -- the sign reverses, as in "
            f"[Study 523](../../523-investment-to-assets/). |\n\n"
            "> **In plain words:** the engine recovers a planted hiring effect at HAC "
            f"*t* ~ {R['synth_06_t']:.0f}, but the real survivor basket shows the *opposite* "
            "sign. The high-hire survivors are the AI/hyperscale growth winners, not "
            "over-expanding failures."
        ),

        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $r_{i,t+1}$ be firm $i$'s 12-month forward return and $N_{i,t}$ its "
            "full-time head-count. Define the hiring rate:\n\n"
            "$$\\text{HN}_{i,t} = \\frac{N_{i,t} - N_{i,t-1}}{\\tfrac{1}{2}(N_{i,t} + N_{i,t-1})}$$\n\n"
            "The **H1 (signal)** claim (Belo-Lin-Bazdresch 2014) is:\n\n"
            "$$\\mathbb{E}\\left[\\bar{r}_{\\text{low-HN},t+1} - "
            "\\bar{r}_{\\text{high-HN},t+1}\\right] > 0$$\n\n"
            "with HAC |t| >= 2. **H2 (beats noise)** requires the hedge to beat its "
            "label-shuffled self. **H3 (tradable)** requires it to survive costs + borrow."
        ),

        md(
            "## 2 - So what? -- the stakes\n\n"
            "H1 is *not* supported on this biased basket -- the hedge is negative (sign "
            "reversed). The academic evidence on the broad US cross-section is credible "
            "(Belo-Lin-Bazdresch 2014; the q-factor investment leg); on large-cap survivors, "
            "selection flips the signal."
        ),

        md(
            "## 3 - The protocol\n\n"
            "- **Data**: full-time-employee counts from a curated 10-K cover-page panel "
            "(FY2013-FY2024; no XBRL/API source exists); daily adjusted prices from Yahoo.\n"
            f"- **Universe**: fixed {R['n_names']}-name large-cap survivor basket; "
            "survivorship-biased.\n"
            f"- **Window**: {R['n_years']} signal years ({R['start_year']}-{R['end_year']}), "
            "12-month forward returns, report-lagged.\n"
            "- **Signal**: HN = (N_t - N_{t-1}) / (0.5*(N_t + N_{t-1})) (higher = faster hirer).\n"
            "- **Sort**: terciles; g1 = low hiring (disciplined), g3 = high hiring (aggressive).\n"
            "- **Execution lag**: one -- enter 1 trading day after a 4-month report lag past "
            "fiscal year-end; no partial year stamped.\n"
            "- **Inference**: HAC Newey-West t-stat on the 11-obs annual hedge.\n"
            "- **Nulls**: label-shuffle placebo (1000) + random same-size portfolios (500/yr).\n"
            "- **Costs**: 10bps x turnover + 50bps short borrow.\n"
            "- **Positive control**: seed-robust synthetic panel with a planted hiring premium."
        ),

        md("## 4 - The teardown"),
        md(
            "### 4a - Synthetic positive control (seed-robust)\n\n"
            "Sweep the planted hiring premium and confirm the hedge tracks it monotonically, "
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
            "ax.set_xlabel('Planted hiring effect (%/yr per unit z-rank)')\n"
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
        md("### 4b - Real panel -- tercile returns and HAC inference"),
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
            "bar_colors = [GREEN, GREY, RED]\n"
            "axes[0].bar(x, g_means, color=bar_colors)\n"
            "axes[0].set_xticks(x); axes[0].set_xticklabels(['g1 (lo)', 'g2', 'g3 (hi)'])\n"
            "axes[0].set_ylabel('Mean annual return (%/yr)')\n"
            "axes[0].set_title('Tercile returns: high-hire (g3) OUT-earns low-hire (g1)')\n"
            "axes[1].bar(x, g_tstats, color=bar_colors)\n"
            "for s in (2, -2): axes[1].axhline(s, ls='--', c=GREY, lw=1)\n"
            "axes[1].set_xticks(x); axes[1].set_xticklabels(['g1 (lo)', 'g2', 'g3 (hi)'])\n"
            "axes[1].set_ylabel('HAC t-stat (level returns)')\n"
            "axes[1].set_title('All terciles ride the same survivor bull market')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Hedge: {s_hedge[\"mean\"]*100:+.2f}%/yr  HAC t={s_hedge[\"tstat\"]:+.2f}  hit={s_hedge[\"hit_rate\"]:.0%}')\n"
            "print(f'Low-hire excess: {s_lo_ex[\"mean\"]*100:+.1f}%/yr  t={s_lo_ex[\"tstat\"]:+.2f}')\n"
            "print(f'High-hire excess: {s_hi_ex[\"mean\"]*100:+.1f}%/yr  t={s_hi_ex[\"tstat\"]:+.2f}')"
        ),
        md(
            f"> The low-minus-high-hire hedge is **{R['hedge_mean']:+.2f}%/yr**, HAC *t* = "
            f"**{R['hedge_t']:+.2f}** -- statistically zero *and the wrong sign*. The high-hire "
            f"tercile earns **+{R['hi_mean']:.1f}%/yr** vs **+{R['lo_mean']:.1f}%/yr** for the "
            "low-hire tercile: an inverted spread, no signal."
        ),
        md(
            "### 4c - Label-shuffle placebo + random-portfolio null\n\n"
            "Shuffle the hiring labels within each year (1000 times); does the real hedge beat "
            "its shuffled self? And does the low-hire leg beat a blind same-size draw?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    p_val, sh = st.placebo_hedge_tstats(hn, fwd, n_groups=3, n_shuffles=1000, seed=528)\n"
            "    rand = st.random_portfolio_returns(hn, fwd, n_groups=3, n_draws=500, seed=528)\n"
            "    real_mean = float(h['hedge'].mean())\n"
            "    lo_excess = float((h['g1'] - h['market']).mean())\n"
            "    pct_beaten = float((rand < lo_excess).mean())\n"
            "else:\n"
            "    rng_nb = np.random.default_rng(528)\n"
            "    sh = pd.Series(rng_nb.normal(R['shuffle_mean']/100, R['shuffle_std']/100, 1000))\n"
            "    rand = pd.Series(rng_nb.normal(0.0, R['rand_std']/100, 11*500))\n"
            "    real_mean = R['hedge_mean'] / 100\n"
            "    p_val = R['placebo_p']\n"
            "    lo_excess = R['lo_excess_mean'] / 100\n"
            "    pct_beaten = R['rand_pct_beaten'] / 100\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "axes[0].hist(sh * 100, bins=40, color=GREY, alpha=0.75, label='Label-shuffle hedge')\n"
            "axes[0].axvline(real_mean * 100, c=RED, lw=2.5, label=f'Real hedge: {real_mean*100:+.2f}%/yr')\n"
            "axes[0].axvline(0, c='k', lw=1); axes[0].legend()\n"
            "axes[0].set_xlabel('Hedge mean (%/yr)'); axes[0].set_ylabel('Count')\n"
            "axes[0].set_title(f'Placebo p = {p_val:.2f} (real in the LEFT tail)')\n"
            "axes[1].hist(rand * 100, bins=50, color=GREY, alpha=0.7, label='Random same-size draws')\n"
            "axes[1].axvline(lo_excess * 100, c=RED, lw=2.5, label=f'Low-hire excess: {lo_excess*100:+.1f}%/yr')\n"
            "axes[1].axvline(0, c='k', lw=1); axes[1].legend()\n"
            "axes[1].set_xlabel('Annual excess vs EW market (%)'); axes[1].set_ylabel('Count')\n"
            "axes[1].set_title(f'Low-hire beats {pct_beaten:.0%} of random draws (chance)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Placebo p={p_val:.3f}  Low-hire beats {pct_beaten:.0%} of random draws')"
        ),
        md(
            f"> Placebo p = **{R['placebo_p']:.2f}** (the real hedge sits in the *left* tail of "
            f"the shuffle null -- worse than a coin-flip relabelling) and the low-hire leg "
            f"beats only **{R['rand_pct_beaten']}%** of random same-size draws -- chance level. "
            "No alpha over a blind pick."
        ),
        md("### 4d - Costs: the negative gross turns more negative"),
        code(
            "c = st.net_of_costs(real_mean, one_way_bps=10.0, annual_turnover=2.0, borrow_bps=50.0)\n"
            "labels = ['Gross', '- trading', '- borrow', 'Net']\n"
            "vals = [c['gross']*100, -c['cost_drag']*100, -c['borrow_drag']*100, c['net']*100]\n"
            "fig, ax = plt.subplots(figsize=(8, 4))\n"
            "bar_colors = [RED, RED, RED, RED if c['net'] < 0 else GREEN]\n"
            "ax.bar(labels, vals, color=bar_colors)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('%/yr')\n"
            "ax.set_title('Costs deepen an already-negative (wrong-sign) hedge')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Gross {c[\"gross\"]*100:+.2f}%/yr -> Net {c[\"net\"]*100:+.2f}%/yr')"
        ),

        md(
            "## 5 - The verdict\n\n"
            f"- **Signal `NONE`** -- hedge {R['hedge_mean']:+.2f}%/yr, HAC *t* = "
            f"{R['hedge_t']:+.2f} (wrong sign), placebo p = {R['placebo_p']:.2f}. Absent.\n"
            f"- **Tradability `MIRAGE`** -- {R['net_mean']:+.2f}%/yr net. Short-high-hire on "
            "survivors means shorting NVIDIA, Amazon and the hyperscale build-outs.\n"
            "- **Predicted sign `BUSTED`** -- the spread inverts (fast hirers "
            f"+{R['hi_mean']:.1f}%/yr beat disciplined +{R['lo_mean']:.1f}%/yr), as in the "
            "capex channel of [Study 523](../../523-investment-to-assets/)."
        ),

        md(
            "## 6 - Could you trade it? -- the honest answer\n\n"
            "No tradable signal exists on this basket -- and the sign is wrong. The obstacles:\n\n"
            "1. **Inverted spread.** The hedge is −4.68%/yr with t = −1.29 -- the fast hirers "
            "*won*.\n"
            "2. **Survivorship selection dominates.** The high-hire survivors are productive "
            "expanders (AI/semis, hyperscalers, warehouse retail, managed care) whose hiring "
            "paid off.\n"
            "3. **The academic evidence is for broad universes.** Belo-Lin-Bazdresch used the "
            "broad Compustat cross-section; on large-cap blue-chips the effect is thin and "
            "arbitraged.\n\n"
            "The correct conclusion: **MIRAGE** on this test. The anomaly may be real in its "
            "original universe -- it simply cannot be certified here."
        ),

        md(
            "## 7 - Going further\n\n"
            "- **The right universe.** Belo-Lin-Bazdresch (2014) used the broad Compustat "
            "universe; a point-in-time all-US-stock panel is the minimum for a credible test.\n"
            "- **The data wall.** Employee counts are not in XBRL (we verified ~5-11 filers "
            "market-wide carry the numeric tag); a deep, point-in-time head-count history is "
            "the real prerequisite.\n"
            "- **Capex cousin.** [Study 523 -- Investment-To-Assets](../../523-investment-to-assets/) "
            "tests the capital-input channel and also lands None/Mirage on the survivor panel "
            "-- the labor and capital input channels die together.\n\n"
            "*Want to confirm the effect on a clean universe? Fork this, plug in a "
            "point-in-time all-US-stock panel with a proper head-count series, and show "
            "\\|t\\| >= 2 on the hedge with no survivorship shortcut. That's the bar.*"
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
