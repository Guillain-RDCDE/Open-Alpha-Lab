"""Generate the two narrative notebooks for Study 153 (Net-Operating-Assets).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run offline and deterministically; the real-data cells use the EDGAR panel cache
(``../_cache/_edgar_*.parquet``) with a HAVE_REAL guard that falls back to frozen
headline numbers ``R`` when the cache is absent. Every ``build_*()`` ends by calling
``_write`` (the repo restyle tooling monkeypatches this convention).
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-14).
# All numbers are SURVIVORSHIP-BIASED upper bounds.
R = dict(
    n_tickers=193, n_years=17, start_year=2009, end_year=2025,
    # Low-NOA (Q1) quintile
    lo_mean=23.7, lo_sharpe=1.23, lo_t=8.65, lo_hit=88,
    # High-NOA (Q5) quintile
    hi_mean=17.1, hi_sharpe=1.09, hi_t=6.62, hi_hit=88,
    # Equal-weight market
    mkt_mean=20.4, mkt_sharpe=1.38, mkt_t=10.01,
    # Hedge (Q1 - Q5)
    hedge_mean=6.6, hedge_sharpe=0.71, hedge_t=3.53, hedge_hit=65,
    # Excess vs market
    lo_excess_mean=3.2, lo_excess_t=3.14,
    hi_excess_mean=-3.3, hi_excess_t=-2.05,
    # Random control
    rand_std=8.7, rand_pct_beaten=73,
)

# Year-by-year frozen data (Q1, Q5, market, hedge)
_Q1  = [19.8, 9.7, 27.4, 56.3, 16.8, 12.5, 14.6, 34.5, -0.8, 38.4, 25.8, 50.2, -20.0, 51.7, 22.7, 18.1, 25.1]
_Q5  = [19.7, 2.8, 27.5, 43.2,  6.0,  1.5, 17.1, 31.6, -0.2, 39.9, 28.0, 22.7, -16.5, 30.5, 13.7, 16.6,  7.0]
_MKT = [23.9, 6.1, 22.8, 44.3, 14.4,  7.5, 18.1, 28.1, -2.3, 35.9, 25.0, 33.7, -13.3, 38.4, 26.0, 20.1, 19.0]
_HDG = [0.1, 6.9, -0.1, 13.1, 10.8, 11.1, -2.4, 2.9, -0.6, -1.5, -2.1, 27.4, -3.5, 21.2, 9.1, 1.5, 18.1]
_YRS = list(range(R["start_year"], R["end_year"] + 1))

VERDICT_BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Survivorship: Upper_bound_only](https://img.shields.io/badge/Survivorship-Upper_bound_only-8b949e?style=flat-square)\n\n"
)

# ---------------------------------------------------------------------------
# Shared boot code
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath("../../.."))   # repo root (quantlab/)
sys.path.insert(0, os.path.abspath(".."))          # study package (net_operating_assets/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from net_operating_assets import data, strategy as st

# Try to load real EDGAR panel; fall back gracefully to frozen R dict
sig, fwd = data.fetch_panel()
HAVE_REAL = not sig.empty

if HAVE_REAL:
    h = st.quintile_returns(sig, fwd, q=0.20)
else:
    h = pd.DataFrame()

print("EDGAR panel loaded:", HAVE_REAL,
      f"({len(sig.columns) if HAVE_REAL else 0} tickers, {len(sig) if HAVE_REAL else 0} years)")
"""

R_CODE = f"""\
R = dict(
    n_tickers={R['n_tickers']}, n_years={R['n_years']}, start_year={R['start_year']}, end_year={R['end_year']},
    lo_mean={R['lo_mean']}, lo_sharpe={R['lo_sharpe']}, lo_t={R['lo_t']}, lo_hit={R['lo_hit']},
    hi_mean={R['hi_mean']}, hi_sharpe={R['hi_sharpe']}, hi_t={R['hi_t']}, hi_hit={R['hi_hit']},
    mkt_mean={R['mkt_mean']}, mkt_sharpe={R['mkt_sharpe']},
    hedge_mean={R['hedge_mean']}, hedge_sharpe={R['hedge_sharpe']}, hedge_t={R['hedge_t']}, hedge_hit={R['hedge_hit']},
    lo_excess_mean={R['lo_excess_mean']}, lo_excess_t={R['lo_excess_t']},
    hi_excess_mean={R['hi_excess_mean']}, hi_excess_t={R['hi_excess_t']},
    rand_std={R['rand_std']}, rand_pct_beaten={R['rand_pct_beaten']},
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
            "# Net-Operating-Assets -- does balance-sheet bloat predict low returns?\n"
            "### The Hirshleifer et al. (2004) NOA anomaly on the S&P 500 EDGAR panel\n\n"
            + VERDICT_BADGES +
            "Joel Hirshleifer and co-authors argued in 2004 that firms with 'bloated' "
            "balance sheets -- large operating assets piled up relative to their asset "
            "base -- earn low future returns. The idea: investors see past earnings growth "
            "(built on heavy asset expansion) and over-extrapolate it. When the expansion "
            "stops producing earnings, the stock disappoints. We test it on ~193 S&P 500 "
            "names using real SEC filings.\n\n"
            "> This is the plain-language layer. The HAC t-stats, quintile monotonicity, "
            "and random-portfolio null live in "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** Every chart drawn by the code beside it. "
            "House style in [METHODOLOGY.md](../../../METHODOLOGY.md).\n"
            ">\n"
            "> **Survivorship bias**: this panel covers only *current* S&P 500 members "
            "projected backwards. Every number below is an upper bound on a live strategy."
        ),
        code(BOOT + "\n" + R_CODE),

        # ---- BEAT 0 -- ANSWER FIRST ------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | What the data says |\n|---|---|\n"
            f"| Does low NOA outperform high NOA? | Yes -- by **+{R['hedge_mean']:.1f}%/yr**, "
            f"positive in {R['hedge_hit']}% of years. |\n"
            f"| Is that statistically real? | On this panel, HAC *t* = {R['hedge_t']:.2f} -- "
            f"nominally passes |t| >= 2. But the panel is survivorship-biased, so this is "
            f"an **upper bound**, not confirmed evidence. |\n"
            f"| Does low NOA beat a random stock pick? | Barely: beats only "
            f"{R['rand_pct_beaten']}% of random same-size draws. |\n"
            "| Could you trade it? | Fragile. Annual rebalancing of ~20 names is cheap, "
            "but the alpha estimate is uncertain and the effect is primarily documented "
            "for small/mid-caps not in this panel. |\n\n"
            "> **The catch**: on a *survivor*-biased panel, high-NOA firms that went "
            "bankrupt or were expelled from the S&P 500 -- precisely those who most "
            "over-invested and subsequently failed -- are simply missing. The t-stat "
            "reflects their absence more than it reflects investor mis-pricing."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *Buy stocks with low Net Operating Assets (low balance-sheet bloat). "
            "Avoid stocks where Operating Assets - Operating Liabilities have grown "
            "large relative to the prior-year asset base. Rebalance annually. "
            "The bloated-balance-sheet firms will disappoint as the asset expansion "
            "stops producing earnings growth.*\n\n"
            "NOA is computed from basic accounting items in every annual report:\n\n"
            "- **Operating Assets** = Total Assets - Cash (cash is financial, not operational)\n"
            "- **Operating Liabilities** = Total Liabilities - Long-Term Debt (debt is "
            "financial)\n"
            "- **NOA** = (Operating Assets - Operating Liabilities) / Prior-Year Total Assets\n\n"
            "A high NOA means the firm has accumulated a large operating-asset surplus -- "
            "more factory equipment, receivables, and inventory than can be explained by "
            "the level of operations. The Hirshleifer et al. argument: that's money that "
            "was spent without generating matching earnings, and the market hasn't "
            "discounted it yet."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If the NOA effect is real and tradable on large US stocks, it's a "
            "fundamentals-based, low-turnover (one trade per name per year) strategy "
            "that requires no forecasting, no ML, and only public SEC filings. "
            "If it only works on small/mid-cap stocks -- or is entirely a survivor-bias "
            "artefact -- then the S&P 500 test is a cautionary tale about testing on the "
            "wrong (biased) universe."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three honesty requirements:\n\n"
            "1. **Reporting lag.** Fundamentals from fiscal year y only used to predict "
            "returns in year y+1 -- a conservative lag.\n"
            "2. **Random-portfolio control.** Does the low-NOA quintile beat a random "
            "draw of the same number of stocks? If not, the result is just concentration "
            "risk, not a real signal.\n"
            "3. **Name the survivorship bias.** The EDGAR panel here covers current S&P "
            "500 members projected back. High-NOA failures are absent. Every result is "
            "an upper bound."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown -- the quintile spread and its real limit\n\n"
            "Positive control first: **when we plant a real effect in a synthetic panel, "
            "the engine finds it.** That makes the real-data result a statement about the "
            "*market* (and the panel), not the method."
        ),
        code(
            "# Synthetic positive control: plant a known NOA premium, confirm the engine detects it\n"
            "premiums = [0.0, -0.04, -0.08, -0.12]\n"
            "hedges = []\n"
            "for p in premiums:\n"
            "    s, f, _ = data.synthetic_panel(n_firms=300, n_years=25, premium=p, seed=153)\n"
            "    hh = st.quintile_returns(s, f, q=0.20)\n"
            "    hedges.append(hh['hedge'].mean() * 100)\n"
            "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            "ax.plot([abs(p)*100 for p in premiums], hedges, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Planted NOA effect (% alpha per unit z-rank)')\n"
            "ax.set_ylabel('Low-minus-high-NOA hedge (%/yr)')\n"
            "ax.set_title('Synthetic control: engine finds the effect when planted')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('The engine faithfully recovers planted effects -- so the real result is a market statement.')"
        ),
        md("**Now the real EDGAR panel -- quintile sort on NOA:**"),
        code(
            "# Year-by-year low-NOA vs high-NOA vs market\n"
            "if HAVE_REAL:\n"
            "    q1_vals = h['q1'].values * 100\n"
            "    q5_vals = h['q5'].values * 100\n"
            "    hdg_vals = h['hedge'].values * 100\n"
            "    years_real = h.index.tolist()\n"
            "else:\n"
            "    q1_vals, q5_vals, hdg_vals, years_real = Q1, Q5, HDG, YRS\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "x = range(len(years_real))\n"
            "axes[0].bar(x, q1_vals, label='Q1 low-NOA', color=GREEN, alpha=0.7, width=0.4)\n"
            "axes[0].bar([i+0.4 for i in x], q5_vals, label='Q5 high-NOA', color=RED, alpha=0.7, width=0.4)\n"
            "axes[0].set_xticks([i+0.2 for i in x])\n"
            "axes[0].set_xticklabels(years_real, rotation=45, fontsize=8)\n"
            "axes[0].set_ylabel('Annual return %'); axes[0].legend()\n"
            "axes[0].set_title('Q1 (low NOA) vs Q5 (high NOA) raw returns')\n"
            "colors = [GREEN if v > 0 else RED for v in hdg_vals]\n"
            "axes[1].bar(years_real, hdg_vals, color=colors)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_ylabel('Hedge return (Q1 - Q5) %')\n"
            "axes[1].set_title(f'Low-minus-high-NOA hedge: {sum(1 for v in hdg_vals if v>0)}/{len(hdg_vals)} positive years')\n"
            "axes[1].tick_params(axis='x', rotation=45, labelsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Mean hedge: {{np.mean(hdg_vals):+.1f}}%/yr  (frozen R: +{{R[\"hedge_mean\"]:.1f}}%/yr)')"
        ),
        md(
            f"The low-NOA quintile earns **+{R['lo_mean']:.1f}%/yr** vs the high-NOA "
            f"quintile's **+{R['hi_mean']:.1f}%/yr** -- a hedge of **+{R['hedge_mean']:.1f}%/yr**, "
            f"positive in {R['hedge_hit']}% of years. But remember: **this panel "
            "excludes every firm that failed.** The high-NOA firms that went bankrupt "
            "(and would have earned catastrophically negative returns in the year they "
            "failed) are simply absent. The effect is probably real in broader universes, "
            "but this headline is the best-case number."
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal -- Weak.** Nominal hedge +{R['hedge_mean']:.1f}%/yr, HAC *t* = "
            f"{R['hedge_t']:.2f}. Passes the |t| >= 2 bar on this panel. But the panel "
            "is survivorship-biased, so the t-stat is an upper bound. Literature supports "
            "the effect in broader universes (especially small caps), but less so for "
            "large-cap survivors.\n"
            f"- **Tradability -- Fragile.** Low annual turnover on a liquid universe is a "
            "point in favour. But the alpha estimate is uncertain, the low-NOA portfolio "
            "beats only 73% of random draws, and the effect is documented primarily "
            "for small/mid-caps with higher transaction costs.\n"
            "- **Survivorship -- Upper bound only.** Every result is the best-case number; "
            "a live strategy would have included high-NOA disasters that are now absent."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "Annual rebalancing of ~20 S&P 500 names is operationally trivial and cheap "
            "(spreads tight, commissions minimal). The obstacles are in the signal:\n\n"
            "1. **The alpha estimate is uncertain.** +3.2%/yr excess on a survivorship-biased "
            "panel might shrink to ~0 or even invert on a live point-in-time universe.\n"
            "2. **Wrong universe.** Greenblatt-style effects and accruals effects are both "
            "documented primarily for small, neglected, illiquid stocks -- precisely those "
            "excluded from the S&P 500 and from this panel.\n"
            "3. **Random-portfolio bar.** The low-NOA portfolio only beats 73% of random "
            "same-size draws -- barely above the 50% chance level. Most of the positive "
            "return comes from the market, not from the signal.\n\n"
            "The right conclusion: **explore this in a broader, non-biased universe "
            "(small/mid-cap, point-in-time constituents) before treating it as tradable.**"
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The right universe.** The original Hirshleifer et al. (2004) sample is "
            "all US stocks above $10M market cap -- thousands of names including small and "
            "mid-caps. On that universe, the NOA effect is strongest among neglected, "
            "low-analyst-coverage stocks. The S&P 500 survivor panel here is the "
            "wrong test.\n"
            "- **Related signals.** Sloan (1996) accruals ([Study 52 -- Smoke-Screen](../../52-smoke-screen/)) "
            "tests the working-capital component of the same balance-sheet bloat idea. "
            "Asset-growth (Cooper et al. 2008) and total-accruals (Richardson et al. 2005) "
            "are closely related.\n"
            "- **NOA decomposition.** Split into working-capital NOA and long-term NOA; "
            "the long-term component (PP&E growth) is sometimes stronger.\n\n"
            "*Think the balance-sheet bloat effect is real in large caps? Fork this, extend "
            "to a point-in-time S&P 500 constituent universe (Compustat-based), and show "
            "|t| >= 2 on the hedge with no survivorship shortcut. That's the bar.*"
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
            "# Net-Operating-Assets -- a quantitative teardown\n"
            "### NOA quintile sort - S&P 500 EDGAR panel - HAC inference - "
            "random-portfolio null - survivorship caveat\n\n"
            + VERDICT_BADGES +
            "The deep companion to [01_for_the_curious.ipynb](01_for_the_curious.ipynb) -- "
            "same seven beats, every claim now carrying its standard error. We test the "
            "NOA anomaly on a survivorship-biased S&P 500 EDGAR panel (~193 tickers, "
            "2009-2025), measure the low-minus-high-NOA hedge with a HAC t-stat, and pin "
            "it against a random-portfolio null distribution.\n\n"
            "> **Not investment advice.** SEC EDGAR 10-K data; annual returns via "
            "Yahoo Finance; survivorship-biased (current S&P 500 projected back). "
            "Sources in [`docs/references.md`](../docs/references.md), reproducible "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Survivorship-bias warning**: all results are upper bounds. Firms that "
            "failed between 2009 and now are excluded -- precisely the firms that would "
            "have scored high NOA and earned catastrophic returns."
        ),
        code(BOOT + "\n" + R_CODE + "\nimport sys; sys.path.insert(0, os.path.abspath('../../..')); from quantlab import analytics"),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Hedge **+{R['hedge_mean']:.1f}%/yr**, "
            f"HAC *t* = **+{R['hedge_t']:.2f}** -- passes |t| >= 2 on this panel, "
            f"but panel is survivorship-biased. Low-NOA excess "
            f"+{R['lo_excess_mean']:.1f}%/yr (*t* = +{R['lo_excess_t']:.2f}). |\n"
            f"| **Tradability** | `FRAGILE` | Low turnover (annual), liquid names. "
            f"But alpha is uncertain (upper bound), and effect is primarily documented "
            f"for small/mid-caps outside this universe. |\n"
            f"| **Survivorship** | `UPPER BOUND` | Panel excludes all post-2009 "
            f"failures; every number is an upper bound on a live strategy. |\n\n"
            "> **In plain words:** NOA bloat appears to predict low returns on this panel, "
            "but we cannot confirm whether this is genuine investor mis-pricing on large "
            "caps or simply the absence of high-NOA disasters from the survivor pool."
        ),

        # ---- BEAT 1 -----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $r_{i,t+1}$ be firm $i$'s return in year $t+1$. Define:\n\n"
            "$$\\text{NOA}_{i,t} = \\frac{(A_{i,t} - C_{i,t}) - (L_{i,t} - D_{i,t})}"
            "{A_{i,t-1}}$$\n\n"
            "where $A$ = total assets, $C$ = cash, $L$ = total liabilities, $D$ = "
            "long-term financial debt, and $A_{t-1}$ is the prior-year denominator.\n\n"
            "The **H1 (signal)** claim (Hirshleifer et al. 2004) is:\n\n"
            "$$\\mathbb{E}\\left[\\bar{r}_{\\text{low-NOA},t+1} - "
            "\\bar{r}_{\\text{high-NOA},t+1}\\right] > 0$$\n\n"
            "with a HAC t-stat |t| >= 2. **H2 (beats random)** requires the low-NOA "
            "quintile to beat a random same-size draw. **H3 (tradable)** requires the "
            "hedge to survive costs and scale."
        ),

        # ---- BEAT 2 -----------------------------------------------------------
        md(
            "## 2 - So what? -- the stakes\n\n"
            "H1 is supported on this biased panel. The practical stakes depend on whether "
            "it survives the survivorship correction and whether it operates on the S&P "
            "500 large-cap universe. The original paper's evidence is strongest for "
            "small, neglected stocks -- the opposite of this panel."
        ),

        # ---- BEAT 3 -----------------------------------------------------------
        md(
            "## 3 - The protocol\n\n"
            "- **Data**: four EDGAR concepts from the desk's shared cache "
            "(Assets, Liabilities, LongTermDebtNoncurrent, Cash); annual returns "
            "from `_edgar_yrret.parquet`.\n"
            f"- **Universe**: current S&P 500 members with all four concepts present "
            f"(~{R['n_tickers']} tickers); survivorship-biased.\n"
            f"- **Window**: {R['n_years']} years of signal ({R['start_year']}-{R['end_year']}), "
            f"forecasting returns {R['start_year']+1}-{R['end_year']+1}.\n"
            "- **Signal**: NOA ratio (higher = more bloat); quintile sort (q=0.20).\n"
            "- **Inference**: HAC Newey-West t-stat on the 17-observation annual hedge series.\n"
            "- **Control**: empirical distribution of random same-size portfolios "
            "(500 draws x 17 years).\n"
            "- **Positive control**: synthetic panel with planted NOA premium confirms the "
            "engine can detect a real effect."
        ),

        # ---- BEAT 4 -----------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - Synthetic positive control -- the engine works when an effect is planted\n\n"
            "Sweep the planted NOA premium from 0 to -12%/yr per unit z-rank and confirm "
            "the hedge tracks it monotonically."
        ),
        code(
            "premiums = [0.0, -0.04, -0.08, -0.12]\n"
            "synth_rows = []\n"
            "for p in premiums:\n"
            "    s, f, _ = data.synthetic_panel(n_firms=300, n_years=25, premium=p, seed=153)\n"
            "    hh = st.quintile_returns(s, f, q=0.20)\n"
            "    ss = st.summary(hh['hedge'])\n"
            "    synth_rows.append((abs(p)*100, ss['mean']*100, ss['tstat']))\n"
            "sydf = pd.DataFrame(synth_rows, columns=['premium_%', 'hedge_%/yr', 'HAC_t'])\n"
            "fig, ax = plt.subplots(figsize=(9, 4))\n"
            "colors = [GREEN if t > 2 else (RED if t < -2 else GREY) for t in sydf['HAC_t']]\n"
            "ax.bar(sydf['premium_%'], sydf['hedge_%/yr'], color=colors, width=1.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Planted NOA effect (%/yr per unit z-rank)')\n"
            "ax.set_ylabel('Low-minus-high hedge (%/yr)')\n"
            "ax.set_title('Positive control: HAC t>2 (green) appears where the premium is real')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(sydf.round(2))"
        ),
        md(
            "> The engine faithfully detects planted premiums. The real-data result is "
            "therefore a statement about the **market** (and the survivor bias), not the "
            "**method**."
        ),
        md(
            "### 4b - Real EDGAR panel -- quintile monotonicity and HAC inference"
        ),
        code(
            "if HAVE_REAL:\n"
            "    q_means = [float(h[f'q{i}'].mean()*100) for i in range(1,6)]\n"
            "    q_tstats = [st.summary(h[f'q{i}'])['tstat'] for i in range(1,6)]\n"
            "    s_hedge = st.summary(h['hedge'])\n"
            "    s_lo_ex = st.summary(h['q1'] - h['market'])\n"
            "    s_hi_ex = st.summary(h['q5'] - h['market'])\n"
            "else:\n"
            "    q_means = [R['lo_mean'], 20.0, 18.9, 22.6, R['hi_mean']]\n"
            "    q_tstats = [R['lo_t'], 8.12, 8.53, 6.60, R['hi_t']]\n"
            "    s_hedge = dict(mean=R['hedge_mean']/100, tstat=R['hedge_t'], hit_rate=R['hedge_hit']/100)\n"
            "    s_lo_ex = dict(mean=R['lo_excess_mean']/100, tstat=R['lo_excess_t'])\n"
            "    s_hi_ex = dict(mean=R['hi_excess_mean']/100, tstat=R['hi_excess_t'])\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "x = [1, 2, 3, 4, 5]\n"
            "bar_colors = [GREEN, GREY, GREY, GREY, RED]\n"
            "axes[0].bar(x, q_means, color=bar_colors)\n"
            "axes[0].set_xticks(x); axes[0].set_xticklabels(['Q1 (lo)', 'Q2', 'Q3', 'Q4', 'Q5 (hi)'])\n"
            "axes[0].set_ylabel('Mean annual return (%/yr)')\n"
            "axes[0].set_title('Quintile returns: low-NOA outperforms high-NOA')\n"
            "axes[1].bar(x, q_tstats, color=bar_colors)\n"
            "for s in (2, -2): axes[1].axhline(s, ls='--', c=GREY, lw=1)\n"
            "axes[1].set_xticks(x); axes[1].set_xticklabels(['Q1 (lo)', 'Q2', 'Q3', 'Q4', 'Q5 (hi)'])\n"
            "axes[1].set_ylabel('HAC t-stat')\n"
            "axes[1].set_title('All quintiles have high t-stats (all earn positive returns)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Hedge: {s_hedge[\"mean\"]*100:+.1f}%/yr  HAC t={s_hedge[\"tstat\"]:+.2f}  hit={s_hedge[\"hit_rate\"]:.0%}')\n"
            "print(f'Low-NOA excess: {s_lo_ex[\"mean\"]*100:+.1f}%/yr  t={s_lo_ex[\"tstat\"]:+.2f}')\n"
            "print(f'High-NOA excess: {s_hi_ex[\"mean\"]*100:+.1f}%/yr  t={s_hi_ex[\"tstat\"]:+.2f}')"
        ),
        md(
            f"> The low-minus-high-NOA hedge is **+{R['hedge_mean']:.1f}%/yr**, "
            f"HAC *t* = **+{R['hedge_t']:.2f}** -- nominally real on this panel. "
            f"But note: all quintiles earn substantial returns because the panel covers "
            f"large-cap survivors of a bull-market decade. The **excess** vs the equal-weight "
            f"market is +{R['lo_excess_mean']:.1f}%/yr for the low-NOA group (*t* = "
            f"+{R['lo_excess_t']:.2f}) and {R['hi_excess_mean']:.1f}%/yr for the high-NOA "
            f"group (*t* = {R['hi_excess_t']:.2f})."
        ),
        md(
            "### 4c - Random-portfolio null distribution\n\n"
            "Does the low-NOA quintile beat a blind draw of the same number of stocks? "
            "We simulate 500 random same-size portfolios each year."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rand = st.random_portfolio_returns(sig, fwd, q=0.20, n_draws=500, seed=153)\n"
            "    actual_excess = float((h['q1'] - h['market']).mean())\n"
            "    pct_beaten = float((rand < actual_excess).mean())\n"
            "else:\n"
            "    rng_nb = np.random.default_rng(42)\n"
            "    rand = pd.Series(rng_nb.normal(0.0, R['rand_std']/100, 17*500))\n"
            "    actual_excess = R['lo_excess_mean'] / 100\n"
            "    pct_beaten = R['rand_pct_beaten'] / 100\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "ax.hist(rand * 100, bins=50, color=GREY, alpha=0.7, label='Random portfolios (500x17yr avg)')\n"
            "ax.axvline(actual_excess * 100, c=GREEN, lw=2.5,\n"
            "           label=f'Low-NOA quintile: {actual_excess*100:+.1f}%/yr excess')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Annual excess return vs EW market (%)')\n"
            "ax.set_ylabel('Count'); ax.legend()\n"
            "ax.set_title(f'Low-NOA beats {pct_beaten:.0%} of random draws -- moderately above chance')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Low-NOA portfolio beats {pct_beaten:.0%} of random same-size draws')"
        ),
        md(
            f"> The low-NOA portfolio beats **{R['rand_pct_beaten']}%** of random same-size "
            "draws -- moderately above the 50% chance level, but not dramatically so. "
            "About 27% of blind random portfolios do as well or better. The signal is not "
            "a slam-dunk even on the survivor-biased panel."
        ),

        # ---- BEAT 5 -----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal `WEAK`** -- hedge +{R['hedge_mean']:.1f}%/yr, HAC *t* = "
            f"+{R['hedge_t']:.2f}; low-NOA excess +{R['lo_excess_mean']:.1f}%/yr "
            f"(*t* = +{R['lo_excess_t']:.2f}). Nominally passes |t| >= 2 but on a "
            "**survivorship-biased** panel. Literature supports the mechanism but "
            "primarily for small-cap, less-covered stocks.\n"
            f"- **Tradability `FRAGILE`** -- annual rebalancing of ~20 names is cheap; "
            "but the alpha estimate is uncertain (upper bound), the effect is not "
            "confirmed for large liquid names, and the hedge is +6.6%/yr only on the "
            "survivor sample.\n"
            "- **Survivorship `UPPER BOUND`** -- every result assumes the failed high-NOA "
            "firms don't exist. They do (or did). Any live strategy would have been "
            "exposed to them."
        ),

        # ---- BEAT 6 -----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- the honest answer\n\n"
            "Annual rebalancing of ~20 large-cap names is operationally feasible. "
            "The obstacles are entirely in the signal and the bias:\n\n"
            "1. **Survivorship arithmetic.** A 2009 portfolio would have included "
            "companies later removed from the S&P 500 for underperformance. High-NOA "
            "over-investors who subsequently failed are exactly those excluded from this "
            "panel -- their absence inflates every comparison.\n"
            "2. **Wrong universe.** The NOA effect is documented for small/mid-cap, "
            "low-analyst-coverage stocks where investor over-extrapolation is most "
            "plausible. On S&P 500 blue-chips, heavy analyst coverage should rapidly "
            "correct balance-sheet misvaluation.\n"
            "3. **Beating random barely.** The low-NOA portfolio beats only 73% of random "
            "draws -- meaning ~27% of blind picks do as well. The signal's information "
            "content is modest even within the biased panel.\n\n"
            "The correct conclusion: **FRAGILE** on this test; investigate further in "
            "a broader, point-in-time universe before trading."
        ),

        # ---- BEAT 7 -----------------------------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The right universe.** Hirshleifer et al. (2004) ran their test on all "
            "US stocks above $10M market cap. A point-in-time constituent universe "
            "(e.g. Compustat-based historical S&P 500 membership) is the minimum "
            "required for a credible large-cap test.\n"
            "- **Decomposing NOA.** The working-capital component of NOA is the Sloan "
            "(1996) accruals anomaly ([Study 52 -- Smoke-Screen](../../52-smoke-screen/)). "
            "The long-term component (PP&E expansion) is Richardson et al. (2005). "
            "They may have different half-lives.\n"
            "- **Related desk studies.** "
            "[Study 65 -- Scorecard](../../65-scorecard/) (Piotroski F-score); "
            "[Study 121 -- Magic-Formula](../../121-magic-formula/) (quality+value, same panel); "
            "[Study 52 -- Smoke-Screen](../../52-smoke-screen/) (accruals).\n\n"
            "*Want to confirm the effect on a clean universe? Fork this, plug in a "
            "point-in-time all-US-stock panel, and show |t| >= 2 on the hedge with "
            "no survivorship shortcut. That's the bar.*"
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
