"""Generate the two narrative notebooks for Study 244 (Asset-Growth).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run offline and deterministically; the real-data cells use the EDGAR panel cache
(``../_cache/_edgar_Assets.parquet``) with a HAVE_REAL guard that falls back to frozen
headline numbers ``R`` when the cache is absent. Every ``build_*()`` ends by calling
``_write`` (the repo restyle tooling monkeypatches this convention).
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-16).
# All numbers are SURVIVORSHIP-BIASED; the predicted hedge direction is INVERTED.
R = dict(
    n_tickers=419, n_years=17, start_year=2009, end_year=2025,
    # Low-AG (Q1) quintile
    lo_mean=19.9, lo_sharpe=1.37, lo_t=9.16, lo_hit=88,
    # High-AG (Q5) quintile
    hi_mean=20.9, hi_sharpe=1.19, hi_t=7.57, hi_hit=94,
    # Equal-weight market
    mkt_mean=17.9, mkt_sharpe=1.37, mkt_t=9.53,
    # Hedge (Q1 - Q5): NEGATIVE — anomaly absent / inverted on this panel
    hedge_mean=-1.0, hedge_sharpe=-0.08, hedge_t=-0.45, hedge_hit=59,
    # Excess vs market
    lo_excess_mean=2.0, lo_excess_t=1.88,
    hi_excess_mean=3.0, hi_excess_t=1.90,
    # Random control
    rand_std=3.7, rand_pct_beaten=74,
)

# Year-by-year frozen data (Q1, Q5, market, hedge)
_Q1  = [22.8, 3.9, 29.9, 50.8, 17.1, -1.2, 25.7, 26.5, -6.8, 33.2, 16.6, 33.2, 1.6, 17.9, 29.9, 15.7, 21.3]
_Q5  = [11.7, 2.6, 28.4, 38.5, 24.8, 11.9, 11.8, 35.0,  2.8, 40.2, 33.9, 31.7,-18.9, 52.7, 24.6, 13.1, 11.0]
_MKT = [22.0, 3.6, 22.1, 40.4, 19.2,  4.6, 18.5, 27.2, -2.8, 34.0, 17.3, 32.7, -8.8, 25.4, 20.0, 15.6, 13.0]
_HDG = [11.0, 1.4,  1.6, 12.3, -7.7,-13.1, 13.9, -8.5, -9.6, -7.0,-17.3,  1.4, 20.6,-34.8,  5.2,  2.6, 10.4]
_YRS = list(range(R["start_year"], R["end_year"] + 1))

VERDICT_BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Survivorship: Selection_dominates](https://img.shields.io/badge/Survivorship-Selection_dominates-8b949e?style=flat-square)\n\n"
)

# ---------------------------------------------------------------------------
# Shared boot code
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath("../../.."))   # repo root (quantlab/)
sys.path.insert(0, os.path.abspath(".."))          # study package (asset_growth/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from asset_growth import data, strategy as st

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
            "# Asset-Growth -- do fast-growing-asset companies underperform?\n"
            "### The Cooper, Gulen & Schill (2008) asset-growth anomaly on the S&P 500 EDGAR panel\n\n"
            + VERDICT_BADGES +
            "Mike Cooper, Huseyin Gulen and Michael Schill documented in 2008 that firms "
            "expanding their total assets aggressively earn lower future returns. The idea: "
            "managers over-invest, markets extrapolate past growth, and returns disappoint "
            "when the expansion stops producing earnings. We test it on ~419 S&P 500 names "
            "using real SEC filings.\n\n"
            "> This is the plain-language layer. The HAC t-stats, quintile monotonicity, "
            "and random-portfolio null live in "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** Every chart drawn by the code beside it. "
            "House style in [METHODOLOGY.md](../../../METHODOLOGY.md).\n"
            ">\n"
            "> **Survivorship bias**: this panel covers only *current* S&P 500 members "
            "projected backwards. Fast-growing survivors skew toward tech giants, "
            "reversing the predicted sign."
        ),
        code(BOOT + "\n" + R_CODE),

        # ---- BEAT 0 -- ANSWER FIRST ------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | What the data says |\n|---|---|\n"
            f"| Does low AG outperform high AG? | No -- the hedge is **{R['hedge_mean']:+.1f}%/yr** "
            f"(wrong direction). High-AG firms *slightly outperform* on this panel. |\n"
            f"| Is that statistically real? | HAC *t* = {R['hedge_t']:.2f} -- "
            f"nowhere near |t| >= 2. The anomaly is **absent**. |\n"
            f"| Does low AG beat a random stock pick? | Barely: beats "
            f"{R['rand_pct_beaten']}% of random same-size draws -- close to chance. |\n"
            "| Could you trade it? | No signal to trade. The predicted long-short inverts "
            "direction on this survivor panel. |\n\n"
            "> **The catch**: on a *survivor*-biased S&P 500 panel, the high-AG quintile "
            "includes Apple, Amazon, Nvidia, and Google -- firms that grew aggressively "
            "and delivered extraordinary returns. Firms that grew aggressively and "
            "subsequently *failed* are simply missing. Survivorship selection dominates "
            "and reverses the predicted sign."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *Sell stocks with high total-asset growth (aggressive expanders). "
            "Buy stocks with low total-asset growth (disciplined growers). "
            "Rebalance annually. The over-investing firms will disappoint as "
            "the expansion stops producing earnings growth.*\n\n"
            "Asset growth is computed from the simplest accounting item:\n\n"
            "- **AG_t** = (Total Assets_t - Total Assets_{t-1}) / Total Assets_{t-1}\n\n"
            "A high AG means the firm expanded its balance sheet aggressively -- "
            "more property, equipment, acquisitions, or working capital relative to "
            "last year. The Cooper et al. argument: that expansion often reflects "
            "over-investment by managers (Jensen 1986 agency costs) or market "
            "extrapolation of past success, and the stock underperforms as a result."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If the asset-growth effect is real and tradable on large US stocks, it "
            "forms the basis of the Fama-French CMA (Conservative Minus Aggressive) "
            "factor -- one of the five factors in the 2015 model. It would also be a "
            "low-turnover, low-ML, fundamentals-only strategy requiring only public "
            "10-K filings. If it only works on small/mid-cap stocks -- or is entirely "
            "a selection-bias artefact -- then the S&P 500 test is a cautionary tale "
            "about testing in the wrong (biased) universe."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three honesty requirements:\n\n"
            "1. **Reporting lag.** Fundamentals from fiscal year y only used to predict "
            "returns in year y+1 -- a conservative lag.\n"
            "2. **Random-portfolio control.** Does the low-AG quintile beat a random "
            "draw of the same number of stocks? If not, the result is just concentration "
            "risk, not a real signal.\n"
            "3. **Name the survivorship bias.** The EDGAR panel here covers current S&P "
            "500 members projected back. Failed high-growers are absent. Every result is "
            "conditional on survival."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown -- the quintile spread and its real (non-) result\n\n"
            "Positive control first: **when we plant a real effect in a synthetic panel, "
            "the engine finds it.** That makes the real-data result a statement about the "
            "*market* (and the panel bias), not the method."
        ),
        code(
            "# Synthetic positive control: plant a known AG premium, confirm the engine detects it\n"
            "premiums = [0.0, -0.04, -0.08, -0.12]\n"
            "hedges = []\n"
            "for p in premiums:\n"
            "    s, f, _ = data.synthetic_panel(n_firms=300, n_years=25, premium=p, seed=244)\n"
            "    hh = st.quintile_returns(s, f, q=0.20)\n"
            "    hedges.append(hh['hedge'].mean() * 100)\n"
            "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            "ax.plot([abs(p)*100 for p in premiums], hedges, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Planted AG effect (% alpha per unit z-rank)')\n"
            "ax.set_ylabel('Low-minus-high-AG hedge (%/yr)')\n"
            "ax.set_title('Synthetic control: engine finds the effect when planted')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('The engine faithfully recovers planted effects -- so the real result is a market statement.')"
        ),
        md("**Now the real EDGAR panel -- quintile sort on asset growth:**"),
        code(
            "# Year-by-year low-AG vs high-AG vs market\n"
            "if HAVE_REAL:\n"
            "    q1_vals = h['q1'].values * 100\n"
            "    q5_vals = h['q5'].values * 100\n"
            "    hdg_vals = h['hedge'].values * 100\n"
            "    years_real = h.index.tolist()\n"
            "else:\n"
            "    q1_vals, q5_vals, hdg_vals, years_real = Q1, Q5, HDG, YRS\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "x = range(len(years_real))\n"
            "axes[0].bar(x, q1_vals, label='Q1 low-AG', color=GREEN, alpha=0.7, width=0.4)\n"
            "axes[0].bar([i+0.4 for i in x], q5_vals, label='Q5 high-AG', color=RED, alpha=0.7, width=0.4)\n"
            "axes[0].set_xticks([i+0.2 for i in x])\n"
            "axes[0].set_xticklabels(years_real, rotation=45, fontsize=8)\n"
            "axes[0].set_ylabel('Annual return %'); axes[0].legend()\n"
            "axes[0].set_title('Q1 (low AG) vs Q5 (high AG) raw returns')\n"
            "colors = [GREEN if v > 0 else RED for v in hdg_vals]\n"
            "axes[1].bar(years_real, hdg_vals, color=colors)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_ylabel('Hedge return (Q1 - Q5) %')\n"
            "axes[1].set_title(f'Low-minus-high-AG hedge: {sum(1 for v in hdg_vals if v>0)}/{len(hdg_vals)} positive years')\n"
            "axes[1].tick_params(axis='x', rotation=45, labelsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Mean hedge: {{np.mean(hdg_vals):+.1f}}%/yr  (frozen R: {{R[\"hedge_mean\"]:+.1f}}%/yr)')"
        ),
        md(
            f"The low-AG quintile earns **+{R['lo_mean']:.1f}%/yr** vs the high-AG "
            f"quintile's **+{R['hi_mean']:.1f}%/yr** -- a hedge of **{R['hedge_mean']:+.1f}%/yr**, "
            f"positive in only {R['hedge_hit']}% of years. **The predicted direction is inverted**: "
            "fast-growing large caps (tech giants) outperform slow growers on this panel. "
            "Survivorship selection dominates the signal completely."
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal -- None.** Hedge {R['hedge_mean']:+.1f}%/yr, HAC *t* = "
            f"{R['hedge_t']:.2f} -- wrong direction, zero signal. The predicted effect "
            "(high AG → low returns) is entirely absent and mildly inverted on this panel.\n"
            f"- **Tradability -- Mirage.** No signal to trade. Long-low-AG / short-high-AG "
            "produces a small negative return with near-zero t-stat. There is no tradable "
            "edge on S&P 500 survivors.\n"
            "- **Survivorship -- Selection dominates.** Fast-growing S&P 500 survivors = "
            "tech titans. Failed fast-growers are gone. The selection reverses the sign."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "No. The signal is absent on this panel. Even setting aside survivorship bias:\n\n"
            "1. **Wrong universe.** Cooper et al. (2008) documented the effect on all "
            "US stocks including small and micro-caps. On large-cap S&P 500 names, "
            "aggressive asset growth often reflects productive expansion by well-managed "
            "companies -- not over-investment.\n"
            "2. **Survivorship reversal.** The S&P 500 panel over-represents successful "
            "high-growers. A live investor in 2009 would have been exposed to high-growth "
            "companies that later failed -- those returns are absent here.\n"
            "3. **The CMA factor paradox.** The asset-growth anomaly is embedded in the "
            "Fama-French five-factor CMA factor -- but the factor evidence is based on "
            "broad cross-sections of all stocks, not large-cap survivors. On liquid, "
            "heavily-covered names the effect is arbitraged away.\n\n"
            "The right conclusion: **explore this in a broader, non-biased universe "
            "(all US stocks, point-in-time constituents) before treating it as tradable.**"
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The right universe.** Cooper et al. (2008) tested on NYSE/AMEX/NASDAQ "
            "stocks above $5M market cap -- thousands of names including small and mid-caps. "
            "On that universe, the asset-growth effect is robust and survives factor controls. "
            "The S&P 500 survivor panel here is the wrong test.\n"
            "- **NOA cousin.** Net Operating Assets ([Study 153](../../153-net-operating-assets/)) "
            "is the Hirshleifer et al. version of the same balance-sheet bloat idea -- "
            "scaled by prior-year assets and focused on operating components. Weak/Fragile "
            "on the same panel (slightly stronger because it filters noise via the NOA ratio).\n"
            "- **CMA factor.** The Fama-French CMA factor operationalises the asset-growth "
            "anomaly at the factor level. The factor has positive returns in broad "
            "cross-sections but much weaker evidence on large-cap-only panels.\n\n"
            "*Think the asset-growth effect is real in large caps? Fork this, extend "
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
            "# Asset-Growth -- a quantitative teardown\n"
            "### AG quintile sort - S&P 500 EDGAR panel - HAC inference - "
            "random-portfolio null - survivorship caveat\n\n"
            + VERDICT_BADGES +
            "The deep companion to [01_for_the_curious.ipynb](01_for_the_curious.ipynb) -- "
            "same seven beats, every claim now carrying its standard error. We test the "
            "asset-growth anomaly on a survivorship-biased S&P 500 EDGAR panel (~419 tickers, "
            "2009-2025), measure the low-minus-high-AG hedge with a HAC t-stat, and pin "
            "it against a random-portfolio null distribution.\n\n"
            "> **Not investment advice.** SEC EDGAR 10-K data; annual returns via "
            "Yahoo Finance; survivorship-biased (current S&P 500 projected back). "
            "Sources in [`docs/references.md`](../docs/references.md), reproducible "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Survivorship-bias warning**: all results are conditional on survival. "
            "Fast-growing firms that failed between 2009 and now are excluded -- "
            "selection reverses the predicted sign."
        ),
        code(BOOT + "\n" + R_CODE + "\nimport sys; sys.path.insert(0, os.path.abspath('../../..'))"),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Hedge **{R['hedge_mean']:+.1f}%/yr**, "
            f"HAC *t* = **{R['hedge_t']:+.2f}** -- wrong direction, zero statistical "
            f"significance. The predicted effect (high AG → low returns) is absent. |\n"
            f"| **Tradability** | `MIRAGE` | No edge to trade. Long-low-AG/short-high-AG "
            f"produces negative returns on this panel. |\n"
            f"| **Survivorship** | `SELECTION DOMINATES` | Fast-growing S&P 500 survivors "
            f"= tech giants. Failed high-growers absent. Selection reverses the predicted "
            f"direction. |\n\n"
            "> **In plain words:** Cooper et al. (2008) found a real anomaly on a broad "
            "US universe. On survivorship-biased S&P 500 large caps, the predicted effect "
            "vanishes -- high-AG tech giants dominate the fast-growth quintile and earn "
            "higher returns, reversing the sign."
        ),

        # ---- BEAT 1 -----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $r_{i,t+1}$ be firm $i$'s return in year $t+1$. Define:\n\n"
            "$$\\text{AG}_{i,t} = \\frac{A_{i,t} - A_{i,t-1}}{A_{i,t-1}}$$\n\n"
            "where $A$ = total assets.\n\n"
            "The **H1 (signal)** claim (Cooper et al. 2008) is:\n\n"
            "$$\\mathbb{E}\\left[\\bar{r}_{\\text{low-AG},t+1} - "
            "\\bar{r}_{\\text{high-AG},t+1}\\right] > 0$$\n\n"
            "with a HAC t-stat |t| >= 2. **H2 (beats random)** requires the low-AG "
            "quintile to beat a random same-size draw. **H3 (tradable)** requires the "
            "hedge to survive costs and scale."
        ),

        # ---- BEAT 2 -----------------------------------------------------------
        md(
            "## 2 - So what? -- the stakes\n\n"
            "H1 is *not* supported on this biased panel -- the direction is inverted. "
            "The academic evidence for the effect in broad cross-sections of US stocks "
            "is credible (Cooper et al. 2008; the Fama-French CMA factor); on S&P 500 "
            "large-cap survivors, survivorship selection dominates the signal."
        ),

        # ---- BEAT 3 -----------------------------------------------------------
        md(
            "## 3 - The protocol\n\n"
            "- **Data**: EDGAR Assets from the desk's shared cache (`_edgar_Assets.parquet`); "
            "annual returns from `_edgar_yrret.parquet`.\n"
            f"- **Universe**: current S&P 500 members with Assets present "
            f"(~{R['n_tickers']} tickers); survivorship-biased.\n"
            f"- **Window**: {R['n_years']} years of signal ({R['start_year']}-{R['end_year']}), "
            f"forecasting returns {R['start_year']+1}-{R['end_year']+1}.\n"
            "- **Signal**: YoY total-asset growth rate (higher = more aggressive expansion).\n"
            "- **Sort**: quintile (q=0.20); Q1 = low AG (disciplined), Q5 = high AG (fast growers).\n"
            "- **Inference**: HAC Newey-West t-stat on the 17-observation annual hedge series.\n"
            "- **Control**: empirical distribution of random same-size portfolios "
            "(500 draws x 17 years).\n"
            "- **Positive control**: synthetic panel with planted AG premium confirms the "
            "engine can detect a real effect."
        ),

        # ---- BEAT 4 -----------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - Synthetic positive control -- the engine works when an effect is planted\n\n"
            "Sweep the planted AG premium from 0 to -12%/yr per unit z-rank and confirm "
            "the hedge tracks it monotonically."
        ),
        code(
            "premiums = [0.0, -0.04, -0.08, -0.12]\n"
            "synth_rows = []\n"
            "for p in premiums:\n"
            "    s, f, _ = data.synthetic_panel(n_firms=300, n_years=25, premium=p, seed=244)\n"
            "    hh = st.quintile_returns(s, f, q=0.20)\n"
            "    ss = st.summary(hh['hedge'])\n"
            "    synth_rows.append((abs(p)*100, ss['mean']*100, ss['tstat']))\n"
            "sydf = pd.DataFrame(synth_rows, columns=['premium_%', 'hedge_%/yr', 'HAC_t'])\n"
            "fig, ax = plt.subplots(figsize=(9, 4))\n"
            "colors = [GREEN if t > 2 else (RED if t < -2 else GREY) for t in sydf['HAC_t']]\n"
            "ax.bar(sydf['premium_%'], sydf['hedge_%/yr'], color=colors, width=1.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Planted AG effect (%/yr per unit z-rank)')\n"
            "ax.set_ylabel('Low-minus-high hedge (%/yr)')\n"
            "ax.set_title('Positive control: HAC t>2 (green) appears where the premium is real')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(sydf.round(2))"
        ),
        md(
            "> The engine faithfully detects planted premiums. The real-data result is "
            "therefore a statement about the **market and survivorship bias**, not the "
            "**method**."
        ),
        md(
            "### 4b - Real EDGAR panel -- quintile returns and HAC inference"
        ),
        code(
            "if HAVE_REAL:\n"
            "    q_means = [float(h[f'q{i}'].mean()*100) for i in range(1,6)]\n"
            "    q_tstats = [st.summary(h[f'q{i}'])['tstat'] for i in range(1,6)]\n"
            "    s_hedge = st.summary(h['hedge'])\n"
            "    s_lo_ex = st.summary(h['q1'] - h['market'])\n"
            "    s_hi_ex = st.summary(h['q5'] - h['market'])\n"
            "else:\n"
            "    q_means = [R['lo_mean'], 16.5, 15.0, 17.0, R['hi_mean']]\n"
            "    q_tstats = [R['lo_t'], 8.39, 9.31, 7.90, R['hi_t']]\n"
            "    s_hedge = dict(mean=R['hedge_mean']/100, tstat=R['hedge_t'], hit_rate=R['hedge_hit']/100)\n"
            "    s_lo_ex = dict(mean=R['lo_excess_mean']/100, tstat=R['lo_excess_t'])\n"
            "    s_hi_ex = dict(mean=R['hi_excess_mean']/100, tstat=R['hi_excess_t'])\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "x = [1, 2, 3, 4, 5]\n"
            "# High-AG (Q5) outperforms -> inverted, so use RED for Q1 and GREEN for Q5\n"
            "bar_colors = [RED, GREY, GREY, GREY, GREEN]\n"
            "axes[0].bar(x, q_means, color=bar_colors)\n"
            "axes[0].set_xticks(x); axes[0].set_xticklabels(['Q1 (lo)', 'Q2', 'Q3', 'Q4', 'Q5 (hi)'])\n"
            "axes[0].set_ylabel('Mean annual return (%/yr)')\n"
            "axes[0].set_title('Quintile returns: HIGH-AG outperforms (inverted vs claim!)')\n"
            "axes[1].bar(x, q_tstats, color=bar_colors)\n"
            "for s in (2, -2): axes[1].axhline(s, ls='--', c=GREY, lw=1)\n"
            "axes[1].set_xticks(x); axes[1].set_xticklabels(['Q1 (lo)', 'Q2', 'Q3', 'Q4', 'Q5 (hi)'])\n"
            "axes[1].set_ylabel('HAC t-stat')\n"
            "axes[1].set_title('All quintiles earn positive returns (survivor bull market)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Hedge: {s_hedge[\"mean\"]*100:+.1f}%/yr  HAC t={s_hedge[\"tstat\"]:+.2f}  hit={s_hedge[\"hit_rate\"]:.0%}')\n"
            "print(f'Low-AG excess: {s_lo_ex[\"mean\"]*100:+.1f}%/yr  t={s_lo_ex[\"tstat\"]:+.2f}')\n"
            "print(f'High-AG excess: {s_hi_ex[\"mean\"]*100:+.1f}%/yr  t={s_hi_ex[\"tstat\"]:+.2f}')"
        ),
        md(
            f"> The low-minus-high-AG hedge is **{R['hedge_mean']:+.1f}%/yr**, "
            f"HAC *t* = **{R['hedge_t']:+.2f}** -- statistically zero. "
            f"Note the direction: **high-AG firms earn {R['hi_mean']:.1f}%/yr** vs "
            f"**low-AG at {R['lo_mean']:.1f}%/yr**. The sign is inverted relative to "
            f"the academic claim. High-AG survivors = successful technology expanders "
            f"(FAANG, semiconductors, cloud computing)."
        ),
        md(
            "### 4c - Random-portfolio null distribution\n\n"
            "Does the low-AG quintile beat a blind draw of the same number of stocks? "
            "We simulate 500 random same-size portfolios each year."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rand = st.random_portfolio_returns(sig, fwd, q=0.20, n_draws=500, seed=244)\n"
            "    actual_excess = float((h['q1'] - h['market']).mean())\n"
            "    pct_beaten = float((rand < actual_excess).mean())\n"
            "else:\n"
            "    rng_nb = np.random.default_rng(42)\n"
            "    rand = pd.Series(rng_nb.normal(0.0, R['rand_std']/100, 17*500))\n"
            "    actual_excess = R['lo_excess_mean'] / 100\n"
            "    pct_beaten = R['rand_pct_beaten'] / 100\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "ax.hist(rand * 100, bins=50, color=GREY, alpha=0.7, label='Random portfolios (500x17yr avg)')\n"
            "ax.axvline(actual_excess * 100, c=RED, lw=2.5,\n"
            "           label=f'Low-AG quintile: {actual_excess*100:+.1f}%/yr excess')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Annual excess return vs EW market (%)')\n"
            "ax.set_ylabel('Count'); ax.legend()\n"
            "ax.set_title(f'Low-AG beats {pct_beaten:.0%} of random draws -- close to chance level')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Low-AG portfolio beats {pct_beaten:.0%} of random same-size draws')"
        ),
        md(
            f"> The low-AG portfolio beats **{R['rand_pct_beaten']}%** of random same-size "
            "draws -- barely above the 50% chance level. The low-AG quintile provides "
            "essentially no alpha over a random draw of the same number of stocks."
        ),

        # ---- BEAT 5 -----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal `NONE`** -- hedge {R['hedge_mean']:+.1f}%/yr, HAC *t* = "
            f"{R['hedge_t']:+.2f}. Wrong direction, zero significance. The predicted "
            "effect (high AG → low returns) is absent on this panel.\n"
            f"- **Tradability `MIRAGE`** -- no signal to trade. Short-high-AG on S&P 500 "
            "survivors means shorting Apple, Nvidia, and Amazon through their growth phases.\n"
            "- **Survivorship `SELECTION DOMINATES`** -- fast-growing S&P 500 survivors "
            "skew heavily toward mega-cap tech. The selection reverses the predicted sign."
        ),

        # ---- BEAT 6 -----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- the honest answer\n\n"
            "No tradable signal exists on this panel. The obstacles:\n\n"
            "1. **Sign inversion.** The hedge earns −1.0%/yr with t = −0.45 -- the "
            "signal direction is the opposite of the academic claim on this survivor panel.\n"
            "2. **Survivorship selection dominates.** The S&P 500 includes Apple, Amazon, "
            "Nvidia, and Google -- all fast-growing and all extraordinary performers. "
            "These companies are in the high-AG quintile. Their growth was productive and "
            "shareholders benefited. The theoretical over-investment mechanism does not "
            "apply to them.\n"
            "3. **The academic evidence is for broad universes.** Cooper et al. (2008) "
            "tested on all NYSE/AMEX/NASDAQ stocks. The CMA factor is estimated on "
            "broad cross-sections. On S&P 500 blue-chips, mispricing is quickly "
            "arbitraged away and productivity of investment varies.\n\n"
            "The correct conclusion: **MIRAGE** on this test. The academic anomaly may "
            "be real in its original universe -- it simply cannot be tested here."
        ),

        # ---- BEAT 7 -----------------------------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The right universe.** Cooper et al. (2008) used NYSE/AMEX/NASDAQ "
            "stocks above $5M market cap from 1968 to 2003. A point-in-time all-US-stock "
            "universe (Compustat-based) is the minimum required for a credible test.\n"
            "- **CMA factor.** The Fama-French five-factor model (2015) includes CMA "
            "(Conservative Minus Aggressive investment). This factor has positive expected "
            "returns in factor regressions on broad cross-sections. It does not imply "
            "the effect is present on a narrow survivor panel.\n"
            "- **NOA cousin.** [Study 153 — Net-Operating-Assets](../../153-net-operating-assets/) "
            "tests the scaled-ratio version (NOA) of the same balance-sheet bloat idea "
            "and finds Weak/Fragile on the same survivor panel -- marginally stronger "
            "because the NOA ratio filters more noise. The comparison is instructive.\n\n"
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
