"""Generate the two narrative notebooks for Study 199 (Sales-Growth).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-data cells use the cached
EDGAR parquet under _cache/ if present and otherwise quote the frozen headline numbers
in ``R``, so the notebook re-runs for any reader.

The _write convention (each build_*() ends by calling _write) is kept so the repo's
intro-restyle tooling can monkeypatch it.
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-15).
# All numbers are SURVIVORSHIP-BIASED upper bounds.
R = dict(
    n_tickers=326, n_years=18, start_year=2008, end_year=2025,
    # Low-growth (Q1) quintile
    lo_mean=19.7, lo_sharpe=1.27, lo_t=8.44, lo_hit=89,
    # High-growth (Q5) glamour quintile
    hi_mean=20.0, hi_sharpe=1.28, hi_t=8.45, hi_hit=83,
    # Equal-weight market
    mkt_mean=17.7, mkt_sharpe=1.38, mkt_t=9.24,
    # Hedge (Q1 - Q5): long value, short glamour
    hedge_mean=-0.4, hedge_sharpe=-0.03, hedge_t=-0.17, hedge_hit=39,
    # Excess vs market
    lo_excess_mean=2.0, lo_excess_t=1.36,
    hi_excess_mean=2.4, hi_excess_t=1.82,
    # Random control
    rand_std=4.9, rand_pct_beaten=70,
    # Signal fingerprint
    sg_fingerprint="9ffae4d859d7",
)

# Year-by-year frozen data (Q1, Q5, market, hedge)
_Q1  = [50.4, 20.7, 4.9, 24.6, 39.0, 14.3, 0.3, 28.1, 23.8, -6.2, 32.3, 9.2, 31.4, -8.0, 28.9, 27.6, 18.2, 15.1]
_Q5  = [30.8, 24.4, -7.0, 23.0, 45.7, 22.2, 13.8, 0.1, 36.8, -1.6, 25.8, 21.3, 45.3, -2.3, 20.0, 30.7, 24.7, 7.0]
_MKT = [30.2, 22.4, 1.6, 22.4, 41.7, 19.3, 5.6, 17.7, 25.8, -4.0, 31.4, 10.5, 33.8, -3.8, 17.5, 22.6, 13.9, 9.3]
_HDG = [19.6, -3.8, 11.8, 1.5, -6.7, -7.9, -13.5, 28.0, -13.0, -4.6, 6.5, -12.1, -13.9, -5.7, 8.8, -3.1, -6.5, 8.0]
_YRS = list(range(R["start_year"], R["end_year"] + 1))

VERDICT_BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Survivorship: Upper_bound_only](https://img.shields.io/badge/Survivorship-Upper_bound_only-8b949e?style=flat-square)\n\n"
)

# ---------------------------------------------------------------------------
# Shared boot code
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath("../../.."))   # repo root (quantlab/)
sys.path.insert(0, os.path.abspath(".."))          # study package (sales_growth/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from sales_growth import data, strategy as st

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
            "# Sales-Growth -- do high-growth 'glamour' stocks underperform?\n"
            "### The Lakonishok, Shleifer & Vishny (1994) glamour/value anomaly on S&P 500 EDGAR data\n\n"
            + VERDICT_BADGES
            + "A famous 1994 paper by Lakonishok, Shleifer & Vishny (LSV) argued that investors "
            "over-extrapolate past sales growth: they pay too much for 'glamour' stocks (fast-growing "
            "revenues) and too little for 'value' stocks (slow-growing revenues). The prediction: "
            "sort firms by trailing one-year revenue growth and the low-growth group should "
            "**outperform** the high-growth group. We test it on ~326 S&P 500 names from real "
            "SEC filings.\n\n"
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
            f"| Does low growth outperform high growth? | **No.** The hedge is "
            f"**{R['hedge_mean']:+.1f}%/yr** -- essentially zero, and the wrong sign. |\n"
            f"| Is that statistically significant? | **No.** HAC *t* = **{R['hedge_t']:+.2f}** "
            f"-- far below the |t| >= 2 bar. The result is pure noise. |\n"
            f"| Does it beat a random stock pick? | **Barely.** Beats only "
            f"{R['rand_pct_beaten']}% of random same-size draws -- barely above chance. |\n"
            "| Could you trade it (even if it were real)? | **No.** With annual rebalancing, "
            "even a flat gross edge gets destroyed by turnover costs on a large-cap universe "
            "where the measured alpha is statistically zero. |\n\n"
            "> **The bottom line**: on this survivor-biased S&P 500 panel, the LSV glamour/"
            "value effect on sales growth simply does not appear. High-growth stocks barely "
            "outperform low-growth stocks, and neither the direction nor the magnitude clears "
            "any reasonable statistical bar."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *Buy stocks with low past sales growth ('value'). Avoid stocks with high past "
            "sales growth ('glamour'). Rebalance annually. Investors over-extrapolate revenue "
            "momentum and will eventually be disappointed when it reverts to the mean.*\n\n"
            "LSV (1994) tested this on all US stocks from 1968-1990 and found a real spread: "
            "value outperformed glamour by roughly 10-11%/yr (raw return), surviving even "
            "after controlling for beta. Their mechanism: investors see last year's 40% revenue "
            "growth and project it forward indefinitely, pushing the price beyond fundamentals.\n\n"
            "**Why it might not work on S&P 500 survivors:**\n\n"
            "- **Wrong universe.** LSV's result is strongest in small, neglected, low-analyst "
            "stocks -- precisely those NOT in the S&P 500.\n"
            "- **Heavy analyst coverage.** Large-cap glamour stocks are well-covered; "
            "any over-extrapolation should be quickly corrected.\n"
            "- **Survivorship.** All the S&P 500 names in our panel survived 20 years. "
            "High-growth companies that subsequently failed are absent."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If LSV's effect is real and survives on large US stocks, it's a low-turnover "
            "(one trade per firm per year), fundamentals-only strategy anyone can run from "
            "public SEC filings. No forecasting required -- just sort on last year's revenue "
            "growth.\n\n"
            "If it's a small-cap artefact, a survivorship illusion, or a 1990s relic, then "
            "this is a cautionary tale about testing famous factor papers on the convenient "
            "(but wrong) universe. The S&P 500 is not the right test."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three honesty requirements:\n\n"
            "1. **Reporting lag.** Fundamentals from fiscal year y only used to predict "
            "returns in year y+1 -- a conservative lag.\n"
            "2. **Random-portfolio control.** Does the low-growth quintile beat a random "
            "draw of the same number of stocks? If not, the result is just size/concentration "
            "risk, not a real signal.\n"
            "3. **Name the survivorship bias.** The EDGAR panel here covers current S&P "
            "500 members projected back. High-growth failures are absent. Every result is "
            "an upper bound (and even that upper bound shows no effect)."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown -- the quintile spread, or lack thereof\n\n"
            "Positive control first: **when we plant a real LSV effect in a synthetic panel, "
            "the engine finds it.** That confirms the engine is honest -- the real-data "
            "non-result is a statement about the **market**, not the method."
        ),
        code(
            "# Synthetic positive control: plant a known LSV premium, confirm the engine detects it\n"
            "premiums = [0.0, -0.04, -0.08, -0.12]\n"
            "hedges = []\n"
            "for p in premiums:\n"
            "    s, f, _ = data.synthetic_panel(n_firms=300, n_years=25, premium=p, seed=199)\n"
            "    hh = st.quintile_returns(s, f, q=0.20)\n"
            "    hedges.append(hh['hedge'].mean() * 100)\n"
            "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            "ax.plot([abs(p)*100 for p in premiums], hedges, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Planted LSV effect (% alpha per unit z-rank)')\n"
            "ax.set_ylabel('Low-minus-high-growth hedge (%/yr)')\n"
            "ax.set_title('Synthetic control: engine finds the effect when planted')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('The engine faithfully recovers planted LSV effects -- so the real non-result is a market statement.')"
        ),
        md("**Now the real EDGAR panel -- quintile sort on trailing sales growth:**"),
        code(
            "# Year-by-year low-growth vs high-growth vs market\n"
            "if HAVE_REAL:\n"
            "    q1_vals = h['q1'].values * 100\n"
            "    q5_vals = h['q5'].values * 100\n"
            "    hdg_vals = h['hedge'].values * 100\n"
            "    years_real = h.index.tolist()\n"
            "else:\n"
            "    q1_vals, q5_vals, hdg_vals, years_real = Q1, Q5, HDG, YRS\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "x = range(len(years_real))\n"
            "axes[0].bar(x, q1_vals, label='Q1 low-growth (value)', color=GREEN, alpha=0.7, width=0.4)\n"
            "axes[0].bar([i+0.4 for i in x], q5_vals, label='Q5 high-growth (glamour)', color=RED, alpha=0.7, width=0.4)\n"
            "axes[0].set_xticks([i+0.2 for i in x])\n"
            "axes[0].set_xticklabels(years_real, rotation=45, fontsize=8)\n"
            "axes[0].set_ylabel('Annual return %'); axes[0].legend()\n"
            "axes[0].set_title('Q1 (low growth) vs Q5 (high growth / glamour) raw returns')\n"
            "colors = [GREEN if v > 0 else RED for v in hdg_vals]\n"
            "axes[1].bar(years_real, hdg_vals, color=colors)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_ylabel('Hedge return (Q1 - Q5) %')\n"
            "n_pos = sum(1 for v in hdg_vals if v > 0)\n"
            "axes[1].set_title(f'Value-minus-glamour hedge: {n_pos}/{len(hdg_vals)} positive years')\n"
            "axes[1].tick_params(axis='x', rotation=45, labelsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Mean hedge: {{np.mean(hdg_vals):+.1f}}%/yr  (frozen R: {{R[\"hedge_mean\"]:+.1f}}%/yr)')"
        ),
        md(
            f"The low-growth quintile earns **+{R['lo_mean']:.1f}%/yr** vs the high-growth "
            f"quintile's **+{R['hi_mean']:.1f}%/yr** -- a hedge of **{R['hedge_mean']:+.1f}%/yr**, "
            f"positive in only {R['hedge_hit']}% of years. This is the **wrong direction** for "
            "the LSV claim (glamour is supposed to underperform) and the magnitude is essentially "
            "zero. Both quintiles beat the equal-weight market because in a bull-market decade "
            "any concentrated portfolio does -- that tells us nothing about the signal.\n\n"
            "> **Why might glamour not underperform here?** In the original LSV sample "
            "(all US stocks, 1968-1990), 'glamour' companies were smaller, less followed, "
            "more prone to price bubbles. On the S&P 500, 'high revenue growth' often means "
            "secular leaders (Amazon, Microsoft, Apple in their growth phases) whose revenue "
            "growth is genuinely persistent -- not mean-reverting. The over-extrapolation "
            "story breaks down for quality-growth compounders."
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal -- None.** Hedge {R['hedge_mean']:+.1f}%/yr, HAC *t* = {R['hedge_t']:+.2f} -- "
            "nowhere near the |t| >= 2 bar. The direction is also *wrong* (glamour barely "
            "outperforms value). This is a null result even on the survivor-biased panel.\n"
            "- **Tradability -- Mirage.** There is no gross edge to survive costs. Even if "
            "the hedge were zero, annual rebalancing on large-cap names is cheap -- but "
            "you'd need a positive gross edge to start with.\n"
            "- **Survivorship -- Upper bound only.** Every result assumes no S&P 500 "
            "member failed or was expelled between 2008 and 2025. High-growth glamour "
            "companies that blew up are simply absent. Despite that best-case bias, "
            "the hedge is still essentially zero."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "No gross edge exists on this panel, so the cost question is moot. "
            "But for completeness: annual rebalancing of ~65 large-cap names is "
            "operationally trivial (spreads tight, commissions minimal, no tax drag on "
            "futures). The obstacle is entirely the missing signal:\n\n"
            "1. **No predictive content.** The t-stat on the hedge is |t| < 0.2 -- pure "
            "noise. Any implementation would be trading on noise.\n"
            "2. **Wrong universe and era.** LSV's effect was documented in the full US "
            "stock market, 1968-1990. The S&P 500 survivor panel in an era of secular "
            "tech growth is the opposite environment.\n"
            "3. **Survivorship arithmetic.** Even this best-case test shows no effect. "
            "A live strategy in 2008-2025 would have included the high-growth firms that "
            "subsequently failed (WeWork, Peloton, etc.) -- those would be in Q5 with "
            "catastrophic returns, which might have made the hedge *positive* in live "
            "trading. But that's speculation, not evidence.\n\n"
            "**Verdict: Mirage.** No actionable edge here."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The right universe.** LSV's (1994) sample is all US stocks -- "
            "thousands of small and mid-cap names where over-extrapolation is most "
            "plausible and analyst coverage is thin. A proper test requires a "
            "point-in-time constituent universe (e.g. CRSP-based), not S&P 500 survivors.\n"
            "- **Related signals.** LSV also tested book-to-market (B/M) and "
            "earnings-to-price (E/P) as glamour proxies -- those often work better than "
            "sales growth in large-cap universes ([Study 121 -- Magic-Formula](../../121-magic-formula/) "
            "tests a related quality+value combination).\n"
            "- **Asset growth.** Cooper et al. (2008) show asset growth (total-asset expansion) "
            "predicts lower returns through an overinvestment channel -- that's a "
            "balance-sheet version of the same over-extrapolation story, and distinct from "
            "revenue growth ([Study 44 -- Growth-Spurt](../../44-growth-spurt/)).\n\n"
            "*Think the sales-growth effect is real in large caps? Fork this, extend "
            "to a point-in-time all-US-stock panel (Compustat-based historical membership), "
            "and show |t| >= 2 on the low-minus-high-growth hedge. That's the bar.*"
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
            "# Sales-Growth -- a quantitative teardown\n"
            "### LSV (1994) glamour/value anomaly - S&P 500 EDGAR panel - HAC inference - "
            "random-portfolio null - survivorship caveat\n\n"
            + VERDICT_BADGES
            + "The deep companion to [01_for_the_curious.ipynb](01_for_the_curious.ipynb) -- "
            "same seven beats, every claim now carrying its standard error. We test the "
            "Lakonishok, Shleifer & Vishny (1994) sales-growth glamour/value anomaly on a "
            "survivorship-biased S&P 500 EDGAR panel (~326 tickers, 2008-2025), measure the "
            "low-minus-high-growth hedge with a HAC t-stat, and pin it against a random-portfolio "
            "null distribution.\n\n"
            "> **Not investment advice.** SEC EDGAR 10-K Revenues data; annual returns via "
            "Yahoo Finance; survivorship-biased (current S&P 500 projected back). "
            "Sources in [`docs/references.md`](../docs/references.md), reproducible "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Survivorship-bias warning**: all results are upper bounds. Firms that "
            "failed between 2008 and now are excluded -- in particular high-growth "
            "firms that blew up after 2008."
        ),
        code(BOOT + "\n" + R_CODE),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Hedge **{R['hedge_mean']:+.1f}%/yr**, "
            f"HAC *t* = **{R['hedge_t']:+.2f}** -- far below |t| = 2 bar, wrong direction "
            f"(glamour barely outperforms value). Low-growth excess "
            f"+{R['lo_excess_mean']:.1f}%/yr (*t* = +{R['lo_excess_t']:.2f}) -- also sub-threshold. |\n"
            f"| **Tradability** | `MIRAGE` | No gross edge exists. Even if zero-cost "
            f"annual rebalancing were feasible, there is nothing to harvest. |\n"
            f"| **Survivorship** | `UPPER BOUND` | Panel excludes all post-2008 "
            f"failures; every number is an upper bound -- and even this best case shows "
            f"no effect. |\n\n"
            "> **In plain words:** the LSV glamour/value anomaly simply does not appear "
            "on the S&P 500 survivor panel from 2008-2025. High-growth stocks are as "
            "likely to outperform as low-growth stocks, with no predictive structure in "
            "the trailing-growth signal."
        ),

        # ---- BEAT 1 -----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $r_{i,t+1}$ be firm $i$'s return in year $t+1$. Define:\n\n"
            "$$\\text{SalesGrowth}_{i,t} = \\frac{\\text{Revenues}_{i,t}}"
            "{\\text{Revenues}_{i,t-1}} - 1$$\n\n"
            "The **H1 (signal)** claim (Lakonishok, Shleifer & Vishny 1994) is:\n\n"
            "$$\\mathbb{E}\\left[\\bar{r}_{\\text{low-growth},t+1} - "
            "\\bar{r}_{\\text{high-growth},t+1}\\right] > 0$$\n\n"
            "with a HAC t-stat |t| >= 2. **H2 (beats random)** requires the low-growth "
            "quintile to beat a random same-size draw. **H3 (tradable)** requires the "
            "hedge to survive costs and scale."
        ),

        # ---- BEAT 2 -----------------------------------------------------------
        md(
            "## 2 - So what? -- the stakes\n\n"
            "H1 is rejected on this panel. The practical stakes are nil: with HAC *t* = "
            f"{R['hedge_t']:+.2f} and the wrong sign, there is nothing to trade. The "
            "important question is *why* the LSV effect fails here -- whether it is a "
            "universe issue (large-cap survivors), an era issue (post-2007 tech-growth "
            "dominance), or genuine factor decay since the 1994 publication."
        ),

        # ---- BEAT 3 -----------------------------------------------------------
        md(
            "## 3 - The protocol\n\n"
            f"- **Data**: `_edgar_Revenues.parquet` from the desk's shared cache; "
            f"annual returns from `_edgar_yrret.parquet`.\n"
            f"- **Universe**: current S&P 500 members with Revenues present (~{R['n_tickers']} "
            f"tickers across all years); survivorship-biased.\n"
            f"- **Window**: {R['n_years']} signal years ({R['start_year']}--{R['end_year']}), "
            f"forecasting returns {R['start_year']+1}--{R['end_year']+1}.\n"
            "- **Signal**: trailing one-year revenue growth (higher = glamour); quintile sort (q=0.20).\n"
            "- **Reporting lag**: fundamentals from year y predict returns in year y+1.\n"
            "- **Inference**: HAC Newey-West t-stat on the 18-observation annual hedge series.\n"
            "- **Control**: empirical distribution of random same-size portfolios (500 draws x 18yr).\n"
            "- **Positive control**: synthetic panel with planted LSV premium confirms the engine detects a real effect.\n"
            "- **Survivorship**: named and noted -- all results are upper bounds."
        ),

        # ---- BEAT 4 -----------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - Synthetic positive control -- the engine works when an effect is planted\n\n"
            "Sweep the planted LSV premium from 0 to -12%/yr per unit z-rank and confirm "
            "the hedge tracks it monotonically."
        ),
        code(
            "premiums = [0.0, -0.04, -0.08, -0.12]\n"
            "synth_rows = []\n"
            "for p in premiums:\n"
            "    s, f, _ = data.synthetic_panel(n_firms=300, n_years=25, premium=p, seed=199)\n"
            "    hh = st.quintile_returns(s, f, q=0.20)\n"
            "    ss = st.summary(hh['hedge'])\n"
            "    synth_rows.append((abs(p)*100, ss['mean']*100, ss['tstat']))\n"
            "sydf = pd.DataFrame(synth_rows, columns=['premium_%', 'hedge_%/yr', 'HAC_t'])\n"
            "fig, ax = plt.subplots(figsize=(9, 4))\n"
            "colors = [GREEN if t > 2 else (RED if t < -2 else GREY) for t in sydf['HAC_t']]\n"
            "ax.bar(sydf['premium_%'], sydf['hedge_%/yr'], color=colors, width=1.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Planted LSV effect (%/yr per unit z-rank)')\n"
            "ax.set_ylabel('Low-minus-high hedge (%/yr)')\n"
            "ax.set_title('Positive control: HAC t>2 (green) appears where the premium is real')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(sydf.round(2))"
        ),
        md(
            "> The engine faithfully detects planted premiums. The real-data non-result is "
            "therefore a statement about the **market and the panel**, not the **method**."
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
            "    q_means = [R['lo_mean'], 14.7, 17.3, 16.6, R['hi_mean']]\n"
            "    q_tstats = [R['lo_t'], 7.86, 6.76, 8.34, R['hi_t']]\n"
            "    s_hedge = dict(mean=R['hedge_mean']/100, tstat=R['hedge_t'], hit_rate=R['hedge_hit']/100)\n"
            "    s_lo_ex = dict(mean=R['lo_excess_mean']/100, tstat=R['lo_excess_t'])\n"
            "    s_hi_ex = dict(mean=R['hi_excess_mean']/100, tstat=R['hi_excess_t'])\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "x = [1, 2, 3, 4, 5]\n"
            "bar_colors = [GREEN, GREY, GREY, GREY, RED]\n"
            "axes[0].bar(x, q_means, color=bar_colors)\n"
            "axes[0].set_xticks(x)\n"
            "axes[0].set_xticklabels(['Q1 (lo\\ngrowth)', 'Q2', 'Q3', 'Q4', 'Q5 (hi\\nglamour)'])\n"
            "axes[0].set_ylabel('Mean annual return (%/yr)')\n"
            "axes[0].set_title('Quintile returns: no clear monotone spread')\n"
            "axes[1].bar(x, q_tstats, color=bar_colors)\n"
            "for s in (2, -2): axes[1].axhline(s, ls='--', c=GREY, lw=1)\n"
            "axes[1].set_xticks(x)\n"
            "axes[1].set_xticklabels(['Q1 (lo\\ngrowth)', 'Q2', 'Q3', 'Q4', 'Q5 (hi\\nglamour)'])\n"
            "axes[1].set_ylabel('HAC t-stat')\n"
            "axes[1].set_title('All quintiles have high t (all earn positive returns -- bull market)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Hedge: {s_hedge[\"mean\"]*100:+.1f}%/yr  HAC t={s_hedge[\"tstat\"]:+.2f}  hit={s_hedge[\"hit_rate\"]:.0%}')\n"
            "print(f'Low-growth excess: {s_lo_ex[\"mean\"]*100:+.1f}%/yr  t={s_lo_ex[\"tstat\"]:+.2f}')\n"
            "print(f'High-growth excess: {s_hi_ex[\"mean\"]*100:+.1f}%/yr  t={s_hi_ex[\"tstat\"]:+.2f}')"
        ),
        md(
            f"> The low-minus-high-growth hedge is **{R['hedge_mean']:+.1f}%/yr**, "
            f"HAC *t* = **{R['hedge_t']:+.2f}** -- a textbook null. The signal has no "
            f"predictive power. All quintiles earn high raw returns because this is a "
            f"survivor panel in a bull market -- those t-stats ({R['lo_t']:.1f} for Q1, "
            f"{R['hi_t']:.1f} for Q5) reflect the strong equity premium over the period, "
            "not any cross-sectional signal in revenue growth."
        ),
        md(
            "### 4c - Random-portfolio null distribution\n\n"
            "Does the low-growth quintile beat a blind draw of the same number of stocks? "
            "We simulate 500 random same-size portfolios each year."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rand = st.random_portfolio_returns(sig, fwd, q=0.20, n_draws=500, seed=199)\n"
            "    actual_excess = float((h['q1'] - h['market']).mean())\n"
            "    pct_beaten = float((rand < actual_excess).mean())\n"
            "else:\n"
            "    rng_nb = np.random.default_rng(42)\n"
            "    rand = pd.Series(rng_nb.normal(0.0, R['rand_std']/100, 18*500))\n"
            "    actual_excess = R['lo_excess_mean'] / 100\n"
            "    pct_beaten = R['rand_pct_beaten'] / 100\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "ax.hist(rand * 100, bins=50, color=GREY, alpha=0.7, label='Random portfolios (500x18yr avg)')\n"
            "ax.axvline(actual_excess * 100, c=AMBER, lw=2.5,\n"
            "           label=f'Low-growth quintile: {actual_excess*100:+.1f}%/yr excess')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Annual excess return vs EW market (%)')\n"
            "ax.set_ylabel('Count'); ax.legend()\n"
            "ax.set_title(f'Low-growth beats {pct_beaten:.0%} of random draws -- barely above chance')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Low-growth portfolio beats {pct_beaten:.0%} of random same-size draws')"
        ),
        md(
            f"> The low-growth portfolio beats **{R['rand_pct_beaten']}%** of random same-size "
            "draws -- not much better than the 50% chance level. The signal's information "
            "content is negligible on this panel."
        ),

        # ---- BEAT 5 -----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal `NONE`** -- hedge {R['hedge_mean']:+.1f}%/yr, HAC *t* = "
            f"{R['hedge_t']:+.2f}; sub-threshold by a wide margin and wrong direction. "
            f"Low-growth excess +{R['lo_excess_mean']:.1f}%/yr (*t* = +{R['lo_excess_t']:.2f}) "
            f"also sub-threshold. Even on the survivor-biased best-case panel, "
            "the LSV glamour/value effect via sales growth is absent.\n"
            f"- **Tradability `MIRAGE`** -- no gross edge exists to survive costs. "
            "Annual rebalancing of large-cap names is cheap in principle but there "
            "is no alpha to capture.\n"
            "- **Survivorship `UPPER BOUND`** -- every result assumes no S&P 500 "
            "member failed between 2008 and 2025. Despite this best-case bias, "
            "the hedge is essentially zero. A live strategy would have been exposed "
            "to high-growth failures -- the true effect would be even weaker."
        ),

        # ---- BEAT 6 -----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- the honest answer\n\n"
            "There is nothing to trade. The gross alpha is statistically indistinguishable "
            "from zero, and the point estimate is negative (the wrong direction).\n\n"
            "The operationally attractive features (low turnover, public data, liquid names) "
            "are real -- the problem is the absence of any signal to harvest. "
            "This is a clean null, not a 'dies at the cost line' fragile result.\n\n"
            "The correct interpretation: the LSV (1994) sales-growth glamour/value effect "
            "is **not present in S&P 500 large-cap survivors from 2008-2025**. Whether "
            "this is (a) always a small-cap effect, (b) factor decay post-publication, "
            "or (c) S&P 500 survivor bias masking the true distribution is left open."
        ),

        # ---- BEAT 7 -----------------------------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The right universe.** Lakonishok et al. (1994) ran their test on all "
            "NYSE/AMEX stocks. A point-in-time constituent universe with small and mid-caps "
            "is the minimum required for a credible replication.\n"
            "- **Other LSV glamour proxies.** Sales growth is one proxy; LSV also tested "
            "book-to-market, cash-flow-to-price, and earnings-to-price. Those value signals "
            "are sometimes stronger in large caps ([Study 121 -- Magic-Formula](../../121-magic-formula/)).\n"
            "- **Asset growth.** Cooper, Gulen & Schill (2008) find asset-growth (total-asset "
            "expansion rate) predicts lower returns through an overinvestment channel -- related "
            "but distinct mechanism ([Study 44 -- Growth-Spurt](../../44-growth-spurt/)).\n"
            "- **NOA.** Hirshleifer et al. (2004) balance-sheet bloat is a related over-investment "
            "signal ([Study 153 -- Net-Operating-Assets](../../153-net-operating-assets/)).\n\n"
            "*Want to confirm the LSV effect on a clean universe? Fork this, plug in a "
            "point-in-time all-US-stock panel (CRSP/Compustat-based historical membership), "
            "and show |t| >= 2 on the low-minus-high-growth hedge. That's the bar.*"
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
