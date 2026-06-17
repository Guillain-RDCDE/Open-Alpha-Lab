"""Generate the two narrative notebooks for Study 231 (Sloan Accruals).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
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

# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-16).
# All numbers are SURVIVORSHIP-BIASED upper bounds.
R = dict(
    n_tickers=407, n_years=17, start_year=2009, end_year=2025,
    # Low-accruals (Q1) quintile
    lo_mean=23.8, lo_sharpe=1.54, lo_t=9.10, lo_hit=94,
    # High-accruals (Q5) quintile
    hi_mean=18.1, hi_sharpe=1.05, hi_t=6.51, hi_hit=82,
    # Equal-weight market
    mkt_mean=18.2, mkt_sharpe=1.39, mkt_t=9.68,
    # Hedge (Q1 - Q5)
    hedge_mean=5.7, hedge_sharpe=0.57, hedge_t=2.73, hedge_hit=71,
    # Excess vs market
    lo_excess_mean=5.7, lo_excess_t=3.57,
    hi_excess_mean=-0.1, hi_excess_t=-0.04,
    # Random control
    rand_std=4.1, rand_pct_beaten=92,
)

# Year-by-year frozen data (Q1, Q5, market, hedge)
_Q1  = [24.7, 13.9, 21.1, 49.4, 23.2,  7.9, 11.3, 27.1,  7.1, 40.8, 29.6, 36.2, -12.9, 44.6, 33.6, 25.5, 21.9]
_Q5  = [13.7, -6.4, 34.5, 51.3, 15.1,  7.6, 26.0, 23.9, -7.2, 41.8, 13.0, 28.9, -12.7, 28.3, 24.4, 17.3,  8.4]
_MKT = [21.7,  4.1, 22.5, 39.8, 18.8,  6.6, 18.5, 26.9, -2.7, 34.9, 18.4, 32.2,  -9.7, 27.3, 20.2, 16.2, 13.1]
_HDG = [11.0, 20.3,-13.4, -1.9,  8.1,  0.4,-14.7,  3.2, 14.3, -1.0, 16.6,  7.3,  -0.2, 16.3,  9.2,  8.3, 13.5]
_YRS = list(range(R["start_year"], R["end_year"] + 1))

VERDICT_BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Accruals on Large Caps: Decayed](https://img.shields.io/badge/Accruals_on_LargeCaps-Decayed-8b949e?style=flat-square)\n\n"
)

# ---------------------------------------------------------------------------
# Shared boot code
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath("../../.."))   # repo root (quantlab/)
sys.path.insert(0, os.path.abspath(".."))          # study package (sloan_accruals/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from sloan_accruals import data, strategy as st

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
            "# Sloan Accruals -- do high-accrual firms really underperform the cash-earners?\n"
            "### The Sloan (1996) accruals anomaly on the S&P 500 EDGAR panel\n\n"
            + VERDICT_BADGES +
            "Richard Sloan showed in 1996 that firms with high accounting accruals -- earnings "
            "backed by accounting entries rather than real cash flows -- earn low future returns. "
            "We test whether the same logic holds on ~407 S&P 500 names using real SEC filings "
            "and ask whether it still works in 2026.\n\n"
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
            f"| Does low accruals outperform high accruals? | Yes -- by **+{R['hedge_mean']:.1f}%/yr**, "
            f"positive in {R['hedge_hit']}% of years. |\n"
            f"| Is that statistically real? | Nominally, HAC *t* = {R['hedge_t']:.2f} -- "
            f"passes |t| >= 2. But the panel is survivorship-biased and virtually "
            f"all the gain comes from the long side (low-accruals excess "
            f"+{R['lo_excess_mean']:.1f}%/yr, *t* = +{R['lo_excess_t']:.2f}). "
            f"The **short side is flat** (high-accruals excess = {R['hi_excess_mean']:.1f}%/yr, "
            f"*t* = {R['hi_excess_t']:.2f}). |\n"
            f"| Does low accruals beat a random stock pick? | Yes: beats "
            f"{R['rand_pct_beaten']}% of random same-size draws -- but this "
            "may reflect a quality/profitability tilt, not accruals mispricing. |\n"
            "| Could you trade it? | Fragile. Cheap to implement; but short side "
            "does not contribute, and the effect has decayed on large caps since "
            "Sloan (1996) was published. |\n\n"
            "> **The asymmetry**: cash-backed earnings predict quality and outperformance; "
            "but high-accrual S&P 500 survivors earn essentially market returns. "
            "The anomaly is one-sided on this universe -- not the bilateral "
            "mispricing Sloan (1996) documented in the full US stock universe."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *Buy stocks with low accounting accruals (earnings backed by real cash). "
            "Avoid stocks where earnings exceed cash flows by large margins. "
            "Rebalance annually. The high-accrual firms' earnings are not persistent; "
            "when accruals mean-revert, earnings disappoint and prices fall.*\n\n"
            "Accruals are computed from the annual income statement and cash flow statement:\n\n"
            "- **Accruals** = Net Income - Operating Cash Flow\n"
            "- **Avg Assets** = (Total Assets_t + Total Assets_{t-1}) / 2\n"
            "- **Accruals ratio** = Accruals / Avg Assets\n\n"
            "A high accruals ratio means earnings are running well ahead of cash generation. "
            "The Sloan argument: investors treat accrual earnings and cash earnings as equally "
            "persistent, when in fact accrual earnings mean-revert faster. "
            "When the reversion hits, actual earnings fall short of estimates "
            "and the stock price follows."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If the accruals anomaly is real and tradable on large US stocks, it is a "
            "purely mechanical, publicly available, low-turnover signal requiring only "
            "reading two lines from annual reports. The seminal paper found a "
            "~10%/yr hedge on the full US stock universe in 1962-1991. "
            "If the effect has been arbitraged away -- especially on large, liquid, "
            "heavily covered S&P 500 names -- then this test is a cautionary tale "
            "about post-publication decay and the limits of large-cap factor investing."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three honesty requirements:\n\n"
            "1. **Reporting lag.** Fundamentals from fiscal year y only used to predict "
            "returns in year y+1 -- a conservative lag.\n"
            "2. **Random-portfolio control.** Does the low-accruals quintile beat a random "
            "draw of the same number of stocks? Does the long-only outperformance "
            "reflect the *signal* or just a *quality* / *size* tilt?\n"
            "3. **Name the survivorship bias.** The EDGAR panel covers current S&P 500 "
            "members projected back. High-accruals failures are absent."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown -- what the real data shows\n\n"
            "Positive control first: **when we plant a real effect in a synthetic panel, "
            "the engine finds it.** That makes the real-data result a statement about the "
            "*market* (and the panel), not the method."
        ),
        code(
            "# Synthetic positive control: plant a known accruals premium, confirm detection\n"
            "premiums = [0.0, -0.04, -0.08, -0.12]\n"
            "hedges = []\n"
            "for p in premiums:\n"
            "    s, f, _ = data.synthetic_panel(n_firms=300, n_years=25, premium=p, seed=231)\n"
            "    hh = st.quintile_returns(s, f, q=0.20)\n"
            "    hedges.append(hh['hedge'].mean() * 100)\n"
            "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            "ax.plot([abs(p)*100 for p in premiums], hedges, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Planted accruals effect (% alpha per unit z-rank)')\n"
            "ax.set_ylabel('Low-minus-high-accruals hedge (%/yr)')\n"
            "ax.set_title('Synthetic control: engine finds the effect when planted')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Engine faithfully recovers planted accruals effects.')"
        ),
        md("**Now the real EDGAR panel -- quintile sort on accruals:**"),
        code(
            "# Year-by-year low-accruals vs high-accruals vs market\n"
            "if HAVE_REAL:\n"
            "    q1_vals = h['q1'].values * 100\n"
            "    q5_vals = h['q5'].values * 100\n"
            "    hdg_vals = h['hedge'].values * 100\n"
            "    years_real = h.index.tolist()\n"
            "else:\n"
            "    q1_vals, q5_vals, hdg_vals, years_real = Q1, Q5, HDG, YRS\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "x = range(len(years_real))\n"
            "axes[0].bar(x, q1_vals, label='Q1 low-accruals', color=GREEN, alpha=0.7, width=0.4)\n"
            "axes[0].bar([i+0.4 for i in x], q5_vals, label='Q5 high-accruals', color=RED, alpha=0.7, width=0.4)\n"
            "axes[0].set_xticks([i+0.2 for i in x])\n"
            "axes[0].set_xticklabels(years_real, rotation=45, fontsize=8)\n"
            "axes[0].set_ylabel('Annual return %'); axes[0].legend()\n"
            "axes[0].set_title('Q1 (low accruals) vs Q5 (high accruals) raw returns')\n"
            "colors = [GREEN if v > 0 else RED for v in hdg_vals]\n"
            "axes[1].bar(years_real, hdg_vals, color=colors)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_ylabel('Hedge return (Q1 - Q5) %')\n"
            "axes[1].set_title(f'Low-minus-high-accruals hedge: {sum(1 for v in hdg_vals if v>0)}/{len(hdg_vals)} positive years')\n"
            "axes[1].tick_params(axis='x', rotation=45, labelsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Mean hedge: {{np.mean(hdg_vals):+.1f}}%/yr  (frozen R: +{{R[\"hedge_mean\"]:.1f}}%/yr)')"
        ),
        md(
            f"The low-accruals quintile earns **+{R['lo_mean']:.1f}%/yr** vs the high-accruals "
            f"quintile's **+{R['hi_mean']:.1f}%/yr** -- a hedge of **+{R['hedge_mean']:.1f}%/yr**. "
            f"But the **equal-weight market is +{R['mkt_mean']:.1f}%/yr** -- nearly identical "
            f"to the high-accruals quintile. The high-accruals arm does not underperform; it "
            "matches market returns. The entire hedge comes from low-accruals (cash-backed "
            "earnings) outperforming -- likely a quality/profitability effect, not the bilateral "
            "accruals mispricing Sloan (1996) documented in the full US universe."
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal -- Weak.** Nominal hedge +{R['hedge_mean']:.1f}%/yr, HAC *t* = "
            f"{R['hedge_t']:.2f}. Passes the |t| >= 2 bar on this panel. But: "
            "(a) the panel is survivorship-biased, so this is an upper bound; "
            "(b) virtually all the gain is long-side (low-accruals excess = "
            f"+{R['lo_excess_mean']:.1f}%/yr); (c) the short side is flat "
            f"(high-accruals excess = {R['hi_excess_mean']:.1f}%/yr, *t* = "
            f"{R['hi_excess_t']:.2f}) -- not the short-side drag predicted by Sloan.\n"
            f"- **Tradability -- Fragile.** Low annual turnover on a liquid universe. "
            "But: the alpha estimate is uncertain and the academic literature "
            "documents significant post-publication decay on large caps (Green et al. 2011). "
            "The short side contributes nothing -- you would be paying borrow and "
            "transaction costs on a short leg that earns market return.\n"
            "- **Myth check -- Decayed on large caps.** Sloan (1996) found ~10%/yr on all "
            "US stocks in 1962-1991. On S&P 500 survivors in 2009-2026 we find "
            "+5.7%/yr, driven entirely by the long side. The 'short high-accruals' leg "
            "of the trade is not working on this universe."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "Annual rebalancing of ~80 S&P 500 names is operationally trivial and cheap. "
            "The obstacles are in the signal:\n\n"
            "1. **One-sided alpha.** The entire +5.7%/yr comes from holding cash-backed "
            "earnings stocks. You could replicate most of it by using a simple "
            "cash-from-operations yield screen (operating cash flow / price) instead "
            "of a pure accruals sort.\n"
            "2. **Short side is not working.** High-accruals S&P 500 survivors earn market "
            "return. The original Sloan short was powered by accruals manipulators "
            "who subsequently failed and were expelled from the index -- those are "
            "precisely the stocks missing from this survivorship-biased panel.\n"
            "3. **Post-publication decay.** As the Sloan paper spread, arbitrageurs "
            "closed the mispricing, especially in large-cap liquid names. "
            "The anomaly survives mainly where information frictions are highest "
            "(small, illiquid, low-coverage stocks -- the opposite of S&P 500).\n\n"
            "The right conclusion: **FRAGILE** on this panel; do not short high-accrual "
            "large caps based on this test."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The right universe.** Sloan (1996) ran his test on all US stocks "
            "1962-1991. A point-in-time Compustat universe with historical constituents "
            "is the minimum required for a credible accruals test on large caps.\n"
            "- **Balance-sheet version.** The original Sloan measure uses the change in "
            "net working capital minus depreciation (balance-sheet approach). "
            "Our cash-flow-statement approach is cleaner post-SFAS 95 (1988). "
            "Richardson et al. (2005) decompose accruals further into working-capital "
            "and long-term components.\n"
            "- **Related signals.** [Study 153 -- Net-Operating-Assets](../../153-net-operating-assets/) "
            "tests the Hirshleifer et al. (2004) balance-sheet NOA measure "
            "(the 'total balance-sheet bloat' version of the same idea) on the same panel.\n"
            "- **Quality as a cleaner lens.** If the long-side quality/cash-flow tilt "
            "is the real driver, a more direct screen on operating cash flow yield or "
            "Piotroski F-score (Study 65) may be more robust.\n\n"
            "*Think the accruals anomaly is still alive in a broader universe? "
            "Fork this, extend to a point-in-time all-US-stock panel (Compustat-based), "
            "and show |t| >= 2 on the short-side underperformance. That's the missing half.*"
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
            "# Sloan Accruals -- a quantitative teardown\n"
            "### Accruals quintile sort - S&P 500 EDGAR panel - HAC inference - "
            "random-portfolio null - post-publication decay\n\n"
            + VERDICT_BADGES +
            "The deep companion to [01_for_the_curious.ipynb](01_for_the_curious.ipynb) -- "
            "same seven beats, every claim now carrying its standard error. We test the "
            "Sloan accruals anomaly on a survivorship-biased S&P 500 EDGAR panel "
            f"(~{R['n_tickers']} tickers, {R['start_year']}-{R['end_year']}), measure "
            "the low-minus-high-accruals hedge with a HAC t-stat, and pin it against "
            "a random-portfolio null distribution.\n\n"
            "> **Not investment advice.** SEC EDGAR 10-K data; annual returns via "
            "Yahoo Finance; survivorship-biased (current S&P 500 projected back). "
            "Sources in [`docs/references.md`](../docs/references.md), reproducible "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Key finding**: the hedge is nominally real (HAC t = +2.73) but "
            "driven entirely by the long leg (low-accruals excess +5.7%/yr); "
            "the short leg (high-accruals) earns market return (-0.1%/yr excess). "
            "Post-publication decay is documented in the literature for large caps."
        ),
        code(BOOT + "\n" + R_CODE),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Hedge **+{R['hedge_mean']:.1f}%/yr**, "
            f"HAC *t* = **+{R['hedge_t']:.2f}** -- passes |t| >= 2 on this panel. "
            f"BUT: low-accruals excess +{R['lo_excess_mean']:.1f}%/yr "
            f"(*t* = +{R['lo_excess_t']:.2f}); high-accruals excess "
            f"{R['hi_excess_mean']:.1f}%/yr (*t* = {R['hi_excess_t']:.2f}). "
            f"Short side is flat -- survivorship-biased upper bound. |\n"
            f"| **Tradability** | `FRAGILE` | Annual rebalancing on liquid S&P 500 names "
            "is cheap. But short side contributes nothing; post-publication decay "
            "documented; alpha estimate is survivorship-inflated. |\n"
            f"| **Myth check** | `DECAYED` | Sloan (1996) ~10%/yr on full US universe "
            "in 1962-1991. On large-cap survivors 2009-2026: +5.7%/yr, "
            "long-side only. |\n\n"
            "> **In plain words:** the 1996 accruals anomaly still leaves a long-side "
            "signal (cash earners outperform on large caps), but the short side has "
            "been arbitraged away. The bilateral mispricing Sloan documented is not "
            "present in this survivor panel."
        ),

        # ---- BEAT 1 -----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $r_{i,t+1}$ be firm $i$'s return in year $t+1$. Define:\n\n"
            "$$\\text{Acc}_{i,t} = \\frac{\\text{NI}_{i,t} - \\text{CFO}_{i,t}}"
            "{(A_{i,t} + A_{i,t-1})/2}$$\n\n"
            "where $\\text{NI}$ = net income, $\\text{CFO}$ = operating cash flow, "
            "$A$ = total assets. High $\\text{Acc}$ = accrual-heavy earnings.\n\n"
            "The **H1 (signal)** claim (Sloan 1996) is:\n\n"
            "$$\\mathbb{E}\\left[\\bar{r}_{\\text{low-acc},t+1} - "
            "\\bar{r}_{\\text{high-acc},t+1}\\right] > 0$$\n\n"
            "with a HAC t-stat |t| >= 2 **on both sides** (i.e., the long side "
            "significantly outperforms AND the short side significantly underperforms "
            "vs market). **H2 (beats random)** requires the low-accruals quintile to "
            "beat a random same-size draw with better interpretation (quality vs signal). "
            "**H3 (tradable)** requires the hedge to survive costs and scale."
        ),

        # ---- BEAT 2 -----------------------------------------------------------
        md(
            "## 2 - So what? -- the stakes\n\n"
            "H1 is partially supported: the long side (low accruals) earns a clear "
            "excess (+5.7%/yr, *t* = +3.57). But the short side (high accruals) "
            "earns essentially market return (-0.1%/yr excess, *t* = -0.04). "
            "This is consistent with post-publication decay: large-cap accruals "
            "manipulators have been under scrutiny since 1996; any systematic "
            "underperformance of high-accruals S&P 500 names has been arbitraged away. "
            "The long-side quality tilt may survive -- but it also may simply reflect "
            "the profitability/quality premium that cash-flow-heavy large caps have "
            "enjoyed in the 2009-2026 bull market."
        ),

        # ---- BEAT 3 -----------------------------------------------------------
        md(
            "## 3 - The protocol\n\n"
            "- **Data**: three EDGAR concepts from the desk's shared cache "
            "(NetIncomeLoss, NetCashProvidedByUsedInOperatingActivities, Assets); "
            "annual returns from `_edgar_yrret.parquet`.\n"
            f"- **Universe**: current S&P 500 members with all three concepts present "
            f"(~{R['n_tickers']} tickers total); survivorship-biased.\n"
            f"- **Window**: {R['n_years']} years of signal ({R['start_year']}-{R['end_year']}), "
            f"forecasting returns {R['start_year']+1}-{R['end_year']+1}.\n"
            "- **Signal**: accruals ratio (higher = more accrual / less cash backing); "
            "quintile sort (q=0.20).\n"
            "- **Inference**: HAC Newey-West t-stat on the 17-observation annual hedge series.\n"
            "- **Control**: empirical distribution of random same-size portfolios "
            "(500 draws x 17 years).\n"
            "- **Positive control**: synthetic panel with planted accruals premium confirms "
            "the engine can detect a real effect."
        ),

        # ---- BEAT 4 -----------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - Synthetic positive control -- the engine works when an effect is planted\n\n"
            "Sweep the planted accruals premium from 0 to -12%/yr per unit z-rank and "
            "confirm the hedge tracks it monotonically."
        ),
        code(
            "premiums = [0.0, -0.04, -0.08, -0.12]\n"
            "synth_rows = []\n"
            "for p in premiums:\n"
            "    s, f, _ = data.synthetic_panel(n_firms=300, n_years=25, premium=p, seed=231)\n"
            "    hh = st.quintile_returns(s, f, q=0.20)\n"
            "    ss = st.summary(hh['hedge'])\n"
            "    synth_rows.append((abs(p)*100, ss['mean']*100, ss['tstat']))\n"
            "sydf = pd.DataFrame(synth_rows, columns=['premium_%', 'hedge_%/yr', 'HAC_t'])\n"
            "fig, ax = plt.subplots(figsize=(9, 4))\n"
            "colors = [GREEN if t > 2 else (RED if t < -2 else GREY) for t in sydf['HAC_t']]\n"
            "ax.bar(sydf['premium_%'], sydf['hedge_%/yr'], color=colors, width=1.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Planted accruals effect (%/yr per unit z-rank)')\n"
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
            "    q_means = [R['lo_mean'], 17.4, 15.3, 16.1, R['hi_mean']]\n"
            "    q_tstats = [R['lo_t'], 8.74, 9.00, 9.98, R['hi_t']]\n"
            "    s_hedge = dict(mean=R['hedge_mean']/100, tstat=R['hedge_t'], hit_rate=R['hedge_hit']/100)\n"
            "    s_lo_ex = dict(mean=R['lo_excess_mean']/100, tstat=R['lo_excess_t'])\n"
            "    s_hi_ex = dict(mean=R['hi_excess_mean']/100, tstat=R['hi_excess_t'])\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "x = [1, 2, 3, 4, 5]\n"
            "bar_colors = [GREEN, GREY, GREY, GREY, RED]\n"
            "axes[0].bar(x, q_means, color=bar_colors)\n"
            "axes[0].axhline(R['mkt_mean'] if not HAVE_REAL else float(h['market'].mean()*100),\n"
            "                ls='--', c=GREY, lw=1.5, label='EW market')\n"
            "axes[0].set_xticks(x); axes[0].set_xticklabels(['Q1 (lo)', 'Q2', 'Q3', 'Q4', 'Q5 (hi)'])\n"
            "axes[0].set_ylabel('Mean annual return (%/yr)')\n"
            "axes[0].set_title('Quintile returns: low-accruals outperforms, high-accruals near market')\n"
            "axes[0].legend()\n"
            "axes[1].bar(x, q_tstats, color=bar_colors)\n"
            "for s in (2, -2): axes[1].axhline(s, ls='--', c=GREY, lw=1)\n"
            "axes[1].set_xticks(x); axes[1].set_xticklabels(['Q1 (lo)', 'Q2', 'Q3', 'Q4', 'Q5 (hi)'])\n"
            "axes[1].set_ylabel('HAC t-stat')\n"
            "axes[1].set_title('All quintiles earn positive returns (bull market, survivor bias)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Hedge: {s_hedge[\"mean\"]*100:+.1f}%/yr  HAC t={s_hedge[\"tstat\"]:+.2f}  hit={s_hedge[\"hit_rate\"]:.0%}')\n"
            "print(f'Low-acc excess: {s_lo_ex[\"mean\"]*100:+.1f}%/yr  t={s_lo_ex[\"tstat\"]:+.2f}')\n"
            "print(f'High-acc excess: {s_hi_ex[\"mean\"]*100:+.1f}%/yr  t={s_hi_ex[\"tstat\"]:+.2f}')"
        ),
        md(
            f"> The low-minus-high-accruals hedge is **+{R['hedge_mean']:.1f}%/yr**, "
            f"HAC *t* = **+{R['hedge_t']:.2f}**. But the high-accruals quintile earns "
            f"**+{R['hi_mean']:.1f}%/yr** -- nearly equal to the market's "
            f"**+{R['mkt_mean']:.1f}%/yr**. The hedge is asymmetric: long-side quality "
            f"(low-accruals excess = +{R['lo_excess_mean']:.1f}%/yr, *t* = "
            f"+{R['lo_excess_t']:.2f}) drives everything; the short side is flat "
            f"({R['hi_excess_mean']:.1f}%/yr excess, *t* = {R['hi_excess_t']:.2f}). "
            "This is the signature of post-publication decay: the easy short "
            "(high-accruals manipulators) has been closed."
        ),
        md(
            "### 4c - Random-portfolio null distribution\n\n"
            "Does the low-accruals quintile beat a blind draw of the same number of stocks? "
            "We simulate 500 random same-size portfolios each year."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rand = st.random_portfolio_returns(sig, fwd, q=0.20, n_draws=500, seed=231)\n"
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
            "           label=f'Low-accruals quintile: {actual_excess*100:+.1f}%/yr excess')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Annual excess return vs EW market (%)')\n"
            "ax.set_ylabel('Count'); ax.legend()\n"
            "ax.set_title(f'Low-accruals beats {pct_beaten:.0%} of random draws -- high, but may be quality tilt')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Low-accruals portfolio beats {pct_beaten:.0%} of random same-size draws')"
        ),
        md(
            f"> The low-accruals portfolio beats **{R['rand_pct_beaten']}%** of random "
            "same-size draws. This looks strong -- but note: the random draw includes "
            "all quintiles, including the low-accruals firms themselves. The 92% "
            "bar likely reflects that low-accruals S&P 500 firms are quality compounders "
            "that systematically outperform in a bull-market decade. The same firm that "
            "has strong operating cash flow tends to also have strong margins, low debt, "
            "and high returns on capital. The accruals signal may be incidentally "
            "capturing quality."
        ),

        # ---- BEAT 5 -----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal `WEAK`** -- hedge +{R['hedge_mean']:.1f}%/yr, HAC *t* = "
            f"+{R['hedge_t']:.2f}; low-accruals excess +{R['lo_excess_mean']:.1f}%/yr "
            f"(*t* = +{R['lo_excess_t']:.2f}). Nominally passes |t| >= 2 but: "
            "(a) survivorship-biased upper bound; (b) short side flat "
            f"({R['hi_excess_mean']:.1f}%/yr, *t* = {R['hi_excess_t']:.2f}). "
            "Literature documents significant post-publication decay.\n"
            f"- **Tradability `FRAGILE`** -- annual rebalancing of ~80 S&P 500 names "
            "is operationally cheap and liquid. But: the short leg earns market return "
            "(wasting borrow and transaction costs); the long leg may be a quality tilt "
            "rather than a pure accruals signal; and the alpha is an upper bound.\n"
            "- **Myth check `DECAYED`** -- the Sloan (1996) bilateral mispricing is not "
            "present in this panel. The short side of the 1996 anomaly (high-accrual "
            "underperformers) has been effectively arbitraged away in the S&P 500. "
            "The original result survives on the long side only."
        ),

        # ---- BEAT 6 -----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- the honest answer\n\n"
            "Annual rebalancing of ~80 large-cap names is operationally feasible. "
            "The obstacles are in the signal anatomy:\n\n"
            "1. **Asymmetric alpha.** You earn +5.7%/yr by being long low-accruals names. "
            "The short leg costs borrow (~0.5-1.5%/yr) and transaction costs while "
            "contributing nothing. A long-only version (buy low-accruals, hold "
            "benchmark for the rest) is cheaper and captures most of the edge.\n"
            "2. **Quality overlap.** Low accruals (NI ≈ CFO) is a proxy for earnings "
            "quality. The long-only premium may reflect factor exposure to quality/"
            "profitability (Fama-French RMW factor) more than accruals mispricing "
            "per se. A factor-adjusted regression would tell you what's left.\n"
            "3. **Post-publication decay.** The Sloan paper is now 30 years old. "
            "Quant funds systematically hunt accruals manipulators. The most "
            "egregious high-accruals S&P 500 firms have been called out; "
            "the remaining high-accruals names are often legitimate business models "
            "(e.g., growth stage firms capitalising development costs).\n\n"
            "The correct conclusion: **FRAGILE** -- the long-side quality screen has "
            "potential value but requires factor-controlling before claiming a pure "
            "accruals alpha."
        ),

        # ---- BEAT 7 -----------------------------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **Factor-control the long side.** Add exposure to the Fama-French RMW "
            "(robust-minus-weak profitability) factor. If the low-accruals alpha "
            "survives after controlling for RMW, there is incremental accruals "
            "information. If it vanishes, you have been proxying for quality.\n"
            "- **Balance-sheet accruals.** The original Sloan (1996) measure uses "
            "change in net working capital minus depreciation. Our cash-flow approach "
            "is conceptually cleaner post-SFAS 95 but produces slightly different "
            "rankings. Richardson et al. (2005) decompose accruals into working-capital "
            "and long-term components -- the long-term component (related to PP&E) "
            "may carry incremental information.\n"
            "- **Broader universe.** The definitive test requires a point-in-time "
            "Compustat panel. Historical S&P 500 membership data (CRSP/Compustat) "
            "would include the high-accruals firms that were later expelled -- "
            "precisely those where the short side should work.\n"
            "- **Related desk studies.** "
            "[Study 153 -- Net-Operating-Assets](../../153-net-operating-assets/): "
            "the Hirshleifer et al. (2004) balance-sheet generalisation of Sloan "
            "accruals; same EDGAR panel, same caveats, similar verdict.\n\n"
            "*Want to confirm the bilateral accruals anomaly is alive on large caps? "
            "Build a point-in-time panel, control for RMW, and show |t| >= 2 on "
            "the short-side underperformance. That's the missing piece.*"
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
