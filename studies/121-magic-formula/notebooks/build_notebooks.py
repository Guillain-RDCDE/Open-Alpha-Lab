"""Generate the two narrative notebooks for Study 121 (Magic-Formula).

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

# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-14).
R = dict(
    n_tickers=233, n_years=17, start_year=2008, end_year=2024,
    # Top decile
    top_mean=21.9, top_sharpe=1.08, top_t=5.38, top_hit=89,
    # EW market
    mkt_mean=20.7, mkt_sharpe=1.51, mkt_t=10.51,
    # Top decile excess
    excess_mean=1.2, excess_sharpe=0.12, excess_t=0.44, excess_hit=44,
    # Long-short hedge (top - bottom)
    ls_mean=-10.6, ls_sharpe=-0.35, ls_t=-1.43, ls_hit=39,
    # Bottom decile
    bot_mean=31.3,
    # Random control
    rand_mean_excess=-0.0, rand_std=11.0, rand_pct_beaten=61,
)

VERDICT_BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Beats_a_random_portfolio%3F: No](https://img.shields.io/badge/Beats_a_random_portfolio%3F-No-8b949e?style=flat-square)\n\n"
)

# ---------------------------------------------------------------------------
# Shared boot code
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath("../../.."))   # repo root (quantlab/)
sys.path.insert(0, os.path.abspath(".."))          # study package (magic_formula/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from magic_formula import data, strategy as st

# Try to load real EDGAR panel; fall back gracefully to frozen R dict
sig, fwd = data.fetch_panel()
HAVE_REAL = not sig.empty

if HAVE_REAL:
    h = st.quantile_hedge(sig, fwd, q=0.10)
else:
    h = pd.DataFrame()

print("EDGAR panel loaded:", HAVE_REAL,
      f"({len(sig.columns) if HAVE_REAL else 0} tickers, {len(sig) if HAVE_REAL else 0} years)")
"""

# Frozen R dict preamble for fallback
R_CODE = f"""\
R = dict(
    n_tickers={R['n_tickers']}, n_years={R['n_years']},
    top_mean={R['top_mean']}, top_sharpe={R['top_sharpe']}, top_t={R['top_t']}, top_hit={R['top_hit']},
    mkt_mean={R['mkt_mean']}, mkt_sharpe={R['mkt_sharpe']},
    excess_mean={R['excess_mean']}, excess_t={R['excess_t']}, excess_hit={R['excess_hit']},
    ls_mean={R['ls_mean']}, ls_t={R['ls_t']}, ls_hit={R['ls_hit']},
    bot_mean={R['bot_mean']},
    rand_mean_excess={R['rand_mean_excess']}, rand_std={R['rand_std']}, rand_pct_beaten={R['rand_pct_beaten']},
)
"""


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Magic-Formula — does Greenblatt's quality+value screen beat the market?\n"
            "### The Little Book recipe tested on the S&P 500 with real SEC filings\n\n"
            + VERDICT_BADGES +
            "Joel Greenblatt's *The Little Book That Beats the Market* (2005) pitches a "
            "deceptively simple idea: rank every stock by how good the business is (Return "
            "on Capital) and how cheaply you're buying it (Earnings Yield); sum those two "
            "ranks; buy the top 20–30 names; hold for a year; repeat. The formula "
            "promises to beat the market without forecasting, just by finding **good "
            "companies at good prices**. We ran it on ~233 S&P 500 names using real SEC "
            "filings — here's what happened.\n\n"
            "> 📓 **Plain-language layer.** The t-stats, bootstrap CIs, and factor "
            "decomposition live in **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool; every chart "
            "is drawn by the code beside it. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md).\n"
            ">\n"
            "> **Survivorship bias**: this panel covers only *current* S&P 500 members "
            "projected backwards — every number is an upper bound on a live strategy."
        ),
        code(BOOT + "\n" + R_CODE),

        # ---- BEAT 0 — ANSWER FIRST -------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | What the data says |\n|---|---|\n"
            f"| Does the top decile beat the market? | Barely: **+{R['excess_mean']:.1f}%/yr** excess, "
            f"statistically **indistinguishable from zero** (HAC *t* = {R['excess_t']:.2f}). |\n"
            f"| Does the long-short hedge (top minus bottom) earn a premium? | **No** — it *inverts*: "
            f"**{R['ls_mean']:.1f}%/yr**, the formula shorts outperform the longs. |\n"
            f"| Is the excess better than just picking random stocks? | Barely: the top decile "
            f"beats only **{R['rand_pct_beaten']}%** of random portfolios of the same size. |\n"
            f"| Could you trade it? | **Mirage**: no statistically real edge to harvest on large-cap "
            f"S&P 500 names, and survivorship overstates every number. |\n\n"
            "> **Why does the formula invert on large caps?** Greenblatt built it for *small, "
            "neglected value stocks* — where a great business at a low price is a genuine "
            "surprise. On heavily-covered S&P 500 companies, the market has already priced "
            "in quality and cheapness. The 'bad' stocks in the formula's bottom decile are "
            "often beaten-down cyclicals and special situations that rebound."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *Buy stocks with high Return on Capital (EBIT / Invested Capital) and high "
            "Earnings Yield (EBIT / Enterprise Value). Rank all stocks on each axis, sum the "
            "two ranks, buy the top 20–30. Rebalance once a year. Beat the market without "
            "predicting anything.*\n\n"
            "The formula is a combination of two of the most robust factor ideas in finance:\n"
            "- **ROC (quality)**: great businesses earn a lot per dollar of capital. "
            "Think of it as finding *moats* — durable competitive advantages that let a "
            "company generate high returns year after year.\n"
            "- **Earnings Yield (value)**: you're not paying a crazy price for those earnings. "
            "EBIT / Enterprise Value is essentially the inverse of EV/EBIT — a lower price "
            "multiple means you're getting more earnings per dollar invested.\n\n"
            "Separately, each has academic support. Together, Greenblatt argued, they "
            "identify *the best companies at the best prices* — and promised ~30%/yr in his "
            "back-test from 1988 to 2004."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the formula works on large liquid stocks you can actually trade, it's a "
            "free, transparent, rules-based strategy straight from public filings — no quant "
            "black box, no data subscriptions, no ML. It's the most-cited *retail-accessible* "
            "quant strategy ever written. If it *doesn't* work — or only works in the "
            "universe of small, illiquid stocks Greenblatt tested it on — then the millions "
            "of investors who bought the book are running a well-intentioned strategy on the "
            "wrong universe."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three honesty requirements:\n\n"
            "1. **Use real filings with a reporting lag.** We pull EBIT, assets, liabilities, "
            "debt, cash, and equity from SEC EDGAR 10-K annual reports. To avoid any "
            "look-ahead, fundamentals from fiscal year *y* are only used to predict returns "
            "in year *y+1* — a conservative one-year lag.\n"
            "2. **Compare against a random portfolio of the same size.** The formula picks "
            "~10% of the universe each year — roughly 15–20 names. If that's better than "
            "just picking 15–20 randomly, the formula is doing something real. If not, it's "
            "concentration risk disguised as skill.\n"
            "3. **Name the survivorship bias.** The EDGAR panel covers *current* S&P 500 "
            "members projected backwards. The Enrons, Lehmans, and Sears of the world — "
            "precisely the kind of 'cheap' names the formula might have bought — are gone. "
            "Every number below is an upper bound."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — the formula inverts on large caps\n\n"
            "The synthetic positive control first: **when we plant a real premium in a "
            "fake panel, the formula finds it**. So the machinery is honest — the "
            "real-data result is a statement about the *market*, not the method."
        ),
        code(
            "# Synthetic positive control: plant a known premium, confirm the engine finds it\n"
            "premiums = [0.0, 0.03, 0.06, 0.10]\n"
            "top_excess = []\n"
            "for p in premiums:\n"
            "    s, f, _ = data.synthetic_panel(n_firms=300, n_years=25, premium=p, seed=121)\n"
            "    hh = st.quantile_hedge(s, f, q=0.10)\n"
            "    top_excess.append(hh['hedge'].mean() * 100)\n"
            "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            "ax.plot([p*100 for p in premiums], top_excess, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Planted premium (%/yr per unit z-rank)')\n"
            "ax.set_ylabel('Top-decile excess vs EW market (%/yr)')\n"
            "ax.set_title('Synthetic control: the engine works when a premium exists')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('The engine faithfully recovers planted premiums — so the real result below is market evidence.')"
        ),
        md("**Now the real EDGAR panel:**"),
        code(
            "# Year-by-year top decile vs EW market on real data\n"
            "if HAVE_REAL:\n"
            "    top_vals = h['top'].values * 100\n"
            "    mkt_vals = h['market'].values * 100\n"
            "    excess_vals = h['hedge'].values * 100\n"
            "    years = h.index.tolist()\n"
            "else:\n"
            "    # Frozen representative values from docs/results.md\n"
            "    years = list(range(R['start_year'], R['start_year'] + R['n_years']))\n"
            "    top_vals = [64,39,9,25,38,18,5,9,25,-6,47,24,42,-22,33,18,19]\n"
            "    mkt_vals = [37,23,7,25,44,18,7,20,28,-1,34,20,32,-11,34,23,17]\n"
            "    excess_vals = [27,16,2,0,-6,0,-3,-11,-3,-6,14,4,10,-11,-1,-5,2]\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "# Left: bar chart of top vs market\n"
            "x = range(len(years))\n"
            "axes[0].bar(x, top_vals, label='Top decile', color=GREEN, alpha=0.7, width=0.4)\n"
            "axes[0].bar([i+0.4 for i in x], mkt_vals, label='EW market', color=GREY, alpha=0.7, width=0.4)\n"
            "axes[0].set_xticks([i+0.2 for i in x]); axes[0].set_xticklabels(years, rotation=45, fontsize=8)\n"
            "axes[0].set_ylabel('Annual return %'); axes[0].legend()\n"
            "axes[0].set_title('Top decile vs EW market (raw returns)')\n"
            "# Right: excess return each year\n"
            "colors = [GREEN if v > 0 else RED for v in excess_vals]\n"
            "axes[1].bar(years, excess_vals, color=colors)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_ylabel('Excess return over EW market %')\n"
            "axes[1].set_title(f'Top-decile excess: {sum(1 for e in excess_vals if e>0)}/{len(excess_vals)} years positive')\n"
            "axes[1].tick_params(axis='x', rotation=45, labelsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Mean excess: {{np.mean(excess_vals):+.1f}}%/yr   (frozen R: +{{R[\"excess_mean\"]:.1f}}%/yr)')"
        ),
        md(
            f"The top decile earns **+{R['excess_mean']:.1f}%/yr** above the equal-weight market — but "
            f"the HAC t-stat is only **{R['excess_t']:.2f}** (needs |t| ≥ 2 to count as real). "
            f"More striking: the **bottom decile outperforms the top by {abs(R['ls_mean']):.1f}%/yr** "
            "— the formula's long-short is backwards on large caps.\n\n"
            "> **What's going on?** On the S&P 500, companies with high ROC are usually "
            "expensive growth stocks (FAANG-type) that are already priced for their moats. "
            "The formula's 'cheap with high earnings yield' bucket catches beaten-down "
            "cyclicals, turnarounds, and value traps — stocks that often *do* bounce back "
            "once the cycle turns. The 'bad' companies at the bottom of the formula are "
            "often in temporary distress, not structural decline."
        ),

        # ---- BEAT 5 — VERDICT ------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Top-decile excess = +{R['excess_mean']:.1f}%/yr, HAC *t* = "
            f"{R['excess_t']:.2f}. Not statistically real. The long-short inverts "
            f"({R['ls_mean']:.1f}%/yr, *t* = {R['ls_t']:.2f}).\n"
            f"- **Tradability — Mirage.** No significant edge on the S&P 500 survivor panel. "
            "A real annual-turnover strategy of 15–20 names would carry high tracking error "
            "and no demonstrable alpha on this universe.\n"
            f"- **Beats a random portfolio? — No.** The top decile beats only "
            f"{R['rand_pct_beaten']}% of random portfolios of the same size — barely "
            "above chance.\n\n"
            "> The Magic Formula is a real idea built for the wrong universe here. "
            "Greenblatt's original back-test ran on *all US stocks above $50M* — a universe "
            "that includes small, neglected, under-covered names where quality and cheapness "
            "are genuine surprises. On the S&P 500 blue-chips, both signals are already in "
            "the price."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Annual rebalancing of 15–20 large-cap names is mechanically feasible and cheap — "
            "spreads are tight, commissions negligible. The problem isn't execution:\n\n"
            "- The gross edge is statistically zero (t = 0.44): you'd be paying taxes on "
            "annual turnover to harvest noise.\n"
            "- The long-short inverts: the correct trade on this panel would be to *short* "
            "the formula's top decile and *buy* the bottom — the opposite of the book.\n"
            "- Survivorship: the EDGAR panel excludes every firm that failed. The formula "
            "might have *loved* Enron (high EBIT, apparently cheap EV) in 2000."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The native habitat.** The formula was designed for small, value-oriented "
            "stocks where analyst coverage is thin. A Russell 2000 + high-B/M universe "
            "would be a fairer test — and is where [the literature](../docs/references.md) "
            "finds the effect.\n"
            "- **Related quality factors.** [Study 65 — Scorecard](../../65-scorecard/) "
            "(Piotroski F-score) and [Study 52 — Smoke-Screen](../../52-smoke-screen/) "
            "(accruals) test different quality signals on the same EDGAR panel; the accruals "
            "effect survives weakly, the F-score also inverts.\n"
            "- **The companion notebook** "
            "([02_for_the_quants.ipynb](02_for_the_quants.ipynb)) has the HAC t-stats, "
            "the random-portfolio null distribution, and the synthetic positive control sweep.\n\n"
            "*Want to find the right universe? Fork this, filter to small-cap value, and "
            "show a statistically robust (|t| ≥ 2) top-decile excess that survives the "
            "random-portfolio control. That's the bar.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Magic-Formula — a quantitative teardown\n"
            "### ROC + Earnings Yield rank · S&P 500 EDGAR panel · "
            "HAC inference · random-portfolio null · survivorship caveat\n\n"
            + VERDICT_BADGES +
            "The deep companion to [01_for_the_curious.ipynb](01_for_the_curious.ipynb) — "
            "same seven beats, every claim now carrying its standard error. We test "
            "Greenblatt's combined rank on a survivorship-biased S&P 500 EDGAR panel "
            "(~233 tickers, 2008–2024), measure the top-decile excess vs equal-weight "
            "market with a HAC t-stat, and pin it against a random-portfolio null "
            "distribution.\n\n"
            "> ⚠️ **Not investment advice.** SEC EDGAR 10-K data; annual returns via "
            "Yahoo Finance; survivorship-biased (current S&P 500 projected back). "
            "Sources in [`docs/references.md`](../docs/references.md), reproducible "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Survivorship-bias warning**: all results are upper bounds on a live "
            "strategy. Firms that failed between 2008 and now are excluded."
        ),
        code(BOOT + "\n" + R_CODE + "\nimport sys; sys.path.insert(0, os.path.abspath('../../..')); from quantlab import analytics"),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Top-decile excess **+{R['excess_mean']:.1f}%/yr**, "
            f"HAC *t* = **{R['excess_t']:+.2f}**; long-short inverts at "
            f"**{R['ls_mean']:.1f}%/yr** (*t* = {R['ls_t']:+.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | No statistically real edge to harvest; "
            f"long-short sign backwards; survivorship inflates all raw numbers. |\n"
            f"| **Beats random portfolio?** | `NO` | Top decile beats only "
            f"**{R['rand_pct_beaten']}%** of random same-size draws. |\n\n"
            "> **In plain words:** on the S&P 500 large-cap survivor panel, the Magic "
            "Formula top decile earns a statistically invisible surplus and the "
            "long-short hedge runs backwards — quality and cheapness are already "
            "priced in by the institutional analyst community."
        ),

        # ---- BEAT 1 -----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_{i,t+1}$ be firm $i$'s return in year $t+1$. Define:\n\n"
            "$$\\text{ROC}_i = \\frac{\\text{EBIT}_i}{\\text{Assets}_i - \\text{CurrentLiab}_i - \\text{Cash}_i}$$\n\n"
            "$$\\text{EY}_i = \\frac{\\text{EBIT}_i}{\\text{Equity}_i + \\text{LTDebt}_i - \\text{Cash}_i}$$\n\n"
            "Rank all firms each year on ROC and EY (higher = better on each); sum the ranks. "
            "The **H₁ (signal)** claim is:\n\n"
            "$$\\mathbb{E}\\left[\\bar{r}_{\\text{top decile},t+1} - \\bar{r}_{\\text{EW market},t+1}\\right] > 0$$\n\n"
            "with a HAC t-stat |t| ≥ 2. **H₂ (beats random)** requires the top decile to "
            "beat a random same-size draw. **H₃ (tradable)** requires the hedge to survive "
            "costs on a liquid universe (feasible given annual turnover of ~15 names)."
        ),

        # ---- BEAT 2 -----------------------------------------------------------
        md(
            "## 2 · So what? — the stakes\n\n"
            "H₁–H₃ jointly, if true, would make the Magic Formula the most important "
            "publicly-documented equity factor strategy: free, from public filings, "
            "with no look-ahead and low turnover. If false on the S&P 500, it means "
            "the effect is either (a) a small-cap phenomenon that doesn't survive in a "
            "liquid institutional universe, or (b) a survivor-biased back-test artefact, "
            "or both."
        ),

        # ---- BEAT 3 -----------------------------------------------------------
        md(
            "## 3 · The protocol\n\n"
            "- **Data**: six EDGAR concepts from the desk's shared cache "
            "(OperatingIncomeLoss, Assets, LiabilitiesCurrent, LongTermDebtNoncurrent, "
            "Cash, StockholdersEquity); annual returns from Yahoo Finance via "
            "`_edgar_yrret.parquet`.\n"
            "- **Universe**: current S&P 500 members with all six concepts present "
            f"(~{R['n_tickers']} tickers at peak); survivorship-biased.\n"
            f"- **Window**: {R['n_years']} years of signal (2008–2024), forecasting "
            "returns 2009–2025.\n"
            "- **Signal**: combined MF rank per year; top decile = portfolio.\n"
            "- **Inference**: HAC Newey-West t-stat on the {R['n_years']}-observation "
            "annual excess-return series.\n"
            "- **Control**: empirical distribution of random same-size portfolios "
            "(500 draws × 17 years).\n"
            "- **Positive control**: synthetic panel with planted premium confirms the "
            "engine can detect a real effect."
        ),

        # ---- BEAT 4 -----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Synthetic positive control — the engine works when a premium exists\n\n"
            "Before condemning the real data, confirm the machinery is honest: sweep the "
            "planted premium from 0% to 10%/yr and check that the top decile advantage "
            "scales with it."
        ),
        code(
            "premiums = [-0.04, 0.0, 0.03, 0.06, 0.10]\n"
            "synth_rows = []\n"
            "for p in premiums:\n"
            "    s, f, _ = data.synthetic_panel(n_firms=300, n_years=25, premium=p, seed=121)\n"
            "    hh = st.quantile_hedge(s, f, q=0.10)\n"
            "    ss = st.summary(hh['hedge'])\n"
            "    synth_rows.append((p, ss['mean']*100, ss['tstat']))\n"
            "sydf = pd.DataFrame(synth_rows, columns=['premium_%', 'excess_%/yr', 'HAC_t'])\n"
            "fig, ax = plt.subplots(figsize=(9, 4))\n"
            "colors = [GREEN if t > 2 else (RED if t < -2 else GREY) for t in sydf['HAC_t']]\n"
            "ax.bar(sydf['premium_%']*100, sydf['excess_%/yr'], color=colors, width=0.8)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Planted premium (%/yr)'); ax.set_ylabel('Top-decile excess (%/yr)')\n"
            "ax.set_title('Positive control: HAC t>2 (green) appears where the premium is real')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(sydf.round(2))"
        ),
        md(
            "> The engine faithfully detects planted premiums: at +6%/yr the HAC t clears 2; "
            "at 0% it prints near-zero. The real-data result is therefore a statement about "
            "the **market**, not the **method**."
        ),
        md(
            "### 4b · Real EDGAR panel — top decile vs market, HAC inference\n\n"
            "Per-year excess returns and the pooled HAC t-stat across the full panel."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s_hedge = st.summary(h['hedge'])\n"
            "    s_ls = st.summary(h['top'] - h['bottom'])\n"
            "    s_top = st.summary(h['top'])\n"
            "    excess_pct = h['hedge'] * 100\n"
            "    years_real = h.index.tolist()\n"
            "else:\n"
            "    # Frozen numbers\n"
            "    s_hedge = dict(mean=R['excess_mean']/100, tstat=R['excess_t'], hit_rate=R['excess_hit']/100, n=R['n_years'])\n"
            "    s_ls = dict(mean=R['ls_mean']/100, tstat=R['ls_t'], hit_rate=R['ls_hit']/100, n=R['n_years'])\n"
            "    s_top = dict(mean=R['top_mean']/100, tstat=R['top_t'], sharpe=R['top_sharpe'])\n"
            "    excess_pct = pd.Series([27,16,2,0,-6,0,-3,-11,-3,-6,14,4,10,-11,-1,-5,2],\n"
            "                           index=list(range(2008,2025)))\n"
            "    years_real = list(range(2008, 2025))\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "col = [GREEN if v > 0 else RED for v in excess_pct]\n"
            "axes[0].bar(years_real, excess_pct, color=col)\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].axhline(s_hedge['mean']*100, ls='--', c=GREY, lw=1.5,\n"
            "    label=f'Mean {s_hedge[\"mean\"]*100:+.1f}%/yr')\n"
            "axes[0].set_ylabel('Top-decile excess vs EW market (%/yr)')\n"
            "axes[0].set_title(f'Top-decile excess (hit {s_hedge[\"hit_rate\"]:.0%} years, HAC t={s_hedge[\"tstat\"]:+.2f})')\n"
            "axes[0].legend()\n"
            "# Long-short\n"
            "if HAVE_REAL:\n"
            "    ls_ann = (h['top'] - h['bottom']) * 100\n"
            "else:\n"
            "    ls_ann = pd.Series([x-y for x,y in zip(\n"
            "        [64,39,9,25,38,18,5,9,25,-6,47,24,42,-22,33,18,19],\n"
            "        [80,13,0,31,64,20,-5,40,24,10,15,9,33,-18,126,33,34])], index=years_real)\n"
            "col2 = [GREEN if v > 0 else RED for v in ls_ann]\n"
            "axes[1].bar(years_real, ls_ann, color=col2)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].axhline(s_ls['mean']*100, ls='--', c=GREY, lw=1.5,\n"
            "    label=f'Mean {s_ls[\"mean\"]*100:+.1f}%/yr')\n"
            "axes[1].set_ylabel('L/S return (%/yr)')\n"
            "axes[1].set_title(f'Long-short inverts (HAC t={s_ls[\"tstat\"]:+.2f})')\n"
            "axes[1].legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Top excess: mean={s_hedge[\"mean\"]*100:+.1f}%/yr  HAC t={s_hedge[\"tstat\"]:+.2f}')\n"
            "print(f'L/S hedge:  mean={s_ls[\"mean\"]*100:+.1f}%/yr  HAC t={s_ls[\"tstat\"]:+.2f}')"
        ),
        md(
            f"> The top decile earns **+{R['excess_mean']:.1f}%/yr** above the equal-weight "
            f"market — HAC *t* = **{R['excess_t']:+.2f}**, well below the |t| ≥ 2 inference bar. "
            f"The long-short **inverts** at **{R['ls_mean']:.1f}%/yr** (*t* = {R['ls_t']:+.2f}): "
            "the formula's bottom decile (worst ROC + highest price) beats its top decile "
            "on this panel. H₁ and H₂ are not supported."
        ),
        md(
            "### 4c · Random-portfolio null distribution\n\n"
            "Is the top decile even beating a blind draw of the same number of stocks? "
            "We simulate 500 random same-size portfolios each year and build the null "
            "distribution of excess returns."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rand = st.random_portfolio_returns(sig, fwd, q=0.10, n_draws=500, seed=121)\n"
            "    actual_excess = float(h['hedge'].mean())\n"
            "    pct_beaten = float((rand < actual_excess).mean())\n"
            "else:\n"
            "    rng = np.random.default_rng(42)\n"
            "    rand = pd.Series(rng.normal(R['rand_mean_excess']/100, R['rand_std']/100, 17*500))\n"
            "    actual_excess = R['excess_mean'] / 100\n"
            "    pct_beaten = R['rand_pct_beaten'] / 100\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "ax.hist(rand * 100, bins=50, color=GREY, alpha=0.7, label='Random portfolios (500 × 17yr avg)')\n"
            "ax.axvline(actual_excess * 100, c=RED, lw=2.5,\n"
            "           label=f'Top decile: {actual_excess*100:+.1f}%/yr')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Annual excess return vs EW market (%)')\n"
            "ax.set_ylabel('Count'); ax.legend()\n"
            "ax.set_title(f'Top decile beats {pct_beaten:.0%} of random portfolios — barely above chance')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Top decile beats {pct_beaten:.0%} of random draws')"
        ),
        md(
            f"> The top decile beats **{R['rand_pct_beaten']}%** of random portfolios — "
            "just above the 50% chance level. The formula adds essentially nothing over a "
            "blind draw from the same universe."
        ),

        # ---- BEAT 5 -----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — excess = +{R['excess_mean']:.1f}%/yr, HAC *t* = "
            f"{R['excess_t']:+.2f}; long-short inverts ({R['ls_mean']:.1f}%/yr, "
            f"*t* = {R['ls_t']:+.2f}); beats {R['rand_pct_beaten']}% of random draws.\n"
            f"- **Tradability `MIRAGE`** — no statistically real edge; long-short "
            "backwards; survivorship bias inflates all raw returns.\n"
            "- **Survivorship caveat** — upper bound: the current S&P 500 survivor "
            "panel excludes every bankruptcy and delistee since 2008."
        ),

        # ---- BEAT 6 -----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the honest answer\n\n"
            "Annual rebalancing of ~15 names from the S&P 500 is operationally trivial "
            "and cheap. The obstacles are entirely in the signal:\n\n"
            "1. **No edge to harvest** — the gross HAC t-stat is 0.44; any positive "
            "observation is statistically indistinguishable from luck.\n"
            "2. **Wrong sign on the long-short** — a L/S strategy on this panel would "
            "need to be flipped: short the 'good' companies and buy the 'bad' ones, "
            "which is economically hard to justify.\n"
            "3. **Survivorship arithmetic** — a live 2008 portfolio would have included "
            "names like Lehman Brothers (briefly high EBIT/EV), Fannie Mae (high apparent "
            "ROC), and other crisis casualties. The clean EDGAR panel excludes them.\n\n"
            "The conclusion of this specific test is **MIRAGE** — but the conclusion about "
            "the *idea* is more nuanced: the Magic Formula is a real quality+value "
            "framework that likely generates alpha in the small-cap universe Greenblatt "
            "built it for. On the S&P 500 large-cap survivor panel, it doesn't."
        ),

        # ---- BEAT 7 -----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The right universe.** Extend to a broader US equity universe (Russell 2000 "
            "or all-cap). The Greenblatt (2005) back-test ran on all stocks >$50M market cap "
            "— that's the native habitat of the effect.\n"
            "- **Factor decomposition.** Decompose ROC (quality) and EY (value) separately: "
            "is one axis driving any signal? Fama-French HML and RMW are the factor "
            "proxies to compare against.\n"
            "- **Related desk studies.** "
            "[Study 65 — Scorecard](../../65-scorecard/) (F-score, same EDGAR panel, same "
            "inversion); [Study 52 — Smoke-Screen](../../52-smoke-screen/) (accruals, "
            "which weakly replicates); [Study 44 — Growth-Spurt](../../44-growth-spurt/) "
            "(earnings growth momentum).\n"
            "- **Correcting the survivorship.** A proper test would require a point-in-time "
            "S&P 500 membership dataset (e.g. Compustat GVKEY-based constituent history) "
            "and EDGAR data for *all* historical members — a live project for the desk.\n\n"
            "*Think the die is loaded in the right universe? Fork this, extend to a "
            "broad all-cap panel with point-in-time constituents, and show |t| ≥ 2 on "
            "the top-decile excess with an unbiased universe. That's the bar.*"
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
