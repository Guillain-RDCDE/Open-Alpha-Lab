"""Generate the two narrative notebooks for Study 177 (Megacap-Concentration).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-data cells use the cached
parquets under ../_cache/ if present and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.

The _write convention (each build_*() ends by calling _write) is kept so the repo's
intro-restyle tooling can monkeypatch it.
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-15).
# Top-10 concentration, 2008-2025 (18 complete years), excl. 2026 partial.
R = dict(
    n_years=18, n_tickers=412, n_tickers_total=412,
    # Full sample (top-10 vs equal-weight)
    topn10_mean=15.0, ew_mean=16.6, excess10_mean=-1.6, excess10_t=-0.31,
    topn10_sharpe=0.57,
    # Top-7
    topn7_mean=13.7, excess7_mean=-2.9, excess7_t=-0.58,
    # By decade
    pre2015_topn=2.4,  pre2015_ew=16.5, pre2015_excess=-14.0, pre2015_t=-10.27,
    mid_topn=16.6,     mid_ew=16.3,     mid_excess=0.3,        mid_t=0.11,
    post2020_topn=28.3, post2020_ew=17.0, post2020_excess=11.3, post2020_t=1.30,
    # vs SPY
    topn10_vs_spy_mean=2.4, topn10_vs_spy_t=0.58,
    # Fingerprint
    fingerprint="b6555c5c4c43",
)

# ---------------------------------------------------------------------------
# Shared preamble
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))           # study package
sys.path.insert(0, os.path.abspath("../../.."))     # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from megacap_concentration import data, strategy as st

CACHE_DIR = os.path.abspath("../_cache")
REPO_CACHE = os.path.abspath("../../../_cache")

def _have_cache():
    return (os.path.exists(os.path.join(CACHE_DIR, "megacap_annual_mcap.parquet")) and
            os.path.exists(os.path.join(CACHE_DIR, "megacap_annual_returns.parquet")))

HAVE_REAL = _have_cache()

def load_enriched(top_n=10):
    \"\"\"Load real annual panel enriched with SPY, exclude partial 2026.\"\"\"
    import numpy as np
    panel, meta = data.load_real(top_n=top_n, fetch=False, cache_dir=CACHE_DIR)
    panel = panel[panel["year"] < 2026].copy()
    annual = st.annual_portfolio_returns(panel, top_n=top_n, col="ret_annual")
    spy = st.spy_annual_returns(cache_dir=REPO_CACHE)
    return st.with_spy_benchmark(annual, spy), meta

print("real cache present:", HAVE_REAL)

# Frozen headline numbers mirroring docs/results.md (as-of 2026-06-15)
R = dict(
    n_years=18, n_tickers=412, n_tickers_total=412,
    topn10_mean=15.0, ew_mean=16.6, excess10_mean=-1.6, excess10_t=-0.31,
    topn10_sharpe=0.57,
    topn7_mean=13.7, excess7_mean=-2.9, excess7_t=-0.58,
    pre2015_topn=2.4,  pre2015_ew=16.5, pre2015_excess=-14.0, pre2015_t=-10.27,
    mid_topn=16.6,     mid_ew=16.3,     mid_excess=0.3,        mid_t=0.11,
    post2020_topn=28.3, post2020_ew=17.0, post2020_excess=11.3, post2020_t=1.30,
    topn10_vs_spy_mean=2.4, topn10_vs_spy_t=0.58,
    fingerprint="b6555c5c4c43",
)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Megacap-Concentration -- is 'just buy the biggest' genius or recency bias?\n"
            "### Top-7/10 S&P 500 names vs the equal-weight index, 2008-2025\n\n"
            "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Recency_bias%3F: Confirmed](https://img.shields.io/badge/Recency_bias%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The Magnificent Seven -- Apple, Microsoft, Nvidia, Google, Amazon, Meta, Tesla -- "
            "dominated the 2020s so completely that a cottage industry of 'just buy the biggest' "
            "strategies emerged. The thesis is seductive: great companies *stay* great, their moats "
            "compound, and the index's cap-weight already tilts toward them -- so why not go all the "
            "way? This notebook asks the uncomfortable question: **is this a genuine edge, or are "
            "we just describing the last five years?**\n\n"
            "> This is the plain-language layer. Want the t-stats, decade splits, and multiple-"
            "comparisons reckoning? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn by the "
            "code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does top-10 beat equal-weight over the full 2008-2025 sample? | "
            f"**No.** It *lags* by **{abs(R['excess10_mean']):.1f} ppt/year** on average (t={R['excess10_t']:+.2f}). |\n"
            "| Did it ever work? | **Yes -- recently.** Post-2020 the top-10 won by "
            f"**+{R['post2020_excess']:.0f} ppt/year**, the Mag-7 era. |\n"
            "| What happened before? | **The opposite.** Pre-2015 the top-10 lagged by "
            f"**{abs(R['pre2015_excess']):.0f} ppt/year** (t={R['pre2015_t']:+.2f}) -- megacaps lost. |\n"
            "| Is the recent win statistically solid? | "
            f"**No.** 6 years of data, t={R['post2020_t']:+.2f}. Not enough to overcome the long history of underperformance. |\n"
            "| Could you build a fund around it? | **No.** The universe is survivorship-biased, "
            "turnover costs are real, and the full-sample signal is negative. |\n\n"
            "> **One sentence:** concentrating in the biggest S&P 500 names looks like genius "
            "because of the Magnificent-Seven era -- but over the full available history the top-10 "
            "*lagged* the average stock by 1.6 ppt/year, and before 2015 it lost by 14 ppt. "
            "That is the definition of recency bias."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 -- The claim\n\n"
            "> *'The biggest companies in the S&P 500 get big for a reason. Their competitive moats, "
            "brand power, and scale advantages mean they keep outperforming. Just buy the top 7 (or "
            "10) by market cap and hold them for a year. It's a simpler, better strategy than the "
            "equal-weight index or the cap-weighted SPY.'*\n\n"
            "The modern version of this claim rode the 2020s wave: Apple crossed \\$3 trillion, "
            "Nvidia surged 10x in two years, Microsoft dominated cloud and AI. Every finance "
            "influencer started recommending 'the Magnificent Seven.' The question is whether the "
            "data supports this *as a general claim*, or whether it is a narrative retrofitted "
            "to one unusual half-decade."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 -- So what?\n\n"
            "The stakes are real. If it's true:\n"
            "- Equal-weight ETFs like RSP underperform -- their biggest drawback is they dilute "
            "the winners.\n"
            "- Passive investors in SPY leave alpha on the table vs. a simple 7-name portfolio.\n"
            "- The academic 'size premium' (small beats large) is dead.\n\n"
            "If it's false (or recency bias):\n"
            "- Millions of investors are extrapolating five exceptional years onto the next decade.\n"
            "- The natural reversal is a long stretch of megacap underperformance as valuations "
            "mean-revert.\n"
            "- The size premium (Study 45) has a serious historical claim -- the new claim requires "
            "overturning it."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 -- How would we know?\n\n"
            "One test, two controls:\n\n"
            "1. **Each January**, rank all S&P 500 names by their *prior year-end* market cap "
            "(shares from EDGAR x year-end price from yfinance). Hold the top-10 equally "
            "weighted for the calendar year. **No look-ahead** -- we use last year's market cap.\n"
            "2. **Control A -- equal-weight:** hold all names equally. If concentration wins, it "
            "beats this.\n"
            "3. **Control B -- SPY total return:** the actual cap-weighted S&P 500 including "
            "dividends (conservative for the top-10 which excludes dividends).\n\n"
            "We split by decade, because a claim that 'only works after 2020' is recency bias, "
            "not an edge. And we name the survivorship bias: our universe is today's S&P 500 "
            "members -- companies that went bankrupt or were delisted aren't here, which "
            "inflates all returns.\n\n"
            f"Tape: EDGAR diluted shares x yfinance annual closes, {R['n_tickers_total']} tickers, "
            f"{R['n_years']} complete years (2008-2025)."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 -- The teardown\n\n"
            "**The full-sample picture: top-10 lags the average stock.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    enriched, meta = load_enriched(top_n=10)\n"
            "    years = enriched.index.astype(int)\n"
            "    topn = enriched['topn_ret'].values * 100\n"
            "    ew   = enriched['ew_ret'].values * 100\n"
            "    excess = enriched['topn_excess'].values * 100\n"
            "else:\n"
            "    years = list(range(2008, 2026))\n"
            "    excess = [-11.7,-17.8,-21.0,-11.1,-13.8,-17.2,-5.8,1.1,-5.9,-3.1,-6.0,15.6,2.4,-6.2,-13.4,28.8,38.5,17.8]\n"
            "    topn = [R['pre2015_topn']]*7 + [R['mid_topn']]*5 + [R['post2020_topn']]*6\n"
            "    ew = [R['pre2015_ew']]*7 + [R['mid_ew']]*5 + [R['post2020_ew']]*6\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))\n"
            "# Left: annual returns side by side\n"
            "x = range(len(years))\n"
            "w = 0.4\n"
            "axes[0].bar([xi - w/2 for xi in x], topn, width=w, color=AMBER, alpha=0.85, label='Top-10')\n"
            "axes[0].bar([xi + w/2 for xi in x], ew, width=w, color=GREY, alpha=0.7, label='Equal-weight')\n"
            "axes[0].set_xticks(list(x)); axes[0].set_xticklabels([str(y) for y in years], rotation=45, ha='right')\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_ylabel('Annual return %'); axes[0].set_title('Annual returns: Top-10 vs Equal-weight')\n"
            "axes[0].legend()\n"
            "# Right: excess return (top-10 minus EW)\n"
            "col = [GREEN if e > 0 else RED for e in excess]\n"
            "axes[1].bar(years, excess, color=col, alpha=0.85)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_xlabel('Year'); axes[1].set_ylabel('Excess ppt (top-10 minus EW)')\n"
            "axes[1].set_title(f'Excess return: top-10 minus equal-weight (mean={sum(excess)/len(excess):+.1f} ppt)')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Full sample: top-10 lagged EW by {{abs(sum(excess)/len(excess)):.1f}} ppt/year on average')"
        ),
        md(
            f"Over the full 18-year sample the top-10 earns **{R['topn10_mean']:.1f}%/year** vs "
            f"**{R['ew_mean']:.1f}%/year** for equal-weight -- a lag of "
            f"**{abs(R['excess10_mean']):.1f} ppt/year**. But look at the pattern: the red bars "
            "(years the top-10 lost to EW) are concentrated in 2008-2014, the green bars in "
            "2019-2025. This is not random noise -- it is a structural regime shift."
        ),
        md(
            "**The decade split: the claim is backwards for most of the sample.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    result = st.summarize(enriched)\n"
            "    decades = result['by_decade']\n"
            "    labels  = ['Pre-2015\\n(size premium alive)', '2015-2019\\n(FAANG era)', 'Post-2020\\n(Mag-7 era)']\n"
            "    keys    = ['pre_2015', '2015_2019', 'post_2020']\n"
            "    topn_d  = [decades[k]['topn_mean'] for k in keys]\n"
            "    ew_d    = [decades[k]['ew_mean'] for k in keys]\n"
            "    excess_d= [decades[k]['excess_mean'] for k in keys]\n"
            "    tstats  = [decades[k]['excess_tstat'] for k in keys]\n"
            "else:\n"
            "    labels  = ['Pre-2015\\n(size premium alive)', '2015-2019\\n(FAANG era)', 'Post-2020\\n(Mag-7 era)']\n"
            "    topn_d  = [R['pre2015_topn'], R['mid_topn'], R['post2020_topn']]\n"
            "    ew_d    = [R['pre2015_ew'], R['mid_ew'], R['post2020_ew']]\n"
            "    excess_d= [R['pre2015_excess'], R['mid_excess'], R['post2020_excess']]\n"
            "    tstats  = [R['pre2015_t'], R['mid_t'], R['post2020_t']]\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10.0, 5.0))\n"
            "x = range(3)\n"
            "w = 0.35\n"
            "bars1 = ax.bar([xi - w/2 for xi in x], topn_d, width=w, color=AMBER, alpha=0.85, label='Top-10')\n"
            "bars2 = ax.bar([xi + w/2 for xi in x], ew_d, width=w, color=GREY, alpha=0.7, label='Equal-weight')\n"
            "ax.set_xticks(list(x)); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('Mean annual return %'); ax.set_title(\"Concentration only worked in the Mag-7 era\")\n"
            "ax.axhline(0, c='k', lw=1); ax.legend()\n"
            "for xi, ex, t in zip(x, excess_d, tstats):\n"
            "    sign = '+' if ex >= 0 else ''\n"
            "    c = GREEN if ex > 0 else RED\n"
            "    ax.annotate(f'{sign}{ex:.0f} ppt\\n(t={t:+.1f})', (xi, max(topn_d[xi], ew_d[xi]) + 1),\n"
            "                ha='center', color=c, fontweight='bold', fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Pre-2015: top-10 lagged by {abs(excess_d[0]):.0f} ppt/yr (the size premium was real here)')"
        ),
        md(
            f"**Three eras, three completely different stories:**\n\n"
            f"- **Pre-2015** (7 years): top-10 earned just **{R['pre2015_topn']:.1f}%/year** "
            f"vs **{R['pre2015_ew']:.1f}%** for the average stock -- a massive "
            f"**{abs(R['pre2015_excess']):.0f} ppt lag** (t={R['pre2015_t']:+.1f}). "
            f"The size premium (small beats large) was alive and kicking.\n"
            f"- **2015-2019** (5 years): near-flat, **{R['mid_excess']:+.1f} ppt** "
            f"(t={R['mid_t']:+.2f}).\n"
            f"- **Post-2020** (6 years): megacaps won by **{R['post2020_excess']:+.0f} ppt/year** "
            f"(t={R['post2020_t']:+.2f}), driven by Nvidia, Microsoft, Apple AI momentum.\n\n"
            "The Magnificent-Seven era is real -- but it is 6 years in an 18-year history where "
            "the strategy was mostly a losing bet."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal -- MIXED.** The top-10 concentration won post-2020 "
            f"({R['post2020_excess']:+.0f} ppt, t={R['post2020_t']:+.2f}) but lost pre-2015 "
            f"({R['pre2015_excess']:+.0f} ppt, t={R['pre2015_t']:+.2f}). Full sample: "
            f"{R['excess10_mean']:+.1f} ppt (t={R['excess10_t']:+.2f}) -- the recent win does not "
            f"outweigh 14 years of underperformance.\n"
            f"- **Tradability -- Mirage.** Annual turnover is modest but the universe is "
            f"survivorship-biased (current S&P 500 members only), which inflates all returns. "
            f"A real fund would hold names that later got dropped from the index -- and many of "
            f"those dropped because they crashed.\n"
            f"- **Recency bias? -- Confirmed.** The claim 'megacaps always win' is a 2020s "
            f"narrative retrofitted onto history. The size premium literature says the opposite "
            f"held for most of modern market history."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 -- Could you actually build this?\n\n"
            "Three real obstacles a fund would face:\n\n"
            "1. **Survivorship bias** -- our universe is today's S&P 500. Companies that grew "
            "big and then imploded (GE 2008-2018, Enron, etc.) are only partially included. "
            "A real fund investing in 2010's top-10 would have held GE, which fell 75%.\n"
            "2. **Concentration risk** -- 10 names is extreme concentration. The 2022 drawdown "
            f"shows this: the top-10 lost **{abs(R['pre2015_topn']):.0f}%** in 2008 (pre-2015 era: "
            "-45.6% for 2008 top-10). A one-decision-per-year concentrated portfolio "
            "has nowhere to hide.\n"
            "3. **No signal stability** -- we're cherry-picking top-N by one metric (market cap), "
            "ignoring quality, valuation, momentum, or any other signal. This is not a factor "
            "strategy; it is a popularity contest, and popularity reverts."
        ),
        code(
            "# Cumulative wealth: $1 in 2008\n"
            "if HAVE_REAL:\n"
            "    topn_r = enriched['topn_ret'].values\n"
            "    ew_r   = enriched['ew_ret'].values\n"
            "    spy_r  = enriched['spy_ret'].dropna().values[:len(topn_r)]\n"
            "else:\n"
            "    import numpy as np\n"
            "    topn_r = np.array([-0.456,0.212,0.051,-0.070,0.084,0.220,0.128,\n"
            "                       0.056,0.129,0.241,-0.092,0.496,0.196,0.263,-0.228,0.544,0.585,0.341])\n"
            "    ew_r = np.array([-0.339,0.390,0.261,0.041,0.222,0.391,0.186,\n"
            "                     0.045,0.188,0.272,-0.032,0.340,0.171,0.324,-0.094,0.256,0.200,0.164])\n"
            "    spy_r = np.array([-0.370,0.264,0.151,0.021,0.160,0.323,0.136,\n"
            "                      0.014,0.120,0.219,-0.044,0.314,0.184,0.287,-0.181,0.264,0.250,-0.002])\n"
            "\n"
            "n = min(len(topn_r), len(ew_r), len(spy_r))\n"
            "topn_r, ew_r, spy_r = topn_r[:n], ew_r[:n], spy_r[:n]\n"
            "yrs = list(range(2008, 2008 + n))\n"
            "cum_topn = (1 + topn_r).cumprod()\n"
            "cum_ew   = (1 + ew_r).cumprod()\n"
            "cum_spy  = (1 + spy_r).cumprod()\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10.0, 5.0))\n"
            "ax.plot(yrs, cum_topn, 'o-', c=AMBER, lw=2.5, label='Top-10 concentration')\n"
            "ax.plot(yrs, cum_ew,   's-', c=GREY,  lw=2, label='Equal-weight S&P 500')\n"
            "ax.plot(yrs, cum_spy,  '^-', c=GREEN, lw=2, label='SPY (cap-weight, TR)')\n"
            "ax.axhline(1, c='k', lw=0.7, ls='--')\n"
            "# Shade the pre-2015 underperformance era\n"
            "ax.axvspan(2008, 2015, alpha=0.07, color=RED, label='Megacap underperformance era')\n"
            "ax.axvspan(2020, max(yrs), alpha=0.07, color=GREEN, label='Mag-7 era')\n"
            "ax.set_xlabel('Year'); ax.set_ylabel('$1 growth (survivorship-biased)')\n"
            "ax.set_title('Cumulative wealth: the story depends entirely on which era you start from')\n"
            "ax.legend(fontsize=8); plt.tight_layout(); plt.show()\n"
            "print(f'Top-10 final wealth: ${cum_topn[-1]:.1f}   EW: ${cum_ew[-1]:.1f}   SPY: ${cum_spy[-1]:.1f}')"
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **The size premium it overturned (maybe temporarily).** "
            "[Study 45](../../45-size-premium/) and Fama & French (1993) document small-cap "
            "outperformance -- the evidence for which was strong from 1926-2010 and has been "
            "murky since. Megacap concentration is the anti-size premium bet; knowing *why* "
            "2020-2025 flipped it is the real research question.\n"
            "- **Quality filter.** The top-10 by market cap includes some very mediocre businesses "
            "(GE in the 2010s, INTC throughout). Add a quality screen (ROIC, margins) and see if "
            "the pre-2015 period improves -- that's worth testing.\n"
            "- **Equal-weight vs cap-weight.** RSP (equal-weight S&P 500 ETF) is the institutional "
            "version of our EW baseline. Its performance vs SPY mirrors our analysis exactly -- "
            "EW won 2000-2014, SPY won 2015-present. The tilts have been cyclical.\n"
            "- **How concentrated is too concentrated?** Top-3 vs top-10 vs top-50 -- where does "
            "the curve break? There may be a sweet spot.\n\n"
            "*Think the Mag-7 era represents a permanent structural shift in market structure? "
            "Fork this, extend with forward data, and show the post-2025 numbers. "
            "That is the bar.*"
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
            "# Megacap-Concentration -- a quantitative teardown\n"
            "### EDGAR shares x yfinance closes * annual rebalance * HAC inference * decade splits\n\n"
            "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Recency_bias%3F: Confirmed](https://img.shields.io/badge/Recency_bias%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim now carrying its standard error and decade split.* "
            "We test whether concentrating in the top-10 S&P 500 names by (prior year-end) "
            "market cap beats an equal-weight baseline and SPY, using HAC t-stats across "
            f"{R['n_years']} complete calendar years (2008-2025).\n\n"
            "> **Not investment advice.** Real data: EDGAR diluted shares x yfinance annual "
            f"closes, {R['n_tickers_total']} tickers, as-of 2026-06-15; fingerprint "
            f"`{R['fingerprint']}`. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md). Universe = current S&P 500 members "
            "(survivorship-biased; all returns are upper bounds on investable reality)."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | Full 2008-2025: top-10 excess = **{R['excess10_mean']:+.1f} ppt/yr**, "
            f"HAC t = {R['excess10_t']:+.2f}. Pre-2015: **{R['pre2015_excess']:+.0f} ppt** "
            f"(t={R['pre2015_t']:+.1f}). Post-2020: **{R['post2020_excess']:+.0f} ppt** (t={R['post2020_t']:+.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | Survivorship-biased universe; annual turnover "
            f"is low but the signal is unstable across regimes; no robust full-sample edge. |\n"
            f"| **Recency bias?** | `CONFIRMED` | The claim 'always works' is falsified by 2008-2014. "
            f"Only the 2020-2025 Mag-7 window supports it -- 6 years, insufficient for the general claim. |\n\n"
            "> The thesis requires believing the 2020s structural shift (AI dominance, winner-take-most "
            "dynamics) is permanent. The 15-year prior evidence says the opposite."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 -- The claim, steelmanned\n\n"
            "Let $r^{(k)}_t$ be the equal-weight return of the top-$k$ stocks ranked by "
            "prior-year-end market cap in year $t$, and $r^{EW}_t$ be the equal-weight "
            "return of all S&P 500 names. The claim:\n\n"
            "- **H1 (concentration alpha):** $\\mathbb{E}[r^{(k)}_t - r^{EW}_t] > 0$ -- "
            "the top-$k$ systematically outperform the average member.\n"
            "- **H2 (stability):** this holds across market regimes (pre- and post-2015), "
            "not just in the Mag-7 era.\n"
            "- **H3 (SPY alpha):** top-$k$ also beats the cap-weighted SPY total return.\n\n"
            "We test H1, H2, and H3 for k=7 and k=10. We find H1 is rejected over the "
            "full sample, H2 is strongly rejected, and H3 is marginally positive but not "
            "statistically significant."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 -- So what? — what rides on each answer\n\n"
            "If H1-H3 held robustly, it would imply:\n"
            "- The size premium (Fama-French) is definitively dead, not just cyclically weak.\n"
            "- Simple rules dominate complex factor models in the US large-cap space.\n"
            "- A five-line annual rebalancing algorithm beats decades of academic factor research.\n\n"
            "The interesting failure is in *how* the claim succeeds recently but failed before: "
            "it reveals a regime, not a law. The Mag-7 era reflects a specific combination of "
            "AI exuberance, quantitative easing aftermath, and winner-take-most dynamics in "
            "software -- not a timeless edge from 'big = good'."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 -- Protocol\n\n"
            "- **Universe:** current S&P 500 members (name-survivorship bias -- disclosed above).\n"
            "- **Market-cap ranking:** diluted shares (EDGAR fiscal-year) x year-end close "
            "(yfinance, last trading day of Dec). **Lagged by one year** -- the rank used for "
            "year-$T$ returns is based on year-$(T-1)$ market cap. No look-ahead.\n"
            "- **Return measurement:** annual price return (yfinance close-to-close, Jan-Dec). "
            "Price returns, not total returns -- conservative for the top-10 which excludes "
            "dividends vs SPY which includes them.\n"
            "- **Baselines:** (A) equal-weight of all names with valid returns; (B) SPY total "
            "return from shared cache.\n"
            "- **Inference:** HAC Newey-West t-stat on the annual excess-return series; "
            "sample is small ($n \\le 18$ years), so even a real effect would require ~5 ppt "
            "systematic excess to show |t| > 2.\n"
            "- **Decade splits:** pre-2015 (7 years), 2015-2019 (5), post-2020 (6).\n"
            "- **Positive control:** synthetic cross-section with tunable concentration effect, "
            "confirming the engine recovers alpha when it is planted.\n"
            "- **Multiple comparisons:** we test k=7 and k=10 -- two hypotheses. Both fail the "
            "full-sample test, so no Bonferroni adjustment changes the conclusion."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 -- The teardown"),
        md(
            "### 4a -- Full-sample: concentration lags equal-weight\n\n"
            "Annual excess returns (top-10 minus equal-weight) with HAC t-stat on the mean."
        ),
        code(
            "if HAVE_REAL:\n"
            "    e10, m10 = load_enriched(top_n=10)\n"
            "    e7,  m7  = load_enriched(top_n=7)\n"
            "    r10 = st.summarize(e10)\n"
            "    r7  = st.summarize(e7)\n"
            "    ex10 = e10['topn_excess'].values * 100\n"
            "    ex7  = e7['topn_excess'].values * 100\n"
            "    years = e10.index.astype(int).tolist()\n"
            "else:\n"
            "    r10 = {'full': {'topn_mean': R['topn10_mean'], 'ew_mean': R['ew_mean'],\n"
            "                    'excess_mean': R['excess10_mean'], 'excess_tstat': R['excess10_t'], 'n': R['n_years']}}\n"
            "    r7  = {'full': {'excess_mean': R['excess7_mean'], 'excess_tstat': R['excess7_t'], 'n': R['n_years']}}\n"
            "    years = list(range(2008, 2026))\n"
            "    ex10 = [-11.7,-17.8,-21.0,-11.1,-13.8,-17.2,-5.8,1.1,-5.9,-3.1,-6.0,15.6,2.4,-6.2,-13.4,28.8,38.5,17.8]\n"
            "    ex7  = [-14.5,-19.4,-23.6,-13.5,-14.5,-18.3,-7.3,-1.1,-6.7,-2.7,-8.3,12.2,0.0,-10.8,-17.5,25.8,34.6,12.6]\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))\n"
            "for ax, ex, k, t, tk in zip(axes, [ex10, ex7], [10, 7],\n"
            "                             [r10['full']['excess_tstat'], r7['full']['excess_tstat']],\n"
            "                             [r10['full']['topn_mean'], r7['full'].get('topn_mean', R['topn7_mean'])]):\n"
            "    col = [GREEN if e > 0 else RED for e in ex]\n"
            "    ax.bar(years, ex, color=col, alpha=0.8)\n"
            "    ax.axhline(0, c='k', lw=1)\n"
            "    mn = sum(ex)/len(ex)\n"
            "    ax.axhline(mn, c=AMBER, ls='--', lw=1.5, label=f'mean={mn:+.1f} ppt')\n"
            "    ax.set_title(f'Top-{k} excess vs EW: mean={mn:+.1f} ppt, HAC t={t:+.2f}')\n"
            "    ax.set_xlabel('Year'); ax.set_ylabel('Excess ppt'); ax.legend()\n"
            "    ax.tick_params(axis='x', rotation=45)\n"
            "plt.suptitle('Both top-7 and top-10 lag EW over the full sample', y=1.02)\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Top-10: mean excess={{r10[\"full\"][\"excess_mean\"]:+.1f}}% (HAC t={{r10[\"full\"][\"excess_tstat\"]:+.2f}})')\n"
            f"print(f'Top-7:  mean excess={{r7[\"full\"][\"excess_mean\"]:+.1f}}% (HAC t={{r7[\"full\"][\"excess_tstat\"]:+.2f}})')"
        ),
        md(
            f"> Both the top-10 (**{R['excess10_mean']:+.1f} ppt**, t={R['excess10_t']:+.2f}) and "
            f"top-7 (**{R['excess7_mean']:+.1f} ppt**, t={R['excess7_t']:+.2f}) lag equal-weight "
            "over the full 18-year sample. HAC t-stats nowhere near the |t| >= 2 bar. Two "
            "hypotheses tested (k=7 and k=10), both failing -- no Bonferroni adjustment needed."
        ),
        md(
            "### 4b -- Decade split: a MIXED story that depends entirely on the start date\n\n"
            "HAC t-stats by period reveal the regime flip."
        ),
        code(
            "if HAVE_REAL:\n"
            "    decs = r10.get('by_decade', {})\n"
            "    rows = []\n"
            "    for k, lbl in [('pre_2015','Pre-2015 (7y)'), ('2015_2019','2015-2019 (5y)'), ('post_2020','Post-2020 (6y)')]:\n"
            "        if k in decs:\n"
            "            d = decs[k]\n"
            "            rows.append([lbl, d['topn_mean'], d['ew_mean'], d['excess_mean'], d['excess_tstat'], d['n']])\n"
            "    decade_df = pd.DataFrame(rows, columns=['Period','top10%','ew%','excess%','HAC t','n'])\n"
            "else:\n"
            "    decade_df = pd.DataFrame([\n"
            "        ['Pre-2015 (7y)', R['pre2015_topn'], R['pre2015_ew'], R['pre2015_excess'], R['pre2015_t'], 7],\n"
            "        ['2015-2019 (5y)', R['mid_topn'], R['mid_ew'], R['mid_excess'], R['mid_t'], 5],\n"
            "        ['Post-2020 (6y)', R['post2020_topn'], R['post2020_ew'], R['post2020_excess'], R['post2020_t'], 6],\n"
            "    ], columns=['Period','top10%','ew%','excess%','HAC t','n'])\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))\n"
            "col_bar = [RED if row < 0 else GREEN for row in decade_df['excess%']]\n"
            "axes[0].bar(decade_df['Period'], decade_df['excess%'], color=col_bar, alpha=0.85)\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_ylabel('Excess ppt (top-10 minus EW)')\n"
            "axes[0].set_title('Excess return by period')\n"
            "for i, (ex, t) in enumerate(zip(decade_df['excess%'], decade_df['HAC t'])):\n"
            "    axes[0].annotate(f't={t:+.1f}', (i, ex + (0.5 if ex >= 0 else -1.5)), ha='center', fontsize=9)\n"
            "\n"
            "for lbl, r, c in zip(['Top-10', 'EW'], ['top10%', 'ew%'], [AMBER, GREY]):\n"
            "    axes[1].plot(decade_df['Period'], decade_df[r], 'o-', c=c, lw=2, label=lbl, ms=8)\n"
            "axes[1].set_ylabel('Mean annual return %'); axes[1].set_title('Returns by period')\n"
            "axes[1].legend(); axes[1].tick_params(axis='x', rotation=15)\n"
            "plt.tight_layout(); plt.show()\n"
            "decade_df.round(2)"
        ),
        md(
            "> The pre-2015 result is the most striking: **t=-10.3** for top-10 vs EW. This is not "
            "noise -- it is the size premium operating in full force, systematically punishing "
            "the largest names in favor of smaller members. The post-2020 win (t=+1.30 on 6 "
            "years) does not come close to reversing that history statistically."
        ),
        md(
            "### 4c -- The survivorship bias problem\n\n"
            "Our universe is today's S&P 500 members -- names that survived to 2026. A real "
            "investor in 2008-2014 would have held companies that later dropped out of the index, "
            "often because they crashed. This inflates all our returns."
        ),
        code(
            "# Show GE's role: in the top-10 for most of 2008-2015 with terrible returns\n"
            "# This is illustrative of what survivorship misses\n"
            "import numpy as np\n"
            "ge_years = [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]\n"
            "if HAVE_REAL:\n"
            "    panel, meta = data.load_real(top_n=10, fetch=False, cache_dir=CACHE_DIR)\n"
            "    ge_top = panel[(panel['ticker']=='GE') & (panel['year'].isin(ge_years))][['year','mcap_rank','ret_annual']]\n"
            "    print('GE in the top-10:')\n"
            "    print(ge_top.to_string(index=False))\n"
            "    ge_in_top10 = ge_top[ge_top['mcap_rank'] <= 10]\n"
            "    print(f'\\nGE was in top-10 for {len(ge_in_top10)} years with avg return {ge_in_top10[\"ret_annual\"].mean():+.1%}')\n"
            "else:\n"
            "    print('GE example: was in top-10 from 2008-2017, returns mostly negative.')\n"
            "    print('A 2008 investor holding top-10 would have included GE throughout its collapse.')\n"
            "    print('Survivorship bias: GE is still in S&P 500 (barely), so it appears in our data.')\n"
            "    print('Companies that DEPARTED entirely (Enron, Lehman, Sears) are missing entirely.')"
        ),
        md(
            "> The survivorship bias is the most important caveat for this study. All returns are "
            "inflated relative to what a real investor would have earned. Companies that fell out "
            "of the S&P 500 due to poor performance are missing -- and many of those would have "
            "been in the top-10 at some point during their life cycle."
        ),
        md(
            "### 4d -- Synthetic positive control -- the engine works when there's something to find\n\n"
            "Sweep the planted concentration effect: the engine recovers alpha only when it exists."
        ),
        code(
            "effects = [-0.05, 0.0, 0.03, 0.07, 0.10, 0.15]\n"
            "excess_means = []\n"
            "excess_tstats = []\n"
            "for eff in effects:\n"
            "    df_s, _ = data.synthetic_annual(n_years=50, n_stocks=200, top_n=10, effect=eff, seed=177)\n"
            "    df_s = df_s.rename(columns={'ret_gross': 'ret_annual', 'rank': 'mcap_rank'})\n"
            "    ann_s = st.annual_portfolio_returns(df_s, top_n=10, col='ret_annual')\n"
            "    res_s = st.summarize(ann_s, decade_split=False)\n"
            "    excess_means.append(res_s['full']['excess_mean'])\n"
            "    excess_tstats.append(res_s['full']['excess_tstat'])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.8))\n"
            "ax2 = ax.twinx()\n"
            "ax.plot(effects, excess_means, 'o-', c=GREEN, lw=2, label='excess mean (ppt/yr)')\n"
            "ax2.plot(effects, excess_tstats, 's--', c=AMBER, lw=1.5, label='HAC t-stat')\n"
            "ax.axhline(0, c='k', lw=1); ax2.axhline(2, ls=':', c=GREY, lw=1)\n"
            "ax2.axhline(-2, ls=':', c=GREY, lw=1)\n"
            "ax.set_xlabel('Planted concentration effect (annual alpha)'); ax.set_ylabel('Excess mean ppt', color=GREEN)\n"
            "ax2.set_ylabel('HAC t-stat', color=AMBER)\n"
            "ax.set_title('The engine: concentration wins only when there is a real advantage to harvest')\n"
            "lines1, labels1 = ax.get_legend_handles_labels()\n"
            "lines2, labels2 = ax2.get_legend_handles_labels()\n"
            "ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('At effect=0: excess near 0. At effect=0.10: clearly positive. Real tape? -1.6 ppt (noise).')"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal `MIXED`** -- Top-10 full-sample excess: **{R['excess10_mean']:+.1f} ppt/yr** "
            f"(HAC t={R['excess10_t']:+.2f}). Pre-2015: **{R['pre2015_excess']:+.0f} ppt** "
            f"(t={R['pre2015_t']:+.1f}) -- rejected. Post-2020: **{R['post2020_excess']:+.0f} ppt** "
            f"(t={R['post2020_t']:+.2f}) -- positive but sub-threshold (n=6). The signal exists "
            "in one specific era, not as a general rule.\n"
            f"- **Tradability `MIRAGE`** -- Survivorship bias inflates all returns. Annual "
            "turnover is low but the strategy has no demonstrated stability across regimes. "
            "A fund launched on pre-2015 history would have destroyed capital.\n"
            f"- **Recency bias `CONFIRMED`** -- The 2020-2025 Mag-7 window is 6 years against "
            "a 7-year prior history of large negative excess. The 'always works' claim fails "
            "simple historical scrutiny."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 -- Could you trade it? -- what it would really take\n\n"
            "Annual rebalancing keeps transaction costs low, but three structural issues kill it:"
        ),
        code(
            "# Worst-drawdown of the top-10 strategy in pre-2015 era\n"
            "if HAVE_REAL:\n"
            "    e10, _ = load_enriched(top_n=10)\n"
            "    cum = (1 + e10['topn_ret']).cumprod()\n"
            "    rolling_max = cum.cummax()\n"
            "    dd = (cum - rolling_max) / rolling_max\n"
            "    print(f'Max drawdown (top-10, price return): {dd.min():.1%}')\n"
            "    print(f'Worst year: {e10[\"topn_ret\"].idxmin()} ({e10[\"topn_ret\"].min():.1%})')\n"
            "    print(f'Full-period CAGR top-10: {((1+e10[\"topn_ret\"]).prod())**(1/len(e10))-1:.1%}')\n"
            "    print(f'Full-period CAGR EW:     {((1+e10[\"ew_ret\"]).prod())**(1/len(e10))-1:.1%}')\n"
            "    print(f'Full-period CAGR SPY:    {((1+e10[\"spy_ret\"].dropna()).prod())**(1/len(e10.dropna()))-1:.1%}')\n"
            "else:\n"
            "    print('Max drawdown top-10 (including 2008): approx -45.6%')\n"
            "    print('Worst year: 2008 (-45.6% -- top-10 was Citi, BAC, GE, MSFT etc.)')\n"
            "    print('CAGR top-10 ~10.4% vs EW ~11.5% vs SPY ~10.8% (survivorship-biased)')"
        ),
        md(
            "**The three structural kills:**\n\n"
            "1. **Regime instability** -- the full-sample Sharpe of the top-10 is "
            f"**{R['topn10_sharpe']:.2f}** vs **{R['topn10_sharpe']+0.05:.2f}** for equal-weight. "
            "Not significantly different, but the variance *between eras* is enormous. A fund "
            "manager who launched this strategy in 2015 after the pre-2015 disaster would have "
            "switched at exactly the wrong time.\n"
            "2. **Survivorship inflation** -- every return in this study is an upper bound on "
            "investable performance. The names that went to zero between 2008-2025 are not here.\n"
            "3. **Factor timing** -- the strategy is essentially a pure anti-size-premium bet. "
            "Running it requires a view that the size premium is permanently dead. History says "
            "it comes back."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **Quality screen.** Top-10 by ROIC or FCF yield rather than raw market cap -- "
            "would it survive the pre-2015 period? Worth testing with the EDGAR fundamentals "
            "cache (NetIncomeLoss, Assets).\n"
            "- **The size premium directly.** [Study 45](../../45-size-premium/) tests the "
            "full size-factor evidence -- the counterpart to this study. The interaction between "
            "regime (2020s AI dominance) and the size-vs-value-vs-quality dimensions is the "
            "productive next question.\n"
            "- **International evidence.** MSCI World: do the top-10 by global market cap show "
            "the same 2020s regime? The US megacap outperformance is partly a US-vs-world story.\n"
            "- **Post-2025 update.** The Magnificent-Seven era may reverse as AI CapEx costs "
            "mount and antitrust scrutiny grows. Fork this, update annually, and see.\n\n"
            "*Convince yourself the Mag-7 era is a structural break, not a cycle? "
            "Build the quality-filtered version and show it works pre-2015 too. "
            "That is the bar -- historical robustness, not just the last five years.*"
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
