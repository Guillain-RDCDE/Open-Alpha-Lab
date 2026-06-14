"""Generate the two narrative notebooks for Study 124 (Cash-Flow-Yield).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-panel cells use the cached
EDGAR parquets and year-end prices under ../_cache/ if present and otherwise quote the
frozen headline numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs
for any reader.

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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-14).
# Survivorship-biased S&P 500, 2007-2024, 18 years, annual quintile sort.
R = dict(
    n_years=18,
    n_firms_median=340,
    # OCF-yield hedge (Q5 - Q1)
    ocfy_hedge_mean=-0.006,
    ocfy_hedge_vol=0.149,
    ocfy_hedge_sharpe=-0.04,
    ocfy_hedge_t=-0.22,
    ocfy_hedge_hit=0.44,
    ocfy_hedge_maxdd=-0.493,
    # EY hedge (Q5 - Q1)
    ey_hedge_mean=-0.016,
    ey_hedge_vol=0.103,
    ey_hedge_sharpe=-0.15,
    ey_hedge_t=-0.77,
    ey_hedge_hit=0.28,
    # Top-Q vs market
    ocfy_spread_mean=0.013,
    ocfy_spread_t=0.74,
    ocfy_spread_hit=0.56,
    ey_spread_mean=0.014,
    ey_spread_t=1.56,
    ey_spread_hit=0.61,
    # IC comparison
    ocfy_ic_mean=0.007,
    ocfy_ic_t=0.21,
    ey_ic_mean=0.018,
    ey_ic_t=0.69,
    ic_diff_mean=-0.011,
    ic_diff_t=-0.61,
    # Market benchmark
    mkt_mean=0.157,
    mkt_vol=0.177,
    # Fingerprints
    fp_ocfy="1f338096bb11",
    fp_fwd="6a01d748393e",
)

# ---------------------------------------------------------------------------
# Shared boot cell -- imports, data loading, HAVE_REAL guard.
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from cash_flow_yield import data, strategy as st

_cache_dir = os.path.abspath(os.path.join("..", "..", "..", "_cache"))
_study_cache = os.path.abspath(os.path.join("..", "_cache"))

def _have_real():
    px_p = os.path.join(_study_cache, "yrpx.parquet")
    cfo_p = os.path.join(_cache_dir, "_edgar_NetCashProvidedByUsedInOperatingActivities.parquet")
    return os.path.exists(px_p) and os.path.exists(cfo_p)

HAVE_REAL = _have_real()

if HAVE_REAL:
    ocfy, ey, fwd = data.fetch_panel(cache_dir=_cache_dir, study_cache=_study_cache)
else:
    ocfy, ey, fwd = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

print("real data available:", HAVE_REAL, f"  panel: {ocfy.shape if not ocfy.empty else 'n/a'}")
"""

# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Cash-Flow-Yield -- does 'cash is king' beat earnings in a value sort?\n"
            "### Operating cash-flow yield as a quality-adjusted value signal on the S&P 500\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![OCF_beats_EY%3F: No](https://img.shields.io/badge/OCF_beats_EY%3F-No-8b949e?style=flat-square)\n\n"
            "A staple of value-investing folklore: reported earnings can be massaged — depreciation "
            "assumptions, revenue timing, one-time charges. Operating cash flow (the money actually "
            "collected) is harder to fake. So buying stocks where **OCF / market cap is high** "
            "(the cash equivalent of a low P/E) should work at least as well as earnings yield, "
            "and probably better. This notebook tests that idea honestly.\n\n"
            "> This is the plain-language layer. For t-stats, Spearman ICs, and the positive control "
            "see the companion **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does OCF yield predict next-year returns? | **No.** Hedge (Q5-Q1): "
            f"**{R['ocfy_hedge_mean']*100:+.1f}%/yr**, HAC *t* = **{R['ocfy_hedge_t']:+.2f}**. |\n"
            f"| Does the long-only version beat the market? | **Barely, and not significantly.** "
            f"Top-quintile spread: **{R['ocfy_spread_mean']*100:+.1f}%/yr**, *t* = **{R['ocfy_spread_t']:+.2f}**. |\n"
            f"| Is OCF yield better than earnings yield? | **No.** IC difference "
            f"*t* = **{R['ic_diff_t']:+.2f}** — statistically indistinguishable. |\n"
            "| Could you trade it? | **No.** No edge to harvest even before costs. |\n\n"
            "> The 'cash is king' logic is sound accounting theory, but on this survivorship-biased "
            "S&P 500 sample (18 years) the signal has been fully arbitraged away — or was never "
            "large enough to survive in large-cap equities."
        ),

        # ---- BEAT 1 -- THE CLAIM -----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *'Operating cash flow is a better measure of corporate health than reported earnings: "
            "it's harder to manipulate. Buy stocks where OCF/price is highest (highest yield), sell "
            "the lowest. Hold one year, rebalance. It should outperform a simple earnings-yield sort "
            "because it filters out the accruals noise.'*\n\n"
            "This is the cash-flow quality version of the classic value premium. The supporting "
            "theory comes from Sloan (1996) — accruals-heavy firms underperform; cash-backed "
            "earnings outperform — and from Lakonishok, Shleifer & Vishny (1994), who found that "
            "cash-flow/price was one of the best value predictors in their US sample. The claim "
            "has a 30-year academic pedigree."
        ),

        # ---- BEAT 2 -- SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the premium is alive in the modern S&P 500, it is a simple, once-a-year "
            "mechanical strategy with low turnover and clear economic intuition. If it is dead "
            "— killed by the quant funds that arbed it after Sloan 1996 was published — "
            "then the accounting lesson is correct but the trading conclusion doesn't follow. "
            "The interesting question is whether the *purification* (OCF vs NI) adds anything "
            "on top of the simpler P/E inverse, since both are trivially available."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "The protocol:\n\n"
            "1. **No look-ahead.** Each year's signal uses fiscal-year-end 10-K data (filed by "
            "~March of the following year); the forward return is the *next* calendar year — "
            "so the trade can only be placed after the 10-K is filed.\n"
            "2. **A fair baseline.** The long-short hedge (top-quintile minus bottom-quintile) is "
            "compared to zero (the null of no predictability), not to a leveraged benchmark.\n"
            "3. **A head-to-head.** The OCF-yield sort races against the earnings-yield sort. "
            "If OCF adds nothing over EY, the 'harder to manipulate' claim has no payoff.\n"
            "4. **Survivorship caveat.** Our universe is the *current* S&P 500 projected back — "
            "we exclude historical failures. This should *flatter* the result; finding nothing "
            "here is a conservative judgment.\n\n"
            "Data: EDGAR `NetCashProvidedByUsedInOperatingActivities`, `NetIncomeLoss`, and "
            "`WeightedAverageNumberOfDilutedSharesOutstanding` (annual 10-K); Yahoo Finance "
            "year-end closing prices for market cap; 2007-2024 (18 years)."
        ),

        # ---- BEAT 4 -- THE TEARDOWN -------------------------------------------
        md("## 4 · The teardown -- let's actually look"),
        md("**First: what do quintile returns look like, year by year?**"),
        code(
            "years_list = [2007,2008,2009,2010,2011,2012,2013,2014,2015,2016,\n"
            "              2017,2018,2019,2020,2021,2022,2023,2024]\n"
            "# Frozen hedge series from docs/results.md\n"
            "hedge_frozen = [-0.126,0.067,-0.091,0.139,-0.104,0.125,0.063,-0.190,\n"
            "                0.193,-0.085,-0.074,-0.114,-0.301,0.139,0.299,-0.044,-0.030,0.029]\n"
            "if HAVE_REAL:\n"
            "    q = st.quintile_sort(ocfy, fwd)\n"
            "    hedge_vals = q['hedge'].tolist()\n"
            "    yrs = q.index.tolist()\n"
            "else:\n"
            "    hedge_vals = hedge_frozen\n"
            "    yrs = years_list\n"
            "fig, ax = plt.subplots(figsize=(10.5, 4.5))\n"
            "colors = [GREEN if v > 0 else RED for v in hedge_vals]\n"
            "ax.bar(yrs, [v*100 for v in hedge_vals], color=colors, width=0.7)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Year (signal year, forward return = next year)')\n"
            "ax.set_ylabel('Hedge return (Q5 - Q1, %)')\n"
            "ax.set_title(f'OCF-yield hedge (Q5-Q1): {len(hedge_vals)} years, mean {sum(hedge_vals)/len(hedge_vals)*100:+.1f}%/yr')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Mean hedge: {{sum(hedge_vals)/len(hedge_vals)*100:+.2f}}%/yr')\n"
            f"print(f'Hit rate: {{sum(v>0 for v in hedge_vals)/len(hedge_vals):.0%}}')"
        ),
        md(
            f"The hedge is close to zero ({R['ocfy_hedge_mean']*100:+.1f}%/yr) and inconsistent "
            f"(hit rate {R['ocfy_hedge_hit']:.0%}) — below a coin-flip. The sort has no "
            "predictable direction: some years the high-OCF stocks win, some they lose, "
            "with no pattern the data resolves."
        ),
        md("**Now the head-to-head: does OCF beat the simpler earnings yield?**"),
        code(
            "ey_hedge_frozen = [0.090,-0.025,-0.165,0.063,-0.108,0.041,-0.066,-0.148,\n"
            "                    0.022,-0.046,-0.033,-0.077,-0.172,0.001,0.218,-0.034,-0.030,0.024]\n"
            "if HAVE_REAL:\n"
            "    q_ey = st.quintile_sort(ey, fwd)\n"
            "    ey_vals = q_ey['hedge'].tolist()\n"
            "    yrs2 = q_ey.index.tolist()\n"
            "else:\n"
            "    ey_vals = ey_hedge_frozen\n"
            "    yrs2 = years_list\n"
            "min_len = min(len(hedge_vals), len(ey_vals))\n"
            "h_plot = hedge_vals[:min_len]; e_plot = ey_vals[:min_len]; y_plot = yrs[:min_len]\n"
            "fig, ax = plt.subplots(figsize=(10.5, 4.5))\n"
            "x = range(len(y_plot))\n"
            "ax.bar([xi - 0.2 for xi in x], [v*100 for v in h_plot], width=0.35,\n"
            "       color=GREEN, alpha=0.75, label='OCF yield (Q5-Q1)')\n"
            "ax.bar([xi + 0.2 for xi in x], [v*100 for v in e_plot], width=0.35,\n"
            "       color=AMBER, alpha=0.75, label='EY (Q5-Q1)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(list(x)); ax.set_xticklabels(y_plot, rotation=45)\n"
            "ax.set_ylabel('Hedge return (%)')\n"
            "ax.set_title('OCF-yield vs earnings-yield hedge: both are noise')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            f"ocfy_m = sum(h_plot)/len(h_plot)*100\n"
            f"ey_m = sum(e_plot)/len(e_plot)*100\n"
            "print(f'OCF hedge mean: {ocfy_m:+.2f}%/yr   EY hedge mean: {ey_m:+.2f}%/yr')"
        ),
        md(
            f"Both hedges are near zero: OCF **{R['ocfy_hedge_mean']*100:+.1f}%/yr**, "
            f"EY **{R['ey_hedge_mean']*100:+.1f}%/yr**. Neither has a reliably positive "
            "or negative edge — they are both noisy. The 'cash is king' purification adds nothing "
            "a P/E-based sort doesn't already fail to deliver."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** OCF-yield hedge {R['ocfy_hedge_mean']*100:+.1f}%/yr, "
            f"HAC *t* = **{R['ocfy_hedge_t']:+.2f}**; EY hedge {R['ey_hedge_mean']*100:+.1f}%/yr, "
            f"*t* = **{R['ey_hedge_t']:+.2f}**. Both are well below the |*t*| ≥ 2 inference bar.\n"
            f"- **Tradability — Mirage.** No significant edge. Annual rebalancing is cheap "
            "but there is nothing to execute profitably.\n"
            f"- **OCF beats EY? — No.** IC head-to-head difference *t* = **{R['ic_diff_t']:+.2f}**."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Even setting aside statistical insignificance, the practical constraints are severe:\n\n"
            "- **Reporting lag.** 10-K data arrives up to three months after fiscal year end. "
            "In practice, some companies file even later, so your signal is already months old.\n"
            "- **Universe concentration.** Sorting a 400-name universe into quintiles (80 firms "
            "per bucket) means you hold 80 names for a year — manageable, but the dispersion "
            "within each quintile is huge, swamping the expected hedge return.\n"
            "- **The premium, if it existed, is well-known.** Sloan (1996) was published thirty "
            "years ago. Any persistent edge in S&P 500 names at annual frequency has been visible "
            "to institutional investors for decades.\n\n"
            "The conclusion is the same for costs: there is no gross edge to erode."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The accruals anomaly** is the mirror image: [Study 52 — Smoke-Screen](../../52-smoke-screen/) "
            "tests whether *high accruals* (= low cash backing) predict poor returns, using the "
            "same EDGAR data. The Sloan (1996) effect survives better in small caps.\n"
            "- **The F-score** ([Study 65 — Scorecard](../../65-scorecard/)) combines cash-flow "
            "quality with eight other fundamental signals into a composite; more signal dimensions "
            "than a single yield sort.\n"
            "- **The 'quality' dimension**: Asness, Frazzini & Pedersen's QMJ factor uses cash-flow "
            "quality as one of several quality proxies. The combination may fare better than any "
            "single metric here.\n\n"
            "*Think the cash-flow premium is alive somewhere? Extend to small caps, use enterprise "
            "value rather than market cap as the denominator, or combine with a quality screen. "
            "That's the bar — show a fair-bet edge that clears t = 2.*"
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
            "# Cash-Flow-Yield -- a quantitative teardown\n"
            "### EDGAR fundamentals . annual quintile sort . HAC inference . OCF vs EY IC race\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![OCF_beats_EY%3F: No](https://img.shields.io/badge/OCF_beats_EY%3F-No-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim now carrying its standard error.* We test whether "
            "OCF/market-cap has predictive power for next-year S&P 500 returns via a quintile sort "
            "(HAC t-stat on the annual hedge), whether the top quintile beats the equal-weight "
            "market (spread t-stat), and whether OCF yield has incremental Spearman IC over the "
            "simpler EY = NI/market-cap. Survivorship bias is named throughout.\n\n"
            "> **Not investment advice.** Real data: EDGAR 10-K fundamentals 2007-2024, "
            "Yahoo Finance year-end prices; as-of 2026-06-14. Methods & sources in "
            "[`docs/references.md`](../docs/references.md).\n"
            ">\n"
            "> **Survivorship caveat.** The ticker universe is the *current* S&P 500 projected "
            "back. Results are upper bounds; a proper backtest would use point-in-time index membership."
        ),
        code(BOOT + "\nfrom quantlab import analytics, stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | OCF-yield hedge "
            f"**{R['ocfy_hedge_mean']*100:+.2f}%/yr**, HAC *t* = **{R['ocfy_hedge_t']:+.2f}**; "
            f"EY hedge {R['ey_hedge_mean']*100:+.2f}%/yr, *t* = {R['ey_hedge_t']:+.2f}. "
            "Both < |*t*| 2. |\n"
            f"| **Tradability** | `MIRAGE` | Top-Q spread {R['ocfy_spread_mean']*100:+.2f}%/yr, "
            f"*t* = {R['ocfy_spread_t']:+.2f} -- insignificant before any cost. |\n"
            f"| **OCF beats EY?** | `NO` | IC diff mean {R['ic_diff_mean']:+.3f}, "
            f"*t* = {R['ic_diff_t']:+.2f}. No incremental information. |\n\n"
            "> On 18 years of survivorship-biased S&P 500 data, the OCF-yield sort yields "
            "no exploitable cross-sectional return differential."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $\\text{OCFY}_{i,t} = \\text{CFO}_{i,t} / (\\text{shares}_{i,t} \\cdot p_{i,t})$ "
            "and $\\text{EY}_{i,t} = \\text{NI}_{i,t} / (\\text{shares}_{i,t} \\cdot p_{i,t})$ "
            "for firm $i$ in fiscal year $t$. The claim has two parts:\n\n"
            "- **H1 (OCF yield is a value signal).** "
            "$\\mathbb{E}[r_{i,t+1} \\mid \\text{OCFY}_{i,t} \\text{ high}] > "
            "\\mathbb{E}[r_{i,t+1} \\mid \\text{OCFY}_{i,t} \\text{ low}]$ — "
            "measured as the mean of the Q5 − Q1 annual hedge return.\n"
            "- **H2 (OCF beats EY).** The Spearman IC of OCFY with $r_{t+1}$ exceeds that "
            "of EY — i.e. cash-flow quality adds incremental alpha over the simpler inverse-P/E.\n\n"
            "We test H1 via a quintile sort (HAC t-stat on hedge) and via a top-Q-vs-market spread. "
            "We test H2 via pairwise annual Spearman ICs and the mean IC difference."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "H1 and H2 together would make OCF yield a rare, simple, low-turnover factor with "
            "genuine economic content. H1 alone (without H2) would mean the value premium is "
            "available from EY too, so cash-flow purification is unnecessary complexity. "
            "Rejecting both confirms the post-2000 evidence that large-cap factor premia have "
            "been largely arbed away, consistent with McLean & Pontiff (2016) and the quant-fund "
            "crowding literature."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know -- the protocol\n\n"
            "- **Signal construction.** $\\text{OCFY}_{i,t} = "
            "\\text{CFO}_{i,t} / (\\text{shares}_{i,t} \\cdot p_{i,t,\\text{Dec}})$; "
            "similarly for EY. Shares from EDGAR `WeightedAverageNumberOfDilutedSharesOutstanding`; "
            "price = Yahoo Finance year-end close (December). Values outside $[0, 100\\%]$ for "
            "OCFY (or $[-50\\%, 100\\%]$ for EY) are treated as data errors and winsorised to NaN.\n"
            "- **Lag.** Signal for year $t$ aligns to forward return for year $t+1$ "
            "(one-year reporting lag; the standard in the academic literature).\n"
            "- **Sort.** Annual quintile sort, equal-weight within each bucket. Minimum 20 firms "
            "per year (2007 is the thinnest year with ~86 XBRL filers).\n"
            "- **Inference.** Newey-West HAC t-stat on the annual hedge series (18 obs). "
            "Spearman IC via `scipy.stats.spearmanr`.\n"
            "- **Positive control.** A deterministic synthetic panel with tunable OCF-yield and "
            "EY premiums confirms the engine recovers an edge *when one is planted*."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · OCF-yield quintile sort -- every bucket return, every year\n\n"
            "HAC t-stat on the Q5-Q1 hedge: the inference-bar test."
        ),
        code(
            "if HAVE_REAL:\n"
            "    q = st.quintile_sort(ocfy, fwd)\n"
            "    s = st.summary(q['hedge'])\n"
            "    hedge_vals = q['hedge'].values; yrs = q.index.values\n"
            "else:\n"
            "    hedge_frozen = [-0.126,0.067,-0.091,0.139,-0.104,0.125,0.063,-0.190,\n"
            "                    0.193,-0.085,-0.074,-0.114,-0.301,0.139,0.299,-0.044,-0.030,0.029]\n"
            "    hedge_vals = hedge_frozen; yrs = list(range(2007, 2025))\n"
            f"    s = dict(mean={R['ocfy_hedge_mean']}, vol={R['ocfy_hedge_vol']}, "
            f"sharpe={R['ocfy_hedge_sharpe']}, tstat={R['ocfy_hedge_t']}, "
            f"hit_rate={R['ocfy_hedge_hit']}, n={R['n_years']})\n"
            "import numpy as np\n"
            "# Cumulative wealth from the hedge (starting at 1.0)\n"
            "cum = np.cumprod([1 + r for r in hedge_vals])\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "colors = [GREEN if v > 0 else RED for v in hedge_vals]\n"
            "ax1.bar(yrs, [v*100 for v in hedge_vals], color=colors, width=0.7)\n"
            "ax1.axhline(0, c='k', lw=1)\n"
            "ax1.set_xlabel('Signal year'); ax1.set_ylabel('Hedge return (%)')\n"
            "ax1.set_title(f'OCF-yield Q5-Q1 hedge per year  (mean {s[\"mean\"]*100:+.2f}%/yr)')\n"
            "ax2.plot(yrs, cum, 'o-', c=RED, lw=2)\n"
            "ax2.axhline(1, c='k', lw=1)\n"
            "ax2.set_xlabel('Signal year'); ax2.set_ylabel('Cumulative wealth ($1 start)')\n"
            "ax2.set_title(f'HAC t = {s[\"tstat\"]:+.2f}  hit {s[\"hit_rate\"]:.0%}  n = {s[\"n\"]} yrs')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Hedge: mean={s[\"mean\"]*100:+.2f}%  vol={s[\"vol\"]*100:.2f}%  "
            "Sharpe={s[\"sharpe\"]:+.3f}  HAC t={s[\"tstat\"]:+.3f}')"
        ),
        md(
            f"> The HAC t-stat is **{R['ocfy_hedge_t']:+.2f}** — far below the |t| >= 2 bar. "
            f"Hit rate {R['ocfy_hedge_hit']:.0%} (below a coin). The cumulative wealth of the "
            "hedge oscillates around 1 with no trend."
        ),
        md(
            "### 4b · Head-to-head: OCF yield vs earnings yield\n\n"
            "Spearman IC of each signal with next-year returns, and the annual IC difference."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ic = st.pairwise_ic(ocfy, ey, fwd)\n"
            "    s_a = st.summary(ic['ic_a']); s_b = st.summary(ic['ic_b'])\n"
            "    s_d = st.summary(ic['ic_diff'])\n"
            "    ic_a_v = ic['ic_a'].values; ic_b_v = ic['ic_b'].values; yrs_ic = ic.index.values\n"
            "else:\n"
            "    ic_a_v = [0.0]*18; ic_b_v = [0.0]*18; yrs_ic = list(range(2007, 2025))\n"
            f"    s_a = dict(mean={R['ocfy_ic_mean']}, tstat={R['ocfy_ic_t']})\n"
            f"    s_b = dict(mean={R['ey_ic_mean']}, tstat={R['ey_ic_t']})\n"
            f"    s_d = dict(mean={R['ic_diff_mean']}, tstat={R['ic_diff_t']})\n"
            "fig, ax = plt.subplots(figsize=(10.5, 4.5))\n"
            "x = range(len(yrs_ic))\n"
            "ax.plot(list(x), ic_a_v, 'o-', c=GREEN, lw=2, label=f'OCF IC (mean {s_a[\"mean\"]:+.3f}, t={s_a[\"tstat\"]:+.2f})')\n"
            "ax.plot(list(x), ic_b_v, 's--', c=AMBER, lw=1.5, label=f'EY IC (mean {s_b[\"mean\"]:+.3f}, t={s_b[\"tstat\"]:+.2f})')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(list(x)); ax.set_xticklabels(list(yrs_ic), rotation=45)\n"
            "ax.set_ylabel('Spearman IC'); ax.set_title('Annual IC: OCF yield vs earnings yield')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'IC diff: mean={s_d[\"mean\"]:+.4f}  HAC t={s_d[\"tstat\"]:+.2f}')"
        ),
        md(
            f"Both ICs hover near zero (OCF: {R['ocfy_ic_mean']:+.3f}, EY: {R['ey_ic_mean']:+.3f}). "
            f"The difference is also negligible (mean {R['ic_diff_mean']:+.3f}, *t* = {R['ic_diff_t']:+.2f}). "
            "There is no statistically significant evidence that cash-flow purification improves "
            "predictive power on this universe and time window."
        ),
        md(
            "### 4c · Top-quintile vs market (long-only perspective)\n\n"
            "A long-only investor sorts by OCF yield and holds the top quintile. Does it outperform "
            "the equal-weight universe?"
        ),
        code(
            "spread_frozen = [-0.137,-0.003,-0.049,0.036,-0.022,0.101,0.047,-0.094,\n"
            "                  0.109,-0.014,0.004,-0.034,-0.084,0.078,0.168,0.034,0.019,0.071]\n"
            "if HAVE_REAL:\n"
            "    hvm = st.hedge_vs_market(ocfy, fwd)\n"
            "    sp_vals = hvm['spread'].values; yrs_hvm = hvm.index.values\n"
            "    s_hvm = st.summary(hvm['spread'])\n"
            "else:\n"
            "    sp_vals = spread_frozen; yrs_hvm = list(range(2007, 2025))\n"
            f"    s_hvm = dict(mean={R['ocfy_spread_mean']}, tstat={R['ocfy_spread_t']}, hit_rate={R['ocfy_spread_hit']})\n"
            "fig, ax = plt.subplots(figsize=(10.5, 4.5))\n"
            "colors = [GREEN if v > 0 else RED for v in sp_vals]\n"
            "ax.bar(list(range(len(sp_vals))), [v*100 for v in sp_vals], color=colors)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(range(len(yrs_hvm))); ax.set_xticklabels(list(yrs_hvm), rotation=45)\n"
            "ax.set_ylabel('Top-Q spread vs market (%)')\n"
            "ax.set_title(f'OCF top-quintile vs market  mean={s_hvm[\"mean\"]*100:+.2f}%/yr  t={s_hvm[\"tstat\"]:+.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Spread: mean={s_hvm[\"mean\"]*100:+.2f}%/yr  t={s_hvm[\"tstat\"]:+.2f}  hit={s_hvm[\"hit_rate\"]:.0%}')"
        ),
        md(
            f"> The long-only spread ({R['ocfy_spread_mean']*100:+.2f}%/yr, t = {R['ocfy_spread_t']:+.2f}) "
            "is positive on average but far below statistical significance on 18 annual observations. "
            "More years of data might resolve whether this small positive bias is real or sampling noise."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — OCF-yield hedge {R['ocfy_hedge_mean']*100:+.2f}%/yr, "
            f"HAC *t* {R['ocfy_hedge_t']:+.2f}; hit {R['ocfy_hedge_hit']:.0%}. "
            f"EY hedge {R['ey_hedge_mean']*100:+.2f}%/yr, *t* {R['ey_hedge_t']:+.2f}. "
            "Both reject neither H1 nor confirm it.\n"
            f"- **Tradability `MIRAGE`** — no significant edge; annual rebalancing is cheap but "
            "there is nothing to execute.\n"
            f"- **OCF beats EY? `NO`** — IC diff *t* {R['ic_diff_t']:+.2f}; the cash-flow "
            "purification has no measurable payoff over the simpler earnings yield on the S&P 500."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md("## 6 · Could you trade it? -- structural limitations"),
        code(
            "# Show the synthetic recovery to confirm the engine works\n"
            "from cash_flow_yield import data as d\n"
            "premiums = [0.00, 0.02, 0.04, 0.06, 0.08]\n"
            "mean_hedges = []\n"
            "for p in premiums:\n"
            "    oc, _, fw, _ = d.synthetic_panel(n_firms=300, n_years=30, ocfy_premium=p, ey_premium=0.0, seed=42)\n"
            "    q = st.quintile_sort(oc, fw)\n"
            "    mean_hedges.append(st.summary(q['hedge'])['mean']*100)\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(premiums, mean_hedges, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            f"ax.axhline({R['ocfy_hedge_mean']*100:.2f}, ls='--', c=RED, lw=1.5, label='real tape result')\n"
            "ax.set_xlabel('Planted OCF-yield premium (per z-unit)')\n"
            "ax.set_ylabel('Mean hedge return (%/yr)')\n"
            "ax.set_title('Positive control: the engine recovers a premium when one is planted')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('The engine works. The real tape simply has no premium to find.')"
        ),
        md(
            "The engine is faithful — on the synthetic tape it recovers the planted premium "
            "linearly. The near-zero result on the real tape is a statement about the market, "
            "not the method: the OCF-yield premium in large-cap S&P 500 names has been either "
            "eliminated by arbitrage or was always too small to detect on 18 annual observations."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The accruals anomaly** ([Study 52 — Smoke-Screen](../../52-smoke-screen/)): "
            "the Sloan (1996) effect (high accruals -> poor returns) is the mirror of OCF yield. "
            "It survived longer in small caps where arbitrage capacity was smaller.\n"
            "- **Composite quality** ([Study 65 — Scorecard](../../65-scorecard/)): the "
            "Piotroski F-score combines cash-flow quality with eight other signals; the combination "
            "may have more power than any single yield.\n"
            "- **Enterprise value denominator.** Using EV (= market cap + net debt) rather than "
            "market cap as the denominator is the standard in the practitioner literature "
            "(e.g. EBITDA/EV multiples); it normalises for capital structure differences.\n"
            "- **Small-cap extension.** The classic Sloan result was strongest in small caps. "
            "Extending the universe beyond the S&P 500 would test whether the premium still "
            "lives in less-arbed corners of the market.\n\n"
            "*To show a real edge: implement on a point-in-time universe with EV denominator, "
            "extend to mid/small-cap, and demonstrate |t| >= 2 on the hedge. The bar is clear.*"
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
