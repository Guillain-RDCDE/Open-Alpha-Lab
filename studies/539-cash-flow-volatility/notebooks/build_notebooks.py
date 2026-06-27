"""Generate the two narrative notebooks for Study 539 (Cash-Flow-Volatility).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
positive control runs anywhere, offline and deterministic; the real-panel cells use the
cached yfinance parquets under ../_cache/ if present and otherwise quote the frozen headline
numbers in ``R`` (a mirror of docs/results.md, as-of 2026-06-26).
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


# Frozen real-panel headline numbers -- mirror of docs/results.md (as-of 2026-06-26).
R = dict(
    n_months=95, year_start=2018, year_end=2025, n_tickers=40, n_per_leg=13,
    ls_mean=-3.43, ls_vol=9.81, ls_sharpe=-0.350, ls_t=-1.027,
    ls_hit=43.2, ls_dd=-34.7,
    low_leg=15.24, high_leg=18.67, spy_mean=14.25,
    placebo_p=0.357, real_mean_mo=-0.286,
    net_mean=-4.61, net_t=-1.381, cost_drag=1.18,
    fp_prices="6385498e2186", fp_cfvol="9f5b76cfb791",
    ctrl={0.00: (0.20, 0.11), 0.04: (8.14, 4.51), 0.08: (16.08, 8.90)},
    splits={"quintile": (-4.56, -0.85), "tercile": (-3.43, -1.03), "half": (-0.87, -0.26)},
    annual={2018: 1.2, 2019: -7.2, 2020: -16.5, 2021: 16.6,
            2022: -8.8, 2023: 1.2, 2024: -11.7, 2025: -1.6},
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Huang--direction%3F: Busted](https://img.shields.io/badge/Huang--direction%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from cash_flow_volatility import data, strategy as st

# Frozen headline numbers (mirror docs/results.md, as-of 2026-06-26) -- used when no cache.
R = dict(
    n_months=95, year_start=2018, year_end=2025, n_tickers=40, n_per_leg=13,
    ls_mean=-3.43, ls_vol=9.81, ls_sharpe=-0.350, ls_t=-1.027,
    ls_hit=43.2, ls_dd=-34.7,
    low_leg=15.24, high_leg=18.67, spy_mean=14.25,
    placebo_p=0.357, real_mean_mo=-0.286,
    net_mean=-4.61, net_t=-1.381, cost_drag=1.18,
    fp_prices="6385498e2186", fp_cfvol="9f5b76cfb791",
    splits={"quintile": (-4.56, -0.85), "tercile": (-3.43, -1.03), "half": (-0.87, -0.26)},
    annual={2018: 1.2, 2019: -7.2, 2020: -16.5, 2021: 16.6,
            2022: -8.8, 2023: 1.2, 2024: -11.7, 2025: -1.6},
)

def _have_cache():
    import os
    study = os.path.abspath("..")
    c = os.path.join(study, "_cache")
    return all(os.path.exists(os.path.join(c, f)) for f in
               ("cfvol_prices.parquet", "cfvol_spy.parquet", "cfvol_signal.parquet"))

HAVE_REAL = _have_cache()
print("yfinance cache present:", HAVE_REAL)

if HAVE_REAL:
    prices, spy, cfvol = data.fetch_panel()
    ls = st.build_real(prices, cfvol, exec_lag=1)
    s_ls  = st.summary(ls["ls_ret"])
    s_low = st.summary(ls["low_ret"])
    s_high= st.summary(ls["high_ret"])
    print(f"N months: {s_ls['n']} | LS mean: {s_ls['mean']*100:+.2f}%/yr"
          f" | HAC t: {s_ls['tstat']:+.3f}")
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Cash-Flow-Volatility -- do steady cash generators quietly win?\n"
            "### Huang (2009): the cash-flow-uncertainty anomaly, tested honestly\n\n"
            + BADGES +
            "In 2009, Alan Huang published a quiet but pointed finding: firms with more "
            "**volatile operating cash flows** earn **lower** future returns. Stable, "
            "predictable cash generators are rewarded; cash-flow lottery tickets are not. "
            "It is a quality-like anomaly -- a cousin of low-volatility and quality-minus-junk.\n\n"
            "We test that claim on a 40-name large-cap survivor basket using yfinance daily "
            "prices and quarterly cash-flow statements -- naming the survivorship bias and the "
            "thin (~5-quarter) fundamental history up front.\n\n"
            "> **This is the plain-language layer.** Want the multi-seed control, the placebo "
            "null, and the split-fraction robustness? See the companion "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do low-CF-vol firms out-earn high-CF-vol firms? | **No -- the opposite, weakly.** "
            f"Long-low / short-high earns **{R['ls_mean']:+.2f}%/yr**, HAC *t* = "
            f"**{R['ls_t']:+.2f}** (wrong sign). |\n"
            "| Is it distinguishable from random labels? | **No.** Label-shuffle placebo "
            f"*p* = **{R['placebo_p']:.2f}**. |\n"
            "| Is it tradable? | **No.** Negative gross and net "
            f"(**{R['net_mean']:+.2f}%/yr** net), Sharpe **{R['ls_sharpe']:+.2f}**. |\n"
            "| Survivorship / data problem? | **Yes, and we name both.** Basket = survivors; "
            "CF-vol rests on only ~5 quarters of yfinance fundamentals. |\n\n"
            "> The Huang (2009) cash-flow-uncertainty premium is documented on long Compustat "
            "histories. On 40 large-cap survivors with a thin yfinance signal window, it is "
            "absent -- and the high-CF-vol leg actually led the tape."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 -- The claim\n\n"
            "> *\"Firms with more volatile operating cash flows earn lower average returns. "
            "Cash-flow uncertainty is a priced characteristic: stable cash generators are "
            "rewarded, volatile ones are penalised.\"*\n\n"
            "-- Huang (2009), *Journal of Empirical Finance*\n\n"
            "The intuition: volatile cash flows raise the cost of external finance, force firms "
            "to skip good investments, and make earnings hard to forecast. Uncertainty-averse "
            "investors discount such firms. So a portfolio **long low-CF-vol** and "
            "**short high-CF-vol** should earn a positive quality-like premium."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 -- So what?\n\n"
            "If true, cash-flow stability is a free lunch hiding inside the income statement: "
            "a signal you can compute from public filings, uncorrelated with size or value, "
            "that tilts you toward boring, dependable compounders. It is one of the 'safety' "
            "inputs in AQR's *Quality Minus Junk* (see "
            "[Study 242](../../242-quality-minus-junk/)).\n\n"
            "The questions for us:\n"
            "1. **Does it survive on a realistic large-cap basket?**\n"
            "2. **Can yfinance even serve enough cash-flow history to measure it?**\n"
            "3. **What is the survivorship bias?**"
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 -- How would we even know?\n\n"
            "Three disciplines keep us honest:\n\n"
            "1. **A synthetic positive control.** Before trusting the engine on real data, we "
            "plant a known CF-vol premium in a fake market and check the long-short recovers it "
            "(averaged over 20 seeds), and returns noise when the premium is zero.\n"
            "2. **A label-shuffle placebo.** Permute the CF-vol labels across names, re-sort, "
            "recompute. If the real long-short sits inside that null, there is no signal.\n"
            "3. **Name the biases.** The basket is *current* survivors -- failed high-CF-vol "
            "firms are gone. And yfinance serves only ~5 quarters, so the signal is thin. "
            "Both inflate the result upward, and it is *still* negative."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md("## 4 -- The teardown\n\n**First, does the engine work when the effect is planted?**"),
        code(
            "ctrl = st.synthetic_control(premiums=(0.0, 0.04, 0.08), n_seeds=20)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREEN if m > 0.5 else GREY for m in ctrl['mean_ls_%']]\n"
            "ax.bar(ctrl['premium'].astype(str), ctrl['mean_ls_%'], color=col)\n"
            "for i, (m, t) in enumerate(zip(ctrl['mean_ls_%'], ctrl['avg_t'])):\n"
            "    ax.text(i, m + 0.3, f't={t:+.1f}', ha='center', fontsize=9)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted CF-vol premium'); ax.set_ylabel('mean long-short (%/yr)')\n"
            "ax.set_title('Synthetic control: engine recovers a planted premium (20-seed avg)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(3).to_string(index=False))"
        ),
        md(
            "The engine is faithful: ~zero at the null (avg *t* = +0.11), strongly positive and "
            "significant when a premium is planted. So the verdict on the real tape reflects "
            "**the market**, not a broken method. *(This control is a power check only -- never "
            "evidence for the real-tape stamp.)*"
        ),
        md("**Now the honest test on the real yfinance basket.**"),
        code(
            "if HAVE_REAL:\n"
            "    low_m, high_m = s_low['mean']*100, s_high['mean']*100\n"
            "    ls_m, ls_t = s_ls['mean']*100, s_ls['tstat']\n"
            "    yb = ls.copy(); yb['year'] = [p.year for p in yb.index]\n"
            "    annual = yb.groupby('year')['ls_ret'].apply(lambda x: float((1+x).prod()-1))*100\n"
            "else:\n"
            "    low_m, high_m = R['low_leg'], R['high_leg']\n"
            "    ls_m, ls_t = R['ls_mean'], R['ls_t']\n"
            "    annual = pd.Series(R['annual'])\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "axes[0].bar(['Low-CF-vol\\n(LONG)', 'SPY', 'High-CF-vol\\n(SHORT)'],\n"
            "            [low_m, R['spy_mean'], high_m], color=[AMBER, GREY, RED], width=0.55)\n"
            "axes[0].axhline(R['spy_mean'], ls='--', c=GREY, lw=1)\n"
            "axes[0].set_ylabel('mean annual return (%/yr)')\n"
            "axes[0].set_title('The SHORT leg WON: Huang sign is reversed here')\n"
            "col_yr = [GREEN if v > 0 else RED for v in annual.values]\n"
            "axes[1].bar(annual.index, annual.values, color=col_yr)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_xlabel('year'); axes[1].set_ylabel('long-short return (%/yr)')\n"
            "axes[1].set_title(f'Year-by-year LS | mean {ls_m:+.1f}% t={ls_t:+.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Long-short: {ls_m:+.2f}%/yr | HAC t = {ls_t:+.3f}')"
        ),
        md(
            f"The long-short earns **{R['ls_mean']:+.1f}%/yr** at HAC *t* = "
            f"**{R['ls_t']:+.2f}** -- below the bar *and* the wrong sign. The high-CF-vol "
            f"leg (**+{R['high_leg']:.1f}%/yr**) beat the low-CF-vol leg "
            f"(**+{R['low_leg']:.1f}%/yr**): on this 2018-2025 large-cap tape, the volatile "
            "cash generators led."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal -- NONE.** HAC *t* = {R['ls_t']:+.2f}, placebo *p* = "
            f"{R['placebo_p']:.2f}, sign reversed. Robust across quintile/tercile/half splits. "
            "Literature support alone never lifts the verdict on this desk -- the real tape "
            "is silent.\n"
            f"- **Tradability -- MIRAGE.** Negative gross ({R['ls_mean']:+.2f}%/yr) and net "
            f"({R['net_mean']:+.2f}%/yr), Sharpe {R['ls_sharpe']:+.2f}, max DD {R['ls_dd']:.1f}%. "
            "Nothing for costs to erode.\n"
            "- **Huang direction -- BUSTED (on this tape).** The high-CF-vol leg won. On "
            "survivors with a thin signal window, the anomaly is absent and the sign flips."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 -- Could you actually trade it?\n\n"
            "There is no gross edge to trade, so the costs question is moot -- but for the "
            "record:\n\n"
            "1. **Short borrow.** The high-CF-vol short leg (often growth/cyclicals) can be "
            "expensive to borrow.\n"
            "2. **Thin signal.** With only ~5 quarters of yfinance data, the CF-vol estimate "
            "is noisy and slow to update -- not a tradable, frequently-refreshed factor.\n"
            "3. **Survivorship.** The names that would have made the short leg pay -- firms "
            "whose cash flows blew up and that delisted -- are not in the basket.\n\n"
            "A Compustat-grade replication over a long history might recover Huang's result. "
            "On this evidence, the verdict is **Mirage**."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **Long history.** Huang (2009) used Compustat back to the 1980s. A real "
            "replication needs quarterly cash flows far beyond yfinance's ~5-quarter window.\n"
            "- **Broader universe.** The premium (like most anomalies) is strongest in "
            "small/microcaps, which are absent from this large-cap basket.\n"
            "- **Quality umbrella.** [Study 242 -- Quality-Minus-Junk](../../242-quality-minus-junk/) "
            "bundles CF-vol 'safety' with profitability and growth.\n"
            "- **Cousins.** [Study 330 -- Low-Volatility-Anomaly](../../330-low-volatility-anomaly/) "
            "(return vol) and [Study 231 -- Sloan-Accruals](../../231-sloan-accruals/) "
            "(cash-flow vs accrual quality).\n\n"
            "*Think CF-vol survives net of costs on a long, unbiased history? Fork this, wire "
            "Compustat-grade quarterly cash flows, expand to Russell 3000, and show t > 2 net. "
            "That is the bar.*"
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
            "# Cash-Flow-Volatility -- a quantitative teardown\n"
            "### yfinance basket * trailing CF-vol sort * long-short * HAC inference * placebo\n\n"
            + BADGES +
            "The quantitative companion to the "
            "[notebook for the curious](01_for_the_curious.ipynb) -- same seven beats, every "
            "claim carrying its standard error. We test Huang (2009): rank by trailing "
            "cash-flow volatility (std of quarterly OCF / total assets), long the low-CF-vol "
            "tercile, short the high-CF-vol tercile, one execution-lag day, monthly returns.\n\n"
            "> **Not investment advice.** Real data: yfinance daily adjusted-close + quarterly "
            "statements, 40 large-caps, 2018-2025, as-of 2026-06-26. Methods in "
            "[`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Survivorship bias is named** (current survivors) and the CF-vol signal rests "
            "on only **~5 quarters** of yfinance fundamentals -- both upper-bound caveats."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | LS mean **{R['ls_mean']:+.2f}%/yr**, HAC *t* = "
            f"**{R['ls_t']:+.3f}** (wrong sign), placebo *p* = **{R['placebo_p']:.2f}**. |\n"
            f"| **Tradability** | `MIRAGE` | Sharpe **{R['ls_sharpe']:+.3f}**, max DD "
            f"**{R['ls_dd']:.1f}%**, net **{R['net_mean']:+.2f}%/yr**. |\n"
            "| **Huang-direction?** | `BUSTED` | High-CF-vol leg led "
            f"(**+{R['high_leg']:.2f}%** vs **+{R['low_leg']:.2f}%/yr**). |\n\n"
            "> Huang (2009) is documented on long Compustat histories. On 40 large-cap "
            "survivors with ~5 quarters of yfinance fundamentals, the effect is absent and "
            "the sign reverses."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 -- The claim, steelmanned\n\n"
            "Let $\\sigma^{CF}_i$ = std of stock $i$'s quarterly operating cash flow scaled "
            "by total assets. Huang (2009) asserts:\n\n"
            "- **H1 (signal).** $E[r_i]$ decreases in $\\sigma^{CF}_i$. The long-short "
            "$r^{\\text{low}} - r^{\\text{high}} > 0$.\n"
            "- **H2 (robust).** Survives controls for size, value, momentum, return-vol, accruals.\n"
            "- **H3 (tradable).** The premium survives costs.\n\n"
            "We **reject H1** on this tape (LS mean negative, *t* = -1.03, sign reversed), "
            "cannot test H2 cleanly with thin data, and **reject H3** trivially (no gross edge)."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 -- So what? -- the economic stakes\n\n"
            "Cash-flow stability is a pillar of 'quality' investing -- embedded in QMJ funds, "
            "low-vol ETFs, and dividend-aristocrat screens. If the standalone CF-vol tilt does "
            "not pay on large-caps (and here it has the wrong sign), the 'safety earns a "
            "premium' story rests on small-caps, long histories, or the *other* quality inputs."
        ),

        # ---- BEAT 3 -- PROTOCOL ----------------------------------------------
        md(
            "## 3 -- The protocol\n\n"
            "- **Signal.** $\\sigma^{CF}_i$ = std of (quarterly OCF / total assets), formed "
            "from yfinance quarterly statements (a stale cross-sectional snapshot).\n"
            "- **Sort.** Cross-sectional terciles by $\\sigma^{CF}$.\n"
            "- **Long leg.** Equal-weight low-CF-vol tercile. **Short leg.** Equal-weight "
            "high-CF-vol tercile.\n"
            "- **LS return.** low_ret - high_ret, monthly.\n"
            "- **Execution lag.** One trading day after the price window opens (no same-bar fill).\n"
            "- **Inference.** Newey-West HAC *t* + a 1000-permutation label-shuffle placebo.\n"
            "- **Costs.** 10 bps one-way x turnover + 100 bps/yr borrow on the short leg.\n"
            "- **Universe caveat.** 40 large-cap survivors; ~5 quarters of fundamentals."
        ),

        # ---- BEAT 4 -- TEARDOWN ----------------------------------------------
        md("## 4 -- The teardown"),
        md(
            "### 4a -- Positive control: faithful detector (20-seed average)\n\n"
            "Sweep the planted CF-vol premium; the long-short mean and HAC *t* should rise "
            "monotonically, averaged over 20 independent synthetic panels."
        ),
        code(
            "ctrl = st.synthetic_control(premiums=(0.0, 0.02, 0.04, 0.06, 0.08), n_seeds=20)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(ctrl['premium'], ctrl['avg_t'], 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='|t|=2 bar')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted CF-vol premium'); ax.set_ylabel('avg HAC t (20 seeds)')\n"
            "ax.set_title('Positive control: avg t is monotone in planted premium')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(3).to_string(index=False))"
        ),
        md(
            "### 4b -- The CF-volatility cross-section (real basket)\n\n"
            "What does the trailing CF-vol distribution look like across the 40 names?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cv = cfvol['cfvol'].sort_values()\n"
            "    k = max(1, len(cv)//3)\n"
            "    colr = [AMBER]*k + [GREY]*(len(cv)-2*k) + [RED]*k\n"
            "    fig, ax = plt.subplots(figsize=(11, 4.3))\n"
            "    ax.bar(range(len(cv)), cv.values*100, color=colr)\n"
            "    ax.set_xticks(range(len(cv))); ax.set_xticklabels(cv.index, rotation=90, fontsize=7)\n"
            "    ax.set_ylabel('CF-vol = std(OCF/assets), %')\n"
            "    ax.set_title('Trailing cash-flow volatility by name (amber=LONG, red=SHORT)')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'Long (low-CF-vol): {list(cv.index[:k])}')\n"
            "    print(f'Short (high-CF-vol): {list(cv.index[-k:])}')\n"
            "else:\n"
            "    print('Cross-section needs the real cache. Frozen: low leg '\n"
            "          f'{R[\"low_leg\"]:+.1f}%/yr vs high leg {R[\"high_leg\"]:+.1f}%/yr')"
        ),
        md("### 4c -- Real long-short: legs and year-by-year"),
        code(
            "if HAVE_REAL:\n"
            "    low_m, high_m = s_low['mean']*100, s_high['mean']*100\n"
            "    ls_m, ls_t = s_ls['mean']*100, s_ls['tstat']\n"
            "    yb = ls.copy(); yb['year'] = [p.year for p in yb.index]\n"
            "    annual = yb.groupby('year')['ls_ret'].apply(lambda x: float((1+x).prod()-1))*100\n"
            "else:\n"
            "    low_m, high_m = R['low_leg'], R['high_leg']\n"
            "    ls_m, ls_t = R['ls_mean'], R['ls_t']\n"
            "    annual = pd.Series(R['annual'])\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "axes[0].bar(['Low-CF-vol\\n(LONG)', 'SPY', 'High-CF-vol\\n(SHORT)'],\n"
            "            [low_m, R['spy_mean'], high_m], color=[AMBER, GREY, RED], width=0.55)\n"
            "axes[0].axhline(R['spy_mean'], ls='--', c=GREY, lw=1)\n"
            "axes[0].set_ylabel('mean annual return (%/yr)')\n"
            "axes[0].set_title('Raw legs: the SHORT leg out-earned the LONG')\n"
            "col_yr = [GREEN if v > 0 else RED for v in annual.values]\n"
            "axes[1].bar(annual.index, annual.values, color=col_yr)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_xlabel('year'); axes[1].set_ylabel('long-short (%/yr)')\n"
            "axes[1].set_title(f'LS year-by-year | mean {ls_m:+.1f}% t={ls_t:+.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'LS: {ls_m:+.2f}%/yr | HAC t = {ls_t:+.3f}')"
        ),
        md(
            f"> The long-short earns **{R['ls_mean']:+.2f}%/yr** at HAC *t* = "
            f"**{R['ls_t']:+.3f}** -- below the |t|>=2 bar and the wrong sign. Statistically "
            "indistinguishable from zero on 95 months."
        ),
        md("### 4d -- Label-shuffle placebo null (1000 permutations)"),
        code(
            "if HAVE_REAL:\n"
            "    monthly = st.monthly_returns_from_prices(prices, exec_lag=1)\n"
            "    pl = st.placebo_pvalue(monthly, cfvol['cfvol'], n_perm=1000)\n"
            "    real_mo, p_val = pl['real_mean']*100, pl['p_value']\n"
            "    null_mean, null_std = pl['null_mean']*100, pl['null_std']*100\n"
            "    import numpy as _np\n"
            "    draws = _np.random.default_rng(0).normal(null_mean, null_std, 5000)\n"
            "    fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "    ax.hist(draws, bins=50, color=GREY, alpha=0.6, label='shuffled-label null')\n"
            "    ax.axvline(real_mo, c=RED, lw=2, label=f'real LS = {real_mo:+.3f}%/mo')\n"
            "    ax.set_xlabel('mean long-short (%/mo)'); ax.set_ylabel('count')\n"
            "    ax.set_title(f'Placebo: real LS sits inside the null (p={p_val:.2f})')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f'real {real_mo:+.3f}%/mo | null {null_mean:+.3f}+/-{null_std:.3f} | p={p_val:.3f}')\n"
            "else:\n"
            "    print(f'Frozen placebo: real {R[\"real_mean_mo\"]:+.3f}%/mo, p={R[\"placebo_p\"]:.3f}')"
        ),
        md("### 4e -- Equity curve, drawdown, and split-fraction robustness"),
        code(
            "if HAVE_REAL:\n"
            "    eq = (1 + ls['ls_ret']).cumprod()\n"
            "    dd = (eq / eq.cummax() - 1) * 100\n"
            "    x = [p.to_timestamp() for p in ls.index]\n"
            "    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)\n"
            "    a1.plot(x, eq.values, c=RED, lw=2); a1.axhline(1, c='k', ls='--', lw=1)\n"
            "    a1.set_ylabel('cumulative wealth (LS)')\n"
            "    a1.set_title('Long-short equity curve -- a slow bleed')\n"
            "    a2.fill_between(x, dd.values, 0, color=RED, alpha=0.4)\n"
            "    a2.set_ylabel('drawdown (%)'); a2.set_xlabel('date')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    rows = []\n"
            "    for frac, nm in [(0.2,'quintile'),(1/3,'tercile'),(0.5,'half')]:\n"
            "        lr = st.build_real(prices, cfvol, tercile=frac)\n"
            "        sr = st.summary(lr['ls_ret'])\n"
            "        rows.append({'split': nm, 'k': int(lr['n_low'].iloc[0]),\n"
            "                     'mean_%': sr['mean']*100, 't': sr['tstat']})\n"
            "    print(pd.DataFrame(rows).round(3).to_string(index=False))\n"
            "else:\n"
            "    print('Frozen splits:', R['splits'])"
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal `NONE`** -- LS {R['ls_mean']:+.2f}%/yr, HAC *t* = {R['ls_t']:+.3f}, "
            f"placebo *p* = {R['placebo_p']:.2f}, sign reversed, robust across splits.\n"
            f"- **Tradability `MIRAGE`** -- Sharpe {R['ls_sharpe']:+.3f}, net "
            f"{R['net_mean']:+.2f}%/yr, max DD {R['ls_dd']:.1f}%. No gross edge.\n"
            "- **Huang-direction `BUSTED`** -- the high-CF-vol leg led. Survivorship + thin "
            "fundamentals both named; the honest outcome for a pointed factor on a small "
            "survivor basket."
        ),

        # ---- BEAT 6 -- TRADABILITY -------------------------------------------
        md(
            "## 6 -- Could you trade it?\n\n"
            "No -- there is no gross premium. Even granting one, the practical frictions are "
            "fatal: a short leg of expensive-to-borrow growth names, a signal refreshed only "
            "quarterly from a ~5-quarter window, and a survivor basket that has already "
            "removed the firms whose cash flows actually imploded. This is a Mirage."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = st.apply_costs(ls)\n"
            "    print(f\"Gross {c['gross_mean']*100:+.2f}%/yr (t {c['gross_t']:+.2f}) -> \"\n"
            "          f\"Net {c['net_mean']*100:+.2f}%/yr (t {c['net_t']:+.2f}); \"\n"
            "          f\"cost drag {c['cost_drag']*100:.2f}%/yr\")\n"
            "else:\n"
            "    print(f'Frozen: net {R[\"net_mean\"]:+.2f}%/yr, t {R[\"net_t\"]:+.2f}, '\n"
            "          f'cost drag {R[\"cost_drag\"]:.2f}%/yr')"
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **Compustat history.** Huang (2009) used decades of quarterly data; yfinance's "
            "~5-quarter window cannot form a stable rolling CF-vol signal.\n"
            "- **Microcaps.** The premium concentrates where financing frictions bind hardest.\n"
            "- **Controls.** Orthogonalise CF-vol against size, value, momentum, return-vol, "
            "and accruals (H2) -- impossible cleanly with thin data here.\n"
            "- **[Study 242 -- Quality-Minus-Junk](../../242-quality-minus-junk/)**, "
            "**[Study 330 -- Low-Volatility-Anomaly](../../330-low-volatility-anomaly/)**, "
            "**[Study 231 -- Sloan-Accruals](../../231-sloan-accruals/)** -- the quality family.\n\n"
            "*Think CF-vol pays net of costs on a long, unbiased history? Fork this, wire "
            "Compustat-grade quarterly cash flows, expand to Russell 3000, and show t > 2 net. "
            "That is the bar.*"
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
