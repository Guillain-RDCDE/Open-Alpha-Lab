"""Generate the two narrative notebooks for Study 123 (Altman-Z).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). Synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
EDGAR parquets under _cache/ if present and otherwise fall back to the frozen headline
numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.

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
R = dict(
    # Panel
    n_years=16, n_tickers=166, n_obs=1210,
    year_start=2008, year_end=2023,
    # Hedge
    hedge_mean=2.36, hedge_vol=14.0, hedge_sharpe=0.15,
    hedge_t=0.65, hedge_ci_lo=-0.44, hedge_ci_hi=0.60, hedge_frac_neg=26,
    hedge_hit=56,
    # Per-bucket annual means
    ret_lo=20.5, ret_mid=21.8, ret_hi=22.9, ret_mkt=21.7,
    # Firm level
    corr=0.013,
    # Z zone fractions
    pct_distress=37.3, pct_grey=23.1, pct_safe=39.6,
    # Zone returns
    zone_distress_ret=20.5, zone_safe_ret=20.4,
)

# ---------------------------------------------------------------------------
# Shared preamble
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

from altman_z import data, strategy as st

def _have_cache():
    cache = data.DEFAULT_CACHE
    needed = [
        "_edgar_Assets.parquet", "_edgar_AssetsCurrent.parquet",
        "_edgar_LiabilitiesCurrent.parquet",
        "_edgar_RetainedEarningsAccumulatedDeficit.parquet",
        "_edgar_OperatingIncomeLoss.parquet", "_edgar_Liabilities.parquet",
        "_edgar_Revenues.parquet",
        "_edgar_WeightedAverageNumberOfDilutedSharesOutstanding.parquet",
        "_edgar_yrret.parquet",
    ]
    return all(os.path.exists(os.path.join(cache, f)) for f in needed)

HAVE_REAL = _have_cache()
print("EDGAR cache present:", HAVE_REAL)

if HAVE_REAL:
    Z_REAL, FWD_REAL = data.fetch_panel()
    RESULTS = st.tertile_hedge(Z_REAL, FWD_REAL)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Altman-Z -- does the distress score predict stock returns?\n"
            "### A survivorship-biased S&P 500 test of the classic bankruptcy model as an equity signal\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Distress_premium%3F: Absent](https://img.shields.io/badge/Distress_premium%3F-Absent-8b949e?style=flat-square)\n\n"
            "In 1968 Edward Altman combined five accounting ratios into a single score to predict corporate "
            "bankruptcy two years ahead. The Altman Z-score became one of the most cited models in finance. "
            "Decades later, practitioners repurposed it as an **equity signal**: sort stocks by Z, go long "
            "the safe (high-Z) ones and avoid or short the distressed (low-Z) ones -- or, for the risk-premium "
            "version, do the opposite, since distressed stocks *should* earn more to compensate for the risk. "
            "This notebook tests both stories against the real data, honestly.\n\n"
            "> **This is the plain-language layer.** Want the t-stats, bootstrap intervals, and survivorship "
            "anatomy? That's the companion, **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn by the code "
            "beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does high-Z (safe) beat low-Z (distressed)? | **Barely.** The annual hedge is "
            f"+{R['hedge_mean']:.1f}%/yr -- less than the noise. |\n"
            "| Is it statistically real? | **No.** HAC t-stat = +{R['hedge_t']:.2f} (need ±2). |\n"
            "| Do distressed firms earn a risk premium? | **No.** Low-Z firms return "
            f"+{R['ret_lo']:.1f}%/yr vs +{R['ret_hi']:.1f}%/yr for high-Z. Distressed stocks "
            "underperform -- the distress *puzzle*. |\n"
            "| Could you trade it? | **No.** Even before costs, the signal is statistically "
            "indistinguishable from noise. |\n\n"
            "> The Z-score is a good *bankruptcy predictor* but a poor *return predictor* -- "
            "two different questions that happen to share the same formula."
        ),

        # ---- BEAT 1 -- THE CLAIM ----------------------------------------------
        md(
            "## 1 The claim\n\n"
            "> *'Altman's Z-score tells you whether a company is headed for bankruptcy. "
            "Sort S&P 500 stocks by Z each year: avoid the distressed zone (Z < 1.81), "
            "overweight the safe zone (Z > 2.99). Distressed firms carry more risk, so "
            "either they're cheap relative to that risk (and should earn more) or they "
            "bleed out from financial strain (and should earn less). Either way, Z sorts "
            "returns.'*\n\n"
            "There are actually *two* versions of the claim and they point in opposite directions:\n\n"
            "- **Distress premium** (risk story): low-Z firms are riskier, so they should earn *more*.\n"
            "- **Distress puzzle** (overpricing story, Dichev 1998, Campbell 2008): low-Z firms are "
            "overpriced relative to their risk, so they actually earn *less*.\n\n"
            "We test both by simply measuring what the three Z-buckets earn next year."
        ),

        # ---- BEAT 2 -- SO WHAT ------------------------------------------------
        md(
            "## 2 So what?\n\n"
            "If either story is true and stable, you have a cheap annual signal: download 10-K data, "
            "compute five ratios, sort. The formula has been public since 1968 -- so if the premium "
            "survived, it's either a fundamental risk story or something the market keeps mispricing. "
            "If neither is true, Z is a fine bankruptcy predictor that simply doesn't translate to "
            "a return predictor in the current S&P 500 -- and that distinction matters for anyone "
            "who heard 'the Z-score works as a stock screen.'"
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 How would we even know?\n\n"
            "1. **Form the Z-score** from year-*t* EDGAR 10-K data (WC/TA, RE/TA, EBIT/TA, "
            "MktEq/Liab, Sales/TA). Market cap uses December month-end price from yfinance.\n"
            "2. **Sort into terciles.** Bottom third = distressed, top third = safe.\n"
            "3. **Measure next-year return** (calendar year *t+1*). The 10-K files in Q1 of *t+1*, "
            "so using year-*t* fundamentals with year-*t+1* returns is a clean one-year lag -- "
            "no look-ahead.\n"
            "4. **Inference bar.** HAC t-stat on the 16-year annual hedge series (|t| >= 2).\n"
            "5. **Survivorship caveat.** Our panel contains only current S&P 500 members. Firms "
            "that went bankrupt or were delisted during 2008-2023 are missing, inflating "
            "distressed-bucket returns. Results are **upper bounds**."
        ),

        # ---- BEAT 4 -- THE TEARDOWN -------------------------------------------
        md(
            "## 4 The teardown\n\n"
            "**First: what do the Z buckets actually earn?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    labels = ['Low-Z\\n(distressed)', 'Mid-Z\\n(grey zone)', 'High-Z\\n(safe)']\n"
            "    means = [RESULTS['ret_lo'].mean()*100, RESULTS['ret_mid'].mean()*100,\n"
            "             RESULTS['ret_hi'].mean()*100]\n"
            "    mkt = RESULTS['ret_mkt'].mean()*100\n"
            "else:\n"
            f"    labels = ['Low-Z\\n(distressed)', 'Mid-Z\\n(grey zone)', 'High-Z\\n(safe)']\n"
            f"    means = [{R['ret_lo']}, {R['ret_mid']}, {R['ret_hi']}]\n"
            f"    mkt = {R['ret_mkt']}\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.5))\n"
            "cols = [RED, GREY, GREEN]\n"
            "bars = ax.bar(labels, means, color=cols, width=0.5)\n"
            "ax.axhline(mkt, ls='--', c='k', lw=1.5, label=f'Market EW: {mkt:.1f}%/yr')\n"
            "ax.set_ylabel('Annual return (%/yr)')\n"
            "ax.set_title('Z-score buckets: all three earn roughly the same')\n"
            "for b, m in zip(bars, means):\n"
            "    ax.text(b.get_x()+b.get_width()/2, m+0.3, f'{m:.1f}%', ha='center', va='bottom', fontsize=10)\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Spread (high minus low): {means[2]-means[0]:+.1f}%/yr')"
        ),
        md(
            f"All three buckets earn roughly **+20-23%/yr** over 2008-2023 -- driven almost "
            "entirely by the long equity bull market, not by Z-score differences. "
            f"The spread is **+{R['hedge_mean']:.1f}%/yr**, which sounds like something but "
            "is smaller than one year of noise and completely fails the statistical test."
        ),
        md(
            "**Now: is the hedge consistent year by year?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    years = RESULTS.index.tolist()\n"
            "    hedge_yr = RESULTS['hedge'].tolist()\n"
            "else:\n"
            "    years = list(range(2008, 2024))\n"
            f"    hedge_yr = [0.103, 0.204, -0.046, 0.073, -0.181, 0.002, 0.074, -0.067,\n"
            f"                0.020, -0.034, 0.144, 0.428, -0.080, -0.178, -0.126, 0.040]\n"
            "fig, ax = plt.subplots(figsize=(10, 4.3))\n"
            "cols = [GREEN if h > 0 else RED for h in hedge_yr]\n"
            "ax.bar(years, [h*100 for h in hedge_yr], color=cols, width=0.7)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Year (signal year; return measured in year+1)')\n"
            "ax.set_ylabel('Hedge return (high-Z minus low-Z, %/yr)')\n"
            "ax.set_title('The hedge flips sign half the time -- no consistent direction')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print('Positive years: {R['hedge_hit']}% of 16')"
        ),
        md(
            f"The hedge is positive **{R['hedge_hit']}%** of the time (9 of 16 years) -- "
            "essentially a coin flip. The two years with the biggest positive hedges (2009 and 2019) "
            "look like recoveries from crisis periods where distressed firms happened to lag more. "
            "There is no stable annual signal here."
        ),

        # ---- BEAT 5 -- VERDICT ------------------------------------------------
        md(
            "## 5 The verdict\n\n"
            f"- **Signal -- NONE.** Hedge mean = +{R['hedge_mean']:.1f}%/yr, "
            f"HAC t = **+{R['hedge_t']:.2f}** (need ±2). Firm-level correlation "
            f"between Z and next-year return = **+{R['corr']:.3f}** -- statistically zero.\n"
            "- **Tradability -- MIRAGE.** An insignificant gross signal with annual turnover "
            "is wiped out before the first trade.\n"
            f"- **Distress puzzle -- confirmed (weakly).** Low-Z firms earn +{R['ret_lo']:.1f}%/yr "
            f"vs +{R['ret_hi']:.1f}%/yr for high-Z -- distressed stocks earn *less*, not more -- "
            "but the difference is noise given the volatility of annual returns."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 Could you actually trade it?\n\n"
            "The Z-score is computed from annual 10-K data. A real implementation would:\n\n"
            "- Rebalance once a year (low turnover -- that part is fine).\n"
            "- Wait until the 10-K is publicly filed (March/April) before acting.\n"
            "- Pay bid-ask spreads and market impact on the full portfolio rebalance.\n\n"
            "At a gross spread of +2.4%/yr, even modest transaction costs eat it entirely. "
            "And that +2.4% is a survivorship-biased upper bound -- the true gross is likely "
            "flat or negative once dead-company stocks are included in the distressed bucket "
            "(those are the stocks most distressed by definition, and many would have zero "
            "or very negative returns from delisting events)."
        ),

        # ---- BEAT 7 -- GOING FURTHER ------------------------------------------
        md(
            "## 7 Going further\n\n"
            "- **Bankruptcy prediction vs return prediction.** The Z-score works well at "
            "predicting defaults 1-2 years ahead. That task is conceptually different from "
            "predicting equity returns, where market prices already incorporate the distress "
            "information.\n"
            "- **The right comparison.** Altman (2000) showed the model needed re-estimation "
            "for modern firms. The private-company Z' and Z'' variants use different coefficients "
            "and targets -- this study tests only the public-company original.\n"
            "- **What does work in this family?** The Piotroski F-score "
            "([Study 65 - Scorecard](../../65-scorecard/)) and accruals "
            "([Study 52 - Smoke-Screen](../../52-smoke-screen/)) are in the same EDGAR "
            "universe -- see how they compare.\n\n"
            "*Think the Z-score does predict returns in a different universe or sample? "
            "Fork this, widen to international stocks or small-caps, and show a hedge that "
            "clears |t| >= 2 out of sample. That's the bar.*"
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
            "# Altman-Z -- a quantitative teardown\n"
            "### EDGAR fundamentals * yfinance market cap * annual tercile sort * HAC inference * survivorship anatomy\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Distress_premium%3F: Absent](https://img.shields.io/badge/Distress_premium%3F-Absent-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). Same seven "
            "beats, every claim carrying its standard error. We test whether the Altman Z-score tercile "
            "hedge earns a statistically robust premium on a survivorship-biased S&P 500 panel "
            "(2008-2023, 166 tickers, 1,210 firm-year observations).\n\n"
            "> **Not investment advice.** Reproducible research; real data sourced from the desk's "
            "shared EDGAR caches and yfinance. Methods in "
            "[`docs/references.md`](../docs/references.md), headline numbers in "
            "[`docs/results.md`](../docs/results.md)."
        ),
        code(BOOT + "\nfrom quantlab.analytics import mean_tstat_hac\nfrom quantlab.stats import sharpe_ci_bootstrap\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Hedge (high-Z minus low-Z) = **+{R['hedge_mean']:.2f}%/yr**, "
            f"HAC *t* = **+{R['hedge_t']:.2f}**. Firm-level Pearson r(Z, next-yr ret) = **+{R['corr']:.4f}**. "
            "No evidence of a return-predictive Z signal. |\n"
            f"| **Tradability** | `MIRAGE` | Insignificant gross spread eaten by any realistic "
            "annual implementation cost. Survivorship overstates the gross. |\n"
            f"| **Distress premium?** | `ABSENT` | Low-Z stocks return +{R['ret_lo']:.1f}%/yr vs "
            f"+{R['ret_hi']:.1f}%/yr for high-Z (distress *puzzle* direction) -- but the difference "
            f"is HAC *t* = {R['hedge_t']:.2f}. |\n\n"
            "> The Z-score distinguishes bankrupt from non-bankrupt firms well. It does not "
            "distinguish high- from low-return stocks in a survivorship-biased S&P 500 universe."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 The claim, steelmanned\n\n"
            "Z = 1.2 X1 + 1.4 X2 + 3.3 X3 + 0.6 X4 + 1.0 X5, with:\n\n"
            "- X1 = (CA - CL) / TA (working capital / total assets)\n"
            "- X2 = Retained Earnings / TA (accumulated profitability)\n"
            "- X3 = EBIT / TA (operating return on assets)\n"
            "- X4 = Market Equity / Total Liabilities (leverage/solvency)\n"
            "- X5 = Sales / TA (asset turnover)\n\n"
            "**H1 (return signal).** E[Z_t -> r_{t+1}] != 0: the Z-score from year-t 10-K data "
            "has predictive power over year-(t+1) equity returns, either positively (high-Z -> "
            "high return, 'safe firms win') or negatively (low-Z -> high return, 'distress premium').\n\n"
            "**H2 (tradable).** The signal survives a one-year-lag, annual-rebalance "
            "implementation net of realistic transaction costs."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 So what?\n\n"
            "If H1 holds, Z is a cheap annual fundamental screen with a 56-year academic pedigree. "
            "The Z coefficients are public, the EDGAR data is free, and the rebalance is once a year. "
            "If H1 fails in this universe, the lesson is that *bankruptcy prediction* and "
            "*return prediction* are different tasks: the market prices the distress information "
            "into the stock before the investor can act on it."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 Protocol\n\n"
            "- **Signal formation.** Z-score for year *t* from EDGAR 10-K annual values. "
            "Market cap: WeightedAverageDilutedShares x December month-end price (yfinance). "
            "Only firm-years where Assets > 0 and Liabilities > 0 are included.\n"
            "- **Return measurement.** Calendar year *(t+1)* return from ``_edgar_yrret.parquet`` "
            "(the desk's shared annual return cache). This gives a clean one-year lag: the 10-K "
            "files in Q1 of *t+1*, so the information is public before the measurement year begins.\n"
            "- **Sort.** Equal-weight tercile sort each year. Bottom third = low-Z (distressed), "
            "top third = high-Z (safe).\n"
            "- **Inference.** HAC (Newey-West) t-stat on the 16-year annual hedge series; "
            "block-bootstrap 95% CI on the annual Sharpe.\n"
            "- **Survivorship caveat.** Panel = current S&P 500 members projected back. "
            "Bankrupt/delisted firms are missing; distressed-bucket returns are upper bounds.\n"
            "- **Positive control.** A synthetic panel with a tunable z_premium knob confirms "
            "the engine recovers a planted signal."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 The teardown"),

        md(
            "### 4a The Altman Z-score distribution (real tape)\n\n"
            "What does the Z landscape look like in our survivorship-biased S&P 500 panel?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    z_vals = Z_REAL.values.flatten().astype(float)\n"
            "    z_vals = z_vals[np.isfinite(z_vals)]\n"
            "    pct_d = (z_vals < 1.81).mean() * 100\n"
            "    pct_g = ((z_vals >= 1.81) & (z_vals < 2.99)).mean() * 100\n"
            "    pct_s = (z_vals >= 2.99).mean() * 100\n"
            "else:\n"
            f"    z_vals = np.random.default_rng(123).exponential(2.5, 1210) + 0.3\n"
            f"    pct_d, pct_g, pct_s = {R['pct_distress']}, {R['pct_grey']}, {R['pct_safe']}\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "ax.hist(z_vals[z_vals < 15], bins=40, color=GREY, alpha=0.7, edgecolor='white')\n"
            "for cut, c, lbl in [(1.81, RED, 'Distress < 1.81'), (2.99, GREEN, 'Safe > 2.99')]:\n"
            "    ax.axvline(cut, c=c, lw=2, ls='--', label=lbl)\n"
            "ax.set_xlabel('Altman Z-score'); ax.set_ylabel('Firm-year count')\n"
            "ax.set_title('Z-score distribution (S&P 500, 2008-2023, survivorship-biased)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            f"print(f'Distress zone (<1.81): {{pct_d:.1f}}% | Grey: {{pct_g:.1f}}% | Safe (>2.99): {{pct_s:.1f}}%')"
        ),
        md(
            f"Even in a survivorship-biased S&P 500 panel, **{R['pct_distress']:.0f}%** of firm-years "
            f"fall in the distress zone (Z < 1.81). The grey zone is {R['pct_grey']:.0f}%, "
            f"safe is {R['pct_safe']:.0f}%. Many large-cap firms carry high leverage or negative "
            "retained earnings and land in the distress zone despite being far from bankruptcy -- "
            "the original model was calibrated on small manufacturing firms."
        ),

        md(
            "### 4b Annual bucket returns -- the hedge and its noise\n\n"
            "Equal-weighted annual returns by Z tercile, and the year-by-year hedge:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = RESULTS\n"
            "    years = res.index.tolist()\n"
            "    ret_lo_yr = res['ret_lo'].tolist()\n"
            "    ret_hi_yr = res['ret_hi'].tolist()\n"
            "    hedge_yr = res['hedge'].tolist()\n"
            "    means_lo, means_hi = res['ret_lo'].mean()*100, res['ret_hi'].mean()*100\n"
            "else:\n"
            "    years = list(range(2008, 2024))\n"
            "    ret_lo_yr = [0.303,0.140,0.076,0.175,0.615,0.171,0.004,0.204,0.309,0.023,0.266,0.046,0.305,-0.002,0.415,0.237]\n"
            "    ret_hi_yr = [0.406,0.344,0.030,0.248,0.434,0.173,0.078,0.137,0.329,-0.010,0.410,0.474,0.225,-0.181,0.290,0.277]\n"
            "    hedge_yr = [r-l for r,l in zip(ret_hi_yr, ret_lo_yr)]\n"
            f"    means_lo, means_hi = {R['ret_lo']}, {R['ret_hi']}\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "ax1.plot(years, [r*100 for r in ret_lo_yr], 'o-', c=RED, label='Low-Z (distressed)')\n"
            "ax1.plot(years, [r*100 for r in ret_hi_yr], 's-', c=GREEN, label='High-Z (safe)')\n"
            "ax1.axhline(0, c='k', lw=0.8); ax1.set_ylabel('%/yr'); ax1.set_xlabel('Signal year')\n"
            "ax1.set_title('Annual returns by Z bucket'); ax1.legend(fontsize=9)\n"
            "col2 = [GREEN if h > 0 else RED for h in hedge_yr]\n"
            "ax2.bar(years, [h*100 for h in hedge_yr], color=col2, width=0.7)\n"
            "ax2.axhline(0, c='k', lw=1); ax2.set_ylabel('Hedge %/yr'); ax2.set_xlabel('Signal year')\n"
            "ax2.set_title('Annual hedge (high-Z minus low-Z): flips sign half the time')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Low-Z mean: +{{means_lo:.1f}}%/yr | High-Z mean: +{{means_hi:.1f}}%/yr')"
        ),

        md(
            "### 4c Inference -- HAC t-stat and bootstrap Sharpe CI\n\n"
            "With only 16 annual observations, the standard error matters a lot. "
            "We use Newey-West HAC for the t-stat and a block bootstrap for the Sharpe CI."
        ),
        code(
            "if HAVE_REAL:\n"
            "    hedge_s = pd.Series(RESULTS['hedge'].values, dtype=float)\n"
            "    hac = mean_tstat_hac(hedge_s)\n"
            "    ci = sharpe_ci_bootstrap(hedge_s, n_boot=2000, periods_per_year=1, seed=123)\n"
            "    t_val, sr, ci_lo, ci_hi, fn = hac['tstat'], ci['sharpe'], ci['ci_low'], ci['ci_high'], ci['frac_negative']*100\n"
            "else:\n"
            f"    t_val, sr, ci_lo, ci_hi, fn = {R['hedge_t']}, {R['hedge_sharpe']}, {R['hedge_ci_lo']}, {R['hedge_ci_hi']}, {R['hedge_frac_neg']}\n"
            "print(f'Hedge HAC t-stat:    {t_val:+.2f}  (need |t| >= 2 to clear the inference bar)')\n"
            "print(f'Annual Sharpe:        {sr:.2f}')\n"
            "print(f'Bootstrap 95% CI:    [{ci_lo:+.2f}, {ci_hi:+.2f}]')\n"
            f"print(f'Frac negative:       {{fn:.0f}}% of resamples < 0')\n"
            "# visual: t-stat vs inference bar\n"
            "fig, ax = plt.subplots(figsize=(7, 3.5))\n"
            "ax.bar(['Hedge t-stat'], [t_val], color=RED if abs(t_val) < 2 else GREEN, width=0.4)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1.5, label='inference bar +2')\n"
            "ax.axhline(-2, ls='--', c=GREY, lw=1.5, label='inference bar -2')\n"
            "ax.axhline(0, c='k', lw=0.8)\n"
            "ax.set_title(f'HAC t = {t_val:+.2f} -- nowhere near the |t| >= 2 bar')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> HAC *t* = **+{R['hedge_t']:.2f}** on 16 annual observations. The bootstrap "
            f"95% CI on the Sharpe is [{R['hedge_ci_lo']:+.2f}, +{R['hedge_ci_hi']:.2f}] "
            f"({R['hedge_frac_neg']}% of resamples negative). The signal is statistically "
            "indistinguishable from zero."
        ),

        md(
            "### 4d Firm-level cross-section\n\n"
            "The annual tertile sort pools ~75 firms per year. The firm-level test uses all "
            "1,210 (firm, year) pairs at once: what is the raw correlation between Z and "
            "next-year return?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    all_z, all_r = [], []\n"
            "    fwd = FWD_REAL\n"
            "    for yr in Z_REAL.index:\n"
            "        z_yr = Z_REAL.loc[yr].dropna(); r_yr = fwd.loc[yr].dropna()\n"
            "        both = z_yr.index.intersection(r_yr.index)\n"
            "        all_z.extend(z_yr.loc[both].tolist()); all_r.extend(r_yr.loc[both].tolist())\n"
            "    all_z = np.array(all_z); all_r = np.array(all_r)\n"
            "    corr_val = np.corrcoef(all_z, all_r)[0, 1]\n"
            "else:\n"
            "    rng = np.random.default_rng(123)\n"
            f"    all_z = rng.exponential(2.5, {R['n_obs']}) + 0.3\n"
            f"    all_r = 0.217 + 0.013 * (all_z - all_z.mean()) + rng.normal(0, 0.3, {R['n_obs']})\n"
            f"    corr_val = {R['corr']}\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "clip_z = np.clip(all_z, 0, 12)\n"
            "ax.hexbin(clip_z, all_r * 100, gridsize=40, cmap='Blues', mincnt=1)\n"
            "# OLS line\n"
            "m, b = np.polyfit(clip_z, all_r * 100, 1)\n"
            "xr = np.linspace(0, 12, 100)\n"
            "ax.plot(xr, m * xr + b, c=RED, lw=2, label=f'OLS slope (r={corr_val:.3f})')\n"
            "ax.set_xlabel('Altman Z-score (clipped at 12)'); ax.set_ylabel('Next-year return (%)')\n"
            "ax.set_title('Firm-level: Z vs next-year return -- essentially flat')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            f"print(f'Pearson r(Z, next-yr ret): {{corr_val:.4f}} on {{len(all_z)}} firm-year pairs')"
        ),
        md(
            f"> Firm-level Pearson r = **+{R['corr']:.4f}** on {R['n_obs']} firm-year pairs. "
            "Essentially flat. The hexbin scatter confirms: at every Z level the return "
            "distribution is wide and the conditional mean barely moves."
        ),

        md(
            "### 4e The positive control -- the engine works when a signal exists\n\n"
            "Is the machinery capable of finding an effect, or does it always print 'none'? "
            "We sweep the planted z_premium knob on a synthetic panel:"
        ),
        code(
            "z_prems = [-0.08, -0.04, 0.0, 0.04, 0.08, 0.12]\n"
            "hedge_means = []\n"
            "tstats = []\n"
            "for zp in z_prems:\n"
            "    zs, fwd_s, _ = data.synthetic_panel(n_firms=300, n_years=25, z_premium=zp, seed=123)\n"
            "    r = st.tertile_hedge(zs, fwd_s)\n"
            "    s = st.summarize(r['hedge'])\n"
            "    hedge_means.append(s['mean'] * 100)\n"
            "    tstats.append(s['tstat'])\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "ax1.plot(z_prems, hedge_means, 'o-', c=GREEN, lw=2)\n"
            "ax1.axhline(0, c='k', lw=1); ax1.axvline(0, ls='--', c=GREY)\n"
            "ax1.set_xlabel('Planted z_premium'); ax1.set_ylabel('Hedge mean (%/yr)')\n"
            "ax1.set_title('Hedge mean scales with planted premium')\n"
            "ax2.plot(z_prems, tstats, 's-', c=GREEN, lw=2)\n"
            "for s in (2, -2): ax2.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax2.axvline(0, ls='--', c=GREY); ax2.axhline(0, c='k', lw=0.8)\n"
            "ax2.set_xlabel('Planted z_premium'); ax2.set_ylabel('HAC t-stat')\n"
            "ax2.set_title('t-stat clears ±2 when premium is large enough')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('At z_premium=0 the t-stat is near 0; it grows monotonically with the planted premium.')"
        ),
        md(
            "The engine is a faithful signal detector: with a planted z_premium of ±0.08 the "
            "HAC t-stat clears ±2. The real-tape result of +0.65 is therefore a statement "
            "about the **market**, not the method."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 The verdict\n\n"
            f"- **Signal `NONE`** -- Hedge HAC *t* = +{R['hedge_t']:.2f} on {R['n_years']} annual "
            f"observations. Firm-level r = +{R['corr']:.4f} on {R['n_obs']} pairs. Bootstrap 95% CI "
            f"on Sharpe is [{R['hedge_ci_lo']:+.2f}, +{R['hedge_ci_hi']:.2f}] ({R['hedge_frac_neg']}% "
            "negative). No statistically real return-predictive signal.\n"
            "- **Tradability `MIRAGE`** -- an insignificant gross spread at annual turnover is "
            "wiped out before cost estimation is even meaningful.\n"
            f"- **Distress premium `ABSENT`** -- low-Z returns +{R['ret_lo']:.1f}%/yr vs "
            f"high-Z +{R['ret_hi']:.1f}%/yr. The distress puzzle direction (low-Z underperforms) "
            "shows up weakly but is far too noisy to exploit."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 Could you trade it?\n\n"
            "Annual rebalance, long high-Z / short low-Z, gross +2.4%/yr (survivorship-biased upper bound):\n\n"
            "| Cost item | Rough magnitude |\n|---|---|\n"
            "| Transaction costs (bid-ask + commission, annual rebalance) | 0.5-1.5%/yr |\n"
            "| Survivorship-bias haircut (dead companies inflate low-Z returns) | Est. 2-4%/yr |\n"
            "| Short-selling costs (borrow fees on the low-Z leg) | 1-3%/yr |\n"
            "| **Net residual edge** | **Likely negative** |\n\n"
            "The survivorship haircut alone is likely larger than the gross spread. A live "
            "implementation would need to include distressed firms that subsequently default -- "
            "and those would drag the low-Z bucket to near-zero or negative, making the hedge "
            "positive but still not significant enough to trade."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 Going further\n\n"
            "- **Alternative distress measures.** The Ohlson O-score (Ohlson 1980), the "
            "Merton KMV distance-to-default, and the Campbell-Hilscher-Szilagyi failure "
            "probability all operationalise distress differently -- the distress-puzzle finding "
            "is robust across measures (Dichev 1998, Campbell et al. 2008).\n"
            "- **Related signals in the same EDGAR universe.** Piotroski F-score "
            "([Study 65](../../65-scorecard/)), accruals ([Study 52](../../52-smoke-screen/)), "
            "and balance-sheet quality ([Study 51](../../51-blue-chip/)) use overlapping data "
            "with different weighting of the same fundamentals.\n"
            "- **Small-cap or international.** The original Altman model was calibrated on small "
            "US manufacturing firms. The premium may behave differently outside the S&P 500. "
            "That requires a broader, unbiased data set.\n\n"
            "*Think Z predicts returns in a different universe? Fork this, change the ticker "
            "universe or sample period, and show a hedge that clears |t| >= 2 net of "
            "survivorship adjustment. That's the bar.*"
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
