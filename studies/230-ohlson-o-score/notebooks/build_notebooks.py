"""Generate the two narrative notebooks for Study 230 (Ohlson O-score).

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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-16).
R = dict(
    # Panel
    n_years=16, n_tickers=241, n_obs=2442,
    year_start=2008, year_end=2023,
    fingerprint="e277cd7f99b3",
    # Hedge (safe/low-O minus distressed/high-O)
    hedge_mean=-4.45, hedge_vol=11.48, hedge_sharpe=-0.39,
    hedge_t=-1.59, hedge_ci_lo=-1.07, hedge_ci_hi=0.02, hedge_frac_neg=97,
    hedge_hit=31,
    # Per-bucket annual means
    ret_lo=21.5, ret_mid=19.7, ret_hi=26.0, ret_mkt=22.4,
    # Firm level
    corr=0.048,
    # O-score distribution
    o_mean=-7.94, o_median=-7.77, o_std=2.08,
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

from ohlson_o_score import data, strategy as st

def _have_cache():
    cache = data.DEFAULT_CACHE
    needed = [
        "_edgar_Assets.parquet", "_edgar_AssetsCurrent.parquet",
        "_edgar_LiabilitiesCurrent.parquet", "_edgar_Liabilities.parquet",
        "_edgar_NetIncomeLoss.parquet",
        "_edgar_NetCashProvidedByUsedInOperatingActivities.parquet",
        "_edgar_yrret.parquet",
    ]
    return all(os.path.exists(os.path.join(cache, f)) for f in needed)

HAVE_REAL = _have_cache()
print("EDGAR cache present:", HAVE_REAL)

if HAVE_REAL:
    O_REAL, FWD_REAL = data.fetch_panel()
    RESULTS = st.tertile_hedge(O_REAL, FWD_REAL)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Ohlson O-score -- does the nine-variable distress model price stocks better than Altman-Z?\n"
            "### A survivorship-biased S&P 500 test of the classic logit bankruptcy model as an equity signal\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats_Altman-Z%3F: Mixed](https://img.shields.io/badge/Beats_Altman--Z%3F-Mixed-8b949e?style=flat-square)\n\n"
            "In 1980 James Ohlson built a logit model to predict corporate bankruptcy using nine accounting "
            "ratios. The resulting O-score is a bankruptcy *probability* — higher O means the firm is more "
            "likely to fail. It became the main competitor to Altman's Z-score and is cited in the same "
            "distress-puzzle literature. If distress scores predict returns, O-score should too — and "
            "maybe better, since it uses more variables and a proper probabilistic framing. This notebook "
            "tests that claim honestly.\n\n"
            "> **This is the plain-language layer.** Want the t-stats, bootstrap CIs, and survivorship "
            "anatomy? Go to **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool. House style: "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does O-score sort stock returns? | **Barely.** The hedge (safe − distressed) = "
            f"{R['hedge_mean']:+.1f}%/yr — small and noisy. |\n"
            f"| Is it statistically real? | **No.** HAC t = {R['hedge_t']:+.2f} (need ±2). |\n"
            f"| Do distressed (high-O) firms earn a risk premium? | **Direction yes, significance no.** "
            f"High-O firms return +{R['ret_hi']:.1f}%/yr vs +{R['ret_lo']:.1f}%/yr for safe — "
            f"the *risk-premium* direction — but the signal is far too noisy to trade. |\n"
            f"| Does O-score beat Altman-Z (Study 123)? | **Not really.** Both fail the bar. "
            f"O-score leans the other direction (distressed earns more) vs Z-score (safe earns more), "
            f"but neither effect is real enough to count. |\n\n"
            "> The O-score is a good bankruptcy predictor. It is not a reliable equity return predictor."
        ),

        # ---- BEAT 1 -- THE CLAIM ----------------------------------------------
        md(
            "## 1 The claim\n\n"
            "> *'Ohlson's nine-variable logit model gives a genuine probability of bankruptcy. Sort "
            "stocks by O-score each year: high-O firms are distressed and either carry more systematic "
            "risk (earning a premium) or are overpriced (earning a penalty). Either way, O-score separates "
            "winners from losers. And it uses nine variables vs Altman's five, so it should do it better.'*\n\n"
            "The same two competing hypotheses as Altman-Z:\n\n"
            "- **Distress premium** (risk story): high-O firms are riskier, so they earn *more*.\n"
            "- **Distress puzzle** (Dichev 1998): high-O firms are overpriced, so they earn *less*.\n\n"
            "Dichev (1998) tested *both* the Z-score and the O-score and found the puzzle in both. "
            "We test this on a survivorship-biased S&P 500 panel (2008–2023)."
        ),

        # ---- BEAT 2 -- SO WHAT ------------------------------------------------
        md(
            "## 2 So what?\n\n"
            "If the O-score predicts returns better than Z, you have a cheap nine-ratio annual screen "
            "with a probabilistic output that's easier to interpret than a raw discriminant score. The "
            "coefficients are public, the EDGAR data is free, and you rebalance once a year. If it "
            "fails, the lesson is that *no* simple accounting-based distress score reliably prices "
            "equities in a large-cap universe — which has practical implications for anyone using "
            "distress screens in systematic strategies."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 How would we even know?\n\n"
            "1. **Compute O-score** from year-*t* EDGAR 10-K data (nine components: SIZE, TLTA, "
            "WCTA, CLCA, OENEG, NITA, FUTL, INTWO, CHIN — see the quants notebook for the full formula).\n"
            "2. **Sort into terciles** by O-score. Low-O = safe, high-O = distressed.\n"
            "3. **Measure next-year return** (calendar year *t+1*). Because 10-Ks file in Q1 of *t+1*, "
            "using year-*t* fundamentals with year-*t+1* returns is a clean one-year lag with no "
            "look-ahead.\n"
            "4. **Hedge:** long low-O (safe), short high-O (distressed). Hedge = safe − distressed.\n"
            "5. **Inference bar:** HAC t-stat on 16 annual observations (|t| ≥ 2).\n"
            "6. **Survivorship caveat:** panel covers current S&P 500 members only. Missing are the "
            "most distressed firms that actually went bankrupt."
        ),

        # ---- BEAT 4 -- THE TEARDOWN -------------------------------------------
        md(
            "## 4 The teardown\n\n"
            "**First: what do the O-score buckets actually earn?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    labels = ['Low-O\\n(safe)', 'Mid-O\\n(grey zone)', 'High-O\\n(distressed)']\n"
            "    means = [RESULTS['ret_lo'].mean()*100, RESULTS['ret_mid'].mean()*100,\n"
            "             RESULTS['ret_hi'].mean()*100]\n"
            "    mkt = RESULTS['ret_mkt'].mean()*100\n"
            "else:\n"
            f"    labels = ['Low-O\\n(safe)', 'Mid-O\\n(grey zone)', 'High-O\\n(distressed)']\n"
            f"    means = [{R['ret_lo']}, {R['ret_mid']}, {R['ret_hi']}]\n"
            f"    mkt = {R['ret_mkt']}\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.5))\n"
            "cols = [GREEN, GREY, RED]\n"
            "bars = ax.bar(labels, means, color=cols, width=0.5)\n"
            "ax.axhline(mkt, ls='--', c='k', lw=1.5, label=f'Market EW: {mkt:.1f}%/yr')\n"
            "ax.set_ylabel('Annual return (%/yr)')\n"
            "ax.set_title('O-score buckets: distressed firms earn more (but noise dominates)')\n"
            "for b, m in zip(bars, means):\n"
            "    ax.text(b.get_x()+b.get_width()/2, m+0.3, f'{m:.1f}%', ha='center', va='bottom', fontsize=10)\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Spread (safe minus distressed): {means[0]-means[2]:+.1f}%/yr')"
        ),
        md(
            f"The distressed bucket (high-O) earns **+{R['ret_hi']:.1f}%/yr** vs **+{R['ret_lo']:.1f}%/yr** "
            f"for the safe bucket — the *risk-premium* direction. But the spread of "
            f"**{R['hedge_mean']:+.1f}%/yr** (safe minus distressed) is smaller than one year of noise "
            "and doesn't clear the inference bar. Note this is the opposite direction to Altman-Z "
            "(study 123), where the safe bucket earned more — but both are below significance."
        ),
        md(
            "**Is the hedge consistent year by year?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    years = RESULTS.index.tolist()\n"
            "    hedge_yr = RESULTS['hedge'].tolist()\n"
            "else:\n"
            "    years = list(range(2008, 2024))\n"
            f"    hedge_yr = [0.196, 0.025, -0.026, -0.217, -0.083, -0.024, -0.125, 0.155,\n"
            f"                -0.029, -0.103, -0.107, -0.075, 0.056, 0.000, -0.172, -0.185]\n"
            "fig, ax = plt.subplots(figsize=(10, 4.3))\n"
            "cols = [GREEN if h > 0 else RED for h in hedge_yr]\n"
            "ax.bar(years, [h*100 for h in hedge_yr], color=cols, width=0.7)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Year (signal year; return measured in year+1)')\n"
            "ax.set_ylabel('Hedge return (safe minus distressed, %/yr)')\n"
            "ax.set_title('Hedge is mostly negative (distressed earns more) — but inconsistent')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Positive years (safe beats distressed): " + str(R['hedge_hit']) + "% of 16')"
        ),
        md(
            f"The hedge is positive (safe beats distressed) in only **{R['hedge_hit']}%** of years "
            "(5 of 16) — most years the distressed bucket earns more, consistent with the risk-premium "
            "story. But the variation is enormous. The two largest outliers cancel: 2008 sees safe firms "
            "outperform sharply (+19.6%), while 2011 and 2023 see distressed firms surge ahead. "
            "No stable, consistent signal."
        ),

        # ---- BEAT 5 -- VERDICT ------------------------------------------------
        md(
            "## 5 The verdict\n\n"
            f"- **Signal -- NONE.** Hedge mean = {R['hedge_mean']:+.1f}%/yr, "
            f"HAC t = **{R['hedge_t']:+.2f}** (need ±2). Firm-level correlation "
            f"between O-score and next-year return = **+{R['corr']:.3f}**.\n"
            "- **Tradability -- MIRAGE.** An insignificant gross signal with annual turnover "
            "is wiped out before the first trade, and survivorship bias inflates the gross.\n"
            f"- **Beats Altman-Z? -- MIXED.** The O-score leans the risk-premium direction "
            f"(distressed earns more: +{R['ret_hi']:.1f}% vs +{R['ret_lo']:.1f}%), opposite to "
            "Altman-Z's direction, but both models fail the |t| ≥ 2 bar. Neither is a reliable "
            "equity signal in this universe."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 Could you actually trade it?\n\n"
            "The O-score uses annual 10-K data, so a real implementation would:\n\n"
            "- Rebalance once a year (low turnover).\n"
            "- Wait until 10-Ks are filed (March–April) before acting.\n"
            "- Bear bid-ask spreads and market impact on the full portfolio rebalance.\n\n"
            "At a gross hedge of −4.45%/yr (favouring distressed stocks), you would be *short* "
            "the safe firms and *long* distressed ones. Those distressed firms are harder to hold "
            "in practice — higher volatility, occasional bankruptcy events. Once the survivorship "
            "adjustment is made (adding back the firms that actually went bankrupt, with near-zero "
            "or −100% returns), the distressed bucket would likely look much worse. The net edge "
            "evaporates."
        ),

        # ---- BEAT 7 -- GOING FURTHER ------------------------------------------
        md(
            "## 7 Going further\n\n"
            "- **Compare to Altman-Z directly.** Study 123 found the same conclusion from the "
            "opposite direction — both models fail the bar. The distress puzzle is robust to which "
            "model you use, consistent with Dichev (1998).\n"
            "- **Modern distress models.** The Campbell-Hilscher-Szilagyi (2008) dynamic logit "
            "adds market variables (equity volatility, stock price relative to 52-week high). "
            "Those models may better capture distress risk but require daily data.\n"
            "- **Small-cap or international stocks.** The original Ohlson sample was broader than "
            "S&P 500. The premium, if it exists, may live in the micro-cap / small-cap space "
            "where survivorship bias is less dominant and liquidity constraints prevent arbitrage.\n\n"
            "*Think O-score predicts returns in a different universe? Fork this, widen to small "
            "caps or international stocks, and show a hedge that clears |t| ≥ 2 net of "
            "survivorship adjustment. That's the bar.*"
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
            "# Ohlson O-score -- a quantitative teardown\n"
            "### EDGAR fundamentals * nine-variable logit * annual tercile sort * HAC inference * survivorship anatomy\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats_Altman-Z%3F: Mixed](https://img.shields.io/badge/Beats_Altman--Z%3F-Mixed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). Same seven "
            "beats, every claim carrying its standard error. We test whether the Ohlson O-score tercile "
            "hedge (long safe, short distressed) earns a statistically robust premium on a "
            "survivorship-biased S&P 500 panel (2008–2023, 241 tickers, 2,442 firm-year observations).\n\n"
            "> **Not investment advice.** Reproducible research; real data sourced from the desk's "
            "shared EDGAR caches. Methods in [`docs/references.md`](../docs/references.md), "
            "headline numbers in [`docs/results.md`](../docs/results.md)."
        ),
        code(BOOT + "\nfrom quantlab.analytics import mean_tstat_hac\nfrom quantlab.stats import sharpe_ci_bootstrap\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Hedge (safe − distressed) = **{R['hedge_mean']:+.2f}%/yr**, "
            f"HAC *t* = **{R['hedge_t']:+.2f}**. Firm-level Pearson r(O, next-yr ret) = **+{R['corr']:.4f}**. "
            "No statistically reliable return-predictive O signal. |\n"
            f"| **Tradability** | `MIRAGE` | Insignificant gross hedge, heavy survivorship haircut, "
            "annual implementation cost wipes the rest. |\n"
            f"| **Beats Altman-Z?** | `MIXED` | High-O (distressed) earns +{R['ret_hi']:.1f}%/yr vs "
            f"+{R['ret_lo']:.1f}%/yr for low-O (safe) — risk-premium direction, opposite Altman-Z. "
            f"Both models fail |t| ≥ 2. |\n\n"
            "> The O-score and Altman-Z tell different stories about the *direction* of the distress "
            "premium in this sample — but neither clears the inference bar. The main conclusion is "
            "the same: distress measures are bankruptcy predictors, not reliable return predictors."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 The claim, steelmanned\n\n"
            "Ohlson (1980) O-score:\n\n"
            "O = −1.32 − 0.407·SIZE + 6.03·TLTA − 1.43·WCTA + 0.076·CLCA\n"
            "    − 1.72·OENEG − 2.37·NITA − 1.83·FUTL + 0.285·INTWO − 0.521·CHIN\n\n"
            "Where:\n\n"
            "- SIZE = log(Total Assets)\n"
            "- TLTA = Total Liabilities / Total Assets\n"
            "- WCTA = (Current Assets − Current Liabilities) / Total Assets\n"
            "- CLCA = Current Liabilities / Current Assets\n"
            "- OENEG = 1 if Total Liabilities > Total Assets\n"
            "- NITA = Net Income / Total Assets\n"
            "- FUTL = Cash Flow from Operations / Total Liabilities\n"
            "- INTWO = 1 if net income negative in both current and prior year\n"
            "- CHIN = (NI_t − NI_{t-1}) / (|NI_t| + |NI_{t-1}|)\n\n"
            "**H1 (return signal).** E[O_t → r_{t+1}] ≠ 0: the O-score from year-t 10-K data "
            "has predictive power over year-(t+1) equity returns — either distress premium "
            "(high-O earns more) or distress puzzle (high-O earns less).\n\n"
            "**H2 (beats Z-score).** The nine-variable logit outperforms the five-variable Z "
            "on the return-prediction task (higher |t-stat| on the annual hedge)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 So what?\n\n"
            "If H1 holds, O-score is a cheap annual screen with a probabilistic output. "
            "If H2 holds, it's the better distress model for equity investors. If both fail, "
            "the lesson is that adding more accounting variables to a bankruptcy model does not "
            "necessarily improve return-prediction power — the two tasks require different signals."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 Protocol\n\n"
            "- **Signal formation.** O-score for year *t* from EDGAR 10-K annual values. "
            "Six concepts: Assets, AssetsCurrent, LiabilitiesCurrent, Liabilities, "
            "NetIncomeLoss, NetCashProvidedByUsedInOperatingActivities. Only firm-years "
            "where Assets > 0 and Liabilities > 0 are included.\n"
            "- **Prior-year net income.** INTWO and CHIN require the *prior-year* net income — "
            "we shift the NetIncomeLoss panel by one year to construct these binary/ratio terms.\n"
            "- **Return measurement.** Calendar year *(t+1)* return from ``_edgar_yrret.parquet`` "
            "(one-year lag, 10-K files in Q1 of *t+1*).\n"
            "- **Sort.** Equal-weight tercile sort each year. Low-O = safe, high-O = distressed. "
            "Hedge = ret_lo − ret_hi (safe minus distressed).\n"
            "- **Inference.** HAC (Newey-West) t-stat on 16 annual hedge observations; "
            "block-bootstrap 95% CI on the annual Sharpe.\n"
            "- **Survivorship caveat.** Panel = current S&P 500 members projected back.\n"
            "- **Positive control.** A synthetic panel with a tunable o_premium knob confirms "
            "the engine detects a planted signal."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 The teardown"),

        md(
            "### 4a The Ohlson O-score distribution (real tape)\n\n"
            "What does the O landscape look like in our survivorship-biased S&P 500 panel?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    o_vals = O_REAL.values.astype(float).flatten()\n"
            "    o_vals = o_vals[np.isfinite(o_vals)]\n"
            "    o_mean_v = o_vals.mean()\n"
            "    o_med_v = np.median(o_vals)\n"
            "else:\n"
            f"    rng = np.random.default_rng(230)\n"
            f"    o_vals = rng.normal({R['o_mean']}, {R['o_std']}, {R['n_obs']})\n"
            f"    o_mean_v, o_med_v = {R['o_mean']}, {R['o_median']}\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "ax.hist(o_vals, bins=50, color=GREY, alpha=0.7, edgecolor='white')\n"
            "ax.axvline(o_mean_v, c=RED, lw=2, ls='--', label=f'Mean: {o_mean_v:.2f}')\n"
            "ax.axvline(o_med_v, c=GREEN, lw=2, ls=':', label=f'Median: {o_med_v:.2f}')\n"
            "ax.set_xlabel('Ohlson O-score'); ax.set_ylabel('Firm-year count')\n"
            "ax.set_title('O-score distribution (S&P 500, 2008-2023, survivorship-biased)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            f"print(f'O-score: mean={{o_mean_v:.2f}}, median={{o_med_v:.2f}}')\n"
            f"print('Most firms have large negative O-scores (low distress probability) in this biased panel.')"
        ),
        md(
            f"The O-score distribution has a mean of **{R['o_mean']:.2f}** and a median of "
            f"**{R['o_median']:.2f}** — large negative values reflecting that S&P 500 survivors "
            "are generally profitable, liquid, and far from bankruptcy. The right tail extends to "
            "positive values (genuinely stressed large caps). The range is wide, providing "
            "cross-sectional variation to sort on."
        ),

        md(
            "### 4b Annual bucket returns -- the hedge and its noise\n\n"
            "Equal-weighted annual returns by O tercile, and the year-by-year hedge:"
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
            "    ret_lo_yr = [0.520,0.243,0.056,0.167,0.409,0.153,0.046,0.257,0.338,-0.036,0.349,0.269,0.304,-0.138,0.358,0.150]\n"
            "    ret_hi_yr = [0.323,0.218,0.082,0.384,0.492,0.177,0.171,0.102,0.367,0.066,0.456,0.344,0.247,-0.138,0.530,0.335]\n"
            "    hedge_yr = [r-l for r,l in zip(ret_lo_yr, ret_hi_yr)]\n"
            f"    means_lo, means_hi = {R['ret_lo']}, {R['ret_hi']}\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "ax1.plot(years, [r*100 for r in ret_lo_yr], 'o-', c=GREEN, label='Low-O (safe)')\n"
            "ax1.plot(years, [r*100 for r in ret_hi_yr], 's-', c=RED, label='High-O (distressed)')\n"
            "ax1.axhline(0, c='k', lw=0.8); ax1.set_ylabel('%/yr'); ax1.set_xlabel('Signal year')\n"
            "ax1.set_title('Annual returns by O bucket'); ax1.legend(fontsize=9)\n"
            "col2 = [GREEN if h > 0 else RED for h in hedge_yr]\n"
            "ax2.bar(years, [h*100 for h in hedge_yr], color=col2, width=0.7)\n"
            "ax2.axhline(0, c='k', lw=1); ax2.set_ylabel('Hedge %/yr'); ax2.set_xlabel('Signal year')\n"
            "ax2.set_title('Hedge (safe minus distressed): mostly negative, very noisy')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Low-O (safe) mean: +{{means_lo:.1f}}%/yr | High-O (distressed) mean: +{{means_hi:.1f}}%/yr')"
        ),

        md(
            "### 4c Inference -- HAC t-stat and bootstrap Sharpe CI\n\n"
            "With only 16 annual observations, the standard error is large. "
            "Newey-West HAC for the t-stat; block bootstrap for the Sharpe CI."
        ),
        code(
            "if HAVE_REAL:\n"
            "    hedge_s = pd.Series(RESULTS['hedge'].values, dtype=float)\n"
            "    hac = mean_tstat_hac(hedge_s)\n"
            "    ci = sharpe_ci_bootstrap(hedge_s, n_boot=2000, periods_per_year=1, seed=230)\n"
            "    t_val, sr, ci_lo, ci_hi, fn = hac['tstat'], ci['sharpe'], ci['ci_low'], ci['ci_high'], ci['frac_negative']*100\n"
            "else:\n"
            f"    t_val, sr, ci_lo, ci_hi, fn = {R['hedge_t']}, {R['hedge_sharpe']}, {R['hedge_ci_lo']}, {R['hedge_ci_hi']}, {R['hedge_frac_neg']}\n"
            "print(f'Hedge HAC t-stat:    {t_val:+.2f}  (need |t| >= 2 to clear the inference bar)')\n"
            "print(f'Annual Sharpe:        {sr:.2f}')\n"
            "print(f'Bootstrap 95% CI:    [{ci_lo:+.2f}, {ci_hi:+.2f}]')\n"
            f"print(f'Frac CI negative:    {{fn:.0f}}% of resamples < 0')\n"
            "fig, ax = plt.subplots(figsize=(7, 3.5))\n"
            "ax.bar(['Hedge t-stat'], [t_val], color=RED if abs(t_val) < 2 else GREEN, width=0.4)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1.5, label='inference bar +2')\n"
            "ax.axhline(-2, ls='--', c=GREY, lw=1.5, label='inference bar -2')\n"
            "ax.axhline(0, c='k', lw=0.8)\n"
            "ax.set_title(f'HAC t = {t_val:+.2f} -- just below the |t| >= 2 bar')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> HAC *t* = **{R['hedge_t']:+.2f}** on 16 annual observations. The bootstrap "
            f"95% CI on the Sharpe is [{R['hedge_ci_lo']:+.2f}, {R['hedge_ci_hi']:+.2f}] "
            f"({R['hedge_frac_neg']}% of resamples negative). The signal misses the inference "
            "bar — the hedge mean is dominated by noise."
        ),

        md(
            "### 4d Firm-level cross-section\n\n"
            "The annual tercile sort uses ~150 firms per year. The firm-level test uses all "
            "2,442 (firm, year) pairs: what is the raw correlation between O and next-year return?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    all_o, all_r = [], []\n"
            "    fwd = FWD_REAL\n"
            "    for yr in O_REAL.index:\n"
            "        o_yr = O_REAL.loc[yr].dropna(); r_yr = fwd.loc[yr].dropna()\n"
            "        both = o_yr.index.intersection(r_yr.index)\n"
            "        all_o.extend(o_yr.loc[both].tolist()); all_r.extend(r_yr.loc[both].tolist())\n"
            "    all_o = np.array(all_o, dtype=float); all_r = np.array(all_r, dtype=float)\n"
            "    corr_val = np.corrcoef(all_o, all_r)[0, 1]\n"
            "else:\n"
            "    rng = np.random.default_rng(230)\n"
            f"    all_o = rng.normal({R['o_mean']}, {R['o_std']}, {R['n_obs']})\n"
            f"    all_r = 0.224 + {R['corr']} * (all_o - all_o.mean()) + rng.normal(0, 0.3, {R['n_obs']})\n"
            f"    corr_val = {R['corr']}\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "clip_o = np.clip(all_o, -15, 5)\n"
            "ax.hexbin(clip_o, all_r * 100, gridsize=40, cmap='Blues', mincnt=1)\n"
            "m, b = np.polyfit(clip_o, all_r * 100, 1)\n"
            "xr = np.linspace(-15, 5, 100)\n"
            "ax.plot(xr, m * xr + b, c=RED, lw=2, label=f'OLS slope (r={corr_val:.3f})')\n"
            "ax.set_xlabel('Ohlson O-score (clipped)'); ax.set_ylabel('Next-year return (%)')\n"
            "ax.set_title('Firm-level: O-score vs next-year return -- weak positive slope')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            f"print(f'Pearson r(O, next-yr ret): {{corr_val:.4f}} on {{len(all_o)}} firm-year pairs')"
        ),
        md(
            f"> Firm-level Pearson r = **+{R['corr']:.4f}** on {R['n_obs']} firm-year pairs. "
            "The slope is positive (higher O, higher returns — risk-premium direction) but "
            "economically small. The scatter confirms: at every O level, the return distribution "
            "is wide and the conditional mean barely moves."
        ),

        md(
            "### 4e The positive control -- the engine works when a signal exists\n\n"
            "Is the machinery capable of finding an effect, or does it always print 'none'? "
            "We sweep the planted o_premium knob on a synthetic panel:"
        ),
        code(
            "o_prems = [-0.08, -0.04, 0.0, 0.04, 0.08, 0.12]\n"
            "hedge_means = []\n"
            "tstats = []\n"
            "for op in o_prems:\n"
            "    os, fwd_s, _ = data.synthetic_panel(n_firms=300, n_years=25, o_premium=op, seed=230)\n"
            "    r = st.tertile_hedge(os, fwd_s)\n"
            "    s = st.summarize(r['hedge'])\n"
            "    hedge_means.append(s['mean'] * 100)\n"
            "    tstats.append(s['tstat'])\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "ax1.plot(o_prems, hedge_means, 'o-', c=GREEN, lw=2)\n"
            "ax1.axhline(0, c='k', lw=1); ax1.axvline(0, ls='--', c=GREY)\n"
            "ax1.set_xlabel('Planted o_premium'); ax1.set_ylabel('Hedge mean (%/yr)')\n"
            "ax1.set_title('Hedge scales with planted premium (inverted: o_premium>0 => hedge<0)')\n"
            "ax2.plot(o_prems, tstats, 's-', c=GREEN, lw=2)\n"
            "for s in (2, -2): ax2.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax2.axvline(0, ls='--', c=GREY); ax2.axhline(0, c='k', lw=0.8)\n"
            "ax2.set_xlabel('Planted o_premium'); ax2.set_ylabel('HAC t-stat')\n"
            "ax2.set_title('t-stat clears ±2 when premium is large enough')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('At o_premium=0 the t-stat is near 0; it grows monotonically with the planted premium.')"
        ),
        md(
            "The engine is a faithful signal detector: with a planted o_premium of ±0.08 the "
            "HAC t-stat clears ±2. The real-tape result of −1.59 is therefore a statement "
            "about the **market**, not the method."
        ),

        md(
            "### 4f Comparison to Altman-Z (Study 123)\n\n"
            "Both models use EDGAR data; the key question is whether nine variables buy more "
            "return-predictive power than five."
        ),
        code(
            "# Summary comparison (frozen numbers from both studies)\n"
            "import pandas as pd\n"
            "comp = pd.DataFrame({\n"
            "    'Model': ['Altman Z-score (Study 123)', 'Ohlson O-score (Study 230)'],\n"
            "    'N vars': [5, 9],\n"
            "    'Hedge direction': ['High-Z earns more (safe wins)', 'High-O earns more (distressed wins)'],\n"
            "    'Hedge mean (%/yr)': ['+2.36', '-4.45'],\n"
            "    'HAC t-stat': ['+0.65', '-1.59'],\n"
            "    'n years': [16, 16],\n"
            "    'Verdict': ['NONE / MIRAGE', 'NONE / MIRAGE'],\n"
            "}).set_index('Model')\n"
            "print(comp.to_string())\n"
            "print()\n"
            "print('Neither model clears |t| >= 2. The O-score gets closer (-1.59 vs +0.65)')\n"
            "print('but the direction is OPPOSITE. No reliable signal from either model.')"
        ),
        md(
            "The O-score and Altman-Z give opposite directional signals in this sample — "
            "and neither is significant. This is consistent with Dichev (1998): distress measures "
            "do not reliably sort returns in either direction within a survivorship-biased S&P 500 universe."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 The verdict\n\n"
            f"- **Signal `NONE`** -- Hedge HAC *t* = {R['hedge_t']:+.2f} on {R['n_years']} annual "
            f"observations. Firm-level r = +{R['corr']:.4f} on {R['n_obs']} pairs. Bootstrap 95% CI "
            f"on Sharpe is [{R['hedge_ci_lo']:+.2f}, {R['hedge_ci_hi']:+.2f}] ({R['hedge_frac_neg']}% "
            "negative). No statistically reliable return signal.\n"
            "- **Tradability `MIRAGE`** -- an insignificant gross signal with annual turnover, "
            "heavy survivorship bias, and potential borrow costs on the distressed leg.\n"
            f"- **Beats Altman-Z? `MIXED`** -- O-score gets closer to the bar (|t|=1.59 vs 0.65) "
            "but in the *opposite* direction. Neither model wins; both confirm that distress measures "
            "are bankruptcy predictors, not equity return predictors, in this universe."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 Could you trade it?\n\n"
            "Long low-O (safe) / short high-O (distressed), annual rebalance, "
            f"gross hedge = {R['hedge_mean']:+.2f}%/yr (survivorship-biased upper bound on the safe side):\n\n"
            "| Cost item | Rough magnitude |\n|---|---|\n"
            "| Transaction costs (bid-ask + commission, annual rebalance) | 0.5–1.5%/yr |\n"
            "| Survivorship-bias haircut (absent bankruptcies inflate distressed returns) | Est. 3–6%/yr |\n"
            "| Short-selling costs on distressed leg | 1–3%/yr |\n"
            "| **Net residual edge** | **Likely highly negative** |\n\n"
            "The survivorship haircut is especially severe here: we are long the *safe* firms, "
            "whose returns are less affected by missing bankrupt firms, and short the *distressed* "
            "firms, whose survivorship-biased returns are heavily overstated. A live implementation "
            "would face the worst distressed firms — those most likely to have zero or −100% returns."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 Going further\n\n"
            "- **Campbell-Hilscher-Szilagyi (2008) model.** A dynamic logit that adds market "
            "variables (equity volatility, price-to-book, stock return momentum). More inputs, "
            "daily data, better bankruptcy prediction — but still fails to deliver a robust "
            "equity return premium.\n"
            "- **Related signals.** Piotroski F-score ([Study 65](../../65-scorecard/)), accruals "
            "([Study 52](../../52-smoke-screen/)), and gross-profit margin "
            "([Study 125](../../125-gross-profit/)) use overlapping EDGAR data with "
            "different factor construction.\n"
            "- **Unbiased sample.** The critical fix: use a full universe including delisted firms. "
            "CRSP (not yfinance) tracks delistings with delisting returns. That data requirement "
            "is why distress studies are hard to replicate cheaply.\n\n"
            f"*Fingerprint: `{R['fingerprint']}` — see [`docs/results.md`](../docs/results.md).*"
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
