"""Generate the two narrative notebooks for Study 130 (Vol-Risk-Premium).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
parquet under ../_cache/ if present and otherwise quote the frozen headline numbers in
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-14).
R = dict(
    n=8379,
    date_start="1993-03-02", date_end="2026-06-12",
    mean_vix=19.53, mean_rv=15.85, mean_vrp=3.68, std_vrp=5.02,
    pct_pos_vrp=85.7, t_vrp=22.95,
    # Quintile table h=1
    q1_mean=1.3, q5_mean=8.3, q5_t=2.94, q1_t=0.41,
    spread_h1=7.0, t_spread_h1=1.67,
    spread_h5=16.5, t_spread_h5=1.11,
    spread_h21=58.6, t_spread_h21=1.51,
    q5_t_h21=5.67, q1_t_h21=2.72,
    # Timing overlay
    sharpe_passive=0.55, sharpe_active=0.50, spread_mean=-1.5, spread_t=-1.69,
    n_switches=981,
    # Short-vol payoff
    sv_n=398, sv_mean=0.86, sv_skew=-0.73, sv_kurt=2.64,
    sv_pct_neg=36.2, sv_p05=-6.6, sv_worst=-19.7,
)

# ---------------------------------------------------------------------------
# Shared analysis preamble.
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from vol_risk_premium import data, strategy as st

_CACHE = os.path.join(os.path.dirname(os.path.abspath(".")), "_cache")

def _have_cache():
    return os.path.exists(data._cache_path(_CACHE))

HAVE_REAL = _have_cache()
print("real daily cache present:", HAVE_REAL)
"""

LOAD_REAL = """\
if HAVE_REAL:
    df = data.fetch_daily(fetch=False, cache_dir=_CACHE)
"""

# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Vol-Risk-Premium -- does implied vol reliably beat realised vol?\n"
            "### The IV-RV spread: a real premium, a complicated trade\n\n"
            "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![VRP_premium%3F: Real](https://img.shields.io/badge/VRP_premium%3F-Real-2ea44f?style=flat-square)\n\n"
            "Every time you buy an option, someone sells it to you. That seller is charging "
            "more for vol insurance than they expect to pay out -- on average. The difference "
            "between what the market *implies* vol will be (the VIX) and what vol actually "
            "*realises* (measured on SPY daily returns) is the **variance risk premium (VRP)**: "
            "a compensation for taking the wrong side of a negative-skew trade. This notebook "
            "asks: how big is it, can you time equities with it, and can you actually trade it?\n\n"
            "> **This is the plain-language layer.** Want the t-stats, bootstrap intervals, and "
            "regime analysis? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is "
            "drawn by the code beside it. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + LOAD_REAL),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Is there really a premium (VIX > RV)? | **Yes.** The average spread is "
            f"**+{R['mean_vrp']:.1f} pp**, positive **{R['pct_pos_vrp']:.0f}%** of days, "
            f"HAC *t* = **{R['t_vrp']:.1f}** across 33 years. |\n"
            f"| Does a high VRP predict good equity returns? | **Weakly.** High-VRP days "
            f"(Q5) show significantly positive returns (t = {R['q5_t']:.1f}) but the Q5-Q1 "
            f"spread (*t* = {R['t_spread_h1']:.1f}) does not clear the inference bar. |\n"
            "| Can you time the market with it? | **No.** A VRP-above-median timer "
            f"underperforms buy-and-hold (Sharpe {R['sharpe_active']:.2f} vs "
            f"{R['sharpe_passive']:.2f}). |\n"
            "| Can you trade short vol? | **Fragile.** You earn a positive mean but the "
            f"distribution has skew = {R['sv_skew']:.1f} and a worst 21-day period of "
            f"**{R['sv_worst']:.0f}%**. |\n\n"
            "> The VRP is one of the most documented premia in finance. It is *real* as a "
            "premium (option sellers are paid). It is *weak* as an equity timing signal. And "
            "harvesting it directly is structurally a negative-skew trade -- steady gains "
            "interrupted by severe losses when vol spikes."
        ),

        # ---- BEAT 1 -- THE CLAIM -----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The VIX -- the market's implied volatility -- consistently overestimates "
            "how volatile the market actually turns out to be. The gap (implied minus "
            "realised) is a risk premium: option buyers pay a fear tax. Smart money "
            "collects it by selling vol. And when the VRP is high, equities tend to "
            "do well -- it signals the market has over-priced fear.\"*\n\n"
            "It's a two-part claim: (1) IV > RV on average -- the premium exists; and "
            "(2) the size of the VRP predicts forward equity returns. Both are extensively "
            "studied (Bollerslev et al. 2009, Carr & Wu 2009) and both have empirical "
            "support of varying strength -- which is exactly why an honest teardown matters."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If both claims hold, you have two edges: you can harvest the premium by "
            "selling options, and you can time equity positions using the VRP level. "
            "If only the first holds (premium is real, timing is not), you need options "
            "infrastructure to collect it -- the VRP is not a free equity signal. If "
            "neither holds, the VIX is just noise. The three outcomes lead to very "
            "different trading strategies, so the distinction matters."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW ----------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "We measure the VRP as ``VIX - RV21`` where ``VIX`` is the CBOE index "
            "(annualised %) and ``RV21`` is the trailing 21-day standard deviation of "
            "SPY log-returns scaled to the same units. Three tests:\n\n"
            "1. **Premium test.** Is the mean VRP positive with HAC t ≥ 2? "
            "(Controls for autocorrelation in daily VIX/RV.)\n"
            "2. **Predictability test.** Sort days into quintiles by rolling VRP rank "
            "(out-of-sample, no look-ahead); check if Q5 earns more than Q1 with HAC "
            "*t* ≥ 2 on the spread.\n"
            "3. **Timer test.** Long SPY when VRP is above its rolling 252-day median, "
            "flat when below; compare to unconditional buy-and-hold.\n\n"
            "Data: ~33 years of daily VIX and SPY (March 1993 -- June 2026), covering "
            "the dot-com crash, GFC, COVID-19 (VIX peak 82.7 in March 2020), and the "
            "2022 rates shock."
        ),

        # ---- BEAT 4 -- THE TEARDOWN -----------------------------------------
        md(
            "## 4 · The teardown -- let's actually look\n\n"
            "### First: is the premium real?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ps = st.vrp_premium_stats(df)\n"
            "    vix_mean, rv_mean, vrp_mean = ps['mean_vix'], ps['mean_rv'], ps['mean_vrp']\n"
            "    pct_pos, t_vrp = ps['pct_positive'], ps['tstat']\n"
            "    vrp_s = df['VRP'].dropna()\n"
            "else:\n"
            f"    vix_mean, rv_mean, vrp_mean = {R['mean_vix']}, {R['mean_rv']}, {R['mean_vrp']}\n"
            f"    pct_pos, t_vrp = {R['pct_pos_vrp']}, {R['t_vrp']}\n"
            "    vrp_s = None\n\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))\n"
            "ax = axes[0]\n"
            "if HAVE_REAL:\n"
            "    ax.hist(df['VRP'].dropna(), bins=80, color=GREEN, alpha=0.7, edgecolor='none')\n"
            "ax.axvline(0, c='k', lw=1.5)\n"
            "ax.axvline(vrp_mean, c=GREEN, lw=2, ls='--', label=f'mean={vrp_mean:.1f}pp')\n"
            "ax.set_xlabel('VRP (VIX minus 21d realised vol, pp)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f'VRP distribution: positive {pct_pos:.0f}% of days, t={t_vrp:.0f}')\n"
            "ax.legend()\n"
            "ax2 = axes[1]\n"
            "bars = ax2.bar(['Mean VIX', 'Mean 21d RV', 'VRP = IV - RV'],\n"
            "               [vix_mean, rv_mean, vrp_mean],\n"
            "               color=[AMBER, GREY, GREEN])\n"
            "ax2.axhline(0, c='k', lw=1)\n"
            "ax2.set_ylabel('Annualised vol (pct)')\n"
            "ax2.set_title('VIX systematically exceeds realised vol')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'VRP: {vrp_mean:.2f} pp average, positive {pct_pos:.1f}% of days, HAC t={t_vrp:.2f}')"
        ),
        md(
            f"The premium is **unambiguously real**: VIX = {R['mean_vix']:.1f}% vs RV = "
            f"{R['mean_rv']:.1f}%, a gap of **{R['mean_vrp']:.1f} percentage points** on average "
            f"(HAC *t* = {R['t_vrp']:.0f}). This is the market saying: 'volatility *feels* worse "
            "than it usually turns out to be.' Option sellers earn this premium for being short "
            "that fear.\n\n"
            "> Why does it persist? Investors are willing to pay above-fair-value for insurance "
            "against vol spikes (the 2008 GFC, March 2020). The VRP is the compensation that "
            "makes selling that insurance rational despite the crash risk."
        ),
        md("### Does a high VRP predict better equity returns?"),
        code(
            "if HAVE_REAL:\n"
            "    qt = st.quintile_table(df, horizons=[1, 21])\n"
            "    q1_m = qt[(qt['quintile']==1)&(qt['horizon']==1)]['mean_bps'].values[0]\n"
            "    q5_m = qt[(qt['quintile']==5)&(qt['horizon']==1)]['mean_bps'].values[0]\n"
            "    q5_t = qt[(qt['quintile']==5)&(qt['horizon']==1)]['t_stat'].values[0]\n"
            "    q1_t_val = qt[(qt['quintile']==1)&(qt['horizon']==1)]['t_stat'].values[0]\n"
            "    tb = st.top_minus_bottom(df, horizon=1)\n"
            "    spread, t_sp = tb['spread_bps'], tb['t_spread']\n"
            "else:\n"
            f"    q1_m, q5_m = {R['q1_mean']}, {R['q5_mean']}\n"
            f"    q5_t, q1_t_val = {R['q5_t']}, {R['q1_t']}\n"
            f"    spread, t_sp = {R['spread_h1']}, {R['t_spread_h1']}\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "qvals = [q1_m, None, None, None, q5_m]\n"
            "if HAVE_REAL:\n"
            "    qt1 = qt[qt['horizon']==1].sort_values('quintile')\n"
            "    qvals = qt1['mean_bps'].tolist()\n"
            "colors = [RED, AMBER, GREY, AMBER, GREEN]\n"
            "ax.bar([f'Q{i+1}' for i in range(5)], qvals, color=colors)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('VRP quintile (Q1=low, Q5=high)')\n"
            "ax.set_ylabel('1-day forward SPY return (bps)')\n"
            "ax.set_title('High-VRP days earn more, but Q1 is not much worse')\n"
            "ax.annotate(f't={q5_t:.1f}', xy=(4, q5_m), ha='center', va='bottom', fontsize=10, color=GREEN)\n"
            "ax.annotate(f't={q1_t_val:.1f}', xy=(0, q1_m), ha='center', va='bottom', fontsize=10, color=RED)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Q5 mean: {q5_m:.1f}bps (t={q5_t:.2f})  Q1 mean: {q1_m:.1f}bps (t={q1_t_val:.2f})')\n"
            "print(f'Q5-Q1 spread: {spread:.1f}bps  HAC t = {t_sp:.2f}  (|t|<2: does not clear the inference bar)')"
        ),
        md(
            f"**The story is nuanced.** High-VRP days (Q5) do earn more: {R['q5_mean']:.1f} bps/day "
            f"with t = {R['q5_t']:.2f} -- statistically real. But low-VRP days (Q1) also earn "
            f"positive returns: {R['q1_mean']:.1f} bps with t = {R['q1_t']:.2f}. The Q5-Q1 spread "
            f"is {R['spread_h1']:.1f} bps, with HAC t = {R['t_spread_h1']:.2f} -- **below the "
            f"|t| ≥ 2 inference bar**. The signal exists in Q5 alone but the relative comparison "
            "is noisier.\n\n"
            "> Why does Q1 also earn positive returns? Low-VRP periods often occur during extended "
            "market rallies where VIX stays depressed for months. The equity risk premium doesn't "
            "disappear just because fear is low -- it can coexist with good returns."
        ),
        md("### The timer: does going flat when VRP is low help?"),
        code(
            "if HAVE_REAL:\n"
            "    to = st.timing_overlay(df, cost_bps=0)\n"
            "    p_s = st.summarize(to['r_passive'].dropna())\n"
            "    a_s = st.summarize(to['r_active'].dropna())\n"
            "    sp_s = st.summarize(to['r_spread'].dropna())\n"
            "    sh_p, sh_a = p_s['sharpe_ann'], a_s['sharpe_ann']\n"
            "else:\n"
            f"    sh_p, sh_a = {R['sharpe_passive']}, {R['sharpe_active']}\n"
            "    sp_s = dict(mean_bps=R['spread_mean'], t_stat=R['spread_t'])\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.5))\n"
            "bars = ax.bar(['Buy-and-hold (passive)', 'VRP timer (active)'], [sh_p, sh_a],\n"
            "              color=[GREY, AMBER], width=0.5)\n"
            "ax.set_ylabel('Annualised Sharpe ratio')\n"
            "ax.set_title('The VRP timer does not beat buy-and-hold')\n"
            "for b in bars:\n"
            "    h = b.get_height()\n"
            "    ax.annotate(f'{h:.2f}', (b.get_x()+b.get_width()/2, h+0.005), ha='center', fontsize=12)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Spread (active-passive): {sp_s[\"mean_bps\"]:.1f}bps/day  t={sp_s[\"t_stat\"]:.2f}')"
        ),
        md(
            f"The timer underperforms: Sharpe {R['sharpe_active']:.2f} vs {R['sharpe_passive']:.2f} for "
            "buy-and-hold. Going flat when the VRP is below its rolling median costs more in missed "
            "equity exposure than it saves in avoided drawdowns. The VRP filter simply does not "
            "improve on the baseline.\n\n"
            "> This doesn't mean VRP is useless for timing -- other formulations (e.g. Bollerslev "
            "et al. 2009 use the VRP^2 and a model-based RV forecast, not the trailing 21-day RV) "
            "show stronger predictability at 3-month horizons. Our simpler version is a lower bound."
        ),
        md("### The skew: what short-vol actually looks like"),
        code(
            "if HAVE_REAL:\n"
            "    sv = st.short_vol_payoff(df, horizon=21)\n"
            "    sv_mean, sv_sk, sv_p05, sv_worst = sv['mean_ret_pct'], sv['skew'], sv['p05'], sv['worst']\n"
            "    log_r = df['SPY_ret'].rolling(21).sum().dropna().to_numpy()\n"
            "else:\n"
            f"    sv_mean, sv_sk = {R['sv_mean']}, {R['sv_skew']}\n"
            f"    sv_p05, sv_worst = {R['sv_p05']}, {R['sv_worst']}\n"
            "    log_r = None\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))\n"
            "ax = axes[0]\n"
            "if log_r is not None:\n"
            "    ax.hist(log_r * 100, bins=60, color=AMBER, alpha=0.75, edgecolor='none')\n"
            "ax.axvline(0, c='k', lw=1.5)\n"
            "ax.axvline(sv_p05, c=RED, lw=2, ls='--', label=f'5th pct: {sv_p05:.1f}%')\n"
            "ax.set_xlabel('21-day SPY log return (%)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Short-vol exposure: mean={sv_mean:.1f}% but left tail hurts')\n"
            "ax.legend()\n"
            "ax2 = axes[1]\n"
            "ax2.bar(['Mean (21d)', '5th pct', 'Worst period'],\n"
            "       [sv_mean, sv_p05, sv_worst],\n"
            "       color=[GREEN, AMBER, RED])\n"
            "ax2.axhline(0, c='k', lw=1)\n"
            "ax2.set_ylabel('Return (%)')\n"
            "ax2.set_title('The crash exposure is the price of the premium')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Skew = {sv_sk:.2f}: March 2020 worst period = {sv_worst:.1f}%')"
        ),
        md(
            f"The distribution is negatively skewed (skew = {R['sv_skew']:.2f}): the "
            f"**{R['sv_pct_neg']:.0f}%** of negative 21-day periods include the 2008 GFC, the "
            f"2010 Flash Crash, the 2020 COVID crash (worst: {R['sv_worst']:.0f}%). The 5th "
            f"percentile loss is {R['sv_p05']:.1f}%. This is the archetypal 'picking up nickels "
            "in front of a steamroller' trade: the mean is positive but you absorb the full "
            "equity crash risk without the long equity upside."
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal -- MIXED.** The VRP premium is statistically overwhelming (t = "
            f"{R['t_vrp']:.0f}) and well-understood. Q5 (high-VRP) days have real forward "
            f"returns (t = {R['q5_t']:.2f}). But the Q5-Q1 spread (t = {R['t_spread_h1']:.2f}) "
            f"and the timing overlay (Sharpe {R['sharpe_active']:.2f} vs "
            f"{R['sharpe_passive']:.2f}) do not clear the bar. REAL premium, WEAK timing.\n"
            f"- **Tradability -- FRAGILE.** The timer underperforms buy-and-hold. Harvesting "
            f"the VRP directly requires short-options infrastructure; the payoff is negatively "
            f"skewed (skew = {R['sv_skew']:.2f}, worst 21-day period = {R['sv_worst']:.0f}%) "
            f"and tail-risk management is non-trivial.\n"
            f"- **VRP premium? -- REAL.** 33 years, t = {R['t_vrp']:.0f}, {R['pct_pos_vrp']:.0f}% "
            "positive. Options sellers are structurally compensated for short-vol risk. This "
            "is not in dispute."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT -----------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Collecting the VRP requires one of three paths -- each with a structural cost:\n\n"
            "1. **Short VIX ETPs (e.g. SVXY, XIV before the 2018 vol spike).** Cheapest to "
            "implement but exposes you to roll cost, daily rebalancing, and in extreme events "
            "near-total loss (XIV fell ~95% on 5 Feb 2018).\n"
            "2. **Short variance swaps.** The clean institutional trade. Requires an ISDA, "
            "prime brokerage, and collateral infrastructure beyond most retail traders.\n"
            "3. **Short straddles / strangles on SPX.** Most accessible but requires active "
            "delta-hedging to isolate the vol premium from directional equity risk.\n\n"
            "In all cases, the payoff is the same structurally: you collect a premium most of "
            "the time and absorb a potentially catastrophic loss during vol spikes."
        ),
        code(
            "# Illustrate the negative-skew / drawdown character\n"
            "if HAVE_REAL:\n"
            "    sv_all = st.short_vol_payoff(df, horizon=21)\n"
            "    skew_val = sv_all['skew']\n"
            "else:\n"
            f"    skew_val = {R['sv_skew']}\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.0))\n"
            "payoff_x = [-30, -20, -10, 0, 5, 8]\n"
            "payoff_y = [0, 0, 0.1, 0.3, 0.7, 1.0]  # stylised CDF\n"
            "ax.plot(payoff_x, payoff_y, c=RED, lw=2.5, label='Short-vol payoff (stylised CDF)')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.fill_betweenx([0, 0.36], -30, 0, color=RED, alpha=0.12, label='36% negative periods')\n"
            "ax.set_xlabel('Period return (%)')\n"
            "ax.set_ylabel('Cumulative probability')\n"
            "ax.set_title(f'Short-vol CDF: fat left tail, skew={skew_val:.2f}')\n"
            "ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('The left tail is the cost of the premium. Crash risk is the product, not a bug.')"
        ),

        # ---- BEAT 7 -- GOING FURTHER ----------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The related CBOE term-structure signal** (VIX vs VIX3M slope -- contango vs "
            "backwardation) is tested in **[Study 111 -- VIX-Term-Structure](../../111-vix-term-structure/)**.\n"
            "- **The indirect short-vol ETP trade** (shorting VIXY, carrying roll) is in "
            "**[Study 63 -- Free-Fall](../../63-free-fall/)**.\n"
            "- **Tail protection strategies** that let you collect the VRP while capping the "
            "crash exposure are studied in **[Study 86 -- Tail-Radar](../../86-tail-radar/)**.\n"
            "- **The academic literature** (Bollerslev et al. 2009, Carr & Wu 2009) shows "
            "stronger VRP predictability using a model-based RV forecast instead of the simple "
            "trailing 21-day RV we use here. That's a worthwhile extension -- model-free RV "
            "from high-frequency data or option-derived RV measures may sharpen the signal.\n\n"
            "*The premium is real. The question is always: can you harvest it at a risk-adjusted "
            "cost that beats what you could earn by just staying long equities?*"
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
            "# Vol-Risk-Premium -- a quantitative teardown\n"
            "### 33 years of VIX vs SPY realised vol · HAC inference · quintile sorts · "
            "timing overlay · short-vol distribution\n\n"
            "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![VRP_premium%3F: Real](https://img.shields.io/badge/VRP_premium%3F-Real-2ea44f?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim now carrying its standard error.* We test the "
            "variance risk premium (VRP = VIX - trailing-21d-RV) across three claims: (1) the "
            "premium exists (IV > RV), (2) a high VRP predicts forward SPY returns, and (3) a "
            "timing rule exploiting (2) beats buy-and-hold.\n\n"
            "> **Not investment advice.** Real data: Yahoo Finance `^VIX` + `SPY` daily, "
            f"as-of {R['date_end']}; offline core runs on a deterministic synthetic tape. "
            "Methods in [`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **`In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT + LOAD_REAL + "\nfrom quantlab import analytics, stats\n"),

        # ---- BEAT 0 -- VERDICT (quant table) ---------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | VRP premium t = **{R['t_vrp']:.1f}** (REAL); "
            f"Q5 forward return t = **{R['q5_t']:.2f}** (REAL); Q5-Q1 spread "
            f"t = **{R['t_spread_h1']:.2f}** (below bar); timing spread t = "
            f"**{R['spread_t']:.2f}** (below bar). |\n"
            f"| **Tradability** | `FRAGILE` | Timer Sharpe {R['sharpe_active']:.2f} vs "
            f"B&H Sharpe {R['sharpe_passive']:.2f}; short-vol skew = {R['sv_skew']:.2f}, "
            f"worst 21d = {R['sv_worst']:.0f}%. |\n"
            f"| **VRP premium?** | `REAL` | Mean VRP = +{R['mean_vrp']:.2f} pp, "
            f"pct_pos = {R['pct_pos_vrp']:.0f}%, HAC t = {R['t_vrp']:.1f} on n = "
            f"{R['n']:,} days. |\n\n"
            "> **In plain words:** The market over-prices fear, consistently. The vol seller "
            "is compensated (REAL premium). Whether that premium signals *good equity days ahead* "
            "is mixed: the top quintile has real returns but the spread is noisy and the timer "
            "loses to buy-and-hold (MIXED, FRAGILE tradability)."
        ),

        # ---- BEAT 1 -- CLAIM (steelmanned) -----------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            r"Let $\sigma^{IV}_t$ be the VIX (annualised pct) and $\sigma^{RV}_t$ the trailing"
            r"21-day realised vol in the same units. The VRP at day $t$ is:"
            "\n\n$$\\text{VRP}_t = \\sigma^{IV}_t - \\sigma^{RV}_t$$\n\n"
            "Three sub-claims:\n\n"
            r"- **H1 (premium exists).** $\mathbb{E}[\text{VRP}_t] > 0$ with $|t| \geq 2$."
            "\n- **H2 (equity predictability).** The conditional expectation "
            r"$\mathbb{E}[r_{t+h} \mid \text{VRP-quintile}_t = 5]$ exceeds the Q1 equivalent"
            " significantly (HAC t on Q5-Q1 spread ≥ 2).\n"
            "- **H3 (timing).** A binary timer that is long SPY when "
            r"$\text{VRP}_t > \text{median}(\text{VRP}_{t-252..t})$"
            " earns a higher Sharpe than unconditional buy-and-hold.\n\n"
            "We confirm H1 strongly; reject H2 at the Q5-Q1 spread level; reject H3. "
            "The theoretical underpinning (recursive preferences, jump risk, "
            "Drechsler & Yaron 2011) does not guarantee H2 in the simple 21-day trailing "
            "RV formulation -- model-based RV or integrated variance (VIX^2 minus option-"
            "implied variance) may restore it."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 · So what? -- what rides on each answer\n\n"
            "H1 confirmed means the derivatives market structurally compensates short-vol "
            "exposure. H2 failing (at the spread level) means you cannot improve a naive "
            "equity long with a simple VRP timing filter. H3 failing means the VRP "
            "timer is worse than buy-and-hold. The practical implication: to harvest H1, "
            "you need options infrastructure and a crash-risk budget; you cannot do it "
            "via a simple long/flat equity rule."
        ),

        # ---- BEAT 3 -- PROTOCOL -----------------------------------------------
        md(
            "## 3 · How we'd know -- the protocol\n\n"
            "- **VRP construction.** $\\text{VIX}_t$ from Yahoo `^VIX` (CBOE index, "
            "ann. pct); $\\text{RV}_{21,t}$ = trailing-21-day `std(log ret, ddof=1) * "
            "sqrt(252) * 100`.\n"
            "- **Quintile rank.** Rolling 252-day percentile of VRP, out-of-sample "
            "(strictly: rank at *t* uses only history through *t*). Quintile 1 = "
            "lowest VRP, 5 = highest.\n"
            "- **Forward return.** Log return over *h* trading days starting the *next* "
            "close (enter at close *t*, exit at close *t+h*).\n"
            "- **Inference.** Newey-West HAC *t*-stat on each quintile's mean and on the "
            "Q5-Q1 sign-adjusted spread.\n"
            "- **Timer.** Signal = 1 if VRP > rolling 252-day median (lagged by 1 day); "
            "flat otherwise. Baseline: unconditional long SPY. Comparison: HAC *t* on the "
            "active-minus-passive daily spread.\n"
            "- **Positive control.** Synthetic tape with tunable VRP predictive power "
            "to verify the engine can detect a signal when one is planted.\n\n"
            "Data: March 1993 -- June 2026 (n = 8,379 daily observations)."
        ),

        # ---- BEAT 4 -- TEARDOWN -----------------------------------------------
        md("## 4 · The teardown"),
        md("### 4a · H1 — The premium: VIX exceeds RV with t = 23"),
        code(
            "if HAVE_REAL:\n"
            "    ps = st.vrp_premium_stats(df)\n"
            "else:\n"
            f"    ps = dict(n={R['n']}, mean_vrp={R['mean_vrp']}, std_vrp={R['std_vrp']},\n"
            f"              pct_positive={R['pct_pos_vrp']}, tstat={R['t_vrp']},\n"
            f"              mean_vix={R['mean_vix']}, mean_rv={R['mean_rv']})\n"
            "print(f\"n = {ps['n']:,}\")\n"
            "print(f\"mean VIX: {ps['mean_vix']:.2f} pp  mean RV21: {ps['mean_rv']:.2f} pp\")\n"
            "print(f\"mean VRP = VIX - RV: {ps['mean_vrp']:.3f} pp\")\n"
            "print(f\"pct days VRP > 0: {ps['pct_positive']:.1f}%\")\n"
            "print(f\"HAC t-stat (H0: mean=0): {ps['tstat']:.2f}  => H1 strongly confirmed\")\n\n"
            "if HAVE_REAL:\n"
            "    ci = stats.sharpe_ci_bootstrap(df['VRP'].dropna().rename('VRP'),\n"
            "                                   periods_per_year=252, seed=130)\n"
            "    print(f\"Bootstrap 95% CI on VRP Sharpe: [{ci['ci_low']:.2f}, {ci['ci_high']:.2f}]\")"
        ),
        md(
            f"> **In plain words:** The null hypothesis H0: mean VRP = 0 is rejected at t = "
            f"{R['t_vrp']:.0f} — one of the highest t-statistics in empirical finance. The "
            "premium is structural, not a sample artifact."
        ),
        md("### 4b · H2 — Equity predictability: real at the quintile level, noisy at the spread"),
        code(
            "if HAVE_REAL:\n"
            "    qt = st.quintile_table(df, horizons=[1, 5, 21])\n"
            "else:\n"
            "    qt = None\n\n"
            "horizons = [1, 5, 21]\n"
            "fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))\n"
            "for ax, h in zip(axes, horizons):\n"
            "    if HAVE_REAL:\n"
            "        row_h = qt[qt['horizon']==h].sort_values('quintile')\n"
            "        means = row_h['mean_bps'].tolist()\n"
            "        tstats = row_h['t_stat'].tolist()\n"
            "    else:\n"
            f"        means = [{R['q1_mean']}, None, None, None, {R['q5_mean']}]\n"
            f"        tstats = [{R['q1_t']}, None, None, None, {R['q5_t']}]\n"
            "    colors = [GREEN if abs(t or 0) >= 2 else RED for t in tstats]\n"
            "    ax.bar([f'Q{i}' for i in range(1,6)], means, color=colors)\n"
            "    ax.axhline(0, c='k', lw=1)\n"
            "    ax.set_title(f'h={h}d')\n"
            "    ax.set_xlabel('VRP quintile')\n"
            "    if ax == axes[0]: ax.set_ylabel('forward return (bps)')\n"
            "    for i, (m, t) in enumerate(zip(means, tstats)):\n"
            "        if t is not None and abs(t) >= 2:\n"
            "            ax.annotate(f't={t:.1f}', (i, m or 0), ha='center', va='bottom', fontsize=8)\n"
            "plt.suptitle('Green bar = t>=2 (real signal); Q5 real at all horizons')\n"
            "plt.tight_layout(); plt.show()\n\n"
            "print('Q5-Q1 spread tests:')\n"
            "for h in horizons:\n"
            "    if HAVE_REAL:\n"
            "        tb = st.top_minus_bottom(df, horizon=h)\n"
            "        print(f'  h={h:2d}: spread={tb[\"spread_bps\"]:+.1f}bps t={tb[\"t_spread\"]:+.2f}')\n"
            f"    else:\n"
            f"        sp = {R['spread_h1']} if h==1 else {R['spread_h5']} if h==5 else {R['spread_h21']}\n"
            f"        ts = {R['t_spread_h1']} if h==1 else {R['t_spread_h5']} if h==5 else {R['t_spread_h21']}\n"
            "        print(f'  h={h:2d}: spread={sp:+.1f}bps t={ts:+.2f}')"
        ),
        md(
            f"> **In plain words:** Q5 (high-VRP days) shows real returns at all horizons — "
            f"t = {R['q5_t']:.2f} at h=1, t = {R['q5_t_h21']:.2f} at h=21. But Q1 is also "
            f"positive (t = {R['q1_t_h21']:.2f} at h=21). The Q5-Q1 spread does not clear the "
            f"|t| ≥ 2 bar at any horizon. The signal is one-sided (Q5 is real), the spread is "
            "weak: MIXED."
        ),
        md("### 4c · H3 — Timing overlay: the VRP timer underperforms buy-and-hold"),
        code(
            "if HAVE_REAL:\n"
            "    to = st.timing_overlay(df, cost_bps=0)\n"
            "    r_p = to['r_passive'].dropna()\n"
            "    r_a = to['r_active'].dropna()\n"
            "    r_sp = to['r_spread'].dropna()\n"
            "    p_s = st.summarize(r_p, 'passive')\n"
            "    a_s = st.summarize(r_a, 'active')\n"
            "    sp_s = st.summarize(r_sp, 'spread')\n"
            "else:\n"
            f"    p_s = dict(sharpe_ann={R['sharpe_passive']}, mean_bps=4.1, t_stat=3.64)\n"
            f"    a_s = dict(sharpe_ann={R['sharpe_active']}, mean_bps=2.6, t_stat=3.31)\n"
            f"    sp_s = dict(mean_bps={R['spread_mean']}, t_stat={R['spread_t']})\n"
            "\n"
            "print('Timing overlay (long SPY when VRP > rolling median):')\n"
            "print(f\"  passive (B&H): Sharpe={p_s['sharpe_ann']:.2f}  mean={p_s['mean_bps']:.1f}bps/day  t={p_s['t_stat']:.2f}\")\n"
            "print(f\"  active (0 cost): Sharpe={a_s['sharpe_ann']:.2f}  mean={a_s['mean_bps']:.1f}bps/day  t={a_s['t_stat']:.2f}\")\n"
            "print(f\"  spread (active-passive): {sp_s['mean_bps']:.1f}bps/day  t={sp_s['t_stat']:.2f}\")\n"
            "print('=> Timer does not beat B&H; H3 rejected.')\n\n"
            "if HAVE_REAL:\n"
            "    cum_p = (1 + r_p).cumprod()\n"
            "    cum_a = (1 + r_a).cumprod()\n"
            "    fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "    ax.plot(cum_p.index, cum_p / cum_p.iloc[0], c=GREY, lw=1.5, label='Buy-and-hold')\n"
            "    ax.plot(cum_a.index, cum_a / cum_a.iloc[0], c=AMBER, lw=1.5, label='VRP timer')\n"
            "    ax.set_yscale('log')\n"
            "    ax.set_ylabel('NAV (log scale)')\n"
            "    ax.set_title('Timing overlay vs buy-and-hold (both nominal, 0-cost)')\n"
            "    ax.legend()\n"
            "    plt.tight_layout(); plt.show()"
        ),
        md(
            "> **In plain words:** The VRP filter removes the investor from the market during "
            "some of its best periods (low-VRP environments can precede sustained rallies). "
            "The timer Sharpe is lower than buy-and-hold at zero cost -- it would only worsen "
            "with transaction costs from the 981 regime switches over 33 years."
        ),
        md("### 4d · Short-vol payoff distribution and crash exposure"),
        code(
            "if HAVE_REAL:\n"
            "    sv = st.short_vol_payoff(df, horizon=21)\n"
            "    sv_mean, sv_sk, sv_kurt = sv['mean_ret_pct'], sv['skew'], sv['kurt']\n"
            "    sv_p05, sv_wst, sv_neg = sv['p05'], sv['worst'], sv['pct_negative']\n"
            "else:\n"
            f"    sv_mean, sv_sk, sv_kurt = {R['sv_mean']}, {R['sv_skew']}, {R['sv_kurt']}\n"
            f"    sv_p05, sv_wst, sv_neg = {R['sv_p05']}, {R['sv_worst']}, {R['sv_pct_neg']}\n"
            "print('Short-vol payoff (21d non-overlapping periods):')\n"
            "print(f'  mean={sv_mean:.2f}%  skew={sv_sk:.2f}  kurt={sv_kurt:.2f}')\n"
            "print(f'  pct_negative={sv_neg:.1f}%  p05={sv_p05:.1f}%  worst={sv_wst:.1f}%')\n\n"
            "# Illustration: short-vol Sharpe vs equity Sharpe\n"
            "if HAVE_REAL:\n"
            "    spy_ann = st.summarize(df['SPY_ret'].dropna(), 'SPY')['sharpe_ann']\n"
            "else:\n"
            f"    spy_ann = {R['sharpe_passive']}\n"
            "sv_sharpe = sv_mean / 4.429 * (12 ** 0.5)  # annualise (12 non-overlapping 21d periods)\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.0))\n"
            "ax.bar(['SPY B&H Sharpe', 'Short-vol Sharpe (approx)'], [spy_ann, sv_sharpe],\n"
            "       color=[GREY, RED])\n"
            "ax.set_ylabel('Annualised Sharpe')\n"
            "ax.set_title(f'Short-vol Sharpe (skew={sv_sk:.2f}) vs buying equity')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Sharpe premium over buy-and-hold: {sv_sharpe - spy_ann:.2f}')"
        ),
        md(
            f"> **In plain words:** The short-vol payoff has a similar Sharpe to equity but "
            f"with a much worse left tail (skew = {R['sv_skew']:.2f} vs equity's roughly "
            "neutral skew). You are being paid for crash risk that doesn't diversify away -- "
            "the worst short-vol periods are exactly the worst equity periods."
        ),

        # ---- BEAT 5 -- VERDICT -------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `MIXED`** — H1 (premium) confirmed at t = {R['t_vrp']:.1f}; "
            f"H2 (predictability) partially confirmed at Q5 level (t = {R['q5_t']:.2f}) "
            f"but Q5-Q1 spread t = {R['t_spread_h1']:.2f} does not clear the bar; H3 "
            "rejected (timer underperforms B&H).\n"
            f"- **Tradability `FRAGILE`** — Timer loses to B&H "
            f"(Sharpe {R['sharpe_active']:.2f} vs {R['sharpe_passive']:.2f}). Harvesting "
            f"the VRP directly requires options infrastructure with skew = {R['sv_skew']:.2f} "
            f"and worst-period = {R['sv_worst']:.0f}%.\n"
            f"- **VRP premium `REAL`** — t = {R['t_vrp']:.1f}, mean = {R['mean_vrp']:.2f} pp, "
            f"positive on {R['pct_pos_vrp']:.0f}% of {R['n']:,} days."
        ),

        # ---- BEAT 6 -- TRADABILITY ----------------------------------------------
        md(
            "## 6 · Could you trade it? -- implementation costs and alternatives\n\n"
            "The VRP is a structural premium but collecting it requires a choice of vehicle, "
            "each with its own cost:\n\n"
            "| Vehicle | Transaction cost | Roll cost | Crash exposure | Practical threshold |\n"
            "|---|---|---|---|---|\n"
            "| VIX ETP short (SVXY) | low | high (negative roll) | extreme (XIV -95% in 2018) | retail |\n"
            "| Short variance swap | zero (OTC) | none | full | prime-brokerage only |\n"
            "| Short SPX straddle | moderate | none | full + gamma | active delta-hedging |\n"
            "| Long equity (proxy) | low | none | equity only | wrong risk budget |\n\n"
            "None of these is a simple 'click buy' trade. The literature (Broadie et al. 2009) "
            "shows that a retail investor who delta-hedges short straddles monthly earns the "
            "VRP but needs substantial collateral and active management."
        ),
        code(
            "# Positive control: planted VRP signal on synthetic tape\n"
            "print('Synthetic positive control: does the engine find a planted VRP signal?')\n"
            "print(f'{\"vrp_signal\":>12} | {\"Q5-Q1 spread\":>12} {\"t_spread\":>9}')\n"
            "print('-' * 40)\n"
            "for sig in [0.0, 0.005, 0.01, 0.02]:\n"
            "    df_s, _ = data.synthetic_daily(n_days=3000, vrp_signal=sig, seed=130)\n"
            "    tb_s = st.top_minus_bottom(df_s, horizon=21)\n"
            "    print(f'{sig:>12.3f} | {tb_s[\"spread_bps\"]:>+12.1f} {tb_s[\"t_spread\"]:>+9.2f}')"
        ),
        md(
            "> The engine correctly recovers the planted signal (spread grows monotonically "
            "with `vrp_signal`, t grows from ~0 to >>2). The real-tape mixed verdict is a "
            "statement about the market, not the method."
        ),

        # ---- BEAT 7 -- GOING FURTHER ---------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Better RV measure.** The trailing 21-day daily-return std is noisy. Using "
            "the model-free integrated variance (5-min realised variance, or CBOE's RVOL) "
            "should sharpen the VRP signal and restore the Bollerslev et al. predictability.\n"
            "- **VRP^2 as the predictor.** Bollerslev et al. (2009) show the *squared* VRP "
            "predicts at the 3-month horizon better than the level. This is a direct fork.\n"
            "- **Conditional on regime.** The VRP may predict better in specific regimes "
            "(post-crash recovery, rising-rate environments). A conditional sort or a "
            "regime-switching model could sharpen the result.\n"
            "- **Adjacent studies.** "
            "[Study 111 (VIX-Term-Structure)](../../111-vix-term-structure/) -- "
            "the VIX/VIX3M slope timing signal; "
            "[Study 63 (Free-Fall)](../../63-free-fall/) -- the VIXY roll-return approach; "
            "[Study 86 (Tail-Radar)](../../86-tail-radar/) -- tail risk management.\n\n"
            "*The premium is real. The signal is mixed. The trade is structurally fragile. "
            "That's the honest answer.*"
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
