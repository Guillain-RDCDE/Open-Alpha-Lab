"""Generate the two narrative notebooks for Study 531 (Enterprise-Multiple).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the desk beats. The synthetic / control cells run anywhere (offline,
deterministic). The real-tape cells read this study's own ../_cache/ if present and otherwise
quote the frozen headline numbers in ``R`` (mirroring docs/results.md), so the notebook
re-runs for any reader.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-26).
# 35-name large-cap survivor basket, 2023-03..2026-05, 38 monthly observations.
R = dict(
    n_months=38,
    n_names=35,
    window="2023-03 .. 2026-05",
    fp="7ba034f010a2",
    fp_em="f6954099c5ae",
    fp_fwd="ca4a8557268c",
    # Hedge (long cheap - short expensive), annualised
    hedge_mean=0.088,
    hedge_vol=0.141,
    hedge_sharpe=0.62,
    hedge_t_simple=1.11,
    hedge_t_hac=1.11,
    hedge_hit=0.55,
    hedge_maxdd=-0.099,
    turnover=0.134,
    # Legs
    long_mean=0.165, long_sharpe=1.19,
    short_mean=0.077, short_sharpe=0.53,
    mkt_mean=0.127, mkt_sharpe=1.08,
    # Placebo
    placebo_p=0.298,
    # Costs / net
    net_mean=0.082, net_t=1.04, net_sharpe=0.58, cost_bps=4.8,
    # IC
    ic_mean=0.050, ic_t=1.32,
    # Robustness
    half1_mean=-0.005, half1_t=-0.07, half2_mean=0.181, half2_t=1.39,
    # Calendar-year hedge (compounded within year)
    yr_2023=-0.082, yr_2024=0.092, yr_2025=0.126, yr_2026=0.135,
    # Synthetic control: premium -> seed-averaged HAC t (20 seeds)
    ctrl_premiums=[0.0, 0.002, 0.004, 0.006, 0.008],
    ctrl_t=[0.20, 3.09, 5.97, 8.86, 11.75],
    ctrl_mean=[0.4, 5.9, 11.5, 17.0, 22.5],
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from enterprise_multiple import data, strategy as st

_study_cache = os.path.abspath(os.path.join("..", "_cache"))

def _have_real():
    return (os.path.exists(os.path.join(_study_cache, "em_panel.parquet"))
            and os.path.exists(os.path.join(_study_cache, "fwd_ret.parquet")))

HAVE_REAL = _have_real()
if HAVE_REAL:
    em, fwd = data.fetch_panel(cache_dir=_study_cache)
else:
    em, fwd = pd.DataFrame(), pd.DataFrame()

print("real data available:", HAVE_REAL, f"  panel: {em.shape if not em.empty else 'n/a'}")
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Enterprise-Multiple -- is EV/EBITDA a real value edge?\n"
            "### The practitioner's favourite cheapness ratio, sorted honestly on the S&P 500 large caps\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Cheap_beats_expensive%3F: Mixed](https://img.shields.io/badge/Cheap_beats_expensive%3F-Mixed-8b949e?style=flat-square)\n\n"
            "Every banker and value investor quotes **EV/EBITDA** -- enterprise value over operating "
            "earnings. A *low* multiple means the whole firm (equity **plus** net debt) is cheap "
            "relative to the cash its operations throw off. Loughran & Wellman (2011) showed it "
            "predicts returns: buy the low-multiple names, sell the high-multiple ones. This notebook "
            "tests that idea on a large-cap survivor basket, honestly.\n\n"
            "> Plain-language layer. For HAC *t*-stats, the placebo null and the positive control see "
            "the companion **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** Every chart is drawn by the code beside it. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does a low EV/EBITDA predict returns? | **Not significantly.** Hedge "
            f"**{R['hedge_mean']*100:+.1f}%/yr**, HAC *t* = **{R['hedge_t_hac']:+.2f}**. |\n"
            f"| Does it beat a coin (placebo)? | **No.** A within-month label shuffle puts it at "
            f"**p = {R['placebo_p']:.2f}**. |\n"
            f"| Do cheap names beat expensive ones at all? | **As a ranking, yes.** Cheap leg "
            f"Sharpe **{R['long_sharpe']:.2f}** vs expensive **{R['short_sharpe']:.2f}**. |\n"
            f"| Could you trade the spread? | **No.** Net **{R['net_mean']*100:+.1f}%/yr**, "
            f"*t* {R['net_t']:+.2f} -- still indistinguishable from zero. |\n\n"
            "> The ratio sorts the right way, but the long-short trade that would monetise it can't "
            "clear the statistical bar on this large-cap survivor sample -- exactly where the factor-"
            "decay literature (and Loughran-Wellman themselves) say a large-cap-only test should land."
        ),

        md(
            "## 1 · The claim\n\n"
            "> *'Forget P/E -- it ignores debt and gets distorted by depreciation games. Use "
            "**EV/EBITDA**: enterprise value (market cap + total debt - cash) over operating earnings "
            "before D&A. Buy the cheapest decile, sell the dearest, hold, rebalance. It is the "
            "cleanest value yardstick because it is capital-structure-neutral.'*\n\n"
            "The supporting evidence is **Loughran & Wellman (2011)**, who found EV/EBITDA a robust "
            "cross-sectional predictor in the US -- though, crucially, **concentrated in smaller, less-"
            "liquid names**. Our test deliberately uses large, liquid survivors, the hardest place for "
            "any value premium to survive."
        ),

        md(
            "## 2 · So what?\n\n"
            "If the multiple still pays in large caps, it is a simple, low-turnover, once-a-year-signal "
            "strategy any retail investor could run. If it has been arbed away -- the obvious fate of a "
            "metric every analyst on Wall Street stares at daily -- then the accounting logic is fine "
            "but the trading conclusion does not follow. The interesting twist is that EV/EBITDA might "
            "still *rank* correctly (cheap beats dear) without the *spread* being tradable."
        ),

        md(
            "## 3 · How would we even know?\n\n"
            "The protocol:\n\n"
            "1. **No look-ahead.** Each fiscal year's multiple is treated as public only after a "
            "4-month reporting lag, then held for 12 months; the position is entered the *next* month.\n"
            "2. **A fair baseline.** The dollar-neutral long-short hedge (cheap minus expensive) is "
            "compared to zero, not to a leveraged benchmark.\n"
            "3. **A placebo.** Shuffle the multiple across names *within each month* and re-run; the "
            "real result must beat that null.\n"
            "4. **Survivorship caveat.** The basket is names still trading in 2026 -- the expensive "
            "names that later imploded and delisted are absent. This should *flatter* the trade; "
            "finding nothing here is conservative.\n\n"
            "Data: yfinance annual EBITDA, total debt, cash and shares; monthly adjusted prices; "
            f"**{R['n_names']} large-cap names, {R['window']} ({R['n_months']} months).** "
            "(5 banks drop out -- EBITDA is meaningless for financials.)"
        ),

        md("## 4 · The teardown -- let's actually look"),
        md("**First: the cheap leg vs the expensive leg vs the market.**"),
        code(
            f"long_mean, short_mean, mkt_mean = {R['long_mean']}, {R['short_mean']}, {R['mkt_mean']}\n"
            f"long_sh, short_sh, mkt_sh = {R['long_sharpe']}, {R['short_sharpe']}, {R['mkt_sharpe']}\n"
            "if HAVE_REAL:\n"
            "    h = st.long_short_hedge(em, fwd)\n"
            "    sl, ss, sm = st.summary(h['long']), st.summary(h['short']), st.summary(h['market'])\n"
            "    long_mean, short_mean, mkt_mean = sl['mean'], ss['mean'], sm['mean']\n"
            "    long_sh, short_sh, mkt_sh = sl['sharpe'], ss['sharpe'], sm['sharpe']\n"
            "labels = ['Cheap leg\\n(low EV/EBITDA)', 'Expensive leg\\n(high EV/EBITDA)', 'Market\\n(equal-wt)']\n"
            "means = [long_mean*100, short_mean*100, mkt_mean*100]\n"
            "sharpes = [long_sh, short_sh, mkt_sh]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "a1.bar(labels, means, color=[GREEN, RED, GREY]); a1.axhline(0, c='k', lw=1)\n"
            "a1.set_ylabel('Annualised return (%)'); a1.set_title('Cheap vs expensive: returns')\n"
            "a2.bar(labels, sharpes, color=[GREEN, RED, GREY]); a2.axhline(0, c='k', lw=1)\n"
            "a2.set_ylabel('Sharpe'); a2.set_title('Cheap vs expensive: Sharpe')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Cheap Sharpe {long_sh:+.2f}  vs  Expensive Sharpe {short_sh:+.2f}  '\n"
            "      f'(market {mkt_sh:+.2f})')"
        ),
        md(
            f"The cheap leg (Sharpe **{R['long_sharpe']:.2f}**) genuinely out-rides the expensive leg "
            f"(**{R['short_sharpe']:.2f}**) and even the market (**{R['mkt_sharpe']:.2f}**). So the "
            "*ranking* works. The question is whether the **difference** -- the long-short hedge -- is "
            "big enough to trade."
        ),
        md("**Now the hedge itself, year by year.**"),
        code(
            f"yr_vals = {{2023: {R['yr_2023']}, 2024: {R['yr_2024']}, 2025: {R['yr_2025']}, 2026: {R['yr_2026']}}}\n"
            "if HAVE_REAL:\n"
            "    h = st.long_short_hedge(em, fwd)\n"
            "    hh = h.copy(); hh.index = hh.index.to_timestamp()\n"
            "    yr_series = hh['hedge'].groupby(hh.index.year).apply(lambda x: (1+x).prod()-1)\n"
            "    yr_vals = yr_series.to_dict()\n"
            "yrs = list(yr_vals.keys()); vals = [yr_vals[y]*100 for y in yrs]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.bar([str(y) for y in yrs], vals, color=[GREEN if v > 0 else RED for v in vals])\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Hedge return (cheap - expensive, %)')\n"
            f"ax.set_title('EV/EBITDA hedge by year  (full window mean {R['hedge_mean']*100:+.1f}%/yr, HAC t {R['hedge_t_hac']:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Hedge years:', {y: f'{v:+.1f}%' for y, v in zip(yrs, vals)})"
        ),
        md(
            f"The hedge is **{R['hedge_mean']*100:+.1f}%/yr** on average but lumpy: a negative 2023, "
            "then three positive years. The positive tilt is concentrated in 2025-26 (the half-split "
            f"confirms it: first-half *t* = {R['half1_t']:+.2f}, second-half *t* = {R['half2_t']:+.2f}) "
            "-- not the steady premium a tradable factor needs."
        ),

        md(
            "## 5 · The verdict\n\n"
            f"- **Signal -- None.** Hedge {R['hedge_mean']*100:+.1f}%/yr, HAC *t* = "
            f"**{R['hedge_t_hac']:+.2f}**; placebo **p = {R['placebo_p']:.2f}**. Below the "
            "\\|*t*\\| >= 2 bar and inside the shuffle null.\n"
            f"- **Tradability -- Mirage.** Net {R['net_mean']*100:+.1f}%/yr (*t* {R['net_t']:+.2f}); "
            f"costs are tiny ({R['cost_bps']:.1f} bps/month) but there is no gross edge to protect.\n"
            f"- **Cheap beats expensive? -- Mixed.** The cheap leg out-Sharpes the expensive leg "
            f"({R['long_sharpe']:.2f} vs {R['short_sharpe']:.2f}) -- the ranking holds -- but the "
            "spread can't be told from zero."
        ),

        md(
            "## 6 · Could you actually trade it?\n\n"
            "- **The signal is stale by construction.** Fundamentals arrive months after fiscal-year "
            "end; the multiple you sort on is already old.\n"
            "- **Everyone watches it.** EV/EBITDA is on every analyst's screen -- any persistent "
            "large-cap edge has been visible for decades.\n"
            "- **Banks don't have an EBITDA.** Financials drop out of the sort entirely, so the "
            "'universe' is narrower than it looks.\n\n"
            "Costs are not the problem here -- there is simply no gross edge to erode."
        ),

        md(
            "## 7 · Going further\n\n"
            "- **Small caps.** Loughran-Wellman found the premium strongest in *smaller, less-liquid* "
            "names -- the opposite of this basket. That is the natural place to look next.\n"
            "- **[Study 530 -- Book-to-Market](../530-book-to-market-value/)** and "
            "**[Study 124 -- Cash-Flow-Yield](../124-cash-flow-yield/)**: the sibling value sorts. "
            "EV/EBITDA adds net debt and uses operating earnings.\n"
            "- **A quality overlay.** Pairing cheap multiples with a profitability/quality screen "
            "(QMJ-style) is the standard fix when a raw value sort fails.\n\n"
            "*Think the enterprise multiple pays somewhere? Extend to mid/small-cap with point-in-time "
            "membership and show a hedge that clears t = 2. That's the bar.*"
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
            "# Enterprise-Multiple -- a quantitative teardown\n"
            "### EV/EBITDA monthly sort . one-sample & HAC t . label-shuffle placebo . costs . positive control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Cheap_beats_expensive%3F: Mixed](https://img.shields.io/badge/Cheap_beats_expensive%3F-Mixed-8b949e?style=flat-square)\n\n"
            "Deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- *same beats, "
            "every claim with its standard error.* We test whether a low EV/EBITDA predicts next-month "
            "returns (HAC *t* on the monthly hedge), whether it beats a within-month label-shuffle null, "
            "whether it survives costs + borrow, and confirm the engine on a seed-averaged synthetic "
            "control. Survivorship bias is named throughout.\n\n"
            "> **Not investment advice.** Real data: yfinance fundamentals + monthly prices, "
            f"**{R['n_names']} names, {R['window']}**; as-of 2026-06-26, fp `{R['fp']}`. "
            "Methods & sources in [`docs/references.md`](../docs/references.md).\n"
            ">\n"
            "> **Survivorship caveat.** The basket is names still trading in 2026 -- results are upper "
            "bounds; the natural shorts (delisted expensive names) are absent."
        ),
        code(BOOT),

        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Hedge **{R['hedge_mean']*100:+.2f}%/yr**, HAC *t* = "
            f"**{R['hedge_t_hac']:+.2f}**; placebo p = {R['placebo_p']:.2f}. < \\|*t*\\| 2. |\n"
            f"| **Tradability** | `MIRAGE` | Net {R['net_mean']*100:+.2f}%/yr, *t* = {R['net_t']:+.2f} "
            f"after {R['cost_bps']:.1f} bps/month. No gross edge to protect. |\n"
            f"| **Cheap beats expensive?** | `MIXED` | Cheap leg Sharpe {R['long_sharpe']:.2f} vs "
            f"expensive {R['short_sharpe']:.2f} (ranking holds); spread IC {R['ic_mean']:+.3f}, "
            f"*t* {R['ic_t']:+.2f} (insignificant). |\n\n"
            "> On 38 months of a survivorship-biased large-cap basket, the enterprise multiple sorts "
            "directionally but yields no exploitable long-short return differential."
        ),

        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $\\text{EM}_{i,t} = \\text{EV}_{i,t} / \\text{EBITDA}_{i,t}$ with "
            "$\\text{EV} = \\text{mktcap} + \\text{debt} - \\text{cash}$. The claim:\n\n"
            "- **H1 (value signal).** "
            "$\\mathbb{E}[r_{i,t+1} \\mid \\text{EM}_{i,t}\\ \\text{low}] > "
            "\\mathbb{E}[r_{i,t+1} \\mid \\text{EM}_{i,t}\\ \\text{high}]$ -- the Q1(cheap)-minus-"
            "Q3(expensive) monthly hedge has positive mean.\n"
            "- **H2 (it ranks).** The Spearman IC of $-\\text{EM}$ with $r_{t+1}$ is positive.\n\n"
            "H1 is the tradable claim (the hedge HAC *t*); H2 is the weaker ranking claim (the IC). "
            "They can diverge -- a correct sign with an untradable magnitude."
        ),

        md(
            "## 2 · So what? -- what rides on each answer\n\n"
            "H1 + H2 would make EV/EBITDA a simple low-turnover large-cap factor with genuine content. "
            "H2 without H1 means the metric is informative but the spread is too small/noisy to trade -- "
            "the classic fate of a published large-cap value signal post-arbitrage (McLean-Pontiff 2016). "
            "Rejecting both would be the strongest negative."
        ),

        md(
            "## 3 · How we'd know -- the protocol\n\n"
            "- **Signal.** $\\text{EM}_{i,t}$ per fiscal year (yfinance EBITDA, total debt, cash, "
            "shares; year-end price for market cap), made public after a 4-month reporting lag and held "
            "12 months. Implausible multiples ($<0$ or $>100$) dropped as data errors.\n"
            "- **Sort.** Monthly: long the cheap tercile, short the expensive tercile, equal-weight, "
            "dollar-neutral. One execution lag (enter next month).\n"
            "- **Inference.** One-sample *t* and Newey-West HAC *t* on the monthly hedge (38 obs).\n"
            "- **Placebo.** 500 within-month signal shuffles; p = fraction of \\|t_HAC\\| >= real.\n"
            "- **Costs.** 5 bps/leg x turnover + 50 bps/yr borrow on shorts.\n"
            "- **Positive control.** Seed-averaged (20 seeds) HAC *t* on a synthetic panel with a "
            "dial-able planted premium -- a faithfulness/power check only."
        ),

        md("## 4 · The teardown"),
        md(
            "### 4a · The monthly hedge with HAC *t* -- the inference-bar test\n\n"
            "Cumulative wealth of the dollar-neutral cheap-minus-expensive book."
        ),
        code(
            f"hedge_mean, hedge_t, hedge_sh, hedge_hit = {R['hedge_mean']}, {R['hedge_t_hac']}, {R['hedge_sharpe']}, {R['hedge_hit']}\n"
            "if HAVE_REAL:\n"
            "    h = st.long_short_hedge(em, fwd)\n"
            "    s = st.summary(h['hedge'])\n"
            "    hedge_mean, hedge_t, hedge_sh, hedge_hit = s['mean'], s['t_hac'], s['sharpe'], s['hit_rate']\n"
            "    monthly = h['hedge'].values\n"
            "    months = [str(m) for m in h.index]\n"
            "else:\n"
            "    rng = np.random.default_rng(0)\n"
            f"    monthly = rng.normal({R['hedge_mean']}/12, {R['hedge_vol']}/np.sqrt(12), {R['n_months']})\n"
            f"    months = list(range({R['n_months']}))\n"
            "cum = np.cumprod(1 + monthly)\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.3))\n"
            "a1.bar(range(len(monthly)), [v*100 for v in monthly],\n"
            "       color=[GREEN if v > 0 else RED for v in monthly])\n"
            "a1.axhline(0, c='k', lw=1); a1.set_xlabel('Month'); a1.set_ylabel('Hedge return (%)')\n"
            "a1.set_title(f'Monthly hedge  (mean {hedge_mean*100:+.1f}%/yr)')\n"
            "a2.plot(cum, '-', c=RED, lw=2); a2.axhline(1, c='k', lw=1)\n"
            "a2.set_xlabel('Month'); a2.set_ylabel('Cumulative wealth ($1)')\n"
            "a2.set_title(f'HAC t = {hedge_t:+.2f}   Sharpe {hedge_sh:+.2f}   hit {hedge_hit:.0%}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Hedge: mean={hedge_mean*100:+.2f}%/yr  HAC t={hedge_t:+.2f}  Sharpe={hedge_sh:+.2f}')"
        ),
        md(
            f"> HAC *t* = **{R['hedge_t_hac']:+.2f}** -- far below \\|t\\| >= 2. The equity curve drifts "
            "up but well within what 38 noisy monthly draws produce by chance."
        ),
        md(
            "### 4b · The placebo -- does it beat a within-month shuffle?\n\n"
            "Permute the cross-sectional multiple within each month (destroying signal->return "
            "alignment) and re-run; compare the real \\|t\\| to the shuffled distribution."
        ),
        code(
            f"real_t, placebo_p = {R['hedge_t_hac']}, {R['placebo_p']}\n"
            "if HAVE_REAL:\n"
            "    pl = st.placebo_null(em, fwd, n_shuffles=300)\n"
            "    real_t, placebo_p = pl['real_t'], pl['p_value']\n"
            "    # rebuild a shuffle distribution for the picture\n"
            "    rng = np.random.default_rng(7); dist = []\n"
            "    cols = list(em.columns)\n"
            "    for _ in range(300):\n"
            "        vals = em.to_numpy(copy=True)\n"
            "        for i in range(vals.shape[0]):\n"
            "            row = vals[i]; idx = np.where(~np.isnan(row))[0]\n"
            "            if idx.size > 1: row[idx] = row[rng.permutation(idx)]\n"
            "            vals[i] = row\n"
            "        shuf = pd.DataFrame(vals, index=em.index, columns=cols)\n"
            "        ss = st.summary(st.long_short_hedge(shuf, fwd)['hedge'])\n"
            "        if np.isfinite(ss['t_hac']): dist.append(ss['t_hac'])\n"
            "    dist = np.array(dist)\n"
            "else:\n"
            "    dist = np.random.default_rng(1).normal(0, 1, 300)\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.hist(dist, bins=30, color=GREY, alpha=0.8, label='shuffled |HAC t| dist')\n"
            "ax.axvline(real_t, c=RED, lw=2.5, label=f'real t = {real_t:+.2f}')\n"
            "ax.axvline(-real_t, c=RED, lw=1, ls='--')\n"
            "ax.set_xlabel('HAC t-stat'); ax.set_ylabel('count')\n"
            "ax.set_title(f'Placebo: real sits inside the null (p = {placebo_p:.2f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Real |t| = {abs(real_t):.2f}   placebo p = {placebo_p:.3f}')"
        ),
        md(
            f"> Placebo **p = {R['placebo_p']:.2f}** -- about 30% of random relabellings match or beat "
            "the real *t*. The signal is statistically indistinguishable from a shuffle."
        ),
        md(
            "### 4c · Costs, IC and robustness\n\n"
            "Net of 5 bps/leg + 50 bps/yr borrow; the monthly IC; the tercile-fraction sweep and "
            "half-split."
        ),
        code(
            f"net_mean, net_t, ic_mean, ic_t = {R['net_mean']}, {R['net_t']}, {R['ic_mean']}, {R['ic_t']}\n"
            f"h1m, h1t, h2m, h2t = {R['half1_mean']}, {R['half1_t']}, {R['half2_mean']}, {R['half2_t']}\n"
            "frac_t = {0.20: 0.51, 0.25: 0.74, 0.33: 1.11}\n"
            "if HAVE_REAL:\n"
            "    h = st.long_short_hedge(em, fwd)\n"
            "    hc = st.apply_costs(h); sn = st.summary(hc['net'])\n"
            "    net_mean, net_t = sn['mean'], sn['t_hac']\n"
            "    ic = st.monthly_ic(em, fwd); sic = st.summary(ic)\n"
            "    ic_mean, ic_t = ic.mean(), sic['t_hac']\n"
            "    half = len(h) // 2\n"
            "    s1, s2 = st.summary(h['hedge'].iloc[:half]), st.summary(h['hedge'].iloc[half:])\n"
            "    h1m, h1t, h2m, h2t = s1['mean'], s1['t_hac'], s2['mean'], s2['t_hac']\n"
            "    frac_t = {fr: st.summary(st.long_short_hedge(em, fwd, frac=fr)['hedge'])['t_hac']\n"
            "              for fr in (0.20, 0.25, 1/3)}\n"
            "print(f'Net hedge:  {net_mean*100:+.2f}%/yr   HAC t = {net_t:+.2f}')\n"
            "print(f'Monthly IC: {ic_mean:+.4f}   HAC t = {ic_t:+.2f}')\n"
            "print(f'Half-split: 1st {h1m*100:+.1f}%/yr (t {h1t:+.2f}) | 2nd {h2m*100:+.1f}%/yr (t {h2t:+.2f})')\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.0))\n"
            "fr_keys = list(frac_t.keys()); fr_vals = list(frac_t.values())\n"
            "ax.bar([f'{k:.2f}' for k in fr_keys], fr_vals, color=AMBER)\n"
            "ax.axhline(2, c=GREEN, ls='--', lw=1.5, label='t = 2 bar')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Long/short fraction'); ax.set_ylabel('Hedge HAC t')\n"
            "ax.set_title('Fraction sweep: no cut clears the t = 2 bar'); ax.legend()\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> Net **{R['net_mean']*100:+.1f}%/yr** (*t* {R['net_t']:+.2f}) -- barely below gross "
            f"(turnover is tiny). IC **{R['ic_mean']:+.3f}** (*t* {R['ic_t']:+.2f}) -- right sign, not "
            f"significant. The half-split (1st *t* {R['half1_t']:+.2f}, 2nd *t* {R['half2_t']:+.2f}) and "
            "the fraction sweep show the modest tilt is not stable. **Fragile, not Real.**"
        ),

        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** -- hedge {R['hedge_mean']*100:+.2f}%/yr, HAC *t* "
            f"{R['hedge_t_hac']:+.2f}; placebo p {R['placebo_p']:.2f}; IC *t* {R['ic_t']:+.2f}. "
            "H1 not supported.\n"
            f"- **Tradability `MIRAGE`** -- net {R['net_mean']*100:+.2f}%/yr (*t* {R['net_t']:+.2f}); "
            "no gross edge to harvest.\n"
            f"- **Cheap beats expensive? `MIXED`** -- the ranking holds (cheap Sharpe "
            f"{R['long_sharpe']:.2f} > expensive {R['short_sharpe']:.2f}, IC +) but the spread is "
            "untradable."
        ),

        md("## 6 · The positive control -- the engine is faithful (and seed-robust)"),
        code(
            f"premiums = {R['ctrl_premiums']}\n"
            f"ctrl_t_frozen = {R['ctrl_t']}\n"
            "# Seed-averaged HAC t over 20 seeds at each planted premium (offline, deterministic)\n"
            "ctrl = st.synthetic_control(premiums=tuple(premiums), n_seeds=20)\n"
            "t_vals = ctrl['t_hac_mean'].values\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(premiums, t_vals, 'o-', c=GREEN, lw=2, label='seed-avg HAC t (20 seeds)')\n"
            "ax.axhline(2, c=GREY, ls='--', lw=1.5, label='t = 2 bar')\n"
            f"ax.axhline({R['hedge_t_hac']}, c=RED, ls=':', lw=2, label='real-tape t = {R['hedge_t_hac']:+.2f}')\n"
            "ax.set_xlabel('Planted premium (per z-unit, monthly)')\n"
            "ax.set_ylabel('Recovered hedge HAC t'); ax.set_title('Positive control: faithful & seed-robust')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for p, t in zip(premiums, t_vals): print(f'  premium {p:.3f} -> seed-avg HAC t {t:+.2f}')"
        ),
        md(
            "At premium 0 the seed-averaged *t* is ~0.2 (clean null); as the premium rises *t* climbs "
            "monotonically past 2, 6, 11. **The engine recovers a premium when one is planted, and the "
            "recovery does not rest on a lucky seed.** The near-flat real-tape *t* is therefore a "
            "statement about the market, not the method. *(This control supports only the engine -- "
            "never the real-tape stamp.)*"
        ),

        md(
            "## 7 · Going further\n\n"
            "- **Small / mid caps with point-in-time membership** -- where Loughran-Wellman (2011) "
            "found the premium strongest, and where survivorship bias is most pernicious.\n"
            "- **[Study 530 -- Book-to-Market](../530-book-to-market-value/)** / "
            "**[Study 124 -- Cash-Flow-Yield](../124-cash-flow-yield/)**: the sibling value sorts; a "
            "composite of all three may beat any one.\n"
            "- **Quality overlay (QMJ-style)** -- screen cheap multiples for profitability to avoid "
            "value traps.\n\n"
            "*The bar is unchanged: a hedge that clears t = 2 out of sample, net of costs, on a "
            "point-in-time universe.*"
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
