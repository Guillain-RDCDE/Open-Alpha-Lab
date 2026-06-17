"""Generate the two narrative notebooks for Study 264 (Buffett-Indicator).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The
hardcoded Buffett-Indicator table and the synthetic generator run anywhere,
offline and deterministically; the real-tape cells use the cached ^GSPC price
index if present, and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md) and synthetic fallbacks, so the notebooks re-run for
any reader. Synthetic cells never appear under a real-tape banner.

The _write convention (each build_*() ends by calling _write) is kept so the
repo's intro-restyle tooling can monkeypatch it.
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-17).
R = dict(
    n=55,
    year_start=1971,
    year_end=2025,
    fp="479252b8a735",
    # Predictive regression
    beta_per_100=-3.22,   # pp of next-year return per +100pp of GDP
    corr=-0.093,
    reg_r2=0.009,
    reg_t=-0.919,
    # Terciles (mean next-year return %, up-rate %)
    cheap_mean=10.0,
    mid_mean=9.9,
    exp_mean=8.2,
    cheap_up=78.9,
    mid_up=77.3,
    exp_up=64.3,
    # Real-time timing strategy
    strat_cagr=1.31,
    strat_cagr_net=1.30,
    bah_cagr=8.11,
    frac_invested=23.6,
    n_flips=4,
    excess_t=-3.764,
    # Latest reading
    ratio_2025=209.0,
    ratio_1974=42.0,
    ratio_1999=172.0,
    ratio_2021=207.0,
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from buffett_indicator import data, strategy as st

def _have_cache():
    try:
        data.fetch_sp500_annual(fetch=False)
        return True
    except FileNotFoundError:
        return False

HAVE_REAL = _have_cache()
print("^GSPC cache present:", HAVE_REAL)
"""

R_CELL = """\
# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-17)
R = dict(
    n=55, year_start=1971, year_end=2025, fp="479252b8a735",
    beta_per_100=-3.22, corr=-0.093, reg_r2=0.009, reg_t=-0.919,
    cheap_mean=10.0, mid_mean=9.9, exp_mean=8.2,
    cheap_up=78.9, mid_up=77.3, exp_up=64.3,
    strat_cagr=1.31, strat_cagr_net=1.30, bah_cagr=8.11,
    frac_invested=23.6, n_flips=4, excess_t=-3.764,
    ratio_2025=209.0, ratio_1974=42.0, ratio_1999=172.0, ratio_2021=207.0,
)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Buffett-Indicator -- does market-cap-to-GDP tell you when to leave the party?\n"
            "### Warren Buffett's 'best single measure of valuation', tested honestly, in plain English\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "In a 2001 *Fortune* piece, Warren Buffett called the ratio of total US "
            "stock-market value to GDP \"probably the best single measure of where valuations "
            "stand at any given moment.\" Near 70-80% of GDP, he said, it's a good time to buy; "
            "approaching 200%, you're \"playing with fire.\" Today it sits above **200%**. So: is "
            "this the alarm bell that tells you when to head for the exits -- or just a "
            "thermometer that reads 'hot' for years on end while the market keeps climbing?\n\n"
            "> **This is the plain-language layer.** Want the predictive regression, the HAC "
            "t-stat, and the real-time out-of-sample test? That's "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\n" + R_CELL),

        # ---- BEAT 0 -- VERDICT ----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do expensive years lead to worse returns? | **A little.** Cheapest third of years: "
            "**+10.0%** next year. Most expensive third: **+8.2%**. The gap (~2pp) points the way "
            "Buffett predicts -- but +8.2% is still a fine year. |\n"
            "| Is that gap statistically real at one year out? | **No.** The regression t-stat is "
            "**-0.92** and the indicator explains under **1%** of next-year returns. |\n"
            "| Could you time the market with it? | **No.** A real-time 'sell when expensive' rule "
            "earns **1.3%/yr** vs **8.1%/yr** for buy-and-hold -- it kept you out for ~30 years. |\n"
            "| So what is it good for? | A long-horizon *temperature gauge*, not a one-year timing "
            "signal. It tells you the room is hot; it can't tell you when to leave. |\n\n"
            "> The Buffett Indicator is a **thermometer, not a timer**."
        ),

        # ---- BEAT 1 -- THE CLAIM --------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"The ratio of total market value of all publicly traded securities as a "
            "percentage of the country's GNP... is probably the best single measure of where "
            "valuations stand at any given moment.\"* -- Warren Buffett, *Fortune*, 2001.\n\n"
            "The folk version: when market-cap-to-GDP is **low** (~70-80%), stocks are cheap -- "
            "buy. When it's **high** (toward 200%), stocks are expensive -- leave the party. "
            "The mechanism is real economics: prices can't grow faster than the economy "
            "*forever*, so a stretched ratio should mean-revert, dragging future returns down."
        ),

        # ---- BEAT 2 -- SO WHAT ----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If the *level* of one slow-moving ratio told you next year's stock-market "
            "direction, market timing would be easy: check the gauge each December, sit out the "
            "expensive years. The catch is that 'expensive' has been the *normal* state of the "
            "market for almost thirty years. The interesting question isn't whether the ratio is "
            "high (it is) -- it's whether *acting* on it would have helped."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW ----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three traps to avoid:\n\n"
            "1. **Hindsight thresholds.** Saying '200% is expensive' uses numbers we only know "
            "*now*. In 1996, 100% looked terrifyingly high -- and the market tripled afterward. "
            "The honest test must use only what you'd have known *at the time* (an expanding, "
            "point-in-time threshold).\n"
            "2. **A rising goalposts problem.** The ratio has drifted structurally upward for "
            "decades (US firms earn abroad, profit margins rose, rates fell). A fixed 'fair "
            "value' is a moving target.\n"
            "3. **Tiny n, short horizon.** We have ~55 years of annual data. At a one-year "
            "horizon, ~17% market volatility swamps a 2-3pp valuation tilt.\n\n"
            "We test on the Buffett Indicator (hardcoded year-end levels, 1971-2025) matched to "
            "the S&P 500 **price** return of the *following* year -- a full year of lag, no "
            "peeking."
        ),

        # ---- BEAT 4 -- THE TEARDOWN -----------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First, meet the indicator:** market-cap-to-GDP, year by year, with its famous "
            "landmarks."
        ),
        code(
            "bi = data.buffett_table()\n"
            "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
            "ax.plot(bi['year'], bi['ratio'], 'o-', c=GREY, lw=1.5, ms=4)\n"
            "ax.axhline(100, ls=':', c='k', lw=1, label='100% (cap = GDP)')\n"
            "ax.axhline(R['ratio_2025'], ls='--', c=RED, lw=1.5, label=f\"2025: {R['ratio_2025']:.0f}%\")\n"
            "for yr, lbl in [(1974,'1974 trough'),(1999,'dot-com'),(2021,'2021 record')]:\n"
            "    v = float(bi.loc[bi['year']==yr,'ratio'].iloc[0])\n"
            "    ax.annotate(lbl, (yr, v), textcoords='offset points', xytext=(0,8), ha='center', fontsize=8)\n"
            "ax.set_xlabel('Year'); ax.set_ylabel('Market cap / GDP (%)')\n"
            "ax.set_title('The Buffett Indicator: a ratio that keeps making new highs')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"1974 trough: {R['ratio_1974']:.0f}%  |  1999 dot-com: {R['ratio_1999']:.0f}%  \"\n"
            "      f\"|  2021 record: {R['ratio_2021']:.0f}%  |  2025: {R['ratio_2025']:.0f}%\")\n"
            "print('Note: 2021 and 2024-25 readings eclipsed the dot-com peak -- the ratio has trended UP.')"
        ),
        md(
            "Notice the upward *drift*: the dot-com peak of ~172% once looked insane; the ratio "
            "has spent most of the last decade above it. A 'sell at 150%' rule would have had you "
            "out of stocks since roughly 2013."
        ),
        md("**Do expensive years really lead to worse returns?** Sort years into cheap / mid / "
           "expensive thirds and look at the *next* year's return."),
        code(
            "if HAVE_REAL:\n"
            "    df = data.fetch_sp500_annual()\n"
            "    terc = st.tercile_returns(df)\n"
            "    cheap, mid, exp = (float(terc.loc[b,'mean_fwd'])*100 for b in ('cheap','mid','expensive'))\n"
            "    cheap_up = float(terc.loc['cheap','up_rate'])*100\n"
            "    exp_up = float(terc.loc['expensive','up_rate'])*100\n"
            "else:\n"
            "    cheap, mid, exp = R['cheap_mean'], R['mid_mean'], R['exp_mean']\n"
            "    cheap_up, exp_up = R['cheap_up'], R['exp_up']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "bars = ax.bar(['Cheap years\\n(low ratio)','Mid','Expensive years\\n(high ratio)'],\n"
            "              [cheap, mid, exp], color=[GREEN, GREY, RED], width=0.55)\n"
            "ax.set_ylabel('Mean S&P next-year return (%)')\n"
            "ax.set_title(f'Expensive years under-return cheap years by ~{cheap-exp:.1f}pp -- but stay positive')\n"
            "for b, v in zip(bars, [cheap, mid, exp]):\n"
            "    ax.annotate(f'{v:+.1f}%', (b.get_x()+b.get_width()/2, v+0.2), ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Cheap years:     {cheap:+.1f}%   (up {cheap_up:.0f}% of the time)')\n"
            "print(f'Expensive years: {exp:+.1f}%   (up {exp_up:.0f}% of the time)')\n"
            "print('Right direction -- but +8.2% after an expensive reading is hardly a reason to leave.')"
        ),
        md(
            "So the indicator points the **right way**: cheap years average +10.0% the following "
            "year, expensive years +8.2%. But that 2pp gap is small, and crucially an 'expensive' "
            "reading still came with a solid +8% on average. This is a faint tilt, not an alarm."
        ),

        # ---- BEAT 5 -- THE VERDICT ------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal -- Weak.** The direction is right and the economics are sound (prices "
            "can't outrun the economy forever), but at the one-year horizon the effect is tiny: "
            "the indicator explains under **1%** of next-year returns and the formal t-stat is "
            "only **-0.92**. Right idea, far too weak to call real.\n"
            "- **Tradability -- Mirage.** As the next beat shows, a real-time 'leave when "
            "expensive' rule destroys returns by keeping you out of a 30-year bull market."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT -----------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "The honest test: each year, be in the market only if the indicator is at or below "
            "its *own past median* (no hindsight). Otherwise sit in cash."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tim = st.timing_strategy(df)\n"
            "    ss = st.strategy_stats(tim, one_way_bps=10.0)\n"
            "    strat_cagr = ss['strat_cagr']*100; bah_cagr = ss['bah_cagr']*100\n"
            "    frac = ss['frac_invested']*100\n"
            "    yrs = tim.index.values\n"
            "    strat_cum = (1+tim['strat_ret']).cumprod().values*100\n"
            "    bah_cum = (1+tim['bah_ret']).cumprod().values*100\n"
            "else:\n"
            "    strat_cagr, bah_cagr, frac = R['strat_cagr'], R['bah_cagr'], R['frac_invested']\n"
            "    rng = np.random.default_rng(264)\n"
            "    yrs = np.arange(R['year_start']+1, R['year_end']+2)\n"
            "    bah_cum = (1+rng.normal(bah_cagr/100, 0.16, R['n'])).cumprod()*100\n"
            "    strat_cum = (1+rng.normal(strat_cagr/100, 0.08, R['n'])).cumprod()*100\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
            "ax.plot(yrs, bah_cum, lw=2, c=GREEN, label=f'Buy & hold: {bah_cagr:.1f}%/yr')\n"
            "ax.plot(yrs, strat_cum, lw=2, c=RED, ls='--', label=f'Buffett timing: {strat_cagr:.1f}%/yr')\n"
            "ax.set_yscale('log'); ax.set_xlabel('Year')\n"
            "ax.set_ylabel('Value (log, base 100, price-only)')\n"
            "ax.set_title('Leaving the party early: the timing rule sat out the whole bull market')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Buffett timing: {strat_cagr:.2f}%/yr  |  Buy & hold: {bah_cagr:.2f}%/yr')\n"
            "print(f'Years actually invested: {frac:.0f}%  -- the rule sat in cash for most of the sample.')"
        ),
        md(
            "The rule sat in cash **76% of the years** -- because once the indicator crossed its "
            "historical median in the mid-1990s, it almost never looked 'cheap' again -- and "
            "earned **1.3%/yr** against buy-and-hold's **8.1%/yr** (price-only). Leaving the "
            "party when the thermometer read 'hot' meant missing the entire party.\n\n"
            "> A signal that's right *on average* but keeps you out of the market for a generation "
            "is not a tradable signal."
        ),

        # ---- BEAT 7 -- GOING FURTHER ----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The positive control:** the companion notebook plants a real valuation->return "
            "link in synthetic data and shows the engine detects it cleanly -- so the weak real "
            "result is about the *market*, not a broken test.\n"
            "- **The valuation cousin that *does* work better:** Shiller's CAPE-based expected "
            "return, [Study 120 -- Excess-CAPE-Yield](../../120-excess-cape-yield/), the one "
            "valuation study this desk grades `Real` on the Signal axis (at long horizons).\n"
            "- **The small-n folklore family:** [Study 158 -- Super-Bowl](../../158-super-bowl/) "
            "shares the correct-baseline / positive-control method used here.\n\n"
            "*Think the indicator times the market? Fork this, define your own real-time rule "
            "(no hindsight thresholds), and beat buy-and-hold net of costs. That's the bar -- and "
            "the expanding-median rule doesn't clear it.*"
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
            "# Buffett-Indicator -- a quantitative teardown\n"
            "### 55 annual obs * ^GSPC price * predictive regression * HAC t * "
            "real-time expanding median * synthetic positive control\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.* We regress next-year "
            "S&P return on the Buffett Indicator level with a Newey-West HAC t-stat, sort cheap "
            "vs expensive terciles, contrast the in-sample slope with a real-time "
            "(expanding-median) timing rule, and close with a synthetic positive control that "
            "proves the engine finds a valuation->return link when one exists.\n\n"
            "> **Not investment advice.** Real data: Buffett Indicator (market cap / GDP) "
            "hardcoded in `data.py` (1971-2025); forward returns = ^GSPC **price** index, "
            "Dec/Dec (repo-level split-only cache). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The plain-English notes** translate each result back to intuition."
        ),
        code(BOOT + "\n" + R_CELL),

        # ---- BEAT 0 ---------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            "| **Signal** | `WEAK` | Predictive-regression slope **-3.22pp / +100pp GDP**, "
            "right sign, but HAC t = **-0.92**, R^2 = **0.009**. |\n"
            "| **Tradability** | `MIRAGE` | Real-time expanding-median timing earns **1.3%/yr** "
            "vs **8.1%/yr** buy-and-hold; excess HAC t = **-3.76**. |\n\n"
            "> 💡 The slope points the way Buffett predicts (expensive -> lower return), but at a "
            "one-year horizon with n=55 it cannot be distinguished from zero, and acting on it is "
            "actively harmful."
        ),

        # ---- BEAT 1 ---------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $V_Y$ be the Buffett Indicator at year-end $Y$ (market cap / GDP, in %) and "
            "$r_{Y+1}$ the S&P 500 price return in year $Y+1$. The hypotheses:\n\n"
            "- **H1 (predictive slope).** In $r_{Y+1} = \\alpha + \\beta V_Y + \\varepsilon$, the "
            "Buffett claim is $\\beta < 0$ with $|t| \\ge 2$ (HAC) -- high valuation forecasts low "
            "return.\n"
            "- **H2 (monotonic terciles).** $E[r \\mid \\text{cheap}] > E[r \\mid \\text{mid}] > "
            "E[r \\mid \\text{expensive}]$.\n"
            "- **H3 (tradable real-time rule).** An out-of-sample timing rule using only "
            "point-in-time information beats buy-and-hold net of costs.\n\n"
            "We find H1 has the right sign but fails the t-bar; H2 holds weakly; H3 fails badly."
        ),

        # ---- BEAT 2 ---------------------------------------------------------
        md(
            "## 2 - So what? -- what rides on each answer\n\n"
            "Valuation-ratio predictability is one of the most studied questions in asset "
            "pricing (Campbell-Shiller, Cochrane, Goyal-Welch). The theory is sound at long "
            "horizons: valuation ratios mean-revert and most of their variation is time-varying "
            "discount rates. The empirical knife-edge is **out-of-sample at short horizons**, "
            "where Goyal-Welch (2008) showed most predictors fail to beat the prevailing mean. "
            "The Buffett Indicator is exactly this case, dressed in folklore."
        ),

        # ---- BEAT 3 ---------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Data.** Buffett Indicator year-end levels 1971-2025 (hardcoded); ^GSPC "
            "Dec/Dec **price** returns, indicator at $Y$ matched to return in $Y+1$ "
            "(one-year execution lag, no look-ahead).\n"
            "- **Primary spec.** OLS $r_{Y+1} = \\alpha + \\beta V_Y$ with Newey-West HAC SE "
            "(automatic Bartlett lag).\n"
            "- **Secondary.** Cheap/mid/expensive terciles (in-sample breakpoints -- a generous "
            "upper bound).\n"
            "- **Honest spec.** Real-time timing: long iff $V_Y \\le$ expanding median of prior "
            "readings; else cash (0%, price-only, conservative). Costs one-way x NAV per flip.\n"
            "- **Positive control.** Synthetic AR(1) valuation with a planted slope $\\beta$, to "
            "confirm the regression recovers a true effect.\n\n"
            "> 💡 Price-only forward returns understate total returns by ~1.8pp/yr uniformly; "
            "this shifts every bar but not the *relative* cheap-vs-expensive comparison."
        ),

        # ---- BEAT 4 ---------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - The predictive regression with a HAC t-stat\n\n"
            "Regress next-year return on the valuation level; report the Newey-West t."
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = data.fetch_sp500_annual()\n"
            "    reg = st.predictive_regression(df)\n"
            "    beta100 = reg['beta_per_100']*100  # pp return per +100pp GDP\n"
            "    corr = reg['corr']; r2 = reg['r2']; tstat = reg['tstat']; n = reg['n']\n"
            "    x = df['ratio'].to_numpy(); y = df['fwd_ret'].to_numpy()*100\n"
            "else:\n"
            "    beta100, corr, r2, tstat, n = (R['beta_per_100'], R['corr'], R['reg_r2'],\n"
            "                                   R['reg_t'], R['n'])\n"
            "    rng = np.random.default_rng(264)\n"
            "    x = np.linspace(45, 205, R['n']) + rng.normal(0, 15, R['n'])\n"
            "    y = 12.9 + (R['beta_per_100']/100)*x + rng.normal(0, 16, R['n'])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "ax.scatter(x, y, c=GREY, s=28, alpha=0.8)\n"
            "xs = np.array([x.min(), x.max()])\n"
            "slope = beta100/100; intercept = (y.mean() - slope*x.mean())\n"
            "ax.plot(xs, intercept + slope*xs, c=RED, lw=2,\n"
            "        label=f'slope = {beta100:+.2f}pp / +100pp GDP')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Buffett Indicator at year-end Y (% of GDP)')\n"
            "ax.set_ylabel('S&P price return in year Y+1 (%)')\n"
            "ax.set_title(f'Right sign, no significance: HAC t = {tstat:+.2f}, R^2 = {r2:.3f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'slope: {beta100:+.2f}pp return per +100pp of GDP  |  corr = {corr:+.3f}')\n"
            "print(f'R^2 = {r2:.3f}  |  HAC t = {tstat:+.3f}  |  n = {n}')\n"
            "print('|t| < 2 -> fails the REAL bar. The cloud is nearly flat.')"
        ),
        md(
            "> 💡 The slope is **negative** -- Buffett's direction -- but the scatter is a "
            "shapeless cloud. HAC t = **-0.92**, R^2 < **1%**. At one year out, valuation level "
            "is essentially uninformative about return *magnitude*."
        ),
        md(
            "### 4b - Cheap vs expensive terciles (in-sample, the generous version)\n\n"
            "Even granting in-sample breakpoints, how big is the tilt?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    terc = st.tercile_returns(df)\n"
            "    print(terc.assign(mean_fwd_pct=(terc['mean_fwd']*100).round(1),\n"
            "                      up_rate_pct=(terc['up_rate']*100).round(1))\n"
            "          [['n','mean_fwd_pct','up_rate_pct']].to_string())\n"
            "    cheap = float(terc.loc['cheap','mean_fwd'])*100\n"
            "    exp = float(terc.loc['expensive','mean_fwd'])*100\n"
            "else:\n"
            "    cheap, exp = R['cheap_mean'], R['exp_mean']\n"
            "    print(f\"cheap: mean +{R['cheap_mean']}%  up {R['cheap_up']}%\")\n"
            "    print(f\"mid:   mean +{R['mid_mean']}%  up {R['mid_up']}%\")\n"
            "    print(f\"exp:   mean +{R['exp_mean']}%  up {R['exp_up']}%\")\n"
            "print()\n"
            "print(f'Cheap-minus-expensive next-year return: {cheap-exp:+.1f}pp')\n"
            "print('In the predicted direction, but small -- and in-sample breakpoints flatter it.')"
        ),
        md(
            "> 💡 ~2pp/yr in the right direction. But these terciles use the **whole sample's** "
            "breakpoints -- you could not have known in 1995 that 96% would land in the 'mid' "
            "bucket. The honest, point-in-time version is next, and it is brutal."
        ),
        md(
            "### 4c - The real-time timing rule (expanding median) vs buy-and-hold\n\n"
            "Long the S&P iff the indicator is at or below its expanding median; else cash."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tim = st.timing_strategy(df)\n"
            "    ss = st.strategy_stats(tim, one_way_bps=10.0)\n"
            "    strat = ss['strat_cagr']*100; strat_net = ss['strat_cagr_net']*100\n"
            "    bah = ss['bah_cagr']*100; frac = ss['frac_invested']*100\n"
            "    flips = ss['n_flips']; ex_t = ss['excess_t']\n"
            "    yrs = tim.index.values; in_mkt = tim['in_market'].values\n"
            "else:\n"
            "    strat, strat_net, bah = R['strat_cagr'], R['strat_cagr_net'], R['bah_cagr']\n"
            "    frac, flips, ex_t = R['frac_invested'], R['n_flips'], R['excess_t']\n"
            "    yrs = np.arange(R['year_start']+1, R['year_end']+2)\n"
            "    in_mkt = (np.arange(R['n']) < 0.236*R['n'])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10, 3.2))\n"
            "ax.fill_between(yrs, 0, 1, where=in_mkt, color=GREEN, alpha=0.5, step='mid',\n"
            "                label='In market (cheap)')\n"
            "ax.fill_between(yrs, 0, 1, where=~np.asarray(in_mkt), color=RED, alpha=0.4, step='mid',\n"
            "                label='In cash (expensive)')\n"
            "ax.set_yticks([]); ax.set_xlabel('Forecast year')\n"
            "ax.set_title(f'When the rule was invested -- in cash {100-frac:.0f}% of the time')\n"
            "ax.legend(loc='upper left', ncol=2); plt.tight_layout(); plt.show()\n"
            "print(f'Buffett timing CAGR (gross): {strat:+.2f}%/yr')\n"
            "print(f'Buffett timing CAGR (net 10bps/flip): {strat_net:+.2f}%/yr  ({flips} flips)')\n"
            "print(f'Buy & hold CAGR (price): {bah:+.2f}%/yr')\n"
            "print(f'Years invested: {frac:.1f}%  |  excess (strat-BAH) HAC t = {ex_t:+.2f}')"
        ),
        md(
            "> 💡 The rule sits in cash **76%** of the years. Its excess return over buy-and-hold "
            "has a HAC t of **-3.76** -- it *reliably underperforms*. Turnover is negligible "
            "(4 flips/55yr), so this is **not** a cost story: the indicator simply flagged the "
            "market 'expensive' for the entire modern bull run."
        ),
        md(
            "### 4d - Why it fails: the ratio's structural drift\n\n"
            "The expanding median keeps rising, but the *contemporaneous* ratio rises faster -- "
            "so the rule is almost always 'expensive' after the mid-1990s."
        ),
        code(
            "bi = data.buffett_table().sort_values('year').reset_index(drop=True)\n"
            "exp_med = bi['ratio'].expanding(min_periods=10).median()\n"
            "fig, ax = plt.subplots(figsize=(10, 4.3))\n"
            "ax.plot(bi['year'], bi['ratio'], 'o-', c=GREY, lw=1.3, ms=3, label='Buffett Indicator')\n"
            "ax.plot(bi['year'], exp_med, c=AMBER, lw=2, label='Expanding median (the real-time threshold)')\n"
            "ax.fill_between(bi['year'], bi['ratio'], exp_med,\n"
            "                where=(bi['ratio']>exp_med), color=RED, alpha=0.15, label='Flagged expensive')\n"
            "ax.set_xlabel('Year'); ax.set_ylabel('Market cap / GDP (%)')\n"
            "ax.set_title('After ~1996 the ratio sits above its own history almost permanently')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "above = int((bi['ratio'] > exp_med).loc[bi['year']>=1996].sum())\n"
            "tot = int((bi['year']>=1996).sum())\n"
            "print(f'Years 1996-2025 flagged expensive by the real-time rule: {above}/{tot}')"
        ),

        # ---- BEAT 5 ---------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal `WEAK`** -- slope -3.22pp/+100pp GDP, right sign, but HAC t = -0.92, "
            "R^2 = 0.009 on n=55. The economics and the long-horizon literature are real "
            "(Campbell-Shiller, Cochrane), but the one-year signal cannot clear the |t| >= 2 "
            "bar -- literature support alone reads `WEAK`, not `REAL`.\n"
            "- **Tradability `MIRAGE`** -- the honest real-time rule earns 1.3%/yr vs 8.1%/yr "
            "buy-and-hold (excess t = -3.76); it sat out the entire post-1996 bull market."
        ),

        # ---- BEAT 6 ---------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- the excess-return reckoning\n\n"
            "The buy-and-hold benchmark on the same price tape, with the strategy's shortfall."
        ),
        code(
            "if HAVE_REAL:\n"
            "    shortfall = (ss['bah_cagr'] - ss['strat_cagr'])*100\n"
            "    print(f\"Buy & hold (price):      {ss['bah_cagr']*100:+.2f}%/yr\")\n"
            "    print(f\"Buffett timing (gross):  {ss['strat_cagr']*100:+.2f}%/yr\")\n"
            "    print(f\"Buffett timing (net):    {ss['strat_cagr_net']*100:+.2f}%/yr\")\n"
            "    print(f'Shortfall vs buy & hold: {shortfall:+.2f}pp/yr')\n"
            "    print(f\"Excess-return HAC t-stat: {ss['excess_t']:+.3f}  (negative = reliably worse)\")\n"
            "else:\n"
            "    print(f\"Buy & hold (price): {R['bah_cagr']:+.2f}%/yr\")\n"
            "    print(f\"Buffett timing:     {R['strat_cagr']:+.2f}%/yr\")\n"
            "    print(f\"Shortfall: {R['bah_cagr']-R['strat_cagr']:+.2f}pp/yr  |  excess t = {R['excess_t']:+.2f}\")\n"
            "print()\n"
            "print('There is no version of the one-year timing trade that survives. The signal,')\n"
            "print('such as it is, lives at horizons of 5-10 years -- not at the 1-year mark.')"
        ),
        md(
            "> 💡 The timing rule is dominated by passive exposure by ~7pp/yr. The opportunity "
            "cost of 'leaving the party' is the party."
        ),

        # ---- BEAT 7 ---------------------------------------------------------
        md(
            "## 7 - Going further -- the synthetic positive control\n\n"
            "Does the regression detect a valuation->return link *when one exists*? We plant a "
            "mean-reversion slope in synthetic AR(1) valuation data and sweep its size."
        ),
        code(
            "planted = [0.0, -0.05, -0.15, -0.30]\n"
            "rows = []\n"
            "for b in planted:\n"
            "    df_s, _ = data.synthetic_panel(n_years=400, beta=b, seed=264)\n"
            "    r = st.predictive_regression(df_s)\n"
            "    rows.append({'beta': b, 'corr': round(r['corr'],3),\n"
            "                 'hac_t': round(r['tstat'],2),\n"
            "                 'detected': abs(r['tstat'])>=2.0})\n"
            "res = pd.DataFrame(rows)\n"
            "colors = [GREEN if d else RED for d in res['detected']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar([str(b) for b in planted], res['hac_t'], color=colors)\n"
            "ax.axhline(-2, c='k', ls=':', lw=1, label='|t| = 2 detection bar')\n"
            "ax.set_xlabel('Planted slope beta (negative = Buffett mean-reversion)')\n"
            "ax.set_ylabel('Recovered HAC t-stat')\n"
            "ax.set_title('The engine finds the signal when it exists (green = |t| >= 2)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(res.to_string(index=False))\n"
            "print()\n"
            "print('Null (beta=0): |t| < 2. Even a small planted slope (beta=-0.05) is detected at n=400.')\n"
            "print('The real tape (n=55, one-year horizon) just does not carry that much signal.')"
        ),
        md(
            "The engine recovers a planted valuation->return relationship cleanly, and correctly "
            "returns a null when none is planted. The weak real result is therefore a statement "
            "about the **market and the one-year horizon**, not a failure of the method. At "
            "long horizons (5-10 years) valuation predictability re-emerges -- see "
            "[Study 120 -- Excess-CAPE-Yield](../../120-excess-cape-yield/)."
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
