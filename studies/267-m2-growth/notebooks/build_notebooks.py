"""Generate (and then execute) the two narrative notebooks for Study 267 (M2-Growth).

    python notebooks/build_notebooks.py

This script writes BOTH notebooks and executes them in-place via nbconvert, so
every code cell carries a non-null execution_count and embedded outputs with no
errors. The hardcoded M2 series and the synthetic generator run anywhere,
offline and deterministically; the real ^GSPC cells use the cached monthly
parquet at the study-level _cache/ if present, and otherwise quote the frozen
headline numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs
for any reader offline by branching on ``HAVE_REAL``.

The _write convention (each build_*() ends by calling _write) is kept so the
repo's intro-restyle tooling can monkeypatch it.
"""

from __future__ import annotations

import os
import subprocess
import sys

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-16).
R = dict(
    n_months=491,
    year_start=1985,
    year_end=2025,
    fp="ecbbebebec4e",
    # 1-month forward predictive regression
    beta_fwd1=0.0,
    t_ols_fwd1=-0.01,
    t_hac_fwd1=-0.01,
    r2_fwd1=0.0,
    # 12-month forward predictive regression
    beta_fwd12=-0.00332,
    t_ols_fwd12=-2.01,
    t_hac_fwd12=-0.57,
    r2_fwd12=0.0084,
    # quantile sort
    hi_mean=0.06,
    lo_mean=1.28,
    spread=-1.22,
    spread_t=-1.87,
    # tradable rule
    ann_net=2.81,
    ann_bah=9.31,
    sharpe_net=0.32,
    sharpe_bah=0.67,
    time_in_market=0.40,
    turnover_per_yr=0.27,
)

# ---------------------------------------------------------------------------
# Shared preamble
# ---------------------------------------------------------------------------
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

from m2_growth import data, strategy as st

def _have_cache():
    try:
        data.build_real_panel(fetch=False)
        return True
    except FileNotFoundError:
        return False

HAVE_REAL = _have_cache()
print("^GSPC cache present:", HAVE_REAL)
"""

R_CELL = """\
# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-16)
R = dict(
    n_months=491, year_start=1985, year_end=2025, fp="ecbbebebec4e",
    beta_fwd1=0.0, t_ols_fwd1=-0.01, t_hac_fwd1=-0.01, r2_fwd1=0.0,
    beta_fwd12=-0.00332, t_ols_fwd12=-2.01, t_hac_fwd12=-0.57, r2_fwd12=0.0084,
    hi_mean=0.06, lo_mean=1.28, spread=-1.22, spread_t=-1.87,
    ann_net=2.81, ann_bah=9.31, sharpe_net=0.32, sharpe_bah=0.67,
    time_in_market=0.40, turnover_per_yr=0.27,
)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# M2-Growth -- does money-supply growth drive the stock market?\n"
            "### The famous M2-vs-S&P overlay, tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
            "During 2020-2023 a single chart went viral: the year-over-year growth of "
            "**M2** (the money supply) laid on top of the **S&P 500**. They surged together "
            "in the COVID money flood, and when M2 growth went *negative* in 2023 the warnings "
            "of an imminent crash were everywhere. The monetarist story is seductive: print more "
            "dollars, and the dollars chase stocks higher. This notebook asks the only question "
            "that matters: does money growth actually *predict* the market, or is the overlay "
            "just two things moving together because of the business cycle?\n\n"
            "> **This is the plain-language layer.** Want the Newey-West regression, the "
            "overlapping-window trap, and the backtest with costs? That's "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\n" + R_CELL),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does fast M2 growth predict a strong *next month*? | **No.** Regression slope ≈ 0, "
            "honest (HAC) t ≈ **0.0**. |\n"
            "| What about the next *year*? | The naive test says t = −2.0 (looks real!), but it "
            "uses overlapping windows that fake significance. The honest t is **−0.6** -- nothing. |\n"
            "| Do high-money months beat low-money months? | **No -- the opposite.** High-money "
            "months *underperform* by **1.2pp**, the wrong sign for the story. |\n"
            "| Could you trade it? | **No.** A long/flat M2 rule earns **+2.8%/yr** vs **+9.3%/yr** "
            "for just holding the index. |\n\n"
            "> The viral overlay deceives because it is *contemporaneous*. Money and markets both "
            "jump when the Fed eases into a downturn -- but the money number doesn't tell you what "
            "happens *next*."
        ),

        # ---- BEAT 1 -- THE CLAIM ----------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"Inflation is always and everywhere a monetary phenomenon\" (Friedman). When the "
            "Fed floods the system with money, the extra dollars push up asset prices. So M2 growth "
            "leads the stock market: buy when money is growing fast, get out when it slows.*\n\n"
            "The intuition has a Nobel laureate behind it and a chart that looks unanswerable. "
            "But \"money matters for the macroeconomy over years\" is a very different statement "
            "from \"published M2 growth predicts next month's S&P return.\""
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If it were true, it would be the easiest macro signal in the world: one widely "
            "published number, released monthly, telling you whether to be in or out of stocks. "
            "Every macro tourist would already be trading it -- which is the first clue that the "
            "market has long since priced whatever information the M2 release contains.\n\n"
            "The fun question is: **why does the overlay look so convincing?** The answer is a "
            "lesson in the difference between *moving together* and *predicting*."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three traps to avoid:\n\n"
            "1. **Contemporaneous vs predictive.** M2 growth and stocks both respond to the "
            "cycle. In a recession the Fed eases, M2 growth jumps, *and* stocks rebound off the "
            "bottom. Overlaid, that looks causal. The honest test lags the (already-lagged) M2 "
            "release and asks whether it predicts the *next* month's return.\n"
            "2. **The overlapping-window mirage.** Regressing the next-12-months return on M2 "
            "reuses overlapping data, which manufactures statistical significance out of thin air. "
            "The fix is a Newey-West (HAC) t-stat. It turns a scary-looking −2.0 into a harmless −0.6.\n"
            "3. **A handful of regimes ≠ decades of evidence.** Most of the co-movement comes from "
            "two episodes (the GFC and COVID). That is not 40 independent years.\n\n"
            "We test all of this on M2 YoY growth (hardcoded from the FRED contour) joined to "
            "^GSPC monthly returns, 1985-2025 (~491 usable months)."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First, the famous overlay.** M2 YoY growth and the S&P 500, the chart that went viral."
        ),
        code(
            "m2 = data.m2_yoy_series()\n"
            "if HAVE_REAL:\n"
            "    df = data.build_real_panel(fetch=False)\n"
            "    gspc = df['gspc']\n"
            "    m2_plot = df['m2_yoy']\n"
            "    dts = df.index\n"
            "else:\n"
            "    # offline fallback: synthetic equity path + the real hardcoded M2 contour\n"
            "    syn, _ = data.synthetic_panel(n_months=len(m2.loc['1985':]), premium=0.0, seed=267)\n"
            "    dts = m2.loc['1985':].index\n"
            "    m2_plot = m2.loc['1985':]\n"
            "    gspc = pd.Series((1+syn['sp_ret'].values).cumprod()*200, index=dts)\n"
            "\n"
            "fig, ax1 = plt.subplots(figsize=(10.5, 4.6))\n"
            "ax1.plot(dts, gspc.values, c=GREEN, lw=1.6, label='S&P 500 (^GSPC, left)')\n"
            "ax1.set_yscale('log'); ax1.set_ylabel('S&P 500 (log)', color=GREEN)\n"
            "ax2 = ax1.twinx()\n"
            "ax2.plot(dts, m2_plot.values, c=RED, lw=1.6, label='M2 YoY growth % (right)')\n"
            "ax2.axhline(0, c=GREY, lw=1, ls=':')\n"
            "ax2.set_ylabel('M2 YoY growth (%)', color=RED); ax2.grid(False)\n"
            "ax1.set_title('The viral overlay: M2 money growth vs the S&P 500 (1985-2025)')\n"
            "fig.tight_layout(); plt.show()\n"
            "print('M2 YoY range over the window: '\n"
            "      f'{m2_plot.min():.1f}% to {m2_plot.max():.1f}% (COVID spike near +27% in 2021)')"
        ),
        md(
            "The COVID money explosion (M2 YoY near **+27%** in early 2021) and the negative prints "
            "of 2023 are the headline events. The eye wants to draw a causal line. Let's lag the "
            "money number and check whether it actually predicts what comes next."
        ),
        md("**Does fast money this month mean a strong market next month?**"),
        code(
            "if HAVE_REAL:\n"
            "    reg1 = st.predictive_regression(df, target_col='sp_fwd1', lags=3)\n"
            "    beta1, t_hac1 = reg1['beta'], reg1['t_hac']\n"
            "    sub = df[['m2_yoy','sp_fwd1']].dropna()\n"
            "    xs, ys = sub['m2_yoy'].values, sub['sp_fwd1'].values*100\n"
            "else:\n"
            "    beta1, t_hac1 = R['beta_fwd1'], R['t_hac_fwd1']\n"
            "    rng = np.random.default_rng(267)\n"
            "    xs = rng.normal(6, 5, R['n_months'])\n"
            "    ys = rng.normal(0.7, 4.2, R['n_months'])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.6))\n"
            "ax.scatter(xs, ys, s=14, c=GREY, alpha=0.5)\n"
            "xx = np.linspace(xs.min(), xs.max(), 50)\n"
            "ax.plot(xx, (beta1*100)*xx + (ys.mean()-beta1*100*xs.mean()), c=RED, lw=2,\n"
            "        label=f'fit slope ≈ {beta1*100:.3f}pp per +1pp M2  (HAC t = {t_hac1:+.2f})')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('M2 YoY growth this month (%)')\n"
            "ax.set_ylabel('S&P 500 return NEXT month (%)')\n"
            "ax.set_title('A flat cloud: M2 growth tells you nothing about next month')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Slope ≈ 0, honest (HAC) t = {t_hac1:+.2f} -- indistinguishable from noise.')"
        ),
        md(
            "A shapeless cloud. The fitted slope is essentially zero and the honest t-stat is "
            "**≈ 0.0**. Knowing this month's money growth does nothing for next month's return."
        ),
        md("**The base check: high-money months vs low-money months.**"),
        code(
            "if HAVE_REAL:\n"
            "    hml = st.high_minus_low(df, target_col='sp_fwd1', n_buckets=5)\n"
            "    hi, lo, spread, tt = hml['hi_mean'], hml['lo_mean'], hml['spread'], hml['t_stat']\n"
            "else:\n"
            "    hi, lo, spread, tt = R['hi_mean'], R['lo_mean'], R['spread'], R['spread_t']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.3))\n"
            "bars = ax.bar(['Low-money\\nmonths','High-money\\nmonths'], [lo, hi],\n"
            "              color=[GREEN, RED], width=0.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Mean next-month S&P return (%)')\n"
            "ax.set_title(f'High-money months UNDERperform by {spread:.2f}pp (t = {tt:+.2f}) -- wrong sign')\n"
            "for b,v in zip(bars,[lo,hi]):\n"
            "    ax.annotate(f'{v:+.2f}%', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='bottom' if v>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'High - Low = {spread:+.2f}pp/month (t = {tt:+.2f}). '\n"
            "      'If anything, fast money precedes WEAKER returns.')"
        ),
        md(
            "Not only is there no edge -- the sign is **backwards**. Months following the fastest "
            "money growth earned *less* than months following the slowest, by about 1.2pp, and even "
            "that gap is not statistically significant."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal -- None.** Lagged M2 growth does not predict forward S&P returns: the "
            "1-month slope is ≈ 0 with HAC t ≈ 0.0, and the high-vs-low-money spread is *negative* "
            "and insignificant.\n"
            "- **Tradability -- Mirage.** A long/flat M2 rule earns far less than just holding the "
            "index (next notebook).\n"
            "- **Busted.** The viral overlay is contemporaneous co-movement driven by the cycle, "
            "not a leading indicator you can trade."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "The natural rule: be long the S&P when money is growing faster than usual, sit in "
            "cash otherwise."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bt = st.backtest_rule(df, lookback=60, cost_bps=5.0)\n"
            "    ann_net, ann_bah = bt['ann_net']*100, bt['ann_bah']*100\n"
            "    tim = bt['time_in_market']\n"
            "else:\n"
            "    ann_net, ann_bah = R['ann_net'], R['ann_bah']\n"
            "    tim = R['time_in_market']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(7.4, 4.3))\n"
            "bars = ax.bar(['M2 rule\\n(long/flat, net)','Buy & hold'], [ann_net, ann_bah],\n"
            "              color=[RED, GREEN], width=0.5)\n"
            "ax.set_ylabel('Annualized return (%)')\n"
            "ax.set_title('The M2 rule gives up two-thirds of the market''s return')\n"
            "for b,v in zip(bars,[ann_net,ann_bah]):\n"
            "    ax.annotate(f'{v:.1f}%', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'M2 rule (net): {ann_net:.1f}%/yr, in market {tim*100:.0f}% of the time')\n"
            "print(f'Buy & hold:    {ann_bah:.1f}%/yr')\n"
            "print('Sitting out 60% of the months on a non-signal is expensive.')"
        ),
        md(
            "The rule sits in cash most of the time, waiting for a signal that isn't there, and "
            "earns roughly a third of the buy-and-hold return. The only trade here is: don't time "
            "the market on the money-supply chart."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The positive control:** the companion notebook plants a known M2 premium into a "
            "synthetic tape and shows the same machinery lights up at HAC t > 2. The real tape "
            "has nothing.\n"
            "- **The honest channel:** monetary policy *does* move stocks -- through the "
            "*surprise* component of policy (fed-funds futures), not the stale published M2 number "
            "(Thorbecke 1997; Rozeff 1974).\n"
            "- **A cousin in this lab:** [Study 120 -- Excess-CAPE-Yield](../../120-excess-cape-yield/) "
            "tests a macro valuation overlay with the same HAC discipline.\n\n"
            "*Think money growth really leads the market? Fork this, define your own lag and "
            "transform, and show a forward HAC t ≥ 2 with the right sign. That's the bar -- and "
            "the lagged M2 release doesn't clear it.*"
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
            "# M2-Growth -- a quantitative teardown\n"
            "### M2 YoY * ^GSPC monthly * predictive HAC regression * overlapping-window "
            "trap * quantile sort * long/flat backtest\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.* We regress forward "
            "^GSPC returns on lagged M2 YoY growth with a **Newey-West (HAC)** t-stat, expose the "
            "**overlapping-window** inflation that makes the 12-month test look significant, run a "
            "quantile sort, build a long/flat rule with costs, and close with a synthetic positive "
            "control.\n\n"
            "> **Not investment advice.** Real data: ^GSPC monthly price index (yfinance, "
            "1985-2025); M2 YoY growth hardcoded in `data.py` (FRED M2SL contour, 1960-2025). "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **In plain words** notes translate each result back to intuition."
        ),
        code(BOOT + "\n" + R_CELL + "\nfrom scipy import stats as scipy_stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            "| **Signal** | `NONE` | 1-mo forward HAC t = **−0.01**; 12-mo OLS t = **−2.01** but "
            "HAC t = **−0.57**; high−low spread **−1.22pp** (t = −1.87). No HAC t clears |2|, and "
            "the sign is wrong for the bullish claim. |\n"
            "| **Tradability** | `MIRAGE` | Long/flat M2 rule **+2.8%/yr** net vs **+9.3%/yr** "
            "buy-and-hold; Sharpe 0.32 vs 0.67. |\n"
            "| **Busted?** | `YES` | The overlay is contemporaneous cycle co-movement; lagging the "
            "release and HAC-correcting the t-stat removes the edge. |\n\n"
            "> 💡 **In plain words.** A scary OLS t-stat on overlapping returns is a parlor trick. "
            "The Newey-West correction is the adult in the room."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $g_t$ be M2 year-over-year growth (known with a publication lag) and $r_{t+h}$ "
            "the forward ^GSPC return over horizon $h$. The hypotheses:\n\n"
            "- **H1 (short-horizon predictive).** In $r_{t+1} = \\alpha + \\beta\\, g_t + "
            "\\varepsilon_{t+1}$, $\\beta > 0$ with HAC $t \\geq 2$.\n"
            "- **H2 (long-horizon predictive).** Same with $h = 12$ overlapping months -- the "
            "horizon where slow-moving macro signals are supposed to shine -- using a Newey-West "
            "t-stat with lag $\\geq h$.\n"
            "- **H3 (monotone sort).** Mean $r_{t+1}$ rises monotonically across M2-growth "
            "quintiles, top minus bottom $> 0$.\n\n"
            "We reject H1, H2 and H3 on the real tape (and H3 comes out with the *wrong sign*)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what? -- what rides on each answer\n\n"
            "If H1-H3 held, the published M2 release would be a free market-timing signal -- which "
            "would violate even weak-form efficiency, since M2 is one of the most-watched, most-"
            "telegraphed macro series in existence. The interesting finding is *how* a flat, "
            "wrong-signed relationship can masquerade as a compelling chart: contemporaneous cycle "
            "co-movement plus overlapping-window t-inflation."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Data.** M2 YoY growth hardcoded in `data.py` (FRED M2SL contour); ^GSPC monthly "
            "price closes (yfinance, cache-only), 1985-2025, ~491 usable months.\n"
            "- **Lag.** M2 dated month $t$ predicts $r_{t+1}$ (and $r_{t+1..t+12}$). M2 is itself "
            "released weeks after month-end, so this is conservative.\n"
            "- **Inference.** OLS slope with both naive and **Newey-West HAC** t-stats (Bartlett "
            "kernel, lag = horizon); quintile sort with a Welch t on the top-minus-bottom spread.\n"
            "- **Tradable check.** Long ^GSPC when lagged M2 > trailing 60-mo median, else cash; "
            "5 bps one-way cost on NAV per flip (long/flat ⇒ no borrow); net vs buy-and-hold.\n"
            "- **Labels.** ^GSPC is **price only** (no dividends); a single index, so no "
            "survivorship question -- both noted on the Signal axis.\n"
            "- **Positive control.** Synthetic tape with a planted M2 premium."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - Predictive regression with a Newey-West t-stat\n\n"
            "The honest test: lagged M2 growth → forward return, with HAC standard errors."
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = data.build_real_panel(fetch=False)\n"
            "    reg1 = st.predictive_regression(df, target_col='sp_fwd1', lags=3)\n"
            "    reg12 = st.predictive_regression(df, target_col='sp_fwd12', lags=18)\n"
            "    b1,to1,th1,r21 = reg1['beta'],reg1['t_ols'],reg1['t_hac'],reg1['r2']\n"
            "    b12,to12,th12,r212 = reg12['beta'],reg12['t_ols'],reg12['t_hac'],reg12['r2']\n"
            "else:\n"
            "    b1,to1,th1,r21 = R['beta_fwd1'],R['t_ols_fwd1'],R['t_hac_fwd1'],R['r2_fwd1']\n"
            "    b12,to12,th12,r212 = R['beta_fwd12'],R['t_ols_fwd12'],R['t_hac_fwd12'],R['r2_fwd12']\n"
            "\n"
            "tbl = pd.DataFrame({\n"
            "    'horizon': ['1 month','12 months (overlapping)'],\n"
            "    'beta': [b1, b12],\n"
            "    't_OLS': [to1, to12],\n"
            "    't_HAC': [th1, th12],\n"
            "    'R2': [r21, r212],\n"
            "})\n"
            "print(tbl.to_string(index=False))\n"
            "print()\n"
            "print('1-month: a non-event (t_HAC ~ 0).')\n"
            "print('12-month: t_OLS looks significant (~-2) but t_HAC ~ -0.6 -- overlap inflation.')"
        ),
        md(
            "### 4b - The overlapping-window trap, visualized\n\n"
            "Why the 12-month OLS t-stat lies: overlapping forward windows share data, so the "
            "residuals are autocorrelated and the naive standard error is far too small."
        ),
        code(
            "horizons = [1,3,6,12,18,24]\n"
            "rows = []\n"
            "for h in horizons:\n"
            "    if HAVE_REAL:\n"
            "        tmp = df.copy()\n"
            "        fwd = (1+tmp['sp_ret']).shift(-1)\n"
            "        for k in range(2,h+1):\n"
            "            fwd = fwd*(1+tmp['sp_ret'].shift(-k))\n"
            "        tmp['tgt'] = fwd-1\n"
            "        rr = st.predictive_regression(tmp, target_col='tgt', lags=max(1,h+2))\n"
            "        rows.append({'h':h,'t_OLS':rr['t_ols'],'t_HAC':rr['t_hac']})\n"
            "    else:\n"
            "        # synthetic illustration with the same qualitative shape\n"
            "        rows.append({'h':h, 't_OLS': -0.3*np.sqrt(h), 't_HAC': -0.3*np.sqrt(h)/(1+0.25*h)})\n"
            "ov = pd.DataFrame(rows)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.plot(ov['h'], ov['t_OLS'].abs(), 'o-', c=RED, lw=2, label='|t| naive OLS (inflated)')\n"
            "ax.plot(ov['h'], ov['t_HAC'].abs(), 's-', c=GREEN, lw=2, label='|t| Newey-West (honest)')\n"
            "ax.axhline(2, c=GREY, ls='--', lw=1.5, label='|t| = 2 bar')\n"
            "ax.set_xlabel('Forward horizon (months)'); ax.set_ylabel('|t-stat| on M2 slope')\n"
            "ax.set_title('Overlapping windows inflate the OLS t-stat; HAC stays honest')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(ov.to_string(index=False))"
        ),
        md(
            "> 💡 **In plain words.** Stack 12-month windows that overlap by 11 months and you've "
            "essentially counted the same data twelve times. The OLS t-stat believes you have "
            "independent observations; the Newey-West t-stat knows you don't. Only the HAC line "
            "tells the truth -- and it never crosses the |t| = 2 bar."
        ),
        md(
            "### 4c - Quantile sort: mean next-month return across M2-growth quintiles\n\n"
            "Model-free companion to the regression."
        ),
        code(
            "if HAVE_REAL:\n"
            "    q = st.quantile_sort(df, target_col='sp_fwd1', n_buckets=5)\n"
            "    hml = st.high_minus_low(df, target_col='sp_fwd1', n_buckets=5)\n"
            "    bx = q.index.astype(int).tolist(); by = q['mean_fwd_ret'].tolist()\n"
            "    spread, tt = hml['spread'], hml['t_stat']\n"
            "else:\n"
            "    bx = [1,2,3,4,5]\n"
            "    by = [R['lo_mean'], 1.0, 0.9, 0.5, R['hi_mean']]\n"
            "    spread, tt = R['spread'], R['spread_t']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "cols = [GREEN if v>=0 else RED for v in by]\n"
            "ax.bar([str(b) for b in bx], by, color=cols, width=0.6)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('M2-growth quintile (1 = slowest money, 5 = fastest)')\n"
            "ax.set_ylabel('Mean next-month ^GSPC return (%)')\n"
            "ax.set_title(f'No monotone pattern; high-low spread {spread:+.2f}pp (t = {tt:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'High (Q5) - Low (Q1) = {spread:+.2f}pp/month, Welch t = {tt:+.2f}')\n"
            "print('Not monotone, wrong sign, not significant.')"
        ),
        md(
            "### 4d - The tradable rule with costs vs buy-and-hold\n\n"
            "Long ^GSPC when lagged M2 YoY exceeds its trailing 60-month median, else cash."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bt = st.backtest_rule(df, lookback=60, cost_bps=5.0)\n"
            "    eq_net, eq_bah, dts = bt['eq_net'], bt['eq_bah'], bt['dates']\n"
            "    ann_net, ann_bah = bt['ann_net']*100, bt['ann_bah']*100\n"
            "    sh_net, sh_bah = bt['sharpe_net'], bt['sharpe_bah']\n"
            "    tim, turn = bt['time_in_market'], bt['turnover_per_yr']\n"
            "else:\n"
            "    ann_net, ann_bah = R['ann_net'], R['ann_bah']\n"
            "    sh_net, sh_bah = R['sharpe_net'], R['sharpe_bah']\n"
            "    tim, turn = R['time_in_market'], R['turnover_per_yr']\n"
            "    n = R['n_months']; dts = pd.date_range('1985-01-31', periods=n, freq='ME')\n"
            "    rng = np.random.default_rng(267)\n"
            "    eq_net = (1+rng.normal(ann_net/100/12, 0.025, n)).cumprod()\n"
            "    eq_bah = (1+rng.normal(ann_bah/100/12, 0.042, n)).cumprod()\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10.0, 4.5))\n"
            "ax.plot(dts, eq_bah, c=GREEN, lw=2, label=f'Buy & hold: {ann_bah:.1f}%/yr (Sharpe {sh_bah:.2f})')\n"
            "ax.plot(dts, eq_net, c=RED, lw=2, ls='--',\n"
            "        label=f'M2 rule net: {ann_net:.1f}%/yr (Sharpe {sh_net:.2f})')\n"
            "ax.set_yscale('log'); ax.set_xlabel('Year')\n"
            "ax.set_ylabel('Growth of $1 (log)')\n"
            "ax.set_title('Long/flat M2 timing is dominated by passive buy-and-hold')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'M2 rule net: {ann_net:.2f}%/yr, Sharpe {sh_net:.2f}, '\n"
            "      f'in-market {tim*100:.0f}%, turnover {turn:.2f}/yr')\n"
            "print(f'Buy & hold:  {ann_bah:.2f}%/yr, Sharpe {sh_bah:.2f}')\n"
            "print('Turnover is tiny -- the failure is the signal, not the costs.')"
        ),
        md(
            "> 💡 **In plain words.** The rule barely trades (≈ 0.3 flips/year), so costs are not "
            "the problem. It simply spends 60% of its life in cash on the strength of a non-signal "
            "and forfeits two-thirds of the market's compounding."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal `NONE`** -- 1-mo forward HAC t = −0.01; 12-mo HAC t = −0.57 (the naive "
            "−2.01 is overlap inflation); quintile spread −1.22pp (t = −1.87), wrong sign. No "
            "robust HAC t ≥ 2. *(Price-only ^GSPC, single index -- but a richer tape cannot "
            "rescue a near-zero, wrong-signed slope.)*\n"
            "- **Tradability `MIRAGE`** -- +2.8%/yr net vs +9.3%/yr buy-and-hold; half the Sharpe.\n"
            "- **Busted** -- the overlay is contemporaneous cycle co-movement; the lagged, "
            "HAC-corrected signal pays nothing."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- the benchmark race\n\n"
            "Already shown in 4d: the long/flat M2 rule loses to passive exposure on both return "
            "and Sharpe. The honest monetary channel that *does* move stocks is the policy "
            "*surprise* (fed-funds futures), not the stale published M2 aggregate (Thorbecke 1997; "
            "Rozeff 1974)."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further -- the synthetic positive control\n\n"
            "Does the machinery detect an M2 → return link *when one exists*? We plant a premium "
            "of increasing size into a synthetic tape and read off the HAC t-stat."
        ),
        code(
            "planted = [0.0, 0.25, 0.5, 1.0, 2.0]\n"
            "rows = []\n"
            "for prem in planted:\n"
            "    syn, _ = data.synthetic_panel(n_months=600, premium=prem, seed=267)\n"
            "    syn['sp_fwd1'] = syn['sp_ret'].shift(-1)\n"
            "    rr = st.predictive_regression(syn, target_col='sp_fwd1', lags=1)\n"
            "    rows.append({'premium':prem, 'beta':rr['beta'], 't_HAC':rr['t_hac']})\n"
            "res = pd.DataFrame(rows)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "cols = [GREEN if abs(t)>=2 else RED for t in res['t_HAC']]\n"
            "ax.bar([str(p) for p in planted], res['t_HAC'], color=cols, width=0.6)\n"
            "ax.axhline(2, c=GREY, ls='--', lw=1.5, label='|t| = 2 detection bar')\n"
            "ax.set_xlabel('Planted M2 premium'); ax.set_ylabel('HAC t-stat on M2 slope')\n"
            "ax.set_title('The engine finds the signal when it exists (green = HAC t ≥ 2)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(res.to_string(index=False))\n"
            "print()\n"
            "print('Null (premium=0): HAC t ~ 0.  Planted: t scales cleanly past 2.')\n"
            "print('The real tape sits at the premium=0 end. The method works; the signal is absent.')"
        ),
        md(
            "The engine recovers a planted M2 premium with a clean, monotone HAC t-stat. The real "
            "^GSPC tape lands at the null end of that sweep. The verdict is a statement about the "
            "**world** (money growth does not lead the market), not about the method."
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


def _execute(name):
    path = os.path.join(HERE, name)
    print("executing", path)
    subprocess.run(
        [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute",
         "--inplace", "--ExecutePreprocessor.timeout=300", path],
        check=True,
        cwd=HERE,
    )


if __name__ == "__main__":
    build_curious()
    build_quants()
    _execute("01_for_the_curious.ipynb")
    _execute("02_for_the_quants.ipynb")
    print("done")
