"""Generate the two narrative notebooks for Study 174 (Bitcoin-Rainbow).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; real-tape cells use the cached BTC daily
parquet (shared from Study 84, Moon-Math) if present and otherwise fall back to frozen
headline numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for any
reader without network access.

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
R = dict(
    # tape
    n=4287, start="2014-09-17", end="2026-06-12", fp="902481b7aa10",
    # in-sample (look-ahead biased)
    is_ann=20.7, is_sharpe=1.16, is_t=4.53, is_time=15.3,
    # walk-forward (honest)
    wf_ann=2.2, wf_sharpe=0.09, wf_t=0.39, wf_time=19.2,
    # buy-and-hold
    bh_ann=33.7,
    # look-ahead gap
    delta_t=4.14,
    # cost sweep (wf)
    wf_ann_c10=2.0, wf_ann_c50=1.4, wf_ann_c100=0.6,
    wf_t_c10=0.36, wf_t_c50=0.25, wf_t_c100=0.11,
    # buy/sell thresholds
    buy_sigma=-1.0, sell_sigma=1.5,
)

# ---------------------------------------------------------------------------
# Shared analysis preamble
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

from bitcoin_rainbow import data, strategy as st

def _have_cache():
    try:
        data.fetch_btc(fetch=False)
        return True
    except FileNotFoundError:
        return False

HAVE_REAL = _have_cache()
print("real BTC cache present:", HAVE_REAL)
"""

# Band colours matching the original rainbow chart (cold to hot).
BAND_COLOURS = ["#0d3b8c", "#1565c0", "#1976d2", "#4caf50",
                "#fdd835", "#ff9800", "#e64a19", "#b71c1c", "#4a0000"]


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Bitcoin-Rainbow -- the prettiest curve-fit in crypto\n"
            "### Log-time regression bands tested honestly: does buying the 'fire sale' and selling the 'bubble' work?\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Overfitting%3F: Busted](https://img.shields.io/badge/Overfitting%3F-Busted-8b949e?style=flat-square)\n\n"
            "You have almost certainly seen it: a Bitcoin price chart with nine coloured bands "
            "stacked like a rainbow -- from icy blue ('Fire sale') at the bottom to deep red "
            "('Maximum bubble territory') at the top. The chart appears to *nail* every Bitcoin "
            "cycle: the 2017 top is in the red zone, the 2018-2019 lows in the blue zone, the "
            "2021 peak flirting with the very top band. Influencers point at it and say: *'the "
            "model.* Buy blue. Sell red. Simple.'\n\n"
            "This notebook asks the only question that matters: **is it actually predictive, or "
            "just a line retrofitted through history?**\n\n"
            "> This is the plain-language layer. For t-stats, bootstrap intervals, and the full "
            "statistical teardown, see **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n\n"
            "> **Not investment advice.** A reproducible research tool; every chart is drawn by "
            "the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the in-sample rainbow look prescient? | **Yes -- spectacularly.** "
            f"Ann. return +{R['is_ann']}%, Sharpe +{R['is_sharpe']:.2f}, t = +{R['is_t']:.2f}. "
            "Every top and bottom is 'called' correctly. |\n"
            "| Does it work without looking at future prices? | **No.** Walk-forward "
            f"(honest) ann. return = +{R['wf_ann']}%, t = +{R['wf_t']:.2f} -- noise. |\n"
            f"| How much of that t = +{R['is_t']:.2f} was look-ahead? | **All of it.** "
            f"The look-ahead gap is +{R['delta_t']:.2f} t-points. |\n"
            "| Could you just hold Bitcoin? | "
            f"**Yes and it trounces the strategy: +{R['bh_ann']}%/yr** vs +{R['wf_ann']}%/yr walk-forward. |\n\n"
            "> The rainbow chart's entire appeal is a curve that was drawn *after* the prices it appears "
            "to predict. Strip that out and the signal evaporates."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *'Bitcoin's price has historically followed a power-law growth curve relative to "
            "its age. The Rainbow Chart plots this log-linear trend with coloured bands that "
            "identify when BTC is over- or undervalued. Buy in the cold bands, sell in the hot "
            "bands.'*  -- Blockchaincenter.net and a thousand crypto Twitter accounts\n\n"
            "The chart fits a single regression:\n\n"
            "    log(price) = alpha + beta * log(days since Bitcoin genesis)\n\n"
            "The bands are placed at multiples of the regression residual standard deviation. "
            "The colour scheme runs from icy blue ('Fire sale', bottom) through green ('HODL!') "
            "to deep red ('Maximum bubble territory', top)."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If it works, it is a simple, regime-based timer for the most volatile major asset "
            "of the last decade. Buy blue, sell red, ride the cycle -- sounds like a plan.\n\n"
            "If it does not work -- if the bands are just a retrospective decoration on a "
            "random-looking path -- then anyone using it to time their Bitcoin purchases is "
            "navigating with a map that was drawn by looking backwards."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "The key question is: **were the bands known at the time, or were they fitted "
            "on the whole history?**\n\n"
            "The original chart is built by fitting the regression on *all* available data. "
            "That means the 2013 chart 'knows' that Bitcoin will hit \\$69,000 in 2021. "
            "Any signal it gives in 2013 is cheating.\n\n"
            "The honest test: **walk-forward refitting.** At each date, fit the regression "
            "using *only* the data available up to that date. Use that fit to produce a "
            "signal. Invest based on that signal the *next* day. Never look ahead.\n\n"
            "Two outcomes to compare:\n"
            "1. **In-sample** -- the standard chart, look-ahead bias included.\n"
            "2. **Walk-forward** -- the honest version, no future data."
        ),

        # ---- BEAT 4 -- TEARDOWN ----------------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "### First: the in-sample rainbow -- why it looks so good"
        ),
        code(
            "# Draw the in-sample (biased) rainbow on the real or synthetic tape.\n"
            "if HAVE_REAL:\n"
            "    df = data.fetch_btc()\n"
            "    df_is = st.fit_rainbow_insample(df)\n"
            "else:\n"
            "    df, _ = data.synthetic_btc(n_days=3000, signal_strength=1.0, seed=174)\n"
            "    df_is = st.fit_rainbow_insample(df)\n\n"
            "fig, ax = plt.subplots(figsize=(10.5, 5.5))\n"
            "ax.semilogy(df_is.index, df_is['close'], c='k', lw=1.2, zorder=5, label='BTC price')\n\n"
            "# Draw 9 coloured bands.\n"
            "BAND_COLOURS = ['#0d3b8c','#1565c0','#1976d2','#4caf50',\n"
            "                '#fdd835','#ff9800','#e64a19','#b71c1c','#4a0000']\n"
            "sigmas = df_is['sigma_insample'].iloc[0]\n"
            "trend = df_is['trend_insample']\n"
            "thresholds = [-4, -3, -2, -1, 0, 1, 2, 3, 4]\n"
            "for i, (lo, hi) in enumerate(zip(thresholds[:-1], thresholds[1:])):\n"
            "    lower = np.exp(trend + lo * sigmas)\n"
            "    upper = np.exp(trend + hi * sigmas)\n"
            "    ax.fill_between(df_is.index, lower, upper,\n"
            "                    color=BAND_COLOURS[i], alpha=0.35)\n\n"
            "ax.set_title('Bitcoin Rainbow Chart -- IN-SAMPLE fit (look-ahead bias)', fontsize=13)\n"
            "ax.set_ylabel('BTC price (log scale)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('This looks prescient. Every top is in the red, every bottom in the blue.')\n"
            "print('But the bands were drawn knowing where the price would go.')"
        ),
        md(
            "The chart looks spectacular. Every major top sits in the orange-red bands; every "
            "major bottom sits in the blue bands. The 2021 peak almost touches the top band. "
            "This chart is catnip for a confirmation bias.\n\n"
            "**Now for the reveal: what did it actually look like in real time?**"
        ),
        code(
            "# Side-by-side: in-sample vs walk-forward fit at an intermediate date.\n"
            "if HAVE_REAL:\n"
            "    df_wf = st.fit_rainbow_walkforward(df)\n"
            "else:\n"
            "    df_wf = st.fit_rainbow_walkforward(df_is)  # df_is has the same columns\n\n"
            "# Pick 2020-01-01 as the 'looking at the chart in real time' date.\n"
            "CUT = '2020-01-01'\n"
            "df_sub = df_is[df_is.index <= CUT]\n"
            "df_wf_sub = df_wf[df_wf.index <= CUT].dropna(subset=['trend_wf'])\n\n"
            "fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharey=True)\n"
            "for ax, label, trend_col, sig_col in [\n"
            "    (axes[0], 'IN-SAMPLE (cheating: uses 2020-2026 data)', 'trend_insample', 'sigma_insample'),\n"
            "    (axes[1], 'WALK-FORWARD (honest: only pre-2020 data)', 'trend_wf', 'sigma_wf'),\n"
            "]:\n"
            "    sub = df_sub if 'insample' in trend_col else df_wf_sub\n"
            "    ax.semilogy(sub.index, sub['close'], c='k', lw=1.2, zorder=5)\n"
            "    if sub[trend_col].notna().sum() > 0:\n"
            "        trend = sub[trend_col].dropna()\n"
            "        sig = sub[sig_col].dropna().iloc[0] if 'insample' in sig_col else sub[sig_col].dropna().iloc[-1]\n"
            "        for i, (lo, hi) in enumerate(zip([-4,-3,-2,-1,0,1,2,3], [-3,-2,-1,0,1,2,3,4])):\n"
            "            lower = np.exp(trend + lo * sig)\n"
            "            upper = np.exp(trend + hi * sig)\n"
            "            ax.fill_between(trend.index, lower, upper, color=BAND_COLOURS[i+1], alpha=0.35)\n"
            "    ax.set_title(label, fontsize=10)\n"
            "    ax.set_ylabel('BTC price (log scale)')\n"
            "plt.suptitle('The same chart, at the same date -- but one cheats', fontsize=12)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Left: the rainbow as you see it today (knows the future).')\n"
            "print('Right: the rainbow as you would have seen it at the time (does not).')"
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal -- None.** Walk-forward HAC t = "
            f"+{R['wf_t']:.2f} (threshold: |t| >= 2). The honest signal is "
            "statistically indistinguishable from random.\n"
            "- **Tradability -- Mirage.** Walk-forward earns "
            f"+{R['wf_ann']}%/yr vs buy-and-hold +{R['bh_ann']}%/yr. The "
            "strategy is flat 80% of the time and misses the bulk of Bitcoin's rise.\n"
            "- **Overfitting? -- Busted.** The look-ahead bonus is "
            f"+{R['delta_t']:.2f} t-points. Every point of the in-sample t = "
            f"+{R['is_t']:.2f} came from future data."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT? ----------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "Let's look at what the walk-forward strategy actually does vs simply holding."
        ),
        code(
            "# Cumulative return: walk-forward rainbow vs buy-and-hold.\n"
            "if HAVE_REAL:\n"
            "    df_wf_full = st.fit_rainbow_walkforward(df)\n"
            "    sig_wf = st.rainbow_signal(df_wf_full, 'band_sigma_wf')\n"
            "    bt_wf = st.backtest(df_wf_full, sig_wf, cost_bps=10.0)\n"
            "    cum_strat = bt_wf['cum_log_net'].apply(lambda x: (np.exp(x)-1)*100)\n"
            "    cum_bh = bt_wf['cum_log_bh'].cumsum().apply(lambda x: (np.exp(x)-1)*100)\n"
            "    times_flat = (bt_wf['position'] == 0).sum() / len(bt_wf) * 100\n"
            "else:\n"
            "    # Synthetic illustration using a BTC-like trending tape.\n"
            "    df_s, _ = data.synthetic_btc(n_days=3000, signal_strength=0.8, seed=174)\n"
            "    df_s_wf = st.fit_rainbow_walkforward(df_s)\n"
            "    sig_wf = st.rainbow_signal(df_s_wf, 'band_sigma_wf')\n"
            "    bt_wf = st.backtest(df_s_wf, sig_wf, cost_bps=10.0)\n"
            "    cum_strat = bt_wf['cum_log_net'].apply(lambda x: (np.exp(x)-1)*100)\n"
            "    cum_bh = bt_wf['cum_log_bh'].cumsum().apply(lambda x: (np.exp(x)-1)*100)\n"
            "    times_flat = (bt_wf['position'] == 0).sum() / len(bt_wf) * 100\n\n"
            "fig, ax = plt.subplots(figsize=(10.5, 5.0))\n"
            "ax.plot(cum_bh.index, cum_bh, c=GREEN, lw=2, label='Buy-and-hold Bitcoin')\n"
            "ax.plot(cum_strat.index, cum_strat, c=RED, lw=1.5, label='Rainbow strategy (WF, 10bp cost)')\n"
            "ax.axhline(0, c='k', lw=0.8)\n"
            "ax.set_ylabel('Cumulative return (%)')\n"
            "ax.set_title('Rainbow walk-forward vs buy-and-hold')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Strategy is flat {times_flat:.0f}% of the time -- missing the uptrend.')\n"
            f"print('Walk-forward ann: +{R['wf_ann']}%/yr vs BH: +{R['bh_ann']}%/yr')"
        ),
        md(
            f"The walk-forward strategy earns **+{R['wf_ann']}%/yr** (gross) vs "
            f"**+{R['bh_ann']}%/yr** for buy-and-hold. It is flat **~80% of the time**, "
            "waiting in the cold bands that never arrive as expected once the look-ahead "
            "is removed. The rainbow strategy is a way to *miss* Bitcoin's best days while "
            "feeling principled about your valuation model."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The same family of curves, different dressing.** The Pi Cycle Top indicator "
            "([Study 117 -- Pi-Cycle-Top](../../117-pi-cycle-top/)) and the Stock-to-Flow model "
            "([Study 84 -- Moon-Math](../../84-moon-math/)) are close relatives -- "
            "all log-time or log-scarcity regressions of BTC price, with the same spurious-"
            "regression diagnosis.\n"
            "- **The halving cycle narrative** ([Study 81 -- Four-Year-Itch](../../81-four-year-itch/)) "
            "is a discrete version of the same idea -- but with n = 4 halvings, "
            "inference is hopeless.\n"
            "- **Does any log-time model work for BTC?** If log(time) genuinely drove BTC price "
            "causally, then out-of-sample forecasts would hit. They haven't: the 2022 crash to "
            "\\$16k was nowhere in the model.\n\n"
            "*Think the bands are real? Fork this, show a walk-forward t >= 2 that beats "
            "buy-and-hold at any cost, and cite a mechanism. That's the bar.*"
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
            "# Bitcoin-Rainbow -- a quantitative teardown\n"
            "### Log-time regression * walk-forward refitting * look-ahead anatomy * HAC inference\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Overfitting%3F: Busted](https://img.shields.io/badge/Overfitting%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "Same seven beats, every claim now carrying its standard error. We test whether the "
            "Bitcoin Rainbow Chart's buy/sell signal (buy below -1sigma, sell above +1.5sigma of "
            "the log-time regression) beats buy-and-hold on a walk-forward basis -- and dissect "
            "exactly how much of the in-sample t-stat is look-ahead.\n\n"
            "> **Not investment advice.** Real data: Yahoo Finance BTC-USD daily bars, "
            f"as-of {R['end']}; offline core and tests run on a deterministic synthetic tape. "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n\n"
            "> The `In plain words` notes translate each result back to intuition."
        ),
        code(BOOT + "\nfrom quantlab import analytics, stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Walk-forward HAC t = **+{R['wf_t']:.2f}** "
            f"(in-sample t = +{R['is_t']:.2f} -- all look-ahead). |\n"
            f"| **Tradability** | `MIRAGE` | WF ann. return **+{R['wf_ann']}%** vs "
            f"BH **+{R['bh_ann']}%**; strategy flat 80% of the time. |\n"
            f"| **Overfitting?** | `BUSTED` | Look-ahead bonus = **+{R['delta_t']:.2f} t-points**. "
            "The in-sample Sharpe of +1.16 is a pure fitting artefact. |\n\n"
            "> In plain words: the rainbow bands describe the past; they do not predict the future. "
            "Once you withold future data, the strategy is indistinguishable from random noise -- "
            "while simply holding Bitcoin returns +34%/yr over the same period."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $x_t = \\log(\\text{days since 2009-01-03})$ and $y_t = \\log(\\text{BTC price})$. "
            "The Rainbow Chart asserts:\n\n"
            "- **H1 (signal).** The standardised residual "
            "$z_t = (y_t - \\hat{\\alpha} - \\hat{\\beta} x_t) / \\hat{\\sigma}$ "
            "carries directional information: low $z_t$ predicts positive future returns "
            "(mean-reversion to the trend), high $z_t$ predicts negative returns.\n"
            "- **H2 (tradable).** Acting on $z_t$ beats buy-and-hold net of transaction costs.\n\n"
            "The confound: $\\hat{\\alpha}, \\hat{\\beta}, \\hat{\\sigma}$ are computed on the "
            "*full sample* in the standard chart, so they embed future prices. The honest test "
            "estimates them on an expanding window of *past data only*."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "The rainbow chart has been shared millions of times and used to justify specific "
            "Bitcoin purchases ('it's in the fire sale zone -- buying more'). If H1-H2 fail "
            "on a look-ahead-free basis, the chart is an expensive aesthetic: it makes no "
            "forward-looking predictions, only a persuasive retrospective fit. The question "
            "is not whether the chart is pretty but whether it is *useful*."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Signal formation.** At each date $t$, fit OLS: "
            "$y_{\\tau} = \\alpha + \\beta x_{\\tau} + \\varepsilon$ for all "
            "$\\tau \\leq t$ (expanding window, minimum 252 obs).\n"
            "- **Band sigma.** $z_t = (y_t - \\hat{\\alpha}_t - \\hat{\\beta}_t x_t) "
            "/ \\hat{\\sigma}_t$, known at the close of $t$.\n"
            f"- **Signal.** Long (+1) when $z_t < {R['buy_sigma']:.1f}\\sigma$; "
            f"flat (0) when ${R['buy_sigma']:.1f} \\leq z_t < {R['sell_sigma']:.1f}$.\n"
            "- **Execution.** Position taken at the open of $t+1$ (no look-ahead).\n"
            "- **Baseline.** Buy-and-hold from the first available bar.\n"
            "- **Inference.** HAC Newey-West t-stat on per-day net returns.\n"
            "- **Positive control.** Synthetic tape with tunable signal_strength; "
            "the engine should recover the signal when one is planted."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),

        md(
            "### 4a - The look-ahead dissection: IS vs walk-forward, side by side\n\n"
            "The same strategy, same data, same signal thresholds. One version is fitted on the "
            "whole history; the other re-fits incrementally."
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = data.fetch_btc()\n"
            "    result = st.insample_vs_walkforward_comparison(df)\n"
            "    s_is, s_wf = result['insample'], result['walkforward']\n"
            "    bt_is, bt_wf = result['bt_is'], result['bt_wf']\n"
            "else:\n"
            "    # Fall back to frozen numbers.\n"
            f"    s_is = dict(ann_ret_pct={R['is_ann']}, sharpe={R['is_sharpe']}, tstat={R['is_t']}, "
            f"time_in_market_pct={R['is_time']}, n_days={R['n']}, ann_bh_pct={R['bh_ann']})\n"
            f"    s_wf = dict(ann_ret_pct={R['wf_ann']}, sharpe={R['wf_sharpe']}, tstat={R['wf_t']}, "
            f"time_in_market_pct={R['wf_time']}, n_days={R['n']}, ann_bh_pct={R['bh_ann']})\n"
            "    bt_is = bt_wf = None\n\n"
            "rows = [('In-sample (look-ahead)', s_is), ('Walk-forward (honest)', s_wf)]\n"
            "tbl = pd.DataFrame([\n"
            "    (lbl, s['ann_ret_pct'], s['sharpe'], s['tstat'], s['time_in_market_pct'])\n"
            "    for lbl, s in rows], columns=['version', 'ann_ret_%', 'sharpe', 'HAC_t', 'time_in_mkt_%'])\n"
            "print(tbl.to_string(index=False))\n"
            f"print(f\"\\nBuy-and-hold: ann = +{R['bh_ann']}%/yr\")\n"
            f"print(f\"Look-ahead bonus: +{R['delta_t']:.2f} t-points\")"
        ),
        md(
            f"> In plain words: the t-stat drops from **+{R['is_t']:.2f}** to **+{R['wf_t']:.2f}** "
            "when we stop using future data. The difference -- the look-ahead bonus -- is "
            f"**+{R['delta_t']:.2f} t-points**. The entire statistical case for the rainbow chart "
            "is contained in those +4 t-points of cheating."
        ),

        md(
            "### 4b - The band sigma distribution -- why the signal is so sparse\n\n"
            "In-sample, the bands are calibrated to the full history so BTC rarely exits the "
            "central range. Walk-forward, the early bands are uncertain and wider."
        ),
        code(
            "if HAVE_REAL and bt_is is not None:\n"
            "    df_is = result['df_is']\n"
            "    df_wf = result['df_wf']\n"
            "    bs_is = df_is['band_sigma_insample'].dropna()\n"
            "    bs_wf = df_wf['band_sigma_wf'].dropna()\n"
            "else:\n"
            "    # Synthetic illustration.\n"
            "    df_s, _ = data.synthetic_btc(n_days=3000, signal_strength=1.0, seed=174)\n"
            "    df_is = st.fit_rainbow_insample(df_s)\n"
            "    df_wf = st.fit_rainbow_walkforward(df_s)\n"
            "    bs_is = df_is['band_sigma_insample'].dropna()\n"
            "    bs_wf = df_wf['band_sigma_wf'].dropna()\n\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "for ax, bs, lbl, col in [\n"
            "    (axes[0], bs_is, 'In-sample (biased)', AMBER),\n"
            "    (axes[1], bs_wf, 'Walk-forward (honest)', GREEN),\n"
            "]:\n"
            "    ax.hist(bs, bins=40, color=col, alpha=0.8)\n"
            f"    ax.axvline({R['buy_sigma']:.1f}, ls='--', c=GREEN, lw=1.5, label='buy threshold')\n"
            f"    ax.axvline({R['sell_sigma']:.1f}, ls='--', c=RED, lw=1.5, label='sell threshold')\n"
            "    ax.set_xlabel('band sigma (z-score)')\n"
            "    ax.set_title(f'{lbl}: mean={bs.mean():+.2f} std={bs.std():.2f}')\n"
            "    ax.legend()\n"
            "plt.suptitle('Band sigma distribution: IS vs walk-forward', fontsize=12)\n"
            "plt.tight_layout(); plt.show()\n"
            f"pct_long_is = (bs_is < {R['buy_sigma']:.1f}).mean() * 100\n"
            f"pct_long_wf = (bs_wf < {R['buy_sigma']:.1f}).mean() * 100\n"
            "print(f'Fraction below buy threshold: IS={pct_long_is:.1f}%  WF={pct_long_wf:.1f}%')"
        ),
        md(
            "In-sample, the regression is calibrated to the full history so the residuals are "
            "mean-zero and std=1 by construction -- and BTC rarely goes below -1σ (the buy "
            "threshold). Walk-forward, the bands drift and the strategy spends more time in the "
            "buy zone, but without any real predictive power.\n\n"
            "> In plain words: the rainbow strategy is mostly *flat*. It waits for extremes that "
            "are defined by a regression that didn't exist in real time. When those extremes "
            "finally arrive in the walk-forward version, they carry no reliable signal."
        ),

        md(
            "### 4c - Positive control: synthetic tape with planted signal_strength\n\n"
            "The engine should find an edge when one genuinely exists. Here we sweep signal_strength "
            "from 0 (pure random walk) to 1 (price perfectly tracks the log-time trend) and show "
            "that the walk-forward signal only recovers an edge when mean-reversion is real."
        ),
        code(
            "strengths = [0.0, 0.25, 0.5, 0.75, 1.0]\n"
            "wf_tstats = []\n"
            "is_tstats = []\n"
            "for ss in strengths:\n"
            "    df_s, _ = data.synthetic_btc(n_days=2000, signal_strength=ss, seed=174)\n"
            "    df_s_is = st.fit_rainbow_insample(df_s)\n"
            "    sig_is = st.rainbow_signal(df_s_is, 'band_sigma_insample')\n"
            "    bt_is_s = st.backtest(df_s_is, sig_is, cost_bps=0.0)\n"
            "    s_is_s = st.summarize(bt_is_s)\n"
            "    is_tstats.append(s_is_s['tstat'])\n\n"
            "    df_s_wf = st.fit_rainbow_walkforward(df_s)\n"
            "    sig_wf = st.rainbow_signal(df_s_wf, 'band_sigma_wf')\n"
            "    bt_wf_s = st.backtest(df_s_wf, sig_wf, cost_bps=0.0)\n"
            "    s_wf_s = st.summarize(bt_wf_s)\n"
            "    wf_tstats.append(s_wf_s['tstat'])\n\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "ax.plot(strengths, is_tstats, 'o--', c=AMBER, lw=2, label='In-sample (biased)')\n"
            "ax.plot(strengths, wf_tstats, 's-', c=GREEN, lw=2, label='Walk-forward (honest)')\n"
            "ax.axhline(2, ls=':', c=GREY, lw=1.5, label='|t|=2 bar')\n"
            "ax.axhline(-2, ls=':', c=GREY, lw=1.5)\n"
            "ax.axhline(0, c='k', lw=0.8)\n"
            "ax.set_xlabel('signal_strength (0=random walk, 1=price tracks log-time trend)')\n"
            "ax.set_ylabel('HAC t-stat (mean daily return)')\n"
            "ax.set_title('Positive control: walk-forward finds signal only when one genuinely exists')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('The engine works. The real tape has signal_strength ~ 0.')"
        ),
        md(
            "> In plain words: the machine is not broken -- it finds signal when signal exists. "
            "The real BTC tape gives a walk-forward t near zero, consistent with signal_strength ~ 0: "
            "Bitcoin's log-time trend provides no reliable mean-reversion signal for the walk-forward "
            "strategy, even if the in-sample fit appears excellent."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal `NONE`** -- walk-forward HAC t = +{R['wf_t']:.2f} (<< 2); "
            f"in-sample t = +{R['is_t']:.2f} is +{R['delta_t']:.2f} points of look-ahead bias.\n"
            f"- **Tradability `MIRAGE`** -- walk-forward +{R['wf_ann']}%/yr vs BH +{R['bh_ann']}%/yr; "
            "strategy flat 80% of the time; t-stat never clears 2 at any cost level.\n"
            "- **Overfitting? `BUSTED`** -- the rainbow chart is a textbook in-sample curve-fit; "
            "the bands are defined by the very prices they appear to predict."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- cost sweep\n\n"
            "Walk-forward strategy net return at various cost levels."
        ),
        code(
            "costs = [0, 10, 50, 100]\n"
            "if HAVE_REAL and bt_wf is not None:\n"
            "    df_wf_full = result['df_wf']\n"
            "    sig_wf_full = result['sig_wf']\n"
            "    net_anns = []\n"
            "    net_ts = []\n"
            "    for c in costs:\n"
            "        bt = st.backtest(df_wf_full, sig_wf_full, cost_bps=c)\n"
            "        s = st.summarize(bt)\n"
            "        net_anns.append(s['ann_ret_pct'])\n"
            "        net_ts.append(s['tstat'])\n"
            "else:\n"
            f"    net_anns = [{R['wf_ann']}, {R['wf_ann_c10']}, {R['wf_ann_c50']}, {R['wf_ann_c100']}]\n"
            f"    net_ts = [{R['wf_t']}, {R['wf_t_c10']}, {R['wf_t_c50']}, {R['wf_t_c100']}]\n\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "ax.plot(costs, net_anns, 'o-', c=RED, lw=2, label='WF ann. return (%)')\n"
            + f"ax.axhline({R['bh_ann']}, ls='--', c=GREEN, lw=1.5, label='Buy-and-hold ({{bh}}%/yr)'.format(bh={R['bh_ann']}))\n"
            + "ax.axhline(0, c='k', lw=0.8)\n"
            "ax.set_xlabel('round-trip cost (bps)')\n"
            "ax.set_ylabel('annualised return (%)')\n"
            "ax.set_title('Walk-forward return vs cost: buy-and-hold dominates throughout')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for c, ann, t in zip(costs, net_anns, net_ts):\n"
            "    print(f'cost={c:4d} bps: ann={ann:+.1f}%  HAC t={t:+.2f}')"
        ),
        md(
            f"> At zero cost the walk-forward strategy earns +{R['wf_ann']}%/yr. "
            f"Buy-and-hold earns +{R['bh_ann']}%/yr. **Costs make a bad strategy worse**, but "
            "the primary problem is not cost -- it is that the strategy spends 80% of the time "
            "flat, missing the asset's dominant trend. The rainbow turns Bitcoin's bull market "
            "into a series of carefully timed holidays from the trade."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **Same family, different clothing.** The BTC Stock-to-Flow model "
            "([Study 84 -- Moon-Math](../../84-moon-math/)) and the Pi Cycle Top indicator "
            "([Study 117 -- Pi-Cycle-Top](../../117-pi-cycle-top/)) are log-time/log-S2F "
            "regressions of BTC price with identical spurious-regression issues -- studied in "
            "full there.\n"
            "- **Does *any* valuation band help?** If instead of the log-time regression you "
            "used a simpler N-year lookback rolling mean with a fixed-width band (no OLS), the "
            "look-ahead bias shrinks but the signal problem remains: BTC does not mean-revert "
            "reliably to any anchor in the short run.\n"
            "- **Why does BTC trend so strongly?** The halving cycle narrative "
            "([Study 81 -- Four-Year-Itch](../../81-four-year-itch/)) tries to explain it; but "
            "with n = 4 halvings the inference is hopeless.\n\n"
            "*Convinced the rainbow works? Fork this, implement a walk-forward version that "
            "clears t >= 2 net of costs versus buy-and-hold BTC, and cite a mechanism that "
            "explains the mean-reversion. That is the bar.*"
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
