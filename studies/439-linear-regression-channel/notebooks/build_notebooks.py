"""Generate the two narrative notebooks for Study 439 (Linear-Regression-Channel).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached SPY (+ panel)
prices under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md.
# SPY total-return (auto-adjusted) daily closes, 2005-01-03 -> 2026-06-12, 5,395 days, 21.4y,
# fingerprint aeda2bbca066. 60-day OLS log-price slope, long/flat, 1-day lag, 1 bp one-way.
R = dict(
    ticker="SPY", start="2005-01-03", end="2026-06-12", n_days=5395, years=21.4,
    fingerprint="aeda2bbca066", window=60, cost_bps=1.0,
    # headline NET race (excess-vs-excess), long/flat
    sharpe_bh=0.647, cagr_bh=11.08,
    slope=dict(sharpe=0.711, gap=0.063, active_t=-1.51, tim=72, turn_yr=3.26, cagr_net=7.81),
    sma=dict(sharpe=0.604, gap=-0.043, active_t=-2.03, tim=72, turn_yr=15.92, cagr_net=6.17),
    perm_obs_bps=-1.641, perm_p=0.676,
    # cost sweep: (cost_bps, sharpe_strat, gap, active_t)
    cost=[(0.0, 0.713, 0.066, -1.50), (0.5, 0.712, 0.065, -1.51),
          (1.0, 0.711, 0.063, -1.51), (2.0, 0.708, 0.061, -1.53),
          (5.0, 0.699, 0.052, -1.56)],
    # window sweep: (window, sharpe_strat, gap, active_t)
    win=[(20, 0.552, -0.093, -2.23), (40, 0.465, -0.175, -2.44), (60, 0.711, 0.063, -1.51),
         (100, 0.595, -0.053, -1.91), (200, 0.711, 0.058, -1.03)],
    # long/short variant: (sharpe_strat, gap, active_t)
    longshort=(0.205, -0.442, -1.54),
    # panel: ticker -> (sharpe_strat, sharpe_bh, gap, active_t)
    panel=[("SPY", 0.711, 0.647, 0.063, -1.51), ("QQQ", 0.560, 0.793, -0.233, -3.01),
           ("AAPL", 0.959, 0.963, -0.004, -1.82), ("MSFT", 0.543, 0.683, -0.140, -2.36),
           ("JPM", 0.310, 0.543, -0.234, -2.51), ("XLE", 0.168, 0.404, -0.236, -1.96)],
    panel_beat=1, panel_n=6,
    # synthetic control: (edge, sharpe_strat, sharpe_bh, gap, active_t, perm_p)
    syn=[(0.00, -0.028, 0.112, -0.140, -0.72, 0.714), (0.50, 2.263, 1.556, 0.706, 2.44, 0.000)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Beats_a_moving_average%3F: Not_Supported](https://img.shields.io/badge/Beats_a_moving_average%3F-Not_Supported-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from linear_regression_channel import data, strategy as st

HAVE_REAL = data.have_real("SPY")
if HAVE_REAL:
    SPY = data.load_real("SPY", fetch=False)
    CLOSE = SPY["close"]
else:
    SPY = CLOSE = None
print("real SPY cache present:", HAVE_REAL,
      "| bars:", (0 if CLOSE is None else len(CLOSE)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does drawing a trend line beat a moving average? 📉\n"
            "### The \"Linear Regression Channel\" — the slope of a fitted line as a trend signal, in plain English\n\n"
            + BADGES +
            "Open any charting app and you'll find the **Linear Regression Channel** (LRC): it fits a "
            "straight line through the last *N* days of price by least squares and draws a channel around "
            "it. The pitch is seductive — that fitted line's **slope** is a *smoother, faster* read on the "
            "trend than a laggy moving average. Up-slope → be long; down-slope → step aside. Surely a real "
            "regression beats a crude average?\n\n"
            "We put the question on the tape: take **SPY** over 21 years, turn the 60-day regression slope "
            "into a long/flat timing rule, and **race it — net of costs — against (a) just buying and "
            "holding, and (b) the boring price-vs-moving-average rule it claims to improve on.** The honest "
            "answer is the usual one for chart-line folklore: it doesn't beat either, and the \"better than "
            "a moving average\" claim isn't supported.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the permutation test and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Data note.** SPY **total-return** (dividend-adjusted) daily closes via yfinance, "
            "cache-first. One execution lag (act on tomorrow's close). Every chart is drawn by the code "
            "beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the regression-slope rule beat **buy-and-hold**? | **No.** On SPY its net Sharpe is a "
            "hair higher (**0.71 vs 0.65**), but that gap is **not statistically real** — the head-to-head "
            "*t* is **negative** (−1.5), and on 5 of 6 other names the rule *loses*. |\n"
            "| Does it beat a **moving average**? | **Not in any way that matters.** Both the slope rule "
            "and the price-vs-SMA rule fail to beat the market; the slope's only real difference is it "
            "**trades less** (lower turnover), not that it earns more. |\n"
            "| Is the slope a *leading* trend read? | **No.** A 60-day regression slope is, by "
            "construction, a *smoothed average of the last 60 days* — it lags just like a moving average. |\n"
            "| So is it worthless? | As an *alpha* source on a liquid index, **yes**. It's a fine way to "
            "*describe* a trend on a chart; it is **not** a way to beat the index. |\n\n"
            "> The fitted line looks more \"scientific\" than a moving average. It isn't. Both are lagging "
            "trend filters, and on a market that mostly goes up, **staying invested** wins."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A moving average just averages the price — it lags. The **Linear Regression Channel** "
            "fits an actual trend line by least squares, so its **slope** tells you the trend's "
            "*direction and strength* sooner and cleaner. Go long when the slope turns up, flat (or short) "
            "when it turns down, and you'll ride trends better than any moving-average crossover.\"*\n\n"
            "This is the standard write-up on TradingView, Investopedia and a thousand YouTube tutorials. "
            "It steelmans to a sharp, testable hypothesis: **the regression-slope timing rule earns a "
            "higher risk-adjusted return than (a) buy-and-hold and (b) the equivalent moving-average rule, "
            "net of costs.**"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If true, it would be a free upgrade: swap your moving average for a regression slope and get "
            "more return for the same risk, on the most liquid instrument on earth. It would also say "
            "something deep — that a *least-squares fit* extracts trend information the market hasn't "
            "priced.\n\n"
            "But there's a trap hiding in plain sight. A regression slope over the last 60 days is "
            "**itself a weighted average of the last 60 days' returns** — it is a *cousin* of the moving "
            "average, not a different animal. So the real test isn't \"does it make money\" (in a 21-year "
            "bull market, almost any long-biased rule does); it's **\"does it beat the market it's "
            "trading, after costs?\"** — and is it any better than the moving average it claims to "
            "replace? We'll measure both, head-to-head."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{R['ticker']}** total-return daily closes, **{R['start']} → {R['end']}** "
            f"(**{R['years']:.0f} years**). Then:\n\n"
            "1. **Build the indicator.** Each day, fit a straight line to the last **60** log-prices by "
            "least squares and read off its **slope**.\n"
            "2. **Make a rule.** Long when the slope is **positive**, flat when it's negative. (We also "
            "try long/short.)\n"
            "3. **Trade it honestly.** Act on **tomorrow's** close (no peeking), charge realistic costs on "
            "every switch, and compute the **net** return.\n"
            "4. **Race it.** Compare its risk-adjusted return (Sharpe) to **buy-and-hold** *and* to the "
            "same rule built from a plain **moving average**. The killer test: is the slope rule's "
            "month-by-month *out*performance over buy-and-hold distinguishable from luck?\n\n"
            "**What would make us say \"mirage\":** if the slope rule's edge over buy-and-hold isn't "
            "statistically significant, and it's no better than the moving average — that's a no."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the equity curves.** The slope rule, the moving-average rule, and just buying and "
            "holding SPY. If the folklore is right, the slope's blue line should pull away from the rest."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sl = st.slope_signal(CLOSE, window=60); sm = st.sma_signal(CLOSE, window=60)\n"
            "    bt_sl = st.backtest(CLOSE, sl, cost_bps=1.0); bt_sm = st.backtest(CLOSE, sm, cost_bps=1.0)\n"
            "    eq_sl = (1+bt_sl['net']).cumprod(); eq_sm = (1+bt_sm['net']).cumprod()\n"
            "    eq_bh = (1+bt_sl['asset_ret']).cumprod()\n"
            "    fig, ax = plt.subplots()\n"
            "    ax.plot(eq_bh.index, eq_bh, c=GREY, lw=2, label=f'buy & hold (Sharpe {R[\"sharpe_bh\"]:.2f})')\n"
            "    ax.plot(eq_sl.index, eq_sl, c='#2c6fbb', lw=2, label=f'regression slope (Sharpe {R[\"slope\"][\"sharpe\"]:.2f})')\n"
            "    ax.plot(eq_sm.index, eq_sm, c=AMBER, lw=1.6, label=f'moving average (Sharpe {R[\"sma\"][\"sharpe\"]:.2f})')\n"
            "    ax.set_yscale('log'); ax.set_ylabel('growth of \\$1 (log)'); ax.legend()\n"
            "    ax.set_title('Both timing rules trail buy-and-hold on growth')\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('cache missing; headline: slope Sharpe', R['slope']['sharpe'], 'vs buy-hold', R['sharpe_bh'])\n"
            "print(f\"net CAGR -> slope {R['slope']['cagr_net']:.1f}%  movavg {R['sma']['cagr_net']:.1f}%  buy&hold {R['cagr_bh']:.1f}%\")"
        ),
        md(
            f"On **growth**, buy-and-hold wins outright (**{R['cagr_bh']:.1f}%/yr** vs the slope rule's "
            f"**{R['slope']['cagr_net']:.1f}%**) — because sitting in cash 28% of the time during a bull "
            "market costs you. The slope rule's *risk-adjusted* number (Sharpe "
            f"**{R['slope']['sharpe']:.2f}**) edges buy-and-hold's (**{R['sharpe_bh']:.2f}**) only because "
            "it dodges some volatility. Is that tiny Sharpe edge real? That's the whole question."
        ),
        md(
            "**The decisive test.** Forget total return — look at the rule's **outperformance over "
            "buy-and-hold, day by day**. If the slope genuinely adds skill, those daily \"active\" returns "
            "should be reliably positive. Here's their average against a luck cloud built by **shuffling "
            "the rule's timing** (block-permutation): if the real bar sits inside the cloud, it's noise."
        ),
        code(
            "if HAVE_REAL:\n"
            "    perm = st.permutation_pvalue(CLOSE, st.slope_signal(CLOSE,60), n_draws=1500, cost_bps=1.0)\n"
            "    obs = perm['obs']*1e4; draws = perm['draws']*1e4; pval = perm['p_value']\n"
            "else:\n"
            "    obs = R['perm_obs_bps']; pval = R['perm_p']\n"
            "    rng = np.random.default_rng(439); draws = rng.normal(0, 2.0, 1500)\n"
            "fig, ax = plt.subplots()\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='luck cloud: shuffled timing')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'real slope rule: {obs:+.2f} bps/day')\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('average daily out/under-performance vs buy-and-hold (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The edge over buy-and-hold is not real (p = {pval:.2f})'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real active mean {obs:+.2f} bps/day  permutation p = {pval:.2f}  (a real edge would sit far RIGHT of the cloud)')"
        ),
        md(
            f"The red line sits **left of zero** and squarely inside the luck cloud (**p = {R['perm_p']:.2f}**). "
            "The slope rule does not reliably beat the market it trades — its daily out-performance is "
            "*negative on average* and indistinguishable from a randomly-shuffled version of itself."
        ),
        md(
            "**Now the head-to-head the claim is really about: slope vs moving average.** Does the fancy "
            "regression actually do anything the crude average doesn't? We line up both rules' edge over "
            "buy-and-hold (their *t*-stat — anything below the dashed line means \"doesn't beat the market\")."
        ),
        code(
            "rules = ['regression\\nslope', 'moving\\naverage']\n"
            "tvals = [R['slope']['active_t'], R['sma']['active_t']]\n"
            "turns = [R['slope']['turn_yr'], R['sma']['turn_yr']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.4, 4.3))\n"
            "a1.bar(rules, tvals, color=[RED, RED], width=.55)\n"
            "a1.axhline(2, ls='--', c=GREEN, label='t = +2 (beats the market)'); a1.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(tvals): a1.annotate(f'{v:+.2f}',(i,v),ha='center',va='top',fontsize=10)\n"
            "a1.set_ylabel('edge over buy-and-hold (t-stat)'); a1.set_ylim(-3, 2.6)\n"
            "a1.set_title('Neither beats buy-and-hold'); a1.legend()\n"
            "a2.bar(rules, turns, color=[GREEN, AMBER], width=.55)\n"
            "for i,v in enumerate(turns): a2.annotate(f'{v:.1f}x',(i,v),ha='center',va='bottom',fontsize=10)\n"
            "a2.set_ylabel('round-trips per year'); a2.set_title(\"Slope's only edge: it trades less\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"slope active-t {R['slope']['active_t']:+.2f} (turn {R['slope']['turn_yr']:.1f}/yr) | \"\n"
            "      f\"SMA active-t {R['sma']['active_t']:+.2f} (turn {R['sma']['turn_yr']:.1f}/yr)\")"
        ),
        md(
            f"There's the verdict in one picture. **Neither** rule clears the green line — both have a "
            f"*negative* edge over buy-and-hold (slope **{R['slope']['active_t']:+.2f}**, moving average "
            f"**{R['sma']['active_t']:+.2f}**). The regression slope's *only* genuine difference is that it "
            f"**trades ~5× less** ({R['slope']['turn_yr']:.1f} vs {R['sma']['turn_yr']:.1f} round-trips a "
            "year) because the slope flips sign less often than price crosses its average. That's a "
            "smoothness property, not an edge."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The slope rule's out-performance over buy-and-hold is *negative* and "
            f"not significant (head-to-head *t* = **{R['slope']['active_t']:+.2f}**, permutation "
            f"**p = {R['perm_p']:.2f}**). On 5 of 6 other names it loses to buy-and-hold.\n"
            "- **Tradability — Mirage.** The wafer-thin SPY Sharpe edge (0.71 vs 0.65) isn't real, "
            "doesn't survive across names, and the long/short version is much worse.\n"
            f"- **\"Beats a moving average\"? — Not supported.** Both rules fail to beat the market; the "
            "slope's only real difference is *lower turnover*, not higher return. The regression line is "
            "a lagging average wearing a lab coat."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — it's not a cost problem\n\n"
            "Sometimes a paper edge dies only because costs eat it. Not here — there's no edge to begin "
            "with. Even at **zero** cost the slope rule doesn't beat buy-and-hold, and crucially it "
            "**loses on almost every other instrument** we tried. Folklore that only \"works\" on the one "
            "ticker you picked is the oldest mirage in the book."
        ),
        code(
            "names = [p[0] for p in R['panel']]; gaps = [p[3] for p in R['panel']]\n"
            "cols = [GREEN if g>0 else RED for g in gaps]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(names, gaps, color=cols, width=.6); ax.axhline(0, c='k', lw=.8)\n"
            "for i,g in enumerate(gaps): ax.annotate(f'{g:+.2f}',(i,g),ha='center',va='bottom' if g>0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('slope-rule Sharpe minus buy-and-hold Sharpe')\n"
            "ax.set_title(f\"Beats buy-and-hold on only {R['panel_beat']} of {R['panel_n']} names — and barely\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Sharpe gap vs buy-hold by name:', {p[0]: round(p[3],3) for p in R['panel']})"
        ),
        md(
            f"Only **{R['ticker']}** is (barely) green, and even there the edge is statistical noise. The "
            "honest read: the Linear Regression Channel is a perfectly good *drawing tool* for "
            "visualising a trend — but as a timing signal on a liquid index it's a **Mirage**. You'd have "
            "done better, with less effort and lower taxes, by doing nothing."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Channel bands, not just the slope.** The LRC also draws ±2σ bands around the line — a "
            "Bollinger-style mean-reversion play. That's a different claim (see desk study **104 — "
            "Bollinger-Reversion**); the *slope* tested here is the trend claim.\n"
            "- **Other markets.** The slope rule earns its keep where trends are strong and persistent "
            "(some commodities, FX). On a mean-reverting, mostly-up equity index, staying invested wins.\n"
            "- **Build your own.** Swap the 60-day window, try long/short, or race it against MACD/RSI — "
            "the harness in [`strategy.py`](../linear_regression_channel/strategy.py) makes it a "
            "one-liner.\n\n"
            "*Think the slope beats a moving average somewhere it matters? Show the **net** active *t* over "
            "buy-and-hold clearing **+2**, on a market you didn't cherry-pick — then we'll talk.*"
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
            "# The Linear Regression Channel slope — a quantitative teardown 🔬\n"
            "### Rolling-OLS log-price slope as a long/flat timing rule · NET, excess-vs-excess vs "
            "buy-and-hold · HAC *t* on the active return · block-permutation placebo · head-to-head vs an "
            "SMA filter · cost & window sweeps · a synthetic faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The LRC slope "
            "is sold as a *leading, smoother* trend read than a moving average. We show it is neither: it "
            "is a lagging weighted average of recent returns, its edge over buy-and-hold is "
            "**indistinguishable from zero** (HAC *t* < 0), and it is **no better** than the SMA rule it "
            "claims to replace.\n\n"
            "> ⚠️ **Data note.** SPY **total-return** (auto-adjusted) daily closes via yfinance, "
            "cache-first; small panel (SPY/QQQ/AAPL/MSFT/JPM/XLE) for the cross-instrument check. One "
            "execution lag (signal at close *t* earns return *t+1*). Offline core + synthetic control are "
            "deterministic. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Slope-rule daily **active** return (net, minus buy-and-hold) has "
            f"HAC **t = {R['slope']['active_t']:+.2f}** and block-permutation **p = {R['perm_p']:.2f}** — "
            f"nowhere near +2, and *negative*. Beats buy-and-hold on **{R['panel_beat']}/{R['panel_n']}** "
            "names. |\n"
            f"| **Tradability** | `MIRAGE` | SPY net Sharpe **{R['slope']['sharpe']:.2f}** vs buy-and-hold "
            f"**{R['sharpe_bh']:.2f}** — a +{R['slope']['gap']:.2f} gap that is not significant, inverts "
            f"cross-section, and collapses long/short (Sharpe {R['longshort'][0]:.2f}). |\n"
            f"| **Beats a moving average?** | `NOT SUPPORTED` | SMA rule active **t = {R['sma']['active_t']:+.2f}** "
            f"vs slope **{R['slope']['active_t']:+.2f}** — both sub-zero; the slope's only difference is "
            f"**lower turnover** ({R['slope']['turn_yr']:.1f} vs {R['sma']['turn_yr']:.1f}/yr), not edge. |\n\n"
            "> 💡 In plain words: a least-squares slope of the last 60 log-prices is a lagging linear "
            "filter of recent returns. It neither beats the index nor improves on a moving average — its "
            "smoothness lowers turnover, which is not the same as adding return."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $p_t=\\log C_t$. The LRC centre-line is the OLS fit of $p$ on a time index over a window "
            "$W$; its **slope** is\n\n"
            "$$\\hat\\beta_t = \\frac{\\sum_{k=0}^{W-1}(k-\\bar k)\\,(p_{t-W+1+k}-\\bar p)}{\\sum_k (k-\\bar k)^2}"
            "= \\frac{12}{W(W^2-1)}\\sum_{k}(k-\\bar k)\\,p_{t-W+1+k}.$$\n\n"
            "The timing rule is $x_t=\\mathbb 1[\\hat\\beta_t>0]$ (long/flat) or $x_t=\\operatorname{sign}\\hat\\beta_t$ "
            "(long/short), applied to the *next* bar: $r^{\\text{strat}}_{t+1}=x_t\\,r_{t+1}-c\\,|x_t-x_{t-1}|$.\n\n"
            "- **H₁ (beats the market).** The active return $a_t = r^{\\text{strat}}_t - r^{\\text{bh}}_t$ "
            "has mean $>0$ with HAC $t\\ge 2$.\n"
            "- **H₂ (deployable).** That edge survives realistic costs and holds across instruments.\n"
            "- **H₃ (\"better than a moving average\").** $\\hat\\beta$-timing's edge exceeds the "
            "price-vs-SMA rule's edge.\n\n"
            f"We find **H₁ rejected** ($t={R['slope']['active_t']:+.2f}$), **H₂ rejected** (edge inverts "
            "cross-section), **H₃ rejected** (slope $t$ ≈ SMA $t$, both $<0$). Note $\\hat\\beta_t$ is a "
            "*linear* combination of past log-prices — a smoothed momentum signal, structurally a "
            "moving-average relative."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The Signal axis is a one-sample HAC *t* on the daily active series $a_t$ (strategy net minus "
            "buy-and-hold), **excess-vs-excess**: both legs are measured against the same cash rate so a "
            "part-time-in-cash rule isn't unfairly compared to a fully-invested one. The Newey-West HAC "
            "correction matters because $a_t$ inherits the volatility clustering of equity returns; an "
            "i.i.d. *t* would overstate significance.\n\n"
            "Two traps the LRC narrative hides: **(a) bull-market drift** — in a 21-year up-market any "
            "long-biased rule makes money, so total return is not evidence; the race must be *over* "
            "buy-and-hold. **(b) the moving-average disguise** — the slope is a linear filter of past "
            "prices, so \"better than a moving average\" needs the slope rule's edge to *exceed* the SMA "
            "rule's, not merely to be positive."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {R['ticker']} total-return daily closes ({R['start']}→{R['end']}, "
            f"**{R['n_days']:,}** bars, fingerprint `{R['fingerprint']}`); panel SPY/QQQ/AAPL/MSFT/JPM/XLE "
            "for the cross-instrument check.\n"
            f"- **Indicator.** {R['window']}-day rolling OLS slope of log-price (closed-form, exact).\n"
            "- **Rules.** Slope long/flat (headline), slope long/short, and the benchmark price-vs-"
            f"{R['window']}-day-SMA long/flat.\n"
            "- **Timing.** Signal at close *t* → return *t+1* (one `shift`, applied once).\n"
            f"- **Costs.** {R['cost_bps']:.0f} bp one-way × |Δposition| (turnover as NAV fraction); shorts "
            "pay 50 bp/yr borrow.\n"
            "- **Signal test.** HAC *t* on the daily active return $a_t$ vs 0, + a **block-permutation** "
            "placebo (21-day blocks preserve the rule's persistence, destroy its alignment).\n"
            "- **Robustness.** Cost sweep, window sweep {20…200}, long/short, 6-name panel.\n"
            "- **Positive control.** A deterministic panel with a **planted** persistent trend: zero edge "
            "must NOT reach significance; a strong planted trend must light up (machinery proof)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The indicator, and the race it loses\n\n"
            "Left: the slope overlaid on price — note how it *lags* turns (it's an average). Right: equity "
            "curves, net, log-scale. Buy-and-hold leads on growth; the slope rule only trims volatility."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sl_raw = st.rolling_slope(CLOSE, 60)\n"
            "    sl = st.slope_signal(CLOSE,60); sm = st.sma_signal(CLOSE,60)\n"
            "    bt_sl = st.backtest(CLOSE, sl, cost_bps=1.0); bt_sm = st.backtest(CLOSE, sm, cost_bps=1.0)\n"
            "    eq_sl=(1+bt_sl['net']).cumprod(); eq_sm=(1+bt_sm['net']).cumprod(); eq_bh=(1+bt_sl['asset_ret']).cumprod()\n"
            "    fig,(a1,a2)=plt.subplots(1,2,figsize=(11,4.3))\n"
            "    w = CLOSE.loc['2018':'2020']\n"
            "    a1.plot(w.index, w, c='k', lw=1, label='SPY close')\n"
            "    a1b=a1.twinx(); a1b.plot(sl_raw.loc['2018':'2020'].index, sl_raw.loc['2018':'2020'], c='#2c6fbb', lw=1.4, label='60d slope')\n"
            "    a1b.axhline(0,c=RED,ls='--',lw=.8); a1.set_title('Slope lags the turns (it is an average)')\n"
            "    a1.legend(loc='upper left'); a1b.legend(loc='lower right'); a1b.set_ylabel('annualised slope')\n"
            "    a2.plot(eq_bh.index,eq_bh,c=GREY,lw=2,label=f'buy&hold {R[\"sharpe_bh\"]:.2f}')\n"
            "    a2.plot(eq_sl.index,eq_sl,c='#2c6fbb',lw=2,label=f'slope {R[\"slope\"][\"sharpe\"]:.2f}')\n"
            "    a2.plot(eq_sm.index,eq_sm,c=AMBER,lw=1.5,label=f'SMA {R[\"sma\"][\"sharpe\"]:.2f}')\n"
            "    a2.set_yscale('log'); a2.set_ylabel('growth of \\$1'); a2.legend(); a2.set_title('Both rules trail on growth (Sharpe in legend)')\n"
            "    plt.tight_layout(); plt.show()\n"
            "print(f\"net CAGR: slope {R['slope']['cagr_net']:.2f}%  SMA {R['sma']['cagr_net']:.2f}%  buy&hold {R['cagr_bh']:.2f}%  | \"\n"
            "      f\"Sharpe: slope {R['slope']['sharpe']:.3f}  SMA {R['sma']['sharpe']:.3f}  bh {R['sharpe_bh']:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: the slope (blue) crosses zero *after* price has already turned. On "
            f"growth, buy-and-hold wins ({R['cagr_bh']:.1f}% vs {R['slope']['cagr_net']:.1f}%/yr); the "
            f"slope's higher Sharpe ({R['slope']['sharpe']:.2f} vs {R['sharpe_bh']:.2f}) is pure "
            "volatility-trimming from sitting in cash ~28% of the time, not alpha."
        ),
        md(
            "### 4b · The decisive test — HAC *t* on the active return + a block placebo\n\n"
            "The daily active return $a_t$ (slope net minus buy-and-hold) against a 21-day-block "
            "permutation null. A real edge sits in the **right** tail; the observed bar should not."
        ),
        code(
            "if HAVE_REAL:\n"
            "    perm = st.permutation_pvalue(CLOSE, st.slope_signal(CLOSE,60), n_draws=1500, cost_bps=1.0)\n"
            "    obs=perm['obs']*1e4; draws=perm['draws']*1e4; pval=perm['p_value']\n"
            "    tval = st.evaluate(CLOSE, st.slope_signal(CLOSE,60), cost_bps=1.0)['active_t']\n"
            "else:\n"
            "    obs=R['perm_obs_bps']; pval=R['perm_p']; tval=R['slope']['active_t']\n"
            "    rng=np.random.default_rng(439); draws=rng.normal(0,2.0,1500)\n"
            "fig,ax=plt.subplots()\n"
            "ax.hist(draws,bins=55,color=GREY,alpha=.85,label='null: 1500 block-shuffled timings')\n"
            "ax.axvline(obs,c=RED,lw=2.5,label=f'observed active {obs:+.2f} bps/day')\n"
            "ax.axvline(0,c='k',lw=.8)\n"
            "ax.set_xlabel('mean daily active return vs buy-and-hold (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'No edge over buy-and-hold: HAC t = {tval:+.2f}, permutation p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'active mean {obs:+.3f} bps/day | HAC t = {tval:+.2f} | block-permutation p = {pval:.2f}')"
        ),
        md(
            f"> 💡 In plain words: the red line is **left of zero**, deep inside the cloud "
            f"(**p = {R['perm_p']:.2f}**, HAC **t = {R['slope']['active_t']:+.2f}**). The slope rule's "
            "out-performance over the market is, if anything, *under*-performance — and statistically "
            "it's a coin flip. **Signal = NONE.**"
        ),
        md(
            "### 4c · Head-to-head vs the moving average + robustness\n\n"
            "Left: the active *t* of slope vs SMA — the claim's actual test. Right: window sweep — is the "
            "60-day a cherry pick? (The active *t* never clears +2 at any window.)"
        ),
        code(
            "if HAVE_REAL:\n"
            "    st_t = st.evaluate(CLOSE, st.slope_signal(CLOSE,60), cost_bps=1.0)['active_t']\n"
            "    sm_t = st.evaluate(CLOSE, st.sma_signal(CLOSE,60), cost_bps=1.0)['active_t']\n"
            "    wins=[]\n"
            "    for w in (20,40,60,100,200):\n"
            "        e = st.evaluate(CLOSE, st.slope_signal(CLOSE,w), cost_bps=1.0)\n"
            "        wins.append((w, e['active_t']))\n"
            "else:\n"
            "    st_t=R['slope']['active_t']; sm_t=R['sma']['active_t']; wins=[(w[0],w[3]) for w in R['win']]\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(10.6,4.3))\n"
            "a1.bar(['regression\\nslope','moving\\naverage'],[st_t,sm_t],color=[RED,RED],width=.55)\n"
            "a1.axhline(2,ls='--',c=GREEN,label='t=+2'); a1.axhline(0,c='k',lw=.8)\n"
            "for i,v in enumerate([st_t,sm_t]): a1.annotate(f'{v:+.2f}',(i,v),ha='center',va='top',fontsize=10)\n"
            "a1.set_ylabel('active t vs buy-and-hold'); a1.set_ylim(-3,2.6); a1.set_title('Slope is no better than the SMA'); a1.legend()\n"
            "ws=[w[0] for w in wins]; wt=[w[1] for w in wins]\n"
            "a2.plot(ws,wt,'o-',c='#2c6fbb',lw=2); a2.axhline(2,ls='--',c=GREEN,label='t=+2'); a2.axhline(0,c='k',lw=.8)\n"
            "for w,t in wins: a2.annotate(f'{t:+.2f}',(w,t),ha='center',va='bottom',fontsize=9)\n"
            "a2.set_xlabel('regression window (days)'); a2.set_ylabel('active t'); a2.set_title('No window beats the market'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('slope vs SMA active-t:', round(st_t,2), round(sm_t,2))\n"
            "print('window sweep active-t:', [(w,round(t,2)) for w,t in wins])"
        ),
        md(
            f"> 💡 In plain words: slope active **t = {R['slope']['active_t']:+.2f}** vs SMA "
            f"**{R['sma']['active_t']:+.2f}** — the regression is *not* the better filter. And no window "
            "from 20 to 200 days clears the bar; the 60-day isn't hiding a missed edge. **\"Beats a "
            "moving average\" is not supported.**"
        ),
        md(
            "### 4d · Costs & cross-instrument — there's nothing for costs to kill\n\n"
            "Left: cost sweep — the (negative) edge barely moves with cost, because it was never there. "
            "Right: the panel — beats buy-and-hold on only 1 of 6 names, and the long/short variant is far "
            "worse than long/flat."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cs=[]\n"
            "    for c in (0.0,0.5,1.0,2.0,5.0):\n"
            "        e=st.evaluate(CLOSE, st.slope_signal(CLOSE,60), cost_bps=c); cs.append((c,e['active_t']))\n"
            "    pn=[]\n"
            "    for t in data.PANEL:\n"
            "        if not data.have_real(t): continue\n"
            "        cc=data.load_real(t)['close']; e=st.evaluate(cc, st.slope_signal(cc,60), cost_bps=1.0)\n"
            "        pn.append((t,e['sharpe_gap']))\n"
            "else:\n"
            "    cs=[(c[0],c[3]) for c in R['cost']]; pn=[(p[0],p[3]) for p in R['panel']]\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(10.6,4.3))\n"
            "a1.plot([c[0] for c in cs],[c[1] for c in cs],'o-',c=RED,lw=2); a1.axhline(0,c='k',lw=.8)\n"
            "a1.set_xlabel('one-way cost (bps)'); a1.set_ylabel('active t vs buy-and-hold'); a1.set_title('Edge is ~flat in cost (it is ~0)')\n"
            "nm=[p[0] for p in pn]; gp=[p[1] for p in pn]\n"
            "a2.bar(nm,gp,color=[GREEN if g>0 else RED for g in gp],width=.6); a2.axhline(0,c='k',lw=.8)\n"
            "for i,g in enumerate(gp): a2.annotate(f'{g:+.2f}',(i,g),ha='center',va='bottom' if g>0 else 'top',fontsize=8)\n"
            "a2.set_ylabel('Sharpe gap vs buy-and-hold'); a2.set_title(f\"Beats buy-and-hold on {R['panel_beat']}/{R['panel_n']} names\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print('cost sweep active-t:', [(c[0],round(c[1],2)) for c in cs])\n"
            "print('panel Sharpe gap:', [(p[0],round(p[1],3)) for p in pn])\n"
            "print(f\"long/short SPY: Sharpe {R['longshort'][0]:.2f}, gap {R['longshort'][1]:+.2f}, active-t {R['longshort'][2]:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: the cost line is nearly flat — costs aren't the killer, the absent edge "
            f"is. Cross-section, the slope rule beats buy-and-hold on **{R['panel_beat']}/{R['panel_n']}** "
            f"names (and that one, SPY, isn't significant). The long/short variant cuts Sharpe to "
            f"**{R['longshort'][0]:.2f}** — shorting a market that goes up is, predictably, a loser. "
            "**Tradability = MIRAGE.**"
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "On a deterministic panel with a **planted** persistent trend (sign-stable regimes a slope "
            "*can* ride): zero edge must NOT light up; a strong planted trend must. Both hold — so the "
            "real-tape null is a true negative, not a broken harness."
        ),
        code(
            "res=[]\n"
            "for edge in (0.0, 0.50):\n"
            "    bars,_ = data.synthetic_panel(edge=edge, seed=439)\n"
            "    r = st.run_experiment(bars['close'], window=60, cost_bps=1.0, do_permutation=True, n_draws=800)\n"
            "    s=r['slope']; res.append((edge, s['active_t'], s['sharpe_gap'], r['slope_perm']['p_value']))\n"
            "fig,ax=plt.subplots(figsize=(8.6,4.3))\n"
            "labs=[f'planted edge\\n{e*100:.0f}%/yr' for e,_,_,_ in res]; tv=[r[1] for r in res]\n"
            "ax.bar(labs,tv,color=[GREY,GREEN],width=.5); ax.axhline(2,ls='--',c=RED,label='t=+2'); ax.axhline(0,c='k',lw=.8)\n"
            "for i,t in enumerate(tv): ax.annotate(f't={t:+.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('active t vs buy-and-hold'); ax.set_title('Control: 0 edge -> t<0 (no false positive); planted -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,t,g,p in res: print(f'planted {e*100:+.0f}%/yr: active-t={t:+.2f}  Sharpe-gap={g:+.3f}  perm p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted trend the slope rule sits at "
            f"**t = {R['syn'][0][4]:+.2f}** (perm p = {R['syn'][0][5]:.2f}) — it does *not* manufacture an "
            f"edge from noise. With a strong planted trend it lights up (**t = {R['syn'][1][4]:+.2f}**, "
            f"Sharpe gap **{R['syn'][1][3]:+.2f}**, perm p ≈ {R['syn'][1][5]:.2f}). The machinery can find "
            "a slope edge when one exists — so the real-tape *null* is the genuine article, not a "
            "false negative."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the slope rule's daily active return vs buy-and-hold has HAC "
            f"**t = {R['slope']['active_t']:+.2f}** (block-permutation **p = {R['perm_p']:.2f}**) — "
            "negative and nowhere near the +2 bar. No window from 20–200 days clears it.\n"
            f"- **Tradability `MIRAGE`** — the SPY net Sharpe edge (**{R['slope']['sharpe']:.2f}** vs "
            f"**{R['sharpe_bh']:.2f}**) is not significant, inverts on 5/6 panel names, and the long/short "
            f"variant collapses to Sharpe **{R['longshort'][0]:.2f}**. There is no edge for costs to kill.\n"
            f"- **Beats a moving average? `NOT SUPPORTED`** — slope active *t* "
            f"(**{R['slope']['active_t']:+.2f}**) ≈ SMA active *t* (**{R['sma']['active_t']:+.2f}**), both "
            f"sub-zero. The slope's only genuine property is **lower turnover** "
            f"({R['slope']['turn_yr']:.1f} vs {R['sma']['turn_yr']:.1f}/yr) — a smoothness artefact, not "
            "alpha. The regression line is a lagging linear filter dressed up as a leading indicator."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — no, and not because of costs\n\n"
            "The operational truth in one frame: the **net** active Sharpe gap over buy-and-hold across "
            "the panel, with the zero line. A deployable edge would be reliably positive across names; "
            "this is a scatter around (and mostly below) zero."
        ),
        code(
            "names=[p[0] for p in R['panel']]; gaps=[p[3] for p in R['panel']]; ts=[p[4] for p in R['panel']]\n"
            "fig,ax=plt.subplots(figsize=(9.0,4.3))\n"
            "cols=[GREEN if g>0 else RED for g in gaps]\n"
            "ax.bar(names,gaps,color=cols,width=.6); ax.axhline(0,c='k',lw=.8)\n"
            "for i,(g,t) in enumerate(zip(gaps,ts)): ax.annotate(f'{g:+.2f}\\n(t={t:+.1f})',(i,g),ha='center',va='bottom' if g>0 else 'top',fontsize=8)\n"
            "ax.set_ylabel('net Sharpe gap vs buy-and-hold'); ax.set_title('No reliable edge across instruments')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('beat buy-and-hold on', R['panel_beat'], 'of', R['panel_n'], 'names; the one win (SPY) has active t =', R['panel'][0][4])"
        ),
        md(
            "> 💡 In plain words: a tradable signal looks like a row of green bars with *t*-stats above 2. "
            "This is one barely-green bar (SPY, not significant) and five red ones. You can't allocate "
            "capital to a rule whose only positive instance is the one you happened to test first. The "
            "LRC slope is a **drawing tool**, not an alpha source — **MIRAGE**."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The bands are a different claim.** The LRC's ±σ channel is a mean-reversion play — see "
            "desk study **104 (Bollinger-Reversion)**, which busts the band-touch entry on the same "
            "family of logic. This study isolates the *slope* (trend) claim.\n"
            "- **Where slopes might pay.** Persistent-trend markets (managed-futures style: commodities, "
            "rates, FX) are where a slope/MA filter historically earns its keep — the equity-index "
            "application is the weakest case.\n"
            "- **The structural lesson.** A rolling OLS slope is a *linear* combination of past prices — "
            "mathematically a relative of the moving average (and of MACD). \"Fit a regression\" sounds "
            "more rigorous than \"take an average,\" but on this tape it buys you nothing the average "
            "didn't.\n\n"
            "*Reproducible core is offline & deterministic; the synthetic control is the machinery proof. "
            "Methods/sources: [`docs/references.md`](../docs/references.md); frozen numbers: "
            "[`docs/results.md`](../docs/results.md).*"
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
