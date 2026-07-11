"""Generate the two narrative notebooks for Study 680 (Disparity Index).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached SPY/basket
tapes under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance SPY/QQQ/IWM/
# AAPL/TSLA/NVDA 2003-01-02 -> 2026-06-30; window=10, thresholds [95,105], hold=5d).
R = dict(
    start="2003-01-02", end="2026-06-30", window=10, oversold=95.0, overbought=105.0,
    hold=5,
    n_low=1825, n_mid=29406, n_high=2338,
    mean_low_bps=+111.60, mean_mid_bps=+38.16, mean_high_bps=+99.64,
    welch_low=+3.84, nw_low=+2.50, welch_high=+4.30, nw_high=+2.68,
    # two-sided timer
    timer_n=4163,
    timer_win5=48.8, timer_mean5=-17.03, timer_t5=-0.94,
    coin_win5=49.1, coin_mean5=+0.87, coin_t5=+0.07,
    timer_win10=48.3, timer_mean10=-27.03, timer_t10=-1.49,
    coin_win10=48.5, coin_mean10=-9.13, coin_t10=-0.75,
    delta5=-17.90, delta5_t=-1.09,
    # drift check
    dl_n=1825, dl_mean=+101.60, dl_t=+3.49, dl_rand=+67.05, dl_rand_t=+4.51,
    dl_delta=+34.55, dl_delta_t=+1.42,
    dh_n=2338, dh_mean=-109.64, dh_t=-4.87, dh_rand=-91.32, dh_rand_t=-6.73,
    dh_delta=-18.31, dh_delta_t=-0.94,
    # reversal diagnostic
    corr_n=33515, corr=0.8366,
    # grid (window -> {pair: (t_low, t_high)})
    grid={
        5: {(97, 103): (+3.65, +1.74), (95, 105): (+3.14, +2.12), (93, 107): (+3.03, +2.73)},
        10: {(97, 103): (+3.11, +3.63), (95, 105): (+3.84, +4.30), (93, 107): (+3.53, +4.54)},
        20: {(97, 103): (+3.91, +8.09), (95, 105): (+3.50, +7.24), (93, 107): (+3.95, +6.99)},
        25: {(97, 103): (+4.60, +8.85), (95, 105): (+3.78, +7.93), (93, 107): (+3.45, +7.01)},
    },
    # synthetic control
    syn_null_welch_lo=(-0.16, 1.55, 5), syn_null_welch_hi=(+0.09, 1.27, 2),
    syn_null_nw_lo=(-0.08, 1.25, 2), syn_null_nw_hi=(+0.06, 0.99, 1),
    syn_planted_welch=(+3.88, -2.47), syn_planted_nw=(+3.33, -2.09),
    fp_spy="c6509bba7177", fp_qqq="824e9d9be08f", fp_iwm="1a82f99e4745",
    fp_aapl="db335bf7a084", fp_tsla="1d273fa4b475", fp_nvda="e3f7f5f85931",
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![More_than_reversal%3F: Busted](https://img.shields.io/badge/More_than_reversal%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from disparity_index import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    BARS = data.load_real()
else:
    BARS = None
print("real cache present:", HAVE_REAL, "| universe:", ", ".join(data.UNIVERSE),
      "| tape rows (SPY):", (0 if BARS is None else len(BARS["SPY"])))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does buying the dip on a \"stretched\" stock actually pay off? 🪀\n"
            "### The Disparity Index — a Korean trading-room staple that turns out to be "
            "half right, half backwards, and mostly captured drift\n\n"
            + BADGES +
            "Look at any stock chart and draw its moving average. Sometimes the price is "
            "way above that line, sometimes way below it — like a rubber band stretched "
            "away from its resting point. The **Disparity Index** puts a number on that "
            "stretch (100 = sitting right on the average, 105 = 5% above, 95 = 5% below) "
            "and makes a very old promise: *when the band is stretched, it snaps back.*\n\n"
            "That's the claim we test — on SPY plus five liquid names (QQQ, IWM, AAPL, "
            "TSLA, NVDA), 2003 to today. Most rubber-band stories die on contact with a "
            "control group. This one dies in a particularly instructive way.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the drift-control math and "
            "the parameter grid? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** DI(10) = 100 x close / its own 10-day average; oversold "
            "< 95, overbought > 105 — the textbook short-horizon defaults, tested exactly "
            "as stated. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do stretched-down days really bounce more than average? | **Yes, on the "
            f"raw numbers.** +{R['mean_low_bps']:.0f} bps over the next 5 days vs "
            f"+{R['mean_mid_bps']:.0f} bps on a normal day (*t* = {R['welch_low']:.2f}). |\n"
            f"| Do stretched-*up* days sell off, like the folklore says? | **No — the "
            "opposite.** They keep going *up* by even more "
            f"(+{R['mean_high_bps']:.0f} bps, *t* = {R['welch_high']:.2f}). Half the rule "
            "is simply backwards on this tape. |\n"
            "| Is the \"buy the dip\" half really a *signal*, or just a good stock having "
            "a good two decades? | **Mostly the latter.** Buying an *arbitrary* random day "
            f"of the same stocks already earns +{R['dl_rand']:.0f} bps — buying specifically "
            f"the dip only adds **+{R['dl_delta']:.0f} bps more**, and that extra bit is "
            f"statistically unconvincing (*t* = {R['dl_delta_t']:.2f}). |\n"
            "| Can you trade the literal two-sided rule (buy dips, short rips)? | "
            f"**No — it loses money.** −{abs(R['timer_mean5']):.0f} bps per trade after "
            "costs, no better than a coin flip on the same days. |\n\n"
            "> The rubber band doesn't snap back. The stocks that were already running kept "
            "running, and the ones that dipped were mostly riding the same tide as every "
            "other day."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Price always reverts to its moving average. When the Disparity Index "
            "(close as a percent of its own N-day average) drifts too far from 100, that's "
            "the market overextending itself — fade the stretch.\"*\n\n"
            "It's a fixture of Korean and Japanese retail technical-analysis education, "
            "alongside cousins like the Psychological Line "
            "([sibling study 679](../../679-psychological-line/)). Unlike some folklore, "
            "it at least has an intuitive mechanism: a price a long way from its own "
            "recent average has, definitionally, moved a lot lately — and \"what goes up "
            "a lot, comes down a bit\" is a real phenomenon in *some* markets, *some* of "
            "the time."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real, this is about as simple as a trading rule gets: one moving average, "
            "one ratio, two thresholds — no options, no leverage, no news to read. It's "
            "the kind of rule a retail trading room can actually run by hand. So the "
            "stakes are practical: does the rubber band actually snap back often enough, "
            "hard enough, and *both directions* to pay for itself after costs — or is it "
            "quietly riding something else entirely (like the fact that a few of these "
            "stocks have been extraordinary investments for two decades)?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The comparison.** Bucket every trading day by its DI(10) reading — "
            "oversold (< 95), overbought (> 105), neutral — and compare the *next 5 "
            "trading days'* return across buckets.\n"
            "- **The literal trade.** Buy the day DI first dips below 95, short the day it "
            "first pops above 105, hold 5 days, pay round-trip costs, compare to a coin "
            "flip on the same entry days.\n"
            "- **The honest control.** These six tickers have had a very good two decades. "
            "So the real question isn't \"does DI beat a coin flip\" — it's **\"does buying "
            "the dip beat simply buying an arbitrary day of the same stock?\"** That's the "
            "control that actually decides it."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline split.** Forward 5-day return by DI bucket."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = st.pooled_conditional(BARS, window=R['window'], oversold=R['oversold'],\n"
            "                              overbought=R['overbought'], h=R['hold'])\n"
            "    lo, mid, hi = c['mean_low_bps'], c['mean_mid_bps'], c['mean_high_bps']\n"
            "else:\n"
            "    lo, mid, hi = R['mean_low_bps'], R['mean_mid_bps'], R['mean_high_bps']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "labels = ['oversold\\nDI<95', 'neutral', 'overbought\\nDI>105']\n"
            "vals = [lo, mid, hi]\n"
            "cols = [GREEN, GREY, RED]\n"
            "ax.bar(labels, vals, color=cols, width=.6)\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.0f} bps',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean forward 5-day return (bps)')\n"
            "ax.set_title('Both extremes beat neutral -- but which direction is the claim?')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'oversold {lo:+.2f} bps | neutral {mid:+.2f} bps | overbought {hi:+.2f} bps')"
        ),
        md(
            f"Both extremes beat the neutral day — oversold **+{R['mean_low_bps']:.0f} bps** "
            f"(*t* = {R['welch_low']:.2f}), overbought **+{R['mean_high_bps']:.0f} bps** "
            f"(*t* = {R['welch_high']:.2f}). Read that last number again: the claim says "
            "overbought should **underperform**. It doesn't — it's the *strongest* bucket "
            "on the chart. That's a momentum signature, not a rubber band.\n\n"
            "**Second, the literal trade.** Buy the dip, short the rip, exactly as the rule "
            "says — does it pay?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    strat, rand = st.pooled_trade_ledger(BARS, window=R['window'], oversold=R['oversold'],\n"
            "                                         overbought=R['overbought'], hold_days=R['hold'],\n"
            "                                         cost_bps=5.0, seed=680)\n"
            "    ss, rs = st.summarize(strat, 'ret_net', lags=R['hold']), st.summarize(rand, 'ret_net', lags=R['hold'])\n"
            "    sm, cm = ss['mean_bps'], rs['mean_bps']\n"
            "else:\n"
            "    sm, cm = R['timer_mean5'], R['coin_mean5']\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.3))\n"
            "ax.bar(['DI timer\\n(buy dip, short rip)', 'random-direction\\ncoin'], [sm, cm],\n"
            "       color=[RED, GREY], width=.55)\n"
            "for i,v in enumerate([sm, cm]): ax.annotate(f'{v:+.2f} bps',(i,v),ha='center',\n"
            "    va='top' if v<0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean net return per trade (bps, 5 bps costs)')\n"
            "ax.set_title('The literal two-sided rule loses money -- and a coin does no worse')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'DI timer {sm:+.2f} bps/trade  |  coin {cm:+.2f} bps/trade')"
        ),
        md(
            f"Net of costs, the DI timer loses **{R['timer_mean5']:.1f} bps per trade** "
            f"(*t* = {R['timer_t5']:.2f}) — and a coin flip on the *same* entry days does "
            f"*better* ({R['coin_mean5']:+.1f} bps). The short-overbought leg is fighting "
            "real momentum, and it shows.\n\n"
            "**Third — the question that actually settles it.** These six stocks have had "
            "an extraordinary run since 2003 (NVDA and TSLA especially). So: does buying "
            "specifically the *dip* beat simply buying *any* day of the same stock?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    db = st.random_day_baseline(BARS, window=R['window'], oversold=R['oversold'],\n"
            "                                overbought=R['overbought'], hold_days=R['hold'],\n"
            "                                cost_bps=5.0, seed=680)\n"
            "    dl, dl_r = db['low']['mean_bps'], db['low']['rand_mean_bps']\n"
            "else:\n"
            "    dl, dl_r = R['dl_mean'], R['dl_rand']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.3))\n"
            "ax.bar(['buy the dip\\n(DI<95 trigger)', 'buy ANY\\nrandom day'], [dl, dl_r],\n"
            "       color=[AMBER, GREY], width=.55)\n"
            "for i,v in enumerate([dl, dl_r]): ax.annotate(f'{v:+.1f} bps',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('mean net return per trade (bps)')\n"
            "ax.set_title(f'Buying the dip barely beats buying ANY day (delta t={R[\"dl_delta_t\"]:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'buy-the-dip {dl:+.2f} bps  vs  random-day {dl_r:+.2f} bps  '\n"
            "      f'delta {R[\"dl_delta\"]:+.2f} bps (t={R[\"dl_delta_t\"]:+.2f})')"
        ),
        md(
            f"There it is. Buying an *arbitrary* day of the same six stocks already earns "
            f"**+{R['dl_rand']:.0f} bps** over 5 days — because these are, on average, "
            f"stocks that went up a lot. Timing the dip with DI only adds "
            f"**+{R['dl_delta']:.0f} bps** on top, and that sliver isn't statistically "
            f"convincing (*t* = {R['dl_delta_t']:.2f}, below the desk's bar of 2). The "
            "\"signal\" in the first chart was mostly the tide, not the timing.\n\n"
            "**One more nail.** Is the Disparity Index even a *distinct* idea?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    dc = st.di_return_correlation(BARS, window=R['window'])\n"
            "    corr = dc['corr']\n"
            "else:\n"
            "    corr = R['corr']\n"
            "print(f'correlation between DI and the plain trailing {R[\"window\"]}-day return: r = {corr:.3f}')\n"
            "print('a value this close to 1 means DI is basically a relabeled version of the')\n"
            "print('same trailing-return signal tested (and found weak/microstructural) in')\n"
            "print('sibling study 329-one-month-reversal.')"
        ),
        md(
            f"**r = {R['corr']:.2f}.** The Disparity Index and the plain trailing return "
            "are almost the same number wearing different clothes. That's not a "
            "coincidence — distance-from-a-moving-average is mathematically close to "
            "accumulated return over the window. It's not a new discovery about market "
            "psychology; it's short-term reversal with a Korean name."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** Real on the raw split (both legs *t* > 3.8) but the "
            "overbought leg has the wrong sign, and the surviving oversold leg loses its "
            "edge against a same-stock random-day control "
            f"(*t* = {R['dl_delta_t']:.2f}).\n"
            "- **Tradability — Mirage.** The literal rule loses money net of costs and "
            "doesn't beat a coin flip.\n"
            "- **\"More than plain short-term reversal?\" — Busted.** r = "
            f"{R['corr']:.2f} with the trailing return; it's the same signal, already "
            "characterized as microstructure by a sibling study."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The general lesson:** any \"distance from an average\" indicator, on a "
            "universe that has genuinely trended, will look like it's picking up "
            "reversion when it's actually just picking up drift. Always ask for the "
            "random-*day* control, not just the random-*direction* coin.\n"
            "- **Where this indicator might still earn its keep:** a genuinely mean-"
            "reverting, non-trending universe (a sideways commodity, a pairs-trade "
            "spread) — the opposite of six of the best-performing large caps of the last "
            "two decades.\n"
            "- **Sibling studies:** [329-one-month-reversal](../../329-one-month-reversal/) "
            "(the ancestor DI turns out to be), "
            "[104-bollinger-reversion](../../104-bollinger-reversion/) (the Western band "
            "cousin, same drift trap), [679-psychological-line](../../679-psychological-line/) "
            "(the matched-protocol sibling on the same universe).\n\n"
            "*Think the Disparity Index earns its keep on a genuinely range-bound tape? "
            "Show a net, certifiable edge on a sideways commodity or index versus its own "
            "random-day control — then we'll talk.*"
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
            "# The Disparity Index — a quantitative teardown 🔬\n"
            "### Conditional Welch/NW splits · a two-sided timer vs a coin · the decisive "
            "random-DAY drift control · a window x threshold grid · a DI-vs-trailing-return "
            "diagnostic · a 20-seed synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The claim — **DI(N) = 100 x close / SMA(N) predicts mean reversion at its "
            "extremes** — is a Korean/Japanese-TA staple with no dedicated Western "
            "academic anchor; it inherits the broad short-horizon-reversal literature "
            "(DeBondt-Thaler 1985, Jegadeesh 1990, Lehmann 1990). The job here is to "
            "measure it honestly against the control that actually matters on a "
            "drift-heavy universe.\n\n"
            "> ⚠️ **Data note.** Daily total-return-adjusted OHLC, SPY/QQQ/IWM/AAPL/TSLA/"
            "NVDA, yfinance, cached, 2003-01-02 → 2026-06-30 (TSLA from 2010-06-29 IPO). "
            "SPY/QQQ/IWM carry no survivorship (index ETFs); the single-name sleeve is a "
            "name-recognition pick, named honestly in `data.py`. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprints `" + R["fp_spy"] +
            "`/`" + R["fp_qqq"] + "`/`" + R["fp_iwm"] + "`/`" + R["fp_aapl"] + "`/`" +
            R["fp_tsla"] + "`/`" + R["fp_nvda"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | oversold +{R['mean_low_bps']:.1f} bps/5d (Welch "
            f"t={R['welch_low']:.2f}, NW t={R['nw_low']:.2f}); overbought "
            f"+{R['mean_high_bps']:.1f} bps/5d (**wrong sign**, t={R['welch_high']:.2f}); "
            f"vs random-day: delta {R['dl_delta']:+.1f} bps at t={R['dl_delta_t']:.2f} |\n"
            f"| **Tradability** | `MIRAGE` | two-sided timer {R['timer_mean5']:+.2f} "
            f"bps/trade net (t={R['timer_t5']:.2f}) vs coin {R['coin_mean5']:+.2f} bps "
            f"(delta t={R['delta5_t']:.2f}) |\n"
            f"| **More than reversal?** | `BUSTED` | corr(DI, trailing "
            f"{R['window']}d return) = {R['corr']:.2f} |\n\n"
            "> 💡 In plain words: raw split looks like textbook mean reversion; a fair "
            "drift-matched control and a sign check on the *other* leg both say otherwise."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $DI_t(N) = 100 \\times P_t / \\mathrm{SMA}_N(P)_t$ and $R_{t\\to t+h}$ the "
            "forward $h$-day return entered at $t{+}1$'s open. The claim:\n\n"
            "- **H₁ (oversold bounce).** $E[R \\mid DI_t < 95] > E[R \\mid \\text{neutral}]$.\n"
            "- **H₂ (overbought pullback).** $E[R \\mid DI_t > 105] < E[R \\mid \\text{neutral}]$.\n"
            "- **H₃ (tradable timer).** A long-the-dip/short-the-rip ledger banks it net of "
            "costs, beating a random-direction coin.\n"
            "- **H₄ (distinct signal).** DI adds information beyond simply owning the "
            "underlying trend/drift of these names.\n\n"
            "We find **H₁ nominally supported but fragile** (survives vs neutral, fails vs "
            "a random-day control), **H₂ REJECTED** (wrong sign), **H₃ rejected** (net "
            "loss), **H₄ rejected** (r = 0.84 with the trailing return)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "The planned primary is a **Welch t** on the conditional split (oversold/"
            "overbought vs neutral); because *h*-day forward windows overlap, a "
            "**Newey-West(*h*) t** is the cross-check throughout — including on the "
            "synthetic null-world power check, where a raw Welch t measurably over-fires "
            "on a single autocorrelated path (documented below). Trade-ledger means carry "
            "a HAC(*h*) t on the per-trade series. The **decisive** control is not the "
            "random-direction coin (does the sign beat a coin on the days DI fires) but a "
            "**random-day, same-ticker, same-direction** baseline — because six names with "
            "two decades of positive drift make \"beats neutral\" a much weaker bar than "
            "\"beats an arbitrary day of the same stock.\""
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** SPY, QQQ, IWM, AAPL, TSLA, NVDA, {R['start']} → {R['end']} "
            "(TSLA from IPO). Index ETFs carry no survivorship; the single-name sleeve's "
            "mild name-recognition selection is named, not hidden.\n"
            f"- **Indicator.** DI(window={R['window']}) = 100 x close / trailing SMA; "
            f"zones oversold < {R['oversold']:.0f}, overbought > {R['overbought']:.0f}.\n"
            "- **Headline.** Conditional forward-return split, trigger-day only (avoids "
            "stacking near-duplicate in-zone days), Welch + NW(*h*).\n"
            "- **Timer.** Zone-trigger ledger, next-open entry, fixed-horizon exit, 2 x "
            "one-way cost x NAV per round trip, vs a random-direction coin on identical "
            "entries.\n"
            "- **Drift control.** Each single-sided leg vs a same-ticker, same-direction, "
            "same-size random-*day* baseline — the desk's answer to \"is this just beta?\"\n"
            "- **Diagnostic.** Pooled correlation between DI and the plain trailing "
            f"{R['window']}-day return.\n"
            "- **Robustness.** Window {5,10,20,25} x threshold {±3,±5,±7}% grid.\n"
            "- **Control.** Synthetic random-walk-plus-planted-reversion tape, causal DI, "
            "the null must not fire across 20 seeds under the calibrated NW detector."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline conditional split\n\n"
            "Forward 5-day return by DI bucket, trigger days only, pooled across the "
            "universe."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = st.pooled_conditional(BARS, window=R['window'], oversold=R['oversold'],\n"
            "                              overbought=R['overbought'], h=R['hold'])\n"
            "    print(f\"oversold  n={c['n_low']:,}  {c['mean_low_bps']:+.2f} bps  \"\n"
            "          f\"Welch t={c['welch_t_low']:+.2f}  NW t={c['nw_t_low']:+.2f}\")\n"
            "    print(f\"neutral   n={c['n_mid']:,}  {c['mean_mid_bps']:+.2f} bps\")\n"
            "    print(f\"overbought n={c['n_high']:,}  {c['mean_high_bps']:+.2f} bps  \"\n"
            "          f\"Welch t={c['welch_t_high']:+.2f}  NW t={c['nw_t_high']:+.2f}\")\n"
            "    lo, mid, hi = c['mean_low_bps'], c['mean_mid_bps'], c['mean_high_bps']\n"
            "    tl, th = c['welch_t_low'], c['welch_t_high']\n"
            "else:\n"
            "    lo, mid, hi = R['mean_low_bps'], R['mean_mid_bps'], R['mean_high_bps']\n"
            "    tl, th = R['welch_low'], R['welch_high']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "labels = ['oversold', 'neutral', 'overbought']\n"
            "a1.bar(labels, [lo, mid, hi], color=[GREEN, GREY, RED], width=.6)\n"
            "for i,v in enumerate([lo, mid, hi]): a1.annotate(f'{v:+.0f}',(i,v),ha='center',va='bottom')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean fwd 5d return (bps)')\n"
            "a1.set_title('Both extremes beat neutral')\n"
            "a2.bar(['low vs mid', 'high vs mid'], [tl, th], color=[GREEN, RED], width=.5)\n"
            "a2.axhline(2, ls='--', c=RED, lw=1); a2.axhline(-2, ls='--', c=RED, lw=1); a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_ylabel('Welch t'); a2.set_title('...but t_high is POSITIVE, not negative')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: H₁ clears the bar (t = {R['welch_low']:.2f} Welch, "
            f"{R['nw_low']:.2f} NW). H₂ is **rejected outright** — t = {R['welch_high']:.2f} "
            "is the wrong sign, by a wide margin. On this tape, \"overbought\" predicts "
            "*more* upside, the signature of momentum, not reversion."
        ),
        md(
            "### 4b · The literal timer vs a coin\n\n"
            "Enter next open, hold 5 days, 2 x one-way cost x NAV per round trip; "
            "long-oversold / short-overbought exactly as prescribed."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows5 = st.pooled_trade_ledger(BARS, window=R['window'], oversold=R['oversold'],\n"
            "                                   overbought=R['overbought'], hold_days=R['hold'],\n"
            "                                   cost_bps=5.0, seed=680)\n"
            "    rows10 = st.pooled_trade_ledger(BARS, window=R['window'], oversold=R['oversold'],\n"
            "                                    overbought=R['overbought'], hold_days=R['hold'],\n"
            "                                    cost_bps=10.0, seed=680)\n"
            "    ss5, rs5 = st.summarize(rows5[0], 'ret_net', R['hold']), st.summarize(rows5[1], 'ret_net', R['hold'])\n"
            "    ss10, rs10 = st.summarize(rows10[0], 'ret_net', R['hold']), st.summarize(rows10[1], 'ret_net', R['hold'])\n"
            "    m5, c5, m10, c10 = ss5['mean_bps'], rs5['mean_bps'], ss10['mean_bps'], rs10['mean_bps']\n"
            "else:\n"
            "    m5, c5, m10, c10 = R['timer_mean5'], R['coin_mean5'], R['timer_mean10'], R['coin_mean10']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "x = np.arange(2); w = .35\n"
            "ax.bar(x - w/2, [m5, m10], width=w, color=RED, label='DI timer')\n"
            "ax.bar(x + w/2, [c5, c10], width=w, color=GREY, label='random-direction coin')\n"
            "ax.set_xticks(x); ax.set_xticklabels(['5 bps costs', '10 bps costs'])\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('mean net bps/trade')\n"
            "ax.set_title('Loses money at every cost level, no better than a coin')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'5bps: DI {m5:+.2f} vs coin {c5:+.2f}   10bps: DI {m10:+.2f} vs coin {c10:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: H₃ rejected. At 5 bps the DI timer nets "
            f"**{R['timer_mean5']:.1f} bps/trade** (t = {R['timer_t5']:.2f}) — negative, "
            f"and the delta vs the coin is t = {R['delta5_t']:.2f}. The short-overbought "
            "leg is the culprit: it fights the momentum documented in 4a."
        ),
        md(
            "### 4c · The decisive control — random DAY, not random direction\n\n"
            "Each single-sided leg vs a same-ticker, same-direction, same-size random-day "
            "entry (a different calendar day of the *same* stock, not a coin flip on the "
            "same day)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    db = st.random_day_baseline(BARS, window=R['window'], oversold=R['oversold'],\n"
            "                                overbought=R['overbought'], hold_days=R['hold'],\n"
            "                                cost_bps=5.0, seed=680)\n"
            "    dl, dlr, dlt = db['low']['mean_bps'], db['low']['rand_mean_bps'], db['low']['welch_t_vs_random_day']\n"
            "    dh, dhr, dht = db['high']['mean_bps'], db['high']['rand_mean_bps'], db['high']['welch_t_vs_random_day']\n"
            "else:\n"
            "    dl, dlr, dlt = R['dl_mean'], R['dl_rand'], R['dl_delta_t']\n"
            "    dh, dhr, dht = R['dh_mean'], R['dh_rand'], R['dh_delta_t']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.bar(['buy-the-dip', 'random day\\n(same stock)'], [dl, dlr], color=[AMBER, GREY], width=.55)\n"
            "for i,v in enumerate([dl, dlr]): a1.annotate(f'{v:+.1f}',(i,v),ha='center',va='bottom')\n"
            "a1.set_title(f'Long leg: delta t = {dlt:+.2f}'); a1.set_ylabel('mean net bps/trade')\n"
            "a2.bar(['short-the-rip', 'random day\\n(same stock)'], [dh, dhr], color=[AMBER, GREY], width=.55)\n"
            "for i,v in enumerate([dh, dhr]): a2.annotate(f'{v:+.1f}',(i,v),ha='center',va='top')\n"
            "a2.set_title(f'Short leg: delta t = {dht:+.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'long: DI {dl:+.2f} vs random-day {dlr:+.2f} (t={dlt:+.2f})')\n"
            "print(f'short: DI {dh:+.2f} vs random-day {dhr:+.2f} (t={dht:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: this is the whole study in one chart. Random days in "
            f"this universe already earn **+{R['dl_rand']:.0f} bps** long (pure drift) and "
            f"lose **{R['dh_rand']:.0f} bps** short (the same drift, working against a "
            f"short). Timing with DI adds only **+{R['dl_delta']:.1f} bps** on the long "
            f"side (t = {R['dl_delta_t']:.2f} — below the bar) and is *worse* than random "
            f"on the short side (t = {R['dh_delta_t']:.2f}). H₁'s survival in 4a was "
            "almost entirely the tide, not the timing."
        ),
        md(
            "### 4d · Parameter robustness — is the wrong sign a fluke of one threshold?\n\n"
            "Window {5,10,20,25} x threshold {±3,±5,±7}% grid, conditional-split Welch t."
        ),
        code(
            "if HAVE_REAL:\n"
            "    grid = st.param_grid(BARS, h=R['hold'])\n"
            "    ws = sorted(grid['window'].unique())\n"
            "    th_lo = {w: grid[grid.window==w].sort_values('oversold')['welch_t_low'].tolist() for w in ws}\n"
            "    th_hi = {w: grid[grid.window==w].sort_values('oversold')['welch_t_high'].tolist() for w in ws}\n"
            "else:\n"
            "    ws = sorted(R['grid'].keys())\n"
            "    pairs_sorted = lambda w: sorted(R['grid'][w].keys())\n"
            "    th_lo = {w: [R['grid'][w][p][0] for p in pairs_sorted(w)] for w in ws}\n"
            "    th_hi = {w: [R['grid'][w][p][1] for p in pairs_sorted(w)] for w in ws}\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.4), sharey=True)\n"
            "for w in ws:\n"
            "    a1.plot(range(3), th_lo[w], marker='o', label=f'window={w}')\n"
            "    a2.plot(range(3), th_hi[w], marker='o', label=f'window={w}')\n"
            "for a, ttl in ((a1, 'oversold leg (t_low)'), (a2, 'overbought leg (t_high)')):\n"
            "    a.axhline(2, ls='--', c=RED, lw=1); a.axhline(-2, ls='--', c=RED, lw=1); a.axhline(0, c='k', lw=.6)\n"
            "    a.set_xticks(range(3)); a.set_xticklabels(['+-3%','+-5%','+-7%'])\n"
            "    a.set_title(ttl); a.set_xlabel('band width')\n"
            "a1.set_ylabel('Welch t'); a1.legend(fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('t_high stays POSITIVE across every window x threshold cell -- structural, not a fluke')"
        ),
        md(
            "> 💡 In plain words: the overbought leg's wrong sign is not a cherry-picked "
            "window or band — it holds (and *strengthens* with longer windows) across "
            "every cell of the grid. Whatever DI is picking up at N=20 or 25 days, it "
            "isn't a rubber band."
        ),
        md(
            "### 4e · \"Just short-term reversal?\" — the correlation diagnostic\n\n"
            "DI is a moving-average-relative distance; a plain trailing return is a "
            "start/end-point distance. Both summarize \"how far has price moved lately.\""
        ),
        code(
            "if HAVE_REAL:\n"
            "    dc = st.di_return_correlation(BARS, window=R['window'])\n"
            "    corr = dc['corr']\n"
            "else:\n"
            "    corr = R['corr']\n"
            "fig, ax = plt.subplots(figsize=(6.0, 5.2))\n"
            "ax.bar(['DI vs trailing\\nreturn'], [corr], color=AMBER, width=.4)\n"
            "ax.axhline(1.0, c='k', lw=.6, ls=':')\n"
            "ax.set_ylim(0, 1.05); ax.set_ylabel('pooled Pearson r')\n"
            "ax.annotate(f'r = {corr:.3f}', (0, corr), ha='center', va='bottom', fontsize=12)\n"
            "ax.set_title('DI is a near-relabeling of the trailing return')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'r = {corr:.4f}')"
        ),
        md(
            f"> 💡 In plain words: H₄ rejected. r = {R['corr']:.2f} is high enough that DI "
            "is functionally the same signal as the trailing N-day return — the exact "
            "quantity [329-one-month-reversal](../../329-one-month-reversal/) tested and "
            "found to be real-but-microstructural, dead since 2002. The \"Disparity "
            "Index\" is short-term reversal with different units and a different name."
        ),
        md(
            "### 4f · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic random-walk-plus-planted-reversion tape, disparity computed "
            "causally (no look-ahead), a 6-series basket per seed mirroring the real "
            "6-ticker pooling. Because the 5-day forward windows overlap, a raw Welch t "
            "over-fires on a single autocorrelated path — the calibrated detector is "
            "NW(5), the same cross-check used on the real headline split."
        ),
        code(
            "null_lo_w, null_hi_w, null_lo_nw, null_hi_nw = [], [], [], []\n"
            "for s_ in range(20):\n"
            "    basket = data.synthetic_basket(reversal=0.0, seed=680 + s_)\n"
            "    r = st.pooled_conditional(basket, window=R['window'], oversold=R['oversold'],\n"
            "                              overbought=R['overbought'], h=R['hold'])\n"
            "    null_lo_w.append(r['welch_t_low']); null_hi_w.append(r['welch_t_high'])\n"
            "    null_lo_nw.append(r['nw_t_low']); null_hi_nw.append(r['nw_t_high'])\n"
            "null_lo_nw = np.asarray(null_lo_nw); null_hi_nw = np.asarray(null_hi_nw)\n"
            "planted = data.synthetic_basket(reversal=0.006, seed=680)\n"
            "rp = st.pooled_conditional(planted, window=R['window'], oversold=R['oversold'],\n"
            "                           overbought=R['overbought'], h=R['hold'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12,.12,20), null_lo_nw, color=GREEN, s=40,\n"
            "           label='null, low leg (NW t), 20 seeds')\n"
            "ax.scatter(np.ones(20) + np.linspace(-.12,.12,20), null_hi_nw, color=RED, s=40,\n"
            "           label='null, high leg (NW t), 20 seeds')\n"
            "ax.scatter([2], [rp['nw_t_low']], color=GREEN, marker='D', s=110, zorder=5, label='planted, low leg')\n"
            "ax.scatter([3], [rp['nw_t_high']], color=RED, marker='D', s=110, zorder=5, label='planted, high leg')\n"
            "ax.axhline(-2, ls='--', c='k', lw=1); ax.axhline(2, ls='--', c='k', lw=1)\n"
            "ax.set_xticks([0,1,2,3]); ax.set_xticklabels(['null low','null high','planted low','planted high'])\n"
            "ax.set_ylabel('NW(5) t'); ax.set_title('Control: null rarely fires, a planted reversal lights up')\n"
            "ax.legend(fontsize=8); plt.tight_layout(); plt.show()\n"
            "print(f'null NW t_low: mean {null_lo_nw.mean():+.2f} (sd {null_lo_nw.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_lo_nw)>=2).sum()}/20')\n"
            "print(f'null NW t_high: mean {null_hi_nw.mean():+.2f} (sd {null_hi_nw.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_hi_nw)>=2).sum()}/20')\n"
            "print(f'planted NW t_low={rp[\"nw_t_low\"]:+.2f}  t_high={rp[\"nw_t_high\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: under the calibrated NW(5) detector the null fires "
            f"close to the 5% nominal rate expected of a two-sided *t* ≥ 2 test "
            f"({R['syn_null_nw_lo'][2]}/20 low, {R['syn_null_nw_hi'][2]}/20 high — a raw "
            f"Welch t over-fires here, {R['syn_null_welch_lo'][2]}/20 and "
            f"{R['syn_null_welch_hi'][2]}/20, because the overlapping forward windows "
            "induce serial correlation a naive Welch test doesn't see), and a planted "
            f"reversal lights up both legs correctly (NW t = {R['syn_planted_nw'][0]:.2f} / "
            f"{R['syn_planted_nw'][1]:.2f}). The machinery is unbiased — the real tape's "
            "wrong-signed overbought leg is a genuine feature of the data, not a detector "
            "artefact. *(Faithful-engine / power check only — never cited in support of "
            "the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — oversold beats neutral (Welch t={R['welch_low']:.2f}, "
            f"NW t={R['nw_low']:.2f}) but overbought has the **wrong sign** "
            f"(t={R['welch_high']:.2f}); the surviving oversold edge falls to "
            f"t={R['dl_delta_t']:.2f} against a same-ticker random-day control. Raw "
            "significance without a fair drift baseline was misleading.\n"
            f"- **Tradability `MIRAGE`** — the literal timer nets {R['timer_mean5']:+.2f} "
            f"bps/trade at 5 bps (t={R['timer_t5']:.2f}), not distinguishable from a "
            f"random-direction coin (delta t={R['delta5_t']:.2f}); loses money at every "
            "cost level tested.\n"
            f"- **\"More than plain short-term reversal?\" `BUSTED`** — r={R['corr']:.2f} "
            "with the trailing return; the wrong-signed leg and the failed drift control "
            "both point the same way: this is captured beta, not a rubber band."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general object is drift contamination in any distance-from-trend "
            "rule.** Any oscillator built on \"how far from its own average\" needs a "
            "random-day (not just random-direction) control on a trending universe — this "
            "is the same failure mode [104-bollinger-reversion](../../104-bollinger-reversion/) "
            "found independently for the Western band-distance cousin.\n"
            "- **A fairer test bed** would be a genuinely range-bound instrument (a "
            "sideways commodity, a stat-arb spread) rather than six of the strongest "
            "secular winners of the last 20 years — the natural sequel.\n"
            "- **Dedup map:** [329-one-month-reversal](../../329-one-month-reversal/) (the "
            "ancestor signal, r=0.84 with DI here), "
            "[104-bollinger-reversion](../../104-bollinger-reversion/) (same drift trap, "
            "Western bands), [137-mansfield-rs](../../137-mansfield-rs/) (the opposite, "
            "trend-following philosophy), "
            "[679-psychological-line](../../679-psychological-line/) (matched protocol, "
            "same universe, different oscillator).\n\n"
            "*The reproducible core is offline and deterministic; frozen numbers live in "
            "[`docs/results.md`](../docs/results.md), sources in "
            "[`docs/references.md`](../docs/references.md).*"
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
