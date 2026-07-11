"""Generate the two narrative notebooks for Study 668 (Williams VIX Fix).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached eight-ticker
basket under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance 8-ticker basket
# 2000-01-03 -> 2026-06-30, IWM from 2000-05-26; WVF(22), 20/2sigma Bollinger onset trigger).
R = dict(
    start="2000-01-03", end="2026-06-30",
    tickers=["SPY", "QQQ", "IWM", "AAPL", "MSFT", "JPM", "XOM", "JNJ"],
    etf_tickers=["SPY", "QQQ", "IWM"],
    # horizon -> (n_spike, spike_bps, rest_bps, gap_bps, welch_t, hit_pct, wilson_lo, wilson_hi,
    #             nw_mean, nw_min, nw_max)
    headline={
        5: (3783, 26.3, 16.7, 9.6, 1.33, 56.3, 54.7, 57.9, 0.59, -0.73, 2.38),
        10: (3782, 29.9, 38.7, -8.8, -0.88, 55.3, 53.7, 56.9, -0.11, -1.57, 0.95),
        20: (3765, 57.8, 81.6, -23.9, -1.74, 58.6, 57.0, 60.2, -0.33, -1.48, 0.32),
    },
    # h=5 individually-significant-looking names (die by h=10) — the multiple-comparisons caution
    h5_spy=(529, 2.44, 1.91), h5_qqq=(478, 3.08, 2.38),
    # per-ticker h=10: (n, spike_bps, rest_bps, welch_t, nw_t, hit_pct)
    per_ticker={
        "SPY": (529, 41.7, 28.3, 0.83, 0.58, 60.1),
        "QQQ": (478, 55.0, 26.3, 1.39, 0.95, 56.9),
        "IWM": (510, 12.8, 28.1, -0.61, -0.38, 54.1),
        "AAPL": (475, 30.6, 86.3, -1.15, -0.81, 55.7),
        "MSFT": (435, -20.1, 38.5, -2.24, -1.57, 49.4),
        "JPM": (449, 53.1, 33.9, 0.56, 0.37, 54.3),
        "XOM": (452, 18.8, 31.6, -0.48, -0.34, 57.9),
        "JNJ": (456, 43.9, 36.1, 0.43, 0.30, 53.1),
    },
    # random-calendar placebo, h=10 pooled
    placebo_obs=29.9, placebo_mean=38.8, placebo_sd=7.8, placebo_p=0.874, placebo_draws=10000,
    # third axis: WVF vs plain close-only drawdown proxy, h=10.
    # ticker -> (overlap_pct, t_wvf_marginal, t_dd_marginal, wvf_only_bps, n_wvf_only,
    #            dd_only_bps, n_dd_only, welch_t_diff)
    wick={
        "SPY": (74.1, -0.04, 0.88, 10.8, 137, 35.1, 169, -0.58),
        "QQQ": (75.9, -0.41, 2.02, 34.3, 115, 98.2, 158, -1.29),
        "IWM": (76.1, -0.86, 1.12, -50.5, 122, 24.6, 129, -1.13),
        "AAPL": (72.2, -0.91, 0.23, 10.5, 132, 88.2, 159, -0.75),
        "MSFT": (74.9, -1.66, -0.06, -23.7, 109, 33.3, 154, -1.02),
        "JPM": (76.8, 0.68, -0.52, 70.6, 104, 12.8, 125, 0.65),
        "XOM": (75.2, -0.07, -0.37, 26.9, 112, 15.1, 126, 0.17),
        "JNJ": (78.3, 0.89, -0.82, 37.9, 99, 0.1, 157, 0.83),
    },
    wick_avg_overlap=75.5, wick_avg_t_wvf=-0.30, wick_avg_t_dd=0.31,
    # the timer, h=10
    timer_gross=(3782, 29.9, 1.76),
    timer_net5=(3782, 19.9, 54.4, 0.033, 1.17),
    timer_net10=(3782, 9.9, 53.1, 0.016, 0.58),
    # ticker -> (n, net5_bps, win_pct, hac_t)
    timer_per_ticker={
        "SPY": (529, 31.7, 59.5, 1.32), "QQQ": (478, 45.0, 56.1, 1.52),
        "IWM": (510, 2.8, 53.9, 0.06), "AAPL": (474, 20.6, 55.5, 0.25),
        "MSFT": (435, -30.1, 48.5, -0.76), "JPM": (449, 43.1, 53.0, 0.77),
        "XOM": (451, 8.8, 56.1, 0.22), "JNJ": (456, 33.9, 51.8, 1.28),
    },
    # synthetic control
    syn_null_welch_mean=-0.29, syn_null_welch_sd=1.22, syn_null_welch_fire=1,
    syn_null_nw_mean=-0.18, syn_null_nw_sd=0.82, syn_null_nw_fire=0, syn_seeds=20,
    syn_planted_welch=3.54, syn_planted_nw=2.47, syn_planted_spike=30.1, syn_planted_rest=-44.8,
    syn_planted_n=351,
    fp={"SPY": "ff6a728ebf89", "QQQ": "dc33a4d14e22", "IWM": "1dd0a989ecb9",
        "AAPL": "a1278fc100bc", "MSFT": "0dcc0426245d", "JPM": "ccf94046c951",
        "XOM": "9907d9f69d09", "JNJ": "62ea82f560f2"},
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Drawdown_proxy%3F: Busted](https://img.shields.io/badge/More_than_a_drawdown_proxy%3F-Busted-8b949e?style=flat-square)\n\n"
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

from williams_vix_fix import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    REAL = data.load_real()
    FRAMES = {t: st.ticker_frame(REAL[t]) for t in data.TICKERS}
else:
    REAL = FRAMES = None
print("real cache present:", HAVE_REAL, "| basket:", data.TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# A \"fear gauge\" built from nothing but price — does it call the bottom? 📉🔧\n"
            "### The Williams VIX Fix — a synthetic VIX with no options data, and the "
            "spike that (mostly) doesn't pay\n\n"
            + BADGES +
            "The real VIX needs a whole options chain to compute. Larry Williams — the same "
            "trader behind %R — wondered: could you fake it with nothing but the daily bar? "
            "His answer, the **VIX Fix**: measure how far today's low sits below the recent "
            "run of closing highs. Panic days (low wicks far down) spike it; calm days barely "
            "move it. The retail rule: when the spike pokes above its own Bollinger band, "
            "that's capitulation — **buy**.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the "
            "drawdown-proxy autopsy? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** Eight-ticker basket (SPY/QQQ/IWM + AAPL/MSFT/JPM/XOM/JNJ), "
            "2000→2026, one entry per capitulation episode, next-open execution. House style "
            "in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the spike predict a bounce? | **Not really.** Pooled across 8 names, "
            "5/10/20-day forward returns after a spike beat an ordinary day by "
            f"**{R['headline'][5][3]:+.1f} bps** (5d), then go **negative**: "
            f"{R['headline'][10][3]:+.1f} bps (10d), {R['headline'][20][3]:+.1f} bps (20d). "
            "None of that clears the statistical bar. |\n"
            "| Is it just luck dressed up? | A random-calendar test says the observed 10-day "
            f"spike-day return **loses to {R['placebo_p']*100:.0f}%** of purely random "
            "3,782-day calendars — worse than a coin flip. |\n"
            "| Is the spike more than \"price fell a lot\"? | **No.** 3 out of every 4 spikes "
            "are days a plain \"how far below its recent high\" filter — no fancy formula "
            "needed — would have caught anyway. The intraday low adds nothing measurable. |\n"
            "| Can you trade it? | Before a single cent of cost, the 10-day timer already "
            f"misses the bar (*t* = {R['timer_gross'][2]:+.2f}). Costs only make it worse. |\n\n"
            "> The chart looks dramatic. The number after it doesn't."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Extreme fear shows up in price itself, not just in the options market. "
            "Measure how far today's low undercuts the recent run of closing highs — when "
            "that gap spikes past its own normal range, you're looking at capitulation. Buy "
            "it.\"*\n\n"
            "It's an appealing idea because it's *true by construction* that WVF spikes on "
            "panic days — the formula is built to react to a low wick. The open question is "
            "whether that reaction predicts anything **useful**, or whether it's just a fancy "
            "way of noticing \"price fell a lot,\" which the market may or may not reward."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real, this would be a genuinely useful tool: a VIX-shaped fear signal for "
            "**any** market, including ones with no listed options — small caps, most "
            "single-name futures, crypto, decades of history before the real VIX existed. "
            "\"Buy the fear spike\" is also one of the most common retail chart-room recipes, "
            "so it's worth testing at full strength rather than waving it away."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "- **The formula, exactly.** `WVF = (highest_close(22) - low) / highest_close(22) "
            "* 100`; spike = the first day WVF pokes above its own 20-day mean + 2 standard "
            "deviations (the canonical Bollinger trigger).\n"
            "- **The basket.** 8 liquid names, 2000→2026: SPY/QQQ/IWM (no survivorship) plus "
            "AAPL/MSFT/JPM/XOM/JNJ (long-history survivors — named honestly).\n"
            "- **The comparison.** Forward returns 5/10/20 days after a spike vs an "
            "unconditional day, one execution lag (enter the next open).\n"
            "- **The luck check.** Draw random days instead of spike days, 10,000 times — how "
            "often does that random pick do as well as the real spikes?\n"
            "- **The fair-review question.** Build the *same* trigger from the close alone — "
            "no intraday low — and see if WVF beats that simpler cousin."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Forward return after a WVF spike vs an ordinary day, at "
            "three horizons."
        ),
        code(
            "hz = [5, 10, 20]\n"
            "gaps = [R['headline'][h][3] for h in hz]\n"
            "ts = [R['headline'][h][4] for h in hz]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "cols = [GREY if abs(t) < 2 else RED for t in ts]\n"
            "ax.bar([f'{h}d' for h in hz], gaps, color=cols, width=.55)\n"
            "for i,(g,t) in enumerate(zip(gaps, ts)):\n"
            "    ax.annotate(f'{g:+.1f} bps\\n(t={t:+.2f})', (i,g), ha='center',\n"
            "        va='bottom' if g>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('spike-day forward return minus unconditional (bps)')\n"
            "ax.set_title('The spike wins small at 5 days, then loses at 10 and 20')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({h: (round(g,1), round(t,2)) for h,g,t in zip(hz, gaps, ts)})"
        ),
        md(
            "That's the whole story in one chart: a faint, uncertifiable edge at 5 days that "
            "**flips negative** by 10 and 20. Two individual names (SPY, QQQ) look convincing "
            f"at 5 days alone (*t* = {R['h5_spy'][1]:+.2f} and {R['h5_qqq'][1]:+.2f}) — but "
            "with 8 names × 3 horizons = 24 separate tests run, seeing 2 cross the bar by pure "
            "luck is *expected*, and neither survives to the next horizon. That's the "
            "signature of noise dressed up as a discovery.\n\n"
            "**Next, the luck check.** Would a random 10-day calendar have done just as well?"
        ),
        code(
            "rng = np.random.default_rng(668)\n"
            "draws = rng.normal(R['placebo_mean'], R['placebo_sd'], 4000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='null: random 10-day calendars')\n"
            "ax.axvline(R['placebo_obs'], c=RED, lw=2.5,\n"
            "           label=f\"observed spike-day mean {R['placebo_obs']:+.1f} bps\")\n"
            "ax.set_xlabel('mean 10-day forward return of a random calendar (bps)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"Beaten by {R['placebo_p']*100:.1f}% of random calendars\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"observed {R['placebo_obs']:+.1f} bps vs placebo mean \"\n"
            "      f\"{R['placebo_mean']:+.1f} bps (sd {R['placebo_sd']:.1f}), p={R['placebo_p']:.3f}\")"
        ),
        md(
            f"The observed spike-day mean sits **inside** the random-calendar cloud, on the "
            f"losing side of it — **{R['placebo_p']*100:.1f}%** of purely random calendars beat "
            "it. A real edge should stick out on the *right*; this one doesn't stick out at all.\n\n"
            "**Now the fair-review question.** Is WVF's extra ingredient — the intraday low — "
            "actually doing anything a plain \"far below its recent high\" close-only filter "
            "wouldn't already catch?"
        ),
        code(
            "tks = list(R['wick'].keys())\n"
            "ov = [R['wick'][t][0] for t in tks]\n"
            "twvf = [R['wick'][t][1] for t in tks]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.4))\n"
            "a1.bar(tks, ov, color=AMBER, width=.6)\n"
            "a1.axhline(R['wick_avg_overlap'], ls='--', c='k', lw=1)\n"
            "a1.set_ylabel('% of WVF spikes also caught by a plain drawdown flag')\n"
            "a1.set_title('Three-quarters overlap, every name')\n"
            "a1.tick_params(axis='x', rotation=45)\n"
            "a2.bar(tks, twvf, color=[RED if abs(t)>=2 else GREY for t in twvf], width=.6)\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.axhline(-2, ls='--', c=RED, lw=1); a2.axhline(2, ls='--', c=RED, lw=1)\n"
            "a2.set_ylabel(\"WVF's marginal t-stat, controlling for the plain drawdown flag\")\n"
            "a2.set_title(f\"Average marginal t = {R['wick_avg_t_wvf']:+.2f} — nothing left\")\n"
            "a2.tick_params(axis='x', rotation=45)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('overlap:', dict(zip(tks, ov)))\n"
            "print('marginal WVF t:', dict(zip(tks, twvf)))"
        ),
        md(
            f"On average **{R['wick_avg_overlap']:.1f}%** of every WVF spike would have fired "
            "on the plain drawdown proxy anyway — and once you control for that overlap, WVF's "
            f"own marginal contribution averages *t* = **{R['wick_avg_t_wvf']:+.2f}**: not one "
            "name shows it adding anything real, and it's negative more often than not. The "
            "intraday-low ingredient — the whole reason WVF calls itself a *volatility* "
            "indicator rather than just a *level* indicator — buys nothing extra.\n\n"
            "**Finally, the trade.** Buy every spike, hold 10 sessions, pay round-trip costs."
        ),
        code(
            "labels = ['gross', 'net 5bps', 'net 10bps']\n"
            "vals = [R['timer_gross'][1], R['timer_net5'][1], R['timer_net10'][1]]\n"
            "ts = [R['timer_gross'][2], R['timer_net5'][4], R['timer_net10'][4]]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(labels, vals, color=[GREY if abs(t)<2 else RED for t in ts], width=.55)\n"
            "for i,(v,t) in enumerate(zip(vals, ts)):\n"
            "    ax.annotate(f'{v:+.1f} bps\\n(t={t:+.2f})', (i,v), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean return per 10-day spike trade (bps)')\n"
            "ax.set_title('Never clears the bar — not even gross of costs')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dict(zip(labels, zip(vals, ts))))"
        ),
        md(
            f"Even with **zero** costs the timer's *t*-stat ({R['timer_gross'][2]:+.2f}) misses "
            "the bar. Charge a realistic 5 bps one-way and it drops further "
            f"({R['timer_net5'][4]:+.2f}); at 10 bps it's essentially nothing "
            f"({R['timer_net10'][4]:+.2f}). There's no edge here for costs to erode — the "
            "mirage is visible before you even open a brokerage account."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** No horizon (5/10/20 days) shows a certifiable edge; the "
            "sign flips negative past 5 days, and a random-calendar test shows the observed "
            f"mean losing to {R['placebo_p']*100:.0f}% of chance.\n"
            "- **Tradability — Mirage.** The timer misses the statistical bar even before "
            "costs; real-world spreads only make it worse.\n"
            "- **\"More than a drawdown proxy?\" — Busted.** Three-quarters of every spike is "
            "just \"price is far below its recent high\" wearing a fancier formula; the "
            "intraday low adds nothing measurable."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The chart-appeal trap.** WVF *looks* dramatic on a chart because it's built "
            "to react hard to panic wicks — but reacting to fear and *predicting* what fear "
            "does next are different claims, and only the first one is true here.\n"
            "- **Where a wick-based signal might still earn its keep** is inside a market with "
            "no close-based drawdown alternative at all (thin order books, gapping futures) — "
            "the drawdown-proxy control here assumed a name liquid enough to have a clean "
            "close series, which SPY/QQQ/AAPL etc. certainly are.\n"
            "- **Sibling studies:** [127-williams-r](../../127-williams-r/) (Larry Williams' "
            "other, unrelated oscillator), [111-vix-term-structure](../../111-vix-term-structure/) "
            "(the *real* VIX curve) and [92-easy-money](../../92-easy-money/) (VIX-futures "
            "carry) — none of them retest this claim.\n\n"
            "*Think a different Bollinger width or lookback saves it? Show a net, certifiable "
            "edge that survives the drawdown-proxy control — then we'll talk.*"
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
            "# The Williams VIX Fix — a quantitative teardown 🔬\n"
            "### Pooled Welch + per-ticker Newey-West across 5/10/20-day horizons · a "
            "10,000-draw random-calendar placebo · a two-dummy HAC drawdown-proxy "
            "head-to-head · a cost-adjusted timer · a 20-seed synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The claim — a **price-only VIX proxy spikes at capitulation bottoms, buy the "
            "spike** — has a plausible mechanism (a wide low-wick is genuine intraday panic) "
            "but no academic anchor: WVF is retail-platform folklore (Larry Williams' formula, "
            "ChrisMoody's Bollinger-trigger port). The job is to measure it honestly against "
            "an eight-name basket, then ask the two questions that matter: *is it real, and "
            "is it anything more than a drawdown filter with extra steps?*\n\n"
            "> ⚠️ **Data note.** 8-ticker daily OHLC basket (SPY/QQQ/IWM/AAPL/MSFT/JPM/XOM/"
            "JNJ), yfinance auto-adjusted, 2000→2026 (IWM from 2000-05-26). "
            "**Survivorship named on the Signal axis:** SPY/QQQ/IWM carry none (index ETFs); "
            "AAPL/MSFT/JPM/XOM/JNJ are named survivors of a long clean tape. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            "| **Signal** | `NONE` | pooled gap/Welch *t*: "
            f"5d {R['headline'][5][3]:+.1f} bps ({R['headline'][5][4]:+.2f}), "
            f"10d {R['headline'][10][3]:+.1f} bps ({R['headline'][10][4]:+.2f}), "
            f"20d {R['headline'][20][3]:+.1f} bps ({R['headline'][20][4]:+.2f}); placebo "
            f"*p* = {R['placebo_p']:.3f} |\n"
            f"| **Tradability** | `MIRAGE` | gross timer *t* = {R['timer_gross'][2]:+.2f}, "
            f"net 5 bps *t* = {R['timer_net5'][4]:+.2f}, net 10 bps *t* = "
            f"{R['timer_net10'][4]:+.2f} |\n"
            f"| **Drawdown proxy?** | `BUSTED` | {R['wick_avg_overlap']:.1f}% overlap, "
            f"marginal WVF *t* = {R['wick_avg_t_wvf']:+.2f} |\n\n"
            "> 💡 In plain words: the formula reacts to real panic, but the reaction predicts "
            "nothing extra — and the panic-detection itself is 3/4 redundant with a one-line "
            "drawdown filter."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $WVF_t = 100 \\times \\frac{HC_{22,t} - L_t}{HC_{22,t}}$ where "
            "$HC_{22,t} = \\max(C_{t-21}, \\ldots, C_t)$. Let $S_t \\in \\{0,1\\}$ flag the "
            "**onset** of $WVF_t \\ge \\mu_{20}(WVF) + 2\\sigma_{20}(WVF)$ (both computed on "
            "data through $t$ — no look-ahead). The claims:\n\n"
            "- **H₁ (bounce).** $E[r^{fwd}_{t,h} \\mid S_t=1] \\gg E[r^{fwd}_{t,h} \\mid S_t=0]$ "
            "for $h \\in \\{5,10,20\\}$ — spikes lead to systematically better forward returns.\n"
            "- **H₂ (not luck).** The observed spike-day mean is not something a random "
            "calendar of equal size would reproduce.\n"
            "- **H₃ (more than a drawdown proxy).** Controlling for a plain close-only "
            "drawdown flag fired on the same day, WVF's own dummy still carries positive "
            "marginal information (the intrabar low earns its keep).\n"
            "- **H₄ (capture).** A spike-onset timer banks the edge net of realistic costs.\n\n"
            "We find **H₁ not supported** (wrong sign past 5 days), **H₂ rejected** (placebo "
            "*p* = 0.874 — the observed mean *loses* to random), **H₃ rejected** (marginal "
            f"*t* = {R['wick_avg_t_wvf']:+.2f}), **H₄ moot** (nothing to capture)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Forward returns of horizon *h* are windows that **overlap** by construction — a "
            "spike on day *t* and one on day *t+3* share most of their forward window when "
            "*h* = 10 — so the planned primary is a **pooled Welch t** (treats each spike "
            "episode as one observation, robust to unequal variances) cross-checked by a "
            "**per-ticker Newey-West (HAC) dummy regression** with lags = *h*, which prices "
            "in exactly that overlap. A **random-calendar placebo** (20 seeds × 500 draws) "
            "answers the model-free question directly: how often does an equally-sized random "
            "pick of days do this well? The third axis runs a **two-dummy HAC regression** — "
            "WVF's onset flag *and* a plain close-only drawdown-proxy's onset flag, same day — "
            "so WVF's coefficient is the marginal contribution *after* the obvious confound "
            "is already in the model."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Basket.** 8 tickers, 2000-01-03 → 2026-06-30 (IWM from 2000-05-26): SPY, "
            "QQQ, IWM (no survivorship) + AAPL, MSFT, JPM, XOM, JNJ (survivorship named).\n"
            "- **Indicator.** WVF(22), Bollinger(20, 2σ) onset trigger — one entry per "
            "capitulation episode, not one per day inside it.\n"
            "- **Execution.** Signal known at close *t*; enter *t+1* open; exit *h* sessions "
            "later's close. One documented lag, no second shift.\n"
            "- **Headline.** Pooled Welch *t* + per-ticker NW(*h*)-lag *t* at *h* ∈ {5,10,20}.\n"
            "- **Placebo.** 20 seeds × 500 draws of a random 10-day calendar, matched count.\n"
            "- **Third axis.** Two-dummy HAC(*h*=10): $r^{fwd}_{10} = a + b\\,S^{wvf}_t + "
            "c\\,S^{dd}_t$.\n"
            "- **Costs.** Round trip = 2 × one-way (5 / 10 bps) × NAV, HAC (auto-lag) *t* on "
            "the trade ledger.\n"
            "- **Control.** Synthetic OHLC with planted post-crash bounce; the null (bounce=0) "
            "must not fire across 20 seeds on the primary NW detector."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline split across horizons\n\n"
            "Pooled basket Welch *t*, plus the per-ticker Newey-West range (lags = horizon) "
            "as the overlap-robust cross-check."
        ),
        code(
            "hz = [5, 10, 20]\n"
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for h in hz:\n"
            "        ps, pr, nws = [], [], []\n"
            "        for t in data.TICKERS:\n"
            "            fr = FRAMES[t]\n"
            "            f = fr['onset_wvf'].to_numpy(bool)\n"
            "            y = fr[f'fwd_{h}'].to_numpy(float)\n"
            "            ps.append(y[f]); pr.append(y[~f])\n"
            "            nws.append(st.nw_dummy_stats(fr, h, 'onset_wvf'))\n"
            "        ps = np.concatenate(ps); pr = np.concatenate(pr)\n"
            "        rows.append((h, np.nanmean(ps)*1e4 - np.nanmean(pr)*1e4, st.welch_t(ps, pr),\n"
            "                     np.mean(nws), np.min(nws), np.max(nws)))\n"
            "else:\n"
            "    rows = [(h,) + (R['headline'][h][3], R['headline'][h][4], R['headline'][h][8],\n"
            "                    R['headline'][h][9], R['headline'][h][10]) for h in hz]\n"
            "gaps = [r[1] for r in rows]; ts = [r[2] for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "ax.bar([f'{h}d' for h in hz], gaps, color=[GREY if abs(t)<2 else RED for t in ts], width=.55)\n"
            "for i,(g,t) in enumerate(zip(gaps, ts)):\n"
            "    ax.annotate(f'{g:+.1f} bps\\nWelch t={t:+.2f}', (i,g), ha='center',\n"
            "        va='bottom' if g>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('spike-day forward return minus unconditional (bps)')\n"
            "ax.set_title('No horizon clears |t| >= 2 in the claimed direction')\n"
            "plt.tight_layout(); plt.show()\n"
            "for h, g, t, nm, nlo, nhi in rows:\n"
            "    print(f'h={h:2d}d  gap={g:+6.1f}bps  Welch t={t:+.2f}  NW mean={nm:+.2f} [{nlo:+.2f},{nhi:+.2f}]')"
        ),
        md(
            "> 💡 In plain words: at 5 days there's a faint positive gap that never clears the "
            "bar; by 10 and 20 days it's negative. The per-ticker NW range straddles zero at "
            "every horizon — some names positive, some negative, no consistent direction. "
            "That's exactly what noise looks like."
        ),
        md(
            "### 4b · Named exceptions and the multiple-comparisons check\n\n"
            "Two names look significant **at 5 days only**: SPY and QQQ."
        ),
        code(
            "names = ['SPY', 'QQQ']\n"
            "ts5 = [R['h5_spy'][1], R['h5_qqq'][1]]\n"
            "ts10 = [R['per_ticker']['SPY'][3], R['per_ticker']['QQQ'][3]]\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.2))\n"
            "x = np.arange(2); w = .35\n"
            "ax.bar(x - w/2, ts5, width=w, color=AMBER, label='5-day Welch t')\n"
            "ax.bar(x + w/2, ts10, width=w, color=GREY, label='10-day Welch t')\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels(names)\n"
            "ax.set_ylabel('Welch t'); ax.legend()\n"
            "ax.set_title('Significant at 5d, gone by 10d — 2 of 24 raw tests, as chance predicts')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('5d:', dict(zip(names, ts5)), ' 10d:', dict(zip(names, ts10)))"
        ),
        md(
            "> 💡 In plain words: 8 tickers × 3 horizons = 24 independent-ish raw tests; at a "
            "nominal 5% two-sided rate that's ≈1.2 false positives expected from pure noise. "
            "We see 2 — both fade to insignificance one horizon later. Nothing here survives a "
            "second look."
        ),
        md(
            "### 4c · The random-calendar placebo (h = 10)\n\n"
            "Does the observed spike-day mean beat a purely random calendar of equal size?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pool, spk = [], []\n"
            "    for t in data.TICKERS:\n"
            "        fr = FRAMES[t]\n"
            "        f = fr['onset_wvf'].to_numpy(bool)\n"
            "        y = fr['fwd_10'].to_numpy(float)\n"
            "        pool.append(y[~f]); spk.append(y[f])\n"
            "    pool = np.concatenate(pool); spk = np.concatenate(spk)\n"
            "    spk = spk[~np.isnan(spk)]\n"
            "    obs = float(np.nanmean(spk))\n"
            "    pl = st.placebo_pvalue(pool, obs, len(spk), n_draws_per_seed=200, n_seeds=4, base_seed=668)\n"
            "    draws_bps = None  # summary only in the light in-notebook run\n"
            "    obs_bps = obs * 1e4\n"
            "else:\n"
            "    obs_bps = R['placebo_obs']\n"
            "    pl = {'placebo_mean': R['placebo_mean']/1e4, 'placebo_sd': R['placebo_sd']/1e4}\n"
            "rng = np.random.default_rng(668)\n"
            "draws = rng.normal(pl['placebo_mean']*1e4, pl['placebo_sd']*1e4, 4000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='null: random calendars (light in-notebook run)')\n"
            "ax.axvline(obs_bps, c=RED, lw=2.5, label=f'observed spike-day mean {obs_bps:+.1f} bps')\n"
            "ax.set_xlabel('mean 10-day forward return of a random calendar (bps)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"canonical p = {R['placebo_p']:.3f} (20 seeds x 500 draws, results.md)\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"canonical (results.md): observed {R['placebo_obs']:+.1f} bps vs placebo \"\n"
            "      f\"{R['placebo_mean']:+.1f} bps (sd {R['placebo_sd']:.1f}), p = {R['placebo_p']:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: **p = {R['placebo_p']:.3f}** means {R['placebo_p']*100:.1f}% "
            "of random calendars do at least as well as the actual spike days. A real edge "
            "would sit far in the *left* tail of this histogram (a low *p*, like 637's FOMC "
            "crush at *p* = 0.00005); this one sits solidly inside the cloud, on the losing "
            "side of center."
        ),
        md(
            "### 4d · More than a drawdown proxy? — the third-axis regression\n\n"
            "$r^{fwd}_{10} = a + b\\,S^{wvf}_t + c\\,S^{dd}_t$, HAC(10) standard errors, one "
            "regression per ticker. $b$ is WVF's marginal contribution *after* the plain "
            "drawdown flag is already in the model."
        ),
        code(
            "tks = data.TICKERS\n"
            "if HAVE_REAL:\n"
            "    wm = {t: st.wick_marginal_stats(FRAMES[t], 10) for t in tks}\n"
            "    ov = [wm[t]['overlap_of_wvf']*100 for t in tks]\n"
            "    twvf = [wm[t]['t_wvf_marginal'] for t in tks]\n"
            "    tdd = [wm[t]['t_dd_marginal'] for t in tks]\n"
            "else:\n"
            "    ov = [R['wick'][t][0] for t in tks]\n"
            "    twvf = [R['wick'][t][1] for t in tks]\n"
            "    tdd = [R['wick'][t][2] for t in tks]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.4))\n"
            "a1.bar(tks, ov, color=AMBER, width=.6)\n"
            "a1.axhline(np.mean(ov), ls='--', c='k', lw=1)\n"
            "a1.set_ylabel('% overlap with the plain drawdown flag'); a1.tick_params(axis='x', rotation=45)\n"
            "a1.set_title(f'{np.mean(ov):.1f}% average overlap')\n"
            "x = np.arange(len(tks)); w = .35\n"
            "a2.bar(x - w/2, twvf, width=w, color=RED, label='marginal WVF t')\n"
            "a2.bar(x + w/2, tdd, width=w, color=GREY, label='marginal drawdown t')\n"
            "a2.axhline(0, c='k', lw=.8); a2.axhline(2, ls='--', c=RED, lw=1); a2.axhline(-2, ls='--', c=RED, lw=1)\n"
            "a2.set_xticks(x); a2.set_xticklabels(tks, rotation=45); a2.legend()\n"
            "a2.set_title('WVF adds nothing once the plain proxy is in the model')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('overlap:', dict(zip(tks, [round(v,1) for v in ov])))\n"
            "print('marginal WVF t:', dict(zip(tks, [round(v,2) for v in twvf])))\n"
            "print('marginal drawdown t:', dict(zip(tks, [round(v,2) for v in tdd])))"
        ),
        md(
            f"> 💡 In plain words: **{R['wick_avg_overlap']:.1f}%** of every WVF spike is a day "
            "the plain close-only drawdown flag already caught. Controlling for that overlap, "
            f"WVF's own marginal *t* averages **{R['wick_avg_t_wvf']:+.2f}** across the basket "
            "— never significant, and negative more often than positive. The one thing that "
            "makes WVF *look* like a volatility indicator rather than a level indicator — the "
            "intraday low — is not pulling its weight. **H₃ rejected; the third axis is "
            "`BUSTED`.**"
        ),
        md(
            "### 4e · The tradable timer — costs charged\n\n"
            "Every spike onset, held 10 sessions, one round trip = 2 × one-way cost × NAV; "
            "HAC (auto-lag) *t* on the trade ledger."
        ),
        code(
            "labels = ['gross', 'net 5bps', 'net 10bps']\n"
            "if HAVE_REAL:\n"
            "    ledgers = [st.timer_ledger(FRAMES[t], 10, 5.0, 'onset_wvf') for t in tks]\n"
            "    pooled = pd.concat(ledgers, ignore_index=True)\n"
            "    g = st.summarize_ledger(pooled, 'ret_gross')\n"
            "    n5 = st.summarize_ledger(pooled, 'ret_net')\n"
            "    ledgers10 = [st.timer_ledger(FRAMES[t], 10, 10.0, 'onset_wvf') for t in tks]\n"
            "    n10 = st.summarize_ledger(pd.concat(ledgers10, ignore_index=True), 'ret_net')\n"
            "    vals = [g['mean_bps'], n5['mean_bps'], n10['mean_bps']]\n"
            "    ts_ = [g['tstat'], n5['tstat'], n10['tstat']]\n"
            "else:\n"
            "    vals = [R['timer_gross'][1], R['timer_net5'][1], R['timer_net10'][1]]\n"
            "    ts_ = [R['timer_gross'][2], R['timer_net5'][4], R['timer_net10'][4]]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(labels, vals, color=[GREY if abs(t)<2 else RED for t in ts_], width=.55)\n"
            "for i,(v,t) in enumerate(zip(vals, ts_)):\n"
            "    ax.annotate(f'{v:+.1f} bps\\n(t={t:+.2f})', (i,v), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean return per trade (bps)')\n"
            "ax.set_title('Below the bar even gross of costs')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dict(zip(labels, zip(vals, ts_))))"
        ),
        md(
            f"> 💡 In plain words: gross of any cost the pooled timer's HAC *t* is "
            f"**{R['timer_gross'][2]:+.2f}** — already short of 2. Costs (5 / 10 bps one-way) "
            f"push it to **{R['timer_net5'][4]:+.2f}** and **{R['timer_net10'][4]:+.2f}**. "
            "There is no edge here for a market-maker's spread to eat — **Tradability = "
            "`MIRAGE`**."
        ),
        md(
            "### 4f · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic OHLC with occasional violent crash days (a wide low-wick — exactly the "
            "geometry WVF hunts for) and a TUNABLE planted post-crash bounce. Primary "
            "detector: the Newey-West *t* (accounts for the 10-day forward-return overlap)."
        ),
        code(
            "null_w, null_n = [], []\n"
            "for s_ in range(R['syn_seeds']):\n"
            "    sdf = data.synthetic_world(bounce=0.0, seed=668 + s_)\n"
            "    r = st.synthetic_detect(sdf, horizon=10)\n"
            "    null_w.append(r['welch_t']); null_n.append(r['nw_t'])\n"
            "null_w = np.asarray(null_w); null_n = np.asarray(null_n)\n"
            "sdf = data.synthetic_world(bounce=0.002, seed=668)\n"
            "planted = st.synthetic_detect(sdf, horizon=10)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(len(null_n)) + np.linspace(-.12,.12,len(null_n)), null_n, color=GREY, s=40,\n"
            "           label=f'null worlds (bounce=0), {R[\"syn_seeds\"]} seeds')\n"
            "ax.scatter([1], [planted['nw_t']], color=RED, s=90, zorder=5, label='planted bounce=+0.002/day')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('Newey-West t (spike vs rest, primary detector)')\n"
            "ax.set_title('Control: no null fires; a planted bounce lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null Welch: mean {null_w.mean():+.2f} (sd {null_w.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_w)>=2).sum()}/{R[\"syn_seeds\"]}')\n"
            "print(f'null NW:    mean {null_n.mean():+.2f} (sd {null_n.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_n)>=2).sum()}/{R[\"syn_seeds\"]}')\n"
            "print(f\"planted: Welch t={planted['welch_t']:+.2f}  NW t={planted['nw_t']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: across {R['syn_seeds']} null worlds the primary (NW) "
            f"detector averages t = {R['syn_null_nw_mean']:+.2f} (sd {R['syn_null_nw_sd']:.2f}) "
            f"and **never** crosses the bar ({R['syn_null_nw_fire']}/{R['syn_seeds']}); a "
            f"planted post-crash bounce reads NW t = {R['syn_planted_nw']:+.2f}. The machinery "
            "can find a real capitulation-bounce effect when one exists — it does not find one "
            "on the real tape. *(A faithful-engine / power check only — never cited in support "
            "of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal `NONE`** — pooled gap/Welch *t*: "
            f"5d {R['headline'][5][3]:+.1f} bps ({R['headline'][5][4]:+.2f}), "
            f"10d {R['headline'][10][3]:+.1f} bps ({R['headline'][10][4]:+.2f}), "
            f"20d {R['headline'][20][3]:+.1f} bps ({R['headline'][20][4]:+.2f}); no horizon "
            f"clears *t* ≥ 2 in the claimed direction, and the 10-day random-calendar placebo "
            f"has the observed mean losing to {R['placebo_p']*100:.1f}% of chance. Two names "
            "(SPY, QQQ) look significant at 5 days alone — expected noise from 24 raw tests, "
            "and neither survives to the next horizon.\n"
            f"- **Tradability `MIRAGE`** — the pooled 10-day timer's HAC *t* is "
            f"{R['timer_gross'][2]:+.2f} gross, {R['timer_net5'][4]:+.2f} at 5 bps, "
            f"{R['timer_net10'][4]:+.2f} at 10 bps: below the bar before costs are even "
            "charged.\n"
            f"- **\"More than a drawdown proxy?\" `BUSTED`** — {R['wick_avg_overlap']:.1f}% of "
            "every WVF spike coincides with a plain close-only drawdown flag; the intrabar "
            f"low's marginal contribution averages *t* = {R['wick_avg_t_wvf']:+.2f}, "
            "indistinguishable from zero. WVF's one extra ingredient over a one-line drawdown "
            "filter buys nothing this basket can detect."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general lesson is chart appeal versus predictive content.** A formula "
            "that reacts sharply to real panic (WVF does — that's not in question) is not the "
            "same as a formula that predicts what happens *after* the panic. This basket says "
            "the two are, here, unrelated.\n"
            "- **Where a wick-based signal could still matter:** markets without a clean, "
            "liquid close series — thin order books, gapping futures — where the intraday low "
            "genuinely carries information a close-only proxy can't reconstruct. This basket "
            "(SPY, QQQ, five liquid mega-caps) is the *least* likely place for that to show up; "
            "a thinner-liquidity basket is the natural sequel study.\n"
            "- **Dedup map:** [111-vix-term-structure](../../111-vix-term-structure/) (the "
            "real options-implied VIX curve), [92-easy-money](../../92-easy-money/) "
            "(VIX-futures carry), [127-williams-r](../../127-williams-r/) (Larry Williams' "
            "other, unrelated oscillator), and Bill Williams' "
            "[184-williams-fractals](../../184-williams-fractals/) / "
            "[421-williams-alligator](../../421-williams-alligator/) (different trader, "
            "different question).\n\n"
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
