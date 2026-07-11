"""Generate the two narrative notebooks for Study 679 (Psychological Line).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached SPY + basket
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance SPY/QQQ/IWM/AAPL/
# TSLA/NVDA, 2003-01-02 -> 2026-06-30; window=12, thresholds 25/75, hold=5 trading days).
R = dict(
    window=12, oversold=25, overbought=75, hold=5,
    start="2003-01-02", end="2026-06-30",
    n_low=97, mean_low=159.13, welch_low=1.32, nw_low=1.64,
    n_mid=33074, mean_mid=45.87,
    n_high=398, mean_high=66.09, welch_high=1.04, nw_high=1.10,
    per_ticker={
        "SPY": (-0.20, 0.60), "QQQ": (1.71, -1.28), "IWM": (0.37, 0.57),
        "AAPL": (0.63, 0.41), "TSLA": (1.10, 1.21), "NVDA": (-0.08, 0.68),
    },
    timer_n=495,
    timer5_strat_win=41.0, timer5_strat_mean=-31.95, timer5_strat_t=-1.49,
    timer5_rand_win=50.3, timer5_rand_mean=18.57, timer5_rand_t=0.69,
    timer10_strat_win=39.6, timer10_strat_mean=-41.95, timer10_strat_t=-1.95,
    timer10_rand_win=48.7, timer10_rand_mean=8.57, timer10_rand_t=0.32,
    delta_bps=-50.52, delta_t=-1.55,
    grid=[
        (10, 20, 80, 57, 1.29, 234, 1.15), (10, 25, 75, 354, 1.33, 949, 1.30),
        (10, 30, 70, 354, 1.33, 949, 1.30), (12, 20, 80, 97, 1.32, 398, 1.04),
        (12, 25, 75, 97, 1.32, 398, 1.04), (12, 30, 70, 454, 2.47, 1175, 0.87),
        (14, 20, 80, 19, 0.53, 162, 0.79), (14, 25, 75, 134, 0.80, 567, -0.35),
        (14, 30, 70, 534, 1.72, 1373, 0.49), (20, 20, 80, 0, None, 31, 0.55),
        (20, 25, 75, 13, 2.60, 158, 0.30), (20, 30, 70, 66, 2.17, 427, 0.98),
    ],
    syn_null_lo_mean=-0.19, syn_null_lo_sd=1.02, syn_null_lo_fire=1,
    syn_null_hi_mean=-0.19, syn_null_hi_sd=1.13, syn_null_hi_fire=1,
    syn_planted_lo=7.03, syn_planted_hi=-7.47,
    fp=dict(SPY="c13d94f59e60", QQQ="1bf28819a80b", IWM="239b32ae95c3",
            AAPL="c3158a02623a", TSLA="1d273fa4b475", NVDA="b6f76f845536"),
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Beats_a_coin%3F: Mixed](https://img.shields.io/badge/Beats_a_coin%3F-Mixed-8b949e?style=flat-square)\n\n"
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

from psychological_line import data, strategy as st

HAVE_REAL = data.have_real()
BARS = data.load_real() if HAVE_REAL else None
print("real cache present:", HAVE_REAL,
      "| universe:", ", ".join(data.UNIVERSE) if HAVE_REAL else "n/a")
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Can you time the market by counting up days? 🧮📈\n"
            "### The Japanese Psychological Line — a decades-old crowd-sentiment gauge, "
            "put through the desk's honest teardown\n\n"
            + BADGES +
            "The **Psychological Line** does one very simple thing: over the last 12 trading "
            "days, what fraction closed up? Above 75%, the story goes, \"almost everyone who "
            "wanted to buy already has\" — the crowd is exhausted, sell. Below 25%, the "
            "opposite — everyone's already sold, buy the capitulation.\n\n"
            "It's an appealing idea: a pure vote-count of the crowd's recent mood, no price "
            "levels, no math beyond a fraction. We test it exactly as the charting books "
            "describe it — 12-day window, 75/25 thresholds — on SPY plus five liquid names, "
            "23 years of daily data.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the parameter grid and the "
            "cost math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** Every trade in this study enters the day *after* the signal, "
            "at that day's open — no trading on information you didn't have yet. House style "
            "in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do stocks bounce after an oversold PSY reading? | **Maybe, but too faintly to "
            f"trust.** After PSY drops below 25, the next 5 days average **+{R['mean_low']:.0f} "
            f"bps** vs +{R['mean_mid']:.0f} bps normally — the right direction, but with only "
            f"{R['n_low']} such events in 23 years, the signal is statistically indistinguishable "
            "from noise. |\n"
            "| Do stocks fall after an overbought PSY reading? | **No — if anything, the "
            f"opposite.** After PSY tops 75, the next 5 days average **+{R['mean_high']:.0f} "
            "bps** — *more* positive than normal, not less. The \"exhausted buyers\" story "
            "doesn't show up in the data. |\n"
            "| Can you trade it? | **Not profitably.** Buying oversold and shorting overbought, "
            f"held 5 days, loses **{R['timer5_strat_mean']:.0f} bps per trade** after realistic "
            "costs — worse the more you pay in fees. |\n"
            f"| Would a coin flip do better? | **On paper, yes.** A random buy-or-sell on the "
            f"exact same days made **+{R['timer5_rand_mean']:.0f} bps** where the PSY rule lost "
            f"{abs(R['timer5_strat_mean']):.0f} — though that gap itself isn't quite big enough "
            "to call statistically certain. |\n\n"
            "> Counting up-days turns out to be a very blunt instrument."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When almost every recent day has closed up, the buyers who were going to buy "
            "have already bought — the next move is down. When almost every recent day has "
            "closed down, the sellers are spent — the next move is up.\"*\n\n"
            "It's one of the older members of the Japanese technical-analysis toolkit — Steve "
            "Nison's *Beyond Candlesticks* carried it to English-speaking traders alongside "
            "candlestick patterns in the 1990s. Unlike most oscillators, PSY throws away *how "
            "much* each day moved and keeps only its sign — a pure vote count."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a simple up/down tally over 12 days genuinely predicted the next week's "
            "direction, that would be a remarkable finding — no price magnitude, no volume, "
            "no volatility adjustment, just a coin-flip tally beating a coin flip. It would "
            "also be trivially easy for anyone to compute and trade, which is exactly why, if "
            "real, it shouldn't still be sitting there unexploited 30+ years later."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The rule, exactly as written.** PSY(12) = 100 x (up-closes in the last 12 "
            f"days) / 12. Below {R['oversold']} = oversold/buy; above {R['overbought']} = "
            "overbought/sell — the textbook thresholds, not tuned.\n"
            "- **The comparison.** The 5-trading-day forward return after each signal vs. every "
            "other day, on SPY, QQQ, IWM, AAPL, TSLA and NVDA, 2003→2026.\n"
            "- **The coin check.** The identical entry days and dates, but with the buy/sell "
            "decision made by a fair coin instead of PSY — the fairest test of whether the rule "
            "adds anything.\n"
            "- **The luck check.** A synthetic world where we know the true answer, to prove the "
            "measuring stick itself isn't broken."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** What happens in the 5 days *after* PSY hits an extreme?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = st.pooled_conditional(BARS, window=R['window'], oversold=R['oversold'],\n"
            "                             overbought=R['overbought'], h=R['hold'])\n"
            "    lo, mid, hi = c['mean_low_bps'], c['mean_mid_bps'], c['mean_high_bps']\n"
            "else:\n"
            "    lo, mid, hi = R['mean_low'], R['mean_mid'], R['mean_high']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "labels = ['oversold\\n(PSY<25)', 'neutral', 'overbought\\n(PSY>75)']\n"
            "vals = [lo, mid, hi]\n"
            "ax.bar(labels, vals, color=[GREEN, GREY, RED], width=.6)\n"
            "for i, v in enumerate(vals): ax.annotate(f'{v:+.0f} bps', (i, v), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean forward return, next 5 trading days (bps)')\n"
            "ax.set_title('All three buckets drift up — the zones barely change the story')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'oversold {lo:+.1f}  neutral {mid:+.1f}  overbought {hi:+.1f} bps')"
        ),
        md(
            f"Oversold days *do* edge out the neutral bucket (**+{R['mean_low']:.0f}** vs "
            f"**+{R['mean_mid']:.0f}** bps) — but with only **{R['n_low']}** such events in 23 "
            f"years across six tickers, the statistics call it noise (Welch *t* = "
            f"{R['welch_low']:.2f}; the bar for \"real\" is 2). And overbought days? They "
            f"average **+{R['mean_high']:.0f}** bps — *more* positive, not less. The market "
            "just tends to drift up in this window (23 years, mostly a bull market), and PSY "
            "extremes don't visibly change that drift either way.\n\n"
            "**Now, the actual trade.** Would buying the oversold dips and shorting the "
            "overbought spikes have made money?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    strat, rand = st.pooled_trade_ledger(BARS, window=R['window'], oversold=R['oversold'],\n"
            "                                         overbought=R['overbought'], hold_days=R['hold'],\n"
            "                                         cost_bps=5.0, seed=679)\n"
            "    ss, rs = st.summarize(strat, 'ret_net', lags=R['hold']), st.summarize(rand, 'ret_net', lags=R['hold'])\n"
            "    sm, rm = ss['mean_bps'], rs['mean_bps']\n"
            "else:\n"
            "    sm, rm = R['timer5_strat_mean'], R['timer5_rand_mean']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.4))\n"
            "ax.bar(['PSY timer\\n(buy low / sell high)', 'coin flip\\n(same days, random side)'],\n"
            "       [sm, rm], color=[RED, GREY], width=.55)\n"
            "for i, v in enumerate([sm, rm]): ax.annotate(f'{v:+.1f} bps', (i, v), ha='center',\n"
            "    va='top' if v < 0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean return per trade, net of costs (bps)')\n"
            "ax.set_title('The rule loses money where a coin flip, on the same days, does not')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'PSY timer {sm:+.1f} bps/trade  |  coin {rm:+.1f} bps/trade')"
        ),
        md(
            f"The PSY rule loses **{R['timer5_strat_mean']:.0f} bps per trade** (5 bps costs, "
            f"495 trades) — while a coin flip on the *exact same* entry days made "
            f"**+{R['timer5_rand_mean']:.0f}**. That's a **{abs(R['delta_bps']):.0f} bps/trade** "
            f"gap favoring the coin — real money on paper, though (the quants notebook shows) "
            "not quite big enough a sample to call it *certain* the coin truly beats the rule "
            "going forward. Either way, the rule itself is a net loser, and it gets worse the "
            "more you pay in trading costs."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Neither the oversold nor the overbought leg reaches "
            "statistical significance, every individual stock/ETF checked out clean, and the "
            "overbought leg points the *wrong* way for the claim.\n"
            "- **Tradability — Mirage.** Actually trading the zone crosses loses money net of "
            "costs, at every cost level tested.\n"
            "- **\"Beats a coin?\" — Mixed.** The coin wins on paper by a wide margin, but the "
            "sample isn't quite large enough to certify that gap as more than luck."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **PSY throws away information on purpose** — no magnitude, no range, no volume. "
            "That's what makes it simple, and probably also what makes it weak: the desk's "
            "other oscillator studies, which *do* use magnitude or range, land in the same "
            "\"None/Mirage\" place anyway (see the dedup list below), so it's not obviously "
            "PSY's simplicity alone that's the problem — it may be that short-horizon "
            "\"crowd exhaustion\" just isn't a reliable phenomenon on liquid daily US equities.\n"
            "- **Sibling studies:** [107-stochastic-oscillator](../../107-stochastic-oscillator/), "
            "[127-williams-r](../../127-williams-r/), [179-aroon](../../179-aroon/) and "
            "[680-disparity-index](../../680-disparity-index/) all test close relatives — none "
            "of them found a certifiable, tradable edge either.\n\n"
            "*Think a longer window, an intraday version, or a volatility-scaled variant would "
            "fare better? The window x threshold grid in the quants notebook is the place to "
            "start — and the parameter-mining trap it walks through is worth reading first.*"
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
            "# The Psychological Line — a quantitative teardown 🔬\n"
            "### Trigger-event design and why it matters · Welch/HAC splits on both legs · "
            "the window x threshold grid and its parameter-mined corners · a random-direction "
            "control · the cost sweep · a 20-seed synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "PSY(*N*) = 100 x (up-closes in the last *N* days) / *N*, tested exactly as the "
            "charting books state it: *N* = 12, thresholds 75/25. The job here is to measure "
            "it with a design that doesn't fool itself, then ask the only question that pays: "
            "*is any of it tradable?*\n\n"
            "> ⚠️ **Data note.** SPY/QQQ/IWM/AAPL/TSLA/NVDA daily OHLC (total-return adjusted), "
            f"{R['start']} → {R['end']}, yfinance, cached. No survivorship on the index sleeve; "
            "the mega-cap sleeve is three well-known liquid names, **not** a re-derived "
            "historical panel (named). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprints " +
            ", ".join(f"`{t}={fp}`" for t, fp in R["fp"].items()) + ").\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | oversold **+{R['mean_low']:.0f} bps/5d** (Welch "
            f"**t={R['welch_low']:.2f}**, NW **t={R['nw_low']:.2f}**, n={R['n_low']}); "
            f"overbought **+{R['mean_high']:.0f} bps/5d** (Welch **t={R['welch_high']:.2f}**, "
            f"NW **t={R['nw_high']:.2f}**, n={R['n_high']}) — wrong sign for the claim |\n"
            f"| **Tradability** | `MIRAGE` | zone timer **{R['timer5_strat_mean']:.1f} "
            f"bps/trade** net (5 bps, HAC t={R['timer5_strat_t']:.2f}); "
            f"**{R['timer10_strat_mean']:.1f} bps** at 10 bps (t={R['timer10_strat_t']:.2f}) |\n"
            f"| **Beats a coin?** | `MIXED` | coin **+{R['timer5_rand_mean']:.1f} bps** vs "
            f"PSY **{R['timer5_strat_mean']:.1f} bps** (Δ={R['delta_bps']:.1f} bps, Welch "
            f"t={R['delta_t']:.2f}) — directionally decisive, not statistically certified |\n\n"
            "> 💡 In plain words: a pure up/down-day tally doesn't buy back the information a "
            "magnitude- or range-aware oscillator would carry — and none of those carry much "
            "either (see the dedup list)."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $u_t \\in \\{0,1\\}$ flag an up-close and $\\text{PSY}_t(N) = 100 \\cdot "
            "\\frac{1}{N}\\sum_{i=0}^{N-1} u_{t-i}$ the trailing share of up-closes, known at "
            "the close of bar $t$. The claims:\n\n"
            "- **H₁ (oversold bounce).** $E[r_{t \\to t+h} \\mid \\text{PSY}_t < 25] > "
            "E[r_{t \\to t+h}]$ — a genuine, tradable-strength bounce.\n"
            "- **H₂ (overbought pullback).** $E[r_{t \\to t+h} \\mid \\text{PSY}_t > 75] < "
            "E[r_{t \\to t+h}]$ — a genuine pullback.\n"
            "- **H₃ (tradable).** A long-the-lows/short-the-highs timer, entered at $t{+}1$'s "
            "open and held $h$ days, beats a same-bar random-direction control net of costs.\n\n"
            "We find **H₁ directionally right but uncertified** (Welch "
            f"*t* = {R['welch_low']:.2f}), **H₂ rejected on sign** (point estimate is "
            "*positive*, not negative), and **H₃ not certified either way** (Δ *t* = "
            f"{R['delta_t']:.2f}, just under the |2| bar)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design, and a trap we had to design around\n\n"
            "PSY is a **rolling** window, so a naive \"score every day PSY is inside the zone\" "
            "design pools heavily overlapping observations — consecutive in-zone days share "
            "nearly the same underlying 12 closes, and their forward-return windows overlap "
            "too. On a synthetic null world (no real structure whatsoever) that naive design "
            "pushed a Welch *t*'s false-positive rate from a nominal ~5% to **20–30%** across "
            "seeds — a silent inflation that would have made this study look more \"real\" "
            "than it is. The fix used throughout: **trigger events only** (the day PSY first "
            "*enters* a zone) with a cooldown equal to the hold period before the next entry "
            "of the same side counts — this brings the null's false-positive rate back to "
            "**1/20 seeds**, right at nominal (see 4e below). Welch *t* is the planned primary "
            "on these near-non-overlapping events; a Newey-West (5-lag) *t* is the residual "
            "overlap cross-check."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** SPY, QQQ, IWM, AAPL, TSLA, NVDA, {R['start']} → {R['end']} "
            "(TSLA from its 2010-06 inception). Total-return adjusted daily OHLC.\n"
            f"- **Rule.** PSY({R['window']}), thresholds [{R['oversold']}, {R['overbought']}], "
            f"hold = {R['hold']} trading days.\n"
            "- **Execution.** One lag: signal known at close *t*, enter next open (*t+1*), "
            "exit close of *t+h*.\n"
            "- **Headline.** Pooled trigger-day Welch *t* + NW(*h*) *t*, both legs vs the "
            "unconditional (non-trigger) rest, plus a per-instrument breakdown.\n"
            "- **Trade test.** Zone-trigger ledger, one-way costs x 2 legs, vs a "
            "random-direction control on identical entries (`seed=679`).\n"
            "- **Robustness.** Window x (oversold, overbought) grid, 12 combinations.\n"
            "- **Control.** Synthetic tape, drift switches on exactly when *that day's* causal "
            "PSY sits in the tested zone — the literal claim planted on the literal condition; "
            "the null must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline split — both legs, pooled universe\n\n"
            "Trigger-day forward 5-day return vs the unconditional rest, Welch *t* (primary) "
            "and NW(5) *t* (overlap cross-check)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = st.pooled_conditional(BARS, window=R['window'], oversold=R['oversold'],\n"
            "                             overbought=R['overbought'], h=R['hold'])\n"
            "    n_lo, n_hi = c['n_low'], c['n_high']\n"
            "    m_lo, m_mid, m_hi = c['mean_low_bps'], c['mean_mid_bps'], c['mean_high_bps']\n"
            "    t_lo, t_hi = c['welch_t_low'], c['welch_t_high']\n"
            "    nw_lo, nw_hi = c['nw_t_low'], c['nw_t_high']\n"
            "else:\n"
            "    n_lo, n_hi = R['n_low'], R['n_high']\n"
            "    m_lo, m_mid, m_hi = R['mean_low'], R['mean_mid'], R['mean_high']\n"
            "    t_lo, t_hi = R['welch_low'], R['welch_high']\n"
            "    nw_lo, nw_hi = R['nw_low'], R['nw_high']\n"
            "print(f'oversold  (n={n_lo}): {m_lo:+.2f} bps/5d  Welch t={t_lo:+.2f}  NW(5) t={nw_lo:+.2f}')\n"
            "print(f'neutral         : {m_mid:+.2f} bps/5d')\n"
            "print(f'overbought(n={n_hi}): {m_hi:+.2f} bps/5d  Welch t={t_hi:+.2f}  NW(5) t={nw_hi:+.2f}')\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "labels = ['oversold', 'neutral', 'overbought']\n"
            "vals = [m_lo, m_mid, m_hi]\n"
            "ts = [t_lo, None, t_hi]\n"
            "cols = [GREEN if abs(t_lo) >= 2 else GREY, GREY, RED if abs(t_hi) >= 2 else GREY]\n"
            "ax.bar(labels, vals, color=cols, width=.6)\n"
            "for i, v in enumerate(vals): ax.annotate(f'{v:+.0f} bps', (i, v), ha='center', va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean forward return, 5d (bps)')\n"
            "ax.set_title(f'Neither leg clears |t|>=2 (dashed) — grey bars = uncertified')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: the oversold leg is directionally consistent with the claim "
            f"(+{R['mean_low']:.0f} vs +{R['mean_mid']:.0f} bps) but *t* = {R['welch_low']:.2f} "
            f"< 2 — not certifiable with n = {R['n_low']}. The overbought leg (+{R['mean_high']:.0f} "
            "bps) runs in the *wrong* direction for a sell signal, though again *t* = "
            f"{R['welch_high']:.2f} < 2 means we can't even certify that it's really positive — "
            "only that it certainly isn't the large negative number the claim predicts."
        ),
        md(
            "### 4b · Per-instrument breakdown\n\n"
            "No single name or ETF carries a certifiable edge either way:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = {}\n"
            "    for t_, bars in BARS.items():\n"
            "        r = st.pooled_conditional({t_: bars}, window=R['window'], oversold=R['oversold'],\n"
            "                                  overbought=R['overbought'], h=R['hold'])\n"
            "        rows[t_] = (r['welch_t_low'], r['welch_t_high'])\n"
            "else:\n"
            "    rows = R['per_ticker']\n"
            "tickers = list(rows)\n"
            "lo_ts = [rows[t_][0] for t_ in tickers]\n"
            "hi_ts = [rows[t_][1] for t_ in tickers]\n"
            "x = np.arange(len(tickers)); w = 0.38\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "ax.bar(x - w/2, lo_ts, width=w, color=GREEN, label='oversold leg t')\n"
            "ax.bar(x + w/2, hi_ts, width=w, color=RED, label='overbought leg t')\n"
            "ax.axhline(2, ls='--', c='k', lw=1); ax.axhline(-2, ls='--', c='k', lw=1)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels(tickers)\n"
            "ax.set_ylabel('Welch t vs unconditional')\n"
            "ax.set_title('No instrument clears the |t|>=2 bar on either leg')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for t_ in tickers: print(f'{t_:>5s}: low t={rows[t_][0]:+.2f}  high t={rows[t_][1]:+.2f}')"
        ),
        md(
            "> 💡 In plain words: pooling six names could in principle manufacture a false "
            "positive if one wild outlier ticker dominates the pool. It doesn't — every "
            "instrument, checked alone, is quiet on both legs."
        ),
        md(
            "### 4c · The trade — zone-trigger timer vs a random-direction control\n\n"
            "Enter next open on the trigger, hold 5 days, one-way cost x 2 per round trip; "
            "pinned against a coin flip on the identical entry bars and dates."
        ),
        code(
            "rows5, rows10 = [], []\n"
            "if HAVE_REAL:\n"
            "    for cb in (5.0, 10.0):\n"
            "        strat, rand = st.pooled_trade_ledger(BARS, window=R['window'], oversold=R['oversold'],\n"
            "                                             overbought=R['overbought'], hold_days=R['hold'],\n"
            "                                             cost_bps=cb, seed=679)\n"
            "        ss, rs = st.summarize(strat, 'ret_net', lags=R['hold']), st.summarize(rand, 'ret_net', lags=R['hold'])\n"
            "        (rows5 if cb == 5.0 else rows10).extend([ss, rs])\n"
            "    sm5, rm5, st5, rt5 = rows5[0]['mean_bps'], rows5[1]['mean_bps'], rows5[0]['tstat'], rows5[1]['tstat']\n"
            "    sm10, rm10 = rows10[0]['mean_bps'], rows10[1]['mean_bps']\n"
            "else:\n"
            "    sm5, rm5, st5, rt5 = R['timer5_strat_mean'], R['timer5_rand_mean'], R['timer5_strat_t'], R['timer5_rand_t']\n"
            "    sm10, rm10 = R['timer10_strat_mean'], R['timer10_rand_mean']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.bar(['PSY timer', 'coin'], [sm5, rm5], color=[RED, GREY], width=.55)\n"
            "for i, v in enumerate([sm5, rm5]): a1.annotate(f'{v:+.1f}', (i, v), ha='center', va='top' if v < 0 else 'bottom')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_title(f'At 5 bps cost (HAC t={st5:+.2f} / {rt5:+.2f})')\n"
            "a1.set_ylabel('bps/trade, net')\n"
            "a2.bar(['5 bps', '10 bps'], [sm5, sm10], color=[AMBER, RED], width=.5)\n"
            "for i, v in enumerate([sm5, sm10]): a2.annotate(f'{v:+.1f}', (i, v), ha='center', va='top')\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_title('PSY timer only: worse at higher cost')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'5bps: PSY {sm5:+.1f} vs coin {rm5:+.1f} bps/trade')\n"
            "print(f'10bps: PSY {sm10:+.1f} vs coin {rm10:+.1f} bps/trade')"
        ),
        md(
            f"> 💡 In plain words: at 5 bps the PSY timer loses "
            f"**{R['timer5_strat_mean']:.1f} bps/trade** (HAC *t* = {R['timer5_strat_t']:.2f}) "
            f"while the coin makes **+{R['timer5_rand_mean']:.1f}** (*t* = "
            f"{R['timer5_rand_t']:.2f}) — a gap of {abs(R['delta_bps']):.1f} bps, Welch "
            f"*t* = {R['delta_t']:.2f} on the delta itself: directionally the coin wins clearly, "
            "but the gap's own significance falls just short of the |2| certification bar "
            "(hence `MIXED`, not `BUSTED`, on the front card). Costs only make the PSY side "
            f"worse: {R['timer5_strat_mean']:.1f} -> {R['timer10_strat_mean']:.1f} bps/trade "
            "from 5 to 10 bps — there is no cost level at which the gross number turns "
            "positive."
        ),
        md(
            "### 4d · Parameter robustness — and the discretization footgun\n\n"
            "Window x (oversold, overbought), 12 combinations. The textbook cell (12, 25/75) "
            "is highlighted."
        ),
        code(
            "if HAVE_REAL:\n"
            "    grid = st.param_grid(BARS, h=R['hold'])\n"
            "    rows = list(zip(grid['window'], grid['oversold'], grid['overbought'],\n"
            "                    grid['n_low'], grid['welch_t_low'], grid['n_high'], grid['welch_t_high']))\n"
            "else:\n"
            "    rows = R['grid']\n"
            "labels = [f\"w{int(w)}\\n[{int(lo)},{int(hi)}]\" for w, lo, hi, *_ in rows]\n"
            "t_lo = [r[4] for r in rows]; t_hi = [r[6] for r in rows]\n"
            "x = np.arange(len(rows)); wdt = 0.38\n"
            "fig, ax = plt.subplots(figsize=(11.5, 4.6))\n"
            "ax.bar(x - wdt/2, [0 if (t is None or (isinstance(t, float) and np.isnan(t))) else t for t in t_lo], width=wdt, color=GREEN, label='oversold t')\n"
            "ax.bar(x + wdt/2, t_hi, width=wdt, color=RED, label='overbought t')\n"
            "ax.axhline(2, ls='--', c='k', lw=1); ax.axhline(-2, ls='--', c='k', lw=1)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)\n"
            "ax.set_ylabel('Welch t'); ax.set_title('The textbook cell (w12, [25,75]) is mid-pack — a few corners cross the bar, one-sided')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            "> 💡 In plain words: a few off-textbook corners cross |*t*| >= 2 on the oversold "
            "leg alone (window 12/[30,70], window 20/[25,75] and [30,70]) — the classic shape "
            "of a parameter-mined false positive: one-sided (never both legs at once), never "
            "the pre-registered rule, and the *n* keeps shrinking as the window grows (window "
            "20 has as few as 0 oversold events). We also flag a discretization footgun: PSY(*N*) "
            "only takes steps of 100/*N* % — for *N* = 10 or 12, several \"different\" threshold "
            "pairs select the *exact same* set of days, because no achievable PSY value falls "
            "between them. A robustness sweep that doesn't account for this can look wider than "
            "it actually is."
        ),
        md(
            "### 4e · Faithful-engine & power control — and the pseudo-replication trap it caught\n\n"
            "Synthetic tape: i.i.d. daily noise plus a drift that switches on exactly when "
            "*that day's* causal PSY(12) sits in the tested zone — the literal claim, planted "
            "on the literal condition. Each seed pools a 6-series basket (mirrors the real "
            "6-ticker universe)."
        ),
        code(
            "null_lo, null_hi = [], []\n"
            "for s_ in range(20):\n"
            "    basket = data.synthetic_basket(reversal=0.0, seed=679 + s_)\n"
            "    r = st.pooled_conditional(basket, window=R['window'], oversold=R['oversold'],\n"
            "                             overbought=R['overbought'], h=R['hold'])\n"
            "    null_lo.append(r['welch_t_low']); null_hi.append(r['welch_t_high'])\n"
            "null_lo, null_hi = np.asarray(null_lo), np.asarray(null_hi)\n"
            "planted = data.synthetic_basket(reversal=0.006, seed=679)\n"
            "rp = st.pooled_conditional(planted, window=R['window'], oversold=R['oversold'],\n"
            "                           overbought=R['overbought'], h=R['hold'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.scatter(np.zeros(20) - .1 + np.linspace(-.06, .06, 20), null_lo, color=GREEN, s=36, label='null, oversold leg (20 seeds)')\n"
            "ax.scatter(np.zeros(20) + .1 + np.linspace(-.06, .06, 20), null_hi, color=RED, s=36, marker='^', label='null, overbought leg (20 seeds)')\n"
            "ax.scatter([1], [rp['welch_t_low']], color=GREEN, s=110, edgecolor='k', zorder=5, label='planted, oversold leg')\n"
            "ax.scatter([1.2], [rp['welch_t_high']], color=RED, s=110, marker='^', edgecolor='k', zorder=5, label='planted, overbought leg')\n"
            "ax.axhline(2, ls='--', c='k', lw=1); ax.axhline(-2, ls='--', c='k', lw=1)\n"
            "ax.set_xticks([0, 1.1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('Welch t'); ax.set_title('No null fires (~nominal rate); a planted reversal lights up hard')\n"
            "ax.legend(fontsize=8); plt.tight_layout(); plt.show()\n"
            "print(f'null: lo mean={null_lo.mean():+.2f} sd={null_lo.std(ddof=1):.2f} |t|>=2: {(abs(null_lo)>=2).sum()}/20')\n"
            "print(f'null: hi mean={null_hi.mean():+.2f} sd={null_hi.std(ddof=1):.2f} |t|>=2: {(abs(null_hi)>=2).sum()}/20')\n"
            "print(f'planted: lo t={rp[\"welch_t_low\"]:+.2f}  hi t={rp[\"welch_t_high\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with the trigger-event + cooldown design, the null world "
            f"fires at **{R['syn_null_lo_fire']}/20** (oversold) and **{R['syn_null_hi_fire']}/20** "
            "(overbought) seeds — right at the nominal ~5% rate a well-calibrated Welch *t* "
            f"should show. A planted reversal the same size as the claim lights up unmistakably "
            f"(*t* = {R['syn_planted_lo']:.2f} / {R['syn_planted_hi']:.2f}). Worth naming: the "
            "naive \"score every in-zone day, no cooldown\" version of this same detector pushed "
            "the null's false-positive rate to **20–30%** — a silent inflation this study would "
            "have shipped with had the control not caught it. The real tape's flat headline "
            "result above is a genuine flat result, not a broken ruler."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — oversold +{R['mean_low']:.0f} bps/5d (Welch "
            f"*t* = {R['welch_low']:.2f}, NW *t* = {R['nw_low']:.2f}, n = {R['n_low']}); "
            f"overbought +{R['mean_high']:.0f} bps/5d (Welch *t* = {R['welch_high']:.2f}, NW "
            f"*t* = {R['nw_high']:.2f}, n = {R['n_high']}, **wrong sign** for the claim). No "
            "instrument individually clears the bar either way. A handful of parameter-grid "
            "corners cross the bar, but one-sided and off the pre-registered rule — a "
            "data-mining shape.\n"
            f"- **Tradability `MIRAGE`** — the zone-trigger timer loses "
            f"{R['timer5_strat_mean']:.1f} bps/trade net at 5 bps cost (HAC *t* = "
            f"{R['timer5_strat_t']:.2f}), {R['timer10_strat_mean']:.1f} bps at 10 bps "
            f"(*t* = {R['timer10_strat_t']:.2f}). No positive break-even cost exists.\n"
            f"- **\"Beats a coin?\" `MIXED`** — coin +{R['timer5_rand_mean']:.1f} bps vs PSY "
            f"{R['timer5_strat_mean']:.1f} bps ({abs(R['delta_bps']):.1f} bps gap, Welch "
            f"*t* = {R['delta_t']:.2f} on the delta) — directionally decisive, statistically "
            "just short of certified."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The design lesson generalizes.** Any rolling-window oscillator study is "
            "vulnerable to the pseudo-replication trap in §4e — scoring every in-zone day "
            "instead of trigger events with a cooldown will quietly inflate significance on "
            "*any* such indicator, real effect or not. Worth re-auditing the desk's other "
            "oscillator studies against this design if they score every in-zone day.\n"
            "- **What might revive PSY** (untested here, an open door): a longer window "
            "(diminishing returns visible in the grid — window 20 starves the oversold leg of "
            "events), a volatility-scaled variant, or conditioning on the *level* of the "
            "market regime (bull vs bear) rather than pooling 23 mostly-bullish years.\n"
            "- **Dedup map:** [107-stochastic-oscillator](../../107-stochastic-oscillator/) "
            "(range position, magnitude-weighted), [127-williams-r](../../127-williams-r/) "
            "(same range family, inverted sign), [179-aroon](../../179-aroon/) (recency since "
            "an extreme, not a frequency count), [680-disparity-index](../../680-disparity-index/) "
            "(close-vs-moving-average deviation) — none of them found a certifiable edge on "
            "this desk's protocol either.\n\n"
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
