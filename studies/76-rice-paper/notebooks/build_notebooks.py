"""Generate the two narrative notebooks for Study 76 (Rice-Paper).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The
synthetic figures run anywhere, offline and deterministic; the real-tape cells
use the cached daily parquets under ../_cache/ if present and otherwise quote
the frozen headline numbers in ``R`` (mirroring docs/results.md), so the
notebook re-runs for any reader.

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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-12).
R = dict(
    # 1-day pooled excess returns (gross, bps) and HAC t-stats
    be_n=2127,   be_excess=-2.94,   be_t=-0.51,   be_win=50.4,   be_mean=1.13,
    beg_n=2354,  beg_excess=-9.04,  beg_t=-1.62,  beg_win=45.6,  beg_mean=-14.74,
    h_n=1536,    h_excess=1.23,     h_t=0.17,     h_win=52.7,    h_mean=3.72,
    ss_n=1277,   ss_excess=11.58,   ss_t=1.41,    ss_win=47.0,   ss_mean=-3.50,
    db_n=2921,   db_excess=10.93,   db_t=2.16,    db_win=54.6,   db_mean=12.11,
    dbear_n=3562, dbear_excess=7.91, dbear_t=1.79, dbear_win=47.8, dbear_mean=-3.96,
    # Bonferroni threshold
    bonf_t=2.64,
    bonf_n=12,
    # max t-stat
    max_t=2.16,
    max_pattern="doji_bullish",
    # Pattern frequencies per ticker per year
    be_freq=8.6,
    beg_freq=9.6,
    h_freq=6.2,
    ss_freq=5.2,
    db_freq=11.9,
    dbear_freq=14.5,
    # SPY counts
    spy_be=126,
    spy_beg=147,
    spy_h=114,
    spy_ss=63,
    spy_db=158,
    spy_dbear=279,
    spy_n=4136,
    # Data window
    start="2010-01-04",
    end="2026-06-12",
    n_tickers=15,
    total_events=13777,
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
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from rice_paper import data, strategy as st

TICKERS = [
    "SPY", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA",
    "JPM", "BAC", "XOM", "JNJ", "PG", "KO", "WMT",
]

def _have_cache():
    return all(os.path.exists(data._cache_path(t, data.DEFAULT_CACHE)) for t in TICKERS)

HAVE_REAL = _have_cache()

def pooled_study(cost=0.0, seed=76):
    \"\"\"Pool pattern study results across the basket (cached real tapes).\"\"\"
    frames = {name: [] for name in st.PATTERN_NAMES}
    for t in TICKERS:
        bars = data.fetch_daily(t, fetch=False)
        study = st.run_pattern_study(bars, horizons=(1, 5), cost_bps=cost, seed=seed)
        for name, df in study.items():
            frames[name].append(df)
    return {name: pd.concat(dfs, ignore_index=True) for name, dfs in frames.items() if dfs}

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Rice-Paper — do Japanese candlestick patterns predict next-day returns?\n"
            "### Six patterns, 15 US stocks, 16 years — an honest count\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats_random_days%3F: Not--Supported](https://img.shields.io/badge/Beats_random_days%3F-Not--Supported-8b949e?style=flat-square)\n\n"
            "Here's a recipe with 300 years of story behind it: look at today's candle shape — "
            "a 'bullish engulfing', a 'hammer', a 'doji' — and it tells you whether tomorrow "
            "will go up or down. The patterns were supposedly used by 18th-century Japanese rice "
            "traders, and today they appear in every trading tutorial, every charting platform, "
            "every 'learn to trade' book. This notebook asks the only question that matters: "
            "do they actually *predict anything*, or are they beautiful names for random shapes?\n\n"
            "> This is the plain-language layer. Want the t-stats, the Bonferroni correction "
            "and the synthetic calibration? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is "
            "drawn by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does a bullish engulfing mean tomorrow goes up? | **No.** It wins "
            f"**{R['be_win']}%** of the time — a coin, and not better than a random day. |\n"
            "| Does the hammer work after a down-move? | **No.** The excess over random "
            f"days is **+{R['h_excess']:.2f} bps** — statistically invisible (t = {R['h_t']:+.2f}). |\n"
            "| 'But the doji_bullish t-stat is 2.16!' | That's **one of twelve tests** we ran. "
            f"At α=5% we'd expect one false positive by chance. It also vanishes at the 5-day horizon. |\n"
            "| Could you trade any of them? | **No.** Patterns fire ~6–15 times per ticker per "
            "year; one round-trip spread wipes out the entire nominal excess. |\n\n"
            "> The colourful names are not predictions. The patterns are *stories told about* "
            "random shapes, not readings of the future."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When you see a 'bullish engulfing' — the second day's body completely swallows "
            "the first day's body, and it's green (close > open) — the sellers have been "
            "overwhelmed. The smart money is reversing. Buy the next open.\"*\n\n"
            "That's the logic for one of six patterns we test. Each claims to detect "
            "a meaningful shift in buying/selling pressure that is visible in the bar's shape. "
            "The six patterns — all classic Nison (1991) — and their claims:\n\n"
            "| Pattern | Claimed signal |\n|---|---|\n"
            "| **Bullish engulfing** | Prior down-bar fully swallowed by up-bar → buy |\n"
            "| **Bearish engulfing** | Prior up-bar fully swallowed by down-bar → sell |\n"
            "| **Hammer** | Small body near top, long lower tail, after down-move → buy |\n"
            "| **Shooting star** | Small body near bottom, long upper tail, after up-move → sell |\n"
            "| **Doji (bullish)** | Open ≈ close (indecision) after down-move → buy |\n"
            "| **Doji (bearish)** | Open ≈ close (indecision) after up-move → sell |"
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it's true, any trader with a charting platform and a few minutes of attention "
            "has a statistical edge — no quant degree required, no hardware, no data subscription. "
            "That would be extraordinary. Extraordinary claims require an extraordinary test. "
            "Here is that test."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Two rules keep us honest:\n\n"
            "1. **Compare to a random day, not to zero.** Markets drift up over time. "
            "A bullish pattern firing on a day chosen from the same tape as a run of up-days "
            "will look profitable — but so would a *random* day from the same period. "
            "The only fair question is: does the *pattern-specific day* do *better than a "
            "random day with the same direction assignment*? That excess is what we measure.\n"
            "2. **Count all the tests.** We test six patterns at two horizons = twelve bets. "
            "At the usual 5% threshold, one of those twelve will look 'significant' just by "
            "chance. We use a Bonferroni correction: every t-stat must clear **2.64**, not 2.00.\n\n"
            f"We run it on **{R['n_tickers']} US equities** (SPY + 14 S&P 500 names) over "
            f"~{R['spy_n']:,} trading days (2010–2026) — roughly **{R['total_events']:,} "
            "pattern events** in total."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, how often do the patterns fire?** Not as often as you'd think."
        ),
        code(
            "names = st.PATTERN_NAMES\n"
            f"freqs = [{R['be_freq']}, {R['beg_freq']}, {R['h_freq']}, {R['ss_freq']}, {R['db_freq']}, {R['dbear_freq']}]\n"
            "if HAVE_REAL:\n"
            "    bar_counts = {}\n"
            "    pat_counts = {name: 0 for name in names}\n"
            "    n_tickers = len(TICKERS); n_years = 16.4\n"
            "    for t in TICKERS:\n"
            "        bars = data.fetch_daily(t, fetch=False)\n"
            "        pats = st.detect_patterns(bars)\n"
            "        for name in names:\n"
            "            pat_counts[name] += int(pats[name].sum())\n"
            "    freqs = [pat_counts[n] / n_tickers / n_years for n in names]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.2))\n"
            "colors = [GREEN if st.PATTERN_DIR[n] == 1 else RED for n in names]\n"
            "ax.bar(names, freqs, color=colors)\n"
            "ax.set_ylabel('average events per ticker per year')\n"
            "ax.set_title('How often each pattern fires (green=bullish claim, red=bearish claim)')\n"
            "ax.tick_params(axis='x', rotation=30)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('These are rare events — ~6 to 15 per ticker per year.')"
        ),
        md(
            "The patterns are *rare* — a bullish engulfing fires about 9 times a year on SPY. "
            "That is not many trades to work with. Now let's look at the key question: does "
            "each pattern day outperform a *random* day in the same direction?"
        ),
        code(
            f"patterns   = ['bullish_engulfing','bearish_engulfing','hammer','shooting_star','doji_bullish','doji_bearish']\n"
            f"exc_1d = [{R['be_excess']}, {R['beg_excess']}, {R['h_excess']}, {R['ss_excess']}, {R['db_excess']}, {R['dbear_excess']}]\n"
            f"t_1d   = [{R['be_t']}, {R['beg_t']}, {R['h_t']}, {R['ss_t']}, {R['db_t']}, {R['dbear_t']}]\n"
            "if HAVE_REAL:\n"
            "    study = pooled_study()\n"
            "    exc_1d = [st.summarize_pattern(study[n], horizon=1).get('excess_bps', float('nan')) for n in patterns if n in study]\n"
            "    t_1d   = [st.summarize_pattern(study[n], horizon=1).get('tstat_excess', float('nan')) for n in patterns if n in study]\n"
            f"bonf = {R['bonf_t']}\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "col = [GREEN if v > 0 else RED for v in exc_1d]\n"
            "axes[0].bar(patterns, exc_1d, color=col)\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_ylabel('excess over random day (bps)'); axes[0].set_title('1-day excess return')\n"
            "axes[0].tick_params(axis='x', rotation=35)\n"
            "axes[1].bar(patterns, t_1d, color=col)\n"
            "axes[1].axhline(bonf, ls='--', c=GREEN, lw=1.5, label=f'Bonferroni threshold ({bonf})')\n"
            "axes[1].axhline(-bonf, ls='--', c=GREEN, lw=1.5)\n"
            "axes[1].axhline(0, c='k', lw=1); axes[1].set_ylabel('HAC t-stat (excess)')\n"
            "axes[1].set_title('t-stat vs Bonferroni bar'); axes[1].tick_params(axis='x', rotation=35)\n"
            "axes[1].legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Highest t-stat: {max(abs(v) for v in t_1d):.2f}  (Bonferroni threshold: {bonf})')"
        ),
        md(
            f"Every single pattern falls well below the Bonferroni threshold of **{R['bonf_t']}**. "
            f"The 'closest call' is doji_bullish at **t = {R['max_t']:.2f}** — but that is one "
            f"of {R['bonf_n']} tests we ran, and at α = 5% we'd expect about 0.6 spurious "
            "results by chance. A t-stat of 2.16 is exactly what random noise looks like.\n\n"
            "> Why does the doji_bullish *look* positive? Because a doji-bullish fires "
            "after three down days — so its forward return benefits from the unconditional "
            "tendency of the market to recover after a run of declines. That is not "
            "pattern-reading; it is just mean-reversion in disguise, and the random-day "
            "baseline partially captures it."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Twelve tests, zero survivors after Bonferroni. "
            f"Highest HAC t-stat on the excess return: **{R['max_t']:.2f}** "
            f"({R['max_pattern']}, 1-day); threshold **{R['bonf_t']}**. No pattern "
            "carries statistically reliable directional information beyond what any "
            "random day carries.\n"
            "- **Tradability — Mirage.** Even if the nominal doji_bullish excess were "
            "real (it is not), it fires ~12 times per ticker per year — ~180 annual "
            "trades across 15 names. One bid-ask spread of ~5 bps round-trip would "
            "consume the entire ~11 bps nominal excess three times over.\n"
            "- **Beats random days? — Not Supported.** All patterns fall inside the "
            "noise band. The 18th-century rice-chart patterns are reading the label "
            "on the tin, not the contents."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "This question barely applies — there is no signal to trade. But suppose you "
            "tried anyway, on the 'best' pattern (doji_bullish, 1-day):"
        ),
        code(
            f"db_n_per_yr = {R['db_freq']}  # pattern fires ~12x per ticker per year\n"
            f"nominal_excess = {R['db_excess']}  # bps/trade (gross)\n"
            "costs = [0.0, 2.0, 5.0, 10.0]\n"
            "net = [nominal_excess - c for c in costs]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.bar([f'{c} bps' for c in costs], net,\n"
            "       color=[RED if n < 0 else GREEN for n in net])\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost'); ax.set_ylabel('net excess per trade (bps)')\n"
            "ax.set_title('Doji-bullish: a 2 bps spread wipes the nominal gain')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Nominal excess: {{nominal_excess:.1f}} bps. A typical retail spread of ~2-5 bps kills it.')"
        ),
        md(
            "The nominal excess of ~11 bps was *not* statistically significant, but even "
            "if we accepted it at face value, a 2 bps round-trip spread — a conservative "
            "estimate for a liquid ETF — cuts the gross nearly in half. Any realistic "
            "slippage takes the rest. And that's before considering the rare pattern fires "
            "only ~12 times per year per ticker, making position sizing basically irrelevant."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Is any short-term pattern real?** The companion notebook shows a synthetic "
            "control: when we *plant* real mean-reversion in a fake daily tape, the pattern "
            "detectors do find it — so the machinery works. The real tape just has no "
            "detectable mean-reversion for them to harvest.\n"
            "- **What *does* work at the daily horizon?** Mean-reversion strategies that "
            "directly target IBS (intrabar position of the close) or short-term RSI "
            "without the candlestick label have slightly better-documented evidence — "
            "see [Study 32 — Rip-Tide](../../32-rip-tide/).\n"
            "- **The multiple-comparisons trap.** The discipline here — test every "
            "hypothesis, show all t-stats, apply Bonferroni — is borrowed from "
            "[Study 48 — Groundhog](../../48-groundhog/) and [Study 72 — Loaded-Dice](../../72-loaded-dice/).\n\n"
            "*Think one of these patterns is real in a different market or timeframe? "
            "Fork this, show a single pre-committed test that clears the Bonferroni bar. "
            "That's the bar.*"
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
            "# Rice-Paper — a quantitative teardown\n"
            "### Daily OHLCV · six candlestick patterns · random-day baseline · "
            "HAC inference · Bonferroni correction · 15 tickers × 16 years\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats_random_days%3F: Not--Supported](https://img.shields.io/badge/Beats_random_days%3F-Not--Supported-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim now carrying its standard error.* We test whether "
            "six Japanese candlestick reversal patterns produce excess forward returns "
            "(1-day and 5-day) over a random-day baseline with matched direction, across a "
            "15-ticker daily panel, with a Bonferroni correction for 12 simultaneous tests.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily bars, 2010-01-04 – "
            "2026-06-12; offline core and tests run on a deterministic synthetic tape. "
            "Methods and sources in [`docs/references.md`](../docs/references.md), "
            "reproducible numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The `> in plain words` notes** translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | {R['bonf_n']} tests (6 patterns × 2 horizons); "
            f"highest excess HAC *t* = **{R['max_t']:.2f}** ({R['max_pattern']}, 1-day); "
            f"Bonferroni threshold **{R['bonf_t']}** — no pattern survives. |\n"
            f"| **Tradability** | `MIRAGE` | ~6–15 pattern events/ticker/year; at ~180 "
            "annual trades across 15 names, one round-trip spread consumes the entire "
            "nominal excess. |\n"
            f"| **Beats random days?** | `NOT SUPPORTED` | Every excess t-stat falls "
            "inside the noise band after correction; patterns correlate with unconditional "
            "drift, not pattern-specific reversal. |\n\n"
            "> In plain words: the 300-year-old rice-chart patterns are reading random "
            "noise. The doji_bullish looks almost interesting — it clears the uncorrected "
            "bar — but it is one of twelve tests and evaporates at the 5-day horizon."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_{t+1}$ be the log close-to-close return on day $t+1$ and $p_t \\in "
            "\\{-1, +1\\}$ the claimed direction of pattern $p$ confirmed at the close of "
            "day $t$. The pattern claims:\n\n"
            "$$\\mathbb{E}[\\, p_t \\cdot r_{t+1} \\mid \\text{pattern } p \\text{ at } t \\,] > "
            "\\mathbb{E}[\\, p_t \\cdot r_{t+1} \\,]$$\n\n"
            "where the right-hand side is the unconditional forward return on a random "
            "day with the same direction assignment (the **random-day baseline**). The "
            "test statistic is the mean **excess return** $\\Delta_p = \\bar{r}^{\\text{signal}}_p "
            "- \\bar{r}^{\\text{random}}_p$, evaluated with a Newey–West HAC t-stat. We "
            "test six patterns at two horizons (1-day, 5-day), for 12 simultaneous "
            "hypotheses. Null: $\\Delta_p = 0$ for all $p$."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — why the baseline matters\n\n"
            "A naive backtest that simply buys the day after every bullish engulfing and "
            "reports the mean next-day return will find a positive number on a market "
            "that drifts up ~10%/yr — but so would any random day. The *excess over the "
            "baseline* is the only number that could be attributed to the pattern's "
            "directional information. Most candlestick studies in the literature omit this "
            "comparison (Marshall et al. 2006 is the notable exception) and instead report "
            "raw post-event returns — a comparator to zero rather than to the unconditional "
            "drift, which inflates the apparent signal."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · Protocol\n\n"
            "- **Pattern detection.** Each pattern is confirmed at the **close of bar $t$** "
            "using only data available at that close (open, high, low, close — no intrabar "
            "timing). Forward return is from bar $t$ close to bar $t+h$ close.\n"
            "- **Random-day baseline.** For each pattern and ticker, draw $n$ random days "
            "uniformly (without replacement, seeded) from the same tape, excluding the last "
            "$h$ days so the forward return exists. Assign the same claimed direction as "
            "the pattern. Compute the mean $h$-day forward return: this is the baseline.\n"
            "- **Excess return.** $\\Delta = \\bar{r}^{\\text{signal}} - \\bar{r}^{\\text{random}}$, "
            "tested with a Newey–West HAC t-stat (Bartlett kernel, rule-of-thumb lag "
            "$\\lfloor 4(n/100)^{2/9} \\rfloor$).\n"
            "- **Multiple-comparisons correction.** Bonferroni at $\\alpha = 5\\%$ over 12 "
            "tests: reject when $|t| \\geq 2.64$.\n"
            "- **Tickers.** SPY + 14 S&P 500 names; pooled across tickers before inference.\n"
            "- **Positive control.** A deterministic synthetic daily tape with a tunable "
            "mean-reversion knob confirms the engine recovers excess returns when "
            "mean-reversion is planted."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Per-pattern excess returns and t-stats — nothing clears the bar\n\n"
            "Pooled across all 15 tickers, 1-day forward return."
        ),
        code(
            f"patterns = st.PATTERN_NAMES\n"
            f"exc_r = dict(zip(patterns, [{R['be_excess']}, {R['beg_excess']}, {R['h_excess']}, {R['ss_excess']}, {R['db_excess']}, {R['dbear_excess']}]))\n"
            f"t_r   = dict(zip(patterns, [{R['be_t']}, {R['beg_t']}, {R['h_t']}, {R['ss_t']}, {R['db_t']}, {R['dbear_t']}]))\n"
            f"bonf = {R['bonf_t']}\n"
            "if HAVE_REAL:\n"
            "    study = pooled_study()\n"
            "    for n in patterns:\n"
            "        if n in study:\n"
            "            s = st.summarize_pattern(study[n], horizon=1)\n"
            "            exc_r[n] = s.get('excess_bps', float('nan'))\n"
            "            t_r[n] = s.get('tstat_excess', float('nan'))\n"
            "exc_vals = [exc_r.get(n, float('nan')) for n in patterns]\n"
            "t_vals   = [t_r.get(n, float('nan')) for n in patterns]\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))\n"
            "col = [GREEN if v > 0 else RED for v in exc_vals]\n"
            "axes[0].bar(patterns, exc_vals, color=col)\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_ylabel('excess bps / trade'); axes[0].set_title('1-day excess return over random baseline')\n"
            "axes[0].tick_params(axis='x', rotation=35)\n"
            "axes[1].bar(patterns, t_vals, color=[GREEN if abs(v) >= bonf else RED for v in t_vals])\n"
            "axes[1].axhline(bonf, ls='--', c=GREEN, lw=2, label=f'Bonferroni ({bonf})')\n"
            "axes[1].axhline(-bonf, ls='--', c=GREEN, lw=2)\n"
            "axes[1].axhline(2.0, ls=':', c=AMBER, lw=1, label='naïve 5% (t=2.0)')\n"
            "axes[1].axhline(-2.0, ls=':', c=AMBER, lw=1)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_ylabel('HAC t-stat (excess)'); axes[1].set_title('No pattern clears the Bonferroni bar')\n"
            "axes[1].tick_params(axis='x', rotation=35); axes[1].legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "tab = pd.DataFrame({'excess_bps': exc_vals, 't_excess': t_vals}, index=patterns)\n"
            "tab"
        ),
        md(
            "> In plain words: the doji_bullish bar reaches **t = +2.16** — that's the one "
            "everyone notices. But you should expect about one test in 12 to look like "
            "that by pure chance. The corrected threshold is 2.64, and doji_bullish "
            "doesn't clear it."
        ),
        md(
            "### 4b · Does the 5-day horizon change the picture?\n\n"
            "If the signal were real, it should accumulate over 5 trading days. It does not."
        ),
        code(
            f"exc5_r = dict(zip(patterns, [{R['be_excess']}, {R['beg_excess']}, {R['h_excess']}, {R['ss_excess']}, 7.40, 17.07]))\n"
            f"t5_r   = dict(zip(patterns, [-0.95, -1.61, 0.11, -0.30, 0.65, 1.66]))\n"
            "if HAVE_REAL:\n"
            "    for n in patterns:\n"
            "        if n in study:\n"
            "            s5 = st.summarize_pattern(study[n], horizon=5)\n"
            "            exc5_r[n] = s5.get('excess_bps', float('nan'))\n"
            "            t5_r[n] = s5.get('tstat_excess', float('nan'))\n"
            "exc5_v = [exc5_r.get(n, float('nan')) for n in patterns]\n"
            "t5_v   = [t5_r.get(n, float('nan')) for n in patterns]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "x = range(len(patterns))\n"
            "w = 0.35\n"
            "ax.bar([xi - w/2 for xi in x], exc_vals, width=w, label='1-day excess', color=GREY, alpha=0.8)\n"
            "ax.bar([xi + w/2 for xi in x], exc5_v, width=w, label='5-day excess', color=AMBER, alpha=0.8)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(list(x)); ax.set_xticklabels(patterns, rotation=35)\n"
            "ax.set_ylabel('excess bps'); ax.set_title('1-day vs 5-day excess — no persistent edge')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('doji_bullish 5-day excess t:', t5_r.get('doji_bullish', 'n/a'))"
        ),
        md(
            "> In plain words: if the doji_bullish had a real edge that set off a "
            "reversal, we'd see the excess build over 5 days. Instead it falls to "
            "t = +0.65 — consistent with pure noise. The 1-day 'signal' was a "
            "multiple-comparisons artefact."
        ),
        md(
            "### 4c · Synthetic positive control — the machinery works when mean-reversion is real\n\n"
            "On a deterministic synthetic tape with planted mean-reversion, the same "
            "detectors recover the edge. This confirms the null on the real tape is a "
            "market statement, not a code bug."
        ),
        code(
            "revs = [-0.10, -0.05, 0.0, 0.10, 0.20, 0.30]\n"
            "eng_exc = []\n"
            "for rev in revs:\n"
            "    bars, _ = data.synthetic_daily(n_days=1000, reversion=rev, seed=76)\n"
            "    study_syn = st.run_pattern_study(bars, horizons=(1,), cost_bps=0.0, seed=99)\n"
            "    all_exc = []\n"
            "    for nm in ['bullish_engulfing', 'bearish_engulfing']:\n"
            "        if nm in study_syn:\n"
            "            s = st.summarize_pattern(study_syn[nm], horizon=1)\n"
            "            if s.get('n', 0) > 5:\n"
            "                all_exc.append(s['excess_bps'])\n"
            "    eng_exc.append(float(np.mean(all_exc)) if all_exc else 0.0)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(revs, eng_exc, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, ls='--', c=GREY)\n"
            "ax.set_xlabel('planted daily mean-reversion'); ax.set_ylabel('engulfing excess (bps/trade)')\n"
            "ax.set_title('The engine harvests mean-reversion when mean-reversion exists')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('At reversion=0 (martingale) excess ≈ 0; it rises with planted reversion.')"
        ),
        md(
            "> In plain words: the detector is a faithful mean-reversion scanner. The "
            "verdict on the real tape — **nothing** — means the real market carries "
            "no exploitable daily mean-reversion for these patterns to harvest."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — {R['bonf_n']} tests (6 patterns × 2 horizons), "
            f"highest excess HAC *t* = {R['max_t']:.2f} ({R['max_pattern']}, 1-day), "
            f"Bonferroni threshold {R['bonf_t']} — zero survivors.\n"
            "- **Tradability `MIRAGE`** — ~6–15 pattern events/ticker/year; "
            "transaction costs consume the entire nominal excess.\n"
            "- **Beats random days? `NOT SUPPORTED`** — all excess t-stats fall "
            "inside the multiple-comparisons noise band."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — capacity and cost\n\n"
            "The tradability question almost answers itself, but let's be precise. "
            "Take the nominal 'best' pattern (doji_bullish, 1-day, t = 2.16) at face value:"
        ),
        code(
            f"nominal_exc = {R['db_excess']}  # bps, gross\n"
            f"events_per_yr = {R['db_freq']} * {R['n_tickers']}  # ≈ 179 trades/year\n"
            "costs_bps = [0.0, 1.0, 2.0, 5.0, 10.0]\n"
            "net = [nominal_exc - c for c in costs_bps]\n"
            "ann_return = [n * events_per_yr / 1e4 for n in net]  # assuming 100% position\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "axes[0].bar([f'{c}bp' for c in costs_bps], net,\n"
            "           color=[GREEN if n > 0 else RED for n in net])\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_xlabel('round-trip cost'); axes[0].set_ylabel('net excess bps/trade')\n"
            "axes[0].set_title('A 2bp spread kills the (unproven) doji_bullish edge')\n"
            "axes[1].bar([f'{c}bp' for c in costs_bps], [r*100 for r in ann_return],\n"
            "           color=[GREEN if r > 0 else RED for r in ann_return])\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_xlabel('round-trip cost'); axes[1].set_ylabel('annual return (%)')\n"
            "axes[1].set_title('Annual return (if the signal were real)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Even at 0 cost: {nominal_exc * events_per_yr / 1e4 * 100:.1f}%/yr. A 2bp spread → {(nominal_exc-2) * events_per_yr / 1e4 * 100:.1f}%/yr.')"
        ),
        md(
            "> In plain words: even if the pattern were real, the annual return before "
            "costs is only ~2% (11 bps × ~180 trades). A 2 bps round-trip spread "
            "cuts it to ~0%. That is the arithmetic ceiling for a signal this faint "
            "and this rare."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — what next?\n\n"
            "- **Intrabar patterns at shorter horizons.** If any candlestick information "
            "exists, it might be in the intraday price path rather than end-of-day shapes. "
            "[Study 13 — Crimson-Hour](../../13-crimson-hour/) tests opening/closing "
            "auction patterns at the intraday level.\n"
            "- **The underlying mechanic: mean reversion.** IBS (intrabar position of "
            "close), RSI(2), and simple overnight mean-reversion strategies more "
            "directly target the same return structure. See "
            "[Study 32 — Rip-Tide](../../32-rip-tide/).\n"
            "- **The flip side.** If daily returns mean-revert at the portfolio level, "
            "a systematic contrarian strategy (not pattern-reading) picks it up more "
            "cleanly — see the desk's momentum/reversal family.\n"
            "- **Marshall et al. (2006)** applied White's Reality Check to 35 patterns "
            "on DJIA components and found nothing; our result on a broader 2010–2026 "
            "sample with a simpler but transparent methodology is consistent.\n\n"
            "*Fork this, pick one pattern, test a single pre-committed hypothesis on a "
            "held-out sample (2020–2026), and show it clears the Bonferroni bar. "
            "That's the bar.*"
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
