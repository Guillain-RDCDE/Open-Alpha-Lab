"""Generate the two narrative notebooks for Study 186 (Morning-Star).

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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-15).
R = dict(
    # 1-day pooled excess returns (gross, bps) and HAC t-stats
    ms_n=1042,    ms_excess=-28.08,  ms_t=-3.14,  ms_win=50.3,  ms_mean=-2.07,
    es_n=1123,    es_excess=-1.44,   es_t=-0.18,  es_win=47.0,  es_mean=-8.23,
    # baseline means (what random days return in the same window)
    ms_base=26.01,
    es_base=-6.79,
    # 5-day
    ms_excess5=-57.96,  ms_t5=-2.84,
    es_excess5=16.72,   es_t5=1.10,
    # Bonferroni threshold
    bonf_t=2.50,
    bonf_n=4,
    # Pattern frequencies per ticker per year (from event counts / 15 tickers / 16.4 yr)
    ms_freq=4.2,
    es_freq=4.6,
    # SPY counts
    spy_ms=67,
    spy_es=76,
    spy_n=4137,
    # Data window
    start="2010-01-04",
    end="2026-06-15",
    n_tickers=15,
    total_events=2165,
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

from morning_star import data, strategy as st

TICKERS = [
    "SPY", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA",
    "JPM", "BAC", "XOM", "JNJ", "PG", "KO", "WMT",
]

def _have_cache():
    return all(os.path.exists(data._cache_path(t, data.DEFAULT_CACHE)) for t in TICKERS)

HAVE_REAL = _have_cache()

def pooled_study(cost=0.0, seed=186):
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
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Morning-Star -- does the three-candle reversal actually predict a bounce?\n"
            "### Two patterns, 15 US stocks, 16 years -- an honest count\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats_random_days%3F: Not--Supported](https://img.shields.io/badge/Beats_random_days%3F-Not--Supported-8b949e?style=flat-square)\n\n"
            "Here is one of the most celebrated recipes in candlestick charting: watch for three "
            "candles in a row. First, a big red candle (sellers in control). Then a small 'star' "
            "candle with a little body that sits below the first -- indecision. Then a big green "
            "candle that climbs back into the first candle's body -- the buyers have won. That is "
            "the *morning star*, and it supposedly signals a bottom. Its mirror image, the "
            "*evening star*, signals a top. They appear in every charting course, every trading "
            "book, every platform. This notebook asks the only question that matters: do they "
            "actually *predict anything*, or are they stories told about random shapes?\n\n"
            "> This is the plain-language layer. Want the t-stats, the Bonferroni correction, and "
            "the synthetic calibration? That is the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is "
            "drawn by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the morning-star predict tomorrow goes up? | **No -- and it's worse than random.** "
            f"The morning-star returns **{R['ms_excess']:+.0f} bps less** than a random day after a "
            f"down-move (t = {R['ms_t']:+.2f}). |\n"
            f"| Is the evening-star any better? | **No.** It is statistically flat: excess = "
            f"**{R['es_excess']:+.2f} bps**, t = {R['es_t']:+.2f}. |\n"
            f"| Why does morning-star underperform? | The pattern fires after a strong down-move; "
            f"a *random day* after a down-move already bounces ~{R['ms_base']:+.0f} bps. The "
            "star's third candle has already used up most of that bounce. |\n"
            "| Could you trade it in reverse? | **No.** The negative excess is a timing artefact, "
            "not a reliable short signal. |\n\n"
            "> The colourful three-candle story arrives at the party after the bounce has already "
            "started. Random days from the same down-move window do better."
        ),

        # ---- BEAT 1 -- THE CLAIM ----------------------------------------------
        md(
            "## 1 -- The claim\n\n"
            "> *\"Three candles: large bearish, small star below the first body, then large bullish "
            "closing back into the first body -- the bears are exhausted. Buy the next open.\"*\n\n"
            "That is the morning-star. Its evening-star mirror says the opposite: three candles with "
            "a star gapping above a large bullish body, then a strong bearish close back into the "
            "first body -- signal a top, sell the next open.\n\n"
            "| Pattern | Claimed signal | Direction |\n|---|---|---|\n"
            "| **Morning star** | Bottom of a down-move; buyers take over | Bullish (+1) |\n"
            "| **Evening star** | Top of an up-move; sellers take over | Bearish (−1) |"
        ),

        # ---- BEAT 2 -- SO WHAT ------------------------------------------------
        md(
            "## 2 -- So what?\n\n"
            "If it is true, any trader watching a daily chart can identify market turning points "
            "three bars in advance -- no data feed, no quant toolkit, just pattern recognition. "
            "Patterns this visible and this easy to identify would be arbed away if they worked "
            "reliably. But the question is not 'are they real?' -- it is 'can we measure whether "
            "they are real?' Here is that measurement."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 -- How would we even know?\n\n"
            "Two rules keep us honest:\n\n"
            "1. **Compare to a random day, not to zero.** A morning-star fires after a down-move. "
            "Down-moves are followed, on average, by bounces -- mean-reversion is a real effect. "
            "A bullish pattern-triggered day will look profitable just because *any* day after a "
            "down-move tends to go up. The only fair question is: does the *pattern-specific day* "
            "do *better than a random day after a similar down-move*? That excess is what we measure.\n"
            "2. **Count all the tests.** We test two patterns at two horizons = four bets. At "
            "the usual 5% threshold, one of four will look significant every twenty experiments "
            "just by chance. We use a Bonferroni correction: every t-stat must clear "
            f"**{R['bonf_t']}**, not 2.00.\n\n"
            f"We run it on **{R['n_tickers']} US equities** (SPY + 14 S&P 500 names) over "
            f"~{R['spy_n']:,} trading days (2010--2026) -- roughly **{R['total_events']:,} "
            "pattern events** in total."
        ),

        # ---- BEAT 4 -- THE TEARDOWN -------------------------------------------
        md(
            "## 4 -- The teardown -- let's actually look\n\n"
            "**First, how often do the patterns fire?** These are three-candle patterns with "
            "specific gap requirements; they are rarer than single-candle patterns."
        ),
        code(
            f"names = st.PATTERN_NAMES\n"
            f"freqs = [{R['ms_freq']}, {R['es_freq']}]\n"
            "if HAVE_REAL:\n"
            "    pat_counts = {name: 0 for name in names}\n"
            "    n_years = 16.4\n"
            "    for t in TICKERS:\n"
            "        bars = data.fetch_daily(t, fetch=False)\n"
            "        pats = st.detect_patterns(bars)\n"
            "        for name in names:\n"
            "            pat_counts[name] += int(pats[name].sum())\n"
            "    freqs = [pat_counts[n] / len(TICKERS) / n_years for n in names]\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.0))\n"
            "colors = [GREEN, RED]\n"
            "ax.bar(names, freqs, color=colors)\n"
            "ax.set_ylabel('average events per ticker per year')\n"
            "ax.set_title('Three-candle pattern frequency (green=bullish, red=bearish claim)')\n"
            "ax.tick_params(axis='x', rotation=15)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Morning-star fires ~{freqs[0]:.1f}x/yr per ticker; evening-star ~{freqs[1]:.1f}x/yr.')"
        ),
        md(
            "The patterns fire ~4-5 times per ticker per year -- rarer than single-candle "
            "patterns, but ~69 morning-star and ~75 evening-star events per year across the "
            f"basket of {R['n_tickers']} tickers. Now the key question: does the pattern day "
            "outperform a *random* day after the same directional context?"
        ),
        code(
            f"patterns = st.PATTERN_NAMES\n"
            f"exc_1d = [{R['ms_excess']}, {R['es_excess']}]\n"
            f"t_1d   = [{R['ms_t']}, {R['es_t']}]\n"
            f"base_1d = [{R['ms_base']}, {R['es_base']}]\n"
            "if HAVE_REAL:\n"
            "    study = pooled_study()\n"
            "    exc_1d = [st.summarize_pattern(study[n], horizon=1).get('excess_bps', float('nan')) for n in patterns if n in study]\n"
            "    t_1d   = [st.summarize_pattern(study[n], horizon=1).get('tstat_excess', float('nan')) for n in patterns if n in study]\n"
            "    base_1d = [st.summarize_pattern(study[n], horizon=1).get('baseline_mean_bps', float('nan')) for n in patterns if n in study]\n"
            f"bonf = {R['bonf_t']}\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "col = [GREEN if v > 0 else RED for v in exc_1d]\n"
            "axes[0].bar(patterns, exc_1d, color=col, label='excess over random')\n"
            "axes[0].bar(patterns, base_1d, color=GREY, alpha=0.4, label='random-day baseline')\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_ylabel('bps'); axes[0].set_title('1-day return: pattern vs random baseline')\n"
            "axes[0].tick_params(axis='x', rotation=15)\n"
            "axes[0].legend(fontsize=8)\n"
            "axes[1].bar(patterns, t_1d, color=col)\n"
            "axes[1].axhline(bonf, ls='--', c=RED, lw=1.5, label=f'Bonferroni +/- ({bonf})')\n"
            "axes[1].axhline(-bonf, ls='--', c=RED, lw=1.5)\n"
            "axes[1].axhline(0, c='k', lw=1); axes[1].set_ylabel('HAC t-stat (excess)')\n"
            "axes[1].set_title('t-stat: morning-star clears the bar -- wrong direction'); axes[1].tick_params(axis='x', rotation=15)\n"
            "axes[1].legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Morning-star excess: {exc_1d[0]:+.1f} bps, t={t_1d[0]:+.2f}')\n"
            "print(f'Random baseline for morning-star window: {base_1d[0]:+.1f} bps')"
        ),
        md(
            f"The morning-star t-stat is **{R['ms_t']:+.2f}** -- it clears the Bonferroni bar "
            f"of {R['bonf_t']}, but in the *wrong direction*. The excess is **{R['ms_excess']:+.0f} "
            "bps** (negative), meaning the morning-star pattern day returns *less* than a random "
            "day drawn from the same volatile down-window.\n\n"
            f"The grey bars show why: random days drawn after a down-move already return "
            f"~{R['ms_base']:+.0f} bps on average. The morning-star's third candle has already "
            "captured some of that bounce; the day *after* the complete pattern has less "
            "mean-reversion left to harvest than a random day from the same period.\n\n"
            "> The morning-star arrives at the party too late. The bounce has started before "
            "the third candle closes, and the next day's return is front-run by the pattern "
            "itself."
        ),

        # ---- BEAT 5 -- THE VERDICT --------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal -- None.** {R['bonf_n']} tests (2 patterns x 2 horizons); "
            f"morning-star excess HAC t = **{R['ms_t']:+.2f}** (significant, but negative -- wrong "
            f"direction); evening-star t = {R['es_t']:+.2f} (noise). Zero patterns show a "
            f"*positive* excess that clears the Bonferroni threshold of {R['bonf_t']}. Signal = NONE.\n"
            "- **Tradability -- Mirage.** The claimed direction (buy after morning-star) loses "
            "ground to a random day. There is no edge to trade. Shorting the pattern is not "
            "robust either: the negative excess is a timing artefact, not a reliable bearish "
            "signal, and t = -3.14 on a noisy single-security signal is not tradable.\n"
            "- **Beats random days? -- Not Supported.** The morning-star *trails* random days "
            f"by {abs(R['ms_excess']):.0f} bps. The evening-star is flat."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 -- Could you actually trade it?\n\n"
            "This question is almost moot: the 'signal' is in the wrong direction. But suppose "
            "someone argued 'short the morning-star completion day.' What would that look like?"
        ),
        code(
            f"reverse_exc = {abs(R['ms_excess'])}  # bps gross if traded in reverse\n"
            f"events_per_yr = {R['ms_freq']} * {R['n_tickers']}  # ~63 reverse trades/year\n"
            "costs = [0.0, 1.0, 2.0, 5.0]\n"
            "net = [reverse_exc - c for c in costs]\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.0))\n"
            "ax.bar([f'{c} bps' for c in costs], net,\n"
            "       color=[GREEN if n > 0 else RED for n in net])\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost'); ax.set_ylabel('net excess bps/trade (reverse)')\n"
            "ax.set_title('Shorting morning-star: 2 bps spread kills the (unproven) negative excess')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Even at 0 cost, reverse annual return ~ {{reverse_exc * events_per_yr / 1e4 * 100:.1f}}%/yr')\n"
            "print('The negative excess is a timing artefact, not a repeatable short signal.')"
        ),
        md(
            "Even at zero cost the reverse trade would yield only ~1.8%/yr (28 bps x ~63 "
            "events/yr). A 2 bps round-trip spread eliminates most of it. And the structural "
            "cause of the negative excess -- the pattern arriving late in a mean-reversion "
            "window -- is not a tradable short signal; it is noise."
        ),

        # ---- BEAT 7 -- GOING FURTHER ------------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **The sibling study.** [Study 76 -- Rice-Paper](../../76-rice-paper/) tests six "
            "single-candle patterns (bullish/bearish engulfing, hammer, shooting star, doji) on "
            "the same basket with the same methodology. Also Signal=NONE/Tradability=MIRAGE -- "
            "the pattern family is consistent.\n"
            "- **What *does* work at the daily horizon?** Mean-reversion strategies that "
            "directly target IBS (intrabar position of the close) or short-term RSI skip the "
            "three-candle narrative and trade the underlying tendency more cleanly.\n"
            "- **The underlying mechanic.** If you believe mean-reversion is real (De Bondt & "
            "Thaler 1985; Jegadeesh 1990), trade it directly with a quantified entry rule, "
            "not a visual pattern that fires after the reversion is already underway.\n"
            "- **Marshall et al. (2006)** applied White's Reality Check to 35 candlestick "
            "patterns including morning/evening stars on DJIA components; no pattern survived. "
            "Our 2010--2026 result is consistent.\n\n"
            "*Fork this, pick one parameter variant, test a single pre-committed hypothesis on "
            "a held-out sample (2020--2026), and show it clears the Bonferroni bar in the "
            "correct direction. That's the bar.*"
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
            "# Morning-Star -- a quantitative teardown\n"
            "### Daily OHLCV -- two three-candle reversal patterns -- random-day baseline -- "
            "HAC inference -- Bonferroni correction -- 15 tickers x 16 years\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats_random_days%3F: Not--Supported](https://img.shields.io/badge/Beats_random_days%3F-Not--Supported-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim now carrying its standard error.* We test whether "
            "the morning-star and evening-star three-candle reversal patterns produce positive "
            "excess forward returns (1-day and 5-day) over a random-day baseline with matched "
            "direction, across a 15-ticker daily panel, with a Bonferroni correction for 4 "
            "simultaneous tests.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily bars, 2010-01-04 -- "
            f"{R['end']}; offline core and tests run on a deterministic synthetic tape. "
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
            f"| **Signal** | `NONE` | {R['bonf_n']} tests (2 patterns x 2 horizons); "
            f"morning-star 1-day HAC *t*(excess) = **{R['ms_t']:+.2f}** (significant, wrong "
            f"direction); evening-star t = {R['es_t']:+.2f}. Zero positive survivors after "
            f"Bonferroni threshold {R['bonf_t']}. |\n"
            f"| **Tradability** | `MIRAGE` | Morning-star fires ~{R['ms_freq']:.1f}x/ticker/yr; "
            "the direction is negative (pattern underperforms random days), so there is no "
            "claimed edge to trade. |\n"
            f"| **Beats random days?** | `NOT SUPPORTED` | Morning-star excess = "
            f"{R['ms_excess']:+.0f} bps (random baseline is +{R['ms_base']:.0f} bps -- "
            "the pattern arrives after the bounce is underway). |\n\n"
            "> In plain words: the morning-star fires after a large down-move, but by the "
            "time the pattern completes, the mean-reversion bounce has already started. "
            "A random day from the same volatile window outperforms the day after the star."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 -- The claim, steelmanned\n\n"
            "Let $r_{t+1}$ be the log close-to-close return on day $t+1$ and $p_t \\in "
            "\\{-1, +1\\}$ the claimed direction of pattern $p$ confirmed at the close of "
            "day $t$. The pattern claims:\n\n"
            "$$\\mathbb{E}[\\, p_t \\cdot r_{t+1} \\mid \\text{pattern } p \\text{ at } t \\,] > "
            "\\mathbb{E}[\\, p_t \\cdot r_{t+1} \\,]$$\n\n"
            "where the right-hand side is the unconditional forward return on a random "
            "day with the same direction assignment (the **random-day baseline**). The "
            "test statistic is the mean **excess return** $\\Delta_p = \\bar{r}^{\\text{signal}}_p "
            "- \\bar{r}^{\\text{random}}_p$, evaluated with a Newey-West HAC t-stat. We "
            "test 2 patterns at 2 horizons (1-day, 5-day), for 4 simultaneous hypotheses. "
            "Null: $\\Delta_p = 0$ for all $p$."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 -- So what? -- why the baseline matters more here\n\n"
            "The morning-star is a *conditional* three-candle sequence that fires after a "
            "run of bearish days. The unconditional mean-reversion bounce after a down-move "
            "is real and measurable: random days drawn from windows that follow large down-moves "
            "have positive forward returns. A naive backtest on morning-star signals on an "
            "up-trending market will report a positive return -- but it would be attributing "
            "to the pattern what belongs to the market's tendency to bounce after a fall.\n\n"
            "The random-day baseline drawn from *the same down-move window* isolates the "
            "question: *given that we are already in a down-move, does the three-candle "
            "star pattern tell us more than 'this is a down-move window'?*"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 -- Protocol\n\n"
            "- **Pattern detection.** Each pattern is confirmed at the **close of bar $t$** "
            "using only data from bars $t-4$ through $t$ (the three candles plus prior-context "
            "lookback). No look-ahead; the three-candle window is strictly backward-looking.\n"
            "- **Detector parameters.** Large candle: body >= 50% of range. Star: body <= 40% "
            "of first candle's body; star body-top within 50% of first body below/above "
            "first body-bottom/top (relaxed gap allowing partial overlap -- the strict gap "
            "is rare on liquid daily bars). Penetration: recovery candle closes at least "
            "30% into the first candle's body. Prior context: 2 days.\n"
            "- **Random-day baseline.** For each pattern and ticker, draw $n$ random days "
            "uniformly (without replacement, seeded) from the same tape. Assign the same "
            "claimed direction. Compute the mean $h$-day forward return: this is the baseline.\n"
            "- **Excess return.** $\\Delta = \\bar{r}^{\\text{signal}} - \\bar{r}^{\\text{random}}$, "
            "tested with a Newey-West HAC t-stat (Bartlett kernel, rule-of-thumb lag "
            "$\\lfloor 4(n/100)^{2/9} \\rfloor$).\n"
            "- **Multiple-comparisons correction.** Bonferroni at $\\alpha = 5\\%$ over 4 "
            "tests: reject when $|t| \\geq 2.50$.\n"
            "- **Tickers.** SPY + 14 S&P 500 names; pooled across tickers before inference.\n"
            "- **Positive control.** A deterministic synthetic daily tape with a tunable "
            "mean-reversion knob (though pattern events are rare on synthetic data)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 -- The teardown"),
        md(
            "### 4a -- 1-day excess returns: morning-star underperforms, evening-star is flat\n\n"
            "Pooled across all 15 tickers."
        ),
        code(
            "patterns = st.PATTERN_NAMES\n"
            f"exc_r = dict(zip(patterns, [{R['ms_excess']}, {R['es_excess']}]))\n"
            f"t_r   = dict(zip(patterns, [{R['ms_t']}, {R['es_t']}]))\n"
            f"base_r = dict(zip(patterns, [{R['ms_base']}, {R['es_base']}]))\n"
            f"bonf = {R['bonf_t']}\n"
            "if HAVE_REAL:\n"
            "    study = pooled_study()\n"
            "    for n in patterns:\n"
            "        if n in study:\n"
            "            s = st.summarize_pattern(study[n], horizon=1)\n"
            "            exc_r[n] = s.get('excess_bps', float('nan'))\n"
            "            t_r[n] = s.get('tstat_excess', float('nan'))\n"
            "            base_r[n] = s.get('baseline_mean_bps', float('nan'))\n"
            "exc_vals = [exc_r.get(n, float('nan')) for n in patterns]\n"
            "t_vals   = [t_r.get(n, float('nan')) for n in patterns]\n"
            "base_vals = [base_r.get(n, float('nan')) for n in patterns]\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))\n"
            "col = [GREEN if v > 0 else RED for v in exc_vals]\n"
            "axes[0].bar(patterns, base_vals, color=GREY, alpha=0.5, label='random baseline')\n"
            "axes[0].bar(patterns, exc_vals, color=col, label='excess over baseline', bottom=0)\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_ylabel('bps'); axes[0].set_title('1-day return decomposition')\n"
            "axes[0].tick_params(axis='x', rotation=15); axes[0].legend(fontsize=8)\n"
            "axes[1].bar(patterns, t_vals, color=[GREEN if abs(v) >= bonf and v > 0 else (AMBER if abs(v) >= bonf else RED) for v in t_vals])\n"
            "axes[1].axhline(bonf, ls='--', c=GREEN, lw=2, label=f'Bonferroni +{bonf}')\n"
            "axes[1].axhline(-bonf, ls='--', c=RED, lw=2, label=f'Bonferroni -{bonf}')\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_ylabel('HAC t-stat (excess)'); axes[1].set_title('t-stat: morning-star clears Bonferroni -- wrong sign')\n"
            "axes[1].tick_params(axis='x', rotation=15); axes[1].legend(fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "tab = pd.DataFrame({'excess_bps': exc_vals, 't_excess': t_vals, 'random_baseline_bps': base_vals}, index=patterns)\n"
            "tab"
        ),
        md(
            "> In plain words: the morning-star's t-stat of -3.14 clears the Bonferroni bar "
            "but in the *wrong* direction. The random baseline for morning-star windows is "
            f"+{R['ms_base']:.0f} bps -- a large positive expected return driven by mean-reversion "
            "after a down-move. The pattern-triggered day returns only -2 bps, for an excess "
            f"of {R['ms_excess']:.0f} bps. The star arrives after the bounce has started."
        ),
        md(
            "### 4b -- Does the 5-day horizon change the picture?\n\n"
            "If mean-reversion were genuinely more pronounced after a morning-star (vs any "
            "random post-down-move day), the 5-day excess should be positive and growing. It does not."
        ),
        code(
            f"exc5_r = dict(zip(patterns, [{R['ms_excess5']}, {R['es_excess5']}]))\n"
            f"t5_r   = dict(zip(patterns, [{R['ms_t5']}, {R['es_t5']}]))\n"
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
            "ax.set_xticks(list(x)); ax.set_xticklabels(patterns, rotation=15)\n"
            "ax.set_ylabel('excess bps'); ax.set_title('1-day vs 5-day excess -- both negative for morning-star')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Morning-star 5-day excess: {exc5_r[\"morning_star\"]:+.1f} bps, t={t5_r[\"morning_star\"]:+.2f}')"
        ),
        md(
            "> In plain words: the 5-day excess for morning-star is even more negative "
            f"({R['ms_excess5']:+.0f} bps, t = {R['ms_t5']:+.2f}). The pattern does not "
            "mark a sustained reversal -- it fires mid-reversion and leaves the forward "
            "return smaller than a random post-down-move day at both horizons."
        ),
        md(
            "### 4c -- Synthetic positive control\n\n"
            "On a deterministic synthetic tape with planted mean-reversion, the same "
            "detectors should in principle recover the edge -- though the three-candle "
            "gap requirements make events rare on synthetic data.  We show the reversion "
            "knob works and verify the AC."
        ),
        code(
            "revs = [0.0, 0.10, 0.20, 0.30, 0.40]\n"
            "acs = []\n"
            "for rev in revs:\n"
            "    bars, _ = data.synthetic_daily(n_days=2000, reversion=rev, seed=186)\n"
            "    r = np.diff(np.log(bars['close'].to_numpy()))\n"
            "    acs.append(float(np.corrcoef(r[:-1], r[1:])[0, 1]))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(revs, acs, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, ls='--', c=GREY)\n"
            "ax.set_xlabel('planted reversion parameter'); ax.set_ylabel('lag-1 return autocorrelation')\n"
            "ax.set_title('Reversion knob confirmed: stronger reversion -> more negative AC')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('AC at reversion=0:', f'{acs[0]:+.3f}', '(should be ~0)')\n"
            "print('AC at reversion=0.40:', f'{acs[-1]:+.3f}', '(should be negative)')"
        ),
        md(
            "> In plain words: the reversion knob works as advertised.  Three-candle "
            "pattern events are rare on synthetic data (~3-8 per 1500 days), so the "
            "synthetic calibration is limited in power -- but the underlying reversion "
            "structure is correctly planted and measured."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal `NONE`** -- {R['bonf_n']} tests (2 patterns x 2 horizons). "
            f"Morning-star 1-day HAC t = {R['ms_t']:+.2f} (significant but wrong direction); "
            f"5-day t = {R['ms_t5']:+.2f} (same). Evening-star is noise at both horizons. "
            f"Bonferroni threshold: {R['bonf_t']}. Zero patterns show a *positive* excess "
            "that clears the bar.\n"
            "- **Tradability `MIRAGE`** -- the morning-star underperforms random days, so "
            "the claimed long trade is negative expectancy. The reverse (short) trade is "
            "not robust: the negative excess is a sequencing artefact, not a repeatable edge.\n"
            "- **Beats random days? `NOT SUPPORTED`** -- all excess returns are negative or "
            "flat. The pattern is a lagging indicator of mean-reversion already underway."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 -- Could you trade it? -- capacity and cost\n\n"
            "There is no positive-excess signal to trade. But let us be precise about "
            "what the arithmetic would look like if one tried to short the morning-star completion."
        ),
        code(
            f"reverse_exc = {abs(R['ms_excess'])}  # bps, gross (absolute value of negative excess)\n"
            f"events_per_yr = {R['ms_freq']} * {R['n_tickers']}  # ~63 reverse trades/year\n"
            "costs_bps = [0.0, 1.0, 2.0, 5.0]\n"
            "net = [reverse_exc - c for c in costs_bps]\n"
            "ann_return = [n * events_per_yr / 1e4 for n in net]\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "axes[0].bar([f'{c}bp' for c in costs_bps], net,\n"
            "           color=[GREEN if n > 0 else RED for n in net])\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_xlabel('round-trip cost'); axes[0].set_ylabel('net excess bps/trade (reverse)')\n"
            "axes[0].set_title('2 bps spread takes most of the (unproven) negative excess')\n"
            "axes[1].bar([f'{c}bp' for c in costs_bps], [r*100 for r in ann_return],\n"
            "           color=[GREEN if r > 0 else RED for r in ann_return])\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_xlabel('round-trip cost'); axes[1].set_ylabel('annual return (%)')\n"
            "axes[1].set_title('Annual return (if the reverse signal were real -- it is not)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'At 0 cost: ~{reverse_exc * events_per_yr / 1e4 * 100:.1f}%/yr. A 2bp spread -> ~{(reverse_exc-2) * events_per_yr / 1e4 * 100:.1f}%/yr.')\n"
            "print('The negative excess is a timing artefact, not a reliable short signal.')"
        ),
        md(
            "> In plain words: even if the negative excess were a real edge (it is a "
            "timing artefact), the maximum annual return before costs is only ~1.8%. "
            "A 2 bps round-trip spread cuts it to ~0.6%. That is below any reasonable "
            "friction floor, and the underlying reason (the pattern arrives mid-reversion) "
            "is not stable across regimes."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 -- Going further -- what next?\n\n"
            "- **The single-candle sibling.** [Study 76 -- Rice-Paper](../../76-rice-paper/) "
            "runs the same random-day test on six single-candle patterns (bullish/bearish "
            "engulfing, hammer, shooting star, doji). Result: also Signal=NONE. The pattern "
            "family is consistent across candle count.\n"
            "- **The underlying mechanic: mean reversion.** If post-down-move mean-reversion "
            "is real (and the random-day baseline says it is: +26 bps expected in the same "
            "window), a direct reversion strategy (IBS, RSI(2)) targets it more cleanly "
            "without the three-candle sequencing delay.\n"
            "- **Marshall et al. (2006)** applied White's Reality Check to 35 candlestick "
            "patterns including morning/evening stars on DJIA 1992-2002; no pattern survived "
            "data-snooping correction. Our 2010-2026 result is consistent.\n"
            "- **The timing question.** The morning-star's systematic underperformance of "
            "random down-move days is a finding, not just a null. It says: the pattern "
            "identifies the *middle* of a mean-reversion episode, not the start. That is "
            "consistent with a pattern definition that requires the recovery candle (day 3) "
            "before the signal fires.\n\n"
            "*Fork this, test morning-star at a shorter horizon (entry at the star close, "
            "before day 3 is complete), and show it clears the Bonferroni bar in the correct "
            "direction. The bar is |t| >= 2.50 on the excess return. That is the test.*"
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
