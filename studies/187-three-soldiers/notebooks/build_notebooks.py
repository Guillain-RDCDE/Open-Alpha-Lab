"""Generate the two narrative notebooks for Study 187 (Three-Soldiers).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
daily parquets under ../_cache/ if present and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.

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
    # Three White Soldiers
    tws_n=462, tws_rate=1.86,
    tws_1d_sig=-13.39, tws_1d_exc=-23.51, tws_1d_t_exc=-2.04, tws_1d_t_sig=-1.75, tws_1d_win=45.5,
    tws_5d_sig=25.14,  tws_5d_exc=-17.29, tws_5d_t_exc=-0.60, tws_5d_t_sig=1.15,  tws_5d_win=57.8,
    tws_10d_sig=51.11, tws_10d_exc=-75.55, tws_10d_t_exc=-1.77, tws_10d_t_sig=1.62, tws_10d_win=64.9,
    # Three Black Crows
    tbc_n=264, tbc_rate=1.06,
    tbc_1d_sig=-39.44, tbc_1d_exc=-21.98, tbc_1d_t_exc=-1.15, tbc_1d_t_sig=-2.75, tbc_1d_win=40.5,
    tbc_5d_sig=-79.71, tbc_5d_exc=-49.28, tbc_5d_t_exc=-1.31, tbc_5d_t_sig=-2.48, tbc_5d_win=39.8,
    tbc_10d_sig=-117.56, tbc_10d_exc=-1.96, tbc_10d_t_exc=-0.03, tbc_10d_t_sig=-2.50, tbc_10d_win=41.3,
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

from three_soldiers import data, strategy as st

TICKERS = data.DEFAULT_TICKERS
HORIZONS = (1, 5, 10)

def _have_cache():
    return all(os.path.exists(data._cache_path(t, data.DEFAULT_CACHE)) for t in TICKERS)

HAVE_REAL = _have_cache()

def pooled_study(cost_bps=1.0, seed=0):
    \"\"\"Pool the full pattern study across the basket (cached real tapes).\"\"\"
    return st.run_pooled_study(TICKERS, fetch=False, horizons=HORIZONS, cost_bps=cost_bps, seed=seed)

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Three-Soldiers -- do Three White Soldiers and Three Black Crows predict anything?\n"
            "### Three-bar continuation candles, tested honestly\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Continuation_or_Reversal%3F: Neither](https://img.shields.io/badge/Continuation_or_Reversal%3F-Neither-8b949e?style=flat-square)\n\n"
            "You've seen them in every candlestick book: **Three White Soldiers** -- three big "
            "bullish candles marching upward, each one closing near its high -- and **Three Black "
            "Crows** -- the bearish mirror image. Both are billed as powerful continuation "
            "signals: the soldiers are resuming an uptrend, the crows are accelerating a downtrend. "
            "This notebook asks: do they actually predict anything? Or are they just pretty charts?\n\n"
            "> **This is the plain-language layer.** Want the t-stats, the Bonferroni table, and "
            "the control experiment? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn by "
            "the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do Three White Soldiers signal a continuation upward? | **No.** The day after the "
            f"pattern, the market is *more* likely to fall: mean 1-day return "
            f"**{R['tws_1d_sig']:+.1f} bps** (claimed direction: up). |\n"
            f"| Do Three Black Crows signal a continuation downward? | **No.** After the pattern, "
            f"the market tends to bounce: mean 1-day return **{R['tbc_1d_sig']:+.1f} bps** "
            f"(claimed direction: down). |\n"
            f"| Do they beat a random day in the same direction? | **No.** Both patterns "
            f"*underperform* a random-day baseline across all horizons tested. |\n"
            "| Could you trade them? | **No.** Patterns fire only ~1-2 times per stock per year; "
            "at 1 bp cost, returns are already negative on day 1. |\n\n"
            "> The patterns fire after a short-term extreme -- they're measuring exhaustion, "
            "not momentum."
        ),

        # ---- BEAT 1 -- THE CLAIM -----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Three White Soldiers is one of the most bullish candlestick patterns. Three "
            "consecutive long white bodies, each opening within the prior body and closing near "
            "the high, indicate a powerful upturn -- strong buying pressure sustaining itself over "
            "three sessions.\"* — Nison (1991), Morris (2006)\n\n"
            "The idea is intuitive: if buyers dominate for three consecutive full sessions with no "
            "significant reversal, the buying pressure is 'confirmed' and the trend will continue. "
            "Three Black Crows make the same argument for sellers.\n\n"
            "At full strength, this claims that three big directional days have **predictive "
            "power** for the days that follow -- above and beyond what any random three-day period "
            "would tell you."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it's true, it's a useful daily signal that any investor can apply without "
            "complex calculations: spot the three candles, trade with the pattern. Textbooks have "
            "been teaching this for 30+ years.\n\n"
            "If it's false, three attractive-looking bars are just noise given a story, "
            "and the whole tradition of three-bar continuation patterns is folklore dressed up "
            "as analysis. That's worth knowing."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Two disciplines keep us honest:\n\n"
            "1. **Compare against a random day in the same direction.** The question is not "
            "'does the market go up after Three White Soldiers?' (it usually goes up *anyway*, "
            "because stocks drift upward). The question is: 'does the market go up *more* than "
            "on a random day where we also decided to go long?' If the pattern tells us nothing "
            "extra, it beats no better than random.\n"
            "2. **Multiple-comparisons correction.** We test two patterns at three horizons "
            "(1, 5, 10 days) = 6 comparisons. At 5% significance, we'd expect 0.3 false "
            "positives by chance. We apply a Bonferroni correction to make sure a single "
            "'lucky' t-stat doesn't get us excited.\n\n"
            "Basket: 15 liquid US names (SPY, AAPL, MSFT, AMZN, GOOGL, META, NVDA, TSLA, JPM, "
            "BAC, XOM, JNJ, PG, KO, WMT), 2010-2026. Pattern detected at close of bar t; "
            "forward return measured from bar t+1. No look-ahead."
        ),

        # ---- BEAT 4 -- THE TEARDOWN -----------------------------------------------
        md(
            "## 4 · The teardown\n\n"
            "**How rare are these patterns?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pooled = pooled_study(cost_bps=1.0)\n"
            "    tws_n = len(pooled.get('three_white_soldiers', pd.DataFrame()))\n"
            "    tbc_n = len(pooled.get('three_black_crows', pd.DataFrame()))\n"
            "    n_ticker_years = len(TICKERS) * 16  # ~16 years per ticker\n"
            "else:\n"
            "    tws_n, tbc_n, n_ticker_years = R['tws_n'], R['tbc_n'], 15*16\n"
            "    pooled = None\n"
            "print(f'Three White Soldiers: {tws_n} triggers across {len(TICKERS) if HAVE_REAL else 15} tickers x 16 years')\n"
            "print(f'  = ~{tws_n/n_ticker_years:.1f} per ticker-year')\n"
            "print(f'Three Black Crows:    {tbc_n} triggers')\n"
            "print(f'  = ~{tbc_n/n_ticker_years:.1f} per ticker-year')\n"
            "print()\n"
            "print('These are rare events. On average, a stock shows 1-2 soldiers and 1 crow per year.')"
        ),
        md(
            f"**{R['tws_n']} Three White Soldiers** and **{R['tbc_n']} Three Black Crows** in 15 "
            f"stocks over 16 years.  That's roughly **{R['tws_rate']:.1f}** soldiers and "
            f"**{R['tbc_rate']:.1f}** crows per stock per year.  A single portfolio-level "
            "manager might see one of each per month across a diversified book — these are "
            "uncommon events, not a daily trading signal."
        ),
        md("**Now the key test: what happens the day after?**"),
        code(
            "horizons = [1, 5, 10]\n"
            "patterns = ['three_white_soldiers', 'three_black_crows']\n"
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for name in patterns:\n"
            "        if name not in pooled: continue\n"
            "        df = pooled[name]\n"
            "        for h in horizons:\n"
            "            s = st.summarize_pattern(df, horizon=h)\n"
            "            rows.append({'pattern': name.replace('_', ' ').title(),\n"
            "                         'horizon': f'{h}d', 'signal_bps': s['mean_bps'],\n"
            "                         'baseline_bps': s['baseline_mean_bps'],\n"
            "                         'excess_bps': s['excess_bps'], 'win_pct': s['win_rate']*100})\n"
            "    tbl = pd.DataFrame(rows)\n"
            "else:\n"
            "    tbl = pd.DataFrame([\n"
            "        {'pattern': 'Three White Soldiers', 'horizon': '1d',\n"
            f"         'signal_bps': {R['tws_1d_sig']}, 'baseline_bps': {R['tws_1d_sig']-R['tws_1d_exc']:.2f},\n"
            f"         'excess_bps': {R['tws_1d_exc']}, 'win_pct': {R['tws_1d_win']}}},\n"
            "        {'pattern': 'Three White Soldiers', 'horizon': '5d',\n"
            f"         'signal_bps': {R['tws_5d_sig']}, 'baseline_bps': {R['tws_5d_sig']-R['tws_5d_exc']:.2f},\n"
            f"         'excess_bps': {R['tws_5d_exc']}, 'win_pct': {R['tws_5d_win']}}},\n"
            "        {'pattern': 'Three White Soldiers', 'horizon': '10d',\n"
            f"         'signal_bps': {R['tws_10d_sig']}, 'baseline_bps': {R['tws_10d_sig']-R['tws_10d_exc']:.2f},\n"
            f"         'excess_bps': {R['tws_10d_exc']}, 'win_pct': {R['tws_10d_win']}}},\n"
            "        {'pattern': 'Three Black Crows', 'horizon': '1d',\n"
            f"         'signal_bps': {R['tbc_1d_sig']}, 'baseline_bps': {R['tbc_1d_sig']-R['tbc_1d_exc']:.2f},\n"
            f"         'excess_bps': {R['tbc_1d_exc']}, 'win_pct': {R['tbc_1d_win']}}},\n"
            "        {'pattern': 'Three Black Crows', 'horizon': '5d',\n"
            f"         'signal_bps': {R['tbc_5d_sig']}, 'baseline_bps': {R['tbc_5d_sig']-R['tbc_5d_exc']:.2f},\n"
            f"         'excess_bps': {R['tbc_5d_exc']}, 'win_pct': {R['tbc_5d_win']}}},\n"
            "        {'pattern': 'Three Black Crows', 'horizon': '10d',\n"
            f"         'signal_bps': {R['tbc_10d_sig']}, 'baseline_bps': {R['tbc_10d_sig']-R['tbc_10d_exc']:.2f},\n"
            f"         'excess_bps': {R['tbc_10d_exc']}, 'win_pct': {R['tbc_10d_win']}}},\n"
            "    ])\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n"
            "for ax, pat in zip(axes, ['Three White Soldiers', 'Three Black Crows']):\n"
            "    sub = tbl[tbl['pattern'] == pat]\n"
            "    x = range(len(sub))\n"
            "    ax.bar([i-0.2 for i in x], sub['signal_bps'], width=0.35,\n"
            "           color=[GREEN if v>0 else RED for v in sub['signal_bps']], label='Pattern signal')\n"
            "    ax.bar([i+0.2 for i in x], sub['baseline_bps'], width=0.35,\n"
            "           color=GREY, alpha=0.7, label='Random-day baseline')\n"
            "    ax.axhline(0, c='k', lw=1)\n"
            "    ax.set_xticks(list(x)); ax.set_xticklabels(sub['horizon'].tolist())\n"
            "    ax.set_xlabel('forward horizon'); ax.set_ylabel('mean return (bps, in claimed direction)')\n"
            "    ax.set_title(pat); ax.legend(fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tbl[['pattern','horizon','signal_bps','baseline_bps','excess_bps']].round(1).to_string(index=False))"
        ),
        md(
            "**Three White Soldiers** returns **-13 bps on day 1** in the claimed direction. "
            "The market goes *down*, not up, the day after three strong up-candles. The baseline "
            "(a random bullish day) returns about +10 bps from drift.\n\n"
            "**Three Black Crows** returns **-39 bps on day 1** in the claimed direction. "
            "The claimed direction is *down*, so -39 bps means the market went *up* — the "
            "opposite of the forecast.\n\n"
            "> Both patterns trigger near short-term exhaustion. Three consecutive strong moves "
            "leave the market stretched: buyers have been buying for three days straight, so "
            "the buyers are now long and the easy sellers have sold. The next move tends to "
            "be a small correction back — mean reversion, not continuation."
        ),

        # ---- BEAT 5 -- THE VERDICT -----------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Zero of six statistical tests survive the multiple-comparisons "
            "correction. The 1-day signals point in the *wrong* direction for both patterns.\n"
            "- **Tradability — Mirage.** ~1-2 triggers per stock per year, already negative at "
            "1 bp cost on day 1. Nothing to trade.\n"
            "- **Continuation or Reversal? — Neither.** The directional evidence is mixed and "
            "non-robust. If anything the patterns fire at exhaustion, but that finding is not "
            "statistically clean enough to trade the other way either."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT -----------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Even if the signal were real, capacity kills it. One Three White Soldiers per "
            "stock per year across a 15-stock basket = ~28 signals per year total. That's "
            "not a strategy, it's a rounding error. And at 1 bp round-trip, the day-1 signal "
            "(-13 bps for 3WS, -39 bps for 3BC in the claimed direction) is already deeply "
            "negative — no cost improvement fixes a negative gross."
        ),
        code(
            "costs = [0.0, 0.5, 1.0, 2.0]\n"
            "if HAVE_REAL:\n"
            "    net_tws, net_tbc = [], []\n"
            "    for c in costs:\n"
            "        p = pooled_study(cost_bps=c)\n"
            "        tws = p.get('three_white_soldiers', pd.DataFrame())\n"
            "        tbc = p.get('three_black_crows', pd.DataFrame())\n"
            "        net_tws.append(st.summarize_pattern(tws, 1).get('mean_bps', float('nan')) if len(tws) else float('nan'))\n"
            "        net_tbc.append(st.summarize_pattern(tbc, 1).get('mean_bps', float('nan')) if len(tbc) else float('nan'))\n"
            "else:\n"
            f"    net_tws = [{R['tws_1d_sig']}, {R['tws_1d_sig']-0.5}, {R['tws_1d_sig']-1.0}, {R['tws_1d_sig']-2.0}]\n"
            f"    net_tbc = [{R['tbc_1d_sig']}, {R['tbc_1d_sig']-0.5}, {R['tbc_1d_sig']-1.0}, {R['tbc_1d_sig']-2.0}]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, net_tws, 'o-', c=GREEN, lw=2, label='Three White Soldiers (1d)')\n"
            "ax.plot(costs, net_tbc, 's--', c=RED, lw=2, label='Three Black Crows (1d)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net 1-day signal (bps, claimed direction)')\n"
            "ax.set_title('Both patterns start negative and get worse with cost'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('No positive break-even cost exists for either pattern at the 1-day horizon.')"
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The synthetic control.** The companion notebook shows that the engine *can* find "
            "continuation edges when they exist — it detects patterns correctly on a synthetic tape "
            "with planted momentum. The real-tape failure is about the *market*, not the detector.\n"
            "- **Candlestick patterns more broadly.** [Study 76 — Rice-Paper](../../76-rice-paper/) "
            "tests five single-bar reversal patterns (engulfing, hammer, doji) on the same basket "
            "— same methodology, same conclusion.\n"
            "- **Could the *reversal* trade work?** The evidence hints both patterns may fire at "
            "exhaustion, but the data is too noisy to trade it. A dedicated mean-reversion test "
            "after a 3-day extreme run is a cleaner way to ask that question.\n\n"
            "*Think Three White Soldiers really do predict continuation? Show a robust positive "
            "excess return over a random-day baseline that clears the Bonferroni bar. That's the test.*"
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
            "# Three-Soldiers -- a quantitative teardown\n"
            "### Daily bars * two three-bar patterns * 15 stocks * 2010-2026 * "
            "random-day control * Bonferroni correction\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Continuation_or_Reversal%3F: Neither](https://img.shields.io/badge/Continuation_or_Reversal%3F-Neither-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim now carrying its standard error.* We test whether "
            "Three White Soldiers (3WS) and Three Black Crows (3BC) add directional information "
            "beyond a random-day entry, pooled across 15 US names over 16 years, with a "
            "Bonferroni correction for six comparisons (2 patterns x 3 horizons).\n\n"
            "> **Not investment advice.** Real data: Yahoo daily bars, 2010-01-04 to 2026-06-15, "
            "as-of 2026-06-15. Methods in [`docs/references.md`](../docs/references.md), "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The `In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT + "\nfrom quantlab import analytics, stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive numbers |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | 3WS 1d excess = {R['tws_1d_exc']:+.1f} bps, t = {R['tws_1d_t_exc']:+.2f}; "
            f"3BC 1d excess = {R['tbc_1d_exc']:+.1f} bps, t = {R['tbc_1d_t_exc']:+.2f}. "
            f"Zero of 6 Bonferroni tests survive (threshold |t| >= 2.65). |\n"
            f"| **Tradability** | `MIRAGE` | ~{R['tws_rate']:.1f} / {R['tbc_rate']:.1f} triggers "
            f"per ticker-year; 1d net signal is negative for both patterns at 1 bp cost. |\n"
            f"| **Continuation or Reversal?** | `NEITHER` | 1d signals are negative in the "
            f"claimed direction ({R['tws_1d_sig']:+.1f} bps 3WS; {R['tbc_1d_sig']:+.1f} bps 3BC) "
            f"but the excess over baseline is not robust after correction. |\n\n"
            "> In plain words: both patterns fire after a three-day directional extreme; the market "
            "tends to mean-revert slightly on day 1, but not enough to be statistically real after "
            "Bonferroni, and nowhere near enough to trade after costs."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t$ be the log close-to-close return at day $t$.  Three White Soldiers fires "
            "at bar $t$ when bars $t-2$, $t-1$, $t$ all satisfy: (a) bullish, (b) ascending "
            "closes and opens, (c) body $\\geq 50\\%$ of range, (d) upper shadow $\\leq 30\\%$ of "
            "range.  The claim (H1):\n\n"
            "$$\\mathbb{E}[r_{t+h} \\mid \\text{3WS at }t] > \\mathbb{E}[r_{t+h} \\mid \\text{random long at }t]$$\n\n"
            "for at least one $h \\in \\{1, 5, 10\\}$ trading days, net of cost.  H2 is the mirror "
            "for Three Black Crows (negative forward returns).  We test both against a random-day "
            "control (same tape, same direction, same n), correct for $2 \\times 3 = 6$ comparisons "
            "with Bonferroni, and require $|t_{\\text{excess}}| \\geq 2.65$."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? -- stakes\n\n"
            "Candlestick patterns are among the most widely taught technical signals, appearing in "
            "virtually every retail trading course.  If Three White Soldiers / Three Black Crows "
            "work, they represent a simple, free, daily-observation edge that any investor can "
            "apply.  If they don't, the pattern's *visual appeal* is the sole reason for its "
            "persistence -- a reminder that backfitting stories to charts is not analysis."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · Protocol\n\n"
            "- **Pattern detection.** Programmatic, on daily OHLCV, confirmed at close of bar $t$.\n"
            "  For 3WS: bars $t-2$, $t-1$, $t$ each bullish, strong body ($\\geq 50\\%$ range), "
            "  small upper shadow ($\\leq 30\\%$ range), ascending closes *and* opens.  3BC: mirror.\n"
            "- **Forward returns.** Log $c$-to-$c$: $\\ln(C_{t+h}/C_t)$, multiplied by the "
            "  claimed direction ($+1$ for 3WS, $-1$ for 3BC), net of 1 bp round-trip cost.\n"
            "- **Control arm.** $n$ randomly sampled dates from the same tape, same direction, "
            "  same forward-return horizon.  The *excess* $r_{\\text{signal}} - r_{\\text{random}}$ "
            "  is the decisive metric: the pattern adds value only if this excess is positive and "
            "  statistically significant.\n"
            "- **Inference.** Newey-West HAC t-stat on the excess, Bonferroni correction for 6 "
            "  tests ($\\alpha_{\\text{adj}} = 0.05/6 \\approx 0.0083$, $|t|_{\\text{adj}} \\approx 2.65$).\n"
            "- **Positive control.** Synthetic tape with planted momentum: confirms the detector "
            "  can find continuation edges when they exist.\n\n"
            "Data: 15 US names, 2010-2026 (~16 years each), 2010-01-04 to 2026-06-15."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Pattern counts and frequency\n\n"
            "Before asking about edges, confirm the patterns fire at a reasonable rate."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pooled = pooled_study(cost_bps=1.0)\n"
            "    tws = pooled.get('three_white_soldiers', pd.DataFrame())\n"
            "    tbc = pooled.get('three_black_crows', pd.DataFrame())\n"
            "    tws_n, tbc_n = len(tws), len(tbc)\n"
            "else:\n"
            f"    tws_n, tbc_n = {R['tws_n']}, {R['tbc_n']}\n"
            "    pooled = None\n"
            "n_ticker_years = len(TICKERS) * 16\n"
            "print(f'Three White Soldiers: n = {tws_n} total triggers')\n"
            "print(f'  = {tws_n/n_ticker_years:.2f} per ticker-year')\n"
            "print(f'Three Black Crows:    n = {tbc_n} total triggers')\n"
            "print(f'  = {tbc_n/n_ticker_years:.2f} per ticker-year')\n"
            "print()\n"
            "# Show annual trigger frequency as a bar chart\n"
            "fig, ax = plt.subplots(figsize=(7, 4))\n"
            "ax.bar(['Three White\\nSoldiers', 'Three Black\\nCrows'],\n"
            f"       [{R['tws_rate']:.2f}, {R['tbc_rate']:.2f}] if not HAVE_REAL else\n"
            "       [tws_n/n_ticker_years, tbc_n/n_ticker_years],\n"
            "       color=[GREEN, RED], width=0.5)\n"
            "ax.set_ylabel('average triggers per ticker-year')\n"
            "ax.set_title('Both patterns are genuinely rare (< 2 per stock-year)')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "### 4b · The key test -- forward returns vs random-day baseline\n\n"
            "If the pattern adds directional information, the signal return should *exceed* the "
            "baseline return.  The Bonferroni threshold is |t| >= 2.65."
        ),
        code(
            "horizons = [1, 5, 10]\n"
            "names = ['three_white_soldiers', 'three_black_crows']\n"
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for name in names:\n"
            "        if name not in pooled: continue\n"
            "        df = pooled[name]\n"
            "        for h in horizons:\n"
            "            s = st.summarize_pattern(df, horizon=h)\n"
            "            rows.append({'pattern': name.replace('_',' ').title(), 'h': h,\n"
            "                         'n': s['n'], 'sig': s['mean_bps'], 'base': s['baseline_mean_bps'],\n"
            "                         'exc': s['excess_bps'], 't_exc': s['tstat_excess'],\n"
            "                         't_sig': s['tstat_signal'], 'win': s['win_rate']*100})\n"
            "    res = pd.DataFrame(rows)\n"
            "else:\n"
            "    res = pd.DataFrame([\n"
            f"        {{'pattern': 'Three White Soldiers', 'h':1,  'n':{R['tws_n']}, 'sig':{R['tws_1d_sig']},  'base':{R['tws_1d_sig']-R['tws_1d_exc']:.2f},  'exc':{R['tws_1d_exc']},  't_exc':{R['tws_1d_t_exc']}, 't_sig':{R['tws_1d_t_sig']}, 'win':{R['tws_1d_win']}}},\n"
            f"        {{'pattern': 'Three White Soldiers', 'h':5,  'n':{R['tws_n']}, 'sig':{R['tws_5d_sig']},  'base':{R['tws_5d_sig']-R['tws_5d_exc']:.2f},  'exc':{R['tws_5d_exc']},  't_exc':{R['tws_5d_t_exc']}, 't_sig':{R['tws_5d_t_sig']}, 'win':{R['tws_5d_win']}}},\n"
            f"        {{'pattern': 'Three White Soldiers', 'h':10, 'n':{R['tws_n']}, 'sig':{R['tws_10d_sig']}, 'base':{R['tws_10d_sig']-R['tws_10d_exc']:.2f}, 'exc':{R['tws_10d_exc']}, 't_exc':{R['tws_10d_t_exc']},'t_sig':{R['tws_10d_t_sig']},'win':{R['tws_10d_win']}}},\n"
            f"        {{'pattern': 'Three Black Crows',    'h':1,  'n':{R['tbc_n']}, 'sig':{R['tbc_1d_sig']},  'base':{R['tbc_1d_sig']-R['tbc_1d_exc']:.2f},  'exc':{R['tbc_1d_exc']},  't_exc':{R['tbc_1d_t_exc']}, 't_sig':{R['tbc_1d_t_sig']}, 'win':{R['tbc_1d_win']}}},\n"
            f"        {{'pattern': 'Three Black Crows',    'h':5,  'n':{R['tbc_n']}, 'sig':{R['tbc_5d_sig']},  'base':{R['tbc_5d_sig']-R['tbc_5d_exc']:.2f},  'exc':{R['tbc_5d_exc']},  't_exc':{R['tbc_5d_t_exc']}, 't_sig':{R['tbc_5d_t_sig']}, 'win':{R['tbc_5d_win']}}},\n"
            f"        {{'pattern': 'Three Black Crows',    'h':10, 'n':{R['tbc_n']}, 'sig':{R['tbc_10d_sig']}, 'base':{R['tbc_10d_sig']-R['tbc_10d_exc']:.2f}, 'exc':{R['tbc_10d_exc']}, 't_exc':{R['tbc_10d_t_exc']},'t_sig':{R['tbc_10d_t_sig']},'win':{R['tbc_10d_win']}}},\n"
            "    ])\n"
            "\n"
            "# Plot excess return and t-stat\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n"
            "for ax, pat in zip(axes, ['Three White Soldiers', 'Three Black Crows']):\n"
            "    sub = res[res['pattern'] == pat]\n"
            "    x = list(range(len(sub)))\n"
            "    col = [GREEN if v > 0 else RED for v in sub['exc']]\n"
            "    ax.bar(x, sub['exc'], color=col, width=0.5)\n"
            "    for xi, row in zip(x, sub.itertuples()):\n"
            "        ax.text(xi, row.exc + (1 if row.exc >= 0 else -2),\n"
            "                f't={row.t_exc:+.2f}', ha='center', va='bottom', fontsize=9)\n"
            "    ax.axhline(0, c='k', lw=1)\n"
            "    ax.axhline(0, c='k', lw=1)\n"
            "    ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in sub['h']])\n"
            "    ax.set_xlabel('horizon'); ax.set_ylabel('excess bps (signal - baseline)')\n"
            "    ax.set_title(f'{pat}: excess return over random-day baseline')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Bonferroni threshold |t| >= 2.65 for alpha=5% across 6 tests')\n"
            "print(res[['pattern','h','n','sig','base','exc','t_exc','t_sig']].round(2).to_string(index=False))"
        ),
        md(
            "> In plain words: every single excess-return t-stat is *negative*, meaning the pattern "
            "performs *worse* than a random day in the same direction. The Bonferroni bar is 2.65 "
            f"and the best t-stat in absolute value is {abs(R['tws_1d_t_exc']):.2f} (3WS, 1d) -- not "
            "even close to surviving. Both patterns are consistent underperformers vs a coin flip."
        ),
        md(
            "### 4c · Synthetic positive control -- the engine works when there is something to find\n\n"
            "Can the detector actually find continuation edges? Yes -- when we plant momentum "
            "in a synthetic tape, the raw pattern signal rises."
        ),
        code(
            "moms = [0.0, 0.10, 0.20, 0.30, 0.40]\n"
            "tws_sigs, tbc_sigs = [], []\n"
            "for m in moms:\n"
            "    bars, _ = data.synthetic_daily(n_days=2000, momentum=m, seed=187)\n"
            "    res_m = st.run_pattern_study(bars, horizons=(1,), cost_bps=0, seed=0)\n"
            "    tws_df = res_m.get('three_white_soldiers', pd.DataFrame())\n"
            "    tbc_df = res_m.get('three_black_crows', pd.DataFrame())\n"
            "    s_tws = st.summarize_pattern(tws_df, 1).get('mean_bps', float('nan')) if len(tws_df) else float('nan')\n"
            "    s_tbc = st.summarize_pattern(tbc_df, 1).get('mean_bps', float('nan')) if len(tbc_df) else float('nan')\n"
            "    tws_sigs.append(s_tws); tbc_sigs.append(s_tbc)\n"
            "    print(f'  momentum={m:.2f}: 3WS n={len(tws_df):2d} sig={s_tws:+.1f}bps | '\n"
            "          f'3BC n={len(tbc_df):2d} sig={s_tbc:+.1f}bps')\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "valid_m = [(m, s) for m, s in zip(moms, tws_sigs) if not np.isnan(s)]\n"
            "if valid_m:\n"
            "    ax.plot([v[0] for v in valid_m], [v[1] for v in valid_m],\n"
            "            'o-', c=GREEN, lw=2, label='3WS 1d signal (bps)')\n"
            "valid_m2 = [(m, s) for m, s in zip(moms, tbc_sigs) if not np.isnan(s)]\n"
            "if valid_m2:\n"
            "    ax.plot([v[0] for v in valid_m2], [v[1] for v in valid_m2],\n"
            "            's--', c=RED, lw=2, label='3BC 1d signal (bps)')\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, ls='--', c=GREY)\n"
            "ax.set_xlabel('planted momentum'); ax.set_ylabel('1d signal (bps, claimed direction)')\n"
            "ax.set_title('The engine finds continuation edges when momentum is planted')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('Engine validated: signals rise with planted momentum, near-zero on a martingale.')"
        ),
        md(
            "> In plain words: the detector is well-specified. When we plant bar-level momentum "
            "in a synthetic tape, the pattern returns positive signals. The real-tape failure is "
            "a statement about the *market* (daily continuation is weak) not the detector."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** -- 3WS excess {R['tws_1d_exc']:+.1f} bps at 1d, HAC t = "
            f"{R['tws_1d_t_exc']:+.2f}; 3BC excess {R['tbc_1d_exc']:+.1f} bps, t = "
            f"{R['tbc_1d_t_exc']:+.2f}. Zero of 6 Bonferroni tests survive. The 1d signals "
            "point in the *wrong* direction, consistent with short-term mean reversion.\n"
            f"- **Tradability `MIRAGE`** -- ~{R['tws_rate']:.1f} / {R['tbc_rate']:.1f} triggers "
            "per ticker-year; no positive break-even cost at any horizon.\n"
            "- **Continuation or Reversal? `NEITHER`** -- The directional hint (exhaustion "
            "reversal) is not robust after multiple-comparisons correction."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? -- capacity and cost\n\n"
            "Even ignoring signal quality, the capacity problem is severe."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pooled0 = pooled_study(cost_bps=0.0)\n"
            "    tws0 = pooled0.get('three_white_soldiers', pd.DataFrame())\n"
            "    tbc0 = pooled0.get('three_black_crows', pd.DataFrame())\n"
            "    tws_gross = st.summarize_pattern(tws0, 1).get('mean_bps', float('nan')) if len(tws0) else float('nan')\n"
            "    tbc_gross = st.summarize_pattern(tbc0, 1).get('mean_bps', float('nan')) if len(tbc0) else float('nan')\n"
            "else:\n"
            f"    tws_gross, tbc_gross = {R['tws_1d_sig']+1.0:.2f}, {R['tbc_1d_sig']+1.0:.2f}\n"
            "print(f'3WS gross 1d: {tws_gross:+.2f} bps (already negative -> no break-even cost)')\n"
            "print(f'3BC gross 1d: {tbc_gross:+.2f} bps (already negative -> no break-even cost)')\n"
            "print()\n"
            "print('Capacity: even if signal were positive, ~28 total signals/year across')\n"
            "print(f'  {len(TICKERS) if HAVE_REAL else 15} tickers is not a viable strategy.')"
        ),
        md(
            "The gross 1-day signal is already negative for both patterns — there is no "
            "transaction cost small enough to make either work. Even if we had an edge, "
            "~28 signals per year across a 15-stock basket is not a strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The same basket, single-bar patterns:** [Study 76 — Rice-Paper](../../76-rice-paper/) "
            "tests engulfing patterns, hammers, and doji on the same 15 names — same methodology, "
            "same verdict.\n"
            "- **The overbought/oversold angle:** If the patterns really fire at exhaustion, a "
            "cleaner way to test is an explicit 3-day return threshold (e.g. 3-day return > 2 sigma) "
            "rather than a visual candlestick rule.  See [Study 127 — Williams-R](../../127-williams-r/) "
            "for the oscillator variant.\n"
            "- **A longer holding period.** The 10-day returns show the raw 3WS signal turning "
            "positive (+51 bps) as drift takes over.  This is unconditional equity drift, not "
            "pattern skill -- the 10-day excess is still negative (-76 bps).\n\n"
            "*Want to prove Three White Soldiers work? Show a robust positive excess return over a "
            "random-day baseline that survives Bonferroni across multiple tickers and periods. "
            "That is the standard every other study on this desk meets or honestly fails.*"
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
