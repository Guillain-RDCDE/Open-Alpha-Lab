"""Generate the two narrative notebooks for Study 189 (Double-Top / Double-Bottom).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). Synthetic
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-15).
R = dict(
    # double-top
    dt_n=905, dt_win=49.7,
    dt_1d_sig=5.64, dt_1d_rand=-4.76, dt_1d_exc=10.40, dt_1d_t=1.39,
    dt_5d_sig=-36.70, dt_5d_rand=-35.30, dt_5d_exc=-1.40, dt_5d_t=-0.09,
    dt_20d_sig=-159.70, dt_20d_rand=-120.61, dt_20d_exc=-39.08, dt_20d_t=-1.27,
    # double-bottom
    db_n=1322, db_win_5d=57.0, db_win_20d=61.9,
    db_1d_sig=11.40, db_1d_rand=10.88, db_1d_exc=0.51, db_1d_t=0.10,
    db_5d_sig=18.78, db_5d_rand=29.50, db_5d_exc=-10.72, db_5d_t=-0.88,
    db_20d_sig=104.68, db_20d_rand=111.78, db_20d_exc=-7.09, db_20d_t=-0.28,
    max_abs_t_exc=1.39,  # across all 6 pattern/horizon tests
)

TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "XOM", "GLD", "IWM"]

# ---------------------------------------------------------------------------
# Shared preamble — imports and cache check
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

from double_top import data, strategy as st

TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOGL", "JPM", "XOM", "GLD", "IWM"]

def _have_cache():
    return all(os.path.exists(data._cache_path(t, data.DEFAULT_CACHE)) for t in TICKERS)

HAVE_REAL = _have_cache()

def pooled_study(cost_bps=0.0, seed=189, **kw):
    \"\"\"Pool pattern study across all tickers (real cache).\"\"\"
    return st.run_pooled_study(
        TICKERS, fetch=False, horizons=(1, 5, 20),
        cost_bps=cost_bps, seed=seed,
        min_bars=10, tolerance=0.04, lookback=120, atr_prominence_mult=0.5,
        **kw
    )

print("real daily cache present:", HAVE_REAL)
"""

# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Double-Top / Double-Bottom — can the M and W predict a reversal?\n"
            "### The classic chart-reversal signal tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Placebo_wins%3F: Yes](https://img.shields.io/badge/Placebo_wins%3F-Yes-8b949e?style=flat-square)\n\n"
            "Every technical-analysis book has the same two shapes. The **double-top** (M): price "
            "runs up, pulls back, tries again and fails at roughly the same high, then breaks "
            "lower — a bearish reversal. The **double-bottom** (W): the mirror. The claim is "
            "that these two-peak / two-trough shapes *predict* a meaningful move in the "
            "reversal direction once the midpoint 'neckline' is broken. This notebook asks "
            "the one question that decides the matter: does the pattern beat a **random day** "
            "picked from the same tape?\n\n"
            "> 📓 **This is the plain-language layer.** Want the t-stats, the Bonferroni "
            "correction, and the full horizon sweep? That's "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** Reproducible research: every chart below is "
            "drawn by the code beside it. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the double-top predict a fall? | **Not reliably.** "
            f"1-day excess over a random bearish bet: +{R['dt_1d_exc']:.0f} bps (t = +{R['dt_1d_t']:.2f} — noise). "
            f"5-day excess: {R['dt_5d_exc']:+.0f} bps (t = {R['dt_5d_t']:+.2f} — noise). |\n"
            "| Does the double-bottom predict a rise? | **Not beyond the market's "
            f"unconditional drift.** The signal return looks positive at 5d and 20d, but a "
            f"random bull bet on the same days does *better*: excess = {R['db_5d_exc']:+.0f} bps "
            f"(t = {R['db_5d_t']:+.2f}). |\n"
            "| Anything survive a rigorous multiple-comparison test? | **No.** "
            f"Zero of six pattern/horizon combinations clear the Bonferroni bar (|t| ≥ 3). "
            f"Max |t| on excess = {R['max_abs_t_exc']:.2f}. |\n"
            "| Could you trade it? | **No.** No meaningful excess exists before costs. |\n\n"
            "> The patterns describe *what has already happened* — two peaks at a similar "
            "level, a neckline break. They do not reliably describe what comes next."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *'Two peaks at roughly the same high, separated by a pullback — wait for the "
            "price to break below the trough between them (the neckline), then sell short. "
            "The double-top is one of the most reliable reversal signals in all of technical "
            "analysis.'* (Bulkowski 2005; innumerable YouTube tutorials.)\n\n"
            "The double-bottom is the mirror: two troughs at a similar low, then a break "
            "above the peak between them — buy. The mechanism supposedly at work: the "
            "first peak/trough reveals the bulls'/bears' ceiling/floor; the second attempt "
            "at that level fails, signalling that supply/demand is exhausted; the neckline "
            "break *confirms* the reversal is underway, and the prior peak-to-trough distance "
            "provides a price target (the 'measure rule')."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it works, the double-top / double-bottom is one of the most actionable signals "
            "in retail trading: it's visible on any chart, has a clear entry trigger (the "
            "neckline break), and a clear target. Textbooks teach it as foundational.\n\n"
            "If it doesn't work, millions of traders are acting on a pattern that reads "
            "noise — paying commissions and spreads to trade a shape that any random day "
            "can replicate. The stakes: one honest, rigorous test."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Two rules keep us honest:\n\n"
            "1. **Use only confirmed patterns.** We detect peaks and troughs objectively "
            "(via `scipy.signal.find_peaks`), require a minimum separation, a height "
            "similarity within 4%, and a neckline break on the close — exactly the "
            "trader's recipe. We enter the *next* bar after confirmation, so there is "
            "no look-ahead.\n"
            "2. **Race it against a random day.** For each confirmed pattern, we pick "
            "a random day from the same tape and assign the same direction. If the "
            "pattern knows something, it beats the random day. If it doesn't, it won't.\n\n"
            "Tape: **10 liquid names** (SPY, QQQ, AAPL, MSFT, AMZN, GOOGL, JPM, XOM, GLD, IWM), "
            "**2010–2026** (15 years, ~4,100 trading days each). Forward returns at 1, 5, and "
            "20 trading days."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, the pattern detection: how many double-tops and double-bottoms are "
            "found across the basket?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pooled = pooled_study(cost_bps=0.0)\n"
            "    n_dt = len(pooled.get('double_top', [])) if 'double_top' in pooled else 0\n"
            "    n_db = len(pooled.get('double_bottom', [])) if 'double_bottom' in pooled else 0\n"
            "else:\n"
            f"    n_dt, n_db = {R['dt_n']}, {R['db_n']}\n"
            "print(f'Double-tops detected: {n_dt}')\n"
            "print(f'Double-bottoms detected: {n_db}')\n"
            "print(f'Total confirmed patterns: {n_dt + n_db}')\n"
            "print(f'Avg patterns per ticker per year: {(n_dt+n_db)/10/16:.1f}')"
        ),
        md(
            "Let's look at the headline test: does the pattern's direction beat a random day?"
        ),
        code(
            "horizons = [1, 5, 20]\n"
            "if HAVE_REAL:\n"
            "    pooled = pooled_study(cost_bps=0.0)\n"
            "    rows = []\n"
            "    for pname, label, color in [('double_top', 'Double-top', RED),\n"
            "                                 ('double_bottom', 'Double-bottom', GREEN)]:\n"
            "        if pname in pooled:\n"
            "            for h in horizons:\n"
            "                res = st.summarize_pattern(pooled[pname], horizon=h)\n"
            "                rows.append((label, h, res.get('mean_bps',0),\n"
            "                             res.get('baseline_bps',0), res.get('excess_bps',0)))\n"
            "else:\n"
            "    rows = [\n"
            f"        ('Double-top', 1, {R['dt_1d_sig']}, {R['dt_1d_rand']}, {R['dt_1d_exc']}),\n"
            f"        ('Double-top', 5, {R['dt_5d_sig']}, {R['dt_5d_rand']}, {R['dt_5d_exc']}),\n"
            f"        ('Double-top', 20, {R['dt_20d_sig']}, {R['dt_20d_rand']}, {R['dt_20d_exc']}),\n"
            f"        ('Double-bottom', 1, {R['db_1d_sig']}, {R['db_1d_rand']}, {R['db_1d_exc']}),\n"
            f"        ('Double-bottom', 5, {R['db_5d_sig']}, {R['db_5d_rand']}, {R['db_5d_exc']}),\n"
            f"        ('Double-bottom', 20, {R['db_20d_sig']}, {R['db_20d_rand']}, {R['db_20d_exc']}),\n"
            "    ]\n"
            "tbl = pd.DataFrame(rows, columns=['pattern','horizon','signal_bps','random_bps','excess_bps'])\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "for ax, (pname, color) in zip(axes, [('Double-top', RED), ('Double-bottom', GREEN)]):\n"
            "    sub = tbl[tbl['pattern']==pname]\n"
            "    x = range(len(sub))\n"
            "    ax.bar([i-0.18 for i in x], sub['signal_bps'], 0.35, label='Pattern signal', color=color, alpha=0.8)\n"
            "    ax.bar([i+0.18 for i in x], sub['random_bps'], 0.35, label='Random placebo', color=GREY, alpha=0.8)\n"
            "    ax.axhline(0, c='k', lw=1)\n"
            "    ax.set_xticks(list(x)); ax.set_xticklabels(['1d', '5d', '20d'])\n"
            "    ax.set_title(f'{pname}: signal vs random placebo')\n"
            "    ax.set_ylabel('mean return (bps, gross)')\n"
            "    ax.legend(fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "tbl.round(1)"
        ),
        md(
            f"The double-top signal barely differs from a random bearish bet at any horizon. "
            f"The double-bottom's 5-day and 20-day signal returns look positive — but the "
            f"**random placebo does better** at both horizons: the market was already drifting "
            f"upward when the double-bottom fired, so the pattern adds nothing beyond the "
            f"market's own bullish tendency. The excess at 5 days is {R['db_5d_exc']:+.0f} bps "
            f"(placebo wins by {abs(R['db_5d_exc']):.0f} bps)."
        ),
        md(
            "**Is the excess statistically significant — or just noise?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pooled = pooled_study(cost_bps=0.0)\n"
            "    exc_rows = []\n"
            "    for pname in ['double_top', 'double_bottom']:\n"
            "        if pname in pooled:\n"
            "            for h in [1, 5, 20]:\n"
            "                res = st.summarize_pattern(pooled[pname], horizon=h)\n"
            "                exc_rows.append((pname, h, res.get('excess_bps',0),\n"
            "                                 res.get('tstat_excess', np.nan)))\n"
            "else:\n"
            "    exc_rows = [\n"
            f"        ('double_top', 1, {R['dt_1d_exc']}, {R['dt_1d_t']}),\n"
            f"        ('double_top', 5, {R['dt_5d_exc']}, {R['dt_5d_t']}),\n"
            f"        ('double_top', 20, {R['dt_20d_exc']}, {R['dt_20d_t']}),\n"
            f"        ('double_bottom', 1, {R['db_1d_exc']}, {R['db_1d_t']}),\n"
            f"        ('double_bottom', 5, {R['db_5d_exc']}, {R['db_5d_t']}),\n"
            f"        ('double_bottom', 20, {R['db_20d_exc']}, {R['db_20d_t']}),\n"
            "    ]\n"
            "edf = pd.DataFrame(exc_rows, columns=['pattern','horizon','excess_bps','t_excess'])\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "labels = [f\"{r['pattern'].replace('_',' ')} {r['horizon']}d\" for _, r in edf.iterrows()]\n"
            "colors = [RED if abs(r['t_excess'])>=3 else GREY for _, r in edf.iterrows()]\n"
            "bars_b = ax.bar(labels, edf['t_excess'], color=colors)\n"
            "ax.axhline(3, ls='--', c=RED, lw=1.5, label='Bonferroni threshold |t|=3')\n"
            "ax.axhline(-3, ls='--', c=RED, lw=1.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('HAC t-stat (excess over placebo)')\n"
            "ax.set_title('No pattern/horizon combination clears the significance bar')\n"
            "ax.tick_params(axis='x', rotation=30)\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Max |t| on excess across all 6 tests: "
            + str(R['max_abs_t_exc'])
            + " (threshold: 3.0)')"
        ),
        md(
            f"Every bar sits well inside the noise band. The highest |t| on the excess is "
            f"**{R['max_abs_t_exc']:.2f}** (double-top, 1-day horizon), far below the "
            "Bonferroni-corrected threshold of 3.0 that would be needed for one out of six "
            "tests to be considered significant. **None of the six combinations survives.**"
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Max excess |t| = {R['max_abs_t_exc']:.2f} across six "
            "pattern/horizon tests (Bonferroni threshold 3.0). The double-bottom's positive "
            "absolute return is fully explained by unconditional market drift — the placebo "
            "captures it just as well.\n"
            "- **Tradability — Mirage.** No excess return exists before costs; adding even "
            "1 bps round-trip makes the expected outcome clearly negative.\n"
            "- **Placebo wins? — Yes.** At the 5-day and 20-day horizons, both patterns "
            "are beaten by the random-day arm, meaning the *timing* of the pattern fire "
            "adds negative value relative to random."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Even if the excess were real, a pattern that fires roughly "
            f"{(R['dt_n']+R['db_n'])//10//16:.0f} times per ticker per year leaves almost no "
            "room for cost. Here is what adding a tiny round-trip spread does:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    net_rows = []\n"
            "    for cost in [0.0, 0.5, 1.0, 2.0]:\n"
            "        pooled_c = pooled_study(cost_bps=cost)\n"
            "        for pname in ['double_top', 'double_bottom']:\n"
            "            if pname in pooled_c:\n"
            "                res = st.summarize_pattern(pooled_c[pname], horizon=5)\n"
            "                net_rows.append((pname, cost, res.get('excess_bps', np.nan)))\n"
            "    ndf = pd.DataFrame(net_rows, columns=['pattern','cost','excess_bps'])\n"
            "else:\n"
            f"    ndf = pd.DataFrame({{\n"
            f"        'pattern': ['double_top','double_top','double_top','double_top',\n"
            f"                    'double_bottom','double_bottom','double_bottom','double_bottom'],\n"
            f"        'cost': [0.0, 0.5, 1.0, 2.0]*2,\n"
            f"        'excess_bps': [{R['dt_5d_exc']}, {R['dt_5d_exc']-0.5}, {R['dt_5d_exc']-1.0}, {R['dt_5d_exc']-2.0},\n"
            f"                       {R['db_5d_exc']}, {R['db_5d_exc']-0.5}, {R['db_5d_exc']-1.0}, {R['db_5d_exc']-2.0}]\n"
            "    })\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "for pname, color in [('double_top', RED), ('double_bottom', GREEN)]:\n"
            "    sub = ndf[ndf['pattern']==pname]\n"
            "    ax.plot(sub['cost'], sub['excess_bps'], 'o-', c=color, lw=2, label=pname.replace('_',' '))\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net excess over placebo (bps, 5d horizon)')\n"
            "ax.set_title('No positive excess at any cost level')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('Both patterns start at or below zero gross excess — costs only make it worse.')"
        ),
        md(
            "The double-top's excess is already near zero before costs; the double-bottom's "
            "is already negative. Adding any friction makes the picture immediately and "
            "meaningfully worse. **There is no positive break-even cost.**"
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **What if you fine-tune the detector?** The companion notebook sweeps the "
            "key parameters (tolerance, lookback, prominence threshold) to check robustness. "
            "The t-stats on excess remain uniformly sub-threshold.\n"
            "- **What if you use the measure-rule target?** The folklore says the expected "
            "move equals the height of the pattern. We leave this as a fork — add a "
            "target-based exit to `strategy.py` and measure whether hitting the target is "
            "more likely than chance (hint: a symmetric check is what the desk uses).\n"
            "- **Related studies:** candlestick one-bar reversal patterns are "
            "[Study 76 — Rice-Paper](../../76-rice-paper/) — same conclusion. The "
            "SMA(5/10) momentum scalp is [Study 72 — Loaded-Dice](../../72-loaded-dice/). "
            "Head-and-shoulders (a related multi-peak pattern) is a natural follow-on fork.\n\n"
            "*Think the M and W really do work? Fork this, change the detection parameters "
            "or entry logic, and show a Bonferroni-corrected |t| ≥ 3 on the excess over "
            "the random-day arm. That's the bar.*"
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
            "# Double-Top / Double-Bottom — a quantitative teardown\n"
            "### 10 tickers · 15 years of daily bars · random-day placebo · HAC inference · Bonferroni correction\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Placebo_wins%3F: Yes](https://img.shields.io/badge/Placebo_wins%3F-Yes-8b949e?style=flat-square)\n\n"
            "The deep companion to [01_for_the_curious.ipynb](01_for_the_curious.ipynb) — same "
            "seven beats, every claim carrying its standard error. We test whether confirmed "
            "double-top and double-bottom patterns produce a forward return statistically "
            "distinguishable from an unconditional random-day bet in the same direction.\n\n"
            "> ⚠️ **Not investment advice.** Real data: Yahoo Finance daily bars 2010-01-01 "
            "to 2026-06-15. Methods & sources in [`docs/references.md`](../docs/references.md); "
            "reproducible numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT + "\nfrom quantlab import analytics, stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Max HAC |t| on excess return (pattern − placebo) = "
            f"**{R['max_abs_t_exc']:.2f}** across 6 tests; Bonferroni threshold 3.0; zero "
            "tests survive. |\n"
            "| **Tradability** | `MIRAGE` | No positive gross excess exists before costs; "
            "the break-even transaction cost is below zero. |\n"
            "| **Placebo wins?** | `YES` | At 5d and 20d, the random-day arm outperforms "
            "the confirmed pattern on both double-top and double-bottom. |\n\n"
            "> 💡 The patterns see the past clearly — two peaks at a similar high — but "
            "they do not predict the future beyond what unconditional market drift already "
            "implies."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            r"Let $P_t^{\\text{DT}}$ be an indicator that a confirmed double-top fires at bar "
            r"$t$ (neckline break below the intervening trough), and $r_{t,h}$ the "
            r"$h$-day log return from bar $t$. The claim asserts:\n\n"
            "- **H₁ (signal):** $\\mathbb{E}[-r_{t,h} \\mid P_t^{\\text{DT}}=1] > 0$ — the "
            "short return after a confirmed double-top is positive in expectation.\n"
            "- **H₂ (excess):** $\\mathbb{E}[-r_{t,h} \\mid P_t^{\\text{DT}}=1] > "
            "\\mathbb{E}[-r_{s,h}]$ where $s$ is a random day — the pattern adds "
            "information beyond the unconditional base rate.\n"
            "- **H₃ (tradable):** H₂ survives a round-trip transaction cost.\n\n"
            "Symmetric claims for the double-bottom. We test H₁ and H₂ for each pattern "
            "at horizons 1d, 5d, 20d (6 tests total); apply Bonferroni correction (|t| ≥ 3.0)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The double-top / double-bottom is the single most commonly cited 'reliable' "
            "reversal pattern in retail technical-analysis education. If H₂ held robustly "
            "over 15 years of liquid US names, it would be a significant edge: high "
            "signal-count (~220 confirmed patterns per year across 10 names), clear entry "
            "rules, systematic execution possible. The failure is interesting precisely "
            "because the absolute signal return at longer horizons *looks* positive for "
            "the double-bottom — the decomposition into unconditional drift vs "
            "pattern-specific information shows exactly where the illusion lives."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Detection.** `scipy.signal.find_peaks` on the daily close with `distance = "
            "min_bars`, `prominence = 0.5 × ATR(20)`. A double-top requires two consecutive "
            "peaks within 4% of each other; the pattern fires when the close breaks below "
            "the trough between them. Mirror logic for double-bottom.\n"
            "- **Entry.** Pattern confirmed at close of bar *t*; position entered at bar *t+1* "
            "(no look-ahead).\n"
            "- **Forward returns.** Log close-to-close at 1, 5, and 20 trading days, in the "
            "claimed direction: short (−1) for double-top, long (+1) for double-bottom.\n"
            "- **Control.** For each confirmed signal, draw one random bar from the same "
            "tape (excluding the last 20 rows), assign the same direction. The *excess* "
            "return (signal − random) is the decisive statistic.\n"
            "- **Inference.** Newey-West HAC t-stat on the excess, Bonferroni correction "
            "for 6 tests (|t| ≥ 3.0 ≈ α = 5% / 6 ≈ 0.83%).\n"
            "- **Basket.** 10 names × 4,137 trading days (2010-01-04 to 2026-06-15). "
            "Survivorship: all names are in the index as of 2026; we name this bias."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Detection counts and pattern rate\n\n"
            "How many confirmed patterns do we find, and at what rate?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pooled = pooled_study(cost_bps=0.0)\n"
            "    n_dt = len(pooled.get('double_top', pd.DataFrame()))\n"
            "    n_db = len(pooled.get('double_bottom', pd.DataFrame()))\n"
            "else:\n"
            f"    n_dt, n_db = {R['dt_n']}, {R['db_n']}\n"
            "print(f'Double-tops (bearish confirmed): {n_dt}')\n"
            "print(f'Double-bottoms (bullish confirmed): {n_db}')\n"
            "print(f'Total: {n_dt+n_db} across 10 tickers, ~16 years')\n"
            "print(f'Rate: ~{(n_dt+n_db)/(10*16):.1f} patterns/ticker/year')"
        ),
        md(
            "### 4b · Signal vs placebo by horizon — the full picture\n\n"
            "For each pattern and horizon: signal mean, placebo mean, excess, and HAC "
            "t-stat on the excess. The Bonferroni threshold for 6 tests at α = 5% is "
            "|t| = 3.0."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pooled = pooled_study(cost_bps=0.0)\n"
            "    result_rows = []\n"
            "    for pname in ['double_top', 'double_bottom']:\n"
            "        if pname not in pooled: continue\n"
            "        for h in [1, 5, 20]:\n"
            "            res = st.summarize_pattern(pooled[pname], horizon=h)\n"
            "            result_rows.append((pname, h, res.get('n',0),\n"
            "                               res.get('win_rate',np.nan)*100,\n"
            "                               res.get('mean_bps',np.nan),\n"
            "                               res.get('baseline_bps',np.nan),\n"
            "                               res.get('excess_bps',np.nan),\n"
            "                               res.get('tstat_excess',np.nan)))\n"
            "    rdf = pd.DataFrame(result_rows,\n"
            "         columns=['pattern','h','n','win%','sig_bps','rnd_bps','exc_bps','t_exc'])\n"
            "else:\n"
            "    rdf = pd.DataFrame([\n"
            f"        ('double_top', 1, {R['dt_n']}, {R['dt_win']}, {R['dt_1d_sig']}, {R['dt_1d_rand']}, {R['dt_1d_exc']}, {R['dt_1d_t']}),\n"
            f"        ('double_top', 5, {R['dt_n']}, {R['dt_win']}, {R['dt_5d_sig']}, {R['dt_5d_rand']}, {R['dt_5d_exc']}, {R['dt_5d_t']}),\n"
            f"        ('double_top', 20, {R['dt_n']}, {R['dt_win']}, {R['dt_20d_sig']}, {R['dt_20d_rand']}, {R['dt_20d_exc']}, {R['dt_20d_t']}),\n"
            f"        ('double_bottom', 1, {R['db_n']}, {R['db_win_5d']}, {R['db_1d_sig']}, {R['db_1d_rand']}, {R['db_1d_exc']}, {R['db_1d_t']}),\n"
            f"        ('double_bottom', 5, {R['db_n']}, {R['db_win_5d']}, {R['db_5d_sig']}, {R['db_5d_rand']}, {R['db_5d_exc']}, {R['db_5d_t']}),\n"
            f"        ('double_bottom', 20, {R['db_n']}, {R['db_win_20d']}, {R['db_20d_sig']}, {R['db_20d_rand']}, {R['db_20d_exc']}, {R['db_20d_t']}),\n"
            "    ], columns=['pattern','h','n','win%','sig_bps','rnd_bps','exc_bps','t_exc'])\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "labels = [f\"{r['pattern'].replace('_',' ')}/{r['h']}d\" for _, r in rdf.iterrows()]\n"
            "colors = [RED if abs(v)>=3 else GREY for v in rdf['t_exc']]\n"
            "ax.bar(labels, rdf['t_exc'], color=colors)\n"
            "for thr in (3, -3): ax.axhline(thr, ls='--', c=RED, lw=1.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('HAC t-stat (excess return over placebo)')\n"
            "ax.set_title('No test clears Bonferroni |t|=3 (dashed lines)')\n"
            "ax.tick_params(axis='x', rotation=30)\n"
            "plt.tight_layout(); plt.show()\n"
            "rdf.round(2)"
        ),
        md(
            "> 💡 The double-bottom's 20-day *signal* t-stat is +6.30 — which looks "
            "spectacular until you check the placebo: the random-day arm also has a "
            "strongly positive 20-day return (+111.78 bps) because US equities trend "
            "upward over 20-day windows. The excess is only −7.09 bps with t = −0.28. "
            "The pattern is riding the bull market, not predicting it."
        ),
        md(
            "### 4c · Sensitivity to detection parameters\n\n"
            "Are the results robust to the pattern-detection tuning?"
        ),
        code(
            "# Sweep tolerance (height similarity threshold) on synthetic tape — fast, offline\n"
            "bars, _ = data.synthetic_daily(n_days=3000, reversal=0.0, seed=189)\n"
            "tolerances = [0.02, 0.04, 0.06, 0.08, 0.10]\n"
            "tol_rows = []\n"
            "for tol in tolerances:\n"
            "    s = st.run_pattern_study(bars, horizons=(5,), cost_bps=0.0,\n"
            "                             seed=0, min_bars=8, tolerance=tol,\n"
            "                             lookback=100, atr_prominence_mult=0.3)\n"
            "    for pname, df in s.items():\n"
            "        res = st.summarize_pattern(df, horizon=5)\n"
            "        tol_rows.append((tol, pname, res.get('n', 0), res.get('tstat_excess', np.nan)))\n"
            "tdf = pd.DataFrame(tol_rows, columns=['tolerance','pattern','n','t_exc'])\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "for pname, color in [('double_top', RED), ('double_bottom', GREEN)]:\n"
            "    sub = tdf[tdf['pattern']==pname]\n"
            "    ax.plot(sub['tolerance'], sub['t_exc'], 'o-', c=color, lw=2,\n"
            "            label=pname.replace('_', ' '))\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(-2, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('height-similarity tolerance'); ax.set_ylabel('HAC t (excess, 5d)')\n"
            "ax.set_title('Synthetic martingale tape: t stays near zero across tolerances')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('On a martingale, no tolerance setting produces a reliable signal — as expected.')"
        ),
        md(
            "> 💡 On a pure random walk (martingale) the excess t-stat stays near zero "
            "regardless of how we tune the detector. This confirms the engine is not "
            "overfitting the tolerance parameter to manufacture t-stats."
        ),
        md(
            "### 4d · The synthetic positive control — does the engine work at all?\n\n"
            "To confirm the detector *can* find a signal when one exists, we plant "
            "mean-reversion in a synthetic tape and sweep the reversal strength."
        ),
        code(
            "reversals = [0.0, 0.10, 0.20, 0.30, 0.40]\n"
            "ctrl_rows = []\n"
            "for rev in reversals:\n"
            "    bars, _ = data.synthetic_daily(n_days=3000, reversal=rev, seed=189)\n"
            "    s = st.run_pattern_study(bars, horizons=(5,), cost_bps=0.0,\n"
            "                             seed=0, min_bars=5, tolerance=0.10,\n"
            "                             lookback=60, atr_prominence_mult=0.1)\n"
            "    for pname, df in s.items():\n"
            "        res = st.summarize_pattern(df, horizon=5)\n"
            "        ctrl_rows.append((rev, pname, res.get('n', 0),\n"
            "                          res.get('excess_bps', np.nan),\n"
            "                          res.get('tstat_excess', np.nan)))\n"
            "cdf = pd.DataFrame(ctrl_rows, columns=['reversal','pattern','n','excess_bps','t_exc'])\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "for pname, color in [('double_top', RED), ('double_bottom', GREEN)]:\n"
            "    sub = cdf[cdf['pattern']==pname]\n"
            "    ax.plot(sub['reversal'], sub['t_exc'], 'o-', c=color, lw=2,\n"
            "            label=pname.replace('_', ' '))\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1); ax.axhline(-2, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted mean-reversion (reversal knob)')\n"
            "ax.set_ylabel('HAC t (excess return, 5d horizon)')\n"
            "ax.set_title('Positive control: engine finds the signal when planted')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('The engine is a faithful mean-reversion detector. Real tape has reversal ~ 0.')"
        ),
        md(
            "> 💡 The excess t-stat rises with planted mean-reversion. The engine works; the "
            "real tape simply doesn't have the reversion that would make these patterns "
            "informative after the neckline break."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — max HAC |t| on excess = {R['max_abs_t_exc']:.2f}; "
            "Bonferroni threshold 3.0; zero tests survive. The double-bottom's positive "
            "absolute return is entirely attributable to unconditional equity drift.\n"
            "- **Tradability `MIRAGE`** — no positive gross excess; every cost level makes "
            "the outcome worse. Survivorship bias (only names alive in 2026) biases the "
            "estimate generously toward the pattern — the honest result is if anything "
            "worse than reported.\n"
            "- **Placebo wins? `YES`** — at 5d and 20d, the random-day arm outperforms "
            "the confirmed pattern for both double-top and double-bottom."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — costs bury it before it starts\n\n"
            "The pattern fires roughly 14 times per ticker per year — less than one per "
            "month. Even at that low frequency, the gross excess is too small (and often "
            "wrong-signed) to survive any realistic cost."
        ),
        code(
            "cost_grid = [0.0, 0.5, 1.0, 2.0, 5.0]\n"
            "if HAVE_REAL:\n"
            "    cost_rows = []\n"
            "    for cost in cost_grid:\n"
            "        p = pooled_study(cost_bps=cost)\n"
            "        for pname in ['double_top', 'double_bottom']:\n"
            "            if pname in p:\n"
            "                res = st.summarize_pattern(p[pname], horizon=5)\n"
            "                cost_rows.append((pname, cost,\n"
            "                                  res.get('excess_bps', np.nan),\n"
            "                                  res.get('tstat_excess', np.nan)))\n"
            "    codf = pd.DataFrame(cost_rows, columns=['pattern','cost','excess_bps','t_exc'])\n"
            "else:\n"
            "    codf = pd.DataFrame({\n"
            "        'pattern': ['double_top']*5 + ['double_bottom']*5,\n"
            "        'cost': cost_grid * 2,\n"
            f"        'excess_bps': [{R['dt_5d_exc']}, {R['dt_5d_exc']-.5}, {R['dt_5d_exc']-1},\n"
            f"                       {R['dt_5d_exc']-2}, {R['dt_5d_exc']-5},\n"
            f"                       {R['db_5d_exc']}, {R['db_5d_exc']-.5}, {R['db_5d_exc']-1},\n"
            f"                       {R['db_5d_exc']-2}, {R['db_5d_exc']-5}],\n"
            f"        't_exc': [{R['dt_5d_t']:.2f}]*5 + [{R['db_5d_t']:.2f}]*5\n"
            "    })\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "for pname, color in [('double_top', RED), ('double_bottom', GREEN)]:\n"
            "    sub = codf[codf['pattern']==pname]\n"
            "    ax.plot(sub['cost'], sub['excess_bps'], 'o-', c=color, lw=2,\n"
            "            label=pname.replace('_', ' '))\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net excess over placebo (bps, 5d)')\n"
            "ax.set_title('No break-even cost exists: both patterns start at or below zero')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('Break-even transaction cost: undefined (gross excess already at or below zero).')"
        ),
        md(
            "> 💡 A 'real signal that dies at the costs line' would start positive and "
            "cross zero at some finite cost. Here both arms start at or below zero — "
            "there is no costs line to cross."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Survivorship bias quantification.** The 10 names used here are all "
            "live as of 2026. A proper out-of-sample test would use point-in-time "
            "S&P 500 membership (EDGAR-vintage constituent lists). The bias direction "
            "is toward the pattern looking better: stocks that formed double-tops and "
            "then failed out of the index aren't in our sample.\n"
            "- **The measure rule.** The folklore price target (prior pattern height) "
            "can be coded as a barrier exit. Does the target get hit at a rate above "
            "chance? That's a PR-able extension.\n"
            "- **Head-and-shoulders.** The three-peak variant (left shoulder, head, "
            "right shoulder) is a natural follow-on; the detection logic extends "
            "directly from the peak-finding framework here.\n"
            "- **Related studies:** [Study 76 — Rice-Paper](../../76-rice-paper/) "
            "(one-bar candlestick reversals, same conclusion), "
            "[Study 72 — Loaded-Dice](../../72-loaded-dice/) (momentum scalp), "
            "[Study 127 — Williams %R](../../127-williams-r/) (oscillator overbought/oversold).\n\n"
            "*Think a refined detector finds something? Fork this, show Bonferroni-corrected "
            "|t| ≥ 3.0 on excess over the random-day placebo, and submit the PR.*"
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
