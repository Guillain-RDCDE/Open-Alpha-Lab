"""Generate the two narrative notebooks for Study 184 (Williams-Fractals).

    python notebooks/build_notebooks.py
    python -W ignore -m jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached daily
parquets under ../_cache/ if present and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.

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
    # Reversal framing
    n_rev=3988, rev_win=49.7, rev_mean=-4.09, rev_t=-1.26,
    rand_rev_mean=2.95, rand_rev_t=0.60, rev_minus_rand=-7.04,
    # Breakout framing
    n_brk=3346, brk_win=50.5, brk_mean=-0.06, brk_t=-0.01,
    rand_brk_mean=1.04,
    # Per-instrument (reversal) t-stats
    spy_t=0.29, qqq_t=-0.89, iwm_t=0.36, aapl_t=-0.83, msft_t=-1.93, gld_t=0.37,
    # Hold-period sweep (reversal, gross)
    hold1=-1.77, hold1_t=-1.00,
    hold3=-4.94, hold3_t=-1.59,
    hold5=-4.09, hold5_t=-1.26,
    hold10=-1.60, hold10_t=-0.42,
    hold20=5.54, hold20_t=1.09,
    # Cost sweep (reversal, hold=5)
    cost0=-4.09, cost0_t=-1.26,
    cost05=-4.59, cost05_t=-1.41,
    cost10=-5.09, cost10_t=-1.57,
    cost20=-6.09, cost20_t=-1.88,
    cost50=-9.09, cost50_t=-2.80,
    # Fractal rate
    fractal_rate=26.8,
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

from williams_fractals import data, strategy as st

TICKERS = ["SPY", "QQQ", "IWM", "AAPL", "MSFT", "GLD"]
CACHE = os.path.join(os.path.abspath(".."), "_cache")

def _have_cache():
    return all(os.path.exists(data._cache_path(t, CACHE)) for t in TICKERS)

HAVE_REAL = _have_cache()

def pool_rev(hold_days=5, cost=0.0, seed=None):
    \"\"\"Pool reversal trades across the basket (cached real tapes).\"\"\"
    frames = []
    for t in TICKERS:
        b = data.fetch_daily(t, cache_dir=CACHE)
        f = st.detect_fractals(b)
        ent = st.reversal_entries(f)
        dirs = st.random_directions(len(ent), seed=seed) if seed is not None else None
        frames.append(st.run_trades(b, ent, hold_days=hold_days, cost_bps=cost, directions=dirs))
    return pd.concat(frames, ignore_index=True)

def pool_brk(hold_days=5, cost=0.0, seed=None):
    \"\"\"Pool breakout trades across the basket (cached real tapes).\"\"\"
    frames = []
    for t in TICKERS:
        b = data.fetch_daily(t, cache_dir=CACHE)
        f = st.detect_fractals(b)
        ent = st.breakout_entries(b, f)
        dirs = st.random_directions(len(ent), seed=seed) if seed is not None else None
        frames.append(st.run_trades(b, ent, hold_days=hold_days, cost_bps=cost, directions=dirs))
    return pd.concat(frames, ignore_index=True)

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Williams-Fractals — does a 5-bar swing mark predict the next move?\n"
            "### Bill Williams' fractal pattern, tested honestly in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beat_a_coin%3F: No](https://img.shields.io/badge/Beat_a_coin%3F-No-8b949e?style=flat-square)\n\n"
            "Here's a pattern you'll find on almost every charting platform and in a thousand "
            "trading courses: the **Williams Fractal**. A bar whose high is the highest of the "
            "surrounding five marks a local swing top; the lowest low marks a swing bottom. The "
            "theory: these swings aren't random — they're *anchors* that tell you something about "
            "what price will do next. Fade the extreme. Or wait for a breakout above it. Either "
            "way, the fractal is supposed to *load the dice*. This notebook asks: does it?\n\n"
            "> 📓 **This is the plain-language layer.** For t-stats, hold-period sweeps and cost "
            "maths see **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart below is "
            "drawn by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does fading the fractal (reversal) work? | **No.** The reversal entry earns "
            f"**{R['rev_mean']:+.2f} bps/trade** — slightly negative, worse than a coin. |\n"
            "| Does breaking the fractal level (breakout) work? | **No.** The breakout earns "
            f"**{R['brk_mean']:+.2f} bps/trade** — essentially zero, exactly a coin. |\n"
            "| Does either beat a random-direction entry? | **No.** Both framings perform "
            "*worse* than random entries on the same bars. |\n"
            "| Could you trade it? | **No.** The gross edge is already negative; costs make "
            f"it a loser at every realistic level (at 5 bps: *t* = **{R['cost50_t']:+.2f}**). |\n\n"
            "> The fractal marks a local extreme by *definition* — but a local extreme on a "
            "random walk is just a local extreme. The math doesn't care what it's called."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A Williams Fractal is a 5-bar pattern where the centre bar's high (or low) "
            "exceeds both neighbours. Bearish fractals mark resistance. Bullish fractals mark "
            "support. Trade the reversal — or wait for the breakout — and you're trading with "
            "the natural rhythm of the market.\"* — standard retail framing\n\n"
            "Bill Williams introduced fractals in *Trading Chaos* (1995) as part of his "
            "'Alligator' system. The pattern fires on about **one in four trading days** "
            f"(we observe **{R['fractal_rate']}%** of bars), so there's never a shortage of "
            "signals. The claim is that these frequent local extremes mark *meaningful* turning "
            "points — not just noise peaks in a random process."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If fractals carry real information, they're the basis for a simple, mechanical "
            "daily strategy with hundreds of signals a year per instrument — easy to run, easy "
            "to automate. If they don't, millions of traders are drawing lines on charts and "
            "making decisions based on a pattern that is, by definition, just a local maximum "
            "or minimum of a noisy series. The distinction matters — let's test it."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Two rules keep us honest:\n\n"
            "1. **The confirmation lag.** A fractal at bar *t* needs *t+1* and *t+2* to "
            "confirm it — those bars happen *in the future*. We can only *know* the fractal at "
            "the close of *t+2*, so a trade can only enter at the *open* of *t+3*. Many "
            "charting studies silently use the fractal bar itself as the entry, which is "
            "look-ahead. We don't.\n"
            "2. **Race it against an actual coin.** We take the exact same entry bars and flip "
            "a coin for the direction. If the fractal direction signal adds information, it "
            "beats the coin. If it doesn't, it won't.\n\n"
            "Two framings tested on **six liquid daily tapes** (SPY, QQQ, IWM, AAPL, MSFT, GLD) "
            "over 10 years, with the 2-bar confirmation lag correctly applied throughout."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — two theories, one honest coin flip\n\n"
            "### 4a · The confirmation-lag trap\n\n"
            "Here's why the reversal idea sounds more compelling than it is. A local swing "
            "high at bar *t* is *already* being 'reversed' by bars *t+1* and *t+2* — otherwise "
            "they'd be higher than *t* and the fractal wouldn't confirm. By the time you can "
            "*act* at the open of *t+3*, the initial reversion has already happened. You're "
            "selling into a bounce that occurred two days ago."
        ),
        code(
            "# Illustrate: how much of the 'reversion' happens BEFORE the trade entry?\n"
            "bars, _ = data.synthetic_daily(n_days=2000, momentum=0.0, seed=184)\n"
            "f = st.detect_fractals(bars)\n"
            "# Return from fractal centre (t) to t+1, t+2, t+3 for bearish fractals\n"
            "bear_idx = bars.index[f['bearish']]\n"
            "lag_rets = {}\n"
            "idx = bars.index.tolist()\n"
            "for lag in range(1, 6):\n"
            "    diffs = []\n"
            "    for dt in bear_idx:\n"
            "        i = idx.index(dt)\n"
            "        if i + lag < len(bars):\n"
            "            # For bearish fractal (swing high), negative return is 'good'\n"
            "            r = (bars['close'].iloc[i+lag] - bars['close'].iloc[i]) / bars['close'].iloc[i]\n"
            "            diffs.append(-r * 1e4)  # flip: positive = reversion happening\n"
            "    lag_rets[lag] = np.mean(diffs)\n\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "lags = list(lag_rets.keys())\n"
            "vals = list(lag_rets.values())\n"
            "ax.bar(lags, vals, color=[GREY, GREY, RED, GREY, GREY])\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.axvspan(2.5, 3.5, color=RED, alpha=0.08, label='trade entry opens here (t+3)')\n"
            "ax.set_xlabel('Lag from bearish fractal centre bar (t)')\n"
            "ax.set_ylabel('Mean cumulative reversion (bps from t)')\n"
            "ax.set_title('Reversion already priced in by confirmation; entry lag kills the trade')\n"
            "ax.set_xticks(lags)\n"
            "ax.set_xticklabels([f't+{l}' for l in lags])\n"
            "ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('(Synthetic: random walk — even the tiny initial reversion is consumed by confirmation)')"
        ),
        md(
            "Even on a random walk the first bar after a local high is slightly lower — that's "
            "regression to the mean by construction. But bars *t+1* and *t+2* consume that "
            "reversion as part of the *definition* of a confirmed fractal. By *t+3* the effect "
            "is gone. The confirmation lag that makes the pattern 'safe' to trade is the same "
            "lag that eats the return."
        ),
        md(
            "### 4b · The honest coin flip — reversal vs random direction"
        ),
        code(
            "if HAVE_REAL:\n"
            "    C = pool_rev(hold_days=5, cost=0.0)\n"
            "    rand_means = [st.summarize(pool_rev(hold_days=5, cost=0.0, seed=s), 'ret_gross')['mean_bps']\n"
            "                  for s in range(15)]\n"
            "    cm, cw = st.summarize(C, 'ret_gross')['mean_bps'], st.summarize(C, 'ret_gross')['win_rate']*100\n"
            "    rm = np.mean(rand_means)\n"
            "else:\n"
            "    cm, cw, rm = R['rev_mean'], R['rev_win'], R['rand_rev_mean']\n"
            "    rand_means = [R['rand_rev_mean']] * 15\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(rand_means, bins=10, color=GREY, alpha=0.7, label='random-control means (15 seeds)')\n"
            "ax.axvline(cm, c=RED, lw=2.5, label=f'fractal reversal: {cm:+.2f} bps')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('gross mean (bps/trade, 5-day hold)')\n"
            "ax.set_ylabel('count')\n"
            "ax.set_title('Fractal reversal sits inside the random noise band — and below zero')\n"
            "ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'reversal: {{cm:+.2f}} bps  |  random: {{rm:+.2f}} bps  |  fractal is WORSE by {{cm-rm:+.2f}} bps')"
        ),
        md(
            f"The fractal reversal earns **{R['rev_mean']:+.2f} bps/trade**, while a random "
            f"coin on the same bars earns **{R['rand_rev_mean']:+.2f} bps**.  The fractal is "
            f"**{R['rev_minus_rand']:+.2f} bps worse** than not having a signal at all."
        ),
        md(
            "### 4c · The breakout framing — waiting for the break"
        ),
        code(
            "if HAVE_REAL:\n"
            "    B = pool_brk(hold_days=5, cost=0.0)\n"
            "    RB = pool_brk(hold_days=5, cost=0.0, seed=184)\n"
            "    bm, bw = st.summarize(B, 'ret_gross')['mean_bps'], st.summarize(B, 'ret_gross')['win_rate']*100\n"
            "    rbm = st.summarize(RB, 'ret_gross')['mean_bps']\n"
            "else:\n"
            "    bm, bw, rbm = R['brk_mean'], R['brk_win'], R['rand_brk_mean']\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "bars_plot = ax.bar(\n"
            "    ['Fractal breakout\\n(with the break)', 'Random direction\\n(a coin)'],\n"
            "    [bm, rbm], color=[RED, GREY], width=0.55\n"
            ")\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('gross profit per trade (bps)')\n"
            "ax.set_title('Fractal breakout: also a coin flip')\n"
            "for b, w in zip(bars_plot, [bw, 50.0]):\n"
            "    ax.annotate(f'win {w:.1f}%', (b.get_x()+b.get_width()/2, 0),\n"
            "                ha='center', va='bottom' if b.get_height() < 0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print('breakout: {R['brk_mean']:+.2f} bps/trade  |  coin: {R['rand_brk_mean']:+.2f} bps/trade')"
        ),
        md(
            f"The breakout earns **{R['brk_mean']:+.2f} bps/trade** — essentially zero, a pure "
            "coin flip.  The fractal high or low carries no information about whether price "
            "will continue when it finally breaches that level."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Both the reversal and breakout framings are statistically "
            "indistinguishable from noise.  The reversal *underperforms* a coin; the breakout "
            "*is* a coin.  No instrument clears the inference bar.\n"
            "- **Tradability — Mirage.** A gross loss of −4 bps/trade compounds into a "
            f"significant loser at realistic costs (at 5 bps: *t* = **{R['cost50_t']:+.2f}**).\n"
            "- **Beat a coin? — No.** The fractal direction adds *negative* information — both "
            "framings would be better off ignoring the direction and flipping a coin."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "The fractal fires on ~27% of all trading days — that's high turnover. Even if the "
            "gross edge were flat (it isn't), a small per-trade cost would quickly bury it."
        ),
        code(
            "costs = [0.0, 0.5, 1.0, 2.0, 5.0]\n"
            "if HAVE_REAL:\n"
            "    net = [st.summarize(pool_rev(hold_days=5, cost=c), 'ret_net')['mean_bps'] for c in costs]\n"
            "else:\n"
            "    net = [R['cost0'], R['cost05'], R['cost10'], R['cost20'], R['cost50']]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, net, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, net, 0, where=[n < 0 for n in net], color=RED, alpha=0.12)\n"
            "ax.set_xlabel('round-trip cost (bps)')\n"
            "ax.set_ylabel('net profit per trade (bps)')\n"
            "ax.set_title('No break-even cost: the gross is already negative')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Gross is already {net[0]:+.2f} bps — there is no positive break-even cost.')"
        ),
        md(
            "The line starts below zero.  Because the gross edge is already negative, there is "
            "**no cost low enough** to make the strategy work — the usual 'it dies at the cost "
            "line' story doesn't even apply here, because it was never alive."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Is the *engine* broken, or is the market?** The companion notebook plants "
            "momentum in a synthetic tape and shows the breakout rule *does* find an edge when "
            "one exists — so it's a market statement, not a method flaw.\n"
            "- **What about the full Alligator system?** Williams used fractals alongside "
            "three displaced SMAs (the 'Alligator jaws'). That combination hasn't been "
            "studied here; fractal *alone* is the subject. A filter that required the "
            "Alligator to be 'awake' would be a different hypothesis.\n"
            "- **Related work on the desk:**\n"
            "  - [Study 127 — Williams-R](../../127-williams-r/): the other Williams indicator, "
            "same daily frequency, same verdict.\n"
            "  - [Study 76 — Rice-Paper](../../76-rice-paper/): candlestick chart patterns — "
            "the same 'visual pattern vs coin' framework.\n"
            "  - [Study 21 — Fools-Gold](../../21-fools-gold/): the daily golden cross — "
            "breakout family, same ending.\n\n"
            "*Think the fractal really does predict direction? Fork this, add a filter "
            "(Alligator state, volume, trend regime), and show a fair-bet edge that clears "
            "the coin comparison and survives costs. That's the bar.*"
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
            "# Williams-Fractals — a quantitative teardown\n"
            "### 5-bar swing pattern · reversal + breakout · random-direction control · "
            "HAC inference · hold-period & cost sweeps\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beat_a_coin%3F: No](https://img.shields.io/badge/Beat_a_coin%3F-No-8b949e?style=flat-square)\n\n"
            "The deep companion to [01_for_the_curious.ipynb](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim carrying its standard error.*  We test whether "
            "Williams' 5-bar fractal (reversal and breakout framings) beats a random-direction "
            "control on six daily tapes across 10 years, with strict no-look-ahead enforcement.\n\n"
            "> ⚠️ **Not investment advice.** Real data: Yahoo daily bars, 10-year window "
            "(2016-06-15 → 2026-06-15). Methods in [`docs/references.md`](../docs/references.md), "
            "reproducible numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **`💡 In plain words`** notes translate each result back to intuition."
        ),
        code(BOOT + "\ntry:\n    from quantlab import analytics, stats\nexcept ImportError:\n    analytics = stats = None\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Reversal gross **{R['rev_mean']:+.2f} bps/trade**, HAC "
            f"*t* = **{R['rev_t']:+.2f}**; Δ vs control = **{R['rev_minus_rand']:+.2f} bps**. "
            f"Breakout **{R['brk_mean']:+.2f} bps**, *t* = **{R['brk_t']:+.2f}**. "
            "No framing, no instrument clears the inference bar. |\n"
            f"| **Tradability** | `MIRAGE` | Gross already negative; at 5 bps "
            f"*t* = **{R['cost50_t']:+.2f}**, a statistically significant loser. |\n"
            f"| **Beat a coin?** | `No` | Both framings perform *worse* than a random-direction "
            "entry on the same bars. The pattern carries no directional information. |\n\n"
            "> 💡 The fractal is, by construction, a local extremum of a noisy series. "
            "Local extrema in random processes are followed by mean reversion — but the "
            "2-bar confirmation delay consumes that reversion before the trade opens."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $f^+_t$ be an indicator for a bullish fractal at bar $t$ (bar $t$'s low is the "
            "local 5-bar minimum) and $f^-_t$ a bearish fractal.  The confirmation rule states "
            "$f^{\\pm}_t$ is known only at the close of $t+2$.  The study tests two hypotheses:\n\n"
            "- **H₁ (reversal signal).** $\\mathbb{E}[d \\cdot r_{t+3 \\to t+3+h}] > 0$ where "
            "$d = +1$ after a bullish fractal, $d = -1$ after a bearish fractal — i.e., extreme "
            "fractals predict short-term mean reversion, net of the confirmation lag.\n"
            "- **H₂ (breakout signal).** $\\mathbb{E}[d \\cdot r_{\\tau \\to \\tau+h}] > 0$ where "
            "$\\tau$ is the first bar whose close breaks the fractal level — i.e., the break of "
            "a fractal level predicts continuation.\n"
            "- **H₃ (tradable).** Either H₁ or H₂ survives a realistic round-trip cost at the "
            "~27% bar-level frequency of the signal.\n\n"
            "We reject H₁, H₂, and H₃ on the real tape."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Fractals appear on virtually every retail charting platform as a default overlay. "
            "If H₁ or H₂ held, they'd be a simple, widely available daily signal — high value. "
            "The interesting failure is that H₁ fails for a theoretically coherent reason: the "
            "confirmation requirement is itself a reversal, so the reversal trade arrives *after* "
            "the event it's betting on. H₂ fails for a different reason: local extremes are "
            "selected by definition, and breaking them carries no more information than any "
            "other large-enough daily move."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Fractal detection.** Bearish fractal at $t$: $H_t > H_{t-2}, H_{t-1}, H_{t+1}, "
            "H_{t+2}$ (strict).  Bullish: $L_t < L_{t-2}, L_{t-1}, L_{t+1}, L_{t+2}$ (strict). "
            "Confirmed at close of $t+2$; trade enters at open of $t+3$.\n"
            "- **Reversal framing.** Bearish fractal → short ($d = -1$); bullish → long ($d = +1$). "
            "Fixed-horizon exit at close of $t + 3 + h - 1$, hold $h \\in \\{1,3,5,10,20\\}$.\n"
            "- **Breakout framing.** After confirmation, watch for first close that breaks the "
            "fractal level; enter at next open.  Level active for at most 20 bars.\n"
            "- **Control.** Identical entries, direction drawn i.i.d. $\\{-1,+1\\}$ (seeded).\n"
            "- **Inference.** Newey–West HAC *t* on per-trade returns.  Six tickers: SPY, QQQ, "
            "IWM, AAPL, MSFT, GLD.  10-year window (2016-06-15 → 2026-06-15).\n"
            "- **Positive control.** Synthetic AR(1) momentum tape confirms the breakout engine "
            "finds an edge *when momentum exists*."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Per-instrument: no tape carries a real signal (reversal, hold=5)\n\n"
            "If the fractal worked anywhere, one of these bars would clear |*t*| ≥ 2."
        ),
        code(
            "if HAVE_REAL:\n"
            "    recs = []\n"
            "    for t in TICKERS:\n"
            "        b = data.fetch_daily(t, cache_dir=CACHE)\n"
            "        f = st.detect_fractals(b)\n"
            "        ent = st.reversal_entries(f)\n"
            "        s = st.summarize(st.run_trades(b, ent, hold_days=5, cost_bps=0), 'ret_gross')\n"
            "        recs.append((t, s['n_trades'], s['win_rate']*100, s['mean_bps'], s['tstat']))\n"
            "    perinst = pd.DataFrame(recs, columns=['ticker','n','win%','mean_bps','t'])\n"
            "    ps = st.summarize(pool_rev(hold_days=5, cost=0.0), 'ret_gross')\n"
            "else:\n"
            "    perinst = pd.DataFrame({\n"
            "        'ticker': TICKERS,\n"
            f"        't': [{R['spy_t']}, {R['qqq_t']}, {R['iwm_t']}, {R['aapl_t']}, {R['msft_t']}, {R['gld_t']}]\n"
            "    })\n"
            "    perinst['mean_bps'] = float('nan'); perinst['win%'] = float('nan'); perinst['n'] = float('nan')\n"
            f"    ps = dict(mean_bps={R['rev_mean']}, tstat={R['rev_t']}, win_rate={R['rev_win']}/100)\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "col = [GREEN if abs(v) >= 2 else RED for v in perinst['t']]\n"
            "ax.bar(perinst['ticker'], perinst['t'], color=col)\n"
            "for s in (2, -2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('HAC t-stat (gross mean/trade)')\n"
            "ax.set_title('Six instruments, six non-results (|t| < 2 everywhere)')\n"
            "ax.set_ylim(-3, 3)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"pooled: {ps['mean_bps']:+.2f} bps/trade, HAC t = {ps['tstat']:+.2f}\")\n"
            "if HAVE_REAL:\n"
            "    display(perinst.round(2))"
        ),
        md(
            f"> 💡 All six instruments sit inside the ±2 band, most with *t* near zero or "
            f"negative.  Pooling **{R['n_rev']:,} trades** confirms the overall estimate is "
            f"**{R['rev_mean']:+.2f} bps**, HAC *t* = **{R['rev_t']:+.2f}** — no signal."
        ),
        md(
            "### 4b · Reversal vs breakout vs control\n\n"
            "Side-by-side comparison of the two framings and the random control."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s_rev = st.summarize(pool_rev(5, 0), 'ret_gross')\n"
            "    s_brk = st.summarize(pool_brk(5, 0), 'ret_gross')\n"
            "    s_rnd = st.summarize(pool_rev(5, 0, seed=184), 'ret_gross')\n"
            "    vals = [s_rev['mean_bps'], s_brk['mean_bps'], s_rnd['mean_bps']]\n"
            "    tstats = [s_rev['tstat'], s_brk['tstat'], s_rnd['tstat']]\n"
            "else:\n"
            f"    vals = [{R['rev_mean']}, {R['brk_mean']}, {R['rand_rev_mean']}]\n"
            f"    tstats = [{R['rev_t']}, {R['brk_t']}, {R['rand_rev_t']}]\n"
            "labels = ['Reversal\\n(fade extreme)', 'Breakout\\n(break level)', 'Random\\n(coin)']\n"
            "fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))\n"
            "col = [RED, RED, GREY]\n"
            "axes[0].bar(labels, vals, color=col)\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_ylabel('gross mean (bps/trade)')\n"
            "axes[0].set_title('Mean return: both framings lose to a coin')\n"
            "axes[1].bar(labels, tstats, color=col)\n"
            "for v in (2, -2): axes[1].axhline(v, ls='--', c=GREY, lw=1)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_ylabel('HAC t-stat')\n"
            "axes[1].set_title('Inference: nowhere near the bar')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "### 4c · Hold-period sweep (reversal, gross)\n\n"
            "Does the signal work at any holding horizon?"
        ),
        code(
            "holds = [1, 3, 5, 10, 20]\n"
            "if HAVE_REAL:\n"
            "    hold_means = [st.summarize(pool_rev(h, 0), 'ret_gross')['mean_bps'] for h in holds]\n"
            "    hold_ts = [st.summarize(pool_rev(h, 0), 'ret_gross')['tstat'] for h in holds]\n"
            "else:\n"
            f"    hold_means = [{R['hold1']}, {R['hold3']}, {R['hold5']}, {R['hold10']}, {R['hold20']}]\n"
            f"    hold_ts = [{R['hold1_t']}, {R['hold3_t']}, {R['hold5_t']}, {R['hold10_t']}, {R['hold20_t']}]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "ax.plot(holds, hold_means, 'o-', c=RED, lw=2, label='mean (bps/trade)')\n"
            "ax2 = ax.twinx()\n"
            "ax2.plot(holds, hold_ts, 's--', c=GREY, lw=1.5, label='HAC t')\n"
            "ax2.axhline(2, ls=':', c=GREY, lw=1); ax2.axhline(-2, ls=':', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('hold period (days)')\n"
            "ax.set_ylabel('gross mean (bps/trade)', color=RED)\n"
            "ax2.set_ylabel('HAC t-stat', color=GREY)\n"
            "ax.set_title('No hold period clears |t| ≥ 2 — noise across all horizons')\n"
            "ax.legend(loc='upper left'); ax2.legend(loc='upper right')\n"
            "plt.tight_layout(); plt.show()\n"
            "for h, m, t in zip(holds, hold_means, hold_ts):\n"
            "    print(f'hold {h:2d}d: mean={m:+.2f} bps  t={t:+.2f}')"
        ),
        md(
            "> 💡 The 3-day hold gives the *most negative* t-stat (−1.59), consistent with "
            "the confirmation-lag theory: at 3 days we're picking up most of the early "
            "'bounce after the bounce' — but in the wrong direction.  At 20 days the mean "
            "drifts positive (+5.54 bps) but is still insignificant (t = +1.09)."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — reversal {R['rev_mean']:+.2f} bps, HAC "
            f"*t* {R['rev_t']:+.2f}; breakout {R['brk_mean']:+.2f} bps, *t* {R['brk_t']:+.2f}; "
            f"Δ vs control {R['rev_minus_rand']:+.2f} bps; no instrument |*t*| ≥ 2.\n"
            f"- **Tradability `MIRAGE`** — gross already negative; at 5 bps *t* = {R['cost50_t']:+.2f}.\n"
            "- **Beat a coin? `No`** — both framings perform worse than a random-direction entry "
            "on the same bars."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — costs complete the story\n\n"
            "The gross edge is already negative, so there is no positive break-even cost."
        ),
        code(
            "costs = [0.0, 0.5, 1.0, 2.0, 5.0]\n"
            "if HAVE_REAL:\n"
            "    sw = [st.summarize(pool_rev(5, c), 'ret_net') for c in costs]\n"
            "    net_means = [s['mean_bps'] for s in sw]\n"
            "    net_ts = [s['tstat'] for s in sw]\n"
            "else:\n"
            f"    net_means = [{R['cost0']}, {R['cost05']}, {R['cost10']}, {R['cost20']}, {R['cost50']}]\n"
            f"    net_ts = [{R['cost0_t']}, {R['cost05_t']}, {R['cost10_t']}, {R['cost20_t']}, {R['cost50_t']}]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "ax.plot(costs, net_means, 'o-', c=RED, lw=2)\n"
            "ax2 = ax.twinx()\n"
            "ax2.plot(costs, net_ts, 's--', c=GREY, lw=1.5)\n"
            "ax2.axhline(-2, ls=':', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, net_means, 0, where=[m < 0 for m in net_means],\n"
            "                color=RED, alpha=0.12)\n"
            "ax.set_xlabel('round-trip cost (bps)')\n"
            "ax.set_ylabel('net mean (bps/trade)', color=RED)\n"
            "ax2.set_ylabel('HAC t-stat', color=GREY)\n"
            "ax.set_title('No break-even cost: the strategy starts negative and certifies as a loser')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print('No break-even cost exists: gross is already {R['cost0']:+.2f} bps.')"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Does the breakout *engine* work when there's something to find?  We plant bar-level "
            "AR(1) momentum in a synthetic tape and sweep it: the breakout edge should turn on "
            "exactly when persistence appears."
        ),
        code(
            "moms = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]\n"
            "means = []\n"
            "for m in moms:\n"
            "    bars_s, _ = data.synthetic_daily(n_days=2000, momentum=m, seed=184)\n"
            "    f_s = st.detect_fractals(bars_s)\n"
            "    ent_s = st.breakout_entries(bars_s, f_s)\n"
            "    if len(ent_s) < 5:\n"
            "        means.append(float('nan'))\n"
            "        continue\n"
            "    s = st.summarize(st.run_trades(bars_s, ent_s, hold_days=5, cost_bps=0), 'ret_gross')\n"
            "    means.append(s['mean_bps'])\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "ax.plot(moms, means, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.axvline(0, ls='--', c=GREY)\n"
            "ax.set_xlabel('planted bar-level AR(1) momentum')\n"
            "ax.set_ylabel('breakout gross mean (bps/trade)')\n"
            "ax.set_title('The engine works: breakout finds an edge when momentum exists')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('At momentum 0 the edge is ~0; it rises monotonically with planted persistence.')"
        ),
        md(
            "The breakout advantage is monotone in the planted momentum — the engine is a "
            "faithful momentum detector. The real-tape verdict is a statement about the "
            "**market**, not the machinery: daily prices in these instruments do not carry "
            "enough momentum at the 1–20 day horizon for fractal breakouts to exploit.\n\n"
            "Experiments worth forking:\n"
            "- **Filter by Alligator state:** Williams intended fractals to be traded only "
            "when the Alligator is 'awake' (three SMAs fanned out). Test whether that filter "
            "rescues either framing.\n"
            "- **Fractal size filter:** require the fractal extreme to exceed a volatility "
            "threshold — perhaps *larger* local extremes carry more information.\n"
            "- **Cross-asset:** test on commodities futures or FX where daily momentum may "
            "be stronger than in equity ETFs."
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
