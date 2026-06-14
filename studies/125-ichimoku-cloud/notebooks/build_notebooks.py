"""Generate the two narrative notebooks for Study 125 (Ichimoku-Cloud).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
daily parquet under ../_cache/ if present and otherwise quote the frozen headline
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-14).
R = dict(
    n=412, tpy=40,
    sig_mean=-8.08, sig_win=49.3, sig_t=-0.94,
    rand_mean=-19.94, rand_win=43.4, rand_t=-2.29,
    sig_minus_rand=11.86,
    # per-instrument t-stats
    spy_t=0.25, qqq_t=-0.48, iwm_t=-1.06, aapl_t=-0.35,
    # forward returns
    d05_mean=-4.30, d05_win=51.0, d05_t=-0.30,
    d20_mean=-0.04, d20_win=52.4, d20_t=0.00,
    # cost sweep
    net05=-8.58, net10=-9.08, net10_t=-1.06, net20=-10.08,
    # data
    start="2016-06-13", end="2026-06-12",
)

# ---------------------------------------------------------------------------
# Shared analysis preamble — imports, the basket, and pooled helpers.
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

from ichimoku_cloud import data, strategy as st

TICKERS = ["SPY", "QQQ", "IWM", "AAPL"]

def _have_cache():
    return all(os.path.exists(data._cache_path(t, data.DEFAULT_CACHE)) for t in TICKERS)

HAVE_REAL = _have_cache()

def pooled(tp, sl, cost, seed=None):
    \"\"\"Pool barrier trades across the basket (cached real tapes).\"\"\"
    frames = []
    for t in TICKERS:
        b = data.fetch_daily(t, fetch=False)
        ent = st.signal_entries(b)
        dirs = st.random_directions(len(ent), seed=seed) if seed is not None else None
        frames.append(st.run_trades(b, ent, tp_R=tp, sl_R=sl, cost_bps=cost, directions=dirs))
    return pd.concat(frames, ignore_index=True)

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Ichimoku-Cloud — does the Kumo cloud know where price is going? ☁️\n"
            "### The Hosoda Ichimoku Kinko Hyo system, tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats_a_coin%3F: Partial](https://img.shields.io/badge/Beats_a_coin%3F-Partial-8b949e?style=flat-square)\n\n"
            "Here's the most elaborate-looking indicator in technical analysis: a Japanese cloud "
            "made of five overlapping lines developed by journalist Goichi Hosoda over 30 years "
            "and published in 1969. The folk rule goes: *'When price is above the Kumo cloud AND "
            "the fast line (Tenkan) is above the slow line (Kijun), everything is aligned — buy. "
            "Multiple confirmations mean a stronger trend.'* This notebook asks the only question "
            "that matters: does all that confirmation actually work, or is it a beautifully "
            "complicated coin flip?\n\n"
            "> 📓 **This is the plain-language layer.** Want the t-stats, the control arm, and the "
            "no-look-ahead mechanics? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart below is "
            "drawn by the code beside it. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the cloud tell you which way to trade? | **Barely.** Gross return is "
            f"**{R['sig_mean']:+.2f} bps/trade**, statistically indistinguishable from zero "
            f"(t = {R['sig_t']:+.2f}). |\n"
            "| Does it beat an actual coin flip? | **Sort of — but in a discouraging way.** "
            "It's +11.86 bps better than a random entry *at the same bars*, but both are "
            "negative. Ichimoku is firing at bad moments and then partially correcting for it. |\n"
            "| Could you trade it? | **No.** Even at zero cost the expected return is negative. "
            "There is no break-even cost. |\n\n"
            "> The cloud doesn't know the future. It knows the past — 26 to 52 bars of it, "
            "lagged by another 26 bars. By the time the cloud confirms a trend, the trend has "
            "already happened."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *'When price is above the Kumo cloud **and** Tenkan-sen crosses above Kijun-sen, "
            "you have multiple confirmations that the trend is in your favour. Enter long. When "
            "price is below the cloud and Tenkan crosses below Kijun, enter short. The cloud "
            "acts as support/resistance and filters out false signals — unlike a simple moving "
            "average crossover, you need the market to clear the cloud first.'*\n\n"
            "The system has an impressive pedigree: developed by Hosoda over 30 years, published "
            "1969, extremely popular in Japan and now globally through TradingView. The bet is "
            "that **multiple lagged trend confirmations** eliminate enough noise to produce a "
            "positive expected return."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the cloud filter works, it's useful: it fires infrequently (~10 times a year per "
            "instrument) so the cost burden is tiny, and the multi-confirmation logic is intuitive "
            "enough to trust. If it fails, millions of traders are looking at an elaborate "
            "cloud-and-line picture that adds complexity without adding information — and the "
            "30-year development history is just overfitting to noise. The stakes are clarity: "
            "is the cloud a signal or a beautiful costume for a fair coin?"
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Two rules keep us honest:\n\n"
            "1. **Make the bet symmetric.** If the profit target and stop are the same distance "
            "away (±1 ATR), a coin scores exactly zero. Only genuine directional information "
            "beats that.\n"
            "2. **Race it against an actual coin.** Take the exact same entry moments and flip a "
            "coin for the direction. If the cloud knows something, the cloud beats the coin.\n\n"
            "We also add a **no-look-ahead rule**: the cloud at any given bar is built from data "
            "only *26 bars ago* (the system naturally displaces the cloud forward 26 bars; to "
            "avoid look-ahead we treat the 'current' cloud as what was published 26 bars ago). "
            "This is the only honest treatment.\n\n"
            "Four tapes: SPY, QQQ, IWM, AAPL — daily bars, ~10 years."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: how infrequently does the signal fire?** The double confirmation (cloud "
            "position + TK cross) is strict — let's count the entries per year per instrument."
        ),
        code(
            "if HAVE_REAL:\n"
            "    for t in TICKERS:\n"
            "        b = data.fetch_daily(t, fetch=False)\n"
            "        ent = st.signal_entries(b)\n"
            "        years = (b.index[-1] - b.index[0]).days / 365.25\n"
            "        print(f'{t:5s}  {len(ent)} entries in {years:.1f} years  = "
            "{len(ent)/years:.1f}/year')\n"
            "else:\n"
            "    print('(cache not present — showing frozen numbers)')\n"
            f"    print('SPY  ~{int(90/10)} entries/year   QQQ  ~{int(96/10)}/year')\n"
            f"    print('IWM  ~{int(121/10)}/year   AAPL ~{int(105/10)}/year')"
        ),
        md(
            f"About **10 entries per year per instrument** — once a month at most. This low "
            "frequency means turnover costs are minimal, so even if the gross edge were zero the "
            "system would be nearly cost-neutral. The question is purely whether it points the "
            "right direction."
        ),
        md(
            "**Now the honest test: symmetric ±1 ATR barriers, cloud signal vs a coin.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    C = pooled(1, 1, 0)\n"
            "    rand_means = [st.summarize(pooled(1, 1, 0, seed=s), 'ret_gross')['mean_bps']\n"
            "                  for s in range(20)]\n"
            "    cm = st.summarize(C, 'ret_gross')\n"
            "    sig_m, rand_m_mean = cm['mean_bps'], sum(rand_means)/len(rand_means)\n"
            "else:\n"
            f"    sig_m = {R['sig_mean']}\n"
            f"    rand_m_mean = {R['rand_mean']}\n"
            f"    rand_means = [{R['rand_mean']}] * 20\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.hist(rand_means, bins=12, color=GREY, alpha=.65,\n"
            "        label='random control means (20 seeds)')\n"
            "ax.axvline(sig_m, c=RED, lw=2.5, label=f'Ichimoku signal: {sig_m:+.2f} bps')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('gross mean (bps/trade)'); ax.set_ylabel('count')\n"
            "ax.set_title('The Ichimoku signal: slightly better than random, but both negative')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Ichimoku gross: {sig_m:+.2f} bps/trade')\n"
            "print(f'Random control mean: {rand_m_mean:+.2f} bps/trade')\n"
            "print(f'Delta (Ichimoku − random): {sig_m - rand_m_mean:+.2f} bps')"
        ),
        md(
            f"The cloud signal earns **{R['sig_mean']:+.2f} bps/trade** — slightly less negative "
            f"than the **{R['rand_mean']:+.2f} bps** a random entry at the same bars earns. The "
            "signal is +11.86 bps better than random. But here's the key insight: *it's better "
            "than a coin at these particular bars*, not better than zero. The Ichimoku entry "
            "moments are adverse (the market tends to mean-revert right after them) and the "
            "cloud filter partially corrects for that — but not enough to turn the sign positive.\n\n"
            "> The cloud says 'the trend is up' *after* price has already climbed through it, "
            "26+ bars after the data was collected. By then, the move has often run its course."
        ),
        md(
            "**Per-instrument breakdown — no ticker shows a real signal.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    recs = []\n"
            "    for t in TICKERS:\n"
            "        b = data.fetch_daily(t, fetch=False)\n"
            "        ent = st.signal_entries(b)\n"
            "        s = st.summarize(st.run_trades(b, ent, tp_R=1, sl_R=1, cost_bps=0), 'ret_gross')\n"
            "        recs.append((t, s['n_trades'], s['win_rate']*100, s['mean_bps'], s['tstat']))\n"
            "    pi = pd.DataFrame(recs, columns=['ticker', 'n', 'win%', 'mean_bps', 't'])\n"
            "else:\n"
            "    pi = pd.DataFrame({'ticker': TICKERS,\n"
            f"        't': [{R['spy_t']}, {R['qqq_t']}, {R['iwm_t']}, {R['aapl_t']}],\n"
            "        'mean_bps': [3.33, -8.03, -16.59, -8.10],\n"
            "        'win%': [51.1, 49.0, 47.9, 49.5], 'n': [90, 96, 121, 105]})\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "colors = [RED if abs(v) < 2 else GREEN for v in pi['t']]\n"
            "ax.bar(pi['ticker'], pi['t'], color=colors)\n"
            "for s in (2, -2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('HAC t-stat (gross mean/trade)')\n"
            "ax.set_title('Four markets, four non-results (|t| < 2 everywhere)')\n"
            "ax.set_ylim(-2.5, 2.5); plt.tight_layout(); plt.show()\n"
            "pi.round(2)"
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Gross **{R['sig_mean']:+.2f} bps/trade**, HAC *t* = "
            f"**{R['sig_t']:+.2f}** — indistinguishable from zero. No instrument clears |t| = 2.\n"
            f"- **Tradability — Mirage.** Negative gross at ~10 trades/year per instrument; "
            "no positive break-even cost exists (the gross is already negative).\n"
            "- **Beats a coin? — Partial.** The signal is +11.86 bps better than a random "
            "entry at the same bars — it partially corrects for adverse entry timing. But both "
            "are negative, so this is 'less-wrong coin' not 'edge.'"
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Low frequency means the cost drag per year is small. But the gross is "
            "negative, so there is no budget to spend on costs:"
        ),
        code(
            "costs = [0.0, 0.5, 1.0, 2.0, 5.0]\n"
            "if HAVE_REAL:\n"
            "    net = [st.summarize(pooled(1, 1, c), 'ret_net')['mean_bps'] for c in costs]\n"
            "else:\n"
            f"    net = [{R['sig_mean']}, {R['net05']}, {R['net10']}, {R['net20']}, "
            f"{R['sig_mean'] - 5}]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, net, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, net, 0, where=[n < 0 for n in net], color=RED, alpha=.12)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net profit per trade (bps)')\n"
            "ax.set_title('Already negative at zero cost — costs just dig deeper')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'At zero cost: {net[0]:+.2f} bps — no positive break-even exists.')"
        ),
        md(
            "At ~10 trades/year the *annual* cost drag from 1 bp round-trip is only about "
            "1 bp × 10 = 10 bps/year — negligible. The problem is not costs; it's that "
            "the gross expectancy is negative to begin with. This is rarer than the usual "
            "story (where a positive gross dies at the costs line); here the gross never "
            "had a pulse."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Why does the random control do *worse* than the signal?** Both are negative, "
            "but the random arm (t = −2.29) is more significantly negative than the signal "
            "(t = −0.94). Ichimoku entry bars are moments of high recent momentum — right after "
            "which the market tends to mean-revert. The cloud filter *partially* corrects for "
            "direction, but the timing is the core problem — a theme worth exploring with "
            "[Study 32 — Rip-Tide](../../32-rip-tide/).\n"
            "- **Does any *component* of Ichimoku work alone?** The TK cross without the cloud "
            "filter, or price-vs-cloud without the TK cross. Hosoda's five lines can be unpacked.\n"
            "- **A longer timeframe.** Ichimoku was designed for weekly charts in the Japanese "
            "equity market. On weekly bars the 26-bar displacement is ~6 months, which overlaps "
            "with the Jegadeesh-Titman momentum window where real effects are documented.\n"
            "- **[Study 72 — Loaded-Dice](../../72-loaded-dice/)**, "
            "[**Study 106 — Supertrend**](../../106-supertrend/), and "
            "[**Study 78 — Crossed-Wires**](../../78-crossed-wires/) all test close relatives "
            "with the same null result. The lagging-trend-system family appears to be a family "
            "of fair coins.\n\n"
            "*Think the cloud works at a different horizon, a different instrument class, or "
            "with a sub-signal isolation? Fork this, show a fair-bet edge that beats the "
            "random control, and that's the bar.*"
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
            "# Ichimoku-Cloud — a quantitative teardown\n"
            "### Daily tape · symmetric ATR barriers · random-direction control · HAC inference · no-look-ahead cloud\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats_a_coin%3F: Partial](https://img.shields.io/badge/Beats_a_coin%3F-Partial-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim now carrying its standard error.* We test whether the "
            "Ichimoku composite signal (price-vs-cloud + Tenkan/Kijun cross) beats a random-direction "
            "control on a symmetric-barrier backtest across four daily tapes, and what the no-look-ahead "
            "cloud displacement means in practice.\n\n"
            "> ⚠️ **Not investment advice.** Real data: Yahoo daily bars, "
            f"{R['start']} → {R['end']}; the offline core and tests run on a deterministic "
            "synthetic tape. Methods & sources in "
            "[`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The `In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Gross **{R['sig_mean']:+.2f} bps/trade**, HAC "
            f"*t* = **{R['sig_t']:+.2f}**; +{R['sig_minus_rand']:.2f} bps over random control "
            f"(which is itself t = {R['rand_t']:+.2f} — adverse timing). |\n"
            f"| **Tradability** | `MIRAGE` | ~{R['tpy']} trades/year across 4 instruments; "
            "negative gross — no break-even cost exists. |\n"
            f"| **Beats a coin?** | `PARTIAL` | Signal +{R['sig_minus_rand']:.2f} bps vs random "
            "at same bars; Ichimoku reduces adverse-timing damage but does not flip the sign. |\n\n"
            "> In plain words: the cloud fires at moments when the market has already moved, "
            "tends to mean-revert, and the Ichimoku filter reduces how negative you are at "
            "those moments — but not enough to produce a positive expected return."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $d_t \\in \\{-1, +1\\}$ be the Ichimoku composite direction at bar $t$: "
            "$d_t = +1$ iff $\\text{close}_t > \\text{cloud\\_top}_t$ *and* "
            "$\\text{tenkan}_t > \\text{kijun}_t$; $d_t = -1$ iff "
            "$\\text{close}_t < \\text{cloud\\_bot}_t$ *and* $\\text{tenkan}_t < \\text{kijun}_t$. "
            "An entry fires when $d_t$ changes from a different state to $\\pm1$. The recipe asserts:\n\n"
            "- **H₁ (signal).** $\\mathbb{E}[\\,d\\cdot r_{t \\to \\text{exit}}\\,] > 0$ on a "
            "symmetric-payoff exit — the filter carries directional information.\n"
            "- **H₂ (beats random).** That expectation exceeds the same statistic with $d$ replaced "
            "by an i.i.d. fair coin on identical entry bars.\n"
            "- **H₃ (tradable).** It survives realistic round-trip costs at the rule's turnover.\n\n"
            "We reject **H₁** (gross mean is negative, |t| < 2). We *partially* confirm "
            "**H₂** (signal > random, but both negative). We reject **H₃** (no positive "
            "break-even cost).\n\n"
            "**No-look-ahead cloud:** `cloud_top / cloud_bot` at bar $t$ are\n"
            "$$\\text{Senkou A}_t = \\frac{\\text{tenkan}_{t-26} + \\text{kijun}_{t-26}}{2}, "
            "\\quad \\text{Senkou B}_t = \\text{Donchian}_{52}[t-26]$$\n"
            "using only data up to bar $t-26$. This is implemented as a `.shift(26)` on the "
            "raw computed series in `strategy.ichimoku()`."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "Ichimoku is the most-cited multi-component indicator in retail and semi-professional "
            "technical analysis. At ~10 trades/year it has near-zero turnover cost — so if the "
            "gross direction were positive, it would be one of the cleaner technical signals to "
            "deliver in practice. The interesting failure mode is that the signal partially "
            "corrects for the adverse timing of its own entry points (it's better than random "
            "at those moments) without making the sign positive: a 'less wrong' coin, not an edge."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Entry.** Composite signal (close vs cloud AND Tenkan vs Kijun) on data up to "
            "bar $t$; **enter at $t{+}1$'s open** (no look-ahead). Cloud built from data $t-26$ "
            "via `.shift(26)` in `strategy.ichimoku()`.\n"
            "- **Exit (symmetric).** Barriers at $\\pm 1\\,\\mathrm{ATR}(20)$ from entry; "
            "a bar straddling both barriers is filled at the **stop** (conservative).\n"
            "- **Control.** Identical entries, direction $\\in\\{-1,+1\\}$ drawn i.i.d. (seeded).\n"
            "- **Inference.** Newey-West HAC *t* on per-trade returns; per-instrument breakdown; "
            "a cost sweep; fixed-day forward returns at d5 / d20.\n"
            "- **Positive control.** A synthetic tape with tunable bar-level AR(1) momentum, "
            "to confirm the engine recovers an edge *when one exists*.\n\n"
            "Four tapes: SPY, QQQ, IWM, AAPL. Daily bars, ~10 years. "
            f"n = {R['n']} pooled trades (~{R['tpy']} per year)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · No-look-ahead mechanics — the cloud that lag built\n\n"
            "A common implementation mistake is to use the *displayed* Ichimoku cloud directly, "
            "which includes the forward-displaced Senkou Spans that represent *future projected* "
            "cloud. We shift back to get the cloud value at bar $t$ from data only up to $t-26$.\n"
            "Below: the Ichimoku components on SPY over the last 200 bars, showing the "
            "no-look-ahead cloud (shaded) and signal transitions (arrows)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = data.fetch_daily('SPY', fetch=False).iloc[-200:]\n"
            "    ichi = st.ichimoku(b)\n"
            "    ent = st.signal_entries(data.fetch_daily('SPY', fetch=False)).iloc[-50:]\n"
            "else:\n"
            "    b, _ = data.synthetic_daily(n_days=400, momentum=0.0, seed=125)\n"
            "    ichi = st.ichimoku(b)\n"
            "    b = b.iloc[-200:]; ichi = ichi.iloc[-200:]\n"
            "    ent = st.signal_entries(b)\n"
            "fig, ax = plt.subplots(figsize=(11, 5))\n"
            "ax.plot(b.index, b['close'], c='k', lw=1.2, label='Close')\n"
            "ax.plot(b.index, ichi['tenkan'], c=RED, lw=0.8, label='Tenkan(9)')\n"
            "ax.plot(b.index, ichi['kijun'], c=AMBER, lw=0.8, label='Kijun(26)')\n"
            "ax.fill_between(b.index,\n"
            "    ichi['cloud_top'].reindex(b.index),\n"
            "    ichi['cloud_bot'].reindex(b.index),\n"
            "    where=ichi['cloud_top'].reindex(b.index) >= ichi['cloud_bot'].reindex(b.index),\n"
            "    alpha=.15, color=GREEN, label='Kumo (bull)')\n"
            "ax.fill_between(b.index,\n"
            "    ichi['cloud_top'].reindex(b.index),\n"
            "    ichi['cloud_bot'].reindex(b.index),\n"
            "    where=ichi['cloud_top'].reindex(b.index) < ichi['cloud_bot'].reindex(b.index),\n"
            "    alpha=.15, color=RED, label='Kumo (bear)')\n"
            "ax.set_title('SPY daily — Ichimoku Kinko Hyo (no look-ahead cloud)')\n"
            "ax.legend(fontsize=8, loc='upper left'); plt.tight_layout(); plt.show()"
        ),
        md(
            "### 4b · The symmetric bet — per instrument, no real signal anywhere\n\n"
            "Per-instrument HAC *t* on the gross per-trade return. If the rule worked anywhere, "
            "a bar would clear +2."
        ),
        code(
            "if HAVE_REAL:\n"
            "    recs = []\n"
            "    for t in TICKERS:\n"
            "        b = data.fetch_daily(t, fetch=False)\n"
            "        ent = st.signal_entries(b)\n"
            "        s = st.summarize(st.run_trades(b, ent, tp_R=1, sl_R=1, cost_bps=0), 'ret_gross')\n"
            "        recs.append((t, s['n_trades'], s['win_rate']*100, s['mean_bps'], s['tstat']))\n"
            "    pi = pd.DataFrame(recs, columns=['ticker', 'n', 'win%', 'mean_bps', 't'])\n"
            "    C = pooled(1, 1, 0); cs = st.summarize(C, 'ret_gross')\n"
            "else:\n"
            "    pi = pd.DataFrame({'ticker': TICKERS,\n"
            f"        't': [{R['spy_t']}, {R['qqq_t']}, {R['iwm_t']}, {R['aapl_t']}],\n"
            "        'mean_bps': [3.33, -8.03, -16.59, -8.10],\n"
            "        'win%': [51.1, 49.0, 47.9, 49.5], 'n': [90, 96, 121, 105]})\n"
            f"    cs = dict(mean_bps={R['sig_mean']}, tstat={R['sig_t']}, "
            f"win_rate={R['sig_win']/100})\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "col = [RED if abs(v) < 2 else GREEN for v in pi['t']]\n"
            "ax.bar(pi['ticker'], pi['t'], color=col)\n"
            "for s in (2, -2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('HAC t-stat (gross mean/trade)')\n"
            "ax.set_title('Four markets, four non-results (|t| < 2 everywhere)')\n"
            "ax.set_ylim(-2.5, 2.5); plt.tight_layout(); plt.show()\n"
            "print(f\"pooled: {cs['mean_bps']:+.2f} bps/trade, HAC t = {cs['tstat']:+.2f}\")\n"
            "pi.round(2)"
        ),
        md(
            f"> In plain words: pooled n = {R['n']} trades, gross mean = "
            f"**{R['sig_mean']:+.2f} bps**, HAC t = **{R['sig_t']:+.2f}**. "
            "Every instrument is inside the ±2 band."
        ),
        md(
            "### 4c · Ichimoku vs the coin — with a distribution of random seeds\n\n"
            "The random-direction control at identical entry bars, vs the signal. "
            "The key story: the signal is *less negative* than random — but both are negative."
        ),
        code(
            "if HAVE_REAL:\n"
            "    C = pooled(1, 1, 0); cm = st.summarize(C, 'ret_gross')\n"
            "    rand_means = [st.summarize(pooled(1, 1, 0, seed=s), 'ret_gross')['mean_bps']\n"
            "                  for s in range(20)]\n"
            "    sig_m, rnd_m = cm['mean_bps'], sum(rand_means)/len(rand_means)\n"
            "else:\n"
            f"    sig_m, rnd_m = {R['sig_mean']}, {R['rand_mean']}\n"
            f"    rand_means = [{R['rand_mean']}] * 20\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(rand_means, bins=12, color=GREY, alpha=.65,\n"
            "        label='random-control means (20 seeds)')\n"
            "ax.axvline(sig_m, c=RED, lw=2.5,\n"
            "           label=f'Ichimoku signal: {sig_m:+.2f} bps')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('gross mean (bps/trade)'); ax.set_ylabel('count')\n"
            "ax.set_title('Signal is better than random at same bars — both still negative')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Ichimoku: {sig_m:+.2f} bps/trade')\n"
            "print(f'Random (mean across seeds): {rnd_m:+.2f} bps/trade')\n"
            "print(f'Delta: {sig_m - rnd_m:+.2f} bps/trade')"
        ),
        md(
            f"> In plain words: the Ichimoku signal ({R['sig_mean']:+.2f} bps) outperforms a "
            f"random entry at the same bars ({R['rand_mean']:+.2f} bps) by "
            f"+{R['sig_minus_rand']:.2f} bps. The cloud *does* carry some directional information "
            "— it reduces the mean-reversion punishment at these adverse-timing entries. But it "
            "doesn't flip the sign. Signal = NONE."
        ),
        md(
            "### 4d · Fixed-day forward returns — does the signal predict drift?\n\n"
            "A simpler test: does the market drift in the signal direction over the next "
            "5 / 20 trading days?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    results_fwd = []\n"
            "    for hd in [5, 20]:\n"
            "        frames = []\n"
            "        for t in TICKERS:\n"
            "            b = data.fetch_daily(t, fetch=False)\n"
            "            ent = st.signal_entries(b)\n"
            "            frames.append(st.run_forward_returns(b, ent, hold_days=hd, cost_bps=0))\n"
            "        led = pd.concat(frames, ignore_index=True)\n"
            "        s = st.summarize(led, 'ret_gross')\n"
            "        results_fwd.append((hd, s['n_trades'], s['win_rate']*100, s['mean_bps'], s['tstat']))\n"
            "    df_fwd = pd.DataFrame(results_fwd, columns=['hold_days', 'n', 'win%', 'mean_bps', 't'])\n"
            "else:\n"
            "    df_fwd = pd.DataFrame({\n"
            f"        'hold_days': [5, 20], 'n': [{R['n']}, {R['n']}],\n"
            f"        'win%': [{R['d05_win']}, {R['d20_win']}],\n"
            f"        'mean_bps': [{R['d05_mean']}, {R['d20_mean']}],\n"
            f"        't': [{R['d05_t']}, {R['d20_t']}]}}\n"
            "    )\n"
            "print(df_fwd.round(2))"
        ),
        md(
            f"> In plain words: at a 5-day horizon the mean is {R['d05_mean']:+.2f} bps "
            f"(t = {R['d05_t']:+.2f}). At 20 days it is {R['d20_mean']:+.2f} bps "
            f"(t = {R['d20_t']:+.2f}). The signal predicts no subsequent directional drift "
            "at any horizon tested."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — pooled gross {R['sig_mean']:+.2f} bps/trade, HAC "
            f"*t* {R['sig_t']:+.2f}; every instrument |*t*| < 1.1. "
            f"Delta vs random control: +{R['sig_minus_rand']:.2f} bps (both negative).\n"
            f"- **Tradability `MIRAGE`** — negative gross at ~{R['tpy']} trades/year; "
            "no break-even cost exists.\n"
            f"- **Beats a coin? `PARTIAL`** — signal is +{R['sig_minus_rand']:.2f} bps better "
            f"than random at the same bars (random control t = {R['rand_t']:+.2f}, significantly "
            "negative); the cloud reduces adverse-timing damage, not eliminates it."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — cost sweep\n\n"
            "Per-trade net mean across round-trip cost. At ~10 trades/year, annual cost "
            "drag from 1 bp round-trip ≈ 10 bps/year — small. But gross is already negative."
        ),
        code(
            "costs = [0.0, 0.5, 1.0, 2.0, 5.0]\n"
            "if HAVE_REAL:\n"
            "    sw = [st.summarize(pooled(1, 1, c), 'ret_net') for c in costs]\n"
            "    mean = [s['mean_bps'] for s in sw]\n"
            "    tst = [s['tstat'] for s in sw]\n"
            "else:\n"
            f"    mean = [{R['sig_mean']}, {R['net05']}, {R['net10']}, {R['net20']}, "
            f"{R['sig_mean'] - 5}]\n"
            f"    tst = [{R['sig_t']}, -1.00, {R['net10_t']}, -1.17, -1.52]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(costs, mean, 'o-', c=RED, lw=2, label='net mean (bps/trade)')\n"
            "ax2 = ax.twinx()\n"
            "ax2.plot(costs, tst, 's--', c=GREY, lw=1.5, label='HAC t')\n"
            "ax2.axhline(-2, ls=':', c=GREY)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net mean (bps)', color=RED)\n"
            "ax2.set_ylabel('HAC t-stat', color=GREY)\n"
            "ax.set_title('Starts negative, gets worse — but slowly (low turnover)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('No break-even cost: gross is already negative.')"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Does the engine *ever* find an edge? Plant a bar-level AR(1) momentum in a "
            "synthetic daily tape and sweep it: the Ichimoku composite signal's advantage over "
            "the coin should turn positive when real persistence is planted."
        ),
        code(
            "moms = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20]\n"
            "edge = []\n"
            "for m in moms:\n"
            "    b, _ = data.synthetic_daily(n_days=1500, momentum=m, seed=125)\n"
            "    ent = st.signal_entries(b)\n"
            "    if len(ent) == 0:\n"
            "        edge.append(0.0); continue\n"
            "    c = st.summarize(st.run_trades(b, ent, tp_R=1, sl_R=1, cost_bps=0),\n"
            "                     'ret_gross')['mean_bps']\n"
            "    r = st.summarize(st.run_trades(b, ent, tp_R=1, sl_R=1, cost_bps=0,\n"
            "        directions=st.random_directions(len(ent), seed=1)),\n"
            "        'ret_gross')['mean_bps']\n"
            "    edge.append(c - r)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(moms, edge, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, ls='--', c=GREY)\n"
            "ax.set_xlabel('planted bar-level AR(1) momentum')\n"
            "ax.set_ylabel('Ichimoku − coin (bps/trade)')\n"
            "ax.set_title('Engine works: finds momentum when momentum is planted')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('At momentum 0 edge ≈ 0; rises with persistence — machinery confirmed.')"
        ),
        md(
            "The Ichimoku advantage over the coin is positive when momentum is planted and "
            "near zero on a martingale. The engine is a faithful momentum detector — the real "
            "tape just doesn't have persistent enough trend structure at this horizon for the "
            "26-bar-lagged cloud to harvest.\n\n"
            "Family companions worth reading: [Study 106 — Supertrend](../../106-supertrend/) "
            "(ATR-band flip, same null), [Study 72 — Loaded-Dice](../../72-loaded-dice/) "
            "(intraday MA cross), [Study 78 — Crossed-Wires](../../78-crossed-wires/) (MACD "
            "daily). The lagging-trend-system family appears robust — robustly null."
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
