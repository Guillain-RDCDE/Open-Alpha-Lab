"""Generate the two narrative notebooks for Study 699 (Butterfly-Harmonic).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached daily
basket tapes under ../_cache/ and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere with
no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily bars,
# SPY/QQQ/AAPL/MSFT/NVDA 2001-07-10 -> 2026-06-30, TSLA 2010-06-29 -> 2026-06-30;
# pct=0.02 zigzag, AB retrace 0.786 +-0.06 of XA, BC retrace 0.382-0.886 of AB,
# D = X - ext*(A-X) with ext in [1.27, 1.618], 120-session touch window).
R = dict(
    basket=("SPY", "QQQ", "AAPL", "MSFT", "TSLA", "NVDA"),
    asof="2026-06-30",
    n_pivots=6055, n_cand_fib=89, n_fib=50,
    n_cand_plac=110, n_plac=36,
    fib={1: dict(hit=48.0, mean=45.34, t=0.86), 5: dict(hit=54.0, mean=133.75, t=2.28),
         10: dict(hit=48.0, mean=-7.99, t=-0.10)},
    wilson_fib=(40.4, 67.0),
    per_instrument={
        "SPY":  dict(n=3,  mean=-186.03, t=None),
        "QQQ":  dict(n=10, mean=297.91,  t=5.39),
        "AAPL": dict(n=10, mean=15.85,   t=0.18),
        "MSFT": dict(n=10, mean=66.87,   t=0.50),
        "TSLA": dict(n=9,  mean=419.34,  t=3.34),
        "NVDA": dict(n=8,  mean=-41.86,  t=-0.30),
    },
    # Signal axis — vs random-day base rate (matched direction mix, 20 seeds)
    base={1: dict(mean=-12.72, t=1.09), 5: dict(mean=-24.11, t=2.18), 10: dict(mean=-57.28, t=0.55)},
    n_base=1000,
    base_per_instrument={
        "SPY":  dict(base_mean=-22.79, t=-2.05),
        "QQQ":  dict(base_mean=-17.17, t=2.62),
        "AAPL": dict(base_mean=-52.38, t=0.74),
        "MSFT": dict(base_mean=-30.65, t=0.52),
        "TSLA": dict(base_mean=-4.05,  t=2.14),
        "NVDA": dict(base_mean=-12.31, t=-0.13),
    },
    bonf_n_tests=7, bonf_thr=2.69, bonf_uncorrected=4, bonf_survive=0,
    # Tradability axis — the fade timer, net of costs
    timer={
        1:  dict(gross=45.34,  t_gross=0.86,  net5=35.34,  t5=0.67,  net10=25.34,  t10=0.48),
        3:  dict(gross=106.21, t_gross=2.70,  net5=96.21,  t5=2.45,  net10=86.21,  t10=2.19),
        5:  dict(gross=133.75, t_gross=2.28,  net5=123.75, t5=2.11,  net10=113.75, t10=1.94),
        10: dict(gross=-7.99,  t_gross=-0.10, net5=-17.99, t5=-0.23, net10=-27.99, t10=-0.36),
        20: dict(gross=45.16,  t_gross=0.31,  net5=35.16,  t5=0.24,  net10=25.16,  t10=0.17),
    },
    # Third axis — vs placebo
    plac={1: dict(mean=40.67, t=0.06), 5: dict(mean=71.75, t=0.55), 10: dict(mean=178.72, t=-1.33)},
    plac_per_instrument={
        "SPY":  dict(n=1,  mean=-70.40,  beats=False),
        "QQQ":  dict(n=6,  mean=164.17,  beats=True),
        "AAPL": dict(n=4,  mean=-47.97,  beats=True),
        "MSFT": dict(n=6,  mean=260.42,  beats=False),
        "TSLA": dict(n=9,  mean=363.37,  beats=True),
        "NVDA": dict(n=10, mean=-297.24, beats=True),
    },
    n_beats_plac=4,
    # synthetic control
    syn_null_mean=0.28, syn_null_sd=0.85, syn_null_fire=1, syn_null_seeds=20,
    syn_planted_n=31, syn_planted_mean=232.23, syn_planted_t=4.30,
    fp={"SPY": "3baddd006483", "QQQ": "8054251241a0", "AAPL": "9dd9c3863af9",
        "MSFT": "291fdd9717fc", "TSLA": "f8ca92e420b8", "NVDA": "8142a8071b07"},
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Beats_a_placebo%3F: Mixed](https://img.shields.io/badge/Beats_a_placebo%3F-Mixed-8b949e?style=flat-square)\n\n"
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

from butterfly_harmonic import data, strategy as st

PCT = 0.02
HAVE_REAL = data.have_real()
BARS = data.load_basket() if HAVE_REAL else {}
print("real cache present:", HAVE_REAL, "| basket:", data.BASKET)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the Butterfly harmonic actually call the turn? 🦋\n"
            "### Five dots, one Fibonacci overshoot, and a stopwatch\n\n"
            + BADGES +
            "Of all the \"harmonic patterns\" retail charting software draws for you, the "
            "**Butterfly** is the boldest: it says price will **blow past** a prior swing low "
            "(or high) by a very specific amount — **1.27 to 1.618 times** the move that "
            "started it — and then, right there, turn around. Not retrace back toward safety. "
            "*Overshoot, then reverse.* That's a striking claim, and it's exactly why the "
            "Butterfly gets its own name in the harmonic-pattern zoo instead of just being "
            "\"a Gartley\" or \"a Bat.\"\n\n"
            "So we built a robot that finds every Butterfly on six liquid tapes over the last "
            "21-25 years — no eyeballing, no cherry-picking — and asked whether fading price "
            "right where the pattern says to would actually have made money.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the Bonferroni correction and "
            "the placebo math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** Pivots are a confirmed percentage zig-zag (known only after "
            "price reverses enough to lock them in — no look-ahead). Every chart is drawn by "
            "the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does price turn where the Butterfly says it should? | **Maybe, barely — and it "
            f"doesn't survive a second look.** Over a 5-day hold the pattern averages "
            f"**+{R['fib'][5]['mean']:.0f} bps**, which *looks* like a real signal on its own "
            f"(a *t*-stat of {R['fib'][5]['t']:.2f}). |\n"
            f"| Is that a fluke of testing six stocks? | **Yes, once you check.** Correcting "
            f"for having looked at six tickers, **{R['bonf_survive']} of {R['bonf_n_tests']}** "
            "tests survive — none, including the headline number itself. |\n"
            f"| Does the exact 0.786 / 1.27-1.618 ratio pair matter? | **Not provably.** Swap "
            "those numbers for random ratios on the *identical* pivots and you do about as "
            f"well (Welch *t* = {R['plac'][5]['t']:.2f} — nowhere near significant). |\n"
            "| Can you trade it? | **Not really.** The pattern completes roughly **once every "
            "4-5 years per stock** — you'd wait most of a market cycle for one signal — and "
            "the apparent edge vanishes by day 10. |\n\n"
            "> A number that looks real, on a pattern that fires once in a blue moon, that "
            "stops looking real the moment you check your work twice."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Find four swing points X-A-B-C. If B retraces **78.6%** of the X-A move and "
            "C bounces back **38-89%** of the A-B move, project a fifth point D that "
            "**overshoots X by 1.27 to 1.618 times** the original X-A leg. That overshoot is "
            "the market running out of gas — a high-probability reversal.\"*\n\n"
            "That's Scott Carney's Butterfly (*Harmonic Trading*, 1998-2010), building on "
            "Bryce Gilmore's extension work. It's baked into TradingView, MetaTrader and "
            "Thinkorswim's auto-scanners as one of four named \"harmonic\" shapes — Gartley, "
            "Bat, Butterfly, Crab — each just a different pair of Fibonacci numbers on the "
            "same five-dot skeleton."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a pattern this specific — two decimal ratios chained across four historical "
            "swings — genuinely forecasts a turn, that's a striking, calculator-testable crack "
            "in market efficiency. And harmonic trading is popular *precisely* because the "
            "charts look uncanny after the fact: hindsight makes any completed pattern look "
            "inevitable.\n\n"
            "The trap is exactly that hindsight. A trader hand-labelling swings on a chart, "
            "with the benefit of already knowing what happened next, will always find the "
            "Butterfly that worked. So we removed the human: a mechanical rule finds every "
            "qualifying X-A-B-C on the tape, in real time, with no knowledge of the future — "
            "then we measure what actually happened at D."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The basket.** {', '.join(R['basket'])} — six liquid names, daily bars, "
            f"2001/2010 → {R['asof']}.\n"
            "- **The rule, mechanically.** A confirmed zig-zag finds swing pivots (only known "
            "once price has reversed enough to lock them in). Every qualifying X-A-B-C "
            "projects a D level; we scan forward up to 120 sessions for the first touch.\n"
            "- **The trade.** Fade at the touch — bet on the reversal the pattern predicts — "
            "and measure the forward return.\n"
            "- **Two honest baselines, announced up front.** (1) A **random-day base rate**: "
            "same ticker, same number of trades, same bullish/bearish mix, just on random "
            "days — kills the \"stocks drift up anyway\" confound. (2) A **placebo ratio "
            "grid**: the identical pivots, but with the 0.786/1.27-1.618 numbers replaced by "
            "random ones — kills the \"any extension projection would work\" confound. And "
            "because we test six tickers, we **correct for that** (Bonferroni) before calling "
            "anything real."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Fade the D-touch, hold 5 days, average across every "
            "qualifying Butterfly on the basket."
        ),
        code(
            "if HAVE_REAL:\n"
            "    frames = []\n"
            "    for t in data.BASKET:\n"
            "        _, _, ledger = st.detect_and_scan(BARS[t], pct=PCT, cost_bps=0.0)\n"
            "        if not ledger.empty:\n"
            "            frames.append(ledger)\n"
            "    FIB = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()\n"
            "    s5 = st.summarize(FIB, 'ret_gross_5')\n"
            "    n5, m5, t5 = s5['n'], s5['mean_bps'], s5['t']\n"
            "else:\n"
            "    n5, m5, t5 = R['n_fib'], R['fib'][5]['mean'], R['fib'][5]['t']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['Butterfly D-touch\\n(5-day fade)'], [m5], color=AMBER, width=.5)\n"
            "ax.annotate(f'{m5:+.0f} bps\\n(n={n5}, t={t5:+.2f})', (0, m5), ha='center',\n"
            "            va='bottom' if m5 >= 0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean forward return (bps)')\n"
            "ax.set_title('The headline number — looks real on its own')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'n={n5}  mean={m5:+.2f}bps  t={t5:+.2f}')"
        ),
        md(
            f"**{R['fib'][5]['mean']:+.0f} bps** on average, *t* = **{R['fib'][5]['t']:.2f}** — "
            "that clears the standard \"is this real\" bar of 2, all on its own. If we stopped "
            "here, this would look like a genuine discovery. We don't stop here.\n\n"
            "**Second, the honest baseline.** Stocks drift up. A random buy on a random day, "
            "matched to the same bullish/bearish mix as our Butterfly trades, earns *something* "
            "just from that drift. Does the Butterfly beat that?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    fib_by_t, base_frames = {}, []\n"
            "    for t in data.BASKET:\n"
            "        _, _, l = st.detect_and_scan(BARS[t], pct=PCT, cost_bps=0.0)\n"
            "        fib_by_t[t] = l\n"
            "        if len(l):\n"
            "            mix = float((l['reversal_dir'] > 0).mean())\n"
            "            for s in range(20):\n"
            "                base_frames.append(st.base_rate_ledger(BARS[t], len(l), mix,\n"
            "                    seed=st._seed_from(699, t, s)))\n"
            "    BASE = pd.concat(base_frames, ignore_index=True) if base_frames else pd.DataFrame()\n"
            "    sb = st.summarize(BASE, 'ret_gross_5')\n"
            "    bm, wt = sb['mean_bps'], st.welch_t(FIB['ret_gross_5'], BASE['ret_gross_5'])\n"
            "else:\n"
            "    bm, wt = R['base'][5]['mean'], R['base'][5]['t']\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.4))\n"
            "ax.bar(['Butterfly D-touch', 'random-day\\nbase rate'], [m5, bm], color=[AMBER, GREY], width=.55)\n"
            "for i, v in enumerate([m5, bm]):\n"
            "    ax.annotate(f'{v:+.0f} bps', (i, v), ha='center', va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean 5-day return (bps)')\n"
            "ax.set_title(f'Beats a matched-direction random day (Welch t = {wt:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Butterfly {m5:+.1f} bps vs base rate {bm:+.1f} bps  Welch t={wt:+.2f}')"
        ),
        md(
            f"Butterfly still beats the random-day baseline (Welch *t* = "
            f"**{R['base'][5]['t']:.2f}**) — so it isn't *purely* the market's drift. But look "
            "at the sample: **50 touches, spread across six stocks, over 21-25 years.** That's "
            "roughly two Butterfly completions a year, *pooled across the whole basket* — and "
            "we tested **six different tickers**. When you look in six places, one of them "
            "looking good by chance is the expectation, not the exception."
        ),
        code(
            "if HAVE_REAL:\n"
            "    per_t = {t: st.summarize(fib_by_t[t], 'ret_gross_5') for t in data.BASKET}\n"
            "    tickers = list(data.BASKET)\n"
            "    means = [per_t[t]['mean_bps'] for t in tickers]\n"
            "else:\n"
            "    tickers = list(R['per_instrument'])\n"
            "    means = [R['per_instrument'][t]['mean'] for t in tickers]\n"
            "cols = [GREEN if m > 0 else RED for m in means]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "ax.bar(tickers, means, color=cols, width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean 5-day fade return (bps)')\n"
            "ax.set_title('Six tickers, six different answers — including one negative')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({t: round(m,1) for t,m in zip(tickers, means)})"
        ),
        md(
            "SPY — the single most liquid, least idiosyncratic name in the basket — comes back "
            "**negative**. QQQ and TSLA look great. That spread is the signature of a thin, "
            "noisy sample dressed up as a discovery: correcting properly for having tested six "
            f"tickers, **{R['bonf_survive']} of {R['bonf_n_tests']}** results survive.\n\n"
            "**Finally, the placebo.** Does the *specific* 0.786 / 1.27-1.618 grid matter, or "
            "would any similarly-shaped extension have done just as well?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    plac_frames = []\n"
            "    for t in data.BASKET:\n"
            "        _, _, lp = st.detect_and_scan(BARS[t], pct=PCT, placebo=True, seed=699, cost_bps=0.0)\n"
            "        if not lp.empty:\n"
            "            plac_frames.append(lp)\n"
            "    PLAC = pd.concat(plac_frames, ignore_index=True) if plac_frames else pd.DataFrame()\n"
            "    sp = st.summarize(PLAC, 'ret_gross_5')\n"
            "    pm, wtp = sp['mean_bps'], st.welch_t(FIB['ret_gross_5'], PLAC['ret_gross_5'])\n"
            "else:\n"
            "    pm, wtp = R['plac'][5]['mean'], R['plac'][5]['t']\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.4))\n"
            "ax.bar(['Butterfly\\n(0.786 / 1.27-1.618)', 'placebo\\n(random ratios)'], [m5, pm],\n"
            "       color=[AMBER, GREY], width=.55)\n"
            "for i, v in enumerate([m5, pm]):\n"
            "    ax.annotate(f'{v:+.0f} bps', (i, v), ha='center', va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean 5-day return (bps)')\n"
            "ax.set_title(f'The magic numbers barely beat random ones (Welch t = {wtp:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Butterfly {m5:+.1f} bps vs placebo {pm:+.1f} bps  Welch t={wtp:+.2f}')"
        ),
        md(
            f"Butterfly comes out ahead of the placebo on paper (**{R['plac'][5]['mean']:+.0f} "
            f"bps** vs its random-ratio control), but the gap is nowhere near statistically "
            f"real (Welch *t* = **{R['plac'][5]['t']:.2f}**). The specific Fibonacci numbers — "
            "0.786, 1.27, 1.618 — are not doing detectably more work than an arbitrary "
            "extension-and-reversal rule would."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** A number that clears the bar on its own (*t* ≈ 2.2-2.3) "
            f"built from only {R['n_fib']} events, that fails a six-ticker correction "
            f"({R['bonf_survive']}/{R['bonf_n_tests']} survive) and doesn't beat a placebo "
            "(*t* = 0.55).\n"
            "- **Tradability — Mirage.** The pattern fires roughly once every 4-5 years per "
            "stock, and whatever edge exists is gone by day 10.\n"
            "- **\"Beats a random ratio grid?\" — Mixed.** Ahead on paper on 4 of 6 tickers, "
            "not statistically distinguishable from zero."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The general lesson: check your work twice.** A single-ticker *t* of 2 means "
            "very little once you've searched six tickers for the best one — that's exactly "
            "what a Bonferroni correction is for, and exactly what kills this result.\n"
            "- **Sibling studies:** [468-gartley-harmonic](../../468-gartley-harmonic/) (the "
            "classic five-point pattern, D stays *inside* the X-A range) and "
            "[698-abcd-harmonic](../../698-abcd-harmonic/) (the bare two-leg version, no X "
            "point at all) both reach the same shape of verdict independently.\n\n"
            "*Think the Butterfly needs its full \"confluence zone\" — several overlapping "
            "Fibonacci projections converging near D — to work? Show it beats this study's "
            "placebo and survives the multiple-comparison correction, then we'll talk.*"
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
            "# The Butterfly-Harmonic — a quantitative teardown 🔬\n"
            "### Mechanical XABCD detection · a random-day base rate · a Bonferroni correction "
            "across the basket · a fade timer with costs · a placebo ratio-grid control · a "
            "20-seed synthetic positive control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The claim — **a D point that extends 1.27-1.618x past X marks a reversal** — is "
            "tested on the tightest mechanical reading a proponent would accept, with two "
            "independent controls (a matched-direction base rate and a randomized ratio "
            "placebo) and an explicit multiple-comparison penalty across the basket.\n\n"
            "> ⚠️ **Data note.** Daily OHLC (2001/2010→2026), yfinance, cached; basket "
            "identical to [468-gartley-harmonic](../../468-gartley-harmonic/) and "
            "[698-abcd-harmonic](../../698-abcd-harmonic/). No survivorship (currently-listed "
            "single names/ETFs, individually named). Methods in "
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
            f"| **Signal** | `WEAK` | pooled 5-day D-touch fade **+{R['fib'][5]['mean']:.2f} "
            f"bps** (HAC *t* = **{R['fib'][5]['t']:.2f}** vs 0; Welch *t* = "
            f"**{R['base'][5]['t']:.2f}** vs base rate), but **{R['bonf_survive']}/"
            f"{R['bonf_n_tests']}** Bonferroni-corrected tests survive (critical "
            f"\\|*t*\\| = {R['bonf_thr']:.2f}) and placebo Welch *t* = "
            f"**{R['plac'][5]['t']:.2f}** |\n"
            f"| **Tradability** | `MIRAGE` | {R['n_fib']} events over 21-25 years pooled across "
            "six names; edge confined to a 3-5 day window, gone by day 10 |\n"
            f"| **Beats a placebo?** | `MIXED` | beats on {R['n_beats_plac']}/6 tickers; pooled "
            f"Welch *t* = {R['plac'][5]['t']:.2f} |\n\n"
            "> 💡 In plain words: a number that clears the naive bar alone, and fails every "
            "check we throw at it once we look twice."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Four confirmed swing pivots $X, A, B, C$, with $XA = A - X$, $AB = B - A$, "
            "$BC = C - B$. The Butterfly grid:\n\n"
            "- $|AB / XA| \\approx 0.786$ (the defining first ratio — distinguishes Butterfly "
            "from Gartley's 0.618).\n"
            "- $|BC / AB| \\in [0.382, 0.886]$ (shared structural band across the XABCD zoo).\n"
            "- $D = X - \\text{ext} \\times XA$, $\\text{ext} \\in [1.27, 1.618]$ — **D extends "
            "*past* X**, not toward it (the defining second ratio; every other zoo member keeps "
            "D inside or near the X-A range).\n\n"
            "$\\text{reversal\\_dir} = \\text{sign}(XA)$: a bullish Butterfly (XA up) projects "
            "D below X and expects a bounce up; a bearish one, the mirror.\n\n"
            "**H₁ (reversal).** Fading at D beats zero and a random-day base rate.\n"
            "**H₂ (specificity).** The 0.786/1.27-1.618 grid beats a randomized ratio placebo "
            "on the identical pivots.\n"
            "**H₃ (robustness).** The result survives correction for testing six tickers.\n\n"
            "We find **H₁ marginally supported before correction** (pooled *t* ≈ 2.2-2.3), "
            "**H₂ not supported** (placebo *t* = 0.55), **H₃ not supported** (0/7 survive "
            "Bonferroni)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Butterfly D-touches on the same tape can cluster and overlap, so within-arm means "
            "are tested with a **Newey-West (HAC) t** against zero. The market's positive "
            "unconditional drift means a signal-vs-zero test alone is not enough — the primary "
            "Signal test is a **Welch t vs a random-day base rate** built to match the *same* "
            "bullish/bearish direction mix as the real touches, pooled over 20 seeds (kills the "
            "drift confound). Because the basket is tested seven ways (pooled + 6 tickers, all "
            "5-day), a **Bonferroni correction** (family-wise alpha 0.05) raises the critical "
            "|*t*| from the naive 2.0 to **2.69** before any test is allowed to \"survive.\" A "
            "**second, independent placebo** — randomized retrace/extension targets on the "
            "identical pivots — tests H₂ separately from H₃."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Basket.** {', '.join(R['basket'])}, daily bars, 2001/2010 → {R['asof']} "
            "(last complete month).\n"
            "- **Pivots.** Percentage-threshold zig-zag, **pct = 2%**, confirmed pivots only "
            "(recorded at the confirmation bar, never the earlier extreme — no look-ahead).\n"
            "- **Candidates.** Every consecutive confirmed X-A-B-C quadruple satisfying the "
            "Butterfly grid; D fully computable the moment C confirms.\n"
            "- **Touch scan.** 120-session forward window from C's confirmation for the first "
            "bar bracketing D.\n"
            "- **Execution.** Enter the fade at the touch bar's own close; one round trip = "
            "2 × one-way cost × NAV.\n"
            "- **Signal control.** Random-day base rate, matched direction mix, 20 seeds; "
            "Bonferroni correction across 7 tests.\n"
            "- **Specificity control.** Randomized ratio placebo on identical pivots.\n"
            "- **Faithful-engine control.** Synthetic mean-reverting price index, tunable "
            "reversion knob; the null must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline split and the per-instrument breakdown\n\n"
            "HAC *t* on the pooled fade at three horizons; per-ticker 5-day breakdown."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fib_by_t = {}\n"
            "    frames = []\n"
            "    for t in data.BASKET:\n"
            "        _, _, l = st.detect_and_scan(BARS[t], pct=PCT, cost_bps=0.0)\n"
            "        fib_by_t[t] = l\n"
            "        if len(l):\n"
            "            frames.append(l)\n"
            "    FIB = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()\n"
            "    hs = [1, 5, 10]\n"
            "    means = [st.summarize(FIB, f'ret_gross_{h}')['mean_bps'] for h in hs]\n"
            "    ts = [st.summarize(FIB, f'ret_gross_{h}')['t'] for h in hs]\n"
            "    per_t = {t: st.summarize(fib_by_t[t], 'ret_gross_5') for t in data.BASKET}\n"
            "    tick_means = [per_t[t]['mean_bps'] for t in data.BASKET]\n"
            "else:\n"
            "    hs = [1, 5, 10]\n"
            "    means = [R['fib'][h]['mean'] for h in hs]\n"
            "    ts = [R['fib'][h]['t'] for h in hs]\n"
            "    tick_means = [R['per_instrument'][t]['mean'] for t in R['basket']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 4.4))\n"
            "cols1 = [AMBER if h == 5 else GREY for h in hs]\n"
            "a1.bar([f'{h}d' for h in hs], means, color=cols1, width=.55)\n"
            "for i, (m, t_) in enumerate(zip(means, ts)):\n"
            "    a1.annotate(f'{m:+.0f}bps\\nt={t_:+.2f}', (i, m), ha='center',\n"
            "                va='bottom' if m >= 0 else 'top', fontsize=8.5)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_title('Pooled fade by horizon')\n"
            "a1.set_ylabel('mean return (bps)')\n"
            "cols2 = [GREEN if m > 0 else RED for m in tick_means]\n"
            "a2.bar(list(R['basket']), tick_means, color=cols2, width=.6)\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_title('5-day fade, per ticker')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('by horizon:', {h: round(m,1) for h,m in zip(hs, means)})\n"
            "print('by ticker:', {t: round(m,1) for t,m in zip(R['basket'], tick_means)})"
        ),
        md(
            f"> 💡 In plain words: the 5-day pooled fade (**+{R['fib'][5]['mean']:.2f} bps**, "
            f"HAC *t* = **{R['fib'][5]['t']:.2f}**) is the best of the three horizons; 1-day "
            f"(*t* = {R['fib'][1]['t']:.2f}) and 10-day (*t* = {R['fib'][10]['t']:.2f}) show no "
            "persistence — the \"signal\", if any, lives in a narrow window. The per-ticker "
            "split shows SPY negative and QQQ/TSLA carrying the whole pooled result — exactly "
            "the pattern a thin, over-searched sample produces."
        ),
        md(
            "### 4b · The Signal axis — random-day base rate, Bonferroni-corrected\n\n"
            "20-seed random-day base rate matched on the empirical bullish/bearish mix of the "
            "real touches; Welch *t* vs the Butterfly ledger, pooled and per ticker; Bonferroni "
            "correction across the resulting 7 tests."
        ),
        code(
            "if HAVE_REAL:\n"
            "    base_frames, base_by_t = [], {}\n"
            "    for t in data.BASKET:\n"
            "        l = fib_by_t[t]\n"
            "        if len(l):\n"
            "            mix = float((l['reversal_dir'] > 0).mean())\n"
            "            bl = pd.concat([st.base_rate_ledger(BARS[t], len(l), mix,\n"
            "                             seed=st._seed_from(699, t, s)) for s in range(20)],\n"
            "                            ignore_index=True)\n"
            "            base_by_t[t] = bl\n"
            "            base_frames.append(bl)\n"
            "    BASE = pd.concat(base_frames, ignore_index=True) if base_frames else pd.DataFrame()\n"
            "    tests = [('pooled', st.welch_t(FIB['ret_gross_5'], BASE['ret_gross_5']))]\n"
            "    for t in data.BASKET:\n"
            "        if t in base_by_t and len(fib_by_t[t]):\n"
            "            tests.append((t, st.welch_t(fib_by_t[t]['ret_gross_5'], base_by_t[t]['ret_gross_5'])))\n"
            "        else:\n"
            "            tests.append((t, float('nan')))\n"
            "    thr = st.bonferroni_threshold(0.05, len(tests))\n"
            "else:\n"
            "    tests = [('pooled', R['base'][5]['t'])] + [\n"
            "        (t, R['base_per_instrument'][t]['t']) for t in R['basket']]\n"
            "    thr = R['bonf_thr']\n"
            "labels = [lbl for lbl, _ in tests]\n"
            "vals = [v for _, v in tests]\n"
            "cols = [RED if (np.isfinite(v) and abs(v) >= thr) else\n"
            "        (AMBER if (np.isfinite(v) and abs(v) >= 2.0) else GREY) for v in vals]\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.6))\n"
            "ax.bar(labels, [v if np.isfinite(v) else 0 for v in vals], color=cols, width=.62)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.axhline(thr, ls='--', c=RED, lw=1.2, label=f'Bonferroni critical |t| = {thr:.2f}')\n"
            "ax.axhline(-thr, ls='--', c=RED, lw=1.2)\n"
            "ax.axhline(2.0, ls=':', c=AMBER, lw=1, label='naive |t| = 2')\n"
            "ax.axhline(-2.0, ls=':', c=AMBER, lw=1)\n"
            "ax.set_ylabel('Welch t (Butterfly vs base rate)')\n"
            "ax.set_title('Nothing clears the Bonferroni-corrected bar')\n"
            "ax.legend(fontsize=8.5); plt.tight_layout(); plt.show()\n"
            "print({lbl: round(v,2) if np.isfinite(v) else None for lbl, v in tests})\n"
            "print(f'Bonferroni critical |t| = {thr:.2f}')"
        ),
        md(
            f"> 💡 In plain words: the pooled test (Welch *t* = **{R['base'][5]['t']:.2f}**) and "
            f"two per-ticker splits clear the naive bar of 2 — but the honest bar, once you "
            f"correct for looking at six tickers, is **{R['bonf_thr']:.2f}**, and "
            f"**{R['bonf_survive']}/{R['bonf_n_tests']}** tests clear it. This is the desk's "
            "multiple-comparison rail doing exactly its job."
        ),
        md(
            "### 4c · Tradability — the fade timer, net of costs\n\n"
            "Sweep the holding period and the one-way cost; ~2 pooled events/year across six "
            "names (roughly once every 4-5 years per ticker)."
        ),
        code(
            "hs = sorted(R['timer'])\n"
            "gross = [R['timer'][h]['gross'] for h in hs]\n"
            "net5 = [R['timer'][h]['net5'] for h in hs]\n"
            "net10 = [R['timer'][h]['net10'] for h in hs]\n"
            "x = np.arange(len(hs)); w = .27\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.6))\n"
            "ax.bar(x - w, gross, width=w, label='gross', color=GREY)\n"
            "ax.bar(x, net5, width=w, label='net 5bps', color=AMBER)\n"
            "ax.bar(x + w, net10, width=w, label='net 10bps', color=RED)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('mean return per event (bps)')\n"
            "ax.set_title('The apparent edge lives in a narrow 3-5 day window')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print({h: (R['timer'][h]['gross'], R['timer'][h]['t_gross']) for h in hs})"
        ),
        md(
            "> 💡 In plain words: costs (5-10 bps round trip) barely move these numbers — "
            "20-130 bps events dwarf them. What actually kills Tradability is the *shape*: "
            "positive and briefly *t* ≥ 2 at 3-5 days, negative by 10, flat by 20 — combined "
            "with a firing rate of roughly one completion every 4-5 years per ticker, there is "
            "no horizon at which this is both certified and frequent enough to matter."
        ),
        md(
            "### 4d · The third axis — does the specific ratio grid matter?\n\n"
            "Identical pivot pipeline; each placebo candidate's AB-retrace and D-extension "
            "*targets* are replaced by a deterministic, seeded, off-Butterfly draw."
        ),
        code(
            "if HAVE_REAL:\n"
            "    plac_by_t, plac_frames = {}, []\n"
            "    for t in data.BASKET:\n"
            "        _, _, lp = st.detect_and_scan(BARS[t], pct=PCT, placebo=True, seed=699, cost_bps=0.0)\n"
            "        plac_by_t[t] = lp\n"
            "        if len(lp):\n"
            "            plac_frames.append(lp)\n"
            "    PLAC = pd.concat(plac_frames, ignore_index=True) if plac_frames else pd.DataFrame()\n"
            "    sp = st.summarize(PLAC, 'ret_gross_5')\n"
            "    pm = sp['mean_bps']\n"
            "    wtp = st.welch_t(FIB['ret_gross_5'], PLAC['ret_gross_5'])\n"
            "    beat_labels, beat_vals = [], []\n"
            "    for t in data.BASKET:\n"
            "        fm = st.summarize(fib_by_t[t], 'ret_gross_5')['mean_bps']\n"
            "        pmi = st.summarize(plac_by_t[t], 'ret_gross_5')['mean_bps']\n"
            "        beat_labels.append(t); beat_vals.append(fm - pmi if np.isfinite(fm) and np.isfinite(pmi) else np.nan)\n"
            "else:\n"
            "    pm, wtp = R['plac'][5]['mean'], R['plac'][5]['t']\n"
            "    beat_labels = list(R['basket'])\n"
            "    beat_vals = [R['per_instrument'][t]['mean'] - R['plac_per_instrument'][t]['mean']\n"
            "                 for t in R['basket']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.0, 4.4))\n"
            "a1.bar(['Butterfly', 'placebo'], [R['fib'][5]['mean'], pm], color=[AMBER, GREY], width=.55)\n"
            "for i, v in enumerate([R['fib'][5]['mean'], pm]):\n"
            "    a1.annotate(f'{v:+.0f}bps', (i, v), ha='center', va='bottom' if v>=0 else 'top')\n"
            "a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_title(f'Pooled: Welch t = {wtp:+.2f} (not significant)')\n"
            "cols = [GREEN if (np.isfinite(v) and v > 0) else RED for v in beat_vals]\n"
            "a2.bar(beat_labels, [v if np.isfinite(v) else 0 for v in beat_vals], color=cols, width=.6)\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_title('Butterfly minus placebo, per ticker')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Butterfly {R[\"fib\"][5][\"mean\"]:+.1f} bps vs placebo {pm:+.1f} bps  Welch t={wtp:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: Butterfly beats its own placebo on "
            f"**{R['n_beats_plac']}/6** tickers and at the pooled short-horizon means, but the "
            f"pooled Welch *t* (**{R['plac'][5]['t']:.2f}**) is nowhere near significant. The "
            "0.786/1.27-1.618 grid is not demonstrably better than an arbitrary equal-legged "
            "extension built from the *same* pivots — H₂ not supported."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic mean-reverting price index (tunable `mean_rev` knob toward a slow EMA), "
            "pooled across a synthetic 6-name basket per seed. The null (`mean_rev = 0`) is "
            "checked over **20 seeds** — never a single stream."
        ),
        code(
            "def pooled_synth(mean_rev, seed, n_names=6):\n"
            "    frames = []\n"
            "    for i in range(n_names):\n"
            "        sbars = data.synthetic_world(mean_rev=mean_rev, seed=seed*1000+i, n_days=6300)\n"
            "        _, _, ledger = st.detect_and_scan(sbars, pct=PCT, cost_bps=0.0)\n"
            "        if not ledger.empty:\n"
            "            frames.append(ledger)\n"
            "    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()\n"
            "\n"
            "null_ts = np.array([st.summarize(pooled_synth(0.0, 699 + s), 'ret_gross_5')['t']\n"
            "                     for s in range(20)], dtype=float)\n"
            "planted = pooled_synth(0.08, 699)\n"
            "planted_t = st.summarize(planted, 'ret_gross_5')['t']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (mean_rev=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=RED, s=90, zorder=5, label='planted mean_rev=0.08')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('HAC t (fade vs 0)')\n"
            "ax.set_title('Control: the null fires near its nominal rate; a planted effect lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {np.nanmean(null_ts):+.2f} (sd {np.nanstd(null_ts, ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(np.abs(null_ts) >= 2).sum()}/20 seeds  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector fires \\|*t*\\| ≥ 2 in "
            f"**{R['syn_null_fire']}/20** seeds — close to the nominal ~5% false-positive rate "
            f"— and a planted mean-reversion tendency lights up sharply "
            f"(*t* = **{R['syn_planted_t']:.2f}**). The detection + inference pipeline is "
            "unbiased; the real-tape result's fragility is genuine, not a bug in the machinery."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — pooled 5-day fade **+{R['fib'][5]['mean']:.2f} bps** "
            f"(HAC *t* = **{R['fib'][5]['t']:.2f}**; Welch *t* vs base rate = "
            f"**{R['base'][5]['t']:.2f}**) on **{R['n_fib']}** touches over 21-25 years; "
            f"**{R['bonf_survive']}/{R['bonf_n_tests']}** Bonferroni-corrected tests survive "
            f"(critical \\|*t*\\| = {R['bonf_thr']:.2f}); 1-/10-day horizons show no "
            "persistence; SPY point-estimates negative.\n"
            f"- **Tradability `MIRAGE`** — ~2 pooled events/year across six names (roughly one "
            "per ticker every 4-5 years); apparent edge confined to a 3-5 day window, gone by "
            "day 10; nothing survives correction.\n"
            f"- **\"Beats a random ratio grid?\" `MIXED`** — ahead on {R['n_beats_plac']}/6 "
            f"tickers and the pooled short-horizon means, but pooled Welch *t* = "
            f"**{R['plac'][5]['t']:.2f}** — not statistically demonstrable."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general lesson is multiple comparisons.** A single \"significant\" ticker "
            "out of six tested is exactly what a Bonferroni correction exists to catch — this "
            "study makes that correction explicit rather than quietly reporting the best-of-six.\n"
            "- **Where a believer would push back:** Carney's full method requires a "
            "*confluence zone* (several overlapping Fibonacci projections converging near D), "
            "not the bare two-ratio skeleton tested here. That's a legitimate next experiment "
            "— but it adds researcher degrees of freedom (which projections? how many must "
            "converge? how tight a zone?) that would need their own placebo before counting as "
            "evidence.\n"
            "- **Dedup map:** [468-gartley-harmonic](../../468-gartley-harmonic/) (D stays "
            "*inside* the X-A range, different B ratio), "
            "[698-abcd-harmonic](../../698-abcd-harmonic/) (no X point, no confluence at all), "
            "[700-bat-harmonic](../../700-bat-harmonic/) and "
            "[701-crab-harmonic](../../701-crab-harmonic/) (different ratio grids on the same "
            "XABCD skeleton) — all independently converging on the same verdict shape.\n\n"
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
