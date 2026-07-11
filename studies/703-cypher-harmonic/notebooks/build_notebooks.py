"""Generate the two narrative notebooks for Study 703 (Cypher-Harmonic).

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
# pct=0.02 zigzag, AB retrace 0.382-0.618 of XA, C overshoots A by 1.13-1.414 of XA,
# D = C - 0.786*(C-X), 120-session touch window).
R = dict(
    basket=("SPY", "QQQ", "AAPL", "MSFT", "TSLA", "NVDA"),
    asof="2026-06-30",
    n_pivots=6055, n_cand_cyp=153, n_cyp=127,
    n_cand_plac=153, n_plac=134,
    cyp={1: dict(hit=57.5, mean=22.89, t=1.08), 5: dict(hit=50.4, mean=45.77, t=0.84),
         10: dict(hit=52.0, mean=27.69, t=0.34)},
    wilson_cyp=(41.8, 58.9),
    per_instrument={
        "SPY":  dict(n=21, mean=-59.62, t=-1.19),
        "QQQ":  dict(n=18, mean=60.34,  t=0.79),
        "AAPL": dict(n=16, mean=128.81, t=1.33),
        "MSFT": dict(n=15, mean=28.98,  t=0.24),
        "TSLA": dict(n=26, mean=77.68,  t=0.64),
        "NVDA": dict(n=31, mean=47.21,  t=0.24),
    },
    # Signal axis — vs random-day base rate (matched direction mix, 20 seeds)
    base={1: dict(mean=-4.78, t=0.99), 5: dict(mean=-2.83, t=0.81), 10: dict(mean=-8.35, t=0.43)},
    n_base=2540,
    base_per_instrument={
        "SPY":  dict(base_mean=-15.57, t=-1.06),
        "QQQ":  dict(base_mean=19.17,  t=0.51),
        "AAPL": dict(base_mean=-29.94, t=1.34),
        "MSFT": dict(base_mean=-18.22, t=0.39),
        "TSLA": dict(base_mean=-42.76, t=0.79),
        "NVDA": dict(base_mean=47.93,  t=-0.00),
    },
    bonf_n_tests=7, bonf_thr=2.69, bonf_uncorrected=0, bonf_survive=0,
    # Tradability axis — the fade timer, net of costs
    timer={
        1:  dict(gross=22.89,  t_gross=1.08,  net5=12.89,  t5=0.61,  net10=2.89,   t10=0.14),
        3:  dict(gross=27.93,  t_gross=0.72,  net5=17.93,  t5=0.47,  net10=7.93,   t10=0.21),
        5:  dict(gross=45.77,  t_gross=0.84,  net5=35.77,  t5=0.66,  net10=25.77,  t10=0.47),
        10: dict(gross=27.69,  t_gross=0.34,  net5=17.69,  t5=0.22,  net10=7.69,   t10=0.09),
        20: dict(gross=-12.92, t_gross=-0.12, net5=-22.92, t5=-0.21, net10=-32.92, t10=-0.31),
    },
    # Third axis — vs placebo
    plac={1: dict(mean=1.82, t=0.58), 5: dict(mean=86.98, t=-0.52), 10: dict(mean=91.39, t=-0.54)},
    plac_per_instrument={
        "SPY":  dict(n=23, mean=29.16,  beats=False),
        "QQQ":  dict(n=21, mean=101.24, beats=False),
        "AAPL": dict(n=16, mean=-44.18, beats=True),
        "MSFT": dict(n=15, mean=17.81,  beats=True),
        "TSLA": dict(n=25, mean=12.86,  beats=True),
        "NVDA": dict(n=34, mean=264.03, beats=False),
    },
    n_beats_plac=3,
    # synthetic control
    syn_null_mean=0.13, syn_null_sd=0.86, syn_null_fire=0, syn_null_seeds=20,
    syn_planted_n=216, syn_planted_mean=99.57, syn_planted_t=5.91,
    fp={"SPY": "4ae3283c45e5", "QQQ": "f0c763128acf", "AAPL": "c5c7b7855e9b",
        "MSFT": "c7a24ba85055", "TSLA": "f8ca92e420b8", "NVDA": "fd06322c7266"},
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Beats_a_placebo%3F: Busted](https://img.shields.io/badge/Beats_a_placebo%3F-Busted-8b949e?style=flat-square)\n\n"
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

from cypher_harmonic import data, strategy as st

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
            "# The pattern that measures itself off its own overshoot 🔑\n"
            "### The Cypher harmonic — five dots, a 78.6% retracement of an "
            "*extended* leg, and a stopwatch\n\n"
            + BADGES +
            "Of all the \"harmonic patterns\" retail charting software draws for "
            "you, the **Cypher** is the odd one out: instead of projecting its "
            "reversal point D from the *original* X-A swing (like the Gartley, "
            "Bat, Butterfly and Crab all do), it lets point C **overshoot** the "
            "pattern's own high or low, then measures D as a **78.6% retracement "
            "of that freshly-extended leg**. Proponents pitch it as a "
            "higher-win-rate filter precisely because that extra overshoot step "
            "supposedly screens out weaker setups.\n\n"
            "So we built a robot that finds every Cypher on six liquid tapes over "
            "the last 21-25 years — no eyeballing, no cherry-picking — and asked "
            "whether fading price right where the pattern says to would actually "
            "have made money.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the Bonferroni "
            "correction and the placebo math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** Pivots are a confirmed percentage zig-zag (known "
            "only after price reverses enough to lock them in — no look-ahead). "
            "Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does price turn where the Cypher says it should? | **Barely, and "
            f"not provably.** Over a 5-day hold the pattern averages "
            f"**{R['cyp'][5]['mean']:+.0f} bps** — a small positive number with a "
            "hit rate landing *exactly* on a coin flip (50.4%), and the *t*-stat "
            f"(**{R['cyp'][5]['t']:.2f}**) is nowhere near the bar for calling it "
            "real. |\n"
            f"| Does it beat what a random trade would have earned? | **Not "
            "provably.** Against a random-day baseline matched to the same "
            f"bullish/bearish mix, the gap is Welch *t* = "
            f"**{R['base'][5]['t']:.2f}** — comfortably inside noise. |\n"
            "| Does the specific 78.6%-of-XC ratio matter? | **No detectable "
            "sign of it.** Swap that number for a random retracement on the "
            "*identical* pivots and, at the pooled level, the random version "
            f"actually does *better* (Welch *t* = {R['plac'][5]['t']:.2f}). |\n"
            "| Can you trade it? | **No.** Every hold length is weak before "
            "costs and turns net-negative by 20 days. |\n\n"
            "> A pattern marketed as a sharper filter than its harmonic cousins "
            "turns out, on the real tape, to be statistically indistinguishable "
            "from a coin flip that occasionally loses to its own placebo."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Find four swing points X-A-B-C. If B retraces **38-62%** of the "
            "X-A move, let C **overshoot the original A swing** by "
            "**113-141%** of X-A — further than a normal pullback would ever go "
            "— then project a fifth point D that retraces **78.6%** of that "
            "*new, extended* X-C leg. The overshoot-then-retrace shape marks an "
            "unusually precise reversal zone: tighter and more selective than "
            "the Gartley or Bat.\"*\n\n"
            "That's the Cypher, generally credited to trader Darren Oglesbee and "
            "absorbed into Scott Carney's broader \"harmonic trading\" system "
            "alongside the Gartley, Bat, Butterfly and Crab. Its selling point is "
            "specificity: while every sibling pattern measures D off the "
            "*original* X-A leg, the Cypher's D references the *C point itself* "
            "— a level that doesn't even exist until the pattern has already "
            "overshot. It's baked into TradingView, MetaTrader and Thinkorswim's "
            "auto-scanners as one of the named \"harmonic\" shapes."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a pattern this specific — a two-step overshoot-then-retrace "
            "chained across four historical swings — genuinely forecasts a turn, "
            "and does so more reliably than its simpler cousins, that's a "
            "striking, calculator-testable crack in market efficiency. And "
            "harmonic trading is popular *precisely* because the charts look "
            "uncanny after the fact: hindsight makes any completed pattern look "
            "inevitable.\n\n"
            "The trap is exactly that hindsight. A trader hand-labelling swings "
            "on a chart, with the benefit of already knowing what happened next, "
            "will always find the Cypher that worked. So we removed the human: a "
            "mechanical rule finds every qualifying X-A-B-C on the tape, in real "
            "time, with no knowledge of the future — then we measure what "
            "actually happened at D."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The basket.** {', '.join(R['basket'])} — six liquid names, "
            f"daily bars, 2001/2010 → {R['asof']}.\n"
            "- **The rule, mechanically.** A confirmed zig-zag finds swing "
            "pivots (only known once price has reversed enough to lock them "
            "in). Every qualifying X-A-B-C (with C overshooting A) projects a D "
            "level; we scan forward up to 120 sessions for the first touch.\n"
            "- **The trade.** Fade at the touch — bet on the reversal the "
            "pattern predicts — and measure the forward return.\n"
            "- **Two honest baselines, announced up front.** (1) A **random-day "
            "base rate**: same ticker, same number of trades, same "
            "bullish/bearish mix, just on random days — kills the \"stocks "
            "drift up anyway\" confound. (2) A **placebo retracement zone**: "
            "the identical pivots, but with the 78.6% number replaced by a "
            "random one — kills the \"any retracement of the extended leg "
            "would work\" confound. And because we test six tickers, we "
            "**correct for that** (Bonferroni) before calling anything real."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Fade the D-touch, hold 5 days, average "
            "across every qualifying Cypher on the basket."
        ),
        code(
            "if HAVE_REAL:\n"
            "    frames = []\n"
            "    for t in data.BASKET:\n"
            "        _, _, ledger = st.detect_and_scan(BARS[t], pct=PCT, cost_bps=0.0)\n"
            "        if not ledger.empty:\n"
            "            frames.append(ledger)\n"
            "    CYP = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()\n"
            "    s5 = st.summarize(CYP, 'ret_gross_5')\n"
            "    n5, m5, t5 = s5['n'], s5['mean_bps'], s5['t']\n"
            "else:\n"
            "    n5, m5, t5 = R['n_cyp'], R['cyp'][5]['mean'], R['cyp'][5]['t']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['Cypher D-touch\\n(5-day fade)'], [m5], color=AMBER, width=.5)\n"
            "ax.annotate(f'{m5:+.0f} bps\\n(n={n5}, t={t5:+.2f})', (0, m5), ha='center',\n"
            "            va='bottom' if m5 >= 0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean forward return (bps)')\n"
            "ax.set_title('The headline number — positive, but a long way from proof')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'n={n5}  mean={m5:+.2f}bps  t={t5:+.2f}')"
        ),
        md(
            f"**{R['cyp'][5]['mean']:+.0f} bps** on average, *t* = "
            f"**{R['cyp'][5]['t']:.2f}** — positive, but the bar for calling "
            "anything real on this desk is *t* ≥ 2, and this is less than half "
            "that. Worse, the hit rate is **50.4%** — a coin flip, dead center.\n\n"
            "**Second, the honest baseline.** Stocks drift up. A random buy on a "
            "random day, matched to the same bullish/bearish mix as our Cypher "
            "trades, earns *something* just from that drift (or loses, if the "
            "mix leans bearish). Does the Cypher actually add anything on top?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cyp_by_t, base_frames = {}, []\n"
            "    for t in data.BASKET:\n"
            "        _, _, l = st.detect_and_scan(BARS[t], pct=PCT, cost_bps=0.0)\n"
            "        cyp_by_t[t] = l\n"
            "        if len(l):\n"
            "            mix = float((l['reversal_dir'] > 0).mean())\n"
            "            for s in range(20):\n"
            "                base_frames.append(st.base_rate_ledger(BARS[t], len(l), mix,\n"
            "                    seed=st._seed_from(703, t, s)))\n"
            "    BASE = pd.concat(base_frames, ignore_index=True) if base_frames else pd.DataFrame()\n"
            "    sb = st.summarize(BASE, 'ret_gross_5')\n"
            "    bm, wt = sb['mean_bps'], st.welch_t(CYP['ret_gross_5'], BASE['ret_gross_5'])\n"
            "else:\n"
            "    bm, wt = R['base'][5]['mean'], R['base'][5]['t']\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.4))\n"
            "ax.bar(['Cypher D-touch', 'random-day\\nbase rate'], [m5, bm], color=[AMBER, GREY], width=.55)\n"
            "for i, v in enumerate([m5, bm]):\n"
            "    ax.annotate(f'{v:+.0f} bps', (i, v), ha='center', va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean 5-day return (bps)')\n"
            "ax.set_title(f'Beats a flat random-day baseline, but not provably (Welch t = {wt:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Cypher {m5:+.1f} bps vs base rate {bm:+.1f} bps  Welch t={wt:+.2f}')"
        ),
        md(
            f"The Cypher does edge out the random-day baseline in raw terms, but "
            f"the gap (Welch *t* = **{R['base'][5]['t']:.2f}**) is comfortably "
            "inside what luck alone could produce. Look at the sample: "
            f"**{R['n_cyp']} touches, spread across six stocks, over 21-25 "
            "years** — and we tested **six different tickers**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    per_t = {t: st.summarize(cyp_by_t[t], 'ret_gross_5') for t in data.BASKET}\n"
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
            "ax.set_title('Five of six positive — but none individually significant')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({t: round(m,1) for t,m in zip(tickers, means)})"
        ),
        md(
            "Five of six tickers land positive, but **not one** clears even the "
            "naive *t* ≥ 2 bar individually — this is a diffuse, low-conviction "
            "average, not a shared mechanism. Correcting properly for having "
            f"tested six tickers, **{R['bonf_survive']} of {R['bonf_n_tests']}** "
            "results survive.\n\n"
            "**Finally, the placebo.** Does the *specific* 78.6% retracement of "
            "XC matter, or would any similarly-shaped retracement have done "
            "just as well?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    plac_frames = []\n"
            "    for t in data.BASKET:\n"
            "        _, _, lp = st.detect_and_scan(BARS[t], pct=PCT, placebo=True, seed=703, cost_bps=0.0)\n"
            "        if not lp.empty:\n"
            "            plac_frames.append(lp)\n"
            "    PLAC = pd.concat(plac_frames, ignore_index=True) if plac_frames else pd.DataFrame()\n"
            "    sp = st.summarize(PLAC, 'ret_gross_5')\n"
            "    pm, wtp = sp['mean_bps'], st.welch_t(CYP['ret_gross_5'], PLAC['ret_gross_5'])\n"
            "else:\n"
            "    pm, wtp = R['plac'][5]['mean'], R['plac'][5]['t']\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.4))\n"
            "ax.bar(['Cypher\\n(0.786 of XC)', 'placebo\\n(random retracement)'], [m5, pm],\n"
            "       color=[AMBER, GREY], width=.55)\n"
            "for i, v in enumerate([m5, pm]):\n"
            "    ax.annotate(f'{v:+.0f} bps', (i, v), ha='center', va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean 5-day return (bps)')\n"
            "ax.set_title(f'The placebo actually does BETTER (Welch t = {wtp:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Cypher {m5:+.1f} bps vs placebo {pm:+.1f} bps  Welch t={wtp:+.2f}')"
        ),
        md(
            f"The placebo — an *arbitrary* retracement of the same extended XC "
            f"leg — actually earns **more** on average (**{R['plac'][5]['mean']:+.0f} "
            f"bps** vs the Cypher's own **{R['cyp'][5]['mean']:+.0f} bps**), though "
            f"the gap (Welch *t* = **{R['plac'][5]['t']:.2f}**) is itself not "
            "statistically real. The specific number — 78.6% — is not doing "
            "detectably better than an arbitrary retracement target would."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The fade is small and positive "
            f"(*t* = {R['cyp'][5]['t']:.2f}), a coin-flip hit rate (50.4%), "
            f"doesn't beat a matched-direction random day provably "
            f"(*t* = {R['base'][5]['t']:.2f}), and fails a six-ticker correction "
            f"({R['bonf_survive']}/{R['bonf_n_tests']} survive).\n"
            "- **Tradability — Mirage.** Every hold length is weak before "
            "costs and turns net-negative by 20 days.\n"
            "- **\"Is 78.6% of XC the pattern's distinctive signature?\" — "
            "Busted.** Swap it for a random retracement on the identical "
            f"pivots and the placebo actually does *better* "
            f"(Welch *t* = {R['plac'][5]['t']:.2f})."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The general lesson: an extra construction step isn't the same "
            "as an extra edge.** The Cypher's whole pitch is that overshooting A "
            "and then referencing the *new* XC leg makes it a sharper filter "
            "than its siblings — on this basket, that extra complexity buys "
            "nothing measurable over a coin flip or a randomized control.\n"
            "- **Sibling studies:** [468-gartley-harmonic](../../468-gartley-harmonic/) "
            "(the same 78.6% *number*, but measured off XA, not XC), "
            "[699-butterfly-harmonic](../../699-butterfly-harmonic/), "
            "[700-bat-harmonic](../../700-bat-harmonic/) and "
            "[701-crab-harmonic](../../701-crab-harmonic/) all reach the same "
            "shape of verdict independently.\n\n"
            "*Think the Cypher needs its full \"confluence zone\" — several "
            "overlapping Fibonacci projections converging near D — to work? "
            "Show it beats this study's placebo and survives the "
            "multiple-comparison correction, then we'll talk.*"
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
            "# The Cypher-Harmonic — a quantitative teardown 🔬\n"
            "### Mechanical XABC detection · a random-day base rate · a "
            "Bonferroni correction across the basket · a fade timer with costs "
            "· a placebo retracement-zone control · a 20-seed synthetic "
            "positive control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The claim — **a C point that overshoots the original A swing, "
            "followed by a D point that retraces 78.6% of the resulting XC leg**, "
            "is the harmonic zoo's uniquely two-step, XC-referenced construction "
            "— is tested on the tightest mechanical reading a proponent would "
            "accept, with two independent controls (a matched-direction base "
            "rate and a randomized retracement placebo) and an explicit "
            "multiple-comparison penalty across the basket.\n\n"
            "> ⚠️ **Data note.** Daily OHLC (2001/2010→2026), yfinance, cached; "
            "basket identical to [468-gartley-harmonic](../../468-gartley-harmonic/) "
            "and [701-crab-harmonic](../../701-crab-harmonic/). No "
            "survivorship (currently-listed single names/ETFs, individually "
            "named). Methods in [`docs/references.md`](../docs/references.md), "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to "
            "intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | pooled 5-day D-touch fade **+{R['cyp'][5]['mean']:.2f} "
            f"bps** (HAC *t* = **{R['cyp'][5]['t']:.2f}** vs 0; Welch *t* = "
            f"**{R['base'][5]['t']:.2f}** vs base rate) — never close to "
            f"significant; **{R['bonf_survive']}/{R['bonf_n_tests']}** "
            "Bonferroni-corrected tests survive (critical \\|*t*\\| = "
            f"{R['bonf_thr']:.2f}) and placebo Welch *t* = "
            f"**{R['plac'][5]['t']:.2f}** (the placebo *outperforms*) |\n"
            f"| **Tradability** | `MIRAGE` | {R['n_cyp']} events over 21-25 years "
            "pooled across six names; weak before costs at every horizon, "
            f"net-negative by 20 days ({R['timer'][20]['gross']:+.0f} bps gross, "
            f"*t*={R['timer'][20]['t_gross']:+.2f}) |\n"
            f"| **Beats a placebo?** | `BUSTED` | beats on {R['n_beats_plac']}/6 "
            f"tickers; pooled Welch *t* = {R['plac'][5]['t']:.2f} (placebo wins "
            "at the pooled level) |\n\n"
            "> 💡 In plain words: a headline number that *looks* promising at "
            "first glance (positive, five of six tickers agree) dissolves the "
            "moment you ask for a *t*-stat, a Bonferroni correction, or a "
            "placebo comparison."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Four confirmed swing pivots $X, A, B, C$, with $XA = A - X$, "
            "$AB = B - A$, $XC = C - X$. The Cypher grid:\n\n"
            "- $|AB / XA| \\in [0.382, 0.618]$ (the same shallow band as the "
            "Crab).\n"
            "- $XC / XA \\in [1.13, 1.414]$, **same sign as $XA$** — C must "
            "*overshoot* the original A swing, in the same direction, unlike "
            "every sibling where C sits between B and A.\n"
            "- $D = C - 0.786 \\times XC$ — **D retraces the freshly-extended "
            "XC leg**, not XA and not AB — the only D formula in the zoo "
            "referenced off a leg that includes the overshoot itself. This is "
            "the defining ratio and the one under test.\n\n"
            "$\\text{reversal\\_dir} = \\text{sign}(XA)$: a bullish Cypher (XA "
            "up) projects D back inside the X-A range and expects a bounce up; "
            "a bearish one, the mirror.\n\n"
            "**H₁ (reversal).** Fading at D beats zero and a random-day base "
            "rate.\n"
            "**H₂ (specificity).** The 0.786-of-XC target beats a randomized "
            "retracement placebo on the identical pivots.\n"
            "**H₃ (robustness).** The result survives correction for testing "
            "six tickers.\n\n"
            "We find **H₁ not supported** (positive point estimate, *t* well "
            "under 2 at every horizon), **H₂ not supported** (placebo *t* = "
            f"{R['plac'][5]['t']:.2f} — the placebo *wins* at the pooled level), "
            "**H₃ moot** (0/7 tests clear even the uncorrected bar, so nothing "
            "survives correction either way)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Cypher D-touches on the same tape can cluster and overlap, so "
            "within-arm means are tested with a **Newey-West (HAC) t** against "
            "zero. The market's positive unconditional drift means a "
            "signal-vs-zero test alone is not enough — the primary Signal test "
            "is a **Welch t vs a random-day base rate** built to match the "
            "*same* bullish/bearish direction mix as the real touches, pooled "
            "over 20 seeds (kills the drift confound). Because the basket is "
            "tested seven ways (pooled + 6 tickers, all 5-day), a **Bonferroni "
            "correction** (family-wise alpha 0.05) raises the critical |*t*| "
            "from the naive 2.0 to **2.69** before any test is allowed to "
            "\"survive.\" A **second, independent placebo** — randomized "
            "retracement targets on the identical pivots — tests H₂ separately "
            "from H₃."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Basket.** {', '.join(R['basket'])}, daily bars, 2001/2010 → "
            f"{R['asof']} (last complete month).\n"
            "- **Pivots.** Percentage-threshold zig-zag, **pct = 2%**, "
            "confirmed pivots only (recorded at the confirmation bar, never "
            "the earlier extreme — no look-ahead).\n"
            "- **Candidates.** Every consecutive confirmed X-A-B-C quadruple "
            "satisfying the Cypher structural bands (AB retrace, XC overshoot); "
            "D fully computable the moment C confirms.\n"
            "- **Touch scan.** 120-session forward window from C's "
            "confirmation for the first bar bracketing D.\n"
            "- **Execution.** Enter the fade at the touch bar's own close; one "
            "round trip = 2 × one-way cost × NAV.\n"
            "- **Signal control.** Random-day base rate, matched direction "
            "mix, 20 seeds; Bonferroni correction across 7 tests.\n"
            "- **Specificity control.** Randomized retracement-target placebo "
            "on identical pivots.\n"
            "- **Faithful-engine control.** Synthetic mean-reverting price "
            "index, tunable reversion knob; the null must not fire across 20 "
            "seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline split and the per-instrument breakdown\n\n"
            "HAC *t* on the pooled fade at three horizons; per-ticker 5-day "
            "breakdown."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cyp_by_t = {}\n"
            "    frames = []\n"
            "    for t in data.BASKET:\n"
            "        _, _, l = st.detect_and_scan(BARS[t], pct=PCT, cost_bps=0.0)\n"
            "        cyp_by_t[t] = l\n"
            "        if len(l):\n"
            "            frames.append(l)\n"
            "    CYP = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()\n"
            "    hs = [1, 5, 10]\n"
            "    means = [st.summarize(CYP, f'ret_gross_{h}')['mean_bps'] for h in hs]\n"
            "    ts = [st.summarize(CYP, f'ret_gross_{h}')['t'] for h in hs]\n"
            "    per_t = {t: st.summarize(cyp_by_t[t], 'ret_gross_5') for t in data.BASKET}\n"
            "    tick_means = [per_t[t]['mean_bps'] for t in data.BASKET]\n"
            "else:\n"
            "    hs = [1, 5, 10]\n"
            "    means = [R['cyp'][h]['mean'] for h in hs]\n"
            "    ts = [R['cyp'][h]['t'] for h in hs]\n"
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
            f"> 💡 In plain words: the 5-day pooled fade (**+{R['cyp'][5]['mean']:.2f} "
            f"bps**, HAC *t* = **{R['cyp'][5]['t']:.2f}**) is positive but not "
            f"even close to significant, and 10-day (*t* = {R['cyp'][10]['t']:.2f}) "
            "fades further rather than strengthening — the opposite of what a "
            "real, slow-building reversal would look like. The per-ticker split "
            "shows five of six tickers positive, but none individually clears "
            "the naive *t* ≥ 2 bar — a diffuse average, not a shared mechanism."
        ),
        md(
            "### 4b · The Signal axis — random-day base rate, Bonferroni-corrected\n\n"
            "20-seed random-day base rate matched on the empirical "
            "bullish/bearish mix of the real touches; Welch *t* vs the Cypher "
            "ledger, pooled and per ticker; Bonferroni correction across the "
            "resulting 7 tests."
        ),
        code(
            "if HAVE_REAL:\n"
            "    base_frames, base_by_t = [], {}\n"
            "    for t in data.BASKET:\n"
            "        l = cyp_by_t[t]\n"
            "        if len(l):\n"
            "            mix = float((l['reversal_dir'] > 0).mean())\n"
            "            bl = pd.concat([st.base_rate_ledger(BARS[t], len(l), mix,\n"
            "                             seed=st._seed_from(703, t, s)) for s in range(20)],\n"
            "                            ignore_index=True)\n"
            "            base_by_t[t] = bl\n"
            "            base_frames.append(bl)\n"
            "    BASE = pd.concat(base_frames, ignore_index=True) if base_frames else pd.DataFrame()\n"
            "    tests = [('pooled', st.welch_t(CYP['ret_gross_5'], BASE['ret_gross_5']))]\n"
            "    for t in data.BASKET:\n"
            "        if t in base_by_t and len(cyp_by_t[t]):\n"
            "            tests.append((t, st.welch_t(cyp_by_t[t]['ret_gross_5'], base_by_t[t]['ret_gross_5'])))\n"
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
            "ax.set_ylabel('Welch t (Cypher vs base rate)')\n"
            "ax.set_title('Nothing clears even the naive bar, let alone the Bonferroni one')\n"
            "ax.legend(fontsize=8.5); plt.tight_layout(); plt.show()\n"
            "print({lbl: round(v,2) if np.isfinite(v) else None for lbl, v in tests})\n"
            "print(f'Bonferroni critical |t| = {thr:.2f}')"
        ),
        md(
            "> 💡 In plain words: the best individual reading (AAPL, Welch *t* "
            f"= **{R['base_per_instrument']['AAPL']['t']:.2f}**) doesn't even "
            "clear the naive bar of 2. The honest bar, once you correct for "
            f"looking at six tickers, is **{R['bonf_thr']:.2f}**, and "
            f"**{R['bonf_survive']}/{R['bonf_n_tests']}** tests clear it. This "
            "is the desk's multiple-comparison rail confirming there was never "
            "much here to begin with."
        ),
        md(
            "### 4c · Tradability — the fade timer, net of costs\n\n"
            "Sweep the holding period and the one-way cost; roughly five pooled "
            "events/year across six names."
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
            "ax.set_title('Weak everywhere, and negative by 20 days')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print({h: (R['timer'][h]['gross'], R['timer'][h]['t_gross']) for h in hs})"
        ),
        md(
            "> 💡 In plain words: costs (5-10 bps round trip) are not what "
            "kills this trade — the *level* of the edge does, before costs are "
            "even applied. The gross number never approaches significance at "
            "any horizon, and by 20 days the sign flips negative on its own. "
            "Combined with the Bonferroni result above and a firing rate of "
            "roughly five completions a year pooled across six names, there is "
            "no regime here worth deploying capital against."
        ),
        md(
            "### 4d · The third axis — does the specific ratio matter?\n\n"
            "Identical pivot pipeline; each placebo candidate's D-retracement "
            "*target* is replaced by a deterministic, seeded, off-Cypher draw."
        ),
        code(
            "if HAVE_REAL:\n"
            "    plac_by_t, plac_frames = {}, []\n"
            "    for t in data.BASKET:\n"
            "        _, _, lp = st.detect_and_scan(BARS[t], pct=PCT, placebo=True, seed=703, cost_bps=0.0)\n"
            "        plac_by_t[t] = lp\n"
            "        if len(lp):\n"
            "            plac_frames.append(lp)\n"
            "    PLAC = pd.concat(plac_frames, ignore_index=True) if plac_frames else pd.DataFrame()\n"
            "    sp = st.summarize(PLAC, 'ret_gross_5')\n"
            "    pm = sp['mean_bps']\n"
            "    wtp = st.welch_t(CYP['ret_gross_5'], PLAC['ret_gross_5'])\n"
            "    beat_labels, beat_vals = [], []\n"
            "    for t in data.BASKET:\n"
            "        fm = st.summarize(cyp_by_t[t], 'ret_gross_5')['mean_bps']\n"
            "        pmi = st.summarize(plac_by_t[t], 'ret_gross_5')['mean_bps']\n"
            "        beat_labels.append(t); beat_vals.append(fm - pmi if np.isfinite(fm) and np.isfinite(pmi) else np.nan)\n"
            "else:\n"
            "    pm, wtp = R['plac'][5]['mean'], R['plac'][5]['t']\n"
            "    beat_labels = list(R['basket'])\n"
            "    beat_vals = [R['per_instrument'][t]['mean'] - R['plac_per_instrument'][t]['mean']\n"
            "                 for t in R['basket']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.0, 4.4))\n"
            "a1.bar(['Cypher', 'placebo'], [R['cyp'][5]['mean'], pm], color=[AMBER, GREY], width=.55)\n"
            "for i, v in enumerate([R['cyp'][5]['mean'], pm]):\n"
            "    a1.annotate(f'{v:+.0f}bps', (i, v), ha='center', va='bottom' if v>=0 else 'top')\n"
            "a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_title(f'Pooled: Welch t = {wtp:+.2f} (placebo ahead, not significant)')\n"
            "cols = [GREEN if (np.isfinite(v) and v > 0) else RED for v in beat_vals]\n"
            "a2.bar(beat_labels, [v if np.isfinite(v) else 0 for v in beat_vals], color=cols, width=.6)\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_title('Cypher minus placebo, per ticker')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Cypher {R[\"cyp\"][5][\"mean\"]:+.1f} bps vs placebo {pm:+.1f} bps  Welch t={wtp:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the Cypher beats its own placebo on only "
            f"**{R['n_beats_plac']}/6** tickers — a coin flip — and at the "
            f"pooled level the placebo actually earns *more* "
            f"(**+{R['plac'][5]['mean']:.0f} bps** vs the Cypher's own "
            f"**+{R['cyp'][5]['mean']:.0f} bps**), though that gap "
            f"(*t* = **{R['plac'][5]['t']:.2f}**) is itself noise. The exact "
            "78.6%-of-XC ratio the pattern is built on is not demonstrably "
            "better than an arbitrary retracement zone drawn from the *same* "
            "pivots — H₂ not supported."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic mean-reverting price index (tunable `mean_rev` knob "
            "toward a slow EMA), pooled across a synthetic 6-name basket per "
            "seed. The null (`mean_rev = 0`) is checked over **20 seeds** — "
            "never a single stream."
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
            "null_ts = np.array([st.summarize(pooled_synth(0.0, 703 + s), 'ret_gross_5')['t']\n"
            "                     for s in range(20)], dtype=float)\n"
            "planted = pooled_synth(0.12, 703)\n"
            "planted_t = st.summarize(planted, 'ret_gross_5')['t']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (mean_rev=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=GREEN, s=90, zorder=5, label='planted mean_rev=0.12')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('HAC t (fade vs 0)')\n"
            "ax.set_title('Control: the null never fires; a planted effect lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {np.nanmean(null_ts):+.2f} (sd {np.nanstd(null_ts, ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(np.abs(null_ts) >= 2).sum()}/20 seeds  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector fires "
            f"\\|*t*\\| ≥ 2 in **{R['syn_null_fire']}/20** seeds — the null "
            f"never fires — and a planted mean-reversion tendency lights up "
            f"sharply (*t* = **{R['syn_planted_t']:.2f}**). The detection + "
            "inference pipeline is unbiased; the real-tape result's flat, "
            "uncertified reading is genuine, not a bug in the machinery."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — pooled 5-day fade **+{R['cyp'][5]['mean']:.2f} "
            f"bps** (HAC *t* = **{R['cyp'][5]['t']:.2f}**; Welch *t* vs base "
            f"rate = **{R['base'][5]['t']:.2f}**) on **{R['n_cyp']}** touches "
            f"over 21-25 years; **{R['bonf_survive']}/{R['bonf_n_tests']}** "
            f"Bonferroni-corrected tests survive (critical \\|*t*\\| = "
            f"{R['bonf_thr']:.2f}); hit rate 50.4% is a coin flip; five of six "
            "tickers are individually positive but none significantly so.\n"
            "- **Tradability `MIRAGE`** — every hold length is weak before "
            f"costs and none clears significance; by 20 days the average "
            f"({R['timer'][20]['gross']:.0f} bps, *t*={R['timer'][20]['t_gross']:+.2f}) "
            "has flipped negative on its own, before costs are even charged.\n"
            "- **\"Does 0.786 of XC beat a placebo retracement?\" `BUSTED`** — "
            f"beats the placebo on only {R['n_beats_plac']}/6 tickers (a coin "
            f"flip), and at the pooled headline horizon the placebo actually "
            f"*outperforms* the pattern's own ratio (Welch *t* = "
            f"**{R['plac'][5]['t']:.2f}**)."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general lesson: an extra construction step reads as more "
            "rigorous, not more predictive.** The Cypher's pitch is that its "
            "C-overshoot-then-XC-retrace geometry is a sharper filter than the "
            "Gartley/Bat/Butterfly/Crab family — the machinery here shows that "
            "extra complexity does not translate into a detectable edge, a "
            "coin-flip hit rate, or specificity over an arbitrary retracement "
            "on the same pivots.\n"
            "- **Where a believer would push back:** Carney's full method "
            "requires a *confluence zone* (several overlapping Fibonacci "
            "projections converging near D), not the bare two-ratio skeleton "
            "tested here. That's a legitimate next experiment — but it adds "
            "researcher degrees of freedom (which projections? how many must "
            "converge? how tight a zone?) that would need their own placebo "
            "before counting as evidence.\n"
            "- **Dedup map:** [468-gartley-harmonic](../../468-gartley-harmonic/) "
            "(the same 78.6% *number*, but measured off XA, not XC), "
            "[698-abcd-harmonic](../../698-abcd-harmonic/) (no X point, no "
            "confluence at all), [699-butterfly-harmonic](../../699-butterfly-harmonic/), "
            "[700-bat-harmonic](../../700-bat-harmonic/) and "
            "[701-crab-harmonic](../../701-crab-harmonic/) (all measure D off "
            "XA or AB, never off an extended XC leg) — all independently "
            "converging on the same verdict shape.\n\n"
            "*The reproducible core is offline and deterministic; frozen "
            "numbers live in [`docs/results.md`](../docs/results.md), sources "
            "in [`docs/references.md`](../docs/references.md).*"
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
