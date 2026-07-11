"""Generate the two narrative notebooks for Study 701 (Crab-Harmonic).

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
# pct=0.02 zigzag, AB retrace 0.382-0.618 of XA, BC retrace 0.382-0.886 of AB,
# D = X - 1.618*(A-X), 120-session touch window).
R = dict(
    basket=("SPY", "QQQ", "AAPL", "MSFT", "TSLA", "NVDA"),
    asof="2026-06-30",
    n_pivots=6055, n_cand_crab=226, n_crab=76,
    n_cand_plac=226, n_plac=79,
    crab={1: dict(hit=53.9, mean=-18.85, t=-0.38), 5: dict(hit=39.5, mean=-82.70, t=-1.30),
          10: dict(hit=40.8, mean=-253.84, t=-2.40)},
    wilson_crab=(29.2, 50.7),
    per_instrument={
        "SPY":  dict(n=4,  mean=312.64,  t=None),
        "QQQ":  dict(n=12, mean=99.34,   t=1.88),
        "AAPL": dict(n=11, mean=-255.74, t=-2.45),
        "MSFT": dict(n=11, mean=143.03,  t=1.14),
        "TSLA": dict(n=16, mean=-57.46,  t=-0.28),
        "NVDA": dict(n=22, mean=-298.59, t=-2.43),
    },
    # Signal axis — vs random-day base rate (matched direction mix, 20 seeds)
    base={1: dict(mean=-1.44, t=-0.36), 5: dict(mean=-34.88, t=-0.61), 10: dict(mean=-42.36, t=-2.11)},
    n_base=1520,
    base_per_instrument={
        "SPY":  dict(base_mean=20.38,  t=1.60),
        "QQQ":  dict(base_mean=12.33,  t=0.94),
        "AAPL": dict(base_mean=-28.13, t=-2.05),
        "MSFT": dict(base_mean=-13.96, t=1.15),
        "TSLA": dict(base_mean=-82.13, t=0.10),
        "NVDA": dict(base_mean=-50.14, t=-1.54),
    },
    bonf_n_tests=7, bonf_thr=2.69, bonf_uncorrected=1, bonf_survive=0,
    # Tradability axis — the fade timer, net of costs
    timer={
        1:  dict(gross=-18.85,  t_gross=-0.38, net5=-28.85,  t5=-0.58, net10=-38.85,  t10=-0.78),
        3:  dict(gross=-83.17,  t_gross=-1.27, net5=-93.17,  t5=-1.42, net10=-103.17, t10=-1.57),
        5:  dict(gross=-82.70,  t_gross=-1.30, net5=-92.70,  t5=-1.46, net10=-102.70, t10=-1.61),
        10: dict(gross=-253.84, t_gross=-2.40, net5=-263.84, t5=-2.50, net10=-273.84, t10=-2.59),
        20: dict(gross=-503.01, t_gross=-2.76, net5=-513.01, t5=-2.81, net10=-523.01, t10=-2.87),
    },
    # Third axis — vs placebo
    plac={1: dict(mean=-56.29, t=0.57), 5: dict(mean=-115.68, t=0.28), 10: dict(mean=-347.94, t=0.59)},
    plac_per_instrument={
        "SPY":  dict(n=4,  mean=-80.99,  beats=True),
        "QQQ":  dict(n=11, mean=134.71,  beats=False),
        "AAPL": dict(n=14, mean=-91.02,  beats=False),
        "MSFT": dict(n=11, mean=-6.68,   beats=True),
        "TSLA": dict(n=17, mean=-335.39, beats=True),
        "NVDA": dict(n=22, mean=-147.58, beats=False),
    },
    n_beats_plac=3,
    # synthetic control
    syn_null_mean=-0.13, syn_null_sd=1.10, syn_null_fire=1, syn_null_seeds=20,
    syn_planted_n=8, syn_planted_mean=394.01, syn_planted_t=4.64,
    fp={"SPY": "58d0459a7599", "QQQ": "c48f22566e73", "AAPL": "75ce4521e4f7",
        "MSFT": "fb9333ad5b2b", "TSLA": "f8ca92e420b8", "NVDA": "1e614f1ea32c"},
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

from crab_harmonic import data, strategy as st

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
            "# Does the Crab harmonic call the sharpest turn of them all? 🦀\n"
            "### Five dots, one 1.618 overshoot, and a stopwatch\n\n"
            + BADGES +
            "Of all the \"harmonic patterns\" retail charting software draws for you, the "
            "**Crab** claims to be the *most precise*: Scott Carney calls it the "
            "\"sharpest\" and most \"exact\" reversal zone in his entire system. Its "
            "signature is a single, tight number — price must extend **1.618 times** "
            "the move that started it, *past* the pattern's own origin point — and "
            "right there, allegedly, it turns.\n\n"
            "So we built a robot that finds every Crab on six liquid tapes over the "
            "last 21-25 years — no eyeballing, no cherry-picking — and asked whether "
            "fading price right where the pattern says to would actually have made "
            "money.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the Bonferroni correction "
            "and the placebo math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** Pivots are a confirmed percentage zig-zag (known only "
            "after price reverses enough to lock them in — no look-ahead). Every chart "
            "is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does price turn where the Crab says it should? | **No — if anything it "
            f"keeps going.** Over a 5-day hold the pattern averages "
            f"**{R['crab'][5]['mean']:+.0f} bps** — negative, not positive, with a hit "
            f"rate *below* a coin flip. |\n"
            f"| Does it get worse or better if you wait longer? | **Worse.** By 10 days "
            f"the loss (**{R['crab'][10]['mean']:+.0f} bps**) is large enough to be "
            "individually significant — in the losing direction. |\n"
            f"| Does the exact 1.618 ratio matter? | **Not provably.** Swap that number "
            "for a random extension on the *identical* pivots and you do about as "
            f"badly (Welch *t* = {R['plac'][5]['t']:.2f} — nowhere near significant, "
            "and both arms lose money). |\n"
            "| Can you trade it? | **No.** Every hold length loses money on average, "
            "gross and net of costs, and the losses only grow the longer you hold. |\n\n"
            "> A pattern billed as the family's most \"exact\" reversal zone turns out "
            "to be, on the real tape, a mechanical way to bet against a trend that "
            "keeps trending."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Find four swing points X-A-B-C. If B retraces **38-62%** of the X-A "
            "move and C bounces back **38-89%** of the A-B move, project a fifth point "
            "D that **overshoots X by exactly 1.618 times** the original X-A leg — a "
            "single, sharp target, not a range. That overshoot marks an exhausted, "
            "overextended move — the highest-probability reversal zone in the whole "
            "system.\"*\n\n"
            "That's Scott Carney's Crab (*Harmonic Trading, Volume 2*, 2007), his own "
            "refinement on the Gartley/Butterfly grid. He markets it specifically as "
            "*more precise* than its siblings — Gartley, Bat, Butterfly — because "
            "1.618 is one number, not a band. It's baked into TradingView, MetaTrader "
            "and Thinkorswim's auto-scanners as one of the four named \"harmonic\" "
            "shapes."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a pattern this specific — a single decimal ratio chained across four "
            "historical swings — genuinely forecasts a turn, and does so *more "
            "reliably* than its siblings, that's a striking, calculator-testable "
            "crack in market efficiency. And harmonic trading is popular *precisely* "
            "because the charts look uncanny after the fact: hindsight makes any "
            "completed pattern look inevitable.\n\n"
            "The trap is exactly that hindsight. A trader hand-labelling swings on a "
            "chart, with the benefit of already knowing what happened next, will "
            "always find the Crab that worked. So we removed the human: a mechanical "
            "rule finds every qualifying X-A-B-C on the tape, in real time, with no "
            "knowledge of the future — then we measure what actually happened at D."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The basket.** {', '.join(R['basket'])} — six liquid names, daily bars, "
            f"2001/2010 → {R['asof']}.\n"
            "- **The rule, mechanically.** A confirmed zig-zag finds swing pivots (only "
            "known once price has reversed enough to lock them in). Every qualifying "
            "X-A-B-C projects a D level; we scan forward up to 120 sessions for the "
            "first touch.\n"
            "- **The trade.** Fade at the touch — bet on the reversal the pattern "
            "predicts — and measure the forward return.\n"
            "- **Two honest baselines, announced up front.** (1) A **random-day base "
            "rate**: same ticker, same number of trades, same bullish/bearish mix, "
            "just on random days — kills the \"stocks drift up anyway\" confound. "
            "(2) A **placebo extension zone**: the identical pivots, but with the "
            "1.618 number replaced by a random one — kills the \"any overshoot "
            "projection would work\" confound. And because we test six tickers, we "
            "**correct for that** (Bonferroni) before calling anything real."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Fade the D-touch, hold 5 days, average across "
            "every qualifying Crab on the basket."
        ),
        code(
            "if HAVE_REAL:\n"
            "    frames = []\n"
            "    for t in data.BASKET:\n"
            "        _, _, ledger = st.detect_and_scan(BARS[t], pct=PCT, cost_bps=0.0)\n"
            "        if not ledger.empty:\n"
            "            frames.append(ledger)\n"
            "    CRAB = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()\n"
            "    s5 = st.summarize(CRAB, 'ret_gross_5')\n"
            "    n5, m5, t5 = s5['n'], s5['mean_bps'], s5['t']\n"
            "else:\n"
            "    n5, m5, t5 = R['n_crab'], R['crab'][5]['mean'], R['crab'][5]['t']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['Crab D-touch\\n(5-day fade)'], [m5], color=RED, width=.5)\n"
            "ax.annotate(f'{m5:+.0f} bps\\n(n={n5}, t={t5:+.2f})', (0, m5), ha='center',\n"
            "            va='bottom' if m5 >= 0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean forward return (bps)')\n"
            "ax.set_title('The headline number — already negative')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'n={n5}  mean={m5:+.2f}bps  t={t5:+.2f}')"
        ),
        md(
            f"**{R['crab'][5]['mean']:+.0f} bps** on average, *t* = "
            f"**{R['crab'][5]['t']:.2f}** — negative from the very first look, before "
            "any control is applied. If the Crab's \"sharpest reversal zone\" claim "
            "were true, this would already be the wrong sign.\n\n"
            "**Second, the honest baseline.** Stocks drift up. A random buy on a "
            "random day, matched to the same bullish/bearish mix as our Crab trades, "
            "earns *something* just from that drift (or loses, if the mix leans "
            "bearish). Does the Crab do worse than even that?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    crab_by_t, base_frames = {}, []\n"
            "    for t in data.BASKET:\n"
            "        _, _, l = st.detect_and_scan(BARS[t], pct=PCT, cost_bps=0.0)\n"
            "        crab_by_t[t] = l\n"
            "        if len(l):\n"
            "            mix = float((l['reversal_dir'] > 0).mean())\n"
            "            for s in range(20):\n"
            "                base_frames.append(st.base_rate_ledger(BARS[t], len(l), mix,\n"
            "                    seed=st._seed_from(701, t, s)))\n"
            "    BASE = pd.concat(base_frames, ignore_index=True) if base_frames else pd.DataFrame()\n"
            "    sb = st.summarize(BASE, 'ret_gross_5')\n"
            "    bm, wt = sb['mean_bps'], st.welch_t(CRAB['ret_gross_5'], BASE['ret_gross_5'])\n"
            "else:\n"
            "    bm, wt = R['base'][5]['mean'], R['base'][5]['t']\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.4))\n"
            "ax.bar(['Crab D-touch', 'random-day\\nbase rate'], [m5, bm], color=[RED, GREY], width=.55)\n"
            "for i, v in enumerate([m5, bm]):\n"
            "    ax.annotate(f'{v:+.0f} bps', (i, v), ha='center', va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean 5-day return (bps)')\n"
            "ax.set_title(f'Underperforms a matched-direction random day (Welch t = {wt:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Crab {m5:+.1f} bps vs base rate {bm:+.1f} bps  Welch t={wt:+.2f}')"
        ),
        md(
            f"The Crab does *worse* than the random-day baseline (Welch *t* = "
            f"**{R['base'][5]['t']:.2f}**) — the mechanical fade loses more than a "
            "coin flip with the same directional lean would. Look at the sample: "
            "**76 touches, spread across six stocks, over 21-25 years** — roughly "
            "three Crab completions a year, pooled across the whole basket — and we "
            "tested **six different tickers**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    per_t = {t: st.summarize(crab_by_t[t], 'ret_gross_5') for t in data.BASKET}\n"
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
            "ax.set_title('Six tickers, no consistent story')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({t: round(m,1) for t,m in zip(tickers, means)})"
        ),
        md(
            "Three of six tickers are positive, three are negative, and the two "
            "worst losers (AAPL, NVDA) individually clear the naive *t* ≥ 2 bar — "
            "**in the losing direction**. That spread is the signature of a thin, "
            "noisy sample with no shared mechanism, not a discovery: correcting "
            f"properly for having tested six tickers, **{R['bonf_survive']} of "
            f"{R['bonf_n_tests']}** results survive.\n\n"
            "**Finally, the placebo.** Does the *specific* 1.618 extension matter, or "
            "would any similarly-shaped overshoot have done just as (badly)?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    plac_frames = []\n"
            "    for t in data.BASKET:\n"
            "        _, _, lp = st.detect_and_scan(BARS[t], pct=PCT, placebo=True, seed=701, cost_bps=0.0)\n"
            "        if not lp.empty:\n"
            "            plac_frames.append(lp)\n"
            "    PLAC = pd.concat(plac_frames, ignore_index=True) if plac_frames else pd.DataFrame()\n"
            "    sp = st.summarize(PLAC, 'ret_gross_5')\n"
            "    pm, wtp = sp['mean_bps'], st.welch_t(CRAB['ret_gross_5'], PLAC['ret_gross_5'])\n"
            "else:\n"
            "    pm, wtp = R['plac'][5]['mean'], R['plac'][5]['t']\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.4))\n"
            "ax.bar(['Crab\\n(1.618)', 'placebo\\n(random extension)'], [m5, pm],\n"
            "       color=[RED, GREY], width=.55)\n"
            "for i, v in enumerate([m5, pm]):\n"
            "    ax.annotate(f'{v:+.0f} bps', (i, v), ha='center', va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean 5-day return (bps)')\n"
            "ax.set_title(f'Both arms lose money; the magic number barely differs (Welch t = {wtp:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Crab {m5:+.1f} bps vs placebo {pm:+.1f} bps  Welch t={wtp:+.2f}')"
        ),
        md(
            f"Both arms lose money — the Crab's exact **{R['crab'][5]['mean']:+.0f} "
            f"bps** and the placebo's **{R['plac'][5]['mean']:+.0f} bps** — and the "
            f"gap between them is nowhere near statistically real (Welch *t* = "
            f"**{R['plac'][5]['t']:.2f}**). The specific number — 1.618 — is not "
            "doing detectably more (or less) damage than an arbitrary extension "
            "target would."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The fade is negative from the first look "
            f"(*t* = {R['crab'][5]['t']:.2f}), underperforms a matched-direction "
            f"random day (*t* = {R['base'][5]['t']:.2f}), and fails a six-ticker "
            f"correction ({R['bonf_survive']}/{R['bonf_n_tests']} survive).\n"
            "- **Tradability — Mirage.** Every hold length loses money on average, "
            "gross and net; the loss only grows the longer you hold.\n"
            "- **\"Is 1.618 the sharpest, most exact ratio?\" — Busted.** Swap it for "
            "a random extension on the identical pivots and you do about as badly "
            f"(Welch *t* = {R['plac'][5]['t']:.2f})."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The general lesson: negative results are still results.** This "
            "isn't just \"no edge found\" — the mechanical fade actively loses "
            "money, and increasingly so with time. If anything, the honest reading "
            "is that price tends to *continue* past a Crab's D, not reverse.\n"
            "- **Sibling studies:** [468-gartley-harmonic](../../468-gartley-harmonic/) "
            "(the classic five-point pattern, D stays *inside* the X-A range), "
            "[699-butterfly-harmonic](../../699-butterfly-harmonic/) (a looser "
            "1.27-1.618 extension *range*) and "
            "[700-bat-harmonic](../../700-bat-harmonic/) (D stays inside X-A) all "
            "reach the same shape of verdict independently.\n\n"
            "*Think the Crab needs its full \"confluence zone\" — several overlapping "
            "Fibonacci projections converging near D — to work? Show it beats this "
            "study's placebo and survives the multiple-comparison correction, then "
            "we'll talk.*"
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
            "# The Crab-Harmonic — a quantitative teardown 🔬\n"
            "### Mechanical XABCD detection · a random-day base rate · a Bonferroni "
            "correction across the basket · a fade timer with costs · a placebo "
            "extension-zone control · a 20-seed synthetic positive control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The claim — **a D point that extends exactly 1.618x past X marks the "
            "\"sharpest, most exact\" reversal in the harmonic zoo** — is tested on "
            "the tightest mechanical reading a proponent would accept, with two "
            "independent controls (a matched-direction base rate and a randomized "
            "extension placebo) and an explicit multiple-comparison penalty across "
            "the basket.\n\n"
            "> ⚠️ **Data note.** Daily OHLC (2001/2010→2026), yfinance, cached; "
            "basket identical to [468-gartley-harmonic](../../468-gartley-harmonic/) "
            "and [699-butterfly-harmonic](../../699-butterfly-harmonic/). No "
            "survivorship (currently-listed single names/ETFs, individually named). "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to "
            "intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | pooled 5-day D-touch fade **{R['crab'][5]['mean']:.2f} "
            f"bps** (HAC *t* = **{R['crab'][5]['t']:.2f}** vs 0; Welch *t* = "
            f"**{R['base'][5]['t']:.2f}** vs base rate) — negative at every "
            f"horizon; **{R['bonf_survive']}/{R['bonf_n_tests']}** Bonferroni-corrected "
            f"tests survive (critical \\|*t*\\| = {R['bonf_thr']:.2f}) and placebo "
            f"Welch *t* = **{R['plac'][5]['t']:.2f}** |\n"
            f"| **Tradability** | `MIRAGE` | {R['n_crab']} events over 21-25 years "
            "pooled across six names; every hold length negative, worst at 20 days "
            f"({R['timer'][20]['gross']:+.0f} bps, *t*={R['timer'][20]['t_gross']:+.2f}) |\n"
            f"| **Beats a placebo?** | `BUSTED` | beats on {R['n_beats_plac']}/6 "
            f"tickers; pooled Welch *t* = {R['plac'][5]['t']:.2f} |\n\n"
            "> 💡 In plain words: a number that is negative from the first look, "
            "underperforms a matched-direction coin flip, and fails every check we "
            "throw at it."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Four confirmed swing pivots $X, A, B, C$, with $XA = A - X$, "
            "$AB = B - A$, $BC = C - B$. The Crab grid:\n\n"
            "- $|AB / XA| \\in [0.382, 0.618]$ (shallower than the Butterfly's fixed "
            "0.786, close to but not identical to the Bat's 0.382-0.50).\n"
            "- $|BC / AB| \\in [0.382, 0.886]$ (shared structural band across the "
            "XABCD zoo).\n"
            "- $D = X - 1.618 \\times XA$ — **D extends *past* X by exactly 1.618**, "
            "a single, sharp target (not a range like the Butterfly's "
            "1.27-1.618) — the defining ratio, and the one Carney calls the "
            "\"sharpest\" and most \"exact\" in the family.\n\n"
            "$\\text{reversal\\_dir} = \\text{sign}(XA)$: a bullish Crab (XA up) "
            "projects D below X and expects a bounce up; a bearish one, the mirror.\n\n"
            "**H₁ (reversal).** Fading at D beats zero and a random-day base rate.\n"
            "**H₂ (specificity).** The 1.618 target beats a randomized extension "
            "placebo on the identical pivots.\n"
            "**H₃ (robustness).** The result survives correction for testing six "
            "tickers.\n\n"
            "We find **H₁ rejected** (pooled *t* is negative at every horizon), "
            "**H₂ not supported** (placebo *t* = 0.28, both arms lose money), "
            "**H₃ moot** (nothing to correct — the signal is negative to begin with)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Crab D-touches on the same tape can cluster and overlap, so within-arm "
            "means are tested with a **Newey-West (HAC) t** against zero. The "
            "market's positive unconditional drift means a signal-vs-zero test alone "
            "is not enough — the primary Signal test is a **Welch t vs a random-day "
            "base rate** built to match the *same* bullish/bearish direction mix as "
            "the real touches, pooled over 20 seeds (kills the drift confound). "
            "Because the basket is tested seven ways (pooled + 6 tickers, all "
            "5-day), a **Bonferroni correction** (family-wise alpha 0.05) raises the "
            "critical |*t*| from the naive 2.0 to **2.69** before any test is "
            "allowed to \"survive.\" A **second, independent placebo** — randomized "
            "extension targets on the identical pivots — tests H₂ separately from "
            "H₃."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Basket.** {', '.join(R['basket'])}, daily bars, 2001/2010 → "
            f"{R['asof']} (last complete month).\n"
            "- **Pivots.** Percentage-threshold zig-zag, **pct = 2%**, confirmed "
            "pivots only (recorded at the confirmation bar, never the earlier "
            "extreme — no look-ahead).\n"
            "- **Candidates.** Every consecutive confirmed X-A-B-C quadruple "
            "satisfying the Crab structural bands; D fully computable the moment C "
            "confirms.\n"
            "- **Touch scan.** 120-session forward window from C's confirmation for "
            "the first bar bracketing D.\n"
            "- **Execution.** Enter the fade at the touch bar's own close; one "
            "round trip = 2 × one-way cost × NAV.\n"
            "- **Signal control.** Random-day base rate, matched direction mix, 20 "
            "seeds; Bonferroni correction across 7 tests.\n"
            "- **Specificity control.** Randomized extension-target placebo on "
            "identical pivots.\n"
            "- **Faithful-engine control.** Synthetic mean-reverting price index, "
            "tunable reversion knob; the null must not fire across 20 seeds."
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
            "    crab_by_t = {}\n"
            "    frames = []\n"
            "    for t in data.BASKET:\n"
            "        _, _, l = st.detect_and_scan(BARS[t], pct=PCT, cost_bps=0.0)\n"
            "        crab_by_t[t] = l\n"
            "        if len(l):\n"
            "            frames.append(l)\n"
            "    CRAB = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()\n"
            "    hs = [1, 5, 10]\n"
            "    means = [st.summarize(CRAB, f'ret_gross_{h}')['mean_bps'] for h in hs]\n"
            "    ts = [st.summarize(CRAB, f'ret_gross_{h}')['t'] for h in hs]\n"
            "    per_t = {t: st.summarize(crab_by_t[t], 'ret_gross_5') for t in data.BASKET}\n"
            "    tick_means = [per_t[t]['mean_bps'] for t in data.BASKET]\n"
            "else:\n"
            "    hs = [1, 5, 10]\n"
            "    means = [R['crab'][h]['mean'] for h in hs]\n"
            "    ts = [R['crab'][h]['t'] for h in hs]\n"
            "    tick_means = [R['per_instrument'][t]['mean'] for t in R['basket']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.4, 4.4))\n"
            "cols1 = [RED if h == 5 else GREY for h in hs]\n"
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
            f"> 💡 In plain words: the 5-day pooled fade (**{R['crab'][5]['mean']:.2f} "
            f"bps**, HAC *t* = **{R['crab'][5]['t']:.2f}**) is negative, and 10-day "
            f"(*t* = {R['crab'][10]['t']:.2f}) is *more* negative, not less — the "
            "opposite of the mean-reverting decay you'd expect if the effect were "
            "real but noisy. The per-ticker split shows three of six tickers "
            "positive, three negative, with the two worst losers (AAPL, NVDA) "
            "individually significant on the losing side."
        ),
        md(
            "### 4b · The Signal axis — random-day base rate, Bonferroni-corrected\n\n"
            "20-seed random-day base rate matched on the empirical bullish/bearish "
            "mix of the real touches; Welch *t* vs the Crab ledger, pooled and per "
            "ticker; Bonferroni correction across the resulting 7 tests."
        ),
        code(
            "if HAVE_REAL:\n"
            "    base_frames, base_by_t = [], {}\n"
            "    for t in data.BASKET:\n"
            "        l = crab_by_t[t]\n"
            "        if len(l):\n"
            "            mix = float((l['reversal_dir'] > 0).mean())\n"
            "            bl = pd.concat([st.base_rate_ledger(BARS[t], len(l), mix,\n"
            "                             seed=st._seed_from(701, t, s)) for s in range(20)],\n"
            "                            ignore_index=True)\n"
            "            base_by_t[t] = bl\n"
            "            base_frames.append(bl)\n"
            "    BASE = pd.concat(base_frames, ignore_index=True) if base_frames else pd.DataFrame()\n"
            "    tests = [('pooled', st.welch_t(CRAB['ret_gross_5'], BASE['ret_gross_5']))]\n"
            "    for t in data.BASKET:\n"
            "        if t in base_by_t and len(crab_by_t[t]):\n"
            "            tests.append((t, st.welch_t(crab_by_t[t]['ret_gross_5'], base_by_t[t]['ret_gross_5'])))\n"
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
            "ax.set_ylabel('Welch t (Crab vs base rate)')\n"
            "ax.set_title('Nothing clears the Bonferroni-corrected bar')\n"
            "ax.legend(fontsize=8.5); plt.tight_layout(); plt.show()\n"
            "print({lbl: round(v,2) if np.isfinite(v) else None for lbl, v in tests})\n"
            "print(f'Bonferroni critical |t| = {thr:.2f}')"
        ),
        md(
            f"> 💡 In plain words: only one per-ticker split (AAPL, Welch *t* = "
            f"**{R['base_per_instrument']['AAPL']['t']:.2f}**) clears the naive bar "
            "of 2 — and it clears it on the *losing* side. The honest bar, once you "
            f"correct for looking at six tickers, is **{R['bonf_thr']:.2f}**, and "
            f"**{R['bonf_survive']}/{R['bonf_n_tests']}** tests clear it. This is the "
            "desk's multiple-comparison rail doing exactly its job — on a result "
            "that was already unpromising before correction."
        ),
        md(
            "### 4c · Tradability — the fade timer, net of costs\n\n"
            "Sweep the holding period and the one-way cost; ~3 pooled events/year "
            "across six names."
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
            "ax.set_title('Every hold length loses money — and the loss grows with time')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print({h: (R['timer'][h]['gross'], R['timer'][h]['t_gross']) for h in hs})"
        ),
        md(
            "> 💡 In plain words: costs (5-10 bps round trip) are a rounding error "
            "next to these 20-500 bps losses. What kills Tradability is the *shape*: "
            "negative at 1 day, more negative at every horizon out to 20 days, "
            "individually significant on the losing side by 10-20 days. Combined "
            "with the Bonferroni result above and a firing rate of roughly three "
            "completions a year pooled across six names, there is no regime here "
            "worth deploying capital against."
        ),
        md(
            "### 4d · The third axis — does the specific ratio matter?\n\n"
            "Identical pivot pipeline; each placebo candidate's D-extension *target* "
            "is replaced by a deterministic, seeded, off-Crab draw."
        ),
        code(
            "if HAVE_REAL:\n"
            "    plac_by_t, plac_frames = {}, []\n"
            "    for t in data.BASKET:\n"
            "        _, _, lp = st.detect_and_scan(BARS[t], pct=PCT, placebo=True, seed=701, cost_bps=0.0)\n"
            "        plac_by_t[t] = lp\n"
            "        if len(lp):\n"
            "            plac_frames.append(lp)\n"
            "    PLAC = pd.concat(plac_frames, ignore_index=True) if plac_frames else pd.DataFrame()\n"
            "    sp = st.summarize(PLAC, 'ret_gross_5')\n"
            "    pm = sp['mean_bps']\n"
            "    wtp = st.welch_t(CRAB['ret_gross_5'], PLAC['ret_gross_5'])\n"
            "    beat_labels, beat_vals = [], []\n"
            "    for t in data.BASKET:\n"
            "        fm = st.summarize(crab_by_t[t], 'ret_gross_5')['mean_bps']\n"
            "        pmi = st.summarize(plac_by_t[t], 'ret_gross_5')['mean_bps']\n"
            "        beat_labels.append(t); beat_vals.append(fm - pmi if np.isfinite(fm) and np.isfinite(pmi) else np.nan)\n"
            "else:\n"
            "    pm, wtp = R['plac'][5]['mean'], R['plac'][5]['t']\n"
            "    beat_labels = list(R['basket'])\n"
            "    beat_vals = [R['per_instrument'][t]['mean'] - R['plac_per_instrument'][t]['mean']\n"
            "                 for t in R['basket']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.0, 4.4))\n"
            "a1.bar(['Crab', 'placebo'], [R['crab'][5]['mean'], pm], color=[RED, GREY], width=.55)\n"
            "for i, v in enumerate([R['crab'][5]['mean'], pm]):\n"
            "    a1.annotate(f'{v:+.0f}bps', (i, v), ha='center', va='bottom' if v>=0 else 'top')\n"
            "a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_title(f'Pooled: Welch t = {wtp:+.2f} (not significant)')\n"
            "cols = [GREEN if (np.isfinite(v) and v > 0) else RED for v in beat_vals]\n"
            "a2.bar(beat_labels, [v if np.isfinite(v) else 0 for v in beat_vals], color=cols, width=.6)\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_title('Crab minus placebo, per ticker')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Crab {R[\"crab\"][5][\"mean\"]:+.1f} bps vs placebo {pm:+.1f} bps  Welch t={wtp:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the Crab beats its own placebo on only "
            f"**{R['n_beats_plac']}/6** tickers — a coin flip — and the pooled "
            f"Welch *t* (**{R['plac'][5]['t']:.2f}**) is nowhere near significant. "
            "Both arms lose money on average; the exact 1.618 ratio Carney calls "
            "\"sharpest\" is not demonstrably better (or worse) than an arbitrary "
            "extension zone built from the *same* pivots — H₂ not supported."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic mean-reverting price index (tunable `mean_rev` knob toward a "
            "slow EMA), pooled across a synthetic 6-name basket per seed. The null "
            "(`mean_rev = 0`) is checked over **20 seeds** — never a single stream."
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
            "null_ts = np.array([st.summarize(pooled_synth(0.0, 701 + s), 'ret_gross_5')['t']\n"
            "                     for s in range(20)], dtype=float)\n"
            "planted = pooled_synth(0.12, 701)\n"
            "planted_t = st.summarize(planted, 'ret_gross_5')['t']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (mean_rev=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=GREEN, s=90, zorder=5, label='planted mean_rev=0.12')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('HAC t (fade vs 0)')\n"
            "ax.set_title('Control: the null fires near its nominal rate; a planted effect lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {np.nanmean(null_ts):+.2f} (sd {np.nanstd(null_ts, ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(np.abs(null_ts) >= 2).sum()}/20 seeds  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector fires "
            f"\\|*t*\\| ≥ 2 in **{R['syn_null_fire']}/20** seeds — close to the "
            f"nominal ~5% false-positive rate — and a planted mean-reversion "
            f"tendency lights up sharply (*t* = **{R['syn_planted_t']:.2f}**). The "
            "detection + inference pipeline is unbiased; the real-tape result's "
            "negative direction is genuine, not a bug in the machinery."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — pooled 5-day fade **{R['crab'][5]['mean']:.2f} "
            f"bps** (HAC *t* = **{R['crab'][5]['t']:.2f}**; Welch *t* vs base rate = "
            f"**{R['base'][5]['t']:.2f}**) on **{R['n_crab']}** touches over 21-25 "
            f"years; **{R['bonf_survive']}/{R['bonf_n_tests']}** Bonferroni-corrected "
            f"tests survive (critical \\|*t*\\| = {R['bonf_thr']:.2f}); 10-day "
            f"horizon is significantly negative (*t* = {R['crab'][10]['t']:.2f}); "
            "three of six tickers are individually negative, two of them "
            "significantly so.\n"
            "- **Tradability `MIRAGE`** — every hold length (1/3/5/10/20 days) loses "
            f"money gross and net; by 20 days the loss ({R['timer'][20]['gross']:.0f} "
            f"bps, *t*={R['timer'][20]['t_gross']:+.2f}) is itself statistically "
            "significant, in the wrong direction for a trader betting on reversal.\n"
            f"- **\"Is 1.618 the sharpest, most exact ratio?\" `BUSTED`** — beats the "
            f"placebo on only {R['n_beats_plac']}/6 tickers (a coin flip), pooled "
            f"Welch *t* = **{R['plac'][5]['t']:.2f}** — not statistically "
            "demonstrable, and both arms lose money regardless."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general lesson is: check the sign before you check "
            "significance.** A pattern billed as the family's most \"exact\" "
            "reversal zone produced a fade that loses money at every horizon on "
            "the real tape — the Bonferroni correction and placebo control here "
            "confirm a negative finding rather than rescue a borderline positive "
            "one, which is itself informative about how far marketing claims can "
            "drift from the data.\n"
            "- **Where a believer would push back:** Carney's full method requires "
            "a *confluence zone* (several overlapping Fibonacci projections "
            "converging near D), not the bare two-ratio skeleton tested here. "
            "That's a legitimate next experiment — but it adds researcher degrees "
            "of freedom (which projections? how many must converge? how tight a "
            "zone?) that would need their own placebo before counting as evidence.\n"
            "- **Dedup map:** [468-gartley-harmonic](../../468-gartley-harmonic/) "
            "(D stays *inside* the X-A range, different B ratio), "
            "[698-abcd-harmonic](../../698-abcd-harmonic/) (no X point, no "
            "confluence at all), [699-butterfly-harmonic](../../699-butterfly-harmonic/) "
            "(a looser 1.27-1.618 extension *range*) and "
            "[700-bat-harmonic](../../700-bat-harmonic/) (D stays inside X-A) — all "
            "independently converging on the same verdict shape.\n\n"
            "*The reproducible core is offline and deterministic; frozen numbers "
            "live in [`docs/results.md`](../docs/results.md), sources in "
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
