"""Generate the two narrative notebooks for Study 702 (Shark-Harmonic).

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
# pct=0.02 zigzag, AB extends XA 1.13-1.618x past X, BC extends AB 1.618-2.24x past
# A, D = a ZONE at [0.886, 1.13] x the original XA leg, 120-session touch window).
R = dict(
    basket=("SPY", "QQQ", "AAPL", "MSFT", "TSLA", "NVDA"),
    asof="2026-06-30",
    n_pivots=6055, n_cand_shark=103, n_shark=94,
    n_cand_plac=103, n_plac=97,
    shark={1: dict(hit=50.0, mean=33.53, t=1.34), 5: dict(hit=54.3, mean=22.81, t=0.42),
          10: dict(hit=53.2, mean=8.88, t=0.10)},
    wilson_shark=(44.2, 64.0),
    per_instrument={
        "SPY":  dict(n=11, mean=181.73,  t=4.63),
        "QQQ":  dict(n=12, mean=112.75,  t=1.24),
        "AAPL": dict(n=16, mean=59.52,   t=0.45),
        "MSFT": dict(n=14, mean=-46.53,  t=-0.48),
        "TSLA": dict(n=14, mean=82.50,   t=0.77),
        "NVDA": dict(n=27, mean=-98.65,  t=-0.72),
    },
    # Signal axis — vs random-day base rate (matched direction mix, 20 seeds)
    base={1: dict(mean=1.73, t=1.14), 5: dict(mean=-15.00, t=0.60), 10: dict(mean=-33.90, t=0.53)},
    n_base=1880,
    base_per_instrument={
        "SPY":  dict(base_mean=-7.04,  t=2.57),
        "QQQ":  dict(base_mean=-23.08, t=1.58),
        "AAPL": dict(base_mean=31.08,  t=0.24),
        "MSFT": dict(base_mean=-3.11,  t=-0.37),
        "TSLA": dict(base_mean=-0.45,  t=0.36),
        "NVDA": dict(base_mean=-55.65, t=-0.27),
    },
    bonf_n_tests=7, bonf_thr=2.69, bonf_uncorrected=1, bonf_survive=0,
    # Tradability axis — the fade timer, net of costs
    timer={
        1:  dict(gross=33.53, t_gross=1.34, net5=23.53, t5=0.94, net10=13.53, t10=0.54),
        3:  dict(gross=31.95, t_gross=0.87, net5=21.95, t5=0.60, net10=11.95, t10=0.33),
        5:  dict(gross=22.81, t_gross=0.42, net5=12.81, t5=0.23, net10=2.81,  t10=0.05),
        10: dict(gross=8.88,  t_gross=0.10, net5=-1.12, t5=-0.01, net10=-11.12, t10=-0.12),
        20: dict(gross=32.38, t_gross=0.21, net5=22.38, t5=0.14, net10=12.38, t10=0.08),
    },
    # Third axis — vs placebo
    plac={1: dict(mean=45.97, t=-0.36), 5: dict(mean=-0.68, t=0.26), 10: dict(mean=56.13, t=-0.42)},
    plac_per_instrument={
        "SPY":  dict(n=12, mean=125.53,  beats=True),
        "QQQ":  dict(n=13, mean=117.40,  beats=False),
        "AAPL": dict(n=14, mean=191.56,  beats=False),
        "MSFT": dict(n=15, mean=-2.84,   beats=False),
        "TSLA": dict(n=14, mean=-29.97,  beats=True),
        "NVDA": dict(n=29, mean=-183.40, beats=True),
    },
    n_beats_plac=3,
    # synthetic control
    syn_null_mean=0.08, syn_null_sd=1.07, syn_null_fire=2, syn_null_seeds=20,
    syn_planted_mean_rev=0.70, syn_planted_n=132, syn_planted_mean=-92.35, syn_planted_t=-5.21,
    fp={"SPY": "fc0e2330906f", "QQQ": "579f714ce50b", "AAPL": "5d82665ed860",
        "MSFT": "740543bb9d77", "TSLA": "f8ca92e420b8", "NVDA": "bdd3551bb11f"},
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

from shark_harmonic import data, strategy as st

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
            "# The Shark harmonic: the one pattern in the zoo that never pulls back 🦈\n"
            "### Two overshoots, a completion ZONE — and a real-tape shrug\n\n"
            + BADGES +
            "Every classic \"harmonic pattern\" retail charting software draws for you — "
            "Gartley, Bat, Butterfly, Crab — shares one habit: somewhere in the middle, "
            "price *pulls back* (a retracement) before the pattern completes. The "
            "**Shark**, introduced later by Scott Carney as a deliberately \"non-standard\" "
            "5-point structure, breaks that habit entirely: both of its middle legs "
            "**overshoot**, never retrace, and its completion isn't even a single price — "
            "it's a whole *zone*, 0.886 to 1.13 times the original move. The nickname is "
            "the \"5-0\" pattern.\n\n"
            "So we built a robot that finds every Shark on six liquid tapes over the "
            "last 21-25 years — no eyeballing, no cherry-picking — and asked whether "
            "fading price the moment it enters that zone would actually have made "
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
            f"| Does price turn where the Shark says it should? | **Not detectably.** "
            f"Over a 5-day hold the pattern averages "
            f"**{R['shark'][5]['mean']:+.0f} bps** — small and positive, but the hit "
            f"rate ({R['shark'][5]['hit']:.1f}%) is a coin flip and the number isn't "
            "statistically different from zero. |\n"
            f"| Does it beat a random day with the same directional lean? | **No.** "
            f"Welch *t* = {R['base'][5]['t']:.2f} vs a matched-direction random-day "
            "baseline — indistinguishable. |\n"
            "| Does the exact 0.886-1.13 zone matter? | **Not provably.** Move that "
            "zone somewhere else on the *identical* pivots and you do about the same "
            f"(Welch *t* = {R['plac'][5]['t']:.2f}, and the real zone only wins on "
            f"{R['n_beats_plac']}/6 tickers). |\n"
            "| Can you trade it? | **No.** No holding period, from 1 to 20 days, ever "
            "clears a basic significance bar, gross or net of costs. |\n\n"
            "> A pattern billed as the odd one out — built entirely from overshoots, "
            "with a real *zone* instead of a single magic number — still lands "
            "exactly where its siblings do: indistinguishable from noise."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Find four swing points X-A-B-C. B must **overshoot** the X-A move by "
            "**1.13-1.618 times**, past point X. C must overshoot the A-B move by a "
            "further **1.618-2.24 times**, past point B. Neither leg retraces — the "
            "structure just keeps extending. Then watch for price to enter the "
            "**0.886-1.13** zone of the *original* X-A move: that's the "
            "reversal zone, the exhaustion point where the overshoot runs out of "
            "steam.\"*\n\n"
            "That's Scott Carney's Shark (*Harmonic Trading, Volume 3*, 2015) — his own "
            "description calls it \"non-standard\" precisely because it breaks from the "
            "Gartley-style retracement grid every other pattern in the family uses. "
            "It's the reason charting platforms label it separately from the rest of "
            "the \"harmonic\" zoo, and traders treat its completion zone (not a single "
            "point) as an extra layer of precision."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the *only* pattern in the zoo built entirely from overshoots — no "
            "retracement leg to hide behind — genuinely marks a reversal, that would "
            "be a striking, mechanically testable edge, and arguably more defensible "
            "than its retracement-based cousins (a real *zone*, rather than a lucky "
            "single number, sounds like it should be more robust). And harmonic "
            "trading remains popular precisely because completed patterns look uncanny "
            "in hindsight — a trader who already knows the outcome will always find "
            "the Shark that worked.\n\n"
            "So we removed the human: a mechanical rule finds every qualifying "
            "X-A-B-C on the tape, in real time, with no knowledge of the future — then "
            "we measure what actually happened once price entered the zone."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The basket.** {', '.join(R['basket'])} — six liquid names, daily bars, "
            f"2001/2010 → {R['asof']}.\n"
            "- **The rule, mechanically.** A confirmed zig-zag finds swing pivots (only "
            "known once price has reversed enough to lock them in). Every qualifying "
            "X-A-B-C projects a D-ZONE; we scan forward up to 120 sessions for the "
            "first bar whose range overlaps it.\n"
            "- **The trade.** Fade at the touch — bet on the reversal the pattern "
            "predicts — and measure the forward return.\n"
            "- **Two honest baselines, announced up front.** (1) A **random-day base "
            "rate**: same ticker, same number of trades, same bullish/bearish mix, "
            "just on random days — kills the \"stocks drift up anyway\" confound. "
            "(2) A **placebo completion zone**: the identical pivots, but the zone "
            "re-centered elsewhere — kills the \"any nearby completion zone would "
            "work\" confound. And because we test six tickers, we **correct for "
            "that** (Bonferroni) before calling anything real."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Fade the D-zone touch, hold 5 days, average "
            "across every qualifying Shark on the basket."
        ),
        code(
            "if HAVE_REAL:\n"
            "    frames = []\n"
            "    for t in data.BASKET:\n"
            "        _, _, ledger = st.detect_and_scan(BARS[t], pct=PCT, cost_bps=0.0)\n"
            "        if not ledger.empty:\n"
            "            frames.append(ledger)\n"
            "    SHARK = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()\n"
            "    s5 = st.summarize(SHARK, 'ret_gross_5')\n"
            "    n5, m5, t5 = s5['n'], s5['mean_bps'], s5['t']\n"
            "else:\n"
            "    n5, m5, t5 = R['n_shark'], R['shark'][5]['mean'], R['shark'][5]['t']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['Shark D-zone touch\\n(5-day fade)'], [m5], color=AMBER, width=.5)\n"
            "ax.annotate(f'{m5:+.0f} bps\\n(n={n5}, t={t5:+.2f})', (0, m5), ha='center',\n"
            "            va='bottom' if m5 >= 0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean forward return (bps)')\n"
            "ax.set_title('The headline number — small, positive, not significant')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'n={n5}  mean={m5:+.2f}bps  t={t5:+.2f}')"
        ),
        md(
            f"**{R['shark'][5]['mean']:+.0f} bps** on average, *t* = "
            f"**{R['shark'][5]['t']:.2f}** — positive, but a *t* that small could "
            "easily be zero. If the Shark's \"5-0\" zone were a genuine reversal "
            "signal, we'd want to see this number stand well clear of noise on its "
            "own, before any control is even applied. It doesn't.\n\n"
            "**Second, the honest baseline.** Stocks drift up. A random buy on a "
            "random day, matched to the same bullish/bearish mix as our Shark trades, "
            "earns *something* just from that drift. Does the Shark beat even that?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    shark_by_t, base_frames = {}, []\n"
            "    for t in data.BASKET:\n"
            "        _, _, l = st.detect_and_scan(BARS[t], pct=PCT, cost_bps=0.0)\n"
            "        shark_by_t[t] = l\n"
            "        if len(l):\n"
            "            mix = float((l['reversal_dir'] > 0).mean())\n"
            "            for s in range(20):\n"
            "                base_frames.append(st.base_rate_ledger(BARS[t], len(l), mix,\n"
            "                    seed=st._seed_from(702, t, s)))\n"
            "    BASE = pd.concat(base_frames, ignore_index=True) if base_frames else pd.DataFrame()\n"
            "    sb = st.summarize(BASE, 'ret_gross_5')\n"
            "    bm, wt = sb['mean_bps'], st.welch_t(SHARK['ret_gross_5'], BASE['ret_gross_5'])\n"
            "else:\n"
            "    bm, wt = R['base'][5]['mean'], R['base'][5]['t']\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.4))\n"
            "ax.bar(['Shark D-zone', 'random-day\\nbase rate'], [m5, bm], color=[AMBER, GREY], width=.55)\n"
            "for i, v in enumerate([m5, bm]):\n"
            "    ax.annotate(f'{v:+.0f} bps', (i, v), ha='center', va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean 5-day return (bps)')\n"
            "ax.set_title(f'No real edge over a matched-direction random day (Welch t = {wt:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Shark {m5:+.1f} bps vs base rate {bm:+.1f} bps  Welch t={wt:+.2f}')"
        ),
        md(
            f"The Shark does look a bit better than the random-day baseline "
            f"(Welch *t* = **{R['base'][5]['t']:.2f}**), but that gap is nowhere near "
            "the desk's *t* ≥ 2 bar for calling something real. Look at the sample: "
            f"**{R['n_shark']} touches, spread across six stocks, over 21-25 years** — "
            "and we tested **six different tickers**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    per_t = {t: st.summarize(shark_by_t[t], 'ret_gross_5') for t in data.BASKET}\n"
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
            "Four of six tickers are positive, two negative — SPY's outsized "
            "+181.73 bps is the only one that individually clears a naive *t* ≥ 2, "
            "and it's built from just 11 events. That's the signature of a thin, "
            "noisy sample dominated by one lucky name, not a discovery: correcting "
            f"properly for having tested six tickers, **{R['bonf_survive']} of "
            f"{R['bonf_n_tests']}** results survive.\n\n"
            "**Finally, the placebo.** Does the *specific* 0.886-1.13 zone matter, or "
            "would any similarly-shaped completion zone have done just as well?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    plac_frames = []\n"
            "    for t in data.BASKET:\n"
            "        _, _, lp = st.detect_and_scan(BARS[t], pct=PCT, placebo=True, seed=702, cost_bps=0.0)\n"
            "        if not lp.empty:\n"
            "            plac_frames.append(lp)\n"
            "    PLAC = pd.concat(plac_frames, ignore_index=True) if plac_frames else pd.DataFrame()\n"
            "    sp = st.summarize(PLAC, 'ret_gross_5')\n"
            "    pm, wtp = sp['mean_bps'], st.welch_t(SHARK['ret_gross_5'], PLAC['ret_gross_5'])\n"
            "else:\n"
            "    pm, wtp = R['plac'][5]['mean'], R['plac'][5]['t']\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.4))\n"
            "ax.bar(['Shark\\n(0.886-1.13)', 'placebo\\n(random zone)'], [m5, pm],\n"
            "       color=[AMBER, GREY], width=.55)\n"
            "for i, v in enumerate([m5, pm]):\n"
            "    ax.annotate(f'{v:+.0f} bps', (i, v), ha='center', va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean 5-day return (bps)')\n"
            "ax.set_title(f'The specific zone barely matters (Welch t = {wtp:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Shark {m5:+.1f} bps vs placebo {pm:+.1f} bps  Welch t={wtp:+.2f}')"
        ),
        md(
            f"The Shark's exact **{R['shark'][5]['mean']:+.0f} bps** and the "
            f"placebo's **{R['plac'][5]['mean']:+.0f} bps** are barely different — "
            f"the gap is nowhere near statistically real (Welch *t* = "
            f"**{R['plac'][5]['t']:.2f}**), and the real zone only beats its own "
            f"placebo on **{R['n_beats_plac']}/6** tickers. The specific 0.886-1.13 "
            "band — the Shark's whole selling point over an arbitrary nearby zone — "
            "is not doing detectable work."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The fade is small and positive from the first look "
            f"(*t* = {R['shark'][5]['t']:.2f}), does not beat a matched-direction "
            f"random day (*t* = {R['base'][5]['t']:.2f}), and fails a six-ticker "
            f"correction ({R['bonf_survive']}/{R['bonf_n_tests']} survive).\n"
            "- **Tradability — Mirage.** No hold length clears significance, gross or "
            "net; the pattern isn't even monotone with horizon.\n"
            "- **\"Is 0.886-1.13 the Shark's defining edge?\" — Busted.** Move the zone "
            "elsewhere on the identical pivots and you do about the same "
            f"(Welch *t* = {R['plac'][5]['t']:.2f})."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The general lesson: a fancier geometry doesn't buy you a real edge.** "
            "The Shark is structurally the most distinctive pattern in the harmonic "
            "zoo — no retracement leg, a genuine completion zone instead of a point — "
            "and none of that distinctiveness translates into a detectable reversal "
            "on the real tape.\n"
            "- **Sibling studies:** [468-gartley-harmonic](../../468-gartley-harmonic/), "
            "[699-butterfly-harmonic](../../699-butterfly-harmonic/), "
            "[700-bat-harmonic](../../700-bat-harmonic/), "
            "[701-crab-harmonic](../../701-crab-harmonic/) and "
            "[703-cypher-harmonic](../../703-cypher-harmonic/) all reach the same "
            "shape of verdict independently, across very different XABCD geometries.\n\n"
            "*Think the Shark needs its full \"confluence zone\" — several overlapping "
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
            "# The Shark-Harmonic — a quantitative teardown 🔬\n"
            "### Mechanical extension-based XABCD detection · a random-day base rate · "
            "a Bonferroni correction across the basket · a fade timer with costs · a "
            "placebo completion-zone control · a 20-seed synthetic positive control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The claim — **a D-ZONE at 0.886-1.13x the original XA leg, reached only "
            "after B and C both overshoot rather than retrace, marks the \"5-0\" "
            "reversal** — is tested on the tightest mechanical reading a proponent "
            "would accept, with two independent controls (a matched-direction base "
            "rate and a randomized completion-zone placebo) and an explicit "
            "multiple-comparison penalty across the basket.\n\n"
            "> ⚠️ **Data note.** Daily OHLC (2001/2010→2026), yfinance, cached; basket "
            "identical to [699-butterfly-harmonic](../../699-butterfly-harmonic/), "
            "[700-bat-harmonic](../../700-bat-harmonic/), "
            "[701-crab-harmonic](../../701-crab-harmonic/) and "
            "[703-cypher-harmonic](../../703-cypher-harmonic/). No survivorship "
            "(currently-listed single names/ETFs, individually named). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
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
            f"| **Signal** | `NONE` | pooled 5-day D-zone fade **{R['shark'][5]['mean']:+.2f} "
            f"bps** (HAC *t* = **{R['shark'][5]['t']:.2f}** vs 0; Welch *t* = "
            f"**{R['base'][5]['t']:.2f}** vs base rate) — small at every horizon; "
            f"**{R['bonf_survive']}/{R['bonf_n_tests']}** Bonferroni-corrected tests "
            f"survive (critical \\|*t*\\| = {R['bonf_thr']:.2f}) and placebo Welch "
            f"*t* = **{R['plac'][5]['t']:.2f}** |\n"
            f"| **Tradability** | `MIRAGE` | {R['n_shark']} events over 21-25 years "
            "pooled across six names; no hold length clears \\|*t*\\| = 2, gross or "
            f"net, and the point estimate isn't monotone with horizon |\n"
            f"| **Beats a placebo?** | `BUSTED` | beats on {R['n_beats_plac']}/6 "
            f"tickers; pooled Welch *t* = {R['plac'][5]['t']:.2f} |\n\n"
            "> 💡 In plain words: a small, positive number that never clears noise, "
            "no matter which of the desk's controls it's raced against."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Four confirmed swing pivots $X, A, B, C$, with $XA = A - X$, "
            "$AB = B - A$, $BC = C - B$. The Shark grid — the only member of the zoo "
            "with **no retracement leg at all**:\n\n"
            "- $|AB / XA| \\in [1.13, 1.618]$ — B **overshoots** past X, in the same "
            "direction the AB leg travels (a retracement-free structure, unlike "
            "Gartley/Bat/Butterfly/Crab/Cypher, all of which retrace XA for B).\n"
            "- $|BC / AB| \\in [1.618, 2.24]$ — C overshoots a further distance past "
            "point A.\n"
            "- $D \\in [X + 0.886 \\times XA,\\ X + 1.13 \\times XA]$ — a **ZONE**, not "
            "a point, measured against the *original* XA leg — the \"5-0\" completion "
            "band and the defining ratio under test.\n\n"
            "$\\text{reversal\\_dir} = \\text{sign}(XA)$: believers read the B/C "
            "overshoot as exhaustion and expect price to resume the *original* XA "
            "direction once the zone is touched.\n\n"
            "**H₁ (reversal).** Fading at the D-zone touch beats zero and a "
            "random-day base rate.\n"
            "**H₂ (specificity).** The 0.886-1.13 zone beats a re-centered zone "
            "placebo on the identical pivots.\n"
            "**H₃ (robustness).** The result survives correction for testing six "
            "tickers.\n\n"
            "We find **H₁ not supported** (pooled *t* is small and positive but never "
            "clears 2 at any horizon), **H₂ not supported** (placebo Welch *t* = 0.26, "
            "beats placebo on only 3/6 tickers), **H₃ moot** (0/7 tests clear the "
            "Bonferroni bar)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Shark D-zone touches on the same tape can cluster and overlap, so "
            "within-arm means are tested with a **Newey-West (HAC) t** against zero. "
            "The market's positive unconditional drift means a signal-vs-zero test "
            "alone is not enough — the primary Signal test is a **Welch t vs a "
            "random-day base rate** built to match the *same* bullish/bearish "
            "direction mix as the real touches, pooled over 20 seeds (kills the "
            "drift confound). Because the basket is tested seven ways (pooled + 6 "
            "tickers, all 5-day), a **Bonferroni correction** (family-wise alpha "
            "0.05) raises the critical |*t*| from the naive 2.0 to **2.69** before "
            "any test is allowed to \"survive.\" A **second, independent placebo** — "
            "randomized completion-zone centers on the identical pivots — tests H₂ "
            "separately from H₃."
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
            "satisfying the Shark structural bands (both legs EXTENSIONS past the "
            "prior pivot); the D-zone is fully computable the moment C confirms.\n"
            "- **Touch scan.** 120-session forward window from C's confirmation for "
            "the first bar whose high-low range overlaps the D-zone.\n"
            "- **Execution.** Enter the fade at the touch bar's own close; one "
            "round trip = 2 × one-way cost × NAV.\n"
            "- **Signal control.** Random-day base rate, matched direction mix, 20 "
            "seeds; Bonferroni correction across 7 tests.\n"
            "- **Specificity control.** Re-centered zone placebo on identical "
            "pivots.\n"
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
            "    shark_by_t = {}\n"
            "    frames = []\n"
            "    for t in data.BASKET:\n"
            "        _, _, l = st.detect_and_scan(BARS[t], pct=PCT, cost_bps=0.0)\n"
            "        shark_by_t[t] = l\n"
            "        if len(l):\n"
            "            frames.append(l)\n"
            "    SHARK = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()\n"
            "    hs = [1, 5, 10]\n"
            "    means = [st.summarize(SHARK, f'ret_gross_{h}')['mean_bps'] for h in hs]\n"
            "    ts = [st.summarize(SHARK, f'ret_gross_{h}')['t'] for h in hs]\n"
            "    per_t = {t: st.summarize(shark_by_t[t], 'ret_gross_5') for t in data.BASKET}\n"
            "    tick_means = [per_t[t]['mean_bps'] for t in data.BASKET]\n"
            "else:\n"
            "    hs = [1, 5, 10]\n"
            "    means = [R['shark'][h]['mean'] for h in hs]\n"
            "    ts = [R['shark'][h]['t'] for h in hs]\n"
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
            f"> 💡 In plain words: the 5-day pooled fade (**{R['shark'][5]['mean']:.2f} "
            f"bps**, HAC *t* = **{R['shark'][5]['t']:.2f}**) is small and positive, "
            f"and 10-day (*t* = {R['shark'][10]['t']:.2f}) fades further toward zero "
            "rather than growing — not the shape of a genuine, strengthening effect. "
            "The per-ticker split shows four of six tickers positive, but only SPY "
            "individually clears the naive |*t*| ≥ 2 bar, on a thin 11-event sample."
        ),
        md(
            "### 4b · The Signal axis — random-day base rate, Bonferroni-corrected\n\n"
            "20-seed random-day base rate matched on the empirical bullish/bearish "
            "mix of the real touches; Welch *t* vs the Shark ledger, pooled and per "
            "ticker; Bonferroni correction across the resulting 7 tests."
        ),
        code(
            "if HAVE_REAL:\n"
            "    base_frames, base_by_t = [], {}\n"
            "    for t in data.BASKET:\n"
            "        l = shark_by_t[t]\n"
            "        if len(l):\n"
            "            mix = float((l['reversal_dir'] > 0).mean())\n"
            "            bl = pd.concat([st.base_rate_ledger(BARS[t], len(l), mix,\n"
            "                             seed=st._seed_from(702, t, s)) for s in range(20)],\n"
            "                            ignore_index=True)\n"
            "            base_by_t[t] = bl\n"
            "            base_frames.append(bl)\n"
            "    BASE = pd.concat(base_frames, ignore_index=True) if base_frames else pd.DataFrame()\n"
            "    tests = [('pooled', st.welch_t(SHARK['ret_gross_5'], BASE['ret_gross_5']))]\n"
            "    for t in data.BASKET:\n"
            "        if t in base_by_t and len(shark_by_t[t]):\n"
            "            tests.append((t, st.welch_t(shark_by_t[t]['ret_gross_5'], base_by_t[t]['ret_gross_5'])))\n"
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
            "ax.set_ylabel('Welch t (Shark vs base rate)')\n"
            "ax.set_title('Nothing clears the Bonferroni-corrected bar')\n"
            "ax.legend(fontsize=8.5); plt.tight_layout(); plt.show()\n"
            "print({lbl: round(v,2) if np.isfinite(v) else None for lbl, v in tests})\n"
            "print(f'Bonferroni critical |t| = {thr:.2f}')"
        ),
        md(
            f"> 💡 In plain words: only one per-ticker split (SPY, Welch *t* = "
            f"**{R['base_per_instrument']['SPY']['t']:.2f}**) clears the naive bar "
            "of 2 — on just 11 events, the smallest sample of the six. The honest "
            f"bar, once you correct for looking at six tickers, is "
            f"**{R['bonf_thr']:.2f}**, and **{R['bonf_survive']}/{R['bonf_n_tests']}** "
            "tests clear it. This is the desk's multiple-comparison rail doing "
            "exactly its job."
        ),
        md(
            "### 4c · Tradability — the fade timer, net of costs\n\n"
            "Sweep the holding period and the one-way cost; ~4 pooled events/year "
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
            "ax.set_title('No hold length ever clears significance, gross or net')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print({h: (R['timer'][h]['gross'], R['timer'][h]['t_gross']) for h in hs})"
        ),
        md(
            "> 💡 In plain words: costs (5-10 bps round trip) already flip the "
            "10-day hold from marginally positive to marginally negative — there is "
            "no cushion here. The *shape* across horizons is not the smooth decay of "
            "a real, fading effect either: gross returns dip near zero at 10 days "
            "then bounce back up at 20, the signature of sampling noise rather than "
            "a decaying edge. Combined with the Bonferroni result above and a firing "
            "rate of roughly four completions a year pooled across six names, there "
            "is no regime here worth deploying capital against."
        ),
        md(
            "### 4d · The third axis — does the specific zone matter?\n\n"
            "Identical pivot pipeline; each placebo candidate's D-zone *center* is "
            "replaced by a deterministic, seeded, off-band draw."
        ),
        code(
            "if HAVE_REAL:\n"
            "    plac_by_t, plac_frames = {}, []\n"
            "    for t in data.BASKET:\n"
            "        _, _, lp = st.detect_and_scan(BARS[t], pct=PCT, placebo=True, seed=702, cost_bps=0.0)\n"
            "        plac_by_t[t] = lp\n"
            "        if len(lp):\n"
            "            plac_frames.append(lp)\n"
            "    PLAC = pd.concat(plac_frames, ignore_index=True) if plac_frames else pd.DataFrame()\n"
            "    sp = st.summarize(PLAC, 'ret_gross_5')\n"
            "    pm = sp['mean_bps']\n"
            "    wtp = st.welch_t(SHARK['ret_gross_5'], PLAC['ret_gross_5'])\n"
            "    beat_labels, beat_vals = [], []\n"
            "    for t in data.BASKET:\n"
            "        fm = st.summarize(shark_by_t[t], 'ret_gross_5')['mean_bps']\n"
            "        pmi = st.summarize(plac_by_t[t], 'ret_gross_5')['mean_bps']\n"
            "        beat_labels.append(t); beat_vals.append(fm - pmi if np.isfinite(fm) and np.isfinite(pmi) else np.nan)\n"
            "else:\n"
            "    pm, wtp = R['plac'][5]['mean'], R['plac'][5]['t']\n"
            "    beat_labels = list(R['basket'])\n"
            "    beat_vals = [R['per_instrument'][t]['mean'] - R['plac_per_instrument'][t]['mean']\n"
            "                 for t in R['basket']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.0, 4.4))\n"
            "a1.bar(['Shark', 'placebo'], [R['shark'][5]['mean'], pm], color=[AMBER, GREY], width=.55)\n"
            "for i, v in enumerate([R['shark'][5]['mean'], pm]):\n"
            "    a1.annotate(f'{v:+.0f}bps', (i, v), ha='center', va='bottom' if v>=0 else 'top')\n"
            "a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_title(f'Pooled: Welch t = {wtp:+.2f} (not significant)')\n"
            "cols = [GREEN if (np.isfinite(v) and v > 0) else RED for v in beat_vals]\n"
            "a2.bar(beat_labels, [v if np.isfinite(v) else 0 for v in beat_vals], color=cols, width=.6)\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_title('Shark minus placebo, per ticker')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Shark {R[\"shark\"][5][\"mean\"]:+.1f} bps vs placebo {pm:+.1f} bps  Welch t={wtp:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the Shark beats its own placebo on only "
            f"**{R['n_beats_plac']}/6** tickers — a coin flip — and the pooled "
            f"Welch *t* (**{R['plac'][5]['t']:.2f}**) is nowhere near significant. "
            "The exact 0.886-1.13 completion zone — the Shark's whole selling point, "
            "a supposedly more precise *zone* rather than a lucky single number — is "
            "not demonstrably better than an arbitrary nearby zone built from the "
            "*same* pivots — H₂ not supported."
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
            "null_ts = np.array([st.summarize(pooled_synth(0.0, 702 + s), 'ret_gross_5')['t']\n"
            "                     for s in range(20)], dtype=float)\n"
            "planted = pooled_synth(R['syn_planted_mean_rev'], 702)\n"
            "planted_t = st.summarize(planted, 'ret_gross_5')['t']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (mean_rev=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=GREEN, s=90, zorder=5,\n"
            "           label=f\"planted mean_rev={R['syn_planted_mean_rev']}\")\n"
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
            f"nominal ~5% false-positive rate — and a strong planted mean-reversion "
            f"tendency lights up sharply (*t* = **{R['syn_planted_t']:.2f}**). Note "
            "the sign runs *opposite* the fade's own `reversal_dir = sign(XA)` "
            "convention: the synthetic world's reversion is a generic pull toward a "
            "moving average, not specifically anchored to the Shark's X-A geometry, "
            "so this is a **power** proof (the pipeline can detect a real reversion "
            "tendency when it exists), not a validation of the directional story. "
            "The detection + inference pipeline is unbiased; the real-tape result's "
            "near-zero signal is genuine, not a bug in the machinery."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — pooled 5-day fade **{R['shark'][5]['mean']:.2f} "
            f"bps** (HAC *t* = **{R['shark'][5]['t']:.2f}**; Welch *t* vs base rate = "
            f"**{R['base'][5]['t']:.2f}**) on **{R['n_shark']}** touches over 21-25 "
            f"years; **{R['bonf_survive']}/{R['bonf_n_tests']}** Bonferroni-corrected "
            f"tests survive (critical \\|*t*\\| = {R['bonf_thr']:.2f}); the point "
            "estimate is not even robust across horizon.\n"
            "- **Tradability `MIRAGE`** — no hold length (1/3/5/10/20 days) clears "
            "|*t*| = 2, gross or net; costs already flip the 10-day hold negative, "
            "and the shape across horizons is noise, not decay.\n"
            f"- **\"Is 0.886-1.13 the Shark's defining edge?\" `BUSTED`** — beats the "
            f"placebo on only {R['n_beats_plac']}/6 tickers (a coin flip), pooled "
            f"Welch *t* = **{R['plac'][5]['t']:.2f}** — not statistically "
            "demonstrable."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general lesson: structural novelty isn't evidence.** The Shark "
            "is the most geometrically distinctive pattern in the harmonic zoo — "
            "extension-only legs, a genuine completion zone — and none of that "
            "distinctiveness produces a detectable edge once tested mechanically "
            "against honest controls. \"Different-looking\" is not \"different-"
            "working.\"\n"
            "- **Where a believer would push back:** Carney's full method requires a "
            "*confluence zone* (several overlapping Fibonacci projections converging "
            "near D), not the bare two-ratio-plus-zone skeleton tested here. That's a "
            "legitimate next experiment — but it adds researcher degrees of freedom "
            "(which projections? how many must converge?) that would need their own "
            "placebo before counting as evidence.\n"
            "- **Dedup map:** [468-gartley-harmonic](../../468-gartley-harmonic/) "
            "(every leg a retracement, D never leaves X-A), "
            "[698-abcd-harmonic](../../698-abcd-harmonic/) (no X point, no "
            "confluence at all), [699-butterfly-harmonic](../../699-butterfly-harmonic/) "
            "(fixed 0.786 B retracement, D a 1.27-1.618 range past X), "
            "[700-bat-harmonic](../../700-bat-harmonic/) (D stays inside X-A), "
            "[701-crab-harmonic](../../701-crab-harmonic/) (a single sharp 1.618 "
            "point target) and [703-cypher-harmonic](../../703-cypher-harmonic/) (D "
            "measured off the XC leg, not XA) — all independently converging on the "
            "same verdict shape across wildly different XABCD geometries.\n\n"
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
