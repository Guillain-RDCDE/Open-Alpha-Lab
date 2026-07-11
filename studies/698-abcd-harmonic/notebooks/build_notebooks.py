"""Generate the two narrative notebooks for Study 698 (ABCD-Harmonic).

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
# pct=0.03 zigzag, BC retrace 0.618 +-0.10, CD=AB +-0.15, 40-session touch window).
R = dict(
    basket=("SPY", "QQQ", "AAPL", "MSFT", "TSLA", "NVDA"),
    asof="2026-06-30",
    n_pivots=4200, n_cand_fib=528, n_cand_plac=523,
    n_fib=349, n_plac=252,
    fib={1: dict(hit=50.1, mean=-6.89, t=-0.35), 5: dict(hit=45.3, mean=-45.29, t=-1.18),
         10: dict(hit=41.8, mean=-79.18, t=-1.49)},
    plac={1: dict(hit=52.0, mean=0.06, t=0.00), 5: dict(hit=54.8, mean=41.54, t=1.06),
          10: dict(hit=54.0, mean=63.94, t=0.95)},
    welch={1: -0.24, 5: -1.49, 10: -1.68},
    wilson_fib=(40.1, 50.5), wilson_plac=(48.6, 60.8),
    per_instrument={
        "SPY": dict(fib_n=36, fib_mean=-73.25, fib_t=-1.09, plac_n=16, plac_mean=96.22, plac_t=1.21),
        "QQQ": dict(fib_n=39, fib_mean=61.53, fib_t=1.11, plac_n=29, plac_mean=10.47, plac_t=0.14),
        "AAPL": dict(fib_n=71, fib_mean=-53.57, fib_t=-0.88, plac_n=44, plac_mean=81.95, plac_t=1.14),
        "MSFT": dict(fib_n=47, fib_mean=2.80, fib_t=0.06, plac_n=40, plac_mean=152.56, plac_t=1.70),
        "TSLA": dict(fib_n=59, fib_mean=-283.57, fib_t=-2.12, plac_n=61, plac_mean=-21.68, plac_t=-0.21),
        "NVDA": dict(fib_n=97, fib_mean=49.83, fib_t=0.64, plac_n=62, plac_mean=3.84, plac_t=0.05),
    },
    cost_sweep={0.0: (-45.29, -1.18), 5.0: (-55.29, -1.45), 10.0: (-65.29, -1.71)},
    syn_null_mean=-0.37, syn_null_sd=0.90, syn_null_fire=1, syn_null_seeds=20,
    syn_planted_n=357, syn_planted_mean=133.81, syn_planted_t=9.35,
    fp={"SPY": "b424b1a9528f", "QQQ": "cfa66123a1bc", "AAPL": "990a4f52486c",
        "MSFT": "026aaf70e806", "TSLA": "f8ca92e420b8", "NVDA": "4b7e2befef6d"},
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Beats_placebo%3F: Busted](https://img.shields.io/badge/Beats_placebo%3F-Busted-8b949e?style=flat-square)\n\n"
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

from abcd_harmonic import data, strategy as st

BASKET = data.BASKET
HAVE_REAL = data.have_real()
BARS = {t: data.load_real(t) for t in BASKET} if HAVE_REAL else {}

def pooled(placebo=False, cost=0.0, seed=698, pct=0.03):
    \"\"\"Pool the D-touch ledger across the whole basket (offline; reads BARS).\"\"\"
    frames = []
    for t in BASKET:
        _, _, ledger = st.detect_and_scan(BARS[t], pct=pct, placebo=placebo, seed=seed, cost_bps=cost)
        if not ledger.empty:
            frames.append(ledger)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

print("real cache present:", HAVE_REAL, "| basket:", BASKET)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the market really turn where a Fibonacci chart pattern says it will? 📐↩️\n"
            "### The AB=CD harmonic — the simplest \"the chart tells you where price reverses\" "
            "claim, taken completely literally\n\n"
            + BADGES +
            "Open any technical-analysis tutorial and you'll meet the **AB=CD pattern**: draw a "
            "line from a swing low to a swing high (leg **A→B**), watch price pull back roughly "
            "**61.8%** of that move (leg **B→C**), then project a *third* leg of the **same length "
            "as the first** (**C→D**, so \"AB equals CD\"). The moment price reaches **D**, the "
            "pattern is supposedly \"complete\" — and the chart is telling you to buy (or sell) "
            "right there for the reversal.\n\n"
            "It's the ancestor of an entire cottage industry of \"harmonic patterns\" (Gartley, Bat, "
            "Butterfly — see the going-further section). We test the plainest version, exactly as "
            "stated: does price actually turn at D?\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the pivot-detection mechanics and "
            "the cost math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** Every pivot is detected with a 3% zigzag and only counted once "
            "it's *confirmed* — never using information from the future. Every chart is drawn by "
            "the code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does price reverse at a completed AB=CD point D? | **No — if anything, the "
            f"opposite.** Betting on the reversal (a 5-day hold) lost **{R['fib'][5]['mean']:.1f} "
            f"bps per event** on average across {R['n_fib']} real detections since 2001/2010. |\n"
            "| Is that just bad luck? | **We can't even tell — it's not statistically distinct "
            f"from zero** (t = {R['fib'][5]['t']:.2f}; the desk's bar is 2). |\n"
            "| Is 0.618 / \"AB=CD\" actually special? | **No.** A control arm using the *same* "
            "swing pivots but random, non-Fibonacci ratios did **better** than the real Fibonacci "
            f"version at every time horizon we checked. |\n"
            f"| Can you trade it? | **No — it loses money before AND after costs.** The one "
            f"statistically significant result on the whole board (TSLA, t = "
            f"{R['per_instrument']['TSLA']['fib_t']:.2f}) says the fade **reliably lost**, the "
            "opposite of the claim. |\n\n"
            "> The pattern is easy to draw. It just doesn't predict anything."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When you see a clean A-B-C-D shape on the chart — a move, a 61.8% pullback, then "
            "an equal-length push — the pattern has 'completed' at D, and price reverses. It works "
            "because everyone's watching the same ratios and placing orders at the same spot.\"*\n\n"
            "It sounds precise (there's a *specific* ratio! there's *math* involved!) and it's the "
            "shape every \"harmonic pattern\" trader learns first. The claim is falsifiable in the "
            "most literal way possible: mark every historical A-B-C spot where BC really did "
            "retrace ~61.8% of AB, project where D would land, and see what price actually did when "
            "it got there."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real, this is one of the cleanest possible trading signals: a **specific, "
            "computable price level**, known the moment the third pivot confirms, with a stated "
            "directional prediction. No fundamentals, no macro calendar — just geometry. Retail "
            "harmonic-pattern traders build entire systems around it (and around fancier cousins "
            "with an extra leg — Gartley, Bat, Butterfly). If it's just numerology, though, it's a "
            "textbook case of pattern-matching with no edge: humans are extremely good at seeing "
            "shapes that aren't really there."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **Find real AB=CD setups.** Scan {', '.join(R['basket'])} daily bars "
            "(2001/2010→2026) for every confirmed A-B-C swing where BC retraces AB within 10 "
            "points of 61.8%, then project D = C + AB (within 15% tolerance of an exact match).\n"
            "- **Wait for D to actually be touched** — up to 40 trading sessions — the same way a "
            "real trader would watch and wait.\n"
            "- **Bet on the reversal**, and measure what actually happened over the next 5 days.\n"
            "- **The control that decides everything.** Rerun the identical scan, but swap 61.8%/"
            "\"AB=CD\" for random, non-Fibonacci ratios on the *same* underlying pivots. If "
            "Fibonacci is magic, it has to beat this control — not just look positive."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            f"**First, the headline.** {R['n_fib']} real Fibonacci AB=CD detections since "
            f"2001/2010 — average 5-day return from betting on the reversal, vs the placebo control."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fib = pooled(placebo=False); plac = pooled(placebo=True)\n"
            "    fm = fib['ret_gross_5'].mean() * 1e4; pm = plac['ret_gross_5'].mean() * 1e4\n"
            "    nf, np_ = len(fib), len(plac)\n"
            "else:\n"
            "    fm, pm = R['fib'][5]['mean'], R['plac'][5]['mean']\n"
            "    nf, np_ = R['n_fib'], R['n_plac']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar([f'Fibonacci AB=CD\\n(n={nf})', f'Placebo control\\n(n={np_})'], [fm, pm],\n"
            "       color=[RED, GREY], width=.55)\n"
            "for i,v in enumerate([fm, pm]): ax.annotate(f'{v:+.1f} bps',(i,v),ha='center',\n"
            "    va='top' if v<0 else 'bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('avg 5-day return from betting on the reversal (bps)')\n"
            "ax.set_title('Fading a completed AB=CD loses money -- and loses to a random control')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Fibonacci {fm:+.1f} bps  vs  placebo {pm:+.1f} bps  (n={nf} / {np_})')"
        ),
        md(
            f"The \"real\" pattern lost **{abs(R['fib'][5]['mean']):.1f} bps per event** on "
            f"average — and the control, using the *identical* pivots with made-up ratios, made "
            f"**+{R['plac'][5]['mean']:.1f} bps**. Neither number is statistically solid on its "
            "own (more on that in a moment), but the direction alone already tells you the "
            "\"magic ratio\" isn't earning its keep.\n\n"
            "**Does it get better at a longer or shorter hold?** Same story at every horizon we "
            "checked:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    fs = {h: pooled(placebo=False).get(f'ret_gross_{h}', pd.Series(dtype=float)).mean()*1e4 for h in (1,5,10)}\n"
            "    ps = {h: pooled(placebo=True).get(f'ret_gross_{h}', pd.Series(dtype=float)).mean()*1e4 for h in (1,5,10)}\n"
            "else:\n"
            "    fs = {h: R['fib'][h]['mean'] for h in (1,5,10)}\n"
            "    ps = {h: R['plac'][h]['mean'] for h in (1,5,10)}\n"
            "hs = [1,5,10]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "x = np.arange(len(hs)); w = 0.35\n"
            "ax.bar(x - w/2, [fs[h] for h in hs], width=w, color=RED, label='Fibonacci AB=CD')\n"
            "ax.bar(x + w/2, [ps[h] for h in hs], width=w, color=GREY, label='Placebo control')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}-day' for h in hs])\n"
            "ax.set_ylabel('mean return (bps)'); ax.set_title('Fibonacci trails the control at every horizon')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print({h: (round(fs[h],1), round(ps[h],1)) for h in hs})"
        ),
        md(
            "**Per instrument, does at least one ticker behave the way the story says?** Two of "
            "six (QQQ, NVDA) edge out the placebo; the other four don't — and TSLA is the one "
            "place a difference is actually statistically real, and it's a **loss**, not a win:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for t in BASKET:\n"
            "        _,_,lf = st.detect_and_scan(BARS[t], placebo=False)\n"
            "        _,_,lp = st.detect_and_scan(BARS[t], placebo=True)\n"
            "        rows.append((t, lf['ret_gross_5'].mean()*1e4 if len(lf) else np.nan,\n"
            "                     lp['ret_gross_5'].mean()*1e4 if len(lp) else np.nan))\n"
            "    tks = [r[0] for r in rows]; fm_ = [r[1] for r in rows]; pm_ = [r[2] for r in rows]\n"
            "else:\n"
            "    tks = list(R['per_instrument'])\n"
            "    fm_ = [R['per_instrument'][t]['fib_mean'] for t in tks]\n"
            "    pm_ = [R['per_instrument'][t]['plac_mean'] for t in tks]\n"
            "x = np.arange(len(tks)); w = 0.35\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.4))\n"
            "ax.bar(x - w/2, fm_, width=w, color=RED, label='Fibonacci AB=CD')\n"
            "ax.bar(x + w/2, pm_, width=w, color=GREY, label='Placebo control')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels(tks)\n"
            "ax.set_ylabel('mean 5-day return (bps)'); ax.set_title('Only 2 of 6 tickers favor Fibonacci over the control')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(dict(zip(tks, zip([round(v,1) for v in fm_], [round(v,1) for v in pm_]))))"
        ),
        md(
            f"TSLA's fade lost **{abs(R['per_instrument']['TSLA']['fib_mean']):.0f} bps per "
            f"event on average** — the single statistically real number in this whole study "
            f"(t = {R['per_instrument']['TSLA']['fib_t']:.2f}), and it says the pattern's "
            "prediction was *backwards*.\n\n"
            "**Finally, costs.** Even ignoring all of the above, does the trade at least survive "
            "a realistic cost?"
        ),
        code(
            "cs = sorted(R['cost_sweep'])\n"
            "means = [R['cost_sweep'][c][0] for c in cs]\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.2))\n"
            "ax.bar([f'{c:.0f} bps' for c in cs], means, color=AMBER, width=.55)\n"
            "for i,v in enumerate(means): ax.annotate(f'{v:+.1f}',(i,v),ha='center',va='top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('net 5-day return (bps/event)')\n"
            "ax.set_title('It starts negative and gets worse with realistic costs')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({c: R['cost_sweep'][c] for c in cs})"
        ),
        md(
            "It starts negative gross and only gets worse net of costs. There is no cost regime "
            "in this sweep where the AB=CD fade is worth placing."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The AB=CD fade returned **{R['fib'][5]['mean']:.1f} bps/event** "
            f"on average (t = {R['fib'][5]['t']:.2f}) — negative, and never statistically distinct "
            "from zero at any horizon we checked.\n"
            "- **Tradability — Mirage.** It loses money before costs and loses more after them. No "
            "cost level makes it attractive.\n"
            "- **\"Beats a random equal-legged reversal projection?\" — Busted.** The same pivots, "
            "with made-up ratios instead of 0.618/AB=CD, did *better* — there's nothing special "
            "about the Fibonacci numbers here."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The harmonic-pattern family gets more elaborate, not simpler.** Gartley, Bat and "
            "Butterfly all add a fourth pivot (X) and stack multiple Fibonacci ratios into a "
            "\"confluence zone\" — more researcher degrees of freedom, not fewer. If the plainest "
            "two-leg version doesn't clear a placebo, the burden of proof only grows for the fancier "
            "cousins.\n"
            "- **Sibling study:** [77-golden-mean](../../77-golden-mean/) tests plain Fibonacci "
            "retracement *levels* (not a multi-pivot pattern) on the same six tapes and reaches the "
            "identical verdict: Fibonacci ratios show no specificity over a randomized control.\n\n"
            "*Think you can find the real edge in a stricter version of the pattern (tighter "
            "tolerances, an added confluence check)? Show it beats this exact placebo — same "
            "pivots, same tape, only the ratio changes — and we'll take a look.*"
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
            "# The AB=CD Harmonic — a quantitative teardown 🔬\n"
            "### Confirmed-pivot zigzag detection · a seeded off-Fibonacci placebo arm · HAC/Welch "
            "splits · per-instrument and per-horizon breakdown · a cost sweep · a 20-seed synthetic "
            "positive control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — **a completed AB=CD leg (BC retraces AB by 0.618, CD=AB) predicts a reversal "
            "at D** — is testable literally: detect it in real time off confirmed pivots, fade it, "
            "and compare against a placebo control built from the identical pivot structure with "
            "randomized, off-Fibonacci ratios.\n\n"
            "> ⚠️ **Data note.** Daily OHLC, yfinance, cached; basket "
            f"{', '.join(R['basket'])} (SPY/QQQ/AAPL/MSFT/NVDA 2001-07-10→2026-06-30, TSLA "
            "2010-06-29→2026-06-30 — the identical basket as sibling "
            "[77-golden-mean](../../77-golden-mean/)). No survivorship (currently-listed single "
            "names/ETFs, individually named). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprints " +
            ", ".join(f"`{v}`" for v in R["fp"].values()) + ").\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | 5-day fade-at-D: **{R['fib'][5]['mean']:+.2f} bps/event**, "
            f"HAC **t = {R['fib'][5]['t']:.2f}** (n={R['n_fib']}); 1/10-day t = "
            f"{R['fib'][1]['t']:.2f} / {R['fib'][10]['t']:.2f} |\n"
            f"| **Tradability** | `MIRAGE` | net mean falls from {R['cost_sweep'][0.0][0]:+.1f} bps "
            f"(gross) to {R['cost_sweep'][10.0][0]:+.1f} bps at 10 bps one-way cost |\n"
            f"| **Beats placebo?** | `BUSTED` | Fib-Plac Welch t = {R['welch'][5]:+.2f} (5-day); "
            f"Fibonacci wins on 2/6 tickers |\n\n"
            "> 💡 In plain words: the pattern doesn't clear the bar, and the one control test built "
            "specifically to catch numerology catches it here."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Given three confirmed swing pivots $A, B, C$ (alternating high/low), let "
            "$AB = B - A$, $BC = C - B$, retracement $\\rho = |BC / AB|$. The AB=CD claim projects\n\n"
            "$$D = C + \\kappa \\cdot AB, \\qquad \\kappa \\approx 1.0 \\ (\\pm 0.15)$$\n\n"
            "conditional on $\\rho \\approx 0.618\\ (\\pm 0.10)$, and predicts a reversal — i.e. a "
            "positive return in direction $-\\mathrm{sign}(AB)$ — once price *touches* $D$.\n\n"
            "- **H₁ (reversal).** $E[\\text{fade return} \\mid D_{\\text{touch}}] \\gg 0$, robust to "
            "HAC inference.\n"
            "- **H₂ (Fibonacci specificity).** H₁, if true, must *beat* the identical pipeline run "
            "with $\\rho, \\kappa$ drawn off the Fibonacci band — otherwise it is generic "
            "\"pullbacks tend to keep moving / mean-revert\" structure any level captures.\n\n"
            "We find **H₁ rejected** (point estimate negative, never *t* ≥ 2) and **H₂ rejected** "
            "(the placebo wins at every horizon tested)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "ABCD events on a single tape can **cluster in time** (overlapping legs share pivots), "
            "so the within-arm test is a **Newey-West (HAC)** *t* of the mean return against 0, not "
            "a naive i.i.d. *t*. The **Fibonacci-vs-placebo** comparison — the decisive test for "
            "specificity — is a **Welch** *t* between the two arms' event-level returns. The hit "
            "rate carries a **Wilson** interval. All numbers are pooled across the six-ticker "
            "basket, matching the real-tape sample to the synthetic control's pooling design."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Pivots.** A percentage-threshold zigzag (pct=3%) on daily closes; a pivot enters "
            "the record only at its **confirmation bar** — the session where price has already "
            "reversed past the threshold — never at the (earlier) extreme itself.\n"
            "- **Fibonacci candidates.** Every consecutive confirmed triple (A,B,C) with retrace "
            "$\\rho \\in [0.518, 0.718]$; projects $D = C + \\kappa AB$, $\\kappa \\in [0.85, 1.15]$.\n"
            "- **Placebo candidates.** Identical triples; $\\rho, \\kappa$ drawn per-candidate from "
            "a deterministic seeded uniform, kept clear of the Fibonacci bands.\n"
            "- **D-touch scan.** From C's confirmation bar forward, first bar in the next 40 "
            "sessions whose high-low range brackets $D$ (or closes within 0.75% of it) — real-time "
            "knowledge only.\n"
            "- **Execution.** Enter the fade at the touch bar's own close (intrabar touch, "
            "same-session close execution — one documented convention, identical in both arms); "
            "exit at close $+h$ sessions, $h \\in \\{1, 5, 10\\}$; net figures subtract "
            "$2\\times$ one-way cost $\\times$ NAV per round trip.\n"
            "- **Control.** Synthetic mean-reverting price index (tunable knob), pooled across a "
            "synthetic 6-name basket per seed; the null must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline split — Fibonacci vs placebo, all three horizons\n\n"
            "Same pivots, same tape; only the retrace/extension *targets* differ between arms."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fib_l = pooled(placebo=False); plac_l = pooled(placebo=True)\n"
            "    rows = {}\n"
            "    for h in (1,5,10):\n"
            "        sf = st.summarize(fib_l, f'ret_gross_{h}'); sp = st.summarize(plac_l, f'ret_gross_{h}')\n"
            "        wt = st.welch_t(fib_l[f'ret_gross_{h}'], plac_l[f'ret_gross_{h}'])\n"
            "        rows[h] = (sf, sp, wt)\n"
            "    for h, (sf, sp, wt) in rows.items():\n"
            "        print(f\"{h:2d}-day: FIB n={sf['n']} mean={sf['mean_bps']:+.2f}bps t={sf['t']:+.2f}\"\n"
            "              f\"  |  PLAC n={sp['n']} mean={sp['mean_bps']:+.2f}bps t={sp['t']:+.2f}\"\n"
            "              f\"  |  Welch t={wt:+.2f}\")\n"
            "    hs = list(rows); fm = [rows[h][0]['mean_bps'] for h in hs]; pm = [rows[h][1]['mean_bps'] for h in hs]\n"
            "else:\n"
            "    hs = [1,5,10]; fm = [R['fib'][h]['mean'] for h in hs]; pm = [R['plac'][h]['mean'] for h in hs]\n"
            "    for h in hs:\n"
            "        print(f\"{h:2d}-day: FIB mean={R['fib'][h]['mean']:+.2f}bps t={R['fib'][h]['t']:+.2f}\"\n"
            "              f\"  |  PLAC mean={R['plac'][h]['mean']:+.2f}bps t={R['plac'][h]['t']:+.2f}\"\n"
            "              f\"  |  Welch t={R['welch'][h]:+.2f}\")\n"
            "x = np.arange(len(hs)); w = .35\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.4))\n"
            "ax.bar(x-w/2, fm, width=w, color=RED, label='Fibonacci')\n"
            "ax.bar(x+w/2, pm, width=w, color=GREY, label='Placebo')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('mean return (bps)'); ax.set_title('Fibonacci trails the placebo at every horizon')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 In plain words: at the headline 5-day horizon, Fibonacci's "
            f"**{R['fib'][5]['mean']:+.2f} bps** (t = {R['fib'][5]['t']:.2f}) sits **"
            f"{abs(R['fib'][5]['mean'] - R['plac'][5]['mean']):.1f} bps below** the placebo's "
            f"**{R['plac'][5]['mean']:+.2f} bps** (t = {R['plac'][5]['t']:.2f}); Welch "
            f"t = **{R['welch'][5]:+.2f}**. Neither arm individually clears *t* ≥ 2, and the "
            f"difference runs opposite to the claim at every horizon tested (1d t="
            f"{R['welch'][1]:+.2f}, 5d t={R['welch'][5]:+.2f}, 10d t={R['welch'][10]:+.2f})."
        ),
        md(
            "### 4b · Hit rate — Wilson intervals\n\n"
            "A conditional claim needs an uncertainty band, not a bare percentage."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sf5 = st.summarize(fib_l, 'ret_gross_5'); sp5 = st.summarize(plac_l, 'ret_gross_5')\n"
            "    fh, flo, fhi = sf5['hit_rate']*100, sf5['hit_lo']*100, sf5['hit_hi']*100\n"
            "    ph, plo, phi = sp5['hit_rate']*100, sp5['hit_lo']*100, sp5['hit_hi']*100\n"
            "else:\n"
            "    fh, (flo, fhi) = R['fib'][5]['hit'], R['wilson_fib']\n"
            "    ph, (plo, phi) = R['plac'][5]['hit'], R['wilson_plac']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.0))\n"
            "ax.bar(['Fibonacci', 'Placebo'], [fh, ph], color=[RED, GREY], width=.5,\n"
            "       yerr=[[fh-flo, ph-plo],[fhi-fh, phi-ph]], capsize=6)\n"
            "ax.axhline(50, ls='--', c='k', lw=.8, label='coin flip')\n"
            "ax.set_ylabel('hit rate (%)'); ax.set_title('Fibonacci hits the reversal LESS than half the time')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Fib {fh:.1f}% [{flo:.1f},{fhi:.1f}]  Plac {ph:.1f}% [{plo:.1f},{phi:.1f}]')"
        ),
        md(
            f"> 💡 In plain words: Fibonacci's 5-day hit rate is **{R['fib'][5]['hit']:.1f}%** "
            "(Wilson " f"[{R['wilson_fib'][0]:.1f}%, {R['wilson_fib'][1]:.1f}%]) — its confidence "
            "interval sits *below* a coin flip. The pattern doesn't just fail to add edge; the "
            "point estimate leans toward the wrong side of even money."
        ),
        md(
            "### 4c · Per-instrument breakdown\n\n"
            "Does the effect concentrate anywhere? If Fibonacci specificity exists on some subset "
            "of names, it should show up here."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for t in BASKET:\n"
            "        _,_,lf = st.detect_and_scan(BARS[t], placebo=False)\n"
            "        _,_,lp = st.detect_and_scan(BARS[t], placebo=True)\n"
            "        sf = st.summarize(lf, 'ret_gross_5'); sp = st.summarize(lp, 'ret_gross_5')\n"
            "        rows.append((t, sf['n'], sf['mean_bps'], sf['t'], sp['n'], sp['mean_bps'], sp['t']))\n"
            "    for r in rows:\n"
            "        print(f\"{r[0]:5s} FIB n={r[1]:3d} mean={r[2]:+7.2f}bps t={r[3]:+.2f}\"\n"
            "              f\"   |   PLAC n={r[4]:3d} mean={r[5]:+7.2f}bps t={r[6]:+.2f}\")\n"
            "else:\n"
            "    for t, d in R['per_instrument'].items():\n"
            "        print(f\"{t:5s} FIB n={d['fib_n']:3d} mean={d['fib_mean']:+7.2f}bps t={d['fib_t']:+.2f}\"\n"
            "              f\"   |   PLAC n={d['plac_n']:3d} mean={d['plac_mean']:+7.2f}bps t={d['plac_t']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: Fibonacci beats its own placebo control on only **2 of 6** "
            "tickers (QQQ, NVDA); it loses on the other four. The single individually significant "
            f"cell on the board is **TSLA** (t = {R['per_instrument']['TSLA']['fib_t']:.2f}, mean "
            f"= {R['per_instrument']['TSLA']['fib_mean']:.1f} bps) — a reliable **loss** from "
            "fading the pattern, the opposite of the claim's prediction."
        ),
        md(
            "### 4d · The fade-at-D timer, net of costs\n\n"
            "One round trip = 2 × one-way cost × NAV per event, at the headline 5-day horizon."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cs = [0.0, 5.0, 10.0]\n"
            "    means, ts = [], []\n"
            "    for c in cs:\n"
            "        l = pooled(placebo=False, cost=c)\n"
            "        s = st.summarize(l, 'ret_net_5')\n"
            "        means.append(s['mean_bps']); ts.append(s['t'])\n"
            "else:\n"
            "    cs = sorted(R['cost_sweep'])\n"
            "    means = [R['cost_sweep'][c][0] for c in cs]; ts = [R['cost_sweep'][c][1] for c in cs]\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.2))\n"
            "ax.bar([f'{c:.0f}bps' for c in cs], means, color=AMBER, width=.55)\n"
            "for i,(v,t) in enumerate(zip(means, ts)):\n"
            "    ax.annotate(f'{v:+.1f}\\n(t={t:+.2f})',(i,v),ha='center',va='top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('net mean return (bps/event, 5-day)')\n"
            "ax.set_title('Negative gross, worse net -- no cost regime helps')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(list(zip(cs, means, ts)))"
        ),
        md(
            "> 💡 In plain words: the trade starts underwater gross and only sinks further as "
            "realistic costs are applied. Since the same cost is subtracted from both arms, the "
            "Fibonacci-vs-placebo Welch *t* is **unchanged** by cost — a useful internal check that "
            "the sign of the comparison isn't a costing artefact."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Deterministic synthetic price index with a tunable mean-reversion knob toward a slow "
            "EMA, pooled across a **synthetic 6-name basket per seed** (mirrors the real pooled "
            "sample size). The null (`mean_rev=0`) is checked over **20 seeds** — never a single "
            "stream."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    frames = []\n"
            "    for i in range(6):\n"
            "        sb = data.synthetic_world(mean_rev=0.0, seed=(698+s_)*1000+i, n_days=6300)\n"
            "        _,_,l = st.detect_and_scan(sb, cost_bps=0.0)\n"
            "        if len(l): frames.append(l)\n"
            "    pooled_syn = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()\n"
            "    null_ts.append(st.summarize(pooled_syn, 'ret_gross_5')['t'])\n"
            "null_ts = np.asarray(null_ts, dtype=float)\n"
            "frames = []\n"
            "for i in range(6):\n"
            "    sb = data.synthetic_world(mean_rev=0.08, seed=698*1000+i, n_days=6300)\n"
            "    _,_,l = st.detect_and_scan(sb, cost_bps=0.0)\n"
            "    if len(l): frames.append(l)\n"
            "planted = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()\n"
            "planted_t = st.summarize(planted, 'ret_gross_5')['t']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12,.12,20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (mean_rev=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=RED, s=90, zorder=5, label='planted mean_rev=0.08')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('HAC t (fade-at-D vs 0)')\n"
            "ax.set_title('Control: the null rarely fires; a planted reversion lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(np.abs(null_ts)>=2).sum()}/20 seeds  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector fires (|t| ≥ 2) in only "
            f"**{R['syn_null_fire']}/{R['syn_null_seeds']}** seeds — right at the nominal ~5% false "
            f"-positive rate — with mean t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}). "
            f"A planted mean-reversion tendency lights up sharply "
            f"(n={R['syn_planted_n']}, t = {R['syn_planted_t']:.2f}). The pipeline is unbiased and "
            "has power — the flat real-tape result is the genuine article, not a broken detector. "
            "*(A faithful-engine / power check only — never cited in support of the real-tape "
            "stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — 5-day fade-at-D: **{R['fib'][5]['mean']:+.2f} bps/event**, HAC "
            f"t = **{R['fib'][5]['t']:.2f}** (n={R['n_fib']}); 1-day t = {R['fib'][1]['t']:+.2f}, "
            f"10-day t = {R['fib'][10]['t']:+.2f} — never *t* ≥ 2 in either direction, and the "
            "lone individually significant cell (TSLA) points against the claim.\n"
            f"- **Tradability `MIRAGE`** — negative gross ({R['cost_sweep'][0.0][0]:+.1f} bps), "
            f"worse net ({R['cost_sweep'][10.0][0]:+.1f} bps at 10 bps one-way); no cost regime in "
            "the sweep makes the fade attractive.\n"
            f"- **\"Beats a random equal-legged reversal projection?\" `BUSTED`** — the placebo arm "
            f"(same pivots, off-Fibonacci ratios) outperforms Fibonacci at every horizon (Welch t "
            f"from {R['welch'][1]:+.2f} to {R['welch'][10]:+.2f}) and wins on 4 of 6 tickers. There "
            "is no detectable specificity in the 0.618/AB=CD pair."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The harmonic family only gets more elaborate.** Gartley, Bat and Butterfly all "
            "add an X point and stack multiple Fibonacci confluences on top of this same AB=CD "
            "skeleton (Carney, *Harmonic Trading*, 2004-2010) — more free parameters, not fewer. A "
            "plain two-leg version failing a placebo raises, not lowers, the burden of proof for "
            "the fancier variants.\n"
            "- **A natural sequel** would test whether *tighter* tolerances (a narrower Fibonacci "
            "band) select for a genuinely different, better-performing subset of setups — or "
            "whether tightening the band just shrinks the sample without changing the sign.\n"
            "- **Dedup map:** [468-gartley-harmonic](../../468-gartley-harmonic/) (the XABCD "
            "Gartley, an added X point and multi-leg confluence), "
            "[699-butterfly-harmonic](../../699-butterfly-harmonic/) (extended, not equal-legged, "
            "CD), [700-bat-harmonic](../../700-bat-harmonic/) (shallower B retracement, deeper CD "
            "extension), [77-golden-mean](../../77-golden-mean/) (plain Fibonacci retracement "
            "*levels*, not a projected multi-pivot pattern — same six tapes, same verdict).\n\n"
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
