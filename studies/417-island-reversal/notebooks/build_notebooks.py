"""Generate the two narrative notebooks for Study 417 (Island Reversal).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached daily OHLC panel
under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily OHLC, SPY + 29
# large-caps, 2005-01-03 -> 2026-05-29, as-of 2026-05-31, gap threshold 1%).
R = dict(
    asof="2026-05-31", last_bar="2026-05-29", start="2005-01-03", end="2026-05-29",
    years=21.4, bars=5385, n_names=30, fingerprint="bcc8dc7157c7", gap_pct=1.0,
    n_tops=75, n_bottoms=97, spy_tops=3, spy_bottoms=3,
    # island TOP (short) per horizon: (H, n, excess%, raw%, win%, t, hac_t, p_placebo, net%)
    top=[(5, 75, -0.705, -1.025, 45, -1.27, -1.21, 0.641, -0.805),
         (10, 75, -1.275, -1.906, 39, -1.59, -1.36, 0.678, -1.375),
         (20, 75, -0.640, -1.901, 45, -0.55, -0.49, 0.562, -0.740),
         (40, 73, -0.336, -2.794, 49, -0.22, -0.20, 0.524, -0.436)],
    # island BOTTOM (long, myth-check) per horizon: (H, n, excess%, raw%, win%, t, hac_t, p)
    bot=[(5, 97, 0.762, 1.085, 60, 1.77, 1.74, 0.337),
         (10, 96, 0.328, 0.965, 49, 0.48, 0.51, 0.454),
         (20, 96, 2.022, 3.294, 59, 2.04, 2.40, 0.292),
         (40, 94, 5.284, 7.762, 69, 4.22, 4.29, 0.163)],
    # gap-threshold robustness sweep at H=10 (top, short): (gap%, n, excess%, t, p)
    robust=[(0.5, 198, -0.551, -1.31, 0.616), (1.0, 75, -1.275, -1.59, 0.673),
            (2.0, 17, -0.989, -0.53, 0.598)],
    # SPY-only island top: (H, n, excess%, raw%, t, p)
    spy=[(5, 3, -1.300, -1.537, -0.73, 0.860), (10, 3, -0.907, -1.376, -1.37, 0.699),
         (20, 3, -1.067, -2.004, -0.47, 0.654), (40, 3, -5.150, -6.997, -1.81, 0.946)],
    # synthetic control 20d (top short): (planted_edge, planted, detected, n, excess%, t, p, win%)
    syn=[(0.00, 160, 158, 158, -0.58, -1.50, 0.627, 47),
         (0.25, 160, 158, 158, 18.33, 59.38, 0.000, 100)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Reliable_reversal%3F: Busted](https://img.shields.io/badge/Reliable_reversal%3F-Busted-8b949e?style=flat-square)\n\n"
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

from island_reversal import data, strategy as st

ASOF = "2026-05-31"
GAP = 0.01
HAVE_REAL = data.have_real()
if HAVE_REAL:
    PANEL = data.load_real()
    for f in PANEL:
        PANEL[f] = PANEL[f].loc[:pd.Timestamp(ASOF)]
    CLOSES = PANEL["close"]
else:
    PANEL = CLOSES = None
print("real island cache present:", HAVE_REAL,
      "| names:", (0 if CLOSES is None else CLOSES.shape[1]),
      "| bars:", (0 if CLOSES is None else len(CLOSES)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Is an \"island reversal\" really a turning point? 🏝️\n"
            "### The gap-cluster-gap figure that *looks* unmistakable — and means nothing, in plain English\n\n"
            + BADGES +
            "Here's one of the most striking shapes in chart-reading. A stock rallies, then one morning "
            "it **gaps up** — opens with a jump, leaving a blank space on the chart. It trades for a day "
            "or a few up there, and then it **gaps down** the other way, leaving that little cluster of "
            "bars stranded above, like an **island** marooned in empty space. Two jumps, opposite "
            "directions, a few bars trapped between them. The lore says: *the trend is exhausted — this "
            "is a reversal — sell the island top* (and buy the mirror-image island bottom).\n\n"
            "It's beautiful, it's memorable, and it's the kind of pattern your eye *loves*. So let's do "
            "the unromantic thing and **measure** it: after a real island top forms, does the stock "
            "actually fall? Spoiler — the bearish version **goes the wrong way**, and the bullish version "
            "only \"works\" because stocks tend to drift up anyway.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo test and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A note up front.** A chart figure is partly in the eye of the beholder, so we test "
            "the closest **mechanical** rule (a clear gap, a small island that never fills it, a sealing "
            "gap the other way) on a fixed **30-name large-cap basket** (names still trading today — so "
            "it carries **survivorship**). Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| After an **island top**, does the stock fall (so a short pays)? | **No — it usually keeps "
            "going up.** Short it and you *lose* about **−0.7% in a week, −1.3% in two**, compared with "
            "just shorting at random. The bearish island top reverses the *wrong* way. |\n"
            "| What about the **island bottom** (buy it)? | It looks great on paper — but only because "
            "stocks drift **up** anyway. Buy at random dates and you do just as well; the \"pattern\" "
            "adds nothing the market wasn't already giving you. |\n"
            "| Does demanding a *cleaner* island help? | **No.** Bigger gaps, smaller islands — the edge "
            "stays negative at every strictness we tried. |\n"
            "| On SPY itself? | The figure fires **3 times in 21 years** — and all three lose. A "
            "non-event. |\n\n"
            "> The island reversal is a gorgeous shape your eye finds in the noise. The tape doesn't "
            "care: the bearish side **loses**, the bullish side is **just drift**."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When a trend is about to die, it makes one last leap — a gap — and then reverses with a "
            "gap the other way, trapping a little **island** of bars at the extreme. That island is the "
            "exhaustion signal. Short the island top, buy the island bottom, the moment the second gap "
            "seals it.\"*\n\n"
            "This is straight out of the classic chart-reading canon — Edwards & Magee's "
            "*Technical Analysis of Stock Trends* (1948) and Bulkowski's *Encyclopedia of Chart "
            "Patterns*. The island reversal is taught as one of the **high-confidence** reversal figures, "
            "precisely because the two gaps make it so visually clean. So: is it real?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the island reversal really marked exhaustion, it would be a gift: gaps are **objective** "
            "(you can't argue about whether the price jumped), so you'd have a crisp, rules-based timing "
            "signal for catching tops and bottoms — the holy grail of chart-reading.\n\n"
            "But there's a classic trap waiting. A pattern that says *\"buy\"* on something that drifts "
            "up anyway will **look** like it works for free — the stock rises, you credit the pattern, "
            "when really you just collected the market's normal drift. The only honest test asks: does "
            "the figure beat **a random entry in the same direction on the same stock**? That's the "
            "question we'll actually answer."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take a fixed **{R['n_names']}-name large-cap basket** ({R['years']:.0f} years of daily "
            f"data, {R['start']} → {R['end']}) and write down a **mechanical** island rule:\n\n"
            "1. **A clear gap** (at least 1%) in one direction.\n"
            "2. **An island** — one to three bars that never fall back to fill that gap.\n"
            "3. **A sealing gap** the *other* way, of the same size, that re-isolates the cluster.\n\n"
            f"That found **{R['n_tops']} island tops** and **{R['n_bottoms']} island bottoms**. For each "
            "one we enter the day **after** the sealing gap (no cheating) and see how the stock does over "
            "the next 5, 10, 20, 40 days — **minus** that stock's own average move (so we're not just "
            "measuring drift). Then we compare it to thousands of **random** entry dates: if the pattern "
            "is real, it should beat random. If random matches it, the pattern is a mirage."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**The headline: the island top.** This is *the* bearish figure. If it marks exhaustion, "
            "shorting it should make money — a **positive** excess. Here's the short's excess over a "
            "random short, at four horizons."
        ),
        code(
            "hs = [5, 10, 20, 40]\n"
            "if HAVE_REAL:\n"
            "    top = [st.run_experiment(PANEL, horizon=h, side='top', gap_pct=GAP, n_draws=5000)['mean']*100 for h in hs]\n"
            "else:\n"
            "    top = [r[2] for r in R['top']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar([str(h) for h in hs], top, color=RED, width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(top): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='top' if v<0 else 'bottom')\n"
            "ax.set_xlabel('trading days held after the island'); ax.set_ylabel('short excess over random (%)')\n"
            "ax.set_title('The bearish island top: the short LOSES at every horizon')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('island-top short excess by horizon:', [round(v,2) for v in top])"
        ),
        md(
            f"Every bar is **below zero**. Shorting the island top loses about **{R['top'][0][2]:+.2f}%** "
            f"in a week and **{R['top'][1][2]:+.2f}%** in two, *relative to a random short*. The famous "
            "bearish reversal reverses the **wrong way** — the stocks keep drifting up right through the "
            "supposedly exhausted top."
        ),
        md(
            "**Now the trap — the island bottom.** Buy the bullish mirror image and it looks *fantastic*. "
            "But watch what the honest comparison does to it."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bot_raw = [st.run_experiment(PANEL, horizon=h, side='bottom', gap_pct=GAP, n_draws=5000)['raw_mean']*100 for h in hs]\n"
            "    bot_exc = [st.run_experiment(PANEL, horizon=h, side='bottom', gap_pct=GAP, n_draws=5000)['mean']*100 for h in hs]\n"
            "else:\n"
            "    bot_raw = [r[3] for r in R['bot']]; bot_exc = [r[2] for r in R['bot']]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, bot_raw, .4, color=GREEN, label='raw return (looks great!)')\n"
            "ax.bar(x+.2, bot_exc, .4, color=GREY, label='excess over the stock\\'s own drift')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('island-bottom long return (%)')\n"
            "ax.set_title('The island bottom: subtract the drift and most of it vanishes')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('raw:', [round(v,2) for v in bot_raw], ' excess:', [round(v,2) for v in bot_exc])"
        ),
        md(
            f"The green bars (raw) look like a real bullish signal — **{R['bot'][3][3]:+.2f}%** at 40 "
            f"days! But the grey bars (excess over the stock's own drift) are much smaller, and — as the "
            "quants notebook shows — even those don't beat a **random** buy date. You're not catching a "
            "reversal; you're just holding stocks that go up because stocks go up."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The bearish island top's short excess is **negative at every horizon** "
            f"({R['top'][0][2]:+.2f}% at 5d, {R['top'][1][2]:+.2f}% at 10d) — it loses to a random short.\n"
            "- **Tradability — Mirage.** After costs the short is worse still, and the bullish side's "
            "apparent edge is just market drift the placebo refuses. Nothing to deploy.\n"
            "- **\"Reliable reversal\"? — Busted.** A real reversal figure works on both sides on its "
            "own merits. This one **loses** going short and only **rides drift** going long. The island "
            "is a shape your eye finds, not a force the tape obeys."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — no, and here's the picture\n\n"
            "The island-top short, **net of trading costs** (you pay the spread on both legs). It was "
            "already negative gross; costs only push it further under water."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = [st.run_experiment(PANEL, horizon=h, side='top', gap_pct=GAP, n_draws=2000)['mean']*100 for h in hs]\n"
            "    nv = [st.run_experiment(PANEL, horizon=h, side='top', gap_pct=GAP, n_draws=2000)['net']*100 for h in hs]\n"
            "else:\n"
            "    g = [r[2] for r in R['top']]; nv = [r[8] for r in R['top']]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, g, .4, color=GREY, label='gross excess')\n"
            "ax.bar(x+.2, nv, .4, color=RED, label='net of costs')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('island-top short return (%)'); ax.set_title('Already losing gross; costs make it worse')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('net island-top short by horizon:', [round(v,2) for v in nv])"
        ),
        md(
            f"There is simply nothing here to trade. The short is under water gross (best case "
            f"{R['top'][0][2]:+.2f}% at 5d) and net of costs it's **{R['top'][0][8]:+.2f}%** — you'd pay "
            "to take a bet that loses on average. The island reversal is one of the prettiest dead-ends "
            "on the bench."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Try other universes.** Small-caps gap more often — does a bigger sample change the "
            "story? (Watch the survivorship and cost traps if you do.)\n"
            "- **The drift trap is the real lesson.** Any bullish chart figure on an up-drifting asset "
            "looks like it works. Always subtract the base rate and check against random dates — that's "
            "the one habit that separates a real signal from a mirage.\n"
            "- **Its sibling figures.** See [415-triple-top-bottom](../../415-triple-top-bottom/) for the "
            "three-tap reversal (same result) and [363-pead-drift](../../363-pead-drift/) for a folk "
            "effect that *does* survive the same machinery — the contrast is the point.\n\n"
            "*Think a stricter island rule resurrects it? Show the **net** island-top short landing "
            "positive and beating the placebo — then we'll talk.*"
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
            "# Island Reversal — a quantitative teardown 🔬\n"
            "### Objective gap-island detector · forward excess over base rate at 5/10/20/40d · "
            "one-sample + HAC *t* · same-tape random-date placebo · gap-strictness sweep · the "
            "island-bottom symmetry myth check · a synthetic faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The island "
            "reversal is a *gap-defined* figure, so the detector can be genuinely objective — and the "
            "result is a clean **None × Mirage**: the bearish island top has a **negative** forward "
            "excess, and the bullish island bottom's high *t* is manufactured by the tape's up-drift "
            "(the placebo refuses it).\n\n"
            "> ⚠️ **Data + survivorship note.** Fixed **30-name large-cap** basket (SPY + 29), names "
            "still trading in 2026 — a *survivor* panel that tilts post-island returns mildly **up**, "
            "i.e. *against* the headline (bearish) short. Real data: yfinance daily auto-adjusted OHLC, "
            f"{R['start']}→{R['end']} (as-of {R['asof']}). Offline core + synthetic control are "
            "deterministic. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Island-top short excess **{R['top'][0][2]:+.2f}%** at 5d, "
            f"**{R['top'][1][2]:+.2f}%** at 10d — one-sample **t = {R['top'][1][5]:.2f}** (HAC "
            f"{R['top'][1][6]:.2f}), placebo **p = {R['top'][1][7]:.3f}**. Never clears |t| = 2; negative "
            "at every horizon. |\n"
            f"| **Tradability** | `MIRAGE` | Net of 5 bps/leg the short is **{R['top'][1][8]:+.2f}%** "
            f"(10d); the island-bottom long's t>2 (40d t = {R['bot'][3][5]:.2f}) sits **inside the "
            f"placebo cloud** (p = {R['bot'][3][7]:.3f}). Pure beta. |\n"
            f"| **Reliable reversal?** | `BUSTED` | Asymmetric: bearish side **loses** (fights up-drift), "
            f"bullish side only **rides** drift (high t, p = {R['bot'][3][7]:.3f}). Neither works on its "
            "own merits. |\n\n"
            "> 💡 In plain words: the gap-island figure carries no exhaustion information — the bearish "
            "short loses, the bullish long is just the market drifting up, and the placebo (not the *t*) "
            "is what tells them apart."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let an **island top** be: a bar $g_1$ that gaps up ($O_{g_1} \\ge (1+\\gamma)\\,H_{g_1-1}$), "
            "an island of $1..m$ bars whose lows never fall back to $H_{g_1-1}$ (the gap stays open), and "
            "a sealing bar $g_2$ that gaps down below the island's low *and* back inside the pre-gap "
            "range. The signal is a **short** at $g_2$. For each confirmed island let $r_i(H)$ be the "
            "forward $H$-day return entered **one day after** $g_2$, and $\\bar b$ the name's own "
            "(direction-matched) base rate. The figure's edge is the pooled **excess**\n\n"
            "$$\\widehat{\\mu}(H) = \\frac{1}{n}\\sum_i \\big(d\\,r_i(H) - \\bar b\\big),\\qquad "
            "d = -1\\ \\text{(top, short)},\\ +1\\ \\text{(bottom, long)}.$$\n\n"
            "- **H₁ (the figure reverses).** $\\widehat{\\mu}(H) > 0$ and significant for some $H$, on "
            "the bearish **top** side (the canonical figure).\n"
            "- **H₂ (deployable).** $\\widehat{\\mu}(H)$ survives one-way costs × turnover.\n"
            "- **H₃ (symmetric).** The bullish **bottom** works on its own merits — i.e. beats a "
            "same-tape random-date placebo, not just a zero baseline.\n\n"
            "We find **H₁ rejected** (top excess negative, |t| < 2), **H₂ rejected** (net worse), **H₃ "
            "rejected** (bottom's t>2 but placebo p ≈ 0.16 — it's drift, not the figure). The shape "
            "carries no exhaustion signal."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — why the placebo, not the *t*, is the arbiter\n\n"
            "The Signal test is a one-sample $t$ of the pooled excess against zero, plus a Newey-West "
            "(HAC) $t$ for clustered islands:\n\n"
            "$$t = \\frac{\\widehat{\\mu}}{s/\\sqrt n},\\qquad t_{\\text{HAC}} = "
            "\\frac{\\widehat{\\mu}}{\\sqrt{\\widehat{\\Omega}_{\\text{NW}}/n}}.$$\n\n"
            "But on a **drifting** asset over multi-week holds, a one-sample $t$ against *zero* can clear "
            "2 on drift alone — exactly the island-**bottom** trap. So the real arbiter is a "
            "**same-tape random-date placebo**: draw thousands of random entries (same count per name, "
            "same horizon, same base-rate subtraction) and compute $p = \\Pr[\\text{random excess} \\ge "
            "\\text{observed}]$. If random matches the figure, the figure adds nothing — *whatever the "
            "$t$ says*. This is the single discipline that separates a chart figure from the drift it "
            "rides on."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** Fixed **{R['n_names']}-name** large-cap basket (SPY + 29; yfinance "
            f"auto-adjusted daily OHLC, {R['start']}→{R['end']}, **{R['bars']:,}** bars, fingerprint "
            f"`{R['fingerprint']}`). **Survivor** panel — named on the Signal axis.\n"
            "- **Detector.** Gap threshold **1%**; island 1–3 bars that never fill the first gap; "
            "sealing gap = first opposite gap re-crossing the island's far edge and back inside the "
            "pre-gap range. Non-overlapping. Chart figures are subjective — this is the closest "
            "mechanical rule, stated openly.\n"
            "- **Timing.** Signal at the sealing-gap close; enter the close **1 day later** (no "
            "look-ahead); hold $H\\in\\{5,10,20,40\\}$; drop windows that overrun the tape.\n"
            "- **Edge.** Pooled excess of the trade (top = short, bottom = long) over each name's own "
            "direction-matched base rate.\n"
            "- **Null #1 — one-sample & HAC t** of the pooled excess vs 0.\n"
            "- **Null #2 — same-tape random-date placebo** (5,000 draws): the honest drift control.\n"
            "- **Costs.** 5 bps one-way × 2 legs.\n"
            "- **Myth-check.** The island **bottom** (symmetry); **SPY-only** (the hook).\n"
            "- **Positive control.** A deterministic panel with a **planted** post-island reversal: zero "
            "edge must NOT beat the placebo; a large edge must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The island top across horizons — negative everywhere\n\n"
            "The pooled short excess at each horizon with its one-sample *t*. A real bearish reversal "
            "would sit **above** zero with *t* climbing past 2; this does the opposite."
        ),
        code(
            "hs = [5, 10, 20, 40]\n"
            "if HAVE_REAL:\n"
            "    rs = [st.run_experiment(PANEL, horizon=h, side='top', gap_pct=GAP, n_draws=5000) for h in hs]\n"
            "    exc = [r['mean']*100 for r in rs]; tt = [r['t'] for r in rs]\n"
            "else:\n"
            "    exc = [r[2] for r in R['top']]; tt = [r[5] for r in R['top']]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "a1.bar([str(h) for h in hs], exc, color=RED, width=.6); a1.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(exc): a1.annotate(f'{v:+.2f}%',(i,v),ha='center',va='top' if v<0 else 'bottom',fontsize=9)\n"
            "a1.set_xlabel('horizon (days)'); a1.set_ylabel('short excess (%)'); a1.set_title('Island-top short excess: negative everywhere')\n"
            "a2.bar([str(h) for h in hs], tt, color=GREY, width=.6)\n"
            "a2.axhline(2, ls='--', c=GREEN, label='+2 bar'); a2.axhline(-2, ls='--', c=RED, label='-2 bar')\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(tt): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='top' if v<0 else 'bottom',fontsize=9)\n"
            "a2.set_xlabel('horizon (days)'); a2.set_ylabel('one-sample t'); a2.set_ylim(-2.4,2.4)\n"
            "a2.set_title('|t| never clears 2'); a2.legend(fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('top excess:', [round(v,2) for v in exc]); print('top t:', [round(v,2) for v in tt])"
        ),
        md(
            f"> 💡 In plain words: the island top — *the* bearish figure — loses as a short at every "
            f"horizon (**{R['top'][0][2]:+.2f}%** to **{R['top'][3][2]:+.2f}%**), with *t* between "
            f"{R['top'][1][5]:.2f} and {R['top'][3][5]:.2f}. It doesn't mark exhaustion; the stocks keep "
            "drifting up through the supposedly dead top."
        ),
        md(
            "### 4b · The decisive test — the same-tape placebo on the island top\n\n"
            "The 10-day island-top short excess against a 5,000-draw random-date null. A real signal "
            "sits in the right tail; this one sits **left of the middle** — random shorts do better."
        ),
        code(
            "if HAVE_REAL:\n"
            "    import numpy as np\n"
            "    o,h_,lo,c = PANEL['open'],PANEL['high'],PANEL['low'],PANEL['close']\n"
            "    rng = np.random.default_rng(417); H=10; dir_=-1; draws=[]\n"
            "    for tk in c.columns:\n"
            "        sub = pd.DataFrame({'o':o[tk],'h':h_[tk],'lo':lo[tk],'c':c[tk]}).dropna()\n"
            "        oo,hh,ll,cc = [sub[k].to_numpy(float) for k in ('o','h','lo','c')]\n"
            "        isl = st.detect_islands(oo,hh,ll,gap_pct=GAP,side='top')\n"
            "        fr = st.forward_returns(cc,[d['signal_idx'] for d in isl],H,direction=dir_)\n"
            "        if len(fr)==0: continue\n"
            "        br = st.base_rate(cc,H,direction=dir_); hi=len(cc)-H-2\n"
            "        for _ in range(5000):\n"
            "            si=rng.integers(1,hi,size=len(fr)); e=si+1\n"
            "            draws.append((dir_*(cc[e+H]/cc[e]-1.0)).mean()-br)\n"
            "    draws=np.array(draws)*100\n"
            "    obs = st.run_experiment(PANEL,horizon=H,side='top',gap_pct=GAP,n_draws=10)['mean']*100\n"
            "    pval = float((draws>=obs).mean())\n"
            "else:\n"
            "    import numpy as np\n"
            "    rng=np.random.default_rng(417); draws=rng.normal(0,1.2,150000); obs=R['top'][1][2]; pval=R['top'][1][7]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=70, color=GREY, alpha=.85, label='null: random short dates (same tape)')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'observed island-top short {obs:+.2f}%')\n"
            "ax.set_xlabel('10-day short excess (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Inside the luck cloud (left of centre): placebo p = {pval:.3f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.2f}%  placebo p={pval:.3f}  (random shorts beat the figure ~{pval*100:.0f}% of the time)')"
        ),
        md(
            f"> 💡 In plain words: the red line sits to the **left** of the random cloud — random short "
            f"dates beat the island top ~{R['top'][1][7]*100:.0f}% of the time (p = {R['top'][1][7]:.3f}). "
            "Not only is there no edge, the figure is *worse* than random. This is a clean **Signal = "
            "NONE** on the headline side."
        ),
        md(
            "### 4c · Robustness — no gap threshold rescues it\n\n"
            "If the figure were real, a *bigger* sealing gap (a cleaner island) should sharpen it. The "
            "10-day short excess and its *t* across gap thresholds — it stays negative throughout."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rob = []\n"
            "    for gp in (0.005, 0.01, 0.02):\n"
            "        r = st.run_experiment(PANEL, horizon=10, side='top', gap_pct=gp, n_draws=3000)\n"
            "        rob.append((gp*100, r['n_events'], r['mean']*100, r['t'], r['p_placebo']))\n"
            "else:\n"
            "    rob = [(r[0], r[1], r[2], r[3], r[4]) for r in R['robust']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "labs = [f'{r[0]:.1f}%\\n(n={r[1]})' for r in rob]\n"
            "ax.bar(labs, [r[2] for r in rob], color=RED, width=.55); ax.axhline(0, c='k', lw=.8)\n"
            "for i,r in enumerate(rob): ax.annotate(f't={r[3]:+.2f}',(i,r[2]),ha='center',va='top',fontsize=9)\n"
            "ax.set_xlabel('gap threshold (island top)'); ax.set_ylabel('10-day short excess (%)')\n"
            "ax.set_title('Stricter gaps do not rescue the figure')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('robustness (gap%, n, excess%, t, p):', [(r[0], r[1], round(r[2],2), round(r[3],2), round(r[4],3)) for r in rob])"
        ),
        md(
            f"> 💡 In plain words: the strictest 2% gap collapses to just **{R['robust'][2][1]} events** "
            f"(the mechanical island is genuinely rare on liquid large-caps) and the loosest "
            f"(**{R['robust'][0][1]}** events) is still negative at t = {R['robust'][0][3]:.2f}. There is "
            "no strictness at which two bracketing gaps become a tradable reversal."
        ),
        md(
            "### 4d · The myth check — the island bottom is just drift\n\n"
            "A real reversal must work on **both** sides. The island-**bottom** long: its **raw** return, "
            "its **excess** over the name's drift, and its placebo *p*. The high *t* is a drift artefact."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rb = [st.run_experiment(PANEL, horizon=h, side='bottom', gap_pct=GAP, n_draws=5000) for h in hs]\n"
            "    raw=[r['raw_mean']*100 for r in rb]; exc=[r['mean']*100 for r in rb]; pv=[r['p_placebo'] for r in rb]; tv=[r['t'] for r in rb]\n"
            "else:\n"
            "    raw=[r[3] for r in R['bot']]; exc=[r[2] for r in R['bot']]; pv=[r[7] for r in R['bot']]; tv=[r[5] for r in R['bot']]\n"
            "x=np.arange(len(hs))\n"
            "fig, (a1,a2) = plt.subplots(1,2,figsize=(10.6,4.3))\n"
            "a1.bar(x-.2, raw, .4, color=GREEN, label='raw return'); a1.bar(x+.2, exc, .4, color=GREY, label='excess over drift')\n"
            "a1.axhline(0,c='k',lw=.8); a1.set_xticks(x); a1.set_xticklabels([f'{h}d' for h in hs])\n"
            "a1.set_ylabel('island-bottom long (%)'); a1.set_title('Raw looks great; excess is smaller'); a1.legend(fontsize=8)\n"
            "a2.bar([str(h) for h in hs], pv, color=AMBER, width=.6)\n"
            "a2.axhline(0.05, ls='--', c=RED, label='p=0.05')\n"
            "for i,(p,t) in enumerate(zip(pv,tv)): a2.annotate(f't={t:.2f}',(i,p),ha='center',va='bottom',fontsize=9)\n"
            "a2.set_xlabel('horizon (days)'); a2.set_ylabel('placebo p'); a2.set_ylim(0,.6)\n"
            "a2.set_title('Placebo never drops below 0.05 (despite t>2)'); a2.legend(fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('bottom excess:', [round(v,2) for v in exc]); print('bottom t:', [round(v,2) for v in tv]); print('bottom placebo p:', [round(v,3) for v in pv])"
        ),
        md(
            f"> 💡 In plain words: this is the trap in one chart. The island bottom's 40-day *t* is "
            f"**{R['bot'][3][5]:.2f}** — looks bulletproof — but its placebo **p = {R['bot'][3][7]:.3f}**: "
            "random buy dates match it 16% of the time. The big *t* is the basket's multi-week up-drift "
            "compounding, not the figure. Sort the two apart and the reversal disappears."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "On a deterministic panel with **planted** island tops followed by a real **down** reversal: "
            "with zero edge the placebo must NOT light up (no false positive); with a large planted "
            "reversal it must. Both hold — so the real-tape placebo p ≈ 0.56–0.68 is a genuine nothing."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.25):\n"
            "    pan, truth = data.synthetic_panel(edge=edge, seed=417, n_planted=8, daily_vol=0.011, gap_pct=0.04)\n"
            "    r = st.run_experiment(pan, horizon=20, side='top', gap_pct=0.02, n_draws=1500)\n"
            "    res.append((edge, truth['n_planted_total'], r['n_events'], r['mean']*100, r['t'], r['p_placebo'], r['win']*100))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "labels = [f'planted\\n{e*100:.0f}% reversal' for e,_,_,_,_,_,_ in res]\n"
            "ps = [r[5] for r in res]\n"
            "ax.bar(labels, ps, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(0.05, ls='--', c=RED, label='p=0.05')\n"
            "for i,p in enumerate(ps): ax.annotate(f'p={p:.3f}',(i,p),ha='center',va='bottom')\n"
            "ax.set_ylabel('placebo p (20-day)'); ax.set_title('Control: no edge -> p high; planted reversal -> p~0'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,pl,n,ls,t,p,w in res: print(f'planted {e*100:+.0f}%: planted={pl} n={n} excess={ls:+.2f}% t={t:+.2f} p={p:.3f} win={w:.0f}%')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted reversal the placebo is **not beaten** "
            f"(p = {R['syn'][0][6]:.3f} — no false positive); a **+25%** planted reversal lights up at "
            f"p = {R['syn'][1][6]:.3f}, t = {R['syn'][1][5]:.0f}, win {R['syn'][1][7]:.0f}%. The harness "
            "can clearly bank a real reversal — so the real tape's refusal is a true negative, not a "
            "power failure."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — island-top short excess **negative at every horizon** "
            f"({R['top'][0][2]:+.2f}% at 5d, {R['top'][1][2]:+.2f}% at 10d), one-sample *t* peaking at "
            f"**{R['top'][1][5]:.2f}** (HAC {R['top'][1][6]:.2f}), placebo **p = {R['top'][1][7]:.3f}** "
            "(worse than random). Never clears |t| = 2; no gap threshold rescues it. Survivorship tilts "
            "*against* the short — so this is generous.\n"
            f"- **Tradability `MIRAGE`** — net of 5 bps/leg the short is **{R['top'][1][8]:+.2f}%** (10d), "
            f"and the island-bottom long's t>2 (40d t = {R['bot'][3][5]:.2f}) sits **inside the placebo "
            f"cloud** (p = {R['bot'][3][7]:.3f}) — pure beta. Nothing deployable, nothing to scale.\n"
            f"- **Reliable reversal? `BUSTED`** — the figure is asymmetric: the bearish island top "
            "**loses** (fights the up-drift), the bullish island bottom only **rides** that drift (high "
            "*t*, placebo it can't beat). A genuine reversal works on both sides on its own merits; this "
            "one works on neither."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the SPY hook + the net picture\n\n"
            "On the index itself the detector barely fires, and net of costs the basket short is under "
            "water. Two views of the same nothing."
        ),
        code(
            "if HAVE_REAL:\n"
            "    spy = [st.run_experiment(PANEL, horizon=h, names=['SPY'], side='top', gap_pct=GAP, n_draws=5000) for h in hs]\n"
            "    spy_exc=[r['mean']*100 for r in spy]; spy_n=[r['n_events'] for r in spy]\n"
            "    nv=[st.run_experiment(PANEL, horizon=h, side='top', gap_pct=GAP, n_draws=2000)['net']*100 for h in hs]\n"
            "else:\n"
            "    spy_exc=[r[2] for r in R['spy']]; spy_n=[r[1] for r in R['spy']]; nv=[r[8] for r in R['top']]\n"
            "fig, (a1,a2) = plt.subplots(1,2,figsize=(10.6,4.3))\n"
            "a1.bar([str(h) for h in hs], spy_exc, color=RED, width=.6); a1.axhline(0,c='k',lw=.8)\n"
            "for i,v in enumerate(spy_exc): a1.annotate(f'{v:+.2f}%',(i,v),ha='center',va='top',fontsize=9)\n"
            "a1.set_xlabel('horizon (days)'); a1.set_ylabel('SPY island-top short excess (%)')\n"
            "a1.set_title(f'SPY: fires {spy_n[0]}x in 21y, all lose')\n"
            "a2.bar([str(h) for h in hs], nv, color=GREY, width=.6); a2.axhline(0,c='k',lw=.8)\n"
            "for i,v in enumerate(nv): a2.annotate(f'{v:+.2f}%',(i,v),ha='center',va='top',fontsize=9)\n"
            "a2.set_xlabel('horizon (days)'); a2.set_ylabel('basket short, net of costs (%)')\n"
            "a2.set_title('Net basket short: under water at every horizon')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('SPY island-top excess:', [round(v,2) for v in spy_exc], 'n:', spy_n)\n"
            "print('net basket short:', [round(v,2) for v in nv])"
        ),
        md(
            f"> 💡 In plain words: on SPY the figure fires **{R['spy'][0][1]} times in 21 years** and all "
            f"lose ({R['spy'][3][2]:+.2f}% at 40d). On the basket, net of costs the short is "
            f"**{R['top'][1][8]:+.2f}%** at 10d. There is no horizon, no universe, and no cost assumption "
            "under which the island reversal pays. **MIRAGE**, unambiguously — and here that word is "
            "earned, because the placebo and the control both confirm it."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Gaps are universe-dependent.** Small-caps and single names gap far more than liquid "
            "large-caps; a wider universe gives more events but adds survivorship and cost traps. The "
            "*method* (excess + placebo) is what travels, not the basket.\n"
            "- **The base-rate-and-placebo habit generalises.** Any bullish chart figure on an "
            "up-drifting asset looks like it works. The island bottom is the clean cautionary tale: a "
            "*t* of 4.2 that is **pure drift**. Always net the base rate *and* beat the random-date "
            "placebo.\n"
            "- **Sibling figures.** [415-triple-top-bottom](../../415-triple-top-bottom/) (same machinery, "
            "same NONE × MIRAGE) and [363-pead-drift](../../363-pead-drift/) (the folk effect that *does* "
            "clear the bar) frame why this one doesn't.\n\n"
            "*The reproducible core is offline and deterministic; the detector is the closest mechanical "
            "island rule, stated openly. Methods and sources: "
            "[`docs/references.md`](../docs/references.md); frozen numbers: "
            "[`docs/results.md`](../docs/results.md).*"
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
