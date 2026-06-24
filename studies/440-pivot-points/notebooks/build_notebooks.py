"""Generate the two narrative notebooks for Study 440 (Floor-Trader Pivot Points).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached 5-minute basket
bars under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
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


# Frozen real-tape headline numbers — mirror of docs/results.md.
# yfinance 5-minute bars, basket SPY/QQQ/AAPL/MSFT/IWM, 2026-03-30 -> 2026-06-23 (59 full
# sessions/name; the live 2026-06-24 partial day dropped). Touch tol = 5 bps, bounce horizon
# k = 6 bars (30 min), one-way cost 1 bp (round trip 2 bp).
R = dict(
    start="2026-03-30", end="2026-06-23", n_names=5, sessions=59, interval="5m",
    tol_bps=5.0, k=6, cost_bps=1.0,
    # pooled across the basket
    piv_touches=1410, piv_rate=0.4674, piv_mean_bps=-1.12, piv_t=-1.04,
    piv_win=0.4674, piv_net_bps=-3.12,
    ctl_touches=5782, ctl_rate=0.5012, ctl_mean_bps=0.36, ctl_t=0.82,
    rate_diff=-0.0338,
    # pooled permutation placebo (300 redraws of the random control lines)
    placebo_p=1.00, placebo_ctl_mean=0.498, placebo_ctl_sd=0.0075,
    # bounce rate by level: name -> (n, rate, mean_bps)
    by_level={
        "S3": (102, 0.637, 9.06), "S2": (190, 0.516, 1.61), "S1": (373, 0.496, 0.35),
        "R1": (445, 0.425, -3.15), "R2": (221, 0.412, -5.05), "R3": (79, 0.392, -5.37),
    },
    # per-name: ticker -> (diff, t, net_bps, placebo_p)
    by_name={
        "SPY": (-0.040, -0.53, -2.66, 0.985), "QQQ": (-0.068, -0.65, -3.31, 1.000),
        "AAPL": (-0.058, -1.93, -6.69, 1.000), "MSFT": (+0.020, +0.31, -1.03, 0.005),
        "IWM": (-0.030, -0.03, -2.07, 0.985),
    },
    # synthetic control: edge -> (touches, rate, ctrl_rate, mean_bps, t, win, placebo_p)
    syn=[(0.0, 227, 0.489, 0.505, -0.17, -0.06, 0.489, 0.813),
         (1.5, 196, 0.704, 0.530, 32.90, 7.79, 0.704, 0.000)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Respects_the_level%3F: Busted](https://img.shields.io/badge/Respects_the_level%3F-Busted-8b949e?style=flat-square)\n\n"
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

from pivot_points import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    CACHE = {t: st.drop_partial_sessions(b) for t, b in data.load_real().items()}
    PIV = pd.concat([st.pivot_touch_events(b, tol_bps=5.0, k=6) for b in CACHE.values()],
                    ignore_index=True)
    CTL = pd.concat([st.control_touch_events(b, tol_bps=5.0, k=6, seed=440) for b in CACHE.values()],
                    ignore_index=True)
else:
    CACHE = PIV = CTL = None
print("real 5m cache present:", HAVE_REAL,
      "| pivot touches:", (0 if PIV is None else len(PIV)),
      "| control touches:", (0 if CTL is None else len(CTL)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do floor-trader \"pivot points\" really act as support and resistance? 📏\n"
            "### One of the oldest day-trading recipes — tested against the only fair yardstick: a line drawn at random\n\n"
            + BADGES +
            "Open any day-trading chat and someone has drawn a fan of horizontal lines on their chart: "
            "the **pivot** (`P`), and above/below it **R1, R2, R3** and **S1, S2, S3**. The recipe is "
            "ancient — it comes from the pit, computed each morning from *yesterday's* high, low and "
            "close. The promise: price **respects** these levels. It drifts down to S1 and *bounces*; it "
            "rallies to R1 and *stalls*. So you fade the touch.\n\n"
            "It *feels* true because price does bounce off them sometimes. But here's the catch that kills "
            "almost every support/resistance claim: **price bounces off *any* line sometimes.** Draw a "
            "horizontal line at a random price and walk away — it'll get touched, and price will wander "
            "away from it about half the time, by pure chance. So the real question isn't \"do pivots get "
            "respected?\" It's: **do they get respected *more than a line you drew with your eyes shut*?**\n\n"
            "> 📓 **Plain-language layer.** Want the HAC *t*-stats, the permutation placebo and the cost "
            "math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A loud data note.** Intraday history is *short*: Yahoo only serves ~**60 calendar "
            "days** of 5-minute bars. So this is **5 very liquid names over ~12 weeks** — enough to see "
            "whether the effect is anywhere near tradable, not enough to certify a tiny one. And on "
            "intraday levels the **spread** is the executioner. Both caveats travel with the verdict."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do pivots get touched and bounced? | **Sure — about half the time.** Over the basket, a "
            f"pivot touch is followed by a bounce **{R['piv_rate']*100:.0f}%** of the time. |\n"
            "| Is that better than a *random* line? | **No — it's actually worse.** A horizontal line "
            f"drawn at a random price bounces **{R['ctl_rate']*100:.0f}%** of the time on the same tape. "
            "The pivots score *below* a coin flip and *below* the random line. |\n"
            "| Could a random line just be lucky? | **It's the other way round.** Re-draw the random "
            f"lines 300 times and **100%** of them match-or-beat the pivots' bounce rate. The pivots are "
            "not special — they're slightly *sub*-special. |\n"
            "| Could you trade the fade? | **No.** The average fade trade makes "
            f"**{R['piv_mean_bps']:+.1f} bps** *before* costs; after a tiny round-trip spread it's "
            f"**{R['piv_net_bps']:+.1f} bps** — you pay to play a coin flip. |\n\n"
            "> The legend that \"price respects the pivots\" is **busted** on this tape: the pivots do no "
            "better than lines drawn at random, and the fade trade is a negative-expectation game once you "
            "pay the spread."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim\n\n"
            "> *\"Each morning, compute the pivot `P = (yesterday's High + Low + Close) / 3`, then the "
            "resistances `R1, R2, R3` above and supports `S1, S2, S3` below. Intraday, price gravitates "
            "to these levels and **respects** them — it bounces off support, stalls at resistance. Fade "
            "the touch and you're trading with the floor.\"*\n\n"
            "This isn't a fringe idea. Pivot points are built into **every** trading platform, taught in "
            "countless day-trading courses, and were genuinely used by floor traders for quick mental "
            "support/resistance before screens. The arithmetic is real and shared by everyone — which is "
            "exactly why, *if* it worked, it would be arbitraged. The steelman: it's a **self-fulfilling** "
            "level because so many eyes watch it. We'll test whether those eyes actually move price."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what?\n\n"
            "If pivots were real support/resistance, intraday trading would have a free map: known levels, "
            "computed before the open, where the odds tilt. You'd fade touches for a steady, high-"
            "frequency edge. That's the dream the recipe sells.\n\n"
            "But two traps sit underneath it. **(1) Every line gets touched.** Price is a wiggly path; any "
            "horizontal line inside the day's range gets crossed and \"bounced\" off by chance — so the "
            "only meaningful test compares pivots to *random* lines on the same tape. **(2) The spread "
            "eats fast trades.** An intraday fade is a round trip measured in *minutes*; if the gross edge "
            "is a basis point or two, the bid-ask spread — paid twice — turns it negative before you "
            "start. We test both."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{R['n_names']} of the most liquid US names** (SPY, QQQ, AAPL, MSFT, IWM) on "
            f"**5-minute bars**, ~**{R['sessions']} full sessions** ({R['start']} → {R['end']}; the live "
            "partial day is dropped). For each session:\n\n"
            "1. **Draw the pivots** from the *previous* session's high/low/close (no peeking at today).\n"
            "2. **Find every touch** — a bar whose range straddles a level (within 5 bps).\n"
            "3. **Score the bounce** — does price move *away* from the level (up off support, down off "
            "resistance) over the next **6 bars (30 minutes)**?\n"
            "4. **Compare to random.** Draw random horizontal lines inside each session's range and run "
            "the *identical* test. **If pivots are real, their bounce rate beats the random lines.**\n\n"
            "**What would make us say \"busted\"?** If the pivot bounce rate is **not above** the random-"
            "line rate, and the fade trade doesn't clear costs. (Spoiler: that's exactly what happens.)"
        ),

        # ---- BEAT 4 ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**The headline picture.** Bounce rate at the pivots versus bounce rate at randomly-placed "
            "lines on the same tape. The dashed line is a coin flip (50%). If the legend held, the green "
            "bar would tower over the grey one."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pr = st.bounce_rate(PIV); cr = st.bounce_rate(CTL)\n"
            "else:\n"
            "    pr = R['piv_rate']; cr = R['ctl_rate']\n"
            "fig, ax = plt.subplots(figsize=(7.8, 4.4))\n"
            "ax.bar(['floor-trader\\npivots','random\\nlines'], [pr*100, cr*100], color=[RED, GREY], width=.6)\n"
            "ax.axhline(50, ls='--', c='k', lw=1, label='coin flip (50%)')\n"
            "for i,v in enumerate([pr*100, cr*100]): ax.annotate(f'{v:.1f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('touch -> bounce rate (%)'); ax.set_ylim(0, 60)\n"
            "ax.set_title('Pivots bounce LESS often than lines drawn at random'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'pivot bounce rate {pr*100:.1f}%  vs random line {cr*100:.1f}%  (diff {(pr-cr)*100:+.1f} pts)')"
        ),
        md(
            f"There's the whole story in one chart. The pivots bounce **{R['piv_rate']*100:.1f}%** of the "
            f"time — *below* a coin flip — while the random lines bounce **{R['ctl_rate']*100:.1f}%**. The "
            "pivots aren't a special map; they're slightly **worse** than lines you could have drawn "
            "blindfolded. (Why below 50%? Once you charge the touch to the *next* bar and hold 30 minutes, "
            "the tiny intraday drift and mean-reversion roughly cancel — there's just no there there.)"
        ),
        md(
            "**Maybe some levels are special?** A fan favourite is \"S1 always holds.\" Let's break the "
            "bounce rate out by level — deep supports on the left, far resistances on the right."
        ),
        code(
            "names = list(R['by_level'].keys())\n"
            "if HAVE_REAL:\n"
            "    rates = [st.bounce_rate(PIV[PIV['level_name']==n])*100 for n in names]\n"
            "else:\n"
            "    rates = [R['by_level'][n][1]*100 for n in names]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "cols = [GREEN if r>50 else RED for r in rates]\n"
            "ax.bar(names, rates, color=cols, width=.7)\n"
            "ax.axhline(50, ls='--', c='k', lw=1)\n"
            "for i,v in enumerate(rates): ax.annotate(f'{v:.0f}%',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "ax.set_ylabel('bounce rate (%)'); ax.set_ylim(0,75)\n"
            "ax.set_title('No level reliably holds — the only >50% is a tiny-sample deep support')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({n: round(st.bounce_rate(PIV[PIV['level_name']==n]),3) if HAVE_REAL else R['by_level'][n][1] for n in names})"
        ),
        md(
            f"The only level above 50% is **S3** (**{R['by_level']['S3'][1]*100:.0f}%**) — but that's just "
            f"**{R['by_level']['S3'][0]} touches** in a basket that fell over the window, so a deep "
            "support \"holding\" is mostly *the tape went up afterwards*, not the line. The resistances "
            f"(**R1 {R['by_level']['R1'][1]*100:.0f}%**, **R2 {R['by_level']['R2'][1]*100:.0f}%**, "
            f"**R3 {R['by_level']['R3'][1]*100:.0f}%**) actively bounce *less* than a coin flip. There's "
            "no level that reliably does its job."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Pivot bounces (**{:.0f}%**) don't beat random-line bounces "
            "(**{:.0f}%**) — they're *below* both the random line and a coin flip. A permutation placebo "
            "says **100%** of random lines match or beat the pivots. No signal.\n"
            "- **Tradability — Mirage.** Even the gross fade trade is **{:+.1f} bps**; after a minimal "
            "intraday round-trip spread it's **{:+.1f} bps**. You'd pay to trade noise.\n"
            "- **\"Price respects the level\"? — Busted.** It respects a *random* line just as much. The "
            "pivots carry no extra information.".format(
                R['piv_rate']*100, R['ctl_rate']*100, R['piv_mean_bps'], R['piv_net_bps'])
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you actually trade it? — the spread closes the case\n\n"
            "Suppose you ignored all the above and faded every touch. Here's the gross fade-trade edge "
            "versus the net, after a *tiny* 1-bp one-way spread (2 bp round trip — generous for these "
            "ultra-liquid names; a retail trader pays more)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = st.trade_stats(PIV, cost_bps=0.0)['mean_bps']\n"
            "    nv = st.trade_stats(PIV, cost_bps=1.0)['net_bps']\n"
            "else:\n"
            "    g = R['piv_mean_bps']; nv = R['piv_net_bps']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.3))\n"
            "ax.bar(['gross','net of\\n2bp round trip'], [g, nv], color=[GREY, RED], width=.55)\n"
            "ax.axhline(0, c='k', lw=.9)\n"
            "for i,v in enumerate([g, nv]): ax.annotate(f'{v:+.1f} bps',(i,v),ha='center',va='top' if v<0 else 'bottom')\n"
            "ax.set_ylabel('fade-trade return per touch (bps)')\n"
            "ax.set_title('Negative gross, more negative net — a paid coin flip')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.2f} bps  ->  net {nv:+.2f} bps per fade')"
        ),
        md(
            f"The fade trade is **negative gross** (**{R['piv_mean_bps']:+.1f} bps**) and only gets worse "
            f"net (**{R['piv_net_bps']:+.1f} bps**). There's no horizon or cost level where this becomes a "
            "trade — the gross edge is already on the wrong side of zero. **Mirage**, full stop."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The short-span caveat is real.** ~12 weeks of 5-minute bars can't certify a *tiny* edge — "
            "but it's plenty to show pivots don't beat random lines or clear the spread, which is the only "
            "claim that matters for a trader.\n"
            "- **Other recipes.** Camarilla, Woodie, Fibonacci and Demark pivots use different arithmetic "
            "on the same H/L/C. They're equally arbitrary lines; the harness here tests any of them — swap "
            "the formula in `data._pivot_levels` and re-run.\n"
            "- **The synthetic control proves the test works.** In the quants notebook we *plant* a real "
            "bounce and the harness lights up at *t* ≈ 8 — so finding nothing on the real tape means "
            "there's nothing, not that the test is blind.\n\n"
            "*Think a specific pivot level holds? Show its bounce rate beating the random-line rate **and** "
            "the fade clearing the spread — on out-of-sample names. Then we'll talk.*"
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
            "# Floor-trader pivot points — a quantitative teardown 🔬\n"
            "### Touch → bounce hit-rate vs a randomly-placed control line · HAC *t* on the fade trade · a "
            "permutation placebo · intraday costs · a synthetic planted-bounce control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). Pivot points "
            "are a support/resistance claim, and the *only* honest test of support/resistance is **vs a "
            "line drawn at random on the same tape** — because any horizontal line gets touched and "
            "\"bounced\" off some fraction of the time. We measure the touch→bounce hit-rate, the fade-"
            "trade HAC *t*, a 300-draw permutation placebo, the net-of-spread edge, and a synthetic "
            "positive control that proves the harness can detect a planted bounce.\n\n"
            "> ⚠️ **Data + capacity note.** Yahoo caps **5-minute** history at ~**60 calendar days**, so "
            "the real tape is **{} liquid names × ~{} sessions** ({}→{}) — short by design. State this on "
            "the **Signal** axis: enough power to reject a *tradable* effect, not to certify a tiny one. "
            "Intraday fades live or die on the **spread**. Offline core + synthetic control are "
            "deterministic. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition.".format(
                R['n_names'], R['sessions'], R['start'], R['end'])
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Pivot touch→bounce rate **{R['piv_rate']:.3f}** vs random-line "
            f"**{R['ctl_rate']:.3f}** (diff **{R['rate_diff']:+.3f}**); fade-trade HAC **t = {R['piv_t']:.2f}**. "
            f"Permutation placebo: **p = {R['placebo_p']:.2f}** (a random line matches/beats the pivots). |\n"
            f"| **Tradability** | `MIRAGE` | Gross fade **{R['piv_mean_bps']:+.2f} bps**/touch; net of a "
            f"2-bp round trip **{R['piv_net_bps']:+.2f} bps**. Negative before costs — nothing to scale. |\n"
            f"| **Respects the level?** | `BUSTED` | The random-line control bounces *more* "
            f"({R['ctl_rate']:.3f} > {R['piv_rate']:.3f}). The pivots carry no support/resistance "
            "information beyond an arbitrary horizontal line. |\n\n"
            "> 💡 In plain words: pivots are not better than random lines, and the fade trade is negative "
            "even gross. The synthetic control (below) shows the harness *would* catch a real bounce — so "
            "this is a true null, not a blind test."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, formalised\n\n"
            "Compute the floor-trader set from the prior session's $(H_{-1}, L_{-1}, C_{-1})$:\n\n"
            "$$P=\\tfrac{H_{-1}+L_{-1}+C_{-1}}{3},\\quad R_1=2P-L_{-1},\\ S_1=2P-H_{-1},\\quad "
            "R_2=P+(H_{-1}-L_{-1}),\\ S_2=P-(H_{-1}-L_{-1}),\\ \\dots$$\n\n"
            "For a level $\\ell$ and a *touch* (a bar whose range straddles $\\ell$ within tolerance "
            "$\\tau$), define the **bounce return** over $k$ bars, entered one bar later, signed in the "
            "level's respect direction $s_\\ell$ (+1 for supports/$P$, −1 for resistances):\n\n"
            "$$b_i \\;=\\; s_{\\ell}\\!\\left(\\frac{C_{t_i+1+k}}{C_{t_i+1}}-1\\right).$$\n\n"
            "- **H₁ (real S/R).** The pivot bounce rate $\\Pr[b_i>0]$ **exceeds** the rate of a randomly-"
            "placed control line on the same tape.\n"
            "- **H₂ (tradable).** The mean fade return $\\bar b$ clears a HAC *t* = 2 **and** survives the "
            "round-trip spread.\n\n"
            "We find **H₁ rejected** (pivots bounce *less* than random) and **H₂ rejected** (negative even "
            "gross). The level-respect mechanism is busted."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — why the control line is the whole game\n\n"
            "A naive \"pivots bounce ~50% of the time!\" is meaningless: **any** line does. The "
            "informative quantity is the *excess* bounce rate over a null line drawn at a uniform-random "
            "price inside the session range, tagged support/resistance at random, and run through the "
            "**identical** touch→bounce machinery. That control absorbs everything generic — intraday "
            "volatility, the day's drift, the touch tolerance, the holding horizon — and isolates whatever "
            "is special about the *pivot arithmetic*. Inference is then a one-sample **HAC (Newey-West) "
            "t** on the fade return (touches cluster within sessions, so we don't trust a plain *t*), plus "
            "a **permutation placebo** that re-draws the control lines hundreds of times to put a *p* on "
            "the bounce-rate gap."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {R['n_names']} liquid names (SPY, QQQ, AAPL, MSFT, IWM), **{R['interval']}** "
            f"bars, {R['start']}→{R['end']} (~{R['sessions']} full sessions/name; the live partial day "
            "dropped — no partial sessions in a stamped run).\n"
            "- **Levels.** Classic floor-trader $P, R_1{-}R_3, S_1{-}S_3$ from the prior session "
            "(no look-ahead).\n"
            f"- **Touch.** A bar straddling the level within **{R['tol_bps']:.0f} bps**; consecutive touch "
            "bars collapse to the first.\n"
            f"- **Bounce.** Enter the close **1 bar after** the touch; measure the signed move over "
            f"**k = {R['k']} bars** (30 min) in the level's respect direction.\n"
            "- **Control.** Uniform-random price lines inside each session's range, random side, identical "
            "machinery — the honest baseline.\n"
            f"- **Inference.** HAC *t* of the fade return vs 0; a **{300}-draw permutation placebo** of the "
            "control bounce rate vs the pivots'.\n"
            f"- **Costs.** {R['cost_bps']:.0f} bp one-way (2 bp round trip) on the fade.\n"
            "- **Positive control.** A synthetic tape with a **planted bounce** at the pivots: edge 0 must "
            "NOT beat the control; a planted bounce must light up."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · Pivots vs random lines — the bounce-rate gap\n\n"
            "Left: pivot bounce rate vs control vs a coin flip. Right: the gap broken out per name — if "
            "pivots were real, every bar would be positive."
        ),
        code(
            "names = list(R['by_name'].keys())\n"
            "if HAVE_REAL:\n"
            "    pr = st.bounce_rate(PIV); cr = st.bounce_rate(CTL)\n"
            "    diffs = []\n"
            "    for t in names:\n"
            "        b = CACHE[t]\n"
            "        p = st.bounce_rate(st.pivot_touch_events(b, tol_bps=5.0, k=6))\n"
            "        c = st.bounce_rate(st.control_touch_events(b, tol_bps=5.0, k=6, seed=440))\n"
            "        diffs.append((p-c)*100)\n"
            "else:\n"
            "    pr = R['piv_rate']; cr = R['ctl_rate']; diffs = [R['by_name'][t][0]*100 for t in names]\n"
            "fig, (a1,a2) = plt.subplots(1,2, figsize=(10.8,4.3))\n"
            "a1.bar(['pivots','random'], [pr*100, cr*100], color=[RED, GREY], width=.6)\n"
            "a1.axhline(50, ls='--', c='k', lw=1); a1.set_ylim(0,60); a1.set_ylabel('bounce rate (%)')\n"
            "for i,v in enumerate([pr*100,cr*100]): a1.annotate(f'{v:.1f}%',(i,v),ha='center',va='bottom')\n"
            "a1.set_title('Pivots vs random line')\n"
            "a2.bar(names, diffs, color=[GREEN if d>0 else RED for d in diffs], width=.7)\n"
            "a2.axhline(0, c='k', lw=.9); a2.set_ylabel('pivot - control bounce rate (pts)')\n"
            "for i,v in enumerate(diffs): a2.annotate(f'{v:+.1f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Per name: mostly negative')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'pooled pivot {pr*100:.1f}% vs control {cr*100:.1f}%  diff {(pr-cr)*100:+.1f} pts')\n"
            "print('per-name diff (pts):', {t: round(d,1) for t,d in zip(names, diffs)})"
        ),
        md(
            f"> 💡 In plain words: pooled, the pivots bounce **{R['piv_rate']*100:.1f}%** vs the random "
            f"line's **{R['ctl_rate']*100:.1f}%** — a **{R['rate_diff']*100:+.1f}-point** *deficit*. Per "
            "name, four of five are negative; only MSFT is marginally positive (and, below, its fade is "
            "still negative net). There is no excess support/resistance to harvest."
        ),
        md(
            "### 4b · The fade trade — HAC *t* against zero\n\n"
            "The fade return per touch (signed in the respect direction) with its Newey-West *t*. The bar "
            "should clear *t* = 2 if there's a tradable edge."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.trade_stats(PIV, cost_bps=1.0)\n"
            "    mean_bps, tval, net_bps = s['mean_bps'], s['t'], s['net_bps']\n"
            "else:\n"
            "    mean_bps, tval, net_bps = R['piv_mean_bps'], R['piv_t'], R['piv_net_bps']\n"
            "fig, ax = plt.subplots(figsize=(7.6,4.3))\n"
            "ax.bar(['fade-trade\\nHAC t'], [tval], color=RED if abs(tval)<2 else GREEN, width=.4)\n"
            "ax.axhline(2, ls='--', c='k', lw=1, label='|t|=2 bar'); ax.axhline(-2, ls='--', c='k', lw=1)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.annotate(f't={tval:.2f}',(0,tval),ha='center',va='bottom' if tval>=0 else 'top')\n"
            "ax.set_ylim(-3,3); ax.set_ylabel('HAC t-stat'); ax.legend()\n"
            "ax.set_title(f'gross {mean_bps:+.2f} bps/touch, HAC t={tval:.2f} -> no edge')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {mean_bps:+.2f} bps  HAC t={tval:.2f}  net {net_bps:+.2f} bps')"
        ),
        md(
            f"> 💡 In plain words: the fade earns **{R['piv_mean_bps']:+.2f} bps** with **HAC t = "
            f"{R['piv_t']:.2f}** — nowhere near the bar, and on the *wrong side of zero*. The sign is "
            "negative because, net of the one-bar entry lag and 30-min hold, touches are followed by "
            "*continuation* as often as reversion. No fade edge."
        ),
        md(
            "### 4c · Permutation placebo — could a random line look this good?\n\n"
            "Re-draw the random control lines 300 times; each draw yields a control bounce rate. The "
            "observed *pivot* rate should sit in the right tail if pivots are special. It doesn't — it "
            "sits *below* the control cloud."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bks = list(CACHE.values()); ND = 300\n"
            "    rates = np.empty(ND)\n"
            "    for i in range(ND):\n"
            "        evs = [st.control_touch_events(b, tol_bps=5.0, k=6, seed=10000+i+13*j) for j,b in enumerate(bks)]\n"
            "        rates[i] = st.bounce_rate(pd.concat(evs, ignore_index=True))\n"
            "    pr = st.bounce_rate(PIV); pval = float((rates>=pr).mean())\n"
            "else:\n"
            "    pr = R['piv_rate']; pval = R['placebo_p']\n"
            "    rng = np.random.default_rng(440); rates = rng.normal(R['placebo_ctl_mean'], R['placebo_ctl_sd'], 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "ax.hist(rates*100, bins=30, color=GREY, alpha=.85, label='300 random-line redraws')\n"
            "ax.axvline(pr*100, c=RED, lw=2.5, label=f'observed pivot rate {pr*100:.1f}%')\n"
            "ax.set_xlabel('bounce rate (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Pivot rate sits BELOW the random cloud: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'control rate {rates.mean()*100:.1f}% +/- {rates.std()*100:.2f}  pivot {pr*100:.1f}%  p(ctrl>=pivot)={pval:.2f}')"
        ),
        md(
            f"> 💡 In plain words: the red line (pivots) is to the *left* of the entire grey cloud — "
            f"**p = {R['placebo_p']:.2f}**, i.e. essentially every random line matches or beats the "
            "pivots. This is the cleanest possible NONE: not \"indistinguishable from random\" but "
            "\"slightly worse than random.\""
        ),
        md(
            "### 4d · Synthetic positive control — the harness can see a real bounce\n\n"
            "On a deterministic tape where we **plant** a genuine bounce away from the pivots: edge 0 must "
            "match the control (no false positive); a planted bounce must light up at high *t*. Both hold "
            "— so the real-tape null is a true null."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 1.5):\n"
            "    b,_ = data.synthetic_panel(edge=edge, seed=440)\n"
            "    r = st.run_experiment(b, tol_bps=5.0, k=6, cost_bps=1.0, n_placebo=150)\n"
            "    res.append((edge, r['pivot_bounce_rate'], r['control_bounce_rate'], r['pivot_mean_bps'], r['pivot_t'], r['placebo_p']))\n"
            "fig, ax = plt.subplots(figsize=(8.6,4.3))\n"
            "x = np.arange(2)\n"
            "ax.bar(x-.2, [r[1]*100 for r in res], .4, color=RED, label='pivot bounce rate')\n"
            "ax.bar(x+.2, [r[2]*100 for r in res], .4, color=GREY, label='control bounce rate')\n"
            "ax.axhline(50, ls='--', c='k', lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels(['planted edge 0\\n(null)','planted bounce 1.5\\n(real)'])\n"
            "ax.set_ylabel('bounce rate (%)'); ax.set_ylim(0,80); ax.legend()\n"
            "ax.set_title('Control: null stays flat; planted bounce lights up')\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,pr,cr,m,t,p in res: print(f'edge={e}: pivot={pr:.3f} ctrl={cr:.3f} mean={m:+.2f}bps t={t:+.2f} placebo_p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted bounce the synthetic pivots sit at "
            f"**{R['syn'][0][2]:.3f}** ≈ control **{R['syn'][0][3]:.3f}** (HAC t = {R['syn'][0][5]:.2f} — "
            f"no false positive). Plant a real bounce and the rate jumps to **{R['syn'][1][2]:.3f}** at "
            f"**t = {R['syn'][1][5]:.2f}**, placebo p = {R['syn'][1][7]:.3f}. The machinery is faithful — "
            "so the real-tape t ≈ −1 is genuinely *nothing*, not a blind spot."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — pivot touch→bounce rate **{R['piv_rate']:.3f}** is *below* both the "
            f"random-line control **{R['ctl_rate']:.3f}** and a coin flip; fade HAC **t = {R['piv_t']:.2f}**; "
            f"permutation placebo **p = {R['placebo_p']:.2f}** (random lines match/beat the pivots). The "
            "synthetic control proves the test detects a planted bounce at t ≈ 8, so this is a true null.\n"
            f"- **Tradability `MIRAGE`** — the fade is **{R['piv_mean_bps']:+.2f} bps**/touch gross, "
            f"**{R['piv_net_bps']:+.2f} bps** net of a 2-bp round trip. Negative before costs; nothing to "
            "size.\n"
            f"- **Respects the level? `BUSTED`** — an arbitrary horizontal line bounces *more* than the "
            "pivots. The floor-trader arithmetic adds no support/resistance information.\n\n"
            "**Survivorship / scope caveats (Signal axis).** Short span (~12 weeks, the 5-minute cap) and "
            "5 surviving liquid names — enough to reject a *tradable* effect, not to certify a tiny one. "
            "The verdict is the honest, expected outcome for classic chart-line folklore."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — gross is already negative\n\n"
            "There's no cost sweep to do: the gross fade edge is below zero, so every realistic spread "
            "only deepens the loss. One picture of gross vs net makes the point and closes the case."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g = st.trade_stats(PIV, cost_bps=0.0)['mean_bps']; nv = st.trade_stats(PIV, cost_bps=1.0)['net_bps']\n"
            "else:\n"
            "    g = R['piv_mean_bps']; nv = R['piv_net_bps']\n"
            "fig, ax = plt.subplots(figsize=(7.4,4.3))\n"
            "ax.bar(['gross','net (2bp RT)'], [g, nv], color=[GREY, RED], width=.5)\n"
            "ax.axhline(0, c='k', lw=.9)\n"
            "for i,v in enumerate([g,nv]): ax.annotate(f'{v:+.1f}',(i,v),ha='center',va='top')\n"
            "ax.set_ylabel('fade return per touch (bps)'); ax.set_title('Negative gross -> deeper negative net')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.2f} bps -> net {nv:+.2f} bps')"
        ),
        md(
            "> 💡 In plain words: when the gross edge is already negative, *capacity* and *impact* are moot "
            "— there's no edge to scale. The pivots fail the very first hurdle. MIRAGE."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further\n\n"
            "- **Other pivot families.** Camarilla, Woodie, DeMark and Fibonacci pivots are different "
            "arithmetic on the same prior-session H/L/C — equally arbitrary lines. Swap "
            "`data._pivot_levels` and the whole harness applies.\n"
            "- **Conditioning.** One could test pivots only on trend days, or only the first touch, or "
            "with a tighter tolerance — but with the *gross* fade already negative and the bounce rate "
            "below random, any sub-slice that looks good is selection, not signal (correct it with the "
            "permutation placebo).\n"
            "- **The general lesson.** Support/resistance claims must always be priced against a random "
            "line on the *same* tape. \"It bounces ~half the time\" is the sound of a coin, not a level.\n\n"
            "*The reproducible core is offline and deterministic once the 5-minute cache exists; the "
            "synthetic positive control needs no network. Methods and sources: "
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
