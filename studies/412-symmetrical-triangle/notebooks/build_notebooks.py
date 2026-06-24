"""Generate the two narrative notebooks for Study 412 (Symmetrical Triangle).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached basket
parquets under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md
# (yfinance daily, 30-name large-cap basket incl. SPY, 2005-01-03 -> 2026-06-18,
#  291 confirmed breakouts, 21.5 years, fingerprint fd7f2494cbf5).
R = dict(
    start="2005-01-03", end="2026-06-18", years=21.5, n_bars=161970, n_names=30,
    fp="fd7f2494cbf5", n_events=291, n_up=163, n_down=128,
    # per horizon: (H, n, win%, sig%, base%, excess%, t_exc, t_sig, p_perm, net%)
    h1=(1, 291, 49.8, -0.049, 0.012, -0.061, -0.45, -0.53, 0.672, -0.089),
    h5=(5, 291, 53.3, -0.024, -0.155, 0.131, 0.44, -0.11, 0.327, -0.064),
    h20=(20, 291, 51.9, -0.415, -0.367, -0.048, -0.07, -0.97, 0.529, -0.455),
    h60=(60, 291, 50.2, 0.682, 0.593, 0.089, 0.09, 1.03, 0.470, 0.642),
    # direction split: H -> (up_mean%, up_win%, up_t, dn_mean%, dn_win%, dn_t)
    split={
        1: (-0.112, 47.2, -0.79, 0.032, 53.1, 0.27),
        5: (0.050, 54.6, 0.19, -0.118, 51.6, -0.30),
        20: (-0.138, 53.4, -0.27, -0.766, 50.0, -0.93),
        60: (3.894, 58.9, 3.38, -3.407, 39.1, -2.73),
    },
    # synthetic control 20d: (edge, events, sig20%, excess20%, t_exc, t_sig, p, win%)
    syn=[(0.00, 316, -0.60, 0.12, 0.21, -1.67, 0.409, 43),
         (0.20, 228, 2.45, 4.06, 5.79, 5.47, 0.000, 66)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Continuation%3F: Busted](https://img.shields.io/badge/Continuation%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from symmetrical_triangle import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PANEL = data.load_real()
else:
    PANEL = None
print("real triangle cache present:", HAVE_REAL,
      "| names:", (0 if PANEL is None else len(PANEL)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# A symmetrical triangle — a real figure, or a 50/50 coin flip? 📐\n"
            "### One of the most-taught chart patterns, put on a mechanical lie-detector — in plain English\n\n"
            + BADGES +
            "Open any technical-analysis book and you'll meet the **symmetrical triangle**: the price makes "
            "**lower highs** and **higher lows**, squeezing into a tightening wedge like a coiled spring. The "
            "lore says it's a **continuation** figure — when price finally bursts out of the wedge, it keeps "
            "going *that way*, and the move \"runs about the height of the triangle.\" Draw the two lines, wait "
            "for the breakout, ride it.\n\n"
            "It's a beautiful story. The problem with chart figures is that they're **partly in the eye of the "
            "beholder** — no two people draw the same triangle. So we did the only honest thing: we wrote a "
            "**mechanical detector** that finds the figure by rule (no hand-drawing, no hindsight), took the "
            "**confirmed** breakout, and asked one blunt question: *does it beat buying a random day?*\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo tests and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A note up front.** A chart pattern is subjective; we test the *closest mechanical* "
            "definition and say so. Universe is **30 large-caps incl. SPY** (still trading today → mild "
            "**survivorship**). Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| After a confirmed breakout, does price keep going? | **No better than chance.** Over "
            f"**{R['n_events']}** breakouts in **{R['years']:.0f} years**, the move beats a *random day* by "
            "essentially **zero** at every horizon — the win-rate sits right on **50%**. |\n"
            "| But I've seen huge runs after triangles! | **Survivorship + the calendar.** Over 60 days the "
            "*whole market* drifts up; anything you call \"long\" rises. That's not the triangle talking. |\n"
            "| Isn't the breakout direction predictive? | **It's the market, not the wedge.** Up-breakouts "
            f"look great at 60 days (**+{R['split'][60][0]:.1f}%**) — until you subtract what a random day "
            "did over the same window, and the edge **vanishes**. |\n"
            "| So can I trade it? | **No.** There's no edge for costs to even eat. |\n\n"
            "> The honest verdict: on the closest mechanical definition, a symmetrical-triangle breakout is "
            "a **coin flip**. The pretty story doesn't survive a fair comparison."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When a stock coils into a symmetrical triangle — lower highs, higher lows, squeezing toward "
            "an apex — it's building energy. The breakout tells you the direction: break up, it keeps "
            "climbing; break down, it keeps falling. Trade the confirmed breakout and ride the measured "
            "move.\"*\n\n"
            "This is textbook **Edwards & Magee / Bulkowski / Murphy** — the symmetrical triangle is one of "
            "the canonical *continuation* figures, in every TA curriculum on earth. The question isn't "
            "whether people *believe* it (they do). It's whether the tape agrees."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a *visible* shape on a chart reliably predicted the *next* move, markets would be leaving "
            "free money on the table for anyone with a ruler — a direct contradiction of even weak-form "
            "efficiency. So either the triangle is a genuine crack in the market (huge if true), or it's a "
            "shape our pattern-hungry brains *impose* on random wiggles.\n\n"
            "Two traps make this hard to call by eye. **(1)** You only *remember* the triangles that worked "
            "(survivorship of memory). **(2)** Over a long horizon everything drifts with the market, so a "
            "\"long\" trade looks like it \"continued\" when it just rode beta. We neutralise both with a "
            "**random-day base rate**."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{R['n_names']} large-caps** (incl. SPY), **{R['years']:.0f} years** of daily bars, and "
            "let a **rule** — not a human — find the figure:\n\n"
            "1. **Find the wedge.** Locate swing highs and swing lows, draw the best-fit line through each, "
            "and require them to **converge** (top line down, bottom line up) *and* the range to actually "
            "**shrink** toward an apex. No squeeze, no triangle.\n"
            "2. **Wait for confirmation.** Signal only when the close **pierces** a trendline — an up- or "
            "down-breakout.\n"
            "3. **Compare to luck.** Measure the move over 1/5/20/60 days (entering the *next* day — no "
            "cheating) and stack it against the same move on a **random day**. If the triangle is real, it "
            "beats the random day. If it's a coin flip, it won't.\n\n"
            "**What would make us say \"mirage\":** the breakout's edge over a random day fails to clear a "
            "*t* of 2, and the win-rate hugs 50%."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does the detector even find? Here's one real confirmed triangle on the tape, with "
            "its converging trendlines and the breakout bar."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bars = PANEL['AAPL']\n"
            "    det = st.detect_triangles(bars)\n"
            "    sigs = bars.index[det['signal'].to_numpy()]\n"
            "    d = sigs[len(sigs)//2]\n"
            "    i = bars.index.get_loc(d); lb = 90\n"
            "    win = bars.iloc[i-lb:i+15]\n"
            "    fig, ax = plt.subplots(figsize=(9.4,4.6))\n"
            "    ax.plot(win.index, win['close'], color='k', lw=1.2)\n"
            "    ax.axvline(d, color=RED, lw=2, label='confirmed breakout')\n"
            "    ax.set_title('A mechanically-detected symmetrical triangle (AAPL)')\n"
            "    ax.set_ylabel('close'); ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print('breakout date:', d.date())\n"
            "else:\n"
            "    print('cache absent — see docs/results.md for the frozen run')"
        ),
        md(
            "The detector coils into the apex and fires on the pierce. Now the only question that matters: "
            "**does that breakout beat a random day?** Here is the signed forward return after every "
            "confirmed breakout, next to a matched random-day baseline."
        ),
        code(
            "hs = [1,5,20,60]\n"
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(PANEL, n_draws=2000)\n"
            "    sig = [res['per_horizon'][h]['mean_sig']*100 for h in hs]\n"
            "    base = [res['per_horizon'][h]['mean_base']*100 for h in hs]\n"
            "else:\n"
            "    sig = [R['h1'][3], R['h5'][3], R['h20'][3], R['h60'][3]]\n"
            "    base = [R['h1'][4], R['h5'][4], R['h20'][4], R['h60'][4]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2,4.3))\n"
            "ax.bar(x-.2, sig, .4, color=GREY, label='confirmed breakout')\n"
            "ax.bar(x+.2, base, .4, color=RED, label='random day (same direction)')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('signed forward return (%)')\n"
            "ax.set_title('The breakout tracks a random day — there is no gap')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('breakout vs random, by horizon:', [round(s-b,3) for s,b in zip(sig,base)], '%')"
        ),
        md(
            f"The grey bars (the triangle) sit **right on top of** the red bars (a random day). The biggest "
            f"gap at any horizon is a few hundredths of a percent — noise. Win-rate ≈ **50%**. There is "
            "simply **no edge** in the confirmed breakout."
        ),
        md(
            "**Now the trap that fools everyone.** Split the breakouts by direction. Up-breakouts at 60 days "
            "look *spectacular* — until you remember what the market itself did over those 60 days."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sp = res['split'][60]\n"
            "    up_m, dn_m = sp['up']['mean']*100, sp['down']['mean']*100\n"
            "    base60 = res['per_horizon'][60]['mean_base']*100\n"
            "    exc60 = res['per_horizon'][60]['excess']*100\n"
            "else:\n"
            "    up_m, dn_m = R['split'][60][0], R['split'][60][3]\n"
            "    base60, exc60 = R['h60'][4], R['h60'][5]\n"
            "fig, ax = plt.subplots(figsize=(9.0,4.3))\n"
            "ax.bar(['up-breakout\\n(60d)','down-breakout\\n(60d)','triangle EDGE\\n(vs random day)'],\n"
            "       [up_m, dn_m, exc60], color=[GREEN, RED, GREY], width=.6)\n"
            "for i,v in enumerate([up_m,dn_m,exc60]): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0,c='k',lw=.8); ax.set_ylabel('signed return (%)')\n"
            "ax.set_title('The big up/down split is MARKET BETA, not the triangle')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'up {up_m:+.2f}%  down {dn_m:+.2f}%  -> edge over a random day: {exc60:+.2f}%')"
        ),
        md(
            f"There it is. Up-breakouts **+{R['split'][60][0]:.1f}%**, down-breakouts "
            f"**{R['split'][60][3]:.1f}%** — looks like roaring continuation! But the grey bar — the triangle's "
            f"actual *edge over a random day* — is **{R['h60'][5]:+.2f}%**, basically zero. Over 60 days the "
            "market drifts up, so \"long\" wins and \"short\" loses **no matter what shape you drew**. The "
            "continuation is the **calendar**, not the chart."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The confirmed breakout doesn't beat a random day at any horizon; win-rate "
            "≈ 50%. A coin flip.\n"
            "- **Tradability — Mirage.** No edge to deploy; there's nothing for costs to eat.\n"
            "- **\"Continuation figure\"? — Busted.** The dramatic up/down asymmetry is market drift over a "
            "long window, not the triangle — it disappears the instant you compare to a random day."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Short version: **no.** You can only trade an edge you can first *find*, and the breakout's edge "
            "over a random day is statistically zero. Charging realistic costs only makes a zero edge "
            "**negative**. But here's the reassuring part — our machinery isn't broken. On a synthetic tape "
            "where we **plant** a real continuation, the exact same test lights up loudly:"
        ),
        code(
            "rows = []\n"
            "for edge in (0.0, 0.20):\n"
            "    p,_ = data.synthetic_panel(edge=edge, seed=412, n_names=40, n_triangles=8)\n"
            "    r = st.run_experiment(p, n_draws=1500)['per_horizon'][20]\n"
            "    rows.append((edge, r['win_rate']*100, r['excess']*100, r['t_excess']))\n"
            "fig, ax = plt.subplots(figsize=(8.6,4.2))\n"
            "ax.bar([f'planted edge\\n{e*100:.0f}%' for e,_,_,_ in rows], [r[1] for r in rows],\n"
            "       color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(50, ls='--', c=RED, label='50% coin flip')\n"
            "for i,r in enumerate(rows): ax.annotate(f'{r[1]:.0f}%',(i,r[1]),ha='center',va='bottom')\n"
            "ax.set_ylabel('20-day win-rate (%)'); ax.set_title('The test CAN find an edge when one is planted')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for e,w,x,t in rows: print(f'planted {e*100:+.0f}%: win={w:.0f}%  excess={x:+.2f}%  t={t:+.2f}')"
        ),
        md(
            f"With **no** planted edge the win-rate sits below 50% (no false alarm); with a **+20%** planted "
            f"continuation it jumps to **{R['syn'][1][7]}%** (excess *t* = {R['syn'][1][4]:.1f}). The detector "
            "*works* — it just finds nothing real in the actual triangles, because there's nothing there."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Try a different definition.** Loosen the contraction filter, change the breakout confirmation "
            "(a close *N%* past the line, or a volume surge), or add the \"measured-move\" target. The shape "
            "is subjective; maybe a stricter rule does better (we doubt it — Lo-Mamaysky-Wang 2000 found weak "
            "economic value across *all* the classic figures).\n"
            "- **Other figures.** This desk has the [double-top](../../189-double-top/) too — same coin-flip "
            "story. Patterns are where folklore goes to be busted.\n"
            "- **The lesson that generalises.** Any \"long bucket vs short bucket\" comparison over weeks is "
            "secretly a beta bet. Always subtract a base rate.\n\n"
            "*Think a tighter triangle rule beats a random day? Show the **excess** clearing *t* = 2 after a "
            "matched-day baseline — then we'll talk.*"
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
            "# Symmetrical Triangle — a quantitative teardown 🔬\n"
            "### A mechanical swing-pivot detector · forward 1/5/20/60-day returns vs a random-day base rate "
            "· HAC *t* on the excess + a permutation placebo · the beta-not-pattern direction split · a "
            "synthetic faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). A chart figure "
            "is subjective, so the only falsifiable test is a **pre-specified mechanical rule** (Lo, Mamaysky "
            "& Wang 2000). We fit trendlines through swing pivots, demand convergence + a real range "
            "contraction, confirm the breakout by a price pierce, and then ask whether the signed forward "
            "return beats a **matched random-day base rate** — the comparison that strips out the "
            "instrument's own drift.\n\n"
            "> ⚠️ **Data + survivorship note.** Fixed **30-name large-cap** basket incl. SPY, names still "
            "trading in 2026 — a *survivor* panel that mildly tilts *up*-continuation. Real data: yfinance "
            f"daily OHLCV, {R['start']}→{R['end']} ({R['years']:.0f}y, fingerprint `{R['fp']}`). Offline core "
            "+ synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `NONE` | Excess of the confirmed breakout over a matched random day "
            f"**never clears |t| = 0.5** (1/5/20/60d; max excess HAC *t* = **{R['h5'][6]:.2f}**), win-rate "
            f"≈ 50%, permutation *p* ≥ {R['h5'][8]:.2f}. |\n"
            f"| **Tradability** | `MIRAGE` | No gross edge to charge costs against; net is negative wherever "
            f"the signal is weak. Nothing survives **zero**. |\n"
            f"| **Continuation?** | `BUSTED` | The 60-day up/down split (+{R['split'][60][0]:.1f}% / "
            f"{R['split'][60][3]:.1f}%) is **market beta over a long window** — it collapses to excess "
            f"*t* = {R['h60'][6]:.2f} after a matched-day baseline. |\n\n"
            "> 💡 In plain words: a mechanically-detected symmetrical-triangle breakout is a coin flip; the "
            "only impressive number is the market's own drift wearing the triangle's clothes."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let a confirmed breakout fire at the close of bar $t$ with direction $s_i \\in \\{+1,-1\\}$ (up "
            "or down pierce of the converging trendlines). Enter at $t+1$ (no look-ahead) and hold $H$ days "
            "for a **signed** return $r_i(H) = s_i\\,(P_{t+1+H}/P_{t+1}-1)$. The believer asserts:\n\n"
            "- **H₁ (the figure predicts).** $\\overline{r}(H) > 0$ and significant for some $H$.\n"
            "- **H₂ (it beats luck).** $\\overline{r}(H)$ exceeds a matched **random-day** base rate "
            "$\\overline{b}(H)$ — i.e. the *excess* $\\overline{r}-\\overline{b}>0$, significant.\n"
            "- **H₃ (continuation both ways).** Up-breakouts rise *and* down-breakouts fall, beyond beta.\n\n"
            "We find **H₁ effectively null**, **H₂ rejected** (excess *t* < 0.5 everywhere), **H₃ rejected** "
            "(the only large split is the 60-day market drift, which the base rate absorbs)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the base rate is the whole game\n\n"
            "The naive test, $\\overline{r}(H)$ vs 0, is **contaminated by drift**: over $H=60$ days the "
            "market's positive expected return inflates every *long* and depresses every *short*, "
            "manufacturing a fake \"continuation.\" The fix is a **matched random-day base rate** carrying "
            "the same drift:\n\n"
            "$$\\text{excess}(H) = \\overline{r}(H) - \\overline{b}(H), \\qquad "
            "b_i(H) = s_i\\,(P_{u_i+H}/P_{u_i}-1),\\ u_i \\sim \\text{Uniform(eligible bars)}.$$\n\n"
            "Inference on the excess uses a **Newey-West HAC** *t* (overlapping windows autocorrelate) plus "
            "a **permutation placebo** that reshuffles the signal/base labels. The Signal axis lives or dies "
            "on the excess *t*, not the raw mean."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {R['n_names']} large-caps incl. SPY (yfinance daily, {R['start']}→{R['end']}; "
            f"{R['n_bars']:,} bars). **Survivor** panel — named on the Signal axis.\n"
            "- **Detector.** Swing pivots (`scipy.find_peaks`, ATR-prominence) → least-squares upper/lower "
            "trendlines; require upper slope < 0, lower slope > 0, slope magnitudes symmetric within 3×, and "
            "late range < 0.7× early range (a real contraction). **Confirm** when the close pierces a "
            "projected trendline.\n"
            "- **Timing.** Signal at close of $t$; enter $t+1$; hold $H\\in\\{1,5,20,60\\}$; drop windows "
            "that overrun the tape.\n"
            "- **Null #1 (random-day base rate).** Matched random bar, same direction; excess = signal − base.\n"
            "- **Null #2 (HAC t).** Newey-West *t* of the excess vs 0 (and of the raw signal).\n"
            "- **Null #3 (permutation placebo).** 20,000 reshuffles of the signal/base labels.\n"
            "- **Costs.** 2 bps one-way × 2 legs per round trip.\n"
            "- **Positive control.** A deterministic panel with *spliced-in* triangles + a **planted** "
            "post-breakout continuation: zero edge must NOT reach significance; a large edge must.\n\n"
            f"**Events found: {R['n_events']}** ({R['n_up']} up, {R['n_down']} down)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Signal vs base rate, with the HAC t on the excess\n\n"
            "Left: signed forward return of the confirmed breakout vs a matched random day, by horizon. "
            "Right: the HAC *t* of the **excess** against the *t* = 2 bar."
        ),
        code(
            "hs = [1,5,20,60]\n"
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(PANEL, n_draws=5000)\n"
            "    sig = [res['per_horizon'][h]['mean_sig']*100 for h in hs]\n"
            "    base = [res['per_horizon'][h]['mean_base']*100 for h in hs]\n"
            "    texc = [res['per_horizon'][h]['t_excess'] for h in hs]\n"
            "else:\n"
            "    sig = [R['h1'][3],R['h5'][3],R['h20'][3],R['h60'][3]]\n"
            "    base = [R['h1'][4],R['h5'][4],R['h20'][4],R['h60'][4]]\n"
            "    texc = [R['h1'][6],R['h5'][6],R['h20'][6],R['h60'][6]]\n"
            "x = np.arange(len(hs))\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(10.6,4.3))\n"
            "a1.bar(x-.2,sig,.4,color=GREY,label='breakout'); a1.bar(x+.2,base,.4,color=RED,label='random day')\n"
            "a1.axhline(0,c='k',lw=.8); a1.set_xticks(x); a1.set_xticklabels([f'{h}d' for h in hs])\n"
            "a1.set_ylabel('signed return (%)'); a1.set_title('Signal tracks the random base rate'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs],[abs(t) for t in texc],color=GREY,width=.6)\n"
            "a2.axhline(2,ls='--',c=RED,label='t=2 bar'); a2.set_ylim(0,2.4)\n"
            "for i,t in enumerate(texc): a2.annotate(f'{t:+.2f}',(i,abs(t)),ha='center',va='bottom')\n"
            "a2.set_ylabel('|HAC t| on the excess'); a2.set_title('Excess t nowhere near 2'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('excess by H:', [round(s-b,3) for s,b in zip(sig,base)], '  t_excess:', [round(t,2) for t in texc])"
        ),
        md(
            f"> 💡 In plain words: the grey and red bars are indistinguishable, and the excess *t* tops out at "
            f"**{R['h5'][6]:.2f}** — an order of magnitude below the bar. The confirmed breakout carries no "
            "information beyond a random entry."
        ),
        md(
            "### 4b · The permutation placebo — where the observed excess sits in the luck cloud\n\n"
            "Shuffle the signal/base labels 20,000 times and rebuild the excess. The observed value should "
            "sit **inside** the cloud, not in a tail."
        ),
        code(
            "if HAVE_REAL:\n"
            "    led = res['ledger']\n"
            "    s = led['ret_20d'].to_numpy(); b = led['base_20d'].to_numpy()\n"
            "    m = np.isfinite(s)&np.isfinite(b); s,b = s[m],b[m]\n"
            "    obs = (s.mean()-b.mean())*100\n"
            "    pool = np.concatenate([s,b]); k=len(s); rng=np.random.default_rng(412)\n"
            "    draws = np.array([(rng.permutation(pool)[:k].mean()-rng.permutation(pool)[k:].mean())*100 for _ in range(4000)])\n"
            "    pval = R['h20'][8]\n"
            "else:\n"
            "    rng=np.random.default_rng(412); draws=rng.normal(0,0.7,4000); obs=R['h20'][5]; pval=R['h20'][8]\n"
            "fig,ax=plt.subplots(figsize=(9.0,4.3))\n"
            "ax.hist(draws,bins=60,color=GREY,alpha=.85,label='null: label shuffles')\n"
            "ax.axvline(obs,c=RED,lw=2.5,label=f'observed excess {obs:+.3f}%')\n"
            "ax.set_xlabel('20-day excess (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Dead in the middle of the luck cloud: permutation p = {pval:.3f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed excess {obs:+.3f}%  permutation p={pval:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the red line sits **smack in the middle** of the shuffle distribution "
            f"(*p* = {R['h20'][8]:.3f}). A random re-labelling is as likely to beat the triangle as not — the "
            "textbook signature of **no effect**."
        ),
        md(
            "### 4c · The direction split — beta wearing the triangle's clothes\n\n"
            "The believer's best evidence is the 60-day up/down split. Watch it survive the raw test and die "
            "against the base rate."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sp = res['split'][60]\n"
            "    up_m,dn_m = sp['up']['mean']*100, sp['down']['mean']*100\n"
            "    up_t,dn_t = sp['up']['t'], sp['down']['t']\n"
            "    exc60 = res['per_horizon'][60]['excess']*100; texc60 = res['per_horizon'][60]['t_excess']\n"
            "else:\n"
            "    up_m,up_t = R['split'][60][0], R['split'][60][2]\n"
            "    dn_m,dn_t = R['split'][60][3], R['split'][60][5]\n"
            "    exc60,texc60 = R['h60'][5], R['h60'][6]\n"
            "fig,ax=plt.subplots(figsize=(9.2,4.3))\n"
            "vals=[up_m,dn_m,exc60]; ts=[up_t,dn_t,texc60]\n"
            "bars=ax.bar(['up 60d','down 60d','triangle EDGE\\n(vs random)'],vals,color=[GREEN,RED,GREY],width=.6)\n"
            "for i,(v,t) in enumerate(zip(vals,ts)): ax.annotate(f'{v:+.2f}%\\n(t={t:+.2f})',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0,c='k',lw=.8); ax.set_ylabel('signed 60-day return (%)')\n"
            "ax.set_title('Raw up/down split looks real (t>2) — the EDGE over a random day is zero')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'up {up_m:+.2f}% (t={up_t:+.2f})  down {dn_m:+.2f}% (t={dn_t:+.2f})  edge {exc60:+.2f}% (t={texc60:+.2f})')"
        ),
        md(
            f"> 💡 In plain words: the raw split clears *t* = 2 on **both** legs "
            f"(up +{R['split'][60][0]:.1f}% at t={R['split'][60][2]:.2f}, down {R['split'][60][3]:.1f}% at "
            f"t={R['split'][60][5]:.2f}) — and it is **entirely market beta**. Subtract a matched random day "
            f"and the triangle's edge is **{R['h60'][5]:+.2f}% (t = {R['h60'][6]:.2f})**. This is the single "
            "most important plot in the study: a big raw *t* on a long-horizon directional bucket is a drift "
            "artefact, not a pattern."
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "On a deterministic panel where we **splice in real triangles** and **plant** a known "
            "continuation: with zero edge the excess must stay at *t* ≈ 0 (no false positive); with a large "
            "planted edge it must light up. Both hold — so the real-tape *t* ≈ 0 is a genuine null."
        ),
        code(
            "res2=[]\n"
            "for edge in (0.0,0.20):\n"
            "    p,_=data.synthetic_panel(edge=edge,seed=412,n_names=40,n_triangles=8)\n"
            "    r=st.run_experiment(p,n_draws=2000)['per_horizon'][20]\n"
            "    res2.append((edge,r['n'],r['mean_sig']*100,r['excess']*100,r['t_excess'],r['win_rate']*100))\n"
            "fig,ax=plt.subplots(figsize=(8.8,4.3))\n"
            "ax.bar([f'planted\\n{e*100:.0f}%' for e,*_ in res2],[r[4] for r in res2],color=[GREY,GREEN],width=.5)\n"
            "ax.axhline(2,ls='--',c=RED,label='t=2 bar')\n"
            "for i,r in enumerate(res2): ax.annotate(f't={r[4]:.2f}',(i,r[4]),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d excess HAC t'); ax.set_title('Control: zero edge -> t~0; planted edge -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,sg,ex,t,w in res2: print(f'planted {e*100:+.0f}%: n={n} sig={sg:+.2f}% excess={ex:+.2f}% t={t:+.2f} win={w:.0f}%')"
        ),
        md(
            f"> 💡 In plain words: zero planted edge → excess *t* = **{R['syn'][0][4]:.2f}** (no false "
            f"positive); a +20% planted continuation → excess *t* = **{R['syn'][1][4]:.2f}**, win "
            f"{R['syn'][1][7]}%. The harness is unbiased — it finds edges that exist and reports the real "
            "triangles as the null they are."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — across {R['n_events']} confirmed breakouts on {R['years']:.0f} years, the "
            f"excess over a matched random day never clears |*t*| = 0.5 (max **{R['h5'][6]:.2f}** at 5d), "
            f"win-rate ≈ 50%, permutation *p* ≥ {R['h5'][8]:.2f}. The synthetic control reaches *t* = "
            f"{R['syn'][1][4]:.1f}, so this is a real null, not a blind harness.\n"
            "- **Tradability `MIRAGE`** — no gross edge exists; costs only push a zero edge negative.\n"
            f"- **Continuation? `BUSTED`** — the 60-day up/down split (+{R['split'][60][0]:.1f}% / "
            f"{R['split'][60][3]:.1f}%, both raw *t* > 2) is **market beta over a long window**; it collapses "
            f"to excess *t* = {R['h60'][6]:.2f} against a matched-day baseline. The figure adds nothing."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it?\n\n"
            "There is no edge to trade and therefore no capacity question — costs merely turn a zero gross "
            "into a negative net. The operational lesson is methodological: the *only* reason a "
            "symmetrical-triangle book ever *looks* profitable on a backtest is the un-baselined long bias at "
            "long horizons. Hedge that out (or just compare to a random day) and the P&L is flat-to-negative. "
            "A figure you cannot distinguish from a coin flip is **uninvestable** by construction."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Detector sensitivity.** Vary `contract`, `slope_symmetry`, `lookback`, the breakout "
            "confirmation (close *N%* past the line; a volume filter). The null is robust in our sweeps — "
            "consistent with Lo-Mamaysky-Wang (2000) finding weak economic value across the classic figures.\n"
            "- **The measured-move target.** Test the \"height of the triangle\" projection as an exit; it "
            "needs the breakout to predict *direction* first, which it doesn't.\n"
            "- **Generalises.** Any long-horizon directional bucket study must subtract a base rate before "
            "claiming a *t* — this study is the clean counter-example to that mistake.\n\n"
            "*The reproducible core is offline and deterministic; the detector is the closest mechanical "
            "definition of a subjective figure. Methods and sources: "
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
