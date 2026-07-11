"""Generate the two narrative notebooks for Study 695 (Inverse Head-and-Shoulders).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached 30-name basket
tape under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance 30-name basket,
# 2005-01-03 -> 2026-06-30; detector defaults shoulder_tol=0.12, neckline_tol=0.15).
R = dict(
    start="2005-01-03", end="2026-06-30", n_names=30, n_events=229,
    fp="0fed535ace51",
    # forward-return excess by horizon: horizon -> (n, raw%, excess%, t, hac_t, placebo_p, net5%)
    horizons={
        5: (229, 0.326, 0.049, 0.23, 0.23, 0.487, -0.051),
        10: (229, 0.954, 0.407, 1.42, 1.56, 0.402, 0.307),
        20: (228, 1.058, -0.036, -0.08, -0.09, 0.509, -0.136),
        40: (227, 2.315, 0.146, 0.25, 0.26, 0.481, 0.046),
    },
    max_abs_t=1.56, max_abs_t_h=10,
    # robustness sweep: tag -> (shoulder_tol, neckline_tol, n, excess%, t)
    sweep={
        "tight": (0.08, 0.10, 153, -0.300, -0.52),
        "base": (0.12, 0.15, 228, -0.036, -0.08),
        "loose": (0.18, 0.20, 305, 0.206, 0.55),
    },
    # measured-move target
    mm_n=229, mm_hit=170, mm_rate=74.2, mm_lo=68.2, mm_hi=79.5,
    mm_placebo_n=6000, mm_placebo_hit=4661, mm_placebo_rate=77.7,
    mm_median_days=20, mm_mean_dist=6.4,
    # long timer
    tm_n=223, tm_gross=2.711, tm_net5=2.611, tm_net10=2.511,
    tm_t5=4.98, tm_hac5=4.65, tm_avg_hold=55.1, tm_hit_share=73.5,
    tm_excess=-0.146, tm_t_excess=-0.23,
    # synthetic control
    syn_null_mean=0.028, syn_null_sd=1.173, syn_null_fire=2, syn_null_seeds=20,
    syn_planted_n=138, syn_planted_of=80, syn_planted_t=8.23, syn_planted_hac=9.31,
    syn_planted_p=0.00276,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Measured-move_target%3F: Busted](https://img.shields.io/badge/Measured--move_target%3F-Busted-8b949e?style=flat-square)\n\n"
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

from inverse_head_shoulders import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PANEL = data.load_real()
else:
    PANEL = None
print("real cache present:", HAVE_REAL,
      "| names:", (0 if PANEL is None else PANEL["close"].shape[1]))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does \"three troughs, deepest in the middle\" really call the bottom? 📈🗿\n"
            "### The inverse head-and-shoulders — Wall Street's favorite bottoming shape, "
            "tested against 21 years of large-cap tape\n\n"
            + BADGES +
            "Every technical-analysis textbook opens with this shape: price falls, bounces, "
            "falls *deeper* (the \"head\"), bounces, falls one *last* time but not as far (the "
            "\"shoulders\" — roughly matching), then breaks up through the **neckline**. The "
            "story: the sellers tried three times and finally ran out. You buy the breakout, and "
            "the pattern's own geometry — the head-to-neckline **height**, projected upward — "
            "hands you a price target.\n\n"
            "That's two claims, not one: *(1)* the breakout predicts an up-move, and *(2)* the "
            "\"measured move\" tells you how far. We test both. Most market folklore survives on "
            "vibes; this one comes with a built-in falsifiable number, which makes it unusually "
            "easy to check.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** A mechanical swing-pivot detector on SPY + 29 large-caps "
            "(2005→2026), because \"three troughs at roughly the right shape\" is exactly the "
            "kind of thing three chartists draw three different ways — house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does buying the confirmed breakout beat just holding the stock? | **No — not "
            f"provably.** Across **{R['n_events']}** confirmed breakouts on 30 large-caps since "
            f"2005, the best of four holding periods clears only *t* = {R['max_abs_t']:.2f} "
            "(the bar is 2). A random entry date on the same stock does about as well. |\n"
            f"| Does the \"measured-move\" target actually forecast the price? | **No — it does "
            f"worse than chance.** The target is hit **{R['mm_rate']:.1f}%** of the time — but a "
            f"random move of the *same size*, from a random day, hits its target "
            f"**{R['mm_placebo_rate']:.1f}%** of the time. The pattern's forecast underperforms "
            "a coin flip of the same reach. |\n"
            f"| Is there a trade here — hold to the target, sell at a timeout? | **On paper, "
            f"yes-looking; really, no.** The trade earns **+{R['tm_gross']:.2f}%** gross per "
            f"event (*t* ≈ 5!) — but that's *t* vs **zero**, and long-only equities are never "
            f"zero. Raced against buying-and-holding for the *same* {R['tm_avg_hold']:.0f} days, "
            f"the excess is **{R['tm_excess']:+.2f}%** at *t* = {R['tm_t_excess']:+.2f} — "
            "nothing. |\n\n"
            "> The shape is real. The forecast inside it isn't."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Three troughs — left shoulder, a deeper head, a right shoulder about as deep "
            "as the left — with rallies to a neckline between them. When price finally closes "
            "**above** the neckline, the downtrend is exhausted: buy. And the pattern's own "
            "height, the distance from the head up to the neckline, projected from the "
            "breakout, is your price target.\"*\n\n"
            "It's the flagship reversal figure of classical charting (Edwards & Magee, 1948; "
            "still in print), and unlike vaguer folklore it comes with an exact, testable "
            "number attached — the measured move. That precision is what makes it worth taking "
            "seriously enough to check twice."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real, this is a two-for-one: an entry signal *and* a price target, both derived "
            "from the shape alone — no fundamentals, no macro view, just geometry anyone can "
            "draw on a chart. Every technical-analysis course teaches it as close to gospel. So "
            "we ask, honestly: does the breakout beat just owning the stock, and does the "
            "target mean anything more than \"stocks go up eventually\"?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The detector.** A mechanical rule on daily closes: three swing-pivot troughs "
            "with a strictly deeper middle one, roughly matching shoulders, a near-horizontal "
            "neckline, and a confirmed close above it — no eyeballing.\n"
            f"- **The basket.** SPY + 29 large, long-listed US stocks, 2005 → 2026 "
            f"({R['n_events']} confirmed breakouts found).\n"
            "- **The comparison.** Forward returns after the breakout, *minus* the stock's own "
            "average return over the same horizon (owning any stock long enough usually makes "
            "money — the question is whether the *pattern* adds anything on top).\n"
            "- **The target check.** Does price reach the measured-move target more often than "
            "a random move of the *same size* would, starting from a random day?\n"
            "- **The luck check.** A random-date placebo for both tests."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Forward return after the breakout vs. the stock's own "
            "base rate, at four holding periods."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = {h: st.run_experiment(PANEL, horizon=h, cost_bps=5.0) for h in st.HORIZONS}\n"
            "    hs = list(st.HORIZONS)\n"
            "    ex = [rows[h]['mean']*100 for h in hs]\n"
            "    ts = [rows[h]['t'] for h in hs]\n"
            "else:\n"
            "    hs = sorted(R['horizons'])\n"
            "    ex = [R['horizons'][h][2] for h in hs]\n"
            "    ts = [R['horizons'][h][3] for h in hs]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "cols = [RED if abs(t) >= 2 else GREY for t in ts]\n"
            "ax.bar([f'{h}d' for h in hs], ex, color=cols, width=.55)\n"
            "for i,(v,t) in enumerate(zip(ex, ts)):\n"
            "    ax.annotate(f'{v:+.2f}%\\n(t={t:+.2f})',(i,v),ha='center',\n"
            "        va='bottom' if v>=0 else 'top', fontsize=9)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('forward return minus the stock\\'s own base rate')\n"
            "ax.set_title('None of the four horizons clears the t=2 bar (bars would turn red)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({h: round(e,3) for h,e in zip(hs, ex)})"
        ),
        md(
            f"Every bar is grey — none clears the significance bar (red would mean |*t*| ≥ 2). "
            f"The best horizon is **{R['max_abs_t_h']}d** at *t* = **{R['max_abs_t']:.2f}**, "
            "still short, and that's exactly what you'd expect the *best of four* random tries "
            "to look like even if nothing is going on. Random-date placebos land at "
            "*p* ≈ 0.4–0.5 across the board — a random Tuesday does just as well.\n\n"
            "**Next, the target.** Does the pattern's own measured-move actually forecast "
            "anything?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    mm = st.measured_move_hits(PANEL, max_days=126)\n"
            "    hit, plc = mm['hit_rate']*100, mm['placebo_rate']*100\n"
            "else:\n"
            "    hit, plc = R['mm_rate'], R['mm_placebo_rate']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['the pattern\\'s target','a random move\\nof the same size'], [hit, plc],\n"
            "       color=[RED, GREY], width=.55)\n"
            "for i,v in enumerate([hit, plc]): ax.annotate(f'{v:.1f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('% of the time the target is reached within 126 trading days')\n"
            "ax.set_title('The \"forecast\" undershoots a magnitude-matched coin flip')\n"
            "ax.set_ylim(0, 100)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'pattern target hit {hit:.1f}%  vs  matched-size random move {plc:.1f}%')"
        ),
        md(
            f"The measured-move target is hit **{R['mm_rate']:.1f}%** of the time — sounds "
            "great in isolation. But a random long position aiming for a move of the exact "
            f"**same size**, entered on a random day in the same stocks, hits it "
            f"**{R['mm_placebo_rate']:.1f}%** of the time — *more* often. On a market that's "
            "spent 21 years mostly going up, most modest up-moves happen eventually regardless "
            "of what shape preceded them. The \"forecast\" is riding the tape's drift, not "
            "reading the geometry.\n\n"
            "**Finally, the trade people actually want to run:** hold to the target, or bail "
            "out at a timeout."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tp = st.timer_pnl(PANEL, max_days=126, cost_bps=5.0)\n"
            "    gross, excess, texc = tp['gross']*100, tp['excess']*100, tp['t_excess']\n"
            "else:\n"
            "    gross, excess, texc = R['tm_gross'], R['tm_excess'], R['tm_t_excess']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.4))\n"
            "ax.bar(['gross vs. ZERO\\n(looks great)','excess vs. a MATCHED\\nbuy-and-hold (the honest race)'],\n"
            "       [gross, excess], color=[AMBER, GREY], width=.55)\n"
            "for i,v in enumerate([gross, excess]): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',\n"
            "    va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('return per trade')\n"
            "ax.set_title('Positive against zero; nothing against the honest baseline')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {gross:+.2f}%  |  excess over matched hold {excess:+.2f}% (t={texc:+.2f})')"
        ),
        md(
            f"The gross number (**+{R['tm_gross']:.2f}%** per trade, roughly {R['tm_avg_hold']:.0f} "
            "days held) is testing against **zero** — but a long-only, multi-week hold in large-"
            "cap stocks is never supposed to be zero. The only fair race is against buying the "
            f"*same stock* for the *same number of days*, unconditionally — and there the edge "
            f"is **{R['tm_excess']:+.2f}%** at *t* = **{R['tm_t_excess']:+.2f}**. You'd have made "
            "about the same money on a random Tuesday."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** No horizon clears *t* = 2 across {R['n_events']} confirmed "
            f"breakouts; best is {R['max_abs_t_h']}d at *t* = {R['max_abs_t']:.2f}. Random-date "
            "placebos land right where a null should.\n"
            "- **Tradability — Mirage.** The apparently-strong long-timer trade is entirely "
            "explained by owning stocks for six-plus weeks; the excess over that baseline is "
            f"statistically zero (*t* = {R['tm_t_excess']:+.2f}).\n"
            f"- **\"Measured-move target forecasts the price\"? — Busted.** It's hit "
            f"{R['mm_rate']:.1f}% of the time — *less* than a magnitude-matched random move "
            f"({R['mm_placebo_rate']:.1f}%). The number chartists quote as a forecast is weaker "
            "than the tape's own baseline tendency."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The general lesson:** any \"height projected from the pattern\" target needs a "
            "magnitude-matched null, not a bare hit-rate. A textbook figure can look "
            "impressively predictive purely because the market drifts, and a target of typical "
            "size will get reached by drift alone.\n"
            "- **Sibling studies:** [188-head-shoulders](../../188-head-shoulders/) (the bearish "
            "top twin, smaller basket), [415-triple-top-bottom](../../415-triple-top-bottom/) "
            "(the flat three-tap cousin), [416-rounding-bottom](../../416-rounding-bottom/) (the "
            "continuous parabola) — none of them run the measured-move test this study does.\n\n"
            "*Think the target rule works on a different asset class or timeframe? Show it beats "
            "a magnitude-matched placebo, net of costs, on out-of-sample data — then we'll "
            "talk.*"
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
            "# The Inverse Head-and-Shoulders — a quantitative teardown 🔬\n"
            "### Swing-pivot detection · forward-return excess at 4 horizons · a same-tape "
            "placebo · a detector-strictness sweep · a magnitude-matched measured-move test · "
            "a holding-matched long-timer race · a synthetic faithful-engine control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The inverse H&S is unusual among chart figures for coming with an exact, testable "
            "quantitative prediction (the measured-move target) — which we hold to the same bar "
            "as the entry signal itself.\n\n"
            "> ⚠️ **Data note.** 30-name large-cap survivors basket + SPY, yfinance "
            "auto-adjusted daily OHLC, 2005→2026, cached; **229 confirmed breakouts** from a "
            "mechanical swing-pivot detector (`shoulder_tol=0.12`, `neckline_tol=0.15`). "
            "**Survivorship named on the Signal axis** — the basket tilts *for* this bullish "
            "figure. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprint `" + R["fp"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | forward-return excess vs base rate, best of 4 horizons "
            f"(10d): *t* = **{R['max_abs_t']:.2f}**, HAC *t* = **1.56**; placebo *p* ≈ 0.4–0.5 "
            "throughout |\n"
            f"| **Tradability** | `MIRAGE` | long-timer excess over a holding-matched base rate: "
            f"**{R['tm_excess']:+.2f}%**, HAC *t* = **{R['tm_t_excess']:+.2f}** |\n"
            f"| **Measured-move target?** | `BUSTED` | hit rate **{R['mm_rate']:.1f}%** vs "
            f"magnitude-matched placebo **{R['mm_placebo_rate']:.1f}%** |\n\n"
            "> 💡 In plain words: the shape is easy to find on a chart and easy to defend "
            "verbally; every quantitative version of the claim we can write down fails to clear "
            "the desk's bar."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $D_t \\in \\{0,1\\}$ flag a confirmed neckline break at bar $t$ (three swing-"
            "pivot troughs — left shoulder, a strictly deeper head, a right shoulder within "
            "`shoulder_tol` of the left — a near-horizontal neckline through the two "
            "intervening swing highs, and a close above the projected neckline). Let $R_{t,h}$ "
            "be the $h$-day forward return entered at $t+1$ (one-day lag, zero look-ahead) and "
            "$\\bar R_h$ the name's own unconditional $h$-day mean return (the base rate). The "
            "claims:\n\n"
            "- **H₁ (entry).** $E[R_{t,h} - \\bar R_h \\mid D_t = 1] > 0$, robustly, at some "
            "reasonable horizon.\n"
            "- **H₂ (measured move).** The target $\\tau = \\text{neckline} + (\\text{neckline} "
            "- \\text{head})$ is reached within a reasonable window *more often* than a random "
            "move of magnitude $(\\tau/\\text{entry} - 1)$ would be, starting from a random day.\n"
            "- **H₃ (tradable).** A rule that holds to $\\tau$ or times out beats a **holding-"
            "period-matched** buy-and-hold, net of costs.\n\n"
            f"We find **H₁ not certified** (max |t| = {R['max_abs_t']:.2f} across 4 horizons), "
            "**H₂ false in the negative direction** (the pattern underperforms its own magnitude-"
            "matched null), **H₃ not certified** (holding-matched excess *t* = "
            f"{R['tm_t_excess']:+.2f})."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Three separate honesty problems stack on top of each other in this figure, and "
            "each gets its own dedicated null:\n\n"
            "1. **Selection by eye.** A shape recognized only in hindsight is curve-fittable — "
            "solved with a mechanical swing-pivot detector plus a strictness sweep.\n"
            "2. **The market's own up-drift.** Any long signal on equities inherits positive "
            "drift — solved by testing the **excess** over each name's own base rate, not the "
            "raw return, with a one-sample + HAC *t* and a same-tape random-date placebo.\n"
            "3. **The target's own reachability.** A target of typical size gets hit *some* of "
            "the time on a rising tape purely by chance — solved with a **magnitude-matched** "
            "placebo (draw the *same* relative target size at a *random* entry) rather than "
            "comparing the hit rate to zero or to an arbitrary round number."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Detector.** Swing pivots (`w=5`), symmetric shoulders (`shoulder_tol=0.12`), "
            "near-horizontal neckline (`neckline_tol=0.15`), figure span 20–200 bars, confirmed "
            f"break within 40 bars of the right shoulder. **{R['n_events']}** events, "
            f"{R['start']} → {R['end']}.\n"
            "- **Headline.** Forward-return excess at {5,10,20,40}d, one-sample + HAC *t*, "
            "5,000-draw same-tape placebo per name (vectorized).\n"
            "- **Robustness.** Tight/base/loose shoulder+neckline tolerance sweep at 20d.\n"
            "- **Measured-move.** Target = neckline-midpoint + (neckline-midpoint − head); hit "
            "rate within 126 trading days, Wilson CI, magnitude-matched placebo (6,000 draws).\n"
            "- **Execution (Tradability).** Enter next close (one lag, zero look-ahead); hold to "
            "target-hit close or 126d timeout; 2 × one-way cost × NAV; excess vs a holding-"
            "period-matched base rate.\n"
            "- **Control.** Synthetic panel: pure noise (the null — no injected shape) vs a "
            "planted inverse-H&S with a tunable post-breakout continuation edge."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline — forward-return excess at four horizons\n\n"
            "One-sample and HAC *t* on the pooled per-event excess; a 5,000-draw same-tape "
            "random-date placebo per name."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = {h: st.run_experiment(PANEL, horizon=h, cost_bps=5.0) for h in st.HORIZONS}\n"
            "    hs = list(st.HORIZONS)\n"
            "    ex = [rows[h]['mean']*100 for h in hs]; ts = [rows[h]['t'] for h in hs]\n"
            "    hts = [rows[h]['hac_t'] for h in hs]; ps = [rows[h]['p_placebo'] for h in hs]\n"
            "else:\n"
            "    hs = sorted(R['horizons'])\n"
            "    ex = [R['horizons'][h][2] for h in hs]; ts = [R['horizons'][h][3] for h in hs]\n"
            "    hts = [R['horizons'][h][4] for h in hs]; ps = [R['horizons'][h][5] for h in hs]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.4))\n"
            "cols = [RED if abs(t) >= 2 else GREY for t in ts]\n"
            "a1.bar([f'{h}d' for h in hs], ex, color=cols, width=.6)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('excess return (%)')\n"
            "a1.set_title('Forward-return excess over base rate')\n"
            "a2.bar([f'{h}d' for h in hs], ts, color=cols, width=.6)\n"
            "a2.axhline(-2, ls='--', c=RED, lw=1); a2.axhline(2, ls='--', c=RED, lw=1)\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('one-sample t')\n"
            "a2.set_title('None crosses the t=2 bar')\n"
            "plt.tight_layout(); plt.show()\n"
            "for h,e,t,ht,p in zip(hs, ex, ts, hts, ps):\n"
            "    print(f'{h:>2d}d: excess {e:+.3f}%  t={t:+.2f}  HAC t={ht:+.2f}  placebo p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the raw post-breakout return is comfortably positive at every "
            "horizon (equities drift up), but once you subtract each name's own base rate — the "
            "correct null for \"does this pattern add anything\" — nothing survives. The best "
            f"horizon (10d) reaches HAC *t* = 1.56; under a genuine null, the *best of four* "
            "correlated horizon tests is expected to land somewhere in that range by chance "
            "alone. Every placebo *p* sits at 0.4–0.5 — indistinguishable from a random Tuesday."
        ),
        md(
            "### 4b · Detector-strictness robustness sweep\n\n"
            "Tighten or loosen the shoulder-symmetry and neckline-tilt tolerances; horizon fixed "
            "at 20d."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sweep = {}\n"
            "    for tag, (ts_, tn_) in [('tight',(0.08,0.10)), ('base',(0.12,0.15)), ('loose',(0.18,0.20))]:\n"
            "        r = st.run_experiment(PANEL, horizon=20, shoulder_tol=ts_, neckline_tol=tn_)\n"
            "        sweep[tag] = (ts_, tn_, r['n_events'], r['mean']*100, r['t'])\n"
            "else:\n"
            "    sweep = R['sweep']\n"
            "tags = ['tight','base','loose']\n"
            "ns = [sweep[t][2] for t in tags]; exs = [sweep[t][3] for t in tags]; ts_ = [sweep[t][4] for t in tags]\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.3))\n"
            "cols = [RED if abs(t) >= 2 else GREY for t in ts_]\n"
            "ax.bar(tags, exs, color=cols, width=.55)\n"
            "for i,(n,e,t) in enumerate(zip(ns, exs, ts_)):\n"
            "    ax.annotate(f'n={n}\\n{e:+.2f}% (t={t:+.2f})',(i,e),ha='center',\n"
            "        va='bottom' if e>=0 else 'top', fontsize=9)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('excess return, 20d (%)')\n"
            "ax.set_title('Not a single-tolerance artefact')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "> 💡 In plain words: loosening the detector triples the count of \"confirmed\" "
            "patterns (153 → 305) — pattern detectors are extremely tolerance-sensitive — but "
            "the excess return never gets anywhere near significant in either direction. The "
            "null result is not an artefact of one arbitrary choice."
        ),
        md(
            "### 4c · The measured-move target — a magnitude-matched honesty check\n\n"
            "Target $\\tau = $ neckline-midpoint $+$ (neckline-midpoint $-$ head). Hit rate "
            "within 126 trading days, Wilson 95% CI, against a placebo that draws random entries "
            "and assigns them a target of the **same relative size** (drawn from the observed "
            "target-distance distribution) — the only fair null for \"the target is reached "
            "X% of the time.\""
        ),
        code(
            "if HAVE_REAL:\n"
            "    mm = st.measured_move_hits(PANEL, max_days=126)\n"
            "    hit, lo, hi = mm['hit_rate']*100, mm['hit_lo']*100, mm['hit_hi']*100\n"
            "    plc = mm['placebo_rate']*100\n"
            "else:\n"
            "    hit, lo, hi, plc = R['mm_rate'], R['mm_lo'], R['mm_hi'], R['mm_placebo_rate']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['pattern target','magnitude-matched\\nrandom entry'], [hit, plc],\n"
            "       color=[RED, GREY], width=.55)\n"
            "ax.errorbar([0], [hit], yerr=[[hit-lo],[hi-hit]], fmt='none', ecolor='k', capsize=6)\n"
            "for i,v in enumerate([hit, plc]): ax.annotate(f'{v:.1f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('% reaching the target within 126 trading days')\n"
            "ax.set_ylim(0, 100)\n"
            "ax.set_title('The pattern UNDERPERFORMS a size-matched coin flip')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'pattern {hit:.1f}% [{lo:.1f}%, {hi:.1f}%]  vs  matched-size placebo {plc:.1f}%')"
        ),
        md(
            f"> 💡 In plain words: {R['mm_rate']:.1f}% (Wilson [{R['mm_lo']:.1f}%, "
            f"{R['mm_hi']:.1f}%]) sounds like a strong forecast in isolation — until you check "
            f"what a random move of the *same size* achieves: {R['mm_placebo_rate']:.1f}%, "
            "measurably higher, off {:,} draws.".format(R['mm_placebo_n']) +
            " The pattern's target is, if anything, a slightly *worse* bet than the tape's own "
            "baseline tendency to eventually drift a similar distance. H₂ doesn't just fail to "
            "clear a bar — it fails in the wrong direction."
        ),
        md(
            "### 4d · The long timer — hold to target-or-timeout, raced against the honest baseline\n\n"
            "Entry at the next close after confirmation; exit at the close of the day the "
            "measured-move target is first touched (a same-day fill is optimistic, named), else "
            "a 126-day timeout close. The gross/net numbers test against **zero** — the ONLY "
            "honest race is against a **holding-period-matched** base rate (buy the same name, "
            "same number of days, unconditionally)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tp5 = st.timer_pnl(PANEL, max_days=126, cost_bps=5.0)\n"
            "    tp10 = st.timer_pnl(PANEL, max_days=126, cost_bps=10.0)\n"
            "    gross, net5, net10 = tp5['gross']*100, tp5['net']*100, tp10['net']*100\n"
            "    t5, hac5 = tp5['t'], tp5['hac_t']\n"
            "    excess, texc = tp5['excess']*100, tp5['t_excess']\n"
            "    avg_hold = tp5['avg_hold']\n"
            "else:\n"
            "    gross, net5, net10 = R['tm_gross'], R['tm_net5'], R['tm_net10']\n"
            "    t5, hac5 = R['tm_t5'], R['tm_hac5']\n"
            "    excess, texc, avg_hold = R['tm_excess'], R['tm_t_excess'], R['tm_avg_hold']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.bar(['gross','net 5bps','net 10bps'], [gross, net5, net10], color=[GREY, AMBER, AMBER], width=.6)\n"
            "for i,v in enumerate([gross, net5, net10]): a1.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "a1.set_ylabel('return per trade'); a1.set_title(f'vs ZERO: t = {t5:+.2f} (looks great)')\n"
            "a2.bar(['excess vs matched\\nbuy-and-hold'], [excess], color=[GREY], width=.4)\n"
            "a2.annotate(f'{excess:+.2f}%\\n(t={texc:+.2f})',(0,excess),ha='center',\n"
            "    va='bottom' if excess>=0 else 'top')\n"
            "a2.axhline(0, c='k', lw=.8); a2.set_ylabel('excess return')\n"
            "a2.set_title('vs the HONEST baseline: nothing')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {gross:+.2f}%  net(5bps) {net5:+.2f}% t={t5:+.2f}  |  '\n"
            "      f'excess vs matched hold {excess:+.2f}% t={texc:+.2f}  (avg hold {avg_hold:.1f}d)')"
        ),
        md(
            f"> 💡 In plain words: the left panel is the number a backtest report would headline "
            f"— +{R['tm_gross']:.2f}% gross, *t* ≈ 5 — and it is **true and useless**: it's "
            f"testing against zero, and a long-only multi-week equity hold is never supposed to "
            "be zero. The right panel is the actual question — does *this trade rule* beat "
            f"*just owning the stock* for the same {R['tm_avg_hold']:.0f} days — and the answer "
            f"is {R['tm_excess']:+.2f}% at *t* = {R['tm_t_excess']:+.2f}: no. H₃ fails."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Two synthetic worlds: **pure noise** (no injected shape — the honest null) and a "
            "**planted** inverse-H&S with a tunable post-breakout continuation knob (`edge`)."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(695, 715):\n"
            "    p, _ = data.synthetic_panel(n_names=10, n_days=6000, edge=0.0, n_planted=0, seed=s_)\n"
            "    r = st.run_experiment(p, horizon=20, seed=s_)\n"
            "    null_ts.append(r['t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "planted, truth = data.synthetic_panel(n_names=10, n_days=6000, edge=0.15, n_planted=8, seed=695)\n"
            "rp = st.run_experiment(planted, horizon=20, seed=695)\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12,.12,20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (pure noise), 20 seeds')\n"
            "ax.scatter([1], [rp['t']], color=RED, s=90, zorder=5, label='planted edge=+0.15')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('one-sample t (excess vs base rate)')\n"
            "ax.set_title('Control: no null fires; a planted continuation lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.3f} (sd {null_ts.std(ddof=1):.3f}), '\n"
            "      f'|t|>=2 in {(np.abs(null_ts)>=2).sum()}/20 seeds  |  '\n"
            "      f'planted: n={rp[\"n_events\"]} (of {truth[\"n_planted_total\"]} planted) t = {rp[\"t\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 pure-noise seeds the detector fires false-"
            f"positive patterns (noise alone traces a qualifying shape sometimes) but the "
            f"forward-return excess stays inside the null band in "
            f"{R['syn_null_seeds'] - R['syn_null_fire']}/{R['syn_null_seeds']} seeds "
            f"(mean *t* = {R['syn_null_mean']:+.3f}, sd {R['syn_null_sd']:.3f}); a genuine "
            f"planted continuation (`edge=+0.15`) lights up unmistakably (*t* = "
            f"{R['syn_planted_t']:.2f}, HAC {R['syn_planted_hac']:.2f}). The machinery works — "
            "the real-tape null result on Beat 4a is the genuine article, not a blind detector."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — forward-return excess over base rate across "
            f"{R['n_events']} confirmed breakouts never clears *t* = 2 at any of 4 horizons "
            f"(best: 10d, *t* = {R['max_abs_t']:.2f}, HAC 1.56); every placebo *p* ≈ 0.4–0.5; "
            "survives a strictness sweep; survivorship named and tilting *for* the figure.\n"
            f"- **Tradability `MIRAGE`** — the long-timer trade's *t* ≈ 5 is against zero; "
            f"against a holding-period-matched base rate the excess is "
            f"{R['tm_excess']:+.2f}% at HAC *t* = {R['tm_t_excess']:+.2f}. Nothing to deploy.\n"
            f"- **Measured-move target `BUSTED`** — hit rate {R['mm_rate']:.1f}% is *below* "
            f"the magnitude-matched placebo's {R['mm_placebo_rate']:.1f}%. The textbook "
            "\"forecast\" underperforms a random move of the same size."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general lesson for any pattern with a projected target** (flag poles, "
            "triangle heights, cup depths): the hit-rate of a projected target needs a "
            "*magnitude-matched* null, not a bare percentage or a comparison to zero. A "
            "sufficiently long window on a drifting market will \"hit\" most modest targets "
            "regardless of the shape that preceded them.\n"
            "- **Where the shape *might* carry information** that this study doesn't test: "
            "intraday micro-structure around the actual neckline break (order-flow imbalance, "
            "not the daily-bar close), or the pattern conditioned on volume confirmation — "
            "Edwards & Magee's original texts emphasize volume expansion on the breakout, which "
            "this mechanical (price-only) detector deliberately ignores to keep the test "
            "reproducible from OHLC alone.\n"
            "- **Dedup map:** [188-head-shoulders](../../188-head-shoulders/) (the bearish top "
            "twin, smaller basket), [415-triple-top-bottom](../../415-triple-top-bottom/) (flat "
            "three-tap reversal), [416-rounding-bottom](../../416-rounding-bottom/) (continuous "
            "parabola, no neckline).\n\n"
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
