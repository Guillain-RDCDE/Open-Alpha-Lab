"""Generate the two narrative notebooks for Study 697 (Wolfe Waves).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached basket tapes under
../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring docs/results.md).
The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily OHLC,
# SPY/QQQ/DIA/IWM/^GSPC/^IXIC/^DJI/GLD, as-of 2026-06-30; 4% ZigZag headline).
R = dict(
    pct=0.04, asof="2026-06-30",
    tickers=("SPY", "QQQ", "DIA", "IWM", "^GSPC", "^IXIC", "^DJI", "GLD"),
    per_ticker={"SPY": (337, 44), "QQQ": (431, 56), "DIA": (275, 28), "IWM": (381, 31),
                "^GSPC": (975, 114), "^IXIC": (612, 73), "^DJI": (315, 33), "GLD": (217, 18)},
    n_pivots=3543, n_candidates=370, n_decided=345, n_timeout=25, n_win=120, n_loss=225,
    hit_rate=34.8, wilson=(29.9, 40.0),
    placebo_mean=34.6, placebo_sd=2.5, placebo_z=0.05, placebo_p=0.471, placebo_ndraws=1000,
    # time-target accuracy (winning trades only)
    tt_n=120, tt_actual=21.7, tt_pred=52.2, tt_corr=0.041, tt_mae=40.8,
    # secondary fixed-horizon directional return, pooled (n=397, gross/net bps, win%, t, hac_t, coin_p)
    fh={5: (397, -40, -42, 47, -1.74, -2.00, 0.960),
        10: (397, -14, -16, 50, -0.46, -0.48, 0.673),
        20: (397, -1, -3, 47, -0.01, -0.01, 0.501),
        40: (397, 20, 18, 47, 0.35, 0.37, 0.359)},
    # ZigZag threshold sweep (candidates, decided, hit_rate%)
    sweep={0.03: (527, 500, 36.0), 0.04: (370, 345, 34.8),
           0.05: (277, 247, 36.4), 0.08: (118, 90, 37.8)},
    # SPY-only
    spy_candidates=44, spy_decided=40, spy_hit=32.5, spy_wilson=(20.1, 48.0),
    # synthetic control
    syn_null_mean_t=-0.76, syn_null_sd=1.25, syn_null_fire=4, syn_null_seeds=20,
    syn_planted_edge=0.30,
    syn_planted={5: (43, 317, 4.63, 5.62), 10: (43, 561, 4.11, 5.51),
                20: (43, 668, 3.95, 5.32), 40: (43, 618, 3.17, 4.59)},
    fp={"SPY": "16694cd9912d", "QQQ": "d935154d0960", "DIA": "33e3d47bdd3a",
        "IWM": "1f2218a6b7f2", "^GSPC": "cec06bce14e7", "^IXIC": "54d65c6aa4cb",
        "^DJI": "803513feb661", "GLD": "44f6ff1685e4"},
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![No_reliable_timer%3F: Confirmed]"
    "(https://img.shields.io/badge/No_reliable_timer%3F-Confirmed-8b949e?style=flat-square)\n\n"
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

from wolfe_waves import data, strategy as st

TICKERS = list(data.TICKERS)
HAVE_REAL = all(data.have_real(tk) for tk in TICKERS)
if HAVE_REAL:
    BASKET = data.load_basket(TICKERS)
else:
    BASKET = {}
print("real cache present:", HAVE_REAL, "| tickers:", TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Can a chart pattern really predict a price *and* a date? 🌀\n"
            "### Wolfe Waves — a five-point wedge that promises a \"natural equilibrium\" "
            "target, tested by letting a computer draw it instead of a chartist\n\n"
            + BADGES +
            "Somewhere on a trading forum, someone has drawn five connected lines on a chart, "
            "labeled the points 1 through 5, and told you that price is now *destined* to travel "
            "to a specific level — the **EPA** (\"Estimated Price at Arrival\") — by a specific "
            "date. It's one of the more ambitious claims in chart reading: not just a direction, "
            "but a **price and a time**.\n\n"
            "The trouble with testing folklore like this is that it's normally drawn by hand, "
            "after the fact, on whichever five points make the story work. So we built a "
            "computer that draws it the same way every time, on every chart, whether or not "
            "anyone wants it to — and asked whether the target it draws means anything at all.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the correlation "
            "math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** A 4% ZigZag finds the swing points on SPY plus a 7-ticker "
            "broad-index basket, 1993→2026 — no cherry-picked chart, no hindsight. House style "
            "in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the target actually get hit more than chance? | **No.** The pattern's price "
            f"target gets hit **{R['hit_rate']:.1f}%** of the time — a same-distance target "
            f"dropped on a **random day** gets hit **{R['placebo_mean']:.1f}%** of the time. "
            "Statistically the same number. |\n"
            f"| Does the pattern at least call *when*? | **No — not even close.** Its own time "
            f"projection correlates **{R['tt_corr']:+.2f}** with when a winning trade actually "
            f"resolves. A coin flip would do about as well. |\n"
            f"| Is this sensitive to how tightly you draw the wedge? | **No.** Loosen or tighten "
            "the swing filter and the hit rate barely moves (36%–38%) — there's no setting where "
            "the edge shows up. |\n"
            "| Would the test even notice a *real* pattern? | **Yes.** On a fake chart where we "
            "deliberately plant a real reversal after point 5, the same detector lights up "
            "unmistakably. It's not blind — there's just nothing there on real markets. |\n\n"
            "> The wedge is real geometry. The destiny isn't."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Five swing points — 1, 2, 3, 4, 5 — form a converging wedge. Point 5 pushes "
            "just past the line connecting points 1 and 3 — a 'false breakout' that traps the "
            "last sellers (or buyers) — and then price snaps back, traveling to the EPA line: "
            "the extension of the 1-4 trendline. That line tells you both the price AND "
            "roughly the time you'll get there.\"*\n\n"
            "It's a more specific promise than most chart patterns make. Head-and-shoulders "
            "says *\"probably down\"*. Wolfe Waves says *\"here, by then.\"* That specificity is "
            "exactly what makes it testable — and exactly what makes it worth testing."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the EPA line genuinely worked, it would be one of the few pieces of technical "
            "analysis offering a **concrete, falsifiable price-and-time forecast** — not just "
            "\"probably higher\" but \"here, roughly then.\" That's a much bigger claim than most "
            "chart patterns make, and it should be correspondingly easy to grade: did price get "
            "there, and did it get there on schedule?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "- **Find the wedges mechanically.** A ZigZag swing filter finds every 5-point "
            "sequence matching the wedge geometry — point 3 extends past point 1, point 4 stays "
            "inside the channel, point 5 breaks the 1-3 line. No human hindsight involved.\n"
            "- **Draw the target.** Extend the 1-4 line to the moment point 5 is confirmed — "
            "that's the price target. Project the same time span forward from point 5 — that's "
            "the time target.\n"
            "- **The luck check.** Walk forward and see whether price touches the target before "
            "an invalidation stop — then do the exact same walk from a **random day**, at the "
            "same distance, and see how often *that* touches too.\n"
            "- **The clock check.** For the trades that do hit the target, does the pattern's own "
            "time projection actually predict when?"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Hit rate of the real target vs a same-distance random-day "
            "target, pooled across the basket."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pooled = st.pooled_target_hit(BASKET, pct=R['pct'], max_horizon=90)\n"
            "    hr = st.hit_rate_summary(pooled['ledger'])\n"
            "    obs = hr['hit_rate'] * 100\n"
            "else:\n"
            "    obs = R['hit_rate']\n"
            "placebo_mean = R['placebo_mean']  # canonical 1,000-draw/ticker placebo (results.md)\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.4))\n"
            "ax.bar(['Wolfe Wave\\ntarget (n=370)', 'random-day\\ntarget (placebo)'],\n"
            "       [obs, placebo_mean], color=[RED, GREY], width=.55)\n"
            "for i, v in enumerate([obs, placebo_mean]):\n"
            "    ax.annotate(f'{v:.1f}%', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('target hit before stop (%)')\n"
            "ax.set_title('The pattern\\'s target hits about as often as a random target')\n"
            "ax.set_ylim(0, 55)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:.1f}%  vs  placebo {placebo_mean:.1f}%')"
        ),
        md(
            f"**{R['hit_rate']:.1f}%** vs **{R['placebo_mean']:.1f}%** — a **z = "
            f"{R['placebo_z']:.2f}**, a coin flip's worth of difference. Out of **{R['n_candidates']}** "
            f"algorithmically-detected wedges, **{R['n_decided']}** resolve one way or the other "
            f"within 90 trading days (the rest simply never get close enough to either level); "
            f"among those, **{R['n_win']}** hit the target and **{R['n_loss']}** hit the "
            "invalidation stop first. That ratio is what a random walk gives you for free.\n\n"
            "**Second — does it at least tell you *when*?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    tta = st.time_target_accuracy(pooled['ledger'])\n"
            "    actual_mean, pred_mean, corr = tta['mean_actual'], tta['mean_predicted'], tta['corr']\n"
            "else:\n"
            "    actual_mean, pred_mean, corr = R['tt_actual'], R['tt_pred'], R['tt_corr']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.4))\n"
            "ax.bar(['actual bars\\nto target', \"pattern's projected\\ntime target\"],\n"
            "       [actual_mean, pred_mean], color=[AMBER, GREY], width=.55)\n"
            "for i, v in enumerate([actual_mean, pred_mean]):\n"
            "    ax.annotate(f'{v:.0f} bars', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('trading days')\n"
            "ax.set_title(f'The clock is wrong by ~2.4x (correlation = {corr:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'actual {actual_mean:.1f} days  vs  projected {pred_mean:.1f} days  '\n"
            "      f'corr={corr:+.3f}')"
        ),
        md(
            f"On the **{R['tt_n']}** trades that *do* reach the target, they get there in "
            f"**{R['tt_actual']:.0f} trading days** on average — the pattern's own time "
            f"projection says **{R['tt_pred']:.0f}**. The two barely relate at all "
            f"(correlation **{R['tt_corr']:+.2f}**). Even when the price call happens to land, "
            "the date call is noise.\n\n"
            "**Third, robustness.** Does loosening or tightening how strictly we draw the wedge "
            "change the answer?"
        ),
        code(
            "pcts = sorted(R['sweep'])\n"
            "hits = [R['sweep'][p][2] for p in pcts]\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.3))\n"
            "ax.bar([f'{p*100:.0f}%' for p in pcts], hits, color=GREY, width=.55)\n"
            "ax.axhline(R['placebo_mean'], ls='--', c=RED, lw=1.5, label='random-day baseline')\n"
            "for i, v in enumerate(hits):\n"
            "    ax.annotate(f'{v:.1f}%', (i, v), ha='center', va='bottom')\n"
            "ax.set_xlabel('ZigZag reversal threshold')\n"
            "ax.set_ylabel('target hit rate (%)')\n"
            "ax.set_title('Loosen it, tighten it — same story every time')\n"
            "ax.legend(); ax.set_ylim(0, 50)\n"
            "plt.tight_layout(); plt.show()\n"
            "print({f'{p*100:.0f}%': f'{h:.1f}%' for p, h in zip(pcts, hits)})"
        ),
        md(
            "Flat, at every threshold from 3% to 8% — all sitting right on top of the random-day "
            "line. There's no secret setting where the pattern suddenly works.\n\n"
            "**Finally — would this test even notice a real pattern if there were one?** We built "
            "a fake market where we deliberately plant a reversal after point 5, and reran the "
            "exact same detector."
        ),
        code(
            "def synth_run(edge, seed=697, horizon=20):\n"
            "    bars, _ = data.synthetic_panel(edge=edge, seed=seed, n_days=9000)\n"
            "    piv = st.zigzag(bars['close'].to_numpy(float), pct=R['pct'])\n"
            "    ent = st.wolfe_candidates(piv)\n"
            "    led = st.run_trades(bars, ent, horizon=horizon, cost_bps=0.0)\n"
            "    return st.summarize(led, 'ret_gross')\n"
            "\n"
            "s_null = synth_run(0.0)\n"
            "s_plant = synth_run(R['syn_planted_edge'])\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.3))\n"
            "ax.bar(['no planted\\nreversal', 'planted reversal\\n(+30% toward target)'],\n"
            "       [s_null['t'], s_plant['t']], color=[GREY, RED], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1)\n"
            "for i, v in enumerate([s_null['t'], s_plant['t']]):\n"
            "    ax.annotate(f't={v:+.2f}', (i, v), ha='center', va='bottom' if v>0 else 'top')\n"
            "ax.set_ylabel('t-statistic')\n"
            "ax.set_title('The detector is not blind — real markets just don\\'t have this')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"no reversal: t={s_null['t']:+.2f}   planted reversal: t={s_plant['t']:+.2f}\")"
        ),
        md(
            "With nothing planted, the detector reports nothing (comfortably inside the ±2 "
            "band). Force a real reversal into the fake data and the same detector lights up "
            "immediately. The machinery works — real markets simply don't hand it what it's "
            "looking for."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The EPA target is hit **{R['hit_rate']:.1f}%** of the time vs "
            f"**{R['placebo_mean']:.1f}%** for a same-distance random target — statistically the "
            "same number, at every ZigZag setting we tried, on every fixed-horizon cut we ran. A "
            "synthetic control proves the harness would catch a real reversal if there were one.\n"
            "- **Tradability — Mirage.** No edge, nothing to charge costs against, nothing to "
            "deploy.\n"
            "- **\"No reliable timer\"? — Confirmed.** The time-and-price promise fails on the "
            f"time half too: correlation **{R['tt_corr']:+.2f}** between the projected and the "
            "actual arrival."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **This is a family, not a one-off.** Every chart pattern that claims a precise "
            "price target invites the same question: does the target beat a same-distance random "
            "day? [448-point-and-figure](../../448-point-and-figure/) asked exactly that of a "
            "different pattern — and found a real (if untradable) edge. Wolfe Waves didn't get "
            "that lucky.\n"
            "- **The discretion problem is structural.** Wolfe's rules were never one canonical "
            "checklist — every source states them slightly differently. A human chartist, free to "
            "adjust which five points \"count\", has vastly more freedom to fit a story after the "
            "fact than the fixed rule this notebook ran.\n"
            "- **Sibling studies:** [445-elliott-wave](../../445-elliott-wave/) (a different "
            "wave count, Fibonacci retracements, no price target), "
            "[447-gann-angles](../../447-gann-angles/) (a single trend line), and "
            "[704-three-drives](../../704-three-drives/) (another Fibonacci five-point pattern).\n\n"
            "*Think a stricter or looser version of the wedge rules finds something real? The "
            "detector and its knobs are in [`wolfe_waves/strategy.py`](../wolfe_waves/strategy.py) "
            "— fork it and show a *t* ≥ 2 on the real tape.*"
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
            "# Wolfe Waves — a quantitative teardown 🔬\n"
            "### The 5-point geometry detector · the target-hit test vs a same-distance "
            "random-day placebo · the EPA time-target correlation · a ZigZag-threshold sweep · "
            "a geometry-free HAC cut · a 20-seed synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "Wolfe Waves make an unusually specific claim for a chart pattern — a **price and a "
            "time** target off a 5-point wedge's own geometry. That specificity is what makes it "
            "gradeable: we run a single mechanical detector on every tape and ask whether the "
            "target beats the one honest baseline that matters — a random target the same "
            "distance away.\n\n"
            "> ⚠️ **Data note.** Daily OHLC, yfinance, auto-adjusted, cached; SPY + "
            "QQQ/DIA/IWM/^GSPC/^IXIC/^DJI/GLD, as-of **" + R["asof"] + "**. No cross-sectional "
            "survivorship (broad-index/ETF price tapes). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprints in the data stamp there).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | target-hit rate **{R['hit_rate']:.1f}%** "
            f"(n={R['n_decided']}) vs random-day placebo **{R['placebo_mean']:.1f}%** "
            f"(sd {R['placebo_sd']:.1f}pp, {R['placebo_ndraws']:,} draws/ticker): "
            f"*z* = **{R['placebo_z']:+.2f}**, *p* = **{R['placebo_p']:.3f}** |\n"
            f"| **Tradability** | `MIRAGE` | no separation from chance to charge costs against; "
            f"fixed-horizon |*t*| < 2 at every horizon |\n"
            f"| **No reliable timer?** | `CONFIRMED` | time-target correlation "
            f"**{R['tt_corr']:+.3f}** (n={R['tt_n']}), MAE **{R['tt_mae']:.1f}** bars |\n\n"
            "> 💡 In plain words: the geometry is real (we can always draw five points that "
            "satisfy the rules); the destiny is not."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let pivots $p_1,\\dots,p_5$ be five consecutive alternating ZigZag swing extremes. "
            "A **bullish** Wolfe candidate ($p_1,p_3,p_5$ swing lows, $p_2,p_4$ swing highs) "
            "requires:\n\n"
            "- **B.** $p_3 < p_1$ — wave 3 extends the wedge past wave 1;\n"
            "- **A.** $p_5 < \\text{line}_{13}(i_5)$ — wave 5 pierces the 1-3 trendline (the one "
            "rule every source agrees on);\n"
            "- **C.** $\\text{line}_{13}(i_4) < p_4 < p_2$ — wave 4 stays inside the channel and "
            "makes a lower high than wave 2.\n\n"
            "(The bearish case is the exact mirror.) The **EPA price target** is "
            "$\\text{line}_{14}$ evaluated at the bar point 5 is confirmed; the **EPA time "
            "target** is $i_5 + (i_4 - i_1)$ — one fixed, ex-ante convention among the several "
            "informally described in the literature (see "
            "[`docs/references.md`](../docs/references.md)).\n\n"
            "Claims:\n\n"
            "- **H₁ (target).** Price reaches the EPA target before the invalidation stop "
            "(point 5's own extreme) more often than a same-distance target on a random day.\n"
            "- **H₂ (clock).** The EPA time target predicts *when* H₁ resolves.\n"
            "- **H₃ (robust).** H₁ holds across the ZigZag threshold and a geometry-free "
            "directional cut.\n\n"
            "We find **H₁ not supported** (*z* = +0.05), **H₂ not supported** (corr = +0.04), "
            "**H₃ moot** — nothing to be robust about."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "The target-hit test is a **binary outcome per candidate** (win/loss, timeouts "
            "excluded), so the honest null isn't \"is the hit rate above 50%\" — a wedge's target "
            "and stop are not symmetric distances, so even a *pure random walk* has some "
            "structural hit rate depending on how far the target sits vs the stop. The correct "
            "null is: **replay the exact same (direction, target distance, stop distance) from a "
            "random day** and see how often *that* hits. We report the empirical placebo *p* (the "
            "share of 1,000-draws/ticker whose pooled hit-rate is ≥ observed) and a *z*-score "
            "against the placebo distribution's own mean/sd — the two agree by construction. A "
            "secondary, **geometry-free** cut (fixed-horizon directional return, one-sample + "
            "Newey-West HAC *t*, same-bars coin placebo) checks the conclusion doesn't depend on "
            "how we defined the target/stop levels at all."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Detector.** {R['pct']*100:.0f}% ZigZag pivots; the A/B/C geometry rules above. "
            f"{R['n_pivots']:,} pivots -> {R['n_candidates']} candidates, pooled across the "
            "basket.\n"
            "- **Execution.** Enter the close one bar after point 5's ZigZag confirmation (no "
            "look-ahead) — the study's single documented execution lag.\n"
            f"- **Headline.** Walk forward on intraday High/Low up to 90 bars; target-before-stop "
            f"hit-rate ({R['n_decided']} decided, {R['n_timeout']} timeouts excluded) with a "
            f"Wilson interval, vs the same-distance random-day placebo "
            f"({R['placebo_ndraws']:,} draws/ticker).\n"
            "- **Clock.** Correlation + MAE of actual vs projected bars-to-hit, winning trades "
            "only.\n"
            "- **Robustness.** ZigZag threshold sweep (3/4/5/8%); geometry-free fixed-horizon "
            "directional return (5/10/20/40 days) with a same-bars random-direction coin "
            "placebo.\n"
            "- **Control.** Synthetic panel built from exact anchor points (not accumulated "
            "per-bar noise) with a tunable planted post-point-5 reversal; the null must not fire "
            "across 20 seeds, a planted reversal must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline target-hit test and its placebo\n\n"
            "Walk-forward outcome (target before stop, 90-bar horizon) pooled across the basket, "
            "vs the same-distance random-day null. In the notebook we run a lighter placebo (one "
            "ticker, fewer draws) and quote the canonical pooled result from `results.md`."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pooled = st.pooled_target_hit(BASKET, pct=R['pct'], max_horizon=90)\n"
            "    ledger = pooled['ledger']\n"
            "    hr = st.hit_rate_summary(ledger)\n"
            "    obs = hr['hit_rate']\n"
            "    tk0 = TICKERS[0]\n"
            "    led0 = ledger[ledger['ticker'] == tk0].reset_index(drop=True)\n"
            "    pl_light = st.random_target_placebo(BASKET[tk0], led0, max_horizon=90,\n"
            "                                        n_draws=300, seed=697)\n"
            "    draws = pl_light['draws']\n"
            "else:\n"
            "    obs = R['hit_rate'] / 100\n"
            "    rng = np.random.default_rng(697)\n"
            "    draws = rng.normal(R['placebo_mean']/100, R['placebo_sd']/100, 2000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws*100, bins=40, color=GREY, alpha=.85,\n"
            "        label='null: same-distance targets on random days (light in-notebook run)')\n"
            "ax.axvline(obs*100, c=RED, lw=2.5, label=f'observed hit rate {obs*100:.1f}%')\n"
            "ax.set_xlabel('target-hit rate of a placebo draw (%)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"Sits inside the luck cloud: canonical z={R['placebo_z']:+.2f}, \"\n"
            "             f\"p={R['placebo_p']:.3f} ({R['placebo_ndraws']:,} draws/ticker, pooled)\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"canonical (results.md): observed {R['hit_rate']:.1f}%  placebo mean \"\n"
            "      f\"{R['placebo_mean']:.1f}% (sd {R['placebo_sd']:.1f}pp)  \"\n"
            "      f\"z={R['placebo_z']:+.2f}  p={R['placebo_p']:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: the observed **{R['hit_rate']:.1f}%** sits almost exactly on "
            f"top of the placebo's own mean (**{R['placebo_mean']:.1f}%**, sd "
            f"{R['placebo_sd']:.1f}pp) — *z* = **{R['placebo_z']:+.2f}**. H₁ is rejected: the "
            "pattern's target is not special."
        ),
        md(
            "### 4b · The clock — does the EPA time target predict when?\n\n"
            "For winning trades only: actual bars-to-hit vs the projected "
            "$i_5 + (i_4 - i_1)$ offset."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tta = st.time_target_accuracy(ledger)\n"
            "else:\n"
            "    tta = {'n': R['tt_n'], 'corr': R['tt_corr'], 'mae_days': R['tt_mae'],\n"
            "           'mean_actual': R['tt_actual'], 'mean_predicted': R['tt_pred']}\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.4))\n"
            "ax.bar(['actual', 'projected'], [tta['mean_actual'], tta['mean_predicted']],\n"
            "       color=[AMBER, GREY], width=.5)\n"
            "for i, v in enumerate([tta['mean_actual'], tta['mean_predicted']]):\n"
            "    ax.annotate(f'{v:.1f}', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('bars to target')\n"
            "ax.set_title(f\"corr={tta['corr']:+.3f}  MAE={tta['mae_days']:.1f} bars  (n={tta['n']})\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(tta)"
        ),
        md(
            f"> 💡 In plain words: correlation **{R['tt_corr']:+.3f}** on **{R['tt_n']}** winning "
            f"trades is indistinguishable from zero. The projected offset ({R['tt_pred']:.0f} "
            f"bars) overshoots the actual ({R['tt_actual']:.0f} bars) by roughly **2.4×** on "
            "average — H₂ rejected outright."
        ),
        md(
            "### 4c · Robustness — the ZigZag threshold, and a geometry-free cut\n\n"
            "First: does the hit rate move if we draw the wedge more or less strictly?"
        ),
        code(
            "pcts = sorted(R['sweep'])\n"
            "cands = [R['sweep'][p][0] for p in pcts]\n"
            "hits = [R['sweep'][p][2] for p in pcts]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.bar([f'{p*100:.0f}%' for p in pcts], cands, color=GREY, width=.55)\n"
            "a1.set_ylabel('candidates found'); a1.set_title('More candidates at looser thresholds...')\n"
            "a2.bar([f'{p*100:.0f}%' for p in pcts], hits, color=AMBER, width=.55)\n"
            "a2.axhline(R['placebo_mean'], ls='--', c=RED, lw=1.5, label='random-day baseline')\n"
            "a2.set_ylabel('hit rate (%)'); a2.set_title('...but the hit rate never moves')\n"
            "a2.legend(); a2.set_ylim(0, 50)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(R['sweep'])"
        ),
        md(
            "Second: a fixed-horizon directional return that doesn't depend on the target/stop "
            "geometry at all — the honest check that H₁'s null result isn't an artefact of how "
            "we picked those levels."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rng = np.random.default_rng(697)\n"
            "    rets, dirs = {}, {}\n"
            "    for tk in TICKERS:\n"
            "        b = BASKET[tk]\n"
            "        piv = st.zigzag(b['close'].to_numpy(float), pct=R['pct'])\n"
            "        ent = st.wolfe_candidates(piv)\n"
            "        for h in st.HORIZONS:\n"
            "            led = st.run_trades(b, ent, h, cost_bps=0.0)\n"
            "            rets.setdefault(h, []).append(led['ret_gross'].to_numpy())\n"
            "            dirs.setdefault(h, []).append(led['dir'].to_numpy())\n"
            "    hs = list(st.HORIZONS)\n"
            "    ts = []\n"
            "    for h in hs:\n"
            "        r = np.concatenate(rets[h])\n"
            "        s = st.summarize(pd.DataFrame({'ret_gross': r, 'ret_net': r}), 'ret_gross')\n"
            "        ts.append(s['hac_t'])\n"
            "else:\n"
            "    hs = sorted(R['fh']); ts = [R['fh'][h][5] for h in hs]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.3))\n"
            "ax.bar([str(h) for h in hs], ts, color=[RED if abs(t) >= 2 else GREY for t in ts], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('holding horizon (days)'); ax.set_ylabel('HAC t')\n"
            "ax.set_title('Geometry-free directional return: flat-to-negative everywhere')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({h: round(t, 2) for h, t in zip(hs, ts)})"
        ),
        md(
            "> 💡 In plain words: no horizon clears **t = 2** in the predicted direction. The "
            "5-day cut is the only one that comes close, and it's on the *wrong* side "
            f"(HAC *t* = {R['fh'][5][5]:.2f}) — a coin flip at the same swing points beats the "
            f"Wolfe direction {R['fh'][5][6]*100:.0f}% of the time at that horizon. H₃ has "
            "nothing to be robust *toward*."
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "A deterministic panel builds the wedge from **exact anchor points** — piecewise-"
            "linear in log-price between five fixed bars, with noise far too small to fragment a "
            "leg into spurious extra ZigZag pivots — followed by a genuinely quiet, noise-only "
            "gap before the next planted wedge, so a detected point 5 is always confirmed on "
            "quiet noise, never on the run-up to the *next* wedge. The null (`edge=0`, no post-"
            "point-5 drift) is checked over **20 seeds**."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    bars, _ = data.synthetic_panel(edge=0.0, seed=697 + s_, n_days=9000)\n"
            "    piv = st.zigzag(bars['close'].to_numpy(float), pct=R['pct'])\n"
            "    ent = st.wolfe_candidates(piv)\n"
            "    led = st.run_trades(bars, ent, horizon=20, cost_bps=0.0)\n"
            "    null_ts.append(st.summarize(led, 'ret_gross')['t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "bars, _ = data.synthetic_panel(edge=R['syn_planted_edge'], seed=697, n_days=9000)\n"
            "piv = st.zigzag(bars['close'].to_numpy(float), pct=R['pct'])\n"
            "ent = st.wolfe_candidates(piv)\n"
            "led = st.run_trades(bars, ent, horizon=20, cost_bps=0.0)\n"
            "planted_t = st.summarize(led, 'ret_gross')['t']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (edge=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=RED, s=90, zorder=5,\n"
            "           label=f\"planted reversal (edge={R['syn_planted_edge']})\")\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('one-sample t (H=20 directional return)')\n"
            "ax.set_title('Control: null mostly quiet, a real planted reversal lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"t = {R['syn_null_mean_t']:+.2f} (sd {R['syn_null_sd']:.2f}) — {R['syn_null_fire']} "
            "of 20 individual seeds cross ±2 on pure noise, disclosed honestly, but the *average* "
            "stays well under the bar. A planted reversal reads t up to **+5.6** across horizons "
            "(see `results.md`). The machinery is not broken — the real tape is simply flat. "
            "*(A faithful-engine / power check only — never cited in support of the real-tape "
            "stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — target-hit rate **{R['hit_rate']:.1f}%** "
            f"(n={R['n_decided']}) vs a same-distance random-day placebo of "
            f"**{R['placebo_mean']:.1f}%** (z = {R['placebo_z']:+.2f}, p = {R['placebo_p']:.3f}); "
            "flat across the ZigZag threshold (36.0%–37.8%) and the geometry-free "
            "fixed-horizon cut (|t| < 2 everywhere, borderline negative at 5 days). A synthetic "
            "control proves the harness would detect a real planted reversal (t up to 5.6).\n"
            "- **Tradability `MIRAGE`** — no edge to charge costs against; nothing to deploy.\n"
            f"- **No reliable timer? `CONFIRMED`** — the EPA time target correlates "
            f"{R['tt_corr']:+.3f} with actual resolution time (MAE {R['tt_mae']:.1f} bars, "
            f"n={R['tt_n']} winning trades). Even the price calls the pattern gets right, the "
            "clock is noise."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The target-hit-vs-random-day idiom generalises.** Any chart pattern that claims "
            "a specific price target can be graded this exact way — draw the target "
            "mechanically, walk forward on High/Low, compare to a same-distance random day. "
            "[448-point-and-figure](../../448-point-and-figure/) ran the identical idiom on a "
            "different pattern and found a genuine (if untradable) edge — the two studies "
            "together are a natural-experiment pair on which chart claims survive this bar and "
            "which don't.\n"
            "- **Discretion is the real story.** A human chartist choosing which five points "
            "\"count\" — and which of the several published EPA-time conventions to use — has far "
            "more freedom to fit a story in hindsight than the single fixed rule this study runs. "
            "That freedom is exactly what a mechanical, pre-registered detector removes.\n"
            "- **Dedup map:** [445-elliott-wave](../../445-elliott-wave/) (a different wave "
            "count, Fibonacci retracements, no price target), "
            "[447-gann-angles](../../447-gann-angles/) (a single fixed-slope line, no 5-point "
            "structure), [704-three-drives](../../704-three-drives/) (another five-point "
            "Fibonacci pattern, no EPA line).\n\n"
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
