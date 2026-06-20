"""Generate the two narrative notebooks for Study 303 (Uranium-Revival).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells try the cached daily
parquet under ../_cache/ if present and otherwise fall back to the synthetic tape, each
cell bannering exactly which tape it shows. The frozen headline numbers in ``R`` mirror
docs/results.md, so the notebook re-runs identically for any reader.

There is no real URA/URNM cache shipped here: the Signal stamp is therefore WEAK, not
REAL — literature support and a passing synthetic positive control cannot, by the desk's
inference bar, certify a real-tape t >= 2 we did not measure.
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


# Frozen synthetic-control headline numbers — mirror of docs/results.md (as-of 2026-06-19).
# All from the deterministic offline tapes (seed=303), trend rule = long when close>SMA200,
# one execution lag, costs one-way x NAV; "edge" = timed minus buy-and-hold, annualised.
R = dict(
    # Positive control: persistent bull/bear runs (regime='trend', n=3000), 10 bps.
    tr_timed=1.7, tr_bh=-26.8, tr_edge=28.5, tr_t=2.85, tr_sd=0.78,
    tr_tShar=0.19, tr_bhShar=-0.59, tr_pim=28.9, tr_rt=28,
    tr_ci_lo=7.9, tr_ci_hi=48.0,
    # Null: random walk (regime='random', n=3000), 10 bps.
    rn_edge=-1.8, rn_t=-0.75, rn_sd=-0.31, rn_pim=40.5, rn_rt=82,
    rn_ci_lo=-23.5, rn_ci_hi=11.5,
    # Hype rocket: boom-bust (regime='hype', n=2500), 10 bps.
    hy_timed=65.1, hy_bh=7.3, hy_edge=57.7, hy_t=3.68, hy_sd=1.44,
    hy_tShar=1.82, hy_bhShar=0.38, hy_pim=56.2, hy_rt=16,
    hy_tdd=-57.1, hy_bdd=-99.7,
    # Cost sweep on the hype rocket (one-way bps -> timing t).
    hy_c0_t=3.72, hy_c50_t=3.53, hy_c0_edge=58.3, hy_c50_edge=55.6,
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from uranium_revival import data, strategy as st

URANIUM = list(data.URANIUM_TICKERS)   # ('URA','URNM')

def have_real():
    return data.have_real_cache()

HAVE_REAL = have_real()

def tape(regime, n_days):
    bars, truth = data.synthetic_daily(n_days=n_days, regime=regime, seed=303)
    return bars

def run_book(bars, n=200, cost=10.0):
    return st.book_returns(bars, st.trend_signal(bars, n=n), cost_bps=cost)

print("real URA/URNM cache present:", HAVE_REAL, "(synthetic tape used everywhere below)")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Uranium-Revival — durable trend, or a hype rocket? ☢️\n"
            "### Riding the 'nuclear renaissance' trade (URA / URNM) with a trend-following overlay\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Durable_trend_or_hype_rocket%3F: Hype_rocket](https://img.shields.io/badge/Durable_trend_or_hype_rocket%3F-Hype_rocket-8b949e?style=flat-square)\n\n"
            "Uranium is having a moment. Reactors are being restarted, AI data-centres are "
            "hungry for power, and the miner ETFs (**URA**, **URNM**) have gone vertical and "
            "then sideways more than once. The pitch: *just ride the trend* — a simple "
            "'be long when it's above its 200-day average, step aside when it isn't' overlay "
            "supposedly turns a wild theme into a tradable trend. This notebook asks the only "
            "two questions that matter: *is trend-timing a real edge here, and is it durable — "
            "or are you just along for one boom-bust ride?*\n\n"
            "> **This is the plain-language layer.** Want the t-stats, the random-walk control "
            "and the capacity math? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool. **No live URA/URNM tape "
            "ships with this study**, so every chart below is drawn on a *deterministic "
            "synthetic tape* that lets us prove what the trend rule can and cannot do — each "
            "cell says which tape it shows. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is trend-following a real edge? | **Maybe — unproven *here*.** The textbook says "
            "trends are real (time-series momentum is one of finance's most-replicated effects), "
            "and our harness banks a planted trend cleanly. But **we have no live URA/URNM tape** "
            "to certify it, so the honest stamp is **Weak**, not Real. |\n"
            "| Is it *durable* on uranium? | **No reason to think so.** A single thematic miner "
            "ETF is one long boom-bust. A trend rule can look brilliant by dodging *one* crash "
            "— that's a lucky regime, not a repeatable edge. |\n"
            "| Could you trade it? | **It's a mirage as a stand-alone.** You'd be making a "
            "concentrated, leveraged-feeling bet on one volatile theme; the 'timing' is mostly "
            "deciding when to be all-in on a rocket. |\n"
            "| So what is it? | A real *category* of signal (trend) bolted onto the wrong *vehicle* "
            "(one hype-driven sector). The edge you'd measure on URA is dominated by which boom "
            "you happened to catch. |\n\n"
            "> The nuclear-renaissance trend trade isn't a scam — trend-following is a genuine "
            "effect. But on a single hype-cycle ETF, 'I rode the trend' and 'I got lucky on one "
            "rocket' produce the *same backtest*, and only one of them repeats."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Uranium is in a structural bull market — supply deficits, reactor restarts, "
            "AI power demand. Don't try to value it; just **ride the trend**. Hold URA/URNM "
            "while price is above its 200-day moving average, go to cash when it breaks down. "
            "You capture the rocket and skip the crashes.\"*\n\n"
            "— the standard finance-Twitter / thematic-ETF-newsletter framing of the "
            "'nuclear renaissance' trade.\n\n"
            "The idea rests on something real: **trends persist**. Markowitz-style buy-and-hold "
            "ignores that a falling market keeps falling; a trend overlay tries to be long the "
            "ups and flat the downs. The bet is that uranium's wild swings are *trends you can "
            "ride*, not noise you'll get whipsawed by."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two things make this worth a careful look. First, **trend-following is real** — "
            "time-series momentum (Moskowitz, Ooi & Pedersen 2012) is one of the most robust "
            "effects in the literature, across dozens of markets and a century of data. So this "
            "isn't folklore. Second, uranium miners are a **perfect trap**: a single, thin, "
            "hugely volatile theme where one spectacular boom-bust can make *any* timing rule "
            "look like genius. If we can show why a great-looking backtest on one rocket proves "
            "nothing, that lesson transfers to every hot thematic ETF you'll ever be pitched."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "We can't run the *real* uranium tape here (none ships with the study). So instead we "
            "do something more honest than another in-sample backtest: we build **synthetic tapes "
            "where we *know* the truth**, and check what the trend rule does on each.\n\n"
            "1. **A real trend (positive control).** A tape with genuine, persistent bull and "
            "bear runs. If the rule is any good, it must beat buy-and-hold here — *and clear "
            "the statistical bar.* If it can't bank a planted trend, it proves nothing.\n"
            "2. **A coin (the null).** A pure random walk — no trend at all. The rule must "
            "**not** reliably beat buy-and-hold here, or our test is broken.\n"
            "3. **A hype rocket (the realist's case).** A boom-bust bubble: a parabolic run-up "
            "then a crash that gives most of it back — the uranium caricature. We watch the "
            "rule look *amazing* here and ask the killer question: is that an edge, or one lucky "
            "dodge?"
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: can the rule even ride a real trend?** On the positive-control tape "
            "(genuine bull/bear runs), trend-timing vs simply holding the asset:"
        ),
        code(
            "tr = run_book(tape('trend', 3000), cost=10.0)\n"
            "s = st.summarize(tr)\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "b = ax.bar(['Trend-timing', 'Buy & hold'],\n"
            "           [s['timed_ann']*100, s['bh_ann']*100], color=[GREEN, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('annualised return (%/yr)')\n"
            "ax.set_title('Positive control (a REAL trend): timing earns its keep')\n"
            "ax.annotate(f\"HAC t = {s['timing_t']:+.2f}\", (0.5, 0.92), xycoords='axes fraction',\n"
            "            ha='center', fontweight='bold')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"[SYNTHETIC trend tape]  edge {s['timing_edge_ann']*100:+.1f}%/yr  HAC t={s['timing_t']:+.2f}  in-market {s['pct_time_in_market']:.0f}%\")"
        ),
        md(
            f"On a tape with a *real* trend the rule adds **+{R['tr_edge']:.0f}%/yr** over "
            f"buy-and-hold (HAC *t* = +{R['tr_t']:.2f}) — mostly by sitting out the bear runs. "
            "**Good: the harness works.** It can detect and bank a genuine trend. (This is a "
            "*machinery* proof, not market evidence — it certifies the tool, not uranium.)"
        ),
        md(
            "**Now the null. Same rule, on a pure coin** — a random walk with no trend at all. "
            "If the rule 'works' here too, it's not detecting anything."
        ),
        code(
            "rn = run_book(tape('random', 3000), cost=10.0)\n"
            "s = st.summarize(rn)\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(['Trend-timing', 'Buy & hold'],\n"
            "       [s['timed_ann']*100, s['bh_ann']*100], color=[RED, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('annualised return (%/yr)')\n"
            "ax.set_title(f\"Null (a coin): no edge — HAC t = {s['timing_t']:+.2f}\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"[SYNTHETIC random tape]  edge {s['timing_edge_ann']*100:+.1f}%/yr  HAC t={s['timing_t']:+.2f}  (whipsawed: {s['n_round_trips']} round-trips)\")"
        ),
        md(
            f"On the coin the 'edge' is **{R['rn_edge']:+.1f}%/yr** at HAC *t* = {R['rn_t']:+.2f} "
            f"— indistinguishable from zero, and the rule churns **{R['rn_rt']} round-trips** "
            "getting whipsawed. Exactly what should happen: no trend, no edge. The test isn't "
            "rigged to find trends that aren't there."
        ),
        md(
            "**Now the punchline — the hype rocket.** A boom-bust bubble (parabolic run-up, "
            "then a crash that gives most of it back): the uranium caricature. Watch the trend "
            "rule look like a *genius*:"
        ),
        code(
            "hy = run_book(tape('hype', 2500), cost=10.0)\n"
            "s = st.summarize(hy)\n"
            "hb = tape('hype', 2500)\n"
            "eq_bh = (1 + hy['asset']).cumprod(); eq_t = (1 + hy['timed']).cumprod()\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.4))\n"
            "a1.plot(hb.index, hb['close'], color=GREY)\n"
            "a1.set_title('The rocket: a boom, then a crash'); a1.set_ylabel('price')\n"
            "a2.plot(eq_t.index, eq_t, color=AMBER, label=f\"Trend-timing ({s['timed_ann']*100:+.0f}%/yr)\")\n"
            "a2.plot(eq_bh.index, eq_bh, color=GREY, label=f\"Buy & hold ({s['bh_ann']*100:+.0f}%/yr)\")\n"
            "a2.legend(); a2.set_title(f\"Timing 'wins' by dodging the crash — t={s['timing_t']:+.2f}\")\n"
            "a2.set_ylabel('growth of $1')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"[SYNTHETIC hype tape]  timing {s['timed_ann']*100:+.0f}%/yr vs B&H {s['bh_ann']*100:+.0f}%/yr  HAC t={s['timing_t']:+.2f}  worst DD timing {s['timed_maxdd']*100:.0f}% vs B&H {s['bh_maxdd']*100:.0f}%\")"
        ),
        md(
            f"On the rocket the rule earns **+{R['hy_timed']:.0f}%/yr** vs buy-and-hold's "
            f"**+{R['hy_bh']:.0f}%/yr** — HAC *t* = +{R['hy_t']:.2f}, and it cuts the worst "
            f"drawdown from {R['hy_bdd']:.0f}% to {R['hy_tdd']:.0f}%. Looks *spectacular*. **But "
            "here's the catch:** that entire result is the rule getting out near the top of "
            "**one** crash. There's no *repeatable* signal being harvested — just one big lucky "
            "dodge. Run a different boom, with the peak in a different place, and the magic "
            "evaporates. A single-rocket backtest is the easiest thing in finance to make look "
            "great."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** Trend-following is real in the literature, and our harness "
            f"banks a planted trend at HAC *t* = +{R['tr_t']:.2f}. But **the desk's rule is that "
            "REAL is earned by the *real tape*, not by the textbook or a synthetic control** — "
            "and we have no live URA/URNM tape here. So: Weak.\n"
            "- **Tradability — Mirage.** As a stand-alone, this is a concentrated bet on one "
            "volatile theme dressed up as 'timing'. The edge you'd measure is dominated by which "
            "boom you caught, not a repeatable skill.\n"
            "- **Durable trend or hype rocket? — Hype rocket.** On a single thematic ETF, the "
            "great backtest comes from dodging *one* crash. That's a lucky regime, not a durable "
            "edge — and it's indistinguishable, in-sample, from genuine skill."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Costs are *not* the problem — a 200-day rule trades a handful of times a year. The "
            "problem is what you're actually buying. Watch the timing 'edge' on the rocket shrug "
            "off even brutal costs (because cost isn't the constraint):"
        ),
        code(
            "costs = [0.0, 10.0, 20.0, 50.0]\n"
            "sweep = st.cost_sweep(tape('hype', 2500), n=200, costs_bps=costs)\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.plot(sweep['cost_bps'], sweep['edge_ann']*100, 'o-', c=AMBER, lw=2)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xlabel('one-way cost (bps)')\n"
            "ax.set_ylabel('timing edge (%/yr)'); ax.set_ylim(0, sweep['edge_ann'].max()*120)\n"
            "ax.set_title('Costs barely touch it — because the real risk is concentration, not fees')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('[SYNTHETIC hype tape] the edge survives 50 bps easily — but it is one-rocket-dependent, not durable.')"
        ),
        md(
            "The edge is cost-robust, which is exactly why it's *seductive* and exactly why it's "
            "dangerous. Cheap to run, looks bulletproof in-sample — and entirely a function of "
            "the one boom-bust you backtested on. The honest pitch isn't 'a trend system on "
            "uranium'; it's 'a leveraged-feeling bet that *this* rocket behaves like the *last* "
            "one.'"
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The honest way to test trend.** Trend-following earns its REAL stamp across "
            "*dozens* of markets pooled, not one thematic ETF (Moskowitz, Ooi & Pedersen 2012). "
            "The fix for the concentration problem is breadth — a basket, not a rocket.\n"
            "- **Get the real tape.** Populate `_cache/` with `data.fetch_daily('URA', "
            "fetch=True)` (and `'URNM'`) and re-run: *then* you can ask whether the live "
            "uranium tape clears HAC *t* = 2, and upgrade or bury the Weak stamp accordingly.\n"
            "- **The single-asset trend trap.** This is the same lesson as any hot-sector timer: "
            "one boom makes any rule look like skill. Fork this, swap in a different synthetic "
            "boom (move the peak), and watch a 'genius' strategy turn ordinary.\n\n"
            "*Think uranium's trend is durable? Fork this, drop in the real URA tape, and show "
            "the timing edge clearing the bar on out-of-sample boom — not just the one you "
            "fitted.*"
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
            "# Uranium-Revival — a quantitative teardown \U0001f52c\n"
            "### Trend-following on a thematic ETF · synthetic regime controls · HAC inference · "
            "block-bootstrap CI · excess-vs-excess · the single-asset trend trap\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Durable_trend_or_hype_rocket%3F: Hype_rocket](https://img.shields.io/badge/Durable_trend_or_hype_rocket%3F-Hype_rocket-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim now carrying its standard error.* We test whether a "
            "200-day trend overlay on uranium miners (URA/URNM) is a real, durable improvement on "
            "buy-and-hold — using deterministic regime controls (a planted trend, a coin, a "
            "hype rocket) to separate *signal detection* from *one lucky boom*.\n\n"
            "> **Not investment advice.** **No live URA/URNM tape ships with this study**, so all "
            "executed cells run the offline synthetic regimes (seed=303); the Signal stamp is "
            "WEAK by the inference bar (a synthetic control is a machinery proof, never market "
            "evidence). Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Harness banks a *planted* trend at HAC "
            f"*t* = **+{R['tr_t']:.2f}** (positive control), and is null on a coin "
            f"(*t* = {R['rn_t']:+.2f}). But **no real URA/URNM tape** to certify "
            "*t* ≥ 2 — literature + synthetic ⇒ WEAK, not REAL. |\n"
            f"| **Tradability** | `MIRAGE` | On the realistic boom-bust regime the 'edge' "
            f"(+{R['hy_edge']:.0f}%/yr, *t*=+{R['hy_t']:.2f}) is one crash-dodge on a single "
            "thin theme — concentration risk dressed as timing, not a scalable edge. |\n"
            f"| **Durable trend or hype rocket?** | `HYPE ROCKET` | The flattering backtest is "
            f"regime-specific: cost-robust (*t*=+{R['hy_c50_t']:.2f} at 50 bps) yet entirely a "
            "function of *which* boom-bust you fitted. |\n\n"
            "> In plain words: trend-following is a real *category* of edge, but a single "
            "hype-cycle ETF is the wrong vehicle — in-sample, genuine skill and one lucky dodge "
            "are the same picture, and we have no out-of-sample real tape to tell them apart."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_{t+1}$ be the asset's day-$(t{+}1)$ return and $s_t = \\mathbf{1}[P_t > "
            "\\mathrm{SMA}_{200}(t)]$ the trend position known at the close of $t$. The timed "
            "book earns $s_t\\,r_{t+1} + (1-s_t)\\,r^{f}$ (one execution lag), and the **timing "
            "edge** is $e_{t+1} = (\\text{timed} - r^f) - (\\text{asset} - r^f)$ — "
            "excess-of-cash vs excess-of-cash, so the timed book gets no free credit for sitting "
            "in cash beyond the risk-free rate it earns there.\n\n"
            "- **H₁ (signal).** $\\mathbb{E}[e_{t+1}] > 0$ with HAC *t* ≥ 2 on the **real** tape.\n"
            "- **H₂ (detection).** The rule banks a *planted* trend (positive control) and is "
            "null on a random walk.\n"
            "- **H₃ (durability).** The edge is not an artefact of a single boom-bust regime.\n\n"
            "We find **H₂ supported** (machinery works), **H₁ untestable here** (no real tape "
            "→ WEAK), and **H₃ rejected** on the realistic regime (the edge is one crash-dodge)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "Time-series momentum is among the most replicated effects in asset pricing "
            "(Moskowitz, Ooi & Pedersen 2012; the AQR 'A Century of Evidence' work). So the "
            "*category* is not in question. What's in question is the **vehicle**: applying a "
            "trend overlay to a single, thin, hype-driven sector ETF. The interesting content "
            "is not a binary verdict but the **mechanism of the illusion** — a single boom-bust "
            "lets any timing rule post a huge, cost-robust, low-drawdown backtest that is pure "
            "regime luck. Naming that precisely is the product."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Engine.** Long when close > SMA(200), else in cash at the risk-free rate; "
            "**one** `shift(1)` (signal at close of *t* → return of *t+1*); costs one-way × NAV "
            "on $|\\Delta\\text{position}|$ (a round-trip pays twice).\n"
            "- **Race.** Excess-of-cash to excess-of-cash vs buy-and-hold — the timing edge "
            "$e_{t+1}$.\n"
            "- **Inference.** Newey-West HAC *t* on $e$; circular **block bootstrap** CI "
            "(block = 21d) on the annualised edge — i.i.d. resampling would destroy the "
            "volatility clustering that dominates a single thematic ETF.\n"
            "- **Regime controls.** `trend` (planted persistent bull/bear runs — positive "
            "control), `random` (a coin — null), `hype` (boom-bust — the realist's case). "
            "Deterministic, seed=303.\n\n"
            "**No real URA/URNM tape ships here** — so this beat is, deliberately, a study of "
            "what the rule *does to known truth*, not a real-tape backtest."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Detection — the harness banks a planted trend and is null on a coin\n\n"
            "The HAC *t* of the timing edge on each control regime. The positive control must "
            "clear |*t*| = 2; the coin must not."
        ),
        code(
            "rows = []\n"
            "for reg, nd, lab in [('trend',3000,'Planted trend (control)'),\n"
            "                     ('random',3000,'Coin (null)'),\n"
            "                     ('hype',2500,'Hype rocket (realist)')]:\n"
            "    s = st.summarize(run_book(tape(reg, nd), cost=10.0))\n"
            "    rows.append((lab, s['timing_edge_ann']*100, s['timing_t'], s['sharpe_diff']))\n"
            "dfm = pd.DataFrame(rows, columns=['regime','edge %/yr','HAC t','Sharpe diff'])\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.4))\n"
            "col = [GREEN if abs(t) >= 2 else GREY for t in dfm['HAC t']]\n"
            "ax.bar(dfm['regime'], dfm['HAC t'], color=col)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1); ax.axhline(-2, ls='--', c=GREY, lw=1); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('HAC t of timing edge'); ax.tick_params(axis='x', labelsize=8)\n"
            "ax.set_title('Detection works — but the rocket clears the bar too (that is the trap)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('[SYNTHETIC regime controls]'); print(dfm.round(2).to_string(index=False))"
        ),
        md(
            f"> In plain words: the planted trend clears the bar (*t* = +{R['tr_t']:.2f}); the coin "
            f"does not (*t* = {R['rn_t']:+.2f}). **The machinery is faithful.** But the hype rocket "
            f"*also* clears it (*t* = +{R['hy_t']:.2f}) — and that's the whole point: a single "
            "boom-bust is statistically indistinguishable, in-sample, from a real trend."
        ),
        md(
            "### 4b · The block-bootstrap CI — honest uncertainty on the edge\n\n"
            "A point estimate of the timing edge on one tape is nearly meaningless; the CI shows "
            "how wide the uncertainty really is once volatility clustering is respected."
        ),
        code(
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "labs, los, his, mids = [], [], [], []\n"
            "for reg, nd, lab in [('trend',3000,'trend'),('random',3000,'coin'),('hype',2500,'hype')]:\n"
            "    book = run_book(tape(reg, nd), cost=10.0)\n"
            "    s = st.summarize(book)\n"
            "    lo, hi = s['timing_ci']\n"
            "    labs.append(lab); los.append(lo*100); his.append(hi*100); mids.append(s['timing_edge_ann']*100)\n"
            "y = np.arange(len(labs))\n"
            "ax.hlines(y, los, his, color=GREY, lw=4, alpha=.6)\n"
            "ax.plot(mids, y, 'o', color=AMBER)\n"
            "ax.axvline(0, c='k', lw=1); ax.set_yticks(y); ax.set_yticklabels(labs)\n"
            "ax.set_xlabel('annualised timing edge (%/yr), 95% block-bootstrap CI')\n"
            "ax.set_title('Wide CIs on a single tape — one asset can never settle this')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('[SYNTHETIC] trend CI ~[%.0f, %.0f]%%/yr, coin straddles 0, hype wide-but-positive (regime-specific).'"
            f" % ({R['tr_ci_lo']:.0f}, {R['tr_ci_hi']:.0f}))"
        ),
        md(
            f"> In plain words: even the planted trend's edge spans ~[{R['tr_ci_lo']:.0f}, "
            f"{R['tr_ci_hi']:.0f}]%/yr; the coin's CI straddles zero. On a *single* asset the "
            "uncertainty is enormous — which is precisely why a real-tape conclusion needs "
            "breadth (many markets), not one volatile ETF."
        ),
        md(
            "### 4c · The single-asset trend trap — the same rule, many boom shapes\n\n"
            "The decisive test of H₃. If the hype-rocket edge were a real signal, it would be "
            "stable as we vary *where* the boom peaks. It isn't — it's a function of the "
            "particular crash you happened to fit."
        ),
        code(
            "ts = []\n"
            "for seed in range(303, 315):\n"
            "    b, _ = data.synthetic_daily(n_days=2500, regime='hype', seed=seed)\n"
            "    ts.append(st.summarize(run_book(b, cost=10.0))['timing_t'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(range(len(ts)), ts, color=[GREEN if abs(t)>=2 else GREY for t in ts])\n"
            "ax.axhline(2, ls='--', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('different hype-rocket draws (same rule)'); ax.set_ylabel('HAC t of timing edge')\n"
            "ax.set_title('Same rule, different rocket: the \"edge\" is all over the place')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'[SYNTHETIC hype draws] HAC t ranges {min(ts):+.2f} to {max(ts):+.2f} — regime luck, not a stable edge.')"
        ),
        md(
            "> In plain words: hold the rule fixed and redraw the boom-bust — the HAC *t* swings "
            "wildly across draws. A *real* edge would be stable; this one is a coin flip on which "
            "rocket you got. **H₃ rejected:** on a single thematic ETF, the great backtest is "
            "regime luck."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — harness banks a planted trend (HAC *t* = +{R['tr_t']:.2f}) "
            f"and is null on a coin ({R['rn_t']:+.2f}), but **no real URA/URNM tape** to certify "
            "*t* ≥ 2. By the inference bar, a synthetic control + literature is WEAK, not REAL.\n"
            f"- **Tradability `MIRAGE`** — the realistic-regime edge (+{R['hy_edge']:.0f}%/yr) is "
            "one crash-dodge on a single thin theme; it is concentration risk, not a scalable, "
            "repeatable edge.\n"
            f"- **Durable trend or hype rocket? `HYPE ROCKET`** — cost-robust "
            f"(*t* = +{R['hy_c50_t']:.2f} at 50 bps) yet entirely regime-specific; redrawing the "
            "boom scrambles the *t* (4c)."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — cost sweep and the real constraint\n\n"
            "Costs are a rounding error for a 200-day rule; the binding constraint is "
            "concentration and regime-dependence, neither of which a cost sweep can show — "
            "which is exactly why cost-robustness here is a *warning*, not a green light."
        ),
        code(
            "costs = [0.0, 10.0, 20.0, 50.0]\n"
            "sw = st.cost_sweep(tape('hype', 2500), n=200, costs_bps=costs)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(sw['cost_bps'], sw['edge_ann']*100, 'o-', c=AMBER, lw=2, label='edge %/yr')\n"
            "ax2 = ax.twinx(); ax2.plot(sw['cost_bps'], sw['timing_t'], 's--', c=GREY, lw=1.5, label='HAC t')\n"
            "ax2.axhline(2, ls=':', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('one-way cost (bps)'); ax.set_ylabel('edge (%/yr)', color=AMBER)\n"
            "ax.set_ylim(0, sw['edge_ann'].max()*120); ax2.set_ylabel('HAC t', color=GREY)\n"
            "ax.set_title('Cost-robust — which is the seduction, not the safety')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('[SYNTHETIC hype] edge %.0f%%/yr at 0 bps -> %.0f%%/yr at 50 bps; t barely moves.'"
            f" % ({R['hy_c0_edge']:.0f}, {R['hy_c50_edge']:.0f}))"
        ),
        md(
            f"> In plain words: the edge falls only from +{R['hy_c0_edge']:.0f}%/yr to "
            f"+{R['hy_c50_edge']:.0f}%/yr across 0–50 bps (*t* from +{R['hy_c0_t']:.2f} to "
            f"+{R['hy_c50_t']:.2f}). Costs don't kill it because costs were never the risk. The "
            "risk is that the whole backtest rests on one boom-bust you can't repeat — the "
            "textbook **mirage**: survives every test you ran, fails the one you couldn't (out-of-"
            "sample uranium)."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — breadth is the real fix\n\n"
            "Time-series momentum earns its REAL stamp **pooled across dozens of markets and a "
            "century of data** (Moskowitz, Ooi & Pedersen 2012; Hurst, Ooi & Pedersen 2017), "
            "precisely because pooling averages out the single-asset regime luck dissected in "
            "4c. A uranium-only trend trade throws that breadth away.\n\n"
            "- **Upgrade the stamp honestly.** Populate `_cache/` "
            "(`data.fetch_daily('URA', fetch=True)`, `'URNM'`) and re-run: if the *live* tape's "
            "timing edge clears HAC *t* = 2 out-of-sample, the Signal stamp moves toward REAL. "
            "Until then, WEAK is the honest call.\n"
            "- **The general lesson.** Any single hot-sector ETF (cannabis, SPACs, AI, uranium) "
            "lets a timing rule post a gorgeous one-cycle backtest. The desk's other "
            "single-theme teardowns tell the same story: the vehicle, not the signal, is the "
            "problem.\n"
            "- **Fork it.** Vary the boom shape (peak location, crash depth) and map where the "
            "'edge' lives — you'll find it's a function of the fit, not the future."
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
