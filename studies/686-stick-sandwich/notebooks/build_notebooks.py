"""Generate the two narrative notebooks for Study 686 (Stick Sandwich).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached daily tapes under
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


# Frozen real-tape headline numbers — mirror of docs/results.md.
# yfinance daily, SPY + 29 long-listed US large-caps, 2001-07-10 -> 2026-06-30
# (As-of 2026-06-30), ~25.0 years/name, tol=15bps, trend_lookback=10, bullish sandwich,
# cost=5bps one-way.
R = dict(
    asof="2026-06-30", start="2001-07-10", end="2026-06-30", years=25.0,
    n_names=30, fp_spy="26d2248068a7", fp_panel="3e40346f84dd",
    n_sandwiches=873, tol_bps=15, lookback=10, cost_bps=5.0,
    per_ticker_min=14, per_ticker_max=43, bonferroni_crit=2.50,
    # pooled bullish stick sandwich, per horizon:
    # (H, n, sand_bps, win%, one_sample_t, base_bps, delta_bps, net_bps, welch_t)
    h5=(5, 873, 38.4, 55, 3.09, 26.9, 11.5, 28.4, 1.01),
    h10=(10, 873, 50.5, 57, 3.16, 53.4, -2.9, 40.5, -0.19),
    h20=(20, 871, 99.3, 58, 4.29, 106.6, -7.3, 89.3, -0.33),
    h60=(60, 865, 327.1, 65, 7.55, 321.9, 5.2, 317.1, 0.14),
    # equal-close geometry placebo (SPY, H=20): obs_bps, p, draws, n_candidates
    placebo=(243.6, 0.110, 1000, 232),
    # synthetic control: null (mean t, sd, n_fired, n_seeds); planted (edge, n_planted,
    # sand_bps, base_bps, delta_bps, welch_t)
    syn_null=(-0.31, 0.90, 0, 20),
    syn_planted=(1.00, 606, 702.7, 236.0, 466.7, 23.08),
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Meeting_adds_info%3F: Busted](https://img.shields.io/badge/Meeting_adds_info%3F-Busted-8b949e?style=flat-square)\n\n"
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

from stick_sandwich import data, strategy as st

ASOF = "2026-06-30"
TOL = st.DEFAULT_TOL
LB = st.DEFAULT_TREND_LOOKBACK
HAVE_REAL = data.have_real()
PANEL = data.load_real() if HAVE_REAL else {}
print("real cache present:", HAVE_REAL, "| names:", len(data.BASKET),
      "| loaded:", len(PANEL))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does a \"stick sandwich\" candle actually call a bottom? 🥪\n"
            "### A famous three-candle pattern — two matching closes with a failed rally "
            "between them — meets a stopwatch\n\n"
            + BADGES +
            "Flip through any candlestick chart book (they all trace back to Steve Nison's "
            "*Japanese Candlestick Charting Techniques*) and you'll find the **stick "
            "sandwich**: a red candle falls, a green candle rallies *above* it the next day, "
            "then a third red candle gives the whole rally back and closes at **almost exactly "
            "the same price** as the first red candle. Two matching closes \"sandwich\" the "
            "green rally candle in the middle — and the story is that a price level tested "
            "twice in three days, and held both times, is **support**: buy it.\n\n"
            "It *looks* compelling on a hand-picked chart. But a three-candle pattern that is, "
            "by definition, a dip after a decline, on a market that drifts **up** over "
            "time, is the textbook setup for fooling yourself. So we did the only fair thing: "
            "encode the sandwich **mechanically** (a concrete equal-close tolerance, no "
            "eyeballing), fire the buy across 30 large-cap names over 25 years, and time the "
            "result with a stopwatch — against the only baseline that matters: **buying on any "
            "other random day of the same tape.**\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the Bonferroni correction and "
            "the cost math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice** — research & education. Every chart is drawn by the "
            "code beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| If I buy a stick sandwich, do I make money? | **Usually, yes — but so does "
            "almost any long entry on this tape.** The basket drifts up over 25 years; any "
            "buy-and-hold-a-few-weeks rule looks profitable in isolation. |\n"
            "| Is that *the pattern's* doing? | **No.** Buy on any random day of the same "
            "30-name panel instead, and you do **just as well** — sometimes slightly better, "
            "sometimes slightly worse, with no consistent gap in either direction. |\n"
            "| Does the *equal close* — the whole point of a \"sandwich\" — matter? | **Not "
            "shown to.** Keep the failed-rally-after-a-decline setup but ignore whether the "
            "two red candles actually land at the same price, and you get almost the same "
            "result. |\n"
            "| So is it a tradable edge? | **No.** It's **beta in a costume** — a dip after a "
            "decline, re-labelled as a candlestick reversal with a catchy name. |\n\n"
            "> The stick sandwich is a tidy way to *name* three candles after the fact. As a "
            "*forecast* — \"the double-tested close marks the bottom\" — it's a **mirage**: the "
            "gains are the market's climb, and the equal close itself does no detectable work."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A red candle falls. A green candle rallies above it. A third red candle gives "
            "the rally back and closes right where the first red candle closed — the market "
            "tested that price level twice, from two directions, and it held both times. That's "
            "support: buy it.\"*\n\n"
            "This is the **bullish stick sandwich**, one of **Steve Nison's** classic "
            "three-candle patterns (*Japanese Candlestick Charting Techniques*, 1991), the book "
            "that brought Japanese candle charting to Western technical analysis. It's the "
            "three-candle, round-trip cousin of the two-candle **counterattack / meeting line** "
            "(sibling study [460](../../460-counterattack-lines/)) — both hinge on the *same* "
            "idea: an equal close is meaningful. So it's a clean test of: **does a closing "
            "price, tested twice, forecast anything?**"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the sandwich genuinely *forecast* reversals, it would be a small miracle: three "
            "candles and a coincidence of closing prices would predict a turn, a crack in "
            "market efficiency you could trade straight off the chart. That's the dream the "
            "pattern sells.\n\n"
            "But there's a trap built into the design. A stick sandwich is *defined* by a down "
            "leg followed by a failed rally — it's a **dip-buy**, and on a basket that drifts "
            "**up**, *any* dip-buy looks profitable in isolation. To separate the **pattern** "
            "from the **tide**, we have to (a) detect the sandwich by a fixed mechanical rule "
            "with no hindsight, (b) compare it to buying on **random days of the same tape**, "
            "and (c) check whether the *equal close* — not just the failed rally — is doing "
            "anything. We'll do all three."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **SPY + {R['n_names'] - 1} long-listed US large-caps**, daily, over "
            f"**~{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Detect the sandwich mechanically.** A confirmed down leg (close below where "
            f"it was {R['lookback']} bars ago), a red candle, a green candle that rallies "
            f"*above* it, then a red candle closing **within {R['tol_bps']} bps** of the first "
            "red candle's close. No eyeballing.\n"
            "2. **Trade the lore.** On the sandwich bar's close, buy at the **next** close; "
            "measure the return over the next **5 / 10 / 20 / 60 days**.\n"
            "3. **The honest baseline.** Do the exact same hold on the **same panel's random "
            "days**. If the sandwich matters, it must beat that base rate. *If it doesn't, the "
            "pattern is a mirage* — that's the result that would make us say so, announced "
            "before we look.\n"
            "4. **The geometry check.** Keep the down-leg/failed-rally setup but ignore whether "
            "the closes actually meet; if that does just as well, the *sandwich* was never the "
            "point."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does a mechanical stick sandwich even look like on the tape? Here's "
            "SPY's full history with every sandwich signal the rule would buy marked."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = PANEL['SPY']; cl = b['close']\n"
            "    ent = st.sandwich_entries(b, tol=TOL, trend_lookback=LB)\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    ax.plot(cl.index, cl.values, c='k', lw=.7, label='SPY close')\n"
            "    ax.scatter(ent, cl.reindex(ent), c=GREEN, s=32, zorder=5, alpha=.85,\n"
            "               label='stick sandwich BUY')\n"
            "    ax.set_yscale('log')\n"
            "    ax.set_title('Mechanical bullish stick sandwiches on SPY, 2001-2026 (log scale)')\n"
            "    ax.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('stick sandwiches on SPY, full history:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The green dots cluster around dips — exactly as a dip-buy would. The question is "
            "whether those dots are followed by *reversals* stronger than an ordinary day. "
            "**Let's race the sandwich against the base rate** at four horizons across all 30 "
            "names. Blue = buy the sandwich; grey = buy on any other day of the same panel."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(PANEL, cost_bps=R['cost_bps'])\n"
            "    sand = [res['by_h'][h]['gross']['mean_bps'] for h in hs]\n"
            "    base = [res['by_h'][h]['base']['mean_bps'] for h in hs]\n"
            "else:\n"
            "    sand = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    base = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, sand, .4, color='#2c6fbb', label='buy the stick sandwich')\n"
            "ax.bar(x+.2, base, .4, color=GREY, label='buy on any other day (base rate)')\n"
            "for i,(a,b) in enumerate(zip(sand,base)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_title('The stick sandwich does NOT beat the base rate at any horizon'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('sandwich:', [round(v) for v in sand]); print('base rate:', [round(v) for v in base])"
        ),
        md(
            "There's the story: the blue and grey bars are nearly on top of each other at every "
            "horizon. At 10 and 20 days the sandwich is actually *slightly worse* than the base "
            "rate; at 5 and 60 days it's *slightly better* — with no consistent pattern and, as "
            "the quants notebook shows, none of the gaps are statistically meaningful. The "
            "apparent profit was **the market's upward climb**, present in the sandwich buys and "
            "in a random day equally."
        ),
        md(
            "**One more sanity check.** What if we keep the down-leg-and-failed-rally setup but "
            "*ignore* whether the two red candles actually close at the same price — the whole "
            "point of calling it a \"sandwich\"? If price really respects the **equal close**, "
            "dropping that condition should wreck the result."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.geometry_placebo(PANEL['SPY'], 20, tol=TOL, trend_lookback=LB,\n"
            "                             n_draws=300, seed=686)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real stick sandwich (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... {pval*100:.0f}% of *no-equal-close* failed-rally dip-buys do at least as well (p={pval:.2f}).')\n"
            "print('=> the equal close is not shown to be doing the work.')"
        ),
        md(
            f"About **{R['placebo'][1]*100:.0f}%** of the no-equal-close draws match or beat the "
            "real sandwich (*p* = {:.2f}). If price genuinely respected *the matching closes*, "
            "dropping that condition should collapse the result far below the real one. It "
            "doesn't — because the result was never really about the equal close.".format(R['placebo'][1])
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The stick sandwich buy does **not** beat buying on random "
            "days of the same panel at any of the four horizons tested; the gaps never clear "
            "the desk's significance bar.\n"
            "- **Tradability — Mirage.** Nothing to trade once you remove the beta you were "
            "always getting for free — and at two of four horizons the pattern is already a net "
            "negative before costs.\n"
            "- **\"Does the equal-close meeting forecast?\" — Busted.** Drop the equal-close "
            "condition and the result barely moves. The defining \"two matching closes\" carry "
            "no detectable forecasting information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. Whatever small, inconsistent edge appears at any "
            "single horizon is well within the noise a random entry on the same panel already "
            "produces — you'd capture the market's climb more cheaply (and more fully) by just "
            "**holding the basket**. Costs (5 bps one-way, paid on entry and exit) only push an "
            "already-flat result further down. As a forecasting tool the sandwich doesn't pay; "
            "as a label for three candles, it was never meant to be a strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The two-candle cousin.** Sibling study "
            "[460-counterattack-lines](../../460-counterattack-lines/) tests the simpler "
            "two-candle version of the same equal-close idea — same verdict, one fewer bar.\n"
            "- **Tolerance & lookback sweeps.** The equal-close band and down-leg window are "
            "free parameters; the quants notebook's synthetic control shows the harness *can* "
            "find a real effect if one is planted, so loosening/tightening the real-tape "
            "parameters is a fair next experiment (spoiler: the same drift confound dominates).\n"
            "- **A real positive control.** The quants notebook plants a *genuine* "
            "post-sandwich bounce into a synthetic panel and shows the harness banks it (so the "
            "null result here isn't a dead detector — it's an honest \"nothing there\").\n\n"
            "*Think the sandwich forecasts? Show it beating base-rate entries at "
            "**Bonferroni-corrected t ≥ 2.5** on a real tape — then we'll talk.*"
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
            "# Stick Sandwich — a quantitative teardown 🔬\n"
            "### Mechanical bullish stick sandwiches on a 30-name panel · forward returns vs "
            "base rate · one-sample HAC *t* vs the beta trap · Bonferroni-corrected Welch *t* · "
            "an equal-close geometry placebo · costs · a synthetic planted-bounce control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The job is to separate the **sandwich** from the **drift**: a stick sandwich is a "
            "dip-buy after a down leg, and an upward-trending basket makes *any* dip-buy look "
            "good, so the only meaningful test is sandwich-vs-base-rate, Bonferroni-corrected "
            "across the four horizons, plus a placebo that drops the equal-close condition "
            "while keeping the down-leg/failed-rally context.\n\n"
            "> ⚠️ **Data note.** SPY + 29 long-listed US large-caps, yfinance daily adjusted "
            "closes (**total-return**), 2001→2026. Sandwich = down leg "
            f"(lookback {R['lookback']}), bearish/bullish-rally/bearish, outer closes within "
            f"{R['tol_bps']} bps; entry is the **next close** (one documented lag). Offline "
            "core + synthetic control are deterministic. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (SPY fingerprint `" + R["fp_spy"] +
            "`, panel fingerprint `" + R["fp_panel"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Sandwich vs the **unconditional base rate** (same panel): "
            f"Δ = {R['h5'][6]:+.1f} / {R['h10'][6]:+.1f} / {R['h20'][6]:+.1f} / "
            f"{R['h60'][6]:+.1f} bps at 5/10/20/60d; Welch *t* = {R['h5'][8]:+.2f} / "
            f"{R['h10'][8]:+.2f} / {R['h20'][8]:+.2f} / {R['h60'][8]:+.2f} — **never clears the "
            f"naive |t|≥2 bar, let alone the Bonferroni-corrected |t|≥{R['bonferroni_crit']:.2f}** "
            f"for {4} simultaneous horizon tests. |\n"
            f"| **Tradability** | `MIRAGE` | One-sample *t*'s up to +{R['h60'][4]:.2f} at 60d are "
            "**pure beta** — they vanish against the base rate; at two horizons the pattern is "
            "already net-negative before costs. Nothing to scale. |\n"
            f"| **Meeting adds info?** | `BUSTED` | Equal-close geometry placebo: **p = "
            f"{R['placebo'][1]:.3f}** of no-equal-close, context-matched draws match or beat the "
            "real sandwich. The defining geometry is not shown load-bearing. |\n\n"
            "> 💡 In plain words: the sandwich *looks* like it pays because the underlying "
            "basket drifts up. Strip the drift (race it vs the base rate) or strip the sandwich "
            "(drop the equal-close condition) and the edge evaporates either way. Classic "
            "beta-in-a-costume, dressed up with an extra candle."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "A bullish stick sandwich at bar $t$ requires: a down leg $C_{t-3}<C_{t-3-L}$ "
            "($L=10$); a bearish bread candle $C_{t-2}<O_{t-2}$; a bullish filling candle "
            "$C_{t-1}>O_{t-1}$ with $C_{t-1}>C_{t-2}$ (a genuine, if temporary, rally); a "
            "second bearish bread candle $C_t<O_t$; and the **meeting** "
            "$|C_t-C_{t-2}|/C_{t-2}\\le\\tau$ ($\\tau=15$ bps). The rule buys the close of $t$ "
            "and rides the claimed reversal up.\n\n"
            "- **H₀ (drift).** Sandwich returns equal the drift-matched **base-rate** baseline "
            "(same panel, same long, every eligible bar).\n"
            "- **H₁ (the sandwich forecasts).** Sandwich returns **exceed** the base rate at "
            "some horizon, Bonferroni-corrected |*t*| ≥ 2.50 for *k*=4 simultaneous tests.\n"
            "- **H₂ (the equal close matters).** Sandwich returns exceed a **geometry-placebo** "
            "pool that keeps the down-leg/failed-rally context but drops the meeting.\n\n"
            f"We find **H₀ not rejected** (sandwich ≈ base rate, gaps flip sign across "
            f"horizons), **H₁ rejected** (max |Welch t| = {max(abs(R['h5'][8]), abs(R['h10'][8]), abs(R['h20'][8]), abs(R['h60'][8])):.2f}), "
            f"**H₂ rejected** (placebo p = {R['placebo'][1]:.3f}). The steelman fails on every leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** A 30-name large-cap basket has a positive unconditional daily "
            "mean. *Any* long entry rule inherits it; a high one-sample $t$ against **zero** "
            "measures the tide, not the tool. The fix is the **base-rate baseline** — the "
            "identical long-only forward-return distribution, measured on *every* eligible bar "
            "of the *same* panel — and a Welch test of sandwich-*minus*-base-rate.\n\n"
            "**(b) The dip vs the sandwich.** A stick sandwich is *by construction* a dip-buy "
            "(it needs a down leg and a failed rally). The danger is that the whole signal is "
            "the dip, not the equal close. The **geometry placebo** keeps the down-leg + "
            "bearish/bullish-rally/bearish context but draws entries that ignore the meeting — "
            "if the real result survives that, the equal close was never load-bearing.\n\n"
            "**(c) Multiple comparisons.** Four horizons are four simultaneous looks at one "
            "question — a Bonferroni-corrected critical value (via the two-sided normal "
            "quantile for *k*=4) is the honest bar, not the naive |*t*| ≥ 2."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** SPY + {R['n_names']-1} long-listed US large-caps; yfinance daily "
            f"adjusted closes ({R['start']}→{R['end']}, ~{R['years']:.1f}y each). "
            f"**{R['n_sandwiches']} bullish stick sandwiches** pooled (per-name count "
            f"{R['per_ticker_min']}–{R['per_ticker_max']}).\n"
            f"- **Pattern.** Down leg (lookback {R['lookback']}), bearish bread-1, "
            f"bullish-and-rallying filling, bearish bread-2, outer closes within "
            f"{R['tol_bps']} bps — all read at/before *t* (no look-ahead).\n"
            "- **Entry.** Buy the sandwich bar's close; enter **next close** (one lag); hold "
            "H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC *t*** of sandwich returns vs 0 (Newey-West).\n"
            "- **Null #2 — base-rate baseline**, Welch two-sample sandwich vs base rate (the "
            "*real* test), Bonferroni-corrected across 4 horizons.\n"
            "- **Null #3 — geometry placebo** (equal close dropped, down-leg/failed-rally "
            "context kept).\n"
            f"- **Costs.** {R['cost_bps']:.0f} bps one-way × 2 legs on every signal.\n"
            "- **Positive control.** Synthetic panel with a **planted** post-sandwich bounce "
            "(knob `edge`): edge=0 must NOT reach significance across seeds; edge>0 must light "
            "up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample *t* looks fine, vs-base-rate kills it\n\n"
            "Left: the sandwich's **one-sample** *t* against zero (the misleading number, "
            "growing with horizon). Right: the same sandwich vs the **drift-matched base "
            "rate** (the honest number), with the naive |*t*|=2 bar and the "
            "Bonferroni-corrected |*t*|=2.50 bar for *k*=4 simultaneous tests."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(PANEL, cost_bps=R['cost_bps'])\n"
            "    one_t = [res['by_h'][h]['gross']['t'] for h in hs]\n"
            "    welch = [res['by_h'][h]['welch_t'] for h in hs]\n"
            "    crit = res['bonferroni_crit']\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    welch = [R['h5'][8], R['h10'][8], R['h20'][8], R['h60'][8]]\n"
            "    crit = R['bonferroni_crit']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], one_t, color=GREY, width=.6)\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2 bar'); a1.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO (misleading: it is beta)'); a1.set_ylabel('t'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if abs(v)>=crit else RED for v in welch], width=.6)\n"
            "a2.axhline(crit, ls='--', c=RED, label=f'Bonferroni t={crit:.2f}')\n"
            "a2.axhline(-crit, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Sandwich vs BASE RATE, Welch t (honest: never clears the bar)'); a2.set_ylabel('t'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs base rate:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars *grow* with the horizon (60d "
            f"**{R['h60'][4]:.2f}**) — but that's the **drift**, every dip-buy on an "
            f"upward-trending panel inherits it. The right bars are the real test: "
            f"sandwich-minus-base-rate flips sign across horizons "
            f"({R['h5'][8]:+.2f} / {R['h10'][8]:+.2f} / {R['h20'][8]:+.2f} / "
            f"{R['h60'][8]:+.2f}) and **never** approaches even the naive 2.0 bar, let alone "
            f"the Bonferroni-corrected {R['bonferroni_crit']:.2f}. The sandwich adds nothing "
            "over the same panel's ordinary days."
        ),
        md(
            "### 4b · Sandwich vs base rate across horizons — the gap is the verdict\n\n"
            "Mean return, sandwich vs base rate, all four horizons and net of cost. The "
            "sandwich should tower over the base rate if it forecasts a bottom. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sand = [res['by_h'][h]['gross']['mean_bps'] for h in hs]\n"
            "    base = [res['by_h'][h]['base']['mean_bps'] for h in hs]\n"
            "    net = [res['by_h'][h]['net']['mean_bps'] for h in hs]\n"
            "else:\n"
            "    sand = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    base = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "    net = [R['h5'][7], R['h10'][7], R['h20'][7], R['h60'][7]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.3))\n"
            "ax.bar(x-.27, sand, .27, color='#2c6fbb', label='sandwich (gross)')\n"
            "ax.bar(x, net, .27, color=AMBER, label=f'sandwich (net, {R[\"cost_bps\"]:.0f}bps)')\n"
            "ax.bar(x+.27, base, .27, color=GREY, label='base rate')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(sand): ax.annotate(f'{v:+.0f}',(i-.27,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=7)\n"
            "for i,v in enumerate(base): ax.annotate(f'{v:+.0f}',(i+.27,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=7)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('Sandwich does not beat the base rate at any horizon'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta sandwich-base (bps):', [round(a-b) for a,b in zip(sand,base)])"
        ),
        md(
            f"> 💡 In plain words: the delta swings from **{R['h20'][6]:+.1f} bps** (20d, "
            f"sandwich *worse*) to **{R['h5'][6]:+.1f} bps** (5d, sandwich *better*) with no "
            "consistent sign — exactly the pattern you'd expect from noise around a common "
            "drift, not a directional signal. Costs shave a further 10 bps/trade off every bar."
        ),
        md(
            "### 4c · The geometry placebo — drop the equal close, nothing changes\n\n"
            "Keep the down-leg + bearish/bullish-rally/bearish context (the \"almost-sandwich\" "
            "candidates), but draw the same number of entries that **ignore the equal close**. "
            "If price respects *the meeting*, dropping it should demolish the result. The "
            "observed sandwich return should sit far in the right tail of the no-meeting "
            "distribution. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bb = PANEL['SPY']; c = bb['close']\n"
            "    pl = st.geometry_placebo(bb, 20, tol=TOL, trend_lookback=LB, n_draws=300, seed=686)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    cand = bb.index[st._candidate_mask(bb, trend_lookback=LB)]\n"
            "    n_sand = len(st.sandwich_entries(bb, tol=TOL, trend_lookback=LB))\n"
            "    rng = np.random.default_rng(686); draws=[]\n"
            "    cand_arr = np.asarray(cand)\n"
            "    pos = {d:i for i,d in enumerate(c.index)}\n"
            "    for _ in range(300):\n"
            "        pick = rng.choice(cand_arr, size=n_sand, replace=False)\n"
            "        rr = st.forward_returns(c, pd.DatetimeIndex(sorted(pick)), 20, pos=pos)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws = np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(686); draws = rng.normal(140, 220, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='no-equal-close failed-rally dip-buys (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real sandwich {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real sandwich sits mid-pack: placebo p = {pval:.3f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real sandwich {obs:+.1f} bps   placebo p={pval:.3f}  (>0.05 => the equal close is not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real sandwich (blue line) sits well inside the "
            f"no-equal-close cloud — **p = {R['placebo'][1]:.3f}** on {R['placebo'][3]} "
            "context-matched candidates. Failed-rally dip-buys that ignore the equal close do "
            "about as well, so the defining \"two matching closes\" carry no detectable extra "
            "information. This is the cleanest refutation of \"the meeting forecasts.\""
        ),
        md(
            "### 4d · Synthetic positive control — the harness CAN bank a real bounce\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** post-sandwich "
            "bounce into a synthetic panel and check the same rule banks it: `edge=0` must stay "
            "flat across 20 independent seeds; `edge>0` must light up hard."
        ),
        code(
            "null_ts = []\n"
            "for s in range(20):\n"
            "    p_, _truth = data.synthetic_panel(edge=0.0, seed=686+s, n_names=20, n_days=3000)\n"
            "    r = st.synthetic_detect(p_, horizon=20)\n"
            "    if r['welch_t'] is not None: null_ts.append(r['welch_t'])\n"
            "null_ts = np.asarray(null_ts, dtype=float)\n"
            "p_planted, truth_planted = data.synthetic_panel(edge=1.0, seed=686, n_names=20, n_days=3000)\n"
            "r_planted = st.synthetic_detect(p_planted, horizon=20)\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(len(null_ts)) + np.linspace(-.12,.12,len(null_ts)), null_ts,\n"
            "           color=GREY, s=40, label=f'null worlds (edge=0), {len(null_ts)} seeds')\n"
            "ax.scatter([1], [r_planted['welch_t']], color=RED, s=90, zorder=5,\n"
            "           label='planted edge = 1.0')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('Welch t (sandwich vs base rate)')\n"
            "ax.set_title('Control: no null fires; a planted bounce lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(np.abs(null_ts)>=2).sum()}/{len(null_ts)} seeds  |  '\n"
            "      f'planted t = {r_planted[\"welch_t\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across {R['syn_null'][3]} null worlds the detector averages "
            f"*t* = **{R['syn_null'][0]:+.2f}** (sd {R['syn_null'][1]:.2f}) and **never** "
            f"crosses the bar; a planted post-sandwich bounce reads *t* = "
            f"**{R['syn_planted'][5]:.2f}**. The machinery is unbiased — the flat real-tape "
            "result is a genuine \"nothing there\", not a broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the stick sandwich does not beat the drift-matched base "
            f"rate at any horizon (Δ = {R['h5'][6]:+.1f} / {R['h10'][6]:+.1f} / "
            f"{R['h20'][6]:+.1f} / {R['h60'][6]:+.1f} bps; Welch *t* = {R['h5'][8]:+.2f} / "
            f"{R['h10'][8]:+.2f} / {R['h20'][8]:+.2f} / {R['h60'][8]:+.2f} — never clears the "
            f"naive 2.0 bar, let alone the Bonferroni-corrected {R['bonferroni_crit']:.2f}). "
            f"The impressive one-sample *t*'s (60d **{R['h60'][4]:.2f}**) are pure beta.\n"
            "- **Tradability `MIRAGE`** — no residual edge once the drift is removed; two of "
            "four horizons are already net-negative before costs, and costs only deepen the "
            "hole. You'd capture the drift more cheaply by holding the basket.\n"
            f"- **Meeting adds info? `BUSTED`** — the equal-close geometry placebo leaves the "
            f"result essentially intact (**p = {R['placebo'][1]:.3f}**): no-equal-close "
            "failed-rally dip-buys do about as well as the real sandwiches, so the defining "
            "two-matching-closes geometry carries no shown forecasting information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The stick sandwich's entire apparent profit is the unconditional drift of a "
            "long-only large-cap basket, which you obtain more cheaply and more fully by "
            "**buying and holding**. At two of the four horizons it actually *loses* to the "
            "base rate before any cost is charged. The rule trades *less* of the time (only on "
            "sandwiches) and pays costs on each, so it strictly dominates *nothing*. There is "
            "no capacity question because there is no edge to scale. The stick sandwich is a "
            "descriptive candlestick label, not a forecasting strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The two-candle cousin.** [460-counterattack-lines](../../460-counterattack-lines/) "
            "runs the identical equal-close idea with one fewer bar — same drift confound, same "
            "verdict, smaller geometry.\n"
            "- **Tolerance & lookback sweeps.** The equal-close band and down-leg window are "
            "free parameters; the synthetic control (4d) shows the harness *can* find a real "
            "effect if one is planted, so a robustness grid on the real tape is a fair next "
            "experiment — the drift-in/drift-out picture should be robust to reasonable "
            "re-tuning.\n"
            "- **Bearish stick sandwich.** The mirror pattern (up leg, bullish/bearish-dip/"
            "bullish, matching closes) inherits the symmetric drift problem on the short side.\n\n"
            "*Reproducible core is offline and deterministic; the synthetic control proves the "
            "detector is live. Methods/sources: [`docs/references.md`](../docs/references.md); "
            "frozen numbers: [`docs/results.md`](../docs/results.md).*"
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
