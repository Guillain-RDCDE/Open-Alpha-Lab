"""Generate the two narrative notebooks for Study 691 (Homing Pigeon).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached basket OHLC
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance un-adjusted daily
# OHLC, 26-name large-cap basket + SPY, 1962-01-02 -> 2026-06-30, 345,846 bars,
# 5,154 homing-pigeon-shaped bars any trend, panel fingerprint 8ddaab4e2e1e,
# SPY data_stamp fingerprint 7987ab25910e, as-of 2026-06-30).
R = dict(
    asof="2026-06-30", start="1962-01-02", end="2026-06-30",
    n_names=26, n_bars=345846, n_shape=5154,
    fp_panel="8ddaab4e2e1e", fp_spy="7987ab25910e",
    # HOMING PIGEON (shape after a downtrend -- the bullish claim): H -> (n, edge%, win%, t, p, p_bonf, net%)
    pigeon={1: (2885, +0.106, 53.2, +2.58, 0.001, 0.005, +0.006),
            3: (2885, +0.224, 55.7, +3.37, 0.000, 0.001, +0.124),
            5: (2882, +0.262, 56.6, +3.16, 0.000, 0.002, +0.162),
            10: (2879, +0.244, 58.0, +2.06, 0.013, 0.051, +0.144)},
    # Alpha vs beta: H -> (pigeon %, any-downtrend-dip %, n_dip, excess Welch t)
    alpha_beta={1: (+0.170, +0.094, 153428, +1.95),
                3: (+0.414, +0.259, 153400, +2.37),
                5: (+0.576, +0.406, 153377, +2.07),
                10: (+0.865, +0.707, 153324, +1.37)},
    # 'wrong side' -- SAME geometry after an UPTREND, traded long as a myth-check
    wrongside={1: (2243, -6.4, 50.2, -1.78, 0.936),
               3: (2242, -3.1, 53.2, -0.51, 0.651),
               5: (2242, -7.4, 54.2, -1.00, 0.789),
               10: (2238, -9.9, 54.2, -0.85, 0.794)},
    # ANY (pooled, ignoring trend, traded long): H -> (n, edge bps, win%, t, p)
    anyside={1: (5152, +3.0, 51.8, +1.12, 0.133),
             3: (5151, +11.0, 54.6, +2.37, 0.007),
             5: (5148, +11.3, 55.5, +1.91, 0.028),
             10: (5141, +9.4, 56.3, +1.03, 0.128)},
    # myth-check filter sweep on pigeon H=3: label -> (edge%, t, p, n)
    filt=[("plain (lookback 10)", +0.224, +3.37, 0.000, 2885),
          ("trend lookback 5", +0.178, +2.89, 0.002, 3198),
          ("trend lookback 20", +0.136, +1.99, 0.020, 2660),
          ("min washout >= 5%", +0.274, +1.68, 0.013, 822),
          ("min washout >= 10%", +0.385, +0.76, 0.077, 196)],
    # cost sweep at H=3 (best horizon): cost_bps -> net bps
    costs=[(0.0, +22.4), (1.0, +20.4), (5.0, +12.4), (10.0, +2.4)],
    # per-name H=3: count of |t|>2, chance baseline
    n_names_over2=3, n_names_chance=1.3,
    # event clustering: n events, distinct weeks, share of events in busiest 10 weeks
    cluster=(2887, 1345, 0.039),
    # synthetic control (H=1, side=any): planted -> (events, edge%, t, p, win%)
    syn=[(0.000, 1219, -0.054, -1.41, 0.916, 49.8),
         (0.006, 1201, +0.534, +13.61, 0.000, 65.4)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Beats_a_downtrend_dip%3F: Mixed](https://img.shields.io/badge/Beats_a_downtrend_dip%3F-Mixed-8b949e?style=flat-square)\n\n"
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

from homing_pigeon import data, strategy as st

HS = [1, 3, 5, 10]
ASOF = "2026-06-30"
HAVE_REAL = data.have_real()
if HAVE_REAL:
    PANEL = {t: b[b.index <= ASOF] for t, b in data.load_real().items()}
else:
    PANEL = None
print("real homing-pigeon cache present:", HAVE_REAL,
      "| names:", (0 if PANEL is None else len(PANEL)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"

HS = [1, 3, 5, 10]


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does a shrinking down-day mark the bottom? 🐦\n"
            "### The homing pigeon — the rarer, same-colour cousin of the harami, on the real tape\n\n"
            + BADGES +
            "After a stock has been sliding, imagine two red (down) days in a row — but the **second** "
            "one is small, and its whole trading range sits **inside** the first, bigger red day's range. "
            "Candlestick traders call this a **homing pigeon**: even though the market is still nominally "
            "falling, the sellers' *conviction* is shrinking — like the down-days are \"returning home\" "
            "to a calmer range. Read as a bottom forming — buy.\n\n"
            "This is the *same-colour* cousin of the "
            "[harami](../../406-harami-pattern/) (which needs *opposite*-coloured candles) and a two-bar "
            "twin of the [inverted hammer](../../684-inverted-hammer/)'s one-bar floor story. We tested "
            "it on **60+ years** of daily bars across 26 big US stocks plus the S&P.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the alpha-vs-beta cut and the per-name "
            "breakdown? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** Fixed **26-name** survivor basket + SPY (names still trading "
            "today), the same panel used by the sibling candlestick studies. Survivors *recover* from "
            "dips that delisted names didn't, so the bias leans **toward** finding a bounce. Charts are "
            "drawn by the code beside them; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the homing pigeon beat the stock's own usual return? | **Yes, on average.** Over 3 "
            f"days it beats the base rate by **{R['pigeon'][3][1]:+.3f}%** (*t* = {R['pigeon'][3][3]:+.2f}) "
            "— and unlike its candlestick cousins on this desk, that edge survives even after a strict "
            "multiple-testing correction at most horizons. This is the best raw reading we've found in "
            "five candlestick studies. |\n"
            "| Is that really *this pattern*, or just \"buy any dip in a downtrend\"? | **Mostly the "
            "pattern, sometimes just the dip.** Compared to buying *any* day in the same downtrend (no "
            "pattern needed), the shape adds something extra at 3 and 5 days — but at 1 and 10 days it's "
            "statistically the same as just buying the dip. |\n"
            "| Would this work if I traded one stock at a time? | **Probably not reliably.** Pooled "
            f"across 26 names it looks solid, but only **{R['n_names_over2']} of 26** stocks individually "
            f"show a statistically real edge — right around the ~{R['n_names_chance']:.1f} you'd expect "
            "from pure chance. |\n"
            "| Could I trade it? | **On paper, yes — this one survives costs.** Unlike every other "
            "candlestick pattern tested on this desk, the net edge stays positive even after realistic "
            "trading costs. |\n\n"
            "> The honest picture: a real, broad, cost-surviving tilt that's *partly* pattern-specific and "
            "*partly* just \"stocks bounce after they fall\" — and too thin, per stock, to bet on any one "
            "name. **Weak signal, fragile trade.**"
        ),

        # ---- BEAT 1 --------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A homing pigeon is two down days in a row where the second, smaller one's whole range "
            "sits inside the first. Appearing after a downtrend, it shows sellers running out of steam — "
            "the bottom is near, buy.\"*\n\n"
            "It appears in **Steve Nison**'s candlestick canon and in Thomas Bulkowski's *Encyclopedia of "
            "Candlestick Charts* (2008), where Bulkowski's own backtest ranks it among the *better* "
            "bullish candle patterns — one of the few with a claimed edge worth a dedicated teardown "
            "rather than a quick dismissal alongside its zoo of siblings."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If shrinking conviction inside a slide reliably marked the bottom, it would be a rare but "
            "clean, mechanical timing signal. But two traps lurk: **(1)** stocks that have been falling "
            "often bounce a little regardless of any candle shape (mean-reversion after a decline is a "
            "well-documented, generic effect) — so the fair test isn't \"does it rise after a homing "
            "pigeon?\" but \"does it rise **more than just buying any dip in the same downtrend**?\" "
            "**(2)** we're testing four different holding periods at once, which needs a multiple-testing "
            "correction, or a lucky horizon can carry a false headline."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We scan **{R['n_bars']:,} daily bars** across **{R['n_names']} stocks + SPY** "
            f"({R['start']} → {R['end']}) and flag every bar pair with the homing-pigeon shape — a "
            f"smaller down day fully inside a bigger prior down day. That's **{R['n_shape']:,}** "
            f"homing-pigeon-shaped pairs in total, of which **{R['pigeon'][3][0]:,}** sit after a genuine "
            "downtrend (the bullish claim).\n\n"
            "1. **The shape.** Detected by exact open/high/low/close rules — no eyeballing.\n"
            "2. **The trend split.** Only pairs after a real 10-day downtrend count as the claim; the "
            "same shape after an *uptrend* is a myth-check contrast.\n"
            "3. **The forward move.** Starting at the **next** day's close (no cheating), how does the "
            "stock do over the next **1, 3, 5, 10** days versus (a) its own usual return, and (b) just "
            "buying any day in the same downtrend?\n"
            "4. **The honesty checks.** A Bonferroni correction for the four horizons, a per-name "
            "breakdown (does *any single stock* actually carry this?), and a check that the events aren't "
            "secretly just a few crash weeks repeated 26 times.\n\n"
            "We'd call it a **mirage** if the edge can't clear a basic luck test or a per-stock "
            "reproducibility check; a clean **real** if it clears everything, including beating the "
            "\"just buy the dip\" baseline."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**The bullish claim first.** How much does the homing pigeon beat the stock's *own normal* "
            "return at each horizon — and does the *t*-stat (the honesty line) clear 2?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(PANEL, side='pigeon', n_draws=1500, seed=691)\n"
            "    edge = [res[h]['edge_mean']*100 for h in HS]; tvals = [res[h]['t'] for h in HS]\n"
            "else:\n"
            "    edge = [R['pigeon'][h][1] for h in HS]; tvals = [R['pigeon'][h][3] for h in HS]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "cols = [GREEN if v>0 else RED for v in edge]\n"
            "a1.bar([f'{h}d' for h in HS], edge, color=cols, width=.6)\n"
            "for i,v in enumerate(edge): a1.annotate(f'{v:+.3f}%',(i,v),ha='center',va='bottom')\n"
            "a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_ylabel(\"pigeon edge over the stock's own base rate (%)\")\n"
            "a1.set_title('A real-looking tilt...')\n"
            "a2.bar([f'{h}d' for h in HS], tvals, color=AMBER, width=.6)\n"
            "a2.axhline(2, ls='--', c=RED, label='t = 2 bar')\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "for i,t in enumerate(tvals): a2.annotate(f't={t:+.2f}',(i,t),ha='center',va='bottom')\n"
            "a2.set_ylabel('HAC t'); a2.set_title('...that clears the bar at every horizon'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('edge by horizon (%):', [round(v,3) for v in edge], ' t:', [round(v,2) for v in tvals])"
        ),
        md(
            f"That's genuinely the best raw reading of any candlestick pattern on this desk — "
            f"**{R['pigeon'][3][1]:+.3f}% at 3 days, *t* = {R['pigeon'][3][3]:+.2f}**, clearing the bar. "
            "But 'beats the stock's own usual return' isn't the whole story — the stock's own usual "
            "return already includes days it wasn't falling. The sharper question: does it beat just "
            "**buying any dip while the stock is already in a downtrend**?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pig, dip, ext = [], [], []\n"
            "    for h in HS:\n"
            "        cr = st.conditional_returns(PANEL, h, side='pigeon')\n"
            "        dtp = st.downtrend_pool(PANEL, h)\n"
            "        pig.append(cr['cond_mean']*100); dip.append(dtp.mean()*100)\n"
            "        ext.append(st.welch_t(cr['cond'], dtp))\n"
            "else:\n"
            "    pig = [R['alpha_beta'][h][0] for h in HS]; dip = [R['alpha_beta'][h][1] for h in HS]\n"
            "    ext = [R['alpha_beta'][h][3] for h in HS]\n"
            "x = np.arange(len(HS))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, pig, .4, color=GREEN, label='after a HOMING PIGEON')\n"
            "ax.bar(x+.2, dip, .4, color=GREY, label='ANY dip in the same downtrend (no shape needed)')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in HS])\n"
            "ax.set_ylabel('long return (%)')\n"
            "ax.set_title('The shape adds a little over plain dip-buying -- but not always enough')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for h, t in zip(HS, ext): print(f'H={h:2d}d  excess over dip-buying, Welch t = {t:+.2f}')"
        ),
        md(
            f"At 3 and 5 days the shape clears the *t* = 2 bar over plain dip-buying "
            f"(*t* = {R['alpha_beta'][3][3]:+.2f} / {R['alpha_beta'][5][3]:+.2f}); at 1 and 10 days it "
            f"doesn't (*t* = {R['alpha_beta'][1][3]:+.2f} / {R['alpha_beta'][10][3]:+.2f}) — a split "
            "verdict, not a clean win."
        ),
        md(
            "**One more honesty check.** If only 3 of 26 stocks individually show a real edge, is the "
            "pooled result something you could actually trade one name at a time?"
        ),
        code(
            "print(f\"names with individually significant edge (|t|>2): {R['n_names_over2']} of \"\n"
            "      f\"{R['n_names']} — chance level is about {R['n_names_chance']:.1f}\")\n"
            "print('Pooled significance here rests on averaging many small, broadly positive but mostly '\n"
            "      'individually-noisy samples -- not on any one stock carrying a robust, repeatable edge.')"
        ),

        # ---- BEAT 5 ------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** The pooled edge clears *t* = 2 at every horizon (best "
            f"{R['pigeon'][3][3]:+.2f} at 3d), but the sharper \"beats plain dip-buying\" test only "
            "clears it at 3-5 days, and only 3 of 26 stocks individually show a real, reproducible edge.\n"
            "- **Tradability — Fragile.** Unlike every sibling candlestick study, the net edge survives "
            "realistic trading costs — but the edge is thin per stock and partly just dip-buying beta.\n"
            "- **Beats plain dip-buying? — Mixed.** Real at 3-5 days, gone at 1 and 10 days."
        ),

        # ---- BEAT 6 ------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Each event is a fresh round trip: a spread in, a spread out. This is the *one* candlestick "
            "pattern on this desk whose 3-day edge stays positive all the way to a punitive 10 bps."
        ),
        code(
            "costs = [c for c,_ in R['costs']]\n"
            "if HAVE_REAL:\n"
            "    cr3 = st.conditional_returns(PANEL, 3, side='pigeon')\n"
            "    net = [st.net_of_costs(cr3['edge_mean'], cost_bps=c)*1e4 for c in costs]\n"
            "else:\n"
            "    net = [n for _,n in R['costs']]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, net, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, net, 0, where=[n<0 for n in net], color=RED, alpha=.12)\n"
            "ax.set_xlabel('one-way cost (bps)'); ax.set_ylabel('net per event (bps, 3-day hold)')\n"
            "ax.set_title('Still net-positive at 10 bps one-way -- costs are not what kills this one')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('gross 3d =', R['pigeon'][3][1], '%  ->  net at 5bps =', R['costs'][2][1], 'bps')"
        ),

        # ---- BEAT 7 ------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The positive control.** The companion notebook plants a real day-after bounce in a "
            "synthetic tape — the detector lights up cleanly (*t* ≈ 13.6). The engine works; the honest "
            "gap here is between a broad pooled tilt and a robust, single-name trade.\n"
            "- **Portfolio, not single-name.** The only way this pattern's edge looks deployable is as a "
            "diversified basket rule across many names — never as a signal to bet on one stock.\n"
            "- **The harami cousin.** [Study 406](../../406-harami-pattern/) tests the *opposite*-colour "
            "containment rule — busted as a two-legged reversal, real only on its long (beta-riding) leg.\n\n"
            "*Think the shape carries more information on a different basket, timeframe, or holding rule? "
            "Fork this, change it, and show the excess-over-dip-buying test clearing *t* = 2 at every "
            "horizon with a majority of names individually significant. That is the bar.*"
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
            "# Homing Pigeon — a quantitative teardown\n"
            "### Two-bar same-colour containment · trend split · alpha-vs-beta · Bonferroni · per-name\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We detect every homing pigeon "
            "across a fixed 26-name basket + SPY, measure the signed forward LONG return (entered after "
            "a downtrend), and ask whether it clears HAC *t* = 2 against (a) the unconditional base rate "
            "and (b) — the decisive cut — the return from buying *any* dip in the same downtrend.\n\n"
            "> **Not investment advice.** Real data: yfinance un-adjusted daily OHLC, "
            f"{R['start']} → {R['end']} (as-of {R['asof']}, panel fingerprint `{R['fp_panel']}`); the "
            "offline core and the synthetic control run with no network. Methods & sources in "
            "[`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Pooled edge clears HAC *t* ≥ 2 at all four horizons "
            f"(best **{R['pigeon'][3][3]:+.2f}** at 3d) and Bonferroni survives 3 of 4 (10d adjusted "
            f"*p* = {R['pigeon'][10][5]:.3f}), but the excess-over-dip-buying Welch *t* clears 2 only at "
            f"3/5 days ({R['alpha_beta'][3][3]:+.2f} / {R['alpha_beta'][5][3]:+.2f}), and only "
            f"**{R['n_names_over2']} of {R['n_names']}** names individually clear \\|*t*\\| > 2 — chance "
            "level. |\n"
            f"| **Tradability** | `FRAGILE` | Net of a 5 bps round trip the 3-day rule nets "
            f"**+{R['pigeon'][3][6]*100:.1f} bps** and stays positive to 10 bps — unlike every sibling "
            "candlestick study — but the edge is thin per name and partly beta. |\n"
            "| **Beats plain dip-buying?** | `MIXED` | Clears *t* ≥ 2 at 3-5 days, misses at 1 and "
            "10 days. |\n\n"
            "> In plain words: the homing pigeon is the strongest *raw* reading of this desk's five "
            "candlestick studies, but every cut that goes beyond the raw pooled *t* — the alpha-vs-beta "
            "test, the per-name breakdown — trims it back from 'real' to 'weak, broad, and thin per name'."
        ),

        # ---- BEAT 1 ------------------------------------------------------
        md(
            "## 1 · The claim, formalised\n\n"
            "Let $(O_0,C_0)$ be the prior bar and $(O_1,C_1)$ the current bar. A **homing pigeon** "
            "requires both bars to be DOWN days with the current body strictly smaller and fully "
            "contained in the prior one:\n\n"
            "$$ O_0 - C_0 > 0,\\quad O_1 - C_1 > 0,\\quad 0 < (O_1-C_1) < (O_0-C_0) $$\n"
            "$$ C_1 \\ge C_0 \\quad\\text{and}\\quad O_1 \\le O_0 $$\n\n"
            "confirmed after a **downtrend** ($\\text{close}_t / \\text{close}_{t-10} - 1 < 0$). Entered "
            "at the open of $t+1$'s close (one execution lag: signal known at close $t$, entry at close "
            "$t+1$), held $H$ bars, long only. The hypotheses:\n\n"
            "- **H₁ (signal vs base rate).** $\\mathbb{E}[r_{\\text{pigeon}}] - \\mathbb{E}[r_{\\text{base}}] "
            "> 0$ with HAC *t* ≥ 2.\n"
            "- **H₂ (alpha vs beta).** The pigeon beats not just the base rate but *any* dip in the same "
            "downtrend (shape-specific information, not generic mean reversion).\n"
            "- **H₃ (trend-conditional).** The identical shape after an *uptrend* should NOT show the "
            "same tilt.\n"
            "- **H₄ (reproducible, not an averaging artefact).** A meaningful share of names individually "
            "clear |*t*| > 2, and events aren't a repackaged handful of crash weeks.\n"
            "- **H₅ (tradable).** Survives a realistic round trip.\n\n"
            "We find H₁ holds cleanly, H₃ holds cleanly (the trend split genuinely discriminates), H₅ "
            "holds (unusually, for this desk) — but H₂ only partially holds (2 of 4 horizons) and H₄ only "
            "partially holds (broad, uncorrelated with crash dates, but individually thin) — hence "
            "**Weak × Fragile × Mixed**."
        ),

        # ---- BEAT 2 ------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "This is the first candlestick pattern on this desk whose *pooled* statistic looks genuinely "
            "real (Marshall, Young & Rose 2006 and Horton 2009 find no value across the candlestick zoo — "
            "the prior every sibling study here has confirmed). If it held up fully, it would be a rare, "
            "specific, tradable exception. The actual finding — real pooled, thinner under the honest "
            "alpha-vs-beta and per-name cuts — is exactly the kind of result the desk's inference bar "
            "exists to catch before it gets oversold as 'the one candlestick pattern that works.'"
        ),

        # ---- BEAT 3 ------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Detector.** Same-colour containment: both bars down, current body strictly smaller and "
            "fully inside the prior body (see `strategy.is_homing_pigeon`).\n"
            "- **Trend.** Sign of the trailing 10-day close-to-close change; `pigeon` = shape after a "
            "downtrend, `wrongside` = shape after an uptrend (myth-check), `any` = shape regardless.\n"
            "- **Entry.** Next bar's *close* (one execution lag). **Exit.** Close of bar $t+1+H$.\n"
            "- **Benchmark 1.** The unconditional forward return of every bar in the same name.\n"
            "- **Benchmark 2 (the decisive cut).** The pooled forward return of *every* bar sitting in a "
            "downtrend, shape not required — isolates shape-specific information from generic "
            "downtrend mean reversion (`strategy.downtrend_pool`, `strategy.welch_t`).\n"
            "- **Inference.** Newey–West HAC *t* on the conditional edge; a per-name label-shuffle "
            "placebo; a **Bonferroni** correction across the four-horizon family "
            "(`strategy.bonferroni`).\n"
            "- **Reproducibility check.** A per-name |HAC *t*| > 2 count against its chance baseline, and "
            "an event-clustering diagnostic (`strategy.event_clustering`) — is the pooled sample really "
            "a handful of shared crash weeks?\n"
            "- **Costs.** One-way bps × NAV charged on the round trip; long-only, no borrow.\n"
            "- **Myth-check.** Does a shorter/longer trend window or a deeper washout filter change it?\n"
            "- **Positive control.** Synthetic tape with a planted day-after bounce — the harness must "
            "recover it and must read ≈ 0 when the edge is 0.\n\n"
            "Basket: 26 long-listed large-caps + SPY (same panel as 403/406/684); yfinance un-adjusted "
            "daily OHLC, 1962→2026, **survivors** (named on the Signal axis)."
        ),

        # ---- BEAT 4 ------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Pooled edge by horizon — and its HAC *t*\n\n"
            "If the pattern is real, the edge over the base rate should clear +2 cleanly at every "
            "horizon, and the Bonferroni-adjusted placebo should hold."
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(PANEL, side='pigeon', n_draws=2000, seed=691)\n"
            "    edge = [res[h]['edge_mean']*100 for h in HS]; tvals = [res[h]['t'] for h in HS]\n"
            "    pbonf = [res[h]['p_bonferroni'] for h in HS]\n"
            "else:\n"
            "    edge = [R['pigeon'][h][1] for h in HS]; tvals = [R['pigeon'][h][3] for h in HS]\n"
            "    pbonf = [R['pigeon'][h][5] for h in HS]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot([f'{h}d' for h in HS], edge, 'o-', c=GREEN, lw=2, label='edge over base rate (%)')\n"
            "ax2 = ax.twinx()\n"
            "ax2.plot([f'{h}d' for h in HS], tvals, 's--', c=GREY, lw=1.5, label='HAC t')\n"
            "ax2.axhline(2, ls=':', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('hold horizon'); ax.set_ylabel('edge (%)', color=GREEN)\n"
            "ax2.set_ylabel('HAC t-stat', color=GREY)\n"
            "ax.set_title('Pooled edge clears t=2 at every horizon -- the strongest raw read on this desk')\n"
            "plt.tight_layout(); plt.show()\n"
            "for h, e, t, pb in zip(HS, edge, tvals, pbonf):\n"
            "    print(f'H={h:2d}  edge={e:+.3f}%  HAC t={t:+.2f}  Bonferroni p={pb:.3f}')"
        ),
        md(
            f"> In plain words: pooled, every horizon clears *t* = 2 (best {R['pigeon'][3][3]:+.2f} at "
            f"3d). Bonferroni survives at 1/3/5 days; at 10 days the adjusted *p* is "
            f"{R['pigeon'][10][5]:.3f} — just over the 5% line. That's already the best raw reading among "
            "five candlestick studies on this desk. Now the harder question."
        ),
        md(
            "### 4b · Alpha vs beta — does the shape beat plain dip-buying?\n\n"
            "The pigeon only ever fires *inside* a downtrend, where short-horizon mean reversion is "
            "already a documented phenomenon (De Bondt & Thaler 1985). The honest test: Welch *t* of the "
            "pigeon's conditional return **minus the pooled downtrend-only return** (shape not required)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ex_t, pig_m, dip_m = [], [], []\n"
            "    for h in HS:\n"
            "        cr = st.conditional_returns(PANEL, h, side='pigeon')\n"
            "        dtp = st.downtrend_pool(PANEL, h)\n"
            "        ex_t.append(st.welch_t(cr['cond'], dtp))\n"
            "        pig_m.append(cr['cond_mean']*100); dip_m.append(dtp.mean()*100)\n"
            "else:\n"
            "    ex_t = [R['alpha_beta'][h][3] for h in HS]\n"
            "    pig_m = [R['alpha_beta'][h][0] for h in HS]; dip_m = [R['alpha_beta'][h][1] for h in HS]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.0))\n"
            "col = [GREEN if t>=2 else AMBER for t in ex_t]\n"
            "ax.bar([f'{h}d' for h in HS], ex_t, color=col)\n"
            "ax.axhline(2, ls=':', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Welch t (pigeon minus any-downtrend-dip)')\n"
            "ax.set_title('The shape beats plain dip-buying only at 3-5 days')\n"
            "plt.tight_layout(); plt.show()\n"
            "for h, t in zip(HS, ex_t): print(f'H={h:2d}  excess Welch t = {t:+.2f}')"
        ),
        md(
            f"> In plain words: excess *t* clears 2 at 3d ({R['alpha_beta'][3][3]:+.2f}) and 5d "
            f"({R['alpha_beta'][5][3]:+.2f}), but not at 1d ({R['alpha_beta'][1][3]:+.2f}) or 10d "
            f"({R['alpha_beta'][10][3]:+.2f}) — a genuinely split answer. At 1 and 10 days, buying any "
            "dip in the same downtrend would have captured essentially the same return."
        ),
        md(
            "### 4c · The trend split — does it genuinely discriminate?\n\n"
            "If the downtrend condition is doing real work (not window-dressing), the *identical* shape "
            "traded long after an **uptrend** should look different — flat or negative."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rp = st.run_experiment(PANEL, side='pigeon', placebo=False, n_draws=1, seed=691)\n"
            "    rw = st.run_experiment(PANEL, side='wrongside', placebo=False, n_draws=1, seed=691)\n"
            "    pig = [rp[h]['edge_mean']*1e4 for h in HS]; wrong = [rw[h]['edge_mean']*1e4 for h in HS]\n"
            "else:\n"
            "    pig = [R['pigeon'][h][1]*100 for h in HS]\n"
            "    wrong = [R['wrongside'][h][1] for h in HS]\n"
            "x = np.arange(len(HS))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, pig, .4, color=GREEN, label='after a DOWNTREND (the claim)')\n"
            "ax.bar(x+.2, wrong, .4, color=RED, label='after an UPTREND (wrong side)')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in HS])\n"
            "ax.set_ylabel('edge over base rate (bps)')\n"
            "ax.set_title('Unlike the inverted hammer, the trend split flips the sign here')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('Downtrend side positive at every horizon; uptrend side flat-to-negative.')"
        ),
        md(
            "> In plain words: every wrong-side horizon is flat-to-negative "
            f"({R['wrongside'][10][1]:+.1f} bps at 10d) — a real contrast with the downtrend side. This "
            "is unlike the inverted hammer's wrong-side test (same sign both ways); here the trend "
            "condition is genuinely load-bearing."
        ),
        md(
            "### 4d · Event clustering — a broad tilt, or a handful of crash weeks?\n\n"
            "Pooling 26 names only inflates a naive *t* if the \"many\" events are really a shared "
            "market-wide date repeated many times."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ec = st.event_clustering(PANEL, side='pigeon')\n"
            "else:\n"
            "    ec = {'n': R['cluster'][0], 'n_weeks': R['cluster'][1], 'top10_week_share': R['cluster'][2]}\n"
            "print(f\"events={ec['n']}  distinct ISO weeks={ec['n_weeks']}  \"\n"
            "      f\"share in busiest 10 weeks={ec['top10_week_share']*100:.1f}%\")"
        ),
        md(
            f"> In plain words: **{R['cluster'][0]:,}** events spread across **{R['cluster'][1]:,}** "
            f"distinct calendar weeks; the busiest 10 weeks account for only **"
            f"{R['cluster'][2]*100:.1f}%** of the total. No single crash (2008, 2020, 2022) is carrying "
            "the pooled result — it's a genuinely broad, low-frequency tilt."
        ),
        md(
            "### 4e · Per-name breakdown — is it reproducible stock by stock?"
        ),
        code(
            f"print(f\"names with individually significant edge (|t|>2): {{R['n_names_over2']}} of \"\n"
            f"      f\"{{R['n_names']}} (chance baseline ~{{R['n_names_chance']:.1f}})\")\n"
            "print('Pooled significance rests on breadth of small, mostly-positive per-name samples, '\n"
            "      'not on a robust, individually-reproducible edge in most single names.')"
        ),

        # ---- BEAT 5 ------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — pooled HAC *t* ≥ 2 at every horizon and Bonferroni holds at 3 of 4, "
            "but the alpha-vs-beta cut clears *t* = 2 only at 3-5 days, and only "
            f"{R['n_names_over2']} of {R['n_names']} names individually clear |*t*| > 2. Real, broad, "
            "not repackaged crash weeks — but not the clean, robust real the pooled *t* alone suggests.\n"
            f"- **Tradability `FRAGILE`** — net of a 5 bps round trip the 3-day rule nets "
            f"+{R['pigeon'][3][6]*100:.1f} bps and survives to 10 bps — unusual for this desk — but the "
            "edge is thin per name and partly a simpler dip-buying rule.\n"
            "- **Beats plain dip-buying? `MIXED`** — clears at 3-5 days, misses at 1 and 10 days."
        ),

        # ---- BEAT 6 ------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the myth-check and the cost ladder\n\n"
            "First the myth-check: does a shorter/longer trend window or a deeper washout filter rescue "
            "or strengthen it (H=3)?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    base = st.run_experiment(PANEL, side='pigeon', horizons=(3,), n_draws=2000, seed=691)[3]\n"
            "    rows.append(('plain (lookback 10)', base['edge_mean']*100, base['t'], base['p_placebo'], base['n']))\n"
            "    for lb in (5, 20):\n"
            "        r = st.run_experiment(PANEL, side='pigeon', horizons=(3,), lookback=lb, n_draws=2000, seed=691)[3]\n"
            "        rows.append((f'trend lookback {lb}', r['edge_mean']*100, r['t'], r['p_placebo'], r['n']))\n"
            "    for ms in (0.05, 0.10):\n"
            "        r = st.run_experiment(PANEL, side='pigeon', horizons=(3,), min_strength=ms, n_draws=2000, seed=691)[3]\n"
            "        rows.append((f'min washout >= {ms:.0%}', r['edge_mean']*100, r['t'], r['p_placebo'], r['n']))\n"
            "else:\n"
            "    rows = R['filt']\n"
            "mc = pd.DataFrame(rows, columns=['variant','edge%','t','p','n'])\n"
            "print(mc.round(3).to_string(index=False))\n"
            "print('\\nA deeper washout filter thins the sample and WEAKENS it -- not a dose-response '\n"
            "      'a real deeper-confirmation effect would show.')"
        ),
        md("Now the cost ladder on the 3-day rule (its best horizon):"),
        code(
            "costs = [c for c,_ in R['costs']]\n"
            "if HAVE_REAL:\n"
            "    g3 = st.conditional_returns(PANEL, 3, side='pigeon')['edge_mean']\n"
            "else:\n"
            "    g3 = R['pigeon'][3][1] / 100\n"
            "net = [st.net_of_costs(g3, cost_bps=c)*1e4 for c in costs]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.0))\n"
            "ax.plot(costs, net, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, net, 0, where=[n<0 for n in net], color=RED, alpha=.12)\n"
            "ax.set_xlabel('one-way cost (bps)'); ax.set_ylabel('net per event (bps, 3-day)')\n"
            "ax.set_title('Net-positive even at 10 bps one-way -- unusual for this desk')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross 3d = {g3*100:+.3f}%  ->  net at 5bps = {net[2]:+.1f}bps')"
        ),

        # ---- BEAT 7 ------------------------------------------------------
        md(
            "## 7 · Going further — the positive control\n\n"
            "Is the *engine* capable of finding a real post-pigeon bounce? Plant a day-after drift in a "
            "synthetic tape (edge = 0 -> must read noise; edge > 0 -> must light up). Note: with a "
            "gapless random walk the homing-pigeon shape is mathematically impossible (the current close "
            "would have to sit on both sides of the prior close at once), so the synthetic generator "
            "includes a small overnight gap — see `data.synthetic_panel`."
        ),
        code(
            "for edge in (0.0, 0.006):\n"
            "    pan, truth = data.synthetic_panel(edge=edge, seed=691)\n"
            "    s = st.run_experiment(pan, side='any', horizons=(1,), n_draws=2000, seed=691)[1]\n"
            "    print(f'planted edge={edge:+.4f}: n={s[\"n\"]:>5}  edge={s[\"edge_mean\"]*100:+.3f}%  '\n"
            "          f'HAC t={s[\"t\"]:+.2f}  placebo p={s[\"p_placebo\"]:.3f}  win={s[\"win\"]*100:.0f}%')\n"
            f"print('\\nNull tape: t≈{R['syn'][0][3]:.2f} (noise). Planted tape: t≈{R['syn'][1][3]:.1f} "
            "— engine is faithful.')"
        ),
        md(
            f"The engine is a faithful reversal detector: on a planted tape it lights up at *t* ≈ "
            f"{R['syn'][1][3]:.0f}, and on a null tape it reads ≈ {R['syn'][0][3]:.2f} (noise, placebo "
            f"*p* = {R['syn'][0][4]:.2f}). The real-tape reading is therefore a genuine, if fragile, "
            "signal — not a blind harness. Forks worth trying: a diversified basket implementation (never "
            "single-name); a stricter containment rule (e.g. requiring the second body's midpoint below "
            "the first's) to test whether it concentrates the alpha-vs-beta excess; international or "
            "small-cap tapes where mean reversion is more native."
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
