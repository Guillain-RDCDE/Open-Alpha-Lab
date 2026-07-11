"""Generate the two narrative notebooks for Study 666 (McClellan Summation Index).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached breadth-basket
tapes under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
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


# Frozen real-tape headline numbers — mirror of docs/results.md. SPY + 9 SPDR sector ETFs,
# 2005-01-04 -> 2026-06-30 (21.4 years; 19.5 post warm-up), classic McClellan EMA spans 19/39.
R = dict(
    asof="2026-06-30", start="2005-01-04", end="2026-06-30", years=21.4, years_post=19.5,
    basket=["SPY", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"],
    fp_spy="8266ce2012be", fp_breadth="d10dadd9e2b7",
    summ_min=0.00, summ_max=128.45, warmup_days=500,
    raw_up_crosses=1, raw_dn_crosses=0, post_up_crosses=0, post_dn_crosses=0,
    # extreme z-score cross event study: n, then per horizon (trig_bps, one_t, rand_bps, delta_bps, welch_t)
    ext_up_n=45,
    ext_up={5: (14.7, 0.94, -24.4, 39.2, 0.78), 10: (30.8, 1.06, 11.9, 18.9, 0.27),
            20: (128.1, 3.02, 125.7, 2.5, 0.03), 60: (240.7, 2.85, 564.6, -323.9, -2.50)},
    ext_dn_n=43,
    ext_dn={5: (24.1, 0.56, -24.4, 48.5, 0.74), 10: (-10.7, -0.17, 11.9, -22.7, -0.24),
            20: (134.6, 1.44, 125.7, 8.9, 0.08), 60: (492.0, 4.00, 564.6, -72.6, -0.45)},
    placebo_ext_obs=128.1, placebo_ext_p=0.309, placebo_ext_draws=500,
    # regime timer
    regime_lvl_switches=0, regime_lvl_excess=0.000,
    regime_z_switches=115, regime_z_per_yr=5.9, regime_z_frac_long=53,
    regime_z_cagr=3.95, bh_cagr=10.90, regime_z_vol=10.23, bh_vol=19.63,
    regime_z_sharpe=0.43, bh_sharpe=0.63,
    regime_z_excess_bps=-3.125, regime_z_excess_t=-2.43,
    placebo_regime_t=-2.43, placebo_regime_p=0.491, placebo_regime_draws=500,
    # robustness: (fast, slow, switches, excess_bps, t)
    spans=[(10, 20, 250, -3.221, -2.53), (19, 39, 115, -3.125, -2.43), (25, 50, 99, -3.032, -2.37)],
    fallback_switches=97, fallback_excess=-2.979, fallback_t=-2.34,
    # synthetic control
    syn_null_lvl_mean=-0.25, syn_null_lvl_sd=1.00, syn_null_lvl_fire=0,
    syn_null_z_mean=-0.36, syn_null_z_sd=0.94, syn_null_z_fire=0,
    syn_planted_lvl_t=10.12, syn_planted_lvl_bps=14.11,
    syn_planted_z_t=4.40, syn_planted_z_bps=6.81,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Confirms_turns%3F: Busted](https://img.shields.io/badge/Confirms_turns%3F-Busted-8b949e?style=flat-square)\n\n"
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

from mcclellan_summation import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    CLOSE = data.load_real(data.TRADED)["close"]
    CLOSE = CLOSE[CLOSE.index <= "2026-06-30"]
    NET = data.load_breadth(asof="2026-06-30").reindex(CLOSE.index).dropna()
    CLOSE = CLOSE.reindex(NET.index)
    SUMM = st.summation_index(NET)
else:
    CLOSE = NET = SUMM = None
print("real cache present:", HAVE_REAL, "| tape days:", (0 if CLOSE is None else len(CLOSE)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Just add up the breadth oscillator — does that finally forecast the market? 📊\n"
            "### The McClellan Summation Index — a good idea that runs into a structural wall\n\n"
            + BADGES +
            "Study [491](../../491-mcclellan-oscillator/) already showed the McClellan Oscillator "
            "(a smoothed daily \"more stocks up than down?\" gauge) doesn't forecast the S&P. "
            "Believers have a comeback: *\"the daily oscillator is noisy — but keep a **running "
            "total** of it, and the level itself becomes a regime gauge. Crossing zero confirms a "
            "new bull or bear market. Extreme readings mark exhaustion.\"* That's the **Summation "
            "Index** — literally the oscillator's running total, nothing more exotic.\n\n"
            "It sounds like exactly the right fix for noise: average many days together, and small "
            "wobbles should cancel out, leaving the real signal. We built it and tested it — and hit "
            "something we didn't expect going in.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebos and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** Breadth proxy = SPY + the 9 classic SPDR sector ETFs "
            "(2005→2026, no survivorship — the actual traded universe throughout). House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the Summation Index cross zero and confirm regimes, like the books say? | "
            f"**It can't.** Over **{R['years']}** years it crossed zero exactly **once** — in its "
            "first week of history — and never again, in either direction. |\n"
            "| Why would a \"running total\" get stuck like that? | Because stocks close up more "
            "often than down, on average, over two decades. A running total of a mildly "
            "positive-biased daily gauge **climbs away from zero and stays away** — like a tally "
            "that starts at zero and almost always adds, rarely subtracts: it heads north and "
            "doesn't look back. |\n"
            "| OK, what about the rescaled version — extreme highs/lows instead of zero? | **Still "
            "no edge.** Forward SPY returns after an extreme breadth reading are statistically the "
            "same as after a random day, at every time horizon we checked, in both directions. |\n"
            "| Can you at least trade the \"stay long in bull regimes\" idea? | You can build a "
            f"working version — but it **loses to just buying and holding**, by about "
            f"**{abs(R['regime_z_excess_bps']):.1f} basis points a day**, badly enough to fail our "
            "statistical bar. |\n\n"
            "> A clever fix for a noisy signal — that turns out to have its own, different problem."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The daily McClellan Oscillator is noisy, sure — but its **running total** filters "
            "that out. When the Summation Index crosses up through zero, breadth momentum has "
            "genuinely turned and a new bull phase is confirmed. Extreme readings (the old books "
            "say ±500, even ±1000) mark overbought or oversold exhaustion — time to fade the move.\"*\n\n"
            "It's a completely reasonable-sounding idea: smoothing by accumulation is a real "
            "technique (that's what a moving average does, after all). The Summation Index just "
            "takes it one step further — instead of averaging the last N days, it keeps *every* "
            "day, forever, in a running tally."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a running total of breadth genuinely confirms regimes, that's a much bigger claim "
            "than 491's single-day trigger — it's a **standing market-timing system**: stay long "
            "while the tally says bull, go flat (or short) when it flips. Every technical-analysis "
            "course teaches the ±500/±1000 extreme levels as textbook overbought/oversold signals. "
            "If that's real, it should show up cleanly in 21 years of data — and if it isn't, we "
            "should be able to say exactly *why* it fails, not just that it does."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "- **Build it honestly.** A running total of the same EMA₁₉−EMA₃₉ breadth gauge from "
            "study 491, computed causally (no peeking ahead), across SPY and the 9 classic sector "
            "ETFs.\n"
            "- **Test the zero-cross literally first.** Does it even cross, and when it does, does "
            "SPY do better afterward than on a random day?\n"
            "- **Rescale the ±500 level honestly.** The textbook numbers assume the full New York "
            "Stock Exchange (thousands of stocks); our 10-name proxy needs its own version of "
            "\"extreme\" — we use a rolling one-year z-score, ±1 standard deviation.\n"
            "- **Build the actual trade.** Stay long while the indicator says bull, go flat "
            "otherwise, pay real costs on every switch, compare to just buying and holding."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First surprise: the zero-cross basically doesn't exist.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    summ = SUMM\n"
            "else:\n"
            "    summ = None\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.4))\n"
            "if summ is not None:\n"
            "    ax.plot(summ.index, summ.values, color=GREY, lw=1.1)\n"
            "    ax.axhline(0, c=RED, lw=1.6, ls='--', label='the textbook regime-turn level (zero)')\n"
            "    ax.set_title('The Summation Index, 2005-2026: it never comes back down to zero')\n"
            "else:\n"
            "    ax.text(0.5, 0.5, 'cache miss — see docs/results.md for the pinned chart',\n"
            "            ha='center', va='center', transform=ax.transAxes)\n"
            "ax.set_ylabel('Summation Index level'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            f"print('post-warm-up zero crosses: {R['post_up_crosses']} up / {R['post_dn_crosses']} down "
            f"over {R['years']} years')"
        ),
        md(
            f"That flat red dashed line at zero is the entire textbook trigger. The Summation Index "
            f"shoots up to its cruising range within the first few months of 2005 and — this is the "
            f"finding — **never dips back to zero again**, not once, in either direction, over "
            f"**{R['years']} years**. Not because of a bug: because the daily breadth reading has a "
            "mild positive bias (stocks close up somewhat more often than down, over a long enough "
            "window), and a running total of a slightly-biased coin flip drifts away and stays away. "
            "The textbook rule simply **can't fire** on this tape.\n\n"
            "**Second attempt: rescale the extreme-level idea instead.** Maybe the zero-cross is the "
            "wrong lens — what about *extreme* readings, the ±500 overbought/oversold levels? We "
            "adapt them honestly (a rolling one-year z-score instead of a literal \"500\", which only "
            "makes sense for a full-exchange breadth count) and test the forward return after each "
            "extreme."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ex_up = st.cross_experiment(CLOSE, SUMM, direction='up', kind='extreme', level=1.0)\n"
            "    ex_dn = st.cross_experiment(CLOSE, SUMM, direction='down', kind='extreme', level=1.0)\n"
            "    trig_up = [ex_up['by_h'][h]['gross']['mean_bps'] for h in st.HORIZONS]\n"
            "    rand_up = [ex_up['by_h'][h]['random']['mean_bps'] for h in st.HORIZONS]\n"
            "else:\n"
            "    trig_up = [R['ext_up'][h][0] for h in (5, 10, 20, 60)]\n"
            "    rand_up = [R['ext_up'][h][2] for h in (5, 10, 20, 60)]\n"
            "labels = [f'{h}d' for h in (5, 10, 20, 60)]\n"
            "x = np.arange(len(labels)); w = 0.35\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.bar(x - w/2, trig_up, width=w, color=AMBER, label='after an extreme-bullish reading')\n"
            "ax.bar(x + w/2, rand_up, width=w, color=GREY, label='after a random day')\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('forward SPY return (bps)')\n"
            "ax.set_title('Extreme-bullish breadth vs. a random day: no gap that survives the test')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            f"print('n extreme-up events: {R['ext_up_n']}')"
        ),
        md(
            "The amber and grey bars sit almost on top of each other at every horizon — an extreme "
            "breadth reading doesn't set up SPY for anything a random Tuesday wouldn't. At the "
            "60-day horizon the amber bar is actually a bit *below* the grey one — backwards from "
            "what \"breadth thrust\" folklore predicts, and about the size you'd expect just from "
            "checking 8 different horizon/direction combinations and getting unlucky once. We also "
            "re-ran the whole thing with the breadth history randomly shuffled in time (destroying "
            "any real momentum pattern while keeping the same rough distribution of daily readings) "
            f"— and the real result was unremarkable next to the shuffled ones (**{R['placebo_ext_draws']}** "
            f"shuffles, **p = {R['placebo_ext_p']:.3f}**). The specific path of breadth history isn't "
            "carrying information.\n\n"
            "**Finally, the actual trade.** Since the textbook zero-level rule can't switch at all, "
            "we built the honest adapted version: stay long SPY while breadth sits above its own "
            "recent one-year average, go flat otherwise."
        ),
        code(
            "if HAVE_REAL:\n"
            "    reg_z = st.regime_from_zscore(SUMM)\n"
            "    rb = st.regime_backtest(CLOSE, reg_z, cost_bps=5.0)\n"
            "    tc, bc = rb['timed']['cagr']*100, rb['buy_hold']['cagr']*100\n"
            "else:\n"
            "    tc, bc = R['regime_z_cagr'], R['bh_cagr']\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['the timer\\n(long/flat)', 'just buy\\nand hold'], [tc, bc], color=[AMBER, GREY], width=.55)\n"
            "for i, v in enumerate([tc, bc]): ax.annotate(f'{v:+.2f}%/yr', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('annualized return (CAGR, %)')\n"
            "ax.set_title('Timing with the Summation Index costs you return, not adds it')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'timed {tc:+.2f}%/yr vs buy&hold {bc:+.2f}%/yr')"
        ),
        md(
            f"The timer earns **{R['regime_z_cagr']:.2f}%/yr** while buy-and-hold earns "
            f"**{R['bh_cagr']:.2f}%/yr** — the strategy that's supposed to keep you on the right "
            "side of bull/bear regimes instead spends **47% of its time in cash**, missing "
            f"upside it can't get back. The shortfall (**{R['regime_z_excess_bps']:.2f} bps a day**) "
            "clears our statistical bar for \"probably not luck\" — but a second shuffle-test shows "
            "even *that* isn't really about breadth: randomly re-timed fake breadth histories "
            "produce a similarly bad result about half the time. It's not that the Summation Index "
            "actively misleads you — it's that **any** part-time market-timing rule gives back some "
            "of the market's average upward drift, and this one has nothing extra to show for the "
            "trade-off."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The zero-cross regime signal cannot fire even once on 21 years of "
            "real data; the rescaled extreme-level version doesn't beat random days; a shuffle test "
            "confirms the breadth path itself carries no information.\n"
            "- **Tradability — Mirage.** The one version of the rule that can actually trade loses "
            "to buy-and-hold, and even that loss isn't really about breadth — it's the generic cost "
            "of sometimes being out of an upward-drifting market.\n"
            "- **\"Does it confirm bull/bear turns?\" — Busted.** It can't confirm anything it "
            "never crosses."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The lesson generalizes.** Any \"just take a running total\" fix for a noisy signal "
            "carries a hidden assumption: that the underlying series has (close to) zero long-run "
            "mean. If it doesn't — and daily market breadth over a mostly-rising 20-year window "
            "doesn't — the running total isn't a smoothed signal, it's a one-way ratchet.\n"
            "- **A fairer future test** would periodically **re-base** the Summation Index (reset it "
            "toward zero on some schedule, as some practitioners actually do) rather than run one "
            "un-rebased 21-year tally — a natural next study.\n"
            "- **Sibling studies:** the [oscillator itself](../../491-mcclellan-oscillator/) (491, "
            "also None), the [bullish percent index](../../494-bullish-percent-index/) (494), the "
            "[advance-decline line](../../168-advance-decline/) (168) and "
            "[new-highs/new-lows](../../493-new-highs-new-lows/) (493) — the whole breadth-indicator "
            "family keeps landing in the same place once you insist on a fair, drift-matched "
            "comparison.\n\n"
            "*Think a re-based Summation Index would behave differently? Show it, with a "
            "drift-matched baseline and real costs, and we'll take a look.*"
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
            "# The McClellan Summation Index — a quantitative teardown 🔬\n"
            "### The causal cumsum construction · the zero-cross non-event · a rolling-z-score "
            "±500 analog · random-entry and shuffled-breadth placebos · a HAC-tested long/flat "
            "regime timer · EMA-span & basket robustness · a 20-seed synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — **the running cumulative sum of the McClellan Oscillator (491) is a regime "
            "gauge: zero-crosses confirm bull/bear phases, ±500 extremes mark exhaustion** — is "
            "tested causally, against drift-matched baselines, with two independent placebos and a "
            "faithful-engine synthetic control.\n\n"
            "> ⚠️ **Data note.** Breadth basket: SPY + the 9 classic SPDR sector ETFs (2005→2026, "
            "yfinance, cached; no survivorship — the actual traded universe throughout). Warm-up = "
            f"**{R['warmup_days']}** sessions dropped as burn-in for the un-rebased cumulative sum. "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprints `" + R["fp_spy"] + "` SPY / `" +
            R["fp_breadth"] + "` breadth).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | post-warm-up zero-crosses over {R['years']}y: "
            f"**{R['post_up_crosses']} up / {R['post_dn_crosses']} down**; extreme-cross Welch *t* "
            "vs random-day baseline indistinguishable from 0 at 7/8 horizon x direction cells "
            f"(the one exception is negative); shuffled-breadth placebo *p* = {R['placebo_ext_p']:.3f} |\n"
            f"| **Tradability** | `MIRAGE` | z-score regime timer excess vs buy&hold "
            f"**{R['regime_z_excess_bps']:.3f} bps/day**, HAC *t* = **{R['regime_z_excess_t']:.2f}**; "
            f"regime-placebo *p* = {R['placebo_regime_p']:.3f} |\n"
            "| **Confirms turns?** | `BUSTED` | the textbook rule cannot switch (0 crosses); its "
            "adapted cousin loses to buy-and-hold |\n\n"
            "> 💡 In plain words: an un-rebased running total of a mildly biased daily gauge is a "
            "ratchet, not a smoothed signal — and the ratchet doesn't even pay for itself."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $A_t$ be the daily net-advances breadth proxy and $M_t = \\mathrm{EMA}_{19}(A)_t - "
            "\\mathrm{EMA}_{39}(A)_t$ the causal McClellan Oscillator (study 491). The Summation "
            "Index is its running integral: $S_t = S_{t-1} + M_t = \\sum_{\\tau \\le t} M_\\tau$, "
            "started at $S_0 = 0$ and never re-based. The claims:\n\n"
            "- **H₁ (zero-cross regime turn).** A cross of $S_t$ through zero from below (above) "
            "confirms a new bull (bear) phase — SPY forward returns after the cross should beat a "
            "drift-matched random-entry baseline.\n"
            "- **H₂ (extreme-level exhaustion).** A cross of $S_t$ through an extreme threshold "
            "(literature: ±500/±1000, full-NYSE scale) marks an overbought/oversold turning point.\n"
            "- **H₃ (tradable regime).** A long/flat filter — long while $S_t$ (or its "
            "scale-adapted version) reads \"bull\" — beats buy-and-hold net of costs.\n\n"
            "We find **H₁ untestable on this tape** (the raw level never revisits zero after its "
            "first week — 1 raw cross ever, 0 post-warm-up, in 21.4 years), **H₂ not supported** "
            "(no correctly-signed Welch-robust edge at any of 4 horizons, either direction), and "
            "**H₃ rejected** (the only switching variant loses to buy-and-hold, HAC "
            f"*t* = {R['regime_z_excess_t']:.2f})."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Three separate statistical problems, three separate tools. **Event studies** (the "
            "zero-cross and extreme-cross triggers) are single, largely non-overlapping events: "
            "Welch *t* vs a drift-matched random-entry baseline is the planned primary, exactly as "
            "sibling 491 tests the oscillator's own trigger. The **regime timer** produces a *daily* "
            "return series with obvious serial correlation (regime persists for weeks at a time), so "
            "its excess-over-buy-and-hold uses a **Newey-West (HAC) one-sample *t*** with automatic "
            "Bartlett-kernel lag selection — Welch would understate the true standard error there. "
            "Both the event-level and regime-level tests carry an independent **shuffled-breadth "
            "placebo**: time-permute the net-advances series (destroying its autocorrelation "
            "structure while preserving its marginal), rebuild the Summation Index from scratch, "
            "re-run the exact same rule. If the real result is just noise dressed up as a pattern, "
            "shuffled worlds should reproduce it about as often as not."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Basket.** {R['basket']} (SPY + the 9 classic SPDR sector ETFs, live continuously "
            f"since 1998 — no survivorship). {R['years']} years, {R['start']} → {R['end']}, "
            f"as-of {R['asof']} (last complete month).\n"
            f"- **Indicator.** $S_t = \\mathrm{{cumsum}}(\\mathrm{{EMA}}_{{19}}(A) - "
            f"\\mathrm{{EMA}}_{{39}}(A))_t$, causal, never re-based. Warm-up = "
            f"**{R['warmup_days']}** sessions dropped as burn-in (the un-rebased sum's own settling "
            "period — a named quirk, not a free parameter tuned to the result).\n"
            "- **Zero-cross.** Event study, forward 5/10/20/60d SPY return, one execution lag "
            "(enter at next close), Welch *t* vs random-entry baseline, 1 bp one-way x 2 legs.\n"
            "- **Extreme threshold.** Causal 252-session rolling z-score, ±1σ cross as the "
            "scale-appropriate analog of the literature's ±500 level; same event-study design.\n"
            "- **Regime timer.** Long/flat, one execution lag (`pos = regime.shift(1)`), 5 bps "
            "one-way x NAV per switch, excess over buy-and-hold via HAC *t*.\n"
            "- **Controls.** Shuffled-breadth placebo (both event- and regime-level); EMA-span "
            "(10/20, 19/39, 25/50) and basket-composition (9-sector vs 5-ticker fallback) "
            "robustness; 20-seed synthetic null + planted-regime-effect positive control."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The Summation Index and the zero-cross that never happens\n\n"
            "Plot the full causal series against its own defining threshold."
        ),
        code(
            "if HAVE_REAL:\n"
            "    summ = SUMM\n"
            "    up_raw = st.zero_cross_dates(summ, 'up', apply_warmup=False)\n"
            "    dn_raw = st.zero_cross_dates(summ, 'down', apply_warmup=False)\n"
            "    up_post = st.zero_cross_dates(summ, 'up')\n"
            "    dn_post = st.zero_cross_dates(summ, 'down')\n"
            "    smin, smax = float(summ.min()), float(summ.max())\n"
            "else:\n"
            "    summ = None\n"
            "    up_raw, dn_raw, up_post, dn_post = [None]*4\n"
            "    smin, smax = R['summ_min'], R['summ_max']\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.6))\n"
            "if summ is not None:\n"
            "    ax.plot(summ.index, summ.values, color=GREY, lw=1.0)\n"
            "    w = st._warmup(len(summ))\n"
            "    ax.axvspan(summ.index[0], summ.index[w], color=AMBER, alpha=.15, label='warm-up (dropped)')\n"
            "ax.axhline(0, c=RED, lw=1.6, ls='--', label='the textbook regime-turn level')\n"
            "ax.set_ylabel('Summation Index'); ax.set_title('Range [%.1f, %.1f] — never revisits 0 after week 1' % (smin, smax))\n"
            "ax.legend(loc='lower right'); plt.tight_layout(); plt.show()\n"
            "print(f'raw crosses (no warm-up filter): {R[\"raw_up_crosses\"]} up / {R[\"raw_dn_crosses\"]} down')\n"
            "print(f'post-warm-up crosses: {R[\"post_up_crosses\"]} up / {R[\"post_dn_crosses\"]} down')"
        ),
        md(
            f"> 💡 In plain words: with a persistent positive mean in the daily breadth reading "
            "(an upward-drifting market has more up-days than down across most 10-name baskets), "
            "an **un-rebased running sum is a one-sided random walk in practice**, not a "
            "mean-reverting oscillator. It settles into a cruising band "
            f"(**[{R['summ_min']:.2f}, {R['summ_max']:.2f}]**) within months and stays there for "
            f"decades. H₁ is not \"rejected\" in the statistical sense — it is **untestable**: the "
            "event count is zero. This is reported as a structural finding of the proxy + un-rebased "
            "construction, not swept under a warm-up filter."
        ),
        md(
            "### 4b · The extreme-threshold analog vs a drift-matched random-day baseline\n\n"
            "A causal rolling 252-session z-score of $S_t$; ±1σ crosses as the scale-appropriate "
            "stand-in for the literature's ±500 level. Per-offset Welch *t* vs random days."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ex_up = st.cross_experiment(CLOSE, SUMM, direction='up', kind='extreme', level=1.0)\n"
            "    ex_dn = st.cross_experiment(CLOSE, SUMM, direction='down', kind='extreme', level=1.0)\n"
            "    hs = list(st.HORIZONS)\n"
            "    tu = [ex_up['by_h'][h]['welch_t'] for h in hs]\n"
            "    td = [ex_dn['by_h'][h]['welch_t'] for h in hs]\n"
            "else:\n"
            "    hs = [5, 10, 20, 60]\n"
            "    tu = [R['ext_up'][h][4] for h in hs]\n"
            "    td = [R['ext_dn'][h][4] for h in hs]\n"
            "x = np.arange(len(hs)); w = 0.35\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x - w/2, tu, width=w, color=AMBER, label='up-cross (extreme bullish)')\n"
            "ax.bar(x + w/2, td, width=w, color=GREY, label='down-cross (extreme bearish)')\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('Welch t (trigger vs random-day baseline)')\n"
            "ax.set_title('7 of 8 cells inside the noise band; the 8th is the wrong sign')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            f"print('extreme up-cross n={R['ext_up_n']}, down-cross n={R['ext_dn_n']}')"
        ),
        md(
            f"> 💡 In plain words: only the 60-day up-cross clears the |t| ≥ 2 bar "
            f"(**{R['ext_up'][60][4]:.2f}**) — and it is **negative**, the opposite sign of the "
            "\"bullish breadth thrust forecasts a rally\" claim. Across 8 (direction x horizon) "
            "cells, one false-positive-looking read is exactly the base rate you'd expect from "
            "chance at a two-sided ~5% level, not evidence of a real (mis-signed) effect. Combined "
            f"with the shuffled-breadth placebo (**p = {R['placebo_ext_p']:.3f}**, {R['placebo_ext_draws']} "
            "draws) — which shows the specific temporal path of breadth carries no information "
            "beyond its marginal distribution — H₂ is not supported."
        ),
        md(
            "### 4c · The regime timer — the honest, switching version of H₃\n\n"
            "Because the textbook zero-level rule cannot switch (4a), the tradable version we test "
            "is the scale-adapted rule: bull while $S_t$ sits above its own trailing 1-year mean. "
            "One execution lag; 5 bps one-way x NAV per switch; excess over buy-and-hold via HAC *t*."
        ),
        code(
            "if HAVE_REAL:\n"
            "    reg_z = st.regime_from_zscore(SUMM)\n"
            "    rb = st.regime_backtest(CLOSE, reg_z, cost_bps=5.0)\n"
            "    tc, bc = rb['timed']['cagr']*100, rb['buy_hold']['cagr']*100\n"
            "    ts, bs = rb['timed']['sharpe'], rb['buy_hold']['sharpe']\n"
            "    ex_bps, ex_t = rb['excess_mean_bps_day'], rb['excess_hac_t']\n"
            "else:\n"
            "    tc, bc = R['regime_z_cagr'], R['bh_cagr']\n"
            "    ts, bs = R['regime_z_sharpe'], R['bh_sharpe']\n"
            "    ex_bps, ex_t = R['regime_z_excess_bps'], R['regime_z_excess_t']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "a1.bar(['timer', 'buy&hold'], [tc, bc], color=[AMBER, GREY], width=.55)\n"
            "for i, v in enumerate([tc, bc]): a1.annotate(f'{v:+.2f}%', (i, v), ha='center', va='bottom')\n"
            "a1.set_ylabel('CAGR (%)'); a1.set_title('Return')\n"
            "a2.bar(['timer', 'buy&hold'], [ts, bs], color=[AMBER, GREY], width=.55)\n"
            "for i, v in enumerate([ts, bs]): a2.annotate(f'{v:.2f}', (i, v), ha='center', va='bottom')\n"
            "a2.set_title(f'Sharpe  (excess = {ex_bps:+.3f} bps/day, HAC t = {ex_t:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'CAGR: timer {tc:+.2f}% vs buy&hold {bc:+.2f}%   Sharpe: {ts:.2f} vs {bs:.2f}')"
        ),
        md(
            f"> 💡 In plain words: the timer is long only **{R['regime_z_frac_long']}%** of days "
            f"(**{R['regime_z_switches']} switches**, {R['regime_z_per_yr']:.1f}/yr) and gives back "
            "more upside than it avoids downside — excess "
            f"**{R['regime_z_excess_bps']:.3f} bps/day**, HAC *t* = **{R['regime_z_excess_t']:.2f}**, "
            "clearing the desk's significance bar for \"probably not zero\", but in the *wrong* "
            "direction for a claim that's supposed to add value. The regime-timer shuffled-breadth "
            f"placebo (**p = {R['placebo_regime_p']:.3f}**, {R['placebo_regime_draws']} draws) shows "
            "this negative excess is **not specific to real breadth** — a randomly re-timed fake "
            "breadth history produces an excess this extreme about half the time. It's the generic "
            "\"beta trap\" sibling 491 already documented for the single-day trigger: any part-time "
            "long/flat filter on an upward-drifting tape gives back drift, with or without genuine "
            "forecasting content behind the filter."
        ),
        md(
            "### 4d · Robustness — EMA spans and basket composition\n\n"
            "Does the (negative, uncertifiable-as-signal) regime-timer excess survive re-parameterization?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for fast, slow in [(10, 20), (19, 39), (25, 50)]:\n"
            "        s2 = st.summation_index(NET, fast, slow)\n"
            "        r2 = st.regime_from_zscore(s2)\n"
            "        rb2 = st.regime_backtest(CLOSE, r2, cost_bps=5.0, warmup=st._warmup(len(CLOSE), slow))\n"
            "        rows.append((fast, slow, rb2['n_switches'], rb2['excess_mean_bps_day'], rb2['excess_hac_t']))\n"
            "else:\n"
            "    rows = R['spans']\n"
            "labels = [f'{f}/{s}' for f, s, *_ in rows]\n"
            "ts = [r[4] for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.3))\n"
            "ax.bar(labels, ts, color=[RED if abs(t) >= 2 else GREY for t in ts], width=.55)\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('excess HAC t (timer - buy&hold)')\n"
            "ax.set_title('Negative and significant at every EMA-span choice')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print('fallback 5-ticker basket: {R['fallback_switches']} switches, "
            f"excess {R['fallback_excess']:+.3f} bps/day, t = {R['fallback_t']:+.2f}')"
        ),
        md(
            "> 💡 In plain words: every span pairing (10/20, the classic 19/39, 25/50) and the "
            f"5-ticker fallback basket land in the same place — a modest, HAC-significant negative "
            "excess versus buy-and-hold, and **zero** post-warm-up zero-level crosses in every "
            "variant. This is not an artefact of one parameter choice."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic mean-reverting breadth tape with a scheduled planted regime effect (knob "
            "`edge`), no network. The null (`edge=0`) is checked over **20 seeds** — never a single "
            "stream — for both the textbook level regime and the scale-adapted z-score regime."
        ),
        code(
            "null_lvl, null_z = [], []\n"
            "for s_ in range(20):\n"
            "    bars, _ = data.synthetic_panel(n_days=4000, edge=0.0, seed=666 + s_)\n"
            "    null_lvl.append(st.synthetic_regime_detect(bars, bars['net_adv'], kind='level')['excess_hac_t'])\n"
            "    null_z.append(st.synthetic_regime_detect(bars, bars['net_adv'], kind='zscore')['excess_hac_t'])\n"
            "null_lvl = np.asarray(null_lvl, dtype=float); null_z = np.asarray(null_z, dtype=float)\n"
            "bars, _ = data.synthetic_panel(n_days=4000, edge=0.6, seed=666)\n"
            "planted_lvl = st.synthetic_regime_detect(bars, bars['net_adv'], kind='level')['excess_hac_t']\n"
            "planted_z = st.synthetic_regime_detect(bars, bars['net_adv'], kind='zscore')['excess_hac_t']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.1, .1, 20), null_lvl, color=GREY, s=36, label='null, level regime')\n"
            "ax.scatter(np.ones(20) + np.linspace(-.1, .1, 20), null_z, color=AMBER, s=36, label='null, z-score regime')\n"
            "ax.scatter([2], [planted_lvl], color=RED, s=90, zorder=5, label='planted, level regime')\n"
            "ax.scatter([3], [planted_z], color=RED, marker='D', s=90, zorder=5, label='planted, z-score regime')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1, 2, 3]); ax.set_xticklabels(['null\\n(level)', 'null\\n(zscore)', 'planted\\n(level)', 'planted\\n(zscore)'])\n"
            "ax.set_ylabel('excess HAC t (timer - buy&hold)')\n"
            "ax.set_title('Control: no null fires; a planted regime effect lights up both ways')\n"
            "ax.legend(fontsize=8); plt.tight_layout(); plt.show()\n"
            "print(f'null level: mean {null_lvl.mean():+.2f} (sd {null_lvl.std(ddof=1):.2f}), fires {(abs(null_lvl)>=2).sum()}/20')\n"
            "print(f'null zscore: mean {null_z.mean():+.2f} (sd {null_z.std(ddof=1):.2f}), fires {(abs(null_z)>=2).sum()}/20')\n"
            "print(f'planted: level t={planted_lvl:+.2f}  zscore t={planted_z:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds neither regime rule ever fires "
            f"(**0/20** for both), and a planted 0.6-strength regime effect lights up sharply "
            f"(level *t* = **{R['syn_planted_lvl_t']:.2f}**, z-score *t* = **{R['syn_planted_z_t']:.2f}**). "
            "The harness is unbiased — the real-tape's structural non-event (4a) and null-ish "
            "extreme-cross result (4b) are genuine \"nothing there\" findings, not a broken detector."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the zero-cross claim is untestable on this tape "
            f"({R['post_up_crosses']} post-warm-up crosses in {R['years']} years, robust to EMA "
            "span and basket choice); the extreme-threshold analog shows no correctly-signed, "
            "Welch-robust edge at 4 horizons in either direction "
            f"(shuffled-breadth placebo *p* = {R['placebo_ext_p']:.3f}).\n"
            f"- **Tradability `MIRAGE`** — the only switching regime rule underperforms buy-and-hold "
            f"by **{R['regime_z_excess_bps']:.3f} bps/day** (HAC *t* = {R['regime_z_excess_t']:.2f}), "
            f"robust across spans/baskets — and the regime-placebo (*p* = {R['placebo_regime_p']:.3f}) "
            "shows even that loss is generic drift give-back, not a real (mis-signed) forecast.\n"
            "- **\"Confirms bull/bear turns?\" `BUSTED`** — the level that's supposed to confirm "
            "regimes never crosses; its rescaled cousin doesn't beat random days; the resulting "
            "timer loses to holding the index."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The general lesson: accumulation isn't free smoothing.** A running sum only "
            "behaves like a mean-reverting, information-carrying level if its input has (close to) "
            "zero long-run mean. Any breadth or sentiment gauge with a persistent directional bias "
            "will turn a cumulative-sum \"fix\" into a ratchet — worth checking *before* building "
            "the indicator, not after finding it never crosses its own trigger level.\n"
            "- **A fairer next test:** periodically re-base the Summation Index (some practitioners "
            "reset it toward zero on a schedule) and re-test the zero-cross with that construction "
            "— a natural, well-defined follow-up.\n"
            "- **Dedup map:** [491-mcclellan-oscillator](../../491-mcclellan-oscillator/) (the "
            "single-day trigger this study's integral is built from — also None), "
            "[494-bullish-percent-index](../../494-bullish-percent-index/) (diffusion, not "
            "cumulative-sum), [168-advance-decline](../../168-advance-decline/) (raw cumulative "
            "line as a divergence signal), [493-new-highs-new-lows](../../493-new-highs-new-lows/) "
            "(extremes-based thrust). The whole breadth-indicator family keeps landing on `None` "
            "once a drift-matched baseline is imposed.\n\n"
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
