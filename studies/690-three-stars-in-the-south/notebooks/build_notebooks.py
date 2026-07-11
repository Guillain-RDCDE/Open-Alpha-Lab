"""Generate the two narrative notebooks for Study 690 (Three Stars in the South).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached daily
tapes under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic positive control runs anywhere with no
network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily OHLCV,
# SPY + 60 large-caps, ~25 years, 2001-07-10 -> 2026-06-30; downtrend-matched base rate).
R = dict(
    n_names=61, years=25.0, total_bars=383_080,
    n_loose=363, n_strict=5, bonferroni_crit=2.50,
    loose_rate_txt="once per ~4.2 ticker-years", strict_rate_txt="once per ~305 ticker-years",
    # loose cut: horizon -> (star_bps, win_pct, base_bps, delta_bps, welch_t, placebo_p, net_bps)
    loose={
        1: (21.9, 55.1, 2.4, 19.5, 2.26, 0.015, 11.9),
        5: (43.3, 54.8, 28.8, 14.5, 0.74, 0.263, 33.3),
        10: (69.2, 57.6, 57.2, 12.0, 0.45, 0.349, 59.2),
        20: (90.3, 55.9, 119.4, -29.1, -0.80, 0.768, 80.3),
    },
    # strict cut: horizon -> (star_bps, win_pct, delta_bps, placebo_p, net_bps) -- n=5, no t-stat
    strict={
        1: (37.6, 60.0, 35.2, 0.302, 27.6),
        5: (-57.9, 40.0, -86.7, 0.719, -67.9),
        10: (27.8, 60.0, -29.4, 0.555, 17.8),
        20: (133.0, 40.0, 13.6, 0.486, 123.0),
    },
    strict_events=[
        ("TXN", "2006-10-18", "2006-10-20", -303.3),
        ("CAT", "2021-06-14", "2021-06-16", -256.3),
        ("BDX", "2016-09-16", "2016-09-20", -210.2),
        ("AXP", "2019-03-27", "2019-03-29", 624.1),
        ("COST", "2024-10-24", "2024-10-28", 810.9),
    ],
    syn_null_mean=-0.34, syn_null_sd=1.08, syn_null_fire=2, syn_null_seeds=20,
    syn_e02_t=7.44, syn_e02_delta=156.0, syn_e04_t=16.16, syn_e04_delta=346.9,
    fp_spy="27b74051eb44", fp_panel="16a85c8367ed",
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Beats_a_downtrend_base_rate%3F: Mixed](https://img.shields.io/badge/Beats_a_downtrend_base_rate%3F-Mixed-8b949e?style=flat-square)\n\n"
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

from three_stars_in_the_south import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PANEL = data.load_real()
else:
    PANEL = None
print("real cache present:", HAVE_REAL, "| basket size:", len(data.BASKET))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The rarest bullish candle in the book, put to the test ⭐⭐⭐\n"
            "### Three stars in the south — three shrinking red candles that are\n"
            "### supposed to say \"the selling is over\"\n\n"
            + BADGES +
            "Deep in a downtrend, the story goes, you sometimes see something odd: three "
            "red (down) candles in a row, but each one is **smaller** than the last, and "
            "each one's low sits **higher** than the one before. Sellers are still "
            "pushing the stock down — but with visibly less and less conviction each day, "
            "like a wave running out of energy on the beach. The third candle barely has "
            "a body at all. Candlestick lore calls this **three stars in the south**, and "
            "claims it marks the exact moment selling pressure runs out.\n\n"
            "That's the claim we test: *does the shrinking, rising-low shape itself signal "
            "exhaustion?* Spoiler up front — this is the rarest pattern this desk has ever "
            "measured, and the honest answer is that there simply isn't enough real-world "
            "evidence to say yes.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the cost "
            "math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** SPY + 60 long-listed US large-caps, ~25 years of daily "
            "bars (yfinance). Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does a shrinking, rising-low three-star block predict a bounce? | "
            f"**Not provably.** The plain (loose) reading fires **{R['n_loose']}** times "
            "across 61 stocks and 25 years — enough to test — and the 1-day reaction "
            "looks promising (+19.5 bps more than a normal downtrend bar) but the effect "
            "**flips negative** by 20 days, and none of it survives the correction for "
            "testing four different holding periods at once. |\n"
            f"| What about the *real*, strict version from the textbook? | It happens "
            f"**{R['n_strict']} times** in 25 years across 61 stocks — {R['strict_rate_txt']}. "
            "That's too few data points for any statistic to mean anything. |\n"
            "| So which is it — real or fake? | **We genuinely can't tell**, and that's "
            "the honest finding. Two of the five real occurrences led to sharp bounces "
            "(+624 bps, +811 bps); three didn't (the stock kept falling). A coin flip "
            "would look similar. |\n"
            "| Can you trade it? | **No.** Even setting statistics aside, a pattern that "
            "shows up once every 305 ticker-years per stock has no capacity to matter. |\n\n"
            "> Not every piece of market folklore gets a clean \"busted\" — some are simply "
            "too rare to ever put on trial properly, and that's worth saying plainly."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"After a sustained decline, watch for three consecutive black candles "
            "where each one is smaller than the last and each low is higher than the "
            "one before. The third candle is barely more than a sliver. This tells you "
            "the bears are exhausted — buy.\"*\n\n"
            "Unlike most reversal patterns, three stars in the south doesn't even need a "
            "green candle to confirm it — the shrinking, rising-low shape of three *red* "
            "candles is claimed to be the signal all by itself. Steve Nison, who "
            "introduced Western traders to Japanese candlesticks, includes it in his "
            "canon — but even Thomas Bulkowski, who has statistically tested hundreds of "
            "chart patterns on decades of data, admits he couldn't find enough real "
            "examples to rank this one with confidence. That admission is itself a "
            "finding worth taking seriously before we even open the data."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real, this would be a genuinely elegant signal: no waiting for a "
            "confirming green candle, no gap required — just read the *shape* of the "
            "selling itself as it happens, and get in a day early. That's valuable if it "
            "works. But it's also the kind of claim that's easy to believe once you've "
            "seen a few pretty chart examples and easy to over-trust because it *sounds* "
            "mechanically sensible (\"of course shrinking sell pressure means selling is "
            "ending\") without ever being tested on enough real cases to know."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "- **The shape, two ways.** A **loose** reading (three shrinking, rising-low "
            "black candles in a downtrend) that fires often enough to test, and a "
            "**strict**, textbook-faithful reading (+ a hammer-like first candle, no gap "
            "on the second, a near-doji third that never undercuts the second) that is "
            "the actual literature claim.\n"
            "- **The comparison.** The forward return after the pattern vs. the forward "
            "return from buying **any** bar that's already in a similar downtrend — "
            "isolating the pattern's own information from plain \"buy the dip.\"\n"
            "- **The luck check.** Draw random downtrend days instead of real "
            "three-star days — how often does chance alone produce as good a result?\n"
            "- **The honesty rule.** If either cut fires too rarely to compute a "
            "meaningful statistic, we say so instead of dressing up a handful of points "
            "as a verdict."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the loose cut.** Forward return after the pattern vs. the "
            "downtrend base rate, at four holding periods."
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(PANEL)\n"
            "    loose = {h: (res['per_horizon'][h]['star']['mean_bps'],\n"
            "                 res['per_horizon'][h]['base']['mean_bps'],\n"
            "                 res['per_horizon'][h]['welch_t']) for h in st.HORIZONS}\n"
            "else:\n"
            "    loose = {h: (R['loose'][h][0], R['loose'][h][2], R['loose'][h][4]) for h in (1,5,10,20)}\n"
            "hs = sorted(loose)\n"
            "star_v = [loose[h][0] for h in hs]\n"
            "base_v = [loose[h][1] for h in hs]\n"
            "x = np.arange(len(hs)); w = 0.35\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.6))\n"
            "ax.bar(x - w/2, star_v, w, color=RED, label='after a three-star block')\n"
            "ax.bar(x + w/2, base_v, w, color=GREY, label='downtrend base rate')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean forward return (bps)')\n"
            "ax.set_title('Loose cut: ahead at first, behind by 20 days')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print({h: (round(loose[h][0],1), round(loose[h][1],1), round(loose[h][2],2)) for h in hs})"
        ),
        md(
            f"At 1 day the pattern looks promising: **{R['loose'][1][0]:+.1f} bps** vs "
            f"**{R['loose'][1][2]:+.1f} bps** for the base rate — a real-looking gap "
            f"(Welch *t* = {R['loose'][1][4]:.2f}). But look at 20 days: "
            f"**{R['loose'][20][0]:+.1f} bps** for the pattern vs "
            f"**{R['loose'][20][2]:+.1f} bps** for the base rate — the pattern is now "
            "*behind*. A genuine exhaustion signal shouldn't reverse itself as you hold "
            "longer. Once you account for testing four different holding periods at "
            "once (the statistical equivalent of getting four lottery tickets instead of "
            "one), **nothing here clears the bar**.\n\n"
            "**Now the real thing — the strict, textbook-faithful cut.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ev = st.pool_events(PANEL)\n"
            "    strict_ev = ev[ev['strict']].copy()\n"
            "    recs = []\n"
            "    for _, row in strict_ev.iterrows():\n"
            "        tkr = row['ticker']; bars = PANEL[tkr]; pos = int(row['pos'])\n"
            "        d0 = bars.index[pos - 2].date(); d1 = bars.index[pos].date()\n"
            "        recs.append((tkr, str(d0), str(d1), row['ret_20'] * 1e4))\n"
            "    recs.sort(key=lambda r: r[3])\n"
            "else:\n"
            "    recs = sorted(R['strict_events'], key=lambda r: r[3])\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "labels = [r[0] for r in recs]; vals = [r[3] for r in recs]\n"
            "ax.barh(labels, vals, color=[RED if v < 0 else GREEN for v in vals])\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('20-day forward return (bps)')\n"
            "ax.set_title(f'Every single strict three-star event this desk found (n={len(recs)})')\n"
            "plt.tight_layout(); plt.show()\n"
            "for r in recs: print(f'{r[0]}: {r[1]} -> {r[2]}   20d {r[3]:+.1f} bps')"
        ),
        md(
            f"This is not a summary statistic — this chart shows **every single** "
            f"literature-faithful three-star block found across 61 stocks and 25 years: "
            f"**{R['n_strict']}**. Two led to sharp bounces (AXP +624 bps, COST +811 bps); "
            "three did not (TXN, CAT, BDX kept falling). With five data points, that "
            "split is exactly what you'd expect from pure chance. We can name the pattern, "
            "chart it beautifully, and still honestly say: **we do not have enough real "
            "occurrences to know if it works.**\n\n"
            "**Finally, does the harness even work?** We plant a fake three-star effect "
            "in simulated data and confirm the same machinery can find it when it's "
            "really there."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    d, truth = data.synthetic_panel(edge=0.0, seed=690 + s_)\n"
            "    r = st.synthetic_detect(d, horizon=20, seed=690 + s_, active_masks=truth['active_masks'])\n"
            "    if r['welch_t'] is not None:\n"
            "        null_ts.append(r['welch_t'])\n"
            "d2, truth2 = data.synthetic_panel(edge=0.04, seed=690)\n"
            "planted = st.synthetic_detect(d2, horizon=20, seed=690, active_masks=truth2['active_masks'])\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.3))\n"
            "ax.scatter(np.zeros(len(null_ts)) + np.linspace(-.12,.12,len(null_ts)), null_ts,\n"
            "           color=GREY, s=40, label='null worlds (no real effect), 20 seeds')\n"
            "ax.scatter([1], [planted['welch_t']], color=RED, s=90, zorder=5, label='a real, planted effect')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['no real effect', 'planted effect'])\n"
            "ax.set_ylabel('detector score (Welch t)')\n"
            "ax.set_title('Proof the tool works: quiet on nothing, loud on something real')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'no-effect worlds: mean {np.mean(null_ts):+.2f}  |  planted-effect world: {planted[\"welch_t\"]:+.2f}')"
        ),
        md(
            "When we simulate 20 fake worlds with **no** real three-star effect, the "
            "detector correctly stays quiet (scores clustered near zero). When we plant a "
            "genuine effect, it lights up unmistakably. So the tool isn't broken — the "
            "real market simply hasn't handed us enough genuine three-star events to say "
            "either way."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The loose cut looks interesting for exactly one day "
            "before reversing, and nothing survives the correction for testing four "
            "holding periods. The strict, textbook cut has only 5 real occurrences in 25 "
            "years — too few for any statistic to mean anything.\n"
            "- **Tradability — Mirage.** Once per 305 ticker-years for the real version "
            "of the pattern. That's not a strategy, it's a museum piece.\n"
            "- **\"Beats a downtrend base rate?\" — Mixed.** Points the right way at "
            "short horizons before flipping; the five real strict events split roughly "
            "50/50 between real bounces and continued declines — a coin flip's worth of "
            "evidence either way."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Rarity itself is the finding.** When a respected chart-pattern "
            "statistician like Bulkowski says he couldn't find enough examples to rank a "
            "pattern, that's worth believing *before* running a single test — and this "
            "study's real-tape count (5 events) backs him up precisely.\n"
            "- **The nearby, testable cousins already exist on this desk:** "
            "[three white soldiers](../../187-three-soldiers/) (the bullish continuation "
            "mirror), [three black crows](../../408-three-black-crows/) (the bearish "
            "momentum cousin that shares the \"three black candles\" costume but claims "
            "the opposite thing), and [ladder bottom](../../687-ladder-bottom/) (a related "
            "5-candle bottoming pattern with enough real occurrences to actually test).\n\n"
            "*Have a bigger basket, or more history? The full detector and event list are "
            "reproducible from [examples/verify.py](../examples/verify.py) — five events "
            "is a starting point, not a final word.*"
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
            "# Three Stars in the South — a quantitative teardown 🔬\n"
            "### The loose-vs-strict detector split · the downtrend-matched base rate · "
            "Bonferroni across 4 horizons · a label-shuffle placebo · costs · a synthetic "
            "planted-bounce control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The claim — **a shrinking-body, rising-low sequence of three black candles "
            "signals downtrend exhaustion, no confirming candle required** — is candlestick "
            "lore's rarest bullish reversal claim, and even its cataloguer (Bulkowski) has "
            "publicly doubted whether enough real occurrences exist to test it. The job here "
            "is to measure exactly how rare it is, run the honest test on what exists, and "
            "say plainly where the evidence runs out.\n\n"
            "> ⚠️ **Data note.** SPY + 60 long-listed US large-caps, daily OHLCV "
            f"(~{R['years']:.1f} years, yfinance), cache-first. No survivorship on the "
            "Signal axis in the sense that matters here — a single-pattern event study, so "
            "a fixed liquid basket affects which *names* contribute events, not the "
            "star-vs-base-rate direction. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprints `" + R["fp_spy"] +
            "` sample / `" + R["fp_panel"] + "` panel).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | loose 1d Welch **t = {R['loose'][1][4]:.2f}** "
            f"(placebo *p* = {R['loose'][1][5]:.3f}) — nominal, short of Bonferroni "
            f"{R['bonferroni_crit']:.2f}; sign reverses at 20d (**t = {R['loose'][20][4]:.2f}**). "
            f"Strict cut n = {R['n_strict']} < `MIN_N_FOR_TEST` -- no *t*-stat computable |\n"
            f"| **Tradability** | `MIRAGE` | strict cut fires {R['strict_rate_txt']} |\n"
            f"| **Beats base rate?** | `MIXED` | loose cut right-signed at 1/5/10d, "
            "flips at 20d; strict cut's 5 events split 2 bounces / 3 non-bounces |\n\n"
            "> 💡 In plain words: this pattern is so rare that the desk's own multiple-"
            "testing and minimum-sample-size rules — designed to stop us from over-"
            "claiming — are themselves the headline finding."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let bars $t-2, t-1, t$ be three consecutive black (bearish, close < open) "
            "candles. The claim requires, at minimum:\n\n"
            "- **H₁ (shrinking, rising-low shape).** range$(t-2) >$ range$(t-1) >$ "
            "range$(t)$ **and** low$(t-2) <$ low$(t-1) <$ low$(t)$, inside a genuine "
            "prior downtrend (close$(t-2) <$ close$(t-2-L)$ for lookback $L$).\n"
            "- **H₂ (literature geometry).** The loose shape *plus* a real lower shadow "
            "on star 1 (a hammer — selling met by intraday buying), star 2 opening "
            "**inside** star 1's real body (no gap down), and star 3 a **near-marubozu** "
            "(small shadows both sides) that never breaks star 2's low.\n"
            "- **H₃ (exhaustion, not calm).** A confirmed block, entered long at the next "
            "open, beats the unconditional downtrend base rate net of costs — and, "
            "because no confirming bullish candle is required by the claim, the edge "
            "should show up **immediately**, not just eventually.\n\n"
            "We find **H₁ fires enough to test (n = 363)** but **does not clear "
            "certification at any horizon**; **H₂ is so rare (n = 5) that it cannot be "
            "tested at all** by the desk's own honesty rule; **H₃ is unsupported** — the "
            "loose cut's best horizon is 1 day, and it reverses by 20."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Three-star events are treated as a **directional long bet vs. the "
            "unconditional downtrend base rate** (never a one-sample test against zero, "
            "which would just measure the basket's unconditional drift). Because four "
            "horizons (1/5/10/20d) are tested simultaneously, we apply a **Bonferroni "
            "correction** (k = 4, critical \\|*t*\\| ≥ "
            f"{R['bonferroni_crit']:.2f}) rather than a naive \\|*t*\\| ≥ 2. Below "
            "`MIN_N_FOR_TEST` = 8 pooled events, no *t*-statistic is computed at all — a "
            "*t* on 5 points is decoration, not evidence, and the honest response is 'too "
            "few to test,' the same discipline sibling studies 685 (tri-star doji) and 687 "
            "(ladder bottom) apply to their own rare-pattern cuts."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Basket.** SPY + 60 long-listed US large-caps, ~{R['years']:.1f} years "
            f"each, {R['total_bars']:,} total daily bars. As-of 2026-06-30 (last complete "
            "month).\n"
            "- **Detector.** `three_stars_flags` (loose) and `strict_three_stars_flags` "
            "(strict), confirmed on the close of the 3rd star.\n"
            "- **Base rate.** The same long bet on every bar already sitting in a "
            "matching downtrend, whether or not the three-star shape fired.\n"
            "- **Execution.** Entered at the **next open** (one documented lag), held a "
            "fixed horizon; 5 bps one-way × 2 round trip; long-only, no borrow.\n"
            "- **Arbiters.** Welch *t* of star mean vs base-rate mean (decisive), a "
            "2,000-draw label-shuffle placebo, Bonferroni across 4 horizons.\n"
            "- **Control.** Synthetic panel, forced blocks planted only where the "
            "underlying random walk is already in a downtrend on its own; the null must "
            "not fire across 20 seeds, and a planted bounce must light up cleanly."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · How rare is the real thing?\n\n"
            "Loose vs strict candidate counts, pooled across the basket."
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(PANEL)\n"
            "    n_loose, n_strict = res['n_loose'], res['n_strict']\n"
            "    yrs, nnames = res['years'], res['n_names']\n"
            "else:\n"
            "    n_loose, n_strict, yrs, nnames = R['n_loose'], R['n_strict'], R['years'], R['n_names']\n"
            "loose_rate = n_loose / (nnames * yrs)\n"
            "strict_rate = n_strict / (nnames * yrs)\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.2))\n"
            "ax.bar(['loose', 'strict\\n(literature-closer)'], [n_loose, n_strict], color=[AMBER, RED], width=.55)\n"
            "for i, v in enumerate([n_loose, n_strict]):\n"
            "    ax.annotate(f'n = {v}', (i, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('pooled event count')\n"
            "ax.set_title(f'{nnames} names x {yrs:.1f}y: the strict geometry collapses the pool '\n"
            "             f'{n_loose/max(n_strict,1):.0f}x')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'loose: {n_loose} events ({loose_rate:.3f}/ticker-year)')\n"
            "print(f'strict: {n_strict} events ({strict_rate:.4f}/ticker-year, ~once per '\n"
            "      f'{1/strict_rate:.0f} ticker-years)')"
        ),
        md(
            f"> 💡 In plain words: the strict, hammer/no-gap/marubozu geometry collapses "
            f"the candidate pool from {R['n_loose']} to just **{R['n_strict']}** — "
            f"{R['strict_rate_txt']}. This directly reproduces Bulkowski's own published "
            "caution about insufficient sample size for this exact pattern."
        ),
        md(
            "### 4b · The loose cut across horizons — vs the downtrend-matched base rate\n\n"
            f"Bonferroni-corrected critical \\|*t*\\| for 4 horizons: "
            f"**{R['bonferroni_crit']:.2f}**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = {h: res['per_horizon'][h] for h in st.HORIZONS}\n"
            "    ts = [rows[h]['welch_t'] for h in st.HORIZONS]\n"
            "    deltas = [rows[h]['delta_bps'] for h in st.HORIZONS]\n"
            "else:\n"
            "    ts = [R['loose'][h][4] for h in (1,5,10,20)]\n"
            "    deltas = [R['loose'][h][3] for h in (1,5,10,20)]\n"
            "hs = [1,5,10,20]\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.0, 6.2), sharex=True,\n"
            "                             gridspec_kw={'height_ratios': [1, 1]})\n"
            "a1.bar([str(h) for h in hs], deltas, color=[RED if d<0 else AMBER for d in deltas], width=.55)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('delta vs base rate (bps)')\n"
            "a1.set_title('Loose cut: positive short-horizon delta reverses by 20 days')\n"
            "a2.bar([str(h) for h in hs], ts, color=[RED if abs(t)>=R['bonferroni_crit'] else GREY for t in ts], width=.55)\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.axhline(-R['bonferroni_crit'], ls='--', c=RED, lw=1)\n"
            "a2.axhline(R['bonferroni_crit'], ls='--', c=RED, lw=1)\n"
            "a2.set_ylabel('Welch t'); a2.set_xlabel('horizon (days)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print({h: (round(d,1), round(t,2)) for h, d, t in zip(hs, deltas, ts)})"
        ),
        md(
            f"> 💡 In plain words: the 1-day delta (**{R['loose'][1][3]:+.1f} bps**, "
            f"*t* = {R['loose'][1][4]:.2f}, placebo *p* = {R['loose'][1][5]:.3f}) is the "
            "closest this study gets to a signal — real-looking on its own, but short of "
            f"the Bonferroni bar ({R['bonferroni_crit']:.2f}), and it does not persist: "
            f"by 20 days the delta is **{R['loose'][20][3]:+.1f} bps** "
            f"(*t* = {R['loose'][20][4]:.2f}) — the star events now trail the base rate. A "
            "genuine exhaustion signal should not reverse sign as the holding period "
            "lengthens; a nominal short-horizon blip that a properly-corrected long-"
            "horizon test contradicts is the textbook signature of noise, not signal."
        ),
        md(
            "### 4c · The strict cut — all five events, named\n\n"
            "Below `MIN_N_FOR_TEST` = 8, no *t*-statistic is computed. The raw numbers "
            "are shown for full transparency, not as evidence."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ev = st.pool_events(PANEL)\n"
            "    strict_ev = ev[ev['strict']].copy()\n"
            "    recs = []\n"
            "    for _, row in strict_ev.iterrows():\n"
            "        tkr = row['ticker']; bars = PANEL[tkr]; pos = int(row['pos'])\n"
            "        d0 = bars.index[pos - 2].date(); d1 = bars.index[pos].date()\n"
            "        recs.append((tkr, str(d0), str(d1), row['ret_20'] * 1e4))\n"
            "    recs.sort(key=lambda r: r[3])\n"
            "else:\n"
            "    recs = sorted(R['strict_events'], key=lambda r: r[3])\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.2))\n"
            "labels = [f'{r[0]}\\n{r[1]}' for r in recs]; vals = [r[3] for r in recs]\n"
            "ax.barh(labels, vals, color=[RED if v < 0 else GREEN for v in vals])\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('20-day forward return (bps)')\n"
            "ax.set_title(f'n = {len(recs)}: every strict three-star event on the whole basket')\n"
            "plt.tight_layout(); plt.show()\n"
            "wins = sum(1 for r in recs if r[3] > 0)\n"
            "print(f'{wins}/{len(recs)} positive at 20d')\n"
            "for r in recs: print(f'{r[0]}: {r[1]} -> {r[2]}   20d {r[3]:+.1f} bps')"
        ),
        md(
            f"> 💡 In plain words: {sum(1 for r in R['strict_events'] if r[3] > 0)} of "
            f"{R['n_strict']} strict events show a positive 20-day return — a coin flip's "
            "worth of evidence in a sample this size. Naming all five (not just the "
            "winners) is the point: a chart-pattern book showing only AXP and COST would "
            "look like proof; showing all five looks like what it is — too few points to "
            "conclude anything."
        ),
        md(
            "### 4d · Costs — net of a 5 bps one-way round trip (10 bps total)"
        ),
        code(
            "if HAVE_REAL:\n"
            "    loose_gross = [res['per_horizon'][h]['star']['mean_bps'] for h in st.HORIZONS]\n"
            "    loose_net = [res['per_horizon'][h]['star']['net_bps'] for h in st.HORIZONS]\n"
            "else:\n"
            "    loose_gross = [R['loose'][h][0] for h in (1,5,10,20)]\n"
            "    loose_net = [R['loose'][h][6] for h in (1,5,10,20)]\n"
            "hs = [1,5,10,20]\n"
            "x = np.arange(len(hs)); w = .35\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.2))\n"
            "ax.bar(x - w/2, loose_gross, w, color=GREY, label='gross')\n"
            "ax.bar(x + w/2, loose_net, w, color=AMBER, label='net (10 bps round trip)')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('bps')\n"
            "ax.set_title('Costs are not the binding constraint here')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print({h: (round(g,1), round(n,1)) for h, g, n in zip(hs, loose_gross, loose_net)})"
        ),
        md(
            "> 💡 In plain words: most cells stay net-positive of a 10 bps round trip — "
            "costs are not what kills this pattern. What kills it is that the point "
            "estimate is not statistically certified at any horizon (loose) and that the "
            "strict, literature-faithful cut fires far too rarely to matter even where the "
            "raw number looks good."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic panel: forced 3-bar shrinking/rising-low blocks planted **only** "
            "where the underlying random walk is already in a downtrend on its own (no "
            "artificial drift injected elsewhere), TUNABLE planted post-block bounce. The "
            "null (`edge = 0`) is checked over **20 seeds** — never a single stream."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    d, truth = data.synthetic_panel(edge=0.0, seed=690 + s_)\n"
            "    r = st.synthetic_detect(d, horizon=20, seed=690 + s_, active_masks=truth['active_masks'])\n"
            "    if r['welch_t'] is not None:\n"
            "        null_ts.append(r['welch_t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "d2, truth2 = data.synthetic_panel(edge=0.04, seed=690)\n"
            "planted = st.synthetic_detect(d2, horizon=20, seed=690, active_masks=truth2['active_masks'])\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(len(null_ts)) + np.linspace(-.12,.12,len(null_ts)), null_ts, color=GREY, s=40,\n"
            "           label=f'null worlds (edge=0), {len(null_ts)} seeds')\n"
            "ax.scatter([1], [planted['welch_t']], color=RED, s=90, zorder=5,\n"
            "           label='planted edge = 0.04')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('Welch t (star vs base rate)')\n"
            "ax.set_title('Control: the null stays small; a planted bounce lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(np.abs(null_ts)>=2).sum()}/{len(null_ts)} seeds  |  '\n"
            "      f'planted t = {planted[\"welch_t\"]:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector averages "
            f"t = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}), firing on "
            f"{R['syn_null_fire']}/{R['syn_null_seeds']} seeds — in the same ballpark as "
            "sibling 685's own tri-star-doji null (a known, named residual of forcing "
            "low-variance engineered days into a random walk, not a claim about the "
            f"market). A planted edge of 0.02 reads t = {R['syn_e02_t']:.2f} and 0.04 "
            f"reads t = {R['syn_e04_t']:.2f} — the machinery finds a real, planted effect "
            "cleanly. *(A faithful-engine / power check only — never cited in support of "
            "the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the loose cut (n = {R['n_loose']}) comes closest at 1 "
            f"day (Welch t = {R['loose'][1][4]:.2f}, placebo p = {R['loose'][1][5]:.3f}) "
            f"but falls short of the Bonferroni bar ({R['bonferroni_crit']:.2f}) and "
            f"reverses sign by 20 days (t = {R['loose'][20][4]:.2f}). The strict cut "
            f"(n = {R['n_strict']}) is below `MIN_N_FOR_TEST` — no *t*-statistic can "
            "honestly be computed, and its raw deltas flip sign at every horizon.\n"
            f"- **Tradability `MIRAGE`** — {R['strict_rate_txt']} for the literature-"
            "faithful pattern. Costs are not the binding constraint (most cells net-"
            "positive of a 10 bps round trip); certification and event frequency both "
            "are.\n"
            "- **Beats a downtrend base rate? `MIXED`** — right-signed at short horizons, "
            "reverses at 20 days on the loose cut; the strict cut's five events split "
            "2 real bounces / 3 continued declines. Not busted, not confirmed — genuinely "
            "too thin to resolve further on this basket."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Rarity as a finding.** Bulkowski's own published caution about "
            "insufficient sample size for this exact pattern is directly reproduced by "
            "this study's real-tape count (5 strict events across 61 names, 25 years) — "
            "worth taking as seriously as any *t*-statistic.\n"
            "- **The next honest step** isn't a bigger *t*-stat on the same data — it's "
            "more data: a wider international basket, or a longer intraday history where "
            "the geometry can be checked bar-by-bar rather than just OHLC, might eventually "
            "clear `MIN_N_FOR_TEST` on the strict cut. Until then, five points is five "
            "points.\n"
            "- **Dedup map:** [187-three-soldiers](../../187-three-soldiers/) (the bullish "
            "*continuation* mirror), [408-three-black-crows](../../408-three-black-crows/) "
            "(three falling candles read *bearish*, no shrinking/rising-low requirement — "
            "the near-opposite claim in the same costume), "
            "[687-ladder-bottom](../../687-ladder-bottom/) (the desk's other black-candle-"
            "into-reversal bottom, five bars with a separate confirming candle vs this "
            "study's three).\n\n"
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
