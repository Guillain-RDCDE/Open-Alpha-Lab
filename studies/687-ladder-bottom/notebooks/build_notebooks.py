"""Generate the two narrative notebooks for Study 687 (Ladder Bottom).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached daily tapes
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily, SPY + 60
# long-listed US large-caps, ~25.0 years each, 383,080 total bars, as-of 2026-06-30).
R = dict(
    n_names=61, years=25.0, total_bars=383080, asof="2026-06-30",
    fp_spy="85f86a841de4", fp_panel="9693c3ea7807",
    n_loose=2543, n_strict=81, bonferroni_crit=2.50,
    # loose cut: horizon -> (n, ladder_bps, win_pct, base_bps, delta_bps, welch_t, placebo_p, net_bps)
    loose={1: (2543, -3.8, 47.2, 2.7, -6.5, -1.87, 0.981, -13.8),
           5: (2543, 46.6, 56.2, 27.5, 19.1, 2.24, 0.009, 36.6),
           10: (2543, 54.5, 56.2, 55.5, -1.0, -0.08, 0.528, 44.5),
           20: (2543, 154.7, 59.9, 120.0, 34.8, 2.12, 0.011, 144.7)},
    # strict cut: horizon -> (n, ladder_bps, win_pct, delta_bps, welch_t, placebo_p, net_bps)
    strict={1: (81, -13.2, 43.2, -15.9, -0.71, 0.805, -23.2),
            5: (81, 107.8, 55.6, 80.3, 1.52, 0.045, 97.8),
            10: (81, 129.5, 53.1, 74.0, 1.11, 0.109, 119.5),
            20: (81, 339.8, 64.2, 219.8, 2.30, 0.006, 329.8)},
    strict_events=[("NKE", "2026-03-10", "2026-03-16", -2028.9),
                   ("LMT", "2003-01-16", "2003-01-23", -1233.2),
                   ("BA", "2009-02-26", "2009-03-04", 2128.7),
                   ("KMB", "2020-03-18", "2020-03-24", 2143.9),
                   ("MS", "2002-10-02", "2002-10-08", 3973.3)],
    # synthetic control
    syn_null_mean=+0.02, syn_null_sd=0.97, syn_null_fire=0, syn_null_seeds=20,
    syn_edge_002_t=+2.85, syn_edge_002_delta=33.8,
    syn_edge_004_t=+4.02, syn_edge_004_delta=49.2,
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
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from ladder_bottom import data, strategy as st

HAVE_REAL = data.have_real()
PANEL = data.load_real() if HAVE_REAL else None
print("real cache present:", HAVE_REAL, "| basket size:", len(data.BASKET))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# A ladder down, then a leap back up 🪜\n"
            "### The ladder bottom: a five-candle bounce story that's almost — but not quite — provable\n\n"
            + BADGES +
            "Picture a stock in a slide: four straight red days, each closing lower than the last, "
            "each one a rung on a ladder heading down. Then, on the fifth day, it snaps back — a "
            "strong green candle that breaks the fall. Candlestick lore calls this the **ladder "
            "bottom**, and says it marks the exact moment sellers run out of gas and buyers take "
            "over.\n\n"
            "It's a great story. We tested it on 61 big US stocks over 25 years — every single time "
            "the shape appeared, both the common reading and the stricter, more literal one — and "
            "the answer turned out to be one of the more interesting kinds of \"no\" this desk has "
            "produced: **not busted, not confirmed, just short.**\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the Bonferroni correction and the "
            "placebo? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** Two cuts: a **loose** one (four falling candles, then a bullish "
            "break) and a **strict** one (adds the textbook \"warning wick\" and a true gap-up "
            "reversal). Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| How often does the *common* reading fire? | **A lot — {R['n_loose']:,} times** "
            f"across {R['n_names']} stocks over {R['years']:.0f} years. But it does nothing "
            "reliable: the 1-day reaction is *negative*, the 10-day reaction is flat, and only two "
            "of four time horizons even nominally beat the desk's significance bar (and neither "
            "survives being tested fairly against three others). |\n"
            f"| How often does the *real*, textbook version fire? | Only **{R['n_strict']}** times "
            "— about once every **19 stock-years**. Genuinely rare, exactly as the old books say. |\n"
            "| Does the rare version do better? | **Yes, tantalizingly.** It points the right way "
            "three horizons out of four, with a 20-day return of **+3.4%** net of costs — but its "
            "best number falls just short of the bar the desk sets for calling something real. |\n"
            "| So is it real? | **We can't say yes, and we can't say no either.** That's the honest "
            "finding — and it happens less often here than you'd think. |\n\n"
            "> Some of the best-looking wins happened right at the 2002, 2009 and 2020 market "
            "bottoms — which is either a beautiful confirmation, or a reminder that when the whole "
            "market bounces, every pattern near the bottom looks like a genius call."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A downtrend that produces four straight falling candles — a visible 'ladder' "
            "stepping down — followed by a fifth candle that reverses and closes higher, marks the "
            "exact bottom. The fourth candle often shows a long upper shadow: a hint that buyers "
            "are already starting to push back before the reversal actually prints.\"* — the ladder "
            "bottom, per Steve Nison's *Japanese Candlestick Charting Techniques*.\n\n"
            "It's a specific, mechanical-sounding claim — which makes it a great one to actually "
            "test instead of just eyeball on a chart."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If real, this is a chartist's dream: a mechanical, countable five-candle shape that "
            "tells you *exactly* when a downtrend is over, before the news does. That would beat "
            "just \"buying the dip\" whenever a stock has fallen for a while — the pattern would "
            "have to add real, specific information on top of ordinary mean-reversion. So the "
            "honest test isn't \"did the stock bounce after four red days\" (stocks often do) — "
            "it's \"did the *specific* ladder shape bounce *more* than an ordinary bar already "
            "sitting in the same kind of downtrend.\""
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The basket.** SPY + **{R['n_names']-1}** long-listed US large-caps across every "
            f"major sector, **{R['years']:.0f} years** of daily bars each ({R['total_bars']:,} "
            "bars total).\n"
            "- **Two honest cuts.** The **loose** ladder (four falling candles, strictly declining "
            "closes, in a genuine downtrend, then a bullish break) and the **strict** one (adds "
            "small selling-conviction shadows on the first three rungs, a longer \"warning\" wick "
            "on the fourth, and a true gap-up on the fifth) — reported side by side.\n"
            "- **The trade.** Buy at the next open (one lag — you can't trade the close you just "
            "saw), hold 1/5/10/20 days.\n"
            "- **The honest bar.** Compare against the same buy on **any other bar already sitting "
            "in a matching downtrend** — not just \"any day\" — and correct for testing four time "
            "horizons at once (a **Bonferroni** correction)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: how rare is the *real* ladder bottom, really?**"
        ),
        code(
            "labels = ['loose\\n(4 falling + bullish break)', 'strict\\n(+ warning wick, gap-up)']\n"
            "counts = [R['n_loose'], R['n_strict']]\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(labels, counts, color=[GREY, RED], width=.55)\n"
            "for i, v in enumerate(counts): ax.annotate(f'{v:,}', (i, v), ha='center', va='bottom', fontsize=12)\n"
            "ax.set_ylabel(f'occurrences across {R[\"n_names\"]} stocks x {R[\"years\"]:.0f} years')\n"
            "ax.set_title('The textbook ladder bottom is >30x rarer than the loose reading')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"loose: {R['n_loose']:,}  strict: {R['n_strict']}  (out of {R['total_bars']:,} trading days pooled)\")"
        ),
        md(
            f"**{R['n_loose']:,}** vs **{R['n_strict']}**. Demanding the actual warning-wick-plus-"
            "gap shape the books describe collapses the count by more than 30x — the pattern's own "
            "reputation for rarity turns out to be earned."
        ),
        code(
            "hs = [1, 5, 10, 20]\n"
            "loose = [R['loose'][h][1] for h in hs]\n"
            "strict = [R['strict'][h][1] for h in hs]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.5))\n"
            "x = np.arange(len(hs))\n"
            "ax.bar(x-.2, loose, .4, color=GREY, label=f'loose (n={R[\"n_loose\"]:,})')\n"
            "ax.bar(x+.2, strict, .4, color=RED, label=f'strict (n={R[\"n_strict\"]})')\n"
            "for i,(a,b) in enumerate(zip(loose,strict)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='top' if a<0 else 'bottom',fontsize=9)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='top' if b<0 else 'bottom',fontsize=9)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('mean forward return (bps)')\n"
            "ax.set_title('The strict cut looks more promising -- but is it real?')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('loose:', loose); print('strict:', strict)"
        ),
        md(
            "The loose reading is a mess: negative on day 1, positive by day 5, roughly flat by day "
            "10, positive again by day 20. That kind of sign-flipping across horizons is exactly "
            "what noise looks like, not a persistent signal.\n\n"
            "The strict reading tells a cleaner story — down on day 1, then a steadily building "
            f"positive return, reaching **+{R['strict'][20][1]:.0f} bps (+3.4%)** by day 20. That's "
            "the interesting number in this whole study. But is it bigger than what an ordinary "
            "downtrend bar would give you anyway?"
        ),
        code(
            "hs = [1, 5, 10, 20]\n"
            "deltas_loose = [R['loose'][h][4] for h in hs]\n"
            "deltas_strict = [R['strict'][h][3] for h in hs]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "x = np.arange(len(hs))\n"
            "ax.bar(x-.2, deltas_loose, .4, color=GREY, label='loose: ladder minus base rate')\n"
            "ax.bar(x+.2, deltas_strict, .4, color=RED, label='strict: ladder minus base rate')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('ladder return minus downtrend base rate (bps)')\n"
            "ax.set_title('The strict cut beats the base rate at 3 of 4 horizons -- but not certifiably')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('loose deltas:', deltas_loose); print('strict deltas:', deltas_strict)"
        ),
        md(
            "The strict cut *does* beat the base rate at 5, 10 and 20 days. The 20-day gap "
            f"(**+{R['strict'][20][3]:.0f} bps**) is the biggest and most consistent number in the "
            f"study, with a statistical score of **t = {R['strict'][20][4]:.2f}** — close to (but "
            "short of) the bar the desk requires to call an effect real once you account for "
            "testing four time horizons at once. The quants notebook has the full Bonferroni "
            "arithmetic; the honest headline is: **suggestive, not certified.**"
        ),
        md(
            "**One more thing worth seeing** — where the biggest wins and losses actually happened:"
        ),
        code(
            "names = [e[0] for e in R['strict_events']]\n"
            "vals = [e[3] for e in R['strict_events']]\n"
            "cols = [RED if v < 0 else GREEN for v in vals]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.4))\n"
            "ax.barh(names, vals, color=cols)\n"
            "ax.axvline(0, c='k', lw=.8)\n"
            "for i, v in enumerate(vals): ax.annotate(f'{v/100:+.1f}%', (v, i), va='center',\n"
            "    ha='left' if v>0 else 'right', fontsize=9)\n"
            "ax.set_xlabel('20-day return after the strict ladder bottom')\n"
            "ax.set_title('Three of the biggest wins landed at famous market-wide bottoms')\n"
            "plt.tight_layout(); plt.show()\n"
            "for tkr, d0, d1, r20 in R['strict_events']: print(f'{tkr}: {d0}->{d1}  {r20/100:+.1f}%')"
        ),
        md(
            "MS (October 2002), BA (February–March 2009) and KMB (March 2020) all fired their "
            "ladder bottom right at, or days before, a **famous, market-wide** bottom — the dot-com "
            "low, the GFC low, and the COVID crash low. That's either the pattern doing exactly "
            "what it claims, or a reminder that near a real market bottom, *almost every* stock "
            "chart draws something that looks like a bullish setup — and with only 81 events, we "
            "can't fully tell those two stories apart."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Neither cut clears the desk's significance bar once you account "
            "for testing four horizons at once. The common reading contradicts itself across "
            "horizons; the rare, textbook-faithful reading comes closer (and points the right way "
            "more often) but still falls short.\n"
            "- **Tradability — Mirage.** The textbook version fires about once every 19 stock-"
            "years — nowhere near often enough to build a strategy on, whatever the statistics say.\n"
            "- **\"Beats a downtrend base rate?\" — Mixed.** The strict cut is economically "
            "interesting and directionally right most of the time — genuinely the study's most "
            "promising thread — but it is neither confirmed nor busted on the sample size real "
            "markets offer."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **\"Almost significant\" is its own honest answer.** Most of this desk's candlestick "
            "teardowns land firmly on None; this one lands on a genuinely close miss — worth "
            "naming precisely instead of rounding up to Real or down to Busted.\n"
            "- **What would move the needle:** more names, more decades, or — the more promising "
            "route — testing whether the pattern's edge concentrates specifically around genuine "
            "market-wide drawdown bottoms (which this basket's five best/worst events hint at) "
            "rather than firing evenly across all conditions.\n"
            "- **Sibling studies:** [455-three-methods](../../455-three-methods/) (a different "
            "5-candle shape, a continuation not a reversal), "
            "[408-three-black-crows](../../408-three-black-crows/) (the same four falling "
            "candles, read bearish instead), [186-morning-star](../../186-morning-star/) (a "
            "3-candle bullish reversal) and [685-tri-star-doji](../../685-tri-star-doji/) (this "
            "study's strict/loose discipline, on three dojis instead).\n\n"
            "*Think the strict cut is onto something real? Show it survives on a broader universe "
            "or a longer tape with the Bonferroni bar still in place — then we'll talk.*"
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
            "# Ladder Bottom — a quantitative teardown 🔬\n"
            "### The loose-vs-strict detector split · downtrend-matched base rate · Bonferroni "
            "across 4 horizons · a label-shuffle placebo · costs · a synthetic planted-bounce control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "ladder bottom is a specific, mechanical five-candle claim — the job here is to encode "
            "it precisely, in two honesty-graded cuts, and grade both against a fair base rate with "
            "a multiple-comparisons correction.\n\n"
            f"> ⚠️ **Data note.** {R['n_names']} names (SPY + {R['n_names']-1} long-listed US "
            f"large-caps), yfinance daily, ~{R['years']:.0f} years each "
            f"({R['total_bars']:,} bars total), as-of {R['asof']}. Reversal bet is long-only "
            "against a matching downtrend base rate, entered at the next open (one documented "
            "lag). Methods in [`docs/references.md`](../docs/references.md), numbers in "
            f"[`docs/results.md`](../docs/results.md) (fingerprints `{R['fp_spy']}` sample / "
            f"`{R['fp_panel']}` panel).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | loose cut (n = {R['n_loose']:,}): sign flips across "
            f"horizons, best |Welch t| = {abs(R['loose'][5][5]):.2f} (5d), short of Bonferroni "
            f"**{R['bonferroni_crit']:.2f}**; strict cut (n = {R['n_strict']}): best 20d Welch "
            f"**t = {R['strict'][20][4]:.2f}**, placebo *p* = {R['strict'][20][5]:.3f}, still "
            "short |\n"
            f"| **Tradability** | `MIRAGE` | strict pattern ≈ once per 19 ticker-years; where "
            "positive, the return survives a 10 bps round trip, so costs are not the binding "
            "constraint — event frequency and certification are |\n"
            "| **Beats a downtrend base rate?** | `MIXED` | strict cut beats the base rate at "
            f"3 of 4 horizons (best delta +{R['strict'][20][3]:.0f} bps at 20d) but doesn't clear "
            "the bar; 3 of 5 best/worst events coincide with famous market-wide bottoms |\n\n"
            "> 💡 In plain words: this is the rare candlestick teardown that comes *close* — close "
            "enough that rounding it down to a flat None feels almost too tidy, and rounding it up "
            "to Real would break the desk's own rule."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $b_t = \\mathbb{1}[c_t < o_t]$ flag a bearish bar. A **loose ladder bottom** "
            "confirms at bar $t$ (the fifth candle) iff: (i) a downtrend context, "
            "$c_{t-4} < c_{t-4-10}$; (ii) four bearish rungs, $b_{t-4}, b_{t-3}, b_{t-2}, "
            "b_{t-1}$; (iii) strictly descending closes, $c_{t-4} > c_{t-3} > c_{t-2} > c_{t-1}$; "
            "(iv) a bullish fifth candle, $c_t > o_t$; (v) a reversal break, $c_t > c_{t-1}$. The "
            "**strict** cut adds: small lower shadows on rungs $t{-}4..t{-}2$ (committed "
            "selling), a longer upper shadow on rung $t{-}1$ than rung $t{-}2$ (the \"warning "
            "wick\"), and a true gap-up, $o_t > c_{t-1}$.\n\n"
            "- **H₀ (no signal).** The reversal return equals the *downtrend-matched* base rate "
            "(the same long bet on every bar already in a matching downtrend, whether or not the "
            "specific shape fired).\n"
            "- **H₁ (real bottom).** The reversal return **exceeds** the base rate, Welch "
            "*t* ≥ 2, surviving a Bonferroni correction across the 4 horizons tested.\n"
            "- **The honest sample-size rule.** Below **8** pooled events, no *t*-stat is "
            "computed.\n\n"
            "We find: loose cut, **H₀ not rejected** (sign flips across horizons — the classic "
            "\"pick a horizon\" trap the correction guards against); strict cut, **closest "
            "approach to H₁** (20d Welch *t* = "
            f"{R['strict'][20][4]:.2f}) but still short of the Bonferroni bar."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Two honesty problems stack here, the same shape sibling study 685 faces. **First, "
            "the base rate must be matched to context, not just \"any day\"** — a bar already "
            "sitting in a downtrend has its own mean-reversion tendency independent of the "
            "five-candle shape; comparing the ladder to *any random bar* would credit the pattern "
            "with ordinary downtrend bounce-back that has nothing to do with its geometry. "
            "**Second, four horizons means four simultaneous looks** at the same question; at the "
            "usual *α* = 5% one spurious hit is expected roughly every 20 independent tries, so we "
            "apply a **Bonferroni correction** (k = 4, critical |*t*| ≥ "
            f"**{R['bonferroni_crit']:.2f}**) — the same discipline siblings 186 and 685 use. The "
            "decisive statistic is a **Welch *t*** of the ladder reversal mean against the "
            "downtrend base-rate mean (never a one-sample *t* against zero, which would just "
            "measure the basket's drift)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** SPY + {R['n_names']-1} long-listed US large-caps, yfinance daily, "
            f"~{R['years']:.0f}y each ({R['total_bars']:,} bars total), as-of {R['asof']}.\n"
            "- **Detector, two cuts.** Loose (4 declining candles + downtrend + bullish break) "
            "and strict (+ committed rungs, warning wick, gap-up) — reported side by side.\n"
            "- **Entry.** the open one session after the fifth candle (one documented lag); "
            "hold 1/5/10/20 days.\n"
            "- **Benchmark.** the downtrend-matched base rate — the same long bet on every bar "
            "already sitting in a matching downtrend context.\n"
            "- **Tests.** Welch *t* of ladder vs base-rate mean (decisive); one-sample HAC *t* "
            "where n permits; a 2,000-draw label-shuffle placebo; Bonferroni across 4 horizons; "
            "below 8 events, no test at all.\n"
            "- **Costs.** 5 bps one-way, 10 bps round trip, long-only, no borrow.\n"
            "- **Control.** synthetic panel, forced 5-bar ladder blocks (engineered downtrend + "
            "4 declining candles + 1 reversal candle), planted post-block bounce knob `edge`; the "
            "null must not fire across 20 seeds; a planted edge must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The detector — loose vs strict, and why it matters\n\n"
            "The pattern's reputation for rarity is earned once the full geometry is required."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ev = st.pool_events(PANEL)\n"
            "    n_loose, n_strict = len(ev), int(ev['strict'].sum())\n"
            "else:\n"
            "    n_loose, n_strict = R['n_loose'], R['n_strict']\n"
            "fig, ax = plt.subplots(figsize=(7.4, 4.3))\n"
            "ax.bar(['loose\\n(4 falling + break)', 'strict\\n(+ wick, gap-up)'], [n_loose, n_strict],\n"
            "       color=[GREY, RED], width=.55)\n"
            "for i,v in enumerate([n_loose, n_strict]): ax.annotate(f'{v:,}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel(f'pooled occurrences ({R[\"n_names\"]} names x {R[\"years\"]:.0f}y)')\n"
            "ax.set_title('The literature-closer cut is >30x rarer than the loose reading')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'loose={n_loose:,}  strict={n_strict}')"
        ),
        md(
            f"> 💡 In plain words: **{R['n_loose']:,}** vs **{R['n_strict']}**. The warning-wick-"
            "plus-gap requirement — the part of the definition that actually encodes \"buyers are "
            "fighting back before the reversal prints\" — is doing almost all of the rarity work."
        ),
        md(
            "### 4b · The loose cut — a sign that flips across horizons\n\n"
            "Welch *t* of the ladder reversal mean vs the downtrend-matched base-rate mean, at "
            "each horizon, against the Bonferroni-corrected critical value."
        ),
        code(
            "hs = list(st.HORIZONS)\n"
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(PANEL)\n"
            "    ts = [res['per_horizon'][h]['welch_t'] for h in hs]\n"
            "    deltas = [res['per_horizon'][h]['delta_bps'] for h in hs]\n"
            "    crit = res['bonferroni_crit']\n"
            "else:\n"
            "    ts = [R['loose'][h][5] for h in hs]; deltas = [R['loose'][h][4] for h in hs]\n"
            "    crit = R['bonferroni_crit']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], deltas, color=GREY, width=.6)\n"
            "a1.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(deltas): a1.annotate(f'{v:+.0f}',(i,v),ha='center',\n"
            "    va='top' if v<0 else 'bottom',fontsize=9)\n"
            "a1.set_ylabel('ladder minus base rate (bps)'); a1.set_title('Delta: sign flips across horizons')\n"
            "a2.bar([f'{h}d' for h in hs], ts, color=GREY, width=.6)\n"
            "a2.axhline(-crit, ls='--', c=RED, label=f'Bonferroni bar (|t|={crit:.2f})')\n"
            "a2.axhline(crit, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(ts): a2.annotate(f'{v:+.2f}',(i,v),ha='center',\n"
            "    va='top' if v<0 else 'bottom',fontsize=9)\n"
            "a2.set_ylabel('Welch t (ladder vs base rate)'); a2.set_title('No horizon clears the bar')\n"
            "a2.legend(); plt.tight_layout(); plt.show()\n"
            "print('deltas (bps):', [round(d,1) for d in deltas]); print('welch t:', [round(t,2) for t in ts])"
        ),
        md(
            f"> 💡 In plain words: negative at 1 day (*t* = {R['loose'][1][5]:.2f}), positive at 5 "
            f"(*t* = {R['loose'][5][5]:.2f}), flat at 10 (*t* = {R['loose'][10][5]:.2f}), positive "
            f"again at 20 (*t* = {R['loose'][20][5]:.2f}). Two of four nominally clear a naive "
            "|*t*| ≥ 2 — precisely the kind of result the Bonferroni correction exists to catch: "
            "with four independent looks, chance alone produces roughly this many marginal hits. "
            "A genuine, persistent edge should not flip sign between its own horizons."
        ),
        md(
            "### 4c · The strict cut — closer, and directionally coherent\n\n"
            "Same test, the literature-closer geometry. n = 81 — small enough that the placebo and "
            "the individual event list both matter as much as the *t*-stat."
        ),
        code(
            "hs = list(st.HORIZONS)\n"
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(PANEL)\n"
            "    ts = [res['strict_per_horizon'][h]['welch_t'] for h in hs]\n"
            "    deltas = [res['strict_per_horizon'][h]['delta_bps'] for h in hs]\n"
            "    crit = res['bonferroni_crit']\n"
            "else:\n"
            "    ts = [R['strict'][h][4] for h in hs]; deltas = [R['strict'][h][3] for h in hs]\n"
            "    crit = R['bonferroni_crit']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], deltas, color=RED, width=.6)\n"
            "a1.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(deltas): a1.annotate(f'{v:+.0f}',(i,v),ha='center',\n"
            "    va='top' if v<0 else 'bottom',fontsize=9)\n"
            "a1.set_ylabel('ladder minus base rate (bps)'); a1.set_title('Delta: positive at 3 of 4 horizons')\n"
            "a2.bar([f'{h}d' for h in hs], ts, color=RED, width=.6)\n"
            "a2.axhline(-crit, ls='--', c=GREY, label=f'Bonferroni bar (|t|={crit:.2f})')\n"
            "a2.axhline(crit, ls='--', c=GREY); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(ts): a2.annotate(f'{v:+.2f}',(i,v),ha='center',\n"
            "    va='top' if v<0 else 'bottom',fontsize=9)\n"
            "a2.set_ylabel('Welch t (ladder vs base rate)'); a2.set_title(f'20d closest: t={ts[-1]:.2f} vs bar {crit:.2f}')\n"
            "a2.legend(); plt.tight_layout(); plt.show()\n"
            "print('deltas (bps):', [round(d,1) for d in deltas]); print('welch t:', [round(t,2) for t in ts])"
        ),
        md(
            f"> 💡 In plain words: the strict cut's 20-day Welch *t* = **{R['strict'][20][4]:.2f}** "
            f"is the study's closest brush with certification — short of the "
            f"**{R['bonferroni_crit']:.2f}** bar by less than 0.2, on n = {R['n_strict']} events. "
            "That is genuinely suggestive and genuinely not enough: with a sample this small, one "
            "or two outsized events (see 4e) can move the mean by hundreds of basis points."
        ),
        md(
            "### 4d · The label-shuffle placebo\n\n"
            "Draw a same-size random set of bars from the downtrend base-rate pool 2,000 times; "
            "the placebo *p* is the share of draws whose mean beats the observed ladder mean."
        ),
        code(
            "hs = list(st.HORIZONS)\n"
            "ps_loose = [R['loose'][h][6] for h in hs]\n"
            "ps_strict = [R['strict'][h][5] for h in hs]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "x = np.arange(len(hs))\n"
            "ax.bar(x-.2, ps_loose, .4, color=GREY, label='loose')\n"
            "ax.bar(x+.2, ps_strict, .4, color=RED, label='strict')\n"
            "ax.axhline(0.05, c='k', lw=.8, ls=':', label='p = 0.05')\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('placebo p (share of random draws beating the ladder)')\n"
            "ax.set_title('Strict 20d placebo (p=0.006) is the most suggestive single number here')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('placebo p, loose:', dict(zip(hs, ps_loose)))\n"
            "print('placebo p, strict:', dict(zip(hs, ps_strict)))"
        ),
        md(
            f"> 💡 In plain words: at 20 days, only **{R['strict'][20][5]*100:.1f}%** of random "
            "same-size draws from the downtrend pool beat the actual strict ladder-bottom mean — "
            "genuinely rare under the null. The placebo and the Welch test tell a consistent story "
            "(both point toward 20d being the study's best evidence), which is reassuring about "
            "the machinery — it just isn't reassuring enough to certify the claim once the "
            "multiple-horizon correction is applied."
        ),
        md(
            "### 4e · The strict cut's event list — where the mean actually comes from\n\n"
            "With n = 81, individual events matter. The five most extreme 20-day outcomes:"
        ),
        code(
            "for tkr, d0, d1, r20 in R['strict_events']:\n"
            "    print(f'{tkr}: block {d0} -> {d1}   20d {r20/100:+.1f}%')"
        ),
        md(
            "MS (October 2002), BA (Feb–Mar 2009) and KMB (March 2020) landed right at, or days "
            "before, textbook broad-market bottoms; LMT (2003) and NKE (2026) fired mid-decline "
            "with no rescue in sight. That mix — some of the biggest wins riding a genuine "
            "market-wide turn, not obviously the five-candle geometry on its own — is exactly the "
            "kind of confound a larger sample would need to resolve, and precisely why n = 81 "
            "can suggest without certifying."
        ),
        md(
            "### 4f · Costs\n\n"
            "5 bps one-way, 10 bps round trip, long-only. Where the point estimate is positive it "
            "survives costs — this pattern's binding constraint is certification and frequency, "
            "not transaction costs."
        ),
        code(
            "hs = list(st.HORIZONS)\n"
            "gross = [R['strict'][h][1] for h in hs]; net = [R['strict'][h][6] for h in hs]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.2))\n"
            "x = np.arange(len(hs))\n"
            "ax.bar(x-.2, gross, .4, color=GREY, label='gross'); ax.bar(x+.2, net, .4, color=RED, label='net')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('bps')\n"
            "ax.set_title('Strict cut: positive returns survive a 10 bps round trip'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('gross:', gross); print('net:', net)"
        ),
        md(
            "### 4g · Synthetic positive control — the machinery is unbiased\n\n"
            "Deterministic panel with forced 5-bar ladder blocks (engineered downtrend, four "
            "declining candles, one reversal candle) and a TUNABLE planted bounce. The decisive "
            "statistic is the same Welch *t* used on the real tape. The null is checked over "
            "**20 seeds** — never a single stream."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    d, _t = data.synthetic_panel(edge=0.0, seed=687 + s_)\n"
            "    r = st.synthetic_detect(d, horizon=20, seed=687 + s_)\n"
            "    if r['welch_t'] is not None:\n"
            "        null_ts.append(r['welch_t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "planted = []\n"
            "for edge in (0.02, 0.04):\n"
            "    d, _t = data.synthetic_panel(edge=edge, seed=687)\n"
            "    r = st.synthetic_detect(d, horizon=20, seed=687)\n"
            "    planted.append((edge, r['welch_t']))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(len(null_ts)) + np.linspace(-.12,.12,len(null_ts)), null_ts,\n"
            "           color=GREY, s=40, label=f'null worlds (edge=0), {len(null_ts)} seeds')\n"
            "for i,(e,t) in enumerate(planted):\n"
            "    ax.scatter([i+1], [t], color=RED, s=90, zorder=5,\n"
            "               label=f'planted edge={e:.2f}')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0]+[i+1 for i in range(len(planted))])\n"
            "ax.set_xticklabels(['null x ' + str(len(null_ts))] + [f'edge={e:.2f}' for e,_ in planted])\n"
            "ax.set_ylabel('Welch t (ladder vs base rate)')\n"
            "ax.set_title('Control: null stays small; a planted bounce lights up cleanly')\n"
            "ax.legend(fontsize=8); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(np.abs(null_ts)>=2).sum()}/{len(null_ts)} seeds')\n"
            "for e,t in planted: print(f'planted edge={e:.2f}: t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across {R['syn_null_seeds']} null worlds the detector averages "
            f"*t* = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and **never** fires at "
            f"|*t*| ≥ 2; a planted bounce of just 0.02–0.04 (far smaller than real market swings) "
            f"lights up cleanly at *t* = {R['syn_edge_002_t']:+.2f} and {R['syn_edge_004_t']:+.2f}. "
            "The machinery is unbiased and has the power to find a real effect of this size. Per "
            "the desk's rule, **a synthetic control never backs a Signal stamp** — it only proves "
            "the harness would catch a genuine planted effect, which it does. The honest, "
            "close-but-short real-tape result above stands on its own."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the loose cut (n = {R['n_loose']:,}) contradicts itself across "
            "horizons (negative at 1d, flat at 10d) and no horizon clears the Bonferroni bar "
            f"(**{R['bonferroni_crit']:.2f}**). The literature-closer strict cut (n = "
            f"{R['n_strict']}) is directionally coherent and comes closest — 20d Welch *t* = "
            f"**{R['strict'][20][4]:.2f}**, placebo *p* = **{R['strict'][20][5]:.3f}** — but still "
            "falls short, and its own 1-day reaction is negative.\n"
            f"- **Tradability `MIRAGE`** — the strict pattern fires about once per 19 ticker-years; "
            "even the loose cut, far more frequent, fails certification. Where positive, the "
            "return survives a 10 bps round trip — costs are not the binding constraint here.\n"
            "- **\"Beats a downtrend base rate?\" `MIXED`** — the strict cut beats the base rate "
            f"at 3 of 4 horizons with an economically large net return (+{R['strict'][20][6]:.0f} "
            "bps at 20d) and a suggestive placebo, but doesn't clear the certification bar, and "
            "3 of its 5 most extreme events coincide with famous *market-wide* bottoms rather than "
            "obviously the candle geometry alone. Not busted, not confirmed."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **A close miss is worth naming precisely.** Rounding this up to Real would break "
            "the desk's own multiple-testing discipline; rounding it down to a flat, uninteresting "
            "None would hide the strict cut's genuinely consistent direction and suggestive "
            "placebo. Both things are true at once, and that tension *is* the finding.\n"
            "- **The market-bottom confound is the natural next question.** Three of the five most "
            "extreme strict events sit at famous, broad drawdown bottoms. A follow-up could "
            "condition on a market-wide (not just single-name) downtrend to see whether the "
            "ladder's apparent edge concentrates there — which would be a *different*, testable "
            "claim from \"any stock's ladder bottom marks a turn.\"\n"
            "- **Dedup map:** [455-three-methods](../../455-three-methods/) (a different 5-candle "
            "shape, a continuation not a reversal, on ETFs), "
            "[408-three-black-crows](../../408-three-black-crows/) (the same four falling candles "
            "read bearish, shorted — not this study's fifth, reversing candle), "
            "[186-morning-star](../../186-morning-star/) (a 3-candle bullish reversal, the same "
            "random-baseline/Bonferroni idiom), [685-tri-star-doji](../../685-tri-star-doji/) "
            "(three dojis, the strict/loose + `MIN_N_FOR_TEST` discipline this study reuses "
            "directly).\n\n"
            "*Reproducible core is offline and deterministic; frozen numbers live in "
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
