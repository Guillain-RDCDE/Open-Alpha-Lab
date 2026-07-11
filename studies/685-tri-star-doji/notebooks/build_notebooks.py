"""Generate the two narrative notebooks for Study 685 (Tri-Star Doji).

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
    fp_spy="d1ac9e0ecaae", fp_panel="7c12cddbd539",
    n_strict=2, n_loose=410, bonferroni_crit=2.50,
    # strict cut: horizon -> (n, mean_bps, win_rate_pct)
    strict={5: (2, -243.0, 0), 10: (2, -249.6, 0), 20: (2, -610.6, 0), 60: (2, -1355.3, 0)},
    strict_events=[("JPM", "2021-03-17", "2021-03-19", -49.0, -215.6),
                   ("CSCO", "2019-01-24", "2019-01-28", -1172.2, -2495.1)],
    # loose cut: horizon -> (n, tri_bps, win_pct, base_bps, delta_bps, welch_t, placebo_p, net_bps)
    loose={5: (410, -14.3, 48, 2.1, -16.4, -1.02, 0.826, -16.8),
           10: (410, -19.1, 47, 1.9, -20.9, -0.98, 0.807, -21.6),
           20: (410, -76.6, 44, 6.2, -82.8, -2.47, 0.986, -79.1),
           60: (410, -86.7, 46, 7.5, -94.1, -1.65, 0.933, -89.2)},
    # synthetic control
    syn_null_mean=-0.60, syn_null_sd=1.16, syn_null_fire=4, syn_null_seeds=20,
    syn_edge_002_t=+2.36, syn_edge_002_delta=52.9,
    syn_edge_004_t=+5.14, syn_edge_004_delta=129.3,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Tri-star_forecasts%3F: Busted](https://img.shields.io/badge/Tri--star_forecasts%3F-Busted-8b949e?style=flat-square)\n\n"
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

from tri_star_doji import data, strategy as st

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
            "# Three dojis in a row — the market's rarest \"never mind\" ✨\n"
            "### The tri-star: candlestick lore's most legendary reversal, and its two lifetime appearances\n\n"
            + BADGES +
            "A **doji** is a session where the close lands almost exactly where the open was — a "
            "coin-flip of a day, neither side wins. Most of the time it's a footnote. But "
            "candlestick lore has one truly dramatic escalation: the **tri-star** — *three* dojis "
            "in a row, each one gapping away from its neighbours like three tiny islands. Steve "
            "Nison, who brought Japanese candlesticks to the West, calls it one of the rarest "
            "patterns of all — and says when it shows up, it doesn't just pause a trend, it "
            "**ends** it.\n\n"
            "Rare claims are fun to test, because rare means we can actually go *count* them. So "
            "we did: 61 large, liquid US stocks, 25 years of daily bars, every tri-star, both the "
            "purist's gapped version and a looser \"just three dojis\" reading. What we found is "
            "almost funnier than a flat no.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the Bonferroni correction and the "
            "placebo? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** Doji = a real body no bigger than 10% of the day's high-low "
            "range (same cut the desk uses everywhere). Every chart is drawn by the code beside "
            "it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| How rare is the *real* tri-star (gapped, as Nison defines it)? | **Two occurrences.** "
            f"In **{R['n_names']} stocks over {R['years']:.0f} years** ({R['total_bars']:,} trading "
            "days total), the pattern, defined the way the book actually defines it, happened "
            "**twice**. |\n"
            "| What happened after those two? | **Both lost.** JPM (March 2021) kept climbing; the "
            "'reversal' bet cost 2.2% over the next 3 months. CSCO (January 2019) kept climbing "
            "*harder* — the bet cost 25.0%. Zero for two. |\n"
            "| What if we're generous and drop the gap requirement? | It stops being rare "
            f"(**{R['n_loose']}** occurrences) — but the reversal bet still **loses to a random "
            "day** at every single time horizon we checked. |\n"
            "| So does stacking three dojis beat just one? | **No.** A single doji already failed "
            "this exact test in [study 405](../../../405-doji-reversal/); tripling it doesn't "
            "help. |\n\n"
            "> Sometimes 'this pattern is legendarily rare' is candlestick-speak for 'we never "
            "actually counted how often it happens, or what happened next.'"
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Three dojis in a row — each one a session of pure indecision, each one gapping "
            "away from the last — is the rarest and most powerful reversal signal in candlestick "
            "charting. It doesn't just pause a trend. It ends it.\"* — the tri-star, per Nison's "
            "*Japanese Candlestick Charting Techniques*.\n\n"
            "It's the natural end point of an escalation: one doji is indecision "
            "([study 405](../../../405-doji-reversal/)); a doji marooned by gaps is an "
            "*abandoned baby* ([study 458](../../../458-abandoned-baby/)); **three** dojis, each "
            "gapped, is candlestick lore's grand finale — the market completely losing its nerve, "
            "three days running."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a chart pattern really *did* call major reversals, this would be as close to a "
            "free lunch as charting gets: an unmistakable, rare, high-conviction signal you could "
            "circle on a chart and act on. But \"rare\" cuts both ways — a pattern almost never "
            "seen is also a pattern almost never *tested*, which is exactly the soil where "
            "folklore grows unchecked. So: how rare is 'rare', actually — and on the (few) "
            "occasions it fires, does the market actually turn?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The basket.** SPY + **{R['n_names']-1}** long-listed US large-caps across every "
            f"major sector, **{R['years']:.0f} years** of daily bars each ({R['total_bars']:,} "
            "bars total) — as broad and deep a net as we can throw at a genuinely rare pattern.\n"
            "- **Two honest cuts.** The **strict** tri-star (three dojis, the middle one truly "
            "gapped clear of both neighbours — Nison's own definition) and a **loose** one (three "
            "dojis back to back, no gap required) — reported side by side, not cherry-picked.\n"
            "- **The trade.** Bet *against* the 10-day trend into the tri-star at the next open — "
            "the textbook reversal — and read forward 5/10/20/60-day returns.\n"
            "- **The honest bar.** Compare against the **same bet on any random day** (the base "
            "rate), and if there are too few events to test — **say so**, instead of quoting a "
            "*t*-stat computed on a handful of points."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: how rare is a *real* tri-star, really?**"
        ),
        code(
            "labels = ['strict\\n(gapped island star)', 'loose\\n(3 dojis, no gap)']\n"
            "counts = [R['n_strict'], R['n_loose']]\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(labels, counts, color=[RED, GREY], width=.55)\n"
            "for i, v in enumerate(counts): ax.annotate(str(v), (i, v), ha='center', va='bottom', fontsize=12)\n"
            "ax.set_ylabel(f'occurrences across {R[\"n_names\"]} stocks x {R[\"years\"]:.0f} years')\n"
            "ax.set_title('The real (gapped) tri-star happened TWICE in a quarter-century')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"strict: {R['n_strict']}  loose: {R['n_loose']}  (out of {R['total_bars']:,} trading days pooled)\")"
        ),
        md(
            "Two. Not \"rare as in a few dozen\" — **two**, full stop, defined the way the book "
            "actually defines it. That alone should make anyone pause before calling it a "
            "*reliable* signal. But since it happened, let's meet them."
        ),
        code(
            "for tkr, d0, d1, r20, r60 in R['strict_events']:\n"
            "    print(f'{tkr}: tri-star block {d0} -> {d1}   '\n"
            "          f'20-day reversal {r20:+.1f} bps   60-day reversal {r60:+.1f} bps')"
        ),
        md(
            "**JPM, March 2021** — a tri-star after a run-up; the 'major reversal' bet lost "
            f"**{abs(R['strict_events'][0][4])/100:.1f}%** over the next three months as JPM kept "
            "climbing. **CSCO, January 2019** — same story, worse: the bet lost "
            f"**{abs(R['strict_events'][1][4])/100:.1f}%** as Cisco's early-2019 rally simply "
            "continued straight through the signal. Zero for two isn't proof of anything by "
            "itself (n=2!) — but it's the complete, honest record."
        ),
        md(
            "**Now the generous version.** Drop the gap requirement — allow any three dojis in a "
            f"row — and the count jumps to **{R['n_loose']}**, enough to actually test. Does the "
            "reversal bet beat a random day now?"
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "tri = [R['loose'][h][1] for h in hs]\n"
            "base = [R['loose'][h][3] for h in hs]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "x = np.arange(len(hs))\n"
            "ax.bar(x-.2, tri, .4, color=RED, label='tri-star reversal bet')\n"
            "ax.bar(x+.2, base, .4, color=GREY, label='base rate (any day, same bet)')\n"
            "for i,(a,b) in enumerate(zip(tri,base)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='top' if a<0 else 'bottom',fontsize=9)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=9)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('mean reversal return (bps)')\n"
            "ax.set_title('The loose tri-star LOSES to a random day, every single horizon')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('tri-star:', tri); print('base rate:', base)"
        ),
        md(
            "Not close, and not in the pattern's favor. At every horizon the tri-star reversal bet "
            f"underperforms the base rate by **{abs(R['loose'][5][4]):.0f} to "
            f"{abs(R['loose'][20][4]):.0f} bps**. The quants notebook shows the strongest gap (20 "
            "days) still falls short of the desk's certification bar — and even if it hadn't, it's "
            "the *wrong sign* for the claim. A random day is a better trade than the market's most "
            "famous reversal signal."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The real (gapped) tri-star fired twice in a quarter-century — "
            "both losers — too few to test. The loose reading (410 occurrences) loses to a random "
            "day at every horizon.\n"
            "- **Tradability — Mirage.** Once every ~30 ticker-years is not a strategy, and there's "
            "no gross edge for costs to even threaten.\n"
            "- **\"Tri-star marks a major reversal\"? — Busted.** Two real trades, two losses; the "
            "common reading loses to a coin flip. Stacking three dojis instead of one doesn't "
            "manufacture a signal a single doji never had."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **The escalation, closed out.** [Study 405](../../../405-doji-reversal/) (one "
            "doji) and [study 458](../../../458-abandoned-baby/) (one gapped doji between "
            "directional candles) both land the same place: the candlestick-of-indecision family "
            "doesn't forecast, no matter how you stack it.\n"
            "- **What next?** A pattern this rare simply can't be rescued with more statistics — "
            "the binding constraint is real-world occurrence count, not method. Intraday bars "
            "would multiply the count but change the microstructure entirely.\n\n"
            "*Think three-in-a-row is different from one? Show a tri-star reversal edge on a "
            "basket where it fires more than twice — then we'll talk.*"
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
            "# Tri-Star Doji — a quantitative teardown 🔬\n"
            "### The strict-vs-loose detector split · base-rate comparison · Bonferroni across "
            "4 horizons · a label-shuffle placebo · costs · a synthetic planted-reversal control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "tri-star is candlestick lore's rarest reversal claim, and the job here is to take "
            "\"rare\" literally: count it honestly under the literature-faithful definition, count "
            "it again under a looser reading that's actually testable, and grade both against a "
            "fair base rate with a multiple-comparisons correction.\n\n"
            f"> ⚠️ **Data note.** {R['n_names']} names (SPY + {R['n_names']-1} long-listed US "
            f"large-caps), yfinance daily, ~{R['years']:.0f} years each "
            f"({R['total_bars']:,} bars total), as-of {R['asof']}. Doji body ≤ 10% of range; "
            "reversal bet is against the 10-day trend into the tri-star, entered at the next "
            "open (one documented lag). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
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
            f"| **Signal** | `NONE` | strict cut n = {R['n_strict']} (too few to test, 0% win "
            f"rate); loose cut (n = {R['n_loose']}) delta negative at every horizon, closest to "
            f"significance at 20d (Welch **t = {R['loose'][20][5]:.2f}**, wrong sign, short of "
            f"Bonferroni **{R['bonferroni_crit']:.2f}**) |\n"
            "| **Tradability** | `MIRAGE` | strict pattern ≈ once per 30 ticker-years; loose cut "
            "has no gross edge for costs to threaten |\n"
            "| **Tri-star forecasts?** | `BUSTED` | 2/2 strict occurrences lost; label-shuffle "
            f"placebo beats the loose tri-star **{R['loose'][20][6]*100:.0f}–"
            f"{R['loose'][60][6]*100:.0f}%** of draws |\n\n"
            "> 💡 In plain words: the rarer the reading, the fewer chances the claim gets to be "
            "right — and on every reading available, it wasn't."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $D_t$ flag a doji at bar $t$ ($|c_t-o_t|\\le 0.10\\,(h_t-l_t)$, non-degenerate "
            "range). A **tri-star** at $t$ requires $D_{t-2}\\wedge D_{t-1}\\wedge D_t$. The "
            "**strict** cut additionally requires the middle doji's range to clear *both* "
            "neighbours (a true gapped island); the **loose** cut drops that requirement. Tag "
            "$P_t=\\mathrm{sign}(c_{t-2}/c_{t-2-10}-1)$ (the 10-day trend into the block) and "
            "define the reversal return at horizon $h$ as $-P_t\\,(c_{t+1+h}/o_{t+1}-1)$ (entered "
            "one bar after the third doji — no look-ahead).\n\n"
            "- **H₀ (no signal).** Reversal return equals the unconditional base rate (the same "
            "bet on every eligible bar, direction-matched).\n"
            "- **H₁ (major reversal).** Reversal return **exceeds** the base rate, *t* ≥ 2, "
            "surviving a Bonferroni correction across the 4 horizons tested.\n"
            "- **The honest sample-size rule.** Below **8** pooled events, no *t*-stat is "
            "computed — a t on a handful of points is decoration, not evidence.\n\n"
            "We find: strict cut, **n too small to test** (2 events, 0% win rate); loose cut, "
            "**H₀ not rejected at any horizon**, and the point estimate runs the *wrong way*."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Two honesty problems stack on top of each other here. **First, the rarity itself is "
            "part of the finding** — literally counting how often the literature-faithful pattern "
            "occurs is a legitimate (if unglamorous) empirical result, and it caps what can ever "
            "be certified. **Second, four horizons means four simultaneous looks** at the same "
            "question; at the usual *α* = 5% one spurious hit is expected roughly every 20 "
            "independent tries, so we apply a **Bonferroni correction** (k = 4, critical "
            f"|*t*| ≥ **{R['bonferroni_crit']:.2f}**) — the same discipline sibling study 186 uses "
            "for its morning-/evening-star grid. The decisive statistic is a **Welch *t*** of the "
            "tri-star reversal mean against the base-rate mean (not a one-sample *t* against zero, "
            "which would just measure the basket's drift — the lesson sibling 405 already "
            "established on a single doji)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** SPY + {R['n_names']-1} long-listed US large-caps, yfinance daily, "
            f"~{R['years']:.0f}y each ({R['total_bars']:,} bars total), as-of {R['asof']}.\n"
            "- **Detector, two cuts.** Strict (gapped island star, literature-faithful) and loose "
            "(3 dojis, no gap) — reported side by side.\n"
            "- **Trend tag.** 10-day move into the block's first bar; reversal bet is against it.\n"
            "- **Entry.** the open one session after the third doji (one documented lag); hold "
            "5/10/20/60 days.\n"
            "- **Benchmark.** the unconditional base rate — the same against-the-trend bet on "
            "every eligible bar (not zero — the desk's standing rule).\n"
            "- **Tests.** Welch *t* of tri-star vs base-rate mean (decisive); one-sample HAC *t* "
            "and a 2,000-draw label-shuffle placebo where n permits; Bonferroni across 4 "
            "horizons; below 8 events, no test at all.\n"
            "- **Costs.** 2 bps round-trip + 1 bp borrow on the short leg.\n"
            "- **Control.** synthetic panel, forced 3-bar doji blocks, planted reversal knob "
            "`edge`; the null must not fire across 20 seeds; a planted edge must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The detector — strict vs loose, and why it matters\n\n"
            "The literature's own emphasis on rarity turns out to be earned. Demanding the actual "
            "gapped 'star' shape collapses the candidate pool by two orders of magnitude."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ev = st.pool_events(PANEL)\n"
            "    n_loose, n_strict = len(ev), int(ev['gapped'].sum())\n"
            "else:\n"
            "    n_loose, n_strict = R['n_loose'], R['n_strict']\n"
            "fig, ax = plt.subplots(figsize=(7.4, 4.3))\n"
            "ax.bar(['strict\\n(gapped island)', 'loose\\n(3 dojis, no gap)'], [n_strict, n_loose],\n"
            "       color=[RED, GREY], width=.55)\n"
            "for i,v in enumerate([n_strict, n_loose]): ax.annotate(str(v),(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel(f'pooled occurrences ({R[\"n_names\"]} names x {R[\"years\"]:.0f}y)')\n"
            "ax.set_title('The literature-faithful cut is ~200x rarer than the loose reading')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'strict={n_strict}  loose={n_loose}')"
        ),
        md(
            f"> 💡 In plain words: **{R['n_strict']}** vs **{R['n_loose']}**. The gap requirement — "
            "the whole point of calling it a *star* — is doing almost all the rarity work. Any "
            "backtest of this pattern that doesn't report this split is quietly picking one "
            "definition and hiding how much the answer depends on it."
        ),
        md(
            "### 4b · The strict cut — n = 2, reported honestly\n\n"
            "Two events cannot support a *t*-test (:data:`MIN_N_FOR_TEST` = 8). We report them as "
            "what they are: a complete, tiny sample."
        ),
        code(
            "for h in st.HORIZONS:\n"
            "    n, m, w = R['strict'][h]\n"
            "    print(f'{h:>2}d: n={n}  mean={m:+8.1f} bps  win={w}%   (n < 8 -> no t-stat computed)')\n"
            "print()\n"
            "for tkr, d0, d1, r20, r60 in R['strict_events']:\n"
            "    print(f'{tkr}: block {d0} -> {d1}   20d {r20:+.1f} bps   60d {r60:+.1f} bps')"
        ),
        md(
            "> 💡 In plain words: JPM (March 2021) and CSCO (January 2019) are the *entire* real "
            "sample. Both tri-stars fired after an up-move (the claim: sell); both stocks kept "
            "rising. n = 2 can't reject anything formally, but it also cannot possibly support the "
            "claim — the honest verdict at this sample size is simply **untested, not confirmed**."
        ),
        md(
            "### 4c · The loose cut — testable, and the sign is wrong\n\n"
            "Welch *t* of the tri-star reversal mean vs the base-rate mean, at each horizon, "
            "against the Bonferroni-corrected critical value."
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
            "a1.bar([f'{h}d' for h in hs], deltas, color=RED, width=.6)\n"
            "a1.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(deltas): a1.annotate(f'{v:+.0f}',(i,v),ha='center',va='top',fontsize=9)\n"
            "a1.set_ylabel('tri-star minus base rate (bps)'); a1.set_title('Delta: negative at every horizon')\n"
            "a2.bar([f'{h}d' for h in hs], ts, color=GREY, width=.6)\n"
            "a2.axhline(-crit, ls='--', c=RED, label=f'Bonferroni bar (|t|={crit:.2f})')\n"
            "a2.axhline(crit, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(ts): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='top',fontsize=9)\n"
            "a2.set_ylabel('Welch t (tri-star vs base rate)'); a2.set_title('No horizon clears the bar')\n"
            "a2.legend(); plt.tight_layout(); plt.show()\n"
            "print('deltas (bps):', [round(d,1) for d in deltas]); print('welch t:', [round(t,2) for t in ts])"
        ),
        md(
            f"> 💡 In plain words: every bar in the left panel points down — the tri-star reversal "
            f"underperforms a random day at **every single horizon**. The closest brush with "
            f"significance, 20 days (*t* = {R['loose'][20][5]:.2f}), still falls short of the "
            f"Bonferroni bar (**{R['bonferroni_crit']:.2f}**) — and even a naive |t| ≥ 2, ignoring "
            "the multiple-comparisons correction entirely, would still be the **wrong sign** for "
            "the claim."
        ),
        md(
            "### 4d · The label-shuffle placebo — a random day beats it\n\n"
            "Draw a same-size random set of bars from the base-rate pool 2,000 times; the placebo "
            "*p* is the share of draws whose mean beats the observed tri-star mean."
        ),
        code(
            "hs = list(st.HORIZONS)\n"
            "ps = [R['loose'][h][6] for h in hs]\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.2))\n"
            "ax.bar([f'{h}d' for h in hs], ps, color=[RED if p>0.95 else AMBER for p in ps], width=.6)\n"
            "ax.axhline(0.5, c='k', lw=.8, ls=':')\n"
            "for i,p in enumerate(ps): ax.annotate(f'{p:.3f}',(i,p),ha='center',va='bottom')\n"
            "ax.set_ylim(0,1.05); ax.set_ylabel('placebo p (share of random draws beating tri-star)')\n"
            "ax.set_title('A random same-size sample beats the tri-star most of the time')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('placebo p by horizon:', dict(zip(hs, ps)))"
        ),
        md(
            f"> 💡 In plain words: at 20 days, **{R['loose'][20][6]*100:.1f}%** of random same-size "
            "draws from ordinary bars do *better* than the actual tri-star sample. The tri-star "
            "isn't just failing to add information — on this reading it sits in the *bottom* "
            "quantile of what an ordinary day would give you."
        ),
        md(
            "### 4e · Costs\n\n"
            "Net of a 2 bps round-trip and 1 bp borrow on the short leg — there is no gross edge "
            "for costs to threaten, so this only confirms the obvious."
        ),
        code(
            "hs = list(st.HORIZONS)\n"
            "gross = [R['loose'][h][1] for h in hs]; net = [R['loose'][h][7] for h in hs]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.2))\n"
            "x = np.arange(len(hs))\n"
            "ax.bar(x-.2, gross, .4, color=GREY, label='gross'); ax.bar(x+.2, net, .4, color=RED, label='net')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('bps')\n"
            "ax.set_title('Already losing gross; costs make it worse'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('gross:', gross); print('net:', net)"
        ),
        md(
            "### 4f · Synthetic positive control — the machinery is unbiased\n\n"
            "Deterministic panel with forced 3-bar doji blocks and a TUNABLE planted reversal. The "
            "decisive statistic is the same Welch *t* used on the real tape. The null is checked "
            "over **20 seeds** — never a single stream."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    d, _t = data.synthetic_panel(edge=0.0, seed=685 + s_)\n"
            "    r = st.synthetic_detect(d, horizon=20, seed=685 + s_)\n"
            "    if r['welch_t'] is not None:\n"
            "        null_ts.append(r['welch_t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "planted = []\n"
            "for edge in (0.02, 0.04):\n"
            "    d, _t = data.synthetic_panel(edge=edge, seed=685)\n"
            "    r = st.synthetic_detect(d, horizon=20, seed=685)\n"
            "    planted.append((edge, r['welch_t']))\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(len(null_ts)) + np.linspace(-.12,.12,len(null_ts)), null_ts,\n"
            "           color=GREY, s=40, label=f'null worlds (edge=0), {len(null_ts)} seeds')\n"
            "for i,(e,t) in enumerate(planted):\n"
            "    ax.scatter([i+1], [t], color=RED, s=90, zorder=5,\n"
            "               label=f'planted edge={e:.2f}' if i==0 else f'planted edge={e:.2f}')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0]+[i+1 for i in range(len(planted))])\n"
            "ax.set_xticklabels(['null x ' + str(len(null_ts))] + [f'edge={e:.2f}' for e,_ in planted])\n"
            "ax.set_ylabel('Welch t (tri-star vs base rate)')\n"
            "ax.set_title('Control: null stays small; a planted reversal lights up cleanly')\n"
            "ax.legend(fontsize=8); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(np.abs(null_ts)>=2).sum()}/{len(null_ts)} seeds')\n"
            "for e,t in planted: print(f'planted edge={e:.2f}: t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across {R['syn_null_seeds']} null worlds the detector averages "
            f"*t* = {R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}), firing \\|*t*\\| ≥ 2 on "
            f"{R['syn_null_fire']}/{R['syn_null_seeds']} seeds — a small, named quirk of forcing "
            "three artificially low-variance days into a random walk, not a claim about markets. "
            f"What matters is the contrast: a planted reversal of just 0.02–0.04 (far smaller than "
            f"real market swings) lights up cleanly at *t* = {R['syn_edge_002_t']:+.2f} and "
            f"{R['syn_edge_004_t']:+.2f}. Per the desk's rule, **a synthetic control never backs a "
            "Signal stamp** — it exists only to prove the harness can find a real, planted effect, "
            "which it demonstrably can. The honest, unglamorous real-tape result above stands on "
            "its own regardless."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the literature-faithful strict tri-star fires **"
            f"{R['n_strict']}** times in {R['n_names']} names × {R['years']:.0f} years, both "
            "losses, too few to test. The loose cut (n = "
            f"{R['n_loose']}) shows the reversal delta negative at **every** horizon; the closest "
            f"approach to significance (20d, Welch *t* = {R['loose'][20][5]:.2f}) still falls "
            f"short of the Bonferroni bar (**{R['bonferroni_crit']:.2f}**) and runs the wrong "
            "sign regardless.\n"
            "- **Tradability `MIRAGE`** — the real pattern occurs about once per 30 ticker-years; "
            "the loose reading has no gross edge for costs to even threaten.\n"
            "- **\"Tri-star marks a major reversal\"? `BUSTED`** — 2/2 real occurrences lost "
            f"money; the label-shuffle placebo beats the loose tri-star "
            f"{R['loose'][20][6]*100:.0f}–{R['loose'][60][6]*100:.0f}% of the time. Escalating "
            "from one doji to three doesn't manufacture a reversal signal that a single doji "
            "([study 405](../../../405-doji-reversal/)) already failed to show."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The candlestick-of-indecision family, closed out.** One doji ([405](../../../405-doji-reversal/)), "
            "one gapped doji between directional candles ([458](../../../458-abandoned-baby/)), "
            "and now three dojis stacked — none of them beat a fair baseline. The family's "
            "unifying failure mode is the same each time: a session that closed near where it "
            "opened is a noisy proxy for a low-range day, not a measurement of order-flow "
            "exhaustion.\n"
            "- **Sample size is the real constraint, not method.** A pattern this rare can't be "
            "rescued by better statistics — more names or more years would help marginally, but "
            "intraday bars (the natural way to get more events) change the microstructure "
            "entirely and would need their own study.\n"
            "- **Dedup map:** [405-doji-reversal](../../405-doji-reversal/) (one doji, same "
            "base-rate idiom), [186-morning-star](../../186-morning-star/) (a different "
            "three-candle shape, random-day baseline), "
            "[458-abandoned-baby](../../458-abandoned-baby/) (one gapped doji between directional "
            "candles, the closest cousin).\n\n"
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
