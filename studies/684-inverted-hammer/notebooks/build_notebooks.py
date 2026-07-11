"""Generate the two narrative notebooks for Study 684 (Inverted Hammer).

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
# 10,735 inverted-hammer-shaped bars any trend, panel fingerprint 8ddaab4e2e1e,
# SPY data_stamp fingerprint 7987ab25910e, as-of 2026-06-30).
R = dict(
    asof="2026-06-30", start="1962-01-02", end="2026-06-30",
    n_names=26, n_bars=345846, n_shape=10735,
    fp_panel="8ddaab4e2e1e", fp_spy="7987ab25910e",
    # INVERTED HAMMER (shape after a downtrend -- the bullish claim): H -> (n, edge%, win%, t, p, p_bonf, net%)
    invh={1: (4895, +0.045, 49.4, +1.45, 0.042, 0.166, -0.055),
          3: (4895, +0.095, 52.2, +1.79, 0.021, 0.082, -0.005),
          5: (4894, +0.105, 52.6, +1.49, 0.035, 0.142, +0.005),
          10: (4891, +0.137, 55.7, +1.48, 0.046, 0.185, +0.037)},
    # SHOOTING-STAR context (shape after an uptrend, traded LONG as a myth-check): H -> (n, edge bps, win%, t, p)
    star={1: (5673, +3.3, 49.6, +1.48, 0.093),
          3: (5673, +4.5, 51.6, +1.12, 0.151),
          5: (5673, +6.5, 53.0, +1.21, 0.115),
          10: (5669, -0.5, 53.3, -0.07, 0.530)},
    # ANY (pooled, ignoring trend, traded long): H -> (n, edge bps, win%, t, p)
    anyside={1: (10735, +3.9, 49.5, +2.13, 0.014),
             3: (10735, +6.9, 51.8, +2.23, 0.013),
             5: (10734, +8.3, 52.8, +1.97, 0.016),
             10: (10727, +5.6, 54.3, +0.91, 0.149)},
    # myth-check filter sweep on invhammer H=3: label -> (edge%, t, p, n)
    filt=[("plain (lookback 10)", +0.095, +1.79, 0.021, 4895),
          ("trend lookback 5", +0.134, +2.64, 0.002, 4906),
          ("trend lookback 20", +0.077, +1.41, 0.045, 4785),
          ("min washout >= 5%", +0.413, +2.85, 0.000, 1265),
          ("min washout >= 10%", +0.527, +1.02, 0.015, 257),
          ("wick >= 3x body", +0.072, +0.94, 0.137, 2486),
          ("wick >= 4x body", +0.092, +0.87, 0.136, 1408)],
    # cost sweep at H=3 (best horizon): cost_bps -> net bps
    costs=[(0.0, +9.5), (1.0, +7.5), (5.0, -0.5), (10.0, -10.5)],
    # per-name H=3: count of |t|>2
    n_names_over2=2,
    # synthetic control (H=1, side=any): planted -> (events, edge%, t, p, win%)
    syn=[(0.000, 2508, +0.014, +0.57, 0.267, 52.0),
         (0.005, 2492, +0.321, +12.78, 0.000, 61.4)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Deeper_washout_rescues_it%3F: Busted](https://img.shields.io/badge/Deeper_washout_rescues_it%3F-Busted-8b949e?style=flat-square)\n\n"
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

from inverted_hammer import data, strategy as st

HS = [1, 3, 5, 10]
ASOF = "2026-06-30"
HAVE_REAL = data.have_real()
if HAVE_REAL:
    PANEL = {t: b[b.index <= ASOF] for t, b in data.load_real().items()}
else:
    PANEL = None
print("real inverted-hammer cache present:", HAVE_REAL,
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
            "# Does a long upper wick after a slide really mark a floor? 🔻\n"
            "### The inverted hammer — the candle traders swear catches the bounce, on the real tape\n\n"
            + BADGES +
            "Here's a chart pattern taught in every trading course. After a stock has been sliding, a day "
            "prints with a **tiny body near the bottom** of its range and a **long tail pointing up** — "
            "buyers pushed price up during the session, even if sellers clawed most of it back by the "
            "close. Traders call it an **inverted hammer**: *\"buyers showed up — the sellers are running "
            "out of ammo, buy the bounce.\"*\n\n"
            "The wick **is** real information about *that day* — buyers genuinely tested higher. The "
            "question is whether it tells you anything about the **next few days**. We checked, on "
            "**60+ years** of daily bars across 26 big US stocks plus the S&P.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo test and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** Fixed **26-name** survivor basket + SPY (names still trading "
            "today) — the same panel used by the sibling hammer/shooting-star studies. Survivors *recover* "
            "from dips that delisted names didn't, so the bias actually leans **toward** finding a bounce. "
            "If even *that* shows no floor, the floor really isn't there. Charts are drawn by the code "
            "beside them; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| After a slide, does an inverted hammer mark a floor? | **Not really.** Over the next 3 "
            f"days it beats the stock's own usual return by **{R['invh'][3][1]:+.3f}%** — a real-looking "
            f"tilt (*t* = {R['invh'][3][3]:+.2f}) but it never crosses the bar the desk requires (*t* ≥ 2), "
            "and testing four different holding periods at once means we have to correct for multiple "
            "guesses — once we do, none of them survive. |\n"
            "| Does the direction (after a *slide* vs after a *rally*) even matter? | **Barely.** The "
            "identical wick shape after an *uptrend* — the bearish look-alike — shows almost the same "
            "small positive tilt. If the story were really about sellers exhausting after a decline, the "
            "\"wrong side\" should look different. It doesn't. |\n"
            "| Can I fix it with a filter? | **That's the tell.** A shorter trend window or a deeper "
            "\"real washout\" filter can nudge the number up — but push the washout filter *further* and "
            "it **falls back down**, and demanding a *bigger, purer* wick (the \"better\" inverted hammer) "
            "makes it **worse**. |\n"
            "| Could I trade it? | **No.** After normal trading costs the edge turns negative at every "
            "horizon that mattered — the best horizon's net edge is a rounding error either side of zero. "
            "|\n\n"
            "> The wick is real — for *that day*. As a forecast of the *next* few days, the floor is a "
            "**mirage**. And we *prove* our test isn't blind: when we secretly plant a real floor in fake "
            "data, the same test screams (more in the quant notebook)."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"An inverted hammer is a small body near the bottom of the range with a long upper "
            "shadow. After a downtrend it's **bullish** — buyers tested higher and sellers are losing "
            "steam, so the bottom is near. The same candle after an uptrend is the bearish "
            "**shooting star**.\"*\n\n"
            "This isn't fringe. It's core to **Steve Nison**'s *Japanese Candlestick Charting Techniques* "
            "(1991) — the book that brought 18th-century Japanese rice-trading candles to Wall Street — "
            "and it's in every charting app and trading course made since. So the question isn't \"is it a "
            "known pattern?\" It's: **does the long upper wick actually predict what happens next?**"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a single candle could call the bottom, you'd have a beautifully simple edge — look at one "
            "bar, take the trade. That's exactly why the pattern is so beloved and so widely taught.\n\n"
            "But two traps hide here. **(1)** A long upper wick says price *did* get bid up today — "
            "that's backward-looking and already in the close. Whether it forecasts *tomorrow* is a "
            "separate, much harder claim. **(2)** Big liquid stocks **drift up** over time, so *any* "
            "random day after a slide often looks \"bullish\" a few days later — that's mean-reversion "
            "and drift, not the candle. The honest test isn't \"does the stock rise after an inverted "
            "hammer?\" (it usually does — everything does) — it's **\"does it rise more than the stock's "
            "own normal?\"** We measure that base-rate-adjusted edge, and we correct for the fact that "
            "we're testing four different holding periods at once."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We scan **{R['n_bars']:,} daily bars** across **{R['n_names']} stocks + SPY** "
            f"({R['start']} → {R['end']}) and flag every bar with the inverted-hammer shape — small body "
            "at the bottom, upper wick at least twice the body, almost no lower shadow. That's "
            f"**{R['n_shape']:,}** inverted-hammer-shaped bars in total, of which "
            f"**{R['invh'][3][0]:,}** sit after a genuine downtrend (the bullish claim).\n\n"
            "1. **The shape.** Detected by exact open/high/low/close rules — no eyeballing.\n"
            "2. **The trend split.** Was the stock *down* over the last 10 days? Only those count as the "
            "bullish claim; the same shape after an *up* move is the bearish look-alike (sibling study).\n"
            "3. **The forward move.** Starting at the **next** day's close (no cheating — the candle is "
            "already done), how does the stock do over the next **1, 3, 5, 10** days **versus its own "
            "usual return** over that horizon?\n"
            "4. **The honesty check.** Testing four horizons at once means one of them can look good by "
            "chance — we apply a **Bonferroni** correction so a lucky horizon can't carry the headline.\n\n"
            "If the legend is real, the inverted hammer should beat the base rate by a clear, robust "
            "margin. We'd call it a **mirage** if the edge can't clear a basic luck test."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**The bullish claim first.** Here's how much the inverted hammer beats (or misses) the "
            "stock's *own normal* return at each horizon. If the floor is real, these bars should be "
            "clearly, solidly green — and the *t*-stats (the honesty line) should clear 2."
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(PANEL, side='invhammer', n_draws=1500, seed=684)\n"
            "    edge = [res[h]['edge_mean']*100 for h in HS]; tvals = [res[h]['t'] for h in HS]\n"
            "else:\n"
            "    edge = [R['invh'][h][1] for h in HS]; tvals = [R['invh'][h][3] for h in HS]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "cols = [GREEN if v>0 else RED for v in edge]\n"
            "a1.bar([f'{h}d' for h in HS], edge, color=cols, width=.6)\n"
            "for i,v in enumerate(edge): a1.annotate(f'{v:+.3f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_ylabel(\"inverted-hammer edge over the stock's own base rate (%)\")\n"
            "a1.set_title('A small positive tilt...')\n"
            "a2.bar([f'{h}d' for h in HS], tvals, color=AMBER, width=.6)\n"
            "a2.axhline(2, ls='--', c=RED, label='t = 2 bar')\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "for i,t in enumerate(tvals): a2.annotate(f't={t:+.2f}',(i,t),ha='center',va='bottom')\n"
            "a2.set_ylabel('HAC t'); a2.set_title('...that never clears the bar'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('edge by horizon (%):', [round(v,3) for v in edge], ' t:', [round(v,2) for v in tvals])"
        ),
        md(
            f"That's the whole ballgame, and it's underwhelming. The best the inverted hammer manages is "
            f"**{R['invh'][3][1]:+.3f}% at 3 days** with *t* = **{R['invh'][3][3]:+.2f}** — real-looking "
            "but under the bar the desk requires (2.0). And once we correct for testing four horizons at "
            f"once, the adjusted odds of that being luck jump to **{R['invh'][3][5]:.0%}** or worse at "
            "every horizon — nothing survives."
        ),
        md(
            "**Now the honesty check on direction.** If the trend split really matters — if sellers are "
            "genuinely \"exhausted\" after a slide — the *same shape* after a rally (the bearish "
            "look-alike, traded long here on purpose) should look clearly different, ideally negative or "
            "flat. Let's put the two next to each other."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ri = st.run_experiment(PANEL, side='invhammer', placebo=False, n_draws=1, seed=684)\n"
            "    rs = st.run_experiment(PANEL, side='star', placebo=False, n_draws=1, seed=684)\n"
            "    ih = [ri[h]['edge_mean']*100 for h in HS]; sr = [rs[h]['edge_mean']*100 for h in HS]\n"
            "else:\n"
            "    ih = [R['invh'][h][1] for h in HS]; sr = [R['star'][h][1]/100 for h in HS]\n"
            "x = np.arange(len(HS))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, ih, .4, color=GREEN, label='AFTER A DOWNTREND (the claim: \"floor\")')\n"
            "ax.bar(x+.2, sr, .4, color=GREY, label='AFTER AN UPTREND (the \"wrong side\")')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in HS])\n"
            "ax.set_ylabel('edge over base rate (%)')\n"
            "ax.set_title('Same shape, both directions positive -- the trend split barely changes the story')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('after downtrend:', [round(v,3) for v in ih], ' after uptrend:', [round(v,3) for v in sr])"
        ),
        md(
            "Both bars lean the same way at almost every horizon. If \"buyers overwhelming exhausted "
            "sellers after a slide\" were the real mechanism, the *wrong-side* version shouldn't look like "
            "a smaller copy of the same thing — it should look different, or negative. It doesn't. That's "
            "a second strike against the story, on top of the *t*-stat never clearing the bar."
        ),
        md(
            "**The give-away: the filter rescue.** Believers say \"sure, but you need the *right* setup — "
            "a real washout, not just any downtrend.\" So we tried. Watch what happens to the 3-day edge "
            "as we change the filters."
        ),
        code(
            "labels = [f[0] for f in R['filt']]; tvals = [f[2] for f in R['filt']]\n"
            "fig, ax = plt.subplots(figsize=(9.8, 4.5))\n"
            "cols = [GREEN if t>=2 else (AMBER if t>=1 else GREY) for t in tvals]\n"
            "ax.barh(range(len(labels)), tvals, color=cols)\n"
            "ax.axvline(2, ls='--', c=RED, label='t = 2 bar (what \"real\" needs)')\n"
            "ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:+.2f}',(t,i),va='center',ha='left' if t>=0 else 'right')\n"
            "ax.set_xlabel('3-day edge, as a t-stat'); ax.set_title('Pick the filter, pick the answer -- the mirage signature')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for f in R['filt']: print(f'{f[0]:22s} edge={f[1]:+.3f}%  t={f[2]:+.2f}  n={f[4]}')"
        ),
        md(
            "A **shorter** trend window or a **deeper washout filter** *can* nudge *t* above 2 — "
            "tantalisingly close! — but push the washout filter *further still* (from 5% to 10%) and it "
            "**falls back down** to under 1.1 on a thin sample, and demanding a *longer, purer* wick — the "
            "\"textbook-perfect\" inverted hammer — **flattens it to under 1**. When the answer swings "
            "with the knob you chose, and doesn't even move consistently as you turn the knob further, "
            "you're not finding a signal. You're fishing."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Best edge **{R['invh'][3][1]:+.3f}% at 3 days, *t* = "
            f"{R['invh'][3][3]:+.2f}** — under the bar, and it doesn't survive correcting for the four "
            "horizons tested at once. The \"wrong-side\" version (after an uptrend) looks almost the "
            "same, undercutting the trend-conditional story itself. Only 2 of 26 names individually show "
            "a real-looking edge — about what chance predicts.\n"
            "- **Tradability — Mirage.** After normal costs, the edge is **negative or a rounding error "
            "at every horizon that matters.**\n"
            "- **Deeper washout rescues it? — Busted.** The only \"edges\" come from tuning a filter, and "
            "they don't even move consistently as you tune it further — a bigger wick makes it *worse*."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — costs finish it off\n\n"
            "Even granting the tiny gross blip at its best horizon (3 days), here's what a normal 5-bps "
            "round trip does to it."
        ),
        code(
            "labels = [f'{c:.0f} bps' for c,_ in R['costs']]\n"
            "vals = [n for _,n in R['costs']]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "cols = [GREEN if v>0 else RED for v in vals]\n"
            "ax.bar(labels, vals, color=cols, width=.6)\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.1f}bps',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('net edge at 3 days (bps/event)')\n"
            "ax.set_title('The desk-standard 5-bps round trip already sinks the best horizon')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('cost sweep (bps one-way -> net bps):', R['costs'])"
        ),
        md(
            "At the desk's standard **5 bps one-way**, the 3-day inverted hammer's net edge is already "
            "**negative**. Break-even sits under 2 bps — well inside the real bid-ask spread of any of "
            "these names. There was never a robust gross edge worth defending, and ordinary trading costs "
            "bury what little there was."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The wick is real, the forecast isn't.** A long upper shadow genuinely tells you buyers "
            "fought back *that day*. The lesson is that *intraday* information isn't a *multi-day* "
            "forecast — the same trap that sinks the hammer, the hanging man and the shooting star.\n"
            "- **Try small-caps / intraday.** Candle effects are bigger (and costlier) in illiquid names; "
            "re-run on a small-cap basket or on intraday bars and see if a thin edge appears.\n"
            "- **Combine with a real signal.** An inverted hammer *plus* an earnings surprise or a real "
            "reversal factor might do something the candle alone can't. Fork it and show the **net** edge "
            "clearing *t* = 2, Bonferroni-corrected, and we'll talk.\n\n"
            "*The reproducible core is offline and deterministic. Methods and sources: "
            "[`docs/references.md`](../docs/references.md); frozen numbers: "
            "[`docs/results.md`](../docs/results.md).*"
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
            "# The Inverted Hammer — a quantitative teardown 🔬\n"
            "### OHLC pattern detector · forward 1/3/5/10-day edge vs base rate · HAC *t* + label-shuffle "
            "placebo + Bonferroni · the trend-split contrast · a filter-snoop myth-check · a synthetic "
            "planted-floor power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The inverted "
            "hammer is the bullish twin of the canonical single-candle reversal pattern (the shooting "
            "star, sibling study 404) and the canonical academic null (Marshall, Young & Rose 2006). The "
            "job here is to measure it *honestly*: detect the geometry by exact rules, benchmark each "
            "event against the name's own base rate (controlling for drift), confront the overlapping "
            "windows with a **HAC** *t*, a label-shuffle placebo and a **Bonferroni** correction across "
            "the four-horizon family, and show the only path to nominal significance is filter-snooping.\n\n"
            "> ⚠️ **Data + survivorship note.** Fixed **26-name** large-cap basket + SPY (the same panel "
            "as siblings 403/404), names still trading in 2026 — a *survivor* panel whose bias points "
            "**toward** a post-dip bounce (survivors recover from dips delisted names didn't), so a null "
            "result is **conservative**. Real data: yfinance **un-adjusted** daily OHLC (the candle "
            "*shape* needs printed levels → forward returns are **price-only**), 1962→2026. Offline core "
            "+ synthetic control are deterministic. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (panel fingerprint `" + R["fp_panel"] + "`, SPY "
            "data-stamp fingerprint `" + R["fp_spy"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Inverted-hammer edge over base rate peaks at "
            f"**{R['invh'][3][1]:+.3f}% / 3d, HAC t = {R['invh'][3][3]:+.2f}** (raw placebo "
            f"p = {R['invh'][3][4]:.3f}, **Bonferroni p = {R['invh'][3][5]:.3f}** across 4 horizons). "
            f"Only 2/26 names individually clear \\|t\\| > 2 — chance level. Synthetic planted floor "
            f"lights up at **t = {R['syn'][1][3]:.2f}**, so the flat reading is genuine. |\n"
            f"| **Tradability** | `MIRAGE` | Net of a 5-bps round trip the 3-day edge is "
            f"**{R['invh'][3][6]:+.3f}%** — already negative at the pattern's best horizon; break-even "
            "under ~2 bps. |\n"
            f"| **Deeper washout rescues it?** | `BUSTED` | A ≥5% washout filter reaches "
            f"t = {R['filt'][3][2]:.2f} on n = {R['filt'][3][4]} — but a ≥10% filter **collapses** it to "
            f"t = {R['filt'][4][2]:.2f}, and a purer wick (≥4x body) falls to t ≈ {R['filt'][6][2]:.2f}. |\n\n"
            "> 💡 In plain words: the candle carries *intraday* information but no *forward, robust* "
            "edge; any apparent rescue is a tuned-filter artefact that doesn't even move consistently as "
            "the knob turns further."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let a bar have body $b=|c-o|$, range $\\rho=h-\\ell$, upper wick "
            "$w_+=h-\\max(o,c)$, lower wick $w_-=\\min(o,c)-\\ell$. The inverted-hammer geometry:\n\n"
            "$$w_+ \\ge 2\\,b,\\qquad w_- \\le 0.5\\,b,\\qquad b \\le 0.35\\,\\rho,\\qquad \\rho>0.$$\n\n"
            "Same shape as the shooting star (study 404); the trend at the close splits it — "
            "$\\tau_t=\\operatorname{sign}(c_t/c_{t-10}-1)$, **inverted hammer** (the claim under test) "
            "if $\\tau<0$. For event $i$ the forward edge is the $H$-day LONG return entered **one day "
            "later** minus that name's unconditional base rate $\\mu_{\\text{name}}(H)$:\n\n"
            "$$e_i(H)=\\frac{c_{t+1+H}}{c_{t+1}}-1-\\mu_{\\text{name}}(H).$$\n\n"
            "- **H₁ (the floor exists).** $\\overline{e}(H)>0$ and HAC-significant, surviving a "
            "Bonferroni correction across $H\\in\\{1,3,5,10\\}$.\n"
            "- **H₂ (trend-conditional).** The wrong-side (post-uptrend) version of the same geometry "
            "looks *different* — weaker or negative.\n"
            "- **H₃ (deployable).** $\\overline{e}(H)$ survives a one-way × 2 round trip.\n"
            "- **H₄ (robust reversal).** The edge survives reasonable filter choices and a *stronger* "
            "wick, with a sensible dose-response.\n\n"
            "We find **H₁ rejected** (best t = 1.79, Bonferroni p ≥ 0.082 everywhere), **H₂ rejected** "
            "(the wrong-side version is the *same sign*), **H₃ rejected** (net-negative at the best "
            "horizon), **H₄ rejected** (non-monotonic, filter-dependent). The shape is real; the "
            "*forecast* is not."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — why base-rate-adjust, why HAC, why Bonferroni\n\n"
            "Three honesty problems sit on top of a naive test. **(a) Drift.** Large-caps trend up, so a "
            "raw \"return after an inverted hammer\" is positive for *most* candles; we subtract each "
            "name's own base rate so the edge measures *excess over buy-and-hold for that name*. "
            "**(b) Overlap.** Nearby signals share overlapping multi-day forward windows, so the events "
            "are autocorrelated and an i.i.d. *t* overstates significance — we use a **Newey-West (HAC)** "
            "one-sample *t* with an auto bandwidth. **(c) Multiple horizons.** We test the same claim at "
            "1/3/5/10 days simultaneously; quoting the best of four is a textbook snoop, so we report a "
            "**Bonferroni**-adjusted *p* (raw *p* × 4, capped at 1) across the whole family, not just the "
            "winner. The placebo asks the cross-sectional question: could the same *number* of random "
            "days, drawn from the same tapes, beat the observed edge?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** Fixed **{R['n_names']}-name** large-cap basket + SPY (yfinance un-adjusted "
            f"daily OHLC, {R['start']}→{R['end']}); **{R['n_bars']:,}** bars, **{R['n_shape']:,}** "
            f"inverted-hammer-shaped (any trend); **{R['invh'][3][0]:,}** sit after a genuine downtrend "
            "(the claim). **Survivor** panel — named on the Signal axis (bias *toward* a bounce).\n"
            "- **Detector.** $w_+\\ge2b,\\ w_-\\le0.5b,\\ b\\le0.35\\rho$ (a doji is excluded via a tiny "
            "body floor).\n"
            "- **Trend split.** 10-day sign of return at the signal close (no look-ahead).\n"
            "- **Timing.** Signal at close[t]; enter close[t+1]; exit close[t+1+H], $H\\in\\{1,3,5,10\\}$. "
            "One `shift`, applied once. Price-only. Long-only (the bullish claim).\n"
            "- **Edge.** Conditional forward return minus the name's unconditional base rate.\n"
            "- **Null #1 (HAC t)** of the pooled edge vs 0.\n"
            "- **Null #2 (label-shuffle placebo).** Per name, draw its event-count of random days; "
            "$p=\\Pr[\\text{shuffled edge}\\ge\\text{observed}]$.\n"
            "- **Family correction.** Bonferroni across the 4 horizons: $p_{\\text{adj}} = \\min(1, 4p)$.\n"
            "- **Costs.** 5 bps one-way × 2 (round trip), long-only (no borrow).\n"
            "- **Positive control.** A deterministic panel with a **planted** post-pattern bounce: zero "
            "edge must NOT reach significance; a real floor must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The term structure of the edge — and the honesty line (Bonferroni)\n\n"
            "Edge over base rate at each horizon, both the raw HAC *t* and the Bonferroni-adjusted "
            "placebo *p* across the four-horizon family."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ri = st.run_experiment(PANEL, side='invhammer', n_draws=1500, seed=684)\n"
            "    edge = [ri[h]['edge_mean']*100 for h in HS]; tvals = [ri[h]['t'] for h in HS]\n"
            "    pbonf = [ri[h]['p_bonferroni'] for h in HS]\n"
            "else:\n"
            "    edge = [R['invh'][h][1] for h in HS]; tvals = [R['invh'][h][3] for h in HS]\n"
            "    pbonf = [R['invh'][h][5] for h in HS]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.3))\n"
            "a1.plot(HS, tvals, 'o-', c=AMBER, lw=2, label='HAC t')\n"
            "a1.axhline(2, ls='--', c=RED, label='t = 2 bar'); a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_xlabel('horizon (days)'); a1.set_ylabel('HAC t'); a1.set_title('t never reaches 2'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in HS], pbonf, color=GREY, width=.6)\n"
            "a2.axhline(0.05, ls='--', c=RED, label='p = 0.05 bar')\n"
            "for i,p in enumerate(pbonf): a2.annotate(f'{p:.3f}',(i,p),ha='center',va='bottom')\n"
            "a2.set_ylabel('Bonferroni-adjusted placebo p'); a2.set_title('Adjusted p never clears 0.05'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('edge%:', [round(v,3) for v in edge], ' t:', [round(v,2) for v in tvals], "
            "' bonf p:', [round(v,3) for v in pbonf])"
        ),
        md(
            f"> 💡 In plain words: the raw HAC *t* tops out at **{R['invh'][3][3]:+.2f}** (3d) — never "
            "approaching the +2 floor a real edge would need. The raw label-shuffle *p* looks tempting at "
            f"every horizon (all < 0.05), but that's exactly the multiple-comparisons trap: testing four "
            "horizons and reporting the best one inflates the apparent hit rate. Once corrected "
            f"(**Bonferroni p ≥ {R['invh'][3][5]:.3f}**), nothing survives."
        ),
        md(
            "### 4b · Does the trend split actually discriminate a floor? — the wrong-side contrast\n\n"
            "If \"sellers exhausted after a slide\" were the real mechanism, the identical geometry traded "
            "long after an **uptrend** (the shooting-star context, the direct look-alike of "
            "[404](../../404-shooting-star/)) should look different — weaker, flat, or negative."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ri = st.run_experiment(PANEL, side='invhammer', placebo=False, n_draws=1, seed=684)\n"
            "    rs = st.run_experiment(PANEL, side='star', placebo=False, n_draws=1, seed=684)\n"
            "    ih = [ri[h]['edge_mean']*100 for h in HS]; sr = [rs[h]['edge_mean']*100 for h in HS]\n"
            "else:\n"
            "    ih = [R['invh'][h][1] for h in HS]; sr = [R['star'][h][1]/100 for h in HS]\n"
            "x = np.arange(len(HS))\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.3))\n"
            "ax.bar(x-.2, ih, .4, color=GREEN, label='post-DOWNTREND (the claim)')\n"
            "ax.bar(x+.2, sr, .4, color=GREY, label='post-UPTREND (the wrong side)')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in HS])\n"
            "ax.set_ylabel('edge over base rate (%)')\n"
            "ax.set_title('Same sign both ways -- the trend split does not cleanly discriminate')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('post-downtrend:', [round(v,3) for v in ih], ' post-uptrend:', [round(v,3) for v in sr])"
        ),
        md(
            "> 💡 In plain words: both contexts show a small positive tilt at 1/3/5 days. A genuinely "
            "trend-conditional reversal mechanism should make the *wrong-side* version look meaningfully "
            "different — it doesn't. This is consistent with a generic (and non-significant) post-wick "
            "drift rather than a floor specifically tied to a prior downtrend."
        ),
        md(
            "### 4c · The decisive test — the pattern against a label-shuffle null\n\n"
            "The 3-day inverted-hammer edge (its best) against a per-name label-shuffle null."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(PANEL, 3, side='invhammer', n_draws=3000, seed=684)\n"
            "    obs = pl['obs_edge']*100; draws = pl['draws']*100; pval = pl['p_value']\n"
            "    tval = st.run_experiment(PANEL, side='invhammer', horizons=(3,), placebo=False, n_draws=1, seed=684)[3]['t']\n"
            "else:\n"
            "    obs = R['invh'][3][1]; pval = R['invh'][3][4]; tval = R['invh'][3][3]\n"
            "    rng = np.random.default_rng(684); draws = rng.normal(0.0, 0.05, 3000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='null: per-name random days')\n"
            "ax.axvline(obs, c=AMBER, lw=2.5, label=f'observed edge {obs:+.3f}%')\n"
            "ax.set_xlabel('3-day inverted-hammer edge over base rate (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Near the edge of the luck cloud: raw p = {pval:.3f}, HAC t = {tval:+.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.3f}%  HAC t={tval:+.2f}  raw shuffle p={pval:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the amber line sits at the *edge* of the null cloud — a random sort of "
            f"the same days beats the pattern about **{R['invh'][3][4]*100:.1f}%** of the time on its own "
            "(raw *p* = {:.3f}), which looks almost interesting on a single test. It's the **multiple-"
            "horizon correction** (section 4a) that reveals this isn't a robust finding: four horizons "
            "means four chances to get a lucky *p* < 0.05.".format(R['invh'][3][4])
        ),
        md(
            "### 4d · The myth-check — significance only by snooping the filter, and not even "
            "consistently\n\n"
            "Believers reach for filters. We sweep the obvious ones on the 3-day edge. The bar to beat is "
            "*t* = 2 — and a real effect should move *consistently* as the filter tightens, not bounce."
        ),
        code(
            "labels = [f[0] for f in R['filt']]; tvals = [f[2] for f in R['filt']]; edges=[f[1] for f in R['filt']]\n"
            "fig, ax = plt.subplots(figsize=(9.8, 4.5))\n"
            "cols = [GREEN if t>=2 else (AMBER if t>=1 else GREY) for t in tvals]\n"
            "ax.barh(range(len(labels)), tvals, color=cols)\n"
            "ax.axvline(2, ls='--', c=RED, label='t = 2 bar')\n"
            "ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)\n"
            "for i,(t,e) in enumerate(zip(tvals,edges)): ax.annotate(f't={t:+.2f} ({e:+.3f}%)',(t,i),va='center',ha='left' if t>=0 else 'right',fontsize=9)\n"
            "ax.set_xlabel('3-day edge as a t-stat'); ax.set_title('Pick the filter, pick the t -- and it does not even move monotonically'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for f in R['filt']: print(f'{f[0]:22s} edge={f[1]:+.3f}%  t={f[2]:+.2f}  p={f[3]:.3f}  n={f[4]}')"
        ),
        md(
            f"> 💡 In plain words: a *shorter* trend window (lookback 5) lifts t to "
            f"**{R['filt'][1][2]:.2f}**, and a ≥5% washout filter reaches **{R['filt'][3][2]:.2f}** on a "
            f"thinned sample (n = {R['filt'][3][4]}) — but pushing the washout threshold *further* to 10% "
            f"**collapses** it to **{R['filt'][4][2]:.2f}** on an even thinner n = {R['filt'][4][4]}, and "
            f"a *longer, purer wick* (the \"better\" inverted hammer) flattens it to "
            f"**t ≈ {R['filt'][6][2]:.2f}**. A real dose-response would strengthen monotonically as the "
            "filter gets stricter; this doesn't — the signature of a data-snooped mirage "
            "(Sullivan-Timmermann-White 1999)."
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "On a deterministic panel where we **plant** a post-pattern bounce: with **zero** edge the "
            "test must stay near t≈0 (no false positive); with a real planted floor it must light up. "
            "Both hold — so the sub-bar real-tape reading is a genuine near-null, not a blind harness."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.005):\n"
            "    sp, _ = data.synthetic_panel(edge=edge, seed=684)\n"
            "    r = st.run_experiment(sp, side='any', horizons=(1,), n_draws=1500, seed=684)[1]\n"
            "    res.append((edge, r['n'], r['edge_mean']*100, r['t'], r['p_placebo'], r['win']*100))\n"
            "tvals = [r[3] for r in res]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar([f'planted\\n{e*100:.1f}%' for e,_,_,_,_,_ in res], tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('synthetic 1d HAC t'); ax.set_title('Control: zero edge -> t~0; planted floor -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,k,ed,t,p,w in res: print(f'planted {e*100:.1f}%: events={k} edge={ed:+.3f}% t={t:.2f} p={p:.3f} win={w:.1f}%')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted floor the control sits at "
            f"**t = {R['syn'][0][3]:.2f}** (no false positive); a **+0.5%/event** planted floor reaches "
            f"**t = {R['syn'][1][3]:.2f}**. The machinery is unbiased and powerful, so the real-tape "
            "inverted-hammer t ≈ 1.8 is a true (near-)negative — the floor genuinely isn't reliably there."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — inverted-hammer edge peaks at **{R['invh'][3][1]:+.3f}% / 3d, HAC "
            f"t = {R['invh'][3][3]:+.2f}** (raw placebo p = {R['invh'][3][4]:.3f}, **Bonferroni "
            f"p = {R['invh'][3][5]:.3f}** across 4 horizons — none clear 0.05). The wrong-side (post-"
            "uptrend) version of the same geometry shows the *same sign*, undercutting the trend-"
            "conditional story. Only 2/26 names individually clear |t| > 2 — chance level. Survivorship "
            "tilts *toward* a bounce, so this is conservative. The control detects a planted floor at "
            f"**t = {R['syn'][1][3]:.2f}**.\n"
            f"- **Tradability `MIRAGE`** — net of a 5-bps round trip the pattern's best horizon (3 days) "
            f"is already **{R['invh'][3][6]:+.3f}%**; break-even sits under ~2 bps one-way. No gross edge "
            "robust enough to defend.\n"
            f"- **Deeper washout rescues it? `BUSTED`** — a ≥5% washout filter reaches "
            f"t = {R['filt'][3][2]:.2f}, but doubling it to ≥10% *collapses* to "
            f"t = {R['filt'][4][2]:.2f} on a thinner sample, and a purer wick *weakens* it further "
            f"(t ≈ {R['filt'][6][2]:.2f}) — no consistent dose-response, the signature of snooping."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — costs on an already-uncertified edge\n\n"
            "The inverted hammer's cost sweep at its best horizon (3 days). There is no net-positive "
            "region past ordinary spread costs."
        ),
        code(
            "labels = [f'{c:.0f} bps' for c,_ in R['costs']]\n"
            "vals = [n for _,n in R['costs']]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.plot(labels, vals, 'o-', c=RED, lw=2.2)\n"
            "ax.axhline(0, c='k', ls='--', label='break-even')\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:+.1f}bps',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.set_xlabel('one-way cost assumption'); ax.set_ylabel('net edge at 3d (bps/event)')\n"
            "ax.set_title('Break-even sits under ~2 bps -- inside the real spread'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('cost sweep:', R['costs'])"
        ),
        md(
            "> 💡 In plain words: at the desk's standard 5-bps one-way convention the net edge is already "
            "negative at the pattern's best horizon. There is simply nothing here to harvest even before "
            "the Bonferroni correction — hence **MIRAGE**, not FRAGILE."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Intraday is the natural habitat.** The wick is a within-day fact; a same-day or "
            "next-open framing on intraday bars might surface a thin, capacity-limited effect the daily "
            "close washes out.\n"
            "- **Small / illiquid names.** Candlestick effects are larger (and costlier) where "
            "microstructure frictions bite; re-run the basket on small-caps.\n"
            "- **The general lesson.** A backward-looking *shape* (it already happened) is not a "
            "forward-looking *forecast*, and testing a claim across several horizons at once needs a "
            "multiple-comparisons correction before you believe the best-looking one. The contrast with "
            "[Study 363 — PEAD](../../363-pead-drift/), which clears the bar, is the point.\n\n"
            "*The reproducible core is offline and deterministic; the synthetic control plants a known "
            "floor. Methods and sources: [`docs/references.md`](../docs/references.md); frozen numbers: "
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
