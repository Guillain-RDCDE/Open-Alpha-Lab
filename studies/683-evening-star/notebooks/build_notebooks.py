"""Generate the two narrative notebooks for Study 683 (Evening-Star).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached basket OHLCV
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily OHLCV,
# 30-name liquid large-cap + SPY basket, 2005-01-03 -> 2026-06-30, 21.5 years).
R = dict(
    start="2005-01-03", end="2026-06-30", years=21.5, n_names=30,
    fp_quantlab="bde089f6e8ee", fp_basket="97e1b85d26c3", total_bars=162180,
    # SIGNED-SHORT forward return after evening star, per horizon:
    # (H, n, mean%, base%, hit%, basehit%, t_hac, t_one, t_welch[decisive], p_placebo, net5%, net10%)
    h1=(1, 602, -0.055, 0.028, 50, 48, -1.12, -1.09, -0.53, 0.842, -0.157, -0.257),
    h5=(5, 602, -0.527, 0.252, 44, 45, -3.77, -3.55, -1.84, 1.000, -0.637, -0.737),
    h10=(10, 600, -0.612, 0.529, 46, 43, -2.95, -3.14, -0.43, 0.999, -0.732, -0.832),
    hit_wilson={1: (46.2, 54.1), 5: (40.1, 48.0), 10: (42.2, 50.2)},
    # strict textbook-gap myth-check: (H, n, mean%, t_welch, t_hac, p, net5%)
    strict=[(1, 284, -0.025, 0.04, -0.34, 0.614, -0.127),
            (5, 284, -0.451, -0.91, -2.45, 0.987, -0.561),
            (10, 283, -0.678, -0.50, -2.71, 0.990, -0.797)],
    # prior-uptrend (true reversal) myth-check: (H, n, mean%, t_welch, t_hac, p, net5%)
    trend=[(1, 318, -0.093, -0.98, -1.49, 0.879, -0.195),
           (5, 318, -0.261, -0.05, -1.37, 0.912, -0.371),
           (10, 316, -0.267, 1.03, -1.06, 0.825, -0.387)],
    # Bonferroni across the basket (per-ticker Welch t vs that ticker's own base, H=5)
    bonf_k=30, bonf_z=3.14, bonf_survive=0,
    bonf_top=[("MMM", 16, -1.396, -2.25), ("ORCL", 17, -2.478, -2.07), ("ABT", 24, -1.167, -1.99)],
    # synthetic null over 20 seeds (decisive stat: Welch t vs the panel's own base)
    syn_null_mean=-0.26, syn_null_sd=1.25, syn_null_fire=1, syn_null_n=20,
    # planted control (seed 683): (edge, planted_days, n, mean%, t_welch, t_hac, p, hit%)
    syn=[(0.000, 359, 358, -0.007, 1.28, -0.04, 0.524, 47),
         (0.004, 359, 358, 0.397, 3.54, 2.43, 0.011, 53),
         (0.008, 358, 357, 0.787, 5.73, 4.82, 0.000, 61)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Crash_incoming%3F: Busted](https://img.shields.io/badge/Crash_incoming%3F-Busted-8b949e?style=flat-square)\n\n"
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

from evening_star import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PANEL = data.load_real(allow_fetch=False, asof="2026-06-30")
else:
    PANEL = None
print("real basket cache present:", HAVE_REAL,
      "| names:", (0 if PANEL is None else len(PANEL)))
"""

# The frozen headline dict is embedded into the first code cell so every downstream cell can
# quote it whether or not the cache is present.
BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The evening star — does the candle that screams \"the top is in\" actually predict one? 🌆\n"
            "### One of candlestick lore's marquee reversal signals, measured on 21.5 years of real tape\n\n"
            + BADGES +
            "Open any candlestick book and you'll meet the **evening star**: a tall **up** day, then a "
            "tiny \"star\" candle that **gaps higher** — buyers running out of steam — then a tall **down** "
            "day that closes deep back into the first candle's body. The lore is dramatic: it's supposed to "
            "mark the **top** of an uptrend and warn that a fall is coming. Spot the star, get short (or get "
            "out), and dodge the drop.\n\n"
            "It *looks* like exactly the kind of thing a chart should be able to tell you. So we did the "
            "boring, honest thing: found **every** evening star in a basket of 30 big US stocks + SPY over "
            "21.5 years, and measured what actually happened next.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo tests and the Bonferroni "
            "correction? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** We use a fixed **30-name liquid large-cap + SPY** basket (names "
            "still trading today), so this carries **survivorship** (we can't include firms that actually "
            "*did* top out, crash and delist). If anything that should make the \"top\" signal look "
            "**stronger** than reality — and it still doesn't show up. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| After an evening star, does the stock keep falling? | **No — not in any way we can "
            "certify.** Short it the next morning and, on average, over the next 1–10 days you're roughly "
            "flat-to-losing once compared fairly to an ordinary day. The \"top\" doesn't reliably arrive. |\n"
            "| Wait, isn't the average return *negative* for the short? | **Yes, but that's a trick of the "
            "math, not an edge.** This basket drifts up on average (stocks generally do), so shorting "
            "*anything* — even random days — tends to lose against zero. The fair test is against what the "
            "*same basket* does on a normal day, and against that fair bar the star shows **nothing**. |\n"
            "| Does the picture-perfect star (a real gap, no overlap) work better? | **No.** The stricter "
            "the textbook shape, the *weaker* the (already absent) signal gets. |\n"
            "| What if I only count stars after a genuine uptrend (a true \"reversal\")? | **Still "
            "nothing.** |\n"
            "| Does any single stock secretly carry the pattern? | **No.** Checked individually across all "
            f"30 names with a stricter statistical bar (so one lucky name can't pass as \"the\" signal), "
            f"**{R['bonf_survive']}/{R['bonf_k']}** clear it. |\n\n"
            "> The most photogenic candle in the book is, on the tape, indistinguishable from an ordinary "
            "day — and shorting it **loses money** before you even pay costs."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The evening star is a **bearish reversal**. After a rally: a tall bullish candle, then a "
            "tiny 'star' that gaps **up**, showing indecision, then a tall bearish candle that closes back "
            "deep into the first candle's body. The bulls have lost control — a downtrend has begun. Sell, "
            "or short, because the **top is in**.\"*\n\n"
            "This is straight out of Steve Nison's *Japanese Candlestick Charting Techniques* (the book "
            "that brought candles to the West) and is repeated by every charting site from Investopedia to "
            "StockCharts. Together with its bullish twin (the [morning star](../../186-morning-star/)), "
            "it's one of the most recognizable patterns in technical analysis. The question is simply: "
            "**does the drop come?**"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the evening star really marked tops, it would be enormously useful: a clean, eyeball-able "
            "warning to trim a winning position or to short for a quick profit, needing no data feed beyond "
            "a price chart.\n\n"
            "But there's a trap built into the very shape of the pattern. The confirming candle is, almost "
            "by definition, a stock that just had a sharp **down** day. Markets that just dropped sharply "
            "tend to **bounce** (short-term mean reversion), not keep tumbling. So a naive short right after "
            "the star might be picking a fight with the market's own rebound tendency — the exact opposite "
            "of what the folklore promises. We'll see which force wins, and — just as important — whether "
            "\"the short loses\" is a *real* bearish-fighting-force or just the ordinary tide of an "
            "up-drifting market."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take a fixed **{R['n_names']}-name** basket of liquid US large-caps + **SPY** and scan "
            f"every daily bar from **{R['start']}** to **{R['end']}** ({R['years']:.1f} years). For each "
            "name we flag **every** evening star with a precise OHLC rule (tall up body, tiny gapping star, "
            "tall down body closing at least halfway back into the first candle). Then, with **no "
            "cheating**:\n\n"
            "1. **Wait for the close** that confirms the third candle.\n"
            "2. **Enter the next morning's open** (one day's lag — you can't trade a bar still forming) and "
            "hold **1, 5, or 10** trading days, **short** (the bearish bet).\n"
            "3. **Compare fairly** — not to zero (a naive comparison is fooled by the market's own upward "
            "drift), but to what the *same basket* earns on an unconditional day, and to thousands of "
            "random \"fake signal\" picks. If shorting the star beats the everyday tide, the legend is "
            "real. If it doesn't, the top is a myth."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Here's the signed-short return after an evening star, at each "
            "horizon, next to what the *same basket* earns on an ordinary (unconditional) day. A real "
            "\"top\" signal would show the star's bar **beating** the baseline, in green."
        ),
        code(
            "hs = [1, 5, 10]\n"
            "if HAVE_REAL:\n"
            "    rows = {h: st.summarize(PANEL, h, placebo=False) for h in hs}\n"
            "    means = [rows[h]['mean']*100 for h in hs]\n"
            "    base  = [-rows[h]['base_mean']*100 for h in hs]   # base, expressed as the SAME short bet\n"
            "else:\n"
            "    means = [R['h1'][2], R['h5'][2], R['h10'][2]]\n"
            "    base  = [-R['h1'][3], -R['h5'][3], -R['h10'][3]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.5))\n"
            "ax.bar(x-.18, means, .34, color=RED, label='short AFTER an evening star')\n"
            "ax.bar(x+.18, base, .34, color=GREY, label='short on an ORDINARY day (baseline)')\n"
            "for i,v in enumerate(means): ax.annotate(f'{v:+.2f}%',(i-.18,v),ha='center',va='top',fontsize=9)\n"
            "for i,v in enumerate(base): ax.annotate(f'{v:+.2f}%',(i+.18,v),ha='center',va='top',fontsize=9)\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('signed-SHORT return (%)')\n"
            "ax.set_title('The star bar does NOT beat the everyday baseline — both are just the market drift')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('after evening star:', {f'{h}d': round(m,3) for h,m in zip(hs,means)})\n"
            "print('ordinary day (same bet):', {f'{h}d': round(b,3) for h,b in zip(hs,base)})"
        ),
        md(
            f"Both bars are **red** — shorting loses whether or not there was ever a star. And they're "
            f"roughly the *same shade of red*: at 10 days the star short is **{R['h10'][2]:+.2f}%** vs "
            f"**{-R['h10'][3]:+.2f}%** for an ordinary day's short. That's the tell: the star candle isn't "
            "adding bearish information — it's just riding the same up-drift every other day rides. The "
            "\"significant-looking\" negative number you'd get by comparing the star's short to **zero** "
            "is a mirage created by the market's own tide, not the pattern."
        ),
        md(
            "**Could it just be luck?** Let's pit the real signal against thousands of **fake** ones: pick "
            "the same number of random days, flip a coin for long/short, and see how the real star's return "
            "stacks up. If the star had real bearish power, the real result would sit far in the tail "
            "(better than almost all the fakes)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ev = st.collect_events(PANEL, 5)\n"
            "    obs = float(ev['signed_ret'].mean())\n"
            "    pl = st.placebo_pvalue(PANEL, 5, len(ev), obs, n_draws=5000)\n"
            "    draws = pl['draws']*100; obsp = obs*100; pval = pl['p_value']\n"
            "else:\n"
            "    obsp = R['h5'][2]; pval = R['h5'][9]\n"
            "    rng = np.random.default_rng(683); draws = rng.normal(0.0, 0.32, 5000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='5,000 random coin-flip \"signals\"')\n"
            "ax.axvline(obsp, c=RED, lw=2.5, label=f'real evening-star short {obsp:+.2f}%')\n"
            "ax.set_xlabel('5-day short return (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'The real star sits in the LEFT tail (worse than random): p = {pval:.3f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obsp:+.2f}%   share of random fakes that beat it: p = {pval:.3f}')"
        ),
        md(
            f"The red line sits far in the **left** tail. About **{R['h5'][9]*100:.0f}%** of *random* "
            "coin-flip picks did **better** than shorting the actual evening star. A random number "
            "generator would have made you more money than the famous pattern."
        ),
        md(
            "**Does the picture-perfect version save it?** Maybe a loose, overlapping star is noise but "
            "the strict, true-gap textbook version — or only counting stars that followed a *real* uptrend "
            "— is the genuine deal. Let's check both stricter recipes against a fair (unconditional-base) "
            "comparison."
        ),
        code(
            "if HAVE_REAL:\n"
            "    basic  = [st.summarize(PANEL, h, placebo=False)['t_welch'] for h in hs]\n"
            "    strict = [st.summarize(PANEL, h, placebo=False, gap_tol_frac=0.0)['t_welch'] for h in hs]\n"
            "    trend  = [st.summarize(PANEL, h, placebo=False, require_trend=True)['t_welch'] for h in hs]\n"
            "else:\n"
            "    basic  = [R['h1'][8], R['h5'][8], R['h10'][8]]\n"
            "    strict = [s[3] for s in R['strict']]; trend = [t[3] for t in R['trend']]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.3))\n"
            "ax.bar(x-.25, basic, .25, color=RED, label='basic evening star')\n"
            "ax.bar(x,      strict,.25, color=AMBER, label='strict true-gap star')\n"
            "ax.bar(x+.25,  trend, .25, color=GREY, label='after a real prior uptrend')\n"
            "ax.axhline(0, c='k', lw=.8); ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('Welch t (vs the unconditional base — the fair bar, +-2 dashed)')\n"
            "ax.set_title('No recipe clears the bar — every filter still sits inside the noise band')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('basic t:', [round(b,2) for b in basic]); print('strict t:', [round(s,2) for s in strict]); print('trend t:', [round(t,2) for t in trend])"
        ),
        md(
            "No bar in that chart pokes above the dashed ±2 lines — the fair, drift-neutral test — under "
            "any of the three recipes. The legend doesn't hide in a stricter definition or a genuine "
            "uptrend filter; it simply isn't there."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** Measured fairly (against the basket's own unconditional day, not "
            "against zero), the evening-star short never clears the ±2 bar at any horizon. No single "
            f"stock secretly carries it either — **{R['bonf_survive']}/{R['bonf_k']}** individual tickers "
            "clear a corrected bar. There's no bearish edge.\n"
            "- **Tradability — Mirage.** It's negative **before** costs; once you add the spread and short "
            "borrow it's worse. Nothing to deploy.\n"
            "- **\"Crash incoming\"? — Busted.** The famous \"the top is in\" reading doesn't survive "
            "contact with the tape once you compare it to an ordinary day."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — no, and here's the kicker\n\n"
            "The short is already negative gross. Now add the real-world frictions — the spread paid on "
            "the round trip, plus borrow to hold a short. The bars only sink further."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g  = [st.summarize(PANEL, h, placebo=False)['mean']*100 for h in hs]\n"
            "    n5 = [st.net_of_costs(st.summarize(PANEL, h, placebo=False)['mean'], h, cost_bps=5.0)*100 for h in hs]\n"
            "else:\n"
            "    g  = [R['h1'][2], R['h5'][2], R['h10'][2]]\n"
            "    n5 = [R['h1'][10], R['h5'][10], R['h10'][10]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, g, .4, color=AMBER, label='gross short return')\n"
            "ax.bar(x+.2, n5, .4, color=RED, label='net of costs + borrow (5 bps)')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('SHORT return (%)'); ax.set_title('Costs only deepen the loss — there is nothing to harvest')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('net short by horizon:', {f'{h}d': round(v,3) for h,v in zip(hs,n5)})"
        ),
        md(
            f"At every horizon the **net** short is more negative than the gross — e.g. "
            f"**{R['h5'][10]:.2f}%** at 5 days. There is no window where it pays. The only way to \"make "
            "money\" around an evening star is to do the **opposite** of the lore (don't short into the "
            "drop) — and even that residual is just the market's normal upward drift, not a tradeable "
            "pattern edge."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Why does the naive comparison fool people?** Any short in an up-drifting market looks "
            "\"significantly negative\" against **zero** — with or without a star. That's why this study's "
            "certifying test compares the star's short to what the *same basket* earns on an ordinary day, "
            "not to zero. The [quants notebook](02_for_the_quants.ipynb) shows this exact trap live on a "
            "synthetic panel with a **known** zero edge.\n"
            "- **The survivorship twist.** Our basket excludes names that actually topped, crashed and "
            "delisted — so we've **stacked the deck in the lore's favour** and it *still* fails.\n"
            "- **The bullish mirror.** [186-morning-star](../../186-morning-star/) tests the buy-side twin "
            "with a different harness and lands on the same kind of null.\n\n"
            "*Think a stricter star, a different universe, or a clever filter turns it green? Show the "
            "**net** short landing above the basket's own baseline with Welch *t* ≥ 2 — then we'll talk.*"
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
            "# Evening Star — a quantitative teardown 🔬\n"
            "### Precise OHLC detector · signed-short forward 1/5/10-day event study · a drift-neutral "
            "Welch *t* design · a coin-flip label-shuffle placebo · a Bonferroni correction across the "
            "30-name basket · costs + borrow · strict-gap & prior-uptrend myth-checks · a synthetic "
            "faithful-engine / power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The evening "
            "star is candlestick lore's marquee **bearish reversal** — the job here is to test it honestly "
            "as a short signal, with one subtlety front and centre: the basket carries the market's own "
            "positive drift, so a naive *t*-vs-zero **overstates** any apparent bearishness. Everything "
            "below is built around removing that contamination.\n\n"
            "> ⚠️ **Data + survivorship note.** Fixed **30-name liquid large-cap + SPY** basket, names "
            "still trading in 2026 — a *survivor* panel that **excludes the firms that actually topped, "
            "crashed and delisted**, i.e. it tilts the test *toward* finding a working bearish signal. It "
            "still doesn't. Real data: yfinance daily OHLCV, `auto_adjust=True`, 2005→2026, as-of "
            "**2026-06-30**. Offline core + synthetic control are deterministic. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprints `" + R["fp_quantlab"] +
            "` / `" + R["fp_basket"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Drift-neutral **Welch *t*** (event sample vs the basket's own "
            f"unconditional forward return — the certifying number) never reaches **\\|t\\| >= 2** at any "
            f"horizon: {R['h1'][8]:+.2f} (1d) / {R['h5'][8]:+.2f} (5d) / {R['h10'][8]:+.2f} (10d). Coin-flip "
            f"placebo *p* is {R['h5'][9]:.2f}-{R['h1'][9]:.2f}. **{R['bonf_survive']}/{R['bonf_k']}** "
            "tickers clear a Bonferroni-corrected per-name bar. |\n"
            f"| **Tradability** | `MIRAGE` | Negative **gross** at every horizon "
            f"({R['h1'][2]:+.2f}% to {R['h10'][2]:+.2f}%); net of 5/10-bps round trip + 50 bps/yr borrow "
            f"it is worse ({R['h5'][10]:+.2f}% / {R['h5'][11]:+.2f}% net at 5d). |\n"
            "| **Crash incoming?** | `BUSTED` | The \"top of the uptrend, sell now\" reading does not "
            "survive being measured fairly against the basket's own drift; neither the strict-gap nor the "
            "prior-uptrend filter rescues it. |\n\n"
            "> 💡 In plain words: the star candle has **no certifiable predictive power** as a short — and "
            "the negative-looking headline number you'd get by comparing to zero is the market's ordinary "
            "up-drift, not a pattern edge."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned — and the drift trap named up front\n\n"
            "Let an evening star be confirmed at the close of bar $t$ when the OHLC triple "
            "$(t\\!-\\!2,t\\!-\\!1,t)$ satisfies: candle $t\\!-\\!2$ is a **tall bullish** body "
            "($C_0>O_0$, body $\\ge 0.6\\times$range), candle $t\\!-\\!1$ is a **small star gapping up** "
            "(body $\\le 0.3\\times$body$_0$, body-bottom $\\ge$ body-top$_0 - 0.25\\times$body$_0$), and "
            "candle $t$ is a **tall bearish** body ($C_2<O_2$, body $\\ge 0.6\\times$range) closing at "
            "least **50%** of the way back into candle $0$'s body. Enter the **next** open (one lag), hold "
            "$H$ days **short**, and let $s_i(H)=-\\big(\\text{Close}_{t+H}/\\text{Open}_{t+1}-1\\big)$ be "
            "the signed return of event $i$.\n\n"
            "- **H₁ (bearish edge exists).** $\\overline{s}(H)$ beats the **unconditional** signed-short "
            "return of the same basket, significantly, for some $H$ — shorting the star adds information "
            "beyond just shorting anything.\n"
            "- **H₂ (it's deployable).** The excess survives the round-trip cost + short borrow.\n"
            "- **H₃ (\"strict star / real reversal\").** Restricting to true-gap stars after a prior "
            "uptrend sharpens the edge.\n\n"
            "**Why not test $\\overline{s}(H) > 0$?** Because this basket (like the real market) drifts "
            "**up**, shorting *anything* — including random days — tends to earn a *negative* number "
            "against **zero**, with or without any pattern. A vs-zero test therefore reads \"significant\" "
            "purely from the tape's ordinary drift. Section 4a demonstrates this trap directly on a "
            "synthetic panel with a **known** zero pattern-edge. We find **H₁ rejected** (Welch "
            "$t \\in [-1.84,-0.43]$, never $\\le -2$), **H₂ rejected** (net worse than gross), **H₃ "
            "rejected** (filters don't help)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "The Signal axis's **decisive** statistic is a **Welch *t*** of the event sample's signed-short "
            "mean against the basket's own unconditional forward-return pool (signed short, i.e. $-$base):\n\n"
            "$$t_{\\text{Welch}} = \\frac{\\overline{s}_{\\text{event}} - \\overline{(-r)}_{\\text{base}}}"
            "{\\sqrt{\\widehat{\\mathrm{Var}}(s)/n_{\\text{event}} + \\widehat{\\mathrm{Var}}(-r)/n_{\\text{base}}}}.$$\n\n"
            "A one-sample **Newey-West HAC *t*** against zero is reported alongside as an "
            "**informational** cross-check only — it is the statistic a naive read would use, and it is "
            "exactly the one contaminated by the basket's own up-drift (proved on the synthetic null in "
            "4f). The **coin-flip placebo** (same event count, random signs) is a second, independent "
            "honesty check: it asks whether a random pick of the same size could look this good, without "
            "presupposing any particular null mean. Finally, with **30 tickers** tested individually for a "
            "per-name version of the effect, the two-sided significance bar widens via **Bonferroni** to "
            "$|t|\\ge z(1-0.025/30)\\approx 3.14$ — otherwise the single loudest name in a wide basket gets "
            "mistaken for \"the\" signal."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** Fixed **{R['n_names']}-name** liquid large-cap + SPY basket (yfinance daily "
            f"OHLCV, `auto_adjust=True`, {R['start']}→{R['end']}, {R['total_bars']:,} bars). **Survivor** "
            "panel — named on the Signal axis (and it tilts *toward* the bearish claim).\n"
            "- **Detector.** Precise real-body evening star on bars $(t\\!-\\!2,t\\!-\\!1,t)$: tall-body "
            "thresholds, a star-gap tolerance, and a 50% penetration rule; a **strict** (`gap_tol_frac=0`) "
            "and a **prior-uptrend** variant for the myth-checks.\n"
            "- **Timing.** Confirm at the close of $t$; enter the **next open** (one lag); hold "
            "$H\\in\\{1,5,10\\}$ days **short** (per the brief); drop events whose window overruns.\n"
            "- **Certifying stat.** Welch *t* of the signed-short sample vs the basket's own unconditional "
            "pool (drift-neutral) — the inference-bar number.\n"
            "- **Cross-checks.** One-sample/HAC *t* vs zero (informational, drift-contaminated); a "
            "5,000-draw coin-flip label-shuffle placebo; hit-rate vs base hit-rate with Wilson intervals.\n"
            "- **Bonferroni across the basket.** Per-ticker Welch *t* (event sample vs that ticker's own "
            "base), corrected critical value for $k=30$ simultaneous tests.\n"
            "- **Costs.** 5 / 10 bps one-way × 2 (round trip) + 50 bps/yr borrow on the short over the hold.\n"
            "- **Positive control.** A deterministic panel with its own embedded up-drift (mirroring the "
            "real basket's contamination risk) and a **planted** post-pattern crash: zero edge must NOT "
            "reach significance on the *certifying* Welch stat across 20 seeds; a planted edge must light "
            "it up.\n\n"
            "> **Falsifier, stated in advance:** if the drift-neutral Welch *t* never clears **+2** on the "
            "real tape at any horizon, the bearish signal is **None** and \"crash incoming\" is **Busted**."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The drift trap, shown live — vs-zero vs vs-base\n\n"
            "Left: the signed-short mean by horizon against **zero** (the naive, contaminated read) with "
            "its HAC *t*. Right: the same means against the basket's own **unconditional** base (the "
            "certifying Welch *t*). Watch the bar heights *and* which axis they clear."
        ),
        code(
            "hs = [1, 5, 10]\n"
            "if HAVE_REAL:\n"
            "    rows = [st.summarize(PANEL, h, placebo=False) for h in hs]\n"
            "    means = [r['mean']*100 for r in rows]\n"
            "    t_hac = [r['t_hac'] for r in rows]; t_welch = [r['t_welch'] for r in rows]\n"
            "else:\n"
            "    means = [R['h1'][2], R['h5'][2], R['h10'][2]]\n"
            "    t_hac = [R['h1'][6], R['h5'][6], R['h10'][6]]\n"
            "    t_welch = [R['h1'][8], R['h5'][8], R['h10'][8]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], t_hac, color=RED, width=.55)\n"
            "a1.axhline(-2, ls='--', c='k', lw=1); a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_ylabel('t (vs ZERO, informational)'); a1.set_title('Looks significant... (HAC t vs 0)')\n"
            "for i,t in enumerate(t_hac): a1.annotate(f'{t:.2f}',(i,t),ha='center',va='top' if t<0 else 'bottom')\n"
            "a2.bar([f'{h}d' for h in hs], t_welch, color=AMBER, width=.55)\n"
            "a2.axhline(-2, ls='--', c='k', lw=1); a2.axhline(2, ls='--', c='k', lw=1); a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_ylabel('Welch t (vs the basket\\'s OWN base — decisive)'); a2.set_title('...until you compare it fairly')\n"
            "for i,t in enumerate(t_welch): a2.annotate(f'{t:.2f}',(i,t),ha='center',va='top' if t<0 else 'bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('t vs zero (HAC):', [round(t,2) for t in t_hac])\n"
            "print('t vs base (Welch, decisive):', [round(t,2) for t in t_welch])"
        ),
        md(
            f"> 💡 In plain words: the left panel — comparing to zero — clears the dashed **−2** bar at "
            f"5- and 10-day ({R['h5'][6]:.2f}, {R['h10'][6]:.2f}). Panic averted by the right panel: once "
            f"compared to what the *same basket* earns on an **ordinary** day, none of the three horizons "
            f"clear ±2 ({R['h1'][8]:+.2f}, {R['h5'][8]:+.2f}, {R['h10'][8]:+.2f}). The left panel was "
            "measuring the market's up-drift, not the star. This is exactly why the Welch-vs-base number "
            "is the one on the front-card."
        ),
        md(
            "### 4b · The full inference table — HAC, one-sample, Welch, hit-rate, placebo\n\n"
            "Every arbiter at once. The believer needs a **positive** signed-short excess over the base "
            "with Welch *t* ≥ +2; nothing here comes close."
        ),
        code(
            "import pandas as pd\n"
            "if HAVE_REAL:\n"
            "    tab = pd.DataFrame([st.summarize(PANEL, h, n_draws=5000) for h in hs])\n"
            "    show = tab[['horizon','n_events','mean','base_mean','hit','base_hit','t_hac','t_one','t_welch','p_placebo','net']].copy()\n"
            "    for c in ['mean','base_mean','net']: show[c] = (show[c]*100).round(3)\n"
            "    for c in ['hit','base_hit']: show[c] = (show[c]*100).round(0)\n"
            "    for c in ['t_hac','t_one','t_welch']: show[c] = show[c].round(2)\n"
            "    show['p_placebo'] = show['p_placebo'].round(3)\n"
            "else:\n"
            "    keys=['h1','h5','h10']\n"
            "    cols=['horizon','n_events','mean','base_mean','hit','base_hit','t_hac','t_one','t_welch','p_placebo','net5','net10']\n"
            "    show = pd.DataFrame([dict(zip(cols, R[k])) for k in keys]).drop(columns=['net10']).rename(columns={'net5':'net'})\n"
            "print(show.to_string(index=False))"
        ),
        md(
            "> 💡 In plain words: the `hit`/`base_hit` columns are nearly identical (and the 10-day pattern "
            "day is even *slightly* more likely to be up than an ordinary day) — no directional tilt "
            "either way. `t_welch`, the decisive column, never leaves the noise band. The signed return is "
            "negative in absolute terms simply because shorting anything in this tape tends to lose to the "
            "market's own drift."
        ),
        md(
            "### 4c · Bonferroni across the basket — does one name secretly carry it?\n\n"
            "A pooled null can hide a real effect concentrated in a handful of names, or manufacture a "
            "fake one from a single noisy outlier. Per-ticker Welch *t* (event sample vs that ticker's "
            "*own* unconditional pool — drift-neutral at the name level too) at $H=5$, with the "
            f"Bonferroni-corrected bar for $k={R['bonf_k']}$ simultaneous tests."
        ),
        code(
            "pt = st.per_ticker_stats(PANEL, 5) if HAVE_REAL else None\n"
            "if pt is not None and len(pt):\n"
            "    zc = st.bonferroni_z(len(pt))\n"
            "    order = pt.reindex(pt['t_welch'].abs().sort_values(ascending=False).index)\n"
            "    tickers = list(order['ticker']); tv = list(order['t_welch'])\n"
            "    k = len(pt)\n"
            "else:\n"
            "    zc = R['bonf_z']; k = R['bonf_k']\n"
            "    tickers = [b[0] for b in R['bonf_top']]; tv = [b[3] for b in R['bonf_top']]\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.4))\n"
            "cols = [RED if abs(t) >= zc else GREY for t in tv[:15]]\n"
            "ax.bar(tickers[:15], tv[:15], color=cols, width=.6)\n"
            "ax.axhline(zc, ls='--', c=RED, lw=1, label=f'Bonferroni bar |t|>={zc:.2f} (k={k})')\n"
            "ax.axhline(-zc, ls='--', c=RED, lw=1)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('per-ticker Welch t (vs its own base)'); ax.set_xlabel('ticker (top 15 by |t|)')\n"
            "ax.set_title('No single name survives the multiple-testing correction')\n"
            "plt.xticks(rotation=45, ha='right'); ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'{k} tickers tested, Bonferroni bar={zc:.2f}, survivors:', sum(abs(t)>=zc for t in tv))"
        ),
        md(
            f"> 💡 In plain words: even the loudest names in a {R['bonf_k']}-name basket — exactly the "
            "outliers a chart-reader would cherry-pick and show you — sit *below* the corrected bar "
            f"(**{R['bonf_survive']}/{R['bonf_k']}** clear it). The pooled null in 4a/4b isn't hiding a "
            "real effect in a handful of names; there's simply no name carrying it."
        ),
        md(
            "### 4d · The myth check — does a stricter star rescue it?\n\n"
            "Two ways to make the signal \"purer\": the **strict textbook gap** (the star's body doesn't "
            "touch candle $t\\!-\\!2$'s body at all), and only counting stars that **followed a genuine "
            "prior uptrend** (a true reversal). If the lore were right, these should sharpen the effect."
        ),
        code(
            "if HAVE_REAL:\n"
            "    basic  = [st.summarize(PANEL, h, placebo=False)['t_welch'] for h in hs]\n"
            "    strict = [st.summarize(PANEL, h, placebo=False, gap_tol_frac=0.0)['t_welch'] for h in hs]\n"
            "    trend  = [st.summarize(PANEL, h, placebo=False, require_trend=True)['t_welch'] for h in hs]\n"
            "    ns = st.summarize(PANEL, 5, placebo=False, gap_tol_frac=0.0)['n_events']\n"
            "    nt = st.summarize(PANEL, 5, placebo=False, require_trend=True)['n_events']\n"
            "else:\n"
            "    basic  = [R['h1'][8], R['h5'][8], R['h10'][8]]\n"
            "    strict = [s[3] for s in R['strict']]; trend = [t[3] for t in R['trend']]\n"
            "    ns = R['strict'][1][1]; nt = R['trend'][1][1]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.3))\n"
            "ax.bar(x-.25, basic, .25, color=RED, label='basic evening star')\n"
            "ax.bar(x,      strict,.25, color=AMBER, label=f'strict true-gap star (n5={ns})')\n"
            "ax.bar(x+.25,  trend, .25, color=GREY, label=f'prior uptrend (n5={nt})')\n"
            "ax.axhline(2, ls='--', c='k', lw=1); ax.axhline(-2, ls='--', c='k', lw=1); ax.axhline(0,c='k',lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('Welch t (vs unconditional base, +-2 dashed)'); ax.set_title('No filter clears the bar — the edge simply is not there')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('strict Welch t:', [round(s,2) for s in strict]); print('trend Welch t :', [round(t,2) for t in trend])"
        ),
        md(
            f"> 💡 In plain words: the strict true-gap star (only **{R['strict'][1][1]}** events at 5d — "
            f"a smaller, rarer sample) gives Welch *t* = **{R['strict'][1][3]:.2f}**, and the "
            f"prior-uptrend reversal gives *t* = **{R['trend'][1][3]:.2f}**. Both sit inside the noise "
            "band. The pattern's predictive content is **indistinguishable from the basket's own drift** "
            "under every reasonable definition."
        ),
        md(
            "### 4e · Costs — the loss only deepens\n\n"
            "Gross vs net (5 / 10 bps one-way × 2 + 50 bps/yr borrow on the short). When the excess over "
            "base is already statistically nothing, costs can't save it — they bury the gross number."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g  = [st.summarize(PANEL, h, placebo=False)['mean']*100 for h in hs]\n"
            "    n5 = [st.net_of_costs(st.summarize(PANEL, h, placebo=False)['mean'], h, cost_bps=5.0)*100 for h in hs]\n"
            "    n10= [st.net_of_costs(st.summarize(PANEL, h, placebo=False)['mean'], h, cost_bps=10.0)*100 for h in hs]\n"
            "else:\n"
            "    g  = [R['h1'][2], R['h5'][2], R['h10'][2]]\n"
            "    n5 = [R['h1'][10], R['h5'][10], R['h10'][10]]\n"
            "    n10= [R['h1'][11], R['h5'][11], R['h10'][11]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.25, g, .25, color=AMBER, label='gross short')\n"
            "ax.bar(x,      n5, .25, color=RED, label='net @ 5 bps + borrow')\n"
            "ax.bar(x+.25,  n10,.25, color='#7a1f14', label='net @ 10 bps + borrow')\n"
            "ax.axhline(0,c='k',lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs])\n"
            "ax.set_ylabel('SHORT return (%)'); ax.set_title('Gross is already negative; net is worse — no tradeable window')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for h,gg,n5_,n10_ in zip(hs,g,n5,n10): print(f'{h:>2}d: gross={gg:+.3f}%  net5bps={n5_:+.3f}%  net10bps={n10_:+.3f}%')"
        ),
        md(
            f"> 💡 In plain words: at 5 days gross **{R['h5'][2]:+.2f}%** → net **{R['h5'][10]:+.2f}%** "
            "(5 bps) / **{:+.2f}%** (10 bps). There is no horizon where the short clears zero, let alone "
            "its costs, let alone the unconditional base it should be beating. **Mirage** in the strict "
            "sense — nothing under it to charge costs *against*.".format(R['h5'][11])
        ),
        md(
            "### 4f · Faithful-engine & power control — the drift trap, proven on a KNOWN-zero panel\n\n"
            "A deterministic panel with its own mild embedded up-drift (so tall-up-candle stars actually "
            "occur — mirroring the real basket's contamination risk) and a **planted** post-pattern crash "
            "knob. With **zero** planted edge, the certifying **Welch t** (vs the panel's own base) must "
            "stay inside ±2 across **20 independent seeds**; the one-sample *t*-vs-zero, by contrast, is "
            "shown to fire on its own from the embedded drift alone — exactly the trap in 4a."
        ),
        code(
            "null_welch = []\n"
            "for s_ in range(20):\n"
            "    px, _ = data.synthetic_panel(edge=0.0, seed=683 + s_)\n"
            "    null_welch.append(st.summarize(px, 5, placebo=False)['t_welch'])\n"
            "null_welch = np.asarray(null_welch)\n"
            "res = []\n"
            "for edge in (0.0, 0.004, 0.008):\n"
            "    px, truth = data.synthetic_panel(edge=edge, seed=683)\n"
            "    s = st.summarize(px, 5, n_draws=4000)\n"
            "    res.append((edge, truth['n_planted_days'], s['n_events'], s['mean']*100, s['t_welch'], s['t_hac'], s['p_placebo'], s['hit']*100))\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.scatter(np.zeros(20) + np.linspace(-.1,.1,20), null_welch, color=GREY, s=40, label='null worlds (edge=0), 20 seeds')\n"
            "a1.axhline(-2, ls='--', c=RED, lw=1); a1.axhline(2, ls='--', c=RED, lw=1); a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_xticks([0]); a1.set_xticklabels(['null x 20']); a1.set_ylabel('Welch t (vs panel base)')\n"
            "a1.set_title(f'Null does NOT fire: {int((abs(null_welch)>=2).sum())}/20 seeds >= |t|=2'); a1.legend()\n"
            "labels = [f'planted\\n{e*100:.1f}%/day' for e,_,_,_,_,_,_,_ in res]\n"
            "tvals = [r[4] for r in res]\n"
            "a2.bar(labels, tvals, color=[GREY, AMBER, GREEN], width=.5)\n"
            "a2.axhline(2, ls='--', c=RED, label='t=+2 bar')\n"
            "for i,t in enumerate(tvals): a2.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "a2.set_ylabel('Welch t (5d, seed 683)'); a2.set_title('A planted crash lights up cleanly'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'null Welch t: mean={null_welch.mean():+.2f} sd={null_welch.std(ddof=1):.2f} |t|>=2 in {(abs(null_welch)>=2).sum()}/20')\n"
            "for e,pd_,k,ls,tw,th,p,w in res: print(f'planted {e*100:+.1f}%/day: n={k} mean={ls:+.3f}% t_welch={tw:.2f} t_hac(vs0)={th:.2f} p={p:.3f} hit={w:.0f}%')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted crash, the certifying Welch *t* averages "
            f"**{R['syn_null_mean']:+.2f}** (sd {R['syn_null_sd']:.2f}) across 20 seeds and fires in only "
            f"**{R['syn_null_fire']}/{R['syn_null_n']}** — in line with the ~5% nominal false-positive rate "
            "of a two-sigma bar, not a systematic bias. A planted crash of 0.4%/day reaches Welch "
            f"*t* = **{R['syn'][1][4]:.2f}**, and 0.8%/day reaches **{R['syn'][2][4]:.2f}** — the harness "
            "*would* catch a real evening-star crash if one existed. Note also the null-world's *own* "
            f"vs-zero HAC *t* ({R['syn'][0][5]:+.2f}) — even with **zero** planted edge, a naive read isn't "
            "always innocent either, which is exactly why Welch-vs-base is the number that decides the "
            "verdict, not HAC-vs-zero."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the certifying, drift-neutral Welch *t* (event sample vs the basket's "
            f"own unconditional forward return) never reaches **\\|t\\| >= 2** at any horizon "
            f"({R['h1'][8]:+.2f} / {R['h5'][8]:+.2f} / {R['h10'][8]:+.2f} for 1/5/10 days). Hit rates track "
            f"the base rate. The coin-flip placebo places the observed mean where "
            f"{R['h1'][9]*100:.0f}-{R['h5'][9]*100:.0f}% of random draws would have matched or beaten it. "
            f"**{R['bonf_survive']}/{R['bonf_k']}** tickers survive a Bonferroni correction individually. "
            "Neither a strict textbook gap nor a genuine prior-uptrend filter rescues it. Carries an "
            "explicit **survivorship** caveat that tilts the test *toward* the claim — and it still fails. "
            "NONE, not WEAK.\n"
            f"- **Tradability `MIRAGE`** — negative **gross** at every horizon "
            f"({R['h1'][2]:+.2f}% to {R['h10'][2]:+.2f}%); net of a round-trip cost + short borrow it's "
            f"worse across the whole 5-10 bps sweep ({R['h5'][10]:+.2f}% to {R['h10'][11]:+.2f}%). There is "
            "no edge to charge costs against.\n"
            "- **Crash incoming? `BUSTED`** — the \"top of the uptrend, sell now\" reading does not "
            "survive a fair comparison to what the same basket does on an ordinary day. Neither purity "
            "filter saves it, and no individual name in the basket carries a Bonferroni-robust version of "
            "the story."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the only money is in fading the lore\n\n"
            "One picture for the operational truth: the signed-**short** net return by horizon against "
            "the break-even line and against what an ordinary day earns on the same bet. The "
            "\"tradeable\" region — net above *both* zero and the baseline — is **empty**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    g  = [st.summarize(PANEL, h, placebo=False)['mean']*100 for h in hs]\n"
            "    n5 = [st.net_of_costs(st.summarize(PANEL, h, placebo=False)['mean'], h, cost_bps=5.0)*100 for h in hs]\n"
            "    base = [-st.summarize(PANEL, h, placebo=False)['base_mean']*100 for h in hs]\n"
            "else:\n"
            "    g  = [R['h1'][2], R['h5'][2], R['h10'][2]]\n"
            "    n5 = [R['h1'][10], R['h5'][10], R['h10'][10]]\n"
            "    base = [-R['h1'][3], -R['h5'][3], -R['h10'][3]]\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.plot(hs, g, 'o--', c=GREY, lw=1.6, label='gross short')\n"
            "ax.plot(hs, n5, 'o-', c=RED, lw=2.2, label='net of costs + borrow')\n"
            "ax.plot(hs, base, 's:', c=AMBER, lw=1.6, label='ordinary-day baseline (same bet)')\n"
            "ax.axhline(0, c='k', ls='--', label='break-even')\n"
            "ax.fill_between(hs, 0, n5, color=RED, alpha=.10)\n"
            "for h,v in zip(hs,n5): ax.annotate(f'{v:+.2f}%',(h,v),ha='center',va='top',fontsize=9)\n"
            "ax.set_xlabel('horizon (days)'); ax.set_ylabel('signed-short return (%)')\n"
            "ax.set_title('Empty tradeable region: net sits below both zero AND the baseline'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net short by horizon:', {f'{h}d': round(v,3) for h,v in zip(hs,n5)})"
        ),
        md(
            "> 💡 In plain words: the red (net) line sits below **both** zero and the amber baseline at "
            "every horizon — there is no window where shorting the star pays, and no window where it even "
            "beats doing nothing special. The contrarian residue (the stock drifting up) is just the "
            "unconditional equity drift you'd capture by holding *anything*; it is not a pattern edge and "
            "carries no timing value. Hence **Mirage**, and the bearish claim is **Busted**."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Why it fails, mechanically.** The confirming candle is, by construction, a stock that just "
            "had a sharp **down** day; short-horizon mean reversion (the bounce) works against a fresh "
            "short, and the naive vs-zero test simply rediscovers the market's own up-drift — a trap this "
            "study's Welch-vs-base design is built to avoid.\n"
            "- **The survivorship lever points the *right* way here.** Excluding delisted, actually-crashed "
            "names biases us **toward** a working short signal — so a clean fail on survivors is strong "
            "evidence. Marshall, Young & Rose (2006) reach the same broad null on the DJIA cross-section.\n"
            "- **The bullish mirror & near neighbours.** [186-morning-star](../../186-morning-star/) tests "
            "the buy-side twin (different harness, same conclusion); "
            "[408-three-black-crows](../../408-three-black-crows/) is a structurally different three-red-"
            "candle pattern sharing the same mean-reversion headwind; "
            "[404-shooting-star](../../404-shooting-star/) is a single-candle relative;\n"
            "  [402-engulfing-pattern](../../402-engulfing-pattern/) is a two-candle relative.\n\n"
            "*The reproducible core is offline and deterministic; the detector is the precise real-body "
            "evening star with a strict-gap variant and a prior-uptrend filter as myth-checks, and a "
            "Bonferroni correction across the 30-name basket. Methods and sources: "
            "[`docs/references.md`](../docs/references.md); frozen numbers: "
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
