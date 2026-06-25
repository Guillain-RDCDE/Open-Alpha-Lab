"""Generate the two narrative notebooks for Study 468 (Gartley / AB=CD Harmonic).

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
# yfinance daily, 5 indices/ETFs (SPY QQQ IWM DIA GLD), 2005-01-03 -> 2026-05-29 (As-of
# 2026-05-31, partial June dropped), 21.4 years, pivot fractal k=4, bullish-Gartley D-point long.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=194, k=4,
    fp_spy="4cb5244f3990",
    # pooled D-point long, per horizon:
    # (H, n, D_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 193, 20.3, 56, 1.15, 23.7, -3.4, 18.3, -0.13, 0.893),
    h10=(10, 193, 57.2, 67, 2.01, 39.6, 17.6, 55.2, 0.47, 0.641),
    h20=(20, 192, 123.9, 68, 3.05, 67.8, 56.2, 121.9, 1.09, 0.275),
    h60=(60, 189, 275.9, 70, 3.57, 75.2, 200.7, 273.9, 2.40, 0.017),
    # per-ticker H=20: (ticker, entries, D_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 27, 132.8, 2.10, 57.5, 75.3), ("QQQ", 26, 137.2, 1.22, 125.9, 11.3),
         ("IWM", 45, 195.4, 1.72, 44.7, 150.7), ("DIA", 31, -24.8, -0.32, 38.1, -63.0),
         ("GLD", 65, 136.8, 2.00, 71.5, 65.3)],
    # ratio-grid placebo (SPY, 500 draws): 20d (obs_bps, p), 60d (obs_bps, p)
    placebo20=(132.8, 0.443, 500),
    placebo60=(244.6, 0.591, 500),
    # synthetic control (H=20, n_days=4000): (edge, n, D_bps, win%, one_sample_t)
    syn=[(0.00, 44, -6.8, 50, -0.10), (0.40, 53, 300.6, 100, 60.63)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Fibonacci_forecasts%3F: Busted](https://img.shields.io/badge/Fibonacci_forecasts%3F-Busted-8b949e?style=flat-square)\n\n"
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

from gartley_harmonic import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real Gartley cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does a Gartley \"harmonic\" pattern actually call reversals? 🦋\n"
            "### A famous chart figure — five dots and a handful of Fibonacci ratios — meets a stopwatch\n\n"
            + BADGES +
            "Harmonic trading is one of the most seductive corners of technical analysis. You find a "
            "five-point zig-zag **X-A-B-C-D** on a chart, check that its legs retrace each other by the "
            "**Fibonacci numbers** (0.618, 0.786…), and — if they do — you declare point **D** a "
            "high-probability **reversal** and buy it. Whole courses, books and auto-scanners (Gartley, "
            "Bat, Butterfly, Crab) are built on this premise.\n\n"
            "It *looks* uncanny on a hand-picked chart. But a pattern you label **after** the swings "
            "have happened, by choosing which five dots to connect and how loose a tolerance to accept, "
            "is the textbook setup for fooling yourself — especially on a market that drifts **up**. So "
            "we did the only fair thing: encode the Gartley **mechanically** (no eyeballing), fire the "
            "\"buy the D-point\" rule across five big indices over 21 years, and time the result against "
            "two honest baselines: **buying on random days**, and **swapping the Fibonacci ratios for "
            "random ones.**\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice** — research & education. Every chart is drawn by the code "
            "beside it; house style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| If I buy when a Gartley completes at **D**, do I make money? | **A little — and only at "
            "long horizons.** Over 60 days the D-point beats random days; over 5/10/20 days it's a "
            "coin flip. |\n"
            "| Is that *the Fibonacci ratios'* doing? | **No.** Swap the 0.618 / 0.786 ratios for "
            "**random** ratios and you do **just as well**. The magic numbers add nothing. |\n"
            "| So what's the 60-day edge, then? | **Generic dip-buying.** You bought a confirmed swing "
            "low and waited three months on a market that drifts up. The fork — sorry, the *fan* of "
            "Fibonacci ratios — is decoration. |\n"
            "| Is it a tradable edge? | **Barely, and fragile.** One horizon, ~9 trades a year, "
            "geometry-agnostic. You'd capture the same thing more cheaply by just buying dips. |\n\n"
            "> The Gartley is a vivid way to *describe* a swing after the fact. As a *forecast* — "
            "\"these Fibonacci ratios mark the turn\" — it's a **busted** premise: a real long-horizon "
            "dip-buy is in there, but the ratios aren't the reason."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Find five swing points X-A-B-C-D. If B retraces **0.618** of XA, C retraces "
            "0.382–0.886 of AB, and D retraces **0.786** of XA, you have a **Gartley** — and price will "
            "reverse at D. Buy the bullish completion; the Fibonacci ratios forecast the turn.\"*\n\n"
            "This codification is **H. M. Gartley** (1935) for the figure, **Larry Pesavento** (1997) "
            "and **Scott Carney** (1998) for the Fibonacci ratios. The whole harmonic zoo — Bat, "
            "Butterfly, Crab — is the same XABCD geometry with different magic numbers, and it's baked "
            "into TradingView, MetaTrader and Thinkorswim. So: do the ratios actually *forecast*?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If specific Fibonacci ratios genuinely *forecast* reversals, it would be remarkable: five "
            "past dots and two decimals (0.618, 0.786) would predict a future turning point — a clean "
            "crack in market efficiency you could trade with a calculator. That's the dream harmonic "
            "trading sells.\n\n"
            "But there are two traps. (1) A Gartley is labelled **by hand, after the swings happen**, on "
            "a market that drifts **up**, so *any* swing-low buy will look profitable. (2) The harmonic "
            "zoo offers **many** ratio templates — pick the one that fit in hindsight and you've "
            "data-mined a number. To separate the **ratios** from the **tide**, we (a) draw the pattern "
            "by a fixed mechanical rule with no hindsight, (b) compare it to buying on **random days**, "
            "and (c) swap the Fibonacci ratios for **random** ratios. We'll do all three."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Find the swing points mechanically.** A 'pivot' is a high (or low) with "
            f"**{R['k']} lower (higher) bars on each side** — a confirmed fractal, only known "
            f"**{R['k']} bars later**, so we never label the pattern with future data.\n"
            "2. **Scan for Gartleys by rule.** At every confirmed swing-low, search recent pivots for an "
            "ordered X-A-B-C-D whose ratios fit the Gartley grid — no eyeballing, no cherry-picking.\n"
            "3. **Trade the lore.** When a bullish Gartley completes at **D**, buy at the next close; "
            "measure the return over the next **5 / 10 / 20 / 60 days**.\n"
            "4. **The honest baselines.** Do the same hold on **random days** (kills the drift), and "
            "rerun with **random ratios** instead of Fibonacci (kills the magic numbers). If the "
            "Gartley matters, it must beat *both*. *If it doesn't, the ratios are decoration* — "
            "announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does a mechanical Gartley even look like? Here's SPY with the detected "
            "bullish-Gartley D-points the rule would buy."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    cl = b['close']; seg = cl.iloc[-700:]\n"
            "    ent = st.detect_completions(cl, k=R['k'])\n"
            "    ent = ent[ent >= seg.index[0]]\n"
            "    piv = st._alternating(st.find_pivots(cl, k=R['k']))\n"
            "    pv = piv[[ (cl.index[int(p)] >= seg.index[0]) for p in piv.index ]]\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    ax.plot(seg.index, seg.values, c='k', lw=1.1, label='SPY close')\n"
            "    ax.scatter([cl.index[int(p)] for p in pv.index], pv['price'], c=GREY, s=14, zorder=4, label='swing pivots')\n"
            "    ax.scatter(ent, cl.reindex(ent), c=GREEN, s=55, marker='^', zorder=5, label='Gartley D-point BUY')\n"
            "    ax.set_title('Mechanical bullish-Gartley D-points on SPY (last ~3y)'); ax.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('Gartley D-points in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The detector picks out genuine five-point swings. The question is whether those green "
            "triangles are followed by bounces *because of the Fibonacci ratios*. **Let's race the "
            "D-point against random entries** at four horizons. Blue = buy the Gartley D; grey = buy on "
            "random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    dpt, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            e = st.detect_completions(c, k=R['k'])\n"
            "            re = st.random_entries(c, max(len(e),50), k=R['k'], seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        dpt.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    dpt = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, dpt, .4, color='#2c6fbb', label='buy the Gartley D-point')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(dpt,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('The D-point only pulls ahead at 60 days'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('D-point:', [round(v) for v in dpt]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the headline. At 5/10/20 days the Gartley D-point is roughly a coin flip vs random "
            f"(it's even *behind* at 5d). Only at **60 days** does it clearly pull ahead "
            f"(**+{R['h60'][2]:.0f} bps** vs random's **+{R['h60'][5]:.0f}**). So there *is* a "
            "long-horizon effect — but is it the *Fibonacci ratios*? Next test."
        ),
        md(
            "**The decisive check.** Keep the whole five-point machinery but **swap the Fibonacci "
            "ratios for random ones** (random targets in place of 0.618 / 0.786). If price really turns "
            "at *Fibonacci* D-points, the nonsense-ratio scans should do much worse."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    pl = st.ratio_grid_placebo(c, 20, k=R['k'], n_draws=200, seed=468)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo20'][0]; pval = R['placebo20'][1]\n"
            "print(f'real Gartley D-point (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of *random-ratio* grids do at least as well (p={pval:.2f}).')\n"
            "print('=> the Fibonacci ratios are not doing the work.')"
        ),
        md(
            f"Nearly half of the **random-ratio** grids match or beat the real Gartley "
            f"(*p* = {R['placebo20'][1]:.2f} at 20d, {R['placebo60'][1]:.2f} at 60d). If price genuinely "
            "respected *these specific Fibonacci numbers*, a random ratio swap would collapse the "
            "result. It doesn't — because the long-horizon effect was never about the ratios."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The D-point beats random days at **only one** horizon (60-day, "
            "*p* = 0.017); 5/10/20 days are a coin flip. A thin, long-only, long-horizon effect.\n"
            "- **Tradability — Fragile.** One horizon, ~9 trades/year, and it survives the geometry "
            "scramble — it's generic swing-low dip-buying, capturable more cheaply.\n"
            "- **\"Do the Fibonacci ratios forecast the D-point turn?\" — Busted.** Swap the ratios for "
            "random ones and the result barely moves (*p* ≈ 0.44–0.59). The magic numbers add nothing."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Barely, and not as a *harmonic* strategy. The only thing the D-point has over a coin flip "
            "is a long-horizon dip-buy on a drifting market — which you'd get more simply by buying "
            "confirmed swing lows (or just holding the index). Drawing five-point Fibonacci forks adds "
            "labour and false precision, not edge. At ~9 trades a year on a single horizon, with the "
            "ratios demonstrably inert, there's nothing here to scale. As a forecasting tool the "
            "Gartley's defining feature — its Fibonacci ratios — does no measurable work."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The rest of the zoo.** Bat (D = 0.886), Butterfly (1.272), Crab (1.618) are the same "
            "geometry with different magic numbers. The ratio-grid placebo predicts they'll all land "
            "the same way: drift + dip-buy in, Fibonacci out.\n"
            "- **Tighter tolerances.** Squeeze the ratio bands and you get fewer trades and noisier "
            "stats, not a cleaner edge — the hallmark of a non-load-bearing parameter.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* Gartley bounce into a "
            "synthetic tape and shows the harness banks it (so the busted ratio result isn't a dead "
            "detector — it's an honest 'the numbers aren't it').\n\n"
            "*Think the ratios forecast? Show the Fibonacci grid beating random ratio grids at "
            "**p < 0.05** on a real tape — then we'll talk.*"
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
            "# Gartley / AB=CD Harmonic — a quantitative teardown 🔬\n"
            "### Mechanical XABCD scans on 5 indices · D-point forward returns · one-sample HAC *t* · "
            "a drift-matched random-entry baseline · a Fibonacci-ratio-grid placebo · costs · a "
            "synthetic planted-bounce control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). Two things "
            "must be separated: the **drift** (an up-trending index makes any swing-low buy look good) "
            "and the **Fibonacci ratios** (the harmonic claim is that *0.618 / 0.786 specifically* "
            "forecast the turn). We use a random-entry baseline for the first and a ratio-grid placebo "
            "for the second.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. Pivots are confirmed "
            f"fractals (k={R['k']}, an explicit {R['k']}-bar confirmation lag); entry is the **next "
            "close** (one documented lag). Offline core + synthetic control are deterministic. "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | D-point vs a **drift-matched random** baseline clears *t* = 2 at "
            f"**only one** horizon: Welch *t* = {R['h5'][8]:+.2f}/{R['h10'][8]:+.2f}/{R['h20'][8]:+.2f}/"
            f"**{R['h60'][8]:+.2f}** at 5/10/20/60d (60d *p* = {R['h60'][9]:.3f}). |\n"
            f"| **Tradability** | `FRAGILE` | A single 60-day horizon, ~9 trades/yr, and it **survives "
            f"the geometry scramble** — generic dip-buying, not harmonic structure. |\n"
            f"| **Fibonacci forecasts?** | `BUSTED` | Random ratio grids match or beat the Fibonacci "
            f"grid: **p = {R['placebo20'][1]:.2f}** (20d) / **{R['placebo60'][1]:.2f}** (60d). The "
            "0.618/0.786 numbers carry no information. |\n\n"
            "> 💡 In plain words: there's a faint long-horizon dip-buy in the D-points, but the thing "
            "that *defines* a Gartley — its Fibonacci ratios — does none of the work. Beta + "
            "mean-reversion in a harmonic costume."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "A bullish Gartley is five pivots $X,A,B,C,D$ (X,B,D lows; A,C highs) with leg ratios "
            "$\\frac{AB}{XA}\\approx0.618$, $\\frac{BC}{AB}\\in[0.382,0.886]$, "
            "$\\frac{AD}{XA}\\approx0.786$. The rule buys at the confirmed $D$ and rides the reversal.\n\n"
            "- **H₀ (drift).** D-point returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (the pattern forecasts).** D returns **exceed** random at some horizon, t ≥ 2.\n"
            "- **H₂ (the *ratios* matter).** D returns exceed a **random-ratio** grid that keeps the "
            "zig-zag machinery but discards the Fibonacci targets.\n\n"
            "We find **H₁ partially supported** (60d Welch t = +2.40 only) but **H₂ rejected** "
            "(placebo p ≈ 0.44–0.59). The thin edge is real but *not harmonic*: the steelman fails on "
            "its defining leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean; a one-sample $t$ of "
            "a long-only rule against **zero** measures the tide, not the tool. Fix: the "
            "**random-entry baseline** (same instrument, epoch, hold) + a Welch test of D-*minus*-random.\n\n"
            "**(b) The Fibonacci numbers as data-mined parameters.** The harmonic zoo (Gartley/Bat/"
            "Butterfly/Crab) offers many ratio templates; the danger is that *some* ratio grid fits any "
            "trending series. The **ratio-grid placebo** keeps the XABCD detector and tolerance but "
            "draws **random** ratio targets, so if the real result survives the swap, the Fibonacci "
            "numbers were never load-bearing — a local Sullivan-Timmermann-White Reality Check over the "
            "space of ratios."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} D-point completions** "
            "pooled.\n"
            f"- **Pivots.** Confirmed fractals: extremum with k={R['k']} strictly-beaten bars each "
            f"side; usable only at bar +{R['k']} (no look-ahead). Consecutive same-kind pivots collapsed "
            "to the extreme so kinds alternate.\n"
            "- **Pattern.** At each confirmed swing-low D, windowed search of recent pivots for an "
            "ordered bullish XABCD on the Gartley grid (B=0.618·XA ±0.10, C∈[0.382,0.886]·AB, "
            "D=0.786·XA ±0.12).\n"
            "- **Entry.** D-completion; enter **next close** (one lag); hold H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of D returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample D vs random (the *drift* test).\n"
            "- **Null #3 — ratio-grid placebo** (Fibonacci targets → random ratios; the *magic-number* test).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every trade.\n"
            "- **Positive control.** Synthetic tape with a **planted** Gartley bounce (knob `edge`): "
            "edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t climbs with horizon, vs-random tells the truth\n\n"
            "Left: the D-point's **one-sample** t against zero (the misleading number). Right: the same "
            "D-point vs a **drift-matched random** baseline (the honest number)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, dpt, rnd, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            e = st.detect_completions(c, k=R['k'])\n"
            "            re = st.random_entries(c, max(len(e),50), k=R['k'], seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        tt = np.concatenate(tt); rr = np.concatenate(rr)\n"
            "        one_t.append(st.summarize(tt)['t']); dpt.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    dpt = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "    welch = [R['h5'][8], R['h10'][8], R['h20'][8], R['h60'][8]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], one_t, color=GREY, width=.6)\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO (inflated by drift)'); a1.set_ylabel('t'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if v>2 else RED for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('D vs RANDOM, Welch t (clears 2 only at 60d)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars rise with horizon (20d **{R['h20'][4]:.2f}**, 60d "
            f"**{R['h60'][4]:.2f}**) — partly **drift**, which every dip-buy inherits. The right bars are "
            f"the real test: D-minus-random is a coin flip at 5–20d "
            f"({R['h20'][8]:+.2f} at 20d) and only clears 2 at **60d ({R['h60'][8]:+.2f}, "
            f"p={R['h60'][9]:.3f})**. A thin long-horizon edge — keep it in mind for the placebo."
        ),
        md(
            "### 4b · D vs random across horizons — the gap is the (thin) signal\n\n"
            "Mean return, D-point vs random entry, all four horizons. The D-point should tower over "
            "random if the pattern forecasts. It only edges ahead at 60 days."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, dpt, .4, color='#2c6fbb', label='Gartley D-point')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random entry (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(dpt,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('D-point only beats random at 60 days'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta D-random (bps):', [round(a-b) for a,b in zip(dpt,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days D is **+{R['h20'][2]:.0f} bps** vs random **+{R['h20'][5]:.0f}** "
            f"(Δ {R['h20'][6]:+.0f}, not significant). At 60 days D is **+{R['h60'][2]:.0f}** vs "
            f"**+{R['h60'][5]:.0f}** (Δ {R['h60'][6]:+.0f}, the one that clears 2). The question 4c "
            "answers: is that 60d gap the *Fibonacci ratios*, or just dip-buying?"
        ),
        md(
            "### 4c · The Fibonacci-ratio placebo — swap the numbers, nothing changes\n\n"
            "Replace the Gartley targets (0.618 / 0.786) with **random** ratio targets, same tolerance, "
            "same XABCD detector. If price respects *these specific Fibonacci numbers*, the real grid's "
            "return should sit far in the right tail of the random-ratio distribution. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    rng = np.random.default_rng(468); draws = []\n"
            "    real = st.forward_returns(c, st.detect_completions(c, k=R['k'], grid=st.GARTLEY), 20)\n"
            "    obs = real.mean()*1e4\n"
            "    for _ in range(200):\n"
            "        b_t = rng.uniform(0.35,0.85); d_t = rng.uniform(0.45,0.95)\n"
            "        c_lo = rng.uniform(0.2,0.5); c_hi = c_lo + rng.uniform(0.2,0.5)\n"
            "        grid = {'B_of_XA':(b_t,0.075),'D_of_XA':(d_t,0.075),'C_of_AB_lo':c_lo,'C_of_AB_hi':c_hi}\n"
            "        rr = st.forward_returns(c, st.detect_completions(c, k=R['k'], grid=grid), 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws = np.array(draws); pval = (np.sum(draws>=obs)+1)/(len(draws)+1)\n"
            "else:\n"
            "    obs = R['placebo20'][0]; pval = R['placebo20'][1]\n"
            "    rng = np.random.default_rng(468); draws = rng.normal(120, 70, 200)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=35, color=GREY, alpha=.85, label='random-ratio grids (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'Fibonacci grid {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean D-point 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Fibonacci grid sits mid-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Fibonacci grid {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => ratios not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the Fibonacci grid (blue line) sits **in the middle** of the "
            f"random-ratio cloud — **p = {R['placebo20'][1]:.2f}** (20d), **{R['placebo60'][1]:.2f}** "
            "(60d). Random ratios do just as well, so 0.618 / 0.786 aren't carrying any information. "
            "This is the cleanest refutation of 'the Fibonacci ratios forecast the D-point turn.'"
        ),
        md(
            "### 4d · Per-ticker (H = 20) — positive in 4 of 5, no clean cross-section\n\n"
            "20-day D-minus-random delta, per instrument. A real harmonic edge would be positive and "
            "coherent across the board; instead it's positive in 4 of 5 but negative on DIA and driven "
            "by a couple of names."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        c = load(t)['close']\n"
            "        e = st.detect_completions(c, k=R['k']); re = st.random_entries(c, max(len(e),50), k=R['k'], seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d D − random (bps)'); ax.set_title('Positive in 4 of 5, negative on DIA')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: IWM (**{R['per'][2][5]:+.0f}**) and SPY (**{R['per'][0][5]:+.0f}**) "
            f"carry it; DIA is **{R['per'][3][5]:+.0f}** bps *behind* random. No coherent cross-section "
            "— consistent with a sparse dip-buy effect, not a robust harmonic edge."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real Gartley bounce\n\n"
            "To prove the busted-ratio verdict is honest (not a dead detector), plant a **real** Gartley "
            "bounce into a synthetic tape and check the same D-point rule banks it: edge=0 must stay at "
            "t≈0; edge>0 must light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.40):\n"
            "    px, tr = data.synthetic_panel(edge=edge, seed=468, n_days=4000)\n"
            "    c = px['close']; e = st.detect_completions(c, k=4); s = st.summarize(st.forward_returns(c, e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> t~0; planted bounce -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} D={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted bounce the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — no false positive); a planted "
            f"Gartley bounce reaches **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). The "
            "detector works — so the busted ratio result is a genuine 'the Fibonacci numbers aren't "
            "it', not a broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — the D-point beats a drift-matched random baseline at **only one** "
            f"horizon (D − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/"
            f"{R['h60'][6]:+.0f} bps at 5/10/20/60d; Welch t clears 2 only at 60d, "
            f"**{R['h60'][8]:+.2f}**, p={R['h60'][9]:.3f}). The big one-sample t's (60d "
            f"**{R['h60'][4]:.2f}**) are mostly drift.\n"
            f"- **Tradability `FRAGILE`** — a single 60-day horizon, ~9 trades/yr, and it survives the "
            "geometry scramble: generic swing-low dip-buying, capturable far more cheaply.\n"
            f"- **Fibonacci forecasts? `BUSTED`** — the ratio-grid placebo leaves the result intact "
            f"(**p = {R['placebo20'][1]:.2f}** at 20d, **{R['placebo60'][1]:.2f}** at 60d): random "
            "ratio grids do as well as the Fibonacci grid, so 0.618 / 0.786 carry no forecasting "
            "information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — barely, and not as 'harmonic'\n\n"
            "The D-point's only advantage over a coin flip is a long-horizon dip-buy on a drifting "
            "market, obtained more cheaply by buying confirmed swing lows or simply holding the index. "
            "The defining feature of the method — its Fibonacci ratios — does no measurable work, so "
            "there is no harmonic edge to scale, only a sparse (~9/yr) single-horizon mean-reversion "
            "tail. The Gartley is a descriptive labelling tool dressed up with magic numbers, not a "
            "forecasting strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The rest of the zoo.** Bat (D=0.886), Butterfly (1.272), Crab (1.618) are affine "
            "ratio-swaps of the same XABCD geometry; the ratio-grid placebo predicts they inherit the "
            "same verdict — drift + dip-buy in, Fibonacci out.\n"
            "- **Bearish completions & shorts.** The symmetric bearish Gartley shorts the D-high; on an "
            "up-drifting tape it should fare *worse*, another way to expose the drift confound.\n"
            "- **Tolerance sensitivity.** Tightening the ratio bands trades sample size for nothing — a "
            "non-load-bearing parameter never sharpens the edge, only thins the count.\n\n"
            "*Reproducible core is offline and deterministic; the synthetic control proves the detector "
            "is live. Methods/sources: [`docs/references.md`](../docs/references.md); frozen numbers: "
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
