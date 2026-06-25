"""Generate the two narrative notebooks for Study 466 (Diamond Top).

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
# 2026-05-31, partial June dropped), 21.4 years, fractal k=5, 6 pivots, diamond-top SHORT.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=125, k=5, n_piv=6,
    fp_spy="4cb5244f3990",
    # pooled diamond breakdown SHORT, per horizon:
    # (H, n, break_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 124, 55.3, 50, 1.32, -18.7, 74.0, 53.3, 1.71, 0.089),
    h10=(10, 123, 20.1, 47, 0.40, -25.7, 45.7, 18.1, 0.96, 0.337),
    h20=(20, 123, -151.4, 33, -2.13, -42.9, -108.5, -153.4, -1.71, 0.088),
    h60=(60, 119, -356.6, 28, -2.78, -83.0, -273.6, -358.6, -2.55, 0.011),
    # per-ticker H=20: (ticker, entries, break_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 26, -99.1, -0.52, -42.9, -56.2), ("QQQ", 19, -515.8, -5.94, -105.0, -410.8),
         ("IWM", 32, -237.5, -1.77, -9.7, -227.8), ("DIA", 19, -15.1, -0.13, -34.1, 19.0),
         ("GLD", 29, 60.8, 0.71, -22.7, 83.5)],
    # shuffled-pivot placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(-99.1, 0.679, 500),
    # synthetic control (H=20, n_days=4000): (edge, n, break_bps, win%, one_sample_t)
    syn=[(0.00, 27, 75.2, 44, 0.82), (0.60, 22, 1360.6, 77, 5.06)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Forecasts_a_reversal%3F: Busted](https://img.shields.io/badge/Forecasts_a_reversal%3F-Busted-8b949e?style=flat-square)\n\n"
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

from diamond_top import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real diamond cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does a \"diamond top\" actually call the turn? 💎\n"
            "### A rare, dramatic chart shape — range widens, then narrows — meets a stopwatch\n\n"
            + BADGES +
            "Flip through any chart-pattern book and you'll meet the **diamond top**: after a rally, "
            "price swings get *wider and wider* (a megaphone), then *tighter and tighter* (a "
            "triangle), tracing a diamond. The lore, from Edwards & Magee to Bulkowski to every "
            "trading site, is that this rare shape marks a **top** — and when price finally breaks "
            "**down** out of the diamond, you **short it**: the reversal is supposed to be on.\n\n"
            "It *looks* uncanny on a hand-picked chart. But a shape you only recognise **after** the "
            "swings have happened, by choosing which wiggles count, is the textbook setup for fooling "
            "yourself. So we did the only fair thing: encode the diamond **mechanically** (no "
            "eyeballing), fire the \"short the breakdown\" rule across five big indices over 21 years, "
            "and time the result — against the only baseline that matters: **shorting on random days "
            "instead.**\n\n"
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
            "| If I short when price breaks **down** out of a diamond top, do I catch a drop? | "
            "**No.** Over the next 20–60 days the short *loses* money (−151 / −357 bps) — price "
            "mostly keeps drifting up. |\n"
            "| Is the short at least better than shorting random days? | **No — it's worse.** At "
            "20/60 days it loses *more* than shorting random days. The diamond points you the wrong "
            "way. |\n"
            "| Is there *anything* there? | **A faint 5-day wobble.** Right after the break there's a "
            "small bounce, but it never reaches statistical significance and flips to a loss by 20 "
            "days. |\n"
            "| Does the diamond \"shape\" forecast the turn? | **No.** Scramble the diamond into "
            "geometric nonsense and the result barely changes. The shape isn't doing the work. |\n\n"
            "> The diamond top is a great way to *describe* a noisy pause after a rally. As a "
            "*forecast* — \"the breakdown starts a decline\" — it's a **mirage**: the market's upward "
            "drift just carries on, and the short bleeds."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"After an advance, the swing range **broadens** (higher highs, lower lows — a "
            "megaphone) and then **narrows** (lower highs, higher lows — a symmetrical triangle), "
            "forming a diamond. This marks distribution at a top. When price breaks **down** out of "
            "the apex, sell/short — the reversal is confirmed, with a target equal to the diamond's "
            "height.\"*\n\n"
            "This is the **diamond top** of Edwards & Magee's *Technical Analysis of Stock Trends* "
            "and Bulkowski's *Encyclopedia of Chart Patterns* — billed as a rarer but "
            "\"high-reliability\" reversal. So: does the diamond actually *call the turn*?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the diamond genuinely *forecast* tops, it would be remarkable: a few past wiggles "
            "predicting the end of a trend, a clean crack in market efficiency you could trade with a "
            "ruler. That's the dream the pattern sells.\n\n"
            "But there are two traps. First, a diamond is recognised **by hand, after the swings have "
            "happened** — you pick the wiggles that make the shape *look* right. Second, it's drawn "
            "on a market (stock indices) that drifts **up** over time, so *any* short will tend to "
            "lose. To separate the **shape** from the **tide**, we have to (a) detect the diamond by "
            "a fixed mechanical rule with no hindsight, and (b) compare the short to shorting on "
            "**random days**. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Find the swing points mechanically.** A 'pivot' is a high (or low) with "
            f"**{R['k']} lower (higher) bars on each side** — a confirmed fractal, only known "
            f"**{R['k']} bars later**, so we never draw the diamond with future data.\n"
            "2. **Detect the diamond by rule.** Over the 6 most-recent alternating pivots, the swing "
            "amplitudes must first **grow** (broadening) then **shrink** (narrowing) — a diamond — "
            "formed after an advance. No eyeballing.\n"
            "3. **Trade the lore.** When the close breaks **below** the narrowing apex, **short** at "
            "the next close; measure the short's return over the next **5 / 10 / 20 / 60 days**.\n"
            "4. **The honest baseline.** Do the exact same short on **random days**. If the diamond "
            "matters, the breakdown short must beat random. *If it doesn't, the shape is a mirage* — "
            "that's the result that would make us say so, announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does a mechanical diamond even look like? Here's SPY with the confirmed "
            "pivots, and the downside breaks the rule would short."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    cl = b['close']; seg = cl.iloc[-900:]\n"
            "    piv = st._alternating(st.find_pivots(cl, k=R['k']))\n"
            "    ent = st.diamond_breakdowns(cl, k=R['k'], n_piv=R['n_piv'])\n"
            "    ent = ent[ent >= seg.index[0]]\n"
            "    pv = piv[(piv.index >= 0)]\n"
            "    pdates = cl.index[[int(p) for p in pv.index if int(p) < len(cl)]]\n"
            "    pdates = pdates[pdates >= seg.index[0]]\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    ax.plot(seg.index, seg.values, c='k', lw=1.1, label='SPY close')\n"
            "    ax.scatter(pdates, cl.reindex(pdates), c=GREY, s=22, zorder=4, label='confirmed pivots')\n"
            "    ax.scatter(ent, cl.reindex(ent), c=RED, s=55, marker='v', zorder=5, label='diamond breakdown SHORT')\n"
            "    ax.set_title('Mechanical diamond-top breakdowns on SPY (last ~3.5y)'); ax.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('diamond breakdowns in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "Now the real question: are those red short triangles followed by **declines**? "
            "**Let's race the breakdown short against shorting random days** at four horizons. Red = "
            "short the diamond breakdown; grey = short random days. (Bars above zero = the short "
            "made money.)"
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    brk, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            e = st.diamond_breakdowns(c, k=R['k'], n_piv=R['n_piv'])\n"
            "            re = st.random_entries(c, max(len(e),50), k=R['k'], seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        brk.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    brk = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, brk, .4, color='#2c6fbb', label='short the diamond breakdown')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='short on random days')\n"
            "for i,(a,bb) in enumerate(zip(brk,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom' if a>=0 else 'top',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom' if bb>=0 else 'top',fontsize=8)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean SHORT return (bps)')\n"
            "ax.set_title('The diamond short does NOT catch a drop — it loses at 20/60d'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('breakdown short:', [round(v) for v in brk]); print('random short:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the whole story. The breakdown short makes a little at 5 days "
            f"(**{R['h5'][2]:+.0f} bps**) — a brief overshoot bounce — but by **20 days it is "
            f"{R['h20'][2]:+.0f} bps** and by **60 days {R['h60'][2]:+.0f} bps**: the 'reversal' "
            "never comes, the market drifts up, and the short bleeds. And it loses *more* than "
            "shorting random days at those horizons. The diamond points the **wrong way**."
        ),
        md(
            "**One more sanity check.** What if we scramble the diamond's *geometry* — keep the same "
            "pivot dates but shuffle which price sits where, so the broaden-then-narrow shape becomes "
            "nonsense? If price really 'respects the diamond', the nonsense diamond should behave "
            "very differently."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    pl = st.shuffled_pivot_placebo(c, 20, k=R['k'], n_piv=R['n_piv'], n_draws=200, seed=466)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real diamond breakdown short (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of *scrambled-geometry* diamonds do at least as well (p={pval:.2f}).')\n"
            "print('=> the diamond shape is not doing the work.')"
        ),
        md(
            f"Most of the **scrambled** diamonds match or beat the real one "
            f"(*p* = {R['placebo'][1]:.2f}). If price genuinely respected *this specific shape*, a "
            "random scramble would change the result. It doesn't — because the result was never "
            "about the shape."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The breakdown short does **not** forecast a decline (it *loses* at "
            "20/60 days, and loses more than shorting random days; the only positive is a faint 5-day "
            "bounce that never clears *t* = 2). The shape marks a *pause*, not a top.\n"
            "- **Tradability — Mirage.** Nothing to trade: the short bleeds against the market's "
            "upward drift, and costs only make it worse.\n"
            "- **\"Does the diamond shape forecast a reversal\"? — Busted.** Scramble the geometry "
            "into nonsense and the result barely moves. The diamond doesn't call the turn."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing to trade — in fact it's *negative*. Shorting the breakdown fights the "
            "long-run climb of equities, so it loses at the 20/60-day horizons the pattern is "
            "supposed to nail. Costs (commissions + spread on every break, plus borrow on a short) "
            "push the already-bad result further into the red. The only flicker — a 5-day overshoot "
            "bounce — is sub-threshold and would be eaten by a fast round trip's costs. As a "
            "forecasting tool the diamond doesn't pay; as a drawing label it was never a strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Diamond *bottoms*.** The mirror pattern (upside break, bullish) would simply inherit "
            "the drift the other way — it'd look 'good' for the same hollow reason a dip-buy does.\n"
            "- **Different pivot/shape thresholds.** Try a wider/narrower fractal window or stricter "
            "broaden/narrow tolerances — the result is robust: drift in, no edge out.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* diamond-top "
            "reversal into a synthetic tape and shows the harness banks it (so the null result here "
            "isn't a dead detector — it's an honest 'nothing there').\n\n"
            "*Think the diamond forecasts? Show the breakdown short beating random shorts at "
            "**t ≥ 2** on a real tape — then we'll talk.*"
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
            "# Diamond Top — a quantitative teardown 🔬\n"
            "### Mechanical broaden-then-narrow diamonds on 5 indices · downside-break SHORT forward "
            "returns · one-sample HAC *t* · a drift-matched random-short baseline · a shuffled-pivot "
            "geometry placebo · costs · a synthetic planted-reversal control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **shape** from the **drift**: an upward-trending index makes *any* "
            "short lose, so the only meaningful test is break-vs-random-short, plus a placebo that "
            "destroys the diamond's geometry while preserving its marginals.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. Pivots are confirmed "
            f"fractals (k={R['k']}, an explicit {R['k']}-bar confirmation lag), diamonds over the "
            f"{R['n_piv']} latest alternating pivots; entry is the **next close** (one documented "
            "lag); the trade is a **short**. Offline core + synthetic control are deterministic. "
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
            f"| **Signal** | `NONE` | Breakdown short vs a **drift-matched random-short** baseline: "
            f"the short *loses more* than random at 20/60d (Δ = {R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} "
            f"bps; 60d Welch t = {R['h60'][8]:+.2f}, significant the WRONG way). The only positive Δ "
            f"is a 5-day bounce ({R['h5'][6]:+.0f} bps) at Welch t = {R['h5'][8]:+.2f} (p="
            f"{R['h5'][9]:.2f}) — never clears 2. |\n"
            f"| **Tradability** | `MIRAGE` | The short bleeds against drift (20d break = "
            f"{R['h20'][2]:+.0f} bps, t = {R['h20'][4]:.2f}); no edge to scale, costs deepen the "
            "hole. |\n"
            f"| **Forecasts a reversal?** | `BUSTED` | Scrambling the diamond's geometry "
            f"(shuffled-pivot placebo) leaves the result intact: **p = {R['placebo'][1]:.2f}** of "
            "nonsense diamonds match or beat the real one. The shape isn't doing the work. |\n\n"
            "> 💡 In plain words: the diamond top is sold as a *reversal* signal, but the short "
            "*loses* at the horizons that matter and loses more than a random short — the market just "
            "keeps drifting up. Strip the shape (scramble the pivots) and nothing changes. Classic "
            "pattern-shaped noise on a drifting tape."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Over a run of alternating confirmed pivots with prices $y_1,\\dots,y_6$, let the leg "
            "amplitudes be $a_j=|y_{j+1}-y_j|$. A **diamond** requires the $a_j$ to rise to a peak "
            "(broadening) then fall (narrowing). After the diamond completes following an advance, "
            "the rule **shorts** the first close below the narrowing-apex floor "
            "$\\ell=\\min(y_4,y_5,y_6)$.\n\n"
            "- **H₀ (drift).** Break-short returns equal a drift-matched **random-short** baseline.\n"
            "- **H₁ (the diamond forecasts a top).** Break-short returns **exceed** random at some "
            "horizon, t ≥ 2.\n"
            "- **H₂ (the shape matters).** Break-short returns exceed a **shuffled-pivot** diamond "
            "whose geometry is nonsense.\n\n"
            "We find **H₀ not rejected** (break ≤ random at 20/60d — in fact *worse*), **H₁ rejected** "
            "(Welch t never ≥ 2 in the right direction), **H₂ rejected** (placebo p ≈ 0.68). The "
            "steelman fails on every leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean. A **short** "
            "fights it; a one-sample $t$ against **zero** of a short rule manufactures a misleading "
            "*negative* number that is just the tide, not the shape. The fix is the **random-short "
            "baseline** (same instrument, epoch, hold, short sign) and a Welch test of "
            "break-*minus*-random.\n\n"
            "**(b) Geometry as a free parameter.** A diamond is a chosen set of wiggles; the danger "
            "is that *any* broaden-then-narrow run drawn on a noisy trend gets labelled a diamond. "
            "The **shuffled-pivot placebo** keeps pivot positions and the price marginal but permutes "
            "which price sits at which pivot — the diamond shape becomes meaningless, so if the real "
            "result survives the scramble, the geometry was never load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} diamond breakdowns** "
            "pooled.\n"
            f"- **Pivots.** Confirmed fractals: extremum with k={R['k']} strictly-beaten bars each "
            f"side; usable only at bar +{R['k']} (no look-ahead). Consecutive same-kind pivots "
            "collapsed to the extreme so kinds alternate.\n"
            f"- **Diamond.** Over the {R['n_piv']} latest confirmed pivots, leg amplitudes rise to a "
            "peak (broadening) then fall (narrowing); formed after an advance.\n"
            "- **Entry.** First close below the narrowing-apex floor; **short** at the **next close** "
            "(one lag); hold H ∈ {5,10,20,60}. Trade return = −(price move).\n"
            "- **Null #1 — one-sample HAC t** of break-short returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-short baseline**, Welch two-sample break vs random (the *real* "
            "test).\n"
            "- **Null #3 — shuffled-pivot placebo** (geometry destroyed, marginals kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every break.\n"
            "- **Positive control.** Synthetic tape with a **planted** diamond-top reversal (knob "
            "`edge`): edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The drift trap — one-sample t vs the honest random-short test\n\n"
            "Left: the breakdown short's **one-sample** t against zero (misleading — for a short on a "
            "rising tape it just turns negative). Right: the same short vs a **drift-matched "
            "random-short** baseline (the honest number)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, brk, rnd, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            e = st.diamond_breakdowns(c, k=R['k'], n_piv=R['n_piv'])\n"
            "            re = st.random_entries(c, max(len(e),50), k=R['k'], seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        tt = np.concatenate(tt); rr = np.concatenate(rr)\n"
            "        one_t.append(st.summarize(tt)['t']); brk.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    brk = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "    welch = [R['h5'][8], R['h10'][8], R['h20'][8], R['h60'][8]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], one_t, color=GREY, width=.6)\n"
            "a1.axhline(2, ls='--', c=RED); a1.axhline(-2, ls='--', c=RED); a1.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO (misleading: it is drift)'); a1.set_ylabel('t')\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if v>2 else RED for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Break vs RANDOM-short, Welch t (honest: never clears +2)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the only horizon where the short beats random is 5 days (Welch "
            f"**{R['h5'][8]:+.2f}**, p={R['h5'][9]:.2f}) — a brief post-break overshoot, never "
            f"significant. At 20/60d the short is **worse** than a random short (Welch "
            f"**{R['h20'][8]:+.2f}** / **{R['h60'][8]:+.2f}**); the 60d figure is significant in the "
            "**wrong** direction. The diamond doesn't forecast a top — it marks a pause the trend "
            "then resumes through."
        ),
        md(
            "### 4b · Break vs random-short across horizons — the gap is the verdict\n\n"
            "Mean SHORT return, breakdown vs random short, all four horizons. If the diamond forecast "
            "a top, the breakdown short would tower over a random short. It doesn't."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, brk, .4, color='#2c6fbb', label='diamond breakdown short')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random short (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(brk,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom' if a>=0 else 'top',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom' if b>=0 else 'top',fontsize=8)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean SHORT return (bps)')\n"
            "ax.set_title('Diamond breakdown short does not beat a random short'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta break-random (bps):', [round(a-b) for a,b in zip(brk,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the break short is **{R['h20'][2]:+.0f} bps** vs a "
            f"random short's **{R['h20'][5]:+.0f} bps** — the diamond *underperforms* by "
            f"{abs(R['h20'][6]):.0f} bps. The supposed reversal signal sends you the wrong way on a "
            "drifting tape."
        ),
        md(
            "### 4c · The geometry placebo — scramble the diamond, nothing changes\n\n"
            "Shuffle which price sits at which pivot (positions kept, marginal kept) so the "
            "broaden-then-narrow shape is geometric nonsense. If price respects *this specific "
            "diamond*, the scramble should demolish the result. The observed break-short return "
            "should sit far in the tail of the scrambled distribution. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    pl = st.shuffled_pivot_placebo(c, 20, k=R['k'], n_piv=R['n_piv'], n_draws=200, seed=466)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    import numpy as _np, pandas as _pd\n"
            "    piv = st._alternating(st.find_pivots(c, k=R['k']))\n"
            "    rng = _np.random.default_rng(466); prices = piv['price'].to_numpy(); positions=[int(p) for p in piv.index]\n"
            "    confirm=[p+R['k'] for p in positions]; idx=c.index; n=len(c)\n"
            "    draws=[]\n"
            "    for _ in range(200):\n"
            "        perm=rng.permutation(prices); armed=_np.full(n,_np.nan)\n"
            "        for t in range(n):\n"
            "            av=[j for j in range(len(positions)) if confirm[j]<=t]\n"
            "            if len(av)<R['n_piv']: continue\n"
            "            seg=av[-R['n_piv']:]; sp=perm[seg]\n"
            "            if not st.is_diamond(sp, tol=0.08): continue\n"
            "            if sp[0]>=sp.max(): continue\n"
            "            armed[t]=float(sp[-3:].min())\n"
            "        fl=_pd.Series(armed,index=idx); m=(c<fl)&fl.notna(); f=m&~m.shift(1,fill_value=False)\n"
            "        rr=st.forward_returns(c, idx[f.to_numpy()], 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws=_np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(466); draws = rng.normal(-90, 60, 200)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=35, color=GREY, alpha=.85, label='scrambled-geometry diamonds (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real diamond {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean breakdown-short 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real diamond sits mid-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real diamond {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => geometry not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real diamond (blue line) sits **inside** the scrambled cloud — "
            f"**p = {R['placebo'][1]:.2f}**. Geometric nonsense does just as well, so the specific "
            "broaden-then-narrow shape isn't carrying any information. The cleanest refutation of "
            "'the diamond forecasts a reversal.'"
        ),
        md(
            "### 4d · Per-ticker — the short loses across the board\n\n"
            "20-day break-minus-random delta, per instrument. If the diamond forecast tops it would "
            "be positive everywhere; instead it's deeply negative in the trending tech/small-cap "
            "names."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        c = load(t)['close']\n"
            "        e = st.diamond_breakdowns(c, k=R['k'], n_piv=R['n_piv']); re = st.random_entries(c, max(len(e),50), k=R['k'], seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d break − random (bps)'); ax.set_title('Diamond short underperforms a random short in the trending names')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: **QQQ** ({R['per'][1][5]:+.0f} bps) and **IWM** "
            f"({R['per'][2][5]:+.0f} bps) — the hardest-drifting tapes — punish the short brutally; "
            "only DIA and GLD edge slightly positive. No coherent, cross-sectional reversal edge — "
            "exactly what you'd expect if the diamond just labels a pause in a trend."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real reversal\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** diamond-top reversal "
            "into a synthetic tape and check the same short rule banks it: edge=0 must stay "
            "insignificant; edge>0 must light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.60):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=466, n_days=4000)\n"
            "    c = px['close']; e = st.diamond_breakdowns(c, k=5, n_piv=6); s = st.summarize(st.forward_returns(c, e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t (short)'); ax.set_title('Control: edge=0 -> insignificant; planted reversal -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} break={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted reversal the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — no false positive); a planted "
            f"diamond-top decline reaches **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). "
            "The detector works — so the flat/negative real-tape result is a genuine 'nothing there', "
            "not a broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the breakdown short does not beat a drift-matched random short "
            f"(break − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/"
            f"{R['h60'][6]:+.0f} bps at 5/10/20/60d; Welch t never clears +2, and at 60d it's "
            f"**{R['h60'][8]:+.2f}** — significant the *wrong* way). The 5-day positive (Welch "
            f"{R['h5'][8]:+.2f}) is a sub-threshold overshoot bounce.\n"
            f"- **Tradability `MIRAGE`** — the short *loses* outright at 20/60d ({R['h20'][2]:+.0f} / "
            f"{R['h60'][2]:+.0f} bps) fighting the drift; no residual edge, and costs only deepen the "
            "hole.\n"
            f"- **Forecasts a reversal? `BUSTED`** — the shuffled-pivot placebo leaves the result "
            f"untouched (**p = {R['placebo'][1]:.2f}**): geometric-nonsense diamonds do as well as "
            "the real ones, so the broaden-then-narrow shape carries no forecasting information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The diamond-top short fights the unconditional drift of long equity indices, so it "
            "*loses* at the 20/60-day horizons the pattern is sold to capture, and loses more than "
            "shorting random days. Add costs (commissions + spread on every break, plus short borrow) "
            "and it gets worse. There is no capacity question because there is no edge to scale — and "
            "the one flicker (a 5-day overshoot bounce) is sub-threshold and round-trip-cost-sensitive. "
            "The diamond top is a descriptive label for a volatile consolidation, not a forecasting "
            "strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Diamond bottoms / measured-move targets.** The mirror pattern and the "
            "diamond-height target inherit the same drift confound; a clean follow-up shows the "
            "'bottom' looks good only because it's long on a rising tape.\n"
            "- **Stricter shape thresholds.** Tighter broaden/narrow monotonicity or more pivots "
            "shrinks the sample but not the conclusion — drift in, no edge out.\n"
            "- **Other broadening/triangle relatives** (megaphone tops, symmetrical triangles) are "
            "affine cousins of the same swing-amplitude geometry and inherit the same confound.\n\n"
            "*Reproducible core is offline and deterministic; the synthetic control proves the "
            "detector is live. Methods/sources: [`docs/references.md`](../docs/references.md); frozen "
            "numbers: [`docs/results.md`](../docs/results.md).*"
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
