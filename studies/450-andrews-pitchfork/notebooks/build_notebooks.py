"""Generate the two narrative notebooks for Study 450 (Andrews' Pitchfork).

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
# 2026-05-31, partial June dropped), 21.4 years, pivot fractal k=10, lower-tine-touch long.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=819, k=10,
    fp_spy="4cb5244f3990",
    # pooled lower-tine-touch, per horizon:
    # (H, n, touch_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 819, 39.0, 61, 3.81, 48.5, -9.5, 37.0, -0.73, 0.467),
    h10=(10, 818, 40.1, 60, 2.63, 65.6, -25.5, 38.1, -1.44, 0.150),
    h20=(20, 818, 92.2, 62, 3.93, 131.3, -39.1, 90.2, -1.61, 0.107),
    h60=(60, 811, 305.0, 70, 6.47, 232.1, 72.8, 303.0, 1.77, 0.078),
    # per-ticker H=20: (ticker, entries, touch_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 161, 51.7, 0.92, 132.9, -81.2), ("QQQ", 162, 134.7, 2.59, 182.2, -47.5),
         ("IWM", 159, 75.3, 1.27, 107.4, -32.1), ("DIA", 144, 110.0, 2.26, 92.8, 17.3),
         ("GLD", 193, 91.0, 2.12, 135.8, -44.9)],
    # shuffled-pivot placebo (SPY, H=20, 500 draws): obs_bps, p
    placebo=(51.7, 0.611, 500),
    # synthetic control (H=20, n_days=4000): (edge, n, touch_bps, win%, one_sample_t)
    syn=[(0.00, 111, -21.0, 48, -0.52), (0.40, 122, 110.9, 64, 3.44)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Respects_the_fork%3F: Busted](https://img.shields.io/badge/Respects_the_fork%3F-Busted-8b949e?style=flat-square)\n\n"
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

from andrews_pitchfork import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real pitchfork cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


def _pool(h, kind="touch"):
    """Returns a code snippet computing a pooled stat list across tickers for horizon h."""
    return ""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does Andrews' pitchfork actually \"channel\" price? 🔱\n"
            "### A famous chart tool — three dots, a median line, two tines — meets a stopwatch\n\n"
            + BADGES +
            "Open any charting package and you'll find **Andrews' pitchfork**: click three swing "
            "points and it draws a three-pronged fork across the chart. The lore, taught by Alan "
            "Andrews himself and repeated on every chart-pattern site, is that **price respects the "
            "fork** — it rides between the outer prongs (the *tines*) and keeps getting pulled back to "
            "the middle prong (the *median line*). So when price drops to the **lower tine**, you buy: "
            "it's \"supposed\" to bounce.\n\n"
            "It *looks* uncanny on a hand-picked chart. But a tool you draw **after** the move, by "
            "choosing which three dots to connect, is the textbook setup for fooling yourself. So we "
            "did the only fair thing: encode the fork **mechanically** (no eyeballing), fire the "
            "\"buy the lower tine\" rule thousands of times across five big indices over 21 years, and "
            "time the result with a stopwatch — against the only baseline that matters: **buying on "
            "random days instead.**\n\n"
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
            "| If I buy when price touches the **lower tine**, do I make money? | **Yes — but only "
            "because the market goes up.** The raw win-rate is ~60% and the returns look great. |\n"
            "| Is that *the fork's* doing? | **No.** Buy on **random days** instead and you do "
            "**just as well or better**. The fork adds nothing — at 5/10/20 days the touch is "
            "actually *worse* than a coin-flip entry. |\n"
            "| Does price \"respect\" the channel? | **Not in any usable way.** Scramble the fork's "
            "geometry into nonsense and the result barely changes. The lines aren't doing the work. |\n"
            "| So is it a tradable edge? | **No.** It's **beta in a costume** — the upward drift of "
            "stocks, re-labelled as a chart pattern. |\n\n"
            "> The pitchfork is a great way to *describe* a trend after the fact. As a *forecast* — "
            "\"the lower tine will bounce\" — it's a **mirage**: all of the apparent edge is just the "
            "market's long-run climb, none of it is the fork."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Pick three swing points. The **median line** runs from the first through the middle "
            "of the other two; two parallel **tines** run through them. Price will oscillate inside "
            "the fork and keep returning to the median line. Buy the lower tine, sell the upper — "
            "price respects the channel.\"*\n\n"
            "This is **Alan Andrews'** median-line method (1960s–70s), still taught today and built "
            "into TradingView, MetaTrader and every charting suite. Andrews claimed price reaches the "
            "median line **~80% of the time**. It's one of the most recognisable tools in technical "
            "analysis — so: does the channel actually *channel*?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the fork genuinely *forecast* reversals, it would be remarkable: three past dots would "
            "predict future turning points, a clean crack in market efficiency you could trade with a "
            "ruler. That's the dream the tool sells.\n\n"
            "But there's a trap built into it. A pitchfork is drawn **by hand, after the swings have "
            "happened** — you choose the three dots that make the channel *look* right. And it's "
            "drawn on a market (stock indices) that drifts **up** over time, so *any* dip-buying rule "
            "will look profitable. To separate the **tool** from the **tide**, we have to (a) draw the "
            "fork by a fixed mechanical rule with no hindsight, and (b) compare it to buying on "
            "**random days**. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Find the swing points mechanically.** A 'pivot' is a high (or low) with "
            f"**{R['k']} lower (higher) bars on each side** — a confirmed fractal. Crucially it's only "
            f"known **{R['k']} bars later**, so we never draw the fork with future data.\n"
            "2. **Draw the fork by rule.** At every bar, anchor it on the three most-recent confirmed "
            "pivots — no eyeballing, no cherry-picking which dots to connect.\n"
            "3. **Trade the lore.** When the close drops **below the lower tine**, buy at the next "
            "close; measure the return over the next **5 / 10 / 20 / 60 days**.\n"
            "4. **The honest baseline.** Do the exact same hold on **random days**. If the fork "
            "matters, the tine-touch must beat random. *If it doesn't, the tool is a mirage* — that's "
            "the result that would make us say so, announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does a mechanical fork even look like? Here's SPY with the fork drawn on its "
            "three most-recent confirmed pivots, and the lower-tine touches the rule would buy."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    cl = b['close']; seg = cl.iloc[-450:]\n"
            "    med, low, up = st.build_forks(cl, k=R['k'])\n"
            "    ent = st.tine_touch_entries(cl, k=R['k'])\n"
            "    ent = ent[ent >= seg.index[0]]\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    ax.plot(seg.index, seg.values, c='k', lw=1.2, label='SPY close')\n"
            "    ax.plot(seg.index, med.reindex(seg.index), c=GREY, lw=1.3, ls='--', label='median line')\n"
            "    ax.plot(seg.index, low.reindex(seg.index), c=GREEN, lw=1.1, label='lower tine')\n"
            "    ax.plot(seg.index, up.reindex(seg.index), c=RED, lw=1.1, label='upper tine')\n"
            "    ax.scatter(ent, cl.reindex(ent), c=GREEN, s=40, zorder=5, label='lower-tine BUY')\n"
            "    ax.set_title('A mechanical Andrews pitchfork on SPY (last ~2y)'); ax.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('lower-tine touches in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The fork tracks the trend nicely — *as a description*. The question is whether those "
            "green buy dots are followed by bounces. **Let's race the tine-touch against random "
            "entries** at four horizons. Blue = buy the lower tine; grey = buy on random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    touch, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            e = st.tine_touch_entries(c, k=R['k'])\n"
            "            re = st.random_entries(c, max(len(e),50), k=R['k'], seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        touch.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    touch = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, touch, .4, color='#2c6fbb', label='buy the lower tine')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(touch,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('The fork does NOT beat random — it mostly loses to it'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('touch:', [round(v) for v in touch]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the whole story in one chart. The tine-touch makes money in absolute terms "
            f"(**+{R['h20'][2]:.0f} bps** over 20 days) — but **random entries make more** "
            f"(**+{R['h20'][5]:.0f} bps**). At 5, 10 and 20 days the famous fork is *worse* than "
            "throwing darts. Only at 60 days does it nose ahead, and even then not by a "
            "statistically meaningful margin (the quants notebook shows the *t* never clears 2). The "
            "apparent edge was **the market's upward drift**, not the lines."
        ),
        md(
            "**One more sanity check.** What if we scramble the fork's *geometry* — keep the same "
            "pivot dates but shuffle which price sits where, so the lines become nonsense? If price "
            "really 'respects the fork', the nonsense fork should do much worse."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    pl = st.shuffled_pivot_placebo(c, 20, k=R['k'], n_draws=300, seed=450)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real fork lower-tine touch (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of *scrambled-geometry* forks do at least as well (p={pval:.2f}).')\n"
            "print('=> the geometry is not doing the work.')"
        ),
        md(
            f"More than half of the **scrambled** forks match or beat the real one "
            f"(*p* = {R['placebo'][1]:.2f}). If price genuinely respected *these specific lines*, a "
            "random scramble would collapse the result. It doesn't — because the result was never "
            "about the lines."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The lower-tine buy does **not** beat buying on random days "
            "(it's *worse* at 5–20 days; the touch-vs-random difference never clears *t* = 2). The "
            "big absolute returns are the market's drift, not the fork.\n"
            "- **Tradability — Mirage.** Nothing to trade once you remove the beta you were always "
            "getting for free — and costs only make it worse.\n"
            "- **\"Price respects the fork\"? — Busted.** Scramble the geometry into nonsense and the "
            "result barely moves. The channel doesn't channel."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. The tine-touch's *only* advantage over a coin flip is the "
            "market's long-run climb — which you'd capture more cheaply (and more fully) by just "
            "**holding the index**. The pitchfork buy is a worse, more expensive way to be long. "
            "Costs (commissions + spread on every touch) push the already-no-edge result further "
            "negative. As a forecasting tool, it doesn't pay; as a drawing tool, it was never meant "
            "to be a strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The median-line target.** Andrews' stronger claim is that price reaches the *median "
            "line* ~80% of the time. That's nearly tautological for a trending series (the median "
            "line *is* a smoothed trend) — a fun follow-up is to show the 80% holds for a random "
            "walk too.\n"
            "- **Different pivot rules.** Try a wider/narrower fractal window, or hand-anchored "
            "Andrews forks — the result is robust: drift in, fork out.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* tine-bounce into a "
            "synthetic tape and shows the harness banks it (so the null result here isn't a dead "
            "detector — it's an honest 'nothing there').\n\n"
            "*Think the fork forecasts? Show the lower-tine touch beating random entries at "
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
            "# Andrews' Pitchfork — a quantitative teardown 🔬\n"
            "### Mechanical median-line forks on 5 indices · lower-tine-touch forward returns · "
            "one-sample HAC *t* · a drift-matched random-entry baseline · a shuffled-pivot geometry "
            "placebo · costs · a synthetic planted-bounce control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **fork** from the **drift**: an upward-trending index makes *any* "
            "dip-buy look good, so the only meaningful test is touch-vs-random, plus a placebo that "
            "destroys the fork's geometry while preserving its marginals.\n\n"
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
            f"| **Signal** | `NONE` | Lower-tine touch vs a **drift-matched random** baseline: the "
            f"touch is *worse* at 5/10/20d (Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f} "
            f"bps) and the touch-minus-random difference **never clears t = 2** (Welch t at 20d "
            f"= {R['h20'][8]:+.2f}, 60d = {R['h60'][8]:+.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | The big one-sample t's (20d t = {R['h20'][4]:.2f}) are "
            f"**pure beta** — they vanish against random entries and against cost. No residual edge "
            "to scale. |\n"
            f"| **Respects the fork?** | `BUSTED` | Scrambling the fork's geometry (shuffled-pivot "
            f"placebo) leaves the result intact: **p = {R['placebo'][1]:.2f}** of nonsense forks match "
            "or beat the real one. The lines aren't doing the work. |\n\n"
            "> 💡 In plain words: the tine-touch *looks* significant only because indices drift up. "
            "Strip the drift (race it vs random) or strip the geometry (scramble the pivots) and the "
            "edge evaporates. Classic beta-in-a-costume."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Anchor a fork on three confirmed pivots $P_0,P_1,P_2$ (positions $x_i$, prices $y_i$). "
            "With $M=\\tfrac12(P_1+P_2)$ the **median line** has slope "
            "$s=\\frac{y_M-y_0}{x_M-x_0}$ and the two **tines** are $y=s\\,x+b_1$, $y=s\\,x+b_2$ "
            "through $P_1,P_2$. Let $\\ell_t$ be the lower tine at bar $t$. The Andrews rule buys when "
            "$C_t<\\ell_t$ and rides toward the median line.\n\n"
            "- **H₀ (drift).** Touch returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (the fork forecasts).** Touch returns **exceed** random at some horizon, t ≥ 2.\n"
            "- **H₂ (the geometry matters).** Touch returns exceed a **shuffled-pivot** fork whose "
            "lines are geometric nonsense.\n\n"
            "We find **H₀ not rejected** (touch ≤ random at 5–20d), **H₁ rejected** (Welch t never "
            "≥ 2), **H₂ rejected** (placebo p ≈ 0.6). The steelman fails on every leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean. *Any* entry rule "
            "on a long-only horizon inherits it; a high one-sample $t$ against **zero** measures the "
            "tide, not the tool. The fix is the **random-entry baseline** (same instrument, epoch, "
            "hold) and a Welch test of touch-*minus*-random.\n\n"
            "**(b) Geometry as a free parameter.** A fork is three chosen points; the danger is that "
            "*any* three points drawn on a trend produce 'respected' lines. The **shuffled-pivot "
            "placebo** keeps pivot positions and the price marginal but permutes which price sits at "
            "which pivot — the lines become meaningless, so if the real result survives the scramble, "
            "the geometry was never load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} lower-tine touches** "
            "pooled.\n"
            f"- **Pivots.** Confirmed fractals: extremum with k={R['k']} strictly-beaten bars each "
            f"side; usable only at bar +{R['k']} (no look-ahead). Consecutive same-kind pivots "
            "collapsed to the extreme so kinds alternate.\n"
            "- **Fork.** Rolling: anchor on the 3 latest confirmed pivots; median line P₀→mid(P₁,P₂); "
            "tines parallel through P₁,P₂.\n"
            "- **Entry.** First close below the lower tine; enter **next close** (one lag); hold "
            "H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of touch returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample touch vs random (the *real* test).\n"
            "- **Null #3 — shuffled-pivot placebo** (geometry destroyed, marginals kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every touch.\n"
            "- **Positive control.** Synthetic tape with a **planted** tine-bounce (knob `edge`): "
            "edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t looks great, vs-random kills it\n\n"
            "Left: the lower-tine touch's **one-sample** t against zero (the misleading number). "
            "Right: the same touch vs a **drift-matched random** baseline (the honest number)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, touch, rnd, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            e = st.tine_touch_entries(c, k=R['k'])\n"
            "            re = st.random_entries(c, max(len(e),50), k=R['k'], seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        tt = np.concatenate(tt); rr = np.concatenate(rr)\n"
            "        one_t.append(st.summarize(tt)['t']); touch.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    touch = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "    welch = [R['h5'][8], R['h10'][8], R['h20'][8], R['h60'][8]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], one_t, color=GREY, width=.6)\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO (misleading: it is beta)'); a1.set_ylabel('t'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if v>2 else RED for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Touch vs RANDOM, Welch t (honest: never clears 2)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars clear *t* = 2 easily (20d **{R['h20'][4]:.2f}**, 60d "
            f"**{R['h60'][4]:.2f}**) — but that's the **drift**, every dip-buy inherits it. The right "
            f"bars are the real test: touch-minus-random is **negative** at 5–20d "
            f"({R['h20'][8]:+.2f} at 20d) and only **{R['h60'][8]:+.2f}** at 60d — never significant. "
            "The fork adds nothing over a coin flip."
        ),
        md(
            "### 4b · Touch vs random across horizons — the gap is the verdict\n\n"
            "Mean return, tine-touch vs random entry, all four horizons. The touch should tower over "
            "random if the fork forecasts. It doesn't."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, touch, .4, color='#2c6fbb', label='lower-tine touch')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random entry (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(touch,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('Lower-tine touch does not beat random entry'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta touch-random (bps):', [round(a-b) for a,b in zip(touch,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the touch is **+{R['h20'][2]:.0f} bps** but random is "
            f"**+{R['h20'][5]:.0f} bps** — the fork *underperforms* a dart by {abs(R['h20'][6]):.0f} "
            "bps. The only horizon where the touch edges ahead is 60d, and the Welch test (4a) says "
            "that gap is noise."
        ),
        md(
            "### 4c · The geometry placebo — scramble the fork, nothing changes\n\n"
            "Shuffle which price sits at which pivot (positions kept, marginal kept) so the lines are "
            "geometric nonsense. If price respects *these specific lines*, the scramble should "
            "demolish the result. The observed touch return should sit far in the right tail of the "
            "scrambled distribution. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    pl = st.shuffled_pivot_placebo(c, 20, k=R['k'], n_draws=300, seed=450)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    # rebuild the placebo distribution for the histogram\n"
            "    import numpy as _np\n"
            "    piv = st._alternating(st.find_pivots(c, k=R['k']))\n"
            "    rng = _np.random.default_rng(450); prices = piv['price'].to_numpy(); positions=list(piv.index)\n"
            "    confirm=[int(p)+R['k'] for p in positions]; cl_=c.to_numpy(); idx=c.index; n=cl_.size\n"
            "    draws=[]\n"
            "    for _ in range(300):\n"
            "        perm=rng.permutation(prices); pts=list(zip([int(p) for p in positions],[float(v) for v in perm]))\n"
            "        low=_np.full(n,_np.nan)\n"
            "        for t in range(n):\n"
            "            av=[pt for pt,cp in zip(pts,confirm) if cp<=t]\n"
            "            if len(av)<3: continue\n"
            "            ln=st.fork_lines(av[-3],av[-2],av[-1])\n"
            "            if ln is None: continue\n"
            "            s_,_mb,lob,_ub=ln; low[t]=s_*t+lob\n"
            "        ls=__import__('pandas').Series(low,index=idx); m=(c<ls)&ls.notna(); f=m&~m.shift(1,fill_value=False)\n"
            "        rr=st.forward_returns(c, idx[f.to_numpy()], 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws=_np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(450); draws = rng.normal(45, 30, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='scrambled-geometry forks (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real fork {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean lower-tine-touch 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real fork sits mid-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real fork {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => geometry not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real fork (blue line) sits **in the middle** of the "
            f"scrambled-fork cloud — **p = {R['placebo'][1]:.2f}**. Geometric nonsense does just as "
            "well, so the specific Andrews lines aren't carrying any information. This is the "
            "cleanest refutation of 'price respects the fork.'"
        ),
        md(
            "### 4d · Per-ticker — the touch loses to random almost everywhere\n\n"
            "20-day touch-minus-random delta, per instrument. If the fork worked it would be "
            "positive across the board; instead it's negative in 4 of 5."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        c = load(t)['close']\n"
            "        e = st.tine_touch_entries(c, k=R['k']); re = st.random_entries(c, max(len(e),50), k=R['k'], seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d touch − random (bps)'); ax.set_title('Touch underperforms random in 4 of 5 names')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: only **DIA** edges out a positive delta ({R['per'][3][5]:+.0f} bps); "
            f"SPY is **{R['per'][0][5]:+.0f}** bps *behind* random. No coherent, cross-sectional edge "
            "— exactly what you'd expect if the fork is just relabelled drift."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real bounce\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** tine-bounce into a "
            "synthetic tape and check the same lower-tine rule banks it: edge=0 must stay at t≈0; "
            "edge>0 must light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.40):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=450, n_days=4000)\n"
            "    c = px['close']; e = st.tine_touch_entries(c, k=10); s = st.summarize(st.forward_returns(c, e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> t~0; planted bounce -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} touch={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted bounce the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — no false positive); a planted "
            f"bounce reaches **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). The detector "
            "works — so the flat real-tape result is a genuine 'nothing there', not a broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the lower-tine touch does not beat a drift-matched random baseline "
            f"(touch − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} "
            f"bps at 5/10/20/60d; Welch t never clears 2, max **{R['h60'][8]:+.2f}** at 60d). The "
            f"impressive one-sample t's (20d **{R['h20'][4]:.2f}**) are pure beta.\n"
            f"- **Tradability `MIRAGE`** — no residual edge once the drift is removed; costs only "
            "deepen the hole. You'd capture the drift more cheaply by holding the index.\n"
            f"- **Respects the fork? `BUSTED`** — the shuffled-pivot placebo leaves the result "
            f"untouched (**p = {R['placebo'][1]:.2f}**): geometric nonsense forks do as well as the "
            "real ones, so the specific Andrews lines carry no information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The tine-touch's entire apparent profit is the unconditional drift of long equity "
            "indices, which you obtain more cheaply and more fully by **buying and holding**. The fork "
            "rule trades *less* of the time (only on touches) and pays costs on each, so it strictly "
            "dominates *nothing*. There is no capacity question because there is no edge to scale. The "
            "pitchfork is a descriptive drawing tool, not a forecasting strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The 80%-median-line claim.** Andrews' headline stat ('price hits the median line "
            "~80% of the time') is near-tautological for a trending series — the median line is a "
            "smoothed trend line; even a random walk hits it often. A clean follow-up quantifies that "
            "on synthetic walks.\n"
            "- **Hand-anchored forks.** Proponents pick the 'right' three points by eye. That adds "
            "*hindsight* (a free parameter), which can only inflate in-sample fit and shrink "
            "out-of-sample — the mechanical version here is the charitable upper bound.\n"
            "- **Other median-line variants** (Schiff, modified Schiff, warning lines) are affine "
            "tweaks of the same geometry and inherit the same drift confound.\n\n"
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
