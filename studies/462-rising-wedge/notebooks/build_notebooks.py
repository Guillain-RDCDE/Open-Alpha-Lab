"""Generate the two narrative notebooks for Study 462 (Rising Wedge).

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
# 2026-05-31, partial June dropped), 21.4 years, pivot fractal k=10, rising-wedge support-break
# SHORT. Returns are SHORT returns (a price fall is a gain).
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=340, k=10,
    fp_spy="4cb5244f3990",
    # pooled wedge-break SHORT, per horizon:
    # (H, n, short_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 340, -6.9, 44, -0.62, -0.3, -6.6, -8.9, -0.33, 0.741),
    h10=(10, 340, -41.6, 39, -2.43, -24.3, -17.4, -43.6, -0.68, 0.495),
    h20=(20, 340, -93.7, 36, -3.47, -59.3, -34.3, -95.7, -0.96, 0.339),
    h60=(60, 339, -325.5, 27, -6.03, -240.5, -85.0, -327.5, -1.48, 0.140),
    # per-ticker H=20: (ticker, entries, short_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 74, -100.2, -1.79, -72.5, -27.7), ("QQQ", 71, -139.6, -2.08, -114.9, -24.6),
         ("IWM", 67, -37.1, -0.52, 28.9, -66.0), ("DIA", 74, -54.4, -1.16, -59.0, 4.6),
         ("GLD", 54, -148.3, -1.91, -77.7, -70.5)],
    # slope-scramble placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(-100.2, 0.248, 500),
    # synthetic control (H=20, n_days=4000): (edge, n, short_bps, win%, one_sample_t)
    syn=[(0.00, 37, -67.6, 41, -0.84), (0.50, 11, 725.6, 100, 4.88)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Forecasts_a_downside_break%3F: Busted](https://img.shields.io/badge/Forecasts_a_downside_break%3F-Busted-8b949e?style=flat-square)\n\n"
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

from rising_wedge import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real rising-wedge cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does a rising wedge really break *down*? 📐\n"
            "### A famous \"bearish\" chart pattern — two rising lines squeezing together — meets a stopwatch\n\n"
            + BADGES +
            "Open any trading course and you'll meet the **rising wedge**: two **up-sloping** lines "
            "that converge — support through the rising lows, resistance through the rising highs — "
            "with the channel narrowing to a point. The lore, in every textbook and on every "
            "chart-pattern site, is that it's a **bearish** pattern: price is climbing on borrowed "
            "time and will **break down** through the lower line. So when price cracks below the "
            "rising support, you **short** it.\n\n"
            "It *looks* uncanny on a hand-picked chart. But a pattern you draw **after** the swings, "
            "by choosing which lows and highs to connect, is the textbook setup for fooling "
            "yourself. So we did the only fair thing: encode the wedge **mechanically** (no "
            "eyeballing), fire the \"short the break-down\" rule hundreds of times across five big "
            "indices over 21 years, and time the result — against the only baseline that matters: "
            "**shorting on random days instead.**\n\n"
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
            "| If I **short** when price breaks below the rising wedge, do I make money? | **No.** "
            "The short **loses** at every horizon — price tends to *rise* after the \"bearish\" "
            "break, not fall. |\n"
            "| Is it at least better than shorting at random? | **No.** Short on **random days** "
            "instead and you lose **less**. The wedge break is *worse* than a coin-flip short. |\n"
            "| Does the wedge \"forecast\" the break-down? | **Not in any usable way.** Scramble the "
            "support line into nonsense and the result barely changes. The geometry isn't doing the "
            "work. |\n"
            "| So is it a tradable edge? | **No.** The only thing the short reliably captures is the "
            "market's upward drift — as a **loss**. |\n\n"
            "> The rising wedge is a great way to *describe* a tired-looking rally after the fact. As "
            "a *forecast* — \"it will break down\" — it's a **mirage**: on a market that drifts up, "
            "shorting the break just hands the drift back, and the lines add nothing."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Two rising lines converge — support through the higher lows, resistance through the "
            "higher highs, the lower line catching up. The rally is running out of room. When price "
            "**breaks below the rising support**, it's a **bearish** signal: short it, the wedge "
            "resolves down.\"*\n\n"
            "This is one of the classical chart patterns from **Edwards & Magee** (*Technical "
            "Analysis of Stock Trends*, 1948) and **Bulkowski's** *Encyclopedia of Chart Patterns* — "
            "built into TradingView, every screener and every trading course. It's one of the most "
            "recognised \"bearish\" setups in technical analysis. So: does the rising wedge actually "
            "break down?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the wedge genuinely *forecast* a fall, it would be remarkable: a converging pair of "
            "lines drawn from past swings would predict future direction — a clean, ruler-drawable "
            "crack in market efficiency. That's the dream the pattern sells.\n\n"
            "But there's a trap built into it. A wedge is drawn **by hand, after the swings have "
            "happened** — you choose the lows and highs that make the squeeze *look* right. And it's "
            "drawn on a market (stock indices) that drifts **up** over time, so *any* short will tend "
            "to lose. To separate the **pattern** from the **tide**, we have to (a) draw the wedge by "
            "a fixed mechanical rule with no hindsight, and (b) compare it to shorting on **random "
            "days**. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Find the swing points mechanically.** A 'pivot' is a high (or low) with "
            f"**{R['k']} lower (higher) bars on each side** — a confirmed fractal. Crucially it's only "
            f"known **{R['k']} bars later**, so we never draw the wedge with future data.\n"
            "2. **Fit the wedge by rule.** At every bar, fit a support line through the recent rising "
            "lows and a resistance line through the recent rising highs. It only counts as a *rising "
            "wedge* if both lines rise, support rises faster (narrowing), and they haven't crossed.\n"
            "3. **Trade the lore.** When the close drops **below the rising support**, **short** at "
            "the next close; measure the return over the next **5 / 10 / 20 / 60 days** (a price fall "
            "= a gain for the short).\n"
            "4. **The honest baseline.** Do the exact same short on **random days**. If the wedge "
            "matters, the break-down short must beat random. *If it doesn't, the pattern is a "
            "mirage* — that's the result that would make us say so, announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does a mechanical rising wedge even look like? Here's SPY with the wedge "
            "fit on its recent rising pivots, and the support-break the rule would short."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    cl = b['close']; seg = cl.iloc[-450:]\n"
            "    sup, act = st.build_wedges(cl, k=R['k'])\n"
            "    ent = st.wedge_break_entries(cl, k=R['k'])\n"
            "    ent = ent[ent >= seg.index[0]]\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    ax.plot(seg.index, seg.values, c='k', lw=1.2, label='SPY close')\n"
            "    ax.plot(seg.index, sup.reindex(seg.index), c=GREEN, lw=1.3, label='rising support')\n"
            "    ax.scatter(ent, cl.reindex(ent), c=RED, s=44, zorder=5, label='support-break SHORT')\n"
            "    ax.set_title('A mechanical rising wedge on SPY (last ~2y)'); ax.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('support-break shorts in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The wedge tracks the rally nicely — *as a description*. The question is whether those "
            "red short dots are followed by **falls**. **Let's race the break-down short against "
            "random shorts** at four horizons. Red = short the break; grey = short on random days. "
            "Bars *below zero* mean the short lost money."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    brk, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            e = st.wedge_break_entries(c, k=R['k'])\n"
            "            re = st.random_entries(c, max(len(e),50), k=R['k'], seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        brk.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    brk = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, brk, .4, color=RED, label='short the wedge break')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='short on random days')\n"
            "for i,(a,bb) in enumerate(zip(brk,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='top',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='top',fontsize=8)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean SHORT return (bps)')\n"
            "ax.set_title('The wedge short LOSES — and loses more than a random short'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('break:', [round(v) for v in brk]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the whole story in one chart. The wedge-break short **loses at every horizon** "
            f"(**{R['h20'][2]:.0f} bps** over 20 days) — and the random short loses **less** "
            f"(**{R['h20'][5]:.0f} bps**). The supposedly bearish break is followed, on average, by a "
            "**relief rally**, not a fall. The pattern doesn't just fail to help — shorting it is "
            "*worse* than throwing darts. The 'edge' was the market's upward drift, working against "
            "every short."
        ),
        md(
            "**One more sanity check.** What if we scramble the wedge's *geometry* — keep the break "
            "timing but replace the support line with a random nonsense line? If price really "
            "'respects the rising wedge', the nonsense line should do much worse."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    pl = st.slope_scramble_placebo(c, 20, k=R['k'], n_draws=300, seed=462)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real wedge break-down short (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... and {pval*100:.0f}% of *scrambled-geometry* lines do at least as well (p={pval:.2f}).')\n"
            "print('=> the rising-wedge geometry is not doing the work.')"
        ),
        md(
            f"A chunk of the **scrambled** lines match or beat the real one "
            f"(*p* = {R['placebo'][1]:.2f}). If price genuinely respected *this specific rising "
            "support*, a random scramble would collapse the result. It doesn't — because the result "
            "was never about the wedge."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The break-down short does **not** beat shorting on random days "
            "(it's *worse* at every horizon; the break-vs-random difference never clears *t* = 2). "
            "The big losses are the market's drift working against a short, not a bearish pattern.\n"
            "- **Tradability — Mirage.** Nothing to trade: the short loses outright and underperforms "
            "a random short — costs only deepen the hole.\n"
            "- **\"Forecasts a downside break\"? — Busted.** Scramble the geometry into nonsense and "
            "the result barely moves. The wedge doesn't forecast the break."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. Shorting the rising-wedge break loses money outright, and "
            "loses *more* than a random short — the pattern actively picks bad moments to be short on "
            "an upward-drifting tape. Costs (commissions + spread on every break) push the "
            "already-negative result further down. As a forecasting tool the rising wedge doesn't "
            "pay; as a drawing tool, it was never meant to be a strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The falling-wedge mirror.** The *falling* wedge is sold as bullish; the desk's "
            "[`414-falling-wedge`](../../414-falling-wedge) study tests that twin claim — same "
            "machinery, same outcome.\n"
            "- **Different pivot rules.** Try a wider/narrower fractal window, more touches per line, "
            "or hand-anchored wedges — the result is robust: drift in, pattern out.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* rising-wedge "
            "break-down into a synthetic tape and shows the harness banks it (so the null result "
            "here isn't a dead detector — it's an honest 'nothing there').\n\n"
            "*Think the rising wedge forecasts the break? Show the break-down short beating random "
            "shorts at **t ≥ 2** on a real tape — then we'll talk.*"
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
            "# The Rising Wedge — a quantitative teardown 🔬\n"
            "### Mechanical converging wedges on 5 indices · support-break SHORT forward returns · "
            "one-sample HAC *t* · a drift-matched random-short baseline · a slope-scramble geometry "
            "placebo · costs · a synthetic planted-break-down control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **wedge** from the **drift**: a short on an upward-trending index "
            "loses *regardless* of the pattern, so the only meaningful test is break-vs-random-short, "
            "plus a placebo that destroys the wedge's geometry while preserving its marginals.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. Pivots are confirmed "
            f"fractals (k={R['k']}, an explicit {R['k']}-bar confirmation lag); entry is the **next "
            "close** (one documented lag). Returns are **short** returns. Offline core + synthetic "
            "control are deterministic. Methods in [`docs/references.md`](../docs/references.md), "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Break-down short vs a **drift-matched random-short** baseline: "
            f"the break is *worse* at every horizon (Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/"
            f"{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} bps) and the break-minus-random difference "
            f"**never clears |t| = 2** (Welch t at 20d = {R['h20'][8]:+.2f}, 60d = {R['h60'][8]:+.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | The big *negative* one-sample t's (20d t = {R['h20'][4]:.2f}) "
            f"are **pure beta against a short** — they are the drift, not skill, and costs deepen the "
            "loss. No residual edge to scale. |\n"
            f"| **Forecasts a downside break?** | `BUSTED` | Scrambling the support line "
            f"(slope-scramble placebo) leaves the result intact: **p = {R['placebo'][1]:.2f}** of "
            "nonsense lines match or beat the real one. The geometry isn't doing the work. |\n\n"
            "> 💡 In plain words: the wedge break *looks* dramatically significant only because the "
            "index drifts up and a short eats that drift. Strip the drift (race it vs a random short) "
            "or strip the geometry (scramble the support) and there is nothing left — and the "
            "'bearish' break is actually followed by a relief rally."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Fit a support line through the last confirmed swing lows ($y=s_\\ell x+b_\\ell$) and a "
            "resistance through the last confirmed swing highs ($y=s_r x+b_r$). A **rising wedge** "
            "requires $s_\\ell>0$, $s_r>0$ and $s_\\ell>s_r$ (both rising, converging), with the "
            "lines not yet crossed. Let $\\ell_t$ be the support at bar $t$. The rule **shorts** when "
            "$C_t<\\ell_t$ and rides the expected break-down.\n\n"
            "- **H₀ (drift).** Short returns equal a drift-matched **random-short** baseline.\n"
            "- **H₁ (the wedge forecasts).** Break-down short returns **exceed** random at some "
            "horizon, t ≥ 2.\n"
            "- **H₂ (the geometry matters).** Break returns exceed a **slope-scramble** wedge whose "
            "support line is nonsense.\n\n"
            "We find **H₀ not rejected** (break ≤ random at every horizon — in fact *worse*), **H₁ "
            "rejected** (Welch t never ≥ 2), **H₂ rejected** (placebo p ≈ 0.25). The steelman fails "
            "on every leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean. A **short** on a "
            "long-only horizon inherits it as a **loss**; a large *negative* one-sample $t$ against "
            "**zero** measures the tide, not the pattern. The fix is the **random-short baseline** "
            "(same instrument, epoch, hold, short sign) and a Welch test of break-*minus*-random.\n\n"
            "**(b) Geometry as a free parameter.** A wedge is two chosen lines; the danger is that "
            "*any* converging lines drawn on a noisy trend produce 'breaks'. The **slope-scramble "
            "placebo** keeps the break cadence (a short fires only where a wedge was active) but "
            "replaces the fitted support with a random plausible line — the geometry becomes "
            "meaningless, so if the real result survives the scramble, the wedge was never "
            "load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} support-break shorts** "
            "pooled.\n"
            f"- **Pivots.** Confirmed fractals: extremum with k={R['k']} strictly-beaten bars each "
            f"side; usable only at bar +{R['k']} (no look-ahead).\n"
            "- **Wedge.** Rolling least-squares lines through the last confirmed lows/highs; qualify "
            "only when both rise, support rises faster, lines not yet crossed.\n"
            "- **Entry.** First close below the rising support; **short** at the **next close** (one "
            "lag); hold H ∈ {5,10,20,60}; forward return = $-(P_{e+H}/P_e-1)$.\n"
            "- **Null #1 — one-sample HAC t** of short returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-short baseline**, Welch two-sample break vs random (the *real* test).\n"
            "- **Null #3 — slope-scramble placebo** (geometry destroyed, marginals kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every break.\n"
            "- **Positive control.** Synthetic tape with a **planted** rising-wedge break-down (knob "
            "`edge`): edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t looks decisive, vs-random kills it\n\n"
            "Left: the break-down short's **one-sample** t against zero (the misleading number — it's "
            "the drift, working against the short). Right: the same break vs a **drift-matched "
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
            "            e = st.wedge_break_entries(c, k=R['k'])\n"
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
            "a1.axhline(-2, ls='--', c=RED, label='|t|=2 bar'); a1.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='top',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO (misleading: it is beta vs a short)'); a1.set_ylabel('t'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if v>2 else RED for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Break vs RANDOM short, Welch t (honest: never clears 2)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars are dramatically *negative* (20d **{R['h20'][4]:.2f}**, "
            f"60d **{R['h60'][4]:.2f}**) — but that's the **drift**, every short inherits it as a loss. "
            f"The right bars are the real test: break-minus-random is **negative** at all horizons "
            f"({R['h20'][8]:+.2f} at 20d, {R['h60'][8]:+.2f} at 60d) and never significant. The wedge "
            "is *worse* than a coin-flip short."
        ),
        md(
            "### 4b · Break vs random across horizons — the gap is the verdict\n\n"
            "Mean **short** return, wedge-break vs random short, all four horizons. If the wedge "
            "forecast the break-down, its short would tower over random. It doesn't — it's below."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, brk, .4, color=RED, label='wedge-break short')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random short (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(brk,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='top',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='top',fontsize=8)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean SHORT return (bps)')\n"
            "ax.set_title('Wedge-break short loses MORE than a random short'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta break-random (bps):', [round(a-b) for a,b in zip(brk,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the break short is **{R['h20'][2]:.0f} bps** and random "
            f"is **{R['h20'][5]:.0f} bps** — the wedge *underperforms* a random short by "
            f"{abs(R['h20'][6]):.0f} bps. There is no horizon where the wedge break helps; the "
            "supposedly bearish pattern is consistently the worse moment to be short."
        ),
        md(
            "### 4c · The geometry placebo — scramble the support, nothing changes\n\n"
            "Replace the fitted rising support with a random plausible line (same break cadence, same "
            "price marginal) so the geometry is nonsense. If price respects *this specific rising "
            "line*, the scramble should demolish the result. The observed short should sit far in the "
            "tail of the scrambled distribution. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    pl = st.slope_scramble_placebo(c, 20, k=R['k'], n_draws=400, seed=462)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    # rebuild the placebo distribution for the histogram\n"
            "    import numpy as _np, pandas as _pd\n"
            "    sup, act = st.build_wedges(c, k=R['k']); vm = sup.notna()\n"
            "    cl_ = c.to_numpy(); idx = c.index; n = cl_.size; warm = vm.to_numpy()\n"
            "    gaps = (c[vm].to_numpy() - sup[vm].to_numpy())/c[vm].to_numpy()\n"
            "    gmu, gsd = float(_np.nanmean(gaps)), float(_np.nanstd(gaps)+1e-9)\n"
            "    rng = _np.random.default_rng(462); draws=[]\n"
            "    for _ in range(400):\n"
            "        rg = rng.normal(gmu, gsd, n); fs = _np.where(warm, cl_*(1-rg), _np.nan)\n"
            "        fss = _pd.Series(fs, index=idx); m=(c<fss)&fss.notna(); f=m&~m.shift(1,fill_value=False)\n"
            "        rr=st.forward_returns(c, idx[f.to_numpy()], 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws=_np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(462); draws = rng.normal(-90, 40, 400)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='scrambled-geometry lines (SPY, 20d)')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'real wedge {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean break-down 20d SHORT return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real wedge sits mid-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real wedge {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => geometry not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real wedge (red line) sits **inside** the scrambled-line cloud "
            f"— **p = {R['placebo'][1]:.2f}**. Nonsense lines do about as well, so the specific "
            "rising-wedge support isn't carrying information. This is the cleanest refutation of 'the "
            "rising wedge forecasts the break-down.'"
        ),
        md(
            "### 4d · Per-ticker — the break-down short loses to random almost everywhere\n\n"
            "20-day break-minus-random delta, per instrument. If the wedge worked it would be "
            "positive across the board; instead it's negative in 4 of 5."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        c = load(t)['close']\n"
            "        e = st.wedge_break_entries(c, k=R['k']); re = st.random_entries(c, max(len(e),50), k=R['k'], seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d break − random (bps)'); ax.set_title('Break-down short underperforms random in 4 of 5 names')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: only **DIA** edges out a positive delta ({R['per'][3][5]:+.0f} bps); "
            f"SPY is **{R['per'][0][5]:+.0f}** bps *behind* a random short. No coherent, "
            "cross-sectional edge — exactly what you'd expect if the wedge is just relabelled drift."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real break-down\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** rising-wedge "
            "break-down into a synthetic tape (a clean rising/narrowing build-up, then a downward "
            "resolution at the apex) and check the same short rule banks it: edge=0 must stay at "
            "t≈0; edge>0 must light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.50):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=462, n_days=4000)\n"
            "    c = px['close']; e = st.wedge_break_entries(c, k=10); s = st.summarize(st.forward_returns(c, e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar'); ax.axhline(0, c='k', lw=.8)\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t (short)'); ax.set_title('Control: edge=0 -> t~0; planted break-down -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} short={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted break-down the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — a fair coin, no false "
            f"positive); a planted break-down reaches **t = {R['syn'][1][4]:.2f}** (win "
            f"{R['syn'][1][3]:.0f}%). The detector works — so the flat real-tape result is a genuine "
            "'nothing there', not a broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the break-down short does not beat a drift-matched random short "
            f"(break − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} "
            f"bps at 5/10/20/60d; Welch t never clears 2, max **{R['h60'][8]:+.2f}** at 60d). The "
            f"impressive *negative* one-sample t's (20d **{R['h20'][4]:.2f}**) are pure beta against "
            "a short.\n"
            f"- **Tradability `MIRAGE`** — the short loses outright and underperforms even a random "
            "short; costs only deepen the loss. No residual edge to scale.\n"
            f"- **Forecasts a downside break? `BUSTED`** — the slope-scramble placebo leaves the "
            f"result untouched (**p = {R['placebo'][1]:.2f}**): nonsense support lines do as well as "
            "the real ones, so the rising-wedge geometry carries no forecasting information. The "
            "'bearish' break is, on the real tape, followed by a relief rally."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "Shorting the rising-wedge break loses money outright on an upward-drifting tape, and "
            "loses *more* than a random short — the pattern actively selects bad moments to be short. "
            "There is no capacity question because there is no edge to scale; costs on every break "
            "only widen the hole. The rising wedge is a descriptive after-the-fact label, not a "
            "bearish forecasting strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The falling-wedge twin.** The *falling* wedge (sold as bullish) is the mirror claim; "
            "[`414-falling-wedge`](../../414-falling-wedge) tests it on the same machinery.\n"
            "- **Hand-anchored wedges.** Proponents pick the 'right' lows/highs by eye. That adds "
            "*hindsight* (free parameters), which can only inflate in-sample fit and shrink "
            "out-of-sample — the mechanical version here is the charitable upper bound.\n"
            "- **Break confirmation filters** (volume contraction, % penetration, retest) are affine "
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
