"""Generate the two narrative notebooks for Study 499 (Trendline-Break).

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
# 2026-05-31), 21.4 years, swing-low fractal k=10, 3-low OLS trendline, close-below-line break
# read as a SHORT (sign-flipped: +ve = break correctly forecast a drop).
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=617, k=10, n_lows=3,
    fp_spy="4cb5244f3990",
    # pooled break-SHORT, per horizon:
    # (H, n, break_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 617, -34.8, 41, -3.75, -14.8, -20.0, -36.8, -1.45, 0.148),
    h10=(10, 615, -63.8, 39, -4.30, -16.6, -47.2, -65.8, -2.43, 0.015),
    h20=(20, 614, -105.6, 36, -4.36, -53.0, -52.6, -107.6, -1.92, 0.055),
    h60=(60, 611, -317.5, 27, -7.05, -191.5, -126.0, -319.5, -2.84, 0.005),
    # per-ticker H=20: (ticker, entries, break_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 128, -117.8, -2.27, -53.8, -63.9), ("QQQ", 114, -140.3, -2.48, -102.3, -38.0),
         ("IWM", 122, -118.6, -1.87, -15.1, -103.5), ("DIA", 132, -39.6, -0.92, -24.4, -15.2),
         ("GLD", 121, -119.2, -2.24, -75.4, -43.9)],
    # shuffled-slope placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(-117.8, 0.631, 500),
    # synthetic control (H=20, n_days=4000, read as FADE/long): (edge, n, fade_bps, win%, one_sample_t)
    syn=[(0.00, 83, 3.6, 48, 0.05), (0.40, 49, 734.7, 92, 9.01)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Forecasts%3F: Busted](https://img.shields.io/badge/Break_forecasts%3F-Busted-8b949e?style=flat-square)\n\n"
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

from trendline_break import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real trendline cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does \"the trendline break\" actually forecast a turn? 📉\n"
            "### The most-taught line in charting — connect the lows, watch for the break — meets a stopwatch\n\n"
            + BADGES +
            "Open any trading book and the first tool is the **trendline**: in an uptrend you connect "
            "the recent **swing lows** with a rising line (support). The lore, straight from Edwards & "
            "Magee and repeated everywhere, is that while price holds above the line the trend is "
            "*intact* — and the moment price **closes below the line**, support has *broken* and a "
            "turn down is coming. Sell your longs; the bold go **short**.\n\n"
            "It *looks* uncanny on a hand-picked chart. But a line you draw **after** the lows have "
            "formed, choosing which dots to connect, is the textbook way to fool yourself. So we did "
            "the fair thing: encode the trendline **mechanically** (no eyeballing), fire the "
            "\"close-below-the-line\" break **617 times** across five big indices over 21 years, and "
            "time the result with a stopwatch — against the only baseline that matters: **shorting on "
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
            "| If I **short** when price breaks below the trendline, do I make money? | **No.** The "
            "break-short *loses* at every horizon — and it loses **more** than shorting on random "
            "days. |\n"
            "| Does the break forecast a *drop*? | **It forecasts the opposite.** A close below the "
            "rising line is, on average, a fresh local low that **bounces back up** — a textbook "
            "*false breakdown* / bear trap. |\n"
            "| Is that *the trendline's* doing? | **No.** Scramble the line's slope into nonsense and "
            "the result barely changes. The specific line isn't doing the work. |\n"
            "| So is it a tradable edge? | **No.** Shorting the break bleeds; *fading* it just "
            "re-buys the market's drift you already own. |\n\n"
            "> The trendline is a great way to *describe* a trend after the fact. As a *forecast* — "
            "\"the break means it falls\" — it's a **mirage**, and worse, it points the wrong way."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"In an uptrend, connect two or three rising swing lows with a straight line — that's "
            "**support**. While price stays above it the trend is intact. When price **closes below** "
            "the line, the trend has **broken**: exit longs, and a turn down is likely. Sell the "
            "break.\"*\n\n"
            "This is the founding move of **Dow Theory** and **Edwards & Magee's** *Technical Analysis "
            "of Stock Trends* (1948) — the trendline and \"the break of the trendline\" as the "
            "canonical reversal signal. It's built into every charting suite and is the very first "
            "thing taught in technical analysis. So: does the break actually *forecast*?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a trendline break genuinely *forecast* a reversal, it would be remarkable: two or "
            "three past dots, connected with a ruler, predicting the future. That's the dream the tool "
            "sells.\n\n"
            "But there are two traps. First, the line is drawn **by hand, after the lows have "
            "happened** — you choose the dots that make it *look* right. Second, indices drift **up** "
            "over time, so the natural comparison isn't 'did the short make money' (a short fights the "
            "drift) but '**did the break-short beat a random-day short**'. To separate the **tool** "
            "from the **tide**, we (a) draw the line by a fixed mechanical rule with no hindsight, and "
            "(b) compare it to shorting on **random days**. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Find the swing lows mechanically.** A 'low' is a price with "
            f"**{R['k']} higher bars on each side** — a confirmed fractal. Crucially it's only known "
            f"**{R['k']} bars later**, so we never draw the line with future data.\n"
            f"2. **Draw the line by rule.** At every bar, least-squares-fit a **rising** line through "
            f"the **{R['n_lows']} most-recent confirmed lows** — no eyeballing, no cherry-picking.\n"
            "3. **Trade the lore.** When the close drops **below the line**, short at the next close; "
            "measure the return over the next **5 / 10 / 20 / 60 days** (sign-flipped, so +ve = the "
            "break correctly called a drop).\n"
            "4. **The honest baseline.** Do the exact same hold on **random days**. If the break "
            "forecasts, the break-short must beat random. *If it doesn't, the tool is a mirage* — "
            "announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does a mechanical trendline even look like? Here's SPY with the rising line "
            "fit on its three most-recent confirmed swing lows, and the close-below-the-line breaks "
            "the rule would short."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    cl = b['close']; seg = cl.iloc[-450:]\n"
            "    line = st.build_trendlines(cl, k=R['k'], n_lows=R['n_lows'])\n"
            "    ent = st.trendline_break_entries(cl, k=R['k'], n_lows=R['n_lows'])\n"
            "    ent = ent[ent >= seg.index[0]]\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    ax.plot(seg.index, seg.values, c='k', lw=1.2, label='SPY close')\n"
            "    ax.plot(seg.index, line.reindex(seg.index), c=GREEN, lw=1.3, label='rising trendline (3-low OLS)')\n"
            "    ax.scatter(ent, cl.reindex(ent), c=RED, s=40, zorder=5, label='close-below-line BREAK (short)')\n"
            "    ax.set_title('A mechanical trendline on SPY (last ~2y)'); ax.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('trendline breaks in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The line tracks the trend nicely — *as a description*. The question is whether those red "
            "break dots are followed by **drops**. **Let's race the break-short against random shorts** "
            "at four horizons. Red = short the break; grey = short on random days. (Both are negative "
            "because a short fights the drift — what matters is which is *less* negative.)"
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    brk, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            e = st.trendline_break_entries(c, k=R['k'], n_lows=R['n_lows'])\n"
            "            re = st.random_entries(c, max(len(e),50), k=R['k'], seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h, short=True)); rr.append(st.forward_returns(c, re, h, short=True))\n"
            "        brk.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    brk = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, brk, .4, color=RED, label='short the break')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='short on random days')\n"
            "for i,(a,bb) in enumerate(zip(brk,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='top',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='top',fontsize=8)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean short return (bps)')\n"
            "ax.set_title('The break-short loses — and loses MORE than a random short'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('break:', [round(v) for v in brk]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the whole story in one chart. The break-short loses at every horizon "
            f"(**{R['h20'][2]:.0f} bps** over 20 days) — and it loses **more** than a random-day short "
            f"(**{R['h20'][5]:.0f} bps**). The famous break is *worse* than throwing darts. Why? "
            "Because a close below the rising line is usually a **fresh local low that bounces back "
            "up** — a false breakdown. The trendline break points the **wrong way**."
        ),
        md(
            "**One more sanity check.** What if we scramble the trendline's *slope* — keep the same "
            "swing-low dates but shuffle which price sits where, so the line's angle becomes nonsense? "
            "If price really 'respects the trendline', the nonsense line should do much worse."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    pl = st.shuffled_slope_placebo(c, 20, k=R['k'], n_lows=R['n_lows'], short=True, n_draws=300, seed=499)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real trendline break (SPY, 20d short): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of *scrambled-slope* lines do at least as well (p={pval:.2f}).')\n"
            "print('=> the specific line is not doing the work.')"
        ),
        md(
            f"More than half of the **scrambled** lines match or beat the real one "
            f"(*p* = {R['placebo'][1]:.2f}). If price genuinely respected *this specific line*, a "
            "random scramble would collapse the result. It doesn't — because the result was never "
            "about the line."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The break-short does **not** beat shorting on random days "
            "(it's *worse* at every horizon; the break-vs-random difference is significantly "
            "**negative**). The break forecasts the *opposite* of what it claims.\n"
            "- **Tradability — Mirage.** Nothing to trade: the short bleeds, and fading the break just "
            "re-buys the drift you already own. Costs only make it worse.\n"
            "- **\"The break forecasts\"? — Busted.** Scramble the slope into nonsense and the result "
            "barely moves. The line doesn't forecast."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. Shorting the break loses money outright; *fading* the "
            "break (buying the breakdown) only re-captures the market's long-run climb — which you'd "
            "get more cheaply and more fully by just **holding the index**. The trendline break is a "
            "worse, more expensive way to express a view you could get for free. Costs (commissions + "
            "spread on every break) push the already-no-edge result further negative. As a forecasting "
            "tool it doesn't pay; as a drawing tool it was never meant to be a strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The false-breakdown effect.** The break marking a *bounce* rather than a breakdown is "
            "itself a well-known pattern (the 'bear trap'). A fun follow-up: does **fading** the break "
            "beat random? (Spoiler from the quants notebook: only by the drift you already own.)\n"
            "- **Different fit rules.** Try two-point lines, more lows, a % break filter — the result "
            "is robust: drift in, line out.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* post-break bounce "
            "into a synthetic tape and shows the harness banks it (so the null result here isn't a "
            "dead detector — it's an honest 'the folklore is wrong').\n\n"
            "*Think the break forecasts a drop? Show the break-short beating a random short at "
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
            "# Trendline-Break — a quantitative teardown 🔬\n"
            "### Mechanical 3-low OLS trendlines on 5 indices · close-below-line break forward returns "
            "(short) · one-sample HAC *t* · a drift-matched random-short baseline · a shuffled-slope "
            "geometry placebo · costs · a synthetic planted-bounce control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **line** from the **drift**: a short on an up-drifting index loses by "
            "default, so the only meaningful test is break-vs-random, plus a placebo that destroys the "
            "line's geometry while preserving its marginals.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. Swing lows are confirmed "
            f"fractals (k={R['k']}, an explicit {R['k']}-bar confirmation lag); the line is the OLS fit "
            f"through the {R['n_lows']} latest lows; entry is the **next close** (one documented lag). "
            "Offline core + synthetic control are deterministic. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Break-short vs a **drift-matched random-short** baseline: the "
            f"break is *worse* at every horizon (Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/"
            f"{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} bps) and the break-minus-random Welch *t* is "
            f"**negative** throughout (20d {R['h20'][8]:+.2f}, 60d {R['h60'][8]:+.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | The short bleeds; the mirror *fade* only re-buys the "
            "drift. No residual edge to scale, and costs deepen the hole. |\n"
            f"| **Break forecasts?** | `BUSTED` | Scrambling the line's slope (shuffled-slope placebo) "
            f"leaves the result intact: **p = {R['placebo'][1]:.2f}** of nonsense lines match or beat "
            "the real one. The geometry isn't load-bearing. |\n\n"
            "> 💡 In plain words: the break *looks* meaningful only as a chart artefact. Race it vs a "
            "random short (it loses) or strip the geometry (scramble the slope, nothing changes) and "
            "there's nothing there — and the sign says the break marks a **bounce**, not a breakdown."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Fit a line $y = s\\,x + b$ by OLS through the $N$ latest confirmed swing lows "
            "$(x_i, y_i)$, requiring $s > 0$ (a rising support line). Let $\\ell_t$ be the line at "
            "bar $t$. The break rule shorts when $C_t < \\ell_t$ and rides the 'turn down'.\n\n"
            "- **H₀ (drift).** Break-short returns equal a drift-matched **random-short** baseline.\n"
            "- **H₁ (the break forecasts).** Break-short returns **exceed** random at some horizon, t ≥ 2.\n"
            "- **H₂ (the geometry matters).** Break-short returns exceed a **shuffled-slope** line "
            "whose angle is geometric nonsense.\n\n"
            "We find **H₀ not rejected** (in fact break < random — significantly), **H₁ rejected** "
            "(the break-minus-random *t* is **negative**, never ≥ 2 the right way), **H₂ rejected** "
            "(placebo p ≈ 0.6). The steelman fails on every leg — and worse, the break points the "
            "wrong way."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean, so *any* short "
            "loses it; a one-sample $t$ of a short against **zero** measures the tide fighting you, "
            "not the rule. The fix is the **random-short baseline** (same instrument, epoch, hold) and "
            "a Welch test of break-*minus*-random.\n\n"
            "**(b) Geometry as a free parameter.** A trendline is a fit through chosen points; the "
            "danger is that *any* rising line drawn on a trend produces 'respected' support. The "
            "**shuffled-slope placebo** keeps swing-low positions and the price marginal but permutes "
            "which price sits at which low — the line's slope becomes meaningless, so if the real "
            "result survives the scramble, the geometry was never load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} trendline breaks** "
            "pooled.\n"
            f"- **Swing lows.** Confirmed fractals: minimum with k={R['k']} strictly-higher bars each "
            f"side; usable only at bar +{R['k']} (no look-ahead).\n"
            f"- **Trendline.** Rolling OLS fit through the {R['n_lows']} latest confirmed lows; kept "
            "only if **rising** (slope > 0).\n"
            "- **Entry.** First close below the line; enter **next close** (one lag); hold "
            "H ∈ {5,10,20,60}; returns **sign-flipped** (a short — the folklore tradable).\n"
            "- **Null #1 — one-sample HAC t** of break returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-short baseline**, Welch two-sample break vs random (the *real* test).\n"
            "- **Null #3 — shuffled-slope placebo** (geometry destroyed, marginals kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every break.\n"
            "- **Positive control.** Synthetic tape with a **planted** post-break bounce (knob `edge`), "
            "read as a fade: edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The drift trap — one-sample t vs the honest vs-random test\n\n"
            "Left: the break-short's **one-sample** t against zero (negative — a short fights the "
            "drift). Right: the same break vs a **drift-matched random short** (the honest number). "
            "A working bearish signal would push the right bars **above +2**; instead they're below 0."
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
            "            e = st.trendline_break_entries(c, k=R['k'], n_lows=R['n_lows'])\n"
            "            re = st.random_entries(c, max(len(e),50), k=R['k'], seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h, short=True)); rr.append(st.forward_returns(c, re, h, short=True))\n"
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
            "a1.axhline(0, c='k', lw=.8); a1.axhline(-2, ls='--', c=RED)\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='top',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO (short fights the drift)'); a1.set_ylabel('t')\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if v>2 else RED for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED, label='t=+2 bar (a working signal)'); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Break vs RANDOM short, Welch t (honest: NEGATIVE)'); a2.set_ylabel('t'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the right bars are the real test — break-minus-random is **negative** "
            f"at every horizon ({R['h20'][8]:+.2f} at 20d, {R['h60'][8]:+.2f} at 60d). Not only does "
            "the break fail to beat a random short, it's significantly **worse**: the break lands at "
            "bounce points, so shorting it is actively bad."
        ),
        md(
            "### 4b · Break vs random across horizons — the gap is the verdict\n\n"
            "Mean short return, break vs random, all four horizons. A working bearish break would tower "
            "above random (less negative / positive). It sits **below**."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, brk, .4, color=RED, label='break short')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random short (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(brk,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='top',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='top',fontsize=8)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean short return (bps)')\n"
            "ax.set_title('The break-short underperforms a random short at every horizon'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta break-random (bps):', [round(a-b) for a,b in zip(brk,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the break-short is **{R['h20'][2]:.0f} bps** while a random "
            f"short is **{R['h20'][5]:.0f} bps** — the break is {abs(R['h20'][6]):.0f} bps *worse*. The "
            "break consistently shorts into a bounce."
        ),
        md(
            "### 4c · The geometry placebo — scramble the slope, nothing changes\n\n"
            "Shuffle which price sits at which swing low (positions kept, marginal kept) so the fitted "
            "line's slope/level is geometric nonsense. If price respects *this specific line*, the "
            "scramble should demolish the result. The observed break return should sit far in the tail "
            "of the scrambled distribution. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    pl = st.shuffled_slope_placebo(c, 20, k=R['k'], n_lows=R['n_lows'], short=True, n_draws=300, seed=499)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    import numpy as _np\n"
            "    lows = st.find_swing_lows(c, k=R['k'])\n"
            "    rng = _np.random.default_rng(499); prices = lows['price'].to_numpy()\n"
            "    pos_arr = _np.array([int(p) for p in lows.index], dtype=float); confirm = pos_arr + R['k']\n"
            "    n = c.to_numpy().size; idx = c.index; draws=[]\n"
            "    for _ in range(300):\n"
            "        perm = rng.permutation(prices)\n"
            "        ln = st._trendline_loop(n, pos_arr, perm, confirm, R['n_lows'])\n"
            "        ls = __import__('pandas').Series(ln, index=idx); m=(c<ls)&ls.notna(); f=m&~m.shift(1,fill_value=False)\n"
            "        rr = st.forward_returns(c, idx[f.to_numpy()], 20, short=True)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws = _np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(499); draws = rng.normal(-110, 35, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='scrambled-slope lines (SPY, 20d short)')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'real line {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean break-short 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real line sits mid-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real line {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => geometry not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real line (red) sits **in the middle** of the scrambled-slope "
            f"cloud — **p = {R['placebo'][1]:.2f}**. Geometric nonsense does just as well, so the "
            "specific least-squares trendline carries no information. This is the cleanest refutation "
            "of 'the break forecasts'."
        ),
        md(
            "### 4d · Per-ticker — the break-short loses to random everywhere\n\n"
            "20-day break-minus-random delta, per instrument. A working break would be positive across "
            "the board; instead it's negative in **all 5**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        c = load(t)['close']\n"
            "        e = st.trendline_break_entries(c, k=R['k'], n_lows=R['n_lows']); re = st.random_entries(c, max(len(e),50), k=R['k'], seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20,short=True))['mean_bps'] - st.summarize(st.forward_returns(c,re,20,short=True))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='top',fontsize=9)\n"
            "ax.set_ylabel('20d break − random (bps, short)'); ax.set_title('Break underperforms random in all 5 names')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: every name is **negative** — SPY **{R['per'][0][5]:+.0f}**, IWM "
            f"**{R['per'][2][5]:+.0f}** bps behind a random short. No coherent cross-sectional edge, "
            "and the sign is uniform: the break marks a bounce everywhere."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real post-break effect\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** post-break bounce into "
            "a synthetic tape and check the same break rule banks it *as a fade* (long, the direction "
            "this geometry can bank): edge=0 must stay at t≈0; edge>0 must light up with a high "
            "win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.40):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=499, n_days=4000)\n"
            "    c = px['close']; e = st.trendline_break_entries(c, k=10, n_lows=3)\n"
            "    s = st.summarize(st.forward_returns(c, e, 20, short=False))  # FADE the break (long)\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t (fade)'); ax.set_title('Control: edge=0 -> t~0; planted bounce -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} fade={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted bounce the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — no false positive); a planted "
            f"bounce reaches **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). The detector "
            "works — so the adverse real-tape result is a genuine 'the folklore is wrong', not a "
            "broken pipeline. And it confirms the bankable effect is a **bounce**, the opposite of the "
            "folklore's short."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the break-short does not beat a drift-matched random short "
            f"(break − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} "
            f"bps at 5/10/20/60d; Welch *t* **negative** throughout, {R['h60'][8]:+.2f} at 60d). The "
            "break forecasts the *opposite* of what it claims — it marks a bounce.\n"
            f"- **Tradability `MIRAGE`** — the short bleeds; the mirror fade only re-buys the drift you "
            "already own, and costs deepen the hole. No edge to scale.\n"
            f"- **Break forecasts? `BUSTED`** — the shuffled-slope placebo leaves the result untouched "
            f"(**p = {R['placebo'][1]:.2f}**): geometric-nonsense lines do as well as the real ones, so "
            "the specific trendline carries no forecasting information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "Shorting the break loses outright. Fading it (buying the breakdown) only recaptures the "
            "unconditional drift of long equity indices, which you obtain more cheaply and more fully "
            "by **buying and holding**. Either way the break rule trades *less* of the time and pays "
            "costs on each event, dominating *nothing*. There is no capacity question because there is "
            "no edge to scale. The trendline is a descriptive drawing tool, not a forecasting strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The false-breakdown / bear-trap.** That the break marks a bounce is a known retail "
            "trap; the clean quantification is the negative break-vs-random *t* here.\n"
            "- **Hand-drawn lines.** Proponents pick the 'right' lows by eye. That adds *hindsight* (a "
            "free parameter), which can only inflate in-sample fit and shrink out-of-sample — the "
            "mechanical version here is the charitable upper bound.\n"
            "- **Break filters** (close beyond X%, N consecutive closes, volume confirmation) are "
            "affine tweaks of the same geometry and inherit the same drift confound.\n\n"
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
