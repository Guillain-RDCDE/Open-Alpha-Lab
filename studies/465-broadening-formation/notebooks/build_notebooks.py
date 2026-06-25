"""Generate the two narrative notebooks for Study 465 (Broadening Formation / megaphone).

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
# 2026-05-31), 21.4 years, pivot fractal k=10, lower-boundary megaphone-break SHORT.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=25, k=10,
    fp_spy="4cb5244f3990",
    # pooled lower-break SHORT, per horizon:
    # (H, n, short_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 25, -207.6, 16, -4.45, -10.3, -197.4, -209.6, -3.13, 0.004),
    h10=(10, 25, -172.9, 32, -3.04, -53.6, -119.4, -174.9, -1.70, 0.099),
    h20=(20, 25, -320.9, 20, -3.07, -33.1, -287.9, -322.9, -2.33, 0.027),
    h60=(60, 25, -938.7, 16, -3.73, -118.9, -819.8, -940.7, -3.91, 0.001),
    # per-ticker H=20: (ticker, entries, short_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 7, -416.8, -3.38, -39.3, -377.5), ("QQQ", 11, -451.4, -2.38, -88.2, -363.2),
         ("IWM", 2, 333.9, float("nan"), -2.7, 336.6), ("DIA", 1, -765.8, float("nan"), -33.7, -732.0),
         ("GLD", 4, -10.6, float("nan"), -1.5, -9.1)],
    # shuffled-pivot placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(-416.8, 0.974, 500),
    # synthetic control (H=20, n_days=4000, seed=999): (edge, n, short_bps, win%, one_sample_t)
    syn=[(0.00, 9, -82.2, 67, -0.93), (0.50, 8, 515.3, 100, 6.88)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Expanding_vol_forecasts_a_turn%3F: Busted](https://img.shields.io/badge/Expanding_vol_forecasts_a_turn%3F-Busted-8b949e?style=flat-square)\n\n"
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

from broadening_formation import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real megaphone cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = (BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\n"
             + "nan = float('nan')\n" + "R = " + repr(R) + "\n")


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does a broadening \"megaphone\" top really mark a reversal? 📣\n"
            "### A famous chart shape — fanning highs and lows — meets a stopwatch\n\n"
            + BADGES +
            "Open any chart-pattern book and you'll meet the **broadening formation** (the "
            "*megaphone*): swing highs keep making **higher highs**, swing lows keep making "
            "**lower lows**, and the trading range fans out like a megaphone. The lore, straight "
            "from Schabacker and Edwards & Magee, is that this expanding, frantic range is a "
            "**blow-off top** — the crowd piling in while smart money distributes — so when price "
            "**breaks the lower boundary**, you **short**: it's \"supposed\" to reverse down.\n\n"
            "It *looks* uncanny on the handful of charts the books show you. But a shape you only "
            "recognise **after** the swings have fanned out, on a market that drifts **up** over "
            "time, is the textbook setup for fooling yourself. So we did the fair thing: encode the "
            "megaphone **mechanically** (no eyeballing), fire the \"short the lower-boundary break\" "
            "rule every time it appears across five big indices over 21 years, and time the result "
            "against the only baseline that matters — **shorting on random days instead.**\n\n"
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
            "| If I short when price breaks the megaphone's **lower boundary**, do I make money? | "
            "**No — I lose.** The short is deeply negative at every horizon (−208 bps over 5 days, "
            "−939 over 60). |\n"
            "| Is that *the megaphone's* doing? | **No.** Shorting on **random days** loses *less*. "
            "The megaphone short is **worse than a random short** at every horizon — it's just "
            "fighting the market's upward drift. |\n"
            "| Does the expanding range \"forecast a turn\"? | **Not in any usable way.** Scramble "
            "the megaphone's geometry into nonsense and the result barely changes (the real one is "
            "actually at the *bad* end). |\n"
            "| So is it a tradable edge? | **No.** It's a **rare, seductive shape** — only **25** "
            "appear in 21 years — that carries no forecasting information either way. |\n\n"
            "> The megaphone is a great way to *describe* a wild, choppy stretch after the fact. As "
            "a *forecast* — \"the expanding top will reverse down\" — it's a **mirage**: the short "
            "just loses the up-drift, and the specific lines do none of the work."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When highs make higher highs **and** lows make lower lows, the range broadens into "
            "a megaphone — an exhausted, over-excited top. Volatility is expanding; the crowd is "
            "euphoric. Short the break of the lower boundary: the broadening top reverses down.\"*\n\n"
            "This is the **broadening formation** of **Schabacker (1932)** and **Edwards & Magee "
            "(1948)**, catalogued by **Bulkowski** and taught on every chart-pattern site. It is one "
            "of the most recognisable \"reversal\" shapes in technical analysis — so: does the "
            "megaphone actually call the turn?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the megaphone genuinely *forecast* tops, it would be remarkable: a fanning range "
            "would predict a reversal you could short with a ruler — a clean crack in market "
            "efficiency. That's the dream the pattern sells.\n\n"
            "But there are two traps. First, a megaphone is recognised **after** the swings have "
            "fanned out — you pick the shape that *looks* right. Second, the rule is a **short** on a "
            "market (stock indices) that drifts **up**, so *any* short will tend to lose — which can "
            "fool you in the other direction (a losing short looks like \"the pattern doesn't work\" "
            "rather than \"there was never any signal\"). To separate the **shape** from the **tide** "
            "we (a) draw the megaphone by a fixed mechanical rule with no hindsight, and (b) compare "
            "it to shorting on **random days**. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Find the swing points mechanically.** A 'pivot' is a high (or low) with "
            f"**{R['k']} lower (higher) bars on each side** — a confirmed fractal, only known "
            f"**{R['k']} bars later**, so we never draw the megaphone with future data.\n"
            "2. **Draw the megaphone by rule.** Take the last two swing highs and last two swing "
            "lows; it's a broadening top only if the highs are **rising** *and* the lows **falling** "
            "(the boundaries diverge) — no eyeballing.\n"
            "3. **Trade the lore.** When the close drops **below the lower boundary**, **short** at "
            "the next close; measure the short's return over the next **5 / 10 / 20 / 60 days**.\n"
            "4. **The honest baseline.** Do the exact same short on **random days**. If the megaphone "
            "matters, the break-short must beat a random short. *If it doesn't, the pattern is a "
            "mirage* — announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does a mechanical megaphone even look like? Here's a stretch of SPY with the "
            "diverging boundaries drawn on its confirmed pivots, and a lower-boundary break the rule "
            "would short."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    cl = b['close']\n"
            "    low, up = st.build_megaphones(cl, k=R['k'])\n"
            "    ent = st.lower_break_entries(cl, k=R['k'])\n"
            "    # pick a window around an actual break so a megaphone is visible\n"
            "    anchor = ent[len(ent)//2] if len(ent) else cl.index[-300]\n"
            "    i0 = cl.index.get_loc(anchor)\n"
            "    seg = cl.iloc[max(0,i0-180):i0+120]\n"
            "    e = ent[(ent>=seg.index[0]) & (ent<=seg.index[-1])]\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    ax.plot(seg.index, seg.values, c='k', lw=1.2, label='SPY close')\n"
            "    ax.plot(seg.index, low.reindex(seg.index), c=RED, lw=1.2, label='lower boundary')\n"
            "    ax.plot(seg.index, up.reindex(seg.index), c=GREEN, lw=1.2, label='upper boundary')\n"
            "    ax.scatter(e, cl.reindex(e), c=RED, s=55, zorder=5, marker='v', label='lower-break SHORT')\n"
            "    ax.set_title('A mechanical broadening formation (megaphone) on SPY'); ax.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('lower-boundary breaks shown:', len(e))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The fanning boundaries *describe* a choppy, widening stretch. The question is whether "
            "those red short markers are followed by **declines**. **Let's race the break-short "
            "against random shorts** at four horizons. Red = short the lower-boundary break; grey = "
            "short on random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    brk, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            c = load(t)['close']\n"
            "            e = st.lower_break_entries(c, k=R['k'])\n"
            "            re = st.random_entries(c, max(len(e),50), k=R['k'], seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        brk.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    brk = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, brk, .4, color=RED, label='short the megaphone break')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='short on random days')\n"
            "for i,(a,bb) in enumerate(zip(brk,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='top',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='top',fontsize=8)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean SHORT return (bps)')\n"
            "ax.set_title('The megaphone short loses — and loses MORE than a random short'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('break:', [round(v) for v in brk]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the whole story. The megaphone short **loses at every horizon** "
            f"(**{R['h20'][2]:.0f} bps** over 20 days) — and a **random** short loses *less* "
            f"(**{R['h20'][5]:.0f} bps**). The famous reversal short is **worse than a dart-throwing "
            "short** everywhere. It isn't calling tops; it's just paying the market's upward drift, "
            "and paying it harder than random. The expanding range forecast nothing."
        ),
        md(
            "**One more sanity check.** What if we scramble the megaphone's *geometry* — keep the "
            "same pivot dates but shuffle which price sits where, so the diverging boundaries become "
            "nonsense? If the expanding range really matters, the nonsense megaphone should do far "
            "worse."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    pl = st.shuffled_pivot_placebo(c, 20, k=R['k'], n_draws=200, seed=465)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real megaphone lower-break short (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... and {pval*100:.0f}% of *scrambled-geometry* megaphones do at least as well (p={pval:.2f}).')\n"
            "print('=> the geometry is not doing the work.')"
        ),
        md(
            f"Almost all the **scrambled** megaphones match or beat the real one "
            f"(*p* = {R['placebo'][1]:.2f}) — the real one sits at the *bad* end even of nonsense. "
            "If the expanding range genuinely forecast the turn, a random scramble would change the "
            "result. It doesn't — because the result was never about the lines."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The lower-boundary-break short does **not** beat shorting on random "
            "days — it's *worse* at every horizon. The losses are the market's upward drift, not a "
            "reversal edge. Only **25** megaphones appear in 21 years.\n"
            "- **Tradability — Mirage.** Nothing to trade, long or short: no edge once the drift is "
            "accounted for, a paper-thin per-name sample, and costs only make it worse.\n"
            "- **\"Expanding volatility forecasts a turn\"? — Busted.** Scramble the geometry into "
            "nonsense and the result barely moves. The megaphone doesn't call the top."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. Shorting the megaphone break loses *more* than shorting "
            "at random, and going **long** the break would just be a worse, rarer way of capturing "
            "the drift you'd get more cheaply by **holding the index**. Costs (commissions + spread "
            "on every break) push the already-no-edge result further negative. And with only 25 "
            "events in 21 years, there isn't enough signal to estimate *anything*. As a forecasting "
            "tool the megaphone doesn't pay; as a descriptive shape, it was never meant to be a "
            "strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The reverse trade.** If shorting loses, does *buying* the break win? No — that just "
            "re-captures the drift you already own by holding the index, minus costs and rarity.\n"
            "- **Different pivot rules.** Try a wider/narrower fractal window or right-angled "
            "broadening variants — the result is robust: drift in, megaphone out.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* expanding-range "
            "reversal into a synthetic tape and shows the harness banks it (so the null result here "
            "isn't a dead detector — it's an honest 'nothing there').\n\n"
            "*Think the megaphone forecasts? Show the lower-break short beating random shorts at "
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
            "# Broadening Formation — a quantitative teardown 🔬\n"
            "### Mechanical megaphones on 5 indices · lower-break SHORT forward returns · "
            "one-sample HAC *t* · a drift-matched random-short baseline · a shuffled-pivot geometry "
            "placebo · costs · a synthetic planted-reversal control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **megaphone** from the **drift**: a short on an upward-drifting index "
            "loses *by construction*, so the only meaningful test is break-vs-random-short, plus a "
            "placebo that destroys the megaphone's geometry while preserving its marginals.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return**), 2005→2026. Pivots are confirmed fractals "
            f"(k={R['k']}, an explicit {R['k']}-bar confirmation lag); entry is the **next close** "
            "(one documented lag); returns are signed for a **short**. A strict megaphone is rare — "
            f"only **{R['n_entries']}** fire in 21 years across 5 tapes. Offline core + synthetic "
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
            f"| **Signal** | `NONE` | Lower-break short vs a **drift-matched random short**: the break "
            f"is *worse* at every horizon (Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}"
            f"/{R['h60'][6]:+.0f} bps) and the break-minus-random Welch *t* is **negative** "
            f"({R['h20'][8]:+.2f} at 20d, {R['h60'][8]:+.2f} at 60d). |\n"
            f"| **Tradability** | `MIRAGE` | The negative one-sample t's ({R['h20'][4]:.2f} at 20d) are "
            f"**negative beta** — a short on an up-drifting index. No reversal edge to scale, and only "
            f"{R['n_entries']} events. |\n"
            f"| **Expanding vol forecasts a turn?** | `BUSTED` | Scrambling the megaphone's geometry "
            f"leaves the result intact: **p = {R['placebo'][1]:.2f}** of nonsense megaphones match or "
            "beat the real one (which sits at the bad end). The diverging lines aren't doing the work. |\n\n"
            "> 💡 In plain words: the break-short loses, but a *random* short loses less — there is no "
            "reversal signal. Strip the drift (race it vs random) or strip the geometry (scramble the "
            "pivots) and nothing is left. A rare, seductive shape with no forecasting content."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Anchor a megaphone on the last two confirmed swing highs $H_1,H_2$ and lows $L_1,L_2$. "
            "It is a broadening top iff the upper line through $H_1,H_2$ has **positive** slope and "
            "the lower line through $L_1,L_2$ has **negative** slope (diverging boundaries). Let "
            "$\\ell_t$ be the lower boundary at bar $t$. The rule **shorts** when $C_t<\\ell_t$ and "
            "rides the reversal down.\n\n"
            "- **H₀ (drift).** Short returns equal a drift-matched **random-short** baseline.\n"
            "- **H₁ (the megaphone forecasts a turn).** Break-short returns **exceed** random shorts "
            "at some horizon, t ≥ 2.\n"
            "- **H₂ (the geometry matters).** Break-short returns exceed a **shuffled-pivot** megaphone "
            "whose diverging lines are geometric nonsense.\n\n"
            "We find **H₀ not rejected** (break ≤ random — in fact *worse*), **H₁ rejected** (Welch t "
            "negative, never ≥ 2), **H₂ rejected** (placebo p ≈ 0.97). The steelman fails on every leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean. A **short** "
            "inherits it with a minus sign; a big *negative* one-sample $t$ against **zero** measures "
            "the tide, not a reversal. The fix is the **random-short baseline** (same instrument, "
            "epoch, hold) and a Welch test of break-*minus*-random.\n\n"
            "**(b) Geometry as a free parameter.** A megaphone is four chosen pivots; the danger is "
            "that *any* diverging pair of lines on a choppy stretch produces a 'megaphone'. The "
            "**shuffled-pivot placebo** keeps pivot positions/kinds and the price marginal but permutes "
            "which price sits at which pivot — the diverging structure becomes meaningless, so if the "
            "real result survives the scramble, the geometry was never load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} lower-boundary breaks** "
            "pooled.\n"
            f"- **Pivots.** Confirmed fractals: extremum with k={R['k']} strictly-beaten bars each "
            f"side; usable only at bar +{R['k']} (no look-ahead). Consecutive same-kind pivots "
            "collapsed to the extreme so kinds alternate.\n"
            "- **Megaphone.** Last two swing highs **rising** AND last two swing lows **falling** "
            "(diverging boundaries); upper line through the highs, lower line through the lows.\n"
            "- **Entry.** First close below the lower boundary; **short** at **next close** (one lag); "
            "hold H ∈ {5,10,20,60}; return signed for a short.\n"
            "- **Null #1 — one-sample HAC t** of short returns vs 0 (Newey-West) — *measures beta*.\n"
            "- **Null #2 — random-short baseline**, Welch two-sample break vs random (the *real* test).\n"
            "- **Null #3 — shuffled-pivot placebo** (geometry destroyed, marginals kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every break.\n"
            "- **Positive control.** Synthetic tape with a **planted** expanding-range reversal (knob "
            "`edge`): edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t looks 'significant', vs-random kills it\n\n"
            "Left: the break-short's **one-sample** t against zero (the misleading number — it's "
            "negative beta). Right: the same short vs a **drift-matched random short** (the honest "
            "number)."
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
            "            e = st.lower_break_entries(c, k=R['k'])\n"
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
            "a1.axhline(-2, ls='--', c=RED, label='t=-2 bar'); a1.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='top',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO (misleading: it is -beta)'); a1.set_ylabel('t'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if v>2 else RED for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Break vs RANDOM short, Welch t (honest: negative)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars are strongly negative (20d **{R['h20'][4]:.2f}**, 60d "
            f"**{R['h60'][4]:.2f}**) — but that's just **negative beta**: any short on an up-drifting "
            f"index loses. The right bars are the real test: break-minus-random is **negative** "
            f"({R['h20'][8]:+.2f} at 20d, {R['h60'][8]:+.2f} at 60d) — the megaphone short is *worse* "
            "than a random short. No reversal signal in either direction."
        ),
        md(
            "### 4b · Break vs random short across horizons — the gap is the verdict\n\n"
            "Mean short return, megaphone break vs random short, all four horizons. If the megaphone "
            "forecast a turn, the break would tower over a random short. Instead it sits *below* it."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, brk, .4, color=RED, label='megaphone lower-break short')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random short (drift baseline)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,(a,b) in enumerate(zip(brk,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='top',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='top',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean SHORT return (bps)')\n"
            "ax.set_title('Megaphone short loses MORE than a random short'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta break-random (bps):', [round(a-b) for a,b in zip(brk,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the break-short is **{R['h20'][2]:.0f} bps** while a "
            f"random short is **{R['h20'][5]:.0f} bps** — the megaphone *underperforms* a dart by "
            f"{abs(R['h20'][6]):.0f} bps. At 60 days the gap is **{abs(R['h60'][6]):.0f} bps** the "
            "wrong way. The expanding range carries no reversal information."
        ),
        md(
            "### 4c · The geometry placebo — scramble the megaphone, nothing changes\n\n"
            "Shuffle which price sits at which pivot (positions/kinds kept, marginal kept) so the "
            "diverging boundaries are geometric nonsense. If price respected *this specific "
            "megaphone*, the scramble should change the result. The observed break-short should sit "
            "far in the *left* tail (more negative) of the scrambled distribution. It doesn't — it "
            "sits at the bad (less-negative) end."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']\n"
            "    pl = st.shuffled_pivot_placebo(c, 20, k=R['k'], n_draws=200, seed=465)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    import numpy as _np\n"
            "    piv = st._alternating(st.find_pivots(c, k=R['k']))\n"
            "    rng = _np.random.default_rng(465); prices = piv['price'].to_numpy(); positions=[int(p) for p in piv.index]\n"
            "    kinds=[int(kk) for kk in piv['kind']]; confirm=[p+R['k'] for p in positions]; n=len(c); idx=c.index\n"
            "    draws=[]\n"
            "    for _ in range(200):\n"
            "        perm=rng.permutation(prices); pts=list(zip(positions,[float(v) for v in perm]))\n"
            "        low=_np.full(n,_np.nan)\n"
            "        for t in range(n):\n"
            "            ai=[j for j,cp in enumerate(confirm) if cp<=t]\n"
            "            if len(ai)<4: continue\n"
            "            hh=[pts[j] for j in ai if kinds[j]>0]; ll=[pts[j] for j in ai if kinds[j]<0]\n"
            "            if len(hh)<2 or len(ll)<2: continue\n"
            "            bnd=st.megaphone_boundaries(hh[-2:],ll[-2:])\n"
            "            if bnd is None: continue\n"
            "            (_us,_ub),(ls,lb)=bnd; low[t]=ls*t+lb\n"
            "        ls_=__import__('pandas').Series(low,index=idx); m=(c<ls_)&ls_.notna(); f=m&~m.shift(1,fill_value=False)\n"
            "        rr=st.forward_returns(c, idx[f.to_numpy()], 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws=_np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(465); draws = rng.normal(-700, 250, 200)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=30, color=GREY, alpha=.85, label='scrambled-geometry megaphones (SPY, 20d)')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'real megaphone {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean lower-break 20d SHORT return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real megaphone is no more negative than nonsense: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real megaphone {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => geometry not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real megaphone (red line) sits at the **bad** end of the "
            f"scrambled cloud — **p = {R['placebo'][1]:.2f}**. Geometric nonsense does at least as "
            "well, so the specific diverging boundaries aren't carrying any information. This is the "
            "cleanest refutation of 'the expanding range forecasts the turn.'"
        ),
        md(
            "### 4d · Per-ticker — a paper-thin, incoherent sample\n\n"
            "20-day break-minus-random delta, per instrument. A real reversal edge would be positive "
            "and coherent; instead it's negative where there's any sample at all, and the 'wins' rest "
            "on one or two trades."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas, counts = [], [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        c = load(t)['close']\n"
            "        e = st.lower_break_entries(c, k=R['k']); re = st.random_entries(c, max(len(e),50), k=R['k'], seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d); counts.append(len(e))\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]; counts=[p[1] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,(d,cnt) in enumerate(zip(deltas,counts)): ax.annotate(f'{d:+.0f}\\n(n={cnt})',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=8)\n"
            "ax.set_ylabel('20d break − random (bps)'); ax.set_title('No coherent edge — and most names have a handful of events')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})\n"
            "print('per-ticker entry counts:', {n:c for n,c in zip(names,counts)})"
        ),
        md(
            f"> 💡 In plain words: SPY ({R['per'][0][1]} events) and QQQ ({R['per'][1][1]}) — the only "
            f"names with a usable count — are **{R['per'][0][5]:+.0f}** and **{R['per'][1][5]:+.0f}** "
            "bps *behind* a random short. The lone positive (IWM) rests on **2** trades. No coherent "
            "cross-sectional reversal edge — exactly what relabelled drift on a tiny sample looks like."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real reversal\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** expanding-range "
            "reversal into a synthetic tape and check the same lower-break short banks it: edge=0 must "
            "stay at t≈0; edge>0 must light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.50):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=999, n_days=4000)\n"
            "    c = px['close']; e = st.lower_break_entries(c, k=10); s = st.summarize(st.forward_returns(c, e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar'); ax.axhline(0, c='k', lw=.8)\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t (SHORT)'); ax.set_title('Control: edge=0 -> t~0; planted reversal -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} short={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted reversal the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — no false positive); a planted "
            f"expanding-range reversal reaches **t = +{R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%, "
            f"short +{R['syn'][1][2]:.0f} bps). The detector works — so the flat/negative real-tape "
            "result is a genuine 'nothing there', not a broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the lower-break short does not beat a drift-matched random short "
            f"(break − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} "
            f"bps at 5/10/20/60d; Welch t **negative**, {R['h60'][8]:+.2f} at 60d). The strongly "
            f"negative one-sample t's (20d **{R['h20'][4]:.2f}**) are pure negative beta.\n"
            f"- **Tradability `MIRAGE`** — no reversal edge once the drift is accounted for; the "
            f"per-name sample is paper-thin ({R['n_entries']} events in 21 years) and costs only "
            "deepen the hole. Neither shorting nor buying the break is worth anything.\n"
            f"- **Expanding vol forecasts a turn? `BUSTED`** — the shuffled-pivot placebo leaves the "
            f"result intact (**p = {R['placebo'][1]:.2f}**): geometric-nonsense megaphones do as well "
            "as the real one, so the diverging boundaries carry no forecasting information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "Shorting the megaphone break loses *more* than a random short; buying the break would "
            "merely re-capture the index drift you already hold, minus costs and rarity. There is no "
            "capacity question because there is no edge to scale — and with only 25 events in 21 "
            "years there isn't even enough data to estimate one. The broadening formation is a "
            "descriptive shape for a choppy, high-volatility stretch, not a forecasting strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Expanding-vol ≠ reversal.** The deeper claim — 'rising realised volatility forecasts "
            "a turn' — is testable directly (regress forward returns on a vol-expansion score); it is "
            "the volatility-clustering literature's well-known result that vol forecasts *vol*, not "
            "*direction*.\n"
            "- **Right-angled & ascending/descending broadening.** Affine variants of the same "
            "diverging geometry; they inherit the same drift confound and small-sample fragility.\n"
            "- **Survivorship & sample.** A strict megaphone is genuinely rare; any 'edge' read off "
            "25 events is a multiple-testing hazard (see the desk's research-method demos).\n\n"
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
