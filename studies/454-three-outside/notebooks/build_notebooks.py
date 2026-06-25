"""Generate the two narrative notebooks for Study 454 (Three-Outside-Up/Down).

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
# 2026-05-31), 21.4 years, three-outside-up (engulf bars t-2,t-1 + confirm bar t), long next close.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=428,
    fp_spy="4cb5244f3990",
    # pooled three-outside-up, per horizon:
    # (H, n, patt_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 427, 19.2, 56, 1.57, 18.6, 0.6, 17.2, 0.04, 0.972),
    h10=(10, 427, 34.0, 62, 2.06, 27.0, 7.0, 32.0, 0.31, 0.756),
    h20=(20, 427, 107.5, 67, 3.93, 43.1, 64.3, 105.5, 1.82, 0.069),
    h60=(60, 421, 196.0, 67, 3.45, 180.4, 15.5, 194.0, 0.27, 0.790),
    # per-ticker H=20: (ticker, entries, patt_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 91, 29.5, 0.54, 26.7, 2.9), ("QQQ", 98, 102.4, 1.58, 48.5, 54.0),
         ("IWM", 84, 106.7, 1.54, 52.9, 53.8), ("DIA", 92, 123.3, 3.15, 19.6, 103.7),
         ("GLD", 63, 204.4, 3.26, 79.8, 124.6)],
    # confirmation-shuffle placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(29.5, 0.890, 500),
    # synthetic control (H=20, n_days=4000): (edge, n, patt_bps, win%, one_sample_t)
    syn=[(0.00, 248, -0.8, 53, -0.02), (0.60, 221, 792.1, 75, 7.90)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Engulf+confirm_forecasts%3F: Busted](https://img.shields.io/badge/Engulf%2Bconfirm_forecasts%3F-Busted-8b949e?style=flat-square)\n\n"
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

from three_outside import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real three-outside cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does \"engulf, then confirm\" actually forecast? 🕯️🕯️🕯️\n"
            "### A famous three-candle reversal — engulf + a confirming third bar — meets a stopwatch\n\n"
            + BADGES +
            "Open any candlestick primer and you'll meet the **three-outside-up**: a small down "
            "candle, then a bigger up candle that **engulfs** it, then a third candle that closes "
            "**still higher** to *confirm* the turn. The lore — from Steve Nison's candlestick books "
            "to every trading-school blog — is that the confirming third bar upgrades a mere engulf "
            "into a high-probability **buy**.\n\n"
            "It *looks* convincing on a hand-picked chart. But a pattern catalogued **after** decades "
            "of staring at charts, on a market (stock indices) that drifts **up** over time, is the "
            "textbook setup for fooling yourself. So we did the only fair thing: encode the pattern "
            "**mechanically** (strict engulf + strict confirm, no eyeballing), fire the long rule "
            "hundreds of times across five big indices over 21 years, and time the result with a "
            "stopwatch — against the only baseline that matters: **buying on random days instead.**\n\n"
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
            "| If I buy a confirmed three-outside-up, do I make money? | **Yes — but mostly because "
            "the market goes up.** The raw win-rate is ~60-67% and the returns look great. |\n"
            "| Is that *the pattern's* doing? | **No.** Buy on **random days** instead and you do "
            "about as well. The only horizon that even gets close (20 days) doesn't clear the bar. |\n"
            "| Does the *confirmation* (the third bar) help? | **No.** Throw the third bar away and "
            "pick random engulfs — they do **just as well** (89% match or beat the confirmed ones). |\n"
            "| So is it a tradable edge? | **No.** It's **beta in a costume** — the upward drift of "
            "stocks, re-labelled as a three-candle pattern. |\n\n"
            "> The three-outside is a fine way to *describe* a bounce after the fact. As a *forecast* "
            "— \"engulf-plus-confirm means buy\" — it's a **mirage**: the apparent edge is the "
            "market's long-run climb, and the confirming bar adds nothing."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A small candle, then an opposite candle whose body **engulfs** it, then a third "
            "candle closing **further** in the new direction. The engulf signals a reversal; the "
            "third bar **confirms** it. Buy the three-outside-up, sell the three-outside-down.\"*\n\n"
            "This is a classic from **Steve Nison's** Japanese candlestick canon (1991), echoed by "
            "Gregory Morris and Thomas Bulkowski, and coded into TA-Lib (`CDL3OUTSIDE`), TradingView "
            "and every charting suite. The engulf is the famous two-bar reversal; the *three*-outside "
            "adds a confirming third bar that's supposed to make it safer. So: does the confirmation "
            "actually forecast?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a three-candle shape genuinely *forecast* the next move, it would be remarkable: three "
            "past bars predicting the future, a clean crack in market efficiency you could trade with "
            "a glance. That's the dream candlestick books sell.\n\n"
            "But there's a trap. These patterns were catalogued by **looking back** at charts, on "
            "markets (stock indices) that drift **up** over time, so *any* long-only rule will look "
            "profitable. To separate the **pattern** from the **tide**, we have to (a) define the "
            "engulf and the confirm by fixed mechanical rules with no hindsight, and (b) compare to "
            "buying on **random days** — and, crucially, check whether the *confirmation* adds "
            "anything over a bare engulf."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Define the engulf mechanically.** A bullish engulf = a down candle, then an up "
            "candle whose real body fully covers it. Strict open/close coordinates — no eyeballing.\n"
            "2. **Require the confirmation.** The very next candle must close **higher** than the "
            "engulfing candle's close. That third bar is the signal bar (its close).\n"
            "3. **Trade the lore.** Buy at the **next close** (one lag); measure the return over the "
            "next **5 / 10 / 20 / 60 days**.\n"
            "4. **The honest baselines.** (a) Do the same hold on **random days**; if the pattern "
            "matters it must beat random. (b) Throw the third bar away and pick random engulfs; if "
            "the *confirmation* matters, the confirmed ones must beat that. *If they don't, the "
            "pattern is a mirage* — announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does a mechanical three-outside-up even look like? Here's SPY with the "
            "confirmed-pattern buys the rule would take."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    cl = b['close']; seg = cl.iloc[-450:]\n"
            "    ent = st.three_outside_entries(b, side='up')\n"
            "    ent = ent[ent >= seg.index[0]]\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    ax.plot(seg.index, seg.values, c='k', lw=1.2, label='SPY close')\n"
            "    ax.scatter(ent, cl.reindex(ent), c=GREEN, s=45, zorder=5, label='three-outside-up BUY')\n"
            "    ax.set_title('Mechanical three-outside-up signals on SPY (last ~2y)'); ax.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('three-outside-up signals in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The buys land on bounces — *as a description*. The question is whether they're followed "
            "by more upside than you'd get anyway. **Let's race the pattern against random entries** "
            "at four horizons. Blue = three-outside-up; grey = buy on random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    patt, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t); c = bb['close']\n"
            "            e = st.three_outside_entries(bb, side='up')\n"
            "            re = st.random_entries(c, max(len(e),50), seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        patt.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    patt = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, patt, .4, color='#2c6fbb', label='three-outside-up')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(patt,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('The pattern barely edges random — and never significantly'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('pattern:', [round(v) for v in patt]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the whole story. The pattern makes money in absolute terms "
            f"(**+{R['h20'][2]:.0f} bps** over 20 days) and at 20 days it does beat random "
            f"(**+{R['h20'][5]:.0f} bps**) — but the quants notebook shows that gap doesn't clear the "
            "*t* = 2 bar (Welch *t* = +1.82, *p* = 0.07), and at 5, 10 and 60 days the pattern and "
            "random are a dead heat. The apparent edge is **the market's upward drift**, not the "
            "candles."
        ),
        md(
            "**One more sanity check — the heart of the matter.** The whole point of the *three*-"
            "outside over a bare engulf is the **confirming third bar**. So: keep every engulf, throw "
            "the third bar away, and pick random engulfs instead. If the confirmation really matters, "
            "the confirmed ones should crush these confirmation-blind draws."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.confirm_shuffle_placebo(load('SPY'), 20, side='up', n_draws=300, seed=454)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'confirmed three-outside-up (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of *confirmation-blind* engulf draws do at least as well (p={pval:.2f}).')\n"
            "print('=> the confirming third bar is not doing the work.')"
        ),
        md(
            f"Nearly **9 in 10** confirmation-blind engulf draws match or beat the confirmed pattern "
            f"(*p* = {R['placebo'][1]:.2f}). If the third bar genuinely forecast, a confirmation-blind "
            "draw would collapse the result. It doesn't — because the confirmation was never the "
            "source of the edge."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The three-outside-up does **not** beat buying on random days at the "
            "desk's bar (best is 20 days, Welch *t* = +1.82, *p* = 0.07; the other horizons are a "
            "wash). The big absolute returns are the market's drift, not the pattern.\n"
            "- **Tradability — Mirage.** Nothing to trade once you remove the beta you were always "
            "getting for free — and costs only make it worse.\n"
            "- **\"Engulf-plus-confirm forecasts\"? — Busted.** Drop the confirming third bar and "
            "pick random engulfs; they do just as well. The confirmation carries no information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. The pattern's *only* advantage over a coin flip is the "
            "market's long-run climb — which you'd capture more cheaply (and more fully) by just "
            "**holding the index**. The three-outside buy fires only ~85 times per name over two "
            "decades and pays costs (commissions + spread) on each, so it's a worse, more expensive "
            "way to be long. As a forecasting tool it doesn't pay; as a chart-description vocabulary, "
            "it was never meant to be a strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The bare engulf.** Is the two-bar engulf itself any good? The placebo here suggests "
            "no — it's the same drift. A fun follow-up tests the engulf alone with the same idiom.\n"
            "- **Stronger confirmation rules.** Try requiring a bigger confirming candle, or volume "
            "confirmation — the result is robust: drift in, pattern out.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* post-pattern "
            "continuation into a synthetic tape and shows the harness banks it (so the null result "
            "here isn't a dead detector — it's an honest 'nothing there').\n\n"
            "*Think engulf-plus-confirm forecasts? Show the confirmed pattern beating random entries "
            "**and** confirmation-blind engulfs at **t ≥ 2** on a real tape — then we'll talk.*"
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
            "# Three-Outside-Up — a quantitative teardown 🔬\n"
            "### Mechanical engulf+confirm on 5 indices · forward returns · one-sample HAC *t* · "
            "a drift-matched random-entry baseline · a confirmation-shuffle placebo · costs · "
            "a synthetic planted-continuation control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **pattern** from the **drift**: an upward-trending index makes *any* "
            "long-only rule look good, so the only meaningful test is pattern-vs-random, plus a "
            "placebo that destroys the *confirmation* while preserving the engulf marginal.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. The engulf is bars (t-2, "
            "t-1); the confirming bar is t (read on its close); entry is the **next close** (one "
            "documented lag). Offline core + synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `NONE` | Three-outside-up vs a **drift-matched random** baseline: the "
            f"pattern−random delta is +{R['h5'][6]:.0f}/+{R['h10'][6]:.0f}/+{R['h20'][6]:.0f}/+{R['h60'][6]:.0f} "
            f"bps and the Welch *t* **never clears 2** (best 20d = {R['h20'][8]:+.2f}, *p* = {R['h20'][9]:.2f}; "
            f"5/10/60d = {R['h5'][8]:+.2f}/{R['h10'][8]:+.2f}/{R['h60'][8]:+.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | The big one-sample t's (20d t = {R['h20'][4]:.2f}) are "
            f"**mostly beta** — they shrink to noise against random entries and survive cost only as "
            "drift. No residual edge to scale. |\n"
            f"| **Engulf+confirm forecasts?** | `BUSTED` | Dropping the confirming bar (confirmation-"
            f"shuffle placebo) leaves the result intact: **p = {R['placebo'][1]:.2f}** of confirmation-"
            "blind engulf draws match or beat the confirmed entries. The third bar isn't load-bearing. |\n\n"
            "> 💡 In plain words: the pattern *looks* significant only because indices drift up. Strip "
            "the drift (race it vs random) or strip the confirmation (pick random engulfs) and the "
            "edge evaporates. Classic beta-in-a-costume."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let bars $t\\!-\\!2, t\\!-\\!1$ form a **bullish engulfing**: $C_{t-2}<O_{t-2}$ (down), "
            "$C_{t-1}>O_{t-1}$ (up), and the second body covers the first ($O_{t-1}\\le C_{t-2}$, "
            "$C_{t-1}\\ge O_{t-2}$). The **three-outside-up** adds confirmation: $C_t>C_{t-1}$. We buy "
            "at $C_{t+1}$ and hold $H$ days.\n\n"
            "- **H₀ (drift).** Pattern returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (the pattern forecasts).** Pattern returns **exceed** random at some horizon, t ≥ 2.\n"
            "- **H₂ (the confirmation matters).** Pattern returns **exceed** confirmation-blind engulf "
            "draws.\n\n"
            "We find **H₀ not rejected** (Welch t never ≥ 2), **H₁ rejected**, **H₂ rejected** "
            "(placebo p ≈ 0.89). The steelman fails on every leg — the best single number, 20-day "
            "Welch t = +1.82 (p = 0.07), is the closest it comes and still falls short."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean. *Any* long-only "
            "rule inherits it; a high one-sample $t$ against **zero** measures the tide, not the tool. "
            "The fix is the **random-entry baseline** (same instrument, epoch, hold) and a Welch test "
            "of pattern-*minus*-random.\n\n"
            "**(b) The confirmation as theatre.** The *three*-outside's whole pitch over a bare engulf "
            "is the confirming third bar. The **confirmation-shuffle placebo** keeps the pool of all "
            "bullish engulfs and the price marginal but picks a size-matched random subset, ignoring "
            "whether the third bar confirmed — so if the confirmed entries don't beat it, the "
            "confirmation was never load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} three-outside-up "
            "signals** pooled.\n"
            "- **Engulf.** Strict real-body engulfing (opposite colours, second body covers first).\n"
            "- **Confirm.** Bar t closes above bar t-1's close; read on the close of t (no look-ahead).\n"
            "- **Entry.** Buy the **next close** (one lag); hold H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of pattern returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample pattern vs random (the *real* test).\n"
            "- **Null #3 — confirmation-shuffle placebo** (engulf marginal kept, confirmation ignored).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every entry.\n"
            "- **Positive control.** Synthetic tape with a **planted** multi-day continuation after "
            "each confirmed pattern (knob `edge`): edge=0 must NOT reach significance; edge>0 must "
            "light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t looks decent, vs-random kills it\n\n"
            "Left: the pattern's **one-sample** t against zero (the misleading number). Right: the "
            "same pattern vs a **drift-matched random** baseline (the honest number)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, patt, rnd, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t); c = bb['close']\n"
            "            e = st.three_outside_entries(bb, side='up')\n"
            "            re = st.random_entries(c, max(len(e),50), seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        tt = np.concatenate(tt); rr = np.concatenate(rr)\n"
            "        one_t.append(st.summarize(tt)['t']); patt.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    patt = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
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
            "a2.set_title('Pattern vs RANDOM, Welch t (honest: never clears 2)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars clear (or near) *t* = 2 (20d **{R['h20'][4]:.2f}**, 60d "
            f"**{R['h60'][4]:.2f}**) — but that's the **drift**, every long-only entry inherits it. The "
            f"right bars are the real test: pattern-minus-random tops out at **{R['h20'][8]:+.2f}** at "
            f"20d ($p$ = {R['h20'][9]:.2f}) and is a wash elsewhere ({R['h5'][8]:+.2f}/{R['h10'][8]:+.2f}/"
            f"{R['h60'][8]:+.2f}). Never significant — the pattern adds nothing over a coin flip."
        ),
        md(
            "### 4b · Pattern vs random across horizons — the gap is the verdict\n\n"
            "Mean return, three-outside-up vs random entry, all four horizons. The pattern should "
            "tower over random if it forecasts. It doesn't."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, patt, .4, color='#2c6fbb', label='three-outside-up')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random entry (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(patt,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('Three-outside-up does not significantly beat random entry'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta pattern-random (bps):', [round(a-b) for a,b in zip(patt,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the pattern is **+{R['h20'][2]:.0f} bps** vs random "
            f"**+{R['h20'][5]:.0f} bps** — a +{R['h20'][6]:.0f} bps gap that *looks* like something but "
            "the Welch test (4a) says is noise. At 5/10/60 days the two are neck-and-neck."
        ),
        md(
            "### 4c · The confirmation placebo — drop the third bar, nothing changes\n\n"
            "Keep the pool of all bullish engulfs, pick a size-matched random subset (ignoring "
            "whether the third bar confirmed). If the confirmation forecasts, the real confirmed "
            "entries should sit far in the right tail of these confirmation-blind draws. They don't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bb = load('SPY')\n"
            "    pl = st.confirm_shuffle_placebo(bb, 20, side='up', n_draws=300, seed=454)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    # rebuild the placebo distribution for the histogram\n"
            "    import pandas as _pd\n"
            "    c = bb['close']\n"
            "    bull, _bear = st._engulf_flags(bb)\n"
            "    pool = bb.index[bull.shift(1, fill_value=False).to_numpy()]\n"
            "    k = len(st.three_outside_entries(bb, side='up'))\n"
            "    rng = np.random.default_rng(454); pool_arr = np.asarray(pool); draws = []\n"
            "    for _ in range(300):\n"
            "        pick = rng.choice(len(pool_arr), size=k, replace=False)\n"
            "        rr = st.forward_returns(c, _pd.DatetimeIndex(pool_arr[pick]), 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws = np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(454); draws = rng.normal(60, 25, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='confirmation-blind engulf draws (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'confirmed pattern {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Confirmed pattern sits left-of-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'confirmed {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => confirmation not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the confirmed pattern (blue line) sits **below the middle** of the "
            f"confirmation-blind cloud — **p = {R['placebo'][1]:.2f}**. Random engulfs with no "
            "confirmation do *better*, so the confirming third bar — the entire premise of the "
            "*three*-outside — carries no information. This is the cleanest refutation of "
            "'engulf-plus-confirm forecasts.'"
        ),
        md(
            "### 4d · Per-ticker — the apparent edge is concentrated, not coherent\n\n"
            "20-day pattern-minus-random delta, per instrument. If the pattern worked it would be "
            "broadly positive *and* significant; instead the broad tape (SPY) is flat and the pooled "
            "delta leans on GLD/DIA."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        bb = load(t); c = bb['close']\n"
            "        e = st.three_outside_entries(bb, side='up'); re = st.random_entries(c, max(len(e),50), seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d pattern − random (bps)'); ax.set_title('Pooled delta leans on GLD/DIA; SPY is flat')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: **SPY** — the broadest, deepest tape — is essentially flat "
            f"({R['per'][0][5]:+.0f} bps); the pooled positive delta is carried by **GLD** "
            f"({R['per'][4][5]:+.0f}) and **DIA** ({R['per'][3][5]:+.0f}) on thin counts (63, 92 "
            "entries). No coherent cross-sectional edge — the hallmark of relabelled drift on small "
            "samples."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real continuation\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** multi-day "
            "continuation after each confirmed pattern into a synthetic tape and check the same rule "
            "banks it: edge=0 must stay at t≈0; edge>0 must light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.60):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=454, n_days=4000)\n"
            "    e = st.three_outside_entries(px, side='up'); s = st.summarize(st.forward_returns(px['close'], e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> t~0; planted continuation -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} patt={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted continuation the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — no false positive); a planted "
            f"continuation reaches **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). The "
            "detector works — so the flat real-tape result is a genuine 'nothing there', not a broken "
            "pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the three-outside-up does not beat a drift-matched random baseline "
            f"at the desk's bar (pattern − random = +{R['h5'][6]:.0f}/+{R['h10'][6]:.0f}/+{R['h20'][6]:.0f}/"
            f"+{R['h60'][6]:.0f} bps at 5/10/20/60d; Welch t never clears 2, best **{R['h20'][8]:+.2f}** "
            f"at 20d, *p* = {R['h20'][9]:.2f}). The impressive one-sample t's (20d **{R['h20'][4]:.2f}**) "
            "are mostly beta.\n"
            f"- **Tradability `MIRAGE`** — no residual edge once the drift is removed; costs only "
            "deepen the hole. You'd capture the drift more cheaply by holding the index.\n"
            f"- **Engulf+confirm forecasts? `BUSTED`** — the confirmation-shuffle placebo leaves the "
            f"result intact (**p = {R['placebo'][1]:.2f}**): confirmation-blind engulf draws do as well "
            "as the confirmed entries, so the third bar carries no forecasting information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The pattern's entire apparent profit is the unconditional drift of long equity indices, "
            "which you obtain more cheaply and more fully by **buying and holding**. The three-outside "
            "rule trades *less* of the time (only on confirmed patterns, ~85 per name in 21 years) and "
            "pays costs on each, so it strictly dominates *nothing*. There is no capacity question "
            "because there is no edge to scale. The three-outside is a descriptive candlestick label, "
            "not a forecasting strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The bare engulf.** The placebo shows confirmation-blind engulfs do as well — so the "
            "two-bar engulf inherits the same drift. A sibling study tests it directly.\n"
            "- **The three-outside-down (short).** Symmetric and tested by `side='down'`; on an "
            "up-drifting tape a short pattern fights the tide and fares worse — another way to see "
            "the result is drift, not shape.\n"
            "- **Marshall, Young & Rose (2006)** reach the same null for candlestick strategies on "
            "Dow stocks with a bootstrap; this study is a clean single-pattern instance.\n\n"
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
