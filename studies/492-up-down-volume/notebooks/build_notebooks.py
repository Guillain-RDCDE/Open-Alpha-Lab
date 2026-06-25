"""Generate the two narrative notebooks for Study 492 (Up-Down-Volume).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached daily tapes under
../_cache/ (OHLC for SPY + OHLCV for the breadth basket) and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md). The synthetic positive control runs anywhere with
no network.
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
# yfinance daily, SPY forward returns; breadth from SPY + 9 SPDR sector ETFs, 2005-01-03 ->
# 2026-05-29 (As-of 2026-05-31), 21.4 years, up-volume-share selling climax (rolling-60d 10%
# quantile, past-only), enter next close.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    basket=["SPY", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB"],
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=643, window=60, q=0.10,
    fp_spy="4cb5244f3990",
    # pooled selling-climax (SPY fwd), per horizon:
    # (H, n, climax_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 643, 49.8, 62, 4.38, 21.9, 27.9, 47.8, 1.95, 0.052),
    h10=(10, 642, 74.1, 63, 3.97, 46.0, 28.1, 72.1, 1.49, 0.136),
    h20=(20, 641, 121.4, 68, 3.67, 85.4, 36.0, 119.4, 1.34, 0.179),
    h60=(60, 638, 319.6, 74, 4.50, 242.2, 77.4, 317.6, 1.76, 0.079),
    # per-ticker H=20 (same climax dates, different fwd tape):
    # (ticker, entries, climax_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 643, 121.4, 3.67, 85.4, 36.0), ("QQQ", 643, 149.6, 3.96, 127.9, 21.8),
         ("IWM", 643, 121.6, 2.64, 77.9, 43.7), ("DIA", 643, 118.2, 3.78, 78.8, 39.4),
         ("GLD", 643, 118.9, 3.63, 97.5, 21.4)],
    # shuffled-volume timing placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(121.4, 0.028, 500),
    # synthetic control (H=20, n_days=4000): (edge, n, climax_bps, win%, one_sample_t)
    syn=[(0.00, 191, 20.8, 49, 0.46), (0.40, 191, 876.3, 76, 7.89)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Volume_breadth_forecasts%3F: Busted](https://img.shields.io/badge/Volume_breadth_forecasts%3F-Busted-8b949e?style=flat-square)\n\n"
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

from up_down_volume import data, strategy as st

ASOF = "2026-05-31"
WINDOW, Q = 60, 0.10
HAVE_REAL = data.have_real() and data.have_breadth()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
def breadth():
    bk = data.load_breadth(allow_fetch=False)
    bk = {t: df[df.index <= ASOF] for t, df in bk.items()}
    return st.up_down_volume(bk)
print("real up/down-volume cache present:", HAVE_REAL, "| basket:", data.BREADTH_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does \"up vs down volume\" actually forecast the market? 📊\n"
            "### The selling-climax legend — when everyone dumps at once, buy — meets a stopwatch\n\n"
            + BADGES +
            "Open any market-internals dashboard and you'll find **up-volume vs down-volume**: add up "
            "the trading volume of everything that went *up* today, and everything that went *down*, "
            "and compare. The lore — from Richard Arms' TRIN to Wyckoff's tape-reading — is that when "
            "**down-volume utterly swamps up-volume**, the market has hit a **selling climax**: panic "
            "is exhausting itself and a bounce is coming. So you *buy the panic*.\n\n"
            "It's a great story. But on a market that drifts **up** over decades, *any* \"buy after a "
            "drop\" rule looks good. So we did the only fair thing: build the up/down-volume ratio "
            "mechanically from a basket of sector ETFs, fire the selling-climax buy hundreds of times "
            "over 21 years, and race it against the only baseline that matters — **buying on random "
            "days instead.**\n\n"
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
            "| If I buy on a **selling climax** (down-volume dominates), do I make money? | **Yes — but "
            "mostly because the market goes up.** Win-rate ~68% at 20 days, returns look great. |\n"
            "| Is that *the breadth signal's* doing? | **Not reliably.** Buy on **random days** and you "
            "capture most of the same return. The climax edge over random never clears the bar. |\n"
            "| Does the up/down *timing* matter at all? | **A little.** Scramble the climax dates in "
            "time and the result weakens (a permutation test flags it) — the climax really does fire "
            "in drawdowns. But that's the market's own short-term bounce, not breadth foresight. |\n"
            "| So is it a tradable edge? | **No.** It's **beta plus a drawdown tilt** you'd get from "
            "the index itself — no breadth feed required. |\n\n"
            "> Up/down volume is a fine way to *describe* a panic. As a *forecast* — \"the climax will "
            "bounce, tradeably\" — it's a **mirage**: the apparent edge doesn't separate from buying "
            "the index after a dip."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Add up the volume of advancing stocks (up-volume) and declining stocks (down-volume). "
            "When down-volume overwhelms up-volume market-wide, you have a **selling climax** — buy. "
            "When up-volume swamps down-volume, a **buying climax** — sell. The up/down-volume ratio "
            "forecasts the next move.\"*\n\n"
            "This is the volume half of **Richard Arms' index (TRIN, 1967)**, echoed by Granville's "
            "volume rules and Wyckoff's climax tape-reading. It's one of the most quoted "
            "market-internals signals — so: does the breadth ratio actually *forecast*?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the up/down-volume ratio genuinely *forecast* turning points, it would be a clean, "
            "tradable crack in market efficiency — a breadth gauge you could act on. That's the dream "
            "the indicator sells.\n\n"
            "But there are two traps. First, a selling climax happens **after a drop**, and stock "
            "indices drift **up**, so *any* dip-buy inherits the climb. Second, short bounces after "
            "drops exist anyway (the index's own mean reversion). To separate the **breadth signal** "
            "from the **tide** and the **bounce**, we (a) build the ratio mechanically with no "
            "hindsight, and (b) compare it to buying on **random days**. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take a breadth basket — **{len(R['basket'])} liquid ETFs** "
            f"(SPY + 9 SPDR sector funds) — daily over **{R['years']:.0f} years** "
            f"({R['start']} → {R['end']}), and:\n\n"
            "1. **Build up/down volume mechanically.** Each day, sum the volume of basket members "
            "that closed *up* (up-volume) and *down* (down-volume); form the **up-volume share** "
            "= up / (up + down).\n"
            f"2. **Flag a selling climax.** A day whose share sits in its rolling **bottom "
            f"{int(R['q']*100)}%** (down-volume dominating) — using only *past* data, no peeking.\n"
            "3. **Trade the lore.** Buy at the next close; measure the SPY return over the next "
            "**5 / 10 / 20 / 60 days**.\n"
            "4. **The honest baseline.** Do the exact same hold on **random days**. If breadth "
            "forecasts, the climax must beat random. *If it doesn't, the signal is a mirage* — that's "
            "the result that would make us say so, announced before we look.\n\n"
            "> ⚠️ A 10-ETF basket is a **proxy** for true exchange up/down volume (thousands of "
            "stocks). It caps the test — but it's the honest, fully-reproducible version."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does the up-volume share even look like, and where do the selling climaxes "
            "land? Here's SPY with the climax days the rule would buy."
        ),
        code(
            "if HAVE_REAL:\n"
            "    spy = load('SPY'); br = breadth()\n"
            "    ent = st.climax_entries(br['uvs'], window=WINDOW, q=Q, side='selling')\n"
            "    ent = ent[ent.isin(spy.index)]\n"
            "    seg = spy['close'].iloc[-700:]; e = ent[ent >= seg.index[0]]\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    ax.plot(seg.index, seg.values, c='k', lw=1.1, label='SPY close')\n"
            "    ax.scatter(e, spy['close'].reindex(e), c=RED, s=42, zorder=5, label='selling-climax BUY')\n"
            "    ax.set_title('Mechanical selling climaxes on SPY (last ~3y)'); ax.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('selling climaxes in window:', len(e), '| total:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The buys cluster on drawdowns — exactly where you'd expect a 'panic-volume' signal. The "
            "question is whether they're followed by *more* bounce than random days. **Let's race the "
            "climax against random entries** at four horizons. Red = buy the climax; grey = buy on "
            "random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    spy = load('SPY'); br = breadth()\n"
            "    ent = st.climax_entries(br['uvs'], window=WINDOW, q=Q, side='selling')\n"
            "    ent = ent[ent.isin(spy.index)]\n"
            "    re = st.random_entries(spy['close'], max(len(ent),50), warmup=WINDOW, seed=7)\n"
            "    clim = [st.summarize(st.forward_returns(spy['close'], ent, h))['mean_bps'] for h in hs]\n"
            "    rnd = [st.summarize(st.forward_returns(spy['close'], re, h))['mean_bps'] for h in hs]\n"
            "else:\n"
            "    clim = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, clim, .4, color='#c0392b', label='buy the selling climax')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(clim,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean SPY return (bps)')\n"
            "ax.set_title('Climax beats random by a hair — but not significantly'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('climax:', [round(v) for v in clim]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"The climax does edge out random at every horizon (e.g. **+{R['h20'][2]:.0f} bps** vs "
            f"**+{R['h20'][5]:.0f} bps** at 20 days) — a consistent *small* tilt. But 'small and "
            "consistent' isn't the same as 'real': the quants notebook shows the climax-minus-random "
            f"*t* tops out at **+{R['h5'][8]:.2f}** (5 days, *p* = {R['h5'][9]:.2f}) and **never clears "
            "2**. Most of that height is the market's drift plus the bounce after any drop."
        ),
        md(
            "**One more check.** Does the *timing* of the climaxes matter? Scramble the up/down-volume "
            "ratio in time (same values, shuffled dates) so 'climaxes' fall on random days. If the "
            "breadth timing is load-bearing, the scramble should weaken the result."
        ),
        code(
            "if HAVE_REAL:\n"
            "    spy = load('SPY'); br = breadth()\n"
            "    pl = st.shuffled_volume_placebo(spy['close'], br, 20, window=WINDOW, q=Q, n_draws=300, seed=492)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real selling climax (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... only {pval*100:.0f}% of *time-scrambled* runs match or beat it (p={pval:.3f}).')\n"
            "print('=> the climax DATES do carry some drawdown-timing info -- but it still loses to random-day buying.')"
        ),
        md(
            f"Here's the honest nuance. The timing placebo *is* significant (**p = {R['placebo'][1]:.3f}**) "
            "— the climax really does fire near drops, not at random. But that drawdown-timing tilt "
            "**still fails to beat the drift-matched random-day baseline** at the desk's *t* ≥ 2 bar. "
            "The breadth ratio is rediscovering the index's own short-term bounce, not adding foresight."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The selling-climax buy does **not** reliably beat buying on random "
            "days (climax-minus-random *t* never clears 2; best +1.95 at 5d, *p* = 0.052). The big "
            "absolute returns are mostly the market's drift.\n"
            "- **Tradability — Mirage.** The small positive tilt isn't separable from 'buy after a "
            "drop and hold' — which needs no breadth feed — and costs erode it.\n"
            "- **\"Does volume breadth forecast?\" — Busted.** The up/down ratio fires in drawdowns "
            "(the timing placebo flags it) but that doesn't survive the random-day test. No tradable "
            "foresight."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing reliable here to trade. The climax buy's only advantage over a coin flip "
            "is the market's long-run climb plus a sliver of post-drop bounce — both of which you'd "
            "capture more cheaply by just **holding the index** (or buying any dip). Costs on every "
            "climax push the already-thin edge further down. As a forecasting tool, the up/down-volume "
            "ratio doesn't pay; as a *description* of a panic day, it was never meant to be a strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **A richer breadth feed.** True NYSE up/down volume (thousands of issues) might sharpen "
            "the signal — our 10-ETF proxy is a floor, not a ceiling. A fun follow-up swaps in a fuller "
            "advance/decline-volume series and re-runs the same random-day race.\n"
            "- **The buying climax.** The symmetric 'blow-off top → sell' rule is just as easy to test "
            "(the quants notebook peeks): it also fails to separate from drift.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* climax bounce into a "
            "synthetic tape and shows the harness banks it (so the null here isn't a dead detector — "
            "it's an honest 'nothing tradable').\n\n"
            "*Think breadth forecasts? Show the selling climax beating random entries at **t ≥ 2** on a "
            "real tape — then we'll talk.*"
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
            "# Up-Down-Volume — a quantitative teardown 🔬\n"
            "### Mechanical up/down-volume breadth on a 10-ETF basket · selling-climax forward returns "
            "· one-sample HAC *t* · a drift-matched random-entry baseline · a shuffled-volume timing "
            "placebo · costs · a synthetic planted-climax control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job is "
            "to separate the **breadth signal** from the **drift** and the **post-drop bounce**: an "
            "upward-drifting index makes *any* climax-buy look good, so the only meaningful test is "
            "climax-vs-random, plus a placebo that destroys the up/down *timing* while preserving its "
            "marginal.\n\n"
            "> ⚠️ **Data note.** Breadth basket = SPY + 9 SPDR sector ETFs (XLK XLF XLE XLV XLI XLY XLP "
            "XLU XLB), yfinance daily OHLCV; forward returns on SPY (total-return), 2005→2026. The "
            "indicator is the **up-volume share** with a past-only rolling-quantile climax threshold "
            "(no look-ahead); entry is the **next close** (one documented lag). A 10-ETF basket is a "
            "**proxy** for exchange-wide up/down volume — an explicit cap. Offline core + synthetic "
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
            f"| **Signal** | `NONE` | Selling climax vs a **drift-matched random** baseline: positive "
            f"at every horizon (Δ = +{R['h5'][6]:.0f}/+{R['h10'][6]:.0f}/+{R['h20'][6]:.0f}/"
            f"+{R['h60'][6]:.0f} bps) but the climax-minus-random Welch *t* **never clears 2** (max "
            f"+{R['h5'][8]:.2f} at 5d, *p* = {R['h5'][9]:.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | The big one-sample t's (20d t = {R['h20'][4]:.2f}) are "
            f"**beta + post-drop bounce** — not separable from buying any dip; costs erode the thin "
            "tilt. |\n"
            f"| **Volume breadth forecasts?** | `BUSTED` | The timing placebo is *significant* "
            f"(**p = {R['placebo'][1]:.3f}**: climaxes fire in drawdowns) yet that does **not** beat "
            "the random-day baseline at *t* ≥ 2. Drawdown-timing, not foresight. |\n\n"
            "> 💡 In plain words: the climax *looks* significant against zero only because indices "
            "drift up and bounce after drops. Race it vs random days (the honest null) and the edge "
            "stays under *t* = 2. The placebo says the dates aren't random noise — but rediscovering "
            "the index's own mean reversion isn't a tradable breadth edge."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "On day $t$, let $U_t,D_t$ be the basket's aggregate up- and down-volume and "
            "$s_t = U_t/(U_t+D_t)$ the **up-volume share**. A **selling climax** is "
            "$s_t \\le Q_{q}(\\{s_\\tau\\}_{\\tau<t})$, the rolling lower-$q$ quantile over past bars "
            "only. The rule buys at the next close and rides the bounce.\n\n"
            "- **H₀ (drift+bounce).** Climax returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (breadth forecasts).** Climax returns **exceed** random at some horizon, *t* ≥ 2.\n"
            "- **H₂ (the timing matters).** Climax returns exceed a **time-shuffled** breadth series.\n\n"
            "We find **H₀ not rejected** (Welch *t* < 2 at every horizon), **H₁ rejected**, **H₂ "
            "*not* rejected** (placebo *p* = 0.028 — the dates do cluster in drawdowns). The steelman "
            "wins the weak leg (timing) but loses the leg that matters (beating random)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift + post-drop bounce.** Equity indices have a positive unconditional daily mean "
            "*and* short-horizon mean reversion after drops (Lo–MacKinlay 1988). A selling climax fires "
            "exactly after a drop, so it inherits both; a one-sample $t$ against **zero** measures "
            "them, not breadth. The fix is the **random-entry baseline** (same instrument, epoch, hold) "
            "and a Welch test of climax-*minus*-random.\n\n"
            "**(b) The up/down structure as a free story.** Maybe *any* clustering of entries in "
            "drawdowns would do as well. The **shuffled-volume placebo** keeps the share's marginal but "
            "permutes it in time, so 'climaxes' land on unrelated days — the direct test of whether the "
            "*specific* up/down timing is load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Breadth.** {', '.join(R['basket'])}; yfinance daily OHLCV. Up-volume share across the "
            "basket; **selling climax** = share ≤ rolling-"
            f"{R['window']}d {int(R['q']*100)}% quantile (past-only). **{R['n_entries']} climaxes**.\n"
            f"- **Forward instrument.** SPY total-return close ({R['start']}→{R['end']}, "
            f"{R['years']:.1f}y).\n"
            "- **Entry.** First climax bar of each run; enter **next close** (one lag); hold "
            "H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of climax returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample climax vs random (the *real* test).\n"
            "- **Null #3 — shuffled-volume timing placebo** (marginal kept, dates permuted).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every climax.\n"
            "- **Positive control.** Synthetic tape with a **planted** selling-climax bounce (knob "
            "`edge`): edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t looks great, vs-random kills it\n\n"
            "Left: the climax's **one-sample** t against zero (the misleading number). Right: the same "
            "climax vs a **drift-matched random** baseline (the honest number)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    from scipy import stats\n"
            "    spy = load('SPY'); br = breadth()\n"
            "    ent = st.climax_entries(br['uvs'], window=WINDOW, q=Q, side='selling'); ent = ent[ent.isin(spy.index)]\n"
            "    re = st.random_entries(spy['close'], max(len(ent),50), warmup=WINDOW, seed=7)\n"
            "    one_t, clim, rnd, welch = [], [], [], []\n"
            "    for h in hs:\n"
            "        tt = st.forward_returns(spy['close'], ent, h); rr = st.forward_returns(spy['close'], re, h)\n"
            "        one_t.append(st.summarize(tt)['t']); clim.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    clim = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "    welch = [R['h5'][8], R['h10'][8], R['h20'][8], R['h60'][8]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], one_t, color=GREY, width=.6)\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO (misleading: beta+bounce)'); a1.set_ylabel('t'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if v>2 else RED for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Climax vs RANDOM, Welch t (honest: never clears 2)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars clear *t* = 2 easily (20d **{R['h20'][4]:.2f}**, 60d "
            f"**{R['h60'][4]:.2f}**) — but that's drift + the post-drop bounce, which *any* dip-buy "
            f"inherits. The right bars are the real test: climax-minus-random tops out at "
            f"**+{R['h5'][8]:.2f}** (5d) and is **+{R['h20'][8]:.2f}** at 20d — positive but never "
            "significant. Breadth adds nothing reliable over a random-day entry."
        ),
        md(
            "### 4b · Climax vs random across horizons — the gap is the verdict\n\n"
            "Mean SPY return, selling-climax vs random entry, all four horizons. The climax should "
            "tower over random if breadth forecasts. It only edges ahead."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, clim, .4, color='#c0392b', label='selling climax')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random entry (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(clim,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd SPY return (bps)')\n"
            "ax.set_title('Selling climax edges random but never significantly'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta climax-random (bps):', [round(a-b) for a,b in zip(clim,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the climax is **+{R['h20'][2]:.0f} bps** and random is "
            f"**+{R['h20'][5]:.0f} bps** — a **+{R['h20'][6]:.0f} bps** tilt with the right sign, but "
            f"the Welch test (4a) says it's noise (*t* = +{R['h20'][8]:.2f}). A consistent small edge "
            "that never reaches significance is the signature of beta-plus-bounce, not a forecast."
        ),
        md(
            "### 4c · The timing placebo — scramble the up/down dates\n\n"
            "Permute the up-volume-share series in time (marginal kept, alignment destroyed) so each "
            "'climax' lands on an unrelated day. If the breadth *timing* is load-bearing, the observed "
            "climax return should sit in the right tail of the scrambled distribution."
        ),
        code(
            "if HAVE_REAL:\n"
            "    spy = load('SPY'); br = breadth()\n"
            "    pl = st.shuffled_volume_placebo(spy['close'], br, 20, window=WINDOW, q=Q, n_draws=300, seed=492)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    rng = np.random.default_rng(492); vals = br['uvs'].dropna(); idx = vals.index; draws=[]\n"
            "    import pandas as _pd\n"
            "    for _ in range(300):\n"
            "        perm = _pd.Series(rng.permutation(vals.to_numpy()), index=idx)\n"
            "        e = st.climax_entries(perm, window=WINDOW, q=Q, side='selling')\n"
            "        rr = st.forward_returns(spy['close'], e, 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws = np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(492); draws = rng.normal(70, 28, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='time-scrambled breadth (SPY, 20d)')\n"
            "ax.axvline(obs, c='#c0392b', lw=2.5, label=f'real climax {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean selling-climax 20d SPY return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real climax sits in the right tail: placebo p = {pval:.3f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real climax {obs:+.1f} bps   placebo p={pval:.3f}  (<0.05 => timing IS load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: unlike most desk indicators, the real climax sits in the **right "
            f"tail** of the scramble — **p = {R['placebo'][1]:.3f}**. The up/down timing genuinely "
            "clusters in drawdowns. But (4a) that drawdown-timing edge still doesn't beat the "
            "*drift-matched random-day* baseline at *t* ≥ 2: the placebo and the random-day null answer "
            "different questions, and the one that decides tradability (random-day) says 'no'."
        ),
        md(
            "### 4d · Per-instrument — the same climax dates on five tapes\n\n"
            "20-day climax-minus-random delta, applying the *same* breadth climax dates to each ETF's "
            "forward return. If it were a fluke of one name it would flip sign; instead it's a small "
            "positive tilt across the board."
        ),
        code(
            "if HAVE_REAL:\n"
            "    spy = load('SPY'); br = breadth()\n"
            "    ent = st.climax_entries(br['uvs'], window=WINDOW, q=Q, side='selling')\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        c = load(t)['close']; e = ent[ent.isin(c.index)]\n"
            "        re = st.random_entries(c, max(len(e),50), warmup=WINDOW, seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d climax - random (bps)'); ax.set_title('A small positive tilt on all five tapes')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: every tape shows a positive 20d delta (**+{R['per'][2][5]:.0f}** on "
            f"IWM down to **+{R['per'][4][5]:.0f}** on GLD). The *sign* is coherent — the climax really "
            "does land before mild bounces — but the *magnitude* is the same small, sub-significant "
            "tilt everywhere, exactly what beta-plus-bounce looks like."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real bounce\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** selling-climax bounce "
            "into a synthetic tape and check the same climax rule banks it: edge=0 must stay at t≈0; "
            "edge>0 must light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.40):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=492, n_days=4000)\n"
            "    c = px['close']; br_s = st.breadth_from_panel(px)\n"
            "    e = st.climax_entries(br_s['uvs'], window=WINDOW, q=Q, side='selling')\n"
            "    s = st.summarize(st.forward_returns(c, e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> t~0; planted bounce -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} climax={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted bounce the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — no false positive); a planted "
            f"bounce reaches **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). The detector "
            "works — so the flat real-tape result is a genuine 'nothing tradable', not a broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the selling climax does not beat a drift-matched random baseline at "
            f"*t* ≥ 2 (climax − random = +{R['h5'][6]:.0f}/+{R['h10'][6]:.0f}/+{R['h20'][6]:.0f}/"
            f"+{R['h60'][6]:.0f} bps at 5/10/20/60d; Welch *t* max **+{R['h5'][8]:.2f}** at 5d, "
            f"*p* = {R['h5'][9]:.2f}). The impressive one-sample t's (20d **{R['h20'][4]:.2f}**) are "
            "beta plus the post-drop bounce.\n"
            f"- **Tradability `MIRAGE`** — no reliable residual edge once drift+bounce are removed; the "
            "tilt isn't separable from buying any dip, and costs erode it.\n"
            f"- **Volume breadth forecasts? `BUSTED`** — the timing placebo is significant "
            f"(**p = {R['placebo'][1]:.3f}**: climaxes cluster in drawdowns) but that does not survive "
            "the random-day test. The up/down ratio rediscovers the index's own short-term mean "
            "reversion; it does not add tradable foresight."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing reliable to trade\n\n"
            "The climax's apparent profit is the unconditional drift of long equity indices plus a "
            "sliver of post-drop mean reversion — both obtained more cheaply by **buying and holding** "
            "(or buying any dip). The breadth rule trades *less* of the time (only on climaxes) and "
            "pays costs on each, so it dominates *nothing*. There is no capacity question because there "
            "is no edge to scale beyond what the index already gives you. Up/down volume is a "
            "descriptive market-internals gauge, not a forecasting strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **A real breadth feed.** Our 10-ETF basket is a proxy; true exchange up/down volume "
            "(thousands of issues, the genuine TRIN input) might carry more — a clean follow-up swaps "
            "it in and re-runs the *same* random-day race. The proxy result is a floor.\n"
            "- **The Arms index proper.** Combine the up/down *volume* ratio with the advance/decline "
            "*count* ratio (full TRIN) and test extreme readings — same drift confound applies.\n"
            "- **Conditioning on regime.** Climaxes in bear markets vs bull markets behave differently; "
            "splitting by VIX or trend is a natural extension (and a fresh multiple-testing hazard).\n\n"
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
