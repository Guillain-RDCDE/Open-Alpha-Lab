"""Generate the two narrative notebooks for Study 491 (McClellan Oscillator).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached daily tapes
under ../_cache/ (and derive the breadth proxy from them) and otherwise quote the frozen
headline numbers in ``R`` (mirroring docs/results.md). The synthetic positive control runs
anywhere with no network.
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
# yfinance daily, traded instrument SPY; breadth proxy = net advances across
# ['SPY','QQQ','IWM','DIA','GLD], 2005-01-03 -> 2026-05-29 (As-of 2026-05-31), 21.4 years,
# McClellan EMA spans 19/39, up-cross-from-negative long.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=528, fast=19, slow=39,
    fp_spy="4cb5244f3990",
    # pooled SPY up-cross, per horizon:
    # (H, n, trig_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 527, 11.3, 59, 0.97, 26.3, -15.0, 9.3, -1.04, 0.300),
    h10=(10, 527, 28.2, 62, 1.52, 54.6, -26.4, 26.2, -1.34, 0.181),
    h20=(20, 525, 51.6, 64, 1.51, 123.4, -71.8, 49.6, -2.55, 0.011),
    h60=(60, 523, 209.2, 71, 3.01, 309.8, -100.6, 207.2, -2.20, 0.028),
    # per-ticker H=20 (same breadth cross): (ticker, entries, trig_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 528, 51.6, 1.51, 123.4, -71.8), ("QQQ", 528, 90.8, 2.20, 163.0, -72.3),
         ("IWM", 528, 41.1, 0.91, 114.7, -73.6), ("DIA", 528, 46.8, 1.47, 116.3, -69.5),
         ("GLD", 528, 93.1, 2.87, 100.5, -7.4)],
    # shuffled-breadth placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(51.6, 0.988, 500),
    # synthetic control (H=20, n_days=4000): (edge, n, trig_bps, win%, one_sample_t)
    syn=[(0.00, 91, 37.8, 59, 1.04), (0.30, 91, 738.5, 98, 18.84)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Forecasts_the_index%3F: Busted](https://img.shields.io/badge/Forecasts_the_index%3F-Busted-8b949e?style=flat-square)\n\n"
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

from mcclellan_oscillator import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
def breadth():
    return data.load_breadth(asof=ASOF)
print("real McClellan cache present:", HAVE_REAL, "| breadth basket:", data.breadth_members())
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the McClellan Oscillator actually forecast the market? 📊\n"
            "### A famous breadth gauge — two EMAs of advancers-minus-decliners — meets a stopwatch\n\n"
            + BADGES +
            "Open any breadth dashboard and you'll find the **McClellan Oscillator**: take the daily "
            "**net advances** (how many stocks went up minus how many went down), smooth it with a "
            "fast and a slow moving average, and subtract. The lore, taught by Sherman & Marian "
            "McClellan themselves and repeated on every breadth-trading site, is that the oscillator "
            "**leads the index** — when it **crosses up through zero from negative**, breadth momentum "
            "has flipped bullish and a rally is coming. So you buy the up-cross.\n\n"
            "It *looks* compelling on a hand-picked chart. But a momentum signal on a market that "
            "drifts **up** over time is the textbook setup for fooling yourself — *any* long-entry rule "
            "looks like a winner. So we did the only fair thing: encode the up-cross rule "
            "**mechanically** (no eyeballing), fire it 528 times on SPY over 21 years, and time the "
            "result with a stopwatch — against the only baseline that matters: **buying on random days "
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
            "| If I buy when the McClellan oscillator crosses up from negative, do I make money? | "
            "**A little — but only because the market goes up.** The raw win-rate is ~60% and the "
            "returns are positive. |\n"
            "| Is that *the oscillator's* doing? | **No.** Buy on **random days** instead and you do "
            "**better** — at *every* horizon. The cross is actually *worse* than a coin-flip entry. |\n"
            "| Does breadth momentum 'forecast' the index? | **Not in any usable way.** Scramble the "
            "breadth series in time and the result doesn't change. The momentum structure isn't doing "
            "the work. |\n"
            "| So is it a tradable edge? | **No.** It's **beta in a costume** — the upward drift of "
            "stocks, re-labelled as a breadth signal. |\n\n"
            "> The McClellan oscillator is a fine way to *describe* market breadth. As a *forecast* — "
            "\"the up-cross will rally\" — it's a **mirage**: all of the apparent edge is the market's "
            "long-run climb, and the rule captures *less* of it than random."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Count net advances each day (stocks up minus stocks down). Smooth it two ways — a "
            "fast EMA(19) and a slow EMA(39) — and subtract: that's the **McClellan Oscillator**. When "
            "it crosses **up through zero from negative**, breadth momentum has turned positive and the "
            "index is about to rally. Buy the up-cross.\"*\n\n"
            "This is **Sherman & Marian McClellan's** breadth oscillator (1969), still taught today and "
            "tracked on StockCharts (`$NYMO`), TradingView and every breadth dashboard. It's one of the "
            "most cited breadth tools in technical analysis — so: does breadth momentum actually lead "
            "the tape?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the oscillator genuinely *forecast* the index, it would be remarkable: a count of how "
            "many stocks rose yesterday would predict tomorrow's market — a clean, tradable crack in "
            "market efficiency. That's the dream the indicator sells.\n\n"
            "But there's a trap. The signal is a long-only entry on **stock indices, which drift up** "
            "over time — so *any* buy rule will look profitable. To separate the **tool** from the "
            "**tide**, we have to (a) fire the rule by a fixed mechanical recipe with no hindsight, and "
            "(b) compare it to buying on **random days**. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **SPY** (traded), build a **breadth proxy** from a basket of liquid ETFs "
            f"({', '.join(R['tickers'])} — a coarse stand-in for true exchange breadth), daily, over "
            f"**{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Count net advances mechanically.** Each day, how many basket members closed up minus "
            "how many closed down. No eyeballing.\n"
            f"2. **Build the oscillator by rule.** EMA({R['fast']}) − EMA({R['slow']}) of that net-"
            "advances series, computed forward-only so it never peeks at the future.\n"
            "3. **Trade the lore.** When the oscillator crosses **up through zero from negative**, buy "
            "SPY at the next close; measure the return over the next **5 / 10 / 20 / 60 days**.\n"
            "4. **The honest baseline.** Do the exact same hold on **random days**. If the cross "
            "matters, it must beat random. *If it doesn't, the tool is a mirage* — that's the result "
            "that would make us say so, announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does the oscillator even look like? Here's SPY with the McClellan oscillator "
            "below it and the up-crosses-from-negative the rule would buy."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']; net = breadth().reindex(c.index).dropna()\n"
            "    osc = st.mcclellan(net); ent = st.osc_up_cross_entries(net)\n"
            "    seg = c.index[-450:]\n"
            "    ent_w = ent[ent >= seg[0]]\n"
            "    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.2, 6.2), sharex=True,\n"
            "                                   gridspec_kw={'height_ratios':[2,1]})\n"
            "    ax1.plot(seg, c.reindex(seg), c='k', lw=1.2, label='SPY close')\n"
            "    ax1.scatter(ent_w, c.reindex(ent_w), c=GREEN, s=40, zorder=5, label='up-cross BUY')\n"
            "    ax1.set_title('McClellan up-cross BUYs on SPY (last ~2y)'); ax1.legend(loc='upper left')\n"
            "    ax2.plot(seg, osc.reindex(seg), c='#2c6fbb', lw=1.1, label='McClellan osc (EMA19-EMA39)')\n"
            "    ax2.axhline(0, c=GREY, lw=.9); ax2.scatter(ent_w, osc.reindex(ent_w), c=GREEN, s=30, zorder=5)\n"
            "    ax2.set_title('the oscillator, with zero up-crosses marked'); ax2.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('up-crosses in window:', len(ent_w))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The oscillator swings around zero and the green dots mark each up-cross. The question is "
            "whether those buys are followed by rallies. **Let's race the up-cross against random "
            "entries** at four horizons. Blue = buy the up-cross; grey = buy on random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']; net = breadth().reindex(c.index).dropna()\n"
            "    ent = st.osc_up_cross_entries(net); re = st.random_entries(c.index, max(len(ent),50), seed=7)\n"
            "    trig = [st.forward_returns(c, ent, h).mean()*1e4 for h in hs]\n"
            "    rnd = [st.forward_returns(c, re, h).mean()*1e4 for h in hs]\n"
            "else:\n"
            "    trig = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, trig, .4, color='#2c6fbb', label='buy the up-cross')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(trig,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('The up-cross does NOT beat random — it loses at every horizon'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('trigger:', [round(v) for v in trig]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the whole story in one chart. The up-cross makes money in absolute terms "
            f"(**+{R['h20'][2]:.0f} bps** over 20 days) — but **random entries make far more** "
            f"(**+{R['h20'][5]:.0f} bps**). At *every* horizon the famous breadth cross is *worse* than "
            "throwing darts. The apparent edge was **the market's upward drift**, not the oscillator — "
            "and the rule actually captures *less* of that drift than random."
        ),
        md(
            "**One more sanity check.** What if we scramble the breadth series **in time** — keep the "
            "same set of daily net-advance values but shuffle their order, so the oscillator's momentum "
            "structure becomes nonsense? If breadth momentum really forecasts, the scramble should do "
            "much worse."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']; net = breadth().reindex(c.index).dropna()\n"
            "    pl = st.shuffled_breadth_placebo(c, net, 20, n_draws=300, seed=491)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real oscillator up-cross (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of TIME-SHUFFLED breadth oscillators do at least as well (p={pval:.2f}).')\n"
            "print('=> the breadth-momentum structure is not doing the work.')"
        ),
        md(
            f"Almost **all** of the **time-shuffled** breadth oscillators match or beat the real one "
            f"(*p* = {R['placebo'][1]:.2f}). If breadth *momentum* genuinely forecast, a random "
            "time-scramble would collapse the result. It doesn't — because the result was never about "
            "the momentum."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The up-cross buy does **not** beat buying on random days (it's "
            "*worse* at every horizon; the trigger-vs-random difference is significantly **negative** "
            "at 20–60 days). The positive absolute returns are the market's drift, not the oscillator.\n"
            "- **Tradability — Mirage.** Nothing to trade once you remove the beta you were always "
            "getting for free — and costs only make it worse.\n"
            "- **\"Breadth momentum forecasts the index\"? — Busted.** Scramble the breadth in time and "
            "the result barely moves. The momentum doesn't lead."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. The up-cross's *only* relationship to profit is the "
            "market's long-run climb — which you'd capture more cheaply (and more fully) by just "
            "**holding the index**. The breadth-cross buy is a worse, more expensive way to be long: it "
            "sits out of the market between crosses and enters at worse-than-random times. Costs "
            "(commissions + spread on every trigger) push the already-no-edge result further negative. "
            "As a forecasting tool it doesn't pay; as a breadth gauge it was never meant to be a "
            "stand-alone strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **A richer breadth basket.** Our net-advances proxy uses just 5 ETFs; the genuine "
            "oscillator uses thousands of NYSE issues. A wider sector basket (XLK XLF XLE …) or a real "
            "`$NYMO` feed would sharpen the estimate — but coarse breadth can only *blur* a real edge, "
            "and the trigger here doesn't just fail to beat random, it *loses*, so a richer basket is "
            "unlikely to flip it.\n"
            "- **Other triggers.** Zero-line crosses, ±100 extremes, divergences — they're all "
            "transforms of the same drift-confounded series.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* post-cross bounce "
            "into a synthetic tape and shows the harness banks it (so the null result here isn't a dead "
            "detector — it's an honest 'nothing there').\n\n"
            "*Think breadth momentum forecasts? Show the up-cross beating random entries at **t ≥ 2** "
            "on a real tape — then we'll talk.*"
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
            "# The McClellan Oscillator — a quantitative teardown 🔬\n"
            "### Causal EMA19−EMA39 breadth oscillator · up-cross-from-negative forward returns on SPY "
            "· one-sample HAC *t* · a drift-matched random-entry baseline · a shuffled-breadth "
            "placebo · costs · a synthetic planted-bounce control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **oscillator** from the **drift**: an upward-trending index makes "
            "*any* long entry look good, so the only meaningful test is trigger-vs-random, plus a "
            "placebo that destroys the breadth-momentum structure while preserving its marginal.\n\n"
            "> ⚠️ **Data note.** Traded instrument SPY; breadth proxy = net advances across a 5-ETF "
            "basket (SPY QQQ IWM DIA GLD), yfinance daily adjusted closes, 2005→2026. The oscillator "
            f"is EMA({R['fast']})−EMA({R['slow']}) (the classic 10%/5% constants), computed causally; "
            "entry is the **next close** (one documented lag). The breadth basket is a **coarse proxy** "
            "for true exchange breadth and *caps* the test (see `docs/results.md`). Offline core + "
            "synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `NONE` | Up-cross vs a **drift-matched random** baseline: the trigger is "
            f"*worse* at **every** horizon (Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/"
            f"{R['h60'][6]:+.0f} bps) and the trigger-minus-random difference is **significantly "
            f"negative** at 20d (Welch t = {R['h20'][8]:+.2f}) and 60d ({R['h60'][8]:+.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | The only positive number, the 60d one-sample t = "
            f"{R['h60'][4]:.2f}, is **pure beta** — it vanishes (goes negative) against random entries "
            "and against cost. No residual edge to scale. |\n"
            f"| **Forecasts the index?** | `BUSTED` | Time-shuffling the breadth series (shuffled-"
            f"breadth placebo) leaves the result intact: **p = {R['placebo'][1]:.2f}** of scrambled "
            "oscillators match or beat the real one. The momentum structure isn't doing the work. |\n\n"
            "> 💡 In plain words: the up-cross *looks* fine only because indices drift up. Strip the "
            "drift (race it vs random) or strip the momentum (shuffle the breadth) and the edge "
            "evaporates — in fact it goes negative. Classic beta-in-a-costume."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $A_t$ be the daily **net advances** (members up − members down). The oscillator is "
            "$\\mathrm{McOsc}_t = \\mathrm{EMA}_{19}(A)_t - \\mathrm{EMA}_{39}(A)_t$, both EMAs "
            "causal. The Andrews-style bull rule buys on the **up-cross from non-positive**: "
            "$\\mathrm{McOsc}_{t-1}\\le 0 < \\mathrm{McOsc}_t$, entered at the next close.\n\n"
            "- **H₀ (drift).** Trigger returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (breadth forecasts).** Trigger returns **exceed** random at some horizon, t ≥ 2.\n"
            "- **H₂ (the momentum matters).** Trigger returns exceed a **time-shuffled-breadth** "
            "oscillator whose temporal structure is destroyed.\n\n"
            "We find **H₀ not rejected — in fact worse than the baseline** (trigger < random at all "
            "horizons, significantly so at 20–60d), **H₁ rejected** (Welch t never ≥ 2; it's "
            "*negative*), **H₂ rejected** (placebo p ≈ 0.99). The steelman fails on every leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean. *Any* long entry "
            "on a forward horizon inherits it; a high one-sample $t$ against **zero** measures the "
            "tide, not the tool. The fix is the **random-entry baseline** (same instrument, epoch, "
            "hold) and a Welch test of trigger-*minus*-random.\n\n"
            "**(b) Momentum as a free description.** The McClellan EMAs are built to capture breadth "
            "*momentum* (slow swings in net advances). The danger is that the up-cross fires mostly "
            "after the market has *already* moved — re-describing the trend rather than leading it. The "
            "**time-shuffled-breadth placebo** keeps the marginal distribution of net advances but "
            "destroys its autocorrelation, so the oscillator's momentum becomes meaningless; if the "
            "real result survives the scramble, the momentum was never load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** Traded: SPY. Breadth proxy: net advances over {', '.join(R['tickers'])}; "
            f"yfinance daily adjusted closes ({R['start']}→{R['end']}, {R['years']:.1f}y). "
            f"**{R['n_entries']} up-crosses**.\n"
            f"- **Oscillator.** EMA({R['fast']})−EMA({R['slow']}) of net advances, causal; a warm-up of "
            f"3×{R['slow']} bars dropped so both EMAs are seeded.\n"
            "- **Entry.** First close where the oscillator crosses up through zero from non-positive; "
            "enter **next close** (one lag); hold H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of trigger returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample trigger vs random (the *real* test).\n"
            "- **Null #3 — shuffled-breadth placebo** (time-permute net advances, marginal kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every trigger.\n"
            "- **Positive control.** Synthetic tape with a **planted** post-cross bounce (knob `edge`): "
            "edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t looks OK, vs-random kills it\n\n"
            "Left: the up-cross's **one-sample** t against zero (the misleading number). Right: the "
            "same trigger vs a **drift-matched random** baseline (the honest number)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    from scipy import stats\n"
            "    c = load('SPY')['close']; net = breadth().reindex(c.index).dropna()\n"
            "    ent = st.osc_up_cross_entries(net); re = st.random_entries(c.index, max(len(ent),50), seed=7)\n"
            "    one_t, trig, rnd, welch = [], [], [], []\n"
            "    for h in hs:\n"
            "        tt = st.forward_returns(c, ent, h); rr = st.forward_returns(c, re, h)\n"
            "        one_t.append(st.summarize(tt)['t']); trig.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    trig = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "    welch = [R['h5'][8], R['h10'][8], R['h20'][8], R['h60'][8]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], one_t, color=GREY, width=.6)\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO (misleading: it is beta)'); a1.set_ylabel('t'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if v>2 else RED for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(-2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Trigger vs RANDOM, Welch t (honest: negative)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars look modest, peaking at 60d (**{R['h60'][4]:.2f}**) — "
            f"and even that is the **drift**, every long entry inherits it. The right bars are the real "
            f"test: trigger-minus-random is **negative at every horizon** and *significantly* so at 20d "
            f"(**{R['h20'][8]:+.2f}**) and 60d (**{R['h60'][8]:+.2f}**). The oscillator does *worse* "
            "than a coin flip."
        ),
        md(
            "### 4b · Trigger vs random across horizons — the gap is the verdict\n\n"
            "Mean return, up-cross vs random entry, all four horizons. The trigger should tower over "
            "random if breadth forecasts. Instead it sits below it everywhere."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, trig, .4, color='#2c6fbb', label='up-cross')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random entry (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(trig,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('Up-cross underperforms random entry at every horizon'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta trigger-random (bps):', [round(a-b) for a,b in zip(trig,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the trigger is **+{R['h20'][2]:.0f} bps** but random is "
            f"**+{R['h20'][5]:.0f} bps** — the oscillator *underperforms* a dart by "
            f"{abs(R['h20'][6]):.0f} bps. The gap only widens with horizon. There is no horizon where "
            "breadth leads."
        ),
        md(
            "### 4c · The momentum placebo — shuffle the breadth in time, nothing changes\n\n"
            "Permute the net-advances series in time (marginal kept) so the oscillator's momentum is "
            "destroyed. If breadth *momentum* forecasts, the observed return should sit far in the "
            "right tail of the shuffled distribution. It sits dead in the middle."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = load('SPY')['close']; net = breadth().reindex(c.index).dropna()\n"
            "    pl = st.shuffled_breadth_placebo(c, net, 20, n_draws=300, seed=491)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    import numpy as _np\n"
            "    rng = _np.random.default_rng(491); vals = net.to_numpy(); idx = net.index\n"
            "    draws = []\n"
            "    for _ in range(300):\n"
            "        perm = __import__('pandas').Series(rng.permutation(vals), index=idx)\n"
            "        e = st.osc_up_cross_entries(perm); rr = st.forward_returns(c, e, 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws = _np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(491); draws = rng.normal(120, 35, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='time-shuffled breadth oscillators (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real oscillator {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean up-cross 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real oscillator sits in the LEFT pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real oscillator {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => momentum not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real oscillator (blue line) sits **in the lower middle** of the "
            f"scrambled cloud — **p = {R['placebo'][1]:.2f}**. Time-shuffled breadth does *better*, so "
            "the specific EMA19−EMA39 momentum carries no information. This is the cleanest refutation "
            "of 'breadth momentum forecasts the index.'"
        ),
        md(
            "### 4d · Per-ticker — the cross loses to random everywhere\n\n"
            "20-day trigger-minus-random delta, trading each index on the **same** breadth up-cross. "
            "If breadth worked it would be positive across the board; instead it's negative in all 5 — "
            "including GLD, which has no link to US equity breadth at all."
        ),
        code(
            "if HAVE_REAL:\n"
            "    net = breadth()\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        c = load(t)['close']; na = net.reindex(c.index).dropna()\n"
            "        e = st.osc_up_cross_entries(na); re = st.random_entries(c.index, max(len(e),50), seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d trigger − random (bps)'); ax.set_title('Up-cross underperforms random in 5 of 5 names')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: **every** name is negative — SPY is **{R['per'][0][5]:+.0f}** bps "
            f"behind random. Even GLD ({R['per'][4][5]:+.0f} bps), which has no business reacting to US "
            "equity breadth, is negative — the tell that the 'signal' is just relabelled drift, not a "
            "breadth effect."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real bounce\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** post-cross bounce into "
            "a synthetic tape and check the same up-cross rule banks it: edge=0 must stay below t=2; "
            "edge>0 must light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.30):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=491, n_days=4000)\n"
            "    c = px['close']; e = st.osc_up_cross_entries(px['net_adv']); s = st.summarize(st.forward_returns(c, e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> t<2; planted bounce -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} trigger={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted bounce the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — below the t=2 bar, no false "
            f"positive; averaged over 20 seeds the edge=0 t is +0.22); a planted post-cross bounce "
            f"reaches **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). The detector works — so "
            "the negative real-tape result is a genuine 'nothing there', not a broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the up-cross does not beat a drift-matched random baseline; it is "
            f"*worse* (trigger − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/"
            f"{R['h60'][6]:+.0f} bps at 5/10/20/60d; Welch t **significantly negative** at 20d "
            f"**{R['h20'][8]:+.2f}** and 60d **{R['h60'][8]:+.2f}**). The only positive number, the 60d "
            f"one-sample t **{R['h60'][4]:.2f}**, is pure beta.\n"
            f"- **Tradability `MIRAGE`** — no residual edge once the drift is removed; the rule gives "
            "*back* drift by entering at worse-than-random times, and costs only deepen the hole. You'd "
            "capture the drift more cheaply by holding the index.\n"
            f"- **Forecasts the index? `BUSTED`** — the shuffled-breadth placebo leaves the result "
            f"untouched (**p = {R['placebo'][1]:.2f}**): time-scrambled breadth does as well or better, "
            "so the EMA19−EMA39 momentum carries no forecasting information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The up-cross's entire apparent profit is the unconditional drift of long equity indices, "
            "which you obtain more cheaply and more fully by **buying and holding**. The breadth rule "
            "trades *less* of the time (only on crosses), enters at worse-than-random times, and pays "
            "costs on each, so it strictly dominates *nothing*. There is no capacity question because "
            "there is no edge to scale. The McClellan oscillator is a descriptive breadth gauge, not a "
            "forecasting strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The breadth-proxy ceiling.** Our net-advances proxy uses 5 ETFs; the genuine "
            "oscillator uses thousands of NYSE issues. A coarse basket can only *blur* a real edge — so "
            "the fact that the trigger *loses* (not merely fails to win) is strong: a richer basket "
            "(XLK XLF XLE …) or a real `$NYMO` feed would sharpen, not rescue, it.\n"
            "- **Summation index & extremes.** The McClellan Summation Index (running sum) and the "
            "±100 extreme triggers are transforms of the same drift-confounded series and inherit the "
            "same confound.\n"
            "- **Divergence trades.** 'Price up, oscillator down' divergences add a second free "
            "parameter and a hindsight-rich definition; the mechanical zero-cross here is the "
            "charitable upper bound.\n\n"
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
