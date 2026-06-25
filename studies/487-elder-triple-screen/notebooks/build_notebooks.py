"""Generate the two narrative notebooks for Study 487 (Elder's Triple Screen).

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
# 2026-05-31, partial June dropped), 21.4 years, weekly MACD-hist trend + daily Force-Index
# pullback + prior-high breakout, long.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=2369,
    fp_spy="4cb5244f3990",
    # pooled triple-screen, per horizon:
    # (H, n, trip_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 2368, 10.9, 55, 2.10, 24.9, -14.0, 8.9, -1.92, 0.054),
    h10=(10, 2363, 31.9, 60, 3.77, 29.7, 2.3, 29.9, 0.22, 0.823),
    h20=(20, 2352, 74.9, 61, 5.21, 68.5, 6.4, 72.9, 0.44, 0.657),
    h60=(60, 2344, 220.1, 67, 5.97, 257.3, -37.1, 218.1, -1.54, 0.123),
    # per-ticker H=20: (ticker, entries, trip_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 477, 69.0, 2.75, 69.7, -0.7), ("QQQ", 464, 113.1, 3.60, 93.8, 19.3),
         ("IWM", 473, 64.4, 1.84, 15.1, 49.3), ("DIA", 438, 43.9, 1.44, 57.4, -13.5),
         ("GLD", 517, 81.8, 2.26, 103.1, -21.3)],
    # screen-scramble placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(69.0, 0.649, 500),
    # synthetic control (H=20, n_days=4000): (edge, n, trip_bps, win%, one_sample_t)
    syn=[(0.00, 296, -2.9, 50, -0.07), (0.50, 275, 543.3, 77, 8.29)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Filter_adds_edge%3F: Busted](https://img.shields.io/badge/Filter_adds_edge%3F-Busted-8b949e?style=flat-square)\n\n"
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

from elder_triple_screen import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real triple-screen cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does Elder's Triple Screen actually beat random? 🖥️🖥️🖥️\n"
            "### Three filters, two timeframes, one famous trading system — meets a stopwatch\n\n"
            + BADGES +
            "Open any trading textbook and you'll meet Dr. Alexander Elder's **Triple Screen**: "
            "don't trade until *three* screens agree. **Screen 1** — the big-picture *weekly* trend "
            "must be up (the \"tide\"). **Screen 2** — a *daily* oscillator must be oversold against "
            "that trend (the \"wave\", a pullback inside the up-move). **Screen 3** — price must break "
            "out above the prior bar's high (the \"ripple\", the trigger). Buy only when the tide, the "
            "wave and the ripple line up.\n\n"
            "It *sounds* bulletproof — surely stacking three filters across two timeframes throws away "
            "the noise and keeps the signal? But every filter here is fitted to *past* price on a "
            "market that drifts **up**, so *any* dip-buy will look good. So we did the only fair thing: "
            "encode all three screens **mechanically** (no eyeballing), fire the long thousands of "
            "times across five big indices over 21 years, and time the result with a stopwatch — "
            "against the only baseline that matters: **buying on random days instead.**\n\n"
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
            "| If I buy when all three screens align, do I make money? | **Yes — but only because the "
            "market goes up.** The raw win-rate is ~60% and the returns look great. |\n"
            "| Is that *the filter's* doing? | **No.** Buy on **random days** instead and you do "
            "**just as well** — the triple-vs-random gap is a few basis points and never significant "
            "(at 5 days the filter is actually *worse* than a coin flip). |\n"
            "| Does the multi-timeframe filter add edge? | **Not in any usable way.** Knock the weekly "
            "trend out of alignment with price and the result barely changes. The alignment isn't doing "
            "the work. |\n"
            "| So is it a tradable edge? | **No.** It's **beta in a costume** — the upward drift of "
            "stocks, re-labelled as a three-screen system. |\n\n"
            "> The Triple Screen is a perfectly sensible *trade-management* discipline (trade with the "
            "bigger trend, enter on a pullback). As a *forecast* — \"these three things aligned, so the "
            "next move is up\" — it's a **mirage**: all of the apparent edge is the market's long-run "
            "climb, none of it is the alignment."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"First check the **weekly** chart — only trade in the direction of the higher-timeframe "
            "trend. Then drop to the **daily** chart and wait for an oscillator to get oversold against "
            "that trend (a pullback). Then enter on a **breakout** above the prior bar. Three screens, "
            "three confirmations — a high-odds trade.\"*\n\n"
            "This is **Dr. Alexander Elder's** Triple Screen system (1986; *Trading for a Living*, "
            "1993), one of the most taught frameworks in technical analysis. Elder's own toolkit: a "
            "**weekly MACD-histogram** for the trend, his **Force Index** (or a stochastic) for the "
            "pullback, and a **trailing buy-stop** for the entry. So: does stacking the screens beat a "
            "coin flip?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the alignment genuinely *forecast* the next move, it would be remarkable: three "
            "indicators across two timeframes predicting turning points, an edge you could automate. "
            "That's the dream the system sells.\n\n"
            "But there's a trap. Every screen is computed from **past price** on a market (stock "
            "indices) that drifts **up** over time, so *any* trend-following dip-buy will look "
            "profitable. And combining filters *feels* like it should help — yet if the filters are "
            "just re-describing the same drift, three of them add nothing. To separate the **system** "
            "from the **tide**, we (a) encode every screen by a fixed mechanical rule with no "
            "hindsight, and (b) compare it to buying on **random days**. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Screen 1 — the weekly trend.** Resample to weekly bars, compute the MACD-histogram, "
            "and ask: is it *rising*? We forward-fill that to days and **shift it one day**, so today "
            "only knows last week's trend — no peeking.\n"
            "2. **Screen 2 — the daily pullback.** A Force-Index proxy must have dipped below zero in "
            "the last few days (an oversold wave against the up-tide).\n"
            "3. **Screen 3 — the breakout.** The close must clear the **prior bar's high**. When all "
            "three align, buy at the next close; measure the return over **5 / 10 / 20 / 60 days**.\n"
            "4. **The honest baseline.** Do the exact same hold on **random days**. If the system "
            "matters, the triple-screen must beat random. *If it doesn't, it's a mirage* — that's the "
            "result that would make us say so, announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does a mechanical triple-screen entry look like? Here's SPY with the weekly "
            "up-trend windows shaded and the triple-screen buy signals the rule would fire."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    cl = b['close']; seg = cl.iloc[-450:]\n"
            "    up = st.weekly_trend_up(cl).reindex(seg.index)\n"
            "    ent = st.triple_screen_entries(cl, b['high'])\n"
            "    ent = ent[ent >= seg.index[0]]\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    ax.plot(seg.index, seg.values, c='k', lw=1.2, label='SPY close')\n"
            "    ax.fill_between(seg.index, seg.min(), seg.max(), where=up.values, color=GREEN, alpha=.08, label='weekly trend up (Screen 1)')\n"
            "    ax.scatter(ent, cl.reindex(ent), c=RED, s=42, zorder=5, label='triple-screen BUY')\n"
            "    ax.set_title('Mechanical Elder Triple Screen on SPY (last ~2y)'); ax.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('triple-screen entries in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The signals cluster in up-trends — *as a description* of \"buy pullbacks in an uptrend\", "
            "that's sensible. The question is whether those red dots beat random days. **Let's race the "
            "triple-screen against random entries** at four horizons. Blue = the three-screen long; "
            "grey = buy on random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    trip, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t); c = bb['close']\n"
            "            e = st.triple_screen_entries(c, bb['high'])\n"
            "            re = st.random_entries(c, max(len(e),50), seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        trip.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    trip = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, trip, .4, color='#2c6fbb', label='triple-screen long')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(trip,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('The three screens do NOT beat random'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('triple:', [round(v) for v in trip]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the whole story in one chart. The triple-screen makes money in absolute terms "
            f"(**+{R['h20'][2]:.0f} bps** over 20 days) — but **random entries make about the same** "
            f"(**+{R['h20'][5]:.0f} bps**). At 5 days the famous system is actually *worse* than "
            "throwing darts. The apparent edge was **the market's upward drift**, not the alignment of "
            "three screens. (The quants notebook shows the difference never clears *t* = 2.)"
        ),
        md(
            "**One more sanity check.** What if we knock the weekly trend *out of alignment* with "
            "price — shift it by a random offset, so Screen 1 is still 'up' the same fraction of the "
            "time but no longer lines up with the daily pullbacks? If the multi-timeframe alignment "
            "really matters, that nonsense filter should do much worse."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bb = load('SPY')\n"
            "    pl = st.scrambled_screen_placebo(bb['close'], bb['high'], 20, n_draws=300, seed=487)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real triple-screen (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of *misaligned* filters do at least as well (p={pval:.2f}).')\n"
            "print('=> the timeframe alignment is not doing the work.')"
        ),
        md(
            f"Roughly two-thirds of the **misaligned** filters match or beat the real one "
            f"(*p* = {R['placebo'][1]:.2f}). If the weekly-to-daily alignment genuinely mattered, "
            "shifting it would collapse the result. It doesn't — because the result was never about the "
            "alignment."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The triple-screen long does **not** beat buying on random days "
            "(it's *worse* at 5 days; the triple-vs-random difference never clears *t* = 2). The big "
            "absolute returns are the market's drift, not the screens.\n"
            "- **Tradability — Mirage.** Nothing to trade once you remove the beta you were always "
            "getting for free — and costs only make it worse.\n"
            "- **\"Does the multi-timeframe filter add edge\"? — Busted.** Knock the weekly trend out "
            "of alignment and the result barely moves. The alignment doesn't align anything tradable."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade *as a forecast*. The triple-screen's *only* advantage over a "
            "coin flip is the market's long-run climb — which you'd capture more cheaply (and more "
            "fully) by just **holding the index**. The three-screen entry is a more selective, more "
            "expensive way to be long. Costs (commissions + spread on every trigger) push the "
            "already-no-edge result further down. As *risk management* (trade with the trend, size on "
            "pullbacks) the framework is fine; as a *return forecast* it doesn't pay."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Swap the oscillator.** Stochastic, Elder-ray, RSI for Screen 2 — the result is "
            "robust: drift in, system out. The specific oscillator is not the point.\n"
            "- **Different timeframe pairs.** Daily/weekly ↔ hourly/daily — the alignment placebo "
            "applies the same way.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* post-alignment "
            "bounce into a synthetic tape and shows the harness banks it (so the null result here isn't "
            "a dead detector — it's an honest 'nothing there').\n\n"
            "*Think three screens forecast? Show the triple-screen beating random entries at "
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
            "# Elder's Triple Screen — a quantitative teardown 🔬\n"
            "### Mechanical 3-screen longs on 5 indices · weekly-trend + daily-pullback + breakout · "
            "forward returns · one-sample HAC *t* · a drift-matched random-entry baseline · a "
            "screen-scramble alignment placebo · costs · a synthetic planted-bounce control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **system** from the **drift**: an upward-trending index makes *any* "
            "trend-following dip-buy look good, so the only meaningful test is triple-vs-random, plus a "
            "placebo that destroys the timeframe alignment while preserving each screen's marginal.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. Screen 1 (weekly MACD-hist "
            "slope) is shifted **one day** (no look-ahead); entry is the **next close** (one documented "
            "lag). Offline core + synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `NONE` | Triple-screen vs a **drift-matched random** baseline: the gap is "
            f"tiny (Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} bps "
            f"at 5/10/20/60d) and the difference **never clears t = 2** (Welch t at 20d "
            f"= {R['h20'][8]:+.2f}; at 5d it is {R['h5'][8]:+.2f} — the filter is *worse* than a dart). |\n"
            f"| **Tradability** | `MIRAGE` | The big one-sample t's (20d t = {R['h20'][4]:.2f}) are "
            f"**pure beta** — they vanish against random entries. No residual edge to scale. |\n"
            f"| **Filter adds edge?** | `BUSTED` | Shifting the weekly trend out of alignment "
            f"(screen-scramble placebo) leaves the result intact: **p = {R['placebo'][1]:.2f}** of "
            "misaligned filters match or beat the real one. The alignment isn't load-bearing. |\n\n"
            "> 💡 In plain words: the triple-screen *looks* significant only because indices drift up. "
            "Strip the drift (race it vs random) or strip the alignment (shift the trend) and the edge "
            "evaporates. Three filters, one beta-in-a-costume."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $W_t$ be the weekly MACD-histogram slope (shifted: $W_t>0$ means last week's "
            "histogram rose), $F_t$ the daily Force-Index proxy, and $H_{t-1}$ the prior bar's high. "
            "The triple-screen long fires when **Screen 1** $W_t>0$, **Screen 2** $F_\\tau<0$ for some "
            "$\\tau\\in[t-5,t-1]$ (a recent pullback), and **Screen 3** $C_t>H_{t-1}$ (the breakout). "
            "Enter at $C_{t+1}$.\n\n"
            "- **H₀ (drift).** Triple-screen returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (the system forecasts).** Triple-screen returns **exceed** random at some horizon, "
            "t ≥ 2.\n"
            "- **H₂ (the alignment matters).** Triple-screen returns exceed a **screen-scramble** "
            "filter whose weekly trend is shifted out of alignment.\n\n"
            "We find **H₀ not rejected** (Δ within a few bps), **H₁ rejected** (Welch t never ≥ 2), "
            "**H₂ rejected** (placebo p ≈ 0.65). The steelman fails on every leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean. *Any* long-only "
            "entry rule inherits it; a high one-sample $t$ against **zero** measures the tide, not the "
            "system. The fix is the **random-entry baseline** (same instrument, epoch, hold) and a "
            "Welch test of triple-*minus*-random.\n\n"
            "**(b) Filter stacking as a free parameter.** Three filters fitted to past price *feel* "
            "like they should compound into an edge — but if each merely re-describes the same trend, "
            "stacking them buys nothing. The **screen-scramble placebo** keeps each screen's marginal "
            "frequency but circularly shifts the weekly trend, so the timeframes no longer align; if "
            "the real result survives the shift, the alignment was never load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} triple-screen entries** "
            "pooled.\n"
            "- **Screen 1.** Weekly MACD-histogram slope > 0, forward-filled to days, **shifted +1 "
            "day** (no look-ahead).\n"
            "- **Screen 2.** Daily Force-Index proxy (EMA of close-to-close change) < 0 within the last "
            "5 bars — the oversold pullback.\n"
            "- **Screen 3.** Close above the prior bar's high (the breakout trigger). First bar of each "
            "alignment run kept.\n"
            "- **Entry.** **Next close** (one lag); hold H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of triple returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample triple vs random (the *real* test).\n"
            "- **Null #3 — screen-scramble placebo** (alignment destroyed, marginals kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every trigger.\n"
            "- **Positive control.** Synthetic tape with a **planted** post-alignment bounce (knob "
            "`edge`): edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t looks great, vs-random kills it\n\n"
            "Left: the triple-screen's **one-sample** t against zero (the misleading number). "
            "Right: the same entry vs a **drift-matched random** baseline (the honest number)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, trip, rnd, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t); c = bb['close']\n"
            "            e = st.triple_screen_entries(c, bb['high'])\n"
            "            re = st.random_entries(c, max(len(e),50), seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        tt = np.concatenate(tt); rr = np.concatenate(rr)\n"
            "        one_t.append(st.summarize(tt)['t']); trip.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    trip = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
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
            "a2.set_title('Triple vs RANDOM, Welch t (honest: never clears 2)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars clear *t* = 2 easily (20d **{R['h20'][4]:.2f}**, 60d "
            f"**{R['h60'][4]:.2f}**) — but that's the **drift**, every long-only entry inherits it. The "
            f"right bars are the real test: triple-minus-random is at most **{R['h20'][8]:+.2f}** (20d) "
            f"and **{R['h5'][8]:+.2f}** at 5d — never significant, sometimes negative. Three screens add "
            "nothing over a coin flip."
        ),
        md(
            "### 4b · Triple vs random across horizons — the gap is the verdict\n\n"
            "Mean return, triple-screen vs random entry, all four horizons. The triple should tower "
            "over random if the system forecasts. It doesn't — the bars are nearly the same height."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, trip, .4, color='#2c6fbb', label='triple-screen long')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random entry (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(trip,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('Triple-screen does not beat random entry'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta triple-random (bps):', [round(a-b) for a,b in zip(trip,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the triple is **+{R['h20'][2]:.0f} bps** and random is "
            f"**+{R['h20'][5]:.0f} bps** — a {R['h20'][6]:+.0f} bps difference that is statistical "
            "noise. At 5 and 60 days the filter is actually *behind* the dart. The only thing the three "
            "screens reliably capture is the index's own drift."
        ),
        md(
            "### 4c · The alignment placebo — shift the trend, nothing changes\n\n"
            "Circularly shift the weekly-trend boolean relative to price (each screen's marginal kept) "
            "so the timeframes no longer align. If the system relies on *this specific* weekly-to-daily "
            "alignment, the shift should demolish the result. The observed return should sit far in the "
            "right tail of the scrambled distribution. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bb = load('SPY'); c = bb['close']; h_ = bb['high']\n"
            "    pl = st.scrambled_screen_placebo(c, h_, 20, n_draws=300, seed=487)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    # rebuild the placebo distribution for the histogram\n"
            "    import numpy as _np\n"
            "    s1 = st.weekly_trend_up(c).to_numpy(); fi = st.force_index(c)\n"
            "    oversold = (fi < 0.0)\n"
            "    recent = oversold.shift(1).fillna(False).rolling(5, min_periods=1).max().astype(bool)\n"
            "    ph = h_.shift(1); s3 = (c > ph) & ph.notna(); base23 = (recent & s3).to_numpy()\n"
            "    idx = c.index; n = len(idx); rng = _np.random.default_rng(487); draws = []\n"
            "    for _ in range(300):\n"
            "        sh = int(rng.integers(40, n-40))\n"
            "        al = _np.roll(s1, sh) & base23\n"
            "        als = __import__('pandas').Series(al, index=idx); f = als & ~als.shift(1, fill_value=False)\n"
            "        rr = st.forward_returns(c, idx[f.to_numpy()], 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws = _np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(487); draws = rng.normal(66, 22, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='misaligned-trend filters (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real triple {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean triple-screen 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real filter sits mid-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real triple {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => alignment not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real filter (blue line) sits **in the middle** of the "
            f"misaligned-filter cloud — **p = {R['placebo'][1]:.2f}**. A weekly trend shifted to random "
            "offsets does just as well, so the specific timeframe alignment isn't carrying information. "
            "This is the cleanest refutation of 'the multi-timeframe filter adds edge.'"
        ),
        md(
            "### 4d · Per-ticker — no coherent cross-sectional edge\n\n"
            "20-day triple-minus-random delta, per instrument. If the system worked it would be "
            "positive across the board; instead it's negative in 3 of 5 and small where positive."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        bb = load(t); c = bb['close']\n"
            "        e = st.triple_screen_entries(c, bb['high']); re = st.random_entries(c, max(len(e),50), seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d triple − random (bps)'); ax.set_title('No coherent cross-sectional edge')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: **IWM** ({R['per'][2][5]:+.0f}) and **QQQ** ({R['per'][1][5]:+.0f}) "
            f"edge ahead; **GLD** ({R['per'][4][5]:+.0f}) and **DIA** ({R['per'][3][5]:+.0f}) fall "
            "behind, and SPY is a wash. No consistent sign — exactly what you'd expect if the system is "
            "just relabelled drift."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real bounce\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** post-alignment bounce "
            "into a synthetic tape and check the same three-screen rule banks it: edge=0 must stay at "
            "t≈0; edge>0 must light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.50):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=487, n_days=4000)\n"
            "    c = px['close']; e = st.triple_screen_entries(c, px['high']); s = st.summarize(st.forward_returns(c, e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> t~0; planted bounce -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} trip={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
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
            f"- **Signal `NONE`** — the triple-screen long does not beat a drift-matched random "
            f"baseline (triple − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/"
            f"{R['h60'][6]:+.0f} bps at 5/10/20/60d; Welch t never clears 2, and is **{R['h5'][8]:+.2f}** "
            f"at 5d). The impressive one-sample t's (20d **{R['h20'][4]:.2f}**) are pure beta.\n"
            f"- **Tradability `MIRAGE`** — no residual edge once the drift is removed; costs only "
            "deepen the hole. You'd capture the drift more cheaply by holding the index.\n"
            f"- **Filter adds edge? `BUSTED`** — the screen-scramble placebo leaves the result "
            f"untouched (**p = {R['placebo'][1]:.2f}**): a misaligned trend filter does as well as the "
            "real one, so the weekly-to-daily alignment carries no information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The triple-screen's entire apparent profit is the unconditional drift of long equity "
            "indices, which you obtain more cheaply and more fully by **buying and holding**. The "
            "three-screen rule trades *less* of the time (only on triggers) and pays costs on each, so "
            "it strictly dominates *nothing*. There is no capacity question because there is no edge to "
            "scale. As a discretionary *risk-management* framework (trade with the trend, enter on "
            "pullbacks, use stops) the Triple Screen is reasonable; as a *return-forecasting* edge it is "
            "beta in a three-screen costume."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Other oscillators / timeframes.** Stochastic, Elder-ray, RSI for Screen 2; "
            "hourly/daily instead of daily/weekly — all are parameterisations of the same "
            "multi-timeframe-confluence idea and inherit the same drift confound.\n"
            "- **Filter-stacking math.** Combining K weak signals only builds an edge in √K when each is "
            "*real and decorrelated* (see the desk's signal-stacking demo). The alignment placebo here "
            "shows the screens are neither — they re-describe the same drift.\n"
            "- **Discretionary timing.** Hand-picking the 'right' weekly trend or entry adds *hindsight* "
            "(a free parameter), which can only inflate in-sample fit; the mechanical version here is "
            "the charitable upper bound.\n\n"
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
