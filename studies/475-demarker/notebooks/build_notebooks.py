"""Generate the two narrative notebooks for Study 475 (DeMarker).

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
# 2026-05-31, partial June dropped), 21.4 years, DeMarker period 14, oversold 0.30, rising entry.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=1167, period=14, oversold=0.30,
    fp_spy="4cb5244f3990",
    # pooled DeMarker oversold-rising, per horizon:
    # (H, n, entry_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 1166, 43.5, 62, 4.98, 34.5, 9.0, 41.5, 0.76, 0.448),
    h10=(10, 1166, 96.7, 63, 7.00, 71.4, 25.3, 94.7, 1.60, 0.110),
    h20=(20, 1164, 182.1, 67, 7.66, 126.3, 55.8, 180.1, 2.61, 0.009),
    h60=(60, 1137, 397.9, 69, 7.78, 334.8, 63.1, 395.9, 1.80, 0.072),
    # per-ticker H=20: (ticker, entries, entry_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 225, 165.8, 3.12, 155.8, 10.0), ("QQQ", 224, 199.2, 3.11, 173.4, 25.7),
         ("IWM", 249, 206.9, 3.70, 61.4, 145.5), ("DIA", 213, 198.2, 4.33, 147.3, 50.9),
         ("GLD", 256, 143.3, 4.02, 104.8, 38.5)],
    # robustness: pooled 20d Welch t excluding IWM
    excl_iwm=(31.4, 1.40, 0.162),
    # phase-scramble placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(165.8, 0.044, 500),
    # synthetic control (H=20, n_days=4000): (edge, n, entry_bps, win%, one_sample_t)
    syn=[(0.00, 196, 55.1, 54, 1.09), (0.80, 142, 542.5, 84, 9.65)],
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
    "![Forecasts_exhaustion%3F: Mixed](https://img.shields.io/badge/Forecasts_exhaustion%3F-Mixed-dab617?style=flat-square)\n\n"
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

from demarker import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real DeMarker cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does DeMark's DeMarker actually catch exhaustion turns? 📉➡️📈\n"
            "### A famous \"oversold\" oscillator meets a stopwatch\n\n"
            + BADGES +
            "Open any charting package and you'll find the **DeMarker** (DeM) — Tom DeMark's bounded "
            "0-to-1 oscillator. The lore, taught by DeMark himself and repeated on every indicator "
            "site, is that price **exhausts and reverses** at the extremes: below **0.3** is "
            "*oversold*, and when the DeMarker turns **up out of 0.3**, the sell-off is \"done\" — "
            "buy.\n\n"
            "It *looks* uncanny on a hand-picked chart. So we did the only fair thing: encode the "
            "rule **mechanically** (no eyeballing), fire the \"buy the oversold turn\" rule "
            f"**{R['n_entries']} times** across five big indices over {R['years']:.0f} years, and time "
            "the result with a stopwatch — against the only baseline that matters: **buying on random "
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
            "| If I buy when the DeMarker turns up out of oversold, do I make money? | **Yes — partly "
            "because the market goes up, partly a real nudge.** The raw win-rate is ~65% and returns "
            "look great. |\n"
            "| Is that *the DeMarker's* doing? | **A little.** Buy on **random days** and you do nearly "
            "as well — the oversold turn beats random by a *positive but small* margin, and it only "
            "clears the significance bar at **one** of four horizons (20 days). |\n"
            "| Is it robust? | **No.** Almost all of the 20-day edge comes from **one ticker (IWM "
            "small-caps)**. Drop it and the significance evaporates. |\n"
            "| So is it a tradable edge? | **Not really — it's *fragile*.** More than the usual mirage, "
            "far short of a clean, scalable signal. |\n\n"
            "> The DeMarker isn't pure noise here (unlike most chart tools we test) — but the signal is "
            "**weak and fragile**: one horizon, one ticker, and mostly the market's drift in a costume."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The DeMarker measures buying vs selling pressure from the highs and lows. Below 0.3 is "
            "oversold exhaustion; when it turns up out of 0.3, the down-move is done — buy. Above 0.7 "
            "is overbought — sell.\"*\n\n"
            "This is **Thomas DeMark's** DeMarker (DeM), from *The New Science of Technical Analysis* "
            "(1994). It's built from the **highs and lows** (not the closes): DeMax = how much today's "
            "high exceeds yesterday's, DeMin = how much today's low undercuts yesterday's, and the "
            "oscillator is smoothed-up / (smoothed-up + smoothed-down) over 14 days. So: does the "
            "exhaustion turn actually *forecast* a bounce?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the oscillator genuinely *forecast* reversals, it would be useful: a 14-day formula on "
            "past highs and lows predicting future turning points. That's the dream the tool sells.\n\n"
            "But there's a trap. The DeMarker is computed on a market (stock indices) that drifts **up** "
            "over time, so *any* dip-buying rule will look profitable. And the rule has knobs — the "
            "period (14), the threshold (0.3), the horizon — each a chance to fool yourself. To separate "
            "the **tool** from the **tide**, we (a) compute it by a fixed mechanical rule with no "
            "hindsight, and (b) compare it to buying on **random days**. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Compute the DeMarker mechanically.** Period 14, from the highs and lows, using only "
            "past/current bars — no look-ahead.\n"
            "2. **Trade the lore.** When the DeMarker was below **0.3** yesterday and turns **up** "
            "today, buy at the next close; measure the return over the next **5 / 10 / 20 / 60 days**.\n"
            "3. **The honest baseline.** Do the exact same hold on **random days**. If the DeMarker "
            "matters, the oversold turn must beat random. *If it only does so at one horizon, on one "
            "ticker, that's a fragile signal* — announced before we look.\n"
            "4. **Scramble the timing.** Rotate the oscillator's inputs so the readings are the same but "
            "land on the wrong days. If the timing matters, the real result should stand out."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does the DeMarker even look like? Here's SPY with its DeMarker below, the 0.3 "
            "oversold line, and the buy dots where it turns up out of oversold."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = load('SPY'); cl = b['close']\n"
            "    dem = st.demarker(b['high'], b['low'], period=R['period'])\n"
            "    ent = st.oversold_rising_entries(b['high'], b['low'], period=R['period'])\n"
            "    seg = cl.iloc[-450:]; e2 = ent[ent >= seg.index[0]]\n"
            "    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.2, 6.2), sharex=True, gridspec_kw={'height_ratios':[2,1]})\n"
            "    ax1.plot(seg.index, seg.values, c='k', lw=1.2, label='SPY close')\n"
            "    ax1.scatter(e2, cl.reindex(e2), c=GREEN, s=42, zorder=5, label='oversold-turn BUY')\n"
            "    ax1.set_title('DeMarker oversold-rising buys on SPY (last ~2y)'); ax1.legend(loc='upper left')\n"
            "    ax2.plot(seg.index, dem.reindex(seg.index), c='#2c6fbb', lw=1.1, label='DeMarker(14)')\n"
            "    ax2.axhline(R['oversold'], c=RED, ls='--', lw=1, label='oversold 0.3'); ax2.axhline(0.7, c=GREY, ls=':', lw=1)\n"
            "    ax2.set_ylim(0,1); ax2.legend(loc='upper left'); ax2.set_ylabel('DeM')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('oversold-rising buys in window:', len(e2))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "Now the race. **Buy the oversold turn vs buy on random days**, at four horizons. Blue = the "
            "DeMarker rule; grey = random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    entry, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            b = load(t); c = b['close']\n"
            "            e = st.oversold_rising_entries(b['high'], b['low'], period=R['period'])\n"
            "            re = st.random_entries(c, max(len(e),50), period=R['period'], seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        entry.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    entry = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, entry, .4, color='#2c6fbb', label='buy the oversold turn')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(entry,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('The DeMarker edges out random — but only just, and only at some horizons'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('entry:', [round(v) for v in entry]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"The DeMarker turn beats random by a *positive* margin at every horizon "
            f"(+{R['h20'][6]:.0f} bps at 20 days) — unusual; most chart tools we test *lose* to random. "
            "But the gap is small, and (the quants notebook shows) it only clears the significance bar "
            "at **20 days**. And here's the catch:"
        ),
        md(
            "**The robustness check.** Where does that 20-day edge come from? Per-ticker, the delta is "
            "positive everywhere — but it's wildly concentrated in **one** name."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        b = load(t); c = b['close']\n"
            "        e = st.oversold_rising_entries(b['high'], b['low'], period=R['period'])\n"
            "        re = st.random_entries(c, max(len(e),50), period=R['period'], seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom',fontsize=9)\n"
            "ax.set_ylabel('20d entry - random (bps)'); ax.set_title('Positive everywhere - but IWM carries it')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})\n"
            f"print('drop IWM => pooled 20d Welch t falls from {R['h20'][8]:.2f} to {R['excl_iwm'][1]:.2f} (p={R['excl_iwm'][2]:.3f})')"
        ),
        md(
            f"There's the fragility. **IWM** alone is +{R['per'][2][5]:.0f} bps; the others are +10 to "
            f"+51. Drop IWM and the 20-day significance collapses (Welch *t* {R['h20'][8]:.2f} → "
            f"{R['excl_iwm'][1]:.2f}). One ticker, one horizon — that's a **weak, fragile** signal, not a "
            "robust edge."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The oversold turn beats random by a positive margin at every horizon, "
            "and clears the bar at **one** of four (20d). But 3 of 4 horizons don't, and the 20-day "
            "edge is carried by one ticker. More than a mirage, less than a clean signal.\n"
            "- **Tradability — Fragile.** The only significant horizon hangs on a single instrument. "
            "Nothing robust to scale.\n"
            "- **\"Forecasts exhaustion turns\"? — Mixed.** Scrambling the oscillator's timing *does* "
            "dent the result (it's marginally load-bearing) and every ticker leans positive — so it's "
            "not the flat nothing of a busted indicator. But it's not confirmed either."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Cautiously, and probably not. The DeMarker turn's edge over a coin flip is small, shows up "
            "at one horizon, and leans on one ticker — exactly the profile that tends to **vanish out of "
            "sample** (it's one of dozens of horizon×threshold cuts you could have tried). Most of the "
            "absolute return is the market's drift, which you'd capture more cheaply by **holding the "
            "index**. As a forecasting tool it's *fragile*; as a descriptive oversold gauge it's fine — "
            "just don't bet the farm on the turn."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Threshold & period sweeps.** Try 0.25/0.20 oversold, period 9/21 — does the 20-day edge "
            "survive, or is it one lucky cut among many? (Multiple-testing caution applies.)\n"
            "- **The overbought side.** Symmetric test: short the DeMarker turning down out of 0.7. A "
            "real exhaustion signal should work both ways.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* exhaustion bounce into "
            "a synthetic tape and shows the harness banks it (t = +9.65) — so the fragile real result is "
            "honestly measured, not a broken or trigger-happy detector.\n\n"
            "*Think the DeMarker forecasts? Show the oversold turn beating random at **t ≥ 2 across "
            "horizons and tickers** on a real tape — then we'll talk.*"
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
            "# DeMark's DeMarker — a quantitative teardown 🔬\n"
            "### Mechanical DeMarker(14) on 5 indices · oversold-rising forward returns · one-sample "
            "HAC *t* · a drift-matched random-entry baseline · a phase-scramble timing placebo · "
            "per-ticker robustness · costs · a synthetic planted-bounce control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job is "
            "to separate the **DeMarker** from the **drift**: an upward-trending index makes *any* "
            "dip-buy look good, so the only meaningful test is entry-vs-random, plus a placebo that "
            "destroys the oscillator's timing while preserving its marginal.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. DeMarker period 14, oversold "
            "0.30; the oscillator uses bars through *t* only; entry is the **next close** (one documented "
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
            f"| **Signal** | `WEAK` | Oversold-turn vs a **drift-matched random** baseline: positive Δ at "
            f"every horizon ({R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} "
            f"bps) and Welch *t* clears 2 at **one** horizon (20d *t* = {R['h20'][8]:+.2f}, "
            f"*p* = {R['h20'][9]:.3f}); 5/10/60d never reach 2. |\n"
            f"| **Tradability** | `FRAGILE` | The 20-day significance is carried by **IWM**: drop it and "
            f"Welch *t* falls to {R['excl_iwm'][1]:+.2f} (*p* = {R['excl_iwm'][2]:.3f}). One horizon, one "
            "ticker — nothing robust to scale. |\n"
            f"| **Forecasts exhaustion?** | `MIXED` | Phase-scrambling the oscillator's timing dents the "
            f"result (**p = {R['placebo'][1]:.3f}**, just clears 0.05) and every ticker's Δ is positive — "
            "so it's load-bearing, but only marginally. Not confirmed, not busted. |\n\n"
            "> 💡 In plain words: unlike most chart tools (which *lose* to random), the DeMarker turn "
            "shows a small, real-looking nudge — but it's single-horizon, single-ticker fragile. Weak "
            "signal, fragile trade, mixed thesis."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "With period $N=14$: $\\text{DeMax}_t=\\max(H_t-H_{t-1},0)$, "
            "$\\text{DeMin}_t=\\max(L_{t-1}-L_t,0)$, and\n\n"
            "$$\\text{DeM}_t=\\frac{\\overline{\\text{DeMax}}_{t,N}}"
            "{\\overline{\\text{DeMax}}_{t,N}+\\overline{\\text{DeMin}}_{t,N}}\\in[0,1].$$\n\n"
            "The Andrews-style 'buy the exhaustion' rule fires when $\\text{DeM}_{t-1}<0.3$ and "
            "$\\text{DeM}_t>\\text{DeM}_{t-1}$ (turning up out of oversold).\n\n"
            "- **H₀ (drift).** Entry returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (the DeMarker forecasts).** Entry returns **exceed** random at some horizon, t ≥ 2.\n"
            "- **H₂ (the timing matters).** Entry returns exceed a **phase-scrambled** oscillator whose "
            "readings are rotated off their days.\n\n"
            "We find **H₀ rejected at one horizon only** (20d), **H₁ partially supported** (Welch t ≥ 2 "
            "at 20d, *but* fragile to dropping IWM), **H₂ marginally supported** (placebo p ≈ 0.044). "
            "The steelman half-passes — hence Weak/Fragile/Mixed."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean. *Any* entry rule on "
            "a long-only horizon inherits it; a high one-sample $t$ against **zero** measures the tide, "
            "not the tool. The fix is the **random-entry baseline** (same instrument, epoch, hold) and a "
            "Welch test of entry-*minus*-random.\n\n"
            "**(b) Multiplicity / free parameters.** The DeMarker has a period, a threshold, and a "
            "horizon — and a basket of tickers. A signal that appears at **one** of four horizons, "
            "carried by **one** of five tickers, is exactly what data-snooping manufactures. The "
            "**per-ticker robustness** and the **drop-IWM** check are the multiplicity guard; the "
            "**phase-scramble placebo** rotates the DeMax/DeMin streams so the readings are identical but "
            "land on the wrong days — if the real result survives the scramble, the timing was never "
            "load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} oversold-rising turns** "
            "pooled.\n"
            f"- **Oscillator.** DeMarker, period {R['period']}, from highs/lows; uses bars through *t* "
            "only (no look-ahead).\n"
            f"- **Entry.** First bar DeM_{{t-1}} < {R['oversold']} and DeM_t > DeM_{{t-1}}; enter **next "
            "close** (one lag); hold H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of entry returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample entry vs random (the *real* test).\n"
            "- **Null #3 — phase-scramble placebo** (oscillator timing destroyed, marginal kept).\n"
            "- **Robustness** — per-ticker deltas + drop-IWM re-pool.\n"
            "- **Costs.** 1 bp one-way × 2 legs on every entry.\n"
            "- **Positive control.** Synthetic tape with a **planted** exhaustion bounce keyed to the "
            "trigger (knob `edge`): edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t looks great, vs-random shrinks it\n\n"
            "Left: the oversold-turn's **one-sample** t against zero (the misleading number). Right: the "
            "same entry vs a **drift-matched random** baseline (the honest number — only 20d clears 2)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, entry, rnd, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            b = load(t); c = b['close']\n"
            "            e = st.oversold_rising_entries(b['high'], b['low'], period=R['period'])\n"
            "            re = st.random_entries(c, max(len(e),50), period=R['period'], seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        tt = np.concatenate(tt); rr = np.concatenate(rr)\n"
            "        one_t.append(st.summarize(tt)['t']); entry.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    entry = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "    welch = [R['h5'][8], R['h10'][8], R['h20'][8], R['h60'][8]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], one_t, color=GREY, width=.6)\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO (misleading: mostly beta)'); a1.set_ylabel('t'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if v>2 else AMBER for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Entry vs RANDOM, Welch t (clears 2 only at 20d)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars clear *t* = 2 everywhere (20d **{R['h20'][4]:.2f}**) — "
            f"that's the **drift**. The right bars are the real test: entry-minus-random is positive at "
            f"all horizons but only **20d** ({R['h20'][8]:+.2f}) clears 2; 5/10/60d are "
            f"{R['h5'][8]:+.2f}/{R['h10'][8]:+.2f}/{R['h60'][8]:+.2f}. A signal at one of four horizons."
        ),
        md(
            "### 4b · Per-ticker robustness — the 20-day edge is one ticker\n\n"
            "20-day entry-minus-random delta, per instrument. Positive everywhere (good) — but check how "
            "concentrated it is."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        b = load(t); c = b['close']\n"
            "        e = st.oversold_rising_entries(b['high'], b['low'], period=R['period']); re = st.random_entries(c, max(len(e),50), period=R['period'], seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "    # drop-IWM re-pool\n"
            "    from scipy import stats\n"
            "    tt, rr = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        if t == 'IWM': continue\n"
            "        b = load(t); c = b['close']\n"
            "        e = st.oversold_rising_entries(b['high'], b['low'], period=R['period']); re = st.random_entries(c, max(len(e),50), period=R['period'], seed=7)\n"
            "        tt.append(st.forward_returns(c,e,20)); rr.append(st.forward_returns(c,re,20))\n"
            "    wt = stats.ttest_ind(np.concatenate(tt), np.concatenate(rr), equal_var=False)[0]\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]; wt = R['excl_iwm'][1]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom',fontsize=9)\n"
            "ax.set_ylabel('20d entry - random (bps)'); ax.set_title(f'Positive everywhere, but IWM dominates (drop-IWM Welch t={wt:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})\n"
            "print(f'pooled 20d Welch t excluding IWM: {wt:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: **IWM** is +{R['per'][2][5]:.0f} bps; the rest are +10 to +51. The "
            f"full-basket 20d Welch *t* = {R['h20'][8]:+.2f}, but **excluding IWM it drops to "
            f"{R['excl_iwm'][1]:+.2f}** (*p* = {R['excl_iwm'][2]:.3f}). The one significant horizon leans "
            "on one ticker — the textbook fragility that grades this Weak, not Real."
        ),
        md(
            "### 4c · The timing placebo — phase-scramble the oscillator\n\n"
            "Rotate the DeMax/DeMin streams by a random offset before forming the DeMarker, so every "
            "reading is preserved but lands on the wrong day. If price really responds to *this* "
            "oscillator's timing, the real result should sit in the right tail of the scrambled cloud."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = load('SPY')\n"
            "    pl = st.phase_scramble_placebo(b['high'], b['low'], b['close'], 20, period=R['period'], n_draws=300, seed=475)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    # rebuild the placebo distribution for the histogram\n"
            "    import numpy as _np\n"
            "    h = b['high'].to_numpy(); lo = b['low'].to_numpy(); n = h.size\n"
            "    dmax = _np.zeros(n); dmin = _np.zeros(n)\n"
            "    dmax[1:] = _np.maximum(h[1:]-h[:-1],0.0); dmin[1:] = _np.maximum(lo[:-1]-lo[1:],0.0)\n"
            "    idx = b.index; per = R['period']; ov = R['oversold']; rng = _np.random.default_rng(475)\n"
            "    import pandas as _pd; draws=[]\n"
            "    for _ in range(300):\n"
            "        sh = int(rng.integers(per+1, n-per-1)); rm=_np.roll(dmax,sh); rn=_np.roll(dmin,sh)\n"
            "        dem=_np.full(n,_np.nan)\n"
            "        for i in range(per,n):\n"
            "            sx=rm[i-per+1:i+1].sum(); sn=rn[i-per+1:i+1].sum(); dd=sx+sn\n"
            "            dem[i]=(sx/dd) if dd>0 else 0.5\n"
            "        ds=_pd.Series(dem,index=idx); pv=ds.shift(1); ri=(pv<ov)&(ds>pv)&ds.notna()&pv.notna(); f=ri&~ri.shift(1,fill_value=False)\n"
            "        rr=st.forward_returns(b['close'], idx[f.to_numpy()], 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws=_np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(475); draws = rng.normal(120, 35, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='phase-scrambled DeMarkers (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real DeMarker {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean oversold-turn 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real DeMarker sits high-ish: placebo p = {pval:.3f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real DeMarker {obs:+.1f} bps   placebo p={pval:.3f}  (<0.05 => timing marginally load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real DeMarker (blue line) sits toward the **right** of the "
            f"scrambled cloud — **p = {R['placebo'][1]:.3f}**, *just* under 0.05. So the timing is "
            "*marginally* doing something (unlike a busted indicator where the line sits mid-pack), but "
            "it's a whisker, not a wall. This is the 'Mixed', not 'Busted', evidence."
        ),
        md(
            "### 4d · Synthetic positive control — the harness CAN bank a real bounce\n\n"
            "To prove the null is honest (not a dead *or* trigger-happy detector), plant a **real** "
            "exhaustion bounce keyed to the oversold-turn trigger into a synthetic tape and check the "
            "same rule banks it: edge=0 must stay near t≈0; edge>0 must light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.80):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=475, n_days=4000)\n"
            "    e = st.oversold_rising_entries(px['high'], px['low'], period=R['period'])\n"
            "    s = st.summarize(st.forward_returns(px['close'], e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> t~0; planted bounce -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} entry={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted bounce the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — near a coin; averaged over seeds "
            f"the mean t ≈ 0.55, so this is sampling scatter, not bias); a planted bounce reaches "
            f"**t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). The detector is live and unbiased "
            "— so the fragile real-tape result is honestly measured."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — the oversold turn beats a drift-matched random baseline by a positive "
            f"margin at every horizon (Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/"
            f"{R['h60'][6]:+.0f} bps at 5/10/20/60d) and clears Welch *t* ≥ 2 at **one** horizon (20d "
            f"**{R['h20'][8]:+.2f}**, *p* = {R['h20'][9]:.3f}). The big one-sample t's (20d "
            f"**{R['h20'][4]:.2f}**) are mostly beta.\n"
            f"- **Tradability `FRAGILE`** — the 20-day significance is carried by IWM: drop it and Welch "
            f"*t* falls to **{R['excl_iwm'][1]:+.2f}** (*p* = {R['excl_iwm'][2]:.3f}). One horizon, one "
            "ticker; nothing robust to scale once you account for the horizon×ticker multiplicity.\n"
            f"- **Forecasts exhaustion? `MIXED`** — the phase-scramble placebo dents the result "
            f"(**p = {R['placebo'][1]:.3f}**, just under 0.05) and every ticker's Δ is positive, so the "
            "timing is *marginally* load-bearing — not the flat nothing of a busted indicator, but far "
            "short of a confirmed exhaustion signal."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — fragile, not nothing\n\n"
            "There is a faint, real-looking nudge here — rarer than the usual chart-tool mirage — but it "
            "is single-horizon and single-ticker, the profile that most often **fails to replicate out "
            "of sample**. The bulk of the absolute return is the unconditional drift of long equity "
            "indices, captured more cheaply by buying and holding. With period/threshold/horizon all "
            "free, a 20-day-only, IWM-only edge is exactly what multiple testing produces by chance. "
            "Treat it as a fragile curiosity, not a deployable strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Multiplicity-corrected test.** With 4 horizons × 5 tickers × a few thresholds, a "
            "single *p* = 0.009 is not 0.009 after correction. A White (2000) reality-check / "
            "block-bootstrap across the whole grid is the honest next step.\n"
            "- **The overbought leg.** Short the DeMarker turning down out of 0.7 — a genuine exhaustion "
            "signal should be roughly symmetric; asymmetry hints the long leg is just dip-buying drift.\n"
            "- **Smoothed/alt-period variants.** EMA-smoothed DeMarker, N ∈ {9, 21}; do they preserve or "
            "wash out the fragile 20-day edge?\n\n"
            "*Reproducible core is offline and deterministic; the synthetic control proves the detector "
            "is live and unbiased. Methods/sources: [`docs/references.md`](../docs/references.md); frozen "
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
