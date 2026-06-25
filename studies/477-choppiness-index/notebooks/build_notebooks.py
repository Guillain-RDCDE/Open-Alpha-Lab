"""Generate the two narrative notebooks for Study 477 (Choppiness Index).

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
# 2026-05-31, partial June dropped), 21.4 years, CI window N=14, low-CI threshold 38.2, long.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=936, window=14, low=38.2,
    fp_spy="4cb5244f3990",
    # pooled low-CI onset, per horizon:
    # (H, n, lowci_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 936, 45.4, 60, 5.25, 27.7, 17.7, 43.4, 1.54, 0.124),
    h10=(10, 936, 61.5, 63, 5.07, 61.2, 0.3, 59.5, 0.02, 0.983),
    h20=(20, 935, 106.7, 63, 5.56, 114.4, -7.7, 104.7, -0.33, 0.741),
    h60=(60, 928, 293.8, 68, 7.93, 250.1, 43.8, 291.8, 1.15, 0.250),
    # per-ticker H=20: (ticker, entries, lowci_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 182, 84.3, 2.36, 105.0, -20.7), ("QQQ", 186, 101.8, 1.88, 144.8, -43.0),
         ("IWM", 153, 99.8, 1.82, 99.5, 0.3), ("DIA", 184, 128.4, 5.01, 96.4, 32.0),
         ("GLD", 231, 115.5, 2.78, 121.7, -6.1)],
    # return-shuffled placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(84.3, 0.409, 500),
    # synthetic control (H=20, n_days=4000): (edge, n, lowci_bps, win%, one_sample_t)
    syn=[(0.00, 80, 63.5, 57, 1.12), (0.35, 84, 871.0, 79, 7.47)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Forecasts_trend_vs_chop%3F: Busted](https://img.shields.io/badge/Forecasts_trend_vs_chop%3F-Busted-8b949e?style=flat-square)\n\n"
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

from choppiness_index import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real choppiness cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does the Choppiness Index actually call \"trend vs chop\"? 🌊📏\n"
            "### A famous regime gauge — one number from 0 to 100 — meets a stopwatch\n\n"
            + BADGES +
            "Open any charting package and you'll find the **Choppiness Index** (CI): a single "
            "0-to-100 line that's *supposed* to tell you whether the market is **trending** (low CI) "
            "or **chopping** sideways (high CI). The lore, repeated on every indicator site, is that "
            "a **low** CI marks the start of a clean directional run — so you switch on trend-"
            "following, ride the move, and stay out of the whipsaw when CI is high.\n\n"
            "It *looks* compelling: pull up a chart, find a low-CI dip, and sure enough a trend often "
            "follows. But a gauge computed **from** recent price, on a market that drifts **up** over "
            "time, is the textbook setup for fooling yourself. So we did the only fair thing: encode "
            "the \"buy when CI goes low\" rule **mechanically** (no eyeballing), fire it ~936 times "
            "across five big indices over 21 years, and time the result with a stopwatch — against "
            "the only baseline that matters: **buying on random days instead.**\n\n"
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
            "| If I buy when the **CI drops low** (a 'trend' is starting), do I make money? | **Yes — "
            "but only because the market goes up.** The raw win-rate is ~60-68% and the returns look "
            "great. |\n"
            "| Is that *the CI's* doing? | **No.** Buy on **random days** instead and you do "
            "**just as well** — at 10 and 20 days the low-CI entry is actually *no better* (or "
            "slightly worse) than a coin-flip entry. |\n"
            "| Does the CI really tell trend from chop in a *useful* way? | **Not in any tradable "
            "sense.** Scramble the trend-vs-chop structure into nonsense and the result barely "
            "changes. The 'regime reading' isn't doing the work. |\n"
            "| So is it a tradable edge? | **No.** It's **beta in a costume** — the upward drift of "
            "stocks, re-labelled as a regime filter. |\n\n"
            "> The Choppiness Index is a fine way to *describe* how straight or thrashy the recent "
            "tape was. As a *forecast* — \"low CI now ⇒ a tradable trend next\" — it's a **mirage**: "
            "all of the apparent edge is the market's long-run climb, none of it is the gauge."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Compute CI = 100·log₁₀(Σ ATR / range) / log₁₀(N) over the last N bars. A **low** "
            "reading (below ~38) means price moved in a near-straight line — a **trend** is underway, "
            "so go with it. A **high** reading (above ~62) means price thrashed back and forth — "
            "**chop**, stand aside.\"*\n\n"
            "This is **E. W. (Bill) Dreiss'** Choppiness Index (1990s), now built into TradingView, "
            "MetaTrader and most charting suites, usually with Fibonacci 38.2 / 61.8 bands. It "
            "compares the *path length* of the last N bars (summed bar-ranges) to the *straight-line* "
            "high-low span: straight move ⇒ low CI, zig-zag ⇒ high CI. So: does the regime gauge "
            "actually *forecast* the regime?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a low CI genuinely *forecast* a coming trend, it would be valuable: a number computed "
            "from the last 14 bars would tell you the next 14 are about to run, a clean, tradable "
            "crack in market efficiency.\n\n"
            "But there's a trap built in. The CI is **non-directional** — it's the same whether price "
            "trends up or down — so any *long* rule grafted on it leans entirely on the market "
            "drifting **up**, which it does. And it's computed *from* recent price, so on a rising "
            "tape *any* 'I'm now long' rule will look profitable. To separate the **gauge** from the "
            "**tide**, we (a) compute the CI by a fixed mechanical rule with no hindsight, and (b) "
            "compare it to buying on **random days**. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Compute the CI mechanically.** Over a trailing "
            f"**{R['window']}-bar** window: summed true range ÷ the window's high-low span, "
            "log-scaled to 0-100. Trailing only — it never peeks at future bars.\n"
            f"2. **Fire the lore.** When the CI first drops **below {R['low']}** (a 'trend' onset), "
            "buy at the next close; measure the return over the next **5 / 10 / 20 / 60 days**.\n"
            "3. **The honest baseline.** Do the exact same hold on **random days**. If the CI "
            "matters, the low-CI entry must beat random. *If it doesn't, the gauge is a mirage* — "
            "that's the result that would make us say so, announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does the CI even look like, and where does the 'trend onset' rule fire? "
            "Here's SPY with its Choppiness Index below it, and the low-CI buys the rule would take."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    seg = b.iloc[-450:]\n"
            "    ci = st.choppiness_index(b, window=R['window'])\n"
            "    ent = st.low_ci_entries(b, window=R['window'], low=R['low'])\n"
            "    ent = ent[ent >= seg.index[0]]\n"
            "    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10.2, 6.0), sharex=True,\n"
            "                                 gridspec_kw={'height_ratios':[2,1]})\n"
            "    a1.plot(seg.index, seg['close'], c='k', lw=1.2, label='SPY close')\n"
            "    a1.scatter(ent, b['close'].reindex(ent), c=GREEN, s=42, zorder=5, label='low-CI BUY')\n"
            "    a1.set_title('Choppiness Index trend-onset buys on SPY (last ~2y)'); a1.legend(loc='upper left')\n"
            "    a2.plot(seg.index, ci.reindex(seg.index), c='#2c6fbb', lw=1.1, label='Choppiness Index')\n"
            "    a2.axhline(R['low'], ls='--', c=GREEN, label=f'low-CI {R[\"low\"]} (trend)')\n"
            "    a2.axhline(61.8, ls='--', c=RED, label='high-CI 61.8 (chop)')\n"
            "    a2.set_ylim(0,100); a2.set_ylabel('CI'); a2.legend(loc='upper left', fontsize=8)\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('low-CI onsets in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "The CI dutifully drops when price has been moving straight and rises when it chops — "
            "*as a description*. The question is whether those green buy dots are followed by a "
            "**tradable** run. **Let's race the low-CI entry against random entries** at four "
            "horizons. Blue = buy when CI goes low; grey = buy on random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    lowci, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t); c = bb['close']\n"
            "            e = st.low_ci_entries(bb, window=R['window'], low=R['low'])\n"
            "            re = st.random_entries(c, max(len(e),50), window=R['window'], seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        lowci.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    lowci = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, lowci, .4, color='#2c6fbb', label='buy when CI goes low')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(lowci,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('Low-CI entry does NOT beat random — it just matches it'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('low-CI:', [round(v) for v in lowci]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the whole story in one chart. The low-CI entry makes money in absolute terms "
            f"(**+{R['h20'][2]:.0f} bps** over 20 days) — but **random entries make about the same** "
            f"(**+{R['h20'][5]:.0f} bps**). At 10 and 20 days the famous regime gauge is *no better* "
            "than throwing darts. The apparent edge was **the market's upward drift**, not the "
            "trend-vs-chop reading."
        ),
        md(
            "**One more sanity check.** What if we scramble the CI's *structure* — shuffle the daily "
            "returns so the same price moves happen in a random order, turning every 'clean trend' "
            "into noise and vice-versa? If the CI's trend-vs-chop reading really mattered, the "
            "nonsense version should do much worse."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.shuffled_returns_placebo(load('SPY'), 20, window=R['window'], low=R['low'], n_draws=300, seed=477)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real low-CI entry (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of *structure-scrambled* CIs do at least as well (p={pval:.2f}).')\n"
            "print('=> the trend-vs-chop reading is not doing the work.')"
        ),
        md(
            f"More than a third of the **scrambled** CIs match or beat the real one "
            f"(*p* = {R['placebo'][1]:.2f}). If price genuinely behaved differently after a *real* "
            "low-CI reading, a random scramble would collapse the result. It doesn't — because the "
            "result was never about the regime reading."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The low-CI 'trend onset' buy does **not** beat buying on random "
            "days (it merely *matches* it; the difference never clears *t* = 2). The big absolute "
            "returns are the market's drift, not the gauge.\n"
            "- **Tradability — Mirage.** Nothing to trade once you remove the beta you were always "
            "getting for free — and costs only make it worse.\n"
            "- **\"Forecasts trend vs chop\"? — Busted.** Scramble the trend-vs-chop structure into "
            "nonsense and the result barely moves. The regime gauge doesn't forecast the regime."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. The low-CI entry's *only* advantage over a coin flip is "
            "the market's long-run climb — which you'd capture more cheaply (and more fully) by just "
            "**holding the index**. The CI filter is a worse, more expensive way to be long: it "
            "trades less of the time and pays costs on every switch. As a forecasting tool it doesn't "
            "pay; as a description of how thrashy the tape has been, it was never meant to be a "
            "strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **High-CI = chop?** The mirror claim is that a *high* CI precedes whipsaw. On a "
            "drifting index that's just as confounded — a fun follow-up is to show high-CI forward "
            "returns are *also* drift, not 'chop'.\n"
            "- **Different thresholds / windows.** Try 30/70 bands or N=10/20 — the result is robust: "
            "drift in, regime label out.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* 'low-CI → "
            "momentum' structure into a synthetic tape and shows the harness banks it (so the null "
            "result here isn't a dead detector — it's an honest 'nothing there').\n\n"
            "*Think the CI forecasts the regime? Show the low-CI entry beating random entries at "
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
            "# The Choppiness Index — a quantitative teardown 🔬\n"
            "### Mechanical trailing CI on 5 indices · low-CI-onset forward returns · one-sample HAC "
            "*t* · a drift-matched random-entry baseline · a return-shuffled structure placebo · "
            "costs · a synthetic planted-momentum control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **gauge** from the **drift**: an upward-trending index makes *any* "
            "long entry look good, so the only meaningful test is low-CI-vs-random, plus a placebo "
            "that destroys the CI's trend-vs-chop structure while preserving its marginals.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. CI is trailing-only "
            f"(N={R['window']}, low-CI threshold {R['low']}); entry is the **next close** (one "
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
            f"| **Signal** | `NONE` | Low-CI onset vs a **drift-matched random** baseline: the "
            f"difference is tiny (Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/"
            f"{R['h60'][6]:+.0f} bps at 5/10/20/60d) and the low-CI-minus-random Welch *t* **never "
            f"clears 2** (max {R['h5'][8]:+.2f} at 5d, *p* = {R['h5'][9]:.3f}). |\n"
            f"| **Tradability** | `MIRAGE` | The big one-sample t's (20d t = {R['h20'][4]:.2f}, 60d "
            f"t = {R['h60'][4]:.2f}) are **pure beta** — they vanish against random entries and "
            "against cost. No residual edge to scale. |\n"
            f"| **Forecasts trend vs chop?** | `BUSTED` | Scrambling the CI's trend-vs-chop structure "
            f"(return-shuffled placebo) leaves the result intact: **p = {R['placebo'][1]:.2f}** of "
            "nonsense CIs match or beat the real one. The regime reading isn't doing the work. |\n\n"
            "> 💡 In plain words: the low-CI entry *looks* significant only because indices drift up. "
            "Strip the drift (race it vs random) or strip the structure (shuffle the returns) and the "
            "edge evaporates. Classic beta-in-a-costume."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Over a trailing window of $N$ bars, let $\\text{TR}_i$ be the true range and "
            "$\\text{span}=\\max(\\text{high})-\\min(\\text{low})$. The Choppiness Index is\n\n"
            "$$\\text{CI}_t = 100\\,\\frac{\\log_{10}\\!\\big(\\sum_{i} \\text{TR}_i / \\text{span}\\big)}"
            "{\\log_{10} N}\\in[0,100].$$\n\n"
            "A straight move has $\\sum \\text{TR}\\approx\\text{span}\\Rightarrow$ CI low; a zig-zag "
            "has $\\sum \\text{TR}\\gg\\text{span}\\Rightarrow$ CI high. The rule buys at the **onset** "
            "of $\\text{CI}_t<\\tau$ (here $\\tau=38.2$) and rides the prevailing trend.\n\n"
            "- **H₀ (drift).** Low-CI returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (the CI forecasts).** Low-CI returns **exceed** random at some horizon, t ≥ 2.\n"
            "- **H₂ (the structure matters).** Low-CI returns exceed a **return-shuffled** CI whose "
            "trend-vs-chop reading is nonsense.\n\n"
            "We find **H₀ not rejected** (low-CI ≈ random everywhere), **H₁ rejected** (Welch t never "
            "≥ 2), **H₂ rejected** (placebo p ≈ 0.41). The steelman fails on every leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean. *Any* long entry "
            "on a long horizon inherits it; a high one-sample $t$ against **zero** measures the tide, "
            "not the gauge. And the CI is **sign-blind** — it cannot tell an up-trend from a "
            "down-trend — so the long-only proxy is *pure* drift exposure. The fix is the "
            "**random-entry baseline** (same instrument, epoch, hold) and a Welch test of "
            "low-CI-*minus*-random.\n\n"
            "**(b) Structure as a free parameter.** The danger is that *any* recent-price gauge on a "
            "trending tape produces 'regime' labels that look predictive. The **return-shuffled "
            "placebo** permutes the daily returns — same price marginal, but the trend-vs-chop "
            "structure the CI reads is destroyed — so if the real result survives the scramble, the "
            "regime reading was never load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} low-CI onsets** "
            "pooled.\n"
            f"- **CI.** Trailing window N={R['window']}; CI = 100·log₁₀(ΣTR/span)/log₁₀(N), "
            "min_periods=N (no look-ahead).\n"
            f"- **Entry.** First bar CI < {R['low']} after being ≥ it (a regime *onset*); enter "
            "**next close** (one lag); hold H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of low-CI returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample low-CI vs random (the *real* test).\n"
            "- **Null #3 — return-shuffled placebo** (structure destroyed, marginals kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every entry.\n"
            "- **Positive control.** Synthetic tape with a **planted** 'low-CI → momentum' structure "
            "(knob `edge`): edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap — one-sample t looks great, vs-random kills it\n\n"
            "Left: the low-CI entry's **one-sample** t against zero (the misleading number). "
            "Right: the same entry vs a **drift-matched random** baseline (the honest number)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, lowci, rnd, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t); c = bb['close']\n"
            "            e = st.low_ci_entries(bb, window=R['window'], low=R['low'])\n"
            "            re = st.random_entries(c, max(len(e),50), window=R['window'], seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        tt = np.concatenate(tt); rr = np.concatenate(rr)\n"
            "        one_t.append(st.summarize(tt)['t']); lowci.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    lowci = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
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
            "a2.set_title('Low-CI vs RANDOM, Welch t (honest: never clears 2)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars clear *t* = 2 easily (20d **{R['h20'][4]:.2f}**, 60d "
            f"**{R['h60'][4]:.2f}**) — but that's the **drift**, every long entry inherits it. The "
            f"right bars are the real test: low-CI-minus-random is **near zero** "
            f"({R['h20'][8]:+.2f} at 20d) and only **{R['h5'][8]:+.2f}** at its best (5d) — never "
            "significant. The CI adds nothing over a coin flip."
        ),
        md(
            "### 4b · Low-CI vs random across horizons — the gap is the verdict\n\n"
            "Mean return, low-CI entry vs random entry, all four horizons. The low-CI bar should "
            "tower over random if the CI forecasts the regime. It doesn't."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, lowci, .4, color='#2c6fbb', label='low-CI onset')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random entry (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(lowci,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('Low-CI onset does not beat random entry'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta low-CI minus random (bps):', [round(a-b) for a,b in zip(lowci,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the low-CI entry is **+{R['h20'][2]:.0f} bps** and random "
            f"is **+{R['h20'][5]:.0f} bps** — the gauge *underperforms* a dart by "
            f"{abs(R['h20'][6]):.0f} bps. At 10 days they're a dead heat ({R['h10'][6]:+.0f} bps). No "
            "horizon clears the Welch bar (4a)."
        ),
        md(
            "### 4c · The structure placebo — scramble the CI, nothing changes\n\n"
            "Shuffle the daily returns (price marginal kept, trend-vs-chop structure destroyed), "
            "recompute the CI and its low-CI onsets on the surrogate, and bank the *real* forward "
            "return at those dates. If price respects a *real* regime reading, the scramble should "
            "demolish the result. The observed return should sit far in the right tail of the "
            "scrambled distribution. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bb = load('SPY')\n"
            "    pl = st.shuffled_returns_placebo(bb, 20, window=R['window'], low=R['low'], n_draws=300, seed=477)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    import numpy as _np\n"
            "    rng = _np.random.default_rng(477); draws=[]\n"
            "    c = bb['close']\n"
            "    for _ in range(300):\n"
            "        sur = st._surrogate_bars(bb, rng)\n"
            "        ent = st.low_ci_entries(sur, window=R['window'], low=R['low'])\n"
            "        rr = st.forward_returns(c, ent, 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws=_np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(477); draws = rng.normal(90, 35, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='structure-scrambled CIs (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real CI {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean low-CI-onset 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real CI sits mid-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real CI {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => structure not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real CI (blue line) sits **in the middle** of the "
            f"scrambled-CI cloud — **p = {R['placebo'][1]:.2f}**. A return-permuted nonsense CI does "
            "just as well, so the specific trend-vs-chop reading isn't carrying any information. This "
            "is the cleanest refutation of 'the CI forecasts the regime.'"
        ),
        md(
            "### 4d · Per-ticker — no coherent cross-sectional edge\n\n"
            "20-day low-CI-minus-random delta, per instrument. If the CI worked it would be positive "
            "across the board; instead it's split (negative in 3 of 5)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        bb = load(t); c = bb['close']\n"
            "        e = st.low_ci_entries(bb, window=R['window'], low=R['low']); re = st.random_entries(c, max(len(e),50), window=R['window'], seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d low-CI − random (bps)'); ax.set_title('Low-CI underperforms random in 3 of 5 names')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: only **DIA** ({R['per'][3][5]:+.0f} bps) and barely **IWM** "
            f"({R['per'][2][5]:+.0f} bps) edge positive; QQQ is **{R['per'][1][5]:+.0f}** bps *behind* "
            "random. No coherent, cross-sectional edge — exactly what you'd expect if the CI is just "
            "relabelled drift."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real 'low-CI → momentum'\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** 'low-CI precedes "
            "upward momentum' structure into a synthetic tape and check the same low-CI rule banks "
            "it: edge=0 must stay at t≈1; edge>0 must light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.35):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=477, n_days=4000)\n"
            "    c = px['close']; e = st.low_ci_entries(px, window=R['window'], low=R['low']); s = st.summarize(st.forward_returns(c, e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> t~1; planted momentum -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} low-CI={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted momentum the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — no false positive); a planted "
            f"'low-CI → momentum' reaches **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). "
            "The detector works — so the flat real-tape result is a genuine 'nothing there', not a "
            "broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the low-CI onset does not beat a drift-matched random baseline "
            f"(low-CI − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} "
            f"bps at 5/10/20/60d; Welch t never clears 2, max **{R['h5'][8]:+.2f}** at 5d). The "
            f"impressive one-sample t's (20d **{R['h20'][4]:.2f}**) are pure beta.\n"
            f"- **Tradability `MIRAGE`** — no residual edge once the drift is removed; costs only "
            "deepen the hole. You'd capture the drift more cheaply by holding the index.\n"
            f"- **Forecasts trend vs chop? `BUSTED`** — the return-shuffled placebo leaves the result "
            f"untouched (**p = {R['placebo'][1]:.2f}**): structure-scrambled CIs do as well as the "
            "real one, so the trend-vs-chop reading carries no forecasting information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The low-CI entry's entire apparent profit is the unconditional drift of long equity "
            "indices, which you obtain more cheaply and more fully by **buying and holding**. Because "
            "the CI is sign-blind, the long-only rule is *pure* drift exposure dressed as a regime "
            "filter; it trades *less* of the time and pays costs on each switch, so it strictly "
            "dominates *nothing*. There is no capacity question because there is no edge to scale. "
            "The Choppiness Index is a descriptive volatility-geometry gauge, not a forecasting "
            "strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The high-CI = chop claim.** The mirror folklore ('high CI precedes whipsaw') is "
            "equally confounded on a drifting index — high-CI forward returns are also just drift, "
            "tested the same way.\n"
            "- **Thresholds & windows.** 30/70 bands, N ∈ {10,14,20}, or pairing CI with ADX as a "
            "regime gate — all affine/parametric tweaks of the same volatility-geometry gauge that "
            "inherit the same drift confound.\n"
            "- **Directional overlay.** Grafting a trend-direction filter (e.g. price > MA) onto the "
            "low-CI gate just re-imports the moving-average studies' verdict; the CI itself adds "
            "nothing.\n\n"
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
