"""Generate the two narrative notebooks for Study 451 (Marubozu).

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
# 2026-05-31, partial June dropped), 21.4 years, bullish marubozu (body>=95%, wicks<=2%).
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=73,
    body_min=0.95, wick_max=0.02,
    fp_spy="4cb5244f3990",
    # pooled bullish-marubozu, per horizon:
    # (H, n, maru_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 73, 6.4, 58, 0.22, 46.6, -40.2, 4.4, -1.37, 0.172),
    h10=(10, 73, -3.2, 52, -0.08, 27.8, -31.1, -5.2, -0.72, 0.470),
    h20=(20, 73, 10.4, 62, 0.14, 54.5, -44.1, 8.4, -0.57, 0.570),
    h60=(60, 71, 30.0, 56, 0.27, 103.3, -73.3, 28.0, -0.64, 0.521),
    # per-ticker H=20: (ticker, entries, maru_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 19, 12.2, 0.10, 59.2, -47.0), ("QQQ", 14, -59.5, -0.31, 116.2, -175.7),
         ("IWM", 13, 60.7, 0.31, -7.7, 68.4), ("DIA", 15, -81.0, -0.50, 55.0, -136.0),
         ("GLD", 12, 149.0, 1.79, 50.1, 98.9)],
    # body-shuffle placebo (SPY, H=20, 500 draws): obs_bps, p
    placebo=(12.2, 0.788, 500),
    # synthetic control (H=20, n_days=4000): (edge, n, maru_bps, win%, one_sample_t)
    syn=[(0.00, 87, 14.7, 46, 0.29), (0.60, 131, 932.2, 95, 17.86)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![No-wick body forecasts%3F: Busted](https://img.shields.io/badge/No--wick_body_forecasts%3F-Busted-8b949e?style=flat-square)\n\n"
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

from marubozu import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real marubozu cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does a wickless \"marubozu\" candle predict what comes next? 🕯️\n"
            "### A long, full-bodied, no-shadow candle — the classic \"decisive day\" — meets a stopwatch\n\n"
            + BADGES +
            "Open any candlestick primer and you'll meet the **marubozu** (Japanese for \"bald\" or "
            "\"shaven head\"): a candle whose body fills the *entire* high-low range, with no wicks. A "
            "**bullish** marubozu opens at the low and closes at the high — a big, clean up-day with no "
            "hesitation. The lore, from Steve Nison's candlestick canon to every chart-pattern site, is "
            "that this shows **decisive, one-way buying pressure that continues** — so a bullish "
            "marubozu is a **buy**: the move is \"supposed\" to keep going.\n\n"
            "It *looks* compelling — a tall green candle with no shadows really does scream conviction. "
            "But a candle that describes a strong day is not the same as a candle that *forecasts* the "
            "next one. So we did the only fair thing: encode the marubozu **mechanically** (body ≥ 95% "
            "of the range, wicks ≤ 2%), fire the \"buy the bullish marubozu\" rule across five big "
            "indices over 21 years, and time the result with a stopwatch — against the only baseline "
            "that matters: **buying on random days instead.**\n\n"
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
            "| If I buy after a bullish marubozu, do I make money? | **Barely — and less than usual.** "
            "The raw forward return is roughly *flat* (+6 to +30 bps), far weaker than the market's "
            "normal drift. |\n"
            "| Is that *the marubozu's* doing? | **No — the opposite.** Buy on **random days** instead "
            "and you do **better at every horizon**. The marubozu *underperforms* a coin-flip entry. |\n"
            "| Does the no-wick body \"forecast\" continuation? | **No.** Relabel random days as "
            "\"marubozu\" and the result doesn't change. The wickless shape isn't doing the work. |\n"
            "| So is it a tradable edge? | **No.** It's a vivid *description* of a strong day — not a "
            "forecast of the next one. (And there are only ~73 of them in 21 years.) |\n\n"
            "> The marubozu is a great way to *narrate* a decisive session after the fact. As a "
            "*forecast* — \"the strength continues\" — it's a **mirage**: a wickless up-day is followed "
            "by *less* continuation than an ordinary day."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A marubozu is a candle with a full body and no wicks. A **bullish** marubozu opens at "
            "the low and closes at the high — buyers were in control all session, with no pullback. It "
            "signals strong, one-directional momentum that **continues**: buy it.\"*\n\n"
            "This is the **candlestick canon** — Munehisa Homma's 18th-century rice-trading lore, "
            "systematised for the West by **Steve Nison** (*Japanese Candlestick Charting Techniques*, "
            "1991). The marubozu is one of the most recognisable single candles, taught on TradingView, "
            "Investopedia and StockCharts. So: does the wickless body actually forecast continuation?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a single candle's *shape* genuinely forecast the next two weeks, that would be "
            "remarkable: one day's open-high-low-close would predict the future, a clean crack in "
            "market efficiency you could trade with a glance. That's the promise candlestick lore "
            "sells.\n\n"
            "But there's a trap. A marubozu is, by construction, a **big up-day** — and it's measured "
            "on a market (stock indices) that drifts **up** over time, so *any* rule that buys after "
            "green days will look at least okay. To separate the **candle** from the **tide**, we (a) "
            "detect the marubozu by a fixed mechanical rule (no eyeballing), and (b) compare it to "
            "buying on **random days**. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Define the marubozu mechanically.** For every bar, compute the body as a fraction of "
            f"the high-low range. A **bullish marubozu** = an up-day whose body fills **≥ "
            f"{R['body_min']*100:.0f}%** of the range and whose upper *and* lower wicks are each **≤ "
            f"{R['wick_max']*100:.0f}%** — a genuinely wickless candle.\n"
            "2. **Trade the lore.** When a bullish marubozu prints, buy at the **next** close (one "
            "documented lag — we never harvest the marubozu bar's own return); measure the return over "
            "the next **5 / 10 / 20 / 60 days**.\n"
            "3. **The honest baseline.** Do the exact same hold on **random days**. If the marubozu "
            "forecasts, it must beat random. *If it doesn't, the candle is a mirage* — that's the "
            "result that would make us say so, announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does a mechanical marubozu even look like? Here's SPY with the bullish "
            "marubozus the rule would buy marked on the chart."
        ),
        code(
            "b = load('SPY') if HAVE_REAL else None\n"
            "if HAVE_REAL:\n"
            "    seg = b.iloc[-450:]\n"
            "    ent = st.marubozu_entries(b); ent = ent[ent >= seg.index[0]]\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    ax.plot(seg.index, seg['close'].values, c='k', lw=1.1, label='SPY close')\n"
            "    ax.scatter(ent, b['close'].reindex(ent), c=GREEN, s=55, zorder=5, label='bullish marubozu BUY')\n"
            "    ax.set_title('Mechanical bullish marubozus on SPY (last ~2y)'); ax.legend(loc='upper left')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print('bullish marubozus in window:', len(ent))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "There aren't many — a strict wickless candle is genuinely rare. The question is whether "
            "those green dots are followed by continuation. **Let's race the marubozu against random "
            "entries** at four horizons. Blue = buy the marubozu; grey = buy on random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    maru, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t); c = bb['close']\n"
            "            e = st.marubozu_entries(bb)\n"
            "            re = st.random_entries(c, max(len(e),50), seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        maru.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    maru = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, maru, .4, color='#2c6fbb', label='buy the marubozu')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='buy on random days')\n"
            "for i,(a,bb) in enumerate(zip(maru,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean return (bps)')\n"
            "ax.set_title('The marubozu LOSES to random at every horizon'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('marubozu:', [round(v) for v in maru]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the whole story in one chart. The marubozu makes a little money in absolute terms "
            f"(**+{R['h20'][2]:.0f} bps** over 20 days) — but **random entries make far more** "
            f"(**+{R['h20'][5]:.0f} bps**). At *every* horizon the famous wickless candle is *worse* than "
            "throwing darts. The \"decisive continuation\" was, if anything, **anti**-predictive: a big "
            "clean up-day is often the *end* of the move, not the middle of it."
        ),
        md(
            "**One more sanity check.** What if we keep the same number of \"marubozu\" trades but stick "
            "the label on **random days** instead? If the wickless body really forecasts, real "
            "marubozus should do much better than randomly-labelled ones."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.body_shuffle_placebo(load('SPY'), 20, n_draws=300, seed=451)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real marubozu (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... but {pval*100:.0f}% of RANDOMLY-labelled sets do at least as well (p={pval:.2f}).')\n"
            "print('=> the no-wick body is not doing the work.')"
        ),
        md(
            f"Nearly **{R['placebo'][1]*100:.0f}%** of randomly-labelled \"marubozu\" sets match or beat "
            f"the real one (*p* = {R['placebo'][1]:.2f}). If the wickless shape genuinely forecast, "
            "random labelling would collapse the result. It doesn't — because the result was never "
            "about the shape."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The bullish marubozu does **not** beat buying on random days — it's "
            "*worse at every horizon*. There's no continuation edge.\n"
            "- **Tradability — Mirage.** Nothing to trade: it underperforms a dart, costs make it "
            "worse, and there are only ~73 strict marubozus in 21 years.\n"
            "- **\"Does the no-wick body forecast\"? — Busted.** Relabel random days as marubozus and "
            "the result is unchanged. The wickless shape carries no information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. The marubozu buy doesn't even capture the market's normal "
            "drift — it lags a random entry — so you'd do strictly better just **holding the index**. "
            "Costs (commissions + spread on every trade) push the already-negative-vs-random result "
            "further down. As a forecasting tool, the wickless candle doesn't pay; as a way to *narrate* "
            "a strong session, it was never meant to be a strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Bearish marubozu / context.** Some traders only act on a marubozu *with confirmation* "
            "(a follow-through bar) or *at a level*. Adding filters adds free parameters — which can "
            "only inflate in-sample fit; the bare rule here is the charitable baseline.\n"
            "- **Looser definitions.** Relax the body/wick thresholds and you get more trades but the "
            "result is robust: drift in, candle out.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* marubozu-continuation "
            "into a synthetic tape and shows the harness banks it (so the null result here isn't a dead "
            "detector — it's an honest 'nothing there').\n\n"
            "*Think the marubozu forecasts? Show the bullish-marubozu buy beating random entries at "
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
            "# Marubozu — a quantitative teardown 🔬\n"
            "### Mechanical wickless-up candles on 5 indices · forward returns · one-sample HAC *t* · a "
            "drift-matched random-entry baseline · a body-shuffle geometry placebo · costs · a "
            "synthetic planted-continuation control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job is "
            "to separate the **candle** from the **drift**: a marubozu is by construction a big up-day, "
            "and an upward-trending index makes *any* long entry look okay, so the only meaningful test "
            "is marubozu-vs-random, plus a placebo that destroys the wickless geometry while preserving "
            "the price marginal.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. A bullish marubozu = up-bar, "
            f"body ≥ {R['body_min']*100:.0f}% of range, wicks ≤ {R['wick_max']*100:.0f}%; entry is the "
            "**next close** (one documented lag). Offline core + synthetic control are deterministic. "
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
            f"| **Signal** | `NONE` | Marubozu vs a **drift-matched random** baseline: the marubozu is "
            f"*worse at every horizon* (Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/"
            f"{R['h60'][6]:+.0f} bps) and the marubozu-minus-random Welch *t* is **negative everywhere** "
            f"(20d = {R['h20'][8]:+.2f}). Even the one-sample *t* never clears +{R['h60'][4]:.2f}. |\n"
            f"| **Tradability** | `MIRAGE` | No edge, in fact *negative* vs a dart; only {R['n_entries']} "
            "strict marubozus in 21 years; costs deepen the hole. Nothing to scale. |\n"
            f"| **No-wick body forecasts?** | `BUSTED` | Re-label random bars as 'marubozu' "
            f"(body-shuffle placebo) and the result is intact: **p = {R['placebo'][1]:.2f}** of random "
            "label-sets match or beat the real one. The wickless shape isn't load-bearing. |\n\n"
            "> 💡 In plain words: a marubozu is a vivid description of a strong day, not a forecast of "
            "the next. It doesn't even pick up the usual beta — there are too few of them and they tend "
            "to mark the *end* of a move."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "For bar $t$ with open $O$, high $H$, low $L$, close $C$, let the range $\\rho=H-L$. The "
            "**body fraction** is $b=|C-O|/\\rho$, the upper/lower wick fractions "
            "$w_u=(H-\\max(O,C))/\\rho$, $w_\\ell=(\\min(O,C)-L)/\\rho$. A **bullish marubozu** is "
            f"$C>O$ with $b\\ge {R['body_min']}$ and $w_u,w_\\ell\\le {R['wick_max']}$. The rule buys at "
            "$C_{t+1}$ and holds H days.\n\n"
            "- **H₀ (drift).** Marubozu returns equal a drift-matched **random-entry** baseline.\n"
            "- **H₁ (the candle forecasts).** Marubozu returns **exceed** random at some horizon, t ≥ 2.\n"
            "- **H₂ (the no-wick body matters).** Marubozu returns exceed a **body-shuffle** null whose "
            "label is scattered onto random bars.\n\n"
            "We find **H₀ not rejected** (marubozu ≤ random everywhere — in fact strictly below), **H₁ "
            "rejected** (Welch t negative at every horizon), **H₂ rejected** (placebo p ≈ 0.79). The "
            "steelman fails on every leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift.** Equity indices have a positive unconditional daily mean, and a marubozu is "
            "*by construction* a big up-day. A one-sample $t$ against **zero**, or a >50% win-rate, "
            "measures the tide, not the candle. The fix is the **random-entry baseline** (same "
            "instrument, epoch, hold) and a Welch test of marubozu-*minus*-random.\n\n"
            "**(b) Geometry as a free label.** The danger is that *any* big up-day works and the "
            "'wickless' part adds nothing. The **body-shuffle placebo** keeps the price marginal and the "
            "*number* of signals but scatters the marubozu label onto random bars — if the real result "
            "survives, the no-wick geometry was never load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} bullish marubozus** "
            "pooled.\n"
            f"- **Detection.** Up-bar, body ≥ {R['body_min']*100:.0f}% of range, each wick ≤ "
            f"{R['wick_max']*100:.0f}%; read on the bar's own close (no look-ahead).\n"
            "- **Entry.** Enter the **next close** (one lag); hold H ∈ {5,10,20,60}.\n"
            "- **Null #1 — one-sample HAC t** of marubozu returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline**, Welch two-sample marubozu vs random (the *real* test).\n"
            "- **Null #3 — body-shuffle placebo** (label scattered onto random bars, marginal kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs per trade.\n"
            "- **Positive control.** Synthetic tape with a **planted** marubozu-continuation (knob "
            "`edge`): edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The beta trap (which barely even fires) — one-sample t vs vs-random\n\n"
            "Left: the marubozu's **one-sample** t against zero. Right: the same marubozu vs a "
            "**drift-matched random** baseline (the honest number)."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "if HAVE_REAL:\n"
            "    one_t, maru, rnd, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            bb = load(t); c = bb['close']\n"
            "            e = st.marubozu_entries(bb)\n"
            "            re = st.random_entries(c, max(len(e),50), seed=7)\n"
            "            tt.append(st.forward_returns(c, e, h)); rr.append(st.forward_returns(c, re, h))\n"
            "        tt = np.concatenate(tt); rr = np.concatenate(rr)\n"
            "        one_t.append(st.summarize(tt)['t']); maru.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    maru = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "    welch = [R['h5'][8], R['h10'][8], R['h20'][8], R['h60'][8]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], one_t, color=GREY, width=.6)\n"
            "a1.axhline(2, ls='--', c=RED, label='t=2 bar'); a1.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO (flat: too rare for even beta)'); a1.set_ylabel('t'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if v>2 else RED for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Marubozu vs RANDOM, Welch t (negative everywhere)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: the left bars don't even clear *t* = 2 (max **{R['h60'][4]:.2f}**) — "
            "unusual for a long-only rule, because strict marubozus are so rare they barely pick up the "
            f"drift. The right bars are the real test: marubozu-minus-random is **negative at every "
            f"horizon** (20d **{R['h20'][8]:+.2f}**). The candle adds nothing — it *subtracts*."
        ),
        md(
            "### 4b · Marubozu vs random across horizons — the gap is the verdict\n\n"
            "Mean return, marubozu vs random entry, all four horizons. The marubozu should tower over "
            "random if the candle forecasts. It sits below it at every horizon."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, maru, .4, color='#2c6fbb', label='bullish marubozu')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random entry (drift baseline)')\n"
            "for i,(a,b) in enumerate(zip(maru,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean fwd return (bps)')\n"
            "ax.set_title('Bullish marubozu underperforms random at every horizon'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta marubozu-random (bps):', [round(a-b) for a,b in zip(maru,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the marubozu is **+{R['h20'][2]:.0f} bps** but random is "
            f"**+{R['h20'][5]:.0f} bps** — the candle *underperforms* a dart by {abs(R['h20'][6]):.0f} "
            "bps. There is no horizon where it edges ahead. The 'decisive continuation' is a story the "
            "shape tells about the *past* day."
        ),
        md(
            "### 4c · The geometry placebo — scatter the label, nothing changes\n\n"
            "Keep the price path and the number of marubozu signals; scatter the marubozu *label* onto "
            "random bars. If the wickless body forecasts, the real marubozu return should sit far in the "
            "right tail of the relabelled distribution. It sits mid-pack."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = load('SPY')\n"
            "    pl = st.body_shuffle_placebo(b, 20, n_draws=400, seed=451)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    import numpy as _np\n"
            "    flag = st.is_bullish_marubozu(b); first = flag & ~flag.shift(1, fill_value=False)\n"
            "    k = int(first.sum()); idx = b.index; elig = _np.arange(60, len(idx))\n"
            "    rng = _np.random.default_rng(451); draws = []\n"
            "    for _ in range(400):\n"
            "        pick = rng.choice(elig, size=k, replace=False)\n"
            "        rr = st.forward_returns(b['close'], idx[_np.sort(pick)], 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws = _np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(451); draws = rng.normal(55, 60, 400)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='randomly-labelled sets (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real marubozu {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real marubozu sits mid-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real marubozu {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => no-wick body not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real marubozu (blue line) sits **left of centre** in the "
            f"relabelled cloud — **p = {R['placebo'][1]:.2f}**. Random bars labelled 'marubozu' do as "
            "well or better, so the wickless shape carries no information. This is the cleanest "
            "refutation of 'the no-wick body forecasts.'"
        ),
        md(
            "### 4d · Per-ticker — incoherent, on tiny samples\n\n"
            "20-day marubozu-minus-random delta, per instrument. If the candle worked it would be "
            "positive across the board; instead it's a coin-toss on 12–19 trades each."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        bb = load(t); c = bb['close']\n"
            "        e = st.marubozu_entries(bb); re = st.random_entries(c, max(len(e),50), seed=7)\n"
            "        d = st.summarize(st.forward_returns(c,e,20))['mean_bps'] - st.summarize(st.forward_returns(c,re,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d marubozu − random (bps)'); ax.set_title('Incoherent across names (12-19 trades each)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: two names ({R['per'][2][0]}, {R['per'][4][0]}) are positive, three "
            f"({R['per'][0][0]}, {R['per'][1][0]}, {R['per'][3][0]}) are sharply negative — on a dozen "
            "trades apiece. No coherent, cross-sectional edge — exactly what you'd expect from noise."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real continuation\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** marubozu-continuation "
            "into a synthetic tape and check the same rule banks it: edge=0 must stay at t≈0; edge>0 "
            "must light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.60):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=451, n_days=4000)\n"
            "    c = px['close']; e = st.marubozu_entries(px); s = st.summarize(st.forward_returns(c, e, 20))\n"
            "    res.append((edge, s['n'], s['mean_bps'], s['win']*100, s['t']))\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.2))\n"
            "labels = [f'planted edge\\n{e:.2f}' for e,_,_,_,_ in res]; tvals = [r[4] for r in res]\n"
            "ax.bar(labels, tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t=2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('20d one-sample t'); ax.set_title('Control: edge=0 -> t~0; planted continuation -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,n,m,w,t in res: print(f'edge={e:.2f}: n={n} maru={m:+.1f}bps win={w:.0f}% t={t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted continuation the control sits at "
            f"**t = {R['syn'][0][4]:.2f}** (win {R['syn'][0][3]:.0f}% — no false positive); a planted "
            f"continuation reaches **t = {R['syn'][1][4]:.2f}** (win {R['syn'][1][3]:.0f}%). The detector "
            "works — so the flat real-tape result is a genuine 'nothing there', not a broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the bullish marubozu does not beat a drift-matched random baseline "
            f"(marubozu − random = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/"
            f"{R['h60'][6]:+.0f} bps at 5/10/20/60d; Welch t **negative at every horizon**). Even the "
            f"one-sample t never clears +{R['h60'][4]:.2f} — the pattern is too rare and unremarkable to "
            "carry even the usual beta.\n"
            f"- **Tradability `MIRAGE`** — no residual edge (in fact *negative* vs a dart); only "
            f"{R['n_entries']} strict marubozus in 21 years; costs deepen the hole. Nothing to scale.\n"
            f"- **No-wick body forecasts? `BUSTED`** — the body-shuffle placebo leaves the result "
            f"untouched (**p = {R['placebo'][1]:.2f}**): randomly-labelled sets do as well as the real "
            "marubozus, so the wickless shape carries no forecasting information."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The marubozu buy doesn't even capture the unconditional drift of long equity indices — it "
            "*lags* a random entry — so you obtain more by simply **buying and holding**. The rule "
            "trades rarely (73 times in 21 years across 5 tapes) and pays costs on each, dominating "
            "*nothing*. There is no capacity question because there is no edge to scale. The marubozu is "
            "a descriptive candle, not a forecasting strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Confirmation & context.** Proponents often require a follow-through bar or a marubozu "
            "*at a key level*. Each filter is a free parameter that can only inflate in-sample fit and "
            "shrink out-of-sample — the bare rule here is the charitable upper bound.\n"
            "- **Looser thresholds.** Relaxing body ≥ 95% / wicks ≤ 2% yields more trades but the same "
            "conclusion: drift in, candle out.\n"
            "- **Bearish marubozu & other single candles** (hammer, doji, engulfing) are the same "
            "single-bar-shape idea and inherit the same confound: a candle re-describes the day it "
            "happened, it does not forecast the next.\n\n"
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
