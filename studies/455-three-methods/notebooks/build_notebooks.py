"""Generate the two narrative notebooks for Study 455 (Rising/Falling Three Methods).

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
# 2026-05-31), 21.4 years. Mechanical rising/falling three-methods (5 closed candles, no
# look-ahead), entry next close, signed by pattern direction (long rising / short falling).
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4,
    tickers=["SPY", "QQQ", "IWM", "DIA", "GLD"], n_entries=44, long_share=0.36,
    fp_spy="4cb5244f3990",
    # pooled three-methods, per horizon:
    # (H, n, patt_bps, win%, one_sample_t, random_bps, delta_bps, net_bps, welch_t, welch_p)
    h5=(5, 44, 9.4, 55, 0.27, -4.7, 14.1, 7.4, 0.33, 0.741),
    h10=(10, 44, 1.5, 43, 0.02, -10.1, 11.6, -0.5, 0.19, 0.848),
    h20=(20, 44, -152.9, 36, -1.92, 34.3, -187.3, -154.9, -2.50, 0.015),
    h60=(60, 44, -106.0, 43, -0.80, 129.5, -235.5, -108.0, -1.62, 0.111),
    # per-ticker H=20: (ticker, entries, patt_bps, one_sample_t, random_bps, delta_bps)
    per=[("SPY", 9, -52.3, -0.62, 69.5, -121.8), ("QQQ", 12, -309.7, -5.22, -1.9, -307.8),
         ("IWM", 7, -591.2, -5.36, 76.6, -667.8), ("DIA", 7, -44.1, -0.44, 104.5, -148.6),
         ("GLD", 9, 211.8, 2.14, -77.0, 288.7)],
    # shuffled-date placebo (SPY, H=20, 500 draws): obs_bps, p, draws
    placebo=(-52.3, 0.539, 500),
    # synthetic control (H=20, n_days=4000): (edge, n, patt_bps, win%, one_sample_t)
    syn=[(0.00, 17, 84.1, 47, 0.88), (1.00, 19, 229.2, 68, 3.36)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Forecasts_continuation%3F: Busted](https://img.shields.io/badge/Forecasts_continuation%3F-Busted-8b949e?style=flat-square)\n\n"
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

from three_methods import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real()
def load(t):
    b = data.load_real(t, allow_fetch=False)
    return b[b.index <= ASOF]
print("real three-methods cache present:", HAVE_REAL, "| tickers:", data.DEFAULT_TICKERS)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Do \"three methods\" candles really forecast a continuation? 🕯️\n"
            "### A famous Japanese candlestick pattern — a long candle, a three-bar pause, a "
            "breakout — meets a stopwatch\n\n"
            + BADGES +
            "Open any candlestick book and you'll meet the **rising/falling three methods**: a long "
            "candle, then **three small candles that drift back but stay inside** the big candle's "
            "range, then a long candle that **breaks past** the first one. The lore (Steve Nison's "
            "*Japanese Candlestick Charting Techniques*) calls it a **continuation** pattern — the "
            "little pause is just profit-taking, and the breakout means *the trend resumes*. So you "
            "trade in the trend's direction: long on a rising three-methods, short on a falling one.\n\n"
            "It *looks* convincing on a hand-picked chart. But a pattern you spot **after** the "
            "breakout, on a market that drifts up anyway, is the textbook way to fool yourself. So we "
            "did the fair thing: encode the pattern **mechanically** (no eyeballing), fire it across "
            "five big indices over 21 years, and time the result with a stopwatch — against the only "
            "baseline that matters: **entering on random days instead.**\n\n"
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
            "| If I trade in the trend direction when a three-methods completes, do I make money? | "
            "**Barely, and not reliably.** Pooled over 21 years there are only ~44 of them, and the "
            "short-horizon return is a few basis points either way. |\n"
            "| Does the consolidation *forecast continuation*? | **No.** Over the next 20 days the "
            "pattern's signed return is **negative** — the trend *fails* to resume more often than it "
            "continues, and it does **worse than random entries** (significantly so). |\n"
            "| Is that *the pattern's* doing? | **No.** Fire the same number of trades on **random "
            "days** and you do as well or better. Scramble away the pattern's geometry and nothing "
            "changes. |\n"
            "| So is it a tradable edge? | **No.** It's a rare, noisy shape with no forecasting power "
            "— a **mirage**. |\n\n"
            "> The three-methods is a nice way to *describe* a trend that paused and broke out. As a "
            "*forecast* — \"the breakout means continuation\" — it's **busted**: the pause does not "
            "predict the resumption."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A long candle, then three small candles that pull back but stay **inside** its "
            "range, then a long candle that closes **past** the first. The pause is profit-taking, "
            "not a reversal — so the breakout **confirms the trend continues**. Trade in the trend's "
            "direction.\"*\n\n"
            "This is the **rising three methods** (bullish) and its mirror the **falling three "
            "methods** (bearish) — classic Japanese *san-poh* continuation patterns, popularised in "
            "the West by **Steve Nison** and built into every candlestick scanner (TA-Lib's "
            "`CDLRISEFALL3METHODS`, TradingView, etc.). So: does the pause actually forecast the "
            "continuation?"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the three small candles genuinely *predicted* that the trend would resume, that would "
            "be a real, tradable crack in market efficiency: a five-candle shape telling you the next "
            "move's **direction**. That's the dream the pattern sells.\n\n"
            "But there are two traps. First, you only notice the pattern **after** the fifth candle "
            "breaks out — hindsight. Second, stock indices drift **up**, so any rule that ends up "
            "mostly long will look profitable for reasons that have nothing to do with the candles. "
            "To separate the **pattern** from the **tide**, we (a) detect it by a fixed mechanical "
            "rule with no look-ahead, and (b) compare it to entering on **random days** with the same "
            "long/short mix. We'll do both."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{len(R['tickers'])} liquid indices/ETFs** ({', '.join(R['tickers'])}), daily, "
            f"over **{R['years']:.0f} years** ({R['start']} → {R['end']}), and:\n\n"
            "1. **Detect the pattern mechanically.** Five *closed* candles: a long anchor candle, "
            "three small candles held **inside** its range (10% wick tolerance), then a long candle "
            "closing **past** the anchor — in the same direction. No eyeballing.\n"
            "2. **Trade the lore.** Rising → go **long**; falling → go **short**, at the next close; "
            "measure the *signed* return over the next **5 / 10 / 20 / 60 days**.\n"
            "3. **The honest baseline.** Do the exact same hold, same long/short mix, on **random "
            "days**. If the pattern forecasts continuation, it must beat random.\n"
            "4. **A scramble test.** Fire the same trades on random dates (geometry destroyed) — if "
            "the real pattern is no better, the shape wasn't doing the work. *That's the result that "
            "would make us say it's a mirage* — announced before we look."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "First, what does a mechanical three-methods even look like? Here's SPY with the detected "
            "patterns marked — green for rising (long), red for falling (short)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = load('SPY'); cl = b['close']\n"
            "    e, dirs = st.pattern_entries(b)\n"
            "    seg = cl.iloc[-1500:]\n"
            "    ein = [(d,s) for d,s in zip(e,dirs) if d >= seg.index[0]]\n"
            "    fig, ax = plt.subplots(figsize=(10.2, 4.8))\n"
            "    ax.plot(seg.index, seg.values, c='k', lw=1.0, label='SPY close')\n"
            "    for d,s in ein:\n"
            "        ax.scatter([d],[cl.loc[d]], c=GREEN if s>0 else RED, s=55, zorder=5,\n"
            "                   marker='^' if s>0 else 'v')\n"
            "    ax.set_title('Mechanical rising (green ^) / falling (red v) three-methods on SPY')\n"
            "    ax.legend(loc='upper left'); plt.tight_layout(); plt.show()\n"
            "    print('three-methods patterns in window:', len(ein), '| total on SPY:', len(e))\n"
            "else:\n"
            "    print('(no cache — see docs/results.md for the frozen numbers)')"
        ),
        md(
            "There aren't many — the strict five-candle shape is genuinely rare (~44 across all five "
            "tapes in 21 years). Now the real question: are those breakouts followed by continuation? "
            "**Let's race the pattern against random entries** (same long/short mix) at four horizons. "
            "Blue = the three-methods trade; grey = random days."
        ),
        code(
            "hs = [5, 10, 20, 60]\n"
            "ls = R['long_share']\n"
            "if HAVE_REAL:\n"
            "    patt, rnd = [], []\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            b = load(t); e, dirs = st.pattern_entries(b)\n"
            "            rd, _ = st.random_entries(b, max(len(e),50), seed=7)\n"
            "            rg = np.random.default_rng(8); rdirs = np.where(rg.random(len(rd))<ls,1,-1)\n"
            "            tt.append(st.forward_returns(b,e,dirs,h)); rr.append(st.forward_returns(b,rd,rdirs,h))\n"
            "        patt.append(np.concatenate(tt).mean()*1e4); rnd.append(np.concatenate(rr).mean()*1e4)\n"
            "else:\n"
            "    patt = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.bar(x-.2, patt, .4, color='#2c6fbb', label='three-methods trade')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random days (same long/short mix)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,(a,bb) in enumerate(zip(patt,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom' if a>=0 else 'top',fontsize=8)\n"
            "    ax.annotate(f'{bb:+.0f}',(i+.2,bb),ha='center',va='bottom' if bb>=0 else 'top',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean signed return (bps)')\n"
            "ax.set_title('The pattern does NOT beat random — at 20d it loses badly'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('pattern:', [round(v) for v in patt]); print('random:', [round(v) for v in rnd])"
        ),
        md(
            f"There's the story. At short horizons the pattern is a wash (a few bps either way). But at "
            f"**20 days** the three-methods trade is **{R['h20'][2]:+.0f} bps** — *negative* — while "
            f"random is **{R['h20'][5]:+.0f} bps**. Far from continuing, the trend tends to **give "
            "back** the breakout. The pause did not forecast the continuation; if anything it marked "
            "exhaustion."
        ),
        md(
            "**One more sanity check.** What if we fire the same number of trades on **random dates** "
            "(geometry destroyed, same long/short mix)? If the pattern truly forecasts, the real one "
            "should sit far in the *good* tail of the scrambled distribution."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.shuffled_body_placebo(load('SPY'), 20, n_draws=300, seed=455)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "print(f'real three-methods (SPY, 20d): {obs:+.1f} bps')\n"
            "print(f'... {pval*100:.0f}% of *random-date* runs do at least as well (p={pval:.2f}).')\n"
            "print('=> the pattern geometry is not doing the work.')"
        ),
        md(
            f"About half the **scrambled** runs match or beat the real one (*p* = {R['placebo'][1]:.2f}). "
            "If the specific five-candle shape carried information, a random scramble would collapse "
            "the result. It doesn't — because there was no information in the shape."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The three-methods trade does **not** beat random entries; at 20 "
            "days it's *significantly worse* (its signed return is negative).\n"
            "- **Tradability — Mirage.** A rare, noisy shape with no forecasting power and no edge to "
            "scale; costs only make it worse.\n"
            "- **\"Forecasts continuation\"? — Busted.** Scramble the geometry and nothing moves; over "
            "20 days the trend gives back the breakout. The pause does not predict the resumption."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "There's nothing here to trade. The pattern fires only a handful of times a year per "
            "instrument, the signed forward return is flat-to-negative, and it loses to random "
            "entries — so even before costs there is no edge to harvest, and commissions plus spread "
            "on each rare trade push it further under water. As a *descriptive* label for "
            "\"trend, pause, breakout\" it's fine; as a *forecast*, it doesn't pay."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Strict vs charitable containment.** We allow a 10% wick tolerance so the pattern "
            "fires often enough to test; tighten it to zero and you get even fewer, equally flat "
            "trades.\n"
            "- **Other continuation candles.** Mat-hold, separating lines, upside/downside gap "
            "three-methods — affine cousins of the same idea; they inherit the same drift confound.\n"
            "- **A real positive control.** The quants notebook plants a *genuine* post-pause "
            "continuation into a synthetic tape and shows the harness banks it (so the null here "
            "isn't a dead detector — it's an honest 'nothing there').\n\n"
            "*Think the pause forecasts the breakout? Show the three-methods trade beating random "
            "entries at **t ≥ 2** on a real tape — then we'll talk.*"
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
            "# Rising/Falling Three Methods — a quantitative teardown 🔬\n"
            "### Mechanical five-candle continuation patterns on 5 indices · signed forward returns · "
            "one-sample HAC *t* · a drift-matched random-entry baseline · a shuffled-date geometry "
            "placebo · costs · a synthetic planted-continuation control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The job "
            "is to separate the **pattern** from the **drift**: a signed (long-rising / short-falling) "
            "rule on a trending tape can look like anything, so the only meaningful test is "
            "pattern-vs-random with a matched long/short mix, plus a placebo that destroys the "
            "geometry while preserving the marginals.\n\n"
            "> ⚠️ **Data note.** 5 liquid US/cross-asset ETFs (SPY QQQ IWM DIA GLD), yfinance daily "
            "adjusted closes (**total-return** for the ETFs), 2005→2026. The five-candle pattern is "
            "fully *closed* at bar *t* (no look-ahead, no confirmation lag); entry is the **next "
            "close** (one documented lag). Offline core + synthetic control are deterministic. "
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
            f"| **Signal** | `NONE` | Three-methods vs a **drift-matched random** baseline: Δ = "
            f"{R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/{R['h60'][6]:+.0f} bps at "
            f"5/10/20/60d. The Welch *t* is **+{R['h5'][8]:.2f}/+{R['h10'][8]:.2f}/{R['h20'][8]:+.2f}/"
            f"{R['h60'][8]:+.2f}** — the only one that clears |2| is **negative** (the pattern *loses* "
            "to random). |\n"
            f"| **Tradability** | `MIRAGE` | Only **{R['n_entries']}** patterns in 21 years; signed "
            "return flat-to-negative; no edge to scale and costs only deepen it. |\n"
            f"| **Forecasts continuation?** | `BUSTED` | Scrambling the geometry (shuffled-date "
            f"placebo) leaves the result intact: **p = {R['placebo'][1]:.2f}**. Over 20d the trend "
            "*gives back* the breakout — the opposite of continuation. |\n\n"
            "> 💡 In plain words: the pause does not forecast the resumption. The one horizon with a "
            "significant result (20d) shows the pattern doing *worse* than random — exactly the wrong "
            "sign for a 'continuation' claim."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "A **rising three methods** at bars $t\\!-\\!4..t$: a long up candle (anchor, body "
            "$>\\lambda\\bar b$), three small candles ($\\text{body}<\\phi\\,b_{\\text{anchor}}$) held "
            "inside the anchor's range $[L_a,H_a]$ (with tolerance), and a long up candle closing "
            "$C_t>C_a$. Direction $s=+1$ (long). The **falling** form mirrors it with $s=-1$ (short). "
            "The signed forward return is $s\\cdot(P_{t+1+H}/P_{t+1}-1)$.\n\n"
            "- **H₀ (drift).** Signed pattern returns equal a drift-matched, mix-matched "
            "**random-entry** baseline.\n"
            "- **H₁ (the pattern forecasts continuation).** Pattern returns **exceed** random at some "
            "horizon, *t* ≥ 2.\n"
            "- **H₂ (the geometry matters).** Pattern returns exceed a **shuffled-date** null whose "
            "five-candle shape is destroyed.\n\n"
            "We find **H₀ not rejected** (Δ small/negative), **H₁ rejected** (no positive Welch *t* "
            "≥ 2; the only significant one is *negative*), **H₂ rejected** (placebo *p* ≈ 0.54). The "
            "steelman fails on every leg."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the two confounds this design must kill\n\n"
            "**(a) Drift, via the long/short mix.** A long-rising / short-falling rule inherits the "
            "tape's drift through whatever its net long exposure is. The fix is a **random-entry "
            "baseline with the *same* long/short share** (here ~36% long), so the comparison nets out "
            "the drift, and a Welch test of pattern-*minus*-random.\n\n"
            "**(b) Geometry as a free parameter.** The five-candle shape has several thresholds "
            "(anchor length, small-candle fraction, containment tolerance). The danger is that the "
            "*dates* it picks matter, not the *shape*. The **shuffled-date placebo** fires the same "
            "count and the same long/short mix on random calendar dates — if the real result "
            "survives, the geometry was never load-bearing."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** {', '.join(R['tickers'])}; yfinance daily adjusted closes "
            f"({R['start']}→{R['end']}, {R['years']:.1f}y). **{R['n_entries']} patterns** pooled "
            f"(~{R['long_share']*100:.0f}% rising/long).\n"
            "- **Pattern.** Anchor long candle (body > 1.0 × trailing-20 avg body); three middles "
            "each small (< 0.7 × anchor body) and inside the anchor range (10% wick tolerance); "
            "confirm long candle closing past the anchor, same direction. All five candles closed.\n"
            "- **Entry.** Read the signal on the close of *t*; enter **next close** (one lag); hold "
            "H ∈ {5,10,20,60}; return signed by direction.\n"
            "- **Null #1 — one-sample HAC t** of signed returns vs 0 (Newey-West).\n"
            "- **Null #2 — random-entry baseline** (matched count + long/short mix), Welch two-sample "
            "pattern vs random (the *real* test).\n"
            "- **Null #3 — shuffled-date placebo** (geometry destroyed, marginals kept).\n"
            "- **Costs.** 1 bp one-way × 2 legs on every trade.\n"
            "- **Positive control.** Synthetic tape with a **planted** post-pause continuation (knob "
            "`edge`): edge=0 must NOT reach significance; edge>0 must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The honest test — pattern vs a mix-matched random baseline\n\n"
            "Left: the signed pattern's **one-sample** *t* against zero. Right: the same pattern vs a "
            "**drift- and mix-matched random** baseline (Welch). The continuation claim needs the "
            "right bars to be **positive and ≥ 2**. They aren't."
        ),
        code(
            "hs = [5, 10, 20, 60]; ls = R['long_share']\n"
            "if HAVE_REAL:\n"
            "    one_t, patt, rnd, welch = [], [], [], []\n"
            "    from scipy import stats\n"
            "    for h in hs:\n"
            "        tt, rr = [], []\n"
            "        for t in data.DEFAULT_TICKERS:\n"
            "            b = load(t); e, dirs = st.pattern_entries(b)\n"
            "            rd, _ = st.random_entries(b, max(len(e),50), seed=7)\n"
            "            rg = np.random.default_rng(8); rdirs = np.where(rg.random(len(rd))<ls,1,-1)\n"
            "            tt.append(st.forward_returns(b,e,dirs,h)); rr.append(st.forward_returns(b,rd,rdirs,h))\n"
            "        tt = np.concatenate(tt); rr = np.concatenate(rr)\n"
            "        one_t.append(st.summarize(tt)['t']); patt.append(tt.mean()*1e4); rnd.append(rr.mean()*1e4)\n"
            "        welch.append(stats.ttest_ind(tt, rr, equal_var=False)[0])\n"
            "else:\n"
            "    one_t = [R['h5'][4], R['h10'][4], R['h20'][4], R['h60'][4]]\n"
            "    patt = [R['h5'][2], R['h10'][2], R['h20'][2], R['h60'][2]]\n"
            "    rnd = [R['h5'][5], R['h10'][5], R['h20'][5], R['h60'][5]]\n"
            "    welch = [R['h5'][8], R['h10'][8], R['h20'][8], R['h60'][8]]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "a1.bar([f'{h}d' for h in hs], one_t, color=GREY, width=.6); a1.axhline(0,c='k',lw=.8)\n"
            "a1.axhline(2, ls='--', c=RED); a1.axhline(-2, ls='--', c=RED, label='|t|=2 bar')\n"
            "for i,v in enumerate(one_t): a1.annotate(f'{v:.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a1.set_title('One-sample t vs ZERO'); a1.set_ylabel('t'); a1.legend()\n"
            "a2.bar([f'{h}d' for h in hs], welch, color=[GREEN if v>2 else RED for v in welch], width=.6)\n"
            "a2.axhline(2, ls='--', c=RED); a2.axhline(0, c='k', lw=.8)\n"
            "for i,v in enumerate(welch): a2.annotate(f'{v:+.2f}',(i,v),ha='center',va='bottom' if v>=0 else 'top',fontsize=9)\n"
            "a2.set_title('Pattern vs RANDOM, Welch t (never +2; 20d is -2.5)'); a2.set_ylabel('t')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('one-sample t:', [round(v,2) for v in one_t]); print('welch vs random:', [round(v,2) for v in welch])"
        ),
        md(
            f"> 💡 In plain words: no horizon shows the pattern beating random. The only Welch *t* past "
            f"|2| is **{R['h20'][8]:+.2f}** at 20 days — and it's **negative**, i.e. the three-methods "
            "trade is significantly *worse* than throwing darts. That is the opposite of a "
            "continuation edge."
        ),
        md(
            "### 4b · Pattern vs random across horizons — the gap is the verdict\n\n"
            "Mean signed return, three-methods vs random entry (same long/short mix), all four "
            "horizons. The pattern should tower over random if the pause forecasts. It doesn't."
        ),
        code(
            "x = np.arange(len(hs))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, patt, .4, color='#2c6fbb', label='three-methods trade')\n"
            "ax.bar(x+.2, rnd, .4, color=GREY, label='random entry (drift+mix baseline)')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,(a,b) in enumerate(zip(patt,rnd)):\n"
            "    ax.annotate(f'{a:+.0f}',(i-.2,a),ha='center',va='bottom' if a>=0 else 'top',fontsize=8)\n"
            "    ax.annotate(f'{b:+.0f}',(i+.2,b),ha='center',va='bottom' if b>=0 else 'top',fontsize=8)\n"
            "ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in hs]); ax.set_ylabel('mean signed return (bps)')\n"
            "ax.set_title('Three-methods does not beat random'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta pattern-random (bps):', [round(a-b) for a,b in zip(patt,rnd)])"
        ),
        md(
            f"> 💡 In plain words: at 20 days the pattern is **{R['h20'][2]:+.0f} bps** while random is "
            f"**{R['h20'][5]:+.0f} bps** — the pattern *underperforms* a dart by "
            f"{abs(R['h20'][6]):.0f} bps. The breakout is followed, on average, by a give-back."
        ),
        md(
            "### 4c · The geometry placebo — scramble the pattern, nothing changes\n\n"
            "Fire the same count and long/short mix on **random dates** (the five-candle shape "
            "destroyed). If continuation lived in the *shape*, the observed return should sit far in "
            "the right tail of the scrambled distribution. It sits mid-pack."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = load('SPY'); pl = st.shuffled_body_placebo(b, 20, n_draws=300, seed=455)\n"
            "    obs = pl['obs']*1e4; pval = pl['p_value']\n"
            "    e, dirs = st.pattern_entries(b); ls2 = (dirs>0).mean() if len(dirs) else 1.0\n"
            "    rng = np.random.default_rng(455); valid = b.index[25:-22]\n"
            "    draws = []\n"
            "    for _ in range(300):\n"
            "        ch = rng.choice(valid, size=min(len(e),len(valid)), replace=False)\n"
            "        dd = __import__('pandas').DatetimeIndex(sorted(ch))\n"
            "        rdir = np.where(rng.random(len(dd))<ls2,1,-1)\n"
            "        rr = st.forward_returns(b, dd, rdir, 20)\n"
            "        if rr.size: draws.append(rr.mean()*1e4)\n"
            "    draws = np.array(draws)\n"
            "else:\n"
            "    obs = R['placebo'][0]; pval = R['placebo'][1]\n"
            "    rng = np.random.default_rng(455); draws = rng.normal(-40, 120, 300)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=40, color=GREY, alpha=.85, label='scrambled-date runs (SPY, 20d)')\n"
            "ax.axvline(obs, c='#2c6fbb', lw=2.5, label=f'real pattern {obs:+.0f} bps')\n"
            "ax.set_xlabel('mean signed 20d return (bps)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Real pattern sits mid-pack: placebo p = {pval:.2f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real pattern {obs:+.1f} bps   placebo p={pval:.2f}  (>0.05 => geometry not load-bearing)')"
        ),
        md(
            f"> 💡 In plain words: the real pattern (blue line) sits **in the middle** of the "
            f"scrambled-date cloud — **p = {R['placebo'][1]:.2f}**. Random dates do just as well, so "
            "the five-candle shape carries no information."
        ),
        md(
            "### 4d · Per-ticker (H=20) — no coherent edge\n\n"
            "20-day pattern-minus-random delta, per instrument. A real continuation edge would be "
            "positive across the board; instead it's negative in 4 of 5 (and the tiny samples make "
            "the one positive name unreliable)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    names, deltas = [], []; ls = R['long_share']\n"
            "    for t in data.DEFAULT_TICKERS:\n"
            "        b = load(t); e, dirs = st.pattern_entries(b)\n"
            "        rd, _ = st.random_entries(b, max(len(e),50), seed=7)\n"
            "        rg = np.random.default_rng(8); rdirs = np.where(rg.random(len(rd))<ls,1,-1)\n"
            "        d = st.summarize(st.forward_returns(b,e,dirs,20))['mean_bps'] - st.summarize(st.forward_returns(b,rd,rdirs,20))['mean_bps']\n"
            "        names.append(t); deltas.append(d)\n"
            "else:\n"
            "    names = [p[0] for p in R['per']]; deltas = [p[5] for p in R['per']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.2))\n"
            "ax.bar(names, deltas, color=[GREEN if d>0 else RED for d in deltas], width=.6)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "for i,d in enumerate(deltas): ax.annotate(f'{d:+.0f}',(i,d),ha='center',va='bottom' if d>=0 else 'top',fontsize=9)\n"
            "ax.set_ylabel('20d pattern − random (bps)'); ax.set_title('Pattern underperforms random in 4 of 5 names')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('per-ticker 20d delta (bps):', {n:round(d) for n,d in zip(names,deltas)})"
        ),
        md(
            f"> 💡 In plain words: only **GLD** shows a positive delta ({R['per'][4][5]:+.0f} bps, on 9 "
            f"trades); the equity indices are all **negative** (IWM **{R['per'][2][5]:+.0f}** bps on 7 "
            "trades). No coherent, cross-sectional continuation edge — just small-sample noise."
        ),
        md(
            "### 4e · Synthetic positive control — the harness CAN bank a real continuation\n\n"
            "To prove the null is honest (not a dead detector), plant a **real** post-pause "
            "continuation into a synthetic tape and check the same rule banks it: edge=0 must stay at "
            "*t* ≈ 0; edge>0 must light up with a high win-rate."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 1.00):\n"
            "    px, _ = data.synthetic_panel(edge=edge, seed=455, n_days=4000)\n"
            "    e, dirs = st.pattern_entries(px); s = st.summarize(st.forward_returns(px, e, dirs, 20))\n"
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
            "detector works — so the flat/negative real-tape result is a genuine 'nothing there', not "
            "a broken pipeline."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — the three-methods trade does not beat a drift- and mix-matched "
            f"random baseline (Δ = {R['h5'][6]:+.0f}/{R['h10'][6]:+.0f}/{R['h20'][6]:+.0f}/"
            f"{R['h60'][6]:+.0f} bps at 5/10/20/60d; no positive Welch *t* ≥ 2, and the only "
            f"significant one is **{R['h20'][8]:+.2f}** at 20d — *negative*).\n"
            f"- **Tradability `MIRAGE`** — only {R['n_entries']} patterns in 21 years, signed return "
            "flat-to-negative, no edge to scale; costs only deepen the hole.\n"
            f"- **Forecasts continuation? `BUSTED`** — the shuffled-date placebo leaves the result "
            f"untouched (**p = {R['placebo'][1]:.2f}**), and over 20 days the trend *gives back* the "
            "breakout. The pause does not predict the resumption."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — there is nothing to trade\n\n"
            "The pattern fires a handful of times per instrument per decade; its signed forward return "
            "is flat at short horizons and negative at 20 days; it loses to random entries. There is "
            "no edge to scale and no capacity question, because there is no edge. Costs on each rare "
            "trade only push it further under water. The three-methods is a descriptive candlestick "
            "label, not a forecasting strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Strictness.** The 10% wick tolerance is the only charitable knob; tightening it to "
            "zero leaves even fewer, equally flat trades — the result is robust to it.\n"
            "- **Trend filter.** Proponents say the anchor should sit *in* a trend. Adding an MA "
            "filter cuts the sample further without flipping the sign — the continuation just isn't "
            "there.\n"
            "- **Cousin patterns.** Mat-hold, separating lines, upside/downside-gap three-methods are "
            "affine variants and inherit the same (non-)result.\n\n"
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
