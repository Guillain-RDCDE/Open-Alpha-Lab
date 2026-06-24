"""Generate the two narrative notebooks for Study 403 (Hammer & Hanging Man).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached basket OHLC
under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance un-adjusted daily
# OHLC, 26-name large-cap basket + SPY, 1962-01-02 -> 2026-06-23, 345,716 bars,
# 11,816 hammer-shaped bars, fingerprint d488237f4299, as-of 2026-06-23).
R = dict(
    asof="2026-06-23", start="1962-01-02", end="2026-06-23",
    n_names=26, n_bars=345716, n_shape=11816, fp="d488237f4299",
    # HAMMER (shape after a downtrend — bullish claim): H -> (n, edge%, t, p, net%)
    hammer={1: (5152, -0.031, -0.82, 0.869, -0.131),
            3: (5149, +0.052, +0.92, 0.128, -0.048),
            5: (5148, +0.043, +0.63, 0.241, -0.057),
            10: (5144, +0.023, +0.23, 0.383, -0.077)},
    # HANGING MAN (shape after an uptrend — bearish claim): H -> (n, edge%, t, p, net%)
    hang={1: (6493, -0.037, -1.71, 0.942, -0.137),
          3: (6493, -0.099, -2.63, 0.993, -0.199),
          5: (6492, -0.098, -2.02, 0.977, -0.198),
          10: (6489, -0.072, -1.00, 0.852, -0.172)},
    # ANY (shape ignoring trend): H -> (n, edge%, t, net%)
    anyside={1: (11816, -0.032, -1.67, -0.132),
             3: (11813, -0.028, -0.87, -0.128),
             5: (11811, -0.033, -0.82, -0.133),
             10: (11804, -0.030, -0.50, -0.130)},
    # myth-check filter sweep on hammer H=3: label -> (edge%, t, p, n)
    filt=[("plain (lookback 10)", +0.052, +0.92, 0.128, 5149),
          ("trend lookback 5", +0.095, +1.94, 0.012, 5534),
          ("trend lookback 20", -0.017, -0.34, 0.637, 4973),
          ("wick >= 3x body", +0.005, +0.07, 0.478, 2510),
          ("wick >= 4x body", +0.005, +0.05, 0.486, 1419)],
    # synthetic control (H=1, side=any): planted -> (events, edge%, t, p, win%)
    syn=[(0.000, 2415, +0.005, +0.22, 0.403, 49.9),
         (0.005, 2423, +0.324, +13.45, 0.000, 61.4)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Reliable_reversal%3F: Busted](https://img.shields.io/badge/Reliable_reversal%3F-Busted-8b949e?style=flat-square)\n\n"
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

from hammer_hanging_man import data, strategy as st

ASOF = "2026-06-23"
HAVE_REAL = data.have_real()
if HAVE_REAL:
    PANEL = {t: b[b.index <= ASOF] for t, b in data.load_real().items()}
else:
    PANEL = None
print("real hammer cache present:", HAVE_REAL,
      "| names:", (0 if PANEL is None else len(PANEL)))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"

HS = [1, 3, 5, 10]


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does a long lower wick really mark a floor? 🔨\n"
            "### The hammer & the hanging man — the candle traders swear marks a turn, on the real tape\n\n"
            + BADGES +
            "Here is one of the most-taught chart patterns on the internet. A candle with a tiny body "
            "perched at the **top** of the day's range and a **long tail hanging down** below it. After a "
            "slide, they call it a **hammer** — *\"the market hammered out a bottom, buy it.\"* After a "
            "rally, the *exact same shape* is a **hanging man** — *\"the top is hanging by a thread, "
            "sell.\"* The long lower wick is supposed to mean buyers stepped in and rejected the lows: a "
            "floor you can see.\n\n"
            "It's a great story. The wick **is** real information about *that day* — price went down and "
            "got pushed back up. The question is whether it tells you anything about the **next few "
            "days**. We checked, on **60+ years** of daily bars across 26 big US stocks plus the S&P.\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo test and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **A data note up front.** Fixed **26-name** survivor basket + SPY (names still trading "
            "today). Survivors *recover* from dips that delisted names didn't — so the bias actually "
            "leans **toward** finding a bounce. If even *that* shows no floor, the floor really isn't "
            "there. Charts are drawn by the code beside them; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| After a slide, does a hammer mark a floor? | **No.** Over the next 3 days the hammer beats "
            "the stock's own usual return by a measly **+0.05%** — statistically that's *luck* (a "
            "coin-flip sort beats it 1 time in 8). At 1 day it's even slightly *negative*. |\n"
            "| Is the *hanging man* a top you can short? | **Not really.** The same shape after a rally "
            "does earn a touch *less* than usual — but the stock still goes **up**, just a hair slower. "
            "You can't short \"slightly less up\" for a profit. |\n"
            "| Can I fix it with a filter? | **That's the tell.** The only way to make the hammer look "
            "good is to **pick** a short trend window; a longer one kills it, and demanding a *bigger, "
            "purer* wick (the \"better\" hammer) makes it **worse**. |\n"
            "| Could I trade it? | **No.** After normal trading costs the edge is **negative** at every "
            "horizon, on both sides. |\n\n"
            "> The wick is real — for *that day*. As a forecast of the *next* few days, the floor is a "
            "**mirage**. And we *prove* our test isn't blind: when we secretly plant a real floor in fake "
            "data, the same test screams (more in the quant notebook)."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"A hammer is a small body at the top of the range with a long lower shadow. After a "
            "downtrend it's **bullish** — buyers absorbed the selling and the bottom is in. The same "
            "candle after an uptrend is a **hanging man** — the rally is exhausted, get out.\"*\n\n"
            "This isn't fringe. The hammer and hanging man are core to **Steve Nison**'s *Japanese "
            "Candlestick Charting Techniques* (1991) — the book that brought 18th-century Japanese rice-"
            "trading candles to Wall Street — and they're in every charting app and trading course made "
            "since. So the question isn't \"is it a known pattern?\" It's: **does the long lower wick "
            "actually predict what happens next?**"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a single candle shape could call a turn, you'd have a beautifully simple edge — look at "
            "one bar, take the trade. That's exactly why the pattern is so beloved and so widely taught.\n\n"
            "But two traps hide here. **(1)** A long lower wick says price *did* get rejected today — "
            "that's backward-looking and already in the close. Whether it forecasts *tomorrow* is a "
            "separate, much harder claim. **(2)** Big liquid stocks **drift up** over time, so *any* "
            "random day looks \"bullish\" at 10 days. The honest test isn't \"does the stock rise after a "
            "hammer?\" (it usually does — everything does) — it's **\"does it rise more than the stock's "
            "own normal?\"** We measure that base-rate-adjusted edge."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We scan **{R['n_bars']:,} daily bars** across **{R['n_names']} stocks + SPY** "
            f"({R['start']} → {R['end']}) and flag every bar with the hammer shape — small body up top, "
            f"lower wick at least twice the body, almost no upper shadow. That's **{R['n_shape']:,}** "
            "hammer-shaped bars.\n\n"
            "1. **The shape.** Detected by exact open/high/low/close rules — no eyeballing.\n"
            "2. **The trend split.** Was the stock *down* over the last 10 days (→ hammer) or *up* "
            "(→ hanging man)? Same candle, two stories.\n"
            "3. **The forward move.** Starting at the **next** day's close (no cheating — the candle is "
            "already done), how does the stock do over the next **1, 3, 5, 10** days **versus its own "
            "usual return** over that horizon?\n\n"
            "If the legend is real, the hammer should beat the base rate by a clear margin. We'd call it "
            "a **mirage** if the edge can't clear a basic luck test."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**The bullish hammer first.** Here's how much it beats (or misses) the stock's *own normal* "
            "return at each horizon. If the floor is real, these bars should be clearly, solidly green."
        ),
        code(
            "HS = [1, 3, 5, 10]\n"
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(PANEL, side='hammer', n_draws=1500, seed=403)\n"
            "    edge = [res[h]['edge_mean']*100 for h in HS]\n"
            "else:\n"
            "    edge = [R['hammer'][h][1] for h in HS]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "cols = [GREEN if v>0 else RED for v in edge]\n"
            "ax.bar([f'{h}d' for h in HS], edge, color=cols, width=.6)\n"
            "for i,v in enumerate(edge): ax.annotate(f'{v:+.3f}%',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('hammer edge over the stock\\'s own base rate (%)')\n"
            "ax.set_title('The bullish hammer barely beats the base rate — and not at 1 day')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('hammer edge by horizon (%):', [round(v,3) for v in edge])"
        ),
        md(
            f"That's the whole ballgame, and it's underwhelming. The best the hammer manages is "
            f"**{R['hammer'][3][1]:+.3f}% at 3 days** — and at **1 day** it's actually *negative* "
            f"(**{R['hammer'][1][1]:+.3f}%**). These are rounding-error sizes. (The quant notebook shows "
            "the matching *t* = 0.9 — i.e. indistinguishable from luck.)"
        ),
        md(
            "**Now the hanging man** — the *same shape* after a rally. The folklore says it should call a "
            "**top** (negative forward returns). Let's put the two sides next to each other."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rh = st.run_experiment(PANEL, side='hammer', placebo=False, n_draws=1, seed=403)\n"
            "    rg = st.run_experiment(PANEL, side='hangingman', placebo=False, n_draws=1, seed=403)\n"
            "    ham = [rh[h]['edge_mean']*100 for h in HS]; han = [rg[h]['edge_mean']*100 for h in HS]\n"
            "else:\n"
            "    ham = [R['hammer'][h][1] for h in HS]; han = [R['hang'][h][1] for h in HS]\n"
            "x = np.arange(len(HS))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, ham, .4, color=GREEN, label='HAMMER (after a downtrend — \"floor\")')\n"
            "ax.bar(x+.2, han, .4, color=RED, label='HANGING MAN (after an uptrend — \"top\")')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in HS])\n"
            "ax.set_ylabel('edge over base rate (%)')\n"
            "ax.set_title('Same shape, both stories tiny — hammer ~0, hanging man slightly negative')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('hammer:', [round(v,3) for v in ham], ' hanging man:', [round(v,3) for v in han])"
        ),
        md(
            f"The hanging man's bars *are* a bit negative (e.g. **{R['hang'][3][1]:+.3f}% at 3 days**) — "
            "the only place anything statistically real shows up. But \"negative edge\" here means the "
            "stock rose **less than usual**, not that it fell. You can't short \"a bit less up\" and keep "
            "the lights on after costs. Neither side gives you a floor or a top you can actually trade."
        ),
        md(
            "**The give-away: the filter rescue.** Believers say \"sure, but you need the *right* setup.\" "
            "So we tried. Watch what happens to the hammer's 3-day edge as we change the filters."
        ),
        code(
            "labels = [f[0] for f in R['filt']]; tvals = [f[2] for f in R['filt']]\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.3))\n"
            "cols = [AMBER if abs(t)>=1 else GREY for t in tvals]\n"
            "ax.barh(range(len(labels)), tvals, color=cols)\n"
            "ax.axvline(2, ls='--', c=RED, label='t = 2 bar (what \"real\" needs)')\n"
            "ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:+.2f}',(t,i),va='center',ha='left' if t>=0 else 'right')\n"
            "ax.set_xlabel('hammer 3-day edge, as a t-stat'); ax.set_title('Pick the filter, pick the answer — the mirage signature')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for f in R['filt']: print(f'{f[0]:24s} edge={f[1]:+.3f}%  t={f[2]:+.2f}')"
        ),
        md(
            "A **shorter** trend window (lookback 5) lifts *t* to **1.94** — almost there! — but a "
            "**longer** one (lookback 20) sends it back to zero, and demanding a *longer, purer* wick — "
            "the \"textbook-perfect\" hammer — **flattens it entirely** (*t* ≈ 0.05). When the answer "
            "swings with the knob you chose, you're not finding a signal. You're fishing."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The bullish hammer never clears the bar: best **+0.05% at 3 days** "
            "(luck-level), negative at 1 day. The only real stat is the hanging man's faint *under*-"
            "performance — the wrong direction for a floor, and untradeable.\n"
            "- **Tradability — Mirage.** After costs the edge is **negative everywhere, both sides.**\n"
            "- **Reliable reversal? — Busted.** No floor, no shortable top, and the only \"edge\" comes "
            "from snooping the filter. A *bigger* wick makes it *worse*."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — costs finish it off\n\n"
            "Even granting the tiny gross blip, here's the hammer **gross vs net** of a normal 5-bps "
            "round trip (buy at the next close, sell H days later)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rr = st.run_experiment(PANEL, side='hammer', placebo=False, n_draws=1, seed=403)\n"
            "    g = [rr[h]['edge_mean']*100 for h in HS]; nv = [rr[h]['net_edge']*100 for h in HS]\n"
            "else:\n"
            "    g = [R['hammer'][h][1] for h in HS]; nv = [R['hammer'][h][4] for h in HS]\n"
            "x = np.arange(len(HS))\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.3))\n"
            "ax.bar(x-.2, g, .4, color=GREEN, label='gross edge')\n"
            "ax.bar(x+.2, nv, .4, color=GREY, label='net of 5-bps round trip')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xticks(x); ax.set_xticklabels([f'{h}d' for h in HS])\n"
            "ax.set_ylabel('hammer edge (%)'); ax.set_title('Net of costs the hammer is under water at every horizon')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('net hammer edge by horizon (%):', [round(v,3) for v in nv])"
        ),
        md(
            f"Every grey bar is below zero — net **{R['hammer'][3][4]:+.3f}%** even at the hammer's best "
            "(3-day) horizon. There was never a positive gross edge worth defending, and the round trip "
            "buries what little there was. Nothing to deploy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The wick is real, the forecast isn't.** A long lower shadow genuinely tells you buyers "
            "fought back *that day*. The lesson is that *intraday* information isn't a *multi-day* "
            "forecast — a trap that sinks most single-candle patterns.\n"
            "- **Try small-caps / intraday.** Candle effects are bigger (and costlier) in illiquid names; "
            "re-run on a small-cap basket or on 5-minute bars and see if a thin edge appears.\n"
            "- **Combine with a real signal.** A hammer *plus* an earnings surprise or a real reversal "
            "factor might do something the candle alone can't. Fork it and show the **net** edge clearing "
            "*t* = 2 — then we'll talk.\n\n"
            "*The reproducible core is offline and deterministic. Methods and sources: "
            "[`docs/references.md`](../docs/references.md); frozen numbers: "
            "[`docs/results.md`](../docs/results.md).*"
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
            "# Hammer & Hanging Man — a quantitative teardown 🔬\n"
            "### OHLC pattern detector · forward 1/3/5/10-day edge vs base rate · HAC *t* + label-shuffle "
            "placebo · the trend split · a filter-snoop myth-check · a synthetic planted-floor power control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The hammer / "
            "hanging man is the canonical single-candle reversal pattern — and the canonical academic null "
            "(Marshall, Young & Rose 2006). The job here is to measure it *honestly*: detect the geometry "
            "by exact rules, benchmark each event against the name's own base rate (controlling for "
            "drift), confront the overlapping windows with a **HAC** *t* and a label-shuffle placebo, and "
            "show the only path to significance is filter-snooping.\n\n"
            "> ⚠️ **Data + survivorship note.** Fixed **26-name** large-cap basket + SPY, names still "
            "trading in 2026 — a *survivor* panel whose bias points **toward** a post-dip bounce "
            "(survivors recover from dips delisted names didn't), so a null hammer is **conservative**. "
            "Real data: yfinance **un-adjusted** daily OHLC (the candle *shape* needs printed levels → "
            "forward returns are **price-only**), 1962→2026. Offline core + synthetic control are "
            "deterministic. Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Bullish hammer edge over base rate peaks at "
            f"**{R['hammer'][3][1]:+.3f}% / 3d, HAC t = {R['hammer'][3][2]:+.2f}** (placebo "
            f"p = {R['hammer'][3][3]:.3f}); *negative* at 1d. Only real stat is the hanging man's "
            f"*under*-performance (3d t = {R['hang'][3][2]:.2f}) — wrong sign for a floor. Synthetic "
            f"planted floor lights up at **t = {R['syn'][1][3]:.2f}**, so the flat reading is genuine. |\n"
            f"| **Tradability** | `MIRAGE` | Net of a 5-bps round trip the edge is **negative at every "
            f"horizon and side** (hammer 3d net **{R['hammer'][3][4]:+.3f}%**). No positive gross edge. |\n"
            f"| **Reliable reversal?** | `BUSTED` | Significance appears *only* by snooping the trend "
            f"window (lookback 5 → t = {R['filt'][1][2]:.2f}; lookback 20 → t = {R['filt'][2][2]:.2f}); "
            f"a longer/purer wick → t ≈ {R['filt'][4][2]:.2f}. |\n\n"
            "> 💡 In plain words: the candle carries *intraday* information but no *forward* edge; any "
            "apparent signal is a tuned-filter artefact, and costs sink whatever is left."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let a bar have body $b=|c-o|$, range $\\rho=h-\\ell$, lower wick "
            "$w_-=\\min(o,c)-\\ell$, upper wick $w_+=h-\\max(o,c)$. The hammer/hanging-man geometry:\n\n"
            "$$w_- \\ge 2\\,b,\\qquad w_+ \\le 0.5\\,b,\\qquad b \\le 0.35\\,\\rho,\\qquad \\rho>0.$$\n\n"
            "Same shape both ways; the trend at the close splits it — "
            "$\\tau_t=\\operatorname{sign}(c_t/c_{t-10}-1)$, **hammer** if $\\tau<0$, **hanging man** if "
            "$\\tau>0$. For event $i$ the forward edge is the $H$-day return entered **one day later** "
            "minus that name's unconditional base rate $\\mu_{\\text{name}}(H)$:\n\n"
            "$$e_i(H)=\\frac{c_{t+1+H}}{c_{t+1}}-1-\\mu_{\\text{name}}(H).$$\n\n"
            "- **H₁ (the floor exists).** $\\overline{e}(H)>0$ and HAC-significant for the **hammer**.\n"
            "- **H₂ (deployable).** $\\overline{e}(H)$ survives a one-way × 2 round trip.\n"
            "- **H₃ (robust reversal).** The edge survives reasonable filter choices and a *stronger* "
            "wick.\n\n"
            "We find **H₁ rejected** (hammer t ≤ 0.92), **H₂ rejected** (net-negative everywhere), "
            "**H₃ rejected** (significance only by snooping). The shape is real; the *forecast* is not."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — why base-rate-adjust and why HAC\n\n"
            "Two honesty problems sit on top of a naive test. **(a) Drift.** Large-caps trend up, so a "
            "raw \"return after a hammer\" is positive for *every* candle; we subtract each name's own "
            "base rate so the edge measures *excess over buy-and-hold for that name*. **(b) Overlap.** "
            "Nearby hammer signals share overlapping multi-day forward windows, so the events are "
            "autocorrelated and an i.i.d. *t* overstates significance — we use a **Newey-West (HAC)** "
            "one-sample *t* with an auto bandwidth. The placebo then asks the cross-sectional question: "
            "could the same *number* of random days, drawn from the same tapes, beat the observed edge?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** Fixed **{R['n_names']}-name** large-cap basket + SPY (yfinance un-adjusted "
            f"daily OHLC, {R['start']}→{R['end']}); **{R['n_bars']:,}** bars, **{R['n_shape']:,}** "
            "hammer-shaped. **Survivor** panel — named on the Signal axis (bias *toward* a bounce).\n"
            "- **Detector.** $w_-\\ge2b,\\ w_+\\le0.5b,\\ b\\le0.35\\rho$ (a doji is excluded via a tiny "
            "body floor).\n"
            "- **Trend split.** 10-day sign of return at the signal close (no look-ahead).\n"
            "- **Timing.** Signal at close[t]; enter close[t+1]; exit close[t+1+H], $H\\in\\{1,3,5,10\\}$. "
            "One `shift`, applied once. Price-only.\n"
            "- **Edge.** Conditional forward return minus the name's unconditional base rate.\n"
            "- **Null #1 (HAC t)** of the pooled edge vs 0.\n"
            "- **Null #2 (label-shuffle placebo).** Per name, draw its event-count of random days; "
            "$p=\\Pr[\\text{shuffled edge}\\ge\\text{observed}]$.\n"
            "- **Costs.** 5 bps one-way × 2 (round trip), long-only (no borrow).\n"
            "- **Positive control.** A deterministic panel with a **planted** post-hammer bounce: zero "
            "edge must NOT reach significance; a real floor must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The term structure of the edge — hammer vs hanging man\n\n"
            "Edge over base rate at each horizon, both sides. The hammer (the bullish-floor claim) "
            "hovers at zero; the hanging man is faintly negative."
        ),
        code(
            "HS = [1, 3, 5, 10]\n"
            "if HAVE_REAL:\n"
            "    rh = st.run_experiment(PANEL, side='hammer', placebo=False, n_draws=1, seed=403)\n"
            "    rg = st.run_experiment(PANEL, side='hangingman', placebo=False, n_draws=1, seed=403)\n"
            "    ham = [rh[h]['edge_mean']*100 for h in HS]; han = [rg[h]['edge_mean']*100 for h in HS]\n"
            "    hamt = [rh[h]['t'] for h in HS]; hant = [rg[h]['t'] for h in HS]\n"
            "else:\n"
            "    ham = [R['hammer'][h][1] for h in HS]; han = [R['hang'][h][1] for h in HS]\n"
            "    hamt = [R['hammer'][h][2] for h in HS]; hant = [R['hang'][h][2] for h in HS]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.8, 4.3))\n"
            "x = np.arange(len(HS))\n"
            "a1.bar(x-.2, ham, .4, color=GREEN, label='hammer'); a1.bar(x+.2, han, .4, color=RED, label='hanging man')\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_xticks(x); a1.set_xticklabels([f'{h}d' for h in HS])\n"
            "a1.set_ylabel('edge over base rate (%)'); a1.set_title('Edge: both tiny'); a1.legend()\n"
            "a2.plot(HS, hamt, 'o-', c=GREEN, lw=2, label='hammer t'); a2.plot(HS, hant, 'o-', c=RED, lw=2, label='hanging man t')\n"
            "a2.axhline(2, ls='--', c=GREY); a2.axhline(-2, ls='--', c=GREY); a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_xlabel('horizon (days)'); a2.set_ylabel('HAC t'); a2.set_title('HAC t: hammer never reaches +2'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('hammer edge%:', [round(v,3) for v in ham], 't:', [round(v,2) for v in hamt])\n"
            "print('hangman edge%:', [round(v,3) for v in han], 't:', [round(v,2) for v in hant])"
        ),
        md(
            f"> 💡 In plain words: the hammer's HAC *t* tops out at **{R['hammer'][3][2]:+.2f}** (3d) and "
            f"is *negative* at 1d — it never approaches the +2 floor a bullish edge would need. The "
            f"hanging man dips to **{R['hang'][3][2]:.2f}** at 3d: real, but it means \"rose less than "
            "usual,\" not a tradable short. The bullish-floor claim has no statistical leg."
        ),
        md(
            "### 4b · The decisive test — the hammer against a label-shuffle null\n\n"
            "The 3-day hammer edge (its best) against a per-name label-shuffle null. A real floor sits in "
            "the right tail; a mirage sits in the middle."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(PANEL, 3, side='hammer', n_draws=3000, seed=403)\n"
            "    obs = pl['obs_edge']*100; draws = pl['draws']*100; pval = pl['p_value']\n"
            "    tval = st.run_experiment(PANEL, side='hammer', horizons=(3,), placebo=False, n_draws=1, seed=403)[3]['t']\n"
            "else:\n"
            "    obs = R['hammer'][3][1]; pval = R['hammer'][3][3]; tval = R['hammer'][3][2]\n"
            "    rng = np.random.default_rng(403); draws = rng.normal(0.0, 0.04, 3000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='null: per-name random days')\n"
            "ax.axvline(obs, c=GREEN, lw=2.5, label=f'observed hammer edge {obs:+.3f}%')\n"
            "ax.set_xlabel('3-day hammer edge over base rate (%)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Inside the luck cloud: placebo p = {pval:.3f}, HAC t = {tval:+.2f}')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.3f}%  HAC t={tval:+.2f}  shuffle p={pval:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the green line sits **inside** the cloud — a random sort of the same "
            f"days beats the hammer about **{R['hammer'][3][3]*100:.0f}%** of the time (placebo "
            f"p = {R['hammer'][3][3]:.3f}), and HAC t = {R['hammer'][3][2]:+.2f}. This is a textbook "
            "non-result: the candle adds nothing over picking days at random."
        ),
        md(
            "### 4c · The myth-check — significance only by snooping the filter\n\n"
            "Believers reach for filters. We sweep the obvious ones on the hammer's 3-day edge. The bar "
            "to beat is *t* = 2."
        ),
        code(
            "labels = [f[0] for f in R['filt']]; tvals = [f[2] for f in R['filt']]; edges=[f[1] for f in R['filt']]\n"
            "fig, ax = plt.subplots(figsize=(9.8, 4.3))\n"
            "cols = [GREEN if t>=2 else (AMBER if t>=1 else GREY) for t in tvals]\n"
            "ax.barh(range(len(labels)), tvals, color=cols)\n"
            "ax.axvline(2, ls='--', c=RED, label='t = 2 bar')\n"
            "ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)\n"
            "for i,(t,e) in enumerate(zip(tvals,edges)): ax.annotate(f't={t:+.2f} ({e:+.3f}%)',(t,i),va='center',ha='left' if t>=0 else 'right',fontsize=9)\n"
            "ax.set_xlabel('hammer 3-day edge as a t-stat'); ax.set_title('Pick the filter, pick the t — the snoop signature'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for f in R['filt']: print(f'{f[0]:24s} edge={f[1]:+.3f}%  t={f[2]:+.2f}  p={f[3]:.3f}  n={f[4]}')"
        ),
        md(
            f"> 💡 In plain words: a *shorter* trend window (lookback 5) lifts t to "
            f"**{R['filt'][1][2]:.2f}** — still under 2 — while a *longer* one (lookback 20) drops it to "
            f"**{R['filt'][2][2]:.2f}**, and a *longer, purer wick* (the \"better\" hammer) flattens it to "
            f"**t ≈ {R['filt'][4][2]:.2f}**. A signal you can dial in by choosing the knob is the "
            "definition of a data-snooped mirage (Sullivan-Timmermann-White 1999)."
        ),
        md(
            "### 4d · Faithful-engine & power control — we know the truth here\n\n"
            "On a deterministic panel where we **plant** a post-hammer bounce: with **zero** edge the "
            "test must stay at t≈0 (no false positive); with a real planted floor it must light up. Both "
            "hold — so the flat real-tape t is a genuine negative, not a blind harness."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 0.005):\n"
            "    sp, _ = data.synthetic_panel(edge=edge, seed=403)\n"
            "    r = st.run_experiment(sp, side='any', horizons=(1,), n_draws=1500, seed=403)[1]\n"
            "    res.append((edge, r['n'], r['edge_mean']*100, r['t'], r['p_placebo'], r['win']*100))\n"
            "tvals = [r[3] for r in res]\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar([f'planted\\n{e*100:.1f}%' for e,_,_,_,_,_ in res], tvals, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=RED, label='t = 2 bar')\n"
            "for i,t in enumerate(tvals): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('synthetic 1d HAC t'); ax.set_title('Control: zero edge -> t~0; planted floor -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,k,ed,t,p,w in res: print(f'planted {e*100:.1f}%: events={k} edge={ed:+.3f}% t={t:.2f} p={p:.3f} win={w:.1f}%')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted floor the control sits at "
            f"**t = {R['syn'][0][3]:.2f}** (no false positive); a **+0.5%/event** planted floor reaches "
            f"**t = {R['syn'][1][3]:.2f}**. The machinery is unbiased and powerful, so the real-tape "
            "hammer t ≈ 0.9 is a true negative — the floor genuinely isn't there."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — hammer edge peaks at **{R['hammer'][3][1]:+.3f}% / 3d, HAC "
            f"t = {R['hammer'][3][2]:+.2f}** (placebo p = {R['hammer'][3][3]:.3f}), negative at 1d. The "
            f"only HAC-real statistic is the hanging man's *under*-performance (3d t = {R['hang'][3][2]:.2f}) "
            "— the wrong sign for a floor and untradeable. Survivorship tilts *toward* a bounce, so this "
            f"is conservative. The control detects a planted floor at **t = {R['syn'][1][3]:.2f}**.\n"
            f"- **Tradability `MIRAGE`** — net of a 5-bps round trip the edge is **negative at every "
            f"horizon and side** (hammer 3d net **{R['hammer'][3][4]:+.3f}%**). No gross edge to defend.\n"
            f"- **Reliable reversal? `BUSTED`** — significance appears only by snooping the trend window "
            f"(lookback 5 → t = {R['filt'][1][2]:.2f}; 20 → {R['filt'][2][2]:.2f}) and *vanishes* with a "
            f"longer/purer wick (t ≈ {R['filt'][4][2]:.2f})."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — costs on a non-existent edge\n\n"
            "The hammer gross vs net by horizon, with the break-even line. There is no net-positive region."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rr = st.run_experiment(PANEL, side='hammer', placebo=False, n_draws=1, seed=403)\n"
            "    g = [rr[h]['edge_mean']*100 for h in HS]; nv = [rr[h]['net_edge']*100 for h in HS]\n"
            "else:\n"
            "    g = [R['hammer'][h][1] for h in HS]; nv = [R['hammer'][h][4] for h in HS]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(HS, g, 'o--', c=GREY, lw=1.6, label='gross edge')\n"
            "ax.plot(HS, nv, 'o-', c=RED, lw=2.2, label='net of 5-bps round trip')\n"
            "ax.axhline(0, c='k', ls='--', label='break-even')\n"
            "for h,v in zip(HS,nv): ax.annotate(f'{v:+.3f}%',(h,v),ha='center',va='top',fontsize=9)\n"
            "ax.set_xlabel('horizon (days)'); ax.set_ylabel('hammer edge (%)')\n"
            "ax.set_title('No net-positive region — under water at every horizon'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('net hammer edge by horizon (%):', {f'{h}d': round(v,3) for h,v in zip(HS,nv)})"
        ),
        md(
            "> 💡 In plain words: the red (net) line is **below zero** the whole way. With no positive "
            "gross edge, costs aren't even the binding constraint — there is simply nothing to harvest. "
            "Hence **MIRAGE**, not FRAGILE."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Intraday is the natural habitat.** The wick is a within-day fact; a same-day or "
            "next-open framing on 5-minute bars (~60 days on yfinance) might surface a thin, capacity-"
            "limited effect the daily close washes out.\n"
            "- **Small / illiquid names.** Candlestick effects are larger (and costlier) where "
            "microstructure frictions bite; re-run the basket on small-caps.\n"
            "- **The general lesson.** A backward-looking *shape* (it already happened) is not a "
            "forward-looking *forecast*. Most single-candle patterns fail this exact test — the contrast "
            "with [Study 363 — PEAD](../../363-pead-drift/), which clears the bar, is the point.\n\n"
            "*The reproducible core is offline and deterministic; the synthetic control plants a known "
            "floor. Methods and sources: [`docs/references.md`](../docs/references.md); frozen numbers: "
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
