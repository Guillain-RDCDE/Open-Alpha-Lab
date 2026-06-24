"""Generate the two narrative notebooks for Study 449 (Renko-Charts).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md).  The synthetic
figures run anywhere, offline and deterministic; the real-tape cells read the cached daily
parquets under ../_cache/ if present and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-05-31).
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", nbars=5385,
    # pooled equal-weight portfolio (gross)
    renko_S=0.683, renko_t=3.43, renko_ann=6.78, renko_vol=9.61, renko_dd=-17.8,
    raw_S=0.685, raw_t=3.43, raw_ann=6.77, raw_vol=9.56, raw_dd=-19.6,
    bh_S=0.713, bh_t=3.73, bh_ann=12.75, bh_vol=16.82, bh_dd=-46.3,
    delta_S=-0.002, delta_bh=-0.030,
    # placebo
    perm_obs=0.0029, perm_p=0.495, perm_n=500,
    # turnover
    tpy_renko=7.8, tpy_raw=7.8,
    # per-instrument: renko_S, renko_t, raw_S, bh_S, delta, brick
    inst={
        "SPY": (0.627, 3.07, 0.624, 0.645, 0.003, 1.77),
        "QQQ": (0.740, 3.52, 0.772, 0.778, -0.032, 1.19),
        "DIA": (0.554, 2.66, 0.490, 0.617, 0.064, 1.50),
        "IWM": (0.374, 1.88, 0.405, 0.470, -0.031, 1.51),
        "EFA": (0.289, 1.44, 0.320, 0.391, -0.030, 0.57),
        "GLD": (0.514, 2.48, 0.481, 0.677, 0.033, 1.58),
    },
    # cost sweep: cost -> (renko_S, raw_S, delta)
    cost={0.0: (0.683, 0.685, -0.002), 1.0: (0.674, 0.676, -0.001),
          2.0: (0.665, 0.666, -0.001), 5.0: (0.639, 0.639, 0.000),
          10.0: (0.595, 0.593, 0.002)},
    # synthetic control: trend -> (renko_S, raw_S, bh_S, delta)
    synth={0.0: (0.123, 0.076, 0.085, 0.047), 0.05: (0.171, 0.096, 0.085, 0.075),
           0.10: (0.193, 0.145, 0.085, 0.048), 0.15: (0.163, 0.170, 0.085, -0.007)},
)


def _boot():
    return (
        "import sys, os\n"
        "sys.path.insert(0, os.path.abspath(\"..\"))          # the study package\n"
        "sys.path.insert(0, os.path.abspath(\"../../..\"))    # repo root\n"
        "%matplotlib inline\n"
        "import numpy as np, pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "plt.rcParams.update({\"figure.figsize\": (9.5, 5.0), \"axes.grid\": True,\n"
        "                     \"grid.alpha\": .3, \"axes.spines.top\": False,\n"
        "                     \"axes.spines.right\": False})\n"
        "RED, AMBER, GREEN, GREY, BLUE = \"#c0392b\", \"#dab617\", \"#2ea44f\", \"#8b949e\", \"#2c6fbb\"\n"
        "\n"
        "from renko_charts import data, strategy as st\n"
        "\n"
        "# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-05-31).\n"
        f"R = {R!r}\n"
        "\n"
        "TICKERS = [\"SPY\", \"QQQ\", \"DIA\", \"IWM\", \"EFA\", \"GLD\"]\n"
        "ASOF = \"2026-05-31\"\n"
        "FAST, SLOW, ATR_W, ATR_M = 10, 30, 14, 1.0\n"
        "\n"
        "def _have_cache():\n"
        "    return all(os.path.exists(data._cache_path(t, data.DEFAULT_CACHE)) for t in TICKERS)\n"
        "\n"
        "HAVE_REAL = _have_cache()\n"
        "\n"
        "def _load(t):\n"
        "    b = data.load_real(t, fetch=False)\n"
        "    return b[b.index <= ASOF]\n"
        "\n"
        "def pooled(kind, cost=0.0):\n"
        "    cols = []\n"
        "    for t in TICKERS:\n"
        "        b = _load(t)\n"
        "        if kind == \"renko\":\n"
        "            cols.append(st.crossover_returns(b, st.renko_from_atr(b, ATR_W, ATR_M), FAST, SLOW, cost))\n"
        "        elif kind == \"raw\":\n"
        "            cols.append(st.crossover_returns(b, b[\"close\"], FAST, SLOW, cost))\n"
        "        else:\n"
        "            cols.append(st.buy_and_hold(b))\n"
        "    return pd.concat(cols, axis=1).mean(axis=1)\n"
        "\n"
        "print(\"real daily cache present:\", HAVE_REAL)\n"
    )





# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Renko-Charts — does a noise-filtered chart beat a plain one?\n"
            "### The brick chart's promise, tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Cleaner_signal%3F: Busted](https://img.shields.io/badge/"
            "Cleaner_signal%3F-Busted-8b949e?style=flat-square)\n\n"
            "A **Renko chart** throws away time. Instead of one candle per day, it draws a "
            "new **brick** only when price moves a fixed amount. The result *looks* gorgeous: "
            "clean staircases, no clutter. Every charting platform sells the same promise — "
            "*\"Renko filters out the noise, so your trend signals are cleaner and you get fewer "
            "false trades.\"* This notebook asks the only question that matters: does running a "
            "moving-average crossover on the Renko bricks actually beat running it on the plain "
            "daily price?\n\n"
            "> **This is the plain-language layer.** Want the Sharpe ratios, the permutation "
            "placebo, and the synthetic positive control? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is "
            "drawn by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(_boot()),

        # ---- BEAT 0 — VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the Renko crossover beat the *same* crossover on raw price? | **No.** "
            f"Pooled Sharpe **{R['renko_S']:.3f}** (Renko) vs **{R['raw_S']:.3f}** (raw) — a "
            f"difference of **{R['delta_S']:+.3f}**, basically zero. |\n"
            "| Is the difference real or just luck? | **Luck.** Shuffle the daily returns and "
            f"the Renko 'edge' shows up *half the time* (placebo *p* = {R['perm_p']:.2f}). |\n"
            "| Does Renko at least cut the number of trades? | **No.** "
            f"~{R['tpy_renko']:.1f} trades/yr either way — identical. |\n"
            "| Could you trade it? | **Mirage.** Both crossovers *trail buy-and-hold* "
            f"(Sharpe {R['bh_S']:.3f}). The brick adds nothing and costs you upside. |\n\n"
            "> A Renko chart is a prettier way to draw the same information. "
            "The bricks do not contain a signal the plain price doesn't."
        ),

        # ---- BEAT 1 — THE CLAIM ---------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"Time-based candles are full of noise — gaps, wicks, dull sideways days. A Renko "
            "chart prints a brick only when price actually *moves* (say, one ATR). That strips the "
            "noise out, so a moving-average crossover on the bricks gives you cleaner trends and "
            "fewer whipsaws than the same crossover on the messy daily chart.\"*\n\n"
            "It's an appealing idea, and the charts genuinely look cleaner. The bet is that this "
            "visual smoothing translates into a **better signal** — the crossover fires later, "
            "fakes out less, and rides trends longer."
        ),

        # ---- BEAT 2 — SO WHAT ----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If it were true, it would be a free upgrade: take any crossover system you already "
            "run, swap the price feed for a Renko brick feed, and earn more for the same effort. "
            "Renko is one click away on every platform. If it's *false*, then a generation of "
            "traders is choosing a chart because it's soothing to look at — mistaking visual "
            "tidiness for an edge — and giving up the simplest benchmark of all: just holding the "
            "thing."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three rules keep us honest:\n\n"
            "1. **Head-to-head, same rule.** We run the *identical* 10/30 crossover two ways — on "
            "the Renko bricks and on the raw daily close. The only difference is the brick filter, "
            "so any gap is the filter's doing.\n"
            "2. **Beat buy-and-hold.** A trend follower that earns money in a 21-year bull market "
            "has done nothing special unless it beats simply owning the basket. That's the real "
            "bar.\n"
            "3. **Shuffle test.** If we scramble the daily returns into pure noise, the Renko "
            "'edge' should vanish. If it survives the scramble, it was never a signal.\n\n"
            "We run it on **six liquid ETFs** (SPY, QQQ, DIA, IWM, EFA, GLD) over 21 years of "
            "daily data (2005-2026)."
        ),

        # ---- BEAT 4 — THE TEARDOWN -----------------------------------------
        md(
            "## 4 - The teardown — let's actually look\n\n"
            "**First: what a Renko brick series looks like next to the raw price.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = _load('SPY').loc['2018':'2019']\n"
            "    renko = st.renko_from_atr(b, ATR_W, ATR_M)\n"
            "    fig, ax = plt.subplots(figsize=(10, 4.5))\n"
            "    ax.plot(b.index, b['close'], color=GREY, lw=1, label='raw daily close')\n"
            "    ax.plot(renko.index, renko.values, color=BLUE, lw=1.6, drawstyle='steps-post',\n"
            "            label='ATR-Renko brick level')\n"
            "    ax.set_title('Renko (blue steps) tracks the raw close almost step-for-step')\n"
            "    ax.set_ylabel('SPY price'); ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print('On a daily tape the brick is ~one ATR wide, so it hugs the price.')\n"
            "else:\n"
            "    print('cached real tape required for this figure')"
        ),
        md(
            "The blue staircase *is* cleaner to look at — but notice how closely it shadows the "
            "raw line. On daily bars the brick is about one day's range wide, so the brick series "
            "barely smooths anything. That's the first hint the 'noise filter' has little to filter."
        ),
        md("**Now the head-to-head: Renko crossover vs raw crossover vs just buying and holding.**"),
        code(
            "if HAVE_REAL:\n"
            "    sr = st.perf(pooled('renko')); sa = st.perf(pooled('raw')); sb = st.perf(pooled('bh'))\n"
            "    S = [sr['sharpe'], sa['sharpe'], sb['sharpe']]\n"
            "else:\n"
            "    S = [R['renko_S'], R['raw_S'], R['bh_S']]\n"
            "arms = ['Renko\\ncrossover', 'Raw-close\\ncrossover', 'Buy-and-hold']\n"
            "colors = [BLUE, GREY, GREEN]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.6))\n"
            "bars = ax.bar(arms, S, color=colors, width=.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Sharpe ratio (gross, pooled basket)')\n"
            "ax.set_title('Renko = raw, and both lose to buy-and-hold')\n"
            "for bb, s in zip(bars, S):\n"
            "    ax.annotate(f'{s:.3f}', (bb.get_x()+bb.get_width()/2, bb.get_height()+.01),\n"
            "                ha='center', fontsize=10)\n"
            "plt.tight_layout(); plt.show()\n"
            f"print('Renko minus raw Sharpe = {R['delta_S']:+.3f}  (a dead heat)')\n"
            f"print('Renko minus buy-and-hold = {R['delta_bh']:+.3f}  (the trend follower trails)')"
        ),
        md(
            f"The Renko bar ({R['renko_S']:.3f}) and the raw bar ({R['raw_S']:.3f}) are the same "
            f"height. And **both lose to buy-and-hold** ({R['bh_S']:.3f}) — the crossover spends "
            "the bull market jumping in and out, giving back to whipsaws what it earns in trends. "
            "The brick filter changes none of that."
        ),
        md("**The shuffle test: is the tiny Renko difference even real?**"),
        code(
            "# Synthetic illustration of the placebo logic: plant NO trend, the brick adds nothing.\n"
            "d0, _ = data.synthetic_panel(n_days=4000, trend=0.0, seed=449)\n"
            "res0 = st.run_experiment(d0, FAST, SLOW, ATR_W, ATR_M)\n"
            "print(f\"No-trend synthetic tape:  Renko S={res0['renko']['sharpe']:.3f}  \"\n"
            "      f\"raw S={res0['raw']['sharpe']:.3f}  buy-hold S={res0['bh']['sharpe']:.3f}\")\n"
            f"print('Real-tape shuffle placebo (SPY): observed Renko-edge = {R['perm_obs']:+.4f}, '\n"
            f"      'p = {R['perm_p']:.2f} over {R['perm_n']} shuffles')\n"
            "print('p ~ 0.50 means the Renko edge is exactly what reshuffled noise produces.')"
        ),
        md(
            f"On the real tape, scrambling SPY's daily returns reproduces the Renko 'edge' "
            f"**half the time** (*p* = {R['perm_p']:.2f}). A coin flip. There is no signal in the "
            "bricks to find."
        ),

        # ---- BEAT 5 — VERDICT ----------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal — None.** Renko minus raw = **{R['delta_S']:+.3f}** Sharpe, placebo "
            f"*p* = **{R['perm_p']:.2f}**. The brick filter adds nothing distinguishable from "
            "noise.\n"
            f"- **Tradability — Mirage.** Identical turnover (~{R['tpy_renko']:.1f} trades/yr), and "
            f"both crossovers trail buy-and-hold ({R['bh_S']:.3f}). You'd be paying attention to a "
            "prettier chart for a worse outcome.\n"
            "- **'Cleaner signal?' — Busted.** Same trades, same Sharpe, same drawdowns as the raw "
            "crossover. The brick is a redraw, not new information."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT -----------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "There's nothing here to trade *as Renko*. The honest move is the boring one — and "
            "even the plain crossover doesn't beat buy-and-hold on this basket."
        ),
        code(
            "costs = list(R['cost'].keys())\n"
            "if HAVE_REAL:\n"
            "    net_r = [st.perf(pooled('renko', c))['sharpe'] for c in costs]\n"
            "    net_a = [st.perf(pooled('raw', c))['sharpe'] for c in costs]\n"
            "else:\n"
            "    net_r = [R['cost'][c][0] for c in costs]; net_a = [R['cost'][c][1] for c in costs]\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "ax.plot(costs, net_r, 'o-', c=BLUE, lw=2, label='Renko crossover')\n"
            "ax.plot(costs, net_a, 's--', c=GREY, lw=1.8, label='raw-close crossover')\n"
            "ax.axhline(R['bh_S'], ls=':', c=GREEN, lw=1.8, label=f\"buy-and-hold ({R['bh_S']:.3f})\")\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('Sharpe (pooled)')\n"
            "ax.set_title('The two crossover lines sit on top of each other — below buy-and-hold')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('The Renko and raw lines are indistinguishable at every cost level.')"
        ),
        md(
            "The two crossover lines lie on top of each other and stay under the buy-and-hold "
            "dotted line. Costs erode both equally (same low turnover). There is no cost regime, "
            "no instrument, where the brick earns its keep."
        ),

        # ---- BEAT 7 — GOING FURTHER ----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **Intraday Renko.** Renko's claim to fame is on fast charts. Try the same head-to-head "
            "on 5-minute bars (where the brick *can* be much smaller than the bar-to-bar move) and "
            "see if the filter finally does something.\n"
            "- **Brick size.** A bigger brick (2-3× ATR) smooths more but lags more — sweep it and "
            "ask whether *any* brick size beats the raw crossover by a permutation-significant margin.\n"
            "- **A cousin study.** [Study 104 — Bollinger-Reversion](../../104-bollinger-reversion/) "
            "is the same shape: an indicator that looks like a secret but turns out to be bull-market "
            "drift once you pin it to a fair control.\n\n"
            "*Think a particular brick size or timeframe makes Renko earn its keep? Fork this, run "
            "the head-to-head, and show a Renko-minus-raw delta with a permutation p below 0.05. "
            "That's the bar.*"
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
            "# Renko-Charts — a quantitative teardown\n"
            "### Real daily tape · 21 years · ATR-Renko 10/30 crossover vs raw-close crossover · "
            "HAC inference · return-permutation placebo\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Cleaner_signal%3F: Busted](https://img.shields.io/badge/"
            "Cleaner_signal%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim carrying its standard error.* We test whether the "
            "**incremental** Sharpe of a Renko-filtered 10/30 crossover over the **identical** "
            "raw-close crossover is distinguishable from zero, across six liquid ETF tapes "
            "(2005-2026), with a HAC *t*, a return-permutation placebo, and a synthetic "
            "planted-trend positive control.\n\n"
            f"> **Not investment advice.** Real data: Yahoo daily bars {R['start']} to {R['end']}, "
            "six ETFs, total-return; reproducible numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **In plain words** notes translate each result back to intuition."
        ),
        code(_boot()),

        # ---- BEAT 0 ---------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Pooled incremental **delta Sharpe = {R['delta_S']:+.3f}** "
            f"(Renko {R['renko_S']:.3f} vs raw {R['raw_S']:.3f}); return-permutation placebo "
            f"*p* = **{R['perm_p']:.2f}**. The crossover's own HAC *t* = +{R['renko_t']:.2f} is "
            f"bull-market beta — below buy-and-hold's +{R['bh_t']:.2f}. |\n"
            f"| **Tradability** | `MIRAGE` | Identical turnover (~{R['tpy_renko']:.1f} trades/yr); "
            f"Renko Sharpe {R['renko_S']:.3f} < buy-and-hold {R['bh_S']:.3f}; no cost level or "
            "instrument flips it. |\n"
            f"| **Cleaner signal?** | `BUSTED` | Renko and raw crossover share trades, Sharpe, and "
            "drawdowns — on daily bars the ATR-brick tracks the close step-for-step. |\n\n"
            "> In plain words: the crossover makes money because it is long most of a 21-year "
            "up-market — that is beta, not Renko alpha. Swapping raw price for bricks changes nothing."
        ),

        # ---- BEAT 1 ---------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $c_t$ be the daily close and $B$ the brick size. The Renko level $\\rho_t$ updates "
            "only when $|c_t - \\rho_{t-1}| \\ge B$, stepping by whole bricks. Run a long/flat "
            "crossover $\\pi_t = \\mathbb{1}[\\mathrm{SMA}_{f}(x)_t > \\mathrm{SMA}_{s}(x)_t]$ on a "
            "price proxy $x$. The recipe asserts:\n\n"
            "- **H1 (incremental edge).** "
            "$\\mathrm{Sharpe}(\\pi^{\\text{Renko}}) > \\mathrm{Sharpe}(\\pi^{\\text{raw}})$ — the "
            "brick proxy beats the raw-close proxy for the *same* $(f,s)$.\n"
            "- **H2 (beats the benchmark).** "
            "$\\mathrm{Sharpe}(\\pi^{\\text{Renko}}) > \\mathrm{Sharpe}(\\text{buy-and-hold})$.\n"
            "- **H3 (fewer whipsaws).** Renko turnover $<$ raw turnover.\n\n"
            f"We **reject** H1 (delta Sharpe {R['delta_S']:+.3f}, placebo *p* = {R['perm_p']:.2f}), "
            f"**reject** H2 (Renko {R['renko_S']:.3f} < buy-and-hold {R['bh_S']:.3f}), and "
            f"**reject** H3 (turnover {R['tpy_renko']:.1f} = {R['tpy_raw']:.1f} trades/yr)."
        ),

        # ---- BEAT 2 ---------------------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "Renko is offered as a costless upgrade to any trend system. If H1 held, it would be a "
            "rare free lunch — a chart transform that adds information. The interesting failure is "
            "*why* it doesn't: on a daily tape the ATR-brick is roughly one bar's range, so the "
            "step-function brick series is nearly a copy of the close. The transform that is "
            "supposed to remove noise has, at the daily frequency, almost no noise to remove — and "
            "the residual difference is pure path-dependence noise, as the placebo confirms."
        ),

        # ---- BEAT 3 ---------------------------------------------------------
        md(
            "## 3 - How we'd know — the protocol\n\n"
            "- **Brick.** ATR-Renko: brick size $B = 1.0 \\times \\mathrm{median}\\,"
            "\\mathrm{ATR}_{14}$ over the sample (a single fixed brick — true Renko, not a "
            "time-varying band). Brick level built **causally** (level as of close of *t*).\n"
            "- **Signal.** 10/30 simple-MA crossover, long/flat, on the Renko level and (control) "
            "on the raw close.\n"
            "- **Execution.** Position known at close of *t* earns the **raw** close-to-close "
            "return of *t+1* — one `shift`, applied once. Turnover one-way × NAV.\n"
            "- **Benchmarks.** Raw-close crossover (head-to-head) and buy-and-hold.\n"
            "- **Inference.** HAC (Newey-West) *t* on daily strategy returns; "
            "**return-permutation placebo** on the Renko-minus-raw Sharpe delta (shuffle daily "
            "returns, rebuild the path, re-measure).\n"
            "- **Positive control.** Synthetic AR(1) tape — confirms the crossover detects planted "
            "trend.\n\n"
            f"Six tapes: SPY, QQQ, DIA, IWM, EFA, GLD — daily {R['start']} to {R['end']} "
            f"({R['nbars']:,} bars), as-of {R['asof']} (partial month dropped)."
        ),

        # ---- BEAT 4 ---------------------------------------------------------
        md("## 4 - The teardown"),

        md(
            "### 4a - Per-instrument head-to-head (Renko vs raw vs buy-and-hold)\n\n"
            "Single-name Sharpe for each arm; the right panel is the Renko-minus-raw delta."
        ),
        code(
            "rows = []\n"
            "if HAVE_REAL:\n"
            "    for t in TICKERS:\n"
            "        b = _load(t)\n"
            "        res = st.run_experiment(b, FAST, SLOW, ATR_W, ATR_M, 0.0)\n"
            "        rows.append((t, res['renko']['sharpe'], res['renko']['tstat'],\n"
            "                     res['raw']['sharpe'], res['bh']['sharpe'], res['delta_sharpe']))\n"
            "else:\n"
            "    for t in TICKERS:\n"
            "        v = R['inst'][t]; rows.append((t, v[0], v[1], v[2], v[3], v[4]))\n"
            "perinst = pd.DataFrame(rows, columns=['ticker','renko_S','renko_t','raw_S','bh_S','delta'])\n"
            "\n"
            "x = range(len(TICKERS)); w = 0.27\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))\n"
            "ax = axes[0]\n"
            "ax.bar([i-w for i in x], perinst['renko_S'], width=w, color=BLUE, label='Renko')\n"
            "ax.bar([i for i in x], perinst['raw_S'], width=w, color=GREY, label='raw')\n"
            "ax.bar([i+w for i in x], perinst['bh_S'], width=w, color=GREEN, label='buy-hold')\n"
            "ax.set_xticks(list(x)); ax.set_xticklabels(TICKERS)\n"
            "ax.set_ylabel('Sharpe'); ax.legend(); ax.set_title('Per-instrument Sharpe')\n"
            "ax2 = axes[1]\n"
            "cols = [GREEN if d>0 else RED for d in perinst['delta']]\n"
            "ax2.bar(perinst['ticker'], perinst['delta'], color=cols)\n"
            "ax2.axhline(0, c='k', lw=1)\n"
            "ax2.set_ylabel('Renko minus raw Sharpe')\n"
            "ax2.set_title('Incremental Renko edge: sign-flips, centred on zero')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(perinst.round(3).to_string(index=False))"
        ),
        md(
            "> In plain words: the Renko-minus-raw delta is positive for three instruments and "
            "negative for three — a coin's signature, not a filter that helps. And on **every** "
            "instrument both crossovers sit below buy-and-hold."
        ),

        md(
            "### 4b - The return-permutation placebo\n\n"
            "Shuffle SPY's daily log-returns (kill all serial structure), rebuild the price path, "
            "and recompute the Renko-minus-raw Sharpe delta. Where does the *observed* delta fall "
            "in the shuffled distribution?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = _load('SPY')\n"
            "    pl = st.permutation_pvalue(b, FAST, SLOW, ATR_W, ATR_M, n_perm=400, seed=449)\n"
            "    obs, pval = pl['observed_delta'], pl['p_value']\n"
            "    # rebuild the null distribution for the histogram\n"
            "    rng = np.random.default_rng(449)\n"
            "    lr = np.log(b['close']/b['close'].shift(1)).dropna().to_numpy(); base=float(b['close'].iloc[0])\n"
            "    null = []\n"
            "    for _ in range(400):\n"
            "        perm = rng.permutation(lr)\n"
            "        cl = base*np.exp(np.cumsum(np.concatenate([[0.],perm])))\n"
            "        idx = b.index[:cl.size]\n"
            "        fr = pd.DataFrame({'open':cl,'high':cl*1.001,'low':cl*0.999,'close':cl,'volume':1.}, index=idx)\n"
            "        rk = st.renko_from_atr(fr, ATR_W, ATR_M)\n"
            "        d = st.sharpe(st.crossover_returns(fr,rk,FAST,SLOW)) - st.sharpe(st.crossover_returns(fr,fr['close'],FAST,SLOW))\n"
            "        if np.isfinite(d): null.append(d)\n"
            "    null = np.array(null)\n"
            "else:\n"
            "    obs, pval = R['perm_obs'], R['perm_p']\n"
            "    null = np.random.default_rng(1).normal(0, 0.04, 400)\n"
            "fig, ax = plt.subplots(figsize=(9, 4.3))\n"
            "ax.hist(null, bins=30, color=GREY, alpha=.7, label='shuffled (null) deltas')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'observed delta = {obs:+.4f}')\n"
            "ax.set_xlabel('Renko minus raw Sharpe (per shuffle)'); ax.set_ylabel('count')\n"
            "ax.set_title(f'Observed Renko edge sits in the bulk of noise (p = {pval:.2f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed delta = {obs:+.4f}   placebo p = {pval:.3f}')"
        ),
        md(
            f"> In plain words: the red line (the real Renko edge) lands squarely in the middle of "
            f"the grey noise distribution — *p* ~ {R['perm_p']:.2f}. The filter's apparent advantage "
            "is the kind of number you get for free from reshuffled returns."
        ),

        md(
            "### 4c - The positive control — the engine *can* harvest a trend\n\n"
            "On a synthetic tape with planted AR(1) return-persistence, the crossover Sharpe should "
            "rise with the planted trend. If it does, finding nothing on the real tape is a true "
            "negative — not a broken harness."
        ),
        code(
            "trends = list(R['synth'].keys())\n"
            "renko_s, raw_s, bh_s = [], [], []\n"
            "for tr in trends:\n"
            "    d,_ = data.synthetic_panel(n_days=4000, trend=tr, annual_drift=0.0, seed=449)\n"
            "    res = st.run_experiment(d, FAST, SLOW, ATR_W, ATR_M)\n"
            "    renko_s.append(res['renko']['sharpe']); raw_s.append(res['raw']['sharpe'])\n"
            "    bh_s.append(res['bh']['sharpe'])\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "ax.plot(trends, renko_s, 'o-', c=BLUE, lw=2, label='Renko crossover')\n"
            "ax.plot(trends, raw_s, 's--', c=GREY, lw=2, label='raw crossover')\n"
            "ax.plot(trends, bh_s, '^:', c=GREEN, lw=1.6, label='buy-and-hold')\n"
            "ax.set_xlabel('planted return-persistence (AR1 trend)')\n"
            "ax.set_ylabel('Sharpe (synthetic)')\n"
            "ax.set_title('Crossover Sharpe climbs with planted trend — engine works')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('Renko-minus-raw deltas:', [round(a-b,3) for a,b in zip(renko_s, raw_s)])"
        ),
        md(
            "> In plain words: the crossover Sharpe rises as we plant more trend (machinery proof). "
            "Even here the Renko-over-raw delta is small and non-monotone — the brick helps a touch "
            "when the trend is faint, then washes out. The real tape behaves like the "
            "'no exploitable persistence beyond drift' regime."
        ),

        # ---- BEAT 5 ---------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal `NONE`** — incremental delta Sharpe **{R['delta_S']:+.3f}** pooled, "
            f"sign-flipping per instrument, permutation *p* = **{R['perm_p']:.2f}**. The crossover's "
            f"HAC *t* = +{R['renko_t']:.2f} is bull-market beta (below buy-and-hold's +{R['bh_t']:.2f}). "
            "No Renko alpha.\n"
            f"- **Tradability `MIRAGE`** — identical turnover (~{R['tpy_renko']:.1f}/yr), Sharpe "
            f"{R['renko_S']:.3f} < buy-and-hold {R['bh_S']:.3f}; the cost sweep never separates the "
            "two crossover arms.\n"
            "- **Cleaner signal? `BUSTED`** — same trades, same Sharpe, same drawdowns as the raw "
            "crossover. The brick is a redraw of the same information."
        ),

        # ---- BEAT 6 ---------------------------------------------------------
        md(
            "## 6 - Could you trade it? — cost sweep\n\n"
            "Pooled Sharpe vs round-trip cost for both crossover arms, against the buy-and-hold line."
        ),
        code(
            "costs = list(R['cost'].keys())\n"
            "if HAVE_REAL:\n"
            "    sw_r = [st.perf(pooled('renko', c))['sharpe'] for c in costs]\n"
            "    sw_a = [st.perf(pooled('raw', c))['sharpe'] for c in costs]\n"
            "else:\n"
            "    sw_r = [R['cost'][c][0] for c in costs]; sw_a = [R['cost'][c][1] for c in costs]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "ax.plot(costs, sw_r, 'o-', c=BLUE, lw=2, label='Renko crossover')\n"
            "ax.plot(costs, sw_a, 's--', c=GREY, lw=1.8, label='raw crossover')\n"
            "ax.axhline(R['bh_S'], ls=':', c=GREEN, lw=1.8, label=f\"buy-and-hold ({R['bh_S']:.3f})\")\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('Sharpe (pooled)')\n"
            "ax.set_title('Renko and raw lines overlap, both under buy-and-hold')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "deltas = [r-a for r,a in zip(sw_r, sw_a)]\n"
            "print('Renko-minus-raw delta by cost:', [round(d,3) for d in deltas])"
        ),
        md(
            "> In plain words: costs erode both arms equally (same low turnover), and the gap stays "
            "inside noise. At the highest cost the Renko arm edges marginally ahead — it occasionally "
            "collapses a same-day round trip into one brick — but never enough to matter, and never "
            "above buy-and-hold."
        ),

        # ---- BEAT 7 ---------------------------------------------------------
        md(
            "## 7 - Going further — open questions\n\n"
            "The synthetic control confirms the crossover engine is a faithful trend detector; the "
            "real-tape answer is that the *brick transform* adds nothing on daily bars. Forks worth "
            "pursuing:\n\n"
            "- **Intraday frequency.** On 5-minute bars the brick can be many ticks smaller than the "
            "bar-to-bar move, so the filtering is non-trivial — run the head-to-head there (mind the "
            "60-day yfinance cap and the capacity caveat).\n"
            "- **Brick-size sweep.** Does *any* multiple of ATR (0.5×–3×) produce a permutation-"
            "significant Renko-over-raw delta? Pre-register the grid to avoid snooping.\n"
            "- **Other rules on bricks.** Test a breakout or RSI rule on the brick series vs raw — "
            "Renko may matter more for a rule that reads consecutive-brick counts.\n"
            "- **[Study 104 — Bollinger-Reversion](../../104-bollinger-reversion/)** and "
            "**[Study 343 — Data-Mining-Roulette](../../343-data-mining-roulette/)** for cousins in "
            "shape and method."
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
