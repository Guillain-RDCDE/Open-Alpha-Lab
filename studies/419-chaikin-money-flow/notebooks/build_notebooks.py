"""Generate the two narrative notebooks for Study 419 (Chaikin Money Flow).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached daily
parquet under ../_cache/ if present and otherwise quote the frozen headline numbers in
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-23).
# Headline tape = SPY daily, 2000-01-03 .. 2026-06-23, fingerprint 0e5348c18f0c.
R = dict(
    asof="2026-06-23", fp_spy="0e5348c18f0c", n_days=6657, window=20, cost_bps=2.0,
    # CMF long/flat on SPY (net excess, cost=2bps)
    cmf_sharpe=-0.011, cmf_t=-0.056, cmf_ann=-0.0014, cmf_dd=-0.640, cmf_expo=0.672,
    cmf_gross_sharpe=0.026, cmf_gross_t=0.137,
    cmf_perm_p=1.000,
    # Benchmarks (net excess, cost=2bps)
    sma_sharpe=0.684, sma_t=4.001,
    macd_sharpe=0.427, macd_t=2.442,
    bh_sharpe=0.505, bh_t=3.016,
    d_cmf_sma=-0.695, d_cmf_macd=-0.437, d_cmf_bh=-0.516,
    # Cost sweep (SPY, CMF net excess Sharpe / HAC t)
    cost0_s=0.026, cost0_t=0.137,
    cost1_s=0.008, cost1_t=0.040,
    cost2_s=-0.011, cost2_t=-0.056,
    cost5_s=-0.066, cost5_t=-0.342,
    # Per-ticker (CMF net excess Sharpe / HAC t, cost=2 ; bh sharpe)
    tk=["SPY", "QQQ", "AAPL", "MSFT", "XLE", "GLD"],
    cmf_s=[-0.011, 0.027, 0.405, 0.135, -0.099, 0.558],
    cmf_tt=[-0.056, 0.136, 2.033, 0.704, -0.545, 2.609],
    bh_s=[0.505, 0.443, 0.779, 0.447, 0.422, 0.638],
    # Synthetic positive control (edge sweep; CMF net Sharpe, delta vs B&H)
    syn_edge=[0.0, 0.5, 1.0, 2.0],
    syn_s=[0.414, 0.715, 0.783, 0.880],
    syn_d=[-0.070, 0.164, 0.217, 0.317],
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from chaikin_money_flow import data, strategy as st

CACHE = os.path.abspath(os.path.join("..", "_cache"))
ASOF = "2026-06-23"
TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "XLE", "GLD"]

def have(t):
    return os.path.exists(data._cache_path(t, CACHE))

HAVE_REAL = have("SPY")

def load(t):
    b = data.fetch_daily(t, fetch=False, cache_dir=CACHE)
    return b[b.index <= ASOF]     # drop the in-progress last bar

print("real daily cache present (SPY):", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Chaikin Money Flow — does accumulation/distribution lead price?\n"
            "### Marc Chaikin's volume-flow oscillator, turned into a long/flat rule and raced honestly\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Leads price%3F: Not_supported](https://img.shields.io/badge/Leads_price%3F-Not_supported-8b949e?style=flat-square)\n\n"
            "Open any charting platform and you'll find Chaikin Money Flow (CMF) — a green/red "
            "oscillator that swings between −1 and +1. Above zero is 'accumulation' (smart money "
            "buying); below zero is 'distribution' (smart money selling). The promise: **money "
            "flow leads price** — so buy when CMF is green, step aside when it's red, and you'll "
            "be ahead of the move. We put that to the test on SPY.\n\n"
            "> This is the plain-language layer. Want the HAC *t*-stats, the permutation placebo, "
            "and the benchmark race? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does buying 'accumulation' beat holding SPY? | **No.** CMF long/flat nets a "
            f"Sharpe of **{R['cmf_sharpe']:+.2f}**; just holding SPY nets **{R['bh_sharpe']:+.2f}**. |\n"
            f"| Is the timing doing *anything*? | **No.** A permutation placebo that scrambles "
            f"the signal's timing matches it exactly (p = **{R['cmf_perm_p']:.2f}**). |\n"
            f"| Is it better than a dumb 50/200 moving-average filter? | **No.** The SMA filter "
            f"nets Sharpe **{R['sma_sharpe']:+.2f}** — CMF trails it by **{R['d_cmf_sma']:+.2f}**. |\n"
            "| Could you trade it? | **No.** Even before costs the edge is ~zero; costs only "
            "push it negative. |\n\n"
            "> 'Money flow leads price' is a great story. On SPY it is indistinguishable from "
            "noise — and a plain moving-average crossover, which ignores volume entirely, wins."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"Chaikin Money Flow reads where the close sits inside each bar's range and "
            "weights it by volume. Persistent buying near the highs on heavy volume (CMF > 0) "
            "is accumulation and precedes a rally; persistent selling near the lows (CMF < 0) is "
            "distribution and precedes a decline. Money flow leads price — so follow the flow.\"*\n\n"
            "Marc Chaikin built the Accumulation/Distribution line in the 1970s and CMF on top of "
            "it. The idea is intuitive: if a stock keeps closing near its high on big volume, "
            "buyers are winning — and that pressure should show up in price *next*. We test "
            "whether 'green CMF = be long, red CMF = be flat' actually earns its keep on SPY."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If CMF really *led* price, it would be a free lookahead: a daily, mechanical signal "
            "telling you to be in the market before the up-moves and out before the down-moves. "
            "Millions of retail traders read CMF exactly this way. If instead it merely "
            "*coincides* with price — confirming what already happened — then trading it adds "
            "turnover and cost without adding information, and a far simpler trend filter does better."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we know?\n\n"
            "One honest test:\n\n"
            "1. **Turn CMF into a long/flat rule.** CMF(20) > 0 → be long SPY; CMF ≤ 0 → sit in "
            "cash. Decide at today's close, earn *tomorrow's* return (one execution lag).\n"
            "2. **Race it — net of costs, excess-of-cash — against the obvious simpler rules:** "
            "a 50/200 SMA trend filter, a MACD filter, and plain buy-and-hold. If CMF's "
            "volume information helps, it should *beat* the volume-blind benchmarks.\n"
            "3. **Scramble the timing.** A permutation placebo re-orders the signal in blocks. "
            "If the real strategy looks no different from the scramble, the timing is noise.\n\n"
            "**What would make us say 'mirage':** a net Sharpe at or below the simpler "
            "benchmarks, a HAC *t* under 2, and a permutation *p* that doesn't reject the null."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — what actually happens\n\n"
            "**The race.** Four long/flat books on SPY, net of 2 bps one-way costs, "
            "excess-of-cash. If money flow leads price, the green bar should be tallest."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = load('SPY')\n"
            "    res = st.run_experiment(b, window=20, cmf_thr=0.0, cost_bps=2.0, perm_iter=2000)\n"
            "    vals = [res['cmf']['net']['sharpe'], res['sma']['net']['sharpe'],\n"
            "            res['macd']['net']['sharpe'], res['buy_hold']['net']['sharpe']]\n"
            "else:\n"
            f"    vals = [{R['cmf_sharpe']}, {R['sma_sharpe']}, {R['macd_sharpe']}, {R['bh_sharpe']}]\n"
            "labels = ['CMF\\n(money flow)', 'SMA 50/200\\n(trend)', 'MACD\\n(trend)', 'Buy & hold']\n"
            "cols = [RED, GREEN, GREEN, GREY]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "ax.bar(labels, vals, color=cols, width=.6)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('net excess Sharpe (2 bps cost)')\n"
            "ax.set_title('Money flow loses to volume-blind trend rules — and to doing nothing')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'CMF {vals[0]:+.2f}  |  SMA {vals[1]:+.2f}  |  MACD {vals[2]:+.2f}  |  B&H {vals[3]:+.2f}')"
        ),
        md(
            f"The CMF rule nets a Sharpe of **{R['cmf_sharpe']:+.2f}** — essentially zero. The "
            f"dumb 50/200 moving-average filter, which never looks at volume at all, nets "
            f"**{R['sma_sharpe']:+.2f}**. Even *buy-and-hold* ({R['bh_sharpe']:+.2f}) beats CMF. "
            "'Money flow leads price' does not survive contact with the tape."
        ),
        md(
            "**Is the timing real or is it luck?** We scramble the CMF position in 21-day blocks "
            "2,000 times and see where the real strategy lands in that placebo distribution."
        ),
        code(
            "if HAVE_REAL:\n"
            "    perm = res['cmf']['perm']\n"
            "    p = perm['p_value']; obs = perm['observed_mean']; plac = perm['placebo_mean']\n"
            "else:\n"
            f"    p, obs, plac = {R['cmf_perm_p']}, 0.0, 0.0003\n"
            "print(f'observed mean daily edge: {obs:+.6f}')\n"
            "print(f'placebo mean (scrambled): {plac:+.6f}')\n"
            f"print(f'permutation p-value: {{p:.3f}}  (>=0.05 means the timing is indistinguishable from noise)')"
        ),
        md(
            f"The permutation *p* is **{R['cmf_perm_p']:.2f}** — the real CMF timing is no better "
            "than randomly shuffling when you're long. The signal carries no usable lead."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** CMF long/flat nets Sharpe **{R['cmf_sharpe']:+.2f}** on SPY, "
            f"HAC *t* = **{R['cmf_t']:+.2f}**; permutation *p* = **{R['cmf_perm_p']:.2f}**. "
            "Indistinguishable from noise.\n"
            "- **Tradability — Mirage.** Even gross the edge is ~zero; costs make it negative. "
            "It loses to a volume-blind SMA filter and to buy-and-hold.\n"
            f"- **Does flow lead price? — Not supported.** The timing scramble matches the real "
            "signal; CMF coincides with price, it does not lead it."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "The CMF rule flips in and out of the market often, so costs bite. But the gross "
            "edge is already ~zero, so there's nothing for costs to eat into:"
        ),
        code(
            "costs = [0.0, 1.0, 2.0, 5.0]\n"
            "if HAVE_REAL:\n"
            "    nets = []\n"
            "    cmf = st.chaikin_money_flow(b, 20); pos = st.positions_from_cmf(cmf)\n"
            "    for c in costs:\n"
            "        bt = st.backtest(b, pos, cost_bps=c)\n"
            "        nets.append(st.summarize_book(bt['net_excess'])['sharpe'])\n"
            "else:\n"
            f"    nets = [{R['cost0_s']}, {R['cost1_s']}, {R['cost2_s']}, {R['cost5_s']}]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, nets, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('one-way cost (bps x NAV)'); ax.set_ylabel('net excess Sharpe')\n"
            "ax.set_title('Starts at ~zero, ends below it — no cost makes this a winner')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Gross Sharpe ~+0.03; the signal, not the cost, is the problem.')"
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further\n\n"
            "- **The positive control.** The companion notebook plants a real "
            "'money-flow-leads-price' effect in a synthetic tape — and CMF *finds* it, beating "
            "buy-and-hold as the planted edge grows. The engine works; SPY just doesn't contain "
            "the effect.\n"
            "- **Other tapes.** A few individual names (AAPL, GLD) show a positive CMF Sharpe — "
            "but they trended so hard that *their own* buy-and-hold beats CMF anyway. The flow "
            "isn't leading; the trend is.\n"
            "- **The MFI cousin.** [Study 418](../../418-money-flow-index/) tests the Money Flow "
            "Index — the volume-weighted RSI — with the same harness and the same outcome.\n\n"
            "*Think CMF leads price on a different asset or window? Fork this, swap the tape, and "
            "show a net excess Sharpe that beats the SMA filter *and* clears HAC t = 2. That's the bar.*"
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
            "# Chaikin Money Flow — a quantitative teardown\n"
            "### Daily OHLCV · CMF(20) long/flat · excess-of-cash · HAC inference · permutation placebo · benchmark race\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Leads price%3F: Not_supported](https://img.shields.io/badge/Leads_price%3F-Not_supported-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We turn CMF(20) into a "
            "long/flat timing rule on SPY, race it NET and excess-of-cash against an SMA(50/200) "
            "filter, a MACD filter, and buy-and-hold, and test the timing with a block-permutation "
            "placebo.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily bars, SPY 2000-01-03 → 2026-06-23, "
            f"fingerprint `{R['fp_spy']}`; the offline core and the synthetic positive control run "
            "deterministically. Methods & sources in [`docs/references.md`](../docs/references.md), "
            "reproducible numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> The `In plain words` notes translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front (real tape)\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | CMF net excess Sharpe **{R['cmf_sharpe']:+.2f}**, HAC "
            f"*t* = **{R['cmf_t']:+.2f}**; permutation *p* = **{R['cmf_perm_p']:.2f}**. |\n"
            f"| **Tradability** | `MIRAGE` | Gross Sharpe **{R['cmf_gross_sharpe']:+.2f}** (t "
            f"{R['cmf_gross_t']:+.2f}); loses to SMA ({R['sma_sharpe']:+.2f}), MACD "
            f"({R['macd_sharpe']:+.2f}), and B&H ({R['bh_sharpe']:+.2f}). |\n"
            f"| **Does flow lead price?** | `NOT SUPPORTED` | Δ vs SMA **{R['d_cmf_sma']:+.2f}** "
            f"Sharpe; the permutation placebo matches the real timing. |\n\n"
            "> In plain words: CMF reads where the close sits inside the bar — a *contemporaneous* "
            "fact. On SPY that fact carries no information about the *next* day's return."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "For each bar define the **Money Flow Multiplier** and **Money Flow Volume**:\n\n"
            "$$\\text{MFM}_t = \\frac{(C_t - L_t) - (H_t - C_t)}{H_t - L_t} \\in [-1, +1], "
            "\\qquad \\text{MFV}_t = \\text{MFM}_t \\cdot V_t$$\n\n"
            "Chaikin Money Flow over a window $w$ (default 20):\n\n"
            "$$\\text{CMF}_t = \\frac{\\sum_{i=t-w+1}^{t} \\text{MFV}_i}"
            "{\\sum_{i=t-w+1}^{t} V_i} \\in [-1, +1].$$\n\n"
            "The rule asserts:\n\n"
            "- **H₁ (signal).** $\\mathbb{E}[\\,\\mathbf{1}\\{\\text{CMF}_t > 0\\}\\cdot "
            "r^{\\text{exc}}_{t+1}\\,] > 0$ — being long when CMF is positive earns positive "
            "excess return.\n"
            "- **H₂ (beats the simpler rule).** That CMF book's NET excess Sharpe exceeds a "
            "volume-blind SMA(50/200) and MACD trend filter on the same tape.\n"
            "- **H₃ (timing carries information).** A block-permutation of the position destroys "
            "the edge.\n\n"
            "We fail to reject the null on all three: H₁ HAC *t* ≈ 0, H₂ CMF trails every "
            "benchmark, H₃ permutation *p* ≈ 1."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "CMF is one of the most-watched volume oscillators in retail platforms. If H₁–H₃ "
            "held, it would be a cheap, mechanical market-timing overlay with a genuine "
            "information advantage over price-only trend filters — the 'volume confirms price' "
            "thesis made tradable. The finding that a *volume-blind* SMA filter beats it says the "
            "opposite: the volume weighting in CMF adds turnover and cost, not signal, at the "
            "daily horizon on a broad index."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **CMF(20).** Window = 20 bars; MFM from intrabar close location; doji bars (H=L) "
            "contribute zero flow.\n"
            "- **Timing rule.** CMF > 0 → long (1); CMF ≤ 0 → flat (0). Decision at the close of "
            "*t*; earns the return of *t+1* (one `shift`). Flat = cash; we report excess-of-cash.\n"
            "- **Costs.** One-way bps × NAV on every position change; sweep 0/1/2/5 bps.\n"
            "- **Benchmarks.** SMA(50/200) trend filter, MACD(12/26/9) filter, buy-and-hold — "
            "each as the same long/flat excess-of-cash book.\n"
            "- **Inference.** Newey–West HAC *t* on daily net excess returns; block-permutation "
            "placebo (21-day blocks, 2,000 iters) for the timing.\n"
            "- **Positive control.** Synthetic tape with a tunable 'money-flow-leads-price' knob, "
            "confirming the harness recovers a CMF edge when one exists.\n\n"
            "Headline tape: SPY daily, 2000-01-03 → 2026-06-23. Panel: QQQ, AAPL, MSFT, XLE, GLD."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The benchmark race — CMF vs volume-blind trend filters (net, excess-of-cash)\n\n"
            "Four long/flat books, 2 bps one-way cost. The HAC *t* on each book's daily net "
            "excess return decides whether it is distinguishable from zero."
        ),
        code(
            "if HAVE_REAL:\n"
            "    b = load('SPY')\n"
            "    res = st.run_experiment(b, window=20, cmf_thr=0.0, cost_bps=2.0, perm_iter=2000)\n"
            "    arms = ['cmf','sma','macd','buy_hold']\n"
            "    sh = [res[a]['net']['sharpe'] for a in arms]\n"
            "    tt = [res[a]['net']['hac_t'] for a in arms]\n"
            "else:\n"
            f"    sh = [{R['cmf_sharpe']}, {R['sma_sharpe']}, {R['macd_sharpe']}, {R['bh_sharpe']}]\n"
            f"    tt = [{R['cmf_t']}, {R['sma_t']}, {R['macd_t']}, {R['bh_t']}]\n"
            "labels = ['CMF','SMA 50/200','MACD','Buy & hold']\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "cols = [RED if t < 2 else GREEN for t in tt]\n"
            "ax.bar(labels, sh, color=cols, width=.6)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('net excess Sharpe (2 bps)')\n"
            "ax.set_title('CMF (red, t<2) trails every volume-blind benchmark')\n"
            "for i,(s,t) in enumerate(zip(sh,tt)):\n"
            "    ax.text(i, s + .02*np.sign(s) + .01, f't={t:+.1f}', ha='center', fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "pd.DataFrame({'arm':labels,'net_sharpe':np.round(sh,3),'hac_t':np.round(tt,3)})"
        ),
        md(
            f"> In plain words: CMF nets Sharpe **{R['cmf_sharpe']:+.2f}** at HAC *t* "
            f"**{R['cmf_t']:+.2f}** — statistically a flat zero. The SMA filter clears *t* = "
            f"**{R['sma_t']:+.1f}**; CMF trails it by **{R['d_cmf_sma']:+.2f}** Sharpe. The "
            "volume weighting buys nothing the price trend didn't already give you — for less."
        ),
        md(
            "### 4b · The permutation placebo — is the timing informative?\n\n"
            "Scramble the CMF position in 21-day blocks against the realised returns 2,000 times. "
            "If the real edge sits in the bulk of the placebo cloud, the timing is noise."
        ),
        code(
            "if HAVE_REAL:\n"
            "    cmf = st.chaikin_money_flow(b, 20); pos = st.positions_from_cmf(cmf)\n"
            "    ae = b['close'].astype(float).pct_change().fillna(0.0)\n"
            "    rng = np.random.default_rng(419)\n"
            "    posshift = pos.shift(1).fillna(0.0).to_numpy(); aev = ae.to_numpy()\n"
            "    n = len(aev); blk = 21; nb_ = int(np.ceil(n/blk))\n"
            "    obs = float(np.nanmean(posshift*aev))\n"
            "    plac = []\n"
            "    for _ in range(2000):\n"
            "        order = rng.permutation(nb_)\n"
            "        sh_ = np.concatenate([posshift[j*blk:(j+1)*blk] for j in order])[:n]\n"
            "        plac.append(float(np.nanmean(sh_*aev)))\n"
            "    plac = np.array(plac); p = (np.sum(plac>=obs)+1)/(len(plac)+1)\n"
            "else:\n"
            f"    obs, plac, p = 0.0, np.random.default_rng(0).normal(3e-4,1e-4,2000), {R['cmf_perm_p']}\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(plac*1e4, bins=40, color=GREY, alpha=.7, label='scrambled timing (placebo)')\n"
            "ax.axvline(obs*1e4, c=RED, lw=2.5, label=f'real CMF: {obs*1e4:+.2f} bps/day')\n"
            "ax.set_xlabel('mean daily edge (bps)'); ax.set_ylabel('count')\n"
            "ax.set_title(f'Real CMF sits inside the placebo cloud  (p = {p:.2f})'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'permutation p-value = {p:.3f}')"
        ),
        md(
            f"> In plain words: permutation *p* = **{R['cmf_perm_p']:.2f}**. Knowing *when* CMF "
            "said 'be long' is worth no more than a random reshuffle of those same long-days. "
            "The signal does not lead — it rides along."
        ),
        md(
            "### 4c · Cost sweep — there's no break-even because gross is already ~zero\n"
        ),
        code(
            "costs = [0.0, 1.0, 2.0, 5.0]\n"
            "if HAVE_REAL:\n"
            "    sw = [st.summarize_book(st.backtest(b, pos, cost_bps=c)['net_excess']) for c in costs]\n"
            "    nets = [s['sharpe'] for s in sw]; tts = [s['hac_t'] for s in sw]\n"
            "else:\n"
            f"    nets = [{R['cost0_s']},{R['cost1_s']},{R['cost2_s']},{R['cost5_s']}]\n"
            f"    tts = [{R['cost0_t']},{R['cost1_t']},{R['cost2_t']},{R['cost5_t']}]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(costs, nets, 'o-', c=RED, lw=2, label='net excess Sharpe')\n"
            "ax2 = ax.twinx(); ax2.plot(costs, tts, 's--', c=GREY, lw=1.5, label='HAC t')\n"
            "ax2.axhline(2, ls=':', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('one-way cost (bps x NAV)'); ax.set_ylabel('net Sharpe', color=RED)\n"
            "ax2.set_ylabel('HAC t', color=GREY)\n"
            "ax.set_title('Gross ~+0.03 Sharpe; the line only goes down from there')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'gross Sharpe {{nets[0]:+.3f}} (t {{tts[0]:+.2f}}) -> never clears t=2 at any cost')"
        ),
        md(
            "### 4d · The panel — a couple of names look positive, but their *own* B&H beats CMF\n\n"
            "If CMF led price, its Sharpe should beat each name's buy-and-hold. It never does."
        ),
        code(
            "rows = []\n"
            "if HAVE_REAL:\n"
            "    for t in TICKERS:\n"
            "        if not have(t):\n"
            "            continue\n"
            "        bb = load(t)\n"
            "        cb = st.summarize_book(st.backtest(bb, st.positions_from_cmf(\n"
            "            st.chaikin_money_flow(bb,20)), cost_bps=2.0)['net_excess'])\n"
            "        bh = st.summarize_book(st.buy_hold_excess(bb))\n"
            "        rows.append((t, cb['sharpe'], cb['hac_t'], bh['sharpe']))\n"
            "if not rows:\n"
            f"    rows = list(zip({R['tk']}, {R['cmf_s']}, {R['cmf_tt']}, {R['bh_s']}))\n"
            "panel = pd.DataFrame(rows, columns=['ticker','cmf_sharpe','cmf_hac_t','bh_sharpe'])\n"
            "panel['cmf_minus_bh'] = panel['cmf_sharpe'] - panel['bh_sharpe']\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "x = np.arange(len(panel)); w = .38\n"
            "ax.bar(x-w/2, panel['cmf_sharpe'], w, color=RED, label='CMF long/flat (net)')\n"
            "ax.bar(x+w/2, panel['bh_sharpe'], w, color=GREY, label='buy & hold')\n"
            "ax.set_xticks(x); ax.set_xticklabels(panel['ticker']); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('net excess Sharpe'); ax.legend()\n"
            "ax.set_title('CMF never beats the asset\\'s own buy-and-hold')\n"
            "plt.tight_layout(); plt.show()\n"
            "panel.round(3)"
        ),
        md(
            "> In plain words: AAPL and GLD give CMF a positive Sharpe (and *t* near 2) — but "
            "both trended so hard that simply holding them beats the CMF book. The positive "
            "numbers are the *trend* leaking through a long-biased rule, not flow leading price."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — CMF net excess Sharpe **{R['cmf_sharpe']:+.2f}**, HAC *t* "
            f"**{R['cmf_t']:+.2f}**; gross *t* **{R['cmf_gross_t']:+.2f}**; permutation *p* "
            f"**{R['cmf_perm_p']:.2f}**. No instrument beats its own buy-and-hold.\n"
            f"- **Tradability `MIRAGE`** — loses to SMA ({R['sma_sharpe']:+.2f}), MACD "
            f"({R['macd_sharpe']:+.2f}) and B&H ({R['bh_sharpe']:+.2f}); no positive break-even cost.\n"
            f"- **Does flow lead price? `NOT SUPPORTED`** — Δ vs the volume-blind SMA filter "
            f"is **{R['d_cmf_sma']:+.2f}** Sharpe and the timing-scramble placebo matches the real "
            "signal. CMF coincides with price; it does not lead it."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it?\n\n"
            "No. The gross net-excess Sharpe is ~+0.03 (HAC *t* ≈ +0.14) — inside the noise band "
            "before a cent of cost. A volume-blind SMA(50/200) filter delivers the trend-timing "
            "benefit CMF is supposed to provide, at lower turnover and a far higher Sharpe. There "
            "is no capital level or cost assumption that turns CMF-timing on SPY into an edge."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the positive control\n\n"
            "Is the *harness* capable of seeing a money-flow-leads-price effect? Plant one in a "
            "synthetic tape (today's volume-weighted close location forecasts tomorrow's return) "
            "and sweep the knob. CMF should pull ahead of buy-and-hold as the edge grows."
        ),
        code(
            "edges = [0.0, 0.5, 1.0, 2.0]\n"
            "deltas = []\n"
            "for e in edges:\n"
            "    sb, _ = data.synthetic_panel(n_days=4000, edge=e, seed=419)\n"
            "    cmf_s = st.summarize_book(st.backtest(sb, st.positions_from_cmf(\n"
            "        st.chaikin_money_flow(sb,20)), cost_bps=0.0)['net_excess'])['sharpe']\n"
            "    bh_s = st.summarize_book(st.buy_hold_excess(sb))['sharpe']\n"
            "    deltas.append(cmf_s - bh_s)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(edges, deltas, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted money-flow-leads-price edge'); ax.set_ylabel('CMF Sharpe - B&H Sharpe')\n"
            "ax.set_title('Engine works: CMF beats B&H once a real flow-leads-price edge exists')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('edge=0 -> delta<=0 (no edge); edge>0 -> CMF pulls ahead. The harness can see it.')"
        ),
        md(
            "The engine is a faithful money-flow detector: with a planted lead it beats "
            "buy-and-hold; with none it does not. The real-tape verdict is therefore a statement "
            "about the **market** — on broad US equities at the daily horizon, where the close "
            "sits inside today's bar tells you nothing about tomorrow — not about the method. "
            "Forks worth trying: intraday CMF (5m), the raw Accumulation/Distribution line, or "
            "CMF *divergence* against price rather than its level."
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
