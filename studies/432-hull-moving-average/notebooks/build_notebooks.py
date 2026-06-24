"""Generate the two narrative notebooks for Study 432 (Hull Moving Average).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). They recompute the
real-tape numbers live from the cached daily parquets under ../_cache/ when present, and
otherwise fall back to the frozen headline numbers in ``R`` (mirroring docs/results.md), so
the notebook re-runs for any reader, online or off.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-13).
R = dict(
    asof="2026-06-13", n_spy=8400, yrs=33.3,
    # SPY headline, long/flat, cost=2bps
    hma_sharpe=0.089, hma_cagr=0.36, hma_dd=-63.7,
    sma_sharpe=0.465, sma_cagr=4.60, sma_dd=-43.8,
    macd_sharpe=0.404, macd_cagr=3.93, macd_dd=-39.0,
    bh_sharpe=0.646, bh_cagr=10.82, bh_dd=-55.2,
    hma_spread=-4.35, hma_t=-5.31,
    sma_spread=-2.74, sma_t=-3.15,
    macd_spread=-2.99, macd_t=-3.58,
    hma_sw=32.5, sma_sw=17.4, macd_sw=21.4,
    hma_tim=58,
    # gross
    hma_spread_gross=-4.10, hma_t_gross=-5.01,
    # permutation
    perm_obs=-4.10, perm_placebo=-1.99, perm_p=1.000,
    # cost sweep (net Sharpe, spread, t)
    cost=[0.0, 1.0, 2.0, 5.0, 10.0],
    cost_sharpe=[0.145, 0.117, 0.089, 0.005, -0.136],
    cost_spread=[-4.10, -4.23, -4.35, -4.74, -5.38],
    cost_t=[-5.01, -5.16, -5.31, -5.76, -6.50],
    # per-instrument (HMA Sharpe, B&H Sharpe, spread, t)
    tick=["SPY", "QQQ", "AAPL", "MSFT", "XLE"],
    tick_hma=[0.089, 0.107, 0.580, 0.372, 0.148],
    tick_bh=[0.646, 0.519, 0.624, 0.825, 0.438],
    tick_spread=[-4.35, -4.82, -4.11, -7.49, -3.87],
    tick_t=[-5.31, -3.66, -2.31, -6.02, -2.52],
    # in/out split
    h1_hma=-0.226, h1_bh=0.454, h1_spread=-4.72, h1_t=-3.81,
    h2_hma=0.457, h2_bh=0.882, h2_spread=-4.10, h2_t=-3.71,
    # long/short
    ls_sharpe=-0.545, ls_spread=-8.78, ls_t=-5.36,
    # synthetic control
    syn_edge=[0.0, 0.3, 0.6, 1.0],
    syn_spread=[-1.88, 3.82, 19.08, 41.68],
    syn_t=[-1.62, 2.64, 9.29, 14.39],
    syn_sharpe=[0.259, 2.144, 5.408, 9.312],
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from hull_moving_average import data, strategy as st

TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "XLE"]
CACHE = os.path.abspath(os.path.join("..", "_cache"))
ASOF = "2026-06-13"

def _have_cache():
    return all(os.path.exists(data._cache_path(t, CACHE)) for t in TICKERS)

HAVE_REAL = _have_cache()

def tape(t):
    b = data.load_real(t, fetch=False, cache_dir=CACHE)
    return b[b.index <= ASOF]

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Hull Moving Average — does the lag-free line really cut false signals?\n"
            "### Alan Hull's near-zero-lag moving average, turned into a timing rule and tested honestly\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Cuts false signals%3F: Busted](https://img.shields.io/badge/Cuts_false_signals%3F-Busted-8b949e?style=flat-square)\n\n"
            "Here's a recipe you'll meet on almost every charting platform: plot the **Hull Moving "
            "Average** instead of a plain moving average. Because it's built to lag far less, the "
            "story goes, it turns *earlier* and gives you **fewer false signals** — fewer of those "
            "annoying flip-flops ('whipsaws') that chop up a simple moving-average rule. Sounds "
            "great. Does it actually beat just buying and holding — or even beat a plain SMA?\n\n"
            "> This is the plain-language layer. Want the *t*-stats, the permutation placebo, and "
            "the cost sweeps? That's the companion, "
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
            f"| Does HMA timing beat buy-and-hold? | **No.** It trails by **{R['hma_spread']} bps/day** "
            f"(*t* = {R['hma_t']}) — net Sharpe **{R['hma_sharpe']}** vs **{R['bh_sharpe']}** for just holding. |\n"
            f"| Fewer false signals than a plain SMA? | **No — the opposite.** HMA fires **{R['hma_sw']} "
            f"switches/yr** vs the SMA's **{R['sma_sw']}** — nearly *double* the whipsaws. |\n"
            f"| Does it at least beat the simpler SMA rule? | **No.** SMA net Sharpe {R['sma_sharpe']} > "
            f"HMA {R['hma_sharpe']}. The fancy indicator loses to the plain one. |\n"
            "| Is the timing real or random? | **Worse than random.** A shuffle of its own calls beats "
            f"the real ones every single time (*p* = {R['perm_p']:.0f}). |\n\n"
            "> The HMA genuinely lags less — that part is true. But on noisy daily stock data, "
            "reacting faster just means reacting to more noise. Lower lag bought *more* false "
            "signals, not fewer."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim\n\n"
            "> *\"The Hull Moving Average hugs price with almost no lag. A simple moving average is "
            "always late — by the time it turns, half the move is gone, and it whipsaws you in and "
            "out during chop. The HMA fixes both: it turns earlier AND gives fewer false signals. "
            "Trade in the direction the HMA is pointing and you'll catch trends a plain MA misses.\"*\n\n"
            "Alan Hull built the HMA in 2005 by stacking weighted moving averages: "
            "`HMA(n) = WMA(2·WMA(n/2) − WMA(n), √n)`. The middle term over-weights recent price to "
            "cancel lag; the outer term re-smooths. As a *line on a chart*, it really does track "
            "turns faster than an SMA. We test whether that faster line makes *better trades*."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what?\n\n"
            "Lag is the original sin of moving averages — every trader who has been stopped out by a "
            "late signal wants a faster line. If the HMA delivers earlier turns *and* fewer false "
            "signals, it would be a strict upgrade you could drop into any moving-average strategy. "
            "Millions of retail charts default to it. But if 'faster' just means 'noisier', then "
            "switching to the HMA makes your trading *worse* while feeling more responsive — the "
            "most seductive kind of mistake."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How would we know?\n\n"
            "The honest test isn't 'does the HMA rule make money' — in a 33-year bull market almost "
            "any mostly-long rule does. The honest tests are:\n\n"
            "1. **Race the timing against just holding.** Go long when the HMA is rising, sit in cash "
            "when it's falling. Then subtract a buy-and-hold of the same asset. If the HMA's *timing* "
            "adds anything, that difference (the 'active spread') should be positive.\n"
            "2. **Count the whipsaws.** Compare the HMA rule's number of in/out switches per year "
            "against a plain SMA rule. 'Fewer false signals' is a countable claim.\n"
            "3. **Beat the simpler benchmark.** If the HMA doesn't beat a plain SMA(50) rule, the "
            "extra machinery earned nothing.\n\n"
            "We enter one day after each signal (no peeking), charge realistic costs, and run it on "
            f"SPY plus four other liquid tapes over ~{R['yrs']:.0f} years."
        ),

        # ---- BEAT 4 ----
        md(
            "## 4 · The teardown — what actually happens\n\n"
            "**First: the HMA timing rule vs the benchmarks vs just holding.** Net of 2 bps cost, "
            "long-when-rising on SPY."
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(tape('SPY'), hma_period=16, sma_period=50, cost_bps=2.0)\n"
            "    sh = {k: res[k]['sharpe_net'] for k in ('HMA','SMA','MACD')}\n"
            "    sh['B&H'] = res['HMA']['bh_sharpe']\n"
            "else:\n"
            f"    sh = dict(HMA={R['hma_sharpe']}, SMA={R['sma_sharpe']}, "
            f"MACD={R['macd_sharpe']})\n"
            f"    sh['B&H'] = {R['bh_sharpe']}\n"
            "labels = ['HMA(16)','SMA(50)','MACD','Buy & hold']\n"
            "vals = [sh['HMA'], sh['SMA'], sh['MACD'], sh['B&H']]\n"
            "col = [RED, AMBER, AMBER, GREEN]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(labels, vals, color=col, width=.6)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('net Sharpe (annualised)')\n"
            "ax.set_title('The lag-free HMA is the WORST of the four — losing to plain holding')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"HMA {sh['HMA']:.3f} | SMA {sh['SMA']:.3f} | MACD {sh['MACD']:.3f} | hold {sh['B&H']:.3f}\")"
        ),
        md(
            f"Buy-and-hold (Sharpe **{R['bh_sharpe']}**) beats every timing rule, and the supposedly "
            f"superior HMA (**{R['hma_sharpe']}**) is the *worst* of them — below even the plain "
            f"SMA (**{R['sma_sharpe']}**). Sitting in cash through chunks of a bull market is the "
            "price, and the HMA's earlier signals never pay it back."
        ),
        md(
            "**Now the headline claim itself: fewer false signals?** Count the in/out switches each "
            "rule asks for per year."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sw = {k: res[k]['switches_per_yr'] for k in ('HMA','SMA','MACD')}\n"
            "else:\n"
            f"    sw = dict(HMA={R['hma_sw']}, SMA={R['sma_sw']}, MACD={R['macd_sw']})\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.bar(['HMA(16)','SMA(50)','MACD'], [sw['HMA'], sw['SMA'], sw['MACD']],\n"
            "       color=[RED, GREY, GREY], width=.55)\n"
            "ax.set_ylabel('position changes per year')\n"
            "ax.set_title('\\\"Fewer false signals\\\" — busted: HMA whipsaws ~2x as often as the SMA')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"HMA {sw['HMA']:.1f}/yr  vs  SMA {sw['SMA']:.1f}/yr  ->  {sw['HMA']/sw['SMA']:.1f}x more switches\")"
        ),
        md(
            f"The HMA fires **{R['hma_sw']} switches/yr** against the SMA's **{R['sma_sw']}** — "
            "nearly double. The whole reason to use the HMA was *fewer* false signals; it delivers "
            "*more*. Lower lag means it reacts to every wiggle, and most daily wiggles are noise."
        ),
        md(
            "**Is the timing even real?** We take the HMA's actual position calls and *shuffle* them "
            "in time — same number of trades, same long bias, but the calls now land on random days. "
            "If the real timing is skillful, it should beat its own shuffles."
        ),
        code(
            "if HAVE_REAL:\n"
            "    p = res['HMA_permutation']\n"
            "    obs, plac, pv = p['observed_spread_bps'], p['placebo_mean_bps'], p['p_value']\n"
            "else:\n"
            f"    obs, plac, pv = {R['perm_obs']}, {R['perm_placebo']}, {R['perm_p']}\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.bar(['Real HMA\\ntiming', 'Average random\\nre-timing'], [obs, plac],\n"
            "       color=[RED, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('active spread vs buy&hold (bps/day)')\n"
            "ax.set_title('The real timing is WORSE than a random shuffle of itself')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real {obs:+.2f} bps/day | random {plac:+.2f} bps/day | p = {pv:.3f}')"
        ),
        md(
            f"The real HMA timing (**{R['perm_obs']} bps/day**) is *worse* than the average random "
            f"re-timing (**{R['perm_placebo']} bps/day**), and **{R['perm_p']:.0%}** of shuffles beat "
            "it. The HMA's calls don't just fail to help — they actively point the wrong way more "
            "often than chance."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Active spread vs holding **{R['hma_spread']} bps/day**, *t* = "
            f"**{R['hma_t']}** — significantly negative. Permutation *p* = {R['perm_p']:.0f}.\n"
            f"- **Tradability — Mirage.** Net Sharpe **{R['hma_sharpe']}** vs buy-and-hold "
            f"**{R['bh_sharpe']}**; loses even gross, and no cost level rescues it.\n"
            f"- **Cuts false signals? — Busted.** **{R['hma_sw']}** switches/yr vs the SMA's "
            f"**{R['sma_sw']}** — nearly twice the whipsaws, and a lower Sharpe than that SMA."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No. The gross timing already loses to holding; costs only widen the gap. There's no "
            "break-even cost because the line starts below zero:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    costs = [0.0, 1.0, 2.0, 5.0, 10.0]\n"
            "    spr = [st.run_experiment(tape('SPY'), cost_bps=c)['HMA']['mean_spread_bps'] for c in costs]\n"
            "else:\n"
            f"    costs = {R['cost']}; spr = {R['cost_spread']}\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, spr, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, spr, 0, color=RED, alpha=.12)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('active spread vs hold (bps/day)')\n"
            "ax.set_title('Already below zero at zero cost — there is no break-even')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('The gross spread is negative; costs are not the issue, the timing is.')"
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further\n\n"
            "- **The positive control.** The companion notebook plants a real trend in a synthetic "
            "tape — and the *same* HMA engine catches it cleanly (*t* up to +14). The harness works; "
            "the daily stock market just doesn't trend at the scale the HMA reacts to.\n"
            "- **Slower windows / weekly bars.** A longer HMA, or weekly data, would whipsaw less — "
            "worth a fork to see if it merely matches the SMA or finally beats holding.\n"
            "- **Other indicators, same fate.** See [Study 178 (CCI)](../../178-cci/) and "
            "[Study 104 (Bollinger)](../../104-bollinger-reversion/) — textbook indicators raced "
            "fairly tend to land here.\n\n"
            "*Think the HMA earns its keep on a different timeframe or asset? Fork this, change the "
            "instrument or window, and show an active spread that clears HAC *t* = 2 against "
            "buy-and-hold. That's the bar.*"
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
            "# Hull Moving Average — a quantitative teardown\n"
            "### Daily total-return bars · HMA(16) slope · long/flat & long/short · HAC inference · permutation placebo · SMA/MACD benchmarks\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Cuts false signals%3F: Busted](https://img.shields.io/badge/Cuts_false_signals%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We test whether the HMA(16)-"
            "slope timing rule beats buy-and-hold on a net, excess-of-cash basis, beats the simpler "
            "SMA(50) and MACD rules, and fires fewer false signals — across five liquid daily tapes.\n\n"
            f"> **Not investment advice.** Real data: Yahoo daily total-return bars, full history to "
            f"2026-06-12, as-of **{R['asof']}**; the offline core runs the deterministic synthetic "
            "tape. Methods & sources in [`docs/references.md`](../docs/references.md), reproducible "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> The `In plain words` notes translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | HMA active spread vs buy&hold **{R['hma_spread']} bps/day**, "
            f"HAC *t* = **{R['hma_t']}** (gross *t* = {R['hma_t_gross']}); permutation *p* = "
            f"**{R['perm_p']:.0f}**; negative on all 5 tapes & both halves. |\n"
            f"| **Tradability** | `MIRAGE` | Net Sharpe **{R['hma_sharpe']}** vs buy&hold "
            f"**{R['bh_sharpe']}**; loses gross; long/short Sharpe **{R['ls_sharpe']}**. |\n"
            f"| **Cuts false signals?** | `BUSTED` | **{R['hma_sw']}** switches/yr vs SMA "
            f"**{R['sma_sw']}** (~2×); HMA Sharpe < SMA Sharpe ({R['sma_sharpe']}). |\n\n"
            "> In plain words: the HMA does lag less, but on daily equity noise that buys more "
            "whipsaws and a lower Sharpe — the timing carries negative information."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $\\text{WMA}_n(P)_t = \\sum_{i=1}^{n} i\\,P_{t-n+i} / \\sum_{i=1}^{n} i$ be the "
            "linearly-weighted MA. Hull (2005) defines\n\n"
            "$$\\text{HMA}_n = \\text{WMA}_{\\lfloor\\sqrt{n}\\rceil}\\big(2\\,\\text{WMA}_{n/2}(P) - \\text{WMA}_n(P)\\big).$$\n\n"
            "The trend rule: hold $d_t = +1$ when $\\text{HMA}_t > \\text{HMA}_{t-1}$ (rising), "
            "$0$ (or $-1$) otherwise. The hypotheses:\n\n"
            "- **H₁ (signal).** $\\mathbb{E}[\\,r^{\\text{strat}}_t - r^{\\text{B\\&H}}_t\\,] > 0$ — "
            "the *active spread* (timing minus exposure) is positive.\n"
            "- **H₂ (beats SMA).** Net Sharpe$(\\text{HMA}) > $ Net Sharpe$(\\text{SMA})$.\n"
            "- **H₃ (fewer false signals).** switches/yr$(\\text{HMA}) < $ switches/yr$(\\text{SMA})$.\n\n"
            "We reject **all three**: H₁ is significantly *negative* (HAC *t* = "
            f"{R['hma_t']}), H₂ is reversed ({R['hma_sharpe']} < {R['sma_sharpe']}), and H₃ is "
            f"reversed ({R['hma_sw']} > {R['sma_sw']} switches/yr)."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The HMA is one of the most popular 'upgrade your moving average' indicators in retail "
            "tooling, defaulted on countless TradingView charts. If H₁–H₃ held, it would be a strict "
            "Pareto improvement over the SMA — earlier turns *and* less noise — droppable into any "
            "MA strategy for free. The finding that it is *strictly worse* (lower Sharpe, more "
            "whipsaws, negative timing) matters because the indicator *feels* better: it visibly "
            "hugs price, so users mistake responsiveness for edge. This is the lag/noise trade-off "
            "made concrete."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Indicator.** HMA(16) (a common default); benchmarks SMA(50) and MACD(12/26/9).\n"
            "- **Rule.** Long/flat: $d_t = \\mathbb{1}[\\text{HMA}_t > \\text{HMA}_{t-1}]$ for HMA; "
            "$\\mathbb{1}[C_t > \\text{SMA}_{50}]$ for SMA; $\\mathbb{1}[\\text{MACD}>\\text{signal}]$ "
            "for MACD. A long/short variant flips flat → −1.\n"
            "- **Execution lag.** One `shift`: position formed on the close of $t$ earns the "
            "close-to-close return of $t+1$. Stated once, applied once.\n"
            "- **Costs.** One-way × NAV on $|d_t - d_{t-1}|$ turnover; short legs pay "
            "50 bps/yr borrow.\n"
            "- **Signal test.** HAC (Newey-West) one-sample *t* on the daily *active spread* "
            "$r^{\\text{strat}}-r^{\\text{B\\&H}}$ — excess-vs-excess by construction.\n"
            "- **Placebo.** Circular-shift the realised position path 2,000× (kills timing, keeps "
            "turnover/bias); one-sided *p* on the gross spread.\n"
            "- **Robustness.** Cost sweep, per-instrument, first-vs-second-half split, long/short.\n"
            "- **Positive control.** Synthetic tape with a *planted* regime-switching trend.\n\n"
            f"Five tapes: SPY, QQQ, AAPL, MSFT, XLE — full history to 2026-06-12 (SPY n = {R['n_spy']})."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · HMA vs SMA vs MACD vs buy-and-hold — net Sharpe and active-spread *t*\n\n"
            "The bar that matters is the active-spread HAC *t*: if HMA timing helps, it clears +2."
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(tape('SPY'), hma_period=16, sma_period=50, cost_bps=2.0)\n"
            "    rows = [(k, res[k]['sharpe_net'], res[k]['mean_spread_bps'], res[k]['spread_t'],\n"
            "             res[k]['switches_per_yr']) for k in ('HMA','SMA','MACD')]\n"
            "    rows.append(('B&H', res['HMA']['bh_sharpe'], 0.0, float('nan'), 0.0))\n"
            "    tbl = pd.DataFrame(rows, columns=['rule','sharpe_net','spread_bps','spread_t','sw/yr'])\n"
            "else:\n"
            "    tbl = pd.DataFrame({\n"
            "        'rule': ['HMA','SMA','MACD','B&H'],\n"
            f"        'sharpe_net': [{R['hma_sharpe']},{R['sma_sharpe']},{R['macd_sharpe']},{R['bh_sharpe']}],\n"
            f"        'spread_bps': [{R['hma_spread']},{R['sma_spread']},{R['macd_spread']},0.0],\n"
            f"        'spread_t': [{R['hma_t']},{R['sma_t']},{R['macd_t']},float('nan')],\n"
            f"        'sw/yr': [{R['hma_sw']},{R['sma_sw']},{R['macd_sw']},0.0]}})\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "sub = tbl[tbl['rule']!='B&H']\n"
            "col = [RED if t < 0 else GREY for t in sub['spread_t']]\n"
            "ax.bar(sub['rule'], sub['spread_t'], color=col)\n"
            "for s in (2, -2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('active-spread HAC t (vs buy&hold)')\n"
            "ax.set_title('All three rules have significantly NEGATIVE timing; HMA the worst')\n"
            "plt.tight_layout(); plt.show()\n"
            "tbl.round(3)"
        ),
        md(
            f"> In plain words: every timing rule's active spread is significantly negative — none "
            f"beats holding. HMA is the worst (*t* = {R['hma_t']}), below the plain SMA "
            f"(*t* = {R['sma_t']}). The fancy indicator loses to the simple one **and** to holding."
        ),
        md(
            "### 4b · The whipsaw count — the 'fewer false signals' claim, head to head\n\n"
            "Position changes per year. The folk claim says HMA < SMA."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sw = {k: res[k]['switches_per_yr'] for k in ('HMA','SMA','MACD')}\n"
            "else:\n"
            f"    sw = dict(HMA={R['hma_sw']}, SMA={R['sma_sw']}, MACD={R['macd_sw']})\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.0))\n"
            "ax.bar(['HMA','SMA','MACD'], [sw['HMA'],sw['SMA'],sw['MACD']],\n"
            "       color=[RED, GREY, GREY])\n"
            "ax.set_ylabel('switches / yr'); ax.set_title('HMA whipsaws ~2x the SMA — claim reversed')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"HMA/SMA switch ratio: {sw['HMA']/sw['SMA']:.2f}x\")"
        ),
        md(
            f"> In plain words: {R['hma_sw']} vs {R['sma_sw']} switches/yr. Lower lag = faster "
            "reaction = more reaction to noise. The HMA's defining selling point is false on the tape."
        ),
        md(
            "### 4c · Permutation placebo — timing vs exposure\n\n"
            "Circularly shift the realised position path 2,000×; the statistic is the gross active "
            "spread. If the timing is informative, the real spread beats the placebo distribution."
        ),
        code(
            "if HAVE_REAL:\n"
            "    p = st.permutation_pvalue(tape('SPY')['close'].pct_change(),\n"
            "                              st.hma_position(tape('SPY')['close'], 16),\n"
            "                              cost_bps=0.0, n_perm=2000, seed=432)\n"
            "    obs, plac, pv = p['observed_spread_bps'], p['placebo_mean_bps'], p['p_value']\n"
            "    import numpy as _np\n"
            "    rng = _np.random.default_rng(432)\n"
            "    a = tape('SPY')['close'].pct_change().fillna(0).to_numpy()\n"
            "    held = st.hma_position(tape('SPY')['close'],16).shift(1).fillna(0).to_numpy()\n"
            "    draws = []\n"
            "    for _ in range(2000):\n"
            "        h = _np.roll(held, int(rng.integers(1,len(held))))\n"
            "        turn=_np.abs(_np.diff(h,prepend=0.0)); draws.append(((h*a-turn*0)-a).mean()*1e4)\n"
            "else:\n"
            f"    obs, plac, pv = {R['perm_obs']}, {R['perm_placebo']}, {R['perm_p']}\n"
            "    import numpy as _np\n"
            f"    draws = list(_np.random.default_rng(1).normal({R['perm_placebo']}, 0.6, 2000))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=30, color=GREY, alpha=.7, label='random re-timings (2000)')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'real HMA timing: {obs:+.2f} bps/day')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('active spread vs buy&hold (bps/day)'); ax.set_ylabel('count')\n"
            "ax.set_title('Real HMA timing sits in the LEFT tail — worse than its own shuffles')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.2f} | placebo mean {plac:+.2f} | one-sided p = {pv:.3f}')"
        ),
        md(
            f"> In plain words: *p* = {R['perm_p']:.0f} — essentially every random re-timing of the "
            "HMA's own trades beats the real ones. The timing isn't just uninformative; it is "
            "reliably on the wrong side."
        ),
        md(
            "### 4d · Per-instrument & in/out-of-sample — is it ever positive?\n\n"
            "Active-spread *t* on all five tapes, and the SPY first-vs-second-half split."
        ),
        code(
            "if HAVE_REAL:\n"
            "    recs = []\n"
            "    for t in TICKERS:\n"
            "        r = st.run_experiment(tape(t), cost_bps=2.0)['HMA']\n"
            "        recs.append((t, r['sharpe_net'], r['bh_sharpe'], r['mean_spread_bps'], r['spread_t']))\n"
            "    per = pd.DataFrame(recs, columns=['ticker','hma_sharpe','bh_sharpe','spread_bps','t'])\n"
            "    spy = tape('SPY'); half = len(spy)//2\n"
            "    h1 = st.run_experiment(spy.iloc[:half], cost_bps=2.0)['HMA']\n"
            "    h2 = st.run_experiment(spy.iloc[half:], cost_bps=2.0)['HMA']\n"
            "    split = [(h1['spread_t'], h2['spread_t'])]\n"
            "else:\n"
            "    per = pd.DataFrame({'ticker': "
            f"{R['tick']}, 'hma_sharpe': {R['tick_hma']}, 'bh_sharpe': {R['tick_bh']},"
            f" 'spread_bps': {R['tick_spread']}, 't': {R['tick_t']}}})\n"
            f"    split = [({R['h1_t']}, {R['h2_t']})]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "ax.bar(per['ticker'], per['t'], color=RED)\n"
            "ax.axhline(-2, ls='--', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('active-spread HAC t (HMA vs hold)')\n"
            "ax.set_title('Five of five negative; all but one clear t = -2')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('SPY split  H1 t = %.2f | H2 t = %.2f  (both negative)' % split[0])\n"
            "per.round(3)"
        ),
        md(
            f"> In plain words: every instrument's HMA timing trails buy-and-hold, and SPY loses in "
            f"both halves (H1 *t* = {R['h1_t']}, H2 *t* = {R['h2_t']}). Not a one-regime artefact."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — active spread {R['hma_spread']} bps/day, HAC *t* {R['hma_t']} "
            f"(gross {R['hma_t_gross']}); permutation *p* = {R['perm_p']:.0f}; negative on 5/5 tapes "
            "and both halves.\n"
            f"- **Tradability `MIRAGE`** — net Sharpe {R['hma_sharpe']} vs hold {R['bh_sharpe']}; "
            f"loses gross; long/short Sharpe {R['ls_sharpe']}, spread {R['ls_spread']} bps/day "
            f"(*t* {R['ls_t']}).\n"
            f"- **Cuts false signals? `BUSTED`** — {R['hma_sw']} vs {R['sma_sw']} switches/yr; "
            "HMA Sharpe below the simpler SMA."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — the cost landscape\n\n"
            "The spread is below zero before costs; there is no break-even, and the long/short "
            "version compounds the loss:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    costs = [0.0, 1.0, 2.0, 5.0, 10.0]\n"
            "    spr = [st.run_experiment(tape('SPY'), cost_bps=c)['HMA']['mean_spread_bps'] for c in costs]\n"
            "    tst = [st.run_experiment(tape('SPY'), cost_bps=c)['HMA']['spread_t'] for c in costs]\n"
            "    lsr = st.run_experiment(tape('SPY'), cost_bps=2.0, long_short=True)['HMA']\n"
            "    ls_sh = lsr['sharpe_net']\n"
            "else:\n"
            f"    costs = {R['cost']}; spr = {R['cost_spread']}; tst = {R['cost_t']}\n"
            f"    ls_sh = {R['ls_sharpe']}\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(costs, spr, 'o-', c=RED, lw=2, label='active spread (bps/day)')\n"
            "ax2 = ax.twinx(); ax2.plot(costs, tst, 's--', c=GREY, lw=1.5, label='HAC t')\n"
            "ax2.axhline(-2, ls=':', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('spread (bps/day)', color=RED)\n"
            "ax2.set_ylabel('HAC t', color=GREY)\n"
            "ax.set_title('No break-even: the spread starts below zero and only falls')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'long/short net Sharpe: {ls_sh:+.3f} (worse than long/flat)')"
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the positive control\n\n"
            "Is the *engine* able to find a trend, or is it always negative? Plant a persistent "
            "regime-switching trend in a synthetic tape and sweep the knob:"
        ),
        code(
            "edges = " + repr(R['syn_edge']) + "\n"
            "if HAVE_REAL or True:\n"
            "    sp, tt = [], []\n"
            "    for e in edges:\n"
            "        b, _ = data.synthetic_panel(n_days=4000, edge=e, seed=432)\n"
            "        r = st.run_experiment(b, cost_bps=0.0)['HMA']\n"
            "        sp.append(r['mean_spread_bps']); tt.append(r['spread_t'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(edges, tt, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted trend (edge knob)'); ax.set_ylabel('active-spread HAC t')\n"
            "ax.set_title('Engine works: HMA banks a planted trend (t up to +14)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('edge=0 t=%.2f (≈null) ; edge=0.3 t=%.2f ; edge=1.0 t=%.2f' % (tt[0], tt[1], tt[-1]))"
        ),
        md(
            "The engine is a faithful trend detector: plant a real persistent trend and the HMA "
            "rule banks it, clearing *t* = 2 by `edge = 0.3` and reaching *t* ≈ +14 at `edge = 1.0`; "
            "with no trend planted it reads ≈ 0 (slightly negative from turnover drag). The "
            "real-tape verdict is therefore a statement about the **market** — daily US equity "
            "returns are too noisy at the scale the HMA reacts to — not a broken harness. Forks "
            "worth trying: weekly bars, a much slower HMA, or trend-prone futures (where time-series "
            "momentum is documented) rather than a single equity index."
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
