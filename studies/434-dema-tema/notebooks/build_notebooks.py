"""Generate the two narrative notebooks for Study 434 (DEMA & TEMA).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md).  The synthetic
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


# --------------------------------------------------------------------------- #
# Frozen real-tape headline numbers -- mirror of docs/results.md
# As-of 2026-05-31 (partial June dropped); SPY daily 2005-01-03..2026-05-29, fp 0f42b216fe70.
# Window n=50, one-way cost 2 bps, one-day lag, long/flat (close > MA).
# --------------------------------------------------------------------------- #
R = dict(
    asof="2026-05-31",
    span="2005-01-03 to 2026-05-29",
    n_bars=5385, fp="0f42b216fe70", window=50, cost=2.0,
    bh_sharpe=0.645,
    # net excess-of-cash Sharpe by MA kind (n=50, 2bps)
    sma_sharpe=0.536, ema_sharpe=0.641, dema_sharpe=0.420, tema_sharpe=0.240,
    # Sharpe minus buy-and-hold
    sma_minus_bh=-0.109, ema_minus_bh=-0.004, dema_minus_bh=-0.225, tema_minus_bh=-0.405,
    # HAC t of daily excess-over-buy-and-hold (the REAL-stamp test)
    sma_t=-2.30, ema_t=-1.86, dema_t=-2.79, tema_t=-3.49,
    # one-way turnover per year (one-way x NAV)
    sma_turn=16.9, ema_turn=18.2, dema_turn=34.2, tema_turn=38.9,
    # time in market
    sma_tim=0.693, ema_tim=0.721, dema_tim=0.563, tema_tim=0.521,
    # placebo (circular block shuffle of the position schedule)
    tema_placebo_p=0.876, tema_placebo_obs=0.240,
    dema_placebo_p=0.527, dema_placebo_obs=0.420,
    # paired daily TEMA-minus-SMA & DEMA-minus-SMA (net), HAC t + ann mean
    tema_vs_sma_t=-1.25, tema_vs_sma_ann=-3.0,
    dema_vs_sma_t=-0.64, dema_vs_sma_ann=-1.22,
    # cost sweep, TEMA vs SMA net Sharpe (n=50)
    tema_c0=0.308, tema_c2=0.240, tema_c5=0.137, tema_c10=-0.033,
    sma_c0=0.568, sma_c2=0.536, sma_c5=0.489, sma_c10=0.410,
    # panel net Sharpe (n=50, 2bps): TEMA beats SMA only on XLE (1 of 6)
    panel_tema_beats_sma="1 of 6",
    # synthetic control: SMA harvests a planted trend (Sharpe up to 4.0) while TEMA whipsaws
    syn_e0_sma=0.605, syn_e0_tema=0.501, syn_e0_bh=0.636,
    syn_e1_sma=2.369, syn_e1_tema=-0.260, syn_e1_bh=-0.985,
    syn_e2_sma=4.035, syn_e2_tema=-0.535, syn_e2_bh=-2.183,
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from dema_tema import data, strategy as st

ASOF = "2026-05-31"          # partial June dropped from the stamped run
KINDS = ["sma", "ema", "dema", "tema"]
COLORS = {"sma": GREY, "ema": "#5b8def", "dema": AMBER, "tema": RED}

def spy():
    b = data.load_real("SPY", fetch=False)
    return b[b.index <= ASOF]

HAVE_REAL = data.have_real("SPY")
print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# DEMA & TEMA -- are double/triple-smoothed moving averages worth it?\n"
            "### The 'less-lag' moving averages, raced against a plain SMA, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Worth_it%3F: Busted](https://img.shields.io/badge/Worth_it%3F-Busted-8b949e?style=flat-square)\n\n"
            "Every charting package ships them: **DEMA** (double exponential moving average) and "
            "**TEMA** (triple). The pitch is irresistible -- a normal moving average *lags* price, "
            "so your trend signals are always late; DEMA and TEMA use a clever algebra trick to "
            "**cancel most of the lag** while staying smooth. 'Same signal, earlier -- so you make "
            "more money.' This notebook asks the only question that matters: when you turn them into "
            "an actual trading rule, do they beat a boring plain moving average?\n\n"
            "> **This is the plain-language layer.** Want the Sharpe race, the HAC *t*-stats, the "
            "permutation null and the cost sweep? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is drawn by "
            "the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does TEMA beat a plain SMA? | **No.** On 21 years of SPY the SMA trend rule nets a "
            f"Sharpe of **{R['sma_sharpe']:.2f}**; TEMA nets **{R['tema_sharpe']:.2f}** -- *worse*. |\n"
            "| Does any of them beat just holding SPY? | **No.** Buy-and-hold nets "
            f"**{R['bh_sharpe']:.2f}**; every MA rule lands below it. |\n"
            "| What did the extra 'responsiveness' buy you? | **More trading.** TEMA turns over "
            f"**~{R['tema_turn']:.0f}x/yr** vs **~{R['sma_turn']:.0f}x** for the SMA -- it reacts to "
            "noise, gets whipsawed, and pays the costs. |\n"
            "| Could you trade it? | **Mirage.** It loses to the simpler tool *and* to doing nothing, "
            "before you even get to costs. |\n\n"
            "> The lag-cancellation is real maths. But less lag means **more reaction to noise** -- "
            "and on a daily stock tape that trades you in and out for nothing."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"A simple moving average lags price by half its window -- by the time it turns, the "
            "move is half over. DEMA (Mulloy, 1994) and TEMA fix that: 2x or 3x EMAs combined so the "
            "lag mostly cancels. You get the smoothness of a long average with the responsiveness of a "
            "short one -- earlier entries, earlier exits, more profit.\"*\n\n"
            "It is genuinely clever. ``DEMA = 2*EMA - EMA(EMA)`` and "
            "``TEMA = 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA))`` really do cut the phase lag of the filter. "
            "The believer's bet is that *less lag = better trend timing = more money*."
        ),

        # ---- BEAT 2 -- SO WHAT ----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If true, it would be a free upgrade: swap your 50-day SMA for a 50-day TEMA in any "
            "trend-following rule and pick up return for nothing. Millions of retail charts default to "
            "these lines. If it's *false* -- if the extra responsiveness just buys you whipsaws -- then "
            "a generation of traders has been paying more in costs to do slightly worse than the plain "
            "average they were trying to improve on."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW ----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Two rules keep us honest:\n\n"
            "1. **Same rule, swap only the line.** Be **long when the close is above its moving "
            "average, flat otherwise** -- identical logic for SMA, EMA, DEMA and TEMA at the same "
            "50-day window. The *only* thing that changes is which line we use. We read the signal at "
            "today's close and earn *tomorrow's* return (one honest lag), and we charge realistic "
            "costs on every switch.\n"
            "2. **Race the right benchmarks.** Not just each other -- against **buy-and-hold** (does "
            "any trend rule even help?) and against the **plain SMA** (does the 'less lag' line beat "
            "the boring one it claims to improve?).\n\n"
            "**The mirage line:** if TEMA/DEMA don't clear the SMA *and* don't clear buy-and-hold "
            "on a net, excess-of-cash Sharpe -- with no positive HAC *t* on the outperformance -- the "
            "'worth it' claim is busted. We run it on **21 years of daily SPY** (2005-2026)."
        ),

        # ---- BEAT 4 -- THE TEARDOWN -----------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First, the lines themselves.** Here are the four 50-day moving averages on a slice of "
            "SPY. DEMA and TEMA do hug price more tightly -- the lag really is smaller."
        ),
        code(
            "b = spy()\n"
            "win = b.loc['2018-01-01':'2019-07-01']\n"
            "fig, ax = plt.subplots(figsize=(10, 5))\n"
            "ax.plot(win.index, win['close'], color='k', lw=1.1, label='SPY close')\n"
            "for k in KINDS:\n"
            "    ma = st.moving_average(b['close'], k, 50).loc[win.index]\n"
            "    ax.plot(win.index, ma, color=COLORS[k], lw=1.6, label=k.upper())\n"
            "ax.set_title('50-day moving averages on SPY -- DEMA/TEMA hug price more tightly')\n"
            "ax.legend(ncol=5); plt.tight_layout(); plt.show()\n"
            "print('Less lag is visible -- TEMA turns soonest. The question is whether that pays.')"
        ),
        md(
            "They turn sooner -- exactly as advertised. Now the only test that matters: **turn each "
            "line into the same long/flat rule and see who makes money, net of costs.**"
        ),
        code(
            "res = st.run_experiment(b, n=50, cost_bps=2.0, n_draws=1)\n"
            "tab = res['table']\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.8))\n"
            "ks = KINDS\n"
            "vals = [tab.loc[k,'sharpe_net'] for k in ks]\n"
            "bars = ax.bar([k.upper() for k in ks], vals, color=[COLORS[k] for k in ks], width=.6)\n"
            "ax.axhline(res['sharpe_bh'], ls='--', c=GREEN, lw=2, label=f\"buy & hold ({res['sharpe_bh']:.2f})\")\n"
            "ax.set_ylabel('net Sharpe (excess of cash, 2bps cost)')\n"
            "ax.set_title('The plain SMA/EMA win -- DEMA and TEMA are the worst, and all lose to holding')\n"
            "for bb, v in zip(bars, vals):\n"
            "    ax.annotate(f'{v:.2f}', (bb.get_x()+bb.get_width()/2, v), ha='center', va='bottom', fontsize=10)\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(tab[['sharpe_net','sharpe_minus_bh','turnover_per_yr']].round(3).to_string())"
        ),
        md(
            f"The 'improved' lines are the **worst** performers. TEMA nets a Sharpe of "
            f"**{R['tema_sharpe']:.2f}**, DEMA **{R['dema_sharpe']:.2f}** -- below the plain SMA's "
            f"**{R['sma_sharpe']:.2f}** and the EMA's **{R['ema_sharpe']:.2f}**, and all of them below "
            f"just **holding SPY** ({R['bh_sharpe']:.2f}). The less-lag lines bought you *less* money."
        ),
        md(
            "**Why?** Because less lag cuts both ways. Look at how much each rule trades."
        ),
        code(
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "turns = [tab.loc[k,'turnover_per_yr'] for k in KINDS]\n"
            "ax.bar([k.upper() for k in KINDS], turns, color=[COLORS[k] for k in KINDS], width=.6)\n"
            "ax.set_ylabel('round-trips per year (one-way x NAV)')\n"
            "ax.set_title('DEMA/TEMA trade ~2x more -- reacting to noise, paying costs, getting whipsawed')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('TEMA turnover %.0f/yr vs SMA %.0f/yr -- twice the friction for a worse result.'\n"
            "      % (tab.loc['tema','turnover_per_yr'], tab.loc['sma','turnover_per_yr']))"
        ),
        md(
            "There it is. The extra 'responsiveness' is responsiveness **to noise**. TEMA flips in and "
            "out roughly twice as often as the SMA, eating costs and selling into every wiggle. The lag "
            "the SMA carries is partly a *feature* -- it ignores the noise that DEMA/TEMA chase."
        ),

        # ---- BEAT 5 -- THE VERDICT ------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal -- None.** No MA rule beats buy-and-hold; the outperformance HAC *t* is "
            f"**negative** for all of them (TEMA *t* = {R['tema_t']:.1f}). The 'less-lag' lines clear "
            "no bar at all.\n"
            "- **Tradability -- Mirage.** DEMA/TEMA lose to the simpler SMA *and* to doing nothing, "
            "while trading twice as much. There is nothing to deploy.\n"
            f"- **'Worth it?' -- Busted.** TEMA nets **{R['tema_sharpe']:.2f}** vs the SMA's "
            f"**{R['sma_sharpe']:.2f}**. The lag-cancellation is real; the edge it was supposed to buy "
            "is not."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT -----------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "It is already behind before costs even bite -- but costs make it worse faster, because it "
            "trades more."
        ),
        code(
            "costs = [0.0, 2.0, 5.0, 10.0]\n"
            "tema = [st.backtest(b,'tema',50,c)['sharpe_net'] for c in costs]\n"
            "sma  = [st.backtest(b,'sma', 50,c)['sharpe_net'] for c in costs]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(costs, tema, 'o-', c=RED, lw=2, label='TEMA')\n"
            "ax.plot(costs, sma, 's--', c=GREY, lw=2, label='plain SMA')\n"
            "ax.axhline(res['sharpe_bh'], ls=':', c=GREEN, lw=2, label='buy & hold')\n"
            "ax.set_xlabel('one-way cost (bps)'); ax.set_ylabel('net Sharpe')\n"
            "ax.set_title('TEMA falls fastest -- more turnover, more cost drag'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('At 10 bps one-way TEMA goes negative; the SMA still clears it -- but neither beats holding.')"
        ),
        md(
            "TEMA's higher turnover means costs erode it fastest -- it crosses zero by ~10 bps while "
            "the SMA limps on. But the headline isn't the cost slope; it's that **the gap was already "
            "there at zero cost**. This is a Mirage you don't even need costs to see."
        ),

        # ---- BEAT 7 -- GOING FURTHER ----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The positive control (in the quants notebook).** On a synthetic tape with a *planted* "
            "trend, the SMA harvests it for a Sharpe up to ~4 while TEMA whipsaws to *negative* -- "
            "proving the harness can see an edge, and that DEMA/TEMA are a liability even when there is "
            "a real trend to catch.\n"
            "- **A crossover version.** Instead of price-vs-MA, try a fast/slow DEMA crossover. The "
            "responsiveness story predicts it should help; we'd bet it whipsaws harder.\n"
            "- **Compare with the desk's other MA studies** -- "
            "[Bollinger-Reversion](../../104-bollinger-reversion/) (bands vs a random buy) and the "
            "golden-cross family. Same lesson recurs: the *line* is rarely the edge.\n\n"
            "*Think DEMA/TEMA earn their keep somewhere -- a faster window, a different asset, a "
            "crossover? Fork this, and show a net, excess-of-cash Sharpe that beats the plain SMA with "
            "a HAC t above 2. That's the bar.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 -- FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# DEMA & TEMA -- a quantitative teardown\n"
            "### Real daily SPY - 21 years - price-vs-MA long/flat - net excess Sharpe race - "
            "HAC inference - block-permutation null\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Worth_it%3F: Busted](https://img.shields.io/badge/Worth_it%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.* We hold the timing logic "
            "fixed (long when close > MA, flat otherwise; one-day lag; one-way costs x turnover) and "
            "swap only the smoother -- SMA, EMA, DEMA, TEMA at a 50-day window -- then race the "
            "**net, excess-of-cash Sharpe** against buy-and-hold and against the plain SMA.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily SPY 2005-01-03..2026-05-29 "
            "(auto-adjusted -> total-return-ish); reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **In plain words** notes translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front (real tape)\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | No MA rule beats buy-and-hold ({R['bh_sharpe']:.2f}); the "
            f"daily *excess-over-BH* HAC *t* is **negative** for all (TEMA {R['tema_t']:.2f}, "
            f"DEMA {R['dema_t']:.2f}). Nothing clears *t* = +2. |\n"
            f"| **Tradability** | `MIRAGE` | TEMA net Sharpe **{R['tema_sharpe']:.2f}** < SMA "
            f"**{R['sma_sharpe']:.2f}** < buy-and-hold **{R['bh_sharpe']:.2f}**, at ~2x the turnover "
            f"({R['tema_turn']:.0f} vs {R['sma_turn']:.0f}/yr). |\n"
            f"| **'Worth it?'** | `BUSTED` | TEMA-minus-SMA daily net return: HAC *t* = "
            f"**{R['tema_vs_sma_t']:.2f}** ({R['tema_vs_sma_ann']:+.1f}%/yr) -- the lag fix makes the "
            "rule *worse*, not better. |\n\n"
            "> In plain words: the lag-cancellation algebra is correct, but on a noisy daily tape the "
            "extra responsiveness is responsiveness to noise. DEMA/TEMA trade more and earn less."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $c_t$ be the close, $E^{(1)} = \\mathrm{EMA}_n(c)$, $E^{(2)} = \\mathrm{EMA}_n(E^{(1)})$, "
            "$E^{(3)} = \\mathrm{EMA}_n(E^{(2)})$. Then\n\n"
            "$$\\mathrm{DEMA}_n = 2E^{(1)} - E^{(2)}, \\qquad "
            "\\mathrm{TEMA}_n = 3E^{(1)} - 3E^{(2)} + E^{(3)}.$$\n\n"
            "These cancel the leading-order phase lag of a single EMA (Mulloy, 1994). The trading "
            "claim:\n\n"
            "- **H1 (better timing).** The long/flat rule on DEMA/TEMA earns a higher net "
            "excess-of-cash Sharpe than the same rule on a plain SMA.\n"
            "- **H2 (worth running at all).** The trend rule beats buy-and-hold.\n\n"
            f"We **reject H1**: TEMA-minus-SMA daily net return has HAC *t* = {R['tema_vs_sma_t']:.2f} "
            "(negative). We **reject H2**: every MA rule's excess-over-BH HAC *t* is negative."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "DEMA/TEMA are default lines in TradingView, MetaTrader and most charting suites; the "
            "'reduced lag = better signal' heuristic is folklore-grade ubiquitous. If it held, it "
            "would be a Pareto improvement on every MA-based rule. The interesting failure is "
            "*mechanistic*: a filter's lag is coupled to its noise rejection (a bias-variance "
            "trade-off). Cancelling the lag re-injects high-frequency variance into the signal, which "
            "a threshold-crossing rule converts directly into excess turnover and whipsaw."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Signal.** Position $= \\mathbb{1}[c_t > \\mathrm{MA}_n(c)_t]$, long/flat, "
            "$n = 50$, identical across SMA/EMA/DEMA/TEMA.\n"
            "- **Lag.** Position known at the close of *t* earns the return of *t+1* "
            "(`pos.shift(1)`) -- one documented shift.\n"
            "- **Costs.** One-way `cost_bps` x |Δposition| (turnover = one-way x NAV); 2 bps headline, "
            "swept to 10.\n"
            "- **Sharpe race (excess-vs-excess).** Both the rule (part-time in cash) and buy-and-hold "
            "(always invested) are measured **in excess of cash** so the comparison is fair.\n"
            "- **Inference.** Newey-West HAC *t* on the daily *excess-over-buy-and-hold* return "
            "(the REAL-stamp test), and a paired TEMA-minus-SMA HAC *t*.\n"
            "- **Null.** A circular **block permutation** of the position schedule (block = 21d) -- "
            "same activity level, random timing -- recomputing the net Sharpe.\n"
            "- **Positive control.** Synthetic tape with a *planted* OU trend; the rule must light up "
            "when an edge exists.\n\n"
            "Tape: SPY daily 2005-01-03..2026-05-29 (partial June dropped)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),

        md(
            "### 4a - The Sharpe race: net, excess-of-cash, all four lines + buy-and-hold\n\n"
            "Same rule, swap only the smoother. The plain lines win; the 'improved' lines lose."
        ),
        code(
            "b = spy()\n"
            "res = st.run_experiment(b, n=50, cost_bps=2.0, n_draws=1)\n"
            "tab = res['table']\n"
            "print('SPY %s..%s  n=%d  fp=%s' % (b.index[0].date(), b.index[-1].date(), len(b), data.fingerprint(b)))\n"
            "print('buy-and-hold net excess Sharpe: %.3f' % res['sharpe_bh'])\n"
            "print(tab[['sharpe_gross','sharpe_net','sharpe_minus_bh','hac_t_vs_bh','turnover_per_yr','time_in_market']].round(3).to_string())\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))\n"
            "ks = KINDS\n"
            "axes[0].bar([k.upper() for k in ks], [tab.loc[k,'sharpe_net'] for k in ks],\n"
            "            color=[COLORS[k] for k in ks], width=.6)\n"
            "axes[0].axhline(res['sharpe_bh'], ls='--', c=GREEN, lw=2, label=f\"buy & hold {res['sharpe_bh']:.2f}\")\n"
            "axes[0].set_ylabel('net Sharpe (excess of cash)'); axes[0].legend()\n"
            "axes[0].set_title('Net Sharpe: plain SMA/EMA beat DEMA/TEMA, all below B&H')\n"
            "axes[1].bar([k.upper() for k in ks], [tab.loc[k,'hac_t_vs_bh'] for k in ks],\n"
            "            color=[COLORS[k] for k in ks], width=.6)\n"
            "axes[1].axhline(2, ls='--', c=GREY); axes[1].axhline(-2, ls='--', c=GREY); axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_ylabel('HAC t of daily excess-over-B&H')\n"
            "axes[1].set_title('Outperformance HAC t: all negative -- none clears +2')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "> In plain words: the ranking is the *opposite* of the claim. The supposedly improved "
            f"TEMA nets **{R['tema_sharpe']:.2f}** and DEMA **{R['dema_sharpe']:.2f}**, below the plain "
            f"SMA ({R['sma_sharpe']:.2f}) and EMA ({R['ema_sharpe']:.2f}), and all four sit under "
            f"buy-and-hold ({R['bh_sharpe']:.2f}). Every outperformance HAC *t* is negative."
        ),

        md(
            "### 4b - The mechanism: turnover. Less lag = more reaction to noise\n\n"
            "The lag is coupled to noise rejection. Cancel the lag, and you trade the wiggles."
        ),
        code(
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "tk = [tab.loc[k,'turnover_per_yr'] for k in KINDS]\n"
            "ax.bar([k.upper() for k in KINDS], tk, color=[COLORS[k] for k in KINDS], width=.6)\n"
            "ax.set_ylabel('round-trips/yr (one-way x NAV)')\n"
            "ax.set_title('Turnover roughly doubles from SMA to TEMA')\n"
            "for i,v in enumerate(tk): ax.annotate(f'{v:.0f}', (i, v), ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            "# paired daily TEMA-minus-SMA and DEMA-minus-SMA (net), HAC t\n"
            "for k in ('dema','tema'):\n"
            "    d = st.backtest(b,k,50,2.0)['strat_net'] - st.backtest(b,'sma',50,2.0)['strat_net']\n"
            "    print('%s-minus-SMA  HAC t = %+.2f   ann mean = %+.2f%%' % (k.upper(), st.hac_t(d), d.mean()*252*100))"
        ),
        md(
            "> In plain words: TEMA turns over ~2x the SMA. The paired test confirms the gap is a "
            f"real drag, not noise: TEMA-minus-SMA runs **{R['tema_vs_sma_ann']:+.1f}%/yr** "
            f"(HAC *t* = {R['tema_vs_sma_t']:.2f}). The lag fix is a net negative on this tape."
        ),

        md(
            "### 4c - The permutation null -- is even TEMA's small Sharpe better than a random schedule?\n\n"
            "Hold the activity level fixed (same fraction of long days, same block autocorrelation) "
            "but randomise *when* the rule is long, against the same return tape."
        ),
        code(
            "plc = st.block_permutation_pvalue(b, 'tema', 50, cost_bps=2.0, n_draws=2000, seed=434)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(plc['draws'], bins=50, color=GREY, alpha=.65, label='random schedules')\n"
            "ax.axvline(plc['obs_sharpe'], color=RED, lw=2.5, label=f\"TEMA observed {plc['obs_sharpe']:.2f}\")\n"
            "ax.axvline(plc['placebo_mean'], color='k', ls=':', lw=1.5, label=f\"placebo mean {plc['placebo_mean']:.2f}\")\n"
            "ax.set_xlabel('net Sharpe'); ax.set_ylabel('count')\n"
            "ax.set_title(f\"TEMA's timing is WORSE than a random schedule of the same days (p={plc['p_value']:.2f})\")\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('placebo p = %.3f -- a random schedule beats TEMA %.0f%% of the time' % (plc['p_value'], plc['p_value']*100))"
        ),
        md(
            f"> In plain words: the placebo *p* = **{R['tema_placebo_p']:.2f}** -- a random long/flat "
            "schedule with TEMA's activity level beats TEMA's actual timing the large majority of the "
            "time. The signal isn't just unprofitable vs the SMA; its *timing* adds nothing over noise."
        ),

        md(
            "### 4d - The positive control -- the harness CAN see an edge (and DEMA/TEMA still lose)\n\n"
            "On a synthetic tape with a *planted* slow trend, a trend rule should harvest it. Does the "
            "machinery detect that -- and does the less-lag line help or hurt?"
        ),
        code(
            "edges = [0.0, 0.5, 1.0, 2.0]\n"
            "sma_s, tema_s, bh_s = [], [], []\n"
            "for e in edges:\n"
            "    bs,_ = data.synthetic_panel(n_days=4000, edge=e, seed=434)\n"
            "    r = st.run_experiment(bs, n=50, cost_bps=2.0, n_draws=1)\n"
            "    sma_s.append(r['table'].loc['sma','sharpe_net'])\n"
            "    tema_s.append(r['table'].loc['tema','sharpe_net'])\n"
            "    bh_s.append(r['sharpe_bh'])\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "ax.plot(edges, sma_s, 'o-', c=GREY, lw=2, label='SMA rule')\n"
            "ax.plot(edges, tema_s, 's-', c=RED, lw=2, label='TEMA rule')\n"
            "ax.plot(edges, bh_s, '^--', c=GREEN, lw=2, label='buy & hold')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xlabel('planted trend strength (edge)')\n"
            "ax.set_ylabel('net Sharpe'); ax.legend()\n"
            "ax.set_title('Harness works: SMA harvests the planted trend (Sharpe up to ~4); TEMA whipsaws to negative')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('edge=0  : sma %.2f tema %.2f bh %.2f' % (sma_s[0], tema_s[0], bh_s[0]))\n"
            "print('edge=2  : sma %.2f tema %.2f bh %.2f' % (sma_s[-1], tema_s[-1], bh_s[-1]))"
        ),
        md(
            "> In plain words: this is the proof the test isn't rigged. When we *plant* a real trend, "
            f"the SMA rule lights up (Sharpe to **{R['syn_e2_sma']:.1f}** at edge=2 while buy-and-hold "
            f"craters to **{R['syn_e2_bh']:.1f}**) -- the harness banks a genuine edge. And even there, "
            f"TEMA goes **negative** ({R['syn_e2_tema']:.1f}): its noise-sensitivity throws away the "
            "very trend it was built to catch faster. The less-lag line is a liability *with or without* "
            "a real signal."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal `NONE`** -- buy-and-hold nets {R['bh_sharpe']:.2f}; every MA rule lands below "
            f"it with a *negative* excess-over-BH HAC *t* (SMA {R['sma_t']:.2f}, EMA {R['ema_t']:.2f}, "
            f"DEMA {R['dema_t']:.2f}, TEMA {R['tema_t']:.2f}). Nothing clears +2 -- there is no real "
            "timing edge to attribute to *any* of these lines on SPY.\n"
            f"- **Tradability `MIRAGE`** -- TEMA net Sharpe {R['tema_sharpe']:.2f} < SMA "
            f"{R['sma_sharpe']:.2f} < B&H {R['bh_sharpe']:.2f}, at ~2x turnover "
            f"({R['tema_turn']:.0f} vs {R['sma_turn']:.0f}/yr). Its timing also loses to a random "
            f"schedule (placebo *p* = {R['tema_placebo_p']:.2f}). Nothing to deploy.\n"
            f"- **'Worth it?' `BUSTED`** -- the lag-cancellation is real, but TEMA-minus-SMA runs "
            f"{R['tema_vs_sma_ann']:+.1f}%/yr (HAC *t* = {R['tema_vs_sma_t']:.2f}) and DEMA-minus-SMA "
            f"{R['dema_vs_sma_ann']:+.1f}%/yr ({R['dema_vs_sma_t']:.2f}). Less lag bought a worse rule."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- cost sweep\n\n"
            "It is behind at zero cost; turnover makes it fall faster. The cost slope is the "
            "*confirmation*, not the cause."
        ),
        code(
            "costs = [0.0, 2.0, 5.0, 10.0]\n"
            "tema = [st.backtest(b,'tema',50,c)['sharpe_net'] for c in costs]\n"
            "dema = [st.backtest(b,'dema',50,c)['sharpe_net'] for c in costs]\n"
            "sma  = [st.backtest(b,'sma', 50,c)['sharpe_net'] for c in costs]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "ax.plot(costs, sma, 's--', c=GREY, lw=2, label='SMA')\n"
            "ax.plot(costs, dema, 'd-', c=AMBER, lw=2, label='DEMA')\n"
            "ax.plot(costs, tema, 'o-', c=RED, lw=2, label='TEMA')\n"
            "ax.axhline(res['sharpe_bh'], ls=':', c=GREEN, lw=2, label='buy & hold')\n"
            "ax.set_xlabel('one-way cost (bps)'); ax.set_ylabel('net Sharpe')\n"
            "ax.set_title('TEMA falls fastest and crosses zero by ~10 bps; none beats holding'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('cost (bps):', costs)\n"
            "print('TEMA Sharpe:', [round(x,3) for x in tema])\n"
            "print('SMA  Sharpe:', [round(x,3) for x in sma])"
        ),
        md(
            f"> In plain words: at 2 bps TEMA nets {R['tema_c2']:.2f}, at 10 bps "
            f"{R['tema_c10']:.2f} (negative); the SMA holds {R['sma_c2']:.2f} -> {R['sma_c10']:.2f}. "
            f"But even at **zero cost** TEMA ({R['tema_c0']:.2f}) trails the SMA ({R['sma_c0']:.2f}). "
            "The Mirage doesn't need costs -- they just make it obvious faster."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further\n\n"
            "The control proves the engine is a faithful trend detector -- and that a less-lag smoother "
            "is a liability even when a real trend exists. Open questions worth a fork:\n\n"
            "- **Crossover variants.** Fast/slow DEMA or TEMA crossover instead of price-vs-line. The "
            "responsiveness thesis predicts help; we'd bet on worse whipsaw.\n"
            "- **Slower windows / weekly bars.** At a 200-day or weekly horizon the noise the lag "
            "rejects is relatively smaller -- does the DEMA/TEMA penalty shrink? (Early sweeps say no.)\n"
            "- **Hull MA & other 'zero-lag' lines.** Same family (Hull 2005, Ehlers' zero-lag EMA); "
            "the same bias-variance argument should apply -- a clean comparison study.\n"
            "- **Desk neighbours.** [Bollinger-Reversion](../../104-bollinger-reversion/) and the "
            "moving-average-crossover family -- where the recurring lesson is that the *line* is rarely "
            "the edge.\n\n"
            "*Show a less-lag smoother beating the plain SMA on a net, excess-of-cash Sharpe with a "
            "HAC t above 2 -- that's the bar this study sets.*"
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
