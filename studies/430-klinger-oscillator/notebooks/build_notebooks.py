"""Generate the two narrative notebooks for Study 430 (Klinger Oscillator).

    cd notebooks
    python build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        01_for_the_curious.ipynb 02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached SPY (and panel)
OHLCV under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
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


# Frozen real-tape headline numbers — mirror of docs/results.md.
# SPY daily OHLCV (yfinance, auto-adjusted), 2005-01-03 -> 2026-05-29 (21.4y, 5,385 days,
# as-of 2026-05-31, fingerprint 63d0db58b0a8). Klinger(34,55,13), long/flat, KVO>0 rule,
# 1bp one-way costs, 1-day execution lag, excess-of-cash.
R = dict(
    asof="2026-05-31", start="2005-01-03", end="2026-05-29", years=21.4, n_days=5385,
    fp="63d0db58b0a8",
    # headline race (zero mode, long/flat, 1bp): annualised net Sharpe (excess of cash)
    k_sharpe=0.382, hold_sharpe=0.645, sma_sharpe=0.776, macd_sharpe=0.548,
    k_cagr=0.037, hold_cagr=0.110, sma_cagr=0.085,
    k_mdd=-0.489, hold_mdd=-0.552, sma_mdd=-0.209,
    k_tim=0.53, k_turn=21.9,
    t_vs_hold=-2.87, t_vs_sma=-1.78, t_vs_macd=-0.94,
    placebo_p=0.669, placebo_median=0.444,
    # variants: (mode, long_short, sharpe, t_vs_hold)
    variants=[("KVO>0 long/flat", 0.382, -2.87), ("KVO>0 long/short", -0.203, -2.92),
              ("KVO>signal long/flat", 0.138, -4.09), ("KVO>signal long/short", -0.481, -4.14)],
    # cost sweep (zero, long/flat): (cost_bps, sharpe, t_vs_hold)
    costs=[(0.0, 0.402, -2.80), (1.0, 0.382, -2.87), (2.0, 0.363, -2.95), (5.0, 0.304, -3.18)],
    # panel (zero, long/flat, 1bp): (ticker, k_sharpe, hold_sharpe, t_vs_hold)
    panel=[("SPY", 0.382, 0.645, -2.87), ("QQQ", 0.652, 0.778, -2.65),
           ("IWM", 0.393, 0.470, -1.43), ("DIA", 0.394, 0.617, -2.48),
           ("EEM", 0.203, 0.400, -2.07), ("GLD", 0.317, 0.677, -2.96)],
    # synthetic control (zero, long/flat): (planted_edge, k_sharpe, hold_sharpe, t_vs_hold, placebo_p)
    syn=[(0.0, -0.633, -0.355, -0.51, 0.981), (3.0, 5.613, 0.439, 8.21, 0.000)],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Leads_price%3F: Busted](https://img.shields.io/badge/Leads_price%3F-Busted-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from klinger_oscillator import data, strategy as st

ASOF = "2026-05-31"
HAVE_REAL = data.have_real("SPY")
if HAVE_REAL:
    SPY = data.load_real("SPY")
    SPY = SPY[SPY.index <= ASOF]      # drop the partial current month (house rule)
else:
    SPY = None
print("real SPY cache present:", HAVE_REAL,
      "| days:", (0 if SPY is None else len(SPY)),
      "| last:", (None if SPY is None else SPY.index[-1].date()))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Does Klinger's volume oscillator *lead* the price? 📊\n"
            "### A 1970s \"volume leads price\" indicator, raced honestly against just buying and holding\n\n"
            + BADGES +
            "Here's a classic piece of charting lore. In the 1970s a trader named **Stephen Klinger** "
            "built an oscillator designed to read **volume** — the idea being that big money "
            "**accumulates** quietly (volume rises) *before* the price actually moves. Read the volume "
            "right, the story goes, and you get an **early warning**: go long when the oscillator turns "
            "positive (accumulation), step aside when it turns negative (distribution).\n\n"
            "It sounds compelling — volume *as a leading indicator*. So we did the obvious thing: we "
            "computed Klinger's oscillator on **SPY** (the S&P 500 ETF) over **21 years**, turned it into "
            "a simple **be-long / be-flat** timing rule, and asked one blunt question: **does it beat "
            "just buying and holding?** And while we were at it: does it beat a *dead-simple* 200-day "
            "moving average?\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebo test and the cost math? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ Every chart is drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does timing SPY with Klinger beat buy-and-hold? | **No.** Over 21 years its "
            f"risk-adjusted return (Sharpe **{R['k_sharpe']:.2f}**) is *worse* than simply holding "
            f"(**{R['hold_sharpe']:.2f}**) — and the gap is statistically real (it's reliably behind). |\n"
            f"| Does it beat a dumb 200-day moving average? | **No.** A one-line trend filter scores "
            f"Sharpe **{R['sma_sharpe']:.2f}** with **far** less trading. Klinger trades ~22× a year and "
            "still loses the race. |\n"
            f"| Is the *timing* any good at all? | **No.** A **random** schedule that's long the same "
            f"fraction of days beats Klinger's actual timing **{R['placebo_p']*100:.0f}%** of the time. "
            "When it's long isn't informative. |\n"
            "| So does volume \"lead price\" here? | **Not in any tradable way.** The oscillator reacts "
            "to volume that has *already* happened; it doesn't anticipate the next move. |\n\n"
            "> The legend is **busted** on the tape: Klinger's oscillator is a fine *description* of "
            "past volume, but as a *leading* timing signal on the index it adds nothing — and costs you "
            "the trading."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Volume precedes price. Klinger's oscillator measures the long-term force of "
            "**accumulation** versus **distribution** by weighting each day's volume by its trend and "
            "range. When the oscillator is **positive**, smart money is accumulating — ride it. When it "
            "turns **negative**, they're distributing — get out. Because it reads volume, it **leads** "
            "the price.\"*\n\n"
            "Stephen Klinger published the **Klinger Volume Oscillator** in *Technical Analysis of "
            "Stocks & Commodities* (1997). It's a staple in charting packages (TradingView, "
            "StockCharts). The selling point is exactly the *leading* property: catch the turn before "
            "the crowd. So the question is simple — **does the volume actually lead, or does the "
            "oscillator just lag the price like everything else?**"
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a volume oscillator truly **led** price, that would be a real crack in efficiency: you "
            "could see the move coming in the order flow before it showed up in the close. A reliable "
            "be-long/be-flat switch on the S&P that **beat buy-and-hold** would be worth a fortune — "
            "you'd capture the upside and dodge the crashes.\n\n"
            "But there's a trap that kills most timing indicators: **buying and holding is a very high "
            "bar.** The market drifts up most of the time, so any rule that sits in cash part of the "
            "year gives up that drift — and has to *more than* make it back by dodging the bad days. "
            "And it has to do that **net of the trading costs** it racks up flipping in and out. We'll "
            "hold Klinger to that bar."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"We take **{R['years']:.0f} years** of daily SPY ({R['start']} → {R['end']}), compute "
            "Klinger's oscillator, and make the simplest honest rule:\n\n"
            "1. **The rule.** Oscillator **above zero** → be **long** SPY. Below zero → be **flat** "
            "(in cash). Signal read at today's close, position taken **tomorrow** (no peeking).\n"
            "2. **The race.** Compare its risk-adjusted return (Sharpe) — *after* trading costs — to "
            "**buy-and-hold**, and to a **200-day moving-average** rule and **MACD** (the obvious "
            "simpler competitors).\n"
            "3. **The luck check.** Shuffle *when* the rule is long into a random schedule with the "
            "same time-in-market. If Klinger's *timing* is informative, the real rule should beat the "
            "random schedules. **If a coin does as well, the signal is noise.**\n\n"
            "What would make us say *mirage*? If Klinger doesn't clear buy-and-hold, gets beaten by the "
            "one-line moving average, and can't beat a random schedule — three strikes."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline race.** Growth of \\$1 (excess of cash) for buy-and-hold, the "
            "Klinger long/flat rule, and the dumb 200-day moving average. If volume \"leads price,\" the "
            "green Klinger line should pull ahead."
        ),
        code(
            "if HAVE_REAL:\n"
            "    kpos = st.klinger_position(SPY, mode='zero')\n"
            "    spos = st.sma_position(SPY)\n"
            "    kb = st.book_returns(SPY, kpos, cost_bps=1.0)\n"
            "    sb = st.book_returns(SPY, spos, cost_bps=1.0)\n"
            "    eq_hold = (1+kb['asset_excess']).cumprod()\n"
            "    eq_k = (1+kb['strat_excess']).cumprod()\n"
            "    eq_s = (1+sb['strat_excess']).cumprod()\n"
            "    idx = SPY.index\n"
            "else:\n"
            "    idx = pd.bdate_range('2005-01-03', periods=R['n_days'])\n"
            "    eq_hold = pd.Series(np.linspace(1, np.exp(R['hold_cagr']*R['years']), R['n_days']), index=idx)\n"
            "    eq_k = pd.Series(np.linspace(1, np.exp(R['k_cagr']*R['years']), R['n_days']), index=idx)\n"
            "    eq_s = pd.Series(np.linspace(1, np.exp(R['sma_cagr']*R['years']), R['n_days']), index=idx)\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.6))\n"
            "ax.plot(idx, eq_hold, c=GREY, lw=2, label=f'buy & hold (Sharpe {R[\"hold_sharpe\"]:.2f})')\n"
            "ax.plot(idx, eq_k, c=RED, lw=2, label=f'Klinger long/flat (Sharpe {R[\"k_sharpe\"]:.2f})')\n"
            "ax.plot(idx, eq_s, c=GREEN, lw=1.6, ls='--', label=f'dumb 200d SMA (Sharpe {R[\"sma_sharpe\"]:.2f})')\n"
            "ax.set_yscale('log'); ax.set_ylabel('growth of $1 (excess of cash, log)')\n"
            "ax.set_title('Klinger times SPY worse than buying & holding — and worse than a 200d average')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Klinger Sharpe {R[\"k_sharpe\"]:.2f}  vs  buy-hold {R[\"hold_sharpe\"]:.2f}  vs  200d SMA {R[\"sma_sharpe\"]:.2f}')"
        ),
        md(
            f"The red Klinger line **lags** the grey buy-and-hold line, and the dashed green "
            f"200-day-average line — about as dumb a trend rule as exists — **beats both**. Klinger's "
            f"Sharpe is **{R['k_sharpe']:.2f}** against buy-and-hold's **{R['hold_sharpe']:.2f}**. The "
            "\"leading\" indicator is *behind*."
        ),
        md(
            "**Is the timing informative at all?** Here's the honest luck test. Klinger is long ~53% of "
            "days. We shuffle *when* it's long into thousands of **random** schedules with the same "
            "time-in-market, and plot the Sharpe each random schedule earns. If Klinger's timing means "
            "anything, the real line should sit in the **right tail**."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.block_permutation_p(kb, n_draws=3000, seed=430)\n"
            "    draws = pl['draws']; obs = pl['obs_sharpe']; pval = pl['p_value']\n"
            "else:\n"
            "    rng = np.random.default_rng(430); draws = rng.normal(R['placebo_median'], 0.18, 3000)\n"
            "    obs = R['k_sharpe']; pval = R['placebo_p']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='3,000 random long/flat schedules')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'Klinger actual timing ({obs:.2f})')\n"
            "ax.set_xlabel('net Sharpe (excess of cash)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'A coin beats Klinger {pval*100:.0f}% of the time — the timing carries no information')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Klinger {obs:.2f}  |  random-schedule median {np.median(draws):.2f}  |  p(random >= Klinger) = {pval:.3f}')"
        ),
        md(
            f"The red line sits **left of the middle** of the random cloud: a random schedule beats "
            f"Klinger's actual timing **{R['placebo_p']*100:.0f}%** of the time, and the *typical* coin "
            f"(median Sharpe **{R['placebo_median']:.2f}**) does **better** than Klinger "
            f"(**{R['k_sharpe']:.2f}**). Knowing *when* the oscillator is positive tells you nothing "
            "useful about the next day."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Klinger's long/flat rule on SPY scores Sharpe **{R['k_sharpe']:.2f}**, "
            f"*below* buy-and-hold (**{R['hold_sharpe']:.2f}**) and below a one-line 200-day average "
            f"(**{R['sma_sharpe']:.2f}**). The difference vs buy-and-hold is significant — significantly "
            "**worse** (*t* ≈ −2.9). A random schedule beats its timing two-thirds of the time.\n"
            "- **Tradability — Mirage.** It loses the race *and* trades ~22× a year; costs only widen "
            "the gap. There is no edge to deploy.\n"
            "- **\"Volume leads price\"? — Busted.** The oscillator reacts to volume that already "
            "printed; on the index it does not anticipate the move."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you actually trade it? — costs only make it worse\n\n"
            "There's nothing to rescue with cheaper execution: Klinger is already *behind* before costs. "
            "Here's its Sharpe as we charge more per trade — it just slides further from the "
            "buy-and-hold line it never reached."
        ),
        code(
            "cb = [c[0] for c in R['costs']]\n"
            "if HAVE_REAL:\n"
            "    shp = [st.run_experiment(SPY, mode='zero', cost_bps=c, with_placebo=False)['klinger']['sharpe'] for c in cb]\n"
            "else:\n"
            "    shp = [c[1] for c in R['costs']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(cb, shp, 'o-', c=RED, lw=2, label='Klinger net Sharpe')\n"
            "ax.axhline(R['hold_sharpe'], c=GREY, ls='--', lw=2, label=f'buy & hold ({R[\"hold_sharpe\"]:.2f})')\n"
            "for x,y in zip(cb,shp): ax.annotate(f'{y:.2f}',(x,y),ha='center',va='bottom',fontsize=9)\n"
            "ax.set_xlabel('one-way cost (bps)'); ax.set_ylabel('net Sharpe (excess of cash)')\n"
            "ax.set_title('Klinger never reaches the buy-and-hold line, at any cost level')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('cost sweep (bps -> Sharpe):', {c:round(s,3) for c,s in zip(cb,shp)})"
        ),
        md(
            "Even at **zero cost** Klinger lands at Sharpe ~0.40 — still under buy-and-hold's 0.65. The "
            "problem isn't friction; it's that the signal doesn't add information. Costs are just salt "
            "in the wound."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Other parameters.** We used Klinger's standard (34, 55, 13). The quants notebook shows "
            "the *signal-line* variant and the long/short variant — both are **worse**, not better.\n"
            "- **Other markets.** We ran a 6-ETF panel (SPY/QQQ/IWM/DIA/EEM/GLD). Klinger loses to "
            "buy-and-hold on **every one**.\n"
            "- **The honest lesson.** \"Volume leads price\" is a seductive story, but a volume "
            "oscillator is still a **moving average of the past**. Lagging averages can't lead. If you "
            "think a different Klinger parameter set beats buy-and-hold net of costs on SPY — fork it "
            "and show the green line on top."
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
            "# The Klinger Volume Oscillator — a quantitative teardown 🔬\n"
            "### KVO(34,55,13) long/flat & long/short on SPY · net Sharpe vs buy-and-hold "
            "(excess-vs-excess) · HAC *t* on the daily difference · a block-permutation placebo · "
            "head-to-head vs SMA & MACD · a synthetic volume-leads-price control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). Klinger's "
            "oscillator is sold as a **leading** signal because it reads volume. We test that literally: "
            "turn it into a daily timing rule, race it **net of costs, excess-of-cash** against "
            "buy-and-hold and the obvious simpler trend rules, and ask whether *when* it is long carries "
            "any information at all.\n\n"
            "> ⚠️ **Data note.** SPY daily OHLCV (yfinance, auto-adjusted = **total-return** proxy), "
            f"{R['start']}→{R['end']}, **{R['years']:.0f} years**, as-of **{R['asof']}** (the partial "
            "current month is dropped). Offline core + synthetic control are deterministic. Methods in "
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
            f"| **Signal** | `NONE` | Klinger long/flat net Sharpe **{R['k_sharpe']:.2f}** vs buy-and-hold "
            f"**{R['hold_sharpe']:.2f}**; HAC *t* of the daily difference = **{R['t_vs_hold']:.2f}** "
            "(significantly *worse*, not better). |\n"
            f"| **Tradability** | `MIRAGE` | Loses the race at every cost level (~{R['k_turn']:.0f} "
            "round-trips/yr), and beaten by a one-line 200d SMA "
            f"(**{R['sma_sharpe']:.2f}**, *t* = {R['t_vs_sma']:.2f}). Nothing to deploy. |\n"
            f"| **Volume leads price?** | `BUSTED` | A block-shuffled random schedule with the same "
            f"time-in-market beats Klinger's timing **p = {R['placebo_p']:.2f}** of the time (median "
            f"random Sharpe **{R['placebo_median']:.2f}** > Klinger's **{R['k_sharpe']:.2f}**). |\n\n"
            "> 💡 In plain words: a *leading* indicator should beat buy-and-hold, or at least beat a "
            "coin. Klinger does neither — it is a lagging moving-average of volume wearing a "
            "leading-indicator label."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Klinger's **Volume Force** weights each day's volume by its range position and trend:\n\n"
            "$$\\mathrm{VF}_t = V_t \\cdot \\left|\\,2\\frac{dm_t}{cm_t} - 1\\,\\right| \\cdot T_t "
            "\\cdot 100,\\qquad T_t=\\mathrm{sign}(TP_t - TP_{t-1}),\\; dm_t = H_t-L_t,$$\n\n"
            "with $cm_t$ a running sum of $dm$ within a trend run, and $TP_t = H_t+L_t+C_t$. The "
            "oscillator and trigger are\n\n"
            "$$\\mathrm{KVO}_t = \\mathrm{EMA}_{34}(\\mathrm{VF})_t - \\mathrm{EMA}_{55}(\\mathrm{VF})_t,"
            "\\qquad \\mathrm{sig}_t = \\mathrm{EMA}_{13}(\\mathrm{KVO})_t.$$\n\n"
            "Let $x_t \\in \\{0,1\\}$ (long/flat) be $\\mathbf 1[\\mathrm{KVO}_t>0]$ and $r_{t+1}$ the "
            "next-day excess return. The believer's hypotheses:\n\n"
            "- **H₁ (it times).** The booked series $x_t\\, r_{t+1} - \\text{costs}$ has a higher Sharpe "
            "than buy-and-hold $r_{t+1}$.\n"
            "- **H₂ (it's better than simple).** It beats a 200-day SMA filter and MACD.\n"
            "- **H₃ (it leads).** The *timing* $x_t$ beats a random schedule of equal exposure.\n\n"
            "We find **H₁ rejected** (Sharpe below buy-and-hold, HAC *t* of the difference < 0), **H₂ "
            "rejected** (loses to a one-line SMA), **H₃ rejected** (a coin wins two-thirds of the time)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the race must be apples-to-apples\n\n"
            "Two honesty rules govern a timing-rule test. **(a) Excess-vs-excess.** A long/flat rule "
            "sits in cash part-time; comparing its *raw* Sharpe to a fully-invested benchmark's "
            "*excess* Sharpe manufactures a verdict. We book **everything excess of the risk-free "
            "rate**, so a flat day earns 0 excess (it is the cash leg), and race excess-Sharpe to "
            "excess-Sharpe. **(b) One execution lag, once.** The position known at the close of $t$ "
            "earns $r_{t+1}$ — a single `shift(1)`. Costs are **one-way × |Δposition| (NAV turnover)**, "
            "and any short leg pays borrow. The decisive statistic is the **HAC (Newey-West) *t*** of "
            "the per-day difference $d_t = (x_{t-1} r_t - \\text{cost}_t) - r_t$ against zero — "
            "autocorrelation-robust because daily strategy returns are serially dependent."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Universe.** SPY daily OHLCV (yfinance, auto-adjusted), {R['start']}→{R['end']}, "
            f"**{R['n_days']:,}** sessions / {R['years']:.0f}y; 6-ETF panel for robustness.\n"
            "- **Indicator.** KVO(34, 55, 13), Klinger's standard spans.\n"
            "- **Rules.** Headline = **KVO > 0 long/flat**; variants = KVO > signal, and long/short.\n"
            "- **Timing.** Signal at close $t$; position over $t+1$ (one shift). Costs **1 bp one-way × "
            "NAV turnover**; borrow 50 bps/yr on short legs.\n"
            "- **Race.** Net annualised Sharpe (excess-of-cash) vs buy-and-hold, **200d SMA**, **MACD**.\n"
            "- **Inference #1 (HAC t).** Newey-West *t* of the daily (Klinger − buy-hold) difference.\n"
            "- **Inference #2 (block-permutation placebo).** 3,000 block-shuffles of the held-position "
            "series (21-day blocks preserve persistence & time-in-market); "
            "$p = \\Pr[\\text{random schedule Sharpe} \\ge \\text{observed}]$.\n"
            "- **Positive control.** A synthetic panel where volume *genuinely* leads price by a day "
            "(knob `edge`): `edge=0` must NOT beat buy-and-hold; a large `edge` must light up."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline race — equity curves & the Sharpe table\n\n"
            "Growth of \\$1 (excess of cash) for buy-and-hold, Klinger long/flat, and a 200d SMA filter."
        ),
        code(
            "if HAVE_REAL:\n"
            "    kpos = st.klinger_position(SPY, mode='zero'); spos = st.sma_position(SPY); mpos = st.macd_position(SPY)\n"
            "    kb = st.book_returns(SPY, kpos, cost_bps=1.0); sb = st.book_returns(SPY, spos, cost_bps=1.0); mb = st.book_returns(SPY, mpos, cost_bps=1.0)\n"
            "    eq_hold=(1+kb['asset_excess']).cumprod(); eq_k=(1+kb['strat_excess']).cumprod(); eq_s=(1+sb['strat_excess']).cumprod()\n"
            "    ks=[st.sharpe(kb['strat_excess'].values), st.sharpe(kb['asset_excess'].values), st.sharpe(sb['strat_excess'].values), st.sharpe(mb['strat_excess'].values)]\n"
            "    idx=SPY.index\n"
            "else:\n"
            "    idx=pd.bdate_range('2005-01-03', periods=R['n_days'])\n"
            "    eq_hold=pd.Series(np.linspace(1,np.exp(R['hold_cagr']*R['years']),R['n_days']),index=idx)\n"
            "    eq_k=pd.Series(np.linspace(1,np.exp(R['k_cagr']*R['years']),R['n_days']),index=idx)\n"
            "    eq_s=pd.Series(np.linspace(1,np.exp(R['sma_cagr']*R['years']),R['n_days']),index=idx)\n"
            "    ks=[R['k_sharpe'],R['hold_sharpe'],R['sma_sharpe'],R['macd_sharpe']]\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.6))\n"
            "ax.plot(idx, eq_hold, c=GREY, lw=2, label=f'buy & hold ({ks[1]:.2f})')\n"
            "ax.plot(idx, eq_k, c=RED, lw=2, label=f'Klinger long/flat ({ks[0]:.2f})')\n"
            "ax.plot(idx, eq_s, c=GREEN, lw=1.6, ls='--', label=f'200d SMA ({ks[2]:.2f})')\n"
            "ax.set_yscale('log'); ax.set_ylabel('growth of $1 (excess, log)')\n"
            "ax.set_title('Net Sharpe race (excess-of-cash): Klinger trails everything'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Sharpe  Klinger={ks[0]:.3f}  buy-hold={ks[1]:.3f}  200dSMA={ks[2]:.3f}  MACD={ks[3]:.3f}')"
        ),
        md(
            f"> 💡 In plain words: every alternative beats Klinger. Klinger **{R['k_sharpe']:.2f}** < MACD "
            f"**{R['macd_sharpe']:.2f}** < buy-hold **{R['hold_sharpe']:.2f}** < the dumbest rule of all, "
            f"a 200-day average **{R['sma_sharpe']:.2f}**. The fancier the volume math, the worse it did."
        ),
        md(
            "### 4b · The decisive test — HAC *t* of the daily difference\n\n"
            "Is Klinger's underperformance vs buy-and-hold *significant*? The Newey-West *t* of the "
            "daily (Klinger − buy-hold) excess-return difference. A *t* below −2 means it is reliably "
            "**worse**, not merely unlucky."
        ),
        code(
            "if HAVE_REAL:\n"
            "    d_hold = kb['strat_excess'].values - kb['asset_excess'].values\n"
            "    d_sma  = kb['strat_excess'].values - sb['strat_excess'].values\n"
            "    d_macd = kb['strat_excess'].values - mb['strat_excess'].values\n"
            "    tt = [st.hac_t(d_hold), st.hac_t(d_sma), st.hac_t(d_macd)]\n"
            "else:\n"
            "    tt = [R['t_vs_hold'], R['t_vs_sma'], R['t_vs_macd']]\n"
            "labels = ['vs buy-hold','vs 200d SMA','vs MACD']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "cols = [RED if t<0 else GREEN for t in tt]\n"
            "ax.bar(labels, tt, color=cols, width=.55)\n"
            "ax.axhline(-2, ls='--', c='k', label='t = -2 (reliably worse)')\n"
            "ax.axhline(2, ls='--', c=GREEN)\n"
            "for i,t in enumerate(tt): ax.annotate(f'{t:.2f}',(i,t),ha='center',va='top' if t<0 else 'bottom')\n"
            "ax.set_ylabel('HAC t of daily return difference'); ax.set_title('Klinger is significantly WORSE than buy-and-hold'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('HAC t  vs buy-hold:', round(tt[0],2), ' vs SMA:', round(tt[1],2), ' vs MACD:', round(tt[2],2))"
        ),
        md(
            f"> 💡 In plain words: the (Klinger − buy-hold) difference has HAC *t* = "
            f"**{R['t_vs_hold']:.2f}** — past the −2 bar, so the underperformance is *real*, not noise. "
            "The desk's `t ≥ 2` bar for a **positive** signal is nowhere in sight; if anything the rule "
            "clears the bar on the **wrong side**."
        ),
        md(
            "### 4c · The luck test — does a coin beat Klinger's timing?\n\n"
            "Block-shuffle the held-position series (21-day blocks keep the persistence and the "
            "time-in-market) into 3,000 random schedules; each one is re-booked net of the same costs. "
            "Where does Klinger's actual Sharpe fall?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.block_permutation_p(kb, n_draws=3000, seed=430)\n"
            "    draws=pl['draws']; obs=pl['obs_sharpe']; pval=pl['p_value']\n"
            "else:\n"
            "    rng=np.random.default_rng(430); draws=rng.normal(R['placebo_median'],0.18,3000); obs=R['k_sharpe']; pval=R['placebo_p']\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "ax.hist(draws, bins=50, color=GREY, alpha=.85, label='3,000 block-shuffled schedules')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'Klinger actual ({obs:.2f})')\n"
            "ax.axvline(np.median(draws), c=GREEN, lw=1.6, ls='--', label=f'random median ({np.median(draws):.2f})')\n"
            "ax.set_xlabel('net Sharpe (excess of cash)'); ax.set_ylabel('frequency')\n"
            "ax.set_title(f'A coin beats Klinger p = {pval:.2f} of the time — timing carries no information'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'obs {obs:.3f}  random-median {np.median(draws):.3f}  p(random>=obs) = {pval:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the placebo *p* = **{R['placebo_p']:.2f}** — the random cloud's "
            f"**median** ({R['placebo_median']:.2f}) is *above* Klinger ({R['k_sharpe']:.2f}). When the "
            "oscillator chooses to be long is no better than (slightly worse than) flipping a coin with "
            "the same time-in-market. This is the cleanest statement of \"no leading information.\""
        ),
        md(
            "### 4d · Variants & costs — nothing rescues it\n\n"
            "Left: the four rule variants (signal-line crossing, long/short) — all *worse*. Right: the "
            "cost sweep — Klinger never reaches the buy-and-hold line even at zero cost."
        ),
        code(
            "if HAVE_REAL:\n"
            "    var = []\n"
            "    for mode, ls, name in [('zero',False,'KVO>0 L/F'),('zero',True,'KVO>0 L/S'),('signal',False,'KVO>sig L/F'),('signal',True,'KVO>sig L/S')]:\n"
            "        rr = st.run_experiment(SPY, mode=mode, long_short=ls, cost_bps=1.0, with_placebo=False)\n"
            "        var.append((name, rr['klinger']['sharpe']))\n"
            "    cb=[c[0] for c in R['costs']]; shp=[st.run_experiment(SPY, mode='zero', cost_bps=c, with_placebo=False)['klinger']['sharpe'] for c in cb]\n"
            "else:\n"
            "    var=[(v[0], v[1]) for v in R['variants']]; cb=[c[0] for c in R['costs']]; shp=[c[1] for c in R['costs']]\n"
            "fig,(a1,a2)=plt.subplots(1,2,figsize=(11.0,4.3))\n"
            "names=[v[0] for v in var]; sv=[v[1] for v in var]\n"
            "a1.barh(names, sv, color=[GREEN if s>R['hold_sharpe'] else RED for s in sv])\n"
            "a1.axvline(R['hold_sharpe'], c=GREY, ls='--', label=f'buy-hold {R[\"hold_sharpe\"]:.2f}')\n"
            "for i,s in enumerate(sv): a1.annotate(f'{s:.2f}',(s,i),va='center',ha='left' if s>=0 else 'right',fontsize=9)\n"
            "a1.set_xlabel('net Sharpe'); a1.set_title('Every Klinger variant < buy-and-hold'); a1.legend()\n"
            "a2.plot(cb, shp, 'o-', c=RED, lw=2, label='Klinger')\n"
            "a2.axhline(R['hold_sharpe'], c=GREY, ls='--', label=f'buy-hold {R[\"hold_sharpe\"]:.2f}')\n"
            "for x,y in zip(cb,shp): a2.annotate(f'{y:.2f}',(x,y),ha='center',va='bottom',fontsize=9)\n"
            "a2.set_xlabel('one-way cost (bps)'); a2.set_ylabel('net Sharpe'); a2.set_title('Behind even at zero cost'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('variants:', [(n,round(s,3)) for n,s in var]); print('cost sweep:', {c:round(s,3) for c,s in zip(cb,shp)})"
        ),
        md(
            f"> 💡 In plain words: the *signal-line* version (**{R['variants'][2][1]:.2f}**) is worse than "
            f"the zero-cross (**{R['variants'][0][1]:.2f}**), and both long/short versions go **negative** "
            f"(**{R['variants'][1][1]:.2f}**, **{R['variants'][3][1]:.2f}**) — shorting on a negative KVO "
            "is actively harmful. And at **zero cost** Klinger is still ~0.40, below buy-hold's 0.65. The "
            "killer is the signal, not the friction."
        ),
        md(
            "### 4e · The panel — it loses everywhere\n\n"
            "Klinger long/flat vs buy-and-hold across a 6-ETF panel. Bars below the dashed line lose to "
            "buying and holding."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pan = []\n"
            "    for t,_,_,_ in R['panel']:\n"
            "        if data.have_real(t):\n"
            "            bb = data.load_real(t); bb=bb[bb.index<=ASOF]\n"
            "            rr = st.run_experiment(bb, mode='zero', cost_bps=1.0, with_placebo=False)\n"
            "            pan.append((t, rr['klinger']['sharpe'], rr['asset_sharpe']))\n"
            "        else:\n"
            "            row=[p for p in R['panel'] if p[0]==t][0]; pan.append((t,row[1],row[2]))\n"
            "else:\n"
            "    pan=[(p[0],p[1],p[2]) for p in R['panel']]\n"
            "x=np.arange(len(pan)); kk=[p[1] for p in pan]; hh=[p[2] for p in pan]\n"
            "fig, ax = plt.subplots(figsize=(9.6, 4.3))\n"
            "ax.bar(x-.2,[p[2] for p in pan],.4,color=GREY,label='buy & hold')\n"
            "ax.bar(x+.2,[p[1] for p in pan],.4,color=RED,label='Klinger long/flat')\n"
            "ax.set_xticks(x); ax.set_xticklabels([p[0] for p in pan]); ax.set_ylabel('net Sharpe')\n"
            "ax.set_title('Klinger loses to buy-and-hold on every ETF in the panel'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('panel (ticker, Klinger, buy-hold):', [(p[0],round(p[1],2),round(p[2],2)) for p in pan])"
        ),
        md(
            "> 💡 In plain words: SPY, QQQ, IWM, DIA, EEM, GLD — Klinger is **below** buy-and-hold on all "
            "six. A real leading signal would win *somewhere*; this loses everywhere, which is what a "
            "non-signal looks like."
        ),
        md(
            "### 4f · Faithful-engine & power control — we know the truth here\n\n"
            "On a deterministic synthetic panel where volume **genuinely leads price** by a day (knob "
            "`edge`): with `edge=0` the Klinger rule must NOT beat buy-and-hold (no false positive); "
            "with a large planted `edge` it must light up. Both hold — so the dead real-tape result is a "
            "true negative, not a broken harness."
        ),
        code(
            "res = []\n"
            "for edge in (0.0, 3.0):\n"
            "    bars, truth = data.synthetic_panel(edge=edge, seed=430)\n"
            "    rr = st.run_experiment(bars, mode='zero', cost_bps=1.0, with_placebo=True, n_draws=2000)\n"
            "    res.append((edge, rr['klinger']['sharpe'], rr['asset_sharpe'], rr['t_vs_hold'], rr['placebo_p']))\n"
            "labels=[f'edge={e:.0f}\\n(volume leads)' if e>0 else f'edge={e:.0f}\\n(no info)' for e,_,_,_,_ in res]\n"
            "tv=[r[3] for r in res]\n"
            "fig, ax = plt.subplots(figsize=(8.6,4.3))\n"
            "ax.bar(labels, tv, color=[GREY, GREEN], width=.5)\n"
            "ax.axhline(2, ls='--', c=GREEN, label='t = +2 bar')\n"
            "for i,t in enumerate(tv): ax.annotate(f't={t:.2f}',(i,t),ha='center',va='bottom')\n"
            "ax.set_ylabel('HAC t of (Klinger - buy-hold)'); ax.set_title('Control: no-info -> t~0; planted lead -> lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for e,k,h,t,p in res: print(f'edge={e:.1f}: Klinger={k:+.2f} hold={h:+.2f} t_vs_hold={t:+.2f} placebo_p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: with **no** planted lead the rule sits at *t* = "
            f"**{R['syn'][0][3]:.2f}** vs buy-hold (placebo *p* = {R['syn'][0][4]:.2f} — no false "
            f"positive); with a **strong** planted lead it explodes to *t* = **{R['syn'][1][3]:.2f}** "
            f"(placebo *p* = {R['syn'][1][4]:.2f}). The engine *can* detect volume-leads-price when it's "
            "really there — and on the real SPY tape, it isn't."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — Klinger long/flat net Sharpe **{R['k_sharpe']:.2f}** vs buy-and-hold "
            f"**{R['hold_sharpe']:.2f}**; HAC *t* of the daily difference = **{R['t_vs_hold']:.2f}** "
            "(reliably *worse*). No variant clears buy-and-hold; the long/short legs go negative. "
            "Nowhere near the desk's *t ≥ 2* bar for a positive edge.\n"
            f"- **Tradability `MIRAGE`** — loses at every cost level (~{R['k_turn']:.0f} round-trips/yr) "
            f"and is beaten by a one-line 200d SMA (**{R['sma_sharpe']:.2f}**). There is no residual edge "
            "to deploy.\n"
            f"- **Volume leads price? `BUSTED`** — a block-shuffled random schedule with the same "
            f"time-in-market beats Klinger's timing **p = {R['placebo_p']:.2f}** of the time. The "
            "oscillator is a lagging moving-average of volume; it does not anticipate the index."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — no, and here's the one picture\n\n"
            "The drawdown story is the only place Klinger looks defensible — sitting in cash trims the "
            "worst losses. But it buys that with so much forgone upside that the **risk-adjusted** "
            "return still loses. Net Sharpe vs max drawdown for the three rules:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pts = [('buy & hold', R['hold_sharpe'], st.max_drawdown(kb['asset_excess'].values), GREY),\n"
            "           ('Klinger', st.sharpe(kb['strat_excess'].values), st.max_drawdown(kb['strat_excess'].values), RED),\n"
            "           ('200d SMA', st.sharpe(sb['strat_excess'].values), st.max_drawdown(sb['strat_excess'].values), GREEN)]\n"
            "else:\n"
            "    pts = [('buy & hold', R['hold_sharpe'], R['hold_mdd'], GREY),\n"
            "           ('Klinger', R['k_sharpe'], R['k_mdd'], RED),\n"
            "           ('200d SMA', R['sma_sharpe'], R['sma_mdd'], GREEN)]\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.6))\n"
            "for name, s, dd, c in pts:\n"
            "    ax.scatter(abs(dd)*100, s, s=180, color=c, zorder=3)\n"
            "    ax.annotate(f'  {name}\\n  Sharpe {s:.2f}', (abs(dd)*100, s), fontsize=9, va='center')\n"
            "ax.set_xlabel('max drawdown (%, lower = safer)'); ax.set_ylabel('net Sharpe (excess of cash)')\n"
            "ax.set_title('Klinger trims drawdown vs buy-hold but gives up too much return; SMA dominates it')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('(Sharpe, maxDD):', [(n, round(s,2), round(dd,2)) for n,s,dd,_ in pts])"
        ),
        md(
            f"> 💡 In plain words: yes, Klinger's drawdown (**{R['k_mdd']*100:.0f}%**) beats buy-and-hold's "
            f"(**{R['hold_mdd']*100:.0f}%**) — but the 200d SMA gets a far better drawdown "
            f"(**{R['sma_mdd']*100:.0f}%**) *and* a higher Sharpe, for a fraction of the trading. On every "
            "axis a trader cares about, Klinger is dominated. **MIRAGE.**"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Parameter search is a trap.** You could grid-search KVO spans until something beats "
            "buy-and-hold in-sample — but that's curve-fitting (see the desk's data-mining demos). The "
            "*standard* (34,55,13) is the honest test, and it fails.\n"
            "- **Volume as a feature, not a filter.** Volume may still help *conditionally* (e.g. "
            "confirming breakouts), but as a standalone leading timing switch on the index it adds "
            "nothing here.\n"
            "- **Why lagging averages can't lead.** KVO is a difference of EMAs of a volume series — a "
            "low-pass filter. Filters delay; they don't anticipate. The \"leads price\" claim is a "
            "category error.\n\n"
            "*The reproducible core is offline and deterministic. Methods and sources: "
            "[`docs/references.md`](../docs/references.md); frozen numbers: "
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
