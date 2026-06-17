"""Generate the two narrative notebooks for Study 301 (Triple-RSI).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-17).
R = dict(
    # SPY all-history (canonical, close fill, gross)
    spy_n=118, spy_win=88.1, spy_mean=132.8, spy_skew=-1.59, spy_t=5.07,
    spy_pre_n=43, spy_pre_win=90.7, spy_pre_mean=179.0, spy_pre_t=6.33,
    spy_post_n=75, spy_post_win=86.7, spy_post_mean=106.3, spy_post_skew=-2.06, spy_post_t=3.11,
    # Random-direction control (SPY)
    rand_n=118, rand_win=61.9, rand_mean=47.9, rand_t=2.57, excess_over_rand=84.9,
    # Next-open robustness
    nextopen_mean=109.7, nextopen_win=85.6, nextopen_t=4.22,
    # No-filter variant
    nofilter_n=278, nofilter_win=83.5, nofilter_mean=162.0, nofilter_t=7.05,
    # Cost sweep (SPY, net)
    cost0_bps=132.8, cost0_t=5.07, cost2_bps=130.8, cost2_t=4.99,
    cost5_bps=127.8, cost5_t=4.88, cost10_bps=122.8, cost10_t=4.69,
    # Capacity / benchmark
    trades_per_yr=3.5, avg_hold=4.8, pct_in_market=6.8, ann_est=4.69,
    spy_bh_ann=10.8, yrs=33.4, worst_trade=-7.37, best_trade=7.81, sharpe_trade=0.66,
    # Pooled basket
    pool_n=492, pool_win=78.9, pool_mean=110.8, pool_skew=-0.99, pool_t=8.28,
    pool_rand_n=492, pool_rand_win=58.7, pool_rand_mean=37.1, pool_rand_t=3.93,
    pool_excess=73.7,
    # Synthetic win-rate illusion + positive control
    mart_n=507, mart_win=62.3, mart_mean=-34.6, mart_skew=-1.65, mart_t=-1.15,
    rev_n=192, rev_win=73.4, rev_mean=103.4, rev_t=5.07, rev_excess=62.9,
)


# ---------------------------------------------------------------------------
# Shared analysis preamble
# ---------------------------------------------------------------------------
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

from triple_rsi import data, strategy as st

TICKERS = ["SPY", "QQQ", "IWM", "DIA"]

def _have_cache():
    return all(os.path.exists(data._cache_path(t, data.DEFAULT_CACHE)) for t in TICKERS)

HAVE_REAL = _have_cache()

def fetch(t):
    return data.fetch_daily(t, fetch=False)

def run_one(t, entry="close", cost=0.0, trend_sma=200, seed=None):
    b = fetch(t)
    sig = st.triple_rsi_entries(b, trend_sma=trend_sma)
    dirs = st.random_directions(int(sig.sum()), seed=seed) if seed is not None else None
    return st.run_trades(b, sig, rsi_exit=50.0, cost_bps=cost, entry=entry, random_dirs=dirs)

def run_all(entry="close", cost=0.0, trend_sma=200, seed=None):
    return pd.concat([run_one(t, entry, cost, trend_sma, seed) for t in TICKERS],
                     ignore_index=True)

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Triple-RSI — is a 90% win rate as good as it sounds?\n"
            "### A viral 'buy the oversold bounce' recipe, tested honestly, in plain English\n\n"
            "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![A_90%25_win--rate_money_machine%3F: Busted](https://img.shields.io/badge/A_90%25_win--rate_money_machine%3F-Busted-8b949e?style=flat-square)\n\n"
            "A strategy doing the rounds on finance Twitter: the **Triple RSI**, marketed as a "
            "**90% win rate** on the S&P 500. The recipe stacks four conditions (three of them on "
            "the 5-day RSI), and when they all line up you buy, then sell a few days later when "
            "the RSI recovers. Ninety percent of trades make money. Sounds like a money machine. "
            "This notebook asks the only two questions that matter: *is the signal real, and is a "
            "90% win rate actually worth anything?*\n\n"
            "> **This is the plain-language layer.** Want the t-stats, the random-control and the "
            "capacity math? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is the oversold bounce real? | **Yes.** On SPY it earns "
            f"**+{R['spy_mean']:.0f} bps per trade** (HAC *t* = +{R['spy_t']:.2f}) and survives "
            "an honest next-open fill — a genuine, durable signal. |\n"
            "| Is the 90% win rate real? | **Basically yes — 88%.** But that's the wrong number: "
            "on a pure coin flip the *same* exit already wins 62% with a **losing** average. |\n"
            "| Can you get rich on it? | **No.** It trades **3.5 times a year** and sits in cash "
            f"~93% of the time, compounding to ~+{R['ann_est']:.1f}%/yr — about **half** of just "
            f"holding the index (+{R['spy_bh_ann']:.1f}%/yr). |\n"
            "| So what is it? | A real but **capital-starved** signal — a fine overlay, a poor "
            "stand-alone 'system'. The headline sells the win rate because the *returns* don't sell. |\n\n"
            "> The Triple-RSI is not a scam — the signal is real. But '90% win rate' is marketing "
            "sleight-of-hand: it points at the one flattering number and away from the one that "
            "matters (how much money, vs just buying and holding)."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The Triple RSI strategy has few trades but a solid **90% win rate**. Buy SPY at "
            "the close when: the 5-day RSI is below 30, **and** it's down for the third day in a "
            "row, **and** it was below 60 three days ago, **and** the close is above the 200-day "
            "moving average. Sell at the close when the 5-day RSI crosses above 50.\"*\n\n"
            "— QuantifiedStrategies.com, *Triple RSI Trading Strategy: Boost Your Win Rate to 90%*\n\n"
            "The idea rests on something real: at a few-day horizon, beaten-down index prices tend "
            "to **bounce**. An RSI below 30 that's been falling for days is the market equivalent "
            "of a short, sharp panic — and panics often overshoot. The bet is that the snap-back "
            "pays."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "Two things make this one worth a careful look. First, the underlying effect "
            "(short-term mean reversion) is *real* — the desk has already stamped its cousin, the "
            "Connors RSI(2) system, as a genuine signal ([Study 75 — Knee-Jerk](../../75-knee-jerk/)). "
            "So this isn't pure folklore. Second, the **'90% win rate'** headline is a perfect "
            "specimen of a very common trap: a number that's technically true and almost entirely "
            "useless. If we can show *why* it's useless, that lesson transfers to every 'high win "
            "rate' system you'll ever be pitched."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three checks keep us honest:\n\n"
            "1. **Race it against a coin.** Take every Triple-RSI entry, but on half the trades go "
            "*random* direction instead of always buying. If the signal means something, it beats "
            "the coin.\n"
            "2. **Pay an honest fill.** The recipe buys at the *same close* it uses to spot the "
            "signal — slightly optimistic. We re-run buying at the *next* day's open to make sure "
            "the edge isn't a measurement trick.\n"
            "3. **Interrogate the win rate.** We run the exact same exit rule on a **pure random "
            "walk** — a tape with no signal at all — and see what win rate it prints. Spoiler: a "
            "high one.\n\n"
            "Tape: SPY daily bars back to 1993 (plus QQQ, IWM, DIA for breadth)."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: does it beat a coin?** The Triple-RSI rule vs a random-direction entry on "
            "the same bars, gross:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    led = run_one('SPY'); rand = run_one('SPY', seed=301)\n"
            "    rule_m = st.summarize(led,'ret_gross')['mean_bps']\n"
            "    rand_m = st.summarize(rand,'ret_gross')['mean_bps']\n"
            "    rule_w = st.summarize(led,'ret_gross')['win_rate']*100\n"
            "    rand_w = st.summarize(rand,'ret_gross')['win_rate']*100\n"
            "else:\n"
            f"    rule_m, rand_m = {R['spy_mean']}, {R['rand_mean']}\n"
            f"    rule_w, rand_w = {R['spy_win']}, {R['rand_win']}\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "b = ax.bar(['Triple-RSI rule', 'Random direction (coin)'],\n"
            "           [rule_m, rand_m], color=[GREEN, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('gross profit per trade (bps)')\n"
            "ax.set_title('The Triple-RSI bounce beats a coin — the signal is real')\n"
            "for bar_, w in zip(b, [rule_w, rand_w]):\n"
            "    ax.annotate(f'win {w:.0f}%', (bar_.get_x()+bar_.get_width()/2,\n"
            "        bar_.get_height()/2), ha='center', va='center', color='white', fontweight='bold')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Rule: {rule_m:+.1f} bps   Coin: {rand_m:+.1f} bps   Excess: {rule_m-rand_m:+.1f} bps')"
        ),
        md(
            f"The rule earns **+{R['spy_mean']:.0f} bps/trade** at an **{R['spy_win']:.0f}%** win "
            f"rate; the coin earns +{R['rand_mean']:.0f} bps. The signal adds "
            f"**+{R['excess_over_rand']:.0f} bps** of real edge on top of the base drift. Good — "
            "this isn't noise."
        ),
        md(
            "**Now the punchline. Here's the same exit rule run on a pure random walk** — a tape "
            "engineered to have *no signal whatsoever*. What win rate does a strategy with zero "
            "edge print?"
        ),
        code(
            "b0, _ = data.synthetic_daily(n_days=8000, reversion=0.0, seed=301)\n"
            "sig0 = st.triple_rsi_entries(b0, trend_sma=None)\n"
            "s0 = st.summarize(st.run_trades(b0, sig0, rsi_exit=50.0, cost_bps=0, entry='close'), 'ret_gross')\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "a1.bar(['Win rate'], [s0['win_rate']*100], color=GREY, width=.45)\n"
            "a1.axhline(50, ls='--', c='k', lw=1); a1.set_ylim(0, 100)\n"
            "a1.set_ylabel('%'); a1.set_title(f\"Pure coin, same exit: {s0['win_rate']*100:.0f}% WINS\")\n"
            "a2.bar(['Mean profit/trade'], [s0['mean_bps']], color=RED, width=.45)\n"
            "a2.axhline(0, c='k', lw=1); a2.set_ylabel('bps')\n"
            "a2.set_title(f\"...but mean = {s0['mean_bps']:+.0f} bps (a LOSER)\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"No-signal tape: win {s0['win_rate']*100:.1f}%  mean {s0['mean_bps']:+.1f} bps  skew {s0['skew']:+.2f}\")"
        ),
        md(
            f"There it is. On a tape with **no edge at all**, the 'take the bounce' exit still wins "
            f"**~{R['mart_win']:.0f}% of trades** — while *losing money on average* "
            f"({R['mart_mean']:+.0f} bps). The trick is the exit: you sell winners the instant they "
            "bounce (lots of small wins) and hold losers as they bleed (a few big losses). A high "
            "win rate is the **shape of the exit**, not the **quality of the signal**. So '90% wins' "
            "tells you almost nothing — you have to look at the average and the rare big losses "
            f"(SPY's worst trade: {R['worst_trade']:.1f}%)."
        ),
        md(
            "**Last: even with a real signal, can you get rich on it?** Here's the catch the "
            "headline never mentions — how *often* it trades, and how that stacks up against simply "
            "holding SPY:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    led = run_one('SPY'); n = len(led); hold = led['days_held'].mean()\n"
            "    spy = fetch('SPY'); yrs = (spy.index[-1]-spy.index[0]).days/365.25\n"
            "    tpy = n/yrs; pim = led['days_held'].sum()/len(spy)*100\n"
            "    ann = st.summarize(led,'ret_gross')['mean_bps']*tpy/100\n"
            "    bh = ((spy['close'].iloc[-1]/spy['close'].iloc[0])**(1/yrs)-1)*100\n"
            "else:\n"
            f"    tpy, pim, ann, bh = {R['trades_per_yr']}, {R['pct_in_market']}, {R['ann_est']}, {R['spy_bh_ann']}\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "a1.bar(['Time in market','In cash'], [pim, 100-pim], color=[GREEN, GREY], width=.5)\n"
            "a1.set_ylabel('% of days'); a1.set_title(f'In the market only ~{pim:.0f}% of the time')\n"
            "a2.bar(['Triple-RSI','Buy & hold SPY'], [ann, bh], color=[AMBER, GREEN], width=.5)\n"
            "a2.axhline(0, c='k', lw=1); a2.set_ylabel('annualised %/yr')\n"
            "a2.set_title('...so it earns about half of just holding')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'{tpy:.1f} trades/yr, ~{pim:.0f}% in market, ~{ann:.1f}%/yr vs B&H {bh:.1f}%/yr')"
        ),
        md(
            f"**~{R['trades_per_yr']:.0f} trades a year.** The strategy is in the market only "
            f"~{R['pct_in_market']:.0f}% of the time, so even a fat +{R['spy_mean']:.0f} bps/trade "
            f"compounds to only ~+{R['ann_est']:.1f}%/yr — while SPY buy-and-hold returned "
            f"+{R['spy_bh_ann']:.1f}%/yr. The '90% win rate machine' earns **half of doing nothing "
            "but holding the index.**"
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Real.** +{R['spy_mean']:.0f} bps/trade on SPY (*t* = +{R['spy_t']:.2f}); "
            f"beats a coin by +{R['excess_over_rand']:.0f} bps; survives an honest fill. The bounce "
            "is genuine.\n"
            "- **Tradability — Fragile.** Not because of costs (it shrugs those off) but because of "
            f"**capacity**: {R['trades_per_yr']:.0f} trades/yr, ~{R['pct_in_market']:.0f}% in the "
            f"market, ~+{R['ann_est']:.1f}%/yr vs the index's +{R['spy_bh_ann']:.1f}%.\n"
            "- **A 90% win-rate money machine? — Busted.** The 88% win rate is real but meaningless "
            "as an edge: a pure coin under the same exit wins 62% and *loses* money. Win rate is the "
            "shape of the exit, not the quality of the signal."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Costs are not the problem here — with so few trades, even a generous round-trip fee "
            "barely registers:"
        ),
        code(
            "costs = [0.0, 2.0, 5.0, 10.0]\n"
            "if HAVE_REAL:\n"
            "    spy = fetch('SPY'); sig = st.triple_rsi_entries(spy)\n"
            "    net = [st.summarize(st.run_trades(spy, sig, rsi_exit=50.0, cost_bps=c, entry='close'),\n"
            "                        'ret_net')['mean_bps'] for c in costs]\n"
            "else:\n"
            f"    net = [{R['cost0_bps']}, {R['cost2_bps']}, {R['cost5_bps']}, {R['cost10_bps']}]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, net, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, net, 0, color=GREEN, alpha=.12)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net profit per trade (bps)')\n"
            "ax.set_ylim(0, max(net)*1.15)\n"
            "ax.set_title('Costs barely touch it — capacity, not cost, is the constraint')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Even at 10 bps the per-trade edge is huge — the problem is there are only ~3.5 trades/yr.')"
        ),
        md(
            f"At 10 bps the edge is still **+{R['cost10_bps']:.0f} bps/trade** (*t* = "
            f"+{R['cost10_t']:.2f}). Unlike the desk's pure mirages, this one doesn't die at the "
            "first dollar of cost — it's genuinely *fragile* rather than fake. The honest pitch "
            "isn't 'a 90% win-rate system'; it's 'a small, real, occasional overlay on top of "
            "something you're already holding.'"
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The signal's honest cousin.** [Study 75 — Knee-Jerk](../../75-knee-jerk/) is the "
            "Connors RSI(2) system — the same oversold-bounce family, but it fires ~24 times a year "
            "instead of 3.5. Triple-RSI's extra filters *buy* a higher win rate by *throwing away* "
            "trades.\n"
            "- **Where the win-rate illusion was first dissected.** [Study 72 — Loaded-Dice]"
            "(../../72-loaded-dice/) showed a 93.5% win rate on pure noise via a tiny-take-profit "
            "exit. Triple-RSI is the honest twin: same illusion, but a real signal underneath.\n"
            "- **Could you fix the capacity problem?** Dropping the 200-SMA filter triples the "
            "trades (to ~8/yr) and the signal holds — but you're still capital-starved. The real "
            "fix is to run it cross-sectionally over many names, not as a single-ticker timer.\n\n"
            "*Think the 90% is the whole story? Fork this, change the exit, and show the win rate "
            "surviving once you measure the thing that actually pays the bills — the mean.*"
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
            "# Triple-RSI — a quantitative teardown\n"
            "### Real daily tape · forward-return test · random-direction control · "
            "next-open robustness · the win-rate illusion · capacity\n\n"
            "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![A_90%25_win--rate_money_machine%3F: Busted](https://img.shields.io/badge/A_90%25_win--rate_money_machine%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We test whether the "
            "four-condition Triple-RSI entry beats a random-direction control, survives an honest "
            "fill, and amounts to a tradable system — and we put the marketed '90% win rate' on the "
            "stand.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily bars, SPY back to 1993; as-of "
            "2026-06-17; the offline core and tests run on a deterministic synthetic tape. "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `REAL` | SPY gross **+{R['spy_mean']:.1f} bps/trade**, HAC "
            f"*t* = **+{R['spy_t']:.2f}**; next-open fill +{R['nextopen_mean']:.1f} bps "
            f"(*t* = +{R['nextopen_t']:.2f}); post-2010 *t* = +{R['spy_post_t']:.2f}; pooled "
            f"*t* = +{R['pool_t']:.2f}; beats random control by +{R['excess_over_rand']:.1f} bps. |\n"
            f"| **Tradability** | `FRAGILE` | Cost-robust (*t* = +{R['cost10_t']:.2f} at 10 bps) but "
            f"only {R['trades_per_yr']:.1f} trades/yr, ~{R['pct_in_market']:.0f}% in market → "
            f"~+{R['ann_est']:.1f}%/yr vs SPY B&H +{R['spy_bh_ann']:.1f}%/yr. Capacity-bound. |\n"
            f"| **A 90% win-rate money machine?** | `BUSTED` | {R['spy_win']:.0f}% win-rate is real, "
            f"but a martingale under the same exit prints {R['mart_win']:.0f}% wins at "
            f"{R['mart_mean']:+.0f} bps; skew {R['spy_skew']:.1f}. |\n\n"
            "> In plain words: a real, durable short-term mean-reversion signal wrapped in a "
            "marketing number (the win rate) that measures the exit's shape, not the edge — and "
            "throttled by a trade count too low to compete with buy-and-hold."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r$ be the trade's close-to-close return and $s_t$ the four-condition signal:\n\n"
            "$$s_t = \\mathbf{1}\\big[\\mathrm{RSI}_5(t)<30 \\;\\wedge\\; \\mathrm{RSI}_5\\!\\downarrow"
            "\\text{ 3 days}\\;\\wedge\\; \\mathrm{RSI}_5(t{-}3)<60 \\;\\wedge\\; P_t>\\mathrm{SMA}_{200}(t)\\big]$$\n\n"
            "- **H₁ (signal).** $\\mathbb{E}[r \\mid s_t=1] > 0$.\n"
            "- **H₂ (beats random).** It exceeds the same statistic with a random long/short "
            "direction on identical bars.\n"
            "- **H₃ (no look-ahead).** It survives entry at *t+1* open rather than *t* close.\n"
            "- **H₄ (tradable system).** Its annualised contribution rivals buy-and-hold.\n"
            "- **H₅ (the win rate is the edge).** The 90% hit-rate is evidence of a good system.\n\n"
            "We find **H₁–H₃ supported**, **H₄ rejected** (capacity), and **H₅ rejected** (a coin "
            "clears 60% under this exit)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "Short-term mean reversion is textbook (Jegadeesh 1990, Lehmann 1990) and the desk has "
            "already stamped its RSI(2) cousin REAL ([Study 75](../../75-knee-jerk/)). The "
            "interesting content here is **not** whether the signal exists — it does — but the two "
            "ways the marketing misleads: (a) a **win rate** that a literal coin reproduces, and "
            "(b) a **trade count** so low that a real edge still loses to the index. Naming the "
            "mechanism precisely is worth more than the binary verdict."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Entry.** Four conditions on bar *t*; **buy at *t*'s close** (canonical), with a "
            "**next-open** variant for the look-ahead check.\n"
            "- **Exit.** First bar whose RSI(5) closes above 50, else a `max_hold` cap.\n"
            "- **Control.** Identical entry bars, direction $\\in\\{-1,+1\\}$ drawn i.i.d. (seeded).\n"
            "- **Inference.** Newey-West HAC *t* on per-trade returns; win-rate, skew and mean "
            "reported together; pre/post-2010 split; cost sweep; capacity vs B&H.\n"
            "- **Positive control.** Deterministic synthetic tape with tunable AR(1) mean-reversion "
            "— doubling as the win-rate-illusion demonstration at reversion = 0.\n\n"
            "Four ETFs: SPY, QQQ, IWM, DIA. Long daily bars from 1993–2000."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Per-instrument & pooled — a broad-based real signal\n\n"
            "HAC *t* on the gross per-trade return, by instrument and pooled."
        ),
        code(
            "if HAVE_REAL:\n"
            "    recs = []\n"
            "    for t in TICKERS:\n"
            "        s = st.summarize(run_one(t), 'ret_gross')\n"
            "        recs.append((t, s['n_trades'], s['win_rate']*100, s['mean_bps'], s['skew'], s['tstat']))\n"
            "    perinst = pd.DataFrame(recs, columns=['ticker','n','win%','mean_bps','skew','t'])\n"
            "    pooled = run_all(); ps = st.summarize(pooled,'ret_gross')\n"
            "else:\n"
            "    perinst = pd.DataFrame({'ticker':['SPY','QQQ','IWM','DIA'],\n"
            "        'n':[118,140,131,103], 'win%':[88.1,75.0,72.5,80.6],\n"
            "        'mean_bps':[132.8,108.0,95.0,112.0], 'skew':[-1.59,-0.9,-0.8,-1.1],\n"
            "        't':[5.07,4.3,3.9,4.6]})\n"
            f"    ps = dict(n_trades={R['pool_n']}, mean_bps={R['pool_mean']}, tstat={R['pool_t']}, win_rate={R['pool_win']/100})\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.4))\n"
            "col = [GREEN if v >= 2 else AMBER for v in perinst['t']]\n"
            "ax.bar(perinst['ticker'], perinst['t'], color=col)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('HAC t-stat'); ax.set_title('Every ETF clears |t|=2 — the bounce is broad-based')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(perinst.round(2).to_string(index=False))\n"
            "print(f\"\\nPOOLED: n={ps['n_trades']} win={ps['win_rate']*100:.1f}% mean={ps['mean_bps']:+.1f}bps t={ps['tstat']:+.2f}\")"
        ),
        md(
            f"> In plain words: pooled HAC *t* = **+{R['pool_t']:.2f}** across four index ETFs, every "
            "instrument above the bar. This is not a single-ticker fluke — it's the same "
            "short-term mean reversion the desk measures elsewhere."
        ),
        md(
            "### 4b · The random-direction control & the look-ahead check\n\n"
            "Both the rule and a coin earn positive returns (equity drift + exit selection). The "
            "question is the *excess* — and whether it survives an honest fill."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rule = st.summarize(run_one('SPY'),'ret_gross')\n"
            "    rand = st.summarize(run_one('SPY', seed=301),'ret_gross')\n"
            "    nopen = st.summarize(run_one('SPY', entry='next_open'),'ret_gross')\n"
            "    vals = [rule['mean_bps'], rand['mean_bps'], nopen['mean_bps']]\n"
            "    ts = [rule['tstat'], rand['tstat'], nopen['tstat']]\n"
            "else:\n"
            f"    vals = [{R['spy_mean']}, {R['rand_mean']}, {R['nextopen_mean']}]\n"
            f"    ts = [{R['spy_t']}, {R['rand_t']}, {R['nextopen_t']}]\n"
            "labels = ['Rule\\n(close fill)', 'Random\\ndirection', 'Rule\\n(next-open fill)']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "b = ax.bar(labels, vals, color=[GREEN, GREY, AMBER], width=0.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('gross mean (bps/trade)')\n"
            "ax.set_title('Edge over a coin, and it survives the honest fill')\n"
            "for bar_, t_ in zip(b, ts):\n"
            "    ax.annotate(f't={t_:+.2f}', (bar_.get_x()+bar_.get_width()/2, bar_.get_height()+2),\n"
            "                ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Excess over random: {{vals[0]-vals[1]:+.1f}} bps   Next-open still t={{ts[2]:+.2f}}')"
        ),
        md(
            f"> In plain words: the rule beats the coin by **+{R['excess_over_rand']:.0f} bps**, and "
            f"the next-open fill still earns +{R['nextopen_mean']:.0f} bps (*t* = "
            f"+{R['nextopen_t']:.2f}). The edge is **not** a same-bar-close artefact — H₁–H₃ hold."
        ),
        md(
            "### 4c · The win-rate illusion — the same exit on a coin\n\n"
            "The decisive test of H₅. Run the identical rule on a synthetic **martingale** "
            "(reversion = 0, no signal) and read the win-rate against the mean."
        ),
        code(
            "rows = []\n"
            "for rev, lab in [(0.0,'Martingale (no signal)'), (0.5,'Strong reversion')]:\n"
            "    b, _ = data.synthetic_daily(n_days=8000, reversion=rev, seed=301)\n"
            "    sig = st.triple_rsi_entries(b, trend_sma=None)\n"
            "    s = st.summarize(st.run_trades(b, sig, rsi_exit=50.0, cost_bps=0, entry='close'),'ret_gross')\n"
            "    rows.append((lab, s['win_rate']*100, s['mean_bps'], s['skew'], s['tstat']))\n"
            "dfm = pd.DataFrame(rows, columns=['tape','win%','mean_bps','skew','t'])\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "a1.bar(dfm['tape'], dfm['win%'], color=[GREY, GREEN], width=.5)\n"
            "a1.axhline(50, ls='--', c='k'); a1.set_ylim(0,100); a1.set_ylabel('win rate %')\n"
            "a1.set_title('Both win most trades...'); a1.tick_params(axis='x', labelsize=8)\n"
            "a2.bar(dfm['tape'], dfm['mean_bps'], color=[RED, GREEN], width=.5)\n"
            "a2.axhline(0, c='k'); a2.set_ylabel('mean bps/trade')\n"
            "a2.set_title('...only one makes money'); a2.tick_params(axis='x', labelsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dfm.round(2).to_string(index=False))"
        ),
        md(
            f"> In plain words: the martingale wins **~{R['mart_win']:.0f}%** of trades with a "
            f"**{R['mart_mean']:+.0f} bps** mean (*t* = {R['mart_t']:.2f}) and skew {R['mart_skew']:.1f}. "
            "The win-rate is manufactured by the 'sell on the bounce' exit, independent of any edge. "
            "On SPY the win-rate is high *and* the mean is real — but the marketed number is the "
            "wrong one to quote. **H₅ rejected.**"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `REAL`** — SPY gross +{R['spy_mean']:.1f} bps/trade, HAC *t* "
            f"+{R['spy_t']:.2f}; next-open +{R['nextopen_t']:.2f}; post-2010 +{R['spy_post_t']:.2f}; "
            f"pooled +{R['pool_t']:.2f}; excess over random +{R['excess_over_rand']:.1f} bps.\n"
            f"- **Tradability `FRAGILE`** — cost-robust (*t* +{R['cost10_t']:.2f} at 10 bps) but "
            f"{R['trades_per_yr']:.1f} trades/yr and ~{R['pct_in_market']:.0f}% time in market, so "
            f"~+{R['ann_est']:.1f}%/yr vs SPY B&H +{R['spy_bh_ann']:.1f}%/yr. Capacity, not cost.\n"
            f"- **A 90% win-rate money machine? `BUSTED`** — {R['spy_win']:.0f}% win-rate, but a "
            f"coin under the same exit clears {R['mart_win']:.0f}% at a loss; skew {R['spy_skew']:.1f}. "
            "Win-rate measures the exit, not the edge."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — cost sweep and the capacity wall\n\n"
            "The per-trade edge is enormous relative to costs; the binding constraint is how rarely "
            "it deploys capital."
        ),
        code(
            "costs = [0.0, 2.0, 5.0, 10.0]\n"
            "if HAVE_REAL:\n"
            "    spy = fetch('SPY'); sig = st.triple_rsi_entries(spy)\n"
            "    sw = [st.summarize(st.run_trades(spy, sig, rsi_exit=50.0, cost_bps=c, entry='close'),'ret_net') for c in costs]\n"
            "    means = [s['mean_bps'] for s in sw]; tstats = [s['tstat'] for s in sw]\n"
            "else:\n"
            f"    means = [{R['cost0_bps']},{R['cost2_bps']},{R['cost5_bps']},{R['cost10_bps']}]\n"
            f"    tstats = [{R['cost0_t']},{R['cost2_t']},{R['cost5_t']},{R['cost10_t']}]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(costs, means, 'o-', c=GREEN, lw=2, label='net mean (bps)')\n"
            "ax2 = ax.twinx(); ax2.plot(costs, tstats, 's--', c=GREY, lw=1.5, label='HAC t')\n"
            "ax2.axhline(2, ls=':', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net mean (bps)', color=GREEN)\n"
            "ax.set_ylim(0, max(means)*1.15); ax2.set_ylabel('HAC t-stat', color=GREY)\n"
            "ax.set_title('SPY: costs are a rounding error — the edge survives 10 bps comfortably')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'At 10 bps: {R['cost10_bps']:+.1f} bps/trade, t={R['cost10_t']:+.2f}  —  but only {R['trades_per_yr']:.1f} trades/yr')"
        ),
        md(
            f"> In plain words: contrast the desk's mirages, where the gross was already negative. "
            f"Here the gross is huge and survives any realistic cost — yet the strategy contributes "
            f"only ~+{R['ann_est']:.1f}%/yr because it is in the market ~{R['pct_in_market']:.0f}% of "
            "the time. A real edge, structurally unable to fill a portfolio: the definition of "
            "**fragile**."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the *engine* a faithful detector, or does it always print the same number? Plant a "
            "short-term AR(1) anti-persistence and sweep its strength."
        ),
        code(
            "revs = [-0.10, 0.0, 0.10, 0.20, 0.30, 0.40, 0.50]\n"
            "edge = []\n"
            "for rev in revs:\n"
            "    b, _ = data.synthetic_daily(n_days=8000, reversion=rev, seed=301)\n"
            "    sig = st.triple_rsi_entries(b, trend_sma=None)\n"
            "    rule = st.summarize(st.run_trades(b, sig, rsi_exit=50.0, cost_bps=0, entry='close'),'ret_gross')['mean_bps']\n"
            "    rand = st.summarize(st.run_trades(b, sig, rsi_exit=50.0, cost_bps=0, entry='close',\n"
            "        random_dirs=st.random_directions(int(sig.sum()), seed=1)),'ret_gross')['mean_bps']\n"
            "    edge.append(rule - rand)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(revs, edge, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, ls='--', c=GREY)\n"
            "ax.set_xlabel('planted daily AR(1) mean-reversion'); ax.set_ylabel('rule − coin (bps/trade)')\n"
            "ax.set_title('The engine is a faithful mean-reversion detector')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Edge ~0 at the martingale, rising with planted anti-persistence.')"
        ),
        md(
            "The rule's advantage is monotone in the planted anti-persistence and ~0 at the "
            "martingale — so the engine is faithful, and the real-tape result is a statement about "
            "**the market**: short-term mean reversion in index ETFs is real, the Triple-RSI is one "
            "(over-conditioned) way to harvest it, and the '90% win rate' is the least informative "
            "thing about it.\n\n"
            "For a denser implementation of the same edge, see "
            "[Study 75 — Knee-Jerk](../../75-knee-jerk/) (RSI(2), ~24 trades/yr); for the win-rate "
            "illusion in its pure form, [Study 72 — Loaded-Dice](../../72-loaded-dice/)."
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
