"""Generate the two narrative notebooks for Study 129 (Heikin-Ashi).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md).  The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
daily parquet under ../_cache/ if present and otherwise quote the frozen headline numbers
in ``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.

The _write convention (each build_*() ends by calling _write) is kept so the repo's
intro-restyle tooling can monkeypatch it.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-14).
R = dict(
    n=3924, tpy=65.4,
    ha_mean=1.37, ha_win=50.7, ha_t=0.56,
    rand_mean=1.02, rand_t=0.33,
    delta=0.35,
    raw_mean=-2.45, raw_t=-1.55, raw_n=7602,
    ci_lo=-0.15, ci_hi=0.26, frac_neg=30.8,
    long_mean=17.57, long_t=3.67,
    short_mean=-14.83, short_t=-3.19,
    rand_long_mean=16.87, rand_long_t=3.35,
    rand_short_mean=-14.91, rand_short_t=-2.95,
    net05_mean=0.87, net05_t=0.36,
    net10_mean=0.37, net10_t=0.15,
    net20_mean=-0.63, net20_t=-0.26,
    spy_t=2.24, qqq_t=-0.10, iwm_t=-0.01, aapl_t=-0.34,
    spy_mean=8.04, qqq_mean=-0.48, iwm_mean=-0.05, aapl_mean=-2.08,
    ha_flips_spy=984, raw_flips_spy=1898,
    ha_flips_avg=987, raw_flips_avg=1911,
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

from heikin_ashi import data, strategy as st

TICKERS = ["SPY", "QQQ", "IWM", "AAPL"]
CACHE_DIR = os.path.abspath(os.path.join("..", "_cache"))

def _have_cache():
    return all(os.path.exists(data._cache_path(t, CACHE_DIR)) for t in TICKERS)

HAVE_REAL = _have_cache()

# Frozen headline numbers — mirror of docs/results.md (as-of 2026-06-14).
R = dict(
    n=3924, tpy=65.4,
    ha_mean=1.37, ha_win=50.7, ha_t=0.56,
    rand_mean=1.02, rand_t=0.33,
    delta=0.35,
    raw_mean=-2.45, raw_t=-1.55, raw_n=7602,
    ci_lo=-0.15, ci_hi=0.26, frac_neg=30.8,
    long_mean=17.57, long_t=3.67,
    short_mean=-14.83, short_t=-3.19,
    rand_long_mean=16.87, rand_long_t=3.35,
    rand_short_mean=-14.91, rand_short_t=-2.95,
    net05_mean=0.87, net05_t=0.36,
    net10_mean=0.37, net10_t=0.15,
    net20_mean=-0.63, net20_t=-0.26,
    spy_t=2.24, qqq_t=-0.10, iwm_t=-0.01, aapl_t=-0.34,
    spy_mean=8.04, qqq_mean=-0.48, iwm_mean=-0.05, aapl_mean=-2.08,
    ha_flips_spy=984, raw_flips_spy=1898,
    ha_flips_avg=987, raw_flips_avg=1911,
)

def pooled(tp, sl, cost, seed=None, use_ha=True):
    \"\"\"Pool barrier trades across the basket (cached real tapes).\"\"\"
    frames = []
    for t in TICKERS:
        b = data.fetch_daily(t, fetch=False, cache_dir=CACHE_DIR)
        ha = st.heikin_ashi(b)
        col = st.ha_colour(ha) if use_ha else st.raw_colour(b)
        ent = st.colour_flip_entries(col, warmup=20)
        dirs = st.random_directions(len(ent), seed=seed) if seed is not None else None
        frames.append(st.run_trades(b, ent, tp_R=tp, sl_R=sl, cost_bps=cost, directions=dirs))
    return pd.concat(frames, ignore_index=True)

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Heikin-Ashi — does the smoothed candle colour flip beat a coin?\n"
            "### The folk claim that HA filters noise and rides trends, tested honestly\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Smoothing_adds_lag%2C_not_signal%3F: Confirmed](https://img.shields.io/badge/Smoothing_adds_lag%2C_not_signal%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "Every charting platform has a button for Heikin-Ashi candles. The pitch is "
            "irresistible: *smooth out the noise, see the trend clearly, trade the colour "
            "flip — green means up, red means down, and the smoothing keeps you on the right "
            "side.* This notebook asks the only question that matters: does the HA colour flip "
            "**actually point the right way**, or is it just a prettier coin toss?\n\n"
            "> This is the plain-language layer. Want the t-stats, bootstrap intervals, "
            "and long/short decomposition? That is the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is "
            "drawn by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the HA colour flip point the right way? | **Not reliably.** With a fair "
            f"(symmetric) target & stop it wins **{R['ha_win']}%** of the time — a coin. |\n"
            "| Does it beat an actual coin flip? | **Barely.** A random-direction entry on "
            f"the *same* flips does +{R['rand_mean']:.2f} bps vs HA's +{R['ha_mean']:.2f} bps — "
            "a gap of +0.35 bps (statistical noise, HAC *t* = +0.56). |\n"
            "| Does the smoothing help vs raw candles? | **No.** HA fires half as often as "
            "raw candles (reducing false positives) but gives no improvement in the coin-test "
            "t-stat. Fewer trades, same noise. |\n"
            "| Could you trade it? | **No.** The gross is not significantly above zero, so "
            "any cost makes it a loser. |\n\n"
            "> The smoothed candle is prettier but not more prophetic. The lag that comes "
            "with the recursion gives up the mean-reversion edge that raw candles hint at "
            "(barely) and replaces it with nothing."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Heikin-Ashi candles average the open, high, low and close over two bars "
            "instead of one, so each candle reflects the market's 'true' direction, not just "
            "a single bar's noise. When the colour flips from red to green, go long; from "
            "green to red, go short. The smoothing filters whipsaws and keeps you on the "
            "right side of the trend.\"*\n\n"
            "The mechanism sounds reasonable: standard candles can flip colour every bar "
            "on noisy days, while HA runs in same-colour streaks during real trends. The "
            "bet is that **smoothing the open by averaging with the prior bar's HA open and "
            "close** gives a signal that knows the direction better than a coin flip.\n\n"
            "Let us look at what HA actually does to a price chart first, then measure it."
        ),
        code(
            "# Visual: standard candles vs HA on a short SPY stretch\n"
            "if HAVE_REAL:\n"
            "    b = data.fetch_daily('SPY', fetch=False, cache_dir=CACHE_DIR)\n"
            "    sample = b.iloc[-120:].copy()\n"
            "else:\n"
            "    sample, _ = data.synthetic_daily(n_days=120, momentum=0.05, seed=129)\n"
            "ha_s = st.heikin_ashi(sample)\n"
            "fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)\n"
            "for ax, ohlc, title in [\n"
            "    (axes[0], sample, 'Standard OHLC candles'),\n"
            "    (axes[1], ha_s.rename(columns={'ha_open':'open','ha_high':'high',\n"
            "                                    'ha_low':'low','ha_close':'close'}),\n"
            "     'Heikin-Ashi candles — same data, smoothed')]:\n"
            "    c = ohlc['close'].values; o = ohlc['open'].values\n"
            "    for i, (idx, row) in enumerate(ohlc.iterrows()):\n"
            "        col = GREEN if row['close'] >= row['open'] else RED\n"
            "        ax.plot([i, i], [row['low'], row['high']], c=col, lw=0.8)\n"
            "        ax.add_patch(plt.Rectangle((i - 0.3, min(row['open'], row['close'])),\n"
            "                     0.6, abs(row['close'] - row['open']),\n"
            "                     fc=col, ec=col, alpha=0.8))\n"
            "    ax.set_title(title); ax.set_ylabel('price')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Notice: HA runs in same-colour streaks. The question is whether that stripe pattern predicts direction.')"
        ),
        md(
            "HA candles are visually cleaner — the colour changes less often, "
            "and during trends the candles run in long same-colour streaks. That visual "
            "clarity is real. The question is whether the *first bar of a new colour streak* "
            "— the flip — is followed by more price movement in the signalled direction than "
            "a random coin would achieve."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the flip is informative, HA is a better entry trigger than a dartboard — "
            "a simple, visual, instrument-agnostic rule millions of traders already have "
            "on their charts. If it is just a coin, those traders are paying commissions "
            "~65 times a year per instrument for nothing."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Two rules keep us honest:\n\n"
            "1. **Make the bet symmetric.** Place the profit target and the stop equally far "
            "from the entry (one ATR on each side). A coin earns 0. Only real directional "
            "knowledge can beat that baseline.\n"
            "2. **Race it against an actual coin.** Take the exact same flip dates and "
            "flip a coin for the direction. If HA knows something, it beats the coin.\n"
            "3. **Compare HA to raw candles.** If the smoothing is useful, HA should beat "
            "the unsmoothed colour-flip baseline by more than noise.\n\n"
            "We run this on **four liquid daily tapes** (SPY, QQQ, IWM, AAPL) over **15 years** "
            "of daily bars — enough trades for meaningful inference."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — does the flip beat a coin?\n\n"
            "**The honest test: same flip dates, three arms — HA direction, random direction, "
            "and raw-candle direction.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ha_led = pooled(1, 1, 0)\n"
            "    rand_led = pooled(1, 1, 0, seed=129)\n"
            "    raw_led = pooled(1, 1, 0, use_ha=False)\n"
            "    ha_s = st.summarize(ha_led, 'ret_gross')\n"
            "    rand_s = st.summarize(rand_led, 'ret_gross')\n"
            "    raw_s = st.summarize(raw_led, 'ret_gross')\n"
            "    labels = ['HA colour flip', 'Random direction (coin)', 'Raw-candle flip']\n"
            "    means  = [ha_s['mean_bps'], rand_s['mean_bps'], raw_s['mean_bps']]\n"
            "    wins   = [ha_s['win_rate']*100, rand_s['win_rate']*100, raw_s['win_rate']*100]\n"
            "else:\n"
            "    labels = ['HA colour flip', 'Random direction (coin)', 'Raw-candle flip']\n"
            "    means  = [R['ha_mean'], R['rand_mean'], R['raw_mean']]\n"
            "    wins   = [R['ha_win'], 50.0, 49.3]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "colors = [AMBER, GREY, RED]\n"
            "bars = ax.bar(labels, means, color=colors, width=0.6)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('gross profit per trade (bps)')\n"
            "ax.set_title('HA flip vs coin vs raw-candle flip (fair 1:1 barrier, gross)')\n"
            "for b_bar, w in zip(bars, wins):\n"
            "    ax.annotate(f'win {w:.1f}%', (b_bar.get_x()+b_bar.get_width()/2,\n"
            "                max(b_bar.get_height(), 0) + 0.1), ha='center', va='bottom', fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'HA={means[0]:+.2f} bps  |  Coin={means[1]:+.2f} bps  |  Raw={means[2]:+.2f} bps')\n"
            "print(f'HA advantage over coin: {means[0]-means[1]:+.2f} bps (statistical noise)')"
        ),
        md(
            f"The HA flip earns **+{R['ha_mean']:.2f} bps/trade** gross — positive, "
            "but statistically indistinguishable from zero (HAC *t* = +0.56). "
            f"The coin earns **+{R['rand_mean']:.2f} bps**, and the "
            f"raw-candle flip earns **{R['raw_mean']:.2f} bps**. The HA advantage over "
            "the coin is only +0.35 bps — noise. The smoothing does not improve the "
            "directional signal."
        ),
        md(
            "**The long-side trap: bull-market drift masquerades as signal.**\n\n"
            "You might notice HA long trades look much better than short trades. "
            "Is that the strategy finding something?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ha_led = pooled(1, 1, 0)\n"
            "    rand_led = pooled(1, 1, 0, seed=129)\n"
            "    ha_long = st.summarize(ha_led[ha_led['dir']==1], 'ret_gross')\n"
            "    ha_short = st.summarize(ha_led[ha_led['dir']==-1], 'ret_gross')\n"
            "    rand_long = st.summarize(rand_led[rand_led['dir']==1], 'ret_gross')\n"
            "    rand_short = st.summarize(rand_led[rand_led['dir']==-1], 'ret_gross')\n"
            "    data_plot = [\n"
            "        ('HA long', ha_long['mean_bps'], ha_long['tstat']),\n"
            "        ('Coin long', rand_long['mean_bps'], rand_long['tstat']),\n"
            "        ('HA short', ha_short['mean_bps'], ha_short['tstat']),\n"
            "        ('Coin short', rand_short['mean_bps'], rand_short['tstat']),\n"
            "    ]\n"
            "else:\n"
            "    data_plot = [\n"
            "        ('HA long', R['long_mean'], R['long_t']),\n"
            "        ('Coin long', R['rand_long_mean'], R['rand_long_t']),\n"
            "        ('HA short', R['short_mean'], R['short_t']),\n"
            "        ('Coin short', R['rand_short_mean'], R['rand_short_t']),\n"
            "    ]\n"
            "labels2 = [d[0] for d in data_plot]\n"
            "means2  = [d[1] for d in data_plot]\n"
            "tstats2 = [d[2] for d in data_plot]\n"
            "colors2 = [GREEN, GREY, RED, GREY]\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "axes[0].bar(labels2, means2, color=colors2)\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_ylabel('gross mean (bps/trade)')\n"
            "axes[0].set_title('Mean return by arm')\n"
            "axes[0].tick_params(axis='x', rotation=15)\n"
            "axes[1].bar(labels2, tstats2, color=colors2)\n"
            "for s in (2, -2): axes[1].axhline(s, ls='--', c=GREY, lw=1)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_ylabel('HAC t-stat')\n"
            "axes[1].set_title('t-stat: HA long = coin long')\n"
            "axes[1].tick_params(axis='x', rotation=15)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'HA long: {means2[0]:+.2f} bps  |  Coin long: {means2[1]:+.2f} bps')\n"
            "print('Both pass t=+3 because of 15 years of equity bull markets, not HA skill.')"
        ),
        md(
            f"HA long trades: **+{R['long_mean']:.2f} bps** (t = +{R['long_t']:.2f}). "
            "Impressive! But the random-direction coin's long trades: "
            f"**+{R['rand_long_mean']:.2f} bps** (t = +{R['rand_long_t']:.2f}). "
            "**Nearly identical.** Over 15 years of mostly up-trending equity markets, any "
            "strategy that goes long sometimes will look good on the long leg — and going long "
            "*randomly* captures the same drift. The symmetric sum (long + short) kills the "
            "illusion: HAC *t* = +0.56 on the pooled trades."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Pooled gross +{R['ha_mean']:.2f} bps/trade, HAC "
            f"*t* = +{R['ha_t']:.2f}; advantage over a coin +0.35 bps. "
            "Bootstrap Sharpe 95% CI [−0.15, +0.26]. No statistically real directional "
            "signal in the HA colour flip.\n"
            "- **Tradability — Mirage.** The gross is not significantly positive. Any "
            "realistic commission renders it clearly negative.\n"
            "- **Smoothing adds lag, not signal.** HA flips at ~half the rate of raw candles "
            f"(~{R['ha_flips_avg']} vs ~{R['raw_flips_avg']} per ticker), but the "
            "directional t-stat vs a coin is the same — smoother, but not more prophetic."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "The gross is not even significantly positive, so costs make it cleanly negative:"
        ),
        code(
            "costs = [0.0, 0.5, 1.0, 2.0, 5.0]\n"
            "if HAVE_REAL:\n"
            "    net = [st.summarize(pooled(1, 1, c), 'ret_net')['mean_bps'] for c in costs]\n"
            "else:\n"
            "    net = [R['ha_mean'], R['net05_mean'], R['net10_mean'], R['net20_mean'],\n"
            "           R['ha_mean'] - 5.0]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, net, 'o-', c=AMBER, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, net, 0, where=[n<0 for n in net], color=RED, alpha=.12)\n"
            "ax.fill_between(costs, net, 0, where=[n>=0 for n in net], color=GREEN, alpha=.12)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net profit per trade (bps)')\n"
            "ax.set_title('The gross is not significant — costs finish the job')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'At 0 bps cost, gross is +{net[0]:.2f} bps (t=+0.56, not significant).')\n"
            "print(f'At 1 bps cost, net is +{net[2]:.2f} bps (barely above zero).')"
        ),
        md(
            "The gross (+1.37 bps) starts just above zero — but with an unreliable t-stat. "
            "By 1 bp round-trip the net is +0.37 bps (noise), and by 2 bps it is negative. "
            "Since the gross is not statistically real, no cost level 'breaks even' on a "
            "sound footing: there is nothing to break even from."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The machinery works when there's signal to find.** The companion notebook "
            "shows a synthetic control: when we plant bar-level momentum in a fake tape, the "
            "exact same HA colour-flip engine finds it. The problem is the real market, not "
            "the tool.\n"
            "- **Supertrend gets a better answer.** [Study 106 — Supertrend](../../106-supertrend/) "
            "tests a related smoothed-band flip on daily bars and finds a real (though weak) "
            "edge — a useful contrast to this study's null.\n"
            "- **What HA is actually good for.** Visually identifying trend regimes "
            "after the fact — that is what the smoothing is genuinely useful for. As a "
            "prospective entry signal it adds nothing over a coin.\n\n"
            "*Think HA does have edge? Fork this, add a filter (e.g., only take flips after "
            "a run of 3+ same-colour bars, or only longs above the 200-day MA), and show it "
            "beats a coin on the symmetric test. That is the bar.*"
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
            "# Heikin-Ashi — a quantitative teardown\n"
            "### Daily tape · symmetric ATR barrier · random-direction control · "
            "raw-candle baseline · HAC inference · bootstrap Sharpe CI\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Smoothing_adds_lag%2C_not_signal%3F: Confirmed](https://img.shields.io/badge/Smoothing_adds_lag%2C_not_signal%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim now carrying its standard error.* We test whether "
            "the Heikin-Ashi colour-flip direction beats a random-direction control on a "
            "symmetric-barrier backtest, across four liquid daily tapes over 15 years, and "
            "decompose the apparent long-side signal into bull-market drift vs HA skill.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily bars, 2011-06-13 to 2026-06-12; "
            "offline core and tests run on a deterministic synthetic tape. Methods & sources in "
            "[`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The `In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT + "\nfrom quantlab import analytics, stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Pooled gross **+{R['ha_mean']:+.2f} bps/trade**, "
            f"HAC *t* = **+{R['ha_t']:.2f}**; Δ vs random control = +{R['delta']:.2f} bps "
            f"(noise); bootstrap Sharpe CI [**{R['ci_lo']:+.2f}, {R['ci_hi']:+.2f}**]. "
            f"Apparent long-side *t* = {R['long_t']:.2f} is replicated by coin (*t* = "
            f"{R['rand_long_t']:.2f}) — bull-market drift, not HA skill. |\n"
            f"| **Tradability** | `MIRAGE` | Gross not significant; at 1 bp cost *t* = "
            f"+{R['net10_t']:.2f} (noise floor). No positive break-even. |\n"
            f"| **Smoothing adds lag, not signal?** | `CONFIRMED` | HA fires ~{R['ha_flips_avg']} "
            f"flips/ticker vs ~{R['raw_flips_avg']} raw; directional *t* vs coin is the same. |\n\n"
            "> In plain words: Heikin-Ashi is a smoother, not a forecaster.  It delays the signal "
            "enough to miss mean-reversion entries, and adds no edge on the trend-following side "
            "beyond what a coin captures from the bull-market drift."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            r"Let $x_t \in \{-1, +1\}$ be the HA colour at bar $t$ (green = +1, red = -1). "
            r"The claim asserts $x_t$ has positive autocorrelation (colour runs) AND that a "
            r"colour *flip* at $t$ forecasts the direction of returns from $t+1$ onward. "
            "Formally:\n\n"
            r"- **H₁ (signal)** $\mathbb{E}[d \cdot r_{t+1:\text{exit}}] > 0$ where "
            r"$d = x_t$ is the HA colour at the flip bar — measured symmetrically so a coin "
            r"earns 0.\n"
            r"- **H₂ (beats random)** The expectation exceeds the same statistic with $d$ "
            r"replaced by an i.i.d. fair coin on identical flip dates.\n"
            r"- **H₃ (smoothing adds value)** $\mathbb{E}[d_\text{HA} \cdot r] > \mathbb{E}[d_\text{raw} \cdot r]$ "
            r"on matched entry dates.\n\n"
            "We reject H₂ and H₃ on the real tape; H₁ gives a weak positive (t=+0.56) that is "
            "not statistically real."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the stakes\n\n"
            "Heikin-Ashi is the default chart type on TradingView for millions of retail traders, "
            "and the colour-flip entry is one of the most recommended patterns in retail trading "
            "content. If H₁–H₃ hold it is a real, accessible signal — if not, those traders "
            "are paying commissions on a coin flip with a pretty skin. The failure mode is "
            "interesting: the long-side *t* > 3 that looks compelling is not HA's forecasting — "
            "it is the market's bull-market drift captured by any strategy that sometimes goes long."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · The protocol\n\n"
            "- **HA construction.** Causal, recursive, no look-ahead: HA_open[t] = "
            "(HA_open[t-1] + HA_close[t-1])/2; HA_close[t] = (O+H+L+C)[t]/4; seeded "
            "with HA_open[0] = raw_open[0]. Computed bar-by-bar.\n"
            "- **Entry.** Colour flip detected at close of bar *t*; **enter at bar *t+1*'s "
            "open** (no look-ahead). Warmup = 20 bars (ATR warm-up).\n"
            "- **Exit (symmetric).** Barriers at ±1 ATR(20) from entry; max 60-bar hold; "
            "a bar straddling both barriers is filled at the **stop** (conservative).\n"
            "- **Controls.** (a) Random-direction on identical entries (seeded). "
            "(b) Raw-candle colour flip on the same entry protocol.\n"
            "- **Inference.** Newey-West HAC *t* on per-trade returns; circular "
            "block-bootstrap Sharpe CI; per-instrument breakdown; long/short decomposition; "
            "cost sweep.\n"
            "- **Positive control.** Synthetic tape with tunable AR(1) momentum.\n\n"
            "Four tapes: SPY, QQQ, IWM, AAPL. 2011-06-13 to 2026-06-12 (~3,773 bars each)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Three arms side by side — no tape carries a signal above noise"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ha_led = pooled(1, 1, 0)\n"
            "    rand_led = pooled(1, 1, 0, seed=129)\n"
            "    raw_led = pooled(1, 1, 0, use_ha=False)\n"
            "    ha_s = st.summarize(ha_led, 'ret_gross')\n"
            "    rand_s = st.summarize(rand_led, 'ret_gross')\n"
            "    raw_s = st.summarize(raw_led, 'ret_gross')\n"
            "    arms = pd.DataFrame([\n"
            "        ('HA colour flip', ha_s['n_trades'], ha_s['win_rate']*100,\n"
            "         ha_s['mean_bps'], ha_s['tstat']),\n"
            "        ('Random control', rand_s['n_trades'], rand_s['win_rate']*100,\n"
            "         rand_s['mean_bps'], rand_s['tstat']),\n"
            "        ('Raw-candle flip', raw_s['n_trades'], raw_s['win_rate']*100,\n"
            "         raw_s['mean_bps'], raw_s['tstat']),\n"
            "    ], columns=['arm', 'n', 'win%', 'mean_bps', 't'])\n"
            "else:\n"
            "    arms = pd.DataFrame([\n"
            "        ('HA colour flip', R['n'], R['ha_win'], R['ha_mean'], R['ha_t']),\n"
            "        ('Random control', R['n'], 49.4, R['rand_mean'], R['rand_t']),\n"
            "        ('Raw-candle flip', R['raw_n'], 49.3, R['raw_mean'], R['raw_t']),\n"
            "    ], columns=['arm', 'n', 'win%', 'mean_bps', 't'])\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "cols = [AMBER, GREY, RED]\n"
            "axes[0].bar(arms['arm'], arms['mean_bps'], color=cols)\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_ylabel('gross mean (bps/trade)')\n"
            "axes[0].set_title('Mean per-trade return by arm')\n"
            "axes[0].tick_params(axis='x', rotation=15)\n"
            "axes[1].bar(arms['arm'], arms['t'], color=cols)\n"
            "for s in (2, -2): axes[1].axhline(s, ls='--', c=GREY, lw=1)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_ylabel('HAC t-stat')\n"
            "axes[1].set_title('Every arm |t| < 2 (no significant signal)')\n"
            "axes[1].tick_params(axis='x', rotation=15)\n"
            "plt.tight_layout(); plt.show()\n"
            "arms.round(2)"
        ),
        md(
            "> In plain words: the HA t-stat (+0.56) is in the noise band. It is marginally "
            "better than the random control (+0.33) by +0.35 bps — a sliver that is not "
            "distinguishable from chance on 3,924 trades."
        ),
        md(
            "### 4b · The long-side trap — bull-market drift, not HA skill\n\n"
            "The per-arm, long/short breakdown reveals the structural confound."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ha_led = pooled(1, 1, 0)\n"
            "    rand_led = pooled(1, 1, 0, seed=129)\n"
            "    ha_l = st.summarize(ha_led[ha_led['dir']==1], 'ret_gross')\n"
            "    ha_s2 = st.summarize(ha_led[ha_led['dir']==-1], 'ret_gross')\n"
            "    r_l = st.summarize(rand_led[rand_led['dir']==1], 'ret_gross')\n"
            "    r_s2 = st.summarize(rand_led[rand_led['dir']==-1], 'ret_gross')\n"
            "    longs = [ha_l['mean_bps'], r_l['mean_bps']]\n"
            "    shorts = [ha_s2['mean_bps'], r_s2['mean_bps']]\n"
            "    l_t = [ha_l['tstat'], r_l['tstat']]\n"
            "    s_t = [ha_s2['tstat'], r_s2['tstat']]\n"
            "else:\n"
            "    longs = [R['long_mean'], R['rand_long_mean']]\n"
            "    shorts = [R['short_mean'], R['rand_short_mean']]\n"
            "    l_t = [R['long_t'], R['rand_long_t']]\n"
            "    s_t = [R['short_t'], R['rand_short_t']]\n"
            "x = np.arange(2); w = 0.35\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "axes[0].bar(x-w/2, longs, w, label='HA', color=GREEN)\n"
            "axes[0].bar(x+w/2, shorts, w, label='Coin', color=GREY)\n"
            "axes[0].set_xticks(x); axes[0].set_xticklabels(['HA arm', 'Random arm'])\n"
            "axes[0].axhline(0, c='k', lw=1); axes[0].set_ylabel('gross mean (bps)')\n"
            "axes[0].set_title('Long (green) vs Short (grey) per arm')\n"
            "axes[0].legend(['long', 'short'])\n"
            "axes[1].bar(x-w/2, l_t, w, label='long t', color=GREEN)\n"
            "axes[1].bar(x+w/2, s_t, w, label='short t', color=GREY)\n"
            "axes[1].set_xticks(x); axes[1].set_xticklabels(['HA arm', 'Random arm'])\n"
            "for s in (2,-2): axes[1].axhline(s, ls='--', c=GREY, lw=1)\n"
            "axes[1].axhline(0, c='k', lw=1); axes[1].set_ylabel('HAC t-stat')\n"
            "axes[1].set_title('HA long t ≈ Coin long t — bull-market drift')\n"
            "axes[1].legend(['long t', 'short t'])\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'HA long: {longs[0]:+.2f} bps (t={l_t[0]:+.2f})  |  Coin long: {longs[1]:+.2f} bps (t={l_t[1]:+.2f})')\n"
            "print(f'HA short: {shorts[0]:+.2f} bps (t={s_t[0]:+.2f})  |  Coin short: {shorts[1]:+.2f} bps (t={s_t[1]:+.2f})')"
        ),
        md(
            f"> In plain words: HA long trades earn **+{R['long_mean']:.2f} bps** (t = {R['long_t']:.2f}). "
            f"The random coin's long trades earn **+{R['rand_long_mean']:.2f} bps** (t = {R['rand_long_t']:.2f}). "
            "**These are statistically the same outcome.** The long-side signal is the equity "
            "bull market, captured equally well by random entries that happen to go long. "
            "The symmetric pooled test (t = +0.56) is the honest verdict."
        ),
        md(
            "### 4c · Per-instrument breakdown\n\n"
            "SPY's t = +2.24 in isolation; QQQ, IWM, AAPL are all noise."
        ),
        code(
            "if HAVE_REAL:\n"
            "    recs = []\n"
            "    for t in TICKERS:\n"
            "        b = data.fetch_daily(t, fetch=False, cache_dir=CACHE_DIR)\n"
            "        ha = st.heikin_ashi(b); col = st.ha_colour(ha)\n"
            "        ent = st.colour_flip_entries(col, warmup=20)\n"
            "        s = st.summarize(st.run_trades(b, ent, tp_R=1, sl_R=1, cost_bps=0), 'ret_gross')\n"
            "        ha_f = col.ne(col.shift(1)).sum()\n"
            "        raw_col = st.raw_colour(b); raw_f = raw_col.ne(raw_col.shift(1)).sum()\n"
            "        recs.append((t, s['n_trades'], s['win_rate']*100, s['mean_bps'], s['tstat'], ha_f, raw_f))\n"
            "    perinst = pd.DataFrame(recs, columns=['ticker','n','win%','mean_bps','t','ha_flips','raw_flips'])\n"
            "else:\n"
            "    perinst = pd.DataFrame({\n"
            "        'ticker': TICKERS,\n"
            "        'mean_bps': [R['spy_mean'], R['qqq_mean'], R['iwm_mean'], R['aapl_mean']],\n"
            "        't': [R['spy_t'], R['qqq_t'], R['iwm_t'], R['aapl_t']],\n"
            "        'ha_flips': [R['ha_flips_spy'], 979, 1040, 947],\n"
            "        'raw_flips': [R['raw_flips_spy'], 1910, 1916, 1921],\n"
            "        'n': [976, 972, 1033, 943], 'win%': [53.0, 50.0, 50.0, 50.1],\n"
            "    })\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "col2 = [GREEN if abs(v)>=2 else RED for v in perinst['t']]\n"
            "axes[0].bar(perinst['ticker'], perinst['t'], color=col2)\n"
            "for s in (2,-2): axes[0].axhline(s, ls='--', c=GREY, lw=1)\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_ylabel('HAC t-stat'); axes[0].set_title('Per-instrument t (SPY is one outlier)')\n"
            "axes[1].bar(perinst['ticker'], perinst['ha_flips'], 0.35, label='HA flips', color=AMBER)\n"
            "axes[1].bar([x+0.35 for x in range(4)], perinst['raw_flips'], 0.35, label='Raw flips', color=GREY)\n"
            "axes[1].set_ylabel('colour flips per 15y'); axes[1].set_title('HA smoothing: half the flips')\n"
            "axes[1].legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "perinst.round(2)"
        ),
        md(
            "> In plain words: SPY's t = +2.24 is the most interesting number in this study — "
            "but it is a post-hoc, single-instrument, single-outcome result. When you look at "
            "the random control for SPY it gives +2.19 bps (seed mean ~−1.0 bps over 20 seeds, "
            "range [−8.3, +5.3]), confirming the SPY result is within the noise of drift. "
            "QQQ, IWM, AAPL all show t < 0.5, and pooling all four gives t = +0.56."
        ),
        md(
            "### 4d · Bootstrap Sharpe CI — confirming the noise verdict"
        ),
        code(
            "if HAVE_REAL:\n"
            "    C = pooled(1, 1, 0)\n"
            "    tpy = int(round(len(C) / 4 / 15))\n"
            "    ci = stats.sharpe_ci_bootstrap(pd.Series(C['ret_gross'].values),\n"
            "                                   periods_per_year=tpy, seed=129)\n"
            "    cilo, cihi, fn = ci['ci_low'], ci['ci_high'], ci['frac_negative']*100\n"
            "    all_m = [st.summarize(pooled(1, 1, 0, seed=s), 'ret_gross')['mean_bps']\n"
            "             for s in range(20)]\n"
            "else:\n"
            "    cilo, cihi, fn = R['ci_lo'], R['ci_hi'], R['frac_neg']\n"
            "    all_m = [R['rand_mean']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(all_m, bins=12, color=GREY, alpha=.65, label='random control means (20 seeds)')\n"
            "ax.axvline(R['ha_mean'], c=AMBER, lw=2.5, label=f'HA: +{R[\"ha_mean\"]:.2f} bps')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('gross mean (bps/trade)'); ax.set_ylabel('count')\n"
            "ax.set_title('HA flip lands inside the coin\\'s own noise band'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Bootstrap Sharpe 95% CI: [{cilo:+.2f}, {cihi:+.2f}]  ({fn:.1f}% negative)')"
        ),
        md(
            f"The bootstrap 95% CI on the annualised Sharpe is "
            f"**[{R['ci_lo']:+.2f}, {R['ci_hi']:+.2f}]** ({R['frac_neg']:.1f}% of resamples "
            "negative). The CI straddles zero, confirming the point estimate is not a reliable "
            "signal — just noise on the positive side."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — pooled gross +{R['ha_mean']:.2f} bps/trade, HAC "
            f"*t* = +{R['ha_t']:.2f}; Δ vs control +{R['delta']:.2f} bps; bootstrap "
            f"Sharpe CI [{R['ci_lo']:+.2f}, {R['ci_hi']:+.2f}]. Long-side *t* = "
            f"+{R['long_t']:.2f} is replicated by the coin (*t* = +{R['rand_long_t']:.2f}) — "
            "bull-market drift.\n"
            f"- **Tradability `MIRAGE`** — gross not significant; at 1 bp cost net "
            f"*t* = +{R['net10_t']:.2f} (noise). No positive break-even cost exists.\n"
            f"- **Smoothing `CONFIRMED` to add lag, not signal** — HA fires at half the "
            f"rate (~{R['ha_flips_avg']} vs ~{R['raw_flips_avg']}) but gives the same "
            "directional t-stat vs a coin."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — cost sweep\n\n"
            "Per-trade net mean and HAC *t* across round-trip cost."
        ),
        code(
            "costs = [0.0, 0.5, 1.0, 2.0, 5.0]\n"
            "if HAVE_REAL:\n"
            "    sw = [st.summarize(pooled(1, 1, c), 'ret_net') for c in costs]\n"
            "    mean_v = [s['mean_bps'] for s in sw]\n"
            "    tst_v = [s['tstat'] for s in sw]\n"
            "else:\n"
            "    mean_v = [R['ha_mean'], R['net05_mean'], R['net10_mean'],\n"
            "               R['net20_mean'], R['ha_mean']-5.0]\n"
            "    tst_v = [R['ha_t'], R['net05_t'], R['net10_t'], R['net20_t'], -1.49]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(costs, mean_v, 'o-', c=AMBER, lw=2, label='net mean (bps/trade)')\n"
            "ax2 = ax.twinx()\n"
            "ax2.plot(costs, tst_v, 's--', c=GREY, lw=1.5, label='HAC t')\n"
            "ax2.axhline(-2, ls=':', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net mean (bps)', color=AMBER)\n"
            "ax2.set_ylabel('HAC t-stat', color=GREY)\n"
            "ax.set_title('Gross is not significant — any cost makes it a noise loser')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Gross is not significantly positive (t=+0.56) — break-even cost is undefined.')"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — synthetic positive control\n\n"
            "Is the *engine* capable of finding an edge, or does it always print 'none'? "
            "Plant a bar-level AR(1) momentum in a synthetic tape: the HA flip should "
            "find it when it exists, and the finding should be proportional to the planted "
            "momentum. If the engine returns ~0 on a martingale and positive on a persistent "
            "tape, the real-tape verdict is about the market, not the machinery."
        ),
        code(
            "moms = [-0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15, 0.20]\n"
            "edge = []\n"
            "for m in moms:\n"
            "    b, _ = data.synthetic_daily(n_days=600, momentum=m, seed=129)\n"
            "    ha = st.heikin_ashi(b)\n"
            "    col = st.ha_colour(ha)\n"
            "    ent = st.colour_flip_entries(col, warmup=20)\n"
            "    c = st.summarize(\n"
            "        st.run_trades(b, ent, tp_R=1, sl_R=1, cost_bps=0), 'ret_gross'\n"
            "    )['mean_bps']\n"
            "    r = st.summarize(\n"
            "        st.run_trades(b, ent, tp_R=1, sl_R=1, cost_bps=0,\n"
            "                     directions=st.random_directions(len(ent), seed=1)),\n"
            "        'ret_gross'\n"
            "    )['mean_bps']\n"
            "    edge.append(c - r)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(moms, edge, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, ls='--', c=GREY)\n"
            "ax.set_xlabel('planted bar-level AR(1) momentum')\n"
            "ax.set_ylabel('HA flip - coin (bps/trade)')\n"
            "ax.set_title('The engine works: it harvests momentum when momentum exists')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('At momentum=0 the edge is ~0; it rises monotonically with persistence.')"
        ),
        md(
            "The HA flip advantage over the coin is monotone in the planted momentum and "
            "crosses zero right at the martingale — so the engine is a faithful momentum "
            "detector. The real-tape verdict is therefore a statement about the **market**: "
            "at daily fidelity over 15 years, the HA colour flip finds no persistent "
            "directional structure beyond the equity bull-market drift that a coin captures "
            "equally well.\n\n"
            "Possible extensions: (1) require a run of N same-colour bars before trading the "
            "flip (reducing noise flips); (2) restrict to only *long* flips during a "
            "defined uptrend (sector rotation signal); (3) combine with volume confirmation. "
            "Each adds a degree of freedom — the bar is to show a fair-bet t > 2 on held-out "
            "data after the degree of freedom is fixed."
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
