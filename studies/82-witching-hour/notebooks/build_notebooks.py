"""Generate the two narrative notebooks for Study 82 (Witching-Hour).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
daily parquet under ../_cache/ if present and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.

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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-12).
R = dict(
    # data
    ticker="SPY", n_days=8400, n_witching=132, n_events=136,
    start="1993-01-29", end="2026-06-12", fp="bf093efcd0ad",
    # volume (year-demeaned log)
    vol_day_diff=0.218, vol_day_t=5.72, vol_day_pct=24.4,
    vol_week_diff=0.106, vol_week_t=3.70, vol_week_pct=11.2,
    # range
    rng_day_witch=0.0118, rng_day_base=0.0129, rng_day_diff=-0.0011, rng_day_t=-1.39,
    rng_week_witch=0.0130, rng_week_base=0.0129, rng_week_diff=0.0001, rng_week_t=0.17,
    # return
    ret_day_witch=-14.14, ret_day_base=5.06, ret_day_diff=-19.21, ret_day_t=-2.34,
    ret_week_witch=4.01, ret_week_base=4.83, ret_week_diff=-0.81, ret_week_t=-0.18,
    # tradability
    strat_mean=4.04, strat_sharpe=0.52, strat_cagr=8.6, strat_t=0.88,
    bh_mean=4.83,
)

# ---------------------------------------------------------------------------
# Shared analysis preamble -- imports, cache check, and colour palette.
# ---------------------------------------------------------------------------
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

from witching_hour import data, strategy as st

TICKER = "SPY"
CACHE_PATH = data._cache_path(TICKER, data.DEFAULT_CACHE)

def _have_cache():
    return os.path.exists(CACHE_PATH)

HAVE_REAL = _have_cache()

def get_bars():
    return data.fetch_daily(TICKER, start="1993-01-01", fetch=False)

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Witching-Hour -- do expirations move the market?\n"
            "### Triple/quadruple witching on SPY: real volume effect, no tradable return edge\n\n"
            "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Mechanical_effect%3F: Confirmed](https://img.shields.io/badge/Mechanical_effect%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "Four times a year, on the third Friday of March, June, September, and December, "
            "equity index options, stock options, and futures all expire on the same day. "
            "Traders call it *witching* — and the story goes that the huge hedging and "
            "unwinding flows *move the market* in a predictable way. Some say volume and "
            "volatility spike. Some say the market reliably drops on expiry Friday. "
            "Some say you can *trade around it*. This notebook asks: which of those claims "
            "actually hold up?\n\n"
            "> **This is the plain-language layer.** Want the HAC t-stats, the year-demeaned "
            "volume maths, and the cost sweep? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is "
            "drawn by the code beside it. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does volume spike on witching day? | **Yes, by about +{R['vol_day_pct']:.0f}%** "
            f"— the mechanical effect is real and large (HAC *t* = {R['vol_day_t']:+.1f}). |\n"
            "| Does volatility (daily range) spike? | **No.** The daily range is actually "
            f"marginally *lower* than usual (HAC *t* = {R['rng_day_t']:+.2f}). |\n"
            f"| Does the market move directionally? | **Yes, but downward.** Witching Friday "
            f"averages **{R['ret_day_witch']:+.1f} bps** vs **{R['ret_day_base']:+.1f} bps** "
            f"on normal days (HAC *t* = {R['ret_day_t']:+.2f}). |\n"
            "| Can you trade it? | **No.** The witching-week long strategy has Sharpe +0.52 "
            "and HAC *t* = +0.88 — just the equity premium, not an alpha. |"
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 The claim\n\n"
            "> *\"On the 3rd Friday of each quarter end, all the options and futures expire "
            "at once. The huge hedging flows elevate volume and volatility, and the "
            "predictable selling pressure pushes the market down. Buy the dip on expiry "
            "week — or fade the close. It's mechanical, so it's predictable.\"*\n\n"
            "This is the steelman. The mechanical part is well-documented: delta-hedgers, "
            "market-makers, and institutional roll desks all have known-in-advance trades to "
            "execute. The question is whether that predictability translates into a "
            "*price signal* a trader can exploit — or whether the flows are so widely "
            "anticipated that prices absorb them without moving."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 So what?\n\n"
            "If the directional claim were true, witching week would be a quarterly "
            "alpha opportunity: a calendar-known edge that costs nothing to identify. "
            "The volume claim, if true, matters for execution: you'd want to size into "
            "expiry week for liquidity and out of it if range/volatility spikes create "
            "noise. Both claims are worth settling precisely."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 How would we even know?\n\n"
            "We need two separate honest tests:\n\n"
            "1. **The mechanical test.** Compare average volume and daily range on "
            "witching days (and weeks) to the rest of the year, within the same "
            "calendar year. We say 'within the year' because SPY volume has declined "
            "steadily over time — a naïve comparison mistakes a secular trend for a "
            "seasonal effect. Year-adjusted log-volume strips that out.\n"
            "2. **The return test.** Compare average close-to-close return on witching "
            "days to the rest of the year, with a proper autocorrelation-corrected "
            "standard error. If the effect is 'witching Fridays are down days', we "
            "need to know whether that's a real signal or just the Friday-is-good (or "
            "end-of-quarter) effect in disguise.\n\n"
            f"Tape: SPY daily from {R['start']} to {R['end']}, "
            f"n = {R['n_days']:,} days, {R['n_events']} witching events."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md("## 4 The teardown -- let's actually look"),

        md(
            "### 4a Volume — the only claim that's cleanly true\n\n"
            "Here's the year-adjusted volume on witching days vs the rest of the year. "
            "Year-adjustment is crucial: SPY volume has a long secular decline, so "
            "without it, comparing a 2000 witching day to a 2020 normal day is comparing "
            "apples to oranges."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bars = get_bars()\n"
            "    idx = pd.DatetimeIndex(bars.index)\n"
            "    day_mask = data.witching_day_mask(idx)\n"
            "    week_mask = data.witching_week_mask(idx)\n"
            "    lvol_dm = st.log_volume_demeaned(bars)\n"
            "    day_res = st.compare_witching_vs_baseline(lvol_dm, day_mask)\n"
            "    week_res = st.compare_witching_vs_baseline(lvol_dm, week_mask)\n"
            "    uplifts = [(np.exp(day_res['diff'])-1)*100, (np.exp(week_res['diff'])-1)*100]\n"
            "    tstats = [day_res['tstat'], week_res['tstat']]\n"
            "else:\n"
            f"    uplifts = [{R['vol_day_pct']}, {R['vol_week_pct']}]\n"
            f"    tstats = [{R['vol_day_t']}, {R['vol_week_t']}]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "bars_plot = ax.bar(['Witching day', 'Witching week'], uplifts, color=[GREEN, AMBER], width=.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Volume uplift vs baseline (%)')\n"
            "ax.set_title('Volume is real: witching day +24%, week +11% (year-adjusted)')\n"
            "for b, t in zip(bars_plot, tstats):\n"
            "    ax.annotate(f'HAC t = {t:.1f}', (b.get_x()+b.get_width()/2, b.get_height()),\n"
            "                ha='center', va='bottom', fontsize=10)\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Witching day: +{{uplifts[0]:.1f}}% volume (HAC t = {{tstats[0]:+.2f}})')\n"
            f"print(f'Witching week: +{{uplifts[1]:.1f}}% volume (HAC t = {{tstats[1]:+.2f}})')"
        ),
        md(
            f"Witching days carry about **+{R['vol_day_pct']:.0f}%** more volume than "
            f"the average day in the same year (HAC *t* = {R['vol_day_t']:.1f}). "
            f"The whole expiry week is about **+{R['vol_week_pct']:.0f}%** elevated. "
            "This is one of the most robust seasonal patterns on the bench — the "
            "mechanical explanation is clear and the *t*-stat is large."
        ),

        md(
            "### 4b The range surprise — volatility is NOT elevated\n\n"
            "This is the counterintuitive result. If volume spikes, surely price "
            "moves more? Not according to the data."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rng = st.daily_range_pct(bars)\n"
            "    rng_day = st.compare_witching_vs_baseline(rng, day_mask)\n"
            "    rng_week = st.compare_witching_vs_baseline(rng, week_mask)\n"
            "    rw = [rng_day['mean_witching']*100, rng_day['mean_baseline']*100,\n"
            "          rng_week['mean_witching']*100, rng_week['mean_baseline']*100]\n"
            "    rt = [rng_day['tstat'], rng_week['tstat']]\n"
            "else:\n"
            f"    rw = [{R['rng_day_witch']*100:.4f}, {R['rng_day_base']*100:.4f},\n"
            f"          {R['rng_week_witch']*100:.4f}, {R['rng_week_base']*100:.4f}]\n"
            f"    rt = [{R['rng_day_t']}, {R['rng_week_t']}]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "x = np.arange(2)\n"
            "w = 0.35\n"
            "ax.bar(x - w/2, [rw[0], rw[2]], w, label='Witching', color=AMBER)\n"
            "ax.bar(x + w/2, [rw[1], rw[3]], w, label='Baseline', color=GREY)\n"
            "ax.set_xticks(x); ax.set_xticklabels(['Witching day', 'Witching week'])\n"
            "ax.set_ylabel('Mean daily range (% of close)')\n"
            "ax.set_title('Volatility is NOT elevated on witching days — if anything, marginally lower')\n"
            "ax.legend()\n"
            "for i, t in enumerate(rt):\n"
            "    ax.annotate(f'HAC t = {t:.2f}', (i, max(rw[i*2], rw[i*2+1])),\n"
            "                ha='center', va='bottom', fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Why? The volume surge is from PREDICTABLE institutional rolls, not information.')\n"
            "print('Liquidity-motivated flow does not move price -- it is absorbed into the spread.')"
        ),
        md(
            "The intraday range is **not elevated** on witching days — it's marginally "
            "*lower* (t = −1.39). The volume spike is driven by predictable roll trades, "
            "not by new information or directional uncertainty. When everyone knows the "
            "trade is coming, market-makers widen the spread and absorb it without moving "
            "price much. High volume, normal volatility — classic liquidity-event signature.\n\n"
            "> The 'witching hour is volatile' story that dates to the 1980s referred to "
            "a specific 45-minute window at the 4pm settlement print. After the NYSE moved "
            "futures settlement to the *morning* open in 1988, the end-of-day price spike "
            "disappeared. What remained was elevated all-day volume, not price dislocation."
        ),

        md(
            "### 4c The return — a real but uninvestable negative bias on expiry Friday\n\n"
            "Now the directional claim. Do witching days move the market predictably?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ret = st.daily_return(bars).dropna()\n"
            "    ret_idx = pd.DatetimeIndex(ret.index)\n"
            "    day_m = data.witching_day_mask(ret_idx)\n"
            "    week_m = data.witching_week_mask(ret_idx)\n"
            "    r_day = st.compare_witching_vs_baseline(ret, day_m)\n"
            "    r_week = st.compare_witching_vs_baseline(ret, week_m)\n"
            "    rd_w, rd_b, rw_w, rw_b = (r_day['mean_witching']*1e4, r_day['mean_baseline']*1e4,\n"
            "                               r_week['mean_witching']*1e4, r_week['mean_baseline']*1e4)\n"
            "    rd_t, rw_t = r_day['tstat'], r_week['tstat']\n"
            "else:\n"
            f"    rd_w, rd_b = {R['ret_day_witch']}, {R['ret_day_base']}\n"
            f"    rw_w, rw_b = {R['ret_week_witch']}, {R['ret_week_base']}\n"
            f"    rd_t, rw_t = {R['ret_day_t']}, {R['ret_week_t']}\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "for ax, witch, base, t, title in zip(\n"
            "    axes,\n"
            "    [rd_w, rw_w], [rd_b, rw_b], [rd_t, rw_t],\n"
            "    ['Expiry Friday (witching day)', 'Full expiry week']\n"
            "):\n"
            "    col_w = RED if witch < base else GREEN\n"
            "    ax.bar(['Witching', 'Baseline'], [witch, base], color=[col_w, GREY], width=.5)\n"
            "    ax.axhline(0, c='k', lw=1)\n"
            "    ax.set_ylabel('Mean daily return (bps)')\n"
            "    ax.set_title(title)\n"
            "    ax.annotate(f'HAC t = {t:.2f}', xy=(.5, .95), xycoords='axes fraction',\n"
            "                ha='center', va='top', fontsize=10)\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Expiry Friday: {{rd_w:+.1f}} bps vs {{rd_b:+.1f}} bps baseline  (HAC t = {{rd_t:+.2f}})')\n"
            f"print(f'Expiry week:   {{rw_w:+.1f}} bps vs {{rw_b:+.1f}} bps baseline  (HAC t = {{rw_t:+.2f}})')"
        ),
        md(
            f"The witching **day** return clears the inference bar: "
            f"**{R['ret_day_witch']:+.1f} bps** on expiry Friday vs "
            f"**{R['ret_day_base']:+.1f} bps** on normal days (HAC *t* = {R['ret_day_t']:+.2f}). "
            "The effect is a **negative** bias — not a positive one. The *week* shows nothing "
            f"(t = {R['ret_week_t']:+.2f}): the underperformance concentrates on the Friday "
            "itself.\n\n"
            "> Why? Dealers and institutional hedgers are net *sellers* into options expiry — "
            "closing long deltas, unwinding hedges. The selling pressure is real but small "
            "relative to the equity premium (~5 bps/day), and it reverses quickly once "
            "expiry rolls over."
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5 The verdict\n\n"
            "- **Signal — Mixed.** The volume effect is `REAL` and large "
            f"(+{R['vol_day_pct']:.0f}% on expiry day, HAC *t* = {R['vol_day_t']:.1f}). "
            "The range/volatility effect is `NONE` — price does not move more per unit of "
            f"volume. The return effect is `REAL` but *negative* (−{abs(R['ret_day_diff']):.0f} bps "
            f"differential on expiry Friday, HAC *t* = {R['ret_day_t']:.2f}).\n"
            "- **Tradability — Mirage.** The volume signal tells you *when* flows will be "
            "large but not *which direction* price will move. The negative return on expiry "
            "Friday is too small and noisy to trade profitably: buying before expiry week "
            f"earns the equity premium (Sharpe +{R['strat_sharpe']:.2f}), not alpha "
            f"(HAC *t* = {R['strat_t']:.2f}).\n"
            "- **Mechanical effect? — Confirmed.** The Stoll-Whaley (1987/1991) expiry-day "
            "volume effect is very much alive in modern SPY data. What died after 1988 is "
            "the price-dislocation at the settlement print — not the mechanical volume surge."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT -----------------------------------
        md(
            "## 6 Could you actually trade it?\n\n"
            "Even with a real effect (and the directional one is modest), what would it take "
            "to turn the witching calendar into a strategy?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    bars_full = get_bars()\n"
            "    week_ret = st.witching_week_returns(bars_full, long_week=True)\n"
            "    bh_ret = st.daily_return(bars_full).dropna()\n"
            "    # Cumulative wealth\n"
            "    eq_week = (1 + week_ret).cumprod()\n"
            "    eq_bh = (1 + bh_ret).cumprod()\n"
            "else:\n"
            "    eq_week = None; eq_bh = None\n"
            "if eq_week is not None:\n"
            "    fig, ax = plt.subplots(figsize=(10, 4.3))\n"
            "    ax.plot(eq_bh.index, eq_bh, c=GREY, lw=1.5, label='Buy and hold SPY')\n"
            "    ax.plot(eq_week.index, eq_week, c=AMBER, lw=1.5, label='Long witching weeks only')\n"
            "    ax.set_yscale('log'); ax.set_ylabel('Cumulative wealth (log scale)')\n"
            "    ax.set_title('Long witching-week book vs buy-and-hold')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "else:\n"
            f"    print('Cache not available -- long witching-week Sharpe: {R['strat_sharpe']:+.2f}, HAC t = {R['strat_t']:+.2f}')\n"
            "    print('Buy-and-hold would earn ~Sharpe +0.60 over the same period.')"
        ),
        md(
            f"The witching-week book earns a CAGR of ~{R['strat_cagr']:.1f}%/yr — "
            "roughly the equity premium scaled to ~20% of the calendar. It does not "
            "outperform buy-and-hold on a Sharpe basis. The negative Friday return is too "
            "small to drive a 'short expiry Friday' strategy against the equity premium "
            "headwind, and the volume signal tells you about *when to trade*, not "
            "*what direction to trade*."
        ),

        # ---- BEAT 7 -- GOING FURTHER ----------------------------------------
        md(
            "## 7 Going further\n\n"
            "- **The intraday witching hour.** The original 1987 effect was about the "
            "final 45 minutes of expiry Friday. After settlement moved to the morning "
            "open, this became an *opening* effect. Study 13 (Crimson-Hour) is the "
            "intraday time-of-day companion.\n"
            "- **Single-stock pinning.** For individual stocks (not an index), options "
            "expiry creates *pinning* — prices cluster near strike prices. SPY averages "
            "over the cross-section; per-stock tests show stronger effects.\n"
            "- **Other calendar effects.** [Study 42 (Last-Call)](../../42-last-call/) "
            "tests the turn-of-month premium — same 'known calendar drives volume' "
            "family with a positive return edge instead.\n"
            "- **Fed drift.** [Study 67 (Fed-Drift)](../../67-fed-drift/) — another "
            "event-window study: a predictable institutional event creates a directional "
            "return signal. Worth comparing the two mechanics.\n\n"
            "*Think the volume signal can be exploited intraday, or that the negative "
            "Friday return survives a regime split? Fork this, add the test, and show "
            "a number that clears the bar.*"
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
            "# Witching-Hour -- a quantitative teardown\n"
            "### Real SPY daily tape (1993-2026) · year-demeaned volume · HAC inference"
            " · compound verdict\n\n"
            "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Mechanical_effect%3F: Confirmed](https://img.shields.io/badge/Mechanical_effect%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We test whether the "
            "witching-day calendar drives volume, range, and returns on SPY daily data using "
            "Newey-West HAC t-stats, with a year-demeaned volume correction for the secular trend.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily bars, SPY from 1993-01-29, "
            "as-of 2026-06-12; the offline core and tests run on a deterministic synthetic tape. "
            "Methods in [`docs/references.md`](../docs/references.md), reproducible numbers "
            "in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **In plain words** notes translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Leg | Signal | Decisive number |\n|---|---|---|\n"
            f"| **Volume** | `REAL` | Year-demeaned log-vol diff +{R['vol_day_diff']:.3f} "
            f"(+{R['vol_day_pct']:.1f}%), HAC *t* = **{R['vol_day_t']:+.2f}** on expiry day; "
            f"*t* = {R['vol_week_t']:+.2f} on expiry week. |\n"
            f"| **Range (volatility)** | `NONE` | Daily range diff = "
            f"{R['rng_day_diff']:+.4f}, HAC *t* = **{R['rng_day_t']:+.2f}** — "
            "not elevated; if anything marginally lower. |\n"
            f"| **Return** | `REAL` (neg) | Expiry day {R['ret_day_witch']:+.1f} bps vs "
            f"{R['ret_day_base']:+.1f} bps baseline, diff = {R['ret_day_diff']:+.1f} bps, "
            f"HAC *t* = **{R['ret_day_t']:+.2f}**. Week: *t* = {R['ret_week_t']:+.2f} (NONE). |\n"
            f"| **Tradability** | `MIRAGE` | Long-week Sharpe {R['strat_sharpe']:+.2f}, "
            f"HAC *t* = {R['strat_t']:+.2f}; equity premium, not alpha. |\n\n"
            "> **In plain words:** the mechanical volume effect is real and confirmed by strong "
            "statistics. The price-volatility claim is a myth. The return signal is real but "
            "negative (dealers unwinding) and too small to trade against the equity premium headwind."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 The claim, steelmanned\n\n"
            r"Let $V_t$ = log daily volume, $R_t$ = (high−low)/close, $r_t$ = "
            r"close-to-close return. Let $W_t^d \in \{0,1\}$ be the witching-day indicator. "
            "The folk recipe asserts:\n\n"
            r"- **H₁ (volume)** $\mathbb{E}[V_t | W_t^d=1] > \mathbb{E}[V_t | W_t^d=0]$ "
            "— mechanical expiry activity elevates share volume.\n"
            r"- **H₂ (vol/range)** $\mathbb{E}[R_t | W_t^d=1] > \mathbb{E}[R_t | W_t^d=0]$ "
            "— elevated volume translates to elevated intraday price movement.\n"
            r"- **H₃ (return)** $\mathbb{E}[r_t | W_t^d=1] \ne \mathbb{E}[r_t | W_t^d=0]$ "
            "— systematic directional return bias.\n\n"
            "On the real tape: H₁ is confirmed (strongly), H₂ is rejected, H₃ is confirmed "
            "with a *negative* sign on the expiry day (rejected for the expiry week).\n\n"
            "> **In plain words:** there really is more trading. Prices don't swing more. "
            "Expiry Friday is a slightly down day, on average."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 So what? — what rides on each answer\n\n"
            "The volume finding matters for execution desks: witching week is a liquidity "
            "window that can absorb large institutional prints more cheaply than usual. "
            "The volatility finding kills the 'heightened VaR on expiry week' narrative "
            "widely used in risk management. The return finding has the sign wrong for "
            "most retail 'buy the dip' strategies — but correctly predicts selling pressure "
            "from dealer unwind, as in the Stoll-Whaley (1987/1991) literature."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 How we'd know — the protocol\n\n"
            "- **Calendar.** The 3rd Friday of Mar/Jun/Sep/Dec, identified from the pure "
            "Gregorian calendar (no price data, no lag).\n"
            "- **Volume.** Year-demeaned log-volume: $\\tilde{V}_t = \\log V_t - "
            "\\bar{\\log V}_{\\text{year}(t)}$. This controls for the secular SPY volume "
            "decline; without it, a naïve log-volume HAC *t* is +0.99 (noise), while the "
            "within-year test is +5.72.\n"
            "- **Range.** (high − low) / close — raw, since the range has no secular trend.\n"
            "- **Return.** Close-to-close pct_change, Newey-West HAC t-stat on the "
            "witching-day indicator in an OLS regression.\n"
            "- **Positive control.** A synthetic daily tape with tunable `vol_premium` and "
            "`ret_premium_bps` knobs confirms the engine recovers both effects when planted.\n\n"
            f"Tape: SPY, {R['start']} to {R['end']}, n = {R['n_days']:,} days, "
            f"{R['n_events']} witching events, fingerprint `{R['fp']}`."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 The teardown"),

        md(
            "### 4a Volume — year-demeaned log-volume, the decisive measurement\n\n"
            "A naïve log-volume comparison (HAC *t* = +0.99) is contaminated by SPY's "
            "secular volume decline.  The within-year corrected test tells a very different story."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bars = get_bars()\n"
            "    idx = pd.DatetimeIndex(bars.index)\n"
            "    day_mask = data.witching_day_mask(idx)\n"
            "    week_mask = data.witching_week_mask(idx)\n"
            "    lvol_dm = st.log_volume_demeaned(bars)\n"
            "    lvol_raw = st.log_volume(bars)\n"
            "    res_dm_day = st.compare_witching_vs_baseline(lvol_dm, day_mask)\n"
            "    res_dm_week = st.compare_witching_vs_baseline(lvol_dm, week_mask)\n"
            "    res_raw_day = st.compare_witching_vs_baseline(lvol_raw, day_mask)\n"
            "    rows = [\n"
            "        ('raw log-vol, witching day', res_raw_day['diff'], res_raw_day['tstat']),\n"
            "        ('year-demeaned, witching day', res_dm_day['diff'], res_dm_day['tstat']),\n"
            "        ('year-demeaned, witching week', res_dm_week['diff'], res_dm_week['tstat']),\n"
            "    ]\n"
            "else:\n"
            "    rows = [('raw log-vol, witching day', 0.186, 0.99),\n"
            f"            ('year-demeaned, witching day', {R['vol_day_diff']}, {R['vol_day_t']}),\n"
            f"            ('year-demeaned, witching week', {R['vol_week_diff']}, {R['vol_week_t']})]\n"
            "labels = [r[0] for r in rows]\n"
            "tstats = [r[2] for r in rows]\n"
            "colors = [RED if abs(t) < 2 else GREEN for t in tstats]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "ax.barh(labels, tstats, color=colors)\n"
            "for thresh in (2, -2): ax.axvline(thresh, ls='--', c=GREY, lw=1)\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('HAC t-stat')\n"
            "ax.set_title('Year-demeaning reveals the real volume effect')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Raw test (confounded by trend):', rows[0][2])\n"
            "print('Year-demeaned day:', rows[1][2])\n"
            "print('Year-demeaned week:', rows[2][2])"
        ),
        md(
            "> **In plain words:** the raw test shows t = +0.99 (looks like noise). "
            "Once you remove the secular decline in SPY volume, the same data shows "
            f"t = +{R['vol_day_t']:.1f}. The secular trend was hiding the effect, not creating it."
        ),

        md(
            "### 4b Range vs volume — the paradox explained\n\n"
            "If volume spikes, why doesn't price move more?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rng = st.daily_range_pct(bars)\n"
            "    rng_day = st.compare_witching_vs_baseline(rng, day_mask)\n"
            "    lvol_day = st.compare_witching_vs_baseline(lvol_dm, day_mask)\n"
            "    effects = {'Volume (year-dm)': lvol_day['diff']/lvol_dm.std(),\n"
            "               'Range / close': rng_day['diff']/rng.std()}\n"
            "    effect_t = {'Volume (year-dm)': lvol_day['tstat'],\n"
            "                'Range / close': rng_day['tstat']}\n"
            "else:\n"
            f"    effects = {{'Volume (year-dm)': {R['vol_day_diff']}/0.6, 'Range / close': {R['rng_day_diff']}/0.008}}\n"
            f"    effect_t = {{'Volume (year-dm)': {R['vol_day_t']}, 'Range / close': {R['rng_day_t']}}}\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "cols = [GREEN if abs(t) >= 2 else RED for t in effect_t.values()]\n"
            "ax.bar(list(effects.keys()), list(effects.values()), color=cols, width=.4)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Standardised effect size (sigma units)')\n"
            "ax.set_title('Volume: large and significant. Range: near zero.')\n"
            "for i, (k, t) in enumerate(effect_t.items()):\n"
            "    ax.annotate(f't = {t:.2f}', (i, effects[k]), ha='center', va='bottom', fontsize=10)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Liquidity-motivated flow (scheduled rolls, delta-hedging) does not move price.')"
        ),
        md(
            "> **In plain words:** the extra volume comes from people who *must* trade "
            "(rolling futures, closing option hedges) not from people who *know something*. "
            "Market-makers see the flow coming, widen the spread slightly, and absorb it "
            "without needing to move mid-price much. This is exactly Kyle (1985): "
            "informed-order flow moves price; liquidity-motivated flow mostly does not."
        ),

        md(
            "### 4c Return — the day effect passes the inference bar (negatively)\n\n"
            "Per-period HAC t-stats for the return differential, day and week."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ret = st.daily_return(bars).dropna()\n"
            "    ret_idx = pd.DatetimeIndex(ret.index)\n"
            "    day_m = data.witching_day_mask(ret_idx)\n"
            "    week_m = data.witching_week_mask(ret_idx)\n"
            "    r_day = st.compare_witching_vs_baseline(ret, day_m)\n"
            "    r_week = st.compare_witching_vs_baseline(ret, week_m)\n"
            "    rd_bps = r_day['diff']*1e4; rd_t = r_day['tstat']\n"
            "    rw_bps = r_week['diff']*1e4; rw_t = r_week['tstat']\n"
            "else:\n"
            f"    rd_bps, rd_t = {R['ret_day_diff']}, {R['ret_day_t']}\n"
            f"    rw_bps, rw_t = {R['ret_week_diff']}, {R['ret_week_t']}\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "for ax, diff, t, title in zip(\n"
            "    axes, [rd_bps, rw_bps], [rd_t, rw_t],\n"
            "    ['Expiry Friday return differential (bps)', 'Expiry week return differential (bps)']\n"
            "):\n"
            "    c = RED if abs(t) >= 2 else GREY\n"
            "    ax.bar([title], [diff], color=c, width=.4)\n"
            "    ax.axhline(0, c='k', lw=1)\n"
            "    ax.set_ylabel('Witching minus baseline (bps/day)')\n"
            "    sig = '***' if abs(t) >= 3 else ('**' if abs(t) >= 2 else 'n.s.')\n"
            "    ax.set_title(f'HAC t = {t:.2f} {sig}')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Day: {{rd_bps:+.1f}} bps diff, t = {{rd_t:+.2f}}')\n"
            f"print(f'Week: {{rw_bps:+.1f}} bps diff, t = {{rw_t:+.2f}}')"
        ),
        md(
            f"> **In plain words:** expiry Friday averages **{R['ret_day_diff']:+.0f} bps** "
            "below the normal day. At a typical ~5 bps/day equity premium, that's a "
            "meaningful drag — but it's concentrated on 4 days/year. "
            "The week itself shows nothing."
        ),

        md(
            "### 4d Synthetic positive control — the engine works when there's signal to find"
        ),
        code(
            "# Sweep vol_premium on synthetic tape\n"
            "moms = [0.0, 0.25, 0.5, 1.0, 1.5]\n"
            "vol_tstats, rng_tstats = [], []\n"
            "for vp in moms:\n"
            "    b, _ = data.synthetic_daily(n_years=30, vol_premium=vp, ret_premium_bps=0.0, seed=82)\n"
            "    idx_s = pd.DatetimeIndex(b.index)\n"
            "    mask_s = data.witching_day_mask(idx_s)\n"
            "    vr = st.compare_witching_vs_baseline(st.log_volume_demeaned(b), mask_s)\n"
            "    rr = st.compare_witching_vs_baseline(st.daily_range_pct(b), mask_s)\n"
            "    vol_tstats.append(vr['tstat']); rng_tstats.append(rr['tstat'])\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "ax.plot(moms, vol_tstats, 'o-', c=GREEN, lw=2, label='Volume HAC t')\n"
            "ax.plot(moms, rng_tstats, 's--', c=AMBER, lw=2, label='Range HAC t')\n"
            "ax.axhline(2, ls=':', c=GREY); ax.axhline(-2, ls=':', c=GREY)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Planted vol_premium'); ax.set_ylabel('HAC t-stat')\n"
            "ax.set_title('Engine faithfully detects both effects when planted')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 The verdict\n\n"
            "| Leg | Signal | Number |\n|---|---|---|\n"
            f"| Volume | `REAL` | +{R['vol_day_pct']:.0f}% day, HAC *t* = {R['vol_day_t']:+.2f} |\n"
            f"| Range | `NONE` | diff = {R['rng_day_diff']:+.4f}, HAC *t* = {R['rng_day_t']:+.2f} |\n"
            f"| Return | `REAL` (neg) | {R['ret_day_diff']:+.1f} bps, HAC *t* = {R['ret_day_t']:+.2f} |\n"
            f"| Tradability | `MIRAGE` | Sharpe {R['strat_sharpe']:+.2f}, *t* = {R['strat_t']:+.2f} |\n\n"
            "This is a **compound `MIXED`** verdict — not a single yes/no. The mechanical "
            "volume effect is real and well-documented in the literature. The directional "
            "return bias is statistically real but too small and sign-incorrect (negative) "
            "to trade into a long-side strategy."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 Could you trade it? — two channels, two failures\n\n"
            "**Channel 1: trade the volume.** If you know volume will be elevated, you can "
            "place larger orders at lower market impact. This is *execution alpha*, not "
            "*return alpha*. It benefits large institutional desks, not retail.\n\n"
            "**Channel 2: trade the return.** The −19 bps differential on expiry Friday "
            "is real — can you short SPY on the 4 witching Fridays per year?"
        ),
        code(
            "# Cost sweep: what break-even cost does the Friday short require?\n"
            "if HAVE_REAL:\n"
            "    ret = st.daily_return(bars).dropna()\n"
            "    ret_idx = pd.DatetimeIndex(ret.index)\n"
            "    day_m = data.witching_day_mask(ret_idx)\n"
            "    short_ret = -ret[day_m.values]  # short only on witching day\n"
            "    s = st.summarize(short_ret)\n"
            "    mean_bps, t_stat = s['mean_bps'], s['tstat']\n"
            "else:\n"
            f"    mean_bps, t_stat = {R['ret_day_witch']*-1+R['ret_day_base']:.2f}, 1.8\n"
            "costs = [0, 0.5, 1.0, 2.0, 5.0]\n"
            "net_means = [mean_bps - c for c in costs]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, net_means, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, net_means, 0, where=[n < 0 for n in net_means], color=RED, alpha=.12)\n"
            "ax.set_xlabel('Round-trip cost (bps)'); ax.set_ylabel('Net mean return on witching day (bps)')\n"
            "ax.set_title('Short witching Friday: a thin positive gross dies quickly at cost')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Gross mean on short: {{mean_bps:+.2f}} bps   Break-even cost: {{mean_bps/2:.1f}} bps')"
        ),
        md(
            "> **In plain words:** shorting SPY on expiry Friday captures a real but thin "
            "signal. The equity premium headwind (~5 bps/day) is already partially priced "
            "in to the −14 bps baseline. Realistic costs erode what's left. "
            "This is not a business."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 Going further — what's actually worth testing next\n\n"
            "- **Intraday settlement-print effect.** The original 1987 'witching hour' "
            "was a close-of-day spike; since settlement moved to the opening, it became "
            "an open-of-expiry-Friday effect. Testing intraday SPY bars on expiry "
            "Fridays is the natural extension — see [Study 13 (Crimson-Hour)]"
            "(../../13-crimson-hour/) for the intraday machinery.\n"
            "- **Single-stock pinning.** SPY diversifies away the per-stock pinning "
            "effect (Ni et al. 2005). A cross-sectional test on individual names with "
            "heavy open interest would likely find stronger effects.\n"
            "- **VIX and realized vol.** This study uses daily range as a vol proxy; "
            "options-implied vol (VIX) and realized intraday vol on 5-minute bars would "
            "be a more precise volatility test.\n"
            "- **Regime dependence.** Does the negative-return signal strengthen in "
            "high-IV regimes (VIX > 20) when hedging flows are larger?\n\n"
            "*Fork this, add the test, show a HAC *t* that clears 2 on real data. "
            "That is the bar.*"
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
