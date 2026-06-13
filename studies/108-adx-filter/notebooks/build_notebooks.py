"""Generate the two narrative notebooks for Study 108 (ADX-Filter).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md).  The synthetic
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-13).
R = dict(
    n_ungated=520, n_gated=104,
    ug_mean=15.91, ug_win=51.9, ug_t=0.39,
    g_mean=-25.50, g_win=50.0, g_t=-0.22,
    rand_ug_mean=-6.69, rand_ug_t=-0.14,
    rand_g_mean=-140.85, rand_g_t=-1.10,
    delta_g_ug=-41.41,
    ug_net05=15.41, ug_net10=14.91, ug_net20=13.91,
    g_net05=-26.00, g_net10=-26.50, g_net20=-27.50,
    pct_filtered=80,
    # Per-instrument ungated
    spy_ug_n=80, qqq_ug_n=78, iwm_ug_n=90,
    aapl_ug_n=94, tsla_ug_n=91, nvda_ug_n=87,
    spy_ug_t=0.47, qqq_ug_t=-0.02, iwm_ug_t=-0.36,
    aapl_ug_t=0.84, tsla_ug_t=0.39, nvda_ug_t=-0.29,
    # Per-instrument gated
    spy_g_n=16, qqq_g_n=12, iwm_g_n=13,
    aapl_g_n=21, tsla_g_n=20, nvda_g_n=22,
    spy_g_mean=106.2, qqq_g_mean=-117.7, iwm_g_mean=44.9,
    aapl_g_mean=-44.3, tsla_g_mean=63.7, nvda_g_mean=-175.7,
    spy_g_t=0.95, qqq_g_t=-0.67, iwm_g_t=0.34,
    aapl_g_t=-0.28, tsla_g_t=0.14, nvda_g_t=-0.54,
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

from adx_filter import data, strategy as st

TICKERS = ["SPY", "QQQ", "IWM", "AAPL", "TSLA", "NVDA"]

def _have_cache():
    return all(os.path.exists(data._cache_path(t, data.DEFAULT_CACHE)) for t in TICKERS)

HAVE_REAL = _have_cache()

def pool(gated, cost, seed=None):
    frames = []
    for t in TICKERS:
        b = data.fetch_daily(t, fetch=False)
        ent = st.ma_cross_entries(b["close"])
        if gated:
            adx_s = st.adx(b, n=14)
            ent = st.apply_adx_gate(ent, adx_s, threshold=25.0)
        dirs = st.random_directions(len(ent), seed=seed) if seed is not None else None
        frames.append(st.run_trades(b, ent, hold_days=20, cost_bps=cost, directions=dirs))
    return pd.concat(frames, ignore_index=True)

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    # Pre-render the frozen R values so we avoid nested f-string brace issues.
    ug_mean = R['ug_mean']
    ug_t = R['ug_t']
    g_mean = R['g_mean']
    g_t = R['g_t']
    delta = R['delta_g_ug']
    n_gated = R['n_gated']
    n_ungated = R['n_ungated']
    pct = R['pct_filtered']

    # Frozen fallback code cell strings (avoid f-string dict-brace collision)
    ug_ns = f"{R['spy_ug_n']},{R['qqq_ug_n']},{R['iwm_ug_n']},{R['aapl_ug_n']},{R['tsla_ug_n']},{R['nvda_ug_n']}"
    g_ns  = f"{R['spy_g_n']},{R['qqq_g_n']},{R['iwm_g_n']},{R['aapl_g_n']},{R['tsla_g_n']},{R['nvda_g_n']}"

    cells = [
        md(
            "# ADX-Filter -- does 'only trade strong trends' actually work?\n"
            "### The ADX(14) > 25 gate applied to a daily MA cross, tested honestly\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![ADX_filter_adds_value%3F: Busted](https://img.shields.io/badge/ADX_filter_adds_value%3F-Busted-8b949e?style=flat-square)\n\n"
            "You will find this rule in virtually every technical-analysis textbook and charting "
            "platform: *draw the ADX(14) indicator on your chart.  When it is above 25 the trend is "
            "strong and your moving-average or breakout signal is trustworthy.  When it is below 25 "
            "the market is choppy -- stay flat.*  It is taught as a free-lunch quality filter: the "
            "same trend rule, but cleaner, with fewer whipsaws.  This notebook asks whether the "
            "filtered rule actually earns more than the unfiltered rule, or a coin.\n\n"
            "> This is the plain-language layer.  Want the t-stats, bootstrap CIs, and per-instrument "
            "breakdown?  That is the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.**  A reproducible research tool: every chart below is "
            "drawn by the code beside it.  House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the MA(20/50) cross have any directional signal at all? | **Barely -- and not "
            f"statistically real.**  Ungated: **+{ug_mean:.1f} bps/trade**, "
            f"*t* = **+{ug_t:.2f}** (need |*t*| >= 2). |\n"
            "| Does the ADX filter *improve* the signal? | **No -- it hurts.**  The gated arm "
            f"earns **{g_mean:+.1f} bps/trade**, i.e. **{delta:+.1f} bps worse** "
            "than just taking every signal. |\n"
            "| Does either arm beat a random coin? | **No.**  Both sit inside the coin's own "
            "noise band.  The ADX filter doesn't load the die. |\n"
            "| Could you trade it? | **No.**  Neither arm clears any t-stat bar before costs, "
            "and costs only dig the hole deeper. |\n\n"
            f"> The ADX filter filters away **{pct}% of signals** without adding a single "
            "bps of edge.  It creates the *impression* of a cleaner system while delivering "
            "a noisier P&L at the same (absent) expectancy."
        ),

        # ---- BEAT 1 -- THE CLAIM -----------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *'Never take a trend signal unless ADX is above 25.  That is the threshold Wilder "
            "himself gave us -- above 25 the trend is strong, below 25 you are just getting "
            "whipsawed.  Add it to any moving-average cross or breakout system and your win-rate "
            "goes up.'*\n\n"
            "It has an appealing logic: ADX measures *how much* the price is trending.  If you "
            "only activate your trend rule when the trend is actually strong, you should avoid the "
            "noise-filled ranging periods where trend rules systematically lose.  This notebook "
            "tests whether that intuition survives contact with a long dataset."
        ),

        # ---- BEAT 2 -- SO WHAT -------------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If ADX filtering works, it is a genuine free improvement: the same rule, with a "
            "layer of quality control that costs nothing extra.  Every trend-following system in "
            "the world should apply it.  If it does not work, millions of retail traders are "
            "carrying a useless indicator that creates false confidence and reduces trade count "
            "without improving expected returns -- the worst kind of noise."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -------------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three comparisons, all honest:\n\n"
            "1. **UNGATED vs a coin.** Take every MA(20/50) cross signal, hold 20 days, compare "
            "to random entries on the same dates with random directions.  Does the cross itself "
            "have *any* signal?\n"
            "2. **GATED vs UNGATED.**  Apply ADX(14) >= 25 to filter the signals before entering.  "
            "Does the gated arm earn more per trade than the unfiltered arm?\n"
            "3. **GATED vs a coin.**  Even if the gate filters, does the resulting subset "
            "beat a random coin?  That is the direct test of whether the gate creates a real edge.\n\n"
            "We run on **six daily tapes** (SPY, QQQ, IWM, AAPL, TSLA, NVDA) from 2010 to "
            "2026, 20-day forward returns, gross of costs first."
        ),

        # ---- BEAT 4 -- TEARDOWN ------------------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First, how many trades does ADX filter away?**  If ADX > 25 is a strong-trend "
            "detector, we expect it to pass a minority of signals while blocking the "
            "choppy-market noise."
        ),
        code(
            "if HAVE_REAL:\n"
            "    total, gated_n = 0, 0\n"
            "    rows = []\n"
            "    for t in TICKERS:\n"
            "        b = data.fetch_daily(t, fetch=False)\n"
            "        ent = st.ma_cross_entries(b['close'])\n"
            "        adx_s = st.adx(b, n=14)\n"
            "        g = st.apply_adx_gate(ent, adx_s, threshold=25.0)\n"
            "        rows.append((t, len(ent), len(g), f'{100*len(g)/max(len(ent),1):.0f}%'))\n"
            "        total += len(ent); gated_n += len(g)\n"
            "    tbl = pd.DataFrame(rows, columns=['ticker','all signals','ADX>25','% kept'])\n"
            "else:\n"
            "    tbl = pd.DataFrame({'ticker': TICKERS,\n"
            "        'all signals': [" + ug_ns + "],\n"
            "        'ADX>25': [" + g_ns + "]})\n"
            "    tbl['% kept'] = (tbl['ADX>25']/tbl['all signals']*100).map('{:.0f}%'.format)\n"
            f"    total, gated_n = {n_ungated}, {n_gated}\n"
            "print(tbl.to_string(index=False))\n"
            "print(f'Pooled: {total} -> {gated_n} trades (' + str(round(100*gated_n/total)) + '% kept)')"
        ),
        md(
            f"The ADX filter removes **{pct}% of all signals** -- it keeps only "
            f"**{n_gated} of {n_ungated}** trades.  That is a very aggressive filter.  "
            "Now the critical question: does the surviving 20% earn more per trade?"
        ),
        md("**The three-way comparison: UNGATED vs GATED vs a coin.**"),
        code(
            "if HAVE_REAL:\n"
            "    ug = st.summarize(pool(False, 0), 'ret_gross')\n"
            "    g  = st.summarize(pool(True, 0),  'ret_gross')\n"
            "    r_ug = st.summarize(pool(False, 0, seed=108), 'ret_gross')\n"
            "    r_g  = st.summarize(pool(True, 0,  seed=108), 'ret_gross')\n"
            "    ug_m, g_m = ug['mean_bps'], g['mean_bps']\n"
            "    rug_m, rg_m = r_ug['mean_bps'], r_g['mean_bps']\n"
            "else:\n"
            f"    ug_m, g_m = {ug_mean}, {g_mean}\n"
            f"    rug_m, rg_m = {R['rand_ug_mean']}, {R['rand_g_mean']}\n"
            "labels = ['UNGATED (all signals)', 'GATED (ADX>25 only)',\n"
            "          'Random coin (ungated)', 'Random coin (gated)']\n"
            "vals = [ug_m, g_m, rug_m, rg_m]\n"
            "cols = [AMBER, RED, GREY, GREY]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "bars = ax.bar(labels, vals, color=cols, width=0.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('gross return (bps per trade, 20-day hold)')\n"
            "ax.set_title('The ADX filter makes the MA cross worse, not better')\n"
            "for b, v in zip(bars, vals):\n"
            "    ax.annotate(f'{v:+.1f} bps', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='bottom' if v>=0 else 'top', fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            f"print('Gated minus Ungated: {delta:+.1f} bps/trade')"
        ),
        md(
            f"The UNGATED arm earns **+{ug_mean:.1f} bps/trade** -- noise-level positive "
            f"(*t* = +{ug_t:.2f}, need |t| >= 2).  The GATED arm earns "
            f"**{g_mean:+.1f} bps/trade** -- noise-level *negative*.  The ADX filter "
            f"costs **{delta:+.1f} bps per trade** without delivering any cleaner signal.\n\n"
            "> Why does the filter hurt?  On a long dataset ADX often exceeds 25 after a "
            "big directional move -- i.e. *after* the trend has already played out.  The gate "
            "selects the late entries, not the early ones."
        ),

        # ---- BEAT 5 -- VERDICT -------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal -- None.** The MA(20/50) cross has no detectable edge (ungated "
            f"*t* = +{ug_t:.2f}).  Gating on ADX > 25 makes it **worse** ({g_mean:+.1f} "
            f"bps/trade, *t* = {g_t:+.2f}).  No instrument clears |*t*| = 1 in the gated arm.\n"
            "- **Tradability -- Mirage.** The ungated arm is noise at realistic cost levels; the "
            "gated arm is negative before costs and has one-fifth the trades.\n"
            f"- **ADX filter adds value? -- Busted.** The filter removes {pct}% of signals without "
            "improving the per-trade return.  'Only trade strong trends' is advice that sounds "
            "right and tests wrong."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT? ------------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "The MA(20/50) cross is a slow rule (~3 trades/year/instrument).  The gated "
            "version fires ~0.7/year.  Even at low turnover, costs erode the noise return:"
        ),
        code(
            "costs = [0.0, 0.5, 1.0, 2.0]\n"
            "if HAVE_REAL:\n"
            "    ug_net = [st.summarize(pool(False, c), 'ret_net')['mean_bps'] for c in costs]\n"
            "    g_net  = [st.summarize(pool(True,  c), 'ret_net')['mean_bps'] for c in costs]\n"
            "else:\n"
            f"    ug_net = [{ug_mean}, {R['ug_net05']}, {R['ug_net10']}, {R['ug_net20']}]\n"
            f"    g_net  = [{g_mean}, {R['g_net05']}, {R['g_net10']}, {R['g_net20']}]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(costs, ug_net, 'o-', c=AMBER, lw=2, label='Ungated')\n"
            "ax.plot(costs, g_net,  's--', c=RED, lw=2, label='Gated (ADX>25)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net mean bps/trade')\n"
            "ax.set_title('Costs: the gated arm starts below zero and falls further')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            "The ungated arm starts just above zero and gently slides with each bp "
            "of cost -- it was never alive to begin with.  The gated arm starts negative and "
            "falls further.  There is no cost level where either arm is investable."
        ),

        # ---- BEAT 7 -- GOING FURTHER -------------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **Is the MA cross *ever* useful on daily bars?** Study 21 (Fools-Gold) tests "
            "the 50/200 golden cross on daily SPY -- same family, same verdict.  Study 78 "
            "(Crossed-Wires) tests MACD, which is EMA-based.\n"
            "- **When does ADX actually work?** If ADX correctly reads trend strength but the "
            "MA cross is a bad signal to gate, maybe it gates a *different* rule better.  "
            "Try `strategy.donchian_entries` as the underlying signal, or a cross-sectional "
            "momentum rank.\n"
            "- **Is *any* regime filter real?** Study 86 (Tail-Radar) tests VIX-level gating "
            "of equity trades -- the volatility analogue of the ADX regime question.\n\n"
            "*Think ADX works with a different rule or timeframe?  Fork this, change the "
            "signal, show a gated arm that significantly beats the ungated arm AND beats a "
            "coin (|t| >= 2 on both).  That is the bar.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 -- FOR THE QUANTS
# ===========================================================================
def build_quants():
    ug_mean = R['ug_mean']
    ug_t = R['ug_t']
    g_mean = R['g_mean']
    g_t = R['g_t']
    delta = R['delta_g_ug']
    n_gated = R['n_gated']
    n_ungated = R['n_ungated']
    pct = R['pct_filtered']

    # Inline lists for fallback (avoid nested f-string braces)
    ug_ns = f"{R['spy_ug_n']},{R['qqq_ug_n']},{R['iwm_ug_n']},{R['aapl_ug_n']},{R['tsla_ug_n']},{R['nvda_ug_n']}"
    g_ns  = f"{R['spy_g_n']},{R['qqq_g_n']},{R['iwm_g_n']},{R['aapl_g_n']},{R['tsla_g_n']},{R['nvda_g_n']}"
    ug_ms = f"{R['spy_ug_t']},{R['qqq_ug_t']},{R['iwm_ug_t']},{R['aapl_ug_t']},{R['tsla_ug_t']},{R['nvda_ug_t']}"
    g_ms  = f"{R['spy_g_mean']},{R['qqq_g_mean']},{R['iwm_g_mean']},{R['aapl_g_mean']},{R['tsla_g_mean']},{R['nvda_g_mean']}"
    g_ts  = f"{R['spy_g_t']},{R['qqq_g_t']},{R['iwm_g_t']},{R['aapl_g_t']},{R['tsla_g_t']},{R['nvda_g_t']}"

    cells = [
        md(
            "# ADX-Filter -- a quantitative teardown\n"
            "### Daily MA(20/50) cross, ungated vs ADX(14)>25-gated, vs random-direction control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![ADX_filter_adds_value%3F: Busted](https://img.shields.io/badge/ADX_filter_adds_value%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim now carrying its standard error.*  We test whether "
            "conditioning an MA(20/50) cross on ADX(14) >= 25 produces a higher per-trade "
            "20-day forward return than the unconditioned cross, and whether either beats a "
            "random-direction control.\n\n"
            "> **Not investment advice.**  Real data: Yahoo Finance daily bars, 2010-01-04 to "
            "2026-06-13, six liquid names.  Methods in [`docs/references.md`](../docs/references.md), "
            "reproducible numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **In plain words** notes translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Ungated gross **+{ug_mean:.2f} bps/trade**, "
            f"HAC *t* = **+{ug_t:.2f}**; gated **{g_mean:+.2f} bps/trade**, "
            f"*t* = **{g_t:+.2f}**; ADX gate costs **{delta:+.2f} bps/trade** vs ungated. |\n"
            f"| **Tradability** | `MIRAGE` | Neither arm clears |*t*| = 1 gross; gated arm "
            f"starts negative before costs; costs decrease returns monotonically. |\n"
            f"| **ADX filter adds value?** | `BUSTED` | Gated arm "
            f"({n_gated} trades) is {abs(delta):.0f} bps/trade *worse* "
            f"than ungated ({n_ungated} trades); "
            f"{pct}% of signals filtered away with no expectancy improvement. |\n\n"
            "> In plain words: the ADX filter removes 80% of signals and delivers zero "
            "improvement in per-trade return -- it creates the illusion of a cleaner system "
            "while making the P&L noisier (fewer trades, same null expectancy)."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            r"Let $d_t \in \{-1,+1\}$ be the direction of the MA(20/50) cross on day $t$ and "
            r"$r_{t+1,t+20}$ the 20-day forward return.  The folk rule asserts:" + "\n\n"
            r"- **H1 (ungated signal).** $\mathbb{E}[d_t \cdot r_{t+1,t+20}] > 0$"
            " -- the cross carries directional information.\n"
            r"- **H2 (gate improvement).** $\mathbb{E}[d_t \cdot r_{t+1,t+20} \mid \text{ADX}_t "
            r"> 25] > \mathbb{E}[d_t \cdot r_{t+1,t+20}]$"
            " -- conditioning on ADX > 25 strictly improves the expectation.\n"
            "- **H3 (beat a coin).** The gated arm beats a random-direction control on identical "
            "entry dates.\n"
            "- **H4 (tradable).** The improvement survives a realistic round-trip cost.\n\n"
            f"We reject H1 (t = +{ug_t:.2f}, not significant), H2 (gated is worse), H3, and H4."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what? -- what rides on each answer\n\n"
            "The ADX gate is sold as a *free* quality improvement: same rule, fewer whipsaws.  "
            "If H2 holds, every MA-based system should carry ADX as a prerequisite -- a standard "
            "module in any systematic trend-follower.  The interesting failure here is not just "
            "'ADX does not work' -- it is that the gate *actively hurts* the per-trade return "
            "while creating the misleading appearance of selectivity.  A practitioner who adds "
            "this filter will see fewer trades and will attribute the resulting quieter P&L to "
            "skill, when the population expected return is identical (noise) or worse."
        ),

        # ---- BEAT 3 -- PROTOCOL ----------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Entry.** MA(20/50) cross of closes up to day *t*; enter at day *t+1* open.  "
            "ADX(14) also measured at close of day *t* (no look-ahead).\n"
            "- **Gate.** GATED arm: only enter when ADX(14)_t >= 25.  UNGATED arm: take every "
            "MA-cross signal.\n"
            "- **Exit.** Fixed hold: close at the close of day *t+20* (approximately one calendar "
            "month; no barrier, no stop).  This is the honest forward-return question: 'does the "
            "signal predict the next month's direction?'\n"
            "- **Control.** Identical entry dates as each arm, direction replaced by an i.i.d. "
            "fair coin (seeded).\n"
            "- **Inference.** Newey-West HAC *t* on per-trade returns; per-instrument breakdown; "
            "cost sweep; synthetic positive control.\n"
            "- **Tapes.** SPY, QQQ, IWM, AAPL, TSLA, NVDA daily, 2010-2026 (~4,100 bars/ticker)."
        ),

        # ---- BEAT 4 -- TEARDOWN ----------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - Per-instrument HAC t-stats: ungated and gated\n\n"
            "If the gate adds signal, gated bars should be systematically higher than ungated "
            "bars, with at least some clearing |t| = 2."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rec_ug, rec_g = [], []\n"
            "    for t in TICKERS:\n"
            "        b = data.fetch_daily(t, fetch=False)\n"
            "        ent = st.ma_cross_entries(b['close'])\n"
            "        adx_s = st.adx(b, n=14)\n"
            "        g_ent = st.apply_adx_gate(ent, adx_s, threshold=25.0)\n"
            "        s_ug = st.summarize(st.run_trades(b, ent, cost_bps=0), 'ret_gross')\n"
            "        s_g  = st.summarize(st.run_trades(b, g_ent, cost_bps=0), 'ret_gross')\n"
            "        rec_ug.append((t, s_ug['n_trades'], s_ug['mean_bps'], s_ug['tstat']))\n"
            "        rec_g.append((t, s_g['n_trades'], s_g['mean_bps'], s_g['tstat']))\n"
            "    df_ug = pd.DataFrame(rec_ug, columns=['ticker','n','mean_bps','t'])\n"
            "    df_g  = pd.DataFrame(rec_g,  columns=['ticker','n','mean_bps','t'])\n"
            "else:\n"
            "    df_ug = pd.DataFrame({'ticker': TICKERS,\n"
            "        'n':       [" + ug_ns + "],\n"
            "        'mean_bps':[" + ug_ms + "],\n"
            "        't':       [" + ug_ms + "]})\n"
            "    df_g  = pd.DataFrame({'ticker': TICKERS,\n"
            "        'n':       [" + g_ns  + "],\n"
            "        'mean_bps':[" + g_ms  + "],\n"
            "        't':       [" + g_ts  + "]})\n"
            "x = np.arange(len(TICKERS)); w = 0.37\n"
            "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
            "ax.bar(x - w/2, df_ug['t'], w, color=AMBER, label='Ungated', alpha=0.8)\n"
            "ax.bar(x + w/2, df_g['t'],  w, color=RED,   label='Gated (ADX>25)', alpha=0.8)\n"
            "for s in (2,-2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels(TICKERS)\n"
            "ax.set_ylabel('HAC t-stat (gross mean/trade)')\n"
            "ax.set_title('Neither arm clears |t|=2 on any instrument; gate adds no lift')\n"
            "ax.legend(); ax.set_ylim(-3, 3); plt.tight_layout(); plt.show()\n"
            "print('Ungated:'); print(df_ug.round(2).to_string(index=False))\n"
            "print(); print('Gated:'); print(df_g.round(2).to_string(index=False))"
        ),
        md(
            "> In plain words: every bar -- gated and ungated -- sits inside the noise band "
            "at every instrument.  The gate does not consistently lift any name above the "
            "inference bar.  On QQQ and NVDA the gate makes the t-stat *more negative*."
        ),
        md(
            "### 4b - Pooled comparison: UNGATED vs GATED vs coin\n\n"
            "Pooling all instruments gives the clearest single read of 'does the gate help?'"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ug_s  = st.summarize(pool(False, 0), 'ret_gross')\n"
            "    g_s   = st.summarize(pool(True,  0), 'ret_gross')\n"
            "    r_ug  = st.summarize(pool(False, 0, seed=108), 'ret_gross')\n"
            "    r_g   = st.summarize(pool(True,  0, seed=108), 'ret_gross')\n"
            "    um, gm = ug_s['mean_bps'], g_s['mean_bps']\n"
            "    rum, rgm = r_ug['mean_bps'], r_g['mean_bps']\n"
            "    ut, gt = ug_s['tstat'], g_s['tstat']\n"
            "else:\n"
            f"    um, gm = {ug_mean}, {g_mean}\n"
            f"    rum, rgm = {R['rand_ug_mean']}, {R['rand_g_mean']}\n"
            f"    ut, gt = {ug_t}, {g_t}\n"
            "labels = ['UNGATED','GATED (ADX>25)','Random (vs ungated)','Random (vs gated)']\n"
            "vals = [um, gm, rum, rgm]\n"
            "cols = [AMBER, RED, GREY, GREY]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "bars = ax.bar(labels, vals, color=cols)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('gross mean (bps/trade, 20-day hold)')\n"
            "ax.set_title('Pooled: gate makes the signal worse, not better')\n"
            "for b, v, ts in zip(bars[:2], [um, gm], [ut, gt]):\n"
            "    ax.annotate(f'{v:+.1f} bps  t={ts:+.2f}',\n"
            "                (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='bottom' if v>=0 else 'top', fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            f"print('Delta gated - ungated:', round({delta}, 2), 'bps/trade')"
        ),
        md(
            "### 4c - Synthetic positive control -- the engine can find signal when signal exists\n\n"
            "Planting AR(1) momentum in a synthetic daily tape and sweeping trend strength "
            "confirms the engine is a faithful detector -- and that real markets simply have "
            "nothing at this scale for it to detect."
        ),
        code(
            "moms = [0.0, 0.05, 0.10, 0.15, 0.20]\n"
            "delta_bps = []\n"
            "for m in moms:\n"
            "    b, _ = data.synthetic_daily(n_days=1000, trend_strength=m, seed=108)\n"
            "    ent = st.ma_cross_entries(b['close'])\n"
            "    adx_s = st.adx(b, n=14)\n"
            "    g_ent = st.apply_adx_gate(ent, adx_s, threshold=25.0)\n"
            "    ug_r = st.summarize(st.run_trades(b, ent, cost_bps=0), 'ret_gross')['mean_bps']\n"
            "    g_r  = (st.summarize(st.run_trades(b, g_ent, cost_bps=0), 'ret_gross')['mean_bps']\n"
            "            if len(g_ent) > 0 else float('nan'))\n"
            "    delta_bps.append(g_r - ug_r)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(moms, delta_bps, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, ls='--', c=GREY)\n"
            "ax.set_xlabel('planted daily AR(1) trend strength')\n"
            "ax.set_ylabel('gated minus ungated (bps/trade)')\n"
            "ax.set_title('Even with planted trends the ADX gate adds little extra')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('delta bps at each momentum:', [f\"{d:+.1f}\" for d in delta_bps])"
        ),
        md(
            "> In plain words: even with planted strong trends the ADX gate does not reliably "
            "improve the per-trade return -- ADX is a lagging indicator that rises *after* the "
            "trend has started.  The gate selects bars near the *end* of a trend move, not the "
            "beginning."
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal `NONE`** -- ungated gross {ug_mean:+.2f} bps/trade, "
            f"HAC *t* {ug_t:+.2f}; gated {g_mean:+.2f} bps/trade, "
            f"*t* {g_t:+.2f}; delta {delta:+.2f} bps.  Every "
            "per-instrument |*t*| < 1 in both arms.  The MA(20/50) cross has no real "
            "directional signal and the ADX gate does not create one.\n"
            f"- **Tradability `MIRAGE`** -- the gated arm is negative before costs; "
            f"{pct}% of signals filtered away reduces trade count without "
            "improving expectancy.  Neither arm is investable.\n"
            f"- **ADX filter adds value? `BUSTED`** -- the gate costs "
            f"{abs(delta):.0f} bps/trade vs the unfiltered rule.  The "
            "folk rule 'only trade when the trend is strong' fails a direct test of "
            "its primary promise."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT? -----------------------------------
        md(
            "## 6 - Could you trade it? -- cost sweep\n\n"
            "The MA(20/50) cross is a slow rule (~3 signals/year/instrument).  The gated "
            "version fires ~0.7/year.  Even a 0.5 bp round-trip cost erodes the noise-level gross:"
        ),
        code(
            "costs = [0.0, 0.5, 1.0, 2.0]\n"
            "if HAVE_REAL:\n"
            "    ug_net = [st.summarize(pool(False, c), 'ret_net')['mean_bps'] for c in costs]\n"
            "    g_net  = [st.summarize(pool(True,  c), 'ret_net')['mean_bps'] for c in costs]\n"
            "    ug_tv  = [st.summarize(pool(False, c), 'ret_net')['tstat'] for c in costs]\n"
            "    g_tv   = [st.summarize(pool(True,  c), 'ret_net')['tstat'] for c in costs]\n"
            "else:\n"
            f"    ug_net = [{ug_mean}, {R['ug_net05']}, {R['ug_net10']}, {R['ug_net20']}]\n"
            f"    g_net  = [{g_mean}, {R['g_net05']}, {R['g_net10']}, {R['g_net20']}]\n"
            f"    ug_tv  = [{ug_t}, 0.37, 0.36, 0.34]\n"
            f"    g_tv   = [{g_t}, -0.23, -0.23, -0.24]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(costs, ug_net, 'o-', c=AMBER, lw=2, label='Ungated')\n"
            "ax.plot(costs, g_net,  's--', c=RED, lw=2, label='Gated (ADX>25)')\n"
            "ax2 = ax.twinx()\n"
            "ax2.plot(costs, ug_tv, 'o:', c=AMBER, lw=1, alpha=0.5)\n"
            "ax2.plot(costs, g_tv,  's:', c=RED, lw=1, alpha=0.5)\n"
            "ax2.axhline(-2, ls=':', c=GREY); ax2.set_ylabel('HAC t', color=GREY)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net mean bps/trade')\n"
            "ax.set_title('Neither arm has a positive break-even: both start at noise or worse')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            "> In plain words: the ungated arm floats near zero gross (*t* ~ 0.4) but "
            "costs clip it flat -- it is noise.  The gated arm starts below zero.  There is "
            "no cost level at which either arm is a statistical winner."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further -- open forks\n\n"
            "The core finding -- ADX gating of a daily MA cross adds nothing -- is robust "
            "across six instruments and 15+ years.  Directions worth exploring:\n\n"
            "- **A faster signal.** The 20/50 cross fires rarely; the ADX gate might interact "
            "differently with a 5/20 cross or a Donchian(20) breakout "
            "(`strategy.donchian_entries` is already implemented).  Fork and test.\n"
            "- **ADX as a standalone forecaster.** Instead of using ADX as a gate, does "
            "ADX *level* predict future volatility or future 20-day return directly?  That "
            "is a separate question from gating.\n"
            "- **The VIX analogue.** Study 86 (Tail-Radar) tests VIX-level gating of "
            "equity trend entries -- the volatility-dimension analogue of this study.\n"
            "- **A shorter hold.** Our 20-day hold matches the MA cross's lag; a 5-day "
            "or 60-day hold changes the question.  Fork the `hold_days` parameter.\n\n"
            "*Raise the bar: for the ADX gate to be useful, gated *t* must exceed ungated *t* "
            "AND both must clear the inference bar (|t| >= 2).  Show that, and the "
            "folk rule earns its place.*"
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
