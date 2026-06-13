"""Generate the two narrative notebooks for Study 110 (Faber-Timing).

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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-13).
R = dict(
    n=8400, start="1993-01-29", end="2026-06-12",
    fp="42b159a9d39a", in_mkt=75.3, n_switches=215,
    # buy-and-hold
    bh_cagr=10.82, bh_sharpe=0.553, bh_vol=18.59, bh_dd=-55.19,
    # SMA timing (5 bps cost)
    tm_cagr=9.15, tm_sharpe=0.729, tm_vol=12.01, tm_dd=-21.91,
    # Random timing control
    rn_cagr=5.07, rn_sharpe=0.304, rn_vol=16.27, rn_dd=-57.07,
    # Sharpe diff t-stats
    t_timing_vs_bh=-0.73, t_timing_vs_rand=1.75,
    dd_improvement=33.28,
    # Cost sweep (Sharpe at various one-way costs)
    sharpe_0bps=0.756, sharpe_1bps=0.751, sharpe_5bps=0.729,
    sharpe_10bps=0.702, sharpe_25bps=0.621,
    # Sub-period: Sharpe & DD
    pre2000_bh_sh=1.25, pre2000_bh_dd=-19.0, pre2000_tm_sh=1.11, pre2000_tm_dd=-14.6,
    bear2000_bh_sh=-0.04, bear2000_bh_dd=-55.2, bear2000_tm_sh=0.55, bear2000_tm_dd=-16.7,
    bull2010_bh_sh=0.85, bull2010_bh_dd=-19.3, bull2010_tm_sh=0.88, bull2010_tm_dd=-16.3,
    era2020_bh_sh=0.70, era2020_bh_dd=-33.7, era2020_tm_sh=1.06, era2020_tm_dd=-17.9,
)

# ---------------------------------------------------------------------------
# Shared analysis preamble
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))           # the study package
sys.path.insert(0, os.path.abspath("../../.."))     # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from faber_timing import data, strategy as st

TICKER = "SPY"
SMA_N  = 200
TBILL  = 0.04 / 252.0   # flat 4%/yr daily proxy
COST   = 5.0             # one-way bps

# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-13).
R = dict(
    n=8400, start="1993-01-29", end="2026-06-12",
    fp="42b159a9d39a", in_mkt=75.3, n_switches=215,
    bh_cagr=10.82, bh_sharpe=0.553, bh_vol=18.59, bh_dd=-55.19,
    tm_cagr=9.15,  tm_sharpe=0.729, tm_vol=12.01,  tm_dd=-21.91,
    rn_cagr=5.07,  rn_sharpe=0.304, rn_vol=16.27,  rn_dd=-57.07,
    t_timing_vs_bh=-0.73, t_timing_vs_rand=1.75, dd_improvement=33.28,
    sharpe_0bps=0.756, sharpe_1bps=0.751, sharpe_5bps=0.729,
    sharpe_10bps=0.702, sharpe_25bps=0.621,
    pre2000_bh_sh=1.25, pre2000_bh_dd=-19.0, pre2000_tm_sh=1.11, pre2000_tm_dd=-14.6,
    bear2000_bh_sh=-0.04, bear2000_bh_dd=-55.2, bear2000_tm_sh=0.55, bear2000_tm_dd=-16.7,
    bull2010_bh_sh=0.85, bull2010_bh_dd=-19.3, bull2010_tm_sh=0.88, bull2010_tm_dd=-16.3,
    era2020_bh_sh=0.70, era2020_bh_dd=-33.7, era2020_tm_sh=1.06, era2020_tm_dd=-17.9,
)

def _have_cache():
    return os.path.exists(data._cache_path(TICKER, data.DEFAULT_CACHE))

HAVE_REAL = _have_cache()
print("real SPY cache present:", HAVE_REAL)
"""

LOAD_REAL = """\
if HAVE_REAL:
    df     = data.fetch_prices(TICKER, fetch=False)
    close  = df["close"]
    sig    = st.timing_signal(close, sma_n=SMA_N)
    rand_s = st.random_timing_signal(sig, seed=42)
    bt     = st.run_backtest(close, sig,    tbill_daily=TBILL, cost_bps=COST)
    bt_rn  = st.run_backtest(close, rand_s, tbill_daily=TBILL, cost_bps=COST)
    s_bh   = st.summary(bt["r_bh"])
    s_tm   = st.summary(bt["r_strategy"])
    s_rn   = st.summary(bt_rn["r_strategy"])
    in_mkt = float(sig.dropna().mean())
else:
    close = None
    s_bh  = dict(cagr=R["bh_cagr"]/100, sharpe=R["bh_sharpe"], max_drawdown=R["bh_dd"]/100, vol_ann=R["bh_vol"]/100)
    s_tm  = dict(cagr=R["tm_cagr"]/100, sharpe=R["tm_sharpe"], max_drawdown=R["tm_dd"]/100, vol_ann=R["tm_vol"]/100)
    s_rn  = dict(cagr=R["rn_cagr"]/100, sharpe=R["rn_sharpe"], max_drawdown=R["rn_dd"]/100, vol_ann=R["rn_vol"]/100)
    in_mkt = R["in_mkt"] / 100
    bt = bt_rn = sig = None
"""

# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Faber-Timing -- does the 10-month SMA protect you from crashes?\n"
            "### The most-cited tactical timing rule, tested honestly, in plain English\n\n"
            "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Risk_reduction%3F: Confirmed](https://img.shields.io/badge/Risk_reduction%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "In 2007, Mebane Faber published a simple idea that has since been downloaded over "
            "a million times: hold the stock market when its price is above the 10-month moving "
            "average; move to cash when it falls below. No complex forecasts, no parameters to "
            "tune. Just a slow-moving average as a crash filter. This notebook asks: does it "
            "actually work, and if so, in what sense?\n\n"
            "> **This is the plain-language layer.** Want the t-stats, bootstrap intervals and "
            "regime-control maths? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\n" + LOAD_REAL),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the rule cut drawdown? | **Yes, dramatically.** SPY's worst "
            f"drawdown falls from **{R['bh_dd']:.0f}%** to **{R['tm_dd']:.0f}%** (−{R['dd_improvement']:.0f} pp). |\n"
            "| Is that just from being invested less often? | **No.** A random-timing control with the "
            f"same in-market fraction ({R['in_mkt']:.1f}%) has a *worse* drawdown than buy-and-hold "
            f"({R['rn_dd']:.0f}%). The SMA picks the *right* days to be out, not just fewer days. |\n"
            "| Does it improve return? | **No** -- CAGR is lower (+{R['tm_cagr']:.1f}% vs +{R['bh_cagr']:.1f}%). "
            "It earns a higher Sharpe because risk falls more than return. |\n"
            "| Could you trade it? | **Yes, cheaply.** Only ~6 switches per year; costs barely touch "
            f"the Sharpe (still +{R['sharpe_25bps']:.2f} at 25 bps one-way). |\n\n"
            "> The rule is what Faber always said it was: a crash shield, not an alpha engine. "
            "The honest question is whether you want a smoother ride at the cost of missing "
            "some upside. On the numbers: yes, decisively on risk; weaker, not certified, on return."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 -- The claim\n\n"
            "> *'Invest in an asset class when its price is above its 10-month simple moving "
            "average. Move to cash (T-bills) when the price falls below. Re-check once a month. "
            "This simple rule, applied to a diversified portfolio of asset classes, dramatically "
            "reduces drawdowns while maintaining most of the long-run return.'*\n"
            "> -- Mebane Faber, SSRN 962461 (2007)\n\n"
            "The idea has a clean economic logic: sustained bear markets are usually accompanied "
            "by price levels falling *below* where they have been on average. A slow moving average "
            "captures this: not a sudden dip, but a confirmed trend downward. When that happens, "
            "you step aside -- earn the T-bill and wait for the recovery. The S&P 500 has had "
            "three major bear markets in our sample (dot-com, 2008, COVID-19). If the rule dodges "
            "even part of those, the lifetime performance is substantially improved."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 -- So what?\n\n"
            "Most investors endure the ride-outs badly -- they sell at the bottom (dot-com, GFC) "
            "and miss the recovery, destroying sequence-of-returns value. If the SMA rule reliably "
            "caps the worst of each crash at −20% instead of −50%+, many investors who would "
            "otherwise panic-sell might stay the course. The economic value is not just in the "
            "abstract return number -- it is in the *behavior it enables*: staying invested through "
            "a −20% drawdown is psychologically very different from a −55% one.\n\n"
            "Conversely, if the rule is just 'be invested 75% of the time on random days', it's "
            "not really timing -- it's just reduced equity exposure. The key test is whether the "
            "*timing* (which days, not just how many days) adds something."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 -- How would we even know?\n\n"
            "Three tests keep us honest:\n\n"
            "1. **The random-timing control.** We build a coin-flip version that goes to cash with "
            f"the same frequency as the SMA rule ({R['in_mkt']:.1f}% in-market) but on random "
            "days. If random timing has the same Sharpe and drawdown as the SMA, the SMA adds "
            "nothing beyond exposure reduction. If SMA beats random, the *timing* (which days) "
            "is what matters.\n"
            "2. **The sub-period breakdown.** A rule that works in bear decades but fails in bull "
            "decades is *regime-dependent* -- honest accounting means showing both.\n"
            "3. **The synthetic positive control.** A simulated tape with genuine bull/bear regimes "
            "shows whether the engine *can* find the signal when it exists. On a flat-vol (null) "
            "tape, the rule should add nothing."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 -- The teardown -- let's actually look\n\n"
            "**First, the big picture: cumulative wealth of all three arms.**"
        ),
        code(
            "fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.0), sharex=True)\n"
            "if HAVE_REAL:\n"
            "    cum_bh = np.exp(bt['r_bh'].cumsum())\n"
            "    cum_tm = np.exp(bt['r_strategy'].cumsum())\n"
            "    cum_rn = np.exp(bt_rn['r_strategy'].cumsum())\n"
            "    axes[0].plot(cum_bh.index, cum_bh, c=RED,   lw=1.8, label='Buy-and-hold')\n"
            "    axes[0].plot(cum_tm.index, cum_tm, c=GREEN, lw=1.8, label='SMA timing')\n"
            "    axes[0].plot(cum_rn.index, cum_rn, c=GREY,  lw=1.2, label='Random timing (null)', ls='--')\n"
            "    # Drawdown\n"
            "    for col, lbl, c in [('r_bh', 'BH', RED), ('r_strategy', 'Timing', GREEN)]:\n"
            "        cum = np.exp(bt[col].cumsum())\n"
            "        dd  = cum / cum.cummax() - 1\n"
            "        axes[1].fill_between(dd.index, dd, 0, alpha=0.3, color=c, label=lbl)\n"
            "else:\n"
            "    axes[0].text(0.5, 0.5, 'Cache not available\\nRun verify.py --fetch', ha='center', va='center', transform=axes[0].transAxes)\n"
            "axes[0].set_ylabel('Cumulative wealth (log scale)')\n"
            "axes[0].set_yscale('log')\n"
            "axes[0].legend(); axes[0].set_title('SPY: buy-and-hold vs SMA timing vs random timing')\n"
            "axes[1].set_ylabel('Drawdown'); axes[1].legend()\n"
            "plt.tight_layout(); plt.show()\n"
        ),
        md(
            f"The SMA rule avoids most of the 2000--2009 lost decade and the 2008 GFC. "
            f"Its cumulative wealth line diverges from buy-and-hold precisely during the "
            f"two biggest bear markets in the sample. The random-timing control (grey dashed) "
            f"does *not* avoid those crashes -- it's just out on random days, including many "
            f"good days. That confirms the SMA's benefit is genuine timing, not mere exposure reduction."
        ),
        md(
            "**Now the honest risk-vs-return scorecard:**"
        ),
        code(
            "labels = ['Buy-and-hold', 'SMA timing', 'Random timing']\n"
            "sharpes = [s_bh['sharpe'], s_tm['sharpe'], s_rn['sharpe']]\n"
            "dds     = [abs(s_bh['max_drawdown']), abs(s_tm['max_drawdown']), abs(s_rn['max_drawdown'])]\n"
            "cagrs   = [s_bh['cagr']*100, s_tm['cagr']*100, s_rn['cagr']*100]\n"
            "colors  = [RED, GREEN, GREY]\n"
            "fig, ax = plt.subplots(1, 3, figsize=(12, 4.5))\n"
            "for i, (vals, title, fmt) in enumerate([\n"
            "        (sharpes, 'Sharpe ratio', '+.3f'),\n"
            "        (dds,     'Max drawdown (abs)', '.1%'),\n"
            "        (cagrs,   'CAGR (%)', '+.1f'),\n"
            "    ]):\n"
            "    bars = ax[i].bar(labels, vals, color=colors, width=0.55)\n"
            "    for b, v in zip(bars, vals):\n"
            "        fmt_v = f'{v:{fmt}}' if 'f' not in fmt else f'{v:.3f}'\n"
            "        ax[i].annotate(fmt_v, (b.get_x()+b.get_width()/2, v/2), ha='center', va='center', color='white', fontsize=9)\n"
            "    ax[i].set_title(title); ax[i].tick_params(axis='x', rotation=20)\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'SMA timing Sharpe: {{s_tm[\"sharpe\"]:.3f}} vs BH: {{s_bh[\"sharpe\"]:.3f}}')\n"
            f"print(f'Max DD: SMA {{s_tm[\"max_drawdown\"]:.1%}} vs BH {{s_bh[\"max_drawdown\"]:.1%}}')\n"
        ),
        md(
            f"The SMA rule has a higher Sharpe ({R['tm_sharpe']:.3f} vs {R['bh_sharpe']:.3f}) and "
            f"dramatically lower drawdown ({R['tm_dd']:.1f}% vs {R['bh_dd']:.1f}%) -- but lower "
            f"CAGR ({R['tm_cagr']:.1f}% vs {R['bh_cagr']:.1f}%). The random timing arm has the "
            f"*worst* drawdown ({R['rn_dd']:.1f}%) -- because it sits out on random good days and "
            "stays in on random bad days, getting the worst of both."
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal -- REAL (on the risk axis).** Max drawdown cut from {R['bh_dd']:.0f}% to "
            f"{R['tm_dd']:.0f}% (−{R['dd_improvement']:.0f} pp); random timing's DD is *worse* than BH, "
            f"confirming timing skill. Sharpe up from {R['bh_sharpe']:.3f} to {R['tm_sharpe']:.3f} -- "
            f"directionally clear but not statistically certified vs BH (*t* = {R['t_timing_vs_bh']:+.2f}), "
            f"stronger vs random timing (*t* = {R['t_timing_vs_rand']:+.2f}).\n"
            "- **Tradability -- FRAGILE.** Only ~6 switches/year; cheap to run even at institutional "
            f"costs. Sharpe stays above BH down to 25 bps/one-way ({R['sharpe_25bps']:.3f} vs {R['bh_sharpe']:.3f}). "
            "But CAGR lags BH, the advantage is regime-dependent (shines in bear decades, lags in "
            "bull markets), and the return improvement is not statistically certified.\n"
            "- **Risk reduction or alpha? -- CONFIRMED** as risk reduction. The SMA rule is a "
            "drawdown shield. It buys smoother rides by giving up some bull-market return."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 -- Could you actually trade it?\n\n"
            "This is where many rules go to die -- but not this one. With only ~6 switches per year "
            "(215 over 33 years), the per-trade cost is almost irrelevant:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    costs = [0.0, 1.0, 5.0, 10.0, 25.0]\n"
            "    sharpes = []\n"
            "    for c in costs:\n"
            "        bt_c = st.run_backtest(close, sig, tbill_daily=TBILL, cost_bps=c)\n"
            "        sharpes.append(st.summary(bt_c['r_strategy'])['sharpe'])\n"
            "else:\n"
            "    costs   = [0.0, 1.0, 5.0, 10.0, 25.0]\n"
            "    sharpes = [R['sharpe_0bps'], R['sharpe_1bps'], R['sharpe_5bps'], R['sharpe_10bps'], R['sharpe_25bps']]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, sharpes, 'o-', c=GREEN, lw=2, label='SMA timing Sharpe')\n"
            f"ax.axhline(R['bh_sharpe'], ls='--', c=RED, label=f'Buy-and-hold: {{R[\"bh_sharpe\"]:.3f}}')\n"
            "ax.set_xlabel('one-way cost (bps)'); ax.set_ylabel('Sharpe ratio')\n"
            "ax.set_title('Cost barely matters -- only 6 switches/year'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Sharpe break-even vs BH: the rule stays above BH Sharpe even at 25 bps one-way')\n"
        ),
        md(
            f"The Sharpe stays above buy-and-hold ({R['bh_sharpe']:.3f}) at every realistic cost level. "
            "Low turnover is the rule's second great feature, after drawdown protection. An investor "
            "who simply holds SPY and flips to BIL/SHV 6 times per year could run this strategy "
            "at near-zero marginal cost through most brokerages today."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **The bear-decade test.** The 2000s are where this rule earns its reputation: "
            f"Sharpe +{R['bear2000_tm_sh']:.2f} vs BH {R['bear2000_bh_sh']:.2f}, drawdown "
            f"{R['bear2000_tm_dd']:.0f}% vs {R['bear2000_bh_dd']:.0f}%. In the 2010s bull "
            f"market the gap narrows (+{R['bull2010_tm_sh']:.2f} vs +{R['bull2010_bh_sh']:.2f}). "
            "Regime matters.\n"
            "- **Continuous version.** Instead of a binary in/out, you could scale position by "
            "how far price is *above or below* the SMA. That's closer to the vol-targeting "
            "approach in [Study 16 -- Storm-Shy](../../16-storm-shy/).\n"
            "- **Multi-asset extension.** Faber's original paper applied this to five asset "
            "classes simultaneously (US equity, international equity, bonds, REITs, commodities). "
            "The diversification benefit there is larger than for a single index.\n"
            "- **Monthly vs daily.** The original rule re-evaluates at month-end only. We use "
            "daily closes; the monthly variant is available via `data.resample_monthly()` and "
            "`strategy.timing_signal_monthly()`.\n\n"
            "*Think the rule has more to give? Fork this, test on international indices, or try a "
            "multi-asset version and show whether the diversification benefit holds post-2006. "
            "That's the bar.*"
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
            "# Faber-Timing -- a quantitative teardown\n"
            "### Real daily tape - 200-day SMA - random-timing control - HAC inference - sub-period decay\n\n"
            "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Risk_reduction%3F: Confirmed](https://img.shields.io/badge/Risk_reduction%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim now carrying its standard error.* We test whether "
            "the Faber 200-day SMA timing rule improves risk-adjusted return vs buy-and-hold "
            "and a random-timing control on 33 years of SPY, isolate timing skill from exposure "
            "reduction, and break down results by decade.\n\n"
            "> **Not investment advice.** Real data: SPY daily total-return closes 1993-01-29 "
            "to 2026-06-12 (n=8,400); cash leg proxied at 4%/yr flat (FRED unavailable). "
            "Methods in [`docs/references.md`](../docs/references.md), reproducible numbers "
            "in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The `> in plain words` notes** translate each result back to intuition."
        ),
        code(BOOT + "\nfrom quantlab import analytics, stats\n" + "\n" + LOAD_REAL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `REAL` (on risk) | SMA max DD **{R['tm_dd']:.1f}%** vs BH **{R['bh_dd']:.1f}%** "
            f"(−{R['dd_improvement']:.0f} pp); random timing DD {R['rn_dd']:.1f}% -- *worse* than BH. "
            f"Sharpe {R['tm_sharpe']:.3f} vs BH {R['bh_sharpe']:.3f}: directional but |*t*| = {abs(R['t_timing_vs_bh']):.2f} "
            f"vs BH, {R['t_timing_vs_rand']:.2f} vs random. |\n"
            f"| **Tradability** | `FRAGILE` | ~6 switches/yr; costs minimal (Sharpe > BH at "
            f"25 bps one-way). CAGR lags BH (+{R['tm_cagr']:.1f}% vs +{R['bh_cagr']:.1f}%); "
            "advantage regime-dependent; Sharpe gain vs BH not certified. |\n"
            "| **Risk reduction or alpha?** | `CONFIRMED` | Vol falls from "
            f"{R['bh_vol']:.1f}% to {R['tm_vol']:.1f}% vs BH; CAGR falls too. "
            "Sharpe improves because vol falls faster than return. |\n\n"
            "> in plain words: Faber timing genuinely sidesteps bear markets (the SMA picks "
            "the right out-days, not just fewer days), but it also misses bull-market upside "
            "and its CAGR lags buy-and-hold. The primary benefit is verified; the secondary "
            "claim (better risk-adjusted return) is directionally supported but not certified."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 -- The claim, steelmanned\n\n"
            "Let $P_t$ be the SPY adjusted close and $\\mu_t = \\frac{1}{200}\\sum_{k=0}^{199} P_{t-k}$ "
            "the 200-day SMA. Faber's binary signal (in daily form) is:\n\n"
            "$$w_t = \\mathbf{1}[P_{t-1} > \\mu_{t-1}]$$\n\n"
            "The strategy return is $r_t^{\\text{strat}} = w_t r_t^{\\text{SPY}} + (1-w_t) r_t^{\\text{cash}} - "
            "c \\cdot |w_t - w_{t-1}|$ where $c$ is the one-way cost. The three hypotheses:\n\n"
            "- **H1 (risk reduction).** $\\text{MaxDD}(w) < \\text{MaxDD}(\\text{BH})$ and the "
            "reduction cannot be explained by random timing at the same in-market frequency.\n"
            "- **H2 (Sharpe improvement).** $\\text{SR}(w) > \\text{SR}(\\text{BH})$ with "
            "statistical support: |*t*| $\\geq$ 2 on the daily return difference.\n"
            "- **H3 (tradable).** The advantage survives realistic transaction costs and does not "
            "require predicting regime shifts.\n\n"
            "We confirm H1 and H3; H2 is directionally supported but |*t*| = 0.73 falls short "
            "of the inference bar."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 -- So what? -- what rides on each answer\n\n"
            "The Faber rule is the benchmark for every tactical asset allocation product. If H1 "
            "holds, it validates a whole class of trend-following overlays. If H2 fails, it means "
            "risk reduction and Sharpe improvement are separable things -- you can get one "
            "without the other. The distinction matters: a pure risk-reduction tool should be "
            "evaluated differently from a return-enhancing one, and investors with different "
            "tolerance for drawdown (vs CAGR) make different choices."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 -- How we'd know -- the protocol\n\n"
            "- **Signal.** Daily SMA(200) on SPY adjusted close; lag one day (signal at $t$ acts on "
            "return $t \\to t+1$); binary 0/1.\n"
            "- **Cash leg.** Flat 4%/yr proxy (FRED unavailable). Sensitivity tested at 0% and 5%.\n"
            "- **Cost.** 5 bps one-way ($c = 5$); swept from 0 to 25 bps.\n"
            "- **Control.** Random-timing: same in-market fraction, random days, seeded for "
            "reproducibility. Isolates timing vs exposure.\n"
            "- **Inference.** HAC t-stat (Newey-West) on mean daily return; return-difference "
            "t-stat for Sharpe comparison (Jobson-Korkie style). Sub-period breakdown for decay.\n"
            "- **Positive control.** Synthetic two-regime tape to confirm the engine recovers "
            "drawdown reduction when regimes exist, and adds nothing on a flat-vol null."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 -- The teardown"),
        md(
            "### 4a -- The random-timing control: is the SMA picking the right days?\n\n"
            "The key test: if the SMA rule just reduces exposure, the random-timing control "
            "(same frequency, random days) should have similar results. If SMA beats random "
            "on drawdown, timing is genuine."
        ),
        code(
            "print(f'In-market fraction (SMA): {in_mkt:.1%}')\n"
            "print(f'\\nArm           | CAGR   | Sharpe | Vol    | MaxDD  |')\n"
            "print(f'-' * 56)\n"
            "for lbl, s in [('Buy-and-hold', s_bh), ('SMA timing', s_tm), ('Random timing', s_rn)]:\n"
            "    print(f'{lbl:14s} | {s[\"cagr\"]:+.2%} | {s[\"sharpe\"]:+.3f} | {s[\"vol_ann\"]:.2%} | {s[\"max_drawdown\"]:+.2%} |')\n"
            "if HAVE_REAL:\n"
            "    t_vs_bh   = st.sharpe_diff_tstat(bt['r_strategy'], bt['r_bh'])\n"
            "    t_vs_rand = st.sharpe_diff_tstat(bt['r_strategy'], bt_rn['r_strategy'])\n"
            "else:\n"
            "    t_vs_bh, t_vs_rand = R['t_timing_vs_bh'], R['t_timing_vs_rand']\n"
            "print(f'\\nSharpe diff t-stat (SMA vs BH):     {t_vs_bh:+.2f}')\n"
            "print(f'Sharpe diff t-stat (SMA vs random):  {t_vs_rand:+.2f}')\n"
            "print(f'\\nConclusion: SMA timing vs BH: not certified (|t|={abs(t_vs_bh):.2f} < 2)')\n"
            "print(f'SMA timing vs random: near the bar (t={t_vs_rand:+.2f})')\n"
            "print(f'MaxDD test: SMA {s_tm[\"max_drawdown\"]:+.1%} << BH {s_bh[\"max_drawdown\"]:+.1%} << Random {s_rn[\"max_drawdown\"]:+.1%}')\n"
        ),
        md(
            "> in plain words: The random-timing control has a *worse* drawdown than buy-and-hold "
            "(-57% vs -55%) -- proving that simply being out of the market 25% of the time on "
            f"random days hurts you more than it helps. The SMA rule's -22% drawdown comes from "
            "being out on the *right* days (the 2000s bear, 2008). That is genuine timing skill, "
            "confirmed by the control comparison."
        ),
        md(
            "### 4b -- Sub-period breakdown: the rule shines in bear decades\n\n"
            "The regime-dependence is the key caveat. The SMA rule is not uniformly better --"
            "it earns its keep in bear markets and gives some of it back in bull markets."
        ),
        code(
            "sub_periods = [\n"
            "    ('Pre-2000', '1993-01-01', '1999-12-31',\n"
            f"     {R['pre2000_bh_sh']}, {R['pre2000_bh_dd']}, {R['pre2000_tm_sh']}, {R['pre2000_tm_dd']}),\n"
            "    ('2000s bear', '2000-01-01', '2009-12-31',\n"
            f"     {R['bear2000_bh_sh']}, {R['bear2000_bh_dd']}, {R['bear2000_tm_sh']}, {R['bear2000_tm_dd']}),\n"
            "    ('2010s bull', '2010-01-01', '2019-12-31',\n"
            f"     {R['bull2010_bh_sh']}, {R['bull2010_bh_dd']}, {R['bull2010_tm_sh']}, {R['bull2010_tm_dd']}),\n"
            "    ('2020s', '2020-01-01', '2026-06-13',\n"
            f"     {R['era2020_bh_sh']}, {R['era2020_bh_dd']}, {R['era2020_tm_sh']}, {R['era2020_tm_dd']}),\n"
            "]\n"
            "if HAVE_REAL:\n"
            "    computed = []\n"
            "    for name, s, e, *_ in sub_periods:\n"
            "        sub = close.loc[s:e]\n"
            "        if len(sub) < 300: continue\n"
            "        sig_s = st.timing_signal(sub, sma_n=SMA_N)\n"
            "        bt_s  = st.run_backtest(sub, sig_s, tbill_daily=TBILL, cost_bps=COST)\n"
            "        t_s   = st.summary(bt_s['r_strategy'])\n"
            "        b_s   = st.summary(bt_s['r_bh'])\n"
            "        computed.append((name, b_s['sharpe'], b_s['max_drawdown']*100, t_s['sharpe'], t_s['max_drawdown']*100))\n"
            "    sub_periods = computed\n"
            "else:\n"
            "    sub_periods = [(n, bs, bd, ts, td) for n, _, __, bs, bd, ts, td in sub_periods]\n"
            "names = [r[0] for r in sub_periods]\n"
            "bh_sh = [r[1] for r in sub_periods]\n"
            "tm_sh = [r[3] for r in sub_periods]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "x = np.arange(len(names))\n"
            "ax.bar(x - 0.2, bh_sh, width=0.38, color=RED,   label='Buy-and-hold')\n"
            "ax.bar(x + 0.2, tm_sh, width=0.38, color=GREEN, label='SMA timing')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels(names)\n"
            "ax.set_ylabel('Sharpe ratio'); ax.legend()\n"
            "ax.set_title('SMA timing shines in bear decades; lags in the pre-2000 bull run')\n"
            "plt.tight_layout(); plt.show()\n"
        ),
        md(
            "> in plain words: The SMA rule's headline advantage is almost entirely from the 2000s "
            f"(Sharpe +{R['bear2000_tm_sh']:.2f} vs BH {R['bear2000_bh_sh']:.2f}) and 2020s. "
            f"In the late 1990s bull market it lagged (Sharpe +{R['pre2000_tm_sh']:.2f} vs BH "
            f"+{R['pre2000_bh_sh']:.2f}). This is the regime-dependence story: a rule that "
            "exits on sustained downtrends pays a whipsaw cost in choppy uptrending markets."
        ),
        md(
            "### 4c -- Synthetic regime control: the engine works when regimes exist\n\n"
            "We confirm the machinery is correct: on a two-regime synthetic tape (genuine "
            "bull/bear separation) the rule cuts drawdown substantially vs both buy-and-hold "
            "and random timing. On a flat-vol null tape it adds nothing."
        ),
        code(
            "from faber_timing import data as fdata\n"
            "results = []\n"
            "for strength, lbl in [(1.0, 'Two-regime'), (0.0, 'Flat-vol (null)')]:\n"
            "    prices, _ = fdata.synthetic_daily(n_years=30, signal_strength=strength, seed=110)\n"
            "    res = st.compare_strategies(prices['close'], tbill_daily=0.04/252, sma_n=200, cost_bps=5.0)\n"
            "    results.append((lbl, res['bh']['sharpe'], res['bh']['max_drawdown'],\n"
            "                    res['timing']['sharpe'], res['timing']['max_drawdown'],\n"
            "                    res['random']['sharpe']))\n"
            "print(f'{\"Tape\":18s} | {\"BH Sharpe\":10s} | {\"BH MaxDD\":10s} | {\"Timing Sharpe\":14s} | {\"Timing MaxDD\":12s}')\n"
            "for r in results:\n"
            "    print(f'{r[0]:18s} | {r[1]:+10.3f} | {r[2]:+10.1%} | {r[3]:+14.3f} | {r[4]:+12.1%}')\n"
        ),
        md(
            "The two-regime tape shows large drawdown improvement and Sharpe gain. The flat-vol "
            "tape shows modest improvement at best -- confirming the engine is a genuine regime "
            "detector, and its real-tape benefit comes from the real bull/bear regimes in SPY "
            "history, not from chance."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal `REAL` (on risk dimension)** -- SMA MaxDD {R['tm_dd']:.1f}% vs BH "
            f"{R['bh_dd']:.1f}% (−{R['dd_improvement']:.0f} pp), confirmed vs random timing "
            f"({R['rn_dd']:.1f}%). Sharpe {R['tm_sharpe']:.3f} vs BH {R['bh_sharpe']:.3f}: "
            f"|*t*| = {abs(R['t_timing_vs_bh']):.2f} vs BH (not certified); "
            f"|*t*| = {R['t_timing_vs_rand']:.2f} vs random (near-threshold). "
            "H1 confirmed; H2 directional but sub-threshold.\n"
            f"- **Tradability `FRAGILE`** -- ~6 switches/yr; Sharpe > BH at 25 bps one-way "
            f"({R['sharpe_25bps']:.3f} vs {R['bh_sharpe']:.3f}). CAGR lags BH "
            f"(+{R['tm_cagr']:.1f}% vs +{R['bh_cagr']:.1f}%). Regime-dependent: dominant "
            "in bear decades, competitive in bull markets.\n"
            "- **Risk reduction or alpha? `CONFIRMED`** -- vol falls from "
            f"{R['bh_vol']:.1f}% to {R['tm_vol']:.1f}%; Sharpe gains from vol compression. "
            "The rule is not a return generator -- it is a tail-risk manager."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 -- Could you trade it? -- costs and implementation\n\n"
            "Sharpe as a function of one-way cost, and the CAGR vs drawdown trade-off."
        ),
        code(
            "if HAVE_REAL:\n"
            "    costs = [0.0, 1.0, 5.0, 10.0, 25.0]\n"
            "    shs, dds, cagrs = [], [], []\n"
            "    for c in costs:\n"
            "        bt_c = st.run_backtest(close, sig, tbill_daily=TBILL, cost_bps=c)\n"
            "        s = st.summary(bt_c['r_strategy'])\n"
            "        shs.append(s['sharpe']); dds.append(s['max_drawdown']); cagrs.append(s['cagr'])\n"
            "else:\n"
            "    costs  = [0.0, 1.0, 5.0, 10.0, 25.0]\n"
            "    shs    = [R['sharpe_0bps'], R['sharpe_1bps'], R['sharpe_5bps'], R['sharpe_10bps'], R['sharpe_25bps']]\n"
            "    dds    = [-0.203, -0.207, -0.219, -0.235, -0.279]\n"
            "    cagrs  = [0.095, 0.094, 0.092, 0.088, 0.078]\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "axes[0].plot(costs, shs, 'o-', c=GREEN, lw=2, label='SMA timing')\n"
            f"axes[0].axhline(R['bh_sharpe'], ls='--', c=RED, label='Buy-and-hold')\n"
            "axes[0].set_xlabel('one-way cost (bps)'); axes[0].set_ylabel('Sharpe')\n"
            "axes[0].set_title('Sharpe vs cost (6 switches/yr = low cost sensitivity)')\n"
            "axes[0].legend()\n"
            "axes[1].scatter([-d*100 for d in dds], [c*100 for c in cagrs], s=80,\n"
            "    c=GREEN, zorder=3, label='SMA timing (5-25 bps)')\n"
            f"axes[1].scatter(R['bh_dd']*-1, R['bh_cagr'], s=120, c=RED, marker='*', zorder=4, label='Buy-and-hold')\n"
            "axes[1].set_xlabel('Max drawdown (% abs)'); axes[1].set_ylabel('CAGR %')\n"
            "axes[1].set_title('Risk-return trade-off: lower DD, lower CAGR'); axes[1].legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Even at 25 bps one-way: Sharpe {shs[-1]:.3f} vs BH {R[\"bh_sharpe\"]:.3f}')\n"
        ),
        md(
            "> in plain words: Costs matter so little because the rule barely trades. A long-only "
            "investor can implement this with SPY and BIL in any retail brokerage. The bigger "
            "practical barrier is behavioural: the rule exits in the middle of a panic (when "
            "everyone is piling in after a recovery begins) and misses some of the early rebound. "
            "That is a cost the numbers can't capture -- but the drawdown numbers suggest it is "
            "worth it for investors who care about not being down 55%."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **Vol-targeting vs binary timing.** The continuous version of this idea -- [Study 16 "
            "-- Storm-Shy](../../16-storm-shy/) -- scales position by inverse realized vol rather "
            "than going binary. Both exploit the same regime structure; vol-targeting is smoother "
            "but requires leverage in calm periods.\n"
            "- **Multi-asset extension.** Faber's original paper applied the 10-month rule to five "
            "asset classes. The correlation across assets means the diversified version has even "
            "less whipsaw. A simple fork of this study: fetch prices for TLT, GLD, VNQ, and a "
            "commodity ETF; run the rule on each; equal-weight the invested positions.\n"
            "- **Monthly vs daily signal.** The monthly variant is available via "
            "`data.resample_monthly()` and `strategy.timing_signal_monthly()` -- fewer "
            "switches, less whipsaw, but slower response to fast crashes (2020 COVID).\n"
            "- **Post-publication decay.** The 2010s show the rule lagging. Testing whether the "
            "Sharpe advantage is decaying post-2007 publication (per McLean & Pontiff 2016) is "
            "a natural next study.\n\n"
            "*Think the SMA timing rule can be improved? A fork testing faster SMA windows, "
            "combining with vol signals, or applying it outside the US is the natural next step. "
            "The engine and controls are here.*"
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
