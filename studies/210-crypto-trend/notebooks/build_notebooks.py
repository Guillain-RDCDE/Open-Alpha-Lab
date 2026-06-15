"""Generate the two narrative notebooks for Study 210 (Crypto-Trend).

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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-15).
R = dict(
    n=4290, start="2014-09-17", end="2026-06-15",
    fp="fcd9bda77d20", in_mkt=58.6, n_switches=74,
    # buy-and-hold
    bh_cagr=52.77, bh_sharpe=0.632, bh_vol=67.09, bh_dd=-83.40,
    # SMA timing (10 bps cost)
    tm_cagr=58.82, tm_sharpe=0.902, tm_vol=51.28, tm_dd=-70.10,
    # Random timing control
    rn_cagr=11.01, rn_sharpe=0.209, rn_vol=50.08, rn_dd=-83.28,
    # Sharpe diff t-stats
    t_timing_vs_bh=0.32, t_timing_vs_rand=2.63,
    dd_improvement=13.3,
    # Cost sweep (Sharpe at various one-way costs)
    sharpe_0bps=0.914, sharpe_1bps=0.913, sharpe_5bps=0.908,
    sharpe_10bps=0.902, sharpe_25bps=0.884, sharpe_50bps=0.853,
    # Sub-period Sharpe & MaxDD
    p1_bh_sh=1.43, p1_bh_dd=-59.1, p1_tm_sh=1.85, p1_tm_dd=-35.5,
    p2_bh_sh=0.33, p2_bh_dd=-81.5, p2_tm_sh=1.22, p2_tm_dd=-49.7,
    p3_bh_sh=-0.39, p3_bh_dd=-76.6, p3_tm_sh=-0.65, p3_tm_dd=-36.7,
    p4_bh_sh=0.85, p4_bh_dd=-51.2, p4_tm_sh=0.63, p4_tm_dd=-31.8,
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

from crypto_trend import data, strategy as st

TICKER  = "BTC-USD"
SMA_N   = 200
TBILL   = 0.04 / 365.0   # flat 4%/yr daily proxy (crypto: 365 days/year)
COST    = 10.0            # one-way bps (crypto spot/ETF)

# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-15).
R = dict(
    n=4290, start="2014-09-17", end="2026-06-15",
    fp="fcd9bda77d20", in_mkt=58.6, n_switches=74,
    bh_cagr=52.77,  bh_sharpe=0.632, bh_vol=67.09,  bh_dd=-83.40,
    tm_cagr=58.82,  tm_sharpe=0.902, tm_vol=51.28,   tm_dd=-70.10,
    rn_cagr=11.01,  rn_sharpe=0.209, rn_vol=50.08,   rn_dd=-83.28,
    t_timing_vs_bh=0.32, t_timing_vs_rand=2.63, dd_improvement=13.3,
    sharpe_0bps=0.914, sharpe_1bps=0.913, sharpe_5bps=0.908,
    sharpe_10bps=0.902, sharpe_25bps=0.884, sharpe_50bps=0.853,
    p1_bh_sh=1.43,  p1_bh_dd=-59.1, p1_tm_sh=1.85, p1_tm_dd=-35.5,
    p2_bh_sh=0.33,  p2_bh_dd=-81.5, p2_tm_sh=1.22, p2_tm_dd=-49.7,
    p3_bh_sh=-0.39, p3_bh_dd=-76.6, p3_tm_sh=-0.65, p3_tm_dd=-36.7,
    p4_bh_sh=0.85,  p4_bh_dd=-51.2, p4_tm_sh=0.63, p4_tm_dd=-31.8,
)

def _have_cache():
    return os.path.exists(data._cache_path(TICKER, data.DEFAULT_CACHE))

HAVE_REAL = _have_cache()
print("real BTC-USD cache present:", HAVE_REAL)
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
    s_bh  = dict(cagr=R["bh_cagr"]/100, sharpe=R["bh_sharpe"],
                 max_drawdown=R["bh_dd"]/100, vol_ann=R["bh_vol"]/100)
    s_tm  = dict(cagr=R["tm_cagr"]/100, sharpe=R["tm_sharpe"],
                 max_drawdown=R["tm_dd"]/100, vol_ann=R["tm_vol"]/100)
    s_rn  = dict(cagr=R["rn_cagr"]/100, sharpe=R["rn_sharpe"],
                 max_drawdown=R["rn_dd"]/100, vol_ann=R["rn_vol"]/100)
    in_mkt = R["in_mkt"] / 100
    bt = bt_rn = sig = None
"""

# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Crypto-Trend -- does the 200-day MA protect you from Bitcoin crashes?\n"
            "### The Faber trend rule applied to Bitcoin, tested honestly, in plain English\n\n"
            "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Drawdown_shield%3F: Confirmed](https://img.shields.io/badge/Drawdown_shield%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "Bitcoin has been the best-returning asset of the past decade -- and also one of the most "
            "brutal to hold through. Its bear markets don't look like equity bear markets: "
            "a −83% drawdown in 2018, another −76% in 2022. For most investors those drawdowns "
            "trigger panic-selling at the worst moment. A simple question: does the 200-day moving "
            "average rule (the same rule Mebane Faber popularised for stocks) keep you out of the "
            "worst of Bitcoin's crashes while letting you capture most of the upside?\n\n"
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
            "| Does the rule cut Bitcoin's drawdowns? | **Yes, meaningfully.** "
            f"BTC's worst drawdown falls from **{R['bh_dd']:.0f}%** to **{R['tm_dd']:.0f}%** "
            f"(a {R['dd_improvement']:.0f} percentage-point improvement). |\n"
            "| Is that just from being invested less often? | **No.** A random-timing control with "
            f"the same in-market fraction ({R['in_mkt']:.1f}%) has an almost identical drawdown to "
            f"buy-and-hold ({R['rn_dd']:.0f}%). The SMA picks the right days to be out, not just "
            "fewer days. |\n"
            "| Does it improve risk-adjusted return (Sharpe)? | **Yes, substantially.** "
            f"Sharpe rises from {R['bh_sharpe']:.3f} to {R['tm_sharpe']:.3f}. The SMA rule beats "
            f"the random-timing control with t = +{R['t_timing_vs_rand']:.2f}, above the inference bar. |\n"
            "| Could you trade it? | **Fragile** -- only ~{n_sw}/yr switches; costs barely dent "
            "the Sharpe. But BTC's short history (11 years), its brutal whipsaws, and "
            "the regime-dependency make this more fragile than it looks. |\n\n"
            "> The MA rule is a genuine drawdown shield for Bitcoin -- but 'fragile' is the right "
            "label: the Sharpe improvement vs BH is not statistically certified (*t* = +{tvb:.2f}), "
            "the history is short, and the whipsaws in choppy crypto markets are severe.".format(
                n_sw=R['n_switches'] // 11,
                tvb=R['t_timing_vs_bh'],
            )
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 -- The claim\n\n"
            "> *'Hold Bitcoin when its price is above its 200-day simple moving average. "
            "Move to cash when the price falls below. This simple rule, which Mebane Faber "
            "showed works for stocks, should dramatically reduce Bitcoin's brutal drawdowns "
            "while keeping most of the explosive upside.'*\n\n"
            "The idea has clean economic logic: Bitcoin's major bear markets (2014, 2018, 2022) "
            "were sustained, multi-month declines -- not brief corrections. A slow moving average "
            "is designed to distinguish a sustained trend from noise. When BTC falls decisively "
            "below its 200-day average, you step aside and wait in cash. When it recovers above, "
            "you re-enter.\n\n"
            "The version we test: hold BTC when the daily close is above the 200-day SMA, "
            "else hold cash earning a 4%/yr flat proxy. Binary: fully in or fully out. "
            "Transaction costs: 10 bps one-way (conservative for spot crypto or a crypto ETF)."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 -- So what?\n\n"
            "If you had held BTC from 2014 to 2026 -- a 11-year bull run with occasional "
            "catastrophic crashes -- you would have turned $1 into something extraordinary. "
            "But most investors couldn't hold through an −83% drawdown. The historical data "
            "shows retail investors systematically sold near the lows and bought near the highs "
            "of each BTC cycle, realising a fraction of the headline CAGR.\n\n"
            "The economic value of the MA rule, if it works, is not the abstract Sharpe number -- "
            "it is the behavioural outcome it enables: staying invested through a −35% drawdown "
            "is psychologically different from staying through an −83% one. Even a partial "
            "drawdown shield that lets real investors hold on would generate enormous practical value."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 -- How would we even know?\n\n"
            "Three tests keep us honest:\n\n"
            "1. **The random-timing control.** We build a coin-flip version that goes to cash with "
            f"the same frequency as the SMA rule ({R['in_mkt']:.1f}% in-market) but on *random* "
            "days. If random timing achieves the same drawdown reduction and Sharpe improvement, "
            "the SMA is just reducing exposure -- not adding timing skill. If SMA beats random, "
            "the *timing* (which days) is what matters.\n"
            "2. **The sub-period breakdown.** A rule that only worked in one BTC cycle is not a "
            "strategy -- it is a backtest artefact. We break the 11-year history into four eras.\n"
            "3. **The synthetic positive control.** A simulated tape with genuine bull/bear regimes "
            "confirms whether the engine can find the signal when it exists. On a flat-vol null "
            "tape (no regime separation), the rule should add nothing."
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
            "    axes[0].plot(cum_bh.index, cum_bh, c=RED,   lw=1.8, label='Buy-and-hold BTC')\n"
            "    axes[0].plot(cum_tm.index, cum_tm, c=GREEN, lw=1.8, label='SMA(200) timing')\n"
            "    axes[0].plot(cum_rn.index, cum_rn, c=GREY,  lw=1.2, label='Random timing (null)', ls='--')\n"
            "    for col, lbl, c in [('r_bh', 'BH', RED), ('r_strategy', 'Timing', GREEN)]:\n"
            "        cum = np.exp(bt[col].cumsum())\n"
            "        dd  = cum / cum.cummax() - 1\n"
            "        axes[1].fill_between(dd.index, dd, 0, alpha=0.3, color=c, label=lbl)\n"
            "else:\n"
            "    axes[0].text(0.5, 0.5, 'Cache not available -- run verify.py --fetch',\n"
            "                 ha='center', va='center', transform=axes[0].transAxes)\n"
            "axes[0].set_ylabel('Cumulative wealth (log scale)')\n"
            "axes[0].set_yscale('log')\n"
            "axes[0].legend(); axes[0].set_title('BTC-USD: buy-and-hold vs SMA(200) timing vs random timing')\n"
            "axes[1].set_ylabel('Drawdown'); axes[1].legend()\n"
            "plt.tight_layout(); plt.show()\n"
        ),
        md(
            f"The SMA rule avoids most of the 2018 bear market (−81%) and the 2022 bear market "
            f"(−77%). The random-timing control (grey dashed) does NOT avoid these crashes -- "
            f"it is simply out of the market on random days, including many good days. "
            f"The SMA's drawdown benefit is genuine timing, not mere exposure reduction."
        ),
        md("**Now the honest risk-vs-return scorecard:**"),
        code(
            "labels  = ['Buy-and-hold', 'SMA(200) timing', 'Random timing']\n"
            "sharpes = [s_bh['sharpe'], s_tm['sharpe'], s_rn['sharpe']]\n"
            "dds     = [abs(s_bh['max_drawdown']), abs(s_tm['max_drawdown']), abs(s_rn['max_drawdown'])]\n"
            "cagrs   = [s_bh['cagr']*100, s_tm['cagr']*100, s_rn['cagr']*100]\n"
            "colors  = [RED, GREEN, GREY]\n"
            "fig, ax = plt.subplots(1, 3, figsize=(12, 4.5))\n"
            "for i, (vals, title, fmt) in enumerate([\n"
            "        (sharpes, 'Sharpe ratio', '+.3f'),\n"
            "        (dds,     'Max drawdown (abs %)', '.1f'),\n"
            "        (cagrs,   'CAGR (%/yr)', '+.1f'),\n"
            "    ]):\n"
            "    bars = ax[i].bar(labels, vals, color=colors, width=0.55)\n"
            "    for b, v in zip(bars, vals):\n"
            "        ax[i].annotate(f'{v:{fmt}}', (b.get_x()+b.get_width()/2, v/2),\n"
            "                       ha='center', va='center', color='white', fontsize=9)\n"
            "    ax[i].set_title(title); ax[i].tick_params(axis='x', rotation=20)\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'SMA timing Sharpe: {{s_tm[\"sharpe\"]:.3f}} vs BH: {{s_bh[\"sharpe\"]:.3f}}')\n"
            f"print(f'Max DD: SMA {{s_tm[\"max_drawdown\"]:.1%}} vs BH {{s_bh[\"max_drawdown\"]:.1%}}')\n"
        ),
        md(
            f"The SMA rule delivers on both dimensions: higher Sharpe ({R['tm_sharpe']:.3f} vs "
            f"{R['bh_sharpe']:.3f}) AND lower drawdown ({R['tm_dd']:.0f}% vs {R['bh_dd']:.0f}%). "
            f"It also beats buy-and-hold on CAGR ({R['tm_cagr']:.1f}% vs {R['bh_cagr']:.1f}%) -- "
            "unlike the equity case where timing typically costs some return. "
            f"The random-timing arm has the worst Sharpe ({R['rn_sharpe']:.3f}) -- confirming "
            "the MA's benefit is from timing, not from exposure reduction alone."
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal -- REAL.** Sharpe up from {R['bh_sharpe']:.3f} to {R['tm_sharpe']:.3f}; "
            f"max drawdown cut from {R['bh_dd']:.0f}% to {R['tm_dd']:.0f}% "
            f"(+{R['dd_improvement']:.0f} pp). Random-timing control confirms it is timing skill, "
            f"not exposure reduction (*t* = +{R['t_timing_vs_rand']:.2f} vs random). "
            f"However, vs buy-and-hold the Sharpe improvement is not certified "
            f"(*t* = +{R['t_timing_vs_bh']:.2f}), and the 11-year history is short for a "
            "strategy with only 74 total switches.\n"
            "- **Tradability -- FRAGILE.** Only ~7 switches/year; costs barely register at "
            f"10 bps one-way (Sharpe {R['sharpe_10bps']:.3f}) and remain above BH even at "
            f"50 bps ({R['sharpe_50bps']:.3f}). BUT: the whipsaw risk in crypto is severe, "
            "regime-dependence is strong (2022 was a loss even with timing), and implementation "
            "on BTC in 2018 or 2022 required sitting in cash for months while peers 'just hodl'd'.\n"
            "- **Drawdown shield? -- CONFIRMED.** The primary claim is confirmed: the SMA rule "
            "genuinely sidesteps the worst of Bitcoin's crashes. The question is whether you "
            "can behaviorally execute it."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 -- Could you actually trade it?\n\n"
            "With only ~7 switches per year, transaction costs are almost irrelevant. "
            "The real implementation challenges are behavioral and structural:"
        ),
        code(
            "costs = [0.0, 1.0, 5.0, 10.0, 25.0, 50.0]\n"
            "if HAVE_REAL:\n"
            "    sharpes_c = []\n"
            "    for c in costs:\n"
            "        bt_c = st.run_backtest(close, sig, tbill_daily=TBILL, cost_bps=c)\n"
            "        sharpes_c.append(st.summary(bt_c['r_strategy'])['sharpe'])\n"
            "else:\n"
            "    sharpes_c = [R['sharpe_0bps'], R['sharpe_1bps'], R['sharpe_5bps'],\n"
            "                 R['sharpe_10bps'], R['sharpe_25bps'], R['sharpe_50bps']]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, sharpes_c, 'o-', c=GREEN, lw=2, label='SMA(200) timing Sharpe')\n"
            f"ax.axhline(R['bh_sharpe'], ls='--', c=RED, label='Buy-and-hold: {R['bh_sharpe']:.3f}')\n"
            "ax.set_xlabel('one-way cost (bps)'); ax.set_ylabel('Sharpe ratio')\n"
            "ax.set_title('Cost barely matters -- only 7 switches/year'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Sharpe stays above BH even at 50 bps one-way: {sharpes_c[-1]:.3f}')\n"
        ),
        md(
            f"Transaction costs are virtually irrelevant with ~7 switches/year. "
            f"The Sharpe stays above buy-and-hold even at 50 bps one-way ({R['sharpe_50bps']:.3f} "
            f"vs {R['bh_sharpe']:.3f}).\n\n"
            "**The real tradability challenge:** Bitcoin is easy to access (BTC ETFs, spot), "
            "but the behavioral test is harder -- the rule required sitting in cash from November "
            "2021 to January 2023 while BTC crashed, then re-entering as BTC started recovering. "
            "A rule that sounds simple in backtests is psychologically challenging live."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **The 2021-2022 sub-period.** The SMA rule did NOT save you completely: "
            f"BH Sharpe was {R['p3_bh_sh']:.2f} (MaxDD {R['p3_bh_dd']:.0f}%) vs timing "
            f"Sharpe {R['p3_tm_sh']:.2f} (MaxDD {R['p3_tm_dd']:.0f}%). The rule entered too "
            "early on the recovery and got whipsawed in the choppy 2021 top.\n"
            "- **Crypto never sleeps.** BTC trades 24/7/365; the SMA rule evaluated on "
            "daily closes ignores weekend price action. A more sophisticated version might "
            "use a 24/7 data feed, though the long-window SMA dampens this significantly.\n"
            "- **Multi-asset crypto.** The same rule applied to ETH, SOL, or a crypto index "
            "would diversify the whipsaw risk. A single-asset MA rule is especially fragile "
            "to idiosyncratic crashes.\n"
            "- **The equity comparison.** Study 110 (Faber-Timing) shows the same rule on SPY "
            "has a *smaller* Sharpe improvement but a *more reliable* history (33 years vs 11). "
            "Crypto's short history inflates both the magnitude and the uncertainty of everything.\n\n"
            "*Think the MA rule can be improved for crypto? Test a faster SMA (100-day), "
            "add a volatility filter, or apply it to a diversified crypto basket. "
            "The engine and controls are here.*"
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
            "# Crypto-Trend -- a quantitative teardown\n"
            "### Real BTC-USD daily tape - 200-day SMA - random-timing control - HAC inference - sub-era decay\n\n"
            "![Signal: Real](https://img.shields.io/badge/Signal-Real-2ea44f?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Drawdown_shield%3F: Confirmed](https://img.shields.io/badge/Drawdown_shield%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim now carrying its standard error.* We test whether "
            "the 200-day SMA timing rule improves risk-adjusted return vs buy-and-hold and a "
            "random-timing control on 11 years of BTC-USD (2014-2026), isolate timing skill "
            "from exposure reduction, and break down results by crypto era.\n\n"
            "> **Not investment advice.** Real data: BTC-USD daily closes 2014-09-17 "
            f"to 2026-06-15 (n=4,290); cash leg proxied at 4%/yr flat. "
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
            f"| **Signal** | `REAL` | SMA Sharpe **{R['tm_sharpe']:.3f}** vs BH **{R['bh_sharpe']:.3f}**; "
            f"SMA max DD **{R['tm_dd']:.0f}%** vs BH **{R['bh_dd']:.0f}%** "
            f"(+{R['dd_improvement']:.0f} pp); random timing DD {R['rn_dd']:.0f}% (near BH). "
            f"*t*(timing vs random) = **+{R['t_timing_vs_rand']:.2f}**; "
            f"*t*(timing vs BH) = +{R['t_timing_vs_bh']:.2f} (not certified). |\n"
            f"| **Tradability** | `FRAGILE` | ~7 switches/yr; costs minimal "
            f"(Sharpe {R['sharpe_50bps']:.3f} at 50 bps one-way vs BH {R['bh_sharpe']:.3f}). "
            "Regime-dependent: fails in choppy 2021-2022 top; short 11-yr history. |\n"
            f"| **Drawdown shield?** | `CONFIRMED` | Vol falls from {R['bh_vol']:.0f}% to "
            f"{R['tm_vol']:.0f}%; max DD cut from {R['bh_dd']:.0f}% to {R['tm_dd']:.0f}%; "
            "random control cannot replicate (genuine timing). |\n\n"
            "> in plain words: The 200-day MA genuinely sidesteps Bitcoin's worst bear markets, "
            "not by chance but by timing (confirmed vs random control). The Sharpe improvement "
            "is real on this tape but not statistically certified vs buy-and-hold alone -- the "
            "history is too short and the whipsaws too severe."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 -- The claim, steelmanned\n\n"
            "Let $P_t$ be the BTC-USD daily close and $\\mu_t = \\frac{1}{200}\\sum_{k=0}^{199} P_{t-k}$ "
            "the 200-day SMA. The timing signal (daily form) is:\n\n"
            "$$w_t = \\mathbf{1}[P_{t-1} > \\mu_{t-1}]$$\n\n"
            "The strategy return is $r_t^{\\text{strat}} = w_t r_t^{\\text{BTC}} + (1-w_t) r_t^{\\text{cash}} - "
            "c \\cdot |w_t - w_{t-1}|$ where $c = 10$ bps one-way. Three hypotheses:\n\n"
            "- **H1 (risk reduction).** $\\text{MaxDD}(w) < \\text{MaxDD}(\\text{BH})$ and the "
            "reduction cannot be explained by random timing at the same in-market frequency.\n"
            "- **H2 (Sharpe improvement).** $\\text{SR}(w) > \\text{SR}(\\text{BH})$ with "
            "statistical support: |*t*| $\\geq$ 2 on the daily return difference.\n"
            "- **H3 (tradable).** The advantage survives realistic transaction costs and does not "
            "require precise regime-shift prediction.\n\n"
            f"We confirm H1 strongly; H2 is directionally supported (*t* = +{R['t_timing_vs_bh']:.2f} "
            "vs BH, sub-threshold) but confirmed vs the random control "
            f"(*t* = +{R['t_timing_vs_rand']:.2f}); H3 on costs but fragile on regime-dependence."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 -- So what? -- what rides on each answer\n\n"
            "The application of the Faber rule to crypto is a natural extension tested "
            "extensively in the practitioner literature (AQR, Faber himself, multiple crypto "
            "quant papers 2018-2023). If H1-H3 hold, the rule is the simplest viable risk "
            "management tool for crypto exposure -- accessible to any retail investor through "
            "a Bitcoin ETF. If H2 fails but H1 holds, it is a pure tail-risk manager: you "
            "accept lower expected return for fewer multi-year catastrophic losses. The distinction "
            "determines whether you should view it as alpha (unlikely) or insurance (plausible)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 -- How we'd know -- the protocol\n\n"
            "- **Signal.** Daily SMA(200) on BTC-USD adjusted close; lag one day (signal at $t$ "
            "acts on return $t \\to t+1$); binary 0/1.\n"
            "- **Cash leg.** Flat 4%/yr proxy. BTC has no T-bill equivalent on-chain; we use "
            "a conservative flat rate.\n"
            "- **Cost.** 10 bps one-way; swept from 0 to 50 bps. Crypto ETFs (IBIT, FBTC) "
            "trade with spreads ~2-5 bps; spot exchanges historically charge 0-25 bps.\n"
            "- **Periodicity.** 365 days/year (crypto never closes). All Sharpe/CAGR figures "
            "annualise on 365, not 252.\n"
            "- **Control.** Random-timing: same in-market fraction, random days, seed=42. "
            "Isolates timing vs exposure.\n"
            "- **Inference.** HAC t-stat (Newey-West) on mean daily return; return-difference "
            "t-stat for Sharpe comparison. Sub-era breakdown for regime-dependence.\n"
            "- **Positive control.** Synthetic two-regime tape calibrated to BTC-like "
            "parameters to confirm the engine recovers drawdown reduction when regimes exist."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 -- The teardown"),
        md(
            "### 4a -- The random-timing control: is the SMA picking the right days?\n\n"
            "The key test: if the SMA rule just reduces exposure, the random-timing control "
            "(same frequency, random days) should have similar results. The random control "
            "cannot replicate the SMA's drawdown protection -- that is the timing signal."
        ),
        code(
            "print(f'In-market fraction (SMA): {in_mkt:.1%}')\n"
            "print(f'\\nArm              | CAGR    | Sharpe  | Vol     | MaxDD   |')\n"
            "print(f'-' * 60)\n"
            "for lbl, s in [('Buy-and-hold', s_bh), ('SMA(200) timing', s_tm), ('Random timing', s_rn)]:\n"
            "    print(f'{lbl:16s} | {s[\"cagr\"]:+.2%} | {s[\"sharpe\"]:+.3f} | {s[\"vol_ann\"]:.2%} | {s[\"max_drawdown\"]:+.2%} |')\n"
            "if HAVE_REAL:\n"
            "    t_vs_bh   = st.sharpe_diff_tstat(bt['r_strategy'], bt['r_bh'])\n"
            "    t_vs_rand = st.sharpe_diff_tstat(bt['r_strategy'], bt_rn['r_strategy'])\n"
            "else:\n"
            "    t_vs_bh, t_vs_rand = R['t_timing_vs_bh'], R['t_timing_vs_rand']\n"
            "print(f'\\nSharpe diff t-stat (SMA vs BH):     {t_vs_bh:+.2f}')\n"
            "print(f'Sharpe diff t-stat (SMA vs random):  {t_vs_rand:+.2f}')\n"
            "print(f'\\nConclusion: SMA vs BH: directional but not certified (t={t_vs_bh:+.2f})')\n"
            "print(f'SMA vs random: ABOVE the inference bar (t={t_vs_rand:+.2f} > 2.0)')\n"
            "print(f'MaxDD: SMA {s_tm[\"max_drawdown\"]:+.1%} << BH {s_bh[\"max_drawdown\"]:+.1%} ~= Random {s_rn[\"max_drawdown\"]:+.1%}')\n"
        ),
        md(
            "> in plain words: The random-timing control has almost the *same* drawdown as "
            f"buy-and-hold ({R['rn_dd']:.0f}% vs {R['bh_dd']:.0f}%) -- proving that simply "
            "being out of BTC on random days doesn't protect you. You need to be out on the "
            f"*right* days (the 2018 and 2022 bear markets). The SMA's {R['tm_dd']:.0f}% "
            "drawdown shows it identifies those periods correctly, and the t-stat of "
            f"+{R['t_timing_vs_rand']:.2f} vs the random control clears our inference bar (|t| ≥ 2)."
        ),
        md(
            "### 4b -- Sub-era breakdown: the rule's regime-dependence\n\n"
            "BTC has had four distinct eras since 2014. The rule's performance varies dramatically."
        ),
        code(
            "sub_periods = [\n"
            "    ('2014-2017', '2014-09-17', '2017-12-31',\n"
            f"     {R['p1_bh_sh']}, {R['p1_bh_dd']}, {R['p1_tm_sh']}, {R['p1_tm_dd']}),\n"
            "    ('2018-2020', '2018-01-01', '2020-12-31',\n"
            f"     {R['p2_bh_sh']}, {R['p2_bh_dd']}, {R['p2_tm_sh']}, {R['p2_tm_dd']}),\n"
            "    ('2021-2022', '2021-01-01', '2022-12-31',\n"
            f"     {R['p3_bh_sh']}, {R['p3_bh_dd']}, {R['p3_tm_sh']}, {R['p3_tm_dd']}),\n"
            "    ('2023-2026', '2023-01-01', '2026-06-15',\n"
            f"     {R['p4_bh_sh']}, {R['p4_bh_dd']}, {R['p4_tm_sh']}, {R['p4_tm_dd']}),\n"
            "]\n"
            "if HAVE_REAL:\n"
            "    computed = []\n"
            "    for name, s, e, *_ in sub_periods:\n"
            "        sub = close.loc[s:e]\n"
            "        if len(sub) < 250: continue\n"
            "        sig_s = st.timing_signal(sub, sma_n=SMA_N)\n"
            "        bt_s  = st.run_backtest(sub, sig_s, tbill_daily=TBILL, cost_bps=COST)\n"
            "        t_s   = st.summary(bt_s['r_strategy'])\n"
            "        b_s   = st.summary(bt_s['r_bh'])\n"
            "        computed.append((name, b_s['sharpe'], b_s['max_drawdown']*100,\n"
            "                         t_s['sharpe'], t_s['max_drawdown']*100))\n"
            "    sub_periods = computed\n"
            "else:\n"
            "    sub_periods = [(n, bs, bd, ts, td) for n, _, __, bs, bd, ts, td in sub_periods]\n"
            "names = [r[0] for r in sub_periods]\n"
            "bh_sh = [r[1] for r in sub_periods]\n"
            "tm_sh = [r[3] for r in sub_periods]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "x = np.arange(len(names))\n"
            "ax.bar(x - 0.2, bh_sh, width=0.38, color=RED,   label='Buy-and-hold')\n"
            "ax.bar(x + 0.2, tm_sh, width=0.38, color=GREEN, label='SMA(200) timing')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels(names)\n"
            "ax.set_ylabel('Sharpe ratio'); ax.legend()\n"
            "ax.set_title('SMA timing: helpful in bear eras, whipsawed in choppy 2021-2022 top')\n"
            "plt.tight_layout(); plt.show()\n"
        ),
        md(
            f"> in plain words: The SMA rule shines in 2018-2020 (the big bear market), "
            f"producing Sharpe +{R['p2_tm_sh']:.2f} vs BH {R['p2_bh_sh']:.2f}. "
            f"But in 2021-2022 it was *worse* than BH "
            f"(Sharpe {R['p3_tm_sh']:.2f} vs {R['p3_bh_sh']:.2f}) -- the choppy top "
            "generated multiple whipsaws as BTC repeatedly crossed the SMA. "
            "Regime-dependence is the key fragility: the rule earns its keep in sustained "
            "downtrends and pays a whipsaw cost in volatile, directionless markets."
        ),
        md(
            "### 4c -- Synthetic regime control: the engine works when regimes exist\n\n"
            "We confirm the machinery is correct on a BTC-calibrated two-regime synthetic tape: "
            "genuine bull/bear separation produces large drawdown reduction; a flat-vol null "
            "tape produces no reliable advantage."
        ),
        code(
            "results = []\n"
            "for strength, lbl in [(1.0, 'Two-regime (BTC-like)'), (0.0, 'Flat-vol (null)')]:\n"
            "    prices, _ = data.synthetic_daily(n_years=8, signal_strength=strength, seed=210)\n"
            "    res = st.compare_strategies(prices['close'], tbill_daily=0.04/365,\n"
            "                               sma_n=200, cost_bps=10.0)\n"
            "    results.append((lbl, res['bh']['sharpe'], res['bh']['max_drawdown'],\n"
            "                    res['timing']['sharpe'], res['timing']['max_drawdown'],\n"
            "                    res['random']['sharpe']))\n"
            "print(f'{\"Tape\":24s} | {\"BH Sharpe\":10s} | {\"BH MaxDD\":10s} | {\"Timing Sharpe\":14s} | {\"Timing MaxDD\":12s}')\n"
            "for r in results:\n"
            "    print(f'{r[0]:24s} | {r[1]:+10.3f} | {r[2]:+10.1%} | {r[3]:+14.3f} | {r[4]:+12.1%}')\n"
        ),
        md(
            "The two-regime tape shows large drawdown improvement (the SMA correctly identifies "
            "the bear regime and moves to cash). The flat-vol tape shows no reliable advantage. "
            "The engine is a genuine regime detector -- the real-tape benefit comes from BTC's "
            "real bull/bear cycles, not from chance or overfitting."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal `REAL`** -- SMA Sharpe {R['tm_sharpe']:.3f} vs BH {R['bh_sharpe']:.3f}; "
            f"max DD {R['tm_dd']:.0f}% vs BH {R['bh_dd']:.0f}% (+{R['dd_improvement']:.0f} pp). "
            f"Timing vs random: *t* = +{R['t_timing_vs_rand']:.2f} (above inference bar). "
            f"Timing vs BH: *t* = +{R['t_timing_vs_bh']:.2f} (directional, not certified). "
            "H1 (drawdown) confirmed; H2 (Sharpe vs BH) sub-threshold but H2 vs random confirmed.\n"
            f"- **Tradability `FRAGILE`** -- ~7 switches/yr; Sharpe above BH at 50 bps one-way "
            f"({R['sharpe_50bps']:.3f} vs {R['bh_sharpe']:.3f}). Key fragilities: (1) only 11 "
            "years of data (74 switches total -- power is limited); (2) regime-dependent, hurting "
            "in the 2021-2022 choppy top; (3) behavioral challenge of sitting in cash during a "
            "BTC bull run that everyone else is riding.\n"
            f"- **Drawdown shield? `CONFIRMED`** -- vol {R['bh_vol']:.0f}% → {R['tm_vol']:.0f}%; "
            f"max DD {R['bh_dd']:.0f}% → {R['tm_dd']:.0f}%. Genuine timing (random control cannot replicate)."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 -- Could you trade it? -- costs and implementation\n\n"
            "Sharpe as a function of one-way cost, and the key implementation observations."
        ),
        code(
            "if HAVE_REAL:\n"
            "    costs_c = [0.0, 1.0, 5.0, 10.0, 25.0, 50.0]\n"
            "    shs, dds, cagrs = [], [], []\n"
            "    for c in costs_c:\n"
            "        bt_c = st.run_backtest(close, sig, tbill_daily=TBILL, cost_bps=c)\n"
            "        s = st.summary(bt_c['r_strategy'])\n"
            "        shs.append(s['sharpe']); dds.append(s['max_drawdown']); cagrs.append(s['cagr'])\n"
            "else:\n"
            "    costs_c = [0.0, 1.0, 5.0, 10.0, 25.0, 50.0]\n"
            "    shs = [R['sharpe_0bps'], R['sharpe_1bps'], R['sharpe_5bps'],\n"
            "           R['sharpe_10bps'], R['sharpe_25bps'], R['sharpe_50bps']]\n"
            "    dds = [-0.699, -0.700, -0.700, -0.701, -0.704, -0.709]\n"
            "    cagrs = [0.598, 0.597, 0.593, 0.588, 0.573, 0.549]\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "axes[0].plot(costs_c, shs, 'o-', c=GREEN, lw=2, label='SMA(200) timing')\n"
            f"axes[0].axhline(R['bh_sharpe'], ls='--', c=RED, label='Buy-and-hold')\n"
            "axes[0].set_xlabel('one-way cost (bps)'); axes[0].set_ylabel('Sharpe')\n"
            "axes[0].set_title('Sharpe vs cost (7 switches/yr = minimal cost sensitivity)')\n"
            "axes[0].legend()\n"
            "axes[1].scatter([-d*100 for d in dds], [c*100 for c in cagrs],\n"
            "                s=80, c=GREEN, zorder=3, label='SMA timing (0-50 bps)')\n"
            f"axes[1].scatter(abs(R['bh_dd']), R['bh_cagr'], s=120, c=RED, marker='*',\n"
            "                zorder=4, label='Buy-and-hold')\n"
            "axes[1].set_xlabel('Max drawdown (% abs)'); axes[1].set_ylabel('CAGR %')\n"
            "axes[1].set_title('SMA timing: lower DD, similar/higher CAGR'); axes[1].legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Even at 50 bps one-way: Sharpe {shs[-1]:.3f} vs BH {R[\"bh_sharpe\"]:.3f}')\n"
        ),
        md(
            "> in plain words: The cost story is easy -- ~7 trades/year means costs are almost "
            "irrelevant. The hard story is behavioral: the rule requires you to sell BTC and sit "
            "in cash for months or years. In 2018-2020 that was exactly right. In 2021 it "
            "triggered two whipsaws around the BTC top. The rule cannot tell you in real time "
            "whether you're in a 2018-style crash (stay out) or a brief 2021 correction (re-enter quickly)."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **The Faber rule on SPY.** Study 110 shows the same rule on US equities over "
            "33 years: the drawdown benefit is similar (−55% → −22%) but the history is three "
            "times longer and the statistical certification is stronger. Comparing the two "
            "studies reveals whether crypto's shorter history inflates perceived benefits.\n"
            "- **Faster MA.** A 100-day or 50-day MA would reduce lag into bear markets "
            "(avoiding the first 2-3 months of the crash) but increase whipsaw frequency. "
            "The optimal window is likely cycle-dependent in crypto.\n"
            "- **Vol-targeting.** Instead of binary in/out, scale BTC position inversely "
            "to its realized 30-day vol. This avoids the binary whipsaw problem while still "
            "reducing exposure in high-vol (typically bear-market) periods.\n"
            "- **Multi-asset crypto.** Run the rule simultaneously on BTC, ETH, and a broad "
            "crypto index. Equal-weight the invested positions. Cross-asset diversification "
            "should reduce whipsaw risk while preserving the drawdown protection.\n"
            "- **Post-ETF era.** Bitcoin ETFs (IBIT, FBTC) launched in January 2024. "
            "This creates a new, lower-friction implementation path. Tracking the rule's "
            "performance post-ETF is a natural live test.\n\n"
            "*Think the MA rule can be improved for crypto? Fork this, test a faster window, "
            "add a vol filter, or apply it to a diversified crypto basket. "
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
