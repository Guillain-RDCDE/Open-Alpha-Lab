"""Generate the two narrative notebooks for Study 437 (Donchian-Breakout).

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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-23).
R = dict(
    n=8400, start="1993-01-29", end="2026-06-12", fp="bc9ada8a0d6f",
    in_mkt=62.2, n_switches=203,
    # four arms (SPY, net 2bps)
    bh_cagr=10.82, bh_sharpe=0.553, bh_vol=18.59, bh_dd=-55.19, bh_t=3.67,
    dc_cagr=5.16, dc_sharpe=0.465, dc_vol=10.81, dc_dd=-41.86, dc_t=2.90,
    sma_cagr=3.68, sma_sharpe=0.325, sma_vol=11.13, sma_dd=-44.33, sma_t=1.94,
    rn_cagr=3.72, rn_sharpe=0.258, rn_vol=14.16, rn_dd=-70.05, rn_t=1.57,
    # head-to-head HAC return-difference t-stats
    t_vs_bh=-2.32, t_vs_random=0.66, t_vs_sma=0.94,
    # permutation placebo (gross, block=20)
    perm_real_bps=2.05, perm_p=0.83,
    # cost sweep: Sharpe
    sharpe_0bps=0.477, sharpe_1bps=0.471, sharpe_2bps=0.465,
    sharpe_5bps=0.449, sharpe_10bps=0.420,
    # window robustness: Donchian Sharpe & t vs bh
    w10_sh=0.331, w10_t=-3.03, w20_sh=0.465, w20_t=-2.32,
    w55_sh=0.523, w55_t=-1.89, w100_sh=0.603, w100_t=-1.44,
    # long/short variant
    ls_sharpe=-0.015, ls_cagr=-0.27, ls_dd=-62.86, ls_t=-2.34,
    # panel (Donchian Sharpe, BH Sharpe, t vs bh)
    panel={
        "SPY": (0.465, 0.553, -2.32), "QQQ": (0.349, 0.385, -1.32),
        "IWM": (0.498, 0.461, -0.53), "TLT": (0.217, 0.255, -0.75),
        "GLD": (0.324, 0.553, -2.29), "USO": (0.183, -0.195, 1.57),
        "UUP": (-0.069, 0.197, -1.59),
    },
    # synthetic control
    syn_null_dc=-0.414, syn_null_bh=-0.295, syn_null_t=0.11, syn_null_p=0.92,
    syn_tr_dc=0.052, syn_tr_bh=-0.446, syn_tr_t=2.24, syn_tr_p=0.044,
)

# ---------------------------------------------------------------------------
# Shared analysis preamble
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))           # the study package
sys.path.insert(0, os.path.abspath("../../.."))     # repo root
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from donchian_breakout import data, strategy as st

TICKER = "SPY"
N      = 20
COST   = 2.0   # one-way bps

# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-23).
R = dict(
    n=8400, start="1993-01-29", end="2026-06-12", fp="bc9ada8a0d6f",
    in_mkt=62.2, n_switches=203,
    bh_cagr=10.82, bh_sharpe=0.553, bh_vol=18.59, bh_dd=-55.19, bh_t=3.67,
    dc_cagr=5.16, dc_sharpe=0.465, dc_vol=10.81, dc_dd=-41.86, dc_t=2.90,
    sma_cagr=3.68, sma_sharpe=0.325, sma_vol=11.13, sma_dd=-44.33, sma_t=1.94,
    rn_cagr=3.72, rn_sharpe=0.258, rn_vol=14.16, rn_dd=-70.05, rn_t=1.57,
    t_vs_bh=-2.32, t_vs_random=0.66, t_vs_sma=0.94,
    perm_real_bps=2.05, perm_p=0.83,
    sharpe_0bps=0.477, sharpe_1bps=0.471, sharpe_2bps=0.465,
    sharpe_5bps=0.449, sharpe_10bps=0.420,
    w10_sh=0.331, w10_t=-3.03, w20_sh=0.465, w20_t=-2.32,
    w55_sh=0.523, w55_t=-1.89, w100_sh=0.603, w100_t=-1.44,
    ls_sharpe=-0.015, ls_cagr=-0.27, ls_dd=-62.86, ls_t=-2.34,
    panel={
        "SPY": (0.465, 0.553, -2.32), "QQQ": (0.349, 0.385, -1.32),
        "IWM": (0.498, 0.461, -0.53), "TLT": (0.217, 0.255, -0.75),
        "GLD": (0.324, 0.553, -2.29), "USO": (0.183, -0.195, 1.57),
        "UUP": (-0.069, 0.197, -1.59),
    },
    syn_null_dc=-0.414, syn_null_bh=-0.295, syn_null_t=0.11, syn_null_p=0.92,
    syn_tr_dc=0.052, syn_tr_bh=-0.446, syn_tr_t=2.24, syn_tr_p=0.044,
)

HAVE_REAL = data.have_real(TICKER)
print("real SPY cache present:", HAVE_REAL)
"""

LOAD_REAL = """\
if HAVE_REAL:
    bars = data.load_real(TICKER)
    exp  = st.run_experiment(bars, n=N, cost_bps=COST, allow_short=False, random_seed=42)
    s_bh, s_dc, s_sma, s_rn = exp["bh"], exp["donchian"], exp["sma"], exp["random"]
    sig  = st.breakout_signal(bars, n=N, allow_short=False)
    bt   = st.run_backtest(bars, sig, cost_bps=COST)
    in_mkt = exp["in_market_frac"]
else:
    bars = sig = bt = None
    s_bh  = dict(cagr=R["bh_cagr"]/100, sharpe=R["bh_sharpe"], vol_ann=R["bh_vol"]/100, max_drawdown=R["bh_dd"]/100, tstat=R["bh_t"])
    s_dc  = dict(cagr=R["dc_cagr"]/100, sharpe=R["dc_sharpe"], vol_ann=R["dc_vol"]/100, max_drawdown=R["dc_dd"]/100, tstat=R["dc_t"])
    s_sma = dict(cagr=R["sma_cagr"]/100, sharpe=R["sma_sharpe"], vol_ann=R["sma_vol"]/100, max_drawdown=R["sma_dd"]/100, tstat=R["sma_t"])
    s_rn  = dict(cagr=R["rn_cagr"]/100, sharpe=R["rn_sharpe"], vol_ann=R["rn_vol"]/100, max_drawdown=R["rn_dd"]/100, tstat=R["rn_t"])
    exp   = dict(t_vs_bh=R["t_vs_bh"], t_vs_random=R["t_vs_random"], t_vs_sma=R["t_vs_sma"])
    in_mkt = R["in_mkt"]/100
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Donchian-Breakout -- does the 20-day channel hold up on its own?\n"
            "### The Turtles' famous entry, stripped to its skeleton and tested honestly\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Better_than_SMA%3F: Not_supported](https://img.shields.io/badge/Better_than_SMA%3F-Not_supported-8b949e?style=flat-square)\n\n"
            "In the 1980s, two traders turned beginners into millionaires with a system whose "
            "entry rule fit on an index card: **buy when price makes a new 20-day high; sell when "
            "it makes a new 20-day low.** That rule -- the Donchian channel breakout -- is still "
            "all over trading forums today, usually shorn of the rest of the Turtle machinery. "
            "This notebook asks the simple question: does the 20-day channel, *on its own*, beat "
            "just buying and holding the index?\n\n"
            "> **This is the plain-language layer.** Want the t-stats, the permutation placebo and "
            "the capacity maths? That's the companion, "
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
            f"| Does the breakout beat buy-and-hold? | **No.** Sharpe **{R['dc_sharpe']:.3f}** vs "
            f"**{R['bh_sharpe']:.3f}** -- and the gap is statistically real *against* it (*t* = {R['t_vs_bh']:+.2f}). |\n"
            f"| Is its timing better than a coin flip? | **No.** A block-shuffle placebo says random "
            f"in/out schedules with the same exposure beat the real rule **{R['perm_p']*100:.0f}%** of the time. |\n"
            f"| Is the channel better than a plain moving average? | **No.** It's a statistical tie with "
            f"an SMA(20) cross (*t* = {R['t_vs_sma']:+.2f}); both lose to buy-and-hold. |\n"
            f"| Does it do *anything*? | **Lower drawdown** ({R['dc_dd']:.0f}% vs {R['bh_dd']:.0f}%) -- "
            f"but only because it sits in cash {100-R['in_mkt']:.0f}% of the time. That's cheaper to buy "
            "with a static stock/cash split. |\n\n"
            "> The breakout's own track record looks positive only because it is long most of the time "
            "in a market that rises. Once you ask the honest question -- *better than just holding?* -- "
            "the answer is a clean no."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 -- The claim\n\n"
            "> *'Buy a new 20-day high; sell a new 20-day low. A new high means the trend is up, so "
            "you ride it; a new low means the trend has turned, so you step aside. This single rule "
            "-- the channel breakout -- was the entry engine of the Turtle traders, who turned a few "
            "thousand dollars into tens of millions.'*\n\n"
            "The logic is clean: a 20-day high is, by definition, evidence that the recent trend is "
            "up. If markets *trend* -- if a rise tends to beget another rise -- then entering on the "
            "breakout and riding it should harvest that momentum. The catch the folklore skips: the "
            "Turtles' result also leaned on volatility-based sizing, pyramiding, dozens of *futures* "
            "markets, and an exit rule. Here we isolate just the **20-day channel on one equity index** "
            "and ask whether *that piece* carries the magic."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 -- So what?\n\n"
            "If a parameter-light breakout genuinely beat buy-and-hold on the most liquid index on "
            "Earth, it would be close to a free lunch -- anyone could run it from a phone. And it "
            "would say something deep about markets: that a simple new-high signal predicts future "
            "returns, i.e. that prices trend in a way you can monetise.\n\n"
            "The honest bar is not 'does the rule make money' -- almost any long-biased rule makes "
            "money in a rising market. The bar is **does it beat just holding the index**, after "
            "costs, on a risk-adjusted basis. That's a much harder test, and it's the one that "
            "separates a real edge from repackaged beta."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 -- How would we even know?\n\n"
            "Four checks keep us honest:\n\n"
            "1. **The buy-and-hold race.** The breakout must beat just holding SPY, net of costs, "
            "on Sharpe -- not merely make a positive number.\n"
            "2. **The random-timing control.** A coin that's in the market the same fraction of days "
            "but on *random* days. If the breakout doesn't beat this, its 'timing' is empty.\n"
            "3. **The plain-SMA benchmark.** A simple 'price above its 20-day average' rule -- same "
            "window, same costs. If the fancy channel can't beat the simplest moving average, the "
            "channel machinery adds nothing.\n"
            "4. **The synthetic positive control.** A simulated tape where we *plant* a real trend. "
            "The rule must light up there (proving the test can detect a breakout edge) and stay "
            "quiet on a pure random walk."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 -- The teardown -- let's actually look\n\n"
            "**First, the big picture: who ends up where?**"
        ),
        code(
            "fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.0), sharex=True)\n"
            "if HAVE_REAL:\n"
            "    sig_sma = st.sma_cross_signal(bars, n=N)\n"
            "    sig_rn  = st.random_timing_signal(sig, seed=42)\n"
            "    bt_sma  = st.run_backtest(bars, sig_sma, cost_bps=COST)\n"
            "    bt_rn   = st.run_backtest(bars, sig_rn,  cost_bps=COST)\n"
            "    cum_bh  = np.exp(bt['r_bh'].cumsum())\n"
            "    cum_dc  = np.exp(bt['r_strategy'].cumsum())\n"
            "    cum_sma = np.exp(bt_sma['r_strategy'].cumsum())\n"
            "    cum_rn  = np.exp(bt_rn['r_strategy'].cumsum())\n"
            "    axes[0].plot(cum_bh.index,  cum_bh,  c=RED,   lw=1.8, label='Buy-and-hold')\n"
            "    axes[0].plot(cum_dc.index,  cum_dc,  c=GREEN, lw=1.8, label='Donchian-20 long/flat')\n"
            "    axes[0].plot(cum_sma.index, cum_sma, c=AMBER, lw=1.2, label='SMA(20) cross', ls='--')\n"
            "    axes[0].plot(cum_rn.index,  cum_rn,  c=GREY,  lw=1.0, label='Random timing', ls=':')\n"
            "    for col, lbl, c in [('r_bh','BH',RED), ('r_strategy','Donchian',GREEN)]:\n"
            "        cum = np.exp(bt[col].cumsum()); dd = cum/cum.cummax()-1\n"
            "        axes[1].fill_between(dd.index, dd, 0, alpha=0.3, color=c, label=lbl)\n"
            "else:\n"
            "    axes[0].text(0.5,0.5,'Cache not available\\nRun verify.py --fetch',ha='center',va='center',transform=axes[0].transAxes)\n"
            "axes[0].set_yscale('log'); axes[0].set_ylabel('Cumulative wealth (log)')\n"
            "axes[0].legend(); axes[0].set_title('SPY: the 20-day breakout trails buy-and-hold')\n"
            "axes[1].set_ylabel('Drawdown'); axes[1].legend(); plt.tight_layout(); plt.show()\n"
        ),
        md(
            f"Buy-and-hold (red) ends well above the breakout (green). The channel rule *does* ride a "
            f"shallower drawdown ({R['dc_dd']:.0f}% vs {R['bh_dd']:.0f}%) -- but look at the random-timing "
            f"line (grey dotted): being out of the market on *random* days actually produces the *worst* "
            f"drawdown of all ({R['rn_dd']:.0f}%). And the plain SMA (amber dashed) ends up roughly where "
            "the Donchian channel does. The fancy channel is buying you essentially what a simple moving "
            "average does, for less than just holding."
        ),
        md("**Now the honest scorecard -- Sharpe, drawdown, return for all four arms:**"),
        code(
            "labels = ['Buy-and-hold','Donchian-20','SMA(20)','Random']\n"
            "sh  = [s_bh['sharpe'], s_dc['sharpe'], s_sma['sharpe'], s_rn['sharpe']]\n"
            "dd  = [abs(s_bh['max_drawdown']), abs(s_dc['max_drawdown']), abs(s_sma['max_drawdown']), abs(s_rn['max_drawdown'])]\n"
            "cg  = [s_bh['cagr']*100, s_dc['cagr']*100, s_sma['cagr']*100, s_rn['cagr']*100]\n"
            "cols= [RED, GREEN, AMBER, GREY]\n"
            "fig, ax = plt.subplots(1,3,figsize=(12,4.3))\n"
            "for i,(vals,title) in enumerate([(sh,'Sharpe ratio'),(dd,'Max drawdown (abs)'),(cg,'CAGR (%)')]):\n"
            "    bars_ = ax[i].bar(labels, vals, color=cols, width=0.6)\n"
            "    ax[i].set_title(title); ax[i].tick_params(axis='x', rotation=25)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Donchian Sharpe {s_dc[\"sharpe\"]:.3f} vs BH {s_bh[\"sharpe\"]:.3f} '\n"
            "      f'(return-diff HAC t = {exp[\"t_vs_bh\"]:+.2f})')\n"
            "print(f'Donchian vs SMA(20): t = {exp[\"t_vs_sma\"]:+.2f}  |  vs random: t = {exp[\"t_vs_random\"]:+.2f}')\n"
        ),
        md(
            f"Buy-and-hold wins on Sharpe ({R['bh_sharpe']:.3f}) and CAGR ({R['bh_cagr']:.1f}%). The "
            f"breakout's only 'win' is lower drawdown -- and the random-timing arm shows that simply "
            "being out of the market on random days makes drawdown *worse*, so even that isn't timing "
            "skill, it's just exposure. The breakout's edge over a plain SMA is a statistical "
            f"non-event (*t* = {R['t_vs_sma']:+.2f})."
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal -- NONE.** The breakout loses to buy-and-hold on a risk-adjusted basis "
            f"(*t* = {R['t_vs_bh']:+.2f}), ties a coin flip with the same exposure, and a block-shuffle "
            f"placebo says random schedules beat it {R['perm_p']*100:.0f}% of the time. Its positive-looking "
            "track record is bull-market beta, not timing skill.\n"
            f"- **Tradability -- MIRAGE.** Even *gross* it trails buy-and-hold ({R['sharpe_0bps']:.3f} vs "
            f"{R['bh_sharpe']:.3f}); the short-selling version goes negative; it fails across a 7-asset panel.\n"
            f"- **Better than a plain SMA? -- NOT SUPPORTED.** Donchian channel vs SMA(20): *t* = "
            f"{R['t_vs_sma']:+.2f}. The channel's extra machinery buys nothing over the simplest filter."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 -- Could you actually trade it?\n\n"
            "There's no edge to monetise here, but it's worth showing *why* costs aren't even the "
            "problem -- the rule is cheap to run (only ~6 switches a year). It loses on merit, gross:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    costs = [0.0,1.0,2.0,5.0,10.0]; shs=[]\n"
            "    for c in costs:\n"
            "        e = st.run_experiment(bars, n=N, cost_bps=c, random_seed=42)\n"
            "        shs.append(e['donchian']['sharpe'])\n"
            "else:\n"
            "    costs=[0.0,1.0,2.0,5.0,10.0]\n"
            "    shs=[R['sharpe_0bps'],R['sharpe_1bps'],R['sharpe_2bps'],R['sharpe_5bps'],R['sharpe_10bps']]\n"
            "fig, ax = plt.subplots(figsize=(8.5,4.3))\n"
            "ax.plot(costs, shs, 'o-', c=GREEN, lw=2, label='Donchian-20')\n"
            "ax.axhline(R['bh_sharpe'], ls='--', c=RED, label=f'Buy-and-hold: {R[\"bh_sharpe\"]:.3f}')\n"
            "ax.set_xlabel('one-way cost (bps)'); ax.set_ylabel('Sharpe'); ax.legend()\n"
            "ax.set_title('Costs are not the problem -- the rule is below BH even at 0 bps')\n"
            "plt.tight_layout(); plt.show()\n"
        ),
        md(
            "The green line never reaches the red dashed buy-and-hold line, not even at zero cost. "
            "When a rule loses *gross*, transaction costs are beside the point -- there is no edge for "
            "costs to eat. The honest bottom line: you would have done better, with less effort and "
            "less tax drag, just holding the index."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **The rest of the Turtle.** The breakout is one piece; "
            "[Study 103 -- Turtle-Trader](../../103-turtle-trader/) runs the *full* system "
            "(ATR sizing + pyramiding + exit). Read together, they ask how much of the legend is the "
            "entry vs the rest.\n"
            "- **Slow channels and diversified futures.** Breakouts earn a real premium when spread "
            "across *many* trending markets (commodities, FX, rates) -- not on one equity index. "
            "Window matters too: the slowest channels here close the Sharpe gap to BH (but never "
            "clear it). A fork on a futures panel is the fair home for this rule.\n"
            "- **Why a slow SMA *does* work.** [Study 110 -- Faber-Timing](../../110-faber-timing/) "
            "shows the 200-day SMA genuinely reduces risk -- because it times volatility, which a "
            "fast 20-day channel doesn't. The contrast is the lesson.\n\n"
            "*Think the channel deserves a friendlier test? Fork this, point it at a diversified "
            "futures basket or a much slower window, and show it beats buy-and-hold there. That's the bar.*"
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
            "# Donchian-Breakout -- a quantitative teardown\n"
            "### Real daily tape - 20-day channel - HAC return-diff inference - block-permutation placebo - planted-edge control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Better_than_SMA%3F: Not_supported](https://img.shields.io/badge/Better_than_SMA%3F-Not_supported-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim now carrying its standard error.* We isolate the 20-day "
            "Donchian channel from the rest of the Turtle system and test, on 33 years of SPY plus a "
            "7-ETF panel, whether the breakout's *timing* carries information beyond bull-market beta.\n\n"
            "> **Not investment advice.** Real data: SPY daily total-return closes "
            "1993-01-29 to 2026-06-12 (n=8,400), `auto_adjust=True`. Methods in "
            "[`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The `> 💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT + "\n" + LOAD_REAL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Donchian Sharpe {R['dc_sharpe']:.3f} vs BH {R['bh_sharpe']:.3f}; "
            f"return-diff HAC *t* = **{R['t_vs_bh']:+.2f}** (loses); permutation placebo *p* = **{R['perm_p']:.2f}**; "
            f"vs random *t* = {R['t_vs_random']:+.2f}; vs SMA *t* = {R['t_vs_sma']:+.2f}. |\n"
            f"| **Tradability** | `MIRAGE` | Gross Sharpe {R['sharpe_0bps']:.3f} < BH {R['bh_sharpe']:.3f}; "
            f"long/short Sharpe {R['ls_sharpe']:+.3f}; fails the whole panel. |\n"
            f"| **Better than SMA?** | `NOT SUPPORTED` | Donchian vs SMA(20): *t* = {R['t_vs_sma']:+.2f} "
            "(indistinguishable); both lose to BH. |\n\n"
            "> 💡 In plain words: the breakout's own *t* of +2.90 is real -- but it measures the rule's "
            "return against *zero*, and the rule is long 62% of the time in a rising market, so of course "
            "it's positive. The bar that matters is the return *against the benchmark*, and there the "
            "channel loses to buy-and-hold and ties both a coin and a plain moving average."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 -- The claim, steelmanned\n\n"
            "Let $H_t^{(N)} = \\max_{1 \\le k \\le N} P_{t-k}$ and $L_t^{(N)} = \\min_{1 \\le k \\le N} P_{t-k}$ "
            "be the Donchian channel (prior-$N$ high/low, both lagged so $t$ only sees the past). "
            "The long/flat position is a state machine: $w_t = 1$ once $P_t > H_t^{(N)}$, held until "
            "$P_t < L_t^{(N)}$ sets $w_t = 0$. Net return "
            "$r_t^{\\text{strat}} = w_{t-1} r_t - c\\,|w_t - w_{t-1}|$ (one execution lag, one-way cost $c$). "
            "Three hypotheses:\n\n"
            "- **H1 (edge over beta).** $\\text{SR}(\\text{Donchian}) > \\text{SR}(\\text{BH})$ with "
            "|*t*| $\\ge$ 2 on the daily return difference.\n"
            "- **H2 (timing skill).** The breakout's days-in-market beat a block-shuffled schedule with "
            "the same exposure (permutation *p* small).\n"
            "- **H3 (channel > MA).** $\\text{SR}(\\text{Donchian}) > \\text{SR}(\\text{SMA}(N))$ at the "
            "same window/lag/cost.\n\n"
            f"All three fail: H1 *t* = {R['t_vs_bh']:+.2f} (wrong sign), H2 *p* = {R['perm_p']:.2f}, "
            f"H3 *t* = {R['t_vs_sma']:+.2f}."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 -- So what? -- what rides on each answer\n\n"
            "The channel breakout is the load-bearing entry of the most mythologised systematic "
            "experiment in trading (the Turtles). If the 20-day channel on a single liquid index beat "
            "buy-and-hold, it would be near-free alpha and strong evidence of monetisable trending in "
            "equity index prices. If it doesn't -- and especially if it can't beat a plain SMA -- then "
            "the Turtle legend's edge lives in the *rest* of the system (sizing, diversification across "
            "trending futures, pyramiding), not the breakout itself. That reattribution is the prize."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 -- How we'd know -- the protocol\n\n"
            "- **Signal.** Donchian(20) on SPY OHLC; channel lagged one bar (uses $\\le t-1$), position "
            "lagged one more (entered at $t+1$) -- one documented execution lag.\n"
            "- **Race.** Excess-vs-excess of the same flat-cash baseline; NET (one-way 2 bps × NAV on "
            "each switch; shorts pay 50 bps/yr borrow).\n"
            "- **Benchmarks.** Buy-and-hold; SMA(20) cross (same window/lag/cost); random-timing control "
            "(matched in-market fraction, random days).\n"
            "- **Inference.** HAC (Newey-West) *t* on each arm's mean and on the return *difference* "
            "(Jobson-Korkie); a 20-day **block-permutation placebo** for timing information.\n"
            "- **Positive control.** Synthetic AR(1) tape with a planted return-autocorrelation `edge` "
            "knob: must light up at `edge>0`, stay null at `edge=0`."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 -- The teardown"),
        md(
            "### 4a -- The four-arm net race\n\n"
            "The headline table: does the channel beat the index, a moving average, or a coin?"
        ),
        code(
            "print(f'In-market fraction: {in_mkt:.1%}')\n"
            "print(f'\\n{\"Arm\":14s}| {\"CAGR\":7s}| {\"Sharpe\":7s}| {\"Vol\":7s}| {\"MaxDD\":8s}| {\"own t\":6s}')\n"
            "print('-'*60)\n"
            "for lbl,s in [('Buy-and-hold',s_bh),('Donchian-20',s_dc),('SMA(20)',s_sma),('Random',s_rn)]:\n"
            "    print(f'{lbl:14s}| {s[\"cagr\"]:+.2%}| {s[\"sharpe\"]:+.3f}| {s[\"vol_ann\"]:.2%}| {s[\"max_drawdown\"]:+.2%}| {s[\"tstat\"]:+.2f}')\n"
            "print(f'\\nReturn-diff HAC t  Donchian vs BH:     {exp[\"t_vs_bh\"]:+.2f}  (loses)')\n"
            "print(f'Return-diff HAC t  Donchian vs random: {exp[\"t_vs_random\"]:+.2f}  (tied)')\n"
            "print(f'Return-diff HAC t  Donchian vs SMA(20):{exp[\"t_vs_sma\"]:+.2f}  (tied)')\n"
        ),
        md(
            "> 💡 In plain words: the breakout's own *t* (+2.90) clears 2 -- but that just says 'this "
            "rule makes money', which any long-biased rule does in a bull market. The three "
            "*difference* t-stats are the real test, and they say: worse than holding, no better than "
            "a coin, no better than a moving average."
        ),
        md(
            "### 4b -- The block-permutation placebo: is there timing information?\n\n"
            "Shuffle the position schedule in 20-day blocks (preserve the clustering of *being* "
            "positioned, destroy its alignment with returns) and ask how often a random schedule with "
            "the same exposure beats the real rule."
        ),
        code(
            "if HAVE_REAL:\n"
            "    perm = st.permutation_pvalue(bars, sig, n_perm=2000, block=N, cost_bps=0.0, seed=437)\n"
            "    p_val, real_bps = perm['p_value'], perm['real_mean_bps']\n"
            "else:\n"
            "    p_val, real_bps = R['perm_p'], R['perm_real_bps']\n"
            "print(f'Real rule mean (gross): {real_bps:+.3f} bps/day')\n"
            "print(f'Permutation p-value (2000 perms, block=20): {p_val:.3f}')\n"
            "print(f'=> {p_val*100:.0f}% of random same-exposure schedules beat the real Donchian schedule.')\n"
        ),
        md(
            f"> 💡 In plain words: *p* = {R['perm_p']:.2f} is about as null as it gets. If the channel were "
            "picking good entry days, the real rule would sit in the right tail (small *p*). Instead it "
            "sits in the middle-to-left: the breakout's chosen days are, if anything, slightly worse than "
            "random days with the same exposure. There is no timing signal in the 20-day high."
        ),
        md(
            "### 4c -- Window & cost sweeps -- is N=20 just unlucky?\n\n"
            "The classic 20 is the *worst* short window; only very slow channels approach BH -- and none "
            "clear it (all return-diff t-stats stay in BH's favour). Costs are not the issue (low turnover)."
        ),
        code(
            "windows = [10,20,55,100]\n"
            "if HAVE_REAL:\n"
            "    w_sh, w_t = [], []\n"
            "    for w in windows:\n"
            "        e = st.run_experiment(bars, n=w, cost_bps=COST, random_seed=42)\n"
            "        w_sh.append(e['donchian']['sharpe']); w_t.append(e['t_vs_bh'])\n"
            "else:\n"
            "    w_sh=[R['w10_sh'],R['w20_sh'],R['w55_sh'],R['w100_sh']]\n"
            "    w_t =[R['w10_t'], R['w20_t'], R['w55_t'], R['w100_t']]\n"
            "fig, ax = plt.subplots(1,2,figsize=(11,4.3))\n"
            "ax[0].plot(windows, w_sh, 'o-', c=GREEN, lw=2, label='Donchian Sharpe')\n"
            "ax[0].axhline(R['bh_sharpe'], ls='--', c=RED, label='Buy-and-hold')\n"
            "ax[0].set_xlabel('channel window N'); ax[0].set_ylabel('Sharpe'); ax[0].legend()\n"
            "ax[0].set_title('Slower channels close the gap but never clear it')\n"
            "ax[1].bar([str(w) for w in windows], w_t, color=[RED if t<0 else GREEN for t in w_t])\n"
            "ax[1].axhline(0, c='k', lw=1); ax[1].axhline(-2, ls=':', c=GREY)\n"
            "ax[1].set_xlabel('channel window N'); ax[1].set_ylabel('return-diff t vs BH')\n"
            "ax[1].set_title('Every window loses to BH (t stays negative)')\n"
            "plt.tight_layout(); plt.show()\n"
        ),
        md(
            "> 💡 In plain words: even hand-picking the friendliest window (N=100) only ties buy-and-hold, "
            "and a 100-day channel is no longer the '20-day breakout' of the folklore. There's no window "
            "where the *classic* rule wins."
        ),
        md(
            "### 4d -- The panel & the long/short blow-up\n\n"
            "Across 7 liquid tapes the rule beats BH on none with significance; adding the short leg "
            "(short new 20-day lows) destroys it outright -- equities trend up, the short side bleeds."
        ),
        code(
            "pan = R['panel']\n"
            "if HAVE_REAL:\n"
            "    pan = {}\n"
            "    for t in data.PANEL:\n"
            "        b = data.load_real(t)\n"
            "        e = st.run_experiment(b, n=N, cost_bps=COST, random_seed=42)\n"
            "        pan[t] = (e['donchian']['sharpe'], e['bh']['sharpe'], e['t_vs_bh'])\n"
            "names = list(pan); dc=[pan[t][0] for t in names]; bh=[pan[t][1] for t in names]\n"
            "x = np.arange(len(names)); fig,ax = plt.subplots(figsize=(9.5,4.3))\n"
            "ax.bar(x-0.2, bh, width=0.38, color=RED, label='Buy-and-hold')\n"
            "ax.bar(x+0.2, dc, width=0.38, color=GREEN, label='Donchian-20')\n"
            "ax.axhline(0,c='k',lw=1); ax.set_xticks(x); ax.set_xticklabels(names)\n"
            "ax.set_ylabel('Sharpe'); ax.legend(); ax.set_title('Donchian-20 vs BH across the panel')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('long/SHORT SPY (net): Sharpe %.3f  CAGR %.2f%%  maxDD %.1f%%  t_vs_bh %.2f' % (\n"
            "      R['ls_sharpe'], R['ls_cagr'], R['ls_dd'], R['ls_t']))\n"
        ),
        md(
            "### 4e -- Synthetic positive control: the engine *can* bank a real breakout\n\n"
            "On a deterministic AR(1) tape with a planted return-autocorrelation `edge`, the breakout "
            "must light up when trend exists and stay null on a random walk."
        ),
        code(
            "rows = []\n"
            "for edge, lbl in [(0.0,'Null (edge=0)'), (0.4,'Trend (edge=0.4)')]:\n"
            "    sb,_ = data.synthetic_panel(n_days=6000, edge=edge, seed=437)\n"
            "    e = st.run_experiment(sb, n=N, cost_bps=COST, random_seed=42)\n"
            "    sg = st.breakout_signal(sb, n=N)\n"
            "    pp = st.permutation_pvalue(sb, sg, n_perm=2000, block=N, seed=437)\n"
            "    rows.append((lbl, e['donchian']['sharpe'], e['bh']['sharpe'], e['t_vs_bh'], pp['p_value']))\n"
            "print(f'{\"Tape\":18s}| {\"Donchian SR\":12s}| {\"BH SR\":8s}| {\"t vs BH\":8s}| perm p')\n"
            "for r in rows:\n"
            "    print(f'{r[0]:18s}| {r[1]:+12.3f}| {r[2]:+8.3f}| {r[3]:+8.2f}| {r[4]:.3f}')\n"
        ),
        md(
            "> 💡 In plain words: with a real trend planted, the breakout beats buy-and-hold at *t* = +2.24 "
            "and the placebo lights up (*p* = 0.04). The harness works -- so SPY's null result is the "
            "engine correctly finding nothing, not the engine being blind. A planted *zero* edge stays null "
            "(*p* = 0.92), confirming the test can't manufacture significance from noise."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal `NONE`** -- return-diff HAC *t* vs BH = {R['t_vs_bh']:+.2f} (loses), vs random "
            f"= {R['t_vs_random']:+.2f}, vs SMA = {R['t_vs_sma']:+.2f}; block-permutation placebo "
            f"*p* = {R['perm_p']:.2f}. The own-mean *t* of {R['dc_t']:+.2f} is bull-market beta, not skill.\n"
            f"- **Tradability `MIRAGE`** -- gross Sharpe {R['sharpe_0bps']:.3f} < BH {R['bh_sharpe']:.3f}; "
            f"long/short Sharpe {R['ls_sharpe']:+.3f}; fails the whole panel. The only deliverable -- "
            f"drawdown {R['dc_dd']:.0f}% vs {R['bh_dd']:.0f}% -- is just lower beta, replicable with a static split.\n"
            f"- **Better than SMA? `NOT SUPPORTED`** -- Donchian vs SMA(20): *t* = {R['t_vs_sma']:+.2f}. "
            "Same window, same lag, same costs, statistically tied; the channel state machine adds nothing."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 -- Could you trade it? -- capacity & costs\n\n"
            "There is no edge to scale, but the cost geometry confirms the rule loses on *merit*, not "
            "on frictions: it sits below buy-and-hold even at zero cost."
        ),
        code(
            "if HAVE_REAL:\n"
            "    costs=[0.0,1.0,2.0,5.0,10.0]; shs=[]\n"
            "    for c in costs:\n"
            "        shs.append(st.run_experiment(bars, n=N, cost_bps=c, random_seed=42)['donchian']['sharpe'])\n"
            "else:\n"
            "    costs=[0.0,1.0,2.0,5.0,10.0]\n"
            "    shs=[R['sharpe_0bps'],R['sharpe_1bps'],R['sharpe_2bps'],R['sharpe_5bps'],R['sharpe_10bps']]\n"
            "fig,ax=plt.subplots(figsize=(8.5,4.3))\n"
            "ax.plot(costs,shs,'o-',c=GREEN,lw=2,label='Donchian-20')\n"
            "ax.axhline(R['bh_sharpe'],ls='--',c=RED,label=f'Buy-and-hold {R[\"bh_sharpe\"]:.3f}')\n"
            "ax.set_xlabel('one-way cost (bps)'); ax.set_ylabel('Sharpe'); ax.legend()\n"
            "ax.set_title('Below buy-and-hold at every cost, including 0 bps')\n"
            "plt.tight_layout(); plt.show()\n"
        ),
        md(
            "> 💡 In plain words: when a rule loses gross, costs and capacity are moot. The ~6 switches/yr "
            "make it cheap to run -- it just isn't worth running. A long-only holder beats it on return, "
            "Sharpe, and tax drag."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **Reattribute the Turtle legend.** [Study 103 -- Turtle-Trader](../../103-turtle-trader/) "
            "runs the full system. This study is the controlled subtraction; together they locate the "
            "Turtles' edge in sizing/diversification/exits, not the 20-day entry.\n"
            "- **The friendly home for breakouts.** Hurst-Ooi-Pedersen (2017) show trend rules earn a "
            "real, *diversified* premium across many futures. A fork applying Donchian-20/55 to a broad "
            "futures panel (commodities, FX, rates) is where the rule should be judged.\n"
            "- **Why a slow SMA works and a fast channel doesn't.** "
            "[Study 110 -- Faber-Timing](../../110-faber-timing/) lands `REAL` on the risk axis because "
            "the 200-day SMA times volatility; the 20-day channel doesn't reach that horizon. Window and "
            "mechanism, not the channel-vs-band distinction, drive the difference.\n"
            "- **Multiple-testing honesty.** Sullivan-Timmermann-White (1999) and Bajgrowicz-Scaillet "
            "(2012) show single-window breakout 'edges' evaporate under Reality-Check correction. Our "
            "permutation placebo is the same caution in miniature.\n\n"
            "*Think the channel earns its keep somewhere? Fork this onto a diversified futures basket and "
            "show a significant return-diff t-stat vs buy-and-hold. That's the bar.*"
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
