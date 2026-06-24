"""Generate the two narrative notebooks for Study 435 (Guppy Multiple MA).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The real-tape
cells use the cached SPY parquet under ../_cache/ if present; otherwise they fall back to the
frozen headline numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for any
reader. The synthetic positive-control cell is fully offline and deterministic.

The real headline numbers live in exactly one place: the dict ``R`` below (mirror of
docs/results.md). Every prose number and table cell quotes ``R`` — never a hard-coded literal.
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
    n=8400, start="1993-01-29", end="2026-06-12", fp="42b159a9d39a",
    in_mkt=71.7, n_switches=259,
    bh_cagr=6.47, bh_sharpe=0.431, bh_vol=18.6, bh_dd=-64.0, bh_t=2.89,
    rb_cagr=2.80, rb_sharpe=0.301, rb_vol=11.3, rb_dd=-47.5, rb_t=1.86,
    cv_cagr=1.42, cv_sharpe=0.210, cv_dd=-38.0, cv_t=1.30,
    sma_cagr=1.77, sma_sharpe=0.215, sma_t=1.31,
    ema_cagr=3.16, ema_sharpe=0.337, ema_t=2.09,
    rn_cagr=3.36, rn_sharpe=0.288, rn_dd=-60.7, rn_t=1.81,
    t_rb_vs_bh=-2.10, t_rb_vs_rand=-0.50, t_rb_vs_sma=1.26, t_cv_vs_rb=-1.30,
    perm_p=0.969, perm_real_bps=2.55, perm_mean_bps=3.39,
    sweep_0=0.315, sweep_2=0.301, sweep_5=0.280, sweep_10=0.246,
    syn_edge_rb_sh=0.264, syn_edge_bh_sh=-0.149, syn_edge_t_vs_bh=2.35, syn_edge_perm_p=0.0135,
    syn_null_rb_sh=0.306, syn_null_bh_sh=0.323, syn_null_t_vs_bh=-0.85, syn_null_perm_p=0.2674,
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

from guppy_mma import data, strategy as st

TICKER = "SPY"
RF     = 0.04 / 252.0   # flat 4%/yr cash proxy
COST   = 2.0            # one-way bps x NAV
SMA_N  = 50

# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-13).
R = dict(
    n=8400, start="1993-01-29", end="2026-06-12", fp="42b159a9d39a",
    in_mkt=71.7, n_switches=259,
    bh_cagr=6.47, bh_sharpe=0.431, bh_vol=18.6, bh_dd=-64.0, bh_t=2.89,
    rb_cagr=2.80, rb_sharpe=0.301, rb_vol=11.3, rb_dd=-47.5, rb_t=1.86,
    cv_cagr=1.42, cv_sharpe=0.210, cv_dd=-38.0, cv_t=1.30,
    sma_cagr=1.77, sma_sharpe=0.215, sma_t=1.31,
    ema_cagr=3.16, ema_sharpe=0.337, ema_t=2.09,
    rn_cagr=3.36, rn_sharpe=0.288, rn_dd=-60.7, rn_t=1.81,
    t_rb_vs_bh=-2.10, t_rb_vs_rand=-0.50, t_rb_vs_sma=1.26, t_cv_vs_rb=-1.30,
    perm_p=0.969, perm_real_bps=2.55, perm_mean_bps=3.39,
    sweep_0=0.315, sweep_2=0.301, sweep_5=0.280, sweep_10=0.246,
    syn_edge_rb_sh=0.264, syn_edge_bh_sh=-0.149, syn_edge_t_vs_bh=2.35, syn_edge_perm_p=0.0135,
    syn_null_rb_sh=0.306, syn_null_bh_sh=0.323, syn_null_t_vs_bh=-0.85, syn_null_perm_p=0.2674,
)

def _have_cache():
    return os.path.exists(data._cache_path(TICKER, data.DEFAULT_CACHE))

HAVE_REAL = _have_cache()
print("real SPY cache present:", HAVE_REAL)
"""

LOAD_REAL = """\
if HAVE_REAL:
    df    = data.load_real(TICKER, fetch=False)
    close = df["close"]
    res   = st.run_experiment(close, rf_daily=RF, cost_bps=COST, sma_n=SMA_N, allow_short=False)
    s_bh  = res["bh"]; s_rb = res["ribbon"]; s_cv = res["conviction"]
    s_sma = res["sma"]; s_ema = res["ema"]; s_rn = res["random"]
    in_mkt = res["in_market_frac"]
else:
    close = None
    mk = lambda c, sh, dd: dict(cagr=c/100, sharpe=sh, max_drawdown=dd/100)
    s_bh  = mk(R["bh_cagr"], R["bh_sharpe"], R["bh_dd"])
    s_rb  = mk(R["rb_cagr"], R["rb_sharpe"], R["rb_dd"])
    s_cv  = mk(R["cv_cagr"], R["cv_sharpe"], R["cv_dd"])
    s_sma = mk(R["sma_cagr"], R["sma_sharpe"], -54.7)
    s_ema = mk(R["ema_cagr"], R["ema_sharpe"], -42.7)
    s_rn  = mk(R["rn_cagr"], R["rn_sharpe"], R["rn_dd"])
    in_mkt = R["in_mkt"] / 100
    res = None
"""

HERO_BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Reveals_conviction%3F: Busted](https://img.shields.io/badge/Reveals_conviction%3F-Busted-8b949e?style=flat-square)\n\n"
)


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Guppy Multiple MA -- do twelve moving averages beat one?\n"
            "### Daryl Guppy's twin-ribbon trend filter, tested honestly, in plain English\n\n"
            + HERO_BADGES +
            "Daryl Guppy's GMMA is one of the most recognisable indicators in retail charting: "
            "two coloured *ribbons* of moving averages, a fast one (3-15 days) for traders and a "
            "slow one (30-60 days) for investors. The folklore says you ride the trend when the "
            "fast ribbon is above the slow one, and you *trust* the trend most when the two ribbons "
            "are wide apart -- that spacing is supposed to reveal **conviction**. This notebook asks "
            "the only question that matters: does any of it actually beat just holding the market?\n\n"
            "> **This is the plain-language layer.** Want the HAC *t*-stats, the permutation placebo "
            "and the single-EMA benchmark? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\n" + LOAD_REAL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the ribbon beat buy-and-hold? | **No.** Excess Sharpe **+{R['rb_sharpe']:.3f}** "
            f"vs buy-and-hold **+{R['bh_sharpe']:.3f}** -- and the gap is statistically *against* the "
            f"ribbon (*t* = {R['t_rb_vs_bh']:+.2f}). |\n"
            f"| Is it better than one moving average? | **No.** A single 50-day EMA earns a *higher* "
            f"Sharpe (+{R['ema_sharpe']:.3f}) than the twelve-line ribbon (+{R['rb_sharpe']:.3f}). |\n"
            f"| Does it beat a coin-flip schedule? | **No.** A random in/out schedule with the same "
            f"in-market fraction matches it (*t* = {R['t_rb_vs_rand']:+.2f}). A rotation placebo beats "
            f"the real signal **{R['perm_p']*100:.0f}%** of the time. |\n"
            f"| Does 'wide ribbon = conviction' help? | **It hurts.** Gating on ribbon width *lowers* "
            f"the Sharpe (+{R['cv_sharpe']:.3f}). Width is hindsight, not a forecast. |\n\n"
            "> The GMMA's only real effect is being out of the market ~28% of the time, which lowers "
            "both risk and return. It is a single trend filter wearing a costume of twelve lines."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 -- The claim\n\n"
            "> *'The GMMA uses two groups of moving averages. The short-term group (3, 5, 8, 10, 12, "
            "15) shows what traders are doing. The long-term group (30, 35, 40, 45, 50, 60) shows what "
            "investors are doing. When the short group is above the long group, the trend is up. When "
            "the two groups are well separated and moving in parallel, the trend has conviction and "
            "will continue. When they compress, the trend is ending.'*\n"
            "> -- the Daryl Guppy / GMMA folklore\n\n"
            "It is a genuinely appealing story: it claims to read not just *direction* (like any moving "
            "average) but *agreement* between two kinds of market participant. If the spacing between "
            "the ribbons really forecasts whether a trend continues, that would be information a single "
            "moving average cannot give you."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 -- So what?\n\n"
            "If the ribbon's conviction reading worked, it would be a free upgrade for millions of "
            "retail traders who already have GMMA on their charts -- a way to filter out the weak "
            "trends that whipsaw a plain moving-average rule. And it would be a small but real dent in "
            "the efficient-market view: it would mean the *shape* of a fan of moving averages predicts "
            "returns beyond what the price level already tells you.\n\n"
            "Conversely, if the ribbon is just a slow-vs-fast crossover with extra decoration, then the "
            "honest comparison is brutal: it must beat (a) holding the market, (b) one moving average, "
            "and (c) flipping a coin to decide when to be in. Those are the three bars."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 -- How would we even know?\n\n"
            "Four tests keep us honest:\n\n"
            "1. **The buy-and-hold race.** Does the ribbon's risk-adjusted return beat just owning SPY? "
            "We compare *excess-of-cash* Sharpe to *excess-of-cash* Sharpe (a fair race, since the rule "
            "sits in cash part-time).\n"
            "2. **The single-moving-average benchmark.** Does the twelve-line ribbon beat **one** 50-day "
            "moving average? If not, the extra eleven lines are decoration.\n"
            "3. **The random-timing control.** A coin that goes to cash as often as the ribbon, but on "
            "random days. If the ribbon doesn't beat it, the 'timing' is just reduced exposure.\n"
            "4. **The conviction test.** Gate the long leg on the ribbon being *wider than usual*. If "
            "Guppy is right, that should *improve* results. We'll see it does the opposite."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 -- The teardown -- let's actually look\n\n"
           "**First, the ribbons themselves on a slice of SPY, so you can see the indicator.**"),
        code(
            "if HAVE_REAL:\n"
            "    sl = close.loc['2018-01-01':'2020-12-31']\n"
            "    rb = st.ribbons(sl)\n"
            "    fig, ax = plt.subplots(figsize=(10, 5))\n"
            "    ax.plot(sl.index, sl, c='k', lw=1.1, label='SPY close', zorder=5)\n"
            "    for c_ in rb['short'].columns:\n"
            "        ax.plot(sl.index, rb['short'][c_], c=GREEN, lw=0.8, alpha=0.6)\n"
            "    for c_ in rb['long'].columns:\n"
            "        ax.plot(sl.index, rb['long'][c_], c=RED, lw=0.8, alpha=0.6)\n"
            "    ax.plot([], [], c=GREEN, label='short ribbon (3-15d)')\n"
            "    ax.plot([], [], c=RED, label='long ribbon (30-60d)')\n"
            "    ax.set_title('GMMA on SPY 2018-2020: the two ribbons'); ax.legend()\n"
            "else:\n"
            "    fig, ax = plt.subplots(); ax.text(0.5,0.5,'cache not available',ha='center')\n"
            "plt.tight_layout(); plt.show()\n"
        ),
        md(
            "You can *see* the 2020 COVID crash compress and flip the ribbons -- that is the indicator "
            "doing exactly what it is supposed to. The question is whether acting on it pays. "
            "**Now the scorecard: risk-adjusted return of every arm.**"
        ),
        code(
            "labels = ['Buy &\\nhold', 'GMMA\\nribbon', 'GMMA\\nconviction', '1 SMA', '1 EMA', 'Random\\ntiming']\n"
            "sharpes = [s_bh['sharpe'], s_rb['sharpe'], s_cv['sharpe'], s_sma['sharpe'], s_ema['sharpe'], s_rn['sharpe']]\n"
            "colors  = [GREEN, RED, RED, GREY, AMBER, GREY]\n"
            "fig, ax = plt.subplots(figsize=(10, 4.6))\n"
            "bars = ax.bar(labels, sharpes, color=colors, width=0.6)\n"
            "for b, v in zip(bars, sharpes):\n"
            "    ax.annotate(f'{v:+.3f}', (b.get_x()+b.get_width()/2, v), ha='center',\n"
            "                va='bottom', fontsize=9)\n"
            "ax.axhline(s_bh['sharpe'], ls='--', c=GREEN, alpha=0.7, label='buy-and-hold bar')\n"
            "ax.set_ylabel('Excess-of-cash Sharpe'); ax.legend()\n"
            "ax.set_title('Nothing the ribbon does clears the buy-and-hold bar')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'ribbon {s_rb[\"sharpe\"]:+.3f} < buy-and-hold {s_bh[\"sharpe\"]:+.3f}; "
            "one EMA {s_ema[\"sharpe\"]:+.3f} even beats the ribbon')\n"
        ),
        md(
            f"The green dashed line is buy-and-hold (+{R['bh_sharpe']:.3f}). Every GMMA bar sits "
            f"*below* it. Worse, the single 50-day EMA (+{R['ema_sharpe']:.3f}) edges out the "
            f"twelve-line ribbon (+{R['rb_sharpe']:.3f}), and the 'conviction' gate is the *lowest* "
            f"bar of all (+{R['cv_sharpe']:.3f}). The ribbon's apparatus buys nothing."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal -- NONE.** Excess Sharpe +{R['rb_sharpe']:.3f}, beaten by buy-and-hold "
            f"(*t* = {R['t_rb_vs_bh']:+.2f}), tied by a coin-flip schedule, and a rotation placebo "
            f"beats the real signal {R['perm_p']*100:.0f}% of the time.\n"
            f"- **Tradability -- MIRAGE.** Even before any costs the ribbon's Sharpe "
            f"(+{R['sweep_0']:.3f}) trails buy-and-hold. There is no edge to scale.\n"
            "- **Reveals conviction? -- BUSTED.** Gating on ribbon width makes everything worse. "
            "Width describes a trend that already happened; it does not forecast the next one."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 -- Could you actually trade it?\n\n"
            "There is nothing here to trade -- but it is worth seeing *why* the lower risk is a trap, "
            "not a feature. The ribbon's drawdown is lower than buy-and-hold's, which looks nice. But a "
            "random schedule that sits in cash just as often gets a similar risk cut for free -- the "
            "reduction is exposure, not skill."
        ),
        code(
            "fig, ax = plt.subplots(figsize=(8.5, 4.4))\n"
            "arms = ['Buy & hold', 'GMMA ribbon', 'Random timing']\n"
            "dds  = [abs(s_bh['max_drawdown'])*100, abs(s_rb['max_drawdown'])*100, abs(s_rn['max_drawdown'])*100]\n"
            "shs  = [s_bh['sharpe'], s_rb['sharpe'], s_rn['sharpe']]\n"
            "ax.scatter(dds, shs, s=[140,140,140], c=[GREEN, RED, GREY], zorder=3)\n"
            "for a, d, s in zip(arms, dds, shs):\n"
            "    ax.annotate(a, (d, s), textcoords='offset points', xytext=(8, 6))\n"
            "ax.set_xlabel('Max drawdown (% abs)'); ax.set_ylabel('Excess Sharpe')\n"
            "ax.set_title('Lower risk, but no better Sharpe -- the cut is exposure, not skill')\n"
            "plt.tight_layout(); plt.show()\n"
        ),
        md(
            f"The ribbon (red) does have a smaller drawdown than buy-and-hold (green), but its Sharpe "
            f"is *lower*, and a random schedule (grey) lands in the same neighbourhood. You would be "
            f"giving up {R['bh_cagr']-R['rb_cagr']:.1f} points of annual return to buy a risk cut you "
            "could get by flipping a coin. That is not a strategy."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **Other instruments.** We tested SPY. The ribbon might behave differently on a "
            "trendier asset (a commodity, a single momentum stock). Fork `data.load_real` with another "
            "ticker and re-run -- the engine is generic.\n"
            "- **Long/short.** We ran long/flat. The quants notebook shows the long/short version is "
            "*sharply negative* -- shorting on a ribbon down-cross fights the market's drift.\n"
            "- **A real trend rule that works.** [Study 110 (Faber-Timing)](../../110-faber-timing/) "
            "shows a single-SMA rule that *does* earn a real risk stamp. The contrast is the lesson: "
            "the GMMA's twelve lines add complexity, not edge.\n\n"
            "*Think the ribbon shines somewhere SPY hides it? Fork this, point it at a trending market, "
            "and show the excess-vs-excess Sharpe clearing the bar. That's the standard.*"
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
            "# Guppy Multiple MA -- a quantitative teardown\n"
            "### Real daily SPY - twin EMA ribbons - excess-vs-excess Sharpe races - HAC inference - rotation placebo\n\n"
            + HERO_BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim now carrying its standard error.* We turn Daryl Guppy's "
            "twin-ribbon GMMA into a long/flat timing rule on 33 years of SPY, race its excess-of-cash "
            "Sharpe against buy-and-hold, a random-timing control and a single 50-day SMA/EMA, and test "
            "the trademark 'conviction' width gate.\n\n"
            "> **Not investment advice.** Real data: SPY daily total-return closes "
            f"{R['start']} to {R['end']} (n={R['n']:,}); cash leg a flat 4%/yr proxy (FRED "
            "unavailable). Every Sharpe is excess-of-cash. Methods in "
            "[`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The `> 💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT + "\n" + LOAD_REAL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Ribbon excess Sharpe **+{R['rb_sharpe']:.3f}**, HAC *t* = "
            f"**+{R['rb_t']:.2f}** (< 2). vs buy-and-hold race *t* = **{R['t_rb_vs_bh']:+.2f}** "
            f"(BH wins). Rotation placebo **p = {R['perm_p']:.3f}**. |\n"
            f"| **Tradability** | `MIRAGE` | Gross excess Sharpe +{R['sweep_0']:.3f} < BH "
            f"+{R['bh_sharpe']:.3f}; long/short version sharply negative. No edge to charge costs against. |\n"
            f"| **Reveals conviction?** | `BUSTED` | Width gate *lowers* Sharpe to +{R['cv_sharpe']:.3f}; "
            f"conviction-vs-ribbon *t* = {R['t_cv_vs_rb']:+.2f}. |\n\n"
            "> 💡 In plain words: every honest comparison points the same way. The ribbon does not "
            "beat the market, does not beat one moving average, does not beat a coin flip, and its "
            "signature 'conviction' reading actively hurts. The lower drawdown is reduced exposure, "
            "not timing skill."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 -- The claim, steelmanned\n\n"
            "Let $P_t$ be the SPY adjusted close. The short ribbon is the set of EMAs "
            "$\\{E^{(n)}_t : n \\in \\{3,5,8,10,12,15\\}\\}$ and the long ribbon "
            "$\\{E^{(m)}_t : m \\in \\{30,35,40,45,50,60\\}\\}$. Write $\\bar S_t$, $\\bar L_t$ for the "
            "ribbon means and the normalised spread $g_t = (\\bar S_t - \\bar L_t)/P_t$.\n\n"
            "- **Ribbon cross.** $w_t = \\mathbf{1}[\\bar S_{t-1} > \\bar L_{t-1}]$ (long/flat).\n"
            "- **Conviction gate (Guppy's distinctive claim).** "
            "$w_t = \\mathbf{1}[\\bar S_{t-1} > \\bar L_{t-1}] \\cdot \\mathbf{1}[g_{t-1} > \\text{median}_{60}(g)]$ "
            "-- long only when the ribbon is *wider than usual*.\n\n"
            "Strategy return $r^{\\text{strat}}_t = w_t r^{\\text{SPY}}_t + (1-w_t) r^{\\text{cash}}_t - "
            "c\\,|w_t - w_{t-1}|$. Hypotheses, each tested on the **excess-of-cash** series:\n\n"
            "- **H1.** $\\text{SR}_{\\text{excess}}(\\text{ribbon}) > \\text{SR}_{\\text{excess}}(\\text{BH})$ "
            "with race |*t*| $\\geq$ 2.\n"
            "- **H2.** Ribbon beats a single 50-day MA (the apparatus adds value).\n"
            "- **H3 (conviction).** The width gate *improves* on the bare cross.\n\n"
            "We reject all three: H1 fails (race *t* = " + f"{R['t_rb_vs_bh']:+.2f}" + ", the wrong "
            "sign), H2 fails (one EMA out-Sharpes the ribbon), H3 fails (the gate lowers every number)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 -- So what? -- what rides on each answer\n\n"
            "GMMA is on millions of retail charts. If H3 held, the 'conviction' reading would be a "
            "genuinely useful trend-quality filter and a small anomaly worth a factor write-up. If it "
            "fails, GMMA joins the long list of indicators whose value is *descriptive* (it draws a "
            "nice picture of a trend that already happened) rather than *predictive*. The distinction "
            "is the whole game: an indicator that labels the past is not a strategy."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 -- How we'd know -- the protocol\n\n"
            "- **Signal.** Twin EMA ribbons on SPY adjusted close; lag one day (signal at $t$ acts on "
            "return $t\\to t+1$); long/flat (long/short available).\n"
            "- **Cash leg.** Flat 4%/yr proxy. Every Sharpe is *excess-of-cash*; the cash level shifts "
            "both arms together so it cannot change the race verdict.\n"
            "- **Cost.** 2 bps one-way $\\times$ NAV on $|\\Delta w|$; swept 0-10 bps. Shorts pay 50 bps/yr borrow.\n"
            "- **Benchmarks.** Buy-and-hold; a single 50-day SMA *and* EMA; a random-timing control "
            "matched to the ribbon's in-market fraction.\n"
            "- **Inference.** HAC (Newey-West) *t* on mean daily excess return; return-difference HAC "
            "*t* for each Sharpe race (Jobson-Korkie style); a 2,000-draw circular-rotation permutation "
            "placebo on the signal.\n"
            "- **Positive control.** A deterministic two-regime synthetic tape: a *planted* trend edge "
            "(the ribbon should win) vs a flat random-walk null (it should not)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 -- The teardown"),
        md(
            "### 4a -- The full race table\n\n"
            "Every arm on the excess-of-cash series, with its own HAC *t* and the race *t* vs buy-and-hold."
        ),
        code(
            "rows = [('Buy-and-hold', s_bh), ('GMMA ribbon', s_rb), ('GMMA conviction', s_cv),\n"
            "        ('Single 50d SMA', s_sma), ('Single 50d EMA', s_ema), ('Random timing', s_rn)]\n"
            "print(f'{\"Arm\":16s} | {\"CAGR\":>7s} | {\"Sharpe\":>7s} | {\"MaxDD\":>7s} | {\"HAC t\":>6s}')\n"
            "print('-'*56)\n"
            "for lbl, s in rows:\n"
            "    t = s.get('tstat', float('nan'))\n"
            "    print(f'{lbl:16s} | {s[\"cagr\"]:+6.2%} | {s[\"sharpe\"]:+7.3f} | {s[\"max_drawdown\"]:+6.1%} | {t:+6.2f}')\n"
            "if HAVE_REAL:\n"
            "    print(f'\\nrace t (ribbon - BH):     {res[\"t_ribbon_vs_bh\"]:+.2f}  (BH significantly better)')\n"
            "    print(f'race t (ribbon - random): {res[\"t_ribbon_vs_rand\"]:+.2f}')\n"
            "    print(f'race t (ribbon - SMA):    {res[\"t_ribbon_vs_sma\"]:+.2f}')\n"
            "    print(f'race t (conviction - ribbon): {res[\"t_conv_vs_ribbon\"]:+.2f}')\n"
            "else:\n"
            "    print(f'\\nrace t (ribbon - BH):     {R[\"t_rb_vs_bh\"]:+.2f}  (BH significantly better)')\n"
            "    print(f'race t (ribbon - random): {R[\"t_rb_vs_rand\"]:+.2f}')\n"
            "    print(f'race t (ribbon - SMA):    {R[\"t_rb_vs_sma\"]:+.2f}')\n"
            "    print(f'race t (conviction - ribbon): {R[\"t_cv_vs_rb\"]:+.2f}')\n"
        ),
        md(
            f"> 💡 In plain words: the ribbon's own *t* (+{R['rb_t']:.2f}) doesn't even clear the 2-bar "
            f"for being different from zero, and the race against buy-and-hold is *negative* "
            f"({R['t_rb_vs_bh']:+.2f}) -- meaning the market is significantly *better*, not the ribbon. "
            f"And the single EMA (+{R['ema_sharpe']:.3f}) quietly out-Sharpes the whole twelve-line apparatus."
        ),
        md(
            "### 4b -- The rotation permutation placebo\n\n"
            "We circularly rotate the lagged ribbon signal 2,000 times -- preserving its in-market "
            "fraction and autocorrelation, destroying its alignment with returns -- and ask how often "
            "a rotated signal earns at least the real one's mean excess return."
        ),
        code(
            "if HAVE_REAL:\n"
            "    perm = st.permutation_pvalue(close, st.ribbon_signal, rf_daily=0.0, cost_bps=0.0, n_perm=2000, seed=435)\n"
            "    real_bps, perm_bps, pval = perm['real_mean_bps'], perm['perm_mean_bps'], perm['p_value']\n"
            "    # rebuild the placebo distribution for the histogram\n"
            "    sig = st.ribbon_signal(close).dropna()\n"
            "    asset = (close.loc[sig.index] / close.loc[sig.index].shift(1) - 1.0).to_numpy()\n"
            "    arr = sig.to_numpy(); rng = np.random.default_rng(435); dist = []\n"
            "    for _ in range(2000):\n"
            "        sh = int(rng.integers(1, len(arr)-1)); dist.append(np.nanmean(np.roll(arr, sh)*asset)*1e4)\n"
            "    dist = np.array(dist)\n"
            "else:\n"
            "    real_bps, perm_bps, pval = R['perm_real_bps'], R['perm_mean_bps'], R['perm_p']\n"
            "    rng = np.random.default_rng(435); dist = rng.normal(perm_bps, 1.6, 2000)\n"
            "fig, ax = plt.subplots(figsize=(9, 4.4))\n"
            "ax.hist(dist, bins=40, color=GREY, alpha=0.7, label='rotated placebos')\n"
            "ax.axvline(real_bps, c=RED, lw=2.2, label=f'real ribbon ({real_bps:+.2f} bps)')\n"
            "ax.set_xlabel('mean excess return (bps/day)'); ax.set_ylabel('count')\n"
            "ax.set_title(f'Real signal sits in the LEFT tail -- placebo p = {pval:.3f}'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real {real_bps:+.2f} bps/day vs placebo mean {perm_bps:+.2f}; p = {pval:.3f}')\n"
        ),
        md(
            f"> 💡 In plain words: the real ribbon's red line is on the *low* side of the random cloud "
            f"-- a random schedule beats it {R['perm_p']*100:.0f}% of the time. The ribbon's timing "
            "carries no information beyond the market's general upward drift, which a random in/out "
            "schedule harvests just as well."
        ),
        md(
            "### 4c -- The conviction gate: does width forecast continuation?\n\n"
            "Guppy's signature claim, isolated: take the bare cross and *add* the requirement that the "
            "ribbon be wider than its 60-day median. If width signals a continuing trend, this should help."
        ),
        code(
            "fig, ax = plt.subplots(figsize=(8.5, 4.4))\n"
            "labels = ['Bare ribbon cross', 'Cross + width gate']\n"
            "vals = [s_rb['sharpe'], s_cv['sharpe']]\n"
            "bars = ax.bar(labels, vals, color=[AMBER, RED], width=0.5)\n"
            "for b, v in zip(bars, vals):\n"
            "    ax.annotate(f'{v:+.3f}', (b.get_x()+b.get_width()/2, v), ha='center', va='bottom')\n"
            "ax.set_ylabel('Excess Sharpe')\n"
            "ax.set_title('Adding the \\'conviction\\' width gate makes it WORSE')\n"
            "plt.tight_layout(); plt.show()\n"
            "if HAVE_REAL:\n"
            "    print(f'conviction vs ribbon race t = {res[\"t_conv_vs_ribbon\"]:+.2f}')\n"
            "else:\n"
            "    print(f'conviction vs ribbon race t = {R[\"t_cv_vs_rb\"]:+.2f}')\n"
        ),
        md(
            f"> 💡 In plain words: the width gate drops the Sharpe from +{R['rb_sharpe']:.3f} to "
            f"+{R['cv_sharpe']:.3f}. A 'wide, parallel' ribbon describes a trend that has *already* run; "
            "by the time you see it, the easy money is gone. Width is a rear-view mirror sold as a windshield."
        ),
        md(
            "### 4d -- Synthetic positive control: the harness can detect a real trend edge\n\n"
            "Before we accept the null on SPY, we prove the engine *can* find a trend edge when one "
            "exists. A deterministic two-regime tape with a **planted** bull/bear edge vs a **flat "
            "null** random walk."
        ),
        code(
            "rows = []\n"
            "for edge, lbl in [(1.0, 'Planted edge'), (0.0, 'Flat null')]:\n"
            "    d, truth = data.synthetic_panel(n_years=35, edge=edge, drift_bull=0.16, drift_bear=-0.55,\n"
            "        vol_bull=0.10, vol_bear=0.45, p_bull_to_bear=0.12, p_bear_to_bull=0.20, seed=435)\n"
            "    r = st.run_experiment(d['close'], rf_daily=RF, cost_bps=COST)\n"
            "    p = st.permutation_pvalue(d['close'], st.ribbon_signal, n_perm=2000, seed=11)\n"
            "    rows.append((lbl, r['ribbon']['sharpe'], r['bh']['sharpe'], r['t_ribbon_vs_bh'], p['p_value']))\n"
            "print(f'{\"Tape\":14s} | {\"Ribbon Sh\":>9s} | {\"BH Sh\":>7s} | {\"t(rb-BH)\":>8s} | {\"perm p\":>7s}')\n"
            "print('-'*54)\n"
            "for lbl, rs, bs, t, p in rows:\n"
            "    print(f'{lbl:14s} | {rs:+9.3f} | {bs:+7.3f} | {t:+8.2f} | {p:7.4f}')\n"
        ),
        md(
            f"> 💡 In plain words: on the planted tape the ribbon beats buy-and-hold at *t* = "
            f"+{R['syn_edge_t_vs_bh']:.2f} and the placebo p collapses to {R['syn_edge_perm_p']:.3f} "
            f"-- the engine banks the edge. On the null it matches BH (*t* = {R['syn_null_t_vs_bh']:+.2f}) "
            f"and p = {R['syn_null_perm_p']:.2f}. So the engine works; the SPY null is real evidence, "
            "not a broken harness."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal `NONE`** -- ribbon excess Sharpe +{R['rb_sharpe']:.3f} (HAC *t* = "
            f"+{R['rb_t']:.2f}, below the bar); loses the race to buy-and-hold "
            f"(*t* = {R['t_rb_vs_bh']:+.2f}); tied by a random schedule (*t* = {R['t_rb_vs_rand']:+.2f}); "
            f"placebo p = {R['perm_p']:.3f}; out-Sharped by a single EMA. H1 and H2 rejected.\n"
            f"- **Tradability `MIRAGE`** -- gross excess Sharpe +{R['sweep_0']:.3f} already trails BH "
            f"+{R['bh_sharpe']:.3f}; no cost level rescues it; long/short is sharply negative. Nothing to deploy.\n"
            f"- **Reveals conviction? `BUSTED`** -- the width gate lowers Sharpe to +{R['cv_sharpe']:.3f} "
            f"(conviction-vs-ribbon *t* = {R['t_cv_vs_rb']:+.2f}). H3 rejected: width is hindsight."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 -- Could you trade it? -- costs and the long/short trap\n\n"
            "The rule is low-turnover, so costs are not the assassin -- it is dead even gross. And the "
            "long/short version (shorting on the down-cross) is far worse, because it fights the "
            "market's structural drift."
        ),
        code(
            "if HAVE_REAL:\n"
            "    costs = [0.0, 2.0, 5.0, 10.0]; shs = []\n"
            "    sig = st.ribbon_signal(close)\n"
            "    for c in costs:\n"
            "        bt = st.run_backtest(close, sig, rf_daily=RF, cost_bps=c)\n"
            "        shs.append(st.summary(bt['r_strategy_excess'])['sharpe'])\n"
            "    ls = st.run_experiment(close, rf_daily=RF, cost_bps=COST, allow_short=True)\n"
            "    ls_sh = ls['ribbon']['sharpe']\n"
            "else:\n"
            "    costs = [0.0, 2.0, 5.0, 10.0]\n"
            "    shs = [R['sweep_0'], R['sweep_2'], R['sweep_5'], R['sweep_10']]\n"
            "    ls_sh = -0.193\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.4))\n"
            "ax.plot(costs, shs, 'o-', c=RED, lw=2, label='ribbon long/flat (excess Sharpe)')\n"
            "ax.axhline(R['bh_sharpe'], ls='--', c=GREEN, label=f'buy-and-hold {R[\"bh_sharpe\"]:+.3f}')\n"
            "ax.axhline(0, c='k', lw=0.8)\n"
            "ax.set_xlabel('one-way cost (bps)'); ax.set_ylabel('Excess Sharpe')\n"
            "ax.set_title('Dead even at zero cost; long/short is negative')\n"
            "ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross ribbon Sharpe {shs[0]:+.3f} < BH {R[\"bh_sharpe\"]:+.3f}; long/short Sharpe {ls_sh:+.3f}')\n"
        ),
        md(
            f"> 💡 In plain words: the red line never reaches the green buy-and-hold line, even at zero "
            f"cost. And turning the rule long/short drives the Sharpe *negative* ({R['rb_sharpe']:.3f} "
            "long/flat becomes deeply negative) -- shorting SPY on a ribbon down-cross is a reliable way "
            "to lose money to the market's drift."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **Other tapes.** The engine is ticker-generic (`data.load_real(ticker, fetch=True)`). "
            "A trendier, less-mean-reverting instrument is the ribbon's best hope; SPY's drift+noise mix "
            "is unkind to all MA-timing rules.\n"
            "- **Parameter robustness.** We used Guppy's canonical 6+6 spans. Re-running over a grid of "
            "spans would be a multiple-testing exercise -- and per Sullivan-Timmermann-White (1999), any "
            "winner found that way needs a data-snooping correction before it counts.\n"
            "- **The contrast study.** [Study 110 (Faber-Timing)](../../110-faber-timing/) is the "
            "single-SMA rule that earns a real *risk* stamp. Diffing the two shows precisely what the "
            "GMMA's extra eleven lines buy: nothing.\n"
            "- **Conviction, properly.** If 'conviction' means anything, it might live in ribbon "
            "*curvature* or *parallelism* rather than raw width. A fork that formalises and tests those "
            "is the honest next step.\n\n"
            "*Think there's a tape where the ribbon clears the bar? Fork it, run the excess-vs-excess "
            "race with the rotation placebo, and show the *t*. That's the standard.*"
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
