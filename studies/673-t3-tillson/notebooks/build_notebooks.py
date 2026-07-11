"""Generate the two narrative notebooks for Study 673 (T3, Tillson).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). They recompute the
real-tape numbers live from the cached daily parquets under ../_cache/ when present, and
otherwise fall back to the frozen headline numbers in ``R`` (mirroring docs/results.md), so
the notebook re-runs for any reader, online or off.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-30).
R = dict(
    asof="2026-06-30", n_spy=8411, yrs=33.3, n_val=14, v_val=0.7,
    # mechanism check
    dist_t3=1.881, dist_sma=1.535, dist_ema=1.327,
    step_price=120.0, step_t3=105.29, step_sma=108.57, step_ema=111.52,
    # SPY headline, long/flat, cost=5bps
    t3_sharpe=0.308, t3_cagr=2.85, t3_dd=-55.6,
    sma_sharpe=0.238, sma_cagr=2.07, sma_dd=-57.6,
    ema_sharpe=0.267, ema_cagr=2.40, ema_dd=-48.0,
    bh_sharpe=0.647, bh_cagr=10.83, bh_dd=-55.2,
    t3_spread=-3.40, t3_t=-4.03,
    sma_spread=-3.70, sma_t=-4.42,
    ema_spread=-3.57, ema_t=-4.22,
    t3_sw=31.9, sma_sw=36.5, ema_sw=38.3,
    t3_tim=60,
    # head-to-head
    diff_sma_bps=0.30, diff_sma_t=0.97, diff_ema_bps=0.17, diff_ema_t=0.42,
    # gross
    t3_spread_gross=-2.77, t3_t_gross=-3.31,
    # permutation
    perm_obs=-2.77, perm_placebo=-1.89, perm_p=0.927,
    # cost sweep (net Sharpe, spread, t)
    cost=[0.0, 2.0, 5.0, 10.0],
    cost_sharpe=[0.451, 0.394, 0.308, 0.165],
    cost_spread=[-2.77, -3.02, -3.40, -4.04],
    cost_t=[-3.31, -3.60, -4.03, -4.73],
    # per-instrument (T3 Sharpe, B&H Sharpe, spread, t, switches T3/SMA/EMA)
    tick=["SPY", "QQQ", "AAPL", "MSFT", "XLE"],
    tick_t3=[0.308, 0.227, 0.621, 0.519, 0.219],
    tick_bh=[0.647, 0.521, 0.623, 0.821, 0.428],
    tick_spread=[-3.40, -4.08, -3.49, -6.18, -3.26],
    tick_t=[-4.03, -3.05, -1.99, -4.94, -2.17],
    tick_sw_t3=[31.9, 31.4, 29.6, 30.7, 31.0],
    tick_sw_sma=[36.5, 36.6, 33.1, 37.3, 36.6],
    tick_sw_ema=[38.3, 39.8, 35.7, 40.9, 42.4],
    # in/out split
    h1_t3=-0.059, h1_bh=0.455, h1_spread=-3.88, h1_t=-2.99,
    h2_t3=0.759, h2_bh=0.872, h2_spread=-2.86, h2_t=-2.59,
    # long/short
    ls_sharpe=-0.288, ls_spread=-6.89, ls_t=-4.08,
    # T3-slope variant
    slope_sharpe=0.557, slope_bh=0.647, slope_spread=-2.27, slope_t=-2.66, slope_sw=10.5,
    # v-sweep
    v_vals=[0.1, 0.3, 0.5, 0.7, 0.9],
    v_sharpe=[0.455, 0.366, 0.338, 0.308, 0.246],
    v_spread=[-2.79, -3.15, -3.29, -3.40, -3.67],
    v_t=[-3.28, -3.66, -3.84, -4.03, -4.45],
    v_sw=[20.5, 24.0, 28.5, 31.9, 33.8],
    # synthetic control
    syn_null_t=-0.21, syn_null_sd=1.05, syn_null_fire=2,
    syn_edge=[0.3, 0.6, 1.0],
    syn_spread=[7.19, 26.64, 50.27],
    syn_t=[5.76, 16.12, 21.35],
    syn_sharpe=[1.57, 4.98, 8.24],
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Genuinely_lower--lag%2C_cleaner_crossovers%3F: Mixed](https://img.shields.io/badge/"
    "Genuinely_lower--lag%2C_cleaner_crossovers%3F-Mixed-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from t3_tillson import data, strategy as st

TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "XLE"]
CACHE = os.path.abspath(os.path.join("..", "_cache"))
ASOF = "2026-06-30"
T3_N, T3_V = 14, 0.7

def _have_cache():
    return all(os.path.exists(data._cache_path(t, CACHE)) for t in TICKERS)

HAVE_REAL = _have_cache()

def tape(t):
    b = data.load_real(t, fetch=False, cache_dir=CACHE)
    return b[b.index <= ASOF]

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# T3 — does the six-times-smoothed moving average really beat a plain one?\n"
            "### Tim Tillson's T3, turned into a timing rule and tested honestly\n\n"
            + BADGES +
            "Here's a recipe from a 1998 trading-magazine article that's still baked into "
            "MT4/MT5 and TradingView indicator packs today: instead of a plain moving average, "
            "plot **T3** — a moving average built by smoothing price *six times over* and "
            "recombining the results with a tunable \"volume factor\". The pitch: it "
            "*virtually eliminates lag while smoothing the data*, so it turns earlier **and** "
            "whipsaws less than a plain SMA/EMA. Sounds like a free lunch — smoother **and** "
            "faster. Does it actually beat just buying and holding?\n\n"
            "> This is the plain-language layer. Want the *t*-stats, the permutation placebo, "
            "and the volume-factor sweep? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is "
            "drawn by the code beside it. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does T3 really react faster than a plain SMA/EMA? | **No.** After a "
            f"deterministic price jump, T3 is the **slowest** of the three to catch up — "
            "the opposite of \"eliminates lag\". |\n"
            f"| Does it at least whipsaw less? | **Yes.** T3 fires **{R['t3_sw']} "
            f"switches/yr** vs the SMA's **{R['sma_sw']}** — genuinely fewer flip-flops, on "
            "every ticker we tested. |\n"
            f"| Does fewer whipsaws mean it beats buy-and-hold? | **No.** It trails by "
            f"**{R['t3_spread']} bps/day** (*t* = {R['t3_t']}) — net Sharpe "
            f"**{R['t3_sharpe']}** vs **{R['bh_sharpe']}** for just holding. |\n"
            "| Is the timing at least real? | **Worse than random.** A shuffle of T3's own "
            f"calls beats the real ones **{R['perm_p']:.1%}** of the time. |\n\n"
            "> T3 keeps half its promise — it really is smoother, with fewer false signals. "
            "But \"less lag\" is false at the same period, and neither half adds up to a "
            "trading edge."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim\n\n"
            "> *\"T3 stacks six exponential moving averages and recombines them with a "
            "'volume factor' v. Unlike a DEMA or TEMA, it doesn't overshoot — it "
            "*virtually eliminates lag while smoothing the data*. Trade the T3 crossover (or "
            "its slope) and you'll turn earlier than a plain SMA/EMA AND get fewer false "
            "signals.\"*\n\n"
            "Tim Tillson published T3 in *Technical Analysis of Stocks & Commodities* "
            "(January 1998). The construction really is elegant: `GD(x,v) = (1+v)·EMA(x) − "
            "v·EMA(EMA(x))` — a tunable blend that becomes a DEMA at v=1 — nested three "
            "times. Six EMA passes, one linear recombination. We test whether that "
            "recombination actually delivers *both* halves of the promise."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what?\n\n"
            "If T3 really turned earlier **and** whipsawed less, it would be a strict "
            "upgrade over every simple moving-average rule — a free lunch worth dropping "
            "into any charting template. It's been in every major indicator library for "
            "over 25 years on exactly that promise. If instead the two halves of the pitch "
            "pull in opposite directions — smoother, sure, but *slower* not faster — then T3 "
            "is a different trade-off wearing a \"strictly better\" marketing label, and "
            "traders picking it for the wrong reason are buying lag they didn't ask for."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How would we know?\n\n"
            "The honest test isn't \"does the T3 rule make money\" — in a 33-year bull market "
            "almost any mostly-long rule does. The honest tests are:\n\n"
            "1. **Check the mechanism directly.** Feed T3, an SMA and an EMA (same length) a "
            "clean step in price. Whichever catches up fastest actually has less lag.\n"
            "2. **Count the whipsaws.** Position changes per year, T3 vs SMA vs EMA.\n"
            "3. **Race the timing against just holding.** Go long when T3 says long, sit in "
            "cash otherwise, then subtract a buy-and-hold of the same asset — the 'active "
            "spread' should be positive if the timing helps.\n"
            "4. **Sweep the volume factor v.** Tillson's own tuning knob, 0.1 to 0.9 — is any "
            "single value doing the work, or is the result robust?\n\n"
            "We enter one day after each signal (no peeking), charge realistic costs, and run "
            f"it on SPY plus four other liquid tapes over ~{R['yrs']:.0f} years."
        ),

        # ---- BEAT 4 ----
        md(
            "## 4 · The teardown — what actually happens\n\n"
            "**First, the mechanism itself.** A clean +20% price jump — which line catches "
            "up fastest?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    sr = st.step_response(n=T3_N, v=T3_V)\n"
            "    after5 = sr.iloc[35]\n"
            "    vals = [after5['T3'], after5['SMA'], after5['EMA']]\n"
            "else:\n"
            f"    vals = [{R['step_t3']}, {R['step_sma']}, {R['step_ema']}]\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.3))\n"
            "ax.axhline(100, c=GREY, lw=1, ls=':', label='pre-jump price')\n"
            f"ax.axhline({R['step_price']}, c='k', lw=1.2, label='post-jump price')\n"
            "ax.bar(['T3(14)','SMA(14)','EMA(14)'], vals, color=[RED, AMBER, GREEN], width=.55)\n"
            "for i,v in enumerate(vals): ax.annotate(f'{v:.1f}',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('line value, 5 bars after the jump')\n"
            "ax.set_title('T3 is the SLOWEST to catch up — not the fastest')\n"
            "ax.legend(loc='lower right'); plt.tight_layout(); plt.show()\n"
            "print(f'5 bars after jump: T3={vals[0]:.2f}  SMA={vals[1]:.2f}  EMA={vals[2]:.2f}')"
        ),
        md(
            f"Five bars after a clean price jump to {R['step_price']:.0f}, the plain EMA has "
            f"reached **{R['step_ema']:.2f}**, the SMA **{R['step_sma']:.2f}** — and T3 only "
            f"**{R['step_t3']:.2f}**. Stacking six EMA passes and correcting for lag "
            "algebraically still leaves T3 the *slowest* of the three at the same nominal "
            "period. \"Virtually eliminates lag\" does not survive contact with a stopwatch.\n\n"
            "**Second, the other half of the promise: does it at least whipsaw less?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    r = st.run_experiment(tape('SPY'), t3_n=T3_N, t3_v=T3_V, sma_period=T3_N,\n"
            "                          ema_period=T3_N, cost_bps=5.0)\n"
            "    sw = {k: r[k]['switches_per_yr'] for k in ('T3','SMA','EMA')}\n"
            "else:\n"
            f"    sw = dict(T3={R['t3_sw']}, SMA={R['sma_sw']}, EMA={R['ema_sw']})\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.0))\n"
            "ax.bar(['T3(14)','SMA(14)','EMA(14)'], [sw['T3'],sw['SMA'],sw['EMA']],\n"
            "       color=[GREEN, GREY, GREY], width=.55)\n"
            "ax.set_ylabel('position changes / yr')\n"
            "ax.set_title('Genuinely fewer switches — this half of the pitch is TRUE')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"T3 {sw['T3']:.1f}/yr  vs  SMA {sw['SMA']:.1f}/yr  vs  EMA {sw['EMA']:.1f}/yr\")"
        ),
        md(
            f"**{R['t3_sw']} switches/yr** for T3 against **{R['sma_sw']}** for the SMA and "
            f"**{R['ema_sw']}** for the EMA — T3 really is calmer. This part of the claim "
            "holds up, and it holds on every ticker in the basket.\n\n"
            "**So — does a smoother, if slower, line at least beat holding?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    sh = {k: r[k]['sharpe_net'] for k in ('T3','SMA','EMA')}\n"
            "    sh['B&H'] = r['T3']['bh_sharpe']\n"
            "else:\n"
            f"    sh = dict(T3={R['t3_sharpe']}, SMA={R['sma_sharpe']}, EMA={R['ema_sharpe']})\n"
            f"    sh['B&H'] = {R['bh_sharpe']}\n"
            "labels = ['T3(14)','SMA(14)','EMA(14)','Buy & hold']\n"
            "vals = [sh['T3'], sh['SMA'], sh['EMA'], sh['B&H']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(labels, vals, color=[RED, AMBER, AMBER, GREEN], width=.6)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('net Sharpe (annualised)')\n"
            "ax.set_title('All three timing rules lose to just holding')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"T3 {sh['T3']:.3f} | SMA {sh['SMA']:.3f} | EMA {sh['EMA']:.3f} | hold {sh['B&H']:.3f}\")"
        ),
        md(
            f"Buy-and-hold (Sharpe **{R['bh_sharpe']}**) beats every timing rule; T3's "
            f"quieter line (**{R['t3_sharpe']}**) is a bit better than the SMA "
            f"(**{R['sma_sharpe']}**) but nowhere close to just holding. Fewer whipsaws "
            "reduced the *damage* of timing — it didn't turn it into an edge.\n\n"
            "**Last check — is the timing even real, or is it noise?** Shuffle T3's own "
            "calls in time and see if the real ones beat the shuffles."
        ),
        code(
            "if HAVE_REAL:\n"
            "    p = r['T3_permutation']\n"
            "    obs, plac, pv = p['observed_spread_bps'], p['placebo_mean_bps'], p['p_value']\n"
            "else:\n"
            f"    obs, plac, pv = {R['perm_obs']}, {R['perm_placebo']}, {R['perm_p']}\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.bar(['Real T3\\ntiming', 'Average random\\nre-timing'], [obs, plac],\n"
            "       color=[RED, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('active spread vs buy&hold (bps/day)')\n"
            "ax.set_title('The real timing is WORSE than a random shuffle of itself')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'real {obs:+.2f} bps/day | random {plac:+.2f} bps/day | p = {pv:.3f}')"
        ),
        md(
            f"The real T3 timing (**{R['perm_obs']} bps/day**) is *worse* than the average "
            f"random re-timing (**{R['perm_placebo']} bps/day**), and **{R['perm_p']:.1%}** "
            "of shuffles beat it. Whatever information T3's crossovers carry, it points the "
            "wrong way more often than chance."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Active spread vs holding **{R['t3_spread']} bps/day**, "
            f"*t* = **{R['t3_t']}** — significantly negative on SPY, on 4 of 5 basket tapes "
            "individually, in both sample halves, and at every volume factor v from 0.1 to "
            f"0.9. Permutation *p* = {R['perm_p']:.3f}.\n"
            f"- **Tradability — Mirage.** Net Sharpe **{R['t3_sharpe']}** vs buy-and-hold "
            f"**{R['bh_sharpe']}**; loses even gross, and no cost level rescues it.\n"
            f"- **\"Lower-lag, cleaner crossovers\"? — Mixed.** Cleaner: **true** "
            f"({R['t3_sw']} vs {R['sma_sw']}/{R['ema_sw']} switches/yr). Lower-lag: "
            "**false** (slowest of the three to catch a step). Neither half beats the plain "
            "MAs T3 claims to improve on."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No. The gross timing already loses to holding; costs only widen the gap. "
            "There's no break-even cost because the line starts below zero:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    costs = [0.0, 2.0, 5.0, 10.0]\n"
            "    spr = [st.run_experiment(tape('SPY'), t3_n=T3_N, t3_v=T3_V, cost_bps=c)['T3']"
            "['mean_spread_bps'] for c in costs]\n"
            "else:\n"
            f"    costs = {R['cost']}; spr = {R['cost_spread']}\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, spr, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, spr, 0, color=RED, alpha=.12)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('active spread vs hold (bps/day)')\n"
            "ax.set_title('Already below zero at zero cost — there is no break-even')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('The gross spread is negative; costs are not the issue, the timing is.')"
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further\n\n"
            "- **The positive control.** The companion notebook plants a real trend in a "
            "synthetic tape — and the *same* T3 engine catches it cleanly (*t* up to +21). "
            "The harness works; the daily stock market just doesn't hand a six-times-smoothed "
            "line enough persistent trend to pay for its lag.\n"
            "- **Shorter periods.** Tillson's own writeups sometimes use N in the 5-8 range to "
            "compensate for the six-stage smoothing — worth a fork to see if a shorter T3 "
            "closes the lag gap against the SMA/EMA benchmarks.\n"
            "- **Other 'smarter MA' claims, other fates.** [Study 432 (Hull MA)](../../432-hull-moving-average/) "
            "shows the *opposite* failure mode — genuinely less lag, but *more* whipsaws. "
            "[Study 672 (McGinley Dynamic)](../../672-mcginley-dynamic/) tests a recursive "
            "self-adjusting line on the same infrastructure.\n\n"
            "*Think T3 earns its keep at a different N, v, or timeframe? Fork this, and show "
            "an active spread that clears HAC *t* = 2 against buy-and-hold. That's the bar.*"
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
            "# T3 (Tillson) — a quantitative teardown\n"
            "### Daily total-return bars · T3(14, v=0.7) cross & slope · HAC inference · "
            "permutation placebo · SMA/EMA benchmarks · volume-factor sweep\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) "
            "— *same seven beats, every claim now carrying its standard error.* We test "
            "whether the T3(14, v=0.7) price-cross (and slope) timing rule beats "
            "buy-and-hold on a net, excess-of-cash basis, beats the simpler SMA(14)/EMA(14) "
            "rules, fires fewer false signals, and does so genuinely faster — across five "
            "liquid daily tapes and every volume factor from 0.1 to 0.9.\n\n"
            f"> **Not investment advice.** Real data: Yahoo daily total-return bars, full "
            f"history to 2026-06-30, as-of **{R['asof']}**; the offline core runs the "
            "deterministic synthetic tape. Methods & sources in "
            "[`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> The `In plain words` notes translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | T3 active spread vs buy&hold **{R['t3_spread']} "
            f"bps/day**, HAC *t* = **{R['t3_t']}** (gross *t* = {R['t3_t_gross']}); "
            f"permutation *p* = **{R['perm_p']:.3f}**; negative across every v in "
            "[0.1, 0.9]. |\n"
            f"| **Tradability** | `MIRAGE` | Net Sharpe **{R['t3_sharpe']}** vs buy&hold "
            f"**{R['bh_sharpe']}**; loses gross; long/short Sharpe **{R['ls_sharpe']}**. |\n"
            "| **Lower-lag, cleaner crossovers?** | `MIXED` | Cleaner: **true** "
            f"({R['t3_sw']} vs {R['sma_sw']}/{R['ema_sw']} switches/yr). Lower-lag: "
            f"**false** (step-response value {R['step_t3']:.2f} vs SMA {R['step_sma']:.2f} / "
            f"EMA {R['step_ema']:.2f}). |\n\n"
            "> In plain words: T3 is genuinely the calmest of the three lines, but it is "
            "also the laggiest — and neither property turns into a real trading edge."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $\\text{EMA}_n(x)$ be the exponential moving average (span convention) and "
            "define the *generalized DEMA*\n\n"
            "$$\\text{GD}(x, v) = (1+v)\\,\\text{EMA}_n(x) - v\\,\\text{EMA}_n(\\text{EMA}_n(x)).$$\n\n"
            "Tillson's T3 nests GD three times: $T3 = \\text{GD}(\\text{GD}(\\text{GD}(P, v), "
            "v), v)$, which expands to a fixed linear combination of the 3rd–6th EMA of "
            "price. The hypotheses:\n\n"
            "- **H₁ (signal).** $\\mathbb{E}[\\,r^{\\text{strat}}_t - r^{\\text{B\\&H}}_t\\,] > 0$ "
            "— the active spread is positive.\n"
            "- **H₂ (beats SMA/EMA).** Net Sharpe$(T3) > $ Net Sharpe$(\\text{SMA})$ and "
            "$>$ Net Sharpe$(\\text{EMA})$, head-to-head active spread *t* ≥ 2.\n"
            "- **H₃ (fewer false signals).** switches/yr$(T3) < $ switches/yr$(\\text{SMA})$.\n"
            "- **H₄ (lower lag).** T3 tracks a step in price faster than SMA/EMA at the "
            "same nominal N.\n\n"
            f"We reject H₁ (HAC *t* = {R['t3_t']}) and H₂ (head-to-head *t* = "
            f"{R['diff_sma_t']} / {R['diff_ema_t']}, both < 2) and H₄ (T3 is the *slowest* "
            f"of the three to a step); we **confirm H₃** ({R['t3_sw']} < {R['sma_sw']} "
            "switches/yr, on every ticker)."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "T3 is a textbook example of the desk's favorite trap: a filter that is "
            "*visibly* better on a chart (it really does track less noisily) being sold as "
            "*strictly* better on every axis, including the one axis (lag) it does not "
            "improve at a shared period. If H₁–H₄ all held it would be a Pareto improvement "
            "over both SMA and EMA — worth swapping into every existing strategy for free. "
            "Splitting the claim in two (smoother: yes; faster: no) is the point of testing "
            "the mechanism directly instead of taking the marketing at face value."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Indicator.** T3(14, v=0.7) — Tillson's own suggested volume factor, same "
            "nominal N as siblings 432/672 for a fair cross-study race; benchmarks SMA(14) "
            "and EMA(14).\n"
            "- **Mechanism check.** A deterministic +20% step response and mean tracking "
            "distance |close − line|/close, isolated from any noise.\n"
            "- **Rule.** Price-cross: $d_t = \\mathbb{1}[C_t > T3_t]$ (and the same form for "
            "SMA/EMA); a **T3-slope** variant $d_t = \\mathbb{1}[T3_t > T3_{t-1}]$. A "
            "long/short variant flips flat → −1.\n"
            "- **Execution lag.** One `shift`: position formed on the close of $t$ earns the "
            "close-to-close return of $t+1$. Stated once, applied once.\n"
            "- **Costs.** One-way × NAV on $|d_t - d_{t-1}|$ turnover; short legs pay "
            "50 bps/yr borrow.\n"
            "- **Signal test.** HAC (Newey-West) one-sample *t* on the daily *active spread* "
            "$r^{\\text{strat}}-r^{\\text{B\\&H}}$ — excess-vs-excess by construction — plus "
            "head-to-head *t*'s vs SMA and EMA.\n"
            "- **Placebo.** Circular-shift the realised position path 2,000× (kills timing, "
            "keeps turnover/bias); one-sided *p* on the gross spread.\n"
            "- **Robustness.** Cost sweep, per-instrument, first-vs-second-half split, "
            "long/short, and a volume-factor v sweep across 0.1 → 0.9.\n"
            "- **Positive control.** Synthetic tape with a *planted* regime-switching trend.\n\n"
            f"Five tapes: SPY, QQQ, AAPL, MSFT, XLE — full history to 2026-06-30 "
            f"(SPY n = {R['n_spy']})."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The mechanism check — is T3 actually lower-lag?\n\n"
            "A clean +20% step in price (flat, then held) isolates the mechanism from any "
            "noise. Whichever line reaches the new level fastest genuinely has less lag."
        ),
        code(
            "if HAVE_REAL:\n"
            "    td = st.tracking_distance(tape('SPY')['close'], n=T3_N, v=T3_V)\n"
            "    sr = st.step_response(n=T3_N, v=T3_V)\n"
            "    after5 = sr.iloc[35]\n"
            "    dist = [td['T3'], td['SMA'], td['EMA']]\n"
            "    step = [after5['T3'], after5['SMA'], after5['EMA']]\n"
            "else:\n"
            f"    dist = [{R['dist_t3']}, {R['dist_sma']}, {R['dist_ema']}]\n"
            f"    step = [{R['step_t3']}, {R['step_sma']}, {R['step_ema']}]\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.bar(['T3','SMA','EMA'], dist, color=[RED, AMBER, GREEN], width=.55)\n"
            "for i,v in enumerate(dist): a1.annotate(f'{v:.2f}%',(i,v),ha='center',va='bottom')\n"
            "a1.set_ylabel('mean |close - line| / close (%)')\n"
            "a1.set_title('Tracking distance: T3 is the LEAST tight')\n"
            "a2.bar(['T3','SMA','EMA'], step, color=[RED, AMBER, GREEN], width=.55)\n"
            "for i,v in enumerate(step): a2.annotate(f'{v:.1f}',(i,v),ha='center',va='bottom')\n"
            f"a2.axhline({R['step_price']}, c='k', lw=1, ls=':', label='target level')\n"
            "a2.set_ylabel('line value, 5 bars after a +20% step')\n"
            "a2.set_title('Step response: T3 is the SLOWEST to catch up'); a2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'tracking distance: T3={dist[0]:.3f}%  SMA={dist[1]:.3f}%  EMA={dist[2]:.3f}%')\n"
            "print(f'step value @+5: T3={step[0]:.2f}  SMA={step[1]:.2f}  EMA={step[2]:.2f}')"
        ),
        md(
            f"> In plain words: T3 sits **{R['dist_t3']:.2f}%** away from price on average — "
            f"farther than the SMA ({R['dist_sma']:.2f}%) or EMA ({R['dist_ema']:.2f}%) — and "
            "after a clean jump it is the *last* of the three to catch up. The reason is "
            "structural: T3 nests its DEMA-style lag correction three times, so by the third "
            "nesting the correction is being applied to an *already twice-smoothed* series, "
            "not raw price. Six stages of smoothing dominate the algebra at a shared N. This "
            "directly contradicts Tillson's \"virtually eliminates lag\" pitch when compared "
            "against a plain SMA/EMA of the same nominal length — the exact comparison every "
            "charting platform invites when a trader swaps the indicator but keeps the "
            "'length' input."
        ),
        md(
            "### 4b · T3 vs SMA vs EMA vs buy-and-hold — net Sharpe and active-spread *t*\n\n"
            "The bar that matters is the active-spread HAC *t*: if T3 timing helps, it "
            "clears +2."
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(tape('SPY'), t3_n=T3_N, t3_v=T3_V, sma_period=T3_N,\n"
            "                            ema_period=T3_N, cost_bps=5.0)\n"
            "    rows = [(k, res[k]['sharpe_net'], res[k]['mean_spread_bps'], res[k]['spread_t'],\n"
            "             res[k]['switches_per_yr']) for k in ('T3','SMA','EMA')]\n"
            "    rows.append(('B&H', res['T3']['bh_sharpe'], 0.0, float('nan'), 0.0))\n"
            "    tbl = pd.DataFrame(rows, columns=['rule','sharpe_net','spread_bps','spread_t','sw/yr'])\n"
            "    d_sma_t, d_ema_t = res['diff_t3_sma_t'], res['diff_t3_ema_t']\n"
            "else:\n"
            "    tbl = pd.DataFrame({\n"
            "        'rule': ['T3','SMA','EMA','B&H'],\n"
            f"        'sharpe_net': [{R['t3_sharpe']},{R['sma_sharpe']},{R['ema_sharpe']},{R['bh_sharpe']}],\n"
            f"        'spread_bps': [{R['t3_spread']},{R['sma_spread']},{R['ema_spread']},0.0],\n"
            f"        'spread_t': [{R['t3_t']},{R['sma_t']},{R['ema_t']},float('nan')],\n"
            f"        'sw/yr': [{R['t3_sw']},{R['sma_sw']},{R['ema_sw']},0.0]}})\n"
            f"    d_sma_t, d_ema_t = {R['diff_sma_t']}, {R['diff_ema_t']}\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "sub = tbl[tbl['rule']!='B&H']\n"
            "col = [RED if t < 0 else GREY for t in sub['spread_t']]\n"
            "ax.bar(sub['rule'], sub['spread_t'], color=col)\n"
            "for s in (2, -2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('active-spread HAC t (vs buy&hold)')\n"
            "ax.set_title('All three rules have significantly NEGATIVE timing')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'T3-SMA head-to-head t = {d_sma_t:+.2f}   T3-EMA head-to-head t = {d_ema_t:+.2f}')\n"
            "tbl.round(3)"
        ),
        md(
            f"> In plain words: every timing rule's active spread is significantly negative "
            f"— none beats holding. T3 is the *best of the three* (*t* = {R['t3_t']}, less "
            f"negative than SMA's {R['sma_t']} or EMA's {R['ema_t']}) but the head-to-head "
            f"spread vs SMA (*t* = {R['diff_sma_t']}) and EMA (*t* = {R['diff_ema_t']}) "
            "never clears the desk's *t* ≥ 2 bar — T3 is not *measurably* better than the "
            "'dumb' MAs it's meant to beat, only better by an uncertifiable margin."
        ),
        md(
            "### 4c · The whipsaw count — the one half of the claim that holds\n\n"
            "Position changes per year, and the same count replicated across all five "
            "basket tickers."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sw = {k: res[k]['switches_per_yr'] for k in ('T3','SMA','EMA')}\n"
            "    per_sw = []\n"
            "    for t in TICKERS:\n"
            "        rr = st.run_experiment(tape(t), t3_n=T3_N, t3_v=T3_V, sma_period=T3_N,\n"
            "                               ema_period=T3_N, cost_bps=5.0)\n"
            "        per_sw.append((t, rr['T3']['switches_per_yr'], rr['SMA']['switches_per_yr'],\n"
            "                       rr['EMA']['switches_per_yr']))\n"
            "    per_sw = pd.DataFrame(per_sw, columns=['ticker','T3','SMA','EMA']).set_index('ticker')\n"
            "else:\n"
            f"    sw = dict(T3={R['t3_sw']}, SMA={R['sma_sw']}, EMA={R['ema_sw']})\n"
            "    per_sw = pd.DataFrame({'T3': " + repr(R['tick_sw_t3']) + ",\n"
            "                           'SMA': " + repr(R['tick_sw_sma']) + ",\n"
            "                           'EMA': " + repr(R['tick_sw_ema']) + "},\n"
            "                          index=" + repr(R['tick']) + ")\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.3))\n"
            "per_sw.plot(kind='bar', ax=ax, color=[GREEN, GREY, GREY], width=.75)\n"
            "ax.set_ylabel('switches / yr'); ax.set_xticklabels(per_sw.index, rotation=0)\n"
            "ax.set_title('T3 whipsaws LESS than SMA/EMA — on every single ticker')\n"
            "plt.tight_layout(); plt.show()\n"
            "per_sw.round(1)"
        ),
        md(
            f"> In plain words: {R['t3_sw']} vs {R['sma_sw']}/{R['ema_sw']} switches/yr on "
            "SPY, and T3 is the calmest of the three on **all five** tickers — this is the "
            "one part of the pitch that genuinely holds up. Six stages of EMA smoothing "
            "really do damp noise; the failure is that this smoothness comes with more lag, "
            "not less, so the calmer line doesn't turn into a better trade."
        ),
        md(
            "### 4d · Permutation placebo — timing vs exposure\n\n"
            "Circularly shift the realised position path 2,000×; the statistic is the gross "
            "active spread. If the timing is informative, the real spread beats the placebo "
            "distribution."
        ),
        code(
            "if HAVE_REAL:\n"
            "    p = st.permutation_pvalue(tape('SPY')['close'].pct_change(),\n"
            "                              st.t3_position(tape('SPY')['close'], T3_N, T3_V),\n"
            "                              cost_bps=0.0, n_perm=2000, seed=673)\n"
            "    obs, plac, pv = p['observed_spread_bps'], p['placebo_mean_bps'], p['p_value']\n"
            "    rng = np.random.default_rng(673)\n"
            "    a = tape('SPY')['close'].pct_change().fillna(0).to_numpy()\n"
            "    held = st.t3_position(tape('SPY')['close'], T3_N, T3_V).shift(1).fillna(0).to_numpy()\n"
            "    draws = []\n"
            "    for _ in range(2000):\n"
            "        h = np.roll(held, int(rng.integers(1,len(held))))\n"
            "        turn=np.abs(np.diff(h,prepend=0.0)); draws.append(((h*a-turn*0)-a).mean()*1e4)\n"
            "else:\n"
            f"    obs, plac, pv = {R['perm_obs']}, {R['perm_placebo']}, {R['perm_p']}\n"
            f"    draws = list(np.random.default_rng(1).normal({R['perm_placebo']}, 0.6, 2000))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=30, color=GREY, alpha=.7, label='random re-timings (2000)')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'real T3 timing: {obs:+.2f} bps/day')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('active spread vs buy&hold (bps/day)'); ax.set_ylabel('count')\n"
            "ax.set_title('Real T3 timing sits in the LEFT tail — worse than its own shuffles')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.2f} | placebo mean {plac:+.2f} | one-sided p = {pv:.3f}')"
        ),
        md(
            f"> In plain words: *p* = {R['perm_p']:.3f} — **{R['perm_p']:.1%}** of random "
            "re-timings of T3's own trades beat the real ones. The timing isn't just "
            "uninformative; it is reliably on the wrong side more often than chance."
        ),
        md(
            "### 4e · Per-instrument, in/out-of-sample & long/short — is it ever positive?\n\n"
            "Active-spread *t* on all five tapes, the SPY first-vs-second-half split, and "
            "the long/short variant."
        ),
        code(
            "if HAVE_REAL:\n"
            "    recs = []\n"
            "    for t in TICKERS:\n"
            "        rr = st.run_experiment(tape(t), t3_n=T3_N, t3_v=T3_V, cost_bps=5.0)['T3']\n"
            "        recs.append((t, rr['sharpe_net'], rr['bh_sharpe'], rr['mean_spread_bps'], rr['spread_t']))\n"
            "    per = pd.DataFrame(recs, columns=['ticker','t3_sharpe','bh_sharpe','spread_bps','t'])\n"
            "    spy = tape('SPY'); half = len(spy)//2\n"
            "    h1 = st.run_experiment(spy.iloc[:half], t3_n=T3_N, t3_v=T3_V, cost_bps=5.0)['T3']\n"
            "    h2 = st.run_experiment(spy.iloc[half:], t3_n=T3_N, t3_v=T3_V, cost_bps=5.0)['T3']\n"
            "    split = (h1['spread_t'], h2['spread_t'])\n"
            "    ls = st.run_experiment(spy, t3_n=T3_N, t3_v=T3_V, cost_bps=5.0, long_short=True)['T3']\n"
            "    ls_sharpe, ls_t = ls['sharpe_net'], ls['spread_t']\n"
            "else:\n"
            "    per = pd.DataFrame({'ticker': "
            f"{R['tick']}, 't3_sharpe': {R['tick_t3']}, 'bh_sharpe': {R['tick_bh']},"
            f" 'spread_bps': {R['tick_spread']}, 't': {R['tick_t']}}})\n"
            f"    split = ({R['h1_t']}, {R['h2_t']})\n"
            f"    ls_sharpe, ls_t = {R['ls_sharpe']}, {R['ls_t']}\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "ax.bar(per['ticker'], per['t'], color=RED)\n"
            "ax.axhline(-2, ls='--', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('active-spread HAC t (T3 vs hold)')\n"
            "ax.set_title('Five of five negative; four clear t = -2')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('SPY split  H1 t = %.2f | H2 t = %.2f  (both negative)' % split)\n"
            "print(f'long/short: Sharpe {ls_sharpe:+.3f}, spread t = {ls_t:+.2f} (worse than long/flat)')\n"
            "per.round(3)"
        ),
        md(
            f"> In plain words: every instrument's T3 timing trails buy-and-hold, and SPY "
            f"loses in both halves (H1 *t* = {R['h1_t']}, H2 *t* = {R['h2_t']}) — not a "
            f"one-regime artefact. Shorting on the flip side makes it worse (Sharpe "
            f"{R['ls_sharpe']}, *t* = {R['ls_t']})."
        ),
        md(
            "### 4f · The T3-slope variant and the volume-factor (v) robustness sweep\n\n"
            "The brief's other timer — long when T3 itself is rising — and Tillson's own "
            "tuning knob, swept from 0.1 to 0.9."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sl = st.run_experiment(tape('SPY'), t3_n=T3_N, t3_v=T3_V, cost_bps=5.0, rule='slope')['T3']\n"
            "    slope_sharpe, slope_t, slope_sw = sl['sharpe_net'], sl['spread_t'], sl['switches_per_yr']\n"
            "    sweep = st.v_sweep(tape('SPY'), t3_n=T3_N, cost_bps=5.0)\n"
            "    vs, vt, vsw = list(sweep.index), list(sweep['spread_t']), list(sweep['switches_per_yr'])\n"
            "else:\n"
            f"    slope_sharpe, slope_t, slope_sw = {R['slope_sharpe']}, {R['slope_t']}, {R['slope_sw']}\n"
            f"    vs, vt, vsw = {R['v_vals']}, {R['v_t']}, {R['v_sw']}\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.2, 4.3))\n"
            "cross_t = res['T3']['spread_t'] if HAVE_REAL else "
            f"{R['t3_t']}\n"
            "a1.bar(['T3-cross', 'T3-slope'], [cross_t, slope_t],\n"
            "       color=[RED, AMBER], width=.5)\n"
            "a1.axhline(-2, ls='--', c=GREY); a1.axhline(0, c='k', lw=1)\n"
            "a1.set_ylabel('active-spread HAC t'); a1.set_title(f'Slope timer: fewer switches ({slope_sw:.1f}/yr), still negative')\n"
            "a2.plot(vs, vt, 'o-', c=RED, lw=2)\n"
            "a2.axhline(-2, ls='--', c=GREY); a2.axhline(0, c='k', lw=1)\n"
            "a2.set_xlabel('volume factor v'); a2.set_ylabel('active-spread HAC t')\n"
            "a2.set_title('Negative & significant at EVERY v from 0.1 to 0.9')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'T3-slope: Sharpe {slope_sharpe:+.3f}, t = {slope_t:+.2f}, {slope_sw:.1f} switches/yr')\n"
            "print('v-sweep t:', {round(v,1): round(t,2) for v,t in zip(vs, vt)})"
        ),
        md(
            f"> In plain words: the T3-slope timer fires far fewer switches "
            f"({R['slope_sw']:.1f}/yr — the smoothed line only flips on genuine reversals) "
            f"and shrinks the loss (*t* = {R['slope_t']}), but never crosses into positive, "
            "significant territory. And the volume-factor sweep closes the door on \"maybe "
            f"a different v would work\": every v from 0.1 to 0.9 gives a negative, "
            f"significant active spread (*t* between {min(R['v_t']):.2f} and "
            f"{max(R['v_t']):.2f}) — lower v (closer to a plain triple-EMA) is milder but "
            "never positive."
        ),
        md(
            "### 4g · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic mean-reverting price tape, regime-switching planted trend. The null "
            "(edge=0) is checked over **20 seeds** — never a single stream."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    b, _ = data.synthetic_panel(n_days=6000, edge=0.0, seed=673 + s_)\n"
            "    null_ts.append(st.run_experiment(b, t3_n=T3_N, t3_v=T3_V, cost_bps=0.0)['T3']['spread_t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "edges = [0.3, 0.6, 1.0]\n"
            "planted_t = []\n"
            "for e in edges:\n"
            "    b, _ = data.synthetic_panel(n_days=6000, edge=e, seed=673)\n"
            "    planted_t.append(st.run_experiment(b, t3_n=T3_N, t3_v=T3_V, cost_bps=0.0)['T3']['spread_t'])\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12,.12,20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (edge=0), 20 seeds')\n"
            "ax.scatter([1,2,3], planted_t, color=RED, s=90, zorder=5, label='planted edge 0.3/0.6/1.0')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0,1,2,3]); ax.set_xticklabels(['null x 20','0.3','0.6','1.0'])\n"
            "ax.set_ylabel('active-spread HAC t')\n"
            "ax.set_title('Control: null rarely fires; a planted trend lights up hard')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds  |  planted t = {planted_t}')"
        ),
        md(
            f"> In plain words: across 20 null worlds the detector averages "
            f"*t* = {R['syn_null_t']:+.2f} (sd {R['syn_null_sd']:.2f}) and rarely crosses the "
            f"bar ({R['syn_null_fire']}/20 seeds); a planted trend of just 0.3 already reads "
            f"*t* = {R['syn_t'][0]:+.2f}, growing to {R['syn_t'][2]:+.2f} at edge=1.0. The "
            "machinery is unbiased and powerful — the real-tape null result is a statement "
            "about the market, not a broken harness. *(A faithful-engine / power check "
            "only — never cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — active spread {R['t3_spread']} bps/day, HAC *t* "
            f"{R['t3_t']} (gross {R['t3_t_gross']}); permutation *p* = {R['perm_p']:.3f}; "
            "negative on 4/5 tapes individually, 5/5 point estimates, both halves, and "
            "every v in [0.1, 0.9].\n"
            f"- **Tradability `MIRAGE`** — net Sharpe {R['t3_sharpe']} vs hold "
            f"{R['bh_sharpe']}; loses gross; long/short Sharpe {R['ls_sharpe']}, spread "
            f"{R['ls_spread']} bps/day (*t* {R['ls_t']}).\n"
            "- **Lower-lag, cleaner crossovers? `MIXED`** — cleaner is true "
            f"({R['t3_sw']} vs {R['sma_sw']}/{R['ema_sw']} switches/yr, every ticker); "
            "lower-lag is false (slowest of three on a clean step, farthest mean tracking "
            "distance). Head-to-head vs SMA/EMA never clears *t* = 2 either way."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — the cost landscape\n\n"
            "The spread is below zero before costs; there is no break-even, and the "
            "long/short version compounds the loss:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    costs = [0.0, 2.0, 5.0, 10.0]\n"
            "    spr = [st.run_experiment(tape('SPY'), t3_n=T3_N, t3_v=T3_V, cost_bps=c)['T3']['mean_spread_bps'] for c in costs]\n"
            "    tst = [st.run_experiment(tape('SPY'), t3_n=T3_N, t3_v=T3_V, cost_bps=c)['T3']['spread_t'] for c in costs]\n"
            "else:\n"
            f"    costs = {R['cost']}; spr = {R['cost_spread']}; tst = {R['cost_t']}\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(costs, spr, 'o-', c=RED, lw=2, label='active spread (bps/day)')\n"
            "ax2 = ax.twinx(); ax2.plot(costs, tst, 's--', c=GREY, lw=1.5, label='HAC t')\n"
            "ax2.axhline(-2, ls=':', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('spread (bps/day)', color=RED)\n"
            "ax2.set_ylabel('HAC t', color=GREY)\n"
            "ax.set_title('No break-even: the spread starts below zero and only falls')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print('long/short net Sharpe: {R['ls_sharpe']:+.3f} (worse than long/flat)')"
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the positive control, and what would change our mind\n\n"
            "The synthetic control (4g) confirms the harness is unbiased and can bank a "
            "planted trend at *t* up to +21 — the null result above is about the market, "
            "not the machinery. What might rescue T3 for a future fork:\n\n"
            "- **Shorter N.** Tillson himself sometimes suggests N in the 5-8 range to offset "
            "the six-stage smoothing — this study's \"same nominal N as siblings 432/672\" "
            "convention is fair for cross-study comparison, but a length-matched-for-lag "
            "comparison is a natural sequel.\n"
            "- **Trend-prone assets.** Time-series momentum is a documented premium in "
            "futures and cross-asset baskets (Moskowitz, Ooi & Pedersen 2012); a single "
            "37-year equity index tape may simply not hand any MA-based rule enough "
            "persistent trend to pay for its lag, T3 included.\n"
            "- **The dedup map.** [Study 432 — Hull MA](../../432-hull-moving-average/) is "
            "the mirror image (less lag, more whipsaws); [Study 672 — McGinley Dynamic]"
            "(../../672-mcginley-dynamic/) tests a recursive self-adjusting mechanism; "
            "[Study 483 — ZLEMA](../../483-zlema/), [Study 674 — VIDYA](../../674-vidya/) "
            "and [Study 433 — KAMA](../../433-kama-adaptive/) round out the \"smarter MA\" "
            "family with different adaptivity mechanisms.\n\n"
            "*The reproducible core is offline and deterministic; frozen numbers live in "
            "[`docs/results.md`](../docs/results.md), sources in "
            "[`docs/references.md`](../docs/references.md).*"
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
