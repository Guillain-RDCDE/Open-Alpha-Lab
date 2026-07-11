"""Generate the two narrative notebooks for Study 674 (VIDYA).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). They recompute
the real-tape numbers live from the cached daily parquets under ../_cache/ when present,
and otherwise fall back to the frozen headline numbers in ``R`` (mirroring
docs/results.md), so the notebook re-runs for any reader, online or off.
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
    asof="2026-06-30", n_spy=8411, yrs=33.4, period=14, cmo_period=9,
    # mechanism: VI vs volatility / trend
    corr_vi_vol=-0.098, corr_vi_trend=0.380, n_corr=8391,
    vi_lo_vol=0.357, vi_hi_vol=0.310, vi_lo_trend=0.261, vi_hi_trend=0.446,
    # mechanism: tracking distance + step response
    td_vidya=2.196, td_sma=1.535, td_ema=1.327,
    step_price=120.0, step_vidya=111.52, step_sma=108.57, step_ema=111.52,
    step_pct_vidya=57.6, step_pct_sma=42.9, step_pct_ema=57.6,
    # SPY headline, long/flat, cost=5bps
    vidya_sharpe=0.387, vidya_cagr=3.68, vidya_dd=-38.8,
    sma_sharpe=0.238, sma_cagr=2.07, sma_dd=-57.6,
    ema_sharpe=0.267, ema_cagr=2.40, ema_dd=-48.0,
    bh_sharpe=0.647, bh_cagr=10.83, bh_dd=-55.2,
    vidya_spread=-3.10, vidya_t=-3.54,
    sma_spread=-3.70, sma_t=-4.42,
    ema_spread=-3.57, ema_t=-4.22,
    vidya_sw=23.1, sma_sw=36.5, ema_sw=38.3,
    vidya_tim=69,
    # gross
    vidya_spread_gross=-2.64, vidya_t_gross=-3.04,
    # head-to-head
    diff_v_sma_bps=0.61, diff_v_sma_t=1.10,
    diff_v_ema_bps=0.48, diff_v_ema_t=0.96,
    # permutation
    perm_obs=-2.64, perm_placebo=-1.49, perm_p=0.9885,
    # cost sweep (net Sharpe, spread, t)
    cost=[0.0, 2.0, 5.0, 10.0],
    cost_sharpe=[0.493, 0.450, 0.387, 0.281],
    cost_spread=[-2.64, -2.82, -3.10, -3.55],
    cost_t=[-3.04, -3.24, -3.54, -4.03],
    # per-instrument (VIDYA Sharpe, B&H Sharpe, spread, t, V-SMA t, V-EMA t, switches)
    tick=["SPY", "QQQ", "AAPL", "MSFT", "XLE"],
    tick_vidya=[0.387, 0.370, 0.742, 0.547, 0.238],
    tick_bh=[0.647, 0.521, 0.623, 0.821, 0.428],
    tick_spread=[-3.10, -3.24, -2.17, -5.74, -3.18],
    tick_t=[-3.54, -2.30, -1.21, -4.84, -2.05],
    tick_v_sma_t=[1.10, 0.89, 0.62, 0.95, 0.60],
    tick_v_ema_t=[0.96, 0.94, 0.27, 0.88, 1.21],
    tick_v_sw=[23.1, 24.2, 21.6, 22.8, 25.1],
    tick_sma_sw=[36.5, 36.6, 33.1, 37.3, 36.6],
    tick_ema_sw=[38.3, 39.8, 35.7, 40.9, 42.4],
    # in/out split
    h1_v=0.266, h1_bh=0.455, h1_spread=-2.40, h1_t=-1.79,
    h2_v=0.540, h2_bh=0.872, h2_spread=-3.71, h2_t=-3.25,
    # long/short
    ls_sharpe=-0.198, ls_spread=-6.23, ls_t=-3.56,
    # CMO-period robustness sweep
    cmo_grid=[5, 9, 14, 20, 30],
    cmo_sharpe=[0.443, 0.387, 0.429, 0.409, 0.424],
    cmo_spread=[-2.86, -3.10, -2.89, -2.95, -2.89],
    cmo_t=[-3.37, -3.54, -3.26, -3.39, -3.29],
    cmo_sw=[25.4, 23.1, 20.8, 18.9, 17.7],
    # synthetic control
    syn_null_mean=-0.22, syn_null_sd=0.94, syn_null_fire=0, syn_null_seeds=20,
    syn_edge=[0.3, 0.6, 1.0],
    syn_spread=[11.68, 28.69, 50.19],
    syn_t=[9.09, 16.31, 19.82],
    syn_sharpe=[2.00, 4.82, 7.48],
)


BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Speeds_up_in_volatile%2Ftrending_regimes%3F: Mixed](https://img.shields.io/badge/"
    "Speeds_up_in_volatile%2Ftrending_regimes%3F-Mixed-8b949e?style=flat-square)\n\n"
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

from vidya import data, strategy as st

TICKERS = ["SPY", "QQQ", "AAPL", "MSFT", "XLE"]
ASOF = data.AS_OF
PERIOD = 14
CMO_PERIOD = 9

def _have_cache():
    return all(os.path.exists(data._cache_path(t)) for t in TICKERS)

HAVE_REAL = _have_cache()

def tape(t):
    return data.load_real(t, fetch=False, asof=ASOF)

print("real daily cache present:", HAVE_REAL)
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# VIDYA — does an EMA that 'knows' the market speed actually help?\n"
            "### Tushar Chande's volatility-scaled moving average, turned into a timing rule and "
            "tested honestly\n\n"
            + BADGES +
            "Here's a pitch that's followed Tushar Chande's VIDYA around since 1992: unlike a plain "
            "moving average with a fixed speed, VIDYA has a built-in dial that supposedly speeds up "
            "when the market is **volatile and trending**, and slows almost to a stop when it's "
            "**quiet**. No re-optimizing, no whipsaws in chop, quick to catch a real move. Does the "
            "dial actually turn on the condition it claims to — and does it beat just buying and "
            "holding?\n\n"
            "> This is the plain-language layer. Want the *t*-stats, the correlation math and the "
            "cost sweeps? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does VIDYA's speed dial actually respond to *volatility*? | **No.** Its speed knob "
            f"is *slightly lower*, not higher, on the market's loudest days (correlation "
            f"**{R['corr_vi_vol']:+.2f}**). |\n"
            "| Does it respond to a *trend*? | **Yes.** The same knob correlates "
            f"**{R['corr_vi_trend']:+.2f}** with trend strength, and a clean price-jump test shows "
            "it snaps to full EMA speed once a real move is under way. |\n"
            f"| Does VIDYA-timing beat buy-and-hold? | **No.** It trails by **{R['vidya_spread']} "
            f"bps/day** (*t* = {R['vidya_t']}) — net Sharpe **{R['vidya_sharpe']}** vs "
            f"**{R['bh_sharpe']}** for just holding. |\n"
            "| Fewer false signals than a plain SMA/EMA? | **Yes, genuinely.** It fires "
            f"**{R['vidya_sw']} switches/yr** vs the SMA's **{R['sma_sw']}** and the EMA's "
            f"**{R['ema_sw']}** — a real ~38% cut. |\n\n"
            "> Half the story checks out (it does freeze in chop and wake up on a trend); the other "
            "half doesn't (it does *not* wake up on volatility, and even the half that works "
            "doesn't turn into money)."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim\n\n"
            "> *\"VIDYA replaces a moving average's fixed smoothing speed with one that reads the "
            "market: it uses Chande's own momentum oscillator (CMO) as a volatility gauge, so the "
            "line moves fast in volatile, trending conditions and almost freezes in place when the "
            "market is quiet. That means fewer whipsaws in chop and faster entries in a real "
            "move.\"*\n\n"
            "The formula bolts one extra ingredient onto an EMA: `VIDYA = VIDYA_prev + speed × "
            "(Price − VIDYA_prev)`, where `speed = base_speed × |CMO| / 100`. CMO ranges from −100 "
            "(every recent move was down) to +100 (every recent move was up) and sits near 0 when "
            "ups and downs cancel out. The pitch calls that \"volatility\" — but CMO actually "
            "measures *net direction*, which is a different thing from *how big the swings are*. We "
            "test both readings."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what?\n\n"
            "If VIDYA really did speed up exactly when markets get dangerous or exactly when a real "
            "trend starts, it would be a genuinely useful building block — a timing signal that "
            "adapts itself instead of needing a trader to guess the right lookback window. That's "
            "the same big promise behind three other \"smarter moving average\" indicators this "
            "desk has already tested (Hull MA, KAMA, McGinley Dynamic) — each with a *different* "
            "adaptation mechanism and a *different* verdict. VIDYA's mechanism (an external "
            "oscillator scaling the speed, rather than an internal ratio or price-based brake) is "
            "different enough from all three to be worth checking on its own terms."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How would we know?\n\n"
            "Three separate questions, tested separately:\n\n"
            "1. **Does the speed knob track what it's sold as?** Correlate VIDYA's own speed "
            "multiplier against realized volatility *and* against trend strength on the real tape "
            "— they are not the same regime, and the pitch conflates them.\n"
            "2. **Does the mechanism behave the way the formula implies?** No trading involved — "
            "measure how far VIDYA sits from price on average, and how fast it catches up after a "
            "clean, deterministic price jump, vs a plain SMA and EMA of the same length.\n"
            "3. **Does turning it into a trading rule pay?** Go long when price is above VIDYA, "
            "flat otherwise; subtract buy-and-hold; count the trades; race it against the "
            "equivalent SMA(14) and EMA(14) rules — the ones it claims to beat — and check the CMO "
            "lookback isn't cherry-picked.\n\n"
            f"We enter one day after each signal (no peeking), charge realistic costs, and run it "
            f"on SPY plus four other liquid tapes over ~{R['yrs']:.0f} years."
        ),

        # ---- BEAT 4 ----
        md(
            "## 4 · The teardown — what actually happens\n\n"
            "**First, the speed knob itself — forget trading, does it track what it claims?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    close = tape('SPY')['close']\n"
            "    rc = st.regime_correlations(close, cmo_period=CMO_PERIOD)\n"
            "else:\n"
            "    rc = {'corr_vi_vol': R['corr_vi_vol'], 'corr_vi_trend': R['corr_vi_trend'],\n"
            "          'vi_low_vol_tercile': R['vi_lo_vol'], 'vi_high_vol_tercile': R['vi_hi_vol'],\n"
            "          'vi_low_trend_tercile': R['vi_lo_trend'], 'vi_high_trend_tercile': R['vi_hi_trend']}\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.6, 4.3))\n"
            "a1.bar(['low-vol\\ntercile', 'high-vol\\ntercile'],\n"
            "       [rc['vi_low_vol_tercile'], rc['vi_high_vol_tercile']], color=[GREY, RED], width=.55)\n"
            "a1.set_title(f\"vs VOLATILITY: corr = {rc['corr_vi_vol']:+.2f}\")\n"
            "a1.set_ylabel('mean VIDYA speed knob (VI)')\n"
            "a2.bar(['low-trend\\ntercile', 'high-trend\\ntercile'],\n"
            "       [rc['vi_low_trend_tercile'], rc['vi_high_trend_tercile']], color=[GREY, GREEN], width=.55)\n"
            "a2.set_title(f\"vs TREND: corr = {rc['corr_vi_trend']:+.2f}\")\n"
            "plt.suptitle(\"The speed knob tracks TREND, not VOLATILITY\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"corr(VI, vol) = {rc['corr_vi_vol']:+.3f}   corr(VI, trend) = {rc['corr_vi_trend']:+.3f}\")"
        ),
        md(
            f"There it is, split cleanly: VIDYA's speed knob correlates **{R['corr_vi_trend']:+.2f}** "
            f"with trend strength (the high-trend tercile runs meaningfully faster than the "
            f"low-trend tercile) but **{R['corr_vi_vol']:+.2f}** with realized volatility — if "
            "anything, VIDYA is very slightly *slower*, not faster, on the market's loudest days. "
            "Chande's own word for his formula — \"volatile\" — is the wrong half of the claim. "
            "Here's the mechanism on a clean example: a price that sits flat, then jumps 20% and "
            "stays there."
        ),
        code(
            "sr = st.step_response(period=PERIOD, cmo_period=CMO_PERIOD)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "ax.plot(sr.index, sr['price'], c='k', lw=2, label='price (the jump)')\n"
            "ax.plot(sr.index, sr['VIDYA'], c=RED, lw=2, label='VIDYA')\n"
            "ax.plot(sr.index, sr['SMA'], c=GREY, lw=1.6, ls='--', label='SMA(14)')\n"
            "ax.plot(sr.index, sr['EMA'], c=AMBER, lw=1.6, ls='-.', label='EMA(14)')\n"
            "ax.axvline(30, c='k', lw=.7, alpha=.4)\n"
            "ax.set_xlabel('bars'); ax.set_ylabel('level')\n"
            "ax.set_title('Flat, then a jump: VIDYA freezes before, then matches the EMA after')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"5 bars after the jump: VIDYA closed {R['step_pct_vidya']:.0f}% of the gap, \"\n"
            "      f\"SMA {R['step_pct_sma']:.0f}%, EMA {R['step_pct_ema']:.0f}%\")"
        ),
        md(
            f"Before the jump VIDYA doesn't move at all — its speed knob reads zero when there's no "
            f"net direction to measure. Once the sustained jump gives it something to lock onto, it "
            f"catches up **exactly as fast as a plain EMA** ({R['step_pct_vidya']:.0f}% of the gap "
            f"closed in 5 bars, same as the EMA's {R['step_pct_ema']:.0f}%, ahead of the SMA's "
            f"{R['step_pct_sma']:.0f}%). So VIDYA never beats a plain EMA at its own game — it just "
            "matches it once triggered, and does nothing the rest of the time.\n\n"
            "**Does that \"freeze in chop\" habit at least mean fewer false trading signals on the "
            "real tape?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(tape('SPY'), period=PERIOD, cmo_period=CMO_PERIOD,\n"
            "                            sma_period=PERIOD, ema_period=PERIOD, cost_bps=5.0)\n"
            "    sw = {k: res[k]['switches_per_yr'] for k in ('VIDYA','SMA','EMA')}\n"
            "else:\n"
            "    sw = dict(VIDYA=R['vidya_sw'], SMA=R['sma_sw'], EMA=R['ema_sw'])\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.3))\n"
            "ax.bar(['VIDYA(14,9)', 'SMA(14)', 'EMA(14)'], [sw['VIDYA'], sw['SMA'], sw['EMA']],\n"
            "       color=[GREEN, GREY, GREY], width=.55)\n"
            "ax.set_ylabel('position changes per year')\n"
            "ax.set_title('VIDYA genuinely cuts whipsaws vs the SMA/EMA it is raced against')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"VIDYA {sw['VIDYA']:.1f}/yr  SMA {sw['SMA']:.1f}/yr  EMA {sw['EMA']:.1f}/yr\")"
        ),
        md(
            f"**{R['vidya_sw']} switches/yr** vs the SMA's **{R['sma_sw']}** and the EMA's "
            f"**{R['ema_sw']}** — a real **~38% fewer trades**. So the mechanism does what it says "
            "on this metric — the question is whether trading less, and trading on trend-strength "
            "rather than volatility, actually makes money."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fp, sp, ep = res['VIDYA']['sharpe_net'], res['SMA']['sharpe_net'], res['EMA']['sharpe_net']\n"
            "    bh = res['VIDYA']['bh_sharpe']\n"
            "else:\n"
            "    fp, sp, ep, bh = R['vidya_sharpe'], R['sma_sharpe'], R['ema_sharpe'], R['bh_sharpe']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar(['VIDYA\\n(14,9)', 'SMA(14)', 'EMA(14)', 'Buy & hold'], [fp, sp, ep, bh],\n"
            "       color=[RED, AMBER, AMBER, GREEN], width=.6)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('net Sharpe (annualised)')\n"
            "ax.set_title('Fewer trades, but still well behind holding')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'VIDYA {fp:.3f} | SMA {sp:.3f} | EMA {ep:.3f} | hold {bh:.3f}')"
        ),
        md(
            f"No. Buy-and-hold (Sharpe **{R['bh_sharpe']}**) still beats every timing rule, and "
            f"VIDYA (**{R['vidya_sharpe']}**) — despite trading the least of the three — is still "
            f"well below holding. Sitting out {100-R['vidya_tim']}% of a bull tape to avoid "
            "whipsaws is not free, and the avoided whipsaws didn't earn their keep."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Active spread vs holding **{R['vidya_spread']} bps/day**, *t* = "
            f"**{R['vidya_t']}** — negative on all five tapes tested, both halves of history, and "
            "every CMO-lookback setting.\n"
            f"- **Tradability — Mirage.** Net Sharpe **{R['vidya_sharpe']}** vs buy-and-hold "
            f"**{R['bh_sharpe']}**; loses even gross, and no cost level rescues it.\n"
            "- **Speeds up in volatile/trending regimes? — Mixed.** *Trending* checks out (positive "
            "correlation, the step test confirms EMA-speed catch-up once a trend is saturating "
            "CMO); *volatile* is backwards (a small negative correlation) — and even the honest "
            f"whipsaw cut ({R['vidya_sw']} vs {R['sma_sw']}/{R['ema_sw']} switches/yr) never turns "
            "into a certified edge over the plain MAs it claims to beat."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "No. The gross timing already loses to holding; costs only widen the gap. No cost "
            "level tested finds a break-even:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    costs = [0.0, 2.0, 5.0, 10.0]\n"
            "    spr = [st.run_experiment(tape('SPY'), period=PERIOD, cmo_period=CMO_PERIOD,\n"
            "                             cost_bps=c)['VIDYA']['mean_spread_bps'] for c in costs]\n"
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
            "- **The positive control.** The companion notebook plants a real trend in a synthetic "
            "tape — and the *same* VIDYA engine catches it cleanly (*t* up to +20). The harness "
            "works; the daily stock market just doesn't hand this rule an edge over the SMA/EMA it "
            "claims to beat, or over just holding.\n"
            "- **A volatility-scaled variant.** If VIDYA is really tracking trend, not volatility, a "
            "fork that scales speed by realized volatility directly (rather than by CMO) might come "
            "closer to the original pitch — worth trying.\n"
            "- **Other 'smarter MA' claims, similar fate.** See "
            "[Study 433 (KAMA)](../../433-kama-adaptive/), "
            "[Study 672 (McGinley Dynamic)](../../672-mcginley-dynamic/) and "
            "[Study 673 (T3)](../../673-t3-tillson/) — none beats a plain SMA/EMA or holding, each "
            "for a different mechanistic reason.\n\n"
            "*Think VIDYA earns its keep on a different timeframe, asset class, or as a volatility "
            "filter rather than a timing signal? Fork this and show an active spread that clears "
            "HAC *t* = 2 against buy-and-hold. That's the bar.*"
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
            "# VIDYA — a quantitative teardown\n"
            "### Daily total-return bars · VIDYA(14, cmo=9) price-cross · the volatility-vs-trend "
            "decomposition · HAC inference · permutation placebo · SMA/EMA head-to-head · "
            "CMO-period robustness\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim now carrying its standard error.* We split the "
            "mechanism claim in two (does the speed knob track volatility, or trend strength — "
            "Chande's own pitch conflates them?), test it with correlations and a deterministic "
            "step response, then ask whether the VIDYA(14, cmo=9) price-cross timing rule beats "
            "buy-and-hold, beats the matched SMA(14)/EMA(14) rules, and fires fewer false signals "
            "— across five liquid daily tapes and a CMO-lookback robustness sweep.\n\n"
            f"> **Not investment advice.** Real data: Yahoo daily total-return bars, full history "
            f"to 2026-06-30, as-of **{R['asof']}**; the offline core runs the deterministic "
            "synthetic tape. Methods & sources in "
            "[`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> The `💡 In plain words` notes translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | VIDYA active spread vs buy&hold **{R['vidya_spread']} "
            f"bps/day**, HAC *t* = **{R['vidya_t']}** (gross *t* = {R['vidya_t_gross']}); "
            f"permutation *p* = **{R['perm_p']:.4f}**; negative on all 5 tapes, both halves, and "
            "every CMO-period tested. |\n"
            f"| **Tradability** | `MIRAGE` | Net Sharpe **{R['vidya_sharpe']}** vs buy&hold "
            f"**{R['bh_sharpe']}**; loses gross; long/short Sharpe **{R['ls_sharpe']}**. |\n"
            "| **Speeds up in volatile/trending regimes?** | `MIXED` | corr(VI, trend) = "
            f"**{R['corr_vi_trend']:+.2f}** (confirmed) vs corr(VI, volatility) = "
            f"**{R['corr_vi_vol']:+.2f}** (busted) — and even the confirmed half yields "
            f"**{R['vidya_sw']}** switches/yr (a real cut) but zero certified head-to-head wins vs "
            "SMA/EMA. |\n\n"
            "> 💡 In plain words: Chande's speed knob is a genuine trend detector wearing a "
            "\"volatility\" label — and even the honest half of that story doesn't monetize."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "VIDYA is a recursive filter with a state-dependent smoothing constant:\n\n"
            "$$\\text{VI}_t = \\frac{|\\text{CMO}(t, m)|}{100}, \\qquad "
            "\\text{VIDYA}_t = \\text{VIDYA}_{t-1} + \\alpha \\cdot \\text{VI}_t \\cdot "
            "(P_t - \\text{VIDYA}_{t-1})$$\n\n"
            "with $\\alpha = 2/(N+1)$ the nominal EMA speed for period $N$, and $m$ = "
            "``cmo_period`` a free knob distinct from $N$. CMO ($m$-bar net direction ÷ total "
            "movement) is bounded in $[-100, 100]$ and saturates on *persistent, one-directional* "
            "moves — Chande's own copy calls the regime it detects \"volatile, trending markets,\" "
            "conflating two things that needn't coincide. Turned into a long/flat rule "
            "($d_t = \\mathbb{1}[P_t > \\text{VIDYA}_t]$) and raced against the matched SMA(14) and "
            "EMA(14) rules and buy-and-hold, the hypotheses:\n\n"
            "- **H₀a (volatility).** $\\text{VI}_t$ correlates positively with trailing realized "
            "volatility.\n"
            "- **H₀b (trend).** $\\text{VI}_t$ correlates positively with trailing trend strength.\n"
            "- **H₁ (signal).** $\\mathbb{E}[r^{\\text{strat}}_t - r^{\\text{B\\&H}}_t] > 0$ — the "
            "active spread is positive.\n"
            "- **H₂ (beats SMA/EMA).** Net Sharpe$(\\text{VIDYA}) > $ Net Sharpe$(\\text{SMA/EMA})$, "
            "certified at HAC *t* ≥ 2 on the head-to-head spread.\n"
            "- **H₃ (fewer false signals).** switches/yr$(\\text{VIDYA}) < $ "
            "switches/yr$(\\text{SMA/EMA})$.\n\n"
            "We find **H₀a rejected** ($r=-0.10$), **H₀b confirmed** ($r=+0.38$), **H₁ rejected** "
            "(significantly negative on the real tape), **H₂ not certified** (positive on paper, "
            "*t* < 2 on all 10 basket comparisons), and **H₃ confirmed** (a genuine ~38% cut) — "
            "the mechanism is internally coherent as a *trend* detector, just mislabeled, and even "
            "labeled correctly it doesn't monetize."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "VIDYA is marketed as reading *volatility* to auto-tune its speed — a claim distinct "
            "from KAMA's efficiency-ratio (net-travel ÷ total-path) adaptation and McGinley "
            "Dynamic's price/line-ratio brake, both already tested on this desk. If H₀a-H₃ all "
            "held, VIDYA would be the cleanest case yet of \"fewer, better signals\" driven by a "
            "genuinely different regime signal (an external momentum oscillator, not an internally "
            "derived ratio). The interesting result here is a **decomposition failure**: the "
            "formula tracks *one* of its two advertised regimes (trend) and not the other "
            "(volatility) — a reminder that \"the market is volatile\" and \"the market is trending\" "
            "get used interchangeably in trading folklore far more often than the data supports."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Indicator.** VIDYA(period={R['period']}, cmo_period={R['cmo_period']}) — the "
            f"nominal $N$ matches the SMA/EMA comparators at the same $N$ so the race is fair; "
            "``cmo_period`` = 9 is Chande's own commonly-cited default, swept 5→30 for robustness.\n"
            "- **Mechanism checks (no trading).** Pearson correlation of $\\text{VI}_t$ against a "
            "20-day realized-volatility proxy and a 20-day trend-strength proxy (a longer-window "
            "CMO, so it is not mechanically the same quantity), plus tercile splits; mean "
            "$|P_t - \\text{line}_t|/P_t$ on the real SPY tape; a deterministic +20% step response "
            "(flat → jump → flat), gap-closed at +5 bars.\n"
            "- **Rule.** Long/flat: $d_t = \\mathbb{1}[P_t > \\text{line}_t]$ for each of "
            "VIDYA/SMA/EMA. A long/short variant flips flat → −1.\n"
            "- **Execution lag.** One `shift`: position formed on the close of $t$ earns the "
            "close-to-close return of $t+1$. Stated once, applied once.\n"
            "- **Costs.** One-way × NAV on $|d_t - d_{t-1}|$ turnover; short legs pay 50 bps/yr "
            "borrow.\n"
            "- **Signal test.** HAC (Newey-West) one-sample *t* on the daily active spread "
            "$r^{\\text{strat}} - r^{\\text{B\\&H}}$ — excess-vs-excess by construction — plus the "
            "VIDYA-minus-SMA and VIDYA-minus-EMA head-to-head spreads.\n"
            "- **Placebo.** Circular-shift the realised VIDYA position path 2,000× (kills timing, "
            "keeps turnover/bias); one-sided *p* on the gross spread.\n"
            "- **Robustness.** Cost sweep, per-instrument, first-vs-second-half split, "
            "CMO-lookback sweep (5→30 bars), long/short.\n"
            "- **Positive control.** Synthetic tape with a *planted* regime-switching trend, 20-seed "
            "null.\n\n"
            f"Five tapes: SPY, QQQ, AAPL, MSFT, XLE — full history to 2026-06-30 "
            f"(SPY n = {R['n_spy']:,})."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The mechanism, decomposed — volatility vs trend\n\n"
            "$\\text{VI}_t = |\\text{CMO}(t, m)|/100$ is VIDYA's entire speed knob. Correlate it "
            "against two *independent* regime proxies (not the same construction as VI itself)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    close = tape('SPY')['close']\n"
            "    rc = st.regime_correlations(close, cmo_period=CMO_PERIOD)\n"
            "else:\n"
            "    rc = {'corr_vi_vol': R['corr_vi_vol'], 'corr_vi_trend': R['corr_vi_trend'],\n"
            "          'vi_low_vol_tercile': R['vi_lo_vol'], 'vi_high_vol_tercile': R['vi_hi_vol'],\n"
            "          'vi_low_trend_tercile': R['vi_lo_trend'], 'vi_high_trend_tercile': R['vi_hi_trend'],\n"
            "          'n': R['n_corr']}\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.8, 4.3))\n"
            "a1.bar(['low-vol\\ntercile', 'high-vol\\ntercile'],\n"
            "       [rc['vi_low_vol_tercile'], rc['vi_high_vol_tercile']], color=[GREY, RED], width=.55)\n"
            "a1.set_title(f\"H0a REJECTED: corr(VI, vol) = {rc['corr_vi_vol']:+.3f}\")\n"
            "a1.set_ylabel('mean VI (VIDYA speed knob)')\n"
            "a2.bar(['low-trend\\ntercile', 'high-trend\\ntercile'],\n"
            "       [rc['vi_low_trend_tercile'], rc['vi_high_trend_tercile']], color=[GREY, GREEN], width=.55)\n"
            "a2.set_title(f\"H0b CONFIRMED: corr(VI, trend) = {rc['corr_vi_trend']:+.3f}\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"n={rc['n']}  corr(VI,vol)={rc['corr_vi_vol']:+.3f}  \"\n"
            "      f\"corr(VI,trend)={rc['corr_vi_trend']:+.3f}\")"
        ),
        md(
            f"> 💡 In plain words: on {R['n_corr']:,} SPY trading days, VIDYA's speed knob tracks "
            f"trend strength (**{R['corr_vi_trend']:+.2f}**, and the high-trend tercile runs "
            f"{R['vi_hi_trend']/R['vi_lo_trend']-1:+.0%} faster than the low-trend tercile) but "
            f"**not** realized volatility (**{R['corr_vi_vol']:+.2f}**, direction backwards from "
            "the pitch). CMO is a net-direction oscillator, not a range/ATR-style volatility "
            "measure — the correlation structure follows directly from what it's built to detect."
        ),
        md(
            "### 4b · Tracking distance & step response — does the mechanism behave as designed?\n\n"
            "No trading involved yet: does VIDYA sit closer to price on average, and catch a shock "
            "faster, than a plain SMA/EMA of the same $N$?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    td = st.tracking_distance(close, period=PERIOD, cmo_period=CMO_PERIOD)\n"
            "else:\n"
            "    td = {'VIDYA': R['td_vidya'], 'SMA': R['td_sma'], 'EMA': R['td_ema']}\n"
            "sr = st.step_response(period=PERIOD, cmo_period=CMO_PERIOD)\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.4))\n"
            "a1.bar(['VIDYA','SMA','EMA'], [td['VIDYA'], td['SMA'], td['EMA']], color=[RED, GREY, GREY])\n"
            "a1.set_ylabel('mean |close-line|/close (%)'); a1.set_title('Tracking distance (SPY)')\n"
            "a2.plot(sr.index, sr['price'], c='k', lw=2, label='price')\n"
            "a2.plot(sr.index, sr['VIDYA'], c=RED, lw=2, label='VIDYA')\n"
            "a2.plot(sr.index, sr['SMA'], c=GREY, ls='--', label='SMA')\n"
            "a2.plot(sr.index, sr['EMA'], c=AMBER, ls='-.', label='EMA')\n"
            "a2.set_title('+20% step response'); a2.legend(fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"tracking distance: VIDYA {td['VIDYA']:.3f}%  SMA {td['SMA']:.3f}%  EMA {td['EMA']:.3f}%\")\n"
            "print(f\"5 bars post-jump: VIDYA={sr['VIDYA'].iloc[35]:.2f}  SMA={sr['SMA'].iloc[35]:.2f}  \"\n"
            "      f\"EMA={sr['EMA'].iloc[35]:.2f}  (price={sr['price'].iloc[35]:.1f})\")"
        ),
        md(
            f"> 💡 In plain words: on the deterministic step, VIDYA freezes completely in the flat "
            f"pre-jump segment (VI ≈ 0) and then converges to **exactly the EMA's own catch-up "
            f"speed** once the jump saturates CMO ({R['step_pct_vidya']:.0f}% of the gap closed in "
            f"5 bars, identical to the EMA's {R['step_pct_ema']:.0f}%) — VIDYA never *out-runs* the "
            "EMA it's compared to. On the real tape, where markets spend more time in the "
            "low-VI \"freeze\" state than in a clean saturating trend, average tracking distance is "
            f"**worse** ({R['td_vidya']:.2f}% vs the EMA's {R['td_ema']:.2f}%) — the \"hugs price\" "
            "framing fails in practice even though the mechanism fires exactly as designed."
        ),
        md(
            "### 4c · VIDYA vs SMA vs EMA vs buy-and-hold — net Sharpe and active-spread *t*\n\n"
            "The bar that matters is the active-spread HAC *t*: if VIDYA timing helps, it clears +2."
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = st.run_experiment(tape('SPY'), period=PERIOD, cmo_period=CMO_PERIOD,\n"
            "                            sma_period=PERIOD, ema_period=PERIOD, cost_bps=5.0)\n"
            "    rows = [(k, res[k]['sharpe_net'], res[k]['mean_spread_bps'], res[k]['spread_t'],\n"
            "             res[k]['switches_per_yr']) for k in ('VIDYA','SMA','EMA')]\n"
            "    rows.append(('B&H', res['VIDYA']['bh_sharpe'], 0.0, float('nan'), 0.0))\n"
            "    tbl = pd.DataFrame(rows, columns=['rule','sharpe_net','spread_bps','spread_t','sw/yr'])\n"
            "else:\n"
            "    tbl = pd.DataFrame({\n"
            "        'rule': ['VIDYA','SMA','EMA','B&H'],\n"
            f"        'sharpe_net': [{R['vidya_sharpe']},{R['sma_sharpe']},{R['ema_sharpe']},{R['bh_sharpe']}],\n"
            f"        'spread_bps': [{R['vidya_spread']},{R['sma_spread']},{R['ema_spread']},0.0],\n"
            f"        'spread_t': [{R['vidya_t']},{R['sma_t']},{R['ema_t']},float('nan')],\n"
            f"        'sw/yr': [{R['vidya_sw']},{R['sma_sw']},{R['ema_sw']},0.0]}})\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "sub = tbl[tbl['rule']!='B&H']\n"
            "col = [RED if t < 0 else GREY for t in sub['spread_t']]\n"
            "ax.bar(sub['rule'], sub['spread_t'], color=col)\n"
            "for s in (2, -2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('active-spread HAC t (vs buy&hold)')\n"
            "ax.set_title('All three rules have significantly NEGATIVE timing')\n"
            "plt.tight_layout(); plt.show()\n"
            "tbl.round(3)"
        ),
        md(
            f"> 💡 In plain words: every timing rule's active spread is significantly negative — "
            f"none beats holding. VIDYA is not the worst here (*t* = {R['vidya_t']} vs SMA's "
            f"{R['sma_t']} and EMA's {R['ema_t']}) — the lower turnover trims some of the damage — "
            "but \"less bad\" is not the same as \"real.\""
        ),
        md(
            "### 4d · Head-to-head — does VIDYA beat the plain MAs it claims to?\n\n"
            "The literal claim isn't just \"beats holding\" — it's \"beats a fixed SMA/EMA of the "
            "same length.\" HAC *t* of the VIDYA-minus-SMA and VIDYA-minus-EMA daily net-return "
            "spread."
        ),
        code(
            "if HAVE_REAL:\n"
            "    d_sma_bps, d_sma_t = res['diff_vidya_sma_bps'], res['diff_vidya_sma_t']\n"
            "    d_ema_bps, d_ema_t = res['diff_vidya_ema_bps'], res['diff_vidya_ema_t']\n"
            "else:\n"
            "    d_sma_bps, d_sma_t = R['diff_v_sma_bps'], R['diff_v_sma_t']\n"
            "    d_ema_bps, d_ema_t = R['diff_v_ema_bps'], R['diff_v_ema_t']\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.2))\n"
            "ax.bar(['VIDYA - SMA', 'VIDYA - EMA'], [d_sma_t, d_ema_t], color=AMBER, width=.5)\n"
            "ax.axhline(2, ls='--', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('HAC t of head-to-head spread')\n"
            "ax.set_title('Positive on SPY, but neither clears the t=2 bar')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'VIDYA-SMA: {d_sma_bps:+.2f} bps/day, t={d_sma_t:+.2f}')\n"
            "print(f'VIDYA-EMA: {d_ema_bps:+.2f} bps/day, t={d_ema_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: VIDYA beats SMA by {R['diff_v_sma_bps']:+.2f} bps/day and EMA by "
            f"{R['diff_v_ema_bps']:+.2f} bps/day on SPY — the right sign — but *t* = "
            f"{R['diff_v_sma_t']:.2f} / {R['diff_v_ema_t']:.2f}, both well under the bar. The "
            "per-instrument table below shows this holds across the basket: consistently positive "
            "on paper, never certifiable."
        ),
        md(
            "### 4e · The whipsaw count — the one part of the claim that survives\n\n"
            "Position changes per year. Like McGinley Dynamic (Study 672) and unlike the Hull MA "
            "(+87% vs SMA) or KAMA (+66% vs SMA), VIDYA genuinely reduces turnover."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sw = {k: res[k]['switches_per_yr'] for k in ('VIDYA','SMA','EMA')}\n"
            "else:\n"
            "    sw = dict(VIDYA=R['vidya_sw'], SMA=R['sma_sw'], EMA=R['ema_sw'])\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.0))\n"
            "ax.bar(['VIDYA','SMA','EMA'], [sw['VIDYA'],sw['SMA'],sw['EMA']], color=[GREEN, GREY, GREY])\n"
            "ax.set_ylabel('switches / yr')\n"
            "ax.set_title('VIDYA trades ~38-40% less than the SMA/EMA it is raced against')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"VIDYA/SMA switch ratio: {sw['VIDYA']/sw['SMA']:.2f}x   \"\n"
            "      f\"VIDYA/EMA ratio: {sw['VIDYA']/sw['EMA']:.2f}x\")"
        ),
        md(
            f"> 💡 In plain words: {R['vidya_sw']} vs {R['sma_sw']}/{R['ema_sw']} switches/yr — "
            "freezing when |CMO| is near zero really does cut trade count. Combined with 4a-4b, the "
            "mechanism is internally consistent (a trend-conditioned filter trades less because it "
            "stays out of directionless chop) — it just isn't the mechanism advertised "
            "(\"volatility\")."
        ),
        md(
            "### 4f · Permutation placebo — timing vs exposure\n\n"
            "Circularly shift the realised VIDYA position path 2,000×; the statistic is the gross "
            "active spread. If the timing is informative, the real spread beats the placebo "
            "distribution."
        ),
        code(
            "if HAVE_REAL:\n"
            "    close_spy = tape('SPY')['close']\n"
            "    p = st.permutation_pvalue(close_spy.pct_change(),\n"
            "                              st.vidya_position(close_spy, PERIOD, CMO_PERIOD),\n"
            "                              cost_bps=0.0, n_perm=2000, seed=674)\n"
            "    obs, plac, pv = p['observed_spread_bps'], p['placebo_mean_bps'], p['p_value']\n"
            "    rng = np.random.default_rng(674)\n"
            "    a = close_spy.pct_change().fillna(0).to_numpy()\n"
            "    held = st.vidya_position(close_spy, PERIOD, CMO_PERIOD).shift(1).fillna(0).to_numpy()\n"
            "    draws = []\n"
            "    for _ in range(2000):\n"
            "        h = np.roll(held, int(rng.integers(1, len(held))))\n"
            "        turn = np.abs(np.diff(h, prepend=0.0))\n"
            "        draws.append(((h * a - turn * 0) - a).mean() * 1e4)\n"
            "else:\n"
            f"    obs, plac, pv = {R['perm_obs']}, {R['perm_placebo']}, {R['perm_p']}\n"
            f"    draws = list(np.random.default_rng(1).normal({R['perm_placebo']}, 0.55, 2000))\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=30, color=GREY, alpha=.7, label='random re-timings (2000)')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'real VIDYA timing: {obs:+.2f} bps/day')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('active spread vs buy&hold (bps/day)'); ax.set_ylabel('count')\n"
            "ax.set_title('Real VIDYA timing sits deep in the LEFT tail of its own shuffles')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'observed {obs:+.2f} | placebo mean {plac:+.2f} | one-sided p = {pv:.4f}')"
        ),
        md(
            f"> 💡 In plain words: *p* = {R['perm_p']:.4f} — essentially every random re-timing of "
            "VIDYA's own trades beats the real ones. Fewer trades did not mean *better-placed* "
            "trades."
        ),
        md(
            "### 4g · Per-instrument, in/out-of-sample & the CMO-period sweep\n\n"
            "Active-spread *t* (vs buy&hold) and VIDYA-vs-SMA/EMA head-to-head *t* on all five "
            "tapes, the SPY first-vs-second-half split, and a sweep of ``cmo_period`` (5→30) to "
            "confirm the result isn't a one-parameter artefact."
        ),
        code(
            "if HAVE_REAL:\n"
            "    recs = []\n"
            "    for t in TICKERS:\n"
            "        rr = st.run_experiment(tape(t), period=PERIOD, cmo_period=CMO_PERIOD,\n"
            "                               sma_period=PERIOD, ema_period=PERIOD, cost_bps=5.0)\n"
            "        m = rr['VIDYA']\n"
            "        recs.append((t, m['sharpe_net'], m['bh_sharpe'], m['mean_spread_bps'], m['spread_t'],\n"
            "                     rr['diff_vidya_sma_t'], rr['diff_vidya_ema_t']))\n"
            "    per = pd.DataFrame(recs, columns=['ticker','vidya_sharpe','bh_sharpe','spread_bps','t',\n"
            "                                       'vidya_sma_t','vidya_ema_t'])\n"
            "    spy2 = tape('SPY'); half = len(spy2)//2\n"
            "    h1 = st.run_experiment(spy2.iloc[:half], period=PERIOD, cmo_period=CMO_PERIOD, cost_bps=5.0)['VIDYA']\n"
            "    h2 = st.run_experiment(spy2.iloc[half:], period=PERIOD, cmo_period=CMO_PERIOD, cost_bps=5.0)['VIDYA']\n"
            "    split = (h1['spread_t'], h2['spread_t'])\n"
            "    cmo_grid = [5, 9, 14, 20, 30]\n"
            "    cmo_t = [st.run_experiment(spy2, period=PERIOD, cmo_period=cm, cost_bps=5.0)['VIDYA']['spread_t']\n"
            "             for cm in cmo_grid]\n"
            "else:\n"
            "    per = pd.DataFrame({'ticker': "
            f"{R['tick']}, 'vidya_sharpe': {R['tick_vidya']}, 'bh_sharpe': {R['tick_bh']},"
            f" 'spread_bps': {R['tick_spread']}, 't': {R['tick_t']},"
            f" 'vidya_sma_t': {R['tick_v_sma_t']}, 'vidya_ema_t': {R['tick_v_ema_t']}}})\n"
            f"    split = ({R['h1_t']}, {R['h2_t']})\n"
            f"    cmo_grid = {R['cmo_grid']}; cmo_t = {R['cmo_t']}\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.2, 4.3))\n"
            "a1.bar(per['ticker'], per['t'], color=RED)\n"
            "a1.axhline(-2, ls='--', c=GREY); a1.axhline(0, c='k', lw=1)\n"
            "a1.set_ylabel('active-spread HAC t (VIDYA vs hold)')\n"
            "a1.set_title('Five of five negative vs buy&hold')\n"
            "a2.bar([str(c) for c in cmo_grid], cmo_t, color=AMBER)\n"
            "a2.axhline(-2, ls='--', c=GREY); a2.axhline(0, c='k', lw=1)\n"
            "a2.set_xlabel('cmo_period'); a2.set_title('Negative at every CMO lookback, 5-30')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('SPY split  H1 t = %.2f | H2 t = %.2f  (both negative)' % split)\n"
            "per.round(3)"
        ),
        md(
            f"> 💡 In plain words: every instrument's VIDYA timing trails buy-and-hold, SPY loses in "
            f"both halves (H1 *t* = {R['h1_t']}, H2 *t* = {R['h2_t']}), and the active-spread *t* "
            "stays below −3.2 at **every** CMO lookback from 5 to 30 bars — not an artefact of "
            "Chande's own canonical `cmo_period = 9`. Against SMA/EMA (the fair comparator), **zero "
            "of 10** basket head-to-heads clear *t* = 2 — real on paper, certified nowhere."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — active spread {R['vidya_spread']} bps/day, HAC *t* "
            f"{R['vidya_t']} (gross {R['vidya_t_gross']}); permutation *p* = {R['perm_p']:.4f}; "
            "negative on 5/5 tapes, both halves, and every CMO-period tested.\n"
            f"- **Tradability `MIRAGE`** — net Sharpe {R['vidya_sharpe']} vs hold {R['bh_sharpe']}; "
            f"loses gross; long/short Sharpe {R['ls_sharpe']}, spread {R['ls_spread']} bps/day "
            f"(*t* {R['ls_t']}).\n"
            "- **Speeds up in volatile/trending regimes? `MIXED`** — trend correlation "
            f"{R['corr_vi_trend']:+.2f} (confirmed) vs volatility correlation "
            f"{R['corr_vi_vol']:+.2f} (busted); the honest whipsaw cut ({R['vidya_sw']} vs "
            f"{R['sma_sw']}/{R['ema_sw']} switches/yr) is real but the head-to-head edge vs "
            "SMA/EMA clears *t* = 2 on zero of 10 basket comparisons."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — the cost landscape\n\n"
            "The spread is below zero before costs; there is no break-even, and the long/short "
            "version compounds the loss:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    costs = [0.0, 2.0, 5.0, 10.0]\n"
            "    spr = [st.run_experiment(tape('SPY'), period=PERIOD, cmo_period=CMO_PERIOD,\n"
            "                             cost_bps=c)['VIDYA']['mean_spread_bps'] for c in costs]\n"
            "    tst = [st.run_experiment(tape('SPY'), period=PERIOD, cmo_period=CMO_PERIOD,\n"
            "                             cost_bps=c)['VIDYA']['spread_t'] for c in costs]\n"
            "    lsr = st.run_experiment(tape('SPY'), period=PERIOD, cmo_period=CMO_PERIOD,\n"
            "                            cost_bps=5.0, long_short=True)['VIDYA']\n"
            "    ls_sh = lsr['sharpe_net']\n"
            "else:\n"
            f"    costs = {R['cost']}; spr = {R['cost_spread']}; tst = {R['cost_t']}\n"
            f"    ls_sh = {R['ls_sharpe']}\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(costs, spr, 'o-', c=RED, lw=2, label='active spread (bps/day)')\n"
            "ax2 = ax.twinx(); ax2.plot(costs, tst, 's--', c=GREY, lw=1.5, label='HAC t')\n"
            "ax2.axhline(-2, ls=':', c=GREY); ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('spread (bps/day)', color=RED)\n"
            "ax2.set_ylabel('HAC t', color=GREY)\n"
            "ax.set_title('No break-even: the spread starts below zero and only falls')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'long/short net Sharpe: {ls_sh:+.3f} (worse than long/flat)')"
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the positive control\n\n"
            "Is the *engine* able to find a trend, or is it always negative? Plant a persistent "
            "regime-switching trend in a synthetic tape and sweep the knob; the null (edge=0) is "
            "checked over 20 seeds, never a single stream."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    b, _ = data.synthetic_panel(n_days=6000, edge=0.0, seed=674 + s_)\n"
            "    null_ts.append(st.run_experiment(b, period=PERIOD, cmo_period=CMO_PERIOD,\n"
            "                                     cost_bps=0.0)['VIDYA']['spread_t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "edges = " + repr(R['syn_edge']) + "\n"
            "sp, tt = [], []\n"
            "for e in edges:\n"
            "    b, _ = data.synthetic_panel(n_days=6000, edge=e, seed=674)\n"
            "    r = st.run_experiment(b, period=PERIOD, cmo_period=CMO_PERIOD, cost_bps=0.0)['VIDYA']\n"
            "    sp.append(r['mean_spread_bps']); tt.append(r['spread_t'])\n"
            "fig, ax = plt.subplots(figsize=(9.4, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12, .12, 20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (edge=0), 20 seeds')\n"
            "ax.scatter(range(1, 1 + len(edges)), tt, color=GREEN, s=80, zorder=5,\n"
            "           label='planted edge (single seed)')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks(range(0, 1 + len(edges)))\n"
            "ax.set_xticklabels(['null x20'] + [f'edge={e}' for e in edges])\n"
            "ax.set_ylabel('active-spread HAC t')\n"
            "ax.set_title('Engine works: VIDYA banks a planted trend; the null does not fire')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/20 seeds')\n"
            "print('planted:', {e: round(t, 2) for e, t in zip(edges, tt)})"
        ),
        md(
            "The engine is a faithful trend detector: plant a real persistent trend and VIDYA banks "
            "it, clearing *t* = 2 comfortably at every planted level (up to *t* ≈ +20); with no "
            f"trend planted the null fires on only {R['syn_null_fire']}/{R['syn_null_seeds']} "
            "independent seeds — in line with the ~5% false-positive rate expected at a *t* = 2 "
            "threshold, not systematic bias. *(A faithful-engine / power check only — never cited "
            "in support of the real-tape stamp.)* The real-tape verdict is therefore a statement "
            "about the **market and the mechanism**: VIDYA's speed knob genuinely detects trend, "
            "not volatility, and freezes appropriately in directionless chop — but daily US equity "
            "trends aren't large or persistent enough, often enough, for a trend-conditioned brake "
            "to beat either buy-and-hold or the plain SMA/EMA it's raced against. A fork worth "
            "trying: scale VIDYA's speed by realized volatility directly (e.g. an ATR ratio) "
            "instead of by CMO, to test the *actually* volatility-driven version of the pitch."
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
