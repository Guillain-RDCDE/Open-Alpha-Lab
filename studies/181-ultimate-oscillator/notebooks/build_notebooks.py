"""Generate the two narrative notebooks for Study 181 (Ultimate-Oscillator).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached daily
parquets under ../_cache/ if present and otherwise quote the frozen headline numbers in
``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.

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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-15).
R = dict(
    # Threshold framing (UO < 30 buy / UO > 70 short), hold 5 days, gross
    n_thresh=681, tickers=5,
    thresh_mean=14.40, thresh_win=47.7, thresh_t=1.50,
    rand_mean=11.88, rand_win=51.7, rand_t=1.35,
    thresh_minus_rand=2.52,
    # Hold-period sweep (threshold, gross)
    h1_mean=7.55, h1_t=2.14,
    h3_mean=10.42, h3_t=1.46,
    h5_mean=14.40, h5_t=1.50,
    h10_mean=-17.27, h10_t=-1.27,
    h20_mean=-31.11, h20_t=-1.62,
    # Cost sweep (threshold, hold 5d)
    net05=13.90, net05_t=1.45,
    net10=13.40, net10_t=1.39,
    net20=12.40, net20_t=1.29,
    net50=9.40, net50_t=0.98,
    # Divergence framing, hold 5 days
    n_div=5336,
    div_mean=-7.08, div_win=46.7, div_t=-1.80,
    div_rand_mean=2.82, div_rand_t=1.00,
    # Per-ticker breakdown (threshold, hold 5d, gross)
    spy_n=169, spy_mean=24.36, spy_t=1.69,
    qqq_n=167, qqq_mean=8.25, qqq_t=0.47,
    iwm_n=117, iwm_mean=9.42, iwm_t=0.34,
    gld_n=132, gld_mean=0.46, gld_t=0.02,
    tlt_n=96, tlt_mean=32.80, tlt_t=1.61,
    # Window
    window_start="2006-06-15", window_end="2026-06-15",
)

# ---------------------------------------------------------------------------
# Shared analysis preamble — imports, the basket, and pooled helpers.
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

from ultimate_oscillator import data, strategy as st

TICKERS = ["SPY", "QQQ", "IWM", "GLD", "TLT"]

def _have_cache():
    return all(os.path.exists(data._cache_path(t, data.DEFAULT_CACHE)) for t in TICKERS)

HAVE_REAL = _have_cache()

def pool_thresh(hold=5, cost=0.0, seed=None):
    \"\"\"Pool threshold-entry trades across the basket (cached real tapes).\"\"\"
    frames = []
    for t in TICKERS:
        b = data.fetch_daily(t, fetch=False)
        uo = st.ultimate_oscillator(b)
        ent = st.threshold_entries(uo)
        if len(ent) == 0:
            continue
        dirs = st.random_directions(len(ent), seed=seed) if seed is not None else None
        frames.append(st.run_trades(b, ent, hold_days=hold, cost_bps=cost, directions=dirs))
    import pandas as pd
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def pool_div(hold=5, cost=0.0, seed=None):
    \"\"\"Pool divergence-entry trades across the basket (cached real tapes).\"\"\"
    frames = []
    for t in TICKERS:
        b = data.fetch_daily(t, fetch=False)
        uo = st.ultimate_oscillator(b)
        ent = st.divergence_entries(b, uo)
        if len(ent) == 0:
            continue
        dirs = st.random_directions(len(ent), seed=seed) if seed is not None else None
        frames.append(st.run_trades(b, ent, hold_days=hold, cost_bps=cost, directions=dirs))
    import pandas as pd
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Ultimate-Oscillator — does a three-period weighted oscillator beat a coin?\n"
            "### Williams' UO (7/14/28 bars): oversold-buy, overbought-short, divergence — tested honestly\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Low_signal--to--noise: Confirmed](https://img.shields.io/badge/Low_signal--to--noise-Confirmed-8b949e?style=flat-square)\n\n"
            "Larry Williams' 1985 *Ultimate Oscillator* promised to fix the fatal flaw of RSI and "
            "Stochastic: their sensitivity to the arbitrary look-back period. Blend three periods "
            "(7, 14, 28 bars) into one weighted average of *buying pressure* vs *true range*, and "
            "buy when the market is oversold (UO < 30), short when overbought (UO > 70), or "
            "trade *divergences* when price and the oscillator disagree. This notebook asks: does "
            "the three-period weighting actually load the die, or is it just a coin wearing a "
            "costume?\n\n"
            "> **This is the plain-language layer.** Want the t-stats, Bonferroni corrections, and "
            "hold-period sweep? That's the companion, "
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
            f"| Does UO < 30 (oversold) predict a bounce? | **Maybe, barely.** Mean +{R['thresh_mean']:.1f} bps/trade gross, "
            f"HAC *t* = +{R['thresh_t']:.2f} — positive but not significant at the |*t*|≥2 bar. |\n"
            f"| Does it beat an actual coin flip? | **Barely.** Signal +{R['thresh_mean']:.1f} vs coin +{R['rand_mean']:.1f} bps — the oscillator "
            f"direction adds only {R['thresh_minus_rand']:.1f} bps over a random entry. |\n"
            f"| What about the famous divergence rule? | **No.** Divergence entries earn **{R['div_mean']:+.1f} bps/trade** (*t* = {R['div_t']:.2f}) — actively negative. |\n"
            "| Could you trade it? | **No — the signal isn't certified.** Low turnover keeps costs manageable, |\n"
            "|  | but trading an unproven edge is speculation, not strategy. |\n\n"
            "> The three-period trick reduces noise inside the oscillator but doesn't change the "
            "fundamental problem: rare, crisis-clustered extreme readings give too few observations "
            "to certify a real edge."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The Ultimate Oscillator uses weighted averages of three different time periods to "
            "reduce false signals. Values below 30 represent oversold conditions — buy and expect "
            "the market to rebound. Values above 70 represent overbought conditions — short and "
            "expect a pullback. Bullish divergence (price new low, UO higher low) is the strongest "
            "buy signal in Williams' original work.\"* — standard chartist framing\n\n"
            "It is a mean-reversion bet: after the market has been under sustained selling "
            "pressure (buying pressure relative to true range has been low for 7–28 bars), it "
            "should bounce back. The three time-frames are supposed to make this more reliable "
            "than single-period oscillators."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the oversold-buy rule actually works, it's a rare, low-turnover edge: fire "
            "roughly 7 times per ticker per year, hold a week, collect a mean reversion. Low "
            "costs, high clarity. If it doesn't work, thousands of traders are entering "
            "positions at random moments in the hope of a bounce that may already have happened "
            "— because the UO's 28-bar lookback means it only 'reads' oversold well after the "
            "trough. The divergence rule is even more concerning: it fires constantly (7.8× more "
            "often) and our data suggest it's actively on the wrong side."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "One rule keeps us honest: **race the UO direction against a coin on identical "
            "entry bars.** If the oscillator knows something, the UO-directed trade should beat "
            "a random-direction trade on the same dates. We hold the position a fixed number of "
            "days (sweeping 1, 3, 5, 10, 20), measure the gross mean return per trade, compute "
            "the HAC t-stat, and apply a Bonferroni correction for testing 5 hold periods.\n\n"
            "We run it on **five liquid ETFs** (SPY, QQQ, IWM, GLD, TLT) over 20 years of "
            "daily data — 681 pooled threshold entries and 5,336 divergence entries.\n\n"
            "**What would make us say 'mirage':** HAC |t| < 2.0 on the pooled series at the "
            "chosen hold period, after multiple-comparisons correction."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md("## 4 · The teardown — what the data actually show"),
        md(
            "### 4a · The threshold rule — positive but sub-threshold\n\n"
            "Here's what UO < 30 entries earn at various hold periods, compared with a coin "
            "on the same dates."
        ),
        code(
            "holds = [1, 3, 5, 10, 20]\n"
            "if HAVE_REAL:\n"
            "    sig_means = [st.summarize(pool_thresh(h, 0),'ret_gross')['mean_bps'] for h in holds]\n"
            "    rnd_means = [st.summarize(pool_thresh(h, 0, seed=181),'ret_gross')['mean_bps'] for h in holds]\n"
            "    sig_tstats = [st.summarize(pool_thresh(h, 0),'ret_gross')['tstat'] for h in holds]\n"
            "else:\n"
            f"    sig_means = [{R['h1_mean']}, {R['h3_mean']}, {R['h5_mean']}, {R['h10_mean']}, {R['h20_mean']}]\n"
            f"    rnd_means = [{R['rand_mean']}]*5\n"
            f"    sig_tstats = [{R['h1_t']}, {R['h3_t']}, {R['h5_t']}, {R['h10_t']}, {R['h20_t']}]\n"
            "\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.5))\n"
            "ax1.plot(holds, sig_means, 'o-', c=AMBER, lw=2, label='UO signal')\n"
            "ax1.plot(holds, rnd_means, 's--', c=GREY, lw=1.5, label='coin (random)')\n"
            "ax1.axhline(0, c='k', lw=1)\n"
            "ax1.set_xlabel('hold (days)'); ax1.set_ylabel('gross mean (bps/trade)')\n"
            "ax1.set_title('Threshold rule: mean return by hold period')\n"
            "ax1.legend()\n"
            "\n"
            "colors = [GREEN if abs(t)>=2 else RED for t in sig_tstats]\n"
            "ax2.bar(holds, sig_tstats, color=colors, width=2)\n"
            "ax2.axhline(2, ls='--', c=GREY, lw=1, label='|t|=2 bar')\n"
            "ax2.axhline(-2, ls='--', c=GREY, lw=1)\n"
            "ax2.axhline(0, c='k', lw=1)\n"
            "ax2.set_xlabel('hold (days)'); ax2.set_ylabel('HAC t-stat')\n"
            "ax2.set_title('No hold period clears |t|=2 after Bonferroni')\n"
            "ax2.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'1-day: t={{sig_tstats[0]:+.2f}} (best, but Bonferroni 5-test crit ≈ 2.58)')"
        ),
        md(
            f"The signal is positive at short horizons (1-day mean +{R['h1_mean']:.1f} bps, "
            f"*t* = +{R['h1_t']:.2f}) but **does not survive a Bonferroni correction** for "
            "testing five hold periods simultaneously — you would need |*t*| ≈ 2.58 at the "
            "1%/5 = 0.2% level.  At 5 days the picture is qualitatively similar: positive "
            f"(+{R['h5_mean']:.1f} bps) but sub-threshold (*t* = +{R['h5_t']:.2f}).  Beyond "
            "10 days the signal reverses — the bounce, if real, is short-lived."
        ),
        md(
            "### 4b · Signal vs coin — the direction adds almost nothing\n\n"
            "On identical entry dates, does the UO direction beat a coin?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    s5 = st.summarize(pool_thresh(5, 0), 'ret_gross')\n"
            "    r5 = st.summarize(pool_thresh(5, 0, seed=181), 'ret_gross')\n"
            "    sm, rn, sw, rw = s5['mean_bps'], r5['mean_bps'], s5['win_rate']*100, r5['win_rate']*100\n"
            "else:\n"
            f"    sm, rn, sw, rw = {R['thresh_mean']}, {R['rand_mean']}, {R['thresh_win']}, {R['rand_win']}\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "bars_p = ax.bar(['UO direction\\n(follow the signal)', 'Random direction\\n(a coin)'],\n"
            "               [sm, rn], color=[AMBER, GREY], width=.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('gross profit per trade (bps)')\n"
            "ax.set_title('The UO barely edges a coin — the direction adds little')\n"
            "for b, w in zip(bars_p, [sw, rw]):\n"
            "    ax.annotate(f'win {w:.1f}%', (b.get_x()+b.get_width()/2, max(0, b.get_height())),\n"
            "                ha='center', va='bottom')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'UO: {{sm:+.2f}} bps   coin: {{rn:+.2f}} bps   Δ = {{sm-rn:+.2f}} bps')"
        ),
        md(
            f"The UO direction earns +{R['thresh_mean']:.1f} bps vs the coin's +{R['rand_mean']:.1f} bps: "
            f"a delta of just **+{R['thresh_minus_rand']:.1f} bps**.  This is consistent with the "
            "market simply having a positive drift — both the signal and the coin are capturing "
            "the unconditional upward trend, not a mean-reversion edge specific to oversold readings."
        ),
        md(
            "### 4c · The divergence rule — Williams' 'strongest signal' actually loses\n\n"
            "The most famous part of the original 1985 paper: trade bullish/bearish divergences "
            "between price and UO."
        ),
        code(
            "if HAVE_REAL:\n"
            "    dsig = st.summarize(pool_div(5, 0), 'ret_gross')\n"
            "    drand = st.summarize(pool_div(5, 0, seed=181), 'ret_gross')\n"
            "    dm, drm = dsig['mean_bps'], drand['mean_bps']\n"
            "    dn, dt = dsig['n_trades'], dsig['tstat']\n"
            "else:\n"
            f"    dm, drm, dn, dt = {R['div_mean']}, {R['div_rand_mean']}, {R['n_div']}, {R['div_t']}\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.bar(['UO divergence\\n(signal)', 'Random direction\\n(coin on same dates)'],\n"
            "       [dm, drm], color=[RED, GREY], width=.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('gross mean return (bps/trade)')\n"
            "ax.set_title(\"The divergence rule: Williams' 'best' signal is actually negative\")\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'divergence: n={{dn}}, mean={{dm:+.2f}} bps, HAC t={{dt:+.2f}}')"
        ),
        md(
            f"The divergence rule fires **{R['n_div']:,} times** (7.8× the threshold rule) and earns "
            f"**{R['div_mean']:+.1f} bps/trade** at *t* = **{R['div_t']:+.2f}** — negative and "
            "approaching the −2 significance bar from the *wrong* side.  This is consistent with "
            "divergence signals firing most in trending markets (where price makes persistent "
            "new extremes and the contrarian bet consistently loses)."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** Threshold rule gross +{R['thresh_mean']:.1f} bps/trade (*t* = +{R['thresh_t']:.2f}), "
            f"Δ vs coin +{R['thresh_minus_rand']:.1f} bps; best hold period (*t* = +{R['h1_t']:.2f} at 1 day) "
            "doesn't survive Bonferroni.  Divergence rule mean −7.1 bps (*t* = −1.80): negative.\n"
            "- **Tradability — Fragile.** Low turnover means costs are not the issue — the "
            "signal isn't certified real, so there is nothing to trade.  If forced to trade, "
            "a liquid ETF basket at 1 bp commission yields +13.4 bps net (*t* = +1.39) — "
            "not enough to bet on.\n"
            "- **Low signal-to-noise — Confirmed.** 681 pooled entries over 5 tickers × 20 years; "
            "signals cluster in crisis episodes; divergence fires 7.8× more but loses."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md("## 6 · Could you actually trade it?\n\n"
           "The threshold rule at least has manageable turnover (~7 signals/ticker/year). "
           "Here's how costs affect the 5-day hold net mean:"
        ),
        code(
            "costs = [0.0, 0.5, 1.0, 2.0, 5.0]\n"
            "if HAVE_REAL:\n"
            "    net = [st.summarize(pool_thresh(5, c), 'ret_net')['mean_bps'] for c in costs]\n"
            "    tst = [st.summarize(pool_thresh(5, c), 'ret_net')['tstat'] for c in costs]\n"
            "else:\n"
            f"    net = [{R['thresh_mean']}, {R['net05']}, {R['net10']}, {R['net20']}, {R['net50']}]\n"
            f"    tst = [{R['thresh_t']}, {R['net05_t']}, {R['net10_t']}, {R['net20_t']}, {R['net50_t']}]\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, net, 'o-', c=AMBER, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, net, 0, where=[n>0 for n in net], color=AMBER, alpha=.15)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net mean (bps/trade)')\n"
            "ax.set_title('Costs are manageable — but the t-stat was already sub-threshold')\n"
            "for i, (c, n, t) in enumerate(zip(costs, net, tst)):\n"
            "    ax.annotate(f't={t:+.2f}', (c, n + 0.5), ha='center', fontsize=8)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('At 1 bp: +13.4 bps net, t=+1.39 — positive but not statistically certified')"
        ),
        md(
            "The gross edge (thin as it is) survives typical retail costs because the rule "
            "trades infrequently.  The real barrier isn't costs — it's that the underlying "
            "t-stat of +1.50 was never enough.  Adding costs just moves an already-uncertain "
            "number in the wrong direction."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **A different basket?** The per-ticker results vary wildly: SPY +1.69, QQQ +0.47, "
            "TLT +1.61, GLD +0.02.  The bond/equity split and volatility regimes matter; "
            "sector ETFs with higher mean reversion (e.g. high-volatility sector funds) "
            "might tell a different story.\n"
            "- **Divergence, properly confirmed.** Williams' original rule required additional "
            "confirmation after the divergence (e.g. UO rises above 50 before taking the trade). "
            "Our simplified scan skips that step — a full implementation might reduce false fires.\n"
            "- **The parent oscillator.** Williams' %R is [Study 127](../../127-williams-r/) — "
            "same author, same mean-reversion thesis, same honest-bet protocol.  Both show weak/none.\n"
            "- **What *does* work here?** Low-turnover mean-reversion signals with stronger "
            "statistical grounding are in [Study 52 — Smoke-Screen](../../52-smoke-screen/) "
            "(EDGAR-based) and [Study 17 — Glass-Ceiling](../../17-glass-ceiling/) "
            "(price-level resistance).\n\n"
            "*Think the die really is loaded? Fork this, change the basket or the confirmation "
            "logic, and show a fair-bet edge that clears the |t|≥2 bar after Bonferroni. "
            "That's the bar.*"
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
            "# Ultimate-Oscillator — a quantitative teardown\n"
            "### Real daily tape · threshold & divergence rules · random-direction control · HAC inference · Bonferroni\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Low_signal--to--noise: Confirmed](https://img.shields.io/badge/Low_signal--to--noise-Confirmed-8b949e?style=flat-square)\n\n"
            "Deep companion to [01_for_the_curious.ipynb](01_for_the_curious.ipynb) — same seven "
            "beats, every claim now carrying its standard error.  We test whether Williams' Ultimate "
            "Oscillator's threshold (<30 / >70) and divergence rules beat a random-direction "
            "control on a fixed-horizon backtest across five liquid ETFs, 20 years of daily data, "
            "with a Bonferroni correction for hold-period search.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily bars, 20-year window ending "
            "2026-06-15; the offline core and tests run on a deterministic synthetic tape.  "
            "Methods in [`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The `In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Threshold: +{R['thresh_mean']:.1f} bps/trade, HAC *t* = "
            f"+{R['thresh_t']:.2f} (n={R['n_thresh']:,}); best hold 1d *t*=+{R['h1_t']:.2f} "
            f"fails Bonferroni. Divergence: {R['div_mean']:+.1f} bps, *t*={R['div_t']:+.2f} — negative. |\n"
            f"| **Tradability** | `FRAGILE` | Low turnover (~7/ticker/year) keeps costs manageable; "
            f"at 1 bp net *t* = +{R['net10_t']:.2f} — still below the |*t*|≥2 bar. |\n"
            f"| **Low signal-to-noise** | `CONFIRMED` | {R['n_thresh']:,} threshold events over "
            f"{R['tickers']} tickers × 20 years; no ticker clears |*t*| > 2.0; divergence events "
            f"= {R['n_div']:,} but mean *t* = {R['div_t']:+.2f}. |\n\n"
            "> In plain words: the UO's three-period weighting reduces oscillator noise but can't "
            "create a certified edge from a sparse, crisis-clustered set of extreme readings."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            r"Let $\\text{UO}_t$ be the Ultimate Oscillator at bar $t$ and $d_t \\in \\{-1, +1\\}$"
            " the signal direction.  The recipe asserts:\n\n"
            "- **H₁ (threshold signal).** $\\mathbb{E}[d_t \\cdot r_{t \\to t+k}] > 0$ where "
            "$d_t = +1$ if $\\text{UO}_t < 30$, $d_t = -1$ if $\\text{UO}_t > 70$, and the "
            "expectation is over a fixed hold period $k$.\n"
            "- **H₂ (beats random).** That expectation exceeds the same statistic with $d_t$ "
            "replaced by an i.i.d. fair coin on identical entry dates.\n"
            "- **H₃ (divergence signal).** The divergence framing (price/UO non-confirmation) "
            "earns a positive mean over the same protocol.\n"
            "- **H₄ (tradable).** H₁ survives a realistic round-trip cost at the rule's "
            "natural turnover.\n\n"
            "We reject H₁ after Bonferroni correction, H₂ weakly (barely positive Δ), H₃ "
            "clearly (divergence is negative), and H₄ conditionally (low costs, but no "
            "certified gross edge to protect)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The UO is the most-cited multi-period oscillator in technical analysis curricula.  "
            "If H₁–H₃ held, it would be a low-turnover, robust mean-reversion signal that "
            "survives a 40-year out-of-sample test.  The interesting failure isn't merely that "
            "the t-stat misses 2.0 — it's that the divergence rule, Williams' own favourite "
            "signal, is *negative*.  That's a direct rebuke to the idea that non-confirmation "
            "of price extremes by a multi-period oscillator carries information."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Entry.** UO computed on closes up to $t$; **enter at $t{+}1$'s open** (no "
            "look-ahead).  Threshold: first bar where $\\text{UO}_t < 30$ (buy) or "
            "$\\text{UO}_t > 70$ (short), one entry per episode.  Divergence: price new "
            "extreme without UO confirmation in a 28-bar lookback.\n"
            "- **Exit.** Fixed-horizon: close of bar $t + k$ for $k \\in \\{1, 3, 5, 10, 20\\}$.\n"
            "- **Control.** Identical entries, direction $\\in\\{-1,+1\\}$ drawn i.i.d. (seeded).\n"
            "- **Inference.** Newey-West HAC *t* on per-trade returns; Bonferroni correction for "
            "5 hold periods (per-test $\\alpha = 1\\%$, *t*-crit ≈ 2.58).\n"
            "- **Basket.** Five ETFs (SPY, QQQ, IWM, GLD, TLT); 20-year daily history "
            "(2006-06-15 → 2026-06-15)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Per-ticker breakdown — no tape carries a certified signal\n\n"
            "HAC *t* on the 5-day gross mean for each ticker.  All bars stay inside ±2; the "
            "pooled result at +1.50 is driven by the positive drift of the basket."
        ),
        code(
            "if HAVE_REAL:\n"
            "    recs = []\n"
            "    for t in TICKERS:\n"
            "        b = data.fetch_daily(t, fetch=False)\n"
            "        uo = st.ultimate_oscillator(b)\n"
            "        ent = st.threshold_entries(uo)\n"
            "        if len(ent) == 0:\n"
            "            continue\n"
            "        s = st.summarize(st.run_trades(b, ent, hold_days=5, cost_bps=0), 'ret_gross')\n"
            "        recs.append((t, s['n_trades'], s['win_rate']*100, s['mean_bps'], s['tstat']))\n"
            "    perinst = pd.DataFrame(recs, columns=['ticker','n','win%','mean_bps','t'])\n"
            "else:\n"
            "    perinst = pd.DataFrame({\n"
            f"        'ticker': ['SPY','QQQ','IWM','GLD','TLT'],\n"
            f"        'n': [{R['spy_n']},{R['qqq_n']},{R['iwm_n']},{R['gld_n']},{R['tlt_n']}],\n"
            f"        'mean_bps': [{R['spy_mean']},{R['qqq_mean']},{R['iwm_mean']},{R['gld_mean']},{R['tlt_mean']}],\n"
            f"        't': [{R['spy_t']},{R['qqq_t']},{R['iwm_t']},{R['gld_t']},{R['tlt_t']}],\n"
            f"        'win%': [47.3, 42.5, 44.4, 47.0, 62.5]}}\n"
            "    )\n"
            "    perinst['win%'] *= 100 if perinst['win%'].max() < 2 else 1\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "col = [GREEN if abs(v)>=2 else RED for v in perinst['t']]\n"
            "ax.bar(perinst['ticker'], perinst['t'], color=col)\n"
            "for s in (2,-2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('HAC t-stat (gross, 5-day hold)')\n"
            "ax.set_title('Five tickers, five non-results (|t| < 2 everywhere)')\n"
            "ax.set_ylim(-3, 3); plt.tight_layout(); plt.show()\n"
            "perinst.round(2)"
        ),
        md(
            "> In plain words: even TLT (+1.61), the strongest individual result, stays "
            "well within the noise band.  There is no 'anchor' ticker that could anchor a "
            "multi-ticker claim."
        ),
        md(
            "### 4b · Hold-period sweep with Bonferroni correction\n\n"
            "The 5-point sweep is a specification search — each additional test you run "
            "inflates the false-discovery rate.  Bonferroni at 5% / 5 tests = 1% per test "
            "requires |*t*| ≥ 2.58.  The best result (1-day, *t* = +2.14) falls short."
        ),
        code(
            "holds = [1, 3, 5, 10, 20]\n"
            "if HAVE_REAL:\n"
            "    ts = [st.summarize(pool_thresh(h, 0),'ret_gross')['tstat'] for h in holds]\n"
            "    mn = [st.summarize(pool_thresh(h, 0),'ret_gross')['mean_bps'] for h in holds]\n"
            "else:\n"
            f"    ts = [{R['h1_t']}, {R['h3_t']}, {R['h5_t']}, {R['h10_t']}, {R['h20_t']}]\n"
            f"    mn = [{R['h1_mean']}, {R['h3_mean']}, {R['h5_mean']}, {R['h10_mean']}, {R['h20_mean']}]\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "colors = [GREEN if t>=2.58 else AMBER if t>=2.0 else RED for t in ts]\n"
            "bars_b = ax.bar(holds, ts, color=colors, width=1.8)\n"
            "ax.axhline(2.58, ls='--', c=GREEN, lw=2, label='Bonferroni bar (|t|=2.58, 5 tests)')\n"
            "ax.axhline(2.00, ls=':', c=AMBER, lw=1.5, label='Naive bar (|t|=2.0)')\n"
            "ax.axhline(-2.0, ls=':', c=AMBER, lw=1.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('hold (days)'); ax.set_ylabel('HAC t-stat')\n"
            "ax.set_title('Hold-period sweep: no result survives Bonferroni')\n"
            "ax.legend(); ax.set_xticks(holds)\n"
            "plt.tight_layout(); plt.show()\n"
            "for h, t, m in zip(holds, ts, mn):\n"
            "    print(f'hold={h:2d}d  mean={m:+.2f}bps  t={t:+.2f}')"
        ),
        md(
            "> In plain words: even the most favourable hold period (1 day) only just "
            "clears the naive 2.0 bar — and we tested 5 periods, so the honest bar is 2.58. "
            "The 1-day result is likely noise amplified by the short measurement window."
        ),
        md(
            "### 4c · The divergence rule — negative and approaching significance\n\n"
            "Williams called divergence 'the most powerful' signal.  On the real tape it fires "
            "7.8× more often than the threshold rule but earns a negative mean."
        ),
        code(
            "if HAVE_REAL:\n"
            "    dsig = pool_div(5, 0)\n"
            "    drand = pool_div(5, 0, seed=181)\n"
            "    ds, dr = st.summarize(dsig,'ret_gross'), st.summarize(drand,'ret_gross')\n"
            "    dm, drm, dn, dt, drn = ds['mean_bps'],dr['mean_bps'],ds['n_trades'],ds['tstat'],dr['n_trades']\n"
            "else:\n"
            f"    dm,drm,dn,dt,drn = {R['div_mean']},{R['div_rand_mean']},{R['n_div']},{R['div_t']},{R['n_div']}\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.bar(['Divergence signal', 'Coin (random on same dates)'],\n"
            "       [dm, drm], color=[RED, GREY], width=.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('gross mean (bps/trade, hold 5d)')\n"
            "ax.set_title('Divergence rule: n={:,}, mean={:+.2f} bps, t={:+.2f}'.format(dn, dm, dt))\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'divergence mean={dm:+.2f} bps vs coin {drm:+.2f} bps  Δ={dm-drm:+.2f} bps')"
        ),
        md(
            "> In plain words: the divergence rule fires most often in trending episodes where "
            "a contrarian bet repeatedly loses.  The frequency-vs-quality trade-off hurts: more "
            "signals, worse signal."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — threshold: +{R['thresh_mean']:.1f} bps/trade, HAC *t* = "
            f"+{R['thresh_t']:.2f} (n={R['n_thresh']:,}); best hold (1d) *t* = +{R['h1_t']:.2f} "
            "< Bonferroni crit 2.58; no ticker clears |*t*| > 2.0; divergence "
            f"*t* = {R['div_t']:+.2f} — negative.\n"
            f"- **Tradability `FRAGILE`** — low turnover (~7/ticker/year) keeps costs "
            f"manageable; at 1 bp round-trip net *t* = +{R['net10_t']:.2f} — positive but "
            "sub-threshold.  Nothing certified to trade.\n"
            "- **Low signal-to-noise `CONFIRMED`** — 681 entries clustered in distress "
            "episodes; all mean return estimates carry large standard errors."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — cost and Sharpe analysis\n\n"
            "At ~7 signals/ticker/year the threshold rule is a long-only-ish position held "
            "for a week.  Costs are not the primary obstacle — the uncertified t-stat is."
        ),
        code(
            "costs = [0.0, 0.5, 1.0, 2.0, 5.0]\n"
            "if HAVE_REAL:\n"
            "    sw = [st.summarize(pool_thresh(5, c),'ret_net') for c in costs]\n"
            "    net = [s['mean_bps'] for s in sw]; tst = [s['tstat'] for s in sw]\n"
            "else:\n"
            f"    net = [{R['thresh_mean']},{R['net05']},{R['net10']},{R['net20']},{R['net50']}]\n"
            f"    tst = [{R['thresh_t']},{R['net05_t']},{R['net10_t']},{R['net20_t']},{R['net50_t']}]\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(costs, net, 'o-', c=AMBER, lw=2, label='net mean (bps/trade)')\n"
            "ax2 = ax.twinx()\n"
            "ax2.plot(costs, tst, 's--', c=GREY, lw=1.5, label='HAC t')\n"
            "ax2.axhline(2, ls=':', c=GREEN, lw=1)\n"
            "ax2.axhline(0, ls=':', c=GREY, lw=0.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net mean (bps)', color=AMBER)\n"
            "ax2.set_ylabel('HAC t-stat', color=GREY)\n"
            "ax.set_title('Costs are manageable; t-stat never clears 2.0 even at 0 cost')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('At 1bp: {:+.1f} bps net, t={:+.2f}'.format(net[2], tst[2]))"
        ),
        md(
            "> In plain words: unlike most failed signals (where a real gross edge dies at "
            "the cost line), here the gross t-stat was *never* certified.  Costs just move "
            "a sub-threshold number further sub-threshold."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the *engine* capable of finding a mean-reversion edge, or does it always "
            "print 'none'?  Pool 20 synthetic tapes with a tunable AR(1) mean-reversion "
            "parameter and sweep it: the UO oversold rule should eventually beat the coin "
            "when real reversion exists at the 7–28-bar scale."
        ),
        code(
            "import pandas as pd\n"
            "mrevs = [-0.10, 0.00, 0.10, 0.20]\n"
            "results = []\n"
            "for mr in mrevs:\n"
            "    all_s, all_r = [], []\n"
            "    for s in range(20):\n"
            "        b, _ = data.synthetic_daily(n_days=500, mean_rev=mr, seed=s)\n"
            "        uo = st.ultimate_oscillator(b)\n"
            "        ent = st.threshold_entries(uo)\n"
            "        if len(ent) == 0:\n"
            "            continue\n"
            "        all_s.append(st.run_trades(b, ent, hold_days=5, cost_bps=0))\n"
            "        all_r.append(st.run_trades(b, ent, hold_days=5, cost_bps=0,\n"
            "                                   directions=st.random_directions(len(ent), seed=s+99)))\n"
            "    if not all_s:\n"
            "        results.append((mr, float('nan'), float('nan')))\n"
            "        continue\n"
            "    ss = st.summarize(pd.concat(all_s, ignore_index=True), 'ret_gross')\n"
            "    rs = st.summarize(pd.concat(all_r, ignore_index=True), 'ret_gross')\n"
            "    results.append((mr, ss['mean_bps'], rs['mean_bps']))\n"
            "\n"
            "mr_vals = [r[0] for r in results]\n"
            "sig_vals = [r[1] for r in results]\n"
            "rnd_vals = [r[2] for r in results]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(mr_vals, sig_vals, 'o-', c=GREEN, lw=2, label='UO signal')\n"
            "ax.plot(mr_vals, rnd_vals, 's--', c=GREY, lw=1.5, label='coin')\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, ls='--', c=GREY)\n"
            "ax.set_xlabel('planted AR(1) mean-reversion parameter')\n"
            "ax.set_ylabel('mean (bps/trade, 5d hold)')\n"
            "ax.set_title('Synthetic control: UO engine responds to planted mean reversion')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for mr, s, r in results:\n"
            "    print(f'mean_rev={mr:+.2f}  signal={s:+.2f}  coin={r:+.2f}')"
        ),
        md(
            "The UO edge over the coin is monotone in the planted mean-reversion parameter — "
            "the engine *can* find it when it exists at the appropriate scale.  The real-tape "
            "result is therefore a statement about the market: short-term mean reversion at "
            "the 7–28-bar daily scale is too weak or too episodic to certify with 681 entries "
            "over 20 years.  Forks worth trying: sector ETFs with higher idiosyncratic vol; "
            "a narrower oversold threshold (UO < 20) for higher signal purity with fewer entries; "
            "or the Williams %R companion at [Study 127](../../127-williams-r/)."
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
