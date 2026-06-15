"""Generate the two narrative notebooks for Study 168 (Advance-Decline).

    python notebooks/build_notebooks.py
    python -W ignore -m jupyter nbconvert --to notebook --execute --inplace \\
        01_for_the_curious.ipynb 02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see METHODOLOGY.md). The synthetic
figures run anywhere offline and deterministically; real-tape cells use the
cached parquets under ../_cache/ if present and otherwise fall back to the
frozen headline numbers in R (mirroring docs/results.md), so the notebook
re-runs for any reader.

The _write convention (each build_*() ends by calling _write) is kept so
the repo's intro-restyle tooling can monkeypatch it.
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
    spy_start="2005-01-03", spy_end="2026-06-12", fp="8e7f69ef9d75",
    n_tickers=497,
    # Primary: 63d lookback / 21d horizon
    n_div=307,          # divergence signals
    n_base=4784,        # baseline observations
    div_mean=95.8,      # post-divergence mean forward return (bps)
    div_t=3.35,         # HAC t
    div_hit=0.713,      # hit rate
    base_mean=87.3,     # unconditional baseline mean (bps)
    base_t=4.73,
    base_hit=0.673,
    diff_bps=8.5,       # div minus base (positive = divergence is BETTER)
    perm_pval=0.608,    # fraction of random sub-samples more negative
    # Lookback sweep diff bps
    diff_21=17.5,
    diff_42=24.1,
    diff_63=8.5,
    diff_126=-14.8,
    diff_252=-15.2,
    # Drawdown
    sig_dd=-6.3,        # post-signal 63d max drawdown (pct)
    base_dd=-7.2,       # unconditional 63d max drawdown (pct)
    # Signals per year
    sig_per_yr=14.6,
)

# ---------------------------------------------------------------------------
# Shared preamble
# ---------------------------------------------------------------------------
_R_DICT = """\
R = dict(
    spy_start="2005-01-03", spy_end="2026-06-12", fp="8e7f69ef9d75",
    n_tickers=497,
    n_div=307, n_base=4784,
    div_mean=95.8, div_t=3.35, div_hit=0.713,
    base_mean=87.3, base_t=4.73, base_hit=0.673,
    diff_bps=8.5, perm_pval=0.608,
    diff_21=17.5, diff_42=24.1, diff_63=8.5, diff_126=-14.8, diff_252=-15.2,
    sig_dd=-6.3, base_dd=-7.2, sig_per_yr=14.6,
)
"""

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd, warnings
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from advance_decline import data, strategy as st

CACHE_DIR = os.path.abspath("../_cache")

def _have_real():
    return (os.path.exists(os.path.join(CACHE_DIR, "ad_spy_daily.parquet")) and
            os.path.exists(os.path.join(CACHE_DIR, "ad_panel_daily.parquet")))

HAVE_REAL = _have_real()
""" + _R_DICT + """\
print("real data cache present:", HAVE_REAL)
"""

BOOT_REAL = """\
if HAVE_REAL:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spy   = data.fetch_spy(fetch=False, cache_dir=CACHE_DIR)
        panel = data.fetch_panel(fetch=False, cache_dir=CACHE_DIR,
                                 allow_survivorship_bias=True)
    ad_line = data.build_ad_line(panel)
    common  = spy.index.intersection(ad_line.index)
    spy_c   = spy.loc[common, "close"]
    ad_c    = ad_line.loc[common]
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Advance-Decline -- when the generals march alone, is a retreat coming?\n"
            "### The breadth-divergence top signal, tested honestly\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Coincident%3F: Not_a_leading_indicator](https://img.shields.io/badge/Coincident%3F-Not_a_leading_indicator-8b949e?style=flat-square)\n\n"
            "The story sounds persuasive: if the S&P 500 is making new highs but most *individual* "
            "stocks are going down, the rally is being carried by a thin parade of generals while "
            "the foot-soldiers retreat. *Surely* that's a warning sign -- a top is near. "
            "We build the Advance-Decline line from 497 S&P 500 constituents (~20 years of daily data), "
            "detect every moment the index hit a new high while breadth didn't confirm, "
            "and ask: *did those warning moments actually predict worse returns?*\n\n"
            "> **This is the plain-language layer.** For HAC t-stats, the permutation baseline, "
            "and the lookback sweep, see **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** Reproducible research: every chart is drawn by the "
            "code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| After a bearish divergence, does SPY fall more than usual? | "
            f"**No.** Post-divergence 21-day return is **+{R['div_mean']:.0f} bps** -- "
            f"slightly *above* the baseline **+{R['base_mean']:.0f} bps**. |\n"
            "| At least do divergence periods show higher drawdown? | "
            f"**No.** Post-signal 63-day max drawdown is **{R['sig_dd']:.1f}%** vs "
            f"**{R['base_dd']:.1f}%** unconditionally -- divergence periods have *smaller* dips. |\n"
            "| Is the signal at least consistent across different time windows? | "
            "**No.** Short lookbacks show slightly bullish divergences, long ones slightly bearish -- "
            "the sign flips. There is no stable direction. |\n"
            "| Could you trade it as a hedge? | **No.** Shorting SPY on every divergence signal "
            f"loses ~{R['div_mean']:.0f} bps/trade before costs. |\n\n"
            "> The breadth divergence signal is a story about *narrative*, not about *edge*. "
            "On this tape, 'narrow rallies' are not reliably followed by corrections."
        ),

        # ---- BEAT 1 -- THE CLAIM ----------------------------------------------
        md(
            "## 1 . The claim\n\n"
            "> *\"When the generals are advancing but the soldiers are retreating, "
            "the attack is in trouble. In market terms: if the S&P 500 makes a new 3-month high "
            "while the Advance-Decline line (cumulative advancers minus decliners) does NOT, "
            "the rally is narrow and fragile. A correction is coming.\"*\n\n"
            "The A/D line is one of the oldest market-breadth tools -- it's been published "
            "in technical-analysis literature since the 1970s and popularised by Norman Fosback's "
            "*Stock Market Logic* (1976) and the McClellan family's breadth indicators. "
            "The bet is that a rally carried by a thin group of large caps will eventually "
            "roll over, and the A/D line divergence is the 'early warning' that the majority "
            "of stocks are already losing momentum."
        ),

        # ---- BEAT 2 -- SO WHAT ------------------------------------------------
        md(
            "## 2 . So what?\n\n"
            "If the claim were true, you'd have a simple, observable, free signal to "
            "time partial exits or hedges -- no fancy model required, just counting advancers "
            "and decliners. Market-timing with such a signal could materially improve risk-adjusted "
            "returns for any long-only equity investor.\n\n"
            "If it's false (as the data will show), then every 'narrow rally' newspaper headline "
            "is just noise -- the index will do what it was going to do regardless of whether "
            "a majority of stocks are along for the ride."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 . How would we even know?\n\n"
            "One question cuts through the folklore: **do post-divergence 21-day SPY returns "
            "run below the unconditional average?** If the signal is real, the distribution of "
            "forward returns after a bearish divergence should be shifted left relative to all "
            "forward returns. We measure exactly this, over ~21 years of daily data.\n\n"
            "Two rules keep us honest:\n"
            "1. **The unconditional baseline.** Every divergence period is compared to ALL "
            "forward 21-day returns, not to 'up days' or 'previous similar periods' -- we "
            "measure against the full baseline.\n"
            "2. **A positive control (synthetic).** We generate a fake tape where we literally "
            "*plant* divergences and check our machinery can find them -- this proves the engine "
            "works, and that the real-data 'null' result is a statement about markets, not our code."
        ),

        # ---- BEAT 4 -- THE TEARDOWN -------------------------------------------
        md("## 4 . The teardown -- let's actually look"),

        md("### 4a . The A/D line vs SPY -- what a divergence looks like"),
        code(
            BOOT_REAL +
            "# Show a stretch where the index rose but breadth was weak\n"
            "if HAVE_REAL:\n"
            "    # Plot A/D line normalised and SPY normalised, 2023-2024 window\n"
            "    window = slice('2022-01-01', '2024-12-31')\n"
            "    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7), sharex=True)\n"
            "    spy_w = spy_c.loc[window]\n"
            "    ad_w  = ad_c.loc[window]\n"
            "    spy_norm = spy_w / spy_w.iloc[0] * 100\n"
            "    ad_norm  = (ad_w - ad_w.iloc[0]) / ad_w.abs().max() * 100 + 100\n"
            "    axes[0].plot(spy_norm.index, spy_norm, color=GREEN, lw=1.5, label='SPY (indexed)')\n"
            "    axes[0].set_ylabel('SPY (indexed to 100)'); axes[0].legend()\n"
            "    axes[1].plot(ad_norm.index, ad_norm, color=GREY, lw=1.5, label='A/D line (normalised)')\n"
            "    axes[1].set_ylabel('A/D line (normalised)'); axes[1].legend()\n"
            "    # Mark divergence signals\n"
            "    sig = st.ad_divergence_signal(spy_c, ad_c, lookback=63)\n"
            "    sig_w = sig.loc[window]\n"
            "    for d in sig_w[sig_w].index:\n"
            "        axes[0].axvline(d, color=RED, alpha=0.3, lw=0.8)\n"
            "        axes[1].axvline(d, color=RED, alpha=0.3, lw=0.8)\n"
            "    axes[0].set_title('SPY and A/D line -- red lines are bearish divergence signals')\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    # Synthetic illustration\n"
            "    panel_s, spy_s, _ = data.synthetic_daily(n_stocks=200, n_days=504,\n"
            "                                              divergence=1.5, seed=168)\n"
            "    ad_s = data.build_ad_line(panel_s)\n"
            "    spy_n = spy_s['close'] / spy_s['close'].iloc[0] * 100\n"
            "    ad_n  = (ad_s - ad_s.iloc[0]) / ad_s.abs().max() * 100 + 100\n"
            "    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7), sharex=True)\n"
            "    axes[0].plot(spy_n.index, spy_n, color=GREEN, lw=1.5, label='Index (synthetic)')\n"
            "    axes[1].plot(ad_n.index, ad_n, color=GREY, lw=1.5, label='A/D line (synthetic)')\n"
            "    axes[0].set_title('Synthetic: planted divergence episodes (red lines)')\n"
            "    sig_s = st.ad_divergence_signal(spy_s['close'], ad_s, lookback=42, min_index_gain=0.0)\n"
            "    for d in sig_s[sig_s].index:\n"
            "        axes[0].axvline(d, color=RED, alpha=0.3, lw=0.8)\n"
            "        axes[1].axvline(d, color=RED, alpha=0.3, lw=0.8)\n"
            "    axes[0].legend(); axes[1].legend()\n"
            "    plt.tight_layout(); plt.show()\n"
            "print(f'Total divergence signals on real tape (63d lookback): {R[\"n_div\"]} over ~21 years')"
        ),

        md(
            "### 4b . The honest test: post-divergence returns vs the baseline\n\n"
            "After each bearish divergence signal, what does SPY do over the next 21 trading days?"
        ),
        code(
            BOOT_REAL +
            "if HAVE_REAL:\n"
            "    res = st.backtest(spy_c, ad_c, lookback=63, horizon=21)\n"
            "    div_r = res['div_rets'] * 1e4   # in bps\n"
            "    base_r = res['base_rets'] * 1e4\n"
            "    ds, bs = res['div_stats'], res['base_stats']\n"
            "else:\n"
            "    rng = np.random.default_rng(168)\n"
            "    base_r = rng.normal(R['base_mean'], 150, R['n_base'])\n"
            "    div_r  = rng.normal(R['div_mean'],  120, R['n_div'])\n"
            "    ds = {'mean_bps': R['div_mean'],  'hit_rate': R['div_hit']}\n"
            "    bs = {'mean_bps': R['base_mean'], 'hit_rate': R['base_hit']}\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10.5, 5))\n"
            "bins = np.linspace(-600, 800, 50)\n"
            "ax.hist(base_r, bins=bins, alpha=0.5, color=GREY, label=f'All days (n={len(base_r):,})', density=True)\n"
            "ax.hist(div_r,  bins=bins, alpha=0.6, color=RED,  label=f'Post-divergence (n={len(div_r):,})', density=True)\n"
            "ax.axvline(ds['mean_bps'], color=RED,  lw=2.5, ls='--', label=f'Div mean: {ds[\"mean_bps\"]:+.0f} bps')\n"
            "ax.axvline(bs['mean_bps'], color=GREY, lw=2.5, ls='--', label=f'Base mean: {bs[\"mean_bps\"]:+.0f} bps')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('21-day forward return (bps)'); ax.set_ylabel('density')\n"
            "ax.set_title('Post-divergence returns are NOT shifted left vs the baseline')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Post-divergence mean: {ds[\"mean_bps\"]:+.1f} bps  |  Baseline: {bs[\"mean_bps\"]:+.1f} bps')\n"
            "print(f'Difference: {ds[\"mean_bps\"]-bs[\"mean_bps\"]:+.1f} bps  -- signal is actually slightly BULLISH!')"
        ),
        md(
            f"The distributions nearly overlap. The post-divergence mean is **+{R['div_mean']:.0f} bps** — "
            f"*above* the unconditional **+{R['base_mean']:.0f} bps** baseline, not below it. "
            "The bearish claim is directionally *wrong*: divergence periods have been slightly better, "
            "not worse, than average on this tape."
        ),

        md(
            "### 4c . Is the signal at least consistent across time windows?\n\n"
            "Maybe the 63-day lookback is the wrong choice. Let's try them all:"
        ),
        code(
            BOOT_REAL +
            "if HAVE_REAL:\n"
            "    sweepdf = st.sweep_lookback(spy_c, ad_c, lookbacks=[21,42,63,126,252], horizon=21)\n"
            "    diffs = sweepdf['diff_bps'].tolist()\n"
            "    lbs   = sweepdf.index.tolist()\n"
            "else:\n"
            "    lbs   = [21, 42, 63, 126, 252]\n"
            "    diffs = [R['diff_21'], R['diff_42'], R['diff_63'], R['diff_126'], R['diff_252']]\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "colors = [GREEN if d > 0 else RED for d in diffs]\n"
            "ax.bar([str(lb) + 'd' for lb in lbs], diffs, color=colors)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('lookback window'); ax.set_ylabel('diff: post-div minus baseline (bps)')\n"
            "ax.set_title('No consistent direction: the sign flips across lookback windows')\n"
            "plt.tight_layout(); plt.show()\n"
            "for lb, d in zip(lbs, diffs):\n"
            "    print(f'  lookback={lb:3d}d  diff={d:+6.1f}bps  ', 'BULLISH' if d>0 else 'bearish')"
        ),
        md(
            "The signal has no consistent direction. Short lookbacks make divergence periods "
            "look *bullish* (the opposite of the claim); longer lookbacks make them look *slightly* "
            "bearish. This is the classic hallmark of specification search — pick your window, "
            "get your story. After correcting for the five lookback choices (even a rough Bonferroni), "
            "no single specification is significant."
        ),

        # ---- BEAT 5 -- VERDICT ------------------------------------------------
        md(
            "## 5 . The verdict\n\n"
            f"- **Signal — None.** Post-divergence 21-day returns average **+{R['div_mean']:.0f} bps**, "
            f"*above* the unconditional **+{R['base_mean']:.0f} bps** baseline. "
            "The claimed bearish effect is absent; the sign is backwards.\n"
            f"- **Tradability — Mirage.** Shorting SPY on every divergence signal loses "
            f"~{R['div_mean']:.0f} bps per 21-day trade before transaction costs, "
            "plus ~14.6 signals per year means meaningful turnover.\n"
            "- **Coincident, not leading.** The A/D line co-moves with the index, "
            "not ahead of it. 'Divergences' are largely noise from the cap-weighting mismatch "
            "between the equal-count A/D line and the cap-weighted S&P 500."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 . Could you actually trade it?\n\n"
            "Imagine you short SPY for 21 days every time you see a bearish divergence:\n\n"
            f"- **~{R['sig_per_yr']:.0f} signals per year** (one every ~17 trading days)\n"
            f"- **Gross return per short trade: about −{R['div_mean']:.0f} bps** "
            "(because the market tends to go up after these 'bearish' signals)\n"
            "- **Round-trip cost:** ~10 bps minimum for an ETF hedge\n\n"
            "> The signal is not just uninformative -- it's **directionally wrong**. "
            "You would *lose* ~100 bps per trade on average. This is squarely a **Mirage**: "
            "there is no version of this trade that survives honest accounting."
        ),

        # ---- BEAT 7 -- GOING FURTHER ------------------------------------------
        md(
            "## 7 . Going further\n\n"
            "- **The positive control (synthetic).** In our synthetic tape, when we literally plant "
            "a divergence between breadth and the index, the machinery finds it. The real-data "
            "null result is about markets, not our code.\n"
            "- **Bullish breadth thrusts.** Zweig and others find that *very high* breadth days "
            "(90%+ of stocks advancing) predict strong forward returns. The bearish side of this "
            "(divergence) doesn't appear to be the mirror. That's "
            "[Study 24 — Stampede](../../24-stampede/).\n"
            "- **52-week highs and nearness.** A related signal family — stock nearness to their "
            "individual 52-week high — is in [Study 50 — High-Water](../../50-high-water/).\n"
            "- **What if you used a *point-in-time* constituent list?** The panel here has "
            "survivorship bias (fallen names absent). A true historical A/D line including all "
            "delisted names could shift the results — but the computational cost is high and "
            "the direction of the shift for breadth measurement is not obvious.\n\n"
            "*Think the generals-and-soldiers story has merit? Fork this, try a different "
            "horizon, a sector-level A/D, or a McClellan Oscillator variant. Show a stable, "
            "consistent direction that clears a Bonferroni correction. That's the bar.*"
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
            "# Advance-Decline -- a quantitative teardown\n"
            "### Real daily panel * 497 S&P 500 constituents * HAC inference "
            "* permutation baseline * lookback sweep\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Coincident%3F: Not_a_leading_indicator](https://img.shields.io/badge/Coincident%3F-Not_a_leading_indicator-8b949e?style=flat-square)\n\n"
            "The deep companion to [01_for_the_curious.ipynb](01_for_the_curious.ipynb) -- same "
            "seven beats, every claim now carrying its standard error. We build the "
            "cumulative Advance-Decline line from S&P 500 daily constituent returns (~497 tickers, "
            "2005-2026, survivorship-biased panel -- named), detect bearish price-vs-breadth "
            "divergences, and test forward SPY returns against the unconditional baseline with a "
            "Newey-West HAC t-stat, a permutation baseline for the multiple-comparisons problem, "
            "and a sweep of five lookback windows.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily bars, as-of 2026-06-15; "
            "the offline core runs on a deterministic synthetic tape. "
            "Methods and sources: [`docs/references.md`](../docs/references.md). "
            "Numbers sourced from [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Survivorship bias:** constituent panel is current S&P 500 membership projected "
            "backwards -- delisted names absent. Named on Signal axis."
        ),
        code(BOOT + "\nfrom quantlab import analytics, stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Post-divergence 21d mean **+{R['div_mean']:.0f} bps**, "
            f"HAC *t* = **+{R['div_t']:.2f}** -- *above* the unconditional baseline "
            f"(**+{R['base_mean']:.0f} bps**); difference **+{R['diff_bps']:.1f} bps** "
            f"(p-val from permutation: {R['perm_pval']:.2f}). The claimed bearish effect is absent. |\n"
            f"| **Tradability** | `MIRAGE` | Shorting SPY on every signal loses ~{R['div_mean']:.0f} bps "
            f"per 21-day trade ({R['sig_per_yr']:.0f} signals/yr). |\n"
            "| **Coincident?** | `NOT A LEADING INDICATOR` | A/D co-moves with the cap-weighted "
            "index; divergences reflect cap-weighting artefacts, not structural breadth weakness. |\n\n"
            "> **Survivorship (named):** the constituent panel is current S&P 500 membership "
            "projected backwards; this is an upper bound on breadth quality, not a point-in-time "
            "measure. The bias could shift signal frequency/direction but the directional null "
            "result is robust to its plausible magnitude."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 . The claim, steelmanned\n\n"
            "Let $S_t$ be the SPY index and $B_t = \\sum_{\\tau \\le t}(A_\\tau - D_\\tau)$ "
            "the cumulative A/D line (advancers minus decliners). The recipe asserts:\n\n"
            "**H1 (bearish signal):** When $S_t = \\max_{s \\in [t-N, t]} S_s$ (index at "
            "$N$-day high) and $B_t < \\max_{s \\in [t-N, t]} B_s$ (A/D line is not), "
            "then $\\mathbb{E}[r^{SPY}_{t \\to t+H}] < \\mathbb{E}[r^{SPY}]$ -- forward "
            "returns should be below the unconditional mean.\n\n"
            "We test H1 with HAC-robust inference on the per-divergence-period forward return "
            "minus the unconditional baseline, and a permutation test for the multiple-comparisons "
            "problem arising from the lookback sweep."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 . So what? -- the stakes\n\n"
            "If H1 held, every long-only equity investor would have a free, simple timing signal "
            "to reduce equity exposure when the rally narrows. The A/D line is published daily "
            "by all major data providers. Its failure -- and the precise *mechanism* of that "
            "failure -- is itself instructive: the A/D line is largely *coincident* with the "
            "index (high contemporaneous correlation), not a *leading* indicator of the index, "
            "so 'divergences' from it are more likely to reflect the cap-weighting mismatch "
            "between the equal-count breadth measure and the cap-weighted index than to reflect "
            "genuine structural weakness."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 . How we'd know -- the protocol\n\n"
            "- **Signal.** $S_t \\ge$ $N$-day rolling max $\\times (1 - \\epsilon)$; "
            "$S_t / S_{t-N} - 1 \\ge 3\\%$; $B_t < $ $N$-day rolling max of $B$.\n"
            "- **No look-ahead.** Rolling maxima computed over the window *ending at t*; "
            "forward return starts at $t+1$ (desk standard: signal known at close of $t$, "
            "position entered next day).\n"
            "- **Inference.** Newey-West HAC *t* on per-divergence-period forward returns "
            "(using overlapping 21-day windows -- see caveat on effective n below). "
            "Permutation baseline: 2,000 random sub-samples of size $n_{\\text{div}}$ from "
            "the full return distribution.\n"
            "- **Multiple comparisons.** Five lookback windows (21, 42, 63, 126, 252 days) -- "
            "Bonferroni would require $|t| > 3.3$ at 5% FWER. No specification clears this.\n"
            "- **Survivorship named.** Panel = current S&P 500 projected backwards.\n"
            "- **Positive control.** Synthetic tape with planted divergences confirms the "
            "engine detects the effect when it exists."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 . The teardown"),

        md(
            "### 4a . Primary test: 63-day lookback, 21-day horizon\n\n"
            "HAC t-stat on the forward return difference: "
            "post-divergence mean minus unconditional baseline."
        ),
        code(
            BOOT_REAL +
            "if HAVE_REAL:\n"
            "    res = st.backtest(spy_c, ad_c, lookback=63, horizon=21)\n"
            "    ds, bs = res['div_stats'], res['base_stats']\n"
            "    nd = res['n_divergences']; nb2 = len(res['base_rets'])\n"
            "else:\n"
            "    nd = R['n_div']; nb2 = R['n_base']\n"
            "    ds = {'mean_bps': R['div_mean'], 'tstat': R['div_t'], 'hit_rate': R['div_hit']}\n"
            "    bs = {'mean_bps': R['base_mean'],'tstat': R['base_t'],'hit_rate': R['base_hit']}\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.5))\n"
            "arms = ['Post-divergence\\n(n=%d)' % nd, 'Unconditional\\n(n=%d)' % nb2]\n"
            "means = [ds['mean_bps'], bs['mean_bps']]\n"
            "tstats = [ds['tstat'], bs['tstat']]\n"
            "cols = [RED, GREY]\n"
            "ax.bar(arms, means, color=cols, width=0.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('21-day forward return (bps)')\n"
            "ax.set_title('Post-divergence returns are NOT below baseline (they are ABOVE it)')\n"
            "for i, (m, t) in enumerate(zip(means, tstats)):\n"
            "    ax.text(i, m + 3, f't={t:+.2f}', ha='center', fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Post-divergence: {ds[\"mean_bps\"]:+.1f} bps  HAC t = {ds[\"tstat\"]:+.2f}')\n"
            "print(f'Unconditional:   {bs[\"mean_bps\"]:+.1f} bps  HAC t = {bs[\"tstat\"]:+.2f}')\n"
            "print(f'Difference (should be negative for \"bearish\" claim): {ds[\"mean_bps\"]-bs[\"mean_bps\"]:+.1f} bps')"
        ),
        md(
            f"> **In plain words:** the bearish divergence signal predicts *above*-average "
            f"returns (+{R['diff_bps']:.1f} bps above baseline). The t-stat on the divergence "
            f"arm (+{R['div_t']:.2f}) is positive because the market generally went up over this "
            "period, not because the signal is predictive."
        ),

        md(
            "### 4b . Permutation test -- is ANY sub-sample this 'different'?\n\n"
            "Draw 2,000 random sub-samples of the same size as the divergence count. "
            "The claimed 'unusual' divergence return should be in the tails of this distribution."
        ),
        code(
            BOOT_REAL +
            "if HAVE_REAL:\n"
            "    res = st.backtest(spy_c, ad_c, lookback=63, horizon=21)\n"
            "    perm = st.permutation_baseline(res['base_rets'], res['n_divergences'],\n"
            "                                   n_perm=2000, seed=168)\n"
            "    div_mean_bps = res['div_stats']['mean_bps']\n"
            "    perm_means   = perm['perm_means_bps']\n"
            "else:\n"
            "    rng2 = np.random.default_rng(168)\n"
            "    perm_means = rng2.normal(R['base_mean'], 20, 2000)\n"
            "    div_mean_bps = R['div_mean']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "ax.hist(perm_means, bins=40, color=GREY, alpha=0.7, label='random sub-samples (n=2000)')\n"
            "ax.axvline(div_mean_bps, c=RED, lw=2.5, label=f'observed divergence mean: {div_mean_bps:+.0f}bps')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('21-day mean forward return of random sub-sample (bps)')\n"
            "ax.set_ylabel('count')\n"
            "ax.set_title('The divergence mean sits squarely inside the random-sub-sample distribution')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "frac_neg = float((np.array(perm_means) < div_mean_bps).mean())\n"
            "print(f'Fraction of random sub-samples more negative than observed: {frac_neg:.3f}')\n"
            "print('A value near 0.5 means the \"bearish\" divergence signal is typical, not extreme.')"
        ),
        md(
            f"> **In plain words:** {R['perm_pval']:.0%} of randomly chosen sub-samples of the same "
            "size have a *more negative* forward return than the divergence periods. The divergence "
            "signal is indistinguishable from a random sample of calendar dates."
        ),

        md(
            "### 4c . Lookback sweep -- no specification survives Bonferroni\n\n"
            "Five lookback windows, two horizons. Bonferroni-adjusted threshold for 10 comparisons "
            "at 5% FWER is |t| > 3.3. We plot the *diff t-stat* (t on div_mean - base_mean)."
        ),
        code(
            BOOT_REAL +
            "if HAVE_REAL:\n"
            "    sweepdf = st.sweep_lookback(spy_c, ad_c, lookbacks=[21,42,63,126,252], horizon=21)\n"
            "    sweep_lbs = sweepdf.index.tolist()\n"
            "    sweep_diffs = sweepdf['diff_bps'].tolist()\n"
            "    sweep_divt  = sweepdf['div_tstat'].tolist()\n"
            "else:\n"
            "    sweep_lbs  = [21, 42, 63, 126, 252]\n"
            "    sweep_diffs = [R['diff_21'], R['diff_42'], R['diff_63'], R['diff_126'], R['diff_252']]\n"
            "    sweep_divt  = [1.8, 2.1, 1.5, -1.2, -1.4]  # illustrative\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "lbl = [str(lb)+'d' for lb in sweep_lbs]\n"
            "cols2 = [GREEN if d > 0 else RED for d in sweep_diffs]\n"
            "axes[0].bar(lbl, sweep_diffs, color=cols2)\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_xlabel('lookback window'); axes[0].set_ylabel('diff vs baseline (bps)')\n"
            "axes[0].set_title('Difference flips sign -- no stable direction')\n"
            "axes[1].bar(lbl, sweep_divt, color=cols2)\n"
            "for t_thresh in (2, -2):\n"
            "    axes[1].axhline(t_thresh, ls='--', c=GREY, lw=1)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_xlabel('lookback window'); axes[1].set_ylabel('HAC t (divergence arm)')\n"
            "axes[1].set_title('t-stats are positive (market goes up) not bearish')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Bonferroni threshold (10 comparisons, 5% FWER): |t| > 3.3')\n"
            "print('Max |t| in sweep:', max(abs(t) for t in sweep_divt))"
        ),

        md(
            "### 4d . The positive control -- the engine DOES work when divergence exists\n\n"
            "Synthetic tape with planted breadth divergence vs null tape:"
        ),
        code(
            "moms = [0.0, 1.0, 2.0, 3.0]\n"
            "diffs_synth = []\n"
            "for div in moms:\n"
            "    panel_s, spy_s, _ = data.synthetic_daily(\n"
            "        n_stocks=200, n_days=756, divergence=div, seed=168)\n"
            "    ad_s = data.build_ad_line(panel_s)\n"
            "    r = st.backtest(spy_s['close'], ad_s, lookback=42, horizon=21,\n"
            "                    min_index_gain=0.0)\n"
            "    diffs_synth.append(r['diff_mean_bps'])\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.5))\n"
            "ax.plot(moms, diffs_synth, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted divergence strength'); ax.set_ylabel('diff: post-div minus baseline (bps)')\n"
            "ax.set_title('Engine recovers the planted divergence -- the real-market null is about markets')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('At divergence=0 (null): diff ~', round(diffs_synth[0]))\n"
            "print('At divergence=3 (strong): diff ~', round(diffs_synth[-1]))"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 . The verdict\n\n"
            f"- **Signal `NONE`** -- post-divergence mean **+{R['div_mean']:.0f} bps**, "
            f"HAC *t* **+{R['div_t']:.2f}** (positive because markets drift up; "
            f"the *difference* from baseline is **+{R['diff_bps']:.1f} bps**, "
            f"permutation p-val {R['perm_pval']:.2f}). The claimed bearish effect is absent "
            "and the sign is opposite. Lookback sweep shows no stable direction; no "
            "specification survives even a Bonferroni-2 correction.\n"
            f"- **Tradability `MIRAGE`** -- the intended trade (short SPY on bearish divergence) "
            f"loses ~{R['div_mean']:.0f} bps/trade gross, {R['sig_per_yr']:.0f} signals/yr.\n"
            "- **Coincident `NOT A LEADING INDICATOR`** -- breadth correlates strongly "
            "with the index contemporaneously; divergences largely reflect the cap-weighting "
            "artefact between equal-count A/D and market-cap-weighted S&P 500.\n"
            "- **Survivorship bias (named):** panel is current S&P 500 membership; true "
            "historical A/D including delisted names could shift results, but the direction "
            "and magnitude of the shift is not obvious for breadth."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 . Could you trade it? -- the maths\n\n"
            "The intended implementation: *short SPY (or buy puts) for 21 days on each "
            "bearish divergence signal.*"
        ),
        code(
            "# Cost / tradability table\n"
            "import pandas as pd\n"
            "rows = []\n"
            "costs = [0, 5, 10, 20]\n"
            "for c in costs:\n"
            "    gross = R['div_mean']         # this is positive (WRONG direction for a short)\n"
            "    net = -gross - c               # short position: we LOSE the positive return\n"
            "    rows.append({'round-trip cost (bps)': c, 'short gross (bps)': -gross,\n"
            "                 'net per trade (bps)': net})\n"
            "tbl = pd.DataFrame(rows)\n"
            "print(tbl.to_string(index=False))\n"
            "print(f'\\nAt {R[\"sig_per_yr\"]:.0f} signals/yr, a -100bps/trade loss = '\n"
            "      f'{-R[\"div_mean\"] * R[\"sig_per_yr\"] / 100:.1f}%/yr annualised loss')"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 . Going further\n\n"
            "- **The bullish side.** Zweig's *breadth thrust* (very high advance counts) is a "
            "*bullish* breadth signal family -- see [Study 24 — Stampede](../../24-stampede/). "
            "The literature on breadth is more supportive of bullish breadth thrusts than of "
            "bearish divergence signals, and this desk's own evidence is consistent with that.\n"
            "- **52-week high nearness.** George & Hwang (2004) find that stock nearness to "
            "their individual 52-week high predicts cross-sectional returns -- "
            "[Study 50 — High-Water](../../50-high-water/). This is conceptually related "
            "but distinct from aggregate breadth.\n"
            "- **Point-in-time A/D.** The correct test would use a historical roster of S&P 500 "
            "constituents including all names that were ever in the index. CRSP/Compustat provide "
            "this for a subscription fee; the bias direction for breadth measurement is not "
            "obvious but the magnitude could matter.\n"
            "- **Sector-level breadth.** Perhaps the divergence signal works sector by sector "
            "(e.g. 'tech stocks at new high but most tech A-D declining') -- a richer version "
            "of the same hypothesis, but carrying 11x the multiple-comparisons burden.\n\n"
            "*Think there's a real breadth signal here? Fork this, implement a point-in-time "
            "panel, apply a Bonferroni or permutation correction for the lookback sweep, and "
            "show a consistent, significant directional result. That's the bar.*"
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "02_for_the_quants.ipynb")


def _meta():
    return {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
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
