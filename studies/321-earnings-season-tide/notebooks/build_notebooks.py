"""Generate the two narrative notebooks for Study 321 (Earnings-Season-Tide).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks walk the seven desk beats (see ../../../METHODOLOGY.md). Every code cell runs
OFFLINE and deterministic: the real-tape cells try the cached SPY total-return parquet under
../_cache/ (or the shared repo cache) and, on any failure, fall back to the synthetic tape
with a clear banner. The frozen real-tape headline numbers live in ONE dict ``R`` below,
mirroring docs/results.md, so a reader who can't reach the cache still sees the verdict.

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


# ---------------------------------------------------------------------------
# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-18).
# ---------------------------------------------------------------------------
R = dict(
    as_of="2026-06-18",
    fingerprint="ff2579c16a7d",
    span="1993-01-29 → 2025-12-31",
    n_days=8288,
    n_in=1737, n_out=6550,
    mean_in_bps=6.86, mean_out_bps=3.28, diff_bps=3.59,
    ann_in_pct=17.3, ann_out_pct=8.3,
    tstat_diff=1.27,
    ci_lo=-1.67, ci_hi=9.17,
    # per window (bps/day, HAC t)
    jan_bps=0.37, jan_t=-0.72,
    apr_bps=9.15, apr_t=1.23,
    jul_bps=5.44, jul_t=0.33,
    oct_bps=11.16, oct_t=1.39,
    # the race
    overlay_ann=3.54, overlay_sr=0.40, overlay_tim=0.21, overlay_turnover=8.0,
    buyhold_ann=10.15, buyhold_sr=0.54,
    # synthetic control
    ctrl_null_diff=-0.71, ctrl_null_t=-0.27,
    ctrl_tide_diff=11.29, ctrl_tide_t=4.24, ctrl_tide_bps=12,
)

R_INJECT = f"""
# Frozen real-tape headline numbers (mirror of ../docs/results.md, as-of {R['as_of']}).
R = {R!r}
"""

# ---------------------------------------------------------------------------
# Shared analysis preamble — loads the tape (real cache-first, synthetic fallback).
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
pd.set_option("display.float_format", lambda v: f"{v:,.3f}")

from earnings_season_tide import data, strategy as st
"""

LOAD_REAL = """\
# Cache-first real tape, synthetic fallback. We BANNER which tape every cell below runs on.
TAPE = "SYNTHETIC"
try:
    bars = data.load_real("SPY", "total_return", fetch=False)
    bars = bars[bars.index <= "2025-12-31"]          # drop the in-progress year
    TAPE = "REAL"
    print(f"REAL tape: SPY total-return, {bars.index.min().date()} -> {bars.index.max().date()}, "
          f"{len(bars)} days, fingerprint {data.fingerprint(bars)}")
except Exception as e:
    bars, _ = data.synthetic_daily(n_years=33, tide_bps=0.0, seed=321)
    print(f"OFFLINE -> SYNTHETIC null tape ({len(bars)} days). Reason: {e}")
    print("Headline real-tape numbers are quoted from the frozen R dict (docs/results.md).")
print("TAPE =", TAPE)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Earnings-Season-Tide 🌊\n"
            "### Does the whole market drift differently during peak earnings weeks?\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Separable tide%3F: Not supported](https://img.shields.io/badge/Separable_tide%3F-Not_supported-8b949e?style=flat-square)\n\n"
            "Four times a year, for a couple of weeks, almost every big company reports its "
            "earnings at once — mid-late January, April, July and October. It *feels* like "
            "the market should move to a different rhythm in those windows. This notebook "
            "asks, in plain language, whether the whole index really has an 'earnings-season "
            "tide' you could lean on — and answers: it drifts a hair faster, but the gap is "
            "lost in the noise, and trying to ride it under-performs just holding the index.\n\n"
            "> 📓 **This is the plain-language layer.** Want the HAC *t*-stats, the bootstrap "
            "CI and the capacity maths? That's the companion notebook, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** An educational, reproducible research tool: "
            "every chart below is generated by the code beside it. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + R_INJECT + "\n" + LOAD_REAL),

        # BEAT 0
        md(
            "## The answer first 🎯\n\n"
            "| Question | Plain answer |\n|---|---|\n"
            "| Do peak-earnings days drift faster? | A little — about **+6.9 bps/day** in-window "
            "vs **+3.3 bps/day** out, on SPY 1993–2025. |\n"
            "| Is that gap *real* (not luck)? | **No.** The robust *t* on the difference is only "
            "**+1.27**; the 95% interval **[−1.7, +9.2] bps/day** straddles zero. |\n"
            "| Could you trade it? | **No.** 'Long only in the windows, cash otherwise' earns "
            "**~3.5%/yr** vs the index's **~10.2%/yr**, while idle in cash 79% of the year. |\n\n"
            "The verdict badges up top say it: **None** on the signal, **Mirage** on tradability."
        ),
        code(
            "# The two group means on whichever tape we're holding.\n"
            "c = st.calendar_contrast(bars)\n"
            "print(f'In-window mean : {c[\"mean_in_bps\"]:+.2f} bps/day  ({c[\"n_in\"]} days)')\n"
            "print(f'Out-of-window  : {c[\"mean_out_bps\"]:+.2f} bps/day  ({c[\"n_out\"]} days)')\n"
            "print(f'Difference     : {c[\"diff_bps\"]:+.2f} bps/day   HAC t = {c[\"tstat_diff\"]:+.2f}')\n"
            "if TAPE != 'REAL':\n"
            "    print('\\n[banner] SYNTHETIC null tape above. Frozen REAL numbers:')\n"
            "    print(f'  REAL diff = {R[\"diff_bps\"]:+.2f} bps/day, HAC t = {R[\"tstat_diff\"]:+.2f}')\n"
        ),

        # BEAT 1
        md(
            "## 1 · The claim\n\n"
            "Stated at full strength, the way a market commentator would: *earnings season "
            "moves the whole market.* When hundreds of S&P 500 companies report at once, the "
            "flood of fresh news supposedly pushes the index into a distinct seasonal drift — "
            "so you could be long (or step aside) only during those four windows and beat a "
            "buy-and-hold investor who ignores the calendar.\n\n"
            "There *is* a real seed here: individual stocks really do behave specially around "
            "their own earnings (post-earnings-announcement drift). The leap is assuming that "
            "aggregates into a directional tide for the *index* — which is what we test."
        ),

        # BEAT 2
        md(
            "## 2 · So what?\n\n"
            "If true, it would be a free lunch hiding in plain sight on a calendar anyone can "
            "read. You wouldn't need to predict anything — just hold the index for ~8 weeks a "
            "year and sit in cash the rest, collecting a seasonal premium with almost no risk "
            "the other 44 weeks. That's the dream. It would also say something deep about "
            "markets: that a *predictable*, *public*, *recurring* event window carries an "
            "un-arbitraged drift. Extraordinary claims need a robust test, not a suggestive gap."
        ),

        # BEAT 3
        md(
            "## 3 · How would we even know?\n\n"
            "We hard-code the four peak-earnings windows (mid-late Jan/Apr/Jul/Oct) **in "
            "advance** — no peeking at returns to choose them — and tag every trading day as "
            "in-window or out. Because the calendar is known ahead of time, there's **no "
            "guessing and no lag**: on 1 January you already know every in-window day for the "
            "year.\n\n"
            "Then we compare the average daily move in-window vs out, and — crucially — put a "
            "proper error bar on the *difference*. **What would make us call it a mirage?** A "
            "difference whose error bar comfortably includes zero. Let's look."
        ),

        # BEAT 4
        md("## 4 · The teardown — let's actually look\n\n"
           "First, the headline gap and its error bar. The bar straddles zero — that's the "
           "whole story in one chart."),
        code(
            "lo, hi = st.block_bootstrap_ci(bars, n_boot=1500, seed=321)\n"
            "fig, ax = plt.subplots()\n"
            "ax.bar(['out-of-window','in-window'], [c['mean_out_bps'], c['mean_in_bps']],\n"
            "       color=[GREY, AMBER], width=.6)\n"
            "ax.errorbar(['in-window'], [c['mean_in_bps']],\n"
            "            yerr=[[c['mean_in_bps']-(c['mean_out_bps']+lo)],"
            "[(c['mean_out_bps']+hi)-c['mean_in_bps']]],\n"
            "            fmt='none', ecolor=RED, capsize=6, lw=2,\n"
            "            label=f'95% CI on the gap: [{lo:+.1f}, {hi:+.1f}] bps')\n"
            "ax.axhline(c['mean_out_bps'], color=GREY, ls='--', lw=1)\n"
            "ax.set_ylabel('mean daily return (bps/day)')\n"
            "ax.set_title(f'Earnings-season tide on {(\"SPY\" if TAPE==\"REAL\" else \"synthetic\")}'\n"
            "             f' — gap {c[\"diff_bps\"]:+.1f} bps/day, t={c[\"tstat_diff\"]:+.2f}')\n"
            "ax.legend(); plt.show()\n"
        ),
        md("And per-quarter — does *any* single earnings window carry a tide? (Cherry-picking "
           "the best of four is itself a trap; watch the *t*-stats.)"),
        code(
            "pw = st.per_window_means(bars)\n"
            "display(pw)\n"
            "fig, ax = plt.subplots()\n"
            "cols = [GREEN if abs(t) >= 2 else GREY for t in pw['tstat']]\n"
            "ax.bar(pw['window'], pw['mean_bps'], color=cols, width=.6)\n"
            "ax.axhline(0, color='k', lw=.8)\n"
            "ax.set_ylabel('mean daily return in window (bps/day)')\n"
            "ax.set_title('Per-window drift — grey = does NOT clear |t|>=2')\n"
            "plt.show()\n"
            "if TAPE != 'REAL':\n"
            "    print('[banner] SYNTHETIC tape above. On the REAL SPY tape, the strongest '\n"
            "          f'window is Oct at {R[\"oct_bps\"]:+.1f} bps/day but only t={R[\"oct_t\"]:+.2f}.')\n"
        ),

        # BEAT 5
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** The in-window days drift a touch faster, but the gap "
            f"(**{R['diff_bps']:+.1f} bps/day** on real SPY) has a robust *t* of only "
            f"**{R['tstat_diff']:+.2f}** and an error bar **[{R['ci_lo']:+.1f}, {R['ci_hi']:+.1f}]** "
            "that includes zero. No single quarter clears the bar either.\n"
            "- **Separable tide? — Not supported.** The real single-stock earnings effect "
            "doesn't aggregate into a directional index seasonal; the in-window days are just "
            "*some* of the ordinary up-drift days, not a separate tide."
        ),

        # BEAT 6
        md("## 6 · Could you actually trade it?\n\n"
           "Suppose you ignored the weak statistics and traded it anyway: long the index only "
           "inside the windows, cash otherwise. Here's that overlay against simply holding."),
        code(
            "race = st.race_vs_buyhold(bars, cost_bps=1.0)\n"
            "print(f'In-window overlay : {race[\"overlay_ann_pct\"]:+.2f}%/yr   '\n"
            "      f'Sharpe {race[\"overlay_sharpe_net\"]:.2f}   in market {race[\"time_in_market\"]:.0%}')\n"
            "print(f'Buy-and-hold      : {race[\"buyhold_ann_pct\"]:+.2f}%/yr   '\n"
            "      f'Sharpe {race[\"buyhold_sharpe\"]:.2f}   in market 100%')\n"
            "if TAPE != 'REAL':\n"
            "    print('\\n[banner] SYNTHETIC tape above. Frozen REAL race:')\n"
            "    print(f'  overlay {R[\"overlay_ann\"]:+.2f}%/yr SR {R[\"overlay_sr\"]:.2f}  vs  '\n"
            "          f'buy-hold {R[\"buyhold_ann\"]:+.2f}%/yr SR {R[\"buyhold_sr\"]:.2f}')\n"
        ),
        md("On the real tape the overlay earns **~3.5%/yr at Sharpe 0.40** — *worse* than the "
           "index's **~10.2%/yr at Sharpe 0.54** — while parking in cash 79% of the year. Even "
           "if the tiny gap were real, slicing it out under-performs the thing it's carved from. "
           "**Mirage.**"),

        # BEAT 7
        md("## 7 · Going further 🚪\n\n"
           "- **Conditioning on surprise.** A directional tide might only appear in quarters "
           "where aggregate surprises skew positive — but that needs forward-looking earnings "
           "data and stops being a pure calendar rule.\n"
           "- **Volatility, not direction.** Maybe earnings season moves *variance* (a vol tide) "
           "even if it doesn't move drift — a cleaner question to fork.\n"
           "- **Window definitions.** We pre-declared the windows; a believer could argue for "
           "different week ranges. The honest way to settle that is a data-snooping correction "
           "across all plausible windows (Sullivan-Timmermann-White), not hunting for the best one."),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "01_for_the_curious.ipynb")


# ===========================================================================
# 02 — FOR THE QUANTS
# ===========================================================================
def build_quants():
    cells = [
        md(
            "# Earnings-Season-Tide — a quantitative teardown 🔬\n"
            "### Calendar contrast · HAC inference · block-bootstrap CI · selection · excess-vs-excess capacity\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Separable tide%3F: Not supported](https://img.shields.io/badge/Separable_tide%3F-Not_supported-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) "
            "— *same seven beats, every claim now carrying its standard error.* We measure the "
            "aggregate-index drift inside four pre-declared peak-earnings windows, put a "
            "Newey-West *t* on the in-window minus out-of-window difference, bound it with a "
            "circular block bootstrap, check the per-window selection trap, and race a "
            "calendar overlay excess-of-cash against buy-and-hold.\n\n"
            "> ⚠️ **Not investment advice.** Data: SPY total-return daily bars (cache-first; "
            "synthetic fallback offline). Methods: HAC/Newey-West, block bootstrap, "
            "excess-vs-excess Sharpe. Literature in [`docs/references.md`](../docs/references.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition. "
            "House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + R_INJECT + "\n" + LOAD_REAL),

        # BEAT 0
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Why |\n|---|---|---|\n"
            "| Signal | **None** | in−out diff **+3.59 bps/day**, HAC *t* **+1.27**; bootstrap "
            "CI **[−1.67, +9.17]** includes 0; no window clears *t*=2 |\n"
            "| Tradability | **Mirage** | overlay **+3.5%/yr, SR 0.40** vs buy-hold **+10.2%/yr, "
            "SR 0.54**; 79% in cash |\n"
            "| Separable tide? | **Not supported** | PEAD is single-stock & surprise-conditioned; "
            "it does not aggregate into a directional index seasonal |\n\n"
            "> 💡 **In plain words:** the index drifts marginally faster during earnings weeks, "
            "but it's the ordinary up-drift showing through a calendar slice — not a separable, "
            "tradable tide."
        ),

        # BEAT 1
        md(
            "## 1 · The claim, steelmanned\n\n"
            "- **H₁ (the tide).** Mean daily index return is higher in-window than out: "
            "`E[r | in] − E[r | out] > 0`, robustly (HAC *t* ≥ 2).\n"
            "- **H₂ (per-window).** At least one of the four windows individually carries a "
            "drift clearing *t* = 2 (and survives the four-way selection).\n"
            "- **H₃ (tradable).** A long-in-window/cash overlay beats buy-and-hold on "
            "*excess-of-cash* Sharpe after one-way costs.\n\n"
            "We reject H₁–H₃ if the differences sit inside their error bars and the overlay "
            "trails the index."
        ),

        # BEAT 2
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "A +3.6 bps/day in-window premium over ~1,740 in-window days is ~+6.2% of cumulative "
            "log-drift parked in 21% of the calendar — *if* real, an attractive risk-budget "
            "deployment. The stakes are entirely in the standard error: a gap of this size is "
            "exactly what ordinary equity drift produces over any 21% slice of days, so the "
            "question is whether the *difference* is separable from baseline drift, not whether "
            "in-window days are positive (they are — so are most days)."
        ),

        # BEAT 3
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "1. **Measure** the in/out contrast on log returns (no fitting).\n"
            "2. **Robust inference** — regress `r` on the in-window dummy with a Newey-West HAC "
            "covariance; the slope is the mean difference, its HAC SE gives the *t*. Bound the "
            "difference with a **circular block bootstrap** (block = 21 ≈ one month, preserving "
            "vol clustering and calendar alignment).\n"
            "3. **Critique magnitude** — split by window to expose the selection trap (best of "
            "four pre-chosen windows).\n"
            "4. **Alpha vs beta** — the in-window return is *mostly market beta you already "
            "hold*; the overlay's contribution is the only thing that isn't.\n"
            "5. **Capacity** — excess-vs-excess Sharpe race, costs one-way × NAV, time-in-market.\n\n"
            "> 💡 **In plain words:** a t-test that respects autocorrelation, a resampling error "
            "bar that respects volatility clusters, and a fair race that doesn't reward sitting "
            "in cash."
        ),

        # BEAT 4
        md("## 4 · The teardown\n\n### 4.1 · The contrast and its HAC *t*"),
        code(
            "c = st.calendar_contrast(bars)\n"
            "lo, hi = st.block_bootstrap_ci(bars, block=21, n_boot=2000, seed=321)\n"
            "tab = pd.DataFrame({\n"
            "    'group': ['in-window','out-of-window','difference (in-out)'],\n"
            "    'mean_bps': [c['mean_in_bps'], c['mean_out_bps'], c['diff_bps']],\n"
            "    'n': [c['n_in'], c['n_out'], np.nan],\n"
            "})\n"
            "display(tab)\n"
            "print(f'HAC t on difference : {c[\"tstat_diff\"]:+.3f}')\n"
            "print(f'block-bootstrap 95% CI on difference : [{lo:+.2f}, {hi:+.2f}] bps/day')\n"
            "print(f'TAPE = {TAPE}')\n"
            "if TAPE != 'REAL':\n"
            "    print(f'[banner] SYNTHETIC null. REAL: diff {R[\"diff_bps\"]:+.2f}, t {R[\"tstat_diff\"]:+.2f}, '\n"
            "          f'CI [{R[\"ci_lo\"]:+.2f}, {R[\"ci_hi\"]:+.2f}]')\n"
        ),
        md("> 💡 **In plain words:** the gap is positive but its *t* is well under 2 and the "
           "interval crosses zero — by the desk's inference bar, that is **not** a real signal."),
        md("### 4.2 · The per-window selection trap"),
        code(
            "pw = st.per_window_means(bars)\n"
            "display(pw)\n"
            "best = pw.loc[pw['mean_bps'].idxmax()]\n"
            "print(f'Strongest window: {best[\"window\"]} at {best[\"mean_bps\"]:+.2f} bps/day, '\n"
            "      f't={best[\"tstat\"]:+.2f}')\n"
            "print('None of the four clears |t|>=2 -> picking the best of four would be pure snooping.')\n"
        ),
        md("### 4.3 · Alpha vs beta — what's actually new"),
        code(
            "# The in-window days are mostly the market you already own. The overlay's only\n"
            "# distinct contribution is the in/out timing; everything else is beta.\n"
            "r = st.log_returns(bars)\n"
            "ann_all = r.mean()*252*100\n"
            "print(f'Whole-sample drift: {ann_all:+.2f}%/yr (this is the beta you hold either way)')\n"
            "print(f'In-window annualised: {c[\"ann_in_pct\"]:+.2f}%/yr  -- higher, but it IS beta,'\n"
            "      ' just sampled on a calendar.')\n"
        ),
        md("### 4.4 · Capacity — the excess-vs-excess race"),
        code(
            "rows = []\n"
            "for cost in (0.0, 1.0, 2.0, 5.0):\n"
            "    rc = st.race_vs_buyhold(bars, cost_bps=cost)\n"
            "    rows.append({'cost_bps': cost, 'overlay_ann%': rc['overlay_ann_pct'],\n"
            "                 'overlay_SR': rc['overlay_sharpe_net'],\n"
            "                 'buyhold_ann%': rc['buyhold_ann_pct'],\n"
            "                 'buyhold_SR': rc['buyhold_sharpe'],\n"
            "                 'time_in_mkt': rc['time_in_market'],\n"
            "                 'turnover/yr': rc['turnover_per_yr']})\n"
            "sweep = pd.DataFrame(rows)\n"
            "display(sweep)\n"
            "print('Turnover is one-way x NAV. The overlay trails buy-and-hold at every cost level.')\n"
        ),
        md("> 💡 **In plain words:** even at zero cost the overlay loses the race — it isn't a "
           "cost problem, it's that you're voluntarily sitting out 79% of the up-drift to chase "
           "a gap that isn't there."),

        # BEAT 5
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** in−out **{R['diff_bps']:+.2f} bps/day**, HAC *t* "
            f"**{R['tstat_diff']:+.2f}**, bootstrap CI **[{R['ci_lo']:+.2f}, {R['ci_hi']:+.2f}]** "
            "includes 0; H₁ and H₂ rejected.\n"
            f"- **Tradability — Mirage.** overlay **{R['overlay_ann']:+.2f}%/yr, SR "
            f"{R['overlay_sr']:.2f}** vs buy-hold **{R['buyhold_ann']:+.2f}%/yr, SR "
            f"{R['buyhold_sr']:.2f}**; H₃ rejected.\n"
            "- **Separable tide? — Not supported.** PEAD aggregates away at the index level."
        ),

        # BEAT 6
        md("## 6 · Could you trade it?\n\n"
           "Break-even is moot: the edge is statistically zero and the overlay loses the race "
           "at **zero** cost, so there is no cost budget to spend. The binding facts are "
           "**time-in-market 21%** and an excess-of-cash Sharpe **below** buy-and-hold. No "
           "square-root-impact capacity analysis is warranted for a non-edge."),

        # BEAT 7
        md("## 7 · Going further\n\n"
           "- **Surprise-conditioned aggregation.** Re-run conditioning on the sign of "
           "aggregate earnings surprise (needs forward earnings data; drops the pure-calendar "
           "property).\n"
           "- **Second moment.** Test an in-window *variance* tide — earnings season may move "
           "vol even where it doesn't move drift.\n"
           "- **Snooping correction.** Sweep all plausible window definitions under a "
           "Sullivan-Timmermann-White Reality Check instead of fixing four — the honest way to "
           "handle the believer's 'but my windows are different' objection.\n\n"
           "*Synthetic positive control (the harness is a faithful detector):*"),
        code(
            "# The same machinery recovers a planted tide and reads the null as null.\n"
            "b0,_ = data.synthetic_daily(n_years=33, tide_bps=0.0, seed=321)\n"
            "b1,_ = data.synthetic_daily(n_years=33, tide_bps=12.0, seed=321)\n"
            "c0, c1 = st.calendar_contrast(b0), st.calendar_contrast(b1)\n"
            "ctrl = pd.DataFrame({\n"
            "    'planted tide (bps/day)': [0, 12],\n"
            "    'detected diff (bps/day)': [c0['diff_bps'], c1['diff_bps']],\n"
            "    'HAC t': [c0['tstat_diff'], c1['tstat_diff']],\n"
            "})\n"
            "display(ctrl)\n"
            "print('Clears the bar when a tide is real; insignificant when none is planted ->'\n"
            "      ' the real-tape t=1.27 is a property of the market, not a blind pipeline.')\n"
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
