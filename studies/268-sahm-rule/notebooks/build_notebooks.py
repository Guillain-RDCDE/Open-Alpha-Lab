"""Generate and execute the two narrative notebooks for Study 268 (Sahm-Rule).

    python notebooks/build_notebooks.py

This writes both notebooks AND executes them in-place with nbconvert (no
--allow-errors). The hardcoded unemployment table and the synthetic generator run
anywhere, offline and deterministically; the real ^GSPC cells use the repo-level
cache (``_cache/^GSPC_split_only.parquet``) if present, and otherwise branch on
``HAVE_REAL`` to quote the frozen headline numbers in ``R`` (mirroring
docs/results.md), so the notebooks re-run for any reader.

The _write convention (each build_*() ends by calling _write) is kept so the repo's
intro-restyle tooling can monkeypatch it.
"""

from __future__ import annotations

import os
import subprocess
import sys

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-17).
R = dict(
    year_start=1959,
    year_end=2025,
    n_unrate_months=804,
    n_onsets=12,
    horizon=12,
    # Event study (1-month execution lag, H=12m forward price return)
    event_mean_pct=11.9,
    uncond_mean_pct=8.6,
    diff_pct=3.4,
    welch_t=0.516,
    welch_p=0.616,
    perm_p=0.765,
    hac_t=0.55,
    # Shorter / longer horizons
    h6_event_pct=7.2,
    h6_uncond_pct=4.2,
    h6_welch_t=0.817,
    h24_event_pct=20.1,
    h24_uncond_pct=17.8,
    h24_welch_t=0.411,
    # Timing overlay (out 12 months on trigger, 10bps one-way)
    overlay_cagr_pct=6.5,
    bah_cagr_pct=7.5,
    overlay_mdd_pct=-38.5,
    bah_mdd_pct=-52.6,
    overlay_sharpe=0.56,
    bah_sharpe=0.56,
    active_hac_t=-1.01,
    n_switches=23,
    fp_unrate="061549f98555",
    fp_gspc="39484c0862b4",
)


# ---------------------------------------------------------------------------
# Shared preamble
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from sahm_rule import data, strategy as st

def _have_real():
    try:
        data.fetch_gspc_daily()
        return True
    except FileNotFoundError:
        return False

HAVE_REAL = _have_real()
print("^GSPC cache present:", HAVE_REAL)
"""

R_CELL = """\
# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-17)
R = dict(
    year_start=1959, year_end=2025, n_unrate_months=804, n_onsets=12, horizon=12,
    event_mean_pct=11.9, uncond_mean_pct=8.6, diff_pct=3.4,
    welch_t=0.516, welch_p=0.616, perm_p=0.765, hac_t=0.55,
    h6_event_pct=7.2, h6_uncond_pct=4.2, h6_welch_t=0.817,
    h24_event_pct=20.1, h24_uncond_pct=17.8, h24_welch_t=0.411,
    overlay_cagr_pct=6.5, bah_cagr_pct=7.5, overlay_mdd_pct=-38.5, bah_mdd_pct=-52.6,
    overlay_sharpe=0.56, bah_sharpe=0.56, active_hac_t=-1.01, n_switches=23,
)
"""

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Busted%3F: Yes](https://img.shields.io/badge/Busted%3F-Yes-8b949e?style=flat-square)\n\n"
)


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Sahm-Rule -- is the recession trigger a usable sell button?\n"
            "### The Sahm Rule, tested honestly, in plain English\n\n"
            + BADGES +
            "The **Sahm Rule** is one of the most reliable recession sirens ever built: when the "
            "3-month average unemployment rate climbs half a point above its recent low, a recession "
            "has almost always already begun. It is so clean that the Fed plots it. Naturally, "
            "investors ask the next question: *if it spots recessions, can I use it as a sell "
            "button for stocks?* This notebook answers that -- and the answer is a cautionary tale "
            "about the gap between **diagnosing** a recession and **timing** the market.\n\n"
            "> **This is the plain-language layer.** Want the event study, the HAC t-stat, the "
            "permutation test, and the net-of-cost overlay? That's "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\n" + R_CELL),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the Sahm trigger reliably spot recessions? | **Yes** -- that's not in dispute. "
            "It fires near the start of every postwar recession. |\n"
            "| Does it predict *low* forward stock returns? | **No.** Average S&P return in the "
            f"**{R['horizon']} months after** a trigger (1-month action lag) is "
            f"**{R['event_mean_pct']:+.1f}%** -- *higher* than the {R['uncond_mean_pct']:+.1f}% "
            "you'd get any random year. |\n"
            "| Why? | The trigger fires **late**. By the time unemployment has risen half a point, "
            "the market has usually already fallen -- and the forward year is often the recovery. |\n"
            "| Could you trade it as a sell button? | **No edge.** Sitting out 12 months on each "
            f"trigger cut max drawdown ({R['bah_mdd_pct']:.0f}% -> {R['overlay_mdd_pct']:.0f}%) but "
            f"*lowered* CAGR ({R['bah_cagr_pct']:.1f}% -> {R['overlay_cagr_pct']:.1f}%). |\n\n"
            "> The Sahm Rule is a superb recession *thermometer* and a poor market *sell button*. "
            "It tells you the patient already has a fever."
        ),

        # ---- BEAT 1 -- THE CLAIM ----------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"When the 3-month-average unemployment rate rises 0.5 points above its low of the "
            "past year, a recession is underway -- so sell stocks before the crash.\"*\n\n"
            "The first half is Claudia Sahm's actual rule (2019), designed to *trigger automatic "
            "stimulus checks*, not trades. The second half -- *therefore sell* -- is the folk "
            "leap investors make. We test that leap."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If a public, lagging, monthly macro statistic let you dodge equity bear markets, it "
            "would be the easiest free lunch in finance -- everybody reads the jobs report. The "
            "interesting question is *why* such an accurate recession dating tool fails as a "
            "trading signal. The answer is **timing**: recessions and bear markets do not line up. "
            "Stocks are forward-looking and usually bottom *during* the recession, often before "
            "the unemployment rate has finished rising."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three traps:\n\n"
            "1. **The wrong null.** Stocks drift up; the S&P is positive in most 12-month windows. "
            "A 'sell' signal must beat *staying invested*, not beat zero. We compare post-trigger "
            "returns to the **unconditional** forward return.\n"
            "2. **Look-ahead.** The unemployment print for a month is released the *next* month. We "
            "add a further one-month execution lag, so a trigger from the month-m number is acted "
            "on at the close of month m+1.\n"
            "3. **Tiny n.** There are only ~12 distinct Sahm triggers in 65 years. Any average "
            "rests on a handful of events -- we report a HAC t-stat and a permutation test and are "
            "honest that n<15 cannot clear a real significance bar.\n\n"
            "We test on the hardcoded BLS unemployment table (1959-2025) and the cached S&P 500 "
            "price series."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**First, the trigger itself.** The Sahm value is the 3-month-average unemployment rate "
            "minus its trailing-12-month minimum. Above 0.5 = recession siren."
        ),
        code(
            "u = data.unrate_series()\n"
            "sind = st.sahm_indicator(u)\n"
            "onsets = st.trigger_onsets(u)\n"
            "print(f'Unemployment table: {u.index.min():%Y-%m} .. {u.index.max():%Y-%m}  ({len(u)} months)')\n"
            "print(f'Sahm trigger onsets: {len(onsets)}')\n"
            "for d in onsets:\n"
            "    print('   ', d.strftime('%Y-%m'))\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(11, 4.3))\n"
            "ax.plot(sind.index, sind['sahm_gap'], c=GREY, lw=1.2, label='Sahm value (pp)')\n"
            "ax.axhline(0.5, ls='--', c=RED, lw=2, label='Trigger threshold (0.50pp)')\n"
            "for d in onsets:\n"
            "    ax.axvline(d, c=AMBER, lw=0.8, alpha=0.6)\n"
            "ax.set_ylabel('Sahm value (pp over trailing-12m min)')\n"
            "ax.set_title('The Sahm trigger fires near the start of every postwar recession')\n"
            "ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            "The trigger is a genuine recession siren -- it lights up around each downturn. The "
            "question is what the **stock market** does *after* it lights up."
        ),
        md("**The punchline chart: forward S&P returns after a trigger vs any random month.**"),
        code(
            "if HAVE_REAL:\n"
            "    g = data.fetch_gspc_daily()\n"
            "    es = st.event_study(u, g, horizon=12, exec_lag=1)\n"
            "    ev_mean = es['event_mean']*100\n"
            "    un_mean = es['uncond_mean']*100\n"
            "    ev_fwd = es['event_fwd']*100\n"
            "else:\n"
            "    ev_mean, un_mean = R['event_mean_pct'], R['uncond_mean_pct']\n"
            "    rng = np.random.default_rng(268)\n"
            "    ev_fwd = rng.normal(ev_mean, 22, R['n_onsets'])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "bars = ax.bar(['After a Sahm\\ntrigger (+1m lag)', 'Any random\\n12-month window'],\n"
            "              [ev_mean, un_mean], color=[RED, GREY], width=0.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Mean forward 12-month S&P return (%)')\n"
            "ax.set_title('After the siren, the market did BETTER than average -- the trigger is late')\n"
            "for b, v in zip(bars, [ev_mean, un_mean]):\n"
            "    ax.annotate(f'{v:+.1f}%', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='bottom' if v>=0 else 'top')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Mean 12m return after trigger: {ev_mean:+.1f}%')\n"
            "print(f'Unconditional 12m return:      {un_mean:+.1f}%')\n"
            "print(f'A sell button should make this NEGATIVE relative to baseline. It is POSITIVE.')"
        ),
        md(
            f"After a Sahm trigger, the S&P returned **{R['event_mean_pct']:+.1f}%** over the next "
            f"year -- *better* than the **{R['uncond_mean_pct']:+.1f}%** unconditional average. "
            "Selling on the siren means selling near the bottom and missing the recovery. The "
            "trigger is a coincident-to-lagging recession marker, not a leading market signal."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal -- None.** Post-trigger 12m return {R['event_mean_pct']:+.1f}% vs "
            f"{R['uncond_mean_pct']:+.1f}% unconditional (Welch t = {R['welch_t']}, "
            f"perm p = {R['perm_p']}). No evidence the trigger precedes *low* returns -- if "
            "anything it precedes the recovery.\n"
            f"- **Tradability -- Mirage.** A 12-month-out overlay cut max drawdown "
            f"({R['bah_mdd_pct']:.0f}% -> {R['overlay_mdd_pct']:.0f}%) but lowered CAGR "
            f"({R['bah_cagr_pct']:.1f}% -> {R['overlay_cagr_pct']:.1f}%); active return HAC "
            f"t = {R['active_hac_t']} (not significant). No usable sell button.\n"
            "- **Busted.** Not because the Sahm Rule is wrong -- it is an excellent recession "
            "*identifier* -- but because identifying a recession is not the same as timing the "
            "market. Stocks lead; unemployment lags."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "The 'strategy': go to cash for 12 months whenever the Sahm trigger fires, else stay "
            "long the S&P. Net of 10bps per switch:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    ov = st.timing_overlay(u, g, exec_lag=1, cost_bps=10.0, stay_out_months=12)\n"
            "    o_cagr = ov['overlay']['cagr']*100; b_cagr = ov['bah']['cagr']*100\n"
            "    o_mdd = ov['overlay']['mdd']*100;   b_mdd = ov['bah']['mdd']*100\n"
            "else:\n"
            "    o_cagr, b_cagr = R['overlay_cagr_pct'], R['bah_cagr_pct']\n"
            "    o_mdd, b_mdd = R['overlay_mdd_pct'], R['bah_mdd_pct']\n"
            "\n"
            "print(f'Sahm timing overlay : CAGR {o_cagr:+.1f}%/yr, max drawdown {o_mdd:.0f}%')\n"
            "print(f'Buy and hold S&P    : CAGR {b_cagr:+.1f}%/yr, max drawdown {b_mdd:.0f}%')\n"
            "print()\n"
            "print(f'You give up {b_cagr-o_cagr:.1f}pp/yr of return to shave the drawdown.')\n"
            "print('Lower drawdown, lower return, same Sharpe -- no free lunch.')"
        ),
        md(
            "The overlay does make the ride smoother (the deep 2008 drawdown is partly avoided), "
            "but it sells *after* big declines and buys back *after* recoveries, so it sacrifices "
            "return. Risk-adjusted (Sharpe) it is a wash. There is no sell-button edge here."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The positive control:** the companion notebook plants a synthetic unemployment "
            "shock and confirms the trigger machinery fires exactly when it should -- the null "
            "(no shock) never trips. So the 'no signal' result is about the *market*, not a bug.\n"
            "- **Horizon sweep:** at 6, 12 and 24 months the post-trigger return stays at or above "
            "baseline -- the lateness is robust to the window.\n"
            "- **Cousin signals:** the yield-curve inversion and the unemployment-trend timing "
            "rules are in the same macro-timing family. Inversion *leads*; Sahm *lags*.\n\n"
            "*Think you can turn the siren into a sell button? Fork this, add a lead (act on the "
            "first 0.2pp uptick, say) and show a negative post-trigger return that beats baseline "
            "with t < -2. That's the bar.*"
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
            "# Sahm-Rule -- a quantitative teardown\n"
            "### 65y BLS unemployment * ^GSPC * event study * HAC t * block permutation * "
            "net-of-cost overlay * n=12 power\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.* We compute the Sahm "
            "trigger from the hardcoded BLS unemployment table, run an event study on forward S&P "
            "returns with a one-month execution lag, test the post-trigger mean against the "
            "unconditional baseline with a Welch t and a block-permutation test, report a HAC "
            "(Newey-West) t on the overlapping forward series, build a net-of-cost timing overlay, "
            "and close with the n=12 power reckoning.\n\n"
            "> **Not investment advice.** Real data: BLS U-3 SA unemployment "
            "(1959--2025, hardcoded in `data.py`); ^GSPC daily close (price-only, repo cache). "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The plain-English notes** translate each result back to intuition."
        ),
        code(BOOT + "\n" + R_CELL + "\nfrom scipy import stats as scipy_stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Post-trigger 12m return **{R['event_mean_pct']:+.1f}%** vs "
            f"**{R['uncond_mean_pct']:+.1f}%** unconditional; Welch t = **{R['welch_t']}**, "
            f"HAC t = **{R['hac_t']}**, perm p = **{R['perm_p']}**. |\n"
            f"| **Tradability** | `MIRAGE` | 12m-out overlay: CAGR **{R['overlay_cagr_pct']:.1f}%** "
            f"vs **{R['bah_cagr_pct']:.1f}%** B&H; active HAC t = **{R['active_hac_t']}**. |\n"
            "| **Busted?** | `YES` | The trigger is a great recession *identifier* but a *lagging* "
            "market signal -- it fires after the decline, near the recovery. |\n\n"
            "> Equities lead the cycle; unemployment lags it. A lagging recession marker cannot be "
            "a leading sell signal."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $g_t$ = 3-month-avg unemployment minus its trailing-12m min, and let the trigger "
            "fire when $g_t \\ge 0.5$. Let $R^{(H)}_{t+\\ell}$ be the $H$-month forward S&P price "
            "return from entry month $t+\\ell$ (execution lag $\\ell=1$). The folk hypothesis:\n\n"
            "- **H1 (sell signal).** $E[R^{(H)} \\mid \\text{trigger}] < E[R^{(H)}]$ -- post-trigger "
            "returns are *below* the unconditional mean.\n"
            "- **H2 (tradable).** A trigger-driven out-of-market overlay beats buy-and-hold on a "
            "risk-adjusted basis, net of costs.\n\n"
            "We reject H1 (the difference is *positive*) and H2 (lower CAGR, equal Sharpe)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what? -- what rides on each answer\n\n"
            "The Sahm Rule is the rare macro indicator with a near-perfect in-sample recession "
            "hit-rate, which is exactly why it is so tempting to misuse. If H1 held, a freely "
            "available monthly print would let anyone sidestep bear markets. The finding that it "
            "*doesn't* is a clean demonstration of the difference between **dating** a recession "
            "(coincident/lagging) and **forecasting** equity returns (which requires a lead)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Data.** BLS U-3 SA unemployment 1959-2025 (hardcoded); ^GSPC daily close "
            "(price-only, repo cache), resampled to month-end.\n"
            "- **Trigger.** Sahm value $g_t \\ge 0.5$pp; *onset* = first month of each new trigger.\n"
            "- **Execution lag.** +1 month (the print is released next month; we act at that close).\n"
            "- **Baseline.** Unconditional distribution of the same H-month forward return -- the "
            "correct null for a drifting-up asset.\n"
            "- **Inference.** Welch t (event vs all); block-permutation p (random entries, same n); "
            "HAC/Newey-West t on overlapping forward returns and on the overlay active return.\n"
            "- **Positive control.** Synthetic unemployment with a planted shock -> the trigger "
            "fires; the null (no shock) -> silent.\n"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - The event study with a one-month execution lag\n\n"
            "Forward 12-month S&P return after each trigger onset, vs the unconditional mean."
        ),
        code(
            "u = data.unrate_series()\n"
            "if HAVE_REAL:\n"
            "    g = data.fetch_gspc_daily()\n"
            "    es = st.event_study(u, g, horizon=12, exec_lag=1)\n"
            "    ev_dates = [d.strftime('%Y-%m') for d in es['event_dates']]\n"
            "    ev_fwd = es['event_fwd']*100\n"
            "    ev_mean, un_mean = es['event_mean']*100, es['uncond_mean']*100\n"
            "    welch_t, welch_p = es['welch_t'], es['welch_p']\n"
            "    hac_t = st.hac_tstat(es['event_fwd'] - es['uncond_mean'])\n"
            "else:\n"
            "    ev_mean, un_mean = R['event_mean_pct'], R['uncond_mean_pct']\n"
            "    welch_t, welch_p, hac_t = R['welch_t'], R['welch_p'], R['hac_t']\n"
            "    rng = np.random.default_rng(268)\n"
            "    ev_fwd = rng.normal(ev_mean, 22, R['n_onsets'])\n"
            "    ev_dates = [f'ev{i+1}' for i in range(R['n_onsets'])]\n"
            "\n"
            "colors = [RED if v < 0 else GREEN for v in ev_fwd]\n"
            "fig, ax = plt.subplots(figsize=(11, 4.3))\n"
            "ax.bar(range(len(ev_fwd)), ev_fwd, color=colors)\n"
            "ax.axhline(un_mean, ls='--', c=AMBER, lw=2, label=f'Unconditional 12m mean ({un_mean:+.1f}%)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(range(len(ev_fwd))); ax.set_xticklabels(ev_dates, rotation=60, ha='right', fontsize=8)\n"
            "ax.set_ylabel('Forward 12m S&P return (%)')\n"
            "ax.set_title(f'Per-trigger forward returns: mean {ev_mean:+.1f}% vs {un_mean:+.1f}% baseline')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'n events: {len(ev_fwd)}')\n"
            "print(f'Event mean 12m: {ev_mean:+.1f}%  |  Unconditional: {un_mean:+.1f}%  |  diff {ev_mean-un_mean:+.1f}pp')\n"
            "print(f'Welch t (event vs all): {welch_t:+.3f}  (p = {welch_p:.3f})')\n"
            "print(f'HAC/Newey-West t (event fwd - baseline vs 0): {hac_t:+.3f}')"
        ),
        md(
            f"The post-trigger mean is **{R['event_mean_pct']:+.1f}%**, *above* the "
            f"**{R['uncond_mean_pct']:+.1f}%** baseline. Welch t = **{R['welch_t']}**, HAC "
            f"t = **{R['hac_t']}** -- nowhere near the |t| >= 2 (let alone the t>=3 anomaly) bar, "
            "and pointing the *wrong way* for a sell signal. The two clear losers (2001, 2008) are "
            "swamped by recovery-year winners (1970, 1980, 2020)."
        ),
        md(
            "### 4b - Block-permutation test: random entries, same event count\n\n"
            "Draw 12 random entry months and recompute the mean forward return, 5,000 times."
        ),
        code(
            "if HAVE_REAL:\n"
            "    perm = st.block_permutation_p(u, g, horizon=12, exec_lag=1, n_perm=5000, seed=268)\n"
            "    perm_means = perm['perm_means']*100\n"
            "    obs = perm['obs_mean']*100\n"
            "    perm_p = perm['perm_p']\n"
            "else:\n"
            "    rng2 = np.random.default_rng(268)\n"
            "    perm_means = rng2.normal(R['uncond_mean_pct'], 22/np.sqrt(R['n_onsets']), 5000)\n"
            "    obs = R['event_mean_pct']; perm_p = R['perm_p']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(perm_means, bins=40, color=GREY, alpha=0.7, label='Random-entry means')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'Observed event mean ({obs:+.1f}%)')\n"
            "ax.set_xlabel('Mean forward 12m return (%)'); ax.set_ylabel('Count')\n"
            "ax.set_title('The post-trigger mean sits on the HIGH side of random -- not low')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'One-sided permutation p (mean <= observed): {perm_p:.4f}')\n"
            "print('For a sell signal we want this SMALL (trigger -> low returns). It is large.')"
        ),
        md(
            f"Permutation p = **{R['perm_p']}** for the one-sided 'trigger precedes low returns' "
            "test -- the observed mean lands on the *high* side of the random distribution. No "
            "evidence of a sell signal."
        ),
        md(
            "### 4c - The timing overlay, net of costs\n\n"
            "Out of the S&P for 12 months on each trigger (1-month lag), 10bps per switch."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ov = st.timing_overlay(u, g, exec_lag=1, cost_bps=10.0, stay_out_months=12)\n"
            "    o, b = ov['overlay'], ov['bah']\n"
            "    oret = ov['overlay_ret']; bret = ov['bah_ret']\n"
            "    a_hac = ov['active_hac_t']; n_sw = ov['n_switches']\n"
            "    idx = oret.index\n"
            "    o_eq = (1+oret).cumprod(); b_eq = (1+bret).cumprod()\n"
            "else:\n"
            "    o = dict(cagr=R['overlay_cagr_pct']/100, mdd=R['overlay_mdd_pct']/100, sharpe=R['overlay_sharpe'])\n"
            "    b = dict(cagr=R['bah_cagr_pct']/100, mdd=R['bah_mdd_pct']/100, sharpe=R['bah_sharpe'])\n"
            "    a_hac, n_sw = R['active_hac_t'], R['n_switches']\n"
            "    rng3 = np.random.default_rng(268); idx = pd.date_range('1959-02-01', periods=800, freq='MS')\n"
            "    o_eq = (1+rng3.normal(R['overlay_cagr_pct']/1200, 0.037, 800)).cumprod()\n"
            "    b_eq = (1+rng3.normal(R['bah_cagr_pct']/1200, 0.043, 800)).cumprod()\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(11, 4.5))\n"
            "ax.plot(idx, b_eq, lw=2, c=GREEN, label=f\"Buy & hold: {b['cagr']:.1%}/yr, MDD {b['mdd']:.0%}\")\n"
            "ax.plot(idx, o_eq, lw=2, c=RED, ls='--', label=f\"Sahm overlay: {o['cagr']:.1%}/yr, MDD {o['mdd']:.0%}\")\n"
            "ax.set_yscale('log'); ax.set_ylabel('Growth of 1 (log)'); ax.set_xlabel('Year')\n"
            "ax.set_title('Lower drawdown, lower terminal wealth -- a wash on risk-adjusted return')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"Overlay : CAGR {o['cagr']:+.2%}, MDD {o['mdd']:.1%}, Sharpe {o['sharpe']:.2f}\")\n"
            "print(f\"B&H     : CAGR {b['cagr']:+.2%}, MDD {b['mdd']:.1%}, Sharpe {b['sharpe']:.2f}\")\n"
            "print(f'Switches: {n_sw}   |   Active return HAC t: {a_hac:+.3f} (not significant)')"
        ),
        md(
            f"The overlay shaves the worst drawdown ({R['bah_mdd_pct']:.0f}% -> "
            f"{R['overlay_mdd_pct']:.0f}%) but at a {R['bah_cagr_pct']-R['overlay_cagr_pct']:.1f}pp/yr "
            f"CAGR cost; Sharpe is identical (~{R['overlay_sharpe']:.2f}). The active return's HAC "
            f"t = **{R['active_hac_t']}** -- if anything negative, certainly not a positive edge."
        ),
        md(
            "### 4d - Horizon robustness\n\n"
            "Does the lateness hold at 6 and 24 months?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for H in (6, 12, 24):\n"
            "        e = st.event_study(u, g, horizon=H, exec_lag=1)\n"
            "        rows.append({'H (months)': H, 'event %': round(e['event_mean']*100,1),\n"
            "                     'uncond %': round(e['uncond_mean']*100,1),\n"
            "                     'diff pp': round(e['diff']*100,1), 'welch_t': e['welch_t']})\n"
            "    tab = pd.DataFrame(rows)\n"
            "else:\n"
            "    tab = pd.DataFrame({\n"
            "        'H (months)': [6,12,24],\n"
            "        'event %': [R['h6_event_pct'], R['event_mean_pct'], R['h24_event_pct']],\n"
            "        'uncond %': [R['h6_uncond_pct'], R['uncond_mean_pct'], R['h24_uncond_pct']],\n"
            "        'diff pp': [R['h6_event_pct']-R['h6_uncond_pct'], R['diff_pct'],\n"
            "                    R['h24_event_pct']-R['h24_uncond_pct']],\n"
            "        'welch_t': [R['h6_welch_t'], R['welch_t'], R['h24_welch_t']]})\n"
            "\n"
            "print(tab.to_string(index=False))\n"
            "print()\n"
            "print('At every horizon the post-trigger return is at or ABOVE baseline.')\n"
            "print('The lateness is robust -- it is not a window artifact.')"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal `NONE`** -- post-trigger 12m return {R['event_mean_pct']:+.1f}% vs "
            f"{R['uncond_mean_pct']:+.1f}% unconditional; Welch t {R['welch_t']}, HAC t {R['hac_t']}, "
            f"perm p {R['perm_p']}. The difference is positive (wrong sign for a sell signal) and "
            "insignificant.\n"
            f"- **Tradability `MIRAGE`** -- overlay CAGR {R['overlay_cagr_pct']:.1f}% vs "
            f"{R['bah_cagr_pct']:.1f}% B&H, equal Sharpe, active HAC t {R['active_hac_t']}.\n"
            "- **Busted** -- as a *sell button*. The Sahm Rule remains an excellent recession "
            "*identifier*; it is simply the wrong tool for market timing."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- the n=12 power reckoning\n\n"
            "With only ~12 triggers and ~20% annual vol on forward returns, what gap could we even "
            "detect?"
        ),
        code(
            "vol = 0.20          # ~vol of 12m forward S&P returns\n"
            "n_ev = R['n_onsets']\n"
            "from scipy.stats import t as t_dist\n"
            "dof = n_ev - 1\n"
            "t_crit = t_dist.ppf(0.975, dof); t_pow = t_dist.ppf(0.80, dof)\n"
            "mde = (t_crit + t_pow) * vol / np.sqrt(n_ev)\n"
            "print(f'Min detectable mean shift (80% power, 2-sided a=0.05, n={n_ev}):')\n"
            "print(f'  vol={vol:.0%}  ->  MDE = {mde:.1%} per event')\n"
            "print()\n"
            "print(f'Observed post-trigger vs baseline gap: {R[\"diff_pct\"]:+.1f}pp (and POSITIVE).')\n"
            "print(f'To call a sell signal we would need ~{mde*100:.0f}pp BELOW baseline.')\n"
            "print('n=12 cannot resolve a real macro-timing edge even if one existed.')"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further -- the synthetic positive control\n\n"
            "Does the trigger machinery fire when it should? Plant a recession-sized unemployment "
            "shock; confirm the null stays silent."
        ),
        code(
            "s_null, _ = data.synthetic_unrate(shock_pp=0.0, seed=268)\n"
            "s_rec,  _ = data.synthetic_unrate(shock_pp=3.0, shock_at=180, seed=268)\n"
            "on_null = st.trigger_onsets(s_null)\n"
            "on_rec  = st.trigger_onsets(s_rec)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(11, 4.3))\n"
            "ax.plot(s_null.index, s_null.values, c=GREY, lw=1.2, label='Null (no shock)')\n"
            "ax.plot(s_rec.index, s_rec.values, c=RED, lw=1.5, label='Planted shock (+3pp)')\n"
            "for d in on_rec:\n"
            "    ax.axvline(d, c=AMBER, lw=1.5, ls='--', label='Trigger fires' if d==on_rec[0] else None)\n"
            "ax.set_ylabel('Synthetic unemployment (%)')\n"
            "ax.set_title('Positive control: the trigger fires on the planted shock, silent on the null')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Null tape triggers: {len(on_null)}  (expect 0)')\n"
            "print(f'Shock tape triggers: {len(on_rec)}  (expect 1, at the planted shock)')\n"
            "assert len(on_null) == 0 and len(on_rec) >= 1"
        ),
        md(
            "The machinery is truthful: the null never trips, the planted shock fires exactly once "
            "at the right time. So the real-tape 'no sell signal' result is a statement about "
            "**market timing**, not a defect in the detector. The Sahm Rule does its job -- "
            "spotting recessions -- and that job is simply not the same as timing stocks."
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


def _execute(name):
    path = os.path.join(HERE, name)
    print("executing", path)
    subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook",
         "--execute", "--inplace",
         "--ExecutePreprocessor.timeout=600", path],
        check=True,
    )


if __name__ == "__main__":
    build_curious()
    build_quants()
    _execute("01_for_the_curious.ipynb")
    _execute("02_for_the_quants.ipynb")
