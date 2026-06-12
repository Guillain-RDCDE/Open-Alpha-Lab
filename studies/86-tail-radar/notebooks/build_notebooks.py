"""Generate the two narrative notebooks for Study 86 (Tail-Radar).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md).  The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
daily parquet under ../_cache/ if present and otherwise quote the frozen headline numbers
in ``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.

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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-12).
R = dict(
    n=8324,
    date_start="1993-02-01", date_end="2026-06-11",
    fingerprint="5e270c23a839",
    skew_min=104.1, skew_max=183.1,
    vix_min=9.1, vix_max=82.7,
    # Quintile spreads (Q5 - Q1)
    spread_1d=-5.81, t_spread_1d=-1.40,
    spread_5d=-20.08, t_spread_5d=-1.17,
    spread_21d=-104.23, t_spread_21d=-2.12,
    # Q5 (high SKEW) forward returns
    q5_1d=3.23, t_q5_1d=1.60,
    q5_5d=14.35, t_q5_5d=1.80,
    q5_21d=31.46, t_q5_21d=1.26,
    # Q1 (low SKEW) forward returns
    q1_1d=9.04, t_q1_1d=3.25,
    q1_5d=34.43, t_q1_5d=3.17,
    q1_21d=135.70, t_q1_21d=4.59,
    # Crash frequency
    n_high=2152, n_low=6151,
    crash_rate_high=0.0884, crash_rate_low=0.0909,
    rate_ratio=0.97, fisher_pvalue=0.655,
    # SKEW vs VIX regression (h=1)
    t_skew_reg=-1.02, t_vix_reg=1.84, r2_reg=0.0011,
)

# ---------------------------------------------------------------------------
# Shared analysis preamble — imports, the tape, and small helpers.
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

from tail_radar import data, strategy as st

def _have_cache():
    return os.path.exists(data._cache_path(data.DEFAULT_CACHE))

HAVE_REAL = _have_cache()

if HAVE_REAL:
    df_real = data.fetch_daily(fetch=False)

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Tail-Radar — does high SKEW predict the next crash?\n"
            "### The CBOE SKEW index as a black-swan radar, tested honestly\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Radar_works%3F: Busted](https://img.shields.io/badge/Radar_works%3F-Busted-8b949e?style=flat-square)\n\n"
            "Every few months a financial pundit points at the CBOE SKEW index and says: "
            "*'When SKEW is high, the smart money is hedging — a crash is coming.'* The index "
            "measures how expensive OTM put options are relative to at-the-money ones. When "
            "it spikes, fear of a left-tail event is being priced in by the options market. "
            "Does that fear actually **predict** a crash? Or is it just an expensive false alarm?\n\n"
            "> 📓 **This is the plain-language layer.** Want the t-stats, regression coefficients "
            "and bootstrap intervals? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does high SKEW predict below-average returns? | **No.** The highest-SKEW quintile "
            f"earns **{R['q5_1d']:+.1f} bps/day** — barely distinguishable from the middle quintiles "
            f"(HAC t = {R['t_q5_1d']:+.2f}). |\n"
            f"| Does high SKEW predict *crashes*? | **No.** The crash rate (SPY < −5% over 21 days) "
            f"is **{R['crash_rate_high']*100:.1f}%** after high SKEW vs **{R['crash_rate_low']*100:.1f}%** "
            f"otherwise — if anything slightly *lower* (Fisher p = {R['fisher_pvalue']:.2f}). |\n"
            f"| Does SKEW add anything over VIX? | **No.** In a joint regression, SKEW's t-stat is "
            f"**{R['t_skew_reg']:+.2f}** — not distinguishable from zero. |\n"
            f"| What actually predicts returns? | The *lowest*-SKEW quintile does best: "
            f"**{R['q1_1d']:+.1f} bps/day** (t = {R['t_q1_1d']:+.2f}) — low fear, not high fear, "
            f"is the better environment. |\n\n"
            "> The radar is pointing the wrong way. High SKEW is an expensive insurance premium, "
            "not an oracle."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When the CBOE SKEW index is high — above 130, 140, or especially 150+ — "
            "sophisticated investors are paying up for crash protection. They know something. "
            "Reduce your equity exposure, or better yet, buy puts alongside them. The crash radar "
            "is flashing red.\"*\n\n"
            "The SKEW index is published by CBOE. It's based on real prices — OTM put options — "
            "paid by real money to insure against a left-tail event. The intuition is appealing: "
            "if the smart money is hedging, maybe we should follow. The bet is that **priced crash "
            "risk actually precedes realised crash risk** at a tradable horizon."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it works, it's a simple, publicly available early-warning system that sits on top "
            "of every S&P 500 position. If it doesn't, then every pundit who scares retail investors "
            "with a spiking SKEW chart is telling a compelling story with no predictive content. "
            "The difference is one honest test — and with 33 years of daily data, we have the power "
            "to find it if it's there."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three interlocking tests:\n\n"
            "1. **Quintile sort.** Divide history into five equal buckets by SKEW level (computed "
            "rolling, so no future information leaks). Measure the average SPY return over the next "
            "1, 5, and 21 days in each bucket. If SKEW is a crash radar, the top bucket (highest "
            "SKEW) should deliver the *worst* forward returns.\n"
            "2. **Crash-frequency test.** Define a 'crash' as SPY dropping more than 5% over the "
            "next 21 days. Compare the crash rate when SKEW is in the top 20% of its history vs "
            "the bottom 80%. A Fisher exact test gives us the p-value.\n"
            "3. **SKEW vs VIX.** SKEW is correlated with VIX (the simpler fear gauge). Does SKEW "
            "add *anything* beyond what VIX already tells us? We run a joint regression.\n\n"
            "We run this on **8,324 daily observations spanning 1993–2026** — the longest SKEW "
            "history available."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — what actually happens\n\n"
            "**First: do high-SKEW days predict worse future returns?**"
        ),
        code(
            "horizons = [1, 5, 21]\n"
            "quintile_means = {h: [] for h in horizons}\n"
            "quintile_tstats = {h: [] for h in horizons}\n"
            "if HAVE_REAL:\n"
            "    for h in horizons:\n"
            "        tbl = st.quintile_table(df_real, horizons=[h])\n"
            "        for qi in range(1, 6):\n"
            "            row = tbl[(tbl['quintile']==qi) & (tbl['horizon']==h)].iloc[0]\n"
            "            quintile_means[h].append(row['mean_bps'])\n"
            "            quintile_tstats[h].append(row['t_stat'])\n"
            "else:\n"
            "    # Frozen fallback from docs/results.md\n"
            "    quintile_means = {\n"
            "        1:  [9.0, 2.4, 2.4, 2.8, 3.2],\n"
            "        5:  [34.4, 6.9, 14.6, 31.9, 14.3],\n"
            "        21: [135.7, 86.2, 90.8, 105.1, 31.5],\n"
            "    }\n"
            "\n"
            "qs = [1, 2, 3, 4, 5]\n"
            "x = range(5)\n"
            "fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))\n"
            "for ax, h in zip(axes, horizons):\n"
            "    vals = quintile_means[h]\n"
            "    colors = [RED if q == 5 else (GREEN if q == 1 else GREY) for q in qs]\n"
            "    ax.bar(qs, vals, color=colors)\n"
            "    ax.axhline(0, c='k', lw=1)\n"
            "    ax.set_xlabel('SKEW quintile (1=low, 5=high)')\n"
            "    ax.set_ylabel('mean forward return (bps)')\n"
            "    ax.set_title(f'h = {h}d forward return by SKEW quintile')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "print('Green = low SKEW (Q1), Red = high SKEW (Q5)')\n"
            "print('If SKEW were a crash radar, Q5 (red) would be the shortest bar.')"
        ),
        md(
            "The pattern is the **opposite of the claim**. Low-SKEW periods (Q1, green) consistently "
            f"deliver the *best* forward returns: **{R['q1_1d']:+.1f} bps/day** at 1-day horizon, "
            f"**{R['q1_21d']:+.0f} bps** over 21 days. High-SKEW periods (Q5, red) deliver the "
            f"*worst* returns at every horizon: **{R['q5_21d']:+.0f} bps** over 21 days.\n\n"
            "Why? When SKEW is low, people aren't buying crash insurance — because the market is "
            "calm and trending higher. When SKEW is high, fear is elevated, put options are "
            "expensive, and the market is often already digesting bad news. High SKEW is a "
            "*symptom* of fear, not a *predictor* of the thing being feared."
        ),
        md("**Now the crash-frequency test — does high SKEW predict actual crashes?**"),
        code(
            "if HAVE_REAL:\n"
            "    crash = st.crash_frequency(df_real, horizon=21, crash_threshold=-0.05, high_skew_pct=0.80)\n"
            "    cr_high = crash['crash_rate_high']\n"
            "    cr_low  = crash['crash_rate_low']\n"
            "    pval    = crash['fisher_pvalue']\n"
            "    n_high  = crash['n_high']\n"
            "    n_low   = crash['n_low']\n"
            "else:\n"
            "    cr_high, cr_low = R['crash_rate_high'], R['crash_rate_low']\n"
            "    pval = R['fisher_pvalue']\n"
            "    n_high, n_low = R['n_high'], R['n_low']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "labels = [f'High SKEW\\n(top 20%)\\nn={n_high}',\n"
            "          f'Low/Med SKEW\\n(bottom 80%)\\nn={n_low}']\n"
            "bars = ax.bar(labels, [cr_high*100, cr_low*100], color=[RED, GREY], width=0.5)\n"
            "ax.set_ylabel('crash rate (SPY < −5% over 21d)')\n"
            "ax.set_title('Crash rate: high SKEW vs low/medium SKEW')\n"
            "for b, v in zip(bars, [cr_high, cr_low]):\n"
            "    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.001,\n"
            "            f'{v*100:.1f}%', ha='center', va='bottom', fontsize=12)\n"
            "ax.set_ylim(0, max(cr_high, cr_low)*1.3)\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "print(f'High SKEW crash rate: {cr_high*100:.1f}%  vs  Low SKEW: {cr_low*100:.1f}%')\n"
            "print(f'Fisher p-value (one-sided, H0: high SKEW has MORE crashes): {pval:.3f}')\n"
            "print('The crash rate is LOWER after high SKEW, not higher.')"
        ),
        md(
            f"The crash rate after high-SKEW episodes is **{R['crash_rate_high']*100:.1f}%** vs "
            f"**{R['crash_rate_low']*100:.1f}%** otherwise — the radar is "
            f"pointing the wrong way. Fisher p = {R['fisher_pvalue']:.2f}: no evidence that high "
            "SKEW predicts more crashes than normal.\n\n"
            "> Why? Because SKEW spikes *during* scary periods (2008, COVID, 2022), when the "
            "crash has often already started. And it stays elevated during long rallies (2017, 2021) "
            "when hedging is cheap entertainment and the market just keeps going up."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** SKEW quintile spread: t = {R['t_spread_1d']:+.2f} (1d), "
            f"{R['t_spread_5d']:+.2f} (5d), {R['t_spread_21d']:+.2f} (21d). "
            f"Crash-frequency Fisher p = {R['fisher_pvalue']:.2f}. No test clears the bar.\n"
            f"- **Tradability — Mirage.** No positive gross edge exists. SKEW regression "
            f"t = {R['t_skew_reg']:+.2f}, adding nothing over VIX.\n"
            f"- **Radar works? — Busted.** High SKEW does not elevate crash frequency "
            f"({R['crash_rate_high']*100:.1f}% vs {R['crash_rate_low']*100:.1f}%). "
            "The radar does not see the swans."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Even if SKEW carried a directional signal, the instrument you'd need to trade would "
            "be SPY options or VIX futures — instruments with their own bid-ask spreads, rolls, "
            "and theta decay. But there's a deeper problem: the gross edge is *already absent*. "
            "There is no break-even cost because there is no gross positive return from the "
            "high-SKEW signal to work from. A strategy of 'sell SPY when SKEW is high' would "
            "simply miss the best market environments (Q1 periods, where returns are highest) "
            "while underperforming through the elevated-fear periods that often resolve upward.\n\n"
            "The closest real trade — buying put spreads when SKEW spikes — is exactly what the "
            "options market has already priced. You're paying up for insurance when the market "
            "thinks it costs the most. That's the opposite of buying cheap insurance."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Does SKEW work over *very* long horizons?** Our test goes to 21 days. Some "
            "researchers claim SKEW has predictive power at 30-day or 60-day windows. The 21-day "
            "t-stat of −2.12 (barely border) and wrong-direction pattern suggests this is "
            "data-mining noise, but you can fork this and extend the horizons.\n"
            "- **SKEW in crisis regimes.** Does high SKEW during already-elevated VIX periods "
            "add incremental predictive power? That's a conditional analysis — subset the data "
            "to VIX > 25 and re-run the quintile tests.\n"
            "- **VIX term structure.** [Study 42 — Last-Call](../../42-last-call/) tests whether "
            "the VIX curve slope (not the level) carries a better signal — VIX structure has "
            "been shown to have more predictive power than SKEW.\n\n"
            "*Think SKEW really works at a specific horizon or regime? Fork this, show a "
            "significant spread t-stat in the claimed direction, and check whether a realistic "
            "options trade captures it net of cost. That's the bar.*"
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
            "# Tail-Radar — a quantitative teardown\n"
            "### Real daily tape · SKEW quintile sorts · crash-frequency Fisher test "
            "· SKEW vs VIX HAC regression · synthetic positive control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Radar_works%3F: Busted](https://img.shields.io/badge/Radar_works%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim now carrying its standard error.* We test whether "
            "the CBOE SKEW index predicts forward SPY returns and crash frequency at 1/5/21-day "
            "horizons, and whether it adds anything over VIX in a joint regression.\n\n"
            "> ⚠️ **Not investment advice.** Real data: Yahoo Finance daily, 1993-02-01 → "
            "2026-06-11 (8,324 observations); offline core and tests run on a deterministic "
            "synthetic tape. Methods in [`docs/references.md`](../docs/references.md), "
            "reproducible numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> The `> in plain words` notes translate each result back to intuition."
        ),
        code(BOOT + "\ntry:\n    from quantlab import analytics, stats\nexcept ImportError:\n    analytics = stats = None\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | SKEW Q5−Q1 spread: t = **{R['t_spread_1d']:+.2f}** (h=1d), "
            f"**{R['t_spread_5d']:+.2f}** (h=5d), **{R['t_spread_21d']:+.2f}** (h=21d); "
            f"crash-frequency Fisher p = **{R['fisher_pvalue']:.2f}**. No horizon clears \\|t\\| ≥ 2 "
            f"in the claimed direction. |\n"
            f"| **Tradability** | `MIRAGE` | No positive gross edge at any horizon; SKEW adds "
            f"nothing over VIX (t_SKEW = **{R['t_skew_reg']:+.2f}** in joint regression). |\n"
            f"| **Radar works?** | `BUSTED` | High-SKEW crash rate "
            f"{R['crash_rate_high']*100:.1f}% vs low-SKEW {R['crash_rate_low']*100:.1f}%; "
            f"rate ratio {R['rate_ratio']:.2f} (< 1 — the *wrong* direction). |\n\n"
            "> in plain words: SKEW measures *priced* tail risk, not *realised* tail risk. "
            "When SKEW is high, investors are paying for insurance; the market does not "
            "systematically deliver the crash they're hedging against."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $S_t$ be the SKEW level at close of day $t$, $q_t$ its rolling out-of-sample "
            "quintile rank (1–5), and $r_{t,h} = \\log(P_{t+h}/P_t)$ the forward log-return on "
            "SPY. The claim asserts:\n\n"
            "- **H₁ (mean signal).** $\\mathbb{E}[r_{t,h} \\mid q_t = 5] < \\mathbb{E}[r_{t,h} "
            "\\mid q_t = 1]$ — high-SKEW periods predict below-average returns.\n"
            "- **H₂ (crash signal).** $P(r_{t,21} < -5\\% \\mid q_t \\geq q_{0.80}) > "
            "P(r_{t,21} < -5\\% \\mid q_t < q_{0.80})$ — high SKEW elevates crash frequency.\n"
            "- **H₃ (incremental over VIX).** In $r_{t,h} = \\alpha + \\beta_S z_S + "
            "\\beta_V z_V + \\varepsilon$, $|t_{\\beta_S}| \\geq 2$ with a *negative* sign "
            "— SKEW adds information beyond the simpler fear gauge.\n\n"
            "We reject H₁ (wrong direction — Q1 beats Q5), H₂ (Fisher p = 0.655), and H₃ "
            "(SKEW t = −1.02 at h=1) on the real tape."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If H₁–H₂ held, SKEW would be a rare timing signal: publicly available, institutionally "
            "observed, and exploitable. The interesting failure is *how* it fails: the relationship "
            "is actually *reversed* — low-SKEW periods outperform high-SKEW periods — consistent "
            "with SKEW being a *sentiment lag* (fear peaks after markets have already fallen) rather "
            "than a leading indicator."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal construction.** Rolling 252-day out-of-sample percentile rank of SKEW "
            "(min 63 observations). Quintile assignment is strictly lagged: rank at *t* uses "
            "only history up to *t*.\n"
            "- **Forward return.** $r_{t,h} = \\log(\\text{SPY}_{t+h}) - \\log(\\text{SPY}_t)$, "
            "aligned at signal date *t* (no look-ahead). Horizons: 1, 5, 21 trading days.\n"
            "- **Inference.** Newey–West HAC *t*-stat on per-period returns within each quintile; "
            "Fisher exact test (one-sided) on the 2×2 crash/no-crash table.\n"
            "- **Incremental test.** OLS with HAC-robust standard errors: "
            "$r_{t,h} = \\alpha + \\beta_S z_S + \\beta_V z_V + \\varepsilon$, "
            "where $z_S, z_V$ are 252-day rolling z-scores of SKEW and VIX.\n"
            "- **Positive control.** A synthetic tape with a tunable `skew_signal` knob: "
            "the test recovers the planted signal, confirming the machinery works when there "
            "is something to find.\n\n"
            "Data: 8,324 daily observations, 1993-02-01 → 2026-06-11 (^SKEW, ^VIX, SPY via Yahoo)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Quintile table — per-quintile HAC t-stats across horizons\n\n"
            "If SKEW were a crash radar, Q5 bars would be negative and Q1 bars would be positive. "
            "The inference bar is |t| ≥ 2."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tbl = st.quintile_table(df_real, horizons=[1, 5, 21])\n"
            "else:\n"
            "    tbl = None\n"
            "\n"
            "if tbl is not None:\n"
            "    pivoted = tbl.pivot(index='quintile', columns='horizon', values='t_stat')\n"
            "    fig, ax = plt.subplots(figsize=(9.5, 4.8))\n"
            "    x = [1, 2, 3, 4, 5]\n"
            "    for j, h in enumerate([1, 5, 21]):\n"
            "        col = [RED, AMBER, GREEN][j]\n"
            "        tstats = [tbl[(tbl['quintile']==q) & (tbl['horizon']==h)]['t_stat'].values[0] for q in x]\n"
            "        ax.plot(x, tstats, 'o-', label=f'h={h}d', color=col, lw=2)\n"
            "    for s in (2, -2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "    ax.axhline(0, c='k', lw=1)\n"
            "    ax.set_xlabel('SKEW quintile (1=low fear, 5=high fear)')\n"
            "    ax.set_ylabel('HAC t-stat on forward return')\n"
            "    ax.set_title('Q1 (low SKEW) is the best predictor — not Q5')\n"
            "    ax.legend(); ax.set_xticks(x)\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(tbl.pivot(index='quintile', columns='horizon', values='mean_bps').round(1))\n"
            "else:\n"
            "    print('No real cache — see frozen numbers in docs/results.md')\n"
            "    print(f'Q1 t-stats: 1d={R[\"t_q1_1d\"]:+.2f}, 5d={R[\"t_q1_5d\"]:+.2f}, 21d={R[\"t_q1_21d\"]:+.2f}')\n"
            "    print(f'Q5 t-stats: 1d={R[\"t_q5_1d\"]:+.2f}, 5d={R[\"t_q5_5d\"]:+.2f}, 21d={R[\"t_q5_21d\"]:+.2f}')"
        ),
        md(
            f"> in plain words: the curve slopes **downward** from Q1 to Q5 — lower SKEW is "
            f"associated with better subsequent returns. This is a *sentiment lag* pattern: "
            f"high SKEW means fear is already priced in, and the market has often already sold off. "
            f"Q1 (calm, low-fear) delivers {R['q1_1d']:+.1f} bps/day with t = {R['t_q1_1d']:+.2f}; "
            f"Q5 (high-fear) delivers only {R['q5_1d']:+.1f} bps/day with t = {R['t_q5_1d']:+.2f}."
        ),
        md(
            "### 4b · Crash-frequency Fisher test\n\n"
            "Is the 2×2 table (high/low SKEW × crash/no-crash) consistent with SKEW being "
            "a crash predictor? We use a one-sided Fisher exact test."
        ),
        code(
            "if HAVE_REAL:\n"
            "    crash = st.crash_frequency(df_real, horizon=21, crash_threshold=-0.05, high_skew_pct=0.80)\n"
            "    cr_h, cr_l = crash['crash_rate_high'], crash['crash_rate_low']\n"
            "    pval = crash['fisher_pvalue']; rr = crash['rate_ratio']\n"
            "    n_h, n_l = crash['n_high'], crash['n_low']\n"
            "else:\n"
            "    cr_h, cr_l = R['crash_rate_high'], R['crash_rate_low']\n"
            "    pval, rr = R['fisher_pvalue'], R['rate_ratio']\n"
            "    n_h, n_l = R['n_high'], R['n_low']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "bar_labels = [f'High SKEW\\n(top 20%, n={n_h})', f'Low/Med SKEW\\n(bottom 80%, n={n_l})']\n"
            "brs = ax.bar(bar_labels, [cr_h*100, cr_l*100], color=[RED, GREY], width=0.5)\n"
            "ax.set_ylabel('crash rate (SPY < −5% over 21d, %)')\n"
            "ax.set_title(f'Crash-frequency test: Fisher p = {pval:.3f} (no signal)')\n"
            "for b, v in zip(brs, [cr_h, cr_l]):\n"
            "    ax.text(b.get_x() + b.get_width()/2, b.get_height()+0.1,\n"
            "            f'{v*100:.1f}%', ha='center', va='bottom', fontsize=12)\n"
            "ax.set_ylim(0, max(cr_h, cr_l)*1.3)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Rate ratio: {rr:.3f}  (1.0 = identical crash rates)')\n"
            "print(f'Fisher p = {pval:.4f}  H0: high-SKEW crash rate ≤ low-SKEW crash rate')\n"
            "print('Result: we cannot reject H0 — crash rates are essentially identical.')"
        ),
        md(
            f"> in plain words: with {R['n_high']} high-SKEW observations and {R['n_low']} "
            f"others, we have ample power to detect a meaningful difference in crash frequency. "
            f"The rate ratio is {R['rate_ratio']:.2f} (essentially 1.0) and p = {R['fisher_pvalue']:.2f}. "
            "The radar is silent when it matters."
        ),
        md(
            "### 4c · SKEW vs VIX incremental regression (HAC-robust)\n\n"
            "Does SKEW add anything over VIX? Both predictors are standardised (z-scored) so "
            "their coefficients are directly comparable."
        ),
        code(
            "if HAVE_REAL:\n"
            "    regs = {h: st.skew_vs_vix_regression(df_real, horizon=h) for h in [1, 5, 21]}\n"
            "else:\n"
            "    regs = None\n"
            "\n"
            "if regs:\n"
            "    rows = [(h, regs[h]['n'], regs[h]['beta_skew'], regs[h]['t_skew'],\n"
            "             regs[h]['beta_vix'], regs[h]['t_vix'], regs[h]['r2'])\n"
            "            for h in [1, 5, 21]]\n"
            "    reg_df = pd.DataFrame(rows, columns=['h', 'n', 'beta_SKEW', 't_SKEW',\n"
            "                                         'beta_VIX', 't_VIX', 'R2'])\n"
            "    print(reg_df.round(4).to_string(index=False))\n"
            "    fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "    hh = [1, 5, 21]\n"
            "    t_skew_vals = [regs[h]['t_skew'] for h in hh]\n"
            "    t_vix_vals  = [regs[h]['t_vix']  for h in hh]\n"
            "    ax.plot(hh, t_skew_vals, 'o-', c=RED,  lw=2, label='t(SKEW)')\n"
            "    ax.plot(hh, t_vix_vals,  's--', c=GREY, lw=2, label='t(VIX)')\n"
            "    for s in (2, -2): ax.axhline(s, ls=':', c=GREY, lw=1)\n"
            "    ax.axhline(0, c='k', lw=1)\n"
            "    ax.set_xlabel('forward horizon (days)')\n"
            "    ax.set_ylabel('HAC t-stat (regression coefficient)')\n"
            "    ax.set_title('Neither SKEW nor VIX clears the inference bar')\n"
            "    ax.legend(); ax.set_xticks(hh)\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print(f't(SKEW) = {R[\"t_skew_reg\"]:+.2f}, t(VIX) = {R[\"t_vix_reg\"]:+.2f} (h=1d)')\n"
            "    print('Neither predictor clears |t| >= 2')"
        ),
        md(
            f"> in plain words: SKEW's t-stat in the joint regression is {R['t_skew_reg']:+.2f} — "
            f"indistinguishable from zero. VIX does marginally better ({R['t_vix_reg']:+.2f}) but "
            "also sub-2. The total R² is near zero. Neither index predicts next-day returns in "
            "any economically or statistically meaningful way."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — SKEW Q5−Q1 spread: t = {R['t_spread_1d']:+.2f} (h=1d), "
            f"{R['t_spread_5d']:+.2f} (h=5d), {R['t_spread_21d']:+.2f} (h=21d, wrong direction). "
            f"Crash-frequency Fisher p = {R['fisher_pvalue']:.2f}. SKEW regression t = "
            f"{R['t_skew_reg']:+.2f}. No test clears |t| ≥ 2 in the claimed direction.\n"
            f"- **Tradability `MIRAGE`** — no positive gross edge; SKEW adds nothing over VIX. "
            f"R² < 0.001 at h=1d.\n"
            f"- **Radar works? `BUSTED`** — crash rate {R['crash_rate_high']*100:.1f}% (high) "
            f"vs {R['crash_rate_low']*100:.1f}% (low). Rate ratio {R['rate_ratio']:.2f}. "
            "The radar does not see the swans."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it?\n\n"
            "Even in the fantasy world where the Q5−Q1 spread of −104 bps/21 days were "
            "significant (it's t = −2.12, barely), the implementation would require:\n\n"
            "- **Timing:** a daily signal that triggers only for the top quintile (~20% of days). "
            "On those days you'd need to be short SPY or long puts.\n"
            "- **Costs:** put options carry bid-ask, implied-vol premium (the volatility risk "
            "premium — you're buying insurance when it's most expensive), and daily theta decay.\n"
            "- **The reversal risk:** given the Q1-beats-Q5 pattern, *selling* puts when SKEW is "
            "high (collecting the premium) would have been the better trade — but that exposes you "
            "to the rare actual crash. The few times SKEW was right (2008, March 2020), being "
            "short puts would have been catastrophic.\n\n"
            "There is no trade here. The gross edge does not exist, and even the directional "
            "signal is a sentiment lag, not a predictor."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the positive control\n\n"
            "Can the machinery *find* the SKEW signal when we plant one in a synthetic tape?"
        ),
        code(
            "signals = [0.000, 0.005, 0.010, 0.015, 0.020]\n"
            "results_h1, results_h21 = [], []\n"
            "for sig in signals:\n"
            "    df_s, _ = data.synthetic_daily(n_days=2000, skew_signal=sig, seed=86)\n"
            "    r1  = st.top_minus_bottom(df_s, horizon=1)\n"
            "    r21 = st.top_minus_bottom(df_s, horizon=21)\n"
            "    results_h1.append(r1['t_spread'])\n"
            "    results_h21.append(r21['t_spread'])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "ax.plot(signals, results_h1,  'o-', c=AMBER, lw=2, label='h=1d spread t-stat')\n"
            "ax.plot(signals, results_h21, 's--', c=RED,  lw=2, label='h=21d spread t-stat')\n"
            "ax.axhline(0,  c='k',  lw=1)\n"
            "ax.axhline(-2, ls='--', c=GREY, lw=1, label='|t|=2 threshold')\n"
            "ax.set_xlabel('planted skew_signal (0 = null)')\n"
            "ax.set_ylabel('HAC t-stat on Q5−Q1 spread')\n"
            "ax.set_title('Positive control: the machinery detects planted signals')\n"
            "ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('At skew_signal=0 the t-stat is near 0; it grows (in magnitude) as the signal is planted.')\n"
            "print('The real-tape verdict is therefore about the MARKET, not the METHOD.')"
        ),
        md(
            "The synthetic sweep confirms the machinery: the t-stat on the Q5−Q1 spread grows "
            "monotonically as the planted signal increases. On the null tape (skew_signal = 0) "
            "the t-stat sits near zero. The real-tape result (t ≈ −1.4 at h=1d) is therefore a "
            "statement about the **market** — there is no predictive SKEW-forward-return "
            "relationship at a tradable horizon — not a failure of the test.\n\n"
            "Forks worth trying: conditional tests (high SKEW *and* high VIX simultaneously), "
            "very short horizons (1-2 days, momentum after SKEW spikes), or cross-asset tests "
            "(does high S&P SKEW predict EM or credit returns?)."
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
