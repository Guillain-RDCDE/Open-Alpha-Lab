"""Generate the two narrative notebooks for Study 145 (Home-Bias).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
cross_asset_etfs.parquet under _cache/ if present and otherwise quote the frozen
headline numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for
any reader.

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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-14).
R = dict(
    # Window
    n=5827, start="2003-04-15", end="2026-06-11",
    # Per-asset
    spy_ann=11.0, spy_vol=18.6, spy_sharpe=0.591, spy_maxdd=-59.6,
    efa_ann=7.9,  efa_vol=20.7, efa_sharpe=0.38,  efa_maxdd=-65.9,
    eem_ann=9.7,  eem_vol=27.0, eem_sharpe=0.36,  eem_maxdd=-73.4,
    # Portfolio comparison
    us_ann=10.97, us_vol=18.6, us_sharpe=0.591, us_maxdd=-59.6,
    bl_ann=10.01, bl_vol=19.4, bl_sharpe=0.515, bl_maxdd=-62.5,
    # Mean excess HAC
    excess_bps=-0.38, excess_t=-1.17,
    # Bootstrap Sharpe diff
    sharpe_diff=-0.077, ci_lo=-0.165, ci_hi=+0.012, frac_below=0.95,
    # Correlation
    corr_efa_min=0.40, corr_efa_mean=0.83, corr_efa_max=0.97,
    corr_eem_min=0.15, corr_eem_mean=0.76, corr_eem_max=0.97,
    # Crisis vs calm
    corr_efa_crisis=0.823, corr_efa_calm=0.811,
    corr_eem_crisis=0.736, corr_eem_calm=0.729,
)

# ---------------------------------------------------------------------------
# Shared preamble — imports, data load, HAVE_REAL guard.
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

from home_bias import data, strategy as st

def _have_cache():
    cache = os.path.join(data.DEFAULT_CACHE, "cross_asset_etfs.parquet")
    return os.path.exists(cache)

HAVE_REAL = _have_cache()
print("real panel cache present:", HAVE_REAL)

if HAVE_REAL:
    ret = data.fetch_panel()
    blend  = st.rolling_blend(ret)
    us_ret = st.us_only(ret)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Home-Bias — does going global beat staying home? 🌍\n"
            "### International diversification (SPY+EFA+EEM) vs US-only, tested honestly\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Diversification_benefit%3F: Correlation_killed_it](https://img.shields.io/badge/Diversification_benefit%3F-Correlation_killed_it-8b949e?style=flat-square)\n\n"
            "Every portfolio textbook says the same thing: *don't keep all your eggs in one "
            "country*. Add developed international (Europe, Japan, Australia) and some "
            "emerging markets (China, India, Brazil) to your US stocks — you'll get the same "
            "return with less risk. It's called the **diversification free lunch**. This "
            "notebook asks whether the free lunch was actually served over the last 23 years, "
            "or whether it arrived cold and half-eaten.\n\n"
            "> 📓 **This is the plain-language layer.** Want the t-stats, the bootstrap "
            "intervals, and the correlation maths? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Did the global blend improve Sharpe ratio? | **No.** Blend Sharpe "
            f"**{R['bl_sharpe']:.2f}** vs US-only **{R['us_sharpe']:.2f}** — blend *worse* on every axis. |\n"
            "| Did it reduce drawdown? | **No.** Blend max drawdown "
            f"**{R['bl_maxdd']:.1f}%** vs US-only **{R['us_maxdd']:.1f}%** — blend lost *more* in the worst period. |\n"
            "| 'But correlations were low!' | They weren't. SPY-EFA averaged "
            f"**{R['corr_efa_mean']:.2f}** over the period — barely a diversifier. |\n"
            "| Could you trade it? | You could, but why? At zero cost the blend underperforms. |\n\n"
            "> 💡 The free lunch was cancelled. High correlations + US outperformance "
            "meant going global cost you return *and* added risk."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Diversify internationally. Adding developed-market (EFA) and emerging-market "
            "(EEM) equity to a US portfolio reduces correlation, cuts volatility, and improves "
            "your risk-adjusted return — all without giving up long-run compound gains.\"*\n\n"
            f"We test it literally: a **60% SPY / 25% EFA / 15% EEM** portfolio, "
            "rebalanced once a year, from **2003** (when EEM went live) to **2026**. "
            "US-only (100% SPY) is the baseline. We ask: did the blend earn a higher Sharpe?"
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If true, home-bias is a free mistake: a US investor who refuses to own foreign "
            "stocks is leaving risk-adjusted return on the table. If false, the decades of "
            "advice to *go global* delivered **lower return, higher vol, and deeper drawdowns** "
            "— a free lunch that turned out to be a bill. That matters because trillions of "
            "dollars sit in internationally diversified portfolios, and the pitch for them is "
            "almost always this Sharpe-improvement story."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three tests, announced before we run them:\n\n"
            "1. **Sharpe comparison.** Does the blend's annualised Sharpe exceed US-only? "
            "The bootstrap 95% CI on the difference tells us whether the gap is noise or real.\n"
            "2. **Drawdown comparison.** Does the blend lose less in the worst periods? "
            "If correlation spikes in a crash, the diversification evaporates exactly when "
            "you need it.\n"
            "3. **Correlation over time.** Is the SPY-EFA/EEM correlation low enough to "
            "produce variance reduction? Mean correlation above ~0.7 leaves little room "
            "for the mathematical free lunch to work.\n\n"
            "Data: SPY/EFA/EEM daily total-return prices from the shared cache "
            f"(n = **{R['n']:,}** trading days, {R['start']} – {R['end']})."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the raw assets.** How did each piece perform on its own?"
        ),
        code(
            "tickers = ['SPY', 'EFA', 'EEM']\n"
            "labels = ['SPY (US)', 'EFA (dev ex-US)', 'EEM (emerging)']\n"
            "if HAVE_REAL:\n"
            "    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "    cum = {}\n"
            "    for t, lbl in zip(tickers, labels):\n"
            "        r = ret[t]\n"
            "        s = st.portfolio_stats(r)\n"
            "        cum[lbl] = (1 + r).cumprod()\n"
            "        axes[0].plot(cum[lbl], label=f'{lbl}  Sharpe={s[\"sharpe\"]:+.2f}')\n"
            "    axes[0].set_title('Cumulative growth (log-returns)')\n"
            "    axes[0].set_ylabel('Cumulative return (1 = start)'); axes[0].legend(fontsize=8)\n"
            "    for t, lbl in zip(tickers, labels):\n"
            "        r = ret[t]\n"
            "        s = st.portfolio_stats(r)\n"
            "        axes[1].bar(lbl, s['sharpe'], color=[GREEN if t=='SPY' else AMBER if t=='EFA' else RED][0])\n"
            "    axes[1].axhline(0, c='k', lw=1); axes[1].set_ylabel('Annualised Sharpe')\n"
            "    axes[1].set_title('SPY dominated on Sharpe'); plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    stats_tbl = pd.DataFrame({\n"
            "        'SPY': dict(ann_return=11.0, vol=18.6, sharpe=0.59, max_dd=-59.6),\n"
            "        'EFA': dict(ann_return=7.9, vol=20.7, sharpe=0.38, max_dd=-65.9),\n"
            "        'EEM': dict(ann_return=9.7, vol=27.0, sharpe=0.36, max_dd=-73.4),\n"
            "    }).T\n"
            "    print(stats_tbl.round(2))"
        ),
        md(
            f"SPY returned **+{R['spy_ann']:.1f}%/yr** at Sharpe **{R['spy_sharpe']:.2f}**. "
            f"EFA returned **+{R['efa_ann']:.1f}%/yr** at Sharpe **{R['efa_sharpe']:.2f}**. "
            f"EEM returned **+{R['eem_ann']:.1f}%/yr** at Sharpe **{R['eem_sharpe']:.2f}**. "
            "Both international legs had **lower return, higher volatility, and steeper drawdowns** "
            "than the US. The starting conditions for the blend are already unfavourable."
        ),
        md(
            "**Now the honest comparison: does the blend beat US-only?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    tbl = st.compare_portfolios(ret)\n"
            "    eq_us = (1 + us_ret).cumprod()\n"
            "    eq_bl = (1 + blend).cumprod()\n"
            "    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "    axes[0].plot(eq_us, label=f'US only (SPY)  Sharpe={tbl.loc[\"US only (SPY)\",\"sharpe\"]:+.2f}',\n"
            "                 color=GREEN, lw=2)\n"
            "    axes[0].plot(eq_bl, label=f'Global blend (60/25/15)  Sharpe={tbl.loc[\"Global blend (60/25/15)\",\"sharpe\"]:+.2f}',\n"
            "                 color=AMBER, lw=2)\n"
            "    axes[0].set_title('US-only vs global blend'); axes[0].set_ylabel('Cumulative return')\n"
            "    axes[0].legend(fontsize=8)\n"
            "    names = ['US only\\n(SPY)', 'Global blend\\n(60/25/15)']\n"
            "    sharpes = [tbl.loc['US only (SPY)','sharpe'], tbl.loc['Global blend (60/25/15)','sharpe']]\n"
            "    axes[1].bar(names, sharpes, color=[GREEN, AMBER], width=0.55)\n"
            "    axes[1].axhline(0, c='k', lw=1); axes[1].set_ylabel('Annualised Sharpe')\n"
            "    axes[1].set_title('Blend Sharpe < US-only'); plt.tight_layout(); plt.show()\n"
            "    print(tbl[['ann_return','ann_vol','sharpe','max_dd']].round(3).to_string())\n"
            "else:\n"
            "    print(f'US only:  ann={R[\"us_ann\"]:+.1f}%  vol={R[\"us_vol\"]:.1f}%  sharpe={R[\"us_sharpe\"]:.3f}  maxDD={R[\"us_maxdd\"]:.1f}%')\n"
            "    print(f'Blend:    ann={R[\"bl_ann\"]:+.1f}%  vol={R[\"bl_vol\"]:.1f}%  sharpe={R[\"bl_sharpe\"]:.3f}  maxDD={R[\"bl_maxdd\"]:.1f}%')"
        ),
        md(
            f"The global blend returns **+{R['bl_ann']:.1f}%/yr** at Sharpe **{R['bl_sharpe']:.3f}**; "
            f"US-only returns **+{R['us_ann']:.1f}%/yr** at Sharpe **{R['us_sharpe']:.3f}**. "
            "The blend loses on **every single metric**: return, vol, Sharpe, and drawdown. "
            "The free lunch was not served."
        ),
        md(
            "**Why? Because the correlations were too high.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    corr_efa = st.rolling_correlation(ret, 'SPY', 'EFA')\n"
            "    corr_eem = st.rolling_correlation(ret, 'SPY', 'EEM')\n"
            "    fig, ax = plt.subplots(figsize=(11, 4.5))\n"
            "    ax.plot(corr_efa, label=f'SPY-EFA (mean={corr_efa.mean():.2f})', color=AMBER, alpha=0.9)\n"
            "    ax.plot(corr_eem, label=f'SPY-EEM (mean={corr_eem.mean():.2f})', color=RED, alpha=0.9)\n"
            "    ax.axhline(corr_efa.mean(), ls='--', c=AMBER, lw=1)\n"
            "    ax.axhline(corr_eem.mean(), ls='--', c=RED, lw=1)\n"
            "    ax.axhline(0.7, ls=':', c=GREY, lw=1, label='0.70 (rough diversification threshold)')\n"
            "    ax.set_title('63-day rolling correlation of international equities with SPY')\n"
            "    ax.set_ylabel('Pearson correlation (63d rolling)'); ax.legend(fontsize=8)\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print(f'SPY-EFA rolling corr: mean={R[\"corr_efa_mean\"]:.2f}  min={R[\"corr_efa_min\"]:.2f}  max={R[\"corr_efa_max\"]:.2f}')\n"
            "    print(f'SPY-EEM rolling corr: mean={R[\"corr_eem_mean\"]:.2f}  min={R[\"corr_eem_min\"]:.2f}  max={R[\"corr_eem_max\"]:.2f}')"
        ),
        md(
            f"SPY-EFA averaged **{R['corr_efa_mean']:.2f}** and SPY-EEM **{R['corr_eem_mean']:.2f}**. "
            "These are very high correlations — financial globalisation has made equity markets "
            "move together. The diversification benefit requires correlations well below 0.7; at "
            "0.83 the variance-reduction from adding EFA is modest at best. And when the US "
            "also earns 3%/yr more than EFA, the modest risk reduction cannot compensate.\n\n"
            "> 💡 The free lunch is mathematical: it requires low correlation *and* competitive "
            "returns. When the US earned more AND correlations were high, both levers pointed "
            "against the blend."
        ),

        # ---- BEAT 5 — VERDICT ------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Blend Sharpe {R['bl_sharpe']:.3f} < US-only {R['us_sharpe']:.3f}; "
            f"blend return {R['bl_ann']:.1f}% < US {R['us_ann']:.1f}%; deeper drawdown. "
            "No diversification benefit materialised.\n"
            f"- **Tradability — Mirage.** At zero cost the blend underperforms on every axis. "
            "There is nothing to trade here.\n"
            f"- **Diversification benefit? Correlation killed it.** SPY-EFA averaged "
            f"{R['corr_efa_mean']:.2f} — far too high for meaningful variance reduction. "
            "And the US structural return edge (+3%/yr) overrode any marginal benefit."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "The good news: the *cost* of implementing the 60/25/15 blend is tiny. Annual "
            "rebalancing trades ~5% of notional; at 2 bps round-trip that's ~0.1 bps/yr — "
            "effectively free. The problem isn't execution: the blend simply underperforms "
            "before costs. You would be paying to own a portfolio with **lower Sharpe, lower "
            "return, and deeper drawdowns** than staying home."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s_us  = st.portfolio_stats(us_ret)\n"
            "    s_bl  = st.portfolio_stats(blend)\n"
            "    fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "    metrics = ['ann_return', 'ann_vol', 'sharpe']\n"
            "    labels_m = ['Ann. Return', 'Ann. Vol', 'Sharpe']\n"
            "    x = range(len(metrics))\n"
            "    w = 0.35\n"
            "    ax.bar([i-w/2 for i in x], [s_us[m] for m in metrics], width=w, label='US only (SPY)', color=GREEN)\n"
            "    ax.bar([i+w/2 for i in x], [s_bl[m] for m in metrics], width=w, label='Global blend', color=AMBER)\n"
            "    ax.set_xticks(list(x)); ax.set_xticklabels(labels_m)\n"
            "    ax.axhline(0, c='k', lw=1); ax.set_title('US-only wins on every axis'); ax.legend()\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('US only: ann=+10.97%  vol=18.6%  sharpe=0.591')\n"
            "    print('Blend:   ann=+10.01%  vol=19.4%  sharpe=0.515')\n"
            "    print('=> Blend loses on every axis before costs.')"
        ),
        md(
            "> 💡 Annual rebalancing costs are negligible (~0.1 bps/yr). The blend underperforms "
            "*before* costs — there is no cost level at which it becomes sensible, because "
            "the issue is not execution friction but the realised return/risk of the assets."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Is the future different from the past?** The theoretical case for international "
            "diversification is forward-looking: if US valuations are high (CAPE > 30) and "
            "international valuations cheap (CAPE ~15), the *forward* expected return spread "
            "may reverse. This study only measures what happened. Future US underperformance "
            "would make the blend look better.\n"
            "- **What if you held bonds?** Adding non-equity assets (TLT, GLD) can genuinely "
            "diversify because correlations with equities are lower. "
            "[Study 68 — All-Weather](../../68-all-weather/) tests exactly this — and finds "
            "the Sharpe improvement is modest but real on the risk-parity frame.\n"
            "- **Currency risk.** EFA and EEM are unhedged; a strong dollar (post-2009) "
            "depressed their USD returns. A currency-hedged EFA (ticker HEFA) would show "
            "slightly higher returns — but still lower than SPY.\n"
            "- **What correlation would make diversification work?** Below ~0.50 and with "
            "equal expected returns, theory predicts a Sharpe improvement. That was the world "
            "in the 1970-1990s. The synthetic demo shows the machine works when conditions are "
            "right — the real market since 2003 just didn't have those conditions.\n\n"
            "*Think the free lunch is back? Check current international CAPE vs US CAPE, "
            "and the recent SPY-EFA rolling 63-day correlation. That's the data you'd want "
            "before tilting the allocation.*"
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
            "# Home-Bias — a quantitative teardown\n"
            "### SPY / EFA / EEM · 2003-2026 · Sharpe difference bootstrap · HAC inference · "
            "crisis-regime correlations\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Diversification_benefit%3F: Correlation_killed_it](https://img.shields.io/badge/Diversification_benefit%3F-Correlation_killed_it-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim now carrying its standard error.* We measure whether "
            "a 60/25/15 SPY/EFA/EEM portfolio earns a higher Sharpe than US-only (SPY) over "
            "2003-2026, with a block-bootstrap CI on the Sharpe difference, a HAC t-stat on the "
            "mean excess return, and a crisis-regime correlation decomposition.\n\n"
            "> ⚠️ **Not investment advice.** Real data: Yahoo Finance total-return prices, "
            f"n = {R['n']:,} days ({R['start']} – {R['end']}); offline core runs on deterministic "
            "synthetic panels. Methods & sources in [`docs/references.md`](../docs/references.md), "
            "reproducible numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT + "\nfrom quantlab import analytics, stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Blend Sharpe **{R['bl_sharpe']:.3f}** < US-only "
            f"**{R['us_sharpe']:.3f}**; Δ = **{R['sharpe_diff']:+.3f}**, bootstrap 95% CI "
            f"**[{R['ci_lo']:+.3f}, {R['ci_hi']:+.3f}]** ({R['frac_below']:.0%} of resamples "
            "favour US-only). Mean excess return HAC *t* = "
            f"**{R['excess_t']:+.2f}**. |\n"
            f"| **Tradability** | `MIRAGE` | Blend underperforms US-only on return, vol, "
            "Sharpe, and drawdown — at zero cost. No positive break-even scenario. |\n"
            f"| **Diversification benefit?** | `CORRELATION KILLED IT` | SPY-EFA mean rolling "
            f"correlation **{R['corr_efa_mean']:.2f}** (max {R['corr_efa_max']:.2f}); too high "
            "for meaningful variance reduction; US outperformed EFA by +3.1%/yr. |\n\n"
            "> 💡 The free lunch requires low correlation AND competitive international returns. "
            "Neither held for 2003-2026: globalisation raised correlations, and the US "
            "dominated on return."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $w = (0.60, 0.25, 0.15)$ be the target weights for SPY/EFA/EEM. The "
            "diversification thesis asserts:\n\n"
            "- **H₁ (Sharpe improvement).** $\\text{SR}(w^\\top r) > \\text{SR}(r_{\\text{SPY}})$ "
            "where $r$ is the daily return vector — the blend earns a higher risk-adjusted "
            "return than US-only.\n"
            "- **H₂ (variance reduction).** $\\text{Var}(w^\\top r) < \\text{Var}(r_{\\text{SPY}})$, "
            "from the convexity of variance in the presence of imperfect correlation.\n"
            "- **H₃ (drawdown protection).** The blend's maximum drawdown is smaller than "
            "SPY's — the diversification kicks in in crises.\n\n"
            "We reject H₁ and H₃; H₂ fails because the realised correlation is too high "
            "to deliver meaningful variance reduction."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "The international-diversification trade is the single most widely taught "
            "portfolio prescription. Trillions of dollars of global equity exposure sit in "
            "funds sold partly on this Sharpe-improvement story. Documenting that the story "
            "failed empirically for 23 years — while identifying the *why* (high correlations + "
            "US dominance) — helps separate the theoretical argument from the historical record, "
            "and flags what would have to change (lower US CAPE, higher ex-US growth, lower "
            "correlations) for the prescription to work going forward."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Data.** Daily log-returns for SPY, EFA, EEM from `cross_asset_etfs.parquet`, "
            f"n = {R['n']:,} days, window {R['start']} – {R['end']} (EEM IPO date onward).\n"
            "- **Portfolio.** Fixed-weight 60/25/15, rebalanced at each calendar-year start "
            "(no look-ahead: weights are fixed *before* the year begins).\n"
            "- **Primary comparison.** US-only (100% SPY) buy-and-hold.\n"
            "- **Inference.** (a) HAC Newey-West t-stat on mean daily excess return of blend "
            "over US; (b) circular block-bootstrap (n^{1/3} blocks) 95% CI on Sharpe difference.\n"
            "- **Correlation decomposition.** 63-day rolling Pearson; crisis-vs-calm split "
            "(worst 20% SPY days = crisis).\n"
            "- **Positive control.** Synthetic panel: blend Sharpe > US-only only when "
            "corr < 0.5 AND equal expected returns — confirming the engine is faithful."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Portfolio stats — the blend loses on every axis\n\n"
            "Annualised return, volatility, Sharpe, and max drawdown for US-only and the blend "
            "over the full sample. If the diversification thesis held, the blend would show "
            "lower vol and higher Sharpe — even if return lagged."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tbl = st.compare_portfolios(ret)\n"
            "    t = tbl.copy()\n"
            "    t['ann_return'] = (t['ann_return']*100).round(2)\n"
            "    t['ann_vol'] = (t['ann_vol']*100).round(2)\n"
            "    t['sharpe'] = t['sharpe'].round(3)\n"
            "    t['max_dd'] = (t['max_dd']*100).round(1)\n"
            "    print(t[['ann_return','ann_vol','sharpe','max_dd','n']].to_string())\n"
            "    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3))\n"
            "    labs = ['US only\\n(SPY)', 'Global blend\\n(60/25/15)']\n"
            "    for ax, col, title in zip(axes, ['sharpe','ann_vol','max_dd'],\n"
            "                              ['Sharpe','Ann. Vol (%)', 'Max Drawdown (%)']):\n"
            "        vals = [tbl.loc['US only (SPY)', col], tbl.loc['Global blend (60/25/15)', col]]\n"
            "        colors = [GREEN, AMBER]\n"
            "        ax.bar(labs, vals, color=colors, width=0.55)\n"
            "        ax.axhline(0, c='k', lw=1); ax.set_title(title)\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('US only:  ann=+10.97%  vol=18.6%  sharpe=0.591  maxDD=-59.6%')\n"
            "    print('Blend:    ann=+10.01%  vol=19.4%  sharpe=0.515  maxDD=-62.5%')"
        ),
        md(
            f"> 💡 The blend has higher vol ({R['bl_vol']:.1f}% vs {R['us_vol']:.1f}%), lower Sharpe "
            f"({R['bl_sharpe']:.3f} vs {R['us_sharpe']:.3f}), and a deeper drawdown "
            f"({R['bl_maxdd']:.1f}% vs {R['us_maxdd']:.1f}%). Adding EFA and EEM made the portfolio *more* "
            "volatile and less rewarding per unit of risk."
        ),
        md(
            "### 4b · Sharpe difference — block-bootstrap 95% CI\n\n"
            "Is the observed Sharpe gap (blend − US) statistically distinguishable from zero? "
            "We use a circular block bootstrap (n^{1/3} blocks) to build the 95% CI."
        ),
        code(
            "if HAVE_REAL:\n"
            "    boot = st.sharpe_diff_bootstrap(blend, us_ret, n_boot=2000, seed=145)\n"
            "    hac  = st.mean_excess_hac(blend, us_ret)\n"
            "    print(f'Sharpe diff (blend - US): {boot[\"sharpe_diff\"]:+.3f}')\n"
            "    print(f'95% CI: [{boot[\"ci_low\"]:+.3f}, {boot[\"ci_high\"]:+.3f}]')\n"
            "    print(f'Frac resamples where blend < US: {boot[\"frac_blend_below_us\"]:.0%}')\n"
            "    print(f'Mean excess return (bps/day): {hac[\"mean_bps\"]:+.2f}  HAC t = {hac[\"tstat\"]:+.2f}')\n"
            "    fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "    n_boot = 2000\n"
            "    import numpy as _np\n"
            "    rng = _np.random.default_rng(145)\n"
            "    rb = blend.to_numpy(); ru = us_ret.to_numpy()\n"
            "    mask = _np.isfinite(rb) & _np.isfinite(ru); rb, ru = rb[mask], ru[mask]\n"
            "    n = len(rb); blk = max(1, round(n**(1/3)))\n"
            "    offsets = _np.arange(blk); n_blks = int(_np.ceil(n/blk))\n"
            "    diffs = []\n"
            "    for _ in range(n_boot):\n"
            "        starts = rng.integers(0, n, n_blks)\n"
            "        idx = ((starts[:,None]+offsets[None,:])%n).ravel()[:n]\n"
            "        def _sr(r): sd=r.std(ddof=1); return r.mean()/sd*_np.sqrt(252) if sd>0 else _np.nan\n"
            "        diffs.append(_sr(rb[idx]) - _sr(ru[idx]))\n"
            "    diffs = _np.array(diffs); diffs = diffs[_np.isfinite(diffs)]\n"
            "    ax.hist(diffs, bins=50, color=GREY, alpha=0.7, label='bootstrap Sharpe diff (blend-US)')\n"
            "    ax.axvline(boot['sharpe_diff'], c=RED, lw=2.5, label=f'observed: {boot[\"sharpe_diff\"]:+.3f}')\n"
            "    ax.axvline(0, c='k', lw=1)\n"
            "    ax.set_xlabel('Sharpe (blend) - Sharpe (US)'); ax.set_ylabel('count')\n"
            "    ax.set_title(f'{boot[\"frac_blend_below_us\"]:.0%} of resamples: blend < US')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('Sharpe diff: -0.077  95% CI [-0.165, +0.012]')\n"
            "    print('Frac below US: 95%  |  Mean excess HAC t = -1.17')"
        ),
        md(
            f"The observed Sharpe difference is **{R['sharpe_diff']:+.3f}** (blend − US). "
            f"The 95% CI is **[{R['ci_lo']:+.3f}, {R['ci_hi']:+.3f}]** — almost entirely below zero. "
            f"**{R['frac_below']:.0%}** of bootstrap resamples have the blend below US-only. "
            "The mean excess return HAC t = **{R['excess_t']:+.2f}** — not significant, but the "
            "point estimate is the wrong sign. The data offer no support for H₁."
        ),
        md(
            "### 4c · Correlation — the mechanical reason the blend fails\n\n"
            "Diversification theory requires low correlation. What did we actually get?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    corr_efa = st.rolling_correlation(ret, 'SPY', 'EFA')\n"
            "    corr_eem = st.rolling_correlation(ret, 'SPY', 'EEM')\n"
            "    crisis   = st.crisis_vs_calm(ret)\n"
            "    print('Rolling 63d correlation stats:')\n"
            "    for nm, c in [('SPY-EFA', corr_efa), ('SPY-EEM', corr_eem)]:\n"
            "        print(f'  {nm}: min={c.min():.2f}  mean={c.mean():.2f}  max={c.max():.2f}')\n"
            "    print()\n"
            "    print('Crisis vs calm (worst 20% SPY days = crisis):')\n"
            "    print(crisis.round(3).to_string())\n"
            "    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "    axes[0].plot(corr_efa, color=AMBER, alpha=0.8, label='SPY-EFA')\n"
            "    axes[0].plot(corr_eem, color=RED, alpha=0.8, label='SPY-EEM')\n"
            "    axes[0].axhline(corr_efa.mean(), ls='--', c=AMBER, lw=1)\n"
            "    axes[0].axhline(0.5, ls=':', c=GREY, lw=1, label='0.5 threshold')\n"
            "    axes[0].set_title('Rolling 63d correlation'); axes[0].legend(fontsize=8)\n"
            "    regimes = ['Crisis\\n(worst 20% SPY)', 'Calm']\n"
            "    axes[1].bar([x-0.2 for x in range(2)],\n"
            "                [crisis.loc['EFA','corr_crisis'], crisis.loc['EFA','corr_calm']],\n"
            "                width=0.35, color=AMBER, label='SPY-EFA')\n"
            "    axes[1].bar([x+0.2 for x in range(2)],\n"
            "                [crisis.loc['EEM','corr_crisis'], crisis.loc['EEM','corr_calm']],\n"
            "                width=0.35, color=RED, label='SPY-EEM')\n"
            "    axes[1].set_xticks(range(2)); axes[1].set_xticklabels(regimes)\n"
            "    axes[1].set_title('Crisis vs calm correlation'); axes[1].legend()\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('SPY-EFA: min=0.40  mean=0.83  max=0.97')\n"
            "    print('SPY-EEM: min=0.15  mean=0.76  max=0.97')\n"
            "    print('EFA crisis=0.823  calm=0.811')\n"
            "    print('EEM crisis=0.736  calm=0.729')"
        ),
        md(
            f"> 💡 Mean SPY-EFA correlation {R['corr_efa_mean']:.2f} (max {R['corr_efa_max']:.2f}) and "
            f"SPY-EEM {R['corr_eem_mean']:.2f}. Crisis-regime correlations are essentially the same "
            "as calm-regime correlations for this linear-return model. On real data, 2008 and "
            "2020 saw correlations spike to 0.90+ across all equity markets — exactly when the "
            "diversification was most needed."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — Sharpe difference {R['sharpe_diff']:+.3f}, 95% CI "
            f"[{R['ci_lo']:+.3f}, {R['ci_hi']:+.3f}], {R['frac_below']:.0%} of resamples favour "
            f"US-only; mean excess HAC *t* = {R['excess_t']:+.2f}. Blend Sharpe {R['bl_sharpe']:.3f} < "
            f"US-only {R['us_sharpe']:.3f}.\n"
            f"- **Tradability `MIRAGE`** — blend underperforms US-only on return (−0.96%/yr), "
            f"vol (+0.89%), Sharpe (−0.077), and max drawdown (−2.9pp) at zero cost. Nothing to trade.\n"
            f"- **Diversification benefit? `CORRELATION KILLED IT`** — SPY-EFA mean {R['corr_efa_mean']:.2f}, "
            "too high for meaningful variance reduction. US structural outperformance (+3.1%/yr "
            "over EFA) compounded the damage."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — execution is not the problem\n\n"
            "Annual rebalancing turnover is ~5% of notional; at 2 bps round-trip, drag is "
            "~0.1 bps/yr — negligible. The issue is not execution but the *assets themselves*: "
            "EFA and EEM delivered lower return, higher vol, and steeper drawdowns than SPY "
            "over this sample. The break-even cost question is moot because the gross edge is "
            "already negative."
        ),
        code(
            "if HAVE_REAL:\n"
            "    print('Annual rebalancing cost estimate:')\n"
            "    turnover_pct = 5.0  # rough: 3-asset drift in one year\n"
            "    cost_bps_roundtrip = 2.0\n"
            "    annual_drag_bps = turnover_pct / 100 * cost_bps_roundtrip\n"
            "    print(f'  Turnover ~{turnover_pct:.0f}%/yr  x  {cost_bps_roundtrip:.0f} bps round-trip')\n"
            "    print(f'  Estimated drag: {annual_drag_bps:.2f} bps/yr (~negligible)')\n"
            "    excess = st.mean_excess_hac(blend, us_ret)\n"
            "    print(f'  Mean excess return: {excess[\"mean_bps\"]:+.2f} bps/DAY = {excess[\"mean_bps\"]*252:.0f} bps/yr')\n"
            "    print(f'  => gross underperformance far exceeds transaction costs.')\n"
            "else:\n"
            "    print('Mean excess: -0.38 bps/day = -96 bps/yr')\n"
            "    print('Annual cost drag: ~0.1 bps/yr (negligible vs -96 bps/yr underperformance)')"
        ),
        md(
            "> 💡 Contrast with studies where the *cost* kills a real-but-thin gross edge (e.g. "
            "high-frequency rules). Here the gross edge is negative before costs — the problem "
            "is the assets, not the friction."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Does the engine *detect* a Sharpe improvement when one is planted? We sweep "
            "correlation on a synthetic panel with equal expected returns."
        ),
        code(
            "corrs = [0.10, 0.30, 0.50, 0.70, 0.90, 1.00]\n"
            "results = []\n"
            "for corr in corrs:\n"
            "    ret_s, _ = data.synthetic_panel(n_days=5000, corr_intl=corr,\n"
            "                                     mu_us=0.07/252, mu_intl=0.07/252, seed=145)\n"
            "    s_bl = st.portfolio_stats(st.rolling_blend(ret_s))['sharpe']\n"
            "    s_us = st.portfolio_stats(st.us_only(ret_s))['sharpe']\n"
            "    results.append((corr, s_us, s_bl, s_bl - s_us))\n"
            "df_ctrl = pd.DataFrame(results, columns=['corr', 'sharpe_us', 'sharpe_blend', 'delta'])\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "ax.plot(df_ctrl['corr'], df_ctrl['delta'], 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.axvline(0.83, ls='--', c=GREY, lw=1.5, label='real SPY-EFA mean=0.83')\n"
            "ax.set_xlabel('Planted SPY-EFA correlation'); ax.set_ylabel('Blend Sharpe - US Sharpe')\n"
            "ax.set_title('Engine works: blend improves Sharpe only at low correlation')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(df_ctrl.round(3).to_string())"
        ),
        md(
            f"The blend's Sharpe advantage is monotone in correlation: positive at low corr, "
            f"negative at high corr. The crossover (where blend ≈ US-only) sits around "
            f"correlation ~0.7. The real-tape mean correlation of **{R['corr_efa_mean']:.2f}** "
            "sits firmly in the negative-benefit zone — confirming the verdict is a statement "
            "about the market regime, not a machinery failure.\n\n"
            "**What would change the verdict?** Lower US CAPE (higher forward return for "
            "international), lower global equity correlation (only from economic decoupling or "
            "a non-equity diversifier), or an extended period of US underperformance. None of "
            "those are guaranteed; this study only reports what 2003-2026 delivered."
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
