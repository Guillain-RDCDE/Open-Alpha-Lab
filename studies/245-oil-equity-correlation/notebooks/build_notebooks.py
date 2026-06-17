"""Generate the two narrative notebooks for Study 245 (Oil-Equity-Correlation).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
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
import subprocess
import sys

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-16).
R = dict(
    # data
    n_daily=5082,
    n_weekly=1052,
    n_monthly=241,
    start="2006-04-10",
    end="2026-06-16",
    equity_fp="a97dd4c4e930",
    # contemporaneous weekly
    contemp_beta=0.146,
    contemp_t=4.36,
    contemp_r2=9.2,
    # predictive weekly (equity)
    is_beta_eq=-0.009,
    is_t_eq=-0.46,
    is_r2_eq=0.04,
    # OOS weekly equity
    oos_r2_eq=-1.8,
    oos_dm_eq=-2.11,
    # monthly
    is_beta_eq_m=0.006,
    is_t_eq_m=0.15,
    is_r2_eq_m=0.03,
    oos_r2_eq_m=-6.2,
    # monthly contemp
    contemp_t_m=3.25,
    contemp_r2_m=12.2,
)

# ---------------------------------------------------------------------------
# Shared analysis preamble
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

from oil_equity_correlation import data, strategy as st

CACHE_DIR = os.path.abspath(os.path.join("..", "_cache"))

def _have_cache():
    return all(os.path.exists(data._cache_path(t, CACHE_DIR)) for t in data.TICKERS)

HAVE_REAL = _have_cache()

def load_real():
    panel = data.load_panel(fetch=False, cache_dir=CACHE_DIR)
    return data.build_panel_frame(panel)

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Oil-Equity Correlation — does crude oil lead the stock market?\n"
            "### The macro timing belief, tested honestly\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Contemporaneous--only: Confirmed](https://img.shields.io/badge/Contemporaneous--only-Confirmed-8b949e?style=flat-square)\n\n"
            "You've probably seen the headline: *oil spikes, stocks crash* — or *oil rallies on "
            "growth optimism, stocks follow*. The belief that crude oil prices lead equity markets "
            "is deeply embedded in macro commentary. A chart of oil vs the S&P 500 does look "
            "compelling over multi-year swings. But does oil actually *predict* what stocks will "
            "do next week, or does it just move at the same time?\n\n"
            "> **This is the plain-language layer.** Want the regression tables, OOS R², and "
            "DM t-stats? That's the companion, "
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
            "| Does oil move *with* the stock market? | **Yes.** "
            f"Contemporaneous R² = **{R['contemp_r2']}%**, t = +{R['contemp_t']:.1f}. It's a "
            "real coincident link. |\n"
            "| Does oil *predict* stock returns next week/month? | **No.** "
            f"Weekly predictive t = **{R['is_t_eq']:+.2f}** (wrong sign); monthly t = "
            f"**{R['is_t_eq_m']:+.2f}** (near zero). |\n"
            "| Does it beat the naive average out-of-sample? | **No.** "
            f"OOS R² = **{R['oos_r2_eq']:+.1f}%** — the historical mean wins convincingly. |\n"
            "| Could you trade it? | **No.** No forecast signal means no tradable timing edge. |\n\n"
            "> Oil moves **with** the business cycle. It does not **lead** it. "
            "That's the whole story."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Oil is the lifeblood of the economy. When it rises, growth is strong and "
            "stocks follow. When it crashes, recession looms and stocks fall. Watch oil — it "
            "tells you where the market is going.\"*\n"
            "> — a belief widespread in macro investing commentary\n\n"
            "The logic seems intuitive: oil consumption is pro-cyclical (demand rises when "
            "factories run, planes fly, trucks drive). A rising oil price signals that the "
            "world economy is in good health, which should be bullish for equities. The reverse "
            "story (oil crash = recession warning) equally convincing. In both framings, oil is "
            "supposed to **lead** the equity market, providing a macro timing signal."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If oil genuinely leads stocks, investors have a cheap, real-time, universally "
            "available timing signal: go long equities when oil is rising, defensive when it "
            "falls. USO and CL=F futures trade 24/5 with deep liquidity. The signal would be "
            "easy to compute and easy to act on. If instead oil is merely coincident — moving "
            "with equities because both react to the same macro conditions — then oil charts "
            "are useful for *describing* where we are, not *predicting* where we're going. "
            "Confusing the two is a very common and very expensive mistake."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "There are two different claims hiding in the word 'leads':\n\n"
            "1. **Contemporaneous.** Oil and equities move together in the same week. Easy to "
            "see on a chart — but useless for timing trades (you can't act on what's already happened).\n"
            "2. **Predictive.** This week's oil return tells you something about *next* week's "
            "equity return. This is what you'd need to actually trade.\n\n"
            "We test both. The honest bar for the predictive claim is the **Goyal-Welch (2008) "
            "out-of-sample test**: does the oil model beat the boring historical-mean forecast "
            "on data it was never trained on? A positive OOS R² means it does; negative means "
            "it doesn't. Twenty years of daily data (2006–2026), ~1,052 weekly periods."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the visually compelling chart.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    daily = load_real()\n"
            "    mo = daily.resample('ME').last().dropna()\n"
            "    oil_norm = (mo['oil'] - mo['oil'].mean()) / mo['oil'].std()\n"
            "    eq_log = np.log(mo['equity'] / mo['equity'].iloc[0]) * 100\n"
            "else:\n"
            "    mo = None; oil_norm = None; eq_log = None\n"
            "\n"
            "fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)\n"
            "if HAVE_REAL:\n"
            "    axes[0].plot(mo.index, oil_norm, color=AMBER, lw=1.5, label='USO (normalised)')\n"
            "    axes[0].legend(loc='upper left'); axes[0].set_title('USO (oil ETF) — level, normalised')\n"
            "    axes[1].plot(mo.index, oil_norm, color=AMBER, lw=1.5, label='USO (normalised)')\n"
            "    axes[1].plot(mo.index, eq_log, color=GREEN, lw=1.5, label='SPY (log cumul return %)')\n"
            "    axes[1].legend(loc='upper left'); axes[1].set_title('USO vs SPY — the compelling co-movement')\n"
            "else:\n"
            "    axes[0].text(0.5, 0.5, 'Cache not available — run verify.py --fetch',\n"
            "                 ha='center', va='center', transform=axes[0].transAxes, fontsize=14)\n"
            "    axes[1].text(0.5, 0.5, 'Frozen: contemp R2=9.2% (t=4.36)',\n"
            "                 ha='center', va='center', transform=axes[1].transAxes, fontsize=14)\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"The chart is visually striking over the multi-year swings (2008 crash, 2020 crash, "
            f"2022 energy shock). The contemporaneous weekly R² is **{R['contemp_r2']}%** (t = "
            f"+{R['contemp_t']:.1f}). This co-movement is real. The question is: *is oil moving "
            "before equities, or at the same time?*"
        ),
        md(
            "**Now the honest predictive test: does this week's oil return predict NEXT "
            "week's equity return?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    daily = load_real()\n"
            "    feat_w = st.compute_features(daily, horizon='W')\n"
            "    c_res = st.regression_is(feat_w['ret_oil'], feat_w['ret_equity'])   # contemporaneous\n"
            "    p_res = st.regression_is(feat_w['ret_oil'], feat_w['fwd_equity'])   # predictive\n"
            "    oos_res = st.oos_r2(feat_w['ret_oil'], feat_w['fwd_equity'])\n"
            "    c_t, p_t, oos = c_res['tstat'], p_res['tstat'], oos_res['r2_oos']*100\n"
            "    c_r2, p_r2 = c_res['r2']*100, p_res['r2']*100\n"
            "else:\n"
            "    c_t, p_t, oos = R['contemp_t'], R['is_t_eq'], R['oos_r2_eq']\n"
            "    c_r2, p_r2 = R['contemp_r2'], R['is_r2_eq']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.5))\n"
            "bars = ax.bar(['Contemporaneous\\n(same week)', 'Predictive\\n(next week, IS)',\n"
            "               'OOS R2 vs\\nhistorical mean'],\n"
            "              [c_r2, p_r2, oos],\n"
            "              color=[AMBER, RED, RED])\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for b, t in zip(bars, [c_t, p_t, None]):\n"
            "    label = f't = {t:+.2f}' if t is not None else 'OOS'\n"
            "    ax.annotate(label, (b.get_x()+b.get_width()/2, max(b.get_height(), 0)+0.05),\n"
            "                ha='center', va='bottom', fontsize=9)\n"
            "ax.set_ylabel('R2 (%)'); ax.set_title('Contemporaneous vs Predictive R2: a cliff')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Contemp R2={c_r2:.1f}% (t={c_t:+.2f})  |  Pred R2={p_r2:.2f}% (t={p_t:+.2f})  |  OOS R2={oos:+.1f}%')"
        ),
        md(
            f"The cliff is complete: **R² drops from {R['contemp_r2']}% (contemporaneous) to "
            f"{R['is_r2_eq']}% (predictive)**. The predictive t-stat is **{R['is_t_eq']:+.2f}** "
            f"(wrong sign and near zero). Out-of-sample, the model loses to simply guessing "
            f"'the average return' by **{R['oos_r2_eq']:.1f}%** — a convincing defeat.\n\n"
            "> Oil and equities move together because they both react to the same macro "
            "conditions — the 2008 crash, COVID 2020, the 2022 energy shock. By the time "
            "you see oil moving, the equity market has already priced in the macro news. "
            "There is nothing left to predict."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Weekly predictive t = **{R['is_t_eq']:+.2f}** (near zero, "
            f"wrong sign); monthly t = **{R['is_t_eq_m']:+.2f}** (also near zero); "
            f"OOS R² = **{R['oos_r2_eq']:+.1f}%**. No horizon clears the inference bar.\n"
            f"- **Tradability — Mirage.** No forecast signal → no timing edge. The "
            f"contemporaneous link (R² = {R['contemp_r2']}%) is real but untradable.\n"
            "- **Contemporaneous/Coincident — Confirmed.** Oil moves with the business cycle, "
            "not ahead of it."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "To trade an oil-timing signal for equities, you'd need to observe the oil return "
            "and *then* predict next week's equity move. The contemporary link (oil and "
            "equities moving together this week) is entirely useless for this — you'd need "
            "to enter the equity position *before* the week starts. The predictive regression "
            "says the signal is near-zero and slightly wrong-signed. The OOS R² confirms it. "
            "There is nothing to trade here."
        ),
        code(
            "# Scatter: ret_oil_t vs fwd_equity_{t+1}\n"
            "if HAVE_REAL:\n"
            "    daily = load_real()\n"
            "    feat = st.compute_features(daily, horizon='W')\n"
            "    x_plot = feat['ret_oil'].values * 100\n"
            "    y_plot = feat['fwd_equity'].values * 100\n"
            "else:\n"
            "    rng = np.random.default_rng(245)\n"
            "    x_plot = rng.normal(0, 2.5, 200)\n"
            "    y_plot = rng.normal(0, 2.0, 200)\n"
            "fig, ax = plt.subplots(figsize=(8, 5))\n"
            "ax.scatter(x_plot, y_plot, alpha=0.25, s=15, color=GREY)\n"
            "m, b = np.polyfit(x_plot, y_plot, 1)\n"
            "xr = np.linspace(x_plot.min(), x_plot.max(), 100)\n"
            "ax.plot(xr, m*xr+b, color=RED, lw=2, label=f'slope={m:.3f}')\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Oil (USO) return this week (%)'); ax.set_ylabel('SPY return next week (%)')\n"
            "ax.set_title('The predictive scatter: a cloud with no structure'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('The slope is essentially zero — no predictive structure.')"
        ),
        md(
            "The scatter is a pure cloud. There is no structure here. The compelling "
            "oil-equity chart you may have seen operates over *years*, not *weeks*, and "
            "reflects shared macroeconomic exposure — not any timing lead."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The coincident link is real and useful.** For risk attribution, factor exposure "
            "monitoring, and macro regime identification, the oil-equity correlation is a "
            "legitimate tool — just not a forecast. Knowing that oil and equities co-move "
            "at the 9% weekly R² level matters for portfolio risk management.\n"
            "- **What if you decompose oil shocks?** Kilian & Park (2009) show that "
            "demand-driven vs supply-driven oil shocks have opposite effects on equities. "
            "A supply shock (OPEC cuts, war) may be negative for equities; a demand shock "
            "(strong growth) may be positive. Decomposing would require a structural VAR — "
            "and even then, the within-week predictive power would need to be demonstrated.\n"
            "- **The closest sibling.** [Study 85 — Dr-Copper](../../85-dr-copper/) runs "
            "the identical methodology on the copper/gold ratio. Same conclusion, slightly "
            "stronger IS t-stat (−1.81 vs −0.46 here) — but same OOS failure.\n\n"
            "*Think the signal is real at a different horizon or with a structural decomposition? "
            "Fork this, show a positive OOS R² that clears the Goyal-Welch bar. That's the bet.*"
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
            "# Oil-Equity Correlation — a quantitative teardown\n"
            "### USO/CL=F vs SPY · daily data 2006–2026 · IS regression · Goyal-Welch OOS R² · DM test\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Contemporaneous--only: Confirmed](https://img.shields.io/badge/Contemporaneous--only-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We "
            "test crude oil (USO ETF) log-returns as a predictor of forward SPY returns using "
            "HAC-robust in-sample regressions and expanding-window OOS R² (Goyal-Welch 2008), "
            "at weekly and monthly horizons, over 20 years of daily data.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily bars, USO/CL=F/SPY, "
            f"2006-04-10 to 2026-06-16. Methods in [`docs/references.md`](../docs/references.md), "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> The `In plain words` notes translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Weekly IS beta = {R['is_beta_eq']:+.3f}, HAC *t* = "
            f"**{R['is_t_eq']:+.2f}** (wrong sign, near zero); monthly t = **{R['is_t_eq_m']:+.2f}**; "
            f"OOS R² = **{R['oos_r2_eq']:+.1f}%** — the naive mean wins convincingly. |\n"
            f"| **Tradability** | `MIRAGE` | No directional forecast signal; the contemporaneous "
            "link is real but untradable. |\n"
            f"| **Contemporaneous** | `CONFIRMED` | Weekly contemporaneous R² = "
            f"**{R['contemp_r2']}%**, t = **+{R['contemp_t']:.2f}** — oil is a coincident "
            "macro indicator for equities. |\n\n"
            "> The key finding: crude oil log-returns are contemporaneously correlated with "
            "equities (R² 9.2%) but have essentially zero predictive power at any tested horizon."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t^{oil}$ be the log-return of USO at period $t$. The claim asserts:\n\n"
            "- **H1 (IS signal).** $\\beta \\ne 0$ with $|t_{HAC}| \\geq 2$ in "
            "$r^{equity}_{t+1} = \\alpha + \\beta \\, r^{oil}_t + \\varepsilon_{t+1}$ "
            "— oil *leads* equities.\n"
            "- **H2 (OOS signal).** Goyal-Welch OOS R² > 0 at the weekly and monthly horizon — "
            "the oil model beats the historical-mean benchmark on out-of-sample data.\n\n"
            "We test H1–H2 and separately document the contemporaneous correlation (not the "
            "claim, but the reason people believe it).\n\n"
            "Note that the sign of beta is ambiguous theoretically: Hamilton (2009) argues "
            "oil *spikes* are recessionary (negative beta); Kilian & Park (2009) show "
            "demand-driven oil increases are growth-positive (positive beta). We test both "
            "and find neither clears the bar."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "A positive IS + OOS signal on a 20-year daily panel would give macro investors a "
            "liquid, cheap-to-compute equity timing signal (oil futures or USO are universally "
            "accessible). The interesting failure here is not just *that* the predictive regression "
            "fails, but *how*: the contemporaneous R² is 9.2% (t = 4.36, statistically overwhelming), "
            "so oil and equity co-movement is definitively real. The question is purely timing — "
            "and on timing, oil tells you nothing about next week."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal construction.** $r^{oil}_t = \\log(\\text{USO}_t) - "
            "\\log(\\text{USO}_{t-1})$ (log-return); weekly/monthly resampling on "
            "week-ending-Friday / month-end. No look-ahead: the oil return through week $t$'s "
            "Friday close predicts the equity return *starting* at that close.\n"
            "- **Contemporaneous baseline.** Same regression with same-period equity return "
            "(not forward). Quantifies the co-movement vs the predictive component.\n"
            "- **IS regression.** OLS with HAC (Newey-West) standard errors — HAC lags "
            "$= \\lfloor 4(n/100)^{2/9} \\rfloor$. The $t_{HAC}$ on $\\beta$ is the "
            "inference-bar number.\n"
            "- **OOS R² (Goyal-Welch 2008).** Expanding window, min 52 weeks training. "
            "$R^2_{OOS} = 1 - \\text{MSFE}_{model}/\\text{MSFE}_{mean}$. Inference via "
            "Diebold-Mariano (1995) t-stat.\n"
            "- **Positive control.** A synthetic daily tape with tunable `pred_r` confirms the "
            "engine recovers an edge when one is planted.\n\n"
            f"Data: USO, CL=F, SPY daily, 2006-04-10 to 2026-06-16 ({R['n_daily']} aligned rows, "
            f"{R['n_weekly']} weekly periods)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Contemporaneous vs predictive — the key decomposition\n\n"
            "The same predictor ($r^{oil}_t$) but two different targets: the *current* week's "
            "equity return (contemporaneous) and the *next* week's (predictive)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    daily = load_real()\n"
            "    feat_w = st.compute_features(daily, horizon='W')\n"
            "    c_res = st.regression_is(feat_w['ret_oil'], feat_w['ret_equity'])\n"
            "    p_res = st.regression_is(feat_w['ret_oil'], feat_w['fwd_equity'])\n"
            "    contemp_t, pred_t = c_res['tstat'], p_res['tstat']\n"
            "    contemp_r2, pred_r2 = c_res['r2']*100, p_res['r2']*100\n"
            "    contemp_b, pred_b = c_res['beta'], p_res['beta']\n"
            "else:\n"
            "    contemp_t, pred_t = R['contemp_t'], R['is_t_eq']\n"
            "    contemp_r2, pred_r2 = R['contemp_r2'], R['is_r2_eq']\n"
            "    contemp_b, pred_b = R['contemp_beta'], R['is_beta_eq']\n"
            "labels = ['Contemporaneous\\n(same week)', 'Predictive\\n(next week)']\n"
            "fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))\n"
            "col = [GREEN if abs(t)>=2 else RED for t in [contemp_t, pred_t]]\n"
            "axes[0].bar(labels, [contemp_r2, pred_r2], color=col)\n"
            "axes[0].set_ylabel('In-sample R2 (%)'); axes[0].set_title('R2 cliff')\n"
            "for i, (lbl, t, r2) in enumerate(zip(labels, [contemp_t, pred_t], [contemp_r2, pred_r2])):\n"
            "    axes[0].annotate(f't={t:+.2f}', (i, r2+0.1), ha='center', fontsize=9)\n"
            "axes[1].bar(labels, [contemp_b, pred_b], color=col)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_ylabel('beta'); axes[1].set_title('Slope sign reversal')\n"
            "for i, (lbl, b) in enumerate(zip(labels, [contemp_b, pred_b])):\n"
            "    axes[1].annotate(f'b={b:+.3f}', (i, b+0.003), ha='center', fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Contemporaneous: b={contemp_b:+.4f}, t={contemp_t:+.3f}, R2={contemp_r2:.1f}%')\n"
            "print(f'Predictive:      b={pred_b:+.4f}, t={pred_t:+.3f}, R2={pred_r2:.3f}%')"
        ),
        md(
            f"> In plain words: the contemporaneous slope is **+{R['contemp_beta']:.3f}** "
            f"(HAC t = +{R['contemp_t']:.2f}, R² = {R['contemp_r2']}%). The predictive slope is "
            f"**{R['is_beta_eq']:+.3f}** (t = {R['is_t_eq']:+.2f}) — essentially zero and "
            "wrong-signed. The oil return that is positive for equities this week is "
            "slightly (but insignificantly) *negative* for equities next week. "
            "Both effects are too small to trade; the sign flip is the fingerprint of a "
            "coincident indicator being mis-read as a predictor."
        ),
        md(
            "### 4b · In-sample regression, weekly and monthly horizons\n\n"
            "Full IS regression table across both forecast horizons."
        ),
        code(
            "rows = []\n"
            "if HAVE_REAL:\n"
            "    daily = load_real()\n"
            "    for hz, lbl in [('W','Weekly'), ('ME','Monthly')]:\n"
            "        res = st.summarize(daily, horizon=hz)\n"
            "        r = res['is_equity']\n"
            "        rows.append((lbl, r['n'], f\"{r['beta']:+.4f}\",\n"
            "                     f\"{r['tstat']:+.3f}\", f\"{r['r2']*100:.3f}%\"))\n"
            "else:\n"
            "    rows = [\n"
            "        ('Weekly',R['n_weekly'],f\"{R['is_beta_eq']:+.4f}\",\n"
            "         f\"{R['is_t_eq']:+.3f}\",f\"{R['is_r2_eq']:.3f}%\"),\n"
            "        ('Monthly',R['n_monthly'],f\"{R['is_beta_eq_m']:+.4f}\",\n"
            "         f\"{R['is_t_eq_m']:+.3f}\",f\"{R['is_r2_eq_m']:.3f}%\"),\n"
            "    ]\n"
            "tbl = pd.DataFrame(rows, columns=['Horizon','n','beta','HAC t','IS R2'])\n"
            "print(tbl.to_string(index=False))"
        ),
        md(
            f"> No cell clears |t| >= 2 on the predictive side. The weekly equity regression "
            f"(t = {R['is_t_eq']:+.2f}) is not only insignificant — it's wrong-signed. "
            f"Monthly equity (t = {R['is_t_eq_m']:+.2f}) is right-signed but essentially zero. "
            "There is no horizon where predictability emerges."
        ),
        md(
            "### 4c · Out-of-sample R² (Goyal-Welch 2008)\n\n"
            "The key test: does the model beat the historical-mean forecast on unseen data? "
            "Expanding window, min 52 weeks training, weekly horizon."
        ),
        code(
            "if HAVE_REAL:\n"
            "    daily = load_real()\n"
            "    feat_w = st.compute_features(daily, horizon='W')\n"
            "    oos_eq = st.oos_r2(feat_w['ret_oil'], feat_w['fwd_equity'])\n"
            "    oos_eq_r2, oos_eq_dm = oos_eq['r2_oos']*100, oos_eq['dm_tstat']\n"
            "    n_oos = oos_eq['n_oos']\n"
            "else:\n"
            "    oos_eq_r2, oos_eq_dm = R['oos_r2_eq'], R['oos_dm_eq']\n"
            "    n_oos = R['n_weekly'] - 52\n"
            "\n"
            "print('OOS R2 results (weekly, expanding window):')\n"
            "print(f'  Equity: OOS R2={oos_eq_r2:+.3f}%  DM t={oos_eq_dm:+.3f}')\n"
            "print(f'  n_oos = {n_oos}')\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(7, 4))\n"
            "ax.bar(['Equity OOS R2 (%)'], [oos_eq_r2],\n"
            "       color=[RED if oos_eq_r2 < 0 else GREEN])\n"
            "ax.axhline(0, c='k', lw=1.5, ls='--')\n"
            "ax.annotate(f'DM t={oos_eq_dm:+.2f}', (0, oos_eq_r2 - 0.1), ha='center', fontsize=11)\n"
            "ax.set_ylabel('Goyal-Welch OOS R2 (%)')\n"
            "ax.set_title('OOS R2: negative — the historical mean wins')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> OOS R² = **{R['oos_r2_eq']:+.1f}%** — the oil model loses to the historical "
            f"mean by a full {abs(R['oos_r2_eq']):.1f}%. The DM t-stat is **{R['oos_dm_eq']:+.2f}** "
            "(significantly negative, confirming underperformance). This is the Goyal-Welch "
            "story repeating with exceptional clarity: a visually compelling co-movement "
            "chart, zero predictive power."
        ),
        md(
            "### 4d · Horizon sensitivity\n\n"
            "Is there any horizon where predictability emerges?"
        ),
        code(
            "horizons_lbl = ['Weekly', 'Monthly']\n"
            "if HAVE_REAL:\n"
            "    daily = load_real()\n"
            "    is_ts, oos_r2s = [], []\n"
            "    for hz in ['W', 'ME']:\n"
            "        res = st.summarize(daily, horizon=hz)\n"
            "        is_ts.append(res['is_equity']['tstat'])\n"
            "        oos_r2s.append(res['oos_equity']['r2_oos']*100)\n"
            "else:\n"
            "    is_ts = [R['is_t_eq'], R['is_t_eq_m']]\n"
            "    oos_r2s = [R['oos_r2_eq'], R['oos_r2_eq_m']]\n"
            "fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))\n"
            "col = [RED if abs(t)<2 else GREEN for t in is_ts]\n"
            "axes[0].bar(horizons_lbl, is_ts, color=col)\n"
            "axes[0].axhline(2, ls='--', c=GREY); axes[0].axhline(-2, ls='--', c=GREY)\n"
            "axes[0].axhline(0, c='k', lw=1); axes[0].set_ylabel('HAC t-stat')\n"
            "axes[0].set_title('IS t-stat by horizon (inference bar = +/-2)')\n"
            "col2 = [RED if v<0 else GREEN for v in oos_r2s]\n"
            "axes[1].bar(horizons_lbl, oos_r2s, color=col2)\n"
            "axes[1].axhline(0, c='k', lw=1); axes[1].set_ylabel('OOS R2 (%)')\n"
            "axes[1].set_title('OOS R2 by horizon')\n"
            "plt.tight_layout(); plt.show()\n"
            "for lbl, t, r in zip(horizons_lbl, is_ts, oos_r2s):\n"
            "    print(f'{lbl}: IS t={t:+.3f}  OOS R2={r:+.3f}%')"
        ),
        md(
            "> Neither horizon clears the bar. Monthly is marginally less wrong (right sign, "
            "but t = 0.15 and OOS R² = −6.2%). There is no 'right' horizon where the signal "
            "reliably emerges — the deeper you look, the more convincingly it fails."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — weekly IS beta = {R['is_beta_eq']:+.4f}, HAC t = "
            f"{R['is_t_eq']:+.2f} (wrong sign, near zero); monthly t = {R['is_t_eq_m']:+.2f}; "
            f"OOS R² = {R['oos_r2_eq']:+.1f}% (weekly) — the inference bar is not cleared "
            "at any horizon.\n"
            f"- **Tradability `MIRAGE`** — no directional forecast signal; the contemporaneous "
            f"link ({R['contemp_r2']}%) is real but untradable.\n"
            f"- **Contemporaneous `CONFIRMED`** — weekly contemporaneous R² = "
            f"{R['contemp_r2']}%, t = +{R['contemp_t']:.2f}; oil is a legitimate "
            "coincident macro indicator."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — capacity and timing issues\n\n"
            "Even hypothetically: a weekly signal with IS R² = 0.04% would have near-zero "
            "trading value after a single commission. The OOS R² being −1.8% means the hurdle "
            "is irrelevant — the historical-mean forecast is better. The coincident use case "
            "(risk attribution, factor exposure estimation, macro regime monitoring) is valid "
            "and well-documented, but requires no forecasting assumption."
        ),
        code(
            "# Synthetic positive control: the engine DOES work when signal is planted\n"
            "pred_rs = [0.0, 0.1, 0.2, 0.3, 0.5]\n"
            "is_ts_syn, oos_r2s_syn = [], []\n"
            "for pr in pred_rs:\n"
            "    syn, _ = data.synthetic_daily(n_years=20, pred_r=pr, seed=245)\n"
            "    feat = st.compute_features(syn, horizon='W')\n"
            "    is_ts_syn.append(st.regression_is(feat['ret_oil'], feat['fwd_equity'])['tstat'])\n"
            "    oos_r2s_syn.append(st.oos_r2(feat['ret_oil'], feat['fwd_equity'])['r2_oos']*100)\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))\n"
            "axes[0].plot(pred_rs, is_ts_syn, 'o-', color=GREEN, lw=2)\n"
            "axes[0].axhline(2, ls='--', c=GREY, lw=1); axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_xlabel('planted pred_r'); axes[0].set_ylabel('IS HAC t-stat')\n"
            "axes[0].set_title('Engine finds signal when planted')\n"
            "axes[1].plot(pred_rs, oos_r2s_syn, 'o-', color=GREEN, lw=2)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_xlabel('planted pred_r'); axes[1].set_ylabel('OOS R2 (%)')\n"
            "axes[1].set_title('OOS R2 turns positive when signal is strong')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Real-tape equivalent: pred_r ~ 0 => no exploitable predictability.')"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Structural decomposition.** Kilian & Park (2009) decompose oil shocks into "
            "supply vs demand components using a SVAR. Demand-driven oil increases may be "
            "positive for equities; supply shocks negative. A structural model could in "
            "principle have predictive power that a reduced-form regression misses — but it "
            "requires identifying assumptions and lagged oil production/inventory data.\n"
            "- **The closest sibling.** [Study 85 — Dr-Copper](../../85-dr-copper/) runs "
            "the identical methodology on the copper/gold ratio (HG=F / GC=F vs ^GSPC). "
            "The copper ratio has slightly stronger IS t-stat (−1.81 vs −0.46 here) but "
            "both fail OOS. The common lesson: macro co-movement is real; macro timing "
            "is a mirage.\n"
            "- **Goyal & Welch (2008)** test 14 macro predictors with identical methodology. "
            "Most fail the OOS test. Oil is squarely in this pattern.\n\n"
            "*Fork this, add a structural oil shock decomposition or regime-switching model, "
            "show a positive OOS R² that holds across sub-periods. That's the bar.*"
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


def execute_notebooks():
    """Execute both notebooks in place using jupyter nbconvert."""
    for nb_name in ["01_for_the_curious.ipynb", "02_for_the_quants.ipynb"]:
        nb_path = os.path.join(HERE, nb_name)
        print(f"Executing {nb_name}...")
        result = subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--inplace",
                "--ExecutePreprocessor.timeout=120",
                nb_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"STDERR: {result.stderr}")
            raise RuntimeError(f"Failed to execute {nb_name}")
        print(f"  done.")


if __name__ == "__main__":
    build_curious()
    build_quants()
    execute_notebooks()
