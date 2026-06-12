"""Generate the two narrative notebooks for Study 85 (Dr-Copper).

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

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-12).
R = dict(
    # data
    n_daily=6493,
    n_weekly=1344,
    n_monthly=309,
    start="2000-08-30",
    end="2026-06-12",
    equity_fp="8d361abc3a09",
    # contemporaneous weekly
    contemp_beta=0.230,
    contemp_t=6.42,
    contemp_r2=12.2,
    # predictive weekly (equity)
    is_beta_eq=-0.044,
    is_t_eq=-1.81,
    is_r2_eq=0.45,
    # predictive weekly (yield)
    is_beta_yi=-0.032,
    is_t_yi=-0.29,
    is_r2_yi=0.01,
    # OOS weekly equity
    oos_r2_eq=-0.30,
    oos_dm_eq=-0.60,
    # OOS weekly yield
    oos_r2_yi=-0.59,
    oos_dm_yi=-1.11,
    # monthly
    is_beta_eq_m=0.061,
    is_t_eq_m=1.17,
    is_r2_eq_m=1.04,
    oos_r2_eq_m=-0.02,
    # monthly contemp
    contemp_t_m=6.04,
    contemp_r2_m=7.2,
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

from dr_copper import data, strategy as st

CACHE_DIR = os.path.abspath(os.path.join("..", "_cache"))

def _have_cache():
    return all(os.path.exists(data._cache_path(t, CACHE_DIR)) for t in data.TICKERS)

HAVE_REAL = _have_cache()

def load_real():
    panel = data.load_panel(fetch=False, cache_dir=CACHE_DIR)
    return data.build_ratio_frame(panel)

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Dr-Copper — does the copper/gold ratio predict markets?\n"
            "### The macro indicator with a 'PhD in economics', tested honestly\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Contemporaneous--only: Confirmed](https://img.shields.io/badge/Contemporaneous--only-Confirmed-8b949e?style=flat-square)\n\n"
            "You've probably heard it: *copper has a PhD in economics.* The idea is that the "
            "copper/gold ratio — copper divided by gold — tracks the world's mood. When copper "
            "outperforms gold, the world is building things, and stocks and bond yields should "
            "follow. It's a neat story, and there's a real chart that looks convincing. But does "
            "it actually *predict* anything, or does it just move at the same time?\n\n"
            "> 📓 **This is the plain-language layer.** Want the regression tables, OOS R², and "
            "DM t-stats? That's the companion, "
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
            "| Does copper/gold move *with* the stock market? | **Yes.** "
            f"Contemporaneous R² = **{R['contemp_r2']}%**, t = +{R['contemp_t']:.1f}. It's a "
            "real coincident indicator. |\n"
            "| Does it *predict* stock returns next week/month? | **No.** "
            f"Weekly predictive t = **{R['is_t_eq']:+.2f}** (wrong sign!); monthly t = "
            f"**{R['is_t_eq_m']:+.2f}** (< 2). |\n"
            "| Does it beat the naive average out-of-sample? | **No.** "
            f"OOS R² = **{R['oos_r2_eq']:+.2f}%** — the historical mean wins. |\n"
            "| Could you trade it? | **No.** No forecast signal means no tradable timing edge. |\n\n"
            "> The copper/gold ratio moves **with** the business cycle. It doesn't **lead** it. "
            "That's the whole story."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The copper/gold ratio has a PhD in economics. When copper is outperforming gold, "
            "the economy is growing and bond yields and equities should follow. It's one of the "
            "best leading indicators in the macro toolkit.\"*\n"
            "> — Jeffrey Gundlach, DoubleLine Capital (paraphrased from 2017–2018 appearances)\n\n"
            "The logic is clean: copper is industrial (demand up when factories run); gold is a "
            "safe haven (demand up when fear runs). The ratio therefore encodes the market's "
            "**growth vs fear** balance. If markets discount the future, the ratio should lead "
            "equities and yields by a few weeks or months."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it's true, macro investors have a reliable, real-time, cheap-to-compute timing "
            "signal: go long equities / short bonds when copper/gold rises, and vice versa. "
            "Millions of Bloomberg terminal users watch this chart daily. If it's a coincident "
            "rather than a leading indicator, all those traders are confusing **the description** "
            "of where we are with a **prediction** of where we're going — a very common and very "
            "expensive mistake."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "There are two very different claims hidden in the word 'predicts':\n\n"
            "1. **Contemporaneous.** The ratio and equities move together in the same week. This "
            "is easy to test and easy to see on a chart — but it's useless for timing trades.\n"
            "2. **Predictive.** This week's ratio change tells you something about *next* week's "
            "equity return. This is what you'd need to actually trade.\n\n"
            "We test both. The honest bar for the predictive claim is the **Goyal-Welch (2008) "
            "out-of-sample test**: does the ratio model beat the boring historical-mean forecast "
            "on data it was never trained on? A positive OOS R² means it does; negative means "
            "it doesn't. Twenty-five years of daily data (2000–2026), ~1,344 weekly periods."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the compelling chart that started all of this.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    daily = load_real()\n"
            "    # Resample to monthly for a cleaner visual\n"
            "    mo = daily.resample('ME').last().dropna()\n"
            "    cu_au_norm = (mo['cu_au'] - mo['cu_au'].mean()) / mo['cu_au'].std()\n"
            "    yield_norm = (mo['yield10'] - mo['yield10'].mean()) / mo['yield10'].std()\n"
            "    eq_log = np.log(mo['equity'] / mo['equity'].iloc[0]) * 100\n"
            "else:\n"
            "    mo = None; cu_au_norm = None; yield_norm = None; eq_log = None\n"
            "\n"
            "fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)\n"
            "if HAVE_REAL:\n"
            "    axes[0].plot(mo.index, cu_au_norm, color=AMBER, lw=1.5, label='Cu/Au ratio (normalised)')\n"
            "    axes[0].plot(mo.index, yield_norm, color=RED, lw=1.5, label='10y yield (normalised)')\n"
            "    axes[0].legend(loc='upper left'); axes[0].set_title('Cu/Au ratio vs 10y yield — the compelling chart')\n"
            "    axes[1].plot(mo.index, cu_au_norm, color=AMBER, lw=1.5, label='Cu/Au ratio (normalised)')\n"
            "    axes[1].plot(mo.index, eq_log, color=GREEN, lw=1.5, label='S&P 500 (log cumul return %)')\n"
            "    axes[1].legend(loc='upper left'); axes[1].set_title('Cu/Au ratio vs S&P 500')\n"
            "else:\n"
            "    axes[0].text(0.5, 0.5, 'Cache not available — run verify.py --fetch',\n"
            "                 ha='center', va='center', transform=axes[0].transAxes, fontsize=14)\n"
            "    axes[1].text(0.5, 0.5, f'Frozen: contemporaneous R²={R[\"contemp_r2\"]}% (t={R[\"contemp_t\"]})',\n"
            "                 ha='center', va='center', transform=axes[1].transAxes, fontsize=14)\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"The chart is visually striking — copper/gold tracks both yields and equities over "
            "the long sweep. The contemporaneous R² is **{R['contemp_r2']}%** (t = +{R['contemp_t']:.1f}). "
            "This is real. The question is: *is it useful for trading?*"
        ),
        md(
            "**Now the honest predictive test: does this week's copper/gold change predict NEXT "
            "week's equity return?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    daily = load_real()\n"
            "    feat_w = st.compute_features(daily, horizon='W')\n"
            "    c_res = st.regression_is(feat_w['delta_cu_au'], feat_w['ret_equity'])   # contemporaneous\n"
            "    p_res = st.regression_is(feat_w['delta_cu_au'], feat_w['fwd_equity'])   # predictive\n"
            "    oos_res = st.oos_r2(feat_w['delta_cu_au'], feat_w['fwd_equity'])\n"
            "    c_t, p_t, oos = c_res['tstat'], p_res['tstat'], oos_res['r2_oos']*100\n"
            "    c_r2, p_r2 = c_res['r2']*100, p_res['r2']*100\n"
            "else:\n"
            "    c_t, p_t, oos = R['contemp_t'], R['is_t_eq'], R['oos_r2_eq']\n"
            "    c_r2, p_r2 = R['contemp_r2'], R['is_r2_eq']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.5))\n"
            "bars = ax.bar(['Contemporaneous\\n(same week)', 'Predictive\\n(next week, IS)',\n"
            "               'OOS R² vs\\nhistorical mean'],\n"
            "              [c_r2, p_r2, oos],\n"
            "              color=[AMBER, RED, RED])\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for b, t in zip(bars, [c_t, p_t, None]):\n"
            "    label = f't = {t:+.2f}' if t is not None else 'OOS'\n"
            "    ax.annotate(label, (b.get_x()+b.get_width()/2, max(b.get_height(), 0)+0.1),\n"
            "                ha='center', va='bottom', fontsize=9)\n"
            "ax.set_ylabel('R² (%)'); ax.set_title('Contemporaneous vs Predictive R²: the cliff')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Contemp R²={c_r2:.1f}% (t={c_t:+.2f})  |  Pred R²={p_r2:.2f}% (t={p_t:+.2f})  |  OOS R²={oos:+.3f}%')"
        ),
        md(
            f"The cliff is stark: **R² drops from {R['contemp_r2']}% (contemporaneous) to "
            f"{R['is_r2_eq']}% (predictive)**. The predictive t-stat is **{R['is_t_eq']:+.2f}** "
            "(wrong sign, meaning copper going up last week was slightly *negative* for equities "
            f"next week). Out-of-sample, the model beats the naive mean by **{R['oos_r2_eq']:+.2f}%** "
            "— in other words, it *loses* to simply guessing 'the average return'.\n\n"
            "> The chart is real. The forecast isn't. The ratio moves *with* markets because they "
            "both react to the same economic conditions. By the time you see copper rising, the "
            "equity market has already priced in the growth — there's nothing left to predict."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak/None.** Weekly predictive t = **{R['is_t_eq']:+.2f}** (wrong sign); "
            f"monthly t = **{R['is_t_eq_m']:+.2f}**; OOS R² = **{R['oos_r2_eq']:+.2f}%**. "
            "The inference bar is not cleared at any horizon.\n"
            f"- **Tradability — Mirage.** No reliable directional signal → no timing edge. "
            f"The contemporaneous link (R² = {R['contemp_r2']}%) is real but untradable.\n"
            f"- **Contemporaneous/Coincident — Confirmed.** The ratio moves with the cycle, "
            "not ahead of it."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Even if the IS t-stat were +2 and the sign were right, an OOS R² that negative tells "
            "you the model would not have helped you in practice. To trade an equity timing signal "
            "you need to know *now* what the next period's return will be. The contemporaneous link "
            "tells you what copper and equities are doing *together* — useful for attribution and "
            "risk monitoring, useless for entry/exit timing."
        ),
        code(
            "# Scatter: delta_cu_au_t vs fwd_equity_{t+1}\n"
            "if HAVE_REAL:\n"
            "    daily = load_real()\n"
            "    feat = st.compute_features(daily, horizon='W')\n"
            "    x_plot = feat['delta_cu_au'].values * 100\n"
            "    y_plot = feat['fwd_equity'].values * 100\n"
            "else:\n"
            "    rng = np.random.default_rng(85)\n"
            "    x_plot = rng.normal(0, 1.5, 200)\n"
            "    y_plot = rng.normal(0, 2.0, 200)\n"
            "fig, ax = plt.subplots(figsize=(8, 5))\n"
            "ax.scatter(x_plot, y_plot, alpha=0.25, s=15, color=GREY)\n"
            "# Fit a line\n"
            "m, b = np.polyfit(x_plot, y_plot, 1)\n"
            "xr = np.linspace(x_plot.min(), x_plot.max(), 100)\n"
            "ax.plot(xr, m*xr+b, color=RED, lw=2, label=f'slope={m:.3f}')\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Cu/Au ratio change this week (%)'); ax.set_ylabel('S&P 500 return next week (%)')\n"
            "ax.set_title('The predictive scatter: a cloud, not a trend'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('The slope is near zero and barely distinguishable from noise.')"
        ),
        md(
            "The scatter is a cloud. Whatever pattern drove the visually compelling chart "
            "operates over *years*, not *weeks* — and by the time a multi-year trend is "
            "identifiable, it's in every Bloomberg terminal and almost certainly priced in."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The coincident link is real and useful.** For risk attribution, factor "
            "decomposition, and macro regime monitoring, the Cu/Au ratio is a legitimate tool — "
            "just not a forecast. The difference matters enormously.\n"
            "- **What if you look further ahead?** The monthly regression is slightly better "
            "(t = +1.17, right sign), but the OOS R² is still essentially zero (−0.02%). "
            "The predictability doesn't emerge at longer horizons either.\n"
            "- **The real 'Dr. Copper' finding.** Ye et al. (2019) document copper as a leading "
            "indicator for *industrial production* at 3–6 month horizons — not equities at 1-week "
            "horizons. The original claim may have been subtly correct about the commodity cycle "
            "but over-extended to financial markets.\n"
            "- **[Study 67 — Fed-Drift](../../67-fed-drift/)**: macro event windows — a cleaner "
            "way to test whether a macro indicator genuinely leads prices.\n\n"
            "*Think the signal is real at a different horizon or with a better model? Fork this, "
            "show a positive OOS R² that clears the Goyal-Welch bar. That's the bet.*"
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
            "# Dr-Copper — a quantitative teardown\n"
            "### Cu/Au ratio · daily data 2000–2026 · IS regression · Goyal-Welch OOS R² · DM test\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Contemporaneous--only: Confirmed](https://img.shields.io/badge/Contemporaneous--only-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). We "
            "test the copper/gold ratio as a predictor of forward equity returns and 10y yield "
            "changes using HAC-robust in-sample regressions and expanding-window OOS R² "
            "(Goyal-Welch 2008), at weekly and monthly horizons, over 25 years of daily data.\n\n"
            "> ⚠️ **Not investment advice.** Real data: Yahoo daily bars, HG=F/GC=F/^GSPC/^TNX, "
            "2000-08-30 to 2026-06-12. Methods in [`docs/references.md`](../docs/references.md), "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT + "\nfrom quantlab import analytics, stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK/NONE` | Weekly IS beta = {R['is_beta_eq']:+.3f}, HAC *t* = "
            f"**{R['is_t_eq']:+.2f}** (wrong sign); monthly t = **{R['is_t_eq_m']:+.2f}**; "
            f"OOS R² = **{R['oos_r2_eq']:+.2f}%** — the naive mean wins out-of-sample. |\n"
            f"| **Tradability** | `MIRAGE` | No reliable directional forecast → no equity timing "
            "edge. The contemporaneous link is real but untradable. |\n"
            f"| **Contemporaneous** | `CONFIRMED` | Weekly contemporaneous R² = "
            f"**{R['contemp_r2']}%**, t = **+{R['contemp_t']:.2f}** — the ratio is a legitimate "
            "coincident macro indicator. |\n\n"
            "> 💡 The key finding: the Cu/Au ratio moves *with* the business cycle, not *before* "
            "it. Contemporaneous and predictive correlations are sharply different."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t^{Cu/Au}$ be the log-change in the copper/gold ratio at period $t$. "
            "The 'Dr. Copper' hypothesis asserts:\n\n"
            "- **H₁ (IS signal).** $\\beta > 0$ with $|t_{HAC}| \\geq 2$ in "
            "$r^{equity}_{t+1} = \\alpha + \\beta \\, r^{Cu/Au}_t + \\varepsilon_{t+1}$ "
            "— the ratio *leads* equities.\n"
            "- **H₂ (OOS signal).** Goyal-Welch OOS R² > 0 at the weekly and monthly horizon — "
            "the ratio model beats the historical-mean benchmark on out-of-sample data.\n"
            "- **H₃ (yield prediction).** Same test for the 10y yield: $\\beta > 0$, $|t| \\geq 2$. "
            "The Gundlach chart implied the strongest visual fit was with yields.\n\n"
            "We test H₁–H₃ and separately document the contemporaneous correlation (not the "
            "claim, but the reason people believe it)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "A positive IS + OOS signal on a 25-year daily panel is a rare thing. If H₁–H₂ "
            "held, macro investors would have a legitimate growth-timing tool with broad "
            "institutional adoption (Bloomberg terminals, risk dashboards). The interesting "
            "failure is not *that* the predictive regression fails — Goyal & Welch (2008) "
            "showed most macro predictors do — but *how*: the contemporaneous R² is 12% "
            "(statistically overwhelming), so the signal is completely real; the question is "
            "purely about *timing*."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal construction.** $r^{Cu/Au}_t = \\log(Cu_t/Au_t) - "
            "\\log(Cu_{t-1}/Au_{t-1})$ (log-ratio change); weekly/monthly resampling on "
            "week-ending-Friday / month-end. No look-ahead: the ratio change through week $t$'s "
            "Friday close predicts the return *starting* at that close.\n"
            "- **Contemporaneous baseline.** Same regression with the same-period equity return "
            "(not forward). Quantifies the co-movement vs the predictive component.\n"
            "- **IS regression.** OLS with HAC (Newey-West) standard errors — HAC lags "
            "$= \\lfloor 4(n/100)^{2/9} \\rfloor$. The $t_{HAC}$ on $\\beta$ is the "
            "inference-bar number.\n"
            "- **OOS R² (Goyal-Welch 2008).** Expanding window, min 52 weeks training. "
            "$R^2_{OOS} = 1 - \\text{MSFE}_{model}/\\text{MSFE}_{mean}$. Inference via "
            "Diebold-Mariano (1995) t-stat.\n"
            "- **Positive control.** A synthetic daily tape with tunable `pred_r` confirms the "
            "engine recovers an edge when one is planted.\n\n"
            "Data: HG=F, GC=F, ^GSPC, ^TNX daily, 2000-08-30 to 2026-06-12 (6,493 aligned rows, "
            "1,344 weekly periods)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Contemporaneous vs predictive — the key decomposition\n\n"
            "The same predictor ($r^{Cu/Au}_t$) but two different targets: the *current* week's "
            "equity return (contemporaneous) and the *next* week's (predictive)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    daily = load_real()\n"
            "    feat_w = st.compute_features(daily, horizon='W')\n"
            "    c_res = st.regression_is(feat_w['delta_cu_au'], feat_w['ret_equity'])\n"
            "    p_res = st.regression_is(feat_w['delta_cu_au'], feat_w['fwd_equity'])\n"
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
            "axes[0].set_ylabel('In-sample R² (%)'); axes[0].set_title('R² cliff')\n"
            "for i, (lbl, t, r2) in enumerate(zip(labels, [contemp_t, pred_t], [contemp_r2, pred_r2])):\n"
            "    axes[0].annotate(f't={t:+.2f}', (i, r2+0.1), ha='center', fontsize=9)\n"
            "axes[1].bar(labels, [contemp_b, pred_b], color=col)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_ylabel('β'); axes[1].set_title('Slope: same sign? No.')\n"
            "for i, (lbl, b) in enumerate(zip(labels, [contemp_b, pred_b])):\n"
            "    axes[1].annotate(f'β={b:+.3f}', (i, b+0.003), ha='center', fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Contemporaneous: β={contemp_b:+.4f}, t={contemp_t:+.3f}, R²={contemp_r2:.1f}%')\n"
            "print(f'Predictive:      β={pred_b:+.4f}, t={pred_t:+.3f}, R²={pred_r2:.3f}%')"
        ),
        md(
            f"> 💡 The contemporaneous slope is **+{R['contemp_beta']:.3f}** (HAC t = "
            f"+{R['contemp_t']:.2f}, R² = {R['contemp_r2']}%). The predictive slope is "
            f"**{R['is_beta_eq']:+.3f}** (t = {R['is_t_eq']:+.2f}) — *wrong sign*. The ratio "
            "that rises with equities simultaneously actually points to a tiny equity *fall* "
            "the next week. Both effects are too small to trade, but the sign reversal is the "
            "fingerprint of a coincident indicator being mis-read as a predictor."
        ),
        md(
            "### 4b · In-sample regression, weekly and monthly horizons\n\n"
            "Full IS regression table across both forecast horizons and both targets."
        ),
        code(
            "rows = []\n"
            "if HAVE_REAL:\n"
            "    daily = load_real()\n"
            "    for hz, lbl in [('W','Weekly'), ('ME','Monthly')]:\n"
            "        res = st.summarize(daily, horizon=hz)\n"
            "        for key, tgt in [('is_equity','Equity'), ('is_yield','Yield')]:\n"
            "            r = res[key]\n"
            "            rows.append((lbl, tgt, r['n'], f\"{r['beta']:+.4f}\",\n"
            "                         f\"{r['tstat']:+.3f}\", f\"{r['r2']*100:.3f}%\"))\n"
            "else:\n"
            "    rows = [\n"
            "        ('Weekly','Equity',R['n_weekly'],f\"{R['is_beta_eq']:+.4f}\",\n"
            "         f\"{R['is_t_eq']:+.3f}\",f\"{R['is_r2_eq']:.3f}%\"),\n"
            "        ('Weekly','Yield',R['n_weekly'],f\"{R['is_beta_yi']:+.4f}\",\n"
            "         f\"{R['is_t_yi']:+.3f}\",f\"{R['is_r2_yi']:.3f}%\"),\n"
            "        ('Monthly','Equity',R['n_monthly'],f\"{R['is_beta_eq_m']:+.4f}\",\n"
            "         f\"{R['is_t_eq_m']:+.3f}\",f\"{R['is_r2_eq_m']:.3f}%\"),\n"
            "    ]\n"
            "tbl = pd.DataFrame(rows, columns=['Horizon','Target','n','β','HAC t','IS R²'])\n"
            "print(tbl.to_string(index=False))"
        ),
        md(
            f"> 💡 No cell clears |t| ≥ 2 on the predictive side. The weekly equity regression "
            f"(t = {R['is_t_eq']:+.2f}) is marginally significant but *wrong-signed*. Monthly equity "
            f"(t = {R['is_t_eq_m']:+.2f}) is right-signed but weak. The yield regressions are "
            "essentially noise despite the compelling visual chart."
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
            "    oos_eq = st.oos_r2(feat_w['delta_cu_au'], feat_w['fwd_equity'])\n"
            "    oos_yi = st.oos_r2(feat_w['delta_cu_au'], feat_w['fwd_yield_chg'])\n"
            "    oos_eq_r2, oos_eq_dm = oos_eq['r2_oos']*100, oos_eq['dm_tstat']\n"
            "    oos_yi_r2, oos_yi_dm = oos_yi['r2_oos']*100, oos_yi['dm_tstat']\n"
            "    n_oos = oos_eq['n_oos']\n"
            "else:\n"
            "    oos_eq_r2, oos_eq_dm = R['oos_r2_eq'], R['oos_dm_eq']\n"
            "    oos_yi_r2, oos_yi_dm = R['oos_r2_yi'], R['oos_dm_yi']\n"
            "    n_oos = R['n_weekly'] - 52\n"
            "\n"
            "print('OOS R² results (weekly, expanding window):')\n"
            "print(f'  Equity: OOS R²={oos_eq_r2:+.3f}%  DM t={oos_eq_dm:+.3f}')\n"
            "print(f'  Yield:  OOS R²={oos_yi_r2:+.3f}%  DM t={oos_yi_dm:+.3f}')\n"
            "print(f'  n_oos = {n_oos}')\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            "bar_labels = ['Equity (OOS R²%)', 'Yield (OOS R²%)']\n"
            "bar_vals = [oos_eq_r2, oos_yi_r2]\n"
            "cols = [RED if v < 0 else GREEN for v in bar_vals]\n"
            "ax.bar(bar_labels, bar_vals, color=cols)\n"
            "ax.axhline(0, c='k', lw=1.5, ls='--')\n"
            "ax.set_ylabel('Goyal-Welch OOS R² (%)')\n"
            "ax.set_title('OOS R²: both negative — the historical mean wins')\n"
            "for i, (v, t) in enumerate(zip(bar_vals, [oos_eq_dm, oos_yi_dm])):\n"
            "    ax.annotate(f'DM t={t:+.2f}', (i, v - 0.04), ha='center', fontsize=10)\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"> 💡 Both OOS R² values are negative: the copper/gold model loses to the "
            f"historical mean, both for equities ({R['oos_r2_eq']:+.2f}%) and yields "
            f"({R['oos_r2_yi']:+.2f}%). The DM t-stats are small and negative — no evidence of "
            "forecasting improvement. This is the Goyal-Welch story repeating: a macro indicator "
            "that 'works' in-sample and 'fails' out-of-sample is the norm, not the exception."
        ),
        md(
            "### 4d · Horizon sensitivity\n\n"
            "Is there a horizon where predictability emerges?"
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
            "axes[0].set_title('IS t-stat by horizon')\n"
            "col2 = [RED if v<0 else GREEN for v in oos_r2s]\n"
            "axes[1].bar(horizons_lbl, oos_r2s, color=col2)\n"
            "axes[1].axhline(0, c='k', lw=1); axes[1].set_ylabel('OOS R² (%)')\n"
            "axes[1].set_title('OOS R² by horizon')\n"
            "plt.tight_layout(); plt.show()\n"
            "for lbl, t, r in zip(horizons_lbl, is_ts, oos_r2s):\n"
            "    print(f'{lbl}: IS t={t:+.3f}  OOS R²={r:+.3f}%')"
        ),
        md(
            "> 💡 Neither horizon clears the bar. Monthly is marginally more promising (right sign, "
            "slightly higher IS t) but the OOS R² is essentially zero. There is no 'right' "
            "horizon where the signal reliably emerges."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK/NONE`** — weekly IS beta = {R['is_beta_eq']:+.4f}, HAC t = "
            f"{R['is_t_eq']:+.2f} (wrong sign); monthly t = {R['is_t_eq_m']:+.2f}; OOS R² = "
            f"{R['oos_r2_eq']:+.2f}% (weekly) — the inference bar is not cleared.\n"
            f"- **Tradability `MIRAGE`** — no directional forecast signal; the contemporaneous "
            f"link (R² = {R['contemp_r2']}%) is real but untradable.\n"
            f"- **Contemporaneous `CONFIRMED`** — weekly contemporaneous R² = "
            f"{R['contemp_r2']}%, t = +{R['contemp_t']:.2f}; the ratio is a legitimate "
            "coincident macro indicator."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — capacity and timing issues\n\n"
            "The practical problems compound: even if a weekly signal existed, the futures roll "
            "costs (HG=F, GC=F) and equity index execution costs would need to be covered by a "
            "sub-0.5% R² weekly edge. The OOS R² being negative means the hurdle is irrelevant — "
            "there is nothing to trade. The coincident use case (regime monitoring, factor exposure "
            "estimation) is valid but requires no forecasting assumption."
        ),
        code(
            "# Synthetic positive control: the engine DOES work when signal is planted\n"
            "pred_rs = [0.0, 0.1, 0.2, 0.3, 0.5]\n"
            "is_ts_syn, oos_r2s_syn = [], []\n"
            "for pr in pred_rs:\n"
            "    syn, _ = data.synthetic_daily(n_years=20, pred_r=pr, seed=85)\n"
            "    syn['cu_au'] = syn['copper'] / syn['gold']\n"
            "    feat = st.compute_features(syn, horizon='W')\n"
            "    is_ts_syn.append(st.regression_is(feat['delta_cu_au'], feat['fwd_equity'])['tstat'])\n"
            "    oos_r2s_syn.append(st.oos_r2(feat['delta_cu_au'], feat['fwd_equity'])['r2_oos']*100)\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))\n"
            "axes[0].plot(pred_rs, is_ts_syn, 'o-', color=GREEN, lw=2)\n"
            "axes[0].axhline(2, ls='--', c=GREY, lw=1); axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_xlabel('planted pred_r'); axes[0].set_ylabel('IS HAC t-stat')\n"
            "axes[0].set_title('Engine finds signal when planted')\n"
            "axes[1].plot(pred_rs, oos_r2s_syn, 'o-', color=GREEN, lw=2)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_xlabel('planted pred_r'); axes[1].set_ylabel('OOS R² (%)')\n"
            "axes[1].set_title('OOS R² turns positive when signal is strong')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Real-tape equivalent: pred_r ≈ 0 → the market has no exploitable predictability here.')"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **What the Cu/Au ratio is good for.** Regime detection, factor exposure, and "
            "real-time GDP nowcasting (not forecasting). Ye et al. (2019) find copper *does* "
            "lead industrial production at 3–6 month horizons — the Gundlach claim may be "
            "right about the real economy and wrong about financial markets' discounting speed.\n"
            "- **Other macro timing signals.** Goyal & Welch (2008) test 14 macro predictors "
            "with identical methodology; most fail the OOS test. The term spread, default premium, "
            "and earnings yield have stronger theoretical foundations but similar empirical fragility.\n"
            "- **[Study 67 — Fed-Drift](../../67-fed-drift/)**: a macro event-window study — the "
            "FOMC announcement channel.\n"
            "- **[Study 70 — Digital-Gold](../../70-digital-gold/)**: the gold side of the ratio "
            "as a safe-haven asset.\n\n"
            "*Fork this, add a regime-switching model or a Bayesian predictor panel, show a "
            "positive OOS R² that holds across sub-periods. That's the bar.*"
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
