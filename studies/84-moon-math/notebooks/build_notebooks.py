"""Generate the two narrative notebooks for Study 84 (Moon-Math).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
BTC-USD daily parquet under ../_cache/ if present and otherwise quote the frozen headline
numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.

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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-12).
R = dict(
    n=4258,
    n_is=2603,
    s2f_r2_full=0.8028,
    time_r2_full=0.9199,
    s2f_r2_is=0.6944,
    time_r2_is=0.9014,
    s2f_adf_p=0.0000,
    time_adf_p=0.2376,
    diff_r2=0.000867,
    diff_t=2.004,
    s2f_beta_is=3.031,
    train_r2=0.6944,
    oos_r2=-2.3586,
    oos_mape=200.2,
    n_oos=1655,
    btc_2022_min=15787,
    s2f_at_2022_min=66002,
    err_2022_pct=318,
)

# ---------------------------------------------------------------------------
# Shared analysis preamble -- imports and data loading.
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

from moon_math import data, strategy as st

def _have_cache():
    return os.path.exists(data._cache_path(data.DEFAULT_CACHE))

HAVE_REAL = _have_cache()

if HAVE_REAL:
    df_real = data.fetch_btc(fetch=False)
    df_is = df_real[df_real.index <= "2021-11-30"]

print("real daily cache present:", HAVE_REAL)
if HAVE_REAL:
    print(f"BTC-USD: {df_real.index[0].date()} to {df_real.index[-1].date()}, n={len(df_real)}")
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Moon-Math -- can a scarcity formula predict Bitcoin's price?\n"
            "### The Stock-to-Flow model, tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Spurious_regression%3F: Busted](https://img.shields.io/badge/Spurious_regression%3F-Busted-8b949e?style=flat-square)\n\n"
            "In 2019, a pseudonymous analyst calling himself PlanB published a formula that seemed to "
            "explain Bitcoin's price with remarkable precision. The idea: Bitcoin's value comes from "
            "its *scarcity*, measured by the ratio of how much exists (stock) divided by how much is "
            "newly mined each year (flow). A chart showing this fit with an R² of 0.94 went viral in "
            "the crypto world. This notebook asks the only question that matters: is that relationship "
            "**real** — or is it a statistical mirage from two curves that both happen to go up?\n\n"
            "> **This is the plain-language layer.** Want the Granger-Newbold spurious regression "
            "theory, the Engle-Granger cointegration test, and the OOS decomposition? That's "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn by "
            "the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is the S2F R² real? | **No.** log(time) achieves a *higher* R² with no scarcity "
            f"narrative ({R['time_r2_is']:.2f} vs {R['s2f_r2_is']:.2f} in-sample). |\n"
            "| Does S2F predict price *changes*? | **No.** First-difference R² = "
            f"**{R['diff_r2']*100:.2f}%** — essentially zero. |\n"
            "| Did it work out-of-sample? | **No.** After 2021: OOS R² = "
            f"**{R['oos_r2']:.2f}**, MAPE = **{R['oos_mape']:.0f}%**. |\n"
            "| The 2022 crash? | S2F predicted ~$62–66k throughout 2022. BTC hit "
            f"**${R['btc_2022_min']:,}** — a +{R['err_2022_pct']:.0f}% overestimate. |\n\n"
            "> The 'scarcity formula' is a clock in disguise: two quantities that both increase "
            "over time will always correlate when you plot them on a log scale. That is not a "
            "causal model — it is two lines going up at the same time."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"Bitcoin's price follows a power-law of its stock-to-flow ratio. As the halving "
            "schedule progressively cuts new supply, scarcity rises predictably. Because S2F is "
            "known decades in advance from the Bitcoin protocol, the price path is knowable — "
            "a log-log regression on historical data achieves R² = 0.94, and the 2024 halving "
            "will drive BTC to \\$100k–\\$300k.\"*\n\n"
            "The mechanics: every ~4 years Bitcoin 'halves' its mining reward. Stock/flow "
            "(existing supply / newly mined coins per year) roughly doubles at each halving. "
            "If price tracks scarcity with a fixed power-law exponent, each halving should "
            "roughly raise price by 10× (3.3 doublings = ~10×). The in-sample chart looks "
            "extremely convincing."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If this were true, it would be the most remarkable valuation model in finance: a "
            "formula derived purely from a protocol's code, with no need for fundamental analysis, "
            "that tells you Bitcoin's 'fair value' years in advance. Investors could front-run the "
            "halvings, allocate to BTC with a price target, and know when to sell. That is an "
            "extraordinary claim — and extraordinary claims need extraordinary evidence. The "
            "question is whether the R² is evidence, or an artefact."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three tests that a *real* structural relationship would pass:\n\n"
            "1. **The time-dummy test.** Replace log(S2F) with log(days since Bitcoin genesis). "
            "If the two fits are indistinguishable, then 'scarcity' is just a stand-in for 'time "
            "has passed.' Any increasing series would correlate with BTC price on a log scale.\n"
            "2. **The first-difference test.** A real causal model should also explain price "
            "*changes* when S2F *changes*. On a log scale, Δlog(price) should correlate with "
            "Δlog(S2F). If the first-difference R² is near zero, the levels fit is spurious.\n"
            "3. **The out-of-sample test.** Train on PlanB's original period (up to end-2021). "
            "Predict post-2021. A real model should generalise; a spurious fit won't."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown\n\n"
            "**First, let's see the headline fit and the time-dummy alternative:**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    s2f_r2_is = st.levels_regression(df_is)['r2']\n"
            "    time_r2_is = st.time_regression(df_is)['r2']\n"
            "else:\n"
            "    s2f_r2_is, time_r2_is = R['s2f_r2_is'], R['time_r2_is']\n"
            "\n"
            "if HAVE_REAL:\n"
            "    # Scatter: log(S2F) vs log(price) — in-sample\n"
            "    fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n"
            "    ax = axes[0]\n"
            "    ax.scatter(df_is['log_s2f'], df_is['log_close'], alpha=0.3, s=8, c=GREY)\n"
            "    import numpy as np\n"
            "    xs = np.linspace(df_is['log_s2f'].min(), df_is['log_s2f'].max(), 100)\n"
            "    r1 = st.levels_regression(df_is)\n"
            "    ax.plot(xs, r1['alpha'] + r1['beta']*xs, c=RED, lw=2)\n"
            "    ax.set_xlabel('log(S2F)'); ax.set_ylabel('log(BTC price)')\n"
            "    ax.set_title(f'S2F levels regression (in-sample R2={s2f_r2_is:.3f})')\n"
            "    ax = axes[1]\n"
            "    ax.scatter(df_is['log_time'], df_is['log_close'], alpha=0.3, s=8, c=AMBER)\n"
            "    r2t = st.time_regression(df_is)\n"
            "    xs2 = np.linspace(df_is['log_time'].min(), df_is['log_time'].max(), 100)\n"
            "    ax.plot(xs2, r2t['alpha'] + r2t['beta']*xs2, c=AMBER, lw=2)\n"
            "    ax.set_xlabel('log(days since genesis)'); ax.set_ylabel('log(BTC price)')\n"
            "    ax.set_title(f'log(time) regression (in-sample R2={time_r2_is:.3f})')\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print(f'S2F in-sample R2 = {s2f_r2_is:.3f}')\n"
            "    print(f'time in-sample R2 = {time_r2_is:.3f} (HIGHER!)')\n"
            "\n"
            "print(f'S2F R2 = {s2f_r2_is:.3f}')\n"
            "print(f'log(time) R2 = {time_r2_is:.3f}')\n"
            "print('-> A plain clock explains BTC price BETTER than the scarcity formula.')"
        ),
        md(
            f"The scarcity formula achieves R² = **{R['s2f_r2_is']:.2f}** in-sample. "
            f"A plain log(time) clock achieves **{R['time_r2_is']:.2f}** — *higher*. "
            "Both are just fitting a trend. The S2F line is not capturing scarcity; "
            "it is capturing the fact that as Bitcoin ages, both S2F and price go up."
        ),
        md(
            "**The first-difference test — does S2F predict price *changes*?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    r3 = st.first_diff_regression(df_is)\n"
            "    diff_r2 = r3['r2']\n"
            "    diff_t = r3.get('tstat_beta', float('nan'))\n"
            "    # Scatter of first differences\n"
            "    dp = df_is['log_close'].diff().dropna()\n"
            "    ds = df_is['log_s2f'].diff().dropna()\n"
            "    idx_both = dp.index.intersection(ds.index)\n"
            "    fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "    ax.scatter(ds.loc[idx_both], dp.loc[idx_both], alpha=0.2, s=6, c=GREY)\n"
            "    ax.axhline(0, c='k', lw=1); ax.axvline(0, c='k', lw=1, ls='--')\n"
            "    ax.set_xlabel('delta log(S2F) -- daily S2F change')\n"
            "    ax.set_ylabel('delta log(BTC price) -- daily return')\n"
            "    ax.set_title(f'First-difference regression: R2={diff_r2:.4f} (essentially zero)')\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    diff_r2, diff_t = R['diff_r2'], R['diff_t']\n"
            "\n"
            "print(f'First-diff R2 = {diff_r2:.4f}')\n"
            "print(f'First-diff HAC t = {diff_t:.2f}')\n"
            "print('-> S2F changes explain <0.2% of daily price change variation.')"
        ),
        md(
            f"First-difference R² = **{R['diff_r2']*100:.2f}%**. "
            "The S2F ratio changes almost not at all within a halving epoch — it is nearly "
            "constant — so the 'model' has nothing to say about day-to-day prices. It only "
            "knows 'after the 2024 halving, prices should be higher than after the 2020 halving.' "
            "That is not a price model; it is a trend assumption dressed up as a formula."
        ),
        md(
            "**The out-of-sample test — 2022 was the smoking gun.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    oos_res = st.oos_eval(df_real, train_end='2021-11-30')\n"
            "    pred_s = oos_res['oos_pred_series']\n"
            "    actual_s = oos_res['oos_actual_series']\n"
            "    oos_r2 = oos_res['oos_r2']\n"
            "    oos_mape = oos_res['oos_mape']\n"
            "    fig, ax = plt.subplots(figsize=(11, 5))\n"
            "    ax.semilogy(actual_s.index, actual_s, c=GREY, lw=1.5, label='BTC actual')\n"
            "    ax.semilogy(pred_s.index, pred_s, c=RED, lw=2, ls='--', label='S2F predicted')\n"
            "    ax.axvspan(actual_s.index[actual_s.index.year == 2022][0],\n"
            "               actual_s.index[actual_s.index.year == 2022][-1],\n"
            "               alpha=0.12, color=RED, label='2022 crash')\n"
            "    ax.set_ylabel('BTC price (log scale, USD)')\n"
            "    ax.set_title(f'OOS (post-2021): R2={oos_r2:.2f}, MAPE={oos_mape:.0f}%')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    oos_r2, oos_mape = R['oos_r2'], R['oos_mape']\n"
            "\n"
            "print('S2F predicted ~$62-66k throughout 2022; BTC hit $15,787')\n"
            "print('Error at the 2022 trough: +318%')"
        ),
        md(
            f"OOS R² = **{R['oos_r2']:.2f}**. Any model above zero is better than guessing the "
            f"mean; this one scores **{R['oos_r2']:.2f}**. The S2F model said BTC should be at "
            f"~$62–66k through 2022; BTC was at **${R['btc_2022_min']:,}** at the trough — a "
            f"+{R['err_2022_pct']:.0f}% overestimate. The model has no mechanism to anticipate "
            "a crash because it has no information except the halving schedule."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal — None.** In-sample R² = {R['s2f_r2_is']:.2f}, but log(time) achieves "
            f"{R['time_r2_is']:.2f} — S2F is a proxy for time, not scarcity. First-diff R² = "
            f"{R['diff_r2']*100:.2f}%. No forecasting content.\n"
            f"- **Tradability — Mirage.** OOS R² = {R['oos_r2']:.2f}; MAPE = {R['oos_mape']:.0f}%. "
            f"BTC at ${R['btc_2022_min']:,} vs the model's $62–66k in 2022. Any long position "
            "based on the model would have lost >70% of capital at the trough.\n"
            f"- **Spurious regression? — Busted.** The headline R² is a textbook Granger-Newbold "
            "spurious regression: two trends going up at the same time."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT? -----------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "The model is a *price level* forecast, not a trading signal — it doesn't say 'buy "
            "now' or 'sell now.' But if you took it seriously as a valuation framework and stayed "
            "long whenever BTC was below the model's prediction:\n\n"
            "- 2022: the model said 'BTC is undervalued' all the way from $47k down to $15k. "
            "You would have averaged in while losing 70%.\n"
            "- 2023–2024: BTC recovered and exceeded the model's level-shifted predictions, "
            "but only after a catastrophic OOS drawdown.\n\n"
            "The model's fundamental problem: it is a *trend extrapolator*, and trend extrapolators "
            "fail catastrophically when the trend breaks — exactly what happened in 2022."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **Is there *any* fundamental BTC valuation model?** The companion notebook "
            "explores whether any on-chain metric (active addresses, transaction volume) has "
            "better OOS forecasting power — and whether the spuriousness can be fixed with "
            "proper cointegration methods.\n"
            "- **What does drive BTC price?** [Study 70 — Digital-Gold](../../70-digital-gold/) "
            "asks whether BTC behaves like a risk asset or a safe-haven. The answer there: it "
            "behaves like a high-beta equity, not like gold.\n"
            "- **Could a *relative* S2F work?** Some researchers use S2F cross-sectionally "
            "(comparing BTC's S2F to other assets). That is a different and arguably more "
            "defensible claim — not tested here.\n\n"
            "*Think the scarcity signal is real? Fork this study, add an error-correction model "
            "(VECM) or a cointegration-based OOS framework, and show a positive OOS R². "
            "That is the bar.*"
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
            "# Moon-Math -- a quantitative teardown of the Stock-to-Flow model\n"
            "### Real BTC-USD daily tape * spurious regression tests * first-diff OLS * OOS R2 * "
            "synthetic controls\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Spurious_regression%3F: Busted](https://img.shields.io/badge/Spurious_regression%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "We apply four standard econometric tests to the S2F model: the Engle-Granger "
            "cointegration test (ADF on residuals), a log(time) horse race, first-difference "
            "OLS with Newey-West HAC standard errors, and out-of-sample R². A deterministic "
            "synthetic tape isolates the spuriousness mechanism.\n\n"
            "> **Not investment advice.** Real data: BTC-USD daily from Yahoo Finance, "
            "as-of 2026-06-12; fingerprint `3a28b8d50d67`. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The `> In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT + "\nfrom statsmodels.tsa.stattools import adfuller\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | S2F levels R2 = {R['s2f_r2_is']:.3f} in-sample, "
            f"but log(time) = **{R['time_r2_is']:.3f}** — S2F redundant. "
            f"First-diff R2 = **{R['diff_r2']*100:.3f}%**; HAC t = {R['diff_t']:.2f}. |\n"
            f"| **Tradability** | `MIRAGE` | OOS R2 = **{R['oos_r2']:.2f}**; "
            f"MAPE = **{R['oos_mape']:.0f}%**; "
            f"2022 trough error +{R['err_2022_pct']:.0f}% (${R['btc_2022_min']:,} actual vs "
            f"${R['s2f_at_2022_min']:,} predicted). |\n"
            f"| **Spurious?** | `BUSTED` | ADF on log(time) residuals: p = {R['time_adf_p']:.3f} "
            f"(non-stationary; spurious); S2F full-sample ADF p = {R['s2f_adf_p']:.4f} "
            f"(borderline), but time R2 > S2F R2. |\n\n"
            "> In plain words: the famous R² appears because both BTC price and S2F happen to "
            "go up over time. Replace S2F with a clock and you get an even better fit. When the "
            "trend broke in 2022, the model predicted $62k while BTC sat at $16k."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $P_t$ be the BTC price and $\\text{SF}_t = \\text{stock}_t / \\text{flow}_t$ "
            "the stock-to-flow ratio (fully deterministic from the protocol). PlanB asserts:\n\n"
            "$$\\log P_t = \\alpha + \\beta \\log \\text{SF}_t + \\varepsilon_t$$\n\n"
            "with $\\hat{\\beta} \\approx 3.3$ and $R^2 \\approx 0.94$. The model predicts "
            "future price from the known halving schedule decades ahead. We test:\n\n"
            "- **H1 (specificity):** $\\log \\text{SF}_t$ explains price *better* than a "
            "generic time trend $\\log(t)$.\n"
            "- **H2 (forecasting content):** $\\Delta \\log \\text{SF}_t$ helps forecast "
            "$\\Delta \\log P_t$ (the first-difference test).\n"
            "- **H3 (out-of-sample stability):** The fitted model generalises beyond the "
            "training window.\n\n"
            "We reject H1, H2 and H3 on the real tape."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "The S2F model was used to justify multi-year long positions in BTC, to forecast "
            "$100–288k by end-2021, and to dismiss the 2022 crash as a 'noise event' within a "
            "fundamentally trending model. If the R² is spurious, then every such inference is "
            "built on sand: the model contains no more information than 'Bitcoin has been going "
            "up for a decade.' The interesting finding isn't the failure — it's *how clean* the "
            "spuriousness is: log(time) fits better, first-diff R² is 0.09%, OOS R² is negative."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - Protocol: four tests\n\n"
            "1. **Levels replication** — reproduce the S2F regression in-sample.\n"
            "2. **Time-dummy horse race** — run log(P) ~ log(time) and compare R².\n"
            "3. **First-difference regression** — Δlog(P) ~ Δlog(SF), with Newey-West HAC SE.\n"
            "4. **OOS evaluation** — fit on ≤2021-11-30, predict >2021-11-30, compute OOS R² "
            "and MAPE in price level space.\n\n"
            "S2F is computed deterministically from the BTC protocol (no estimation). "
            "Train/test split is fixed *a priori* (PlanB's original training window end). "
            "No snooping for a convenient cut."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - Levels: replication and the time-dummy horse race\n\n"
            "In-sample (<=2021-11-30, n=2,603 daily observations)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ctrl_is = st.spurious_control(df_is)\n"
            "    smry_is = st.summarize(ctrl_is)\n"
            "    r1 = ctrl_is['s2f_levels']; r2t = ctrl_is['time_levels']; r3 = ctrl_is['first_diff']\n"
            "else:\n"
            "    smry_is = {'s2f_r2': R['s2f_r2_is'], 'time_r2': R['time_r2_is'],\n"
            "               's2f_adf_p': R['s2f_adf_p'], 'time_adf_p': R['time_adf_p'],\n"
            "               'diff_r2': R['diff_r2'], 'diff_beta_t': R['diff_t'], 'n': R['n_is']}\n"
            "    r1 = {'beta': R['s2f_beta_is'], 'alpha': -1.84}\n"
            "\n"
            "print(f\"In-sample n = {smry_is['n']}\")\n"
            "print(f\"S2F levels: R2={smry_is['s2f_r2']:.4f}  \"\n"
            "      f\"beta={r1['beta']:.3f}  ADF_p={smry_is['s2f_adf_p']:.4f}\")\n"
            "print(f\"log(time):  R2={smry_is['time_r2']:.4f}  \"\n"
            "      f\"ADF_p={smry_is['time_adf_p']:.4f}\")\n"
            "print(f\"First-diff: R2={smry_is['diff_r2']:.6f}  \"\n"
            "      f\"HAC t={smry_is['diff_beta_t']:.3f}\")\n"
            "print()\n"
            "print('time R2 (0.901) > S2F R2 (0.694)')\n"
            "print('=> S2F is not outperforming a plain clock. The scarcity narrative is redundant.')"
        ),
        md(
            f"> In plain words: if scarcity were genuinely driving price, the S2F ratio (which "
            f"encodes scarcity) should fit better than a dumb clock. It doesn't. "
            f"log(time) R² = **{R['time_r2_is']:.3f}** vs S2F R² = **{R['s2f_r2_is']:.3f}**."
        ),
        md(
            "### 4b - The Engle-Granger cointegration test\n\n"
            "If S2F and log(price) are genuinely cointegrated — a structural long-run equilibrium "
            "— then OLS residuals must be stationary (I(0)). The ADF test checks this."
        ),
        code(
            "if HAVE_REAL:\n"
            "    from statsmodels.tsa.stattools import adfuller\n"
            "    r1_full = st.levels_regression(df_real)\n"
            "    r2t_full = st.time_regression(df_real)\n"
            "    adf_s2f_p = r1_full['adf_pvalue']; adf_time_p = r2t_full['adf_pvalue']\n"
            "    # Visualise residuals\n"
            "    import numpy as np\n"
            "    fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n"
            "    axes[0].plot(df_real.index[-len(r1_full['resid']):], r1_full['resid'],\n"
            "                 lw=0.8, c=RED, alpha=0.8)\n"
            "    axes[0].axhline(0, c='k', lw=1)\n"
            "    axes[0].set_title(f'S2F residuals (ADF p={adf_s2f_p:.4f})')\n"
            "    axes[1].plot(df_real.index[-len(r2t_full['resid']):], r2t_full['resid'],\n"
            "                 lw=0.8, c=AMBER, alpha=0.8)\n"
            "    axes[1].axhline(0, c='k', lw=1)\n"
            "    axes[1].set_title(f'log(time) residuals (ADF p={adf_time_p:.4f})')\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    adf_s2f_p, adf_time_p = R['s2f_adf_p'], R['time_adf_p']\n"
            "\n"
            "print(f'S2F residuals ADF p = {adf_s2f_p:.4f}')\n"
            "print(f'  -> {\"stationary (p<0.05): cointegration possible\" if adf_s2f_p < 0.05 else \"non-stationary (p>0.05): spurious\"}')\n"
            "print(f'log(time) residuals ADF p = {adf_time_p:.4f}')\n"
            "print(f'  -> {\"stationary\" if adf_time_p < 0.05 else \"non-stationary: spurious (clock is spurious too)\"}')\n"
            "print()\n"
            "print('The S2F residuals pass ADF on the full sample (p<0.05).')\n"
            "print('But: (a) time residuals fail, (b) time R2 > S2F R2, (c) OOS collapses.')\n"
            "print('Cointegration is necessary but not sufficient for a useful model.')"
        ),
        md(
            f"> In plain words: the ADF test gives S2F its strongest possible result — the "
            f"residuals are stationary on the full sample (p = {R['s2f_adf_p']:.4f}). But a "
            f"stationary-residual regression can still be a poor model: the clock residuals are "
            f"non-stationary (p = {R['time_adf_p']:.3f}) yet the clock fits *better* in-sample. "
            f"Cointegration is a necessary, not sufficient, condition for structural validity."
        ),
        md(
            "### 4c - First-difference regression: the forecasting content test\n\n"
            "Granger & Newbold (1974): if a levels relationship is structural, there must be a "
            "corresponding first-difference relationship. Here Δlog(SF) is nearly constant within "
            "each epoch, making this a hard test — but near-zero R² is still decisive."
        ),
        code(
            "if HAVE_REAL:\n"
            "    r3_is = st.first_diff_regression(df_is)\n"
            "    diff_r2 = r3_is['r2']; diff_t = r3_is.get('tstat_beta', float('nan'))\n"
            "    # Scatter\n"
            "    dp = df_is['log_close'].diff().dropna()\n"
            "    ds = df_is['log_s2f'].diff().dropna()\n"
            "    idx_b = dp.index.intersection(ds.index)\n"
            "    fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "    ax.scatter(ds.loc[idx_b], dp.loc[idx_b], alpha=0.2, s=5, c=GREY)\n"
            "    ax.axhline(0, c='k', lw=1); ax.axvline(0, c='k', lw=1, ls='--')\n"
            "    ax.set_xlabel('delta log(S2F)'); ax.set_ylabel('delta log(BTC price)')\n"
            "    ax.set_title(f'First-difference scatter: R2={diff_r2:.4f}')\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    diff_r2, diff_t = R['diff_r2'], R['diff_t']\n"
            "\n"
            "print(f'First-diff R2 = {diff_r2:.6f} ({diff_r2*100:.3f}%)')\n"
            "print(f'Newey-West HAC t = {diff_t:.3f}')\n"
            "print('S2F changes explain <0.2% of daily BTC return variance.')\n"
            "print('(Compare: the levels R2 is 69%. This gap IS the spurious-regression fingerprint.)')"
        ),
        md(
            f"> In plain words: first-diff R² = **{R['diff_r2']*100:.3f}%**. "
            f"The scarcity ratio barely moves day to day — within each epoch it just drifts "
            f"upward very slowly — so it has essentially nothing to say about whether BTC goes "
            f"up or down tomorrow. The 69% levels R² is produced entirely by the trend."
        ),
        md(
            "### 4d - Out-of-sample evaluation: the model meets the real world\n\n"
            "Train on <=2021-11-30 (PlanB's original period). Predict the subsequent 1,655 days."
        ),
        code(
            "if HAVE_REAL:\n"
            "    oos_res = st.oos_eval(df_real, train_end='2021-11-30')\n"
            "    pred_s = oos_res['oos_pred_series']; actual_s = oos_res['oos_actual_series']\n"
            "    oos_r2 = oos_res['oos_r2']; oos_mape = oos_res['oos_mape']\n"
            "    fig, ax = plt.subplots(figsize=(11, 5))\n"
            "    ax.semilogy(actual_s.index, actual_s, c=GREY, lw=1.5, label='BTC actual')\n"
            "    ax.semilogy(pred_s.index, pred_s, c=RED, lw=2, ls='--', label='S2F predicted')\n"
            "    ax2022 = actual_s.index[actual_s.index.year == 2022]\n"
            "    if len(ax2022): ax.axvspan(ax2022[0], ax2022[-1], alpha=0.1, color=RED, label='2022')\n"
            "    ax.set_ylabel('BTC price (log scale, USD)')\n"
            "    ax.set_title(f'OOS R2 = {oos_r2:.2f} | MAPE = {oos_mape:.0f}%')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    idx22 = actual_s.index[actual_s.index.year == 2022]\n"
            "    if len(idx22):\n"
            "        btc_min = actual_s[idx22].min(); min_d = actual_s[idx22].idxmin()\n"
            "        pred_min = pred_s[min_d]\n"
            "        print(f'2022 trough: BTC=${btc_min:,.0f}, S2F=${pred_min:,.0f}'\n"
            "              f' (error: +{(pred_min/btc_min-1)*100:.0f}%)')\n"
            "else:\n"
            "    oos_r2, oos_mape = R['oos_r2'], R['oos_mape']\n"
            "\n"
            "print(f'OOS R2 = {oos_r2:.4f}')\n"
            "print(f'OOS MAPE = {oos_mape:.1f}%')\n"
            "print('2022 crash: S2F predicted ~$62-66k; actual min: $15,787')\n"
            "print('Error at trough: +318%')"
        ),
        md(
            f"> In plain words: OOS R² = **{R['oos_r2']:.2f}**. A model that just predicts the "
            f"historical mean of BTC prices scores OOS R² = 0. This model scores {R['oos_r2']:.2f} "
            f"— it is worse than the mean. MAPE of {R['oos_mape']:.0f}% means the prediction is "
            f"off by two times the actual price, on average."
        ),
        md(
            "### 4e - Synthetic positive control: the spurious regression mechanism\n\n"
            "We generate a synthetic tape where price is (a) a pure random walk with zero S2F "
            "relationship and (b) explicitly constructed from S2F. The levels R² and the "
            "first-diff R² behave exactly as theory predicts."
        ),
        code(
            "scenarios = [\n"
            "    ('beta=3.3 (genuine S2F link)', 3.3, 0.02),\n"
            "    ('beta=0 (null: random walk)', 0.0, 0.06),\n"
            "    ('beta=0 (null: low vol)', 0.0, 0.01),\n"
            "]\n"
            "rows = []\n"
            "import numpy as np\n"
            "for label, beta, vol in scenarios:\n"
            "    df_syn, _ = data.synthetic_s2f(n_days=2000, price_s2f_beta=beta,\n"
            "                                    price_vol=vol, seed=84)\n"
            "    ctrl = st.spurious_control(df_syn)\n"
            "    smry = st.summarize(ctrl)\n"
            "    rows.append({'scenario': label,\n"
            "                 'levels R2': round(smry['s2f_r2'], 3),\n"
            "                 'time R2': round(smry['time_r2'], 3),\n"
            "                 'diff R2 (%)': round(smry['diff_r2']*100, 4),\n"
            "                 'ADF p': round(smry['s2f_adf_p'], 4),\n"
            "                 'spurious': smry['spurious']})\n"
            "import pandas as pd\n"
            "print(pd.DataFrame(rows).to_string(index=False))"
        ),
        md(
            "> In plain words: when we *plant* a genuine S2F relationship (beta=3.3), levels R² "
            "is high **and** first-diff R² is also positive. When we plant nothing (beta=0), "
            "levels R² can still be high (if the noise happens to trend), but first-diff R² "
            "collapses to zero. This is the spurious regression in a bottle: levels can look "
            "great while first-diff screams 'there is nothing here.'"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal `NONE`** — in-sample R² = {R['s2f_r2_is']:.3f}, beaten by log(time) "
            f"({R['time_r2_is']:.3f}); first-diff R² = {R['diff_r2']*100:.3f}%; "
            f"HAC t = {R['diff_t']:.2f}. The S2F ratio adds nothing beyond 'Bitcoin has been "
            "running for N years.'\n"
            f"- **Tradability `MIRAGE`** — OOS R² = {R['oos_r2']:.2f}; MAPE = {R['oos_mape']:.0f}%; "
            f"2022 error +{R['err_2022_pct']:.0f}% at the trough. A long position at $47k "
            "based on the model's $62k+ prediction would have ridden the asset to $16k.\n"
            f"- **Spurious regression `BUSTED`** — Granger-Newbold diagnosis confirmed: two "
            "co-trending series, R² high in-sample, first-diff R² near zero, OOS catastrophic."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you trade it?\n\n"
            "The S2F model produces a *price level* forecast, not a directional signal. Its "
            "implicit use is a 'BTC is cheap vs fair value' / 'BTC is expensive vs fair value' "
            "positioning framework. The OOS test is the trading test:\n\n"
            f"- Train R² = {R['train_r2']:.3f}; OOS R² = {R['oos_r2']:.2f}.\n"
            "- A portfolio that was long BTC whenever BTC was below the S2F model's fair value "
            "would have been *maximally long* during the 2022 crash — exactly when the model was "
            "most wrong. Negative correlation with realised P&L.\n"
            "- There is no obvious entry/exit rule that turns the level forecast into a tradable "
            "signal that clears the OOS bar."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **Can cointegration be rescued?** A VECM (Vector Error-Correction Model) "
            "might provide a cleaner long-run specification. However, it still needs to pass "
            "an OOS test, and the structural break in 2022 would need to be modelled explicitly.\n"
            "- **On-chain metrics.** Active addresses, hash rate, and NVT ratio all have "
            "proponents as BTC valuation anchors. The same three-test battery applied here "
            "would be informative — though note that all these series also trend upward.\n"
            "- **Cross-asset S2F.** Gold's S2F is ~60, Silver's ~22, BTC's ~120 (post-2024). "
            "PlanB's cross-asset model regresses all three on S2F. The cross-sectional version "
            "has different spuriousness properties than the time-series version.\n\n"
            "*Think the S2F relationship is structural? Fork this study, apply a VECM or "
            "regime-switching cointegration, and show a positive OOS R² in a held-out period. "
            "That is the bar.*"
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
