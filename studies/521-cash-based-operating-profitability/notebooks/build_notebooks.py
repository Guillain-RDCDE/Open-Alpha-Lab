"""Generate the two narrative notebooks for Study 521 (Cash-Based Operating Profitability).

    python notebooks/build_notebooks.py
    python -W ignore -m jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

The synthetic positive control runs anywhere, offline and deterministic. The real-tape
cells read the cached yfinance parquets under ../_cache/ if present and otherwise quote the
frozen headline numbers in ``R`` (the single dict mirroring docs/results.md, as-of
2026-06-26).
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


# Frozen real-panel headline numbers -- the ONE dict mirroring docs/results.md (2026-06-26).
R = dict(
    n_years=3, year_start=2022, year_end=2024, n_fiscal_years=5, n_tickers=35,
    cop_mean=10.63, cop_net=9.96, cop_vol=10.51, cop_sharpe=1.011, cop_t=1.751,
    cop_hit=100.0, cop_dd=0.0, turnover=42.9, cost=0.67,
    high_mean=34.50, low_mean=23.87, mkt_mean=23.69, high_vs_mkt=10.81,
    gpa_mean=4.05, gpa_t=0.27,
    placebo_real=10.63, placebo_null=-0.52, placebo_std=12.63, placebo_p=0.392,
    fp_cop="574e2b84dcd3", fp_gpa="6f15c9dde4be", fp_fwd="fe3d772b4c50",
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from cash_op import data, strategy as st

def _have_cache():
    cd = os.path.abspath(os.path.join("..", "_cache"))
    return all(os.path.exists(os.path.join(cd, f))
               for f in ("cop.parquet", "gpa.parquet", "fwd.parquet"))

HAVE_REAL = _have_cache()
print("yfinance cache present:", HAVE_REAL)

if HAVE_REAL:
    cop, gpa, fwd = data.fetch_panel(fetch=False)
    H = st.apply_costs(st.quintile_hedge(cop, fwd, q=0.20))
    Hg = st.quintile_hedge(gpa, fwd, q=0.20)
    s_cop = st.summary(H["hedge"]); s_net = st.summary(H["net"])
    s_hi = st.summary(H["high"]); s_lo = st.summary(H["low"]); s_mkt = st.summary(H["market"])
    s_gpa = st.summary(Hg["hedge"])
    print(f"N years: {s_cop['n']} | cash-OP hedge: {s_cop['mean']*100:+.2f}%/yr"
          f" | one-sample t: {s_cop['tstat']:+.3f}")
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Cash-Based Operating Profitability — does stripping accruals sharpen the signal?\n"
            "### Ball, Gerakos, Linnainmaa & Nikolaev (2016) on a large-cap survivor basket, tested honestly\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Cash beats accrual%3F: Mixed](https://img.shields.io/badge/Cash_beats_accrual%3F-Mixed-8b949e?style=flat-square)\n\n"
            "In 2016, Ray Ball and co-authors made a sharp point about the famous quality "
            "factor: the profitability that predicts returns is the **cash** part. Accounting "
            "operating profit mixes durable cash earnings with *accruals* — paper entries "
            "like receivables and inventory changes that tend to reverse. Strip the accruals "
            "out and you get **cash-based operating profitability**, which (they show) beats "
            "the accrual-laden gross-profitability of Novy-Marx (2013) in the cross-section. "
            "We test that claim on a large-cap basket — and name the data limits up front.\n\n"
            "> **This is the plain-language layer.** Want the t-stats, the placebo "
            "distribution, and the head-to-head? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** Every chart below is drawn by the code beside it. "
            "House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does cash-OP predict next-year returns? | **Weakly yes** — hedge "
            f"**+{R['cop_mean']:.1f}%/yr**, but one-sample *t* = **+{R['cop_t']:.2f}** on only "
            f"**{R['n_years']} usable years**, below the |t|≥2 bar. |\n"
            "| Does it beat the accrual-laden version? | **Directionally yes.** Cash-OP "
            f"**+{R['cop_mean']:.1f}%/yr** vs gross-profitability **+{R['gpa_mean']:.1f}%/yr** — "
            "the Ball et al. ordering holds, but on 3 points. |\n"
            "| Is the hedge tradable? | **Mirage.** The 100% hit-rate and 0% drawdown are "
            "small-sample illusions; the spread fails a label-shuffle placebo. |\n"
            "| Is there a data problem? | **Yes, and we name it.** yfinance gives only ~5 "
            "fiscal years; two lack a forward window → just 3 usable hedge years. |\n\n"
            "> The cash-vs-accrual *direction* is real in the literature. On 3 years of "
            "large-cap survivors it shows the right sign but cannot clear the bar."
        ),

        md(
            "## 1 — The claim\n\n"
            "> *\"Operating profitability predicts returns — but it is the cash component "
            "that does the work. Accruals are the reversing, noisy part of earnings; once "
            "you strip them out, cash-based operating profitability subsumes both the "
            "gross-profitability premium and the accruals anomaly.\"*\n\n"
            "— Ball, Gerakos, Linnainmaa & Nikolaev (2016), *JFE*\n\n"
            "It is a refinement of the **quality factor**. Novy-Marx (2013, our "
            "[Study 122](../../122-gross-profitability)) used GrossProfit / Assets. BGLN say: "
            "good idea, wrong measure — use the *cash* operating profit. We build cash-OP as "
            "operating income plus depreciation/amortisation plus the change in working "
            "capital (the accrual block that the cash-flow statement reports directly), all "
            "scaled by total assets."
        ),

        md(
            "## 2 — So what?\n\n"
            "If cash-OP genuinely beats accrual-laden profitability, every quality ETF and "
            "five-factor model anchored on operating profit is using a noisier signal than it "
            "could. The fix is nearly free: the data is on the same filings. The interesting "
            "question for us is **how much survives on a realistic large-cap pull, with the "
            "survivorship and short-history limits named rather than hidden.**"
        ),

        md(
            "## 3 — How would we even know?\n\n"
            "1. **Lag the signal.** Cash-OP from fiscal year *y* predicts the next 12-month "
            "forward return, entered **one trading day after** the fiscal-year-end is public. "
            "No look-ahead.\n"
            "2. **Contrast like-for-like.** Run the *same* quintile engine on cash-OP and on "
            "gross-profitability GP/A, on the *same* basket and years.\n"
            "3. **Name the limits.** The basket is names still trading in 2026 "
            "(survivorship), and yfinance only exposes ~5 fiscal years — so only 3 have a "
            "full forward window. Three annual points cannot be robust, full stop."
        ),

        md(
            "## 4 — The teardown\n\n"
            "**First, does the long-short work in a world where the cash-OP effect is planted?**"
        ),
        code(
            "ctrl = st.synthetic_control()\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREEN if m > 1 else (RED if m < -1 else GREY) for m in ctrl['hedge_mean_%']]\n"
            "ax.bar(ctrl['premium'].astype(str), ctrl['hedge_mean_%'], color=col)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted cash-OP premium'); ax.set_ylabel('hedge mean (%/yr)')\n"
            "ax.set_title('Synthetic control: the engine finds the effect when planted')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(3).to_string(index=False))"
        ),
        md(
            "The engine is faithful: a strong hedge when a premium is planted, noise at zero "
            "(t ≈ −1.1). So the real-tape verdict reflects **the market**, not the method."
        ),
        md("**Now the honest test: cash-OP vs the accrual-laden GP/A on the real basket.**"),
        code(
            "if HAVE_REAL:\n"
            "    cop_m = s_cop['mean']*100; cop_t = s_cop['tstat']\n"
            "    gpa_m = s_gpa['mean']*100; gpa_t = s_gpa['tstat']\n"
            "    hi_m = s_hi['mean']*100; mk_m = s_mkt['mean']*100; lo_m = s_lo['mean']*100\n"
            "else:\n"
            "    cop_m, cop_t = R['cop_mean'], R['cop_t']\n"
            "    gpa_m, gpa_t = R['gpa_mean'], R['gpa_t']\n"
            "    hi_m, mk_m, lo_m = R['high_mean'], R['mkt_mean'], R['low_mean']\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))\n"
            "axes[0].bar(['Cash-OP', 'Gross-prof\\n(GP/A)'], [cop_m, gpa_m],\n"
            "            color=[AMBER, GREY], width=0.55)\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_ylabel('hedge mean (%/yr)')\n"
            "axes[0].set_title(f'Cash beats accrual-laden\\n(t {cop_t:+.2f} vs {gpa_t:+.2f})')\n"
            "axes[1].bar(['High\\ncash-OP', 'EW\\nmarket', 'Low\\ncash-OP'],\n"
            "            [hi_m, mk_m, lo_m], color=[GREEN, GREY, RED], width=0.55)\n"
            "axes[1].axhline(mk_m, ls='--', c=GREY, lw=1)\n"
            "axes[1].set_ylabel('mean next-year return (%/yr)')\n"
            "axes[1].set_title('Cash-OP legs vs market')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Cash-OP hedge: {cop_m:+.2f}%/yr (t={cop_t:+.2f}) | "
            "GP/A hedge: {gpa_m:+.2f}%/yr (t={gpa_t:+.2f})')"
        ),
        md(
            f"The cash-OP hedge earns **+{R['cop_mean']:.1f}%/yr** (*t* = "
            f"**+{R['cop_t']:.2f}**) vs the accrual-laden GP/A at **+{R['gpa_mean']:.1f}%/yr** "
            f"(*t* = +{R['gpa_t']:.2f}) — the Ball et al. direction holds. But both sit on "
            f"**{R['n_years']} years** and neither clears |t| ≥ 2."
        ),

        md(
            "## 5 — The verdict\n\n"
            f"- **Signal — WEAK.** Cash-OP hedge +{R['cop_mean']:.1f}%/yr at one-sample *t* = "
            f"+{R['cop_t']:.2f} (< 2) on {R['n_years']} usable years, and it **fails** the "
            f"label-shuffle placebo (*p* = {R['placebo_p']:.2f}). Right sign, strong "
            "literature prior (BGLN 2016) → above `NONE`, but the tape does not clear the bar.\n"
            "- **Tradability — MIRAGE.** Sharpe +1.01, 100% hit-rate, 0% drawdown are all "
            "3-year-window artefacts; the placebo says the spread is chance.\n"
            "- **Cash beats accrual? — MIXED.** Directionally yes, but on 3 points neither "
            "leg is significant."
        ),

        md(
            "## 6 — Could you actually trade it?\n\n"
            "Annual rebalancing means low turnover (~43%/yr here → just 0.67%/yr in costs "
            "including borrow), so costs are **not** the obstacle. The obstacles are:\n\n"
            "1. **Three years of data.** yfinance's statement history is the binding "
            "constraint; you cannot infer a factor premium from 3 annual points.\n"
            "2. **The placebo.** A within-year relabelling reproduces the spread 39% of the "
            "time — there is no robust edge to trade yet.\n"
            "3. **Survivorship.** The basket is winners that survived to 2026; the true "
            "premium (with delistings) is smaller."
        ),

        md(
            "## 7 — Going further\n\n"
            "- **A real multi-decade panel.** BGLN used Compustat back to the 1960s. A "
            "Compustat/CRSP pull (or EDGAR-derived full history) would give the statistical "
            "power yfinance cannot.\n"
            "- **The accrual decomposition.** Split cash-OP into its cash and accrual legs "
            "explicitly and confirm the accrual leg carries no premium.\n"
            "- **[Study 122 — Gross-Profitability](../../122-gross-profitability)**: the "
            "accrual-laden sibling this study is meant to beat.\n"
            "- **[Study 124 — Cash-Flow Yield](../../124-cash-flow-yield)**: another "
            "cash-based cross-sectional signal.\n\n"
            "*Think cash-OP still earns a robust premium in large-caps? Fork this, pull a "
            "multi-decade delisting-inclusive panel, and show a one-sample t > 2 that "
            "survives the placebo. That is the bar.*"
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
            "# Cash-Based Operating Profitability — a quantitative teardown\n"
            "### survivor basket × quintile sort × one-sample t × label-shuffle placebo × cash-vs-accrual head-to-head\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Cash beats accrual%3F: Mixed](https://img.shields.io/badge/Cash_beats_accrual%3F-Mixed-8b949e?style=flat-square)\n\n"
            "The quantitative companion to the "
            "[notebook for the curious](01_for_the_curious.ipynb) — same beats, every claim "
            "carrying its standard error. We test Ball-Gerakos-Linnainmaa-Nikolaev (2016): "
            "sort on cash-based operating profitability annually, long top quintile vs short "
            "bottom quintile, and contrast head-to-head with the accrual-laden GP/A of "
            "[Study 122](../../122-gross-profitability).\n\n"
            "> **Not investment advice.** Real data: yfinance annual statements + daily adj "
            "close, as-of 2026-06-26; the synthetic control and tests run offline and "
            "deterministic. Methods in [`docs/references.md`](../docs/references.md), "
            "reproducible numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Limits named:** basket = large-caps still trading 2026 (survivorship); "
            "yfinance exposes ~5 fiscal years → **3 usable hedge years**."
        ),
        code(BOOT),

        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Cash-OP hedge **+{R['cop_mean']:.2f}%/yr**, one-sample "
            f"*t* = **+{R['cop_t']:.3f}** (< 2), placebo *p* = **{R['placebo_p']:.3f}**; "
            "only 3 years. |\n"
            f"| **Tradability** | `MIRAGE` | Sharpe **+{R['cop_sharpe']:.2f}**, hit "
            f"**{R['cop_hit']:.0f}%**, DD **{R['cop_dd']:.0f}%** — small-sample artefacts. |\n"
            f"| **Cash beats accrual?** | `MIXED` | cash-OP *t* **+{R['cop_t']:.2f}** vs "
            f"GP/A *t* **+{R['gpa_t']:.2f}**; right order, neither significant. |\n\n"
            "> BGLN's direction is real in the literature; on 3 years of survivors it shows "
            "the right sign and fails the robustness gate."
        ),

        md(
            "## 1 — The claim, steelmanned\n\n"
            "Let $\\text{cashOP}_{i,y}=(\\text{OpInc}_{i,y}+\\text{D\\&A}_{i,y}+"
            "\\Delta\\text{WC}_{i,y})/\\text{Assets}_{i,y}$ and "
            "$\\text{GPA}_{i,y}=\\text{GrossProfit}_{i,y}/\\text{Assets}_{i,y}$. BGLN (2016) "
            "assert:\n\n"
            "- **H1 (signal).** Top-quintile cash-OP earns more than bottom-quintile the "
            "following year.\n"
            "- **H2 (dominance).** The cash-OP spread exceeds the accrual-laden GP/A spread.\n"
            "- **H3 (tradable).** The premium survives realistic costs at low turnover.\n\n"
            "On this tape we find H1 *directionally* (t = +1.75 < 2), H2 *directionally* "
            "(cash > accrual), and H3 in principle (costs trivial) — but the panel is far "
            "too short to confirm any of them."
        ),

        md(
            "## 2 — So what? — the economic stakes\n\n"
            "Operating profitability is the *P* in the Fama-French five-factor RMW factor and "
            "the anchor of an entire quality-ETF ecosystem. If the cash construction "
            "dominates — for free, from the same filings — then the standard accrual-laden "
            "measure is leaving robustness on the table. BGLN's contribution is precisely to "
            "show *which* version of profitability to use."
        ),

        md(
            "## 3 — The protocol\n\n"
            "- **Signal.** cash-OP = (Operating Income + D&A + ΔWorking-capital) / Total "
            "Assets, from year-*y* yfinance statements. ΔWorking-capital is the accrual "
            "block the cash-flow statement reports directly.\n"
            "- **Lag.** One trading day after the fiscal-year-end is public → next 252-day "
            "forward total return. No same-bar fills, no look-ahead.\n"
            "- **Sort.** Top quintile long, bottom quintile short, equal-weight.\n"
            "- **Costs.** 4 × 10 bps × turnover (two legs, in+out) + 50 bps/yr borrow on the "
            "short leg.\n"
            "- **Inference.** One-sample *t* on the annual hedge series (panel too short for "
            "HAC) + a 500-draw within-year label-shuffle placebo.\n"
            "- **Control.** Synthetic panel with a tunable cash-OP premium; the engine must "
            "recover it monotonically.\n"
            "- **Limits.** Basket = large-caps still trading 2026 (survivorship); 3 usable "
            "hedge years."
        ),

        md("## 4 — The teardown"),
        md(
            "### 4a — Positive control: the engine is a faithful detector\n\n"
            "Sweep the planted premium negative→positive on a 300-firm / 18-year synthetic "
            "panel. The hedge mean should be monotone and cross zero at premium = 0."
        ),
        code(
            "ctrl = st.synthetic_control()\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREEN if m > 0 else RED for m in ctrl['hedge_mean_%']]\n"
            "ax.bar(ctrl['premium'].astype(str), ctrl['hedge_mean_%'], color=col)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for i, (m, t) in enumerate(zip(ctrl['hedge_mean_%'], ctrl['tstat'])):\n"
            "    ax.text(i, m + (0.6 if m >= 0 else -1.6), f't={t:+.0f}',\n"
            "            ha='center', va='bottom', fontsize=8)\n"
            "ax.set_xlabel('planted cash-OP premium'); ax.set_ylabel('hedge mean (%/yr)')\n"
            "ax.set_title('Positive control: hedge is monotone in planted premium')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(ctrl.round(3).to_string(index=False))"
        ),

        md("### 4b — Real tape: cash-OP hedge, its legs, and the head-to-head vs GP/A"),
        code(
            "if HAVE_REAL:\n"
            "    cop_m, cop_t = s_cop['mean']*100, s_cop['tstat']\n"
            "    net_m = s_net['mean']*100\n"
            "    gpa_m, gpa_t = s_gpa['mean']*100, s_gpa['tstat']\n"
            "    hi_m, lo_m, mk_m = s_hi['mean']*100, s_lo['mean']*100, s_mkt['mean']*100\n"
            "    hedge_series = H['hedge']\n"
            "else:\n"
            "    cop_m, cop_t, net_m = R['cop_mean'], R['cop_t'], R['cop_net']\n"
            "    gpa_m, gpa_t = R['gpa_mean'], R['gpa_t']\n"
            "    hi_m, lo_m, mk_m = R['high_mean'], R['low_mean'], R['mkt_mean']\n"
            "    hedge_series = None\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "axes[0].bar(['Cash-OP', 'GP/A'], [cop_m, gpa_m], color=[AMBER, GREY], width=0.5)\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_ylabel('hedge mean (%/yr)')\n"
            "axes[0].set_title(f'Cash beats accrual: t {cop_t:+.2f} vs {gpa_t:+.2f}')\n"
            "if hedge_series is not None:\n"
            "    yrs = hedge_series.index.astype(int); hv = hedge_series.values*100\n"
            "    axes[1].bar(yrs, hv, color=[GREEN if v > 0 else RED for v in hv])\n"
            "    axes[1].axhline(0, c='k', lw=1)\n"
            "    axes[1].set_xticks(yrs)\n"
            "else:\n"
            "    axes[1].text(0.5, 0.5, f'Hedge {cop_m:+.1f}%/yr\\nt={cop_t:+.2f}',\n"
            "                 ha='center', va='center', transform=axes[1].transAxes)\n"
            "axes[1].set_xlabel('year'); axes[1].set_ylabel('cash-OP hedge (%/yr)')\n"
            "axes[1].set_title(f'Year-by-year | mean {cop_m:+.1f}% net {net_m:+.1f}%')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Cash-OP: gross {cop_m:+.2f}%/yr | net {net_m:+.2f}%/yr | t={cop_t:+.3f}"
            " | GP/A {gpa_m:+.2f}%/yr t={gpa_t:+.3f}')"
        ),
        md(
            f"> Cash-OP hedge **+{R['cop_mean']:.2f}%/yr** gross / **+{R['cop_net']:.2f}%/yr** "
            f"net (one-sample *t* = **+{R['cop_t']:.3f}**) vs accrual-laden GP/A "
            f"**+{R['gpa_mean']:.2f}%/yr** (*t* = +{R['gpa_t']:.2f}). The high cash-OP leg "
            f"beats the EW market by **+{R['high_vs_mkt']:.1f}%/yr** — but on "
            f"{R['n_years']} points."
        ),

        md(
            "### 4c — The decisive robustness check: a within-year label-shuffle placebo\n\n"
            "Shuffle the cash-OP labels *within each year* (preserving each year's marginal "
            "return and signal distributions) and recompute the hedge mean, 500 times. If the "
            "real spread is genuine it should sit far in the tail of this null."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.placebo_pvalue(cop, fwd, n_shuffles=500)\n"
            "    real_m, null_m, null_s, pval = (pl['real_mean']*100, pl['null_mean']*100,\n"
            "                                    pl['null_std']*100, pl['p_value'])\n"
            "    rng = np.random.default_rng(521); draws = []\n"
            "    for _ in range(500):\n"
            "        shuf = cop.copy()\n"
            "        for y in shuf.index:\n"
            "            v = shuf.loc[y].to_numpy(); m = np.isfinite(v)\n"
            "            p = v.copy(); p[m] = rng.permutation(v[m]); shuf.loc[y] = p\n"
            "        h = st.quintile_hedge(shuf, fwd)\n"
            "        if not h.empty: draws.append(h['hedge'].mean()*100)\n"
            "    draws = np.array(draws)\n"
            "else:\n"
            "    real_m, null_m, null_s, pval = (R['placebo_real'], R['placebo_null'],\n"
            "                                    R['placebo_std'], R['placebo_p'])\n"
            "    draws = np.random.default_rng(521).normal(null_m, null_s, 500)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "ax.hist(draws, bins=30, color=GREY, alpha=0.7, label='shuffled null')\n"
            "ax.axvline(real_m, c=RED, lw=2, label=f'real {real_m:+.1f}%/yr')\n"
            "ax.set_xlabel('hedge mean under shuffled labels (%/yr)'); ax.set_ylabel('count')\n"
            "ax.set_title(f'Label-shuffle placebo: p = {pval:.3f} (signal NOT in the tail)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Real {real_m:+.2f}%/yr | null {null_m:+.2f}% (std {null_s:.2f}) | p = {pval:.3f}')"
        ),
        md(
            f"> The placebo *p* = **{R['placebo_p']:.3f}**: the real spread is reproduced by a "
            "random within-year relabelling **39% of the time**. This is the decisive "
            "robustness number, and the signal **fails** it — confirming the verdict is "
            "WEAK, not REAL."
        ),

        md(
            "## 5 — The verdict\n\n"
            f"- **Signal `WEAK`** — cash-OP +{R['cop_mean']:.2f}%/yr, one-sample *t* = "
            f"+{R['cop_t']:.3f} (< 2), placebo *p* = {R['placebo_p']:.3f}, {R['n_years']} "
            "years. Literature prior (BGLN 2016) prevents `NONE`; the tape clears neither "
            "the t-bar nor the placebo.\n"
            f"- **Tradability `MIRAGE`** — Sharpe +{R['cop_sharpe']:.2f}, hit "
            f"{R['cop_hit']:.0f}%, DD {R['cop_dd']:.0f}% are small-sample illusions; "
            f"net of {R['cost']:.2f}%/yr costs the spread is still chance.\n"
            f"- **Cash beats accrual `MIXED`** — right order (t +{R['cop_t']:.2f} vs "
            f"+{R['gpa_t']:.2f}) but neither significant on 3 points.\n"
            "- **Survivorship `NAMED`** — basket = large-caps still trading 2026; upper bound."
        ),

        md(
            "## 6 — Could you trade it?\n\n"
            "Costs are trivial (annual rebalance, ~43% turnover, 0.67%/yr all-in). The "
            "barriers are entirely statistical:\n\n"
            "1. **Three usable years.** yfinance's ~5-year statement window, minus the two "
            "years without a forward return, leaves 3 annual hedge observations.\n"
            "2. **The placebo fails.** No robust spread to capture yet.\n"
            "3. **Survivorship.** Winners-only basket inflates every leg."
        ),
        code(
            "# Turnover profile (cash-OP top quintile, real basket if cached).\n"
            "if HAVE_REAL:\n"
            "    turn = H['turnover'].dropna()*100\n"
            "    fig, ax = plt.subplots(figsize=(8.5, 3.8))\n"
            "    ax.bar(turn.index.astype(int), turn.values, color=AMBER)\n"
            "    ax.axhline(turn.mean(), ls='--', c=RED, lw=1.5, label=f'mean {turn.mean():.0f}%')\n"
            "    ax.set_xticks(turn.index.astype(int))\n"
            "    ax.set_ylabel('top-quintile turnover (%/yr)'); ax.set_xlabel('year')\n"
            "    ax.set_title('Annual rebalancing turnover is modest'); ax.legend()\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'Mean turnover: {turn.mean():.1f}%/yr | mean cost: {H[\"cost\"].mean()*100:.2f}%/yr')\n"
            "else:\n"
            "    print(f'Frozen: turnover {R[\"turnover\"]:.1f}%/yr | cost {R[\"cost\"]:.2f}%/yr')"
        ),

        md(
            "## 7 — Going further\n\n"
            "- **A multi-decade panel.** BGLN used Compustat since the 1960s. The binding "
            "constraint here is purely yfinance's short history.\n"
            "- **Decompose the accrual leg.** Split cash-OP into cash and accrual components "
            "and verify the accrual part earns nothing — the heart of the BGLN result.\n"
            "- **Sector-neutral sort.** Profitability levels vary by industry; neutralising "
            "removes a confound.\n"
            "- **[Study 122 — Gross-Profitability](../../122-gross-profitability)**: the "
            "accrual-laden sibling.\n"
            "- **[Study 124 — Cash-Flow Yield](../../124-cash-flow-yield)**: a related "
            "cash-based valuation signal.\n\n"
            "*Think cash-OP earns a robust premium in large-caps post-2016? Fork this, pull a "
            "delisting-inclusive multi-decade panel, and show a one-sample t > 2 that "
            "survives the placebo. That is the bar.*"
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
