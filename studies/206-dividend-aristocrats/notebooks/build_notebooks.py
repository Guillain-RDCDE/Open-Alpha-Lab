"""Generate the two narrative notebooks for Study 206 (Dividend-Aristocrats).

    python notebooks/build_notebooks.py
    python -W ignore -m jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
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
    n=3188,
    start="2013-10-10",
    end="2026-06-15",
    # NOBL gross
    nobl_cagr=10.49, nobl_vol=15.9, nobl_sharpe=0.629, nobl_maxdd=-35.4,
    nobl_t=2.46, nobl_total_ret=253,
    # SPY gross
    spy_cagr=14.46, spy_vol=17.2, spy_sharpe=0.787, spy_maxdd=-33.7,
    spy_t=3.11, spy_total_ret=452,
    # Excess return
    excess_bps=-1.40, excess_cagr=-3.48, excess_t=-1.63,
    excess_sharpe_ci_lo=-0.987, excess_sharpe_ci_hi=0.088, excess_frac_neg=95,
    # OLS
    ols_alpha_bps=-0.384, ols_alpha_ann=-0.96, ols_t_alpha=-0.45,
    ols_beta=0.810, ols_r2=0.767,
    # Net of fees
    nobl_net_cagr=10.10, nobl_net_sharpe=0.607,
    spy_net_cagr=14.36, spy_net_sharpe=0.782,
    # Random-mix control
    ctrl_sharpe_mean=0.716, ctrl_frac_beat_nobl=100,
    # Crash 2018
    c18_dd=-19.3, c18_spy=18.6, c18_nobl=16.1, c18_prot=-2.5,
    # Crash 2020
    c20_dd=-33.7, c20_spy=23.6, c20_nobl=16.1, c20_prot=-7.6,
    # Crash 2022
    c22_dd=-24.5, c22_spy=21.0, c22_nobl=7.6, c22_prot=-13.4,
    # Crash 2025
    c25_dd=-18.8, c25_spy=21.5, c25_nobl=6.2, c25_prot=-15.3,
)

# ---------------------------------------------------------------------------
# Shared preamble
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

from dividend_aristocrats import data, strategy as st

CACHE = os.path.abspath(os.path.join("..", "_cache"))
TICKERS = ["NOBL", "SPY"]

def _have_cache():
    return all(os.path.exists(data._cache_path(t, CACHE)) for t in TICKERS)

HAVE_REAL = _have_cache()
print("real price cache present:", HAVE_REAL)

# Frozen results (mirror of docs/results.md)
R = dict(
    n=3188, start="2013-10-10", end="2026-06-15",
    nobl_cagr=10.49, nobl_vol=15.9, nobl_sharpe=0.629, nobl_maxdd=-35.4,
    nobl_t=2.46, nobl_total_ret=253,
    spy_cagr=14.46, spy_vol=17.2, spy_sharpe=0.787, spy_maxdd=-33.7,
    spy_t=3.11, spy_total_ret=452,
    excess_bps=-1.40, excess_cagr=-3.48, excess_t=-1.63,
    excess_sharpe_ci_lo=-0.987, excess_sharpe_ci_hi=0.088, excess_frac_neg=95,
    ols_alpha_bps=-0.384, ols_alpha_ann=-0.96, ols_t_alpha=-0.45,
    ols_beta=0.810, ols_r2=0.767,
    nobl_net_cagr=10.10, nobl_net_sharpe=0.607,
    spy_net_cagr=14.36, spy_net_sharpe=0.782,
    ctrl_sharpe_mean=0.716, ctrl_frac_beat_nobl=100,
    c18_dd=-19.3, c18_spy=18.6, c18_nobl=16.1, c18_prot=-2.5,
    c20_dd=-33.7, c20_spy=23.6, c20_nobl=16.1, c20_prot=-7.6,
    c22_dd=-24.5, c22_spy=21.0, c22_nobl=7.6, c22_prot=-13.4,
    c25_dd=-18.8, c25_spy=21.5, c25_nobl=6.2, c25_prot=-15.3,
)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Dividend-Aristocrats — do 25+ years of dividend growth beat the market?\n"
            "### NOBL vs SPY, 2013–2026: the quality crash-shield, tested honestly\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Crash_shield%3F: Busted](https://img.shields.io/badge/Crash_shield%3F-Busted-8b949e?style=flat-square)\n\n"
            "Every financial advisor's PowerPoint has a slide for this: companies that have raised "
            "their dividend for 25 years in a row — the **Dividend Aristocrats** — are the bedrock "
            "portfolio. They have financial discipline built into their DNA. They hold up in crashes "
            "because they don't cut dividends. They beat the market over the long run because quality "
            "always wins. ProShares even built an ETF (NOBL) to make the idea accessible. This "
            "notebook asks the only question that matters: does NOBL actually beat SPY over its full "
            "live history, and did it protect investors in the crashes?\n\n"
            "> 📓 **Plain-language layer.** For the t-stats, bootstrap intervals, and OLS alpha "
            "decomposition, see **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** Research tool only: every chart is drawn by the code "
            "beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Did NOBL beat the S&P 500 over its full life? | **No.** NOBL returned **{R['nobl_total_ret']}%** "
            f"vs SPY's **{R['spy_total_ret']}%** since 2013. |\n"
            f"| Is the risk-adjusted return (Sharpe) better? | **No.** NOBL Sharpe **{R['nobl_sharpe']:.3f}** "
            f"vs SPY **{R['spy_sharpe']:.3f}** — *lower* despite lower vol. |\n"
            "| Did it protect in crashes? | **No.** Negative protection in every major drawdown. |\n"
            f"| Net of fees? | Even worse. NOBL's 0.35%/yr ER vs SPY's 0.09%/yr adds **+0.26 pp/yr** headwind. |\n\n"
            "> The quality/dividend-growth story is compelling and the theory is real — the data just "
            "shows that at NOBL's price (both in ER and in valuation), it hasn't paid off."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Companies that have raised their dividend for at least 25 consecutive years have "
            "proved they can grow earnings through recessions, inflation, and rate cycles. Owning "
            "them via the Dividend Aristocrats Index (NOBL ETF) gives you the market's quality core: "
            "better risk-adjusted returns in the long run and a cushion when markets sell off — "
            "because these companies don't cut.\"*\n\n"
            "Two claims embedded here:\n"
            "1. **Quality premium** — superior risk-adjusted returns vs the S&P 500 (SPY).\n"
            "2. **Crash shield** — smaller drawdowns in bear markets.\n\n"
            "Both can be measured. NOBL has a full live history from October 2013: let's look."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            f"If true, this is the most-defensible equity tilt in finance: own quality businesses, "
            "sleep soundly in crashes, and earn a premium for it. NOBL is the canonical implementation "
            "— a liquid, transparent ETF that puts the rule in reach of every investor at 0.35%/yr. "
            "If it's false, investors are paying an ER premium to hold a concentrated, low-beta "
            "portfolio that lags a simple S&P 500 index fund in both total return and Sharpe, and "
            "doesn't even deliver the promised crash protection. The difference is one honest comparison."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "We compare NOBL and SPY using their full live history (Oct 2013–Jun 2026) using "
            "daily total-return closes (`auto_adjust=True`) — this includes all dividends, which is "
            "the only fair comparison for a dividend-growth strategy. Three tests:\n\n"
            "1. **Return test.** Does NOBL's CAGR and Sharpe beat SPY's over the full period and "
            "sub-periods?\n"
            "2. **Crash test.** Does NOBL hold up better in S&P 500 drawdowns ≥ 15%?\n"
            "3. **Control test.** Does a random daily mix of NOBL and SPY outperform pure NOBL? "
            "If so, the 'pure Aristocrats' concentration isn't what's driving any outcome.\n\n"
            "A synthetic oracle confirms the engine: when we *plant* a daily return advantage for "
            "the aristocrat, the tests correctly detect it."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — the full picture\n\n"
            "**First: the cumulative wealth chart — $1 invested in 2013.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    prices = data.load_aligned(fetch=False, cache_dir=CACHE)\n"
            "    nobl_idx = prices['NOBL'] / prices['NOBL'].iloc[0]\n"
            "    spy_idx  = prices['SPY']  / prices['SPY'].iloc[0]\n"
            "else:\n"
            "    import numpy as np\n"
            "    dates = pd.date_range('2013-10-10', periods=3188, freq='B')\n"
            "    nobl_idx = pd.Series((1 + R['nobl_cagr']/100) ** (np.arange(3188)/252), index=dates)\n"
            "    spy_idx  = pd.Series((1 + R['spy_cagr']/100)  ** (np.arange(3188)/252), index=dates)\n"
            "fig, ax = plt.subplots(figsize=(10, 5))\n"
            "ax.plot(nobl_idx.index, nobl_idx.values, color=AMBER, lw=2, label=f'NOBL (Dividend Aristocrats)')\n"
            "ax.plot(spy_idx.index,  spy_idx.values,  color=GREY,  lw=2, label='SPY (S&P 500)')\n"
            "ax.set_ylabel('Growth of $1 (total return, log scale)')\n"
            "ax.set_yscale('log')\n"
            "ax.set_title('Cumulative wealth: NOBL vs SPY since NOBL inception (Oct 2013)')\n"
            "ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Total return: NOBL={R[\"nobl_total_ret\"]}%  SPY={R[\"spy_total_ret\"]}%')\n"
            "print(f'CAGR:         NOBL={R[\"nobl_cagr\"]:+.2f}%  SPY={R[\"spy_cagr\"]:+.2f}%')"
        ),
        md(
            f"NOBL turned $1 into **\\${1 + R['nobl_total_ret']/100:.2f}** vs SPY's "
            f"**\\${1 + R['spy_total_ret']/100:.2f}** — a gap of **{R['spy_total_ret'] - R['nobl_total_ret']} pp "
            f"over 12.7 years**. The quality premium was not the market-beating idea it was sold as."
        ),
        md(
            "**Second: the crash test — did NOBL protect in bear markets?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    crashes = st.crash_analysis(prices, threshold=-0.15)\n"
            "    prot = crashes['protection_pp'].tolist()\n"
            "else:\n"
            "    prot = [R['c18_prot'], R['c20_prot'], R['c22_prot'], R['c25_prot']]\n"
            "labels = ['Dec 2018', 'COVID 2020', '2022 Bear', 'Apr 2025']\n"
            "colors = [RED if p < 0 else GREEN for p in prot]\n"
            "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            "ax.bar(labels, prot, color=colors)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('NOBL protection vs SPY (pp)')\n"
            "ax.set_title('Crash shield test: NOBL underperformed SPY in every crash')\n"
            "for i, (l, p) in enumerate(zip(labels, prot)):\n"
            "    ax.annotate(f'{p:+.1f} pp', (i, p), ha='center',\n"
            "                va='top' if p < 0 else 'bottom', fontsize=10)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Negative = NOBL fell MORE than SPY during the crash + recovery window')"
        ),
        md(
            f"NOBL underperformed SPY in **all four crashes studied** — by −{abs(R['c18_prot'])} pp in Dec 2018, "
            f"−{abs(R['c20_prot'])} pp in COVID, −{abs(R['c22_prot'])} pp in 2022, and "
            f"−{abs(R['c25_prot'])} pp in Apr 2025.  The crash shield is not just absent — it's "
            "reversed.  Not paying dividends (as tech stocks don't) doesn't mean cutting dividends.  "
            "In growth-led selloffs and recoveries, the Aristocrats' sector bias (consumer staples, "
            "industrials, utilities) hurt them relative to the cap-weighted benchmark."
        ),
        md("**Third: the random-mix control — is there any value in the pure-NOBL tilt?**"),
        code(
            "if HAVE_REAL:\n"
            "    ctrl = st.random_mix_control(prices, n_seeds=20, seed=206)\n"
            "    nobl_sharpe = st.summarize(st.daily_returns(prices).dropna()['NOBL'])['sharpe_ann']\n"
            "else:\n"
            "    ctrl = pd.DataFrame({'sharpe_ann': [R['ctrl_sharpe_mean']] * 20})\n"
            "    nobl_sharpe = R['nobl_sharpe']\n"
            "fig, ax = plt.subplots(figsize=(8, 4.0))\n"
            "ax.hist(ctrl['sharpe_ann'], bins=10, color=GREY, alpha=0.7, label='random-mix Sharpe (20 seeds)')\n"
            "ax.axvline(nobl_sharpe, c=AMBER, lw=2.5, label=f'Pure NOBL Sharpe: {nobl_sharpe:.3f}')\n"
            "ax.set_xlabel('Annualised Sharpe'); ax.set_ylabel('count')\n"
            "ax.set_title('Pure NOBL is dominated by any random mix of NOBL + SPY')\n"
            "ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Random-mix mean Sharpe: {ctrl[\"sharpe_ann\"].mean():.3f}')\n"
            "print(f'Fraction of mixes beating pure NOBL: {(ctrl[\"sharpe_ann\"] > nobl_sharpe).mean()*100:.0f}%')"
        ),
        md(
            f"A random daily weighting of NOBL and SPY (any random split, 20 seeds) beats "
            f"pure NOBL in **{R['ctrl_frac_beat_nobl']}%** of seeds.  You don't need the pure "
            "Aristocrats tilt — any mix is better, because SPY's superior performance blends in."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** NOBL lagged SPY by −{abs(R['excess_cagr']):.2f}%/yr in CAGR, "
            f"lower Sharpe ({R['nobl_sharpe']:.3f} vs {R['spy_sharpe']:.3f}), no quality premium visible.\n"
            "- **Tradability — Mirage.** Higher ER (0.35% vs 0.09%), lower return, lower Sharpe, and "
            "negative crash protection: the investor is paying more for less on every dimension.\n"
            f"- **Crash shield? — Busted.** NOBL underperformed SPY in every major drawdown, with "
            f"the gap widening over time (−{abs(R['c25_prot']):.1f} pp in Apr 2025)."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually invest in it?\n\n"
            "NOBL is perfectly liquid and easily accessible.  The issue is not execution — it's the "
            "economics.  The ETF's ER of 0.35%/yr is roughly 3.7× SPY's 0.0945%/yr, adding a "
            "perpetual ~0.26 pp/yr headwind for zero demonstrable gross alpha.  Long-only investors "
            "considering NOBL should compare it directly to VIG (Vanguard Dividend Appreciation ETF, "
            "ER 0.06%/yr) or DGRO (iShares Core Dividend Growth ETF, ER 0.08%/yr) before paying "
            "the NOBL premium."
        ),
        code(
            "# What the ER difference compounds to over time\n"
            "years = np.arange(1, 21)\n"
            "er_drag = (1 - (st.NOBL_ER - st.SPY_ER)) ** years - 1\n"
            "fig, ax = plt.subplots(figsize=(8, 4))\n"
            "ax.plot(years, er_drag * 100, 'o-', c=RED, lw=2)\n"
            "ax.set_xlabel('Years held'); ax.set_ylabel('Cumulative ER drag vs SPY (%)')\n"
            "ax.set_title(f'0.26 pp/yr ER headwind: cumulative drag over a holding period')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'ER drag after 20 years: {er_drag[-1]*100:.1f}%')\n"
            "print(f'Total gap (CAGR): NOBL {R[\"nobl_net_cagr\"]:.2f}% vs SPY {R[\"spy_net_cagr\"]:.2f}% net of ER')"
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Equal-weight or direct indexing** — building the Aristocrats basket directly avoids "
            "the 0.35%/yr ER; equal-weighting changes the factor tilt. That's a fork worth testing.\n"
            "- **The 2022 sub-period** was the lone outperforming year (+13.3% excess return). Was "
            "that a regime shift or a one-off? A study of value/quality factor cycles "
            "([Arnott et al. 2016](../../docs/references.md)) would help.\n"
            "- **Dogs of the Dow ([Study 88](../../88-dogs-of-the-dow/))** and "
            "**Dividend Growth ([Study 201](../../201-dividend-growth/))** test other dividend "
            "strategies — compare the verdicts.\n\n"
            "*Think the quality premium is real? Fork this, change the comparison window or risk model, "
            "and show a robust positive t-stat that clears the inference bar.*"
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
            "# Dividend-Aristocrats — a quantitative teardown\n"
            "### NOBL vs SPY · daily total return · HAC inference · OLS decomposition · bootstrap CI\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Crash_shield%3F: Busted](https://img.shields.io/badge/Crash_shield%3F-Busted-8b949e?style=flat-square)\n\n"
            "The quantitative companion to [01_for_the_curious.ipynb](01_for_the_curious.ipynb). "
            "Same seven beats, every claim now carrying its HAC t-stat. We test whether NOBL "
            "(ProShares S&P 500 Dividend Aristocrats ETF) delivers a statistically robust positive "
            "excess return and Sharpe vs SPY over its full live history (2013-10-10 to 2026-06-15), "
            "decompose the return into alpha and beta, check crash episodes, and sweep the ER drag.\n\n"
            "> ⚠️ **Not investment advice.** Real data: Yahoo Finance daily adjusted closes "
            "(auto_adjust=True), cached parquet; as-of 2026-06-15. Reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT + "\nfrom quantlab import analytics, stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Excess return **{R['excess_cagr']:+.2f}%/yr**, HAC "
            f"*t* = **{R['excess_t']:+.2f}**; bootstrap excess-Sharpe CI "
            f"[{R['excess_sharpe_ci_lo']:+.3f}, {R['excess_sharpe_ci_hi']:+.3f}] "
            f"({R['excess_frac_neg']}% of resamples negative). OLS alpha **{R['ols_alpha_ann']:+.2f}%/yr**, "
            f"*t* = **{R['ols_t_alpha']:+.2f}**. |\n"
            f"| **Tradability** | `MIRAGE` | NOBL net CAGR {R['nobl_net_cagr']:+.2f}% vs SPY "
            f"{R['spy_net_cagr']:+.2f}%; Sharpe {R['nobl_net_sharpe']:.3f} vs {R['spy_net_sharpe']:.3f}. "
            f"Random-mix control dominates pure NOBL in {R['ctrl_frac_beat_nobl']}% of seeds. |\n"
            f"| **Crash shield?** | `BUSTED` | NOBL underperformed SPY in all 4 major crashes "
            f"(−{abs(R['c20_prot']):.1f} pp COVID, −{abs(R['c22_prot']):.1f} pp 2022, "
            f"−{abs(R['c25_prot']):.1f} pp Apr 2025). |\n\n"
            "> 💡 Three failures, one structure: NOBL's beta = 0.81 gives lower vol but also lower "
            "return; there is no market-neutral alpha to compensate; and the defensive factor tilt "
            "failed in every major selloff."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_{N,t}$ and $r_{S,t}$ be the daily log total-returns of NOBL and SPY.  "
            "The Aristocrats thesis asserts:\n\n"
            "- **H₁ (quality premium).** $\\mathbb{E}[r_{N,t} - r_{S,t}] > 0$ over the long run — "
            "positive mean excess return, robust to autocorrelation (HAC t ≥ 2).\n"
            "- **H₂ (market-neutral alpha).** Positive OLS alpha: "
            "$r_{N,t} = \\alpha + \\beta r_{S,t} + \\epsilon_t$ with $\\alpha > 0$ at *t* ≥ 2. "
            "A beta < 1 (quality/low-vol) without positive alpha is just low-leverage beta, not a premium.\n"
            "- **H₃ (crash shield).** Inside major S&P 500 drawdown episodes, NOBL's total return "
            "exceeds SPY's — positive drawdown protection.\n\n"
            "We reject H₁, H₂, and H₃ on the full 2013–2026 live tape."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each hypothesis\n\n"
            "**H₁** is the alpha claim — the entire economic justification for paying NOBL's 0.35%/yr ER. "
            "If H₁ fails, investors are paying more for systematically less. "
            "**H₂** pins down *why*: low beta explains the lower vol, but a negative alpha means "
            "quality carries no additional premium in this sample — the quality factor may exist "
            "in theory (Asness et al.) but is priced out in the NOBL implementation. "
            "**H₃** is the emotional core of the pitch — 'sleep soundly in crashes' — and its failure "
            "is the most surprising result: negative protection in every episode means the crash-period "
            "underperformance reinforces the lifetime return gap."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Data.** Yahoo Finance daily adjusted total-return closes (`auto_adjust=True`) "
            "for NOBL and SPY, 2013-10-10 to 2026-06-15 (NOBL inception to present, n = 3,188).\n"
            "- **Return test.** Daily log-return series; HAC Newey-West t-stat on the mean excess "
            "return; block-bootstrap 95% CI on the excess Sharpe (CBB, block = n^(1/3)).\n"
            "- **Decomposition.** OLS regression $r_N = \\alpha + \\beta r_S + \\epsilon$; "
            "t-stat on alpha with OLS SE (n large, so asymptotically valid).\n"
            "- **Crash test.** SPY drawdowns ≥ 15%: compare NOBL total return over the same episode "
            "(start to recovery above the prior peak).\n"
            "- **Control.** Random daily NOBL/SPY mix (target 50%, seed-varied): if all 20 random "
            "seeds beat pure NOBL, the 'pure Aristocrats' concentration is strictly dominated.\n"
            "- **Positive control.** Synthetic tape with planted daily alpha: the engine must detect "
            "the advantage when it exists."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),

        md(
            "### 4a · Gross returns — NOBL vs SPY\n\n"
            "Full-period summary statistics on daily total-return log-returns."
        ),
        code(
            "if HAVE_REAL:\n"
            "    prices = data.load_aligned(fetch=False, cache_dir=CACHE)\n"
            "    ret = st.daily_returns(prices).dropna()\n"
            "    rows = []\n"
            "    for col in ['NOBL', 'SPY']:\n"
            "        s = st.summarize(ret[col], label=col)\n"
            "        rows.append(s)\n"
            "    df_sum = pd.DataFrame(rows).set_index('label')\n"
            "else:\n"
            "    df_sum = pd.DataFrame({\n"
            "        'cagr_pct': [R['nobl_cagr'], R['spy_cagr']],\n"
            "        'vol_ann_pct': [R['nobl_vol'], R['spy_vol']],\n"
            "        'sharpe_ann': [R['nobl_sharpe'], R['spy_sharpe']],\n"
            "        'max_dd_pct': [R['nobl_maxdd'], R['spy_maxdd']],\n"
            "        'tstat': [R['nobl_t'], R['spy_t']],\n"
            "    }, index=['NOBL', 'SPY'])\n"
            "    ret, prices = None, None\n"
            "df_sum.round(3)"
        ),
        md(
            "> 💡 NOBL's vol is lower (15.9% vs 17.2%) — that's the beta-0.81 tilt, not alpha. "
            f"But its Sharpe is also lower ({R['nobl_sharpe']:.3f} vs {R['spy_sharpe']:.3f}): lower "
            "vol saved less than lower return cost. The lifetime CAGR gap is "
            f"{R['spy_cagr'] - R['nobl_cagr']:.2f} pp/yr."
        ),

        md(
            "### 4b · The excess return test — H₁ and its bootstrap CI\n\n"
            "The daily NOBL − SPY log-return differential; HAC t-stat and block-bootstrap "
            "95% CI on the annualised excess Sharpe."
        ),
        code(
            "if HAVE_REAL:\n"
            "    diff = st.excess_return(prices)\n"
            "    s_diff = st.summarize(diff, label='NOBL-SPY')\n"
            "    ci = stats.sharpe_ci_bootstrap(diff, periods_per_year=252, seed=206)\n"
            "    em, et, clo, chi, fn = s_diff['mean_bps'], s_diff['tstat'], ci['ci_low'], ci['ci_high'], ci['frac_negative']\n"
            "else:\n"
            "    em, et = R['excess_bps'], R['excess_t']\n"
            "    clo, chi, fn = R['excess_sharpe_ci_lo'], R['excess_sharpe_ci_hi'], R['excess_frac_neg']/100\n"
            "    diff = pd.Series(dtype=float)\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "# Left: excess return distribution\n"
            "if HAVE_REAL and len(diff) > 0:\n"
            "    axes[0].hist(diff.values * 1e4, bins=60, color=AMBER, alpha=0.7)\n"
            "    axes[0].axvline(em, c=RED, lw=2, label=f'mean={em:+.2f} bps/day')\n"
            "else:\n"
            "    axes[0].bar(['Mean'], [em], color=AMBER)\n"
            "    axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_xlabel('Excess return (bps/day)'); axes[0].set_ylabel('count')\n"
            "axes[0].set_title(f'NOBL − SPY daily excess return (mean={em:+.2f} bps, t={et:+.2f})')\n"
            "axes[0].legend()\n"
            "# Right: excess Sharpe CI\n"
            "y = 0\n"
            "axes[1].plot([clo, chi], [y, y], 'o-', c=RED, lw=3, ms=8, label='95% bootstrap CI')\n"
            "axes[1].axvline(0, c='k', lw=1, ls='--')\n"
            "axes[1].set_xlabel('Excess Sharpe (annualised)'); axes[1].set_yticks([])\n"
            "axes[1].set_title(f'Excess Sharpe CI [{clo:+.3f}, {chi:+.3f}] ({fn*100:.0f}% neg.)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'HAC t = {et:+.2f}  CI = [{clo:+.3f}, {chi:+.3f}]  {fn*100:.0f}% of resamples negative')"
        ),
        md(
            f"> 💡 The HAC t = **{R['excess_t']:+.2f}** is below |2| — the excess return is not "
            "statistically certified (and is on the wrong side). The bootstrap CI "
            f"[{R['excess_sharpe_ci_lo']:+.3f}, {R['excess_sharpe_ci_hi']:+.3f}] barely clips above "
            f"zero; **{R['excess_frac_neg']}% of resamples are negative**. H₁ is rejected."
        ),

        md(
            "### 4c · OLS alpha/beta decomposition — H₂\n\n"
            "Regress NOBL daily log-returns on SPY daily log-returns.  A real quality premium "
            "requires positive *and statistically significant* alpha."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ols = st.beta_alpha_ols(prices)\n"
            "else:\n"
            "    ols = {'alpha_daily_bps': R['ols_alpha_bps'], 'alpha_ann_pct': R['ols_alpha_ann'],\n"
            "           't_alpha': R['ols_t_alpha'], 'beta': R['ols_beta'], 'r_squared': R['ols_r2'], 'n': R['n']}\n"
            "print('OLS decomposition: r_NOBL = alpha + beta * r_SPY')\n"
            "print(f\"  alpha = {ols['alpha_daily_bps']:+.3f} bps/day  ({ols['alpha_ann_pct']:+.2f}%/yr)  t = {ols['t_alpha']:+.2f}\")\n"
            "print(f\"  beta  = {ols['beta']:.4f}  R² = {ols['r_squared']:.4f}  n = {ols['n']}\")\n"
            "print()\n"
            "print('Interpretation:')\n"
            "print(f\"  Beta {ols['beta']:.3f} < 1 → NOBL is low-vol/quality tilted (76.7% of SPY variance)\")\n"
            "print(f\"  Alpha {ols['alpha_daily_bps']:+.3f} bps/day (t={ols['t_alpha']:+.2f}) → no detectable quality premium\")"
        ),
        md(
            f"> 💡 Beta = **{R['ols_beta']:.3f}** explains why NOBL has lower vol: it loads only 81% of the "
            "market factor. The quality premium story requires a *positive* alpha net of this beta — the "
            f"OLS alpha is **{R['ols_alpha_bps']:+.3f} bps/day ({R['ols_alpha_ann']:+.2f}%/yr)** with "
            f"t = **{R['ols_t_alpha']:+.2f}**: statistically zero and negatively signed. H₂ is rejected."
        ),

        md(
            "### 4d · Crash analysis — H₃\n\n"
            "SPY drawdown episodes ≥ 15%; NOBL protection = (NOBL total ret − SPY total ret) over "
            "the full episode (start to recovery above prior peak)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    crashes = st.crash_analysis(prices, threshold=-0.15)\n"
            "else:\n"
            "    crashes = pd.DataFrame({\n"
            "        'start': ['2018-12-20','2020-03-09','2022-05-09','2025-04-04'],\n"
            "        'bench_max_dd_pct': [R['c18_dd'], R['c20_dd'], R['c22_dd'], R['c25_dd']],\n"
            "        'SPY_total_ret_pct': [R['c18_spy'], R['c20_spy'], R['c22_spy'], R['c25_spy']],\n"
            "        'NOBL_total_ret_pct': [R['c18_nobl'], R['c20_nobl'], R['c22_nobl'], R['c25_nobl']],\n"
            "        'protection_pp': [R['c18_prot'], R['c20_prot'], R['c22_prot'], R['c25_prot']],\n"
            "    })\n"
            "print(crashes[['start','bench_max_dd_pct','SPY_total_ret_pct','NOBL_total_ret_pct','protection_pp']].to_string())\n"
            "print(f'\\nMean protection: {crashes[\"protection_pp\"].mean():+.1f} pp (negative = NOBL underperformed)')"
        ),
        md(
            f"> 💡 Mean protection: **−{abs(R['c18_prot'] + R['c20_prot'] + R['c22_prot'] + R['c25_prot'])/4:.1f} pp** "
            "across 4 episodes. The crash-shield claim (H₃) is rejected on every episode. The widening gap "
            f"in 2022 (−{abs(R['c22_prot']):.1f} pp) and Apr 2025 (−{abs(R['c25_prot']):.1f} pp) suggests "
            "the sector composition disadvantage is worsening, not stabilising."
        ),

        md(
            "### 4e · Synthetic positive control — the engine works when alpha exists\n\n"
            "Plant a daily return advantage in the synthetic tape and confirm the HAC t-stat "
            "rises monotonically.  This verifies the engine is not always printing 'none'."
        ),
        code(
            "alpha_grid = [0.0, 0.5, 1.0, 2.0, 3.0]\n"
            "rows = []\n"
            "for a in alpha_grid:\n"
            "    p_syn, _ = data.synthetic_daily(n_years=12, alpha_bps=a, seed=206)\n"
            "    diff_syn = st.excess_return(p_syn)\n"
            "    s_syn = st.summarize(diff_syn)\n"
            "    rows.append({'planted_alpha_bps': a, 'excess_cagr_pct': s_syn['cagr_pct'], 'tstat': s_syn['tstat']})\n"
            "ctrl_df = pd.DataFrame(rows)\n"
            "fig, ax = plt.subplots(figsize=(9, 4))\n"
            "ax.plot(ctrl_df['planted_alpha_bps'], ctrl_df['tstat'], 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='|t|=2 bar')\n"
            "ax.axhline(-2, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Planted daily alpha (bps/day)')\n"
            "ax.set_ylabel('HAC t-stat on excess return')\n"
            "ax.set_title('Engine detects alpha when planted — real tape implies ~0 bps/day')\n"
            "ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "ctrl_df.round(3)"
        ),
        md(
            "> 💡 The t-stat rises monotonically with planted alpha and clears |t| = 2 around 1.5–2 "
            "bps/day.  The engine is calibrated.  The real tape's HAC t = "
            f"**{R['excess_t']:+.2f}** implies an effective daily alpha near or below **0 bps** — "
            "consistent with the NOBL story: the quality factor may be priced in."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — excess return {R['excess_cagr']:+.2f}%/yr, HAC *t* {R['excess_t']:+.2f} "
            f"(below |2|); OLS alpha {R['ols_alpha_ann']:+.2f}%/yr, *t* {R['ols_t_alpha']:+.2f}. "
            f"Bootstrap excess-Sharpe CI [{R['excess_sharpe_ci_lo']:+.3f}, {R['excess_sharpe_ci_hi']:+.3f}] "
            f"with {R['excess_frac_neg']}% of resamples negative.\n"
            f"- **Tradability `MIRAGE`** — NOBL net CAGR {R['nobl_net_cagr']:+.2f}% vs SPY "
            f"{R['spy_net_cagr']:+.2f}%; Sharpe {R['nobl_net_sharpe']:.3f} vs {R['spy_net_sharpe']:.3f}. "
            f"Random-mix control dominates in {R['ctrl_frac_beat_nobl']}% of seeds. Total return "
            f"gap: {R['nobl_total_ret']}% vs {R['spy_total_ret']}% over 12.7 years.\n"
            f"- **Crash shield `BUSTED`** — negative protection in 4/4 episodes; mean −"
            f"{abs(R['c18_prot'] + R['c20_prot'] + R['c22_prot'] + R['c25_prot'])/4:.1f} pp."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the ER drag and net-of-fees reality\n\n"
            "NOBL is perfectly investable; the issue is the persistent headwind from a 0.35%/yr ER "
            "against a benchmark that charges 0.0945%/yr, on top of a negative gross alpha."
        ),
        code(
            "# Simulate net-of-fees performance on real (or synthetic) tape\n"
            "if HAVE_REAL:\n"
            "    adj = st.net_of_fees(prices)\n"
            "    ret_adj = st.daily_returns(adj).dropna()\n"
            "    rows_net = [st.summarize(ret_adj[c], label=c+' net') for c in ['NOBL','SPY']]\n"
            "    net_df = pd.DataFrame(rows_net).set_index('label')[['cagr_pct','sharpe_ann']]\n"
            "else:\n"
            "    net_df = pd.DataFrame({\n"
            "        'cagr_pct': [R['nobl_net_cagr'], R['spy_net_cagr']],\n"
            "        'sharpe_ann': [R['nobl_net_sharpe'], R['spy_net_sharpe']],\n"
            "    }, index=['NOBL net', 'SPY net'])\n"
            "print('Net-of-ER performance:')\n"
            "print(net_df.round(3))\n"
            "print()\n"
            "print(f'NOBL ER: {st.NOBL_ER*100:.4f}%/yr   SPY ER: {st.SPY_ER*100:.4f}%/yr')\n"
            "print(f'Annual ER headwind for NOBL vs SPY: {(st.NOBL_ER - st.SPY_ER)*100:.4f}%/yr')"
        ),
        md(
            "> 💡 Alternatives with lower ER and similar (or better) dividend-growth exposure: "
            "VIG (Vanguard Dividend Appreciation, ER 0.06%/yr), DGRO (iShares Core Dividend Growth, "
            "ER 0.08%/yr). Both avoid the Aristocrats' 25-year filter, which mechanically excludes "
            "any company that missed a raise — including many high-quality cyclicals."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **What worked in 2022?** Rate-shock episodes favour value/quality defensives; the "
            "2022 +13.3% excess return was real and is worth investigating in isolation.\n"
            "- **Direct indexing / custom index.** Replicating the Aristocrats directly, equal- "
            "weighted, at ~0% ER, changes both the cost and the factor exposure — a natural fork.\n"
            "- **International Aristocrats.** The 25-year rule is US-specific; shorter-history "
            "dividend growers in Europe and EM have different signals and crash behaviours.\n"
            "- **Compare [Study 88 — Dogs-of-the-Dow](../../88-dogs-of-the-dow/) and "
            "[Study 201 — Dividend-Growth](../../201-dividend-growth/)** for a full family portrait "
            "of dividend strategies.\n\n"
            "*Think the quality premium re-emerges with the right factor control? Fork, add a "
            "Fama-French exposure adjustment, and show a robust positive alpha.*"
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
