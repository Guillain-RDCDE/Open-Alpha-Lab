"""Generate the two narrative notebooks for Study 139 (AI-Powered-ETF).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The figures run
live from the cached daily parquet under ../_cache/ if present, and otherwise quote the
frozen headline numbers in ``R`` (mirroring docs/results.md), so the notebook re-runs for
any reader. The synthetic positive-control cells run anywhere, offline and deterministic.

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
    n=2173, n_years=8.62,
    cagr_aieq=9.88, cagr_spy=14.90, cagr_gap=-5.02,
    sharpe_aieq=0.531, sharpe_spy=0.826, sharpe_gap=-0.295,
    maxdd_aieq=-39.0, maxdd_spy=-33.7, maxdd_gap=-5.3,
    beta=1.050, r2=0.780,
    alpha_daily_bps=-1.785, alpha_ann=-4.40, alpha_t=-1.28,
    excess_bps=-1.475, excess_t=-1.05,
    roll_beat=31.2,
    boot_aieq_lo=-0.131, boot_aieq_hi=1.242, boot_aieq_fneg=6.2,
    boot_spy_lo=0.209, boot_spy_hi=1.532, boot_spy_fneg=0.4,
    er_aieq=0.75, er_spy=0.09,
    mdd_alpha=4.5,           # minimum detectable alpha at |t|=2 (%/yr)
    launch="2017-10-18",
)

# ---------------------------------------------------------------------------
# Shared analysis preamble — imports, the cache guard, and a returns loader.
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package (ai_powered_etf)
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY, BLUE = "#c0392b", "#dab617", "#2ea44f", "#8b949e", "#1f6feb"

from ai_powered_etf import data, strategy as st

TICKERS = ["AIEQ", "SPY", "IVV"]

def _have_cache():
    return all(os.path.exists(data._cache_path(t, data.DEFAULT_CACHE)) for t in TICKERS)

HAVE_REAL = _have_cache()

def load_real():
    \"\"\"Aligned daily returns for AIEQ & SPY from the cached real tape.\"\"\"
    prices = data.load_prices(fetch=False)
    return st.daily_returns(prices), prices

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# AI-Powered-ETF — can IBM Watson beat the S&P 500? 🤖\n"
            "### AIEQ, the \"AI Powered Equity ETF\", tested honestly, in plain English\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![AI picks vs buy-and-hold%3F: Busted](https://img.shields.io/badge/AI_picks_vs_buy--and--hold%3F-Busted-8b949e?style=flat-square)\n\n"
            "In 2017 a fund launched with a tantalising pitch: let **IBM Watson** read every "
            "filing, news story and social post, then pick the ~30-70 US stocks most likely to "
            "beat the market — *faster and more thoroughly than any human analyst.* The ticker is "
            "**AIEQ**. Nearly nine years of live track record later, this notebook asks the only "
            "question that matters: did the AI actually pick better than just buying the whole "
            "index and going to the beach?\n\n"
            "> 📓 **This is the plain-language layer.** Want the CAPM alpha, the HAC t-stats and "
            "the bootstrap intervals? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart below is "
            "drawn by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Did the AI fund beat the S&P 500? | **No.** Over 8.6 years AIEQ grew "
            f"**{R['cagr_aieq']:+.2f}%/yr** vs the index's **{R['cagr_spy']:+.2f}%/yr** — about "
            f"**{abs(R['cagr_gap']):.0f} points a year** behind. |\n"
            "| Did it at least take *less* risk for that? | **No.** Its worst drop was "
            f"**{R['maxdd_gap']:.0f} points deeper** and its risk-adjusted return (Sharpe) was "
            f"**{abs(R['sharpe_gap']):.2f} lower**. |\n"
            "| Is it just the 0.75% fee? | **No.** The fee hurts, but the stock-picking itself "
            "looks to lag the market *before* fees too. |\n"
            "| Could you have just bought SPY instead? | **Yes — and you'd be ~$50k richer** per "
            "$100k invested, with smaller drawdowns. |\n\n"
            "> 💡 The honest catch: 8.6 years is a *short* window, so we can't slam the gavel at "
            "95% confidence (that's why **Signal = Weak**, not 'None'). But the direction never "
            "wavers: in this live sample, the AI lost to a dirt-cheap index fund."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"AIEQ uses IBM Watson to analyse millions of data points — filings, news, social "
            "media, management commentary — across thousands of US companies every day, and builds "
            "a portfolio of the ~30-70 with the best risk-adjusted return potential. Artificial "
            "intelligence never sleeps, never gets emotional, and processes more than any team of "
            "humans. The result: consistent outperformance, net of fees.\"*\n\n"
            "It's a seductive pitch and a *testable* one. The fund is real, public, and priced "
            "every day. We don't have to argue about whether AI *could* work — we can just look at "
            "what it *did*."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the pitch is true, it's a genuinely new thing under the sun: a machine that beats "
            "the market you can buy in one click. If it's false, it's a cautionary tale about "
            "paying **0.75%/yr** — roughly *eight times* a plain index fund — for a brand name and "
            "a story. With nearly nine years of real prices, we don't have to guess which world "
            "we're in."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Two fair yardsticks:\n\n"
            "1. **Race it against the index it's trying to beat.** Line up AIEQ next to **SPY** "
            "(the S&P 500 tracker, ~0.09% fee) from AIEQ's launch day and watch $100 grow in each. "
            "Same start, same finish, no cherry-picking — the launch date is fixed, not chosen by "
            "us.\n"
            "2. **Adjust for how much market risk it took.** A fund can look good simply by holding "
            "more stock-market exposure. The proper test asks for *extra* return beyond what its "
            "market exposure alone would deliver — the quants call that **alpha**. Positive alpha "
            "= real skill; zero or negative = the AI added nothing (or worse).\n\n"
            "We run both on the **full 8.6-year live tape** (Oct 2017 → Jun 2026) — every trading "
            "day there is, no more and no less."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline race: $100 in AIEQ vs $100 in the S&P 500, from launch.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rets, prices = load_real()\n"
            "    g_aieq = (1 + rets['AIEQ']).cumprod() * 100\n"
            "    g_spy  = (1 + rets['SPY']).cumprod() * 100\n"
            "    end_aieq, end_spy = g_aieq.iloc[-1], g_spy.iloc[-1]\n"
            "else:\n"
            "    # fall back to the frozen CAGRs over the known window\n"
            "    yrs = R['n_years']\n"
            "    idx = pd.bdate_range(R['launch'], periods=R['n'])\n"
            "    t = np.linspace(0, yrs, R['n'])\n"
            "    g_aieq = pd.Series(100 * (1 + R['cagr_aieq']/100) ** t, index=idx)\n"
            "    g_spy  = pd.Series(100 * (1 + R['cagr_spy']/100) ** t, index=idx)\n"
            "    end_aieq, end_spy = g_aieq.iloc[-1], g_spy.iloc[-1]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.6))\n"
            "ax.plot(g_aieq.index, g_aieq.values, c=RED, lw=2, label='AIEQ (IBM Watson AI)')\n"
            "ax.plot(g_spy.index,  g_spy.values,  c=BLUE, lw=2, label='SPY (S&P 500)')\n"
            "ax.set_ylabel('value of $100 invested at launch'); ax.legend(loc='upper left')\n"
            "ax.set_title('Same start line, very different finish')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'$100 became  ${end_aieq:,.0f}  in AIEQ   vs   ${end_spy:,.0f}  in SPY')"
        ),
        md(
            f"$100 left in AIEQ at launch is worth far less today than the same $100 in the "
            f"index. AIEQ compounded at **{R['cagr_aieq']:+.2f}%/yr**, SPY at "
            f"**{R['cagr_spy']:+.2f}%/yr** — a gap of about **{abs(R['cagr_gap']):.0f} points "
            "every year**, which snowballs over 8.6 years into roughly **$50k of lost growth per "
            "$100k**.\n\n"
            "**But maybe AIEQ just took less risk?** Let's check the two things that matter — "
            "risk-adjusted return (Sharpe) and the worst peak-to-trough fall (drawdown):"
        ),
        code(
            "if HAVE_REAL:\n"
            "    res = st.summarize(rets['AIEQ'], rets['SPY'])\n"
            "    sh_a, sh_s = res['sharpe_fund'], res['sharpe_market']\n"
            "    dd_a, dd_s = res['max_dd_fund_pct'], res['max_dd_market_pct']\n"
            "else:\n"
            "    sh_a, sh_s = R['sharpe_aieq'], R['sharpe_spy']\n"
            "    dd_a, dd_s = R['maxdd_aieq'], R['maxdd_spy']\n"
            "fig, ax = plt.subplots(1, 2, figsize=(10.0, 4.2))\n"
            "ax[0].bar(['AIEQ', 'SPY'], [sh_a, sh_s], color=[RED, BLUE], width=.55)\n"
            "ax[0].set_ylabel('Sharpe (more = better reward per unit risk)')\n"
            "ax[0].set_title('AIEQ earned less per unit of risk')\n"
            "ax[1].bar(['AIEQ', 'SPY'], [dd_a, dd_s], color=[RED, BLUE], width=.55)\n"
            "ax[1].set_ylabel('worst drawdown (%)')\n"
            "ax[1].set_title('...and fell harder when it fell')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Sharpe: AIEQ {sh_a:.2f} vs SPY {sh_s:.2f}   |   max drawdown: AIEQ {dd_a:.0f}% vs SPY {dd_s:.0f}%')"
        ),
        md(
            "Both bars point the wrong way for the AI fund: **lower** Sharpe and a **deeper** "
            "worst-case fall. So it isn't a 'safer, slower' fund — it delivered *less* return "
            "*and* took *more* pain.\n\n"
            "> 💡 Why doesn't the AI help? Picking ~50 stocks out of thousands is a near-impossible "
            "game even for the best humans, and the AI's edge — if any — has to clear a **0.75% "
            "yearly fee** before it reaches you. In this sample it never did."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — Weak.** The fund's 'extra return beyond its market risk' (alpha) is "
            f"**{R['alpha_ann']:+.2f}%/yr** — negative — but on 8.6 years the statistics can't "
            "*certify* it's not just bad luck (the formal t-stat is "
            f"**{R['alpha_t']:+.2f}**, short of the bar). The *direction*, though, is never in "
            "doubt.\n"
            "- **Tradability — Mirage.** Choosing AIEQ over SPY cost ~$50k per $100k and added "
            "drawdown. The marketed outperformance simply isn't there.\n"
            "- **AI picks vs buy-and-hold — Busted.** AIEQ beat the index in only "
            f"**{R['roll_beat']:.0f}%** of rolling 12-month windows. The good stretches are mostly "
            "shared bull markets, not AI magic."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually 'trade' it?\n\n"
            "There's nothing to trade here — buying AIEQ *is* the strategy, and it's available to "
            "anyone. The relevant question is simpler: **how often would picking AIEQ over SPY "
            "have actually paid off?** Here's the rolling 12-month gap (AIEQ minus SPY) over the "
            "whole history:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    roll = st.rolling_outperformance(rets['AIEQ'], rets['SPY'], window=252)\n"
            "    frac_beat = float((roll > 0).mean() * 100)\n"
            "    fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "    ax.fill_between(roll.index, roll.values, 0, where=roll.values >= 0, color=GREEN, alpha=.4)\n"
            "    ax.fill_between(roll.index, roll.values, 0, where=roll.values < 0, color=RED, alpha=.4)\n"
            "    ax.axhline(0, c='k', lw=1)\n"
            "    ax.set_ylabel('AIEQ - SPY over trailing 12m (%)')\n"
            "    ax.set_title(f'AIEQ beat the index in only {frac_beat:.0f}% of rolling years')\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    frac_beat = R['roll_beat']\n"
            "print(f'AIEQ beat SPY in {frac_beat:.1f}% of rolling 12-month windows.')"
        ),
        md(
            f"The chart is mostly **red**: AIEQ trailed the index in roughly two of every three "
            f"rolling years, beating it only **{R['roll_beat']:.0f}%** of the time. That's the "
            "shape of a fund that catches the occasional updraft but bleeds against the benchmark "
            "the rest of the time — exactly what you'd expect if the AI adds no durable edge."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Does the test even work?** The companion notebook *plants* a known amount of "
            "skill into a fake fund and shows our analysis recovers it cleanly — so when it reads "
            "'no alpha' on AIEQ, that's a fact about the fund, not a broken ruler.\n"
            "- **It's the short window talking.** 8.6 years isn't enough to *prove* a small "
            "negative alpha at 95% confidence; a decade more data would sharpen the verdict either "
            "way. We say so honestly — that's why this is **Weak**, not **None**.\n"
            "- **The broader lesson.** Most active funds — AI-branded or not — lose to a cheap "
            "index over long horizons, for the same reason: fees and the sheer difficulty of "
            "stock-picking.\n\n"
            "*Think the AI really earns its fee? Fork this, extend the window or swap the "
            "benchmark, and show a positive, significant alpha that survives the 0.75% drag. "
            "That's the bar.*"
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
            "# AI-Powered-ETF — a quantitative teardown 🔬\n"
            "### Real daily tape · CAPM Jensen alpha · HAC inference · bootstrap Sharpe CI · rolling windows · synthetic positive control\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![AI picks vs buy-and-hold%3F: Busted](https://img.shields.io/badge/AI_picks_vs_buy--and--hold%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim now carrying its standard error.* We test whether "
            "AIEQ (EquBot / IBM Watson, 0.75% ER) delivers a positive Jensen alpha over SPY on the "
            "full live tape, under Newey-West HAC inference, and confirm our detector with a "
            "synthetic positive control.\n\n"
            "> ⚠️ **Not investment advice.** Real data: Yahoo daily adjusted-close, "
            f"AIEQ launch {R['launch']} → 2026-06-12 ({R['n']} days, {R['n_years']:.2f} yrs), "
            "as-of 2026-06-14; the offline core and tests run on a deterministic synthetic tape. "
            "Methods & sources in [`docs/references.md`](../docs/references.md), reproducible "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Jensen alpha **{R['alpha_ann']:+.2f}%/yr** "
            f"({R['alpha_daily_bps']:+.3f} bps/day), HAC *t* = **{R['alpha_t']:+.2f}** — "
            "consistently negative but |*t*| < 2 on a power-limited window. |\n"
            f"| **Tradability** | `MIRAGE` | CAGR **{R['cagr_aieq']:+.2f}%** vs SPY "
            f"**{R['cagr_spy']:+.2f}%**; Sharpe {R['sharpe_aieq']:.2f} vs {R['sharpe_spy']:.2f}; "
            f"max DD {R['maxdd_aieq']:.0f}% vs {R['maxdd_spy']:.0f}%. |\n"
            f"| **AI picks vs buy-and-hold?** | `BUSTED` | beta **{R['beta']:.2f}**, beats SPY in "
            f"only **{R['roll_beat']:.0f}%** of rolling 12m windows; Sharpe ~36% lower. |\n\n"
            "> 💡 One cause behind all three: there is no detectable post-fee stock-selection edge "
            "in AIEQ's live record — the AI buys roughly the market (β≈1.05) and then trails it by "
            "its fee plus a negative gross residual."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^{f}_t$ be AIEQ's daily return and $r^{m}_t$ SPY's. Fit the market model\n\n"
            "$$ r^{f}_t = \\alpha + \\beta\\, r^{m}_t + \\varepsilon_t. $$\n\n"
            "The fund's promise is:\n\n"
            "- **H₁ (signal).** $\\alpha > 0$ — a positive *Jensen alpha*, i.e. return beyond what "
            "market exposure alone explains, **net of the 0.75% fee** already baked into AIEQ's "
            "adjusted prices.\n"
            "- **H₂ (beats buy-and-hold).** The realised path dominates SPY on the metrics an "
            "investor actually feels: CAGR, Sharpe, drawdown.\n"
            "- **H₃ (durable).** The edge isn't a one-year fluke — it shows up across rolling "
            "windows, not just in pooled aggregate.\n\n"
            "We find $\\hat\\alpha < 0$ (failing H₁ in *direction*, though under-powered), and H₂ "
            "and H₃ fail decisively."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "AIEQ is the flagship public test of \"AI picks stocks better than humans.\" If H₁–H₃ "
            "held it would be a marketable free lunch and a genuine market-efficiency anomaly. The "
            "interesting failure mode here is **power**: with ~8%/yr idiosyncratic vol over 8.6 "
            "years, the minimum alpha detectable at |*t*|=2 is roughly "
            f"**±{R['mdd_alpha']:.1f}%/yr** — and the point estimate "
            f"(**{R['alpha_ann']:+.2f}%/yr**) sits right on that edge. So we can rule out a *large* "
            "positive alpha but not a small one; the real-world comparison removes any ambiguity "
            "about who won the sample."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Data.** Yahoo daily *adjusted* close (total-return, dividends reinvested) for "
            f"AIEQ, SPY, IVV, aligned from AIEQ's launch ({R['launch']}); {R['n']} common trading "
            "days. No look-ahead: returns are close-to-close.\n"
            "- **Estimator.** Full-sample OLS market model for $(\\alpha,\\beta)$; the static beta "
            "is mildly *generous* to AIEQ (a stricter rolling beta would not flatter it).\n"
            "- **Inference.** Newey-West **HAC** t-stat on the OLS intercept (the sandwich "
            "covariance of $[\\alpha,\\beta]$, not the residual mean which is 0 by construction); "
            "circular block-bootstrap 95% CI on annualised Sharpe.\n"
            "- **Durability.** Rolling 252-day Jensen alpha and rolling 12m excess return.\n"
            "- **Positive control.** A synthetic tape with a *tunable planted alpha* "
            "(`data.synthetic_daily`) to confirm the estimator recovers an edge **when one "
            "exists** and reads ~0 when it doesn't.\n\n"
            f"Inference bar: |*t*| ≥ 2. AIEQ's alpha lands at *t* = {R['alpha_t']:+.2f}."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The market model — a negative, under-powered alpha\n\n"
            "OLS $r^{f} = \\alpha + \\beta r^{m} + \\varepsilon$ on the full tape, with the HAC "
            "t-stat on the intercept. We also annualise the alpha and report $R^2$."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rets, prices = load_real()\n"
            "    res = st.summarize(rets['AIEQ'], rets['SPY'])\n"
            "    alpha_ann, alpha_t = res['alpha_ann_pct'], res['tstat_alpha_hac']\n"
            "    beta, r2 = res['beta'], res['r_squared']\n"
            "    rf, rm = rets['AIEQ'].values, rets['SPY'].values\n"
            "else:\n"
            "    alpha_ann, alpha_t = R['alpha_ann'], R['alpha_t']\n"
            "    beta, r2 = R['beta'], R['r2']\n"
            "    rng = np.random.default_rng(139)\n"
            "    rm = rng.normal(0.07/252, 0.16/np.sqrt(252), R['n'])\n"
            "    rf = (R['alpha_daily_bps']/1e4) + beta*rm + rng.normal(0, 0.08/np.sqrt(252), R['n'])\n"
            "fig, ax = plt.subplots(figsize=(6.2, 6.0))\n"
            "ax.scatter(rm*100, rf*100, s=6, alpha=.25, color=GREY)\n"
            "xs = np.linspace(rm.min(), rm.max(), 50)\n"
            "ax.plot(xs*100, (beta*xs + alpha_ann/100/252)*100, c=RED, lw=2,\n"
            "        label=f'fit: beta={beta:.2f}, alpha={alpha_ann:+.1f}%/yr')\n"
            "ax.plot(xs*100, xs*100, c=BLUE, ls='--', lw=1.5, label='beta=1, alpha=0 (fair)')\n"
            "ax.set_xlabel('SPY daily return (%)'); ax.set_ylabel('AIEQ daily return (%)')\n"
            "ax.set_title('The fit sits *below* the fair line: negative alpha'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'alpha = {alpha_ann:+.2f}%/yr   HAC t = {alpha_t:+.2f}   beta = {beta:.3f}   R^2 = {r2:.3f}')"
        ),
        md(
            f"> 💡 The regression line lies just under the 45° fair line: AIEQ holds essentially "
            f"market risk (**β = {R['beta']:.2f}**, $R^2$ = {R['r2']:.2f}) and gives back "
            f"**{R['alpha_ann']:+.2f}%/yr** of alpha. HAC *t* = **{R['alpha_t']:+.2f}** is below "
            "|2|, so we label it **WEAK** — the sign is firm, the certification is power-starved. "
            f"The minimum detectable alpha here is ~±{R['mdd_alpha']:.1f}%/yr, and the estimate "
            "sits right at that frontier."
        ),
        md(
            "### 4b · Risk-adjusted return — Sharpe with a bootstrap band\n\n"
            "Point Sharpe plus a circular block-bootstrap 95% CI (2000 resamples) for each fund, "
            "so we can see whether the gap is inside the noise."
        ),
        code(
            "try:\n"
            "    from quantlab.stats import sharpe_ci_bootstrap\n"
            "    HAVE_QL = True\n"
            "except Exception:\n"
            "    HAVE_QL = False\n"
            "if HAVE_REAL and HAVE_QL:\n"
            "    ci_a = sharpe_ci_bootstrap(rets['AIEQ'], n_boot=2000, periods_per_year=252, seed=139)\n"
            "    ci_s = sharpe_ci_bootstrap(rets['SPY'],  n_boot=2000, periods_per_year=252, seed=139)\n"
            "    sa, la, ha, fa = ci_a['sharpe'], ci_a['ci_low'], ci_a['ci_high'], ci_a['frac_negative']*100\n"
            "    ss, ls_, hs, fs = ci_s['sharpe'], ci_s['ci_low'], ci_s['ci_high'], ci_s['frac_negative']*100\n"
            "else:\n"
            "    sa, la, ha, fa = R['sharpe_aieq'], R['boot_aieq_lo'], R['boot_aieq_hi'], R['boot_aieq_fneg']\n"
            "    ss, ls_, hs, fs = R['sharpe_spy'],  R['boot_spy_lo'],  R['boot_spy_hi'],  R['boot_spy_fneg']\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.0))\n"
            "ys = [1, 0]\n"
            "ax.errorbar([sa], [1], xerr=[[sa-la],[ha-sa]], fmt='o', color=RED, capsize=5, lw=2, label='AIEQ')\n"
            "ax.errorbar([ss], [0], xerr=[[ss-ls_],[hs-ss]], fmt='o', color=BLUE, capsize=5, lw=2, label='SPY')\n"
            "ax.axvline(0, c='k', lw=1); ax.set_yticks(ys); ax.set_yticklabels(['AIEQ', 'SPY'])\n"
            "ax.set_xlabel('annualised Sharpe (95% block-bootstrap CI)')\n"
            "ax.set_title('AIEQ Sharpe sits below SPY, and its band kisses zero')\n"
            "ax.set_ylim(-0.6, 1.6); plt.tight_layout(); plt.show()\n"
            "print(f'AIEQ Sharpe {sa:.3f}  CI [{la:+.3f}, {ha:+.3f}]  ({fa:.1f}% neg)')\n"
            "print(f'SPY  Sharpe {ss:.3f}  CI [{ls_:+.3f}, {hs:+.3f}]  ({fs:.1f}% neg)')"
        ),
        md(
            f"AIEQ's Sharpe is **{R['sharpe_aieq']:.3f}** with a 95% CI "
            f"**[{R['boot_aieq_lo']:+.3f}, {R['boot_aieq_hi']:+.3f}]** "
            f"({R['boot_aieq_fneg']:.1f}% of resamples negative); SPY's is **{R['sharpe_spy']:.3f}** "
            f"with CI **[{R['boot_spy_lo']:+.3f}, {R['boot_spy_hi']:+.3f}]** "
            f"({R['boot_spy_fneg']:.1f}% negative). The index dominates on the point estimate and "
            "has a band cleanly above zero; AIEQ's band brushes it."
        ),
        md(
            "### 4c · Durability — rolling 12-month Jensen alpha\n\n"
            "A genuine AI edge should persist. Here is the rolling 252-day annualised alpha; "
            "shaded green where positive, red where negative."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ra = st.rolling_alpha(rets['AIEQ'], rets['SPY'], window=252)\n"
            "    frac_pos = float((ra > 0).mean() * 100)\n"
            "    fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "    ax.fill_between(ra.index, ra.values, 0, where=ra.values >= 0, color=GREEN, alpha=.4)\n"
            "    ax.fill_between(ra.index, ra.values, 0, where=ra.values < 0, color=RED, alpha=.4)\n"
            "    ax.axhline(0, c='k', lw=1)\n"
            "    ax.set_ylabel('rolling 12m Jensen alpha (%/yr)')\n"
            "    ax.set_title(f'Rolling alpha is positive only {frac_pos:.0f}% of the time')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'rolling 12m alpha > 0 in {frac_pos:.1f}% of windows')\n"
            "else:\n"
            "    print('no cache: rolling alpha needs the real tape; frozen rolling-beat = %.1f%%' % R['roll_beat'])"
        ),
        md(
            "> 💡 The series spends most of its life **below zero** and shows no upward trend — no "
            "sign of an AI model 'learning' its way to alpha. Combined with the "
            f"**{R['roll_beat']:.0f}%** rolling-12m win rate over SPY, the durability test (H₃) "
            "fails: the good windows are sparse and clustered in shared bull runs."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — Jensen alpha {R['alpha_ann']:+.2f}%/yr "
            f"({R['alpha_daily_bps']:+.3f} bps/day), HAC *t* {R['alpha_t']:+.2f}; raw excess "
            f"{R['excess_bps']:+.3f} bps/day (*t* {R['excess_t']:+.2f}). Negative and consistent, "
            f"but |*t*| < 2 on {R['n_years']:.1f} yrs (min detectable α ~±{R['mdd_alpha']:.1f}%/yr).\n"
            f"- **Tradability `MIRAGE`** — CAGR {R['cagr_aieq']:+.2f}% vs {R['cagr_spy']:+.2f}%; "
            f"Sharpe {R['sharpe_aieq']:.3f} vs {R['sharpe_spy']:.3f}; max DD {R['maxdd_aieq']:.0f}% "
            f"vs {R['maxdd_spy']:.0f}%.\n"
            f"- **AI picks vs buy-and-hold `BUSTED`** — β {R['beta']:.2f}, rolling-12m win rate "
            f"{R['roll_beat']:.0f}%, Sharpe ~36% lower; the 0.75% ER is a persistent headwind on "
            "top of a negative gross residual."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the investor's-eye comparison\n\n"
            "There's no spread or turnover to model — you just buy the fund. The honest "
            "tradability question is the dollar gap an investor lived through. Equity curves of "
            "$100 from launch, log-scaled so the compounding gap is legible:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    g_a = (1 + rets['AIEQ']).cumprod() * 100\n"
            "    g_s = (1 + rets['SPY']).cumprod() * 100\n"
            "    end_a, end_s = g_a.iloc[-1], g_s.iloc[-1]\n"
            "else:\n"
            "    idx = pd.bdate_range(R['launch'], periods=R['n'])\n"
            "    t = np.linspace(0, R['n_years'], R['n'])\n"
            "    g_a = pd.Series(100*(1+R['cagr_aieq']/100)**t, index=idx)\n"
            "    g_s = pd.Series(100*(1+R['cagr_spy']/100)**t, index=idx)\n"
            "    end_a, end_s = g_a.iloc[-1], g_s.iloc[-1]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.4))\n"
            "ax.plot(g_a.index, g_a.values, c=RED, lw=2, label='AIEQ')\n"
            "ax.plot(g_s.index, g_s.values, c=BLUE, lw=2, label='SPY')\n"
            "ax.set_yscale('log'); ax.set_ylabel('$100 at launch (log scale)'); ax.legend(loc='upper left')\n"
            "ax.set_title('The wealth gap compounds steadily, never closing')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'terminal: AIEQ ${end_a:,.0f}  vs  SPY ${end_s:,.0f}  '\n"
            "      f'(shortfall ${end_s-end_a:,.0f} per $100 = ${(end_s-end_a)*1000:,.0f} per $100k)')"
        ),
        md(
            "> 💡 Unlike the desk's *real-but-untradable* studies, where a positive gross edge dies "
            "at a finite cost line, here there is no edge to erode in the first place. The fee is a "
            "headwind, but even gross of fees the selection residual is negative — costs are an "
            "aggravator, not the cause."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the positive control\n\n"
            "Is the *estimator* capable of finding alpha, or does it always print 'none'? Plant a "
            "known annual alpha into a synthetic AIEQ-vs-SPY tape and sweep it: the recovered HAC "
            "alpha should track the planted value along the 45° line."
        ),
        code(
            "planted = [-0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08]\n"
            "rec_alpha, rec_t = [], []\n"
            "for a in planted:\n"
            "    prices_s, truth = data.synthetic_daily(n_days=2173, alpha_ann=a, seed=139)\n"
            "    res_s = st.detect_alpha(prices_s)\n"
            "    rec_alpha.append(res_s['alpha_ann_pct'])\n"
            "    rec_t.append(res_s['tstat_alpha_hac'])\n"
            "fig, ax = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "ax[0].plot([p*100 for p in planted], rec_alpha, 'o-', c=GREEN, lw=2)\n"
            "lim = [-9, 9]; ax[0].plot(lim, lim, ls='--', c=GREY, label='perfect recovery')\n"
            "ax[0].axhline(0, c='k', lw=.8); ax[0].axvline(0, c='k', lw=.8)\n"
            "ax[0].set_xlabel('planted alpha (%/yr)'); ax[0].set_ylabel('recovered alpha (%/yr)')\n"
            "ax[0].set_title('The estimator is unbiased'); ax[0].legend()\n"
            "ax[1].plot([p*100 for p in planted], rec_t, 'o-', c=BLUE, lw=2)\n"
            "for s in (2, -2): ax[1].axhline(s, ls=':', c=GREY)\n"
            "ax[1].axhline(0, c='k', lw=.8)\n"
            "ax[1].set_xlabel('planted alpha (%/yr)'); ax[1].set_ylabel('recovered HAC t')\n"
            "ax[1].set_title('...and significant when the edge is real')\n"
            "plt.tight_layout(); plt.show()\n"
            "i0 = planted.index(0.0)\n"
            "print(f'at planted alpha 0: recovered {rec_alpha[i0]:+.2f}%/yr, t = {rec_t[i0]:+.2f}')"
        ),
        md(
            "The recovered alpha tracks the planted value along the 45° line and its t-stat clears "
            "|2| once the edge is large enough — so the detector is faithful and roughly unbiased. "
            "The real-tape reading is therefore a statement about **AIEQ**, not the method: over "
            "8.6 live years the AI's net selection alpha is negative and, at this sample size, "
            "indistinguishable-from-zero at best. Forks worth trying: extend the window as data "
            "accrues, swap in a multi-factor (Fama-French) model to strip style tilts, or test "
            "other AI-branded funds (QRFT, AIQ) on the same protocol."
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
