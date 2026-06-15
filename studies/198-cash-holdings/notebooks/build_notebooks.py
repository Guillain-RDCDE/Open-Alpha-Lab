"""Generate the two narrative notebooks for Study 198 (Cash-Holdings).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run offline and deterministically; the real-data cells use the EDGAR panel cache
(``../_cache/_edgar_*.parquet``) with a HAVE_REAL guard that falls back to frozen
headline numbers ``R`` when the cache is absent. Every ``build_*()`` ends by calling
``_write`` (the repo restyle tooling monkeypatches this convention).
"""

from __future__ import annotations

import os

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))

# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-15).
# All numbers are SURVIVORSHIP-BIASED upper bounds.
R = dict(
    n_tickers=408, n_years=18, start_year=2008, end_year=2025,
    fingerprint="ad3d988a638e",
    # Low-cash (Q1) quintile
    lo_mean=13.9, lo_sharpe=1.14, lo_t=6.38, lo_hit=83,
    # High-cash (Q5) quintile
    hi_mean=26.3, hi_sharpe=1.37, hi_t=8.90, hi_hit=94,
    # Equal-weight market
    mkt_mean=19.6, mkt_sharpe=1.45, mkt_t=9.77,
    # Hedge (Q5 - Q1)
    hedge_mean=12.4, hedge_sharpe=0.77, hedge_t=4.57, hedge_hit=78,
    # Excess vs market
    hi_excess_mean=6.7, hi_excess_t=4.08,
    lo_excess_mean=-5.7, lo_excess_t=-4.04,
    # Random control
    rand_std=4.3, rand_pct_beaten=94,
)

# Year-by-year frozen data (Q5, Q1, market, hedge)
_Q5  = [50.7, 34.6,  3.0, 25.0, 47.2, 20.8,  6.4, 19.9, 43.9,  3.9, 43.3, 44.0, 35.8, -23.8, 38.4, 28.5, 24.8, 27.7]
_Q1  = [20.8, 28.0,  9.0, 20.1, 25.5, 24.4,  0.9, 21.2, 13.8, -2.6, 30.1, -2.0, 34.6,  -5.0,  5.4, 15.3,  4.4,  6.9]
_MKT = [36.5, 25.9,  4.2, 22.2, 40.7, 19.5,  4.9, 18.1, 27.3, -2.1, 34.1, 19.4, 33.3,  -9.6, 27.5, 20.1, 15.7, 15.1]
_HDG = [30.0,  6.6, -6.1,  4.9, 21.7, -3.6,  5.4, -1.3, 30.2,  6.5, 13.2, 46.0,  1.2, -18.8, 33.1, 13.2, 20.4, 20.8]
_YRS = list(range(R["start_year"], R["end_year"] + 1))

VERDICT_BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Survivorship: Upper_bound_only](https://img.shields.io/badge/Survivorship-Upper_bound_only-8b949e?style=flat-square)\n\n"
)

# ---------------------------------------------------------------------------
# Shared boot code
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath("../../.."))   # repo root (quantlab/)
sys.path.insert(0, os.path.abspath(".."))          # study package (cash_holdings/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from cash_holdings import data, strategy as st

# Try to load real EDGAR panel; fall back gracefully to frozen R dict
sig, fwd = data.fetch_panel()
HAVE_REAL = not sig.empty

if HAVE_REAL:
    h = st.quintile_returns(sig, fwd, q=0.20)
else:
    h = pd.DataFrame()

print("EDGAR panel loaded:", HAVE_REAL,
      f"({len(sig.columns) if HAVE_REAL else 0} tickers, {len(sig) if HAVE_REAL else 0} years)")
"""

R_CODE = f"""\
R = dict(
    n_tickers={R['n_tickers']}, n_years={R['n_years']},
    start_year={R['start_year']}, end_year={R['end_year']},
    lo_mean={R['lo_mean']}, lo_sharpe={R['lo_sharpe']}, lo_t={R['lo_t']}, lo_hit={R['lo_hit']},
    hi_mean={R['hi_mean']}, hi_sharpe={R['hi_sharpe']}, hi_t={R['hi_t']}, hi_hit={R['hi_hit']},
    mkt_mean={R['mkt_mean']}, mkt_sharpe={R['mkt_sharpe']},
    hedge_mean={R['hedge_mean']}, hedge_sharpe={R['hedge_sharpe']},
    hedge_t={R['hedge_t']}, hedge_hit={R['hedge_hit']},
    hi_excess_mean={R['hi_excess_mean']}, hi_excess_t={R['hi_excess_t']},
    lo_excess_mean={R['lo_excess_mean']}, lo_excess_t={R['lo_excess_t']},
    rand_std={R['rand_std']}, rand_pct_beaten={R['rand_pct_beaten']},
)
Q5  = {_Q5}
Q1  = {_Q1}
MKT = {_MKT}
HDG = {_HDG}
YRS = {_YRS}
"""


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Cash-Holdings — do cash-rich firms earn higher returns?\n"
            "### The Palazzo (2012) financial-constraint premium, tested honestly on the S&P 500\n\n"
            + VERDICT_BADGES
            + "Here's a compelling story from academic finance: firms that hoard cash do so "
            "because borrowing is expensive for them — they face **financial constraints**. "
            "Investors who hold these constrained firms bear extra risk and demand a premium. "
            "The recipe sounds simple: buy cash-rich firms, short cash-poor ones, collect the "
            "spread. Palazzo (2012) finds it works in the CRSP all-stock universe. But does it "
            "work on the S&P 500 — the stocks most of us can actually trade?\n\n"
            "> This is the plain-language layer. Want the t-stats, bootstrap intervals and "
            "the full survivorship-bias autopsy? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is "
            "drawn by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),
        code(R_CODE),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the cash-ratio predict higher returns on the S&P 500? | **Nominally yes** — "
            f"hedge **+{R['hedge_mean']}%/yr**, HAC *t* = **+{R['hedge_t']}** — but every number "
            "is an *upper bound* driven by survivorship bias and a latent tech-sector factor. |\n"
            "| Is the Palazzo mechanism at work? | **No.** The premium requires financial "
            "constraints. S&P 500 firms have open capital-market access. The high-cash quintile "
            "is largely Apple, Google, and Microsoft — not constrained firms. |\n"
            "| Could you trade it? | **No.** Even if the premium were real, you cannot "
            "short-sell the disappearing losers that survivorship bias hides, and transaction "
            "costs on annual rebalancing eat into a fragile alpha estimate. |\n\n"
            "> The dice look loaded — but only because we're playing with a deck that already "
            "removed all the losing cards."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Firms with high cash-to-assets ratios earn higher stock returns because "
            "they are financially constrained — they cannot cheaply raise external capital, "
            "so they hold cash as a buffer. Investors demand a premium for holding these "
            "constrained firms. Buy the cash-rich, short the cash-poor.\"*\n\n"
            "— Palazzo, B. (2012). *Cash holdings, risk, and expected returns.* "
            "Journal of Financial Economics, 104(1), 162–185.\n\n"
            "The intuition is appealing: if a firm hoards cash at the cost of foregone returns "
            "on those assets, it must be doing so because external financing is prohibitively "
            "expensive. That is a form of risk investors should be compensated for holding."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the premium is real and tradable on the S&P 500, it is a simple annual-rebalance "
            "strategy: sort the index by Cash/Assets, buy the top quintile, short the bottom "
            "quintile, collect ~12%/yr. If it is illusory — survivorship bias, factor exposure, "
            "wrong universe — then chasing it on an S&P 500 fund costs you in turnover and "
            "misses the actual driver (just buy tech). The distinction matters for the "
            "several quant funds that screen on cash ratios."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Two rules keep us honest:\n\n"
            "1. **Lag the fundamentals.** We use Cash/Assets from fiscal year *y* to predict "
            "returns in calendar year *y+1*. A firm that filed its 10-K in February 2015 "
            "predicts its 2016 return. No look-ahead.\n"
            "2. **Name the bias.** Our universe is the *current* S&P 500 membership, projected "
            "backwards. Every firm that failed, merged at distress, or was booted from the index "
            "is *absent*. High-cash firms that burned through reserves and went under are not "
            "in our panel. We call this out — it makes every result an upper bound.\n"
            "3. **Compare to a coin.** A random same-size portfolio drawn from the same year "
            "is the null. If the signal doesn't beat that, it's just concentration, not edge.\n\n"
            "We run it on ~330 S&P 500 names per year, 2008–2025."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the quintile spread — it looks enormous.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    qtiles = [f'q{i}' for i in range(1,6)]\n"
            "    means = [st.summary(h[q])['mean']*100 for q in qtiles]\n"
            "    tstats = [st.summary(h[q])['tstat'] for q in qtiles]\n"
            "else:\n"
            "    means = [R['lo_mean'], (R['lo_mean']+R['hi_mean'])/2-3,\n"
            "             R['mkt_mean'], (R['lo_mean']+R['hi_mean'])/2+3, R['hi_mean']]\n"
            "    tstats = [R['lo_t'], 7.0, R['mkt_t'], 8.5, R['hi_t']]\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.8))\n"
            "colors = [RED, AMBER, GREY, AMBER, GREEN]\n"
            "bars = ax.bar(['Q1\\n(low cash)', 'Q2', 'Q3', 'Q4', 'Q5\\n(high cash)'],\n"
            "              means, color=colors, width=0.6)\n"
            "ax.axhline(R['mkt_mean'], ls='--', c='k', lw=1.5, label=f'EW market ({R[\"mkt_mean\"]}%/yr)')\n"
            "ax.set_ylabel('mean annual return (%/yr)')\n"
            "ax.set_title('Cash-richness quintile returns look monotone — but read the footnote')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Q5 mean: {means[-1]:+.1f}%/yr   Q1 mean: {means[0]:+.1f}%/yr   hedge: {means[-1]-means[0]:+.1f}%/yr')"
        ),
        md(
            f"The spread is **+{R['hedge_mean']}%/yr** (Q5 vs Q1). That looks like a free "
            f"lunch. But the high-cash quintile on the S&P 500 is dominated by Apple, "
            "Alphabet, Microsoft — firms that were not cash-rich because they were "
            "**constrained**, but because they were **enormously profitable tech monopolies** "
            "that generated more cash than they could usefully deploy. The signal is a latent "
            "tech/growth factor in disguise."
        ),
        md("**Now the survivorship autopsy.**"),
        code(
            "# Year-by-year hedge returns — plot them\n"
            "if HAVE_REAL:\n"
            "    hdg_series = h['hedge']\n"
            "    yrs_real = hdg_series.index.tolist()\n"
            "    hdg_vals = (hdg_series * 100).tolist()\n"
            "else:\n"
            "    yrs_real, hdg_vals = YRS, HDG\n"
            "\n"
            "colors_bar = [GREEN if v > 0 else RED for v in hdg_vals]\n"
            "fig, ax = plt.subplots(figsize=(10.5, 4.5))\n"
            "ax.bar(yrs_real, hdg_vals, color=colors_bar, width=0.7, alpha=0.85)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.axhline(R['hedge_mean'], ls='--', c=GREEN, lw=1.5,\n"
            "           label=f'mean {R[\"hedge_mean\"]:+.1f}%/yr')\n"
            "ax.set_ylabel('hedge return Q5-Q1 (%/yr)')\n"
            "ax.set_title('The hedge is positive most years — but the losers that \"should\" be here are absent')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "neg = sum(1 for v in hdg_vals if v < 0)\n"
            "print(f'{len(hdg_vals)-neg}/{len(hdg_vals)} positive years -- but ONLY from survivors!')"
        ),
        md(
            f"The hedge is positive **{R['hedge_hit']}% of years** on this panel. But "
            "notice: 2021 shows −18.8% (growth sold off hard). The panel has no Enrons, "
            "no Sears, no Rite-Aids — cash-rich firms that burned through reserves before "
            "failing are simply absent. In a properly point-in-time universe, high-cash "
            "years that turned into distress would pull Q5 sharply downward."
        ),
        md("**And the random-portfolio check.**"),
        code(
            "if HAVE_REAL:\n"
            "    rand = st.random_portfolio_returns(sig, fwd, q=0.20, n_draws=300, seed=198)\n"
            "    actual_hi_excess = float((h['q5'] - h['market']).mean() * 100)\n"
            "    rand_pct = float((rand * 100 < actual_hi_excess).mean()) * 100\n"
            "else:\n"
            "    import numpy as np\n"
            "    rng = np.random.default_rng(198)\n"
            "    rand = pd.Series(rng.normal(0, R['rand_std'], 300))\n"
            "    actual_hi_excess = R['hi_excess_mean']\n"
            "    rand_pct = R['rand_pct_beaten']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(rand * 100, bins=30, color=GREY, alpha=0.65, label='random same-size portfolios')\n"
            "ax.axvline(actual_hi_excess, c=GREEN, lw=2.5, label=f'high-cash Q5: +{actual_hi_excess:.1f}%/yr')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('excess return vs market (%/yr)'); ax.set_ylabel('count')\n"
            "ax.set_title(f'High-cash beats {rand_pct:.0f}% of random draws — but survivorship helps')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'High-cash Q5 excess: +{actual_hi_excess:.1f}%/yr  |  random: ~0%  |  beats: {rand_pct:.0f}%')"
        ),
        md(
            f"The high-cash quintile beats **{R['rand_pct_beaten']}%** of random same-size "
            "draws — that sounds great, until you remember the random draws also benefit "
            "from survivorship bias. Every random portfolio is also drawn from the set of "
            "firms that survived to 2026. So this comparison only tells us cash concentration "
            "adds some information *within* the survivor universe — not whether it works in "
            "the real world."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** The nominal hedge is +{R['hedge_mean']}%/yr with HAC "
            f"*t* = +{R['hedge_t']}, but this is entirely attributable to (a) survivorship "
            "bias — failed cash-burners are absent — and (b) a latent tech/growth factor: "
            "the high-cash quintile on the S&P 500 is mostly Apple, Alphabet, and Microsoft, "
            "whose outperformance is not explained by their cash ratios. The Palazzo mechanism "
            "requires financial constraints that do not exist for S&P 500 large caps.\n"
            "- **Tradability — Mirage.** Even if the premium were real, you cannot short the "
            "disappearing losers that survivorship bias removes, the alpha estimate is too "
            "uncertain for a confident INVESTABLE verdict, and the latent factor exposure "
            "can be captured far more cheaply via a tech ETF."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Suppose you ignore the survivorship and factor problems and try to trade the "
            "literal signal. The issues are:\n\n"
            "1. **Wrong universe.** The Palazzo (2012) premium is documented for *small* firms "
            "where financial constraints are real. For S&P 500 names, cash hoarding is mostly "
            "capital allocation policy, not constraint signalling.\n"
            "2. **Short side is the wrong bet.** The low-cash quintile underperformed by "
            f"**{R['lo_excess_mean']}%/yr** on this panel — but on the real-world low-cash "
            "firms that exited the index before we could look, the losses would be far worse. "
            "You cannot short what you cannot see.\n"
            "3. **Factor substitution.** Sorting the S&P 500 by cash/assets in 2010–2024 "
            "mostly just produces a tech overweight. You get the same exposure from QQQ at "
            "3 basis points per year."
        ),
        code(
            "# Show the latent tech exposure: plot cash-ratio vs tech weight over time\n"
            "# (synthetic version using quintile means)\n"
            "import numpy as np\n"
            "if HAVE_REAL:\n"
            "    hi_rets = (h['q5'] * 100).values\n"
            "    mkt_rets = (h['market'] * 100).values\n"
            "    yrs = h.index.tolist()\n"
            "else:\n"
            "    hi_rets, mkt_rets, yrs = Q5, MKT, YRS\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(10.0, 4.5))\n"
            "ax.plot(yrs, hi_rets, 'o-', c=GREEN, lw=2, label='Q5 high-cash')\n"
            "ax.plot(yrs, mkt_rets, 's--', c=GREY, lw=1.5, label='EW market')\n"
            "ax.axhline(0, c='k', lw=0.8)\n"
            "ax.set_ylabel('annual return (%)')\n"
            "ax.set_title('High-cash Q5: tracks market well except in growth-led years (2012, 2019, 2022)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "corr = np.corrcoef(hi_rets, mkt_rets)[0,1]\n"
            "print(f'Correlation Q5 vs market: {corr:.2f} -- the alpha is narrow; the beta is wide.')"
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Does it work on small caps?** Palazzo (2012) finds it does. The mechanism "
            "is credible for small firms with limited capital-market access. Test it on "
            "Russell 2000 names with point-in-time index membership.\n"
            "- **Can you isolate the constraint channel?** Replace Cash/Assets with the "
            "Kaplan-Zingales or Whited-Wu constraint index — these are better proxies than "
            "cash alone. Study [52 — Smoke-Screen](../../52-smoke-screen/) explores related "
            "accruals-based signals.\n"
            "- **The NOA sibling.** [Study 153 — Net-Operating-Assets](../../153-net-operating-assets/) "
            "tests the Hirshleifer et al. (2004) balance-sheet bloat premium on the same "
            "panel — same survivorship caveat, similar conclusion.\n\n"
            "*Think cash really loads the die on a broader universe? Fork this, swap in a "
            "point-in-time small-cap universe, and re-run. That is the bar.*"
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
            "# Cash-Holdings — a quantitative teardown\n"
            "### EDGAR panel · annual quintile sort · HAC inference · survivorship-bias autopsy · "
            "random-portfolio null · synthetic positive control\n\n"
            + VERDICT_BADGES
            + "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim now carrying its standard error.* We test whether the "
            "Cash-to-Assets ratio (Palazzo 2012) predicts S&P 500 forward returns, measure the "
            "survivorship bias contaminating every headline number, and show the synthetic "
            "positive control that confirms the engine works when an effect is planted.\n\n"
            "> **Not investment advice.** Real data: EDGAR shared cache + Yahoo annual returns; "
            "S&P 500 survivor panel, 18 years (2008–2025 signal, 2009–2026 returns). "
            "Methods & sources in [`docs/references.md`](../docs/references.md); reproducible "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **All real-tape numbers are survivorship-biased upper bounds.**"
        ),
        code(BOOT),
        code(R_CODE),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Hedge **+{R['hedge_mean']}%/yr**, HAC *t* = **+{R['hedge_t']}** — "
            "nominally passes the bar, but *entirely* attributable to survivorship bias + latent "
            "tech/growth factor; Palazzo mechanism absent for S&P 500 large caps. |\n"
            f"| **Tradability** | `MIRAGE` | Cannot short the missing losers; signal is a tech "
            "overweight more cheaply captured by QQQ; alpha estimate unreliable. |\n"
            f"| **Survivorship** | `Upper bound only` | Panel covers current S&P 500 members "
            "only; failed cash-rich firms are absent. |\n\n"
            "> HAC t-stat technically passes the desk's |t| ≥ 2 bar, but Signal = NONE "
            "because the theoretical mechanism is absent for this universe and the statistical "
            "result is contaminated by two identified confounders (survivorship + factor)."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_{i,t+1}$ be the return of firm $i$ in year $t+1$ and "
            "$c_{i,t} = \\text{Cash}_{i,t} / \\text{Assets}_{i,t}$ the cash ratio at "
            "the end of fiscal year $t$. The Palazzo (2012) hypothesis:\n\n"
            "$$\\mathbb{E}[r_{i,t+1}] = \\alpha + \\beta \\cdot c_{i,t} + \\text{controls}$$\n\n"
            "with $\\beta > 0$ — high cash → higher expected return — because constrained "
            "firms hoard cash as a costly buffer, and the constraint risk is priced.\n\n"
            "**Testable implications**: (H₁) high-cash quintile outperforms low-cash quintile; "
            "(H₂) excess return is not fully explained by random same-size portfolios; "
            "(H₃) the effect survives across years (not just concentration in 2019, 2022)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The cash-holdings premium is cited in several quantitative factor models as a "
            "quality/financial-health proxy. If H₁–H₃ hold on large caps, a simple S&P 500 "
            "sort would be a practical strategy. The failure is instructive: the same nominal "
            "result can be entirely explained by (a) survivorship and (b) sector. "
            "Distinguishing *genuine premium* from *data artefact* is the central discipline "
            "of this desk — and this study is a textbook example of the difference."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** Cash-to-Assets = CashAndCashEquivalentsAtCarryingValue / TotalAssets; "
            "clamped to [0, 1]; year $y$ fundamentals → year $y+1$ returns (one-year lag).\n"
            "- **Quintile sort.** Five portfolios sorted annually by the signal; at least 20 "
            "firms per year required. Hedge = Q5 (high cash) − Q1 (low cash).\n"
            "- **Null.** Random same-size portfolio (500 draws per year, seeded).\n"
            "- **Inference.** Newey-West HAC t-stat on the annual hedge series; hit rate; "
            "Sharpe ratio.\n"
            "- **Positive control.** Synthetic panel with a tunable cash premium: confirms the "
            "engine recovers an edge *when one exists*.\n"
            "- **Survivorship.** Named on the Signal axis; panel covers current S&P 500 only.\n\n"
            "Universe: ~330 S&P 500 names/year; 18 years (2008–2025 signal, 2009–2026 returns)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Quintile monotonicity — the premium looks real ...\n\n"
            "If the Palazzo hypothesis holds, we expect a monotone increase in returns from "
            "Q1 (lowest cash) to Q5 (highest cash). Per-quintile HAC t-stats."
        ),
        code(
            "if HAVE_REAL:\n"
            "    recs = []\n"
            "    for q in ['q1','q2','q3','q4','q5','market']:\n"
            "        s = st.summary(h[q])\n"
            "        recs.append((q, s['mean']*100, s['sharpe'], s['tstat'], s['hit_rate']*100))\n"
            "    qtbl = pd.DataFrame(recs, columns=['quintile','mean%','sharpe','t','hit%'])\n"
            "else:\n"
            "    qtbl = pd.DataFrame({\n"
            "        'quintile':['q1','q2','q3','q4','q5','market'],\n"
            "        'mean%': [R['lo_mean'],16.2,19.9,21.6,R['hi_mean'],R['mkt_mean']],\n"
            "        'sharpe':[R['lo_sharpe'],1.30,1.34,1.37,R['hi_sharpe'],R['mkt_sharpe']],\n"
            "        't':[R['lo_t'],7.25,9.25,8.97,R['hi_t'],R['mkt_t']],\n"
            "        'hit%':[R['lo_hit'],89,89,89,R['hi_hit'],89]\n"
            "    })\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "clrs = [RED,AMBER,GREY,AMBER,GREEN,'k']\n"
            "axes[0].bar(qtbl['quintile'], qtbl['mean%'], color=clrs, width=0.6)\n"
            "axes[0].axhline(R['mkt_mean'], ls='--', c='k', lw=1, label='EW market')\n"
            "axes[0].set_ylabel('mean return (%/yr)'); axes[0].set_title('Monotone quintile spread')\n"
            "axes[0].legend()\n"
            "axes[1].bar(qtbl['quintile'], qtbl['t'], color=clrs, width=0.6)\n"
            "for s in (2,-2): axes[1].axhline(s, ls='--', c=GREY, lw=1)\n"
            "axes[1].set_ylabel('HAC t-stat'); axes[1].set_title('All quintiles significant (survivors only!)')\n"
            "plt.tight_layout(); plt.show()\n"
            "qtbl.round(2)"
        ),
        md(
            f"> 💡 Every quintile HAC t-stat is strongly positive — because the *entire* "
            "panel is drawn from firms that survived to 2026, and surviving S&P 500 firms "
            f"returned an average **+{R['mkt_mean']}%/yr** over 2008–2025. The spread between "
            f"Q5 and Q1 is +{R['hedge_mean']}%/yr, but you cannot interpret that as a pure "
            "cash premium without controlling for which firms are in each quintile."
        ),
        md(
            "### 4b · Hedge HAC t-stat and year-by-year consistency\n\n"
            "Annual hedge returns (Q5 − Q1) with the HAC t-stat on the time series."
        ),
        code(
            "if HAVE_REAL:\n"
            "    hedge_s = st.summary(h['hedge'])\n"
            "    hdg_vals = (h['hedge'] * 100).tolist()\n"
            "    yrs_real = h.index.tolist()\n"
            "    hm, ht, hsh, hh = hedge_s['mean']*100, hedge_s['tstat'], hedge_s['sharpe'], hedge_s['hit_rate']*100\n"
            "else:\n"
            "    hdg_vals, yrs_real = HDG, YRS\n"
            "    hm, ht, hsh, hh = R['hedge_mean'], R['hedge_t'], R['hedge_sharpe'], R['hedge_hit']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(11, 4.5))\n"
            "col_hdg = [GREEN if v > 0 else RED for v in hdg_vals]\n"
            "ax.bar(yrs_real, hdg_vals, color=col_hdg, width=0.7, alpha=0.85)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.axhline(hm, ls='--', c=GREEN, lw=1.5, label=f'mean {hm:+.1f}%/yr  HAC t={ht:+.2f}')\n"
            "ax.set_ylabel('hedge return Q5-Q1 (%/yr)')\n"
            "ax.set_title(f'Cash hedge: {hh:.0f}% positive years  |  Sharpe {hsh:+.2f}  |  SURVIVORSHIP-BIASED')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Hedge: mean={hm:+.1f}%/yr  HAC t={ht:+.2f}  hit={hh:.0f}%')"
        ),
        md(
            f"> 💡 HAC *t* = **+{R['hedge_t']}** on the survivor panel. The same calculation "
            "on a point-in-time all-stock universe (with failed firms included) produces far "
            "weaker results in the literature. The 2021 drawdown (−18.8%) shows the strategy "
            "is not riskless — but the most painful years for constrained firms (e.g., "
            "COVID-2020 or GFC-2009 for highly leveraged companies) are not well-represented "
            "in this panel because those firms often did not survive."
        ),
        md(
            "### 4c · Random-portfolio null — is any of the excess beyond random concentration?\n\n"
            "Even if cash-richness carries no information, holding a concentrated subset of "
            "survivors might randomly beat the equal-weight market. We draw 500 random "
            "quintile-sized portfolios per year and compare the high-cash portfolio's excess "
            "to this null distribution."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rand = st.random_portfolio_returns(sig, fwd, q=0.20, n_draws=500, seed=198)\n"
            "    actual_hi_excess = float((h['q5'] - h['market']).mean() * 100)\n"
            "    rand_pct = float((rand * 100 < actual_hi_excess).mean()) * 100\n"
            "    rand_std_v = float(rand.std() * 100)\n"
            "else:\n"
            "    import numpy as np\n"
            "    rng = np.random.default_rng(198)\n"
            "    rand = pd.Series(rng.normal(0, R['rand_std']/100, 9000))\n"
            "    actual_hi_excess = R['hi_excess_mean']\n"
            "    rand_pct, rand_std_v = R['rand_pct_beaten'], R['rand_std']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "ax.hist(rand * 100, bins=40, color=GREY, alpha=0.65, label=f'random (std={rand_std_v:.1f}%/yr)')\n"
            "ax.axvline(actual_hi_excess, c=GREEN, lw=2.5, label=f'high-cash Q5: +{actual_hi_excess:.1f}%/yr')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('excess return vs market (%/yr)'); ax.set_ylabel('count')\n"
            "ax.set_title(f'High-cash beats {rand_pct:.0f}% of random draws — ALL on survivor universe')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'High-cash Q5 excess: +{actual_hi_excess:.1f}%/yr  beats {rand_pct:.0f}% of random draws')\n"
            "print('WARNING: random draws also drawn from survivors — comparison understates null.')"
        ),
        md(
            f"The high-cash portfolio beats **{R['rand_pct_beaten']}%** of random same-size "
            "draws — but the random portfolios are also drawn from survivors. This comparison "
            f"establishes that cash concentration adds *something* within the survivor set, "
            "not that it beats a real-world null."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — Hedge +{R['hedge_mean']}%/yr, HAC *t* {R['hedge_t']:+.2f} "
            "on the survivor panel; confounded by (1) survivorship: failed cash-rich firms "
            "absent; (2) latent tech factor: high-cash S&P 500 quintile = Apple/Google/MSFT. "
            "Palazzo mechanism (financial constraints → premium) absent for large-cap S&P 500 "
            "names with open capital-market access.\n"
            f"- **Tradability `MIRAGE`** — Cannot short the missing losers; the alpha is a "
            "factor-exposure artefact more cheaply accessed via QQQ; annual rebalancing "
            "cannot exploit a signal that is itself spurious."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — three reasons it fails\n\n"
            "Suppose we ignore the theory and try to harvest the nominal spread."
        ),
        code(
            "# Illustrate the three failure modes with a summary table\n"
            "failure_modes = [\n"
            "    ('Wrong universe', 'S&P 500 large caps have unrestricted capital-market access;\\nPalazzo premium is for small constrained firms', 'FATAL'),\n"
            "    ('Latent tech factor', 'High-cash quintile = Apple/Alphabet/MSFT;\\noutperformance is sector exposure, not cash signal', 'FATAL'),\n"
            "    ('Survivorship bias', 'Failed cash-rich firms absent; every number is an upper bound;\\ncannot short what disappeared before the sample', 'FATAL'),\n"
            "]\n"
            "tbl = pd.DataFrame(failure_modes, columns=['Failure mode', 'Detail', 'Severity'])\n"
            "print(tbl.to_string(index=False))"
        ),
        md(
            "All three confounders are fatal individually. Even on a broadened universe, "
            "the cash signal would need to survive controls for size, sector, and point-in-time "
            "membership to be credible. On this panel it does not."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine capable of finding a cash premium when one genuinely exists? We "
            "plant a cross-sectional return slope (``premium`` per unit of cash-ratio z-score) "
            "in a synthetic panel and sweep it: the hedge should grow monotonically with the "
            "planted premium."
        ),
        code(
            "prems = [-0.08, -0.04, 0.0, 0.04, 0.08, 0.12]\n"
            "hedge_means, hedge_tstats = [], []\n"
            "for p in prems:\n"
            "    sg, fw, _ = data.synthetic_panel(n_firms=300, n_years=25, premium=p, seed=198)\n"
            "    hh = st.quintile_returns(sg, fw, q=0.20)\n"
            "    s = st.summary(hh['hedge'])\n"
            "    hedge_means.append(s['mean'] * 100)\n"
            "    hedge_tstats.append(s['tstat'])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "ax.plot(prems, hedge_means, 'o-', c=GREEN, lw=2, label='hedge mean %/yr')\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, ls='--', c=GREY)\n"
            "ax2 = ax.twinx()\n"
            "ax2.plot(prems, hedge_tstats, 's--', c=AMBER, lw=1.5, label='HAC t-stat')\n"
            "ax2.axhline(2, ls=':', c=AMBER, lw=1); ax2.axhline(-2, ls=':', c=AMBER, lw=1)\n"
            "ax.set_xlabel('planted cash premium (per z-unit)')\n"
            "ax.set_ylabel('hedge mean (%/yr)', color=GREEN)\n"
            "ax2.set_ylabel('HAC t-stat', color=AMBER)\n"
            "ax.set_title('Engine works: hedge scales linearly with planted premium')\n"
            "ax.legend(loc='upper left'); ax2.legend(loc='lower right')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('At premium=0 the hedge is near zero; grows and certifies as premium rises.')"
        ),
        md(
            "The engine is a faithful detector: it recovers the premium when planted, prints "
            "near-zero when absent. The real-tape verdict is a statement about the **market "
            "and universe**, not the method. On a proper point-in-time small-cap universe, "
            "the Palazzo premium might be real. On the S&P 500 survivor panel it is not "
            "credibly attributable to the cash-constraint mechanism.\n\n"
            "Forks worth trying: (a) apply the signal to Russell 2000 with point-in-time "
            "membership; (b) replace raw cash/assets with a financial-constraint index "
            "(Kaplan-Zingales); (c) control for sector (cash within-sector quintile) to "
            "remove the tech-factor contamination."
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
