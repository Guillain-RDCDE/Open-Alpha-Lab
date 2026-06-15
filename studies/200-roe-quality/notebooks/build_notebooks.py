"""Generate the two narrative notebooks for Study 200 (ROE-Quality).

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
# All numbers are SURVIVORSHIP-BIASED upper bounds (2009-2025, 17yr).
R = dict(
    n_tickers=403, n_years=17, start_year=2009, end_year=2025,
    # ROE quintile returns (%/yr)
    q1_mean=22.1, q2_mean=17.2, q3_mean=19.0, q4_mean=17.3, q5_mean=17.1,
    mkt_mean=18.1,
    q1_t=7.04, q5_t=7.78, mkt_t=9.82,
    q1_hit=82, q5_hit=76,
    # Hedge (Q5 - Q1)
    hedge_mean=-5.0, hedge_sharpe=-0.42, hedge_t=-1.48, hedge_hit=35,
    # Excess vs market
    hi_excess_mean=-1.0, hi_excess_t=-0.94,
    lo_excess_mean=4.0, lo_excess_t=1.58,
    # Random portfolio control
    rand_std=4.2, rand_pct_above_hi=40,
    # GP/Assets comparison (restricted universe, ~207 tickers)
    gp_hedge_mean=2.6, gp_hedge_t=0.62,
    gp_hi_excess=-2.5, gp_hi_excess_t=-0.89,
)

# Year-by-year frozen data (Q1, Q5, market, hedge) -- 2009 through 2025
_Q1  = [19.5, 0.7, 36.5, 58.5, 20.6, 5.0, 22.7, 25.3, -2.0, 33.0, 13.6, 30.1, -13.0, 50.1, 16.8, 29.6, 28.6]
_Q5  = [28.2, -0.1, 18.0, 31.2, 13.3, 4.2, 13.7, 32.4, -3.4, 38.5, 24.1, 36.0, -9.9, 27.1, 13.6, 14.5, 9.3]
_MKT = [24.3, 3.3, 22.7, 38.9, 18.7, 6.3, 18.7, 27.8, -4.0, 35.0, 17.9, 33.0, -8.8, 27.4, 18.7, 15.6, 12.7]
_HDG = [8.7, -0.8, -18.6, -27.3, -7.3, -0.9, -9.0, 7.1, -1.4, 5.5, 10.6, 5.9, 3.1, -23.0, -3.2, -15.1, -19.4]
_YRS = list(range(R["start_year"], R["end_year"] + 1))

VERDICT_BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![ROE_priced%3F: Yes--already_in](https://img.shields.io/badge/ROE_priced%3F-Yes--already_in-8b949e?style=flat-square)\n\n"
)

# ---------------------------------------------------------------------------
# Shared boot code
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath("../../.."))   # repo root (quantlab/)
sys.path.insert(0, os.path.abspath(".."))          # study package (roe_quality/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from roe_quality import data, strategy as st

# Try to load real EDGAR panel; fall back gracefully to frozen R dict
roe, gpr, fwd = data.fetch_panel()
HAVE_REAL = not roe.empty

if HAVE_REAL:
    roe_w = roe.loc[2009:]
    gpr_w = gpr.loc[2009:]
    fwd_w = fwd.loc[2009:]
    h_roe = st.quintile_returns(roe_w, fwd_w, q=0.20)
    h_gp  = st.quintile_returns(gpr_w, fwd_w, q=0.20)
else:
    h_roe = pd.DataFrame()
    h_gp  = pd.DataFrame()

print("EDGAR panel loaded:", HAVE_REAL,
      f"({roe.shape[1] if HAVE_REAL else 0} tickers, {len(roe) if HAVE_REAL else 0} years)")
"""

R_CODE = f"""\
R = dict(
    n_tickers={R['n_tickers']}, n_years={R['n_years']}, start_year={R['start_year']}, end_year={R['end_year']},
    q1_mean={R['q1_mean']}, q5_mean={R['q5_mean']}, mkt_mean={R['mkt_mean']},
    q1_t={R['q1_t']}, q5_t={R['q5_t']}, mkt_t={R['mkt_t']},
    q1_hit={R['q1_hit']}, q5_hit={R['q5_hit']},
    hedge_mean={R['hedge_mean']}, hedge_sharpe={R['hedge_sharpe']}, hedge_t={R['hedge_t']}, hedge_hit={R['hedge_hit']},
    hi_excess_mean={R['hi_excess_mean']}, hi_excess_t={R['hi_excess_t']},
    lo_excess_mean={R['lo_excess_mean']}, lo_excess_t={R['lo_excess_t']},
    rand_std={R['rand_std']}, rand_pct_above_hi={R['rand_pct_above_hi']},
    gp_hedge_mean={R['gp_hedge_mean']}, gp_hedge_t={R['gp_hedge_t']},
    gp_hi_excess={R['gp_hi_excess']}, gp_hi_excess_t={R['gp_hi_excess_t']},
)
Q1  = {_Q1}
Q5  = {_Q5}
MKT = {_MKT}
HDG = {_HDG}
YRS = {_YRS}
"""


def md(text: str):
    return new_markdown_cell(text)


def code(text: str):
    return new_code_cell(text)


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# ROE-Quality — do high-return-on-equity firms earn higher returns?\n"
            "### A quality-factor test on 17 years of S&P 500 fundamentals\n\n"
            + VERDICT_BADGES +
            "An old investment idea: companies that earn high returns on their shareholders' equity "
            "(ROE = net income / book equity) are **quality businesses** — efficient capital "
            "allocators — and should reward investors with premium future stock returns. "
            "It is a key leg of the famous **Quality-Minus-Junk** factor (Asness et al. 2019). "
            "But is ROE actually informative about *next year's* stock returns, or has the market "
            "already priced in the quality premium?\n\n"
            "> 📓 **This is the plain-language layer.** Want the t-stats, quintile tables and "
            "the synthetic positive control? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** Reproducible research: every chart is drawn by the "
            "code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\n" + R_CODE),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do high-ROE firms earn higher 1yr forward returns? | **No.** High-ROE (Q5) earns "
            f"**{R['q5_mean']:.1f}%/yr** vs **{R['q1_mean']:.1f}%/yr** for low-ROE (Q1). |\n"
            f"| Does high-ROE beat the market? | **No.** Q5 underperforms the equal-weight "
            f"market by **{R['hi_excess_mean']:.1f}%/yr** (*t* = {R['hi_excess_t']:+.2f}). |\n"
            f"| Does the hedge (Q5 − Q1) work? | **No.** "
            f"**{R['hedge_mean']:+.1f}%/yr** (*t* = {R['hedge_t']:+.2f}), hitting only "
            f"**{R['hedge_hit']}%** of years — the wrong sign of the theory. |\n"
            "| Is gross profitability (Novy-Marx) better? | **Also no**, on this panel. |\n"
            "| Could you trade it? | **No.** Negative point estimate, no statistical foundation. |\n\n"
            "> The ROE quality signal is **priced in** on the S&P 500 survivor universe: "
            "high-ROE names have already been bid up and subsequently underperform. "
            "The quality story, if real, lives in a broader or survivorship-free universe."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Firms that earn high returns on equity are quality businesses — efficient "
            "capital allocators with durable competitive advantages. Buy them; they will reward "
            "you with higher stock returns. Sell or avoid the 'junk': low-ROE, unprofitable, "
            "highly-leveraged firms that burn capital.\"* — the quality-investing thesis.\n\n"
            "This idea has serious backing: it underpins AQR's Quality-Minus-Junk factor "
            "(Asness et al. 2019), Novy-Marx's profitability premium (2013), and the Fama-French "
            "RMW (Robust Minus Weak profitability) factor. The claim is testable: build ROE "
            "from EDGAR fundamentals, sort S&P 500 names into quintiles each year, and measure "
            "whether the top quintile earns more next year."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If ROE reliably predicts forward returns, every fundamental investor should tilt "
            "toward high-quality (high-ROE) stocks. The quality factor would be an investable, "
            "systematic edge — a reason to favour profitable compounders over cheap junkers. "
            "If it doesn't, the quality narrative may be true in a philosophical sense while "
            "the premium is completely priced away inside a well-known index like the S&P 500."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "**The protocol:**\n\n"
            "1. **Build ROE** from EDGAR: NetIncomeLoss ÷ lagged StockholdersEquity, "
            "cross-sectionally winsorised at 1%/99% to reduce outlier leverage.\n"
            "2. **One-year reporting lag**: fiscal-year-y fundamentals predict calendar-year-y+1 "
            "returns, so we can't see the 10-K before it's filed.\n"
            "3. **Quintile sort**: five bins from low ROE (Q1) to high ROE (Q5); compute equal-weight "
            "annual returns within each bin and the equal-weight-market return.\n"
            "4. **Honest controls**: (a) the equal-weight market benchmark, (b) a random portfolio of "
            "the same size (500 draws × 17 years) — does high-ROE beat a coin?\n"
            "5. **Head-to-head**: gross profitability (GP/Assets, Novy-Marx 2013) on the same universe.\n"
            "6. **Survivorship bias named**: the EDGAR cache is current S&P 500 members projected "
            "backwards — every firm survived to 2026, so results are upper bounds."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown\n\n"
            "Let's see what the data actually say — quintile by quintile, year by year."
        ),
        code(
            "# The quintile return bar chart\n"
            "if HAVE_REAL:\n"
            "    q_means = [st.summary(h_roe[f'q{i}'])['mean']*100 for i in range(1,6)]\n"
            "    mkt_m = st.summary(h_roe['market'])['mean']*100\n"
            "else:\n"
            "    q_means = [R['q1_mean'], R['q2_mean'], R['q3_mean'], R['q4_mean'], R['q5_mean']]\n"
            "    mkt_m = R['mkt_mean']\n"
            "\n"
            "cols = [GREEN if q < mkt_m else RED for q in q_means]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.8))\n"
            "bars = ax.bar([f'Q{i}' for i in range(1,6)], q_means, color=cols, width=0.6)\n"
            "ax.axhline(mkt_m, ls='--', c=GREY, lw=1.5, label=f'EW market: {mkt_m:.1f}%/yr')\n"
            "ax.set_ylabel('forward 1yr return (%/yr)')\n"
            "ax.set_title('ROE quintiles: low-ROE (Q1) BEATS high-ROE (Q5) on S&P 500 survivors')\n"
            "ax.legend()\n"
            "for b, v in zip(bars, q_means):\n"
            "    ax.annotate(f'{v:.1f}%', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Q5 (high ROE) mean: {q_means[4]:.1f}%/yr  Q1 (low ROE) mean: {q_means[0]:.1f}%/yr')\n"
            "print(f'Hedge (Q5-Q1): {q_means[4]-q_means[0]:.1f}%/yr  -- the WRONG sign of the theory')"
        ),
        md(
            f"The chart is striking: **Q1 (low ROE) outperforms Q5 (high ROE)** by roughly "
            f"{abs(R['hedge_mean']):.1f}%/yr on average over {R['n_years']} years. The quality "
            "premium is not just absent — it is *reversed*. Low-quality firms (by ROE) beat "
            "high-quality firms inside this universe.\n\n"
            "> The explanation: S&P 500 is already a quality filter. Every firm in it survived, "
            "earned profits, and attracted analyst coverage. High-ROE names get bid up by quality "
            "investors until future returns are depressed; low-ROE names that *remain* in the "
            "index (despite poor historical ROE) are often cyclical recoveries or cheap assets "
            "that subsequently outperform."
        ),
        md("**Year-by-year: is there any consistency?**"),
        code(
            "yrs_arr = YRS\n"
            "q1_arr  = Q1\n"
            "q5_arr  = Q5\n"
            "if HAVE_REAL:\n"
            "    yrs_arr = [int(y) for y in h_roe.index]\n"
            "    q1_arr  = list(h_roe['q1']*100)\n"
            "    q5_arr  = list(h_roe['q5']*100)\n"
            "    hdg_arr = list(h_roe['hedge']*100)\n"
            "else:\n"
            "    hdg_arr = HDG\n"
            "\n"
            "fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)\n"
            "x = range(len(yrs_arr))\n"
            "ax1.bar(x, q1_arr, label='Q1 low-ROE', color=GREEN, alpha=0.7, width=0.4, align='edge')\n"
            "ax1.bar([i-0.4 for i in x], q5_arr, label='Q5 high-ROE', color=RED, alpha=0.7, width=0.4, align='edge')\n"
            "ax1.axhline(0, c='k', lw=1); ax1.set_ylabel('return (%/yr)'); ax1.legend()\n"
            "ax1.set_title('Year-by-year: Q1 and Q5 returns')\n"
            "\n"
            "cols_h = [GREEN if v > 0 else RED for v in hdg_arr]\n"
            "ax2.bar(x, hdg_arr, color=cols_h, width=0.65)\n"
            "ax2.axhline(0, c='k', lw=1); ax2.set_ylabel('hedge (Q5-Q1, %/yr)')\n"
            "ax2.set_title(f'Hedge: {sum(1 for v in hdg_arr if v>0)}/{len(hdg_arr)} positive years')\n"
            "ax2.set_xticks(list(x)); ax2.set_xticklabels(yrs_arr, rotation=45, ha='right')\n"
            "plt.tight_layout(); plt.show()\n"
            "n_pos = sum(1 for v in hdg_arr if v>0)\n"
            "print(f'Hedge positive in {n_pos}/{len(hdg_arr)} years ({n_pos/len(hdg_arr):.0%} hit rate)')"
        ),
        md(
            f"The hedge hits only **{R['hedge_hit']}% of years** (6 out of 17). The "
            "overwhelming direction is negative — low-ROE firms outperform high-ROE firms. "
            "Particularly large reverse-quality years: 2011, 2012, 2015, 2022, 2024, 2025."
        ),

        # ---- BEAT 5 — VERDICT ------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Hedge {R['hedge_mean']:+.1f}%/yr, HAC *t* = "
            f"{R['hedge_t']:+.2f} (wrong sign); high-ROE Q5 excess vs market "
            f"{R['hi_excess_mean']:+.1f}%/yr (*t* = {R['hi_excess_t']:+.2f}); "
            f"hit rate {R['hedge_hit']}%. The ROE quality premium is absent and reversed.\n"
            "- **Tradability — Mirage.** Negative point estimate, no statistical foundation. "
            "Annual rebalancing costs (50–70 names/leg) further worsen the economics.\n"
            "- **ROE is 'priced'?** Yes — inside the S&P 500 survivor universe. The quality "
            "premium, if real, requires a survivorship-clean, broader universe and likely "
            "factor-model neutralisation (size, sector, valuation).\n\n"
            "> This is an important lesson: a factor that works *in theory* and *in academic "
            "studies on broad universes* may be fully priced within a narrow, already-screened "
            "universe like the S&P 500. The index itself acts as a quality filter."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Even setting aside the negative signal, the practical challenges are severe:"
        ),
        code(
            "barriers = [\n"
            "    ('Wrong direction', f\"{R['hedge_mean']:+.1f}%/yr hedge\", RED),\n"
            "    ('No stat. foundation', f\"HAC t = {R['hedge_t']:+.2f}\", RED),\n"
            "    ('Transaction costs', '~50-70 names per leg, annual rebalance', AMBER),\n"
            "    ('Survivorship bias', 'S&P 500 is already quality-screened', AMBER),\n"
            "    ('Factor crowding', 'Quality factor widely known since 2013+', AMBER),\n"
            "]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 3.5))\n"
            "ax.barh([b[0] for b in barriers], [1]*len(barriers),\n"
            "        color=[b[2] for b in barriers], alpha=0.8, height=0.5)\n"
            "for i, (_, label, _) in enumerate(barriers):\n"
            "    ax.text(0.5, i, label, ha='center', va='center', fontsize=10, color='white', fontweight='bold')\n"
            "ax.set_xlim(0, 1); ax.axis('off')\n"
            "ax.set_title('Barriers to trading ROE-quality inside the S&P 500')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Verdict: not tradable.')"
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Where does quality actually work?** Asness et al. (2019) find QMJ "
            "is most powerful on broad universe (not just S&P 500) and when combined "
            "with a value tilt (high quality at a cheap price).\n"
            "- **Gross profitability (Novy-Marx 2013)**: the companion notebook shows "
            "that GP/Assets also fails to deliver a positive quality premium on this "
            "S&P 500 panel — same problem, different signal.\n"
            "- **Survivorship-free universe**: [Study 65 — Scorecard](../../65-scorecard/) "
            "and [Study 121 — Magic-Formula](../../121-magic-formula/) explore related "
            "quality ideas on the same EDGAR cache with explicit survivorship warnings.\n"
            "- **The Fama-French RMW factor**: built from a broad universe with "
            "short-selling; the Q5-Q1 long-only version inside the S&P 500 survivor "
            "set is a much weaker test of the same idea.\n\n"
            "*Think ROE quality actually works? Fork this, extend to a broader universe "
            "(e.g., Russell 1000 or all EDGAR filers), add a value screen, or "
            "neutralise market beta. That is the bar.*"
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
            "# ROE-Quality — a quantitative teardown\n"
            "### EDGAR fundamentals · quintile sort · HAC inference · survivorship-biased S&P 500\n\n"
            + VERDICT_BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim now carrying its standard error.* We build ROE "
            "(NetIncomeLoss / lagged StockholdersEquity, winsorised 1%/99%) from the EDGAR "
            "cache, sort ~300 S&P 500 names into quintiles, test whether the high-ROE quintile "
            "earns higher forward returns, compare to gross profitability (Novy-Marx 2013), "
            "and benchmark against a random-portfolio null. Window: 2009–2025 (17 years).\n\n"
            "> ⚠️ **Not investment advice.** EDGAR fundamentals, S&P 500 universe, "
            "as-of 2026-06-15; the offline core runs on a deterministic synthetic panel. "
            "Results are **survivorship-biased upper bounds**. "
            "Methods in [`docs/references.md`](../docs/references.md), reproducible numbers "
            "in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **`💡 In plain words`** notes translate each result back to intuition."
        ),
        code(BOOT + "\n" + R_CODE + "\nfrom quantlab import analytics, stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Hedge (Q5−Q1) **{R['hedge_mean']:+.1f}%/yr**, "
            f"HAC *t* = **{R['hedge_t']:+.2f}**; high-ROE Q5 excess "
            f"**{R['hi_excess_mean']:+.1f}%/yr** (*t* = {R['hi_excess_t']:+.2f}); "
            f"hit rate {R['hedge_hit']}%. Wrong sign of the theory. |\n"
            f"| **Tradability** | `MIRAGE` | Negative point estimate, no statistical "
            "foundation; annual rebalancing costs are additive headwinds. |\n"
            f"| **ROE priced?** | `YES` | High-ROE names in the S&P 500 survivor universe "
            "are already bid up; the quality premium lives in broader, survivorship-free "
            "universes, not inside an already-screened index. |\n\n"
            "> 💡 The S&P 500 is itself a quality filter — membership requires sustained "
            "earnings and market capitalisation. Sorting *within* it by ROE asks whether "
            "the best-of-the-best outperforms the merely-good, and the answer is no."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $\\text{ROE}_{i,y}$ = $\\text{NI}_{i,y}$ / $\\text{SE}_{i,y-1}$ "
            "(lagged book equity, winsorised). The quality hypothesis asserts:\n\n"
            "- **H₁ (signal).** $\\mathbb{E}[r_{i,y+1}]$ is monotone increasing in "
            "$\\text{ROE}_{i,y}$ rank — the top quintile earns higher forward returns than the "
            "bottom quintile.\n"
            "- **H₂ (outperforms market).** The high-ROE quintile earns more than the "
            "equal-weight market: a positive excess return, statistically significant.\n"
            "- **H₃ (beats random).** The high-ROE portfolio beats a random same-size "
            "draw from the universe — the ROE sort adds value beyond size effects.\n\n"
            "We reject H₁, H₂, and H₃ on the real EDGAR tape."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The quality/ROE thesis is one of the most widely cited in factor investing. "
            "If H₁–H₃ hold inside the S&P 500, every systematic equity strategy should "
            "include a quality tilt. The failure here isn't a rejection of quality investing "
            "broadly — it is a precise statement: **within the S&P 500 survivor universe, "
            "the ROE rank has already been incorporated into prices, and the residual signal "
            "is negative**. The factor may survive in broader, survivorship-clean, or "
            "factor-neutralised contexts."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** $\\text{ROE}_{i,y} = \\text{NetIncomeLoss}_{i,y} / "
            "\\text{StockholdersEquity}_{i,y-1}$. Non-positive lagged equity → NaN. "
            "Cross-sectional winsorisation at 1%/99%.\n"
            "- **Reporting lag.** Fundamentals from fiscal year $y$ → predict calendar-year "
            "$y+1$ returns (conservative: assume 10-K not actionable until the following year).\n"
            "- **Universe.** S&P 500 names in the shared EDGAR cache (survivorship-biased).\n"
            "- **Quintile sort.** Quintile 1 = lowest ROE, Quintile 5 = highest ROE.\n"
            "- **Baseline.** Equal-weight all firms with valid fundamentals that year.\n"
            "- **Random control.** 500 draws × 17 years of same-size random portfolios.\n"
            "- **Comparison.** GP/Assets (Novy-Marx 2013) on the same universe.\n"
            "- **Inference.** Newey-West HAC *t* on annual hedge and excess returns.\n"
            "- **Window.** 2009–2025 (17 years, enough EDGAR coverage), "
            f"~{R['n_tickers']} tickers with valid ROE data."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Quintile return table — no monotone pattern\n\n"
            "If ROE signals quality, returns should rise from Q1 to Q5. Instead they fall."
        ),
        code(
            "if HAVE_REAL:\n"
            "    q_data = []\n"
            "    for i in range(1, 6):\n"
            "        s = st.summary(h_roe[f'q{i}'])\n"
            "        q_data.append((f'Q{i}', s['mean']*100, s['sharpe'], s['tstat'], s['hit_rate']*100))\n"
            "    s_mkt = st.summary(h_roe['market'])\n"
            "    s_hdg = st.summary(h_roe['hedge'])\n"
            "    q_df = pd.DataFrame(q_data, columns=['quintile','mean%/yr','sharpe','HAC_t','hit%'])\n"
            "else:\n"
            "    q_df = pd.DataFrame({\n"
            "        'quintile': [f'Q{i}' for i in range(1,6)],\n"
            "        'mean%/yr': [R['q1_mean'], R['q2_mean'], R['q3_mean'], R['q4_mean'], R['q5_mean']],\n"
            "        'sharpe': [0.78, 1.00, 1.02, 0.84, 0.73],\n"
            "        'HAC_t': [R['q1_t'], 6.45, 6.44, 6.11, R['q5_t']],\n"
            "        'hit%': [R['q1_hit']]*4 + [R['q5_hit']],\n"
            "    })\n"
            "    s_mkt = {'mean': R['mkt_mean']/100, 'tstat': R['mkt_t']}\n"
            "    s_hdg = {'mean': R['hedge_mean']/100, 'sharpe': R['hedge_sharpe'], 'tstat': R['hedge_t']}\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "mkt_val = s_mkt['mean']*100 if HAVE_REAL else R['mkt_mean']\n"
            "colors = [GREEN if v >= mkt_val else RED for v in q_df['mean%/yr']]\n"
            "bars = ax.bar(q_df['quintile'], q_df['mean%/yr'], color=colors, width=0.6)\n"
            "ax.axhline(mkt_val, ls='--', c=GREY, lw=1.5, label=f'EW market: {mkt_val:.1f}%/yr')\n"
            "ax.set_ylabel('forward 1yr return (%/yr)')\n"
            "ax.set_title('ROE quintiles: Q1 (low ROE) outperforms Q5 (high ROE)')\n"
            "ax.legend()\n"
            "for b, v in zip(bars, q_df['mean%/yr']):\n"
            "    ax.annotate(f'{v:.1f}%', (b.get_x()+b.get_width()/2, v),\n"
            "                ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(q_df.round(2).to_string(index=False))\n"
            "hdg_mean = s_hdg['mean']*100 if HAVE_REAL else R['hedge_mean']\n"
            "hdg_t = s_hdg['tstat'] if HAVE_REAL else R['hedge_t']\n"
            "print(f\"\\nHedge (Q5-Q1): {hdg_mean:+.1f}%/yr  HAC t = {hdg_t:+.2f}\")"
        ),
        md(
            "> 💡 The pattern is the *opposite* of the quality thesis: the lowest-ROE firms "
            f"(Q1, {R['q1_mean']:.1f}%/yr) outperform the highest-ROE firms "
            f"(Q5, {R['q5_mean']:.1f}%/yr) by ~{abs(R['hedge_mean']):.0f}pp/yr on average. "
            "The market has already valued the quality — high-ROE stocks trade at premium "
            "multiples, and that premium embeds the expected outperformance, leaving little "
            "residual for the investor."
        ),
        md(
            "### 4b · Random-portfolio null — high-ROE beats fewer than half of random draws\n\n"
            "Does the high-ROE portfolio outperform a random same-size selection from the "
            "universe? If ROE adds value beyond sheer concentration, it should beat most "
            "random draws."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rand = st.random_portfolio_returns(roe_w, fwd_w, q=0.20, n_draws=500, seed=200)\n"
            "    hi_exc = float((h_roe['q5'] - h_roe['market']).mean())\n"
            "    rand_std = rand.std()*100\n"
            "    pct_above = (rand > hi_exc).mean()*100\n"
            "else:\n"
            "    rng = np.random.default_rng(42)\n"
            "    rand = pd.Series(rng.normal(0, R['rand_std']/100, 500*17))\n"
            "    hi_exc = R['hi_excess_mean']/100\n"
            "    rand_std = R['rand_std']\n"
            "    pct_above = 100 - R['rand_pct_above_hi']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "ax.hist(rand*100, bins=40, color=GREY, alpha=0.65, label='random same-size portfolios')\n"
            "ax.axvline(hi_exc*100, c=RED, lw=2.5, label=f'high-ROE Q5 excess: {hi_exc*100:+.1f}%/yr')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('excess return vs market (%/yr)'); ax.set_ylabel('count')\n"
            "ax.set_title('High-ROE Q5 falls BELOW the median random portfolio')\n"
            "ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'High-ROE excess: {hi_exc*100:+.1f}%/yr')\n"
            "print(f'Random portfolio std: {rand_std:.1f}%/yr')\n"
            "print(f'Fraction of random draws ABOVE high-ROE excess: {pct_above:.0f}%')"
        ),
        md(
            "> 💡 The high-ROE portfolio lands *below* the median random draw — "
            f"~{100 - R['rand_pct_above_hi']}% of random portfolios beat it. The ROE sort "
            "adds negative value relative to an uninformed selection of the same size. "
            "This is the clearest signal that ROE is not just uninformative but "
            "mildly negatively informative within this universe."
        ),
        md(
            "### 4c · Head-to-head: ROE vs Gross Profitability (Novy-Marx 2013)\n\n"
            "GP/Assets (gross profit / total assets) is theoretically a cleaner signal. "
            "Does it fare better?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    s_gp_hdg = st.summary(h_gp['hedge'])\n"
            "    s_gp_hi  = st.summary(h_gp['q5'] - h_gp['market'])\n"
            "    gp_hedge_m = s_gp_hdg['mean']*100\n"
            "    gp_hedge_t = s_gp_hdg['tstat']\n"
            "    gp_hi_exc  = s_gp_hi['mean']*100\n"
            "    gp_hi_exc_t = s_gp_hi['tstat']\n"
            "else:\n"
            "    gp_hedge_m = R['gp_hedge_mean']; gp_hedge_t = R['gp_hedge_t']\n"
            "    gp_hi_exc = R['gp_hi_excess']; gp_hi_exc_t = R['gp_hi_excess_t']\n"
            "\n"
            "roe_hedge_m = R['hedge_mean']; roe_hedge_t = R['hedge_t']\n"
            "roe_hi_exc = R['hi_excess_mean']; roe_hi_exc_t = R['hi_excess_t']\n"
            "if HAVE_REAL:\n"
            "    s_r = st.summary(h_roe['hedge'])\n"
            "    roe_hedge_m = s_r['mean']*100; roe_hedge_t = s_r['tstat']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "labels = ['ROE (this study)', 'GP/Assets (Novy-Marx)']\n"
            "hedge_vals = [roe_hedge_m, gp_hedge_m]\n"
            "hi_exc_vals = [roe_hi_exc, gp_hi_exc]\n"
            "x = np.arange(2); w = 0.35\n"
            "b1 = ax.bar(x - w/2, hedge_vals, w, label='Hedge (Q5-Q1) %/yr', color=[RED, AMBER])\n"
            "b2 = ax.bar(x + w/2, hi_exc_vals, w, label='Q5 excess vs mkt %/yr', color=[RED, AMBER], alpha=0.6)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('%/yr')\n"
            "ax.set_title('ROE vs GP/Assets: neither signal works inside S&P 500')\n"
            "ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'ROE hedge: {roe_hedge_m:+.1f}%/yr  t={roe_hedge_t:+.2f}')\n"
            "print(f'GP hedge:  {gp_hedge_m:+.1f}%/yr  t={gp_hedge_t:+.2f}')"
        ),
        md(
            "> 💡 Neither ROE nor GP/Assets delivers a positive, significant quality premium "
            f"on this panel. The ROE hedge is **{R['hedge_mean']:+.1f}%/yr** "
            f"(*t* = {R['hedge_t']:+.2f}); the GP/Assets hedge is "
            f"**{R['gp_hedge_mean']:+.1f}%/yr** (*t* = {R['gp_hedge_t']:+.2f}). Both are "
            "consistent with the same story: quality is priced within the S&P 500 survivor "
            "universe, regardless of which definition of profitability we use."
        ),
        md(
            "### 4d · Synthetic positive control — the engine works when an effect exists\n\n"
            "To confirm the method can detect a real premium when one is planted, we run "
            "the same quintile sort on a synthetic panel with a tunable quality premium."
        ),
        code(
            "premiums = [-0.04, 0.0, 0.04, 0.08, 0.12]\n"
            "results = []\n"
            "for prem in premiums:\n"
            "    roe_s, gp_s, fwd_s, truth = data.synthetic_panel(\n"
            "        n_firms=300, n_years=25, premium=prem, seed=200\n"
            "    )\n"
            "    h_s = st.quintile_returns(roe_s, fwd_s, q=0.20)\n"
            "    s_s = st.summary(h_s['hedge'])\n"
            "    results.append((prem, s_s['mean']*100, s_s['tstat']))\n"
            "\n"
            "res_df = pd.DataFrame(results, columns=['premium', 'hedge_%/yr', 'HAC_t'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "colors = [GREEN if v > 0 else RED for v in res_df['hedge_%/yr']]\n"
            "ax.bar(res_df['premium'].astype(str), res_df['hedge_%/yr'], color=colors, width=0.6)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted quality premium (per z-rank unit)')\n"
            "ax.set_ylabel('realised hedge (Q5-Q1) %/yr')\n"
            "ax.set_title('Synthetic control: engine detects premium when it exists')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(res_df.round(2).to_string(index=False))\n"
            "print('\\nOn the real tape: hedge = {:.1f}%/yr -- consistent with premium ≈ 0 or slightly negative'.format(R['hedge_mean']))"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — hedge {R['hedge_mean']:+.1f}%/yr, HAC *t* {R['hedge_t']:+.2f} "
            f"(wrong sign); Q5 excess {R['hi_excess_mean']:+.1f}%/yr (*t* {R['hi_excess_t']:+.2f}); "
            f"hit rate {R['hedge_hit']}%. Both ROE and GP/Assets fail to deliver a positive, "
            "significant quality premium on this panel.\n"
            "- **Tradability `MIRAGE`** — negative point estimate for the hedge; the long-quality "
            "leg (Q5) underperforms the market; no evidence of a tradable premium.\n"
            "- **ROE priced `YES`** — survivorship-biased S&P 500 is an already-screened quality "
            "universe; the ROE premium is embedded in valuations, not in forward returns."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the practical arithmetic\n\n"
            "Annual rebalancing with ~50–70 names per quintile leg:"
        ),
        code(
            "# Cost breakdown for the long-quality leg\n"
            "n_names_per_leg = 65  # approximate quintile size\n"
            "annual_turnover_pct = 30  # typical for annual rebalance\n"
            "cost_per_trade_bps = 5  # realistic institutional round-trip\n"
            "gross_alpha_pct = R['hi_excess_mean']  # -1.0%\n"
            "\n"
            "# Effective annual cost\n"
            "annual_cost_bps = annual_turnover_pct / 100 * cost_per_trade_bps\n"
            "net_alpha = gross_alpha_pct - annual_cost_bps / 100\n"
            "\n"
            "print(f'Gross alpha (Q5 vs market):  {gross_alpha_pct:+.1f}%/yr')\n"
            "print(f'~30% turnover × 5bp cost:    -{annual_cost_bps/100:.2f}%/yr')\n"
            "print(f'Net alpha:                   {net_alpha:+.2f}%/yr')\n"
            "print()\n"
            "print('Even at zero transaction cost, the gross alpha is negative.')\n"
            "print('There is no break-even cost that makes this investable.')"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — where quality *can* work\n\n"
            "This study tests one of the most restrictive possible contexts for a quality "
            "factor: a survivorship-biased, already-screened universe with no factor "
            "neutralisation. The literature suggests quality works in:\n\n"
            "1. **Broader universes**: Russell 1000 / Russell 2000 or all EDGAR filers — "
            "the distressed firms the S&P 500 excludes are where the quality short leg bites.\n"
            "2. **Combined with value**: 'Quality at a reasonable price' (QARP) or the "
            "Asness et al. (2019) QMJ approach that combines profitability, growth, safety "
            "and payout — not just ROE.\n"
            "3. **With factor neutralisation**: removing market beta, size, and sector "
            "exposures often reveals a cleaner quality signal.\n"
            "4. **Pre-publication windows**: quality premia tend to erode post-publication "
            "(McLean & Pontiff 2016) — the ROE signal was widely known by 2013.\n\n"
            "Related desk studies: "
            "[Study 65 — Scorecard](../../65-scorecard/), "
            "[Study 121 — Magic-Formula](../../121-magic-formula/), "
            "[Study 153 — Net-Operating-Assets](../../153-net-operating-assets/) — "
            "all working with related quality/profitability ideas on the same EDGAR cache."
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
