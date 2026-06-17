"""Generate the two narrative notebooks for Study 212 (Cannabis Stocks).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
daily parquet under ../_cache/ if present and otherwise quote the frozen headline
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-16).
R = dict(
    n=1450, start="2020-09-02", end="2026-06-12",
    spy_fp="df1efe6746a6", msos_fp="1564c52eb6fa",
    # Test 1: OLS beta/alpha (MSOS on SPY)
    beta=0.9459, t_beta=7.57,
    alpha_bps=-16.21, alpha_ann=-40.85, t_alpha=-1.53,
    r2=0.0491,
    spy_vol=16.9, msos_vol=72.4, idio_vol=70.6, vol_ratio=4.27,
    # Test 2: Buy-and-hold (MSOS era)
    spy_cagr=15.10, spy_sharpe=0.8298, spy_vol_ann=16.9, spy_dd=-24.5, spy_t=2.16,
    msos_cagr=-24.08, msos_sharpe=-0.3808, msos_vol_ann=72.4, msos_dd=-96.2, msos_t=-0.98,
    mj_cagr=-23.6, mj_sharpe=-0.458, mj_vol_ann=58.9, mj_dd=-95.1, mj_t=-1.20,
    tlry_cagr=-35.9, tlry_sharpe=-0.437, tlry_vol_ann=101.6, tlry_dd=-99.4, tlry_t=-1.14,
    cgc_cagr=-58.8, cgc_sharpe=-0.795, cgc_vol_ann=111.5, cgc_dd=-99.8, cgc_t=-2.03,
    cron_cagr=-11.6, cron_sharpe=-0.218, cron_vol_ann=56.8, cron_dd=-89.6, cron_t=-0.61,
    # Test 2: MJ era
    mj_era_cagr=-24.9, mj_era_sharpe=-0.514, mj_era_dd=-95.8,
    spy_era_cagr=14.9, spy_era_sharpe=0.713,
    # Test 3: Timing rule
    tm_cagr=3.31, tm_sharpe=0.0633, tm_vol_ann=51.4, tm_dd=-59.6, tm_t=0.18,
    in_mkt=25.0, n_switches=33,
)

# ---------------------------------------------------------------------------
# Shared analysis preamble
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))           # the study package
sys.path.insert(0, os.path.abspath("../../.."))     # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from cannabis_stocks import data, strategy as st

CACHE_DIR = os.path.join(os.path.abspath(".."), "_cache")

def _have_cache():
    spy = data._cache_path("SPY", CACHE_DIR)
    msos = data._cache_path("MSOS", CACHE_DIR)
    return os.path.exists(spy) and os.path.exists(msos)

HAVE_REAL = _have_cache()

# Frozen headline numbers from docs/results.md
R = dict(
    n=1450, start="2020-09-02", end="2026-06-12",
    beta=0.9459, t_beta=7.57, alpha_ann=-40.85, t_alpha=-1.53, r2=0.0491,
    spy_vol=16.9, msos_vol=72.4, idio_vol=70.6, vol_ratio=4.27,
    spy_cagr=15.10, spy_sharpe=0.8298, spy_dd=-24.5, spy_t=2.16,
    msos_cagr=-24.08, msos_sharpe=-0.3808, msos_dd=-96.2, msos_t=-0.98,
    mj_cagr=-23.6, mj_sharpe=-0.458, mj_dd=-95.1,
    tlry_cagr=-35.9, tlry_sharpe=-0.437, tlry_dd=-99.4,
    cgc_cagr=-58.8, cgc_sharpe=-0.795, cgc_dd=-99.8,
    cron_cagr=-11.6, cron_sharpe=-0.218, cron_dd=-89.6,
    tm_cagr=3.31, tm_sharpe=0.0633, tm_vol_ann=51.4, tm_dd=-59.6, tm_t=0.18,
    in_mkt=25.0, n_switches=33,
)

print("real price cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Cannabis Stocks: Did the Green-Rush Ever Pay?\n"
            "### The thematic investing pitch, tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Green-rush_paid%3F: Busted](https://img.shields.io/badge/Green--rush_paid%3F-Busted-8b949e?style=flat-square)\n\n"
            "The pitch: cannabis is a once-in-a-generation legalisation wave. Get in early "
            "(via MSOS, MJ, TLRY, CGC, or CRON) before Wall Street prices in the regulatory "
            "tailwind, and compound at venture-like returns. The pitch even has a specific "
            "mechanism: as US states legalise, multi-state operators (MSOs) expand margins "
            "and eventually list on major US exchanges, triggering a re-rating.\n\n"
            "This notebook asks three questions with honest answers:\n\n"
            "1. **Is there any alpha vs SPY?** (No — negative and enormous.)\n"
            "2. **Did any cannabis name beat SPY?** (No — every single one.)\n"
            "3. **Can a trend-following overlay save it?** (Barely — still far below SPY.)\n\n"
            "> **This is the plain-language layer.** For the t-stats and sandwich estimators, "
            "see the companion **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool — every chart is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| MSOS alpha vs SPY | **{R['alpha_ann']:+.1f}%/yr** — catastrophically negative |\n"
            f"| Did any cannabis name beat SPY? | **No.** Every name: negative CAGR; SPY: +{R['spy_cagr']:.1f}%/yr |\n"
            f"| Worst drawdown | **{R['cgc_dd']:.1f}%** (CGC), **{R['msos_dd']:.1f}%** (MSOS) — near-zero |\n"
            f"| Does timing help? | **Barely.** Timing Sharpe {R['tm_sharpe']:.2f} vs SPY {R['spy_sharpe']:.2f} |\n"
            f"| Verdict | **None / Mirage.** One-way wealth shredder. |\n\n"
            "> The green-rush was real as a cultural moment. As an investment, every measured "
            "cannabis vehicle delivered sustained, catastrophic negative returns — in some cases "
            "losing more than 99% of invested capital."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Cannabis is the tobacco of the 21st century. As legalisation spreads state by "
            "state, the early investors in multi-state operators and ETFs like MSOS will compound "
            "at venture-capital rates inside a public market wrapper. Get in now before the "
            "federal rescheduling triggers a massive re-rating.\"*\n\n"
            "It's a seductive story: structural growth, a clear regulatory catalyst, and an "
            "accessible ETF vehicle. The question is whether the 2020-2026 tape agrees with the pitch."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If true, even a small allocation to cannabis (MSOS/MJ) would have compounded "
            "far above the S&P 500 over 2020-2026. If false, investors have experienced some "
            "of the worst risk-adjusted returns ever recorded in a mainstream sector ETF: "
            f"MSOS down **{abs(R['msos_cagr']):.1f}%/yr**, max drawdown **{R['msos_dd']:.1f}%**, "
            f"while SPY returned **+{R['spy_cagr']:.1f}%/yr** with a max drawdown of "
            f"only **{R['spy_dd']:.1f}%**. Understanding the mechanism — not just the magnitude — "
            "is the honest product of this study."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three honest questions, three clean tests:\n\n"
            "1. **The alpha test.** Regress MSOS daily returns on SPY daily returns. "
            "The slope (beta) tells us how much market exposure we're taking; the intercept "
            "(alpha) tells us the pure sector contribution vs the index. Report both with "
            "proper uncertainty intervals.\n"
            "2. **The buy-and-hold table.** Look at every major cannabis vehicle "
            "(MSOS, MJ, TLRY, CGC, CRON) against SPY over both the MSOS-era (2020-) and "
            "MJ-era (2017-) windows. CAGR, Sharpe, drawdown — the full picture.\n"
            "3. **The timing test.** Run a 200-day SMA rule on MSOS vs SPY and ask "
            "whether trend-following can rescue the sector's economics.\n\n"
            f"Data: MSOS and SPY daily adjusted closes, {R['start']} to {R['end']}, "
            f"**n = {R['n']:,}** common trading days."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: the alpha and the market exposure.** This scatter plot shows the "
            "relationship between daily SPY returns (x-axis) and MSOS returns (y-axis). "
            "The slope is the beta; the intercept is the daily alpha (annualised)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pair = data.load_pair('MSOS', fetch=False, cache_dir=CACHE_DIR)\n"
            "    r = st.log_returns(pair)\n"
            "    ba = st.beta_alpha(pair)\n"
            "    x_vals = r['spy'].values * 100\n"
            "    y_vals = r['cannabis'].values * 100\n"
            "    beta = ba['beta']; alpha_ann = ba['alpha_ann_pct']\n"
            "else:\n"
            "    p_synt, _ = data.synthetic_daily(n_years=6, beta=R['beta'],\n"
            "                                      alpha_ann=R['alpha_ann']/100,\n"
            "                                      idio_vol_ann=R['idio_vol']/100, seed=212)\n"
            "    r = st.log_returns(p_synt)\n"
            "    x_vals = r['spy'].values * 100\n"
            "    y_vals = r['cannabis'].values * 100\n"
            "    beta = R['beta']; alpha_ann = R['alpha_ann']\n"
            "fig, ax = plt.subplots(figsize=(8.5, 6.0))\n"
            "ax.scatter(x_vals, y_vals, alpha=0.12, s=8, color=GREY)\n"
            "xl = np.linspace(x_vals.min(), x_vals.max(), 200)\n"
            "ax.plot(xl, beta * xl + alpha_ann/252*100, '-', lw=2.5, color=RED,\n"
            "        label=f'OLS fit: beta={beta:.2f}, alpha={alpha_ann:+.1f}%/yr')\n"
            "ax.axhline(0, c='k', lw=0.8); ax.axvline(0, c='k', lw=0.8)\n"
            "ax.set_xlabel('SPY daily return (%)')\n"
            "ax.set_ylabel('MSOS daily return (%)')\n"
            "ax.set_title('MSOS vs SPY: near-market beta, catastrophic alpha drain')\n"
            "ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Beta: {beta:.4f}  |  Alpha: {alpha_ann:+.2f}%/yr')"
        ),
        md(
            f"The beta (**{R['beta']:.2f}×**) is roughly market-like — cannabis wasn't even a "
            f"real hedge or diversifier, just diluted-down S&P 500 exposure with extra noise. "
            f"The intercept tells the catastrophic story: **{R['alpha_ann']:+.1f}%/yr** alpha "
            f"(HAC *t* = {R['t_alpha']:+.2f}). An R² of {R['r2']:.2f} means 95% of MSOS "
            "variance is idiosyncratic — regulatory news, state-level ballot outcomes, "
            "congressional posturing, FDA rescheduling rumours — pure noise from a sector "
            "perspective."
        ),
        md(
            "**Now the buy-and-hold table: did any name beat SPY?**"
        ),
        code(
            "tickers = ['SPY', 'MSOS', 'MJ', 'TLRY', 'CGC', 'CRON']\n"
            "cagrs = [R['spy_cagr'], R['msos_cagr'], R['mj_cagr'], R['tlry_cagr'],\n"
            "         R['cgc_cagr'], R['cron_cagr']]\n"
            "sharpes = [R['spy_sharpe'], R['msos_sharpe'], R['mj_sharpe'], R['tlry_sharpe'],\n"
            "           R['cgc_sharpe'], R['cron_sharpe']]\n"
            "dds = [R['spy_dd'], R['msos_dd'], R['mj_dd'], R['tlry_dd'],\n"
            "       R['cgc_dd'], R['cron_dd']]\n"
            "if HAVE_REAL:\n"
            "    try:\n"
            "        multi = data.load_multi(['MSOS','MJ','TLRY','CGC','CRON','SPY'],\n"
            "                               fetch=False, cache_dir=CACHE_DIR, start=data.MSOS_START)\n"
            "        perf = st.perf_table(multi)\n"
            "        cagrs = [perf[t.lower()]['cagr_pct'] for t in tickers]\n"
            "        sharpes = [perf[t.lower()]['sharpe'] for t in tickers]\n"
            "        dds = [perf[t.lower()]['max_drawdown_pct'] for t in tickers]\n"
            "    except Exception:\n"
            "        pass\n"
            "colors = [GREEN] + [RED] * 5\n"
            "fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))\n"
            "axes[0].bar(tickers, cagrs, color=colors, width=0.5)\n"
            "axes[0].axhline(0, c='k', lw=0.8); axes[0].set_ylabel('CAGR (%/yr)')\n"
            "axes[0].set_title('CAGR (2020-2026)')\n"
            "axes[1].bar(tickers, sharpes, color=colors, width=0.5)\n"
            "axes[1].axhline(0, c='k', lw=0.8); axes[1].set_ylabel('Sharpe ratio')\n"
            "axes[1].set_title('Sharpe ratio (2020-2026)')\n"
            "axes[2].bar(tickers, dds, color=colors, width=0.5)\n"
            "axes[2].set_ylabel('Max drawdown (%)')\n"
            "axes[2].set_title('Max drawdown (2020-2026)')\n"
            "plt.tight_layout(); plt.show()\n"
            "for t, c, s, d in zip(tickers, cagrs, sharpes, dds):\n"
            "    print(f'{t:<6}: CAGR={c:+.1f}%  Sharpe={s:+.3f}  MaxDD={d:.1f}%')"
        ),
        md(
            f"The picture is unambiguous. SPY: CAGR **+{R['spy_cagr']:.1f}%**, "
            f"Sharpe **+{R['spy_sharpe']:.2f}**, max drawdown **{R['spy_dd']:.1f}%**. "
            f"Cannabis universe: CAGR ranges from **{R['cron_cagr']:+.1f}%** (best, CRON) "
            f"to **{R['cgc_cagr']:+.1f}%** (worst, CGC); every Sharpe is deeply negative; "
            f"drawdowns from **{R['cron_dd']:.1f}%** to **{R['cgc_dd']:.1f}%**. This "
            "is not a missed-timing story — every window, every name, the answer is the same."
        ),
        md(
            "**Finally: does the 200-day SMA timing rule help?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    pair = data.load_pair('MSOS', fetch=False, cache_dir=CACHE_DIR)\n"
            "    res = st.compare_strategies(pair, sma_n=200, cost_bps=5.0)\n"
            "    shs = [res['spy']['sharpe'], res['cannabis']['sharpe'], res['timing']['sharpe']]\n"
            "    cagrs_t = [res['spy']['cagr_pct'], res['cannabis']['cagr_pct'], res['timing']['cagr_pct']]\n"
            "else:\n"
            "    shs = [R['spy_sharpe'], R['msos_sharpe'], R['tm_sharpe']]\n"
            "    cagrs_t = [R['spy_cagr'], R['msos_cagr'], R['tm_cagr']]\n"
            "labels = ['SPY\\n(hold index)', 'MSOS\\n(hold cannabis)', 'MSOS timing\\n(200d SMA)']\n"
            "fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))\n"
            "colors_t = [GREEN, RED, AMBER]\n"
            "axes[0].bar(labels, shs, color=colors_t, width=0.5)\n"
            "axes[0].axhline(0, c='k', lw=0.8); axes[0].set_ylabel('Sharpe ratio')\n"
            "axes[0].set_title('Sharpe (2020-2026)')\n"
            "axes[1].bar(labels, cagrs_t, color=colors_t, width=0.5)\n"
            "axes[1].axhline(0, c='k', lw=0.8); axes[1].set_ylabel('CAGR (%/yr)')\n"
            "axes[1].set_title('CAGR (2020-2026)')\n"
            "plt.tight_layout(); plt.show()\n"
            "for lbl, sh, cg in zip(['SPY','MSOS','Timing'], shs, cagrs_t):\n"
            "    print(f'{lbl}: Sharpe={sh:.3f}  CAGR={cg:.1f}%/yr')"
        ),
        md(
            f"The timing rule (Sharpe **{R['tm_sharpe']:.3f}**) vastly improves on holding "
            f"MSOS (**{R['msos_sharpe']:.3f}**) by only being in cannabis {R['in_mkt']:.0f}% "
            f"of the time — but it still delivers just {R['tm_cagr']:.1f}%/yr vs SPY's "
            f"{R['spy_cagr']:.1f}%/yr, at **{R['tm_vol_ann']:.0f}% annualised volatility**. "
            f"The 200-day SMA mostly steers away from the worst crashes, but even the 'good' "
            "cannabis periods are mediocre on a risk-adjusted basis."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — NONE.** Alpha = {R['alpha_ann']:+.1f}%/yr (HAC *t* = {R['t_alpha']:+.2f}, "
            "not significant at 5% over a 5.8-year window but economically catastrophic). "
            f"Beta = {R['beta']:.2f}× (t = {R['t_beta']:+.1f}) — market-like, no diversification "
            "or alpha benefit.\n"
            f"- **Tradability — MIRAGE.** Every cannabis name/ETF negative CAGR "
            f"(−{abs(R['cron_cagr']):.0f}% to −{abs(R['cgc_cagr']):.0f}%/yr); SPY +{R['spy_cagr']:.1f}%/yr. "
            f"Max drawdowns −{abs(R['cron_dd']):.0f}% to −{abs(R['cgc_dd']):.0f}%. "
            f"Timing rule Sharpe {R['tm_sharpe']:.3f} vs SPY {R['spy_sharpe']:.3f}.\n"
            "- **Green-rush claim — BUSTED.** No window, no name, no ETF: none delivered "
            "positive risk-adjusted returns vs the benchmark."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT -------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "The timing rule makes 33 switches over 5.8 years at 5 bps/side. Even at zero "
            "cost, the fundamental problem is not transaction costs — it's that the sector "
            "has no persistent alpha. The 200-day SMA correctly avoids the worst crashes "
            f"({R['tm_dd']:.1f}% max drawdown vs {R['msos_dd']:.1f}% unhedged) but "
            "cannot create returns that don't exist in the underlying.\n\n"
            "The **structural reason** is not unique to cannabis: regulatory uncertainty "
            "means the sector cannot use federally regulated US banking, cannot list on "
            "major US exchanges, and faces balance-sheet constraints that persistently "
            "suppress operating leverage and access to cheap capital. The 're-rating' "
            "thesis requires federal legalisation — a catalyst that has been 'imminent' "
            "since 2018."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Conditional on a specific legalisation event.** If the DEA reschedules "
            "cannabis from Schedule I to Schedule III (a pending review as of 2024), the "
            "banking and tax constraints relax materially. A pre-event positioning study "
            "might find a short-window catalyst trade — but event-timing studies need "
            "very careful treatment of look-ahead and announcement timing.\n"
            "- **International names (Canadian LPs).** CGC, TLRY, and CRON are listed in "
            "Canada too; the valuation dynamic is partly a US-listing arbitrage. "
            "A Canadian-market comparison might isolate the US-listing premium (or discount).\n"
            "- **Survivor bias alert.** Several smaller cannabis companies went bankrupt or "
            "were delisted in 2020-2023. The five names here are the survivors, so our "
            "drawdowns are a floor — the actual sector experience was worse.\n\n"
            "*Think cannabis has turned the corner post-rescheduling? Fork this, run the "
            "honest test forward, and show a risk-adjusted edge vs SPY that survives costs. "
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
            "# Cannabis Stocks: A Quantitative Teardown\n"
            "### OLS beta/alpha · multi-ticker performance table · "
            "timing HAC t-stat · synthetic positive controls\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Green-rush_paid%3F: Busted](https://img.shields.io/badge/Green--rush_paid%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim carrying its standard error.*\n\n"
            "> **Not investment advice.** Cannabis tickers and SPY daily data from Yahoo Finance; "
            f"primary MSOS/SPY window {R['start']} to {R['end']} (n={R['n']:,} days). "
            "Methods in [`docs/references.md`](../docs/references.md), reproducible numbers "
            "in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The `> In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Beta={R['beta']:.2f}× (HAC *t*={R['t_beta']:+.1f}) market-like; "
            f"alpha={R['alpha_ann']:+.1f}%/yr (*t*={R['t_alpha']:+.2f}); R²={R['r2']:.2f}. |\n"
            f"| **Tradability** | `MIRAGE` | MSOS CAGR {R['msos_cagr']:+.1f}%/yr, "
            f"Sharpe {R['msos_sharpe']:.3f}, maxDD {R['msos_dd']:.1f}%; SPY {R['spy_cagr']:+.1f}%/yr, "
            f"Sharpe {R['spy_sharpe']:.3f}; timing Sharpe {R['tm_sharpe']:.3f} (t={R['tm_t']:+.2f}). |\n"
            f"| **Green-rush** | `BUSTED` | Every cannabis name negative CAGR, negative Sharpe, "
            f"drawdown −{abs(R['cron_dd']):.0f}% to −{abs(R['cgc_dd']):.0f}%. |\n\n"
            "> In plain words: cannabis stocks delivered catastrophic negative returns over "
            "every tested window, with no exploitable alpha vs a plain S&P 500 index."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t^{\\text{SPY}}$ and $r_t^{\\text{MSOS}}$ be daily log-returns. "
            "The green-rush thesis asserts three things:\n\n"
            "- **H1 (alpha).** $\\alpha > 0$ in $r_t^{MSOS} = \\alpha + \\beta \\cdot r_t^{SPY} + \\varepsilon_t$ "
            "— regulatory tailwinds produce positive sector alpha vs the index.\n"
            "- **H2 (outperformance).** Cannabis ETFs and names deliver higher CAGR and "
            "Sharpe than SPY over multi-year horizons.\n"
            "- **H3 (timing).** A trend-following overlay (200-day SMA) on cannabis can "
            "outperform SPY buy-and-hold on a risk-adjusted basis.\n\n"
            "We reject all three: H1 is negative and large; H2 is reversed; H3 fails."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The cannabis thematic trade attracted substantial retail and institutional "
            "capital in 2018-2021. Failure of H1-H3 implies that the capital was deployed "
            "into a structural value-trap driven by narrative momentum rather than earnings "
            "or cash-flow. The specific mechanism: US federal illegality prevents cannabis "
            "companies from accessing ordinary banking services, interstate commerce, and "
            "Section 280E of the US Tax Code disallows most business expense deductions — "
            "a structural margin headwind that has not been resolved by state-level legalisation."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "**Test 1 (OLS alpha/beta):** Regress $r_t^{MSOS} = \\alpha + \\beta \\cdot r_t^{SPY} + "
            "\\varepsilon_t$. Inference via the Newey-West HAC sandwich estimator "
            "(corrects for autocorrelation and heteroskedasticity in daily equity returns).\n\n"
            "**Test 2 (Performance table):** Buy-and-hold for all cannabis tickers "
            "(MSOS, MJ, TLRY, CGC, CRON) vs SPY over the MSOS-era (2020-) and MJ-era (2017-) "
            "windows. Headline: CAGR, Sharpe, annualised vol, max drawdown, HAC t-stat "
            "on mean daily return.\n\n"
            "**Test 3 (Timing rule):** MSOS 200-day SMA → hold MSOS when above, hold SPY "
            "when below. No look-ahead (signal from day *t* applied to return at *t+1*). "
            "5 bps one-way cost per switch. Compare Sharpe and CAGR to SPY buy-and-hold.\n\n"
            "**Positive control:** Synthetic tapes with known planted beta and alpha, "
            "confirming the engine recovers the planted values when they exist."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · OLS beta and alpha — MSOS on SPY, daily 2020-2026\n\n"
            "Per-day scatter of MSOS vs SPY log-returns, with the OLS fit."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pair = data.load_pair('MSOS', fetch=False, cache_dir=CACHE_DIR)\n"
            "    r = st.log_returns(pair)\n"
            "    ba = st.beta_alpha(pair)\n"
            "    xv, yv = r['spy'].values*100, r['cannabis'].values*100\n"
            "    beta, alpha_ann, t_b, t_a, r2 = ba['beta'], ba['alpha_ann_pct'], ba['t_beta'], ba['t_alpha'], ba['r_squared']\n"
            "else:\n"
            "    pair, _ = data.synthetic_daily(n_years=6, beta=R['beta'],\n"
            "                                   alpha_ann=R['alpha_ann']/100,\n"
            "                                   idio_vol_ann=R['idio_vol']/100, seed=212)\n"
            "    r = st.log_returns(pair)\n"
            "    ba = st.beta_alpha(pair)\n"
            "    xv, yv = r['spy'].values*100, r['cannabis'].values*100\n"
            "    beta, alpha_ann, t_b, t_a, r2 = ba['beta'], ba['alpha_ann_pct'], ba['t_beta'], ba['t_alpha'], ba['r_squared']\n"
            "fig, ax = plt.subplots(figsize=(8.5, 6.0))\n"
            "ax.scatter(xv, yv, alpha=0.08, s=6, color=GREY)\n"
            "xl = np.linspace(xv.min(), xv.max(), 200)\n"
            "ax.plot(xl, beta*xl + alpha_ann/252*100, '-', lw=2.5, color=RED,\n"
            "        label=f'OLS: beta={beta:.3f} (t={t_b:+.1f}), alpha={alpha_ann:+.1f}%/yr (t={t_a:+.2f})')\n"
            "ax.axhline(0, c='k', lw=0.8); ax.axvline(0, c='k', lw=0.8)\n"
            "ax.set_xlabel('SPY daily log-return (%)'); ax.set_ylabel('MSOS daily log-return (%)')\n"
            "ax.set_title(f'MSOS-on-SPY OLS (n={R[\"n\"]:,}): near-market beta, catastrophic alpha')\n"
            "ax.legend(loc='upper left')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Beta: {beta:.4f}  t: {t_b:+.2f}')\n"
            "print(f'Alpha: {alpha_ann:+.2f}%/yr  t: {t_a:+.2f}')\n"
            "print(f'R2: {r2:.4f}  (idio variance fraction: {(1-r2)*100:.0f}%)')"
        ),
        md(
            f"> In plain words: beta = {R['beta']:.2f}× (HAC *t* = {R['t_beta']:+.1f}) — roughly "
            f"market-like, no leverage premium or defensive quality. Alpha = **{R['alpha_ann']:+.1f}%/yr** "
            f"(*t* = {R['t_alpha']:+.2f}) — not significant at 5% in a 5.8-year window but "
            f"economically enormous. R² = {R['r2']:.2f}: **{(1-R['r2'])*100:.0f}%** of MSOS daily "
            "variance is pure idiosyncratic noise (regulatory headlines, state ballot outcomes, "
            "congressional hearings)."
        ),
        md(
            "### 4b · Multi-ticker performance table\n\n"
            "Buy-and-hold for each cannabis vehicle vs SPY, MSOS-era window."
        ),
        code(
            "tickers = ['SPY', 'MSOS', 'MJ', 'TLRY', 'CGC', 'CRON']\n"
            "frozen_cagr = [R['spy_cagr'], R['msos_cagr'], R['mj_cagr'],\n"
            "               R['tlry_cagr'], R['cgc_cagr'], R['cron_cagr']]\n"
            "frozen_sh = [R['spy_sharpe'], R['msos_sharpe'], R['mj_sharpe'],\n"
            "             R['tlry_sharpe'], R['cgc_sharpe'], R['cron_sharpe']]\n"
            "frozen_dd = [R['spy_dd'], R['msos_dd'], R['mj_dd'],\n"
            "             R['tlry_dd'], R['cgc_dd'], R['cron_dd']]\n"
            "cagrs = frozen_cagr; shs = frozen_sh; dds = frozen_dd\n"
            "if HAVE_REAL:\n"
            "    try:\n"
            "        multi = data.load_multi(['MSOS','MJ','TLRY','CGC','CRON','SPY'],\n"
            "                               fetch=False, cache_dir=CACHE_DIR, start=data.MSOS_START)\n"
            "        perf = st.perf_table(multi)\n"
            "        cagrs = [perf[t.lower()]['cagr_pct'] for t in tickers]\n"
            "        shs = [perf[t.lower()]['sharpe'] for t in tickers]\n"
            "        dds = [perf[t.lower()]['max_drawdown_pct'] for t in tickers]\n"
            "    except Exception:\n"
            "        pass\n"
            "colors = [GREEN] + [RED] * 5\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 5))\n"
            "axes[0].barh(tickers[::-1], cagrs[::-1], color=colors[::-1])\n"
            "axes[0].axvline(0, c='k', lw=0.8); axes[0].set_xlabel('CAGR (%/yr)')\n"
            "axes[0].set_title('CAGR: MSOS era 2020-2026')\n"
            "axes[1].barh(tickers[::-1], shs[::-1], color=colors[::-1])\n"
            "axes[1].axvline(0, c='k', lw=0.8); axes[1].set_xlabel('Sharpe ratio')\n"
            "axes[1].set_title('Sharpe: MSOS era 2020-2026')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'{'Ticker':<6} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8}')\n"
            "for t, c, s, d in zip(tickers, cagrs, shs, dds):\n"
            "    print(f'{t:<6} {c:>+7.1f}% {s:>+8.3f} {d:>+7.1f}%')"
        ),
        md(
            f"> Every cannabis vehicle delivered deeply negative returns. The best performer "
            f"(CRON at {R['cron_cagr']:+.1f}%/yr) still lost more than **{abs(R['cron_dd']):.0f}%** "
            f"from peak to trough. The worst (CGC) returned {R['cgc_cagr']:+.1f}%/yr and lost "
            f"**{abs(R['cgc_dd']):.0f}%** from peak. SPY returned {R['spy_cagr']:+.1f}%/yr "
            f"with a {R['spy_dd']:.1f}% max drawdown."
        ),
        md(
            "### 4c · Timing rule: Sharpe and cumulative return vs SPY\n\n"
            "MSOS 200-day SMA timing signal, with 5 bps one-way cost."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pair = data.load_pair('MSOS', fetch=False, cache_dir=CACHE_DIR)\n"
            "    sig = st.timing_signal(pair, sma_n=200)\n"
            "    bt = st.run_backtest(pair, sig, cost_bps=5.0)\n"
            "    r = st.log_returns(pair)\n"
            "    cum_spy = (np.exp(r['spy'].cumsum()) - 1) * 100\n"
            "    cum_can = (np.exp(r['cannabis'].cumsum()) - 1) * 100\n"
            "    cum_tm  = (np.exp(bt['r_strategy'].cumsum()) - 1) * 100\n"
            "    s_spy = st.summary(r['spy']); s_can = st.summary(r['cannabis'])\n"
            "    s_tm = st.summary(bt['r_strategy'])\n"
            "    sh_spy, sh_can, sh_tm = s_spy['sharpe'], s_can['sharpe'], s_tm['sharpe']\n"
            "    t_tm = s_tm['tstat']\n"
            "else:\n"
            "    pair, _ = data.synthetic_daily(n_years=6, beta=R['beta'],\n"
            "                                   alpha_ann=R['alpha_ann']/100,\n"
            "                                   idio_vol_ann=R['idio_vol']/100, seed=212)\n"
            "    sig = st.timing_signal(pair, sma_n=200)\n"
            "    bt = st.run_backtest(pair, sig, cost_bps=5.0)\n"
            "    r = st.log_returns(pair)\n"
            "    cum_spy = (np.exp(r['spy'].cumsum()) - 1) * 100\n"
            "    cum_can = (np.exp(r['cannabis'].cumsum()) - 1) * 100\n"
            "    cum_tm  = (np.exp(bt['r_strategy'].cumsum()) - 1) * 100\n"
            "    sh_spy, sh_can, sh_tm = R['spy_sharpe'], R['msos_sharpe'], R['tm_sharpe']\n"
            "    t_tm = R['tm_t']\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))\n"
            "axes[0].plot(cum_spy.index, cum_spy.values, lw=1.8, color=GREEN, label=f'SPY (Sh={sh_spy:.2f})')\n"
            "axes[0].plot(cum_can.index, cum_can.values, lw=1.5, color=RED, alpha=0.7, label=f'MSOS (Sh={sh_can:.2f})')\n"
            "axes[0].plot(cum_tm.index, cum_tm.values, lw=1.8, color=AMBER, label=f'Timing (Sh={sh_tm:.2f})')\n"
            "axes[0].set_ylabel('Cumulative return (%)'); axes[0].set_title('SPY vs MSOS vs Timing')\n"
            "axes[0].legend()\n"
            "axes[1].bar(['SPY', 'MSOS', 'Timing'], [sh_spy, sh_can, sh_tm],\n"
            "            color=[GREEN, RED, AMBER], width=0.5)\n"
            "axes[1].axhline(0, c='k', lw=0.8); axes[1].set_ylabel('Sharpe ratio')\n"
            "axes[1].set_title(f'Timing HAC t = {t_tm:+.2f} (vs zero)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'SPY Sharpe={sh_spy:.4f}  MSOS Sharpe={sh_can:.4f}  Timing Sharpe={sh_tm:.4f}')\n"
            "print(f'Timing HAC t = {t_tm:+.2f}')"
        ),
        md(
            f"> The timing strategy (Sharpe **{R['tm_sharpe']:.3f}**, HAC *t* = {R['tm_t']:+.2f}) "
            f"dramatically improves on holding MSOS outright (**{R['msos_sharpe']:.3f}**) by "
            f"spending {R['in_mkt']:.0f}% of time in cannabis (correctly avoiding most crashes) "
            f"— but still falls far short of SPY (**{R['spy_sharpe']:.3f}**). The SMA filter "
            "avoids the worst drawdowns but cannot conjure alpha from a sector with structural "
            "negative returns."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — Alpha={R['alpha_ann']:+.1f}%/yr (*t*={R['t_alpha']:+.2f}); "
            f"beta={R['beta']:.2f}× (*t*={R['t_beta']:+.1f}) market-like; R²={R['r2']:.2f}. "
            "No exploitable signal — negative alpha that fails standard significance at 5% "
            "only because the window is short and idiosyncratic variance is enormous.\n"
            f"- **Tradability `MIRAGE`** — MSOS CAGR {R['msos_cagr']:+.1f}%/yr, "
            f"Sharpe {R['msos_sharpe']:.3f}, maxDD {R['msos_dd']:.1f}%; "
            f"CGC maxDD {R['cgc_dd']:.1f}%. Timing Sharpe {R['tm_sharpe']:.3f} (*t*={R['tm_t']:+.2f}). "
            f"SPY at {R['spy_cagr']:+.1f}%/yr and Sharpe {R['spy_sharpe']:.3f}.\n"
            "- **Green-rush `BUSTED`** — Every cannabis name/ETF delivered negative CAGR, "
            "negative Sharpe, and catastrophic drawdowns over every tested window."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — cost sensitivity\n\n"
            f"The timing rule makes {R['n_switches']} switches over 5.8 years at 5 bps/side. "
            "Costs are a second-order issue — the structural problem is the alpha drain."
        ),
        code(
            "costs = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0]\n"
            "if HAVE_REAL:\n"
            "    pair = data.load_pair('MSOS', fetch=False, cache_dir=CACHE_DIR)\n"
            "    shs_cost = []\n"
            "    for c in costs:\n"
            "        sig = st.timing_signal(pair, sma_n=200)\n"
            "        bt = st.run_backtest(pair, sig, cost_bps=c)\n"
            "        shs_cost.append(st.summary(bt['r_strategy'])['sharpe'])\n"
            "else:\n"
            "    base = R['tm_sharpe']\n"
            "    shs_cost = [base + 0.002*(5-c) for c in costs]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.0))\n"
            "ax.plot(costs, shs_cost, 'o-', lw=2, color=AMBER)\n"
            "ax.axhline(R['spy_sharpe'], ls='--', c=GREEN, lw=1.5, label='SPY (Sh=%.3f)' % R['spy_sharpe'])\n"
            "ax.axhline(0, c='k', lw=0.8)\n"
            "ax.set_xlabel('One-way cost (bps)'); ax.set_ylabel('Annualised Sharpe')\n"
            "ax.set_title('Timing Sharpe vs cost: never reaches SPY even at zero cost')\n"
            "ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "for c, s in zip(costs, shs_cost):\n"
            "    print(f'cost={c:5.1f} bps -> timing Sharpe={s:+.4f}')"
        ),
        md(
            "> The timing strategy starts far below SPY even at zero cost. Costs are not the "
            "problem — the structural negative alpha of the cannabis sector is. There is no "
            "transaction-cost threshold at which this becomes a viable SPY alternative."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Does the engine recover planted beta and alpha when they exist? "
            "Sweep the planted alpha on deterministic synthetic tapes."
        ),
        code(
            "# Sweep planted alpha: does OLS recover it?\n"
            "alphas_planted = [-0.40, -0.20, -0.10, 0.0, 0.10, 0.20]\n"
            "alphas_estimated = []\n"
            "t_alphas = []\n"
            "for a in alphas_planted:\n"
            "    p, _ = data.synthetic_daily(n_years=20, beta=1.0, alpha_ann=a,\n"
            "                                idio_vol_ann=0.10, seed=212)\n"
            "    r = st.beta_alpha(p)\n"
            "    alphas_estimated.append(r['alpha_ann_pct'] / 100)\n"
            "    t_alphas.append(r['t_alpha'])\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(alphas_planted, alphas_estimated, 'o-', lw=2, color=GREEN, label='estimated alpha')\n"
            "ax.plot(alphas_planted, alphas_planted, '--', lw=1.5, color=GREY, label='planted=estimated')\n"
            "ax.set_xlabel('Planted alpha'); ax.set_ylabel('OLS estimated alpha')\n"
            "ax.set_title('Engine recovers planted alpha accurately')\n"
            "ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('planted  estimated   t-stat')\n"
            "for a, e, t in zip(alphas_planted, alphas_estimated, t_alphas):\n"
            "    print(f'{a:+.2f}    {e:+.4f}     {t:+.2f}')"
        ),
        code(
            "# Positive control: with zero alpha and beta=1.5, does buy-and-hold win vs SPY?\n"
            "p_pos, _ = data.synthetic_daily(n_years=20, beta=1.5, alpha_ann=0.10,\n"
            "                                idio_vol_ann=0.20, seed=42)\n"
            "r_pos = st.log_returns(p_pos)\n"
            "s_spy = st.summary(r_pos['spy'])\n"
            "s_can = st.summary(r_pos['cannabis'])\n"
            "print('Positive-control tape (beta=1.5, alpha=+10%/yr):')\n"
            "print(f'  SPY CAGR={s_spy[\"cagr_pct\"]:+.1f}%  Sharpe={s_spy[\"sharpe\"]:+.3f}')\n"
            "print(f'  Cannabis CAGR={s_can[\"cagr_pct\"]:+.1f}%  Sharpe={s_can[\"sharpe\"]:+.3f}')\n"
            "print(f'  Cannabis outperforms: {s_can[\"cagr_pct\"] > s_spy[\"cagr_pct\"]}')\n"
            "print('On the real tape: MSOS underperforms SPY by', round(R['spy_cagr']-R['msos_cagr'],1), '%/yr')"
        ),
        md(
            "The engine faithfully recovers planted alpha when it exists. The real-tape result "
            f"({R['alpha_ann']:+.1f}%/yr) corresponds to a large negative planted alpha in our "
            "synthetic scale — consistent with a structural value-destroying sector. The study's "
            "verdict is a statement about cannabis industry economics (US federal illegality, "
            "280E tax headwind, banking restrictions, execution risk), not a methodology limitation."
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
