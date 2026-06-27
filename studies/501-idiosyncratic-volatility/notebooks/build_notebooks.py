"""Generate the two narrative notebooks for Study 501 (Idiosyncratic-Volatility).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic figures
run anywhere, offline and deterministic; the real-panel cells use the cached yfinance parquets
under ../_cache/ if present and otherwise quote the frozen headline numbers in ``R``.

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


# Frozen real-panel headline numbers -- mirror of docs/results.md (as-of 2026-06-26).
R = dict(
    n_months=142, year_start=2014, year_end=2025, n_tickers=50,
    ls_gross=-21.54, ls_vol=18.49, ls_sharpe=-1.164, ls_t=-4.293,
    ls_hit=36.6, ls_dd=-94.2, placebo_p=0.000,
    ls_net=-22.24, ls_net_t=-4.434, turnover=0.345, cost_bps=5, borrow_bps=50,
    low_mean=13.15, high_mean=34.69, mkt_mean=14.27,
    avg_ivol_low=13.9, avg_ivol_high=34.8,
    fp_prices="31bdf138449e", fp_spy="b0f3386738bb",
    quintiles={"Q1": 13.15, "Q2": 16.69, "Q3": 15.12, "Q4": 20.51, "Q5": 34.69},
    annual={
        2014: -4.7, 2015: -30.3, 2016: -21.4, 2017: -11.3, 2018: -19.2,
        2019: -10.6, 2020: -34.9, 2021: -21.6, 2022: 10.4, 2023: -49.4,
        2024: -24.7, 2025: -14.1,
    },
    control={-0.06: -2.74, 0.0: -0.54, 0.06: 1.65, 0.12: 3.85, 0.30: 10.15, 0.50: 17.44},
)

# ---------------------------------------------------------------------------
# Shared preamble
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from idiosyncratic_volatility import data, strategy as st

def _have_cache():
    study = os.path.abspath("..")
    return (os.path.exists(os.path.join(study, "_cache", "ivol_prices.parquet"))
            and os.path.exists(os.path.join(study, "_cache", "ivol_spy.parquet")))

HAVE_REAL = _have_cache()
print("yfinance cache present:", HAVE_REAL)

if HAVE_REAL:
    prices, spy = data.fetch_panel()
    res      = st.ivol_long_short(prices, spy, window=252, n_quantiles=5, min_stocks=15)
    res_net  = st.ivol_long_short(prices, spy, window=252, n_quantiles=5, min_stocks=15,
                                  cost_bps=5.0, borrow_ann_bps=50.0)
    s_ls     = st.summary(res["ls_gross"])
    s_net    = st.summary(res_net["ls_net"])
    s_low    = st.summary(res["low_ivol_ret"])
    s_high   = st.summary(res["high_ivol_ret"])
    s_mkt    = st.summary(res["mkt_ret"])
    placebo_p = st.placebo_pvalue(res["ls_gross"])
    print(f"N months: {s_ls['n']} | low-minus-high {s_ls['mean']*100:+.2f}%/yr"
          f" | HAC t {s_ls['tstat']:+.3f} | placebo p {placebo_p:.3f}")
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Idiosyncratic-Volatility -- do 'wild residual' stocks really earn less?\n"
            "### Ang-Hodrick-Xing-Zhang (2006): the IVOL puzzle, tested honestly\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Puzzle replicates%3F: Busted](https://img.shields.io/badge/Puzzle%20replicates%3F-Busted-8b949e?style=flat-square)\n\n"
            "In 2006 Ang, Hodrick, Xing and Zhang published a genuine puzzle. Take each stock's "
            "**idiosyncratic** volatility -- the part of its wiggle that is NOT explained by the "
            "market (the std of the residual after you regress the stock on the market). Standard "
            "theory says this idiosyncratic risk should earn *no* premium: you can diversify it "
            "away, so the market shouldn't pay you for it. Instead AHXZ found something stranger: "
            "high-idiosyncratic-vol stocks earned **lower** future returns. Priced -- and with the "
            "*wrong* sign. That is the **IVOL puzzle**.\n\n"
            "We test it on a large-cap S&P 500 survivor panel via yfinance -- and we name the "
            "survivorship bias up front, because here it is the whole story.\n\n"
            "> **This is the plain-language layer.** Want the rolling residual-vol construction, "
            "the placebo null, and window sweeps? See "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the puzzle replicate (low IVOL beats high IVOL)? | **No -- it inverts.** "
            f"Low-minus-high = **{R['ls_gross']:.1f}%/yr**, HAC *t* = **{R['ls_t']:.2f}**. "
            "High IVOL hugely *out*-earned low IVOL. |\n"
            "| Is the relation statistically real? | **Yes, overwhelmingly -- the wrong way.** "
            f"Placebo p = **{R['placebo_p']:.3f}**. But the sign is opposite the published puzzle. |\n"
            "| Can you trade it? | **No.** The reversed leg (long high-IVOL) is a survivorship "
            "artifact -- you only know which high-IVOL names survive with hindsight. |\n"
            "| Is there a survivorship problem? | **Yes, and it is the entire mechanism.** A "
            "basket of *survivors* keeps the high-IVOL lottery winners (NVDA, TSLA, AMD, META). |\n\n"
            "> The IVOL puzzle is real and replicated internationally on broad, delisting-complete "
            "universes. On 50 large-cap survivors it **flips sign** -- a clean illustration of how "
            "survivorship can manufacture the opposite of a known anomaly."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 -- The claim\n\n"
            "> *\"Stocks with high idiosyncratic volatility relative to the Fama-French model "
            "have abysmally low average returns... a substantial difference of -1.06% per month "
            "between the highest and lowest quintiles.\"*\n\n"
            "-- Ang, Hodrick, Xing & Zhang (2006), *Journal of Finance*\n\n"
            "The intuition for why it is a *puzzle*: idiosyncratic risk is, by definition, "
            "diversifiable. A rational, well-diversified investor holds the market and bears only "
            "systematic risk, so idiosyncratic vol should command **zero** premium. AHXZ find it "
            "commands a *negative* one -- you pay (in foregone return) to hold the wild names. "
            "Crucially, IVOL is the **residual** after removing the market -- not raw total vol "
            "(that's [Study 330](../../330-low-volatility-anomaly/)) and not market beta "
            "(that's [Study 238](../../238-betting-against-beta/))."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 -- So what?\n\n"
            "The IVOL puzzle is one of the most studied anomalies in modern asset pricing. If "
            "idiosyncratic risk is priced, the textbook diversification argument is incomplete, "
            "and a long-low / short-high IVOL portfolio would be a near-pure mispricing harvest. "
            "Competing explanations abound: lottery demand (Bali-Cakici-Whitelaw 2011), "
            "arbitrage asymmetry (Stambaugh-Yu-Yuan 2015), and measurement (Fu 2009 even finds a "
            "*positive* relation with conditional IVOL). Our questions:\n\n"
            "1. **Does the negative low-minus-high spread show up on a realistic large-cap panel?**\n"
            "2. **Is whatever we find a survivorship artifact?**\n"
            "3. **Is any of it tradable forward?**"
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 -- How would we even know?\n\n"
            "Three disciplines keep us honest:\n\n"
            "1. **Residualise, don't just take total vol.** IVOL is the std of the CAPM "
            "*residual* on a trailing 252-day window -- the market beta is stripped out first, so "
            "we are not secretly re-testing low-beta or low-total-vol.\n"
            "2. **One execution lag.** The signal is formed at month end; the position is entered "
            "the *next* trading day. No same-bar fill, no look-ahead.\n"
            "3. **Name the survivorship bias.** The universe is the *current* S&P 500 projected "
            "backwards. The high-IVOL names that *failed* (and which AHXZ's broad universe "
            "includes) are simply absent -- so our high-IVOL leg is stuffed with survivors that, "
            "by selection, did well."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 -- The teardown\n\n"
            "**First, does the IVOL engine work when the puzzle is planted?**"
        ),
        code(
            "# Synthetic positive control: plant a known IVOL puzzle and verify recovery.\n"
            "premiums = [-0.06, 0.0, 0.06, 0.12, 0.30, 0.50]\n"
            "means, ts = [], []\n"
            "for prem in premiums:\n"
            "    sp, mk, _ = data.synthetic_panel(ivol_premium=prem, n_days=3500, seed=501)\n"
            "    rr = st.ivol_long_short(sp, mk, min_stocks=8)\n"
            "    s = st.summary(rr['ls_gross']) if not rr.empty else {'mean': np.nan, 'tstat': np.nan}\n"
            "    means.append(s['mean']*100); ts.append(s['tstat'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col = [GREEN if m > 0 else (RED if m < 0 else GREY) for m in means]\n"
            "ax.bar([str(p) for p in premiums], means, color=col)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for i, (m, t) in enumerate(zip(means, ts)):\n"
            "    ax.text(i, m + (0.4 if m >= 0 else -0.9), f't={t:+.1f}', ha='center', fontsize=8)\n"
            "ax.set_xlabel('planted IVOL premium (low-IVOL alpha)')\n"
            "ax.set_ylabel('low-minus-high (%/yr)')\n"
            "ax.set_title('Synthetic control: engine recovers the puzzle when it is planted')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Monotone in the planted premium, |t|>2 once the effect is large enough.')"
        ),
        md(
            "The engine is faithful: a positive planted premium yields a positive low-minus-high "
            "spread, and the *t*-stat clears 2 once the planted effect is large. So whatever sign "
            "we get on the real tape reflects **the market and the basket** -- not a broken method."
        ),
        md("**Now the honest test on the real yfinance panel.**"),
        code(
            "if HAVE_REAL:\n"
            "    low_m, high_m, mkt_m = s_low['mean']*100, s_high['mean']*100, s_mkt['mean']*100\n"
            "    ls_m, ls_t = s_ls['mean']*100, s_ls['tstat']\n"
            "    qm = st.quintile_means(prices, spy)\n"
            "    qvals = (qm*100).to_dict()\n"
            "else:\n"
            "    low_m, high_m, mkt_m = R['low_mean'], R['high_mean'], R['mkt_mean']\n"
            "    ls_m, ls_t = R['ls_gross'], R['ls_t']\n"
            "    qvals = R['quintiles']\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "# Left: quintile mean returns (the AHXZ table, mirrored)\n"
            "qs = list(qvals.keys()); vs = list(qvals.values())\n"
            "axes[0].bar(qs, vs, color=[GREEN, AMBER, AMBER, AMBER, RED])\n"
            "axes[0].axhline(mkt_m, ls='--', c=GREY, lw=1, label=f'SPY {mkt_m:+.1f}%')\n"
            "axes[0].set_ylabel('mean annual return (%/yr)')\n"
            "axes[0].set_title('Quintile returns INCREASE in IVOL (puzzle inverted)')\n"
            "axes[0].legend()\n"
            "# Right: three legs\n"
            "axes[1].bar(['Low-IVOL\\n(long)', 'SPY', 'High-IVOL\\n(short)'],\n"
            "            [low_m, mkt_m, high_m], color=[GREEN, GREY, RED], width=0.5)\n"
            "axes[1].axhline(mkt_m, ls='--', c=GREY, lw=1)\n"
            "axes[1].set_ylabel('mean annual return (%/yr)')\n"
            "axes[1].set_title(f'low-minus-high {ls_m:+.1f}%/yr | t={ls_t:+.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Low-IVOL {low_m:+.1f}%/yr  vs  High-IVOL {high_m:+.1f}%/yr  (SPY {mkt_m:+.1f}%)')"
        ),
        md(
            f"The puzzle is **reversed**. The high-IVOL quintile earned **+{R['high_mean']:.1f}%/yr** "
            f"against the low-IVOL quintile's **+{R['low_mean']:.1f}%/yr** -- a low-minus-high spread "
            f"of **{R['ls_gross']:.1f}%/yr** at HAC *t* = **{R['ls_t']:.2f}** (the *opposite* sign "
            "to AHXZ). On a survivor basket the names with the wildest residuals -- NVDA, TSLA, AMD, "
            "META -- are exactly the ones that survived *because* their gambles paid off."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal -- NONE (for the puzzle).** The AHXZ puzzle predicts a *positive* "
            f"low-minus-high spread. We get a strongly *negative* one ({R['ls_gross']:.1f}%/yr, "
            f"HAC *t* = {R['ls_t']:.2f}, placebo p = {R['placebo_p']:.3f}). The relation is "
            "statistically overwhelming but points the wrong way -- the published puzzle does not "
            "replicate on this basket.\n"
            "- **Tradability -- MIRAGE.** The only money is in the *reversed* leg (long high-IVOL), "
            f"a survivorship artifact. Sharpe {R['ls_sharpe']:.2f}, max drawdown {R['ls_dd']:.0f}%. "
            "Net of costs it is marginally worse. Nothing here is investable forward.\n"
            "- **Puzzle replicates? -- Busted (survivorship).** The inversion IS the survivorship "
            "mechanism. On a broad, delisting-complete universe AHXZ's negative IVOL-return "
            "relation holds; our survivor sample flips it."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 -- Could you actually trade it?\n\n"
            "No -- and the reason is instructive:\n\n"
            "1. **The 'edge' is hindsight.** The reversed spread (long high-IVOL) only worked "
            "because the *survivors* among high-IVOL names won. Going forward you cannot know "
            "which high-IVOL gamblers survive; many will blow up.\n"
            "2. **The real puzzle requires the failures.** To harvest AHXZ's actual (negative) "
            "premium you need the delisted, distressed, bankrupt high-IVOL names in your short "
            "leg -- the exact names a survivor panel deletes.\n"
            "3. **Drawdowns are savage.** The puzzle spread has a -94% max drawdown on this tape; "
            "the 2023 AI rally alone cost it -49%.\n\n"
            "This study is a worked example of how a clean, statistically powerful result can "
            "still be **worthless** -- because the data was selected on the outcome."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **Delisting-complete universe.** Re-run on CRSP with delisting returns; the sign "
            "should flip back to the AHXZ negative relation.\n"
            "- **Control for lottery demand.** Bali-Cakici-Whitelaw (2011) argue IVOL proxies for "
            "MAX (extreme daily returns) -- see [Study 365](../../365-lottery-max-effect/).\n"
            "- **Conditional IVOL.** Fu (2009) finds a *positive* relation using EGARCH IVOL -- the "
            "sign is measurement-sensitive.\n"
            "- **Compare with the cousins.** [Study 330](../../330-low-volatility-anomaly/) "
            "(total vol) and [Study 238](../../238-betting-against-beta/) (market beta) -- IVOL is "
            "what is left once you remove the beta 238 trades.\n\n"
            "*Think the IVOL puzzle survives once you put the failures back in? Fork this, add a "
            "delisting-complete universe, and show a low-minus-high t > 2 of the **right** sign. "
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
            "# Idiosyncratic-Volatility -- a quantitative teardown\n"
            "### yfinance panel * rolling CAPM-residual vol * quintile sort * HAC inference\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Puzzle replicates%3F: Busted](https://img.shields.io/badge/Puzzle%20replicates%3F-Busted-8b949e?style=flat-square)\n\n"
            "The quantitative companion to the [notebook for the curious]"
            "(01_for_the_curious.ipynb) -- same seven beats, every claim carrying its standard "
            "error. We test Ang-Hodrick-Xing-Zhang (2006): rank by trailing-252-day CAPM-residual "
            "volatility, long the lowest-IVOL quintile, short the highest, dollar-neutral, monthly "
            "rebalance, one execution-day lag.\n\n"
            "> **Not investment advice.** Real data: yfinance daily adjusted-close, 50 S&P 500 "
            "large-caps, 2013-2025, as-of 2026-06-26. Methods in "
            "[`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **Survivorship bias is named:** universe = current S&P 500 projected backwards. "
            "Here it is the entire mechanism -- it inverts the puzzle's sign."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | low-minus-high **{R['ls_gross']:.2f}%/yr**, HAC *t* = "
            f"**{R['ls_t']:.3f}**, placebo p = **{R['placebo_p']:.3f}** -- statistically "
            "overwhelming but the *wrong sign* for the puzzle (it inverts). |\n"
            f"| **Tradability** | `MIRAGE` | Sharpe **{R['ls_sharpe']:.3f}**, max DD "
            f"**{R['ls_dd']:.1f}%**, net ~= gross. Reversed leg is hindsight-only. |\n"
            "| **Puzzle replicates?** | `BUSTED` | Quintile returns *increase* in IVOL; "
            "high-IVOL out-earns low-IVOL by +21.5%/yr -- textbook survivorship inversion. |\n\n"
            "> AHXZ's negative IVOL-return relation is robust on broad, delisting-complete "
            "universes. On 50 large-cap survivors the conditioning-on-survival flips it: the "
            "high-IVOL names that survived are the ones whose idiosyncratic gambles paid off."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 -- The claim, steelmanned\n\n"
            "Let $\\mathrm{IVOL}_{i,t}$ = annualised std of the residual $\\varepsilon_{i,s}$ from "
            "$r_{i,s} - r_f = \\alpha_i + \\beta_i (r_{m,s} - r_f) + \\varepsilon_{i,s}$ estimated "
            "on the trailing 252 daily observations to time $t$. AHXZ assert:\n\n"
            "- **H1 (signal).** $E[r^{\\text{low}} - r^{\\text{high}}] > 0$: the low-IVOL "
            "quintile out-earns the high-IVOL quintile (the negative IVOL-return relation).\n"
            "- **H2 (residual, not beta).** The effect survives removing market beta -- it is "
            "*idiosyncratic*, not a low-beta re-label.\n"
            "- **H3 (tradable).** A long-low / short-high book harvests it net of costs.\n\n"
            "On this survivor panel we **reject H1 with the opposite sign** ($t=-4.29$), confirm "
            "H2 mechanically (we residualise before sorting), and reject H3 (the only positive "
            "spread is the reversed, survivorship-driven one)."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 -- So what? -- the economic stakes\n\n"
            "The IVOL puzzle anchors a literature on limits-to-arbitrage and lottery preferences. "
            "If it were a clean, tradable mispricing, a market-neutral low-minus-high IVOL book "
            "would be a near-pure alpha source. The cautionary tale here is methodological: a "
            "*survivor* basket does not merely shrink the effect (as survivorship usually does) -- "
            "it can **reverse** it, because the high-residual-risk names that survive are a "
            "selected-on-success sample. Any backtest of IVOL on 'today's index, projected back' "
            "is structurally misleading."
        ),

        # ---- BEAT 3 -- PROTOCOL ----------------------------------------------
        md(
            "## 3 -- The protocol\n\n"
            "- **Signal.** Rolling 252-day OLS of each stock's daily excess return on SPY's daily "
            "excess return; IVOL = annualised std of the residual. RF = 3%/yr constant.\n"
            "- **Ranking.** Monthly, sort into 5 quintiles by IVOL (Q1 lowest, Q5 highest).\n"
            "- **Book.** Long Q1 (equal-weight), short Q5 (equal-weight), dollar-neutral.\n"
            "- **Execution lag.** Enter one trading day after the signal date (no same-bar fill).\n"
            "- **Costs.** 5 bps one-way x turnover + 50 bps/yr borrow on the short leg.\n"
            "- **Inference.** Newey-West HAC *t* on monthly low-minus-high; sign-flip placebo "
            "p-value.\n"
            "- **Universe caveat.** 50 large-cap S&P 500 names (yfinance); survivorship-biased."
        ),

        # ---- BEAT 4 -- TEARDOWN ----------------------------------------------
        md("## 4 -- The teardown"),
        md(
            "### 4a -- Positive control: the IVOL engine is a faithful detector\n\n"
            "Sweep the planted IVOL premium. The low-minus-high spread should be monotone "
            "increasing in it, crossing zero at the null and clearing |t|>2 when the planted "
            "effect is large."
        ),
        code(
            "premiums = [-0.06, 0.0, 0.06, 0.12, 0.30, 0.50]\n"
            "means, ts = [], []\n"
            "for prem in premiums:\n"
            "    sp, mk, _ = data.synthetic_panel(ivol_premium=prem, n_days=3500, seed=501)\n"
            "    rr = st.ivol_long_short(sp, mk, min_stocks=8)\n"
            "    s = st.summary(rr['ls_gross']) if not rr.empty else {'mean': np.nan, 'tstat': np.nan}\n"
            "    means.append(s['mean']*100); ts.append(s['tstat'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(premiums, means, 'o-', c=AMBER, lw=2)\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, c=GREY, lw=1, ls=':')\n"
            "for p, m, t in zip(premiums, means, ts):\n"
            "    ax.annotate(f't={t:+.1f}', (p, m), textcoords='offset points', xytext=(0, 7),\n"
            "                ha='center', fontsize=8)\n"
            "ax.set_xlabel('planted IVOL premium'); ax.set_ylabel('low-minus-high (%/yr)')\n"
            "ax.set_title('Positive control: spread monotone in the planted premium')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "### 4b -- Rolling IVOL distribution on the real panel\n\n"
            "What does the cross-section of residual vol look like? We expect a wide spread "
            "(~12% to ~40%+ annualised)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rets = prices.pct_change().dropna(how='all'); spy_r = spy.pct_change().dropna()\n"
            "    iv = st.rolling_ivol(rets.sub(st.RF_DAILY), spy_r.sub(st.RF_DAILY))\n"
            "    last = iv.iloc[-1].dropna()*100\n"
            "    fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "    ax.hist(last, bins=20, color=AMBER, edgecolor='white', alpha=0.85)\n"
            "    ax.axvline(last.median(), c=RED, lw=2, ls='--', label=f'median={last.median():.1f}%')\n"
            "    ax.set_xlabel('annualised idiosyncratic vol (CAPM residual std)')\n"
            "    ax.set_ylabel('count'); ax.set_title('IVOL cross-section (latest snapshot)')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f'IVOL range: {last.min():.1f}% -- {last.max():.1f}% | median {last.median():.1f}%')\n"
            "else:\n"
            "    print(f\"Frozen: avg IVOL low={R['avg_ivol_low']:.1f}% high={R['avg_ivol_high']:.1f}%\")"
        ),
        md("### 4c -- The quintile table (the AHXZ shape, mirrored)"),
        code(
            "if HAVE_REAL:\n"
            "    qm = (st.quintile_means(prices, spy)*100)\n"
            "    qvals = qm.to_dict()\n"
            "else:\n"
            "    qvals = R['quintiles']\n"
            "qs, vs = list(qvals.keys()), list(qvals.values())\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(qs, vs, color=[GREEN, AMBER, AMBER, AMBER, RED])\n"
            "ax.set_ylabel('mean annual return (%/yr)')\n"
            "ax.set_title('Quintile returns rise with IVOL -- the puzzle inverted')\n"
            "for i, v in enumerate(vs):\n"
            "    ax.text(i, v+0.4, f'{v:+.1f}', ha='center', fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Q1(low) ->', ' '.join(f'{k}:{v:+.1f}%' for k, v in qvals.items()), '-> Q5(high)')"
        ),
        md("### 4d -- Low-minus-high: equity curve, drawdown, and the placebo null"),
        code(
            "if HAVE_REAL:\n"
            "    s = res['ls_gross']\n"
            "    eq = (1+s).cumprod(); dd = (eq/eq.cummax()-1)*100\n"
            "    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)\n"
            "    ax1.plot(s.index, eq.values, c=RED, lw=2); ax1.axhline(1, c='k', lw=1, ls='--')\n"
            "    ax1.set_ylabel('cumulative wealth')\n"
            "    ax1.set_title(f'low-minus-high IVOL book | {s_ls[\"mean\"]*100:+.1f}%/yr t={s_ls[\"tstat\"]:+.2f}')\n"
            "    ax2.fill_between(s.index, dd.values, 0, color=RED, alpha=0.4)\n"
            "    ax2.set_ylabel('drawdown (%)'); ax2.set_xlabel('date')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(f'Max DD {dd.min():.1f}% | hit rate {s_ls[\"hit_rate\"]*100:.1f}% | placebo p {placebo_p:.3f}')\n"
            "else:\n"
            "    print(f\"Frozen: max DD {R['ls_dd']:.1f}% | hit {R['ls_hit']:.1f}% | placebo p {R['placebo_p']:.3f}\")"
        ),
        md(
            "The placebo (sign-flip) p-value is ~0: the negative spread is not luck. But a "
            "statistically certain result of the *wrong sign* is not the anomaly -- it is "
            "survivorship dressed as alpha."
        ),
        md("### 4e -- Robustness: window and quantile sweeps"),
        code(
            "if HAVE_REAL:\n"
            "    print('IVOL window sweep (gross low-minus-high):')\n"
            "    for w in (126, 189, 252, 504):\n"
            "        rw = st.ivol_long_short(prices, spy, window=w, min_stocks=15)\n"
            "        sw = st.summary(rw['ls_gross'])\n"
            "        print(f'  {w:3d}d: {sw[\"mean\"]*100:+6.2f}%/yr  t={sw[\"tstat\"]:+.2f}')\n"
            "    print('Quantile-count sweep:')\n"
            "    for nq in (3, 5, 10):\n"
            "        rq = st.ivol_long_short(prices, spy, n_quantiles=nq, min_stocks=max(12, nq*2))\n"
            "        sq = st.summary(rq['ls_gross'])\n"
            "        print(f'  {nq:2d}q: {sq[\"mean\"]*100:+6.2f}%/yr  t={sq[\"tstat\"]:+.2f}')\n"
            "else:\n"
            "    print('Robustness sweeps require the real cache. Frozen baseline:',\n"
            "          f\"{R['ls_gross']:.1f}%/yr at t={R['ls_t']:.2f}\")"
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal `NONE`** -- low-minus-high {R['ls_gross']:.2f}%/yr, HAC *t* = "
            f"{R['ls_t']:.3f}, placebo p = {R['placebo_p']:.3f}. Statistically overwhelming, but "
            "the *opposite* sign to AHXZ. The published puzzle does not replicate here; the "
            "negative relation is the wrong way round, so the puzzle's signal is NONE.\n"
            f"- **Tradability `MIRAGE`** -- Sharpe {R['ls_sharpe']:.3f}, max DD {R['ls_dd']:.1f}%, "
            f"net low-minus-high {R['ls_net']:.2f}%/yr. The only positive spread (long high-IVOL) "
            "is hindsight on which survivors won. Not investable forward.\n"
            "- **Puzzle replicates? `BUSTED`** -- the inversion IS the survivorship mechanism. "
            "Robust to window (126-504d) and bucket count (3-10) -- not a knob artifact, a "
            "structural feature of a survivor basket."
        ),

        # ---- BEAT 6 -- TRADABILITY -------------------------------------------
        md(
            "## 6 -- Could you trade it?\n\n"
            "The reversed spread is not a forward strategy:\n\n"
            "1. **Selection on survival.** Going long high-IVOL only worked because the high-IVOL "
            "*survivors* (NVDA, TSLA, AMD, META) are in the basket. Ex ante you cannot pick them.\n"
            "2. **The genuine puzzle needs the failures.** AHXZ's negative premium lives in the "
            "delisted/distressed high-IVOL names -- exactly what a survivor index omits.\n"
            "3. **Costs and borrow.** Net low-minus-high "
            f"({R['ls_net']:.1f}%/yr) is marginally worse than gross; turnover ~{R['turnover']:.2f} "
            "one-sided/month at 5 bps plus 50 bps/yr borrow.\n"
            "4. **Drawdown.** A -94% peak-to-trough on the puzzle spread; the 2023 AI rally cost "
            "it -49% alone."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rc = res.copy(); rc['year'] = rc.index.year\n"
            "    ann = rc.groupby('year')['ls_gross'].apply(lambda x: float((1+x).prod()-1))*100\n"
            "    fig, ax = plt.subplots(figsize=(10, 3.8))\n"
            "    ax.bar(ann.index, ann.values, color=[GREEN if v > 0 else RED for v in ann.values])\n"
            "    ax.axhline(0, c='k', lw=1)\n"
            "    ax.set_ylabel('low-minus-high (%/yr)'); ax.set_xlabel('year')\n"
            "    ax.set_title('Year-by-year: negative in 11 of 12 years (2022 bear = only positive)')\n"
            "    plt.tight_layout(); plt.show()\n"
            "    print(ann.round(1).to_string())\n"
            "else:\n"
            "    print('Frozen year-by-year:', R['annual'])"
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **Delisting-complete universe.** Re-run on CRSP with delisting returns -- the sign "
            "should revert to AHXZ's negative relation.\n"
            "- **Lottery-demand control.** Bali-Cakici-Whitelaw (2011): IVOL proxies for MAX. See "
            "[Study 365 -- Lottery-MAX-Effect](../../365-lottery-max-effect/).\n"
            "- **Conditional IVOL.** Fu (2009) finds a *positive* relation with EGARCH IVOL -- the "
            "sign is measurement-sensitive; try a forward-looking estimator.\n"
            "- **The cousins.** [Study 330 -- Low-Volatility-Anomaly](../../330-low-volatility-anomaly/) "
            "(total vol) and [Study 238 -- Betting-Against-Beta](../../238-betting-against-beta/) "
            "(market beta) -- IVOL is the residual once 238's beta is removed.\n\n"
            "*Think the IVOL puzzle survives once the failures are back in the universe? Fork "
            "this, swap in a delisting-complete tape, and show a low-minus-high t > 2 of the "
            "**right** sign. That is the bar.*"
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
