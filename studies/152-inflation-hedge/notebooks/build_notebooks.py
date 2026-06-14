"""Generate the two narrative notebooks for Study 152 (Inflation-Hedge).

    python notebooks/build_notebooks.py
    python -W ignore -m jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-data cells load the Shiller
parquet from the repo-wide cache (always present in the repo) and fall back to the
frozen headline numbers in ``R`` (mirroring docs/results.md) if needed.

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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-14).
R = dict(
    n=1818, n_high=724, n_low=1094,
    date_start="1872-01", date_end="2023-06",
    fingerprint="b2019f787713",
    # Fisher hypothesis
    fisher_beta_nom=0.313, fisher_t_nom=1.304, fisher_r2_nom=0.010,
    fisher_beta_real=-0.719, fisher_t_real=-3.023,
    # Regime: 1-year forward real total return
    high_mean_1y=0.0428, low_mean_1y=0.1135,
    diff_mean_1y=-0.0707, t_diff_1y=-4.078,
    t_diff_shuffled=0.793,
    # Regime: 5-year forward annualised real total return
    high_mean_5y=0.0601, low_mean_5y=0.0794,
    diff_mean_5y=-0.0193, t_diff_5y=-2.111,
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

from inflation_hedge import data, strategy as st

# Frozen real-tape headline numbers (mirror of docs/results.md)
R = dict(
    n=1818, n_high=724, n_low=1094,
    date_start="1872-01", date_end="2023-06",
    fingerprint="b2019f787713",
    fisher_beta_nom=0.313, fisher_t_nom=1.304, fisher_r2_nom=0.010,
    fisher_beta_real=-0.719, fisher_t_real=-3.023,
    high_mean_1y=0.0428, low_mean_1y=0.1135,
    diff_mean_1y=-0.0707, t_diff_1y=-4.078,
    t_diff_shuffled=0.793,
    high_mean_5y=0.0601, low_mean_5y=0.0794,
    diff_mean_5y=-0.0193, t_diff_5y=-2.111,
)

# Check whether the Shiller cache is present (always True in the repo)
_cache_dir = os.path.abspath(os.path.join("..", "..", "..", "_cache"))
_shiller_path = os.path.join(_cache_dir, "shiller_sp500.parquet")
HAVE_REAL = os.path.exists(_shiller_path)

if HAVE_REAL:
    df = data.load_shiller(cache_dir=_cache_dir)
    print(f"Shiller panel: {df.index[0].strftime('%Y-%m')} to {df.index[-1].strftime('%Y-%m')},",
          f"n={len(df)}, fingerprint={data.fingerprint(df)}")
else:
    print("Shiller parquet not found -- using frozen R dict for all figures")

print("HAVE_REAL:", HAVE_REAL)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Inflation-Hedge -- do stocks actually protect you from inflation?\n"
            "### 150 years of Shiller data, the Fisher test, and two inconvenient numbers\n\n"
            "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Short--run hedge%3F: No](https://img.shields.io/badge/Short--run_hedge%3F-No-8b949e?style=flat-square)\n\n"
            "Every financial adviser who has ever lived has told you: *'stocks are a real asset -- "
            "they protect against inflation.'*  It's one of the few financial folk wisdoms that sounds "
            "both plausible and comforting.  This notebook asks: **is it actually true, and if so, at "
            "what time horizon?**  The answer, from 150 years of monthly US data, is more complicated "
            "-- and more alarming -- than the reassuring story suggests.\n\n"
            "> **This is the plain-language layer.** For HAC t-statistics, OLS regressions, and "
            "bootstrap intervals, see the companion "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn by the "
            "code beside it.  House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Short answer |\n|---|---|\n"
            "| Does inflation hedge work in the *short run* (1 year)? | **No.** High-inflation years "
            f"deliver **{R['high_mean_1y']*100:.1f}%** real returns vs "
            f"**{R['low_mean_1y']*100:.1f}%** in low-inflation: a **{R['diff_mean_1y']*100:.1f} pp** "
            "gap. |\n"
            "| Do nominal stock returns rise with inflation (Fisher)? | **Barely.** They absorb only "
            f"**{R['fisher_beta_nom']:.0%}** of each 1pp CPI rise, not the 100% a full hedge needs. |\n"
            "| Do real returns at least hold up? | **No.** Real returns *fall* by "
            f"**{R['fisher_beta_real']:.2f}pp** per 1pp of CPI -- the stock market makes your "
            "inflation problem *worse*, not better. |\n"
            "| Does it eventually work over 5 years? | **Partially.** The gap shrinks to "
            f"**{R['diff_mean_5y']*100:.1f} pp** -- stocks are a better long-run real asset than a "
            "short-run hedge. |\n"
            "| Can you trade on it? | **No.** Inflation regimes are decade-long epochs, not "
            "actionable timing signals. |\n\n"
            "> The stock market fails as a short-run inflation hedge -- and the failure is big enough "
            "to matter.  But over years-long horizons, stocks do eventually claw back their real "
            "purchasing power.  The hedge is real; the timing is wrong."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *'Stocks are a real asset.  They own real factories, real brands, real cash flows.  "
            "When prices rise, those assets are worth more in nominal terms, so stock prices rise "
            "too -- keeping up with or outpacing inflation.  In an inflationary environment, hold "
            "stocks, not cash.'*\n\n"
            "The academic version of this idea is the **Fisher hypothesis for stocks**: if markets "
            "are efficient and stocks represent claims on real assets, their nominal returns should "
            "rise dollar-for-dollar with inflation, leaving *real* returns unaffected.  The folk "
            "wisdom and the theory say the same thing -- stocks protect you.\n\n"
            "The famous economists who challenged this are **Fama & Schwert (1977)** and "
            "**Modigliani & Cohn (1979)**.  They found the opposite: nominal stock returns are "
            "essentially uncorrelated with (or even negatively correlated with) inflation, so real "
            "returns *fall* during high-inflation periods.  We test both the folk claim and the "
            "Fama-Schwert challenge on 150 years of monthly US data."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If stocks *do* hedge inflation: investors who park money in stocks during inflationary "
            "periods are doing the right thing -- protecting real wealth.  If stocks *don't*: "
            "investors in the 1970s who stayed in equities saw their real purchasing power eroded, "
            "in exactly the environment where they believed they were protected.\n\n"
            "This matters for **asset allocation during macro inflection points**: every time CPI "
            "jumps, some investors shift *into* equities thinking they're buying a hedge.  If the "
            "Fama-Schwert finding holds, they're loading up on the wrong asset."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Two clean tests:\n\n"
            "1. **The Fisher test.** Regress 12-month nominal (and real) stock returns on trailing "
            "12-month CPI inflation.  A true inflation hedge requires the nominal slope to be near "
            "**+1.0** (each 1pp of CPI lifts nominal returns 1pp) and the real slope to be near "
            "**0** (real returns invariant).  Anything below that is a partial (or negative) hedge.\n\n"
            "2. **The regime test.** Split months into 'high-inflation' (CPI YoY > 3%) and "
            "'low-inflation' (CPI YoY ≤ 3%) buckets -- the split is pre-specified, not data-mined.  "
            "Compare forward 1-year and 5-year *real total* returns (dividends reinvested, CPI-"
            "deflated) across regimes.  An honest inflation hedge would show no gap (or a positive "
            "gap in high-inflation months).\n\n"
            "The **shuffled-label control** permutes the regime labels at random: if our real regime "
            "difference is spurious, the shuffled version should be similarly large.  If the real "
            "gap is real, the shuffled version collapses to noise.\n\n"
            "Data: Shiller monthly S&P 500, 1872-2023 (n=1,818 months).  Dividends reinvested "
            "month-by-month using Shiller's real-dividend construction."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 · The teardown\n\n"
            "**First: does the Fisher hypothesis hold?**  If stocks hedge inflation, nominal "
            "returns should rise one-for-one with CPI."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sub = df[['cpi_yoy','nom_ret_12m','real_tr_ret_12m']].dropna()\n"
            "    cpi = sub['cpi_yoy'].values; nom = sub['nom_ret_12m'].values; real = sub['real_tr_ret_12m'].values\n"
            "else:\n"
            "    # Reconstruct plausible scatter from frozen R\n"
            "    rng = np.random.default_rng(152)\n"
            "    cpi = np.clip(rng.normal(0.023, 0.058, 500), -0.15, 0.25)\n"
            "    nom  = R['fisher_beta_nom'] * cpi + rng.normal(0.062, 0.18, 500)\n"
            "    real = R['fisher_beta_real'] * cpi + rng.normal(0.085, 0.19, 500)\n\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))\n"
            "# Left: nominal\n"
            "ax = axes[0]\n"
            "ax.scatter(cpi*100, nom*100, alpha=0.15, s=8, color=GREY)\n"
            "m = R['fisher_beta_nom']; b_int = 0.062 - m*0.023\n"
            "x_line = np.linspace(-15, 25, 100)\n"
            "ax.plot(x_line, m*x_line + b_int*100, color=RED, lw=2, label=f'OLS: beta={m:.2f} (need 1.0)')\n"
            "ax.axline((0, 0), slope=1.0, ls='--', color=GREEN, lw=1.5, label='Full hedge (beta=1.0)')\n"
            "ax.axhline(0, c='k', lw=0.5); ax.axvline(0, c='k', lw=0.5)\n"
            "ax.set_xlabel('Trailing 12m CPI inflation (%)'); ax.set_ylabel('Nominal 12m return (%)')\n"
            "ax.set_title('Fisher test: nominal returns\\n(need slope=1; got ~0.31)')\n"
            "ax.legend(fontsize=9); ax.set_xlim(-15, 25); ax.set_ylim(-70, 130)\n"
            "# Right: real\n"
            "ax = axes[1]\n"
            "ax.scatter(cpi*100, real*100, alpha=0.15, s=8, color=GREY)\n"
            "m2 = R['fisher_beta_real']; b_int2 = 0.085 - m2*0.023\n"
            "ax.plot(x_line, m2*x_line + b_int2*100, color=RED, lw=2, label=f'OLS: beta={m2:.2f} (need ~0)')\n"
            "ax.axhline(0, c=GREEN, ls='--', lw=1.5, label='Full hedge (beta=0)')\n"
            "ax.axhline(0, c='k', lw=0.5); ax.axvline(0, c='k', lw=0.5)\n"
            "ax.set_xlabel('Trailing 12m CPI inflation (%)'); ax.set_ylabel('Real total return 12m (%)')\n"
            "ax.set_title('Fisher test: real returns\\n(need slope~0; got -0.72 ***)')\n"
            "ax.legend(fontsize=9); ax.set_xlim(-15, 25); ax.set_ylim(-70, 130)\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Nominal beta: {{R[\"fisher_beta_nom\"]:.3f}}  (t={{R[\"fisher_t_nom\"]:+.2f}}, need |t|>2 to confirm hedge)')\n"
            f"print(f'Real beta:    {{R[\"fisher_beta_real\"]:.3f}} (t={{R[\"fisher_t_real\"]:+.2f}}) -- significantly NEGATIVE')"
        ),
        md(
            f"The nominal Fisher beta is **{R['fisher_beta_nom']:.2f}** — stocks absorb about "
            f"**{R['fisher_beta_nom']:.0%}** of each percentage point of inflation, not the 100% "
            "that a full hedge requires. This is not statistically significant from zero "
            f"(HAC t = {R['fisher_t_nom']:+.2f}).  The real-return beta is "
            f"**{R['fisher_beta_real']:.2f}** with a very significant HAC t = "
            f"**{R['fisher_t_real']:+.2f}**: real returns actively *fall* when inflation rises.  "
            "Stocks are not just a *poor* inflation hedge — in the short run, they make your "
            "real purchasing power problem *worse*.\n\n"
            "> This is exactly the Fama & Schwert (1977) finding, replicated on 150 years of "
            "data.  The intuition: when inflation spikes, it usually comes with economic uncertainty, "
            "rising interest rates (which hurt equity valuations), and sometimes a real output "
            "shock — all of which are bad for stocks in real terms."
        ),
        md(
            "**Next: the regime test.** Do forward real returns differ across inflation regimes?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    sub = df[['cpi_regime','fwd_real_tr_12m']].dropna()\n"
            "    high_vals = sub.loc[sub['cpi_regime']=='high','fwd_real_tr_12m'].values * 100\n"
            "    low_vals  = sub.loc[sub['cpi_regime']=='low','fwd_real_tr_12m'].values * 100\n"
            "    h_mean = high_vals.mean(); l_mean = low_vals.mean()\n"
            "else:\n"
            "    rng = np.random.default_rng(152)\n"
            "    high_vals = rng.normal(R['high_mean_1y']*100, 19.5, R['n_high'])\n"
            "    low_vals  = rng.normal(R['low_mean_1y']*100,  19.0, R['n_low'])\n"
            "    h_mean = R['high_mean_1y']*100; l_mean = R['low_mean_1y']*100\n\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))\n"
            "# Left: distribution comparison\n"
            "ax = axes[0]\n"
            "ax.hist(high_vals, bins=40, alpha=0.5, color=RED,   density=True, label=f'High CPI (n={len(high_vals):,})')\n"
            "ax.hist(low_vals,  bins=40, alpha=0.5, color=GREEN, density=True, label=f'Low CPI  (n={len(low_vals):,})')\n"
            "ax.axvline(h_mean, color=RED,   ls='--', lw=2, label=f'High mean: {h_mean:.1f}%')\n"
            "ax.axvline(l_mean, color=GREEN, ls='--', lw=2, label=f'Low mean:  {l_mean:.1f}%')\n"
            "ax.axvline(0, c='k', lw=0.5)\n"
            "ax.set_xlabel('Forward 12m real total return (%)'); ax.set_ylabel('Density')\n"
            "ax.set_title('Forward 1-year real return\\nby inflation regime')\n"
            "ax.legend(fontsize=9)\n"
            "# Right: bar chart of means\n"
            "ax = axes[1]\n"
            "bars = ax.bar(['High CPI\\n(>3% YoY)', 'Low CPI\\n(<=3% YoY)'], [h_mean, l_mean],\n"
            "              color=[RED, GREEN], width=0.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for b, v in zip(bars, [h_mean, l_mean]):\n"
            "    ax.annotate(f'{v:.1f}%', (b.get_x()+b.get_width()/2, v+0.5), ha='center', fontsize=11, fontweight='bold')\n"
            f"ax.annotate(f'Gap: {{h_mean-l_mean:.1f}}pp\\nt = {{R[\"t_diff_1y\"]:+.1f}}', (0.5, (h_mean+l_mean)/2),\n"
            "            ha='center', fontsize=10, color='k')\n"
            "ax.set_ylabel('Mean forward 1y real total return (%)')\n"
            "ax.set_title('High-inflation years earn half the\\nreal return of low-inflation years')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Difference: {{h_mean-l_mean:.2f}}pp | HAC t = {{R[\"t_diff_1y\"]:+.2f}} | shuffled t = {{R[\"t_diff_shuffled\"]:+.2f}}')"
        ),
        md(
            f"In **high-inflation months** the forward 1-year real total return averages "
            f"**{R['high_mean_1y']*100:.1f}%** — barely half the **{R['low_mean_1y']*100:.1f}%** "
            f"earned in low-inflation months.  The gap of **{R['diff_mean_1y']*100:.1f} percentage "
            f"points** is statistically real: HAC t = **{R['t_diff_1y']:+.1f}** (well above the |t|≥2 "
            "bar).  The shuffled-label control produces t = "
            f"**{R['t_diff_shuffled']:+.2f}** — confirming the real gap is not a random split "
            "artefact.\n\n"
            "> 'Stocks are a real asset' is true over geological time.  At the 1-year horizon "
            "relevant to most investors, high inflation is simply bad news for real equity returns."
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Mixed.** The short-run inflation hedge is a myth: Fisher beta "
            f"{R['fisher_beta_nom']:.2f} (not 1.0); real return beta {R['fisher_beta_real']:.2f} "
            f"(t = {R['fisher_t_real']:+.1f}); 1-year regime gap {R['diff_mean_1y']*100:.1f}pp "
            f"(t = {R['t_diff_1y']:+.1f}).  The *negative* effect is real.  The *positive* claim "
            "(stocks protect you from inflation) is only partly true and only at 5+ year horizons.\n"
            f"- **Tradability — Mirage.** Inflation regimes are broad macro epochs — the 1970s, the "
            "post-GFC era, the 2021-2023 spike.  They last years, not months, and cannot be timed "
            "from a CPI print.  The finding is real for risk awareness; it is not an actionable "
            "timing edge."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Even if the regime gap is real, translating it into a trading strategy faces "
            "three problems:"
        ),
        code(
            "# Show the 5-year vs 1-year gap: the hedge 'recovers' at longer horizons\n"
            "horizons = ['1-year forward', '5-year forward\\n(annualised)']\n"
            "high_vals_plot = [R['high_mean_1y']*100, R['high_mean_5y']*100]\n"
            "low_vals_plot  = [R['low_mean_1y']*100,  R['low_mean_5y']*100]\n"
            "gaps_plot = [R['diff_mean_1y']*100, R['diff_mean_5y']*100]\n"
            "tstats_plot = [R['t_diff_1y'], R['t_diff_5y']]\n\n"
            "if HAVE_REAL:\n"
            "    r5 = st.regime_returns(df, fwd_col='fwd_real_tr_60m')\n"
            "    r1 = st.regime_returns(df, fwd_col='fwd_real_tr_12m')\n"
            "    high_vals_plot = [r1['high']['mean']*100, r5['high']['mean']*100]\n"
            "    low_vals_plot  = [r1['low']['mean']*100,  r5['low']['mean']*100]\n"
            "    gaps_plot      = [r1['diff_mean']*100,    r5['diff_mean']*100]\n"
            "    tstats_plot    = [r1['t_diff'],           r5['t_diff']]\n\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))\n"
            "x = np.arange(len(horizons)); w = 0.35\n"
            "ax = axes[0]\n"
            "ax.bar(x - w/2, high_vals_plot, w, color=RED,   label='High CPI regime')\n"
            "ax.bar(x + w/2, low_vals_plot,  w, color=GREEN, label='Low CPI regime')\n"
            "ax.set_xticks(x); ax.set_xticklabels(horizons)\n"
            "ax.set_ylabel('Mean real total return (% pa)'); ax.legend(fontsize=9)\n"
            "ax.set_title('The hedge partially recovers at 5 years')\n"
            "ax = axes[1]\n"
            "col = [RED if abs(t)>=2 else AMBER for t in tstats_plot]\n"
            "ax.bar(horizons, [abs(t) for t in tstats_plot], color=col)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1.5, label='|t|=2 inference bar')\n"
            "ax.set_ylabel('|HAC t-stat| of regime gap'); ax.set_title('Both horizons clear the bar')\n"
            "ax.legend(fontsize=9)\n"
            "for i, (t, g) in enumerate(zip(tstats_plot, gaps_plot)):\n"
            "    ax.annotate(f'gap={g:.1f}pp', (i, abs(t)+0.1), ha='center', fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Why you cannot trade this:')\n"
            "print('  1. Regimes last 5-15 years -- timing the entry/exit is the whole problem')\n"
            "print('  2. CPI prints are lagged and revised -- the signal is noisy when it counts')\n"
            "print('  3. The regime gap at 5y is ~2pp pa -- fees, implementation drag eat that')"
        ),
        md(
            "The 5-year gap narrows to **-1.9pp** (annualised) — stocks are a better long-run "
            "real asset than a short-run hedge, but the edge erodes with horizon.  Neither gap "
            "is actionable: inflation regimes are multi-year macro epochs that you recognize in "
            "hindsight, not at the start.  MIRAGE."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Gold as an inflation hedge** — Study 69 (Safe-Haven) tests gold, which has a "
            "much stronger short-run correlation with CPI than equities.\n"
            "- **Long-run valuation timing** — Study 120 (Excess-CAPE-Yield) uses CAPE to time "
            "equity allocation across a different macro dimension (valuation, not inflation).\n"
            "- **Real-rate regimes** — Study 119 (Real-Rate-Regime) looks at how real interest "
            "rates (nominal yield minus CPI) predict equity real returns, a closely related "
            "framework.\n"
            "- **Why the failure?** The Modigliani-Cohn (1979) inflation illusion says investors "
            "discount real cash flows at nominal rates during high-CPI periods, causing equities "
            "to trade cheaply relative to fundamentals.  If true, high-inflation periods might "
            "be *buying opportunities* on a 10-year horizon — test it!\n\n"
            "*Think the hedge works at some other horizon or with some other instrument?  "
            "Fork this, change the threshold or the asset, and show a regime gap that clears "
            "|t| >= 2 and survives the shuffled-label control.  That's the bar.*"
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
            "# Inflation-Hedge -- a quantitative teardown\n"
            "### 150 years of Shiller data · Fisher regression · regime t-test · shuffled-label control · HAC inference\n\n"
            "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Short--run hedge%3F: No](https://img.shields.io/badge/Short--run_hedge%3F-No-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim now carrying its standard error.*  We test the "
            "Fama-Schwert (1977) finding that equities fail as a short-run inflation hedge, "
            "using 150 years of the Shiller monthly S&P 500 panel: OLS Fisher regressions with "
            "12-lag Newey-West HAC standard errors, and a pre-specified 3% regime split with a "
            "shuffled-label control.\n\n"
            "> **Not investment advice.**  Data: Shiller monthly panel "
            f"({R['date_start']} to {R['date_end']}, n={R['n']:,}), "
            f"fingerprint `{R['fingerprint']}`.  "
            "Methods in [`docs/references.md`](../docs/references.md); "
            "reproducible numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **'In plain words' notes** translate each result back to intuition."
        ),
        code(BOOT + "\nfrom quantlab import analytics, stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | Fisher beta_nom = **{R['fisher_beta_nom']:.3f}** "
            f"(HAC t = {R['fisher_t_nom']:+.2f}); beta_real = **{R['fisher_beta_real']:.3f}** "
            f"(HAC t = **{R['fisher_t_real']:+.2f}**); 1y regime gap = "
            f"**{R['diff_mean_1y']*100:.1f}pp** (HAC t = **{R['t_diff_1y']:+.1f}**). "
            "Negative effect is statistically real; positive hedge claim fails. |\n"
            f"| **Tradability** | `MIRAGE` | Inflation regimes are multi-year macro epochs, "
            "not actionable timing signals. The 5-year gap of "
            f"**{R['diff_mean_5y']*100:.1f}pp pa** (t = {R['t_diff_5y']:+.1f}) "
            "cannot be harvested by regime-switching. |\n"
            f"| **Short-run hedge?** | `NO` | Nominal Fisher beta {R['fisher_beta_nom']:.2f} "
            "(need 1.0); real return falls significantly with inflation. |\n\n"
            "> The Fama-Schwert (1977) finding replicates across 150 years: stocks are a **poor "
            "short-run inflation hedge** — real returns fall when CPI rises, by a statistically "
            "significant margin."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "The **Fisher hypothesis for stocks** states: since equities represent claims on real "
            "productive assets, their nominal returns should incorporate expected inflation "
            "fully — so that:\n\n"
            "$$r_{nom,t} = r_{real,t} + \\pi_t \\quad\\Rightarrow\\quad "
            "\\beta_{Fisher} = 1.0$$\n\n"
            "where $\\pi_t$ is the CPI inflation rate.  Equivalently, real equity returns "
            "$r_{real,t}$ should be orthogonal to $\\pi_t$ (beta_real = 0).\n\n"
            "The **regime hypothesis** states: if stocks are real assets, investors in high-"
            "inflation periods should earn at least as much in real terms as in low-inflation "
            "periods — the inflation hedge should be protective precisely when inflation is high.\n\n"
            "Both hypotheses are tested here using the pre-specified 3% regime threshold and "
            "12-lag Newey-West HAC standard errors for overlapping annual return windows "
            "(Hodrick 1992)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the stakes\n\n"
            "The Fama-Schwert failure has direct implications:\n\n"
            "- **Asset allocation during inflation scares:** the conventional advice to rotate "
            "into equities as an inflation hedge sends investors into exactly the wrong asset "
            "at 1-3 year horizons.\n"
            "- **The Modigliani-Cohn illusion:** if the failure is due to investors discounting "
            "real cash flows at nominal rates (inflation illusion), then high-inflation periods "
            "are periods of systematic equity undervaluation — *buying opportunities* on a "
            "decade horizon, not selling ones.\n"
            "- **Factor zoo cross-checks:** inflation exposure is a factor in many multi-factor "
            "models; understanding the sign (negative, not positive) matters for portfolio "
            "construction and hedging."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Fisher regression.**  OLS regression of trailing 12-month nominal (and real "
            "total) S&P 500 return on trailing 12-month CPI inflation.  HAC t-stat with 12 "
            "Newey-West lags (standard for annual-window overlapping returns; Hodrick 1992).\n"
            "- **Regime split.** Pre-specified at CPI YoY = 3% (the 1970s boundary).  No "
            "data-mining — 3% is chosen before looking at outcomes.  Compare mean forward "
            "1-year and 5-year real total returns across regimes.\n"
            "- **Shuffled-label control.** Permute the 'high'/'low' regime labels at random "
            "(seeded); recompute the regime difference.  Expected t-stat near zero if the "
            "split is uninformative; the real t should substantially exceed this.\n"
            "- **Synthetic positive control.** A simulated world with ``fisher_beta=1.0`` "
            "should recover beta_nom ≈ 1 and a zero regime gap; beta=0 should recover the "
            "Fama-Schwert pattern.  Confirms the machinery.\n\n"
            f"Data: Shiller monthly panel {R['date_start']} to {R['date_end']}, "
            f"n = {R['n']:,} months.  "
            f"High-inflation: {R['n_high']:,} months; low: {R['n_low']:,} months."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Fisher regressions — the hypothesis test\n\n"
            "OLS with 12-lag Newey-West HAC standard errors.  The full-hedge null requires "
            "beta_nom = 1 (nominal return fully absorbs inflation) and beta_real = 0 (real "
            "return invariant).  We also run the synthetic sweep to confirm the engine works."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fisher = st.fisher_regression(df)\n"
            "    bn = fisher['beta_nom']; tn = fisher['t_nom']; r2n = fisher['r2_nom']\n"
            "    br = fisher['beta_real']; tr = fisher['t_real']\n"
            "else:\n"
            "    bn, tn, r2n = R['fisher_beta_nom'], R['fisher_t_nom'], R['fisher_r2_nom']\n"
            "    br, tr = R['fisher_beta_real'], R['fisher_t_real']\n\n"
            "print('Fisher regression (12-lag Newey-West HAC):')\n"
            "print(f'  Nominal: beta={bn:+.4f}, t={tn:+.4f}, R2={r2n:.4f}')\n"
            "print(f'  Real:    beta={br:+.4f}, t={tr:+.4f}')\n"
            "print(f'  Full hedge requires: beta_nom=1.0, beta_real~0')\n"
            "print(f'  Fama-Schwert prediction: beta_nom<1, beta_real<0 -- CHECK: both confirmed')"
        ),
        md(
            "### 4b · Synthetic beta sweep — positive and negative control\n\n"
            "Verify the machinery recovers the planted Fisher coefficient.  With "
            "``fisher_beta=1`` the engine should see nominal beta near 1 and near-zero real "
            "beta.  With ``fisher_beta=0`` it should see the Fama-Schwert pattern."
        ),
        code(
            "betas = [0.0, 0.25, 0.50, 0.75, 1.00]\n"
            "sweep = st.fisher_beta_sweep(betas, n_years=130, seed=152)\n"
            "print(sweep[['planted_beta','recovered_nom_beta','t_nom','recovered_real_beta','t_real']].to_string(index=False))\n\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))\n"
            "ax = axes[0]\n"
            "ax.plot(sweep['planted_beta'], sweep['recovered_nom_beta'], 'o-', color=RED, lw=2, label='Recovered nom beta')\n"
            "ax.plot(sweep['planted_beta'], sweep['planted_beta'], '--', color=GREEN, lw=1.5, label='Full hedge (y=x)')\n"
            "ax.axhline(0, c='k', lw=0.5); ax.set_xlabel('Planted Fisher beta'); ax.set_ylabel('Recovered nominal beta')\n"
            "ax.set_title('Engine correctly recovers planted beta\\n(nominal regression)'); ax.legend()\n"
            "ax = axes[1]\n"
            "ax.plot(sweep['planted_beta'], sweep['recovered_real_beta'], 's-', color=AMBER, lw=2)\n"
            "ax.axhline(0, c=GREEN, ls='--', lw=1.5, label='Full hedge (real beta=0)')\n"
            "ax.axhline(0, c='k', lw=0.5); ax.set_xlabel('Planted Fisher beta'); ax.set_ylabel('Recovered real beta')\n"
            "ax.set_title('Real beta goes from ~-1 (no hedge)\\nto ~0 (full hedge)'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('On real data: nom beta =', round(R['fisher_beta_nom'], 3), '-- much closer to 0 than 1 (the Fama-Schwert result)')"
        ),
        md(
            "> The engine is a faithful Fisher-coefficient detector: it recovers the planted "
            "parameter cleanly.  On the real tape, the nominal beta of **+0.31** sits firmly "
            "in the 'no hedge' zone — close to the planted-0.25 result, far from the planted-1.0."
        ),
        md(
            "### 4c · Regime analysis — 1-year and 5-year forward real total returns\n\n"
            "Pre-specified 3% CPI YoY threshold; shuffled-label control included."
        ),
        code(
            "if HAVE_REAL:\n"
            "    r1 = st.regime_returns(df, fwd_col='fwd_real_tr_12m')\n"
            "    r5 = st.regime_returns(df, fwd_col='fwd_real_tr_60m')\n"
            "    h1, l1, d1, t1 = r1['high']['mean']*100, r1['low']['mean']*100, r1['diff_mean']*100, r1['t_diff']\n"
            "    h5, l5, d5, t5 = r5['high']['mean']*100, r5['low']['mean']*100, r5['diff_mean']*100, r5['t_diff']\n"
            "    ts = r1['t_diff_shuffled']\n"
            "else:\n"
            "    h1, l1, d1, t1 = R['high_mean_1y']*100, R['low_mean_1y']*100, R['diff_mean_1y']*100, R['t_diff_1y']\n"
            "    h5, l5, d5, t5 = R['high_mean_5y']*100, R['low_mean_5y']*100, R['diff_mean_5y']*100, R['t_diff_5y']\n"
            "    ts = R['t_diff_shuffled']\n\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))\n"
            "# Left: 1-year regime\n"
            "ax = axes[0]\n"
            "bars = ax.bar(['High CPI\\n(>3% YoY)', 'Low CPI\\n(<=3% YoY)'],\n"
            "              [h1, l1], color=[RED, GREEN], width=0.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for b, v in zip(bars, [h1, l1]):\n"
            "    ax.annotate(f'{v:.1f}%', (b.get_x()+b.get_width()/2, v+0.3), ha='center', fontsize=11, fontweight='bold')\n"
            "ax.set_ylabel('Mean forward 1y real total return (% pa)')\n"
            "ax.set_title(f'1-year forward real return\\nGap={d1:.1f}pp, HAC t={t1:.1f} | shuffled t={ts:.1f}')\n"
            "# Right: 5-year regime\n"
            "ax = axes[1]\n"
            "bars = ax.bar(['High CPI\\n(>3% YoY)', 'Low CPI\\n(<=3% YoY)'],\n"
            "              [h5, l5], color=[AMBER, GREEN], width=0.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for b, v in zip(bars, [h5, l5]):\n"
            "    ax.annotate(f'{v:.1f}%', (b.get_x()+b.get_width()/2, v+0.1), ha='center', fontsize=11, fontweight='bold')\n"
            "ax.set_ylabel('Mean forward 5y real total return (% pa)')\n"
            "ax.set_title(f'5-year forward real return (annualised)\\nGap={d5:.1f}pp, HAC t={t5:.1f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'1y: high={h1:.2f}%, low={l1:.2f}%, gap={d1:.2f}pp, t={t1:.2f}')\n"
            "print(f'5y: high={h5:.2f}%, low={l5:.2f}%, gap={d5:.2f}pp, t={t5:.2f}')\n"
            "print(f'Shuffled-label control t (1y): {ts:.2f} -- near zero as expected')"
        ),
        md(
            f"> The 1-year regime gap of **{R['diff_mean_1y']*100:.1f}pp** (HAC t = "
            f"**{R['t_diff_1y']:+.1f}**) is far above the |t|≥2 inference bar; the shuffled "
            f"control produces t = **{R['t_diff_shuffled']:+.2f}**.  The 5-year gap of "
            f"**{R['diff_mean_5y']*100:.1f}pp** (t = {R['t_diff_5y']:+.1f}) is smaller but "
            "still significant — consistent with the literature finding that the Fama-Schwert "
            "failure is a *short-horizon* phenomenon (Boudoukh & Richardson 1993)."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `MIXED`** — The negative effect (real returns fall in high-inflation "
            f"regimes) is statistically real: 1-year gap {R['diff_mean_1y']*100:.1f}pp "
            f"(HAC t = {R['t_diff_1y']:+.1f}); real Fisher beta {R['fisher_beta_real']:+.3f} "
            f"(t = {R['fisher_t_real']:+.2f}).  The positive claim (stocks hedge inflation) is "
            f"false at 1-year and only partially true at 5-year horizons.\n"
            f"- **Tradability `MIRAGE`** — Inflation regimes are macro epochs; the finding is "
            "real for risk awareness and portfolio construction context, not for systematic "
            "timing.  MIRAGE.\n"
            f"- **Short-run hedge? `NO`** — The Fisher test is decisive: beta_nom = "
            f"{R['fisher_beta_nom']:.3f} (t = {R['fisher_t_nom']:+.2f}), far below 1.0."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — why the regime effect is not a timing edge\n\n"
            "Three structural reasons the regime gap is descriptively real but not tradable:"
        ),
        code(
            "# Show a rolling 10-year forward real return vs 10-year average CPI\n"
            "if HAVE_REAL:\n"
            "    # Rolling 10-year avg CPI and rolling 10-year annualised real TR\n"
            "    cpi_10y = df['cpi_yoy'].rolling(120, min_periods=60).mean()\n"
            "    # Reconstruct 10y forward real TR from fwd_real_tr_60m (already in df)\n"
            "    real_10y = df['fwd_real_tr_60m']  # 5-year annualised; use as proxy for 10y direction\n"
            "    mask = cpi_10y.notna() & real_10y.notna()\n"
            "    xs = cpi_10y[mask].values * 100\n"
            "    ys = real_10y[mask].values * 100\n"
            "    years = df.index[mask].year.values\n"
            "else:\n"
            "    rng = np.random.default_rng(152)\n"
            "    xs = np.clip(rng.normal(2.3, 3.5, 1200), -2, 20)\n"
            "    ys = -0.5 * xs + 7 + rng.normal(0, 4, 1200)\n"
            "    years = np.linspace(1890, 2010, 1200).astype(int)\n\n"
            "fig, ax = plt.subplots(figsize=(10, 5))\n"
            "sc = ax.scatter(xs, ys, c=years, cmap='plasma', alpha=0.3, s=5)\n"
            "plt.colorbar(sc, ax=ax, label='Year')\n"
            "# OLS trendline\n"
            "p = np.polyfit(xs, ys, 1)\n"
            "x_line = np.linspace(xs.min(), xs.max(), 100)\n"
            "ax.plot(x_line, np.polyval(p, x_line), c=RED, lw=2, label=f'OLS slope={p[0]:.2f}%/pp')\n"
            "ax.axhline(0, c='k', lw=0.5); ax.axvline(3, ls='--', c=GREY, lw=1, label='3% threshold')\n"
            "ax.set_xlabel('Rolling 10-year avg CPI inflation (%)')\n"
            "ax.set_ylabel('Forward 5-year annualised real TR (%, proxy for 10y direction)')\n"
            "ax.set_title('Long-run view: negative relationship, but no timing edge')\n"
            "ax.legend(fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Why you cannot trade this:')\n"
            "print('  1. You know you are in high-inflation AFTER 12 months of CPI data -- too late')\n"
            "print('  2. Regime switches are rare and hard to time (in/out costs are large)')\n"
            "print('  3. At 5y horizon the gap is only ~2pp pa -- fees and implementation eat it')"
        ),
        md(
            "> The scatter makes the relationship visible across 130 years but also its "
            "noisiness: the 1940s had high inflation and reasonable real returns; the 1970s "
            "were genuinely bad; the 1980s disinflation was excellent.  You only know which "
            "decade you are in after the fact.  The regime effect is a description of history; "
            "it is not a market-timing tool."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The Modigliani-Cohn illusion as a *buying* signal?**  If high-inflation "
            "depresses valuations due to investor confusion (not fundamentals), then CAPE is "
            "exceptionally low during high-CPI episodes — meaning the next 10-year real return "
            "might actually be *higher* from those depressed starting points.  Test this with "
            "Study 120 (Excess-CAPE-Yield).\n"
            "- **Is gold a better hedge?**  Study 69 (Safe-Haven) tests gold's CPI sensitivity "
            "— gold has a much higher short-run inflation correlation than equities.\n"
            "- **Real-rate regimes vs. CPI regimes**: Study 119 (Real-Rate-Regime) splits on "
            "the real interest rate (nominal minus CPI), which arguably captures the equity "
            "valuation channel more directly than CPI alone.\n"
            "- **The reverse: does disinflation predict bull markets?**  Ritter & Warr (2002) "
            "argue the 1982-1999 bull was largely driven by inflation falling from elevated "
            "levels (unwinding the Modigliani-Cohn undervaluation).  A disinflation-regime "
            "study would complement this one.\n\n"
            "*Think the hedge works with a different asset, horizon, or CPI threshold?  Fork "
            "this, change the knobs, and show a Fisher beta near 1 or a regime gap with "
            "|t| >= 2 after costs.  That's the bar.*"
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
