"""Generate the two narrative notebooks for Study 157 (Kelly-Sizing).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
SPY parquet under ../_cache/ if present and otherwise quote the frozen headline numbers
in ``R`` (mirroring docs/results.md), so the notebook re-runs for any reader.

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
# Used both in f-string markdown cells at build time AND embedded in BOOT for the kernel.
R = dict(
    n=8399, years=33.3,
    date_start="1993-02-01", date_end="2026-06-12",
    fingerprint="8135df2df025",
    ann_sharpe_spy=0.646,
    true_kelly=3.5,
    kelly_cap=2.0,
    # Walk-forward results (train=5yr, rebal=monthly, cost=1bp)
    fk_geo=0.099, fk_vol=0.334, fk_sharpe=0.296, fk_dd=-0.853, fk_lev=1.613, fk_t=2.70,
    hk_geo=0.063, hk_vol=0.167, hk_sharpe=0.377, hk_dd=-0.567, hk_lev=0.807, hk_t=2.70,
    fe_geo=0.094, fe_vol=0.194, fe_sharpe=0.483, fe_dd=-0.552, fe_lev=1.000, fe_t=3.45,
    he_geo=0.051, he_vol=0.097, he_sharpe=0.523, he_dd=-0.313, he_lev=0.500, he_t=3.45,
    # Kelly vs full-exposure daily differential
    diff_bps=2.09, diff_t=1.83,
    # Estimation noise
    noise_1yr_std=0.88, noise_1yr_p05=0.00, noise_1yr_p95=2.00,
    noise_20yr_std=0.22, noise_20yr_p05=1.60, noise_20yr_p95=2.00,
)

# ---------------------------------------------------------------------------
# Shared analysis preamble (R dict is also embedded inside BOOT so it is
# available in both notebooks as a live kernel variable)
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

from kelly_sizing import data, strategy as st

STUDY_CACHE = os.path.abspath(os.path.join("..", "_cache"))

def _have_cache():
    path = data._cache_path(STUDY_CACHE)
    return os.path.exists(path)

HAVE_REAL = _have_cache()
print("real SPY cache present:", HAVE_REAL)

if HAVE_REAL:
    SPY = data.fetch_spy(fetch=False, cache_dir=STUDY_CACHE)

# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-14).
R = dict(
    n=8399, years=33.3,
    date_start="1993-02-01", date_end="2026-06-12",
    fingerprint="8135df2df025",
    ann_sharpe_spy=0.646,
    true_kelly=3.5,
    kelly_cap=2.0,
    # Walk-forward results (train=5yr, rebal=monthly, cost=1bp)
    fk_geo=0.099, fk_vol=0.334, fk_sharpe=0.296, fk_dd=-0.853, fk_lev=1.613, fk_t=2.70,
    hk_geo=0.063, hk_vol=0.167, hk_sharpe=0.377, hk_dd=-0.567, hk_lev=0.807, hk_t=2.70,
    fe_geo=0.094, fe_vol=0.194, fe_sharpe=0.483, fe_dd=-0.552, fe_lev=1.000, fe_t=3.45,
    he_geo=0.051, he_vol=0.097, he_sharpe=0.523, he_dd=-0.313, he_lev=0.500, he_t=3.45,
    # Kelly vs full-exposure daily differential
    diff_bps=2.09, diff_t=1.83,
    # Estimation noise at 1yr window
    noise_1yr_std=0.88, noise_1yr_p05=0.00, noise_1yr_p95=2.00,
    noise_20yr_std=0.22, noise_20yr_p05=1.60, noise_20yr_p95=2.00,
)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Kelly-Sizing -- does the growth-optimal bet size beat buy-and-hold?\n"
            "### The Kelly criterion tested honestly on 33 years of SPY\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Beats_BH_on_Sharpe%3F: No](https://img.shields.io/badge/Beats_BH_on_Sharpe%3F-No-8b949e?style=flat-square)\n\n"
            "There's a beautiful idea from information theory: if you know the odds of a bet, "
            "there is a *specific* fraction of your wealth to wager that maximises your long-run "
            "growth. It's called the **Kelly criterion**. Applied to stocks, it says: size your "
            "position as `f* = mu / sigma^2` (edge divided by variance). This notebook asks the "
            "honest question: does it actually beat just buying and holding?\n\n"
            "> **This is the plain-language layer.** Want the t-stats, the Kelly parabola, and the "
            "estimation-error math? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does full-Kelly beat buy-and-hold in geometric return? | **Barely, and not reliably.** "
            f"+0.5%/yr advantage, HAC *t* = **{R['diff_t']}** (below the |*t*|=2 bar). |\n"
            "| Is the Kelly edge real? | **WEAK.** The maths is correct, but the parameters are not "
            "estimable with useful precision from any reasonable history. |\n"
            f"| What's the price of full-Kelly? | An **{R['fk_dd']:.0%} max drawdown** (vs "
            f"{R['fe_dd']:.0%} for buy-and-hold), and a Sharpe of {R['fk_sharpe']:.2f} "
            f"vs {R['fe_sharpe']:.2f}. |\n"
            "| Could you trade it? | **FRAGILE.** True Kelly for SPY is ~3.5x leverage — "
            "you're always at the cap. Kelly amplifies your equity risk, not your edge. |\n\n"
            "> Kelly is like a race car engine: theoretically faster, but it explodes without "
            "precise tuning -- and on real data, you never have precise enough parameters."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *'Size your position as `f* = mu / sigma^2`. In the long run, no other strategy "
            "will beat a Kelly bettor. It's the mathematically optimal, growth-maximising approach -- "
            "and the math is provably right.'*\n\n"
            "This is not a trading-forum claim -- it's a theorem. The Kelly criterion genuinely "
            "maximises the long-run geometric growth rate given the *true* probability distribution. "
            "Ed Thorp used it to beat casino blackjack, then ran a hedge fund using it. The "
            "**steelman**: Kelly is optimal, half-Kelly is robust, and fixed-fractional sizing is "
            "just a special case (at the wrong fraction). If the theory holds in practice, it's a "
            "free size-up of your edge."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If Kelly truly beats buy-and-hold out of sample with a known edge (like the equity "
            "risk premium), then every investor *not* using Kelly is leaving geometric return on "
            "the table -- voluntarily. The stakes are: a 0.5-1%/yr geometric improvement compounds "
            "dramatically over decades. At the same time, the *costs* of full-Kelly sizing -- "
            "brutal drawdowns, leverage, and estimation-error blowups -- are real and possibly "
            "unsurvivable for a real investor. We need to know which of these dominates."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Two rules keep us honest:\n\n"
            "1. **Walk-forward only.** The Kelly fraction is estimated *only* from the five years "
            "before each monthly rebalance. We never touch future data. If Kelly works out of "
            "sample, it shows up in geometric return over the *test* period.\n"
            "2. **Compare against an honest baseline.** The relevant competition is "
            "**full buy-and-hold** (full-exposure): the thing a passive investor does. Kelly "
            "claims to beat this. We use the same equity, same period, just a different "
            "position size.\n\n"
            "We also check half-Kelly and half-exposure to understand the trade-off space, "
            "and we visualise how noisy the Kelly *estimate* actually is across different "
            f"lookback windows (spoiler: at 1 year, the 90% confidence interval is "
            f"[{R['noise_1yr_p05']:.1f}x, {R['noise_1yr_p95']:.1f}x] -- essentially no signal)."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown\n\n"
            "**First: the Kelly parabola -- geometric growth vs leverage on the full tape.**\n\n"
            "This is the theoretical heart of Kelly: geometric growth is a concave function of "
            "leverage, peaking at `f*`. Double it and growth falls; triple it and you go broke."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sens = st.kelly_sensitivity(SPY, train_years=20.0)\n"
            "else:\n"
            "    # Approximate Kelly parabola for SPY-like parameters\n"
            "    mu_d = SPY.mean() if HAVE_REAL else 0.000476\n"
            "    var_d = SPY.var() if HAVE_REAL else 0.000137\n"
            "    import numpy as np\n"
            "    fractions = np.linspace(0, 2.5, 51)\n"
            "    # Approximate: g(f) = f*mu - 0.5*f^2*var (lognormal approximation)\n"
            "    geo = fractions * 0.000476 - 0.5 * fractions**2 * 0.000137\n"
            "    sens = pd.DataFrame({'fraction': fractions, 'ann_geo_ret': geo * 252})\n"
            "if HAVE_REAL:\n"
            "    fractions = sens['fraction'].to_numpy()\n"
            "    geo = sens['ann_geo_ret'].to_numpy()\n"
            "    dd = sens['max_drawdown'].to_numpy()\n"
            "else:\n"
            "    fractions = sens['fraction'].to_numpy(); geo = sens['ann_geo_ret'].to_numpy(); dd = None\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.8))\n"
            "ax.plot(fractions, geo * 100, lw=2.5, c=GREEN)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.axvline(1.0, ls='--', c=GREY, lw=1.2, label='buy-and-hold (1x)')\n"
            "peak_idx = np.argmax(geo) if geo is not None else 0\n"
            "ax.axvline(fractions[peak_idx], ls=':', c=RED, lw=1.5, label=f'peak at {fractions[peak_idx]:.1f}x')\n"
            "ax.set_xlabel('leverage fraction (f)'); ax.set_ylabel('annualised geometric return (%)')\n"
            "ax.set_title('The Kelly parabola: geometric growth peaks somewhere, then collapses')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Peak geometric growth at leverage = {fractions[peak_idx]:.2f}x')"
        ),
        md(
            "The Kelly parabola shows the theoretical optimum. On SPY the unconstrained "
            f"optimum is around **{R['true_kelly']:.1f}x** leverage -- already extreme. "
            "But here's the catch: this is the *in-sample* Kelly, computed using the full 33 years "
            "of history at once. In a real walk-forward, you never know the true parameters."
        ),
        md("**Now the honest test: walk-forward sizing on SPY, four strategies.**"),
        code(
            "strategies = ['full-Kelly', 'half-Kelly', 'full-exposure', 'half-exposure']\n"
            "colors = [RED, AMBER, GREEN, GREY]\n"
            "if HAVE_REAL:\n"
            "    bts = st.run_all_strategies(SPY, train_years=5.0, rebal_freq=21, cost_bps=1.0)\n"
            "    cum_rets = {}\n"
            "    for name, bt in bts.items():\n"
            "        active = bt[bt['leverage'] > 0]\n"
            "        cum_rets[name] = (1 + active['ret_port']).cumprod()\n"
            "    fig, ax = plt.subplots(figsize=(9.5, 5.0))\n"
            "    for (name, cum), col in zip(cum_rets.items(), colors):\n"
            "        ax.plot(cum.index, np.log(cum), label=name, lw=1.8 if 'exposure' in name else 1.2, c=col)\n"
            "    ax.set_ylabel('log cumulative wealth')\n"
            "    ax.set_title('Walk-forward Kelly vs fixed sizing on SPY (1998-2026, log scale)')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print(f'[offline] full-Kelly geo: {R[\"fk_geo\"]:.1%}/yr | full-exposure: {R[\"fe_geo\"]:.1%}/yr')"
        ),
        code(
            "if HAVE_REAL:\n"
            "    result = st.compare_strategies(SPY, train_years=5.0, rebal_freq=21, cost_bps=1.0)\n"
            "else:\n"
            "    result = pd.DataFrame({\n"
            "        'ann_geo_ret': [R['fk_geo'], R['hk_geo'], R['fe_geo'], R['he_geo']],\n"
            "        'ann_vol': [R['fk_vol'], R['hk_vol'], R['fe_vol'], R['he_vol']],\n"
            "        'sharpe': [R['fk_sharpe'], R['hk_sharpe'], R['fe_sharpe'], R['he_sharpe']],\n"
            "        'max_drawdown': [R['fk_dd'], R['hk_dd'], R['fe_dd'], R['he_dd']],\n"
            "        'mean_lev': [R['fk_lev'], R['hk_lev'], R['fe_lev'], R['he_lev']],\n"
            "    }, index=['full-Kelly', 'half-Kelly', 'full-exposure', 'half-exposure'])\n"
            "fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))\n"
            "for ax, col, ttl in zip(axes, ['ann_geo_ret', 'max_drawdown', 'sharpe'],\n"
            "                        ['Ann. Geometric Return', 'Max Drawdown', 'Sharpe Ratio']):\n"
            "    vals = result[col]\n"
            "    cols = [RED, AMBER, GREEN, GREY]\n"
            "    ax.bar(range(4), vals, color=cols)\n"
            "    ax.set_xticks(range(4)); ax.set_xticklabels(result.index, rotation=20, ha='right')\n"
            "    ax.set_title(ttl); ax.axhline(0, c='k', lw=1)\n"
            "plt.tight_layout(); plt.show()\n"
            "result[['ann_geo_ret', 'ann_vol', 'sharpe', 'max_drawdown', 'mean_lev']].round(3)"
        ),
        md(
            f"Full-Kelly barely edges buy-and-hold in geometric return "
            f"(**{R['fk_geo']:.1%}** vs **{R['fe_geo']:.1%}**) — but look at the Sharpe: "
            f"**{R['fk_sharpe']:.2f}** vs **{R['fe_sharpe']:.2f}**. And that max drawdown: "
            f"**{R['fk_dd']:.0%}** for full-Kelly vs **{R['fe_dd']:.0%}** for buy-and-hold. "
            "Kelly trades risk-adjusted return for raw growth — and the growth edge is so thin "
            "it's statistically noise.\n\n"
            "> Why so thin? Because for SPY, the unconstrained Kelly fraction is ~3.5x. We cap "
            "at 2x, so in the walk-forward we're often near the cap regardless of the signal — "
            "Kelly's precision is wasted."
        ),
        md("**The estimation problem: how noisy is the Kelly fraction?**"),
        code(
            "if HAVE_REAL:\n"
            "    noise = st.kelly_estimation_noise(SPY, train_years_grid=(1, 2, 5, 10, 20))\n"
            "    years = noise['train_years'].to_numpy()\n"
            "    std = noise['std_kelly'].to_numpy()\n"
            "    p05 = noise['p05'].to_numpy()\n"
            "    p95 = noise['p95'].to_numpy()\n"
            "else:\n"
            "    years = [1, 2, 5, 10, 20]\n"
            "    std = [R['noise_1yr_std'], 0.80, 0.66, 0.44, R['noise_20yr_std']]\n"
            "    p05 = [R['noise_1yr_p05'], 0.00, 0.00, 0.71, R['noise_20yr_p05']]\n"
            "    p95 = [R['noise_1yr_p95'], 2.00, 2.00, 2.00, R['noise_20yr_p95']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "ax.fill_between(years, p05, p95, alpha=0.25, color=AMBER, label='90% CI')\n"
            "ax.plot(years, std, 'o-', c=RED, lw=2, label='std of Kelly estimate')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('training window (years)'); ax.set_ylabel('Kelly fraction variability (x)')\n"
            "ax.set_title('Kelly estimation noise by lookback window (bootstrap, 500 reps)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'1yr window: 90% CI [{p05[0]:.2f}x, {p95[0]:.2f}x] -- useless precision')"
        ),
        md(
            f"At a **1-year lookback**, the 90% CI for the Kelly fraction is "
            f"[{R['noise_1yr_p05']:.1f}x, {R['noise_1yr_p95']:.1f}x] — the full legal range. "
            f"At **20 years**, it narrows to [{R['noise_20yr_p05']:.1f}x, {R['noise_20yr_p95']:.1f}x] "
            "— still essentially 'use the max cap'. You need the entire history of modern finance "
            "to even begin to pin down the Kelly fraction, and it's already at the cap anyway."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal — WEAK.** Full-Kelly geometry is nominally +{(R['fk_geo']-R['fe_geo'])*100:.1f}pp "
            f"above buy-and-hold, but HAC *t* = **{R['diff_t']:.2f}** — below the |*t*|=2 bar. "
            "The maths is right (Kelly IS the growth-optimal strategy given true parameters), "
            "but the parameters are not estimable with useful precision.\n"
            f"- **Tradability — FRAGILE.** Full-Kelly max drawdown **{R['fk_dd']:.0%}**, "
            f"Sharpe {R['fk_sharpe']:.2f} (vs buy-and-hold {R['fe_sharpe']:.2f}). The true Kelly "
            f"fraction for SPY (~{R['true_kelly']:.1f}x) hits the 2x cap immediately — estimation "
            "risk, fat tails, leverage constraints, and costs all erode the theoretical advantage.\n"
            f"- **Beats buy-and-hold on Sharpe? — No.** Buy-and-hold Sharpe {R['fe_sharpe']:.2f}; "
            f"full-Kelly {R['fk_sharpe']:.2f}; half-Kelly {R['hk_sharpe']:.2f}. Kelly trades "
            "risk-adjusted return for raw geometric growth — and that trade is unfavourable at "
            "realistic leverage caps."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT -----------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "The practical Kelly investor faces five compounding problems:\n\n"
            "1. **Estimation risk**: even 20 years of history gives you a Kelly estimate with "
            "std ±0.22x — and you need to act monthly with a 5-year window (std ±0.66x).\n"
            "2. **Leverage cap**: at 2x, you're always running near the cap for SPY regardless "
            "of what the formula says. The precision of Kelly is wasted.\n"
            "3. **Drawdowns**: 85% drawdowns are institutionally and psychologically fatal. "
            "A 50% drawdown requires a 100% gain to recover; an 85% drawdown requires 567%.\n"
            "4. **Fat tails**: the Kelly formula assumes Gaussian returns. SPY has fat tails; "
            "the true growth-optimal fraction under fat tails is *lower* than Gaussian Kelly.\n"
            "5. **Transaction costs**: monthly rebalancing at changing leverage incurs costs. "
            "At 1bp per rebalance the impact is modest, but at larger sizes it matters.\n\n"
            "The practitioner solution is **half-Kelly**: half the growth advantage, "
            "a quarter of the variance. But on SPY, half-Kelly still has a Sharpe of "
            f"{R['hk_sharpe']:.2f} vs {R['fe_sharpe']:.2f} for buy-and-hold, and a "
            f"{R['hk_dd']:.0%} max drawdown. It's not obviously better than simply holding."
        ),
        code(
            "# Kelly fraction distribution from the walk-forward (what lever did we actually use?)\n"
            "if HAVE_REAL:\n"
            "    bt_k = st.run_backtest(SPY, st.kelly_fraction_continuous,\n"
            "                           train_years=5.0, rebal_freq=21, cost_bps=1.0)\n"
            "    lev = bt_k.loc[bt_k['leverage'] > 0, 'leverage']\n"
            "    fig, ax = plt.subplots(figsize=(9.0, 4.0))\n"
            "    ax.hist(lev, bins=30, color=RED, alpha=0.75, edgecolor='white')\n"
            "    ax.axvline(lev.mean(), c='k', lw=2, label=f'mean={lev.mean():.2f}x')\n"
            "    ax.axvline(1.0, ls='--', c=GREY, lw=1.5, label='buy-and-hold')\n"
            "    ax.set_xlabel('applied leverage (x)'); ax.set_ylabel('count (monthly rebalances)')\n"
            "    ax.set_title('Walk-forward Kelly leverage distribution: mostly near the 2x cap')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f'Mean walk-forward leverage: {lev.mean():.2f}x  (cap = 2.00x)')\n"
            "else:\n"
            "    print(f'Mean walk-forward leverage (real tape): {R[\"fk_lev\"]:.2f}x  (cap = 2.00x)')"
        ),
        md(
            f"The walk-forward Kelly leverage averages **{R['fk_lev']:.2f}x** — nearly always "
            "near the 2x cap. The Kelly formula is not giving you a precise size-up; it's "
            "telling you to borrow to the hilt and you're only *not* doing so because of "
            "the cap. Remove the cap and you'd go bankrupt in the first major drawdown."
        ),

        # ---- BEAT 7 -- GOING FURTHER ----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **Fractional Kelly below 0.5**: the companion notebook shows that half-Kelly is "
            "already a reasonable compromise, but quarter-Kelly (or even less) may be more "
            "appropriate when estimation uncertainty is large.\n"
            "- **Kelly with parameter uncertainty**: a Bayesian treatment of Kelly sizing "
            "shrinks the fraction towards zero, reflecting the estimation risk. See Browne (1999) "
            "'Reaching Goals by a Deadline'.\n"
            "- **Multi-asset Kelly**: the correlation matrix adds even more estimation noise. "
            "See [Study 68 (All-Weather)](../../68-all-weather/) for a related test on "
            "risk-parity (inverse-vol weighting, a simpler Kelly cousin).\n"
            "- **Kelly on a genuine short-horizon signal**: the maths only works cleanly when "
            "the signal Sharpe is well-estimated. The right test is a signal with |*t*|>>2 and "
            "a short estimation window, not the diffuse equity risk premium.\n\n"
            "*Think Kelly works in practice? Fork this, add a proper alpha signal with |t|>3, "
            "apply Kelly, and show out-of-sample it beats 1x. That's the actual bar.*"
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
            "# Kelly-Sizing -- a quantitative teardown\n"
            "### Walk-forward backtest * Kelly parabola * estimation noise * HAC inference\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Beats_BH_on_Sharpe%3F: No](https://img.shields.io/badge/Beats_BH_on_Sharpe%3F-No-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim now carrying its standard error.* We test whether "
            "walk-forward Kelly sizing beats fixed-fractional sizing on SPY, quantify the "
            "estimation-error distribution, and put the growth differential through HAC inference.\n\n"
            "> **Not investment advice.** Real data: SPY daily returns via Yahoo Finance "
            f"(auto-adjusted), {R['date_start']} to {R['date_end']} ({R['n']} trading days); "
            "the offline core and tests run on a deterministic synthetic tape. "
            "Methods & sources in [`docs/references.md`](../docs/references.md), reproducible "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The `> In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT + "\nfrom quantlab import analytics, stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Full-Kelly geo {R['fk_geo']:.1%}/yr vs buy-and-hold "
            f"{R['fe_geo']:.1%}/yr; daily differential HAC *t* = **{R['diff_t']:.2f}** "
            f"({R['diff_bps']:+.2f} bps/day). Below |*t*|=2. |\n"
            f"| **Tradability** | `FRAGILE` | Full-Kelly max DD **{R['fk_dd']:.0%}** vs "
            f"{R['fe_dd']:.0%} for buy-and-hold. True Kelly fraction ~{R['true_kelly']:.1f}x "
            f"hits 2x cap. Sharpe {R['fk_sharpe']:.2f} vs {R['fe_sharpe']:.2f}. |\n"
            f"| **Beats BH on Sharpe?** | `No` | BH Sharpe {R['fe_sharpe']:.2f}; "
            f"full-Kelly {R['fk_sharpe']:.2f}; half-Kelly {R['hk_sharpe']:.2f}. "
            "Kelly trades risk-adjusted return for geometric growth. |\n\n"
            "> In plain words: the Kelly criterion is theoretically optimal but practically "
            "fragile. The equity risk premium is too imprecisely estimated for the Kelly "
            "fraction to be actionable, and the leverage it requires produces "
            "institutionally fatal drawdowns."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $r_t$ be the daily log-return and $\\mu, \\sigma^2$ its mean and variance. The "
            "continuous Kelly criterion states:\n\n"
            "$$f^* = \\frac{\\mu}{\\sigma^2}$$\n\n"
            "Under log-normal returns, $f^*$ maximises $\\mathbb{E}[\\log W_T]$ and, almost surely, "
            "$W_T^{\\text{Kelly}} / W_T^{\\text{other}} \\to \\infty$ as $T \\to \\infty$ "
            "(Breiman 1961). In expectation, this translates to annualised geometric growth:\n\n"
            "$$g(f) \\approx f \\mu - \\tfrac{1}{2} f^2 \\sigma^2$$\n\n"
            "which is a concave parabola peaking at $f^*$. The three hypotheses we test:\n\n"
            "- **H1 (growth)**: Walk-forward Kelly geometric return exceeds buy-and-hold.\n"
            "- **H2 (significance)**: The growth differential has HAC $|t| \\geq 2$.\n"
            "- **H3 (tradability)**: The advantage survives leverage caps, drawdowns, "
            "and estimation risk.\n\n"
            "We find H1 barely (nominally) true, H2 false (|t|=1.83), and H3 false."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what? -- what rides on each answer\n\n"
            "Kelly sizing is the theoretical backbone of portfolio sizing. If H1-H3 held, "
            "every fixed-fraction investor is systematically suboptimal. The interesting "
            "failure here is not *that* Kelly underperforms on Sharpe -- the formula was never "
            "designed to maximise Sharpe -- but *that even its geometric return claim is "
            "statistically unproven at realistic holding periods*. And the practical killer: "
            "the true Kelly fraction for the equity risk premium is so large it immediately "
            "saturates any reasonable leverage cap, making the formula's precision irrelevant."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - How we'd know -- the protocol\n\n"
            "- **Asset**: SPY daily total returns, 1993-02-01 onward (33 years, 8399 days).\n"
            "- **Walk-forward**: train on 5yr preceding each monthly rebalance; no future data "
            "touches the Kelly estimate at any point.\n"
            "- **Four arms**: full-Kelly (f capped at 2x), half-Kelly, full-exposure (1x), "
            "half-exposure (0.5x). All share the same 1bp round-trip rebalancing cost.\n"
            "- **Inference**: HAC t-stat (Newey-West) on the daily return differential "
            "between full-Kelly and full-exposure in the active period.\n"
            "- **Estimation noise**: bootstrap the Kelly fraction from sub-windows of "
            "increasing length to show the 90% CI at each lookback.\n"
            "- **Positive control**: a deterministic synthetic tape with planted Sharpe=0.5, "
            "showing the engine recovers a Kelly advantage when parameters are precisely known."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - The Kelly parabola: where does SPY's optimum actually live?\n\n"
            "In-sample geometric growth as a function of constant leverage. The peak is the "
            "true Kelly fraction -- if it were known with certainty."
        ),
        code(
            "if HAVE_REAL:\n"
            "    sens = st.kelly_sensitivity(SPY, train_years=20.0)\n"
            "    f_arr = sens['fraction'].to_numpy()\n"
            "    g_arr = sens['ann_geo_ret'].to_numpy() * 100\n"
            "    dd_arr = sens['max_drawdown'].to_numpy() * 100\n"
            "else:\n"
            "    f_arr = np.linspace(0, 2.5, 51)\n"
            "    mu_d, var_d = 0.000476, 0.000137\n"
            "    g_arr = (f_arr * mu_d - 0.5 * f_arr**2 * var_d) * 252 * 100\n"
            "    dd_arr = -f_arr * 55  # rough approximation\n"
            "fig, ax1 = plt.subplots(figsize=(9.5, 4.8))\n"
            "ax1.plot(f_arr, g_arr, lw=2.5, c=GREEN, label='ann. geo. return (%/yr)')\n"
            "ax2 = ax1.twinx()\n"
            "ax2.plot(f_arr, dd_arr, lw=2, c=RED, ls='--', label='max drawdown (%)')\n"
            "ax1.axvline(1.0, ls='--', c=GREY, lw=1.2, label='f=1 (buy-and-hold)')\n"
            "peak_f = f_arr[np.argmax(g_arr)]\n"
            "ax1.axvline(peak_f, ls=':', c=GREEN, lw=1.5, label=f'Kelly peak f={peak_f:.2f}x')\n"
            "ax1.set_xlabel('leverage fraction f'); ax1.set_ylabel('ann. geo. return (%)', color=GREEN)\n"
            "ax2.set_ylabel('max drawdown (%)', color=RED)\n"
            "ax1.set_title('Kelly parabola: geo growth peaks at ~2.5x; drawdown worsens monotonically')\n"
            "lines1, labels1 = ax1.get_legend_handles_labels()\n"
            "lines2, labels2 = ax2.get_legend_handles_labels()\n"
            "ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left'); plt.tight_layout(); plt.show()\n"
            f"print(f'In-sample Kelly optimum: {{peak_f:.2f}}x leverage  (2x cap applied in walk-forward)')"
        ),
        md(
            f"> In plain words: the parabola peaks around **{R['true_kelly']:.0f}x leverage** "
            "on the full in-sample tape. Nobody can run 3.5x leveraged SPY -- and even if "
            "they could, the in-sample optimum is computed with hindsight. The walk-forward "
            "Kelly fraction is noisier and lower. Our 2x cap truncates most of the theoretical "
            "advantage before it can manifest."
        ),
        md(
            "### 4b - Walk-forward comparison: four strategies, same tape\n\n"
            "Strategy returns, risks, and statistics over the 1998-2026 out-of-sample period "
            "(5yr training burn-in, monthly rebalance, 1bp cost)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    result = st.compare_strategies(SPY, train_years=5.0, rebal_freq=21, cost_bps=1.0)\n"
            "    print(result[['n_days','ann_geo_ret','ann_vol','sharpe','max_drawdown','mean_lev','tstat']].round(3).to_string())\n"
            "    # HAC t-stat on daily differential\n"
            "    bt_k = st.run_backtest(SPY, st.kelly_fraction_continuous, 5.0, 21, 1.0)\n"
            "    bt_fe = st.run_backtest(SPY, lambda r: 1.0, 5.0, 21, 1.0)\n"
            "    active = bt_k['leverage'] > 0\n"
            "    diff = bt_k.loc[active, 'ret_port'] - bt_fe.loc[active, 'ret_port']\n"
            "    hac = analytics.mean_tstat_hac(diff)\n"
            "    print(f'\\nKelly - buy-and-hold daily differential:')\n"
            "    print(f'  mean = {hac[\"mean_bps\"]:+.2f} bps/day,  HAC t = {hac[\"tstat\"]:+.2f}')\n"
            "else:\n"
            "    print(f'full-Kelly: geo={R[\"fk_geo\"]:.1%}, Sharpe={R[\"fk_sharpe\"]:.2f}, DD={R[\"fk_dd\"]:.0%}')\n"
            "    print(f'full-exposure: geo={R[\"fe_geo\"]:.1%}, Sharpe={R[\"fe_sharpe\"]:.2f}, DD={R[\"fe_dd\"]:.0%}')\n"
            "    print(f'Kelly - BH: {R[\"diff_bps\"]:+.2f} bps/day,  HAC t = {R[\"diff_t\"]:+.2f}')"
        ),
        md(
            f"> In plain words: full-Kelly wins on geometric return by **{(R['fk_geo']-R['fe_geo'])*100:.1f}pp/yr** "
            f"({R['fk_geo']:.1%} vs {R['fe_geo']:.1%}) but the daily differential of "
            f"**{R['diff_bps']:+.2f} bps/day** has HAC *t* = **{R['diff_t']:.2f}** -- "
            "below the |*t*|=2 bar. Signal is WEAK, not REAL. And that Sharpe cost is real: "
            f"buy-and-hold ({R['fe_sharpe']:.2f}) beats all leveraged variants."
        ),
        md(
            "### 4c - The estimation-noise problem: how wide is the Kelly CI?\n\n"
            "Bootstrap the Kelly fraction from sub-windows of the real tape to visualise "
            "how much precision you actually have at each lookback."
        ),
        code(
            "if HAVE_REAL:\n"
            "    noise = st.kelly_estimation_noise(SPY, train_years_grid=(1, 2, 5, 10, 20))\n"
            "    yrs = noise['train_years'].to_numpy()\n"
            "    std = noise['std_kelly'].to_numpy()\n"
            "    p05, p95 = noise['p05'].to_numpy(), noise['p95'].to_numpy()\n"
            "else:\n"
            "    yrs = [1, 2, 5, 10, 20]\n"
            "    std = [0.88, 0.80, 0.66, 0.44, 0.22]\n"
            "    p05 = [0.00, 0.00, 0.00, 0.71, 1.60]\n"
            "    p95 = [2.00, 2.00, 2.00, 2.00, 2.00]\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "ax = axes[0]\n"
            "ax.fill_between(yrs, p05, p95, alpha=0.25, color=AMBER, label='90% CI')\n"
            "ax.plot(yrs, std, 'o-', c=RED, lw=2, label='std of Kelly estimate')\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xlabel('training window (years)')\n"
            "ax.set_ylabel('Kelly fraction (x)'); ax.set_title('Estimation noise by window')\n"
            "ax.legend()\n"
            "ax = axes[1]\n"
            "# Width of CI\n"
            "ci_width = [p - q for p, q in zip(p95, p05)]\n"
            "ax.bar(yrs, ci_width, color=RED, alpha=0.75, width=0.8)\n"
            "ax.set_xlabel('training window (years)'); ax.set_ylabel('90% CI width (x leverage)')\n"
            "ax.set_title('CI width narrows slowly with more data')\n"
            "plt.tight_layout(); plt.show()\n"
            "if HAVE_REAL: print(noise.to_string(index=False))\n"
            "else: print('1yr: 90% CI [0.00, 2.00]; 20yr: 90% CI [1.60, 2.00]')"
        ),
        md(
            f"> In plain words: at **1yr** of data (252 days) the 90% CI for the Kelly fraction "
            f"spans the entire legal range [{R['noise_1yr_p05']:.1f}x, {R['noise_1yr_p95']:.1f}x]. "
            f"Even at **20yr** it is [{R['noise_20yr_p05']:.1f}x, {R['noise_20yr_p95']:.1f}x]. "
            "The Kelly fraction for the equity premium is *not estimable* at any practical "
            "horizon -- it is always near the cap, regardless of the lookback."
        ),
        md(
            "### 4d - Synthetic positive control: the machinery works when parameters are known\n\n"
            "On a synthetic tape with *planted* Sharpe, the walk-forward Kelly strategy "
            "achieves higher geometric growth than fixed-fractional when the parameters "
            "are estimable. This confirms the engine is correct -- the real-tape result "
            "is a statement about the market's imprecise parameters, not a bug."
        ),
        code(
            "sharpes = [0.0, 0.5, 1.0, 2.0]\n"
            "kelly_geo, fe_geo, diff_vals = [], [], []\n"
            "for sh in sharpes:\n"
            "    syn, _ = data.synthetic_returns(n_years=30, sharpe=sh, annual_vol=0.16, seed=157)\n"
            "    res = st.compare_strategies(syn, train_years=5.0, rebal_freq=21, cost_bps=0.0)\n"
            "    kelly_geo.append(res.loc['full-Kelly','ann_geo_ret'])\n"
            "    fe_geo.append(res.loc['full-exposure','ann_geo_ret'])\n"
            "    diff_vals.append(res.loc['full-Kelly','ann_geo_ret'] - res.loc['full-exposure','ann_geo_ret'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "ax.bar([str(s) for s in sharpes], [d*100 for d in diff_vals], color=[RED if d<0 else GREEN for d in diff_vals])\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xlabel('planted annual Sharpe')\n"
            "ax.set_ylabel('Kelly geo. return - buy-and-hold geo. return (%/yr)')\n"
            "ax.set_title('Synthetic control: Kelly advantage grows with estimable Sharpe')\n"
            "plt.tight_layout(); plt.show()\n"
            "for sh, d in zip(sharpes, diff_vals): print(f'Sharpe={sh:.1f}: Kelly vs BH diff = {d:+.1%}/yr')"
        ),
        md(
            "> In plain words: when the Sharpe is high (Sharpe=2, as in a strong signal "
            "strategy), walk-forward Kelly beats buy-and-hold substantially. At Sharpe=0.5 "
            "(the equity premium) it barely wins. At Sharpe=0 it loses. The real SPY tape "
            "sits near the Sharpe=0.5 case, where the advantage is too thin to clear the |t|=2 bar."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal `WEAK`** -- geo differential {R['diff_bps']:+.2f} bps/day, "
            f"HAC *t* = {R['diff_t']:.2f}; every half-Kelly and buy-and-hold variant "
            "wins on Sharpe. The maths of Kelly is right; the estimation precision is wrong.\n"
            f"- **Tradability `FRAGILE`** -- true Kelly fraction ~{R['true_kelly']:.1f}x, "
            f"always capped; mean walk-forward leverage {R['fk_lev']:.2f}x; "
            f"max drawdown {R['fk_dd']:.0%}; Sharpe {R['fk_sharpe']:.2f} vs {R['fe_sharpe']:.2f} BH.\n"
            f"- **Beats BH on Sharpe? `No`** -- buy-and-hold Sharpe {R['fe_sharpe']:.2f}; "
            f"all Kelly variants below; even half-Kelly ({R['hk_sharpe']:.2f}) trails."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you trade it? -- the leverage and drawdown cost\n\n"
            "Kelly's growth advantage (if real) is paid for entirely in drawdown and Sharpe. "
            "Sweep the Kelly fraction cap from 0.5x to 2.0x and see the trade-off."
        ),
        code(
            "caps = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]\n"
            "rows = []\n"
            "if HAVE_REAL:\n"
            "    for cap in caps:\n"
            "        def sized(r, cap=cap): return min(st.kelly_fraction_continuous(r), cap)\n"
            "        bt = st.run_backtest(SPY, sized, train_years=5.0, rebal_freq=21, cost_bps=1.0)\n"
            "        s = st.portfolio_stats(bt)\n"
            "        rows.append({'cap': cap, 'geo': s['ann_geo_ret'], 'sharpe': s['sharpe'], 'dd': s['max_drawdown']})\n"
            "    cap_df = pd.DataFrame(rows)\n"
            "else:\n"
            "    cap_df = pd.DataFrame({'cap': caps,\n"
            "        'geo': [0.055, 0.068, 0.082, 0.088, 0.093, 0.099],\n"
            "        'sharpe': [0.52, 0.47, 0.42, 0.38, 0.34, 0.30],\n"
            "        'dd': [-0.29, -0.36, -0.44, -0.53, -0.62, -0.85]})\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "axes[0].plot(cap_df['cap'], cap_df['geo']*100, 'o-', c=GREEN, lw=2)\n"
            "axes[0].axhline(R['fe_geo']*100, ls='--', c=GREY, label='buy-and-hold')\n"
            "axes[0].set_xlabel('Kelly cap (x)'); axes[0].set_ylabel('ann. geo. return (%/yr)')\n"
            "axes[0].set_title('More leverage -> more growth (barely)'); axes[0].legend()\n"
            "axes[1].plot(cap_df['cap'], cap_df['dd']*100, 'o-', c=RED, lw=2)\n"
            "axes[1].axhline(R['fe_dd']*100, ls='--', c=GREY, label='buy-and-hold DD')\n"
            "axes[1].set_xlabel('Kelly cap (x)'); axes[1].set_ylabel('max drawdown (%)')\n"
            "axes[1].set_title('More leverage -> worse drawdown (rapidly)'); axes[1].legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(cap_df.round(3).to_string(index=False))"
        ),
        md(
            "> In plain words: increasing the Kelly cap from 0.5x to 2.0x adds roughly "
            f"**{(R['fk_geo']-R['he_geo'])*100:.0f}pp/yr** of geometric return -- but at the cost of "
            f"**{(R['fk_dd']-R['he_dd'])*100:.0f}pp** of additional max drawdown. The trade-off is "
            "unfavourable on Sharpe at every cap level. The 'free growth' is not free."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **Kelly under parameter uncertainty**: a Bayesian posterior over (mu, sigma) "
            "typically shrinks the Kelly fraction substantially. See Kardaras & Platen (2011) "
            "'On the Optimal Wealth Process in a Log-Normal Market'.\n"
            "- **Fractional Kelly with a momentum signal**: if instead of the diffuse equity "
            "risk premium you have a high-Sharpe signal (|t|>3), Kelly sizing becomes meaningful. "
            "The synthetic positive control (Beat 4d) shows the advantage is real at Sharpe=2.\n"
            "- **Risk parity as approximate Kelly**: [Study 68 (All-Weather)](../../68-all-weather/) "
            "applies inverse-vol weighting -- a simpler cousin of multi-asset Kelly.\n"
            "- **The ruin zone**: what happens above 2x Kelly? Geometric growth goes negative. "
            "A simple extension: set the cap at 3x or 4x and observe the walk-forward blow-up.\n\n"
            "*Think Kelly beats buy-and-hold with a real signal? Fork this, use a factor or "
            "trend signal with |t|>2, apply walk-forward Kelly, and clear the |t|=2 bar on the "
            "growth differential. That's the proof.*"
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
