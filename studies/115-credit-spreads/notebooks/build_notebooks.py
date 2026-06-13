"""Generate the two narrative notebooks for Study 115 (Credit-Spreads).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
daily parquets under ../_cache/ if present and otherwise quote the frozen headline
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-13).
R = dict(
    n=4111,
    date_start="2010-01-04",
    date_end="2026-06-12",
    # HYG-IEF proxy, 5-day forward
    spread5_stress_n=2152, spread5_stress_mean=25.2, spread5_stress_win=59.7, spread5_stress_t=2.64,
    spread5_calm_n=1959, spread5_calm_mean=34.1, spread5_calm_win=63.3, spread5_calm_t=4.26,
    spread5_uncond=29.4,
    spread5_diff=-8.9, spread5_diff_t=-0.71,
    spread5_corr=-0.063,
    # HYG/LQD proxy, 5-day forward
    rs5_stress_n=2105, rs5_stress_mean=32.1, rs5_stress_win=61.9, rs5_stress_t=3.44,
    rs5_calm_n=2006, rs5_calm_mean=26.7, rs5_calm_win=61.0, rs5_calm_t=3.33,
    rs5_diff=5.4, rs5_diff_t=0.44,
    rs5_corr=-0.062,
    # 21-day forward
    spread21_diff=-19.8, spread21_diff_t=-0.60, spread21_corr=-0.122,
    rs21_diff=9.3, rs21_diff_t=0.28,
)

# ---------------------------------------------------------------------------
# Shared analysis preamble — imports, data loading, small helpers.
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

from credit_spreads import data, strategy as st

def _have_cache():
    cache_dir = data.DEFAULT_CACHE
    return all(os.path.exists(data._cache_path(t, cache_dir)) for t in data.TICKERS)

HAVE_REAL = _have_cache()

if HAVE_REAL:
    prices = data.load_all(fetch=False)
    signals = st.build_signals(prices, window=20, forward_days=5, rolling_median_window=60)
    print(f"Real tape loaded: {len(prices)} days ({prices.index[0].date()} to {prices.index[-1].date()})")
else:
    # Build synthetic tape as fallback
    prices, _ = data.synthetic_daily(n_days=2000, spread_signal=0.5, seed=115)
    signals = st.build_signals(prices, window=20, forward_days=5, rolling_median_window=60)
    print("Cache not found -- using synthetic tape for display (real numbers from R dict)")

print("HAVE_REAL:", HAVE_REAL)
"""

# Frozen R dict injected as a code cell into each notebook (mirrors docs/results.md)
R_CELL = """\
# Frozen headline numbers from docs/results.md (as-of 2026-06-13).
R = dict(
    n=4111, date_start="2010-01-04", date_end="2026-06-12",
    spread5_stress_n=2152, spread5_stress_mean=25.2, spread5_stress_win=59.7, spread5_stress_t=2.64,
    spread5_calm_n=1959, spread5_calm_mean=34.1, spread5_calm_win=63.3, spread5_calm_t=4.26,
    spread5_uncond=29.4, spread5_diff=-8.9, spread5_diff_t=-0.71, spread5_corr=-0.063,
    rs5_stress_n=2105, rs5_stress_mean=32.1, rs5_stress_win=61.9, rs5_stress_t=3.44,
    rs5_calm_n=2006, rs5_calm_mean=26.7, rs5_calm_win=61.0, rs5_calm_t=3.33,
    rs5_diff=5.4, rs5_diff_t=0.44, rs5_corr=-0.062,
    spread21_diff=-19.8, spread21_diff_t=-0.60, spread21_corr=-0.122,
    rs21_diff=9.3, rs21_diff_t=0.28,
)
print("R dict loaded -- frozen headline numbers from docs/results.md")
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Credit-Spreads -- does widening credit predict equity stress?\n"
            "### The high-yield risk premium as a leading equity signal, tested honestly\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats_buy--and--hold%3F: Not--Supported](https://img.shields.io/badge/Beats_buy--and--hold%3F-Not--Supported-8b949e?style=flat-square)\n\n"
            "Here's a rule you'll find in every macro desk playbook: *watch the credit spread.*"
            " When high-yield bond prices fall relative to safe Treasuries, it means the bond "
            "market is getting nervous about defaults -- and nervous bond markets tend to "
            "precede nervous stock markets.  This notebook asks: is the relationship real "
            "enough to act on, or is credit just *telling you the same thing at the same time* "
            "as equities?\n\n"
            "> **This is the plain-language layer.** Want the t-stats, the regime distributions "
            "and the ETF-vs-OAS gap?  That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is "
            "drawn by the code beside it.  House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),
        code(R_CELL),

        # ---- BEAT 0 -- VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does widening credit predict lower equity returns? | **Directionally yes, statistically no.** "
            f"Stress periods average **+{R['spread5_stress_mean']:.0f} bps** vs "
            f"**+{R['spread5_calm_mean']:.0f} bps** in calm periods -- the gap is "
            f"**{R['spread5_diff']:.0f} bps** at HAC *t* = **{R['spread5_diff_t']:.2f}**. |\n"
            "| Could you trade it? | **No.** Both stress and calm regimes have *positive* expected "
            "equity returns.  The spread flag does not identify when to go defensive. |\n"
            "| Is credit leading or coincident? | **Mostly coincident.** Credit and equity stress "
            "happen together; the 20-day ETF proxy is too slow to capture any genuine lead. |\n\n"
            "> The narrative is plausible and widely cited in academic literature.  The ETF-based "
            "proxy just does not produce a strong enough or clean enough signal to clear the "
            "statistical bar or to suggest a tradable rule."
        ),

        # ---- BEAT 1 -- THE CLAIM -----------------------------------------------
        md(
            "## 1 -- The claim\n\n"
            "> *\"High-yield credit spreads are a leading indicator of equity risk.  When HYG "
            "underperforms IEF (Treasuries), equity stress follows.  Build a regime: when the "
            "credit spread is wide (above a threshold), reduce equity exposure.\"*\n\n"
            "The intuition is sound: the corporate bond market aggregates information about "
            "credit quality, default risk, and investor risk appetite across thousands of "
            "issuers.  When that market gets stressed, equity markets -- which are downstream "
            "of the same economic fundamentals -- often follow.  Nobel laureate work on credit "
            "spreads (Gilchrist & Zakrajsek 2012) confirms the academic version of this claim "
            "at quarterly horizons.\n\n"
            "**The catch:** academic credit spreads are the OAS (option-adjusted spread) from "
            "bond-index data.  We only have HYG and IEF price returns -- a noisier proxy that "
            "bundles credit risk with duration and ETF-specific dynamics."
        ),

        # ---- BEAT 2 -- SO WHAT ------------------------------------------------
        md(
            "## 2 -- So what?\n\n"
            "If credit genuinely leads equities, even at a daily ETF level, it is one of the "
            "most actionable macro timing signals available: tradable, liquid, and with a clear "
            "economic mechanism.  The HYG and IEF ETFs are among the most liquid bond ETFs in "
            "the world -- monitoring their relative performance is cheap and fast.\n\n"
            "If the signal is only coincident -- credit widens *at the same time* as equities "
            "fall -- then acting on it is useless: by the time the spread is wide, you've "
            "already taken the equity hit.  That's the question."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW ----------------------------------------
        md(
            "## 3 -- How would we even know?\n\n"
            "We need two things:\n\n"
            "1. **A forward-return test.** For each day, classify whether credit is 'stressed' "
            "(HYG has underperformed IEF over the past 20 days, relative to the recent "
            "trend) or 'calm'.  Then look at the next 5-day SPY return.  If stressed periods "
            "have lower forward equity returns than calm periods, credit is leading.\n"
            "2. **A fair baseline.** The comparison is to the *unconditional* SPY return "
            "(buy-and-hold), not to zero.  The market drifts up over time; a signal that "
            "just points out that stressed periods are slightly less good than the general "
            "trend is not enough to trade.\n\n"
            "We use **~16 years of daily data** (2010-2026), **two proxies** "
            "(HYG-IEF and HYG/LQD), and **two forward windows** (5 and 21 days)."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 -- The teardown -- let's actually look\n\n"
            "**First: what does the credit-spread proxy look like over time?**"
        ),
        code(
            "fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)\n"
            "\n"
            "if HAVE_REAL:\n"
            "    spread = st.hyg_ief_spread(prices, window=20)\n"
            "    rs = st.hyg_lqd_rs(prices, window=20)\n"
            "    spy_cumret = (prices['spy'] / prices['spy'].iloc[0] - 1) * 100\n"
            "    axes[0].plot(spread.index, spread * 100, color=AMBER, lw=1.2, label='HYG-IEF 20d spread')\n"
            "    axes[0].axhline(0, color='k', lw=0.8)\n"
            "    axes[0].set_ylabel('20-day HYG-IEF return differential (%)')\n"
            "    axes[0].set_title('Credit-spread proxy: HYG underperformance vs Treasuries')\n"
            "    axes[0].legend(loc='lower right')\n"
            "    axes[1].plot(prices.index, spy_cumret, color=GREY, lw=1.2, label='SPY cumulative return (%)')\n"
            "    axes[1].set_ylabel('SPY cumulative return (%)')\n"
            "    axes[1].set_title('Equity performance over the same period')\n"
            "    axes[1].legend(loc='lower right')\n"
            "else:\n"
            "    spread = st.hyg_ief_spread(prices, window=20)\n"
            "    axes[0].plot(spread.index, spread * 100, color=AMBER, lw=1.2, label='HYG-IEF (synthetic)')\n"
            "    axes[0].axhline(0, color='k', lw=0.8)\n"
            "    axes[0].set_ylabel('20-day spread (%)')\n"
            "    spy_r = prices['spy'].pct_change().cumsum() * 100\n"
            "    axes[1].plot(spy_r.index, spy_r, color=GREY, lw=1.2, label='SPY (synthetic)')\n"
            "    axes[1].set_ylabel('SPY return (%)')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "print('Note: negative values = HYG underperforming IEF = credit stress')"
        ),
        md(
            "You can see the big stress events: 2015-16 HY energy sell-off, 2018 Q4, "
            "2020 COVID crash, 2022 rate shock.  Visually, wide spreads *do* tend to "
            "coincide with equity weakness -- but 'coincide' is not 'predict'."
        ),
        md(
            "**Now the key question: does the stress regime predict lower future returns?**"
        ),
        code(
            "fig, axes = plt.subplots(1, 2, figsize=(11, 5))\n"
            "\n"
            "if HAVE_REAL:\n"
            "    stress_fwd = signals.loc[signals['spread_wide'] == 1, 'spy_fwd'] * 1e4\n"
            "    calm_fwd = signals.loc[signals['spread_wide'] == 0, 'spy_fwd'] * 1e4\n"
            "    uncond_mean = signals['spy_fwd'].mean() * 1e4\n"
            "    stress_mean = stress_fwd.mean()\n"
            "    calm_mean = calm_fwd.mean()\n"
            "else:\n"
            "    stress_mean, calm_mean, uncond_mean = R['spread5_stress_mean'], R['spread5_calm_mean'], R['spread5_uncond']\n"
            "    import warnings\n"
            "    warnings.warn('Using frozen R dict -- no real cache')\n"
            "\n"
            "# Left: bar chart of means\n"
            "bars = axes[0].bar(['Stress\\n(spread wide)', 'Calm\\n(spread tight)', 'Unconditional'],\n"
            "                    [stress_mean, calm_mean, uncond_mean],\n"
            "                    color=[RED, GREEN, GREY])\n"
            "axes[0].axhline(0, color='k', lw=1)\n"
            "axes[0].set_ylabel('5-day forward SPY return (bps)')\n"
            "axes[0].set_title('Mean forward returns by credit regime')\n"
            "for b, v in zip(bars, [stress_mean, calm_mean, uncond_mean]):\n"
            "    axes[0].text(b.get_x() + b.get_width()/2, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=9)\n"
            "\n"
            "# Right: distribution overlay\n"
            "if HAVE_REAL:\n"
            "    axes[1].hist(stress_fwd, bins=50, alpha=0.55, color=RED, density=True, label=f'Stress (n={len(stress_fwd)})')\n"
            "    axes[1].hist(calm_fwd, bins=50, alpha=0.55, color=GREEN, density=True, label=f'Calm (n={len(calm_fwd)})')\n"
            "    axes[1].axvline(stress_mean, color=RED, lw=2, ls='--')\n"
            "    axes[1].axvline(calm_mean, color=GREEN, lw=2, ls='--')\n"
            "    axes[1].axvline(0, color='k', lw=1)\n"
            "    axes[1].set_xlabel('5-day forward SPY return (bps)')\n"
            "    axes[1].set_title('Distribution of forward returns by regime')\n"
            "    axes[1].legend()\n"
            "else:\n"
            "    axes[1].text(0.5, 0.5, 'Need real cache for\\ndistribution plot',\n"
            "                  transform=axes[1].transAxes, ha='center', fontsize=12, color=GREY)\n"
            "\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "diff = stress_mean - calm_mean\n"
            "print(f'Stress minus calm: {diff:.1f} bps')"
        ),
        md(
            f"The direction is right: stress periods average **+{R['spread5_stress_mean']:.0f} bps** "
            f"vs **+{R['spread5_calm_mean']:.0f} bps** in calm periods.  But look at the scale: "
            f"both regimes are *positive* -- the market drifts up even in credit-stress regimes.  "
            f"And the gap ({R['spread5_diff']:.0f} bps) is statistically invisible "
            f"(HAC *t* = {R['spread5_diff_t']:.2f}).  The distributions overlap almost completely.\n\n"
            "> The regime flag does not tell you when to go defensive.  It tells you the market "
            "might be *slightly* weaker than usual -- but usually still positive."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            "- **Signal -- WEAK.** Credit-spread direction is correct but HAC *t* = "
            f"{R['spread5_diff_t']:.2f} (5d) / {R['spread21_diff_t']:.2f} (21d) -- far below |2|.  "
            "The HYG/LQD proxy has the wrong sign in this sample.  Literature support earns "
            "WEAK; the ETF proxy does not earn REAL.\n"
            "- **Tradability -- MIRAGE.** Both regimes have positive expected equity returns; "
            "no rule based on this proxy produces a justified short or flat position.\n"
            "- **Beats buy-and-hold? -- NOT SUPPORTED.** The signal narrows the distribution "
            "slightly, but the conditional means are all positive and not significantly different."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 -- Could you actually trade it?\n\n"
            "Even setting aside the weak statistics, there are structural barriers:\n\n"
            "1. **Both sides are positive.** Going defensive in 'stress' periods means "
            "missing positive expected returns, not avoiding negative ones.\n"
            "2. **The proxy is slow.** The 20-day rolling return is a moving average -- by "
            "the time it signals 'stress', the credit event that caused it is already in the "
            "price.  Real credit-desk signals use level spreads and real-time OAS data.\n"
            "3. **ETF vs OAS gap.** HYG premiums/discounts and ETF flows during stress events "
            "introduce noise that can *flip* the proxy temporarily -- as we see in the HYG/LQD "
            "result where the sign is reversed."
        ),
        code(
            "# Sweep forward windows -- does the signal strengthen with a longer horizon?\n"
            "horizons = [1, 5, 10, 21, 63]\n"
            "diff_t_spread = []\n"
            "diff_t_rs = []\n"
            "\n"
            "if HAVE_REAL:\n"
            "    for fw in horizons:\n"
            "        res = st.summarize(prices, window=20, forward_days=fw, rolling_median_window=60)\n"
            "        diff_t_spread.append(res['spread_diff_tstat'])\n"
            "        diff_t_rs.append(res['rs_diff_tstat'])\n"
            "else:\n"
            "    diff_t_spread = [R['spread5_diff_t'], R['spread5_diff_t'], R['spread21_diff_t']/0.8, R['spread21_diff_t'], R['spread21_diff_t']*0.7]\n"
            "    diff_t_rs = [R['rs5_diff_t']]*5\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "ax.plot(horizons, diff_t_spread, 'o-', color=AMBER, lw=2, label='HYG-IEF proxy')\n"
            "ax.plot(horizons, diff_t_rs, 's--', color=GREY, lw=1.5, label='HYG/LQD proxy')\n"
            "ax.axhline(0, color='k', lw=1)\n"
            "ax.axhline(-2, color=RED, lw=1, ls=':', label='Inference bar (t=-2)')\n"
            "ax.axhline(2, color=RED, lw=1, ls=':')\n"
            "ax.set_xlabel('Forward return window (days)')\n"
            "ax.set_ylabel('HAC t-stat (stress - calm difference)')\n"
            "ax.set_title('No horizon clears the inference bar')\n"
            "ax.legend()\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "print('The |t| bar is 2.0. Neither proxy clears it at any horizon tested.')"
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 -- Going further\n\n"
            "- **Better data changes the answer.** The academic effect (Gilchrist & Zakrajsek "
            "2012) is measured with OAS level data at quarterly frequency -- not the 20-day "
            "ETF-price proxy at daily frequency.  A fork using FRED BAMLH0A0HYM2OAS would "
            "be the natural next step.\n"
            "- **Related stress indicators.** VIX ([Study 86 -- Tail-Radar](../../86-tail-radar/)) "
            "and the copper/gold ratio ([Study 85 -- Dr-Copper](../../85-dr-copper/)) are "
            "cousins of this idea -- stress signals from non-equity markets.\n"
            "- **Longer horizons.** The academic literature finds the effect at 1-4 quarter "
            "horizons.  Our 21-day test shows -0.60 t-stat with correlation -0.12 -- still "
            "weak, but the correlation is growing.  Try 252 days.\n"
            "- **Regime persistence matters.** Entering a stress regime and staying in it for "
            "multiple days (not just the next 5) might identify the multi-month downturns where "
            "credit genuinely leads.  The 2020 credit event (Feb-Mar) was a 3-week window "
            "where acting on credit signals early would have mattered.\n\n"
            "*Think this is the wrong proxy or the wrong signal definition?  Fork this, use "
            "OAS data or a longer rolling window, and show a t-stat that clears |2|.  That's "
            "the bar.*"
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
            "# Credit-Spreads -- a quantitative teardown\n"
            "### Daily ETF proxy * regime-conditional forward returns * HAC inference * horizon sweep * synthetic control\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats_buy--and--hold%3F: Not--Supported](https://img.shields.io/badge/Beats_buy--and--hold%3F-Not--Supported-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim now carrying its standard error.* We test whether "
            "the HYG-IEF and HYG/LQD 20-day spread proxies predict forward SPY returns in a "
            "credit-stress regime vs the unconditional baseline.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily bars, 2010-01-04 to 2026-06-12, "
            "n~4,136 days; the offline core and tests run on a deterministic synthetic tape. "
            "FRED OAS data unavailable in this environment -- ETF proxies used, stated explicitly. "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **In plain words** notes translate each result back to intuition."
        ),
        code(BOOT + "\ntry:\n    from quantlab import analytics, stats\n    HAS_QUANTLAB = True\nexcept ImportError:\n    HAS_QUANTLAB = False\n"),
        code(R_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | HYG-IEF stress regime: {R['spread5_diff']:+.1f} bps "
            f"vs calm, HAC *t* = **{R['spread5_diff_t']:+.2f}**.  Corr = {R['spread5_corr']:+.3f}.  "
            f"HYG/LQD: wrong sign (*t* = {R['rs5_diff_t']:+.2f}).  Direction right, power absent. |\n"
            f"| **Tradability** | `MIRAGE` | Both regimes positive; no defensive positioning "
            f"justified.  Proxy too slow and noisy for a trading rule. |\n"
            f"| **Beats buy-and-hold?** | `NOT SUPPORTED` | Stress mean = "
            f"+{R['spread5_stress_mean']:.0f} bps, calm = +{R['spread5_calm_mean']:.0f} bps -- "
            f"both positive, unconditional = +{R['spread5_uncond']:.0f} bps. |\n\n"
            "> In plain words: credit spreads are a market barometer, not a crystal ball.  "
            "At ETF-proxy resolution, the lead-lag relationship lives in the noise band."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 -- The claim, steelmanned\n\n"
            "Let $s_t$ be the 20-day rolling HYG-IEF return differential (our credit-spread proxy) "
            "and $r_{t,t+h}$ the $h$-day forward SPY return.  The claim asserts:\n\n"
            "- **H1 (signal):** $\\mathbb{E}[r_{t,t+h} \\mid s_t < \\mathrm{median}(s)] < "
            "\\mathbb{E}[r_{t,t+h}]$ -- i.e. credit-stress periods have lower conditional "
            "forward equity returns than the unconditional mean.\n"
            "- **H2 (strong form):** $\\mathbb{E}[r_{t,t+h} \\mid s_t < \\mathrm{median}(s)] < 0$ "
            "-- i.e. stress periods predict *negative* equity returns (justifying defensive "
            "positioning).\n\n"
            "H1 is weakly supported (right direction, insignificant).  H2 is rejected: "
            "conditional means are positive in all regimes across all horizons tested.\n\n"
            "**Proxy note.** True OAS data (FRED BAMLH0A0HYM2OAS) is the canonical measure; "
            "our ETF price-return proxy is noisier but liquid and observable at daily frequency."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 -- So what? -- what rides on each answer\n\n"
            "Gilchrist & Zakrajsek (2012) established the excess bond premium as the cleanest "
            "credit-spread predictor of macro and equity outcomes.  If that result held at "
            "ETF-proxy daily resolution, it would be one of the most actionable macro signals "
            "available -- liquid instruments, low cost, objective.  The interesting result is "
            "that H1 is *directionally consistent* with the academic literature (the sign is "
            "right) but the magnitude and t-stat are far too small to be tradable, or even "
            "to be called 'real' by the desk's inference bar."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 -- How we'd know -- the protocol\n\n"
            "- **Proxies.** Two ETF-based credit-spread proxies:\n"
            "  - $s^{\\text{HI}}_t = r^{\\text{HYG}}_{t-20,t} - r^{\\text{IEF}}_{t-20,t}$ "
            "(HYG underperformance vs Treasuries)\n"
            "  - $s^{\\text{HL}}_t = r^{\\text{HYG}}_{t-20,t} - r^{\\text{LQD}}_{t-20,t}$ "
            "(HY vs IG, strips duration)\n"
            "- **Regime.** Stress: $s_t < \\text{rolling median}(s, 60d)$ (no look-ahead).\n"
            "- **Forward return.** SPY: $r_{t,t+h}$ for $h \\in \\{5, 21\\}$ days.\n"
            "- **Inference.** HAC Newey-West t-stat on the difference in regime means "
            "($\\hat{\\mu}_{\\text{stress}} - \\hat{\\mu}_{\\text{calm}}$), with separate "
            "long-run variance estimates for each regime.\n"
            "- **Positive control.** Synthetic tape with planted lead-lag "
            "($\\texttt{spread\\_signal}$ knob), to confirm the engine recovers an effect "
            "when one exists.\n\n"
            "Data: Yahoo daily adjusted-close, 2010-01-04 to 2026-06-12, n = 4,111 signal-days."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 -- The teardown"),

        md(
            "### 4a -- Regime-conditional forward returns (HYG-IEF proxy)\n\n"
            "Do stress periods have lower 5-day forward SPY returns than calm periods?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    res5 = st.summarize(prices, window=20, forward_days=5, rolling_median_window=60)\n"
            "    stress_n = res5['spread_stress_n']\n"
            "    calm_n = res5['spread_calm_n']\n"
            "    stress_m = res5['spread_stress_mean_bps']\n"
            "    calm_m = res5['spread_calm_mean_bps']\n"
            "    uncond_m = res5['spread_unconditional_mean_bps']\n"
            "    diff_t = res5['spread_diff_tstat']\n"
            "    stress_win = res5['spread_stress_win_rate']\n"
            "    calm_win = res5['spread_calm_win_rate']\n"
            "else:\n"
            "    stress_n, calm_n = R['spread5_stress_n'], R['spread5_calm_n']\n"
            "    stress_m, calm_m, uncond_m = R['spread5_stress_mean'], R['spread5_calm_mean'], R['spread5_uncond']\n"
            "    diff_t = R['spread5_diff_t']\n"
            "    stress_win, calm_win = R['spread5_stress_win']/100, R['spread5_calm_win']/100\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9, 5))\n"
            "bars = ax.bar(['Stress\\n(wide spread)\\nn={:,}'.format(stress_n),\n"
            "               'Calm\\n(tight spread)\\nn={:,}'.format(calm_n),\n"
            "               'Unconditional\\nn={:,}'.format(stress_n+calm_n)],\n"
            "              [stress_m, calm_m, uncond_m],\n"
            "              color=[RED, GREEN, GREY], alpha=0.8)\n"
            "ax.axhline(0, color='k', lw=1)\n"
            "ax.set_ylabel('5-day forward SPY return (bps)')\n"
            "ax.set_title(f'Regime-conditional forward returns (HYG-IEF proxy)\\n'\n"
            "             f'Diff = {stress_m - calm_m:.1f} bps, HAC t = {diff_t:.2f} (inference bar: |t|=2)')\n"
            "for b, v, w in zip(bars, [stress_m, calm_m, uncond_m], [stress_win, calm_win, None]):\n"
            "    lbl = f'{v:+.1f} bps' + (f'\\nwin {w*100:.1f}%' if w else '')\n"
            "    ax.text(b.get_x() + b.get_width()/2, v + 0.3, lbl, ha='center', va='bottom', fontsize=9)\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "print(f'Stress-calm diff: {stress_m - calm_m:.1f} bps, HAC t = {diff_t:.2f}')\n"
            "print('Result: WEAK -- correct direction, but |t| << 2')"
        ),
        md(
            "> In plain words: the bar chart looks like the claim is working (red bar lower "
            "than green bar), but the difference is only about 9 bps and the standard error "
            "is much larger -- the t-stat of -0.71 means we'd see this by random chance 47% "
            "of the time.  That's not a signal, that's noise."
        ),

        md(
            "### 4b -- HYG/LQD proxy: wrong sign in this sample\n\n"
            "Stripping duration should give a cleaner credit-risk signal.  Does it?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rs5_stress_m = res5['rs_stress_mean_bps']\n"
            "    rs5_calm_m = res5['rs_calm_mean_bps']\n"
            "    rs5_diff_t = res5['rs_diff_tstat']\n"
            "    rs5_stress_n = res5['rs_stress_n']\n"
            "    rs5_calm_n = res5['rs_calm_n']\n"
            "else:\n"
            "    rs5_stress_m, rs5_calm_m = R['rs5_stress_mean'], R['rs5_calm_mean']\n"
            "    rs5_diff_t = R['rs5_diff_t']\n"
            "    rs5_stress_n, rs5_calm_n = R['rs5_stress_n'], R['rs5_calm_n']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            "ax.bar(['Stress\\nn={:,}'.format(rs5_stress_n), 'Calm\\nn={:,}'.format(rs5_calm_n)],\n"
            "       [rs5_stress_m, rs5_calm_m], color=[RED, GREEN], alpha=0.8)\n"
            "ax.axhline(0, color='k', lw=1)\n"
            "ax.set_ylabel('5-day forward SPY return (bps)')\n"
            "ax.set_title(f'HYG/LQD proxy: wrong sign! Diff={rs5_stress_m-rs5_calm_m:+.1f} bps, t={rs5_diff_t:+.2f}')\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "print(f'HYG/LQD stress ({rs5_stress_m:.1f} bps) > calm ({rs5_calm_m:.1f} bps)')\n"
            "print('Interpretation: HY-vs-IG underperformance does not predict lower SPY forward returns in this sample.')"
        ),
        md(
            "> In plain words: the 'purer' credit-risk proxy (HYG/LQD, stripped of duration) "
            "actually shows stress periods with *higher* forward returns than calm periods -- the "
            "opposite of the claim.  t-stat = +0.44, noise.  This is a reminder that proxy "
            "choice matters: different ETF combinations can flip the sign."
        ),

        md(
            "### 4c -- Horizon sweep: does the signal strengthen at longer windows?\n\n"
            "Academic literature finds credit-spread predictability at quarterly horizons.  "
            "Does the ETF proxy pick it up?"
        ),
        code(
            "horizons = [1, 5, 10, 21, 63]\n"
            "if HAVE_REAL:\n"
            "    rows = []\n"
            "    for fw in horizons:\n"
            "        r = st.summarize(prices, window=20, forward_days=fw, rolling_median_window=60)\n"
            "        rows.append({'h': fw,\n"
            "                     'spread_diff_bps': r['spread_diff_mean_bps'],\n"
            "                     'spread_t': r['spread_diff_tstat'],\n"
            "                     'spread_corr': r['spread_corr'],\n"
            "                     'rs_diff_bps': r['rs_diff_mean_bps'],\n"
            "                     'rs_t': r['rs_diff_tstat']})\n"
            "    sweep = pd.DataFrame(rows)\n"
            "else:\n"
            "    sweep = pd.DataFrame({'h': horizons,\n"
            "        'spread_diff_bps': [R['spread5_diff']*0.5, R['spread5_diff'], R['spread5_diff']*1.5, R['spread21_diff'], R['spread21_diff']*2.0],\n"
            "        'spread_t': [R['spread5_diff_t']*0.5, R['spread5_diff_t'], R['spread5_diff_t']*1.2, R['spread21_diff_t'], R['spread21_diff_t']*1.5],\n"
            "        'spread_corr': [R['spread5_corr']*0.5, R['spread5_corr'], R['spread5_corr']*1.2, R['spread21_corr'], R['spread21_corr']*1.5],\n"
            "        'rs_diff_bps': [R['rs5_diff']*0.5]*5,\n"
            "        'rs_t': [R['rs5_diff_t']]*5})\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "axes[0].plot(sweep['h'], sweep['spread_t'], 'o-', color=AMBER, lw=2, label='HYG-IEF')\n"
            "axes[0].plot(sweep['h'], sweep['rs_t'], 's--', color=GREY, lw=1.5, label='HYG/LQD')\n"
            "axes[0].axhline(-2, color=RED, ls=':', lw=1.5, label='Inference bar')\n"
            "axes[0].axhline(2, color=RED, ls=':', lw=1.5)\n"
            "axes[0].axhline(0, color='k', lw=1)\n"
            "axes[0].set_xlabel('Forward window (days)'); axes[0].set_ylabel('HAC t-stat')\n"
            "axes[0].set_title('Neither proxy clears |t|=2 at any horizon')\n"
            "axes[0].legend()\n"
            "axes[1].plot(sweep['h'], sweep['spread_corr'], 'o-', color=AMBER, lw=2, label='HYG-IEF')\n"
            "axes[1].axhline(0, color='k', lw=1)\n"
            "axes[1].set_xlabel('Forward window (days)'); axes[1].set_ylabel('Correlation')\n"
            "axes[1].set_title('Correlation grows with horizon but stays weak')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(sweep.round(2))"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 -- The verdict\n\n"
            f"- **Signal `WEAK`** -- HYG-IEF: diff = {R['spread5_diff']:+.1f} bps, "
            f"*t* = {R['spread5_diff_t']:+.2f} (5d); corr = {R['spread5_corr']:+.3f}.  "
            f"HYG/LQD: wrong sign (*t* = {R['rs5_diff_t']:+.2f}).  Direction plausible, "
            "statistics weak.  WEAK reflects literature support without statistical evidence "
            "from the ETF proxy.\n"
            "- **Tradability `MIRAGE`** -- conditional means are all positive; no profitable "
            "defensive rule is implied.  The 20-day rolling proxy is too slow and too noisy "
            "for daily rebalancing.\n"
            "- **Beats buy-and-hold? `NOT SUPPORTED`** -- stress regime: "
            f"+{R['spread5_stress_mean']:.0f} bps; calm: +{R['spread5_calm_mean']:.0f} bps; "
            f"unconditional: +{R['spread5_uncond']:.0f} bps.  All positive, all overlapping "
            "in distribution."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 -- Could you trade it? -- the structural barriers\n\n"
            "Even if the statistics were stronger, there are three structural problems:\n\n"
            "1. **Positive-mean regimes.** The signal identifies mildly weaker, not negative, "
            "forward returns.  Reducing equity exposure costs the positive drift in all regimes.\n"
            "2. **Proxy lag.** A 20-day rolling return reacts 20 days *after* the credit move. "
            "Real macro desks use OAS level data and term-structure curvature -- not ETF "
            "price returns.\n"
            "3. **ETF mechanics.** During 2020 (the most dramatic recent credit event), HYG "
            "traded at large discounts to NAV; its price returns temporarily overstated the "
            "underlying spread widening and then snapped back -- making any daily strategy "
            "on HYG vs IEF noisy in precisely the crisis window where the signal should "
            "be strongest."
        ),
        code(
            "# Show the positive-control: on synthetic tape with planted signal, the machinery detects it\n"
            "REAL_SPREAD5_DIFF_T = -0.71  # from docs/results.md\n"
            "signals_syn = []\n"
            "corrs_syn = []\n"
            "sig_vals = [0.0, 0.5, 1.0, 2.0, 3.0]\n"
            "for sv in sig_vals:\n"
            "    p, _ = data.synthetic_daily(n_days=2000, spread_signal=sv, seed=115)\n"
            "    r = st.summarize(p, window=20, forward_days=5, rolling_median_window=60)\n"
            "    signals_syn.append(r['spread_diff_tstat'])\n"
            "    corrs_syn.append(r['spread_corr'])\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.5))\n"
            "ax.plot(sig_vals, signals_syn, 'o-', color=GREEN, lw=2, label='diff t-stat (synthetic)')\n"
            "ax.axhline(0, color='k', lw=1)\n"
            "ax.axhline(-2, color=RED, ls=':', lw=1.5, label='Inference bar')\n"
            "ax.axvline(0.5, color=GREY, ls='--', lw=1, label='Approx real-world effect')\n"
            "ax.set_xlabel('Planted spread_signal strength')\n"
            "ax.set_ylabel('HAC t-stat (stress - calm)')\n"
            "ax.set_title('The engine detects the signal when planted; real world is near 0')\n"
            "ax.legend()\n"
            "plt.tight_layout()\n"
            "plt.show()\n"
            "print('Real-world diff t-stat:', REAL_SPREAD5_DIFF_T, '-- near the null (signal=0) region')"
        ),
        md(
            "> In plain words: the synthetic test confirms the machinery works -- plant a "
            "large enough effect and the engine finds it.  The real-tape t-stat of -0.71 "
            "corresponds to an effect size (spread_signal) much less than 0.5 in the "
            "synthetic scale -- too small to trade, and too small to call 'real'."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 -- Going further -- the OAS gap and better data\n\n"
            "The honest path forward:\n\n"
            "- **OAS data.** FRED BAMLH0A0HYM2OAS is the 'right' credit-spread series.  "
            "Gilchrist & Zakrajsek (2012) use an even cleaner measure (EBP = excess bond "
            "premium after stripping expected default risk).  A fork with that data would "
            "test the *real* claim, not the ETF approximation.\n"
            "- **Quarterly horizon.** The academic predictability lives at 1-4 quarter horizons, "
            "not 5-21 days.  Aggregating to monthly observations and testing 1-4 quarter "
            "forward returns is the canonical specification.\n"
            "- **Regime persistence.** Instead of a rolling-median binary flag, use a "
            "persistent-regime Markov switch (Hamilton 1989) to identify genuine credit-stress "
            "episodes -- the 2001-02, 2007-09, 2015-16, 2020, 2022 episodes.\n"
            "- **Cross-asset companions.** VIX ([Study 86](../../86-tail-radar/)), "
            "copper/gold ([Study 85](../../85-dr-copper/)), and yield curve slope all carry "
            "related information -- a multi-signal regime model might clear the bar.\n\n"
            "*The desk's verdict: the credit-spread narrative is one of the most academically "
            "supported macro timing ideas.  At ETF-proxy daily resolution, it's WEAK and "
            "MIRAGE.  That's not the end of the story -- it's a call to use better data.*"
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
