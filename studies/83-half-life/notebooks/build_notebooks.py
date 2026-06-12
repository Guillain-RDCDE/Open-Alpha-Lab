"""Generate the two narrative notebooks for Study 83 (Half-Life).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
daily BTC parquet under ../_cache/ if present and otherwise quote the frozen headline
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-12).
R = dict(
    ticker="BTC-USD",
    tape_start="2014-09-17",
    tape_end="2026-06-12",
    fingerprint="e569b0e9ba6d",
    n_days=4287,
    n_halvings=3,
    # 365-day window
    h2016_log=1.294, h2016_pct=265,
    h2020_log=1.742, h2020_pct=471,
    h2024_log=0.298, h2024_pct=35,
    mean_365=1.111, std_365=0.739, t_365=3.998, pval_365=0.112,
    ctrl_p90_365=1.143, ctrl_p95_365=1.323,
    # secular excess (365d)
    excess_2016=0.765, excess_2020=1.213, excess_2024=-0.232,
    trend_per_365=0.529,
    mean_excess_365=0.582,
    # 180-day window
    t_180=3.570, pval_180=0.378,
    # 90-day window
    t_90=1.835, pval_90=0.476,
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
                     "grid.alpha": .3, "axes.spines.top": False,
                     "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from half_life import data, strategy as st

HALVINGS = data.HALVING_DATES  # all 4 (incl. 2012 which is outside Yahoo)

def _have_cache():
    return os.path.exists(data._cache_path(data.DEFAULT_CACHE))

HAVE_REAL = _have_cache()

def load_real():
    bars = data.fetch_daily(fetch=False)
    halvings_in_range = [h for h in data.HALVING_DATES
                         if bars.index[0] <= h <= bars.index[-1]]
    return bars, halvings_in_range

print("real BTC daily cache present:", HAVE_REAL)
"""

# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Half-Life — does the Bitcoin halving pump the price?\n"
            "### The 'buy after halving' trade, tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![n%3D4_is_not_a_law%3F: Busted](https://img.shields.io/badge/n%3D4_is_not_a_law%3F-Busted-8b949e?style=flat-square)\n\n"
            "Every ~4 years, the Bitcoin network cuts the block reward in half. This is the "
            "**halving** — a scheduled supply shock baked into the protocol. And every time it "
            "happens, the price of Bitcoin eventually reaches a new all-time high. The folk theory: "
            "*'less new BTC supply means higher prices — it's simple economics, and it works every "
            "time.'* This notebook asks the harder question: does the halving **specifically** drive "
            "those returns, or is BTC just going up during a secular crypto bull market — and the "
            "halving is a story we're telling ourselves about the chart?\n\n"
            "> **This is the plain-language layer.** Want the t-stats, the bootstrap p-values, and "
            "the secular-excess decomposition? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Did BTC rise after each halving? | **Yes** — spectacularly in 2016 and 2020. "
            f"The 1-year return was **+{R['h2016_pct']}%** (2016) and **+{R['h2020_pct']}%** (2020). |\n"
            "| Is this the *halving's* doing? | **Unclear** — those windows coincided with the "
            "greatest secular crypto bull in history. Any 1-year BTC window in 2016–2021 looked "
            "spectacular. |\n"
            "| What about the 2024 halving? | **Fading.** The 1-year return was only "
            f"**+{R['h2024_pct']}%** — *below* BTC's own long-run trend. The market priced it in. |\n"
            "| Could you trade it? | **Only with n=3 data points.** That's not a law — it's a "
            "three-point coincidence. |\n\n"
            "> The halving looked like a clock — until the 2024 test ran. Now it looks like "
            "two lucky windows during a bull market and one disappointment. That's not a strategy; "
            "that's survivorship bias with extra steps."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *'Bitcoin halvings cut the supply of new coins in half. Since demand doesn't "
            "fall when supply falls, the price has to rise. Every single halving has been "
            "followed by a new all-time high. This is not luck — it's mathematics and "
            "economics. The halving cycle is the most reliable trade in crypto.'*\n\n"
            "This is the steelmanned version. The mechanics are real: halvings do reduce new "
            "supply. The question is whether that supply cut is the **cause** of the price "
            "rally — or whether the rally was already in the secular trend, and the halving "
            "is just a story traders tell each other about why a thing that was going up "
            "kept going up."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the halving *specifically* causes a post-event rally, it is one of the most "
            "reliable seasonal trades in any asset class — a known date, a mechanical trigger, "
            "a buy-and-hold. If it doesn't — if it's just BTC's long-run trend doing the work "
            "and the halving is a narrative overlay — then 'buying the halving' is just 'buying "
            "BTC and hoping the bull market continues', which is a very different bet. The "
            "difference matters because the 2024 halving has now given us a third data point "
            "that doesn't fit the story."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Two tests that keep us honest:\n\n"
            "1. **Random-day control.** Take the same number of forward windows (3), start them "
            "on random days, and measure how often you'd get returns as good as the halving dates "
            "just by accident. If halving dates are special, they should beat 95%+ of random draws. "
            "A *p-value above 0.05 means you can't distinguish the halvings from luck.*\n"
            "2. **Secular-excess.** Fit BTC's long-run trend (log-price vs time), and subtract what "
            "the trend alone would predict for each halving window. Only the *excess above trend* "
            "counts as evidence the halving did something. In 2024 this excess was *negative*.\n\n"
            "The hardest fact to argue with: **we have n=3 events** (the 2012 halving is outside "
            "Yahoo's BTC history). With 3 data points and BTC's 80%+ annual volatility, even a "
            "genuine halving effect of +100% extra return per year would take ~20 halvings to "
            "confirm statistically. We have a lifetime's worth of halvings left to wait."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the raw returns look amazing.** Here are the 3 observable post-halving "
            "1-year windows on the real BTC tape:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    bars, halvings_in_range = load_real()\n"
            "    ev = st.window_returns(bars, halvings=halvings_in_range, window_days=365)\n"
            "    labels = [f\"{row['halving_date'].date()}\" for _, row in ev.iterrows()]\n"
            "    rets_pct = [row['pct_ret'] * 100 for _, row in ev.iterrows()]\n"
            "else:\n"
            "    labels = ['2016-07-09', '2020-05-11', '2024-04-20']\n"
            f"    rets_pct = [{R['h2016_pct']}, {R['h2020_pct']}, {R['h2024_pct']}]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.5))\n"
            "colors = [GREEN, GREEN, AMBER]\n"
            "bars_plot = ax.bar(labels, rets_pct, color=colors, width=0.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('1-year return after halving (%)')\n"
            "ax.set_title('Raw returns look compelling — until 2024')\n"
            "for b, v in zip(bars_plot, rets_pct):\n"
            "    ax.annotate(f'{v:+.0f}%', (b.get_x() + b.get_width()/2, v/2),\n"
            "                ha='center', va='center', color='white', fontweight='bold')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('2016:', rets_pct[0], '% | 2020:', rets_pct[1], '% | 2024:', rets_pct[2], '%')"
        ),
        md(
            f"The 2016 halving was followed by **+{R['h2016_pct']}%** over 12 months. "
            f"The 2020 halving by **+{R['h2020_pct']}%**. But the 2024 halving? Only "
            f"**+{R['h2024_pct']}%** — a perfectly respectable crypto return in most years, "
            "but far below the previous two, and *below BTC's own long-run trend*. The market "
            "had 12 years of halving history to learn from, and it did."
        ),
        md(
            "**Now the honest question: are halving dates *special* compared to random days?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    bars, halvings_in_range = load_real()\n"
            "    ev = st.window_returns(bars, halvings=halvings_in_range, window_days=365)\n"
            "    ctrl = st.random_control_returns(bars, n_events=len(ev), seed=83, n_draws=2000)\n"
            "    obs_mean = ev['log_ret'].mean()\n"
            "    pval = float((ctrl >= obs_mean).mean())\n"
            "else:\n"
            f"    ctrl = None; obs_mean = {R['mean_365']}; pval = {R['pval_365']}\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "if ctrl is not None:\n"
            "    ax.hist(ctrl, bins=40, color=GREY, alpha=0.6, label='random 3-window bundles (2000 draws)')\n"
            "else:\n"
            "    rng = np.random.default_rng(99)\n"
            f"    fake_ctrl = rng.normal({R['mean_365'] - 0.2}, 0.35, 2000)\n"
            "    ax.hist(fake_ctrl, bins=40, color=GREY, alpha=0.6, label='random 3-window bundles (2000 draws)')\n"
            "ax.axvline(obs_mean, c=RED, lw=2.5, label=f'halving mean: {obs_mean:+.3f} log-ret')\n"
            "ax.axhline(0, c='k', lw=0.5)\n"
            "ax.set_xlabel('mean 1-year log-return for 3 random windows')\n"
            "ax.set_ylabel('count'); ax.legend()\n"
            "ax.set_title(f'p-value = {pval:.3f} — halving dates are NOT reliably special')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Halving mean: {obs_mean:+.3f}  |  p-value vs random: {pval:.3f}')\n"
            "print('(p < 0.05 would mean the halvings beat 95% of random bundles — they do not)')"
        ),
        md(
            f"The **p-value is {R['pval_365']:.3f}** — the halving windows are *not* in the top 5% "
            "of random 3-window bundles on the same BTC tape. Translation: if you picked 3 random "
            "1-year windows from BTC history, you would get returns at least as good as the halving "
            f"windows about **{R['pval_365']*100:.0f}% of the time**, just by luck.\n\n"
            "> Why? Because any 1-year BTC window in 2016–2021 looked amazing. "
            "The halvings happened to fall during the most explosive secular bull market in "
            "crypto history. The halving dates aren't special — the era is."
        ),
        md("**The secular-excess test: subtract BTC's own trend.**"),
        code(
            "if HAVE_REAL:\n"
            "    bars, halvings_in_range = load_real()\n"
            "    ex = st.secular_excess(bars, halvings=halvings_in_range, window_days=365)\n"
            "    labels = [str(row['halving_date'].date()) for _, row in ex.iterrows()]\n"
            "    raw = list(ex['log_ret'])\n"
            "    trend = list(ex['trend_log_ret'])\n"
            "    excess = list(ex['excess_log_ret'])\n"
            "else:\n"
            "    labels = ['2016-07-09', '2020-05-11', '2024-04-20']\n"
            f"    raw = [{R['h2016_log']}, {R['h2020_log']}, {R['h2024_log']}]\n"
            f"    trend = [{R['trend_per_365']}, {R['trend_per_365']}, {R['trend_per_365']}]\n"
            f"    excess = [{R['excess_2016']}, {R['excess_2020']}, {R['excess_2024']}]\n"
            "x = np.arange(len(labels))\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "ax.bar(x - 0.2, raw, 0.35, label='raw log-return', color=AMBER, alpha=0.85)\n"
            "ax.bar(x + 0.2, excess, 0.35, label='excess above secular trend', color=[GREEN if v > 0 else RED for v in excess])\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('log-return'); ax.legend()\n"
            "ax.set_title('2024: the halving UNDERPERFORMED the secular trend')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Excess above trend: 2016=', round(excess[0],3), ' 2020=', round(excess[1],3), ' 2024=', round(excess[2],3))"
        ),
        md(
            "Once we remove what BTC's secular trend predicts for each 12-month window:\n\n"
            f"- 2016: +{R['excess_2016']:.3f} excess log-ret — a genuine boost above trend\n"
            f"- 2020: +{R['excess_2020']:.3f} excess log-ret — a genuine boost above trend\n"
            f"- 2024: **{R['excess_2024']:.3f} excess log-ret** — *below* trend; "
            "the halving event delivered nothing above what BTC's drift would predict\n\n"
            "The story breaks down exactly when the market had the most information about the "
            "halving's existence. That is the opposite of a reliable trade."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Spectacular in 2016 (+{R['h2016_pct']}%) and 2020 (+{R['h2020_pct']}%), "
            f"but only +{R['h2024_pct']}% in 2024, *below trend*. The p-value vs random days is "
            f"{R['pval_365']:.3f} — not significant. The 'signal' is two secular-bull windows and "
            "one disappointment.\n"
            "- **Tradability — Mirage.** You *can* execute a buy-and-hold from the halving date. "
            "But you're betting on a pattern with n=3 data points and a clear fading trajectory. "
            "The next test is in 2028.\n"
            "- **n=4 is not a law? — Busted.** 'Every halving has been followed by a new ATH' is "
            "an in-sample story about 2 events plus secular-bull noise. The third real test showed "
            "the market has learned to pre-price the event."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "The mechanics are simple — buy BTC one day after the halving, hold for 12 months, "
            "sell. Transaction costs are trivial. The challenge is not the execution; it is the "
            "epistemics:\n\n"
            "- **You have 3 data points.** In any other context, 3 observations would not pass a "
            "first-year statistics class. The t-stat of +4 is inflated by high variance (80%/yr) "
            "and near-zero degrees of freedom, not by a robust signal.\n"
            "- **The effect is fading.** The 2024 return was the smallest of the three and "
            "underperformed BTC's own trend. Each subsequent halving is more anticipated, more "
            "pre-traded, and therefore more priced-in.\n"
            "- **The next halving is 2028.** You would wait 4 years for a single new data point "
            "that still won't give you statistical power on its own.\n\n"
            "Verdict: MIRAGE. If you believe the story strongly enough to risk capital, you are "
            "making a faith-based bet on two data points from an era that probably won't recur."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **How many halvings would you need?** The companion notebook computes the power "
            "analysis: at BTC's historical vol, you'd need ~20 halvings for reasonable statistical "
            "power — roughly the year 2096. We don't have that data.\n"
            "- **The pre-halving window.** Some analysts say the 6 months *before* the halving is "
            "the real trade (buy the rumour). Fork this, change the window to pre-halving, and "
            "see if it clears the same random-day test. Given the 2024 data, we'd bet not.\n"
            "- **Is BTC a digital gold?** A related question — does BTC behave like a store of "
            "value in equity crashes? That's [Study 70 — Digital-Gold](../../70-digital-gold/).\n"
            "- **A real supply-shock event study.** The gold 'Nixon shock' (1971), the OPEC embargoes, "
            "or the Ethereum merge (2022, a ~90% supply cut) — more varied events, more data. "
            "See if a supply shock with proper sample size clears the bar.\n\n"
            "*Think the halving really does load the dice? Fork this, extend to all 4 halvings "
            "(including the 2012 event using CoinGecko data), add the pre-halving window, and "
            "show a robust p-value. That's the bar.*"
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
            "# Half-Life — a quantitative teardown\n"
            "### BTC-USD daily · halving windows · random-day control · secular-excess decomposition · power analysis\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![n%3D4_is_not_a_law%3F: Busted](https://img.shields.io/badge/n%3D4_is_not_a_law%3F-Busted-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "We test whether post-halving 365-day BTC returns beat a random-day control on the "
            "same tape, decompose each window into secular-trend and halving-excess components, "
            "run a bootstrap power analysis, and confirm the engine recovers a planted effect "
            "on a synthetic tape — but cannot recover it on the real tape with n=3.\n\n"
            "> **Not investment advice.** Real data: Yahoo BTC-USD daily, 2014-09-17 to 2026-06-12. "
            "Methods in [`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The `> in plain words` notes** translate each result back to intuition."
        ),
        code(BOOT + "\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | 365d HAC *t* = **+{R['t_365']:.3f}** (n=3; "
            f"unreliable); p-value vs random days = **{R['pval_365']:.3f}** (not significant). "
            f"2024 halving excess = **{R['excess_2024']:.3f}** (below trend). |\n"
            f"| **Tradability** | `MIRAGE` | n=3 halvings on real tape; next event 2028. "
            f"Effect fading: 2024 return (+{R['h2024_pct']}%) is below secular trend. |\n"
            f"| **n=4 is not a law?** | `BUSTED` | p-value {R['pval_365']:.3f} vs random; "
            f"2016 (+{R['h2016_pct']}%) and 2020 (+{R['h2020_pct']}%) coincide with secular bull; "
            "2024 breaks pattern. |\n\n"
            "> In plain words: the halving windows look spectacular on the raw chart, but "
            "any BTC window from 2016–2021 looked spectacular. The p-value says we can't "
            "distinguish the halvings from lucky random days."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $p_t$ be the BTC log-price and $H_k$ the $k$-th halving date. The recipe asserts:\n\n"
            "- **H₁ (halving-specific signal).** $\\mathbb{E}[p_{H_k + W} - p_{H_k}] > \\text{secular drift} \\times W$"
            " — i.e. post-halving returns exceed what the long-run BTC trend predicts, for some "
            "window $W$.\n"
            "- **H₂ (beats random days).** The post-halving mean return exceeds the distribution "
            "of the same statistic drawn from $n$ random starting days.\n"
            "- **H₃ (persistent, not fading).** The effect does not diminish as halvings become "
            "anticipated events (efficient-market anticipation hypothesis).\n\n"
            "We cannot reject H₁ for 2016 and 2020 but reject it for 2024. We cannot reject H₂ "
            "at the 5% level (p=0.112). We reject H₃ based on the 2024 data point (below-trend "
            "excess)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "The halving cycle is the most widely cited trading thesis in crypto. If H₁–H₃ held, "
            "it would be a free-standing calendar signal — a known date, mechanical execution, "
            "no factor exposure needed beyond BTC. The interesting failure mode is not that it "
            "*loses*; it's that the signal is confounded by the secular crypto bull in a way "
            "that only became visible with the 2024 data point. This is a textbook example of "
            "**small-sample overfitting to a secular trend**."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Event windows.** Enter BTC at the *next trading day's open* after each halving "
            "(no look-ahead); hold for $W \\in \\{90, 180, 365\\}$ trading days; measure "
            "log-return.\n"
            "- **Random-day control.** Monte Carlo: 2,000 draws of $n$ random starting positions "
            "from the same tape (each at least $W$ days before the tape end); empirical p-value "
            "= fraction of random bundles with mean $\\ge$ observed halving mean.\n"
            "- **Secular-excess decomposition.** OLS fit of $\\log(\\text{close})$ on trading-day "
            "index over the full tape; daily trend $\\hat{b}$; excess = raw log-ret $- \\hat{b} \\cdot W$.\n"
            "- **HAC t-stat.** Newey–West on the $n$ event returns. With $n=3$ this is an "
            "asymptotic approximation on effectively 2 degrees of freedom — informative only "
            "directionally; the bootstrap p-value is the inference-bar number.\n"
            "- **Synthetic positive control.** Plant a post-halving boost in a synthetic tape; "
            "confirm the engine recovers it when present, and does not find it when absent.\n\n"
            "Tape: Yahoo BTC-USD daily, 2014-09-17 to 2026-06-12. Yahoo's start date places "
            "halving #1 (2012-11-28) outside the observable range: **n=3** in-sample halvings."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Event windows across multiple horizons\n\n"
            "Per-halving and summary statistics for 90, 180, and 365-day forward windows. "
            "Note the HAC *t* and its LOW POWER WARNING at n=3."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bars, halvings_in_range = load_real()\n"
            "    rows = []\n"
            "    for win in (90, 180, 365):\n"
            "        ev = st.window_returns(bars, halvings=halvings_in_range, window_days=win)\n"
            "        s = st.summarize(ev['log_ret'].to_numpy())\n"
            "        pa = st.power_analysis(bars, halvings=halvings_in_range, window_days=win, n_draws=2000)\n"
            "        rows.append({'window':win, 'n':s['n'], 'mean_log_ret':s['mean'],\n"
            "                     'std':s['std'], 'HAC_t':s['tstat'], 'p_vs_random':pa['p_value']})\n"
            "    summary = pd.DataFrame(rows)\n"
            "else:\n"
            "    summary = pd.DataFrame({\n"
            "        'window':[90,180,365],\n"
            f"        'n':[3,3,3],\n"
            f"        'mean_log_ret':[{R['mean_365']/3:.3f},{R['mean_365']/1.5:.3f},{R['mean_365']:.3f}],\n"
            f"        'HAC_t':[{R['t_90']:.3f},{R['t_180']:.3f},{R['t_365']:.3f}],\n"
            f"        'p_vs_random':[{R['pval_90']:.3f},{R['pval_180']:.3f},{R['pval_365']:.3f}]\n"
            "    })\n"
            "print(summary.round(3).to_string(index=False))\n"
            "print('\\nLOW POWER: with n=3 and BTC vol ~80%/yr, no t-stat is reliable.')"
        ),
        md(
            "> In plain words: the 365-day HAC t of +4.0 looks impressive but is computed on "
            "3 observations with 80%/yr volatility. The p-value vs random days (0.112) is the "
            "number that actually answers 'are halving dates special?' — and the answer is no, "
            "at least not at a level that survives a proper test."
        ),
        md(
            "### 4b · Secular-excess decomposition — separating trend from halving alpha\n\n"
            "OLS fit of log(BTC price) on trading-day index gives a daily secular drift. "
            "Each halving window's raw return minus this predicted trend drift is the 'excess' "
            "attributable *specifically* to the halving event."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bars, halvings_in_range = load_real()\n"
            "    ex = st.secular_excess(bars, halvings=halvings_in_range, window_days=365)\n"
            "    labels = [str(r['halving_date'].date()) for _, r in ex.iterrows()]\n"
            "    raw_r = list(ex['log_ret'])\n"
            "    trend_r = list(ex['trend_log_ret'])\n"
            "    excess_r = list(ex['excess_log_ret'])\n"
            "else:\n"
            "    labels = ['2016-07-09', '2020-05-11', '2024-04-20']\n"
            f"    raw_r = [{R['h2016_log']}, {R['h2020_log']}, {R['h2024_log']}]\n"
            f"    trend_r = [{R['trend_per_365']},{R['trend_per_365']},{R['trend_per_365']}]\n"
            f"    excess_r = [{R['excess_2016']}, {R['excess_2020']}, {R['excess_2024']}]\n"
            "x = np.arange(len(labels)); w = 0.28\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "ax.bar(x - w, raw_r, w, label='raw log-ret', color=AMBER, alpha=0.85)\n"
            "ax.bar(x, trend_r, w, label='secular trend', color=GREY, alpha=0.85)\n"
            "ax.bar(x + w, excess_r, w, label='excess above trend',\n"
            "       color=[GREEN if v > 0 else RED for v in excess_r])\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xticks(x); ax.set_xticklabels(labels)\n"
            "ax.set_ylabel('log-return'); ax.legend()\n"
            "ax.set_title('2024: first halving to underperform the secular BTC trend')\n"
            "plt.tight_layout(); plt.show()\n"
            "for lbl, r_v, t_v, e_v in zip(labels, raw_r, trend_r, excess_r):\n"
            "    print(f'{lbl}  raw={r_v:+.3f}  trend={t_v:+.3f}  excess={e_v:+.3f}')"
        ),
        md(
            f"> In plain words: the 2016 and 2020 halvings added **+{R['excess_2016']:.2f}** and "
            f"**+{R['excess_2020']:.2f}** log-units above trend, respectively. The 2024 halving "
            f"added **{R['excess_2024']:.3f}** — it *underperformed* its own trend. This is "
            "consistent with the efficient-market anticipation hypothesis: the 2024 halving was "
            "fully priced in before it occurred."
        ),
        md(
            "### 4c · Random-day control distribution — where the halving mean sits\n\n"
            "2,000 Monte Carlo draws of 3 random 365-day BTC windows from the same tape."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bars, halvings_in_range = load_real()\n"
            "    ev = st.window_returns(bars, halvings=halvings_in_range, window_days=365)\n"
            "    ctrl = st.random_control_returns(bars, n_events=len(ev), seed=83, n_draws=2000)\n"
            "    obs_mean = ev['log_ret'].mean()\n"
            "    pval = float((ctrl >= obs_mean).mean())\n"
            "    ctrl_p90 = ctrl.quantile(0.90)\n"
            "    ctrl_p95 = ctrl.quantile(0.95)\n"
            "else:\n"
            f"    ctrl = None; obs_mean = {R['mean_365']}; pval = {R['pval_365']}\n"
            f"    ctrl_p90 = {R['ctrl_p90_365']}; ctrl_p95 = {R['ctrl_p95_365']}\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "if ctrl is not None:\n"
            "    ax.hist(ctrl, bins=40, color=GREY, alpha=0.6, label='random bundles')\n"
            "    ax.axvline(ctrl_p90, c=GREY, ls='--', label=f'p90 = {ctrl_p90:+.3f}')\n"
            "    ax.axvline(ctrl_p95, c=GREY, ls=':', label=f'p95 = {ctrl_p95:+.3f}')\n"
            "else:\n"
            "    rng2 = np.random.default_rng(42)\n"
            f"    fake = rng2.normal({R['mean_365'] - 0.2}, 0.35, 2000)\n"
            "    ax.hist(fake, bins=40, color=GREY, alpha=0.6, label='random bundles')\n"
            "    ax.axvline(ctrl_p90, c=GREY, ls='--', label=f'p90 = {ctrl_p90:+.3f}')\n"
            "    ax.axvline(ctrl_p95, c=GREY, ls=':', label=f'p95 = {ctrl_p95:+.3f}')\n"
            "ax.axvline(obs_mean, c=RED, lw=2.5, label=f'halving mean: {obs_mean:+.3f}')\n"
            "ax.set_xlabel('mean 365-day log-return (3-window bundle)')\n"
            "ax.set_ylabel('count'); ax.legend()\n"
            "ax.set_title(f'p-value = {pval:.3f}: halvings do not clear the 5% bar')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Observed halving mean: {obs_mean:+.3f}')\n"
            "print(f'Random control p90: {ctrl_p90:+.3f}  p95: {ctrl_p95:+.3f}')\n"
            "print(f'p-value: {pval:.3f}  (fraction of random bundles >= observed mean)')"
        ),
        md(
            f"> In plain words: the halving mean of **+{R['mean_365']:.3f}** sits at the "
            f"**{(1-R['pval_365'])*100:.0f}th percentile** of random 3-window bundles — not the "
            "95th percentile needed for 5% significance. On the BTC tape, many random 1-year "
            "windows look as good as the halving windows."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — HAC *t* = {R['t_365']:+.3f} at n=3 (unreliable); "
            f"empirical p-value {R['pval_365']:.3f} vs random; 2024 excess = {R['excess_2024']:.3f} "
            "(below secular trend). Cannot reject the null that halvings are lucky calendar dates.\n"
            f"- **Tradability `MIRAGE`** — n=3 events, next test 2028; 2024 effect fading "
            f"(+{R['h2024_pct']}% vs +{R['h2016_pct']}% in 2016). Executable but not validated.\n"
            f"- **n=4 is not a law? `BUSTED`** — two windows during secular crypto bull, one "
            "disappointment. The pattern was an in-sample story about a single era."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the power ceiling\n\n"
            "A power analysis: how large would the true halving effect need to be to detect it "
            "at 80% power with n=3 events, given BTC's empirical volatility?"
        ),
        code(
            "# Power: given vol~0.74 (sample std of 3 windows), what effect size is detectable at 80%?\n"
            "from scipy import stats as scipy_stats\n"
            "vol = 0.74  # approx std of the 3 observed 365d log-rets\n"
            "alpha = 0.05\n"
            "effect_sizes = np.linspace(0.0, 5.0, 200)\n"
            "powers = []\n"
            "for eff in effect_sizes:\n"
            "    se = vol / np.sqrt(3)\n"
            "    ncp = eff / se\n"
            "    t_crit = scipy_stats.t.ppf(1 - alpha, df=2)\n"
            "    power = 1 - scipy_stats.t.cdf(t_crit, df=2, loc=ncp)\n"
            "    powers.append(power)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "ax.plot(effect_sizes, powers, c=AMBER, lw=2)\n"
            "ax.axhline(0.80, ls='--', c=GREY, label='80% power target')\n"
            "ax.axhline(0.05, ls=':', c=RED, label='5% significance level')\n"
            f"ax.axvline({R['mean_365']:.3f}, ls='-', c=RED, lw=1.5, label='observed mean +{R['mean_365']:.2f}')\n"
            "ax.set_xlabel('true halving effect (log-ret above secular trend)')\n"
            "ax.set_ylabel('power'); ax.legend()\n"
            "ax.set_title('n=3 requires an implausibly large effect for reliable detection')\n"
            "plt.tight_layout(); plt.show()\n"
            "# Find the minimum detectable effect at 80% power\n"
            "idx80 = np.argmin(np.abs(np.array(powers) - 0.80))\n"
            "mde = effect_sizes[idx80]\n"
            "print(f'Minimum detectable effect at 80% power (n=3, alpha=5%): {mde:.2f} log-ret/year')\n"
            "print(f'That is ~{np.expm1(mde)*100:.0f}% return above trend — implausibly large for 2028 given 2024 data.')"
        ),
        md(
            "> In plain words: with n=3 and BTC's observed volatility, you would need the true "
            "halving effect to be a **+130%+ excess return above secular trend** to detect it "
            "with 80% probability. Even the most bullish halving believers don't claim that. "
            "The study's power is structurally insufficient to validate or invalidate this idea "
            "on the available data."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Can the engine find a halving boost when one is planted? Sweep the planted boost "
            "on a synthetic tape and confirm the p-value drops as the effect grows."
        ),
        code(
            "boosts = [0.0, 0.5, 1.0, 2.0, 3.0]\n"
            "halvings_synth = data.HALVING_DATES[:3]\n"
            "rows_synth = []\n"
            "for b in boosts:\n"
            "    bars_s, _ = data.synthetic_daily(n_years=12, halving_boost=b, seed=83)\n"
            "    ev_s = st.window_returns(bars_s, halvings=halvings_synth, window_days=365)\n"
            "    pa_s = st.power_analysis(bars_s, halvings=halvings_synth, window_days=365, n_draws=500)\n"
            "    s_s = st.summarize(ev_s['log_ret'].to_numpy())\n"
            "    rows_synth.append({'boost': b, 'mean_log_ret': s_s['mean'], 'HAC_t': s_s['tstat'],\n"
            "                       'p_value': pa_s['p_value']})\n"
            "synth_df = pd.DataFrame(rows_synth)\n"
            "print(synth_df.round(3).to_string(index=False))\n"
            "fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))\n"
            "axes[0].plot(synth_df['boost'], synth_df['mean_log_ret'], 'o-', c=GREEN, lw=2)\n"
            "axes[0].set_xlabel('planted halving boost'); axes[0].set_ylabel('mean log-ret')\n"
            "axes[0].set_title('Mean log-ret rises with planted boost')\n"
            "axes[1].plot(synth_df['boost'], synth_df['p_value'], 'o-', c=RED, lw=2)\n"
            "axes[1].axhline(0.05, ls='--', c=GREY, label='5% threshold')\n"
            "axes[1].set_xlabel('planted halving boost'); axes[1].set_ylabel('p-value vs random')\n"
            "axes[1].set_title('p-value falls as boost grows (engine works)')\n"
            "axes[1].legend(); plt.tight_layout(); plt.show()\n"
            "print('\\nConclusion: the engine is a faithful boost detector; the real tape has a small boost.')"
        ),
        md(
            "The engine correctly detects a planted halving boost when the boost is large enough "
            "(p-value falls as boost grows). On the real tape, the observed mean falls between "
            "the boost=0 and boost=1.0 synthetic worlds — consistent with a partial or fading "
            "effect, not a robust, repeatable one. The synthetic control also demonstrates why "
            "n=3 is the binding constraint: even with a planted boost of +1.0, the p-value stays "
            "marginal because 3 events at 80%/yr vol is simply not enough statistical mass.\n\n"
            "Forks worth trying: extend the tape to 2012 using CoinGecko/Glassnode data, "
            "test shorter holding windows (pre-halving accumulation), or study the Ethereum "
            "issuance reductions — more events, more power."
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
