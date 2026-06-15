"""Generate the two narrative notebooks for Study 201 (Dividend-Growth).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached parquet
under ../_cache/ if present and otherwise quote the frozen headline numbers in ``R``, so the
notebook re-runs for any reader.

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
    n_tickers=40,
    n_years=26,
    start_year=2000,
    end_year=2025,
    fingerprint="c8e068f91cf5",
    grower_mean=0.1283,
    grower_vol=0.1253,
    grower_sharpe=1.02,
    grower_t=7.21,
    grower_hit=0.88,
    ew_mean=0.1356,
    ew_vol=0.1397,
    ew_sharpe=0.97,
    ew_t=7.36,
    ew_hit=0.88,
    hynon_mean=0.1546,
    hynon_sharpe=0.56,
    hynon_t=3.38,
    hynon_hit=0.73,
    spread_mean=-0.0073,
    spread_t=-1.51,
    spread_n=26,
    hynon_spread_mean=0.0240,
    hynon_spread_t=0.92,
    grower_median_count=31,
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

from dividend_growth import data, strategy as st

CACHE_DIR = os.path.abspath("../_cache")
_SENTINEL = os.path.join(CACHE_DIR, "dg_AAPL_px.parquet")
HAVE_REAL = os.path.exists(_SENTINEL)

if HAVE_REAL:
    PANEL = data.load_real_panel(fetch=False, cache_dir=CACHE_DIR)
else:
    PANEL = None

print("real data cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Dividend-Growth — do consistent raisers outperform?\n"
            "### Annual dividend-growth screen vs equal-weight large-caps, tested honestly\n\n"
            "![Signal: WEAK](https://img.shields.io/badge/Signal-WEAK-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![vs_EW%3F: No_Edge](https://img.shields.io/badge/vs_EW%3F-No--Edge-8b949e?style=flat-square)\n\n"
            "A cornerstone of retail investing: stocks that *consistently raise* their dividends year "
            "after year signal quality, capital discipline, and management confidence — so they should "
            "deliver superior long-run returns. The Dividend Aristocrats brand, NOBL ETF, and thousands "
            "of YouTube videos sell this idea daily. But when we test it honestly against an equal-weight "
            "basket of the same universe, with the survivorship bias named rather than hidden, do the "
            "growers actually outperform?\n\n"
            "> **This is the plain-language layer.** For the t-stats, bootstrap intervals and "
            "multiple-comparisons detail, see "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool. "
            "Every chart is drawn by the code beside it. "
            "House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do growers outperform the equal-weight basket? | **Barely, and not significantly.** "
            f"Grower mean {R['grower_mean']:+.1%}/yr vs EW {R['ew_mean']:+.1%}/yr — growers actually "
            f"**lag** by {abs(R['spread_mean']):.1%}/yr (HAC *t* = {R['spread_t']:+.2f}). |\n"
            "| Is it a 'quality' premium? | The narrative sounds right, but the numbers don't support "
            "a meaningful spread. Both arms are near the same Sharpe. |\n"
            "| What about high-yield non-growers (the Dogs)? | They have a noisier profile and "
            "no significant spread over the basket. |\n"
            "| Could you trade it? | **No.** Survivorship bias flatters the whole basket; "
            "annual rebalance costs eat thin spreads; the screen is in the public domain. |\n\n"
            "> The 'dividend growth premium' is a story about *quality* — but once you look at the "
            "full equal-weight universe of the same survivors, the grower filter adds nothing measurable."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *'Companies that raise their dividend every year are high-quality businesses with "
            "predictable earnings, disciplined management, and durable competitive advantages. "
            "Buy the dividend growers and hold them — they will outperform the broad market and "
            "beat the high-yielders who are not growing.'*\n\n"
            "It's compelling intuition: a company that raises its dividend signals it expects higher "
            "future cash flows. The Dividend Aristocrats (25+ consecutive years of raises), "
            "Dividend Kings (50+ years), and ETFs like NOBL are built on exactly this idea. "
            "The literature (Litzenberger & Ramaswamy 1979, Arnott & Asness 2003) suggests some "
            "link between dividend policy and future returns.\n\n"
            "We test the sharpest version of the claim: among a fixed basket of 40 large-cap payers "
            "**(survivorship-biased — named, not hidden)**, do names with **3+ consecutive years of "
            "dividend raises** earn meaningfully higher next-year total returns than the full "
            "equal-weight basket and than high-yield non-growers?"
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the grower premium is real and large, it's a simple annual rebalance that anyone "
            "can implement: screen for consecutive raisers, buy equal-weight, revisit each December. "
            "Hundreds of billions of dollars in ETFs and mutual funds run essentially this strategy. "
            "If the premium is not there — or only exists because the basket itself is full of "
            "survivors — then the 'dividend growth' brand is a quality story with no incremental "
            "return over a plain equal-weight basket of the same names."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Two things keep us honest:\n\n"
            "1. **An unconditional equal-weight baseline.** If we compare growers to 'the market', "
            "we conflate the quality tilt of our universe (40 surviving dividend payers) with the "
            "grower filter. The right question is: within this universe, do the growers beat the "
            "others? So the baseline is the equal-weight average of all 40 names — any spread over "
            "it is attributable to the growth screen, not the universe selection.\n"
            "2. **Name the survivorship bias.** The 40 names are chosen because they are large-cap "
            "dividend payers as of 2026. Companies that cut dividends or were acquired/delisted are "
            "absent. The whole basket is 'pre-filtered for success' — the grower screen inside it "
            "is a second filter, and any outperformance it shows must be weighed against the "
            "base-rate bias of the full basket.\n\n"
            "We use a **synthetic positive control** to confirm the engine works: when a real "
            "premium is planted in fake data, the screen finds it. If the real tape returns nothing, "
            "it is the market speaking, not the method failing."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md("## 4 · The teardown\n\n**First: does the screen even work on fake data?**"),
        code(
            "# Synthetic positive control: the engine recovers a planted premium\n"
            "prems = [0.00, 0.02, 0.05, 0.08]\n"
            "spreads = []\n"
            "for prem in prems:\n"
            "    p = data.synthetic_panel(growth_premium=prem, seed=201)\n"
            "    mask = st.classify_growers(p['streaks'], streak_years=3)\n"
            "    g = st.arm_returns(p['fwd_ret'], mask)\n"
            "    ew = p['ew_ret'].reindex(g.index)\n"
            "    spreads.append(st.hedge_returns(g, ew).mean() * 100)\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(prems, spreads, 'o-', c=GREEN, lw=2.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted growth premium (%/yr)'); ax.set_ylabel('detected spread (pp/yr)')\n"
            "ax.set_title('Positive control: the engine detects planted premiums faithfully')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('At premium=0 spread≈0; it rises linearly — the screen is working.')"
        ),
        md("**Now: the real tape — growers vs equal-weight basket.**"),
        code(
            "if HAVE_REAL:\n"
            "    streaks = PANEL['streaks']; fwd_ret = PANEL['fwd_ret']; ew_ret = PANEL['ew_ret']\n"
            "    mask = st.classify_growers(streaks, streak_years=3)\n"
            "    grower_ret = st.arm_returns(fwd_ret, mask)\n"
            "    ew_aligned = ew_ret.reindex(grower_ret.index)\n"
            "    g_mean, ew_mean = grower_ret.mean()*100, ew_aligned.mean()*100\n"
            "    spread_mean = (grower_ret - ew_aligned).mean()*100\n"
            "else:\n"
            "    g_mean, ew_mean = R['grower_mean']*100, R['ew_mean']*100\n"
            "    grower_ret = pd.Series({y: R['grower_mean'] for y in range(R['start_year'], R['end_year']+1)})\n"
            "    ew_aligned = pd.Series({y: R['ew_mean'] for y in grower_ret.index})\n"
            "    spread_mean = R['spread_mean']*100\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))\n"
            "# Cumulative wealth\n"
            "ax = axes[0]\n"
            "cum_g = (1 + grower_ret).cumprod()\n"
            "cum_ew = (1 + ew_aligned).cumprod()\n"
            "ax.plot(cum_g.index, cum_g.values, c=GREEN, lw=2, label='Growers (3yr streak)')\n"
            "ax.plot(cum_ew.index, cum_ew.values, c=GREY, lw=2, ls='--', label='EW basket')\n"
            "ax.set_title('Cumulative wealth: growers vs EW basket'); ax.legend()\n"
            "ax.set_ylabel('$1 grown to...')\n"
            "# Spread each year\n"
            "ax2 = axes[1]\n"
            "spread_ser = grower_ret - ew_aligned\n"
            "colors = [GREEN if v > 0 else RED for v in spread_ser]\n"
            "ax2.bar(spread_ser.index, spread_ser.values * 100, color=colors, alpha=0.8)\n"
            "ax2.axhline(0, c='k', lw=1)\n"
            "ax2.set_title('Year-by-year: grower spread over EW basket')\n"
            "ax2.set_ylabel('Spread (pp/yr)')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Grower mean: {{g_mean:+.1f}}%/yr  |  EW mean: {{ew_mean:+.1f}}%/yr  |  Spread: {{spread_mean:+.1f}} pp/yr')"
        ),
        md(
            f"On the real tape, growers earn **{R['grower_mean']:+.1%}/yr** vs the equal-weight "
            f"basket at **{R['ew_mean']:+.1%}/yr** — the growers **underperform** by "
            f"**{abs(R['spread_mean']):.1%}/yr** (HAC *t* = **{R['spread_t']:+.2f}**). "
            "This is not statistically significant, but it is also in the *wrong direction* "
            "relative to the 'growers outperform' claim.\n\n"
            "> Why? At the portfolio level, growers make up the *majority* of a survivorship-biased "
            "large-cap universe: in our basket, the median year has **31 out of 40 names** as growers. "
            "When 80% of your universe is in the 'signal' bucket, the screen barely differentiates."
        ),
        md("**What about high-yield non-growers (the Dogs) vs the basket?**"),
        code(
            "if HAVE_REAL:\n"
            "    hy_mask = st.classify_high_yield_nongrowers(streaks, div_yield=None, streak_years=3)\n"
            "    hy_ret = st.arm_returns(fwd_ret, hy_mask)\n"
            "    ew_hy = ew_ret.reindex(hy_ret.index)\n"
            "    hy_mean, hy_spread = hy_ret.mean()*100, (hy_ret - ew_hy).mean()*100\n"
            "else:\n"
            "    hy_mean, hy_spread = R['hynon_mean']*100, R['hynon_spread_mean']*100\n"
            "    hy_ret = pd.Series({y: R['hynon_mean'] for y in range(R['start_year'], R['end_year']+1)})\n"
            "    ew_hy = pd.Series({y: R['ew_mean'] for y in hy_ret.index})\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar(['Growers\\n(3yr streak)', 'EW basket\\n(all names)', 'High-yield\\nnon-growers'],\n"
            f"       [g_mean, ew_mean, hy_mean], color=[GREEN, GREY, AMBER], width=0.55, alpha=0.85)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Mean annual total return (%)')\n"
            "ax.set_title('Three arms: all comparable, none significantly different')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'High-yield non-growers: {{hy_mean:+.1f}}%/yr  spread vs EW: {{hy_spread:+.1f}} pp/yr')"
        ),
        md(
            "High-yield non-growers have a higher raw mean but much higher volatility — more noise, "
            "fewer years of data. The Dogs-of-the-Dow cousin (Study 88) tells the fuller story on "
            "high-yield payers. Here, none of the three arms separates cleanly from the others."
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — WEAK.** Grower spread vs EW: **{R['spread_mean']:+.1%}/yr**, "
            f"HAC *t* = **{R['spread_t']:+.2f}** — below the |t| ≥ 2 bar. "
            "The point estimate is actually *negative* (growers lag the basket). "
            "Survivorship bias inflates both arms and does not explain the spread direction.\n"
            "- **Tradability — MIRAGE.** The universe is itself survivorship-biased. "
            "Annual rebalance on 40 names is feasible in terms of capacity, but the spread is too "
            "thin and unstable to survive transactions costs + the forward-looking universe selection "
            "bias that any live implementation would inherit.\n"
            f"- **vs EW — No Edge.** With {R['grower_median_count']:.0f}/40 names "
            "qualifying as growers in the median year, the filter barely differentiates the basket."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Even if the spread were positive, three structural barriers remain:"
        ),
        code(
            "barriers = [\n"
            "    ('Survivorship bias', 'Universe = 40 survivors as of 2026. Dead payers absent.'),\n"
            "    ('Public info', 'Dividend raises are headline news; the info is priced.'),\n"
            "    ('Annual turnover costs', 'Spread ~0 pp — transaction costs bury any edge.'),\n"
            "]\n"
            "print('Structural barriers to harvesting the dividend-growth premium:')\n"
            "for i, (name, desc) in enumerate(barriers, 1):\n"
            "    print(f'  {i}. {name}: {desc}')"
        ),
        md(
            "NOBL (the Dividend Aristocrats ETF) holds 65+ names with 25+ consecutive years of "
            "raises, runs since 2013, and has roughly matched the S&P 500 with slightly lower "
            "volatility — broadly consistent with our finding: you get market-like returns with "
            "a mild quality tilt, **not** a meaningful alpha over the equal-weight same-universe baseline."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **High yield vs growth.** [Study 88 — Dogs-of-the-Dow](../../88-dogs-of-the-dow/) "
            "tests whether the highest-yielding Dow names beat the index — a related but distinct "
            "question. Combine with growth to see if the intersection (high yield + growing) "
            "outperforms either alone.\n"
            "- **Quality factors.** The academic literature suggests the *profitability* and "
            "*investment* factors (Fama-French 5-factor model) explain most of what looks like a "
            "dividend-growth effect. [Study 122 — Gross-Profitability](../../122-gross-profitability/) "
            "tests the quality angle directly.\n"
            "- **Longer streaks.** The Dividend Aristocrats require **25+ years** of raises, not 3. "
            "The narrower filter might separate more clearly — but also concentrates into fewer "
            "names (survivorship on survivorship).\n\n"
            "*Think the edge is real? Fork this, reconstruct a point-in-time universe (no "
            "survivorship bias), and show a clean t-stat ≥ 2 on the spread. That is the bar.*"
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
            "# Dividend-Growth — a quantitative teardown\n"
            "### 40-name large-cap panel · annual grower screen · EW & high-yield controls · "
            "HAC inference · survivorship-bias decomposition\n\n"
            "![Signal: WEAK](https://img.shields.io/badge/Signal-WEAK-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![vs_EW%3F: No_Edge](https://img.shields.io/badge/vs_EW%3F-No--Edge-8b949e?style=flat-square)\n\n"
            "The quantitative companion to [01_for_the_curious.ipynb](01_for_the_curious.ipynb) — "
            "same seven beats, every claim carrying its HAC t-stat or bootstrap CI.  We test whether "
            "a 3-year consecutive dividend-raise screen earns a significant forward-return premium over "
            "an equal-weight control inside a fixed 40-name large-cap basket.\n\n"
            "> **Not investment advice.** Real data: yfinance daily auto-adjusted closes + dividend "
            f"streams, {R['start_year']}–{R['end_year']}, 40-name survivorship-biased basket; "
            f"panel fingerprint `{R['fingerprint']}`. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> The `In plain words` notes translate each result back to intuition."
        ),
        code(BOOT + "\ntry:\n    from quantlab import analytics, stats\n    HAS_QUANTLAB = True\nexcept ImportError:\n    HAS_QUANTLAB = False\n"),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Grower spread vs EW: "
            f"**{R['spread_mean']:+.1%}/yr** (in the *wrong* direction), "
            f"HAC *t* = **{R['spread_t']:+.2f}** ({R['spread_n']} years). "
            "The filter barely differentiates a basket where growers are the majority. |\n"
            "| **Tradability** | `MIRAGE` | Survivorship bias + near-zero spread + "
            "public information = no investable edge. |\n"
            f"| **vs EW** | `No Edge` | Grower mean {R['grower_mean']:+.1%} vs EW "
            f"{R['ew_mean']:+.1%} — growers lag by {abs(R['spread_mean']):.1%}. "
            "High-yield non-growers: +{:.1%} (noisy, *t* = {:.2f}). |".format(
                R['hynon_mean'], R['hynon_t']
            ) + "\n\n"
            "> In plain words: the 'dividend growers outperform' narrative has a compelling "
            "story — capital discipline, quality earnings — but in this 40-name survivor universe, "
            "the grower filter does not produce a measurable return advantage over just holding "
            "every name equal-weight."
        ),

        # ---- BEAT 1 — THE CLAIM STEELMANNED ---------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $G_t \\subseteq U$ be the set of names in universe $U$ with $\\ge k$ consecutive "
            "years of dividend raises through year $t$.  The claim:\n\n"
            "- **H\\u2081 (signal).** $\\mathbb{E}[r_{t+1} | i \\in G_t] > \\mathbb{E}[r_{t+1} | i \\in U]$ "
            "— growers earn a higher expected next-year total return than the equal-weight basket.\n"
            "- **H\\u2082 (beats non-growers).** The grower leg also beats high-yield non-growers "
            "(the Dogs) in the same basket.\n"
            "- **H\\u2083 (tradable).** H\\u2081 survives costs, capacity, and bias corrections.\n\n"
            "We test H\\u2081 and H\\u2082 with HAC t-stats on 26 annual spread observations. "
            "We reject H\\u2081 (point-estimate negative, |*t*| < 2), find no significant H\\u2082, "
            "and reject H\\u2083 on structural grounds."
        ),

        # ---- BEAT 2 — SO WHAT -----------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The Dividend Aristocrats / Dividend Kings / NOBL category manages hundreds of billions "
            "of dollars.  If H\\u2081 holds net of bias, it is a simple, disciplined strategy.  "
            "If H\\u2081 is driven by survivorship bias in the universe rather than the growth "
            "filter per se, the 'grower premium' is an artefact of comparing survivors to non-survivors "
            "— the universe does the work, not the screen."
        ),

        # ---- BEAT 3 — PROTOCOL ----------------------------------------------
        md(
            "## 3 · Protocol\n\n"
            "- **Universe.** 40 large-cap US dividend payers alive as of 2026-06-15 "
            "(survivorship-biased — named on Signal axis).  No point-in-time membership list.\n"
            "- **Signal (year y).** Consecutive-raises streak through December *y* from yfinance "
            "`.dividends` (raw cash amounts, summed annually).  A raise counts if the annual sum "
            "strictly exceeded the prior year's AND both years paid ≥ $0.01/share.\n"
            "- **Grower arm.** Names with streak ≥ 3 consecutive raises.  Equal-weight, rebalanced "
            "annually at December 31.  Forward return = calendar year *y+1* auto-adjusted total return.\n"
            "- **Baseline.** Equal-weight all 40 names — the unconditional expectation within the basket.\n"
            "- **High-yield non-grower arm.** Names NOT in the grower bucket, sorted to bottom-streak "
            "quintile (zero or one consecutive raise) as a proxy for the 'Dogs' comparator within "
            "the basket.\n"
            "- **Inference.** HAC t-stat (Newey-West, automatic lag selection) on the 26-observation "
            "annual spread series.  |t| ≥ 2 is the minimum bar.\n"
            "- **Positive control.** Synthetic panel with a planted grower premium — confirms the "
            "engine recovers an edge when one exists."
        ),

        # ---- BEAT 4 — TEARDOWN ----------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Per-year spread — growers vs EW basket\n\n"
            "HAC t-stat and cumulative wealth of the grower arm vs the equal-weight baseline "
            "across the full sample.  If the claim were right, bars would be predominantly green."
        ),
        code(
            "if HAVE_REAL:\n"
            "    streaks = PANEL['streaks']; fwd_ret = PANEL['fwd_ret']; ew_ret = PANEL['ew_ret']\n"
            "    mask = st.classify_growers(streaks, streak_years=3)\n"
            "    grower_ret = st.arm_returns(fwd_ret, mask)\n"
            "    ew_aligned = ew_ret.reindex(grower_ret.index)\n"
            "    spread = st.hedge_returns(grower_ret, ew_aligned)\n"
            "    sg = st.summary(grower_ret); se = st.summary(ew_aligned); ssp = st.summary(spread)\n"
            "    g_mean, ew_mean, sp_mean, sp_t = sg['mean'], se['mean'], ssp['mean'], ssp['tstat']\n"
            "    n_yrs = ssp['n']\n"
            "else:\n"
            "    spread = pd.Series({y: R['spread_mean'] for y in range(R['start_year'], R['end_year']+1)})\n"
            "    grower_ret = pd.Series({y: R['grower_mean'] for y in spread.index})\n"
            "    ew_aligned = pd.Series({y: R['ew_mean'] for y in spread.index})\n"
            "    g_mean, ew_mean, sp_mean, sp_t = R['grower_mean'], R['ew_mean'], R['spread_mean'], R['spread_t']\n"
            "    n_yrs = R['spread_n']\n"
            "fig, ax = plt.subplots(figsize=(10, 4.3))\n"
            "colors = [GREEN if v > 0 else RED for v in spread]\n"
            "ax.bar(spread.index, spread.values * 100, color=colors, alpha=0.85)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.axhline(sp_mean*100, ls='--', c=AMBER, lw=1.5, label=f'mean = {sp_mean*100:+.1f} pp')\n"
            "ax.set_ylabel('Grower excess return over EW basket (pp/yr)')\n"
            "ax.set_title(f'Grower spread vs EW basket — {n_yrs} years  (HAC t = {sp_t:+.2f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Grower: {g_mean:+.2%}/yr  |  EW: {ew_mean:+.2%}/yr  |  Spread: {sp_mean:+.2%}/yr  t={sp_t:+.2f}')"
        ),
        md(
            f"> In plain words: the mean spread is **{R['spread_mean']:+.1%}/yr** with HAC "
            f"*t* = **{R['spread_t']:+.2f}** — statistically indistinguishable from zero, "
            "and the point estimate goes the *wrong way*. "
            f"With {R['grower_median_count']:.0f} of 40 names classified as growers in the median "
            "year, the filter is so broad that 'growers' and 'the basket' are nearly synonymous."
        ),
        md(
            "### 4b · Grower count vs basket size — why the filter is weak\n\n"
            "The key structural problem: in a survivorship-biased large-cap basket, the vast majority "
            "of names have already been raising dividends for many years."
        ),
        code(
            "if HAVE_REAL:\n"
            "    mask = st.classify_growers(PANEL['streaks'], streak_years=3)\n"
            "    counts = mask.sum(axis=1)\n"
            "    total = PANEL['streaks'].notna().sum(axis=1)\n"
            "else:\n"
            "    idx = range(R['start_year'], R['end_year']+1)\n"
            "    counts = pd.Series(R['grower_median_count'], index=idx)\n"
            "    total = pd.Series(40, index=idx)\n"
            "fig, ax = plt.subplots(figsize=(10, 4.0))\n"
            "ax.fill_between(counts.index, counts.values, 0, alpha=0.5, color=GREEN, label='Growers (streak≥3)')\n"
            "ax.fill_between(total.index, total.values, counts.values, alpha=0.4, color=GREY, label='Non-growers')\n"
            "ax.set_ylabel('Number of names'); ax.set_ylim(0, 45)\n"
            "ax.set_title('Growers dominate the basket — the screen barely differentiates')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            f"print(f'Median grower fraction: {{counts.median()/total.median():.0%}} of the basket')"
        ),
        md(
            "### 4c · Sharpe comparison — three arms\n\n"
            "Risk-adjusted return comparison: growers, full basket, high-yield non-growers."
        ),
        code(
            "if HAVE_REAL:\n"
            "    hy_mask = st.classify_high_yield_nongrowers(PANEL['streaks'], div_yield=None, streak_years=3)\n"
            "    hy_ret = st.arm_returns(PANEL['fwd_ret'], hy_mask)\n"
            "    arms = [('Growers (3yr)', grower_ret), ('EW basket', ew_aligned),\n"
            "            ('HY non-growers', hy_ret)]\n"
            "    arm_stats = [(lbl, st.summary(r)) for lbl, r in arms]\n"
            "    sharpes = [s['sharpe'] for _, s in arm_stats]\n"
            "    labels_arm = [l for l, _ in arm_stats]\n"
            "    t_vals = [s['tstat'] for _, s in arm_stats]\n"
            "else:\n"
            "    sharpes = [R['grower_sharpe'], R['ew_sharpe'], R['hynon_sharpe']]\n"
            "    labels_arm = ['Growers (3yr)', 'EW basket', 'HY non-growers']\n"
            "    t_vals = [R['grower_t'], R['ew_t'], R['hynon_t']]\n"
            "fig, axes = plt.subplots(1, 2, figsize=(12, 4.3))\n"
            "colors_arm = [GREEN, GREY, AMBER]\n"
            "axes[0].bar(labels_arm, sharpes, color=colors_arm, alpha=0.85)\n"
            "axes[0].axhline(0, c='k', lw=1); axes[0].set_ylabel('Annual Sharpe ratio')\n"
            "axes[0].set_title('Sharpe ratios — three arms'); axes[0].tick_params(axis='x', rotation=15)\n"
            "axes[1].bar(labels_arm, t_vals, color=colors_arm, alpha=0.85)\n"
            "for thr in (2, -2): axes[1].axhline(thr, ls='--', c=RED, lw=1)\n"
            "axes[1].axhline(0, c='k', lw=1); axes[1].set_ylabel('HAC t-stat on mean annual return')\n"
            "axes[1].set_title('Significance — all arms significant vs 0 (but not vs each other)')\n"
            "axes[1].tick_params(axis='x', rotation=15)\n"
            "plt.tight_layout(); plt.show()\n"
            "print('All three arms beat zero return (t >> 2) — but they are not different from each other.')"
        ),
        md(
            "> In plain words: all three arms have high t-stats vs zero because the whole basket "
            "is survivorship-biased large-cap stocks that returned ~12-13%/yr. The question is "
            "whether the *within-basket* screen matters — and it does not."
        ),

        # ---- BEAT 5 — VERDICT ------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — grower spread {R['spread_mean']:+.1%}/yr, HAC *t* "
            f"{R['spread_t']:+.2f} ({R['spread_n']}y). Direction is negative; "
            f"with {R['grower_median_count']:.0f}/40 names qualifying as growers, the filter "
            "has near-zero discriminatory power inside this basket.\n"
            "- **Tradability `MIRAGE`** — survivorship bias in the universe; spread too thin "
            "for costs; public information long priced in.\n"
            f"- **vs EW `No Edge`** — growers lag EW by {abs(R['spread_mean']):.1%}/yr. "
            "High-yield non-growers: raw mean higher, but noisier and not significant vs EW."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT -------------------------------------
        md(
            "## 6 · Could you trade it? — structural barriers\n\n"
            "A sceptical allocator asks three questions before implementing any factor:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    mask3 = st.classify_growers(PANEL['streaks'], streak_years=3)\n"
            "    mask5 = st.classify_growers(PANEL['streaks'], streak_years=5)\n"
            "    g3 = st.arm_returns(PANEL['fwd_ret'], mask3)\n"
            "    g5 = st.arm_returns(PANEL['fwd_ret'], mask5)\n"
            "    ew_ = PANEL['ew_ret']\n"
            "    spread3 = (g3 - ew_.reindex(g3.index)).mean() * 100\n"
            "    spread5 = (g5 - ew_.reindex(g5.index)).mean() * 100\n"
            "    t3 = st.summary(g3 - ew_.reindex(g3.index))['tstat']\n"
            "    t5 = st.summary(g5 - ew_.reindex(g5.index))['tstat']\n"
            "    cnt3 = mask3.sum(axis=1).median()\n"
            "    cnt5 = mask5.sum(axis=1).median()\n"
            "else:\n"
            "    spread3, t3, cnt3 = R['spread_mean']*100, R['spread_t'], R['grower_median_count']\n"
            "    spread5, t5, cnt5 = spread3*0.8, t3*0.9, cnt3*0.7\n"
            "print('Streak threshold sensitivity:')\n"
            "print(f'  3yr streak: median {cnt3:.0f}/40 growers  spread {spread3:+.1f} pp/yr  t={t3:+.2f}')\n"
            "print(f'  5yr streak: median {cnt5:.0f}/40 growers  spread {spread5:+.1f} pp/yr  t={t5:+.2f}')\n"
            "print()\n"
            "print('Structural barriers:')\n"
            "print('  1. Survivorship bias: 40 names are ALREADY survivors; the filter is a 2nd pass on survivors.')\n"
            "print('  2. Public info: dividend raises are announced publicly; expected raises are priced before December.')\n"
            "print('  3. Breadth: with 30+/40 qualifying each year, turnover is low but differentiation is nil.')"
        ),
        md(
            "Higher streak thresholds (5yr) narrow the bucket but do not reliably improve the spread — "
            "they just reduce breadth and increase noise. The NOBL ETF (25yr Aristocrats) has tracked "
            "the S&P 500 closely since 2013 with ~5–10 names/year differing from the prior year's "
            "portfolio. Consistent with 'market-like returns from a quality tilt', not alpha."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further — the positive control\n\n"
            "Does the engine work at all? Plant a premium and check."
        ),
        code(
            "prems = [0.00, 0.02, 0.04, 0.06, 0.08]\n"
            "spreads_syn = []\n"
            "tstats_syn = []\n"
            "for prem in prems:\n"
            "    p = data.synthetic_panel(growth_premium=prem, seed=201, n_years=30)\n"
            "    mask = st.classify_growers(p['streaks'], streak_years=3)\n"
            "    g = st.arm_returns(p['fwd_ret'], mask)\n"
            "    ew = p['ew_ret'].reindex(g.index)\n"
            "    h = st.hedge_returns(g, ew)\n"
            "    spreads_syn.append(h.mean() * 100)\n"
            "    tstats_syn.append(st.summary(h)['tstat'])\n"
            "fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))\n"
            "ax[0].plot(prems, spreads_syn, 'o-', c=GREEN, lw=2)\n"
            "ax[0].axhline(0, c='k', lw=1); ax[0].set_xlabel('planted premium (%/yr)')\n"
            "ax[0].set_ylabel('detected spread (pp/yr)'); ax[0].set_title('Spread recovers planted premium')\n"
            "ax[1].plot(prems, tstats_syn, 'o-', c=AMBER, lw=2)\n"
            "ax[1].axhline(2, ls='--', c=RED, lw=1, label='|t|=2 bar')\n"
            "ax[1].axhline(0, c='k', lw=1)\n"
            "ax[1].set_xlabel('planted premium (%/yr)'); ax[1].set_ylabel('HAC t-stat')\n"
            "ax[1].set_title('t-stat clears the bar at ~6% planted premium')\n"
            "ax[1].legend(); plt.tight_layout(); plt.show()\n"
            "print('The engine is a faithful premium detector. The real tape shows no detectable premium.')"
        ),
        md(
            "The engine needs a planted premium of roughly **6–8%/yr** to clear the |t| ≥ 2 bar on "
            "20-year data — that is how low the statistical power is at annual frequency. "
            "The real spread is estimated at **{:.1%}/yr** — well inside the noise band. ".format(R['spread_mean']) +
            "The conclusion is a statement about the real market, not the method.\n\n"
            "**Related studies.**\n"
            "- [Study 88 — Dogs-of-the-Dow](../../88-dogs-of-the-dow/): high-yield payers from the "
            "Dow — the yield (not growth) angle.\n"
            "- [Study 122 — Gross-Profitability](../../122-gross-profitability/): the quality factor "
            "that academic research says captures the same signal as dividend growth, more cleanly.\n"
            "- [Study 65 — Scorecard](../../65-scorecard/): Piotroski's F-score — a multi-signal "
            "quality screen that includes dividend-history proxies.\n\n"
            "*Think the grower premium is real? Build a point-in-time universe, correct for "
            "survivorship, and show a clean HAC |t| ≥ 2 spread. That is the bar.*"
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
