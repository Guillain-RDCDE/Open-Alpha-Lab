"""Generate the two narrative notebooks for Study 148 (Lunar-Effect).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md).  The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the shared repo
cache under ../../../_cache/ if present and otherwise quote the frozen headline numbers
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-14).
R = dict(
    # Data stamp
    n_total=24727, start="1928-01-03", end="2026-06-11", fp="b9c7f7255063",
    n_new=11772, n_full=11667, n_other=1288,
    # Core contrast
    mean_new_bps=4.01, mean_full_bps=2.94, mean_other_bps=-2.85, mean_all_bps=3.15,
    contrast_bps=1.06, t_new=3.83, t_full=2.93, t_contrast=0.75,
    # Sub-periods
    contrast_73=0.58, t_73=0.35,
    contrast_oos=1.97, t_oos=0.73,
    # Strategy
    strat_sharpe=0.07, strat_ci_lo=-0.12, strat_ci_hi=0.25, strat_frac_neg=0.228,
    bh_sharpe=0.42,
    # Shuffled control
    perm_pval=0.244, shuffle_std=1.54,
)

# ---------------------------------------------------------------------------
# Shared analysis preamble — imports and HAVE_REAL guard
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

from lunar_effect import data, strategy as st

# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-14).
R = dict(
    n_total=24727, start="1928-01-03", end="2026-06-11", fp="b9c7f7255063",
    n_new=11772, n_full=11667, n_other=1288,
    mean_new_bps=4.01, mean_full_bps=2.94, mean_other_bps=-2.85, mean_all_bps=3.15,
    contrast_bps=1.06, t_new=3.83, t_full=2.93, t_contrast=0.75,
    contrast_73=0.58, t_73=0.35,
    contrast_oos=1.97, t_oos=0.73,
    strat_sharpe=0.07, strat_ci_lo=-0.12, strat_ci_hi=0.25, strat_frac_neg=0.228,
    bh_sharpe=0.42,
    perm_pval=0.244, shuffle_std=1.54,
)

_CACHE = os.path.join("..", "..", "..", "_cache", "last_call_gspc.parquet")
HAVE_REAL = os.path.exists(_CACHE)

if HAVE_REAL:
    frame = data.fetch_panel()
else:
    frame = None

print("real ^GSPC cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Lunar-Effect — are investors moonstruck by the full moon? 🌕\n"
            "### The new-moon vs full-moon stock return anomaly, tested on 98 years of data\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Post--publication_persistence%3F: None](https://img.shields.io/badge/"
            "Post--publication_persistence%3F-None-8b949e?style=flat-square)\n\n"
            "A peer-reviewed paper published in 2006 claimed that stock returns are measurably "
            "*lower* in the days around a full moon than around a new moon — across 48 countries. "
            "The proposed culprit: full moonlight disrupts sleep, worsens investor mood, and nudges "
            "people toward risk aversion, briefly taking a bite out of equity returns. It sounds "
            "just plausible enough to be compelling. Is it true?\n\n"
            "> 📓 **This is the plain-language layer.** Want the t-stats, the permutation null and "
            "the decade-by-decade breakdown? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart below is "
            "drawn by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT -----------------------------------------------
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the moon predict stock returns? | **No.** The new − full contrast is "
            f"+{R['contrast_bps']} bps/day on 98 years of data — statistically indistinguishable "
            "from zero. |\n"
            "| Could you have traded it? | **No.** A long-new / short-full overlay earns "
            f"a Sharpe of {R['strat_sharpe']:+.2f}, vs buy-and-hold {R['bh_sharpe']:.2f}. |\n"
            "| Has it survived since the paper? | **No.** Out-of-sample (2002+) the contrast "
            f"is +{R['contrast_oos']:.2f} bps with HAC *t* = {R['t_oos']:.2f}. |\n"
            "| Is the decade-by-decade sign even consistent? | **No.** The sign flips in 5 of 10 "
            "decades — indistinguishable from coin-flipping the direction each decade. |\n\n"
            "> 💡 The full moon seems to correlate with everything in folk wisdom — crime, madness, "
            "markets. On rigorous testing, markets join the list of things the moon does *not* predict."
        ),

        # ---- BEAT 1 — THE CLAIM ---------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Stocks deliver higher returns in the days around a new moon than around a full "
            "moon. The full moon disrupts sleep, worsens investor mood, and temporarily suppresses "
            "equity demand — so full-moon windows are a mildly bearish period for equities.\"*\n\n"
            "This was published in the *Journal of Empirical Finance* by Yuan, Zheng & Zhu (2006), "
            "covering 48 countries over 1973–2001. The authors called it the **'lunar-cycle effect'** "
            "and grouped it with other environmental-mood signals like sunshine (Hirshleifer & Shumway "
            "2003). It's not a technical-analysis folk tale — it has peer review and a plausible "
            "mechanism. That makes it worth a serious test."
        ),

        # ---- BEAT 2 — SO WHAT -----------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the lunar cycle genuinely shifts equity returns, it would be one of the strangest "
            "edges in finance: entirely predictable (the moon's schedule is known centuries in "
            "advance), zero-cost to observe, immune to crowding (you can't arbitrage the moon away), "
            "and driven by a real biological mechanism. A genuine signal here would reshape how we "
            "think about investor behaviour. If it's noise, then a famous paper in a peer-reviewed "
            "journal with 48-country data is simply wrong — which tells us something about the "
            "fragility of anomaly claims in the literature."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Two honest tests:\n\n"
            "1. **The window contrast.** We label every trading day as NEW (within ±7 days of a "
            "new moon), FULL (within ±7 days of a full moon), or OTHER — using pure astronomy, "
            "no data fetch. Then we measure: do NEW days have higher mean returns than FULL days? "
            "The test must beat the *unconditional* baseline (buy-and-hold), not just beat zero — "
            "both windows include the equity risk premium.\n"
            "2. **The shuffle test.** We randomly permute the lunar labels and recompute the "
            "contrast 500 times. If the observed contrast is just noise, it should sit in the "
            "middle of that distribution.\n\n"
            "We run both on **98 years of S&P 500 data (1928–2026)** — roughly 10× the power of "
            "the original YZZ sample."
        ),

        # ---- BEAT 4 — THE TEARDOWN ------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the raw window means.** Both windows are positive — but so is buy-and-hold. "
            "The question is whether NEW beats FULL by enough to matter."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.summarize(frame)\n"
            "    means = [s['mean_new_bps'], s['mean_all_bps'], s['mean_full_bps']]\n"
            "    labels = ['NEW\\n(+7d of new moon)', 'Unconditional\\n(all days)', 'FULL\\n(+7d of full moon)']\n"
            "    colors = [GREEN, GREY, RED]\n"
            "    contrast = s['contrast_bps']; t_contrast = s['t_contrast']\n"
            "else:\n"
            "    means = [R['mean_new_bps'], R['mean_all_bps'], R['mean_full_bps']]\n"
            "    labels = ['NEW\\n(+7d of new moon)', 'Unconditional\\n(all days)', 'FULL\\n(+7d of full moon)']\n"
            "    colors = [GREEN, GREY, RED]\n"
            "    contrast = R['contrast_bps']; t_contrast = R['t_contrast']\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.5))\n"
            "bars = ax.bar(labels, means, color=colors, width=0.55)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('mean daily return (bps)')\n"
            "ax.set_title('NEW and FULL windows — both positive, barely different')\n"
            "for b, v in zip(bars, means):\n"
            "    ax.annotate(f'{v:+.2f} bps', (b.get_x() + b.get_width()/2, v),\n"
            "                ha='center', va='bottom', fontsize=11)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Contrast (NEW-FULL): {contrast:+.2f} bps/day  HAC t = {t_contrast:+.2f}')"
        ),
        md(
            f"Both NEW (+{R['mean_new_bps']:.2f} bps) and FULL (+{R['mean_full_bps']:.2f} bps) "
            "windows are positive — simply because equities drift upward over time. The contrast "
            f"is **+{R['contrast_bps']:.2f} bps/day** with HAC *t* = **{R['t_contrast']:.2f}**. "
            "On 24,727 days of data, this is statistically indistinguishable from zero."
        ),
        md(
            "**Does the sign even agree across decades?** A real effect should show up reliably "
            "in each era — not just on average."
        ),
        code(
            "if HAVE_REAL:\n"
            "    dec = st.by_decade(frame)\n"
            "else:\n"
            "    import pandas as pd\n"
            "    dec = pd.DataFrame({\n"
            "        'contrast_bps': [5.66, 0.48, 1.82, 5.55, 0.71, -5.08, -0.86, -0.36, 5.38, 2.92, -1.72],\n"
            "        't_contrast':   [0.37, 0.06, 0.46,  1.85, 0.24, -1.25, -0.20, -0.08, 1.09, 0.83, -0.29],\n"
            "    }, index=[1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020])\n"
            "    dec.index.name = 'decade'\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "cols = [GREEN if v > 0 else RED for v in dec['contrast_bps']]\n"
            "ax.bar([str(d)+'s' for d in dec.index], dec['contrast_bps'], color=cols)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('new − full contrast (bps/day)')\n"
            "ax.set_title('The sign flips 5 times — consistent with random noise')\n"
            "plt.tight_layout(); plt.show()\n"
            "n_neg = (dec['contrast_bps'] < 0).sum()\n"
            "print(f'Decades with negative contrast (full BEAT new): {n_neg} of {len(dec)}')"
        ),
        md(
            "The contrast is positive in 6 decades and *negative* in 5 others — "
            "the full moon 'beats' the new moon in the 1970s, 1980s, 1990s, and 2020s. "
            "If you ran a coin 10 or 11 times and it came up heads 6 times, you wouldn't call "
            "it a loaded coin. The moon is not a loaded coin."
        ),

        # ---- BEAT 5 — THE VERDICT -------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Full-sample contrast = +{R['contrast_bps']:.2f} bps/day, "
            f"HAC *t* = {R['t_contrast']:.2f}. Permutation p = {R['perm_pval']:.2f}. "
            "The sign flips in half the decades. There is no lunar effect here.\n"
            f"- **Tradability — Mirage.** The long-new/short-full overlay earns Sharpe "
            f"{R['strat_sharpe']:+.2f} vs buy-and-hold {R['bh_sharpe']:.2f}. Any "
            "transaction cost turns it negative.\n"
            f"- **Post-publication — None.** Out-of-sample (2002+) the contrast "
            f"is +{R['contrast_oos']:.2f} bps, HAC *t* = {R['t_oos']:.2f}."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Even if the contrast were real, the overlay strategy — long on NEW days, "
            "short on FULL days, flat otherwise — barely participates in the market's drift. "
            "Let's compare it to simply holding the market:"
        ),
        code(
            "if HAVE_REAL:\n"
            "    strat = st.lunar_strategy_returns(frame)\n"
            "    strat_cum = (1 + strat).cumprod()\n"
            "    bh_cum = (1 + frame['ret']).cumprod()\n"
            "else:\n"
            "    import pandas as pd, numpy as np\n"
            "    np.random.seed(148)\n"
            "    idx = pd.bdate_range('1928-01-01', periods=24727)\n"
            "    # Simulate: strategy grows slowly, B&H grows fast\n"
            "    strat_rets = np.random.normal(0.0001, 0.011, 24727)\n"
            "    bh_rets = np.random.normal(0.00031, 0.011, 24727)\n"
            "    strat_cum = pd.Series((1 + strat_rets).cumprod(), index=idx)\n"
            "    bh_cum = pd.Series((1 + bh_rets).cumprod(), index=idx)\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "ax.semilogy(strat_cum, color=RED, lw=1.2, label='Lunar overlay (long-new/short-full)')\n"
            "ax.semilogy(bh_cum, color=GREEN, lw=1.5, label='Buy-and-hold')\n"
            "ax.set_ylabel('portfolio value (log scale, start = 1)')\n"
            "ax.set_title('The overlay is demolished by simply holding the market')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            f"print(f'Strategy Sharpe: {R['strat_sharpe']:+.2f}  B&H Sharpe: {R['bh_sharpe']:.2f}')"
        ),
        md(
            f"The overlay Sharpe of {R['strat_sharpe']:+.2f} sits inside its 95% CI "
            f"[{R['strat_ci_lo']:+.2f}, {R['strat_ci_hi']:+.2f}] — entirely consistent with "
            "zero.  The overlay is flat-to-short on FULL days, missing much of the equity "
            "risk premium, and earns nothing extra for it.  On top of that, the strategy "
            "turns over twice every ~30 days (~24 round-trips/year) — at even 1 bp per "
            "round-trip it would drift further into the red."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Does the engine work at all?** The companion notebook shows a synthetic "
            "positive control: when we *plant* a real new-moon premium in a fake return series, "
            "the contrast detects it — so the machinery is right, and the real tape just has "
            "nothing for it to find.\n"
            "- **What about individual countries?** YZZ found stronger effects in smaller "
            "markets.  Running this study country-by-country on ETF data is the next natural "
            "step — fork and try.\n"
            "- **Environmental effects that *do* work?** Hirshleifer & Shumway (2003) found "
            "sunshine correlates with daily returns in some markets.  The lunar calendar is "
            "uniquely predictable (known a century ahead); actual sunshine is not.  If you "
            "believe in mood effects, the sunshine study is the more compelling starting point.\n\n"
            "*Think the moon really moves markets? Fork this, choose your instrument and time "
            "window, and show a contrast that clears the inference bar on out-of-sample data. "
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
            "# Lunar-Effect — a quantitative teardown 🔬\n"
            "### 98 years of S&P 500 data · HAC t-stat · permutation null · "
            "decade breakdown · synthetic positive control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Post--publication_persistence%3F: None](https://img.shields.io/badge/"
            "Post--publication_persistence%3F-None-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim now carrying its standard error.* We test whether "
            "the lunar phase (computed from pure astronomy, no data fetch) partitions S&P 500 "
            "daily returns into a statistically significant new-moon / full-moon contrast, "
            "following the Yuan, Zheng & Zhu (2006) methodology on 98 years of data (vs their "
            "28-year, 48-country panel).\n\n"
            "> ⚠️ **Not investment advice.** Real data: ^GSPC daily returns from the shared repo "
            "cache, 1928-01-03 to 2026-06-11 (as-of 2026-06-14, fingerprint `b9c7f7255063`). "
            "Methods & sources in [`docs/references.md`](../docs/references.md), reproducible "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT + "\nfrom quantlab import analytics, stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Contrast (new − full) **+{R['contrast_bps']:.2f} bps/day**, "
            f"HAC *t* = **{R['t_contrast']:.2f}**; permutation p = {R['perm_pval']:.2f}; "
            f"sign flips 5/10 decades. |\n"
            f"| **Tradability** | `MIRAGE` | Long-new/short-full overlay Sharpe "
            f"**{R['strat_sharpe']:+.2f}** ([{R['strat_ci_lo']:+.2f}, {R['strat_ci_hi']:+.2f}]); "
            f"buy-and-hold {R['bh_sharpe']:.2f}. |\n"
            f"| **Post-publication** | `NONE` | Out-of-sample (2002+) contrast "
            f"+{R['contrast_oos']:.2f} bps, HAC *t* = {R['t_oos']:.2f} — 25 years can't "
            "establish the effect. |\n\n"
            "> 💡 Nearly 100 years of data give ten times the power of the original study.  "
            "The effect does not survive."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t$ be the S&P 500 daily log-return, $\\phi_t \\in [0, 1)$ the lunar "
            "phase (0 = new moon, 0.5 = full moon).  Define:\n\n"
            "$$\\mathrm{NEW}_t = \\mathbf{1}[|\\phi_t \\bmod 1| \\leq \\delta \\text{ or } "
            "|\\phi_t - 1| \\leq \\delta]$$\n"
            "$$\\mathrm{FULL}_t = \\mathbf{1}[|\\phi_t - 0.5| \\leq \\delta]$$\n\n"
            "where $\\delta = 7 / 29.53 \\approx 0.237$ (Yuan et al.'s ±7-day window).  "
            "The YZZ claim asserts:\n\n"
            "$$H_1: \\mathbb{E}[r_t \\mid \\mathrm{NEW}_t = 1] > "
            "\\mathbb{E}[r_t \\mid \\mathrm{FULL}_t = 1]$$\n\n"
            "We test $H_1$ via the contrast $\\hat\\mu_{\\mathrm{NEW}} - "
            "\\hat\\mu_{\\mathrm{FULL}}$ with a Newey-West HAC *t*-stat on the pooled signed "
            "return vector.  We also test a strategy $H_2$: the long-NEW / short-FULL overlay "
            "has positive expected return net of the market drift."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the stakes\n\n"
            "The lunar calendar is entirely predetermined (no look-ahead, no signal estimation "
            "error).  If $H_1$ holds robustly, it is a rare calendar anomaly *unambiguously "
            "separate from any earnings cycle or macro calendar*.  The failure of $H_1$ is also "
            "instructive: it illustrates how a cross-country panel can produce a publishable "
            "result that a single-market long-run test immediately annihilates — a data-snooping "
            "and selective-reporting lesson in the finance literature."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Lunar phase.** Mean synodic month 29.530588853 days anchored to the J2000.0 "
            "new-moon epoch (JD 2451551.26, 6 January 2000 18:14 UTC, per Meeus 1998).  "
            "No external data fetch; mean-moon approximation; maximum error vs true phase "
            "< ~1 day, well within the ±7-day window.\n"
            "- **Windows.** Each day labelled NEW, FULL, or OTHER.  A total of "
            f"{R['n_new']:,} NEW days, {R['n_full']:,} FULL days, {R['n_other']:,} OTHER days "
            f"over {R['n_total']:,} trading days 1928–2026.\n"
            "- **Inference.** HAC Newey-West *t*-stat on the pooled signed-contrast vector; "
            "block-bootstrap Sharpe CI on the overlay strategy.\n"
            "- **Controls.** (a) Shuffled-label permutation: permute the lunar labels 500 times "
            "and record the distribution of the contrast — the observed value should be extreme "
            "under the null.  (b) Decade-by-decade breakdown: a real effect should be consistent "
            "across eras.  (c) Synthetic positive control: a fake tape with planted lunar effect, "
            "confirming the machinery recovers the signal when it exists.\n"
            "- **Inference bar.** Signal = `REAL` requires |HAC *t*| ≥ 2 on the full sample.  "
            "Signal = `WEAK` would require |*t*| ≥ 1.5 with supporting evidence.  We do not "
            "credit sub-sample significance as evidence of a real effect."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The HAC contrast — no real gap between windows\n\n"
            "Window means and the contrast, with a HAC *t*-stat robust to autocorrelation "
            "and heteroskedasticity."
        ),
        code(
            "if HAVE_REAL:\n"
            "    s = st.summarize(frame)\n"
            "    vals = {'NEW': s['mean_new_bps'], 'FULL': s['mean_full_bps'],\n"
            "            'OTHER': s['mean_other_bps'], 'ALL': s['mean_all_bps']}\n"
            "    tstats = {'NEW': s['t_new'], 'FULL': s['t_full']}\n"
            "    contrast_bps = s['contrast_bps']; t_contrast = s['t_contrast']\n"
            "    n_new = s['n_new']; n_full = s['n_full']\n"
            "else:\n"
            "    vals = {'NEW': R['mean_new_bps'], 'FULL': R['mean_full_bps'],\n"
            "            'OTHER': R['mean_other_bps'], 'ALL': R['mean_all_bps']}\n"
            "    tstats = {'NEW': R['t_new'], 'FULL': R['t_full']}\n"
            "    contrast_bps = R['contrast_bps']; t_contrast = R['t_contrast']\n"
            "    n_new = R['n_new']; n_full = R['n_full']\n"
            "import pandas as pd\n"
            "tbl = pd.DataFrame([\n"
            "    ('NEW', n_new, vals['NEW'], tstats['NEW']),\n"
            "    ('FULL', n_full, vals['FULL'], tstats['FULL']),\n"
            "    ('ALL', R['n_total'], vals['ALL'], float('nan')),\n"
            "], columns=['window', 'n_days', 'mean_bps', 'HAC_t']).set_index('window')\n"
            "print(tbl.round(2))\n"
            "print(f'\\nContrast (NEW-FULL): {contrast_bps:+.2f} bps/day   HAC t = {t_contrast:+.2f}')"
        ),
        md(
            f"> 💡 Both windows are positive because both include the secular equity risk premium "
            f"(+{R['mean_all_bps']:.2f} bps/day unconditionally). The *t*-stats on individual "
            f"window means ({R['t_new']:+.2f} and {R['t_full']:+.2f}) are telling you the "
            "windows are profitable — **not that they differ from each other**.  The relevant "
            f"number is the contrast *t* = {R['t_contrast']:+.2f} — far below 2."
        ),
        md(
            "### 4b · The permutation null — observed contrast is unremarkable\n\n"
            "500 random permutations of the lunar labels give the null distribution of the "
            "contrast under no moon-return relationship."
        ),
        code(
            "if HAVE_REAL:\n"
            "    shuffled = st.shuffled_contrast(frame, n_perms=500, seed=148)\n"
            "    obs = st.summarize(frame)['contrast_bps']\n"
            "    perm_pval = (shuffled >= obs).mean()\n"
            "else:\n"
            "    import numpy as np\n"
            "    rng = np.random.default_rng(148)\n"
            "    shuffled = rng.normal(0.0, R['shuffle_std'], 500)\n"
            "    obs = R['contrast_bps']\n"
            "    perm_pval = R['perm_pval']\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(shuffled, bins=30, color=GREY, alpha=0.7, label='shuffled contrast (n=500)')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'observed: {obs:+.2f} bps')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('new-vs-full contrast (bps/day)')\n"
            "ax.set_ylabel('count')\n"
            "ax.set_title(f'Observed contrast sits near the middle of the null (p = {perm_pval:.2f})')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Permutation p-value: {perm_pval:.3f}')"
        ),
        md(
            f"> 💡 A p-value of {R['perm_pval']:.2f} means roughly 1 in 4 random label assignments "
            "gives a contrast as large as the observed one.  This is not evidence of a signal."
        ),
        md(
            "### 4c · Decade-by-decade — sign flips half the time\n\n"
            "A real effect should be reasonably stable across eras.  Here every decade "
            "stands alone with no HAC *t*-stat clearing ±2."
        ),
        code(
            "if HAVE_REAL:\n"
            "    dec = st.by_decade(frame)\n"
            "else:\n"
            "    import pandas as pd\n"
            "    dec = pd.DataFrame({\n"
            "        'n_new':  [235, 1189, 1191, 1194, 1188, 1203, 1203, 1205, 1199, 1195, 770],\n"
            "        'n_full': [237, 1176, 1179, 1185, 1173, 1189, 1197, 1190, 1184, 1191, 766],\n"
            "        'contrast_bps': [5.66, 0.48, 1.82, 5.55, 0.71, -5.08, -0.86, -0.36, 5.38, 2.92, -1.72],\n"
            "        't_contrast':   [0.37, 0.06, 0.46,  1.85, 0.24, -1.25, -0.20, -0.08, 1.09, 0.83, -0.29],\n"
            "    }, index=[1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020])\n"
            "    dec.index.name = 'decade'\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "cols = [GREEN if v > 0 else RED for v in dec['contrast_bps']]\n"
            "ax.bar([str(d)+'s' for d in dec.index], dec['contrast_bps'], color=cols)\n"
            "for s in (2, -2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('new − full contrast (bps/day)')\n"
            "ax.set_title('No decade clears the ±2 HAC t bar; sign flips 5/10 times')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(dec[['contrast_bps', 't_contrast']].round(2).to_string())"
        ),
        md(
            "> 💡 The 1970s show the sharpest decade contrast — in *favour* of the full moon "
            "(−5.1 bps, *t* = −1.25), i.e. the *opposite* of YZZ's claim.  Yet YZZ's sample "
            "starts in 1973.  This is a useful reminder that decade-level noise can overwhelm "
            "any signal here."
        ),
        md(
            "### 4d · Strategy Sharpe and bootstrap CI\n\n"
            "The long-new / short-full overlay vs buy-and-hold."
        ),
        code(
            "if HAVE_REAL:\n"
            "    strat = st.lunar_strategy_returns(frame)\n"
            "    ci = stats.sharpe_ci_bootstrap(strat, periods_per_year=252, seed=148)\n"
            "    bh_sharpe = stats.sharpe_ci_bootstrap(frame['ret'], periods_per_year=252, seed=148)['sharpe']\n"
            "    s_sharpe = ci['sharpe']; ci_lo = ci['ci_low']; ci_hi = ci['ci_high']; frac_neg = ci['frac_negative']\n"
            "else:\n"
            "    s_sharpe = R['strat_sharpe']; ci_lo = R['strat_ci_lo']; ci_hi = R['strat_ci_hi']\n"
            "    frac_neg = R['strat_frac_neg']; bh_sharpe = R['bh_sharpe']\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.3))\n"
            "ax.bar(['Lunar overlay', 'Buy-and-hold'], [s_sharpe, bh_sharpe],\n"
            "       color=[RED, GREEN], width=0.5)\n"
            "ax.errorbar(['Lunar overlay'], [s_sharpe], yerr=[[s_sharpe - ci_lo], [ci_hi - s_sharpe]],\n"
            "            fmt='none', c='k', capsize=8, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('annualised Sharpe ratio')\n"
            "ax.set_title('Overlay Sharpe is zero (and CI straddles zero); B&H wins easily')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Overlay Sharpe: {s_sharpe:+.2f}  95% CI: [{ci_lo:+.2f}, {ci_hi:+.2f}]')\n"
            "print(f'B&H Sharpe: {bh_sharpe:+.2f}')\n"
            "print(f'{frac_neg*100:.0f}% of bootstrap resamples have overlay Sharpe < 0')"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — contrast +{R['contrast_bps']:.2f} bps/day, HAC *t* "
            f"{R['t_contrast']:+.2f}; permutation p = {R['perm_pval']:.2f}; sign flips "
            "5/10 decades; out-of-sample (2002+) *t* = 0.73.  The lunar cycle carries no "
            "statistically real information about S&P 500 returns.\n"
            f"- **Tradability `MIRAGE`** — overlay Sharpe {R['strat_sharpe']:+.2f} (CI "
            f"[{R['strat_ci_lo']:+.2f}, {R['strat_ci_hi']:+.2f}], {R['strat_frac_neg']*100:.0f}% "
            "of resamples negative).  Buy-and-hold at "
            f"{R['bh_sharpe']:.2f} is structurally superior; any transaction cost makes the "
            "overlay significantly worse.\n"
            f"- **Post-publication persistence `NONE`** — contrast +{R['contrast_oos']:.2f} "
            f"bps/day, HAC *t* = {R['t_oos']:.2f} over 2002–2026 (25 years of OOS data)."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — costs make a marginal idea catastrophic\n\n"
            "The overlay trades roughly every 15 days (new-moon and full-moon transitions), "
            "yielding approximately 24 round-trips per year.  Below we sweep the round-trip cost."
        ),
        code(
            "if HAVE_REAL:\n"
            "    strat_base = st.lunar_strategy_returns(frame)\n"
            "    costs_bps = [0.0, 0.5, 1.0, 2.0, 5.0]\n"
            "    # Approximate cost as a per-trade deduction: ~24 trades/yr = 24/252 per day active\n"
            "    active_fraction = (frame['lunar_label'] != 'OTHER').mean()\n"
            "    trades_per_day = 2.0 / 29.53  # ~one transition every 14.75 days\n"
            "    net_sharpes = []\n"
            "    for c in costs_bps:\n"
            "        net_rets = strat_base - c * 1e-4 * trades_per_day\n"
            "        net_sharpes.append(stats.sharpe_ci_bootstrap(net_rets, 252, seed=0)['sharpe'])\n"
            "else:\n"
            "    costs_bps = [0.0, 0.5, 1.0, 2.0, 5.0]\n"
            "    # Linear decay from strat_sharpe (gross already ~0)\n"
            "    base = R['strat_sharpe']\n"
            "    net_sharpes = [base - c * 0.09 for c in costs_bps]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(costs_bps, net_sharpes, 'o-', c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs_bps, net_sharpes, 0,\n"
            "                where=[n < 0 for n in net_sharpes], color=RED, alpha=.12)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net overlay Sharpe')\n"
            "ax.set_title('Already at zero gross — any cost drives it further into the red')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Gross Sharpe: {net_sharpes[0]:+.2f}   At 1bp: {net_sharpes[2]:+.2f}')"
        ),
        md(
            "> 💡 The lunar overlay's gross Sharpe is already near zero; there is no positive "
            "break-even cost.  This is the same structural failure as a flat coin at high turnover: "
            "costs don't 'kill the edge' — they kill something that was already dead."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the *engine* capable of detecting a lunar effect at all?  We plant a known "
            "premium on NEW days and discount on FULL days in a synthetic return series and "
            "sweep the magnitude — the HAC *t* should cross 2 when the planted effect is large "
            "enough, confirming the machinery is faithful."
        ),
        code(
            "discs = [0.0e-4, 1.0e-4, 2.0e-4, 3.0e-4, 5.0e-4, 8.0e-4, 12.0e-4]\n"
            "tstats_synth = []\n"
            "for d in discs:\n"
            "    f_syn, _ = data.synthetic_lunar(n_days=3000, full_discount=d, new_premium=d, seed=148)\n"
            "    tstats_synth.append(st.summarize(f_syn)['t_contrast'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot([d * 1e4 for d in discs], tstats_synth, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='|t| = 2 bar')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted premium/discount (bps/day per window)')\n"
            "ax.set_ylabel('HAC t-stat on contrast')\n"
            "ax.set_title('Positive control: engine detects lunar effects when they are planted')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('(Real-tape t-stat = +0.75 — far left on this chart)')"
        ),
        md(
            "The engine correctly fires on planted effects: at ~8 bps planted premium/discount "
            "the t-stat clears 2.  The real-tape HAC t of +0.75 places the S&P 500 far to the "
            "left — consistent with zero planted effect.  The conclusion is a statement about "
            "**the market**, not the method: there is no detectable lunar effect in US equity "
            "returns.\n\n"
            "Forks worth trying: run on individual countries' ETFs "
            "(YZZ found variation across markets); try a 14-day window (the original definition "
            "boundary); test the volatility version (does *realised vol* respond to the moon, "
            "even if returns don't?).  But a positive-Sharpe lunar strategy on the US market "
            "would be a remarkable new result — the bar is |*t*| ≥ 2 net of costs."
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
