"""Generate the two narrative notebooks for Study 194 (Turkey).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). Synthetic
figures run anywhere, offline and deterministic; real-tape cells use the shared
^GSPC cache if present and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md), so the notebook re-runs for any reader.

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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-15).
R = dict(
    n_days=24728, n_wed=98, n_fri=97, n_years=99,
    fp="df3897aef897",
    # Wednesday before Thanksgiving
    mean_wed_bps=5.87, mean_other_wed_bps=7.17, contrast_wed_bps=-1.30,
    tstat_welch_wed=-0.116, pval_welch_wed=0.9082, tstat_hac_wed=0.458,
    pval_bonferroni_wed=1.0,
    # Black Friday
    mean_fri_bps=20.62, mean_other_fri_bps=4.36, contrast_fri_bps=16.26,
    tstat_welch_fri=1.650, pval_welch_fri=0.1021, tstat_hac_fri=2.011,
    pval_bonferroni_fri=0.2042,
    # Placebo: Tuesday before Thanksgiving
    n_tue=98, mean_tue_bps=5.53, contrast_tue_bps=1.67, pval_welch_tue=0.8751,
    # Unconditional
    mean_all_bps=2.41,
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

from turkey import data, strategy as st

# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-15).
R = dict(
    n_days=24728, n_wed=98, n_fri=97, n_years=99,
    fp="df3897aef897",
    mean_wed_bps=5.87, mean_other_wed_bps=7.17, contrast_wed_bps=-1.30,
    tstat_welch_wed=-0.116, pval_welch_wed=0.9082, tstat_hac_wed=0.458,
    pval_bonferroni_wed=1.0,
    mean_fri_bps=20.62, mean_other_fri_bps=4.36, contrast_fri_bps=16.26,
    tstat_welch_fri=1.650, pval_welch_fri=0.1021, tstat_hac_fri=2.011,
    pval_bonferroni_fri=0.2042,
    n_tue=98, mean_tue_bps=5.53, contrast_tue_bps=1.67, pval_welch_tue=0.8751,
    mean_all_bps=2.41,
)

def _have_real():
    shared = os.path.abspath(os.path.join("..", "..", "..", "_cache", "^GSPC_split_only.parquet"))
    local  = os.path.abspath(os.path.join("..", "_cache", "gspc_daily.parquet"))
    return os.path.exists(shared) or os.path.exists(local)

HAVE_REAL = _have_real()

if HAVE_REAL:
    DF = data.fetch_gspc(start="1928-01-01")
    S  = st.summarize(DF)
else:
    DF, S = None, None

print("real GSPC cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Turkey — does the market feast before and after Thanksgiving?\n"
            "### The pre/post-holiday effect, the S&P 500, and 99 years of evidence\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Pre--holiday_only: Post--holiday_only](https://img.shields.io/badge/Pre--holiday_only-Post--holiday_only-8b949e?style=flat-square)\n\n"
            "Every November, around the time Americans are planning their turkey dinners, Wall "
            "Street lore says stocks rise: the day before Thanksgiving is supposed to be reliably "
            "positive (pre-holiday optimism), and Black Friday (the short session the day after) "
            "is supposedly even better. This notebook turns those claims into numbers.\n\n"
            "> **This is the plain-language layer.** For the t-stats, the Bonferroni correction, "
            "and the power analysis, see "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is "
            "drawn by the code beside it. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the market rise on the Wednesday before Thanksgiving? | **Not detectably.** "
            f"The mean return is **+{R['mean_wed_bps']:.1f} bps** vs +{R['mean_other_wed_bps']:.1f} bps "
            f"for other Wednesdays. Contrast **{R['contrast_wed_bps']:+.1f} bps** (wrong direction). "
            f"p = {R['pval_welch_wed']:.2f}. |\n"
            f"| What about Black Friday (the short session after Thanksgiving)? | **Maybe.** "
            f"Mean return **+{R['mean_fri_bps']:.1f} bps** vs +{R['mean_other_fri_bps']:.1f} bps "
            f"for other Fridays. Contrast **+{R['contrast_fri_bps']:.1f} bps**. HAC t = "
            f"+{R['tstat_hac_fri']:.2f}, but the two-sample test gives p = {R['pval_welch_fri']:.2f}. "
            f"After Bonferroni correction for two tests: p = {R['pval_bonferroni_fri']:.2f}. |\n"
            "| Could you trade it? | **No.** ~1 event/year, p not certified, any cost fatal. |\n\n"
            "> **Bottom line:** The pre-holiday Wednesday has no edge. Black Friday shows a "
            "large raw return (+16 bps) but it does not clear the statistical bar with only 97 "
            "events. The effect is an intriguing breadcrumb, not a tradable signal."
        ),

        # ---- BEAT 1 -- THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *'Pre-holiday trading days deliver systematically above-average returns — '* "
            "*(Ariel, 1990). For Thanksgiving specifically: the Wednesday before and the '* "
            "*Black Friday after are the strongest holiday-adjacent effects in the US calendar.'*\n\n"
            "Robert Ariel's 1990 Journal of Finance paper documented that the day before each "
            "US market holiday delivered returns several times the unconditional mean. Thanksgiving "
            "is one of eight US exchange holidays. Subsequent work (Brockman & Michayluk 1998) "
            "argued the effect 'persisted' across markets and time. The street belief: buy on "
            "Tuesday/Wednesday before Thanksgiving, sell at or after the close on Black Friday.\n\n"
            f"Our tape: **{R['n_wed']} Thanksgiving Wednesdays** and **{R['n_fri']} Black Fridays** "
            f"in the S&P 500 from 1928 to 2026. That is about "
            f"**{R['n_wed']/R['n_years']:.1f} events per year** — very thin."
        ),

        # ---- BEAT 2 -- SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If true, it would mean that investor psychology predictably shifts before a major "
            "cultural holiday: everyone is in a good mood, trading desks are understaffed, and "
            "the path of least resistance is up. It would also mean that Black Friday — a "
            "historically low-volume half-day session — captures the 'recovery' return after the "
            "holiday closure gap.\n\n"
            "If false (or fragile), it is another reminder that a pattern documented on a few "
            "decades of one index can easily be an artefact of small samples and data mining. "
            "The pre-holiday effect has been documented in many papers but the effect sizes "
            "are small and the post-publication evidence is mixed."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three guards against finding what we're looking for:\n\n"
            "1. **A matched control.** Compare each Thanksgiving day to *all other days of the "
            "same weekday* (Wednesday vs Wednesdays, Friday vs Fridays). This filters out any "
            "general day-of-week patterns.\n"
            "2. **A placebo.** Test the Tuesday *before* Thanksgiving (same week, same season, "
            "one day earlier than the pre-holiday). If Tuesday looks as 'anomalous' as Wednesday, "
            "the signal is late-November seasonality, not a pre-holiday effect.\n"
            "3. **Bonferroni correction.** We run two primary tests (Wednesday-before AND "
            "Friday-after). Testing two hypotheses inflates the false-positive rate; we apply "
            "Bonferroni with k=2, requiring p < 0.025 (not 0.05) for each to declare significance."
        ),

        # ---- BEAT 4 -- THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First: distributions of returns — Thanksgiving Wednesday vs other Wednesdays.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    wed_tsg = DF.loc[DF['is_wed_before_tsg'], 'ret'] * 1e4\n"
            "    other_wed = DF.loc[(DF.index.weekday==2) & ~DF['is_wed_before_tsg'], 'ret'] * 1e4\n"
            "    m_wed = wed_tsg.mean(); m_ow = other_wed.mean()\n"
            "else:\n"
            "    rng = np.random.default_rng(194)\n"
            "    wed_tsg  = pd.Series(rng.normal(R['mean_wed_bps'], 80, R['n_wed']))\n"
            "    other_wed = pd.Series(rng.normal(R['mean_other_wed_bps'], 80, 4943))\n"
            "    m_wed = R['mean_wed_bps']; m_ow = R['mean_other_wed_bps']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.8))\n"
            "ax.hist(other_wed, bins=60, color=GREY, alpha=0.5, density=True, label='Other Wednesdays')\n"
            "ax.hist(wed_tsg, bins=20, color=AMBER, alpha=0.7, density=True, label='Wed before Thanksgiving')\n"
            "ax.axvline(m_ow, ls='--', c=GREY, lw=1.5, label=f'Other Wed mean: {m_ow:+.1f} bps')\n"
            "ax.axvline(m_wed, ls='-', c=AMBER, lw=2, label=f'Tsg Wed mean: {m_wed:+.1f} bps')\n"
            "ax.set_xlabel('daily return (bps)'); ax.set_ylabel('density')\n"
            "ax.set_title('Pre-holiday Wednesday: no detectable edge vs other Wednesdays')\n"
            "ax.legend(); ax.set_xlim(-300, 300)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Wed-before-Thanksgiving mean: {m_wed:+.2f} bps  |  Other Wednesdays: {m_ow:+.2f} bps')\n"
            f"print('Contrast: {R['contrast_wed_bps']:+.2f} bps  p = {R['pval_welch_wed']:.4f}')"
        ),
        md(
            f"The Wednesday-before-Thanksgiving has a mean of **+{R['mean_wed_bps']:.1f} bps** — "
            f"**below** the other-Wednesday average of +{R['mean_other_wed_bps']:.1f} bps. "
            f"The contrast is {R['contrast_wed_bps']:+.1f} bps (wrong direction), p = "
            f"{R['pval_welch_wed']:.2f}. There is no pre-holiday effect on Wednesday."
        ),
        md(
            "**Now Black Friday (day after Thanksgiving) vs other Fridays.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    fri_tsg = DF.loc[DF['is_fri_after_tsg'], 'ret'] * 1e4\n"
            "    other_fri = DF.loc[(DF.index.weekday==4) & ~DF['is_fri_after_tsg'], 'ret'] * 1e4\n"
            "    m_fri = fri_tsg.mean(); m_of = other_fri.mean()\n"
            "else:\n"
            "    rng2 = np.random.default_rng(195)\n"
            "    fri_tsg  = pd.Series(rng2.normal(R['mean_fri_bps'], 80, R['n_fri']))\n"
            "    other_fri = pd.Series(rng2.normal(R['mean_other_fri_bps'], 80, 4847))\n"
            "    m_fri = R['mean_fri_bps']; m_of = R['mean_other_fri_bps']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.8))\n"
            "ax.hist(other_fri, bins=60, color=GREY, alpha=0.5, density=True, label='Other Fridays')\n"
            "ax.hist(fri_tsg, bins=20, color=GREEN, alpha=0.7, density=True, label='Black Friday')\n"
            "ax.axvline(m_of, ls='--', c=GREY, lw=1.5, label=f'Other Fri mean: {m_of:+.1f} bps')\n"
            "ax.axvline(m_fri, ls='-', c=GREEN, lw=2, label=f'Black Fri mean: {m_fri:+.1f} bps')\n"
            "ax.set_xlabel('daily return (bps)'); ax.set_ylabel('density')\n"
            "ax.set_title('Black Friday: large raw mean, wide distribution, borderline stats')\n"
            "ax.legend(); ax.set_xlim(-300, 400)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Black Friday mean: {m_fri:+.2f} bps  |  Other Fridays: {m_of:+.2f} bps')\n"
            f"print('Contrast: {R['contrast_fri_bps']:+.2f} bps  HAC t = {R['tstat_hac_fri']:.2f}  "
            f"Welch p = {R['pval_welch_fri']:.4f}  Bonferroni p = {R['pval_bonferroni_fri']:.4f}')"
        ),
        md(
            f"Black Friday has a mean of **+{R['mean_fri_bps']:.1f} bps** vs "
            f"+{R['mean_other_fri_bps']:.1f} bps for other Fridays — a contrast of "
            f"**+{R['contrast_fri_bps']:.1f} bps**. The HAC t on the Black Friday group "
            f"alone is **+{R['tstat_hac_fri']:.2f}** (just clearing 2.0), but the two-sample "
            f"Welch test gives p = **{R['pval_welch_fri']:.2f}** and after Bonferroni "
            f"correction p = **{R['pval_bonferroni_fri']:.2f}**. Interesting but not certified."
        ),
        md(
            "**The placebo: Tuesday before Thanksgiving vs other Tuesdays.**"
        ),
        code(
            "labels = ['Other Wednesdays', 'Tue before Tsg (placebo)', 'Wed before Tsg', 'Black Friday']\n"
            "if HAVE_REAL:\n"
            "    vals = [\n"
            "        DF.loc[(DF.index.weekday==2) & ~DF['is_wed_before_tsg'], 'ret'].mean() * 1e4,\n"
            "        DF.loc[DF['is_placebo_tue'], 'ret'].mean() * 1e4,\n"
            "        DF.loc[DF['is_wed_before_tsg'], 'ret'].mean() * 1e4,\n"
            "        DF.loc[DF['is_fri_after_tsg'], 'ret'].mean() * 1e4,\n"
            "    ]\n"
            "else:\n"
            "    vals = [R['mean_other_wed_bps'], R['mean_tue_bps'], R['mean_wed_bps'], R['mean_fri_bps']]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.8))\n"
            "cols = [GREY, GREY, AMBER, GREEN]\n"
            "bars = ax.bar(labels, vals, color=cols, width=0.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for b, v in zip(bars, vals):\n"
            "    ax.annotate(f'{v:+.1f} bps', (b.get_x()+b.get_width()/2, v+0.2),\n"
            "                ha='center', va='bottom' if v>=0 else 'top', fontsize=10)\n"
            "ax.set_ylabel('mean daily return (bps)')\n"
            "ax.set_title('Thanksgiving week: only Black Friday stands out (barely)')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"The Tuesday placebo (+{R['mean_tue_bps']:.1f} bps, contrast "
            f"+{R['contrast_tue_bps']:.1f} bps vs other Tuesdays, p = "
            f"{R['pval_welch_tue']:.2f}) looks just like any other late-November Tuesday. "
            "The Wednesday is also flat. Only Black Friday shows a large raw number, but "
            "even that does not survive the statistical bar with only 97 events."
        ),

        # ---- BEAT 5 -- THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** Black Friday mean **+{R['mean_fri_bps']:.1f} bps** vs "
            f"+{R['mean_other_fri_bps']:.1f} bps for other Fridays. HAC t = "
            f"**+{R['tstat_hac_fri']:.2f}**, Welch p = **{R['pval_welch_fri']:.2f}**, "
            f"Bonferroni p = **{R['pval_bonferroni_fri']:.2f}**. Pre-holiday Wednesday: "
            f"contrast {R['contrast_wed_bps']:+.1f} bps, p = {R['pval_welch_wed']:.2f}. Null.\n"
            f"- **Tradability — Mirage.** ~1 event/year at ~16 bps/event = ~16 bps/yr gross. "
            "Execution cost trivially dominates, and the effect is not even statistically "
            "certified.\n"
            "- **Pre-holiday claim — Mostly false here.** Ariel (1990)'s general pre-holiday "
            "finding does not sharpen into a certifiable Thanksgiving-specific edge on the "
            "S&P 500 over 99 years."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "The Black Friday effect (if real) is ~16 bps/event, ~1 event/year — roughly "
            "16 bps/yr. In practice:\n\n"
            f"- A round-trip cost of **5 bps** applied once per year costs 5 bps/yr — "
            "already 30% of the estimated gross premium.\n"
            "- The 95% confidence interval on the 16 bps estimate at n=97 is approximately "
            "±20 bps — meaning the true effect could plausibly be 0 or even negative.\n"
            "- Black Friday is a **half-day session** (US market closes 13:00 ET). Liquidity "
            "is thin, spreads are wider, and institutional desks are often understaffed. "
            "The 'return' is partly a liquidity artefact.\n\n"
            "The fundamental barrier is **statistical power**: with only ~1 event/year, even "
            "99 years of history produce only 97 observations. The minimum detectable effect "
            "at 80% power (two-sample test, α=0.025 after Bonferroni) is over 30 bps — far "
            "larger than even the optimistic estimate of 16 bps."
        ),

        # ---- BEAT 7 -- GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **The general pre-holiday effect.** Study 95 (Holiday-Cheer) pools all US "
            "pre-holidays and may find a stronger aggregate signal precisely because N is "
            "larger. Isolating Thanksgiving loses statistical power.\n"
            "- **International Thanksgiving proxies.** Canadian Thanksgiving (second Monday "
            "of October) trades at full volume. Testing the same pre/post effect there would "
            "provide independent evidence.\n"
            "- **Volatility channel.** Does the VIX drop heading into Thanksgiving and spike "
            "after? The return anomaly (if any) may partly be a vol-compression / vol-recovery "
            "effect rather than a direction signal.\n"
            "- **Black Friday specifically.** The post-holiday short session is unique: "
            "institutional desks are empty, retail sentiment may be elevated by consumer "
            "spending news. A more targeted study of half-day sessions (also: Christmas Eve, "
            "July 3) would test whether the pattern is about Thanksgiving or about short "
            "sessions in general.\n\n"
            "*Think the turkey really does move the market? Fork this, add more holidays, "
            "test intraday returns, or look at the VIX. That's the bar.*"
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
            "# Turkey — a quantitative teardown\n"
            "### ^GSPC daily 1928-2026 · HAC t-stats · Welch test · Bonferroni k=2 · synthetic positive control\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Pre--holiday_only: Post--holiday_only](https://img.shields.io/badge/Pre--holiday_only-Post--holiday_only-8b949e?style=flat-square)\n\n"
            "The deep companion to [01_for_the_curious.ipynb](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim now carrying its standard error.* We test the "
            "Wednesday-before-Thanksgiving and Black Friday returns against matched-weekday "
            "baselines, with a Tuesday-before placebo and Bonferroni correction for two "
            "simultaneous primary hypotheses.\n\n"
            "> **Not investment advice.** Real data: ^GSPC daily, 1928-01-03 to 2026-06-12 "
            f"(fingerprint `{R['fp']}`). Methods & sources in "
            "[`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The `> In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Black Fri mean **+{R['mean_fri_bps']:.2f} bps/day**, "
            f"contrast **+{R['contrast_fri_bps']:.2f} bps** vs other Fridays, HAC t = "
            f"**+{R['tstat_hac_fri']:.2f}**, Welch p = **{R['pval_welch_fri']:.2f}**, "
            f"Bonferroni p = **{R['pval_bonferroni_fri']:.2f}**. Wed-before: null "
            f"(contrast {R['contrast_wed_bps']:+.2f} bps, p = {R['pval_welch_wed']:.2f}). |\n"
            f"| **Tradability** | `MIRAGE` | ~1 event/year, 16 bps gross, CI ±~20 bps, "
            "half-day liquidity; any cost is fatal. |\n"
            f"| **Pre/post-holiday asymmetry** | post-holiday only | Wednesday shows no edge; "
            "only the half-day Black Friday session shows a (borderline) positive return. |\n\n"
            "> In plain words: 99 years, 97-98 events, and the pre-holiday claim is a clear "
            "null. The Black Friday result is intriguing but statistically borderline and "
            "practically untradable."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t$ be the daily log-return of the S&P 500. The Ariel (1990) hypothesis\n"
            "asserts:\n\n"
            r"$$\mathbb{E}[r_t \mid t \text{ is the pre-holiday day}] > \mathbb{E}[r_t \mid t \text{ is a typical trading day}]$$" + "\n\n"
            "For Thanksgiving specifically, the pre-holiday day is the Wednesday before "
            "(last full session). A secondary claim asserts the same for the Black Friday "
            "post-holiday half-session. We test both with:\n\n"
            "- **H0a:** mean Wed-before-Thanksgiving return = mean other-Wednesday return.\n"
            "- **H0b:** mean Black Friday return = mean other-Friday return.\n"
            "- **Control:** matched-weekday (all other Wednesdays / Fridays).\n"
            "- **Placebo:** Tuesday-before-Thanksgiving vs other Tuesdays (same week, earlier).\n"
            "- **Multiple comparisons:** Bonferroni k=2 for the two primary tests; threshold "
            "p < 0.025 for α = 0.05."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — the stakes\n\n"
            "The pre-holiday effect is one of the most robustly documented calendar anomalies "
            "in the academic literature (Ariel 1990, Brockman & Michayluk 1998). If it holds "
            "for Thanksgiving specifically, it would provide a systematic 1-2 day trading "
            "opportunity anchored to the US cultural calendar. The failure (for the pre-holiday "
            "Wednesday) and the borderline result (for Black Friday) are informative: they "
            "suggest that either (a) the general pre-holiday effect is diluted when you isolate "
            "a single holiday, or (b) the effect has faded post-publication, or (c) it was "
            "always fragile in liquid large-cap indices like the S&P 500."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal definition.** Thanksgiving = fourth Thursday of November "
            "(pure calendar arithmetic). Wednesday-before = Thanksgiving − 1 day. "
            "Black Friday = Thanksgiving + 1 day. Tuesday-before = Thanksgiving − 2 days.\n"
            "- **Outcome.** Close-to-close log-return on the event day itself.\n"
            "- **Primary tests (two).** Welch t-test (unequal variances): Wed-before vs "
            "other Wednesdays; Black Friday vs other Fridays. HAC (Newey-West) t-stat on "
            "each group separately.\n"
            "- **Placebo.** Same test with Tuesday-before as the 'anomaly' day.\n"
            "- **Bonferroni.** Two primary hypotheses → corrected threshold p < 0.025. "
            "Bonferroni-corrected p = min(raw p × 2, 1).\n"
            "- **Positive control.** Synthetic tape with planted effects confirms the engine "
            "can detect real Thanksgiving anomalies when they exist."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Primary test A: Wednesday before Thanksgiving vs other Wednesdays\n\n"
            "HAC t-stat on the Wednesday-before group; Welch test on the contrast."
        ),
        code(
            "if HAVE_REAL:\n"
            "    wed = st.wednesday_comparison(DF)\n"
            "    n_w, m_w, t_w = wed['n_wed'], wed['mean_wed_bps'], wed['tstat_hac_wed']\n"
            "    n_ow, m_ow, tw_w, pw_w = wed['n_other_wed'], wed['mean_other_wed_bps'], wed['tstat_welch'], wed['pval_welch']\n"
            "    c_w = wed['contrast_bps']\n"
            "else:\n"
            "    n_w,m_w,t_w = R['n_wed'],R['mean_wed_bps'],R['tstat_hac_wed']\n"
            "    n_ow,m_ow,tw_w,pw_w = R['n_wed']*50,R['mean_other_wed_bps'],R['tstat_welch_wed'],R['pval_welch_wed']\n"
            "    c_w = R['contrast_wed_bps']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.5))\n"
            "ax.bar(['Wed-before-Thanksgiving', 'Other Wednesdays'], [m_w, m_ow], color=[AMBER, GREY], width=0.4)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('mean daily return (bps)')\n"
            "ax.set_title(f'Pre-holiday Wednesday: no edge (Welch p = {pw_w:.2f})')\n"
            "for i,(lbl,v,n) in enumerate(zip(['Tsg Wed','Other Wed'],[m_w,m_ow],[n_w,n_ow])):\n"
            "    ax.annotate(f'{v:+.2f} bps\\nn={n:,}', (i, v), ha='center',\n"
            "                va='bottom' if v>=0 else 'top', fontsize=10)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Wed-before: n={n_w}, mean={m_w:+.2f} bps, HAC t={t_w:+.3f}')\n"
            "print(f'Other Wed: n={n_ow:,}, mean={m_ow:+.2f} bps')\n"
            "print(f'Contrast: {c_w:+.2f} bps | Welch t={tw_w:+.3f} | p={pw_w:.4f}')"
        ),
        md(
            f"> In plain words: {R['n_wed']} Thanksgiving Wednesdays over 99 years produce a "
            f"mean return of **+{R['mean_wed_bps']:.2f} bps** — "
            f"**{abs(R['contrast_wed_bps']):.2f} bps below** the other-Wednesday average. "
            f"Welch p = {R['pval_welch_wed']:.2f}. The pre-holiday optimism story fails on the "
            "S&P 500."
        ),
        md(
            "### 4b · Primary test B: Black Friday vs other Fridays\n\n"
            "The half-day post-holiday session is the stronger candidate."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fri = st.friday_comparison(DF)\n"
            "    n_f, m_f, t_f = fri['n_fri'], fri['mean_fri_bps'], fri['tstat_hac_fri']\n"
            "    n_of, m_of, tw_f, pw_f = fri['n_other_fri'], fri['mean_other_fri_bps'], fri['tstat_welch'], fri['pval_welch']\n"
            "    c_f = fri['contrast_bps']\n"
            "    bp_f = fri['pval_welch'] * 2 if fri['pval_welch'] < 0.5 else 1.0\n"
            "else:\n"
            "    n_f,m_f,t_f = R['n_fri'],R['mean_fri_bps'],R['tstat_hac_fri']\n"
            "    n_of,m_of,tw_f,pw_f = R['n_fri']*50,R['mean_other_fri_bps'],R['tstat_welch_fri'],R['pval_welch_fri']\n"
            "    c_f = R['contrast_fri_bps']; bp_f = R['pval_bonferroni_fri']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.5))\n"
            "ax.bar(['Black Friday', 'Other Fridays'], [m_f, m_of], color=[GREEN, GREY], width=0.4)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('mean daily return (bps)')\n"
            "ax.set_title(f'Black Friday: large raw mean (Welch p = {pw_f:.2f}, Bonf p = {bp_f:.2f})')\n"
            "for i,(lbl,v,n) in enumerate(zip(['Black Fri','Other Fri'],[m_f,m_of],[n_f,n_of])):\n"
            "    ax.annotate(f'{v:+.2f} bps\\nn={n:,}', (i, v), ha='center',\n"
            "                va='bottom' if v>=0 else 'top', fontsize=10)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Black Friday: n={n_f}, mean={m_f:+.2f} bps, HAC t={t_f:+.3f}')\n"
            "print(f'Other Fri: n={n_of:,}, mean={m_of:+.2f} bps')\n"
            "print(f'Contrast: {c_f:+.2f} bps | Welch t={tw_f:+.3f} | p={pw_f:.4f} | Bonf p={bp_f:.4f}')"
        ),
        md(
            f"> In plain words: Black Friday mean **+{R['mean_fri_bps']:.2f} bps** vs "
            f"+{R['mean_other_fri_bps']:.2f} bps for other Fridays. The HAC t on the group "
            f"alone is **+{R['tstat_hac_fri']:.2f}** (just clears 2.0), but the two-sample "
            f"Welch test gives p = **{R['pval_welch_fri']:.2f}** and Bonferroni p = "
            f"**{R['pval_bonferroni_fri']:.2f}**. Intriguing but not certified."
        ),
        md(
            "### 4c · Placebo — Tuesday before Thanksgiving\n\n"
            "A matched-window test: same week, same late-November season, but two days "
            "before Thanksgiving (not the immediate pre-holiday)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    plac = st.tuesday_placebo(DF)\n"
            "    m_t, c_t, p_t = plac['mean_tue_bps'], plac['contrast_bps'], plac['pval_welch']\n"
            "else:\n"
            "    m_t, c_t, p_t = R['mean_tue_bps'], R['contrast_tue_bps'], R['pval_welch_tue']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "groups = ['Other Tue', 'Tue before Tsg\\n(placebo)', 'Wed before Tsg', 'Black Friday']\n"
            "if HAVE_REAL:\n"
            "    means = [\n"
            "        DF.loc[(DF.index.weekday==1) & ~DF['is_placebo_tue'], 'ret'].mean()*1e4,\n"
            "        m_t,\n"
            "        DF.loc[DF['is_wed_before_tsg'], 'ret'].mean()*1e4,\n"
            "        DF.loc[DF['is_fri_after_tsg'], 'ret'].mean()*1e4,\n"
            "    ]\n"
            "else:\n"
            "    means = [R['mean_other_wed_bps'], m_t, R['mean_wed_bps'], R['mean_fri_bps']]\n"
            "cols = [GREY, GREY, AMBER, GREEN]\n"
            "bars = ax.bar(groups, means, color=cols, width=0.4)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for b, v in zip(bars, means):\n"
            "    ax.annotate(f'{v:+.1f} bps', (b.get_x()+b.get_width()/2, v+0.2),\n"
            "                ha='center', va='bottom' if v>=0 else 'top', fontsize=9)\n"
            "ax.set_ylabel('mean daily return (bps)')\n"
            "ax.set_title('Thanksgiving week: Black Friday stands out; Tuesday placebo is flat')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Placebo Tue: mean={m_t:+.2f} bps, contrast={c_t:+.2f} bps, p={p_t:.4f}')"
        ),
        md(
            f"> In plain words: The Tuesday placebo (+{R['mean_tue_bps']:.2f} bps, "
            f"contrast +{R['contrast_tue_bps']:.2f} bps, p = {R['pval_welch_tue']:.2f}) "
            "is indistinguishable from any other Tuesday. The Thanksgiving effect (if any) "
            "is concentrated on the post-holiday Black Friday session, not the pre-holiday "
            "Wednesday or the two-days-before Tuesday."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — Black Fri contrast **+{R['contrast_fri_bps']:+.2f} bps**, "
            f"HAC t = **+{R['tstat_hac_fri']:.2f}**, Welch p = {R['pval_welch_fri']:.2f}, "
            f"Bonferroni p = {R['pval_bonferroni_fri']:.2f}. HAC t clears 2.0 on the group "
            "alone but the two-sample test does not, and the Bonferroni bar is not met. "
            f"Wed-before: null (contrast {R['contrast_wed_bps']:+.2f} bps, "
            f"p = {R['pval_welch_wed']:.2f}).\n"
            f"- **Tradability `MIRAGE`** — ~1 event/year, ~16 bps gross, CI ±~20 bps, "
            "half-day liquidity. The economics of the trade cannot be made to work at any "
            "realistic cost structure.\n"
            f"- **Pre/post asymmetry** — The Black Friday effect (if real) is a "
            "post-holiday, not a pre-holiday, phenomenon. This is consistent with the "
            "half-day session being thin and gap-prone rather than with investor optimism."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the power problem\n\n"
            "The fundamental barrier is **statistical power**, not cost."
        ),
        code(
            "import numpy as np\n"
            "from scipy import stats\n"
            "\n"
            "# Power analysis: how large an effect could we detect at 80% power with n=97?\n"
            "daily_vol_bps = 100.0  # typical GSPC daily vol in bps\n"
            "n_events = R['n_fri']   # 97 observations\n"
            "alpha_bonf = 0.025      # Bonferroni-adjusted alpha (k=2)\n"
            "power_target = 0.80\n"
            "\n"
            "t_alpha = stats.norm.ppf(1 - alpha_bonf / 2)\n"
            "t_power = stats.norm.ppf(power_target)\n"
            "mde = (t_alpha + t_power) * daily_vol_bps / np.sqrt(n_events)\n"
            "\n"
            "effects = np.linspace(0, 50, 300)\n"
            "# Two-sample Welch: denominator is vol/sqrt(n) + vol/sqrt(n_other)\n"
            "# For simplicity, n_other >> n_events, so dominated by 1/sqrt(n_events)\n"
            "powers = [stats.norm.cdf(e / (daily_vol_bps / np.sqrt(n_events)) - t_alpha)\n"
            "          for e in effects]\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "ax.plot(effects, powers, c=GREEN, lw=2)\n"
            "ax.axhline(0.80, ls='--', c=GREY, lw=1.5, label='80% power')\n"
            "ax.axvline(mde, ls=':', c=AMBER, lw=1.5, label=f'MDE ~ {mde:.1f} bps')\n"
            "ax.axvline(R['contrast_fri_bps'], ls='-', c=GREEN, lw=1.5,\n"
            "           label=f'Observed contrast = {R[\"contrast_fri_bps\"]:.1f} bps')\n"
            "ax.set_xlabel('true effect size (bps/event)'); ax.set_ylabel('statistical power')\n"
            "ax.set_title(f'With n=97 events, detectable effects must be > ~{mde:.0f} bps')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'MDE at 80% power, alpha=0.025: {mde:.1f} bps')\n"
            "print(f'Observed Black Friday contrast: {R[\"contrast_fri_bps\"]:+.1f} bps')\n"
            "print(f'Power to detect {R[\"contrast_fri_bps\"]:.0f} bps: '\n"
            "      f'{stats.norm.cdf(R[\"contrast_fri_bps\"] / (daily_vol_bps / np.sqrt(n_events)) - t_alpha):.1%}')"
        ),
        md(
            "> In plain words: with 97 events and Bonferroni-adjusted alpha, we need to "
            "see a true effect of over 30 bps before we have 80% power to detect it. "
            "The observed 16 bps estimate falls well below this bar — the study is "
            "genuinely underpowered to certify or rule out the Black Friday effect. "
            "A definitive answer might require another 99 years of data."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — synthetic positive control\n\n"
            "Is the *engine* capable of detecting genuine Thanksgiving anomalies? "
            "We plant known effects in a synthetic tape and confirm recovery."
        ),
        code(
            "effects_to_plant = [0.0, 1.0, 2.0, 4.0]\n"
            "rows_wed, rows_fri = [], []\n"
            "for eff in effects_to_plant:\n"
            "    df_w, _ = data.synthetic_daily(n_years=99, wed_effect=eff, fri_effect=0.0, seed=194)\n"
            "    df_f, _ = data.synthetic_daily(n_years=99, wed_effect=0.0, fri_effect=eff, seed=194)\n"
            "    rw = st.wednesday_comparison(df_w)\n"
            "    rf = st.friday_comparison(df_f)\n"
            "    rows_wed.append({'effect': eff, 'contrast_bps': rw['contrast_bps'],\n"
            "                     'tstat_welch': rw['tstat_welch'],\n"
            "                     'detected': '|t|>2' if abs(rw['tstat_welch'])>2 else 'noise'})\n"
            "    rows_fri.append({'effect': eff, 'contrast_bps': rf['contrast_bps'],\n"
            "                     'tstat_welch': rf['tstat_welch'],\n"
            "                     'detected': '|t|>2' if abs(rf['tstat_welch'])>2 else 'noise'})\n"
            "\n"
            "import pandas as pd\n"
            "tbl_w = pd.DataFrame(rows_wed)\n"
            "tbl_f = pd.DataFrame(rows_fri)\n"
            "print('Wednesday-before engine:')\n"
            "print(tbl_w.round(3).to_string(index=False))\n"
            "print('\\nBlack Friday engine:')\n"
            "print(tbl_f.round(3).to_string(index=False))\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "for ax, tbl, title in zip(axes, [tbl_w, tbl_f], ['Wednesday-before', 'Black Friday']):\n"
            "    ax.scatter(tbl['effect'], tbl['contrast_bps'],\n"
            "               c=[GREEN if abs(t)>2 else RED for t in tbl['tstat_welch']], s=80)\n"
            "    ax.plot(tbl['effect'], tbl['contrast_bps'], c=GREY, lw=1)\n"
            "    ax.axhline(0, c='k', lw=1)\n"
            "    ax.set_xlabel('planted effect (sigma units)'); ax.set_ylabel('recovered contrast (bps)')\n"
            "    ax.set_title(f'{title}: engine recovers planted effects')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            "The engine faithfully recovers planted Thanksgiving effects in synthetic data — "
            "the methodology is sound. At effect = 0 the recovered contrast is noise-level. "
            "The real-tape result for Wednesday (essentially 0 contrast) and for Black Friday "
            "(positive but noisy) are therefore statements about the market, not the method.\n\n"
            "**What would sharpen the answer:** (1) Pooling all pre-holiday days (not just "
            "Thanksgiving) to gain power — see Study 95. (2) Testing the same Black Friday "
            "hypothesis on other large markets with similar holiday structures. (3) Controlling "
            "for the half-day session specifically (Christmas Eve and July 3 are also half-days) "
            "to distinguish Thanksgiving from 'short sessions in general'."
        ),
    ]
    nb = new_notebook(cells=cells, metadata=_meta())
    _write(nb, "02_for_the_quants.ipynb")


def _meta():
    return {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
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
