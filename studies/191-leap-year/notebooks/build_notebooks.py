"""Generate the two narrative notebooks for Study 191 (Leap-Year).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). Synthetic
figures run anywhere, offline and deterministic; real-tape cells use the shared
Shiller cache if present and otherwise quote the frozen headline numbers in ``R``
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


# Frozen real-tape headline numbers — mirror of docs/results.md (as-of 2026-06-15).
R = dict(
    # Modern era 1928-2025
    n_total=98, n_leap=25, n_non_leap=73,
    mean_leap_pct=8.77, mean_non_leap_pct=7.75, mean_all_pct=8.01,
    std_leap_pct=16.53, std_non_leap_pct=19.14, std_all_pct=18.43,
    contrast_pct=1.02, tstat_welch=0.255, pval_welch=0.7999,
    tstat_leap_hac=3.05, pval_leap_bonf=1.00,
    # Term cycle
    term_y1_mean=8.77, term_y2_mean=7.46, term_y3_mean=2.58, term_y4_mean=13.24,
    term_y1_t=3.05, term_y3_t=0.69, term_y4_t=4.61,
    # Sub-period
    contrast_pre1980=4.29, p_pre1980=0.4483,
    contrast_post1980=-2.73, p_post1980=0.6458,
    # Full Shiller (1872-2025)
    n_leap_full=38, contrast_full_pct=0.90, pval_full=0.7811,
    # Fingerprint
    fp_modern="9a990fb006c3", fp_full="248be906654f",
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

from leap_year import data, strategy as st

# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-15).
R = dict(
    n_total=98, n_leap=25, n_non_leap=73,
    mean_leap_pct=8.77, mean_non_leap_pct=7.75, mean_all_pct=8.01,
    std_leap_pct=16.53, std_non_leap_pct=19.14, std_all_pct=18.43,
    contrast_pct=1.02, tstat_welch=0.255, pval_welch=0.7999,
    tstat_leap_hac=3.05, pval_leap_bonf=1.00,
    term_y1_mean=8.77, term_y2_mean=7.46, term_y3_mean=2.58, term_y4_mean=13.24,
    term_y1_t=3.05, term_y3_t=0.69, term_y4_t=4.61,
    contrast_pre1980=4.29, p_pre1980=0.4483,
    contrast_post1980=-2.73, p_post1980=0.6458,
    n_leap_full=38, contrast_full_pct=0.90, pval_full=0.7811,
    fp_modern="9a990fb006c3", fp_full="248be906654f",
)

def _have_real():
    return os.path.exists(data.SHILLER_PATH)

HAVE_REAL = _have_real()

if HAVE_REAL:
    DF = data.load_shiller(start_year=1928)
    S  = st.summarize(DF)
else:
    DF, S = None, None

print("Shiller cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Leap-Year — do stocks know it's a leap year?\n"
            "### The quadrennial folk claim, tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Confound_exposed: Election_year_in_disguise](https://img.shields.io/badge/"
            "Confound_exposed-Election_year_in_disguise-8b949e?style=flat-square)\n\n"
            "Every four years a leap year rolls around — and so does a US presidential election, "
            "and the Summer Olympics. The folk story says this is good (or bad) for stocks: "
            "election-year spending, Olympic optimism, or simply something auspicious about "
            "the extra February 29th. This notebook asks the only question that matters: "
            "**is there actually a leap-year effect, or is it just statistical coincidence?**\n\n"
            "> This is the plain-language layer. Want the t-stats, the Bonferroni table, and the "
            "presidential-cycle breakdown? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Do stocks return more in leap years? | **No.** Leap years averaged "
            f"**{R['mean_leap_pct']:.1f}%** vs **{R['mean_non_leap_pct']:.1f}%** for other years — "
            f"a +{R['contrast_pct']:.1f} pp gap with a p-value of **{R['pval_welch']:.2f}**. "
            "Statistically indistinguishable from noise. |\n"
            "| Is the election-year overlap the real story? | **Also no.** "
            "In the modern era leap years and election years are the *same years* (both "
            "divisible by 4). Election years are not the strongest year-of-term either — "
            f"that title goes to **mid-term years** (+{R['term_y4_mean']:.1f}%). |\n"
            "| Could you trade it? | **No.** One trade per year, with no edge over the "
            "unconditional mean — there is nothing here to execute. |\n\n"
            "> The leap-year premium is an illusion: +1 pp of noise on a backdrop of "
            "18% annual volatility, over 25 observations."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Leap years are special for stocks because they coincide with "
            "US presidential elections — and markets historically do well in election "
            "years — and with the Summer Olympics, which boost consumer spending and "
            "sentiment. Buy in January of a leap year and sell in December.\"*\n\n"
            "It is a cocktail of three overlapping stories:\n\n"
            "1. **The calendar coincidence.** Every leap year (year divisible by 4) is "
            "also a US presidential election year and an Olympics year. They co-occur by "
            "construction — not because they cause each other.\n"
            "2. **The election-year story.** Some almanacs claim election years are "
            "systematically bullish because incumbents stimulate the economy before "
            "voting day.\n"
            "3. **The Olympic story.** Less credible but common on finance social media: "
            "global sporting optimism lifts sentiment.\n\n"
            "All three claims collapse to the same test: does the year-mod-4 == 0 label "
            "predict above-average calendar-year equity returns?"
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "With only **25 leap years** in the modern GSPC era (1928–2025), a genuine "
            "+5 pp annual premium would be barely detectable — and the claimed effect "
            "is much smaller. This is a study where the sample size is fundamentally "
            "limited by the calendar: you cannot get more leap years without waiting "
            "centuries. If the effect were real and large enough to trade, it would be "
            "visible. If it is small enough to be invisible in 97 years of data, "
            "it is too small to matter."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "One rule keeps us honest: **compare the right things**.\n\n"
            "- We use the **Welch t-test** (unequal variances) to compare the mean "
            "calendar-year return in leap years vs non-leap years on the Shiller "
            "S&P 500 dataset going back to 1928.\n"
            "- We apply a **Bonferroni correction** over the three related hypotheses "
            "a researcher could have tested (leap-year premium, election-year premium, "
            "mid-term-year premium). If you test three hypotheses and then pick the "
            "most exciting one, your true false-positive rate is 3× the nominal alpha.\n"
            "- We show the **presidential-cycle breakdown** (year-of-term 1–4) to "
            "expose the structural confound: are leap years special within the cycle, "
            "or just mediocre?\n"
            "- We verify the effect is not a statistical artefact by checking a "
            "**synthetic tape** where we control the planted premium."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the raw comparison: leap vs non-leap years.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    leap_ret = DF.loc[DF['is_leap'], 'ret'] * 100\n"
            "    nonleap_ret = DF.loc[~DF['is_leap'], 'ret'] * 100\n"
            "    lm, nm = leap_ret.mean(), nonleap_ret.mean()\n"
            "    lsd, nsd = leap_ret.std(), nonleap_ret.std()\n"
            "else:\n"
            "    lm, nm = R['mean_leap_pct'], R['mean_non_leap_pct']\n"
            "    lsd, nsd = R['std_leap_pct'], R['std_non_leap_pct']\n"
            "    leap_ret = None; nonleap_ret = None\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "# Left: bar chart of means\n"
            "ax = axes[0]\n"
            "bars = ax.bar(['Leap years\\n(n=25)', 'Non-leap years\\n(n=73)', 'All years\\n(n=98)'],\n"
            "              [lm, nm, R['mean_all_pct']], color=[AMBER, GREY, 'steelblue'], width=0.55)\n"
            "ax.axhline(R['mean_all_pct'], ls='--', c='steelblue', lw=1.5, label='unconditional mean')\n"
            "ax.set_ylabel('Mean calendar-year return (%)')\n"
            "ax.set_title(f'Leap year premium: +{R[\"contrast_pct\"]:.1f} pp — noise (p={R[\"pval_welch\"]:.2f})')\n"
            "ax.legend(fontsize=9)\n"
            "for bar, val in zip(bars, [lm, nm, R['mean_all_pct']]):\n"
            "    ax.text(bar.get_x() + bar.get_width()/2, val + 0.3, f'{val:.1f}%',\n"
            "            ha='center', va='bottom', fontsize=10)\n"
            "# Right: scatter of leap vs non-leap returns\n"
            "ax2 = axes[1]\n"
            "if HAVE_REAL:\n"
            "    ax2.scatter(DF.loc[~DF['is_leap']].index, nonleap_ret, c=GREY, s=20, alpha=0.6, label='non-leap')\n"
            "    ax2.scatter(DF.loc[DF['is_leap']].index, leap_ret, c=AMBER, s=50, zorder=5, label='leap year')\n"
            "    ax2.axhline(lm, ls='--', c=AMBER, lw=1.5)\n"
            "    ax2.axhline(nm, ls='--', c=GREY, lw=1.0)\n"
            "ax2.axhline(0, c='k', lw=1)\n"
            "ax2.set_xlabel('Year'); ax2.set_ylabel('Calendar-year return (%)')\n"
            "ax2.set_title('Leap years (amber) mixed into the full distribution')\n"
            "ax2.legend(fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Leap: {lm:.2f}% ± {lsd:.1f}%   Non-leap: {nm:.2f}% ± {nsd:.1f}%')\n"
            "print(f'With only 25 leap observations, the noise band is ±{lsd/25**0.5:.1f} pp — the +1 pp gap is invisible.')"
        ),
        md(
            f"On the real tape, leap years averaged **+{R['mean_leap_pct']:.2f}%** and "
            f"non-leap years **+{R['mean_non_leap_pct']:.2f}%**. The raw gap is "
            f"**+{R['contrast_pct']:.1f} pp** — but with 25 observations and 16-17% "
            "annual volatility, the standard error is ~3-4 pp. The gap is pure noise.\n\n"
            "> The annual vol alone (~18%) is 18× larger than the claimed premium. "
            "You'd need centuries of data to prove a 1 pp annual edge."
        ),
        md(
            "**Now the presidential-cycle confound.** Leap years *are* election years "
            "(year % 4 == 0). Is the election year itself special within the four-year cycle?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    tc = st.term_cycle_comparison(DF)\n"
            "    means = list(tc['mean_pct'])\n"
            "    tstats = list(tc['tstat_hac'])\n"
            "else:\n"
            "    means = [R['term_y1_mean'], R['term_y2_mean'], R['term_y3_mean'], R['term_y4_mean']]\n"
            "    tstats = [R['term_y1_t'], None, R['term_y3_t'], R['term_y4_t']]\n"
            "labels = ['Year 1\\n(election=leap)', 'Year 2\\n(post-elect)', 'Year 3\\n(mid-term+1)', 'Year 4\\n(mid-term)']\n"
            "colors = [AMBER, GREY, RED, GREEN]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.8))\n"
            "bars = ax.bar(labels, means, color=colors, width=0.55)\n"
            "ax.axhline(R['mean_all_pct'], ls='--', c='steelblue', lw=1.5, label=f'unconditional mean ({R[\"mean_all_pct\"]:.1f}%)')\n"
            "ax.set_ylabel('Mean calendar-year return (%)')\n"
            "ax.set_title('The four-year presidential cycle: year 4 beats year 1 (election=leap)')\n"
            "ax.legend(fontsize=9)\n"
            "for bar, val in zip(bars, means):\n"
            "    ax.text(bar.get_x() + bar.get_width()/2, val + 0.3, f'{val:.1f}%',\n"
            "            ha='center', va='bottom', fontsize=10)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Year 4 (mid-term): {means[3]:.1f}%  vs  Year 1 (election=leap): {means[0]:.1f}%')"
        ),
        md(
            f"Within the four-year cycle, year 1 (election = leap) is **not special**. "
            f"The strongest year is **year 4 (mid-term)** at **+{R['term_y4_mean']:.1f}%**, "
            f"and the weakest is year 3 at **+{R['term_y3_mean']:.1f}%**. The leap-year "
            "label lands squarely in the middle — not even the election-year narrative "
            "holds up within the cycle's own data."
        ),
        md(
            "**Finally, the sub-period reality check.** A genuine structural effect "
            "should be consistent across sub-periods — not reverse sign."
        ),
        code(
            "if HAVE_REAL:\n"
            "    df_pre = DF[DF.index < 1980]\n"
            "    df_post = DF[DF.index >= 1980]\n"
            "    r_pre = st.leap_comparison(df_pre)\n"
            "    r_post = st.leap_comparison(df_post)\n"
            "    c_pre, c_post = r_pre['contrast_pct'], r_post['contrast_pct']\n"
            "    p_pre, p_post = r_pre['pval_welch'], r_post['pval_welch']\n"
            "else:\n"
            "    c_pre, c_post = R['contrast_pre1980'], R['contrast_post1980']\n"
            "    p_pre, p_post = R['p_pre1980'], R['p_post1980']\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "subperiods = ['Pre-1980\\n(n_leap=13)', 'Post-1980\\n(n_leap=12)', 'Full 1928-2025\\n(n_leap=25)']\n"
            "contrasts = [c_pre, c_post, R['contrast_pct']]\n"
            "colors_sp = [AMBER if c > 0 else RED for c in contrasts]\n"
            "ax.bar(subperiods, contrasts, color=colors_sp, width=0.45)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Leap-year premium (pp)')\n"
            "ax.set_title('The \"premium\" reverses sign across halves — a hallmark of noise')\n"
            "for i, (sp, c, p) in enumerate(zip(subperiods, contrasts, [p_pre, p_post, R['pval_welch']])):\n"
            "    ax.text(i, c + (0.3 if c >= 0 else -0.8), f'{c:+.1f}pp\\n(p={p:.2f})',\n"
            "            ha='center', fontsize=9)\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"Pre-1980: **+{R['contrast_pre1980']:.1f} pp** (p = {R['p_pre1980']:.2f}). "
            f"Post-1980: **{R['contrast_post1980']:+.1f} pp** (p = {R['p_post1980']:.2f}). "
            "The sign flips. A real structural effect doesn't do that."
        ),

        # ---- BEAT 5 — VERDICT ------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Welch t = +{R['tstat_welch']:.3f}, p = {R['pval_welch']:.2f}; "
            f"Bonferroni-corrected p = {R['pval_leap_bonf']:.2f}. The +{R['contrast_pct']:.1f} pp "
            "gap is pure noise in a 18% vol world with 25 leap-year observations.\n"
            "- **Tradability — Mirage.** One trade per year, no edge over the unconditional mean. "
            "Annual transaction costs trivially exceed any claimed benefit.\n"
            f"- **Confound exposed.** Leap years = election years in the modern era. Neither is "
            f"special within the four-year presidential cycle — year 4 (mid-term, +{R['term_y4_mean']:.1f}%) "
            f"beats year 1 (election=leap, +{R['term_y1_mean']:.1f}%)."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "The strategy is: on January 1st of every leap year, move to 100% equity; "
            "on January 1st of non-leap years, switch to something else (or reduce equity). "
            "With one rebalance per year and no edge, any transaction cost immediately turns "
            "a coin into a loser. Typical annual rebalancing costs — 5–20 bp/year of "
            "transaction costs — exceed the claimed +1 pp edge within a confidence interval "
            "that spans ±4 pp. The strategy is not investable."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Is there *anything* in the four-year cycle?** The year-4 (mid-term) "
            "pattern has the most empirical support, but it too fails Bonferroni "
            "correction once you account for testing four year-types. That's "
            "[Study 81 — Four-Year-Itch](../../81-four-year-itch/).\n"
            "- **Other calendar folklore.** [Study 48 — Groundhog](../../48-groundhog/) "
            "and [Study 163 — Friday-13th](../../163-friday-13th/) test comparable "
            "folk claims with the same null result.\n"
            "- **January effect.** Monthly calendar seasonality — not quadrennial — is "
            "[Study 25 — January-Effect](../../25-january-effect/), which has a WEAK "
            "signal in small-caps but NONE in large-caps.\n\n"
            "*Think the leap-year effect is real? Fork this, restrict to a specific "
            "sector, time-window, or international market, and show a Bonferroni-clean "
            "signal. That's the bar.*"
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
            "# Leap-Year — a quantitative teardown\n"
            "### Shiller S&P 500 (1928-2025) · Welch t-test · HAC inference · "
            "Bonferroni correction · presidential-cycle confound\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Confound_exposed: Election_year_in_disguise](https://img.shields.io/badge/"
            "Confound_exposed-Election_year_in_disguise-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim now carrying its standard error.* We test whether "
            "leap-year calendar returns differ from non-leap years on 98 years of Shiller data, "
            "control for the presidential-cycle confound, and apply Bonferroni correction over "
            "the three quadrennial hypotheses a data-miner could have tested.\n\n"
            "> **Not investment advice.** Shiller S&P 500 monthly data staged at "
            "`_cache/shiller_sp500.parquet`; December-to-December price return; "
            "1928–2025; as-of 2026-06-15. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The `In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT + "\nfrom quantlab import analytics, stats\n"),

        # ---- BEAT 0 — VERDICT -----------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Welch *t* = **{R['tstat_welch']:+.3f}**, "
            f"p = **{R['pval_welch']:.2f}**; Bonferroni p = **{R['pval_leap_bonf']:.2f}**. "
            f"Contrast = **+{R['contrast_pct']:.1f} pp** on a ±18% vol background with "
            f"n = {R['n_leap']} leap years. |\n"
            f"| **Tradability** | `MIRAGE` | One trade per year; no exploitable edge over "
            f"the unconditional mean (+{R['mean_all_pct']:.1f}%/yr). Any cost kills it. |\n"
            f"| **Confound** | `ELECTION YEAR IN DISGUISE` | In the modern era leap years "
            f"and election years are the *same years* (year % 4 == 0). The election-year "
            f"story itself is not the strongest year-of-term (year 4 = +{R['term_y4_mean']:.1f}% "
            f"> year 1 = +{R['term_y1_mean']:.1f}%). |\n\n"
            "> In plain words: there is no leap-year effect. There is barely a "
            "presidential-cycle effect once you correct for multiple comparisons."
        ),

        # ---- BEAT 1 — CLAIM --------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_Y$ be the calendar-year S&P 500 return and $L_Y = \\mathbf{1}[Y \\mod 4 = 0]$ "
            "the leap-year indicator. The claim is:\n\n"
            "- **H₁ (leap-year premium):** $\\mathbb{E}[r_Y \\mid L_Y = 1] > "
            "\\mathbb{E}[r_Y \\mid L_Y = 0]$.\n"
            "- **H₂ (election-year premium):** same test, re-labelled 'election year'. "
            "Identical to H₁ in the modern era (year % 4 == 0 for both).\n"
            "- **H₃ (mid-term-year premium):** $\\mathbb{E}[r_Y \\mid Y \\mod 4 = 3]$ is the "
            "highest of the four year-types — the rival four-year-itch claim.\n\n"
            "We test all three with Bonferroni correction ($k = 3$, corrected $\\alpha = 0.017$)."
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what? — the power problem\n\n"
            "The leap-year test is structurally low-power. With annual vol $\\sigma \\approx 18\\%$ "
            "and $n_{\\text{leap}} = 25$, the standard error of the leap-year mean is "
            "$\\approx \\sigma / \\sqrt{n} \\approx 3.6$ pp. To achieve $|t| \\geq 2$, the true "
            "premium would need to exceed $\\approx 7$ pp — a claim so large it would have been "
            "spotted decades ago. The +1 pp observed contrast is 7× smaller than the detection "
            "threshold. This is not a close call: the study is correctly powered to rule out "
            "effects of the claimed magnitude."
        ),

        # ---- BEAT 3 — PROTOCOL -----------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Data.** Shiller monthly S&P 500 (December-to-December price return, "
            "1928–2025, n = 98 annual observations).\n"
            "- **Primary test.** Welch t-test (unequal variances) on mean contrast "
            "leap vs non-leap; HAC Newey-West t-stat on each group separately.\n"
            "- **Control.** Presidential-cycle year-of-term (1–4) breakdown as the "
            "structural confound, and the unconditional all-years mean as the baseline.\n"
            "- **Multiple comparisons.** Bonferroni correction over $k = 3$ quadrennial "
            "hypotheses; corrected $\\alpha = 0.05/3 \\approx 0.017$.\n"
            "- **Sub-period check.** Split at 1980 (two sub-periods of 13/12 leap years).\n"
            "- **Positive control.** Synthetic annual tape with tunable planted premium, "
            "to confirm the engine has power when the signal exists."
        ),

        # ---- BEAT 4 — TEARDOWN -----------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Leap vs non-leap: the primary test\n\n"
            "Welch t-test on calendar-year returns; HAC t-stats (Newey-West) on "
            "each group's mean-vs-zero."
        ),
        code(
            "if HAVE_REAL:\n"
            "    lc = st.leap_comparison(DF)\n"
            "    lm, nm = lc['mean_leap_pct'], lc['mean_non_leap_pct']\n"
            "    ct, tw, pw = lc['contrast_pct'], lc['tstat_welch'], lc['pval_welch']\n"
            "    hl, hn = lc['tstat_leap_hac'], lc['tstat_non_leap_hac']\n"
            "else:\n"
            "    lm, nm, ct = R['mean_leap_pct'], R['mean_non_leap_pct'], R['contrast_pct']\n"
            "    tw, pw = R['tstat_welch'], R['pval_welch']\n"
            "    hl, hn = R['tstat_leap_hac'], None\n"
            "hl_str = f'{hl:+.3f}' if hl is not None else 'n/a'\n"
            "print(f'Leap years (n={R[\"n_leap\"]}):     mean = {lm:+.2f}%  HAC t = {hl_str}')\n"
            "print(f'Non-leap  (n={R[\"n_non_leap\"]}):  mean = {nm:+.2f}%')\n"
            "print(f'Contrast:       {ct:+.2f} pp')\n"
            "print(f'Welch t:        {tw:+.4f}')\n"
            "print(f'p-value:        {pw:.4f}')\n"
            "print(f'Bonf-p (k=3):  {min(pw*3,1):.4f}')"
        ),
        md(
            f"> In plain words: the {R['n_leap']}-observation leap-year mean is "
            f"**{R['mean_leap_pct']:.2f}%**, with a standard error of "
            f"~{R['std_leap_pct'] / R['n_leap']**0.5:.1f} pp. "
            f"The +{R['contrast_pct']:.2f} pp gap (Welch t = +{R['tstat_welch']:.3f}) "
            f"is well within one standard error of zero — statistically invisible."
        ),
        md(
            "### 4b · Presidential-cycle breakdown\n\n"
            "Year-of-term 1 = election = leap (in the modern era). If the election-year "
            "premium were real, year 1 should stand out. It doesn't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    tc = st.term_cycle_comparison(DF)\n"
            "    means_tc = list(tc['mean_pct'])\n"
            "    tstats_tc = list(tc['tstat_hac'])\n"
            "    ns_tc = list(tc['n'])\n"
            "else:\n"
            "    means_tc = [R['term_y1_mean'], R['term_y2_mean'], R['term_y3_mean'], R['term_y4_mean']]\n"
            "    tstats_tc = [R['term_y1_t'], None, R['term_y3_t'], R['term_y4_t']]\n"
            "    ns_tc = [25, 25, 24, 24]\n"
            "lbl = ['Y1\\nelection=leap', 'Y2\\npost-elect', 'Y3\\nmid-term+1', 'Y4\\nmid-term']\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))\n"
            "# Left: means\n"
            "col = [AMBER, GREY, RED, GREEN]\n"
            "axes[0].bar(lbl, means_tc, color=col, width=0.55)\n"
            "axes[0].axhline(R['mean_all_pct'], ls='--', c='steelblue', lw=1.5,\n"
            "               label=f'unconditional ({R[\"mean_all_pct\"]:.1f}%)')\n"
            "axes[0].set_ylabel('Mean calendar-year return (%)')\n"
            "axes[0].set_title('Year-of-term: year 4 wins, year 1 (leap) is middling')\n"
            "axes[0].legend(fontsize=9)\n"
            "# Right: HAC t-stats\n"
            "ts_plot = [t if t is not None else 0 for t in tstats_tc]\n"
            "axes[1].bar(lbl, ts_plot, color=col, width=0.55)\n"
            "for s in (2, -2): axes[1].axhline(s, ls='--', c=GREY, lw=1)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "axes[1].set_ylabel('HAC t-stat (mean vs zero)')\n"
            "axes[1].set_title('All four years significantly positive (equities trend up)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Note: HAC t >> 2 for all groups means each group\\'s mean is > 0 (equity trend),\\n'\n"
            "      '      NOT that the groups differ from each other.')"
        ),
        md(
            f"> In plain words: the large HAC t-stats for each group measure whether "
            "the group mean is significantly above zero — not whether it beats the others. "
            f"Year 4 (mid-term) at **{R['term_y4_mean']:.1f}%** is the strongest. "
            f"Year 1 (election = leap) at **{R['term_y1_mean']:.1f}%** is the second-best — "
            "not special, just average."
        ),
        md(
            "### 4c · Multiple-comparisons reckoning (Bonferroni, k = 3)\n\n"
            "If we tested all three quadrennial hypotheses and then picked the strongest "
            "one, the true false-positive rate is 3× the nominal alpha. Nothing survives."
        ),
        code(
            "if HAVE_REAL:\n"
            "    mc = st.multiple_comparisons(DF)\n"
            "else:\n"
            "    mc = None\n"
            "if mc is not None:\n"
            "    print(mc[['hypothesis','n_group','mean_group_pct','contrast_pct','pval_raw','pval_bonferroni']].to_string(index=False))\n"
            "else:\n"
            "    print(f'Leap year premium:              p_raw={R[\"pval_welch\"]:.4f}  p_bonf={R[\"pval_leap_bonf\"]:.4f}')\n"
            "    print(f'Election year premium:          p_raw={R[\"pval_welch\"]:.4f}  p_bonf={R[\"pval_leap_bonf\"]:.4f} (identical to leap)')\n"
            "    print(f'Mid-term year premium (year 4): p_raw=0.1021           p_bonf=0.3064')"
        ),
        md(
            f"Bonferroni-corrected p for the leap-year hypothesis: "
            f"**{R['pval_leap_bonf']:.2f}** (raw p = {R['pval_welch']:.2f}, × 3 = {R['pval_welch']*3:.2f}). "
            "Even the most credible hypothesis (mid-term year premium, raw p ≈ 0.10) "
            "falls to Bonferroni p ≈ 0.31."
        ),
        md(
            "### 4d · The synthetic positive control — the engine has power\n\n"
            "Does the engine detect a leap-year premium when one genuinely exists? "
            "Plant a large effect in a synthetic annual series and confirm it is recovered."
        ),
        code(
            "prems = [0.0, 0.03, 0.06, 0.09, 0.12, -0.06]\n"
            "rows = []\n"
            "for p in prems:\n"
            "    df_s, _ = data.synthetic_annual(n_years=300, leap_premium=p, seed=191)\n"
            "    r = st.leap_comparison(df_s)\n"
            "    rows.append({'planted': p*100, 'contrast': r['contrast_pct'],\n"
            "                 'tstat': r['tstat_welch'], 'pval': r['pval_welch']})\n"
            "syn = pd.DataFrame(rows)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "col_syn = [GREEN if abs(r['tstat']) >= 2 else RED for _, r in syn.iterrows()]\n"
            "ax.bar([f'{p:+.0f}pp' for p in syn['planted']], syn['tstat'], color=col_syn, width=0.55)\n"
            "for s in (2, -2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('Planted leap-year premium (pp/yr)')\n"
            "ax.set_ylabel('Welch t-stat')\n"
            "ax.set_title('The engine has power: it detects a large planted effect (n=300 synthetic years)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(syn.to_string(index=False))"
        ),
        md(
            "The engine correctly recovers the planted effect when it is large enough. "
            "The real-data result (+1.02 pp, t = +0.255) is consistent with a true premium "
            "near zero — the engine would easily detect anything as large as 5+ pp. "
            "The real market simply has no leap-year premium at this scale."
        ),

        # ---- BEAT 5 — VERDICT -----------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — Welch *t* = {R['tstat_welch']:+.3f}, p = {R['pval_welch']:.2f}; "
            f"Bonferroni p = {R['pval_leap_bonf']:.2f}. Contrast = +{R['contrast_pct']:.2f} pp "
            f"on a {R['std_all_pct']:.0f}% vol background, n = {R['n_leap']} leap years. "
            "Sub-period sign flip confirms noise.\n"
            f"- **Tradability `MIRAGE`** — no edge over the unconditional mean "
            f"({R['mean_all_pct']:.1f}%/yr); any annual cost kills a one-trade-per-year strategy.\n"
            f"- **Confound `ELECTION YEAR IN DISGUISE`** — leap = election year in modern era. "
            f"Election year (year 1) at {R['term_y1_mean']:.1f}% < mid-term year (year 4) at "
            f"{R['term_y4_mean']:.1f}%; neither clears Bonferroni."
        ),

        # ---- BEAT 6 — TRADABILITY -------------------------------------------
        md(
            "## 6 · Could you trade it? — the turnover arithmetic\n\n"
            "The strategy has exactly one signal per four years: a binary bet on "
            "the full calendar year. Even at zero transaction costs it earns the "
            "leap-year mean (+8.77%/yr in leap years, +7.75%/yr otherwise) — "
            "a 1 pp edge that does not exist statistically and is 18× smaller than "
            "annual vol. Annual rebalancing costs alone (5–20 bp/yr) consume this "
            "claimed premium within the confidence interval. The strategy is not "
            "worth implementing even in a frictionless world."
        ),

        # ---- BEAT 7 — GOING FURTHER -----------------------------------------
        md(
            "## 7 · Going further — extensions and sister studies\n\n"
            "- **Four-year itch ([Study 81](../../81-four-year-itch/)):** the "
            "presidential-cycle year-of-term effect in full. Year 4 (mid-term) "
            "is the strongest, but does it survive the same Bonferroni bar? "
            "That study has the answer.\n"
            "- **January Effect ([Study 25](../../25-january-effect/)):** monthly "
            "calendar seasonality — a finer-grained but better-powered test.\n"
            "- **Groundhog Day ([Study 48](../../48-groundhog/)):** another folk "
            "calendar claim tested against the same Shiller tape.\n"
            "- **International markets.** Does the Olympic-year / leap-year story "
            "show up outside the US? The confound (election cycle) is US-specific, "
            "so if the effect were truly from Olympics rather than elections it should "
            "appear in non-US markets. Fork this and check — the bar is the same."
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
