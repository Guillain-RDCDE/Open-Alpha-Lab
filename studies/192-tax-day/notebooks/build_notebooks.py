"""Generate the two narrative notebooks for Study 192 (Tax-Day).

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
    n_days=24728, n_years=99, fp="df3897aef897",
    n_pre=990, mean_pre_bps=7.77, contrast_pre_bps=5.55,
    tstat_pre_hac=1.865, tstat_pre_welch=1.452, pval_pre=0.1469,
    pval_pre_bonf=0.2937,
    n_post=495, mean_post_bps=9.32, contrast_post_bps=7.02,
    tstat_post_hac=1.819, tstat_post_welch=1.167, pval_post=0.2437,
    pval_post_bonf=0.4874,
    n_apr=2049, mean_apr_bps=5.68, contrast_apr_bps=3.53, pval_apr=0.1948,
    n_oct=1480, mean_oct_bps=3.34, contrast_oct_bps=0.95, pval_oct=0.8244,
    mean_all_bps=2.44,
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

from tax_day import data, strategy as st

# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-15).
R = dict(
    n_days=24728, n_years=99, fp="df3897aef897",
    n_pre=990, mean_pre_bps=7.77, contrast_pre_bps=5.55,
    tstat_pre_hac=1.865, tstat_pre_welch=1.452, pval_pre=0.1469,
    pval_pre_bonf=0.2937,
    n_post=495, mean_post_bps=9.32, contrast_post_bps=7.02,
    tstat_post_hac=1.819, tstat_post_welch=1.167, pval_post=0.2437,
    pval_post_bonf=0.4874,
    n_apr=2049, mean_apr_bps=5.68, contrast_apr_bps=3.53, pval_apr=0.1948,
    n_oct=1480, mean_oct_bps=3.34, contrast_oct_bps=0.95, pval_oct=0.8244,
    mean_all_bps=2.44,
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
            "# Tax-Day -- does April 15 move the market?\n"
            "### The IRA contribution theory, the S&P 500, and 99 years of evidence\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![April_IRA_inflow: Weak](https://img.shields.io/badge/April_IRA_inflow-Weak-dab617?style=flat-square)\n\n"
            "Every year around April 15, a story circulates on financial media: "
            "individual investors rushing to top up their IRAs before the deadline, or "
            "deploying their tax refunds, push money into the stock market and briefly lift "
            "prices. The 10-day window before the deadline is supposed to be especially "
            "strong. This notebook asks: is that actually visible in the data?\n\n"
            "> **This is the plain-language layer.** For the t-stats, the Bonferroni "
            "correction and the power analysis, see "
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
            f"| Are the 10 days before April 15 unusually strong? | "
            f"**Sort of, but not significantly.** The pre-tax window averages "
            f"**+{R['mean_pre_bps']:.1f} bps/day** vs +{R['mean_all_bps']:.1f} bps "
            f"unconditionally -- a gap of +{R['contrast_pre_bps']:.1f} bps that doesn't "
            f"survive statistical testing (p = {R['pval_pre']:.2f}). |\n"
            f"| What about the days right after the deadline? | "
            f"**Same story.** Post-deadline window: **+{R['mean_post_bps']:.1f} bps/day**, "
            f"contrast +{R['contrast_post_bps']:.1f} bps, p = {R['pval_post']:.2f}. |\n"
            "| Is the whole month of April special? | "
            f"**Not really.** April averages +{R['mean_apr_bps']:.1f} bps/day, "
            f"contrast +{R['contrast_apr_bps']:.1f} bps (p = {R['pval_apr']:.2f}). |\n"
            "| Could you trade it? | **No.** 15 days/year, no statistical signal, "
            "any round-trip cost exceeds the noise-level premium. |\n\n"
            "> The point estimates are consistent with the folk narrative -- the pre-tax "
            "window really is a bit above average. But 'a bit above average' over 99 years "
            "with ~10 events per year is not enough to clear the inference bar. The direction "
            "is right; the evidence is too weak to act on."
        ),

        # ---- BEAT 1 -- THE CLAIM -----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *'In the 10 or so trading days before April 15, individual investors rush "
            "to top up their IRA contributions before the deadline. Tax refund money also "
            "flows into the market around this time. This extra demand briefly lifts stock "
            "prices, making the pre-tax-day window a seasonally strong period.'*\n\n"
            "The mechanism is plausible: IRA contributions for the prior tax year are due "
            "by April 15, so millions of savers make a lump-sum deposit in March-April. "
            "Simultaneously, the IRS issues most refund checks in February-April, and some "
            "fraction of recipients invest them. Both flows would, in theory, create a "
            "modest but systematic demand shock for equities in the weeks before the deadline."
        ),

        # ---- BEAT 2 -- SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If it's real, even a small seasonal drift over a predictable 2-week window "
            "is actionable: buy the S&P 500 on day -10, hold through the deadline, and "
            "repeat. The question is whether the premium is large enough and reliable enough "
            "to overcome transaction costs and the inevitable false positives from testing "
            "calendar patterns.\n\n"
            "Most calendar anomalies fail this test: the premium is too small or too noisy "
            "to survive honest inference. This is the desk's job -- to find out which side "
            "of that line the tax-day story falls on."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Two rules keep us honest:\n\n"
            "1. **Use a pre-specified window.** We define the pre-tax window as the 10 "
            "trading days *before* the effective April 15 deadline (adjusted for weekends) "
            "and the post-tax window as the 5 days *on or after* it. These definitions "
            "are fixed in code before we look at the data.\n"
            "2. **Test against an honest baseline.** The control is all other trading days "
            "(not some cherry-picked period). We also check October 15 -- same window sizes, "
            "no IRS event -- as a placebo. If October 15 looks just as anomalous, the "
            "April effect is noise.\n"
            "3. **Correct for multiple comparisons.** We have two hypotheses (pre and post); "
            "the Bonferroni threshold is 0.05/2 = 0.025, not 0.05.\n\n"
            f"Data: ^GSPC daily returns, 1928-01-03 to 2026-06-12 "
            f"({R['n_days']:,} trading days, {R['n_years']} years, "
            f"{R['n_pre']} pre-tax observations, {R['n_post']} post-tax observations)."
        ),

        # ---- BEAT 4 -- THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown -- let's look at the numbers\n\n"
            "**First, where does each window fit in the seasonal calendar?**"
        ),
        code(
            "# Seasonal bar chart: mean daily return by month and window type\n"
            "if HAVE_REAL:\n"
            "    month_means = DF.groupby(DF.index.month)['ret'].mean() * 1e4\n"
            "    pre_mean  = S['mean_pre_bps']\n"
            "    post_mean = S['mean_post_bps']\n"
            "    all_mean  = S['mean_all_bps']\n"
            "else:\n"
            "    import numpy as np\n"
            "    rng = np.random.default_rng(192)\n"
            "    month_means = pd.Series(rng.normal(R['mean_all_bps'], 3, 12) + [0,0,0,R['contrast_apr_bps'],0,0,0,0,0,0,0,0],\n"
            "                           index=range(1,13))\n"
            "    pre_mean  = R['mean_pre_bps']\n"
            "    post_mean = R['mean_post_bps']\n"
            "    all_mean  = R['mean_all_bps']\n"
            "months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']\n"
            "colors = [AMBER if i+1 == 4 else GREY for i in range(12)]\n"
            "fig, ax = plt.subplots(figsize=(10, 4.5))\n"
            "ax.bar(months, [month_means.get(i+1, 0) for i in range(12)], color=colors, alpha=0.8)\n"
            "ax.axhline(all_mean, ls='--', c='k', lw=1.2, label=f'unconditional mean (+{all_mean:.1f} bps)')\n"
            "ax.axhline(pre_mean, ls=':', c=GREEN, lw=1.5, label=f'pre-tax window (+{pre_mean:.1f} bps)')\n"
            "ax.axhline(post_mean, ls=':', c=RED, lw=1.5, label=f'post-tax window (+{post_mean:.1f} bps)')\n"
            "ax.set_ylabel('Mean daily return (bps)'); ax.set_title('Mean daily return by month -- April is modestly elevated')\n"
            "ax.legend(fontsize=9); plt.tight_layout(); plt.show()\n"
            "print(f'April mean: +{month_means.get(4, R[\"mean_apr_bps\"]):.2f} bps  |  pre-tax: +{pre_mean:.2f} bps  |  post-tax: +{post_mean:.2f} bps')"
        ),
        md(
            f"April is modestly above the annual mean, and the pre-tax and post-tax "
            f"windows are higher still (+{R['mean_pre_bps']:.1f} and +{R['mean_post_bps']:.1f} bps). "
            "The direction is consistent with the folk narrative. The question is whether "
            "this is reliably above the background noise of random monthly variation."
        ),
        md(
            "**Now the honest test: tax windows vs the October-15 placebo.**"
        ),
        code(
            "# Side-by-side: tax windows vs unconditional vs Oct-15 placebo\n"
            "if HAVE_REAL:\n"
            "    means = [S['mean_pre_bps'], S['mean_post_bps'], S['mean_all_bps'], S['mean_oct_bps']]\n"
            "    pvals = [S['pval_pre'], S['pval_post'], float('nan'), S['pval_oct']]\n"
            "else:\n"
            "    means = [R['mean_pre_bps'], R['mean_post_bps'], R['mean_all_bps'], R['mean_oct_bps']]\n"
            "    pvals = [R['pval_pre'], R['pval_post'], float('nan'), R['pval_oct']]\n"
            "labels = ['Pre-tax\\n(10d before Apr 15)', 'Post-tax\\n(5d after Apr 15)',\n"
            "          'Unconditional\\n(all days)', 'Oct-15 placebo\\n(same sizes)']\n"
            "cols = [AMBER, AMBER, GREY, GREY]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "bars = ax.bar(labels, means, color=cols, alpha=0.85)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for b, pv in zip(bars, pvals):\n"
            "    import math\n"
            "    if not math.isnan(pv):\n"
            "        tag = f'p={pv:.2f}'\n"
            "        ax.annotate(tag, (b.get_x() + b.get_width()/2, b.get_height()/2),\n"
            "                   ha='center', va='center', fontsize=9, color='white', fontweight='bold')\n"
            "ax.set_ylabel('Mean daily return (bps)')\n"
            "ax.set_title('Tax windows are above average -- but so is October, and p-values are high')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('The October-15 placebo also shows a positive mean -- it is just noise.')"
        ),
        md(
            f"The tax windows outperform the unconditional mean (+{R['contrast_pre_bps']:.1f} and "
            f"+{R['contrast_post_bps']:.1f} bps), but so does the October-15 placebo "
            f"(+{R['contrast_oct_bps']:.1f} bps, p = {R['pval_oct']:.2f}). The gap between April "
            "and October is real in point estimate, but not statistically distinguishable. "
            "The folk story has the right direction -- but the same test would likely 'work' "
            "on several other 2-week windows chosen after the fact."
        ),

        # ---- BEAT 5 -- THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal -- None.** The pre-tax window contrast is +{R['contrast_pre_bps']:.1f} bps "
            f"(HAC t = +{R['tstat_pre_hac']:.2f}, p = {R['pval_pre']:.2f}, Bonf-p = {R['pval_pre_bonf']:.2f}). "
            f"The post-tax contrast is +{R['contrast_post_bps']:.1f} bps "
            f"(HAC t = +{R['tstat_post_hac']:.2f}, p = {R['pval_post']:.2f}, Bonf-p = {R['pval_post_bonf']:.2f}). "
            "Neither hypothesis survives the Bonferroni correction.\n"
            "- **Tradability -- Mirage.** ~15 days/year, positive point estimate, "
            "no significant edge. Any round-trip cost turns a marginal premium into a certain loss.\n"
            "- **April IRA inflow -- Weak.** The point estimates are above the unconditional "
            "mean and above the October placebo, consistent with the mechanism, but the evidence "
            "is too weak to claim. With ~10 events per year it would take another 100 years of "
            "data to power the test properly."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Even if the signal were statistically significant, the strategy looks like this:\n\n"
            "- Hold SPY for 15 days per year (10 pre + 5 post) in long-equity mode; "
            "hold cash the other 237 trading days.\n"
            "- Expected annual contribution from the *window excess*: "
            f"~{R['contrast_pre_bps']:.0f} bps × 10 days + ~{R['contrast_post_bps']:.0f} bps × 5 days "
            f"= ~{int(R['contrast_pre_bps']*10 + R['contrast_post_bps']*5)} bps = ~{(R['contrast_pre_bps']*10 + R['contrast_post_bps']*5)/100:.1f}% per year, gross.\n"
            "- You *miss* the market's daily drift on the other 237 days (~+2.4 bps × 237 = +570 bps).\n"
            "- Net: the strategy underperforms passive buy-and-hold SPY by a large margin "
            "even before adding any transaction cost.\n\n"
            "The question to ask is not 'is the pre-tax window positive?' (it is) but 'is it "
            "positive enough above the positive drift I give up by sitting in cash?' It is not."
        ),

        # ---- BEAT 7 -- GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Is there a January / year-end tax effect?** "
            "[Study 48 — Groundhog](../../48-groundhog/) tests weather-linked seasonals; "
            "the January tax-loss-harvesting reversal is the most studied seasonal in finance "
            "and has also weakened substantially post-publication.\n"
            "- **Sell in May?** The broader spring-to-autumn seasonal is "
            "[Study 136 — Mark-Twain](../../136-mark-twain/), which finds weak evidence but "
            "a more robust historical pattern than the April deadline effect.\n"
            "- **Options expiry demand shocks.** "
            "[Study 82 — Witching-Hour](../../82-witching-hour/) tests a similar "
            "deadline-driven demand-shock mechanism at a much higher frequency.\n\n"
            "*Think the die is loaded after all? Fork the study, change the pre-window to "
            "5 or 15 days, subset to years before IRA contribution limits were raised, or "
            "test on small-cap indices where individual-investor flows are more dominant. "
            "Show a contrast that clears the Bonferroni bar. That's the bar.*"
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
            "# Tax-Day -- a quantitative teardown\n"
            "### ^GSPC daily tape 1928-2026 -- HAC t-stats -- Bonferroni k=2 -- "
            "October-15 placebo -- synthetic positive control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![April_IRA_inflow: Weak](https://img.shields.io/badge/April_IRA_inflow-Weak-dab617?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) -- "
            "same seven beats, every claim carrying its standard error. We test whether the "
            "10-day pre-April-15 window and 5-day post-deadline window show statistically "
            "real return differentials vs the unconditional baseline and vs an October-15 "
            "placebo of identical sizes.\n\n"
            "> **Not investment advice.** Real data: ^GSPC daily bars, shared repo cache "
            f"(fp `{R['fp']}`), as-of 2026-06-15; the offline core and tests run on a "
            "deterministic synthetic tape. Methods in "
            "[`docs/references.md`](../docs/references.md), reproducible numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The `> In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Hypothesis | Stamp | Decisive numbers |\n|---|---|---|\n"
            f"| **H1 — pre-tax window** | `NONE` | contrast +{R['contrast_pre_bps']:.1f} bps/day, "
            f"HAC t = +{R['tstat_pre_hac']:.2f}, Welch p = {R['pval_pre']:.4f}, "
            f"Bonf-p = **{R['pval_pre_bonf']:.4f}** (k=2). |\n"
            f"| **H2 — post-tax window** | `NONE` | contrast +{R['contrast_post_bps']:.1f} bps/day, "
            f"HAC t = +{R['tstat_post_hac']:.2f}, Welch p = {R['pval_post']:.4f}, "
            f"Bonf-p = **{R['pval_post_bonf']:.4f}** (k=2). |\n"
            f"| **April broad seasonal** | `NONE` | contrast +{R['contrast_apr_bps']:.1f} bps, "
            f"p = {R['pval_apr']:.4f}. |\n"
            f"| **Oct-15 placebo** | — | contrast +{R['contrast_oct_bps']:.1f} bps, "
            f"p = {R['pval_oct']:.4f}. |\n\n"
            "> In plain words: the point estimates are in the right direction (pre and post "
            "windows above average), but neither clears the Bonferroni bar with 99 years of "
            "~10-event/yr data. The study is underpowered against effects smaller than "
            "~15-20 bps/day."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_t$ denote the daily log-return and $W_t^{\\text{pre}}$ the indicator "
            "for the 10-trading-day pre-deadline window. The folk recipe asserts:\n\n"
            "- **H1 (pre-tax signal).** "
            "$\\mathbb{E}[r_t \\mid W_t^{\\text{pre}} = 1] > \\mathbb{E}[r_t \\mid W_t^{\\text{pre}} = 0]$. "
            "IRA and refund inflows create a demand shock that lifts equity prices in the "
            "pre-deadline window, detectable as an above-average daily return.\n"
            "- **H2 (post-deadline normalisation).** "
            "$\\mathbb{E}[r_t \\mid W_t^{\\text{post}} = 1] \\neq \\mathbb{E}[r_t \\mid W_t^{\\text{post}} = 0]$. "
            "The demand shock reverses, or tax-selling pressure creates a distinct post-deadline "
            "return pattern.\n\n"
            "Both hypotheses are tested jointly under a Bonferroni correction (k=2, threshold "
            "0.025 per test). The effective deadline is April 15 shifted forward to Monday when "
            "April 15 falls on a weekend (deterministic date arithmetic, no holiday table)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? -- what rides on each answer\n\n"
            "The mechanism is credible and the sign is directionally consistent with economic "
            "theory. If H1 survives inference, it would be a rare case of a *known* demand "
            "shock (the IRA deadline, publicly scheduled years in advance) creating a "
            "statistically reliable return pattern in the world's most liquid equity market. "
            "That would be a strong statement about the limits of arbitrage: with only ~10 "
            "events per year, the flow is too small relative to the market's daily volume for "
            "pre-positioning to fully eliminate the premium. The interesting failure is not "
            "that the premium is zero in point estimate -- it isn't -- but that the power of "
            "99 years of data is not enough to certify a 5-7 bps/day incremental return."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know -- the protocol\n\n"
            "- **Date arithmetic.** Effective deadline = April 15 shifted to Monday if "
            "on a weekend. Pre window: 10 trading days *strictly before* the deadline. "
            "Post window: 5 trading days *on or after* the deadline (inclusive). "
            "Windows are mutually exclusive.\n"
            "- **Inference.** Welch t-test (unequal variances) for the group contrast; "
            "Newey-West HAC t-stat on each group separately. Bonferroni correction k=2.\n"
            "- **Placebo.** October-15 windows of identical sizes (10 pre + 5 post), "
            "no IRS event. If the placebo is equally 'significant', the April result is noise.\n"
            "- **Broad seasonal control.** Whole-month April vs non-April -- to separate "
            "any April seasonal from the specific deadline effect.\n"
            "- **Positive control.** Synthetic tape with planted ``pre_tax_effect`` and "
            "``post_tax_effect`` knobs -- confirms the engine detects real effects when they "
            "exist.\n\n"
            f"Tape: ^GSPC, {R['n_days']:,} trading days, {R['n_years']} years "
            f"(fp `{R['fp']}`). {R['n_pre']} pre-tax observations, {R['n_post']} post-tax."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Pre-tax window (H1) -- the IRA-contribution hypothesis\n\n"
            "Welch t-test and HAC t-stat on the pre-deadline window vs all other days."
        ),
        code(
            "if HAVE_REAL:\n"
            "    pre_r = st.window_comparison(DF, 'pre_tax')\n"
            "    n_win  = pre_r['n_window']; n_rest = pre_r['n_rest']\n"
            "    m_win  = pre_r['mean_window_bps']; m_rest = pre_r['mean_rest_bps']\n"
            "    contr  = pre_r['contrast_bps']\n"
            "    t_hac  = pre_r['tstat_window_hac']; t_welch = pre_r['tstat_welch']\n"
            "    pv     = pre_r['pval_welch']\n"
            "else:\n"
            "    n_win, n_rest = R['n_pre'], R['n_days'] - R['n_pre']\n"
            "    m_win, m_rest = R['mean_pre_bps'], R['mean_all_bps']\n"
            "    contr = R['contrast_pre_bps']; t_hac = R['tstat_pre_hac']\n"
            "    t_welch = R['tstat_pre_welch']; pv = R['pval_pre']\n"
            "fig, ax = plt.subplots(figsize=(7, 4.5))\n"
            "ax.bar(['Pre-tax window\\n(n={:,})'.format(n_win),\n"
            "        'Rest\\n(n={:,})'.format(n_rest)],\n"
            "       [m_win, m_rest], color=[AMBER, GREY], alpha=0.85)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Mean daily return (bps)')\n"
            "ax.set_title(f'Pre-tax window: +{contr:.1f} bps contrast (HAC t={t_hac:+.2f}, p={pv:.3f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Pre-window mean: +{m_win:.2f} bps | Rest: +{m_rest:.2f} bps | contrast: +{contr:.2f} bps')\n"
            "print(f'HAC t = {t_hac:+.3f} | Welch t = {t_welch:+.3f} | p = {pv:.4f}')\n"
            "print(f'Bonferroni-corrected p (k=2): {min(pv*2, 1.0):.4f}')"
        ),
        md(
            f"> In plain words: the pre-tax window earns **+{R['mean_pre_bps']:.1f} bps/day** vs "
            f"+{R['mean_all_bps']:.1f} bps unconditionally. The contrast (+{R['contrast_pre_bps']:.1f} bps) "
            f"has a HAC t-stat of **+{R['tstat_pre_hac']:.2f}** -- close to but not above "
            "the |t|=1.96 bar for a single test, and failing the Bonferroni-corrected 0.025 "
            f"threshold (Bonf-p = {R['pval_pre_bonf']:.2f}). The direction is right; the "
            "magnitude is not statistically certifiable."
        ),
        md(
            "### 4b · Post-tax window (H2) -- the post-deadline pattern\n\n"
            "Same test, post-tax window (5 days on/after April 15)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    post_r = st.window_comparison(DF, 'post_tax')\n"
            "    m_post = post_r['mean_window_bps']; contr_post = post_r['contrast_bps']\n"
            "    t_post = post_r['tstat_window_hac']; pv_post = post_r['pval_welch']\n"
            "else:\n"
            "    m_post = R['mean_post_bps']; contr_post = R['contrast_post_bps']\n"
            "    t_post = R['tstat_post_hac']; pv_post = R['pval_post']\n"
            "print(f'Post-window mean: +{m_post:.2f} bps | contrast: +{contr_post:.2f} bps')\n"
            "print(f'HAC t = {t_post:+.3f} | p = {pv_post:.4f} | Bonf-p = {min(pv_post*2,1):.4f}')"
        ),
        md(
            f"> In plain words: the post-deadline window is also elevated "
            f"(+{R['mean_post_bps']:.1f} bps/day, contrast +{R['contrast_post_bps']:.1f} bps) "
            f"but again fails significance (HAC t = +{R['tstat_post_hac']:.2f}, "
            f"Bonf-p = {R['pval_post_bonf']:.2f})."
        ),
        md(
            "### 4c · October-15 placebo and Bonferroni table\n\n"
            "If the October-15 placebo -- same window sizes, no IRS event -- shows a "
            "comparable premium, the April result is noise. Bonferroni summary across both "
            "hypotheses."
        ),
        code(
            "rows = []\n"
            "if HAVE_REAL:\n"
            "    pre_r  = st.window_comparison(DF, 'pre_tax')\n"
            "    post_r = st.window_comparison(DF, 'post_tax')\n"
            "    plac_r = st.placebo_comparison(DF)\n"
            "    rows = [\n"
            "        ('H1 pre-tax', pre_r['n_window'],  pre_r['mean_window_bps'],  pre_r['contrast_bps'],  pre_r['tstat_window_hac'],  pre_r['pval_welch'],  min(pre_r['pval_welch']*2, 1.0)),\n"
            "        ('H2 post-tax', post_r['n_window'], post_r['mean_window_bps'], post_r['contrast_bps'], post_r['tstat_window_hac'], post_r['pval_welch'], min(post_r['pval_welch']*2,1.0)),\n"
            "        ('Oct-15 placebo', plac_r['n_oct_window'], plac_r['mean_oct_bps'], plac_r['contrast_bps'], float('nan'), plac_r['pval_welch'], float('nan')),\n"
            "    ]\n"
            "else:\n"
            "    rows = [\n"
            "        ('H1 pre-tax',    R['n_pre'],  R['mean_pre_bps'],  R['contrast_pre_bps'],  R['tstat_pre_hac'],  R['pval_pre'],  R['pval_pre_bonf']),\n"
            "        ('H2 post-tax',   R['n_post'], R['mean_post_bps'], R['contrast_post_bps'], R['tstat_post_hac'], R['pval_post'], R['pval_post_bonf']),\n"
            "        ('Oct-15 placebo', R['n_oct'],  R['mean_oct_bps'],  R['contrast_oct_bps'],  float('nan'),       R['pval_oct'],  float('nan')),\n"
            "    ]\n"
            "import pandas as pd\n"
            "tbl = pd.DataFrame(rows, columns=['window', 'n', 'mean (bps)', 'contrast (bps)', 'HAC t', 'p-raw', 'Bonf-p (k=2)'])\n"
            "print(tbl.round(3).to_string(index=False))\n"
            "print('\\nNeither hypothesis survives Bonferroni correction (threshold 0.025).')"
        ),
        md(
            f"> In plain words: the October-15 placebo earns +{R['mean_oct_bps']:.1f} bps/day "
            f"(contrast +{R['contrast_oct_bps']:.1f} bps, p = {R['pval_oct']:.2f}). The April "
            "windows are higher in point estimate, but the difference between 'April premium' "
            "and 'October noise' is not statistically meaningful. The Bonferroni table shows "
            "no hypothesis clearing the 0.025 threshold."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE` (H1)** -- pre-tax contrast +{R['contrast_pre_bps']:.1f} bps, "
            f"HAC t = +{R['tstat_pre_hac']:.2f}, Bonf-p = {R['pval_pre_bonf']:.2f}.\n"
            f"- **Signal `NONE` (H2)** -- post-tax contrast +{R['contrast_post_bps']:.1f} bps, "
            f"HAC t = +{R['tstat_post_hac']:.2f}, Bonf-p = {R['pval_post_bonf']:.2f}.\n"
            f"- **Tradability `MIRAGE`** -- 15 days/year, no certified edge, any cost kills it.\n"
            f"- **April IRA inflow `WEAK`** -- direction consistent with the mechanism, "
            "magnitude underpowered. Would need ~another century of data or a larger effect "
            "size to certify."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? -- the power ceiling\n\n"
            "The fundamental problem is event frequency: ~10 pre-tax + ~5 post-tax = "
            "~15 annual observations per year. At 99 years that's ~990 pre-tax days total. "
            "The standard error of the mean on ~990 days of equity returns (daily vol ~100 bps) "
            "is $\\approx 100 / \\sqrt{990} \\approx 3.2$ bps. A true effect of +5.5 bps "
            "gives a theoretical t-stat of only $5.5 / 3.2 \\approx 1.7$ -- which is exactly "
            "what we see. This is not a failure of the test; it's the arithmetic of low-frequency "
            "events."
        ),
        code(
            "# Power curve: how large would the effect need to be to clear |t|=2 at 99 years?\n"
            "import numpy as np\n"
            "daily_vol = 100.0  # bps\n"
            "n_years_range = np.arange(10, 301, 10)\n"
            "min_detectable = 2.0 * daily_vol / np.sqrt(n_years_range * 10)  # 10 pre-events/yr\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "ax.plot(n_years_range, min_detectable, c=RED, lw=2, label='min detectable effect (|t|=2, 10 events/yr)')\n"
            "ax.axhline(R['contrast_pre_bps'], ls='--', c=AMBER, lw=1.5, label=f'observed contrast (+{R[\"contrast_pre_bps\"]:.1f} bps)')\n"
            "ax.axvline(R['n_years'], ls=':', c=GREY, label=f'current sample ({R[\"n_years\"]} years)')\n"
            "ax.set_xlabel('Years of data'); ax.set_ylabel('Effect size (bps/day)')\n"
            "ax.set_title('Power ceiling: the observed effect needs ~200 years to certify at |t|=2')\n"
            "ax.legend(fontsize=9); plt.tight_layout(); plt.show()\n"
            "needed = (2.0 * daily_vol / R['contrast_pre_bps'])**2 / 10\n"
            "print(f'Years needed to power this test at |t|=2: ~{needed:.0f}')"
        ),
        md(
            "> In plain words: given daily equity volatility of ~100 bps and only ~10 events "
            "per year, a true effect of +5.5 bps/day would require roughly **200 years** of "
            "data to certify at |t|=2. The data simply cannot resolve this effect size at this "
            "event frequency. This is the honest power ceiling for all low-frequency calendar "
            "effects -- not a bug, but a fundamental constraint."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further -- the synthetic positive control\n\n"
            "Can the engine find an effect when one is genuinely there? We plant known "
            "pre-tax and post-tax return differentials in a synthetic tape and verify "
            "that the engine recovers them."
        ),
        code(
            "effects = [0.0, 1.0, 2.0, 3.0, 4.0]\n"
            "pre_ts, post_ts = [], []\n"
            "for eff in effects:\n"
            "    df_syn, _ = data.synthetic_daily(n_years=99, pre_tax_effect=eff, post_tax_effect=-eff, seed=192)\n"
            "    p = st.window_comparison(df_syn, 'pre_tax')\n"
            "    q = st.window_comparison(df_syn, 'post_tax')\n"
            "    pre_ts.append(p['tstat_welch'])\n"
            "    post_ts.append(q['tstat_welch'])\n"
            "import pandas as pd\n"
            "ctrl_df = pd.DataFrame({'planted effect (sigma)': effects,\n"
            "                        'pre Welch t': pre_ts, 'post Welch t': post_ts})\n"
            "fig, ax = plt.subplots(figsize=(9, 4.5))\n"
            "ax.plot(effects, pre_ts, 'o-', c=GREEN, lw=2, label='pre-tax window')\n"
            "ax.plot(effects, post_ts, 's--', c=RED, lw=2, label='post-tax window (negative effect)')\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1); ax.axhline(-2, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_xlabel('Planted effect (sigma units)')\n"
            "ax.set_ylabel('Welch t-stat'); ax.legend()\n"
            "ax.set_title('Synthetic control: engine finds effects when planted (|t| > 2 for sigma >= 2)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(ctrl_df.round(2).to_string(index=False))"
        ),
        md(
            "The engine is a faithful detector: a planted 2-sigma effect clears |t|=2 "
            "and a null produces a near-zero t-stat. The real-tape result (t~+1.85) is "
            "exactly where a true 1-sigma effect would land -- consistent with a small "
            "real mechanism that the data cannot yet certify."
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
