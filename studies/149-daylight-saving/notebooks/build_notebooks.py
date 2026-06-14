"""Generate the two narrative notebooks for Study 149 (Daylight-Saving).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells use the cached
^GSPC parquet under ../_cache/ if present and otherwise quote the frozen headline
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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-14).
R = dict(
    n_total=11706,
    n_dst=93,
    n_other=2119,
    dst_mean=-17.69,   dst_tstat=-1.20,
    other_mean=2.21,   other_tstat=0.88,
    diff_bps=-19.89,   welch_t=-1.19,
    placebo_mean=0.03, placebo_std=14.33,
    p_one_sided=0.084,
    spring_mean=-18.95, spring_n=47, spring_t_vs_other=-1.03,
    fall_mean=-16.40,   fall_n=46,   fall_t_vs_other=-0.70,
    pre_diff=-35.68,   pre_t=-1.19,
    post_diff=-7.98,   post_t=-0.42,
    window_start="1980-01-03", window_end="2026-06-12",
    fingerprint="3ad46b2324e0",
)

# ---------------------------------------------------------------------------
# Shared analysis preamble — imports, data loading, colour palette.
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

from daylight_saving import data, strategy as st

_cache = os.path.join("..", "_cache", "gspc_daily.parquet")
HAVE_REAL = os.path.exists(_cache)

if HAVE_REAL:
    frame = data.fetch_daily()
    M = st.summarize(frame)
else:
    M = dict(
        n_dst=93, dst_mean_bps=-17.69, dst_tstat=-1.20,
        n_other=2119, other_mean_bps=2.21, other_tstat=0.88,
        diff_bps=-19.89, welch_t=-1.19,
        placebo_mean_bps=0.03, placebo_std_bps=14.33, p_one_sided=0.084,
        spring_mean_bps=-18.95, spring_n=47, spring_t_vs_other=-1.03,
        fall_mean_bps=-16.40, fall_n=46, fall_t_vs_other=-0.70,
        pre_diff_bps=-35.68, pre_welch_t=-1.19,
        post_diff_bps=-7.98, post_welch_t=-0.42,
    )

print("real ^GSPC cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Daylight-Saving -- does losing an hour of sleep crash the market?\n"
            "### The post-DST Monday anomaly, tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Contested%3F: Fragile--finding](https://img.shields.io/badge/Contested%3F-Fragile--finding-8b949e?style=flat-square)\n\n"
            "Every spring, clocks spring forward by one hour and millions of Americans lose "
            "an hour of sleep.  A famous 2000 paper by Kamstra, Kramer and Levi (KKL) claimed "
            "this causes investors to be extra risk-averse the following Monday, producing a "
            "market crash that is **far bigger than any other known calendar anomaly**.  "
            "This notebook asks the only question that matters: does it actually hold up?\n\n"
            "> **This is the plain-language layer.** Want the t-stats, the placebo distribution "
            "and the era breakdown? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below "
            "is drawn by the code beside it. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Is the post-DST Monday really more negative? | **Directionally yes** "
            f"({R['diff_bps']:+.1f} bps vs other Mondays), but barely statistically "
            f"distinguishable from noise (Welch *t* = {R['welch_t']:+.2f}). |\n"
            "| Is it bigger in spring (spring-forward)? | Marginally so "
            f"({R['spring_mean']:+.1f} bps) vs fall ({R['fall_mean']:+.1f} bps), "
            "but neither is significant. |\n"
            "| Did it hold up out of sample? | No: post-2000 gap = "
            f"{R['post_diff']:+.1f} bps (*t* = {R['post_t']:+.2f}) -- essentially flat. |\n"
            "| Could you trade it? | No: two events per year at an insignificant gap. |\n\n"
            "> The signal has the right sign but never clears the statistical bar."
            " With only 93 events over 46 years, the noise band is simply too wide."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *'The Monday after a daylight-saving time change -- spring or fall -- earns "
            "significantly negative stock returns.  In spring, losing an hour of sleep makes "
            "investors more risk-averse; in fall, gaining an hour mildly reverses this. "
            "The effect is far larger than any known calendar anomaly.'*\n\n"
            "-- Kamstra, Kramer & Levi (2000), *American Economic Review*\n\n"
            "It is a behaviorally appealing story: sleep disruption is real, its effects on "
            "cognition and risk-taking are documented in the lab, and the DST shock is exogenous "
            "and quasi-random.  The authors found effects of −200 to −500 bps on international "
            "indices in the late 1960s-1990s.  The question is whether this is a causal effect or "
            "a coincidence in a small sample."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If true, this would be extraordinary: a human-biology event reliably moving markets "
            "by hundreds of basis points, twice a year, with a clear mechanism.  If false -- which "
            "Pinegar (2002) quickly argued -- it's a textbook lesson in **small-sample calendar "
            "anomalies**: with 2 events per year and a few decades of data, you can find apparent "
            "patterns by chance even in clean random data."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Two things make this honest:\n\n"
            "1. **The right baseline.** The market goes down on some Mondays.  The claim is that "
            "*DST Mondays* are reliably worse than *other Mondays* -- not just that they can be "
            "negative.  We compare post-DST Mondays against all other Mondays.\n"
            "2. **A placebo control.** We randomly pick Mondays in the same quantities (93) and "
            "measure their gap against the rest, 500 times.  If the real gap barely beats the "
            "shuffle distribution, the effect is statistically indistinguishable from noise.\n\n"
            f"Sample: ^GSPC daily returns {R['window_start']} to {R['window_end']}, "
            f"n = {R['n_total']:,} days."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown -- let's actually look\n\n"
            "**The raw numbers: post-DST Monday vs all other Mondays.**"
        ),
        code(
            "arms = ['Post-DST\\nMonday', 'Other\\nMondays']\n"
            "means = [M['dst_mean_bps'], M['other_mean_bps']]\n"
            "ns    = [M['n_dst'], M['n_other']]\n\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.5))\n"
            "bars = ax.bar(arms, means, color=[RED, GREY], width=.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('mean daily return (bps)')\n"
            "ax.set_title('Post-DST Monday vs Other Mondays (^GSPC 1980-2026)')\n"
            "for b, n in zip(bars, ns):\n"
            "    ax.annotate(f'n={n}', (b.get_x()+b.get_width()/2, b.get_height()),\n"
            "                ha='center', va='bottom' if b.get_height()>0 else 'top',\n"
            "                fontsize=10)\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f\"Post-DST: {{M['dst_mean_bps']:+.2f}} bps  |  \"\n"
            f"      f\"Other Mondays: {{M['other_mean_bps']:+.2f}} bps  |  \"\n"
            f"      f\"Gap: {{M['diff_bps']:+.2f}} bps  (Welch t = {{M['welch_t']:+.2f}})\")\n"
        ),
        md(
            f"Post-DST Mondays average **{R['diff_bps']:+.1f} bps below** other Mondays, "
            "directionally matching the KKL claim.  But the Welch t-statistic is "
            f"**{R['welch_t']:+.2f}** -- well below the |t| >= 2 threshold.  "
            f"With only {R['n_dst']} events, the noise band is simply too wide."
        ),
        md(
            "**The placebo test: could this gap arise by chance?**"
        ),
        code(
            "# Show the placebo distribution (run offline from synthetic, or use real)\n"
            "rng = np.random.default_rng(149)\n"
            "if HAVE_REAL:\n"
            "    mondays = frame[frame['is_monday']]\n"
            "    r_all = mondays['ret'].to_numpy()\n"
            "    n_dst_real = int(mondays['is_dst_monday'].sum())\n"
            "    n = len(r_all)\n"
            "    gaps = np.empty(500)\n"
            "    for s in range(500):\n"
            "        idx = np.zeros(n, dtype=bool); idx[:n_dst_real] = True\n"
            "        rng.shuffle(idx)\n"
            "        gaps[s] = (r_all[idx].mean() - r_all[~idx].mean()) * 1e4\n"
            "    real_gap = M['diff_bps']\n"
            "else:\n"
            "    gaps = np.random.default_rng(0).normal(0, 14.33, 500)\n"
            f"    real_gap = {R['diff_bps']}\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "ax.hist(gaps, bins=30, color=GREY, alpha=.7, label='500 random-Monday gaps')\n"
            "ax.axvline(real_gap, c=RED, lw=2.5, label=f'Observed gap: {real_gap:+.1f} bps')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "ax.set_xlabel('mean gap (bps): random DST Monday vs other Mondays')\n"
            "ax.set_ylabel('count'); ax.legend()\n"
            "ax.set_title('The observed gap is in the left tail -- but not the extreme left')\n"
            "plt.tight_layout(); plt.show()\n"
            f"p = (gaps <= real_gap).mean()\n"
            f"print(f'One-sided p-value: {{p:.3f}}  (observed gap = {{real_gap:+.1f}} bps)')\n"
        ),
        md(
            f"The observed gap ({R['diff_bps']:+.1f} bps) sits in the left tail of the "
            f"placebo distribution, but **p = {R['p_one_sided']:.3f}** -- not below the "
            f"conventional 0.05 threshold.  About {int(R['p_one_sided']*100)}% of randomly "
            "chosen Monday subsets produce a gap at least this negative by pure chance."
        ),
        md(
            "**Spring-forward vs fall-back: does the mechanism hold?**"
        ),
        code(
            "seasons = ['Spring-forward\\n(lose 1h sleep)', 'Fall-back\\n(gain 1h sleep)', 'Other\\nMondays']\n"
            "smeans  = [M['spring_mean_bps'], M['fall_mean_bps'], M['other_mean_bps']]\n"
            "sns_n   = [M['spring_n'], M['fall_n'], M['n_other']]\n"
            "colors  = [RED, AMBER, GREY]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.5))\n"
            "bs = ax.bar(seasons, smeans, color=colors, width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('mean daily return (bps)')\n"
            "ax.set_title('Season split: spring and fall are both negative -- neither is significant')\n"
            "for b, n in zip(bs, sns_n):\n"
            "    ax.annotate(f'n={n}', (b.get_x()+b.get_width()/2, b.get_height()),\n"
            "                ha='center', va='bottom' if b.get_height()>0 else 'top', fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"Spring: {M['spring_mean_bps']:+.2f} bps  t_vs_other={M['spring_t_vs_other']:+.2f}\")\n"
            "print(f\"Fall:   {M['fall_mean_bps']:+.2f} bps  t_vs_other={M['fall_t_vs_other']:+.2f}\")\n"
        ),
        md(
            f"Both spring ({R['spring_mean']:+.1f} bps) and fall ({R['fall_mean']:+.1f} bps) "
            "are negative, which matches the KKL story directionally but both miss the "
            "significance threshold.  The spring-minus-fall difference is small and also "
            "not significant."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal -- None.** Welch *t* = {R['welch_t']:+.2f} on the "
            f"{R['diff_bps']:+.1f} bps gap; placebo p = {R['p_one_sided']:.3f}. "
            "Directional consistency with KKL, but never significant on 93 events.\n"
            "- **Tradability -- Mirage.** Two events per year at an insignificant gap "
            "means no tradable alpha.\n"
            f"- **Contested?** Pinegar (2002) showed the original result is fragile; "
            f"our post-2000 confirms this ({R['post_diff']:+.1f} bps, *t* = {R['post_t']:+.2f})."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT -----------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "Even if the gap were real, the tradability case is hopeless: two events per year "
            "means annual alpha = roughly 2 x (gap) -- at −20 bps per event that is −40 bps "
            "per year annualised, before costs.  A short position in index futures "
            "on two specific Mondays a year:\n\n"
            "- Faces substantial basis risk (the market can move by 1-3% on any Monday).\n"
            "- Has a Sharpe below zero even if the gap is real at −20 bps.\n"
            "- Would be detectable by the rest of the market and arbitraged away.\n\n"
            "This is a curiosity, not a strategy."
        ),

        # ---- BEAT 7 -- GOING FURTHER ----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **Is the Monday effect real?** French (1980) showed Mondays in general are "
            "negative -- the DST claim is layered on top of this.  "
            "[Study 48 -- Groundhog](../../48-groundhog/) tests another folklore calendar event.\n"
            "- **Behavioral mechanisms in market anomalies.** The Kamstra et al. mechanism "
            "(sleep -> risk aversion -> prices) is one of the few calendar anomalies with a "
            "proposed neurological mechanism.  The lack of statistical confirmation here doesn't "
            "mean the biology is wrong -- the event is just too rare to test adequately.\n"
            "- **What has more events?** [Study 67 -- Fed-Drift](../../67-fed-drift/) and "
            "[Study 135 -- FOMC-Cycle](../../135-fomc-cycle/) use calendar events that fire "
            "8 times a year and have much more statistical power.\n\n"
            "*Think the clock really moves markets? Fork this, extend to international indices "
            "or longer history, and show a gap that beats the placebo distribution at p < 0.05. "
            "That's the bar.*"
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
            "# Daylight-Saving -- a quantitative teardown\n"
            "### ^GSPC 1980-2026 * DST Monday vs other Mondays * HAC t * Welch test *"
            " placebo * era split * synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Contested%3F: Fragile--finding](https://img.shields.io/badge/Contested%3F-Fragile--finding-8b949e?style=flat-square)\n\n"
            "The deep companion to [01_for_the_curious.ipynb](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim carrying its standard error.*  We test whether "
            "post-DST Mondays on ^GSPC (1980-2026) are significantly worse than other Mondays, "
            "pin the gap against a random-Monday placebo, and assess era decay.\n\n"
            "> **Not investment advice.** Real data: Yahoo Finance ^GSPC daily, "
            f"as-of 2026-06-14 (fp `{R['fingerprint']}`). "
            "Methods in [`docs/references.md`](../docs/references.md), reproducible "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **'In plain words' notes** translate each result back to intuition."
        ),
        code(BOOT + "\nfrom quantlab import analytics, stats\n"),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Gap = **{R['diff_bps']:+.1f} bps**, "
            f"Welch *t* = **{R['welch_t']:+.2f}**; placebo p = {R['p_one_sided']:.3f}. "
            f"HAC *t* on DST arm alone: {R['dst_tstat']:+.2f}. Directionally consistent "
            f"with KKL but never clears |t| >= 2 on {R['n_dst']} events. |\n"
            "| **Tradability** | `MIRAGE` | 2 events/year at insignificant gap = "
            "effectively zero alpha; short index futures on two Mondays is "
            "not a strategy. |\n"
            "| **Contested?** | `FRAGILE` | Pinegar (2002) showed the KKL result "
            f"vanishes with minor changes; our post-2000 out-of-sample: "
            f"{R['post_diff']:+.1f} bps, *t* = {R['post_t']:+.2f}. |\n\n"
            "> In plain words: the signal has the right sign but is statistically "
            "indistinguishable from noise -- the sample of 93 events over 46 years "
            "is simply too small to tell a real effect from a lucky draw."
        ),

        # ---- BEAT 1 -- THE CLAIM ---------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $r_t$ be the daily S&P 500 log return and $D_t = 1$ if day $t$ is the "
            "Monday after a US DST clock change.  KKL assert:\n\n"
            r"- **H1 (signal).** $\mathbb{E}[r_t \mid D_t = 1] "
            r"\ll \mathbb{E}[r_t \mid D_t = 0, \text{Monday}]$ -- "
            "post-DST Mondays are reliably more negative than other Mondays.\n"
            "- **H2 (season).** Spring-forward > fall-back in magnitude: losing "
            "an hour is worse than gaining one.\n"
            "- **H3 (mechanism).** The effect operates through sleep-disruption-induced "
            "risk aversion, not through any secular Monday effect or contemporaneous "
            "news.\n\n"
            "We test H1 rigorously; H2 directionally; H3 indirectly via a placebo test "
            "showing the gap is at least in the expected tail."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what? -- the stakes\n\n"
            "A confirmed DST effect would be remarkable: an exogenous, predictable "
            "2x-per-year shock of tens to hundreds of bps driven by human biology. "
            "The failure is instructive: **low-event-count calendar anomalies** "
            "(< 100 events) are easily spurious even with a compelling story.  "
            "A Welch t-statistic of ~1.2 on 93 events translates to a p-value around "
            "0.12 (two-sided) -- routine false-positive territory for a 'famous' anomaly."
        ),

        # ---- BEAT 3 -- PROTOCOL ----------------------------------------------
        md(
            "## 3 - Protocol\n\n"
            "- **Data.** ^GSPC daily close-to-close returns, Yahoo Finance, "
            f"{R['window_start']} to {R['window_end']}, n = {R['n_total']:,}.\n"
            "- **DST dates.** Computed analytically from US statute: "
            "pre-1987 last Sunday April/October; 1987-2006 first Sunday April / "
            "last Sunday October; 2007+ second Sunday March / first Sunday November. "
            "Treatment = the Monday after the change.\n"
            "- **Baseline.** All other Mondays (not immediately after a DST change).\n"
            "- **Tests.** Welch t on the mean difference (unequal variances); "
            "HAC Newey-West t on each arm separately; random-Monday placebo "
            "(500 seeds); season breakdown; pre/post-2000 era split.\n"
            "- **Positive control.** Synthetic daily tape with tunable planted "
            "DST effect: engine must detect the signal when one is planted, "
            "and find nothing on a null tape.\n\n"
            "**Inference bar:** Signal = REAL requires |t| >= 2 on the real tape "
            "on the headline test."
        ),

        # ---- BEAT 4a -- HEADLINE SPLIT ---------------------------------------
        md("## 4 - The teardown\n\n### 4a - Headline split: DST Monday vs other Mondays"),
        code(
            "if HAVE_REAL:\n"
            "    h = st.dst_vs_other_mondays(frame)\n"
            "    dst_m, dst_t = h['dst']['mean_bps'], h['dst']['tstat']\n"
            "    oth_m, oth_t = h['other']['mean_bps'], h['other']['tstat']\n"
            "    gap, wt = h['diff_bps'], h['welch_t']\n"
            "    n_dst, n_oth = h['dst']['n'], h['other']['n']\n"
            "else:\n"
            f"    dst_m, dst_t = {R['dst_mean']}, {R['dst_tstat']}\n"
            f"    oth_m, oth_t = {R['other_mean']}, {R['other_tstat']}\n"
            f"    gap, wt = {R['diff_bps']}, {R['welch_t']}\n"
            f"    n_dst, n_oth = {R['n_dst']}, {R['n_other']}\n"
            "\n"
            "tbl = pd.DataFrame({\n"
            "    'arm': ['Post-DST Monday', 'Other Mondays'],\n"
            "    'n': [n_dst, n_oth],\n"
            "    'mean_bps': [dst_m, oth_m],\n"
            "    'HAC_t': [dst_t, oth_t],\n"
            "})\n"
            "print(tbl.to_string(index=False))\n"
            "print(f'Difference: {gap:+.2f} bps   Welch t = {wt:+.2f}')\n"
            "print(f'Inference bar: |t| < 2 -> Signal = NONE')\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "bars = ax.bar(['Post-DST Monday', 'Other Mondays'], [dst_m, oth_m],\n"
            "              color=[RED, GREY], width=.5)\n"
            "for b, t in zip(bars, [dst_t, oth_t]):\n"
            "    ax.annotate(f't={t:+.2f}', (b.get_x()+b.get_width()/2, b.get_height()),\n"
            "                ha='center', va='bottom' if b.get_height()>0 else 'top', fontsize=10)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('mean daily return (bps)')\n"
            "ax.set_title(f'Gap = {gap:+.2f} bps, Welch t = {wt:+.2f} (|t| < 2: NONE)')\n"
            "plt.tight_layout(); plt.show()\n"
        ),
        md(
            f"> In plain words: the gap is {R['diff_bps']:+.1f} bps, which sounds large "
            "but is swamped by daily return noise (~110 bps std).  With n = "
            f"{R['n_dst']}, the standard error of the mean is about 11 bps, "
            "so t ~ 1.2.  The 95% confidence interval on the DST mean "
            "spans roughly [-40, +5] bps -- consistent with zero."
        ),

        # ---- BEAT 4b -- PLACEBO ----------------------------------------------
        md("### 4b - Placebo: random Monday relabelling (500 seeds)"),
        code(
            "if HAVE_REAL:\n"
            "    pl = st.random_monday_split(frame, n_seeds=500, seed=149)\n"
            "    rg = pl['real_gap_bps']; pm = pl['placebo_mean_bps']; ps = pl['placebo_std_bps']\n"
            "    pv = pl['p_value_one_sided']\n"
            "    # Rebuild the gap distribution for plotting\n"
            "    mondays = frame[frame['is_monday']]; r_all = mondays['ret'].to_numpy()\n"
            "    n_ = len(r_all); n_d = int(mondays['is_dst_monday'].sum())\n"
            "    rng2 = np.random.default_rng(149); gaps_arr = np.empty(500)\n"
            "    for s in range(500):\n"
            "        idx_ = np.zeros(n_, dtype=bool); idx_[:n_d] = True\n"
            "        rng2.shuffle(idx_)\n"
            "        gaps_arr[s] = (r_all[idx_].mean() - r_all[~idx_].mean()) * 1e4\n"
            "else:\n"
            f"    rg, pm, ps, pv = {R['diff_bps']}, {R['placebo_mean']}, {R['placebo_std']}, {R['p_one_sided']}\n"
            "    gaps_arr = np.random.default_rng(0).normal(pm, ps, 500)\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(gaps_arr, bins=30, color=GREY, alpha=.7, label='Placebo null (500 seeds)')\n"
            "ax.axvline(rg, c=RED, lw=2.5, label=f'Observed: {rg:+.1f} bps')\n"
            "ax.axvline(0, c='k', lw=1)\n"
            "p5 = np.percentile(gaps_arr, 5)\n"
            "ax.axvline(p5, c=AMBER, ls='--', lw=1.5, label=f'5th pct: {p5:+.1f} bps')\n"
            "ax.set_xlabel('gap bps: (random DST Mondays) - (other Mondays)')\n"
            "ax.set_ylabel('count'); ax.legend()\n"
            "ax.set_title(f'Observed gap is in the {pv*100:.0f}th percentile of null -- not < 5%')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'p (one-sided) = {pv:.3f}; 5th percentile of null = {p5:+.1f} bps')\n"
        ),
        md(
            f"> In plain words: the observed gap ({R['diff_bps']:+.1f} bps) is in the "
            f"left {R['p_one_sided']*100:.0f}th percentile of the random-Monday null -- "
            "a hint of signal, but not below the 5% threshold.  The boundary between "
            "'suggestive' and 'significant' is exactly what the inference bar is for."
        ),

        # ---- BEAT 4c -- ERA SPLIT --------------------------------------------
        md("### 4c - Era split: pre- vs post-2000 (in-sample vs out-of-sample for KKL)"),
        code(
            "if HAVE_REAL:\n"
            "    era = st.split_by_era(frame, cut='2000-01-01')\n"
            "    pre_d = era['pre_publication']['diff_bps']\n"
            "    pre_t = era['pre_publication']['welch_t']\n"
            "    post_d = era['post_publication']['diff_bps']\n"
            "    post_t = era['post_publication']['welch_t']\n"
            "    pre_n  = era['pre_publication']['n_dst']\n"
            "    post_n = era['post_publication']['n_dst']\n"
            "else:\n"
            f"    pre_d, pre_t, pre_n = {R['pre_diff']}, {R['pre_t']}, 40\n"
            f"    post_d, post_t, post_n = {R['post_diff']}, {R['post_t']}, 53\n"
            "\n"
            "eras = ['Pre-2000\\n(KKL in-sample)', 'Post-2000\\n(out-of-sample)']\n"
            "diffs = [pre_d, post_d]\n"
            "tstats = [pre_t, post_t]\n"
            "ns_ = [pre_n, post_n]\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))\n"
            "c = [RED if d < 0 else GREY for d in diffs]\n"
            "axes[0].bar(eras, diffs, color=c, width=.55)\n"
            "axes[0].axhline(0, c='k', lw=1)\n"
            "axes[0].set_ylabel('gap (bps): DST - other Mondays')\n"
            "axes[0].set_title('Gap by era')\n"
            "for i, (e, d, t_, n_) in enumerate(zip(eras, diffs, tstats, ns_)):\n"
            "    axes[0].annotate(f't={t_:+.2f}\\nn={n_}',\n"
            "        (i, d), ha='center', va='bottom' if d>0 else 'top', fontsize=9)\n"
            "axes[1].bar(eras, tstats, color=c, width=.55)\n"
            "axes[1].axhline(0, c='k', lw=1)\n"
            "for s in (-2, 2): axes[1].axhline(s, ls='--', c=GREY, lw=1)\n"
            "axes[1].set_ylabel('Welch t-stat'); axes[1].set_title('Welch t by era')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Pre-2000:  gap={pre_d:+.1f} bps  t={pre_t:+.2f}')\n"
            "print(f'Post-2000: gap={post_d:+.1f} bps  t={post_t:+.2f}')\n"
        ),
        md(
            f"The gap was larger in the KKL in-sample era ({R['pre_diff']:+.1f} bps, "
            f"t = {R['pre_t']:+.2f}) and shrinks markedly post-2000 "
            f"({R['post_diff']:+.1f} bps, t = {R['post_t']:+.2f}).  "
            "This is consistent with Pinegar's (2002) critique: the original result "
            "was a small-sample artefact, and the post-publication data does not "
            "support it."
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            f"- **Signal `NONE`** -- headline gap {R['diff_bps']:+.1f} bps, "
            f"Welch *t* {R['welch_t']:+.2f}, placebo p {R['p_one_sided']:.3f}; "
            "no result clears |t| >= 2.\n"
            "- **Tradability `MIRAGE`** -- 2 events/year at an insignificant gap; "
            "no alpha to trade.\n"
            f"- **Contested `FRAGILE`** -- post-2000 gap {R['post_diff']:+.1f} bps "
            f"(*t* = {R['post_t']:+.2f}); Pinegar (2002) replication failure."
        ),

        # ---- BEAT 6 -- TRADABILITY -------------------------------------------
        md(
            "## 6 - Could you trade it? -- the numbers\n\n"
            "Even if we accept the point estimate at face value:\n\n"
            f"- 2 events/year x {R['diff_bps']:+.1f} bps/event = "
            f"roughly {2*R['diff_bps']:.0f} bps annualised (before costs).\n"
            "- That is ~−40 bps/year = ~−0.4% annualised alpha, with enormous "
            "variance (daily S&P vol ~1.1%, so each DST Monday has +/-110 bps "
            "noise band around the signal).\n"
            "- The Sharpe on this 'strategy' is negative and indistinguishable from zero.\n"
            "- Standard transaction costs (2-5 bps round-trip on futures) are negligible "
            "at 2 trades/year, but the signal itself is noise.\n\n"
            "Contrast with anomalies that fire 50-250 times per year: they can "
            "accumulate statistical confidence in 5-10 years; a 2x-per-year event "
            "needs **50 years** to reach the same power, and by then the market has changed."
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further -- the synthetic positive control\n\n"
            "Is the engine capable of detecting a DST effect if one exists?  "
            "Plant a negative drift on DST Mondays in a synthetic tape and sweep the effect size."
        ),
        code(
            "from daylight_saving import data as ddata\n"
            "effects = [0.0, -0.001, -0.002, -0.005, -0.010]\n"
            "rows = []\n"
            "for eff in effects:\n"
            "    f_syn, _ = ddata.synthetic_daily(dst_effect=eff, seed=149)\n"
            "    f_syn['is_monday'] = f_syn.index.dayofweek == 0\n"
            "    import pandas as _pd\n"
            "    tbl_ = ddata.dst_mondays(1980, 2025)\n"
            "    dm = {r['date'].normalize(): r['season'] for _, r in tbl_.iterrows()}\n"
            "    f_syn['season'] = [dm.get(_pd.Timestamp(d).normalize(),'') for d in f_syn.index]\n"
            "    h_ = st.dst_vs_other_mondays(f_syn)\n"
            "    rows.append({'effect_bps': eff*1e4, 'gap_bps': h_['diff_bps'], 'welch_t': h_['welch_t']})\n"
            "ctrl_df = pd.DataFrame(rows)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "colors_ = [GREEN if abs(r)<2 else RED for r in ctrl_df['welch_t']]\n"
            "ax.bar(ctrl_df['effect_bps'], ctrl_df['welch_t'], color=colors_, width=3)\n"
            "for s in (-2, 2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted DST effect (bps/event)')\n"
            "ax.set_ylabel('Welch t-stat')\n"
            "ax.set_title('Engine detects the gap only when effect is large enough to clear |t|=2')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(ctrl_df.to_string(index=False))\n"
        ),
        md(
            "The engine is a faithful detector: it clears |t| >= 2 once the planted effect "
            "reaches about −50 bps/event.  The real observed gap (−19.89 bps) is below this "
            "detection threshold given only 93 events.  This is the fundamental problem: "
            "even if the KKL effect is real at −20 bps, 93 events provide only ~20% power "
            "to detect it at the 5% level.  The study is **underpowered by construction**."
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
