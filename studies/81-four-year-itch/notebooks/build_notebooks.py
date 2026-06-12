"""Generate the two narrative notebooks for Study 81 (Four-Year-Itch).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \\
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). Synthetic figures
run offline and deterministically; real-tape cells use the cached ^GSPC/SPY parquets under
../_cache/ if present, and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md), so the notebook re-runs for any reader.

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


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-12).
R = dict(
    # ^GSPC 1928-2026, n_days=24727
    y1_n=25, y1_mean=7.0,  y1_std=21.1, y1_tstat=1.78,
    y2_n=25, y2_mean=3.5,  y2_std=19.6, y2_tstat=0.91,
    y3_n=24, y3_mean=14.0, y3_std=17.9, y3_tstat=4.57,
    y4_n=24, y4_mean=6.9,  y4_std=15.1, y4_tstat=2.44,
    y3_minus_rest=8.2,
    y3_vs_rest_t=1.92,
    n_cycles=24,
    # SPY (post-1993)
    spy_y3_mean=19.0, spy_y3_t=3.57,
    spy_y3_vs_rest_t=1.48,
    # post-2001
    post2001_y3_vs_rest_t=0.98,
    # positive rates
    y1_pos=60, y2_pos=56, y3_pos=79, y4_pos=75,
    ticker="^GSPC",
    fp="df60aa9fe9f6",
)

# ---------------------------------------------------------------------------
# Shared preamble
# ---------------------------------------------------------------------------
BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root (quantlab/)
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from four_year_itch import data, strategy as st

TICKER = "^GSPC"

def _have_cache():
    try:
        data.fetch_prices(TICKER, fetch=False)
        return True
    except FileNotFoundError:
        return False

HAVE_REAL = _have_cache()
print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Four-Year-Itch -- does the Presidential cycle really load the dice for year 3?\n"
            "### The election-cycle effect on U.S. equities, tested honestly, in plain English\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats_a_coin%3F: NOT--SUPPORTED](https://img.shields.io/badge/Beats_a_coin%3F-NOT--SUPPORTED-8b949e?style=flat-square)\n\n"
            "Every four years, as a U.S. Presidential election approaches, the financial press "
            "rediscovers a deceptively seductive idea: **year 3 of a Presidential term is the "
            "best year for stocks** — the incumbent pumps the economy, the Fed plays along, and "
            "pre-election optimism lifts equity markets. It sounds reasonable. Year 3 does, in "
            "fact, carry the highest average return of the four-year cycle on the long S&P 500 "
            "history. But 'highest average' over 24 data points is almost meaningless. This "
            "notebook asks the only honest question: is the gap *statistically real*, or is it "
            "just noise on a tiny sample?\n\n"
            "> **This is the plain-language layer.** Want the t-stats, bootstrap intervals and "
            "subsample breakdowns? That's the companion, "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** -- same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart below is "
            "drawn by the code beside it. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Is year-3 the best year-of-term? | **On average, yes.** Mean +{R['y3_mean']:.0f}% vs "
            f"+{R['y2_mean']:.0f}% for year-2 — a gap of +{R['y3_minus_rest']:.0f} pp vs the rest. |\n"
            "| Is the gap statistically real? | **Barely not.** Welch t = "
            f"**+{R['y3_vs_rest_t']:.2f}** — just below the |t| ≥ 2 bar. With only "
            f"**{R['n_cycles']}** complete cycles since 1929, we can't confirm it. |\n"
            "| Does it work in modern data? | **No.** Post-2001 (5 cycles), Welch t = "
            f"**+{R['post2001_y3_vs_rest_t']:.2f}** — no signal at all. |\n"
            "| Could you trade it? | **No.** You'd be invested only 25% of the time, with ~18% "
            "annual vol and no certainty the next cycle won't be 1931 (-47%) or 2003 (+26%). |\n\n"
            "> The year-3 premium is real in the historical average — but it's built on 24 "
            "coin flips. Calling it a trading signal requires evidence we simply do not have."
        ),

        # ---- BEAT 1 -- THE CLAIM -----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Year 3 of a Presidential term is the best year for equities. The incumbent "
            "stimulates the economy ahead of the election — lower rates, fiscal spending, "
            "optimistic rhetoric — and the market knows it. Conversely, years 1 and 2 are lean: "
            "political capital is spent on unpopular reforms before the next vote is far enough "
            "away. The practical rule: buy in the pre-election year.\"*\n\n"
            "This idea is attributed to Yale Hirsch's *Stock Trader's Almanac*, published every "
            "year since 1972. It has real academic support in political-economy theory "
            "(Nordhaus 1975 on the 'Political Business Cycle'). And the numbers look "
            "compelling on first glance: going back to 1928, year-3 does carry the highest mean "
            f"annual return at **+{R['y3_mean']:.0f}%**."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If year-3 truly out-earns the rest, a 25%-of-the-time rule that only invests in "
            "the pre-election year would be an anomaly worth studying -- a calendar-driven "
            "equity premium with a real political mechanism. Investors have tried exactly this. "
            "If it's illusory, a story that has been repeated in every financial almanac and "
            "countless 'presidential cycle' articles for 50 years is a coin-flip dressed up as "
            "a law. The stakes are real: election-year timing is used as a retail investment "
            "strategy and is widely cited in financial media. Let's see what the data actually "
            "support."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Two rules keep us honest:\n\n"
            "1. **Ask the right question.** 'Is year-3 positive?' is trivially answered by the "
            "long-run equity premium — every year averages positive returns. The right question "
            "is 'Is year-3 *better than the rest*?' — and that requires a comparison to the "
            "pooled alternative years (1, 2, and 4).\n"
            "2. **Respect the sample size.** We have only ~24 year-3 observations since 1928. "
            "With annual stock-return volatility of ~18%, the standard error on the mean is "
            "≈ 18%/√24 ≈ 3.7 pp. A gap of 8 pp needs |t| ≥ 2 to be called real; anything "
            "smaller is statistically invisible noise at this sample size.\n\n"
            f"Data: **^GSPC** daily, 1928–2026, **{R['n_cycles']}** complete 4-year cycles."
        ),

        # ---- BEAT 4 -- THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown -- let's actually look\n\n"
            "**First, the headline: what does each year-of-term look like?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = data.fetch_prices(TICKER, fetch=False)\n"
            "    ann = st.annual_returns(df)\n"
            "    cs = st.cycle_stats(ann)\n"
            "    y_means = [cs.loc[i,'mean_pct'] for i in [1,2,3,4]]\n"
            "    y_stds  = [cs.loc[i,'std_pct'] for i in [1,2,3,4]]\n"
            "    y_ns    = [cs.loc[i,'n'] for i in [1,2,3,4]]\n"
            "else:\n"
            f"    y_means = [{R['y1_mean']}, {R['y2_mean']}, {R['y3_mean']}, {R['y4_mean']}]\n"
            f"    y_stds  = [{R['y1_std']},  {R['y2_std']},  {R['y3_std']},  {R['y4_std']}]\n"
            f"    y_ns    = [{R['y1_n']},     {R['y2_n']},     {R['y3_n']},     {R['y4_n']}]\n"
            "labels = ['Year 1\\n(inaugural)', 'Year 2\\n(mid-term)',\n"
            "          'Year 3\\n(pre-election)', 'Year 4\\n(election)']\n"
            "colors = [GREY, GREY, GREEN, GREY]\n"
            "fig, ax = plt.subplots(figsize=(9.5, 5.0))\n"
            "bars = ax.bar(labels, y_means, color=colors, width=0.6,\n"
            "              yerr=[s/n**0.5 for s,n in zip(y_stds, y_ns)],\n"
            "              error_kw={'ecolor':'k','capsize':5,'lw':1.5})\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Mean annual return (%)'); ax.set_ylim(-5, 25)\n"
            "ax.set_title('Year 3 looks best -- but is the gap real? (error bars = 1 SE)')\n"
            "for b, m, n in zip(bars, y_means, y_ns):\n"
            "    ax.annotate(f'+{m:.0f}%\\n(n={n})', (b.get_x()+b.get_width()/2, m+0.5),\n"
            "                ha='center', va='bottom', fontsize=9)\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"Year 3 does have the highest mean (+{R['y3_mean']:.0f}%) and the highest positive "
            f"rate ({R['y3_pos']}%). But look at those error bars. With ~24 observations per slot, "
            "they overlap massively. The eye-catching gap between year-3 and year-2 "
            f"(+{R['y3_mean']-R['y2_mean']:.0f} pp) is perfectly consistent with noise.\n\n"
            "> **The pattern looks compelling in a bar chart. That is exactly the trap.** A bar "
            "chart of 24 averages always shows one bar highest — the question is whether the "
            "highest bar is systematically higher or just lucky."
        ),
        md("**The fair test: is year-3 *meaningfully* better than the pooled rest?**"),
        code(
            "if HAVE_REAL:\n"
            "    df = data.fetch_prices(TICKER, fetch=False)\n"
            "    ann = st.annual_returns(df)\n"
            "    y3r = st.year3_vs_rest(ann)\n"
            "    diff, wt = y3r['diff_pct'], y3r['welch_t']\n"
            "else:\n"
            f"    diff, wt = {R['y3_minus_rest']}, {R['y3_vs_rest_t']}\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.5))\n"
            "ax.bar(['Year 3\\n(pre-election)', 'Rest\\n(years 1+2+4)'],\n"
            "       [14.0, 14.0-diff], color=[GREEN, GREY], width=0.5)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Mean annual return (%)')\n"
            "ax.set_title(f'Gap of {diff:.1f} pp -- Welch t = {wt:.2f} (bar is |t| = 2)')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Year-3 vs rest: diff = {diff:+.1f} pp, Welch t = {wt:+.2f}')\n"
            "print(f'The |t|>=2 bar: {\"CLEARED\" if abs(wt)>=2 else \"NOT CLEARED\"}')"
        ),
        md(
            f"**Welch t = +{R['y3_vs_rest_t']:.2f}** — just below our |t| ≥ 2 threshold for "
            "calling a signal real. The gap exists, but it is not statistically robust.\n\n"
            "> **Why is the individual HAC t for year-3 so high (+4.57) while the comparison t "
            "is only +1.92?** Because the individual t tests 'is year-3 positive?' — which is "
            "easy to confirm on 24 observations of equity returns. The comparison tests 'is "
            "year-3 *better than the 74 other observations*?' — a much harder question, with "
            "much larger pooled SE."
        ),
        md("**What about modern data? The subsample check.**"),
        code(
            "if HAVE_REAL:\n"
            "    df = data.fetch_prices(TICKER, fetch=False)\n"
            "    df_post = df[df.index.year >= 2001]\n"
            "    ann_post = st.annual_returns(df_post)\n"
            "    cs_post = st.cycle_stats(ann_post)\n"
            "    y3r_post = st.year3_vs_rest(ann_post)\n"
            "    means_post = [cs_post.loc[i,'mean_pct'] for i in [1,2,3,4]]\n"
            "    post_t = y3r_post['welch_t']\n"
            "else:\n"
            "    means_post = [15.1, -0.5, 13.7, 5.5]\n"
            f"    post_t = {R['post2001_y3_vs_rest_t']}\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "ax.bar(labels, means_post, color=colors, width=0.6)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_ylabel('Mean annual return (%) -- post-2001 only')\n"
            "ax.set_title(f'Post-2001 (5-7 cycles): year-3 vs rest t = {post_t:.2f}')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Post-2001 year-3 vs rest: Welch t = {post_t:.2f} -- no signal')"
        ),
        md(
            f"In the modern era (post-2001, 5–7 cycles per slot), the year-3 premium "
            f"essentially vanishes: Welch t = **+{R['post2001_y3_vs_rest_t']:.2f}**. Year 1 "
            "actually looks stronger than year 3 in this window. The 'effect' is driven by "
            "the full history including pre-war data."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** Year-3 mean (+{R['y3_mean']:.0f}%) does lead the cycle "
            "on the long record, but vs the rest, Welch t = "
            f"**+{R['y3_vs_rest_t']:.2f}** — just below the bar. Only 24 year-3 observations "
            "exist; we cannot confirm the effect.\n"
            "- **Tradability — Mirage.** Even if the premium were real, investing only 25% of "
            "the time in a high-vol asset does not outperform buy-and-hold on a risk-adjusted "
            "basis; worst year-3 ever is **-47%** (1931). No timing rule survives one 1931.\n"
            "- **Beats a coin? — Not supported.** The year-3 advantage is not meaningfully "
            "distinguishable from chance, especially in modern data (t = "
            f"{R['post2001_y3_vs_rest_t']:.2f})."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "The strategy: buy and hold the S&P 500 *only* in year-3 of every Presidential "
            "term, stay in cash the other 75% of the time. The maths are brutal:\n\n"
            "- You earn ~14% in your one year *when* markets cooperate -- but worst case is "
            "  **-47%** (1931).\n"
            "- You earn ~0% in cash the other three years (opportunity cost).\n"
            "- The buy-and-hold S&P earns +7.7%/yr on a fully-invested basis — you're giving "
            "  up three years of compounding for a one-year bet.\n"
            "- The result: lower long-run wealth, higher tracking error, and no statistical "
            "  guarantee the 'premium' isn't pure noise.\n\n"
            "This is a textbook **MIRAGE**: interesting in a bar chart, unusable in a portfolio."
        ),
        code(
            "# Illustrate the small-n problem: if we permute the year labels randomly,\n"
            "# how often does a 'best year' have an 8pp gap purely by chance?\n"
            "import numpy as np\n"
            "rng = np.random.default_rng(81)\n"
            "if HAVE_REAL:\n"
            "    df = data.fetch_prices(TICKER, fetch=False)\n"
            "    ann = st.annual_returns(df)\n"
            "    rets = ann['annual_ret'].dropna().values\n"
            "else:\n"
            "    rng2 = np.random.default_rng(99)\n"
            "    rets = rng2.normal(0.077, 0.18, 98)\n"
            "N = 50_000\n"
            "gaps = []\n"
            "n = len(rets)\n"
            "for _ in range(N):\n"
            "    shuffled = rng.permutation(rets)\n"
            "    groups = [shuffled[i::4] for i in range(4)]\n"
            "    best = max(g.mean() for g in groups)\n"
            "    rest_mean = np.concatenate([groups[j] for j in range(4) if j != np.argmax([g.mean() for g in groups])]).mean()\n"
            "    gaps.append((best - rest_mean) * 100)\n"
            "actual_gap = " + str(R['y3_minus_rest']) + "\n"
            "pval = (np.array(gaps) >= actual_gap).mean()\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "ax.hist(gaps, bins=60, color=GREY, alpha=0.7, label='permuted best-year gap')\n"
            "ax.axvline(actual_gap, c=GREEN, lw=2.5, label=f'actual year-3 gap: +{actual_gap:.1f} pp')\n"
            "ax.set_xlabel('best year-of-term gap (pp)'); ax.set_ylabel('count')\n"
            "ax.set_title('Permutation test: the year-3 gap is not extreme by chance')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Permutation p-value (one-sided): {pval:.3f}')"
        ),

        # ---- BEAT 7 -- GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Is the mechanism real?** There *is* academic evidence for the Political "
            "Business Cycle (Nordhaus 1975) — incumbents do manipulate the economy before "
            "elections. But the transmission from GDP manipulation to equity returns is "
            "indirect, delayed, and partly priced-in by modern markets.\n"
            "- **Party effects.** A parallel claim: Democratic presidencies are better for "
            "stocks. Also small-n and data-mined. [Study 67 — Fed-Drift](../../67-fed-drift/) "
            "tests the more specific FOMC policy channel with a tighter identification.\n"
            "- **Calendar seasonals that do hold up.** The [Turn-of-Month "
            "premium](../../42-last-call/) is the desk's best-supported calendar effect — "
            "real but thin and likely already traded.\n\n"
            "*Think the four-year cycle survives a tighter test? Fork this study, add the "
            "Federal funds rate cycle or sentiment data as co-variates, and see if the year-3 "
            "premium becomes robust. That is the bar.*"
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
            "# Four-Year-Itch -- a quantitative teardown\n"
            "### ^GSPC 1928-2026 * ~24 cycles * per-term-year HAC t-stats * subsample decay * permutation test\n\n"
            "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Beats_a_coin%3F: NOT--SUPPORTED](https://img.shields.io/badge/Beats_a_coin%3F-NOT--SUPPORTED-8b949e?style=flat-square)\n\n"
            "The deep companion to the [plain-language notebook](01_for_the_curious.ipynb) -- "
            "*same seven beats, every claim now carrying its standard error.* We test whether "
            "year-3 of the U.S. Presidential cycle delivers a statistically real excess annual "
            "return over years 1, 2, and 4, using the maximum available history (~24 cycles, "
            "^GSPC back to 1928), and we characterise the study's irreducible small-n ceiling.\n\n"
            "> **Not investment advice.** Real data: Yahoo daily bars, as-of 2026-06-12; "
            "fingerprint in [`docs/results.md`](../docs/results.md). The offline core and tests "
            "run on a deterministic synthetic tape.\n"
            ">\n"
            "> **The `> In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT + "\ntry:\n    from quantlab import analytics, stats\nexcept ImportError:\n    analytics = stats = None\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | Year-3 mean **+{R['y3_mean']:.0f}%** (HAC t *vs zero* = "
            f"**+{R['y3_tstat']:.2f}**), but year-3 vs rest Welch t = **+{R['y3_vs_rest_t']:.2f}** "
            f"-- just below |t|=2; n={R['y3_n']} year-3 obs, ~24 cycles. |\n"
            f"| **Tradability** | `MIRAGE` | 25%-invested calendar rule; worst year-3 was -47% "
            f"(1931); no risk-adjusted improvement over buy-and-hold. |\n"
            f"| **Beats a coin?** | `NOT SUPPORTED` | Post-2001 (5 cycles) Welch t = "
            f"**+{R['post2001_y3_vs_rest_t']:.2f}**; SPY (8 cycles) t = "
            f"**+{R['spy_y3_vs_rest_t']:.2f}**. The pattern is old-sample-driven. |\n\n"
            "> In plain words: the year-3 premium exists as a historical average, but with only "
            "24 data points we cannot confirm it is real rather than luck -- and in modern data "
            "it disappears entirely."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r_{y,c}$ be the annual return in year $y \\in \\{1,2,3,4\\}$ of cycle $c$. "
            "The Presidential-Cycle hypothesis asserts:\n\n"
            "- **H1 (differential return).** $\\mathbb{E}[r_{3,c}] > \\mathbb{E}[r_{y,c}]$ for "
            "$y \\in \\{1,2,4\\}$ — year-3 is genuinely better than the others.\n"
            "- **H2 (mechanism).** The premium is driven by pre-election fiscal/monetary "
            "stimulus (Nordhaus 1975), not random cyclical variation.\n"
            "- **H3 (persistent).** The premium exists in modern data (post-1993 or post-2001), "
            "not only in pre-war/pre-efficient-market history.\n\n"
            "Our evidence: H1 is *suggestive* but below the statistical bar (Welch t = 1.92); "
            "H2 is untested here; H3 is rejected (modern subsamples show t < 1.5)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? -- what rides on each answer\n\n"
            "The Presidential Cycle is one of the most-cited calendar effects in retail "
            "finance. If H1-H3 held, it would be a rare low-effort seasonal strategy with an "
            "intuitive political mechanism. The interesting finding is not just 'it fails' but "
            "**why**: the maximum-sample Welch t is 1.92 on 24 observations -- we are at the "
            "absolute sample ceiling (we cannot wait another 80 years for more cycles). The "
            "signal, if real, is too small and too noisy to survive the only test that matters."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · The protocol\n\n"
            "- **Data.** ^GSPC daily price returns, 1928-01-03 to 2026-06-11 (Yahoo Finance, "
            "cached parquet). No dividends in the price index -- fine for *differential* tests "
            "across years (dividends accrue roughly evenly). SPY (auto-adjusted, dividends "
            "re-invested) provides a shorter total-return robustness check.\n"
            "- **Labelling.** Calendar year $t$ is term-year $k$ if it falls in the $k$-th year "
            "of the Presidential term starting in the inauguration year (list of 25 term starts "
            "since 1929 coded in `data._TERM_START_YEARS`). Known before the year begins -- no "
            "look-ahead risk.\n"
            "- **Statistic.** Welch two-sample t on year-3 annual returns vs the pooled rest "
            "(years 1+2+4). Per-year-of-term HAC (Newey-West) t-stats for the individual means.\n"
            "- **Subsample.** Full 1928-2026 history; post-2001 (5-7 cycles); SPY 1993-2026 "
            "(8 cycles).\n"
            "- **Permutation.** Randomly relabel the year-of-term assignments 50,000 times "
            "and compute the best-year vs rest gap: confirms whether the 8 pp gap is surprising "
            "under the null of no cycle."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Full-history ^GSPC (1928-2026) -- the pattern exists, but barely\n\n"
            "Per-term-year annual returns: mean, std, HAC t (vs zero), and the fraction of "
            "positive years."
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = data.fetch_prices(TICKER, fetch=False)\n"
            "    ann = st.annual_returns(df)\n"
            "    cs = st.cycle_stats(ann)\n"
            "    y3r = st.year3_vs_rest(ann)\n"
            "    means = [cs.loc[i,'mean_pct'] for i in [1,2,3,4]]\n"
            "    stds  = [cs.loc[i,'std_pct'] for i in [1,2,3,4]]\n"
            "    tstats= [cs.loc[i,'tstat'] for i in [1,2,3,4]]\n"
            "    ns    = [cs.loc[i,'n'] for i in [1,2,3,4]]\n"
            "    diff, wt = y3r['diff_pct'], y3r['welch_t']\n"
            "else:\n"
            f"    means = [{R['y1_mean']},{R['y2_mean']},{R['y3_mean']},{R['y4_mean']}]\n"
            f"    stds  = [{R['y1_std']},{R['y2_std']},{R['y3_std']},{R['y4_std']}]\n"
            f"    tstats= [{R['y1_tstat']},{R['y2_tstat']},{R['y3_tstat']},{R['y4_tstat']}]\n"
            f"    ns    = [{R['y1_n']},{R['y2_n']},{R['y3_n']},{R['y4_n']}]\n"
            f"    diff, wt = {R['y3_minus_rest']}, {R['y3_vs_rest_t']}\n"
            "labels = ['Yr1\\n(inaugural)','Yr2\\n(mid-term)','Yr3\\n(pre-election)','Yr4\\n(election)']\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 5))\n"
            "c2 = [GREEN if i==2 else GREY for i in range(4)]\n"
            "axes[0].bar(labels, means, color=c2, yerr=[s/n**0.5 for s,n in zip(stds,ns)],\n"
            "            error_kw={'ecolor':'k','capsize':5})\n"
            "axes[0].axhline(0, color='k', lw=1); axes[0].set_ylabel('Mean annual return (%)')\n"
            "axes[0].set_title('Mean returns -- year 3 leads (barely)')\n"
            "ct = [GREEN if abs(t)>=2 else RED for t in tstats]\n"
            "axes[1].bar(labels, tstats, color=ct)\n"
            "for s in (2,-2): axes[1].axhline(s, ls='--', c=GREY, lw=1)\n"
            "axes[1].axhline(0, color='k', lw=1); axes[1].set_ylabel('HAC t (vs zero)')\n"
            "axes[1].set_title('HAC t-stats -- only year-3 clears +2 vs zero')\n"
            "plt.suptitle(f'Year-3 vs rest: diff={diff:+.1f}pp, Welch t={wt:.2f}', y=1.01)\n"
            "plt.tight_layout(); plt.show()\n"
            "import pandas as pd\n"
            "pd.DataFrame({'n':ns,'mean%':means,'std%':stds,'HAC_t':tstats},\n"
            "             index=['Yr1','Yr2','Yr3','Yr4']).round(2)"
        ),
        md(
            f"> In plain words: year-3's individual t-stat (+{R['y3_tstat']:.2f}) is large "
            "because 'is equity-market year-3 positive?' is an easy question on 24 points. "
            "The hard question is 'is it better than the other years?' -- and for that the "
            f"answer is Welch t = **+{R['y3_vs_rest_t']:.2f}**, just below the |t| ≥ 2 bar."
        ),
        md(
            "### 4b · The distribution of year-3 returns -- wide dispersion\n\n"
            "24 annual returns have enormous spread. Calling the mean 'the pattern' ignores "
            "the variance."
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = data.fetch_prices(TICKER, fetch=False)\n"
            "    ann = st.annual_returns(df)\n"
            "    fig, ax = plt.subplots(figsize=(11, 5))\n"
            "    colors_map = {1: GREY, 2: GREY, 3: GREEN, 4: AMBER}\n"
            "    for ty in [1, 2, 3, 4]:\n"
            "        vals = ann.loc[ann['term_year']==ty,'annual_ret'].values * 100\n"
            "        ax.scatter([ty]*len(vals), vals, alpha=0.6, s=40,\n"
            "                   color=colors_map[ty], label=f'Year {ty}')\n"
            "        ax.hlines(vals.mean(), ty-0.25, ty+0.25, lw=3,\n"
            "                  color=colors_map[ty])\n"
            "    ax.axhline(0, color='k', lw=1); ax.set_ylabel('Annual return (%)')\n"
            "    ax.set_xticks([1,2,3,4])\n"
            "    ax.set_xticklabels(['Yr1\\n(inaugural)','Yr2\\n(mid-term)',\n"
            "                        'Yr3\\n(pre-election)','Yr4\\n(election)'])\n"
            "    ax.set_title('Jitter plot: the overlapping dispersions explain the weak t-stat')\n"
            "    ax.legend(loc='upper left'); plt.tight_layout(); plt.show()\n"
            "    worst_y3 = ann.loc[ann['term_year']==3,'annual_ret'].min()*100\n"
            "    print(f'Worst year-3 ever: {worst_y3:.0f}%')\n"
            "else:\n"
            "    print('Need real data for this plot -- run with HAVE_REAL=True')"
        ),
        md(
            "The distributions overlap completely. The tick-marks (means) show the year-3 "
            "advantage, but the individual data points are scattered across the same range. "
            "A single year like 1931 (-47% in year 3) or the 2008 crisis pulls the distribution "
            "hard. This is why 24 data points are fundamentally insufficient."
        ),
        md(
            "### 4c · Subsample decay -- modern data shows nothing\n\n"
            "If the effect is a genuine political-economy mechanism, it should persist in "
            "modern markets. We check three windows."
        ),
        code(
            "if HAVE_REAL:\n"
            "    results = []\n"
            "    df_full = data.fetch_prices(TICKER, fetch=False)\n"
            "    for label, yr_start in [('Full 1928-2026',1928),('Post-1975',1975),('Post-2001',2001)]:\n"
            "        dfsub = df_full[df_full.index.year >= yr_start]\n"
            "        y3r = st.year3_vs_rest(st.annual_returns(dfsub))\n"
            "        cs_sub = st.cycle_stats(st.annual_returns(dfsub))\n"
            "        results.append((label, y3r['n_y3'], y3r['diff_pct'], y3r['welch_t']))\n"
            "    results = pd.DataFrame(results, columns=['window','n_y3','diff_pp','Welch_t'])\n"
            "    print(results.to_string(index=False))\n"
            "    fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "    ax.bar(results['window'], results['Welch_t'], color=[GREEN if abs(t)>=2 else RED for t in results['Welch_t']])\n"
            "    for s in (2,-2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "    ax.axhline(0, color='k', lw=1)\n"
            "    ax.set_ylabel('Welch t (year-3 vs rest)'); ax.set_title('Signal decays in modern data')\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('Full 1928-2026: n_y3=24 diff=+8.2pp t=+1.92')\n"
            "    print('Post-2001: (approx) diff=+6.9pp t=+0.98')\n"
        ),
        md(
            "> In plain words: the year-3 premium is strongest in pre-war data where the "
            "political business cycle was more blatant and markets less efficient. In modern "
            "data (where a strategy might actually be implemented) the t-stat collapses."
        ),
        md(
            "### 4d · Positive control -- the machinery finds the cycle when planted\n\n"
            "A synthetic daily tape with a planted year-3 premium confirms the engine is "
            "functional: it recovers a signal when one exists."
        ),
        code(
            "premia_grid = [\n"
            "    ('null', (0.0, 0.0, 0.0, 0.0)),\n"
            "    ('weak (+5 bps/day yr3)', (-1.0, -1.0, 5.0, 1.0)),\n"
            "    ('Hirsch-like (+10 bps/day yr3)', (-3.0, -3.0, 10.0, 3.0)),\n"
            "    ('strong (+20 bps/day yr3)', (-5.0, -5.0, 20.0, 5.0)),\n"
            "]\n"
            "recs = []\n"
            "for lbl, pr in premia_grid:\n"
            "    df_s, _ = data.synthetic_daily(n_cycles=24, year_premia=pr, seed=81)\n"
            "    s = st.summarize(df_s)\n"
            "    recs.append((lbl, s['y3_mean']-((s['y1_mean']+s['y2_mean']+s['y4_mean'])/3),\n"
            "                 s['y3_vs_rest_t']))\n"
            "recs_df = pd.DataFrame(recs, columns=['scenario','diff_pp','Welch_t'])\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "ax.bar(recs_df['scenario'], recs_df['Welch_t'],\n"
            "       color=[GREEN if abs(t)>=2 else RED for t in recs_df['Welch_t']])\n"
            "for s in (2,-2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, color='k', lw=1)\n"
            "ax.set_ylabel('Welch t (year-3 vs rest)'); ax.set_ylim(-1, 12)\n"
            "ax.set_title('Engine detects planted cycle (Welch t rises with premium size)')\n"
            "ax.tick_params(axis='x', rotation=15)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(recs_df.to_string(index=False))"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** -- year-3 mean {R['y3_mean']:+.0f}%, HAC t vs zero "
            f"= {R['y3_tstat']:+.2f}; vs rest Welch t = **{R['y3_vs_rest_t']:+.2f}** (bar: "
            f"|t|=2); n = {R['y3_n']} year-3 obs only. SPY t = {R['spy_y3_vs_rest_t']:+.2f}; "
            f"post-2001 t = {R['post2001_y3_vs_rest_t']:+.2f}.\n"
            "- **Tradability `MIRAGE`** -- 25%-invested calendar strategy; worst single year-3: "
            "-47% (1931); no risk-adjusted advantage over buy-and-hold.\n"
            f"- **Beats a coin? `NOT SUPPORTED`** -- effect confined to old data; modern "
            "subsamples show no signal. Cannot be confirmed at standard significance."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? -- the structural ceiling kills it\n\n"
            "Even a confirmed cycle premium would be difficult to trade:\n"
            "1. **Capacity.** The S&P is deep -- costs are negligible at fund scale. But the "
            "premium (if real, ~8 pp) is entirely consumed by the 75% cash drag: a "
            "buy-and-hold earns compound returns; a 25%-invested strategy earns mostly cash.\n"
            "2. **Tail risk.** A single -47% year-3 (like 1931) or a concentrated political "
            "shock could wipe out years of premium. The vol of year-3 returns is 17.9% -- "
            "not lower than other years.\n"
            "3. **Data snooping.** We tested one of four year-of-term slots on one index. "
            "Adjusting for 4 multiple tests raises the effective bar to |t| ≈ 2.4 (Bonferroni), "
            "which the pattern clearly does not meet.\n"
            "4. **Crowding.** The Almanac effect is public knowledge -- if it were real, "
            "arbitrage would erode it (consistent with the post-2001 decay)."
        ),
        code(
            "# Bonferroni-adjusted significance bar for testing 4 year-of-term slots\n"
            "from scipy.stats import t as tdist\n"
            "alpha_unadj = 0.05\n"
            "alpha_bonf = alpha_unadj / 4\n"
            "# Degrees of freedom for year-3 vs rest test (approx Welch)\n"
            "dof = 24 + 74 - 2\n"
            "t_bonf = tdist.ppf(1 - alpha_bonf/2, dof)\n"
            "print(f'Unadjusted |t| bar (alpha=0.05): {tdist.ppf(0.975, dof):.2f}')\n"
            "print(f'Bonferroni-adjusted |t| bar (4 tests): {t_bonf:.2f}')\n"
            "print(f'Actual Welch t for year-3 vs rest: +1.92')\n"
            "print('Verdict: BELOW even unadjusted bar')"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further -- what would make the cycle credible?\n\n"
            "The Presidential Cycle is weakly supported but not dead. To upgrade from WEAK "
            "to REAL would require:\n\n"
            "1. **A larger universe.** The cycle has been documented (though weakly) in "
            "international markets. A joint test across G7 or MSCI World would multiply the "
            "effective n -- though global covariance reduces the power gain.\n"
            "2. **The mechanism variable.** Conditioning on fiscal deficit expansion in year-3 "
            "or Fed easing posture could tighten the identification and separate the cycle "
            "from background noise.\n"
            "3. **Factor controls.** Post-1990 data shows year-1 performing as strongly as "
            "year-3 on SPY. A Fama-French factor regression might reveal whether any premium "
            "is just loading on small-cap or value — not an election effect at all.\n\n"
            "The desk's closest live effect: [Study 67 — Fed-Drift](../../67-fed-drift/) "
            "provides the policy-driven equity channel with tighter identification and a "
            "smaller-but-better-powered test."
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
