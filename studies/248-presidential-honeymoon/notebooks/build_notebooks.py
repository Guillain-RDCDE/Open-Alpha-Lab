"""Generate the two narrative notebooks for Study 248 (Presidential-Honeymoon).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). Synthetic figures
run offline and deterministically; real-tape cells use the cached ^GSPC parquet under
../_cache/ if present, and otherwise quote the frozen headline numbers in ``R``
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
    # ^GSPC 1928-2026, n_days=24729, fingerprint=cc8b1977fb03
    n=25,
    hm_mean=3.29,
    hm_std=20.08,
    hm_tstat=0.82,
    rest_mean=4.64,
    rest_std=14.90,
    rest_tstat=1.56,
    diff_mean=-1.34,
    diff_std=22.70,
    diff_tstat=-0.29,
    pos_rate=0.44,
    # First-year extended test
    fy_n=25,
    fy_yr1_mean=8.57,
    fy_yr2_mean=4.72,
    fy_diff_mean=3.85,
    fy_diff_tstat=0.65,
    # Post-2001 subsample (7 terms)
    post_n=7,
    post_hm_mean=1.42,
    post_ctrl_mean=14.06,
    post_diff_mean=-12.64,
    post_diff_tstat=-3.83,
    ticker="^GSPC",
    fp="cc8b1977fb03",
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

from presidential_honeymoon import data, strategy as st

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
            "# Presidential-Honeymoon — does the market enjoy a honeymoon in a new president's first 100 days?\n"
            "### The 'new-president rally' claim, tested honestly against 25 inaugurations, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Myth_check: BUSTED](https://img.shields.io/badge/Myth_check-BUSTED-8b949e?style=flat-square)\n\n"
            "Every four years the financial press asks: will markets rally in the new president's "
            "first 100 days? The idea feels intuitive — relief at the transition, optimism about "
            "new policies, a bipartisan honeymoon before the hard choices. And sometimes it does "
            "rally. But is the first 100 days *systematically better* than the rest of the year? "
            "This notebook asks that honest question across 25 inaugurations since Hoover (1929) "
            "and finds a bracing answer.\n\n"
            "> **This is the plain-language layer.** For t-stats, bootstrap intervals and "
            "subsample breakdown, see the companion "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> **Not investment advice.** Reproducible research: every chart is drawn by the "
            "code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Do first-100-day returns beat the next window? | **No.** Honeymoon mean +{R['hm_mean']:.2f}% "
            f"vs +{R['rest_mean']:.2f}% for the control — a gap of **{R['diff_mean']:+.2f} pp** in the *wrong* direction. |\n"
            f"| Is the gap statistically real? | **Not at all.** Paired HAC t = "
            f"**{R['diff_tstat']:+.2f}** across {R['n']} inaugurations — effectively zero. |\n"
            "| Does the win-rate beat a coin? | **No.** Honeymoon wins only 44% of terms — *below* 50%. |\n"
            "| Does it work in modern data? | **Reversed.** Post-2001: diff = "
            f"**{R['post_diff_mean']:+.2f} pp**, t = **{R['post_diff_tstat']:+.2f}**. |\n\n"
            "> The market honeymoon is a myth. Across 25 data points the honeymoon window "
            "actually *underperforms* the control, and the pattern actively reverses in modern data."
        ),

        # ---- BEAT 1 -- THE CLAIM -----------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When a new president takes office markets enjoy a honeymoon — relief at the "
            "transition, optimism about policy, a brief suspension of partisan fear. The S&P 500 "
            "typically rallies in the first 100 days. It is one of the most reliable short-term "
            "political seasonals.\"*\n\n"
            "This idea appears in every election-year market almanac and financial-media preview "
            "piece. There is superficial logic: a new administration brings fresh expectations; "
            "even a change from one party to the other can provide relief from uncertainty. "
            "Sometimes single-data-point examples like FDR 1933 (+78%!) are cited as evidence. "
            f"But we have {R['n']} inaugurations since 1929 — let's count all of them."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the first-100-day window systematically outperforms the rest of the year, it "
            "would be a tradeable political calendar seasonal — a reason to be long equities "
            "around inaugurations and to reduce exposure in months 4–12. It would validate the "
            "institutional practice of 'inauguration positioning' and the media narrative of a "
            "market honeymoon.\n\n"
            "If it is illusory, a story repeated in thousands of articles has no empirical "
            "support. Given how much investor behaviour (ETF positioning, options hedges, "
            "sentiment surveys) is influenced by this narrative, the stakes of getting it wrong "
            "are real."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Two rules keep us honest:\n\n"
            "1. **Ask the right question.** 'Are first-100-day returns positive?' is trivially "
            "answered by the long-run equity premium — almost any 100-day window averages "
            "positive. The honest question is 'Are first-100-day returns *better than the next "
            "265 days* of the same term?'. That within-term paired design removes trend and "
            "removes decade-level differences in market regimes.\n"
            "2. **Respect the sample size.** With 25 inauguration events and ~20% annual vol, "
            "the standard error on a 100-day cumulative return is about 10 pp. To clear the "
            "|t| ≥ 2 bar we need the honeymoon gap to exceed ~20 pp — a very high bar.\n\n"
            f"Data: **^GSPC** daily, 1928–2026, {R['n']} inaugurations from Hoover through "
            "Trump's second term."
        ),

        # ---- BEAT 4 -- THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the raw data: honeymoon vs control for every inauguration.**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = data.fetch_prices(TICKER, fetch=False)\n"
            "    periods = st.period_returns(df)\n"
            "    hm_rets = periods['hm_ret'].values * 100\n"
            "    ctrl_rets = periods['rest_ret'].values * 100\n"
            "    labels_p = [f\"{row['president']}\\n{idx.year}\" for idx, row in periods.iterrows()]\n"
            "    stats = st.honeymoon_stats(periods)\n"
            "    hm_mean, ctrl_mean, diff_mean, diff_t = (\n"
            "        stats['hm_mean'], stats['rest_mean'], stats['diff_mean'], stats['diff_tstat'])\n"
            "else:\n"
            f"    hm_mean, ctrl_mean, diff_mean, diff_t = {R['hm_mean']}, {R['rest_mean']}, {R['diff_mean']}, {R['diff_tstat']}\n"
            "fig, ax = plt.subplots(figsize=(11, 5))\n"
            "if HAVE_REAL:\n"
            "    x = range(len(hm_rets))\n"
            "    ax.bar([i - 0.2 for i in x], hm_rets, width=0.4, label='Honeymoon (100d)', color=AMBER, alpha=0.85)\n"
            "    ax.bar([i + 0.2 for i in x], ctrl_rets, width=0.4, label='Control (265d)', color=GREY, alpha=0.7)\n"
            "    ax.axhline(0, color='k', lw=1)\n"
            "    ax.set_ylabel('Cumulative return (%)')\n"
            "    ax.set_title(f'Honeymoon vs control: mean gap = {diff_mean:+.2f} pp (HAC t = {diff_t:+.2f})')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('Need real data for per-inauguration chart')\n"
            f"    print(f'Frozen: HM mean={{hm_m:+.2f}}%, Ctrl mean={{ctrl_m:+.2f}}%, Diff={{diff_m:+.2f}}pp, t={{diff_t:+.2f}}')"
        ),
        md(
            f"The first thing to notice: FDR 1933 (+78.3%) dominates the chart. That is not a "
            "'honeymoon' — it is an emergency New Deal bounce from Depression-era lows. Excluding "
            "that outlier makes the honeymoon hypothesis look *worse*. Even including it, the "
            f"honeymoon mean (+{R['hm_mean']:.2f}%) is *lower* than the control (+{R['rest_mean']:.2f}%).\n\n"
            "> **The key number: 44% of terms show the honeymoon outperforming the control.** "
            "That is below a fair coin."
        ),
        md("**The summary view: is the gap real?**"),
        code(
            "if HAVE_REAL:\n"
            "    df = data.fetch_prices(TICKER, fetch=False)\n"
            "    periods = st.period_returns(df)\n"
            "    stats = st.honeymoon_stats(periods)\n"
            "    diff_vals = (periods['hm_ret'] - periods['rest_ret']).values * 100\n"
            "    hm_m, ctrl_m, diff_m, diff_t = (\n"
            "        stats['hm_mean'], stats['rest_mean'], stats['diff_mean'], stats['diff_tstat'])\n"
            "else:\n"
            "    diff_vals = None\n"
            f"    hm_m, ctrl_m, diff_m, diff_t = {R['hm_mean']}, {R['rest_mean']}, {R['diff_mean']}, {R['diff_tstat']}\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 5))\n"
            "axes[0].bar(['Honeymoon\\n(first 100 days)', 'Control\\n(next 265 days)'],\n"
            "            [hm_m, ctrl_m], color=[AMBER, GREY], width=0.5)\n"
            "axes[0].axhline(0, color='k', lw=1)\n"
            "axes[0].set_ylabel('Mean cumulative return (%)')\n"
            "axes[0].set_title('Honeymoon UNDERPERFORMS the control')\n"
            "if diff_vals is not None:\n"
            "    axes[1].hist(diff_vals, bins=15, color=GREY, alpha=0.75, edgecolor='k', lw=0.5)\n"
            "    axes[1].axvline(diff_m, color=RED, lw=2.5, label=f'mean diff = {diff_m:+.2f} pp')\n"
            "    axes[1].axvline(0, color='k', lw=1, ls='--')\n"
            "    axes[1].set_xlabel('HM - Control (pp)'); axes[1].set_ylabel('Count')\n"
            "    axes[1].set_title(f'Distribution of paired diffs (t = {diff_t:+.2f})')\n"
            "    axes[1].legend()\n"
            "else:\n"
            "    axes[1].text(0.5, 0.5, f'HM-Ctrl diff mean:\\n{diff_m:+.2f} pp\\nHAC t = {diff_t:+.2f}',\n"
            "                 ha='center', va='center', transform=axes[1].transAxes, fontsize=16)\n"
            "    axes[1].set_title('Paired difference (frozen numbers)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Honeymoon mean: {hm_m:+.2f}%')\n"
            "print(f'Control mean:   {ctrl_m:+.2f}%')\n"
            "print(f'Paired diff:    {diff_m:+.2f} pp   HAC t = {diff_t:+.2f}')\n"
            "print(f'The |t|>=2 bar: {\"CLEARED\" if abs(diff_t)>=2 else \"NOT CLEARED - the honeymoon is not real\"}')"
        ),
        md(
            f"**Paired HAC t = {R['diff_tstat']:+.2f}.** That is not only far below the |t| ≥ 2 threshold — "
            "it is essentially zero and *negative*, meaning the honeymoon direction is wrong.\n\n"
            "> **There is no market honeymoon. The data say the opposite of the claim.**"
        ),
        md("**What about modern data?**"),
        code(
            "if HAVE_REAL:\n"
            "    df = data.fetch_prices(TICKER, fetch=False)\n"
            "    df_post = df[df.index.year >= 2001]\n"
            "    periods_post = st.period_returns(df_post)\n"
            "    s_post = st.honeymoon_stats(periods_post)\n"
            "    n_post, hm_post, ctrl_post, diff_post, t_post = (\n"
            "        s_post['n'], s_post['hm_mean'], s_post['rest_mean'],\n"
            "        s_post['diff_mean'], s_post['diff_tstat'])\n"
            "else:\n"
            f"    n_post, hm_post, ctrl_post, diff_post, t_post = ({R['post_n']}, {R['post_hm_mean']},\n"
            f"        {R['post_ctrl_mean']}, {R['post_diff_mean']}, {R['post_diff_tstat']})\n"
            "fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            "ax.bar(['Full 1929–2025\\n(n=25)', f'Post-2001\\n(n={n_post})'],\n"
            f"       [{R['diff_mean']}, diff_post], color=[RED, RED], width=0.5, alpha=0.85)\n"
            "ax.axhline(0, color='k', lw=1)\n"
            "ax.set_ylabel('HM − Control mean (pp)')\n"
            "ax.set_title('Honeymoon vs control: full history and modern subsample')\n"
            f"ax.annotate('t={R['diff_tstat']:+.2f}', (0, {R['diff_mean']} - 0.5),\n"
            "            ha='center', va='top', fontsize=12, color=RED)\n"
            "ax.annotate(f't={t_post:+.2f}', (1, diff_post - 0.5),\n"
            "            ha='center', va='top', fontsize=12, color=RED)\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Post-2001: HM mean = {hm_post:+.2f}%, Control = {ctrl_post:+.2f}%, diff = {diff_post:+.2f}pp, t = {t_post:+.2f}')"
        ),
        md(
            f"In modern data (post-2001, {R['post_n']} terms), the honeymoon is *clearly worse* than "
            f"the control: diff = {R['post_diff_mean']:+.2f} pp, t = **{R['post_diff_tstat']:+.2f}**. "
            "The pattern does not just fail to confirm — it reverses with statistical significance.\n\n"
            "> In recent inaugurations (Bush 2001, Obama, Trump, Biden, Trump 2025) the first "
            "100 days have been the *worst* stretch, not the best. Crisis episodes (2001, 2009, "
            "2025) dominate the modern sample."
        ),

        # ---- BEAT 5 -- THE VERDICT -------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — None.** Paired HAC t = {R['diff_tstat']:+.2f} across {R['n']} inaugurations. "
            "The honeymoon direction is wrong (underperformance, not outperformance). Post-2001 "
            f"t = {R['post_diff_tstat']:+.2f} — actively contradicts the claim.\n"
            "- **Tradability — Mirage.** A honeymoon-only buy rule earns less than staying "
            "invested. Before costs. No edge to extract.\n"
            "- **Beats a coin? — Busted.** 44% of terms show HM > Control. Below 50%."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "The strategy: buy S&P 500 at inauguration, sell 100 days later, sit in cash.\n\n"
            "- Raw return: +3.29% mean over 100 days.\n"
            "- But the *next* 265 days averages +4.64%. You give up the better window.\n"
            "- In cash 73% of the year: you lose compounding.\n"
            "- You incur two round-trip commissions and the bid-ask on entry and exit.\n"
            "- Worst single case: -10.3% (FDR third term, WWII mobilisation).\n"
            "- Modern record (post-2001): avg -12.64 pp vs control.\n\n"
            "There is no edge. There is no honeymoon. This is a **textbook MIRAGE**."
        ),
        code(
            "# Simple illustration: is the honeymoon win-rate above chance?\n"
            "import numpy as np\n"
            "rng = np.random.default_rng(248)\n"
            "n_terms = 25\n"
            "# Simulate what a 'coin-flip' honeymoon win-rate distribution looks like\n"
            "sim_winrates = []\n"
            "for _ in range(20_000):\n"
            "    wins = rng.binomial(n_terms, 0.5)\n"
            "    sim_winrates.append(wins / n_terms)\n"
            "sim_winrates = np.array(sim_winrates)\n"
            f"actual_wr = {R['pos_rate']}\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "ax.hist(sim_winrates, bins=40, color=GREY, alpha=0.7, label='coin-flip null distribution')\n"
            "ax.axvline(actual_wr, c=RED, lw=2.5, label=f'actual win-rate: {actual_wr:.0%}')\n"
            "ax.axvline(0.5, c='k', lw=1.5, ls='--', label='50% (fair coin)')\n"
            "ax.set_xlabel('Honeymoon win-rate (fraction of terms HM > Control)')\n"
            "ax.set_ylabel('Count')\n"
            "ax.set_title('The 44% win-rate is unremarkable — consistent with noise')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "p_below = (sim_winrates <= actual_wr).mean()\n"
            "print(f'Fraction of coin-flip runs with win-rate <= {actual_wr:.0%}: {p_below:.2f}')\n"
            "print('A fair coin beats the honeymoon more than half the time.')"
        ),

        # ---- BEAT 7 -- GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **Is there any political seasonal that works?** The desk's best-supported calendar "
            "effects are driven by structural micro-mechanics (flows, fund rules, options "
            "expiry) — not political narratives. [Study 89 — Turn-of-Month](../../89-turn-of-month/) "
            "is the clearest example: real, thin, probably capacity-constrained.\n"
            "- **The four-year cycle (year-3 premium).** [Study 81 — Four-Year-Itch](../81-four-year-itch/) "
            "tests whether year 3 of the term outperforms other years. The answer: weakly "
            "supported (Welch t ≈ 1.9) but not robust. Stronger than the honeymoon, but still "
            "not a trading signal.\n"
            "- **The party effect.** [Study 159 — Presidential-Party](../159-presidential-party/) "
            "tests Democrat vs Republican returns. Also not supported.\n\n"
            "*Convinced the honeymoon effect should appear in a different subset or with a "
            "different window? Fork this study, change the 100-day parameter in "
            "`data.HONEYMOON_DAYS`, and see if any window survives the paired test across "
            "all 25 inaugurations.*"
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
            "# Presidential-Honeymoon — a quantitative teardown\n"
            "### ^GSPC 1928-2026 · 25 inaugurations · paired HAC t · subsample reversal · positive control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Myth_check: BUSTED](https://img.shields.io/badge/Myth_check-BUSTED-8b949e?style=flat-square)\n\n"
            "The deep companion to the [plain-language notebook](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim now carrying its standard error.* We test whether "
            "the S&P 500 delivers a statistically real excess return in the first 100 calendar "
            "days after a U.S. presidential inauguration ('the honeymoon window'), using a "
            "within-term paired design on 25 inaugurations since Hoover (1929).\n\n"
            "> **Not investment advice.** Real data: Yahoo daily bars, as-of 2026-06-15; "
            f"fingerprint `{R['fp']}` in [`docs/results.md`](../docs/results.md). The offline "
            "core and tests run on a deterministic synthetic tape.\n"
            ">\n"
            "> **The `> In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Paired HAC t = **{R['diff_tstat']:+.2f}** across n={R['n']} inaugurations; "
            f"honeymoon mean {R['hm_mean']:+.2f}% vs control {R['rest_mean']:+.2f}% — gap is "
            f"**{R['diff_mean']:+.2f} pp** in the wrong direction. |\n"
            "| **Tradability** | `MIRAGE` | Honeymoon buy-and-hold earns less than the control window; "
            "negative net edge before costs. |\n"
            f"| **Beats a coin?** | `BUSTED` | Win-rate = {R['pos_rate']:.0%} (below 50%); "
            f"post-2001 diff = {R['post_diff_mean']:+.2f} pp (t = {R['post_diff_tstat']:+.2f}), "
            "actively contradicts the claim. |\n\n"
            "> In plain words: there is no market honeymoon. The first 100 days after inauguration "
            "underperform the next 265 days on average, with a near-zero t-stat and a post-2001 "
            "reversal that is statistically significant — in the *wrong* direction."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $r^{HM}_i$ be the cumulative S&P 500 return over the first 100 calendar days "
            "after inauguration $i$, and $r^{Ctrl}_i$ over the subsequent 265 calendar days. "
            "The honeymoon hypothesis asserts:\n\n"
            "- **H1 (directional).** $\\mathbb{E}[r^{HM}_i] > \\mathbb{E}[r^{Ctrl}_i]$ — "
            "the honeymoon window systematically outperforms the rest-of-year control.\n"
            "- **H2 (causal).** The excess return is driven by political optimism / policy "
            "uncertainty resolution at inauguration — not by coincidental market conditions.\n"
            "- **H3 (persistent).** The premium persists in modern data (post-2001), where "
            "it might be expected to be most relevant for today's practitioner.\n\n"
            "Our evidence: H1 is rejected (diff < 0, t ≈ 0); H2 is untestable here; "
            "H3 is strongly rejected (post-2001 t = −3.83, sign reversed)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "The honeymoon narrative is used to justify inauguration-period equity overweights "
            "by retail and institutional investors. If H1-H3 held, a simple 100-day long "
            "position around each inauguration would be a rare low-effort seasonal with a "
            "political mechanism. Instead, we find the opposite: the honeymoon window is "
            "systematically *worse*, driven (in modern data) by 2001, 2009, and 2025 — all "
            "crises that landed in the first 100 days."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · The protocol\n\n"
            "- **Data.** ^GSPC daily price-index returns, 1928-01-03 to 2026-06-15 (Yahoo "
            f"Finance, cached parquet, fingerprint `{R['fp']}`). No dividends in the price "
            "index — fine for a *relative* honeymoon vs control comparison (dividends accrue "
            "roughly evenly across both windows).\n"
            "- **Labelling.** For each of the 25 inaugurations since Hoover (hardcoded in "
            "`data.INAUGURATIONS`), we define:\n"
            "  - **Honeymoon window**: trading days $t \\in [\\text{inau}, \\text{inau} + 100\\text{d})$\n"
            "  - **Control window**: trading days $t \\in [\\text{inau}+100\\text{d}, \\text{inau}+365\\text{d})$\n"
            "- **Statistic.** Paired Newey-West HAC t-test on the within-term difference "
            "$d_i = r^{HM}_i - r^{Ctrl}_i$, $i = 1, \\ldots, 25$.\n"
            "- **Inference bar.** |HAC t| ≥ 2 required to call a signal real. With n = 25 and "
            "per-term diff standard deviation ~23 pp, the minimum detectable gap at |t| = 2 "
            "is about ±9 pp — a demanding bar for a 100-day window.\n"
            "- **Subsamples.** Full 1929–2025; post-2001 (7 inaugurations).\n"
            "- **Extended test.** First calendar year vs second year after inauguration "
            "(a broader window)."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Full-history ^GSPC — the honeymoon underperforms\n\n"
            "Per-inauguration raw returns and summary statistics."
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = data.fetch_prices(TICKER, fetch=False)\n"
            "    periods = st.period_returns(df)\n"
            "    stats = st.honeymoon_stats(periods)\n"
            "    hm_vals = periods['hm_ret'].values * 100\n"
            "    ctrl_vals = periods['rest_ret'].values * 100\n"
            "    diff_vals = hm_vals - ctrl_vals\n"
            "    hm_m, ctrl_m, diff_m, diff_t = (\n"
            "        stats['hm_mean'], stats['rest_mean'], stats['diff_mean'], stats['diff_tstat'])\n"
            "    pos_r = stats['pos_rate']\n"
            "else:\n"
            f"    hm_m, ctrl_m, diff_m, diff_t, pos_r = ({R['hm_mean']}, {R['rest_mean']},\n"
            f"        {R['diff_mean']}, {R['diff_tstat']}, {R['pos_rate']})\n"
            "    hm_vals = ctrl_vals = diff_vals = None\n"
            "import pandas as pd\n"
            "summary = pd.DataFrame({\n"
            "    'Window': ['Honeymoon (100d)', 'Control (265d)', 'Diff (HM-Ctrl)'],\n"
            "    'Mean (%)': [hm_m, ctrl_m, diff_m],\n"
            f"    'HAC t': [{R['hm_tstat']}, {R['rest_tstat']}, {R['diff_tstat']}],\n"
            "})\n"
            "print(summary.to_string(index=False))\n"
            f"print(f'\\nPositive rate (HM > Control): {{pos_r:.0%}} across {R['n']} inaugurations')\n"
            "print('|HAC t| >= 2 bar:', abs(diff_t) >= 2)"
        ),
        md(
            f"> In plain words: the paired t of **{R['diff_tstat']:+.2f}** is the key number. "
            "It is not only far below our |t| ≥ 2 bar — it is negative, meaning the mean "
            "diff points *against* the honeymoon claim. With n = 25, the SE on the diff is "
            f"about {R['diff_std']:.1f}%/√25 ≈ {R['diff_std']/25**0.5:.1f} pp. The observed "
            f"gap of {R['diff_mean']:+.2f} pp is swamped by this noise."
        ),
        md(
            "### 4b · Per-inauguration scatter — one giant outlier, otherwise noise\n\n"
            "Plotting each inauguration's honeymoon vs control return reveals the FDR 1933 outlier."
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = data.fetch_prices(TICKER, fetch=False)\n"
            "    periods = st.period_returns(df)\n"
            "    hm_vals = periods['hm_ret'].values * 100\n"
            "    ctrl_vals = periods['rest_ret'].values * 100\n"
            "    years = [idx.year for idx in periods.index]\n"
            "    fig, ax = plt.subplots(figsize=(9, 6))\n"
            "    colors_pt = [AMBER if h > c else GREY for h, c in zip(hm_vals, ctrl_vals)]\n"
            "    ax.scatter(ctrl_vals, hm_vals, c=colors_pt, s=60, zorder=3, edgecolors='k', lw=0.5)\n"
            "    lim = max(abs(hm_vals).max(), abs(ctrl_vals).max()) * 1.1\n"
            "    ax.plot([-lim, lim], [-lim, lim], 'k--', lw=1, alpha=0.5, label='HM = Control (diagonal)')\n"
            "    # label outliers\n"
            "    for i, (h, c, yr) in enumerate(zip(hm_vals, ctrl_vals, years)):\n"
            "        if abs(h) > 15 or abs(c) > 25:\n"
            "            ax.annotate(str(yr), (c, h), fontsize=8, ha='left', va='bottom')\n"
            "    ax.set_xlabel('Control return (%, next 265d)')\n"
            "    ax.set_ylabel('Honeymoon return (%, first 100d)')\n"
            "    ax.set_title('Scatter: above diagonal = HM wins; amber = HM > Control')\n"
            "    ax.legend(); plt.tight_layout(); plt.show()\n"
            "    print(f'Points above diagonal (HM wins): {sum(h > c for h, c in zip(hm_vals, ctrl_vals))}/{len(hm_vals)}')\n"
            "else:\n"
            "    print('Need real data for per-inauguration scatter')\n"
            f"    print(f'From frozen R: {R['pos_rate']:.0%} of terms HM wins (below diagonal)')"
        ),
        md(
            "The scatter shows no systematic tendency for points to be above the diagonal "
            "(honeymoon > control). The 1933 point at (+78%, +6%) is a massive outlier that "
            "misleads simple averages. With or without it, the data do not support the claim."
        ),
        md(
            "### 4c · Subsample breakdown — modern data actively contradicts the claim\n\n"
            "Full history vs post-2001 (7 inaugurations) comparison."
        ),
        code(
            "if HAVE_REAL:\n"
            "    df = data.fetch_prices(TICKER, fetch=False)\n"
            "    sub_results = []\n"
            "    for label, yr_start in [('Full 1929-2025', 1929), ('Post-2001', 2001)]:\n"
            "        dfsub = df[df.index.year >= yr_start]\n"
            "        p_sub = st.period_returns(dfsub)\n"
            "        s_sub = st.honeymoon_stats(p_sub)\n"
            "        sub_results.append((label, s_sub['n'], s_sub['hm_mean'],\n"
            "                            s_sub['rest_mean'], s_sub['diff_mean'], s_sub['diff_tstat']))\n"
            "    sr = pd.DataFrame(sub_results, columns=['Window','n','HM_mean%','Ctrl_mean%','Diff_pp','HAC_t'])\n"
            "    print(sr.to_string(index=False))\n"
            "    fig, ax = plt.subplots(figsize=(8, 4.5))\n"
            "    ax.bar(sr['Window'], sr['Diff_pp'], color=[RED if d < 0 else GREEN for d in sr['Diff_pp']],\n"
            "           width=0.5, alpha=0.85)\n"
            "    ax.axhline(0, color='k', lw=1)\n"
            "    for s in (2, -2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "    for i, row in sr.iterrows():\n"
            "        ax.annotate(f't={row[\"HAC_t\"]:+.2f}', (i, row['Diff_pp']-0.8),\n"
            "                    ha='center', va='top', fontsize=12, color='white', fontweight='bold')\n"
            "    ax.set_ylabel('HM − Control mean (pp)')\n"
            "    ax.set_title('Honeymoon underperforms in both windows (post-2001: significantly)')\n"
            "    plt.tight_layout(); plt.show()\n"
            "else:\n"
            f"    print(f'Full history: n={R['n']}, diff={R['diff_mean']:+.2f}pp, t={R['diff_tstat']:+.2f}')\n"
            f"    print(f'Post-2001:    n={R['post_n']}, diff={R['post_diff_mean']:+.2f}pp, t={R['post_diff_tstat']:+.2f}')"
        ),
        md(
            f"> In plain words: the post-2001 t-stat of {R['post_diff_tstat']:+.2f} clears the |t| ≥ 2 bar — "
            "but in the *wrong* direction. The market has had a systematic *anti*-honeymoon in "
            "modern inaugurations: 2001 (dot-com bust), 2009 (GFC recovery, good, but control "
            "was 2009H2 +31%), 2025 (tariff shock, -7.3% honeymoon vs +24.8% subsequent). "
            "There is no market honeymoon. The evidence points the other way."
        ),
        md(
            "### 4d · Positive control — the machinery finds a premium when planted\n\n"
            "Synthetic tape with a large planted honeymoon premium (+20 bps/day) confirms "
            "the engine would detect a signal if one existed."
        ),
        code(
            "scenarios = [\n"
            "    ('null (0 bps)', 0.0),\n"
            "    ('weak (+5 bps)', 5.0),\n"
            "    ('signal (+20 bps)', 20.0),\n"
            "]\n"
            "recs = []\n"
            "for lbl, prem in scenarios:\n"
            "    df_s, _ = data.synthetic_daily(n_terms=25, honeymoon_premium_bp=prem, seed=248)\n"
            "    periods_s = st.period_returns(df_s)\n"
            "    stats_s = st.honeymoon_stats(periods_s)\n"
            "    recs.append((lbl, prem, stats_s['diff_mean'], stats_s['diff_tstat']))\n"
            "recs_df = pd.DataFrame(recs, columns=['scenario', 'planted_bp', 'diff_pp', 'HAC_t'])\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "cols = [GREEN if abs(t) >= 2 else RED for t in recs_df['HAC_t']]\n"
            "ax.bar(recs_df['scenario'], recs_df['HAC_t'], color=cols, alpha=0.85)\n"
            "for s in (2, -2): ax.axhline(s, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, color='k', lw=1)\n"
            "ax.set_ylabel('HAC t-stat (HM − Control)')\n"
            "ax.set_title('Engine recovers signal when planted; real tape shows None')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(recs_df.to_string(index=False))\n"
            f"print(f'\\nReal tape (frozen): HAC t = {R['diff_tstat']:+.2f} -- None')"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — paired HAC t = {R['diff_tstat']:+.2f} across {R['n']} inaugurations. "
            f"Honeymoon mean {R['hm_mean']:+.2f}% vs control {R['rest_mean']:+.2f}%; diff "
            f"{R['diff_mean']:+.2f} pp, SE ≈ {R['diff_std']/R['n']**0.5:.1f} pp. "
            f"Post-2001 t = {R['post_diff_tstat']:+.2f} (reversed). Win-rate = {R['pos_rate']:.0%} < 50%.\n"
            "- **Tradability `MIRAGE`** — negative expected edge before costs; worse in modern data; "
            "opportunity cost from sitting out 73% of the year.\n"
            "- **Beats a coin? `BUSTED`** — 44% win-rate, near-zero t on full sample, "
            "significant reversal post-2001."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the structural ceiling is reversed\n\n"
            "Unlike most failed calendar effects where the signal is 'absent', here it is "
            "*negative* in modern data. The practical implications:\n\n"
            "1. **Crowded-out by crises.** The modern sample is dominated by inaugurations "
            "that coincided with crises starting or ending (9/11 aftermath, GFC, 2025 tariffs). "
            "If the next inauguration is calm, the honeymoon might be better — but you cannot "
            "know in advance.\n"
            "2. **FDR 1933 is not evidence.** A +78% return during an unprecedented "
            "Depression-era emergency bank closure is not a 'honeymoon rally' — it is a "
            "crisis-rebound. Citing it as honeymoon evidence is data mining.\n"
            "3. **Small-n ceiling.** We have 25 data points. Even if the true honeymoon premium "
            "were +5 pp, the SE is ~4.5 pp. We cannot detect it at standard significance. "
            "The sample size does not allow confirmation or rejection of a small positive effect.\n"
            "4. **Conclusion for the practitioner.** There is no evidence to justify an "
            "inauguration-period overweight."
        ),
        code(
            "# SE computation to illustrate the small-n ceiling\n"
            f"import numpy as np\n"
            f"n_terms = {R['n']}\n"
            f"diff_std = {R['diff_std']}  # pp, estimated from real data\n"
            "se = diff_std / np.sqrt(n_terms)\n"
            "print(f'n inaugurations: {n_terms}')\n"
            "print(f'SE of mean diff: {se:.2f} pp')\n"
            "print(f'Minimum detectable gap at |t|=2: {2 * se:.2f} pp')\n"
            f"print(f'Observed gap: {R['diff_mean']:+.2f} pp -- well inside the noise band')\n"
            "print()\n"
            "# Additional terms needed to detect a +5pp true premium at |t|=2\n"
            "target_gap = 5.0  # pp\n"
            "n_needed = (2 * diff_std / target_gap) ** 2\n"
            "print(f'To detect +{target_gap:.0f}pp true premium at |t|=2 would require ~{n_needed:.0f} inaugurations')\n"
            "print('(We have 25 since 1929 -- a 4-year cycle would need ~170 more years)')"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — what would make the honeymoon credible?\n\n"
            "The honeymoon narrative is not just unsupported — the modern data actively contradicts "
            "it. To upgrade from NONE to any positive signal would require:\n\n"
            "1. **A different window definition.** Perhaps the effect is concentrated in the "
            "first 20 days (not 100), or in the last 30 days of the 100. Testing multiple "
            "window sizes introduces a multiple-testing problem — any reported window would "
            "need Bonferroni correction.\n"
            "2. **Crisis exclusion logic.** If inaugurations that coincide with crises are "
            "excluded, perhaps a residual honeymoon exists. But the exclusion criterion "
            "cannot be defined ex-ante without look-ahead.\n"
            "3. **Mechanism variable.** Conditioning on policy uncertainty indices or approval "
            "ratings might sharpen identification — but it also adds a second degree of freedom "
            "and reduces the already-tiny effective n.\n\n"
            "The desk's verdict: the presidential honeymoon is a narrative with no statistical "
            "footing. [Study 81](../81-four-year-itch/) and [Study 159](../159-presidential-party/) "
            "test adjacent political calendar claims with similar results — weak or none."
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
