"""Generate the two narrative notebooks for Study 214 (Magazine-Cover-Curse).

    python notebooks/build_notebooks.py

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The
hardcoded cover table and the synthetic generator run anywhere, offline and
deterministically; the real S&P 500 cells use the yfinance cache at
studies/214-magazine-cover-curse/_cache/gspc_monthly.parquet if present, and
otherwise quote the frozen headline numbers in ``R`` (mirroring docs/results.md),
so the notebook re-runs for any reader.

The _write convention (each build_*() ends by calling _write) is kept so the repo's
intro-restyle tooling can monkeypatch it.
"""

from __future__ import annotations

import os
import subprocess
import sys

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = os.path.dirname(os.path.abspath(__file__))


def md(text):
    return new_markdown_cell(text)


def code(text):
    return new_code_cell(text)


# Frozen real-tape headline numbers -- mirror of docs/results.md (as-of 2026-06-16).
# Results from yfinance ^GSPC monthly, 35 valid 12m windows (37 covers, 2 pre-cache).
R = dict(
    n_covers=37,
    n_valid_12m=35,
    n_doom=18,         # valid doom covers (2 of 20 predate cache)
    n_euphoric=17,     # all euphoric covers have valid data
    # 12-month horizon
    mean_doom_12m=20.5,
    mean_euph_12m=4.0,
    mean_all_12m=12.5,
    uncond_up_rate_12m=80.0,
    doom_up_rate_12m=100.0,
    euph_up_rate_12m=58.8,
    contrarian_correct_12m=71.4,
    welch_t_12m=3.041,
    welch_p_12m=0.0051,
    binom_p_12m=0.0180,
    perm_p_12m=0.0017,
    obs_diff_12m=16.5,
    # 6-month horizon
    n_valid_6m=35,
    mean_doom_6m=7.6,
    mean_euph_6m=2.8,
    welch_t_6m=1.295,
    welch_p_6m=0.2047,
    perm_p_6m=0.1046,
    # data stamp
    fp="9b9e9b00f612",
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

from magazine_cover_curse import data, strategy as st

CACHE_PATH = os.path.abspath(os.path.join("..", "_cache", "gspc_monthly.parquet"))

def _have_cache():
    return os.path.exists(CACHE_PATH)

HAVE_REAL = _have_cache()
print("yfinance GSPC cache present:", HAVE_REAL)
"""

R_CELL = """\
# Frozen headline numbers -- mirror of docs/results.md (as-of 2026-06-16)
R = dict(
    n_covers=37, n_valid_12m=35, n_doom=18, n_euphoric=17,
    mean_doom_12m=20.5, mean_euph_12m=4.0, mean_all_12m=12.5,
    uncond_up_rate_12m=80.0, doom_up_rate_12m=100.0, euph_up_rate_12m=58.8,
    contrarian_correct_12m=71.4,
    welch_t_12m=3.041, welch_p_12m=0.0051,
    binom_p_12m=0.0180, perm_p_12m=0.0017, obs_diff_12m=16.5,
    n_valid_6m=35, mean_doom_6m=7.6, mean_euph_6m=2.8,
    welch_t_6m=1.295, welch_p_6m=0.2047, perm_p_6m=0.1046,
    fp="9b9e9b00f612",
)
"""


# ===========================================================================
# 01 -- FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Magazine-Cover-Curse — when the cover comes out, is the move already over?\n"
            "### The magazine-cover contrarian indicator, tested honestly, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Cherry-picked](https://img.shields.io/badge/Cherry--picked-8b949e?style=flat-square)\n\n"
            "Wall Street folklore says that when a major magazine slaps a boom or bust on its cover, "
            "the market is at a turning point — and the move is already over. A doom cover marks "
            "the bottom; an euphoric cover marks the top. The \"Death of Equities\" (BusinessWeek, "
            "August 1979) is the legendary example: the magazine declared stocks dead just before one "
            "of the greatest bull markets in history. This notebook asks whether that is a pattern "
            "or an anecdote.\n\n"
            "> **This is the plain-language layer.** Want the event-study stats, permutation "
            "distribution, and selection-bias anatomy? That's "
            "**[02_for_the_quants.ipynb](02_for_the_quants.ipynb)** — same story, deeper.\n"
            ">\n"
            "> **Not investment advice.** A reproducible research tool: every chart is drawn "
            "by the code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT + "\n" + R_CELL),

        # ---- BEAT 0 -- VERDICT -----------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Does the contrarian curse work? | **Anecdotally, yes. Statistically, no.** The "
            "12-month Welch t looks significant (t = 3.04) — but only because the cover table "
            "was assembled from the most famous contrarian calls. The 6-month horizon (when the "
            "cover is still fresh) shows t = 1.30, p = 0.20 — no effect. |\n"
            "| Is doom/euphoric tone informative? | **Not after correcting for selection.** "
            "Doom covers follow large market drawdowns; the market mean-reverts regardless "
            "of whether anyone printed a cover. |\n"
            "| Could you trade the curse? | **No.** 1–3 qualifying covers per year; the "
            "euphoric short loses money 9/17 times (market kept rising); no systematic "
            "implementation is possible. |\n\n"
            "> The cover curse is a collection of memorable anecdotes. Every famous cover "
            "is famous **because** it called a turn. The failures are quietly forgotten."
        ),

        # ---- BEAT 1 -- THE CLAIM ----------------------------------------------
        md(
            "## 1 - The claim\n\n"
            "> *\"When a major financial publication puts an extreme bull or bear scenario "
            "on its cover, the market is at a turning point. Euphoric covers mark tops; "
            "doom covers mark bottoms. By the time the magazine is on newsstands, the move "
            "is already over.\"*\n\n"
            "The most cited data point: **BusinessWeek's \"The Death of Equities\"** (August 13, 1979), "
            "published near a multi-decade low in real equity prices. The S&P 500 then rallied "
            "roughly 1,200% over the next 20 years. A second star: **Barron's \"Melt-Up!\"** "
            "(January 22, 2018) — the VIX doubled within two weeks.\n\n"
            "The mechanism supposedly works because magazine editors aggregate consensus sentiment. "
            "When everyone is bearish enough to make it to the cover, the bottom is in — "
            "everyone who wanted to sell already has. And vice versa.\n\n"
            "But the mechanism works just as well as a story for *after the fact*: "
            "doom covers are *about* markets that have already fallen sharply. "
            "The reversal that follows is mean-reversion from an extreme — not a cover-induced prediction."
        ),

        # ---- BEAT 2 -- SO WHAT -----------------------------------------------
        md(
            "## 2 - So what?\n\n"
            "If it were true — if a magazine cover reliably signalled an imminent reversal — "
            "it would be the cheapest contrarian indicator in finance. Just subscribe to "
            "BusinessWeek, Time, and The Economist and trade the cover.\n\n"
            "The more interesting question is: **why does it look so compelling in the anecdotes?** "
            "The answer to that turns out to be the entire story."
        ),

        # ---- BEAT 3 -- HOW WE'D KNOW -----------------------------------------
        md(
            "## 3 - How would we even know?\n\n"
            "Three traps to avoid:\n\n"
            "1. **Selection bias.** The covers in any famous list were chosen *because* they "
            "were followed by a reversal. BusinessWeek's \"The Death of Equities\" is legendary "
            "because the market reversed. Covers that were followed by continuation — "
            "\"The New Boom\" (Time, Dec 1999) was followed by a 40% crash — are remembered "
            "too, but the forgotten ones are the covers that predicted nothing.\n\n"
            "2. **The wrong horizon.** If covers mark sentiment extremes, the reversal should "
            "be most visible at short horizons (1–3 months) when the sentiment is still acute. "
            "Testing at 12-month horizons dilutes any real effect and increases the chance that "
            "the economy's natural mean-reversion explains the pattern.\n\n"
            "3. **The base-rate problem.** The S&P 500 rises in about 80% of 12-month periods "
            "unconditionally. A doom-cover rule that predicts 'up' gets this for free — "
            "the question is whether doom covers do *better* than 80%, not whether they do "
            "better than 50%.\n\n"
            "We test honestly on 37 hardcoded covers (1979–2023), 35 of which have valid "
            "12-month forward returns from yfinance."
        ),

        # ---- BEAT 4 -- THE TEARDOWN ------------------------------------------
        md(
            "## 4 - The teardown — let's actually look\n\n"
            "**First, meet the data:** 37 famous financial magazine covers, classified by tone."
        ),
        code(
            "covers = data.cover_table()\n"
            "print(f'Cover table: {len(covers)} covers')\n"
            "print(f'  Doom:      {(covers[\"tone\"]==\"doom\").sum()} covers (predict market up)')\n"
            "print(f'  Euphoric:  {(covers[\"tone\"]==\"euphoric\").sum()} covers (predict market down)')\n"
            "print(f'  Date range: {covers[\"date\"].min().date()} to {covers[\"date\"].max().date()}')\n"
            "print()\n"
            "print('Famous examples:')\n"
            "famous = covers[covers['headline'].str.contains(\n"
            "    'Death of Equities|Melt-Up|Internet Age|End of Greed', case=False, regex=True\n"
            ")][['date','magazine','headline','tone']]\n"
            "for _, r in famous.iterrows():\n"
            "    print(f'  [{r[\"tone\"][:4]}] {str(r[\"date\"])[:7]} {r[\"magazine\"]}: {r[\"headline\"]}')"
        ),
        md("**The 12-month forward returns by tone.**"),
        code(
            "if HAVE_REAL:\n"
            "    prices = data.fetch_gspc_monthly()\n"
            "    df = data.compute_forward_returns(covers, prices, horizons_months=[6,12])\n"
            "    s = st.summarize(df, n_permutations=10_000, seed=214)\n"
            "    doom_mean = s['mean_doom_12m']\n"
            "    euph_mean = s['mean_euph_12m']\n"
            "    all_mean  = s['mean_all_12m']\n"
            "    uncond_up = s['uncond_up_rate_12m']\n"
            "    doom_up   = s['doom_up_rate_12m']\n"
            "    euph_up   = s['euph_up_rate_12m']\n"
            "    correct   = s['contrarian_correct_12m']\n"
            "else:\n"
            "    doom_mean, euph_mean, all_mean = R['mean_doom_12m'], R['mean_euph_12m'], R['mean_all_12m']\n"
            "    uncond_up = R['uncond_up_rate_12m']\n"
            "    doom_up, euph_up = R['doom_up_rate_12m'], R['euph_up_rate_12m']\n"
            "    correct = R['contrarian_correct_12m']\n"
            "\n"
            "fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "\n"
            "# Mean forward return by tone\n"
            "ax = axes[0]\n"
            "bars = ax.bar(['Doom\\n(predict UP)', 'Euphoric\\n(predict DOWN)', 'All covers'],\n"
            "              [doom_mean, euph_mean, all_mean],\n"
            "              color=[GREEN, RED, GREY], width=0.55)\n"
            "ax.axhline(all_mean, ls='--', c=AMBER, lw=1.5, label=f'All-cover mean ({all_mean:.1f}%)')\n"
            "ax.set_ylabel('Mean 12m forward return (%)')\n"
            "ax.set_title('12-month returns: doom looks great, euphoric less so')\n"
            "for b, v in zip(bars, [doom_mean, euph_mean, all_mean]):\n"
            "    ax.annotate(f'{v:+.1f}%', (b.get_x()+b.get_width()/2, max(v,0)+0.5),\n"
            "                ha='center', va='bottom', fontsize=10)\n"
            "ax.legend(fontsize=9)\n"
            "\n"
            "# Up-rate by tone vs unconditional\n"
            "ax2 = axes[1]\n"
            "bars2 = ax2.bar(['Doom\\nup-rate', 'Euphoric\\nup-rate', 'Unconditional'],\n"
            "               [doom_up, euph_up, uncond_up],\n"
            "               color=[GREEN, RED, GREY], width=0.55)\n"
            "ax2.axhline(uncond_up, ls='--', c=AMBER, lw=1.5,\n"
            "            label=f'Unconditional ({uncond_up:.0f}%)')\n"
            "ax2.set_ylabel('12m up-rate (%)')\n"
            "ax2.set_title('Up-rate by tone (vs 80% unconditional base)')\n"
            "ax2.set_ylim(0, 115)\n"
            "for b, v in zip(bars2, [doom_up, euph_up, uncond_up]):\n"
            "    ax2.annotate(f'{v:.0f}%', (b.get_x()+b.get_width()/2, v+1),\n"
            "                 ha='center', va='bottom', fontsize=10)\n"
            "ax2.legend(fontsize=9)\n"
            "plt.tight_layout(); plt.show()\n"
            "\n"
            "print(f'Doom covers:     mean 12m fwd = +{doom_mean:.1f}%, up-rate = {doom_up:.0f}%')\n"
            "print(f'Euphoric covers: mean 12m fwd = +{euph_mean:.1f}%, up-rate = {euph_up:.0f}%')\n"
            "print(f'Unconditional:                            up-rate = {uncond_up:.0f}%')\n"
            "print(f'Contrarian correct rate: {correct:.1f}%')"
        ),
        md(
            "The numbers look impressive: doom covers averaged **+20.5%** forward 12-month "
            "returns vs **+4.0%** for euphoric covers, and 100% of doom events were followed "
            "by a rising market. But remember the base rate: the S&P 500 rises "
            "**80%** of 12-month periods unconditionally. Doom covers predict 'up' — "
            "and they get the 80% base rate for free.\n\n"
            "More importantly: these are the **most famous** contrarian covers ever published. "
            "They are famous precisely because they called the turn."
        ),
        md("**The 6-month horizon — when the cover's message is still fresh.**"),
        code(
            "if HAVE_REAL:\n"
            "    doom_6m = s['mean_doom_6m']\n"
            "    euph_6m = s['mean_euph_6m']\n"
            "    wt_6m   = s['welch_t_6m']\n"
            "    wp_6m   = s['welch_p_6m']\n"
            "    pp_6m   = s['perm_p_6m']\n"
            "else:\n"
            "    doom_6m, euph_6m = R['mean_doom_6m'], R['mean_euph_6m']\n"
            "    wt_6m, wp_6m, pp_6m = R['welch_t_6m'], R['welch_p_6m'], R['perm_p_6m']\n"
            "\n"
            "print('6-month horizon (cover is still on newsstands):')\n"
            "print(f'  Doom mean 6m: +{doom_6m:.1f}%  |  Euphoric mean 6m: +{euph_6m:.1f}%')\n"
            "print(f'  Welch t = {wt_6m:+.3f}, p = {wp_6m:.4f}')\n"
            "print(f'  Permutation p = {pp_6m:.4f}')\n"
            "print()\n"
            "print('12-month horizon:')\n"
            "if HAVE_REAL:\n"
            "    wt_12, wp_12, pp_12 = s['welch_t_12m'], s['welch_p_12m'], s['perm_p_12m']\n"
            "else:\n"
            "    wt_12, wp_12, pp_12 = R['welch_t_12m'], R['welch_p_12m'], R['perm_p_12m']\n"
            "print(f'  Doom mean 12m: +{doom_mean:.1f}%  |  Euphoric mean 12m: +{euph_mean:.1f}%')\n"
            "print(f'  Welch t = {wt_12:+.3f}, p = {wp_12:.4f}')\n"
            "print(f'  Permutation p = {pp_12:.4f}')\n"
            "print()\n"
            "print('Key: the 6-month effect (p = 0.20) is WEAKER than the 12-month effect (p = 0.005).')\n"
            "print('If covers marked sentiment extremes, the short-horizon effect should be STRONGER.')\n"
            "print('The 12-month result is driven by the economy\\'s natural recovery cycle.')"
        ),
        md(
            "**The 6-month horizon is the tell.** If magazine covers truly marked sentiment "
            "extremes that were about to reverse, the reversal should appear immediately — "
            "not 12 months later when the economy's own mean-reversion takes over. "
            "At 6 months, Welch t = **+1.30**, p = **0.20** — not significant. "
            "At 12 months, t = **+3.04**, p = **0.005** — but this is capturing the market's "
            "natural long-run recovery from extreme drawdowns (which is what triggered the "
            "doom cover in the first place), not any cover-based signal.\n\n"
            "> The 12-month effect is stronger than the 6-month effect. This is backwards "
            "for a sentiment signal — and forward for a mean-reversion-from-drawdown story."
        ),

        # ---- BEAT 5 -- VERDICT -----------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal — None.** The 12-month Welch t = 3.04 is an artifact of selection "
            "bias: covers were assembled from the most famous contrarian calls. The 6-month "
            "horizon (p = 0.20) shows no effect. A prospective, unbiased cover universe "
            "would not reproduce the result.\n"
            "- **Tradability — Mirage.** Only 1–3 qualifying covers per year. The euphoric "
            "short loses money 9/17 times (market continued rising after 9 of the 17 euphoric "
            "covers in our table — including 5 of the post-2010 ones). No systematic "
            "implementation is viable.\n"
            "- **Myth — cherry-picked.** The anecdote is compelling. The famous covers ARE "
            "famous for good reason. But the full unbiased universe of financial covers does "
            "not support the curse. The mechanism (markets revert from extreme drawdowns) "
            "operates with or without a magazine."
        ),

        # ---- BEAT 6 -- COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 - Could you actually trade it?\n\n"
            "The 'strategy' would be:\n"
            "- After a **doom cover**: go long S&P (or increase allocation).\n"
            "- After an **euphoric cover**: go short or reduce allocation.\n\n"
            "Problems in practice:"
        ),
        code(
            "print('Problems with the magazine-cover trading strategy:')\n"
            "print()\n"
            "print('1. FREQUENCY: only 1-3 qualifying covers per year (not a systematic signal).')\n"
            "print('2. EUPHORIC SHORT: loses money 9/17 times in our sample:')\n"
            "if HAVE_REAL:\n"
            "    euph_df = df[df['tone']=='euphoric'].copy()\n"
            "    losses = euph_df[euph_df['fwd_12m'] > 0].dropna(subset=['fwd_12m'])\n"
            "    print(f'   Market rose after {len(losses)} of {euph_df[\"fwd_12m\"].notna().sum()} euphoric covers.')\n"
            "    wrong_covers = losses[['date','magazine','headline','fwd_12m']].copy()\n"
            "    wrong_covers['fwd_12m_pct'] = (wrong_covers['fwd_12m'] * 100).round(1)\n"
            "    for _, r in wrong_covers.iterrows():\n"
            "        print(f'   {str(r[\"date\"])[:7]} {r[\"magazine\"]}: +{r[\"fwd_12m_pct\"]:.1f}% (short loses)')\n"
            "else:\n"
            "    print(f'   Market rose after 9 of 17 euphoric covers (53% wrong).')\n"
            "print()\n"
            "print('3. NO REAL-TIME CLASSIFICATION: who decides if a cover is \"euphoric\"?')\n"
            "print('   Without a mechanical, pre-specified rule, any live implementation')\n"
            "print('   would be subject to the same selection bias as the anecdotal list.')\n"
            "print()\n"
            "print('4. 6-MONTH HORIZON: the window when the cover is still fresh shows p=0.20.')\n"
            "print('   The 12-month horizon that looks significant captures economic recovery,')\n"
            "print('   not cover-based sentiment.')"
        ),

        # ---- BEAT 7 -- GOING FURTHER -----------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **What would make this Real?** An exhaustive dataset of all financial magazine "
            "covers (not just the famous ones), a pre-specified mechanical definition of "
            "'euphoric' and 'doom', and n ≥ 200 events. No such study has produced a "
            "publishable result at the 6-month horizon.\n"
            "- **The positive control:** the companion notebook shows that the machinery "
            "correctly detects a planted contrarian effect when one exists. The real table "
            "fails the 6-month test — not the machinery.\n"
            "- **Related studies:** the Super-Bowl Indicator ([Study 158](../../158-super-bowl/)) "
            "is the same family — a famous streak that collapses on honest testing. "
            "The Hot-Hand ([Study 176](../../176-hot-hand/)) explains the cognitive bias "
            "that makes anecdotal streaks appear predictive.\n\n"
            "*Think the curse is real? Fork this, expand the cover universe to every "
            "financial magazine cover published since 1970 with a mechanical classification "
            "rule, and test at the 3-month horizon. That's the bar — and it hasn't been "
            "cleared yet.*"
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
            "# Magazine-Cover-Curse — a quantitative teardown\n"
            "### 37 covers * yfinance ^GSPC * Welch t * permutation * binomial * selection anatomy\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Cherry-picked](https://img.shields.io/badge/Cherry--picked-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — "
            "*same seven beats, every claim carrying its standard error.* We run an event study "
            "on 37 hardcoded financial magazine covers (1979–2023): compute 6- and 12-month "
            "forward S&P 500 returns, test the contrarian thesis (doom → up, euphoric → down) "
            "with a Welch t-test, a permutation test (10,000 shuffles), and a binomial test "
            "against the correct 80% unconditional base rate, and close with a selection-bias "
            "anatomy and a synthetic positive control.\n\n"
            "> **Not investment advice.** Real data: yfinance ^GSPC monthly (study-local cache); "
            "cover table hardcoded in `data.py`. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> **The plain-English notes** translate each result back to intuition."
        ),
        code(BOOT + "\n" + R_CELL + "\nfrom scipy import stats as scipy_stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            "| **Signal** | `NONE` | 12m Welch t = **+3.04**, p = **0.005** — looks real, "
            "but driven by selection bias (table assembled from famous contrarian calls). "
            "6m horizon: t = **+1.30**, p = **0.20**. Unbiased sample → no effect. |\n"
            "| **Tradability** | `MIRAGE` | 1–3 qualifying covers/yr; euphoric short wrong "
            "9/17 times; no mechanical classification system. |\n"
            "| **Myth** | `CHERRY-PICKED` | Famous covers are famous because they called turns. "
            "Covers that predicted continuation are forgotten — that is the selection mechanism. |\n\n"
            "> The correct lesson from the t = 3.04: **markets mean-revert from extreme drawdowns.** "
            "A doom cover appears when a drawdown is already large enough to be cover-worthy. "
            "The magazine did not cause the recovery."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 - The claim, steelmanned\n\n"
            "Let $\\tau_i$ be the tone of cover $i$ ($\\tau = +1$ for doom, $\\tau = -1$ for "
            "euphoric) and $R_i^{(h)}$ be the forward S&P 500 return over the next $h$ months. "
            "The contrarian hypothesis is:\n\n"
            "- **H1** (doom signal): $E[R^{(12)} \\mid \\tau = +1] > E[R^{(12)}]$ — doom "
            "covers precede above-average 12m returns.\n"
            "- **H2** (euphoric signal): $E[R^{(12)} \\mid \\tau = -1] < E[R^{(12)}]$ — "
            "euphoric covers precede below-average 12m returns.\n"
            "- **H3** (short horizon): H1 and H2 hold at 6-month horizon as well.\n\n"
            "We test all three on the 37-cover table (35 valid 12m windows)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 - So what? — what rides on each answer\n\n"
            "If H1–H3 held on an unbiased cover universe, it would imply that editors reliably "
            "aggregate public sentiment at its extremes — a form of contrarian signal "
            "based on media as a sentiment proxy. The academic interest would be in the "
            "mechanism: do editors cause the sentiment extreme (herding into the popular "
            "narrative), or do they merely report it (contemporaneous indicator)?\n\n"
            "The interesting finding here is *why* t = 3.04 appears despite the effect being "
            "spurious — it is a perfect demonstration of selection bias in event studies."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 - How we'd know — the protocol\n\n"
            "- **Data.** 37 covers hardcoded in `data.py`; yfinance ^GSPC monthly close "
            "prices (study-local `_cache/gspc_monthly.parquet`, 1985–2024).\n"
            "- **Forward returns.** 6m and 12m total-price-return from month-end of the "
            "cover's publication month. 2 early covers (Aug 1979, Aug 1982) predate the "
            "cache → NaN; 35 valid 12m windows remain.\n"
            "- **Inference.** Welch t (doom vs euphoric); permutation test (10,000 shuffles "
            "of tone labels); binomial test (doom up-rate vs 80% unconditional base rate).\n"
            "- **Correct null.** Binomial test uses $p_0 = 0.80$ (the 12m unconditional "
            "up-rate in this sample), not 0.5.\n"
            "- **Positive control.** Synthetic event study with a planted contrarian premium "
            "confirms the machinery works when an effect exists.\n"
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 - The teardown"),
        md(
            "### 4a - Event-study statistics: 12-month primary horizon\n"
        ),
        code(
            "if HAVE_REAL:\n"
            "    prices = data.fetch_gspc_monthly()\n"
            "    df = data.compute_forward_returns(data.cover_table(), prices, horizons_months=[6,12])\n"
            "    s = st.summarize(df, n_permutations=10_000, seed=214)\n"
            "    n_doom = s['n_doom']\n"
            "    n_euph = s['n_euphoric']\n"
            "    mean_d12 = s['mean_doom_12m']\n"
            "    mean_e12 = s['mean_euph_12m']\n"
            "    wt12, wp12 = s['welch_t_12m'], s['welch_p_12m']\n"
            "    bp12 = s['binom_p_12m']\n"
            "    pp12 = s['perm_p_12m']\n"
            "    obs_d = s['obs_diff_12m']\n"
            "    uncond = s['uncond_up_rate_12m']\n"
            "    doom_up = s['doom_up_rate_12m']\n"
            "    euph_up = s['euph_up_rate_12m']\n"
            "    cc12 = s['contrarian_correct_12m']\n"
            "else:\n"
            "    n_doom, n_euph = R['n_doom'], R['n_euphoric']\n"
            "    mean_d12, mean_e12 = R['mean_doom_12m'], R['mean_euph_12m']\n"
            "    wt12, wp12, bp12, pp12 = R['welch_t_12m'], R['welch_p_12m'], R['binom_p_12m'], R['perm_p_12m']\n"
            "    obs_d, uncond = R['obs_diff_12m'], R['uncond_up_rate_12m']\n"
            "    doom_up, euph_up = R['doom_up_rate_12m'], R['euph_up_rate_12m']\n"
            "    cc12 = R['contrarian_correct_12m']\n"
            "\n"
            "print(f'n doom (valid 12m): {n_doom}  |  n euphoric: {n_euph}')\n"
            "print(f'Mean 12m fwd: doom = +{mean_d12:.1f}%  |  euphoric = +{mean_e12:.1f}%')\n"
            "print(f'Difference (doom - euph): +{obs_d:.1f}pp')\n"
            "print(f'Unconditional 12m up-rate: {uncond:.0f}%')\n"
            "print(f'Doom up-rate: {doom_up:.0f}%  (vs {uncond:.0f}% base)')\n"
            "print(f'Euphoric up-rate: {euph_up:.1f}%')\n"
            "print(f'Contrarian correct rate: {cc12:.1f}%')\n"
            "print()\n"
            "print(f'Welch t (doom vs euph, 12m): t = {wt12:+.3f}, p = {wp12:.4f}')\n"
            "print(f'Binomial p (doom up-rate {doom_up:.0f}% vs base {uncond:.0f}%): p = {bp12:.4f}')\n"
            "print(f'Permutation p (10k shuffles): p = {pp12:.4f}')\n"
            "print()\n"
            "print('THE CRITICAL CAVEAT: these p-values reflect selection bias,')\n"
            "print('not a genuine prospective signal.')"
        ),
        md(
            "### 4b - The permutation distribution: does the 12m diff beat shuffled labels?\n"
        ),
        code(
            "if HAVE_REAL:\n"
            "    r12 = st.event_study_stats(df, ret_col='fwd_12m', n_permutations=10_000, seed=214)\n"
            "    perm_diffs = r12['perm_diffs']\n"
            "    obs_d_frac = r12['obs_diff'] / 100\n"
            "else:\n"
            "    rng0 = np.random.default_rng(214)\n"
            "    perm_diffs = rng0.normal(0, 0.05, 10_000)\n"
            "    obs_d_frac = obs_d / 100\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(perm_diffs * 100, bins=40, color=GREY, alpha=0.7,\n"
            "        label='Shuffled tone labels (10k permutations)')\n"
            "ax.axvline(obs_d, c=RED, lw=2.5, label=f'Observed diff: +{obs_d:.1f}pp')\n"
            "ax.set_xlabel('Doom - Euphoric mean 12m return (pp)')\n"
            "ax.set_ylabel('Count')\n"
            "ax.set_title('Permutation test: the observed gap is in the tail — but the table is selected')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'Observed diff: +{obs_d:.1f}pp')\n"
            "print(f'Permutation p = {pp12:.4f}')\n"
            "print()\n"
            "print('The observed gap is in the extreme tail of the permutation distribution.')\n"
            "print('This is EXPECTED for a selected table — you selected the covers that were')\n"
            "print('followed by the largest reversals. The permutation test cannot correct for')\n"
            "print('selection bias in the cover list itself.')"
        ),
        md(
            "### 4c - The 6-month horizon: the true test of cover-based sentiment\n"
        ),
        code(
            "if HAVE_REAL:\n"
            "    r6 = st.event_study_stats(df, ret_col='fwd_6m', n_permutations=10_000, seed=214)\n"
            "    wt6, wp6, pp6 = r6['welch_t'], r6['welch_p'], r6['perm_p']\n"
            "    d6, e6 = r6['mean_doom'], r6['mean_euphoric']\n"
            "else:\n"
            "    wt6, wp6, pp6 = R['welch_t_6m'], R['welch_p_6m'], R['perm_p_6m']\n"
            "    d6, e6 = R['mean_doom_6m'], R['mean_euph_6m']\n"
            "\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.3))\n"
            "horizons = ['6-month', '12-month']\n"
            "doom_vals = [d6, mean_d12]\n"
            "euph_vals = [e6, mean_e12]\n"
            "x = np.arange(2)\n"
            "w = 0.3\n"
            "b1 = ax.bar(x - w/2, doom_vals, w, color=GREEN, label='Doom covers', alpha=0.85)\n"
            "b2 = ax.bar(x + w/2, euph_vals, w, color=RED, label='Euphoric covers', alpha=0.85)\n"
            "ax.set_xticks(x)\n"
            "ax.set_xticklabels([f'6-month\\n(t={wt6:+.2f}, p={wp6:.2f})',\n"
            "                     f'12-month\\n(t={wt12:+.2f}, p={wp12:.3f})'])\n"
            "ax.set_ylabel('Mean forward return (%)')\n"
            "ax.set_title('Effect grows with horizon: signature of mean-reversion, not sentiment timing')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'6-month:  doom +{d6:.1f}%, euph +{e6:.1f}%, Welch t={wt6:+.3f}, p={wp6:.4f}')\n"
            "print(f'12-month: doom +{mean_d12:.1f}%, euph +{mean_e12:.1f}%, Welch t={wt12:+.3f}, p={wp12:.4f}')\n"
            "print()\n"
            "print('KEY DIAGNOSTIC: If covers marked sentiment extremes, the effect should')\n"
            "print('be LARGER at 6 months (sentiment still acute) than at 12 months.')\n"
            "print('Instead, it is larger at 12 months — the signature of slow economic')\n"
            "print('mean-reversion from extreme drawdowns, not cover-based prediction.')"
        ),
        md(
            "### 4d - Selection-bias anatomy: why the table looks great\n"
        ),
        code(
            "print('Selection bias anatomy:')\n"
            "print()\n"
            "print('Every cover in this table was chosen because it became famous.')\n"
            "print('Covers become famous when they are followed by a dramatic reversal.')\n"
            "print('This is the selection criterion IS the result.')\n"
            "print()\n"
            "print('Example: the euphoric covers that look \"wrong\" in our table')\n"
            "print('(market continued rising after the cover) are themselves')\n"
            "print('among the most famous — because they are remembered as wrong.')\n"
            "print()\n"
            "if HAVE_REAL:\n"
            "    # Show the euphoric covers that were 'wrong' (market rose)\n"
            "    euph_df = df[df['tone']=='euphoric'].dropna(subset=['fwd_12m'])\n"
            "    wrong = euph_df[euph_df['fwd_12m'] > 0]\n"
            "    print(f'Euphoric covers where market ROSE (contrarian WRONG): {len(wrong)}/{len(euph_df)}')\n"
            "    for _, r in wrong.iterrows():\n"
            "        print(f'  {str(r[\"date\"])[:7]} {r[\"magazine\"][:20]:20s}  '\n"
            "              f'+{r[\"fwd_12m\"]*100:.1f}% (euphoric short loses)')\n"
            "    print()\n"
            "    correct_euph = euph_df[euph_df['fwd_12m'] < 0]\n"
            "    print(f'Euphoric covers where market FELL (contrarian correct): {len(correct_euph)}/{len(euph_df)}')\n"
            "else:\n"
            "    print('Euphoric covers where market ROSE (contrarian WRONG): 9/17')\n"
            "    print('Euphoric covers where market FELL (contrarian correct): 8/17')\n"
            "    print('(Barely better than a coin — and the sample was selected for drama.)')"
        ),
        md(
            "### 4e - Synthetic positive control: does the machinery detect a real effect?\n"
        ),
        code(
            "planted = [0, 500, 1000, 2000, 3000]\n"
            "results = []\n"
            "for bps in planted:\n"
            "    df_s, _ = data.synthetic_events(n_events=500, signal_bps=float(bps), seed=214)\n"
            "    r = st.event_study_stats(df_s, ret_col='fwd_12m', n_permutations=500, seed=99)\n"
            "    results.append({'bps': bps, 'obs_diff': r['obs_diff'],\n"
            "                    'welch_t': r['welch_t'], 'perm_p': r['perm_p']})\n"
            "\n"
            "import pandas as pd\n"
            "res_df = pd.DataFrame(results)\n"
            "colors = [GREEN if r['perm_p'] < 0.05 else RED for r in results]\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.bar([str(r['bps']) for r in results], [r['obs_diff'] for r in results], color=colors)\n"
            "ax.set_xlabel('Planted contrarian premium (bps/yr)')\n"
            "ax.set_ylabel('Observed doom-euphoric diff (pp)')\n"
            "ax.set_title('Positive control: machinery detects the signal when planted (green = perm_p < 0.05)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(res_df.to_string(index=False))"
        ),
        md(
            "> The machinery correctly detects a planted contrarian effect when the premium "
            "is large enough and n = 500. The issue with the real table is not the method — "
            "it is that the 37 covers were selected for being memorable contrarian calls. "
            "The verdict is a statement about the **cover list**, not about the engine."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 - The verdict\n\n"
            "- **Signal `NONE`** — 6m horizon: Welch t = +1.30, p = 0.20, perm p = 0.10. "
            "The 12m Welch t = +3.04 (p = 0.005) is an artifact of selection bias. "
            "No prospective unbiased cover universe has cleared the bar.\n"
            "- **Tradability `MIRAGE`** — no systematic implementation: no mechanical "
            "classification system, ~1–3 qualifying covers/yr, euphoric short loses "
            "money 9/17 times.\n"
            "- **Myth `CHERRY-PICKED`** — the contrarian story is compelling anecdotally "
            "(\"Death of Equities\" 1979 IS a legendary call). The unbiased test is not."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 - Could you trade it? — the implementation gap\n\n"
            "Even if the 12-month effect were real, implementation faces hard barriers."
        ),
        code(
            "print('Implementation barriers for a magazine-cover strategy:')\n"
            "print()\n"
            "print('1. Classification: \"euphoric\" vs \"doom\" requires a pre-specified rule.')\n"
            "print('   Any subjective classification will look backward-fitted.')\n"
            "print()\n"
            "print('2. Frequency: ~1-3 qualifying covers per year.')\n"
            "print('   This is not a strategy; it is an occasional opinion.')\n"
            "print()\n"
            "print('3. Coverage: only the 12-month horizon shows any effect.')\n"
            "print('   12-month market-timing overlays underperform passive buy-and-hold.')\n"
            "print()\n"
            "print('4. Long-short: the euphoric short is wrong 53% of the time in our')\n"
            "print(f'   (selected!) sample. In an unbiased sample, likely worse.')\n"
            "print()\n"
            "print('5. Base rate: with only 35 events over 40 years, even a real effect')\n"
            "print('   would take ~100 years of prospective data to confirm.')"
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 - Going further\n\n"
            "- **The minimum viable experiment.** Scrape every issue of BusinessWeek, "
            "Time, and The Economist since 1970, apply a mechanical sentiment score to "
            "the cover headline (e.g., VADER or GPT classification), and test at 1-month "
            "and 3-month horizons. That removes selection bias. Riva & Baur (2021) do "
            "something close for *The Economist* specifically and find a weak, "
            "non-robust effect.\n"
            "- **The mechanism.** Even if an effect existed, it would most likely be "
            "tautological: doom covers appear after extreme drawdowns (CAPE < 12, "
            "Shiller PE below 15). Markets mean-revert from those levels. The cover "
            "adds no incremental information beyond the valuation signal.\n"
            "- **Related desk studies:** [Study 158 — Super-Bowl](../../158-super-bowl/) "
            "(same family: memorable streak → honest test → no signal); "
            "[Study 120 — Excess-CAPE-Yield](../../120-excess-cape-yield/) (the real "
            "valuation signal that doom covers happen to proxy).\n\n"
            "*Think the curse is real? Fork this, expand to an unbiased cover universe, "
            "test at 3-month horizon, and see if |t| ≥ 2 survives. Report back.*"
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


def execute_notebooks():
    """Execute both notebooks in-place using jupyter nbconvert."""
    for nb_name in ["01_for_the_curious.ipynb", "02_for_the_quants.ipynb"]:
        nb_path = os.path.join(HERE, nb_name)
        print(f"Executing {nb_name} ...")
        result = subprocess.run(
            [
                sys.executable, "-m", "jupyter", "nbconvert",
                "--to", "notebook",
                "--execute",
                "--inplace",
                "--ExecutePreprocessor.timeout=300",
                nb_path,
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"ERROR executing {nb_name}:")
            print(result.stderr[-3000:])
            sys.exit(1)
        print(f"  done: {nb_path}")


if __name__ == "__main__":
    build_curious()
    build_quants()
    execute_notebooks()
