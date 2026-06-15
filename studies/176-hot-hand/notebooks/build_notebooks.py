"""Generate the two narrative notebooks for Study 176 (Hot-Hand).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). The synthetic
figures run anywhere, offline and deterministic; the real-tape cells fall back to the
frozen headline numbers in ``R`` (mirroring docs/results.md) when no cache is available,
so the notebook re-runs for any reader.

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
    n_days=8419,
    date_start="1993-01-04", date_end="2026-06-12",
    fingerprint="7bb4886dbdf1",
    uncond_bps=3.37, uncond_t=3.04,
    # Up-streak follow (hot-hand)
    hot_n=2379, hot_gross=0.28, hot_t=0.15, hot_net=-0.72, hot_net_t=-0.39,
    # Down-streak reversal (gambler's fallacy right)
    dn_n=742, dn_gross=22.31, dn_t=4.91, dn_net=21.31, dn_net_t=4.69,
    dn_win=0.598, dn_cagr=4.57, dn_sharpe=0.65,
    # Bonferroni
    n_tests=12, alpha_bonf=0.0042, sig_raw=8, sig_bonf=5,
    # Full streak table — continuation rates
    cont_up=[0.532, 0.551, 0.484, 0.495, 0.513, 0.462],
    cont_dn=[0.446, 0.464, 0.417, 0.396, 0.311, 0.261],
    ns_up=[2163, 1150, 634, 307, 152, 78],
    ns_dn=[2164, 965, 448, 187, 74, 23],
    excess_up=[-3.0, 0.0, -4.7, -10.4, -2.5, -0.7],
    excess_dn=[0.3, -0.7, 10.8, 17.4, 37.1, 56.2],
    tstats_up=[-1.13, 0.00, -1.24, -1.98, -0.30, -0.11],
    tstats_dn=[0.15, -0.17, 1.89, 1.27, 1.84, 1.58],
    bh_cagr=8.49,
    # Cost sweep
    dn_net_costs=[22.31, 21.81, 21.31, 20.31, 17.31],
    dn_net_t_costs=[4.91, 4.80, 4.69, 4.47, 3.81],
)

# ---------------------------------------------------------------------------
# Shared preamble
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

from hot_hand import data, strategy as st

CACHE = os.path.abspath(os.path.join("..", "_cache"))

def _have_cache():
    path = data._cache_path("^GSPC", CACHE)
    return os.path.exists(path)

HAVE_REAL = _have_cache()

def get_bars():
    return data.load_gspc(fetch=False, cache_dir=CACHE)

print("real daily cache present:", HAVE_REAL)
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Hot-Hand — does a winning streak predict the next day? 🎲\n"
            "### On a roll, or due for a turn? Both folk beliefs, one honest test\n\n"
            "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Streak_length%3F: None](https://img.shields.io/badge/Streak_length%3F-None-c0392b?style=flat-square)\n\n"
            "Two tribes of market watchers. The **hot-handers**: *\"SPY has been up five days "
            "in a row — it's on a roll, keep riding it.\"* The **gamblers**: *\"Five up days in "
            "a row? It's due for a breather — fade it.\"* Both beliefs are everywhere. Both "
            "cannot be right. And it turns out **one of them is completely wrong, the other is "
            "half-right for the wrong reason, and neither makes you money**. Let's look.\n\n"
            "> 📓 **This is the plain-language layer.** Want the binomial tests, the Bonferroni "
            "correction, and the HAC t-stats? That's **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** Every chart below is drawn by the code beside it. "
            "House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Streak type | Hot-hand (follow) | Gambler's fallacy (fade) |\n"
            "|---|---|---|\n"
            f"| **Up-streaks (1–6 days)** | **No edge.** The die is fair. | Also no edge. |\n"
            f"| **Down-streaks (3+ days)** | No. | **Yes — bounce is real.** *t* = **{R['dn_t']:.2f}**. |\n"
            f"| **Could you trade it?** | No. | Barely — only **{R['dn_cagr']:.1f}%/yr** vs "
            f"B&H **{R['bh_cagr']:.1f}%/yr**. |\n\n"
            "> The hot-hand on up-streaks is a myth. The gambler's fallacy on down-streaks "
            "is accidentally right — but it's not really a fallacy winning, it's panic "
            "selling followed by a well-known bounce."
        ),

        # ---- BEAT 1 — THE CLAIM ----------------------------------------------
        md(
            "## 1 · The two claims\n\n"
            "> *\"When the market has been up N days in a row, it's on a hot streak — "
            "trend-following says keep going.\"*\n\n"
            "vs.\n\n"
            "> *\"When the market has been up N days in a row, a reversal is overdue — "
            "the law of averages says it has to come back.\"*\n\n"
            "These are the **hot-hand fallacy** and the **gambler's fallacy**, translated "
            "from casino floors and basketball courts to the stock market. On a true random "
            "walk (a fair coin) *both* are equally wrong — there is no memory. On real markets "
            "the answer might be different, but it depends on the direction of the streak and "
            "how long it has run.\n\n"
            "We test both claims systematically: for each streak length 1 to 6, in both "
            "directions (up and down), does the next day look different from any other day?"
        ),

        # ---- BEAT 2 — SO WHAT ------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If up-streaks continued, a trend-follower would have an edge on the fastest "
            "possible signal: just count the last few days and ride. If down-streaks reversed, "
            "a contrarian could buy the dip mechanically after every three-day sell-off. "
            "These are dead-simple rules anyone can implement. The question is whether the "
            "pattern is real — or whether we're just seeing randomness and telling stories."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "Three things make the test honest:\n\n"
            "1. **The baseline.** The comparison isn't *zero* — it's the unconditional "
            f"next-day return, which is **{R['uncond_bps']:.2f} bps** (the market's long-run "
            "upward drift). A streak-conditioned bet has to beat that, not just beat a coin.\n"
            "2. **Multiple comparisons.** Testing 12 sub-hypotheses (6 lengths × 2 directions) "
            "means some will look significant by chance. We apply a Bonferroni correction — "
            f"the threshold tightens from 5% to {R['alpha_bonf']:.2%}. Of {R['n_tests']} "
            f"tests, {R['sig_raw']} look exciting raw; only {R['sig_bonf']} survive correction.\n"
            "3. **Honest stats.** We use a HAC t-stat (Newey-West) that accounts for the "
            "fact that daily returns are not perfectly independent."
        ),

        # ---- BEAT 4 — THE TEARDOWN -------------------------------------------
        md(
            "## 4 · The teardown\n\n"
            "**First: up-streaks — the hot-hand.** What is the continuation rate (probability "
            "the next day is also up) after 1, 2, 3 ... 6 consecutive up days?"
        ),
        code(
            "lens = list(range(1, 7))\n"
            "if HAVE_REAL:\n"
            "    bars = get_bars()\n"
            "    tbl = st.streak_table(bars['close'], max_streak=6)\n"
            "    cont_up = tbl[tbl['direction']=='up']['continuation'].tolist()\n"
            "    ns_up = tbl[tbl['direction']=='up']['n'].tolist()\n"
            f"else:\n"
            f"    cont_up = {R['cont_up']}\n"
            f"    ns_up = {R['ns_up']}\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "bars_plot = ax.bar(lens, [c*100 for c in cont_up], color=AMBER, alpha=0.85)\n"
            "ax.axhline(50, ls='--', c='k', lw=1.2, label='fair coin (50%)')\n"
            "ax.axhline(53.2, ls=':', c=GREY, lw=1, label='up-1 baseline (53.2%)')\n"
            "for b, n in zip(bars_plot, ns_up):\n"
            "    ax.annotate(f'n={n:,}', (b.get_x()+b.get_width()/2, 2),\n"
            "                ha='center', fontsize=8.5, color='white')\n"
            "ax.set_xlabel('Up-streak length'); ax.set_ylabel('P(next day also up) %')\n"
            "ax.set_title('Hot-hand on up-streaks: the rate stays near 50% — no trend')\n"
            "ax.set_ylim(0, 75); ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            "The continuation rate for up-streaks hovers around 50–55% regardless of "
            "how long the run has been. There is no 'getting hotter'. After 4, 5, or 6 "
            "consecutive up days, next-day continuation is statistically indistinguishable "
            "from a coin flip. **The hot-hand on up-streaks: absent.**\n\n"
            "**Now: down-streaks — does the gambler have a point?**"
        ),
        code(
            "if HAVE_REAL:\n"
            "    cont_dn = tbl[tbl['direction']=='down']['continuation'].tolist()\n"
            "    ns_dn = tbl[tbl['direction']=='down']['n'].tolist()\n"
            f"else:\n"
            f"    cont_dn = {R['cont_dn']}\n"
            f"    ns_dn = {R['ns_dn']}\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.5))\n"
            "bars_plot = ax.bar(lens, [c*100 for c in cont_dn], color=RED, alpha=0.85)\n"
            "ax.axhline(50, ls='--', c='k', lw=1.2, label='fair coin (50%)')\n"
            "for b, n in zip(bars_plot, ns_dn):\n"
            "    ax.annotate(f'n={n:,}', (b.get_x()+b.get_width()/2, 2),\n"
            "                ha='center', fontsize=8.5, color='white')\n"
            "ax.set_xlabel('Down-streak length'); ax.set_ylabel('P(next day also down) %')\n"
            "ax.set_title('Gambler on down-streaks: continuation drops as the sell-off deepens')\n"
            "ax.set_ylim(0, 75); ax.legend(); plt.tight_layout(); plt.show()"
        ),
        md(
            "This is the punchline. After each additional down day, the probability of "
            "**another** down day keeps falling — from 44.6% after 1 down day to 26.1% "
            "after 6 consecutive down days. The market bounces. The gambler's fallacy is "
            "correct here — but *not* because of some law of averages. It's because sustained "
            "sell-offs trigger panic, and panic tends to overshoot, and overshooting tends to "
            "reverse. That's a real, well-studied market mechanism — not a fallacy at all."
        ),
        md("**What does the excess return look like?**"),
        code(
            "if HAVE_REAL:\n"
            "    excess_up = tbl[tbl['direction']=='up']['excess_bps'].tolist()\n"
            "    excess_dn = tbl[tbl['direction']=='down']['excess_bps'].tolist()\n"
            f"else:\n"
            f"    excess_up = {R['excess_up']}\n"
            f"    excess_dn = {R['excess_dn']}\n"
            "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "ax1.bar(lens, excess_up, color=[GREEN if v>0 else RED for v in excess_up], alpha=0.85)\n"
            "ax1.axhline(0, c='k', lw=1)\n"
            "ax1.set_title('Up-streak: excess return vs buy-and-hold')\n"
            "ax1.set_xlabel('Streak length'); ax1.set_ylabel('excess bps/day')\n"
            "ax2.bar(lens, excess_dn, color=[GREEN if v>0 else RED for v in excess_dn], alpha=0.85)\n"
            "ax2.axhline(0, c='k', lw=1)\n"
            "ax2.set_title('Down-streak fade: excess return vs buy-and-hold')\n"
            "ax2.set_xlabel('Streak length'); ax2.set_ylabel('excess bps/day')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'After 3+ down days: mean gross excess = +10.8 to +56.2 bps vs uncond baseline')"
        ),

        # ---- BEAT 5 — THE VERDICT --------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Mixed.** Up-streak hot-hand: **None** (*t* = +{R['hot_t']:.2f}, a coin). "
            f"Down-streak reversal: **Real** (*t* = +{R['dn_t']:.2f}, n = {R['dn_n']:,}, "
            "robust to excluding the 2008–09 crisis, *t* = +5.24).\n"
            "- **Tradability — Fragile.** The down-reversal survives costs — but delivers only "
            f"~{R['dn_cagr']:.1f}%/yr CAGR vs S&P 500 B&H ~{R['bh_cagr']:.1f}%/yr. It fires "
            "only ~22 times per year, sitting in cash otherwise — you underperform simply holding.\n"
            "- **Streak length info? — None.** Longer up-streaks carry zero extra information; "
            "longer down-streaks show stronger reversal but tiny n (74 events at length 5, 23 at "
            f"length 6) — and only {R['sig_bonf']} of {R['n_tests']} cells survive Bonferroni."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ------------------------------------
        md(
            "## 6 · Could you actually trade it?\n\n"
            "The hot-hand rule dies instantly — it's a coin. What about the down-reversal?"
        ),
        code(
            "costs = [0.0, 0.5, 1.0, 2.0, 5.0]\n"
            f"net_means = {R['dn_net_costs']}\n"
            "if HAVE_REAL:\n"
            "    bars = get_bars()\n"
            "    net_means = []\n"
            "    for c in costs:\n"
            "        ld = st.trade_streak(bars['close'], min_streak=3, direction=-1,\n"
            "                             follow_streak=False, cost_bps=c)\n"
            "        col = 'ret_net' if c > 0 else 'ret_gross'\n"
            "        net_means.append(st.summarize(ld, col)['mean_bps'])\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(costs, net_means, 'o-', c=AMBER, lw=2)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.fill_between(costs, net_means, 0, where=[n>0 for n in net_means], "
            "color=GREEN, alpha=.12)\n"
            f"ax.axhline({R['uncond_bps']}, ls='--', c=GREY, lw=1.2, "
            "label='unconditional B&H')\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net mean (bps/trade)')\n"
            "ax.set_title('Down-reversal survives costs — but still lags buy-and-hold')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            f"print('CAGR net (@1bp): ~{R['dn_cagr']:.1f}%/yr  |  S&P 500 B&H: ~{R['bh_cagr']:.1f}%/yr')"
        ),
        md(
            f"The down-reversal survives realistic costs — the line stays positive even at 5 bps. "
            "But with ~22 trades per year, **sitting in cash the rest of the time**, you earn "
            f"only about **{R['dn_cagr']:.1f}%/yr** while the index itself returns "
            f"**{R['bh_cagr']:.1f}%/yr**. You're getting paid to buy during panics — that's a "
            "real edge — but the opportunity cost of missing the calm drift eats most of it."
        ),

        # ---- BEAT 7 — GOING FURTHER ------------------------------------------
        md(
            "## 7 · Going further\n\n"
            "- **What would make the hot-hand real?** The companion notebook shows a synthetic "
            "positive control: when we plant daily AR(1) autocorrelation, the streak-follow rule "
            "finds it. The real S&P 500 simply has near-zero daily autocorrelation at the index "
            "level — individual stocks and shorter windows are different.\n"
            "- **The down-reversal is just short-term reversion.** See "
            "[Study 72 — Loaded-Dice](../../72-loaded-dice/) for the intraday analogue, "
            "and De Bondt & Thaler (1985) for the long-horizon version.\n"
            "- **Does the reversal work on individual stocks?** Different question — microstructure "
            "effects are much stronger there. Fork this and replace ^GSPC with a single ticker.\n\n"
            "*Think you can turn the down-reversal into something that beats B&H? Add volatility "
            "scaling (size by the size of the sell-off), a proper overnight position, or a "
            "longer hold period. That's the fork worth making.*"
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
            "# Hot-Hand — a quantitative teardown\n"
            "### S&P 500 daily streaks · binomial tests · HAC inference · Bonferroni · positive control\n\n"
            "![Signal: Mixed](https://img.shields.io/badge/Signal-Mixed-dab617?style=flat-square)\n"
            "![Tradability: Fragile](https://img.shields.io/badge/Tradability-Fragile-dab617?style=flat-square)\n"
            "![Streak_length%3F: None](https://img.shields.io/badge/Streak_length%3F-None-c0392b?style=flat-square)\n\n"
            "The deep companion to [01_for_the_curious.ipynb](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We test whether N "
            "consecutive up-days (or down-days) predicts the direction of day N+1 on the S&P 500, "
            "across streak lengths 1–6 in both directions, with exact binomial tests, HAC t-stats, "
            "and a Bonferroni correction across 12 simultaneous hypotheses.\n\n"
            "> ⚠️ **Not investment advice.** Real data: Yahoo daily ^GSPC 1993-01-04 to "
            f"2026-06-12 (n = {R['n_days']:,}); the offline core and tests run on a "
            "deterministic synthetic tape. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **`💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT + "\nfrom quantlab import analytics, stats\n"),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `MIXED` | Up-streak hot-hand: *t* = **+{R['hot_t']:.2f}** (NONE). "
            f"Down-streak reversal: *t* = **+{R['dn_t']:.2f}**, n = {R['dn_n']:,} (REAL). "
            f"Bonferroni: {R['sig_bonf']}/{R['n_tests']} cells survive (all down). |\n"
            f"| **Tradability** | `FRAGILE` | Down-reversal net *t* = +{R['dn_net_t']:.2f} at 1 bp, "
            f"CAGR ~{R['dn_cagr']:.1f}%/yr — underperforms B&H (~{R['bh_cagr']:.1f}%/yr). |\n"
            f"| **Streak length?** | `NONE` | Longer up-streaks: zero incremental signal. "
            f"Longer down-streaks: stronger reversal but n ↓ precipitously (n=74 at length 5). |\n\n"
            "> 💡 The folk debate is settled asymmetrically: up-streaks are a coin; "
            "down-streaks really do reverse — but via a known mechanism (panic overreaction), "
            "not via the gambler's fallacy logic."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claims, steelmanned\n\n"
            "Let $r_t$ be the daily log return of the S&P 500 and $L_t$ the number of "
            "consecutive same-direction days ending on day $t$.\n\n"
            "- **H₁ (hot-hand).** $\\mathbb{E}[r_{t+1} \\mid L_t = n, \\text{sign}(r_t)=+1] > "
            "\\mathbb{E}[r_{t+1}]$ — after $n$ consecutive up days, the conditional expectation "
            "exceeds unconditional buy-and-hold.\n"
            "- **H₂ (gambler).** $\\mathbb{E}[r_{t+1} \\mid L_t = n, \\text{sign}(r_t)=-1] > "
            "\\mathbb{E}[r_{t+1}]$ when *fading* — the reverse bet after $n$ down days beats "
            "unconditional buy-and-hold.\n"
            "- **H₃ (streak length matters).** The conditional expectation is monotone in $n$ — "
            "longer runs carry proportionally more information.\n\n"
            "Verdict: H₁ rejected, H₂ confirmed for $n\\geq 3$, H₃ rejected."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "H₁ failing is the first lesson: at the one-day horizon the S&P 500 has negligible "
            "positive autocorrelation at the index level (Lo & MacKinlay 1988 found it in small "
            "caps, not the index). H₂ holding for down-streaks is the second lesson: it is the "
            "well-known short-term mean reversion / overreaction correction (Cox & Peterson 1994, "
            "De Bondt & Thaler 1985), not a folk superstition being vindicated. H₃ failing is the "
            "main guard against over-engineering: you cannot progressively exploit longer streaks "
            "because the n shrinks too fast for reliable inference."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Data.** S&P 500 daily close (^GSPC, Yahoo!, auto-adjusted), 1993-01-04 to "
            f"2026-06-12 (n = {R['n_days']:,} days, fingerprint `{R['fingerprint']}`).\n"
            "- **Streak label.** For each day $t$, compute $L_t$ = length of the current run. "
            "Enter a long position at $t$'s close, exit at $(t+1)$'s close (one holding day).\n"
            "- **Baseline.** Unconditional mean next-day return = "
            f"**{R['uncond_bps']:.2f} bps** (HAC *t* = +{R['uncond_t']:.2f}).\n"
            "- **Inference.** Exact two-sided binomial test on continuation count; HAC t-stat "
            "(Newey-West) on excess return over the unconditional mean.\n"
            "- **Multiple comparisons.** Bonferroni across 12 tests "
            f"(alpha_adj = {R['alpha_bonf']:.4f}).\n"
            "- **Positive control.** Synthetic tape with tunable daily AR(1) — confirms the "
            "engine recovers signal when it exists."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · Full streak table — 12 cells, 5 survive Bonferroni\n\n"
            "For each streak length 1–6 and direction, the continuation rate and the "
            "HAC *t*-stat on the excess return."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bars = get_bars()\n"
            "    tbl = st.streak_table(bars['close'], max_streak=6, cost_bps=1.0)\n"
            "    bf = st.bonferroni_summary(tbl)\n"
            "    alpha_bonf = bf['alpha_bonf']\n"
            "    cont_up = tbl[tbl['direction']=='up']['continuation'].tolist()\n"
            "    cont_dn = tbl[tbl['direction']=='down']['continuation'].tolist()\n"
            "    tstats_up = tbl[tbl['direction']=='up']['tstat'].tolist()\n"
            "    tstats_dn = tbl[tbl['direction']=='down']['tstat'].tolist()\n"
            "    binom_up = tbl[tbl['direction']=='up']['binom_p'].tolist()\n"
            "    binom_dn = tbl[tbl['direction']=='down']['binom_p'].tolist()\n"
            f"else:\n"
            f"    alpha_bonf = {R['alpha_bonf']}\n"
            f"    cont_up, cont_dn = {R['cont_up']}, {R['cont_dn']}\n"
            f"    tstats_up, tstats_dn = {R['tstats_up']}, {R['tstats_dn']}\n"
            f"    binom_up = [None]*6; binom_dn = [None]*6\n"
            "lens = list(range(1, 7))\n"
            "fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))\n"
            "for ax, cont, ts, lbl, col in [\n"
            "        (axs[0], cont_up, tstats_up, 'Up-streak continuation', AMBER),\n"
            "        (axs[1], cont_dn, tstats_dn, 'Down-streak continuation (reversal = 1-cont)', RED)]:\n"
            "    ax.bar(lens, [c*100 for c in cont],\n"
            "           color=[GREEN if abs(t) >= 2 else col for t in ts], alpha=0.85)\n"
            "    ax.axhline(50, ls='--', c='k', lw=1.2)\n"
            "    for i, (c, t) in enumerate(zip(cont, ts)):\n"
            "        ax.annotate(f't={t:+.1f}', (i+1, c*100+1), ha='center', fontsize=8)\n"
            "    ax.set_xlabel('Streak length'); ax.set_ylabel('Continuation %')\n"
            "    ax.set_title(lbl); ax.set_ylim(0, 70)\n"
            "plt.tight_layout(); plt.show()\n"
            f"print(f'Bonferroni alpha_adj={{alpha_bonf:.4f}}')"
        ),
        md(
            "> 💡 Up-streaks (left panel): t-stats hover near zero and the continuation "
            "rate barely moves from 50%. No hot-hand.\n"
            "> Down-streaks (right panel): continuation keeps falling — the market "
            "reverses more often after longer sell-offs. The t-stats on the excess return "
            "are +1.89 (down-3) through +1.84 (down-5), borderline by HAC standards but "
            "real when pooled (down 3+: *t* = +4.91)."
        ),
        md(
            "### 4b · Down-streak reversal pooled (≥ 3 consecutive down days)\n\n"
            "Aggregating all observations with a down-streak of 3 or more days for power."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bars = get_bars()\n"
            "    ld = st.trade_streak(bars['close'], min_streak=3, direction=-1,\n"
            "                         follow_streak=False, cost_bps=0.0)\n"
            "    sg = st.summarize(ld, 'ret_gross')\n"
            "    ld_net = st.trade_streak(bars['close'], min_streak=3, direction=-1,\n"
            "                              follow_streak=False, cost_bps=1.0)\n"
            "    sn = st.summarize(ld_net, 'ret_net')\n"
            "    dn_n = sg['n_trades']; dn_gross = sg['mean_bps']; dn_t = sg['tstat']\n"
            "    dn_win = sg['win_rate']; dn_net = sn['mean_bps']; dn_net_t = sn['tstat']\n"
            f"else:\n"
            f"    dn_n, dn_gross, dn_t = {R['dn_n']}, {R['dn_gross']}, {R['dn_t']}\n"
            f"    dn_win, dn_net, dn_net_t = {R['dn_win']}, {R['dn_net']}, {R['dn_net_t']}\n"
            "print(f'Down-streak reversal (3+): n={dn_n}, gross={dn_gross:+.2f}bps, '\n"
            "      f't={dn_t:+.3f}, win={dn_win:.3f}, net={dn_net:+.2f}bps, net_t={dn_net_t:+.3f}')\n"
            "print(f'Excluding 2008-09:')\n"
            "if HAVE_REAL:\n"
            "    bars_ex = bars[~bars.index.year.isin([2008, 2009])]\n"
            "    ld_ex = st.trade_streak(bars_ex['close'], min_streak=3, direction=-1,\n"
            "                             follow_streak=False, cost_bps=0.0)\n"
            "    se = st.summarize(ld_ex, 'ret_gross')\n"
            "    print(f'  n={se[\"n_trades\"]}, gross={se[\"mean_bps\"]:+.2f}bps, t={se[\"tstat\"]:+.3f}')\n"
            "else:\n"
            "    print('  n=701, gross=+21.18bps, t=+5.24 (more significant without crisis)')"
        ),
        md(
            "> 💡 The reversal *strengthens* when the 2008–09 crisis is excluded — confirming "
            "it is not driven by a single extreme regime. The effect is a persistent property "
            "of the S&P 500's short-term return dynamics."
        ),
        md(
            "### 4c · Hot-hand rule and gambler's fade on up-streaks — both are coins\n\n"
            "For completeness: both hot-hand and gambler's fade applied to up-streaks."
        ),
        code(
            "if HAVE_REAL:\n"
            "    bars = get_bars()\n"
            "    lh = st.trade_streak(bars['close'], min_streak=2, direction=1,\n"
            "                          follow_streak=True, cost_bps=0.0)\n"
            "    lg = st.trade_streak(bars['close'], min_streak=2, direction=1,\n"
            "                          follow_streak=False, cost_bps=0.0)\n"
            "    sh = st.summarize(lh, 'ret_gross')\n"
            "    sg = st.summarize(lg, 'ret_gross')\n"
            "    hm, ht = sh['mean_bps'], sh['tstat']\n"
            "    gm, gt = sg['mean_bps'], sg['tstat']\n"
            f"else:\n"
            f"    hm, ht = {R['hot_gross']}, {R['hot_t']}\n"
            f"    gm, gt = -{R['hot_gross']}, -{R['hot_t']}\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "results = [('Hot-hand\\n(follow up-streak)', hm, ht),\n"
            "           ('Gambler fade\\n(fade up-streak)', gm, gt),\n"
            f"           ('Unconditional\\nbuy-and-hold', {R['uncond_bps']}, {R['uncond_t']})]\n"
            "ax.bar([r[0] for r in results], [r[1] for r in results],\n"
            "       color=[GREY, GREY, GREEN], alpha=0.85)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "for r in results:\n"
            "    ax.annotate(f't={r[2]:+.2f}', (r[0], r[1]+(0.3 if r[1]>=0 else -0.5)),\n"
            "                ha='center', fontsize=9)\n"
            "ax.set_ylabel('gross mean (bps/trade)'); ax.set_title('Up-streak rules vs baseline')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'Hot-hand: {hm:+.2f} bps, t={ht:+.2f} | Gambler: {gm:+.2f} bps, t={gt:+.2f}')"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `MIXED`** — Hot-hand (up-streaks): *t* = **+{R['hot_t']:.2f}**, NONE. "
            f"Down-streak reversal (≥3): *t* = **+{R['dn_t']:.2f}**, REAL. "
            f"Bonferroni: {R['sig_bonf']}/{R['n_tests']} cells survive.\n"
            f"- **Tradability `FRAGILE`** — Down-reversal: net *t* = +{R['dn_net_t']:.2f} at 1 bp, "
            f"CAGR **{R['dn_cagr']:.1f}%/yr** vs B&H **{R['bh_cagr']:.1f}%/yr**. "
            f"Annualised Sharpe {R['dn_sharpe']:.2f}. Underperforms passive.\n"
            f"- **Streak length `NONE`** — no monotone info in up-streaks; "
            "down-streak reversal stronger at longer lengths but n collapses (23 events at length 6)."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Could you trade it? — the cost sweep\n\n"
            "Down-streak reversal at varying round-trip cost, ~22 trades/year."
        ),
        code(
            "costs = [0.0, 0.5, 1.0, 2.0, 5.0]\n"
            f"net_means = {R['dn_net_costs']}\n"
            f"net_tsts = {R['dn_net_t_costs']}\n"
            "if HAVE_REAL:\n"
            "    bars = get_bars()\n"
            "    net_means, net_tsts = [], []\n"
            "    for c in costs:\n"
            "        ld = st.trade_streak(bars['close'], min_streak=3, direction=-1,\n"
            "                             follow_streak=False, cost_bps=c)\n"
            "        col = 'ret_net' if c > 0 else 'ret_gross'\n"
            "        s = st.summarize(ld, col)\n"
            "        net_means.append(s['mean_bps']); net_tsts.append(s['tstat'])\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(costs, net_means, 'o-', c=AMBER, lw=2, label='net mean (bps/trade)')\n"
            "ax2 = ax.twinx(); ax2.plot(costs, net_tsts, 's--', c=GREY, lw=1.5, label='HAC t')\n"
            "ax2.axhline(2, ls=':', c=GREEN, lw=1); ax2.axhline(0, ls=':', c=GREY, lw=0.8)\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('round-trip cost (bps)'); ax.set_ylabel('net mean (bps)', color=AMBER)\n"
            "ax2.set_ylabel('HAC t-stat', color=GREY)\n"
            "ax.set_title('Down-reversal: survives costs but CAGR lags buy-and-hold')\n"
            "plt.tight_layout(); plt.show()\n"
            f"print('CAGR net (@1bp): ~{R['dn_cagr']:.1f}%/yr  vs  B&H ~{R['bh_cagr']:.1f}%/yr')\n"
            "print('The strategy is in cash ~95% of the time -- opportunity cost dominates')"
        ),
        md(
            "> 💡 The t-stat stays above 2 even at 5 bps round-trip — the gross edge is "
            "genuine. But the *opportunity cost* kills it: ~22 trades per year with ~1-day "
            "holds means you are in market about 22/252 = 9% of the time. The remaining "
            "91% in cash earns nothing while buy-and-hold compounds. This is a structural "
            "ceiling no amount of cost cutting can fix."
        ),

        # ---- BEAT 7 ----------------------------------------------------------
        md(
            "## 7 · Going further — the positive control\n\n"
            "Is the streak engine capable of finding signal, or does it always print 'none'? "
            "Plant a daily AR(1) autocorrelation in a synthetic tape:"
        ),
        code(
            "autocorrs = [-0.20, -0.10, 0.0, 0.10, 0.20]\n"
            "results = []\n"
            "for ac in autocorrs:\n"
            "    b, _ = data.synthetic_daily(n_days=5000, autocorr=ac, seed=176)\n"
            "    lh = st.trade_streak(b['close'], min_streak=2, direction=0,\n"
            "                          follow_streak=True, cost_bps=0.0)\n"
            "    sh = st.summarize(lh, 'ret_gross')\n"
            "    lf = st.trade_streak(b['close'], min_streak=2, direction=0,\n"
            "                          follow_streak=False, cost_bps=0.0)\n"
            "    sf = st.summarize(lf, 'ret_gross')\n"
            "    results.append((ac, sh['mean_bps'], sf['mean_bps']))\n"
            "acs, follow_means, fade_means = zip(*results)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.plot(acs, follow_means, 'o-', c=AMBER, lw=2, label='follow streak (hot-hand)')\n"
            "ax.plot(acs, fade_means, 's--', c=GREEN, lw=2, label='fade streak (gambler)')\n"
            "ax.axhline(0, c='k', lw=1); ax.axvline(0, ls=':', c=GREY)\n"
            "ax.set_xlabel('planted daily AR(1) autocorrelation')\n"
            "ax.set_ylabel('gross excess bps/trade')\n"
            "ax.set_title('Positive control: follow wins with momentum, fade wins with reversion')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('The engine works — it just finds near-zero autocorrelation in real S&P 500 up-runs')"
        ),
        md(
            "The streak engine is a faithful autocorrelation detector: follow wins when "
            "autocorr > 0 (momentum), fade wins when autocorr < 0 (reversion). On the real "
            "S&P 500, up-streaks show autocorr ≈ 0 (hot-hand absent), while the down-streak "
            "reversal shows autocorr < 0 (panic overshoot → reversion). The signal is real; "
            "it's just a known market mechanism at a specific horizon, not magic."
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
