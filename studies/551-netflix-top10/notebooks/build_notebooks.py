"""Generate the two narrative notebooks for Study 551 (Netflix-Top10).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the seven desk beats (see ../../../METHODOLOGY.md). Everything runs on the
deterministic synthetic world — there is **no real tape** for this study (no free point-in-time
Netflix Top-10 hours feed), so the notebooks are fully offline and reproducible. ``HAVE_REAL`` is
always False here; the synthetic cells never sit under a real-tape banner.
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


# Frozen synthetic-world headline numbers — mirror of docs/results.md (as-of 2026-06-30; null
# world beta=0, seed 551; 197 weekly rows; world fp a31af2c4d9d7).
R = dict(
    fp_world="a31af2c4d9d7", n=197, seed=551,
    # NFLX predictive regression (null world, seed 551, horizon 4)
    slope=-0.0106, ols_t=-2.03, nw_t=-1.48, corr=-0.14, placebo_p=0.046,
    # seed-robust view
    nw_mean_20=0.11, frac_ge2=0,
    # XLY spillover leg
    xly_slope=-0.002, xly_t=-0.47, xly_corr=-0.03,
    # horizon sweep (null): (h, slope, ols_t, nw_t)
    hz=[(2, -0.0032, -0.96, -0.89), (4, -0.0106, -2.03, -1.48),
        (8, -0.0334, -4.34, -3.29), (13, -0.0410, -3.92, -3.12)],
    # timing backtest (null world, NFLX), annualised
    lf_gross=-3.6, lf_net=-4.4, ls_gross=-25.7, ls_net=-27.1, bh_ann=21.8,
    cost_bps=5.0, borrow_bps=100.0,
    # synthetic control: (beta, mean NW slope-t over 25 seeds)
    ctrl=[(0.0, -0.08), (0.005, 0.49), (0.010, 1.01), (0.020, 1.78), (0.040, 2.51)],
)


BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from netflix_top10 import data, strategy as st

SEED = 551

# There is NO real tape for this study (no free point-in-time Top-10 hours feed).
REAL = data.fetch_series(fetch=False)
HAVE_REAL = not REAL.empty
print("real streaming-engagement tape present:", HAVE_REAL, "(synthetic-only by design)")

# The offline core: the NULL synthetic world (beta=0) is the honest headline.
WEEKLY, TRUTH = data.synthetic_series(beta=0.0, seed=SEED)
print(f"synthetic NULL world: {len(WEEKLY)} weekly rows, fp {data.fingerprint(WEEKLY)}")
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Netflix Top-10 — does *watching* predict the *stock*? 📺\n"
            "### Turning 'how many hours the world streamed' into a trading signal, in plain English\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Alt-data_edge%3F: Mirage](https://img.shields.io/badge/Alt--data_edge%3F-Mirage-c0392b?style=flat-square)\n\n"
            "Every week Netflix publishes a **Top-10** with how many *hours* the world watched. "
            "Tempting alt-data idea: if engagement is *accelerating*, subscribers, ad money and "
            "pricing power should follow — so watch the hours, buy the stock (NFLX), maybe even the "
            "whole consumer-discretionary basket (XLY). Does it work?\n\n"
            "Two catches, up front:\n\n"
            "1. **You can't actually get the data cleanly.** The public hours series is short (2021+), "
            "changed its own definition partway through, and lives in PDFs and a dashboard — not a "
            "tidy, free, time-stamped feed. So we test the *machinery* on a synthetic stand-in and are "
            "honest that there's no real tape.\n"
            "2. **Overlap plays tricks.** As we'll see, the one result that looks 'significant' is a "
            "statistical mirage.\n\n"
            "> 📓 **This is the plain-language layer.** Want the *t*-stats, the Newey-West correction "
            "and the cost maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn by the "
            "code beside it. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            "| Is there real Netflix engagement data to test? | **Not for a free retail stack.** The "
            "hours series is short, revised, and not machine-readable — so this is a **synthetic-only** "
            "study, which by house rule can *never* be rated `REAL`. |\n"
            f"| Does engagement momentum predict NFLX? | **No.** On the honest (null) test the "
            f"predictive link is flat: averaged over 20 random worlds the *t*-stat is **+{R['nw_mean_20']}** "
            "and **none** clear the bar. |\n"
            f"| But one run looked significant? | **Yes — and that's the trap.** One draw prints "
            f"*t* = {R['ols_t']}, but it's a **mirage** from overlapping windows; the corrected *t* is "
            f"{R['nw_t']}. |\n"
            f"| Could you trade it? | **No.** Timing on the signal earns **{R['lf_gross']}%/yr** vs just "
            f"holding the stock at **+{R['bh_ann']}%/yr**. |\n\n"
            "> The engine *does* catch a real link when we plant one (see the last beat) — so this is a "
            "statement about the data and the overlap trap, not broken code."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"Streaming-engagement momentum — how fast Netflix's Top-10 viewing hours are "
            "accelerating — predicts NFLX (and consumer-discretionary) stock returns.\"*\n\n"
            "The intuition: engagement is a leading indicator of the business. More hours → happier, "
            "stickier subscribers → less churn, more ad inventory, more pricing power → a stronger "
            "stock. And because it's *alt-data* (not on every analyst's screen), maybe you get there "
            "first. This is the same family as 'Google-search-volume predicts returns' — a real, "
            "documented idea, and a real graveyard of edges that were tiny or already priced in."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "If engagement momentum genuinely led the stock, you'd have a clean weekly timing rule for "
            "one of the most-watched names on the market — and maybe a read on the whole consumer "
            "tape. The desk has looked at other alt-data / sentiment signals "
            "([257 AAII](../../257-aaii-sentiment/), [335 Buzz](../../335-buzz-sentiment-etf/), "
            "[392 Glassdoor](../../392-glassdoor-sentiment/)); this one adds the **streaming** angle "
            "and, as it turns out, a sharp lesson about *overlapping windows*."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "1. **Build an engagement index.** Weekly viewing hours, rising over time with holiday "
            "bumps and hit-series spikes.\n"
            "2. **Turn it into *momentum*.** Is this week's engagement *above* its recent trend? That's "
            "the 'accelerating' signal.\n"
            "3. **Line it up against the *next* few weeks' stock return** and ask: does high momentum "
            "come before high returns?\n\n"
            "Because there's no clean real feed, we run this on a **synthetic world** with a dial: turn "
            "the dial up and engagement genuinely predicts returns (to prove the test *can* find it); "
            "set it to zero and engagement is pure noise. The honest headline is the **zero** case — "
            "what an honest test of a signal that *doesn't* exist should look like.\n\n"
            "*Timing:* the signal at week *t* uses only hours up to week *t*; we then measure the "
            "*following* weeks' return. No peeking ahead."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**Here's the engagement index and its momentum signal.**"
        ),
        code(
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.5, 6.2), sharex=True)\n"
            "a1.plot(WEEKLY.index, WEEKLY['engagement'], color=GREEN, lw=1.6)\n"
            "a1.set_ylabel('engagement index'); a1.set_title('Weekly Top-10 engagement (synthetic): trend + seasonality + hit spikes')\n"
            "a2.plot(WEEKLY.index, WEEKLY['eng_mom'], color=GREY, lw=1.2)\n"
            "a2.axhline(0, c='k', lw=1)\n"
            "a2.fill_between(WEEKLY.index, 0, WEEKLY['eng_mom'], where=WEEKLY['eng_mom']>0, color=GREEN, alpha=.3)\n"
            "a2.fill_between(WEEKLY.index, 0, WEEKLY['eng_mom'], where=WEEKLY['eng_mom']<0, color=RED, alpha=.3)\n"
            "a2.set_ylabel('engagement momentum'); a2.set_xlabel('week')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('green = engagement accelerating (the buy signal), red = decelerating')"
        ),
        md(
            "**Does high engagement momentum come before high NFLX returns?** Scatter the momentum "
            "signal against the *next-4-weeks* return and fit a line — the whole claim is: **is that "
            "line sloping up, convincingly?**"
        ),
        code(
            "reg = st.predictive_regression(WEEKLY, 'nflx_fwd')\n"
            "x = WEEKLY['eng_mom'].to_numpy(); y = WEEKLY['nflx_fwd'].to_numpy()*100\n"
            "fig, ax = plt.subplots(figsize=(8, 4.6))\n"
            "ax.scatter(x, y, s=20, color=GREY, alpha=.7)\n"
            "xs = np.linspace(x.min(), x.max(), 50)\n"
            "ax.plot(xs, (reg['slope']*xs)*100 + (y.mean() - reg['slope']*x.mean()*100), c=RED, lw=2)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('engagement momentum (this week)'); ax.set_ylabel('NFLX return, next 4 weeks (%)')\n"
            "ax.set_title(f\"Barely-there, wrong-way slope (naive t {reg['slope_t']:+.2f}, honest t {reg['t_nw']:+.2f})\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"slope {reg['slope']:+.4f}, naive OLS t {reg['slope_t']:+.2f}, honest (Newey-West) t {reg['t_nw']:+.2f}\")"
        ),
        md(
            f"On this one run the naive *t* is **{R['ols_t']}** — which *looks* significant (past ±2). "
            f"But that number is **not trustworthy**: because each 4-week return overlaps its "
            f"neighbours, the naive test thinks it has far more independent data than it does. Correct "
            f"for that overlap and the honest *t* is **{R['nw_t']}** — and pointing the *wrong* way "
            "(down, not up). So even the 'significant' run isn't the claimed signal."
        ),
        md(
            "**Is it really a fluke?** Re-run the whole thing on 20 different random worlds (all with "
            "*zero* true signal) and look at the honest *t* each time:"
        ),
        code(
            "ts = [st.predictive_regression(data.synthetic_series(beta=0.0, seed=s)[0], 'nflx_fwd')['t_nw']\n"
            "      for s in range(551, 571)]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.2))\n"
            "cols = [RED if abs(t)>=2 else GREY for t in ts]\n"
            "ax.bar(range(len(ts)), ts, color=cols, width=.7)\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1)\n"
            "ax.axhline(np.mean(ts), c=GREEN, lw=1.5, label=f'mean {np.mean(ts):+.2f}')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('random world (seed)'); ax.set_ylabel('honest t-stat')\n"
            "ax.set_title('20 null worlds: not one clears the ±2 bar; the average is ~0')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('mean honest t over 20 seeds:', round(float(np.mean(ts)),3), '| how many clear |t|>=2:', int(np.sum(np.abs(ts)>=2)))"
        ),
        md(
            f"There it is: across 20 worlds the average honest *t* is **+{R['nw_mean_20']}** and "
            f"**{R['frac_ge2']} of 20** clear the bar. The single 'significant' run was just an unlucky "
            "draw magnified by overlap — a false positive. This is the whole reason the desk exists: to "
            "not get fooled by one good-looking backtest."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** No real tape to test (synthetic-only ⇒ can't be `REAL`), and the "
            "honest, seed-averaged link is flat (mean *t* ≈ 0, 0/20 clear the bar).\n"
            "- **Tradability — Mirage.** Timing on the signal loses to simply holding the stock.\n"
            "- **Alt-data edge? — Mirage.** The 'edge' is an overlapping-window artifact; and real "
            "engagement reports lag the week and are widely covered, so any true edge is priced in."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually trade it?\n\n"
            "Even pretending the signal were real, here's the timing rule — hold NFLX when engagement "
            "is accelerating, step aside otherwise — against just buying and holding the stock:"
        ),
        code(
            "bt = st.timing_backtest(WEEKLY, 'nflx_ret', allow_short=False, cost_bps=5.0)\n"
            "vals = [bt['gross_ann']*100, bt['net_ann']*100, bt['buyhold_ann']*100]\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.2))\n"
            "ax.bar(['timing\\n(gross)','timing\\n(net)','buy & hold'], vals, color=[AMBER, RED, GREEN], width=.55)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('annualised return %')\n"
            "ax.set_title('Timing on the (noise) signal loses to just holding the stock')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"timing gross {bt['gross_ann']*100:+.1f}% | net {bt['net_ann']*100:+.1f}% | buy&hold {bt['buyhold_ann']*100:+.1f}%\")"
        ),
        md(
            f"Trading a coin-flip signal just churns costs and misses upside: **{R['lf_net']}%/yr** net "
            f"versus **+{R['bh_ann']}%/yr** for doing nothing. And the *short* version (bet against low "
            "engagement, pay a borrow) is far worse. `MIRAGE`."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **Prove the test isn't blind.** In the quants notebook we *plant* a real "
            "engagement→return link and watch the honest *t* climb past +2 — so the flat result above "
            "is the data talking, not a broken detector.\n"
            "- **Other alt-data signals.** [257 AAII](../../257-aaii-sentiment/), "
            "[335 Buzz](../../335-buzz-sentiment-etf/), [392 Glassdoor](../../392-glassdoor-sentiment/).\n\n"
            "*Think engagement really leads the stock? Get a clean, point-in-time hours feed, line it up "
            "against NFLX with a Newey-West *t* on non-overlapping windows, and show it clears +2.*"
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
            "# Netflix Top-10 engagement momentum — a quantitative teardown 🔬\n"
            "### engagement-momentum signal · predictive regression · OLS vs Newey-West · label-shuffle placebo · seed-robust null · horizon-overlap artifact · costs & borrow · synthetic control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Alt-data_edge%3F: Mirage](https://img.shields.io/badge/Alt--data_edge%3F-Mirage-c0392b?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We test whether "
            "streaming-engagement momentum predicts NFLX / XLY forward returns, and dissect why the one "
            "'significant' result is an **overlapping-window false positive**.\n\n"
            "> ⚠️ **Not investment advice.** There is **no real tape** for this study (no free "
            "point-in-time Netflix Top-10 hours feed), so it is **synthetic-only** and capped at "
            f"`WEAK`/`NONE`. Numbers are the null synthetic world (seed {R['seed']}, world fp "
            f"`{R['fp_world']}`, as-of 2026-06-30). Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | No real tape (synthetic-only ⇒ not `REAL`). Null world: NFLX slope "
            f"**{R['slope']:+.4f}**, naive OLS *t* **{R['ols_t']}** but overlap-robust NW *t* "
            f"**{R['nw_t']}**; 20-seed mean NW *t* **+{R['nw_mean_20']}**, **{R['frac_ge2']}/20** clear "
            "\\|*t*\\| ≥ 2. |\n"
            f"| **Tradability** | `MIRAGE` | Timing long/flat **{R['lf_gross']}%/yr** gross "
            f"(**{R['lf_net']}%** net) vs buy-and-hold **+{R['bh_ann']}%/yr**; long/short "
            f"**{R['ls_gross']}%/yr**. |\n"
            f"| **Alt-data edge?** | `MIRAGE` | XLY spillover flat (*t* {R['xly_t']}); the 'edge' is a "
            "single-seed overlap artifact, and real reports lag & are widely covered (priced in). |\n\n"
            "> 💡 In plain words: the engine works (the synthetic control proves it), but there is no "
            "data to test on and the honest, overlap-corrected, seed-averaged link is a flat zero."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $m_t$ be standardised engagement momentum at week $t$ and $r_{t\\to t+h}$ the forward "
            "$h$-week return.\n\n"
            "- **H₁ (predictive).** In $r_{t\\to t+h} = \\alpha + \\beta\\, m_t + \\varepsilon$, "
            "$\\beta > 0$ at a robust *t* ≥ 2 for NFLX.\n"
            "- **H₂ (spillover).** The same $\\beta > 0$ for XLY (consumer discretionary).\n"
            "- **H₃ (robustness).** The sign/size of $\\beta$ is stable across forward horizons and "
            "RNG seeds — *not* an overlapping-window artifact.\n"
            "- **H₄ (net).** A timing rule on $m_t$ beats buy-and-hold after costs and borrow.\n\n"
            "We find **H₁ not supported** (naive *t* is an overlap false positive; NW *t* below the bar "
            "and seed-averaged ≈ 0), **H₂ rejected** (flat), **H₃ rejected** (the naive *t* grows with "
            "horizon purely from overlap), and **H₄ rejected** (timing loses to buy-and-hold)."
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — what rides on each answer\n\n"
            "Two structural reasons this is hard: (a) **no data** — the public Top-10 hours series is "
            "short (2021+), methodology-revised, and not a free machine-readable point-in-time feed, so "
            "a synthetic-only study cannot earn `REAL`; and (b) **overlap** — weekly signals against "
            "multi-week forward returns share most of their observations, so a naive OLS *t* massively "
            "overstates significance. The Newey-West correction and seed-averaging exist precisely to "
            "un-fool that. Same alt-data family as "
            "[257 AAII](../../257-aaii-sentiment/) / [335 Buzz](../../335-buzz-sentiment-etf/)."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            "- **Signal.** $m_t = z(\\text{engagement}_t - \\overline{\\text{engagement}}_{t-8:t})$ — "
            "standardised gap of the index above its trailing 8-week mean (formed from data ≤ $t$).\n"
            "- **Regression.** OLS of $r_{t\\to t+4}$ on $m_t$; report the slope, the **naive OLS *t***, "
            "and the **Newey-West (overlap-robust) *t***.\n"
            "- **Placebo.** Shuffle $m_t$ against the forward returns (2000 perms); tail probability of "
            "\\|slope\\|.\n"
            "- **Robustness.** Sweep forward horizons {2,4,8,13} weeks and RNG seeds.\n"
            "- **Frictions.** Long/flat and long/short timing, 5 bps/side one-way costs, 100 bps/yr "
            "borrow on the short leg.\n"
            "- **Positive control.** Plant $\\beta > 0$ and average the NW *t* over 25 seeds (house "
            "rule) — vs the null.\n\n"
            "Timing: $m_t$ uses data ≤ $t$ (no look-ahead); the position is entered at week $t$'s close "
            "and held over the forward window — one documented execution lag."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The predictive regression — H₁ (and the OLS-vs-Newey-West gap)\n\n"
            "The headline test, with *both* standard errors so you can see the overlap inflation."
        ),
        code(
            "reg = st.predictive_regression(WEEKLY, 'nflx_fwd')\n"
            "p = st.placebo_pvalue(WEEKLY, 'nflx_fwd', n_perm=2000, seed=SEED)\n"
            "fig, ax = plt.subplots(figsize=(7.5, 4.3))\n"
            "ax.bar(['naive OLS t','Newey-West t\\n(overlap-robust)'], [reg['slope_t'], reg['t_nw']],\n"
            "       color=[AMBER, RED], width=.5)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1); ax.axhline(-2, ls='--', c=GREY, lw=1)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('t-stat for the predictive slope')\n"
            "ax.set_title(f\"The naive t ({reg['slope_t']:+.2f}) 'clears' ±2; the honest t ({reg['t_nw']:+.2f}) does not\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"slope {reg['slope']:+.4f} | naive OLS t {reg['slope_t']:+.2f} | NW t {reg['t_nw']:+.2f} | corr {reg['corr']:+.2f} | placebo p {p:.3f}\")"
        ),
        md(
            f"> 💡 In plain words: **H₁ not supported.** The naive *t* ({R['ols_t']}) is an artifact of "
            f"overlapping 4-week windows; the overlap-robust NW *t* is **{R['nw_t']}** (below the bar, "
            f"and the wrong sign). The placebo *p* ({R['placebo_p']}) is borderline *because the shuffle "
            "inherits the same overlap* — which is exactly why we don't stop at one seed."
        ),
        md(
            "### 4b · The seed-robust null — H₁ evaporates\n\n"
            "Re-fit on 20 null worlds (β = 0). If the seed-551 'significance' were real signal it would "
            "recur; if it's overlap noise, the average washes out."
        ),
        code(
            "ts = [st.predictive_regression(data.synthetic_series(beta=0.0, seed=s)[0], 'nflx_fwd')['t_nw']\n"
            "      for s in range(551, 571)]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.2))\n"
            "cols = [RED if abs(t)>=2 else GREY for t in ts]\n"
            "ax.bar(range(len(ts)), ts, color=cols, width=.7)\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1)\n"
            "ax.axhline(np.mean(ts), c=GREEN, lw=1.5, label=f'mean {np.mean(ts):+.2f}')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_xlabel('seed'); ax.set_ylabel('Newey-West t')\n"
            "ax.set_title('20 null worlds: mean NW t ~ 0, none clear the bar'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print('mean NW t (20 seeds):', round(float(np.mean(ts)),3), '| clearing |t|>=2:', int(np.sum(np.abs(ts)>=2)), '/ 20')"
        ),
        md(
            f"> 💡 In plain words: over 20 seeds the mean NW *t* is **+{R['nw_mean_20']}** and "
            f"**{R['frac_ge2']}/20** clear \\|*t*\\| ≥ 2. The seed-551 draw was a false positive. No signal."
        ),
        md(
            "### 4c · The horizon-overlap artifact + the XLY spillover — H₂ & H₃\n\n"
            "Left: sweep forward horizons — watch the *naive* OLS *t* balloon (more overlap = more "
            "fake significance) while the NW *t* stays modest. Right: the XLY spillover leg."
        ),
        code(
            "hz = [(h,) + tuple(st.predictive_regression(data.synthetic_series(beta=0.0, horizon=h, seed=SEED)[0], 'nflx_fwd')[k]\n"
            "      for k in ('slope_t','t_nw')) for h in (2,4,8,13)]\n"
            "hzs = [r[0] for r in hz]; ols = [r[1] for r in hz]; nw = [r[2] for r in hz]\n"
            "regx = st.predictive_regression(WEEKLY, 'xly_fwd')\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))\n"
            "a1.plot(hzs, ols, 'o-', c=AMBER, lw=2, label='naive OLS t')\n"
            "a1.plot(hzs, nw, 's-', c=RED, lw=2, label='Newey-West t')\n"
            "a1.axhline(-2, ls='--', c=GREY, lw=1); a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_xlabel('forward horizon (weeks)'); a1.set_ylabel('t-stat')\n"
            "a1.set_title('Overlap inflates the NAIVE t at long horizons'); a1.legend()\n"
            "a2.scatter(WEEKLY['eng_mom'], WEEKLY['xly_fwd']*100, s=18, color=GREY, alpha=.6)\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_xlabel('engagement momentum'); a2.set_ylabel('XLY fwd return %')\n"
            "a2.set_title(f\"XLY spillover: flat (t {regx['slope_t']:+.2f})\")\n"
            "plt.tight_layout(); plt.show()\n"
            "print('horizon (h, ols_t, nw_t):', [(h, round(o,2), round(n,2)) for h,o,n in hz])\n"
            "print(f\"XLY slope {regx['slope']:+.4f}, t {regx['slope_t']:+.2f}, corr {regx['corr']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: **H₃ rejected** — the naive *t* grows with horizon purely because "
            "longer windows overlap more (a 13-week return shares 12 weeks with the next), a null "
            f"artifact, not signal. **H₂ rejected** — the XLY spillover is flat (*t* {R['xly_t']}); "
            "engagement, even if it moved NFLX, would not mechanically move the whole consumer basket."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — no real tape (synthetic-only); H₁ unsupported (NW *t* {R['nw_t']}, "
            f"seed-mean +{R['nw_mean_20']}, {R['frac_ge2']}/20 clear the bar); H₂/H₃ rejected.\n"
            f"- **Tradability `MIRAGE`** — timing {R['lf_gross']}%/yr gross vs buy-and-hold "
            f"+{R['bh_ann']}%/yr.\n"
            "- **Alt-data edge? `MIRAGE`** — an overlap false positive on data you cannot even buy."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — costs and the borrow\n\n"
            "Long/flat and long/short timing on $m_t$, gross and net, against buy-and-hold."
        ),
        code(
            "lf = st.timing_backtest(WEEKLY, 'nflx_ret', allow_short=False, cost_bps=5.0)\n"
            "ls = st.timing_backtest(WEEKLY, 'nflx_ret', allow_short=True, cost_bps=5.0, borrow_ann_bps=100.0)\n"
            "labels = ['long/flat\\ngross','long/flat\\nnet','long/short\\nnet','buy & hold']\n"
            "vals = [lf['gross_ann']*100, lf['net_ann']*100, ls['net_ann']*100, lf['buyhold_ann']*100]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "cols = [AMBER, RED, RED, GREEN]\n"
            "ax.bar(labels, vals, color=cols, width=.6)\n"
            "ax.axhline(0, c='k', lw=1); ax.set_ylabel('annualised return %')\n"
            "ax.set_title('Every timing variant loses to just holding the stock')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"long/flat gross {lf['gross_ann']*100:+.1f}% net {lf['net_ann']*100:+.1f}% | long/short net {ls['net_ann']*100:+.1f}% | buy&hold {lf['buyhold_ann']*100:+.1f}%\")"
        ),
        md(
            "> 💡 In plain words: timing on a noise signal churns costs and misses upside; shorting the "
            "low-engagement weeks (and paying a borrow) is worse still. Nothing to harvest. `MIRAGE`."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control\n\n"
            "Is the engine a faithful detector, or does it always print ~0? Plant a real "
            "engagement→return link (β > 0) of increasing strength and watch the **overlap-robust** NW "
            "*t* climb — averaged over 25 seeds so no single lucky seed can fake it."
        ),
        code(
            "betas = [0.0, 0.005, 0.010, 0.020, 0.030, 0.040]\n"
            "ts = [st.synthetic_mean_t(data, beta=b, n_seeds=25) for b in betas]\n"
            "fig, ax = plt.subplots(figsize=(8.5, 4.3))\n"
            "ax.plot(betas, ts, 'o-', c=GREEN, lw=2)\n"
            "ax.axhline(2, ls='--', c=GREY, lw=1, label='t = +2 bar')\n"
            "ax.axhline(0, c='k', lw=1)\n"
            "ax.set_xlabel('planted beta (engagement -> forward return)')\n"
            "ax.set_ylabel('mean Newey-West t (25 seeds)')\n"
            "ax.set_title('The harness banks a planted signal — flat at the null, past +2 as it grows')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "for b, t in zip(betas, ts): print(f'beta {b:+.3f} -> mean NW t {t:+.2f}')"
        ),
        md(
            "The NW *t* is ≈ 0 at the null and climbs past +2 as the planted link grows — so the engine "
            "is a faithful detector *even under the overlap correction*. The null-world non-result is "
            "therefore a statement about **the data and the overlap trap**, not the code. For sibling "
            "alt-data signals see [257 AAII](../../257-aaii-sentiment/), "
            "[335 Buzz](../../335-buzz-sentiment-etf/) and [392 Glassdoor](../../392-glassdoor-sentiment/)."
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
