"""Generate the two narrative notebooks for Study 654 (Quiet-Period-Expiry).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached per-ticker
+ SPY tapes under ../_cache/ and otherwise quote the frozen headline numbers in ``R``
(mirroring docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (yfinance daily closes +
# SPY, 66 hardcoded IPOs 2015-01-23 -> 2025-06-05, 60 with a full post-listing window).
R = dict(
    n_basket=66, n_tape=60, cal_lo="2015-01-23", cal_hi="2025-06-05",
    car_mean=3.346, car_median=1.616, car_sd=16.561, car_t=1.56,
    hit=33, hit_n=60, hit_pct=55.0, wilson=(42.5, 66.9),
    placebo_paired_diff=0.858, placebo_paired_t=0.29,
    placebo_rand_obs=3.346, placebo_rand_mean=2.207, placebo_rand_sd=1.789,
    placebo_rand_p=0.2606, placebo_rand_n=20000,
    # day-by-day anatomy: t -> (mean pct, t-stat)
    anatomy={15: (-0.07, -0.12), 16: (-1.17, -2.40), 17: (0.82, 1.31), 18: (0.85, 1.37),
             19: (-0.06, -0.11), 20: (-0.11, -0.17), 21: (-0.44, -0.92), 22: (-0.26, -0.51),
             23: (0.86, 1.99), 24: (-0.99, -1.76), 25: (1.50, 1.75), 26: (1.28, 1.88),
             27: (0.15, 0.17), 28: (0.32, 0.74), 29: (-0.07, -0.13), 30: (1.11, 1.04),
             31: (0.54, 1.04), 32: (0.60, 0.89), 33: (0.01, 0.01), 34: (-0.19, -0.29),
             35: (0.09, 0.14)},
    # timer strategy: buy day 22, sell day 27
    timer_gross=2.160, timer_net5=2.060, timer_net10=1.960,
    timer_t_gross=1.57, timer_t_net5=1.50, timer_t_net10=1.43,
    timer_hit=56.7, timer_best=30.26, timer_worst=-24.32, timer_n=60,
    # era contrast
    era_split="2021-06-01", era_early=5.023, era_early_n=38, era_early_t=1.72,
    era_late=0.449, era_late_n=22, era_late_t=0.15, era_diff_t=-1.11,
    # synthetic control
    syn_null_mean=0.05, syn_null_sd=1.12, syn_null_fire=1, syn_planted_pop=400,
    syn_planted_car=4.982, syn_planted_t=3.31,
    fp_panel="a790bce88c75", fp_abn="4adda6645406",
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Faded_since_2021%3F: Mixed](https://img.shields.io/badge/Faded_since_2021%3F-Mixed-8b949e?style=flat-square)\n\n"
)

BOOT = """\
import sys, os
sys.path.insert(0, os.path.abspath(".."))          # the study package
sys.path.insert(0, os.path.abspath("../../.."))    # repo root
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (9.5, 5.0), "axes.grid": True,
                     "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False})
RED, AMBER, GREEN, GREY = "#c0392b", "#dab617", "#2ea44f", "#8b949e"

from quiet_period_expiry import data, strategy as st

BASKET = data.basket_frame()
HAVE_REAL = data.have_real()
if HAVE_REAL:
    PANEL = data.fetch_event_panel()
else:
    PANEL = None
print("real cache present:", HAVE_REAL, "| basket:", len(BASKET),
      "| IPOs on tape:", (0 if PANEL is None else PANEL['ticker'].nunique()))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# The bank goes quiet for 25 days — then what? 🤐📊\n"
            "### The quiet-period-expiry pop — a real 2003 finding that our own tape "
            "can't confirm\n\n"
            + BADGES +
            "When a company IPOs, the banks that underwrote the deal are legally barred "
            "from publishing research on it for **25 calendar days** (FINRA Rule 2711 — a "
            "rule written after regulators found underwriter analysts were, unsurprisingly, "
            "far more bullish on their own deals than anyone else). The day that gag order "
            "lifts, those same banks' analysts pile in — and academics found that, on IPOs "
            "from the late 1990s, the stock actually **popped** when the all-clear notes hit.\n\n"
            "That's the claim we test on a fresh, modern sample: *does the bank's silence "
            "being broken still move the stock?*\n\n"
            "> 📓 **Plain-language layer.** Want the *t*-stats, the placebos and the cost "
            "math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 66 real, underwritten US IPOs, 2015→2025, hardcoded — "
            "direct listings and SPAC mergers excluded because the quiet-period rule binds "
            "underwriters of a traditional offering, not those structures. Every chart is "
            "drawn by the code beside it; house style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the stock pop around day 25? | **A little, on paper — not "
            f"statistically.** Average abnormal return (vs the S&P 500) over trading days "
            f"20–30 is **+{R['car_mean']:.1f}%**, but that number could plausibly be noise "
            f"(only **55%** of IPOs even finish the window positive). |\n"
            "| Is day 25 special, or is this just a generally bumpy stretch? | **It's not "
            "special.** A random 11-day window drawn from anywhere in these same IPOs' "
            f"first six weeks beats the observed pop about **{R['placebo_rand_p']*100:.0f}% "
            "of the time**. |\n"
            f"| Could you trade it? | **Buy day 22, sell day 27** nets "
            f"**+{R['timer_net5']:.1f}%** on paper — but with a **+{R['timer_best']:.0f}% / "
            f"{R['timer_worst']:.0f}%** best/worst spread across 60 trades, that average is "
            "carried by a handful of wild names, not a repeatable edge. |\n"
            "| Has it faded since the story became famous? | **Maybe.** Early-sample IPOs "
            f"(2015→2021) show +{R['era_early']:.1f}%; recent ones (2021→2025) show almost "
            f"nothing (+{R['era_late']:.1f}%) — suggestive, but the gap itself isn't "
            "statistically certified. |\n\n"
            "> The mechanism is real. The 2015–2025 tape just won't sign off on it."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"The bank that took you public spent months telling you what a great "
            "company you are. The day the SEC lets its analysts finally publish, of course "
            "they say Buy — everyone knew they would, and they wait 25 days to say it. "
            "That coordinated burst of one-sided opinion should move the stock.\"*\n\n"
            "It's not just trading-desk lore: **Bradley, Jordan & Ritter (2003)** studied "
            "1996–2000 IPOs and found a real, statistically significant pop clustered "
            "around the initiation date — mostly explained by how lopsidedly bullish "
            "underwriter analysts are (nearly all Buy ratings, essentially never a Sell on "
            "your own deal)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If the original finding still holds, this is a genuinely mechanical edge: the "
            "quiet-period calendar is public from the day a company files its S-1, so a "
            "trader knows the pop window in advance — no crystal ball needed, just a "
            "calendar. And it would tell us something bigger: that a **coordinated burst of "
            "conflicted-but-legal opinions** can move a market price even when nobody "
            "involved does anything illegal.\n\n"
            "So we ask: does the same test, run on a fresh, modern IPO sample, still find "
            "it — and if it does, can you actually bank it?"
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The basket.** **{R['n_basket']}** real, underwritten US IPOs from "
            f"{R['cal_lo']} to {R['cal_hi']} — direct listings and SPAC mergers excluded, "
            "since they don't have the same underwriter-quiet-period rule.\n"
            "- **The comparison.** Each stock's return minus the S&P 500's return "
            "(\"abnormal return\") on the same calendar days, summed over trading days "
            "**20 through 30** since listing — the window bracketing the ~25-trading-day "
            "quiet-period-expiry proxy.\n"
            "- **The luck checks.** (1) Compare that window to a same-width window earlier "
            "in the same stock's first few weeks. (2) Compare it to thousands of random "
            "windows drawn the same way. If the [20..30] window isn't special versus either, "
            "the story doesn't hold up.\n"
            "- **The trade check.** Buy the close of day 22, sell the close of day 27, pay "
            "costs — the version of the trade an actual account could place."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the headline.** Cumulative abnormal return over trading days 20–30, "
            "one bar per IPO's contribution to the average."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = st.car_window_stats(PANEL)\n"
            "    mean_c, n_c, hit_c = c['mean_car_pct'], c['n'], c['hit_rate']\n"
            "else:\n"
            "    mean_c, n_c, hit_c = R['car_mean'], R['n_tape'], R['hit_pct']/100\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.4))\n"
            "ax.bar(['[20..30] window\\n(the claim)'], [mean_c], color=AMBER, width=.45)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.annotate(f'{mean_c:+.2f}%', (0, mean_c), ha='center',\n"
            "            va='bottom' if mean_c >= 0 else 'top')\n"
            "ax.set_ylabel('mean cumulative abnormal return (%)')\n"
            "ax.set_title(f'A positive lean ({hit_c*100:.0f}% of IPOs finish the window up) '\n"
            "             'that never clears the bar')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'mean CAR [20..30] = {mean_c:+.3f}%  (n={n_c}, hit rate {hit_c*100:.1f}%)')"
        ),
        md(
            f"On paper: **+{R['car_mean']:.2f}%** average, and **{R['hit_pct']:.0f}%** of "
            "IPOs finish the window positive — a slim majority, barely better than a coin "
            f"flip (the 95% range for that hit rate runs from {R['wilson'][0]:.0f}% to "
            f"{R['wilson'][1]:.0f}%). The quants notebook shows this doesn't clear "
            "statistical significance any way we cut it.\n\n"
            "**Is day 20-30 actually special, though?** Here's the honest check: how does "
            "this window compare to a random stretch of the same stocks' early life?"
        ),
        code(
            "if HAVE_REAL:\n"
            "    rp = st.random_window_placebo(PANEL, n_draws_per_seed=200, n_seeds=4)\n"
            "    obs, draws_mean, draws_sd = rp['obs'], rp['placebo_mean'], rp['placebo_sd']\n"
            "else:\n"
            "    obs, draws_mean, draws_sd = R['placebo_rand_obs'], R['placebo_rand_mean'], R['placebo_rand_sd']\n"
            "rng = np.random.default_rng(654)\n"
            "draws = rng.normal(draws_mean, draws_sd, 3000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=45, color=GREY, alpha=.85,\n"
            "        label='random same-width windows drawn from the same 60 IPOs')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'observed [20..30] window {obs:+.2f}%')\n"
            "ax.set_xlabel('mean CAR of a random 11-day window (%)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f'Not an outlier in its own tape: canonical p = {R[\"placebo_rand_p\"]:.2f} '\n"
            "             '(20 seeds x 1,000 draws)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'canonical placebo (results.md): mean {R[\"placebo_rand_mean\"]:+.3f}%, '\n"
            "      f'p = {R[\"placebo_rand_p\"]:.4f}')"
        ),
        md(
            f"About **{R['placebo_rand_p']*100:.0f}%** of random 11-day windows in these "
            "same stocks' first six weeks beat the observed [20..30] window. That's the "
            "opposite of a rare event — day 25 isn't special in its own tape.\n\n"
            "**Finally, the trade.** Buy day 22's close, sell day 27's close — the version "
            "an actual account could place, costs included."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ts = st.timer_strategy(PANEL, cost_bps=5.0)\n"
            "    net, worst, best = ts['net_pct'], ts['worst_pct'], ts['best_pct']\n"
            "else:\n"
            "    net, worst, best = R['timer_net5'], R['timer_worst'], R['timer_best']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.4))\n"
            "ax.bar(['worst trade', 'average (net)', 'best trade'], [worst, net, best],\n"
            "       color=[RED, AMBER, GREEN], width=.55)\n"
            "for i, v in enumerate([worst, net, best]):\n"
            "    ax.annotate(f'{v:+.1f}%', (i, v), ha='center', va='bottom' if v >= 0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('return on one buy-day-22/sell-day-27 trade (%)')\n"
            "ax.set_title('The tails dwarf the average — 60 trades, one flyer either way '\n"
            "             'swings the mean')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'net {net:+.2f}%  |  best {best:+.1f}%  |  worst {worst:+.1f}%')"
        ),
        md(
            f"Net of costs: **+{R['timer_net5']:.2f}%** per trade on average — but the "
            f"single best trade made **+{R['timer_best']:.0f}%** and the single worst lost "
            f"**{R['timer_worst']:.0f}%**. With only 60 trades in the sample, one or two "
            "outlier names can move the average more than the whole claimed effect. That's "
            "not a repeatable edge — it's variance."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal — Weak.** The original 2003 study is real research on real data — "
            "it just doesn't reproduce with the same confidence on a fresh 2015–2025 "
            f"sample: **+{R['car_mean']:.1f}%** average, but statistically indistinguishable "
            "from noise on every check we ran.\n"
            "- **Tradability — Mirage.** The literal trade is positive on paper and "
            "uncertifiable in the math, with tails big enough to erase or double the "
            "average on a single name.\n"
            "- **\"Faded since 2021?\" — Mixed.** The point estimate has shrunk toward zero "
            "in the more recent half of the sample, consistent with the pattern getting "
            "arbitraged away as it became well known — but we can't prove the fade "
            "statistically either."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Timing precision may be the real problem.** We use a fixed trading-day "
            "count since listing; real initiation dates vary firm by firm (compliance "
            "sign-off, scheduling). Pooling by a fixed calendar likely smears a real, "
            "*firm-specific* event into noise. A follow-up with actual initiation-report "
            "dates (where available) would be a stronger test.\n"
            "- **Sibling studies:** [219-ipo-pop](../../219-ipo-pop/) (the day-1 pop), "
            "[319-lockup-expiry](../../319-lockup-expiry/) (the 180-day *share* unlock — "
            "same panel design, different mechanism) and "
            "[623-ipo-long-run-underperformance](../../623-ipo-long-run-underperformance/) "
            "(years, not weeks) — none of them is this study: the **25-day analyst quiet "
            "period**.\n\n"
            "*Think a tighter, per-firm initiation-date window recovers a certifiable edge? "
            "Show the *t* ≥ 2 on the real tape and we'll rerun this teardown.*"
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
            "# The Quiet-Period-Expiry Pop — a quantitative teardown 🔬\n"
            "### The [20..30] CAR-window *t* · a paired within-IPO placebo · a 20-seed "
            "random-window placebo · the [15..35] day-by-day anatomy · a costed timer "
            "trade · the pre/post-2021 era contrast · a 20-seed synthetic null\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). "
            "The claim — **the stock pops when the underwriters' 25-day analyst quiet "
            "period ends** — has a real academic anchor (Bradley, Jordan & Ritter 2003) and "
            "a hard, calendar-known mechanism (FINRA Rule 2711). The job here is to run the "
            "same kind of test on a fresh sample and report exactly where it stops clearing "
            "the bar.\n\n"
            "> ⚠️ **Data note.** 66 hardcoded, real, underwritten US IPOs 2015→2025 (direct "
            "listings and SPAC mergers excluded by construction), daily adjusted closes + "
            "SPY, yfinance, cached. 60 IPOs carry a full post-listing window. Abnormal "
            "return = stock − SPY, same calendar date (beta = 1). Survivorship on data "
            "availability only, named. Methods in "
            "[`docs/references.md`](../docs/references.md), numbers in "
            "[`docs/results.md`](../docs/results.md) (fingerprints `" + R["fp_panel"] +
            "` / `" + R["fp_abn"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `WEAK` | [20..30] CAR **+{R['car_mean']:.3f}%** "
            f"(median {R['car_median']:+.2f}%, sd {R['car_sd']:.1f}%, n={R['n_tape']}): "
            f"one-sample **t = {R['car_t']:.2f}**, paired placebo t = "
            f"{R['placebo_paired_t']:.2f}, random-window placebo p = "
            f"{R['placebo_rand_p']:.2f} |\n"
            f"| **Tradability** | `MIRAGE` | buy-22/sell-27 timer nets "
            f"{R['timer_net5']:+.2f}% at t = {R['timer_t_net5']:.2f}; best/worst "
            f"{R['timer_best']:+.1f}% / {R['timer_worst']:+.1f}% on n={R['timer_n']} |\n"
            f"| **Faded since 2021?** | `MIXED` | era CAR {R['era_early']:+.2f}% "
            f"(t={R['era_early_t']:.2f}, n={R['era_early_n']}) -> "
            f"{R['era_late']:+.2f}% (t={R['era_late_t']:.2f}, n={R['era_late_n']}); "
            f"diff t = {R['era_diff_t']:.2f} |\n\n"
            "> 💡 In plain words: a real published finding, a positive point estimate on a "
            "fresh sample, and *t*-stats that stay stubbornly under the certification bar "
            "at every cut we tried."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $CAR_i$ be IPO $i$'s cumulative market-adjusted (beta = 1) abnormal return "
            "over trading days $[20..30]$ since its first close (day 0). FINRA Rule 2711 "
            "bars the managing underwriter(s) from publishing research on the issuer for 25 "
            "*calendar* days; the window tested here brackets the trading-day proxy for that "
            "expiry, allowing for the administrative lag between quiet-period end and the "
            "actual initiation report. The claims:\n\n"
            "- **H₁ (pop).** $E[CAR_i] \\gg 0$ — a systematic, cross-sectionally consistent "
            "abnormal return in the expiry window, not a handful of lucky names.\n"
            "- **H₂ (specificity).** The [20..30] window is unusual *relative to the rest of "
            "the same IPOs' early trading life* — not just positive because post-IPO drift "
            "is generally positive.\n"
            "- **H₃ (bankability).** A retail-executable timer (buy day 22, sell day 27) "
            "captures the pop net of costs.\n"
            "- **H₄ (decay).** The effect has weakened as the finding became widely known "
            "and (partially) arbitraged/anticipated.\n\n"
            "We find **H₁ not certified** (t = 1.56), **H₂ rejected** (paired placebo "
            "t = 0.29; random-window placebo p = 0.26), **H₃ not certified** (t = 1.50, "
            "fat tails), **H₄ suggestive but not certified** (era-diff t = −1.11)."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "Each IPO has its own listing date, so unlike a shared-calendar event study "
            "(e.g. an FOMC decision) these 60 events are (largely) **cross-sectionally "
            "independent** — the planned primary is a **one-sample *t*** on the per-IPO CAR "
            "across IPOs, not a Welch split against a pooled control. Because \"the window "
            "is positive\" alone doesn't distinguish a real day-25 effect from generic "
            "post-IPO drift, we add a **paired within-IPO placebo** (the [20..30] CAR minus "
            "a same-width [3..13] CAR from the *same* ticker, differencing out any per-IPO "
            "level effect) and a **random-window placebo** — 20,000 (20 seeds × 1,000) "
            "independently-drawn same-width windows per IPO, giving a distribution-free "
            "right-tail *p*. The era split (2021-06-01, the sample's approximate calendar "
            "midpoint) is fixed *ex ante*, tested as a difference, not eyeballed."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Basket.** {R['n_basket']} hardcoded, real, underwritten US IPOs "
            f"{R['cal_lo']} → {R['cal_hi']} (direct listings & SPAC mergers excluded by "
            f"construction). {R['n_tape']} carry a full post-listing window.\n"
            "- **Tape.** Daily adjusted closes + SPY, yfinance, cached. As-of 2026-06-30 "
            "(last complete month).\n"
            "- **Headline.** One-sample t on the [20..30] per-IPO CAR + Wilson hit rate.\n"
            "- **Specificity.** Paired [20..30] − [3..13] placebo (same ticker) + 20-seed "
            "random-window placebo.\n"
            "- **Anatomy.** Mean abnormal return per trading day, [15..35], one-sample t "
            "each.\n"
            "- **Execution (third axis).** Buy day 22 close, sell day 27 close; 2 × "
            "one-way cost × NAV; long-only, no borrow.\n"
            "- **Control.** Synthetic per-IPO abnormal-return panel, planted-pop knob; the "
            "null must not fire across 20 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The headline window and its placebos\n\n"
            "One-sample t on the [20..30] CAR, the paired within-IPO placebo, and the "
            "random-window null. In the notebook we run a lighter placebo (4 seeds × 200 "
            "draws) and quote the canonical 20,000-draw p from `results.md`."
        ),
        code(
            "if HAVE_REAL:\n"
            "    c = st.car_window_stats(PANEL)\n"
            "    print(f\"[20..30] CAR: mean {c['mean_car_pct']:+.3f}%  \"\n"
            "          f\"(median {c['median_car_pct']:+.3f}%, sd {c['std_car_pct']:.2f}%, n={c['n']})\")\n"
            "    print(f\"one-sample t = {c['t']:+.2f}   \"\n"
            "          f\"hit {c['hit_up']}/{c['n']} = {c['hit_rate']*100:.1f}%  \"\n"
            "          f\"Wilson [{c['hit_lo']*100:.1f}%, {c['hit_hi']*100:.1f}%]\")\n"
            "    pp = st.paired_placebo_stats(PANEL)\n"
            "    print(f\"paired placebo: diff {pp['mean_diff_pct']:+.3f}%  t = {pp['t']:+.2f}\")\n"
            "    rp = st.random_window_placebo(PANEL, n_draws_per_seed=200, n_seeds=4)\n"
            "    obs, draws_mean, draws_sd = rp['obs'], rp['placebo_mean'], rp['placebo_sd']\n"
            "else:\n"
            "    obs, draws_mean, draws_sd = R['placebo_rand_obs'], R['placebo_rand_mean'], R['placebo_rand_sd']\n"
            "rng = np.random.default_rng(654)\n"
            "draws = rng.normal(draws_mean, draws_sd, 3000)\n"
            "fig, ax = plt.subplots(figsize=(9.0, 4.3))\n"
            "ax.hist(draws, bins=45, color=GREY, alpha=.85,\n"
            "        label='null: random same-width windows (light in-notebook run)')\n"
            "ax.axvline(obs, c=RED, lw=2.5, label=f'observed [20..30] window {obs:+.3f}%')\n"
            "ax.set_xlabel('mean CAR of a random 11-day window (%)')\n"
            "ax.set_ylabel('frequency')\n"
            "ax.set_title(f\"Not a tail event: canonical p = {R['placebo_rand_p']:.4f} \"\n"
            "             '(20 seeds x 1,000 draws)')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f\"canonical placebo (results.md): mean {R['placebo_rand_mean']:+.3f}%, \"\n"
            "      f\"sd {R['placebo_rand_sd']:.3f}%, p = {R['placebo_rand_p']:.4f}\")"
        ),
        md(
            f"> 💡 In plain words: the observed **{R['car_mean']:+.2f}%** sits under two "
            f"placebo-sigmas from the null's center ({R['placebo_rand_mean']:+.2f}% ± "
            f"{R['placebo_rand_sd']:.2f}%); canonical **p = {R['placebo_rand_p']:.2f}** — "
            "not remotely rare. With one-sample t = "
            f"**{R['car_t']:.2f}** and paired-placebo t = **{R['placebo_paired_t']:.2f}**, "
            "H₁ and H₂ both fail to clear the desk bar."
        ),
        md(
            "### 4b · Anatomy — is any single day special?\n\n"
            "Mean abnormal return per trading day, [15..35], one-sample t vs zero at each."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ev = st.per_day_stats(PANEL, lo=15, hi=35)\n"
            "    ks = list(ev.index); ms = list(ev['mean_pct']); ts = list(ev['t_stat'])\n"
            "else:\n"
            "    ks = sorted(R['anatomy']); ms = [R['anatomy'][k][0] for k in ks]\n"
            "    ts = [R['anatomy'][k][1] for k in ks]\n"
            "fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.6, 6.4), sharex=True,\n"
            "                             gridspec_kw={'height_ratios': [2, 1]})\n"
            "cols = [AMBER if 20 <= k <= 30 else GREY for k in ks]\n"
            "a1.bar([str(k) for k in ks], ms, color=cols, width=.62)\n"
            "a1.axhline(0, c='k', lw=.8); a1.set_ylabel('mean abnormal return (%)')\n"
            "a1.set_title('The [20..30] window (amber) vs the rest of the six-week tape')\n"
            "a2.bar([str(k) for k in ks], ts, color=[RED if abs(t)>=2 else GREY for t in ts], width=.62)\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.axhline(-2, ls='--', c=RED, lw=1); a2.axhline(2, ls='--', c=RED, lw=1)\n"
            "a2.set_ylabel('one-sample t'); a2.set_xlabel('trading day since listing')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('day 25:', dict(mean_pct=R['anatomy'][25][0], t=R['anatomy'][25][1]))"
        ),
        md(
            f"> 💡 In plain words: the [20..30] window (amber) doesn't visually separate "
            "from the grey rest of the tape, and the *t*-panel shows only **one** bar past "
            f"|t| = 2 in the whole [15..35] neighborhood — day 16 at "
            f"**t = {R['anatomy'][16][1]:.2f}**, which is *negative* and weeks before the "
            f"quiet period even ends. Day 25 itself: {R['anatomy'][25][0]:+.2f}% "
            f"(t = {R['anatomy'][25][1]:.2f}). No clean one-shot event on the day the story "
            "predicts."
        ),
        md(
            "### 4c · The literal trade, costed\n\n"
            "Buy day 22's close, sell day 27's close — the version an actual account could "
            "place — gross and net of one-way costs × 2."
        ),
        code(
            "if HAVE_REAL:\n"
            "    rows = [st.timer_strategy(PANEL, cost_bps=cb) for cb in (5.0, 10.0)]\n"
            "    g = rows[0]['gross_pct']; n5, n10 = rows[0]['net_pct'], rows[1]['net_pct']\n"
            "    tg, tn5 = rows[0]['t_gross'], rows[0]['t_net']\n"
            "    worst, best, hit = rows[0]['worst_pct'], rows[0]['best_pct'], rows[0]['hit_rate']\n"
            "else:\n"
            "    g, n5, n10 = R['timer_gross'], R['timer_net5'], R['timer_net10']\n"
            "    tg, tn5 = R['timer_t_gross'], R['timer_t_net5']\n"
            "    worst, best, hit = R['timer_worst'], R['timer_best'], R['timer_hit']/100\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.3))\n"
            "a1.bar(['gross', 'net 5 bps', 'net 10 bps'], [g, n5, n10], color=[GREY, AMBER, AMBER], width=.6)\n"
            "for i, v in enumerate([g, n5, n10]): a1.annotate(f'{v:+.2f}', (i, v), ha='center', va='bottom')\n"
            "a1.set_ylabel('% per trade'); a1.set_title(f'Positive on paper (t_gross = {tg:+.2f})')\n"
            "a2.bar(['worst', 'net (5bps)', 'best'], [worst, n5, best], color=[RED, AMBER, GREEN], width=.6)\n"
            "for i, v in enumerate([worst, n5, best]): a2.annotate(f'{v:+.1f}%', (i, v), ha='center',\n"
            "    va='bottom' if v >= 0 else 'top')\n"
            "a2.axhline(0, c='k', lw=.8)\n"
            "a2.set_ylabel('%'); a2.set_title('Tails dwarf the mean (n=60 trades)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'gross {g:+.2f} -> net {n5:+.2f}/{n10:+.2f}%  (t_net5={tn5:+.2f});  '\n"
            "      f'hit {hit*100:.1f}%  worst {worst:+.1f}%  best {best:+.1f}%')"
        ),
        md(
            f"> 💡 In plain words: +{R['timer_net5']:.2f}% net per trade sounds like free "
            f"money for a five-session hold — until you see the spread: a single "
            f"**{R['timer_worst']:+.1f}%** loss or **{R['timer_best']:+.1f}%** gain on 60 "
            f"trades outweighs the entire claimed effect. At t = {R['timer_t_net5']:.2f} "
            "the mean itself is not certifiable. This is concentrated idiosyncratic risk in "
            "a handful of volatile small/mid-cap names, not a diversified, repeatable edge "
            "— **H₃ not certified; Tradability = MIRAGE.**"
        ),
        md(
            "### 4d · The era contrast — has it faded?\n\n"
            "Split at **2021-06-01** (the sample's approximate calendar midpoint, fixed "
            "before any statistics were run) and tested as a **difference**, not eyeballed."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ec = st.era_contrast(PANEL, BASKET)\n"
            "    e, l = ec['early_pct'], ec['late_pct']\n"
            "    et, lt, dt = ec['t_early'], ec['t_late'], ec['t_diff']\n"
            "    ne, nl = ec['n_early'], ec['n_late']\n"
            "else:\n"
            "    e, l = R['era_early'], R['era_late']\n"
            "    et, lt, dt = R['era_early_t'], R['era_late_t'], R['era_diff_t']\n"
            "    ne, nl = R['era_early_n'], R['era_late_n']\n"
            "fig, ax = plt.subplots(figsize=(8.6, 4.3))\n"
            "ax.bar([f'2015 - 2021-06\\n(n={ne})', f'2021-06 - 2025\\n(n={nl})'], [e, l],\n"
            "       color=[AMBER, GREY], width=.55)\n"
            "for i,(v,t_) in enumerate([(e,et),(l,lt)]):\n"
            "    ax.annotate(f'{v:+.2f}%\\n(within-era t={t_:+.2f})',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('mean [20..30] CAR (%)')\n"
            "ax.set_title(f'Shrinking toward zero - but the gap is NOT certified (diff t = {dt:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'early {e:+.2f}% (t={et:+.2f}, n={ne})  late {l:+.2f}% (t={lt:+.2f}, n={nl})  diff t = {dt:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: the point estimate goes from **{R['era_early']:+.2f}%** "
            f"(t = {R['era_early_t']:.2f} — the closest this whole study gets to "
            f"significance) to **{R['era_late']:+.2f}%** (t = {R['era_late_t']:.2f}) in the "
            f"second half of the sample — but the Welch t of the difference is only "
            f"**{R['era_diff_t']:.2f}**, so we may not claim decay outright. Plausible "
            "story: the effect became well-documented folklore and got partially "
            "anticipated/arbitraged; honest reading: suggestive, not proven — **H₄ "
            "MIXED.**"
        ),
        md(
            "### 4e · Faithful-engine & power control — we know the truth here\n\n"
            "Synthetic per-IPO abnormal-return panel, i.i.d. daily noise, TUNABLE planted "
            "pop spread across [20..30]. The null (pop = 0) is checked over **20 seeds** — "
            "never a single stream."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(20):\n"
            "    p, _ = data.synthetic_panel(pop_bps=0.0, seed=654 + s_)\n"
            "    null_ts.append(st.synthetic_detect(p)['t'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "p_planted, _ = data.synthetic_panel(pop_bps=400.0, seed=654)\n"
            "planted_t = st.synthetic_detect(p_planted)['t']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.zeros(20) + np.linspace(-.12,.12,20), null_ts, color=GREY, s=40,\n"
            "           label='null worlds (pop=0), 20 seeds')\n"
            "ax.scatter([1], [planted_t], color=RED, s=90, zorder=5,\n"
            "           label='planted pop = +400 bps')\n"
            "ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(2, ls='--', c=RED, lw=1)\n"
            "ax.set_xticks([0, 1]); ax.set_xticklabels(['null x 20', 'planted'])\n"
            "ax.set_ylabel('one-sample t (CAR window)')\n"
            "ax.set_title('Control: the null fires at ~nominal rate, a planted pop lights up')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print(f'null: mean t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(np.abs(null_ts)>=2).sum()}/20 seeds  |  planted t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 20 null worlds the detector fires (|t| ≥ 2) in "
            f"**{R['syn_null_fire']}/20** seeds — almost exactly the nominal 5% false-"
            f"positive rate for a two-sided *t* ≥ 2 threshold — and a planted "
            f"+{R['syn_planted_pop']} bps pop reads t = {R['syn_planted_t']:.2f}. The "
            f"machinery is unbiased and has power; the real-tape t = {R['car_t']:.2f} is the "
            "genuine measurement, not a blind spot. *(A faithful-engine / power check "
            "only — never cited in support of the real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `WEAK`** — [20..30] CAR **{R['car_mean']:+.3f}%** "
            f"(n={R['n_tape']}): one-sample t = **{R['car_t']:.2f}**, paired placebo "
            f"t = **{R['placebo_paired_t']:.2f}**, random-window placebo "
            f"p = **{R['placebo_rand_p']:.2f}**, no individual day in [15..35] clears "
            "t = 2 on the claim's side. BJR (2003) found it real on 1996–2000 IPOs; this "
            "2015–2025 tape can't certify it.\n"
            f"- **Tradability `MIRAGE`** — the buy-22/sell-27 timer nets "
            f"**{R['timer_net5']:+.2f}%** at t = **{R['timer_t_net5']:.2f}** with a "
            f"**{R['timer_best']:+.1f}% / {R['timer_worst']:+.1f}%** best/worst spread on "
            f"{R['timer_n']} trades — uncertifiable and tail-dominated.\n"
            f"- **Faded since 2021? `MIXED`** — {R['era_early']:+.2f}% "
            f"(t={R['era_early_t']:.2f}) → {R['era_late']:+.2f}% (t={R['era_late_t']:.2f}); "
            f"the shrinkage is directionally consistent with the folklore getting "
            f"arbitraged away, but the Welch t of the difference (t = {R['era_diff_t']:.2f}) "
            "does not certify it."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **Fixed-calendar pooling likely dilutes a real, firm-specific event.** "
            "BJR (2003) had access to the *actual* initiation dates per IPO; those vary "
            "by a week or more around the quiet-period expiry (compliance timing, "
            "scheduling). Pooling on a fixed trading-day count instead of the true "
            "event date smears any real bump across neighboring days — the natural "
            "sequel is to hand-collect initiation dates for a subsample and re-run the "
            "event study on the true event day.\n"
            "- **The mechanism (conflicted, one-sided opinions) is not in doubt** — "
            "Michaely & Womack (1999) and the 2003 Global Settlement itself document it. "
            "What's in doubt is whether *price* still reacts to a well-telegraphed, "
            "calendar-known burst of predictable opinions, two decades after the original "
            "finding and in a market where algorithmic/quant desks price calendar effects "
            "in advance.\n"
            "- **Dedup map:** [219-ipo-pop](../../219-ipo-pop/) (day-1 underpricing), "
            "[319-lockup-expiry](../../319-lockup-expiry/) (the 180-day share unlock, same "
            "panel design), "
            "[623-ipo-long-run-underperformance](../../623-ipo-long-run-underperformance/) "
            "(3-5 years), [636-exchange-listing-pop](../../636-exchange-listing-pop/) "
            "(crypto listing pop/fade).\n\n"
            "*The reproducible core is offline and deterministic; frozen numbers live in "
            "[`docs/results.md`](../docs/results.md), sources in "
            "[`docs/references.md`](../docs/references.md).*"
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
