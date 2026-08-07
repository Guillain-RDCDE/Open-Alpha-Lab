"""Generate the two narrative notebooks for Study 836 (Rebalance Timing Luck).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks follow the desk's seven beats. This is a synthetic-only method demo: every figure
runs offline and deterministic on the seed-836 panel — there is NO real-tape cell (real free data
can never certify "zero momentum edge"), so the study is capped at NONE. The single-tape headline
is recomputed live (it is fast and deterministic, so it matches docs/results.md exactly); the
25-seed robustness numbers are quoted from the ``R`` dict below (mirror of docs/results.md).
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


# Frozen headline numbers — mirror of docs/results.md (as-of 2026-06-30; null panel fp abbc37b5f962;
# seed 836 single tape + 25-seed robustness). The single-tape numbers are recomputed live below.
R = dict(
    fp="abbc37b5f962", start="2012-01-02", end="2021-12-17", n_assets=30, n_days=2600,
    common_days=2453, period=21, lookback=126,
    best_off=14, best_sh=0.168, worst_off=1, worst_sh=-0.242,
    spread=0.410, sd=0.111, mean=-0.017, rank_corr=0.044,
    tranched_sharpe=-0.019, tranched_mean_bps=-0.079, tranched_t_nw=-0.06, tranched_t_1s=-0.06,
    timer_1_gross=-0.079, timer_1_cost=0.327, timer_1_net=-0.406, timer_1_sh=-0.097, timer_1_t=-0.30,
    timer_5_net=-1.168, timer_5_sh=-0.278, timer_5_t=-0.87,
    null25_spread=0.442, null25_corr=-0.088, null25_tsh=-0.047, null25_tnw=-0.147, null25_fires=0,
    edge25_spread=0.447, edge25_corr=0.003, edge25_tsh=1.387, edge25_tnw=4.244, edge25_fires=24,
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

from timing_luck import data, strategy as st

PERIOD, LOOKBACK, TOP_FRAC = 21, 126, 0.3
# The whole demo runs on the deterministic, offline seed-836 NULL panel (a cross-section where a
# momentum sort has ZERO genuine edge). Synthetic-only method demo — there is no real-tape cell.
RET, TRUTH = data.synthetic_panel(mom_edge=0.0, seed=836)
print("null panel:", RET.shape[1], "assets x", RET.shape[0], "days,",
      RET.index[0].date(), "->", RET.index[-1].date(),
      "| mom_edge %g (fp %s)" % (TRUTH.mom_edge, data.fingerprint(RET)))
"""


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Rebalance Timing Luck — the *same* strategy, a *different* Sharpe 🎰\n"
            "### How the arbitrary day you rebalance can make one book look like a winner and a loser at once\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Does_rebalance_timing_break_inference%3F: Confirmed](https://img.shields.io/badge/Does_rebalance_timing_break_inference%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "Every backtest of a *\"monthly\"* strategy makes one silent choice nobody thinks about: "
            "**which day of the month do you rebalance?** The 1st? The 15th? The last? It is supposed "
            "to be a trivial detail. It is not. The *same* rule, rebalanced on a different day, can "
            "trace a completely different equity curve — and print a completely different Sharpe "
            "ratio.\n\n"
            "So we take **one** momentum strategy, run it **21 times** — once for each possible "
            "rebalance day — on a return world we *built to have no real edge*, and watch what "
            "happens. Then we show the free fix that makes the illusion disappear.\n\n"
            "> 📓 **This is the plain-language layer.** Want the persistence rank-correlation, the "
            "Newey-West *t* and the maths? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Not investment advice.** A reproducible research tool: every chart is drawn by the "
            "code beside it, on a deterministic synthetic tape. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT),

        # ---- BEAT 0 — VERDICT ----
        md(
            "## The answer first 🎯\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the rebalance *day* change your Sharpe? | **Hugely.** The same book prints "
            f"**{R['worst_sh']:+.2f}** on its unluckiest day and **{R['best_sh']:+.2f}** on its "
            f"luckiest — a **{R['spread']:.2f}-Sharpe** swing on the *identical* strategy. |\n"
            f"| Is that gap real skill? | **No — pure luck.** The lucky day this year is a coin-flip "
            f"next year (rank correlation ≈ **{R['rank_corr']:+.2f}**). You cannot forecast it. |\n"
            f"| Can you fix it? | **Yes, for free.** *Tranch* the rebalance across every day at once "
            f"(overlapping portfolios) and the dispersion collapses to a single curve. |\n"
            "| Is there any real edge here? | **None.** The tape was built with zero momentum edge; "
            "the dispersion is noise dressed up as performance. |\n\n"
            "> A smooth, high-Sharpe track record can be an accident of *when* the fund rebalances — "
            "not proof of skill."
        ),

        # ---- BEAT 1 — THE CLAIM ----
        md(
            "## 1 · The claim\n\n"
            "> *\"The same rules-based strategy, rebalanced on a different day of the period, produces "
            "materially different returns and Sharpe ratios — a phantom dispersion that is pure luck, "
            "not skill.\"* — Hoffstein et al, *Rebalance Timing Luck*\n\n"
            "Two index funds can follow the *identical* methodology, differ only in the month they "
            "reconstitute, and drift apart by hundreds of basis points a year. Same idea, same rules "
            "— different luck of the calendar."
        ),

        # ---- BEAT 2 — SO WHAT ----
        md(
            "## 2 · So what?\n\n"
            "Because allocators hire and fire strategies on a Sharpe ratio. If an arbitrary rebalance "
            "date can swing that number from *fire* to *hire*, the number is nearly meaningless "
            "without knowing how the rebalance was scheduled — and a backtester can quietly try a few "
            "rebalance days and keep the prettiest. The desk has shown you can fake a Sharpe by "
            "**searching** for a lucky rule ([344 Backtest-Overfitting](../../344-backtest-overfitting/)) "
            "or by **reshaping** the returns ([590 Sharpe-Hacking](../../590-sharpe-hacking/)); here it "
            "gets faked by the calendar."
        ),

        # ---- BEAT 3 — HOW WE'D KNOW ----
        md(
            "## 3 · How would we even know?\n\n"
            "We need a world where we *know* there is no real edge, so any Sharpe dispersion must be "
            "luck. So we **build** one: 30 assets, ~10 years of daily returns, a shared market and "
            "pure noise — a cross-sectional **momentum** sort on this tape predicts *nothing*.\n\n"
            "Then we run **one** momentum book (buy the 6-month winners, short the losers, rebalance "
            "every 21 days) — but we run it once for **each rebalance offset**: starting on day 0, "
            "day 1, … day 20 of the cycle. Same rule, same data; only the rebalance day moves. If the "
            "21 Sharpes fan out, that fan is pure timing luck."
        ),

        # ---- BEAT 4 — THE TEARDOWN ----
        md(
            "## 4 · The teardown — let's actually run it\n\n"
            "**The fan of luck.** Here are the 21 Sharpe ratios of the identical strategy, one per "
            "rebalance day. Watch them spread out."
        ),
        code(
            "tl = st.timing_luck(RET, PERIOD, LOOKBACK, TOP_FRAC)\n"
            "sh = tl['sharpes']\n"
            "fig, ax = plt.subplots(figsize=(10, 4.6))\n"
            "colors = [GREEN if i == tl['best_offset'] else RED if i == tl['worst_offset'] else GREY for i in range(len(sh))]\n"
            "ax.bar(np.arange(len(sh)), sh, color=colors, width=.72)\n"
            "ax.axhline(tl['sharpe_mean'], ls='--', c='k', lw=1, label=f\"average {tl['sharpe_mean']:+.2f}\")\n"
            "ax.set_xlabel('rebalance offset (which day of the 21-day cycle)')\n"
            "ax.set_ylabel('annualised Sharpe'); ax.set_xticks(range(0, 21, 2))\n"
            "ax.set_title('ONE momentum strategy, 21 rebalance days — a %.2f-Sharpe fan of pure luck' % tl['sharpe_spread'])\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('luckiest offset #%d: Sharpe %+.3f' % (tl['best_offset'], tl['sharpe_best']))\n"
            "print('unluckiest offset #%d: Sharpe %+.3f' % (tl['worst_offset'], tl['sharpe_worst']))\n"
            "print('PHANTOM GAP: %.3f Sharpe units (same rule, same data)' % tl['sharpe_spread'])"
        ),
        md(
            f"There it is. The **green** bar (offset {R['best_off']}) is a book you'd *keep*; the "
            f"**red** bar (offset {R['worst_off']}) is one you'd *fire* — and they are the **exact "
            f"same strategy** on the **exact same data**. The only difference is the day of the month "
            f"the rebalance fell on. That's a **{R['spread']:.2f}-Sharpe** gap of nothing but luck."
        ),
        md(
            "**Is the lucky day *forecastable*?** If it were, the offset that won in the first half of "
            "history would tend to win in the second half too. Let's check the ranking's persistence."
        ),
        code(
            "pr = st.offset_persistence(RET, PERIOD, LOOKBACK, TOP_FRAC)\n"
            "fig, ax = plt.subplots(figsize=(6.2, 5.4))\n"
            "ax.scatter(pr['sharpe_h1'], pr['sharpe_h2'], c=GREY, s=45, zorder=3)\n"
            "for i,(a,b) in enumerate(zip(pr['sharpe_h1'], pr['sharpe_h2'])):\n"
            "    ax.annotate(str(i), (a,b), fontsize=7, ha='center', va='center')\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('offset Sharpe — first half'); ax.set_ylabel('offset Sharpe — second half')\n"
            "ax.set_title('The lucky offset does NOT persist (rank corr %+.2f)' % pr['rank_corr'])\n"
            "plt.tight_layout(); plt.show()\n"
            "print('Spearman rank correlation first-half vs second-half: %+.3f  (~0 => pure luck)' % pr['rank_corr'])"
        ),
        md(
            f"The cloud is shapeless — the offsets that were lucky early are a coin-flip later (rank "
            f"correlation **{R['rank_corr']:+.2f}**). You cannot pick the winning rebalance day in "
            "advance, because there is nothing there to pick. **Luck, not skill.**"
        ),
        md(
            "**The fix — tranching.** Instead of betting the whole book on one arbitrary day, run all "
            "21 offsets at once with 1/21 of the capital each (rebalance a slice every day). Their "
            "average is a *single* overlapping portfolio — there is nothing left to be lucky about."
        ),
        code(
            "tr = st.tranched_portfolio(RET, PERIOD, LOOKBACK, TOP_FRAC)\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "start = tr['start']; idx = RET.index[start:]\n"
            "for off in range(PERIOD):\n"
            "    port, _ = st.offset_portfolio(RET.to_numpy(), st.trailing_return(RET.to_numpy(), LOOKBACK), off, PERIOD, TOP_FRAC, 6)\n"
            "    ax.plot(idx, np.cumsum(port[start:]), c=GREY, lw=.7, alpha=.5)\n"
            "ax.plot(idx, np.cumsum(tr['seg']), c=GREEN, lw=2.4, label='tranched / overlapping (the fix)')\n"
            "ax.plot([], [], c=GREY, lw=.7, label='the 21 individual rebalance-day books')\n"
            "ax.set_ylabel('cumulative return (sum of daily)'); ax.set_title('21 lucky/unlucky curves -> ONE tranched curve')\n"
            "ax.legend(); plt.tight_layout(); plt.show()\n"
            "print('tranched Sharpe %+.3f (NW t = %+.2f) — dispersion collapsed to a single curve' % (tr['sharpe'], tr['t_nw']))"
        ),
        md(
            f"The grey spaghetti is the 21 timing-luck curves; the **green** line is the single "
            f"tranched book. The phantom dispersion is **gone** — and on this null tape the tranched "
            f"Sharpe is **{R['tranched_sharpe']:+.2f}** (NW *t* = **{R['tranched_t_nw']:+.2f}**), "
            "confirming what we knew: there was never any real edge, just noise the rebalance-day "
            "choice was amplifying."
        ),

        # ---- BEAT 5 — VERDICT ----
        md(
            "## 5 · The verdict\n\n"
            "- **Signal — None.** On a tape with no real edge, the 21 rebalance days fan out by "
            f"{R['spread']:.2f} Sharpe units of pure luck; the lucky day doesn't persist; the "
            "dispersion-free tranched book finds nothing. Nothing real to detect.\n"
            "- **Tradability — Mirage.** You can't harvest a spread whose winner is a coin-flip, and "
            "the tranched null book loses net of costs. Nothing to spend.\n"
            "- **Does rebalance timing break inference? — Confirmed.** Yes — an arbitrary rebalance "
            "date swings the reported Sharpe by ~0.4 units on the identical strategy, and tranching "
            "collapses it."
        ),

        # ---- BEAT 6 — COULD YOU TRADE IT ----
        md(
            "## 6 · Could you actually 'trade' it?\n\n"
            "No. To profit from the dispersion you'd have to *know* which rebalance day will be lucky "
            "— and it's unforecastable. Picking the best day *in hindsight* is curve-fitting; it "
            "won't repeat. The only sane response is the tranched book, which trades every day (so it "
            "pays costs) and, on a no-edge tape, earns nothing. There's no free Sharpe to harvest — "
            "only a phantom to eliminate."
        ),

        # ---- BEAT 7 — GOING FURTHER ----
        md(
            "## 7 · Going further 🚪\n\n"
            "- **The other ways to fake a Sharpe.** [344 Backtest-Overfitting](../../344-backtest-overfitting/) "
            "fakes it by *searching* many rules; [590 Sharpe-Hacking](../../590-sharpe-hacking/) fakes "
            "it by *reshaping* the reported returns. This study fakes it with the *calendar*.\n"
            "- **The fix is public and free.** Overlapping / tranched portfolios (Blitz–van der "
            "Grient–van Vliet 2010) — see the quants notebook for the persistence test and the "
            "planted-edge control.\n\n"
            "*Think your monthly backtest's Sharpe is robust? Fork this, sweep the rebalance offset on "
            "your own strategy, and see how wide the fan is before you trust the headline number.*"
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
            "# Rebalance Timing Luck — a quantitative teardown 🔬\n"
            "### per-offset Sharpe fan · out-of-sample offset-persistence rank correlation · tranched Newey-West *t* · cost math · 25-seed synthetic positive control\n\n"
            "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
            "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
            "![Does_rebalance_timing_break_inference%3F: Confirmed](https://img.shields.io/badge/Does_rebalance_timing_break_inference%3F-Confirmed-8b949e?style=flat-square)\n\n"
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb) — *same "
            "seven beats, every claim now carrying its standard error.* We run one cross-sectional "
            "momentum book across all 21 rebalance offsets on a zero-edge synthetic panel, quantify "
            "the phantom Sharpe dispersion, test its persistence, and collapse it with the tranched "
            "portfolio.\n\n"
            "> ⚠️ **Not investment advice.** A synthetic-only method demo: the tape is built to have "
            f"zero momentum edge (null fp `{R['fp']}`), so real free data can never certify it and the "
            "study is capped at `NONE`. Methods in [`docs/references.md`](../docs/references.md), "
            "numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back into intuition."
        ),
        code(BOOT),
        code("R = %r" % (R,)),

        # ---- BEAT 0 ----
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** | `NONE` | Same book, 21 offsets: Sharpe {R['worst_sh']:+.2f} … "
            f"{R['best_sh']:+.2f} (gap **{R['spread']:.2f}**); offset ranking rank-corr "
            f"**{R['rank_corr']:+.2f}** (unforecastable); tranched NW *t* **{R['tranched_t_nw']:+.2f}**. "
            "Synthetic-only — no real tape. |\n"
            f"| **Tradability** | `MIRAGE` | Lucky offset a coin-flip across seeds; tranched null book "
            f"nets {R['timer_1_net']:+.2f} bps/day at 1 bp (gross {R['timer_1_gross']:+.2f}). |\n"
            f"| **Does rebalance timing break inference?** | `CONFIRMED` | 25-seed mean phantom spread "
            f"**{R['null25_spread']:.2f}** Sharpe units; tranching -> single curve (dispersion 0); a "
            f"planted premium still fires (tranched NW *t* {R['edge25_tnw']:+.2f}, {R['edge25_fires']}/25). |\n\n"
            "> 💡 In plain words: the rebalance-day choice injects a ~0.4-Sharpe artefact with nothing "
            "underneath; the persistence test proves it's luck, and tranching removes it while a real "
            "edge survives."
        ),

        # ---- BEAT 1 ----
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let a strategy rebalance every $P$ days. The *offset* $o \\in \\{0, \\dots, P-1\\}$ is the "
            "phase of the rebalance schedule — day-of-cycle it reconstitutes. Write $SR_o$ for the "
            "realized annualised Sharpe of the identical rule run at offset $o$.\n\n"
            "- **H₁ (dispersion).** $\\{SR_o\\}$ has **material spread** — $\\max_o SR_o - \\min_o "
            "SR_o \\gg 0$ — even though the rule and data are fixed.\n"
            "- **H₂ (luck, not skill).** The offset ranking does **not** persist: the Spearman rank "
            "correlation of $\\{SR_o\\}$ between the first and second half of the sample is $\\approx "
            "0$ — the best offset is unforecastable.\n"
            "- **H₃ (the fix).** The tranched/overlapping portfolio $\\bar r_t = \\frac1P \\sum_o "
            "r^{(o)}_t$ is a **single** curve — dispersion $= 0$ by construction — and preserves any "
            "genuine premium.\n\n"
            "We **confirm H₁** (a %.2f-Sharpe gap), **confirm H₂** (rank-corr ≈ %+.2f), and **confirm "
            "H₃** (one tranched curve; silent on the null, positive on a planted edge)." % (R['spread'], R['rank_corr'])
        ),

        # ---- BEAT 2 ----
        md(
            "## 2 · So what? — the mechanism\n\n"
            "At each rebalance the book locks in the cross-section's *current* winners/losers and "
            "holds them for $P$ days. Two offsets rebalance on **different** days, so they lock in "
            "**different** snapshots of a noisy signal and ride **different** subsequent paths. Over a "
            "finite sample those path differences don't wash out — they accumulate into a Sharpe "
            "spread. It is the same family of construction-choice fragility as regime-dependence "
            "([349](../../349-regime-dependence/)), but the *only* moving part is the rebalance phase. "
            "The cure — overlapping portfolios — is Blitz–van der Grient–van Vliet (2010)."
        ),

        # ---- BEAT 3 ----
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Tape.** {R['n_assets']} assets, {R['n_days']} daily rows, common market factor "
            "(beta = 1 for all names, so a dollar-neutral book cancels the market **exactly**) + "
            "idiosyncratic noise. Null: `mom_edge = 0` (a momentum sort predicts nothing).\n"
            f"- **Rule.** Trailing-{R['lookback']}d momentum, long top 30% / short bottom 30%, "
            f"equal-weight, dollar-neutral, rebalanced every {R['period']} days; signal known at the "
            "close of `d-1`, held from `d` (one documented lag, no look-ahead).\n"
            "- **The sweep.** Run the identical rule at every offset `0..20`; evaluate on the common "
            "window once all offsets are live.\n"
            "- **Persistence.** Rank offsets by Sharpe in each half; Spearman rank correlation.\n"
            "- **The fix.** Tranch/overlap all offsets into one book; Newey-West (HAC, 10-lag) *t*.\n"
            "- **Costs.** One-way × NAV on the slice rotated each day + 50 bps/yr borrow on the short "
            "leg.\n"
            "- **Positive control.** The same machinery on tapes with a *planted* momentum premium "
            "(`mom_edge > 0`), averaged over 25 seeds (house rule)."
        ),

        # ---- BEAT 4 ----
        md("## 4 · The teardown"),
        md(
            "### 4a · The per-offset Sharpe fan — H₁\n\n"
            "The identical book at every rebalance offset. The spread is the phantom dispersion."
        ),
        code(
            "tl = st.timing_luck(RET, PERIOD, LOOKBACK, TOP_FRAC)\n"
            "sh = tl['sharpes']\n"
            "fig, ax = plt.subplots(figsize=(10, 4.4))\n"
            "colors = [GREEN if i==tl['best_offset'] else RED if i==tl['worst_offset'] else GREY for i in range(len(sh))]\n"
            "ax.bar(np.arange(len(sh)), sh, color=colors, width=.72)\n"
            "ax.axhline(tl['sharpe_mean'], ls='--', c='k', lw=1)\n"
            "ax.set_xlabel('rebalance offset'); ax.set_ylabel('annualised Sharpe'); ax.set_xticks(range(0,21,2))\n"
            "ax.set_title('Same strategy, 21 rebalance days: Sharpe fan = %.2f' % tl['sharpe_spread'])\n"
            "plt.tight_layout(); plt.show()\n"
            "print('best #%d %+.3f | worst #%d %+.3f | spread %.3f | sd %.3f | mean %+.3f | n=%d'\n"
            "      % (tl['best_offset'], tl['sharpe_best'], tl['worst_offset'], tl['sharpe_worst'],\n"
            "         tl['sharpe_spread'], tl['sharpe_sd'], tl['sharpe_mean'], tl['n_days']))"
        ),
        md(
            f"> 💡 In plain words: **H₁ confirmed.** The identical rule spans "
            f"{R['worst_sh']:+.2f} … {R['best_sh']:+.2f} — a **{R['spread']:.2f}-Sharpe** gap "
            f"(sd {R['sd']:.2f}) driven purely by the rebalance day. The offsets average "
            f"{R['mean']:+.2f}: there is no edge, only dispersion."
        ),
        md(
            "### 4b · Luck, not skill — the offset ranking does not persist (H₂)\n\n"
            "Rank offsets by Sharpe in the first half vs the second half. A 45° line would mean the "
            "lucky offset is forecastable; a shapeless cloud means it's a coin-flip."
        ),
        code(
            "pr = st.offset_persistence(RET, PERIOD, LOOKBACK, TOP_FRAC)\n"
            "fig, ax = plt.subplots(figsize=(6.2, 5.4))\n"
            "ax.scatter(pr['sharpe_h1'], pr['sharpe_h2'], c=GREY, s=45, zorder=3)\n"
            "for i,(a,b) in enumerate(zip(pr['sharpe_h1'], pr['sharpe_h2'])):\n"
            "    ax.annotate(str(i), (a,b), fontsize=7, ha='center', va='center')\n"
            "ax.axhline(0, c='k', lw=.8); ax.axvline(0, c='k', lw=.8)\n"
            "ax.set_xlabel('offset Sharpe — first half'); ax.set_ylabel('offset Sharpe — second half')\n"
            "ax.set_title('Offset ranking does not persist (Spearman %+.2f)' % pr['rank_corr'])\n"
            "plt.tight_layout(); plt.show()\n"
            "print('rank corr first-half vs second-half: %+.3f' % pr['rank_corr'])"
        ),
        md(
            f"> 💡 In plain words: **H₂ confirmed.** Rank correlation **{R['rank_corr']:+.2f}** ≈ 0 — "
            "the offset that won early is unrelated to the one that wins late. The dispersion carries "
            "**zero forecastable information**: it is luck, definitionally."
        ),
        md(
            "### 4c · The fix — tranching collapses the dispersion (H₃)\n\n"
            "Average all 21 offset books into one overlapping portfolio and read its Newey-West *t*."
        ),
        code(
            "tr = st.tranched_portfolio(RET, PERIOD, LOOKBACK, TOP_FRAC)\n"
            "start = tr['start']; idx = RET.index[start:]\n"
            "mom = st.trailing_return(RET.to_numpy(), LOOKBACK)\n"
            "fig, ax = plt.subplots(figsize=(9.5, 4.5))\n"
            "for off in range(PERIOD):\n"
            "    port, _ = st.offset_portfolio(RET.to_numpy(), mom, off, PERIOD, TOP_FRAC, 6)\n"
            "    ax.plot(idx, np.cumsum(port[start:]), c=GREY, lw=.7, alpha=.5)\n"
            "ax.plot(idx, np.cumsum(tr['seg']), c=GREEN, lw=2.4, label='tranched / overlapping')\n"
            "ax.plot([], [], c=GREY, lw=.7, label='21 individual offset books')\n"
            "ax.set_ylabel('cumulative daily return'); ax.legend()\n"
            "ax.set_title('Dispersion collapsed: 21 curves -> 1 (tranched Sharpe %+.2f, NW t %+.2f)' % (tr['sharpe'], tr['t_nw']))\n"
            "plt.tight_layout(); plt.show()\n"
            "print('tranched: Sharpe %+.3f | mean %+.3f bps/day | NW(10) t %+.2f | one-sample t %+.2f | n=%d'\n"
            "      % (tr['sharpe'], tr['mean_bps'], tr['t_nw'], tr['t_1s'], tr['n_days']))"
        ),
        md(
            f"> 💡 In plain words: **H₃ confirmed.** One curve, no dispersion, tranched NW *t* = "
            f"**{R['tranched_t_nw']:+.2f}** — the phantom is gone and, correctly, no edge remains on "
            "the null tape."
        ),
        md(
            "### 4d · The cost of the tranched book\n\n"
            "The tranched book trades a slice every day. Charge it and read the net."
        ),
        code(
            "for cb in (1.0, 5.0):\n"
            "    tm = st.timer_stats(RET, PERIOD, LOOKBACK, TOP_FRAC, cost_bps=cb, borrow_bps_yr=50.0)\n"
            "    print('cost=%.1f bps/side: gross %+.3f -> net %+.3f bps/day (cost %.3f/day, Sharpe net %+.3f, t %+.2f)'\n"
            "          % (cb, tm['gross_bps'], tm['net_bps'], tm['cost_bps_per_day'], tm['sharpe_net'], tm['t_net']))"
        ),
        md(
            f"> 💡 In plain words: flat gross ({R['timer_1_gross']:+.2f} bps/day), negative net "
            f"({R['timer_1_net']:+.2f} at 1 bp, {R['timer_5_net']:+.2f} at 5 bps). Nothing to harvest — "
            "`MIRAGE`."
        ),

        # ---- BEAT 5 ----
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal `NONE`** — a {R['spread']:.2f}-Sharpe phantom fan with rank-corr "
            f"{R['rank_corr']:+.2f} and a dispersion-free tranched NW *t* of {R['tranched_t_nw']:+.2f}; "
            "synthetic-only, so never `REAL`.\n"
            f"- **Tradability `MIRAGE`** — unforecastable winner; tranched null book nets "
            f"{R['timer_1_net']:+.2f} bps/day at 1 bp.\n"
            f"- **Does rebalance timing break inference? `CONFIRMED`** — ~{R['null25_spread']:.2f}-Sharpe "
            "artefact (25-seed mean), collapsed by tranching."
        ),

        # ---- BEAT 6 ----
        md(
            "## 6 · Could you trade it? — the honest read\n\n"
            "You cannot trade an unforecastable dispersion; picking the best offset in hindsight is "
            "curve-fitting and won't repeat. The tranched book is the only defensible construction, "
            "and on a no-edge tape it earns nothing net of costs. The point of the study is *not* a "
            "strategy — it's a warning that a backtest's Sharpe is contaminated by an arbitrary "
            "calendar choice unless you tranch it away."
        ),

        # ---- BEAT 7 ----
        md(
            "## 7 · Going further — the synthetic positive control (25 seeds)\n\n"
            "Is the tranched book just numb to everything, or does it reward a **real** edge? Plant a "
            "genuine momentum premium (`mom_edge > 0`) and check: the tranched book must light up, "
            "while the timing-luck dispersion stays present (it's an artefact of *when*, not *what*). "
            "Averaged over seeds so no lucky seed can fake it — live below on a few seeds, with the "
            "frozen 25-seed numbers from `docs/results.md`."
        ),
        code(
            "lo = st.seed_robust(data, mom_edge=0.0, n_seeds=6)\n"
            "hi = st.seed_robust(data, mom_edge=1.0, n_seeds=6)\n"
            "print('live (6 seeds):')\n"
            "print('  NULL   : phantom spread %.3f | rank corr %+.3f | tranched Sharpe %+.3f (NW t %+.3f) | fires %d/6'\n"
            "      % (lo['mean_sharpe_spread'], lo['mean_rank_corr'], lo['mean_tranched_sharpe'], lo['mean_tranched_t_nw'], lo['tranched_t_fires']))\n"
            "print('  PLANTED: phantom spread %.3f | rank corr %+.3f | tranched Sharpe %+.3f (NW t %+.3f) | fires %d/6'\n"
            "      % (hi['mean_sharpe_spread'], hi['mean_rank_corr'], hi['mean_tranched_sharpe'], hi['mean_tranched_t_nw'], hi['tranched_t_fires']))\n"
            "print()\n"
            "print('frozen (25 seeds, docs/results.md):')\n"
            "print('  NULL   : spread %.3f | rank corr %+.3f | tranched Sharpe %+.3f (NW t %+.3f) | fires %d/25'\n"
            "      % (R['null25_spread'], R['null25_corr'], R['null25_tsh'], R['null25_tnw'], R['null25_fires']))\n"
            "print('  PLANTED: spread %.3f | rank corr %+.3f | tranched Sharpe %+.3f (NW t %+.3f) | fires %d/25'\n"
            "      % (R['edge25_spread'], R['edge25_corr'], R['edge25_tsh'], R['edge25_tnw'], R['edge25_fires']))\n"
            "fig, ax = plt.subplots(figsize=(7.6, 4.3))\n"
            "ax.bar(['null\\n(mom_edge 0)','planted\\n(mom_edge 1)'], [R['null25_tsh'], R['edge25_tsh']],\n"
            "       color=[RED, GREEN], width=.5)\n"
            "ax.axhline(0, c='k', lw=.8)\n"
            "ax.set_ylabel('tranched annualised Sharpe (25-seed mean)')\n"
            "ax.set_title('Tranched book: silent on the null, fires on a planted premium')\n"
            "plt.tight_layout(); plt.show()"
        ),
        md(
            f"The tranched book is **silent on the null** (25-seed Sharpe {R['null25_tsh']:+.2f}, NW "
            f"*t* {R['null25_tnw']:+.2f}, fires {R['null25_fires']}/25) and **robustly positive when a "
            f"real premium is planted** ({R['edge25_tsh']:+.2f}, NW *t* {R['edge25_tnw']:+.2f}, fires "
            f"{R['edge25_fires']}/25) — the machinery detects genuine edge and is not itself the "
            f"artefact. Crucially the phantom dispersion is ~{R['null25_spread']:.2f} Sharpe units in "
            "**both** worlds: it is a property of *when* you rebalance, independent of whether there is "
            "anything to trade. For the other ways construction choices fake a Sharpe, see "
            "[344 Backtest-Overfitting](../../344-backtest-overfitting/) and "
            "[590 Sharpe-Hacking](../../590-sharpe-hacking/)."
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
