"""Generate the two narrative notebooks for Study 798 (Deferred-Revenue Signal).

    python notebooks/build_notebooks.py
    jupyter nbconvert --to notebook --execute --inplace \
        notebooks/01_for_the_curious.ipynb notebooks/02_for_the_quants.ipynb

Both notebooks run offline and deterministic. Real-tape cells read the cached prices + EDGAR
events under ../_cache/ and otherwise quote the frozen headline numbers in ``R`` (mirroring
docs/results.md). The synthetic positive control runs anywhere with no network.
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


# Frozen real-tape headline numbers — mirror of docs/results.md (EDGAR deferred revenue +
# yfinance prices, 40 SaaS-type names, period ends 2009-11 -> 2026-04, as-of 2026-06-30).
R = dict(
    as_of="2026-06-30", n_names=40, n_events=1238,
    end_lo="2009-11-27", end_hi="2026-04-30", fp_prices="f8a44ad79534",
    # primary calendar long-short (defrev_yoy, terciles, staleness 200)
    n_months=195, avg_n=19.6, ls_span_lo="2010-04-30", ls_span_hi="2026-06-30",
    ls_mean_bps=56.2, ls_ann=6.74, ls_t_iid=1.04, ls_t_nw=0.94, ls_sharpe=0.26,
    ls_hit=55, ls_long_bps=230.3, ls_short_bps=174.1, ls_turn=0.10, ls_cum=1.72,
    ls120_mean_bps=50.8, ls120_t_nw=0.84,
    scaled_mean_bps=75.9, scaled_t_nw=1.63, scaled_sharpe=0.43,
    xsec_early=8.8, xsec_late=27.2,
    # pooled event drift  horizon -> (n, top%, bot%, ls%, t, win%, placebo p)
    drift={
        21: (1232, 1.25, 1.54, -0.29, -0.42, 48, 0.6676),
        63: (1213, 5.70, 5.37, 0.33, 0.20, 49, 0.4062),
        126: (1185, 10.15, 9.62, 0.53, 0.22, 47, 0.4096),
    },
    mono63=(5.37, 5.12, 5.70), mono126=(9.62, 9.36, 10.15),
    # era split (calendar LS)
    era_early_n=105, era_early_bps=53.3, era_early_t=0.86,
    era_late_n=90, era_late_bps=59.6, era_late_t=0.55,
    # leads sales
    ls_n=917, ls_slope=0.434, ls_reg_t=27.05, ls_r2=0.444, ls_corr=0.667,
    ls_top_sales=40.4, ls_bot_sales=8.4, ls_spread=32.0, contemp_corr=0.678,
    # tradability net   (cost, borrow) -> (net bps, net ann, nw t, sharpe)
    net={(10, 50): (49.9, 5.99, 0.83, 0.23), (20, 100): (43.7, 5.24, 0.73, 0.20)},
    # synthetic control
    syn_null_mean=-0.35, syn_null_sd=1.08, syn_null_fire=1, syn_null_seeds=12,
    syn_planted_bps=236.8, syn_planted_t=6.56,
)

BADGES = (
    "![Signal: Weak](https://img.shields.io/badge/Signal-Weak-dab617?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Leads sales%3F: Confirmed](https://img.shields.io/badge/Leads_sales%3F-Confirmed-8b949e?style=flat-square)\n\n"
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

from deferred_revenue import data, strategy as st

HAVE_REAL = data.have_real()
if HAVE_REAL:
    PX, EV = data.load_real()
else:
    PX = EV = None
print("real cache present:", HAVE_REAL, "| events:", (0 if EV is None else len(EV)),
      "| names:", (0 if EV is None else EV['ticker'].nunique()))
"""

BOOT_CELL = BOOT + "\n# frozen headline numbers (mirror of docs/results.md)\nR = " + repr(R) + "\n"


# ===========================================================================
# 01 — FOR THE CURIOUS
# ===========================================================================
def build_curious():
    cells = [
        md(
            "# Deferred revenue front-runs the sales print. Can you trade it? 📥\n"
            "### A SaaS balance-sheet signal that *genuinely* predicts next quarter's revenue — "
            "and that the market has already figured out\n\n"
            + BADGES +
            "Here's a signal with real accounting logic behind it. When a software company sells "
            "you an annual subscription, it can't book all that cash as revenue on day one — it "
            "parks it on the balance sheet as **deferred revenue** (a liability) and bleeds it "
            "into the income statement over the next four quarters. So a swelling deferred-revenue "
            "balance is, quite literally, *revenue that hasn't shown up yet*. The pitch writes "
            "itself: watch deferred-revenue growth, and you're seeing the future sales print "
            "early — so the stock should follow.\n\n"
            "Half of that is dead right. The other half is where the money would be — and that's "
            "the half that doesn't work.\n\n"
            "> 📓 **Plain-language layer.** Want the Newey-West *t*-stats, the placebo and the "
            "cost math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 40 subscription/SaaS-type US names that report a deferred-"
            "revenue balance on EDGAR, 2009→2026; a genuinely **thin, uneven panel** (the "
            "cross-section is single digits early on). House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does deferred-revenue growth predict future **sales**? | **Yes — strongly.** The "
            f"names with the fastest-growing deferred revenue go on to grow next quarter's sales "
            f"**+{R['ls_spread']:.0f} percentage points** faster than the slowest (correlation "
            f"**+{R['ls_corr']:.2f}**). The accounting lead is real and mechanical. |\n"
            f"| Does it predict future **returns**? | **Not in any way we can certify.** A "
            f"long-short that buys high-deferred-growth names and shorts low ones earns a "
            f"right-signed **+{R['ls_ann']:.1f}%/yr** on paper — but the statistics say that's "
            f"indistinguishable from luck (robust *t* = **+{R['ls_t_nw']:.2f}**, the bar is 2). |\n"
            "| Why the gap? | The lead is **public information**. Anyone can read the balance "
            "sheet; the market prices the coming revenue as soon as the 10-Q lands. A real "
            "signal about the *business* is not automatically a signal about the *stock*. |\n"
            "| Can you trade it? | **No.** The return edge doesn't clear the bar even before "
            "costs — there's nothing to charge costs against. |\n\n"
            "> Real information about the company. No alpha in the stock. That distinction is the "
            "whole study."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"Deferred revenue is bookings that haven't hit the income statement yet. If it's "
            "growing, revenue is coming — so buy the names whose deferred revenue is swelling "
            "fastest.\"*\n\n"
            "It's a specific case of a respectable academic idea — the market misprices the "
            "*accrual* pieces of earnings (Sloan 1996). Deferred revenue is an unusually clean, "
            "*good-news* accrual: unlike most accruals it doesn't reverse against future "
            "earnings, it **becomes** them. The question is whether that accounting lead is also "
            "a *trading* lead."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a plain balance-sheet line predicted stock returns, it would be one of the "
            "easiest anomalies going — no estimates feed, no alt-data, just one number in every "
            "10-Q. That's exactly why it deserves suspicion: it's *too* readable. Everyone with "
            "a brokerage account can pull deferred revenue. If markets price public information "
            "at all, this is the kind they should price fastest."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The signal.** For each of {R['n_names']} subscription-type names, year-over-year "
            "growth in the current deferred-revenue balance, known only on the **filing date** of "
            "the 10-Q/10-K (never before — no peeking at a number that isn't public yet).\n"
            "- **The sales test.** Does this quarter's deferred-revenue growth predict *next* "
            "quarter's revenue growth? (The accounting claim.)\n"
            "- **The return test.** Each month, rank the names on the signal, buy the top third, "
            "short the bottom third, and hold for the next month. Does the spread make money — "
            "and can we tell it apart from noise?\n"
            "- **The mirage check.** If the return spread can't beat a coin-flip relabelling of "
            "the names, it's not a signal, however good the story sounds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the part that works: does deferred revenue really lead sales?** For every "
            "filing we line up this quarter's deferred-revenue growth against the *next* quarter's "
            "actual revenue growth, and split the names into thirds."
        ),
        code(
            "if HAVE_REAL:\n"
            "    q = st.leads_sales(EV)\n"
            "    top, bot, corr = q['top_sales']*100, q['bot_sales']*100, q['corr']\n"
            "else:\n"
            "    top, bot, corr = R['ls_top_sales'], R['ls_bot_sales'], R['ls_corr']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(['bottom third\\n(slow deferred\\ngrowth)','top third\\n(fast deferred\\ngrowth)'],\n"
            "       [bot, top], color=[GREY, GREEN], width=.55)\n"
            "for i,v in enumerate([bot, top]): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel(\"next quarter's revenue growth (YoY)\")\n"
            "ax.set_title(f'Deferred-revenue growth genuinely leads sales (corr {corr:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'future sales growth: top third {top:+.1f}% vs bottom third {bot:+.1f}% '\n"
            "      f'-> spread {top-bot:+.1f} pp, correlation {corr:+.2f}')"
        ),
        md(
            f"That's a **+{R['ls_spread']:.0f} percentage-point** gap in *future* sales growth "
            f"between the fastest- and slowest-deferred-growth names, correlation "
            f"**+{R['ls_corr']:.2f}**. The accounting lead is not folklore — it's about as real "
            "as a cross-sectional signal gets. Deferred revenue really is next year's revenue, "
            "sitting on the balance sheet in plain sight.\n\n"
            "**So now the money question: do the stocks follow?** Same ranking, but instead of "
            "future sales we measure the forward *return* of a buy-the-top-third, short-the-"
            "bottom-third portfolio."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.calendar_ls(PX, EV, signal_col='defrev_yoy', n_buckets=3, min_names=6, staleness_days=200)\n"
            "    s = st.calendar_ls_stats(ls)\n"
            "    ann, tnw = s['ann_pct'], s['t_nw']\n"
            "    cum = (1+ls['ls']).cumprod()\n"
            "else:\n"
            "    ann, tnw = R['ls_ann'], R['ls_t_nw']\n"
            "    cum = None\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "if cum is not None:\n"
            "    ax.plot(cum.index, cum.values, color=AMBER, lw=1.8)\n"
            "    ax.axhline(1.0, c='k', lw=.8)\n"
            "    ax.set_ylabel('growth of $1 (gross, long-short)')\n"
            "    ax.set_title(f'A pretty line that means nothing: +{ann:.1f}%/yr but robust t = {tnw:+.2f}')\n"
            "else:\n"
            "    ax.text(.5,.5,'run with cache for the equity curve',ha='center'); ax.set_axis_off()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'long-short: +{ann:.1f}%/yr gross, but Newey-West t = {tnw:+.2f} (the bar is 2)')"
        ),
        md(
            f"It *slopes up* — **+{R['ls_ann']:.1f}%/yr** gross, growing \\$1 to "
            f"\\${R['ls_cum']:.2f} over {R['n_months']} months. Tempting! But that's the trap. "
            f"Once you account for how noisy and bunchy those monthly returns are, the "
            f"robust *t*-statistic is **+{R['ls_t_nw']:.2f}** — nowhere near the **2** we require "
            f"to call something real. A line that drifts up and a *reliable* edge are different "
            "things, and this is the former.\n\n"
            "**The tell:** if the signal really sorted returns, the middle third should land "
            "between the top and bottom. It doesn't — the thirds are essentially tied."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fr = st.event_drift_frame(PX, EV, horizon=63)\n"
            "    mono = st.bucket_means(fr, 3)*100\n"
            "else:\n"
            "    mono = np.array(R['mono63'])\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.2))\n"
            "ax.bar(['bottom\\nthird','middle\\nthird','top\\nthird'], mono,\n"
            "       color=[GREY, GREY, GREY], width=.55)\n"
            "for i,v in enumerate(mono): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('average 3-month forward return')\n"
            "ax.set_title('No ladder: the return thirds are basically tied (non-monotone)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('3-month forward return by deferred-growth third (low->high):', [f'{v:+.2f}%' for v in mono])"
        ),
        md(
            "No staircase. The fastest-deferred-growth third doesn't reliably out-return the "
            "slowest over the next quarter — the bars wobble instead of climbing. Compare that to "
            "the *sales* chart above, which is a clean, monotone lead. Same signal, two totally "
            "different answers: it forecasts the **business**, not the **stock**."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Does it lead sales? — Confirmed.** +{R['ls_spread']:.0f} pp of future sales "
            f"growth between top and bottom thirds, correlation +{R['ls_corr']:.2f}. Real, "
            "mechanical, not in dispute.\n"
            "- **Signal (returns) — Weak.** The return spread is right-signed and ~"
            f"+{R['ls_ann']:.1f}%/yr on paper, but it never clears the significance bar "
            f"(robust *t* = +{R['ls_t_nw']:.2f}) and the thirds don't form a ladder. There's a "
            "whiff of the effect the literature describes, but this tape can't certify it.\n"
            "- **Tradability — Mirage.** You can't get paid for an edge you can't distinguish "
            "from luck. It fails before costs are even charged.\n\n"
            "> The honest one-liner: *deferred revenue tells you where the company is going, and "
            "the market already knows.*"
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Where an edge might still hide.** Not in the *level* of the public number, but "
            "maybe in the *surprise* — deferred-revenue growth vs what a naïve model would "
            "expect — or in the gap between deferred-revenue growth and the growth analysts are "
            "modelling. That's the residual this study didn't chase.\n"
            "- **The coverage caveat is real.** Early in the sample only a handful of names "
            f"reported the concept (the cross-section grows from ≈{R['xsec_early']:.0f} names in "
            f"2010 to ≈{R['xsec_late']:.0f} by the mid-2020s). A wider, deeper panel might sharpen "
            "the return test — though the flatness of the event-drift makes a hidden edge "
            "unlikely.\n"
            "- **Sibling studies:** [199-sales-growth](../../199-sales-growth/) (ranks on the "
            "*recognised* top line), [534-revenue-surprise-drift](../../534-revenue-surprise-drift/) "
            "(the drift after a *revenue surprise*), and [799-order-backlog-drift](../../799-order-backlog-drift/) "
            "(order backlog / RPO, one step earlier in the cash cycle). See "
            "[docs/references.md](docs/references.md) for the exact dedup.\n\n"
            "*Think the alpha is in the surprise, not the level? Build the expectation model, "
            "show a certifiable net spread on the size you'd actually run — then we'll talk.*"
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
            "# Deferred-Revenue Signal — a quantitative teardown 🔬\n"
            "### A point-in-time calendar-time tercile long-short (Newey-West) · a pooled "
            "event-drift cross-check with a label-shuffle placebo · an era split · the "
            "leads-future-sales regression · a 12-seed synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — **deferred-revenue growth leads future sales *and* returns** — splits cleanly "
            "into two testable pieces, and the two pieces give opposite answers. This is distinct "
            "from every sibling on the desk: [199](../../199-sales-growth/) ranks on *recognised* "
            "sales growth, [534](../../534-revenue-surprise-drift/) on a *revenue surprise* event, "
            "[799](../../799-order-backlog-drift/) on *order backlog*. This is the balance-sheet "
            "**contract-liability** line.\n\n"
            "> ⚠️ **Data note.** EDGAR `DeferredRevenueCurrent` / `ContractWithCustomerLiability"
            "Current` (longest per-name history) + yfinance adjusted closes, "
            + R["n_names"].__str__() + " names, ends "
            + R["end_lo"] + " → " + R["end_hi"] + ", as-of " + R["as_of"] + ". Point-in-time on "
            "the **filing date**. Survivorship named on the Signal axis (current-survivors "
            "basket). Thin/uneven coverage is a first-class caveat. Numbers in "
            "[`docs/results.md`](../docs/results.md) (prices fingerprint `" + R["fp_prices"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** (returns) | `WEAK` | calendar tercile long-short "
            f"**{R['ls_mean_bps']:+.1f} bps/mo** (+{R['ls_ann']:.1f}%/yr gross), one-sample "
            f"*t* = {R['ls_t_iid']:+.2f}, **Newey-West *t* = {R['ls_t_nw']:+.2f}**; asset-scaled "
            f"variant NW *t* = {R['scaled_t_nw']:+.2f} — right-signed, never clears 2 |\n"
            f"| **Tradability** | `MIRAGE` | net of 20 bps + 100 bps borrow: NW *t* = "
            f"{R['net'][(20, 100)][2]:+.2f}, Sharpe {R['net'][(20, 100)][3]:.2f}; fails "
            "**before** costs |\n"
            f"| **Leads future sales?** | `CONFIRMED` | next-q sales-growth regression slope "
            f"+{R['ls_slope']:.2f} (corr +{R['ls_corr']:.2f}), top−bottom tercile spread "
            f"**+{R['ls_spread']:.0f} pp** |\n\n"
            "> 💡 In plain words: the accounting lead is bulletproof and the return edge is a "
            "near-null. A public balance-sheet number forecasts the business, and the market has "
            "already priced the stock accordingly."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $D_{i,q}$ be name $i$'s current deferred-revenue balance at fiscal quarter $q$, "
            "disclosed on filing date $F_{i,q}$. The signal is the point-in-time YoY growth "
            "$g_{i,q} = D_{i,q}/D_{i,q-4} - 1$, known at $F_{i,q}$. The claims:\n\n"
            "- **H₁ (leads sales).** $g_{i,q}$ predicts next quarter's recognised revenue growth "
            "$\\Delta\\text{Rev}_{i,q+1}$ — the accounting mechanism.\n"
            "- **H₂ (leads returns).** A cross-sectional long-short on $g$ earns a positive "
            "forward return spread — the market-mispricing claim.\n"
            "- **H₃ (tradable).** That spread survives realistic long-short costs + borrow.\n\n"
            "We find **H₁ strongly supported** (slope +"
            f"{R['ls_slope']:.2f}, corr +{R['ls_corr']:.2f}, +{R['ls_spread']:.0f} pp tercile "
            f"spread), **H₂ not supported** (NW *t* = {R['ls_t_nw']:+.2f}, flat non-monotone "
            "event drift), and therefore **H₃ moot** (nothing to trade). The literature (Sloan "
            "1996; Prakash-Sinha 2013) supports a real *mispricing*; our tape supports the "
            "*mechanism* but not the *mispricing* — hence a `WEAK`, not `NONE`, on returns."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "The primary is a **calendar-time** long-short, not a pooled event study, precisely "
            "because balance-sheet signals are persistent and filings cluster: a calendar series "
            "of monthly long-short returns lets a **Newey-West (6-lag) HAC *t*** do the honest "
            "work the desk's `REAL` bar is written against. The panel is thin, so we sort into "
            "**terciles** (not quintiles) and require ≥ 6 names in the cross-section. The pooled "
            "event drift + a **label-shuffle placebo** is the cross-check; the leads-sales "
            "regression is graded on **magnitude** (its pooled *t* ignores quarter clustering, "
            "so we read the +32 pp spread and +0.67 correlation, not the literal +27)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Panel.** {R['n_events']:,} (ticker, filing) deferred-revenue quarters across "
            f"{R['n_names']} names, ends {R['end_lo']} → {R['end_hi']}, each stamped with its "
            "10-Q/10-K filing date (point-in-time).\n"
            "- **Primary.** Monthly tercile long-short on `defrev_yoy`, one execution lag "
            "(rank at month $M$ close, earn month $M{+}1$); Newey-West + one-sample *t*, Sharpe, "
            "hit rate.\n"
            "- **Cross-check.** Pooled event drift over 21/63/126 trading days, one-day-lag "
            "entry, top-minus-bottom tercile, one-sample *t* + 10k-draw label-shuffle placebo, "
            "and the tercile monotonicity picture.\n"
            "- **Robustness.** Staleness 120 vs 200 days; the asset-scaled signal "
            "`(ΔDefRev)/Assets`; an era split at 2019 (when the SaaS cohort widens the panel).\n"
            "- **Mechanism.** Pooled OLS of next-quarter revenue growth on `defrev_yoy`.\n"
            "- **Execution.** Long-short net of one-way cost × turnover (both legs) + short "
            "borrow.\n"
            "- **Control.** Synthetic panel, planted-lead knob; the null must not fire across "
            "12 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The primary — calendar-time tercile long-short (Newey-West)\n\n"
            "Rank fresh signals into terciles each month, long top / short bottom equal-weight, "
            "earn next month's return. The decisive statistic is the HAC *t* of the monthly "
            "long-short series."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.calendar_ls(PX, EV, signal_col='defrev_yoy', n_buckets=3, min_names=6, staleness_days=200)\n"
            "    s = st.calendar_ls_stats(ls)\n"
            "    ls_sc = st.calendar_ls(PX, EV, signal_col='defrev_chg_assets', n_buckets=3, min_names=6, staleness_days=200)\n"
            "    s_sc = st.calendar_ls_stats(ls_sc)\n"
            "    print(f\"calendar long-short: {s['mean_bps']:+.1f} bps/mo (+{s['ann_pct']:.2f}%/yr gross), \"\n"
            "          f\"n={s['n_months']} months, avg cross-section {s['avg_n']:.1f}\")\n"
            "    print(f\"  one-sample t = {s['t_iid']:+.2f}   Newey-West(6) t = {s['t_nw']:+.2f}   \"\n"
            "          f\"Sharpe {s['sharpe']:.2f}   hit {s['hit']*100:.0f}%   turnover {s['avg_turnover']:.2f}\")\n"
            "    print(f\"  asset-scaled signal: {s_sc['mean_bps']:+.1f} bps/mo, NW t = {s_sc['t_nw']:+.2f}, Sharpe {s_sc['sharpe']:.2f}\")\n"
            "    cum = (1+ls['ls']).cumprod(); nser = ls['n']\n"
            "else:\n"
            "    cum = None; nser = None\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.3))\n"
            "if cum is not None:\n"
            "    a1.plot(cum.index, cum.values, color=AMBER, lw=1.8); a1.axhline(1, c='k', lw=.8)\n"
            "    a1.set_ylabel('growth of $1 (gross)'); a1.set_title(f\"Long-short: +{R['ls_ann']:.1f}%/yr, NW t = {R['ls_t_nw']:+.2f}\")\n"
            "    a2.plot(nser.index, nser.values, color=GREY, lw=1.5); a2.set_ylabel('names in cross-section')\n"
            "    a2.set_title('Thin & uneven: the panel widens with the SaaS IPO wave')\n"
            "else:\n"
            "    for a in (a1,a2): a.text(.5,.5,'run with cache',ha='center'); a.set_axis_off()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"frozen: {R['ls_mean_bps']:+.1f} bps/mo, NW t = {R['ls_t_nw']:+.2f}, \"\n"
            "      f\"staleness-120 NW t = {R['ls120_t_nw']:+.2f}, asset-scaled NW t = {R['scaled_t_nw']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: **+{R['ls_mean_bps']:.0f} bps/month** is economically real "
            f"money (~+{R['ls_ann']:.1f}%/yr gross), but the HAC *t* is **+{R['ls_t_nw']:.2f}** — "
            f"the monthly returns are too noisy and too few (n={R['n_months']}, avg cross-section "
            f"{R['avg_n']:.0f}) to distinguish from zero. Every specification agrees: staleness-120 "
            f"NW *t* = +{R['ls120_t_nw']:.2f}, and the best case — the asset-scaled signal — tops "
            f"out at NW *t* = +{R['scaled_t_nw']:.2f}. Right-signed, never certified."
        ),
        md(
            "### 4b · The cross-check — pooled event drift + placebo + monotonicity\n\n"
            "Bucket all events by signal; top-minus-bottom forward drift with a label-shuffle "
            "null. If there were a sort, the terciles would form a ladder."
        ),
        code(
            "rows = []\n"
            "if HAVE_REAL:\n"
            "    for h in st.HORIZONS:\n"
            "        es = st.event_summary(PX, EV, horizon=h, n_buckets=3, n_draws=4000)\n"
            "        rows.append((h, es['ls_mean']*100, es['t'], es['ls_win']*100, es['p_placebo']))\n"
            "    mono = st.bucket_means(st.event_drift_frame(PX, EV, horizon=63), 3)*100\n"
            "else:\n"
            "    for h in st.HORIZONS:\n"
            "        d = R['drift'][h]; rows.append((h, d[3], d[4], d[5], d[6]))\n"
            "    mono = np.array(R['mono63'])\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.3))\n"
            "hs = [r[0] for r in rows]; ts = [r[2] for r in rows]\n"
            "a1.bar([f'{h}d' for h in hs], ts, color=GREY, width=.55)\n"
            "a1.axhline(2, ls='--', c=RED, lw=1); a1.axhline(-2, ls='--', c=RED, lw=1); a1.axhline(0, c='k', lw=.8)\n"
            "a1.set_ylabel('one-sample t (long-short drift)'); a1.set_title('Flat: no horizon clears |t|=2')\n"
            "a2.bar(['bottom','middle','top'], mono, color=GREY, width=.55)\n"
            "for i,v in enumerate(mono): a2.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('3-month forward return'); a2.set_title('No ladder (non-monotone terciles)')\n"
            "plt.tight_layout(); plt.show()\n"
            "for h,ls_,t,win,p in rows:\n"
            "    print(f'  H={h:>3}d: long-short {ls_:+.2f}%  t={t:+.2f}  win={win:.0f}%  placebo p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the pooled long-short drift is **flat** at every horizon "
            f"(*t* from {R['drift'][21][4]:+.2f} to {R['drift'][126][4]:+.2f}), the label-shuffle "
            f"placebo *p* sits around **0.4** (a random tercile split beats the real one ~4 times "
            f"in 10), and the terciles are **non-monotone** ({R['mono63'][0]:+.2f}% / "
            f"{R['mono63'][1]:+.2f}% / {R['mono63'][2]:+.2f}% low→high at 63d). The event study "
            "and the calendar long-short agree: no return sort."
        ),
        md(
            "### 4c · Era split — nothing hiding in a regime\n\n"
            "Split the calendar long-short at 2019 (the SaaS-cohort widening)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    e = ls[ls.index < '2019-01-01']['ls'].to_numpy(); l = ls[ls.index >= '2019-01-01']['ls'].to_numpy()\n"
            "    eb, et, en = e.mean()*1e4, st.newey_west_t(e), len(e)\n"
            "    lb, lt, ln = l.mean()*1e4, st.newey_west_t(l), len(l)\n"
            "else:\n"
            "    eb, et, en = R['era_early_bps'], R['era_early_t'], R['era_early_n']\n"
            "    lb, lt, ln = R['era_late_bps'], R['era_late_t'], R['era_late_n']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.2))\n"
            "ax.bar([f'2010-2018\\n(n={en})', f'2019-2026\\n(n={ln})'], [eb, lb], color=[AMBER, AMBER], width=.5)\n"
            "for i,(v,t_) in enumerate([(eb,et),(lb,lt)]): ax.annotate(f'{v:+.0f} bps\\n(NW t={t_:+.2f})',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('long-short mean (bps/mo)')\n"
            "ax.set_title('Both eras: right-signed, neither significant')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'2010-2018: {eb:+.1f} bps NW t={et:+.2f} (n={en})  |  2019-2026: {lb:+.1f} bps NW t={lt:+.2f} (n={ln})')"
        ),
        md(
            f"> 💡 In plain words: {R['era_early_bps']:+.0f} bps (NW *t* = {R['era_early_t']:+.2f}) "
            f"early, {R['era_late_bps']:+.0f} bps (NW *t* = {R['era_late_t']:+.2f}) late — the "
            "same right-signed-but-insignificant positive in both halves. It wasn't a live edge "
            "that decayed; it never certified."
        ),
        md(
            "### 4d · The mechanism — deferred revenue *does* lead sales\n\n"
            "Pooled OLS of next-quarter revenue growth on this quarter's deferred-revenue growth, "
            "and the future-sales spread between the top and bottom deferred-growth terciles."
        ),
        code(
            "if HAVE_REAL:\n"
            "    q = st.leads_sales(EV)\n"
            "    fr = EV.dropna(subset=['defrev_yoy','next_rev_yoy'])\n"
            "    x, y = fr['defrev_yoy'].to_numpy(), fr['next_rev_yoy'].to_numpy()\n"
            "    slope, corr, top, bot = q['slope'], q['corr'], q['top_sales']*100, q['bot_sales']*100\n"
            "else:\n"
            "    x = y = None\n"
            "    slope, corr, top, bot = R['ls_slope'], R['ls_corr'], R['ls_top_sales'], R['ls_bot_sales']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.3))\n"
            "if x is not None:\n"
            "    m = (np.abs(x)<2)&(np.abs(y)<3)\n"
            "    a1.scatter(x[m]*100, y[m]*100, s=8, alpha=.25, color=GREEN)\n"
            "    xs = np.linspace(np.percentile(x,2), np.percentile(x,98), 50)\n"
            "    a1.plot(xs*100, (q['slope']*xs + (y.mean()-q['slope']*x.mean()))*100, color=RED, lw=2)\n"
            "    a1.set_xlabel('deferred-revenue YoY growth (%)'); a1.set_ylabel('NEXT-quarter revenue growth (%)')\n"
            "    a1.set_title(f'Leads sales: slope {slope:+.2f}, corr {corr:+.2f}')\n"
            "else:\n"
            "    a1.text(.5,.5,'run with cache',ha='center'); a1.set_axis_off()\n"
            "a2.bar(['bottom third','top third'], [bot, top], color=[GREY, GREEN], width=.5)\n"
            "for i,v in enumerate([bot, top]): a2.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel(\"next quarter's revenue growth\"); a2.set_title(f'+{top-bot:.0f} pp future-sales spread')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'leads-sales: slope {slope:+.3f}, corr {corr:+.3f}, future-sales spread {top-bot:+.1f} pp (top {top:+.1f}% vs bottom {bot:+.1f}%)')"
        ),
        md(
            f"> 💡 In plain words: this is the study's one unambiguous result — deferred-revenue "
            f"growth **leads recognised sales** with correlation **+{R['ls_corr']:.2f}** and a "
            f"**+{R['ls_spread']:.0f} pp** top-minus-bottom spread in next-quarter revenue growth. "
            "The accounting mechanism is exactly as advertised. The return null is therefore not a "
            "data problem — it's a **market-efficiency** result: the lead is public and priced. "
            "(The regression *t* is huge but pooled across clustered filings; we cite the "
            "magnitude, not the *t*.)"
        ),
        md(
            "### 4e · Tradability — the timer\n\n"
            "For completeness, the calendar long-short net of one-way costs × turnover (both "
            "legs) + short borrow — though a sub-2 gross *t* already settles it."
        ),
        code(
            "rows = []\n"
            "if HAVE_REAL:\n"
            "    for cb, bb in [(10.0,50.0),(20.0,100.0)]:\n"
            "        nt = st.calendar_ls_net(ls, cost_bps=cb, borrow_bps_ann=bb)\n"
            "        rows.append((cb, bb, nt['net_ann_pct'], nt['net_t_nw'], nt['net_sharpe']))\n"
            "else:\n"
            "    for (cb,bb),v in R['net'].items(): rows.append((cb, bb, v[1], v[2], v[3]))\n"
            "labels = [f'{int(cb)}bps +\\n{int(bb)}bps borrow' for cb,bb,_,_,_ in rows]\n"
            "anns = [r[2] for r in rows]\n"
            "fig, ax = plt.subplots(figsize=(8.0, 4.2))\n"
            "ax.bar(labels, anns, color=[AMBER, RED], width=.5)\n"
            "for i,(cb,bb,a,t,sh) in enumerate(rows): ax.annotate(f'+{a:.1f}%/yr\\n(NW t={t:+.2f})',(i,a),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('net long-short (%/yr)')\n"
            "ax.set_title('Costs barely dent it — but the gross edge was never significant')\n"
            "plt.tight_layout(); plt.show()\n"
            "for cb,bb,a,t,sh in rows: print(f'  cost {int(cb)}bps + borrow {int(bb)}bps/yr: +{a:.2f}%/yr net, NW t={t:+.2f}, Sharpe {sh:.2f}')"
        ),
        md(
            f"> 💡 In plain words: turnover is low (~{R['ls_turn']:.2f}/mo — a slow quarterly "
            f"balance-sheet signal), so costs only trim ~+{R['ls_ann']-R['net'][(20,100)][1]:.1f}%/yr. "
            f"But that's irrelevant: net NW *t* = {R['net'][(20,100)][2]:+.2f}, and you cannot be "
            "paid for a spread that wasn't distinguishable from zero to begin with. "
            "**Tradability = MIRAGE.**"
        ),
        md(
            "### 4f · Faithful-engine & power control\n\n"
            "Synthetic price + signal panel with a TUNABLE planted lead (high-deferred-growth "
            "names drift up). The null (edge = 0) is checked over **12 seeds**."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(12):\n"
            "    p0, e0 = data.synthetic_panel(edge=0.0, seed=798 + s_)\n"
            "    null_ts.append(st.synthetic_detect(p0, e0)['t_nw'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "p1, e1 = data.synthetic_panel(edge=0.15, seed=798)\n"
            "planted_t = st.synthetic_detect(p1, e1)['t_nw']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.linspace(-.12,.12,12), null_ts, color=GREY, s=45, label='null worlds (edge=0), 12 seeds')\n"
            "ax.scatter([1], [planted_t], color=GREEN, s=95, zorder=5, label='planted lead (edge=0.15)')\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(0, c='k', lw=.6)\n"
            "ax.set_xticks([0,1]); ax.set_xticklabels(['null x 12','planted'])\n"
            "ax.set_ylabel('calendar long-short Newey-West t')\n"
            "ax.set_title('Control: the null barely fires; a planted lead lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'null: mean NW t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/12  |  planted NW t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 12 null worlds the detector averages NW *t* = "
            f"{R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and crosses |t|=2 only "
            f"{R['syn_null_fire']}/12 times — about what chance gives you for a HAC *t* over ~190 "
            f"months. A planted lead reads NW *t* = {R['syn_planted_t']:.2f}. The machinery is "
            f"unbiased and powered, so the real-tape +{R['ls_t_nw']:.2f} is a genuine near-null, "
            "not a broken pipeline. *(Power check only — never cited in support of a real-tape "
            "stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal (returns) `WEAK`** — calendar tercile long-short {R['ls_mean_bps']:+.1f} "
            f"bps/mo (+{R['ls_ann']:.1f}%/yr gross), one-sample *t* = {R['ls_t_iid']:+.2f}, "
            f"**Newey-West *t* = {R['ls_t_nw']:+.2f}**; asset-scaled best case NW *t* = "
            f"{R['scaled_t_nw']:+.2f}; pooled event drift flat and non-monotone (placebo *p* ≈ "
            "0.4); both eras right-signed but insignificant. Literature-supported, "
            "tape-uncertified.\n"
            f"- **Tradability `MIRAGE`** — net of 20 bps + 100 bps borrow: +"
            f"{R['net'][(20, 100)][1]:.2f}%/yr, NW *t* = {R['net'][(20, 100)][2]:+.2f}, Sharpe "
            f"{R['net'][(20, 100)][3]:.2f}; fails before costs.\n"
            f"- **Leads future sales? `CONFIRMED`** — regression slope +{R['ls_slope']:.2f}, "
            f"correlation +{R['ls_corr']:.2f}, **+{R['ls_spread']:.0f} pp** top-minus-bottom "
            "future-sales spread. The mechanism is real; the market prices it."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The residual worth chasing** is the *surprise*, not the *level*: deferred-"
            "revenue growth relative to a seasonal expectation, or relative to the sales growth "
            "analysts already model. A public level that forecasts the business but not the stock "
            "is the textbook signature of an efficiently-priced fundamental — the alpha, if any, "
            "lives in the part the market can't already see.\n"
            "- **Coverage honesty:** the cross-section runs from ≈"
            f"{R['xsec_early']:.0f} names (2010) to ≈{R['xsec_late']:.0f} (mid-2020s); terciles "
            "on a single-digit early cross-section are noisy, and the flat event drift across "
            "1,200+ pooled events is the more decisive evidence of the return null.\n"
            "- **Dedup map:** [199-sales-growth](../../199-sales-growth/) (recognised top-line "
            "growth, the LSV premium), [534-revenue-surprise-drift](../../534-revenue-surprise-drift/) "
            "(the seasonal-random-walk revenue *surprise* and its drift), "
            "[799-order-backlog-drift](../../799-order-backlog-drift/) (order backlog / RPO — one "
            "step earlier in the cash cycle than deferred revenue). None ranks on the "
            "contract-liability balance itself.\n\n"
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
