"""Generate the two narrative notebooks for Study 858 (Margin / Inventory Divergence).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (EDGAR revenue/cost/inventory +
# yfinance prices, 40 inventory-carrying names, period ends 2009-06 -> 2026-05, as-of 2026-06-30).
R = dict(
    as_of="2026-06-30", n_names=40, n_events=957,
    end_lo="2009-06-30", end_hi="2026-05-03", fp_prices="bc81e3fc2938",
    # primary calendar long-short (divergence, terciles, staleness 200)
    n_months=200, avg_n=18.0, ls_span_lo="2009-11-30", ls_span_hi="2026-06-30",
    ls_mean_bps=-8.1, ls_ann=-0.97, ls_t_iid=-0.20, ls_t_nw=-0.18, ls_sharpe=-0.05,
    ls_hit=51, ls_long_bps=180.9, ls_short_bps=189.0, ls_turn=0.16,
    ls120_mean_bps=4.7, ls120_t_nw=0.09,
    gap_mean_bps=19.7, gap_t_nw=0.43,
    xsec_early=19.9, xsec_late=17.0,
    # pooled event drift  horizon -> (n, top%, bot%, ls%, t, win%, placebo p)
    drift={
        21: (952, 1.53, 0.82, 0.71, 1.09, 52, 0.1153),
        63: (946, 5.78, 4.75, 1.03, 0.72, 50, 0.2009),
        126: (942, 12.39, 9.13, 3.26, 1.34, 52, 0.0580),
    },
    mono63=(4.75, 3.07, 5.78), mono126=(9.13, 6.58, 12.39),
    # era split (calendar LS)
    era_early_n=74, era_early_bps=18.6, era_early_t=0.30,
    era_late_n=126, era_late_bps=-23.8, era_late_t=-0.39,
    # leads margin (mechanism, in pp of gross margin)
    lm_n=834, lm_slope=-0.000, lm_t=-0.74, lm_r2=0.001, lm_corr=-0.026,
    lm_top=0.23, lm_bot=0.79, lm_spread=-0.56,
    # tradability net   (cost, borrow) -> (net bps, net ann, nw t, sharpe)
    net={(10, 50): (-15.6, -1.87, -0.34, -0.09), (20, 100): (-23.0, -2.76, -0.51, -0.14)},
    # synthetic control
    syn_null_mean=-0.02, syn_null_sd=0.92, syn_null_fire=1, syn_null_seeds=12,
    syn_planted_bps=141.3, syn_planted_t=6.15,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Leads margin%3F: No](https://img.shields.io/badge/Leads_margin%3F-No-8b949e?style=flat-square)\n\n"
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

from margin_inventory import data, strategy as st

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
            "# A rising margin *and* a warehouse filling up faster than sales. Contradiction? 📦\n"
            "### A famous accounting signal says that combination is a warning. On 40 real names, "
            "it warns you of nothing.\n\n"
            + BADGES +
            "Here's a story with real accounting logic. When a company's **gross margin is going "
            "up** it looks healthy — it's keeping more of every sales dollar. But if at the same "
            "time its **inventory is piling up faster than its sales are growing**, something "
            "doesn't add up. Either that fat margin is about to be discounted away to clear the "
            "shelves, or the pile of unsold goods is heading for a write-down. Two signals "
            "pointing in *opposite* directions — a **contradiction**. The famous "
            "Abarbanell–Bushee research says: trust the warning, short the contradictory names, "
            "reward the coherent ones.\n\n"
            "It's a great story. On this tape it turns out to be *just* a story.\n\n"
            "> 📓 **Plain-language layer.** Want the Newey-West *t*-stats, the placebo and the "
            "cost math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** 40 inventory-carrying US names (retailers, manufacturers, "
            "staples, hardware) that report revenue, cost of goods and inventory on EDGAR, "
            "2009→2026; a genuinely **thin, uneven panel**. House style in "
            "[METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does the contradiction predict future **returns**? | **No.** A long-short that "
            f"buys the coherent names and shorts the contradictory ones earns "
            f"**{R['ls_ann']:+.1f}%/yr** — a wrong-signed near-zero (robust *t* = "
            f"**{R['ls_t_nw']:+.2f}**, the bar is 2), and it even flips sign between the first "
            "and second halves of the sample. |\n"
            f"| Does it predict the future **fundamentals**? | **No.** The divergence doesn't "
            f"forecast next year's gross-margin change either (correlation **{R['lm_corr']:+.2f}**, "
            "a wrong-signed tercile spread). The accounting mechanism it's built on doesn't show "
            "up. |\n"
            "| Why does the story fail? | Gross margins **mean-revert**, inventory swings are "
            "**seasonal and lumpy**, and every number here is in a public 10-Q the whole market "
            "has read. A tidy narrative about two numbers disagreeing isn't the same as a real, "
            "repeatable edge. |\n"
            "| Can you trade it? | **No** — it's negative before you even pay costs. |\n\n"
            "> A famous, heavily-studied fundamental signal that, taken literally on real filings, "
            "sorts *nothing*. That's the whole study."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"If gross margin is rising while inventory grows faster than sales, the two are "
            "telling contradictory stories — the margin is unsustainable or the inventory is "
            "about to be marked down. Short the contradiction; go long the coherent names.\"*\n\n"
            "It comes from a respectable place: the **fundamental-analysis** literature "
            "(Lev–Thiagarajan 1993; Abarbanell–Bushee 1997/1998) that built hand-crafted "
            "accounting signals to forecast earnings and returns. The inventory signal and the "
            "gross-margin signal are two of the classics. We fuse them into one **divergence** "
            "score and ask whether the fusion actually works."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If a combination of two plain balance-sheet/income-statement lines predicted stock "
            "returns, it would be one of the cheapest anomalies going — no alt-data, no "
            "estimates feed, just three numbers in every 10-Q. That is exactly why it deserves "
            "suspicion. Everyone can compute gross margin, inventory growth and sales growth. If "
            "markets price public accounting at all, this is the kind they should price fastest — "
            "and a tidy 'contradiction' story is precisely the sort of narrative that survives in "
            "folklore long after the data has stopped supporting it."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            f"- **The signal.** For each of {R['n_names']} inventory-carrying names, "
            "`divergence = (change in gross margin) − (inventory growth − sales growth)`, all "
            "year-over-year, known only on the **filing date** of the 10-Q/10-K (no peeking at a "
            "number that isn't public yet). High = coherent; low = contradictory.\n"
            "- **The return test.** Each month, rank the names, buy the top third (coherent), "
            "short the bottom third (contradictory), hold for the next month. Does the spread "
            "make money — and can we tell it apart from noise?\n"
            "- **The fundamentals test.** Does today's divergence predict *next year's* "
            "gross-margin change? (The accounting mechanism the whole thesis rests on.)\n"
            "- **The mirage check.** If the return spread can't beat a coin-flip relabelling of "
            "the names, it's not a signal, however good the story sounds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**Start with the money question: do the stocks follow?** Rank the names on the "
            "divergence each month, buy the coherent third, short the contradictory third, and "
            "track the long-short."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.calendar_ls(PX, EV, signal_col='divergence', n_buckets=3, min_names=6, staleness_days=200)\n"
            "    s = st.calendar_ls_stats(ls)\n"
            "    ann, tnw = s['ann_pct'], s['t_nw']\n"
            "    cum = (1+ls['ls']).cumprod()\n"
            "else:\n"
            "    ann, tnw = R['ls_ann'], R['ls_t_nw']\n"
            "    cum = None\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "if cum is not None:\n"
            "    ax.plot(cum.index, cum.values, color=RED, lw=1.8)\n"
            "    ax.axhline(1.0, c='k', lw=.8)\n"
            "    ax.set_ylabel('growth of $1 (gross, long-short)')\n"
            "    ax.set_title(f'A wander around $1: {ann:+.1f}%/yr, robust t = {tnw:+.2f}')\n"
            "else:\n"
            "    ax.text(.5,.5,'run with cache for the equity curve',ha='center'); ax.set_axis_off()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'long-short: {ann:+.1f}%/yr gross, Newey-West t = {tnw:+.2f} (the bar is 2)')"
        ),
        md(
            f"There's no edge here to get excited about. The long-short drifts sideways around "
            f"\\$1, ends up **{R['ls_ann']:+.1f}%/yr** (the *wrong* sign for the claim), and the "
            f"robust *t*-statistic is **{R['ls_t_nw']:+.2f}** — a rounding error away from zero. "
            "Worse, it isn't even stable: tweak how long a filing stays 'fresh' and the sign "
            "flips.\n\n"
            "**The tell:** if the signal really sorted returns, the middle third should sit "
            "between the top and bottom. It doesn't — the thirds are jumbled."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fr = st.event_drift_frame(PX, EV, horizon=63)\n"
            "    mono = st.bucket_means(fr, 3)*100\n"
            "else:\n"
            "    mono = np.array(R['mono63'])\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.2))\n"
            "ax.bar(['bottom third\\n(contradictory)','middle\\nthird','top third\\n(coherent)'], mono,\n"
            "       color=[GREY, GREY, GREY], width=.55)\n"
            "for i,v in enumerate(mono): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('average 3-month forward return')\n"
            "ax.set_title('No ladder: the middle third is the WORST (non-monotone)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('3-month forward return by divergence third (low->high):', [f'{v:+.2f}%' for v in mono])"
        ),
        md(
            f"No staircase — the *middle* third ({R['mono63'][1]:+.2f}%) actually underperforms "
            f"both the bottom ({R['mono63'][0]:+.2f}%) and the top ({R['mono63'][2]:+.2f}%). "
            "That's the fingerprint of noise, not a signal. A real cross-sectional edge lines the "
            "thirds up in order; this one scrambles them.\n\n"
            "**Maybe the return doesn't move but the *fundamentals* do?** Abarbanell–Bushee's own "
            "logic says the contradictory names should suffer a later margin markdown. Let's line "
            "up today's divergence against *next year's* change in gross margin."
        ),
        code(
            "if HAVE_REAL:\n"
            "    q = st.leads_margin(EV)\n"
            "    top, bot, corr = q['top_margin']*100, q['bot_margin']*100, q['corr']\n"
            "else:\n"
            "    top, bot, corr = R['lm_top'], R['lm_bot'], R['lm_corr']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(['bottom third\\n(contradictory)','top third\\n(coherent)'], [bot, top],\n"
            "       color=[GREY, GREY], width=.55)\n"
            "for i,v in enumerate([bot, top]): ax.annotate(f'{v:+.2f}pp',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel(\"next YEAR's gross-margin change (pp)\")\n"
            "ax.set_title(f'The mechanism fails too: corr {corr:+.2f} (and the wrong way)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'next-year margin change: coherent third {top:+.2f}pp vs contradictory third {bot:+.2f}pp '\n"
            "      f'-> spread {top-bot:+.2f} pp, correlation {corr:+.2f}')"
        ),
        md(
            f"Flat, and if anything backwards: the 'contradictory' names go on to post a "
            f"*slightly larger* margin gain (spread **{R['lm_spread']:+.2f} pp**, correlation "
            f"**{R['lm_corr']:+.2f}**). This is the opposite of the deferred-revenue study "
            "([798](../../798-deferred-revenue-signal/)), where the accounting lead was "
            "bulletproof even though the stock didn't follow. Here **neither** the returns "
            "**nor** the fundamentals cooperate. Gross margins simply mean-revert, and a "
            "one-quarter 'contradiction' tells you nothing durable about where they head next."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal (returns) — None.** The long-short is a wrong-signed null (robust *t* = "
            f"{R['ls_t_nw']:+.2f}), sign-unstable across eras, with a non-monotone, "
            "placebo-insignificant event drift. Nothing to certify.\n"
            f"- **Leads the fundamentals? — No.** Correlation {R['lm_corr']:+.2f} with next "
            "year's margin change; the mechanism is absent.\n"
            "- **Tradability — Mirage.** Negative before costs; costs only make it worse.\n\n"
            "> The honest one-liner: *a contradiction you can spot in a 10-Q is not a secret, and "
            "on this tape it isn't even true.*"
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Where an edge might still hide.** Not in the *level* of a public divergence, but "
            "perhaps in the *surprise* — inventory or margin moves versus a seasonal expectation "
            "— or in narrower, more homogeneous sub-industries where inventory means the same "
            "thing across names. The fused signal on a mixed basket is the version that fails "
            "here.\n"
            "- **The coverage caveat is real.** The quarterly-span filter drops fiscal-Q4 "
            f"figures and the cross-section averages ≈{R['xsec_late']:.0f}–{R['xsec_early']:.0f} "
            "names; a deeper, industry-matched panel might sharpen the test — though a flat event "
            "drift across ~950 pooled events makes a hidden edge unlikely.\n"
            "- **Sibling studies:** [529-inventory-growth](../../529-inventory-growth/) (inventory "
            "growth *level*), [854-cash-conversion-cycle](../../854-cash-conversion-cycle/) "
            "(working-capital cycle), [122-gross-profitability](../../122-gross-profitability/) "
            "(gross profits ÷ assets), [231-sloan-accruals](../../231-sloan-accruals/) (total "
            "accruals). See [docs/references.md](docs/references.md) for the exact dedup.\n\n"
            "*Think the alpha is in the surprise, not the level, or inside one clean industry? "
            "Build it, show a certifiable net spread on the size you'd actually run — then we'll "
            "talk.*"
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
            "# Margin ÷ Inventory Divergence — a quantitative teardown 🔬\n"
            "### A point-in-time calendar-time tercile long-short (Newey-West) · a pooled "
            "event-drift cross-check with a label-shuffle placebo · a sign-flipping era split · "
            "the leads-margin regression · a 12-seed synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "claim — the **Abarbanell–Bushee contradiction** (rising margin + inventory outrunning "
            "sales = a negative signal) — splits into a return test and an accounting-mechanism "
            "test, and **both come back null**. This is distinct from every sibling on the desk: "
            "[529](../../529-inventory-growth/) ranks on inventory-growth *level*, "
            "[854](../../854-cash-conversion-cycle/) on the *cash-conversion cycle*, "
            "[122](../../122-gross-profitability/) on *gross profits ÷ assets*, "
            "[231](../../231-sloan-accruals/) on *total accruals*. This is the fused "
            "margin-vs-inventory-vs-sales **divergence**.\n\n"
            "> ⚠️ **Data note.** EDGAR `Revenues`/`CostOfRevenue`/`InventoryNet` (+ tag "
            "fallbacks) + yfinance adjusted closes, "
            + str(R["n_names"]) + " names, ends "
            + R["end_lo"] + " → " + R["end_hi"] + ", as-of " + R["as_of"] + ". Point-in-time on "
            "the **filing date**. Survivorship named on the Signal axis (current-survivors "
            "basket). Thin/uneven coverage is a first-class caveat. Numbers in "
            "[`docs/results.md`](../docs/results.md) (prices Fingerprint `" + R["fp_prices"] + "`).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** (returns) | `NONE` | calendar tercile long-short "
            f"**{R['ls_mean_bps']:+.1f} bps/mo** ({R['ls_ann']:+.1f}%/yr gross), one-sample "
            f"*t* = {R['ls_t_iid']:+.2f}, **Newey-West *t* = {R['ls_t_nw']:+.2f}**; "
            f"staleness-120 flips to {R['ls120_t_nw']:+.2f}; era split "
            f"{R['era_early_bps']:+.0f}→{R['era_late_bps']:+.0f} bps — wrong-signed, sign-unstable |\n"
            f"| **Tradability** | `MIRAGE` | net of 20 bps + 100 bps borrow: "
            f"{R['net'][(20, 100)][1]:+.2f}%/yr, NW *t* = {R['net'][(20, 100)][2]:+.2f}, "
            f"Sharpe {R['net'][(20, 100)][3]:.2f}; negative before costs |\n"
            f"| **Leads next-year margin?** | `NO` | slope {R['lm_slope']:+.3f} "
            f"(corr {R['lm_corr']:+.2f}), top−bottom tercile spread "
            f"**{R['lm_spread']:+.2f} pp** (wrong-signed) |\n\n"
            "> 💡 In plain words: the return sort is a wrong-signed null, and the accounting "
            "mechanism it was built on is also absent. Neither half of the thesis survives contact "
            "with the tape."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let name $i$ at fiscal quarter $q$ (disclosed on filing date $F_{i,q}$) have gross "
            "margin $M = (\\text{Rev}-\\text{Cost})/\\text{Rev}$. The point-in-time signal is\n\n"
            "$$\\text{div}_{i,q} = \\underbrace{(M_{i,q}-M_{i,q-4})}_{\\Delta\\text{gross margin}} "
            "- \\underbrace{\\big(\\tfrac{I_{i,q}}{I_{i,q-4}}-1 \\;-\\; \\tfrac{\\text{Rev}_{i,q}}"
            "{\\text{Rev}_{i,q-4}}+1\\big)}_{\\text{inventory outrunning sales}},$$\n\n"
            "high = coherent (the long), low = contradictory (the short). The claims:\n\n"
            "- **H₁ (leads returns).** A cross-sectional long-short on $\\text{div}$ earns a "
            "positive forward return spread — the market-mispricing claim.\n"
            "- **H₂ (leads fundamentals).** $\\text{div}$ predicts next year's gross-margin change "
            "$\\Delta M_{i,q+4}$ — the accounting mechanism.\n"
            "- **H₃ (tradable).** Any spread survives realistic long-short costs + borrow.\n\n"
            "We find **H₁ not supported** (NW *t* = "
            f"{R['ls_t_nw']:+.2f}, sign-unstable), **H₂ not supported** (corr "
            f"{R['lm_corr']:+.2f}, wrong-signed spread), and therefore **H₃ moot**. The literature "
            "(Abarbanell–Bushee 1997/1998) reports a real effect in an older, broader,"
            " earnings-based sample; our point-in-time return/margin test on this basket does not "
            "reproduce it — hence a `NONE`, not a `WEAK`."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "The primary is a **calendar-time** long-short, not a pooled event study, because "
            "fundamental signals are persistent and filings cluster: a calendar series of monthly "
            "long-short returns lets a **Newey-West (6-lag) HAC *t*** do the honest work the "
            "desk's `REAL` bar is written against. The panel is thin, so we sort into **terciles** "
            "(not quintiles) and require ≥ 6 names in the cross-section. The pooled event drift + "
            "a **label-shuffle placebo** is the cross-check; the leads-margin regression is graded "
            "on **magnitude** (its pooled *t* ignores quarter clustering)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Panel.** {R['n_events']:,} (ticker, filing) quarters across {R['n_names']} "
            f"names, ends {R['end_lo']} → {R['end_hi']}, each stamped with its 10-Q/10-K filing "
            "date (point-in-time).\n"
            "- **Primary.** Monthly tercile long-short on `divergence`, one execution lag "
            "(rank at month $M$ close, earn month $M{+}1$); Newey-West + one-sample *t*, Sharpe, "
            "hit rate.\n"
            "- **Cross-check.** Pooled event drift over 21/63/126 trading days, one-day-lag "
            "entry, top-minus-bottom tercile, one-sample *t* + 10k-draw label-shuffle placebo, "
            "and the tercile monotonicity picture.\n"
            "- **Robustness.** Staleness 120 vs 200 days; the classic `−inv_sales_gap` signal "
            "alone; an era split at 2016.\n"
            "- **Mechanism.** Pooled OLS of next-year gross-margin change on `divergence`.\n"
            "- **Execution.** Long-short net of one-way cost × turnover (both legs) + short "
            "borrow.\n"
            "- **Control.** Synthetic panel, planted-effect knob; the null must not fire across "
            "12 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The primary — calendar-time tercile long-short (Newey-West)\n\n"
            "Rank fresh signals into terciles each month, long top (coherent) / short bottom "
            "(contradictory) equal-weight, earn next month's return. The decisive statistic is "
            "the HAC *t* of the monthly long-short series."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.calendar_ls(PX, EV, signal_col='divergence', n_buckets=3, min_names=6, staleness_days=200)\n"
            "    s = st.calendar_ls_stats(ls)\n"
            "    ls_gap = st.calendar_ls(PX, EV, signal_col='inv_sales_gap', n_buckets=3, min_names=6, staleness_days=200)\n"
            "    s_gap = st.calendar_ls_stats(ls_gap)\n"
            "    print(f\"calendar long-short: {s['mean_bps']:+.1f} bps/mo ({s['ann_pct']:+.2f}%/yr gross), \"\n"
            "          f\"n={s['n_months']} months, avg cross-section {s['avg_n']:.1f}\")\n"
            "    print(f\"  one-sample t = {s['t_iid']:+.2f}   Newey-West(6) t = {s['t_nw']:+.2f}   \"\n"
            "          f\"Sharpe {s['sharpe']:.2f}   hit {s['hit']*100:.0f}%   turnover {s['avg_turnover']:.2f}\")\n"
            "    print(f\"  classic -inv_sales_gap alone (long LOW gap): {-s_gap['mean_bps']:+.1f} bps/mo, NW t = {-s_gap['t_nw']:+.2f}\")\n"
            "    cum = (1+ls['ls']).cumprod(); nser = ls['n']\n"
            "else:\n"
            "    cum = None; nser = None\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.3))\n"
            "if cum is not None:\n"
            "    a1.plot(cum.index, cum.values, color=RED, lw=1.8); a1.axhline(1, c='k', lw=.8)\n"
            "    a1.set_ylabel('growth of $1 (gross)'); a1.set_title(f\"Long-short: {R['ls_ann']:+.1f}%/yr, NW t = {R['ls_t_nw']:+.2f}\")\n"
            "    a2.plot(nser.index, nser.values, color=GREY, lw=1.5); a2.set_ylabel('names in cross-section')\n"
            "    a2.set_title('Thin & uneven: the panel wobbles around ~18 names')\n"
            "else:\n"
            "    for a in (a1,a2): a.text(.5,.5,'run with cache',ha='center'); a.set_axis_off()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"frozen: {R['ls_mean_bps']:+.1f} bps/mo, NW t = {R['ls_t_nw']:+.2f}, \"\n"
            "      f\"staleness-120 NW t = {R['ls120_t_nw']:+.2f}, gap-alone NW t = {R['gap_t_nw']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: **{R['ls_mean_bps']:+.0f} bps/month** is a wrong-signed "
            f"rounding error (~{R['ls_ann']:+.1f}%/yr), and the HAC *t* is **{R['ls_t_nw']:+.2f}**. "
            f"Every specification agrees it's a null: staleness-120 flips to NW *t* = "
            f"{R['ls120_t_nw']:+.2f}, and the classic inventory-vs-sales gap alone reaches only NW "
            f"*t* = {R['gap_t_nw']:+.2f}. Nothing here approaches the bar."
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
            "a2.set_ylabel('3-month forward return'); a2.set_title('No ladder (middle third is worst)')\n"
            "plt.tight_layout(); plt.show()\n"
            "for h,ls_,t,win,p in rows:\n"
            "    print(f'  H={h:>3}d: long-short {ls_:+.2f}%  t={t:+.2f}  win={win:.0f}%  placebo p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the pooled long-short drift is **right-signed but flat** at "
            f"every horizon (*t* from {R['drift'][63][4]:+.2f} to {R['drift'][126][4]:+.2f}), the "
            f"label-shuffle placebo *p* runs **0.06–0.20** (a random tercile split beats the real "
            f"one one time in five to six), and the terciles are **non-monotone** "
            f"({R['mono63'][0]:+.2f}% / {R['mono63'][1]:+.2f}% / {R['mono63'][2]:+.2f}% low→high — "
            "the middle is worst). The event study and the calendar long-short agree: no return "
            "sort."
        ),
        md(
            "### 4c · Era split — the sign isn't even stable\n\n"
            "Split the calendar long-short at 2016 (≈ halves the sample)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    e = ls[ls.index < '2016-01-01']['ls'].to_numpy(); l = ls[ls.index >= '2016-01-01']['ls'].to_numpy()\n"
            "    eb, et, en = e.mean()*1e4, st.newey_west_t(e), len(e)\n"
            "    lb, lt, ln = l.mean()*1e4, st.newey_west_t(l), len(l)\n"
            "else:\n"
            "    eb, et, en = R['era_early_bps'], R['era_early_t'], R['era_early_n']\n"
            "    lb, lt, ln = R['era_late_bps'], R['era_late_t'], R['era_late_n']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.2))\n"
            "ax.bar([f'pre-2016\\n(n={en})', f'2016-2026\\n(n={ln})'], [eb, lb], color=[GREY, GREY], width=.5)\n"
            "for i,(v,t_) in enumerate([(eb,et),(lb,lt)]): ax.annotate(f'{v:+.0f} bps\\n(NW t={t_:+.2f})',(i,v),ha='center',va='bottom' if v>=0 else 'top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('long-short mean (bps/mo)')\n"
            "ax.set_title('Sign flips across eras — the signature of noise')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'pre-2016: {eb:+.1f} bps NW t={et:+.2f} (n={en})  |  2016-2026: {lb:+.1f} bps NW t={lt:+.2f} (n={ln})')"
        ),
        md(
            f"> 💡 In plain words: {R['era_early_bps']:+.0f} bps (NW *t* = {R['era_early_t']:+.2f}) "
            f"early, {R['era_late_bps']:+.0f} bps (NW *t* = {R['era_late_t']:+.2f}) late — "
            "**right-signed then wrong-signed**, neither significant. A live edge decays in one "
            "direction; a non-edge flips sign. This is the latter."
        ),
        md(
            "### 4d · The mechanism — does the divergence lead next-year margin?\n\n"
            "Pooled OLS of next-year gross-margin change on this quarter's divergence, and the "
            "future-margin spread between the top and bottom divergence terciles. This is the "
            "Abarbanell–Bushee accounting claim itself."
        ),
        code(
            "if HAVE_REAL:\n"
            "    q = st.leads_margin(EV)\n"
            "    fr = EV.dropna(subset=['divergence','next_d_gross_margin'])\n"
            "    x, y = fr['divergence'].to_numpy(), fr['next_d_gross_margin'].to_numpy()\n"
            "    slope, corr, top, bot = q['slope'], q['corr'], q['top_margin']*100, q['bot_margin']*100\n"
            "else:\n"
            "    x = y = None\n"
            "    slope, corr, top, bot = R['lm_slope'], R['lm_corr'], R['lm_top'], R['lm_bot']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.3))\n"
            "if x is not None:\n"
            "    m = (np.abs(x)<1.0)&(np.abs(y)<0.3)\n"
            "    a1.scatter(x[m]*100, y[m]*100, s=8, alpha=.2, color=GREY)\n"
            "    xs = np.linspace(np.percentile(x,2), np.percentile(x,98), 50)\n"
            "    a1.plot(xs*100, (q['slope']*xs + (y.mean()-q['slope']*x.mean()))*100, color=RED, lw=2)\n"
            "    a1.set_xlabel('divergence (%)'); a1.set_ylabel('NEXT-year gross-margin change (pp)')\n"
            "    a1.set_ylim(-15, 15)\n"
            "    a1.set_title(f'No lead: slope {slope:+.3f}, corr {corr:+.2f}')\n"
            "else:\n"
            "    a1.text(.5,.5,'run with cache',ha='center'); a1.set_axis_off()\n"
            "a2.bar(['contradictory\\n(bottom)','coherent\\n(top)'], [bot, top], color=[GREY, GREY], width=.5)\n"
            "for i,v in enumerate([bot, top]): a2.annotate(f'{v:+.2f}pp',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel(\"next year's gross-margin change (pp)\"); a2.set_title(f'{top-bot:+.2f} pp spread (wrong way)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'leads-margin: slope {slope:+.3f}, corr {corr:+.3f}, next-yr margin spread {top-bot:+.2f} pp (coherent {top:+.2f}pp vs contradictory {bot:+.2f}pp)')"
        ),
        md(
            f"> 💡 In plain words: this is the study's decisive negative — the divergence **does "
            f"not lead** recognised margins (correlation **{R['lm_corr']:+.2f}**, and the "
            f"top-minus-bottom future-margin spread is **{R['lm_spread']:+.2f} pp**, the *wrong* "
            "sign). Contrast [798](../../798-deferred-revenue-signal/), where the accounting lead "
            "was real (+0.67 correlation) and only the *return* was a null. Here the return null "
            "is joined by a **mechanism null**: gross margins mean-revert, and a one-quarter "
            "contradiction carries no durable information about them."
        ),
        md(
            "### 4e · Tradability — the timer\n\n"
            "For completeness, the calendar long-short net of one-way costs × turnover (both "
            "legs) + short borrow — though a wrong-signed gross null already settles it."
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
            "ax.bar(labels, anns, color=[RED, RED], width=.5)\n"
            "for i,(cb,bb,a,t,sh) in enumerate(rows): ax.annotate(f'{a:+.1f}%/yr\\n(NW t={t:+.2f})',(i,a),ha='center',va='top')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('net long-short (%/yr)')\n"
            "ax.set_title('Negative before costs, more negative after')\n"
            "plt.tight_layout(); plt.show()\n"
            "for cb,bb,a,t,sh in rows: print(f'  cost {int(cb)}bps + borrow {int(bb)}bps/yr: {a:+.2f}%/yr net, NW t={t:+.2f}, Sharpe {sh:.2f}')"
        ),
        md(
            f"> 💡 In plain words: the gross spread is a wrong-signed null, so costs only push it "
            f"further under water (net NW *t* = {R['net'][(20,100)][2]:+.2f}, Sharpe "
            f"{R['net'][(20,100)][3]:.2f}). **Tradability = MIRAGE** — there was never a paycheck "
            "to charge costs against."
        ),
        md(
            "### 4f · Faithful-engine & power control\n\n"
            "Synthetic price + signal panel with a TUNABLE planted effect (high-divergence names "
            "drift up). The null (edge = 0) is checked over **12 seeds**."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(12):\n"
            "    p0, e0 = data.synthetic_panel(edge=0.0, seed=858 + s_)\n"
            "    null_ts.append(st.synthetic_detect(p0, e0)['t_nw'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "p1, e1 = data.synthetic_panel(edge=0.15, seed=858)\n"
            "planted_t = st.synthetic_detect(p1, e1)['t_nw']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.linspace(-.12,.12,12), null_ts, color=GREY, s=45, label='null worlds (edge=0), 12 seeds')\n"
            "ax.scatter([1], [planted_t], color=GREEN, s=95, zorder=5, label='planted effect (edge=0.15)')\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(0, c='k', lw=.6)\n"
            "ax.set_xticks([0,1]); ax.set_xticklabels(['null x 12','planted'])\n"
            "ax.set_ylabel('calendar long-short Newey-West t')\n"
            "ax.set_title('Control: the null barely fires; a planted effect lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'null: mean NW t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/12  |  planted NW t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 12 null worlds the detector averages NW *t* = "
            f"{R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and crosses |t|=2 only "
            f"{R['syn_null_fire']}/12 times — about what chance gives you for a HAC *t* over ~200 "
            f"months. A planted effect reads NW *t* = {R['syn_planted_t']:.2f}. The machinery is "
            f"unbiased and powered, so the real-tape {R['ls_t_nw']:+.2f} is a genuine null, not a "
            "broken pipeline. *(Power check only — never cited in support of a real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal (returns) `NONE`** — calendar tercile long-short {R['ls_mean_bps']:+.1f} "
            f"bps/mo ({R['ls_ann']:+.1f}%/yr gross), one-sample *t* = {R['ls_t_iid']:+.2f}, "
            f"**Newey-West *t* = {R['ls_t_nw']:+.2f}**; staleness-120 flips to "
            f"{R['ls120_t_nw']:+.2f}; era split {R['era_early_bps']:+.0f}→{R['era_late_bps']:+.0f} "
            "bps; pooled event drift right-signed but flat, non-monotone, placebo-insignificant. "
            "Wrong-signed and sign-unstable.\n"
            f"- **Tradability `MIRAGE`** — net of 20 bps + 100 bps borrow: "
            f"{R['net'][(20, 100)][1]:+.2f}%/yr, NW *t* = {R['net'][(20, 100)][2]:+.2f}, Sharpe "
            f"{R['net'][(20, 100)][3]:.2f}; negative before costs.\n"
            f"- **Leads next-year margin? `NO`** — slope {R['lm_slope']:+.3f}, correlation "
            f"{R['lm_corr']:+.2f}, **{R['lm_spread']:+.2f} pp** (wrong-signed) top-minus-bottom "
            "future-margin spread. The accounting mechanism is absent too."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The residual worth chasing** is the *surprise*, not the *level*: inventory or "
            "margin moves relative to a seasonal expectation, or the divergence measured inside a "
            "single homogeneous industry where 'inventory' means the same thing across names. The "
            "fused signal on a mixed retail/manufacturing/staples/hardware basket is the version "
            "that fails here.\n"
            "- **Coverage honesty:** the quarterly-span filter drops fiscal-Q4 figures, six "
            "basket names lack usable history, and the cross-section averages ≈"
            f"{R['xsec_late']:.0f}–{R['xsec_early']:.0f} names; but the flat, sign-flipping "
            "evidence across ~950 pooled events is decisive for the null.\n"
            "- **Dedup map:** [529-inventory-growth](../../529-inventory-growth/) (inventory "
            "growth level), [854-cash-conversion-cycle](../../854-cash-conversion-cycle/) "
            "(working-capital cycle), [122-gross-profitability](../../122-gross-profitability/) "
            "(gross profits ÷ assets), [231-sloan-accruals](../../231-sloan-accruals/) (total "
            "accruals). None ranks on the margin-vs-inventory-vs-sales divergence itself.\n\n"
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
