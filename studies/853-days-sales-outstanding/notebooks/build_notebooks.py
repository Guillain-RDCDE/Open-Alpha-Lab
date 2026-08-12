"""Generate the two narrative notebooks for Study 853 (Days-Sales-Outstanding Buildup).

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


# Frozen real-tape headline numbers — mirror of docs/results.md (EDGAR receivables + revenue +
# yfinance prices, 38 large US filers with matched history, period ends 2009-07-31 → 2026-04-30,
# as-of 2026-06-30). Reproduce with examples/verify.py.
R = dict(
    as_of="2026-06-30", n_names=38, n_events=1173,
    end_lo="2009-07-31", end_hi="2026-04-30", fp_prices="83a4ccfa2027",
    n_months=191, avg_n=21.4, ls_span_lo="2010-08-31", ls_span_hi="2026-06-30",
    ls_mean_bps=14.2, ls_ann=1.71, ls_t_iid=0.55, ls_t_nw=0.53, ls_sharpe=0.14,
    ls_hit=52, ls_hit_lo=45, ls_hit_hi=59, ls_long_bps=148.6, ls_short_bps=134.3,
    ls_turn=0.17, ls_cum=1.16,
    ls120_mean_bps=4.8, ls120_t_nw=0.16,
    pct_mean_bps=6.6, pct_t_nw=0.27, pct_sharpe=0.06,
    xsec_early=27.3, xsec_late=12.1,
    drift={21: (1172, 1.31, 1.30, 0.01, 0.01, 50, 0.4927),
           63: (1163, 4.43, 4.32, 0.11, 0.11, 51, 0.4363),
           126: (1157, 8.10, 9.12, -1.01, -0.74, 50, 0.8123)},
    mono63=(4.32, 3.55, 4.43),
    era_early_n=89, era_early_bps=29.8, era_early_t=1.03,
    era_late_n=102, era_late_bps=0.7, era_late_t=0.02,
    ws_n=997, ws_slope=-0.0031, ws_t=-2.67, ws_r2=0.007, ws_corr=-0.0843,
    ws_low_sales=8.2, ws_high_sales=4.9, ws_spread=3.3,
    net={(10, 50): (6.6, 0.79, 0.25, 0.06), (20, 100): (-1.0, -0.13, -0.04, -0.01)},
    syn_null_mean=0.01, syn_null_sd=0.73, syn_null_fire=0, syn_null_seeds=12,
    syn_planted_bps=860.8, syn_planted_t=18.51,
)

BADGES = (
    "![Signal: None](https://img.shields.io/badge/Signal-None-c0392b?style=flat-square)\n"
    "![Tradability: Mirage](https://img.shields.io/badge/Tradability-Mirage-c0392b?style=flat-square)\n"
    "![Warns on sales%3F: Faint](https://img.shields.io/badge/Warns_on_sales%3F-Faint-8b949e?style=flat-square)\n\n"
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

from dso_signal import data, strategy as st

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
            "# A rising pile of unpaid invoices is supposed to be a red flag. Is it? 🧾\n"
            "### Days Sales Outstanding — the receivables-vs-sales warning light from the "
            "forensic-accounting playbook — meets a liquid tape of big US companies\n\n"
            + BADGES +
            "Here's a signal with real forensic-accounting logic behind it. When a company ships "
            "product but books the sale on credit, the money it's owed piles up as **accounts "
            "receivable**. Divide that pile by daily sales and you get **Days Sales Outstanding "
            "(DSO)** — how many days of revenue are stuck in unpaid invoices. When DSO *climbs* — "
            "receivables growing faster than sales — the textbooks get nervous: maybe the company "
            "is **channel-stuffing** (jamming product into the channel on soft credit terms to "
            "make the quarter), maybe it's booking revenue too aggressively, maybe customers just "
            "aren't paying. The pitch: rising DSO is tomorrow's bad news today — so **avoid (or "
            "short) the names whose DSO is building, and favour the ones keeping it lean.**\n\n"
            "It's a famous fundamental signal (Abarbanell-Bushee). Does the warning light actually "
            "predict anything you can trade?\n\n"
            "> 📓 **Plain-language layer.** Want the Newey-West *t*-stats, the placebo and the "
            "cost math? See **[02_for_the_quants.ipynb](02_for_the_quants.ipynb)**.\n"
            ">\n"
            "> ⚠️ **Method note.** ~38 large US filers with real trade receivables (EDGAR "
            "`AccountsReceivableNetCurrent` + `Revenues`), 2009→2026; a genuinely **thin, uneven "
            "panel**. House style in [METHODOLOGY.md](../../../METHODOLOGY.md)."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## The answer first\n\n"
            "| Question | Answer |\n|---|---|\n"
            f"| Does a building DSO predict **lower future returns**, as the red flag claims? | "
            f"**No — not on this tape.** Long-lean / short-building earns a *right-signed but "
            f"trivial* **+{R['ls_ann']:.1f}%/yr** gross, robust *t* = **+{R['ls_t_nw']:.2f}** (the "
            f"bar is 2). More conservative versions of the signal are flat-zero, and the whole "
            f"thing is **dead after 2018** (*t* = +{R['era_late_t']:.2f}). |\n"
            f"| Does it at least **warn on future sales**? | **Faintly.** Lean-DSO names do grow "
            f"next quarter's sales a bit faster than building-DSO names (+{R['ws_spread']:.1f} "
            f"points), correctly signed — but the correlation is a whisker (**{R['ws_corr']:+.2f}**). "
            "The mechanism is there in the sign, essentially absent in the magnitude. |\n"
            "| Can you trade it? | **No.** It fails *before* costs, and after realistic costs + "
            f"borrow it goes **negative** ({R['net'][(20, 100)][1]:+.1f}%/yr). |\n"
            "| Why so weak? | These are **big, liquid, heavily-covered** companies. A ratio anyone "
            "can compute from the 10-Q is the last place a durable edge survives — and here it "
            "didn't survive at all. |\n\n"
            "> A famous forensic red flag that, on blue chips, lights up for neither the stock nor "
            "(much) the sales it's supposed to foretell."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim\n\n"
            "> *\"When receivables grow faster than sales, Days Sales Outstanding rises. That's a "
            "red flag — channel-stuffing, aggressive revenue recognition, weak collections — and "
            "it precedes disappointment. Buy the lean-DSO names, short the building-DSO names.\"*\n\n"
            "It's one of the nine fundamental signals in **Abarbanell & Bushee (1997/1998)**, a "
            "cousin of Sloan's accrual anomaly (a receivable *is* an accrual) and a term inside "
            "both the Piotroski F-score and the Beneish M-score. The receivables-vs-sales gap is "
            "as respectable as forensic red flags get."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what?\n\n"
            "If one plain balance-sheet ratio flagged future losers, it would be one of the "
            "easiest anomalies going — no estimates feed, no alt-data, just receivables ÷ sales "
            "out of every 10-Q. That's exactly why it deserves suspicion: it's *too* readable. "
            "Everyone with a brokerage account can compute DSO. If markets price public accounting "
            "information at all, this is the kind they should price fastest."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How would we even know?\n\n"
            "- **The signal.** For each name, the year-over-year **change in DSO** (this quarter's "
            "DSO minus the same quarter a year ago), known only on the **filing date** of the "
            "10-Q/10-K — never before the number is public.\n"
            "- **The sales test.** Does a rising DSO this quarter precede *weaker* revenue growth "
            "next quarter? (The channel-stuffing mechanism.)\n"
            "- **The return test.** Each month, rank the names on DSO change, buy the leanest "
            "third, short the building third, hold one month. Does the spread make money — and can "
            "we tell it apart from noise?\n"
            "- **The mirage check.** If the return spread can't beat a coin-flip relabelling of the "
            "names, it's not a signal, however good the story sounds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md(
            "## 4 · The teardown — let's actually look\n\n"
            "**First, the part the story leans on: does a building DSO warn on sales?** For every "
            "filing we line up this quarter's DSO change against *next* quarter's actual revenue "
            "growth, and split the names into a lean third and a building third."
        ),
        code(
            "if HAVE_REAL:\n"
            "    q = st.warns_sales(EV)\n"
            "    low, high, corr = q['low_sales']*100, q['high_sales']*100, q['corr']\n"
            "else:\n"
            "    low, high, corr = R['ws_low_sales'], R['ws_high_sales'], R['ws_corr']\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.4))\n"
            "ax.bar(['building DSO\\n(top third)','lean DSO\\n(bottom third)'],\n"
            "       [high, low], color=[GREY, GREEN], width=.55)\n"
            "for i,v in enumerate([high, low]): ax.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel(\"next quarter's revenue growth (YoY)\")\n"
            "ax.set_title(f'Does a building DSO precede weaker sales? (corr {corr:+.2f})')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'future sales growth: lean-DSO third {low:+.1f}% vs building-DSO third {high:+.1f}% '\n"
            "      f'-> spread {low-high:+.1f} pp, correlation {corr:+.2f}')"
        ),
        md(
            f"So there's a *nudge* in the right direction: lean-DSO names go on to grow next "
            f"quarter's sales about **+{R['ws_spread']:.1f} points** faster than building-DSO names, "
            f"and the correlation is negative as the red flag predicts. But look at how *small* it "
            f"is — correlation **{R['ws_corr']:+.2f}**, basically a flat scatter. The channel-"
            "stuffing mechanism exists in the *sign* and almost vanishes in the *magnitude*. Hold "
            "that thought — because it's the strongest thing in this whole study.\n\n"
            "**Now the money question: do the stocks follow?** Same ranking, but instead of future "
            "sales we measure the forward *return* of a buy-the-lean-third, short-the-building-"
            "third portfolio."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.calendar_ls(PX, EV, signal_col='dso_score', n_buckets=3, min_names=6, staleness_days=200)\n"
            "    s = st.calendar_ls_stats(ls)\n"
            "    ann, tnw = s['ann_pct'], s['t_nw']\n"
            "    cum = (1+ls['ls']).cumprod()\n"
            "else:\n"
            "    ann, tnw = R['ls_ann'], R['ls_t_nw']\n"
            "    cum = None\n"
            "fig, ax = plt.subplots(figsize=(9.2, 4.4))\n"
            "if cum is not None:\n"
            "    ax.plot(cum.index, cum.values, color=AMBER, lw=1.8); ax.axhline(1.0, c='k', lw=.8)\n"
            "    ax.set_ylabel('growth of $1 (gross, long lean / short building)')\n"
            "    ax.set_title(f'Long lean-DSO / short building-DSO: {ann:+.1f}%/yr, robust t = {tnw:+.2f}')\n"
            "else:\n"
            "    ax.text(.5,.5,'run with cache for the equity curve',ha='center'); ax.set_axis_off()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'long-short: {ann:+.1f}%/yr gross, Newey-West t = {tnw:+.2f} (the bar is 2)')"
        ),
        md(
            f"It drifts up a touch — **+{R['ls_ann']:.1f}%/yr** gross, growing \\$1 to only "
            f"\\${R['ls_cum']:.2f} over {R['n_months']} months — and it's right-signed (lean beats "
            f"building, as the claim wants). But the robust *t*-statistic is **+{R['ls_t_nw']:.2f}**, "
            f"nowhere near the **2** we require. This is about as close to a flat line as a "
            f"long-short gets. And it's worse than it looks: a more conservative version of the "
            f"signal (drop stale filings) is **+{R['ls120_mean_bps']:.0f} bps/mo** (*t* = "
            f"+{R['ls120_t_nw']:.2f}), and after 2018 the whole thing is **stone dead** "
            f"(+{R['era_late_bps']:.0f} bps/mo, *t* = +{R['era_late_t']:.2f}).\n\n"
            "**The tell:** if the signal really sorted returns, the thirds would form a ladder from "
            "building to lean. They don't."
        ),
        code(
            "if HAVE_REAL:\n"
            "    fr = st.event_drift_frame(PX, EV, horizon=63)\n"
            "    mono = st.bucket_means(fr, 3)*100\n"
            "else:\n"
            "    mono = np.array(R['mono63'])\n"
            "fig, ax = plt.subplots(figsize=(8.4, 4.2))\n"
            "ax.bar(['building\\nDSO','middle','lean\\nDSO'], mono, color=[GREY, GREY, GREY], width=.55)\n"
            "for i,v in enumerate(mono): ax.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "ax.set_ylabel('average 3-month forward return')\n"
            "ax.set_title('The return thirds (high buildup -> low buildup)')\n"
            "plt.tight_layout(); plt.show()\n"
            "print('3-month forward return by DSO third (building->lean):', [f'{v:+.2f}%' for v in mono])"
        ),
        md(
            f"No staircase — the thirds come in **{R['mono63'][0]:+.2f}% / {R['mono63'][1]:+.2f}% / "
            f"{R['mono63'][2]:+.2f}%** (building → lean), a wobble, not a ladder. Over a longer "
            "2-quarter window the ordering actually **flips** (the building third out-returns the "
            "lean one). When the sign of your 'edge' depends on the horizon you measure, you don't "
            "have an edge — you have noise. Compare that to the *sales* nudge above, which at least "
            "pointed the right way. Same signal: a faint truth about the business, nothing about "
            "the stock."
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Warns on future sales? — Faint.** Lean-DSO names grow next quarter's sales "
            f"+{R['ws_spread']:.1f} points faster than building-DSO names, correctly signed — but "
            f"correlation just {R['ws_corr']:+.2f}. The mechanism is real in direction, negligible "
            "in size.\n"
            f"- **Signal (returns) — None.** The long-short is right-signed but statistically zero "
            f"(robust *t* = +{R['ls_t_nw']:.2f}), the thirds don't form a ladder (and flip sign at "
            f"2 quarters), and it's dead after 2018. No certifiable return predictability on this "
            "tape.\n"
            "- **Tradability — Mirage.** It fails before costs and turns negative after them. "
            "Nothing to trade.\n\n"
            "> The honest one-liner: *on big liquid names, a rising DSO is a real-but-tiny hint "
            "about next quarter's sales and — as far as this tape can tell — no hint at all about "
            "the stock.*"
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further 🚪\n\n"
            "- **Where an edge might still hide.** Not in the *level* of the public ratio, but "
            "maybe in the *surprise* — DSO change vs a seasonal expectation — or by conditioning on "
            "firms where a receivables build is genuinely unusual (small, opaque, or fast-growing "
            "names, not mega-cap blue chips). This study deliberately used large, liquid, "
            "heavily-covered names — the *hardest* place for a public red flag to survive.\n"
            "- **The coverage caveat is real.** Only a subset of names expose a long, cleanly-"
            "matched receivables+revenue history on EDGAR; the cross-section is thin early on.\n"
            "- **Sibling studies:** [231-sloan-accruals](../../231-sloan-accruals/) (total "
            "accruals), [522-percent-operating-accruals](../../522-percent-operating-accruals/), "
            "[529-inventory-growth](../../529-inventory-growth/) (the *inventory* red flag), and "
            "[855-accrual-quality](../../855-accrual-quality/). See "
            "[docs/references.md](docs/references.md) for the exact dedup.\n\n"
            "*Think the alpha is in the surprise, or in small opaque names, not the level on blue "
            "chips? Build it, show a certifiable net spread on the size you'd actually run — then "
            "we'll talk.*"
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
            "# Days-Sales-Outstanding Buildup — a quantitative teardown 🔬\n"
            "### A point-in-time calendar-time tercile long-short (Newey-West) · a pooled "
            "event-drift cross-check with a label-shuffle placebo · an era split · the "
            "warns-on-sales regression · a 12-seed synthetic control\n\n"
            + BADGES +
            "The deep companion to the [notebook for the curious](01_for_the_curious.ipynb). The "
            "Abarbanell-Bushee claim — **rising DSO is a red flag that predicts weaker future "
            "sales *and* returns** — splits into testable pieces. This is distinct from every "
            "sibling on the desk: [231](../../231-sloan-accruals/) and "
            "[522](../../522-percent-operating-accruals/) rank on *aggregate accruals*, "
            "[529](../../529-inventory-growth/) on *inventory*, [855](../../855-accrual-quality/) "
            "on *accrual quality*. This is the **receivables-vs-sales (DSO)** line specifically.\n\n"
            "> ⚠️ **Data note.** EDGAR `AccountsReceivableNetCurrent` (→ `ReceivablesNetCurrent`) "
            "and `Revenues` (→ `RevenueFromContractWithCustomerExcludingAssessedTax`), longest "
            "per-name history, + yfinance adjusted closes, "
            + "as-of " + R["as_of"] + ". Point-in-time on the **filing date**. Survivorship named "
            "on the Signal axis (current-survivors basket). Thin/uneven coverage is a first-class "
            "caveat. Numbers in [`docs/results.md`](../docs/results.md).\n"
            ">\n"
            "> 💡 **The `💡 In plain words` notes** translate each result back to intuition."
        ),
        code(BOOT_CELL),

        # ---- BEAT 0 ----------------------------------------------------------
        md(
            "## Verdict, up front\n\n"
            "| Axis | Stamp | Decisive number |\n|---|---|---|\n"
            f"| **Signal** (returns) | `NONE` | calendar tercile long-short (long lean / short "
            f"building) **{R['ls_mean_bps']:+.1f} bps/mo** ({R['ls_ann']:+.1f}%/yr gross), "
            f"one-sample *t* = {R['ls_t_iid']:+.2f}, **Newey-West *t* = {R['ls_t_nw']:+.2f}**; "
            f"staleness-120 *t* = {R['ls120_t_nw']:+.2f}, pct-change *t* = {R['pct_t_nw']:+.2f}; "
            f"event drift flat and sign-flipping; post-2018 *t* = {R['era_late_t']:+.2f} |\n"
            f"| **Tradability** | `MIRAGE` | net of 20 bps + 100 bps borrow: {R['net'][(20, 100)][1]:+.2f}%/yr, "
            f"NW *t* = {R['net'][(20, 100)][2]:+.2f}, Sharpe {R['net'][(20, 100)][3]:.2f}; **negative** "
            "after costs |\n"
            f"| **Warns on sales?** | `FAINT` | next-q sales-growth regression slope {R['ws_slope']:+.4f} "
            f"(corr {R['ws_corr']:+.2f}), lean−building tercile spread **+{R['ws_spread']:.1f} pp** — "
            "correctly signed, economically negligible |\n\n"
            "> 💡 In plain words: the red flag is a near-perfect null on returns and only a whisper "
            "on sales. On big, liquid, heavily-covered names, a public accounting ratio predicts "
            "neither the stock nor (much) the fundamentals it's built from."
        ),

        # ---- BEAT 1 ----------------------------------------------------------
        md(
            "## 1 · The claim, steelmanned\n\n"
            "Let $A_{i,q}$ be name $i$'s current net receivables at fiscal quarter $q$ and "
            "$S_{i,q}$ its quarterly revenue, disclosed on filing date $F_{i,q}$. "
            "$\\text{DSO}_{i,q} = A_{i,q} / (S_{i,q}\\,/\\,91.25)$ (days). The signal is the "
            "point-in-time YoY change $\\Delta_{i,q} = \\text{DSO}_{i,q} - \\text{DSO}_{i,q-4}$, "
            "known at $F_{i,q}$; we rank on the signed score $-\\Delta$ (high = lean = long). The "
            "claims:\n\n"
            "- **H₁ (warns on sales).** $\\Delta_{i,q}$ predicts next quarter's recognised revenue "
            "growth *negatively* — the channel-stuffing mechanism.\n"
            "- **H₂ (leads returns).** A cross-sectional long-short (long lean / short building) "
            "earns a positive forward return spread — the market-mispricing claim.\n"
            "- **H₃ (tradable).** That spread survives realistic long-short costs + borrow.\n\n"
            f"We find **H₁ only faintly supported** (slope {R['ws_slope']:+.4f}, corr "
            f"{R['ws_corr']:+.2f}, +{R['ws_spread']:.1f} pp tercile spread — right sign, trivial "
            f"size), **H₂ not supported** (NW *t* = {R['ls_t_nw']:+.2f}, flat non-monotone event "
            "drift that flips sign by two quarters, dead post-2018), and therefore **H₃ moot** "
            "(a negative net spread). The Abarbanell-Bushee literature documents this signal in a "
            "*broad* cross-section including small, opaque firms; our deliberately large-cap, "
            "liquid tape is the hardest place for it to survive, and it does not — hence a `NONE`, "
            "not a `WEAK`, on returns."
        ),

        # ---- BEAT 2 ----------------------------------------------------------
        md(
            "## 2 · So what? — inference design\n\n"
            "The primary is a **calendar-time** long-short, not a pooled event study, precisely "
            "because balance-sheet signals are persistent and filings cluster: a calendar series "
            "of monthly long-short returns lets a **Newey-West (6-lag) HAC *t*** do the honest "
            "work the desk's `REAL` bar is written against. The panel is thin, so we sort into "
            "**terciles** (not quintiles) and require ≥ 6 names in the cross-section. The pooled "
            "event drift + a **label-shuffle placebo** is the cross-check; the warns-on-sales "
            "regression is graded on **magnitude and sign** (its pooled *t* ignores quarter "
            "clustering)."
        ),

        # ---- BEAT 3 ----------------------------------------------------------
        md(
            "## 3 · How we'd know — the protocol\n\n"
            f"- **Panel.** {R['n_events']:,} (ticker, filing) receivables quarters across "
            f"{R['n_names']} names, period ends {R['end_lo']} → {R['end_hi']}, each stamped with "
            "its 10-Q/10-K filing date (point-in-time).\n"
            "- **Primary.** Monthly tercile long-short on `dso_score` (= −ΔDSO), one execution lag "
            "(rank at month $M$ close, earn month $M{+}1$); Newey-West + one-sample *t*, Sharpe, "
            "Wilson-interval hit rate.\n"
            "- **Cross-check.** Pooled event drift over 21/63/126 trading days, one-day-lag entry, "
            "top-minus-bottom tercile, one-sample *t* + 10k-draw label-shuffle placebo, and the "
            "tercile monotonicity picture.\n"
            "- **Robustness.** Staleness 120 vs 200 days; the unit-free percentage-change signal "
            "`ΔDSO/DSO`; an era split at 2018 (the ASC-606 revenue-tagging break).\n"
            "- **Mechanism.** Pooled OLS of next-quarter revenue growth on `dso_yoy_chg` "
            "(winsorised 1/99% — a few near-zero prior-quarter revenues create absurd YoY "
            "outliers); expected slope < 0.\n"
            "- **Execution.** Long-short net of one-way cost × turnover (both legs) + short "
            "borrow.\n"
            "- **Control.** Synthetic panel, planted-red-flag knob; the null must not fire across "
            "12 seeds."
        ),

        # ---- BEAT 4 ----------------------------------------------------------
        md("## 4 · The teardown"),
        md(
            "### 4a · The primary — calendar-time tercile long-short (Newey-West)\n\n"
            "Rank fresh signals into terciles each month, long lean / short building equal-weight, "
            "earn next month's return. The decisive statistic is the HAC *t* of the monthly "
            "long-short series."
        ),
        code(
            "if HAVE_REAL:\n"
            "    ls = st.calendar_ls(PX, EV, signal_col='dso_score', n_buckets=3, min_names=6, staleness_days=200)\n"
            "    s = st.calendar_ls_stats(ls)\n"
            "    ls_pct = st.calendar_ls(PX, EV, signal_col='dso_pct_score', n_buckets=3, min_names=6, staleness_days=200)\n"
            "    s_pct = st.calendar_ls_stats(ls_pct)\n"
            "    print(f\"calendar long-short: {s['mean_bps']:+.1f} bps/mo ({s['ann_pct']:+.2f}%/yr gross), \"\n"
            "          f\"n={s['n_months']} months, avg cross-section {s['avg_n']:.1f}\")\n"
            "    print(f\"  one-sample t = {s['t_iid']:+.2f}   Newey-West(6) t = {s['t_nw']:+.2f}   \"\n"
            "          f\"Sharpe {s['sharpe']:.2f}   hit {s['hit']*100:.0f}%   turnover {s['avg_turnover']:.2f}\")\n"
            "    print(f\"  pct-change signal: {s_pct['mean_bps']:+.1f} bps/mo, NW t = {s_pct['t_nw']:+.2f}, Sharpe {s_pct['sharpe']:.2f}\")\n"
            "    cum = (1+ls['ls']).cumprod(); nser = ls['n']\n"
            "else:\n"
            "    cum = None; nser = None\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.3))\n"
            "if cum is not None:\n"
            "    a1.plot(cum.index, cum.values, color=AMBER, lw=1.8); a1.axhline(1, c='k', lw=.8)\n"
            "    a1.set_ylabel('growth of $1 (gross)'); a1.set_title(f\"Long-short: {R['ls_ann']:+.1f}%/yr, NW t = {R['ls_t_nw']:+.2f}\")\n"
            "    a2.plot(nser.index, nser.values, color=GREY, lw=1.5); a2.set_ylabel('names in cross-section')\n"
            "    a2.set_title('Thin & uneven panel coverage')\n"
            "else:\n"
            "    for a in (a1,a2): a.text(.5,.5,'run with cache',ha='center'); a.set_axis_off()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f\"frozen: {R['ls_mean_bps']:+.1f} bps/mo, NW t = {R['ls_t_nw']:+.2f}, \"\n"
            "      f\"staleness-120 NW t = {R['ls120_t_nw']:+.2f}, pct-change NW t = {R['pct_t_nw']:+.2f}\")"
        ),
        md(
            f"> 💡 In plain words: **+{R['ls_mean_bps']:.0f} bps/month** ({R['ls_ann']:+.1f}%/yr "
            f"gross) is right-signed — lean beats building, as the claim wants — but the HAC *t* is "
            f"**+{R['ls_t_nw']:.2f}**, indistinguishable from zero over n={R['n_months']} months "
            f"(avg cross-section {R['avg_n']:.0f}). And the point estimate is fragile: the "
            f"staleness-120 cut is +{R['ls120_mean_bps']:.0f} bps (*t* = +{R['ls120_t_nw']:.2f}) and "
            f"the unit-free percentage-change signal is +{R['pct_mean_bps']:.0f} bps (*t* = "
            f"+{R['pct_t_nw']:.2f}) — both essentially nil. The full-sample positive is a pre-2018 "
            "residue (see 4c), not a stable effect."
        ),
        md(
            "### 4b · The cross-check — pooled event drift + placebo + monotonicity\n\n"
            "Bucket all events by signed score; top-minus-bottom forward drift with a label-shuffle "
            "null. If there were a sort, the terciles would form a ladder (building → lean)."
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
            "a1.set_ylabel('one-sample t (long-short drift)'); a1.set_title('Event-drift t by horizon')\n"
            "a2.bar(['building','middle','lean'], mono, color=GREY, width=.55)\n"
            "for i,v in enumerate(mono): a2.annotate(f'{v:+.2f}%',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel('3-month forward return'); a2.set_title('Tercile ladder (building -> lean)')\n"
            "plt.tight_layout(); plt.show()\n"
            "for h,ls_,t,win,p in rows:\n"
            "    print(f'  H={h:>3}d: long-short {ls_:+.2f}%  t={t:+.2f}  win={win:.0f}%  placebo p={p:.3f}')"
        ),
        md(
            f"> 💡 In plain words: the pooled long-short drift is **flat** and *changes sign* with "
            f"the horizon — *t* = {R['drift'][21][4]:+.2f} at 21d, {R['drift'][63][4]:+.2f} at 63d, "
            f"then **{R['drift'][126][4]:+.2f}** at 126d (the building third actually out-returns "
            f"the lean one over two quarters). The label-shuffle placebo *p* is around "
            f"**0.44-0.81** (a random tercile split beats the real one about half the time, and "
            f"comfortably at 126d), and the 63d terciles are **non-monotone** "
            f"({R['mono63'][0]:+.2f}% / {R['mono63'][1]:+.2f}% / {R['mono63'][2]:+.2f}% "
            "building→lean). A horizon-dependent sign is the signature of noise, not a signal."
        ),
        md(
            "### 4c · Era split — nothing hiding in a regime\n\n"
            "Split the calendar long-short at 2018 (the ASC-606 revenue-tagging break)."
        ),
        code(
            "if HAVE_REAL:\n"
            "    e = ls[ls.index < '2018-01-01']['ls'].to_numpy(); l = ls[ls.index >= '2018-01-01']['ls'].to_numpy()\n"
            "    eb, et, en = e.mean()*1e4, st.newey_west_t(e), len(e)\n"
            "    lb, lt, ln = l.mean()*1e4, st.newey_west_t(l), len(l)\n"
            "else:\n"
            "    eb, et, en = R['era_early_bps'], R['era_early_t'], R['era_early_n']\n"
            "    lb, lt, ln = R['era_late_bps'], R['era_late_t'], R['era_late_n']\n"
            "fig, ax = plt.subplots(figsize=(8.2, 4.2))\n"
            "ax.bar([f'2010-2017\\n(n={en})', f'2018-2026\\n(n={ln})'], [eb, lb], color=[AMBER, AMBER], width=.5)\n"
            "for i,(v,t_) in enumerate([(eb,et),(lb,lt)]): ax.annotate(f'{v:+.0f} bps\\n(NW t={t_:+.2f})',(i,v),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('long-short mean (bps/mo)')\n"
            "ax.set_title('Long-short by era')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'2010-2017: {eb:+.1f} bps NW t={et:+.2f} (n={en})  |  2018-2026: {lb:+.1f} bps NW t={lt:+.2f} (n={ln})')"
        ),
        md(
            f"> 💡 In plain words: what little positive there is lives **entirely before 2018** "
            f"(+{R['era_early_bps']:.0f} bps/mo, NW *t* = +{R['era_early_t']:.2f} — still short of "
            f"the bar) and **evaporates after** (+{R['era_late_bps']:.0f} bps/mo, NW *t* = "
            f"+{R['era_late_t']:.2f}). Note the cross-section actually *shrinks* over the sample "
            f"(≈{R['xsec_early']:.0f} matched names early, ≈{R['xsec_late']:.0f} late) as ASC-606 "
            "revenue-tag switches break the YoY revenue match — so the modern half is both thinner "
            "and deader. Either way, there is no era in which this certifies."
        ),
        md(
            "### 4d · The mechanism — does a building DSO warn on sales?\n\n"
            "Pooled OLS of next-quarter revenue growth on this quarter's DSO change (expect a "
            "*negative* slope), and the future-sales spread between lean- and building-DSO terciles."
        ),
        code(
            "if HAVE_REAL:\n"
            "    q = st.warns_sales(EV)\n"
            "    fr = EV.dropna(subset=['dso_yoy_chg','next_rev_yoy'])\n"
            "    x, y = fr['dso_yoy_chg'].to_numpy(), fr['next_rev_yoy'].to_numpy()\n"
            "    slope, corr, low, high = q['slope'], q['corr'], q['low_sales']*100, q['high_sales']*100\n"
            "else:\n"
            "    x = y = None\n"
            "    slope, corr, low, high = R['ws_slope'], R['ws_corr'], R['ws_low_sales'], R['ws_high_sales']\n"
            "fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.4, 4.3))\n"
            "if x is not None:\n"
            "    m = (np.abs(x)<40)&(np.abs(y)<1.5)\n"
            "    a1.scatter(x[m], y[m]*100, s=8, alpha=.25, color=GREEN)\n"
            "    xs = np.linspace(np.percentile(x,2), np.percentile(x,98), 50)\n"
            "    a1.plot(xs, (slope*xs + (y.mean()-slope*x.mean()))*100, color=RED, lw=2)\n"
            "    a1.set_xlabel('DSO YoY change (days)'); a1.set_ylabel('NEXT-quarter revenue growth (%)')\n"
            "    a1.set_title(f'Warns on sales? slope {slope:+.3f}, corr {corr:+.2f}')\n"
            "else:\n"
            "    a1.text(.5,.5,'run with cache',ha='center'); a1.set_axis_off()\n"
            "a2.bar(['building third','lean third'], [high, low], color=[GREY, GREEN], width=.5)\n"
            "for i,v in enumerate([high, low]): a2.annotate(f'{v:+.1f}%',(i,v),ha='center',va='bottom')\n"
            "a2.set_ylabel(\"next quarter's revenue growth\"); a2.set_title(f'{low-high:+.0f} pp future-sales spread')\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'warns-sales: slope {slope:+.4f}, corr {corr:+.3f}, future-sales spread {low-high:+.1f} pp (lean {low:+.1f}% vs building {high:+.1f}%)')"
        ),
        md(
            f"> 💡 In plain words: this is the study's *strongest* result and it's still faint. The "
            f"slope is **{R['ws_slope']:+.4f}** (correctly negative — more DSO buildup, weaker "
            f"next-quarter sales) and the lean third grows sales **+{R['ws_spread']:.1f} pp** faster "
            f"than the building third — but the correlation is **{R['ws_corr']:+.2f}** and R² is "
            f"**{R['ws_r2']:.3f}**. The pooled *t* ({R['ws_t']:+.2f}) clears 2 only because n is "
            "large and clustered; we read the magnitude, and the magnitude says the channel-"
            "stuffing mechanism is a real-but-negligible tilt on these names, not a lever. (Signal "
            "winsorised at 1/99% to kill a handful of near-zero-prior-revenue YoY artefacts.)"
        ),
        md(
            "### 4e · Tradability — the timer\n\n"
            "The calendar long-short net of one-way costs × turnover (both legs) + short borrow."
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
            "for i,(cb,bb,a,t,sh) in enumerate(rows): ax.annotate(f'{a:+.1f}%/yr\\n(NW t={t:+.2f})',(i,a),ha='center',va='bottom')\n"
            "ax.axhline(0, c='k', lw=.8); ax.set_ylabel('net long-short (%/yr)')\n"
            "ax.set_title('Net of costs + borrow')\n"
            "plt.tight_layout(); plt.show()\n"
            "for cb,bb,a,t,sh in rows: print(f'  cost {int(cb)}bps + borrow {int(bb)}bps/yr: {a:+.2f}%/yr net, NW t={t:+.2f}, Sharpe {sh:.2f}')"
        ),
        md(
            f"> 💡 In plain words: turnover is modest (~{R['ls_turn']:.2f}/mo), but a gross edge that "
            f"was already indistinguishable from zero cannot pay a toll. Light friction (10 bps + "
            f"50 bps borrow) leaves +{R['net'][(10, 50)][1]:.1f}%/yr (*t* = {R['net'][(10, 50)][2]:+.2f}); "
            f"a realistic 20 bps + 100 bps borrow tips it **negative** ({R['net'][(20, 100)][1]:+.1f}%/yr, "
            f"*t* = {R['net'][(20, 100)][2]:+.2f}). **Tradability = MIRAGE.**"
        ),
        md(
            "### 4f · Faithful-engine & power control\n\n"
            "Synthetic price + signal panel with a TUNABLE planted red flag (building-DSO names "
            "drift down). The null (edge = 0) is checked over **12 seeds**."
        ),
        code(
            "null_ts = []\n"
            "for s_ in range(12):\n"
            "    p0, e0 = data.synthetic_panel(edge=0.0, seed=853 + s_)\n"
            "    null_ts.append(st.synthetic_detect(p0, e0)['t_nw'])\n"
            "null_ts = np.asarray(null_ts)\n"
            "p1, e1 = data.synthetic_panel(edge=0.15, seed=853)\n"
            "planted_t = st.synthetic_detect(p1, e1)['t_nw']\n"
            "fig, ax = plt.subplots(figsize=(8.8, 4.3))\n"
            "ax.scatter(np.linspace(-.12,.12,12), null_ts, color=GREY, s=45, label='null worlds (edge=0), 12 seeds')\n"
            "ax.scatter([1], [planted_t], color=GREEN, s=95, zorder=5, label='planted red flag (edge=0.15)')\n"
            "ax.axhline(2, ls='--', c=RED, lw=1); ax.axhline(-2, ls='--', c=RED, lw=1); ax.axhline(0, c='k', lw=.6)\n"
            "ax.set_xticks([0,1]); ax.set_xticklabels(['null x 12','planted'])\n"
            "ax.set_ylabel('calendar long-short Newey-West t')\n"
            "ax.set_title('Control: the null barely fires; a planted red flag lights up'); ax.legend()\n"
            "plt.tight_layout(); plt.show()\n"
            "print(f'null: mean NW t = {null_ts.mean():+.2f} (sd {null_ts.std(ddof=1):.2f}), '\n"
            "      f'|t|>=2 in {(abs(null_ts)>=2).sum()}/12  |  planted NW t = {planted_t:+.2f}')"
        ),
        md(
            f"> 💡 In plain words: across 12 null worlds the detector averages NW *t* = "
            f"{R['syn_null_mean']:+.2f} (sd {R['syn_null_sd']:.2f}) and crosses |t|=2 in "
            f"**{R['syn_null_fire']}/12** seeds — an unbiased, well-behaved null. A planted red flag "
            f"reads NW *t* = **{R['syn_planted_t']:.1f}** ({R['syn_planted_bps']:.0f} bps/mo). So the "
            f"machinery is faithful and powerful, which means the real-tape +{R['ls_t_nw']:.2f} is a "
            "genuine null, not a broken pipeline. *(Power check only — never cited in support of a "
            "real-tape stamp.)*"
        ),

        # ---- BEAT 5 ----------------------------------------------------------
        md(
            "## 5 · The verdict\n\n"
            f"- **Signal (returns) `NONE`** — calendar tercile long-short {R['ls_mean_bps']:+.1f} "
            f"bps/mo ({R['ls_ann']:+.1f}%/yr gross), one-sample *t* = {R['ls_t_iid']:+.2f}, "
            f"**Newey-West *t* = {R['ls_t_nw']:+.2f}**; staleness-120 and percentage-change variants "
            f"both ~0 ({R['ls120_t_nw']:+.2f}, {R['pct_t_nw']:+.2f}); pooled event drift flat and "
            f"sign-flipping (placebo *p* 0.44-0.81); the small full-sample positive lives entirely "
            f"pre-2018 and dies after (*t* = {R['era_late_t']:+.2f}). Right-signed in one spec, but "
            "no coherent, robust effect — the desk's `NONE`.\n"
            f"- **Tradability `MIRAGE`** — net of 20 bps + 100 bps borrow: {R['net'][(20, 100)][1]:+.2f}%/yr, "
            f"NW *t* = {R['net'][(20, 100)][2]:+.2f}, Sharpe {R['net'][(20, 100)][3]:.2f}; negative after "
            "costs.\n"
            f"- **Warns on sales? `FAINT`** — regression slope {R['ws_slope']:+.4f}, correlation "
            f"{R['ws_corr']:+.2f}, **+{R['ws_spread']:.1f} pp** lean-minus-building future-sales "
            "spread. Correctly signed, economically negligible — the mechanism exists but is a "
            "whisper, not the lead the red flag promises."
        ),

        # ---- BEAT 6 ----------------------------------------------------------
        md(
            "## 6 · Going further\n\n"
            "- **The residual worth chasing** is the *surprise*, not the *level*: DSO change "
            "relative to a seasonal expectation, or conditioned on small/opaque names where a "
            "receivables build is genuinely informative rather than noise on a mega-cap. This "
            "study used large, liquid, heavily-covered filers — the *hardest* place for a public "
            "red flag to survive.\n"
            "- **Coverage honesty:** only a subset of the basket exposes a long, cleanly-matched "
            "receivables+revenue history; terciles on a thin early cross-section are noisy, and the "
            "pooled event drift across 1,000+ events is the more decisive evidence.\n"
            "- **Dedup map:** [231-sloan-accruals](../../231-sloan-accruals/) (total accruals), "
            "[522-percent-operating-accruals](../../522-percent-operating-accruals/), "
            "[529-inventory-growth](../../529-inventory-growth/) (the inventory red flag), "
            "[855-accrual-quality](../../855-accrual-quality/). None ranks on the **YoY change in "
            "DSO** itself.\n\n"
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
